//! # Pivot Points (PIVOT)
//!
//! Support (S) and resistance (R) levels from High, Low, Close, Open prices.
//! Multiple calculation modes supported (Standard, Fibonacci, Demark, Camarilla, Woodie).
//!
//! ## Parameters
//! - **mode**: Calculation method. 0=Standard, 1=Fibonacci, 2=Demark, 3=Camarilla (default), 4=Woodie
//!
//! ## Inputs
//! - **high**: High price data
//! - **low**: Low price data
//! - **close**: Close price data
//! - **open**: Open price data
//!
//! ## Returns
//! - **r4, r3, r2, r1**: Resistance levels (4 vectors)
//! - **pp**: Pivot point (1 vector)
//! - **s1, s2, s3, s4**: Support levels (4 vectors)
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Implemented with per-lane validity masks; tails fallback to scalar
//! - **Streaming**: Implemented with O(1) update performance (pure calculations)
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations
//! - Decision: Scalar path structured as per-mode loops (jammed) with safe indexing to aid LLVM vectorization; observed ~10–15% improvement at 100k locally. AVX2/AVX512 kernels implemented and selected via runtime kernel detection. Batch uses row-per-mode fan-out; no additional row-specific sharing beyond single-pass p/d reuse per element.

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

// ========== CONSTANTS ==========

const N_LEVELS: usize = 9; // r4, r3, r2, r1, pp, s1, s2, s3, s4 in this order

// ========== HELPER FUNCTIONS ==========

#[inline(always)]
fn first_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    let len = high.len();
    for i in 0..len {
        if !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()) {
            return Some(i);
        }
    }
    None
}

// Unified kernel dispatch like ALMA
#[inline(always)]
fn pivot_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    k: Kernel,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    unsafe {
        match k {
            Kernel::Scalar | Kernel::ScalarBatch => pivot_scalar(
                high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => pivot_avx2(
                high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => pivot_avx512(
                high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
            ),
            _ => unreachable!(),
        }
    }
}

// ========== DATA/INPUT/OUTPUT STRUCTS ==========

#[derive(Debug, Clone)]
pub enum PivotData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        open: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PivotParams {
    pub mode: Option<usize>,
}
impl Default for PivotParams {
    fn default() -> Self {
        Self { mode: Some(3) }
    }
}

#[derive(Debug, Clone)]
pub struct PivotInput<'a> {
    pub data: PivotData<'a>,
    pub params: PivotParams,
}
impl<'a> PivotInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: PivotParams) -> Self {
        Self {
            data: PivotData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        open: &'a [f64],
        params: PivotParams,
    ) -> Self {
        Self {
            data: PivotData::Slices {
                high,
                low,
                close,
                open,
            },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, PivotParams::default())
    }
    #[inline]
    pub fn get_mode(&self) -> usize {
        self.params
            .mode
            .unwrap_or_else(|| PivotParams::default().mode.unwrap())
    }
}
impl<'a> AsRef<PivotData<'a>> for PivotInput<'a> {
    fn as_ref(&self) -> &PivotData<'a> {
        &self.data
    }
}

#[derive(Debug, Clone)]
pub struct PivotOutput {
    pub r4: Vec<f64>,
    pub r3: Vec<f64>,
    pub r2: Vec<f64>,
    pub r1: Vec<f64>,
    pub pp: Vec<f64>,
    pub s1: Vec<f64>,
    pub s2: Vec<f64>,
    pub s3: Vec<f64>,
    pub s4: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum PivotError {
    #[error("pivot: One or more required fields is empty.")]
    EmptyData,
    #[error("pivot: All values are NaN.")]
    AllValuesNaN,
    #[error("pivot: Not enough valid data after the first valid index.")]
    NotEnoughValidData,
}

// ========== BUILDER ==========

#[derive(Copy, Clone, Debug)]
pub struct PivotBuilder {
    mode: Option<usize>,
    kernel: Kernel,
}
impl Default for PivotBuilder {
    fn default() -> Self {
        Self {
            mode: None,
            kernel: Kernel::Auto,
        }
    }
}
impl PivotBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn mode(mut self, mode: usize) -> Self {
        self.mode = Some(mode);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<PivotOutput, PivotError> {
        let params = PivotParams { mode: self.mode };
        let input = PivotInput::from_candles(candles, params);
        pivot_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        open: &[f64],
    ) -> Result<PivotOutput, PivotError> {
        let params = PivotParams { mode: self.mode };
        let input = PivotInput::from_slices(high, low, close, open, params);
        pivot_with_kernel(&input, self.kernel)
    }
}

// ========== MAIN INTERFACE FUNCTIONS ==========

#[inline]
pub fn pivot(input: &PivotInput) -> Result<PivotOutput, PivotError> {
    pivot_with_kernel(input, Kernel::Auto)
}

pub fn pivot_with_kernel(input: &PivotInput, kernel: Kernel) -> Result<PivotOutput, PivotError> {
    let (high, low, close, open) = match &input.data {
        PivotData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let open = source_type(candles, "open");
            (high, low, close, open)
        }
        PivotData::Slices {
            high,
            low,
            close,
            open,
        } => (*high, *low, *close, *open),
    };
    let len = high.len();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(PivotError::EmptyData);
    }
    if low.len() != len || close.len() != len || open.len() != len {
        return Err(PivotError::EmptyData);
    }
    let mode = input.get_mode();

    let mut first_valid_idx = None;
    for i in 0..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !(h.is_nan() || l.is_nan() || c.is_nan()) {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(PivotError::AllValuesNaN),
    };
    if first_valid_idx >= len {
        return Err(PivotError::NotEnoughValidData);
    }

    // Allocate output vectors with NaN prefix
    let mut r4 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut r3 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut r2 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut r1 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut pp = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s1 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s2 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s3 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s4 = alloc_with_nan_prefix(len, first_valid_idx);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    pivot_compute_into(
        high,
        low,
        close,
        open,
        mode,
        first_valid_idx,
        chosen,
        &mut r4,
        &mut r3,
        &mut r2,
        &mut r1,
        &mut pp,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
    );
    Ok(PivotOutput {
        r4,
        r3,
        r2,
        r1,
        pp,
        s1,
        s2,
        s3,
        s4,
    })
}

#[inline]
pub fn pivot_into_slices(
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
    input: &PivotInput,
    kern: Kernel,
) -> Result<(), PivotError> {
    let (high, low, close, open) = match &input.data {
        PivotData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let open = source_type(candles, "open");
            (high, low, close, open)
        }
        PivotData::Slices {
            high,
            low,
            close,
            open,
        } => (*high, *low, *close, *open),
    };

    let len = high.len();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(PivotError::EmptyData);
    }
    if low.len() != len || close.len() != len || open.len() != len {
        return Err(PivotError::EmptyData);
    }
    if r4.len() != len
        || r3.len() != len
        || r2.len() != len
        || r1.len() != len
        || pp.len() != len
        || s1.len() != len
        || s2.len() != len
        || s3.len() != len
        || s4.len() != len
    {
        return Err(PivotError::EmptyData);
    }

    let mode = input.get_mode();

    let mut first_valid_idx = None;
    for i in 0..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !(h.is_nan() || l.is_nan() || c.is_nan()) {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(PivotError::AllValuesNaN),
    };
    if first_valid_idx >= len {
        return Err(PivotError::NotEnoughValidData);
    }

    // Match ALMA pattern: compute first, then set warmup NaNs
    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    pivot_compute_into(
        high,
        low,
        close,
        open,
        mode,
        first_valid_idx,
        chosen,
        r4,
        r3,
        r2,
        r1,
        pp,
        s1,
        s2,
        s3,
        s4,
    );

    // Now set warmup NaNs after computation
    for i in 0..first_valid_idx {
        r4[i] = f64::NAN;
        r3[i] = f64::NAN;
        r2[i] = f64::NAN;
        r1[i] = f64::NAN;
        pp[i] = f64::NAN;
        s1[i] = f64::NAN;
        s2[i] = f64::NAN;
        s3[i] = f64::NAN;
        s4[i] = f64::NAN;
    }

    Ok(())
}

#[inline]
pub unsafe fn pivot_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    let len = high.len();
    if first >= len {
        return;
    }

    let nan = f64::NAN;

    match mode {
        // ========================== STANDARD ==========================
        0 => {
            for i in first..len {
                let h = high[i];
                let l = low[i];
                let c = close[i];
                if h.is_nan() || l.is_nan() || c.is_nan() {
                    r4[i] = nan; r3[i] = nan; r2[i] = nan; r1[i] = nan; pp[i] = nan; s1[i] = nan; s2[i] = nan; s3[i] = nan; s4[i] = nan;
                    continue;
                }
                let d = h - l;
                let p = (h + l + c) * (1.0 / 3.0);
                let t2 = p + p;
                pp[i] = p;
                r1[i] = t2 - l;
                r2[i] = p + d;
                s1[i] = t2 - h;
                s2[i] = p - d;
                r3[i] = nan; r4[i] = nan; s3[i] = nan; s4[i] = nan;
            }
        }

        // ========================== FIBONACCI ==========================
        1 => {
            for i in first..len {
                let h = high[i];
                let l = low[i];
                let c = close[i];
                if h.is_nan() || l.is_nan() || c.is_nan() {
                    r4[i] = nan; r3[i] = nan; r2[i] = nan; r1[i] = nan; pp[i] = nan; s1[i] = nan; s2[i] = nan; s3[i] = nan; s4[i] = nan;
                    continue;
                }
                let d = h - l;
                let p = (h + l + c) * (1.0 / 3.0);
                let d38 = d * 0.382_f64;
                let d62 = d * 0.618_f64;
                pp[i] = p;
                r1[i] = p + d38;
                r2[i] = p + d62;
                r3[i] = p + d;
                s1[i] = p - d38;
                s2[i] = p - d62;
                s3[i] = p - d;
                r4[i] = nan; s4[i] = nan;
            }
        }

        // ========================== DEMARK ==========================
        2 => {
            for i in first..len {
                let h = high[i];
                let l = low[i];
                let c = close[i];
                let o = open[i];
                if h.is_nan() || l.is_nan() || c.is_nan() {
                    r4[i] = nan; r3[i] = nan; r2[i] = nan; r1[i] = nan; pp[i] = nan; s1[i] = nan; s2[i] = nan; s3[i] = nan; s4[i] = nan;
                    continue;
                }
                let p = if c < o { (h + (l + l) + c) * 0.25 } else if c > o { ((h + h) + l + c) * 0.25 } else { (h + l + (c + c)) * 0.25 };
                pp[i] = p;
                let num = if c < o { (h + (l + l) + c) * 0.5 } else if c > o { ((h + h) + l + c) * 0.5 } else { (h + l + (c + c)) * 0.5 };
                r1[i] = num - l;
                s1[i] = num - h;
                r2[i] = nan; r3[i] = nan; r4[i] = nan; s2[i] = nan; s3[i] = nan; s4[i] = nan;
            }
        }

        // ========================== CAMARILLA (default) ==========================
        3 => {
            const C1: f64 = 0.0916_f64;
            const C2: f64 = 0.183_f64;
            const C3: f64 = 0.275_f64;
            const C4: f64 = 0.55_f64;
            for i in first..len {
                let h = high[i];
                let l = low[i];
                let c = close[i];
                if h.is_nan() || l.is_nan() || c.is_nan() {
                    r4[i] = nan; r3[i] = nan; r2[i] = nan; r1[i] = nan; pp[i] = nan; s1[i] = nan; s2[i] = nan; s3[i] = nan; s4[i] = nan;
                    continue;
                }
                let d = h - l;
                let p = (h + l + c) * (1.0 / 3.0);
                pp[i] = p;
                let d1 = d * C1; let d2 = d * C2; let d3 = d * C3; let d4 = d * C4;
                r1[i] = d1 + c; r2[i] = d2 + c; r3[i] = d3 + c; r4[i] = d4 + c;
                s1[i] = c - d1; s2[i] = c - d2; s3[i] = c - d3; s4[i] = c - d4;
            }
        }

        // ========================== WOODIE ==========================
        4 => {
            for i in first..len {
                let h = high[i];
                let l = low[i];
                let c = close[i];
                let o = open[i];
                if h.is_nan() || l.is_nan() || c.is_nan() {
                    r4[i] = nan; r3[i] = nan; r2[i] = nan; r1[i] = nan; pp[i] = nan; s1[i] = nan; s2[i] = nan; s3[i] = nan; s4[i] = nan;
                    continue;
                }
                let d = h - l;
                let p = (h + l + (o + o)) * 0.25; // (H+L+2*O)/4
                pp[i] = p;
                let t2p = p + p; let t2l = l + l; let t2h = h + h;
                let r3v = (t2p - t2l) + h; r3[i] = r3v; r4[i] = r3v + d; r2[i] = p + d; r1[i] = t2p - l;
                s1[i] = t2p - h; s2[i] = p - d; let s3v = (l + t2p) - t2h; s3[i] = s3v; s4[i] = s3v - d;
            }
        }

        _ => {}
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pivot_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = high.len();
    if first >= len {
        return;
    }

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let op = open.as_ptr();

    let r4p = r4.as_mut_ptr();
    let r3p = r3.as_mut_ptr();
    let r2p = r2.as_mut_ptr();
    let r1p = r1.as_mut_ptr();
    let ppp = pp.as_mut_ptr();
    let s1p = s1.as_mut_ptr();
    let s2p = s2.as_mut_ptr();
    let s3p = s3.as_mut_ptr();
    let s4p = s4.as_mut_ptr();

    let v_nan = _mm256_set1_pd(f64::NAN);
    let v_third = _mm256_set1_pd(1.0 / 3.0);
    let v_quart = _mm256_set1_pd(0.25);
    let v_half = _mm256_set1_pd(0.5);
    let v_one  = _mm256_set1_pd(1.0);
    let v_c0916 = _mm256_set1_pd(0.0916);
    let v_c0183 = _mm256_set1_pd(0.183);
    let v_c0275 = _mm256_set1_pd(0.275);
    let v_c0550 = _mm256_set1_pd(0.55);
    let v_c0382 = _mm256_set1_pd(0.382);
    let v_c0618 = _mm256_set1_pd(0.618);
    let v_neg1  = _mm256_set1_pd(-1.0);
    let v_n0382 = _mm256_set1_pd(-0.382);
    let v_n0618 = _mm256_set1_pd(-0.618);

    let mut i = first;
    let end4 = first + ((len - first) & !3);

    #[inline(always)]
    unsafe fn valid_mask_avx2(h: __m256d, l: __m256d, c: __m256d) -> __m256d {
        let ord_h = _mm256_cmp_pd(h, h, _CMP_ORD_Q);
        let ord_l = _mm256_cmp_pd(l, l, _CMP_ORD_Q);
        let ord_c = _mm256_cmp_pd(c, c, _CMP_ORD_Q);
        _mm256_and_pd(_mm256_and_pd(ord_h, ord_l), ord_c)
    }

    #[inline(always)]
    unsafe fn blendv(a: __m256d, b: __m256d, mask: __m256d) -> __m256d {
        _mm256_blendv_pd(a, b, mask)
    }

    match mode {
        0 => {
            while i < end4 {
                let h = _mm256_loadu_pd(hp.add(i));
                let l = _mm256_loadu_pd(lp.add(i));
                let c = _mm256_loadu_pd(cp.add(i));
                let vld = valid_mask_avx2(h, l, c);

                let p = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, l), c), v_third);
                let d = _mm256_sub_pd(h, l);
                let t2 = _mm256_add_pd(p, p);

                let r1v = _mm256_sub_pd(t2, l);
                let r2v = _mm256_fmadd_pd(d, v_one, p);
                let s1v = _mm256_sub_pd(t2, h);
                let s2v = _mm256_fmadd_pd(d, v_neg1, p);

                _mm256_storeu_pd(ppp.add(i), blendv(v_nan, p, vld));
                _mm256_storeu_pd(r1p.add(i), blendv(v_nan, r1v, vld));
                _mm256_storeu_pd(r2p.add(i), blendv(v_nan, r2v, vld));
                _mm256_storeu_pd(s1p.add(i), blendv(v_nan, s1v, vld));
                _mm256_storeu_pd(s2p.add(i), blendv(v_nan, s2v, vld));
                _mm256_storeu_pd(r3p.add(i), v_nan);
                _mm256_storeu_pd(r4p.add(i), v_nan);
                _mm256_storeu_pd(s3p.add(i), v_nan);
                _mm256_storeu_pd(s4p.add(i), v_nan);

                i += 4;
            }
        }
        1 => {
            while i < end4 {
                let h = _mm256_loadu_pd(hp.add(i));
                let l = _mm256_loadu_pd(lp.add(i));
                let c = _mm256_loadu_pd(cp.add(i));
                let vld = valid_mask_avx2(h, l, c);

                let p = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, l), c), v_third);
                let d = _mm256_sub_pd(h, l);
                let r1v = _mm256_fmadd_pd(d, v_c0382, p);
                let r2v = _mm256_fmadd_pd(d, v_c0618, p);
                let r3v = _mm256_fmadd_pd(d, v_one,   p);
                let s1v = _mm256_fmadd_pd(d, v_n0382, p);
                let s2v = _mm256_fmadd_pd(d, v_n0618, p);
                let s3v = _mm256_fmadd_pd(d, v_neg1,  p);

                _mm256_storeu_pd(ppp.add(i), blendv(v_nan, p, vld));
                _mm256_storeu_pd(r1p.add(i), blendv(v_nan, r1v, vld));
                _mm256_storeu_pd(r2p.add(i), blendv(v_nan, r2v, vld));
                _mm256_storeu_pd(r3p.add(i), blendv(v_nan, r3v, vld));
                _mm256_storeu_pd(s1p.add(i), blendv(v_nan, s1v, vld));
                _mm256_storeu_pd(s2p.add(i), blendv(v_nan, s2v, vld));
                _mm256_storeu_pd(s3p.add(i), blendv(v_nan, s3v, vld));
                _mm256_storeu_pd(r4p.add(i), v_nan);
                _mm256_storeu_pd(s4p.add(i), v_nan);

                i += 4;
            }
        }
        2 => {
            while i < end4 {
                let h = _mm256_loadu_pd(hp.add(i));
                let l = _mm256_loadu_pd(lp.add(i));
                let c = _mm256_loadu_pd(cp.add(i));
                let o = _mm256_loadu_pd(op.add(i));
                let vld = valid_mask_avx2(h, l, c);

                let mlt = _mm256_cmp_pd(c, o, _CMP_LT_OQ);
                let mgt = _mm256_cmp_pd(c, o, _CMP_GT_OQ);

                let p_lt = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, _mm256_add_pd(l, l)), c), v_quart);
                let p_gt = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(h, h), l), c), v_quart);
                let p_eq = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, l), _mm256_add_pd(c, c)), v_quart);

                let mut p = blendv(p_eq, p_gt, mgt);
                p = blendv(p, p_lt, mlt);
                _mm256_storeu_pd(ppp.add(i), blendv(v_nan, p, vld));

                let n_lt = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, _mm256_add_pd(l, l)), c), v_half);
                let n_gt = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(h, h), l), c), v_half);
                let n_eq = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, l), _mm256_add_pd(c, c)), v_half);

                let mut n = blendv(n_eq, n_gt, mgt);
                n = blendv(n, n_lt, mlt);

                let r1v = _mm256_sub_pd(n, l);
                let s1v = _mm256_sub_pd(n, h);

                _mm256_storeu_pd(r1p.add(i), blendv(v_nan, r1v, vld));
                _mm256_storeu_pd(s1p.add(i), blendv(v_nan, s1v, vld));
                _mm256_storeu_pd(r2p.add(i), v_nan);
                _mm256_storeu_pd(r3p.add(i), v_nan);
                _mm256_storeu_pd(r4p.add(i), v_nan);
                _mm256_storeu_pd(s2p.add(i), v_nan);
                _mm256_storeu_pd(s3p.add(i), v_nan);
                _mm256_storeu_pd(s4p.add(i), v_nan);

                i += 4;
            }
        }
        3 => {
            while i < end4 {
                let h = _mm256_loadu_pd(hp.add(i));
                let l = _mm256_loadu_pd(lp.add(i));
                let c = _mm256_loadu_pd(cp.add(i));
                let vld = valid_mask_avx2(h, l, c);

                let p = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, l), c), v_third);
                _mm256_storeu_pd(ppp.add(i), blendv(v_nan, p, vld));

                let d = _mm256_sub_pd(h, l);
                let d1 = _mm256_mul_pd(d, v_c0916);
                let d2 = _mm256_mul_pd(d, v_c0183);
                let d3 = _mm256_mul_pd(d, v_c0275);
                let d4 = _mm256_mul_pd(d, v_c0550);

                let r1v = _mm256_fmadd_pd(d, v_c0916, c);
                let r2v = _mm256_fmadd_pd(d, v_c0183, c);
                let r3v = _mm256_fmadd_pd(d, v_c0275, c);
                let r4v = _mm256_fmadd_pd(d, v_c0550, c);

                let s1v = _mm256_fmadd_pd(d, _mm256_sub_pd(_mm256_setzero_pd(), v_c0916), c);
                let s2v = _mm256_fmadd_pd(d, _mm256_sub_pd(_mm256_setzero_pd(), v_c0183), c);
                let s3v = _mm256_fmadd_pd(d, _mm256_sub_pd(_mm256_setzero_pd(), v_c0275), c);
                let s4v = _mm256_fmadd_pd(d, _mm256_sub_pd(_mm256_setzero_pd(), v_c0550), c);

                _mm256_storeu_pd(r1p.add(i), blendv(v_nan, r1v, vld));
                _mm256_storeu_pd(r2p.add(i), blendv(v_nan, r2v, vld));
                _mm256_storeu_pd(r3p.add(i), blendv(v_nan, r3v, vld));
                _mm256_storeu_pd(r4p.add(i), blendv(v_nan, r4v, vld));
                _mm256_storeu_pd(s1p.add(i), blendv(v_nan, s1v, vld));
                _mm256_storeu_pd(s2p.add(i), blendv(v_nan, s2v, vld));
                _mm256_storeu_pd(s3p.add(i), blendv(v_nan, s3v, vld));
                _mm256_storeu_pd(s4p.add(i), blendv(v_nan, s4v, vld));

                i += 4;
            }
        }
        4 => {
            while i < end4 {
                let h = _mm256_loadu_pd(hp.add(i));
                let l = _mm256_loadu_pd(lp.add(i));
                let c = _mm256_loadu_pd(cp.add(i));
                let o = _mm256_loadu_pd(op.add(i));
                let vld = valid_mask_avx2(h, l, c);

                let p = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(h, l), _mm256_add_pd(o, o)), v_quart);
                let t2p = _mm256_add_pd(p, p);
                let t2l = _mm256_add_pd(l, l);
                let t2h = _mm256_add_pd(h, h);
                let d = _mm256_sub_pd(h, l);

                let r3v = _mm256_add_pd(_mm256_sub_pd(t2p, t2l), h);
                let r4v = _mm256_fmadd_pd(d, v_one, r3v);
                let r2v = _mm256_fmadd_pd(d, v_one, p);
                let r1v = _mm256_sub_pd(t2p, l);

                let s1v = _mm256_sub_pd(t2p, h);
                let s2v = _mm256_fmadd_pd(d, v_neg1, p);
                let s3v = _mm256_sub_pd(_mm256_add_pd(l, t2p), t2h);
                let s4v = _mm256_fmadd_pd(d, v_neg1, s3v);

                _mm256_storeu_pd(ppp.add(i), blendv(v_nan, p, vld));
                _mm256_storeu_pd(r1p.add(i), blendv(v_nan, r1v, vld));
                _mm256_storeu_pd(r2p.add(i), blendv(v_nan, r2v, vld));
                _mm256_storeu_pd(r3p.add(i), blendv(v_nan, r3v, vld));
                _mm256_storeu_pd(r4p.add(i), blendv(v_nan, r4v, vld));
                _mm256_storeu_pd(s1p.add(i), blendv(v_nan, s1v, vld));
                _mm256_storeu_pd(s2p.add(i), blendv(v_nan, s2v, vld));
                _mm256_storeu_pd(s3p.add(i), blendv(v_nan, s3v, vld));
                _mm256_storeu_pd(s4p.add(i), blendv(v_nan, s4v, vld));

                i += 4;
            }
        }
        _ => {}
    }

    if i < len {
        pivot_scalar(high, low, close, open, mode, i, r4, r3, r2, r1, pp, s1, s2, s3, s4);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    if high.len() <= 32 {
        pivot_avx512_short(
            high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
        )
    } else {
        pivot_avx512_long(
            high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx512_long(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn pivot_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = high.len();
    if first >= len {
        return;
    }

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let op = open.as_ptr();

    let r4p = r4.as_mut_ptr();
    let r3p = r3.as_mut_ptr();
    let r2p = r2.as_mut_ptr();
    let r1p = r1.as_mut_ptr();
    let ppp = pp.as_mut_ptr();
    let s1p = s1.as_mut_ptr();
    let s2p = s2.as_mut_ptr();
    let s3p = s3.as_mut_ptr();
    let s4p = s4.as_mut_ptr();

    let v_nan = _mm512_set1_pd(f64::NAN);
    let v_third = _mm512_set1_pd(1.0 / 3.0);
    let v_quart = _mm512_set1_pd(0.25);
    let v_half = _mm512_set1_pd(0.5);
    let v_one  = _mm512_set1_pd(1.0);
    let v_c0916 = _mm512_set1_pd(0.0916);
    let v_c0183 = _mm512_set1_pd(0.183);
    let v_c0275 = _mm512_set1_pd(0.275);
    let v_c0550 = _mm512_set1_pd(0.55);
    let v_c0382 = _mm512_set1_pd(0.382);
    let v_c0618 = _mm512_set1_pd(0.618);
    let v_neg1  = _mm512_set1_pd(-1.0);
    let v_n0382 = _mm512_set1_pd(-0.382);
    let v_n0618 = _mm512_set1_pd(-0.618);

    let mut i = first;
    let step = 8;

    #[inline(always)]
    unsafe fn valid_mask_avx512(h: __m512d, l: __m512d, c: __m512d) -> u8 {
        let mh = _mm512_cmp_pd_mask(h, h, _CMP_ORD_Q);
        let ml = _mm512_cmp_pd_mask(l, l, _CMP_ORD_Q);
        let mc = _mm512_cmp_pd_mask(c, c, _CMP_ORD_Q);
        mh & ml & mc
    }

    match mode {
        0 => {
            while i + step <= len {
                let h = _mm512_loadu_pd(hp.add(i));
                let l = _mm512_loadu_pd(lp.add(i));
                let c = _mm512_loadu_pd(cp.add(i));
                let mk = valid_mask_avx512(h, l, c);

                let p = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, l), c), v_third);
                let d = _mm512_sub_pd(h, l);
                let t2 = _mm512_add_pd(p, p);

                let r1v = _mm512_sub_pd(t2, l);
                let r2v = _mm512_fmadd_pd(d, v_one, p);
                let s1v = _mm512_sub_pd(t2, h);
                let s2v = _mm512_fmadd_pd(d, v_neg1, p);

                _mm512_storeu_pd(ppp.add(i), _mm512_mask_blend_pd(mk, v_nan, p));
                _mm512_storeu_pd(r1p.add(i), _mm512_mask_blend_pd(mk, v_nan, r1v));
                _mm512_storeu_pd(r2p.add(i), _mm512_mask_blend_pd(mk, v_nan, r2v));
                _mm512_storeu_pd(s1p.add(i), _mm512_mask_blend_pd(mk, v_nan, s1v));
                _mm512_storeu_pd(s2p.add(i), _mm512_mask_blend_pd(mk, v_nan, s2v));
                _mm512_storeu_pd(r3p.add(i), v_nan);
                _mm512_storeu_pd(r4p.add(i), v_nan);
                _mm512_storeu_pd(s3p.add(i), v_nan);
                _mm512_storeu_pd(s4p.add(i), v_nan);

                i += step;
            }
        }
        1 => {
            while i + step <= len {
                let h = _mm512_loadu_pd(hp.add(i));
                let l = _mm512_loadu_pd(lp.add(i));
                let c = _mm512_loadu_pd(cp.add(i));
                let mk = valid_mask_avx512(h, l, c);

                let p = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, l), c), v_third);
                let d = _mm512_sub_pd(h, l);
                let r1v = _mm512_fmadd_pd(d, v_c0382, p);
                let r2v = _mm512_fmadd_pd(d, v_c0618, p);
                let r3v = _mm512_fmadd_pd(d, v_one,   p);
                let s1v = _mm512_fmadd_pd(d, v_n0382, p);
                let s2v = _mm512_fmadd_pd(d, v_n0618, p);
                let s3v = _mm512_fmadd_pd(d, v_neg1,  p);

                _mm512_storeu_pd(ppp.add(i), _mm512_mask_blend_pd(mk, v_nan, p));
                _mm512_storeu_pd(r1p.add(i), _mm512_mask_blend_pd(mk, v_nan, r1v));
                _mm512_storeu_pd(r2p.add(i), _mm512_mask_blend_pd(mk, v_nan, r2v));
                _mm512_storeu_pd(r3p.add(i), _mm512_mask_blend_pd(mk, v_nan, r3v));
                _mm512_storeu_pd(s1p.add(i), _mm512_mask_blend_pd(mk, v_nan, s1v));
                _mm512_storeu_pd(s2p.add(i), _mm512_mask_blend_pd(mk, v_nan, s2v));
                _mm512_storeu_pd(s3p.add(i), _mm512_mask_blend_pd(mk, v_nan, s3v));
                _mm512_storeu_pd(r4p.add(i), v_nan);
                _mm512_storeu_pd(s4p.add(i), v_nan);

                i += step;
            }
        }
        2 => {
            while i + step <= len {
                let h = _mm512_loadu_pd(hp.add(i));
                let l = _mm512_loadu_pd(lp.add(i));
                let c = _mm512_loadu_pd(cp.add(i));
                let o = _mm512_loadu_pd(op.add(i));
                let mk = valid_mask_avx512(h, l, c);

                let mlt = _mm512_cmp_pd_mask(c, o, _CMP_LT_OQ);
                let mgt = _mm512_cmp_pd_mask(c, o, _CMP_GT_OQ);
                let meq = (!mlt) & (!mgt);

                let p_lt = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, _mm512_add_pd(l, l)), c), v_quart);
                let p_gt = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(_mm512_add_pd(h, h), l), c), v_quart);
                let p_eq = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, l), _mm512_add_pd(c, c)), v_quart);

                let mut p = p_eq;
                p = _mm512_mask_blend_pd(mgt, p, p_gt);
                p = _mm512_mask_blend_pd(mlt, p, p_lt);
                _mm512_storeu_pd(ppp.add(i), _mm512_mask_blend_pd(mk, v_nan, p));

                let n_lt = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, _mm512_add_pd(l, l)), c), v_half);
                let n_gt = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(_mm512_add_pd(h, h), l), c), v_half);
                let n_eq = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, l), _mm512_add_pd(c, c)), v_half);

                let mut n = n_eq;
                n = _mm512_mask_blend_pd(mgt, n, n_gt);
                n = _mm512_mask_blend_pd(mlt, n, n_lt);

                let r1v = _mm512_sub_pd(n, l);
                let s1v = _mm512_sub_pd(n, h);

                _mm512_storeu_pd(r1p.add(i), _mm512_mask_blend_pd(mk, v_nan, r1v));
                _mm512_storeu_pd(s1p.add(i), _mm512_mask_blend_pd(mk, v_nan, s1v));
                _mm512_storeu_pd(r2p.add(i), v_nan);
                _mm512_storeu_pd(r3p.add(i), v_nan);
                _mm512_storeu_pd(r4p.add(i), v_nan);
                _mm512_storeu_pd(s2p.add(i), v_nan);
                _mm512_storeu_pd(s3p.add(i), v_nan);
                _mm512_storeu_pd(s4p.add(i), v_nan);

                i += step;
            }
        }
        3 => {
            while i + step <= len {
                let h = _mm512_loadu_pd(hp.add(i));
                let l = _mm512_loadu_pd(lp.add(i));
                let c = _mm512_loadu_pd(cp.add(i));
                let mk = valid_mask_avx512(h, l, c);

                let p = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, l), c), v_third);
                _mm512_storeu_pd(ppp.add(i), _mm512_mask_blend_pd(mk, v_nan, p));

                let d = _mm512_sub_pd(h, l);
                let d1 = _mm512_mul_pd(d, v_c0916);
                let d2 = _mm512_mul_pd(d, v_c0183);
                let d3 = _mm512_mul_pd(d, v_c0275);
                let d4 = _mm512_mul_pd(d, v_c0550);

                let r1v = _mm512_fmadd_pd(d, v_c0916, c);
                let r2v = _mm512_fmadd_pd(d, v_c0183, c);
                let r3v = _mm512_fmadd_pd(d, v_c0275, c);
                let r4v = _mm512_fmadd_pd(d, v_c0550, c);

                let s1v = _mm512_fmadd_pd(d, _mm512_sub_pd(_mm512_setzero_pd(), v_c0916), c);
                let s2v = _mm512_fmadd_pd(d, _mm512_sub_pd(_mm512_setzero_pd(), v_c0183), c);
                let s3v = _mm512_fmadd_pd(d, _mm512_sub_pd(_mm512_setzero_pd(), v_c0275), c);
                let s4v = _mm512_fmadd_pd(d, _mm512_sub_pd(_mm512_setzero_pd(), v_c0550), c);

                _mm512_storeu_pd(r1p.add(i), _mm512_mask_blend_pd(mk, v_nan, r1v));
                _mm512_storeu_pd(r2p.add(i), _mm512_mask_blend_pd(mk, v_nan, r2v));
                _mm512_storeu_pd(r3p.add(i), _mm512_mask_blend_pd(mk, v_nan, r3v));
                _mm512_storeu_pd(r4p.add(i), _mm512_mask_blend_pd(mk, v_nan, r4v));
                _mm512_storeu_pd(s1p.add(i), _mm512_mask_blend_pd(mk, v_nan, s1v));
                _mm512_storeu_pd(s2p.add(i), _mm512_mask_blend_pd(mk, v_nan, s2v));
                _mm512_storeu_pd(s3p.add(i), _mm512_mask_blend_pd(mk, v_nan, s3v));
                _mm512_storeu_pd(s4p.add(i), _mm512_mask_blend_pd(mk, v_nan, s4v));

                i += step;
            }
        }
        4 => {
            while i + step <= len {
                let h = _mm512_loadu_pd(hp.add(i));
                let l = _mm512_loadu_pd(lp.add(i));
                let c = _mm512_loadu_pd(cp.add(i));
                let o = _mm512_loadu_pd(op.add(i));
                let mk = valid_mask_avx512(h, l, c);

                let p = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(h, l), _mm512_add_pd(o, o)), v_quart);
                let t2p = _mm512_add_pd(p, p);
                let t2l = _mm512_add_pd(l, l);
                let t2h = _mm512_add_pd(h, h);
                let d = _mm512_sub_pd(h, l);

                let r3v = _mm512_add_pd(_mm512_sub_pd(t2p, t2l), h);
                let r4v = _mm512_fmadd_pd(d, v_one, r3v);
                let r2v = _mm512_fmadd_pd(d, v_one, p);
                let r1v = _mm512_sub_pd(t2p, l);

                let s1v = _mm512_sub_pd(t2p, h);
                let s2v = _mm512_fmadd_pd(d, v_neg1, p);
                let s3v = _mm512_sub_pd(_mm512_add_pd(l, t2p), t2h);
                let s4v = _mm512_fmadd_pd(d, v_neg1, s3v);

                _mm512_storeu_pd(ppp.add(i), _mm512_mask_blend_pd(mk, v_nan, p));
                _mm512_storeu_pd(r1p.add(i), _mm512_mask_blend_pd(mk, v_nan, r1v));
                _mm512_storeu_pd(r2p.add(i), _mm512_mask_blend_pd(mk, v_nan, r2v));
                _mm512_storeu_pd(r3p.add(i), _mm512_mask_blend_pd(mk, v_nan, r3v));
                _mm512_storeu_pd(r4p.add(i), _mm512_mask_blend_pd(mk, v_nan, r4v));
                _mm512_storeu_pd(s1p.add(i), _mm512_mask_blend_pd(mk, v_nan, s1v));
                _mm512_storeu_pd(s2p.add(i), _mm512_mask_blend_pd(mk, v_nan, s2v));
                _mm512_storeu_pd(s3p.add(i), _mm512_mask_blend_pd(mk, v_nan, s3v));
                _mm512_storeu_pd(s4p.add(i), _mm512_mask_blend_pd(mk, v_nan, s4v));

                i += step;
            }
        }
        _ => {}
    }

    if i < len {
        pivot_scalar(high, low, close, open, mode, i, r4, r3, r2, r1, pp, s1, s2, s3, s4);
    }
}

// ========== ROW "BATCH" VECTORIZED API ==========

#[inline(always)]
pub unsafe fn pivot_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx2(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx512(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx512_short(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx512_long(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}

// ========== NEW BATCH HELPERS ==========

// Compute one row-block (9 rows) for a given mode into pre-sliced outputs
#[inline(always)]
unsafe fn pivot_rows_scalar_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}

// New: flat batch "inner_into" mirroring alma_batch_inner_into
#[inline(always)]
fn pivot_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PivotParams>, PivotError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PivotError::EmptyData);
    }
    let cols = high.len();
    if cols == 0 || low.len() != cols || close.len() != cols || open.len() != cols {
        return Err(PivotError::EmptyData);
    }

    let first = first_valid_ohlc(high, low, close).ok_or(PivotError::AllValuesNaN)?;
    if first >= cols {
        return Err(PivotError::NotEnoughValidData);
    }

    let rows = combos.len() * N_LEVELS;
    // out is expected length rows*cols
    assert_eq!(
        out.len(),
        rows * cols,
        "pivot_batch_inner_into: out len mismatch"
    );

    // Warm prefixes for each of the 9 rows of each combo
    let warm: Vec<usize> = vec![first; rows];

    // Poison + warm NaNs using your helper
    let out_mu = unsafe {
        let mu = std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        );
        init_matrix_prefixes(mu, cols, &warm);
        mu
    };

    // Resolve kernel (scalar path is fine; stubs keep parity)
    let chosen = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicPtr, Ordering};

            // Wrap pointer in AtomicPtr for thread safety
            let out_ptr = AtomicPtr::new(out.as_mut_ptr());
            let out_len = out.len();

            // Drive indices; compute offsets and re-slice
            (0..combos.len()).into_par_iter().for_each(|ci| {
                let mode = combos[ci].mode.unwrap_or(3);
                // Compute 9 rows window for this combo
                let base = ci * N_LEVELS * cols;
                unsafe {
                    let ptr = out_ptr.load(Ordering::Relaxed);
                    let mu = std::slice::from_raw_parts_mut(
                        ptr as *mut std::mem::MaybeUninit<f64>,
                        out_len,
                    );
                    let mut rows_mu = mu[base..base + N_LEVELS * cols].chunks_mut(cols);
                    let mut cast = |mu: &mut [std::mem::MaybeUninit<f64>]| {
                        std::slice::from_raw_parts_mut(mu.as_mut_ptr() as *mut f64, mu.len())
                    };
                    let r4_mu = rows_mu.next().unwrap();
                    let r3_mu = rows_mu.next().unwrap();
                    let r2_mu = rows_mu.next().unwrap();
                    let r1_mu = rows_mu.next().unwrap();
                    let pp_mu = rows_mu.next().unwrap();
                    let s1_mu = rows_mu.next().unwrap();
                    let s2_mu = rows_mu.next().unwrap();
                    let s3_mu = rows_mu.next().unwrap();
                    let s4_mu = rows_mu.next().unwrap();

                    let r4 = cast(r4_mu);
                    let r3 = cast(r3_mu);
                    let r2 = cast(r2_mu);
                    let r1 = cast(r1_mu);
                    let pp = cast(pp_mu);
                    let s1 = cast(s1_mu);
                    let s2 = cast(s2_mu);
                    let s3 = cast(s3_mu);
                    let s4 = cast(s4_mu);

                    match chosen {
                        Kernel::Scalar | Kernel::ScalarBatch => pivot_rows_scalar_into(
                            high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
                        ),
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        Kernel::Avx2 | Kernel::Avx2Batch => pivot_row_avx2(
                            high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
                        ),
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        Kernel::Avx512 | Kernel::Avx512Batch => pivot_row_avx512(
                            high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
                        ),
                        _ => unreachable!(),
                    }
                }
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            // Sequential execution for WASM
            let mut row_chunks = out_mu.chunks_mut(cols);
            for p in &combos {
                let mode = p.mode.unwrap_or(3);
                unsafe {
                    let r4_mu = row_chunks.next().unwrap();
                    let r3_mu = row_chunks.next().unwrap();
                    let r2_mu = row_chunks.next().unwrap();
                    let r1_mu = row_chunks.next().unwrap();
                    let pp_mu = row_chunks.next().unwrap();
                    let s1_mu = row_chunks.next().unwrap();
                    let s2_mu = row_chunks.next().unwrap();
                    let s3_mu = row_chunks.next().unwrap();
                    let s4_mu = row_chunks.next().unwrap();

                    // Cast MU -> f64 slices without extra alloc/copy
                    let mut cast = |mu: &mut [std::mem::MaybeUninit<f64>]| {
                        std::slice::from_raw_parts_mut(mu.as_mut_ptr() as *mut f64, mu.len())
                    };
                    let (r4, r3, r2, r1, pp, s1, s2, s3, s4) = (
                        cast(r4_mu),
                        cast(r3_mu),
                        cast(r2_mu),
                        cast(r1_mu),
                        cast(pp_mu),
                        cast(s1_mu),
                        cast(s2_mu),
                        cast(s3_mu),
                        cast(s4_mu),
                    );

                    pivot_rows_scalar_into(
                        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
                    );
                }
            }
        }
    } else {
        // Sequential execution
        let mut row_chunks = out_mu.chunks_mut(cols);
        for p in &combos {
            let mode = p.mode.unwrap_or(3);
            unsafe {
                let r4_mu = row_chunks.next().unwrap();
                let r3_mu = row_chunks.next().unwrap();
                let r2_mu = row_chunks.next().unwrap();
                let r1_mu = row_chunks.next().unwrap();
                let pp_mu = row_chunks.next().unwrap();
                let s1_mu = row_chunks.next().unwrap();
                let s2_mu = row_chunks.next().unwrap();
                let s3_mu = row_chunks.next().unwrap();
                let s4_mu = row_chunks.next().unwrap();

                // Cast MU -> f64 slices without extra alloc/copy
                let mut cast = |mu: &mut [std::mem::MaybeUninit<f64>]| {
                    std::slice::from_raw_parts_mut(mu.as_mut_ptr() as *mut f64, mu.len())
                };
                let (r4, r3, r2, r1, pp, s1, s2, s3, s4) = (
                    cast(r4_mu),
                    cast(r3_mu),
                    cast(r2_mu),
                    cast(r1_mu),
                    cast(pp_mu),
                    cast(s1_mu),
                    cast(s2_mu),
                    cast(s3_mu),
                    cast(s4_mu),
                );

                pivot_rows_scalar_into(
                    high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
                );
            }
        }
    }
    Ok(combos)
}

// ========== BATCH (RANGE) API ==========

#[derive(Clone, Debug)]
pub struct PivotBatchRange {
    pub mode: (usize, usize, usize),
}
impl Default for PivotBatchRange {
    fn default() -> Self {
        Self { mode: (3, 3, 1) }
    }
}
#[derive(Clone, Debug, Default)]
pub struct PivotBatchBuilder {
    range: PivotBatchRange,
    kernel: Kernel,
}
impl PivotBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn mode_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.mode = (start, end, step);
        self
    }
    #[inline]
    pub fn mode_static(mut self, m: usize) -> Self {
        self.range.mode = (m, m, 1);
        self
    }
    pub fn apply_slice(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        open: &[f64],
    ) -> Result<PivotBatchOutput, PivotError> {
        pivot_batch_with_kernel(high, low, close, open, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<PivotBatchOutput, PivotError> {
        let high = source_type(candles, "high");
        let low = source_type(candles, "low");
        let close = source_type(candles, "close");
        let open = source_type(candles, "open");
        self.apply_slice(high, low, close, open)
    }
    pub fn with_default_candles(candles: &Candles) -> Result<PivotBatchOutput, PivotError> {
        PivotBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles)
    }
}

pub fn pivot_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    k: Kernel,
) -> Result<PivotBatchOutput, PivotError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PivotError::EmptyData),
    };
    pivot_batch_inner(high, low, close, open, sweep, kernel)
}

#[derive(Clone, Debug)]
pub struct PivotBatchOutput {
    pub levels: Vec<[Vec<f64>; 9]>,
    pub combos: Vec<PivotParams>,
    pub rows: usize,
    pub cols: usize,
}

// New: flat batch container like ALMA
#[derive(Clone, Debug)]
pub struct PivotBatchFlatOutput {
    pub values: Vec<f64>,         // row-major, rows = combos*9, cols = len
    pub combos: Vec<PivotParams>, // one per combo
    pub rows: usize,              // combos*9
    pub cols: usize,              // len
}

pub fn pivot_batch_flat_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    k: Kernel,
) -> Result<PivotBatchFlatOutput, PivotError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PivotError::EmptyData),
    };
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PivotError::EmptyData);
    }
    let cols = high.len();
    let rows = combos.len() * N_LEVELS;

    // Single allocation, MU -> f64 without copies
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> =
        vec![first_valid_ohlc(high, low, close).ok_or(PivotError::AllValuesNaN)?; rows];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    pivot_batch_inner_into(high, low, close, open, sweep, kernel, true, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(PivotBatchFlatOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn expand_grid(r: &PivotBatchRange) -> Vec<PivotParams> {
    let (start, end, step) = r.mode;
    let mut v = Vec::new();
    let mut m = start;
    while m <= end {
        v.push(PivotParams { mode: Some(m) });
        m += step;
    }
    v
}
fn pivot_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    kernel: Kernel,
) -> Result<PivotBatchOutput, PivotError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PivotError::EmptyData);
    }
    let len = high.len();
    let mut levels = Vec::with_capacity(combos.len());
    for p in &combos {
        let mode = p.mode.unwrap_or(3);
        let mut first = None;
        for i in 0..len {
            if !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()) {
                first = Some(i);
                break;
            }
        }
        let first = first.unwrap_or(len);

        // Allocate output vectors with NaN prefix
        let mut r4 = alloc_with_nan_prefix(len, first);
        let mut r3 = alloc_with_nan_prefix(len, first);
        let mut r2 = alloc_with_nan_prefix(len, first);
        let mut r1 = alloc_with_nan_prefix(len, first);
        let mut pp = alloc_with_nan_prefix(len, first);
        let mut s1 = alloc_with_nan_prefix(len, first);
        let mut s2 = alloc_with_nan_prefix(len, first);
        let mut s3 = alloc_with_nan_prefix(len, first);
        let mut s4 = alloc_with_nan_prefix(len, first);
        unsafe {
            match kernel {
                Kernel::Scalar | Kernel::ScalarBatch => pivot_row_scalar(
                    high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1,
                    &mut pp, &mut s1, &mut s2, &mut s3, &mut s4,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => pivot_row_avx2(
                    high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1,
                    &mut pp, &mut s1, &mut s2, &mut s3, &mut s4,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => pivot_row_avx512(
                    high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1,
                    &mut pp, &mut s1, &mut s2, &mut s3, &mut s4,
                ),
                _ => unreachable!(),
            }
        }
        levels.push([r4, r3, r2, r1, pp, s1, s2, s3, s4]);
    }
    let rows = combos.len();
    let cols = high.len();
    Ok(PivotBatchOutput {
        levels,
        combos,
        rows,
        cols,
    })
}

// ========== STREAMING INTERFACE ==========

/// Streaming pivot calculation
/// Note: Pivot is not truly a streaming indicator as it requires complete period data.
/// This implementation maintains a single pivot level based on the most recent data point.
pub struct PivotStream {
    mode: usize,
}

impl PivotStream {
    pub fn new(mode: usize) -> Self {
        Self { mode }
    }

    pub fn try_new(params: PivotParams) -> Result<Self, PivotError> {
        let mode = params.mode.unwrap_or(3);
        if mode > 4 {
            return Err(PivotError::EmptyData); // Using existing error for invalid mode
        }
        Ok(Self { mode })
    }

    /// Update with new OHLC data and return pivot levels
    /// Returns tuple of (r4, r3, r2, r1, pp, s1, s2, s3, s4)
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        open: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        if high.is_nan() || low.is_nan() || close.is_nan() || open.is_nan() {
            return None;
        }

        let p = match self.mode {
            2 => {
                // Demark
                if close < open {
                    (high + 2.0 * low + close) / 4.0
                } else if close > open {
                    (2.0 * high + low + close) / 4.0
                } else {
                    (high + low + 2.0 * close) / 4.0
                }
            }
            4 => (high + low + 2.0 * open) / 4.0, // Woodie
            _ => (high + low + close) / 3.0,      // Standard/Fibonacci/Camarilla
        };

        let (r4, r3, r2, r1, s1, s2, s3, s4) = match self.mode {
            0 => {
                // Standard
                let r1 = 2.0 * p - low;
                let r2 = p + (high - low);
                let s1 = 2.0 * p - high;
                let s2 = p - (high - low);
                (f64::NAN, f64::NAN, r2, r1, s1, s2, f64::NAN, f64::NAN)
            }
            1 => {
                // Fibonacci
                let r1 = p + 0.382 * (high - low);
                let r2 = p + 0.618 * (high - low);
                let r3 = p + 1.0 * (high - low);
                let s1 = p - 0.382 * (high - low);
                let s2 = p - 0.618 * (high - low);
                let s3 = p - 1.0 * (high - low);
                (f64::NAN, r3, r2, r1, s1, s2, s3, f64::NAN)
            }
            2 => {
                // Demark
                let r1 = if close < open {
                    (high + 2.0 * low + close) / 2.0 - low
                } else if close > open {
                    (2.0 * high + low + close) / 2.0 - low
                } else {
                    (high + low + 2.0 * close) / 2.0 - low
                };
                let s1 = if close < open {
                    (high + 2.0 * low + close) / 2.0 - high
                } else if close > open {
                    (2.0 * high + low + close) / 2.0 - high
                } else {
                    (high + low + 2.0 * close) / 2.0 - high
                };
                (
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    r1,
                    s1,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                )
            }
            3 => {
                // Camarilla
                let r4 = (0.55 * (high - low)) + close;
                let r3 = (0.275 * (high - low)) + close;
                let r2 = (0.183 * (high - low)) + close;
                let r1 = (0.0916 * (high - low)) + close;
                let s1 = close - (0.0916 * (high - low));
                let s2 = close - (0.183 * (high - low));
                let s3 = close - (0.275 * (high - low));
                let s4 = close - (0.55 * (high - low));
                (r4, r3, r2, r1, s1, s2, s3, s4)
            }
            4 => {
                // Woodie
                let r3 = high + 2.0 * (p - low);
                let r4 = r3 + (high - low);
                let r2 = p + (high - low);
                let r1 = 2.0 * p - low;
                let s1 = 2.0 * p - high;
                let s2 = p - (high - low);
                let s3 = low - 2.0 * (high - p);
                let s4 = s3 - (high - low);
                (r4, r3, r2, r1, s1, s2, s3, s4)
            }
            _ => return None,
        };

        Some((r4, r3, r2, r1, p, s1, s2, s3, s4))
    }
}

// ========== PYTHON BINDINGS ==========

#[cfg(feature = "python")]
#[pyfunction(name = "pivot")]
#[pyo3(signature = (high, low, close, open, mode=3, kernel=None))]
pub fn pivot_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    open: PyReadonlyArray1<'py, f64>,
    mode: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let open_slice = open.as_slice()?;

    let kern = validate_kernel(kernel, false)?;

    let params = PivotParams { mode: Some(mode) };
    let input = PivotInput::from_slices(high_slice, low_slice, close_slice, open_slice, params);

    let result = py
        .allow_threads(|| pivot_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        result.r4.into_pyarray(py),
        result.r3.into_pyarray(py),
        result.r2.into_pyarray(py),
        result.r1.into_pyarray(py),
        result.pp.into_pyarray(py),
        result.s1.into_pyarray(py),
        result.s2.into_pyarray(py),
        result.s3.into_pyarray(py),
        result.s4.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "PivotStream")]
pub struct PivotStreamPy {
    inner: PivotStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PivotStreamPy {
    #[new]
    fn new(mode: Option<usize>) -> PyResult<Self> {
        let params = PivotParams { mode };
        let inner =
            PivotStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PivotStreamPy { inner })
    }

    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        open: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.inner.update(high, low, close, open)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "pivot_batch")]
#[pyo3(signature = (high, low, close, open, mode_range, kernel=None))]
pub fn pivot_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    open: PyReadonlyArray1<'py, f64>,
    mode_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let (h, l, c, o) = (
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        open.as_slice()?,
    );
    let sweep = PivotBatchRange { mode: mode_range };
    let kern = validate_kernel(kernel, true)?;

    // Compute flat once using zero-copy path
    let flat = py
        .allow_threads(|| pivot_batch_flat_with_kernel(h, l, c, o, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Split flat values into 9 views of shape (combos, cols) flattened (row-major)
    let combos = flat.combos.len();
    let cols = flat.cols;
    let vals = flat.values; // take ownership; move into NumPy without copy

    // One NumPy buffer, then create 9 arrays as views without extra copies
    let arr = unsafe { PyArray1::<f64>::new(py, [vals.len()], false) };
    unsafe {
        arr.as_slice_mut()?.copy_from_slice(&vals);
    } // single copy from our owned Vec into NumPy storage

    // Produce 9 arrays as separate 2D arrays
    let dict = PyDict::new(py);
    let names = ["r4", "r3", "r2", "r1", "pp", "s1", "s2", "s3", "s4"];

    for (li, name) in names.iter().enumerate() {
        // Create a new array for this level
        let level_arr = unsafe { PyArray1::<f64>::new(py, [combos * cols], false) };
        let level_slice = unsafe { level_arr.as_slice_mut()? };

        // Copy data for this level from the flat array
        // The data is organized as [combo0_r4...combo0_s4, combo1_r4...combo1_s4, ...]
        // We need to extract every 9th element starting from li
        for combo_idx in 0..combos {
            let base_idx = combo_idx * N_LEVELS * cols;
            let level_base = li * cols;
            for col_idx in 0..cols {
                level_slice[combo_idx * cols + col_idx] = vals[base_idx + level_base + col_idx];
            }
        }

        dict.set_item(*name, level_arr.reshape((combos, cols))?)?;
    }

    dict.set_item(
        "modes",
        flat.combos
            .iter()
            .map(|p| p.mode.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows_per_level", combos)?; // for each level
    dict.set_item("cols", cols)?;
    Ok(dict)
}

// ========== WASM BINDINGS ==========

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
) -> Result<Vec<f64>, JsValue> {
    // Check for mismatched lengths explicitly
    let len = high.len();
    if low.len() != len || close.len() != len || open.len() != len {
        return Err(JsValue::from_str(
            "pivot: Input arrays must have the same length",
        ));
    }

    let params = PivotParams { mode: Some(mode) };
    let input = PivotInput::from_slices(high, low, close, open, params);
    let out =
        pivot_with_kernel(&input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let cols = high.len();
    let mut values = Vec::with_capacity(N_LEVELS * cols);
    values.extend_from_slice(&out.r4);
    values.extend_from_slice(&out.r3);
    values.extend_from_slice(&out.r2);
    values.extend_from_slice(&out.r1);
    values.extend_from_slice(&out.pp);
    values.extend_from_slice(&out.s1);
    values.extend_from_slice(&out.s2);
    values.extend_from_slice(&out.s3);
    values.extend_from_slice(&out.s4);
    Ok(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    open_ptr: *const f64,
    r4_ptr: *mut f64,
    r3_ptr: *mut f64,
    r2_ptr: *mut f64,
    r1_ptr: *mut f64,
    pp_ptr: *mut f64,
    s1_ptr: *mut f64,
    s2_ptr: *mut f64,
    s3_ptr: *mut f64,
    s4_ptr: *mut f64,
    len: usize,
    mode: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || open_ptr.is_null() {
        return Err(JsValue::from_str("Null input pointer provided"));
    }

    if r4_ptr.is_null()
        || r3_ptr.is_null()
        || r2_ptr.is_null()
        || r1_ptr.is_null()
        || pp_ptr.is_null()
        || s1_ptr.is_null()
        || s2_ptr.is_null()
        || s3_ptr.is_null()
        || s4_ptr.is_null()
    {
        return Err(JsValue::from_str("Null output pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let open = std::slice::from_raw_parts(open_ptr, len);

        let params = PivotParams { mode: Some(mode) };
        let input = PivotInput::from_slices(high, low, close, open, params);

        // Check for any aliasing between inputs and outputs
        let input_ptrs = [
            high_ptr as *const u8,
            low_ptr as *const u8,
            close_ptr as *const u8,
            open_ptr as *const u8,
        ];
        let output_ptrs = [
            r4_ptr as *const u8,
            r3_ptr as *const u8,
            r2_ptr as *const u8,
            r1_ptr as *const u8,
            pp_ptr as *const u8,
            s1_ptr as *const u8,
            s2_ptr as *const u8,
            s3_ptr as *const u8,
            s4_ptr as *const u8,
        ];

        let has_aliasing = input_ptrs
            .iter()
            .any(|&inp| output_ptrs.iter().any(|&out| inp == out));

        if has_aliasing {
            // Use single temporary buffer if there's aliasing
            let mut temp = vec![0.0; len * 9];

            // Split into slices
            let (r4_temp, rest) = temp.split_at_mut(len);
            let (r3_temp, rest) = rest.split_at_mut(len);
            let (r2_temp, rest) = rest.split_at_mut(len);
            let (r1_temp, rest) = rest.split_at_mut(len);
            let (pp_temp, rest) = rest.split_at_mut(len);
            let (s1_temp, rest) = rest.split_at_mut(len);
            let (s2_temp, rest) = rest.split_at_mut(len);
            let (s3_temp, s4_temp) = rest.split_at_mut(len);

            pivot_into_slices(
                r4_temp,
                r3_temp,
                r2_temp,
                r1_temp,
                pp_temp,
                s1_temp,
                s2_temp,
                s3_temp,
                s4_temp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let r4_out = std::slice::from_raw_parts_mut(r4_ptr, len);
            let r3_out = std::slice::from_raw_parts_mut(r3_ptr, len);
            let r2_out = std::slice::from_raw_parts_mut(r2_ptr, len);
            let r1_out = std::slice::from_raw_parts_mut(r1_ptr, len);
            let pp_out = std::slice::from_raw_parts_mut(pp_ptr, len);
            let s1_out = std::slice::from_raw_parts_mut(s1_ptr, len);
            let s2_out = std::slice::from_raw_parts_mut(s2_ptr, len);
            let s3_out = std::slice::from_raw_parts_mut(s3_ptr, len);
            let s4_out = std::slice::from_raw_parts_mut(s4_ptr, len);

            r4_out.copy_from_slice(r4_temp);
            r3_out.copy_from_slice(r3_temp);
            r2_out.copy_from_slice(r2_temp);
            r1_out.copy_from_slice(r1_temp);
            pp_out.copy_from_slice(pp_temp);
            s1_out.copy_from_slice(s1_temp);
            s2_out.copy_from_slice(s2_temp);
            s3_out.copy_from_slice(s3_temp);
            s4_out.copy_from_slice(s4_temp);
        } else {
            // Direct computation into output slices
            let r4_out = std::slice::from_raw_parts_mut(r4_ptr, len);
            let r3_out = std::slice::from_raw_parts_mut(r3_ptr, len);
            let r2_out = std::slice::from_raw_parts_mut(r2_ptr, len);
            let r1_out = std::slice::from_raw_parts_mut(r1_ptr, len);
            let pp_out = std::slice::from_raw_parts_mut(pp_ptr, len);
            let s1_out = std::slice::from_raw_parts_mut(s1_ptr, len);
            let s2_out = std::slice::from_raw_parts_mut(s2_ptr, len);
            let s3_out = std::slice::from_raw_parts_mut(s3_ptr, len);
            let s4_out = std::slice::from_raw_parts_mut(s4_ptr, len);

            pivot_into_slices(
                r4_out,
                r3_out,
                r2_out,
                r1_out,
                pp_out,
                s1_out,
                s2_out,
                s3_out,
                s4_out,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PivotBatchConfig {
    pub mode_range: (usize, usize, usize), // (start, end, step)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PivotBatchFlatJsOutput {
    pub values: Vec<f64>, // row-major, rows = combos*9, cols = len
    pub modes: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = pivot_batch)]
pub fn pivot_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: PivotBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let sweep = PivotBatchRange {
        mode: cfg.mode_range,
    };
    let flat = pivot_batch_flat_with_kernel(high, low, close, open, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let modes = flat.combos.iter().map(|p| p.mode.unwrap()).collect();
    let out = PivotBatchFlatJsOutput {
        values: flat.values,
        modes,
        rows: flat.rows,
        cols: flat.cols,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;
    use paste::paste;

    fn check_pivot_default_mode_camarilla(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = PivotParams { mode: None };
        let input = PivotInput::from_candles(&candles, params);
        let result = pivot_with_kernel(&input, kernel)?;

        assert_eq!(result.r4.len(), candles.close.len());
        assert_eq!(result.r3.len(), candles.close.len());
        assert_eq!(result.r2.len(), candles.close.len());
        assert_eq!(result.r1.len(), candles.close.len());
        assert_eq!(result.pp.len(), candles.close.len());
        assert_eq!(result.s1.len(), candles.close.len());
        assert_eq!(result.s2.len(), candles.close.len());
        assert_eq!(result.s3.len(), candles.close.len());
        assert_eq!(result.s4.len(), candles.close.len());

        // Spot-check Camarilla outputs for a few points
        let last_five_r4 = &result.r4[result.r4.len().saturating_sub(5)..];
        let expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
        for (i, &val) in last_five_r4.iter().enumerate() {
            let exp = expected_r4[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla r4 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_pivot_nan_values(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, f64::NAN, 30.0];
        let low = [9.0, 8.5, f64::NAN];
        let close = [9.5, 9.0, 29.0];
        let open = [9.1, 8.8, 28.5];

        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel)?;
        assert_eq!(result.pp.len(), high.len());
        Ok(())
    }

    fn check_pivot_no_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let open: [f64; 0] = [];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("One or more required fields"),
                "Expected 'EmptyData' error, got: {}",
                e
            );
        }
        Ok(())
    }

    fn check_pivot_all_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let open = [f64::NAN, f64::NAN];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'AllValuesNaN' error, got: {}",
                e
            );
        }
        Ok(())
    }

    fn check_pivot_fibonacci_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(1) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r3.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_standard_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(0) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r2.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_demark_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(2) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r1.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_woodie_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(4) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r4.len(), candles.close.len());
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_pivot_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy to generate diverse OHLC data including edge cases
        let strat = (10usize..=200).prop_flat_map(|len| {
            // Mix of normal and edge case data generation
            prop_oneof![
                // Normal case: realistic price movements
                prop::collection::vec(
                    (100f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                    len,
                )
                .prop_flat_map(move |base_prices| {
                    let ohlc_strat = prop::collection::vec(
                        (0f64..1f64, 0f64..1f64, 0f64..1f64, 0f64..1f64),
                        len,
                    );

                    (ohlc_strat, 0usize..=4).prop_map(move |(factors, mode)| {
                        let mut high_data = Vec::with_capacity(len);
                        let mut low_data = Vec::with_capacity(len);
                        let mut close_data = Vec::with_capacity(len);
                        let mut open_data = Vec::with_capacity(len);

                        for (i, base) in base_prices.iter().enumerate() {
                            let (high_factor, low_factor, close_factor, open_factor) = factors[i];

                            // Generate realistic OHLC where high >= low
                            let range = base * 0.1; // 10% range
                            let low = base - range * low_factor;
                            let high = base + range * high_factor;
                            let open = low + (high - low) * open_factor;
                            let close = low + (high - low) * close_factor;

                            high_data.push(high);
                            low_data.push(low);
                            open_data.push(open);
                            close_data.push(close);
                        }

                        (high_data, low_data, close_data, open_data, mode)
                    })
                }),
                // Edge case: flat market (all prices equal)
                (100f64..1000f64, 0usize..=4).prop_map(move |(price, mode)| {
                    let data = vec![price; len];
                    (data.clone(), data.clone(), data.clone(), data, mode)
                }),
                // Edge case: very small price differences
                (100f64..1000f64, 0usize..=4).prop_map(move |(base, mode)| {
                    let mut high_data = Vec::with_capacity(len);
                    let mut low_data = Vec::with_capacity(len);
                    let mut close_data = Vec::with_capacity(len);
                    let mut open_data = Vec::with_capacity(len);

                    for _ in 0..len {
                        let epsilon = 1e-10;
                        let low = base;
                        let high = base + epsilon;
                        let open = base + epsilon * 0.3;
                        let close = base + epsilon * 0.7;

                        high_data.push(high);
                        low_data.push(low);
                        open_data.push(open);
                        close_data.push(close);
                    }

                    (high_data, low_data, close_data, open_data, mode)
                }),
            ]
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(high, low, close, open, mode)| {
                let params = PivotParams { mode: Some(mode) };
                let input = PivotInput::from_slices(&high, &low, &close, &open, params);

                let output = pivot_with_kernel(&input, kernel)?;
                let ref_output = pivot_with_kernel(&input, Kernel::Scalar)?;

                // Verify output lengths
                prop_assert_eq!(output.pp.len(), high.len());
                prop_assert_eq!(output.r1.len(), high.len());
                prop_assert_eq!(output.s1.len(), high.len());

                for i in 0..high.len() {
                    let h = high[i];
                    let l = low[i];
                    let c = close[i];
                    let o = open[i];

                    // Skip if any input is NaN
                    if h.is_nan() || l.is_nan() || c.is_nan() || o.is_nan() {
                        continue;
                    }

                    let pp = output.pp[i];
                    let r4 = output.r4[i];
                    let r3 = output.r3[i];
                    let r2 = output.r2[i];
                    let r1 = output.r1[i];
                    let s1 = output.s1[i];
                    let s2 = output.s2[i];
                    let s3 = output.s3[i];
                    let s4 = output.s4[i];

                    let tolerance = 1e-9;
                    let range = h - l;

                    // Comprehensive formula verification for each mode
                    match mode {
                        0 => {
                            // Standard Mode
                            let expected_pp = (h + l + c) / 3.0;
                            prop_assert!(
                                (pp - expected_pp).abs() < tolerance,
                                "Standard PP at {}: {} vs {}",
                                i,
                                pp,
                                expected_pp
                            );

                            // R1 = 2*PP - L
                            let expected_r1 = 2.0 * pp - l;
                            prop_assert!(
                                (r1 - expected_r1).abs() < tolerance,
                                "Standard R1 at {}: {} vs {}",
                                i,
                                r1,
                                expected_r1
                            );

                            // R2 = PP + (H - L)
                            let expected_r2 = pp + range;
                            prop_assert!(
                                (r2 - expected_r2).abs() < tolerance,
                                "Standard R2 at {}: {} vs {}",
                                i,
                                r2,
                                expected_r2
                            );

                            // S1 = 2*PP - H
                            let expected_s1 = 2.0 * pp - h;
                            prop_assert!(
                                (s1 - expected_s1).abs() < tolerance,
                                "Standard S1 at {}: {} vs {}",
                                i,
                                s1,
                                expected_s1
                            );

                            // S2 = PP - (H - L)
                            let expected_s2 = pp - range;
                            prop_assert!(
                                (s2 - expected_s2).abs() < tolerance,
                                "Standard S2 at {}: {} vs {}",
                                i,
                                s2,
                                expected_s2
                            );

                            // R3, R4, S3, S4 should be NaN for Standard mode
                            prop_assert!(r3.is_nan(), "Standard R3 should be NaN at {}", i);
                            prop_assert!(r4.is_nan(), "Standard R4 should be NaN at {}", i);
                            prop_assert!(s3.is_nan(), "Standard S3 should be NaN at {}", i);
                            prop_assert!(s4.is_nan(), "Standard S4 should be NaN at {}", i);

                            // Verify ordering: S2 < S1 < PP < R1 < R2
                            prop_assert!(s2 <= s1 + tolerance, "S2 > S1 at {}", i);
                            prop_assert!(s1 <= pp + tolerance, "S1 > PP at {}", i);
                            prop_assert!(pp <= r1 + tolerance, "PP > R1 at {}", i);
                            prop_assert!(r1 <= r2 + tolerance, "R1 > R2 at {}", i);
                        }
                        1 => {
                            // Fibonacci Mode
                            let expected_pp = (h + l + c) / 3.0;
                            prop_assert!(
                                (pp - expected_pp).abs() < tolerance,
                                "Fibonacci PP at {}: {} vs {}",
                                i,
                                pp,
                                expected_pp
                            );

                            // Fibonacci ratios
                            let expected_r1 = pp + 0.382 * range;
                            let expected_r2 = pp + 0.618 * range;
                            let expected_r3 = pp + 1.0 * range;
                            let expected_s1 = pp - 0.382 * range;
                            let expected_s2 = pp - 0.618 * range;
                            let expected_s3 = pp - 1.0 * range;

                            prop_assert!(
                                (r1 - expected_r1).abs() < tolerance,
                                "Fibonacci R1 at {}: {} vs {}",
                                i,
                                r1,
                                expected_r1
                            );
                            prop_assert!(
                                (r2 - expected_r2).abs() < tolerance,
                                "Fibonacci R2 at {}: {} vs {}",
                                i,
                                r2,
                                expected_r2
                            );
                            prop_assert!(
                                (r3 - expected_r3).abs() < tolerance,
                                "Fibonacci R3 at {}: {} vs {}",
                                i,
                                r3,
                                expected_r3
                            );
                            prop_assert!(
                                (s1 - expected_s1).abs() < tolerance,
                                "Fibonacci S1 at {}: {} vs {}",
                                i,
                                s1,
                                expected_s1
                            );
                            prop_assert!(
                                (s2 - expected_s2).abs() < tolerance,
                                "Fibonacci S2 at {}: {} vs {}",
                                i,
                                s2,
                                expected_s2
                            );
                            prop_assert!(
                                (s3 - expected_s3).abs() < tolerance,
                                "Fibonacci S3 at {}: {} vs {}",
                                i,
                                s3,
                                expected_s3
                            );

                            // R4, S4 should be NaN for Fibonacci
                            prop_assert!(r4.is_nan(), "Fibonacci R4 should be NaN at {}", i);
                            prop_assert!(s4.is_nan(), "Fibonacci S4 should be NaN at {}", i);

                            // Verify ordering
                            prop_assert!(s3 <= s2 + tolerance, "S3 > S2 at {}", i);
                            prop_assert!(s2 <= s1 + tolerance, "S2 > S1 at {}", i);
                            prop_assert!(s1 <= pp + tolerance, "S1 > PP at {}", i);
                            prop_assert!(pp <= r1 + tolerance, "PP > R1 at {}", i);
                            prop_assert!(r1 <= r2 + tolerance, "R1 > R2 at {}", i);
                            prop_assert!(r2 <= r3 + tolerance, "R2 > R3 at {}", i);
                        }
                        2 => {
                            // Demark Mode
                            let expected_pp = if c < o {
                                (h + 2.0 * l + c) / 4.0
                            } else if c > o {
                                (2.0 * h + l + c) / 4.0
                            } else {
                                (h + l + 2.0 * c) / 4.0
                            };
                            prop_assert!(
                                (pp - expected_pp).abs() < tolerance,
                                "Demark PP at {}: {} vs {}",
                                i,
                                pp,
                                expected_pp
                            );

                            // Demark has special R1/S1 calculations
                            let expected_r1 = if c < o {
                                (h + 2.0 * l + c) / 2.0 - l
                            } else if c > o {
                                (2.0 * h + l + c) / 2.0 - l
                            } else {
                                (h + l + 2.0 * c) / 2.0 - l
                            };
                            let expected_s1 = if c < o {
                                (h + 2.0 * l + c) / 2.0 - h
                            } else if c > o {
                                (2.0 * h + l + c) / 2.0 - h
                            } else {
                                (h + l + 2.0 * c) / 2.0 - h
                            };

                            prop_assert!(
                                (r1 - expected_r1).abs() < tolerance,
                                "Demark R1 at {}: {} vs {}",
                                i,
                                r1,
                                expected_r1
                            );
                            prop_assert!(
                                (s1 - expected_s1).abs() < tolerance,
                                "Demark S1 at {}: {} vs {}",
                                i,
                                s1,
                                expected_s1
                            );

                            // Other levels should be NaN for Demark
                            prop_assert!(r2.is_nan(), "Demark R2 should be NaN at {}", i);
                            prop_assert!(r3.is_nan(), "Demark R3 should be NaN at {}", i);
                            prop_assert!(r4.is_nan(), "Demark R4 should be NaN at {}", i);
                            prop_assert!(s2.is_nan(), "Demark S2 should be NaN at {}", i);
                            prop_assert!(s3.is_nan(), "Demark S3 should be NaN at {}", i);
                            prop_assert!(s4.is_nan(), "Demark S4 should be NaN at {}", i);
                        }
                        3 => {
                            // Camarilla Mode
                            let expected_pp = (h + l + c) / 3.0;
                            prop_assert!(
                                (pp - expected_pp).abs() < tolerance,
                                "Camarilla PP at {}: {} vs {}",
                                i,
                                pp,
                                expected_pp
                            );

                            // Camarilla specific multipliers
                            let expected_r4 = 0.55 * range + c;
                            let expected_r3 = 0.275 * range + c;
                            let expected_r2 = 0.183 * range + c;
                            let expected_r1 = 0.0916 * range + c;
                            let expected_s1 = c - 0.0916 * range;
                            let expected_s2 = c - 0.183 * range;
                            let expected_s3 = c - 0.275 * range;
                            let expected_s4 = c - 0.55 * range;

                            prop_assert!(
                                (r4 - expected_r4).abs() < tolerance,
                                "Camarilla R4 at {}: {} vs {}",
                                i,
                                r4,
                                expected_r4
                            );
                            prop_assert!(
                                (r3 - expected_r3).abs() < tolerance,
                                "Camarilla R3 at {}: {} vs {}",
                                i,
                                r3,
                                expected_r3
                            );
                            prop_assert!(
                                (r2 - expected_r2).abs() < tolerance,
                                "Camarilla R2 at {}: {} vs {}",
                                i,
                                r2,
                                expected_r2
                            );
                            prop_assert!(
                                (r1 - expected_r1).abs() < tolerance,
                                "Camarilla R1 at {}: {} vs {}",
                                i,
                                r1,
                                expected_r1
                            );
                            prop_assert!(
                                (s1 - expected_s1).abs() < tolerance,
                                "Camarilla S1 at {}: {} vs {}",
                                i,
                                s1,
                                expected_s1
                            );
                            prop_assert!(
                                (s2 - expected_s2).abs() < tolerance,
                                "Camarilla S2 at {}: {} vs {}",
                                i,
                                s2,
                                expected_s2
                            );
                            prop_assert!(
                                (s3 - expected_s3).abs() < tolerance,
                                "Camarilla S3 at {}: {} vs {}",
                                i,
                                s3,
                                expected_s3
                            );
                            prop_assert!(
                                (s4 - expected_s4).abs() < tolerance,
                                "Camarilla S4 at {}: {} vs {}",
                                i,
                                s4,
                                expected_s4
                            );

                            // Verify ordering
                            prop_assert!(s4 <= s3 + tolerance, "S4 > S3 at {}", i);
                            prop_assert!(s3 <= s2 + tolerance, "S3 > S2 at {}", i);
                            prop_assert!(s2 <= s1 + tolerance, "S2 > S1 at {}", i);
                            prop_assert!(r1 <= r2 + tolerance, "R1 > R2 at {}", i);
                            prop_assert!(r2 <= r3 + tolerance, "R2 > R3 at {}", i);
                            prop_assert!(r3 <= r4 + tolerance, "R3 > R4 at {}", i);
                        }
                        4 => {
                            // Woodie Mode
                            let expected_pp = (h + l + 2.0 * o) / 4.0;
                            prop_assert!(
                                (pp - expected_pp).abs() < tolerance,
                                "Woodie PP at {}: {} vs {}",
                                i,
                                pp,
                                expected_pp
                            );

                            // Woodie calculations
                            let expected_r1 = 2.0 * pp - l;
                            let expected_r2 = pp + range;
                            let expected_r3 = h + 2.0 * (pp - l);
                            let expected_r4 = expected_r3 + range;
                            let expected_s1 = 2.0 * pp - h;
                            let expected_s2 = pp - range;
                            let expected_s3 = l - 2.0 * (h - pp);
                            let expected_s4 = expected_s3 - range;

                            prop_assert!(
                                (r1 - expected_r1).abs() < tolerance,
                                "Woodie R1 at {}: {} vs {}",
                                i,
                                r1,
                                expected_r1
                            );
                            prop_assert!(
                                (r2 - expected_r2).abs() < tolerance,
                                "Woodie R2 at {}: {} vs {}",
                                i,
                                r2,
                                expected_r2
                            );
                            prop_assert!(
                                (r3 - expected_r3).abs() < tolerance,
                                "Woodie R3 at {}: {} vs {}",
                                i,
                                r3,
                                expected_r3
                            );
                            prop_assert!(
                                (r4 - expected_r4).abs() < tolerance,
                                "Woodie R4 at {}: {} vs {}",
                                i,
                                r4,
                                expected_r4
                            );
                            prop_assert!(
                                (s1 - expected_s1).abs() < tolerance,
                                "Woodie S1 at {}: {} vs {}",
                                i,
                                s1,
                                expected_s1
                            );
                            prop_assert!(
                                (s2 - expected_s2).abs() < tolerance,
                                "Woodie S2 at {}: {} vs {}",
                                i,
                                s2,
                                expected_s2
                            );
                            prop_assert!(
                                (s3 - expected_s3).abs() < tolerance,
                                "Woodie S3 at {}: {} vs {}",
                                i,
                                s3,
                                expected_s3
                            );
                            prop_assert!(
                                (s4 - expected_s4).abs() < tolerance,
                                "Woodie S4 at {}: {} vs {}",
                                i,
                                s4,
                                expected_s4
                            );

                            // Verify ordering
                            prop_assert!(s4 <= s3 + tolerance, "S4 > S3 at {}", i);
                            prop_assert!(s3 <= s2 + tolerance, "S3 > S2 at {}", i);
                            prop_assert!(s2 <= s1 + tolerance, "S2 > S1 at {}", i);
                            prop_assert!(r1 <= r2 + tolerance, "R1 > R2 at {}", i);
                            prop_assert!(r2 <= r3 + tolerance, "R2 > R3 at {}", i);
                            prop_assert!(r3 <= r4 + tolerance, "R3 > R4 at {}", i);
                        }
                        _ => {}
                    }

                    // Verify kernel consistency
                    prop_assert!(
                        (pp - ref_output.pp[i]).abs() < tolerance,
                        "PP kernel mismatch at {}",
                        i
                    );
                    prop_assert!(
                        (r1 - ref_output.r1[i]).abs() < tolerance
                            || (r1.is_nan() && ref_output.r1[i].is_nan()),
                        "R1 kernel mismatch at {}",
                        i
                    );
                    prop_assert!(
                        (s1 - ref_output.s1[i]).abs() < tolerance
                            || (s1.is_nan() && ref_output.s1[i].is_nan()),
                        "S1 kernel mismatch at {}",
                        i
                    );

                    // Check for poison values in debug builds
                    #[cfg(debug_assertions)]
                    {
                        let check_poison = |val: f64, name: &str| {
                            if !val.is_nan() {
                                let bits = val.to_bits();
                                prop_assert_ne!(
                                    bits,
                                    0x11111111_11111111,
                                    "{} poison at {}",
                                    name,
                                    i
                                );
                                prop_assert_ne!(
                                    bits,
                                    0x22222222_22222222,
                                    "{} poison at {}",
                                    name,
                                    i
                                );
                                prop_assert_ne!(
                                    bits,
                                    0x33333333_33333333,
                                    "{} poison at {}",
                                    name,
                                    i
                                );
                            }
                            Ok(())
                        };

                        check_poison(pp, "PP")?;
                        check_poison(r4, "R4")?;
                        check_poison(r3, "R3")?;
                        check_poison(r2, "R2")?;
                        check_poison(r1, "R1")?;
                        check_poison(s1, "S1")?;
                        check_poison(s2, "S2")?;
                        check_poison(s3, "S3")?;
                        check_poison(s4, "S4")?;
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_pivot_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations covering all modes
        let test_params = vec![
            PivotParams::default(),        // mode: 3 (Camarilla)
            PivotParams { mode: Some(0) }, // Standard
            PivotParams { mode: Some(1) }, // Fibonacci
            PivotParams { mode: Some(2) }, // Demark
            PivotParams { mode: Some(3) }, // Camarilla (explicit)
            PivotParams { mode: Some(4) }, // Woodie
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = PivotInput::from_candles(&candles, params.clone());
            let output = pivot_with_kernel(&input, kernel)?;

            // Check all 9 output arrays
            let arrays = vec![
                ("r4", &output.r4),
                ("r3", &output.r3),
                ("r2", &output.r2),
                ("r1", &output.r1),
                ("pp", &output.pp),
                ("s1", &output.s1),
                ("s2", &output.s2),
                ("s3", &output.s3),
                ("s4", &output.s4),
            ];

            for (array_name, values) in arrays {
                for (i, &val) in values.iter().enumerate() {
                    if val.is_nan() {
                        continue; // NaN values are expected during warmup
                    }

                    let bits = val.to_bits();

                    // Check all three poison patterns
                    if bits == 0x11111111_11111111 {
                        panic!(
							"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
							 in array {} with params: {:?} (param set {})",
							test_name, val, bits, i, array_name, params, param_idx
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
							 in array {} with params: {:?} (param set {})",
							test_name, val, bits, i, array_name, params, param_idx
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
							 in array {} with params: {:?} (param set {})",
							test_name, val, bits, i, array_name, params, param_idx
						);
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_pivot_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    fn check_pivot_batch_default_row(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let output = PivotBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;
        let default = PivotParams::default();
        let def_idx = output
            .combos
            .iter()
            .position(|p| p.mode == default.mode)
            .expect("default row missing");
        for arr in &output.levels[def_idx] {
            assert_eq!(arr.len(), candles.close.len());
        }
        Ok(())
    }

    // Macro for all kernel variants
    macro_rules! generate_all_pivot_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() { let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() { let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2); }
                    #[test]
                    fn [<$test_fn _avx512>]() { let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512); }
                )*
                $(
                    #[test]
                    fn [<$test_fn _auto_detect>]() { let _ = $test_fn(stringify!([<$test_fn _auto_detect>]), Kernel::Auto); }
                )*
            }
        }
    }

    generate_all_pivot_tests!(
        check_pivot_default_mode_camarilla,
        check_pivot_nan_values,
        check_pivot_no_data,
        check_pivot_all_nan,
        check_pivot_fibonacci_mode,
        check_pivot_standard_mode,
        check_pivot_demark_mode,
        check_pivot_woodie_mode,
        check_pivot_batch_default_row,
        check_pivot_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_pivot_tests!(check_pivot_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let output = PivotBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;

        let def = PivotParams::default();
        let row = output
            .combos
            .iter()
            .position(|p| p.mode == def.mode)
            .expect("default row missing");
        let levels = &output.levels[row];

        // Spot check: each level should be the right length
        for arr in levels.iter() {
            assert_eq!(arr.len(), candles.close.len());
        }

        // Optionally, spot-check some values (e.g. Camarilla r4)
        let expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
        let r4 = &levels[0];
        let last_five_r4 = &r4[r4.len().saturating_sub(5)..];
        for (i, &val) in last_five_r4.iter().enumerate() {
            let exp = expected_r4[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "[{test}] Camarilla r4 mismatch at idx {i}: {val} vs {exp:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations for mode
        let test_configs = vec![
            (0, 2, 1), // Small range: Standard, Fibonacci, Demark
            (0, 4, 1), // Full range: all modes
            (0, 4, 2), // Skip modes: Standard, Demark, Woodie
            (1, 3, 1), // Middle modes: Fibonacci, Demark, Camarilla
            (3, 4, 1), // Last two: Camarilla, Woodie
            (2, 2, 1), // Single mode: Demark only
            (0, 0, 1), // Single mode: Standard only
        ];

        for (cfg_idx, &(mode_start, mode_end, mode_step)) in test_configs.iter().enumerate() {
            let output = PivotBatchBuilder::new()
                .kernel(kernel)
                .mode_range(mode_start, mode_end, mode_step)
                .apply_candles(&c)?;

            // Check all 9 arrays for each parameter combination
            for (row_idx, levels) in output.levels.iter().enumerate() {
                let combo = &output.combos[row_idx];

                // Check each of the 9 arrays
                for (level_idx, level_array) in levels.iter().enumerate() {
                    let level_name = match level_idx {
                        0 => "r4",
                        1 => "r3",
                        2 => "r2",
                        3 => "r1",
                        4 => "pp",
                        5 => "s1",
                        6 => "s2",
                        7 => "s3",
                        8 => "s4",
                        _ => "unknown",
                    };

                    for (col, &val) in level_array.iter().enumerate() {
                        if val.is_nan() {
                            continue;
                        }

                        let bits = val.to_bits();

                        // Check all three poison patterns with detailed context
                        if bits == 0x11111111_11111111 {
                            panic!(
								"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
								 at row {} col {} in array {} with params: {:?}",
								test, cfg_idx, val, bits, row_idx, col, level_name, combo
							);
                        }

                        if bits == 0x22222222_22222222 {
                            panic!(
								"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
								 at row {} col {} in array {} with params: {:?}",
								test, cfg_idx, val, bits, row_idx, col, level_name, combo
							);
                        }

                        if bits == 0x33333333_33333333 {
                            panic!(
								"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
								 at row {} col {} in array {} with params: {:?}",
								test, cfg_idx, val, bits, row_idx, col, level_name, combo
							);
                        }
                    }
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
        Ok(()) // No-op in release builds
    }

    // Kernel variant macro expansion (as in alma.rs)
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
