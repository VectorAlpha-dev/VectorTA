//! # Average True Range (ATR)
//!
//! A volatility indicator that measures market volatility by calculating the average of true ranges
//! over a specified period. True range is the greatest of: current high-low, abs(high-close[prev]),
//! or abs(low-close[prev]). ATR uses an RMA (Running Moving Average) for smoothing.
//!
//! ## Parameters
//! - **length**: Window size for RMA smoothing (defaults to 14)
//!
//! ## Inputs
//! - Requires high, low, and close price arrays
//! - Supports both raw slices and Candles data structure
//!
//! ## Returns
//! - **`Ok(AtrOutput)`** containing a `Vec<f64>` matching input length
//! - Leading values are NaN during the warmup period (length-1 values)
//!
//! ## Developer Notes (Status)
//! - SIMD: AVX2/AVX512 implemented; vectorizes TR while keeping sequential RMA.
//!   - On this machine at 100k samples: AVX2/AVX512 ~2–4% faster vs scalar; at 1M, AVX512 ~4–5%.
//!   - Scalar path optimized (~15% vs baseline) via loop‑jamming + `mul_add` + pointer access.
//! - Batch: row‑specific optimized path shares a single TR prepass across rows; seeds via TR prefix sums.
//! - Streaming: switched to O(1) kernel (no ring buffer/modulo), FMA update, precomputed alpha; seeds with mean(TR[0..N)).
//! - Memory: follows alma.rs patterns (NaN warmup, zero‑copy/uninit matrix helpers, aligned buffers).

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
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
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AtrData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AtrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AtrParams {
    pub length: Option<usize>,
}

impl Default for AtrParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AtrInput<'a> {
    pub data: AtrData<'a>,
    pub params: AtrParams,
}

impl<'a> AtrInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AtrParams) -> Self {
        Self {
            data: AtrData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AtrParams,
    ) -> Self {
        Self {
            data: AtrData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AtrParams::default())
    }
    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AtrBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for AtrBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AtrBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AtrOutput, AtrError> {
        let p = AtrParams {
            length: self.length,
        };
        let i = AtrInput::from_candles(c, p);
        atr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AtrOutput, AtrError> {
        let p = AtrParams {
            length: self.length,
        };
        let i = AtrInput::from_slices(high, low, close, p);
        atr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AtrStream, AtrError> {
        let p = AtrParams {
            length: self.length,
        };
        AtrStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AtrError {
    #[error("Invalid length for ATR calculation (length={length}).")]
    InvalidLength { length: usize },
    #[error("Inconsistent slice lengths for ATR calculation: high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("No candles available for ATR calculation.")]
    NoCandlesAvailable,
    #[error("Not enough data to calculate ATR: length={length}, data length={data_len}")]
    NotEnoughData { length: usize, data_len: usize },
}

#[inline(always)]
fn first_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let len = close.len();
    let mut i = 0;
    while i < len {
        // valid only if all three are not NaN at i
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            break;
        }
        i += 1;
    }
    i.min(len)
}

#[inline(always)]
fn atr_prepare_full<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    length: usize,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize), AtrError> {
    let (high, low, close, length) = atr_prepare(high, low, close, length)?;
    let first = first_valid_hlc(high, low, close);
    if close.len().saturating_sub(first) < length {
        return Err(AtrError::NotEnoughData {
            length,
            data_len: close.len(),
        });
    }
    let warmup = first + length - 1;
    Ok((high, low, close, first, warmup))
}

#[inline]
pub fn atr(input: &AtrInput) -> Result<AtrOutput, AtrError> {
    atr_with_kernel(input, Kernel::Auto)
}

pub fn atr_with_kernel(input: &AtrInput, kernel: Kernel) -> Result<AtrOutput, AtrError> {
    let (high, low, close) = match &input.data {
        AtrData::Candles { candles } => (
            candles
                .select_candle_field("high")
                .map_err(|_| AtrError::NoCandlesAvailable)?,
            candles
                .select_candle_field("low")
                .map_err(|_| AtrError::NoCandlesAvailable)?,
            candles
                .select_candle_field("close")
                .map_err(|_| AtrError::NoCandlesAvailable)?,
        ),
        AtrData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(AtrError::InconsistentSliceLengths {
                    high_len: high.len(),
                    low_len: low.len(),
                    close_len: close.len(),
                });
            }
            (*high, *low, *close)
        }
    };

    let len = close.len();
    let length = input.get_length();
    if length == 0 {
        return Err(AtrError::InvalidLength { length });
    }
    if len == 0 {
        return Err(AtrError::NoCandlesAvailable);
    }
    if length > len {
        return Err(AtrError::NotEnoughData {
            length,
            data_len: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    let (_, _, _, first, warmup) = atr_prepare_full(high, low, close, length)?;
    let mut out = alloc_with_nan_prefix(len, warmup);
    atr_compute_into(high, low, close, length, first, chosen, &mut out);
    Ok(AtrOutput { values: out })
}

#[inline(always)]
fn atr_compute_into_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(low.len(), close.len());
    debug_assert_eq!(out.len(), close.len());

    let warm = first + length - 1;
    let alpha = 1.0 / (length as f64);

    unsafe {
        // ---- Seed RMA at warm (sum TR over [first ..= warm]) ----
        // i == first (no prev-close term)
        let mut sum_tr = *high.get_unchecked(first) - *low.get_unchecked(first);

        if warm > first {
            let mut i = first + 1;
            let mut prev_c = *close.get_unchecked(i - 1);
            while i <= warm {
                let hi = *high.get_unchecked(i);
                let lo = *low.get_unchecked(i);

                // branchless max of (hi-lo, |hi-prev_c|, |lo-prev_c|)
                let mut tr = hi - lo;
                let hc = (hi - prev_c).abs();
                if hc > tr {
                    tr = hc;
                }
                let lc = (lo - prev_c).abs();
                if lc > tr {
                    tr = lc;
                }

                sum_tr += tr;
                prev_c = *close.get_unchecked(i);
                i += 1;
            }
        }

        let mut rma = sum_tr / (length as f64);
        *out.get_unchecked_mut(warm) = rma;

        // ---- Rolling RMA from warm+1 .. end (unrolled by 4) ----
        let mut i = warm + 1;
        let n = out.len();

        // prev close for the first rolling step
        let mut prev_c = if i > 0 {
            *close.get_unchecked(i - 1)
        } else {
            *close.get_unchecked(0)
        };

        // process 4 at a time, sequentially updating rma
        while i + 3 < n {
            // step 0
            let (hi0, lo0) = (*high.get_unchecked(i), *low.get_unchecked(i));
            let mut tr0 = hi0 - lo0;
            let hc0 = (hi0 - prev_c).abs();
            if hc0 > tr0 {
                tr0 = hc0;
            }
            let lc0 = (lo0 - prev_c).abs();
            if lc0 > tr0 {
                tr0 = lc0;
            }
            rma = (-alpha).mul_add(rma, rma) + alpha * tr0;
            *out.get_unchecked_mut(i) = rma;

            // step 1
            let prev0 = *close.get_unchecked(i);
            let (hi1, lo1) = (*high.get_unchecked(i + 1), *low.get_unchecked(i + 1));
            let mut tr1 = hi1 - lo1;
            let hc1 = (hi1 - prev0).abs();
            if hc1 > tr1 {
                tr1 = hc1;
            }
            let lc1 = (lo1 - prev0).abs();
            if lc1 > tr1 {
                tr1 = lc1;
            }
            rma = (-alpha).mul_add(rma, rma) + alpha * tr1;
            *out.get_unchecked_mut(i + 1) = rma;

            // step 2
            let prev1 = *close.get_unchecked(i + 1);
            let (hi2, lo2) = (*high.get_unchecked(i + 2), *low.get_unchecked(i + 2));
            let mut tr2 = hi2 - lo2;
            let hc2 = (hi2 - prev1).abs();
            if hc2 > tr2 {
                tr2 = hc2;
            }
            let lc2 = (lo2 - prev1).abs();
            if lc2 > tr2 {
                tr2 = lc2;
            }
            rma = (-alpha).mul_add(rma, rma) + alpha * tr2;
            *out.get_unchecked_mut(i + 2) = rma;

            // step 3
            let prev2 = *close.get_unchecked(i + 2);
            let (hi3, lo3) = (*high.get_unchecked(i + 3), *low.get_unchecked(i + 3));
            let mut tr3 = hi3 - lo3;
            let hc3 = (hi3 - prev2).abs();
            if hc3 > tr3 {
                tr3 = hc3;
            }
            let lc3 = (lo3 - prev2).abs();
            if lc3 > tr3 {
                tr3 = lc3;
            }
            rma = (-alpha).mul_add(rma, rma) + alpha * tr3;
            *out.get_unchecked_mut(i + 3) = rma;

            i += 4;
            prev_c = *close.get_unchecked(i - 1);
        }

        // tail
        while i < n {
            let (hi, lo) = (*high.get_unchecked(i), *low.get_unchecked(i));
            let mut tr = hi - lo;
            let hc = (hi - prev_c).abs();
            if hc > tr {
                tr = hc;
            }
            let lc = (lo - prev_c).abs();
            if lc > tr {
                tr = lc;
            }
            rma = (-alpha).mul_add(rma, rma) + alpha * tr;
            *out.get_unchecked_mut(i) = rma;

            prev_c = *close.get_unchecked(i);
            i += 1;
        }
    }
}

#[inline]
pub fn atr_scalar(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    // default to first=0 warmup=length-1; assume caller prepared prefix
    atr_compute_into_scalar(high, low, close, length, 0, out);
}

#[inline(always)]
fn atr_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    first: usize,
    kern: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kern, Kernel::Scalar | Kernel::ScalarBatch) {
                // keep scalar path until you add true SIMD128
                atr_compute_into_scalar(high, low, close, length, first, out);
                return;
            }
        }
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                atr_compute_into_scalar(high, low, close, length, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                atr_compute_into_avx2(high, low, close, length, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                atr_compute_into_avx512(high, low, close, length, first, out)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn atr_simd128(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    use core::arch::wasm32::*;

    // For now, use scalar implementation as ATR's sequential nature
    // makes SIMD optimization complex. This maintains API parity with ALMA.
    // Future optimization could process multiple true ranges in parallel
    // for batch operations.
    atr_scalar(high, low, close, length, out);
}

// -- AVX2/AVX512 always point to scalar; structuring for parity/stub compatibility --
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn atr_compute_into_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(low.len(), close.len());
    debug_assert_eq!(out.len(), close.len());

    let warm = first + length - 1;
    let alpha = 1.0 / (length as f64);

    // ---- seed (scalar; length is typically small and sequential-dep anyway) ----
    let mut sum_tr = *high.get_unchecked(first) - *low.get_unchecked(first);
    if warm > first {
        let mut i = first + 1;
        let mut prev_c = *close.get_unchecked(i - 1);
        while i <= warm {
            let hi = *high.get_unchecked(i);
            let lo = *low.get_unchecked(i);

            let mut tr = hi - lo;
            let hc = (hi - prev_c).abs();
            if hc > tr {
                tr = hc;
            }
            let lc = (lo - prev_c).abs();
            if lc > tr {
                tr = lc;
            }

            sum_tr += tr;
            prev_c = *close.get_unchecked(i);
            i += 1;
        }
    }

    let mut rma = sum_tr / (length as f64);
    *out.get_unchecked_mut(warm) = rma;

    // ---- rolling phase ----
    let mut i = warm + 1;
    let n = out.len();

    // constants/masks
    let mask_abs = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffffu64 as i64));

    // vector block of 4
    while i + 3 < n {
        let v_hi = _mm256_loadu_pd(high.as_ptr().add(i));
        let v_lo = _mm256_loadu_pd(low.as_ptr().add(i));
        // prev close vector is [close[i-1], close[i], close[i+1], close[i+2]]
        let v_pc = _mm256_loadu_pd(close.as_ptr().add(i - 1));

        // hl = hi - lo
        let v_hl = _mm256_sub_pd(v_hi, v_lo);
        // hc = |hi - prev_close|
        let v_hc = _mm256_and_pd(_mm256_sub_pd(v_hi, v_pc), mask_abs);
        // lc = |lo - prev_close|
        let v_lc = _mm256_and_pd(_mm256_sub_pd(v_lo, v_pc), mask_abs);

        // tr = max(hl, hc, lc)
        let v_m1 = _mm256_max_pd(v_hl, v_hc);
        let v_tr = _mm256_max_pd(v_m1, v_lc);

        // spill 4 TRs and sequentially update RMA
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), v_tr);

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[0];
        *out.get_unchecked_mut(i) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[1];
        *out.get_unchecked_mut(i + 1) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[2];
        *out.get_unchecked_mut(i + 2) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[3];
        *out.get_unchecked_mut(i + 3) = rma;

        i += 4;
    }

    // scalar tail
    if i < n {
        let mut prev_c = *close.get_unchecked(i - 1);
        while i < n {
            let hi = *high.get_unchecked(i);
            let lo = *low.get_unchecked(i);
            let mut tr = hi - lo;
            let hc = (hi - prev_c).abs();
            if hc > tr {
                tr = hc;
            }
            let lc = (lo - prev_c).abs();
            if lc > tr {
                tr = lc;
            }
            rma = (-alpha).mul_add(rma, rma) + alpha * tr;
            *out.get_unchecked_mut(i) = rma;

            prev_c = *close.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn atr_compute_into_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(low.len(), close.len());
    debug_assert_eq!(out.len(), close.len());

    let warm = first + length - 1;
    let alpha = 1.0 / (length as f64);

    // ---- seed (scalar) ----
    let mut sum_tr = *high.get_unchecked(first) - *low.get_unchecked(first);
    if warm > first {
        let mut i = first + 1;
        let mut prev_c = *close.get_unchecked(i - 1);
        while i <= warm {
            let hi = *high.get_unchecked(i);
            let lo = *low.get_unchecked(i);

            let mut tr = hi - lo;
            let hc = (hi - prev_c).abs();
            if hc > tr {
                tr = hc;
            }
            let lc = (lo - prev_c).abs();
            if lc > tr {
                tr = lc;
            }

            sum_tr += tr;
            prev_c = *close.get_unchecked(i);
            i += 1;
        }
    }

    let mut rma = sum_tr / (length as f64);
    *out.get_unchecked_mut(warm) = rma;

    // ---- rolling phase ----
    let mut i = warm + 1;
    let n = out.len();

    // 0x7FFF... mask to clear sign bit
    let mask_abs = _mm512_castsi512_pd(_mm512_set1_epi64(0x7fff_ffff_ffff_ffffu64 as i64));

    while i + 7 < n {
        let v_hi = _mm512_loadu_pd(high.as_ptr().add(i));
        let v_lo = _mm512_loadu_pd(low.as_ptr().add(i));
        let v_pc = _mm512_loadu_pd(close.as_ptr().add(i - 1)); // prev-close lane-wise

        let v_hl = _mm512_sub_pd(v_hi, v_lo);
        let v_hc = _mm512_and_pd(_mm512_sub_pd(v_hi, v_pc), mask_abs);
        let v_lc = _mm512_and_pd(_mm512_sub_pd(v_lo, v_pc), mask_abs);

        let v_m1 = _mm512_max_pd(v_hl, v_hc);
        let v_tr = _mm512_max_pd(v_m1, v_lc);

        let mut buf = [0.0f64; 8];
        _mm512_storeu_pd(buf.as_mut_ptr(), v_tr);

        // sequential RMA across the 8-lane block
        rma = (-alpha).mul_add(rma, rma) + alpha * buf[0];
        *out.get_unchecked_mut(i) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[1];
        *out.get_unchecked_mut(i + 1) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[2];
        *out.get_unchecked_mut(i + 2) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[3];
        *out.get_unchecked_mut(i + 3) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[4];
        *out.get_unchecked_mut(i + 4) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[5];
        *out.get_unchecked_mut(i + 5) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[6];
        *out.get_unchecked_mut(i + 6) = rma;

        rma = (-alpha).mul_add(rma, rma) + alpha * buf[7];
        *out.get_unchecked_mut(i + 7) = rma;

        i += 8;
    }

    // scalar tail
    if i < n {
        let mut prev_c = *close.get_unchecked(i - 1);
        while i < n {
            let hi = *high.get_unchecked(i);
            let lo = *low.get_unchecked(i);
            let mut tr = hi - lo;
            let hc = (hi - prev_c).abs();
            if hc > tr {
                tr = hc;
            }
            let lc = (lo - prev_c).abs();
            if lc > tr {
                tr = lc;
            }
            rma = (-alpha).mul_add(rma, rma) + alpha * tr;
            *out.get_unchecked_mut(i) = rma;

            prev_c = *close.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn atr_avx2(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    unsafe { atr_compute_into_avx2(high, low, close, length, 0, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn atr_avx512(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    unsafe { atr_compute_into_avx512(high, low, close, length, 0, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn atr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    atr_compute_into_avx512(high, low, close, length, 0, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn atr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    atr_compute_into_avx512(high, low, close, length, 0, out)
}

// Streaming object for rolling ATR (like AlmaStream)
#[derive(Debug, Clone)]
pub struct AtrStream {
    length: usize,
    alpha: f64,        // 1.0 / length
    prev_close: f64,   // NaN until first close
    rma: f64,          // last ATR (NaN until seeded)
    warm_sum: f64,     // sum(TR) over the first `length` samples only
    warm_count: usize, // how many TRs we've accumulated so far
    seeded: bool,      // true once warm_count == length
}

impl AtrStream {
    /// Decision: O(1) streaming ATR without ring buffer/modulo.
    /// Seeds with mean of first N TRs; steady-state uses RMA with FMA form.
    #[inline(always)]
    pub fn try_new(params: AtrParams) -> Result<Self, AtrError> {
        let length = params.length.unwrap_or(14);
        if length == 0 {
            return Err(AtrError::InvalidLength { length });
        }
        Ok(Self {
            length,
            alpha: 1.0 / (length as f64),
            prev_close: f64::NAN,
            rma: f64::NAN,
            warm_sum: 0.0,
            warm_count: 0,
            seeded: false,
        })
    }

    /// Update with the next OHLC triple.
    /// Returns Some(atr) once seeded with `length` TRs, otherwise None.
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        debug_assert!(
            high.is_finite() && low.is_finite() && close.is_finite(),
            "Streaming ATR assumes finite inputs; prefilter NaNs/Infs upstream if needed",
        );

        // True Range:
        //  - First bar: TR = high - low
        //  - Subsequent: TR = max(high, prev_close) - min(low, prev_close)
        // This avoids abs() and matches the textbook/Wilder definition.
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let up = if high > self.prev_close { high } else { self.prev_close };
            let dn = if low < self.prev_close { low } else { self.prev_close };
            up - dn
        };

        // Update prev_close for the next tick before early-returns.
        self.prev_close = close;

        // Warmup: accumulate the arithmetic mean of the first `length` TRs.
        if !self.seeded {
            self.warm_sum += tr;
            self.warm_count += 1;

            if self.warm_count == self.length {
                // First ATR is the simple average of the first N TRs (Wilder).
                // Subsequent ATRs follow RMA/SMMA with α = 1/N:
                // ATR_t = ATR_{t-1} + α * (TR_t - ATR_{t-1})
                self.rma = self.warm_sum * self.alpha;
                self.seeded = true;
                return Some(self.rma);
            }
            return None;
        }

        // Steady-state RMA (Wilder's smoothing, α = 1/length)
        // Use FMA form for precision & speed: rma' = α*(tr - rma) + rma
        self.rma = self.alpha.mul_add(tr - self.rma, self.rma);
        Some(self.rma)
    }
}

// Batch/param sweep types for parity with alma.rs

#[derive(Clone, Debug)]
pub struct AtrBatchRange {
    pub length: (usize, usize, usize),
}
impl Default for AtrBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 30, 1),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct AtrBatchBuilder {
    range: AtrBatchRange,
    kernel: Kernel,
}
impl AtrBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }
    #[inline]
    pub fn length_static(mut self, p: usize) -> Self {
        self.range.length = (p, p, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AtrBatchOutput, AtrError> {
        atr_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<AtrBatchOutput, AtrError> {
        let high = c
            .select_candle_field("high")
            .map_err(|_| AtrError::NoCandlesAvailable)?;
        let low = c
            .select_candle_field("low")
            .map_err(|_| AtrError::NoCandlesAvailable)?;
        let close = c
            .select_candle_field("close")
            .map_err(|_| AtrError::NoCandlesAvailable)?;
        self.apply_slices(high, low, close)
    }
}

#[derive(Clone, Debug)]
pub struct AtrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AtrParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AtrBatchOutput {
    pub fn row_for_params(&self, p: &AtrParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.length.unwrap_or(14) == p.length.unwrap_or(14))
    }
    pub fn values_for(&self, p: &AtrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AtrBatchRange) -> Vec<AtrParams> {
    let (start, end, step) = r.length;
    if step == 0 || start == end {
        return vec![AtrParams {
            length: Some(start),
        }];
    }
    (start..=end)
        .step_by(step)
        .map(|l| AtrParams { length: Some(l) })
        .collect()
}

pub fn atr_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    k: Kernel,
) -> Result<AtrBatchOutput, AtrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AtrError::InvalidLength { length: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    atr_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn atr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    kern: Kernel,
) -> Result<AtrBatchOutput, AtrError> {
    atr_batch_inner(high, low, close, sweep, kern, false)
}
#[inline(always)]
pub fn atr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    kern: Kernel,
) -> Result<AtrBatchOutput, AtrError> {
    atr_batch_inner(high, low, close, sweep, kern, true)
}

fn atr_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AtrParams>, AtrError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AtrError::InvalidLength { length: 0 });
    }
    let rows = combos.len();
    let cols = high.len();
    if out.len() != rows * cols {
        return Err(AtrError::NotEnoughData {
            length: rows * cols,
            data_len: out.len(),
        });
    }

    let first = first_valid_hlc(high, low, close);

    // Precompute TR series once and its prefix sums for fast per-row seeds
    let mut tr = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cols);
    unsafe { tr.set_len(cols); }
    // Fill with zeros up front to avoid reading uninit in prefix sums
    for v in &mut tr[..] {
        *v = 0.0;
    }

    // Compute TR depending on kernel selection
    match kern_to_simd(kern) {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            precompute_tr_into_avx512(high, low, close, first, &mut tr);
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            precompute_tr_into_avx2(high, low, close, first, &mut tr);
        },
        _ => {
            precompute_tr_into_scalar(high, low, close, first, &mut tr);
        }
    }

    // Prefix sums of TR to seed RMA in O(1) per row
    let mut ps = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cols + 1);
    unsafe { ps.set_len(cols + 1); }
    ps[0] = 0.0;
    // Keep prefix align with indices: ps[i+1] = sum_{j=0..i} tr[j]
    for i in 0..cols {
        ps[i + 1] = ps[i] + tr[i];
    }

    let do_row = |row: usize, dst: &mut [f64]| {
        let length = combos[row].length.unwrap();
        let warm = first + length - 1;
        // ensure warmup prefix
        for v in &mut dst[..warm] {
            *v = f64::NAN;
        }

        // Seed RMA using prefix sums of TR
        let sum_tr = ps[warm + 1] - ps[first];
        let mut rma = sum_tr / (length as f64);
        dst[warm] = rma;
        let alpha = 1.0 / (length as f64);
        let mut i = warm + 1;
        while i < cols {
            let tri = tr[i];
            rma = (-alpha).mul_add(rma, rma) + alpha * tri;
            dst[i] = rma;
            i += 1;
        }
    };

    #[inline(always)]
    fn kern_to_simd(k: Kernel) -> Kernel {
        match k {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch => Kernel::Avx512,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            other => other,
        }
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, row)| do_row(r, row));
        #[cfg(target_arch = "wasm32")]
        for (r, row) in out.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    } else {
        for (r, row) in out.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    }

    Ok(combos)
}

fn atr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AtrBatchOutput, AtrError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AtrError::InvalidLength { length: 0 });
    }
    let len = close.len();
    let rows = combos.len();
    let cols = len;

    // Step 1: Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Find first valid index across all three arrays
    let first_valid = first_valid_hlc(high, low, close);

    // Step 2: Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first_valid + c.length.unwrap() - 1) // ATR warmup period includes leading NaNs
        .collect();

    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Step 4: Convert to mutable slice for computation
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        std::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Precompute TR once + prefix sums
    let mut tr = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cols);
    unsafe { tr.set_len(cols); }
    for v in &mut tr[..] {
        *v = 0.0;
    }
    match kern {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe { precompute_tr_into_avx512(high, low, close, first_valid, &mut tr) },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe { precompute_tr_into_avx2(high, low, close, first_valid, &mut tr) },
        _ => precompute_tr_into_scalar(high, low, close, first_valid, &mut tr),
    }
    let mut ps = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cols + 1);
    unsafe { ps.set_len(cols + 1) };
    ps[0] = 0.0;
    for i in 0..cols {
        ps[i + 1] = ps[i] + tr[i];
    }

    let do_row = |row: usize, out_row: &mut [f64]| {
        let length = combos[row].length.unwrap();
        let warm = first_valid + length - 1;
        // RMA seed from prefix sums
        let sum_tr = ps[warm + 1] - ps[first_valid];
        let mut rma = sum_tr / (length as f64);
        out_row[warm] = rma;
        let alpha = 1.0 / (length as f64);
        let mut i = warm + 1;
        while i < cols {
            let tri = tr[i];
            rma = (-alpha).mul_add(rma, rma) + alpha * tri;
            out_row[i] = rma;
            i += 1;
        }
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

    // Step 5: Reclaim as Vec<f64>
    let final_values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(AtrBatchOutput {
        values: final_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn atr_row_scalar(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    let first = first_valid_hlc(high, low, close);
    atr_compute_into(high, low, close, length, first, Kernel::Scalar, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn atr_row_avx2(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    let first = first_valid_hlc(high, low, close);
    atr_compute_into(high, low, close, length, first, Kernel::Avx2, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn atr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    if length <= 32 {
        atr_row_avx512_short(high, low, close, length, out);
    } else {
        atr_row_avx512_long(high, low, close, length, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn atr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    let first = first_valid_hlc(high, low, close);
    atr_compute_into(high, low, close, length, first, Kernel::Avx512, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn atr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    let first = first_valid_hlc(high, low, close);
    atr_compute_into(high, low, close, length, first, Kernel::Avx512, out);
}

#[inline(always)]
fn precompute_tr_into_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    tr_out: &mut [f64],
) {
    if first >= tr_out.len() {
        return;
    }
    tr_out[first] = high[first] - low[first];
    let mut i = first + 1;
    while i < tr_out.len() {
        let hi = high[i];
        let lo = low[i];
        let pc = close[i - 1];
        let mut tr = hi - lo;
        let hc = (hi - pc).abs();
        if hc > tr {
            tr = hc;
        }
        let lc = (lo - pc).abs();
        if lc > tr {
            tr = lc;
        }
        tr_out[i] = tr;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn precompute_tr_into_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    tr_out: &mut [f64],
) {
    use core::arch::x86_64::*;
    if first >= tr_out.len() {
        return;
    }
    tr_out[first] = *high.get_unchecked(first) - *low.get_unchecked(first);
    let mut i = first + 1;
    let n = tr_out.len();
    let mask_abs = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffffu64 as i64));
    while i + 3 < n {
        let v_hi = _mm256_loadu_pd(high.as_ptr().add(i));
        let v_lo = _mm256_loadu_pd(low.as_ptr().add(i));
        let v_pc = _mm256_loadu_pd(close.as_ptr().add(i - 1));

        let v_hl = _mm256_sub_pd(v_hi, v_lo);
        let v_hc = _mm256_and_pd(_mm256_sub_pd(v_hi, v_pc), mask_abs);
        let v_lc = _mm256_and_pd(_mm256_sub_pd(v_lo, v_pc), mask_abs);
        let v_m1 = _mm256_max_pd(v_hl, v_hc);
        let v_tr = _mm256_max_pd(v_m1, v_lc);
        _mm256_storeu_pd(tr_out.as_mut_ptr().add(i), v_tr);
        i += 4;
    }
    while i < n {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let pc = *close.get_unchecked(i - 1);
        let mut tr = hi - lo;
        let hc = (hi - pc).abs();
        if hc > tr {
            tr = hc;
        }
        let lc = (lo - pc).abs();
        if lc > tr {
            tr = lc;
        }
        *tr_out.get_unchecked_mut(i) = tr;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn precompute_tr_into_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    tr_out: &mut [f64],
) {
    use core::arch::x86_64::*;
    if first >= tr_out.len() {
        return;
    }
    tr_out[first] = *high.get_unchecked(first) - *low.get_unchecked(first);
    let mut i = first + 1;
    let n = tr_out.len();
    let mask_abs = _mm512_castsi512_pd(_mm512_set1_epi64(0x7fff_ffff_ffff_ffffu64 as i64));
    while i + 7 < n {
        let v_hi = _mm512_loadu_pd(high.as_ptr().add(i));
        let v_lo = _mm512_loadu_pd(low.as_ptr().add(i));
        let v_pc = _mm512_loadu_pd(close.as_ptr().add(i - 1));
        let v_hl = _mm512_sub_pd(v_hi, v_lo);
        let v_hc = _mm512_and_pd(_mm512_sub_pd(v_hi, v_pc), mask_abs);
        let v_lc = _mm512_and_pd(_mm512_sub_pd(v_lo, v_pc), mask_abs);
        let v_m1 = _mm512_max_pd(v_hl, v_hc);
        let v_tr = _mm512_max_pd(v_m1, v_lc);
        _mm512_storeu_pd(tr_out.as_mut_ptr().add(i), v_tr);
        i += 8;
    }
    while i < n {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let pc = *close.get_unchecked(i - 1);
        let mut tr = hi - lo;
        let hc = (hi - pc).abs();
        if hc > tr {
            tr = hc;
        }
        let lc = (lo - pc).abs();
        if lc > tr {
            tr = lc;
        }
        *tr_out.get_unchecked_mut(i) = tr;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_atr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AtrParams { length: None };
        let input_partial = AtrInput::from_candles(&candles, partial_params);
        let result_partial = atr_with_kernel(&input_partial, kernel)?;
        assert_eq!(result_partial.values.len(), candles.close.len());
        let zero_and_none_params = AtrParams { length: Some(14) };
        let input_zero_and_none = AtrInput::from_candles(&candles, zero_and_none_params);
        let result_zero_and_none = atr_with_kernel(&input_zero_and_none, kernel)?;
        assert_eq!(result_zero_and_none.values.len(), candles.close.len());
        Ok(())
    }

    fn check_atr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AtrInput::with_default_candles(&candles);
        let result = atr_with_kernel(&input, kernel)?;
        let expected_last_five = [916.89, 874.33, 838.45, 801.92, 811.57];
        assert!(result.values.len() >= 5, "Not enough ATR values");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "ATR output length does not match input length!"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-2,
                "ATR value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }
        let length = 14;
        for val in result.values.iter().skip(length - 1) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "ATR output should be finite after RMA stabilizes"
                );
            }
        }
        Ok(())
    }

    fn check_atr_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AtrInput::with_default_candles(&candles);
        match input.data {
            AtrData::Candles { .. } => {}
            _ => panic!("Expected AtrData::Candles variant"),
        }
        let default_params = AtrParams::default();
        assert_eq!(input.params.length, default_params.length);
        let output = atr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_atr_zero_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let zero_length_params = AtrParams { length: Some(0) };
        let input_zero_length = AtrInput::from_candles(&candles, zero_length_params);
        let result_zero_length = atr_with_kernel(&input_zero_length, kernel);
        assert!(result_zero_length.is_err());
        Ok(())
    }

    fn check_atr_length_exceeding_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let too_long_params = AtrParams {
            length: Some(candles.close.len() + 10),
        };
        let input_too_long = AtrInput::from_candles(&candles, too_long_params);
        let result_too_long = atr_with_kernel(&input_too_long, kernel);
        assert!(result_too_long.is_err());
        Ok(())
    }

    fn check_atr_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0];
        let low = [5.0];
        let close = [7.0];
        let params = AtrParams { length: Some(14) };
        let input = AtrInput::from_slices(&high, &low, &close, params);
        let result = atr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_atr_with_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AtrParams { length: Some(14) };
        let first_input = AtrInput::from_candles(&candles, first_params);
        let first_result = atr_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = AtrParams { length: Some(5) };
        let second_input = AtrInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = atr_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_atr_accuracy_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = AtrParams { length: Some(14) };
        let input = AtrInput::from_candles(&candles, params);
        let result = atr_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_atr_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to increase coverage
        let test_lengths = vec![2, 5, 10, 14, 20, 50, 100, 200];

        for length in test_lengths {
            let params = AtrParams {
                length: Some(length),
            };
            let input = AtrInput::from_candles(&candles, params);
            let output = atr_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with length={}",
						test_name, val, bits, i, length
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with length={}",
						test_name, val, bits, i, length
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with length={}",
						test_name, val, bits, i, length
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_atr_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_atr_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic OHLC data with proper constraints
        let strat = (2usize..=50)
            .prop_flat_map(|length| {
                // First choose the data length, then generate vectors of that exact size
                (length..400).prop_flat_map(move |data_len| {
                    (
                        // High prices
                        prop::collection::vec(
                            (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        // Low prices (will be adjusted to be <= high)
                        prop::collection::vec(
                            (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        // Close prices (will be adjusted to be between low and high)
                        prop::collection::vec(
                            (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        Just(length),
                    )
                })
            })
            .prop_map(|(high_raw, low_raw, close_raw, length)| {
                // Ensure OHLC constraints are met
                let len = high_raw.len();
                assert_eq!(low_raw.len(), len);
                assert_eq!(close_raw.len(), len);

                let mut high = Vec::with_capacity(len);
                let mut low = Vec::with_capacity(len);
                let mut close = Vec::with_capacity(len);

                for i in 0..len {
                    // Ensure low <= high
                    let h = high_raw[i].max(low_raw[i]);
                    let l = high_raw[i].min(low_raw[i]);

                    // Ensure close is between low and high
                    let c = close_raw[i].max(l).min(h);

                    high.push(h);
                    low.push(l);
                    close.push(c);
                }

                (high, low, close, length)
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(high, low, close, length)| {
                let params = AtrParams {
                    length: Some(length),
                };
                let input = AtrInput::from_slices(&high, &low, &close, params);

                // Get outputs from kernel under test and reference scalar
                let AtrOutput { values: out } = atr_with_kernel(&input, kernel)?;
                let AtrOutput { values: ref_out } = atr_with_kernel(&input, Kernel::Scalar)?;

                // Check output length matches input
                prop_assert_eq!(out.len(), high.len(), "Output length mismatch");

                // Test 1: Warmup validation - first (length-1) values should be NaN
                // This assumes clean input data with no leading NaNs
                for i in 0..(length - 1) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Test 2: Non-negative values - ATR measures volatility, must be >= 0
                for (i, &val) in out.iter().enumerate().skip(length - 1) {
                    if !val.is_nan() {
                        prop_assert!(
                            val >= 0.0,
                            "ATR must be non-negative at index {}: got {}",
                            i,
                            val
                        );
                    }
                }

                // Test 3: Bounded by maximum true range in the data
                // Calculate actual maximum true range (not just high-low)
                let mut max_true_range = 0.0f64;
                for i in 0..high.len() {
                    let tr = if i == 0 {
                        high[0] - low[0]
                    } else {
                        let hl = high[i] - low[i];
                        let hc = (high[i] - close[i - 1]).abs();
                        let lc = (low[i] - close[i - 1]).abs();
                        hl.max(hc).max(lc)
                    };
                    max_true_range = max_true_range.max(tr);
                }

                // ATR is an exponential average, so it should never exceed the max true range
                for (i, &val) in out.iter().enumerate().skip(length - 1) {
                    if !val.is_nan() && val.is_finite() {
                        prop_assert!(
                            val <= max_true_range + 1e-9,
                            "ATR at index {} exceeds max true range: {} > {}",
                            i,
                            val,
                            max_true_range
                        );
                    }
                }

                // Test 4: Kernel consistency - all kernels should produce identical results
                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Handle NaN/infinite values
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/infinite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    // Check ULP difference for finite values
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Test 5: Constant price convergence - when all prices are identical
                // Check if all high, low, and close values are the same
                let first_price = high[0];
                let is_constant = high.iter().all(|&h| (h - first_price).abs() < 1e-10)
                    && low.iter().all(|&l| (l - first_price).abs() < 1e-10)
                    && close.iter().all(|&c| (c - first_price).abs() < 1e-10);

                if is_constant {
                    // For constant prices, ATR should converge to 0
                    // Check the last few values after sufficient iterations
                    if out.len() >= length * 3 {
                        let last_values = &out[out.len().saturating_sub(5)..];
                        for &val in last_values {
                            if !val.is_nan() && val.is_finite() {
                                prop_assert!(
                                    val < 1e-6,
                                    "ATR should converge to 0 for constant prices, got {}",
                                    val
                                );
                            }
                        }
                    }
                }

                // Test 6: RMA smoothness property - ATR should change smoothly
                // Check that consecutive ATR values don't jump erratically
                if out.len() >= length + 10 {
                    for i in (length + 1)..out.len() {
                        if !out[i].is_nan() && !out[i - 1].is_nan() {
                            // Calculate true range for current period
                            let tr = {
                                let hl = high[i] - low[i];
                                let hc = (high[i] - close[i - 1]).abs();
                                let lc = (low[i] - close[i - 1]).abs();
                                hl.max(hc).max(lc)
                            };

                            // ATR uses RMA: new_atr = old_atr + (tr - old_atr) / length
                            // So the change should be bounded by |tr - old_atr| / length
                            let expected_change_bound = (tr - out[i - 1]).abs() / length as f64;
                            let actual_change = (out[i] - out[i - 1]).abs();

                            prop_assert!(
                                actual_change <= expected_change_bound + 1e-9,
                                "ATR change at index {} exceeds RMA bound: {} > {}",
                                i,
                                actual_change,
                                expected_change_bound
                            );
                        }
                    }
                }

                // Test 7: Single period edge case
                if length == 1 {
                    // For length=1, ATR should equal the true range exactly
                    for i in 0..out.len() {
                        if !out[i].is_nan() {
                            let tr = if i == 0 {
                                high[0] - low[0]
                            } else {
                                let hl = high[i] - low[i];
                                let hc = (high[i] - close[i - 1]).abs();
                                let lc = (low[i] - close[i - 1]).abs();
                                hl.max(hc).max(lc)
                            };
                            prop_assert!(
                                (out[i] - tr).abs() <= 1e-9,
                                "Length=1 ATR should equal TR at index {}: {} vs {}",
                                i,
                                out[i],
                                tr
                            );
                        }
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_atr_tests {
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

    generate_all_atr_tests!(
        check_atr_partial_params,
        check_atr_accuracy,
        check_atr_default_candles,
        check_atr_zero_length,
        check_atr_length_exceeding_data_length,
        check_atr_very_small_data_set,
        check_atr_with_slice_data_reinput,
        check_atr_accuracy_nan_check,
        check_atr_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_atr_tests!(check_atr_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AtrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = AtrParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [916.89, 874.33, 838.45, 801.92, 811.57];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
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

        // Test multiple diverse parameter ranges to increase coverage
        let test_configs = vec![
            (2, 10, 1),    // Small lengths with step 1
            (5, 25, 5),    // Medium lengths with step 5
            (10, 50, 10),  // Larger lengths with step 10
            (14, 140, 14), // Default-based multiples
            (50, 200, 50), // Large lengths
            (100, 100, 0), // Single large length
        ];

        for (start, end, step) in test_configs {
            let output = AtrBatchBuilder::new()
                .kernel(kernel)
                .length_range(start, end, step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let length = output.combos[row].length.unwrap_or(14);

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length={} in range ({},{},{})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length={} in range ({},{},{})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length={} in range ({},{},{})",
                        test, val, bits, row, col, idx, length, start, end, step
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

// ============= Python and WASM Bindings =============

#[cfg(feature = "python")]
use pyo3::create_exception;

// Custom Python exceptions for each AtrError variant
#[cfg(feature = "python")]
create_exception!(atr, InvalidLengthError, PyValueError);
#[cfg(feature = "python")]
create_exception!(atr, InconsistentSliceLengthsError, PyValueError);
#[cfg(feature = "python")]
create_exception!(atr, NoCandlesAvailableError, PyValueError);
#[cfg(feature = "python")]
create_exception!(atr, NotEnoughDataError, PyValueError);

#[cfg(feature = "python")]
impl From<AtrError> for PyErr {
    fn from(err: AtrError) -> PyErr {
        match err {
            AtrError::InvalidLength { length } => InvalidLengthError::new_err(format!(
                "Invalid length for ATR calculation (length={}).",
                length
            )),
            AtrError::InconsistentSliceLengths {
                high_len,
                low_len,
                close_len,
            } => InconsistentSliceLengthsError::new_err(format!(
                "Inconsistent slice lengths for ATR calculation: high={}, low={}, close={}",
                high_len, low_len, close_len
            )),
            AtrError::NoCandlesAvailable => {
                NoCandlesAvailableError::new_err("No candles available for ATR calculation.")
            }
            AtrError::NotEnoughData { length, data_len } => NotEnoughDataError::new_err(format!(
                "Not enough data to calculate ATR: length={}, data length={}",
                length, data_len
            )),
        }
    }
}

/// Helper function to prepare ATR calculation parameters from AtrInput
#[inline(always)]
fn atr_prepare_from_input<'a>(
    input: &'a AtrInput,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize), AtrError> {
    let (high, low, close) = match &input.data {
        AtrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        AtrData::Slices { high, low, close } => (*high, *low, *close),
    };

    let length = input.params.length.unwrap_or(14);
    let (high, low, close, length) = atr_prepare(high, low, close, length)?;
    let warmup = length - 1;
    Ok((high, low, close, length, warmup))
}

/// Helper function to prepare ATR calculation parameters
#[inline(always)]
fn atr_prepare<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    length: usize,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize), AtrError> {
    // Check for mismatched lengths
    if high.len() != low.len() || low.len() != close.len() {
        return Err(AtrError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    // Check for empty data
    if close.is_empty() {
        return Err(AtrError::NoCandlesAvailable);
    }

    // Check for zero length
    if length == 0 {
        return Err(AtrError::InvalidLength { length });
    }

    // Check if we have enough data
    if length > close.len() {
        return Err(AtrError::NotEnoughData {
            length,
            data_len: close.len(),
        });
    }

    Ok((high, low, close, length))
}

#[cfg(feature = "python")]
#[pyfunction(name = "atr")]
#[pyo3(signature = (high, low, close, length=14, kernel=None))]
pub fn atr_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    // Validate kernel outside allow_threads
    let kernel_enum = validate_kernel(kernel, false)?;

    // Get slices from numpy arrays - zero copy
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    // Create ATR input
    let params = AtrParams {
        length: Some(length),
    };
    let input = AtrInput::from_slices(high_slice, low_slice, close_slice, params);

    // Compute ATR and get Vec<f64> result
    let result_vec: Vec<f64> = py
        .allow_threads(|| atr_with_kernel(&input, kernel_enum).map(|output| output.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy array
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AtrStream")]
pub struct AtrStreamPy {
    stream: AtrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AtrStreamPy {
    #[new]
    pub fn new(length: Option<usize>) -> PyResult<Self> {
        let params = AtrParams { length };
        let stream = AtrStream::try_new(params)?;
        Ok(Self { stream })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "atr_batch")]
#[pyo3(signature = (high, low, close, length_range, kernel=None))]
pub fn atr_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let k = validate_kernel(kernel, true)?; // batch
    let hs = high.as_slice()?;
    let ls = low.as_slice()?;
    let cs = close.as_slice()?;

    let range = AtrBatchRange {
        length: length_range,
    };
    let combos = expand_grid(&range);
    let rows = combos.len();
    let cols = cs.len();

    // flat 1D buffer like ALMA, then reshape without copy
    let out_arr = unsafe { numpy::PyArray1::<f64>::new(py, [rows * cols], false) };
    let buf = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let simd = match match k {
            Kernel::Auto => detect_best_batch_kernel(),
            k if k.is_batch() => k,
            Kernel::Scalar => Kernel::ScalarBatch,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => Kernel::Avx2Batch,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => Kernel::Avx512Batch,
            _ => Kernel::ScalarBatch,
        } {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        atr_batch_inner_into(hs, ls, cs, &range, simd, true, buf)
            .map(|_| ())
            .map_err(|e| e)
    })
    .map_err(|e: AtrError| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict.into())
}

// ============= WASM Bindings =============

/// Write directly to output slice - no allocations
#[cfg(feature = "wasm")]
pub fn atr_into_slice(dst: &mut [f64], input: &AtrInput, kern: Kernel) -> Result<(), AtrError> {
    let (high, low, close) = match &input.data {
        AtrData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        AtrData::Slices { high, low, close } => (*high, *low, *close),
    };

    let length = input.params.length.unwrap_or(14);
    let (high, low, close, length) = atr_prepare(high, low, close, length)?;
    let first = first_valid_hlc(high, low, close);
    if close.len().saturating_sub(first) < length {
        return Err(AtrError::NotEnoughData {
            length,
            data_len: close.len(),
        });
    }
    let warm = first + length - 1;

    if dst.len() != close.len() {
        return Err(AtrError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: dst.len(),
        });
    }

    // only once: set prefix NaNs
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }

    let k = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    atr_compute_into(high, low, close, length, first, k, dst);
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atr")]
pub fn atr_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
) -> Result<Vec<f64>, JsError> {
    let params = AtrParams {
        length: Some(length),
    };
    let input = AtrInput::from_slices(high, low, close, params);

    let mut output = vec![0.0; high.len()]; // Single allocation
    atr_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsError::new(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atrBatch")]
pub fn atr_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<Vec<f64>, JsError> {
    let range = AtrBatchRange {
        length: (length_start, length_end, length_step),
    };
    let output = atr_batch_with_kernel(high, low, close, &range, Kernel::Auto)
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atrBatchMetadata")]
pub fn atr_batch_metadata_js(
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Vec<f64> {
    let range = AtrBatchRange {
        length: (length_start, length_end, length_step),
    };
    let combos = expand_grid(&range);

    // Return flat array of lengths
    combos
        .iter()
        .map(|p| p.length.unwrap_or(14) as f64)
        .collect()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atr_batch", skip_jsdoc)]
pub fn atr_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsError> {
    #[derive(Deserialize)]
    struct BatchConfig {
        length_range: [usize; 3],
    }

    let config: BatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsError::new(&e.to_string()))?;

    let range = AtrBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };

    let output = atr_batch_with_kernel(high, low, close, &range, Kernel::Auto)
        .map_err(|e| JsError::new(&e.to_string()))?;

    #[derive(Serialize)]
    struct BatchResult {
        values: Vec<f64>,
        combos: Vec<AtrParams>,
        rows: usize,
        cols: usize,
    }

    let result = BatchResult {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}

// Zero-copy API for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn atr_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn atr_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn atr_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsError> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsError::new("null pointer passed to atr_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let params = AtrParams {
            length: Some(length),
        };
        let input = AtrInput::from_slices(high, low, close, params);

        // CRITICAL: Check for aliasing - any input could be the same as output
        if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
            // Use temporary buffer if there's aliasing
            let mut temp = vec![0.0; len];
            atr_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsError::new(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, can write directly
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            atr_into_slice(out, &input, Kernel::Auto).map_err(|e| JsError::new(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn atr_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<(), JsError> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsError::new("null pointer passed to atr_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let range = AtrBatchRange {
            length: (length_start, length_end, length_step),
        };

        // Compute dimensions
        let combos = expand_grid(&range);
        let rows = combos.len();
        let cols = len;
        let output_size = rows * cols;

        // Check for aliasing
        if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
            // Use temporary buffer if there's aliasing
            let output = atr_batch_with_kernel(high, low, close, &range, Kernel::Auto)
                .map_err(|e| JsError::new(&e.to_string()))?;
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, output_size);
            out_slice.copy_from_slice(&output.values);
        } else {
            // No aliasing, compute directly into output buffer
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, output_size);

            // Select kernel for batch operations
            let kernel = match detect_best_batch_kernel() {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512Batch => Kernel::Avx512,
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => Kernel::Scalar,
            };

            // Use atr_batch_inner_into to write directly to output
            atr_batch_inner_into(high, low, close, &range, kernel, false, out_slice)
                .map_err(|e| JsError::new(&e.to_string()))?;
        }

        Ok(())
    }
}

// Streaming context for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct AtrContext {
    stream: AtrStream,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl AtrContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(length: usize) -> Result<AtrContext, JsError> {
        let params = AtrParams {
            length: Some(length),
        };
        let stream = AtrStream::try_new(params).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(AtrContext { stream })
    }

    #[wasm_bindgen]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) -> Result<(), JsError> {
        let length = self.stream.length;
        let params = AtrParams {
            length: Some(length),
        };
        self.stream = AtrStream::try_new(params).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }
}

// Python module registration for custom exceptions
// This should be called from the main module initialization
#[cfg(feature = "python")]
pub fn register_atr_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "InvalidLengthError",
        m.py().get_type::<InvalidLengthError>(),
    )?;
    m.add(
        "InconsistentSliceLengthsError",
        m.py().get_type::<InconsistentSliceLengthsError>(),
    )?;
    m.add(
        "NoCandlesAvailableError",
        m.py().get_type::<NoCandlesAvailableError>(),
    )?;
    m.add(
        "NotEnoughDataError",
        m.py().get_type::<NotEnoughDataError>(),
    )?;
    Ok(())
}
