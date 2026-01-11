//! # Normalized Average True Range (NATR)
//!
//! Normalizes ATR by closing price as a percentage for cross-asset volatility comparison.
//!
//! ## Parameters
//! - **high**: High price data
//! - **low**: Low price data
//! - **close**: Close price data
//! - **period**: ATR calculation period (default: 14)
//!
//! ## Returns
//! - `Vec<f64>` - NATR values (percentage) matching input length
//!
//! ## Developer Status
//! **SIMD**: Implemented AVX2/AVX512 for TR batching; recurrence remains scalar
//! **Streaming**: O(1) – minimal state Wilder update (prev_close, sum_tr warmup, atr)
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`
//! Note: Streaming path refactored to lean Wilder kernel; matches batch outputs.
//! **Decision log**: SIMD enabled (AVX2/AVX512 TR batching with scalar recurrence); CUDA wrapper present; Python CUDA interop exposes CAI v3 + DLPack v1.x; scalar path remains the reference implementation.

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

#[derive(Debug, Clone)]
pub enum NatrData<'a> {
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
pub struct NatrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct NatrParams {
    pub period: Option<usize>,
}

impl Default for NatrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct NatrInput<'a> {
    pub data: NatrData<'a>,
    pub params: NatrParams,
}

impl<'a> NatrInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: NatrParams) -> Self {
        Self {
            data: NatrData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: NatrParams,
    ) -> Self {
        Self {
            data: NatrData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: NatrData::Candles { candles },
            params: NatrParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NatrBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for NatrBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NatrBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<NatrOutput, NatrError> {
        let p = NatrParams {
            period: self.period,
        };
        let i = NatrInput::from_candles(c, p);
        natr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<NatrOutput, NatrError> {
        let p = NatrParams {
            period: self.period,
        };
        let i = NatrInput::from_slices(high, low, close, p);
        natr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NatrStream, NatrError> {
        let p = NatrParams {
            period: self.period,
        };
        NatrStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum NatrError {
    #[error("natr: Empty data provided for NATR.")]
    EmptyInputData,
    #[error("natr: All values are NaN.")]
    AllValuesNaN,
    #[error("natr: Empty data provided for NATR.")]
    EmptyData,
    #[error("natr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("natr: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("natr: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("natr: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },
    #[error("natr: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("natr: Mismatched lengths: expected = {expected}, actual = {actual}")]
    MismatchedLength { expected: usize, actual: usize },
}

#[inline]
pub fn natr(input: &NatrInput) -> Result<NatrOutput, NatrError> {
    natr_with_kernel(input, Kernel::Auto)
}

pub fn natr_with_kernel(input: &NatrInput, kernel: Kernel) -> Result<NatrOutput, NatrError> {
    let (high, low, close) = match &input.data {
        NatrData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            (high, low, close)
        }
        NatrData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(NatrError::EmptyInputData);
    }

    let len_h = high.len();
    let len_l = low.len();
    let len_c = close.len();
    if len_h != len_l || len_h != len_c {
        return Err(NatrError::MismatchedLength {
            expected: len_h,
            actual: if len_l != len_h { len_l } else { len_c },
        });
    }
    let len = len_h;

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(NatrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_idx = {
        let first_valid_idx_h = high.iter().position(|&x| !x.is_nan());
        let first_valid_idx_l = low.iter().position(|&x| !x.is_nan());
        let first_valid_idx_c = close.iter().position(|&x| !x.is_nan());

        match (first_valid_idx_h, first_valid_idx_l, first_valid_idx_c) {
            (Some(h), Some(l), Some(c)) => Some(h.max(l).max(c)),
            _ => None,
        }
    };

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(NatrError::AllValuesNaN),
    };

    if (len - first_valid_idx) < period {
        return Err(NatrError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let mut out = alloc_with_nan_prefix(len, first_valid_idx + period - 1);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                natr_scalar(high, low, close, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                natr_avx2(high, low, close, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                natr_avx512(high, low, close, period, first_valid_idx, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(NatrOutput { values: out })
}

#[inline]
pub fn natr_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = out.len();
    if first >= len {
        return;
    }

    let inv_p: f64 = 1.0 / (period as f64);
    let k100: f64 = 100.0;

    // Warmup: arithmetic mean of TR over the first `period` values starting at `first`.
    let warm_end = first + period - 1;
    let mut sum_tr = 0.0;
    // i == first
    sum_tr += high[first] - low[first];
    for i in (first + 1)..=warm_end {
        let hi = high[i];
        let lo = low[i];
        let pc = close[i - 1];
        let tr = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
        sum_tr += tr;
    }

    let mut atr = sum_tr * inv_p;
    let c_we = close[warm_end];
    out[warm_end] = if c_we.is_finite() && c_we != 0.0 {
        (atr / c_we) * k100
    } else {
        f64::NAN
    };

    // Main loop: unrolled by 4 for better ILP; Wilder smoothing remains sequential.
    let mut idx = warm_end + 1;
    while idx + 3 < len {
        let pc0 = close[idx - 1];
        let pc1 = close[idx + 0];
        let pc2 = close[idx + 1];
        let pc3 = close[idx + 2];

        let h0 = high[idx + 0];
        let h1 = high[idx + 1];
        let h2 = high[idx + 2];
        let h3 = high[idx + 3];
        let l0 = low[idx + 0];
        let l1 = low[idx + 1];
        let l2 = low[idx + 2];
        let l3 = low[idx + 3];

        let tr0 = (h0 - l0).max((h0 - pc0).abs()).max((l0 - pc0).abs());
        let tr1 = (h1 - l1).max((h1 - pc1).abs()).max((l1 - pc1).abs());
        let tr2 = (h2 - l2).max((h2 - pc2).abs()).max((l2 - pc2).abs());
        let tr3 = (h3 - l3).max((h3 - pc3).abs()).max((l3 - pc3).abs());

        atr = (tr0 - atr).mul_add(inv_p, atr);
        let c0 = close[idx + 0];
        out[idx + 0] = if c0.is_finite() && c0 != 0.0 {
            (atr / c0) * k100
        } else {
            f64::NAN
        };

        atr = (tr1 - atr).mul_add(inv_p, atr);
        let c1v = close[idx + 1];
        out[idx + 1] = if c1v.is_finite() && c1v != 0.0 {
            (atr / c1v) * k100
        } else {
            f64::NAN
        };

        atr = (tr2 - atr).mul_add(inv_p, atr);
        let c2v = close[idx + 2];
        out[idx + 2] = if c2v.is_finite() && c2v != 0.0 {
            (atr / c2v) * k100
        } else {
            f64::NAN
        };

        atr = (tr3 - atr).mul_add(inv_p, atr);
        let c3v = close[idx + 3];
        out[idx + 3] = if c3v.is_finite() && c3v != 0.0 {
            (atr / c3v) * k100
        } else {
            f64::NAN
        };

        idx += 4;
    }

    while idx < len {
        let hi = high[idx];
        let lo = low[idx];
        let pc = close[idx - 1];
        let tr = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
        atr = (tr - atr).mul_add(inv_p, atr);

        let cv = close[idx];
        out[idx] = if cv.is_finite() && cv != 0.0 {
            (atr / cv) * k100
        } else {
            f64::NAN
        };
        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn natr_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe {
        // Unified body works for all periods
        natr_avx512_body(high, low, close, period, first, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn natr_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    debug_assert!(high.len() == low.len() && high.len() == close.len() && high.len() == out.len());
    let len = out.len();
    if first >= len {
        return;
    }

    let inv_p: f64 = 1.0 / (period as f64);
    let k100: f64 = 100.0;

    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let o = out.as_mut_ptr();

    // Warm-up sum
    let mut i = first;
    let mut sum_tr = *h.add(i) - *l.add(i);
    i += 1;
    let warm_end = first + period - 1;

    let sign = _mm256_set1_pd(-0.0_f64);
    while i + 3 <= warm_end {
        let vh = _mm256_loadu_pd(h.add(i));
        let vl = _mm256_loadu_pd(l.add(i));
        let vpc = _mm256_loadu_pd(c.add(i - 1));

        let vhl = _mm256_sub_pd(vh, vl);
        let vhc = _mm256_sub_pd(vh, vpc);
        let vlc = _mm256_sub_pd(vl, vpc);
        let abs_hc = _mm256_andnot_pd(sign, vhc);
        let abs_lc = _mm256_andnot_pd(sign, vlc);
        let mx1 = _mm256_max_pd(vhl, abs_hc);
        let vtr = _mm256_max_pd(mx1, abs_lc);

        let mut tmp = [0.0f64; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), vtr);
        sum_tr += tmp[0] + tmp[1] + tmp[2] + tmp[3];

        i += 4;
    }
    while i <= warm_end {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let pc = *c.add(i - 1);
        let tr = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
        sum_tr += tr;
        i += 1;
    }

    let mut atr = sum_tr * inv_p;
    let c_we = *c.add(warm_end);
    *o.add(warm_end) = if c_we.is_finite() && c_we != 0.0 {
        (atr / c_we) * k100
    } else {
        f64::NAN
    };

    // Main loop: batch TR in AVX2, sequential recurrence
    let mut idx = warm_end + 1;
    while idx + 3 < len {
        let vh = _mm256_loadu_pd(h.add(idx));
        let vl = _mm256_loadu_pd(l.add(idx));
        let vpc = _mm256_loadu_pd(c.add(idx - 1));

        let vhl = _mm256_sub_pd(vh, vl);
        let vhc = _mm256_sub_pd(vh, vpc);
        let vlc = _mm256_sub_pd(vl, vpc);
        let abs_hc = _mm256_andnot_pd(sign, vhc);
        let abs_lc = _mm256_andnot_pd(sign, vlc);
        let mx1 = _mm256_max_pd(vhl, abs_hc);
        let vtr = _mm256_max_pd(mx1, abs_lc);

        let mut tr = [0.0f64; 4];
        _mm256_storeu_pd(tr.as_mut_ptr(), vtr);

        atr = (tr[0] - atr).mul_add(inv_p, atr);
        let c0 = *c.add(idx + 0);
        *o.add(idx + 0) = if c0.is_finite() && c0 != 0.0 {
            (atr / c0) * k100
        } else {
            f64::NAN
        };

        atr = (tr[1] - atr).mul_add(inv_p, atr);
        let c1 = *c.add(idx + 1);
        *o.add(idx + 1) = if c1.is_finite() && c1 != 0.0 {
            (atr / c1) * k100
        } else {
            f64::NAN
        };

        atr = (tr[2] - atr).mul_add(inv_p, atr);
        let c2 = *c.add(idx + 2);
        *o.add(idx + 2) = if c2.is_finite() && c2 != 0.0 {
            (atr / c2) * k100
        } else {
            f64::NAN
        };

        atr = (tr[3] - atr).mul_add(inv_p, atr);
        let c3 = *c.add(idx + 3);
        *o.add(idx + 3) = if c3.is_finite() && c3 != 0.0 {
            (atr / c3) * k100
        } else {
            f64::NAN
        };

        idx += 4;
    }
    while idx < len {
        let hi = *h.add(idx);
        let lo = *l.add(idx);
        let pc = *c.add(idx - 1);
        let tr = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
        atr = (tr - atr).mul_add(inv_p, atr);

        let cv = *c.add(idx);
        *o.add(idx) = if cv.is_finite() && cv != 0.0 {
            (atr / cv) * k100
        } else {
            f64::NAN
        };
        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn natr_avx512_body(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    debug_assert!(high.len() == low.len() && high.len() == close.len() && high.len() == out.len());
    let len = out.len();
    if first >= len {
        return;
    }

    let inv_p: f64 = 1.0 / (period as f64);
    let k100: f64 = 100.0;

    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let o = out.as_mut_ptr();

    // Warm-up
    let mut i = first;
    let mut sum_tr = *h.add(i) - *l.add(i);
    i += 1;
    let warm_end = first + period - 1;

    let sign = _mm512_set1_pd(-0.0_f64);
    while i + 7 <= warm_end {
        let vh = _mm512_loadu_pd(h.add(i));
        let vl = _mm512_loadu_pd(l.add(i));
        let vpc = _mm512_loadu_pd(c.add(i - 1));

        let vhl = _mm512_sub_pd(vh, vl);
        let vhc = _mm512_sub_pd(vh, vpc);
        let vlc = _mm512_sub_pd(vl, vpc);
        let abs_hc = _mm512_andnot_pd(sign, vhc);
        let abs_lc = _mm512_andnot_pd(sign, vlc);
        let mx1 = _mm512_max_pd(vhl, abs_hc);
        let vtr = _mm512_max_pd(mx1, abs_lc);

        let mut tmp = [0.0f64; 8];
        _mm512_storeu_pd(tmp.as_mut_ptr(), vtr);
        sum_tr += tmp.iter().sum::<f64>();

        i += 8;
    }
    while i <= warm_end {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let pc = *c.add(i - 1);
        let tr = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
        sum_tr += tr;
        i += 1;
    }

    let mut atr = sum_tr * inv_p;
    let c_we = *c.add(warm_end);
    *o.add(warm_end) = if c_we.is_finite() && c_we != 0.0 {
        (atr / c_we) * k100
    } else {
        f64::NAN
    };

    // Main loop
    let mut idx = warm_end + 1;
    while idx + 7 < len {
        let vh = _mm512_loadu_pd(h.add(idx));
        let vl = _mm512_loadu_pd(l.add(idx));
        let vpc = _mm512_loadu_pd(c.add(idx - 1));

        let vhl = _mm512_sub_pd(vh, vl);
        let vhc = _mm512_sub_pd(vh, vpc);
        let vlc = _mm512_sub_pd(vl, vpc);
        let abs_hc = _mm512_andnot_pd(sign, vhc);
        let abs_lc = _mm512_andnot_pd(sign, vlc);
        let mx1 = _mm512_max_pd(vhl, abs_hc);
        let vtr = _mm512_max_pd(mx1, abs_lc);

        let mut tr = [0.0f64; 8];
        _mm512_storeu_pd(tr.as_mut_ptr(), vtr);

        atr = (tr[0] - atr).mul_add(inv_p, atr);
        let c0 = *c.add(idx + 0);
        *o.add(idx + 0) = if c0.is_finite() && c0 != 0.0 {
            (atr / c0) * k100
        } else {
            f64::NAN
        };
        atr = (tr[1] - atr).mul_add(inv_p, atr);
        let c1 = *c.add(idx + 1);
        *o.add(idx + 1) = if c1.is_finite() && c1 != 0.0 {
            (atr / c1) * k100
        } else {
            f64::NAN
        };
        atr = (tr[2] - atr).mul_add(inv_p, atr);
        let c2 = *c.add(idx + 2);
        *o.add(idx + 2) = if c2.is_finite() && c2 != 0.0 {
            (atr / c2) * k100
        } else {
            f64::NAN
        };
        atr = (tr[3] - atr).mul_add(inv_p, atr);
        let c3 = *c.add(idx + 3);
        *o.add(idx + 3) = if c3.is_finite() && c3 != 0.0 {
            (atr / c3) * k100
        } else {
            f64::NAN
        };
        atr = (tr[4] - atr).mul_add(inv_p, atr);
        let c4 = *c.add(idx + 4);
        *o.add(idx + 4) = if c4.is_finite() && c4 != 0.0 {
            (atr / c4) * k100
        } else {
            f64::NAN
        };
        atr = (tr[5] - atr).mul_add(inv_p, atr);
        let c5 = *c.add(idx + 5);
        *o.add(idx + 5) = if c5.is_finite() && c5 != 0.0 {
            (atr / c5) * k100
        } else {
            f64::NAN
        };
        atr = (tr[6] - atr).mul_add(inv_p, atr);
        let c6 = *c.add(idx + 6);
        *o.add(idx + 6) = if c6.is_finite() && c6 != 0.0 {
            (atr / c6) * k100
        } else {
            f64::NAN
        };
        atr = (tr[7] - atr).mul_add(inv_p, atr);
        let c7 = *c.add(idx + 7);
        *o.add(idx + 7) = if c7.is_finite() && c7 != 0.0 {
            (atr / c7) * k100
        } else {
            f64::NAN
        };

        idx += 8;
    }
    while idx < len {
        let hi = *h.add(idx);
        let lo = *l.add(idx);
        let pc = *c.add(idx - 1);
        let tr = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
        atr = (tr - atr).mul_add(inv_p, atr);

        let cv = *c.add(idx);
        *o.add(idx) = if cv.is_finite() && cv != 0.0 {
            (atr / cv) * k100
        } else {
            f64::NAN
        };
        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn natr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn natr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out);
}

/// Decision: Streaming uses minimal-state Wilder kernel; matches batch within tests.
#[derive(Debug, Clone)]
pub struct NatrStream {
    period: usize,
    alpha: f64, // 1 / period
    k100: f64,  // 100.0
    // Running state
    count: usize,    // how many updates we've seen
    sum_tr: f64,     // used only until warmup completes
    atr: f64,        // valid once ready == true
    prev_close: f64, // last close (for TR)
    have_prev: bool, // do we have a previous close yet?
    ready: bool,     // becomes true at count == period
}

impl NatrStream {
    #[inline(always)]
    pub fn try_new(params: NatrParams) -> Result<Self, NatrError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(NatrError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            alpha: 1.0 / (period as f64),
            k100: 100.0,
            count: 0,
            sum_tr: 0.0,
            atr: 0.0,
            prev_close: 0.0,
            have_prev: false,
            ready: false,
        })
    }

    /// O(1) update. Returns:
    /// - None until warmup completes (i.e., after `period` updates),
    /// - Some(NaN) if `close` is non-finite or zero at emission step,
    /// - Some(natr) otherwise.
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        // True Range computation (Wilder):
        // TR = max(high, prev_close) - min(low, prev_close)
        // For the very first bar, TR := high - low (no prev_close).
        let tr = if self.have_prev {
            let pc = self.prev_close;
            high.max(pc) - low.min(pc)
        } else {
            // first bar only
            high - low
        };

        // Prepare next step's prev_close
        self.prev_close = close;
        self.have_prev = true;

        self.count += 1;

        if !self.ready {
            // Warmup: accumulate TRs; when we reach `period`,
            // initialize ATR as the arithmetic mean of the first `period` TRs.
            self.sum_tr += tr;
            if self.count == self.period {
                self.atr = self.sum_tr / (self.period as f64);
                self.ready = true;
            } else {
                return None;
            }
        } else {
            // Wilder smoothing (EMA with alpha = 1/period):
            // atr = atr + alpha * (tr - atr)
            self.atr = (tr - self.atr).mul_add(self.alpha, self.atr);
        }

        // NATR = (ATR / close) * 100
        if close.is_finite() && close != 0.0 {
            Some((self.atr / close) * self.k100)
        } else {
            Some(f64::NAN)
        }
    }
}

#[derive(Clone, Debug)]
pub struct NatrBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for NatrBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NatrBatchBuilder {
    range: NatrBatchRange,
    kernel: Kernel,
}

impl NatrBatchBuilder {
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<NatrBatchOutput, NatrError> {
        natr_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<NatrBatchOutput, NatrError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<NatrBatchOutput, NatrError> {
        NatrBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub fn natr_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    k: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(NatrError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    natr_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct NatrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NatrParams>,
    pub rows: usize,
    pub cols: usize,
}
impl NatrBatchOutput {
    pub fn row_for_params(&self, p: &NatrParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &NatrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &NatrBatchRange) -> Result<Vec<NatrParams>, NatrError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, NatrError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut values = Vec::new();
        let step_u = step;

        if start <= end {
            let mut v = start;
            loop {
                if v > end {
                    break;
                }
                values.push(v);
                match v.checked_add(step_u) {
                    Some(next) => v = next,
                    None => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                if v < end {
                    break;
                }
                values.push(v);
                match v.checked_sub(step_u) {
                    Some(next) => v = next,
                    None => break,
                }
            }
        }

        if values.is_empty() {
            return Err(NatrError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }

        Ok(values)
    }

    let periods = axis_usize(r.period)?;

    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(NatrParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn natr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
    natr_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn natr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
    natr_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn natr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<NatrBatchOutput, NatrError> {
    let combos = expand_grid(sweep)?;

    let len_h = high.len();
    let len_l = low.len();
    let len_c = close.len();
    if len_h != len_l || len_h != len_c {
        return Err(NatrError::MismatchedLength {
            expected: len_h,
            actual: if len_l != len_h { len_l } else { len_c },
        });
    }
    let len = len_h;

    let first = high
        .iter()
        .position(|x| !x.is_nan())
        .unwrap_or(0)
        .max(low.iter().position(|x| !x.is_nan()).unwrap_or(0))
        .max(close.iter().position(|x| !x.is_nan()).unwrap_or(0));
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(NatrError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let use_tr_shared = combos.len() >= 24; // enable shared TR only for larger sweeps

    if use_tr_shared {
        // Precompute TR once and reuse across rows (row-optimized batch)
        let mut tr: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, len);
        tr.resize(len, 0.0);
        if first < len {
            tr[first] = high[first] - low[first];
            let mut i = first + 1;
            while i < len {
                let hi = high[i];
                let lo = low[i];
                let pc = close[i - 1];
                let trv = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
                tr[i] = trv;
                i += 1;
            }
        }
        // Prefix sums for O(1) warmup ATR per row
        let mut pref: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, len + 1);
        pref.resize(len + 1, 0.0);
        for i in first..len {
            pref[i + 1] = pref[i] + tr[i];
        }

        let do_row = |row: usize, out_row: &mut [f64]| {
            let period = combos[row].period.unwrap();
            let inv_p = 1.0 / (period as f64);
            let k100 = 100.0;
            let warm_end = first + period - 1;

            // Warmup ATR via prefix sum
            let sum_tr = pref[warm_end + 1] - pref[first];
            let mut atr = sum_tr * inv_p;
            let cw = close[warm_end];
            out_row[warm_end] = if cw.is_finite() && cw != 0.0 {
                (atr / cw) * k100
            } else {
                f64::NAN
            };

            // Main recurrence using TR
            let mut i = warm_end + 1;
            while i < len {
                atr = (tr[i] - atr).mul_add(inv_p, atr);
                let cv = close[i];
                out_row[i] = if cv.is_finite() && cv != 0.0 {
                    (atr / cv) * k100
                } else {
                    f64::NAN
                };
                i += 1;
            }
        };

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
    } else {
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            match kern {
                Kernel::Scalar => natr_row_scalar(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => natr_row_avx2(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => natr_row_avx512(high, low, close, first, period, out_row),
                _ => unreachable!(),
            }
        };

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
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(NatrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn natr_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<NatrParams>, NatrError> {
    let combos = expand_grid(sweep)?;

    let len_h = high.len();
    let len_l = low.len();
    let len_c = close.len();
    if len_h != len_l || len_h != len_c {
        return Err(NatrError::MismatchedLength {
            expected: len_h,
            actual: if len_l != len_h { len_l } else { len_c },
        });
    }
    let len = len_h;

    let first = high
        .iter()
        .position(|x| !x.is_nan())
        .unwrap_or(0)
        .max(low.iter().position(|x| !x.is_nan()).unwrap_or(0))
        .max(close.iter().position(|x| !x.is_nan()).unwrap_or(0));
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(NatrError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    // Initialize warmup positions with NaN for each row
    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warmup_end = first + period - 1;
        let row_start = row * cols;
        for i in 0..warmup_end.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    let use_tr_shared = combos.len() >= 24;
    if use_tr_shared {
        // Precompute TR once and reuse for all rows
        let mut tr: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, len);
        tr.resize(len, 0.0);
        if first < len {
            tr[first] = high[first] - low[first];
            let mut i = first + 1;
            while i < len {
                let hi = high[i];
                let lo = low[i];
                let pc = close[i - 1];
                let trv = (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
                tr[i] = trv;
                i += 1;
            }
        }
        // Prefix sums for O(1) warmup ATR per row
        let mut pref: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, len + 1);
        pref.resize(len + 1, 0.0);
        for i in first..len {
            pref[i + 1] = pref[i] + tr[i];
        }

        let do_row = |row: usize, out_row: &mut [f64]| {
            let period = combos[row].period.unwrap();
            let inv_p = 1.0 / (period as f64);
            let k100 = 100.0;
            let warm_end = first + period - 1;

            let sum_tr = pref[warm_end + 1] - pref[first];
            let mut atr = sum_tr * inv_p;
            let cw = close[warm_end];
            out_row[warm_end] = if cw.is_finite() && cw != 0.0 {
                (atr / cw) * k100
            } else {
                f64::NAN
            };

            let mut i = warm_end + 1;
            while i < len {
                atr = (tr[i] - atr).mul_add(inv_p, atr);
                let cv = close[i];
                out_row[i] = if cv.is_finite() && cv != 0.0 {
                    (atr / cv) * k100
                } else {
                    f64::NAN
                };
                i += 1;
            }
        };

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
    } else {
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            match kern {
                Kernel::Scalar => natr_row_scalar(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => natr_row_avx2(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => natr_row_avx512(high, low, close, first, period, out_row),
                _ => unreachable!(),
            }
        };

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
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn natr_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_scalar(high, low, close, period, first, out)
}

#[inline(always)]
unsafe fn natr_row_scalar_from_tr(
    tr: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = out.len();
    if first >= len {
        return;
    }
    let inv_p = 1.0 / (period as f64);
    let k100 = 100.0;
    let warm_end = first + period - 1;

    // Warmup ATR from TR
    let mut sum_tr = 0.0;
    for i in first..=warm_end {
        sum_tr += tr[i];
    }
    let mut atr = sum_tr * inv_p;
    let cw = close[warm_end];
    out[warm_end] = if cw.is_finite() && cw != 0.0 {
        (atr / cw) * k100
    } else {
        f64::NAN
    };

    // Main recurrence using TR stream
    let mut i = warm_end + 1;
    while i < len {
        atr = (tr[i] - atr).mul_add(inv_p, atr);
        let cv = close[i];
        out[i] = if cv.is_finite() && cv != 0.0 {
            (atr / cv) * k100
        } else {
            f64::NAN
        };
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn natr_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx2(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx512(high, low, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn natr_row_avx2_from_tr(
    tr: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    // Recurrence is scalar; use the TR-optimized scalar row path
    natr_row_scalar_from_tr(tr, close, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn natr_row_avx512_from_tr(
    tr: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_row_scalar_from_tr(tr, close, first, period, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[test]
    fn test_natr_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Prepare a non-trivial OHLC input from the repo dataset
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Build input with default params (period = 14)
        let input = NatrInput::with_default_candles(&candles);

        // Baseline via existing Vec-returning API
        let baseline = natr(&input)?.values;

        // Preallocate output and call the new into API
        let mut out = vec![0.0; candles.close.len()];
        #[allow(unused_variables)]
        {
            // Native path only; wasm provides a different symbol for natr_into
            #[cfg(not(feature = "wasm"))]
            {
                natr_into(&input, &mut out)?;
            }
        }

        assert_eq!(baseline.len(), out.len());

        // Helper: treat NaN == NaN; otherwise compare with tight epsilon
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..baseline.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "NATR into parity mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }

    fn check_natr_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = NatrParams { period: None };
        let input_default = NatrInput::from_candles(&candles, default_params);
        let output_default = natr_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period_7 = NatrParams { period: Some(7) };
        let input_period_7 = NatrInput::from_candles(&candles, params_period_7);
        let output_period_7 = natr_with_kernel(&input_period_7, kernel)?;
        assert_eq!(output_period_7.values.len(), candles.close.len());
        Ok(())
    }

    fn check_natr_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close").unwrap();
        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_candles(&candles, params.clone());
        let natr_result = natr_with_kernel(&input, kernel)?;
        assert_eq!(natr_result.values.len(), close_prices.len());
        let expected_last_five = [
            1.5465877404905772,
            1.4773840355794576,
            1.4201627494720954,
            1.3556212509014807,
            1.3836271128536142,
        ];
        let start_index = natr_result.values.len() - 5;
        let result_last_five = &natr_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "NATR mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        let period = params.period.unwrap();
        for i in 0..(period - 1) {
            assert!(natr_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_natr_with_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 10.0, 15.0];
        let close = [7.0, 14.0, 25.0];
        let params = NatrParams { period: Some(0) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 10.0, 15.0];
        let close = [7.0, 14.0, 25.0];
        let params = NatrParams { period: Some(10) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [40.0];
        let close = [41.0];
        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let params = NatrParams { period: Some(2) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 10.0];
        let low = [f64::NAN, 5.0];
        let close = [f64::NAN, 7.0];
        let params = NatrParams { period: Some(5) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = NatrParams { period: Some(14) };
        let first_input = NatrInput::from_candles(&candles, first_params);
        let first_result = natr_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = NatrParams { period: Some(14) };
        let second_input = NatrInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = natr_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());

        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 28, but found NaN at index {}",
                i
            );
        }
        Ok(())
    }

    fn check_natr_nan_check(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_candles(&candles, params);
        let natr_result = natr_with_kernel(&input, kernel)?;
        assert_eq!(natr_result.values.len(), candles.close.len());
        if natr_result.values.len() > 30 {
            for i in 30..natr_result.values.len() {
                assert!(
                    !natr_result.values[i].is_nan(),
                    "Expected no NaN after index 30, but found NaN at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_natr_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NatrInput::with_default_candles(&candles);
        match input.data {
            NatrData::Candles { .. } => {}
            _ => panic!("Expected NatrData::Candles variant"),
        }
        let output = natr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_natr_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let high = &candles.high;
        let low = &candles.low;
        let close = &candles.close;
        let input = NatrInput::from_slices(
            high,
            low,
            close,
            NatrParams {
                period: Some(period),
            },
        );
        let batch_output = natr_with_kernel(&input, kernel)?.values;

        let mut stream = NatrStream::try_new(NatrParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(close.len());
        for ((&h, &l), &c) in high.iter().zip(low.iter()).zip(close.iter()) {
            match stream.update(h, l, c) {
                Some(natr_val) => stream_values.push(natr_val),
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
                "[{}] NATR streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_natr_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            NatrParams::default(),            // period: 14
            NatrParams { period: Some(2) },   // minimum viable period
            NatrParams { period: Some(5) },   // small period
            NatrParams { period: Some(7) },   // small period
            NatrParams { period: Some(10) },  // small-medium period
            NatrParams { period: Some(20) },  // medium period
            NatrParams { period: Some(30) },  // medium-large period
            NatrParams { period: Some(50) },  // large period
            NatrParams { period: Some(100) }, // very large period
            NatrParams { period: Some(200) }, // extra large period
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = NatrInput::from_candles(&candles, params.clone());
            let output = natr_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_natr_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = NatrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = NatrParams::default();
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
            (2, 10, 2),    // Small periods
            (5, 25, 5),    // Medium periods
            (30, 60, 15),  // Large periods
            (2, 5, 1),     // Dense small range
            (10, 20, 2),   // Mid-range with small step
            (50, 100, 10), // Large range
            (14, 14, 0),   // Single period (default)
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = NatrBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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

    #[cfg(feature = "proptest")]
    fn check_natr_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Strategy: generate period and length, then create realistic OHLC data
        let strat = (2usize..=50, 50usize..=400, 0usize..=2)
            .prop_flat_map(|(period, len, scenario)| {
                // Generate base close prices with different scenarios
                let close_strategy = match scenario {
                    0 => {
                        // Normal range prices
                        prop::collection::vec(
                            (1.0f64..1000.0f64).prop_filter("finite", |x| x.is_finite()),
                            len,
                        )
                        .boxed()
                    }
                    1 => {
                        // Very small prices (edge case)
                        prop::collection::vec(
                            (0.01f64..1.0f64).prop_filter("finite", |x| x.is_finite()),
                            len,
                        )
                        .boxed()
                    }
                    _ => {
                        // Constant prices
                        (1.0f64..100.0f64)
                            .prop_map(move |val| vec![val; len])
                            .boxed()
                    }
                };

                (close_strategy, Just(period), Just(len), Just(scenario))
            })
            .prop_flat_map(|(close_prices, period, len, scenario)| {
                // Generate high/low based on close with realistic spreads
                let mut high_vec = Vec::with_capacity(len);
                let mut low_vec = Vec::with_capacity(len);

                // Use a deterministic approach for generating high/low
                for (i, &close) in close_prices.iter().enumerate() {
                    if scenario == 2 {
                        // Constant prices scenario
                        high_vec.push(close);
                        low_vec.push(close);
                    } else {
                        // Create volatility that varies over time (expanded range: 0.1% to 20%)
                        let volatility_factor = 0.001 + 0.20 * ((i * 7919) % 100) as f64 / 100.0;
                        let spread = close * volatility_factor;

                        // Ensure high >= close >= low
                        let high = close + spread * 0.5;
                        let low = close - spread * 0.5;

                        high_vec.push(high);
                        low_vec.push(low.max(0.001)); // Ensure positive but allow very small values
                    }
                }

                (
                    Just(high_vec),
                    Just(low_vec),
                    Just(close_prices),
                    Just(period),
                    Just(scenario),
                )
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(high, low, close, period, scenario)| {
                let params = NatrParams {
                    period: Some(period),
                };
                let input = NatrInput::from_slices(&high, &low, &close, params);

                // Test with specified kernel
                let result = natr_with_kernel(&input, kernel)?;

                // Test with scalar kernel as reference
                let ref_result = natr_with_kernel(&input, Kernel::Scalar)?;

                // Property 1: Output length matches input length
                prop_assert_eq!(result.values.len(), high.len());
                prop_assert_eq!(result.values.len(), low.len());
                prop_assert_eq!(result.values.len(), close.len());

                // Property 2: Warmup period handling - first (period - 1) values should be NaN
                for i in 0..(period - 1) {
                    prop_assert!(
                        result.values[i].is_nan(),
                        "Expected NaN at index {} during warmup, got {}",
                        i,
                        result.values[i]
                    );
                }

                // Property 3: NATR values should be non-negative after warmup (TR is always positive)
                for i in period..result.values.len() {
                    if result.values[i].is_finite() {
                        prop_assert!(
                            result.values[i] >= 0.0,
                            "NATR should be non-negative at index {}: got {}",
                            i,
                            result.values[i]
                        );

                        // NATR values should be reasonable (with expanded volatility, could be higher)
                        prop_assert!(
                            result.values[i] < 10000.0,
                            "NATR seems unreasonably high at index {}: got {}",
                            i,
                            result.values[i]
                        );
                    }
                }

                // Property 4: Kernel consistency - different SIMD implementations should produce same results
                for i in 0..result.values.len() {
                    let val = result.values[i];
                    let ref_val = ref_result.values[i];

                    if val.is_nan() && ref_val.is_nan() {
                        continue;
                    }

                    if val.is_finite() && ref_val.is_finite() {
                        // Allow small tolerance for floating point differences
                        let diff = (val - ref_val).abs();
                        let tolerance = (ref_val.abs() * 1e-10).max(1e-10);
                        prop_assert!(
                            diff <= tolerance,
                            "Kernel mismatch at index {}: {} vs {} (diff: {})",
                            i,
                            val,
                            ref_val,
                            diff
                        );
                    } else {
                        // Both should have same finite/infinite status
                        prop_assert_eq!(
                            val.is_finite(),
                            ref_val.is_finite(),
                            "Finite status mismatch at index {}: {} vs {}",
                            i,
                            val,
                            ref_val
                        );
                    }
                }

                // Property 5: Test special case - when all prices are constant (scenario 2)
                // NATR should be exactly 0 after initial warmup
                if scenario == 2 {
                    // Check that high == low == close for all values (constant prices)
                    let is_constant = high
                        .iter()
                        .zip(&low)
                        .zip(&close)
                        .all(|((h, l), c)| (*h - *l).abs() < 1e-10 && (*h - *c).abs() < 1e-10);

                    if is_constant && result.values.len() > period + 5 {
                        // After the initial period, NATR should be 0 for constant prices
                        // Start checking a few indices after warmup to allow stabilization
                        for i in (period + 5)..result.values.len() {
                            if result.values[i].is_finite() {
                                prop_assert!(
                                    result.values[i].abs() < 1e-10,
                                    "NATR should be 0 for constant prices at index {}, got {}",
                                    i,
                                    result.values[i]
                                );
                            }
                        }
                    }
                }

                // Property 6: Test edge case with very small prices (scenario 1)
                if scenario == 1 {
                    // Verify that NATR still works correctly with small prices
                    for i in period..result.values.len() {
                        if result.values[i].is_finite() && close[i] > 0.0 {
                            // NATR should still be reasonable even with small prices
                            prop_assert!(
                                result.values[i] >= 0.0 && result.values[i] < 100000.0,
                                "NATR out of bounds with small prices at index {}: got {}",
                                i,
                                result.values[i]
                            );
                        }
                    }
                }

                // Property 7: Division by zero handling
                // Check if any close prices are near zero
                if close.iter().any(|&c| c.abs() < 1e-10) {
                    // NATR should return 0 when close is 0, not NaN or infinity
                    for (i, &c) in close.iter().enumerate() {
                        if c.abs() < 1e-10 && i >= period - 1 {
                            prop_assert!(
                                result.values[i] == 0.0 || result.values[i].is_nan(),
                                "NATR should be 0 or NaN when close is 0, got {} at index {}",
                                result.values[i],
                                i
                            );
                        }
                    }
                }

                // Property 8: Check poison values aren't present
                #[cfg(debug_assertions)]
                {
                    for (i, &val) in result.values.iter().enumerate() {
                        if val.is_finite() {
                            let bits = val.to_bits();
                            prop_assert!(
                                bits != 0x11111111_11111111
                                    && bits != 0x22222222_22222222
                                    && bits != 0x33333333_33333333,
                                "Found poison value at index {}: {} (0x{:016X})",
                                i,
                                val,
                                bits
                            );
                        }
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_natr_tests {
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

    #[cfg(feature = "proptest")]
    generate_all_natr_tests!(check_natr_property);

    generate_all_natr_tests!(
        check_natr_partial_params,
        check_natr_accuracy,
        check_natr_with_zero_period,
        check_natr_period_exceeds_length,
        check_natr_very_small_dataset,
        check_natr_all_values_nan,
        check_natr_not_enough_valid_data,
        check_natr_slice_data_reinput,
        check_natr_nan_check,
        check_natr_default_candles,
        check_natr_streaming,
        check_natr_no_poison
    );

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

#[cfg(feature = "python")]
#[pyfunction(name = "natr")]
#[pyo3(signature = (high, low, close, period, kernel=None))]
pub fn natr_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = NatrParams {
        period: Some(period),
    };
    let input = NatrInput::from_slices(high_slice, low_slice, close_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| natr_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "natr_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
pub fn natr_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err("natr: Mismatched input lengths"));
    }
    let cols = high_slice.len();

    let kern = validate_kernel(kernel, true)?;
    let sweep = NatrBatchRange {
        period: period_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("natr_batch: rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
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
                _ => kernel,
            };
            natr_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                simd,
                true,
                slice_out,
            )
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

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "NatrStream")]
pub struct NatrStreamPy {
    stream: NatrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NatrStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = NatrParams {
            period: Some(period),
        };
        let stream =
            NatrStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NatrStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

/// Write directly to output slice - no allocations
pub fn natr_into_slice(dst: &mut [f64], input: &NatrInput, kern: Kernel) -> Result<(), NatrError> {
    let (high, low, close, period) = match &input.data {
        NatrData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            input.params.period.unwrap_or(14),
        ),
        NatrData::Slices { high, low, close } => {
            (*high, *low, *close, input.params.period.unwrap_or(14))
        }
    };

    let len = high.len().min(low.len()).min(close.len());

    if dst.len() != len {
        return Err(NatrError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    // Validate parameters
    if len == 0 {
        return Err(NatrError::EmptyInputData);
    }
    if period == 0 {
        return Err(NatrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period > len {
        return Err(NatrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // Find first valid index
    let first_valid_idx = {
        let first_valid_idx_h = high.iter().position(|&x| !x.is_nan());
        let first_valid_idx_l = low.iter().position(|&x| !x.is_nan());
        let first_valid_idx_c = close.iter().position(|&x| !x.is_nan());

        match (first_valid_idx_h, first_valid_idx_l, first_valid_idx_c) {
            (Some(h), Some(l), Some(c)) => Some(h.max(l).max(c)),
            _ => None,
        }
    };

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(NatrError::AllValuesNaN),
    };

    if (len - first_valid_idx) < period {
        return Err(NatrError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    // Choose kernel
    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    // Compute NATR directly into dst
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                natr_scalar(high, low, close, period, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                natr_avx2(high, low, close, period, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                natr_avx512(high, low, close, period, first_valid_idx, dst)
            }
            _ => unreachable!(),
        }
    }

    // Fill warmup with NaN
    for v in &mut dst[..(first_valid_idx + period - 1)] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Zero-allocation native API that writes results into a caller-provided buffer.
///
/// - Preserves NaN warmups exactly like the Vec-returning API.
/// - `out.len()` must equal the input length; returns an error on mismatch.
/// - Uses `Kernel::Auto` dispatch.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn natr_into(input: &NatrInput, out: &mut [f64]) -> Result<(), NatrError> {
    natr_into_slice(out, input, Kernel::Auto)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, JsValue> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(JsValue::from_str("natr: Mismatched input lengths"));
    }
    let params = NatrParams {
        period: Some(period),
    };
    let input = NatrInput::from_slices(high, low, close, params);
    let mut output = vec![0.0; high.len()];
    natr_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let params = NatrParams {
            period: Some(period),
        };
        let input = NatrInput::from_slices(high, low, close, params);

        // Check if any input pointer equals output pointer (aliasing)
        if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            natr_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            natr_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NatrBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NatrBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NatrParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = natr_batch)]
pub fn natr_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(JsValue::from_str("natr: Mismatched input lengths"));
    }
    let cfg: NatrBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = NatrBatchRange {
        period: cfg.period_range,
    };

    let out = natr_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = NatrBatchJsOutput {
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
pub fn natr_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to natr_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let sweep = NatrBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("natr_batch_into: rows*cols overflow"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        natr_batch_inner_into(high, low, close, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "python")]
pub fn register_natr_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(natr_py, m)?)?;
    m.add_function(wrap_pyfunction!(natr_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<NatrDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(natr_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(natr_cuda_many_series_one_param_dev_py, m)?)?;
    }
    Ok(())
}

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::natr_wrapper::DeviceArrayF32Natr;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyBufferError;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::PyObject;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct NatrDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl NatrDeviceArrayF32Py {
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
        // Producer stream is synchronized before returning; omit `stream`.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
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
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyBufferError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyBufferError::new_err(
                            "__dlpack__: requested device does not match producer buffer",
                        ));
                    }
                }
            }
        }
        let _ = stream;

        // copy=True is not yet supported (no cross-device copy path here).
        if let Some(copy_obj) = copy.as_ref() {
            let do_copy: bool = copy_obj.extract(py)?;
            if do_copy {
                return Err(PyBufferError::new_err(
                    "__dlpack__(copy=True) not supported for natr CUDA buffers",
                ));
            }
        }

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let rows = self.rows;
        let cols = self.cols;
        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "natr_cuda_batch_dev")]
#[pyo3(signature = (high, low, close, period_range, device_id=0))]
pub fn natr_cuda_batch_dev_py(
    py: Python<'_>,
    high: numpy::PyReadonlyArray1<'_, f32>,
    low: numpy::PyReadonlyArray1<'_, f32>,
    close: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<NatrDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaNatr;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    if h.len() != l.len() || h.len() != c.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }
    let sweep = NatrBatchRange {
        period: period_range,
    };
    let dev = py.allow_threads(|| {
        let mut cuda =
            CudaNatr::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.natr_batch_dev(h, l, c, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let DeviceArrayF32Natr { buf, rows, cols, ctx, device_id } = dev;
    Ok(NatrDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        ctx,
        device_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "natr_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, close_tm, period, device_id=0))]
pub fn natr_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm: numpy::PyReadonlyArray2<'_, f32>,
    low_tm: numpy::PyReadonlyArray2<'_, f32>,
    close_tm: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<NatrDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaNatr;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm.as_slice()?;
    let l = low_tm.as_slice()?;
    let c = close_tm.as_slice()?;
    let rows = high_tm.shape()[0];
    let cols = high_tm.shape()[1];

    let dev = py.allow_threads(|| {
        let mut cuda =
            CudaNatr::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.natr_many_series_one_param_time_major_dev(h, l, c, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let DeviceArrayF32Natr { buf, rows, cols, ctx, device_id } = dev;
    Ok(NatrDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        ctx,
        device_id,
    })
}
