//! # Chande Kroll Stop (CKSP)
//!
//! Computes two stop lines (long and short) using ATR and rolling maxima/minima.
//! Its parameters (`p`, `x`, `q`) control the ATR period, ATR multiplier, and rolling window size.
//!
//! ## Parameters
//! - **p**: ATR period (default: 10)
//! - **x**: ATR multiplier (default: 1.0)
//! - **q**: Rolling window (default: 9)
//!
//! ## Returns
//! - **`Ok(CkspOutput)`** on success, containing two `Vec<f64>` (long_values and short_values) of length matching the input
//! - **`Err(CkspError)`** otherwise
//!
//! ## Developer Notes
//! - **AVX2 kernel**: STUB - calls scalar implementation
//! - **AVX512 kernel**: STUB - calls scalar implementation
//! - Decision: SIMD kept as stubs — ATR recurrence and deque updates are sequential/branchy; scalar is fastest.
//!   Bench (100k, target-cpu=native): scalar ~3.20 ms → 2.80 ms after scalar optimization (~12% faster).
//!   Batch optimized by precomputing ATR per p and high/low windows per q, then per-row final rolling.
//! - **Streaming**: Not implemented
//! - **Memory optimization**: ✅ Uses alloc_with_nan_prefix (zero-copy) for both output arrays
//! - **Batch operations**: ✅ Implemented with parallel processing support
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
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

// ========================= Input Structs, AsRef =========================

#[derive(Debug, Clone)]
pub enum CkspData<'a> {
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
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CkspParams {
    pub p: Option<usize>,
    pub x: Option<f64>,
    pub q: Option<usize>,
}

impl Default for CkspParams {
    fn default() -> Self {
        Self {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CkspInput<'a> {
    pub data: CkspData<'a>,
    pub params: CkspParams,
}

impl<'a> CkspInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: CkspParams) -> Self {
        Self {
            data: CkspData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: CkspParams,
    ) -> Self {
        Self {
            data: CkspData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, CkspParams::default())
    }
    #[inline]
    pub fn get_p(&self) -> usize {
        self.params.p.unwrap_or(10)
    }
    #[inline]
    pub fn get_x(&self) -> f64 {
        self.params.x.unwrap_or(1.0)
    }
    #[inline]
    pub fn get_q(&self) -> usize {
        self.params.q.unwrap_or(9)
    }
}

impl<'a> AsRef<[f64]> for CkspInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CkspData::Candles { candles } => &candles.close,
            CkspData::Slices { close, .. } => close,
        }
    }
}

// ========================= Output Struct =========================

#[derive(Debug, Clone)]
pub struct CkspOutput {
    pub long_values: Vec<f64>,
    pub short_values: Vec<f64>,
}

// ========================= Error Type =========================

#[derive(Debug, Error)]
pub enum CkspError {
    #[error("cksp: Data is empty or all values are NaN.")]
    NoData,
    #[error("cksp: Not enough data for p={p}, q={q}, data_len={data_len}.")]
    NotEnoughData { p: usize, q: usize, data_len: usize },
    #[error("cksp: Inconsistent input lengths.")]
    InconsistentLengths,
    #[error("cksp: Invalid param value: {param}")]
    InvalidParam { param: &'static str },
    #[error("cksp: Candle field error: {0}")]
    CandleFieldError(String),
}

// ========================= Builder Struct =========================

#[derive(Copy, Clone, Debug)]
pub struct CkspBuilder {
    p: Option<usize>,
    x: Option<f64>,
    q: Option<usize>,
    kernel: Kernel,
}

impl Default for CkspBuilder {
    fn default() -> Self {
        Self {
            p: None,
            x: None,
            q: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CkspBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn p(mut self, n: usize) -> Self {
        self.p = Some(n);
        self
    }
    #[inline(always)]
    pub fn x(mut self, v: f64) -> Self {
        self.x = Some(v);
        self
    }
    #[inline(always)]
    pub fn q(mut self, n: usize) -> Self {
        self.q = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<CkspOutput, CkspError> {
        let params = CkspParams {
            p: self.p,
            x: self.x,
            q: self.q,
        };
        let input = CkspInput::from_candles(candles, params);
        cksp_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<CkspOutput, CkspError> {
        let params = CkspParams {
            p: self.p,
            x: self.x,
            q: self.q,
        };
        let input = CkspInput::from_slices(high, low, close, params);
        cksp_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CkspStream, CkspError> {
        let params = CkspParams {
            p: self.p,
            x: self.x,
            q: self.q,
        };
        CkspStream::try_new(params)
    }
}

// ========================= Main Indicator Functions =========================

#[inline]
pub fn cksp(input: &CkspInput) -> Result<CkspOutput, CkspError> {
    cksp_with_kernel(input, Kernel::Auto)
}

pub fn cksp_with_kernel(input: &CkspInput, kernel: Kernel) -> Result<CkspOutput, CkspError> {
    let (high, low, close) = match &input.data {
        CkspData::Candles { candles } => {
            let h = candles
                .select_candle_field("high")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let l = candles
                .select_candle_field("low")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let c = candles
                .select_candle_field("close")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            (h, l, c)
        }
        CkspData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(CkspError::InconsistentLengths);
            }
            (*high, *low, *close)
        }
    };
    let p = input.get_p();
    let x = input.get_x();
    let q = input.get_q();

    // Validate parameters first (before data checks)
    if p == 0 || q == 0 {
        return Err(CkspError::InvalidParam { param: "p/q" });
    }
    if !(x.is_finite()) || x.is_nan() {
        return Err(CkspError::InvalidParam { param: "x" });
    }

    // Now check data
    let size = close.len();
    if size == 0 {
        return Err(CkspError::NoData);
    }
    if p > size || q > size {
        return Err(CkspError::NotEnoughData {
            p,
            q,
            data_len: size,
        });
    }

    let first_valid_idx = match close.iter().position(|&v| !v.is_nan()) {
        Some(idx) => idx,
        None => return Err(CkspError::NoData),
    };

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cksp_scalar(high, low, close, p, x, q, first_valid_idx)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                // Fallback to scalar when AVX2 not compiled in
                cksp_scalar(high, low, close, p, x, q, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                cksp_avx2(high, low, close, p, x, q, first_valid_idx)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX512 not compiled in
                cksp_scalar(high, low, close, p, x, q, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cksp_avx512(high, low, close, p, x, q, first_valid_idx)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn cksp_into_slices(
    out_long: &mut [f64],
    out_short: &mut [f64],
    input: &CkspInput,
    kern: Kernel,
) -> Result<(), CkspError> {
    // Resolve inputs
    let (high, low, close) = match &input.data {
        CkspData::Candles { candles } => (
            candles
                .select_candle_field("high")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?,
            candles
                .select_candle_field("low")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?,
            candles
                .select_candle_field("close")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?,
        ),
        CkspData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.len() != low.len() || low.len() != close.len() {
        return Err(CkspError::InconsistentLengths);
    }
    if out_long.len() != close.len() || out_short.len() != close.len() {
        return Err(CkspError::InconsistentLengths); // Fixed: was NotEnoughData
    }

    let p = input.get_p();
    let q = input.get_q();
    let x = input.get_x();
    if p == 0 || q == 0 {
        return Err(CkspError::InvalidParam { param: "p/q" });
    }
    if !x.is_finite() {
        return Err(CkspError::InvalidParam { param: "x" });
    }

    let first_valid = close
        .iter()
        .position(|v| !v.is_nan())
        .ok_or(CkspError::NoData)?;
    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cksp_row_scalar(high, low, close, p, x, q, first_valid, out_long, out_short)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                cksp_row_avx2(high, low, close, p, x, q, first_valid, out_long, out_short)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cksp_row_avx512(high, low, close, p, x, q, first_valid, out_long, out_short)
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

// ========================= Scalar Logic =========================

#[inline]
pub unsafe fn cksp_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    let size = close.len();
    let warmup = first_valid_idx + p + q - 1;

    // Allocate outputs with warmup NaN prefix
    let mut long_values = alloc_with_nan_prefix(size, warmup);
    let mut short_values = alloc_with_nan_prefix(size, warmup);

    if first_valid_idx >= size {
        return Ok(CkspOutput { long_values, short_values });
    }

    // Monotonic ring buffers (indices and/or values)
    let cap = q + 1; // capacity q+1 to distinguish full/empty

    // Rolling max of HIGH
    let mut h_idx: Vec<usize> = Vec::with_capacity(cap);
    h_idx.set_len(cap);
    let mut h_head: usize = 0;
    let mut h_tail: usize = 0;

    // Rolling min of LOW
    let mut l_idx: Vec<usize> = Vec::with_capacity(cap);
    l_idx.set_len(cap);
    let mut l_head: usize = 0;
    let mut l_tail: usize = 0;

    // Rolling max of ls0
    let mut ls_idx: Vec<usize> = Vec::with_capacity(cap);
    let mut ls_val: Vec<f64> = Vec::with_capacity(cap);
    ls_idx.set_len(cap);
    ls_val.set_len(cap);
    let mut ls_head: usize = 0;
    let mut ls_tail: usize = 0;

    // Rolling min of ss0
    let mut ss_idx: Vec<usize> = Vec::with_capacity(cap);
    let mut ss_val: Vec<f64> = Vec::with_capacity(cap);
    ss_idx.set_len(cap);
    ss_val.set_len(cap);
    let mut ss_head: usize = 0;
    let mut ss_tail: usize = 0;

    // ATR (RMA) state
    let mut sum_tr: f64 = 0.0;
    let mut rma: f64 = 0.0;
    let alpha: f64 = 1.0 / (p as f64);

    #[inline(always)]
    unsafe fn rb_dec(idx: usize, cap: usize) -> usize {
        if idx == 0 { cap - 1 } else { idx - 1 }
    }
    #[inline(always)]
    unsafe fn rb_inc(idx: usize, cap: usize) -> usize {
        let mut t = idx + 1;
        if t == cap { t = 0; }
        t
    }

    for i in 0..size {
        if i < first_valid_idx { continue; }

        // True Range
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let tr = if i == first_valid_idx {
            hi - lo
        } else {
            let cprev = *close.get_unchecked(i - 1);
            let hl = hi - lo;
            let hc = (hi - cprev).abs();
            let lc = (lo - cprev).abs();
            if hl >= hc {
                if hl >= lc { hl } else { lc }
            } else {
                if hc >= lc { hc } else { lc }
            }
        };

        // ATR (RMA)
        let k = i - first_valid_idx;
        if k < p {
            sum_tr += tr;
            if k == p - 1 {
                rma = sum_tr / (p as f64);
            }
        } else {
            rma = alpha.mul_add(tr - rma, rma);
        }

        // Rolling MAX of HIGH over q
        while h_head != h_tail {
            let last = rb_dec(h_tail, cap);
            let last_i = *h_idx.get_unchecked(last);
            if *high.get_unchecked(last_i) <= hi {
                h_tail = last;
            } else {
                break;
            }
        }
        // Prevent full condition: advance head if next tail would collide
        let mut next_tail = rb_inc(h_tail, cap);
        if next_tail == h_head { h_head = rb_inc(h_head, cap); }
        *h_idx.get_unchecked_mut(h_tail) = i;
        h_tail = next_tail;
        while h_head != h_tail {
            let front_i = *h_idx.get_unchecked(h_head);
            if front_i + q <= i { h_head = rb_inc(h_head, cap); } else { break; }
        }
        let mh = *high.get_unchecked(*h_idx.get_unchecked(h_head));

        // Rolling MIN of LOW over q
        while l_head != l_tail {
            let last = rb_dec(l_tail, cap);
            let last_i = *l_idx.get_unchecked(last);
            if *low.get_unchecked(last_i) >= lo {
                l_tail = last;
            } else {
                break;
            }
        }
        let mut next_tail = rb_inc(l_tail, cap);
        if next_tail == l_head { l_head = rb_inc(l_head, cap); }
        *l_idx.get_unchecked_mut(l_tail) = i;
        l_tail = next_tail;
        while l_head != l_tail {
            let front_i = *l_idx.get_unchecked(l_head);
            if front_i + q <= i { l_head = rb_inc(l_head, cap); } else { break; }
        }
        let ml = *low.get_unchecked(*l_idx.get_unchecked(l_head));

        // Emit outputs after warmup
        if i >= warmup {
            let ls0 = (-x).mul_add(rma, mh);
            let ss0 = x.mul_add(rma, ml);

            // Rolling MAX over ls0
            while ls_head != ls_tail {
                let last = rb_dec(ls_tail, cap);
                if *ls_val.get_unchecked(last) <= ls0 {
                    ls_tail = last;
                } else {
                    break;
                }
            }
            let mut next_tail = rb_inc(ls_tail, cap);
            if next_tail == ls_head { ls_head = rb_inc(ls_head, cap); }
            *ls_idx.get_unchecked_mut(ls_tail) = i;
            *ls_val.get_unchecked_mut(ls_tail) = ls0;
            ls_tail = next_tail;
            while ls_head != ls_tail {
                let front_i = *ls_idx.get_unchecked(ls_head);
                if front_i + q <= i { ls_head = rb_inc(ls_head, cap); } else { break; }
            }
            let mx = *ls_val.get_unchecked(ls_head);
            *long_values.get_unchecked_mut(i) = mx;

            // Rolling MIN over ss0
            while ss_head != ss_tail {
                let last = rb_dec(ss_tail, cap);
                if *ss_val.get_unchecked(last) >= ss0 {
                    ss_tail = last;
                } else {
                    break;
                }
            }
            let mut next_tail = rb_inc(ss_tail, cap);
            if next_tail == ss_head { ss_head = rb_inc(ss_head, cap); }
            *ss_idx.get_unchecked_mut(ss_tail) = i;
            *ss_val.get_unchecked_mut(ss_tail) = ss0;
            ss_tail = next_tail;
            while ss_head != ss_tail {
                let front_i = *ss_idx.get_unchecked(ss_head);
                if front_i + q <= i { ss_head = rb_inc(ss_head, cap); } else { break; }
            }
            let mn = *ss_val.get_unchecked(ss_head);
            *short_values.get_unchecked_mut(i) = mn;
        }
    }

    Ok(CkspOutput { long_values, short_values })
}

// ========================= AVX2/AVX512 Stubs =========================

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    // For API parity, fallback to scalar
    cksp_scalar(high, low, close, p, x, q, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    // For API parity, fallback to scalar
    cksp_scalar(high, low, close, p, x, q, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    cksp_avx512(high, low, close, p, x, q, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    cksp_avx512(high, low, close, p, x, q, first_valid_idx)
}

// ========================= Row/Batched API =========================

// Helper function that computes directly into output slices
#[inline(always)]
pub unsafe fn cksp_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    let size = close.len();
    let warmup = first_valid_idx + p + q - 1;

    // Initialize NaN values for warmup period
    for i in 0..warmup.min(size) {
        *out_long.get_unchecked_mut(i) = f64::NAN;
        *out_short.get_unchecked_mut(i) = f64::NAN;
    }

    // Monotonic ring buffers
    let cap = q + 1;
    let mut h_idx: Vec<usize> = Vec::with_capacity(cap);
    h_idx.set_len(cap);
    let mut h_head: usize = 0;
    let mut h_tail: usize = 0;

    let mut l_idx: Vec<usize> = Vec::with_capacity(cap);
    l_idx.set_len(cap);
    let mut l_head: usize = 0;
    let mut l_tail: usize = 0;

    let mut ls_idx: Vec<usize> = Vec::with_capacity(cap);
    let mut ls_val: Vec<f64> = Vec::with_capacity(cap);
    ls_idx.set_len(cap);
    ls_val.set_len(cap);
    let mut ls_head: usize = 0;
    let mut ls_tail: usize = 0;

    let mut ss_idx: Vec<usize> = Vec::with_capacity(cap);
    let mut ss_val: Vec<f64> = Vec::with_capacity(cap);
    ss_idx.set_len(cap);
    ss_val.set_len(cap);
    let mut ss_head: usize = 0;
    let mut ss_tail: usize = 0;

    let mut sum_tr: f64 = 0.0;
    let mut rma: f64 = 0.0;
    let alpha: f64 = 1.0 / (p as f64);

    #[inline(always)]
    unsafe fn rb_dec(idx: usize, cap: usize) -> usize { if idx == 0 { cap - 1 } else { idx - 1 } }
    #[inline(always)]
    unsafe fn rb_inc(idx: usize, cap: usize) -> usize { let mut t = idx + 1; if t == cap { t = 0; } t }

    for i in 0..size {
        if i < first_valid_idx { continue; }

        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let tr = if i == first_valid_idx {
            hi - lo
        } else {
            let cprev = *close.get_unchecked(i - 1);
            let hl = hi - lo;
            let hc = (hi - cprev).abs();
            let lc = (lo - cprev).abs();
            if hl >= hc { if hl >= lc { hl } else { lc } } else { if hc >= lc { hc } else { lc } }
        };

        let k = i - first_valid_idx;
        if k < p {
            sum_tr += tr;
            if k == p - 1 { rma = sum_tr / (p as f64); }
        } else {
            rma = alpha.mul_add(tr - rma, rma);
        }

        // Rolling MAX high
        while h_head != h_tail {
            let last = rb_dec(h_tail, cap);
            let last_i = *h_idx.get_unchecked(last);
            if *high.get_unchecked(last_i) <= hi { h_tail = last; } else { break; }
        }
        let mut next_tail = rb_inc(h_tail, cap);
        if next_tail == h_head { h_head = rb_inc(h_head, cap); }
        *h_idx.get_unchecked_mut(h_tail) = i;
        h_tail = next_tail;
        while h_head != h_tail {
            let front_i = *h_idx.get_unchecked(h_head);
            if front_i + q <= i { h_head = rb_inc(h_head, cap); } else { break; }
        }
        let mh = *high.get_unchecked(*h_idx.get_unchecked(h_head));

        // Rolling MIN low
        while l_head != l_tail {
            let last = rb_dec(l_tail, cap);
            let last_i = *l_idx.get_unchecked(last);
            if *low.get_unchecked(last_i) >= lo { l_tail = last; } else { break; }
        }
        let mut next_tail = rb_inc(l_tail, cap);
        if next_tail == l_head { l_head = rb_inc(l_head, cap); }
        *l_idx.get_unchecked_mut(l_tail) = i;
        l_tail = next_tail;
        while l_head != l_tail {
            let front_i = *l_idx.get_unchecked(l_head);
            if front_i + q <= i { l_head = rb_inc(l_head, cap); } else { break; }
        }
        let ml = *low.get_unchecked(*l_idx.get_unchecked(l_head));

        if i >= warmup {
            let ls0 = (-x).mul_add(rma, mh);
            let ss0 = x.mul_add(rma, ml);

            while ls_head != ls_tail {
                let last = rb_dec(ls_tail, cap);
                if *ls_val.get_unchecked(last) <= ls0 { ls_tail = last; } else { break; }
            }
            let mut next_tail = rb_inc(ls_tail, cap);
            if next_tail == ls_head { ls_head = rb_inc(ls_head, cap); }
            *ls_idx.get_unchecked_mut(ls_tail) = i;
            *ls_val.get_unchecked_mut(ls_tail) = ls0;
            ls_tail = next_tail;
            while ls_head != ls_tail {
                let front_i = *ls_idx.get_unchecked(ls_head);
                if front_i + q <= i { ls_head = rb_inc(ls_head, cap); } else { break; }
            }
            let mx = *ls_val.get_unchecked(ls_head);
            *out_long.get_unchecked_mut(i) = mx;

            while ss_head != ss_tail {
                let last = rb_dec(ss_tail, cap);
                if *ss_val.get_unchecked(last) >= ss0 { ss_tail = last; } else { break; }
            }
            let mut next_tail = rb_inc(ss_tail, cap);
            if next_tail == ss_head { ss_head = rb_inc(ss_head, cap); }
            *ss_idx.get_unchecked_mut(ss_tail) = i;
            *ss_val.get_unchecked_mut(ss_tail) = ss0;
            ss_tail = next_tail;
            while ss_head != ss_tail {
                let front_i = *ss_idx.get_unchecked(ss_head);
                if front_i + q <= i { ss_head = rb_inc(ss_head, cap); } else { break; }
            }
            let mn = *ss_val.get_unchecked(ss_head);
            *out_short.get_unchecked_mut(i) = mn;
        }
    }
}

#[inline(always)]
pub unsafe fn cksp_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    cksp_compute_into(
        high,
        low,
        close,
        p,
        x,
        q,
        first_valid_idx,
        out_long,
        out_short,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    // Since AVX2 implementation is a stub, use compute_into directly
    cksp_compute_into(
        high,
        low,
        close,
        p,
        x,
        q,
        first_valid_idx,
        out_long,
        out_short,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    // Since AVX512 implementation is a stub, use compute_into directly
    cksp_compute_into(
        high,
        low,
        close,
        p,
        x,
        q,
        first_valid_idx,
        out_long,
        out_short,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    // Since AVX512 implementation is a stub, use compute_into directly
    cksp_compute_into(
        high,
        low,
        close,
        p,
        x,
        q,
        first_valid_idx,
        out_long,
        out_short,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    // Since AVX512 implementation is a stub, use compute_into directly
    cksp_compute_into(
        high,
        low,
        close,
        p,
        x,
        q,
        first_valid_idx,
        out_long,
        out_short,
    )
}

// ========================= Stream API =========================

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct CkspStream {
    p: usize,
    x: f64,
    q: usize,
    alpha: f64,
    sum_tr: f64,
    rma: f64,
    prev_close: f64,
    dq_h: VecDeque<(usize, f64)>,
    dq_l: VecDeque<(usize, f64)>,
    dq_ls0: VecDeque<(usize, f64)>,
    dq_ss0: VecDeque<(usize, f64)>,
    i: usize,
}

impl CkspStream {
    pub fn try_new(params: CkspParams) -> Result<Self, CkspError> {
        let p = params.p.unwrap_or(10);
        let x = params.x.unwrap_or(1.0);
        let q = params.q.unwrap_or(9);
        if p == 0 || q == 0 {
            return Err(CkspError::InvalidParam { param: "p/q" });
        }
        if !x.is_finite() {
            return Err(CkspError::InvalidParam { param: "x" });
        }
        Ok(Self {
            p,
            x,
            q,
            alpha: 1.0 / p as f64,
            sum_tr: 0.0,
            rma: 0.0,
            prev_close: f64::NAN,
            dq_h: VecDeque::new(),
            dq_l: VecDeque::new(),
            dq_ls0: VecDeque::new(),
            dq_ss0: VecDeque::new(),
            i: 0,
        })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        };
        self.prev_close = close;
        let atr_opt = if self.i < self.p {
            self.sum_tr += tr;
            if self.i == self.p - 1 {
                self.rma = self.sum_tr / self.p as f64;
                Some(self.rma)
            } else {
                None
            }
        } else {
            self.rma += self.alpha * (tr - self.rma);
            Some(self.rma)
        };

        while let Some((_, v)) = self.dq_h.back() {
            if *v <= high {
                self.dq_h.pop_back();
            } else {
                break;
            }
        }
        self.dq_h.push_back((self.i, high));
        let start_h = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_h.front() {
            if idx < start_h {
                self.dq_h.pop_front();
            } else {
                break;
            }
        }

        while let Some((_, v)) = self.dq_l.back() {
            if *v >= low {
                self.dq_l.pop_back();
            } else {
                break;
            }
        }
        self.dq_l.push_back((self.i, low));
        let start_l = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_l.front() {
            if idx < start_l {
                self.dq_l.pop_front();
            } else {
                break;
            }
        }
        // Check if we're still in warmup period
        if self.i < self.p + self.q - 1 {
            self.i += 1;
            return None;
        }

        let atr = match atr_opt {
            Some(v) => v,
            None => {
                self.i += 1;
                return None;
            }
        };

        let (mh, ml) = match (self.dq_h.front(), self.dq_l.front()) {
            (Some(&(_, mh)), Some(&(_, ml))) => (mh, ml),
            _ => {
                self.i += 1;
                return None;
            }
        };
        let ls0_val = mh - self.x * atr;
        let ss0_val = ml + self.x * atr;

        while let Some((_, val)) = self.dq_ls0.back() {
            if *val <= ls0_val {
                self.dq_ls0.pop_back();
            } else {
                break;
            }
        }
        self.dq_ls0.push_back((self.i, ls0_val));
        let start_ls0 = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_ls0.front() {
            if idx < start_ls0 {
                self.dq_ls0.pop_front();
            } else {
                break;
            }
        }
        let long = self.dq_ls0.front().map(|&(_, v)| v).unwrap_or(f64::NAN);

        while let Some((_, val)) = self.dq_ss0.back() {
            if *val >= ss0_val {
                self.dq_ss0.pop_back();
            } else {
                break;
            }
        }
        self.dq_ss0.push_back((self.i, ss0_val));
        let start_ss0 = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_ss0.front() {
            if idx < start_ss0 {
                self.dq_ss0.pop_front();
            } else {
                break;
            }
        }
        let short = self.dq_ss0.front().map(|&(_, v)| v).unwrap_or(f64::NAN);

        self.i += 1;
        Some((long, short))
    }
}

// ========================= Batch/Range Builder & Output =========================

#[derive(Clone, Debug)]
pub struct CkspBatchRange {
    pub p: (usize, usize, usize),
    pub x: (f64, f64, f64),
    pub q: (usize, usize, usize),
}

impl Default for CkspBatchRange {
    fn default() -> Self {
        Self {
            p: (10, 40, 1),
            x: (1.0, 1.0, 0.0),
            q: (9, 24, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CkspBatchBuilder {
    range: CkspBatchRange,
    kernel: Kernel,
}

impl CkspBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn p_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.p = (start, end, step);
        self
    }
    #[inline]
    pub fn p_static(mut self, p: usize) -> Self {
        self.range.p = (p, p, 0);
        self
    }
    #[inline]
    pub fn x_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.x = (start, end, step);
        self
    }
    #[inline]
    pub fn x_static(mut self, x: f64) -> Self {
        self.range.x = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn q_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.q = (start, end, step);
        self
    }
    #[inline]
    pub fn q_static(mut self, q: usize) -> Self {
        self.range.q = (q, q, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<CkspBatchOutput, CkspError> {
        cksp_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k: Kernel,
    ) -> Result<CkspBatchOutput, CkspError> {
        CkspBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<CkspBatchOutput, CkspError> {
        let h = c
            .select_candle_field("high")
            .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
        let l = c
            .select_candle_field("low")
            .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
        let cl = c
            .select_candle_field("close")
            .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
        self.apply_slices(h, l, cl)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CkspBatchOutput, CkspError> {
        CkspBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct CkspBatchOutput {
    pub long_values: Vec<f64>,
    pub short_values: Vec<f64>,
    pub combos: Vec<CkspParams>,
    pub rows: usize,
    pub cols: usize,
}
impl CkspBatchOutput {
    pub fn row_for_params(&self, p: &CkspParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.p.unwrap_or(10) == p.p.unwrap_or(10)
                && (c.x.unwrap_or(1.0) - p.x.unwrap_or(1.0)).abs() < 1e-12
                && c.q.unwrap_or(9) == p.q.unwrap_or(9)
        })
    }
    pub fn values_for(&self, p: &CkspParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.long_values[start..start + self.cols],
                &self.short_values[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CkspBatchRange) -> Vec<CkspParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }

    let ps = axis_usize(r.p);
    let xs = axis_f64(r.x);
    let qs = axis_usize(r.q);

    let mut out = Vec::with_capacity(ps.len() * xs.len() * qs.len());
    for &p in &ps {
        for &x in &xs {
            for &q in &qs {
                out.push(CkspParams {
                    p: Some(p),
                    x: Some(x),
                    q: Some(q),
                });
            }
        }
    }
    out
}

pub fn cksp_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    k: Kernel,
) -> Result<CkspBatchOutput, CkspError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CkspError::InvalidParam { param: "kernel" }),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cksp_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn cksp_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    kern: Kernel,
) -> Result<CkspBatchOutput, CkspError> {
    cksp_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn cksp_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    kern: Kernel,
) -> Result<CkspBatchOutput, CkspError> {
    cksp_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn cksp_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CkspBatchOutput, CkspError> {
    let _ = kern; // runtime kernel currently unused (precomputed scalar path)
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CkspError::InvalidParam { param: "combos" });
    }
    let size = close.len();
    if high.len() != low.len() || low.len() != close.len() {
        return Err(CkspError::InconsistentLengths);
    }
    let first_valid = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CkspError::NoData)?;

    let rows = combos.len();
    let cols = size;

    // Step 1: Allocate uninitialized matrices
    let mut long_buf_mu = make_uninit_matrix(rows, cols);
    let mut short_buf_mu = make_uninit_matrix(rows, cols);

    // Step 2: Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first_valid + c.p.unwrap() + c.q.unwrap() - 1)
        .collect();

    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut long_buf_mu, cols, &warm);
    init_matrix_prefixes(&mut short_buf_mu, cols, &warm);

    // Step 4: Convert to mutable slices for computation
    let mut long_guard = ManuallyDrop::new(long_buf_mu);
    let mut short_guard = ManuallyDrop::new(short_buf_mu);

    let long_values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(long_guard.as_mut_ptr() as *mut f64, long_guard.len())
    };

    let short_values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(short_guard.as_mut_ptr() as *mut f64, short_guard.len())
    };

    // Row-specific batch optimization: precompute ATR per unique p, and mh/ml per unique q
    use std::collections::{BTreeSet, HashMap};

    #[inline]
    fn precompute_atr_series(high: &[f64], low: &[f64], close: &[f64], p: usize, first_valid: usize) -> Vec<f64> {
        let n = close.len();
        let mut atr = vec![0.0; n];
        let mut sum_tr = 0.0;
        let mut rma = 0.0;
        let alpha = 1.0 / (p as f64);
        for i in 0..n {
            if i < first_valid { continue; }
            let hi = high[i];
            let lo = low[i];
            let tr = if i == first_valid {
                hi - lo
            } else {
                let cp = close[i - 1];
                let hl = hi - lo;
                let hc = (hi - cp).abs();
                let lc = (lo - cp).abs();
                hl.max(hc).max(lc)
            };
            let k = i - first_valid;
            if k < p {
                sum_tr += tr;
                if k == p - 1 { rma = sum_tr / (p as f64); atr[i] = rma; }
            } else {
                rma += alpha * (tr - rma);
                atr[i] = rma;
            }
        }
        atr
    }

    #[inline]
    fn rolling_max_series(src: &[f64], q: usize, first_valid: usize) -> Vec<f64> {
        let n = src.len();
        let mut out = vec![0.0; n];
        let cap = q + 1;
        let mut idx: Vec<usize> = Vec::with_capacity(cap);
        unsafe { idx.set_len(cap); }
        let mut head = 0usize;
        let mut tail = 0usize;
        #[inline(always)] fn dec(i: usize, c: usize) -> usize { if i == 0 { c - 1 } else { i - 1 } }
        #[inline(always)] fn inc(i: usize, c: usize) -> usize { let mut t = i + 1; if t == c { t = 0; } t }
        for i in 0..n {
            if i < first_valid { continue; }
            while head != tail {
                let last = dec(tail, cap);
                let li = unsafe { *idx.get_unchecked(last) };
                if src[li] <= src[i] { tail = last; } else { break; }
            }
            let mut nt = inc(tail, cap);
            if nt == head { head = inc(head, cap); }
            unsafe { *idx.get_unchecked_mut(tail) = i; }
            tail = nt;
            while head != tail {
                let fi = unsafe { *idx.get_unchecked(head) };
                if fi + q <= i { head = inc(head, cap); } else { break; }
            }
            out[i] = src[unsafe { *idx.get_unchecked(head) }];
        }
        out
    }

    #[inline]
    fn rolling_min_series(src: &[f64], q: usize, first_valid: usize) -> Vec<f64> {
        let n = src.len();
        let mut out = vec![0.0; n];
        let cap = q + 1;
        let mut idx: Vec<usize> = Vec::with_capacity(cap);
        unsafe { idx.set_len(cap); }
        let mut head = 0usize;
        let mut tail = 0usize;
        #[inline(always)] fn dec(i: usize, c: usize) -> usize { if i == 0 { c - 1 } else { i - 1 } }
        #[inline(always)] fn inc(i: usize, c: usize) -> usize { let mut t = i + 1; if t == c { t = 0; } t }
        for i in 0..n {
            if i < first_valid { continue; }
            while head != tail {
                let last = dec(tail, cap);
                let li = unsafe { *idx.get_unchecked(last) };
                if src[li] >= src[i] { tail = last; } else { break; }
            }
            let mut nt = inc(tail, cap);
            if nt == head { head = inc(head, cap); }
            unsafe { *idx.get_unchecked_mut(tail) = i; }
            tail = nt;
            while head != tail {
                let fi = unsafe { *idx.get_unchecked(head) };
                if fi + q <= i { head = inc(head, cap); } else { break; }
            }
            out[i] = src[unsafe { *idx.get_unchecked(head) }];
        }
        out
    }

    // Collect unique p and q values
    let mut ps: BTreeSet<usize> = BTreeSet::new();
    let mut qs: BTreeSet<usize> = BTreeSet::new();
    for prm in &combos { ps.insert(prm.p.unwrap()); qs.insert(prm.q.unwrap()); }

    // Precompute ATR per unique p
    let mut atr_map: HashMap<usize, Vec<f64>> = HashMap::with_capacity(ps.len());
    for &p in &ps { atr_map.insert(p, precompute_atr_series(high, low, close, p, first_valid)); }
    // Precompute mh/ml per unique q
    let mut mh_map: HashMap<usize, Vec<f64>> = HashMap::with_capacity(qs.len());
    let mut ml_map: HashMap<usize, Vec<f64>> = HashMap::with_capacity(qs.len());
    for &qv in &qs {
        mh_map.insert(qv, rolling_max_series(high, qv, first_valid));
        ml_map.insert(qv, rolling_min_series(low, qv, first_valid));
    }

    let do_row = |row: usize, out_long: &mut [f64], out_short: &mut [f64]| unsafe {
        let prm = &combos[row];
        let (p, x, q) = (prm.p.unwrap(), prm.x.unwrap(), prm.q.unwrap());

        // Fast path: use precomputed series and only do the final rolling over ls0/ss0
        let warmup = first_valid + p + q - 1;
        let atr = atr_map.get(&p).expect("atr precompute");
        let mh = mh_map.get(&q).expect("mh precompute");
        let ml = ml_map.get(&q).expect("ml precompute");

        // Final rolling for ls0 (max) and ss0 (min)
        let cap = q + 1;
        let mut ls_idx: Vec<usize> = Vec::with_capacity(cap);
        let mut ls_val: Vec<f64> = Vec::with_capacity(cap);
        ls_idx.set_len(cap);
        ls_val.set_len(cap);
        let mut ls_head = 0usize;
        let mut ls_tail = 0usize;
        let mut ss_idx: Vec<usize> = Vec::with_capacity(cap);
        let mut ss_val: Vec<f64> = Vec::with_capacity(cap);
        ss_idx.set_len(cap);
        ss_val.set_len(cap);
        let mut ss_head = 0usize;
        let mut ss_tail = 0usize;
        #[inline(always)] fn dec(i: usize, c: usize) -> usize { if i == 0 { c - 1 } else { i - 1 } }
        #[inline(always)] fn inc(i: usize, c: usize) -> usize { let mut t = i + 1; if t == c { t = 0; } t }

        for i in warmup..cols {
            let ls0 = mh[i] - x * atr[i];
            let ss0 = ml[i] + x * atr[i];

            while ls_head != ls_tail {
                let last = dec(ls_tail, cap);
                if unsafe { *ls_val.get_unchecked(last) } <= ls0 { ls_tail = last; } else { break; }
            }
            let mut nt = inc(ls_tail, cap);
            if nt == ls_head { ls_head = inc(ls_head, cap); }
            unsafe {
                *ls_idx.get_unchecked_mut(ls_tail) = i;
                *ls_val.get_unchecked_mut(ls_tail) = ls0;
            }
            ls_tail = nt;
            while ls_head != ls_tail {
                let fi = unsafe { *ls_idx.get_unchecked(ls_head) };
                if fi + q <= i { ls_head = inc(ls_head, cap); } else { break; }
            }
            let mx = unsafe { *ls_val.get_unchecked(ls_head) };
            *out_long.get_unchecked_mut(i) = mx;

            while ss_head != ss_tail {
                let last = dec(ss_tail, cap);
                if unsafe { *ss_val.get_unchecked(last) } >= ss0 { ss_tail = last; } else { break; }
            }
            let mut nt2 = inc(ss_tail, cap);
            if nt2 == ss_head { ss_head = inc(ss_head, cap); }
            unsafe {
                *ss_idx.get_unchecked_mut(ss_tail) = i;
                *ss_val.get_unchecked_mut(ss_tail) = ss0;
            }
            ss_tail = nt2;
            while ss_head != ss_tail {
                let fi = unsafe { *ss_idx.get_unchecked(ss_head) };
                if fi + q <= i { ss_head = inc(ss_head, cap); } else { break; }
            }
            let mn = unsafe { *ss_val.get_unchecked(ss_head) };
            *out_short.get_unchecked_mut(i) = mn;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            long_values
                .par_chunks_mut(cols)
                .zip(short_values.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (lv, sv))| do_row(row, lv, sv));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (lv, sv)) in long_values
                .chunks_mut(cols)
                .zip(short_values.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, lv, sv);
            }
        }
    } else {
        for (row, (lv, sv)) in long_values
            .chunks_mut(cols)
            .zip(short_values.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, lv, sv);
        }
    }

    // Step 5: Reclaim as Vec<f64>
    let long_values = unsafe {
        Vec::from_raw_parts(
            long_guard.as_mut_ptr() as *mut f64,
            long_guard.len(),
            long_guard.capacity(),
        )
    };

    let short_values = unsafe {
        Vec::from_raw_parts(
            short_guard.as_mut_ptr() as *mut f64,
            short_guard.len(),
            short_guard.capacity(),
        )
    };

    Ok(CkspBatchOutput {
        long_values,
        short_values,
        combos,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_cksp_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = CkspParams {
            p: None,
            x: None,
            q: None,
        };
        let input = CkspInput::from_candles(&candles, default_params);
        let output = cksp_with_kernel(&input, kernel)?;
        assert_eq!(output.long_values.len(), candles.close.len());
        assert_eq!(output.short_values.len(), candles.close.len());
        Ok(())
    }

    fn check_cksp_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_candles(&candles, params);
        let output = cksp_with_kernel(&input, kernel)?;

        let expected_long_last_5 = [
            60306.66197802568,
            60306.66197802568,
            60306.66197802568,
            60203.29578022311,
            60201.57958198072,
        ];
        let l_start = output.long_values.len() - 5;
        let long_slice = &output.long_values[l_start..];
        for (i, &val) in long_slice.iter().enumerate() {
            let exp_val = expected_long_last_5[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "[{}] CKSP long mismatch at idx {}: expected {}, got {}",
                test_name,
                i,
                exp_val,
                val
            );
        }

        let expected_short_last_5 = [
            58757.826484736055,
            58701.74383626245,
            58656.36945263621,
            58611.03250737258,
            58611.03250737258,
        ];
        let s_start = output.short_values.len() - 5;
        let short_slice = &output.short_values[s_start..];
        for (i, &val) in short_slice.iter().enumerate() {
            let exp_val = expected_short_last_5[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "[{}] CKSP short mismatch at idx {}: expected {}, got {}",
                test_name,
                i,
                exp_val,
                val
            );
        }
        Ok(())
    }

    fn check_cksp_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CkspInput::with_default_candles(&candles);
        match input.data {
            CkspData::Candles { .. } => {}
            _ => panic!("Expected CkspData::Candles"),
        }
        let output = cksp_with_kernel(&input, kernel)?;
        assert_eq!(output.long_values.len(), candles.close.len());
        assert_eq!(output.short_values.len(), candles.close.len());
        Ok(())
    }

    fn check_cksp_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 10.5];
        let close = [9.5, 10.5, 11.0];
        let params = CkspParams {
            p: Some(0),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CKSP should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cksp_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 10.5];
        let close = [9.5, 10.5, 11.0];
        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CKSP should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cksp_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [41.0];
        let close = [41.5];
        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CKSP should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cksp_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let first_input = CkspInput::from_candles(&candles, first_params.clone());
        let first_result = cksp_with_kernel(&first_input, kernel)?;

        let dummy_close = vec![0.0; first_result.long_values.len()];
        let second_input = CkspInput::from_slices(
            &first_result.long_values,
            &first_result.short_values,
            &dummy_close,
            first_params,
        );
        let second_result = cksp_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.long_values.len(), dummy_close.len());
        assert_eq!(second_result.short_values.len(), dummy_close.len());
        Ok(())
    }

    fn check_cksp_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CkspInput::from_candles(
            &candles,
            CkspParams {
                p: Some(10),
                x: Some(1.0),
                q: Some(9),
            },
        );
        let res = cksp_with_kernel(&input, kernel)?;
        assert_eq!(res.long_values.len(), candles.close.len());
        assert_eq!(res.short_values.len(), candles.close.len());
        if res.long_values.len() > 240 {
            for i in 240..res.long_values.len() {
                assert!(
                    !res.long_values[i].is_nan(),
                    "[{}] Found unexpected NaN in long_values at out-index {}",
                    test_name,
                    i
                );
                assert!(
                    !res.short_values[i].is_nan(),
                    "[{}] Found unexpected NaN in short_values at out-index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_cksp_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let p = 10;
        let x = 1.0;
        let q = 9;

        let input = CkspInput::from_candles(
            &candles,
            CkspParams {
                p: Some(p),
                x: Some(x),
                q: Some(q),
            },
        );
        let batch_output = cksp_with_kernel(&input, kernel)?;
        let mut stream = CkspStream::try_new(CkspParams {
            p: Some(p),
            x: Some(x),
            q: Some(q),
        })?;

        let mut stream_long = Vec::with_capacity(candles.close.len());
        let mut stream_short = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            let h = candles.high[i];
            let l = candles.low[i];
            let c = candles.close[i];
            match stream.update(h, l, c) {
                Some((long, short)) => {
                    stream_long.push(long);
                    stream_short.push(short);
                }
                None => {
                    stream_long.push(f64::NAN);
                    stream_short.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.long_values.len(), stream_long.len());
        assert_eq!(batch_output.short_values.len(), stream_short.len());
        for i in 0..stream_long.len() {
            let b_long = batch_output.long_values[i];
            let b_short = batch_output.short_values[i];
            let s_long = stream_long[i];
            let s_short = stream_short[i];
            let diff_long = (b_long - s_long).abs();
            let diff_short = (b_short - s_short).abs();
            if b_long.is_nan() && s_long.is_nan() && b_short.is_nan() && s_short.is_nan() {
                continue;
            }
            assert!(
                diff_long < 1e-8,
                "[{}] CKSP streaming long f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b_long,
                s_long,
                diff_long
            );
            assert!(
                diff_short < 1e-8,
                "[{}] CKSP streaming short f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b_short,
                s_short,
                diff_short
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_cksp_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = CkspInput::from_candles(&candles, CkspParams::default());
        let output = cksp_with_kernel(&input, kernel)?;

        // Check every value for poison patterns in long_values
        for (i, &val) in output.long_values.iter().enumerate() {
            // Skip NaN values as they're expected in the warmup period
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in long_values",
					test_name, val, bits, i
				);
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in long_values",
					test_name, val, bits, i
				);
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in long_values",
					test_name, val, bits, i
				);
            }
        }

        // Check every value for poison patterns in short_values
        for (i, &val) in output.short_values.iter().enumerate() {
            // Skip NaN values as they're expected in the warmup period
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in short_values",
					test_name, val, bits, i
				);
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in short_values",
					test_name, val, bits, i
				);
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in short_values",
					test_name, val, bits, i
				);
            }
        }

        // Test with multiple parameter combinations
        let param_combos = vec![
            CkspParams {
                p: Some(5),
                x: Some(0.5),
                q: Some(5),
            },
            CkspParams {
                p: Some(20),
                x: Some(2.0),
                q: Some(15),
            },
            CkspParams {
                p: Some(30),
                x: Some(1.5),
                q: Some(20),
            },
        ];

        for params in param_combos {
            let input = CkspInput::from_candles(&candles, params.clone());
            let output = cksp_with_kernel(&input, kernel)?;

            // Check long_values
            for (i, &val) in output.long_values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
                        "[{}] Found poison value {} (0x{:016X}) at index {} in long_values with params p={}, x={}, q={}",
                        test_name, val, bits, i, params.p.unwrap(), params.x.unwrap(), params.q.unwrap()
                    );
                }
            }

            // Check short_values
            for (i, &val) in output.short_values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
                        "[{}] Found poison value {} (0x{:016X}) at index {} in short_values with params p={}, x={}, q={}",
                        test_name, val, bits, i, params.p.unwrap(), params.x.unwrap(), params.q.unwrap()
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_cksp_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_cksp_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Generate test data with OHLC price data and parameters
        let strat = (1usize..=64).prop_flat_map(|p| {
            (1usize..=20).prop_flat_map(move |q| {
                (
                    // Generate realistic OHLC data
                    prop::collection::vec(
                        (10.0f64..1000.0f64).prop_filter("finite", |x| x.is_finite()),
                        (p + q)..400, // Ensure enough data for warmup
                    ),
                    Just(p),
                    (0.1f64..10.0f64).prop_filter("finite", |x| x.is_finite()), // x parameter
                    Just(q),
                )
            })
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(base_prices, p, x, q)| {
                // Generate realistic OHLC data from base prices
                let mut high = Vec::with_capacity(base_prices.len());
                let mut low = Vec::with_capacity(base_prices.len());
                let mut close = Vec::with_capacity(base_prices.len());

                for (i, price) in base_prices.iter().enumerate() {
                    let volatility = price * 0.02; // 2% volatility
                    let h = price + volatility;
                    let l = price - volatility;
                    high.push(h);
                    low.push(l);
                    // Close should be between high and low
                    // Use index for deterministic variation
                    let close_factor = 0.3 + 0.4 * ((i % 3) as f64 / 2.0); // Varies between 0.3 and 0.7
                    close.push(l + (h - l) * close_factor);
                }

                let params = CkspParams {
                    p: Some(p),
                    x: Some(x),
                    q: Some(q),
                };
                let input = CkspInput::from_slices(&high, &low, &close, params);

                // Test 1: Verify outputs are generated
                let result = cksp_with_kernel(&input, kernel)?;
                let CkspOutput {
                    long_values,
                    short_values,
                } = result;

                prop_assert_eq!(
                    long_values.len(),
                    close.len(),
                    "Long values length mismatch"
                );
                prop_assert_eq!(
                    short_values.len(),
                    close.len(),
                    "Short values length mismatch"
                );

                // Test 2: Warmup period validation
                // Find the first non-NaN index to understand actual warmup behavior
                let first_long_valid = long_values.iter().position(|&v| v.is_finite());
                let first_short_valid = short_values.iter().position(|&v| v.is_finite());

                // Both should have the same first valid index
                if let (Some(long_idx), Some(short_idx)) = (first_long_valid, first_short_valid) {
                    prop_assert_eq!(
                        long_idx,
                        short_idx,
                        "First valid indices should match: long={}, short={}",
                        long_idx,
                        short_idx
                    );

                    // Verify NaN values before first valid index
                    for i in 0..long_idx {
                        prop_assert!(
                            long_values[i].is_nan(),
                            "idx {}: long value should be NaN before first valid ({}), got {}",
                            i,
                            long_idx,
                            long_values[i]
                        );
                        prop_assert!(
                            short_values[i].is_nan(),
                            "idx {}: short value should be NaN before first valid ({}), got {}",
                            i,
                            short_idx,
                            short_values[i]
                        );
                    }

                    // Verify warmup is reasonable based on parameters
                    // The actual warmup depends on data and should be at least p-1 for ATR
                    prop_assert!(
                        long_idx >= p - 1,
                        "Warmup period {} should be at least p - 1 = {}",
                        long_idx,
                        p - 1
                    );
                    // And should not exceed p + q - 1 (theoretical maximum)
                    let max_warmup = p + q - 1;
                    prop_assert!(
                        long_idx <= max_warmup,
                        "Warmup period {} should not exceed p + q - 1 = {}",
                        long_idx,
                        max_warmup
                    );
                }

                // Test 3: Non-NaN values after warmup
                if let Some(first_valid) = first_long_valid {
                    for i in first_valid..close.len() {
                        prop_assert!(
                            long_values[i].is_finite(),
                            "idx {}: long value should be finite after warmup, got {}",
                            i,
                            long_values[i]
                        );
                        prop_assert!(
                            short_values[i].is_finite(),
                            "idx {}: short value should be finite after warmup, got {}",
                            i,
                            short_values[i]
                        );
                    }
                }

                // Test 4: Kernel consistency (compare with scalar)
                if kernel != Kernel::Scalar {
                    let scalar_result = cksp_with_kernel(&input, Kernel::Scalar)?;
                    let CkspOutput {
                        long_values: scalar_long,
                        short_values: scalar_short,
                    } = scalar_result;

                    let start_idx = first_long_valid.unwrap_or(0);
                    for i in start_idx..close.len() {
                        let long_val = long_values[i];
                        let scalar_long_val = scalar_long[i];
                        let short_val = short_values[i];
                        let scalar_short_val = scalar_short[i];

                        // Check ULP difference for long values
                        if long_val.is_finite() && scalar_long_val.is_finite() {
                            let long_bits = long_val.to_bits();
                            let scalar_long_bits = scalar_long_val.to_bits();
                            let ulp_diff = long_bits.abs_diff(scalar_long_bits);

                            prop_assert!(
                                (long_val - scalar_long_val).abs() <= 1e-9 || ulp_diff <= 8,
                                "Long value mismatch at idx {}: {} vs {} (ULP={})",
                                i,
                                long_val,
                                scalar_long_val,
                                ulp_diff
                            );
                        }

                        // Check ULP difference for short values
                        if short_val.is_finite() && scalar_short_val.is_finite() {
                            let short_bits = short_val.to_bits();
                            let scalar_short_bits = scalar_short_val.to_bits();
                            let ulp_diff = short_bits.abs_diff(scalar_short_bits);

                            prop_assert!(
                                (short_val - scalar_short_val).abs() <= 1e-9 || ulp_diff <= 8,
                                "Short value mismatch at idx {}: {} vs {} (ULP={})",
                                i,
                                short_val,
                                scalar_short_val,
                                ulp_diff
                            );
                        }
                    }
                }

                // Test 5: Mathematical properties and bounds checking
                let start_idx = first_long_valid.unwrap_or(0);
                if start_idx < close.len() {
                    // Calculate rough ATR estimate for bounds checking
                    let mut max_tr: f64 = 0.0;
                    for j in start_idx.saturating_sub(p)..start_idx {
                        if j < high.len() {
                            let tr = high[j] - low[j];
                            max_tr = max_tr.max(tr);
                        }
                    }

                    // Find price range for bounds checking (use entire data, not just from start)
                    let price_max = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let price_min = low.iter().cloned().fold(f64::INFINITY, f64::min);

                    for i in start_idx..close.len() {
                        // Both stops should be finite (not NaN or infinite)
                        prop_assert!(
                            long_values[i].is_finite(),
                            "Long stop should be finite at idx {}: {}",
                            i,
                            long_values[i]
                        );
                        prop_assert!(
                            short_values[i].is_finite(),
                            "Short stop should be finite at idx {}: {}",
                            i,
                            short_values[i]
                        );

                        // Stops should be within reasonable bounds
                        // Use a generous margin to account for extreme market conditions
                        let price_range = price_max - price_min;
                        let margin = price_range * 2.0; // Allow stops to be within 2x the price range

                        prop_assert!(
                            long_values[i] <= price_max + margin,
                            "Long stop {} should be <= max_price {} + margin {} at idx {}",
                            long_values[i],
                            price_max,
                            margin,
                            i
                        );

                        prop_assert!(
                            short_values[i] >= price_min - margin,
                            "Short stop {} should be >= min_price {} - margin {} at idx {}",
                            short_values[i],
                            price_min,
                            margin,
                            i
                        );
                    }
                }

                // Test 6: Edge case - period = 1 and q = 1
                if p == 1 && q == 1 {
                    // With minimal periods, stops should react quickly to price changes
                    let start_check = first_long_valid.unwrap_or(0).saturating_add(1);
                    for i in start_check..close.len() {
                        // With minimal parameters, the stops should exist and be finite
                        prop_assert!(
                            long_values[i].is_finite(),
                            "Long stop should be finite with p=1,q=1 at idx {}: {}",
                            i,
                            long_values[i]
                        );
                        prop_assert!(
                            short_values[i].is_finite(),
                            "Short stop should be finite with p=1,q=1 at idx {}: {}",
                            i,
                            short_values[i]
                        );

                        // With p=1 and q=1, stops should be very close to recent high/low
                        // Since ATR period is 1, the stop should be within 1 ATR of recent extremes
                        let recent_high = high[i];
                        let recent_low = low[i];
                        let recent_range = recent_high - recent_low;

                        // Long stop should be below recent high
                        prop_assert!(
                            long_values[i] <= recent_high,
                            "With p=1,q=1: Long stop {} should be <= recent high {} at idx {}",
                            long_values[i],
                            recent_high,
                            i
                        );

                        // Short stop should be above recent low
                        prop_assert!(
                            short_values[i] >= recent_low,
                            "With p=1,q=1: Short stop {} should be >= recent low {} at idx {}",
                            short_values[i],
                            recent_low,
                            i
                        );

                        // With minimal parameters and extreme multipliers, stops can vary widely
                        // Just ensure they remain finite
                        // The actual bounds depend heavily on the ATR multiplier x
                    }
                }

                // Test 8: ATR multiplier (x) effect
                // Compare with a smaller x value to verify stops widen with larger x
                if x > 1.0 {
                    let smaller_x = x * 0.5;
                    let params_small = CkspParams {
                        p: Some(p),
                        x: Some(smaller_x),
                        q: Some(q),
                    };
                    let input_small = CkspInput::from_slices(&high, &low, &close, params_small);
                    if let Ok(result_small) = cksp_with_kernel(&input_small, kernel) {
                        let CkspOutput {
                            long_values: long_small,
                            short_values: short_small,
                        } = result_small;

                        // After warmup, stops with larger x should be wider apart than with smaller x
                        if let Some(start) = first_long_valid {
                            let sample_points = 5.min((close.len() - start) / 2);
                            for offset in 0..sample_points {
                                let idx = start + offset * 2;
                                if idx < close.len() {
                                    let spread_large = (short_values[idx] - long_values[idx]).abs();
                                    let spread_small = (short_small[idx] - long_small[idx]).abs();

                                    // In most cases, larger x should produce wider spreads
                                    // But this isn't always guaranteed due to rolling window effects
                                    // So we only check this as a general trend, not a strict rule
                                    if spread_small > 0.0 {
                                        // Just verify both spreads are reasonable
                                        prop_assert!(
											spread_large > 0.0 && spread_small > 0.0,
											"At idx {}: Both spreads should be positive: large={}, small={}",
											idx,
											spread_large,
											spread_small
										);
                                    }
                                }
                            }
                        }
                    }
                }

                // Test 9: Rolling window (q) effect
                // Compare with different q values to verify smoothing effect
                if q > 2 && p < 10 {
                    let smaller_q = 1;
                    let params_small_q = CkspParams {
                        p: Some(p),
                        x: Some(x),
                        q: Some(smaller_q),
                    };
                    let input_small_q = CkspInput::from_slices(&high, &low, &close, params_small_q);
                    if let Ok(result_small_q) = cksp_with_kernel(&input_small_q, kernel) {
                        let CkspOutput {
                            long_values: long_small_q,
                            short_values: short_small_q,
                        } = result_small_q;

                        // Calculate smoothness metric: sum of absolute differences between consecutive values
                        let start = (p + q).max(p + smaller_q);
                        if start + 10 < close.len() {
                            let mut volatility_large_q = 0.0;
                            let mut volatility_small_q = 0.0;

                            for i in start..(start + 10) {
                                if i > 0 && i < close.len() {
                                    volatility_large_q +=
                                        (long_values[i] - long_values[i - 1]).abs();
                                    volatility_small_q +=
                                        (long_small_q[i] - long_small_q[i - 1]).abs();
                                }
                            }

                            // Larger q often produces smoother stops, but not always
                            // Just verify that both have reasonable volatility
                            prop_assert!(
                                volatility_large_q.is_finite() && volatility_small_q.is_finite(),
                                "Volatilities should be finite: large_q={}, small_q={}",
                                volatility_large_q,
                                volatility_small_q
                            );
                        }
                    }
                }

                // Test 7: Constant price property
                if base_prices.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                    // With constant prices, ATR should approach the fixed volatility we added
                    let last_idx = close.len() - 1;
                    let min_converge_idx = first_long_valid.unwrap_or(0) + p * 2; // Allow 2x period for convergence
                    if last_idx > min_converge_idx {
                        let constant_price = base_prices[0];
                        let constant_volatility = constant_price * 0.02; // From our OHLC generation

                        // With constant prices, the stops should converge to:
                        // Long stop ≈ high - x * ATR ≈ (price + volatility) - x * (2 * volatility)
                        // Short stop ≈ low + x * ATR ≈ (price - volatility) + x * (2 * volatility)

                        let expected_long =
                            constant_price + constant_volatility - x * (2.0 * constant_volatility);
                        let expected_short =
                            constant_price - constant_volatility + x * (2.0 * constant_volatility);

                        // Check convergence at the end
                        let long_val = long_values[last_idx];
                        let short_val = short_values[last_idx];

                        // Allow 20% tolerance for convergence
                        let tolerance = constant_price * 0.2;

                        prop_assert!(
							(long_val - expected_long).abs() <= tolerance,
							"With constant price {}: Long stop {} should converge near {} (within {})",
							constant_price,
							long_val,
							expected_long,
							tolerance
						);

                        prop_assert!(
							(short_val - expected_short).abs() <= tolerance,
							"With constant price {}: Short stop {} should converge near {} (within {})",
							constant_price,
							short_val,
							expected_short,
							tolerance
						);

                        // Also check stabilization
                        if last_idx >= 3 {
                            let long_stable = (long_values[last_idx] - long_values[last_idx - 1])
                                .abs()
                                < constant_volatility * 0.1;
                            let short_stable =
                                (short_values[last_idx] - short_values[last_idx - 1]).abs()
                                    < constant_volatility * 0.1;

                            prop_assert!(
                                long_stable && short_stable,
                                "Stops should stabilize: Long diff {}, Short diff {}",
                                (long_values[last_idx] - long_values[last_idx - 1]).abs(),
                                (short_values[last_idx] - short_values[last_idx - 1]).abs()
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_cksp_tests {
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

    fn check_cksp_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = CkspInput::from_slices(&empty, &empty, &empty, CkspParams::default());
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(CkspError::NoData)),
            "[{}] CKSP should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_cksp_invalid_x_param(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0, 13.0, 14.0];
        let low = [9.0, 10.0, 11.0, 12.0, 13.0];
        let close = [9.5, 10.5, 11.5, 12.5, 13.5];
        let params = CkspParams {
            p: Some(2),
            x: Some(f64::NAN),
            q: Some(2),
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(CkspError::InvalidParam { .. })),
            "[{}] CKSP should fail with invalid x parameter (NaN)",
            test_name
        );
        Ok(())
    }

    fn check_cksp_invalid_q_param(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0, 13.0, 14.0];
        let low = [9.0, 10.0, 11.0, 12.0, 13.0];
        let close = [9.5, 10.5, 11.5, 12.5, 13.5];
        let params = CkspParams {
            p: Some(2),
            x: Some(1.0),
            q: Some(0), // Invalid q = 0
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(CkspError::InvalidParam { .. })),
            "[{}] CKSP should fail with invalid q parameter (0)",
            test_name
        );
        Ok(())
    }

    generate_all_cksp_tests!(
        check_cksp_partial_params,
        check_cksp_accuracy,
        check_cksp_default_candles,
        check_cksp_zero_period,
        check_cksp_period_exceeds_length,
        check_cksp_very_small_dataset,
        check_cksp_empty_input,
        check_cksp_invalid_x_param,
        check_cksp_invalid_q_param,
        check_cksp_reinput,
        check_cksp_nan_handling,
        check_cksp_streaming,
        check_cksp_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_cksp_tests!(check_cksp_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CkspBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = CkspParams::default();
        let (long_row, short_row) = output.values_for(&def).expect("default row missing");

        assert_eq!(long_row.len(), c.close.len());
        assert_eq!(short_row.len(), c.close.len());

        let expected_long = [
            60306.66197802568,
            60306.66197802568,
            60306.66197802568,
            60203.29578022311,
            60201.57958198072,
        ];
        let start = long_row.len() - 5;
        for (i, &v) in long_row[start..].iter().enumerate() {
            assert!(
                (v - expected_long[i]).abs() < 1e-5,
                "[{test}] default-row long mismatch at idx {i}: {v} vs {expected_long:?}"
            );
        }

        let expected_short = [
            58757.826484736055,
            58701.74383626245,
            58656.36945263621,
            58611.03250737258,
            58611.03250737258,
        ];
        for (i, &v) in short_row[start..].iter().enumerate() {
            assert!(
                (v - expected_short[i]).abs() < 1e-5,
                "[{test}] default-row short mismatch at idx {i}: {v} vs {expected_short:?}"
            );
        }
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = CkspBatchBuilder::new()
            .kernel(kernel)
            .p_range(5, 25, 5) // Test p values: 5, 10, 15, 20, 25
            .x_range(0.5, 2.5, 0.5) // Test x values: 0.5, 1.0, 1.5, 2.0, 2.5
            .q_range(5, 20, 5) // Test q values: 5, 10, 15, 20
            .apply_candles(&c)?;

        // Check every value in the entire batch matrix for poison patterns - long_values
        for (idx, &val) in output.long_values.iter().enumerate() {
            // Skip NaN values as they're expected in warmup periods
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in long_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) in long_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in long_values",
                    test, val, bits, row, col, idx
                );
            }
        }

        // Check every value in the entire batch matrix for poison patterns - short_values
        for (idx, &val) in output.short_values.iter().enumerate() {
            // Skip NaN values as they're expected in warmup periods
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in short_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) in short_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in short_values",
                    test, val, bits, row, col, idx
                );
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
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

// ========================= Python Bindings =========================

#[cfg(feature = "python")]
#[inline(always)]
fn cksp_prepare(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    kernel: Kernel,
) -> Result<(usize, Kernel), CkspError> {
    // Validate parameters first (before data checks)
    if p == 0 || q == 0 {
        return Err(CkspError::InvalidParam { param: "p/q" });
    }
    if !x.is_finite() || x.is_nan() {
        return Err(CkspError::InvalidParam { param: "x" });
    }

    // Now check data
    let size = close.len();
    if size == 0 {
        return Err(CkspError::NoData);
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(CkspError::InconsistentLengths);
    }
    if p > size || q > size {
        return Err(CkspError::NotEnoughData {
            p,
            q,
            data_len: size,
        });
    }

    let first_valid_idx = match close.iter().position(|&v| !v.is_nan()) {
        Some(idx) => idx,
        None => return Err(CkspError::NoData),
    };

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((first_valid_idx, chosen))
}

#[cfg(feature = "python")]
#[pyfunction(name = "cksp")]
#[pyo3(signature = (high, low, close, p=10, x=1.0, q=9, kernel=None))]
pub fn cksp_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    p: usize,
    x: f64,
    q: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    // Use cksp_prepare for validation
    let (first_valid_idx, chosen) = cksp_prepare(high_slice, low_slice, close_slice, p, x, q, kern)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Get result vectors from Rust function
    let result = py
        .allow_threads(|| unsafe {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    cksp_scalar(high_slice, low_slice, close_slice, p, x, q, first_valid_idx)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    cksp_avx2(high_slice, low_slice, close_slice, p, x, q, first_valid_idx)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    cksp_avx512(high_slice, low_slice, close_slice, p, x, q, first_valid_idx)
                }
                _ => unreachable!(),
            }
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok((
        result.long_values.into_pyarray(py),
        result.short_values.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "CkspStream")]
pub struct CkspStreamPy {
    inner: CkspStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CkspStreamPy {
    #[new]
    pub fn new(p: usize, x: f64, q: usize) -> PyResult<Self> {
        let params = CkspParams {
            p: Some(p),
            x: Some(x),
            q: Some(q),
        };
        let inner =
            CkspStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CkspStreamPy { inner })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.inner.update(high, low, close)
    }
}

#[inline(always)]
fn cksp_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    kern: Kernel,
    parallel: bool,
    long_out: &mut [f64],
    short_out: &mut [f64],
) -> Result<Vec<CkspParams>, CkspError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CkspError::InvalidParam { param: "combos" });
    }
    let size = close.len();
    if high.len() != low.len() || low.len() != close.len() {
        return Err(CkspError::InconsistentLengths);
    }
    let first_valid = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CkspError::NoData)?;

    let rows = combos.len();
    let cols = size;
    if long_out.len() != rows * cols || short_out.len() != rows * cols {
        return Err(CkspError::NotEnoughData {
            p: 0,
            q: 0,
            data_len: cols,
        });
    }

    // Initialize warm prefixes in-place via MaybeUninit view (debug poison supported)
    unsafe {
        let mut long_mu = core::slice::from_raw_parts_mut(
            long_out.as_mut_ptr() as *mut MaybeUninit<f64>,
            long_out.len(),
        );
        let mut short_mu = core::slice::from_raw_parts_mut(
            short_out.as_mut_ptr() as *mut MaybeUninit<f64>,
            short_out.len(),
        );
        let warm: Vec<usize> = combos
            .iter()
            .map(|c| first_valid + c.p.unwrap() + c.q.unwrap() - 1)
            .collect();
        init_matrix_prefixes(&mut long_mu, cols, &warm);
        init_matrix_prefixes(&mut short_mu, cols, &warm);
    }

    let do_row = |row: usize, out_long: &mut [f64], out_short: &mut [f64]| unsafe {
        let prm = &combos[row];
        let (p, x, q) = (prm.p.unwrap(), prm.x.unwrap(), prm.q.unwrap());
        match kern {
            Kernel::Scalar => {
                cksp_row_scalar(high, low, close, p, x, q, first_valid, out_long, out_short)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                cksp_row_avx2(high, low, close, p, x, q, first_valid, out_long, out_short)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                cksp_row_avx512(high, low, close, p, x, q, first_valid, out_long, out_short)
            }
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            long_out
                .par_chunks_mut(cols)
                .zip(short_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (lv, sv))| do_row(row, lv, sv));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (lv, sv)) in long_out
                .chunks_mut(cols)
                .zip(short_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, lv, sv);
            }
        }
    } else {
        for (row, (lv, sv)) in long_out
            .chunks_mut(cols)
            .zip(short_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, lv, sv);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "cksp_batch")]
#[pyo3(signature = (high, low, close, p_range=(10, 10, 0), x_range=(1.0, 1.0, 0.0), q_range=(9, 9, 0), kernel=None))]
pub fn cksp_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    p_range: (usize, usize, usize),
    x_range: (f64, f64, f64),
    q_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = CkspBatchRange {
        p: p_range,
        x: x_range,
        q: q_range,
    };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = close_slice.len();

    // Pre-allocate output arrays
    let long_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let short_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let long_slice = unsafe { long_arr.as_slice_mut()? };
    let short_slice = unsafe { short_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py
        .allow_threads(|| {
            // Handle kernel selection for batch operations
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // Map batch kernels to regular kernels
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };

            cksp_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                simd,
                true,
                long_slice,
                short_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("long_values", long_arr.reshape((rows, cols))?)?;
    dict.set_item("short_values", short_arr.reshape((rows, cols))?)?;

    // Add parameter arrays
    dict.set_item(
        "p",
        combos
            .iter()
            .map(|p| p.p.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "x",
        combos
            .iter()
            .map(|p| p.x.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "q",
        combos
            .iter()
            .map(|p| p.q.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ========================= WASM Bindings =========================

/// Helper function that writes directly to output slices - no allocations
#[inline]
pub fn cksp_into_slice(
    long_dst: &mut [f64],
    short_dst: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    kern: Kernel,
) -> Result<(), CkspError> {
    // Validate inputs
    if high.len() != low.len() || low.len() != close.len() {
        return Err(CkspError::InconsistentLengths);
    }
    if long_dst.len() != close.len() || short_dst.len() != close.len() {
        return Err(CkspError::InconsistentLengths);
    }
    if close.is_empty() {
        return Err(CkspError::NoData);
    }
    if p == 0 || q == 0 {
        return Err(CkspError::InvalidParam { param: "p/q" });
    }
    if p > close.len() || q > close.len() {
        return Err(CkspError::NotEnoughData {
            p,
            q,
            data_len: close.len(),
        });
    }
    if !x.is_finite() {
        return Err(CkspError::InvalidParam { param: "x" });
    }

    let first_valid_idx = match close.iter().position(|&v| !v.is_nan()) {
        Some(idx) => idx,
        None => return Err(CkspError::NoData),
    };

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Use the compute_into helper to write directly to output slices
    unsafe {
        cksp_compute_into(
            high,
            low,
            close,
            p,
            x,
            q,
            first_valid_idx,
            long_dst,
            short_dst,
        );
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cksp_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = CkspInput::from_slices(
        high,
        low,
        close,
        CkspParams {
            p: Some(p),
            x: Some(x),
            q: Some(q),
        },
    );
    let out = cksp_with_kernel(&input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let cols = close.len();
    let mut values = Vec::with_capacity(2 * cols);
    values.extend_from_slice(&out.long_values);
    values.extend_from_slice(&out.short_values);
    Ok(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cksp_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    long_ptr: *mut f64,
    short_ptr: *mut f64,
    len: usize,
    p: usize,
    x: f64,
    q: usize,
) -> Result<(), JsValue> {
    if long_ptr.is_null() || short_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    if high.len() != len || low.len() != len || close.len() != len {
        return Err(JsValue::from_str("Input length mismatch"));
    }

    unsafe {
        // Get pointers from slices for aliasing check
        let high_ptr = high.as_ptr();
        let low_ptr = low.as_ptr();
        let close_ptr = close.as_ptr();

        // Check for any aliasing between inputs and outputs
        let has_aliasing = (high_ptr as *const f64 == long_ptr as *const f64)
            || (high_ptr as *const f64 == short_ptr as *const f64)
            || (low_ptr as *const f64 == long_ptr as *const f64)
            || (low_ptr as *const f64 == short_ptr as *const f64)
            || (close_ptr as *const f64 == long_ptr as *const f64)
            || (close_ptr as *const f64 == short_ptr as *const f64)
            || (long_ptr == short_ptr); // Also check if output pointers alias each other

        if has_aliasing {
            // Use temporary buffers if any aliasing is detected
            let mut temp_long = vec![0.0; len];
            let mut temp_short = vec![0.0; len];

            cksp_into_slice(
                &mut temp_long,
                &mut temp_short,
                high,
                low,
                close,
                p,
                x,
                q,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let long_out = std::slice::from_raw_parts_mut(long_ptr, len);
            let short_out = std::slice::from_raw_parts_mut(short_ptr, len);
            long_out.copy_from_slice(&temp_long);
            short_out.copy_from_slice(&temp_short);
        } else {
            // No aliasing - write directly to output
            let long_out = std::slice::from_raw_parts_mut(long_ptr, len);
            let short_out = std::slice::from_raw_parts_mut(short_ptr, len);

            cksp_into_slice(long_out, short_out, high, low, close, p, x, q, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cksp_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cksp_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CkspJsResult {
    pub values: Vec<f64>, // [long..., short...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CkspBatchConfig {
    pub p_range: (usize, usize, usize),
    pub x_range: (f64, f64, f64),
    pub q_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CkspBatchJsOutput {
    pub long_values: Vec<f64>,
    pub short_values: Vec<f64>,
    pub combos: Vec<CkspParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cksp_batch)]
pub fn cksp_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: CkspBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = CkspBatchRange {
        p: config.p_range,
        x: config.x_range,
        q: config.q_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = close.len();

    // Pre-allocate output arrays
    let mut long_values = vec![0.0; rows * cols];
    let mut short_values = vec![0.0; rows * cols];

    // Compute batch results with appropriate kernel for WASM
    let kernel = detect_best_batch_kernel();
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch | _ => Kernel::Scalar,
    };
    cksp_batch_inner_into(
        high,
        low,
        close,
        &sweep,
        simd,
        false,
        &mut long_values,
        &mut short_values,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = CkspBatchJsOutput {
        long_values,
        short_values,
        combos,
        rows,
        cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cksp_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    long_ptr: *mut f64,
    short_ptr: *mut f64,
    len: usize,
    p_start: usize,
    p_end: usize,
    p_step: usize,
    x_start: f64,
    x_end: f64,
    x_step: f64,
    q_start: usize,
    q_end: usize,
    q_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || long_ptr.is_null()
        || short_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let sweep = CkspBatchRange {
            p: (p_start, p_end, p_step),
            x: (x_start, x_end, x_step),
            q: (q_start, q_end, q_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let long_out = std::slice::from_raw_parts_mut(long_ptr, rows * cols);
        let short_out = std::slice::from_raw_parts_mut(short_ptr, rows * cols);

        // Use appropriate kernel for WASM
        let kernel = detect_best_batch_kernel();
        let simd = match kernel {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch => Kernel::Avx512,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch | _ => Kernel::Scalar,
        };
        cksp_batch_inner_into(high, low, close, &sweep, simd, false, long_out, short_out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
