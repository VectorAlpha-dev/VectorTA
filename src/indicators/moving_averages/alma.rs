#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

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
use std::alloc::{alloc, dealloc, Layout};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for AlmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AlmaData::Slice(slice) => slice,
            AlmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AlmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AlmaParams {
    pub period: Option<usize>,
    pub offset: Option<f64>,
    pub sigma: Option<f64>,
}

impl Default for AlmaParams {
    fn default() -> Self {
        Self {
            period: Some(9),
            offset: Some(0.85),
            sigma: Some(6.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlmaInput<'a> {
    pub data: AlmaData<'a>,
    pub params: AlmaParams,
}

impl<'a> AlmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: AlmaParams) -> Self {
        Self {
            data: AlmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: AlmaParams) -> Self {
        Self {
            data: AlmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", AlmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
    #[inline]
    pub fn get_offset(&self) -> f64 {
        self.params.offset.unwrap_or(0.85)
    }
    #[inline]
    pub fn get_sigma(&self) -> f64 {
        self.params.sigma.unwrap_or(6.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AlmaBuilder {
    period: Option<usize>,
    offset: Option<f64>,
    sigma: Option<f64>,
    kernel: Kernel,
}

impl Default for AlmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            offset: None,
            sigma: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AlmaBuilder {
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
    pub fn offset(mut self, x: f64) -> Self {
        self.offset = Some(x);
        self
    }
    #[inline(always)]
    pub fn sigma(mut self, s: f64) -> Self {
        self.sigma = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AlmaOutput, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma: self.sigma,
        };
        let i = AlmaInput::from_candles(c, "close", p);
        alma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<AlmaOutput, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma: self.sigma,
        };
        let i = AlmaInput::from_slice(d, p);
        alma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AlmaStream, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma: self.sigma,
        };
        AlmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AlmaError {
    #[error("alma: Input data slice is empty.")]
    EmptyInputData,
    #[error("alma: All values are NaN.")]
    AllValuesNaN,

    #[error("alma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("alma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("alma: Invalid sigma: {sigma}")]
    InvalidSigma { sigma: f64 },

    #[error("alma: Invalid offset: {offset}")]
    InvalidOffset { offset: f64 },
}

#[inline]
pub fn alma(input: &AlmaInput) -> Result<AlmaOutput, AlmaError> {
    alma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn alma_compute_into(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    inv_n: f64,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                alma_scalar(data, weights, period, first, inv_n, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => alma_avx2(data, weights, period, first, inv_n, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_avx512(data, weights, period, first, inv_n, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                alma_scalar(data, weights, period, first, inv_n, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
fn alma_prepare<'a>(
    input: &'a AlmaInput,
    kernel: Kernel,
) -> Result<
    (
        /*data*/ &'a [f64],
        /*weights*/ AVec<f64>,
        /*period*/ usize,
        /*first*/ usize,
        /*inv_norm*/ f64,
        /*chosen*/ Kernel,
    ),
    AlmaError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(AlmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlmaError::AllValuesNaN)?;
    let period = input.get_period();
    let offset = input.get_offset();
    let sigma = input.get_sigma();

    // ---- same guards as before ----
    if period == 0 || period > len {
        return Err(AlmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(AlmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if sigma <= 0.0 {
        return Err(AlmaError::InvalidSigma { sigma });
    }
    if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
        return Err(AlmaError::InvalidOffset { offset });
    }

    // ---- build weights once ----
    let m = offset * (period - 1) as f64;
    let s = period as f64 / sigma;
    let s2 = 2.0 * s * s;

    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period);
    weights.resize(period, 0.0);
    let mut norm = 0.0;
    for i in 0..period {
        let w = (-(i as f64 - m).powi(2) / s2).exp();
        weights[i] = w;
        norm += w;
    }
    let inv_norm = 1.0 / norm;

    // kernel auto-detection only once
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, weights, period, first, inv_norm, chosen))
}

pub fn alma_with_kernel(input: &AlmaInput, kernel: Kernel) -> Result<AlmaOutput, AlmaError> {
    let (data, weights, period, first, inv_n, chosen) = alma_prepare(input, kernel)?;

    // identical warm-up allocation utility you already have
    let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);

    alma_compute_into(data, &weights, period, first, inv_n, chosen, &mut out);

    Ok(AlmaOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn hsum_pd_zmm(v: __m512d) -> f64 {
    let hi256 = _mm512_extractf64x4_pd(v, 1);
    let lo256 = _mm512_castpd512_pd256(v);
    let sum256 = _mm256_add_pd(hi256, lo256);
    let hi128 = _mm256_extractf128_pd(sum256, 1);
    let lo128 = _mm256_castpd256_pd128(sum256);
    let sum128 = _mm_add_pd(hi128, lo128);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, hi64);
    _mm_cvtsd_f64(sum64)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub fn alma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { alma_avx512_short(data, weights, period, first_valid, inv_norm, out) }
    } else {
        unsafe { alma_avx512_long(data, weights, period, first_valid, inv_norm, out) }
    }
}

#[inline]
pub fn alma_scalar(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    assert_eq!(weights.len(), period, "weights.len() must equal `period`");
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    let p4 = period & !3;

    for i in (first_val + period - 1)..data.len() {
        let start = i + 1 - period;
        let window = &data[start..start + period];

        let mut sum = 0.0;
        for (d4, w4) in window[..p4]
            .chunks_exact(4)
            .zip(weights[..p4].chunks_exact(4))
        {
            sum += d4[0] * w4[0] + d4[1] * w4[1] + d4[2] * w4[2] + d4[3] * w4[3];
        }

        for (d, w) in window[p4..].iter().zip(&weights[p4..]) {
            sum += d * w;
        }

        out[i] = sum * inv_norm;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn alma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    const STEP: usize = 4;
    let chunks = period / STEP;
    let tail = period % STEP;

    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm256_setzero_pd();

        for blk in 0..chunks {
            let idx = blk * STEP;
            let w = _mm256_loadu_pd(weights.as_ptr().add(idx));
            let d = _mm256_loadu_pd(data.as_ptr().add(start + idx));
            acc = _mm256_fmadd_pd(d, w, acc);
        }

        if tail != 0 {
            let w_tail = _mm256_maskload_pd(weights.as_ptr().add(chunks * STEP), tail_mask);
            let d_tail = _mm256_maskload_pd(data.as_ptr().add(start + chunks * STEP), tail_mask);
            acc = _mm256_fmadd_pd(d_tail, w_tail, acc);
        }

        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let sum2 = _mm_add_pd(hi, lo);
        let sum1 = _mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2));
        let sum = _mm_cvtsd_f64(sum1);

        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    debug_assert!(period >= 1);
    debug_assert!(data.len() == out.len());
    debug_assert!(weights.len() >= period);

    const STEP: usize = 8;
    let chunks = period / STEP; // 0,1,2,3,4 (≤ 4 because period ≤ 32)
    let tail_len = period % STEP;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    // ---------- fast path for period ≤ 8 ---------------------------------------
    if chunks == 0 {
        let w_vec = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr());
        for i in (first_valid + period - 1)..data.len() {
            let start = i + 1 - period;
            let d_vec = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start));
            let sum = hsum_pd_zmm(_mm512_mul_pd(d_vec, w_vec)) * inv_norm;
            *out.get_unchecked_mut(i) = sum;
        }
        return;
    }

    // ---------- general short‑kernel (1 ≤ chunks ≤ 4) ---------------------------
    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        // handle all full 8‑wide blocks
        for blk in 0..chunks {
            let w = _mm512_loadu_pd(weights.as_ptr().add(blk * STEP));
            let d = _mm512_loadu_pd(data.as_ptr().add(start + blk * STEP));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        // optional tail
        if tail_len != 0 {
            let w_tail = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(chunks * STEP));
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + chunks * STEP));
            acc = _mm512_fmadd_pd(d_tail, w_tail, acc);
        }

        *out.get_unchecked_mut(i) = hsum_pd_zmm(acc) * inv_norm;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn alma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let n_chunks = period / STEP;
    let tail_len = period % STEP;
    let paired = n_chunks & !3;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    debug_assert!(period >= 1 && n_chunks > 0);
    debug_assert_eq!(data.len(), out.len());
    debug_assert!(weights.len() >= period);

    let mut wregs: Vec<__m512d> = Vec::with_capacity(n_chunks + (tail_len != 0) as usize);

    for blk in 0..n_chunks {
        wregs.push(_mm512_load_pd(weights.as_ptr().add(blk * STEP)));
    }
    let w_tail = if tail_len != 0 {
        let wt = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(n_chunks * STEP));
        wregs.push(wt);
        Some(wt)
    } else {
        None
    };

    let mut data_ptr = data.as_ptr().add(first_valid);
    let stop_ptr = data.as_ptr().add(data.len());

    let mut dst_ptr = out.as_mut_ptr().add(first_valid + period - 1);

    if tail_len == 0 {
        while data_ptr.add(period) <= stop_ptr {
            let mut s0 = _mm512_setzero_pd();
            let mut s1 = _mm512_setzero_pd();
            let mut s2 = _mm512_setzero_pd();
            let mut s3 = _mm512_setzero_pd();

            for blk in (0..paired).step_by(4) {
                _mm_prefetch(data_ptr.add((blk + 8) * STEP) as *const i8, _MM_HINT_T0);

                let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
                let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
                let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
                let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));

                s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
                s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
                s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
                s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);
            }

            for blk in paired..n_chunks {
                let d = _mm512_loadu_pd(data_ptr.add(blk * STEP));
                s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(blk), s0);
            }

            // horizontal reduction (hand-coded)
            let tot = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
            *dst_ptr = hsum_pd_zmm(tot) * inv_norm;

            data_ptr = data_ptr.add(1);
            dst_ptr = dst_ptr.add(1);
        }
    } else {
        let wt = w_tail.expect("tail_len != 0 but w_tail missing");

        while data_ptr.add(period) <= stop_ptr {
            let mut s0 = _mm512_setzero_pd();
            let mut s1 = _mm512_setzero_pd();
            let mut s2 = _mm512_setzero_pd();
            let mut s3 = _mm512_setzero_pd();

            for blk in (0..paired).step_by(4) {
                _mm_prefetch(data_ptr.add((blk + 8) * STEP) as *const i8, _MM_HINT_T0);

                let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
                let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
                let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
                let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));

                s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
                s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
                s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
                s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);
            }

            for blk in paired..n_chunks {
                let d = _mm512_loadu_pd(data_ptr.add(blk * STEP));
                s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(blk), s0);
            }

            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data_ptr.add(n_chunks * STEP));
            s0 = _mm512_fmadd_pd(d_tail, wt, s0);

            let tot = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
            *dst_ptr = hsum_pd_zmm(tot) * inv_norm;

            data_ptr = data_ptr.add(1);
            dst_ptr = dst_ptr.add(1);
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlmaStream {
    period: usize,
    weights: Vec<f64>,
    inv_norm: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl AlmaStream {
    pub fn try_new(params: AlmaParams) -> Result<Self, AlmaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(AlmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let offset = params.offset.unwrap_or(0.85);
        if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
            return Err(AlmaError::InvalidOffset { offset });
        }
        let sigma = params.sigma.unwrap_or(6.0);
        if sigma <= 0.0 {
            return Err(AlmaError::InvalidSigma { sigma });
        }

        let m = offset * (period - 1) as f64;
        let s = period as f64 / sigma;
        let s2 = 2.0 * s * s;

        let mut weights = Vec::with_capacity(period);
        let mut norm = 0.0;
        for i in 0..period {
            let diff = i as f64 - m;
            let w = (-(diff * diff) / s2).exp();
            weights.push(w);
            norm += w;
        }
        let inv_norm = 1.0 / norm;

        Ok(Self {
            period,
            weights,
            inv_norm,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(self.dot_ring())
    }

    #[inline(always)]
    fn dot_ring(&self) -> f64 {
        let mut sum = 0.0;
        let mut idx = self.head;
        for &w in &self.weights {
            sum += w * self.buffer[idx];
            idx = (idx + 1) % self.period;
        }
        sum * self.inv_norm
    }
}

#[derive(Clone, Debug)]
pub struct AlmaBatchRange {
    pub period: (usize, usize, usize),
    pub offset: (f64, f64, f64),
    pub sigma: (f64, f64, f64),
}

impl Default for AlmaBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
            offset: (0.85, 0.85, 0.0),
            sigma: (6.0, 6.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AlmaBatchBuilder {
    range: AlmaBatchRange,
    kernel: Kernel,
}

impl AlmaBatchBuilder {
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

    #[inline]
    pub fn offset_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.offset = (start, end, step);
        self
    }
    #[inline]
    pub fn offset_static(mut self, x: f64) -> Self {
        self.range.offset = (x, x, 0.0);
        self
    }

    #[inline]
    pub fn sigma_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.sigma = (start, end, step);
        self
    }
    #[inline]
    pub fn sigma_static(mut self, s: f64) -> Self {
        self.range.sigma = (s, s, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<AlmaBatchOutput, AlmaError> {
        alma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<AlmaBatchOutput, AlmaError> {
        AlmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<AlmaBatchOutput, AlmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<AlmaBatchOutput, AlmaError> {
        AlmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn alma_batch_with_kernel(
    data: &[f64],
    sweep: &AlmaBatchRange,
    k: Kernel,
) -> Result<AlmaBatchOutput, AlmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(AlmaError::InvalidPeriod {
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
    alma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AlmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AlmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AlmaBatchOutput {
    pub fn row_for_params(&self, p: &AlmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(9) == p.period.unwrap_or(9)
                && (c.offset.unwrap_or(0.85) - p.offset.unwrap_or(0.85)).abs() < 1e-12
                && (c.sigma.unwrap_or(6.0) - p.sigma.unwrap_or(6.0)).abs() < 1e-12
        })
    }

    pub fn values_for(&self, p: &AlmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AlmaBatchRange) -> Vec<AlmaParams> {
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

    let periods = axis_usize(r.period);
    let offsets = axis_f64(r.offset);
    let sigmas = axis_f64(r.sigma);

    let mut out = Vec::with_capacity(periods.len() * offsets.len() * sigmas.len());
    for &p in &periods {
        for &o in &offsets {
            for &s in &sigmas {
                out.push(AlmaParams {
                    period: Some(p),
                    offset: Some(o),
                    sigma: Some(s),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn alma_batch_slice(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
) -> Result<AlmaBatchOutput, AlmaError> {
    alma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn alma_batch_par_slice(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
) -> Result<AlmaBatchOutput, AlmaError> {
    alma_batch_inner(data, sweep, kern, true)
}

#[inline]
fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

#[inline(always)]
fn alma_batch_inner(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AlmaBatchOutput, AlmaError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    let rows = combos.len();
    
    if cols == 0 {
        return Err(AlmaError::AllValuesNaN);
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c|                                 //  first + period - 1
             data.iter()
                 .position(|x| !x.is_nan())
                 .unwrap_or(0) + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm); // ──┘

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);          // keep capacity
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
        )
    };

    alma_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(buf_guard.as_mut_ptr() as *mut f64,
                            buf_guard.len(),
                            buf_guard.capacity())
    };
    
    Ok(AlmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn alma_batch_inner_into(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AlmaParams>, AlmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AlmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlmaError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| round_up8(c.period.unwrap()))
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(AlmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut inv_norms = vec![0.0; rows];

    let cap = rows * max_p;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);
    let flat_slice = flat_w.as_mut_slice();

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let offset = prm.offset.unwrap();
        let sigma = prm.sigma.unwrap();

        if sigma <= 0.0 {
            return Err(AlmaError::InvalidSigma { sigma });
        }
        if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
            return Err(AlmaError::InvalidOffset { offset });
        }

        let m = offset * (period - 1) as f64;
        let s = period as f64 / sigma;
        let s2 = 2.0 * s * s;

        let mut norm = 0.0;
        for i in 0..period {
            let w = (-(i as f64 - m).powi(2) / s2).exp();
            flat_w[row * max_p + i] = w;
            norm += w;
        }
        inv_norms[row] = 1.0 / norm;
    }


    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let inv_n = *inv_norms.get_unchecked(row);

        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                alma_row_scalar(data, first, period, max_p, w_ptr, inv_n, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                alma_row_avx2(data, first, period, max_p, w_ptr, inv_n, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_row_avx512(data, first, period, max_p, w_ptr, inv_n, dst)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_row_scalar(data, first, period, max_p, w_ptr, inv_n, dst)
            }
            Kernel::Auto => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn alma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let p4 = period & !3;
    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for k in (0..p4).step_by(4) {
            let w = std::slice::from_raw_parts(w_ptr.add(k), 4);
            let d = &data[start + k..start + k + 4];
            sum += d[0] * w[0] + d[1] * w[1] + d[2] * w[2] + d[3] * w[3];
        }
        for k in p4..period {
            sum += *data.get_unchecked(start + k) * *w_ptr.add(k);
        }
        out[i] = sum * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn alma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 4;
    let vec_blocks = period / STEP;
    let tail = period % STEP;
    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm256_setzero_pd();

        for blk in 0..vec_blocks {
            let d = _mm256_loadu_pd(data.as_ptr().add(start + blk * STEP));
            let w = _mm256_loadu_pd(w_ptr.add(blk * STEP));
            acc = _mm256_fmadd_pd(d, w, acc);
        }

        if tail != 0 {
            let d = _mm256_maskload_pd(data.as_ptr().add(start + vec_blocks * STEP), tail_mask);
            let w = _mm256_maskload_pd(w_ptr.add(vec_blocks * STEP), tail_mask);
            acc = _mm256_fmadd_pd(d, w, acc);
        }

        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let s2 = _mm_add_pd(hi, lo);
        let s1 = _mm_add_pd(s2, _mm_unpackhi_pd(s2, s2));
        out[i] = _mm_cvtsd_f64(s1) * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma,avx512dq")]
pub unsafe fn alma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        alma_row_avx512_short(data, first, period, stride, w_ptr, inv_n, out);
    } else {
        alma_row_avx512_long(data, first, period, stride, w_ptr, inv_n, out);
    }

    _mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    debug_assert!(period <= 32);
    const STEP: usize = 8;

    let chunks = period / STEP;
    let tail_len = period % STEP;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    if chunks == 0 {
        let w_tail = _mm512_maskz_loadu_pd(tail_mask, w_ptr);
        for i in (first + period - 1)..data.len() {
            let start = i + 1 - period;
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start));
            let res = hsum_pd_zmm(_mm512_mul_pd(d_tail, w_tail)) * inv_n;
            _mm_stream_sd(out.get_unchecked_mut(i) as *mut f64, _mm_set_sd(res));
        }
        return;
    }

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        for blk in 0..chunks {
            let w = _mm512_loadu_pd(w_ptr.add(blk * STEP));
            let d = _mm512_loadu_pd(data.as_ptr().add(start + blk * STEP));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        if tail_len != 0 {
            let w_tail = _mm512_maskz_loadu_pd(tail_mask, w_ptr.add(chunks * STEP));
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + chunks * STEP));
            acc = _mm512_fmadd_pd(d_tail, w_tail, acc);
        }

        let res = hsum_pd_zmm(acc) * inv_n;
        _mm_stream_sd(out.get_unchecked_mut(i) as *mut f64, _mm_set_sd(res));
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn alma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let n_chunks = period / STEP;
    let tail_len = period % STEP;
    let tmask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    const MAX_CHUNKS: usize = 512;
    debug_assert!(n_chunks + (tail_len != 0) as usize <= MAX_CHUNKS);

    let mut wregs: [core::mem::MaybeUninit<__m512d>; MAX_CHUNKS] =
        core::mem::MaybeUninit::uninit().assume_init();

    for blk in 0..n_chunks {
        wregs[blk]
            .as_mut_ptr()
            .write(_mm512_loadu_pd(w_ptr.add(blk * STEP)));
    }
    if tail_len != 0 {
        wregs[n_chunks]
            .as_mut_ptr()
            .write(_mm512_maskz_loadu_pd(tmask, w_ptr.add(n_chunks * STEP)));
    }

    let wregs: &[__m512d] = core::slice::from_raw_parts(
        wregs.as_ptr() as *const __m512d,
        n_chunks + (tail_len != 0) as usize,
    );

    if tail_len == 0 {
        long_kernel_no_tail(data, first, n_chunks, wregs, inv_n, out);
    } else {
        long_kernel_with_tail(data, first, n_chunks, tail_len, tmask, wregs, inv_n, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn long_kernel_no_tail(
    data: &[f64],
    first: usize,
    n_chunks: usize,
    wregs: &[__m512d],
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let paired = n_chunks & !3;

    let mut data_ptr = data.as_ptr().add(first);
    let stop_ptr = data.as_ptr().add(data.len());
    let mut dst_ptr = out.as_mut_ptr().add(first + n_chunks * STEP - 1);

    while data_ptr < stop_ptr {
        let mut s0 = _mm512_setzero_pd();
        let mut s1 = _mm512_setzero_pd();
        let mut s2 = _mm512_setzero_pd();
        let mut s3 = _mm512_setzero_pd();

        let mut blk = 0;
        while blk < paired {
            _mm_prefetch(data_ptr.add((blk + 16) * STEP) as *const i8, _MM_HINT_T0);

            let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
            let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));

            s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
            s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
            s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
            s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);

            blk += 4;
        }

        for r in blk..n_chunks {
            let d = _mm512_loadu_pd(data_ptr.add(r * STEP));
            s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(r), s0);
        }

        let sum = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
        let res = hsum_pd_zmm(sum) * inv_n;

        _mm_stream_sd(dst_ptr as *mut f64, _mm_set_sd(res));

        data_ptr = data_ptr.add(1);
        dst_ptr = dst_ptr.add(1);
        if data_ptr.add(n_chunks * STEP) > stop_ptr {
            break;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn long_kernel_with_tail(
    data: &[f64],
    first: usize,
    n_chunks: usize,
    tail_len: usize,
    tmask: __mmask8,
    wregs: &[__m512d],
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let paired = n_chunks & !3;

    let w_tail = *wregs.get_unchecked(n_chunks);

    let mut data_ptr = data.as_ptr().add(first);
    let stop_ptr = data.as_ptr().add(data.len());
    let mut dst_ptr = out.as_mut_ptr().add(first + n_chunks * STEP + tail_len - 1);

    while data_ptr < stop_ptr {
        let mut s0 = _mm512_setzero_pd();
        let mut s1 = _mm512_setzero_pd();
        let mut s2 = _mm512_setzero_pd();
        let mut s3 = _mm512_setzero_pd();

        let mut blk = 0;
        while blk < paired {
            _mm_prefetch(data_ptr.add((blk + 16) * STEP) as *const i8, _MM_HINT_T0);

            let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
            let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));

            s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
            s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
            s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
            s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);

            blk += 4;
        }

        for r in blk..n_chunks {
            let d = _mm512_loadu_pd(data_ptr.add(r * STEP));
            s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(r), s0);
        }

        let d_tail = _mm512_maskz_loadu_pd(tmask, data_ptr.add(n_chunks * STEP));
        s0 = _mm512_fmadd_pd(d_tail, w_tail, s0);

        let sum = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
        let res = hsum_pd_zmm(sum) * inv_n;

        _mm_stream_sd(dst_ptr as *mut f64, _mm_set_sd(res));

        data_ptr = data_ptr.add(1);
        dst_ptr = dst_ptr.add(1);
        if data_ptr.add(n_chunks * STEP + tail_len) > stop_ptr {
            break;
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_alma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = AlmaParams {
            period: None,
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_candles(&candles, "close", default_params);
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_alma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::from_candles(&candles, "close", AlmaParams::default());
        let result = alma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59286.72216704,
            59273.53428138,
            59204.37290721,
            59155.93381742,
            59026.92526112,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] ALMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_alma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::with_default_candles(&candles);
        match input.data {
            AlmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected AlmaData::Candles"),
        }
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_alma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(0),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&input_data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_alma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(10),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&data_small, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_alma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&single_point, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_alma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = AlmaInput::from_slice(&empty, AlmaParams::default());
        let res = alma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlmaError::EmptyInputData)),
            "[{}] ALMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_alma_invalid_sigma(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = AlmaParams {
            period: Some(2),
            offset: None,
            sigma: Some(0.0),
        };
        let input = AlmaInput::from_slice(&data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlmaError::InvalidSigma { .. })),
            "[{}] ALMA should fail with invalid sigma",
            test_name
        );
        Ok(())
    }

    fn check_alma_invalid_offset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = AlmaParams {
            period: Some(2),
            offset: Some(f64::NAN),
            sigma: None,
        };
        let input = AlmaInput::from_slice(&data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlmaError::InvalidOffset { .. })),
            "[{}] ALMA should fail with invalid offset",
            test_name
        );
        Ok(())
    }

    fn check_alma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let first_input = AlmaInput::from_candles(&candles, "close", first_params);
        let first_result = alma_with_kernel(&first_input, kernel)?;

        let second_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let second_input = AlmaInput::from_slice(&first_result.values, second_params);
        let second_result = alma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        let expected_last_five = [
            59140.73195170,
            59211.58090986,
            59238.16030697,
            59222.63528822,
            59165.14427332,
        ];
        let start = second_result.values.len().saturating_sub(5);
        for (i, &val) in second_result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] ALMA Slice Reinput {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_alma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::from_candles(
            &candles,
            "close",
            AlmaParams {
                period: Some(9),
                offset: None,
                sigma: None,
            },
        );
        let res = alma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_alma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 9;
        let offset = 0.85;
        let sigma = 6.0;

        let input = AlmaInput::from_candles(
            &candles,
            "close",
            AlmaParams {
                period: Some(period),
                offset: Some(offset),
                sigma: Some(sigma),
            },
        );
        let batch_output = alma_with_kernel(&input, kernel)?.values;

        let mut stream = AlmaStream::try_new(AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma: Some(sigma),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(alma_val) => stream_values.push(alma_val),
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
                "[{}] ALMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_alma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = AlmaInput::from_candles(&candles, "close", AlmaParams::default());
        let output = alma_with_kernel(&input, kernel)?;
        
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
    fn check_alma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_alma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=64) // period
            .prop_flat_map(|period| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        period..400, // len ≥ period
                    ),
                    Just(period),
                    0f64..1f64,      // offset
                    0.1f64..10.0f64, // sigma
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, offset, sigma)| {
                let params = AlmaParams {
                    period: Some(period),
                    offset: Some(offset),
                    sigma: Some(sigma),
                };
                let input = AlmaInput::from_slice(&data, params);

                let AlmaOutput { values: out } = alma_with_kernel(&input, kernel).unwrap();
                let AlmaOutput { values: ref_out } =
                    alma_with_kernel(&input, Kernel::Scalar).unwrap();

                for i in (period - 1)..data.len() {
                    let window = &data[i + 1 - period..=i];
                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];
                    let r = ref_out[i];

                    prop_assert!(
                        y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                        "idx {i}: {y} ∉ [{lo}, {hi}]"
                    );

                    if period == 1 {
                        prop_assert!((y - data[i]).abs() <= f64::EPSILON);
                    }

                    if data.windows(2).all(|w| w[0] == w[1]) {
                        prop_assert!((y - data[0]).abs() <= 1e-9);
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_alma_tests {
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

    generate_all_alma_tests!(
        check_alma_partial_params,
        check_alma_accuracy,
        check_alma_default_candles,
        check_alma_zero_period,
        check_alma_period_exceeds_length,
        check_alma_very_small_dataset,
        check_alma_empty_input,
        check_alma_invalid_sigma,
        check_alma_invalid_offset,
        check_alma_reinput,
        check_alma_nan_handling,
        check_alma_streaming,
        check_alma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_alma_tests!(check_alma_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = AlmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = AlmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59286.72216704,
            59273.53428138,
            59204.37290721,
            59155.93381742,
            59026.92526112,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    
    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = AlmaBatchBuilder::new()
            .kernel(kernel)
            .period_range(9, 20, 1)
            .offset_range(0.5, 1.0, 0.1)
            .sigma_range(3.0, 9.0, 1.0)
            .apply_candles(&c, "close")?;

        let expected_combos = 12 * 6 * 7;
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
        
        let output = AlmaBatchBuilder::new()
            .kernel(kernel)
            .period_range(9, 15, 3)
            .offset_range(0.8, 0.9, 0.1)
            .sigma_range(6.0, 8.0, 2.0)
            .apply_candles(&c, "close")?;
        
        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }
            
            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;
            
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }
            
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }
            
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }
        }
        
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "alma")]
#[pyo3(signature = (data, period, offset, sigma, kernel=None))]

pub fn alma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    offset: f64,
    sigma: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };
    let alma_in = AlmaInput::from_slice(slice_in, params);

    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| -> Result<(), AlmaError> {
        let (data, weights, per, first, inv_n, chosen) = alma_prepare(&alma_in, kern)?;
        

        if first + per - 1 > 0 {
            slice_out[..first + per - 1].fill(f64::NAN);
        }
        
        alma_compute_into(data, &weights, per, first, inv_n, chosen, slice_out);
        
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "AlmaStream")]
pub struct AlmaStreamPy {
    stream: AlmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AlmaStreamPy {
    #[new]
    fn new(period: usize, offset: f64, sigma: f64) -> PyResult<Self> {
        let params = AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma: Some(sigma),
        };
        let stream =
            AlmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AlmaStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "alma_batch")]
#[pyo3(signature = (data, period_range, offset_range, sigma_range, kernel=None))]

pub fn alma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    offset_range: (f64, f64, f64),
    sigma_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = AlmaBatchRange {
        period: period_range,
        offset: offset_range,
        sigma: sigma_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py.allow_threads(|| {
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
        alma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
        "offsets",
        combos
            .iter()
            .map(|p| p.offset.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sigmas",
        combos
            .iter()
            .map(|p| p.sigma.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_js(data: &[f64], period: usize, offset: f64, sigma: f64) -> Result<Vec<f64>, JsValue> {
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };
    let input = AlmaInput::from_slice(data, params);

    alma_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    offset_start: f64,
    offset_end: f64,
    offset_step: f64,
    sigma_start: f64,
    sigma_end: f64,
    sigma_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AlmaBatchRange {
        period: (period_start, period_end, period_step),
        offset: (offset_start, offset_end, offset_step),
        sigma: (sigma_start, sigma_end, sigma_step),
    };

    alma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    offset_start: f64,
    offset_end: f64,
    offset_step: f64,
    sigma_start: f64,
    sigma_end: f64,
    sigma_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AlmaBatchRange {
        period: (period_start, period_end, period_step),
        offset: (offset_start, offset_end, offset_step),
        sigma: (sigma_start, sigma_end, sigma_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len() * 3);

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
        metadata.push(combo.offset.unwrap());
        metadata.push(combo.sigma.unwrap());
    }

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlmaBatchConfig {
    pub period_range: (usize, usize, usize),
    pub offset_range: (f64, f64, f64),
    pub sigma_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AlmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = alma_batch)]
pub fn alma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: AlmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AlmaBatchRange {
        period: config.period_range,
        offset: config.offset_range,
        sigma: config.sigma_range,
    };

    let output = alma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = AlmaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
