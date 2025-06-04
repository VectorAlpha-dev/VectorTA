//! # Arnaud Legoux Moving Average (ALMA)
//!
//! A smooth yet responsive moving average that uses Gaussian weighting. Its parameters
//! (`period`, `offset`, `sigma`) control the window size, the weighting center, and
//! the Gaussian smoothness. ALMA can also be re-applied to its own output, allowing
//! iterative smoothing on previously computed results.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **offset**: Shift in [0.0, 1.0] for the Gaussian center (defaults to 0.85).
//! - **sigma**: Controls the Gaussian curve’s width (defaults to 6.0).
//!
//! ## Errors
//! - **AllValuesNaN**: alma: All input data values are `NaN`.
//! - **InvalidPeriod**: alma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: alma: Not enough valid data points for the requested `period`.
//! - **InvalidSigma**: alma: `sigma` ≤ 0.0.
//! - **InvalidOffset**: alma: `offset` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(AlmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(AlmaError)`** otherwise.
//!

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use numpy::{PyArray1, IntoPyArray};

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
use std::alloc::{alloc, dealloc, Layout};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
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

pub fn alma_with_kernel(input: &AlmaInput, kernel: Kernel) -> Result<AlmaOutput, AlmaError> {
    let data: &[f64] = match &input.data {
        AlmaData::Candles { candles, source } => source_type(candles, source),
        AlmaData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlmaError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();
    let offset = input.get_offset();
    let sigma = input.get_sigma();

    if period == 0 || period > len {
        return Err(AlmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
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

    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                alma_scalar(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                alma_avx2(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_avx512(data, &weights, period, first, inv_norm, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(AlmaOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
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
    let chunks = period / STEP;
    let tail_len = period % STEP;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    if chunks == 0 {
        let w_vec = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr());
        for i in (first_valid + period - 1)..data.len() {
            let start = i + 1 - period;
            let d_vec = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start));
            let sum = _mm512_reduce_add_pd(_mm512_mul_pd(d_vec, w_vec));
            *out.get_unchecked_mut(i) = sum * inv_norm;
        }
        return;
    }

    let w0 = _mm512_loadu_pd(weights.as_ptr());
    let w1 = if chunks >= 2 {
        Some(_mm512_loadu_pd(weights.as_ptr().add(STEP)))
    } else {
        None
    };

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;

        let mut acc = _mm512_setzero_pd();

        let d0 = _mm512_loadu_pd(data.as_ptr().add(start));
        acc = _mm512_fmadd_pd(d0, w0, acc);

        if let Some(w1v) = w1 {
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + STEP));
            acc = _mm512_fmadd_pd(d1, w1v, acc);
        }

        if tail_len != 0 {
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + chunks * STEP));
            let w_tail = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(chunks * STEP));
            acc = _mm512_fmadd_pd(d_tail, w_tail, acc);
        }

        let sum = _mm512_reduce_add_pd(acc);
        *out.get_unchecked_mut(i) = sum * inv_norm;
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
    let tail_mask = (1u8 << tail_len).wrapping_sub(1);

    debug_assert!(period >= 1 && n_chunks > 0);
    debug_assert!(data.len() == out.len());
    debug_assert!(weights.len() >= period);

    let mut wregs: Vec<__m512d> = Vec::with_capacity(n_chunks + (tail_len != 0) as usize);
    for blk in 0..n_chunks {
        wregs.push(_mm512_loadu_pd(weights.as_ptr().add(blk * STEP)));
    }
    if tail_len != 0 {
        wregs.push(_mm512_maskz_loadu_pd(
            tail_mask,
            weights.as_ptr().add(n_chunks * STEP),
        ));
    }

    let mut data_ptr = data.as_ptr().add(first_valid);
    let stop = data.as_ptr().add(data.len());

    for dst in &mut out[first_valid + period - 1..] {
        let mut sum0 = _mm512_setzero_pd();
        let mut sum1 = _mm512_setzero_pd();
        let mut sum2 = _mm512_setzero_pd();
        let mut sum3 = _mm512_setzero_pd();

        for blk in (0..paired).step_by(4) {
            _mm_prefetch(data_ptr.add((blk + 8) * STEP) as *const i8, _MM_HINT_T0);

            let d0 = _mm512_loadu_pd(data_ptr.add(blk * STEP));
            let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));

            sum0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk), sum0);
            sum1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), sum1);
            sum2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), sum2);
            sum3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), sum3);
        }

        for blk in paired..n_chunks {
            let d = _mm512_loadu_pd(data_ptr.add(blk * STEP));
            sum0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(blk), sum0);
        }

        if tail_len != 0 {
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data_ptr.add(n_chunks * STEP));
            sum0 = _mm512_fmadd_pd(d_tail, *wregs.get_unchecked(n_chunks), sum0);
        }

        let mut total = _mm512_add_pd(_mm512_add_pd(sum0, sum1), _mm512_add_pd(sum2, sum3));
        let value = _mm512_reduce_add_pd(total) * inv_norm;
        *dst = value;

        data_ptr = data_ptr.add(1);
        if data_ptr.add(period) > stop {
            break;
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

    let mut raw = make_uninit_matrix(rows, cols);      // step 1

    // collect warm-up lengths per row once
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };  // step 2

    // turn into Vec<f64>
    let mut values: Vec<f64> = unsafe {
        let ptr = raw.as_mut_ptr() as *mut f64;
        let cap = raw.capacity();
        std::mem::forget(raw);
        Vec::from_raw_parts(ptr, rows * cols, cap)
    };
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let inv_n = *inv_norms.get_unchecked(row);

        match kern {
            Kernel::Scalar => alma_row_scalar(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => alma_row_avx2(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => alma_row_avx512(data, first, period, max_p, w_ptr, inv_n, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(AlmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
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

    let w0 = _mm512_loadu_pd(w_ptr);
    let w1 = if chunks >= 2 {
        Some(_mm512_loadu_pd(w_ptr.add(STEP)))
    } else {
        None
    };

    if tail_len == 0 {
        for i in (first + period - 1)..data.len() {
            let start = i + 1 - period;
            let mut acc = _mm512_fmadd_pd(
                _mm512_loadu_pd(data.as_ptr().add(start)),
                w0,
                _mm512_setzero_pd(),
            );

            if let Some(w1v) = w1 {
                let d1 = _mm512_loadu_pd(data.as_ptr().add(start + STEP));
                acc = _mm512_fmadd_pd(d1, w1v, acc);
            }

            let res = _mm512_reduce_add_pd(acc) * inv_n;
            _mm_stream_sd(out.get_unchecked_mut(i) as *mut f64, _mm_set_sd(res));
        }
        return;
    }

    let tmask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);
    let w_tail = _mm512_maskz_loadu_pd(tmask, w_ptr.add(chunks * STEP));

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm512_fmadd_pd(
            _mm512_loadu_pd(data.as_ptr().add(start)),
            w0,
            _mm512_setzero_pd(),
        );

        if let Some(w1v) = w1 {
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + STEP));
            acc = _mm512_fmadd_pd(d1, w1v, acc);
        }

        let d_tail = _mm512_maskz_loadu_pd(tmask, data.as_ptr().add(start + chunks * STEP));
        acc = _mm512_fmadd_pd(d_tail, w_tail, acc);

        let res = _mm512_reduce_add_pd(acc) * inv_n;
        _mm_stream_sd(out.get_unchecked_mut(i) as *mut f64, _mm_set_sd(res));
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub(crate) unsafe fn alma_row_avx512_long(
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
        let res = _mm512_reduce_add_pd(sum) * inv_n;

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
        let res = _mm512_reduce_add_pd(sum) * inv_n;

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
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

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
        check_alma_reinput,
        check_alma_nan_handling,
        check_alma_streaming
    );

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
    gen_batch_tests!(check_batch_default_row);
}

#[cfg(feature = "python")]
#[pyfunction]
fn alma<'py>(py: Python<'py>,
                     input: &'py PyArray1<f64>,
                     period: usize,
                     offset: f64,
                     sigma: f64) -> PyResult<&'py PyArray1<f64>> {
    let slice: &[f64] = unsafe {
        input.as_slice().map_err(|_| {
            PyValueError::new_err("Input must be a contiguous 1-D numpy.ndarray[float64]")
        })?
    };
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };
    let alma_input = AlmaInput::from_slice(slice, params);
    let values: Vec<f64> = py.allow_threads(|| {
        let AlmaOutput { values } = alma_with_kernel(&alma_input, Default::default())
            .expect("ALMA computation failed");
        values
    });
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pymodule]
fn my_project(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(alma, m)?)?;
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_js(data: &[f64], period: usize, offset: f64, sigma: f64) -> Vec<f64> {
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };
    let input = AlmaInput::from_slice(data, params);

    let AlmaOutput { values } = alma_with_kernel(&input, Default::default())
        .expect("ALMA computation failed");

    values
}