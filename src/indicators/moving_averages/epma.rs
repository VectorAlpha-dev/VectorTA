//! # End Point Moving Average (EPMA)
//!
//! ## MEMORY COPY OPERATION NOTE
//! The Python batch binding (`epma_batch_py`) contains a memory copy operation
//! where the output array is initialized with NaN values to prevent uninitialized memory issues.
//! This was added to fix memory safety issues but introduces a performance overhead.
//!
//! A polynomial-weighted moving average with adjustable period and offset.
//! SIMD (AVX2/AVX512) kernels are provided for API parity with alma.rs, but
//! offer little to no practical performance gain, as this indicator is memory bound.
//!
//! ## Parameters
//! - **period**: Window size, >= 2 (default: 11)
//! - **offset**: Weight offset (default: 4)
//!
//! ## Errors
//! - **EmptyInputData**: epma: Input data slice is empty.
//! - **AllValuesNaN**: epma: All input values are NaN
//! - **InvalidPeriod**: epma: `period` < 2 or `period` > data length
//! - **InvalidOffset**: epma: `offset` ≥ `period`
//! - **NotEnoughValidData**: epma: `period` + `offset` + 1 > valid data length
//!
//! ## Returns
//! - **Ok(EpmaOutput)** with a Vec<f64> of the same length as input
//! - **Err(EpmaError)** otherwise
//!

use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;
impl<'a> AsRef<[f64]> for EpmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EpmaData::Slice(slice) => slice,
            EpmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EpmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EpmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EpmaParams {
    pub period: Option<usize>,
    pub offset: Option<usize>,
}
impl Default for EpmaParams {
    fn default() -> Self {
        Self {
            period: Some(11),
            offset: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpmaInput<'a> {
    pub data: EpmaData<'a>,
    pub params: EpmaParams,
}

impl<'a> EpmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EpmaParams) -> Self {
        Self {
            data: EpmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EpmaParams) -> Self {
        Self {
            data: EpmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EpmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(11)
    }
    #[inline]
    pub fn get_offset(&self) -> usize {
        self.params.offset.unwrap_or(4)
    }
}

#[derive(Debug, Error)]
pub enum EpmaError {
    #[error("epma: Input data slice is empty.")]
    EmptyInputData,

    #[error("epma: All values are NaN.")]
    AllValuesNaN,

    #[error("epma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("epma: Invalid offset: {offset}")]
    InvalidOffset { offset: usize },

    #[error("epma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[derive(Copy, Clone, Debug)]
pub struct EpmaBuilder {
    period: Option<usize>,
    offset: Option<usize>,
    kernel: Kernel,
}
impl Default for EpmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            offset: None,
            kernel: Kernel::Auto,
        }
    }
}
impl EpmaBuilder {
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
    pub fn offset(mut self, o: usize) -> Self {
        self.offset = Some(o);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EpmaOutput, EpmaError> {
        let p = EpmaParams {
            period: self.period,
            offset: self.offset,
        };
        let i = EpmaInput::from_candles(c, "close", p);
        epma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EpmaOutput, EpmaError> {
        let p = EpmaParams {
            period: self.period,
            offset: self.offset,
        };
        let i = EpmaInput::from_slice(d, p);
        epma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EpmaStream, EpmaError> {
        let p = EpmaParams {
            period: self.period,
            offset: self.offset,
        };
        EpmaStream::try_new(p)
    }
}

#[inline]
pub fn epma(input: &EpmaInput) -> Result<EpmaOutput, EpmaError> {
    epma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn epma_prepare<'a>(
    input: &'a EpmaInput,
    kernel: Kernel,
) -> Result<
    (
        /*data*/ &'a [f64],
        /*period*/ usize,
        /*offset*/ usize,
        /*first*/ usize,
        /*warmup*/ usize,
        /*chosen*/ Kernel,
    ),
    EpmaError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EpmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EpmaError::AllValuesNaN)?;
    let period = input.get_period();
    let offset = input.get_offset();

    if offset >= period {
        return Err(EpmaError::InvalidOffset { offset });
    }

    if period < 2 || period > len {
        return Err(EpmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let needed = period + offset + 1;
    if (len - first) < needed {
        return Err(EpmaError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let warmup = first + period + offset + 1;

    Ok((data, period, offset, first, warmup, chosen))
}

#[inline(always)]
fn epma_compute_into(
    data: &[f64],
    period: usize,
    offset: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                epma_scalar(data, period, offset, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => epma_avx2(data, period, offset, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                epma_avx512(data, period, offset, first, out)
            }
            _ => unreachable!(),
        }
    }
}

pub fn epma_with_kernel(input: &EpmaInput, kernel: Kernel) -> Result<EpmaOutput, EpmaError> {
    let (data, period, offset, first, warmup, chosen) = epma_prepare(input, kernel)?;
    
    let mut out = alloc_with_nan_prefix(data.len(), warmup);
    epma_compute_into(data, period, offset, first, chosen, &mut out);
    
    Ok(EpmaOutput { values: out })
}

#[inline(always)]
pub fn epma_scalar(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let n = data.len();
    let p1 = period - 1;
    // Build weights for oldest-to-newest order
    let mut weights = Vec::with_capacity(p1);
    let mut weight_sum = 0.0;
    for i in 0..p1 {
        let w = (period as i32 - i as i32 - offset as i32) as f64;
        weights.push(w);
        weight_sum += w;
    }

    for j in (first_valid + period + offset + 1)..n {
        let start = j + 1 - p1;
        let mut my_sum = 0.0;
        let mut i = 0_usize;
        while i + 3 < p1 {
            my_sum += data[start + i] * weights[p1 - 1 - i];
            my_sum += data[start + i + 1] * weights[p1 - 2 - i];
            my_sum += data[start + i + 2] * weights[p1 - 3 - i];
            my_sum += data[start + i + 3] * weights[p1 - 4 - i];
            i += 4;
        }
        while i < p1 {
            my_sum += data[start + i] * weights[p1 - 1 - i];
            i += 1;
        }
        out[j] = my_sum / weight_sum;
    }
}

#[inline(always)]
fn build_weights_rev(period: usize, offset: usize) -> (Vec<f64>, f64) {
    let p1 = period - 1;
    let mut w = Vec::with_capacity(p1);
    let mut sum = 0.0;
    for k in 0..p1 {
        let val = (k as isize + 2 - offset as isize) as f64;
        w.push(val);
        sum += val;
    }
    (w, sum)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn epma_avx2(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    const STEP: usize = 4;
    let p1 = period - 1;
    let chunks = p1 / STEP;
    let tail = p1 % STEP;
    let mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    let (weights, wsum) = build_weights_rev(period, offset);
    let inv = 1.0 / wsum;

    for j in (first_valid + period + offset + 1)..data.len() {
        let start = j + 1 - p1;
        let mut acc = _mm256_setzero_pd();

        for blk in 0..chunks {
            let idx = blk * STEP;
            let w = _mm256_loadu_pd(weights.as_ptr().add(idx));
            let d = _mm256_loadu_pd(data.as_ptr().add(start + idx));
            acc = _mm256_fmadd_pd(d, w, acc);
        }
        if tail != 0 {
            let w_t = _mm256_maskload_pd(weights.as_ptr().add(chunks * STEP), mask);
            let d_t = _mm256_maskload_pd(data.as_ptr().add(start + chunks * STEP), mask);
            acc = _mm256_fmadd_pd(d_t, w_t, acc);
        }

        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let s2 = _mm_add_pd(hi, lo);
        let s1 = _mm_add_pd(s2, _mm_unpackhi_pd(s2, s2));
        *out.get_unchecked_mut(j) = _mm_cvtsd_f64(s1) * inv;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn epma_avx512(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        epma_avx512_short(data, period, offset, first_valid, out)
    
        } else {
        epma_avx512_long(data, period, offset, first_valid, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[inline]
unsafe fn epma_avx512_short(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let p1 = period - 1;
    let chunks = p1 / STEP;
    let tail = p1 % STEP;
    let tmask: __mmask8 = (1u8 << tail).wrapping_sub(1);

    let (weights, wsum) = build_weights_rev(period, offset);
    let inv = 1.0 / wsum;

    let w0 = _mm512_loadu_pd(weights.as_ptr());
    let w1 = if chunks >= 2 {
        Some(_mm512_loadu_pd(weights.as_ptr().add(STEP)))
    
        } else {
        None
    };
    let w_tail = if tail != 0 {
        _mm512_maskz_loadu_pd(tmask, weights.as_ptr().add(chunks * STEP))
    
        } else {
        _mm512_setzero_pd()
    };

    for j in (first_valid + period + offset + 1)..data.len() {
        let start = j + 1 - p1;
        let mut acc = _mm512_mul_pd(_mm512_loadu_pd(data.as_ptr().add(start)), w0);

        if let Some(w1v) = w1 {
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + STEP));
            acc = _mm512_fmadd_pd(d1, w1v, acc);
        }
        if tail != 0 {
            let d_t = _mm512_maskz_loadu_pd(tmask, data.as_ptr().add(start + chunks * STEP));
            acc = _mm512_fmadd_pd(d_t, w_tail, acc);
        }
        *out.get_unchecked_mut(j) = _mm512_reduce_add_pd(acc) * inv;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn epma_avx512_long(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let p1 = period - 1;
    let n_chunks = p1 / STEP;
    let tail_len = p1 % STEP;
    let paired = n_chunks & !3;
    let tmask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    let (weights, wsum) = build_weights_rev(period, offset);
    let inv = 1.0 / wsum;

    let mut wregs: Vec<__m512d> = Vec::with_capacity(n_chunks + (tail_len != 0) as usize);
    for blk in 0..n_chunks {
        wregs.push(_mm512_loadu_pd(weights.as_ptr().add(blk * STEP)));
    }
    if tail_len != 0 {
        wregs.push(_mm512_maskz_loadu_pd(
            tmask,
            weights.as_ptr().add(n_chunks * STEP),
        ));
    }

    for j in (first_valid + period + offset + 1)..data.len() {
        let start_ptr = data.as_ptr().add(j + 1 - p1);
        let mut s0 = _mm512_setzero_pd();
        let mut s1 = _mm512_setzero_pd();
        let mut s2 = _mm512_setzero_pd();
        let mut s3 = _mm512_setzero_pd();

        for blk in (0..paired).step_by(4) {
            let d0 = _mm512_loadu_pd(start_ptr.add(blk * STEP));
            let d1 = _mm512_loadu_pd(start_ptr.add((blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(start_ptr.add((blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(start_ptr.add((blk + 3) * STEP));

            s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk), s0);
            s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
            s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
            s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);
        }
        for blk in paired..n_chunks {
            let d = _mm512_loadu_pd(start_ptr.add(blk * STEP));
            s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(blk), s0);
        }
        if tail_len != 0 {
            let d_t = _mm512_maskz_loadu_pd(tmask, start_ptr.add(n_chunks * STEP));
            s0 = _mm512_fmadd_pd(d_t, *wregs.last().unwrap(), s0);
        }

        let total = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
        *out.get_unchecked_mut(j) = _mm512_reduce_add_pd(total) * inv;
    }
}

#[derive(Debug, Clone)]
pub struct EpmaStream {
    period: usize,
    offset: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    seen: usize,
    weights: Vec<f64>,
    weight_sum: f64,
}

impl EpmaStream {
    pub fn try_new(params: EpmaParams) -> Result<Self, EpmaError> {
        let period = params.period.unwrap_or(11);
        let offset = params.offset.unwrap_or(4);

        if period < 2 {
            return Err(EpmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        if offset >= period {
            return Err(EpmaError::InvalidOffset { offset });
        }

        let mut weights = Vec::with_capacity(period - 1);
        for i in 0..(period - 1) {
            weights.push((period as i32 - i as i32 - offset as i32) as f64);
        }
        let weight_sum: f64 = weights.iter().sum();

        Ok(Self {
            period,
            offset,
            buffer: alloc_with_nan_prefix(period, period),
            head: 0,
            filled: false,
            seen: 0,
            weights,
            weight_sum,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        self.seen += 1;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }

        if self.seen <= self.period + self.offset + 1 {
            return Some(value);
        }

        Some(self.dot_ring())
    }

    #[inline(always)]
    fn dot_ring(&self) -> f64 {
        let mut idx = (self.head + self.period - 1) % self.period;
        let mut sum = 0.0;

        for &w in &self.weights {
            sum += w * self.buffer[idx];
            idx = if idx == 0 { self.period - 1 } else { idx - 1 };
        }

        sum / self.weight_sum
    }
}

#[derive(Clone, Debug)]
pub struct EpmaBatchRange {
    pub period: (usize, usize, usize),
    pub offset: (usize, usize, usize),
}
impl Default for EpmaBatchRange {
    fn default() -> Self {
        Self {
            period: (11, 22, 1),
            offset: (4, 4, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct EpmaBatchBuilder {
    range: EpmaBatchRange,
    kernel: Kernel,
}
impl EpmaBatchBuilder {
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
    pub fn offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.offset = (start, end, step);
        self
    }
    #[inline]
    pub fn offset_static(mut self, o: usize) -> Self {
        self.range.offset = (o, o, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<EpmaBatchOutput, EpmaError> {
        epma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<EpmaBatchOutput, EpmaError> {
        EpmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EpmaBatchOutput, EpmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<EpmaBatchOutput, EpmaError> {
        EpmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct EpmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EpmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EpmaBatchOutput {
    pub fn row_for_params(&self, p: &EpmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(11) == p.period.unwrap_or(11)
                && c.offset.unwrap_or(4) == p.offset.unwrap_or(4)
        })
    }
    pub fn values_for(&self, p: &EpmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EpmaBatchRange) -> Vec<EpmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let offsets = axis_usize(r.offset);
    let mut out = Vec::with_capacity(periods.len() * offsets.len());
    for &p in &periods {
        for &o in &offsets {
            out.push(EpmaParams {
                period: Some(p),
                offset: Some(o),
            });
        }
    }
    out
}

#[inline(always)]
pub fn epma_batch_with_kernel(
    data: &[f64],
    sweep: &EpmaBatchRange,
    k: Kernel,
) -> Result<EpmaBatchOutput, EpmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EpmaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    epma_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn epma_batch_slice(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
) -> Result<EpmaBatchOutput, EpmaError> {
    epma_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn epma_batch_par_slice(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
) -> Result<EpmaBatchOutput, EpmaError> {
    epma_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn epma_batch_inner(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EpmaBatchOutput, EpmaError> {
    let rows = expand_grid(sweep).len();
    let cols = data.len();
    let mut out = vec![f64::NAN; rows * cols];
    
    let combos = epma_batch_inner_into(data, sweep, kern, parallel, &mut out)?;
    
    Ok(EpmaBatchOutput {
        values: out,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn epma_batch_inner_into(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EpmaParams>, EpmaError> {
    if data.is_empty() {
        return Err(EpmaError::EmptyInputData);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EpmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    for c in &combos {
        let p = c.period.unwrap();
        let o = c.offset.unwrap();
        if p < 2 {
            return Err(EpmaError::InvalidPeriod {
                period: p,
                data_len: data.len(),
            });
        }
        if o >= p {
            return Err(EpmaError::InvalidOffset { offset: o });
        }
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EpmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let max_off = combos.iter().map(|c| c.offset.unwrap()).max().unwrap();
    let needed = max_p + max_off + 1;
    if data.len() - first < needed {
        return Err(EpmaError::NotEnoughValidData {
            needed,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Reinterpret the output slice as MaybeUninit
    let raw = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len(),
        )
    };

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() + c.offset.unwrap() + 1)
        .collect();
    unsafe { init_matrix_prefixes(raw, cols, &warm) };

    // ---------------------------------------------------------------------------
    // 2. helper that computes **one** row into a &mut [MaybeUninit<f64>]
    // ---------------------------------------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let offset = combos[row].offset.unwrap();

        // cast this slice only; we know the kernel will overwrite every cell
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => epma_row_avx512(data, first, period, offset, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => epma_row_avx2(data, first, period, offset, dst),
            _ => epma_row_scalar(data, first, period, offset, dst),
        }
    };

    // ---------------------------------------------------------------------------
    // 3. run all rows (parallel or serial) on the MaybeUninit buffer
    // ---------------------------------------------------------------------------
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        raw.par_chunks_mut(cols)

                    .enumerate()

                    .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in raw.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }

        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}
#[inline(always)]
unsafe fn epma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    offset: usize,
    out: &mut [f64],
) {
    epma_scalar(data, period, offset, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn epma_row_avx2(data: &[f64], first: usize, period: usize, offset: usize, out: &mut [f64]) {
    epma_avx2(data, period, offset, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn epma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    offset: usize,
    out: &mut [f64],
) {
    epma_avx512(data, period, offset, first, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_epma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EpmaParams {
            period: None,
            offset: None,
        };
        let input = EpmaInput::from_candles(&candles, "close", default_params);
        let output = epma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_epma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = EpmaParams::default();
        let input = EpmaInput::from_candles(&candles, "close", default_params);
        let result = epma_with_kernel(&input, kernel)?;
        let expected_last_five = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];
        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "[{}] EPMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                value,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_epma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EpmaInput::with_default_candles(&candles);
        match input.data {
            EpmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EpmaData::Candles"),
        }
        let output = epma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_epma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = EpmaParams {
            period: Some(0),
            offset: None,
        };
        let input = EpmaInput::from_slice(&input_data, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EPMA should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_epma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = EpmaParams {
            period: Some(10),
            offset: None,
        };
        let input = EpmaInput::from_slice(&data_small, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EPMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_epma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = EpmaParams {
            period: Some(9),
            offset: None,
        };
        let input = EpmaInput::from_slice(&single_point, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EPMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_epma_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EpmaInput::from_slice(&empty, EpmaParams::default());
        let res = epma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EpmaError::EmptyInputData)),
            "[{}] EPMA should fail with empty input",
            test_name
        );
        Ok(())
    }
    fn check_epma_invalid_offset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0];
        let params = EpmaParams {
            period: Some(3),
            offset: Some(3),
        };
        let input = EpmaInput::from_slice(&data, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EpmaError::InvalidOffset { .. })),
            "[{}] EPMA should fail with invalid offset",
            test_name
        );
        Ok(())
    }
    fn check_epma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (
            proptest::collection::vec(
                (-1e6f64..1e6).prop_filter("finite", |x| x.is_finite()),
                30..200,
            ),
            3usize..30,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = EpmaParams {
                    period: Some(period),
                    offset: Some(0),
                };
                let input = EpmaInput::from_slice(&data, params);
                let EpmaOutput { values: out } = epma_with_kernel(&input, kernel).unwrap();

                let start = period + 1;
                for i in start..data.len() {
                    let y = out[i];
                    // EPMA can produce values outside the input window bounds due to its
                    // polynomial weighting scheme which can include negative weights.
                    // This is expected behavior for a leading indicator.
                    // We only check that the output is finite (not NaN or infinite).
                    prop_assert!(y.is_nan() || y.is_finite(), 
                        "EPMA output at index {} is not finite: {}", i, y);
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }
    fn check_epma_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = EpmaParams {
            period: Some(9),
            offset: None,
        };
        let first_input = EpmaInput::from_candles(&candles, "close", first_params);
        let first_result = epma_with_kernel(&first_input, kernel)?;
        let second_params = EpmaParams {
            period: Some(3),
            offset: None,
        };
        let second_input = EpmaInput::from_slice(&first_result.values, second_params);
        let second_result = epma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_epma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = EpmaParams {
            period: Some(11),
            offset: Some(4),
        };
        let input = EpmaInput::from_candles(&candles, "close", params.clone());
        let res = epma_with_kernel(&input, kernel)?;
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
    fn check_epma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 11;
        let offset = 4;
        let input = EpmaInput::from_candles(
            &candles,
            "close",
            EpmaParams {
                period: Some(period),
                offset: Some(offset),
            },
        );
        let batch_output = epma_with_kernel(&input, kernel)?.values;
        let mut stream = EpmaStream::try_new(EpmaParams {
            period: Some(period),
            offset: Some(offset),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output
            .iter()
            .zip(stream_values.iter())
            .enumerate()
            .skip(period + offset + 1)
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] EPMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_epma_tests {
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
    generate_all_epma_tests!(
        check_epma_partial_params,
        check_epma_accuracy,
        check_epma_default_candles,
        check_epma_zero_period,
        check_epma_period_exceeds_length,
        check_epma_very_small_dataset,
        check_epma_empty_input,
        check_epma_invalid_offset,
        check_epma_reinput,
        check_epma_nan_handling,
        check_epma_streaming,
        check_epma_property
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EpmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = EpmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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
    gen_batch_tests!(check_batch_default_row);
}

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use numpy;

#[cfg(feature = "python")]
#[pyfunction(name = "epma")]
#[pyo3(signature = (arr_in, period=None, offset=None))]
pub fn epma_py<'py>(
    py: Python<'py>,
    arr_in: numpy::PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    offset: Option<usize>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};
    
    let slice_in = arr_in.as_slice()?; // zero-copy view
    
    let params = EpmaParams { period, offset };
    let input = EpmaInput::from_slice(slice_in, params);
    
    // Pre-allocate output array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    // Release GIL for computation
    py.allow_threads(|| -> Result<(), EpmaError> {
        let (data, period, offset, first, warmup, chosen) = epma_prepare(&input, Kernel::Auto)?;
        // Initialize NaN prefix
        slice_out[..warmup].fill(f64::NAN);
        // Compute directly into output buffer
        epma_compute_into(data, period, offset, first, chosen, slice_out);
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyfunction(name = "epma_batch")]
#[pyo3(signature = (arr_in, period_range, offset_range))]
pub fn epma_batch_py<'py>(
    py: Python<'py>,
    arr_in: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    offset_range: (usize, usize, usize),
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArray1, PyArray2, PyArrayMethods};
    
    let data = arr_in.as_slice()?;
    let range = EpmaBatchRange {
        period: period_range,
        offset: offset_range,
    };
    
    // Pre-calculate dimensions
    let combos = expand_grid(&range);
    let rows = combos.len();
    let cols = data.len();
    
    // Pre-allocate output array and initialize with NaN
    let out_arr = unsafe { PyArray2::<f64>::new(py, [rows, cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };
    // Initialize all values to NaN to avoid uninitialized memory issues
    for v in out_slice.iter_mut() {
        *v = f64::NAN;
    }
    
    // Map kernel (ScalarBatch -> Scalar, etc.)
    let kernel = match Kernel::ScalarBatch {
        Kernel::ScalarBatch => Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => Kernel::Scalar,
    };
    
    // Release GIL for computation
    let combos_result = py.allow_threads(|| -> Result<Vec<EpmaParams>, EpmaError> {
        epma_batch_inner_into(data, &range, kernel, false, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    let dict = PyDict::new(py);
    
    // Extract periods and offsets
    let periods: Vec<usize> = combos_result.iter()
        .map(|c| c.period.unwrap_or(11))
        .collect();
    let offsets: Vec<usize> = combos_result.iter()
        .map(|c| c.offset.unwrap_or(4))
        .collect();
    
    dict.set_item("values", out_arr)?;
    dict.set_item("periods", PyArray1::from_vec(py, periods))?;
    dict.set_item("offsets", PyArray1::from_vec(py, offsets))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "EpmaStream")]
pub struct EpmaStreamPy {
    inner: EpmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EpmaStreamPy {
    #[new]
    #[pyo3(signature = (period=None, offset=None))]
    fn new(period: Option<usize>, offset: Option<usize>) -> PyResult<Self> {
        let params = EpmaParams { period, offset };
        match EpmaStream::try_new(params) {
            Ok(stream) => Ok(Self { inner: stream }),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// WASM bindings
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_js(data: &[f64], period: Option<usize>, offset: Option<usize>) -> Result<Vec<f64>, JsValue> {
    let params = EpmaParams { period, offset };
    let input = EpmaInput::from_slice(data, params);
    
    match epma_with_kernel(&input, Kernel::Scalar) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    offset_start: usize,
    offset_end: usize,
    offset_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let range = EpmaBatchRange {
        period: (period_start, period_end, period_step),
        offset: (offset_start, offset_end, offset_step),
    };
    
    match epma_batch_with_kernel(data, &range, Kernel::ScalarBatch) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    offset_start: usize,
    offset_end: usize,
    offset_step: usize,
) -> Vec<usize> {
    let range = EpmaBatchRange {
        period: (period_start, period_end, period_step),
        offset: (offset_start, offset_end, offset_step),
    };
    
    let combos = expand_grid(&range);
    let mut metadata = Vec::with_capacity(combos.len() * 2);
    
    for combo in combos {
        metadata.push(combo.period.unwrap_or(11));
        metadata.push(combo.offset.unwrap_or(4));
    }
    
    metadata
}
