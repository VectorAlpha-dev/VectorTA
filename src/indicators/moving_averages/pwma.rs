//! # Pascal Weighted Moving Average (PWMA)
//!
//! A weighted moving average using Pascal's triangle coefficients for weights.
//! The weights follow binomial coefficients, giving more emphasis to middle values
//! in the window, creating a bell-curve-like weighting pattern.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). Defaults to 5.
//!
//! ## Returns
//! - **`Ok(PwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(PwmaError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ✅ Fully implemented - 4-wide SIMD with FMA operations for weighted averaging
//! - **AVX512 kernel**: ✅ Fully implemented - Dual-path optimization (short ≤32, long >32 periods), 8-wide SIMD
//! - **Streaming update**: ⚠️ O(n) complexity - dot_ring() iterates through all Pascal weights
//!   - TODO: Could optimize to O(1) with incremental updates using Pascal's triangle properties
//! - **Memory optimization**: ✅ Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) for output vectors
//! - **Note**: Pascal weights are precomputed and normalized, suitable for SIMD vectorization

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaPwma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for PwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PwmaData::Slice(slice) => slice,
            PwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct PwmaParams {
    pub period: Option<usize>,
}

impl Default for PwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct PwmaInput<'a> {
    pub data: PwmaData<'a>,
    pub params: PwmaParams,
}

impl<'a> PwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PwmaParams) -> Self {
        Self {
            data: PwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PwmaParams) -> Self {
        Self {
            data: PwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for PwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PwmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<PwmaOutput, PwmaError> {
        let p = PwmaParams {
            period: self.period,
        };
        let i = PwmaInput::from_candles(c, "close", p);
        pwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PwmaOutput, PwmaError> {
        let p = PwmaParams {
            period: self.period,
        };
        let i = PwmaInput::from_slice(d, p);
        pwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PwmaStream, PwmaError> {
        let p = PwmaParams {
            period: self.period,
        };
        PwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PwmaError {
    #[error("pwma: All values are NaN.")]
    AllValuesNaN,
    #[error("pwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("pwma: Pascal weights sum to zero for period = {period}")]
    PascalWeightsSumZero { period: usize },
}

#[inline]
pub fn pwma(input: &PwmaInput) -> Result<PwmaOutput, PwmaError> {
    pwma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn pwma_prepare<'a>(
    input: &'a PwmaInput,
    kernel: Kernel,
) -> Result<
    (
        // data
        &'a [f64],
        // weights
        Vec<f64>,
        // period
        usize,
        // first
        usize,
        // chosen
        Kernel,
    ),
    PwmaError,
> {
    let data: &[f64] = input.as_ref();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PwmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(PwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let weights = pascal_weights(period)?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, weights, period, first, chosen))
}

#[inline(always)]
fn pwma_compute_into(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => pwma_scalar(data, weights, period, first, out),

            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => pwma_avx2(data, weights, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => pwma_avx512(data, weights, period, first, out),

            // Fallbacks when AVX code is not compiled in:
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                pwma_scalar(data, weights, period, first, out)
            }
            _ => unreachable!(),
        }
    }
}

pub fn pwma_with_kernel(input: &PwmaInput, kernel: Kernel) -> Result<PwmaOutput, PwmaError> {
    let (data, weights, period, first, chosen) = pwma_prepare(input, kernel)?;

    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    pwma_compute_into(data, &weights, period, first, chosen, &mut out);

    Ok(PwmaOutput { values: out })
}

/// Computes PWMA directly into a provided output slice, avoiding allocation.
/// The output slice must be the same length as the input data.
#[inline]
pub fn pwma_into_slice(dst: &mut [f64], input: &PwmaInput, kern: Kernel) -> Result<(), PwmaError> {
    let (data, weights, period, first, chosen) = pwma_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(PwmaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    // Compute PWMA values directly into dst
    pwma_compute_into(data, &weights, period, first, chosen, dst);

    // Fill warmup period with NaN
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn pwma_scalar(data: &[f64], weights: &[f64], period: usize, first: usize, out: &mut [f64]) {
    assert_eq!(weights.len(), period, "weights.len() must equal `period`");
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let window = &data[start..start + period];
        let mut sum = 0.0;
        for (d, w) in window.iter().zip(weights.iter()) {
            sum += d * w;
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pwma_avx512(data: &[f64], weights: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { pwma_avx512_short(data, weights, period, first, out) }
    } else {
        unsafe { pwma_avx512_long(data, weights, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pwma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let vecs = period / 4; // AVX2 processes 4 doubles per vector
    let tail = period % 4;

    for i in (first + period - 1)..len {
        let start = i + 1 - period;
        let mut acc = _mm256_setzero_pd();

        // Process full 256-bit vectors (4 doubles at a time)
        for v in 0..vecs {
            let d = _mm256_loadu_pd(data.as_ptr().add(start + v * 4));
            let w = _mm256_loadu_pd(weights.as_ptr().add(v * 4));
            acc = _mm256_fmadd_pd(d, w, acc);
        }

        // Horizontal sum of acc
        // Split 256-bit into two 128-bit halves and add
        let low128 = _mm256_castpd256_pd128(acc);
        let high128 = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(low128, high128);

        // Final horizontal add of 2 doubles
        let high64 = _mm_unpackhi_pd(sum128, sum128);
        let final_sum = _mm_add_sd(sum128, high64);
        let mut total = _mm_cvtsd_f64(final_sum);

        // Process tail elements
        for t in 0..tail {
            let d = *data.as_ptr().add(start + vecs * 4 + t);
            let w = *weights.as_ptr().add(vecs * 4 + t);
            total = d.mul_add(w, total);
        }

        *out.as_mut_ptr().add(i) = total;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn pwma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let vecs = period / 8;
    let tail = period % 8;
    let len = data.len();

    // Create tail mask for AVX512 masked operations
    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..len {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        // Process all full vectors
        for v in 0..vecs {
            let d = _mm512_loadu_pd(data.as_ptr().add(start + v * 8));
            let w = _mm512_loadu_pd(weights.as_ptr().add(v * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        // Process tail with AVX512 masking
        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(vecs * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        // Use optimized horizontal reduction intrinsic
        let total = _mm512_reduce_add_pd(acc);

        // Use stream store to avoid cache pollution
        _mm_stream_sd(out.as_mut_ptr().add(i), _mm_set_sd(total));
    }

    // Ensure all stream stores are committed
    _mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn pwma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let full_vecs = period / 8;
    let tail = period % 8;

    // Create tail mask
    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..len {
        let start = i + 1 - period;

        // Use 4 accumulators for maximum ILP with AVX512's 32 registers
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        // Prefetch next window's data
        if i + 1 < len {
            _mm_prefetch(data.as_ptr().add(start + period) as *const i8, _MM_HINT_T0);
        }

        // Process vectors in groups of 4 for better ILP
        let quads = full_vecs / 4;
        let remaining = full_vecs % 4;

        for q in 0..quads {
            let base = q * 4 * 8;
            let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + base + 8));
            let d2 = _mm512_loadu_pd(data.as_ptr().add(start + base + 16));
            let d3 = _mm512_loadu_pd(data.as_ptr().add(start + base + 24));

            let w0 = _mm512_loadu_pd(weights.as_ptr().add(base));
            let w1 = _mm512_loadu_pd(weights.as_ptr().add(base + 8));
            let w2 = _mm512_loadu_pd(weights.as_ptr().add(base + 16));
            let w3 = _mm512_loadu_pd(weights.as_ptr().add(base + 24));

            acc0 = _mm512_fmadd_pd(d0, w0, acc0);
            acc1 = _mm512_fmadd_pd(d1, w1, acc1);
            acc2 = _mm512_fmadd_pd(d2, w2, acc2);
            acc3 = _mm512_fmadd_pd(d3, w3, acc3);
        }

        // Process remaining vectors
        let base = quads * 4 * 8;
        match remaining {
            3 => {
                let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
                let d1 = _mm512_loadu_pd(data.as_ptr().add(start + base + 8));
                let d2 = _mm512_loadu_pd(data.as_ptr().add(start + base + 16));
                let w0 = _mm512_loadu_pd(weights.as_ptr().add(base));
                let w1 = _mm512_loadu_pd(weights.as_ptr().add(base + 8));
                let w2 = _mm512_loadu_pd(weights.as_ptr().add(base + 16));
                acc0 = _mm512_fmadd_pd(d0, w0, acc0);
                acc1 = _mm512_fmadd_pd(d1, w1, acc1);
                acc2 = _mm512_fmadd_pd(d2, w2, acc2);
            }
            2 => {
                let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
                let d1 = _mm512_loadu_pd(data.as_ptr().add(start + base + 8));
                let w0 = _mm512_loadu_pd(weights.as_ptr().add(base));
                let w1 = _mm512_loadu_pd(weights.as_ptr().add(base + 8));
                acc0 = _mm512_fmadd_pd(d0, w0, acc0);
                acc1 = _mm512_fmadd_pd(d1, w1, acc1);
            }
            1 => {
                let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
                let w0 = _mm512_loadu_pd(weights.as_ptr().add(base));
                acc0 = _mm512_fmadd_pd(d0, w0, acc0);
            }
            _ => {}
        }

        // Process tail with masking
        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + full_vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(full_vecs * 8));
            acc0 = _mm512_fmadd_pd(d, w, acc0);
        }

        // Combine all accumulators
        let acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

        // Use optimized horizontal reduction intrinsic
        let total = _mm512_reduce_add_pd(acc);

        // Use stream store to avoid cache pollution
        _mm_stream_sd(out.as_mut_ptr().add(i), _mm_set_sd(total));
    }

    // Ensure all stream stores are committed
    _mm_sfence();
}

#[derive(Debug, Clone)]
pub struct PwmaStream {
    period: usize,
    weights: Vec<f64>,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl PwmaStream {
    pub fn try_new(params: PwmaParams) -> Result<Self, PwmaError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(PwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let weights = pascal_weights(period)?;
        Ok(Self {
            period,
            weights,
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
        sum
    }
}

#[derive(Clone, Debug)]
pub struct PwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for PwmaBatchRange {
    fn default() -> Self {
        Self { period: (5, 30, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PwmaBatchBuilder {
    range: PwmaBatchRange,
    kernel: Kernel,
}

impl PwmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<PwmaBatchOutput, PwmaError> {
        pwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PwmaBatchOutput, PwmaError> {
        PwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PwmaBatchOutput, PwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PwmaBatchOutput, PwmaError> {
        PwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub struct PwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl PwmaBatchOutput {
    pub fn row_for_params(&self, p: &PwmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &PwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid(r: &PwmaBatchRange) -> Vec<PwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(PwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn pwma_batch_slice(
    data: &[f64],
    sweep: &PwmaBatchRange,
    kern: Kernel,
) -> Result<PwmaBatchOutput, PwmaError> {
    pwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn pwma_batch_par_slice(
    data: &[f64],
    sweep: &PwmaBatchRange,
    kern: Kernel,
) -> Result<PwmaBatchOutput, PwmaError> {
    pwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn pwma_batch_with_kernel(
    data: &[f64],
    sweep: &PwmaBatchRange,
    k: Kernel,
) -> Result<PwmaBatchOutput, PwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PwmaError::InvalidPeriod {
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
    pwma_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn pwma_batch_inner(
    data: &[f64],
    sweep: &PwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PwmaBatchOutput, PwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PwmaError::InvalidPeriod {
            period: max_p,
            data_len: data.len(),
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, rows * max_p);
    weights.resize(rows * max_p, 0.0);
    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let row_weights = pascal_weights(period)?;
        for (i, w) in row_weights.iter().enumerate() {
            weights[row * max_p + i] = *w;
        }
    }
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    let mut raw = make_uninit_matrix(rows, cols); // Vec<MaybeUninit<f64>>
    init_matrix_prefixes(&mut raw, cols, &warm); // write NaN prefixes

    // --- closure that fills a single row -------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = weights.as_ptr().add(row * max_p);

        // reinterpret this row as &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => pwma_row_scalar(data, first, period, max_p, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => pwma_row_avx2(data, first, period, max_p, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => pwma_row_avx512(data, first, period, max_p, w_ptr, out_row),
            _ => unreachable!(),
        }
    };

    // --- run the rows in parallel or serial ----------------------------------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // --- convert to fully-initialised Vec<f64> without UB --------------------
    use core::mem::ManuallyDrop;
    let mut guard = ManuallyDrop::new(raw);
    let ptr = guard.as_mut_ptr() as *mut f64;
    let len = guard.len();
    let cap = guard.capacity();
    let values: Vec<f64> = unsafe { Vec::from_raw_parts(ptr, len, cap) };

    Ok(PwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn pwma_batch_inner_into(
    data: &[f64],
    sweep: &PwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PwmaParams>, PwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PwmaError::InvalidPeriod {
            period: max_p,
            data_len: data.len(),
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Pre-compute all weights
    let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, rows * max_p);
    weights.resize(rows * max_p, 0.0);
    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let row_weights = pascal_weights(period)?;
        for (i, w) in row_weights.iter().enumerate() {
            weights[row * max_p + i] = *w;
        }
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // SAFETY: We're reinterpreting the output slice as MaybeUninit
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    init_matrix_prefixes(out_uninit, cols, &warm);

    // Closure that fills a single row
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = weights.as_ptr().add(row * max_p);

        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => pwma_row_scalar(data, first, period, max_p, w_ptr, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => pwma_row_avx2(data, first, period, max_p, w_ptr, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => pwma_row_avx512(data, first, period, max_p, w_ptr, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
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
unsafe fn pwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for k in 0..period {
            sum += *data.get_unchecked(start + k) * *w_ptr.add(k);
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn pwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    // Create a slice from the weight pointer for pwma_avx2
    let weights = std::slice::from_raw_parts(w_ptr, period);
    pwma_avx2(data, weights, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    if period <= 32 {
        pwma_row_avx512_short(data, first, period, stride, w_ptr, out);
    } else {
        pwma_row_avx512_long(data, first, period, stride, w_ptr, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
unsafe fn pwma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    let vecs = period / 8;
    let tail = period % 8;

    // Create tail mask for AVX512 masked operations
    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        // Process full 512-bit vectors
        for v in 0..vecs {
            let d = _mm512_loadu_pd(data.as_ptr().add(start + v * 8));
            let w = _mm512_loadu_pd(w_ptr.add(v * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        // Process tail with AVX512 masking
        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, w_ptr.add(vecs * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        // Use optimized horizontal reduction intrinsic
        let total = _mm512_reduce_add_pd(acc);

        out[i] = total;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
unsafe fn pwma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    let full_vecs = period / 8;
    let tail = period % 8;

    // Create tail mask
    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;

        // Use 4 accumulators for maximum ILP
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        // Prefetch next window's data
        if i + 1 < data.len() {
            _mm_prefetch(data.as_ptr().add(start + period) as *const i8, _MM_HINT_T0);
        }

        // Process vectors in groups of 4 for better ILP
        let quads = full_vecs / 4;
        let remaining = full_vecs % 4;

        for q in 0..quads {
            let base = q * 4 * 8;
            let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + base + 8));
            let d2 = _mm512_loadu_pd(data.as_ptr().add(start + base + 16));
            let d3 = _mm512_loadu_pd(data.as_ptr().add(start + base + 24));

            let w0 = _mm512_loadu_pd(w_ptr.add(base));
            let w1 = _mm512_loadu_pd(w_ptr.add(base + 8));
            let w2 = _mm512_loadu_pd(w_ptr.add(base + 16));
            let w3 = _mm512_loadu_pd(w_ptr.add(base + 24));

            acc0 = _mm512_fmadd_pd(d0, w0, acc0);
            acc1 = _mm512_fmadd_pd(d1, w1, acc1);
            acc2 = _mm512_fmadd_pd(d2, w2, acc2);
            acc3 = _mm512_fmadd_pd(d3, w3, acc3);
        }

        // Process remaining vectors
        let base = quads * 4 * 8;
        match remaining {
            3 => {
                let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
                let d1 = _mm512_loadu_pd(data.as_ptr().add(start + base + 8));
                let d2 = _mm512_loadu_pd(data.as_ptr().add(start + base + 16));
                let w0 = _mm512_loadu_pd(w_ptr.add(base));
                let w1 = _mm512_loadu_pd(w_ptr.add(base + 8));
                let w2 = _mm512_loadu_pd(w_ptr.add(base + 16));
                acc0 = _mm512_fmadd_pd(d0, w0, acc0);
                acc1 = _mm512_fmadd_pd(d1, w1, acc1);
                acc2 = _mm512_fmadd_pd(d2, w2, acc2);
            }
            2 => {
                let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
                let d1 = _mm512_loadu_pd(data.as_ptr().add(start + base + 8));
                let w0 = _mm512_loadu_pd(w_ptr.add(base));
                let w1 = _mm512_loadu_pd(w_ptr.add(base + 8));
                acc0 = _mm512_fmadd_pd(d0, w0, acc0);
                acc1 = _mm512_fmadd_pd(d1, w1, acc1);
            }
            1 => {
                let d0 = _mm512_loadu_pd(data.as_ptr().add(start + base));
                let w0 = _mm512_loadu_pd(w_ptr.add(base));
                acc0 = _mm512_fmadd_pd(d0, w0, acc0);
            }
            _ => {}
        }

        // Process tail with masking
        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + full_vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, w_ptr.add(full_vecs * 8));
            acc0 = _mm512_fmadd_pd(d, w, acc0);
        }

        // Combine all accumulators
        let acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

        // Use optimized horizontal reduction intrinsic
        let total = _mm512_reduce_add_pd(acc);

        out[i] = total;
    }
}

#[inline]
fn pascal_weights(period: usize) -> Result<Vec<f64>, PwmaError> {
    if period == 0 {
        return Err(PwmaError::InvalidPeriod {
            period,
            data_len: 0,
        });
    }
    let n = period - 1;
    let mut row = Vec::with_capacity(period);
    for r in 0..=n {
        let c = combination_f64(n, r);
        row.push(c);
    }
    let sum: f64 = row.iter().sum();
    if sum == 0.0 {
        return Err(PwmaError::PascalWeightsSumZero { period });
    }
    for val in row.iter_mut() {
        *val /= sum;
    }
    Ok(row)
}

#[inline]
fn combination_f64(n: usize, r: usize) -> f64 {
    let r = r.min(n - r);
    if r == 0 {
        return 1.0;
    }
    let mut result = 1.0;
    for i in 0..r {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ---- Tests (macro parity, batch tests, kernel detection) ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_pwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = PwmaParams { period: None };
        let input = PwmaInput::from_candles(&candles, "close", default_params);
        let output = pwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five = [59313.25, 59309.6875, 59249.3125, 59175.625, 59094.875];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PwmaInput::from_candles(&candles, "close", PwmaParams::default());
        let result = pwma_with_kernel(&input, kernel)?;
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-3,
                "[{}] PWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_pwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PwmaInput::with_default_candles(&candles);
        match input.data {
            PwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected PwmaData::Candles"),
        }
        let output = pwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = PwmaParams { period: Some(0) };
        let input = PwmaInput::from_slice(&input_data, params);
        let res = pwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_pwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = PwmaParams { period: Some(10) };
        let input = PwmaInput::from_slice(&data_small, params);
        let res = pwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_pwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = PwmaParams { period: Some(5) };
        let input = PwmaInput::from_slice(&single_point, params);
        let res = pwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_pwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = PwmaParams { period: Some(5) };
        let first_input = PwmaInput::from_candles(&candles, "close", first_params);
        let first_result = pwma_with_kernel(&first_input, kernel)?;
        let second_params = PwmaParams { period: Some(3) };
        let second_input = PwmaInput::from_slice(&first_result.values, second_params);
        let second_result = pwma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_pwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PwmaInput::from_candles(&candles, "close", PwmaParams { period: Some(5) });
        let res = pwma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_pwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = PwmaInput::from_candles(
            &candles,
            "close",
            PwmaParams {
                period: Some(period),
            },
        );
        let batch_output = pwma_with_kernel(&input, kernel)?.values;
        let mut stream = PwmaStream::try_new(PwmaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] PWMA streaming mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }

    macro_rules! generate_all_pwma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); })*
            }
        }
    }
    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_pwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations
        // PWMA typically uses smaller periods than other MAs
        let test_cases = vec![
            PwmaParams { period: Some(5) },  // default
            PwmaParams { period: Some(3) },  // minimum practical
            PwmaParams { period: Some(10) }, // medium
            PwmaParams { period: Some(15) }, // larger
            PwmaParams { period: Some(7) },  // different value
            PwmaParams { period: Some(20) }, // large for PWMA
            PwmaParams { period: Some(2) },  // very small
            PwmaParams { period: Some(12) }, // another medium
            PwmaParams { period: None },     // None value (use default)
        ];

        for params in test_cases {
            let input = PwmaInput::from_candles(&candles, "close", params);
            let output = pwma_with_kernel(&input, kernel)?;

            // Check every value for poison patterns
            for (i, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in the warmup period
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_pwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_pwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Load real market data for realistic testing
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_data = &candles.close;

        // Strategy: test various parameter combinations with real data slices
        // PWMA typically uses smaller periods due to Pascal coefficient growth
        let strat = (
            2usize..=30, // period (PWMA typically uses smaller periods)
            0usize..close_data.len().saturating_sub(200), // starting index
            100usize..=200, // length of data slice to use
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                // Ensure we have valid slice bounds
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period + 10 {
                    return Ok(()); // Skip invalid combinations
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = PwmaParams {
                    period: Some(period),
                };
                let input = PwmaInput::from_slice(data_slice, params);

                // Test the specified kernel
                let result = pwma_with_kernel(&input, kernel);

                // Also compute with scalar kernel for reference
                let scalar_result = pwma_with_kernel(&input, Kernel::Scalar);

                // Both should succeed or fail together
                match (result, scalar_result) {
                    (Ok(PwmaOutput { values: out }), Ok(PwmaOutput { values: ref_out })) => {
                        // Verify output length
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        // Find first non-NaN value
                        let first = data_slice.iter().position(|x| !x.is_nan()).unwrap_or(0);
                        let expected_warmup = first + period - 1;

                        // Check NaN pattern during warmup
                        for i in 0..expected_warmup {
                            prop_assert!(
                                out[i].is_nan(),
                                "Expected NaN at index {} during warmup, got {}",
                                i,
                                out[i]
                            );
                        }

                        // Test Pascal weight properties
                        let weights = pascal_weights(period).unwrap();

                        // Verify weights sum to 1.0 (already normalized)
                        let weight_sum: f64 = weights.iter().sum();
                        prop_assert!(
                            (weight_sum - 1.0).abs() < 1e-10,
                            "Pascal weights don't sum to 1.0: sum = {}",
                            weight_sum
                        );

                        // Verify symmetry of Pascal weights
                        for i in 0..period / 2 {
                            let diff = (weights[i] - weights[period - 1 - i]).abs();
                            prop_assert!(
                                diff < 1e-10,
                                "Pascal weights not symmetric at positions {} and {}: {} vs {}",
                                i,
                                period - 1 - i,
                                weights[i],
                                weights[period - 1 - i]
                            );
                        }

                        // Test specific properties for valid outputs
                        for i in expected_warmup..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            // Both should be valid
                            prop_assert!(!y.is_nan(), "Unexpected NaN at index {}", i);
                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

                            // Kernel consistency check
                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();

                            if !y.is_finite() || !r.is_finite() {
                                prop_assert_eq!(
                                    y_bits,
                                    r_bits,
                                    "NaN/Inf mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                                continue;
                            }

                            // ULP difference check for floating-point precision
                            let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                "Kernel mismatch at {}: {} vs {} (ULP={})",
                                i,
                                y,
                                r,
                                ulp_diff
                            );

                            // Output bounds check - PWMA output should be within window bounds
                            if i >= period - 1 {
                                let window_start = i + 1 - period;
                                let window = &data_slice[window_start..=i];
                                let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max_val =
                                    window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                                // PWMA is a weighted average, so it must be within min/max
                                prop_assert!(
                                    y >= min_val - 1e-9 && y <= max_val + 1e-9,
                                    "PWMA value {} outside window bounds [{}, {}] at index {}",
                                    y,
                                    min_val,
                                    max_val,
                                    i
                                );
                            }
                        }

                        // Test constant data property
                        let const_data = vec![42.0; period + 10];
                        let const_input = PwmaInput::from_slice(&const_data, params);
                        if let Ok(PwmaOutput { values: const_out }) =
                            pwma_with_kernel(&const_input, kernel)
                        {
                            for (i, &val) in const_out.iter().enumerate() {
                                if !val.is_nan() {
                                    prop_assert!(
										(val - 42.0).abs() < 1e-9,
										"PWMA of constant data should equal the constant at {}: got {}",
										i, val
									);
                                }
                            }
                        }
                    }
                    (Err(e1), Err(e2)) => {
                        // Both kernels should fail with similar errors
                        prop_assert_eq!(
                            std::mem::discriminant(&e1),
                            std::mem::discriminant(&e2),
                            "Different error types: {:?} vs {:?}",
                            e1,
                            e2
                        );
                    }
                    _ => {
                        prop_assert!(
                            false,
                            "Kernel consistency failure: one succeeded, one failed"
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_pwma_tests!(
        check_pwma_partial_params,
        check_pwma_accuracy,
        check_pwma_default_candles,
        check_pwma_zero_period,
        check_pwma_period_exceeds_length,
        check_pwma_very_small_dataset,
        check_pwma_reinput,
        check_pwma_nan_handling,
        check_pwma_streaming,
        check_pwma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_pwma_tests!(check_pwma_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = PwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = PwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }
    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations with different parameter ranges
        // PWMA typically uses smaller periods, so adjust ranges accordingly
        let batch_configs = vec![
            // Original test case
            (3, 10, 1), // Comprehensive small range
            // Edge cases
            (5, 5, 0),   // Single parameter (default)
            (2, 8, 2),   // Very small periods
            (10, 20, 5), // Medium periods for PWMA
            (4, 12, 4),  // Different step
            (3, 15, 3),  // Wider range
            (6, 18, 6),  // Different pattern
            (2, 10, 1),  // Full small range
        ];

        for (p_start, p_end, p_step) in batch_configs {
            let output = PwmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "pwma")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn pwma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = PwmaParams {
        period: Some(period),
    };
    let pwma_in = PwmaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| pwma_with_kernel(&pwma_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "PwmaStream")]
pub struct PwmaStreamPy {
    stream: PwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PwmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = PwmaParams {
            period: Some(period),
        };
        let stream =
            PwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PwmaStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "pwma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn pwma_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = PwmaBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
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
                _ => unreachable!(),
            };
            pwma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pwma_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, device_id=0))]
pub fn pwma_cuda_batch_dev_py(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data.as_slice()?;
    let sweep = PwmaBatchRange {
        period: period_range,
    };
    let data_f32: Vec<f32> = slice_in.iter().map(|&v| v as f32).collect();

    let inner = py.allow_threads(|| {
        let cuda = CudaPwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.pwma_batch_dev(&data_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pwma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn pwma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = PwmaParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaPwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.pwma_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pwma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = PwmaParams {
        period: Some(period),
    };
    let input = PwmaInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    pwma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pwma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = PwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    pwma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pwma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = PwmaBatchRange {
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
pub fn pwma_batch_rows_cols_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    data_len: usize,
) -> Vec<usize> {
    let sweep = PwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = data_len;

    vec![rows, cols]
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pwma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    core::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pwma_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pwma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to pwma_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Calculate PWMA
        let params = PwmaParams {
            period: Some(period),
        };
        let input = PwmaInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            pwma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            pwma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pwma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to pwma_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = PwmaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Use optimized batch processing
        pwma_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
