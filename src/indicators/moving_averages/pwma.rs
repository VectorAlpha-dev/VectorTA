#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::pwma_wrapper::DeviceArrayF32Pwma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaPwma;

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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
pub struct PrimaryCtxGuardPwma {
    dev: i32,
    ctx: cust::sys::CUcontext,
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl PrimaryCtxGuardPwma {
    fn new(device_id: u32) -> Result<Self, cust::error::CudaError> {
        unsafe {
            let mut ctx: cust::sys::CUcontext = core::ptr::null_mut();
            let dev = device_id as i32;
            let rc = cust::sys::cuDevicePrimaryCtxRetain(&mut ctx as *mut _, dev);
            if rc != cust::sys::CUresult::CUDA_SUCCESS {
                return Err(cust::error::CudaError::UnknownError);
            }
            Ok(PrimaryCtxGuardPwma { dev, ctx })
        }
    }
    #[inline]
    unsafe fn push_current(&self) {
        let _ = cust::sys::cuCtxSetCurrent(self.ctx);
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl Drop for PrimaryCtxGuardPwma {
    fn drop(&mut self) {
        unsafe {
            let _ = cust::sys::cuDevicePrimaryCtxRelease_v2(self.dev);
        }
    }
}

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
    #[error("pwma: empty input data")]
    EmptyInputData,
    #[error("pwma: All values are NaN.")]
    AllValuesNaN,
    #[error("pwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("pwma: Pascal weights sum to zero for period = {period}")]
    PascalWeightsSumZero { period: usize },
    #[error("pwma: not enough valid data: needed {needed}, valid {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("pwma: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("pwma: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("pwma: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn pwma(input: &PwmaInput) -> Result<PwmaOutput, PwmaError> {
    pwma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn pwma_prepare<'a>(
    input: &'a PwmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], Vec<f64>, usize, usize, Kernel), PwmaError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(PwmaError::EmptyInputData);
    }
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

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline(always)]
pub fn pwma_into(input: &PwmaInput, out: &mut [f64]) -> Result<(), PwmaError> {
    let (data, weights, period, first, chosen) = pwma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(PwmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let warmup_end = first + period - 1;
    let end = warmup_end.min(out.len());
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..end] {
        *v = qnan;
    }

    pwma_compute_into(data, &weights, period, first, chosen, out);

    Ok(())
}

#[inline]
pub fn pwma_into_slice(dst: &mut [f64], input: &PwmaInput, kern: Kernel) -> Result<(), PwmaError> {
    let (data, weights, period, first, chosen) = pwma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(PwmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    pwma_compute_into(data, &weights, period, first, chosen, dst);

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

    let n = data.len();
    let d_base = data.as_ptr();
    let w_ptr = weights.as_ptr();
    let o_ptr = out.as_mut_ptr();

    unsafe {
        let mut i = first + period - 1;
        while i < n {
            let start = i + 1 - period;
            let d_ptr = d_base.add(start);

            let mut s0 = 0.0f64;
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            let mut s3 = 0.0f64;

            let mut k = 0usize;
            let k_end = period & !3usize;
            while k < k_end {
                let d0 = *d_ptr.add(k + 0);
                let d1 = *d_ptr.add(k + 1);
                let d2 = *d_ptr.add(k + 2);
                let d3 = *d_ptr.add(k + 3);

                let w0 = *w_ptr.add(k + 0);
                let w1 = *w_ptr.add(k + 1);
                let w2 = *w_ptr.add(k + 2);
                let w3 = *w_ptr.add(k + 3);

                s0 = d0.mul_add(w0, s0);
                s1 = d1.mul_add(w1, s1);
                s2 = d2.mul_add(w2, s2);
                s3 = d3.mul_add(w3, s3);

                k += 4;
            }

            let mut sum = (s0 + s1) + (s2 + s3);
            while k < period {
                sum = (*d_ptr.add(k)).mul_add(*w_ptr.add(k), sum);
                k += 1;
            }

            *o_ptr.add(i) = sum;
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pwma_avx512(data: &[f64], weights: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if cfg!(target_feature = "avx512vl") || std::is_x86_feature_detected!("avx512vl") {
        unsafe { pwma_avx512_vl(data, weights, period, first, out) }
    } else if period <= 32 {
        unsafe { pwma_avx512_short(data, weights, period, first, out) }
    } else {
        unsafe { pwma_avx512_long(data, weights, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512vl,fma")]
unsafe fn pwma_avx512_vl(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let vecs = period / 4;
    let tail = period % 4;
    let tail_mask: __mmask8 = match tail {
        0 => 0,
        t => ((1u8 << t) - 1) as __mmask8,
    };

    for i in (first + period - 1)..len {
        let start = i + 1 - period;

        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();

        let pairs = vecs / 2;
        for p in 0..pairs {
            let base = p * 8;
            let d0 = _mm256_loadu_pd(data.as_ptr().add(start + base));
            let w0 = _mm256_loadu_pd(weights.as_ptr().add(base));
            acc0 = _mm256_fmadd_pd(d0, w0, acc0);

            let d1 = _mm256_loadu_pd(data.as_ptr().add(start + base + 4));
            let w1 = _mm256_loadu_pd(weights.as_ptr().add(base + 4));
            acc1 = _mm256_fmadd_pd(d1, w1, acc1);
        }

        if (vecs & 1) != 0 {
            let base = pairs * 8;
            let d = _mm256_loadu_pd(data.as_ptr().add(start + base));
            let w = _mm256_loadu_pd(weights.as_ptr().add(base));
            acc0 = _mm256_fmadd_pd(d, w, acc0);
        }

        if tail_mask != 0 {
            let d = _mm256_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + vecs * 4));
            let w = _mm256_maskz_loadu_pd(tail_mask, weights.as_ptr().add(vecs * 4));
            acc0 = _mm256_fmadd_pd(d, w, acc0);
        }

        let acc = _mm256_add_pd(acc0, acc1);
        let low128 = _mm256_castpd256_pd128(acc);
        let high128 = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(low128, high128);
        let high64 = _mm_unpackhi_pd(sum128, sum128);
        let total = _mm_cvtsd_f64(_mm_add_sd(sum128, high64));

        _mm_stream_sd(out.as_mut_ptr().add(i), _mm_set_sd(total));
    }

    _mm_sfence();
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
    let vecs = period / 4;
    let tail = period % 4;

    for i in (first + period - 1)..len {
        let start = i + 1 - period;

        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();

        let pairs = vecs / 2;
        for p in 0..pairs {
            let base = p * 8;
            let d0 = _mm256_loadu_pd(data.as_ptr().add(start + base));
            let w0 = _mm256_loadu_pd(weights.as_ptr().add(base));
            acc0 = _mm256_fmadd_pd(d0, w0, acc0);

            let d1 = _mm256_loadu_pd(data.as_ptr().add(start + base + 4));
            let w1 = _mm256_loadu_pd(weights.as_ptr().add(base + 4));
            acc1 = _mm256_fmadd_pd(d1, w1, acc1);
        }

        if (vecs & 1) != 0 {
            let base = pairs * 8;
            let d = _mm256_loadu_pd(data.as_ptr().add(start + base));
            let w = _mm256_loadu_pd(weights.as_ptr().add(base));
            acc0 = _mm256_fmadd_pd(d, w, acc0);
        }

        let acc = _mm256_add_pd(acc0, acc1);
        let low128 = _mm256_castpd256_pd128(acc);
        let high128 = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(low128, high128);
        let high64 = _mm_unpackhi_pd(sum128, sum128);
        let mut total = _mm_cvtsd_f64(_mm_add_sd(sum128, high64));

        for t in 0..tail {
            let idx = vecs * 4 + t;
            total = (*data.get_unchecked(start + idx)).mul_add(*weights.get_unchecked(idx), total);
        }

        _mm_stream_sd(out.as_mut_ptr().add(i), _mm_set_sd(total));
    }

    _mm_sfence();
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

    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..len {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        for v in 0..vecs {
            let d = _mm512_loadu_pd(data.as_ptr().add(start + v * 8));
            let w = _mm512_loadu_pd(weights.as_ptr().add(v * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(vecs * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        let total = _mm512_reduce_add_pd(acc);

        _mm_stream_sd(out.as_mut_ptr().add(i), _mm_set_sd(total));
    }

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

    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..len {
        let start = i + 1 - period;

        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        if i + 1 < len {
            _mm_prefetch(data.as_ptr().add(start + period) as *const i8, _MM_HINT_T0);
        }

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

        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + full_vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(full_vecs * 8));
            acc0 = _mm512_fmadd_pd(d, w, acc0);
        }

        let acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

        let total = _mm512_reduce_add_pd(acc);

        _mm_stream_sd(out.as_mut_ptr().add(i), _mm_set_sd(total));
    }

    _mm_sfence();
}

#[derive(Debug, Clone)]
pub struct PwmaStream {
    period: usize,
    n: usize,

    prev: Vec<f64>,

    seen: usize,

    norm: f64,
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
        let n = period.saturating_sub(1);

        let norm = fast_pow2_neg_i32(n as i32);

        let mut prev = Vec::with_capacity(n);
        prev.resize(n, f64::NAN);

        Ok(Self {
            period,
            n,
            prev,
            seen: 0,
            norm,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        if self.n == 0 {
            return Some(x);
        }

        let mut a = x;
        for p in &mut self.prev {
            let out = a + *p;
            *p = a;
            a = out;
        }

        if self.seen < self.n {
            self.seen += 1;
            return None;
        }

        Some(a * self.norm)
    }
}

#[inline(always)]
fn fast_pow2_neg_i32(e: i32) -> f64 {
    if (0..=1023).contains(&e) {
        let bits = ((1023 - e) as u64) << 52;
        f64::from_bits(bits)
    } else if (1024..=1074).contains(&e) {
        let s = e - 1023;
        let mant = 1u64 << (52 - s as u32);
        f64::from_bits(mant)
    } else {
        (2.0f64).powi(-e)
    }
}

#[derive(Clone, Debug)]
pub struct PwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for PwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 254, 1),
        }
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
        if start < end {
            if step == 0 {
                return vec![start];
            }
            return (start..=end).step_by(step).collect();
        }

        let mut v = Vec::new();
        if step == 0 {
            v.push(start);
            return v;
        }
        let mut cur = start;
        loop {
            v.push(cur);
            if cur == end {
                break;
            }
            cur = match cur.checked_sub(step) {
                Some(next) if next >= end => next,
                _ => break,
            };
        }
        v
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
    if data.is_empty() {
        return Err(PwmaError::EmptyInputData);
    }
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(PwmaError::InvalidKernelForBatch(other)),
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
    if data.is_empty() {
        return Err(PwmaError::EmptyInputData);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PwmaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PwmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let rows_x_max = rows.checked_mul(max_p).ok_or(PwmaError::InvalidRange {
        start: rows,
        end: max_p,
        step: 0,
    })?;
    let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, rows_x_max);
    weights.resize(rows_x_max, 0.0);
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

    let _rows_x_cols = rows.checked_mul(cols).ok_or(PwmaError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;
    let mut raw = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut raw, cols, &warm);

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = weights.as_ptr().add(row * max_p);

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
    if data.is_empty() {
        return Err(PwmaError::EmptyInputData);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PwmaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PwmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    if let Some(expected) = rows.checked_mul(cols) {
        if out.len() != expected {
            return Err(PwmaError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    } else {
        return Err(PwmaError::InvalidRange {
            start: rows,
            end: cols,
            step: 0,
        });
    }

    let rows_x_max = rows.checked_mul(max_p).ok_or(PwmaError::InvalidRange {
        start: rows,
        end: max_p,
        step: 0,
    })?;
    let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, rows_x_max);
    weights.resize(rows_x_max, 0.0);
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

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    init_matrix_prefixes(out_uninit, cols, &warm);

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
    let n = data.len();
    let d_base = data.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let mut i = first + period - 1;
    while i < n {
        let start = i + 1 - period;
        let d_ptr = d_base.add(start);

        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut s3 = 0.0f64;
        let mut s4 = 0.0f64;
        let mut s5 = 0.0f64;
        let mut s6 = 0.0f64;
        let mut s7 = 0.0f64;

        let mut k = 0usize;
        let k_end = period & !7usize;
        while k < k_end {
            let d0 = *d_ptr.add(k + 0);
            let d1 = *d_ptr.add(k + 1);
            let d2 = *d_ptr.add(k + 2);
            let d3 = *d_ptr.add(k + 3);
            let d4 = *d_ptr.add(k + 4);
            let d5 = *d_ptr.add(k + 5);
            let d6 = *d_ptr.add(k + 6);
            let d7 = *d_ptr.add(k + 7);

            let w0 = *w_ptr.add(k + 0);
            let w1 = *w_ptr.add(k + 1);
            let w2 = *w_ptr.add(k + 2);
            let w3 = *w_ptr.add(k + 3);
            let w4 = *w_ptr.add(k + 4);
            let w5 = *w_ptr.add(k + 5);
            let w6 = *w_ptr.add(k + 6);
            let w7 = *w_ptr.add(k + 7);

            s0 = d0.mul_add(w0, s0);
            s1 = d1.mul_add(w1, s1);
            s2 = d2.mul_add(w2, s2);
            s3 = d3.mul_add(w3, s3);
            s4 = d4.mul_add(w4, s4);
            s5 = d5.mul_add(w5, s5);
            s6 = d6.mul_add(w6, s6);
            s7 = d7.mul_add(w7, s7);

            k += 8;
        }

        let mut sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
        while k < period {
            sum = (*d_ptr.add(k)).mul_add(*w_ptr.add(k), sum);
            k += 1;
        }

        *o_ptr.add(i) = sum;
        i += 1;
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

    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        for v in 0..vecs {
            let d = _mm512_loadu_pd(data.as_ptr().add(start + v * 8));
            let w = _mm512_loadu_pd(w_ptr.add(v * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, w_ptr.add(vecs * 8));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

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

    let tail_mask: __mmask8 = if tail > 0 {
        ((1u8 << tail) - 1) as __mmask8
    } else {
        0
    };

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;

        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        if i + 1 < data.len() {
            _mm_prefetch(data.as_ptr().add(start + period) as *const i8, _MM_HINT_T0);
        }

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

        if tail_mask != 0 {
            let d = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + full_vecs * 8));
            let w = _mm512_maskz_loadu_pd(tail_mask, w_ptr.add(full_vecs * 8));
            acc0 = _mm512_fmadd_pd(d, w, acc0);
        }

        let acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

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

    #[cfg(debug_assertions)]
    fn check_pwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_cases = vec![
            PwmaParams { period: Some(5) },
            PwmaParams { period: Some(3) },
            PwmaParams { period: Some(10) },
            PwmaParams { period: Some(15) },
            PwmaParams { period: Some(7) },
            PwmaParams { period: Some(20) },
            PwmaParams { period: Some(2) },
            PwmaParams { period: Some(12) },
            PwmaParams { period: None },
        ];

        for params in test_cases {
            let input = PwmaInput::from_candles(&candles, "close", params);
            let output = pwma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

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

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_data = &candles.close;

        let strat = (
            2usize..=30,
            0usize..close_data.len().saturating_sub(200),
            100usize..=200,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period + 10 {
                    return Ok(());
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = PwmaParams {
                    period: Some(period),
                };
                let input = PwmaInput::from_slice(data_slice, params);

                let result = pwma_with_kernel(&input, kernel);

                let scalar_result = pwma_with_kernel(&input, Kernel::Scalar);

                match (result, scalar_result) {
                    (Ok(PwmaOutput { values: out }), Ok(PwmaOutput { values: ref_out })) => {
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        let first = data_slice.iter().position(|x| !x.is_nan()).unwrap_or(0);
                        let expected_warmup = first + period - 1;

                        for i in 0..expected_warmup {
                            prop_assert!(
                                out[i].is_nan(),
                                "Expected NaN at index {} during warmup, got {}",
                                i,
                                out[i]
                            );
                        }

                        let weights = pascal_weights(period).unwrap();

                        let weight_sum: f64 = weights.iter().sum();
                        prop_assert!(
                            (weight_sum - 1.0).abs() < 1e-10,
                            "Pascal weights don't sum to 1.0: sum = {}",
                            weight_sum
                        );

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

                        for i in expected_warmup..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            prop_assert!(!y.is_nan(), "Unexpected NaN at index {}", i);
                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

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

                            let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                "Kernel mismatch at {}: {} vs {} (ULP={})",
                                i,
                                y,
                                r,
                                ulp_diff
                            );

                            if i >= period - 1 {
                                let window_start = i + 1 - period;
                                let window = &data_slice[window_start..=i];
                                let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max_val =
                                    window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let batch_configs = vec![
            (3, 10, 1),
            (5, 5, 0),
            (2, 8, 2),
            (10, 20, 5),
            (4, 12, 4),
            (3, 15, 3),
            (6, 18, 6),
            (2, 10, 1),
        ];

        for (p_start, p_end, p_step) in batch_configs {
            let output = PwmaBatchBuilder::new()
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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

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

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_pwma_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data = Vec::with_capacity(256);
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..253usize {
            let x = (i as f64 * 0.07).sin() * 2.5 + (i as f64) * 0.01 + 100.0;
            data.push(x);
        }

        let input = PwmaInput::from_slice(&data, PwmaParams::default());

        let baseline = pwma(&input)?;

        let mut out = vec![0.0; data.len()];
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            pwma_into(&input, &mut out)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            pwma_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.values.len(), out.len());

        for (a, b) in baseline.values.iter().copied().zip(out.iter().copied()) {
            let both_nan = a.is_nan() && b.is_nan();
            assert!(both_nan || a == b, "mismatch: got {b:?}, expected {a:?}");
        }
        Ok(())
    }
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
) -> PyResult<PwmaDeviceArrayF32Py> {
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

    make_pwma_device_array_py(device_id, inner)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pwma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn pwma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<PwmaDeviceArrayF32Py> {
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

    make_pwma_device_array_py(device_id, inner)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32Pwma", unsendable)]
pub struct PwmaDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32Pwma,
    device_id: u32,
    pc_guard: PrimaryCtxGuardPwma,
}

#[cfg(all(feature = "python", feature = "cuda"))]
pub fn make_pwma_device_array_py(
    device_id: usize,
    inner: DeviceArrayF32Pwma,
) -> PyResult<PwmaDeviceArrayF32Py> {
    let guard = PrimaryCtxGuardPwma::new(device_id as u32)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PwmaDeviceArrayF32Py {
        inner,
        device_id: device_id as u32,
        pc_guard: guard,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl PwmaDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = &self.inner;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr_val: usize = if inner.rows == 0 || inner.cols == 0 {
            0
        } else {
            inner.device_ptr() as usize
        };
        d.set_item("data", (ptr_val, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self.device_id as i32))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        let (kdl, alloc_dev) = self.__dlpack_device__()?;
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
        let ctx_clone = self.inner.ctx.clone();
        let dev_id = self.device_id;
        let inner = core::mem::replace(
            &mut self.inner,
            DeviceArrayF32Pwma {
                buf: dummy,
                rows: 0,
                cols: 0,
                ctx: ctx_clone,
                device_id: dev_id,
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
impl Drop for PwmaDeviceArrayF32Py {
    fn drop(&mut self) {
        unsafe {
            self.pc_guard.push_current();
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pwma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = PwmaParams {
        period: Some(period),
    };
    let input = PwmaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    pwma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pwma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    core::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pwma_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pwma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to pwma_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = PwmaParams {
            period: Some(period),
        };
        let input = PwmaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            pwma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            pwma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

        pwma_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
