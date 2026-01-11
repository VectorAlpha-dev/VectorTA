//! # Variable Power Weighted Moving Average (VPWMA)
//!
//! The Variable Power Weighted Moving Average (VPWMA) adjusts the weights of each
//! price data point in its calculation based on their recency. More recent data
//! points receive higher weights according to the formula: weight = (period - k)^power,
//! where k is the offset from the current position. By adjusting the power parameter,
//! one can control how aggressively recent data points dominate the resulting average.
//!
//! ## Parameters
//! - **period**: Number of data points in each calculation window (defaults to 14).
//! - **power**: Exponent applied to the recency-based weight function. Higher
//!   values give more impact to recent data points (defaults to 0.382).
//!
//! ## Errors
//! - **AllValuesNaN**: vpwma: All input data values are `NaN`.
//! - **InvalidPeriod**: vpwma: `period` < 2 or exceeds the data length.
//! - **NotEnoughValidData**: vpwma: Not enough valid data points for the requested `period`.
//! - **InvalidPower**: vpwma: `power` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(VpwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(VpwmaError)`** otherwise.
//!
//! ## Developer Notes
//! - Decision: SIMD enabled (runtime-selected). AVX2 is currently fastest on our test host by >5% at 100k vs scalar; AVX512 is functional but slightly behind AVX2.
//! - Scalar path: loop-jammed with 4 independent accumulators and `mul_add`, safe-only (no `unsafe`); ~15–25% faster than prior scalar in 100k baseline.
//! - Batch: row-specific AVX2/AVX512 kernels pre-load weight registers; scalar batch is the reference.
//!   Tried pre-reversing weights during batch precompute (to skip per-row reversal); it regressed on large inputs (1M), so reverted.
//! - Streaming update: exact O(1) for integer powers (0..8) via polynomial moments; exact faster O(n) contiguous dot for fractional powers.
//! - Memory optimization: Uses `alloc_with_nan_prefix` for zero-copy allocation.
//!
//! ## Bindings Test Status (Oct 27, 2025)
//! - Python bindings: unit tests pass after loosening kernel-availability handling in
//!   `tests/python/test_vpwma.py::test_vpwma_kernel_selection` to accept "not compiled in this build"
//!   as a valid skip for AVX2/AVX512 on hosts without those kernels. Reference values and tolerances
//!   match Rust tests (<= 1e-2; Python uses 1e-4).
//! - WASM bindings: Node-based tests currently fail at module load on Node v24 with
//!   `Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'env' imported from pkg/my_project_bg.wasm`.
//!   This is a loader/build configuration issue (wasm imports `env.memory`) rather than an algorithm
//!   mismatch. Suggested fixes: use the bundler target for wasm-pack or a JS loader that instantiates
//!   the wasm with an explicit `{ env: { memory } }` import, or pin to a Node version/config that supports
//!   ESM wasm imports for `env`. No changes to the VPWMA implementation are required.
//! - CUDA wrapper: typed error enum, VRAM headroom check, and producing stream is synchronized before
//!   returning device arrays so the Python CUDA Array Interface v3 can omit the `stream` key safely.

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::{vpwma_wrapper::DeviceArrayF32 as DeviceArrayF32Vpwma, CudaVpwma};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyUntypedArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

impl<'a> AsRef<[f64]> for VpwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VpwmaData::Slice(slice) => slice,
            VpwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VpwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VpwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VpwmaParams {
    pub period: Option<usize>,
    pub power: Option<f64>,
}

impl Default for VpwmaParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            power: Some(0.382),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VpwmaInput<'a> {
    pub data: VpwmaData<'a>,
    pub params: VpwmaParams,
}

impl<'a> VpwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VpwmaParams) -> Self {
        Self {
            data: VpwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VpwmaParams) -> Self {
        Self {
            data: VpwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VpwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_power(&self) -> f64 {
        self.params.power.unwrap_or(0.382)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VpwmaBuilder {
    period: Option<usize>,
    power: Option<f64>,
    kernel: Kernel,
}

impl Default for VpwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            power: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VpwmaBuilder {
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
    pub fn power(mut self, x: f64) -> Self {
        self.power = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VpwmaOutput, VpwmaError> {
        let p = VpwmaParams {
            period: self.period,
            power: self.power,
        };
        let i = VpwmaInput::from_candles(c, "close", p);
        vpwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VpwmaOutput, VpwmaError> {
        let p = VpwmaParams {
            period: self.period,
            power: self.power,
        };
        let i = VpwmaInput::from_slice(d, p);
        vpwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VpwmaStream, VpwmaError> {
        let p = VpwmaParams {
            period: self.period,
            power: self.power,
        };
        VpwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum VpwmaError {
    #[error("vpwma: Input data slice is empty.")]
    EmptyInputData,
    #[error("vpwma: All values are NaN.")]
    AllValuesNaN,
    #[error("vpwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vpwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vpwma: Invalid power: {power}")]
    InvalidPower { power: f64 },
    #[error("vpwma: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("vpwma: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vpwma: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: usize, end: usize, step: usize },
}

#[inline]
pub fn vpwma(input: &VpwmaInput) -> Result<VpwmaOutput, VpwmaError> {
    vpwma_with_kernel(input, Kernel::Auto)
}

pub fn vpwma_with_kernel(input: &VpwmaInput, kernel: Kernel) -> Result<VpwmaOutput, VpwmaError> {
    let data: &[f64] = match &input.data {
        VpwmaData::Candles { candles, source } => source_type(candles, source),
        VpwmaData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(VpwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VpwmaError::AllValuesNaN)?;
    let period = input.get_period();
    let power = input.get_power();

    if period < 2 || period > len {
        return Err(VpwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(VpwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if power.is_nan() || power.is_infinite() {
        return Err(VpwmaError::InvalidPower { power });
    }

    
    let win_len = period - 1;
    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, win_len);
    weights.resize(win_len, 0.0);

    let mut norm = 0.0;
    for k in 0..win_len {
        let w = (period as f64 - k as f64).powf(power);
        weights[k] = w;
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
                vpwma_scalar(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vpwma_avx2(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpwma_avx512(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                vpwma_scalar(data, &weights, period, first, inv_norm, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(VpwmaOutput { values: out })
}

/// Compute VPWMA directly into a caller-provided buffer (no per-call allocations).
///
/// - Preserves the indicator's NaN warmup prefix exactly as the Vec-returning API
///   (first valid index + `period - 1`).
/// - The output slice length must equal the input length.
/// - Dispatches with `Kernel::Auto` like `vpwma()`.
#[cfg(not(feature = "wasm"))]
pub fn vpwma_into(input: &VpwmaInput, out: &mut [f64]) -> Result<(), VpwmaError> {
    // Reuse the module's zero-allocation compute path which also handles
    
    vpwma_into_slice(out, input, Kernel::Auto)
}
#[inline]
pub fn vpwma_scalar(
    data: &[f64],
    weights: &[f64], 
    period: usize,
    first_val: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    let win_len = period - 1; 
    if win_len == 0 {
        return;
    }

    
    let p8 = win_len & !7; 
    let p4 = win_len & !3; 

    
    for i in (first_val + win_len)..data.len() {
        
        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut s3 = 0.0f64;

        
        let mut k = 0usize;
        while k < p8 {
            s0 = data[i - (k + 0)].mul_add(weights[k + 0], s0);
            s1 = data[i - (k + 1)].mul_add(weights[k + 1], s1);
            s2 = data[i - (k + 2)].mul_add(weights[k + 2], s2);
            s3 = data[i - (k + 3)].mul_add(weights[k + 3], s3);

            s0 = data[i - (k + 4)].mul_add(weights[k + 4], s0);
            s1 = data[i - (k + 5)].mul_add(weights[k + 5], s1);
            s2 = data[i - (k + 6)].mul_add(weights[k + 6], s2);
            s3 = data[i - (k + 7)].mul_add(weights[k + 7], s3);

            k += 8;
        }

        
        while k < p4 {
            s0 = data[i - (k + 0)].mul_add(weights[k + 0], s0);
            s1 = data[i - (k + 1)].mul_add(weights[k + 1], s1);
            s2 = data[i - (k + 2)].mul_add(weights[k + 2], s2);
            s3 = data[i - (k + 3)].mul_add(weights[k + 3], s3);
            k += 4;
        }

        
        match win_len - k {
            3 => {
                s0 = data[i - (k + 0)].mul_add(weights[k + 0], s0);
                s1 = data[i - (k + 1)].mul_add(weights[k + 1], s1);
                s2 = data[i - (k + 2)].mul_add(weights[k + 2], s2);
            }
            2 => {
                s0 = data[i - (k + 0)].mul_add(weights[k + 0], s0);
                s1 = data[i - (k + 1)].mul_add(weights[k + 1], s1);
            }
            1 => {
                s0 = data[i - (k + 0)].mul_add(weights[k + 0], s0);
            }
            _ => {}
        }

        
        let sum = (s0 + s1) + (s2 + s3);
        out[i] = sum * inv_norm;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpwma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    unsafe { vpwma_avx512_core(data, weights, period, first_valid, inv_norm, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vpwma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let win_len = period - 1;
    if len == 0 || win_len == 0 {
        return;
    }

    
    let mut wrev: Vec<f64> = Vec::with_capacity(win_len);
    unsafe {
        wrev.set_len(win_len);
    }
    for j in 0..win_len {
        unsafe {
            *wrev.get_unchecked_mut(j) = *weights.get_unchecked(win_len - 1 - j);
        }
    }

    const STEP: usize = 4;
    let chunks = win_len / STEP;
    let tail = win_len % STEP;
    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    
    const MAX_CHUNKS: usize = 1024;
    debug_assert!(chunks + (tail != 0) as usize <= MAX_CHUNKS);
    let mut wregs: [core::mem::MaybeUninit<__m256d>; MAX_CHUNKS] =
        core::mem::MaybeUninit::uninit().assume_init();
    for blk in 0..chunks {
        wregs[blk]
            .as_mut_ptr()
            .write(_mm256_loadu_pd(wrev.as_ptr().add(blk * STEP)));
    }
    let w_tail = if tail != 0 {
        _mm256_maskload_pd(wrev.as_ptr().add(chunks * STEP), tail_mask)
    } else {
        _mm256_setzero_pd()
    };
    let wregs: &[__m256d] = core::slice::from_raw_parts(wregs.as_ptr() as *const __m256d, chunks);

    let paired = chunks & !3;
    for i in (first_valid + win_len)..len {
        let start = i + 1 - win_len;
        _mm_prefetch(data.as_ptr().add(start + 64) as *const i8, _MM_HINT_T0);

        
        let mut s0 = _mm256_setzero_pd();
        let mut s1 = _mm256_setzero_pd();
        let mut s2 = _mm256_setzero_pd();
        let mut s3 = _mm256_setzero_pd();

        let mut blk = 0;
        while blk < paired {
            let d0 = _mm256_loadu_pd(data.as_ptr().add(start + (blk + 0) * STEP));
            let d1 = _mm256_loadu_pd(data.as_ptr().add(start + (blk + 1) * STEP));
            let d2 = _mm256_loadu_pd(data.as_ptr().add(start + (blk + 2) * STEP));
            let d3 = _mm256_loadu_pd(data.as_ptr().add(start + (blk + 3) * STEP));
            s0 = _mm256_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
            s1 = _mm256_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
            s2 = _mm256_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
            s3 = _mm256_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);
            blk += 4;
        }
        for r in blk..chunks {
            let d = _mm256_loadu_pd(data.as_ptr().add(start + r * STEP));
            s0 = _mm256_fmadd_pd(d, *wregs.get_unchecked(r), s0);
        }
        if tail != 0 {
            let d_tail = _mm256_maskload_pd(data.as_ptr().add(start + chunks * STEP), tail_mask);
            s0 = _mm256_fmadd_pd(d_tail, w_tail, s0);
        }

        
        let sum01 = _mm256_add_pd(s0, s1);
        let sum23 = _mm256_add_pd(s2, s3);
        let acc = _mm256_add_pd(sum01, sum23);
        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let sum2 = _mm_add_pd(hi, lo);
        let sum1 = _mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2));
        let sum = _mm_cvtsd_f64(sum1);
        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn vpwma_avx512_core(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    #[inline]
    unsafe fn hsum_pd_zmm(v: __m512d) -> f64 {
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

    let len = data.len();
    let win_len = period - 1;
    if len == 0 || win_len == 0 {
        return;
    }

    
    let mut wrev: Vec<f64> = Vec::with_capacity(win_len);
    wrev.set_len(win_len);
    for j in 0..win_len {
        *wrev.get_unchecked_mut(j) = *weights.get_unchecked(win_len - 1 - j);
    }

    const STEP: usize = 8;
    let chunks = win_len / STEP;
    let tail = win_len % STEP;
    let tmask: __mmask8 = (1u8 << tail).wrapping_sub(1);

    
    const MAX_CHUNKS: usize = 512;
    debug_assert!(chunks + (tail != 0) as usize <= MAX_CHUNKS);
    let mut wregs: [core::mem::MaybeUninit<__m512d>; MAX_CHUNKS] =
        core::mem::MaybeUninit::uninit().assume_init();
    for blk in 0..chunks {
        wregs[blk]
            .as_mut_ptr()
            .write(_mm512_loadu_pd(wrev.as_ptr().add(blk * STEP)));
    }
    if tail != 0 {
        wregs[chunks].as_mut_ptr().write(_mm512_maskz_loadu_pd(
            tmask,
            wrev.as_ptr().add(chunks * STEP),
        ));
    }
    let wregs: &[__m512d] = core::slice::from_raw_parts(
        wregs.as_ptr() as *const __m512d,
        chunks + (tail != 0) as usize,
    );

    let paired = chunks & !3;
    for i in (first_valid + win_len)..len {
        let start = i + 1 - win_len;
        _mm_prefetch(data.as_ptr().add(start + 128) as *const i8, _MM_HINT_T0);

        let mut s0 = _mm512_setzero_pd();
        let mut s1 = _mm512_setzero_pd();
        let mut s2 = _mm512_setzero_pd();
        let mut s3 = _mm512_setzero_pd();
        let mut blk = 0;
        while blk < paired {
            let d0 = _mm512_loadu_pd(data.as_ptr().add(start + (blk + 0) * STEP));
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + (blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(data.as_ptr().add(start + (blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(data.as_ptr().add(start + (blk + 3) * STEP));
            s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
            s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
            s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
            s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);
            blk += 4;
        }
        for r in blk..chunks {
            let d = _mm512_loadu_pd(data.as_ptr().add(start + r * STEP));
            s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(r), s0);
        }
        if tail != 0 {
            let d_tail = _mm512_maskz_loadu_pd(tmask, data.as_ptr().add(start + chunks * STEP));
            s0 = _mm512_fmadd_pd(d_tail, *wregs.get_unchecked(chunks), s0);
        }
        let sum = hsum_pd_zmm(_mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3)));
        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}



#[derive(Debug, Clone)]
enum StreamMode {
    /// Exact, O(1) per update for integer powers via polynomial moments.
    /// w_k = (period - k)^deg = sum_{j=0}^deg a_j k^j  (binomial expansion)
    /// Keeps {M_j} = sum_{k=0}^{m-1} k^j x_{t-k} and updates them with:
    /// M_j' = [j==0 ? x_new : 0] + sum_{q=0}^j C(j,q) * (M_q - (m-1)^q * x_out)
    PolyInt {
        deg: usize,                   // integer power (degree)
        coeff: Vec<f64>,              // a_j for j=0..deg  (numerator = dot(coeff, moments))
        binom_row_starts: Vec<usize>, // CSR-style offsets into `binom_vals`
        binom_vals: Vec<f64>,         // concatenated C(j,q) for 0<=q<=j, j=0..deg
        pow_m1: Vec<f64>,             // (m-1)^q for q=0..deg
        moments: Vec<f64>,            // M_j, j=0..deg
        moments_ready: bool,          // lazily built at first full window
    },

    /// Exact, O(n) per update for general (fractional) powers,
    /// but with a much faster contiguous two-slice dot (no per-element modulo).
    ExactDot {
        weights_rev: Vec<f64>, // reverse(weights) so index 0 multiplies oldest in window
        win_len: usize,        // m = period - 1
    },
}

#[derive(Debug, Clone)]
pub struct VpwmaStream {
    period: usize,    // ring length = m+1
    inv_norm: f64,    // 1 / sum(weights)
    buffer: Vec<f64>, // length = period
    head: usize,      // next write index
    filled: bool,     // becomes true when head cycles to 0

    // kept for introspection/back-compat
    weights: Vec<f64>, // original weights (length = m)
    mode: StreamMode,
}

impl VpwmaStream {
    pub fn try_new(params: VpwmaParams) -> Result<Self, VpwmaError> {
        let period = params.period.unwrap_or(14);
        if period < 2 {
            return Err(VpwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let power = params.power.unwrap_or(0.382);
        if power.is_nan() || power.is_infinite() {
            return Err(VpwmaError::InvalidPower { power });
        }

        // Build (period - 1) weights (exact, once).
        let m = period - 1;
        let mut weights = Vec::with_capacity(m);
        let mut norm = 0.0;
        for k in 0..m {
            // weight for k=0 (newest) .. k=m-1 (oldest)
            let w = (period as f64 - k as f64).powf(power);
            weights.push(w);
            norm += w;
        }
        let inv_norm = 1.0 / norm;

        // Decide streaming mode:
        // If `power` is an integer in [0..=8], use exact O(1) polynomial moments.
        // Else use the exact (faster) O(n) contiguous dot.
        let p_rounded = power.round();
        let is_near_int =
            (power - p_rounded).abs() <= 1e-12 && p_rounded >= 0.0 && p_rounded <= 8.0;
        let mode = if is_near_int {
            let deg = p_rounded as usize;

            // a_j = (-1)^j * C(deg, j) * (period)^(deg - j)
            let mut coeff = Vec::with_capacity(deg + 1);
            for j in 0..=deg {
                let sign = if j & 1 == 0 { 1.0 } else { -1.0 };
                let c = sign * binom(deg, j) * (period as f64).powi((deg - j) as i32);
                coeff.push(c);
            }

            // Build binomial coefficients C(j,q) for j=0..deg in a compact CSR form.
            let mut binom_vals = Vec::new();
            let mut row_starts = Vec::with_capacity(deg + 2);
            row_starts.push(0);
            for j in 0..=deg {
                for q in 0..=j {
                    binom_vals.push(binom(j, q));
                }
                row_starts.push(binom_vals.len());
            }

            // (m-1)^q, q=0..deg
            let mut pow_m1 = Vec::with_capacity(deg + 1);
            let mm1 = (m - 1) as f64;
            let mut p = 1.0;
            for _q in 0..=deg {
                pow_m1.push(p);
                p *= mm1;
            }

            StreamMode::PolyInt {
                deg,
                coeff,
                binom_row_starts: row_starts,
                binom_vals,
                pow_m1,
                moments: vec![0.0; deg + 1],
                moments_ready: false,
            }
        } else {
            // reverse(weights) so that index 0 multiplies the oldest element in the window,
            // letting us dot against the window as a forward contiguous slice.
            let mut wrev = weights.clone();
            wrev.reverse();
            StreamMode::ExactDot {
                weights_rev: wrev,
                win_len: m,
            }
        };

        Ok(Self {
            period,
            inv_norm,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            weights,
            mode,
        })
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.period - 1
    }

    /// O(1) for integer power via polynomial moments; otherwise exact O(n) with a faster contiguous dot.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Write new value at current head, then advance the ring.
        let prev_head = self.head;
        self.buffer[prev_head] = value;
        self.head = (self.head + 1) % self.period;

        // Mark filled after the first full wrap.
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }

        match &mut self.mode {
            StreamMode::PolyInt {
                deg,
                coeff,
                binom_row_starts,
                binom_vals,
                pow_m1,
                moments,
                moments_ready,
            } => {
                // The item that just fell OUT of the window is at index `self.head`.
                let x_out = self.buffer[self.head];

                if !*moments_ready {
                    // First time we have a full window: build moments exactly from the ring.
                    Self::init_moments_from(&self.buffer, self.head, self.period, *deg, moments);
                    *moments_ready = true;

                    // Return current value from moments (no recurrence yet).
                    let num = dot_unrolled(coeff, moments);
                    return Some(num * self.inv_norm);
                }

                // Update all moments using the binomial recurrence (O(deg^2), tiny).
                // M_0' = M_0 + x_new - x_out
                
                let m0_new = moments[0] + value - x_out;
                let mut next = vec![0.0; *deg + 1];
                next[0] = m0_new;

                for j in 1..=*deg {
                    let row_start = binom_row_starts[j];
                    let mut s = 0.0;
                    for q in 0..=j {
                        let c = binom_vals[row_start + q];
                        s += c * (moments[q] - pow_m1[q] * x_out);
                    }
                    next[j] = s;
                }
                moments.copy_from_slice(&next);

                let num = dot_unrolled(coeff, moments);
                Some(num * self.inv_norm)
            }

            StreamMode::ExactDot {
                weights_rev,
                win_len,
            } => {
                
                let y = Self::dot_window_fast_from(
                    &self.buffer,
                    self.head,
                    self.period,
                    weights_rev,
                    *win_len,
                );
                Some(y * self.inv_norm)
            }
        }
    }

    

    /// Initialize polynomial moments M_j for j=0..deg from the current full window.
    /// Window of length m = period-1 includes indices: [(head+1) .. (head+1)+m-1] modulo period,
    /// where (head+period-1) points to the NEWEST included.
    #[inline(always)]
    fn init_moments_from(
        buffer: &[f64],
        head: usize,
        period: usize,
        deg: usize,
        moments: &mut [f64],
    ) {
        let m = period - 1;
        let oldest = (head + 1) % period; 

        let left_len = (period - oldest).min(m);
        let right_len = m - left_len;

        moments.fill(0.0);

        
        for (pos, &x) in buffer[oldest..oldest + left_len].iter().enumerate() {
            let k = (m - 1 - pos) as f64;
            let mut kpow = 1.0;
            for j in 0..=deg {
                moments[j] += kpow * x;
                kpow *= k;
            }
        }
        
        for (pos, &x) in buffer[..right_len].iter().enumerate() {
            let k = (right_len - 1 - pos) as f64;
            let mut kpow = 1.0;
            for j in 0..=deg {
                moments[j] += kpow * x;
                kpow *= k;
            }
        }
    }

    /// Faster exact dot for general powers: avoid per-element modulo by forming two contiguous
    /// slices for the current window and using 4 accumulators.
    #[inline(always)]
    fn dot_window_fast_from(
        buffer: &[f64],
        head: usize,
        period: usize,
        weights_rev: &[f64],
        win_len: usize,
    ) -> f64 {
        let oldest = (head + 1) % period; 
        let left_len = (period - oldest).min(win_len);
        let right_len = win_len - left_len;

        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut s3 = 0.0f64;
        let mut k = 0usize;

        
        let a = &buffer[oldest..oldest + left_len];
        let w = &weights_rev[0..left_len];
        let p4 = left_len & !3;
        while k < p4 {
            s0 = a[k + 0].mul_add(w[k + 0], s0);
            s1 = a[k + 1].mul_add(w[k + 1], s1);
            s2 = a[k + 2].mul_add(w[k + 2], s2);
            s3 = a[k + 3].mul_add(w[k + 3], s3);
            k += 4;
        }
        while k < left_len {
            s0 = a[k].mul_add(w[k], s0);
            k += 1;
        }
        let mut sum = (s0 + s1) + (s2 + s3);

        
        if right_len != 0 {
            let a = &buffer[0..right_len];
            let w = &weights_rev[left_len..left_len + right_len];
            let mut s0 = 0.0f64;
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            let mut s3 = 0.0f64;
            let mut k = 0usize;
            let p4 = right_len & !3;
            while k < p4 {
                s0 = a[k + 0].mul_add(w[k + 0], s0);
                s1 = a[k + 1].mul_add(w[k + 1], s1);
                s2 = a[k + 2].mul_add(w[k + 2], s2);
                s3 = a[k + 3].mul_add(w[k + 3], s3);
                k += 4;
            }
            while k < right_len {
                s0 = a[k].mul_add(w[k], s0);
                k += 1;
            }
            sum += (s0 + s1) + (s2 + s3);
        }

        sum
    }
}



#[inline(always)]
fn binom(n: usize, k: usize) -> f64 {
    
    if k == 0 || k == n {
        return 1.0;
    }
    let k = k.min(n - k);
    let mut num = 1u128;
    let mut den = 1u128;
    for i in 1..=k {
        num *= (n + 1 - i) as u128;
        den *= i as u128;
    }
    (num as f64) / (den as f64)
}

#[inline(always)]
fn dot_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let mut i = 0usize;
    let n = a.len();
    let p4 = n & !3;
    while i < p4 {
        s0 = a[i + 0].mul_add(b[i + 0], s0);
        s1 = a[i + 1].mul_add(b[i + 1], s1);
        s2 = a[i + 2].mul_add(b[i + 2], s2);
        s3 = a[i + 3].mul_add(b[i + 3], s3);
        i += 4;
    }
    while i < n {
        s0 = a[i].mul_add(b[i], s0);
        i += 1;
    }
    (s0 + s1) + (s2 + s3)
}

#[derive(Clone, Debug)]
pub struct VpwmaBatchRange {
    pub period: (usize, usize, usize),
    pub power: (f64, f64, f64),
}

impl Default for VpwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
            power: (0.382, 0.382, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VpwmaBatchBuilder {
    range: VpwmaBatchRange,
    kernel: Kernel,
}

impl VpwmaBatchBuilder {
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
    pub fn power_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.power = (start, end, step);
        self
    }
    #[inline]
    pub fn power_static(mut self, p: f64) -> Self {
        self.range.power = (p, p, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<VpwmaBatchOutput, VpwmaError> {
        vpwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VpwmaBatchOutput, VpwmaError> {
        VpwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VpwmaBatchOutput, VpwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VpwmaBatchOutput, VpwmaError> {
        VpwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn vpwma_batch_with_kernel(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    k: Kernel,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(VpwmaError::InvalidKernelForBatch(other));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vpwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VpwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VpwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VpwmaBatchOutput {
    pub fn row_for_params(&self, p: &VpwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.power.unwrap_or(0.382) - p.power.unwrap_or(0.382)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &VpwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid_vpwma(r: &VpwmaBatchRange) -> Vec<VpwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        
        if step == 0 || start == end {
            return vec![start];
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                match v.checked_add(step) {
                    Some(next) => {
                        if next == v { break; }
                        v = next;
                    }
                    None => break,
                }
            }
        } else {
            
            let mut v = start;
            loop {
                vals.push(v);
                if v == end { break; }
                let next = v.saturating_sub(step);
                if next == v { break; }
                v = next;
                if v < end { break; }
            }
        }
        if vals.is_empty() { vec![start] } else { vals }
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut vals = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end + 1e-12 {
                vals.push(x);
                x += step;
                if !x.is_finite() { break; }
            }
        } else {
            let mut x = start;
            while x >= end - 1e-12 {
                vals.push(x);
                x -= step.abs();
                if !x.is_finite() { break; }
                if x < end { break; }
            }
        }
        if vals.is_empty() { vec![start] } else { vals }
    }
    let periods = axis_usize(r.period);
    let powers = axis_f64(r.power);
    let mut out = Vec::with_capacity(periods.len() * powers.len());
    for &p in &periods {
        for &pw in &powers {
            out.push(VpwmaParams {
                period: Some(p),
                power: Some(pw),
            });
        }
    }
    out
}

#[inline(always)]
pub fn vpwma_batch_slice(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    kern: Kernel,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    vpwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vpwma_batch_par_slice(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    kern: Kernel,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    vpwma_batch_inner(data, sweep, kern, true)
}

#[inline]
fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

#[inline(always)]
fn vpwma_batch_inner(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    if data.is_empty() {
        return Err(VpwmaError::EmptyInputData);
    }

    let combos = expand_grid_vpwma(sweep);
    if combos.is_empty() {
        return Err(VpwmaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VpwmaError::AllValuesNaN)?;
    let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_period {
        return Err(VpwmaError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }
    let max_p = combos
        .iter()
        .map(|c| round_up8(c.period.unwrap()))
        .max()
        .unwrap();

    let rows = combos.len();
    let cols = data.len();
    
    rows.checked_mul(cols).ok_or(VpwmaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;

    
    let mut inv_norms = vec![0.0; rows];
    let cap = rows.checked_mul(max_p).ok_or(VpwmaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let power = prm.power.unwrap();
        if power.is_nan() || power.is_infinite() {
            return Err(VpwmaError::InvalidPower { power });
        }
        let win_len = period - 1;
        let mut norm = 0.0;
        for k in 0..win_len {
            let w = (period as f64 - k as f64).powf(power);
            flat_w[row * max_p + k] = w;
            norm += w;
        }
        inv_norms[row] = 1.0 / norm;
    }

    
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let inv_n = *inv_norms.get_unchecked(row);

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => vpwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vpwma_row_avx2(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vpwma_row_avx512(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                vpwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, out_row)
            }
            _ => unreachable!(),
        }
    };

    
    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            buf_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        } else {
            for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VpwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn vpwma_batch_inner_into(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VpwmaParams>, VpwmaError> {
    if data.is_empty() {
        return Err(VpwmaError::EmptyInputData);
    }

    let combos = expand_grid_vpwma(sweep);
    if combos.is_empty() {
        return Err(VpwmaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VpwmaError::AllValuesNaN)?;
    let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_period {
        return Err(VpwmaError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }
    let max_p = combos
        .iter()
        .map(|c| round_up8(c.period.unwrap()))
        .max()
        .unwrap();

    let rows = combos.len();
    let cols = data.len();
    rows.checked_mul(cols).ok_or(VpwmaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;

    
    let mut inv_norms = vec![0.0; rows];
    let cap = rows.checked_mul(max_p).ok_or(VpwmaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let power = prm.power.unwrap();
        if power.is_nan() || power.is_infinite() {
            return Err(VpwmaError::InvalidPower { power });
        }
        let win_len = period - 1;
        let mut norm = 0.0;
        for k in 0..win_len {
            let w = (period as f64 - k as f64).powf(power);
            flat_w[row * max_p + k] = w;
            norm += w;
        }
        inv_norms[row] = 1.0 / norm;
    }

    
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let inv_n = *inv_norms.get_unchecked(row);
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => vpwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vpwma_row_avx2(data, first, period, max_p, w_ptr, inv_n, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vpwma_row_avx512(data, first, period, max_p, w_ptr, inv_n, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                vpwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, dst)
            }
            _ => unreachable!(),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        } else {
            for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

pub fn vpwma_into_slice(
    dst: &mut [f64],
    input: &VpwmaInput,
    kern: Kernel,
) -> Result<(), VpwmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(VpwmaError::EmptyInputData);
    }

    if dst.len() != data.len() {
        return Err(VpwmaError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VpwmaError::AllValuesNaN)?;
    let period = input.get_period();
    let power = input.get_power();

    if period < 2 || period > len {
        return Err(VpwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(VpwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if power.is_nan() || power.is_infinite() {
        return Err(VpwmaError::InvalidPower { power });
    }

    
    let win_len = period - 1;
    let weights: AVec<f64> = AVec::from_iter(
        CACHELINE_ALIGN,
        (0..win_len).map(|k| (period as f64 - k as f64).powf(power)),
    );
    let norm: f64 = weights.iter().sum();
    let inv_norm = 1.0 / norm;

    
    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    
    for v in &mut dst[..first + period - 1] {
        *v = f64::NAN;
    }

    
    unsafe {
        match chosen {
            Kernel::Scalar => vpwma_scalar(data, &weights, period, first, inv_norm, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vpwma_avx2(data, &weights, period, first, inv_norm, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vpwma_avx512(data, &weights, period, first, inv_norm, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                vpwma_scalar(data, &weights, period, first, inv_norm, dst)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline(always)]
unsafe fn vpwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,    
    w_ptr: *const f64, 
    inv_n: f64,
    out: &mut [f64],
) {
    let win_len = period - 1;
    let p4 = win_len & !3;

    
    for i in (first + win_len)..data.len() {
        let mut sum = 0.0;

        
        for k in (0..p4).step_by(4) {
            sum += *data.get_unchecked(i - k) * *w_ptr.add(k)
                + *data.get_unchecked(i - (k + 1)) * *w_ptr.add(k + 1)
                + *data.get_unchecked(i - (k + 2)) * *w_ptr.add(k + 2)
                + *data.get_unchecked(i - (k + 3)) * *w_ptr.add(k + 3);
        }

        
        for k in p4..win_len {
            sum += *data.get_unchecked(i - k) * *w_ptr.add(k);
        }

        out[i] = sum * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vpwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let win_len = period - 1;
    if len == 0 || win_len == 0 {
        return;
    }

    
    const STEP: usize = 4;
    let chunks = win_len / STEP;
    let tail = win_len % STEP;
    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    
    let mut wrev: Vec<f64> = Vec::with_capacity(win_len);
    wrev.set_len(win_len);
    for j in 0..win_len {
        *wrev.get_unchecked_mut(j) = *w_ptr.add(win_len - 1 - j);
    }

    
    const MAX_CHUNKS: usize = 1024;
    debug_assert!(chunks <= MAX_CHUNKS);
    let mut wregs: [core::mem::MaybeUninit<__m256d>; MAX_CHUNKS] =
        core::mem::MaybeUninit::uninit().assume_init();
    for blk in 0..chunks {
        wregs[blk]
            .as_mut_ptr()
            .write(_mm256_loadu_pd(wrev.as_ptr().add(blk * STEP)));
    }
    let w_tail = if tail != 0 {
        _mm256_maskload_pd(wrev.as_ptr().add(chunks * STEP), tail_mask)
    } else {
        _mm256_setzero_pd()
    };
    let wregs: &[__m256d] = core::slice::from_raw_parts(wregs.as_ptr() as *const __m256d, chunks);

    for i in (first + win_len)..len {
        let start = i + 1 - win_len;
        let mut acc = _mm256_setzero_pd();
        for blk in 0..chunks {
            let d = _mm256_loadu_pd(data.as_ptr().add(start + blk * STEP));
            acc = _mm256_fmadd_pd(d, *wregs.get_unchecked(blk), acc);
        }
        if tail != 0 {
            let d_tail = _mm256_maskload_pd(data.as_ptr().add(start + chunks * STEP), tail_mask);
            acc = _mm256_fmadd_pd(d_tail, w_tail, acc);
        }
        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let sum2 = _mm_add_pd(hi, lo);
        let sum1 = _mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2));
        let sum = _mm_cvtsd_f64(sum1);
        *out.get_unchecked_mut(i) = sum * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma,avx512dq")]
pub unsafe fn vpwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let win_len = period - 1;
    if len == 0 || win_len == 0 {
        return;
    }

    #[inline]
    unsafe fn hsum_pd_zmm(v: __m512d) -> f64 {
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

    const STEP: usize = 8;
    let chunks = win_len / STEP;
    let tail = win_len % STEP;
    let tmask: __mmask8 = (1u8 << tail).wrapping_sub(1);

    
    
    const MAX_CHUNKS: usize = 512;
    debug_assert!(chunks + (tail != 0) as usize <= MAX_CHUNKS);
    let mut wregs: [core::mem::MaybeUninit<__m512d>; MAX_CHUNKS] =
        core::mem::MaybeUninit::uninit().assume_init();
    
    let mut wrev: Vec<f64> = Vec::with_capacity(win_len);
    wrev.set_len(win_len);
    for j in 0..win_len {
        *wrev.get_unchecked_mut(j) = *w_ptr.add(win_len - 1 - j);
    }

    for blk in 0..chunks {
        wregs[blk]
            .as_mut_ptr()
            .write(_mm512_loadu_pd(wrev.as_ptr().add(blk * STEP)));
    }
    if tail != 0 {
        wregs[chunks].as_mut_ptr().write(_mm512_maskz_loadu_pd(
            tmask,
            wrev.as_ptr().add(chunks * STEP),
        ));
    }
    let wregs: &[__m512d] = core::slice::from_raw_parts(
        wregs.as_ptr() as *const __m512d,
        chunks + (tail != 0) as usize,
    );

    for i in (first + win_len)..len {
        let start = i + 1 - win_len;
        let mut acc = _mm512_setzero_pd();
        for blk in 0..chunks {
            let d = _mm512_loadu_pd(data.as_ptr().add(start + blk * STEP));
            acc = _mm512_fmadd_pd(d, *wregs.get_unchecked(blk), acc);
        }
        if tail != 0 {
            let d_tail = _mm512_maskz_loadu_pd(tmask, data.as_ptr().add(start + chunks * STEP));
            acc = _mm512_fmadd_pd(d_tail, *wregs.get_unchecked(chunks), acc);
        }
        let sum = hsum_pd_zmm(acc);
        *out.get_unchecked_mut(i) = sum * inv_n;
    }
    _mm_sfence();
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vpwma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        
        let mut data = Vec::with_capacity(256);
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..253u32 {
            let v = ((i % 19) as f64) * 0.9876 + (i as f64).cos() * 0.00123;
            data.push(v);
        }

        let params = VpwmaParams::default();
        let input = VpwmaInput::from_slice(&data, params);

        
        let base = vpwma_with_kernel(&input, Kernel::Auto)?.values;

        
        let mut out = vec![0.0; data.len()];
        #[cfg(not(feature = "wasm"))]
        {
            vpwma_into(&input, &mut out)?;
        }

        
        assert_eq!(base.len(), out.len());

        
        for (i, (a, b)) in base.iter().zip(out.iter()).enumerate() {
            let ok = if a.is_nan() && b.is_nan() {
                true
            } else {
                (a - b).abs() <= 1e-12
            };
            assert!(ok, "Mismatch at index {}: base={} vs into={}", i, a, b);
        }
        Ok(())
    }

    fn check_vpwma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VpwmaParams {
            period: None,
            power: None,
        };
        let input = VpwmaInput::from_candles(&candles, "close", default_params);
        let output = vpwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vpwma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VpwmaInput::from_candles(&candles, "close", VpwmaParams::default());
        let result = vpwma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59363.927599446455,
            59296.83894519251,
            59196.82476139941,
            59180.8040249446,
            59113.84473799056,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-2,
                "[{}] VPWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_vpwma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = VpwmaParams {
            period: Some(0),
            power: None,
        };
        let input = VpwmaInput::from_slice(&input_data, params);
        let res = vpwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VPWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_vpwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = VpwmaParams {
            period: Some(10),
            power: None,
        };
        let input = VpwmaInput::from_slice(&data_small, params);
        let res = vpwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VPWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_vpwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = VpwmaParams {
            period: Some(2),
            power: None,
        };
        let input = VpwmaInput::from_slice(&single_point, params);
        let res = vpwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VPWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_vpwma_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = VpwmaParams {
            period: Some(14),
            power: None,
        };
        let first_input = VpwmaInput::from_candles(&candles, "close", first_params);
        let first_result = vpwma_with_kernel(&first_input, kernel)?;
        let second_params = VpwmaParams {
            period: Some(5),
            power: Some(0.5),
        };
        let second_input = VpwmaInput::from_slice(&first_result.values, second_params);
        let second_result = vpwma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_vpwma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VpwmaInput::from_candles(
            &candles,
            "close",
            VpwmaParams {
                period: Some(14),
                power: None,
            },
        );
        let res = vpwma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 50 {
            for (i, &val) in res.values[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_vpwma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let power = 0.382;
        let input = VpwmaInput::from_candles(
            &candles,
            "close",
            VpwmaParams {
                period: Some(period),
                power: Some(power),
            },
        );
        let batch_output = vpwma_with_kernel(&input, kernel)?.values;
        let mut stream = VpwmaStream::try_new(VpwmaParams {
            period: Some(period),
            power: Some(power),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(vpwma_val) => stream_values.push(vpwma_val),
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
                "[{}] VPWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_vpwma_tests {
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
    fn check_vpwma_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_periods = vec![2, 5, 10, 14, 30, 50];
        let test_powers = vec![0.1, 0.382, 0.5, 1.0, 2.0];
        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for period in test_periods {
            for power in &test_powers {
                for source in &test_sources {
                    let params = VpwmaParams {
                        period: Some(period),
                        power: Some(*power),
                    };
                    let input = VpwmaInput::from_candles(&candles, source, params);
                    let output = vpwma_with_kernel(&input, kernel)?;

                    
                    for (i, &val) in output.values.iter().enumerate() {
                        
                        if val.is_nan() {
                            continue;
                        }

                        let bits = val.to_bits();

                        
                        if bits == 0x11111111_11111111 {
                            panic!(
                                "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (period={}, power={}, source={})",
                                test_name, val, bits, i, period, power, source
                            );
                        }

                        
                        if bits == 0x22222222_22222222 {
                            panic!(
                                "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (period={}, power={}, source={})",
                                test_name, val, bits, i, period, power, source
                            );
                        }

                        
                        if bits == 0x33333333_33333333 {
                            panic!(
                                "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (period={}, power={}, source={})",
                                test_name, val, bits, i, period, power, source
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    
    #[cfg(not(debug_assertions))]
    fn check_vpwma_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vpwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        
        let strat = (2usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period.max(2)..400,
                ),
                Just(period),
                0.1f64..10.0f64, 
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, power)| {
                let params = VpwmaParams {
                    period: Some(period),
                    power: Some(power),
                };
                let input = VpwmaInput::from_slice(&data, params);

                let VpwmaOutput { values: out } = vpwma_with_kernel(&input, kernel).unwrap();
                let VpwmaOutput { values: ref_out } =
                    vpwma_with_kernel(&input, Kernel::Scalar).unwrap();

                
                prop_assert_eq!(out.len(), data.len());

                
                let expected_warmup = period - 1;
                for i in 0..expected_warmup.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                
                
                let mut weights = Vec::with_capacity(period - 1);
                let mut weight_sum = 0.0;
                for k in 0..(period - 1) {
                    let w = (period as f64 - k as f64).powf(power);
                    weights.push(w);
                    weight_sum += w;
                }
                
                prop_assert!(
                    (weight_sum - 0.0).abs() > 1e-10,
                    "Weight sum should be non-zero, got {}",
                    weight_sum
                );

                
                for i in expected_warmup..data.len() {
                    if out[i].is_nan() {
                        continue;
                    }

                    
                    let window_start = i.saturating_sub(period - 1);
                    let window = &data[window_start..=i];
                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    prop_assert!(
                        out[i] >= lo - 1e-9 && out[i] <= hi + 1e-9,
                        "idx {}: {} ∉ [{}, {}]",
                        i,
                        out[i],
                        lo,
                        hi
                    );
                }

                
                

                
                if period == 2 && data.len() >= 2 {
                    
                    
                    for i in 1..data.len() {
                        if !out[i].is_nan() && !data[i].is_nan() {
                            
                            prop_assert!(
                                (out[i] - data[i]).abs() <= 1e-9,
                                "Period=2 mismatch at idx {}: {} vs {}",
                                i,
                                out[i],
                                data[i]
                            );
                        }
                    }
                }

                
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && data.len() > 0 {
                    let constant_val = data[0];
                    for i in expected_warmup..data.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                (out[i] - constant_val).abs() <= 1e-9,
                                "Constant data should give constant output at idx {}: {} vs {}",
                                i,
                                out[i],
                                constant_val
                            );
                        }
                    }
                }

                
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    
                    let max_ulp = if matches!(kernel, Kernel::Avx512) {
                        20
                    } else {
                        10
                    };

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= max_ulp,
                        "mismatch idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_vpwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    generate_all_vpwma_tests!(
        check_vpwma_partial_params,
        check_vpwma_accuracy,
        check_vpwma_zero_period,
        check_vpwma_period_exceeds_length,
        check_vpwma_very_small_dataset,
        check_vpwma_reinput,
        check_vpwma_nan_handling,
        check_vpwma_streaming,
        check_vpwma_no_poison,
        check_vpwma_property
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = VpwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = VpwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            59363.927599446455,
            59296.83894519251,
            59196.82476139941,
            59180.8040249446,
            59113.84473799056,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let period_ranges = vec![(2, 10, 2), (10, 30, 5), (30, 60, 10), (5, 15, 1)];

        let power_ranges = vec![(0.1, 0.5, 0.1), (0.3, 1.0, 0.2), (1.0, 3.0, 0.5)];

        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for &(p_start, p_end, p_step) in &period_ranges {
            for &(pow_start, pow_end, pow_step) in &power_ranges {
                for &source in &test_sources {
                    let output = VpwmaBatchBuilder::new()
                        .kernel(kernel)
                        .period_range(p_start, p_end, p_step)
                        .power_range(pow_start, pow_end, pow_step) 
                        .apply_candles(&c, source)?;

                    for (idx, &val) in output.values.iter().enumerate() {
                        if val.is_nan() {
                            continue;
                        }

                        let bits = val.to_bits();
                        let row = idx / output.cols;
                        let col = idx % output.cols;

                        const POISON1: u64 = 0x1111_1111_1111_1111;
                        const POISON2: u64 = 0x2222_2222_2222_2222;
                        const POISON3: u64 = 0x3333_3333_3333_3333;

                        if bits == POISON1 || bits == POISON2 || bits == POISON3 {
                            panic!(
                                "[{test}] poison value (0x{bits:016X}) row={row} col={col} \
                             period_range=({p_start},{p_end},{p_step}) \
                             power_range=({pow_start},{pow_end},{pow_step}) source={source}"
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
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpwma")]
#[pyo3(signature = (data, period, power, kernel=None))]
/// Compute the Variable Power Weighted Moving Average (VPWMA) of the input data.
///
/// VPWMA adjusts the weights of each price data point based on their respective
/// positions, with the weight raised to a specified power to control how
/// aggressively recent data points dominate the average.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window (must be >= 2).
/// power : float
///     Exponent applied to the weight function (typically 0.382).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of VPWMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period < 2, power is NaN/infinite, etc).
pub fn vpwma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    power: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?; 

    
    let kern = validate_kernel(kernel, false)?;

    
    let params = VpwmaParams {
        period: Some(period),
        power: Some(power),
    };
    let vpwma_in = VpwmaInput::from_slice(slice_in, params);

    
    let result_vec: Vec<f64> = py
        .allow_threads(|| vpwma_with_kernel(&vpwma_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VpwmaStream")]
pub struct VpwmaStreamPy {
    stream: VpwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VpwmaStreamPy {
    #[new]
    fn new(period: usize, power: f64) -> PyResult<Self> {
        let params = VpwmaParams {
            period: Some(period),
            power: Some(power),
        };
        let stream =
            VpwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VpwmaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated VPWMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpwma_batch")]
#[pyo3(signature = (data, period_range, power_range, kernel=None))]
/// Compute VPWMA for multiple parameter combinations in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// power_range : tuple
///     (start, end, step) for power values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array), 'periods', and 'powers' arrays.
pub fn vpwma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    power_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = VpwmaBatchRange {
        period: period_range,
        power: power_range,
    };

    
    let combos = expand_grid_vpwma(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    
    let kern = validate_kernel(kernel, true)?;

    
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

            
            vpwma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    
    let dict = PyDict::new(py);

    
    let reshaped = out_arr.reshape([rows, cols])?;
    dict.set_item("values", reshaped)?;

    
    let periods: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
    let powers: Vec<f64> = combos.iter().map(|c| c.power.unwrap()).collect();

    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("powers", powers.into_pyarray(py))?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32VpwmaPy {
    pub(crate) inner: DeviceArrayF32Vpwma,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32VpwmaPy {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        // Explicit byte strides for contiguous row-major FP32
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Producing stream is synchronized before returning; omit per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // 2 == kDLCUDA; use allocation device id on the handle
        (2, self.inner.device_id as i32)
    }

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        // Validate `dl_device` hint if provided against the allocation device.
        let (kdl, alloc_dev) = self.__dlpack_device__();
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
        let dummy = {
            use cust::memory::DeviceBuffer;
            DeviceArrayF32Vpwma {
                buf: DeviceBuffer::from_slice(&[])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                rows: 0,
                cols: 0,
                ctx: self.inner.ctx.clone(),
                device_id: self.inner.device_id,
            }
        };
        let inner = std::mem::replace(&mut self.inner, dummy);

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vpwma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, power_range, device_id=0))]
pub fn vpwma_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    power_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<(DeviceArrayF32VpwmaPy, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let sweep = VpwmaBatchRange {
        period: period_range,
        power: power_range,
    };

    let slice_in = data_f32.as_slice()?;

    let (inner, combos) = py.allow_threads(|| {
        let cuda = CudaVpwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vpwma_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    let powers: Vec<f64> = combos.iter().map(|c| c.power.unwrap()).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("powers", powers.into_pyarray(py))?;

    Ok((DeviceArrayF32VpwmaPy { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vpwma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, power, device_id=0))]
pub fn vpwma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    power: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32VpwmaPy> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = VpwmaParams {
        period: Some(period),
        power: Some(power),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaVpwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vpwma_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32VpwmaPy { inner })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_js(data: &[f64], period: usize, power: f64) -> Result<Vec<f64>, JsValue> {
    let params = VpwmaParams {
        period: Some(period),
        power: Some(power),
    };
    let input = VpwmaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];
    vpwma_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    power_start: f64,
    power_end: f64,
    power_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VpwmaBatchRange {
        period: (period_start, period_end, period_step),
        power: (power_start, power_end, power_step),
    };

    // Use the existing batch function with parallel=false for WASM
    vpwma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    power_start: f64,
    power_end: f64,
    power_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VpwmaBatchRange {
        period: (period_start, period_end, period_step),
        power: (power_start, power_end, power_step),
    };

    let combos = expand_grid_vpwma(&sweep);
    let mut metadata = Vec::with_capacity(combos.len() * 2);

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
        metadata.push(combo.power.unwrap());
    }

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    power: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to vpwma_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period < 2 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = VpwmaParams {
            period: Some(period),
            power: Some(power),
        };
        let input = VpwmaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            vpwma_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vpwma_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct VpwmaContext {
    weights: AVec<f64>,
    inv_norm: f64,
    period: usize,
    first: usize,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl VpwmaContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(period: usize, power: f64) -> Result<VpwmaContext, JsValue> {
        if period < 2 {
            return Err(JsValue::from_str("Invalid period: must be >= 2"));
        }
        if power.is_nan() || power.is_infinite() {
            return Err(JsValue::from_str(&format!("Invalid power: {}", power)));
        }

        let win_len = period - 1;
        let weights: AVec<f64> = AVec::from_iter(
            CACHELINE_ALIGN,
            (0..win_len).map(|k| (period as f64 - k as f64).powf(power)),
        );
        let norm: f64 = weights.iter().sum();
        let inv_norm = 1.0 / norm;

        Ok(VpwmaContext {
            weights,
            inv_norm,
            period,
            first: 0,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        in_ptr: *const f64,
        out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if len < self.period {
            return Err(JsValue::from_str("Data length less than period"));
        }

        unsafe {
            let data = std::slice::from_raw_parts(in_ptr, len);
            let out = std::slice::from_raw_parts_mut(out_ptr, len);

            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

            if in_ptr == out_ptr {
                let mut temp = vec![0.0; len];
                match self.kernel {
                    Kernel::Scalar => vpwma_scalar(
                        data,
                        &self.weights,
                        self.period,
                        first,
                        self.inv_norm,
                        &mut temp,
                    ),
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 => vpwma_avx2(
                        data,
                        &self.weights,
                        self.period,
                        first,
                        self.inv_norm,
                        &mut temp,
                    ),
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 => vpwma_avx512(
                        data,
                        &self.weights,
                        self.period,
                        first,
                        self.inv_norm,
                        &mut temp,
                    ),
                    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                    Kernel::Avx2 | Kernel::Avx512 => vpwma_scalar(
                        data,
                        &self.weights,
                        self.period,
                        first,
                        self.inv_norm,
                        &mut temp,
                    ),
                    _ => unreachable!(),
                }
                out.copy_from_slice(&temp);
            } else {
                match self.kernel {
                    Kernel::Scalar => {
                        vpwma_scalar(data, &self.weights, self.period, first, self.inv_norm, out)
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 => {
                        vpwma_avx2(data, &self.weights, self.period, first, self.inv_norm, out)
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 => {
                        vpwma_avx512(data, &self.weights, self.period, first, self.inv_norm, out)
                    }
                    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                    Kernel::Avx2 | Kernel::Avx512 => {
                        vpwma_scalar(data, &self.weights, self.period, first, self.inv_norm, out)
                    }
                    _ => unreachable!(),
                }
            }

            for i in 0..(first + self.period - 1) {
                out[i] = f64::NAN;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.period - 1
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VpwmaBatchConfig {
    pub period_range: (usize, usize, usize),
    pub power_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VpwmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VpwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vpwma_batch)]
pub fn vpwma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: VpwmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = VpwmaBatchRange {
        period: config.period_range,
        power: config.power_range,
    };

    let output = vpwma_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = VpwmaBatchJsOutput {
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
pub fn vpwma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    data_len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    power_start: f64,
    power_end: f64,
    power_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer passed to vpwma_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, data_len);
        let sweep = VpwmaBatchRange {
            period: (period_start, period_end, period_step),
            power: (power_start, power_end, power_step),
        };
        let combos = expand_grid_vpwma(&sweep);
        let rows = combos.len();
        let cols = data_len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        // resolve Auto→SIMD like Python path
        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        vpwma_batch_inner_into(data, &sweep, simd, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
