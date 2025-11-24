//! # Ehlers Distance Coefficient Filter (EDCF)
//!
//! John Ehlers' Distance Coefficient Filter (EDCF) uses squared distances between successive points to build a non-linear, volatility-sensitive weighted average. Higher weights are assigned to prices following larger recent price changes, smoothing out trendless noise. Re-applying EDCF to its own output can provide multi-stage smoothing.
//!
//! ## WASM Performance Warning
//! **⚠️ IMPORTANT: This indicator has severe performance limitations in WebAssembly (WASM).**
//!
//! EDCF requires a full-size distance buffer that scales with input length (8MB for 1M data points).
//! The algorithm's second pass performs random access across this entire buffer, which is extremely
//! inefficient in WASM's linear memory model. This results in EDCF being **20-60x slower** in WASM
//! compared to native execution, while most other indicators are only 2-3x slower.
//!
//! For WASM/browser applications, consider using alternative smoothing indicators like ALMA, EMA,
//! or HMA which have better memory access patterns and near-native WASM performance.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). (defaults to 15)
//!
//! ## Errors
//! - **NoData**: edcf: No data provided to EDCF filter.
//! - **AllValuesNaN**: edcf: All input data values are `NaN`.
//! - **InvalidPeriod**: edcf: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: edcf: Not enough valid data points for the requested `period`.
//! - **OutputLenMismatch**: edcf: Output buffer length doesn't match input data length.
//!
//! ## Returns
//! - **`Ok(EdcfOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(EdcfError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: Fully implemented - vectorized distance calculations with 4-wide SIMD
//! - **AVX512 kernel**: Fully implemented - optimized with 8-wide SIMD operations
//! - **Streaming update**: O(1) per update using rolling sums of prices and squared prices
//! - **Memory optimization**: Uses alloc_with_nan_prefix but requires full-size distance buffer (memory intensive)
//! - **WASM caution**: Distance buffer access remains costly in WASM (see warning above)
//!
//! Decision: Streaming path switched to O(1) kernel using rolling sums; matches batch warmup (2·period) and returns None when denominator is zero.
//! CUDA: Wrapper present (FP32 rolling/tiled); Python interop uses the shared CAI v3 + DLPack v1.x wrapper with primary-context RAII for correct context lifetime.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// =============== Python: reuse shared DeviceArrayF32Py (CAI v3 + DLPack v1.x) ===============
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::{make_device_array_py, DeviceArrayF32Py};

// Thread-local storage for WASM dist buffer to avoid repeated allocations
#[cfg(target_arch = "wasm32")]
thread_local! {
    static WASM_DIST_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

#[derive(Debug, Clone)]
pub enum EdcfData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for EdcfInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EdcfData::Slice(slice) => slice,
            EdcfData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
/// Parameters controlling the EDCF calculation.
pub struct EdcfParams {
    /// Window size for the filter.
    pub period: Option<usize>,
}

impl Default for EdcfParams {
    fn default() -> Self {
        Self { period: Some(15) }
    }
}

#[derive(Debug, Clone)]
pub struct EdcfOutput {
    /// Filtered values matching the input length.
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EdcfInput<'a> {
    /// Input data series.
    pub data: EdcfData<'a>,
    /// Filter parameters.
    pub params: EdcfParams,
}

impl<'a> EdcfInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EdcfParams) -> Self {
        Self {
            data: EdcfData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EdcfParams) -> Self {
        Self {
            data: EdcfData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EdcfParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(15)
    }
}

#[derive(Debug, Error)]
pub enum EdcfError {
    #[error("edcf: No data provided to EDCF filter.")]
    NoData,
    #[error("edcf: Empty input data.")]
    EmptyInputData,
    #[error("edcf: All values are NaN.")]
    AllValuesNaN,
    #[error("edcf: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("edcf: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("edcf: Output buffer length mismatch: expected = {expected}, got = {got}")]
    OutputLenMismatch { expected: usize, got: usize },
    #[error("edcf: Invalid kernel specified")]
    InvalidKernel,
    #[error("edcf: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("edcf: Invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("edcf: size overflow during allocation ({op})")]
    SizeOverflow { op: &'static str },
}

#[derive(Copy, Clone, Debug)]
/// Builder providing a fluent API for [`EdcfOutput`].
pub struct EdcfBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EdcfBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EdcfBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EdcfOutput, EdcfError> {
        let p = EdcfParams {
            period: self.period,
        };
        let i = EdcfInput::from_candles(c, "close", p);
        edcf_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EdcfOutput, EdcfError> {
        let p = EdcfParams {
            period: self.period,
        };
        let i = EdcfInput::from_slice(d, p);
        edcf_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EdcfStream, EdcfError> {
        let p = EdcfParams {
            period: self.period,
        };
        EdcfStream::try_new(p)
    }
}

#[inline]
pub fn edcf(input: &EdcfInput) -> Result<EdcfOutput, EdcfError> {
    edcf_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn edcf_prepare<'a>(
    input: &'a EdcfInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), EdcfError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EdcfError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EdcfError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EdcfError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let needed = 2 * period;
    if (len - first) < needed {
        return Err(EdcfError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let warm = first + 2 * period;
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, warm, chosen))
}

#[inline(always)]
fn edcf_compute_into(data: &[f64], period: usize, first: usize, chosen: Kernel, out: &mut [f64]) {
    // Use optimized WASM version with reusable buffer
    #[cfg(target_arch = "wasm32")]
    {
        if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
            edcf_scalar_wasm(data, period, first, out);
            return;
        }
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => edcf_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => edcf_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => edcf_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }
}

pub fn edcf_with_kernel(input: &EdcfInput, kernel: Kernel) -> Result<EdcfOutput, EdcfError> {
    let (data, period, first, warm, chosen) = edcf_prepare(input, kernel)?;
    let len = data.len();
    let mut out = alloc_with_nan_prefix(len, warm);
    edcf_compute_into(data, period, first, chosen, &mut out);
    Ok(EdcfOutput { values: out })
}

/// Computes EDCF into a caller-provided buffer (no allocations).
///
/// - Preserves NaN warmups exactly as the Vec-returning API.
/// - The output slice length must equal the input length.
/// - Uses `Kernel::Auto` for kernel selection.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn edcf_into(input: &EdcfInput, out: &mut [f64]) -> Result<(), EdcfError> {
    let (data, period, first, warm, chosen) = edcf_prepare(input, Kernel::Auto)?;

    // Validate output buffer size
    if out.len() != data.len() {
        return Err(EdcfError::OutputLenMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    // Guard against aliasing with the input slice (in-place not supported)
    let in_ptr = data.as_ptr();
    let out_ptr = out.as_ptr();
    if core::ptr::eq(in_ptr, out_ptr) {
        let mut temp = alloc_with_nan_prefix(out.len(), warm);
        edcf_compute_into(data, period, first, chosen, &mut temp);
        out.copy_from_slice(&temp);
        return Ok(());
    }

    // Prefill warmup with the same quiet-NaN pattern used by alloc_with_nan_prefix
    let warm = warm.min(out.len());
    for v in &mut out[..warm] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    // Compute the remainder directly into the provided buffer
    edcf_compute_into(data, period, first, chosen, out);

    Ok(())
}

/// Computes EDCF directly into a provided output slice, avoiding allocation.
/// The output slice must be the same length as the input data.
#[inline]
pub fn edcf_into_slice(dst: &mut [f64], input: &EdcfInput, kern: Kernel) -> Result<(), EdcfError> {
    let (data, period, first, warm, chosen) = edcf_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(EdcfError::OutputLenMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    // Compute directly into the output buffer
    edcf_compute_into(data, period, first, chosen, dst);

    // Fill warmup period with NaN (post-compute write pattern like ALMA)
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
pub fn edcf_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    let len = data.len();

    // Allocate uninitialized memory for dist buffer
    let mut dist: Vec<f64> = Vec::with_capacity(len);
    unsafe {
        // Initialize only the portion that will be read before being written
        // The computation reads from indices [j-period+1..=j] where j starts at first_valid + 2*period
        // So we need zeros up to first_valid + period
        let zero_end = (first_valid + period).min(len);
        dist.set_len(zero_end);
        dist.fill(0.0);

        // Extend to full length without initialization
        dist.set_len(len);
    }

    unsafe {
        let dp = data.as_ptr();
        let wp = dist.as_mut_ptr();

        let dist_start = first_valid + period;
        for k in dist_start..len {
            let xk = *dp.add(k);
            let mut sum_sq = 0.0;
            for lb in 1..period {
                let diff = xk - *dp.add(k - lb);
                sum_sq = diff.mul_add(diff, sum_sq);
            }
            *wp.add(k) = sum_sq;
        }

        let start_j = first_valid + 2 * period;
        for j in start_j..len {
            let mut num = 0.0;
            let mut coef_sum = 0.0;
            for i in 0..period {
                let k = j - i;
                let w = *wp.add(k);
                let v = *dp.add(k);

                num = w.mul_add(v, num);
                coef_sum += w;
            }
            if coef_sum != 0.0 {
                *out.get_unchecked_mut(j) = num / coef_sum;
            }
        }
    }
}

#[inline(always)]
fn edcf_scalar_into_with_scratch(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
    scratch: &mut Vec<f64>,
) {
    let len = data.len();
    // Ensure capacity, then expose uninit tail. We will only read initialized parts.
    if scratch.capacity() < len {
        scratch.reserve_exact(len - scratch.capacity());
    }
    unsafe {
        scratch.set_len(len);
    }
    let zero_end = (first_valid + period).min(len);
    scratch[..zero_end].fill(0.0);

    unsafe {
        let dp = data.as_ptr();
        let wp = scratch.as_mut_ptr();

        // Fill distances for indices that will be read
        for k in (first_valid + period)..len {
            let xk = *dp.add(k);
            let mut sum_sq = 0.0;
            for lb in 1..period {
                let diff = xk - *dp.add(k - lb);
                sum_sq = diff.mul_add(diff, sum_sq);
            }
            *wp.add(k) = sum_sq;
        }

        // Weighted average using the distance weights
        let start_j = first_valid + 2 * period;
        for j in start_j..len {
            let mut num = 0.0;
            let mut coef_sum = 0.0;
            for i in 0..period {
                let k = j - i;
                let w = *wp.add(k);
                let v = *dp.add(k);
                num = w.mul_add(v, num);
                coef_sum += w;
            }
            if coef_sum != 0.0 {
                *out.get_unchecked_mut(j) = num / coef_sum;
            }
        }
    }
}

// WASM-optimized version that reuses dist buffer to avoid repeated allocations
#[cfg(target_arch = "wasm32")]
#[inline]
fn edcf_scalar_wasm(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    let len = data.len();

    WASM_DIST_BUFFER.with(|buffer| {
        let mut dist = buffer.borrow_mut();

        // Resize buffer if needed, reusing existing capacity
        if dist.len() < len {
            dist.resize(len, 0.0);
        }

        // Zero out the portion that will be read before being written
        let zero_end = (first_valid + period).min(len);
        dist[..zero_end].fill(0.0);

        unsafe {
            let dp = data.as_ptr();
            let wp = dist.as_mut_ptr();

            let dist_start = first_valid + period;
            for k in dist_start..len {
                let xk = *dp.add(k);
                let mut sum_sq = 0.0;
                for lb in 1..period {
                    let diff = xk - *dp.add(k - lb);
                    sum_sq = diff.mul_add(diff, sum_sq);
                }
                *wp.add(k) = sum_sq;
            }

            let start_j = first_valid + 2 * period;
            for j in start_j..len {
                let mut num = 0.0;
                let mut coef_sum = 0.0;
                for i in 0..period {
                    let k = j - i;
                    let w = *wp.add(k);
                    let v = *dp.add(k);

                    num = w.mul_add(v, num);
                    coef_sum += w;
                }
                if coef_sum != 0.0 {
                    *out.get_unchecked_mut(j) = num / coef_sum;
                }
            }
        }
    });
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum_m256d(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum2 = _mm_add_pd(hi, lo);
    let hi64 = _mm_unpackhi_pd(sum2, sum2);
    _mm_cvtsd_f64(_mm_add_sd(sum2, hi64))
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn edcf_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    const STEP: usize = 4;
    let len = data.len();
    let chunks = period / STEP;

    // Allocate uninitialized memory for dist buffer
    let mut dist: Vec<f64> = Vec::with_capacity(len);
    unsafe {
        // Initialize only the portion that will be read before being written
        let zero_end = (first_valid + period).min(len);
        dist.set_len(zero_end);
        dist.fill(0.0);

        // Extend to full length without initialization
        dist.set_len(len);
    }
    let dp = data.as_ptr();
    let wp = dist.as_mut_ptr();

    for k in (first_valid + period)..len {
        let xk_vec = _mm256_broadcast_sd(&*dp.add(k));
        let mut acc = _mm256_setzero_pd();

        for blk in 0..chunks {
            let ptr = dp.add(k - (blk + 1) * STEP);
            let d = _mm256_loadu_pd(ptr);
            let diff = _mm256_sub_pd(xk_vec, d);
            acc = _mm256_fmadd_pd(diff, diff, acc);
        }

        let mut sum_tail = 0.0;
        for lb in (chunks * STEP + 1)..period {
            let diff = *dp.add(k) - *dp.add(k - lb);
            sum_tail += diff * diff;
        }

        *wp.add(k) = hsum_m256d(acc) + sum_tail;
    }

    for j in (first_valid + 2 * period)..len {
        let start_k = j + 1 - period;

        let mut num_vec = _mm256_setzero_pd();
        let mut coef_vec = _mm256_setzero_pd();
        for blk in 0..chunks {
            let idx = start_k + blk * STEP;
            let d = _mm256_loadu_pd(dp.add(idx));
            let w = _mm256_loadu_pd(wp.add(idx));
            num_vec = _mm256_fmadd_pd(w, d, num_vec);
            coef_vec = _mm256_add_pd(coef_vec, w);
        }

        let mut num = hsum_m256d(num_vec);
        let mut coef = hsum_m256d(coef_vec);

        for i in (chunks * STEP)..period {
            let k = start_k + i;
            let w = *wp.add(k);
            num += w * *dp.add(k);
            coef += w;
        }

        if coef != 0.0 {
            *out.get_unchecked_mut(j) = num / coef;
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn edcf_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    const STEP: usize = 8;

    let len = data.len();
    let p_minus1 = period - 1;
    let chunks = p_minus1 / STEP;
    let tail_len = p_minus1 % STEP;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    // Allocate uninitialized memory for dist buffer
    let mut dist: Vec<f64> = Vec::with_capacity(len);
    unsafe {
        // Initialize only the portion that will be read before being written
        let zero_end = (first_valid + period).min(len);
        dist.set_len(zero_end);
        dist.fill(0.0);

        // Extend to full length without initialization
        dist.set_len(len);
    }
    let dp = data.as_ptr();
    let wp = dist.as_mut_ptr();

    for k in (first_valid + period)..len {
        let xk_vec = _mm512_set1_pd(*dp.add(k));
        let mut acc = _mm512_setzero_pd();

        let start = k - p_minus1;
        for blk in 0..chunks {
            let d = _mm512_loadu_pd(dp.add(start + blk * STEP));
            let diff = _mm512_sub_pd(xk_vec, d);
            acc = _mm512_fmadd_pd(diff, diff, acc);
        }

        if tail_len != 0 {
            let base = dp.add(start + chunks * STEP);
            let d = _mm512_maskz_loadu_pd(tail_mask, base);
            let diff = _mm512_mask_sub_pd(_mm512_setzero_pd(), tail_mask, xk_vec, d);
            let sq = _mm512_mul_pd(diff, diff);
            acc = _mm512_add_pd(acc, sq);
        }

        *wp.add(k) = _mm512_reduce_add_pd(acc);
    }

    for j in (first_valid + 2 * period)..len {
        let start_k = j - p_minus1;

        let mut num_vec = _mm512_setzero_pd();
        let mut coef_vec = _mm512_setzero_pd();

        for blk in 0..chunks {
            let idx = start_k + blk * STEP;
            let d = _mm512_loadu_pd(dp.add(idx));
            let w = _mm512_loadu_pd(wp.add(idx));
            num_vec = _mm512_fmadd_pd(w, d, num_vec);
            coef_vec = _mm512_add_pd(coef_vec, w);
        }

        if tail_len != 0 {
            let idx = start_k + chunks * STEP;
            let d = _mm512_maskz_loadu_pd(tail_mask, dp.add(idx));
            let w = _mm512_maskz_loadu_pd(tail_mask, wp.add(idx));
            num_vec = _mm512_fmadd_pd(w, d, num_vec);
            coef_vec = _mm512_add_pd(coef_vec, w);
        }

        let w0 = *wp.add(j);
        let v0 = *dp.add(j);
        let num = _mm512_reduce_add_pd(num_vec) + w0 * v0;
        let coef = _mm512_reduce_add_pd(coef_vec) + w0;

        if coef != 0.0 {
            *out.get_unchecked_mut(j) = num / coef;
        }
    }
}

#[derive(Debug, Clone)]
/// Streaming variant of the EDCF filter with O(1) updates.
pub struct EdcfStream {
    period: usize,
    // price ring buffer
    buffer: Vec<f64>,
    // weight ring buffer (w_k for the price at the same index)
    dist: Vec<f64>,
    // index to overwrite next (points at the oldest sample)
    head: usize,
    // number of accepted finite samples since last reset
    count: usize,

    // rolling sums over the *previous p-1* samples (exclude current)
    sum_prev: f64,
    sum_prev_sq: f64,

    // rolling window aggregates over the last *p* (w_k, x_k) pairs
    den: f64,
    num: f64,

    // cached (p-1) as f64
    p_minus1_f: f64,
}

impl EdcfStream {
    /// Creates a new stream with the provided parameters.
    #[inline]
    pub fn try_new(params: EdcfParams) -> Result<Self, EdcfError> {
        let period = params.period.unwrap_or(15);
        if period == 0 {
            return Err(EdcfError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        // Keep parity with existing semantics: price buffer starts as NaN,
        // weight buffer starts as 0.0 (weights are not defined before p-1 prevs).
        let buffer = alloc_with_nan_prefix(period, period);
        let dist = vec![0.0; period];

        Ok(Self {
            period,
            buffer,
            dist,
            head: 0,
            count: 0,
            sum_prev: 0.0,
            sum_prev_sq: 0.0,
            den: 0.0,
            num: 0.0,
            p_minus1_f: (period - 1) as f64,
        })
    }

    #[inline(always)]
    fn bump_head(&mut self) {
        // branchy increment is faster than % on hot paths
        let n = self.head + 1;
        self.head = if n == self.period { 0 } else { n };
    }

    /// Feed a new value and return the filtered result once warmup is satisfied.
    ///
    /// Warmup matches batch kernels: returns `None` until `count >= 2*period`.
    /// Returns `None` as well if the current denominator is zero (e.g., constant window).
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            // keep state unchanged on non-finite input
            return None;
        }

        let p = self.period;

        // Values that are about to leave the window (valid only once we've seen >= p samples).
        let old_x = self.buffer[self.head];
        let old_w = self.dist[self.head];
        let had_full_window = self.count >= p;

        // --- 1) Compute new weight w_t in O(1) ---
        // Only defined once we have p-1 previous values; otherwise keep it 0
        // to mirror the batch kernel's early zeros in the distance buffer.
        let w_new = if self.count >= (p - 1) {
            // w_t = (p-1)*x^2 - 2*x*sum_prev + sum_prev_sq
            let x2 = value * value;
            self.p_minus1_f.mul_add(x2, self.sum_prev_sq) - (2.0 * value * self.sum_prev)
        } else {
            0.0
        };

        // --- 2) Update rolling aggregates (den,num) for the last p weights ---
        if had_full_window {
            // drop the contribution that slides out
            self.den -= old_w;
            self.num -= old_w * old_x;
        }
        // add the new contribution
        self.den += w_new;
        self.num = w_new.mul_add(value, self.num);

        // --- 3) Commit the new sample & weight to the rings ---
        self.buffer[self.head] = value;
        self.dist[self.head] = w_new;
        self.bump_head();

        // --- 4) Maintain sums of the *previous p-1* values for next step ---
        // After evaluating w_new with the *current* previous set, we now
        // insert x_t into that set for the next update and, if necessary, remove the oldest.
        self.sum_prev += value;
        self.sum_prev_sq = value.mul_add(value, self.sum_prev_sq);
        if self.count >= (p - 1) {
            // remove x_{t-p+1}, which was sitting at `old_x` (head pointed to it before overwrite)
            if had_full_window {
                self.sum_prev -= old_x;
                self.sum_prev_sq -= old_x * old_x;
            }
        }

        // --- 5) Advance sample count and decide output ---
        self.count += 1;
        if self.count < 2 * p {
            return None;
        }
        if self.den != 0.0 {
            // Division isolated for easy "fast-math" swap (see helper below)
            Some(fast_div(self.num, self.den))
        } else {
            None
        }
    }
}

// ------------ Fast-math division hook (optional) ------------
#[inline(always)]
fn fast_div(num: f64, den: f64) -> f64 {
    // Default: rely on the platform's IEEE-754 division.
    // Swap this for a gated approximation below if you enable the "fast-math" cfg.
    num / den
}

/*  If you want a faster (approximate) division, you can enable a feature
    and use a Newton–Raphson reciprocal seeded from f32. This replaces a costly
    f64 divide with: 1 f32 divide + 2 f64 muls + 1 f64 add (+ casts).

    Accuracy: after one NR step the relative error is typically < 1e-7 for well-scaled inputs,
    which is more than sufficient for smoothing indicators. Add a second iteration if you
    want ~full f64 precision at still-lower latency than a hardware f64 divide on many CPUs.

    Uncomment and compile with `--cfg fast_math_edcf` to use.

#[inline(always)]
#[cfg(fast_math_edcf)]
fn fast_div(num: f64, den: f64) -> f64 {
    // initial guess from single-precision reciprocal
    let mut r = (1.0f32 / den as f32) as f64;
    // one Newton step: r <- r * (2 - den*r)
    r = r * (2.0 - den * r);
    // OPTIONAL second step for near-IEEE accuracy:
    // r = r * (2.0 - den * r);
    num * r
}
*/

#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EdcfBatchRange {
    /// Range for `period` as `(start, end, step)`.
    pub period: (usize, usize, usize),
}

impl Default for EdcfBatchRange {
    fn default() -> Self {
        Self {
            period: (15, 50, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
/// Builder for [`EdcfBatchOutput`] across a sweep of parameters.
pub struct EdcfBatchBuilder {
    range: EdcfBatchRange,
    kernel: Kernel,
}

impl EdcfBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<EdcfBatchOutput, EdcfError> {
        edcf_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<EdcfBatchOutput, EdcfError> {
        EdcfBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EdcfBatchOutput, EdcfError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<EdcfBatchOutput, EdcfError> {
        EdcfBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn edcf_batch_with_kernel(
    data: &[f64],
    sweep: &EdcfBatchRange,
    k: Kernel,
) -> Result<EdcfBatchOutput, EdcfError> {
    if data.is_empty() {
        return Err(EdcfError::EmptyInputData);
    }
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EdcfError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    edcf_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EdcfBatchOutput {
    /// Flattened matrix of batch results.
    pub values: Vec<f64>,
    /// Parameter combinations for each row.
    pub combos: Vec<EdcfParams>,
    /// Number of rows in `values`.
    pub rows: usize,
    /// Number of columns in `values`.
    pub cols: usize,
}
impl EdcfBatchOutput {
    pub fn row_for_params(&self, p: &EdcfParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(15) == p.period.unwrap_or(15))
    }
    pub fn values_for(&self, p: &EdcfParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EdcfBatchRange) -> Vec<EdcfParams> {
    let (mut start, mut end, step) = r.period;

    // Normalize reversed bounds and treat zero step or equal bounds as singleton.
    if start > end {
        core::mem::swap(&mut start, &mut end);
    }
    let periods: Vec<usize> = if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    };

    // Map periods → EdcfParams
    periods
        .into_iter()
        .map(|p| EdcfParams { period: Some(p) })
        .collect()
}

#[inline(always)]
pub fn edcf_batch_slice(
    data: &[f64],
    sweep: &EdcfBatchRange,
    kern: Kernel,
) -> Result<EdcfBatchOutput, EdcfError> {
    edcf_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn edcf_batch_par_slice(
    data: &[f64],
    sweep: &EdcfBatchRange,
    kern: Kernel,
) -> Result<EdcfBatchOutput, EdcfError> {
    edcf_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn edcf_batch_inner(
    data: &[f64],
    sweep: &EdcfBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EdcfBatchOutput, EdcfError> {
    if data.is_empty() {
        return Err(EdcfError::EmptyInputData);
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EdcfError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Checked sizing before allocating rows*cols matrix
    let _total = rows
        .checked_mul(cols)
        .ok_or(EdcfError::SizeOverflow { op: "rows*cols" })?;

    // Use zero-copy allocation pattern from alma.rs
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| data.iter().position(|x| !x.is_nan()).unwrap_or(0) + 2 * c.period.unwrap_or(15))
        .collect();

    // Initialize NaN prefixes
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to mutable slice for computation
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let result_combos = edcf_batch_inner_into(data, sweep, kern, parallel, out)?;

    // Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(EdcfBatchOutput {
        values,
        combos: result_combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn edcf_batch_inner_into(
    data: &[f64],
    sweep: &EdcfBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EdcfParams>, EdcfError> {
    // ─────────────────── guards unchanged ───────────────────
    if data.is_empty() {
        return Err(EdcfError::EmptyInputData);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EdcfError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EdcfError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < 2 * max_p {
        return Err(EdcfError::NotEnoughValidData {
            needed: 2 * max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;

            out.par_chunks_mut(cols).enumerate().for_each(|(row, dst)| {
                let period = combos[row].period.unwrap();
                match kern {
                    Kernel::Scalar => {
                        // For parallel execution, each thread gets its own scratch buffer
                        let mut scratch = Vec::<f64>::new();
                        edcf_scalar_into_with_scratch(data, period, first, dst, &mut scratch);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 => unsafe { edcf_row_avx2(data, first, period, dst) },
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 => unsafe { edcf_row_avx512(data, first, period, dst) },
                    _ => unsafe { edcf_row_scalar(data, first, period, dst) }, // wasm path
                }
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                let period = combos[row].period.unwrap();
                unsafe { edcf_row_scalar(data, first, period, dst) }
            }
        }
    } else {
        // serial: single reusable scratch
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut scratch = Vec::<f64>::new();
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                let period = combos[row].period.unwrap();
                match kern {
                    Kernel::Scalar => {
                        edcf_scalar_into_with_scratch(data, period, first, dst, &mut scratch)
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 => unsafe { edcf_row_avx2(data, first, period, dst) },
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 => unsafe { edcf_row_avx512(data, first, period, dst) },
                    _ => unsafe { edcf_row_scalar(data, first, period, dst) },
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                let period = combos[row].period.unwrap();
                unsafe { edcf_row_scalar(data, first, period, dst) }
            }
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn edcf_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    edcf_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn edcf_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    edcf_avx2(data, period, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn edcf_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    edcf_avx512(data, period, first, out);
}

// ==================== PYTHON MODULE REGISTRATION ====================
#[cfg(feature = "python")]
pub fn register_edcf_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edcf_py, m)?)?;
    m.add_function(wrap_pyfunction!(edcf_batch_py, m)?)?;
    m.add_class::<EdcfStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(edcf_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(edcf_cuda_many_series_one_param_dev_py, m)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;
    use std::error::Error;

    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_edcf_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small, non-trivial input with NaN warmup and varying values
        let mut data: Vec<f64> = Vec::new();
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..250usize {
            let x = (i as f64).sin() * 3.0 + (i as f64) * 0.05 + ((i % 7) as f64) * 0.1;
            data.push(x);
        }

        let input = EdcfInput::from_slice(&data, EdcfParams::default());

        // Baseline via Vec-returning API
        let baseline = edcf(&input)?.values;

        // Preallocate output and call into API
        let mut out = vec![0.0; data.len()];
        edcf_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: expected {:?}, got {:?}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_edcf_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EdcfInput::from_candles(&candles, "close", EdcfParams { period: None });
        let result = edcf_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_edcf_accuracy_last_five(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EdcfInput::from_candles(&candles, "hl2", EdcfParams { period: Some(15) });
        let result = edcf_with_kernel(&input, kernel)?;
        let expected = [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847,
        ];
        let len = result.values.len();
        let start = len - expected.len();
        for (i, &v) in result.values[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{}] EDCF mismatch at {}: got {}, expected {}",
                test_name,
                start + i,
                v,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_edcf_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EdcfInput::with_default_candles(&candles);
        match input.data {
            EdcfData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EdcfData::Candles"),
        }
        let period = input.get_period();
        assert_eq!(period, 15);
        Ok(())
    }

    fn check_edcf_with_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(0) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_with_no_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: [f64; 0] = [];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_with_period_exceeding_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(10) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_with_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input =
            EdcfInput::from_candles(&candles, "close", EdcfParams { period: Some(15) });
        let first_result = edcf_with_kernel(&first_input, kernel)?;
        let first_values = first_result.values;
        let second_input = EdcfInput::from_slice(&first_values, EdcfParams { period: Some(5) });
        let second_result = edcf_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_values.len());
        if second_result.values.len() > 240 {
            for (i, &val) in second_result.values.iter().enumerate().skip(240) {
                assert!(
                    !val.is_nan(),
                    "Found NaN in second EDCF output at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_edcf_accuracy_nan_check(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 15;
        let input = EdcfInput::from_candles(
            &candles,
            "close",
            EdcfParams {
                period: Some(period),
            },
        );
        let result = edcf_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let start_index = 2 * period;
        if result.values.len() > start_index {
            for (i, &val) in result.values.iter().enumerate().skip(start_index) {
                assert!(!val.is_nan(), "Found NaN in EDCF output at index {}", i);
            }
        }
        Ok(())
    }

    fn check_edcf_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EdcfInput::from_candles(&candles, "close", EdcfParams { period: Some(15) });
        let _batch = edcf_with_kernel(&input, kernel)?;

        let mut stream = EdcfStream::try_new(EdcfParams { period: Some(15) })?;
        let mut vals = Vec::with_capacity(candles.close.len());
        for &v in &candles.close {
            vals.push(stream.update(v).unwrap_or(f64::NAN));
        }
        for (i, &v) in vals.iter().enumerate().skip(30) {
            assert!(!v.is_nan(), "[{test_name}] NaN at {i}");
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_edcf_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to better catch uninitialized memory bugs
        let test_periods = vec![3, 5, 10, 15, 30, 50, 100, 200];
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for period in &test_periods {
            for source in &test_sources {
                let input = EdcfInput::from_candles(
                    &candles,
                    source,
                    EdcfParams {
                        period: Some(*period),
                    },
                );
                let output = edcf_with_kernel(&input, kernel)?;

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_edcf_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[allow(clippy::float_cmp)]
    fn check_edcf_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // ─ 1. Strategy ───────────────────────────────────────────────────────────
        // choose period first (3‥=30), then a ≥-2·period finite vector
        let strat = (3usize..=30).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    2 * period..400,
                ),
                Just(period),
                // affine parameters: non-zero scale, arbitrary shift
                (-1e3f64..1e3f64).prop_filter("a≠0", |a| a.abs() > 1e-12),
                -1e3f64..1e3f64,
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period, a, b)| {
            // ─ 2. Run kernels safely (don’t unwrap blindly) ───────────────────
            let params = EdcfParams {
                period: Some(period),
            };
            let input = EdcfInput::from_slice(&data, params.clone());

            let fast = edcf_with_kernel(&input, kernel);
            let slow = edcf_with_kernel(&input, Kernel::Scalar);

            match (fast, slow) {
                // ➊ Same-kind error ⇒ property holds
                (Err(e1), Err(e2))
                    if std::mem::discriminant(&e1) == std::mem::discriminant(&e2) =>
                {
                    return Ok(());
                }

                // ➊′ Different error kinds → fail
                (Err(e1), Err(e2)) => {
                    prop_assert!(false, "different errors: fast={:?} slow={:?}", e1, e2)
                }

                // ➋ Kernels disagree on success/error
                (Err(e), Ok(_)) => prop_assert!(false, "fast errored {e:?} but scalar succeeded"),
                (Ok(_), Err(e)) => prop_assert!(false, "scalar errored {e:?} but fast succeeded"),

                // ➌ Both succeeded – full invariant suite
                (Ok(fast), Ok(reference)) => {
                    let EdcfOutput { values: out } = fast;
                    let EdcfOutput { values: rref } = reference;

                    // pre-compute streaming and affine-transformed outputs
                    let mut stream = EdcfStream::try_new(params.clone()).unwrap();
                    let mut s_out = Vec::with_capacity(data.len());
                    for &v in &data {
                        s_out.push(stream.update(v).unwrap_or(f64::NAN));
                    }

                    let transformed: Vec<f64> = data.iter().map(|x| a * x + b).collect();
                    let t_out = edcf(&EdcfInput::from_slice(&transformed, params.clone()))?.values;

                    let warm = 2 * period; // first usable index

                    for i in warm..data.len() {
                        let win = &data[i + 1 - period..=i];
                        let (lo, hi) = win
                            .iter()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                                (l.min(v), h.max(v))
                            });
                        let y = out[i];
                        let yr = rref[i];
                        let ys = s_out[i];
                        let yt = t_out[i];

                        // 1️⃣ Window-boundedness
                        prop_assert!(
                            y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                            "idx {i}: {y} ∉ [{lo}, {hi}]"
                        );

                        // 2️⃣ Constant-series ⇒ all NaN
                        if win.iter().all(|v| *v == win[0]) {
                            prop_assert!(y.is_nan(), "idx {i}: expected NaN on constant series");
                        }

                        // 3️⃣ Affine equivariance (scale & translation)
                        if y.is_finite() && yt.is_finite() {
                            let expect = a * y + b;
                            let diff = (yt - expect).abs();
                            let tol = 1e-9_f64.max(expect.abs() * 1e-9);
                            let ulp = yt.to_bits().abs_diff(expect.to_bits());
                            prop_assert!(
                                diff <= tol || ulp <= 8,
                                "idx {i}: affine mismatch diff={diff:e}  ULP={ulp}"
                            );
                        }

                        // 4️⃣ SIMD ≡ scalar (ULP ≤ 4 or abs ≤ 1e-9)
                        let ulp = y.to_bits().abs_diff(yr.to_bits());
                        prop_assert!(
                            (y - yr).abs() <= 1e-9 || ulp <= 4,
                            "idx {i}: fast={y} ref={yr} ULP={ulp}"
                        );

                        // 5️⃣ Streaming parity
                        prop_assert!(
                            (y - ys).abs() <= 1e-9 || (y.is_nan() && ys.is_nan()),
                            "idx {i}: stream mismatch"
                        );
                    }

                    // 6️⃣ Warm-up NaNs
                    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(data.len());
                    let warm_expected = first + warm;
                    prop_assert!(out[..warm_expected].iter().all(|v| v.is_nan()));
                }
            }

            Ok(())
        })?;

        // 🔟 Error-path smoke tests (uniform across indicators)
        assert!(edcf(&EdcfInput::from_slice(&[], EdcfParams::default())).is_err());
        assert!(edcf(&EdcfInput::from_slice(
            &[f64::NAN; 12],
            EdcfParams::default()
        ))
        .is_err());
        assert!(edcf(&EdcfInput::from_slice(
            &[1.0; 5],
            EdcfParams { period: Some(8) }
        ))
        .is_err());
        assert!(edcf(&EdcfInput::from_slice(
            &[1.0; 5],
            EdcfParams { period: Some(0) }
        ))
        .is_err());

        Ok(())
    }

    fn check_edcf_invalid_kernel(
        test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let data = [1.0, 2.0, 3.0];
        let range = EdcfBatchRange::default();
        let res = edcf_batch_with_kernel(&data, &range, Kernel::Avx2);
        assert!(matches!(res, Err(EdcfError::InvalidKernelForBatch(Kernel::Avx2))), "{}", test_name);
        Ok(())
    }

    macro_rules! generate_all_edcf_tests {
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

    generate_all_edcf_tests!(
        check_edcf_partial_params,
        check_edcf_accuracy_last_five,
        check_edcf_with_default_candles,
        check_edcf_with_zero_period,
        check_edcf_with_no_data,
        check_edcf_with_period_exceeding_data_length,
        check_edcf_very_small_data_set,
        check_edcf_with_slice_data_reinput,
        check_edcf_accuracy_nan_check,
        check_edcf_streaming,
        check_edcf_property,
        check_edcf_invalid_kernel,
        check_edcf_no_poison
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EdcfBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = EdcfParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations to better catch uninitialized memory bugs
        // Test different period ranges and sources
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for source in &test_sources {
            // Test with various period ranges
            let output = EdcfBatchBuilder::new()
                .kernel(kernel)
                .period_range(3, 200, 5) // Wide range: 3 to 200 with step 5
                .apply_candles(&c, source)?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }
            }
        }

        // Also test edge cases with very small and very large periods
        let edge_case_ranges = vec![(3, 5, 1), (190, 200, 2), (50, 100, 10)];
        for (start, end, step) in edge_case_ranges {
            let output = EdcfBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
						"[{}] Found poison value {} (0x{:016X}) at row {} col {} with range ({},{},{})",
						test, val, bits, row, col, start, end, step
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
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "edcf")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn edcf_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = EdcfParams {
        period: Some(period),
    };
    let edcf_in = EdcfInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| edcf_with_kernel(&edcf_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EdcfStream")]
pub struct EdcfStreamPy {
    stream: EdcfStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EdcfStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = EdcfParams {
            period: Some(period),
        };
        let stream =
            EdcfStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(EdcfStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "edcf_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn edcf_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;

    let sweep = EdcfBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
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
            edcf_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
#[pyfunction(name = "edcf_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn edcf_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaEdcf;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = EdcfBatchRange {
        period: period_range,
    };

    let (inner, dev_id) = py.allow_threads(|| {
        let cuda = CudaEdcf::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let out = cuda
            .edcf_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((out, dev_id))
    })?;

    make_device_array_py(dev_id as usize, inner)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "edcf_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn edcf_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaEdcf;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in: &[f32] = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = EdcfParams {
        period: Some(period),
    };

    let (inner, dev_id) = py.allow_threads(|| {
        let mut cuda =
            CudaEdcf::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let out = cuda
            .edcf_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((out, dev_id))
    })?;

    make_device_array_py(dev_id as usize, inner)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn edcf_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = EdcfParams {
        period: Some(period),
    };
    let input = EdcfInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    edcf_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn edcf_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = EdcfBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    edcf_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn edcf_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = EdcfBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap() as f64)
        .collect();

    Ok(metadata)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn edcf_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn edcf_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn edcf_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to edcf_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Create input structure
        let params = EdcfParams {
            period: Some(period),
        };
        let input = EdcfInput::from_slice(data, params);

        // Check for aliasing (same input and output)
        if in_ptr == out_ptr {
            // Need temporary buffer for in-place operation
            let mut temp = vec![0.0; len];
            edcf_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy result to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // Direct computation into output buffer
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            edcf_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Batch Processing with Serde ==================

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EdcfBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EdcfBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EdcfParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = edcf_batch)]
pub fn edcf_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: EdcfBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EdcfBatchRange {
        period: config.period_range,
    };

    let output = edcf_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = EdcfBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn edcf_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to edcf_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = EdcfBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Use optimized batch processing
        edcf_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
