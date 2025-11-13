//! # End Point Moving Average (EPMA)
//!
//! A polynomial-weighted moving average with adjustable period and offset.
//! SIMD (AVX2/AVX512) kernels generate weights on-the-fly (no weight tables)
//! and use FMA to tighten hot loops. Despite EPMA being relatively memory bound,
//! these kernels still provide measurable speedups over scalar.
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
//! - **OutputLengthMismatch**: epma: Output buffer length doesn't match input data length
//! - **InvalidKernelForBatch**: epma: Non-batch kernel provided to batch operation
//!
//! ## Returns
//! - **Ok(EpmaOutput)** with a Vec<f64> of the same length as input
//! - **Err(EpmaError)** otherwise
//!
//! ## Developer Notes
//! - Scalar: ✅ On-the-fly linear weight ramp (no temp Vec); ~20–30% faster in 100k
//! - AVX2: ✅ FMA + weight synthesis; ~1.7–2.2× vs scalar at 100k
//! - AVX512: ✅ Short/long variants; ~3.3–3.5× vs scalar at 100k
//! - Batch (rows): ✅ AVX2/AVX512 batch paths use shared prefix sums (P,Q) across rows for O(1) per-tick updates; >5% faster than scalar batch at 100k
//! - Streaming update: ✅ O(1) per update via rolling S/R with Kahan compensation
//! - Memory: ✅ Uses `alloc_with_nan_prefix` for zero-copy outputs and avoids O(N) temporaries on single-series paths
//! - Status: SIMD enabled; fastest on AVX512 > AVX2 > Scalar by clear margins at realistic sizes (10k–100k)
//! - Notes: For batch, scalar path remains per-row kernels; SIMD batch selects the shared-prefix implementation.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaEpma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;

// Decision: Streaming uses O(1) updates maintaining S and R with Kahan compensation; matches batch outputs.
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
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
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

    #[error("epma: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("epma: Invalid kernel for batch operation: got {0:?}")]
    InvalidKernelForBatch(Kernel),

    // Additional non-breaking coverage for batch/range and sizing guards
    #[error("epma: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },

    #[error("epma: size overflow computing rows*cols: rows={rows}, cols={cols}")]
    SizeOverflow { rows: usize, cols: usize },
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
        // data
        &'a [f64],
        // period
        usize,
        // offset
        usize,
        // first
        usize,
        // warmup
        usize,
        // chosen
        Kernel,
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
        // For WASM, use SIMD128 when available instead of scalar
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                epma_simd128(data, period, offset, first, out);
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => epma_scalar(data, period, offset, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => epma_avx2(data, period, offset, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => epma_avx512(data, period, offset, first, out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                epma_scalar(data, period, offset, first, out)
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

/// Writes EPMA results into a caller-provided output slice without allocating.
///
/// - Preserves NaN warmup semantics by pre-filling the warmup prefix with the
///   module's quiet-NaN pattern (same as `alloc_with_nan_prefix`).
/// - The `out` slice length must equal the input length; otherwise returns
///   `EpmaError::OutputLengthMismatch`.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn epma_into(input: &EpmaInput, out: &mut [f64]) -> Result<(), EpmaError> {
    let (data, period, offset, first, warmup, chosen) = epma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(EpmaError::OutputLengthMismatch { expected: data.len(), got: out.len() });
    }

    // Prefill warmup with the same quiet-NaN pattern used elsewhere.
    let w = warmup.min(out.len());
    for v in &mut out[..w] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    // Compute values into the provided buffer.
    epma_compute_into(data, period, offset, first, chosen, out);
    Ok(())
}

/// Computes EPMA directly into a provided output slice, avoiding allocation.
/// The output slice must be the same length as the input data.
#[inline]
pub fn epma_into_slice(dst: &mut [f64], input: &EpmaInput, kern: Kernel) -> Result<(), EpmaError> {
    let (data, period, offset, first, warmup, chosen) = epma_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(EpmaError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }

    // Compute EPMA values directly into dst
    epma_compute_into(data, period, offset, first, chosen, dst);

    // Fill warmup period with NaN
    for v in &mut dst[..warmup] {
        // Match alloc_with_nan_prefix quiet-NaN pattern
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    Ok(())
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

    // Linear ramp weights for oldest-to-newest index i:
    // w(i) = (2 - offset) + i, i in [0, p1)
    // Sum of weights = p1*(2 - offset) + (p1-1)*p1/2
    let c0 = 2.0 - (offset as f64);
    let p1f = p1 as f64;
    let weight_sum = p1f.mul_add(c0, 0.5 * (p1f - 1.0) * p1f);
    let inv = 1.0 / weight_sum;

    for j in (first_valid + period + offset + 1)..n {
        let start = j + 1 - p1;

        // Unroll by 4 while generating weights on the fly to avoid
        // memory loads of a weight table.
        let mut sum = 0.0;
        let mut i = 0usize;
        let mut wi = c0; // weight for i

        // Process blocks of 4
        while i + 3 < p1 {
            // Weights for i, i+1, i+2, i+3 are wi, wi+1, wi+2, wi+3
            sum = data[start + i].mul_add(wi, sum);
            sum = data[start + i + 1].mul_add(wi + 1.0, sum);
            sum = data[start + i + 2].mul_add(wi + 2.0, sum);
            sum = data[start + i + 3].mul_add(wi + 3.0, sum);
            i += 4;
            wi += 4.0;
        }

        // Tail
        while i < p1 {
            sum = data[start + i].mul_add(wi, sum);
            i += 1;
            wi += 1.0;
        }

        out[j] = sum * inv;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn epma_simd128(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::wasm32::*;

    // SIMD128 processes 2 f64 values at a time
    const STEP: usize = 2;
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

    let chunks = p1 / STEP;
    let tail = p1 % STEP;

    for j in (first_valid + period + offset + 1)..n {
        let start = j + 1 - p1;
        let mut acc = f64x2_splat(0.0);

        // Process chunks of 2
        for blk in 0..chunks {
            let idx = blk * STEP;
            // Load 2 weights (reversed order)
            let w0 = weights[p1 - 1 - idx];
            let w1 = weights[p1 - 2 - idx];
            let w = f64x2(w0, w1);

            // Load 2 data values
            let d = v128_load(data.as_ptr().add(start + idx) as *const v128);
            acc = f64x2_add(acc, f64x2_mul(d, w));
        }

        // Process remaining element if period is odd
        let mut sum = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);

        if tail != 0 {
            sum += data[start + p1 - 1] * weights[0];
        }

        out[j] = sum / weight_sum;
    }
}

#[inline(always)]
fn build_weights_rev(period: usize, offset: usize) -> (Vec<f64>, f64) {
    let p1 = period - 1;
    let mut w = Vec::with_capacity(p1);
    let mut sum = 0.0;
    // Match the scalar implementation's weight formula
    // Scalar uses: weights[i] = (period - i - offset) for i in 0..p1
    // and then accesses as weights[p1 - 1 - i] when processing
    // So for this reversed version, we want w[k] to match what would be weights[k] in scalar
    for k in 0..p1 {
        // This gives us the weight that would be at position k in the reversed array
        let val = (period as isize - (p1 - 1 - k) as isize - offset as isize) as f64;
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

    // Weight ramp parameters
    let c0 = 2.0 - (offset as f64);
    let p1f = p1 as f64;
    let wsum = p1f.mul_add(c0, 0.5 * (p1f - 1.0) * p1f);
    let inv = 1.0 / wsum;

    // Constant ramp [0,1,2,3]
    let ramp = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);

    for j in (first_valid + period + offset + 1)..data.len() {
        let start = j + 1 - p1;
        let base_ptr = data.as_ptr().add(start);

        let mut acc = _mm256_setzero_pd();

        // Full 4-lane blocks with on-the-fly weights
        for blk in 0..chunks {
            let base_w = c0 + (blk * STEP) as f64;
            let w = _mm256_add_pd(_mm256_set1_pd(base_w), ramp);
            let d = _mm256_loadu_pd(base_ptr.add(blk * STEP));
            acc = _mm256_fmadd_pd(d, w, acc);
        }

        // Tail
        if tail != 0 {
            let base_w = c0 + (chunks * STEP) as f64;
            let w_t = _mm256_add_pd(_mm256_set1_pd(base_w), ramp);
            let d_t = _mm256_maskload_pd(base_ptr.add(chunks * STEP), mask);
            acc = _mm256_fmadd_pd(d_t, w_t, acc);
        }

        // Horizontal reduction
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

    // Weight ramp parameters
    let c0 = 2.0 - (offset as f64);
    let p1f = p1 as f64;
    let wsum = p1f.mul_add(c0, 0.5 * (p1f - 1.0) * p1f);
    let inv = 1.0 / wsum;

    // ramp [0..7]
    let ramp = _mm512_setr_pd(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);

    for j in (first_valid + period + offset + 1)..data.len() {
        let start = j + 1 - p1;
        let base_ptr = data.as_ptr().add(start);
        let mut acc = _mm512_setzero_pd();

        for blk in 0..chunks {
            let base_w = c0 + (blk * STEP) as f64;
            let w = _mm512_add_pd(_mm512_set1_pd(base_w), ramp);
            let d = _mm512_loadu_pd(base_ptr.add(blk * STEP));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        if tail != 0 {
            let base_w = c0 + (chunks * STEP) as f64;
            let w_t = _mm512_add_pd(_mm512_set1_pd(base_w), ramp);
            let d_t = _mm512_maskz_loadu_pd(tmask, base_ptr.add(chunks * STEP));
            acc = _mm512_fmadd_pd(d_t, w_t, acc);
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

    // Weight ramp parameters
    let c0 = 2.0 - (offset as f64);
    let p1f = p1 as f64;
    let wsum = p1f.mul_add(c0, 0.5 * (p1f - 1.0) * p1f);
    let inv = 1.0 / wsum;

    // ramp [0..7]
    let ramp = _mm512_setr_pd(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);

    for j in (first_valid + period + offset + 1)..data.len() {
        let start_ptr = data.as_ptr().add(j + 1 - p1);

        // Four accumulators for better ILP on longer periods
        let mut s0 = _mm512_setzero_pd();
        let mut s1 = _mm512_setzero_pd();
        let mut s2 = _mm512_setzero_pd();
        let mut s3 = _mm512_setzero_pd();

        // process 4*8 lanes per iteration
        let mut blk = 0usize;
        while blk < paired {
            let base0 = c0 + ((blk + 0) * STEP) as f64;
            let base1 = c0 + ((blk + 1) * STEP) as f64;
            let base2 = c0 + ((blk + 2) * STEP) as f64;
            let base3 = c0 + ((blk + 3) * STEP) as f64;

            let w0 = _mm512_add_pd(_mm512_set1_pd(base0), ramp);
            let w1 = _mm512_add_pd(_mm512_set1_pd(base1), ramp);
            let w2 = _mm512_add_pd(_mm512_set1_pd(base2), ramp);
            let w3 = _mm512_add_pd(_mm512_set1_pd(base3), ramp);

            let d0 = _mm512_loadu_pd(start_ptr.add((blk + 0) * STEP));
            let d1 = _mm512_loadu_pd(start_ptr.add((blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(start_ptr.add((blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(start_ptr.add((blk + 3) * STEP));

            s0 = _mm512_fmadd_pd(d0, w0, s0);
            s1 = _mm512_fmadd_pd(d1, w1, s1);
            s2 = _mm512_fmadd_pd(d2, w2, s2);
            s3 = _mm512_fmadd_pd(d3, w3, s3);

            blk += 4;
        }

        // remaining full blocks
        while blk < n_chunks {
            let base = c0 + (blk * STEP) as f64;
            let w = _mm512_add_pd(_mm512_set1_pd(base), ramp);
            let d = _mm512_loadu_pd(start_ptr.add(blk * STEP));
            s0 = _mm512_fmadd_pd(d, w, s0);
            blk += 1;
        }

        // tail
        if tail_len != 0 {
            let base = c0 + (n_chunks * STEP) as f64;
            let w_t = _mm512_add_pd(_mm512_set1_pd(base), ramp);
            let d_t = _mm512_maskz_loadu_pd(tmask, start_ptr.add(n_chunks * STEP));
            s0 = _mm512_fmadd_pd(d_t, w_t, s0);
        }

        let total = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
        *out.get_unchecked_mut(j) = _mm512_reduce_add_pd(total) * inv;
    }
}

#[derive(Debug, Clone)]
pub struct EpmaStream {
    period: usize, // ring capacity (p), EPMA window is p1 = p - 1
    offset: usize,
    p1: usize, // period - 1

    // ring buffer and head (head always points to the excluded slot)
    buffer: Vec<f64>,
    head: usize,

    // counts
    seen: usize,     // total updates seen
    included: usize, // number of elements currently in the EPMA window (<= p1)

    // rolling stats over last p1 samples (oldest->newest)
    sum: f64,    // S_t = sum x
    sum_c: f64,  // Kahan compensation for sum
    ramp: f64,   // R_t = sum i*x (i = 0..p1-1 oldest->newest)
    ramp_c: f64, // Kahan compensation for ramp

    // normalization
    c0: f64,       // 2 - offset
    inv_wsum: f64, // 1 / weight_sum
}

// small helper for compensated addition
#[inline(always)]
fn kahan_add(sum: &mut f64, c: &mut f64, x: f64) {
    let y = x - *c;
    let t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
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

        let p1 = period - 1;
        let c0 = 2.0 - (offset as f64);

        // Sum of weights over the p1-sample EPMA window:
        // W = p1*c0 + 0.5*(p1-1)*p1
        let p1f = p1 as f64;
        let wsum = p1f.mul_add(c0, 0.5 * (p1f - 1.0) * p1f);
        let inv_wsum = 1.0 / wsum; // if wsum==0, IEEE will propagate ±Inf/NaN like batch

        Ok(Self {
            period,
            offset,
            p1,

            // Using zeros; the excluded slot (head) is never read into stats,
            // and we only read x_out after we have at least p1 included elements.
            buffer: vec![0.0; period],
            head: 0,

            seen: 0,
            included: 0,

            sum: 0.0,
            sum_c: 0.0,
            ramp: 0.0,
            ramp_c: 0.0,

            c0,
            inv_wsum,
        })
    }

    /// O(1) update. Returns the raw input during warmup (<= period+offset+1),
    /// otherwise the EPMA value for the window ending at this tick.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let p = self.period;
        let p1m1 = (self.p1 - 1) as f64;

        // Oldest included value (drops only when we already have p1 included)
        let idx_out = (self.head + 1) % p;
        let x_out = if self.included == self.p1 {
            // window full: this will be dropped from the included set
            self.buffer[idx_out]
        } else {
            0.0
        };

        // Write the new value into the excluded slot, then advance head
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % p;
        self.seen += 1;

        // Update rolling statistics
        if self.included < self.p1 {
            // growth phase: window size m increases by 1, weights are 0..m
            let m = self.included as f64;

            kahan_add(&mut self.sum, &mut self.sum_c, value);
            // R_{new} = R_{old} + m * value
            kahan_add(&mut self.ramp, &mut self.ramp_c, m * value);

            self.included += 1;
        } else {
            // steady phase: constant-size window (size = p1)
            let s_old = self.sum;

            // S_{new} = S_{old} + (value - x_out)
            let delta_s = value - x_out;
            kahan_add(&mut self.sum, &mut self.sum_c, delta_s);

            // R_{new} = R_{old} + [(x_out - S_{old}) + (p1-1)*value]
            let delta_r = p1m1.mul_add(value, x_out - s_old);
            kahan_add(&mut self.ramp, &mut self.ramp_c, delta_r);
        }

        // Warmup convention: return raw value until we pass (period + offset + 1) samples
        if self.seen <= (self.period + self.offset + 1) {
            return Some(value);
        }

        // num = c0 * S + R
        let num = self.c0.mul_add(self.sum, self.ramp);
        Some(num * self.inv_wsum)
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
    // Robust axis expansion:
    // - step == 0 => singleton
    // - reversed bounds => expand over min..=max
    // - keep behavior stable for normal inputs
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 {
            return vec![start];
        }
        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
        (lo..=hi).step_by(step).collect()
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
        other => return Err(EpmaError::InvalidKernelForBatch(other)),
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
    // First calculate dimensions to allocate the right size
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();

    // Checked arithmetic guard (no behavior change unless overflow would occur)
    let _total_cells = rows
        .checked_mul(cols)
        .ok_or(EpmaError::SizeOverflow { rows, cols })?;

    // Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Pass to inner function which will initialize and compute
    let combos = epma_batch_inner_into_uninit(data, sweep, kern, parallel, &mut buf_mu)?;

    // Convert from MaybeUninit to Vec<f64>
    let values = unsafe {
        let mut buf_guard = ManuallyDrop::new(buf_mu);
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(EpmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[cfg(any(feature = "python", feature = "wasm"))]
#[inline(always)]
pub fn epma_batch_inner_into(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EpmaParams>, EpmaError> {
    // Safety: We're creating a MaybeUninit view of the output buffer
    // The epma_batch_inner_into_uninit function will initialize all values
    let buf_mu = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    epma_batch_inner_into_uninit(data, sweep, kern, parallel, buf_mu)
}

#[inline(always)]
fn epma_batch_inner_into_uninit(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
    parallel: bool,
    buf_mu: &mut [MaybeUninit<f64>],
) -> Result<Vec<EpmaParams>, EpmaError> {
    if data.is_empty() {
        return Err(EpmaError::EmptyInputData);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        // Defensive: axis_usize should prevent empty expansions, but keep a typed error.
        return Err(EpmaError::InvalidRange { start: 0, end: 0, step: 0 });
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
    let cols = data.len();

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() + c.offset.unwrap() + 1)
        .collect();
    init_matrix_prefixes(buf_mu, cols, &warm);

    // Row-specific batch: shared prefix sums across rows (O(1) per timestamp)
    // Enabled for SIMD batch selections; scalar batch falls back to per-row kernel
    let use_prefix = {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        {
            matches!(kern, Kernel::Avx2 | Kernel::Avx512)
        }
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            false
        }
    };

    if use_prefix {
        // Build prefix sums P[k] = sum_{u<=k} x[u] and Q[k] = sum_{u<=k} u*x[u]
        let mut p: Vec<f64> = Vec::with_capacity(cols);
        let mut q: Vec<f64> = Vec::with_capacity(cols);
        let mut ps = 0.0f64;
        let mut qs = 0.0f64;
        for (idx, &x) in data.iter().enumerate() {
            ps += x;
            qs = (idx as f64).mul_add(x, qs);
            p.push(ps);
            q.push(qs);
        }

        let do_row_prefix = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
            let period = combos[row].period.unwrap();
            let offset = combos[row].offset.unwrap();
            let p1 = period - 1;
            let c0 = 2.0 - (offset as f64);
            let p1f = p1 as f64;
            let wsum = p1f.mul_add(c0, 0.5 * (p1f - 1.0) * p1f);
            let inv = 1.0 / wsum;
            let warmup = warm[row];

            // Safety: we will initialize all entries from warmup..cols
            let dst = unsafe {
                core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
            };

            for j in warmup..cols {
                let a = j + 1 - p1; // window start
                let b = j; // inclusive end
                           // SumX = P[b] - P[a-1]
                let sumx = if a > 0 { p[b] - p[a - 1] } else { p[b] };
                // SumAbs = Q[b] - Q[a-1]
                let sum_abs = if a > 0 { q[b] - q[a - 1] } else { q[b] };
                // Sum_i x = SumAbs - a * SumX
                let sum_ix = sum_abs - (a as f64) * sumx;
                let y = (c0 * sumx + sum_ix) * inv;
                // Safety: warmup regions were prefilled; we only write post-warmup
                unsafe {
                    *dst.get_unchecked_mut(j) = y;
                }
            }
        };

        // Run all rows with shared prefixes
        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                buf_mu
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(row, slice)| do_row_prefix(row, slice));
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
                    do_row_prefix(row, slice);
                }
            }
        } else {
            for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
                do_row_prefix(row, slice);
            }
        }
    } else {
        // Helper that computes one row into a &mut [MaybeUninit<f64>] using per-row kernel
        let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
            let period = combos[row].period.unwrap();
            let offset = combos[row].offset.unwrap();

            // Cast this slice only; we know the kernel will overwrite every cell after warmup
            let dst =
                core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

            match kern {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => epma_row_avx512(data, first, period, offset, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => epma_row_avx2(data, first, period, offset, dst),
                _ => epma_row_scalar(data, first, period, offset, dst),
            }
        };

        // Run all rows (parallel or serial) on the MaybeUninit buffer
        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                buf_mu
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(row, slice)| do_row(row, slice));
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
                    do_row(row, slice);
                }
            }
        } else {
            for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
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
    use std::error::Error;

    #[test]
    fn test_epma_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Prepare a small but non-trivial input with some NaNs at the start
        let mut data = Vec::with_capacity(256);
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..253u32 {
            let v = (i as f64).sin() * 10.0 + (i as f64) * 0.01;
            data.push(v);
        }

        let input = EpmaInput::from_slice(&data, EpmaParams::default());

        // Baseline via existing Vec-returning API
        let baseline = epma(&input)?.values;

        // Preallocate output and call the new into API
        let mut out = vec![0.0; data.len()];
        #[cfg(not(feature = "wasm"))]
        {
            epma_into(&input, &mut out)?;
        }

        // Helper for NaN-aware equality
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        assert_eq!(baseline.len(), out.len());
        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }

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

        // Enhanced property testing strategy - expanded range for better coverage
        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                // Generate data with length >= period + offset + warmup buffer
                // Need at least period + max_offset + 1, plus some buffer for testing
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    (period * 2 + 10)..500, // Ensure enough data even for max offset
                ),
                Just(period),
                0usize..period, // offset must be < period
            )
        });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period, offset)| {
				let params = EpmaParams {
					period: Some(period),
					offset: Some(offset),
				};
				let input = EpmaInput::from_slice(&data, params);

				// Test with the specified kernel
				let EpmaOutput { values: out } = epma_with_kernel(&input, kernel).unwrap();

				// Also compute reference with scalar kernel for consistency check
				let EpmaOutput { values: ref_out } = epma_with_kernel(&input, Kernel::Scalar).unwrap();

				// Find first non-NaN index
				let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				// The implementations start outputting at first_valid + period + offset + 1
				// So everything before that index should be NaN
				let warmup = first_valid + period + offset + 1;

				// Property 1: Warmup period validation
				for i in 0..warmup.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"[{}] Expected NaN during warmup at index {}, got {}",
						test_name,
						i,
						out[i]
					);
				}

				// Property 2: First valid output at correct index
				// Note: EPMA may still produce NaN if weight_sum is 0 or if all data in window is NaN
				if warmup < out.len() && data[warmup].is_finite() {
					// Check if weight_sum would be non-zero
					let p1 = period - 1;
					let mut weight_sum = 0.0;
					for i in 0..p1 {
						let w = (period as i32 - i as i32 - offset as i32) as f64;
						weight_sum += w;
					}

					// Only check for valid output if weight_sum is non-zero
					if weight_sum.abs() > 1e-10 {
						prop_assert!(
							!out[warmup].is_nan(),
							"[{}] Expected valid value at warmup index {}, got NaN",
							test_name,
							warmup
						);
					}
				}

				// Property 3: Values after warmup should be finite (but can be outside input bounds)
				// STRICT CHECK: when weight_sum is zero, ALL kernels should produce consistent NaN/Inf
				let p1 = period - 1;
				let mut weight_sum = 0.0;
				for i in 0..p1 {
					let w = (period as i32 - i as i32 - offset as i32) as f64;
					weight_sum += w;
				}

				if weight_sum.abs() > 1e-10 {
					// Normal case: weight_sum is non-zero, outputs should be finite
					for i in warmup..data.len() {
						let y = out[i];
						prop_assert!(
							y.is_finite(),
							"[{}] EPMA output at index {} is not finite: {} (period={}, offset={}, weight_sum={})",
							test_name,
							i,
							y,
							period,
							offset,
							weight_sum
						);
					}
				} else {
					// Edge case: weight_sum is zero, should produce consistent behavior across all kernels
					// Either both NaN or both Inf (division by zero)
					for i in warmup..data.len() {
						let both_nan = out[i].is_nan() && ref_out[i].is_nan();
						let both_inf = out[i].is_infinite() && ref_out[i].is_infinite();
						prop_assert!(
							both_nan || both_inf,
							"[{}] With weight_sum=0, expected consistent NaN or Inf at index {} but got: kernel={}, scalar={} (period={}, offset={})",
							test_name,
							i,
							out[i],
							ref_out[i],
							period,
							offset
						);
					}
				}

				// Property 4: Special case - period=2, offset=0
				// EPMA uses period-1 values, so for period=2 it uses 1 value
				// Weight = (2 - 0 - 0) = 2, but since p1=1, it just copies the value
				if period == 2 && offset == 0 && warmup < data.len() {
					// For period=2, p1=1, so it uses only 1 value with weight (2-0-0)=2
					for i in warmup..data.len() {
						if data[i].is_finite() {
							// EPMA with period=2 just uses the current value
							prop_assert!(
								(out[i] - data[i]).abs() < 1e-9,
								"[{}] Period=2,offset=0 mismatch at {}: got {}, expected {}",
								test_name,
								i,
								out[i],
								data[i]
							);
						}
					}
				}

				// Property 5: Constant non-zero data should produce that constant
				// (after warmup, regardless of weights)
				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && data.iter().any(|x| x.is_finite() && x.abs() > 1e-10) {
					let constant = *data.iter().find(|x| x.is_finite()).unwrap();
					// Check if weight_sum would be non-zero
					let p1 = period - 1;
					let mut weight_sum = 0.0;
					for i in 0..p1 {
						let w = (period as i32 - i as i32 - offset as i32) as f64;
						weight_sum += w;
					}

					if weight_sum.abs() > 1e-10 {
						for i in warmup..data.len() {
							prop_assert!(
								(out[i] - constant).abs() < 1e-9,
								"[{}] Constant data mismatch at {}: got {}, expected {}",
								test_name,
								i,
								out[i],
								constant
							);
						}
					}
				}

				// Property 6: Kernel consistency - all kernels should produce identical results
				// Start from warmup (not warmup itself) to avoid boundary issues
				for i in (warmup.saturating_add(1))..data.len() {
					let y = out[i];
					let r = ref_out[i];

					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"[{}] finite/NaN mismatch at idx {}: {} vs {}",
							test_name,
							i,
							y,
							r
						);
						continue;
					}

					let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
					// Check for relative error for large values
					let rel_error = if r.abs() > 1e-10 {
						(y - r).abs() / r.abs()
					} else {
						(y - r).abs()
					};

					// Allow small relative error (0.01%) or small absolute error or small ULP difference
					prop_assert!(
						rel_error <= 1e-4 || (y - r).abs() <= 1e-9 || ulp_diff <= 100,
						"[{}] Kernel mismatch at idx {}: {} vs {} (ULP={}, rel_err={})",
						test_name,
						i,
						y,
						r,
						ulp_diff,
						rel_error
					);
				}

				// Property 7: EPMA polynomial weight verification
				// Verify the weighted sum formula for a few random positions
				if warmup + 5 < data.len() {
					// Build weights for verification
					let p1 = period - 1;
					let mut weights = Vec::with_capacity(p1);
					let mut weight_sum = 0.0;
					for i in 0..p1 {
						let w = (period as i32 - i as i32 - offset as i32) as f64;
						weights.push(w);
						weight_sum += w;
					}

					// Only verify formula if weight_sum is non-zero
					if weight_sum.abs() > 1e-10 {
						// Test a few positions, avoiding the exact warmup boundary
						for idx in [warmup + 1, warmup + 2, data.len() - 1].iter().copied() {
							if idx > warmup && idx < data.len() {
								let start = idx + 1 - p1;
								let mut expected_sum = 0.0;
								for i in 0..p1 {
									expected_sum += data[start + i] * weights[p1 - 1 - i];
								}
								let expected = expected_sum / weight_sum;

								// Both should be finite for comparison
								if out[idx].is_finite() && expected.is_finite() {
									// Use relative tolerance for large values, absolute for small
									let tolerance = if expected.abs() > 1000.0 {
										expected.abs() * 1e-12  // Relative tolerance
									} else {
										1e-9  // Absolute tolerance
									};
									prop_assert!(
										(out[idx] - expected).abs() < tolerance,
										"[{}] EPMA formula mismatch at {}: got {}, expected {} (diff: {})",
										test_name,
										idx,
										out[idx],
										expected,
										(out[idx] - expected).abs()
									);
								} else {
									// Both should have the same NaN/Inf status
									prop_assert!(
										out[idx].is_nan() == expected.is_nan() &&
										out[idx].is_infinite() == expected.is_infinite(),
										"[{}] EPMA formula NaN/Inf mismatch at {}: got {}, expected {}",
										test_name,
										idx,
										out[idx],
										expected
									);
								}
							}
						}
					}
				}

				// Property 8: Edge case - offset = period - 1
				// This gives weights like [1, 0, -1, -2, ...] which can produce extreme values
				// When weight_sum is zero, NaN/Inf is expected
				if offset == period - 1 && warmup < data.len() && weight_sum.abs() > 1e-10 {
					// Just verify outputs are finite, as they can be quite extreme
					for i in warmup..data.len() {
						prop_assert!(
							out[i].is_finite(),
							"[{}] Edge case offset={} produced non-finite at {}",
							test_name,
							offset,
							i
						);
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }
    fn check_epma_invalid_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Test specific parameter combinations that produce weight_sum = 0
        // These should either error or produce consistent NaN
        let zero_weight_cases = vec![
            (4, 3), // weights: [1, 0, -1] -> sum = 0
            (5, 3), // weights: [2, 1, 0, -1] -> sum = 2
            (6, 4), // weights: [2, 1, 0, -1, -2] -> sum = 0
            (8, 6), // weights: [2, 1, 0, -1, -2, -3, -4] -> sum = -7
        ];

        for (period, offset) in zero_weight_cases {
            // Calculate actual weight sum
            let p1 = period - 1;
            let mut weight_sum = 0.0;
            for i in 0..p1 {
                let w = (period as i32 - i as i32 - offset as i32) as f64;
                weight_sum += w;
            }

            // Test with some simple data
            let data = vec![1.0; period * 2];
            let params = EpmaParams {
                period: Some(period),
                offset: Some(offset),
            };
            let input = EpmaInput::from_slice(&data, params);

            // If weight_sum is zero, this should produce NaN consistently
            if weight_sum.abs() < 1e-10 {
                let out = epma_with_kernel(&input, kernel)?;
                let scalar_out = epma_with_kernel(&input, Kernel::Scalar)?;

                let warmup = period + offset + 1;
                for i in warmup..data.len() {
                    let both_nan = out.values[i].is_nan() && scalar_out.values[i].is_nan();
                    let both_inf =
                        out.values[i].is_infinite() && scalar_out.values[i].is_infinite();
                    assert!(
						both_nan || both_inf,
						"[{}] Period={}, Offset={} (weight_sum=0) should produce consistent NaN or Inf, got kernel={}, scalar={}",
						test_name,
						period,
						offset,
						out.values[i],
						scalar_out.values[i]
					);
                }
            }
        }

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

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_epma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to better catch uninitialized memory bugs
        let test_cases = vec![
            // Default parameters
            EpmaParams::default(),
            // Small period
            EpmaParams {
                period: Some(2),
                offset: Some(0),
            },
            // Medium period with various offsets
            EpmaParams {
                period: Some(5),
                offset: Some(1),
            },
            EpmaParams {
                period: Some(10),
                offset: Some(3),
            },
            EpmaParams {
                period: Some(10),
                offset: Some(9),
            },
            // Large period
            EpmaParams {
                period: Some(20),
                offset: Some(5),
            },
            EpmaParams {
                period: Some(30),
                offset: Some(10),
            },
            // Edge case: period - 1 offset
            EpmaParams {
                period: Some(15),
                offset: Some(14),
            },
        ];

        for params in test_cases {
            let input = EpmaInput::from_candles(&candles, "close", params.clone());
            let output = epma_with_kernel(&input, kernel)?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params period={:?}, offset={:?}",
                        test_name, val, bits, i, params.period, params.offset
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params period={:?}, offset={:?}",
                        test_name, val, bits, i, params.period, params.offset
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params period={:?}, offset={:?}",
                        test_name, val, bits, i, params.period, params.offset
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_epma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
                // Test WASM SIMD128 implementation
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
    generate_all_epma_tests!(
        check_epma_partial_params,
        check_epma_accuracy,
        check_epma_default_candles,
        check_epma_zero_period,
        check_epma_period_exceeds_length,
        check_epma_very_small_dataset,
        check_epma_empty_input,
        check_epma_invalid_offset,
        check_epma_invalid_params,
        check_epma_reinput,
        check_epma_nan_handling,
        check_epma_streaming,
        check_epma_property,
        check_epma_no_poison
    );

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_epma_simd128_correctness() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let period = 5;
        let offset = 2;

        // Compute with scalar version (force scalar kernel)
        let params = EpmaParams {
            period: Some(period),
            offset: Some(offset),
        };
        let input = EpmaInput::from_slice(&data, params);

        // First compute using scalar explicitly
        let mut scalar_out = vec![0.0; data.len()];
        epma_scalar(&data, period, offset, 0, &mut scalar_out);

        // Compute with SIMD128 (via Scalar kernel on WASM which will use SIMD128)
        let simd128_output = epma_with_kernel(&input, Kernel::Scalar).unwrap();

        // Compare results after warmup period
        let warmup = period + offset + 1;
        for i in warmup..data.len() {
            assert!(
                (scalar_out[i] - simd128_output.values[i]).abs() < 1e-10,
                "SIMD128 mismatch at index {}: scalar={}, simd128={}",
                i,
                scalar_out[i],
                simd128_output.values[i]
            );
        }
    }

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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations to better catch uninitialized memory bugs
        let test_configs = vec![
            // Small periods with various offsets
            ((2, 5, 1), (0, 2, 1)),
            // Medium periods with edge-case offsets
            ((10, 20, 5), (0, 19, 3)),
            // Large periods
            ((20, 30, 2), (5, 15, 5)),
            // Edge case: large offset relative to period
            ((15, 25, 5), (10, 14, 2)),
            // Dense parameter sweep
            ((5, 10, 1), (0, 9, 1)),
        ];

        for (period_range, offset_range) in test_configs {
            let output = EpmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_range.0, period_range.1, period_range.2)
                .offset_range(offset_range.0, offset_range.1, offset_range.2)
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
                let params = &output.combos[row];

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, offset={:?})",
                        test, val, bits, row, col, params.period, params.offset
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, offset={:?})",
                        test, val, bits, row, col, params.period, params.offset
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, offset={:?})",
                        test, val, bits, row, col, params.period, params.offset
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

    #[test]
    fn test_invalid_output_len_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = EpmaParams {
            period: Some(3),
            offset: Some(1),
        };
        let input = EpmaInput::from_slice(&data, params);
        let mut wrong_size_dst = vec![0.0; 3]; // Wrong size, should be 5

        let result = epma_into_slice(&mut wrong_size_dst, &input, Kernel::Scalar);
        assert!(result.is_err());

        // Verify we get the correct error variant
        match result {
            Err(EpmaError::OutputLengthMismatch { expected, got }) => {
                assert_eq!(expected, 5);
                assert_eq!(got, 3);
            }
            _ => panic!("Expected OutputLengthMismatch error"),
        }
    }

    #[test]
    fn test_invalid_kernel_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sweep = EpmaBatchRange {
            period: (3, 5, 1),
            offset: (1, 2, 1),
        };

        // Try to use a non-batch kernel with batch operation
        let result = epma_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(result.is_err());

        // Verify we get the correct error variant
        match result {
            Err(EpmaError::InvalidKernelForBatch(k)) => {
                assert_eq!(k, Kernel::Scalar);
            }
            _ => panic!("Expected InvalidKernelForBatch error"),
        }
    }
}

// Python bindings
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray2;
#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[cfg(feature = "python")]
#[pyfunction(name = "epma")]
#[pyo3(signature = (data, period, offset, kernel=None))]
pub fn epma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    offset: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let params = EpmaParams {
        period: Some(period),
        offset: Some(offset),
    };
    let input = EpmaInput::from_slice(slice_in, params);
    let kern = validate_kernel(kernel, false)?;

    // Zero extra copies: allocate NumPy output then compute directly into it.
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| epma_into_slice(out_slice, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyfunction(name = "epma_batch")]
#[pyo3(signature = (data, period_range, offset_range, kernel=None))]
pub fn epma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    offset_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let slice_in = data.as_slice()?;

    let sweep = EpmaBatchRange {
        period: period_range,
        offset: offset_range,
    };

    // Build combos to know rows/cols up front.
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Allocate a flat NumPy buffer and compute into it directly. No NaN prefill copy.
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos_done = py
        .allow_threads(|| {
            let actual = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match actual {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            epma_batch_inner_into(slice_in, &sweep, simd, true, out_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos_done
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "offsets",
        combos_done
            .iter()
            .map(|p| p.offset.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "epma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, offset_range, device_id=0))]
pub fn epma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    offset_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<EpmaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let sweep = EpmaBatchRange {
        period: period_range,
        offset: offset_range,
    };
    let slice_in = data_f32.as_slice()?;

    let (inner, ctx_guard, dev_id) = py.allow_threads(|| {
        let cuda = CudaEpma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let arr = cuda
            .epma_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev))
    })?;

    Ok(EpmaDeviceArrayF32Py { inner, _ctx_guard: ctx_guard, device_id: dev_id as u32 })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "epma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, offset, device_id=0))]
pub fn epma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    offset: usize,
    device_id: usize,
) -> PyResult<EpmaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let rows = shape[0];
    let cols = shape[1];
    let params = EpmaParams {
        period: Some(period),
        offset: Some(offset),
    };

    let (inner, ctx_guard, dev_id) = py.allow_threads(|| {
        let cuda = CudaEpma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let arr = cuda
            .epma_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev))
    })?;

    Ok(EpmaDeviceArrayF32Py { inner, _ctx_guard: ctx_guard, device_id: dev_id as u32 })
}

// EPMA-specific CUDA Array Interface v3 + DLPack handle with context guard
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct EpmaDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx_guard: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl EpmaDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        // 2D row-major FP32: (rows, cols)
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Producing stream is synchronized before returning; omit 'stream' per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // 2 == kDLCUDA
        (2, self.device_id as i32)
    }

    fn __dlpack__<'py>(&self, py: Python<'py>, _stream: Option<i64>) -> PyResult<PyObject> {
        use std::ffi::c_void;

        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLManagedTensor {
            dl_tensor: DLTensor,
            manager_ctx: *mut c_void,
            deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
        }
        struct DlpGuard {
            _shape: Box<[i64; 2]>,
            _strides: Box<[i64; 2]>,
            _ctx: Arc<Context>,
        }

        extern "C" fn managed_deleter(p: *mut DLManagedTensor) {
            unsafe {
                if p.is_null() { return; }
                let guard_ptr = (*p).manager_ctx as *mut DlpGuard;
                if !guard_ptr.is_null() {
                    drop(Box::from_raw(guard_ptr));
                }
                drop(Box::from_raw(p));
            }
        }

        extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            unsafe {
                let name = b"dltensor\0";
                let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr() as *const _) as *mut DLManagedTensor;
                if !ptr.is_null() {
                    if let Some(del) = (*ptr).deleter { del(ptr); }
                    let used = b"used_dltensor\0";
                    pyo3::ffi::PyCapsule_SetName(capsule, used.as_ptr() as *const _);
                }
            }
        }

        let shape = Box::new([self.inner.rows as i64, self.inner.cols as i64]);
        let strides = Box::new([self.inner.cols as i64, 1i64]);
        let data_ptr = self.inner.device_ptr() as usize as *mut c_void;

        let guard = Box::new(DlpGuard { _shape: shape, _strides: strides, _ctx: self._ctx_guard.clone() });
        let guard_ref = &*guard;
        let mt = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice { device_type: 2, device_id: self.device_id as i32 },
                ndim: 2,
                dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                shape: guard_ref._shape.as_ptr() as *mut i64,
                strides: guard_ref._strides.as_ptr() as *mut i64,
                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(guard) as *mut c_void,
            deleter: Some(managed_deleter),
        });
        let mt_raw = Box::into_raw(mt);
        let name = b"dltensor\0";
        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(mt_raw as *mut c_void, name.as_ptr() as *const _, Some(capsule_destructor))
        };
        if capsule.is_null() {
            unsafe { managed_deleter(mt_raw); }
            return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create DLPack capsule"));
        }
        Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "EpmaStream")]
pub struct EpmaStreamPy {
    stream: EpmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EpmaStreamPy {
    #[new]
    fn new(period: usize, offset: usize) -> PyResult<Self> {
        let params = EpmaParams {
            period: Some(period),
            offset: Some(offset),
        };
        let stream =
            EpmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

// WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_js(data: &[f64], period: usize, offset: usize) -> Result<Vec<f64>, JsValue> {
    let params = EpmaParams {
        period: Some(period),
        offset: Some(offset),
    };
    let input = EpmaInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    epma_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EpmaBatchConfig {
    pub period_range: (usize, usize, usize),
    pub offset_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EpmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EpmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = epma_batch)]
pub fn epma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: EpmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EpmaBatchRange {
        period: cfg.period_range,
        offset: cfg.offset_range,
    };
    let out = epma_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&EpmaBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    offset: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = EpmaParams {
            period: Some(period),
            offset: Some(offset),
        };
        let input = EpmaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // Safe in-place fallback
            let mut tmp = vec![0.0; len];
            epma_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            epma_into_slice(
                std::slice::from_raw_parts_mut(out_ptr, len),
                &input,
                detect_best_kernel(),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn epma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    p_start: usize,
    p_end: usize,
    p_step: usize,
    o_start: usize,
    o_end: usize,
    o_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = EpmaBatchRange {
            period: (p_start, p_end, p_step),
            offset: (o_start, o_end, o_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        epma_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
