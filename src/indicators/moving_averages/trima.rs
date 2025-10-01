//! # Triangular Moving Average (TRIMA)
//!
//! A moving average computed by averaging an underlying Simple Moving Average (SMA) over
//! the specified `period`, resulting in a smoother output than a single SMA.
//! TRIMA supports different compute kernels and batch processing via builder APIs.
//!
//! ## Parameters
//! - **period**: Window size (must be > 3).
//!
//! ## Returns
//! - **`Ok(TrimaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(TrimaError)`** otherwise.
//!
//! ## Developer Status
//! - SIMD enabled (AVX2/AVX512): vectorizes initial accumulation; rolling core remains scalar.
//! - Streaming update: O(1) via two rolling sums; no full-size temporaries.
//! - Memory: uses zero-copy helpers for outputs; tiny O(period) ring buffer only.
//! - Rationale: TRIMA = SMA(SMA(x, m1), m2); SIMD helps initial sums; main loop is sequential.
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaTrima;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
use crate::indicators::sma::{sma, SmaData, SmaInput, SmaParams};
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
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for TrimaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrimaData::Slice(slice) => slice,
            TrimaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrimaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrimaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TrimaParams {
    pub period: Option<usize>,
}

impl Default for TrimaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct TrimaInput<'a> {
    pub data: TrimaData<'a>,
    pub params: TrimaParams,
}

impl<'a> TrimaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TrimaParams) -> Self {
        Self {
            data: TrimaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TrimaParams) -> Self {
        Self {
            data: TrimaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TrimaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrimaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TrimaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrimaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TrimaOutput, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        let i = TrimaInput::from_candles(c, "close", p);
        trima_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TrimaOutput, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        let i = TrimaInput::from_slice(d, p);
        trima_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TrimaStream, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        TrimaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TrimaError {
    #[error("trima: All values are NaN.")]
    AllValuesNaN,

    #[error("trima: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("trima: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("trima: Period too small: {period}")]
    PeriodTooSmall { period: usize },

    #[error("trima: No data provided.")]
    NoData,
}

#[inline]
pub fn trima(input: &TrimaInput) -> Result<TrimaOutput, TrimaError> {
    trima_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn trima_prepare<'a>(
    input: &'a TrimaInput,
    kernel: Kernel,
) -> Result<
    (
        // data
        &'a [f64],
        // period
        usize,
        // m1
        usize,
        // m2
        usize,
        // first
        usize,
        // chosen
        Kernel,
    ),
    TrimaError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(TrimaError::NoData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrimaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(TrimaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period <= 3 {
        return Err(TrimaError::PeriodTooSmall { period });
    }
    if (len - first) < period {
        return Err(TrimaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, period, m1, m2, first, chosen))
}

#[inline(always)]
fn trima_compute_into(
    data: &[f64],
    period: usize,
    m1: usize,
    m2: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                trima_simd128(data, m1, m2, first, out);
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trima_scalar_optimized(data, period, m1, m2, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                trima_avx2(data, period, first, out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trima_avx512(data, period, first, out);
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                trima_scalar_optimized(data, period, m1, m2, first, out)
            }
            Kernel::Auto => {
                // Auto should have been resolved to a specific kernel by this point
                trima_scalar_optimized(data, period, m1, m2, first, out)
            }
        }
    }
}

#[inline(always)]
unsafe fn trima_scalar_optimized(
    data: &[f64],
    period: usize,
    m1: usize,
    m2: usize,
    first: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(data.len(), out.len());
    let n = data.len();
    if n == 0 {
        return;
    }
    let warm = first + period - 1;
    if warm >= n {
        // nothing to write (caller already validated lengths)
        return;
    }

    // Precompute reciprocals to turn divisions into multiplies
    let inv_m1 = 1.0 / (m1 as f64);
    let inv_m2 = 1.0 / (m2 as f64);

    // 1) Initial m1-sum (partially unrolled)
    let base = data.as_ptr().add(first);
    let mut sum1 = 0.0;
    {
        let mut j = 0usize;
        let end_unroll = m1 & !3usize; // floor to multiple of 4
        while j < end_unroll {
            sum1 += *base.add(j)
                + *base.add(j + 1)
                + *base.add(j + 2)
                + *base.add(j + 3);
            j += 4;
        }
        while j < m1 {
            sum1 += *base.add(j);
            j += 1;
        }
    }

    // 2) Build m2 of SMA1s into a small ring
    let mut ring: Vec<f64> = Vec::with_capacity(m2);
    let mut sum2 = 0.0;

    // time index t corresponds to the index of the SMA1 we just produced
    let mut t = first + m1 - 1;

    // Pointers to advance sum1 in O(1) without bounds checks
    let mut p_new = data.as_ptr().add(first + m1); // element to be added next
    let mut p_old = data.as_ptr().add(first); // element to be removed next

    // First SMA1
    {
        let s1 = sum1 * inv_m1;
        ring.push(s1);
        sum2 += s1;
    }

    // Fill the remaining (m2 - 1) SMA1s and accumulate sum2
    while ring.len() < m2 {
        t += 1; // move to the index for the next SMA1
        // Maintain the rolling m1-sum
        sum1 += *p_new - *p_old;
        p_new = p_new.add(1);
        p_old = p_old.add(1);

        let s1 = sum1 * inv_m1;
        ring.push(s1);
        sum2 += s1;
    }

    // At this point, t == warm and sum2 holds the sum of the last m2 SMA1s.
    *out.get_unchecked_mut(warm) = sum2 * inv_m2;

    // 3) Main rolling loop
    let mut head = 0usize; // ring write index / oldest element
    t += 1; // next index to produce TRIMA for
    while t < n {
        // Update m1 rolling sum
        sum1 += *p_new - *p_old;
        p_new = p_new.add(1);
        p_old = p_old.add(1);

        // New SMA1 and ring maintenance for sum2
        let new_s1 = sum1 * inv_m1;
        let old_s1 = *ring.get_unchecked(head);
        sum2 += new_s1 - old_s1;
        *ring.get_unchecked_mut(head) = new_s1;

        head += 1;
        if head == m2 {
            head = 0;
        }

        // Write TRIMA aligned at index t
        *out.get_unchecked_mut(t) = sum2 * inv_m2;

        t += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn trima_simd128(data: &[f64], m1: usize, m2: usize, first: usize, out: &mut [f64]) {
    use core::arch::wasm32::*;

    const STEP: usize = 2;
    let n = data.len();

    // Use the optimized algorithm from scalar but with SIMD acceleration
    // First pass: compute SMA of length m1
    let mut sma1 = vec![f64::NAN; n];

    if first + m1 <= n {
        // Initialize first value with SIMD
        let chunks = m1 / STEP;
        let tail = m1 % STEP;

        let mut acc = f64x2_splat(0.0);
        for i in 0..chunks {
            let idx = first + i * STEP;
            let d = v128_load(data.as_ptr().add(idx) as *const v128);
            acc = f64x2_add(acc, d);
        }

        let mut sum = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
        if tail != 0 {
            sum += data[first + chunks * STEP];
        }

        sma1[first + m1 - 1] = sum / m1 as f64;

        // Continue with scalar running sum for efficiency
        for i in (first + m1)..n {
            sum += data[i] - data[i - m1];
            sma1[i] = sum / m1 as f64;
        }
    }

    // Second pass: compute SMA of length m2 on the first SMA
    if first + m1 + m2 - 1 <= n {
        // Find first valid value in sma1
        let sma1_first = first + m1 - 1;

        // Initialize second SMA with SIMD
        let chunks2 = m2 / STEP;
        let tail2 = m2 % STEP;

        let mut acc2 = f64x2_splat(0.0);
        for i in 0..chunks2 {
            let idx = sma1_first + i * STEP;
            let d = v128_load(sma1.as_ptr().add(idx) as *const v128);
            acc2 = f64x2_add(acc2, d);
        }

        let mut sum2 = f64x2_extract_lane::<0>(acc2) + f64x2_extract_lane::<1>(acc2);
        if tail2 != 0 {
            sum2 += sma1[sma1_first + chunks2 * STEP];
        }

        out[sma1_first + m2 - 1] = sum2 / m2 as f64;

        // Continue with scalar running sum
        for i in (sma1_first + m2)..n {
            sum2 += sma1[i] - sma1[i - m2];
            out[i] = sum2 / m2 as f64;
        }
    }
}

pub fn trima_with_kernel(input: &TrimaInput, kernel: Kernel) -> Result<TrimaOutput, TrimaError> {
    let (data, period, m1, m2, first, chosen) = trima_prepare(input, kernel)?;
    let len = data.len();
    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);
    trima_compute_into(data, period, m1, m2, first, chosen, &mut out);
    Ok(TrimaOutput { values: out })
}

#[inline]
pub fn trima_into_slice(
    output: &mut [f64],
    input: &TrimaInput,
    kernel: Kernel,
) -> Result<(), TrimaError> {
    let (data, period, m1, m2, first, chosen) = trima_prepare(input, kernel)?;

    // Compute TRIMA values first
    trima_compute_into(data, period, m1, m2, first, chosen, output);

    // Then set warmup period to NaN (like ALMA does)
    let warmup = first + period - 1;
    for i in 0..warmup.min(output.len()) {
        output[i] = f64::NAN;
    }

    Ok(())
}

#[inline]
/// Classic kernel - optimized loop-jammed implementation without function calls
pub fn trima_scalar_classic(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // ─── OPTIMIZED TWO-PASS SMA APPROACH (LOOP-JAMMED) ───
    //
    // Let m1 = (period+1)/2,  m2 = period − m1 + 1.
    // First pass: compute the m1-period SMA of `data` inline.
    // Second pass: compute the m2-period SMA of that first-pass result inline.
    // This avoids function calls and allocations.

    let n = data.len();
    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;

    // Allocate intermediate buffer for first pass
    let mut sma1 = vec![f64::NAN; n];

    // FIRST PASS: Inline SMA of length m1
    if first + m1 <= n {
        // Initial sum for first valid SMA value
        let mut sum1 = 0.0;
        for j in 0..m1 {
            sum1 += data[first + j];
        }
        sma1[first + m1 - 1] = sum1 / m1 as f64;

        // Rolling sum for remaining values
        for i in (first + m1)..n {
            sum1 += data[i] - data[i - m1];
            sma1[i] = sum1 / m1 as f64;
        }
    }

    // SECOND PASS: Inline SMA of length m2 on sma1 results
    let warmup_end = first + period - 1;
    if warmup_end < n {
        // Find first valid index in sma1
        let first_valid_sma1 = first + m1 - 1;
        let first_valid_sma2 = first_valid_sma1 + m2 - 1;

        if first_valid_sma2 < n {
            // Initial sum for first valid TRIMA value
            let mut sum2 = 0.0;
            for j in 0..m2 {
                sum2 += sma1[first_valid_sma1 + j];
            }

            if warmup_end < n {
                out[warmup_end] = sum2 / m2 as f64;
            }

            // Rolling sum for remaining values
            for i in (warmup_end + 1)..n {
                let old_idx = i - m2;
                if old_idx >= first_valid_sma1 {
                    sum2 += sma1[i] - sma1[old_idx];
                    out[i] = sum2 / m2 as f64;
                }
            }
        }
    }
}

/// Regular kernel - uses function calls for flexibility
pub fn trima_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // ─── TWO-PASS SMA APPROACH ───
    //
    // Let m1 = (period+1)/2,  m2 = period − m1 + 1.
    // First pass: compute the m1-period SMA of `data`.
    // Second pass: compute the m2-period SMA of that first-pass result.
    // That is exactly the standard definition of a "Triangular MA."

    let n = data.len();
    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;

    // FIRST PASS:  length-m1 SMA on `data`.
    let sma1_in = SmaInput {
        data: SmaData::Slice(data),
        params: SmaParams { period: Some(m1) },
    };
    // We know this cannot error, because `period <= n` and m1 > 0.
    let pass1 = sma(&sma1_in).unwrap();

    // SECOND PASS: length-m2 SMA on the first-pass values.
    let sma2_in = SmaInput {
        data: SmaData::Slice(&pass1.values),
        params: SmaParams { period: Some(m2) },
    };
    let pass2 = sma(&sma2_in).unwrap();

    // Copy only the valid values from pass2, respecting the warmup period
    // The batch implementation expects us to preserve the NaN values set by init_matrix_prefixes
    let warmup_end = first + period - 1;
    for i in warmup_end..n {
        out[i] = pass2.values[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn trima_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { trima_avx512_short(data, period, first, out) }
    } else {
        unsafe { trima_avx512_long(data, period, first, out) }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert_eq!(data.len(), out.len());
    let n = data.len();
    if n == 0 {
        return;
    }

    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;
    let warm = first + period - 1;
    if warm >= n {
        return;
    }

    let inv_m1 = 1.0 / (m1 as f64);
    let inv_m2 = 1.0 / (m2 as f64);

    // Vectorized initial m1-sum (unaligned loads)
    let mut sum1 = sum_u_avx2(data.as_ptr().add(first), m1);

    // Ramp: build m2 SMA1s into a small ring
    let mut ring: Vec<f64> = Vec::with_capacity(m2);
    let mut sum2 = 0.0;

    let mut t = first + m1 - 1;
    let mut p_new = data.as_ptr().add(first + m1);
    let mut p_old = data.as_ptr().add(first);

    // First SMA1
    {
        let s1 = sum1 * inv_m1;
        ring.push(s1);
        sum2 += s1;
    }

    while ring.len() < m2 {
        t += 1;
        sum1 += *p_new - *p_old;
        p_new = p_new.add(1);
        p_old = p_old.add(1);

        let s1 = sum1 * inv_m1;
        ring.push(s1);
        sum2 += s1;
    }

    *out.get_unchecked_mut(warm) = sum2 * inv_m2;

    // Rolling (sequential) core
    let mut head = 0usize;
    t += 1;
    while t < n {
        sum1 += *p_new - *p_old;
        p_new = p_new.add(1);
        p_old = p_old.add(1);

        let new_s1 = sum1 * inv_m1;
        let old_s1 = *ring.get_unchecked(head);
        sum2 += new_s1 - old_s1;
        *ring.get_unchecked_mut(head) = new_s1;

        head += 1;
        if head == m2 {
            head = 0;
        }

        *out.get_unchecked_mut(t) = sum2 * inv_m2;
        t += 1;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert_eq!(data.len(), out.len());
    let n = data.len();
    if n == 0 {
        return;
    }

    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;
    let warm = first + period - 1;
    if warm >= n {
        return;
    }

    let inv_m1 = 1.0 / (m1 as f64);
    let inv_m2 = 1.0 / (m2 as f64);

    // Vectorized initial m1-sum
    let mut sum1 = sum_u_avx512(data.as_ptr().add(first), m1);

    let mut ring: Vec<f64> = Vec::with_capacity(m2);
    let mut sum2 = 0.0;

    let mut t = first + m1 - 1;
    let mut p_new = data.as_ptr().add(first + m1);
    let mut p_old = data.as_ptr().add(first);

    let s1 = sum1 * inv_m1;
    ring.push(s1);
    sum2 += s1;

    while ring.len() < m2 {
        t += 1;
        sum1 += *p_new - *p_old;
        p_new = p_new.add(1);
        p_old = p_old.add(1);

        let s1 = sum1 * inv_m1;
        ring.push(s1);
        sum2 += s1;
    }

    *out.get_unchecked_mut(warm) = sum2 * inv_m2;

    let mut head = 0usize;
    t += 1;
    while t < n {
        sum1 += *p_new - *p_old;
        p_new = p_new.add(1);
        p_old = p_old.add(1);

        let new_s1 = sum1 * inv_m1;
        let old_s1 = *ring.get_unchecked(head);
        sum2 += new_s1 - old_s1;
        *ring.get_unchecked_mut(head) = new_s1;

        head += 1;
        if head == m2 {
            head = 0;
        }

        *out.get_unchecked_mut(t) = sum2 * inv_m2;
        t += 1;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    trima_avx512_short(data, period, first, out)
}

// ────────────────────── AVX helpers ──────────────────────
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum256d(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(lo, hi);
    let shuffled = _mm_unpackhi_pd(sum128, sum128);
    _mm_cvtsd_f64(_mm_add_sd(sum128, shuffled))
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sum_u_avx2(ptr: *const f64, len: usize) -> f64 {
    let mut acc = _mm256_setzero_pd();
    let mut p = ptr;
    let chunks = len / 4;
    for _ in 0..chunks {
        acc = _mm256_add_pd(acc, _mm256_loadu_pd(p));
        p = p.add(4);
    }
    let mut s = hsum256d(acc);
    for i in 0..(len & 3) {
        s += *p.add(i);
    }
    s
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum512d(v: __m512d) -> f64 {
    // Reduce 8 lanes → 4 → 2 → 1
    let v4 = _mm256_add_pd(_mm512_castpd512_pd256(v), _mm512_extractf64x4_pd(v, 1));
    let v2 = _mm_add_pd(_mm256_castpd256_pd128(v4), _mm256_extractf128_pd(v4, 1));
    let hi = _mm_unpackhi_pd(v2, v2);
    _mm_cvtsd_f64(_mm_add_sd(v2, hi))
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sum_u_avx512(ptr: *const f64, len: usize) -> f64 {
    let mut acc = _mm512_setzero_pd();
    let mut p = ptr;
    let chunks = len / 8;
    for _ in 0..chunks {
        acc = _mm512_add_pd(acc, _mm512_loadu_pd(p));
        p = p.add(8);
    }
    let mut s = hsum512d(acc);
    for i in 0..(len & 7) {
        s += *p.add(i);
    }
    s
}

#[inline(always)]
pub fn trima_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;
    unsafe { trima_scalar_optimized(data, period, m1, m2, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512_short(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512_long(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct TrimaStream {
    // the overall TRIMA period
    period: usize,
    // the two intermediate SMA window‐sizes
    m1: usize,
    m2: usize,

    // first SMA window
    buffer1: Vec<f64>,
    sum1: f64,
    head1: usize,
    filled1: bool,

    // second SMA window
    buffer2: Vec<f64>,
    sum2: f64,
    head2: usize,
    filled2: bool,
}

impl TrimaStream {
    pub fn try_new(params: TrimaParams) -> Result<Self, TrimaError> {
        let period = params.period.unwrap_or(30);
        if period == 0 || period <= 3 {
            return Err(TrimaError::PeriodTooSmall { period });
        }
        // compute m₁ and m₂ exactly as in the “two‐pass” formula:
        //   m₁ = (period+1)/2,    m₂ = period−m₁+1
        let m1 = (period + 1) / 2;
        let m2 = period - m1 + 1;

        Ok(Self {
            period,
            m1,
            m2,
            buffer1: vec![f64::NAN; m1],
            sum1: 0.0,
            head1: 0,
            filled1: false,
            buffer2: vec![f64::NAN; m2],
            sum2: 0.0,
            head2: 0,
            filled2: false,
        })
    }

    /// Feed a single new raw price into the TRIMA‐stream.
    /// Returns `Some(trima_value)` only once enough data has been seen for both sub‐windows;
    /// otherwise returns `None` (which the test harness will compare as NaN).
    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        // ──  STEP 1:  Update the m₁‐window (compute first‐stage SMA)  ──
        let old1 = self.buffer1[self.head1];
        self.buffer1[self.head1] = x;
        self.head1 = (self.head1 + 1) % self.m1;
        if !self.filled1 && self.head1 == 0 {
            self.filled1 = true;
        }
        // Adjust sum1, always ignoring NaNs:
        if !old1.is_nan() {
            self.sum1 -= old1;
        }
        if !x.is_nan() {
            self.sum1 += x;
        }
        // Once filled1 is true, we can compute SMA₁ = sum1 / m₁:
        let sma1 = if self.filled1 {
            Some(self.sum1 / (self.m1 as f64))
        } else {
            None
        };

        // ──  STEP 2:  Once we have an SMA₁, feed it into the m₂‐window (second pass)  ──
        if let Some(s1) = sma1 {
            let old2 = self.buffer2[self.head2];
            self.buffer2[self.head2] = s1;
            self.head2 = (self.head2 + 1) % self.m2;
            if !self.filled2 && self.head2 == 0 {
                self.filled2 = true;
            }
            if !old2.is_nan() {
                self.sum2 -= old2;
            }
            if !s1.is_nan() {
                self.sum2 += s1;
            }
            // Once filled2 == true, we can output TRIMA = sum2 / m₂
            if self.filled2 {
                return Some(self.sum2 / (self.m2 as f64));
            }
        }

        None
    }
}

#[derive(Clone, Debug)]
pub struct TrimaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TrimaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrimaBatchBuilder {
    range: TrimaBatchRange,
    kernel: Kernel,
}

impl TrimaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<TrimaBatchOutput, TrimaError> {
        trima_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TrimaBatchOutput, TrimaError> {
        TrimaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TrimaBatchOutput, TrimaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<TrimaBatchOutput, TrimaError> {
        TrimaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn trima_batch_with_kernel(
    data: &[f64],
    sweep: &TrimaBatchRange,
    k: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => other, // allow explicit per-row kernels for tests
    };

    // Map batch selector to per-row compute kernel (ALMA pattern)
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx512 | Kernel::Avx2 | Kernel::Scalar => kernel,
        _ => Kernel::Scalar,
    };

    trima_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TrimaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrimaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TrimaBatchOutput {
    pub fn row_for_params(&self, p: &TrimaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }

    pub fn values_for(&self, p: &TrimaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TrimaBatchRange) -> Vec<TrimaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TrimaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn trima_batch_slice(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    trima_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn trima_batch_par_slice(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    trima_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn trima_batch_inner(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TrimaBatchOutput, TrimaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TrimaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrimaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TrimaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // ---------- 2. allocate rows×cols buffer and stamp NaN prefixes ----------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 3. closure that writes one row in-place ----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast *just this row* to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => {
                trima_row_scalar(data, first, period, 0, core::ptr::null(), 1.0, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => trima_row_avx2(data, first, period, 0, core::ptr::null(), 1.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                trima_row_avx512(data, first, period, 0, core::ptr::null(), 1.0, out_row)
            }
            _ => trima_row_scalar(data, first, period, 0, core::ptr::null(), 1.0, out_row),
        }
    };

    // ---------- 4. run every row (parallel or serial) ----------
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

    // ---------- 5. convert Vec<MaybeUninit<f64>> → Vec<f64> ----------
    let mut buf_guard = core::mem::ManuallyDrop::new(raw);
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(TrimaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn trima_batch_inner_into(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TrimaParams>, TrimaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TrimaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrimaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TrimaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Stamp warm prefixes via helper on the destination buffer
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    let out_mu = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    // Per-row writer (cast to f64 slice only for the row)
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match kern {
            Kernel::Scalar => {
                trima_row_scalar(data, first, period, 0, core::ptr::null(), 1.0, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => trima_row_avx2(data, first, period, 0, core::ptr::null(), 1.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                trima_row_avx512(data, first, period, 0, core::ptr::null(), 1.0, out_row)
            }
            _ => trima_row_scalar(data, first, period, 0, core::ptr::null(), 1.0, out_row),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

// Python bindings
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "trima")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Triangular Moving Average (TRIMA) of the input data.
///
/// TRIMA is a double-smoothed simple moving average that places more weight
/// on the middle portion of the smoothing period.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Window size for the moving average (must be > 3).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of TRIMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period <= 3, period > data length, etc).
pub fn trima_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = TrimaParams {
        period: Some(period),
    };
    let trima_in = TrimaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| trima_with_kernel(&trima_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrimaStream")]
pub struct TrimaStreamPy {
    stream: TrimaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrimaStreamPy {
    #[new]
    fn new(period: Option<usize>) -> PyResult<Self> {
        let params = TrimaParams { period };
        let stream =
            TrimaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TrimaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated TRIMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trima_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute TRIMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' array.
pub fn trima_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let sweep = TrimaBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate NumPy output and write into it directly
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Resolve kernel like ALMA
    let kern = validate_kernel(kernel, true)?;
    let kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match kern {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx512 | Kernel::Avx2 | Kernel::Scalar => kern,
        _ => Kernel::Scalar,
    };

    let combos = py
        .allow_threads(|| trima_batch_inner_into(slice_in, &sweep, simd, true, slice_out))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap_or(30) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "trima_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, device_id=0))]
pub fn trima_cuda_batch_dev_py(
    py: Python<'_>,
    data: numpy::PyReadonlyArray1<'_, f64>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data.as_slice()?;
    let sweep = TrimaBatchRange {
        period: period_range,
    };

    let data_f32: Vec<f32> = slice_in.iter().map(|&v| v as f32).collect();

    let inner = py.allow_threads(|| {
        let cuda = CudaTrima::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.trima_batch_dev(&data_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "trima_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn trima_cuda_many_series_one_param_dev_py(
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
    let params = TrimaParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaTrima::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.trima_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

// WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use serde_wasm_bindgen;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Compute the Triangular Moving Average (TRIMA) of the input data.
///
/// # Arguments
/// * `data` - Input data array
/// * `period` - Window size for the moving average (must be > 3)
///
/// # Returns
/// Array of TRIMA values, same length as input
pub fn trima_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = TrimaParams {
        period: Some(period),
    };
    let input = TrimaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    trima_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TrimaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TrimaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrimaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = trima_batch)]
pub fn trima_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: TrimaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = TrimaBatchRange {
        period: config.period_range,
    };

    // resolve per-row kernel for batch like ALMA
    let output = trima_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = TrimaBatchJsOutput {
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
/// Compute TRIMA for multiple period values in a single pass.
///
/// # Arguments
/// * `data` - Input data array
/// * `period_start`, `period_end`, `period_step` - Period range parameters
///
/// # Returns
/// Flattened array of values (row-major order)
pub fn trima_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TrimaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    trima_batch_inner(data, &sweep, Kernel::Auto, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Get metadata about batch computation.
///
/// # Arguments
/// * Period range parameters (same as trima_batch_js)
///
/// # Returns
/// Array containing period values
pub fn trima_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TrimaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(30) as f64)
        .collect();

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trima_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trima_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trima_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to trima_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }
        let params = TrimaParams {
            period: Some(period),
        };
        if in_ptr == out_ptr {
            // compute into a temp result buffer, then copy
            let mut temp = vec![0.0; len];
            let input = TrimaInput::from_slice(data, params);
            trima_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let input = TrimaInput::from_slice(data, params);
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            trima_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trima_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to trima_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = TrimaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        trima_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For streaming patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct TrimaContext {
    period: usize,
    m1: usize,
    m2: usize,
    first: usize,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl TrimaContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For streaming patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(period: usize) -> Result<TrimaContext, JsValue> {
        if period == 0 {
            return Err(JsValue::from_str("Invalid period: 0"));
        }
        if period <= 3 {
            return Err(JsValue::from_str(&format!("Period too small: {}", period)));
        }

        let m1 = (period + 1) / 2;
        let m2 = period - m1 + 1;

        Ok(TrimaContext {
            period,
            m1,
            m2,
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
                trima_compute_into(
                    data,
                    self.period,
                    self.m1,
                    self.m2,
                    first,
                    self.kernel,
                    &mut temp,
                );

                out.copy_from_slice(&temp);
            } else {
                trima_compute_into(data, self.period, self.m1, self.m2, first, self.kernel, out);
            }

            // Ensure proper warmup period
            let warmup = first + self.period - 1;
            for i in 0..warmup {
                out[i] = f64::NAN;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.period - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_trima_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = TrimaParams { period: None };
        let input = TrimaInput::from_candles(&candles, "close", default_params);
        let output = trima_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_10 = TrimaParams { period: Some(10) };
        let input2 = TrimaInput::from_candles(&candles, "hl2", params_period_10);
        let output2 = trima_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = TrimaParams { period: Some(14) };
        let input3 = TrimaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = trima_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());

        Ok(())
    }

    fn check_trima_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;
        let params = TrimaParams { period: Some(30) };
        let input = TrimaInput::from_candles(&candles, "close", params);
        let trima_result = trima_with_kernel(&input, kernel)?;

        assert_eq!(
            trima_result.values.len(),
            close_prices.len(),
            "TRIMA output length should match input data length"
        );
        let expected_last_five_trima = [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ];
        assert!(
            trima_result.values.len() >= 5,
            "Not enough TRIMA values for the test"
        );
        let start_index = trima_result.values.len() - 5;
        let result_last_five_trima = &trima_result.values[start_index..];
        for (i, &value) in result_last_five_trima.iter().enumerate() {
            let expected_value = expected_last_five_trima[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "[{}] TRIMA value mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        let period = input.params.period.unwrap_or(14);
        for i in 0..(period - 1) {
            assert!(
                trima_result.values[i].is_nan(),
                "[{}] Expected NaN at early index {} for TRIMA, got {}",
                test_name,
                i,
                trima_result.values[i]
            );
        }
        Ok(())
    }

    fn check_trima_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TrimaInput::with_default_candles(&candles);
        match input.data {
            TrimaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TrimaData::Candles"),
        }
        let output = trima_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_trima_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TrimaParams { period: Some(0) };
        let input = TrimaInput::from_slice(&input_data, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_trima_period_too_small(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0, 40.0];
        let params = TrimaParams { period: Some(3) };
        let input = TrimaInput::from_slice(&input_data, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with period <= 3",
            test_name
        );
        Ok(())
    }

    fn check_trima_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TrimaParams { period: Some(10) };
        let input = TrimaInput::from_slice(&data_small, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_trima_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TrimaParams { period: Some(14) };
        let input = TrimaInput::from_slice(&single_point, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_trima_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = TrimaParams { period: Some(14) };
        let first_input = TrimaInput::from_candles(&candles, "close", first_params);
        let first_result = trima_with_kernel(&first_input, kernel)?;

        let second_params = TrimaParams { period: Some(10) };
        let second_input = TrimaInput::from_slice(&first_result.values, second_params);
        let second_result = trima_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(val.is_finite());
        }
        Ok(())
    }

    fn check_trima_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TrimaInput::from_candles(&candles, "close", TrimaParams { period: Some(14) });
        let res = trima_with_kernel(&input, kernel)?;
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

    fn check_trima_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;

        let input = TrimaInput::from_candles(
            &candles,
            "close",
            TrimaParams {
                period: Some(period),
            },
        );
        let batch_output = trima_with_kernel(&input, kernel)?.values;

        let mut stream = TrimaStream::try_new(TrimaParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(trima_val) => stream_values.push(trima_val),
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
                diff < 1e-8,
                "[{}] TRIMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_trima_tests {
        ($($test_fn:ident),*) => {
            paste! {
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

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_trima_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase coverage
        let test_periods = vec![4, 10, 14, 30, 50, 100];
        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for period in test_periods {
            for source in &test_sources {
                let params = TrimaParams {
                    period: Some(period),
                };
                let input = TrimaInput::from_candles(&candles, source, params);
                let output = trima_with_kernel(&input, kernel)?;

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (period={}, source={})",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (period={}, source={})",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (period={}, source={})",
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
    fn check_trima_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_trima_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::indicators::sma::{sma, SmaData, SmaInput, SmaParams};
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Test strategy: generate period first (4 is minimum valid), then data of appropriate length
        let strat = (4usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = TrimaParams {
                period: Some(period),
            };
            let input = TrimaInput::from_slice(&data, params);

            // Compute TRIMA with the specified kernel and scalar reference
            let result = trima_with_kernel(&input, kernel)?;
            let scalar_result = trima_with_kernel(&input, Kernel::Scalar)?;

            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            let warmup_end = first + period - 1;

            // Property 1: Warmup period - values before warmup_end should be NaN
            for i in 0..warmup_end.min(data.len()) {
                prop_assert!(
                    result.values[i].is_nan(),
                    "Expected NaN during warmup at index {}, got {}",
                    i,
                    result.values[i]
                );
            }

            // Property 2: Valid values after warmup
            for i in warmup_end..data.len() {
                prop_assert!(
                    result.values[i].is_finite() || data[i].is_nan(),
                    "Expected finite value after warmup at index {}, got {}",
                    i,
                    result.values[i]
                );
            }

            // Property 3: Constant input produces constant output
            if data[first..]
                .windows(2)
                .all(|w| (w[0] - w[1]).abs() < 1e-10)
                && data.len() > first
            {
                let constant_val = data[first];
                for i in warmup_end..data.len() {
                    prop_assert!(
							(result.values[i] - constant_val).abs() < 1e-9,
							"Constant input should produce constant output at index {}: expected {}, got {}",
							i,
							constant_val,
							result.values[i]
						);
                }
            }

            // Property 4: Cross-kernel consistency
            for i in 0..data.len() {
                let val = result.values[i];
                let ref_val = scalar_result.values[i];

                if val.is_nan() && ref_val.is_nan() {
                    continue;
                }

                if !val.is_finite() || !ref_val.is_finite() {
                    prop_assert_eq!(
                        val.to_bits(),
                        ref_val.to_bits(),
                        "NaN/Inf mismatch at index {}: {} vs {}",
                        i,
                        val,
                        ref_val
                    );
                } else {
                    let ulp_diff = val.to_bits().abs_diff(ref_val.to_bits());
                    prop_assert!(
                        (val - ref_val).abs() < 1e-9 || ulp_diff <= 4,
                        "Cross-kernel mismatch at index {}: {} vs {} (ULP diff: {})",
                        i,
                        val,
                        ref_val,
                        ulp_diff
                    );
                }
            }

            // Property 5: Numerical stability - no infinite values
            for (i, &val) in result.values.iter().enumerate() {
                prop_assert!(
                    val.is_nan() || val.is_finite(),
                    "Value should be finite or NaN at index {}, got {}",
                    i,
                    val
                );
            }

            // Property 6: Bounds check - output within window bounds (with tolerance for double smoothing)
            // TRIMA is double-smoothed, so it should be well within bounds
            for i in warmup_end..data.len() {
                if i >= period - 1 {
                    let start = if i >= period - 1 { i + 1 - period } else { 0 };
                    let window = &data[start..=i];
                    let min_val = window
                        .iter()
                        .filter(|x| x.is_finite())
                        .fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = window
                        .iter()
                        .filter(|x| x.is_finite())
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                    if min_val.is_finite() && max_val.is_finite() {
                        let val = result.values[i];
                        // Allow small tolerance for numerical errors
                        prop_assert!(
                            val >= min_val - 1e-6 && val <= max_val + 1e-6,
                            "TRIMA value {} at index {} outside window bounds [{}, {}]",
                            val,
                            i,
                            min_val,
                            max_val
                        );
                    }
                }
            }

            // Property 7: Period=4 edge case with actual calculation verification
            if period == 4 {
                // m1 = (4+1)/2 = 2, m2 = 4-2+1 = 3
                // Verify the actual two-pass calculation for minimum period
                let m1 = 2;
                let m2 = 3;

                // Compute the expected two-pass SMA
                let sma1_input = SmaInput {
                    data: SmaData::Slice(&data),
                    params: SmaParams { period: Some(m1) },
                };
                let pass1 = sma(&sma1_input)?;

                let sma2_input = SmaInput {
                    data: SmaData::Slice(&pass1.values),
                    params: SmaParams { period: Some(m2) },
                };
                let expected = sma(&sma2_input)?;

                // Verify TRIMA matches the two-pass calculation
                for i in warmup_end..data.len().min(warmup_end + 5) {
                    prop_assert!(
                        (result.values[i] - expected.values[i]).abs() < 1e-9,
                        "Period=4: TRIMA mismatch at index {}: got {}, expected {}",
                        i,
                        result.values[i],
                        expected.values[i]
                    );
                }
            }

            // Property 8: Two-pass SMA formula verification
            // TRIMA must equal SMA(SMA(data, m1), m2) where m1=(period+1)/2, m2=period-m1+1
            {
                let m1 = (period + 1) / 2;
                let m2 = period - m1 + 1;

                // Compute the expected two-pass SMA
                let sma1_input = SmaInput {
                    data: SmaData::Slice(&data),
                    params: SmaParams { period: Some(m1) },
                };
                let pass1 = sma(&sma1_input)?;

                let sma2_input = SmaInput {
                    data: SmaData::Slice(&pass1.values),
                    params: SmaParams { period: Some(m2) },
                };
                let expected = sma(&sma2_input)?;

                // Spot check several points to verify formula
                let check_points = vec![
                    warmup_end,
                    warmup_end + period / 2,
                    warmup_end + period,
                    data.len() - 1,
                ];

                for &idx in &check_points {
                    if idx < data.len() {
                        let trima_val = result.values[idx];
                        let expected_val = expected.values[idx];

                        if trima_val.is_finite() && expected_val.is_finite() {
                            prop_assert!(
                                (trima_val - expected_val).abs() < 1e-9,
                                "Two-pass SMA formula mismatch at index {}: TRIMA={}, Expected={}",
                                idx,
                                trima_val,
                                expected_val
                            );
                        }
                    }
                }
            }

            // Property 9: Smoothness verification - TRIMA should be smoother than single SMA
            if data.len() >= warmup_end + 20 {
                // Compute single SMA for comparison
                let sma_input = SmaInput {
                    data: SmaData::Slice(&data),
                    params: SmaParams {
                        period: Some(period),
                    },
                };
                let single_sma = sma(&sma_input)?;

                // Calculate roughness (sum of absolute differences between consecutive values)
                let trima_roughness: f64 = result.values[warmup_end..warmup_end + 20]
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .sum();

                let sma_roughness: f64 = single_sma.values[warmup_end..warmup_end + 20]
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .sum();

                // TRIMA should generally be smoother (lower roughness) than single SMA
                // Allow for some tolerance as this depends on data pattern
                if sma_roughness > 1e-10 {
                    // Only check if there's actual variation in the data
                    prop_assert!(
							trima_roughness <= sma_roughness * 1.1, // Allow 10% tolerance
							"TRIMA should be smoother than single SMA: TRIMA roughness={}, SMA roughness={}",
							trima_roughness,
							sma_roughness
						);
                }
            }

            // Property 10: Edge case - exact period length data
            if data.len() == period {
                // Should have exactly one valid output value at the last index
                prop_assert!(
                    result.values[period - 1].is_finite(),
                    "With data.len()==period, last value should be finite, got {}",
                    result.values[period - 1]
                );
                // All other values should be NaN
                for i in 0..period - 1 {
                    prop_assert!(
                        result.values[i].is_nan(),
                        "With data.len()==period, value at {} should be NaN, got {}",
                        i,
                        result.values[i]
                    );
                }
            }

            // Property 11: Monotonicity preservation
            // For strictly monotonic data, TRIMA should preserve the trend
            let is_monotonic_increasing = data[first..].windows(2).all(|w| w[1] >= w[0] - 1e-10);
            let is_monotonic_decreasing = data[first..].windows(2).all(|w| w[1] <= w[0] + 1e-10);

            if is_monotonic_increasing || is_monotonic_decreasing {
                let valid_trima = &result.values[warmup_end..];
                if valid_trima.len() >= 2 {
                    if is_monotonic_increasing {
                        for w in valid_trima.windows(2) {
                            prop_assert!(
                                w[1] >= w[0] - 1e-9,
                                "TRIMA should preserve increasing trend: {} < {}",
                                w[1],
                                w[0]
                            );
                        }
                    } else {
                        for w in valid_trima.windows(2) {
                            prop_assert!(
                                w[1] <= w[0] + 1e-9,
                                "TRIMA should preserve decreasing trend: {} > {}",
                                w[1],
                                w[0]
                            );
                        }
                    }
                }
            }

            // Property 12: Poison detection (debug mode only)
            #[cfg(debug_assertions)]
            {
                for (i, &val) in result.values.iter().enumerate() {
                    if !val.is_nan() {
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
        })?;

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_trima_tests!(check_trima_property);

    generate_all_trima_tests!(
        check_trima_partial_params,
        check_trima_accuracy,
        check_trima_default_candles,
        check_trima_zero_period,
        check_trima_period_exceeds_length,
        check_trima_period_too_small,
        check_trima_very_small_dataset,
        check_trima_reinput,
        check_trima_nan_handling,
        check_trima_streaming,
        check_trima_no_poison
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TrimaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = TrimaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // You can use expected values as appropriate for TRIMA.
        let expected = [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter range combinations
        let period_ranges = vec![
            (4, 20, 4),    // Small periods
            (20, 50, 10),  // Medium periods
            (50, 100, 25), // Large periods
            (5, 15, 1),    // Dense small range
        ];

        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for (start, end, step) in period_ranges {
            for source in &test_sources {
                let output = TrimaBatchBuilder::new()
                    .kernel(kernel)
                    .period_range(start, end, step)
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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
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
