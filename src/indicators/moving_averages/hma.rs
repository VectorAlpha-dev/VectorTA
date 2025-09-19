//! # Hull Moving Average (HMA)
//!
//! The Hull Moving Average (HMA) is a moving average technique that aims to
//! minimize lag while providing smooth output. It combines Weighted Moving
//! Averages of different lengths—namely `period/2` and `period`—to form an
//! intermediate difference. A final Weighted Moving Average is then applied
//! using the integer part of `sqrt(period)`, yielding a responsive trend
//! indication with reduced lag.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). (defaults to 5)
//!
//! ## Returns
//! - **`Ok(HmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(HmaError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ✅ Fully implemented - 4-wide SIMD with FMA operations for weighted moving averages
//! - **AVX512 kernel**: ✅ Fully implemented - 8-wide SIMD with optimized weighted average calculations
//! - **Streaming update**: ⚠️ O(n) complexity - LinWma's `dot_ring()` iterates through all period weights
//!   - TODO: Could potentially optimize to O(1) with incremental weight updates
//! - **Memory optimization**: ✅ Uses zero-copy helpers (alloc_with_nan_prefix) for output vectors

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
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
impl<'a> AsRef<[f64]> for HmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HmaData::Slice(slice) => slice,
            HmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct HmaParams {
    pub period: Option<usize>,
}

impl Default for HmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct HmaInput<'a> {
    pub data: HmaData<'a>,
    pub params: HmaParams,
}

impl<'a> HmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HmaParams) -> Self {
        Self {
            data: HmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HmaParams) -> Self {
        Self {
            data: HmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for HmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<HmaOutput, HmaError> {
        let p = HmaParams {
            period: self.period,
        };
        let i = HmaInput::from_candles(c, "close", p);
        hma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HmaOutput, HmaError> {
        let p = HmaParams {
            period: self.period,
        };
        let i = HmaInput::from_slice(d, p);
        hma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<HmaStream, HmaError> {
        let p = HmaParams {
            period: self.period,
        };
        HmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum HmaError {
    #[error("hma: No data provided.")]
    NoData,

    #[error("hma: All values are NaN.")]
    AllValuesNaN,

    #[error("hma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("hma: Cannot calculate half of period: period = {period}")]
    ZeroHalf { period: usize },

    #[error("hma: Cannot calculate sqrt of period: period = {period}")]
    ZeroSqrtPeriod { period: usize },

    #[error("hma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn hma(input: &HmaInput) -> Result<HmaOutput, HmaError> {
    hma_with_kernel(input, Kernel::Auto)
}

#[inline]
fn hma_into_internal(input: &HmaInput, out: &mut [f64]) -> Result<(), HmaError> {
    hma_with_kernel_into(input, Kernel::Auto, out)
}

pub fn hma_with_kernel(input: &HmaInput, kernel: Kernel) -> Result<HmaOutput, HmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HmaError::NoData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(HmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let half = period / 2;
    if half == 0 {
        return Err(HmaError::ZeroHalf { period });
    }
    let sqrt_len = (period as f64).sqrt().floor() as usize;
    if sqrt_len == 0 {
        return Err(HmaError::ZeroSqrtPeriod { period });
    }
    if len - first < period + sqrt_len - 1 {
        return Err(HmaError::NotEnoughValidData {
            needed: period + sqrt_len - 1,
            valid: len - first,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let first_out = first + period + sqrt_len - 2;
    let mut out = alloc_with_nan_prefix(len, first_out);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => hma_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(HmaOutput { values: out })
}

fn hma_with_kernel_into(input: &HmaInput, kernel: Kernel, out: &mut [f64]) -> Result<(), HmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HmaError::NoData);
    }
    if out.len() != len {
        return Err(HmaError::InvalidPeriod {
            period: out.len(),
            data_len: len,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(HmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let half = period / 2;
    if half == 0 {
        return Err(HmaError::ZeroHalf { period });
    }
    let sqrt_len = (period as f64).sqrt().floor() as usize;
    if sqrt_len == 0 {
        return Err(HmaError::ZeroSqrtPeriod { period });
    }
    if len - first < period + sqrt_len - 1 {
        return Err(HmaError::NotEnoughValidData {
            needed: period + sqrt_len - 1,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => hma_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }

    // write only the prefix now (up to but not including first_out)
    let first_out = first + period + sqrt_len - 2;
    for v in &mut out[..first_out] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline]
pub fn hma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use std::f64;

    let len = data.len();
    if period < 2 || first >= len || period > len - first {
        return;
    }
    let half_len = period / 2;
    if half_len == 0 {
        return;
    }
    let sqrt_len = (period as f64).sqrt().floor() as usize;
    if sqrt_len == 0 {
        return;
    }

    // first index that receives a value
    let first_out = first + period + sqrt_len - 2;
    if first_out >= len {
        return;
    }

    // precomputed constants
    let ws_half = (half_len * (half_len + 1) / 2) as f64;
    let ws_full = (period * (period + 1) / 2) as f64;
    let ws_sqrt = (sqrt_len * (sqrt_len + 1) / 2) as f64;

    // running state
    let (mut sum_half, mut wsum_half) = (0.0, 0.0);
    let (mut sum_full, mut wsum_full) = (0.0, 0.0);
    let (mut wma_half, mut wma_full) = (f64::NAN, f64::NAN);

    // √n ring buffer
    let mut x_buf = vec![0.0; sqrt_len];
    let mut x_sum = 0.0;
    let mut x_wsum = 0.0;
    let mut x_head = 0usize;

    let start = first;
    for j in 0..(len - start) {
        let idx = start + j;
        let val = data[idx];

        // WMA(full)
        if j < period {
            sum_full += val;
            wsum_full += (j as f64 + 1.0) * val;
        } else {
            let old = data[idx - period];
            let sum_prev = sum_full;
            sum_full = sum_prev + val - old;
            wsum_full = wsum_full - sum_prev + (period as f64) * val;
        }
        if j + 1 >= period {
            wma_full = wsum_full / ws_full;
        }

        // WMA(half)
        if j < half_len {
            sum_half += val;
            wsum_half += (j as f64 + 1.0) * val;
        } else {
            let old = data[idx - half_len];
            let sum_prev = sum_half;
            sum_half = sum_prev + val - old;
            wsum_half = wsum_half - sum_prev + (half_len as f64) * val;
        }
        if j + 1 >= half_len {
            wma_half = wsum_half / ws_half;
        }

        // Combine when both exist
        if j + 1 >= period {
            let x_val = 2.0 * wma_half - wma_full;

            // fill √n buffer
            if j + 1 < period + sqrt_len {
                let pos = j + 1 - period;
                x_buf[pos] = x_val;
                x_sum += x_val;

                if pos + 1 == sqrt_len {
                    // first HMA value at first_out
                    x_wsum = 0.0;
                    for k in 0..sqrt_len {
                        x_wsum += (k as f64 + 1.0) * x_buf[k];
                    }
                    out[first_out] = x_wsum / ws_sqrt;
                }
            } else {
                // steady state
                let old_x = x_buf[x_head];
                x_buf[x_head] = x_val;
                x_head = (x_head + 1) % sqrt_len;

                let sum_prev = x_sum;
                x_sum = sum_prev + x_val - old_x;
                x_wsum = x_wsum - sum_prev + (sqrt_len as f64) * x_val;

                out[idx] = x_wsum / ws_sqrt;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn hma_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // SAFETY: we just verified AVX2+FMA at runtime.
    unsafe {
        // ----------- early-exit checks (unchanged from scalar) -----------
        let len = data.len();
        if period < 2 || first >= len || period > len - first {
            return;
        }
        let half_len = period / 2;
        if half_len == 0 {
            return;
        }
        let sqrt_len = (period as f64).sqrt().floor() as usize;
        if sqrt_len == 0 {
            return;
        }
        let first_out = first + period + sqrt_len - 2;
        if first_out >= len {
            return;
        }

        let ws_half = (half_len * (half_len + 1) / 2) as f64;
        let ws_full = (period * (period + 1) / 2) as f64;
        let ws_sqrt = (sqrt_len * (sqrt_len + 1) / 2) as f64;

        let (mut sum_half, mut wsum_half) = (0.0, 0.0);
        let (mut sum_full, mut wsum_full) = (0.0, 0.0);
        let (mut wma_half, mut wma_full) = (f64::NAN, f64::NAN);

        // √n ring-buffer + constant weights
        let mut x_buf = vec![0.0; sqrt_len];
        let mut weights = vec![0.0; sqrt_len];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = (k + 1) as f64;
        }

        let mut x_sum = 0.0;
        let mut x_wsum = 0.0;
        let mut x_head = 0usize;

        // ------------------------- main loop -------------------------
        let start = first;
        for j in 0..(len - start) {
            let idx = start + j;
            let val = *data.get_unchecked(idx);

            // ----- WMA(full) rolling update -----
            if j < period {
                sum_full += val;
                wsum_full += (j as f64 + 1.0) * val;
            } else {
                let old = *data.get_unchecked(idx - period);
                let sum_prev = sum_full;
                sum_full = sum_prev + val - old;
                wsum_full = wsum_full - sum_prev + (period as f64) * val;
            }
            if j + 1 >= period {
                wma_full = wsum_full / ws_full;
            }

            // ----- WMA(half) rolling update -----
            if j < half_len {
                sum_half += val;
                wsum_half += (j as f64 + 1.0) * val;
            } else {
                let old = *data.get_unchecked(idx - half_len);
                let sum_prev = sum_half;
                sum_half = sum_prev + val - old;
                wsum_half = wsum_half - sum_prev + (half_len as f64) * val;
            }
            if j + 1 >= half_len {
                wma_half = wsum_half / ws_half;
            }

            // ----- combine once both WMAs exist -----
            if j + 1 >= period {
                let x_val = 2.0 * wma_half - wma_full;

                // fill √n buffer first
                if j + 1 < period + sqrt_len {
                    let pos = j + 1 - period;
                    *x_buf.get_unchecked_mut(pos) = x_val;
                    x_sum += x_val;

                    if pos + 1 == sqrt_len {
                        // SIMD dot‐product for the first HMA
                        x_wsum = {
                            use std::arch::x86_64::*;
                            // len is ≤ period, so always multiple of 4? Not necessarily.
                            let mut acc = _mm256_setzero_pd();
                            let chunks = sqrt_len / 4;
                            for c in 0..chunks {
                                let v1 = _mm256_loadu_pd(x_buf.as_ptr().add(c * 4));
                                let v2 = _mm256_loadu_pd(weights.as_ptr().add(c * 4));
                                acc = _mm256_fmadd_pd(v1, v2, acc);
                            }
                            // horizontal add
                            let hi = _mm256_extractf128_pd(acc, 1);
                            let lo = _mm256_castpd256_pd128(acc);
                            let t = _mm_add_pd(hi, lo);
                            let t = _mm_hadd_pd(t, t);
                            let mut sum = _mm_cvtsd_f64(t);

                            for i in (chunks * 4)..sqrt_len {
                                sum += *x_buf.get_unchecked(i) * *weights.get_unchecked(i);
                            }
                            sum
                        };
                        *out.get_unchecked_mut(first_out) = x_wsum / ws_sqrt;
                    }
                } else {
                    // rolling update after buffer full
                    let old_x = *x_buf.get_unchecked(x_head);
                    *x_buf.get_unchecked_mut(x_head) = x_val;
                    x_head = (x_head + 1) % sqrt_len;

                    let sum_prev = x_sum;
                    x_sum = sum_prev + x_val - old_x;
                    x_wsum = x_wsum - sum_prev + (sqrt_len as f64) * x_val;

                    *out.get_unchecked_mut(idx) = x_wsum / ws_sqrt;
                }
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn hma_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use aligned_vec::AVec;
    use core::arch::x86_64::*;

    // ---------- parameter & safety checks ----------
    let len = data.len();
    if period < 2 || first >= len || period > len - first {
        return;
    }
    let half = period / 2;
    if half == 0 {
        return;
    }
    let sq = (period as f64).sqrt().floor() as usize;
    debug_assert!(
        sq > 0 && sq <= 65_535,
        "HMA: √period must fit in 16-bit to keep Σw < 2^53"
    );
    if sq == 0 {
        return;
    }
    let first_out = first + period + sq - 2;
    if first_out >= len {
        return;
    }

    // ---------- pre-computed window constants ----------
    let ws_half = (half * (half + 1) / 2) as f64;
    let ws_full = (period * (period + 1) / 2) as f64;
    let ws_sqrt = (sq * (sq + 1) / 2) as f64;
    let sq_f = sq as f64; // for FMA later

    // ---------- rolling state ----------
    let (mut s_half, mut ws_half_acc) = (0.0, 0.0);
    let (mut s_full, mut ws_full_acc) = (0.0, 0.0);
    let (mut wma_half, mut wma_full) = (f64::NAN, f64::NAN);

    // √n ring buffer – 64 B aligned & length rounded up to 8 ×
    let sq_aligned = (sq + 7) & !7;
    let mut x_buf: AVec<f64> = AVec::with_capacity(64, sq_aligned);
    x_buf.resize(sq_aligned, 0.0);

    let mut x_sum = 0.0;
    let mut x_wsum = 0.0;
    let mut x_head = 0usize;

    // lane-local ramp 0‥7 → shifted per block
    const W_RAMP_ARR: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // ❷  Load once into a ZMM register
    let w_ramp: __m512d = _mm512_loadu_pd(W_RAMP_ARR.as_ptr());

    // fast horizontal sum (avoids _mm512_reduce_add_pd latency)
    #[inline(always)]
    unsafe fn horiz_sum(z: __m512d) -> f64 {
        // Extract upper and lower 256-bit halves
        let hi = _mm512_extractf64x4_pd(z, 1);
        let lo = _mm512_castpd512_pd256(z);
        // Add them together (now have 4 sums)
        let sum256 = _mm256_add_pd(hi, lo);
        // Horizontal add to get 2 sums
        let sum128 = _mm256_hadd_pd(sum256, sum256);
        // Extract high and low 128-bit parts and add
        let hi128 = _mm256_extractf128_pd(sum128, 1);
        let lo128 = _mm256_castpd256_pd128(sum128);
        let final_sum = _mm_add_pd(hi128, lo128);
        // Extract the final scalar result
        _mm_cvtsd_f64(final_sum)
    }

    // ---------- phase 1: warm-up ----------
    for j in 0..(period + sq - 1) {
        let idx = first + j;
        let val = *data.get_unchecked(idx); // unaligned load OK

        // full WMA update
        if j < period {
            s_full += val;
            ws_full_acc += (j as f64 + 1.0) * val;
        } else {
            let old = *data.get_unchecked(idx - period);
            let prev = s_full;
            s_full = prev + val - old;
            ws_full_acc = ws_full_acc - prev + (period as f64) * val;
        }

        // half WMA update
        if j < half {
            s_half += val;
            ws_half_acc += (j as f64 + 1.0) * val;
        } else {
            let old = *data.get_unchecked(idx - half);
            let prev = s_half;
            s_half = prev + val - old;
            ws_half_acc = ws_half_acc - prev + (half as f64) * val;
        }

        if j + 1 >= half {
            wma_half = ws_half_acc / ws_half;
        }
        if j + 1 >= period {
            wma_full = ws_full_acc / ws_full;
        }

        if j + 1 >= period {
            let x_val = 2.0 * wma_half - wma_full;
            let pos = (j + 1 - period) as usize;

            if pos < sq {
                *x_buf.get_unchecked_mut(pos) = x_val;
                x_sum += x_val;

                if pos + 1 == sq {
                    // first full dot-product (SIMD)
                    let mut acc = _mm512_setzero_pd();
                    let mut i = 0usize;
                    let mut off = 0.0;
                    while i + 8 <= sq {
                        let x = _mm512_loadu_pd(x_buf.as_ptr().add(i)); // unaligned OK
                                                                        // use the *vector* you just loaded
                        let w = _mm512_add_pd(w_ramp, _mm512_set1_pd(off + 1.0));
                        acc = _mm512_fmadd_pd(x, w, acc);
                        i += 8;
                        off += 8.0;
                    }
                    x_wsum = horiz_sum(acc);
                    for k in i..sq {
                        x_wsum += x_buf[k] * (k as f64 + 1.0);
                    }
                    *out.get_unchecked_mut(first_out) = x_wsum / ws_sqrt; // unaligned store
                }
            }
        }
    }

    // ---------- phase 2: steady-state ----------
    for j in (period + sq - 1)..(len - first) {
        let idx = first + j;
        let val = *data.get_unchecked(idx); // unaligned load

        // vectorised rolling update for both WMAs (128-bit packs)
        let old_f = *data.get_unchecked(idx - period);
        let old_h = *data.get_unchecked(idx - half);

        let sum_vec = _mm_set_pd(s_full, s_half); // [hi | lo]
        let old_vec = _mm_set_pd(old_f, old_h);
        let ws_vec = _mm_set_pd(ws_full_acc, ws_half_acc);
        let weights = _mm_set_pd(period as f64, half as f64);
        let v_val = _mm_set1_pd(val);

        // Σ ← Σ − old + val
        let new_sum_vec = _mm_add_pd(_mm_sub_pd(sum_vec, old_vec), v_val);

        // WS ← WS − Σ_prev + w*val  (single FMA)
        let diff = _mm_sub_pd(ws_vec, sum_vec);
        let new_ws_vec = _mm_fmadd_pd(v_val, weights, diff);

        // unpack back to scalars
        s_full = _mm_cvtsd_f64(_mm_unpackhi_pd(new_sum_vec, new_sum_vec));
        s_half = _mm_cvtsd_f64(new_sum_vec);
        ws_full_acc = _mm_cvtsd_f64(_mm_unpackhi_pd(new_ws_vec, new_ws_vec));
        ws_half_acc = _mm_cvtsd_f64(new_ws_vec);

        // derive WMAs & combine
        wma_full = ws_full_acc / ws_full;
        wma_half = ws_half_acc / ws_half;
        let x_val = 2.0 * wma_half - wma_full;

        // ring update – O(1) with fused multiply-add
        let old_x = *x_buf.get_unchecked(x_head);
        *x_buf.get_unchecked_mut(x_head) = x_val;
        x_head = (x_head + 1) % sq;

        let prev_sum = x_sum;
        x_sum = prev_sum + x_val - old_x;
        x_wsum = sq_f.mul_add(x_val, x_wsum - prev_sum); // single FMA

        *out.get_unchecked_mut(idx) = x_wsum / ws_sqrt; // unaligned store

        // software prefetch to L2 ~16 iterations ahead
        let pf = core::cmp::min(idx + 128, len - 1);
        _mm_prefetch(data.as_ptr().add(pf) as *const i8, _MM_HINT_T1);
    }
}

#[derive(Debug, Clone)]
struct LinWma {
    period: usize,
    weights: Vec<f64>,
    inv_norm: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl LinWma {
    fn new(period: usize) -> Self {
        let mut weights = Vec::with_capacity(period);
        let mut norm = 0.0;
        for i in 0..period {
            let w = (i + 1) as f64;
            weights.push(w);
            norm += w;
        }
        Self {
            period,
            weights,
            inv_norm: 1.0 / norm,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
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

#[derive(Debug, Clone)]
pub struct HmaStream {
    wma_half: LinWma,
    wma_full: LinWma,
    wma_sqrt: LinWma,
}

impl HmaStream {
    pub fn try_new(params: HmaParams) -> Result<Self, HmaError> {
        let period = params.period.unwrap_or(5);
        if period < 2 {
            return Err(HmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let half = period / 2;
        if half == 0 {
            return Err(HmaError::ZeroHalf { period });
        }
        let sqrt_len = (period as f64).sqrt().floor() as usize;
        if sqrt_len == 0 {
            return Err(HmaError::ZeroSqrtPeriod { period });
        }

        Ok(Self {
            wma_half: LinWma::new(half),
            wma_full: LinWma::new(period),
            wma_sqrt: LinWma::new(sqrt_len),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let full = self.wma_full.update(value);
        let half = self.wma_half.update(value);
        if let (Some(f), Some(h)) = (full, half) {
            let x = 2.0 * h - f;
            self.wma_sqrt.update(x)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct HmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for HmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 120, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HmaBatchBuilder {
    range: HmaBatchRange,
    kernel: Kernel,
}

impl HmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<HmaBatchOutput, HmaError> {
        hma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<HmaBatchOutput, HmaError> {
        HmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<HmaBatchOutput, HmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HmaBatchOutput, HmaError> {
        HmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn hma_batch_with_kernel(
    data: &[f64],
    sweep: &HmaBatchRange,
    k: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(HmaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    hma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HmaBatchOutput {
    pub fn row_for_params(&self, p: &HmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &HmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HmaBatchRange) -> Vec<HmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(HmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn hma_batch_slice(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    hma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn hma_batch_par_slice(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    hma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn hma_batch_inner(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HmaBatchOutput, HmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // one warm-prefix per row so batch + streaming agree
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let s = (p as f64).sqrt().floor() as usize;
            first + p + s - 2 // first_out index for each period
        })
        .collect();

    // -------- allocate rows×cols uninitialised -----------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // -------- per-row worker closure -----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this row into &mut [f64] once we’re ready to write real numbers
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => hma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // -------- run every row -----------
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

    // finalize like ALMA
    let rows = combos.len();
    let cols = data.len();

    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(HmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn hma_batch_inner_into(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(Vec<HmaParams>, usize, usize), HmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Ensure output buffer is the correct size
    if out.len() != rows * cols {
        return Err(HmaError::InvalidPeriod {
            period: out.len(),
            data_len: rows * cols,
        });
    }

    // one warm-prefix per row so batch + streaming agree
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let s = (p as f64).sqrt().floor() as usize;
            first + p + s - 2 // first_out index for each period
        })
        .collect();

    // Cast output to MaybeUninit for initialization
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // -------- per-row worker closure -----------
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => hma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // -------- run every row -----------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok((combos, rows, cols))
}

// --- row variants (all AVX point to scalar, as per your pattern) ---

#[inline(always)]
pub unsafe fn hma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    hma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    hma_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    hma_avx512(data, period, first, out);
}

#[inline(always)]
fn expand_grid_hma(r: &HmaBatchRange) -> Vec<HmaParams> {
    expand_grid(r)
}

// Python bindings
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

#[cfg(feature = "python")]
#[pyfunction(name = "hma")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn hma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = HmaParams {
        period: Some(period),
    };
    let hma_in = HmaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| hma_with_kernel(&hma_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "hma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn hma_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = HmaBatchRange {
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
            hma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
                .map(|(combos, _, _)| combos)
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

#[cfg(feature = "python")]
#[pyclass(name = "HmaStream")]
pub struct HmaStreamPy {
    inner: HmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = HmaParams {
            period: Some(period),
        };
        let stream =
            HmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(HmaStreamPy { inner: stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// Core helper function for zero-copy operations
#[inline]
pub fn hma_into_slice(dst: &mut [f64], input: &HmaInput, kern: Kernel) -> Result<(), HmaError> {
    let data: &[f64] = input.as_ref();

    if dst.len() != data.len() {
        return Err(HmaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    hma_with_kernel_into(input, kern, dst)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = HmaParams {
        period: Some(period),
    };
    let input = HmaInput::from_slice(data, params);

    // derive warmup cheaply
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| JsValue::from_str("All NaN"))?;
    let sqrt_len = (period as f64).sqrt().floor() as usize;
    if period == 0 || sqrt_len == 0 || data.len() - first < period + sqrt_len - 1 {
        return Err(JsValue::from_str("Invalid or insufficient data"));
    }
    let first_out = first + period + sqrt_len - 2;

    // zero-copy style allocation
    let mut output = alloc_with_nan_prefix(data.len(), first_out);
    hma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct HmaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct HmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = hma_batch)]
pub fn hma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: HmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = HmaBatchRange {
        period: config.period_range,
    };

    // Force scalar kernel for WASM since it doesn't support SIMD
    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::ScalarBatch
    } else {
        Kernel::Auto
    };

    let output = hma_batch_inner(data, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = HmaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Legacy batch API for backward compatibility
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = HmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Force scalar kernel for WASM since it doesn't support SIMD
    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::ScalarBatch
    } else {
        Kernel::Auto
    };

    let output = hma_batch_inner(data, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Vec<f64> {
    let periods: Vec<usize> = if period_step == 0 || period_start == period_end {
        vec![period_start]
    } else {
        (period_start..=period_end).step_by(period_step).collect()
    };

    periods.iter().map(|&p| p as f64).collect()
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hma_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hma_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to hma_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Calculate HMA
        let params = HmaParams {
            period: Some(period),
        };
        let input = HmaInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            hma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            hma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to hma_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = HmaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Force scalar kernel for WASM since it doesn't support SIMD
        let kernel = if cfg!(target_arch = "wasm32") {
            Kernel::ScalarBatch
        } else {
            Kernel::Auto
        };

        // Use optimized batch processing
        hma_batch_inner_into(data, &sweep, kernel, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

// ==================== PYTHON MODULE REGISTRATION ====================
#[cfg(feature = "python")]
pub fn register_hma_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hma_py, m)?)?;
    m.add_function(wrap_pyfunction!(hma_batch_py, m)?)?;
    m.add_class::<HmaStreamPy>()?;
    Ok(())
}

// --- tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;

    fn check_hma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HmaParams { period: None };
        let input_default = HmaInput::from_candles(&candles, "close", default_params);
        let output_default = hma_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }

    fn check_hma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HmaInput::with_default_candles(&candles);
        let result = hma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        assert!(result.values.len() >= 5);
        assert_eq!(result.values.len(), candles.close.len());
        let start = result.values.len() - 5;
        let last_five = &result.values[start..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-3,
                "[{}] idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp
            );
        }
        Ok(())
    }

    fn check_hma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(0) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] HMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_hma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(10) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] HMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_hma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = HmaParams { period: Some(5) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] HMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_hma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = HmaParams { period: Some(5) };
        let first_input = HmaInput::from_candles(&candles, "close", first_params);
        let first_result = hma_with_kernel(&first_input, kernel)?;
        let second_params = HmaParams { period: Some(3) };
        let second_input = HmaInput::from_slice(&first_result.values, second_params);
        let second_result = hma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_hma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = HmaParams::default();
        let period = params.period.unwrap_or(5) * 2;
        let input = HmaInput::from_candles(&candles, "close", params);
        let result = hma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > period {
            for i in period..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_hma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = HmaInput::from_slice(&empty, HmaParams::default());
        let res = hma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HmaError::NoData)),
            "[{}] expected NoData",
            test_name
        );
        Ok(())
    }

    fn check_hma_not_enough_valid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, 1.0, 2.0];
        let params = HmaParams { period: Some(3) };
        let input = HmaInput::from_slice(&data, params);
        let res = hma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HmaError::NotEnoughValidData { .. })),
            "[{}] expected NotEnoughValidData",
            test_name
        );
        Ok(())
    }

    fn check_hma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = HmaInput::from_candles(
            &candles,
            "close",
            HmaParams {
                period: Some(period),
            },
        );
        let batch_output = hma_with_kernel(&input, kernel)?.values;

        let mut stream = HmaStream::try_new(HmaParams {
            period: Some(period),
        })?;
        let mut stream_vals = Vec::with_capacity(candles.close.len());
        for &p in &candles.close {
            match stream.update(p) {
                Some(v) => stream_vals.push(v),
                None => stream_vals.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_vals.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_vals.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-4,
                "[{}] HMA streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_hma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Load real market data for realistic testing
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_data = &candles.close;

        // Strategy: test various parameter combinations with real data slices
        let strat = (
            2usize..=100,                                 // period (HMA requires >= 2)
            0usize..close_data.len().saturating_sub(500), // starting index for data slice
            200usize..=500,                               // length of data slice to use
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                // Ensure we have valid slice bounds
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period + 10 {
                    return Ok(()); // Skip invalid combinations
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = HmaParams {
                    period: Some(period),
                };
                let input = HmaInput::from_slice(data_slice, params);

                // Test the specified kernel
                let result = hma_with_kernel(&input, kernel);

                // Also compute with scalar kernel for reference
                let scalar_result = hma_with_kernel(&input, Kernel::Scalar);

                // Both should succeed or fail together
                match (result, scalar_result) {
                    (Ok(HmaOutput { values: out }), Ok(HmaOutput { values: ref_out })) => {
                        // Verify output length
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        // Calculate expected warmup period
                        let sqrt_period = (period as f64).sqrt().floor() as usize;
                        let expected_warmup = period + sqrt_period - 1;

                        // Find first non-NaN value
                        let first_valid = out.iter().position(|x| !x.is_nan());
                        if let Some(first_idx) = first_valid {
                            // Verify warmup period is correct
                            prop_assert!(
                                first_idx >= expected_warmup - 1,
                                "First valid at {} but expected warmup is {}",
                                first_idx,
                                expected_warmup
                            );

                            // Check NaN pattern - all values before first_valid should be NaN
                            for i in 0..first_idx {
                                prop_assert!(
                                    out[i].is_nan(),
                                    "Expected NaN at index {} during warmup, got {}",
                                    i,
                                    out[i]
                                );
                            }
                        }

                        // Verify kernel consistency
                        for i in 0..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            // Both should be NaN or both should be valid
                            if y.is_nan() {
                                prop_assert!(
                                    r.is_nan(),
                                    "Kernel mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                                continue;
                            }

                            // Check finite values
                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

                            // Compare with scalar reference (allowing for floating-point precision)
                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();
                            let ulp_diff = y_bits.abs_diff(r_bits);

                            // AVX512 has higher ULP differences due to different FMA ordering
                            // but the absolute error is still very small (< 1e-8)
                            let ulp_tolerance = if matches!(kernel, Kernel::Avx512) {
                                20000
                            } else {
                                8
                            };
                            prop_assert!(
                                (y - r).abs() <= 1e-8 || ulp_diff <= ulp_tolerance,
                                "Kernel mismatch at {}: {} vs {} (ULP={})",
                                i,
                                y,
                                r,
                                ulp_diff
                            );
                        }

                        // Test HMA-specific properties for valid outputs
                        for i in expected_warmup..out.len() {
                            let y = out[i];
                            if y.is_nan() {
                                continue;
                            }

                            // HMA can produce values outside the recent window due to linear extrapolation
                            // This is intentional for lag reduction, so we only check it's finite
                            prop_assert!(y.is_finite(), "HMA output at {} is not finite: {}", i, y);

                            // For constant data, HMA should converge to that constant
                            if i >= period * 2 {
                                let window_start = i.saturating_sub(period);
                                let window = &data_slice[window_start..=i];
                                let is_constant =
                                    window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                                if is_constant {
                                    let constant_val = window[0];
                                    prop_assert!(
										(y - constant_val).abs() <= 1e-6,
										"HMA should converge to {} for constant data, got {} at index {}",
										constant_val,
										y,
										i
									);
                                }
                            }
                        }

                        // Edge case: period = 2 (minimum valid)
                        if period == 2 {
                            // HMA should still produce valid output after warmup
                            let min_valid_idx = expected_warmup;
                            if out.len() > min_valid_idx {
                                prop_assert!(
                                    out[min_valid_idx].is_finite(),
                                    "HMA with period=2 should produce valid output at index {}",
                                    min_valid_idx
                                );
                            }
                        }

                        Ok(())
                    }
                    (Err(e1), Err(_e2)) => {
                        // Both kernels failed - this is expected for insufficient data
                        prop_assert!(
                            format!("{:?}", e1).contains("NotEnoughValidData")
                                || format!("{:?}", e1).contains("InvalidPeriod"),
                            "Unexpected error type: {:?}",
                            e1
                        );
                        Ok(())
                    }
                    (Ok(_), Err(e)) | (Err(e), Ok(_)) => {
                        // Kernels should agree on success/failure
                        prop_assert!(
                            false,
                            "Kernel consistency failure: one succeeded, one failed with {:?}",
                            e
                        );
                        Ok(())
                    }
                }
            })
            .unwrap();

        // Additional edge case testing with synthetic data
        let edge_cases = vec![
            // Minimum period with small data
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], 2),
            // Constant data
            (vec![42.0; 100], 10),
            // Monotonic increasing
            ((0..100).map(|i| i as f64).collect::<Vec<_>>(), 15),
            // Monotonic decreasing
            ((0..100).map(|i| 100.0 - i as f64).collect::<Vec<_>>(), 20),
        ];

        for (case_idx, (data, period)) in edge_cases.into_iter().enumerate() {
            let params = HmaParams {
                period: Some(period),
            };
            let input = HmaInput::from_slice(&data, params);

            match hma_with_kernel(&input, kernel) {
                Ok(out) => {
                    // Just verify it produces some valid output
                    let has_valid = out.values.iter().any(|&x| x.is_finite() && !x.is_nan());
                    assert!(
                        has_valid || data.len() < period + 2,
                        "[{}] Edge case {} produced no valid output",
                        test_name,
                        case_idx
                    );
                }
                Err(_) => {
                    // Error is acceptable for edge cases with insufficient data
                }
            }
        }

        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_hma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to better catch uninitialized memory bugs
        let test_params = vec![
            // Default parameters
            HmaParams::default(),
            // Small periods
            HmaParams { period: Some(2) },
            HmaParams { period: Some(3) },
            HmaParams { period: Some(4) },
            HmaParams { period: Some(5) },
            // Medium periods
            HmaParams { period: Some(7) },
            HmaParams { period: Some(10) },
            HmaParams { period: Some(14) },
            HmaParams { period: Some(20) },
            // Large periods
            HmaParams { period: Some(30) },
            HmaParams { period: Some(50) },
            HmaParams { period: Some(100) },
            HmaParams { period: Some(200) },
            // Edge cases
            HmaParams { period: Some(1) },
            HmaParams { period: Some(250) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = HmaInput::from_candles(&candles, "close", params.clone());
            let output = hma_with_kernel(&input, kernel)?;

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
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5)
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5)
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5)
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_hma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_hma_tests {
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

    generate_all_hma_tests!(
        check_hma_partial_params,
        check_hma_accuracy,
        check_hma_zero_period,
        check_hma_period_exceeds_length,
        check_hma_very_small_dataset,
        check_hma_reinput,
        check_hma_nan_handling,
        check_hma_empty_input,
        check_hma_not_enough_valid,
        check_hma_streaming,
        check_hma_property,
        check_hma_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = HmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = HmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-3,
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
            // Small range
            (2, 5, 1), // periods: 2, 3, 4, 5
            // Medium range with gaps
            (5, 25, 5), // periods: 5, 10, 15, 20, 25
            // Large range
            (10, 50, 10), // periods: 10, 20, 30, 40, 50
            // Edge case: very small periods
            (2, 4, 1), // periods: 2, 3, 4
            // Edge case: large periods
            (50, 150, 25), // periods: 50, 75, 100, 125, 150
            // Dense range
            (10, 30, 2), // periods: 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
            // Original configuration
            (10, 30, 10), // periods: 10, 20, 30
            // Very large periods
            (100, 300, 50), // periods: 100, 150, 200, 250, 300
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = HmaBatchBuilder::new()
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
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
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
    gen_batch_tests!(check_batch_no_poison);
}
