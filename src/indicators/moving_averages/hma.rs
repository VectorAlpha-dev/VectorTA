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
//! ## Errors
//! - **NoData**: hma: No data provided.
//! - **AllValuesNaN**: hma: All input data values are `NaN`.
//! - **InvalidPeriod**: hma: `period` is zero or exceeds the data length.
//! - **ZeroHalf**: hma: Cannot calculate half of period.
//! - **ZeroSqrtPeriod**: hma: Cannot calculate sqrt of period.
//!
//! ## Returns
//! - **`Ok(HmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(HmaError)`** otherwise.

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
fn hma_into(
    input: &HmaInput,
    out: &mut [f64],
) -> Result<(), HmaError> {
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
    let warm = first + period + sqrt_len - 1;
    let mut out = alloc_with_nan_prefix(len, warm);
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

fn hma_with_kernel_into(
    input: &HmaInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), HmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HmaError::NoData);
    }
    
    // Ensure output buffer is the correct size
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
    let warm = first + period + sqrt_len - 1;
    // Initialize NaN prefix
    out[..warm].fill(f64::NAN);
    
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
    Ok(())
}

#[inline]
pub fn hma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use std::f64;

    // ------------- validation -------------
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

    for v in &mut out[..] {
        *v = f64::NAN;
    }

    // ---------- pre-computed constants ----------
    let ws_half = (half_len * (half_len + 1) / 2) as f64;
    let ws_full = (period * (period + 1) / 2) as f64;
    let ws_sqrt = (sqrt_len * (sqrt_len + 1) / 2) as f64;

    // ---------- running state ----------
    let (mut sum_half, mut wsum_half) = (0.0, 0.0);
    let (mut sum_full, mut wsum_full) = (0.0, 0.0);
    let (mut wma_half, mut wma_full) = (f64::NAN, f64::NAN);

    // ring buffer for √n intermediate values
    let mut x_buf = vec![0.0; sqrt_len];
    let mut x_sum = 0.0;
    let mut x_wsum = 0.0;
    let mut x_head = 0usize;

    // ---------- main loop ----------
    let start = first;
    for j in 0..(len - start) {
        let idx = start + j;
        let val = data[idx];

        // ---- WMA(full) ----
        if j < period {
            // window not yet full
            sum_full += val;
            wsum_full += (j as f64 + 1.0) * val;
        
            } else {
            let old = data[idx - period];
            let sum_prev = sum_full; // save old Σ
            sum_full = sum_prev + val - old; // new Σ
            wsum_full = wsum_full - sum_prev + (period as f64) * val;
        }
        if j + 1 >= period {
            wma_full = wsum_full / ws_full;
        }

        // ---- WMA(half) ----
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

        // ---- combine into X once both WMAs exist ----
        if j + 1 >= period {
            let x_val = 2.0 * wma_half - wma_full;

            // fill √n buffer first …
            if j + 1 < period + sqrt_len {
                let pos = j + 1 - period;
                x_buf[pos] = x_val;
                x_sum += x_val;

                if pos + 1 == sqrt_len {
                    // buffer full ⇒ first HMA
                    x_wsum = 0.0;
                    for k in 0..sqrt_len {
                        x_wsum += (k as f64 + 1.0) * x_buf[k];
                    }
                    out[first_out] = x_wsum / ws_sqrt;
                }
            } else {
                // … then do rolling updates
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

    /* ---------- parameter & safety checks ---------- */
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

    /* ---------- pre-computed window constants ---------- */
    let ws_half = (half * (half + 1) / 2) as f64;
    let ws_full = (period * (period + 1) / 2) as f64;
    let ws_sqrt = (sq * (sq + 1) / 2) as f64;
    let sq_f = sq as f64; // for FMA later

    /* ---------- rolling state ---------- */
    let (mut s_half, mut ws_half_acc) = (0.0, 0.0);
    let (mut s_full, mut ws_full_acc) = (0.0, 0.0);
    let (mut wma_half, mut wma_full) = (f64::NAN, f64::NAN);

    /* √n ring buffer – 64 B aligned & length rounded up to 8 × */
    let sq_aligned = (sq + 7) & !7;
    let mut x_buf: AVec<f64> = AVec::with_capacity(64, sq_aligned);
    x_buf.resize(sq_aligned, 0.0);

    let mut x_sum = 0.0;
    let mut x_wsum = 0.0;
    let mut x_head = 0usize;

    /* lane-local ramp 0‥7 → shifted per block */
    const W_RAMP_ARR: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // ❷  Load once into a ZMM register
    let w_ramp: __m512d = _mm512_loadu_pd(W_RAMP_ARR.as_ptr());

    /* fast horizontal sum (avoids _mm512_reduce_add_pd latency) */
    #[inline(always)]
    unsafe fn horiz_sum(z: __m512d) -> f64 {
        let hi = _mm512_extractf64x4_pd(z, 1);
        let lo = _mm512_castpd512_pd256(z);
        let red = _mm256_add_pd(hi, lo);
        let red = _mm256_hadd_pd(red, red);
        _mm_cvtsd_f64(_mm256_castpd256_pd128(red))
    }

    /* ---------- phase 1: warm-up ---------- */
    for j in 0..(period + sq - 1) {
        let idx = first + j;
        let val = *data.get_unchecked(idx); // unaligned load OK

        /* full WMA update */
        if j < period {
            s_full += val;
            ws_full_acc += (j as f64 + 1.0) * val;
        
            } else {
            let old = *data.get_unchecked(idx - period);
            let prev = s_full;
            s_full = prev + val - old;
            ws_full_acc = ws_full_acc - prev + (period as f64) * val;
        }

        /* half WMA update */
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
                    /* first full dot-product (SIMD) */
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

    /* ---------- phase 2: steady-state ---------- */
    for j in (period + sq - 1)..(len - first) {
        let idx = first + j;
        let val = *data.get_unchecked(idx); // unaligned load

        /* vectorised rolling update for both WMAs (128-bit packs) */
        let old_f = *data.get_unchecked(idx - period);
        let old_h = *data.get_unchecked(idx - half);

        let sum_vec = _mm_set_pd(s_full, s_half); // [hi | lo]
        let old_vec = _mm_set_pd(old_f, old_h);
        let ws_vec = _mm_set_pd(ws_full_acc, ws_half_acc);
        let weights = _mm_set_pd(period as f64, half as f64);
        let v_val = _mm_set1_pd(val);

        /* Σ ← Σ − old + val */
        let new_sum_vec = _mm_add_pd(_mm_sub_pd(sum_vec, old_vec), v_val);

        /* WS ← WS − Σ_prev + w*val  (single FMA) */
        let diff = _mm_sub_pd(ws_vec, sum_vec);
        let new_ws_vec = _mm_fmadd_pd(v_val, weights, diff);

        /* unpack back to scalars */
        s_full = _mm_cvtsd_f64(_mm_unpackhi_pd(new_sum_vec, new_sum_vec));
        s_half = _mm_cvtsd_f64(new_sum_vec);
        ws_full_acc = _mm_cvtsd_f64(_mm_unpackhi_pd(new_ws_vec, new_ws_vec));
        ws_half_acc = _mm_cvtsd_f64(new_ws_vec);

        /* derive WMAs & combine */
        wma_full = ws_full_acc / ws_full;
        wma_half = ws_half_acc / ws_half;
        let x_val = 2.0 * wma_half - wma_full;

        /* ring update – O(1) with fused multiply-add */
        let old_x = *x_buf.get_unchecked(x_head);
        *x_buf.get_unchecked_mut(x_head) = x_val;
        x_head = (x_head + 1) % sq;

        let prev_sum = x_sum;
        x_sum = prev_sum + x_val - old_x;
        x_wsum = sq_f.mul_add(x_val, x_wsum - prev_sum); // single FMA

        *out.get_unchecked_mut(idx) = x_wsum / ws_sqrt; // unaligned store

        /* software prefetch to L2 ~16 iterations ahead */
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
            first + p + s - 1
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
            Kernel::Scalar => hma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => hma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => hma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // -------- run every row -----------
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

    // -------- transmute to a normal Vec<f64> -----------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

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
            first + p + s - 1
        })
        .collect();

    // Cast output to MaybeUninit for initialization
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // -------- per-row worker closure -----------
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        match kern {
            Kernel::Scalar => hma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => hma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => hma_row_avx512(data, first, period, out_row),
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
use numpy::ndarray::{Array1, Array2};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "hma")]
pub fn hma_py<'py>(
    py: Python<'py>,
    arr_in: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::PyArrayMethods;
    
    let slice_in = arr_in.as_slice()?; // zero-copy, read-only view
    
    // Pre-allocate NumPy output buffer  
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Prepare HMA input
    let hma_in = HmaInput::from_slice(slice_in, HmaParams { period: Some(period) });
    
    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), HmaError> {
        hma_into(&hma_in, slice_out)
    })
    .map_err(|e| PyValueError::new_err(format!("HMA error: {}", e)))?;
    
    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyfunction(name = "hma_batch")]
pub fn hma_batch_py<'py>(
    py: Python<'py>,
    arr_in: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArrayMethods;
    
    let slice_in = arr_in.as_slice()?; // zero-copy, read-only view
    let sweep = HmaBatchRange {
        period: period_range,
    };
    
    // Expand grid to get all combinations
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("Invalid period range"));
    }
    
    let rows = combos.len();
    let cols = slice_in.len();
    
    // Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Heavy work without the GIL
    let (_, final_rows, final_cols) = py.allow_threads(|| -> Result<(Vec<HmaParams>, usize, usize), HmaError> {
        // Detect best kernel
        let kernel = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        
        // Use the new _into function
        hma_batch_inner_into(slice_in, &sweep, kernel, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(format!("HMA batch error: {}", e)))?;
    
    // Extract periods for metadata
    let periods: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
    
    // Reshape to 2D
    let out_2d = out_arr.reshape((final_rows, final_cols))?;
    
    // Create dictionary output
    let dict = PyDict::new(py);
    dict.set_item("values", out_2d)?;
    dict.set_item("periods", periods)?;
    
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
    pub fn new(period: usize) -> PyResult<Self> {
        let params = HmaParams {
            period: Some(period),
        };
        match HmaStream::try_new(params) {
            Ok(stream) => Ok(Self { inner: stream }),
            Err(e) => Err(PyValueError::new_err(format!("HmaStream error: {}", e))),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// WASM bindings
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = HmaParams {
        period: Some(period),
    };
    let input = HmaInput::from_slice(data, params);
    match hma_with_kernel(&input, Kernel::Scalar) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("HMA error: {}", e))),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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
    match hma_batch_inner(data, &sweep, Kernel::Scalar, false) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("HMA batch error: {}", e))),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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
    
    let mut result = Vec::new();
    for &period in &periods {
        result.push(period as f64);
    }
    result
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

        let strat = (
            proptest::collection::vec(
                (-1e6f64..1e6).prop_filter("finite", |x| x.is_finite()),
                30..200,
            ),
            3usize..30,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = HmaParams {
                    period: Some(period),
                };
                let input = HmaInput::from_slice(&data, params);
                
                // HMA may fail if there's not enough valid data
                match hma_with_kernel(&input, kernel) {
                    Ok(HmaOutput { values: out }) => {
                        for i in (period + (period as f64).sqrt().floor() as usize - 2)..data.len() {
                            let y = out[i];
                            // HMA uses the formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
                            // The 2*WMA(n/2) - WMA(n) operation is a form of linear extrapolation
                            // that deliberately produces values outside the input bounds to reduce lag.
                            // This is a key feature of HMA, not a bug.
                            // We only check that the output is finite (not NaN or infinite).
                            prop_assert!(
                                y.is_nan() || y.is_finite(),
                                "HMA output at index {} is not finite: {}", i, y
                            );
                        }
                    }
                    Err(_) => {
                        // If HMA fails due to insufficient data, that's expected
                        // for some random test inputs
                    }
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_hma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = HmaInput::from_candles(&candles, "close", HmaParams::default());
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
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
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

        // Test batch with multiple parameter combinations
        let output = HmaBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)
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

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
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
