//! # Qstick
//!
//! Qstick measures the average difference between the Close and Open over a specified period.
//! A positive Qstick indicates that, on average, the market closes above its open, while a
//! negative Qstick indicates the opposite.
//! Decision log: SIMD enabled (AVX2/AVX512) for initial sums; CUDA batch/many-series wrappers present with VRAM handles and Python CAI v3 + DLPack v1.x; scalar path remains the reference and numerical outputs stay unchanged.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 5.
//!
//! ## Returns
//! - **`Ok(QstickOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(QstickError)`** on failure
//!
//! ## Developer Notes
//! - SIMD enabled for initial window sum (AVX2/AVX512); rolling update stays scalar and unrolled.
//! - Scalar path is optimized, safe, and warmup-compatible (no temporaries, single pass).
//! - Batch (Avx2/Avx512) uses shared prefix of (close - open) across rows to remove redundant work.
//! - Memory: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)

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
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum QstickData<'a> {
    Candles {
        candles: &'a Candles,
        open_source: &'a str,
        close_source: &'a str,
    },
    Slices {
        open: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct QstickOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct QstickParams {
    pub period: Option<usize>,
}

impl Default for QstickParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct QstickInput<'a> {
    pub data: QstickData<'a>,
    pub params: QstickParams,
}

impl<'a> QstickInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        open_source: &'a str,
        close_source: &'a str,
        params: QstickParams,
    ) -> Self {
        Self {
            data: QstickData::Candles {
                candles,
                open_source,
                close_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(open: &'a [f64], close: &'a [f64], params: QstickParams) -> Self {
        Self {
            data: QstickData::Slices { open, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: QstickData::Candles {
                candles,
                open_source: "open",
                close_source: "close",
            },
            params: QstickParams::default(),
        }
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct QstickBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for QstickBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl QstickBuilder {
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
    pub fn apply(self, candles: &Candles) -> Result<QstickOutput, QstickError> {
        let params = QstickParams {
            period: self.period,
        };
        let input = QstickInput::from_candles(candles, "open", "close", params);
        qstick_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, open: &[f64], close: &[f64]) -> Result<QstickOutput, QstickError> {
        let params = QstickParams {
            period: self.period,
        };
        let input = QstickInput::from_slices(open, close, params);
        qstick_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<QstickStream, QstickError> {
        let params = QstickParams {
            period: self.period,
        };
        QstickStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum QstickError {
    #[error("qstick: Input data slice is empty.")]
    EmptyInputData,
    #[error("qstick: All values are NaN.")]
    AllValuesNaN,
    #[error("qstick: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("qstick: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("qstick: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("qstick: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("qstick: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("qstick: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn qstick(input: &QstickInput) -> Result<QstickOutput, QstickError> {
    qstick_with_kernel(input, Kernel::Auto)
}

pub fn qstick_with_kernel(
    input: &QstickInput,
    kernel: Kernel,
) -> Result<QstickOutput, QstickError> {
    let (open, close) = match &input.data {
        QstickData::Candles {
            candles,
            open_source,
            close_source,
        } => {
            let open = source_type(candles, open_source);
            let close = source_type(candles, close_source);
            (open, close)
        }
        QstickData::Slices { open, close } => (*open, *close),
    };

    let len = open.len().min(close.len());
    let period = input.get_period();

    if len == 0 {
        return Err(QstickError::EmptyInputData);
    }
    if period == 0 || period > len {
        return Err(QstickError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // Find first valid index by checking both open and close
    let mut first = 0;
    for i in 0..len {
        if !open[i].is_nan() && !close[i].is_nan() {
            first = i;
            break;
        }
        if i == len - 1 {
            return Err(QstickError::AllValuesNaN);
        }
    }

    if (len - first) < period {
        return Err(QstickError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let warmup_end = first
        .checked_add(period)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| QstickError::InvalidInput("warmup index overflow".into()))?;

    let mut out = alloc_with_nan_prefix(len, warmup_end);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                qstick_scalar(open, close, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => qstick_avx2(open, close, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                qstick_avx512(open, close, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(QstickOutput { values: out })
}

/// Write Qstick values into the provided output buffer without allocating.
///
/// - Preserves NaN warmups exactly like the Vec-returning API (alloc_with_nan_prefix):
///   fills the warmup prefix with quiet-NaNs and writes the first finite value at index
///   `first_valid + period - 1`.
/// - The output slice length must equal the input length (min of open/close lengths for slices).
/// - Uses `Kernel::Auto` to dispatch to the best available compute path.
#[cfg(not(feature = "wasm"))]
pub fn qstick_into(input: &QstickInput, out: &mut [f64]) -> Result<(), QstickError> {
    let (open, close) = match &input.data {
        QstickData::Candles {
            candles,
            open_source,
            close_source,
        } => {
            let open = source_type(candles, open_source);
            let close = source_type(candles, close_source);
            (open, close)
        }
        QstickData::Slices { open, close } => (*open, *close),
    };

    let len = open.len().min(close.len());
    let period = input.get_period();

    // Length/period validation mirrors the Vec API and `qstick_into_slice` behavior
    if len == 0 {
        return Err(QstickError::EmptyInputData);
    }
    if period == 0 || period > len {
        return Err(QstickError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if out.len() != len {
        return Err(QstickError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    // Find first valid index where both open and close are finite
    let mut first = 0usize;
    for i in 0..len {
        if !open[i].is_nan() && !close[i].is_nan() {
            first = i;
            break;
        }
        if i == len - 1 {
            return Err(QstickError::AllValuesNaN);
        }
    }

    if (len - first) < period {
        return Err(QstickError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // Warmup prefix: identical to alloc_with_nan_prefix (quiet NaN payload)
    let warm = first
        .checked_add(period)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| QstickError::InvalidInput("warmup index overflow".into()))?
        .min(len);
    for v in &mut out[..warm] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    // Dispatch to the best compute kernel
    let chosen = match Kernel::Auto {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => qstick_scalar(open, close, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => qstick_avx2(open, close, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => qstick_avx512(open, close, period, first, out),
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline]
pub fn qstick_scalar(
    open: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // Pre-checked by caller: period in [1..], len - first_valid >= period
    let len = open.len().min(close.len());
    if len == 0 {
        return;
    }

    let start = first_valid;
    let warm = start + period - 1;
    let inv_p = 1.0 / (period as f64);

    // Fast path: period == 1 ⇒ QStick = close - open
    if period == 1 {
        let mut i = start;
        // Unroll by 4, safe indexing
        while i + 3 < len {
            out[i] = close[i] - open[i];
            out[i + 1] = close[i + 1] - open[i + 1];
            out[i + 2] = close[i + 2] - open[i + 2];
            out[i + 3] = close[i + 3] - open[i + 3];
            i += 4;
        }
        while i < len {
            out[i] = close[i] - open[i];
            i += 1;
        }
        return;
    }

    // Initial window sum of (close - open)
    let mut sum = 0.0f64;
    let end_init = start + period;
    let mut k = start;
    // Unroll by 4 for fewer branches
    let end_unroll = start + ((period) & !3usize);
    while k < end_unroll {
        sum += (close[k] - open[k])
            + (close[k + 1] - open[k + 1])
            + (close[k + 2] - open[k + 2])
            + (close[k + 3] - open[k + 3]);
        k += 4;
    }
    while k < end_init {
        sum += close[k] - open[k];
        k += 1;
    }

    out[warm] = sum * inv_p;

    // Rolling update: add entering diff, subtract leaving diff; unrolled by 4
    let mut i_new = warm + 1;
    let mut i_old = start;
    while i_new + 3 < len {
        // 0
        sum = (sum + (close[i_new] - open[i_new])) - (close[i_old] - open[i_old]);
        out[i_new] = sum * inv_p;
        // 1
        sum = (sum + (close[i_new + 1] - open[i_new + 1])) - (close[i_old + 1] - open[i_old + 1]);
        out[i_new + 1] = sum * inv_p;
        // 2
        sum = (sum + (close[i_new + 2] - open[i_new + 2])) - (close[i_old + 2] - open[i_old + 2]);
        out[i_new + 2] = sum * inv_p;
        // 3
        sum = (sum + (close[i_new + 3] - open[i_new + 3])) - (close[i_old + 3] - open[i_old + 3]);
        out[i_new + 3] = sum * inv_p;

        i_new += 4;
        i_old += 4;
    }
    while i_new < len {
        sum = (sum + (close[i_new] - open[i_new])) - (close[i_old] - open[i_old]);
        out[i_new] = sum * inv_p;
        i_new += 1;
        i_old += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512(
    open: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { qstick_avx512_impl(open, close, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx2(
    open: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { qstick_avx2_impl(open, close, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512_short(
    open: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    qstick_avx512(open, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512_long(
    open: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    qstick_avx512(open, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn qstick_avx2_impl(
    open: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let len = open.len().min(close.len());
    if len == 0 {
        return;
    }
    let start = first_valid;
    let warm = start + period - 1;
    let inv_p = 1.0 / (period as f64);

    // period == 1
    if period == 1 {
        let mut i = start;
        while i + 3 < len {
            let c = _mm256_loadu_pd(close.as_ptr().add(i));
            let o = _mm256_loadu_pd(open.as_ptr().add(i));
            let d = _mm256_sub_pd(c, o);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), d);
            i += 4;
        }
        while i < len {
            *out.get_unchecked_mut(i) = *close.get_unchecked(i) - *open.get_unchecked(i);
            i += 1;
        }
        return;
    }

    // Initial window sum using AVX2
    let mut v_sum = _mm256_setzero_pd();
    let mut k = 0usize;
    let vec_end = period & !3usize; // multiple of 4
    while k < vec_end {
        let idx = start + k;
        let c = _mm256_loadu_pd(close.as_ptr().add(idx));
        let o = _mm256_loadu_pd(open.as_ptr().add(idx));
        let d = _mm256_sub_pd(c, o);
        v_sum = _mm256_add_pd(v_sum, d);
        k += 4;
    }
    // horizontal reduce v_sum
    let hi = _mm256_extractf128_pd(v_sum, 1);
    let lo = _mm256_castpd256_pd128(v_sum);
    let s2 = _mm_add_pd(lo, hi);
    let s1 = _mm_hadd_pd(s2, s2);
    let mut sum = _mm_cvtsd_f64(s1);

    while k < period {
        let idx = start + k;
        sum += *close.get_unchecked(idx) - *open.get_unchecked(idx);
        k += 1;
    }

    *out.get_unchecked_mut(warm) = sum * inv_p;

    // Rolling update remains scalar; unroll by 4
    let mut i_new = warm + 1;
    let mut i_old = start;
    while i_new + 3 < len {
        // 0
        sum = (sum + (*close.get_unchecked(i_new) - *open.get_unchecked(i_new)))
            - (*close.get_unchecked(i_old) - *open.get_unchecked(i_old));
        *out.get_unchecked_mut(i_new) = sum * inv_p;
        // 1
        sum = (sum + (*close.get_unchecked(i_new + 1) - *open.get_unchecked(i_new + 1)))
            - (*close.get_unchecked(i_old + 1) - *open.get_unchecked(i_old + 1));
        *out.get_unchecked_mut(i_new + 1) = sum * inv_p;
        // 2
        sum = (sum + (*close.get_unchecked(i_new + 2) - *open.get_unchecked(i_new + 2)))
            - (*close.get_unchecked(i_old + 2) - *open.get_unchecked(i_old + 2));
        *out.get_unchecked_mut(i_new + 2) = sum * inv_p;
        // 3
        sum = (sum + (*close.get_unchecked(i_new + 3) - *open.get_unchecked(i_new + 3)))
            - (*close.get_unchecked(i_old + 3) - *open.get_unchecked(i_old + 3));
        *out.get_unchecked_mut(i_new + 3) = sum * inv_p;

        i_new += 4;
        i_old += 4;
    }
    while i_new < len {
        sum = (sum + (*close.get_unchecked(i_new) - *open.get_unchecked(i_new)))
            - (*close.get_unchecked(i_old) - *open.get_unchecked(i_old));
        *out.get_unchecked_mut(i_new) = sum * inv_p;
        i_new += 1;
        i_old += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn qstick_avx512_impl(
    open: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let len = open.len().min(close.len());
    if len == 0 {
        return;
    }
    let start = first_valid;
    let warm = start + period - 1;
    let inv_p = 1.0 / (period as f64);

    // period == 1
    if period == 1 {
        let mut i = start;
        while i + 7 < len {
            let c = _mm512_loadu_pd(close.as_ptr().add(i));
            let o = _mm512_loadu_pd(open.as_ptr().add(i));
            let d = _mm512_sub_pd(c, o);
            _mm512_storeu_pd(out.as_mut_ptr().add(i), d);
            i += 8;
        }
        while i < len {
            *out.get_unchecked_mut(i) = *close.get_unchecked(i) - *open.get_unchecked(i);
            i += 1;
        }
        return;
    }

    // Initial window sum using AVX512
    let mut v_sum = _mm512_setzero_pd();
    let mut k = 0usize;
    let vec_end = period & !7usize; // multiple of 8
    while k < vec_end {
        let idx = start + k;
        let c = _mm512_loadu_pd(close.as_ptr().add(idx));
        let o = _mm512_loadu_pd(open.as_ptr().add(idx));
        let d = _mm512_sub_pd(c, o);
        v_sum = _mm512_add_pd(v_sum, d);
        k += 8;
    }
    // reduce 8-lane vector to scalar
    let lo256 = _mm512_castpd512_pd256(v_sum);
    let hi256 = _mm512_extractf64x4_pd(v_sum, 1);
    let lo_hi128 = _mm256_extractf128_pd(lo256, 1);
    let lo_lo128 = _mm256_castpd256_pd128(lo256);
    let lo_s2 = _mm_add_pd(lo_lo128, lo_hi128);
    let lo_s1 = _mm_hadd_pd(lo_s2, lo_s2);
    let s_lo = _mm_cvtsd_f64(lo_s1);

    let hi_hi128 = _mm256_extractf128_pd(hi256, 1);
    let hi_lo128 = _mm256_castpd256_pd128(hi256);
    let hi_s2 = _mm_add_pd(hi_lo128, hi_hi128);
    let hi_s1 = _mm_hadd_pd(hi_s2, hi_s2);
    let s_hi = _mm_cvtsd_f64(hi_s1);

    let mut sum = s_lo + s_hi;
    while k < period {
        let idx = start + k;
        sum += *close.get_unchecked(idx) - *open.get_unchecked(idx);
        k += 1;
    }

    *out.get_unchecked_mut(warm) = sum * inv_p;

    // Rolling update remains scalar; unroll by 4
    let mut i_new = warm + 1;
    let mut i_old = start;
    while i_new + 3 < len {
        sum = (sum + (*close.get_unchecked(i_new) - *open.get_unchecked(i_new)))
            - (*close.get_unchecked(i_old) - *open.get_unchecked(i_old));
        *out.get_unchecked_mut(i_new) = sum * inv_p;

        sum = (sum + (*close.get_unchecked(i_new + 1) - *open.get_unchecked(i_new + 1)))
            - (*close.get_unchecked(i_old + 1) - *open.get_unchecked(i_old + 1));
        *out.get_unchecked_mut(i_new + 1) = sum * inv_p;

        sum = (sum + (*close.get_unchecked(i_new + 2) - *open.get_unchecked(i_new + 2)))
            - (*close.get_unchecked(i_old + 2) - *open.get_unchecked(i_old + 2));
        *out.get_unchecked_mut(i_new + 2) = sum * inv_p;

        sum = (sum + (*close.get_unchecked(i_new + 3) - *open.get_unchecked(i_new + 3)))
            - (*close.get_unchecked(i_old + 3) - *open.get_unchecked(i_old + 3));
        *out.get_unchecked_mut(i_new + 3) = sum * inv_p;

        i_new += 4;
        i_old += 4;
    }
    while i_new < len {
        sum = (sum + (*close.get_unchecked(i_new) - *open.get_unchecked(i_new)))
            - (*close.get_unchecked(i_old) - *open.get_unchecked(i_old));
        *out.get_unchecked_mut(i_new) = sum * inv_p;
        i_new += 1;
        i_old += 1;
    }
}

#[inline]
pub fn qstick_batch_with_kernel(
    open: &[f64],
    close: &[f64],
    sweep: &QstickBatchRange,
    kernel: Kernel,
) -> Result<QstickBatchOutput, QstickError> {
    let kern = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(QstickError::InvalidKernelForBatch(kernel))
        }
    };
    let simd = match kern {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    qstick_batch_par_slice(open, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct QstickBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for QstickBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 254, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct QstickBatchBuilder {
    range: QstickBatchRange,
    kernel: Kernel,
}

impl QstickBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<QstickBatchOutput, QstickError> {
        qstick_batch_with_kernel(open, close, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        open_src: &str,
        close_src: &str,
    ) -> Result<QstickBatchOutput, QstickError> {
        let open = source_type(c, open_src);
        let close = source_type(c, close_src);
        self.apply_slices(open, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<QstickBatchOutput, QstickError> {
        QstickBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "open", "close")
    }
}

#[derive(Clone, Debug)]
pub struct QstickBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<QstickParams>,
    pub rows: usize,
    pub cols: usize,
}

impl QstickBatchOutput {
    pub fn row_for_params(&self, p: &QstickParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &QstickParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &QstickBatchRange) -> Result<Vec<QstickParams>, QstickError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, QstickError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                let next = cur.saturating_add(step);
                if next == cur {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                let next = cur.saturating_sub(step);
                if next == cur {
                    break;
                }
                cur = next;
                if cur == 0 && end > 0 {
                    break;
                }
            }
        }
        if v.is_empty() {
            return Err(QstickError::InvalidRange { start, end, step });
        }
        Ok(v)
    }

    let (start, end, step) = r.period;
    let periods = axis_usize((start, end, step))?;
    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(QstickParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn qstick_batch_slice(
    open: &[f64],
    close: &[f64],
    sweep: &QstickBatchRange,
    kern: Kernel,
) -> Result<QstickBatchOutput, QstickError> {
    qstick_batch_inner(open, close, sweep, kern, false)
}

#[inline(always)]
pub fn qstick_batch_par_slice(
    open: &[f64],
    close: &[f64],
    sweep: &QstickBatchRange,
    kern: Kernel,
) -> Result<QstickBatchOutput, QstickError> {
    qstick_batch_inner(open, close, sweep, kern, true)
}

#[inline(always)]
fn qstick_batch_inner(
    open: &[f64],
    close: &[f64],
    sweep: &QstickBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<QstickBatchOutput, QstickError> {
    let combos = expand_grid(sweep)?;
    let len = open.len().min(close.len());
    if len == 0 {
        return Err(QstickError::EmptyInputData);
    }

    // Find first valid index
    let mut first = 0;
    for i in 0..len {
        if !open[i].is_nan() && !close[i].is_nan() {
            first = i;
            break;
        }
        if i == len - 1 {
            return Err(QstickError::AllValuesNaN);
        }
    }

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(QstickError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    let total_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| QstickError::InvalidInput("rows*cols overflow".into()))?;

    // Use proper uninitialized memory allocation like ALMA
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each parameter combination
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            first
                .checked_add(c.period.unwrap_or(0))
                .and_then(|v| v.checked_sub(1))
                .unwrap_or(first)
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to mutable slice
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    qstick_batch_inner_into(open, close, sweep, kern, parallel, out)?;

    // Take ownership of the buffer
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            total_elems,
            buf_guard.capacity(),
        )
    };

    Ok(QstickBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn qstick_batch_inner_into(
    open: &[f64],
    close: &[f64],
    sweep: &QstickBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<QstickParams>, QstickError> {
    let combos = expand_grid(sweep)?;

    let len = open.len().min(close.len());
    if len == 0 {
        return Err(QstickError::EmptyInputData);
    }
    let cols = len;

    // first valid across both inputs
    let first = (0..len)
        .find(|&i| !open[i].is_nan() && !close[i].is_nan())
        .ok_or(QstickError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(QstickError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    // Initialize NaN prefixes for each row based on warmup period
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first
            .checked_add(combo.period.unwrap_or(0))
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| QstickError::InvalidInput("warmup index overflow".into()))?;
        let row_start = row
            .checked_mul(cols)
            .ok_or_else(|| QstickError::InvalidInput("row*cols overflow".into()))?;
        for i in 0..warmup.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    // If using AVX2/AVX512 batch kernels, compute shared prefix of (close - open)
    // to eliminate redundant work across rows. ScalarBatch path remains per-row.
    match kern {
        Kernel::Avx2Batch | Kernel::Avx512Batch => {
            qstick_batch_shared_prefix_into(open, close, &combos, first, cols, out);
            return Ok(combos);
        }
        _ => {}
    }

    // Treat output as uninitialized, like alma_batch_inner_into for per-row path
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        // cast current row to f64 slice
        let dst: &mut [f64] =
            std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                qstick_scalar(open, close, period, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => qstick_avx2(open, close, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => qstick_avx512(open, close, period, first, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            _ => qstick_scalar(open, close, period, first, dst),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
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

#[inline(always)]
fn qstick_batch_shared_prefix_into(
    open: &[f64],
    close: &[f64],
    combos: &[QstickParams],
    first: usize,
    cols: usize,
    out: &mut [f64],
) {
    let len = cols;
    if len == 0 {
        return;
    }
    // Build diff d[i] = close[i] - open[i] and prefix P where P[0]=0, P[i+1]=P[i]+d[i]
    let cap = len
        .checked_add(1)
        .unwrap_or(len);
    let mut prefix = Vec::with_capacity(cap);
    prefix.push(0.0);
    let mut acc = 0.0f64;
    // Before `first`, diffs are treated as zero; start accumulation at `first`
    let mut i = 0usize;
    while i < first && i < len {
        prefix.push(acc);
        i += 1;
    }
    while i + 3 < len {
        let d0 = close[i] - open[i];
        let d1 = close[i + 1] - open[i + 1];
        let d2 = close[i + 2] - open[i + 2];
        let d3 = close[i + 3] - open[i + 3];
        acc += d0;
        prefix.push(acc);
        acc += d1;
        prefix.push(acc);
        acc += d2;
        prefix.push(acc);
        acc += d3;
        prefix.push(acc);
        i += 4;
    }
    while i < len {
        acc += close[i] - open[i];
        prefix.push(acc);
        i += 1;
    }

    // For each row, fill from warmup onward using prefix sums
    for (row, combo) in combos.iter().enumerate() {
        let p = combo.period.unwrap_or(5);
        let warm = first
            .checked_add(p)
            .and_then(|v| v.checked_sub(1))
            .unwrap_or(first);
        if warm >= len {
            continue;
        }
        let row_start = row
            .checked_mul(cols)
            .unwrap_or(0);
        let inv_p = 1.0 / (p as f64);
        let mut j = warm;
        while j + 3 < len {
            let s0 = prefix[j + 1] - prefix[j + 1 - p];
            let s1 = prefix[j + 2] - prefix[j + 2 - p];
            let s2 = prefix[j + 3] - prefix[j + 3 - p];
            let s3 = prefix[j + 4] - prefix[j + 4 - p];
            out[row_start + j] = s0 * inv_p;
            out[row_start + j + 1] = s1 * inv_p;
            out[row_start + j + 2] = s2 * inv_p;
            out[row_start + j + 3] = s3 * inv_p;
            j += 4;
        }
        while j < len {
            let s = prefix[j + 1] - prefix[j + 1 - p];
            out[row_start + j] = s * inv_p;
            j += 1;
        }
    }
}

#[inline(always)]
unsafe fn qstick_row_scalar(
    open: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    qstick_scalar(open, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx2(
    open: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    qstick_avx2(open, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512(
    open: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    qstick_avx512(open, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512_short(
    open: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    qstick_avx512_short(open, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512_long(
    open: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    qstick_avx512_long(open, close, period, first, out)
}

#[derive(Debug, Clone)]
pub struct QstickStream {
    /// Decision: streaming kernel uses O(1) ring buffer with length counter and reciprocal.
    /// Same warmup semantics; avoids NaN sentinel and uses bitmask wrap when period is power-of-two.
    period: usize,
    inv_p: f64,       // 1.0 / period, avoids a division every tick
    buffer: Vec<f64>, // ring buffer of last `period` diffs (close - open)
    head: usize,      // write index
    len: usize,       // number of valid elements in buffer, clamped to period
    sum: f64,         // rolling sum of diffs in the current window
    mask: usize,      // period - 1 if period is power-of-two, else 0
}

impl QstickStream {
    #[inline(always)]
    pub fn try_new(params: QstickParams) -> Result<Self, QstickError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(QstickError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let mask = if period.is_power_of_two() {
            period - 1
        } else {
            0
        };
        Ok(Self {
            period,
            inv_p: 1.0 / (period as f64),
            buffer: vec![0.0; period], // contents irrelevant until len == period
            head: 0,
            len: 0,
            sum: 0.0,
            mask,
        })
    }

    /// O(1) update: add newest (close - open), subtract the element we overwrite (once full).
    /// Warmup: returns None until `period` samples have been seen, then always returns Some(avg).
    #[inline(always)]
    pub fn update(&mut self, open: f64, close: f64) -> Option<f64> {
        let diff = close - open;
        let h = self.head;

        if self.len < self.period {
            // Filling phase: no subtraction yet
            self.buffer[h] = diff;
            self.sum += diff;
            self.head = if self.mask != 0 {
                (h + 1) & self.mask
            } else if h + 1 == self.period {
                0
            } else {
                h + 1
            };
            self.len += 1;
            if self.len == self.period {
                Some(self.sum * self.inv_p)
            } else {
                None
            }
        } else {
            // Steady-state: subtract the one we overwrite, add the new one
            let old = self.buffer[h];
            self.sum += diff - old;
            self.buffer[h] = diff;
            self.head = if self.mask != 0 {
                (h + 1) & self.mask
            } else if h + 1 == self.period {
                0
            } else {
                h + 1
            };
            Some(self.sum * self.inv_p)
        }
    }

    /// Optional helper: reset without reallocating. Keeps buffer contents; len=0 disables subtraction until refilled.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
        self.sum = 0.0;
    }

    /// Optional fast path if you already have (close - open) upstream.
    #[inline(always)]
    pub fn update_diff(&mut self, diff: f64) -> Option<f64> {
        let h = self.head;

        if self.len < self.period {
            self.buffer[h] = diff;
            self.sum += diff;
            self.head = if self.mask != 0 {
                (h + 1) & self.mask
            } else if h + 1 == self.period {
                0
            } else {
                h + 1
            };
            self.len += 1;
            if self.len == self.period {
                Some(self.sum * self.inv_p)
            } else {
                None
            }
        } else {
            let old = self.buffer[h];
            self.sum += diff - old;
            self.buffer[h] = diff;
            self.head = if self.mask != 0 {
                (h + 1) & self.mask
            } else if h + 1 == self.period {
                0
            } else {
                h + 1
            };
            Some(self.sum * self.inv_p)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[test]
    fn test_qstick_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Use the repo CSV data to mirror other qstick tests
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = QstickParams { period: Some(5) }; // default behavior
        let input = QstickInput::from_candles(&candles, "open", "close", params);

        // Baseline via existing Vec-returning API
        let baseline = qstick(&input)?.values;

        // New into API writes in-place
        let mut into_out = vec![0.0; baseline.len()];
        #[cfg(not(feature = "wasm"))]
        qstick_into(&input, &mut into_out)?;

        assert_eq!(baseline.len(), into_out.len());
        for (a, b) in baseline.iter().zip(into_out.iter()) {
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(equal, "qstick_into mismatch: a={}, b={}", a, b);
        }
        Ok(())
    }
    fn check_qstick_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = QstickParams { period: None };
        let input_default = QstickInput::from_candles(&candles, "open", "close", default_params);
        let output_default = qstick_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period_7 = QstickParams { period: Some(7) };
        let input_period_7 = QstickInput::from_candles(&candles, "open", "close", params_period_7);
        let output_period_7 = qstick_with_kernel(&input_period_7, kernel)?;
        assert_eq!(output_period_7.values.len(), candles.close.len());
        Ok(())
    }
    fn check_qstick_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = QstickParams { period: Some(5) };
        let input = QstickInput::from_candles(&candles, "open", "close", params);
        let result = qstick_with_kernel(&input, kernel)?;
        let expected_last_five_qstick = [219.4, 61.6, -51.8, -53.4, -123.2];
        let start_index = result.values.len() - 5;
        let result_last_five = &result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five_qstick[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "[{}] Qstick mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                value,
                expected_value
            );
        }
        Ok(())
    }
    fn check_qstick_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let open_data = [10.0, 20.0, 30.0];
        let close_data = [15.0, 25.0, 35.0];
        let params = QstickParams { period: Some(0) };
        let input = QstickInput::from_slices(&open_data, &close_data, params);
        let res = qstick_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Qstick should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_qstick_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let open_data = [10.0, 20.0, 30.0];
        let close_data = [15.0, 25.0, 35.0];
        let params = QstickParams { period: Some(10) };
        let input = QstickInput::from_slices(&open_data, &close_data, params);
        let res = qstick_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Qstick should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_qstick_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let open_data = [50.0];
        let close_data = [55.0];
        let params = QstickParams { period: Some(5) };
        let input = QstickInput::from_slices(&open_data, &close_data, params);
        let res = qstick_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Qstick should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_qstick_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = QstickParams { period: Some(5) };
        let first_input = QstickInput::from_candles(&candles, "open", "close", first_params);
        let first_result = qstick_with_kernel(&first_input, kernel)?;
        let second_params = QstickParams { period: Some(5) };
        let second_input =
            QstickInput::from_slices(&first_result.values, &first_result.values, second_params);
        let second_result = qstick_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 10..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Qstick Slice Reinput: Expected no NaN after idx 10, found NaN at idx {}",
                test_name,
                i
            );
        }
        Ok(())
    }
    fn check_qstick_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = QstickParams { period: Some(5) };
        let input = QstickInput::from_candles(&candles, "open", "close", params);
        let qstick_result = qstick_with_kernel(&input, kernel)?;
        if qstick_result.values.len() > 50 {
            for i in 50..qstick_result.values.len() {
                assert!(
                    !qstick_result.values[i].is_nan(),
                    "[{}] Expected no NaN after index 50, found NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_qstick_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            QstickParams::default(),            // period: 5
            QstickParams { period: Some(2) },   // minimum viable
            QstickParams { period: Some(3) },   // small
            QstickParams { period: Some(7) },   // small-medium
            QstickParams { period: Some(10) },  // medium
            QstickParams { period: Some(20) },  // medium-large
            QstickParams { period: Some(30) },  // large
            QstickParams { period: Some(50) },  // very large
            QstickParams { period: Some(100) }, // extra large
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = QstickInput::from_candles(&candles, "open", "close", params.clone());
            let output = qstick_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_qstick_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_qstick_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating test data
        let strat = (1usize..=64).prop_flat_map(|period| {
            (period..=400usize).prop_flat_map(move |len| {
                (
                    // Generate open prices
                    prop::collection::vec(
                        (1.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    // Generate close prices as deltas from open
                    prop::collection::vec(
                        (-100.0f64..100.0f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    Just(period),
                )
            })
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(open_prices, close_deltas, period)| {
                // Create close prices from open + delta
                let close_prices: Vec<f64> = open_prices
                    .iter()
                    .zip(close_deltas.iter())
                    .map(|(o, d)| o + d)
                    .collect();

                let params = QstickParams {
                    period: Some(period),
                };
                let input = QstickInput::from_slices(&open_prices, &close_prices, params);

                let QstickOutput { values: out } = qstick_with_kernel(&input, kernel).unwrap();
                let QstickOutput { values: ref_out } =
                    qstick_with_kernel(&input, Kernel::Scalar).unwrap();

                // Test 1: Warmup period - first (period - 1) values should be NaN
                for i in 0..(period - 1) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Test valid output values
                for i in (period - 1)..open_prices.len() {
                    let window_start = i + 1 - period;
                    let window_end = i + 1;

                    // Calculate the differences in the window
                    let diffs: Vec<f64> = (window_start..window_end)
                        .map(|j| close_prices[j] - open_prices[j])
                        .collect();

                    // Test 2: Bounds property - QStick should be within min/max of differences
                    let min_diff = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_diff = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];

                    prop_assert!(
                        y.is_nan() || (y >= min_diff - 1e-9 && y <= max_diff + 1e-9),
                        "idx {}: QStick {} not in bounds [{}, {}]",
                        i,
                        y,
                        min_diff,
                        max_diff
                    );

                    // Test 3: Period=1 property - should equal close - open
                    if period == 1 {
                        let expected = close_prices[i] - open_prices[i];
                        prop_assert!(
                            (y - expected).abs() <= 1e-10,
                            "Period=1: expected {}, got {} at index {}",
                            expected,
                            y,
                            i
                        );
                    }

                    // Test 4: Constant difference property
                    if diffs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                        let expected = diffs[0];
                        prop_assert!(
                            (y - expected).abs() <= 1e-9,
                            "Constant diff: expected {}, got {} at index {}",
                            expected,
                            y,
                            i
                        );
                    }

                    // Test 5: Zero difference property - if open == close everywhere
                    if diffs.iter().all(|&d| d.abs() < 1e-10) {
                        prop_assert!(
                            y.abs() <= 1e-9,
                            "Zero diff: expected 0, got {} at index {}",
                            y,
                            i
                        );
                    }

                    // Test 6: Manual calculation verification
                    let expected_qstick = diffs.iter().sum::<f64>() / (period as f64);
                    prop_assert!(
                        (y - expected_qstick).abs() <= 1e-9,
                        "Manual calc: expected {}, got {} at index {}",
                        expected_qstick,
                        y,
                        i
                    );

                    // Test 7: Kernel consistency - compare with scalar reference
                    let r = ref_out[i];
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

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

                    // ULP tolerance for floating point comparison
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch idx {}: {} vs {} (ULP={})",
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

    macro_rules! generate_all_qstick_tests {
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
    generate_all_qstick_tests!(
        check_qstick_partial_params,
        check_qstick_accuracy,
        check_qstick_zero_period,
        check_qstick_period_exceeds_length,
        check_qstick_very_small_dataset,
        check_qstick_reinput,
        check_qstick_nan_handling,
        check_qstick_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_qstick_tests!(check_qstick_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = QstickBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "open", "close")?;
        let def = QstickParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [219.4, 61.6, -51.8, -53.4, -123.2];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            (2, 10, 2),   // Small periods
            (5, 25, 5),   // Medium periods
            (30, 60, 15), // Large periods
            (2, 5, 1),    // Dense small range
            (10, 50, 10), // Wide range medium step
            (15, 30, 5),  // Medium range small step
            (2, 100, 20), // Full range large step
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = QstickBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "open", "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
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

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(name = "QstickDeviceArrayF32Py")]
pub struct QstickDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl QstickDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
            .as_device_ptr()
            .as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        // Producing stream is synchronized before return; omit the 'stream' key
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        // Compute target device id and validate `dl_device` hint if provided,
        // matching the shared DLPack CUDA helpers.
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

        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = self.rows;
        let cols = self.cols;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}
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
#[pyfunction(name = "qstick")]
#[pyo3(signature = (open, close, period, kernel=None))]
pub fn qstick_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let open_slice = open.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = QstickParams {
        period: Some(period),
    };
    let input = QstickInput::from_slices(open_slice, close_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| qstick_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "QstickStream")]
pub struct QstickStreamPy {
    stream: QstickStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl QstickStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = QstickParams {
            period: Some(period),
        };
        let stream =
            QstickStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(QstickStreamPy { stream })
    }

    pub fn update(&mut self, open: f64, close: f64) -> Option<f64> {
        self.stream.update(open, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "qstick_batch")]
#[pyo3(signature = (open, close, period_range, kernel=None))]
pub fn qstick_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open_slice = open.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = QstickBatchRange {
        period: period_range,
    };

    let combos =
        expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = open_slice.len();

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
                _ => kernel,
            };

            // Use the optimized batch_inner_into function
            qstick_batch_inner_into(open_slice, close_slice, &sweep, simd, true, slice_out)
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
pub fn register_qstick_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qstick_py, m)?)?;
    m.add_function(wrap_pyfunction!(qstick_batch_py, m)?)?;
    m.add_class::<QstickStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_class::<QstickDeviceArrayF32Py>()?;
    }
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(qstick_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            qstick_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    Ok(())
}

// ==================== PYTHON: CUDA BINDINGS (zero-copy) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "qstick_cuda_batch_dev")]
#[pyo3(signature = (open_f32, close_f32, period_range, device_id=0))]
pub fn qstick_cuda_batch_dev_py(
    py: Python<'_>,
    open_f32: numpy::PyReadonlyArray1<'_, f32>,
    close_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<QstickDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaQstick;
    use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
    use cust::context::Context;
    use cust::memory::DeviceBuffer;
    use std::sync::Arc;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let open_slice = open_f32.as_slice()?;
    let close_slice = close_f32.as_slice()?;
    let sweep = QstickBatchRange {
        period: period_range,
    };
    let (buf, rows, cols, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaQstick::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out: DeviceArrayF32 = cuda
            .qstick_batch_dev(open_slice, close_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx_arc: Arc<Context> = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((out.buf, out.rows, out.cols, ctx_arc, cuda.device_id()))
    })?;
    Ok(QstickDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "qstick_cuda_many_series_one_param_dev")]
#[pyo3(signature = (open_tm_f32, close_tm_f32, period, device_id=0))]
pub fn qstick_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    open_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    close_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<QstickDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaQstick;
    use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
    use cust::context::Context;
    use cust::memory::DeviceBuffer;
    use numpy::PyUntypedArrayMethods;
    use std::sync::Arc;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if open_tm_f32.shape() != close_tm_f32.shape() {
        return Err(PyValueError::new_err("open/close shapes differ"));
    }
    let flat_open: &[f32] = open_tm_f32.as_slice()?;
    let flat_close: &[f32] = close_tm_f32.as_slice()?;
    let rows = open_tm_f32.shape()[0];
    let cols = open_tm_f32.shape()[1];

    let (buf, r_out, c_out, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaQstick::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out: DeviceArrayF32 = cuda
            .qstick_many_series_one_param_time_major_dev(flat_open, flat_close, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx_arc: Arc<Context> = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((out.buf, out.rows, out.cols, ctx_arc, cuda.device_id()))
    })?;
    Ok(QstickDeviceArrayF32Py {
        buf: Some(buf),
        rows: r_out,
        cols: c_out,
        _ctx: ctx,
        device_id: dev_id,
    })
}

/// Write qstick directly to output slice - no allocations
pub fn qstick_into_slice(
    dst: &mut [f64],
    open: &[f64],
    close: &[f64],
    period: usize,
    kern: Kernel,
) -> Result<(), QstickError> {
    // Validate inputs
    let len = open.len().min(close.len());
    if len == 0 {
        return Err(QstickError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if dst.len() != len {
        return Err(QstickError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }
    if period == 0 || period > len {
        return Err(QstickError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // Find first valid index
    let mut first_valid = 0;
    for i in 0..len {
        if !open[i].is_nan() && !close[i].is_nan() {
            first_valid = i;
            break;
        }
        if i == len - 1 {
            return Err(QstickError::AllValuesNaN);
        }
    }

    // Check if we have enough valid data
    if len - first_valid < period {
        return Err(QstickError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid,
        });
    }

    // Compute directly into output
    let kernel = match kern {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    match kernel {
        Kernel::Scalar | Kernel::ScalarBatch => {
            qstick_scalar(open, close, period, first_valid, dst)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => qstick_avx2(open, close, period, first_valid, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            qstick_avx512(open, close, period, first_valid, dst)
        }
        _ => unreachable!(),
    }

    // match alma.rs: apply warmup NaNs after compute
    let warmup_end = first_valid + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_js(open: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    // Validate minimum inputs
    let len = open.len();
    if len != close.len() {
        return Err(JsValue::from_str(
            "Open and close arrays must have the same length",
        ));
    }

    // Single allocation
    let mut output = vec![0.0; len];

    qstick_into_slice(&mut output, open, close, period, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        // CRITICAL: Check for aliasing
        // If either input overlaps with output, we need a temporary buffer
        if open_ptr == out_ptr || close_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            qstick_into_slice(&mut temp, open, close, period, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            qstick_into_slice(out, open, close, period, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct QstickBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct QstickBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<QstickParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = qstick_batch)]
pub fn qstick_batch_unified_js(
    open: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: QstickBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let len = open.len();
    if len != close.len() {
        return Err(JsValue::from_str(
            "Open and close arrays must have the same length",
        ));
    }

    let sweep = QstickBatchRange {
        period: config.period_range,
    };

    let output = qstick_batch_inner(open, close, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = QstickBatchJsOutput {
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
pub fn qstick_batch_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let sweep = QstickBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total_size = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("size overflow"))?;

        // For batch, we write directly to the output - no aliasing check needed
        // as batch output is different size than input
        let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

        qstick_batch_inner_into(open, close, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
