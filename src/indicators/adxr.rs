//! # Average Directional Index Rating (ADXR)
//!
//! ## OPTIMIZED PYTHON BINDINGS
//! The Python bindings (`adxr_py` and `adxr_batch_py`) use zero-copy transfers
//! for optimal performance, following the same patterns as alma.rs.
//!
//! A smoothed trend indicator: the average of the current ADX value and the ADX from `period` bars ago.
//! API and features modeled after alma.rs, including batch, AVX, and streaming support.
//!
//! ## Parameters
//! - **period**: Window size (number of bars), default 14.
//!
//! ## Errors
//! - **CandleFieldError**: ADXR: Failed to retrieve `high`, `low`, or `close`.
//! - **HlcLengthMismatch**: ADXR: Provided slices are of different lengths.
//! - **AllValuesNaN**: ADXR: All input values are `NaN`.
//! - **InvalidPeriod**: ADXR: period is zero or exceeds data length.
//! - **NotEnoughValidData**: ADXR: Insufficient valid data for requested period.
//!
//! ## Returns
//! - **`Ok(AdxrOutput)`** on success, else **`Err(AdxrError)`**.
//!
//! ## Decision Log
//! - SIMD: enabled but delegates to scalar; ADXR’s Wilder smoothing is time‑recursive and single‑series SIMD shows limited wins, so scalar is the reference.
//! - CUDA: FP32 batch and many‑series kernels enabled via `CudaAdxr`, matched against scalar ADXR within relaxed tolerances.
//! - Python: zero‑copy NumPy, CuPy CAI v3, and DLPack v1.x (with versioned capsules) wired for VRAM‑resident ADXR outputs.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaAdxr;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

// Note: AVec<f64> already implements Send since f64 is Send
// No wrapper types needed

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AdxrData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdxrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AdxrParams {
    pub period: Option<usize>,
}

impl Default for AdxrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AdxrInput<'a> {
    pub data: AdxrData<'a>,
    pub params: AdxrParams,
}

impl<'a> AdxrInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: AdxrParams) -> Self {
        Self {
            data: AdxrData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(h: &'a [f64], l: &'a [f64], c: &'a [f64], p: AdxrParams) -> Self {
        Self {
            data: AdxrData::Slices {
                high: h,
                low: l,
                close: c,
            },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, AdxrParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdxrBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for AdxrBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdxrBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<AdxrOutput, AdxrError> {
        let p = AdxrParams {
            period: self.period,
        };
        let i = AdxrInput::from_candles(c, p);
        adxr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, h: &[f64], l: &[f64], c: &[f64]) -> Result<AdxrOutput, AdxrError> {
        let p = AdxrParams {
            period: self.period,
        };
        let i = AdxrInput::from_slices(h, l, c, p);
        adxr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AdxrStream, AdxrError> {
        let p = AdxrParams {
            period: self.period,
        };
        AdxrStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AdxrError {
    #[error("adxr: Candle field error: {0}")]
    CandleFieldError(String),
    #[error("adxr: Empty input data (All values are NaN).")]
    EmptyInputData,
    #[error("adxr: HLC data length mismatch: high={high_len}, low={low_len}, close={close_len}")]
    HlcLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("adxr: All values are NaN.")]
    AllValuesNaN,
    #[error("adxr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    // Keep message text stable for existing tests that expect "Not enough data"
    #[error("adxr: Not enough data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("adxr: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("adxr: Invalid kernel type - expected batch kernel, got {kernel:?}")]
    InvalidKernel { kernel: Kernel },
    #[error("adxr: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("adxr: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn adxr(input: &AdxrInput) -> Result<AdxrOutput, AdxrError> {
    adxr_with_kernel(input, Kernel::Auto)
}

pub fn adxr_with_kernel(input: &AdxrInput, kernel: Kernel) -> Result<AdxrOutput, AdxrError> {
    let (high, low, close, period, first, chosen) = adxr_prepare(input, kernel)?;

    let len = close.len();

    // ADXR needs warmup period of first + 2 * period
    // (ADX appears at first + 2*period, then ADXR averages current and period-ago ADX)
    let warmup_period = first + 2 * period;
    let mut out = alloc_with_nan_prefix(len, warmup_period);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                adxr_scalar(high, low, close, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                adxr_avx2(high, low, close, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                adxr_avx512(high, low, close, period, first, &mut out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                adxr_scalar(high, low, close, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(AdxrOutput { values: out })
}

/// Prepare ADXR computation parameters
#[inline(always)]
fn adxr_prepare<'a>(
    input: &'a AdxrInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), AdxrError> {
    let (high, low, close) = match &input.data {
        AdxrData::Candles { candles } => (
            candles
                .select_candle_field("high")
                .map_err(|e| AdxrError::CandleFieldError(e.to_string()))?,
            candles
                .select_candle_field("low")
                .map_err(|e| AdxrError::CandleFieldError(e.to_string()))?,
            candles
                .select_candle_field("close")
                .map_err(|e| AdxrError::CandleFieldError(e.to_string()))?,
        ),
        AdxrData::Slices { high, low, close } => (*high, *low, *close),
    };

    let len = close.len();
    if len == 0 {
        return Err(AdxrError::EmptyInputData);
    }
    if high.len() != len || low.len() != len {
        return Err(AdxrError::HlcLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: len,
        });
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AdxrError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(AdxrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    // Need at least period + 1 bars of valid data to compute anything
    // (The actual warmup is longer but we handle that with NaN values)
    if len - first < period + 1 {
        return Err(AdxrError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((high, low, close, period, first, chosen))
}

/// Write ADXR values directly to output slice - no allocations
#[inline]
pub fn adxr_into_slice(dst: &mut [f64], input: &AdxrInput, kern: Kernel) -> Result<(), AdxrError> {
    let (high, low, close, period, first, chosen) = adxr_prepare(input, kern)?;

    let len = close.len();
    if dst.len() != len {
        return Err(AdxrError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    // Compute directly into dst
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                adxr_scalar(high, low, close, period, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => adxr_avx2(high, low, close, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                adxr_avx512(high, low, close, period, first, dst)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                adxr_scalar(high, low, close, period, first, dst)
            }
            _ => unreachable!(),
        }
    }

    // Fill warmup with NaN
    // ADXR needs warmup until first + 2*period
    let warmup_end = (first + 2 * period).min(dst.len());
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

/// ADXR into (no allocations): writes results into a caller-provided buffer.
///
/// - Preserves NaN warmups exactly as the Vec-returning API (`adxr`) does.
/// - The output slice length must equal the input length; otherwise returns
///   `AdxrError::OutputLengthMismatch`.
/// - Uses `Kernel::Auto` for runtime kernel selection.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn adxr_into(input: &AdxrInput, out: &mut [f64]) -> Result<(), AdxrError> {
    adxr_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn adxr_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    let p = period as f64;
    let rp = 1.0 / p;
    let om = 1.0 - rp;
    let pm1 = p - 1.0;
    let warmup_start = first + 2 * period;

    // Wilder bootstrap sums over the first `period` deltas.
    let mut atr_sum = 0.0;
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;

    // ---- Bootstrap: accumulate TR/+DM/-DM over [first+1 .. first+period]
    // Using safe indexing to keep scalar path safe.
    let stop = (first + period).min(len.saturating_sub(1));
    for i in (first + 1)..=stop {
        let prev_close = close[i - 1];
        let ch = high[i];
        let cl = low[i];
        let ph = high[i - 1];
        let pl = low[i - 1];

        // True range = max( high-low, |high-prev_close|, |low-prev_close| )
        let a = ch - cl;
        let b = (ch - prev_close).abs();
        let c = (cl - prev_close).abs();
        let tr = a.max(b).max(c);
        atr_sum += tr;

        let up = ch - ph;
        let down = pl - cl;
        if up > down && up > 0.0 {
            plus_dm_sum += up;
        }
        if down > up && down > 0.0 {
            minus_dm_sum += down;
        }
    }

    // First DX from bootstrap (ATR cancels): DX0 = 100 * |PDM - MDM| / (PDM + MDM)
    let denom0 = plus_dm_sum + minus_dm_sum;
    let initial_dx = if denom0 > 0.0 {
        100.0 * (plus_dm_sum - minus_dm_sum).abs() / denom0
    } else {
        0.0
    };

    // Running state for Wilder smoothing.
    let mut atr = atr_sum;
    let mut pdm_s = plus_dm_sum;
    let mut mdm_s = minus_dm_sum;

    // Build initial ADX as the average of the first `period` DX values
    let mut dx_sum = initial_dx;
    let mut dx_count: usize = 1;
    let mut adx_last = f64::NAN;
    let mut have_adx = false;

    // Keep only the last `period` ADX values to form ADXR without a full-sized buffer.
    let mut adx_ring = vec![f64::NAN; period];
    let mut head = 0usize;

    // Main pass over bars [first+period+1 .. len)
    let mut i = first + period + 1;
    while i < len {
        let prev_close = close[i - 1];
        let ch = high[i];
        let cl = low[i];
        let ph = high[i - 1];
        let pl = low[i - 1];

        // TR and directional moves
        let a = ch - cl;
        let b = (ch - prev_close).abs();
        let c = (cl - prev_close).abs();
        let tr = a.max(b).max(c);

        let up = ch - ph;
        let down = pl - cl;
        let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
        let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };

        // Wilder smoothing via (p-1)/p * prev + current
        atr = atr.mul_add(om, tr);
        pdm_s = pdm_s.mul_add(om, plus_dm);
        mdm_s = mdm_s.mul_add(om, minus_dm);

        // DX with one division (ATR cancels): 100 * |PDMs - MDMs| / (PDMs + MDMs)
        let denom = pdm_s + mdm_s;
        let dx = if denom > 0.0 {
            100.0 * (pdm_s - mdm_s).abs() / denom
        } else {
            0.0
        };

        // Build/extend ADX
        if dx_count < period {
            dx_sum += dx;
            dx_count += 1;

            if i >= warmup_start {
                out[i] = f64::NAN;
            }

            if dx_count == period {
                // First ADX (simple average of the first `period` DX values)
                adx_last = dx_sum * rp;
                have_adx = true;

                // Push into ring; ADXR requires the ADX from `period` bars ago
                let prev_adx = adx_ring[head];
                adx_ring[head] = adx_last;
                head += 1;
                if head == period {
                    head = 0;
                }

                if i >= warmup_start {
                    let v = if prev_adx.is_finite() {
                        0.5 * (adx_last + prev_adx)
                    } else {
                        f64::NAN
                    };
                    out[i] = v;
                }
            }
        } else if have_adx {
            // Wilder-smoothed ADX
            let adx_curr = (adx_last * pm1 + dx) * rp;
            adx_last = adx_curr;

            let prev_adx = adx_ring[head];
            adx_ring[head] = adx_curr;
            head += 1;
            if head == period {
                head = 0;
            }

            if i >= warmup_start {
                let v = if prev_adx.is_finite() {
                    0.5 * (adx_curr + prev_adx)
                } else {
                    f64::NAN
                };
                out[i] = v;
            }
        }

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adxr_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe {
        if period <= 32 {
            adxr_avx512_short(high, low, close, period, first, out)
        } else {
            adxr_avx512_long(high, low, close, period, first, out)
        }
    }
}

#[inline]
pub fn adxr_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // Use an unchecked, fused variant to remove bounds checks while
    // keeping the scalar algorithm. This leverages FMA via mul_add.
    unsafe { adxr_scalar_unchecked(high, low, close, period, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adxr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adxr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

/// Unchecked single-series ADXR kernel used by AVX2 wrapper.
/// Identical math to `adxr_scalar` but uses unchecked indexing to
/// avoid bounds checks in the hot loops. The caller must guarantee
/// valid slice lengths and indices.
#[inline]
unsafe fn adxr_scalar_unchecked(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    let p = period as f64;
    let rp = 1.0 / p;
    let om = 1.0 - rp;
    let pm1 = p - 1.0;
    let warmup_start = first + 2 * period;

    // Bootstrap sums over first window
    let mut atr_sum = 0.0;
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;

    let mut i = first + 1;
    let stop = core::cmp::min(first + period, len - 1);
    while i <= stop {
        let prev_close = *close.get_unchecked(i - 1);
        let ch = *high.get_unchecked(i);
        let cl = *low.get_unchecked(i);
        let ph = *high.get_unchecked(i - 1);
        let pl = *low.get_unchecked(i - 1);

        let a = ch - cl;
        let b = (ch - prev_close).abs();
        let c = (cl - prev_close).abs();
        let tr = a.max(b).max(c);
        atr_sum += tr;

        let up = ch - ph;
        let down = pl - cl;
        if up > down && up > 0.0 {
            plus_dm_sum += up;
        }
        if down > up && down > 0.0 {
            minus_dm_sum += down;
        }
        i += 1;
    }

    // Initial DX from bootstrap sums (ATR cancels)
    let denom0 = plus_dm_sum + minus_dm_sum;
    let initial_dx = if denom0 > 0.0 {
        100.0 * (plus_dm_sum - minus_dm_sum).abs() / denom0
    } else {
        0.0
    };

    // Running Wilder state
    let mut atr = atr_sum;
    let mut pdm_s = plus_dm_sum;
    let mut mdm_s = minus_dm_sum;

    // ADX accumulation
    let mut dx_sum = initial_dx;
    let mut dx_count: usize = 1;
    let mut adx_last = f64::NAN;
    let mut have_adx = false;

    // Ring buffer to compute ADXR without O(N) temp
    let mut adx_ring = vec![f64::NAN; period];
    let mut head = 0usize;

    i = first + period + 1;
    while i < len {
        let prev_close = *close.get_unchecked(i - 1);
        let ch = *high.get_unchecked(i);
        let cl = *low.get_unchecked(i);
        let ph = *high.get_unchecked(i - 1);
        let pl = *low.get_unchecked(i - 1);

        let a = ch - cl;
        let b = (ch - prev_close).abs();
        let c = (cl - prev_close).abs();
        let tr = a.max(b).max(c);

        let up = ch - ph;
        let down = pl - cl;
        let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
        let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };

        atr = atr.mul_add(om, tr);
        pdm_s = pdm_s.mul_add(om, plus_dm);
        mdm_s = mdm_s.mul_add(om, minus_dm);

        // DX with one division (ATR cancels)
        let denom = pdm_s + mdm_s;
        let dx = if denom > 0.0 {
            100.0 * (pdm_s - mdm_s).abs() / denom
        } else {
            0.0
        };

        if dx_count < period {
            dx_sum += dx;
            dx_count += 1;

            if i >= warmup_start {
                *out.get_unchecked_mut(i) = f64::NAN;
            }

            if dx_count == period {
                adx_last = dx_sum * rp;
                have_adx = true;

                let prev_adx = *adx_ring.get_unchecked(head);
                *adx_ring.get_unchecked_mut(head) = adx_last;
                head += 1;
                if head == period {
                    head = 0;
                }

                if i >= warmup_start {
                    let v = if prev_adx.is_finite() {
                        0.5 * (adx_last + prev_adx)
                    } else {
                        f64::NAN
                    };
                    *out.get_unchecked_mut(i) = v;
                }
            }
        } else if have_adx {
            let adx_curr = (adx_last * pm1 + dx) * rp;
            adx_last = adx_curr;

            let prev_adx = *adx_ring.get_unchecked(head);
            *adx_ring.get_unchecked_mut(head) = adx_curr;
            head += 1;
            if head == period {
                head = 0;
            }

            if i >= warmup_start {
                let v = if prev_adx.is_finite() {
                    0.5 * (adx_curr + prev_adx)
                } else {
                    f64::NAN
                };
                *out.get_unchecked_mut(i) = v;
            }
        }

        i += 1;
    }
}

#[inline(always)]
pub fn adxr_batch_with_kernel(
    h: &[f64],
    l: &[f64],
    c: &[f64],
    sweep: &AdxrBatchRange,
    k: Kernel,
) -> Result<AdxrBatchOutput, AdxrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AdxrError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    // allocate once with helpers, then reuse into
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = c.len();
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|p| {
            let first = c.iter().position(|x| !x.is_nan()).unwrap_or(0);
            // ADXR warmup: first + 2 * period
            first + 2 * p.period.unwrap()
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // compute into the same allocation
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let combos = adxr_batch_inner_into(h, l, c, sweep, simd, true, out)?;
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(AdxrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[derive(Clone, Debug)]
pub struct AdxrBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for AdxrBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdxrBatchBuilder {
    range: AdxrBatchRange,
    kernel: Kernel,
}
impl AdxrBatchBuilder {
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
        h: &[f64],
        l: &[f64],
        c: &[f64],
    ) -> Result<AdxrBatchOutput, AdxrError> {
        adxr_batch_with_kernel(h, l, c, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<AdxrBatchOutput, AdxrError> {
        let h = &candles.high;
        let l = &candles.low;
        let c = &candles.close;
        self.apply_slices(h, l, c)
    }
}

#[derive(Clone, Debug)]
pub struct AdxrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdxrParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AdxrBatchOutput {
    pub fn row_for_params(&self, p: &AdxrParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &AdxrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AdxrBatchRange) -> Result<Vec<AdxrParams>, AdxrError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Option<Vec<usize>> {
        // Zero step or equal bounds => static
        if step == 0 || start == end {
            return Some(vec![start]);
        }
        if start < end {
            return Some((start..=end).step_by(step).collect());
        }
        // Reversed bounds: walk down using `step`
        if step == 0 { return Some(vec![start]); }
        let mut v = Vec::new();
        let mut cur = start;
        while cur >= end {
            v.push(cur);
            if let Some(next) = cur.checked_sub(step) { cur = next; } else { break; }
            if cur == usize::MAX { break; }
            if cur < end { break; }
        }
        Some(v)
    }
    let periods = axis(r.period).unwrap_or_default();
    if periods.is_empty() {
        return Err(AdxrError::InvalidRange {
            start: r.period.0,
            end: r.period.1,
            step: r.period.2,
        });
    }
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(AdxrParams { period: Some(p) });
    }
    Ok(out)
}

// Shared precomputation reused by all rows in batch ADXR.
#[inline]
fn shared_precompute_tr_dm(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut tr_all = vec![0.0; len];
    let mut pdm_all = vec![0.0; len];
    let mut mdm_all = vec![0.0; len];

    // Compute TR, +DM, -DM for all i >= first+1
    for i in (first + 1)..len {
        let prev_close = close[i - 1];
        let ch = high[i];
        let cl = low[i];
        let ph = high[i - 1];
        let pl = low[i - 1];

        let a = ch - cl;
        let b = (ch - prev_close).abs();
        let c = (cl - prev_close).abs();
        tr_all[i] = a.max(b).max(c);

        let up = ch - ph;
        let down = pl - cl;
        pdm_all[i] = if up > down && up > 0.0 { up } else { 0.0 };
        mdm_all[i] = if down > up && down > 0.0 { down } else { 0.0 };
    }

    // Build prefix sums starting at start = first+1 so that prefix[k] = sum of first k elements
    let start = first + 1;
    let pre_len = len.saturating_sub(start);
    let mut prefix_tr = vec![0.0; pre_len + 1];
    let mut prefix_pdm = vec![0.0; pre_len + 1];
    let mut prefix_mdm = vec![0.0; pre_len + 1];
    for k in 1..=pre_len {
        let i = start + (k - 1);
        prefix_tr[k] = prefix_tr[k - 1] + tr_all[i];
        prefix_pdm[k] = prefix_pdm[k - 1] + pdm_all[i];
        prefix_mdm[k] = prefix_mdm[k - 1] + mdm_all[i];
    }

    (tr_all, pdm_all, mdm_all, prefix_tr, prefix_pdm, prefix_mdm)
}

#[inline]
fn adxr_row_from_precomputed(
    tr_all: &[f64],
    pdm_all: &[f64],
    mdm_all: &[f64],
    prefix_tr: &[f64],
    prefix_pdm: &[f64],
    prefix_mdm: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = tr_all.len();
    if len == 0 {
        return;
    }

    let p = period as f64;
    let rp = 1.0 / p;
    let om = 1.0 - rp;
    let pm1 = p - 1.0;
    let warmup_start = first + 2 * period;

    // Bootstrap sums via prefix sums
    let atr0 = prefix_tr.get(period).copied().unwrap_or(0.0);
    let pdm0 = prefix_pdm.get(period).copied().unwrap_or(0.0);
    let mdm0 = prefix_mdm.get(period).copied().unwrap_or(0.0);

    // Initial DX using one-division form (ATR cancels)
    let denom0 = pdm0 + mdm0;
    let initial_dx = if denom0 > 0.0 {
        100.0 * (pdm0 - mdm0).abs() / denom0
    } else {
        0.0
    };

    let mut atr = atr0;
    let mut pdm_s = pdm0;
    let mut mdm_s = mdm0;

    let mut dx_sum = initial_dx;
    let mut dx_count: usize = 1;
    let mut adx_last = f64::NAN;
    let mut have_adx = false;
    let mut adx_ring = vec![f64::NAN; period];
    let mut head = 0usize;

    let mut i = first + period + 1;
    while i < len {
        let tr = tr_all[i];
        let plus_dm = pdm_all[i];
        let minus_dm = mdm_all[i];

        atr = atr.mul_add(om, tr);
        pdm_s = pdm_s.mul_add(om, plus_dm);
        mdm_s = mdm_s.mul_add(om, minus_dm);

        // DX with one division (ATR cancels)
        let denom = pdm_s + mdm_s;
        let dx = if denom > 0.0 {
            100.0 * (pdm_s - mdm_s).abs() / denom
        } else {
            0.0
        };

        if dx_count < period {
            dx_sum += dx;
            dx_count += 1;
            if i >= warmup_start {
                out[i] = f64::NAN;
            }
            if dx_count == period {
                adx_last = dx_sum * rp;
                have_adx = true;

                let prev_adx = adx_ring[head];
                adx_ring[head] = adx_last;
                head += 1;
                if head == period {
                    head = 0;
                }

                if i >= warmup_start {
                    let v = if prev_adx.is_finite() {
                        0.5 * (adx_last + prev_adx)
                    } else {
                        f64::NAN
                    };
                    out[i] = v;
                }
            }
        } else if have_adx {
            let adx_curr = (adx_last * pm1 + dx) * rp;
            adx_last = adx_curr;

            let prev_adx = adx_ring[head];
            adx_ring[head] = adx_curr;
            head += 1;
            if head == period {
                head = 0;
            }

            if i >= warmup_start {
                let v = if prev_adx.is_finite() {
                    0.5 * (adx_curr + prev_adx)
                } else {
                    f64::NAN
                };
                out[i] = v;
            }
        }

        i += 1;
    }
}

#[inline(always)]
pub fn adxr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    kern: Kernel,
) -> Result<AdxrBatchOutput, AdxrError> {
    adxr_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn adxr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    kern: Kernel,
) -> Result<AdxrBatchOutput, AdxrError> {
    adxr_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn adxr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AdxrBatchOutput, AdxrError> {
    let combos = expand_grid(sweep)?;
    let len = close.len();
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AdxrError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    // Need at least period + 1 bars of valid data to compute anything
    if len - first < max_p + 1 {
        return Err(AdxrError::NotEnoughValidData {
            needed: max_p + 1,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    // Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            // ADXR warmup: first + 2 * period
            // (ADX appears at first + 2*period, then ADXR averages current and period-ago ADX)
            first + 2 * c.period.unwrap()
        })
        .collect();

    // Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to mutable slice for computation
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        std::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Shared precompute of TR, +DM, -DM and their prefix sums once per dataset
    let (tr_all, pdm_all, mdm_all, prefix_tr, prefix_pdm, prefix_mdm) =
        shared_precompute_tr_dm(high, low, close, first);

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        match kern {
            // Delegate to a fused row kernel that reuses shared precompute
            Kernel::Scalar => adxr_row_from_precomputed(
                &tr_all,
                &pdm_all,
                &mdm_all,
                &prefix_tr,
                &prefix_pdm,
                &prefix_mdm,
                first,
                period,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx512 => adxr_row_from_precomputed(
                &tr_all,
                &pdm_all,
                &mdm_all,
                &prefix_tr,
                &prefix_pdm,
                &prefix_mdm,
                first,
                period,
                out_row,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => adxr_row_from_precomputed(
                &tr_all,
                &pdm_all,
                &mdm_all,
                &prefix_tr,
                &prefix_pdm,
                &prefix_mdm,
                first,
                period,
                out_row,
            ),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(AdxrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn adxr_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AdxrParams>, AdxrError> {
    let combos = expand_grid(sweep)?;

    let len = close.len();
    if high.len() != len || low.len() != len {
        return Err(AdxrError::HlcLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: len,
        });
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AdxrError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    // Need at least period + 1 bars of valid data to compute anything
    if len - first < max_p + 1 {
        return Err(AdxrError::NotEnoughValidData {
            needed: max_p + 1,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    if let Some(expected) = rows.checked_mul(cols) {
        if out.len() != expected {
            return Err(AdxrError::OutputLengthMismatch { expected, got: out.len() });
        }
    } else {
        return Err(AdxrError::InvalidRange { start: rows, end: cols, step: 0 });
    }

    // Warmup = first + 2*period per row
    // (ADX appears at first + 2*period, then ADXR averages current and period-ago ADX)
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + 2 * c.period.unwrap())
        .collect();

    // Cast the Python/WASM buffer to MaybeUninit and initialize warm prefixes with helpers
    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    // Shared precompute of TR, +DM, -DM and prefix sums once per dataset
    let (tr_all, pdm_all, mdm_all, prefix_tr, prefix_pdm, prefix_mdm) =
        shared_precompute_tr_dm(high, low, close, first);

    // Row executor
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let dst = std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => adxr_row_from_precomputed(
                &tr_all,
                &pdm_all,
                &mdm_all,
                &prefix_tr,
                &prefix_pdm,
                &prefix_mdm,
                first,
                period,
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx512 => adxr_row_from_precomputed(
                &tr_all,
                &pdm_all,
                &mdm_all,
                &prefix_tr,
                &prefix_pdm,
                &prefix_mdm,
                first,
                period,
                dst,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => adxr_row_from_precomputed(
                &tr_all,
                &pdm_all,
                &mdm_all,
                &prefix_tr,
                &prefix_pdm,
                &prefix_mdm,
                first,
                period,
                dst,
            ),
            _ => unreachable!("pass non-batch kernel"),
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
unsafe fn adxr_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_avx512(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

/// Decision: Streaming ring fixed; constants precomputed; DX uses one-division form.
#[derive(Debug, Clone)]
pub struct AdxrStream {
    period: usize,
    // Precomputed constants for Wilder smoothing / ADX update
    rp: f64,  // 1/p
    om: f64,  // 1 - 1/p
    pm1: f64, // p - 1

    // Wilder smoothing state (running SMMA of TR, +DM, -DM)
    atr: f64,
    pdm_s: f64,
    mdm_s: f64,

    // Bootstrap DX accumulation to build first ADX
    dx_sum: f64,
    dx_count: usize,
    adx_last: f64,
    have_adx: bool,

    // Ring buffer of last `period` ADX values (to fetch ADX from `period` bars ago)
    adx_ring: Vec<f64>,
    head: usize, // next write index (element at `head` is the oldest)

    // Previous bar values to form TR/+DM/-DM deltas
    prev_hlc: Option<(f64, f64, f64)>,

    // Number of deltas processed since we first had a previous bar
    seen: usize,
}

impl AdxrStream {
    #[inline(always)]
    pub fn try_new(params: AdxrParams) -> Result<Self, AdxrError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(AdxrError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let p = period as f64;
        Ok(Self {
            period,
            rp: 1.0 / p,
            om: 1.0 - 1.0 / p,
            pm1: p - 1.0,
            atr: 0.0,
            pdm_s: 0.0,
            mdm_s: 0.0,
            dx_sum: 0.0,
            dx_count: 0,
            adx_last: f64::NAN,
            have_adx: false,
            adx_ring: vec![f64::NAN; period],
            head: 0,
            prev_hlc: None,
            seen: 0,
        })
    }

    /// O(1) per-bar update. Returns Some(ADXR) after warmup, otherwise None.
    ///
    /// Warmup timeline (assuming no NaNs and contiguous bars):
    /// 1) Accumulate TR/+DM/-DM for `period` bars to form DX₀ from sums.
    /// 2) Accumulate `period` DX values to form first ADX.
    /// 3) After an additional `period` ADX updates the ring is full and ADXR becomes available.
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        // Ignore non-finite updates (keeps state unchanged)
        if !(high.is_finite() && low.is_finite() && close.is_finite()) {
            return None;
        }

        // First observation just primes the previous values
        if self.prev_hlc.is_none() {
            self.prev_hlc = Some((high, low, close));
            return None;
        }

        let (ph, pl, pc) = unsafe { self.prev_hlc.unwrap_unchecked() };
        self.prev_hlc = Some((high, low, close));
        self.seen = self.seen.wrapping_add(1);

        // True Range (Wilder): max(high-low, |high-prev_close|, |low-prev_close|)
        let tr = {
            let a = high - low;
            let b = (high - pc).abs();
            let c = (low - pc).abs();
            a.max(b).max(c)
        };

        // Directional Movement
        let up = high - ph;
        let down = pl - low;
        let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
        let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };

        // --- Bootstrap over the first `period` deltas
        if self.seen <= self.period {
            self.atr += tr;
            self.pdm_s += plus_dm;
            self.mdm_s += minus_dm;

            if self.seen == self.period {
                // First DX from bootstrap sums.
                let denom = self.pdm_s + self.mdm_s;
                let dx0 = if denom > 0.0 {
                    100.0 * (self.pdm_s - self.mdm_s).abs() / denom
                } else {
                    0.0
                };
                self.dx_sum = dx0;
                self.dx_count = 1;
            }
            return None;
        }

        // Wilder smoothing (equivalent to EMA with alpha = 1/period)
        self.atr = self.atr.mul_add(self.om, tr);
        self.pdm_s = self.pdm_s.mul_add(self.om, plus_dm);
        self.mdm_s = self.mdm_s.mul_add(self.om, minus_dm);

        // Compute DX from smoothed directional movement (ATR cancels out)
        let denom = self.pdm_s + self.mdm_s;
        let dx = if denom > 0.0 {
            100.0 * (self.pdm_s - self.mdm_s).abs() / denom
        } else {
            0.0
        };

        // Build first ADX as the average of the first `period` DX values
        if !self.have_adx {
            if self.dx_count + 1 < self.period {
                self.dx_sum += dx;
                self.dx_count += 1;
                return None;
            } else {
                // Now dx_count + 1 == period
                self.dx_sum += dx;
                self.adx_last = self.dx_sum * self.rp;
                self.have_adx = true;

                // Seed ring with the first ADX; no ADXR until a full ring cycle
                self.adx_ring[self.head] = self.adx_last;
                self.head = (self.head + 1) % self.period;
                return None;
            }
        }

        // Subsequent ADX: Wilder-smoothed ADX_t = ((p-1)*ADX_{t-1} + DX_t) / p
        let adx_curr = (self.adx_last.mul_add(self.pm1, dx)) * self.rp;
        self.adx_last = adx_curr;

        // Correct ring logic: read oldest before overwrite
        let adx_period_ago = self.adx_ring[self.head];
        self.adx_ring[self.head] = adx_curr;
        self.head = (self.head + 1) % self.period;

        if adx_period_ago.is_finite() {
            Some(0.5 * (adx_curr + adx_period_ago))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_adxr_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdxrInput::from_candles(&candles, AdxrParams { period: None });
        let output = adxr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adxr_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdxrInput::from_candles(&candles, AdxrParams::default());
        let result = adxr_with_kernel(&input, kernel)?;
        let expected = [37.10, 37.3, 37.0, 36.2, 36.3];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] ADXR {:?} mismatch at idx {}: got {}, expected {}",
                test,
                kernel,
                i,
                val,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_adxr_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 20.0, 30.0];
        let low = [9.0, 19.0, 29.0];
        let close = [9.5, 19.5, 29.5];
        let input = AdxrInput::from_slices(&high, &low, &close, AdxrParams { period: Some(0) });
        let res = adxr_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ADXR should fail with zero period", test);
        Ok(())
    }

    fn check_adxr_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 20.0];
        let low = [9.0, 19.0];
        let close = [9.5, 19.5];
        let input = AdxrInput::from_slices(&high, &low, &close, AdxrParams { period: Some(10) });
        let res = adxr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ADXR should fail with period > data.len()",
            test
        );
        Ok(())
    }

    fn check_adxr_very_small_dataset(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [100.0];
        let low = [99.0];
        let close = [99.5];
        let input = AdxrInput::from_slices(&high, &low, &close, AdxrParams { period: Some(14) });
        let res = adxr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ADXR should fail with insufficient data",
            test
        );
        Ok(())
    }

    fn check_adxr_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = AdxrInput::from_candles(&candles, AdxrParams { period: Some(14) });
        let first_result = adxr_with_kernel(&first_input, kernel)?;
        let high = &candles.high;
        let low = &candles.low;
        let close = &candles.close;
        let second_input = AdxrInput::from_slices(high, low, close, AdxrParams { period: Some(5) });
        let second_result = adxr_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adxr_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdxrInput::from_candles(&candles, AdxrParams { period: Some(14) });
        let res = adxr_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test,
                    240 + i
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_adxr_tests {
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

    #[cfg(debug_assertions)]
    fn check_adxr_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            AdxrParams::default(),
            AdxrParams { period: Some(5) },
            AdxrParams { period: Some(10) },
            AdxrParams { period: Some(14) },
            AdxrParams { period: Some(20) },
            AdxrParams { period: Some(25) },
            AdxrParams { period: Some(30) },
            AdxrParams { period: Some(50) },
            AdxrParams { period: Some(100) },
            AdxrParams { period: Some(2) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = AdxrInput::from_candles(&candles, params.clone());
            let output = adxr_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_adxr_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn check_adxr_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test scenarios with various market conditions
        let strat = (2usize..=50)
            .prop_flat_map(|period| {
                // Generate data size between period+10 and 400
                let min_size = (period * 3).max(period + 10);
                let max_size = 400;
                (
                    // Base price
                    10.0f64..1000.0f64,
                    // Volatility percentage (0% to 10%)
                    0.0f64..0.1f64,
                    // Trend strength (-0.01 to 0.01 per bar)
                    -0.01f64..0.01f64,
                    // Data size
                    min_size..max_size,
                    // Period
                    Just(period),
                    // Market type: 0=ranging, 1=trending, 2=zero volatility
                    0u8..3,
                )
            })
            .prop_map(
                |(base_price, volatility_pct, trend, size, period, market_type)| {
                    let mut high_data = Vec::with_capacity(size);
                    let mut low_data = Vec::with_capacity(size);
                    let mut close_data = Vec::with_capacity(size);

                    for i in 0..size {
                        let price = match market_type {
                            0 => {
                                // Ranging market with variable volatility
                                let cycle = (i as f64 * 0.1).sin();
                                base_price * (1.0 + cycle * volatility_pct)
                            }
                            1 => {
                                // Trending market
                                base_price * (1.0 + trend * i as f64)
                            }
                            2 => {
                                // Zero/minimal volatility - constant price
                                base_price
                            }
                            _ => base_price,
                        };

                        // Generate high/low based on market type
                        let (high, low, close) = if market_type == 2 {
                            // Zero volatility case - high = low = close
                            (price, price, price)
                        } else {
                            // Variable volatility
                            let daily_volatility =
                                price * volatility_pct * (0.5 + 0.5 * (i as f64 * 0.05).cos());
                            let close = price;
                            let high = close + daily_volatility.abs();
                            let low = close - daily_volatility.abs();
                            (high, low, close)
                        };

                        high_data.push(high);
                        low_data.push(low);
                        close_data.push(close);
                    }

                    (high_data, low_data, close_data, period, market_type)
                },
            );

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(high_data, low_data, close_data, period, market_type)| {
                let params = AdxrParams {
                    period: Some(period),
                };
                let input = AdxrInput::from_slices(&high_data, &low_data, &close_data, params);

                // Test with the specified kernel
                let result = adxr_with_kernel(&input, kernel);
                prop_assert!(result.is_ok(), "ADXR computation failed: {:?}", result);
                let AdxrOutput { values: out } = result.unwrap();

                // Test with scalar kernel as reference
                let ref_result = adxr_with_kernel(&input, Kernel::Scalar);
                prop_assert!(ref_result.is_ok(), "Reference ADXR computation failed");
                let AdxrOutput { values: ref_out } = ref_result.unwrap();

                // Find first non-NaN index in data
                let first = close_data.iter().position(|x| !x.is_nan()).unwrap_or(0);

                // ADXR warmup period is first + 2 * period
                let warmup_period = first + 2 * period;

                // Test properties
                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if i < warmup_period {
                        // During warmup, values should be NaN
                        prop_assert!(
                            y.is_nan(),
                            "Expected NaN during warmup at index {}, got {}",
                            i,
                            y
                        );
                    } else {
                        // After warmup, values should be valid

                        // ADXR values should be in [0, 100] range (since it's based on ADX)
                        if !y.is_nan() {
                            prop_assert!(
                                y >= -1e-9 && y <= 100.0 + 1e-9,
                                "ADXR value {} at index {} is outside [0, 100] range",
                                y,
                                i
                            );
                        }

                        // Different kernels should produce very similar results
                        if !y.is_nan() && !r.is_nan() {
                            let diff = (y - r).abs();
                            prop_assert!(
                                diff < 1e-6,
                                "Kernel {:?} and Scalar differ by {} at index {}: {} vs {}",
                                kernel,
                                diff,
                                i,
                                y,
                                r
                            );
                        }
                    }
                }

                // Market-specific validations
                if market_type == 2 && out.len() > warmup_period + period {
                    // Zero volatility: ADXR should converge toward 0
                    let last_values = &out[out.len().saturating_sub(10)..];
                    let non_nan_values: Vec<f64> = last_values
                        .iter()
                        .filter(|v| !v.is_nan())
                        .copied()
                        .collect();

                    if !non_nan_values.is_empty() {
                        let avg_last =
                            non_nan_values.iter().sum::<f64>() / non_nan_values.len() as f64;
                        prop_assert!(
                            avg_last < 25.0,
                            "ADXR should be low with zero volatility, got average {}",
                            avg_last
                        );
                    }
                }

                // Special case: period = 2 (minimum valid)
                if period == 2 {
                    // Just verify it doesn't crash and produces valid output
                    prop_assert!(out.len() == close_data.len());
                }

                // Constant data test
                let is_constant = close_data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && high_data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && low_data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

                if is_constant && out.len() > warmup_period {
                    // With constant prices, ADXR should stabilize
                    let stable_values = &out[warmup_period..];
                    let non_nan: Vec<f64> = stable_values
                        .iter()
                        .filter(|v| !v.is_nan())
                        .copied()
                        .collect();

                    if non_nan.len() > 10 {
                        // Check that values are stabilizing (low standard deviation)
                        let mean = non_nan.iter().sum::<f64>() / non_nan.len() as f64;
                        let variance = non_nan.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / non_nan.len() as f64;
                        let std_dev = variance.sqrt();

                        prop_assert!(
                            std_dev < 5.0,
                            "ADXR should stabilize with constant data, std_dev = {}",
                            std_dev
                        );
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    generate_all_adxr_tests!(
        check_adxr_partial_params,
        check_adxr_accuracy,
        check_adxr_zero_period,
        check_adxr_period_exceeds_length,
        check_adxr_very_small_dataset,
        check_adxr_reinput,
        check_adxr_nan_handling,
        check_adxr_no_poison,
        check_adxr_property
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AdxrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = AdxrParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2), // period_start, period_end, period_step
            (5, 25, 5),
            (10, 20, 2),
            (14, 50, 6),
            (20, 100, 10),
            (2, 30, 7),
            (8, 40, 8),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = AdxrBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

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
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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
    fn test_adxr_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Small but non-trivial synthetic HLC series
        let len = 256usize;
        let mut high = vec![0.0f64; len];
        let mut low = vec![0.0f64; len];
        let mut close = vec![0.0f64; len];
        for i in 0..len {
            // Trend + oscillation; ensure high >= close >= low
            let base = 100.0 + (i as f64) * 0.1 + (i as f64 * 0.07).sin();
            low[i] = base - 1.0;
            close[i] = base - 0.3;
            high[i] = base + 0.8;
        }

        let input = AdxrInput::from_slices(&high, &low, &close, AdxrParams::default());

        // Baseline via Vec-returning API
        let baseline = adxr(&input)?.values;

        // Preallocate output and compute via into API
        let mut out = vec![0.0f64; len];
        #[cfg(not(feature = "wasm"))]
        {
            adxr_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, call the slice version directly for parity check
            adxr_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), out.len());
        for (a, b) in baseline.iter().zip(out.iter()) {
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(equal, "Mismatch: baseline={} out={}", a, b);
        }

        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "adxr")]
#[pyo3(signature = (high, low, close, period=None, kernel=None))]
pub fn adxr_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    // Validate input lengths
    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err(format!(
            "HLC data length mismatch: high={}, low={}, close={}",
            high_slice.len(),
            low_slice.len(),
            close_slice.len()
        )));
    }

    // Validate kernel before entering allow_threads
    let kern = validate_kernel(kernel, false)?;

    let params = AdxrParams {
        period: period.or(Some(14)),
    };
    let adxr_in = AdxrInput::from_slices(high_slice, low_slice, close_slice, params);

    // Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| adxr_with_kernel(&adxr_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdxrStream")]
pub struct AdxrStreamPy {
    stream: AdxrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdxrStreamPy {
    #[new]
    #[pyo3(signature = (period=None))]
    fn new(period: Option<usize>) -> PyResult<Self> {
        let params = AdxrParams {
            period: period.or(Some(14)),
        };
        let stream =
            AdxrStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AdxrStreamPy { stream })
    }

    /// Updates the stream with new high, low, close values and returns the calculated ADXR value.
    /// Returns `None` if the buffer is not yet full.
    ///
    /// Note: ADXR streaming requires maintaining full ADX history,
    /// so this implementation is simplified and may not produce exact values.
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "adxr_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
pub fn adxr_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;

    if h.len() != l.len() || h.len() != c.len() {
        return Err(PyValueError::new_err(format!(
            "HLC data length mismatch: high={}, low={}, close={}",
            h.len(),
            l.len(),
            c.len()
        )));
    }

    let sweep = AdxrBatchRange { period: period_range };
    let combos_probe = expand_grid(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos_probe.len();
    let cols = c.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    // Pre-allocate NumPy array and fill it in-place without extra copies.
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    // Resolve to non-batch kernel before releasing the GIL.
    let k = crate::utilities::kernel_validation::validate_kernel(kernel, true)?;
    let simd = match k {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        },
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        other => other, // allow explicit non-batch for tests
    };

    let combos = py
        .allow_threads(|| adxr_batch_inner_into(h, l, c, &sweep, simd, true, out_slice))
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

// ==================== PYTHON CUDA BINDINGS ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "adxr_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, period_range, device_id=0))]
pub fn adxr_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<AdxrDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let c = close_f32.as_slice()?;
    let sweep = AdxrBatchRange {
        period: period_range,
    };
    let (inner, ctx_arc, dev_id) = py.allow_threads(|| {
        let cuda = CudaAdxr::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (dev, _combos) = cuda
            .adxr_batch_dev(h, l, c, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((dev, cuda.context_arc_clone(), cuda.device_id()))
    })?;
    Ok(AdxrDeviceArrayF32Py { inner: Some(inner), _ctx: ctx_arc, device_id: dev_id })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "adxr_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, period, device_id=0))]
pub fn adxr_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    close_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<AdxrDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape = high_tm_f32.shape();
    if shape.len() != 2 || low_tm_f32.shape() != shape || close_tm_f32.shape() != shape {
        return Err(PyValueError::new_err("expected three matching 2D arrays"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let c = close_tm_f32.as_slice()?;
    let (inner, ctx_arc, dev_id) = py.allow_threads(|| {
        let cuda = CudaAdxr::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .adxr_many_series_one_param_time_major_dev(h, l, c, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((dev, cuda.context_arc_clone(), cuda.device_id()))
    })?;
    Ok(AdxrDeviceArrayF32Py { inner: Some(inner), _ctx: ctx_arc, device_id: dev_id })
}

// Python wrapper that keeps the CUDA context alive for returned VRAM handles
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "AdxrDeviceArrayF32", unsendable)]
pub struct AdxrDeviceArrayF32Py {
    pub(crate) inner: Option<DeviceArrayF32>,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl AdxrDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
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
        d.set_item("data", (inner.device_ptr() as usize, false))?;
        // Producing stream is synchronized before return; omit "stream" per CAI v3
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // 2 == kDLCUDA
        (2, self.device_id as i32)
    }

    // DLPack producer with v1.x negotiation and legacy fallback, delegating to the
    // shared helper in `src/utilities/dlpack_cuda.rs`. Array API stream semantics
    // are accepted but ignored here since the stream is synchronized.
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
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
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = AdxrParams {
        period: Some(period),
    };
    let input = AdxrInput::from_slices(high, low, close, params);

    let mut output = vec![0.0; close.len()];

    adxr_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AdxrBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    adxr_batch_inner(high, low, close, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AdxrBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdxrBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdxrBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdxrParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = adxr_batch)]
pub fn adxr_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: AdxrBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AdxrBatchRange {
        period: config.period_range,
    };

    // 2. Run the existing core logic
    let output = adxr_batch_inner(high, low, close, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = AdxrBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = AdxrParams {
            period: Some(period),
        };
        let input = AdxrInput::from_slices(high, low, close, params);

        // Check for aliasing - any input pointer equals output pointer
        if high_ptr == out_ptr as *const f64
            || low_ptr == out_ptr as *const f64
            || close_ptr == out_ptr as *const f64
        {
            let mut temp = vec![0.0; len];
            adxr_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            adxr_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let h = std::slice::from_raw_parts(high_ptr, len);
        let l = std::slice::from_raw_parts(low_ptr, len);
        let c = std::slice::from_raw_parts(close_ptr, len);

        let sweep = AdxrBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        // Use Scalar on wasm for portability; change to detect if you ship wasm simd kernels.
        adxr_batch_inner_into(h, l, c, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
