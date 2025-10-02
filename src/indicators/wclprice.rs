//! # Weighted Close Price (WCLPRICE)
//!
//! WCLPRICE calculates a weighted average price that gives double weight to the closing price,
//! useful for identifying price levels with heavier emphasis on closing values.
//!
//! ## Parameters
//! - None (uses high, low, and close price data)
//!
//! ## Returns
//! - **`Ok(WclpriceOutput)`** containing a `Vec<f64>` of weighted close values matching input length.
//! - **`Err(WclpriceError)`** on invalid data or mismatched input lengths.
//!
//! ## Developer Notes
//! - SIMD Status: AVX2/AVX512 kernels implemented (vectorized `(h + l + 2c)/4`).
//!   - Nightly AVX benches (100k, `-C target-cpu=native`):
//!     - scalar: ~49 µs (single-series bench group)
//!     - avx2: ~23.6 µs, avx512: ~23.3 µs (>5% vs scalar)
//!   - Stable scalar-only bench (legacy group `wclprice_bench`): ~31 µs after scalar optimizations.
//! - Scalar path optimized: branch-free NaN propagation, unrolled by 8/4, uses `mul_add`.
//! - Streaming Performance: O(1) per element; stateless.
//! - Memory: uses `alloc_with_nan_prefix` and `make_uninit_matrix` for batch.
//! - Batch: single-row only (no params); row SIMD reuses single-series kernels.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaWclprice};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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
use std::error::Error;
use thiserror::Error;

// Note: Unlike ALMA, WCLPRICE doesn't implement AsRef<[f64]> because it needs multiple slices
// This is intentional as WCLPRICE requires high, low, and close data

#[derive(Debug, Clone)]
pub enum WclpriceData<'a> {
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
pub struct WclpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct WclpriceParams;

impl Default for WclpriceParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct WclpriceInput<'a> {
    pub data: WclpriceData<'a>,
    pub params: WclpriceParams,
}

impl<'a> WclpriceInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: WclpriceData::Candles { candles },
            params: WclpriceParams::default(),
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: WclpriceData::Slices { high, low, close },
            params: WclpriceParams::default(),
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WclpriceBuilder {
    kernel: Kernel,
}
impl Default for WclpriceBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}
impl WclpriceBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<WclpriceOutput, WclpriceError> {
        let i = WclpriceInput::from_candles(candles);
        wclprice_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WclpriceOutput, WclpriceError> {
        let i = WclpriceInput::from_slices(high, low, close);
        wclprice_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn into_stream(self) -> WclpriceStream {
        WclpriceStream::default()
    }
}

#[derive(Debug, Error)]
pub enum WclpriceError {
    #[error("wclprice: empty input")]
    EmptyData,
    #[error("wclprice: all values are NaN")]
    AllValuesNaN,
    #[error("wclprice: output len {out} must equal min(high,low,close)={min_len}")]
    InvalidOutputLen { out: usize, min_len: usize },
    #[error("wclprice: missing candle field '{field}'")]
    MissingField { field: &'static str },
    #[error("wclprice: invalid kernel for batch mode: {0:?}")]
    InvalidBatchKernel(Kernel),
}

#[inline(always)]
fn wclprice_prepare<'a>(
    input: &'a WclpriceInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), WclpriceError> {
    let (high, low, close) = match &input.data {
        WclpriceData::Candles { candles } => {
            let h = candles
                .select_candle_field("high")
                .map_err(|_| WclpriceError::MissingField { field: "high" })?;
            let l = candles
                .select_candle_field("low")
                .map_err(|_| WclpriceError::MissingField { field: "low" })?;
            let c = candles
                .select_candle_field("close")
                .map_err(|_| WclpriceError::MissingField { field: "close" })?;
            (h, l, c)
        }
        WclpriceData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyData);
    }
    let lh = high.len();
    let ll = low.len();
    let lc = close.len();
    let len = lh.min(ll).min(lc);
    // Note: We compute on min(len) for compatibility, though ideally all lengths should match
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::ScalarBatch => Kernel::Scalar,
        k => k,
    };
    Ok((high, low, close, len, first, chosen))
}

#[inline]
pub fn wclprice(input: &WclpriceInput) -> Result<WclpriceOutput, WclpriceError> {
    wclprice_with_kernel(input, Kernel::Auto)
}

pub fn wclprice_with_kernel(
    input: &WclpriceInput,
    kernel: Kernel,
) -> Result<WclpriceOutput, WclpriceError> {
    let (high, low, close, len, first, chosen) = wclprice_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(len, first);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                wclprice_scalar(high, low, close, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wclprice_avx2(high, low, close, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wclprice_avx512(high, low, close, first, &mut out)
            }
            _ => wclprice_scalar(high, low, close, first, &mut out),
        }
    }
    Ok(WclpriceOutput { values: out })
}

#[inline]
pub fn wclprice_into_slice(
    dst: &mut [f64],
    input: &WclpriceInput,
    kern: Kernel,
) -> Result<(), WclpriceError> {
    let (high, low, close, len, first, chosen) = wclprice_prepare(input, kern)?;
    if dst.len() != len {
        return Err(WclpriceError::InvalidOutputLen {
            out: dst.len(),
            min_len: len,
        });
    }
    // warmup prefix
    if first > 0 {
        dst[..first].fill(f64::NAN);
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wclprice_scalar(high, low, close, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wclprice_avx2(high, low, close, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wclprice_avx512(high, low, close, first, dst),
            _ => wclprice_scalar(high, low, close, first, dst),
        }
    }
    Ok(())
}

#[inline]
pub fn wclprice_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    // Process up to the minimum common length; callers ensure `out` is that length
    let len = high.len().min(low.len()).min(close.len());
    debug_assert_eq!(out.len(), len);

    // Using arithmetic to naturally propagate NaNs avoids a branch.
    // y = (h + l + 2c) / 4 = c*0.5 + (h + l)*0.25
    const HALF: f64 = 0.5;
    const QUARTER: f64 = 0.25;

    // Manually unroll by 8 (then 4) to improve ILP and reduce loop overhead while staying safe.
    let mut i = first_valid;
    let end = len;
    while i + 8 <= end {
        let h0 = high[i + 0];
        let l0 = low[i + 0];
        let c0 = close[i + 0];
        out[i + 0] = c0.mul_add(HALF, (h0 + l0) * QUARTER);

        let h1 = high[i + 1];
        let l1 = low[i + 1];
        let c1 = close[i + 1];
        out[i + 1] = c1.mul_add(HALF, (h1 + l1) * QUARTER);

        let h2 = high[i + 2];
        let l2 = low[i + 2];
        let c2 = close[i + 2];
        out[i + 2] = c2.mul_add(HALF, (h2 + l2) * QUARTER);

        let h3 = high[i + 3];
        let l3 = low[i + 3];
        let c3 = close[i + 3];
        out[i + 3] = c3.mul_add(HALF, (h3 + l3) * QUARTER);

        let h4 = high[i + 4];
        let l4 = low[i + 4];
        let c4 = close[i + 4];
        out[i + 4] = c4.mul_add(HALF, (h4 + l4) * QUARTER);

        let h5 = high[i + 5];
        let l5 = low[i + 5];
        let c5 = close[i + 5];
        out[i + 5] = c5.mul_add(HALF, (h5 + l5) * QUARTER);

        let h6 = high[i + 6];
        let l6 = low[i + 6];
        let c6 = close[i + 6];
        out[i + 6] = c6.mul_add(HALF, (h6 + l6) * QUARTER);

        let h7 = high[i + 7];
        let l7 = low[i + 7];
        let c7 = close[i + 7];
        out[i + 7] = c7.mul_add(HALF, (h7 + l7) * QUARTER);

        i += 8;
    }
    while i + 4 <= end {
        let h0 = high[i];
        let l0 = low[i];
        let c0 = close[i];
        out[i] = c0.mul_add(HALF, (h0 + l0) * QUARTER);

        let h1 = high[i + 1];
        let l1 = low[i + 1];
        let c1 = close[i + 1];
        out[i + 1] = c1.mul_add(HALF, (h1 + l1) * QUARTER);

        let h2 = high[i + 2];
        let l2 = low[i + 2];
        let c2 = close[i + 2];
        out[i + 2] = c2.mul_add(HALF, (h2 + l2) * QUARTER);

        let h3 = high[i + 3];
        let l3 = low[i + 3];
        let c3 = close[i + 3];
        out[i + 3] = c3.mul_add(HALF, (h3 + l3) * QUARTER);

        i += 4;
    }
    while i < end {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        out[i] = c.mul_add(HALF, (h + l) * QUARTER);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn wclprice_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = high.len().min(low.len()).min(close.len());
    debug_assert_eq!(out.len(), len);

    let mut i = first_valid;
    let end = len;

    let vhalf = _mm256_set1_pd(0.5);
    let vquart = _mm256_set1_pd(0.25);

    const STEP: usize = 4;
    // Unroll x2: process 8 doubles per iteration
    while i + 2 * STEP <= end {
        // i..i+3
        let h0 = _mm256_loadu_pd(high.as_ptr().add(i));
        let l0 = _mm256_loadu_pd(low.as_ptr().add(i));
        let c0 = _mm256_loadu_pd(close.as_ptr().add(i));
        let hl0 = _mm256_add_pd(h0, l0);
        let t0 = _mm256_mul_pd(hl0, vquart);
        // i+4..i+7
        let h1 = _mm256_loadu_pd(high.as_ptr().add(i + STEP));
        let l1 = _mm256_loadu_pd(low.as_ptr().add(i + STEP));
        let c1 = _mm256_loadu_pd(close.as_ptr().add(i + STEP));
        let hl1 = _mm256_add_pd(h1, l1);
        let t1 = _mm256_mul_pd(hl1, vquart);

        let y0 = _mm256_fmadd_pd(c0, vhalf, t0);
        let y1 = _mm256_fmadd_pd(c1, vhalf, t1);

        _mm256_storeu_pd(out.as_mut_ptr().add(i), y0);
        _mm256_storeu_pd(out.as_mut_ptr().add(i + STEP), y1);

        i += 2 * STEP;
    }
    // Remainder vector
    while i + STEP <= end {
        let h = _mm256_loadu_pd(high.as_ptr().add(i));
        let l = _mm256_loadu_pd(low.as_ptr().add(i));
        let c = _mm256_loadu_pd(close.as_ptr().add(i));
        let hl = _mm256_add_pd(h, l);
        let t = _mm256_mul_pd(hl, vquart);
        let y = _mm256_fmadd_pd(c, vhalf, t);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), y);
        i += STEP;
    }
    while i < end {
        let h = *high.get_unchecked(i);
        let l = *low.get_unchecked(i);
        let c = *close.get_unchecked(i);
        *out.get_unchecked_mut(i) = c.mul_add(0.5, (h + l) * 0.25);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn wclprice_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = high.len().min(low.len()).min(close.len());
    debug_assert_eq!(out.len(), len);

    let mut i = first_valid;
    let end = len;

    let vhalf = _mm512_set1_pd(0.5);
    let vquart = _mm512_set1_pd(0.25);

    const STEP: usize = 8;
    // Unroll x2: process 16 doubles per iteration
    while i + 2 * STEP <= end {
        // i..i+7
        let h0 = _mm512_loadu_pd(high.as_ptr().add(i));
        let l0 = _mm512_loadu_pd(low.as_ptr().add(i));
        let c0 = _mm512_loadu_pd(close.as_ptr().add(i));
        let hl0 = _mm512_add_pd(h0, l0);
        let t0 = _mm512_mul_pd(hl0, vquart);

        // i+8..i+15
        let h1 = _mm512_loadu_pd(high.as_ptr().add(i + STEP));
        let l1 = _mm512_loadu_pd(low.as_ptr().add(i + STEP));
        let c1 = _mm512_loadu_pd(close.as_ptr().add(i + STEP));
        let hl1 = _mm512_add_pd(h1, l1);
        let t1 = _mm512_mul_pd(hl1, vquart);

        let y0 = _mm512_fmadd_pd(c0, vhalf, t0);
        let y1 = _mm512_fmadd_pd(c1, vhalf, t1);

        _mm512_storeu_pd(out.as_mut_ptr().add(i), y0);
        _mm512_storeu_pd(out.as_mut_ptr().add(i + STEP), y1);

        i += 2 * STEP;
    }
    // Remainder vector
    while i + STEP <= end {
        let h = _mm512_loadu_pd(high.as_ptr().add(i));
        let l = _mm512_loadu_pd(low.as_ptr().add(i));
        let c = _mm512_loadu_pd(close.as_ptr().add(i));
        let hl = _mm512_add_pd(h, l);
        let t = _mm512_mul_pd(hl, vquart);
        let y = _mm512_fmadd_pd(c, vhalf, t);
        _mm512_storeu_pd(out.as_mut_ptr().add(i), y);
        i += STEP;
    }
    while i < end {
        let h = *high.get_unchecked(i);
        let l = *low.get_unchecked(i);
        let c = *close.get_unchecked(i);
        *out.get_unchecked_mut(i) = c.mul_add(0.5, (h + l) * 0.25);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[inline]
pub fn wclprice_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { wclprice_avx2(high, low, close, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { wclprice_avx512(high, low, close, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx512_short(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx512_long(high, low, close, first_valid, out)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchRange; // No parameters

impl Default for WclpriceBatchRange {
    fn default() -> Self {
        Self
    }
}

#[derive(Clone, Debug, Default)]
pub struct WclpriceBatchBuilder {
    kernel: Kernel,
}
impl WclpriceBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WclpriceBatchOutput, WclpriceError> {
        wclprice_batch_with_kernel(high, low, close, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
        let h = c
            .select_candle_field("high")
            .map_err(|_| WclpriceError::MissingField { field: "high" })?;
        let l = c
            .select_candle_field("low")
            .map_err(|_| WclpriceError::MissingField { field: "low" })?;
        let cl = c
            .select_candle_field("close")
            .map_err(|_| WclpriceError::MissingField { field: "close" })?;
        self.apply_slices(h, l, cl)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
        WclpriceBatchBuilder::new().apply_candles(c)
    }
}

pub fn wclprice_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(WclpriceError::InvalidBatchKernel(other)),
    };
    wclprice_batch_par_slice(high, low, close, kernel)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WclpriceParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WclpriceBatchOutput {
    pub fn values_for(&self, _params: &WclpriceParams) -> Option<&[f64]> {
        if self.rows == 1 {
            Some(&self.values[..self.cols])
        } else {
            None
        }
    }
}

#[inline(always)]
pub fn wclprice_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    wclprice_batch_inner(high, low, close, kern, false)
}
#[inline(always)]
pub fn wclprice_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    wclprice_batch_inner(high, low, close, kern, true)
}
#[inline(always)]
pub fn wclprice_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyData);
    }
    let len = high.len().min(low.len()).min(close.len());
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;

    // 1 row matrix
    let mut buf_mu = make_uninit_matrix(1, len);
    init_matrix_prefixes(&mut buf_mu, len, &[first]);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let simd = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let map = match simd {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        other => other,
    };

    wclprice_batch_inner_into(high, low, close, map, _parallel, out_slice)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(WclpriceBatchOutput {
        values,
        combos: vec![WclpriceParams],
        rows: 1,
        cols: len,
    })
}

#[inline(always)]
fn expand_grid(_r: &WclpriceBatchRange) -> Vec<WclpriceParams> {
    vec![WclpriceParams]
}

#[inline(always)]
fn wclprice_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<WclpriceParams>, WclpriceError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyData);
    }
    let len = high.len().min(low.len()).min(close.len());
    if out.len() < len {
        return Err(WclpriceError::InvalidOutputLen {
            out: out.len(),
            min_len: len,
        });
    }
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;

    // Since WCLPRICE has no parameters, we only have one row
    // Fill the output slice with NaN up to first valid index
    if first > 0 {
        out[..first].fill(f64::NAN);
    }

    unsafe {
        match kern {
            Kernel::Scalar => wclprice_row_scalar(high, low, close, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => wclprice_row_avx2(high, low, close, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => wclprice_row_avx512(high, low, close, first, out),
            _ => wclprice_row_scalar(high, low, close, first, out),
        }
    }

    Ok(vec![WclpriceParams])
}

#[derive(Debug, Clone)]
pub struct WclpriceStream;
impl Default for WclpriceStream {
    fn default() -> Self {
        Self
    }
}
impl WclpriceStream {
    #[inline(always)]
    pub fn update(&mut self, h: f64, l: f64, c: f64) -> Option<f64> {
        if h.is_nan() || l.is_nan() || c.is_nan() {
            None
        } else {
            Some((h + l + 2.0 * c) / 4.0)
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "wclprice")]
#[pyo3(signature = (high, low, close, kernel=None))]
pub fn wclprice_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};
    let hs = high.as_slice()?;
    let ls = low.as_slice()?;
    let cs = close.as_slice()?;
    let len = hs.len().min(ls.len()).min(cs.len());
    let out = unsafe { PyArray1::<f64>::new(py, [len], false) };
    let out_slice = unsafe { out.as_slice_mut()? };
    let input = WclpriceInput::from_slices(hs, ls, cs);
    let kern = validate_kernel(kernel, false)?;
    py.allow_threads(|| wclprice_into_slice(out_slice, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "python")]
#[pyclass(name = "WclpriceStream")]
pub struct WclpriceStreamPy {
    stream: WclpriceStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WclpriceStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(WclpriceStreamPy {
            stream: WclpriceStream::default(),
        })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "wclprice_batch")]
#[pyo3(signature = (high, low, close, kernel=None))]
pub fn wclprice_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let hs = high.as_slice()?;
    let ls = low.as_slice()?;
    let cs = close.as_slice()?;

    let rows = 1usize;
    let cols = hs.len().min(ls.len()).min(cs.len());

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch_kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match batch_kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            other => other,
        };
        wclprice_batch_inner_into(hs, ls, cs, simd, true, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    // Alma-compatible param vectors: single row, so length-1 placeholders
    dict.set_item("periods", vec![0u64].into_pyarray(py))?;
    dict.set_item("offsets", vec![0.0f64].into_pyarray(py))?;
    dict.set_item("sigmas", vec![0.0f64].into_pyarray(py))?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wclprice_cuda_dev")]
#[pyo3(signature = (high, low, close, device_id=0))]
pub fn wclprice_cuda_dev_py(
    py: Python<'_>,
    high: numpy::PyReadonlyArray1<'_, f32>,
    low: numpy::PyReadonlyArray1<'_, f32>,
    close: numpy::PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let hs = high.as_slice()?;
    let ls = low.as_slice()?;
    let cs = close.as_slice()?;

    let inner = py.allow_threads(|| {
        let cuda =
            CudaWclprice::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.wclprice_dev(hs, ls, cs)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_js(high: &[f64], low: &[f64], close: &[f64]) -> Result<Vec<f64>, JsValue> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(JsValue::from_str("wclprice: Empty data provided"));
    }

    let input = WclpriceInput::from_slices(high, low, close);
    let mut output = vec![0.0; high.len().min(low.len()).min(close.len())];

    wclprice_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let input = WclpriceInput::from_slices(high, low, close);

        // Check for aliasing with any input pointer
        if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            wclprice_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            wclprice_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WclpriceBatchConfig {
    // intentionally empty; reserved for future
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WclpriceBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WclpriceParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = wclprice_batch)]
pub fn wclprice_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    cfg: JsValue,
) -> Result<JsValue, JsValue> {
    let _cfg: WclpriceBatchConfig =
        serde_wasm_bindgen::from_value(cfg).unwrap_or(WclpriceBatchConfig {});
    let out = wclprice_batch_inner(high, low, close, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = WclpriceBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        // WCLPRICE has no parameters, so only 1 row
        let rows = 1;

        // Check for aliasing with any input pointer
        if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            wclprice_batch_inner_into(high, low, close, detect_best_kernel(), false, &mut temp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            wclprice_batch_inner_into(high, low, close, detect_best_kernel(), false, out)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_wclprice_slices(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![59230.0, 59220.0, 59077.0, 59160.0, 58717.0];
        let low = vec![59222.0, 59211.0, 59077.0, 59143.0, 58708.0];
        let close = vec![59225.0, 59210.0, 59080.0, 59150.0, 58710.0];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice_with_kernel(&input, kernel)?;
        let expected = vec![59225.5, 59212.75, 59078.5, 59150.75, 58711.25];
        for (i, &v) in output.values.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
                "[{test}] mismatch at {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }
    fn check_wclprice_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let input = WclpriceInput::from_candles(&candles);
        let output = wclprice_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_wclprice_empty_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let res = wclprice_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with empty data", test);
        Ok(())
    }
    fn check_wclprice_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![f64::NAN, f64::NAN];
        let low = vec![f64::NAN, f64::NAN];
        let close = vec![f64::NAN, f64::NAN];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let res = wclprice_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with all NaN", test);
        Ok(())
    }
    fn check_wclprice_partial_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![f64::NAN, 59000.0];
        let low = vec![f64::NAN, 58950.0];
        let close = vec![f64::NAN, 58975.0];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice_with_kernel(&input, kernel)?;
        assert!(output.values[0].is_nan());
        assert!((output.values[1] - (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0).abs() < 1e-8);
        Ok(())
    }

    fn check_wclprice_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WclpriceInput::from_candles(&candles);
        let result = wclprice_with_kernel(&input, kernel)?;

        // Reference values provided by user
        let expected_last_five = [59225.5, 59212.75, 59078.5, 59150.75, 58711.25];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] WCLPRICE {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_wclprice_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Test with real candle data
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with full candle data
        let input = WclpriceInput::from_candles(&candles);
        let output = wclprice_with_kernel(&input, kernel)?;

        for (i, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue; // NaN values are expected during warmup
            }

            let bits = val.to_bits();

            // Check all three poison patterns
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
					 in WCLPRICE with candle data",
                    test_name, val, bits, i
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
					 in WCLPRICE with candle data",
                    test_name, val, bits, i
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
					 in WCLPRICE with candle data",
                    test_name, val, bits, i
                );
            }
        }

        // Test with various slice patterns to ensure no poison in different scenarios
        let test_patterns = vec![
            // (description, high, low, close)
            (
                "small dataset",
                vec![100.0, 101.0, 102.0],
                vec![99.0, 100.0, 101.0],
                vec![100.5, 101.5, 102.5],
            ),
            (
                "large values",
                vec![59000.0, 60000.0, 61000.0],
                vec![58900.0, 59900.0, 60900.0],
                vec![58950.0, 59950.0, 60950.0],
            ),
            (
                "with leading NaN",
                vec![f64::NAN, 100.0, 101.0],
                vec![f64::NAN, 99.0, 100.0],
                vec![f64::NAN, 100.5, 101.5],
            ),
            (
                "with trailing NaN",
                vec![100.0, 101.0, f64::NAN],
                vec![99.0, 100.0, f64::NAN],
                vec![100.5, 101.5, f64::NAN],
            ),
            (
                "mixed NaN pattern",
                vec![100.0, f64::NAN, 101.0, f64::NAN, 102.0],
                vec![99.0, f64::NAN, 100.0, f64::NAN, 101.0],
                vec![100.5, f64::NAN, 101.5, f64::NAN, 102.5],
            ),
            (
                "single valid value",
                vec![f64::NAN, f64::NAN, 100.0],
                vec![f64::NAN, f64::NAN, 99.0],
                vec![f64::NAN, f64::NAN, 100.5],
            ),
            (
                "extreme values",
                vec![1e-10, 1e10, 1e-10],
                vec![1e-10, 1e10, 1e-10],
                vec![1e-10, 1e10, 1e-10],
            ),
            (
                "zero values",
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.5, 0.0],
            ),
            (
                "negative values",
                vec![-100.0, -50.0, -25.0],
                vec![-101.0, -51.0, -26.0],
                vec![-100.5, -50.5, -25.5],
            ),
            (
                "large dataset",
                (0..1000).map(|i| 100.0 + i as f64).collect(),
                (0..1000).map(|i| 99.0 + i as f64).collect(),
                (0..1000).map(|i| 100.5 + i as f64).collect(),
            ),
        ];

        for (pattern_idx, (desc, high, low, close)) in test_patterns.iter().enumerate() {
            let input = WclpriceInput::from_slices(high, low, close);
            let output = wclprice_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in WCLPRICE with pattern '{}' (pattern {})",
                        test_name, val, bits, i, desc, pattern_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in WCLPRICE with pattern '{}' (pattern {})",
                        test_name, val, bits, i, desc, pattern_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in WCLPRICE with pattern '{}' (pattern {})",
                        test_name, val, bits, i, desc, pattern_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_wclprice_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_wclprice_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic price data with proper OHLC constraints
        let strat = (2usize..=400).prop_flat_map(|len| {
            prop::collection::vec(
                // Generate low first, then high >= low, then close within [low, high]
                // This ensures valid OHLC data where close is always between low and high
                (0.0f64..1e6f64)
                    .prop_filter("finite non-negative price", |x| x.is_finite() && *x >= 0.0)
                    .prop_flat_map(|low| {
                        (0.0f64..10000.0f64) // Allow larger spreads for more realistic volatility
                            .prop_filter("finite diff", |x| x.is_finite())
                            .prop_flat_map(move |high_diff| {
                                let high = low + high_diff;
                                (
                                    Just(low),
                                    Just(high),
                                    (low..=high).prop_filter("finite close", |x| x.is_finite()),
                                )
                            })
                    }),
                len,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |price_data| {
                // Extract high, low, close arrays from generated data
                let mut high = Vec::with_capacity(price_data.len());
                let mut low = Vec::with_capacity(price_data.len());
                let mut close = Vec::with_capacity(price_data.len());

                for (l, h, c) in price_data.iter() {
                    low.push(*l);
                    high.push(*h);
                    close.push(*c);
                }

                // Create input and compute output
                let input = WclpriceInput::from_slices(&high, &low, &close);
                let WclpriceOutput { values: out } = wclprice_with_kernel(&input, kernel)?;

                // Also compute with scalar kernel for reference
                let WclpriceOutput { values: ref_out } =
                    wclprice_with_kernel(&input, Kernel::Scalar)?;

                // Verify properties for each output value
                for i in 0..price_data.len() {
                    let h = high[i];
                    let l = low[i];
                    let c = close[i];
                    let y = out[i];
                    let r = ref_out[i];

                    // Property 1: Formula correctness
                    // WCLPRICE = (high + low + 2*close) / 4
                    if h.is_finite() && l.is_finite() && c.is_finite() {
                        let expected = (h + l + 2.0 * c) / 4.0;
                        prop_assert!(
                            (y - expected).abs() <= 1e-9,
                            "Formula mismatch at idx {}: got {} expected {} (h={}, l={}, c={})",
                            i,
                            y,
                            expected,
                            h,
                            l,
                            c
                        );
                    } else {
                        // If any input is NaN/infinite, output should be NaN
                        prop_assert!(
                            y.is_nan(),
                            "Expected NaN at idx {} when input has non-finite values, got {}",
                            i,
                            y
                        );
                    }

                    // Property 2: Output bounds
                    // WCLPRICE should be within the range of input values
                    if h.is_finite() && l.is_finite() && c.is_finite() {
                        let min_val = h.min(l).min(c);
                        let max_val = h.max(l).max(c);
                        prop_assert!(
                            y >= min_val - 1e-9 && y <= max_val + 1e-9,
                            "Output {} at idx {} outside bounds [{}, {}]",
                            y,
                            i,
                            min_val,
                            max_val
                        );
                    }

                    // Property 3: Kernel consistency
                    // All kernels should produce identical results
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        // For NaN/infinite values, bit patterns should match exactly
                        prop_assert!(
                            y_bits == r_bits,
                            "NaN/infinite mismatch at idx {}: {} vs {} (bits: {:016x} vs {:016x})",
                            i,
                            y,
                            r,
                            y_bits,
                            r_bits
                        );
                    } else {
                        // For finite values, allow small ULP difference
                        let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }

                    // Property 4: Special cases
                    // When all prices are equal, WCLPRICE should equal that price
                    if (h - l).abs() < f64::EPSILON && (h - c).abs() < f64::EPSILON {
                        prop_assert!(
                            (y - h).abs() <= 1e-9,
                            "When all prices equal {}, WCLPRICE should be {}, got {}",
                            h,
                            h,
                            y
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_wclprice_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                   #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }
    generate_all_wclprice_tests!(
        check_wclprice_slices,
        check_wclprice_candles,
        check_wclprice_empty_data,
        check_wclprice_all_nan,
        check_wclprice_partial_nan,
        check_wclprice_accuracy,
        check_wclprice_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_wclprice_tests!(check_wclprice_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = WclpriceBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let row = output
            .values_for(&WclpriceParams)
            .expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch processing with candle data
        let output = WclpriceBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        // WCLPRICE has no parameters, so there's only one row
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, c.close.len());

        // Check for poison values in the batch output
        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
					 in WCLPRICE batch at index {} (candle data)",
                    test, val, bits, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) \
					 in WCLPRICE batch at index {} (candle data)",
                    test, val, bits, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) \
					 in WCLPRICE batch at index {} (candle data)",
                    test, val, bits, idx
                );
            }
        }

        // Test batch processing with various slice patterns
        let test_configs = vec![
            // (description, high, low, close)
            (
                "small data",
                vec![100.0, 101.0, 102.0],
                vec![99.0, 100.0, 101.0],
                vec![100.5, 101.5, 102.5],
            ),
            (
                "medium data",
                (0..100).map(|i| 100.0 + i as f64).collect(),
                (0..100).map(|i| 99.0 + i as f64).collect(),
                (0..100).map(|i| 100.5 + i as f64).collect(),
            ),
            (
                "large data",
                (0..5000).map(|i| 100.0 + (i as f64 * 0.1)).collect(),
                (0..5000).map(|i| 99.0 + (i as f64 * 0.1)).collect(),
                (0..5000).map(|i| 100.5 + (i as f64 * 0.1)).collect(),
            ),
            (
                "with NaN prefix",
                [
                    vec![f64::NAN; 10],
                    (0..90).map(|i| 100.0 + i as f64).collect(),
                ]
                .concat(),
                [
                    vec![f64::NAN; 10],
                    (0..90).map(|i| 99.0 + i as f64).collect(),
                ]
                .concat(),
                [
                    vec![f64::NAN; 10],
                    (0..90).map(|i| 100.5 + i as f64).collect(),
                ]
                .concat(),
            ),
            (
                "sparse NaN pattern",
                (0..50)
                    .map(|i| {
                        if i % 5 == 0 {
                            f64::NAN
                        } else {
                            100.0 + i as f64
                        }
                    })
                    .collect(),
                (0..50)
                    .map(|i| {
                        if i % 5 == 0 {
                            f64::NAN
                        } else {
                            99.0 + i as f64
                        }
                    })
                    .collect(),
                (0..50)
                    .map(|i| {
                        if i % 5 == 0 {
                            f64::NAN
                        } else {
                            100.5 + i as f64
                        }
                    })
                    .collect(),
            ),
            (
                "extreme values",
                vec![1e-100, 1e100, 1e-50, 1e50],
                vec![1e-100, 1e100, 1e-50, 1e50],
                vec![1e-100, 1e100, 1e-50, 1e50],
            ),
        ];

        for (cfg_idx, (desc, high, low, close)) in test_configs.iter().enumerate() {
            let output = WclpriceBatchBuilder::new()
                .kernel(kernel)
                .apply_slices(high, low, close)?;

            // Verify batch structure
            assert_eq!(
                output.rows, 1,
                "[{}] Config {}: Expected 1 row for WCLPRICE",
                test, cfg_idx
            );
            assert_eq!(
                output.cols,
                high.len(),
                "[{}] Config {}: Cols mismatch",
                test,
                cfg_idx
            );

            // Check for poison values
            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Config {} ({}): Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 in WCLPRICE batch at index {}",
						test, cfg_idx, desc, val, bits, idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {} ({}): Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 in WCLPRICE batch at index {}",
						test, cfg_idx, desc, val, bits, idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {} ({}): Found make_uninit_matrix poison value {} (0x{:016X}) \
						 in WCLPRICE batch at index {}",
						test, cfg_idx, desc, val, bits, idx
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
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        }
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
