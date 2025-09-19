//! # Volume Price Trend (VPT)
//!
//! Implements the cumulative Volume Price Trend indicator, which accumulates
//! volume-weighted price changes over time. This is the standard definition of VPT.
//!
//! Note: This implementation calculates cumulative VPT where each value is the
//! running sum of all previous volume * (price_change / previous_price) values.
//! Some implementations (like certain Python libraries) may use a non-cumulative
//! version that only adds the current and previous period values.
//!
//! ## Parameters
//! None (uses price/volume arrays).
//!
//! ## Returns
//! - **Ok(VptOutput)** with output array.
//! - **Err(VptError)** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Stub implementations that delegate to scalar. Comments indicate "API parity only". Cumulative nature makes SIMD challenging but could vectorize price change calculations.
//! - **Streaming Performance**: O(1) implementation with simple cumulative sum tracking. Very efficient - only stores last price and VPT value.
//! - **Memory Optimization**: Uses `alloc_with_nan_prefix` and batch helpers properly. Streaming is optimal with minimal state.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use std::error::Error;
use thiserror::Error;

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
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum VptData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VptOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VptParams;

#[derive(Debug, Clone)]
pub struct VptInput<'a> {
    pub data: VptData<'a>,
    pub params: VptParams,
}

impl<'a> VptInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
        Self {
            data: VptData::Candles { candles, source },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn from_slices(price: &'a [f64], volume: &'a [f64]) -> Self {
        Self {
            data: VptData::Slices { price, volume },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VptData::Candles {
                candles,
                source: "close",
            },
            params: VptParams::default(),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VptBuilder {
    kernel: Kernel,
}

impl VptBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VptOutput, VptError> {
        let i = VptInput::with_default_candles(c);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
        let i = VptInput::from_slices(price, volume);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> VptStream {
        VptStream::default()
    }
}

#[derive(Debug, Error)]
pub enum VptError {
    #[error("vpt: Empty data provided.")]
    EmptyData,
    #[error("vpt: All values are NaN.")]
    AllValuesNaN,
    #[error("vpt: Not enough valid data (fewer than 2 valid points).")]
    NotEnoughValidData,
    #[error("vpt: Invalid output length. expected={expected}, got={got}")]
    InvalidLength { expected: usize, got: usize },
}

#[inline]
fn vpt_first_valid(price: &[f64], volume: &[f64]) -> Option<usize> {
    // VPT always has NaN at index 0, so warmup is at least 1
    // Find earliest i >= 1 where the formula can produce a finite value
    // needs p[i-1] finite and != 0, p[i] finite, v[i] finite
    for i in 1..price.len() {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
            return Some(i);
        }
    }
    None
}

#[inline]
pub fn vpt(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, Kernel::Auto)
}

pub fn vpt_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    let (price, volume) = match &input.data {
        VptData::Candles { candles, source } => {
            let price = source_type(candles, source);
            let vol = candles
                .select_candle_field("volume")
                .map_err(|_| VptError::EmptyData)?;
            (price, vol)
        }
        VptData::Slices { price, volume } => (*price, *volume),
    };

    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyData);
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData);
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => vpt_scalar(price, volume),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_avx2(price, volume),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => vpt_avx512(price, volume),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn vpt_scalar(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    let n = price.len();
    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData)?;
    let mut res = alloc_with_nan_prefix(n, first + 1);

    // seed cumulative with vpt_val at `first`
    let p0 = price[first - 1];
    let p1 = price[first];
    let v1 = volume[first];
    let mut prev_cum = v1 * ((p1 - p0) / p0);

    for i in (first + 1)..n {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        let cur = if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        };
        res[i] = if cur.is_nan() || prev_cum.is_nan() {
            f64::NAN
        } else {
            cur + prev_cum
        };
        prev_cum = res[i];
    }
    Ok(VptOutput { values: res })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx2(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    // For API parity only; reuses scalar logic.
    vpt_scalar(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    // For API parity only; reuses scalar logic.
    vpt_scalar(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_short(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_long(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[inline]
pub fn vpt_indicator(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt(input)
}

#[inline]
pub fn vpt_indicator_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, kernel)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx2(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx2(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_short(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_short(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_long(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_long(price, volume)
    }
}

#[inline]
pub fn vpt_indicator_scalar(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_scalar(price, volume)
    }
}

#[inline]
pub fn vpt_expand_grid() -> Vec<VptParams> {
    vec![VptParams::default()]
}

/// Write VPT directly to output slice - no allocations
pub fn vpt_into_slice(
    dst: &mut [f64],
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<(), VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyData);
    }

    if dst.len() != price.len() {
        return Err(VptError::InvalidLength {
            expected: price.len(),
            got: dst.len(),
        });
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData);
    }

    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData)?;
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                vpt_row_scalar_from(price, volume, first + 1, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_row_avx2_from(price, volume, first + 1, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpt_row_avx512_from(price, volume, first + 1, dst)
            }
            _ => vpt_row_scalar_from(price, volume, first + 1, dst),
        }
    }
    for v in &mut dst[..=first] {
        *v = f64::NAN;
    }
    Ok(())
}

pub fn vpt_batch_inner_into(
    price: &[f64],
    volume: &[f64],
    _range: &VptBatchRange,
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VptParams>, VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyData);
    }
    let combos = vec![VptParams::default()];
    let cols = price.len();
    if out.len() != cols {
        return Err(VptError::InvalidLength {
            expected: cols,
            got: out.len(),
        });
    }

    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData)?;

    // change calls to start at first + 1
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                vpt_row_scalar_from(price, volume, first + 1, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_row_avx2_from(price, volume, first + 1, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpt_row_avx512_from(price, volume, first + 1, out)
            }
            _ => vpt_row_scalar_from(price, volume, first + 1, out),
        }
    }
    Ok(combos)
}

#[derive(Clone, Debug, Default)]
pub struct VptStream {
    last_price: f64,
    last_vpt: f64,
    is_initialized: bool,
}

impl VptStream {
    #[inline]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        if !self.is_initialized {
            self.last_price = price;
            self.last_vpt = f64::NAN; // Start with NaN to match array behavior
            self.is_initialized = true;
            return None;
        }
        if self.last_price.is_nan() || self.last_price == 0.0 || price.is_nan() || volume.is_nan() {
            self.last_price = price;
            self.last_vpt = f64::NAN; // Keep as NaN to propagate
            return Some(f64::NAN);
        }
        let vpt_val = volume * ((price - self.last_price) / self.last_price);
        // Cumulative sum: current VPT value + previous VPT sum
        // First actual calculation returns NaN because last_vpt starts as NaN
        let out = if self.last_vpt.is_nan() {
            // This is the first actual VPT calculation, return NaN but save the value for next time
            self.last_price = price;
            self.last_vpt = vpt_val; // Save first value for cumulative calculation
            f64::NAN
        } else {
            let result = vpt_val + self.last_vpt;
            self.last_price = price;
            self.last_vpt = result; // Store cumulative sum
            result
        };
        Some(out)
    }
}

#[derive(Clone, Debug, Default)]
pub struct VptBatchRange;

#[derive(Clone, Debug, Default)]
pub struct VptBatchBuilder {
    kernel: Kernel,
}

impl VptBatchBuilder {
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptBatchOutput, VptError> {
        vpt_batch_with_kernel(price, volume, self.kernel)
    }

    pub fn with_default_slices(
        price: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new().kernel(k).apply_slices(price, volume)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VptBatchOutput, VptError> {
        let price = source_type(c, src);
        let volume = c
            .select_candle_field("volume")
            .map_err(|_| VptError::EmptyData)?;
        self.apply_slices(price, volume)
    }

    pub fn with_default_candles(c: &Candles) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn vpt_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    k: Kernel,
) -> Result<VptBatchOutput, VptError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    vpt_batch_par_slice(price, volume, kernel)
}

#[derive(Clone, Debug)]
pub struct VptBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VptParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VptBatchOutput {
    pub fn row_for_params(&self, _p: &VptParams) -> Option<usize> {
        Some(0)
    }

    pub fn values_for(&self, _p: &VptParams) -> Option<&[f64]> {
        Some(&self.values[..])
    }
}

#[inline(always)]
pub fn vpt_batch_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, false)
}

#[inline(always)]
pub fn vpt_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, true)
}

#[inline(always)]
fn vpt_batch_inner(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<VptBatchOutput, VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyData);
    }

    let combos = vpt_expand_grid();
    let rows = 1usize;
    let cols = price.len();

    // uninit matrix, then fill warmup prefixes with NaN
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // For VPT, warmup is always at least 1 (index 0 is always NaN)
    // but might be more if there are NaN values in the data
    let first_valid = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData)?;
    let warm = vec![first_valid + 1];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // get &mut [f64] view over MaybeUninit<f64> buffer
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    vpt_batch_inner_into(price, volume, &VptBatchRange, kern, _parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VptBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn vpt_row_scalar(price: &[f64], volume: &[f64], out: &mut [f64]) {
    // full coverage writer for python path
    // Find first valid index and set everything before and including it to NaN
    let n = price.len();
    if let Some(first) = vpt_first_valid(price, volume) {
        // Set warmup prefix to NaN
        for i in 0..=first {
            out[i] = f64::NAN;
        }
        // Use the _from variant starting at first + 1
        vpt_row_scalar_from(price, volume, first + 1, out);
    } else {
        // No valid data, all NaN
        for i in 0..n {
            out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
pub unsafe fn vpt_row_scalar_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    let n = price.len();
    // seed = VPT value at index (start_i - 1); not written to out
    let mut prev_vpt_val = if start_i >= 2 {
        let k = start_i - 1;
        let p0 = price[k - 1];
        let p1 = price[k];
        let v1 = volume[k];
        if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    } else {
        0.0
    };

    for i in start_i..n {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        let cur = if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        };
        out[i] = if cur.is_nan() || prev_vpt_val.is_nan() {
            f64::NAN
        } else {
            cur + prev_vpt_val
        };
        prev_vpt_val = out[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    vpt_row_scalar_from(price, volume, start_i, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    vpt_row_scalar_from(price, volume, start_i, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_short(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_long(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpt")]
#[pyo3(signature = (price, volume, kernel=None))]
pub fn vpt_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let price_slice = price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let input = VptInput::from_slices(price_slice, volume_slice);

    let result_vec: Vec<f64> = py
        .allow_threads(|| vpt_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VptStream")]
pub struct VptStreamPy {
    stream: VptStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VptStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(VptStreamPy {
            stream: VptStream::default(),
        })
    }

    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(price, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpt_batch")]
#[pyo3(signature = (price, volume, kernel=None))]
pub fn vpt_batch_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let price_slice = price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    // VPT has no parameters, so single row output
    let rows = 1;
    let cols = price_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Initialize NaN prefix for VPT (indices 0..=first_valid)
    let first_valid = vpt_first_valid(price_slice, volume_slice)
        .ok_or_else(|| PyValueError::new_err("Not enough valid data"))?;

    // set [0..=first_valid] to NaN
    for i in 0..=first_valid {
        slice_out[i] = f64::NAN;
    }

    let _combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            vpt_batch_inner_into(
                price_slice,
                volume_slice,
                &VptBatchRange,
                kernel,
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    // No parameters for VPT, but include empty list for consistency
    dict.set_item("params", Vec::<f64>::new().into_pyarray(py))?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_js(price: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
    let mut output = vec![0.0; price.len()];

    vpt_into_slice(&mut output, price, volume, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let price = std::slice::from_raw_parts(price_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        // Check if either input aliases with output
        if price_ptr == out_ptr || volume_ptr == out_ptr {
            // Need temporary buffer for aliasing
            let mut temp = vec![0.0; len];
            vpt_into_slice(&mut temp, price, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, write directly
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vpt_into_slice(out, price, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VptBatchConfig {
    // VPT has no parameters, so empty config
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VptBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VptParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vpt_batch)]
pub fn vpt_batch_js(price: &[f64], volume: &[f64], _config: JsValue) -> Result<JsValue, JsValue> {
    // VPT has no parameters, so batch returns single row
    let output = vpt_batch_with_kernel(price, volume, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = VptBatchJsOutput {
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
pub fn vpt_batch_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let price = std::slice::from_raw_parts(price_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);

        // VPT has no parameters, so just compute once
        vpt_batch_inner_into(price, volume, &VptBatchRange, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Return number of parameter combinations (always 1 for VPT)
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_vpt_basic_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vpt_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [1.0, 1.1, 1.05, 1.2, 1.3];
        let volume = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
        let input = VptInput::from_slices(&price, &volume);
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), price.len());
        Ok(())
    }

    fn check_vpt_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [100.0];
        let volume = [500.0];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN, f64::NAN];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_accuracy_from_csv(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;

        // NOTE: The Rust implementation calculates cumulative VPT (standard definition)
        // Python reference values were for a non-cumulative version:
        // [-0.40358334248536065, -0.16292768139917702, -0.4792942916867958,
        //  -0.1188231211518107, -3.3492674990910025]
        //
        // Our implementation accumulates all historical VPT values, while the Python
        // version only adds the current and previous period values.
        let expected_last_five = [
            -18292.323972247592,
            -18292.510374716476,
            -18292.803266539282,
            -18292.62919783763,
            -18296.152568643138,
        ];

        assert!(output.values.len() >= 5);
        let start_index = output.values.len() - 5;
        for (i, &value) in output.values[start_index..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "VPT mismatch at final bars, index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    macro_rules! generate_all_vpt_tests {
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
    fn check_vpt_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Since VPT has no parameters, we'll test with different data sources
        let test_sources = vec!["close", "open", "high", "low"];

        for (source_idx, &source) in test_sources.iter().enumerate() {
            let input = VptInput::from_candles(&candles, source);
            let output = vpt_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vpt_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vpt_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy to generate price and volume data
        // Prices: non-negative values (including zero to test edge case)
        // Volume: non-negative values (realistic for trading volume)
        // Length: at least 2 points (minimum for VPT calculation)
        let strat = (2usize..=400).prop_flat_map(|len| {
            (
                prop::collection::vec(
                    (0.0f64..1e6f64)
                        .prop_filter("finite non-negative price", |x| x.is_finite() && *x >= 0.0),
                    len,
                ),
                prop::collection::vec(
                    (0.0f64..1e9f64)
                        .prop_filter("finite non-negative volume", |x| x.is_finite() && *x >= 0.0),
                    len,
                ),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(price, volume)| {
            let input = VptInput::from_slices(&price, &volume);

            // Get output from the kernel being tested
            let VptOutput { values: out } = vpt_with_kernel(&input, kernel)?;

            // Get reference output from scalar kernel
            let VptOutput { values: ref_out } = vpt_with_kernel(&input, Kernel::Scalar)?;

            // Verify properties
            prop_assert_eq!(out.len(), price.len(), "Output length mismatch");
            prop_assert_eq!(
                ref_out.len(),
                price.len(),
                "Reference output length mismatch"
            );

            // First value should always be NaN (warmup period)
            prop_assert!(
                out[0].is_nan(),
                "First VPT value should be NaN, got {}",
                out[0]
            );
            prop_assert!(
                ref_out[0].is_nan(),
                "First reference VPT value should be NaN, got {}",
                ref_out[0]
            );

            // Manually calculate VPT to verify correctness
            let mut expected_vpt = vec![f64::NAN; price.len()];
            let mut prev_vpt_val = f64::NAN;

            for i in 1..price.len() {
                let p0 = price[i - 1];
                let p1 = price[i];
                let v1 = volume[i];

                // Calculate current VPT value
                let vpt_val = if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
                    f64::NAN
                } else {
                    v1 * ((p1 - p0) / p0)
                };

                // Output is current VPT value + previous VPT value (shifted array approach)
                expected_vpt[i] = if vpt_val.is_nan() || prev_vpt_val.is_nan() {
                    f64::NAN
                } else {
                    vpt_val + prev_vpt_val
                };

                // Save current VPT value for next iteration
                prev_vpt_val = vpt_val;
            }

            // Compare outputs with expected calculations
            for i in 0..price.len() {
                let y = out[i];
                let r = ref_out[i];
                let e = expected_vpt[i];

                // Check consistency between kernels
                if y.is_nan() && r.is_nan() {
                    // Both NaN is fine
                    continue;
                } else if !y.is_nan() && !r.is_nan() {
                    // Both should be very close
                    let diff = (y - r).abs();
                    prop_assert!(
                        diff < 1e-9,
                        "Kernel mismatch at idx {}: {} vs {} (diff: {})",
                        i,
                        y,
                        r,
                        diff
                    );

                    // Also check against expected value
                    if !e.is_nan() {
                        let diff_expected = (y - e).abs();
                        prop_assert!(
                            diff_expected < 1e-9,
                            "Value mismatch at idx {}: got {} expected {} (diff: {})",
                            i,
                            y,
                            e,
                            diff_expected
                        );
                    }
                } else {
                    prop_assert!(
                        false,
                        "NaN mismatch at idx {}: kernel={}, scalar={}",
                        i,
                        y,
                        r
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_vpt_tests!(
        check_vpt_basic_candles,
        check_vpt_basic_slices,
        check_vpt_not_enough_data,
        check_vpt_empty_data,
        check_vpt_all_nan,
        check_vpt_accuracy_from_csv,
        check_vpt_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vpt_tests!(check_vpt_property);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various data sources since VPT has no parameters
        let test_sources = vec!["close", "open", "high", "low"];

        for (src_idx, &source) in test_sources.iter().enumerate() {
            let output = VptBatchBuilder::new()
                .kernel(kernel)
                .apply_candles(&c, source)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Source {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Source {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Source {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
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
                    let kernel = detect_best_batch_kernel();
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), kernel);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_no_poison);
}
