//! # Rate of Change (ROC)
//!
//! Measures the percentage change in price between the current value and the value `period` bars ago.
//! Implements kernel auto-detection and AVX2/AVX512 stubs for SIMD compatibility, with a fully
//! featured builder/batch/stream API and input validation parity with alma.rs.
//!
//! ## Parameters
//! - **period**: Lookback window (defaults to 9)
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **values**: Vector of ROC percentage values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - **SIMD**: AVX2/AVX512 single-series kernels implemented; batch SIMD rows currently delegate to scalar.
//! - **Streaming**: Implemented with O(1) update performance (circular buffer)
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations
//! - Decision: Scalar is the reference path. SIMD is available and benchmarked; row-specific batch SIMD not attempted (little shared work). Keep selection to scalar where SIMD underperforms.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

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

// --- Data Types ---

#[derive(Debug, Clone)]
pub enum RocData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct RocParams {
    pub period: Option<usize>,
}

impl Default for RocParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct RocInput<'a> {
    pub data: RocData<'a>,
    pub params: RocParams,
}

impl<'a> AsRef<[f64]> for RocInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RocData::Slice(slice) => slice,
            RocData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> RocInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RocParams) -> Self {
        Self {
            data: RocData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slice(slice: &'a [f64], params: RocParams) -> Self {
        Self {
            data: RocData::Slice(slice),
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", RocParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

// --- Builder/Stream/Batch Structs ---

#[derive(Copy, Clone, Debug)]
pub struct RocBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for RocBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RocBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<RocOutput, RocError> {
        let p = RocParams {
            period: self.period,
        };
        let i = RocInput::from_candles(c, "close", p);
        roc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<RocOutput, RocError> {
        let p = RocParams {
            period: self.period,
        };
        let i = RocInput::from_slice(d, p);
        roc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<RocStream, RocError> {
        let p = RocParams {
            period: self.period,
        };
        RocStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum RocError {
    #[error("roc: Input data slice is empty.")]
    EmptyData,
    #[error("roc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("roc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("roc: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn roc(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn roc_prepare<'a>(
    input: &'a RocInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), RocError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(RocError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(RocError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(RocError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    // ROC is memory-bound and sees no consistent SIMD wins across CPUs.
    // Prefer Scalar for Auto to avoid regressions; explicit SIMD remains for benchmarking.
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    Ok((data, period, first, chosen))
}

#[inline(always)]
fn roc_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
    unsafe {
        match kernel {
            Kernel::Scalar => roc_scalar(data, period, first, out),
            Kernel::ScalarBatch => roc_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => roc_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch => {
                roc_row_avx2(data, first, period, 0, std::ptr::null(), 0.0, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => roc_avx512(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch => {
                roc_row_avx512(data, first, period, 0, std::ptr::null(), 0.0, out)
            }
            _ => unreachable!(),
        }
    }
}

pub fn roc_with_kernel(input: &RocInput, kernel: Kernel) -> Result<RocOutput, RocError> {
    let (data, period, first, chosen) = roc_prepare(input, kernel)?;
    // ROC first valid index is first + period
    let mut out = alloc_with_nan_prefix(data.len(), first + period);
    roc_compute_into(data, period, first, chosen, &mut out);
    Ok(RocOutput { values: out })
}

// --- Indicator Functions ---

#[inline(always)]
pub unsafe fn roc_indicator(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Auto)
}
#[inline(always)]
pub unsafe fn roc_indicator_with_kernel(
    input: &RocInput,
    k: Kernel,
) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, k)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx512(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx512)
}
#[inline(always)]
pub unsafe fn roc_indicator_scalar(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Scalar)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx2(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx2)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx512_short(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx512)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx512_long(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx512)
}

// --- Core Scalar & SIMD ---

#[inline(always)]
pub fn roc_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let start = first + period;
    // Safe, bounds-check-minimized iteration using slice zips
    let dst = &mut out[start..];
    let curr = &data[start..];
    let prev = &data[first..(len - period)];
    for ((d, &c), &p) in dst.iter_mut().zip(curr.iter()).zip(prev.iter()) {
        if p == 0.0 || p.is_nan() {
            *d = 0.0;
        } else {
            // Use mul_add to enable FMA where available: (c/p)*100 - 100
            *d = (c / p).mul_add(100.0, -100.0);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn roc_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // Process 4 lanes at a time using AVX2
    let len = data.len();
    let start = first + period;
    if start >= len { return; }

    let n = len - start;
    let base_curr = data.as_ptr().add(start);
    let base_prev = data.as_ptr().add(first);
    let base_out = out.as_mut_ptr().add(start);

    let v_zero = _mm256_set1_pd(0.0);
    let v_m100 = _mm256_set1_pd(-100.0);
    let v_100 = _mm256_set1_pd(100.0);

    let mut i = 0usize;
    while i + 4 <= n {
        let c = _mm256_loadu_pd(base_curr.add(i));
        let p = _mm256_loadu_pd(base_prev.add(i));

        // invalid when p == 0.0 or p is NaN
        let mask_zero = _mm256_cmp_pd(p, v_zero, _CMP_EQ_OQ);
        let mask_nan = _mm256_cmp_pd(p, p, _CMP_UNORD_Q);
        let mask_invalid = _mm256_or_pd(mask_zero, mask_nan);

        // r = (c / p)*100 - 100, using FMA for the mul-add part
        let div = _mm256_div_pd(c, p);
        let res = _mm256_fmadd_pd(div, v_100, v_m100);

        // Blend: if invalid -> 0.0 else res
        let blended = _mm256_blendv_pd(res, v_zero, mask_invalid);
        _mm256_storeu_pd(base_out.add(i), blended);
        i += 4;
    }

    // Scalar tail
    while i < n {
        let p = *base_prev.add(i);
        let c = *base_curr.add(i);
        *base_out.add(i) = if p == 0.0 || p.is_nan() { 0.0 } else { ((c / p) - 1.0) * 100.0 };
        i += 1;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn roc_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // Process 8 lanes at a time using AVX-512
    let len = data.len();
    let start = first + period;
    if start >= len { return; }

    let n = len - start;
    let base_curr = data.as_ptr().add(start);
    let base_prev = data.as_ptr().add(first);
    let base_out = out.as_mut_ptr().add(start);

    let v_zero = _mm512_set1_pd(0.0);
    let v_m100 = _mm512_set1_pd(-100.0);
    let v_100 = _mm512_set1_pd(100.0);

    let mut i = 0usize;
    while i + 8 <= n {
        let c = _mm512_loadu_pd(base_curr.add(i));
        let p = _mm512_loadu_pd(base_prev.add(i));

        // invalid when p == 0.0 or p is NaN
        let k_zero = _mm512_cmp_pd_mask(p, v_zero, _CMP_EQ_OQ);
        let k_nan = _mm512_cmp_pd_mask(p, p, _CMP_UNORD_Q);
        let k_invalid = k_zero | k_nan;

        // r = (c / p)*100 - 100
        let div = _mm512_div_pd(c, p);
        let res = _mm512_fmadd_pd(div, v_100, v_m100);

        // zero for invalid lanes
        let blended = _mm512_mask_mov_pd(res, k_invalid, v_zero);
        _mm512_storeu_pd(base_out.add(i), blended);
        i += 8;
    }

    // Scalar tail
    while i < n {
        let p = *base_prev.add(i);
        let c = *base_curr.add(i);
        *base_out.add(i) = if p == 0.0 || p.is_nan() { 0.0 } else { ((c / p) - 1.0) * 100.0 };
        i += 1;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn roc_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn roc_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    roc_scalar(data, period, first, out)
}

// --- Row/Batch Parity ---

#[inline(always)]
pub unsafe fn roc_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    // Optimized row kernel mirroring roc_scalar semantics with pointer arithmetic.
    let len = data.len();
    let start = first + period;
    if start >= len {
        return;
    }

    let base_ptr = data.as_ptr();
    let prev_ptr = base_ptr.add(first);
    let curr_ptr = base_ptr.add(start);
    let dst_ptr = out.as_mut_ptr().add(start);

    let n = len - start;
    let mut i = 0usize;

    // Unroll by 4
    while i + 4 <= n {
        let p0 = *prev_ptr.add(i + 0);
        let p1 = *prev_ptr.add(i + 1);
        let p2 = *prev_ptr.add(i + 2);
        let p3 = *prev_ptr.add(i + 3);

        let c0 = *curr_ptr.add(i + 0);
        let c1 = *curr_ptr.add(i + 1);
        let c2 = *curr_ptr.add(i + 2);
        let c3 = *curr_ptr.add(i + 3);

        *dst_ptr.add(i + 0) = if p0 == 0.0 || p0.is_nan() { 0.0 } else { ((c0 / p0) - 1.0) * 100.0 };
        *dst_ptr.add(i + 1) = if p1 == 0.0 || p1.is_nan() { 0.0 } else { ((c1 / p1) - 1.0) * 100.0 };
        *dst_ptr.add(i + 2) = if p2 == 0.0 || p2.is_nan() { 0.0 } else { ((c2 / p2) - 1.0) * 100.0 };
        *dst_ptr.add(i + 3) = if p3 == 0.0 || p3.is_nan() { 0.0 } else { ((c3 / p3) - 1.0) * 100.0 };

        i += 4;
    }

    // Tail
    while i < n {
        let p = *prev_ptr.add(i);
        let c = *curr_ptr.add(i);
        *dst_ptr.add(i) = if p == 0.0 || p.is_nan() { 0.0 } else { ((c / p) - 1.0) * 100.0 };
        i += 1;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_row_scalar(data, first, period, _stride, _weights, _inv_n, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_row_scalar(data, first, period, _stride, _weights, _inv_n, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_row_scalar(data, first, period, _stride, _weights, _inv_n, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_row_scalar(data, first, period, _stride, _weights, _inv_n, out)
}

// --- Stream API ---

#[derive(Debug, Clone)]
pub struct RocStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    count: usize,
}

impl RocStream {
    pub fn try_new(params: RocParams) -> Result<Self, RocError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(RocError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.count += 1;

        // We need at least period+1 values before we can calculate ROC
        // (current value plus period previous values)
        if self.count <= self.period {
            // Store the new value and advance
            self.buffer[self.head] = value;
            self.head = (self.head + 1) % self.period;
            return None;
        }

        // Get the value that's about to be replaced (period values ago)
        let old_value = self.buffer[self.head];

        // Store the new value
        self.buffer[self.head] = value;

        // Move head forward
        self.head = (self.head + 1) % self.period;

        // Calculate ROC
        if old_value.is_nan() || old_value == 0.0 {
            Some(0.0)
        } else {
            Some(((value / old_value) - 1.0) * 100.0)
        }
    }
}

// --- Batch API ---

#[derive(Clone, Debug)]
pub struct RocBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for RocBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RocBatchBuilder {
    range: RocBatchRange,
    kernel: Kernel,
}

impl RocBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<RocBatchOutput, RocError> {
        roc_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RocBatchOutput, RocError> {
        RocBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RocBatchOutput, RocError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<RocBatchOutput, RocError> {
        RocBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn roc_batch_with_kernel(
    data: &[f64],
    sweep: &RocBatchRange,
    k: Kernel,
) -> Result<RocBatchOutput, RocError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(RocError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    roc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RocBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<RocParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RocBatchOutput {
    pub fn row_for_params(&self, p: &RocParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
    }
    pub fn values_for(&self, p: &RocParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub(crate) fn expand_grid(r: &RocBatchRange) -> Vec<RocParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![RocParams {
            period: Some(start),
        }];
    }
    (start..=end)
        .step_by(step)
        .map(|p| RocParams { period: Some(p) })
        .collect()
}

#[inline(always)]
pub fn roc_batch_slice(
    data: &[f64],
    sweep: &RocBatchRange,
    kern: Kernel,
) -> Result<RocBatchOutput, RocError> {
    roc_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn roc_batch_par_slice(
    data: &[f64],
    sweep: &RocBatchRange,
    kern: Kernel,
) -> Result<RocBatchOutput, RocError> {
    roc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn roc_batch_inner(
    data: &[f64],
    sweep: &RocBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RocBatchOutput, RocError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RocError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(RocError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Use uninitialized memory for better performance
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => {
                roc_row_scalar(data, first, period, 0, std::ptr::null(), 0.0, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => roc_row_avx2(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                roc_row_avx512(data, first, period, 0, std::ptr::null(), 0.0, out_row)
            }
            _ => unreachable!(),
        }
    };

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

    // Convert uninitialized memory to Vec
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(RocBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn roc_batch_inner_into(
    data: &[f64],
    sweep: &RocBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<RocParams>, RocError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RocError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(RocError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // 1) View output as uninitialized and write NaN warm prefixes via helper
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    // 2) Row worker: write valid region only
    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let dst: &mut [f64] =
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
        match kern {
            Kernel::Scalar => roc_row_scalar(data, first, period, 0, std::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => roc_row_avx2(data, first, period, 0, std::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => roc_row_avx512(data, first, period, 0, std::ptr::null(), 0.0, dst),
            _ => unreachable!(),
        }
    };

    // 3) Iterate rows
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_mu)| do_row(row, row_mu));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, row_mu);
            }
        }
    } else {
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    }

    Ok(combos)
}

// --- Python Bindings ---

#[cfg(feature = "python")]
#[pyfunction(name = "roc")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn roc_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = RocParams {
        period: Some(period),
    };
    let input = RocInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| roc_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "roc_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn roc_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = RocBatchRange {
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

            roc_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
#[pyclass(name = "RocStream")]
pub struct RocStreamPy {
    inner: RocStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RocStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = RocParams {
            period: Some(period),
        };
        let inner = RocStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(RocStreamPy { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// --- WASM Bindings ---

/// Write ROC values directly to output slice - no allocations
#[inline]
pub fn roc_into_slice(dst: &mut [f64], input: &RocInput, kern: Kernel) -> Result<(), RocError> {
    let (data, period, first, chosen) = roc_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(RocError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    let warmup_end = first + period;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    roc_compute_into(data, period, first, chosen, dst);
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn roc_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = RocParams {
        period: Some(period),
    };
    let input = RocInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    roc_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn roc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn roc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn roc_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = RocParams {
            period: Some(period),
        };
        let input = RocInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // Handle aliasing - use temporary buffer
            let mut temp = vec![0.0; len];
            roc_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            roc_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RocBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RocBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<RocParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn roc_batch(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: RocBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let batch_range = RocBatchRange {
        period: config.period_range,
    };

    let result = roc_batch_with_kernel(data, &batch_range, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let output = RocBatchJsOutput {
        values: result.values,
        combos: result.combos,
        rows: result.rows,
        cols: result.cols,
    };

    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_roc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = RocParams { period: None };
        let input_default = RocInput::from_candles(&candles, "close", default_params);
        let output_default = roc_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = RocParams { period: Some(14) };
        let input_period_14 = RocInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = roc_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = RocParams { period: Some(20) };
        let input_custom = RocInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = roc_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());

        Ok(())
    }

    fn check_roc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let close_prices = &candles.close;
        let params = RocParams { period: Some(10) };
        let input = RocInput::from_candles(&candles, "close", params);
        let roc_result = roc_with_kernel(&input, kernel)?;

        assert_eq!(roc_result.values.len(), close_prices.len());

        let expected_last_five_roc = [
            -0.22551709049294377,
            -0.5561903481650754,
            -0.32752013235864963,
            -0.49454153980722504,
            -1.5045927020536976,
        ];
        assert!(roc_result.values.len() >= 5);
        let start_index = roc_result.values.len() - 5;
        let result_last_five_roc = &roc_result.values[start_index..];
        for (i, &value) in result_last_five_roc.iter().enumerate() {
            let expected_value = expected_last_five_roc[i];
            assert!(
                (value - expected_value).abs() < 1e-7,
                "[{}] ROC mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        let period = input.get_period();
        for i in 0..(period - 1) {
            assert!(roc_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_roc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = RocInput::with_default_candles(&candles);
        match input.data {
            RocData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected RocData::Candles"),
        }
        let output = roc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_roc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = RocParams { period: Some(0) };
        let input = RocInput::from_slice(&input_data, params);
        let res = roc_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_roc_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = RocParams { period: Some(10) };
        let input = RocInput::from_slice(&data_small, params);
        let res = roc_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_roc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = RocParams { period: Some(9) };
        let input = RocInput::from_slice(&single_point, params);
        let res = roc_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_roc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = RocParams { period: Some(14) };
        let first_input = RocInput::from_candles(&candles, "close", first_params);
        let first_result = roc_with_kernel(&first_input, kernel)?;

        let second_params = RocParams { period: Some(14) };
        let second_input = RocInput::from_slice(&first_result.values, second_params);
        let second_result = roc_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index 28, found NaN at {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_roc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = RocInput::from_candles(&candles, "close", RocParams { period: Some(9) });
        let res = roc_with_kernel(&input, kernel)?;
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

    fn check_roc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 9;

        let input = RocInput::from_candles(
            &candles,
            "close",
            RocParams {
                period: Some(period),
            },
        );
        let batch_output = roc_with_kernel(&input, kernel)?.values;

        let mut stream = RocStream::try_new(RocParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] ROC streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_roc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            RocParams::default(),            // period: 9
            RocParams { period: Some(2) },   // minimum viable
            RocParams { period: Some(5) },   // small
            RocParams { period: Some(7) },   // small
            RocParams { period: Some(9) },   // small
            RocParams { period: Some(14) },  // medium
            RocParams { period: Some(20) },  // medium
            RocParams { period: Some(30) },  // medium
            RocParams { period: Some(50) },  // large
            RocParams { period: Some(100) }, // large
            RocParams { period: Some(200) }, // very large
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = RocInput::from_candles(&candles, "close", params.clone());
            let output = roc_with_kernel(&input, kernel)?;

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
                        params.period.unwrap_or(9),
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
                        params.period.unwrap_or(9),
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
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_roc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_roc_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy to generate diverse test data
        let strat = (2usize..=100).prop_flat_map(|period| {
            prop_oneof![
                // Normal random data
                (
                    prop::collection::vec(
                        (1f64..1e6f64)
                            .prop_filter("finite positive", |x| x.is_finite() && *x > 0.0),
                        period..400,
                    ),
                    Just(period),
                ),
                // Data with some constant sequences
                (
                    prop::collection::vec(
                        prop_oneof![
                            (1f64..1000f64).prop_filter("finite", |x| x.is_finite()),
                            Just(100.0), // Constant value
                        ],
                        period..400,
                    ),
                    Just(period),
                ),
                // Monotonic increasing data
                (
                    (100f64..10000f64, 0.01f64..0.1f64).prop_map(move |(start, step)| {
                        let len = period + (400 - period) / 2; // Average length
                        (0..len)
                            .map(|i| start + (i as f64) * step)
                            .collect::<Vec<_>>()
                    }),
                    Just(period),
                ),
                // Monotonic decreasing data
                (
                    (10000f64..100000f64, 0.01f64..0.1f64).prop_map(move |(start, step)| {
                        let len = period + (400 - period) / 2; // Average length
                        (0..len)
                            .map(|i| start - (i as f64) * step)
                            .collect::<Vec<_>>()
                    }),
                    Just(period),
                ),
            ]
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = RocParams {
                period: Some(period),
            };
            let input = RocInput::from_slice(&data, params);

            let RocOutput { values: out } = roc_with_kernel(&input, kernel)?;
            let RocOutput { values: ref_out } = roc_with_kernel(&input, Kernel::Scalar)?;

            // Check output length matches input
            prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

            // Check warmup period - first 'period' values should be NaN
            for i in 0..period {
                prop_assert!(
                    out[i].is_nan(),
                    "Expected NaN during warmup at index {}, got {}",
                    i,
                    out[i]
                );
            }

            // Check mathematical correctness after warmup
            for i in period..data.len() {
                let current = data[i];
                let previous = data[i - period];
                let roc_val = out[i];

                // Expected ROC calculation
                let expected_roc = if previous == 0.0 || previous.is_nan() {
                    0.0
                } else {
                    ((current / previous) - 1.0) * 100.0
                };

                // Verify calculation correctness
                if !roc_val.is_nan() {
                    prop_assert!(
							(roc_val - expected_roc).abs() < 1e-9,
							"ROC calculation mismatch at {}: got {}, expected {} (current={}, previous={})",
							i, roc_val, expected_roc, current, previous
						);

                    // Sign properties
                    if current > previous && previous > 0.0 {
                        prop_assert!(
								roc_val > -1e-9,
								"ROC should be positive when current > previous at {}: roc={}, current={}, previous={}",
								i, roc_val, current, previous
							);
                    }
                    if current < previous && previous > 0.0 {
                        prop_assert!(
								roc_val < 1e-9,
								"ROC should be negative when current < previous at {}: roc={}, current={}, previous={}",
								i, roc_val, current, previous
							);
                    }
                    if (current - previous).abs() < 1e-12 && previous > 0.0 {
                        prop_assert!(
								roc_val.abs() < 1e-9,
								"ROC should be ~0 when current ≈ previous at {}: roc={}, current={}, previous={}",
								i, roc_val, current, previous
							);
                    }
                }
            }

            // Check constant data property
            let is_constant = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
            if is_constant && data.len() > period {
                for i in period..data.len() {
                    if !out[i].is_nan() {
                        prop_assert!(
                            out[i].abs() < 1e-9,
                            "ROC of constant data should be 0 at {}: got {}",
                            i,
                            out[i]
                        );
                    }
                }
            }

            // Check monotonic properties
            let is_monotonic_increasing = data.windows(2).all(|w| w[1] >= w[0]);
            let is_monotonic_decreasing = data.windows(2).all(|w| w[1] <= w[0]);

            if is_monotonic_increasing && !is_constant {
                // For strictly increasing data, ROC should be positive
                for i in period..data.len() {
                    if !out[i].is_nan() && data[i] > data[i - period] {
                        prop_assert!(
                            out[i] > -1e-9,
                            "ROC should be positive for increasing data at {}: got {}",
                            i,
                            out[i]
                        );
                    }
                }
            }

            if is_monotonic_decreasing && !is_constant {
                // For strictly decreasing data, ROC should be negative
                for i in period..data.len() {
                    if !out[i].is_nan() && data[i] < data[i - period] {
                        prop_assert!(
                            out[i] < 1e-9,
                            "ROC should be negative for decreasing data at {}: got {}",
                            i,
                            out[i]
                        );
                    }
                }
            }

            // Verify kernel consistency
            prop_assert_eq!(out.len(), ref_out.len(), "Kernel output length mismatch");

            for i in 0..out.len() {
                let y = out[i];
                let r = ref_out[i];

                if !y.is_finite() || !r.is_finite() {
                    prop_assert!(
                        y.to_bits() == r.to_bits(),
                        "NaN/Inf mismatch at {}: {} vs {}",
                        i,
                        y,
                        r
                    );
                } else {
                    let tolerance = 1e-9;
                    prop_assert!(
                        (y - r).abs() <= tolerance,
                        "Kernel mismatch at {}: {} vs {}, diff={}",
                        i,
                        y,
                        r,
                        (y - r).abs()
                    );
                }
            }

            // Check for poison values in debug builds
            #[cfg(debug_assertions)]
            {
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        let bits = val.to_bits();
                        prop_assert_ne!(
                            bits,
                            0x11111111_11111111,
                            "Found alloc_with_nan_prefix poison at {}",
                            i
                        );
                        prop_assert_ne!(
                            bits,
                            0x22222222_22222222,
                            "Found init_matrix_prefixes poison at {}",
                            i
                        );
                        prop_assert_ne!(
                            bits,
                            0x33333333_33333333,
                            "Found make_uninit_matrix poison at {}",
                            i
                        );
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_roc_tests {
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
    generate_all_roc_tests!(
        check_roc_partial_params,
        check_roc_accuracy,
        check_roc_default_candles,
        check_roc_zero_period,
        check_roc_period_exceeds_length,
        check_roc_very_small_dataset,
        check_roc_reinput,
        check_roc_nan_handling,
        check_roc_streaming,
        check_roc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_roc_tests!(check_roc_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = RocBatchBuilder::new()
            .period_static(10) // Use period=10 to match expected values
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let test_params = RocParams { period: Some(10) };
        let row = output
            .values_for(&test_params)
            .expect("period=10 row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -0.22551709049294377,
            -0.5561903481650754,
            -0.32752013235864963,
            -0.49454153980722504,
            -1.5045927020536976,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
                "[{test}] period=10 row mismatch at idx {i}: {v} vs {expected:?}"
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
            (9, 9, 0),    // Single period (edge case)
            (14, 21, 7),  // Week-based periods
            (10, 50, 10), // Decade periods
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = RocBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

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
                        combo.period.unwrap_or(9)
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
                        combo.period.unwrap_or(9)
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
                        combo.period.unwrap_or(9)
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
