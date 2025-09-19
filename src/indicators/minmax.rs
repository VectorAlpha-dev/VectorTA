//! # MinMax (Local Extrema)
//!
//! Identifies local minima and maxima over a specified `order` range.
//! Provides SIMD and batch APIs, builder pattern, streaming, and robust input validation.
//!
//! ## Parameters
//! - **order**: Neighborhood range (defaults to 3)
//!
//! ## Returns
//! - **`Ok(MinmaxOutput)`** on success, containing local extrema and forward-filled values.
//! - **`Err(MinmaxError)`** on failure
//!
//! ## Developer Notes
//! - **AVX2**: Stub implementation - calls scalar function
//! - **AVX512**: Multiple stub functions (minmax_avx512_short, minmax_avx512_long) - all call scalar
//! - **Streaming**: O(n) for each update - scans entire window for min/max comparison (needs optimization)
//! - **Memory**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
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
use std::error::Error;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// --- DATA TYPES ---

#[derive(Debug, Clone)]
pub enum MinmaxData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MinmaxOutput {
    pub is_min: Vec<f64>,
    pub is_max: Vec<f64>,
    pub last_min: Vec<f64>,
    pub last_max: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MinmaxParams {
    pub order: Option<usize>,
}

impl Default for MinmaxParams {
    fn default() -> Self {
        Self { order: Some(3) }
    }
}

#[derive(Debug, Clone)]
pub struct MinmaxInput<'a> {
    pub data: MinmaxData<'a>,
    pub params: MinmaxParams,
}

impl<'a> MinmaxInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        params: MinmaxParams,
    ) -> Self {
        Self {
            data: MinmaxData::Candles {
                candles,
                high_src,
                low_src,
            },
            params,
        }
    }
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MinmaxParams) -> Self {
        Self {
            data: MinmaxData::Slices { high, low },
            params,
        }
    }
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "high", "low", MinmaxParams::default())
    }
    pub fn get_order(&self) -> usize {
        self.params.order.unwrap_or(3)
    }
}

// --- BUILDER ---

#[derive(Copy, Clone, Debug)]
pub struct MinmaxBuilder {
    order: Option<usize>,
    kernel: Kernel,
}

impl Default for MinmaxBuilder {
    fn default() -> Self {
        Self {
            order: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MinmaxBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn order(mut self, n: usize) -> Self {
        self.order = Some(n);
        self
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply(self, candles: &Candles) -> Result<MinmaxOutput, MinmaxError> {
        let params = MinmaxParams { order: self.order };
        let input = MinmaxInput::from_candles(candles, "high", "low", params);
        minmax_with_kernel(&input, self.kernel)
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MinmaxOutput, MinmaxError> {
        let params = MinmaxParams { order: self.order };
        let input = MinmaxInput::from_slices(high, low, params);
        minmax_with_kernel(&input, self.kernel)
    }
    pub fn into_stream(self) -> Result<MinmaxStream, MinmaxError> {
        let params = MinmaxParams { order: self.order };
        MinmaxStream::try_new(params)
    }
}

// --- ERRORS ---

#[derive(Debug, Error)]
pub enum MinmaxError {
    #[error("minmax: Empty data provided.")]
    EmptyData,
    #[error("minmax: Invalid order: order = {order}, data length = {data_len}")]
    InvalidOrder { order: usize, data_len: usize },
    #[error("minmax: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("minmax: All values are NaN.")]
    AllValuesNaN,
}

// --- KERNEL API ---

#[inline]
pub fn minmax(input: &MinmaxInput) -> Result<MinmaxOutput, MinmaxError> {
    minmax_with_kernel(input, Kernel::Auto)
}

/// Write directly to output slices - no allocations
#[inline]
pub fn minmax_into_slice(
    is_min_dst: &mut [f64],
    is_max_dst: &mut [f64],
    last_min_dst: &mut [f64],
    last_max_dst: &mut [f64],
    input: &MinmaxInput,
    kern: Kernel,
) -> Result<(), MinmaxError> {
    let (high, low) = match &input.data {
        MinmaxData::Candles {
            candles,
            high_src,
            low_src,
        } => {
            let h = source_type(candles, high_src);
            let l = source_type(candles, low_src);
            (h, l)
        }
        MinmaxData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }

    let len = high.len();
    if is_min_dst.len() != len
        || is_max_dst.len() != len
        || last_min_dst.len() != len
        || last_max_dst.len() != len
    {
        return Err(MinmaxError::InvalidOrder {
            order: is_min_dst.len(),
            data_len: len,
        });
    }

    let order = input.get_order();
    if order == 0 || order > len {
        return Err(MinmaxError::InvalidOrder {
            order,
            data_len: len,
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
        .ok_or(MinmaxError::AllValuesNaN)?;

    if (len - first_valid_idx) < order {
        return Err(MinmaxError::NotEnoughValidData {
            needed: order,
            valid: len - first_valid_idx,
        });
    }

    // only warm prefix
    for i in 0..first_valid_idx {
        is_min_dst[i] = f64::NAN;
        is_max_dst[i] = f64::NAN;
        last_min_dst[i] = f64::NAN;
        last_max_dst[i] = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => minmax_scalar(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => minmax_avx2(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => minmax_avx512(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
            _ => minmax_scalar(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
        }
    }

    Ok(())
}

pub fn minmax_with_kernel(
    input: &MinmaxInput,
    kernel: Kernel,
) -> Result<MinmaxOutput, MinmaxError> {
    let (high, low) = match &input.data {
        MinmaxData::Candles {
            candles,
            high_src,
            low_src,
        } => {
            let h = source_type(candles, high_src);
            let l = source_type(candles, low_src);
            (h, l)
        }
        MinmaxData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }
    let len = high.len();
    let order = input.get_order();
    if order == 0 || order > len {
        return Err(MinmaxError::InvalidOrder {
            order,
            data_len: len,
        });
    }
    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
        .ok_or(MinmaxError::AllValuesNaN)?;

    if (len - first_valid_idx) < order {
        return Err(MinmaxError::NotEnoughValidData {
            needed: order,
            valid: len - first_valid_idx,
        });
    }
    // NaN only for the warm prefix; everything else will be written once in the scalar
    let mut is_min = alloc_with_nan_prefix(len, first_valid_idx);
    let mut is_max = alloc_with_nan_prefix(len, first_valid_idx);
    let mut last_min = alloc_with_nan_prefix(len, first_valid_idx);
    let mut last_max = alloc_with_nan_prefix(len, first_valid_idx);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => minmax_scalar(
                high,
                low,
                order,
                first_valid_idx,
                &mut is_min,
                &mut is_max,
                &mut last_min,
                &mut last_max,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => minmax_avx2(
                high,
                low,
                order,
                first_valid_idx,
                &mut is_min,
                &mut is_max,
                &mut last_min,
                &mut last_max,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => minmax_avx512(
                high,
                low,
                order,
                first_valid_idx,
                &mut is_min,
                &mut is_max,
                &mut last_min,
                &mut last_max,
            ),
            _ => unreachable!(),
        }
    }
    Ok(MinmaxOutput {
        is_min,
        is_max,
        last_min,
        last_max,
    })
}

// --- SCALAR LOGIC ---

#[inline]
pub fn minmax_scalar(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    let len = high.len();

    // prefix [0..first_valid_idx)
    for i in 0..first_valid_idx {
        is_min[i] = f64::NAN;
        is_max[i] = f64::NAN;
        last_min[i] = f64::NAN;
        last_max[i] = f64::NAN;
    }

    let mut last_min_val = f64::NAN;
    let mut last_max_val = f64::NAN;

    for i in first_valid_idx..len {
        let ch = high[i];
        let cl = low[i];

        // default to NaN at this index
        let mut min_here = f64::NAN;
        let mut max_here = f64::NAN;

        // only consider if the window is fully inside bounds and center is finite
        if i >= order && i + order < len && ch.is_finite() && cl.is_finite() {
            // neighbors must all be finite
            let mut less_than_neighbors = true;
            let mut greater_than_neighbors = true;

            for o in 1..=order {
                let lh = high[i - o];
                let rh = high[i + o];
                let ll = low[i - o];
                let rl = low[i + o];

                if !(ll.is_finite() && rl.is_finite()) {
                    less_than_neighbors = false;
                } else if !(cl < ll && cl < rl) {
                    less_than_neighbors = false;
                }

                if !(lh.is_finite() && rh.is_finite()) {
                    greater_than_neighbors = false;
                } else if !(ch > lh && ch > rh) {
                    greater_than_neighbors = false;
                }

                if !less_than_neighbors && !greater_than_neighbors {
                    break;
                }
            }

            if less_than_neighbors {
                min_here = cl;
            }
            if greater_than_neighbors {
                max_here = ch;
            }
        }

        // write this row fully
        is_min[i] = min_here;
        is_max[i] = max_here;

        if min_here.is_finite() {
            last_min_val = min_here;
        }
        if max_here.is_finite() {
            last_max_val = max_here;
        }

        last_min[i] = last_min_val;
        last_max[i] = last_max_val;
    }
}

// --- AVX2/AVX512 STUBS ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx2(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid_idx,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    if order <= 16 {
        minmax_avx512_short(
            high,
            low,
            order,
            first_valid_idx,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    } else {
        minmax_avx512_long(
            high,
            low,
            order,
            first_valid_idx,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512_short(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid_idx,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512_long(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid_idx,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

// --- STREAMING ---

#[derive(Debug, Clone)]
pub struct MinmaxStream {
    order: usize,
    high_buf: Vec<f64>,
    low_buf: Vec<f64>,
    idx: usize,
    filled: bool,
    len: usize,
    last_min: f64,
    last_max: f64,
}

impl MinmaxStream {
    pub fn try_new(params: MinmaxParams) -> Result<Self, MinmaxError> {
        let order = params.order.unwrap_or(3);
        if order == 0 {
            return Err(MinmaxError::InvalidOrder { order, data_len: 0 });
        }
        Ok(Self {
            order,
            high_buf: vec![f64::NAN; order * 2 + 1],
            low_buf: vec![f64::NAN; order * 2 + 1],
            idx: 0,
            filled: false,
            len: order * 2 + 1,
            last_min: f64::NAN,
            last_max: f64::NAN,
        })
    }
    pub fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, f64, f64) {
        self.high_buf[self.idx] = high;
        self.low_buf[self.idx] = low;
        self.idx = (self.idx + 1) % self.len;

        if !self.filled && self.idx == 0 {
            self.filled = true;
        }
        if !self.filled {
            return (None, None, self.last_min, self.last_max);
        }
        let center = (self.idx + self.len - self.order) % self.len;
        let center_high = self.high_buf[center];
        let center_low = self.low_buf[center];

        let mut is_min = true;
        let mut is_max = true;

        for o in 1..=self.order {
            let l_idx = (center + self.len - o) % self.len;
            let r_idx = (center + o) % self.len;
            if center_low >= self.low_buf[l_idx] || center_low >= self.low_buf[r_idx] {
                is_min = false;
            }
            if center_high <= self.high_buf[l_idx] || center_high <= self.high_buf[r_idx] {
                is_max = false;
            }
            if !is_min && !is_max {
                break;
            }
        }
        let min_val = if is_min { center_low } else { f64::NAN };
        let max_val = if is_max { center_high } else { f64::NAN };

        if !min_val.is_nan() {
            self.last_min = min_val;
        }
        if !max_val.is_nan() {
            self.last_max = max_val;
        }
        (
            if is_min { Some(center_low) } else { None },
            if is_max { Some(center_high) } else { None },
            self.last_min,
            self.last_max,
        )
    }
}

// --- BATCH API ---

#[derive(Clone, Debug)]
pub struct MinmaxBatchRange {
    pub order: (usize, usize, usize),
}

impl Default for MinmaxBatchRange {
    fn default() -> Self {
        Self { order: (3, 20, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MinmaxBatchBuilder {
    range: MinmaxBatchRange,
    kernel: Kernel,
}

impl MinmaxBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn order_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.order = (start, end, step);
        self
    }
    pub fn order_static(mut self, o: usize) -> Self {
        self.range.order = (o, o, 0);
        self
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MinmaxBatchOutput, MinmaxError> {
        minmax_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<MinmaxBatchOutput, MinmaxError> {
        MinmaxBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<MinmaxBatchOutput, MinmaxError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MinmaxBatchOutput, MinmaxError> {
        MinmaxBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub fn minmax_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    k: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MinmaxError::InvalidOrder {
                order: 0,
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
    minmax_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MinmaxBatchOutput {
    pub is_min: Vec<f64>,
    pub is_max: Vec<f64>,
    pub last_min: Vec<f64>,
    pub last_max: Vec<f64>,
    pub combos: Vec<MinmaxParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MinmaxBatchOutput {
    pub fn row_for_params(&self, p: &MinmaxParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.order.unwrap_or(3) == p.order.unwrap_or(3))
    }
    pub fn is_min_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.is_min[start..start + self.cols]
        })
    }
    pub fn is_max_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.is_max[start..start + self.cols]
        })
    }
    pub fn last_min_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.last_min[start..start + self.cols]
        })
    }
    pub fn last_max_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.last_max[start..start + self.cols]
        })
    }
}

// --- BATCH EXPANSION ---

#[inline(always)]
fn expand_grid(r: &MinmaxBatchRange) -> Vec<MinmaxParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let orders = axis_usize(r.order);
    let mut out = Vec::with_capacity(orders.len());
    for &o in &orders {
        out.push(MinmaxParams { order: Some(o) });
    }
    out
}

#[inline(always)]
pub fn minmax_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    minmax_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn minmax_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    minmax_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn minmax_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: 0,
        });
    }

    let len = high.len();
    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
        .ok_or(MinmaxError::AllValuesNaN)?;
    let max_o = combos.iter().map(|c| c.order.unwrap()).max().unwrap();
    if len - first < max_o {
        return Err(MinmaxError::NotEnoughValidData {
            needed: max_o,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;

    // 4 matrices, uninitialized
    let mut min_mu = make_uninit_matrix(rows, cols);
    let mut max_mu = make_uninit_matrix(rows, cols);
    let mut lmin_mu = make_uninit_matrix(rows, cols);
    let mut lmax_mu = make_uninit_matrix(rows, cols);

    // warm prefix per row
    let warm = vec![first; rows];
    init_matrix_prefixes(&mut min_mu, cols, &warm);
    init_matrix_prefixes(&mut max_mu, cols, &warm);
    init_matrix_prefixes(&mut lmin_mu, cols, &warm);
    init_matrix_prefixes(&mut lmax_mu, cols, &warm);

    // flatten to &mut [f64] without copies
    let mut min_guard = core::mem::ManuallyDrop::new(min_mu);
    let mut max_guard = core::mem::ManuallyDrop::new(max_mu);
    let mut lmin_guard = core::mem::ManuallyDrop::new(lmin_mu);
    let mut lmax_guard = core::mem::ManuallyDrop::new(lmax_mu);

    let is_min: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(min_guard.as_mut_ptr() as *mut f64, min_guard.len())
    };
    let is_max: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(max_guard.as_mut_ptr() as *mut f64, max_guard.len())
    };
    let last_min: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lmin_guard.as_mut_ptr() as *mut f64, lmin_guard.len())
    };
    let last_max: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lmax_guard.as_mut_ptr() as *mut f64, lmax_guard.len())
    };

    // row worker
    let do_row = |row: usize,
                  out_min: &mut [f64],
                  out_max: &mut [f64],
                  out_lmin: &mut [f64],
                  out_lmax: &mut [f64]| unsafe {
        let o = combos[row].order.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                minmax_row_avx2(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                minmax_row_avx512(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            _ => minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            is_min
                .par_chunks_mut(cols)
                .zip(is_max.par_chunks_mut(cols))
                .zip(
                    last_min
                        .par_chunks_mut(cols)
                        .zip(last_max.par_chunks_mut(cols)),
                )
                .enumerate()
                .for_each(|(row, ((m, x), (lm, lx)))| do_row(row, m, x, lm, lx));
        }
        #[cfg(target_arch = "wasm32")]
        for (row, ((m, x), (lm, lx))) in is_min
            .chunks_mut(cols)
            .zip(is_max.chunks_mut(cols))
            .zip(last_min.chunks_mut(cols).zip(last_max.chunks_mut(cols)))
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    } else {
        for (row, ((m, x), (lm, lx))) in is_min
            .chunks_mut(cols)
            .zip(is_max.chunks_mut(cols))
            .zip(last_min.chunks_mut(cols).zip(last_max.chunks_mut(cols)))
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    }

    // reconstitute Vec<f64> without copies
    let is_min = unsafe {
        Vec::from_raw_parts(
            min_guard.as_mut_ptr() as *mut f64,
            min_guard.len(),
            min_guard.capacity(),
        )
    };
    let is_max = unsafe {
        Vec::from_raw_parts(
            max_guard.as_mut_ptr() as *mut f64,
            max_guard.len(),
            max_guard.capacity(),
        )
    };
    let last_min = unsafe {
        Vec::from_raw_parts(
            lmin_guard.as_mut_ptr() as *mut f64,
            lmin_guard.len(),
            lmin_guard.capacity(),
        )
    };
    let last_max = unsafe {
        Vec::from_raw_parts(
            lmax_guard.as_mut_ptr() as *mut f64,
            lmax_guard.len(),
            lmax_guard.capacity(),
        )
    };

    Ok(MinmaxBatchOutput {
        is_min,
        is_max,
        last_min,
        last_max,
        combos,
        rows,
        cols,
    })
}

// Direct buffer writing variant for Python/WASM bindings
#[inline(always)]
fn minmax_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
    parallel: bool,
    is_min_out: &mut [f64],
    is_max_out: &mut [f64],
    last_min_out: &mut [f64],
    last_max_out: &mut [f64],
) -> Result<Vec<MinmaxParams>, MinmaxError> {
    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: 0,
        });
    }

    let len = high.len();
    let rows = combos.len();
    let cols = len;

    if is_min_out.len() != rows * cols
        || is_max_out.len() != rows * cols
        || last_min_out.len() != rows * cols
        || last_max_out.len() != rows * cols
    {
        return Err(MinmaxError::InvalidOrder {
            order: is_min_out.len(),
            data_len: rows * cols,
        });
    }

    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
        .ok_or(MinmaxError::AllValuesNaN)?;
    let max_o = combos.iter().map(|c| c.order.unwrap()).max().unwrap();
    if len - first < max_o {
        return Err(MinmaxError::NotEnoughValidData {
            needed: max_o,
            valid: len - first,
        });
    }

    // Treat raw outputs as MaybeUninit and use prefix helper
    let warm = vec![first; rows];
    let (min_mu, max_mu, lmin_mu, lmax_mu) = unsafe {
        (
            core::slice::from_raw_parts_mut(
                is_min_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                rows * cols,
            ),
            core::slice::from_raw_parts_mut(
                is_max_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                rows * cols,
            ),
            core::slice::from_raw_parts_mut(
                last_min_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                rows * cols,
            ),
            core::slice::from_raw_parts_mut(
                last_max_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                rows * cols,
            ),
        )
    };
    init_matrix_prefixes(min_mu, cols, &warm);
    init_matrix_prefixes(max_mu, cols, &warm);
    init_matrix_prefixes(lmin_mu, cols, &warm);
    init_matrix_prefixes(lmax_mu, cols, &warm);

    let do_row = |row: usize,
                  out_min: &mut [f64],
                  out_max: &mut [f64],
                  out_lmin: &mut [f64],
                  out_lmax: &mut [f64]| unsafe {
        let o = combos[row].order.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                minmax_row_avx2(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                minmax_row_avx512(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            _ => minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            is_min_out
                .par_chunks_mut(cols)
                .zip(is_max_out.par_chunks_mut(cols))
                .zip(
                    last_min_out
                        .par_chunks_mut(cols)
                        .zip(last_max_out.par_chunks_mut(cols)),
                )
                .enumerate()
                .for_each(|(row, ((m, x), (lm, lx)))| do_row(row, m, x, lm, lx));
        }
        #[cfg(target_arch = "wasm32")]
        for (row, ((m, x), (lm, lx))) in is_min_out
            .chunks_mut(cols)
            .zip(is_max_out.chunks_mut(cols))
            .zip(
                last_min_out
                    .chunks_mut(cols)
                    .zip(last_max_out.chunks_mut(cols)),
            )
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    } else {
        for (row, ((m, x), (lm, lx))) in is_min_out
            .chunks_mut(cols)
            .zip(is_max_out.chunks_mut(cols))
            .zip(
                last_min_out
                    .chunks_mut(cols)
                    .zip(last_max_out.chunks_mut(cols)),
            )
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    }

    Ok(combos)
}

// --- BATCH ROW VARIANTS ---

#[inline(always)]
pub unsafe fn minmax_row_scalar(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx2(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_row_scalar(
        high,
        low,
        first_valid,
        order,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    if order <= 16 {
        minmax_row_avx512_short(
            high,
            low,
            first_valid,
            order,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    } else {
        minmax_row_avx512_long(
            high,
            low,
            first_valid,
            order,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_row_scalar(
        high,
        low,
        first_valid,
        order,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_row_scalar(
        high,
        low,
        first_valid,
        order,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

// --- PYTHON BINDINGS ---

#[cfg(feature = "python")]
#[pyfunction(name = "minmax")]
#[pyo3(signature = (high, low, order, kernel=None))]
pub fn minmax_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    order: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MinmaxParams { order: Some(order) };
    let input = MinmaxInput::from_slices(high_slice, low_slice, params);

    let output = py
        .allow_threads(|| minmax_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        output.is_min.into_pyarray(py),
        output.is_max.into_pyarray(py),
        output.last_min.into_pyarray(py),
        output.last_max.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "MinmaxStream")]
pub struct MinmaxStreamPy {
    stream: MinmaxStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MinmaxStreamPy {
    #[new]
    fn new(order: usize) -> PyResult<Self> {
        let params = MinmaxParams { order: Some(order) };
        let stream =
            MinmaxStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MinmaxStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, f64, f64) {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "minmax_batch")]
#[pyo3(signature = (high, low, order_range, kernel=None))]
pub fn minmax_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    order_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = MinmaxBatchRange { order: order_range };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high_slice.len();

    // Pre-allocate arrays for batch operation
    let is_min_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let is_max_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let last_min_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let last_max_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

    let is_min_slice = unsafe { is_min_arr.as_slice_mut()? };
    let is_max_slice = unsafe { is_max_arr.as_slice_mut()? };
    let last_min_slice = unsafe { last_min_arr.as_slice_mut()? };
    let last_max_slice = unsafe { last_max_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // Map batch kernels to regular SIMD kernels
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };

            // Write directly to pre-allocated buffers
            minmax_batch_inner_into(
                high_slice,
                low_slice,
                &sweep,
                simd,
                true,
                is_min_slice,
                is_max_slice,
                last_min_slice,
                last_max_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("is_min", is_min_arr.reshape((rows, cols))?)?;
    dict.set_item("is_max", is_max_arr.reshape((rows, cols))?)?;
    dict.set_item("last_min", last_min_arr.reshape((rows, cols))?)?;
    dict.set_item("last_max", last_max_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "orders",
        combos
            .iter()
            .map(|p| p.order.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// --- WASM BINDINGS ---

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MinmaxResult {
    pub values: Vec<f64>, // [is_min..., is_max..., last_min..., last_max...]
    pub rows: usize,      // 4
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_js(high: &[f64], low: &[f64], order: usize) -> Result<JsValue, JsValue> {
    let input = MinmaxInput::from_slices(high, low, MinmaxParams { order: Some(order) });

    let out =
        minmax_with_kernel(&input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let len = high.len();
    let mut values = Vec::with_capacity(4 * len);
    values.extend_from_slice(&out.is_min);
    values.extend_from_slice(&out.is_max);
    values.extend_from_slice(&out.last_min);
    values.extend_from_slice(&out.last_max);

    let result = MinmaxResult {
        values,
        rows: 4,
        cols: len,
    };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    is_min_ptr: *mut f64,
    is_max_ptr: *mut f64,
    last_min_ptr: *mut f64,
    last_max_ptr: *mut f64,
    len: usize,
    order: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || is_min_ptr.is_null()
        || is_max_ptr.is_null()
        || last_min_ptr.is_null()
        || last_max_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to minmax_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        if order == 0 || order > len {
            return Err(JsValue::from_str("Invalid order"));
        }

        let params = MinmaxParams { order: Some(order) };
        let input = MinmaxInput::from_slices(high, low, params);

        // Check for aliasing between any input and output pointers
        let input_ptrs = [high_ptr as *const u8, low_ptr as *const u8];
        let output_ptrs = [
            is_min_ptr as *mut u8,
            is_max_ptr as *mut u8,
            last_min_ptr as *mut u8,
            last_max_ptr as *mut u8,
        ];

        let mut needs_temp = false;
        for &inp in &input_ptrs {
            for &out in &output_ptrs {
                if inp == out {
                    needs_temp = true;
                    break;
                }
            }
            if needs_temp {
                break;
            }
        }

        if needs_temp {
            // Use temporary buffers
            let mut temp_is_min = vec![0.0; len];
            let mut temp_is_max = vec![0.0; len];
            let mut temp_last_min = vec![0.0; len];
            let mut temp_last_max = vec![0.0; len];

            minmax_into_slice(
                &mut temp_is_min,
                &mut temp_is_max,
                &mut temp_last_min,
                &mut temp_last_max,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let is_min_out = std::slice::from_raw_parts_mut(is_min_ptr, len);
            let is_max_out = std::slice::from_raw_parts_mut(is_max_ptr, len);
            let last_min_out = std::slice::from_raw_parts_mut(last_min_ptr, len);
            let last_max_out = std::slice::from_raw_parts_mut(last_max_ptr, len);

            is_min_out.copy_from_slice(&temp_is_min);
            is_max_out.copy_from_slice(&temp_is_max);
            last_min_out.copy_from_slice(&temp_last_min);
            last_max_out.copy_from_slice(&temp_last_max);
        } else {
            // Direct output
            let is_min_out = std::slice::from_raw_parts_mut(is_min_ptr, len);
            let is_max_out = std::slice::from_raw_parts_mut(is_max_ptr, len);
            let last_min_out = std::slice::from_raw_parts_mut(last_min_ptr, len);
            let last_max_out = std::slice::from_raw_parts_mut(last_max_ptr, len);

            minmax_into_slice(
                is_min_out,
                is_max_out,
                last_min_out,
                last_max_out,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MinmaxBatchConfig {
    pub order_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MinmaxBatchJsOutput {
    pub values: Vec<f64>, // concatenated by series, then by combo
    pub combos: Vec<MinmaxParams>,
    pub rows: usize, // 4 * combos.len()
    pub cols: usize, // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = minmax_batch)]
pub fn minmax_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: MinmaxBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = MinmaxBatchRange {
        order: cfg.order_range,
    };
    let out = minmax_batch_with_kernel(high, low, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let rows = out.rows; // combos
    let cols = out.cols;

    let mut values = Vec::with_capacity(4 * rows * cols);
    // series-major layout: is_min, is_max, last_min, last_max blocks
    for series in 0..4 {
        for r in 0..rows {
            let (src, start) = match series {
                0 => (&out.is_min, r * cols),
                1 => (&out.is_max, r * cols),
                2 => (&out.last_min, r * cols),
                _ => (&out.last_max, r * cols),
            };
            values.extend_from_slice(&src[start..start + cols]);
        }
    }

    let js_out = MinmaxBatchJsOutput {
        values,
        combos: out.combos,
        rows: 4 * rows,
        cols,
    };
    serde_wasm_bindgen::to_value(&js_out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    is_min_ptr: *mut f64,
    is_max_ptr: *mut f64,
    last_min_ptr: *mut f64,
    last_max_ptr: *mut f64,
    len: usize,
    order_start: usize,
    order_end: usize,
    order_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || is_min_ptr.is_null()
        || is_max_ptr.is_null()
        || last_min_ptr.is_null()
        || last_max_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to minmax_batch_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = MinmaxBatchRange {
            order: (order_start, order_end, order_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let is_min_out = std::slice::from_raw_parts_mut(is_min_ptr, rows * cols);
        let is_max_out = std::slice::from_raw_parts_mut(is_max_ptr, rows * cols);
        let last_min_out = std::slice::from_raw_parts_mut(last_min_ptr, rows * cols);
        let last_max_out = std::slice::from_raw_parts_mut(last_max_ptr, rows * cols);

        minmax_batch_inner_into(
            high,
            low,
            &sweep,
            Kernel::Auto,
            false,
            is_min_out,
            is_max_out,
            last_min_out,
            last_max_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

// --- TESTS ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_minmax_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MinmaxParams { order: None };
        let input = MinmaxInput::from_candles(&candles, "high", "low", params);
        let output = minmax_with_kernel(&input, kernel)?;
        assert_eq!(output.is_min.len(), candles.close.len());
        Ok(())
    }

    fn check_minmax_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MinmaxParams { order: Some(3) };
        let input = MinmaxInput::from_candles(&candles, "high", "low", params);
        let output = minmax_with_kernel(&input, kernel)?;
        assert_eq!(output.is_min.len(), candles.close.len());
        let count = output.is_min.len();
        assert!(count >= 5, "Not enough data to check last 5");
        let start_index = count - 5;
        for &val in &output.is_min[start_index..] {
            assert!(val.is_nan());
        }
        for &val in &output.is_max[start_index..] {
            assert!(val.is_nan());
        }
        let expected_last_five_min = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
        let last_min_slice = &output.last_min[start_index..];
        for (i, &val) in last_min_slice.iter().enumerate() {
            let expected_val = expected_last_five_min[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "Minmax last_min mismatch at idx {}: {} vs {}",
                i,
                val,
                expected_val
            );
        }
        let expected_last_five_max = [60102.0, 60102.0, 60102.0, 60102.0, 60102.0];
        let last_max_slice = &output.last_max[start_index..];
        for (i, &val) in last_max_slice.iter().enumerate() {
            let expected_val = expected_last_five_max[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "Minmax last_max mismatch at idx {}: {} vs {}",
                i,
                val,
                expected_val
            );
        }
        Ok(())
    }

    fn check_minmax_zero_order(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [1.0, 2.0, 3.0];
        let params = MinmaxParams { order: Some(0) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with zero order",
            test_name
        );
        Ok(())
    }

    fn check_minmax_order_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [1.0, 2.0, 3.0];
        let params = MinmaxParams { order: Some(10) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with order > length",
            test_name
        );
        Ok(())
    }

    fn check_minmax_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = MinmaxParams { order: Some(1) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with all NaN data",
            test_name
        );
        Ok(())
    }

    fn check_minmax_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 10.0];
        let low = [f64::NAN, 5.0];
        let params = MinmaxParams { order: Some(3) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with not enough valid data",
            test_name
        );
        Ok(())
    }

    fn check_minmax_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [50.0, 55.0, 60.0, 55.0, 50.0, 45.0, 50.0, 55.0];
        let low = [40.0, 38.0, 35.0, 38.0, 40.0, 42.0, 41.0, 39.0];
        let params = MinmaxParams { order: Some(2) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let output = minmax_with_kernel(&input, kernel)?;
        assert_eq!(output.is_min.len(), 8);
        assert_eq!(output.is_max.len(), 8);
        assert_eq!(output.last_min.len(), 8);
        assert_eq!(output.last_max.len(), 8);
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_minmax_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            MinmaxParams::default(),           // order: 3
            MinmaxParams { order: Some(1) },   // minimum viable
            MinmaxParams { order: Some(2) },   // small
            MinmaxParams { order: Some(5) },   // small-medium
            MinmaxParams { order: Some(10) },  // medium
            MinmaxParams { order: Some(20) },  // medium-large
            MinmaxParams { order: Some(50) },  // large
            MinmaxParams { order: Some(100) }, // very large
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MinmaxInput::from_candles(&candles, "high", "low", params.clone());
            let output = minmax_with_kernel(&input, kernel)?;

            // Check all four output arrays
            let arrays = [
                (&output.is_min, "is_min"),
                (&output.is_max, "is_max"),
                (&output.last_min, "last_min"),
                (&output.last_max, "last_max"),
            ];

            for (array, array_name) in arrays.iter() {
                for (i, &val) in array.iter().enumerate() {
                    if val.is_nan() {
                        continue; // NaN values are expected during warmup
                    }

                    let bits = val.to_bits();

                    // Check all three poison patterns
                    if bits == 0x11111111_11111111 {
                        panic!(
							"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
							 in {} with params: order={} (param set {})",
							test_name, val, bits, i, array_name, 
							params.order.unwrap_or(3), param_idx
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
							 in {} with params: order={} (param set {})",
							test_name, val, bits, i, array_name,
							params.order.unwrap_or(3), param_idx
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
							 in {} with params: order={} (param set {})",
							test_name, val, bits, i, array_name,
							params.order.unwrap_or(3), param_idx
						);
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_minmax_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_minmax_tests {
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

    generate_all_minmax_tests!(
        check_minmax_partial_params,
        check_minmax_accuracy,
        check_minmax_zero_order,
        check_minmax_order_exceeds_length,
        check_minmax_nan_handling,
        check_minmax_very_small_dataset,
        check_minmax_basic_slices,
        check_minmax_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_minmax_tests!(check_minmax_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MinmaxBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = MinmaxParams::default();
        let row = output.is_min_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            (2, 10, 2),    // Small periods with step
            (5, 25, 5),    // Medium periods with step
            (30, 60, 15),  // Large periods with step
            (2, 5, 1),     // Dense small range
            (1, 1, 0),     // Single value (step 0)
            (10, 50, 10),  // Wide range
            (100, 100, 0), // Single large value
        ];

        for (cfg_idx, &(order_start, order_end, order_step)) in test_configs.iter().enumerate() {
            let output = MinmaxBatchBuilder::new()
                .kernel(kernel)
                .order_range(order_start, order_end, order_step)
                .apply_candles(&c)?;

            // Check all four output arrays
            let arrays = [
                (&output.is_min, "is_min"),
                (&output.is_max, "is_max"),
                (&output.last_min, "last_min"),
                (&output.last_max, "last_max"),
            ];

            for (array, array_name) in arrays.iter() {
                for (idx, &val) in array.iter().enumerate() {
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
							 at row {} col {} (flat index {}) in {} with params: order={}",
							test, cfg_idx, val, bits, row, col, idx, array_name,
							combo.order.unwrap_or(3)
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} with params: order={}",
							test, cfg_idx, val, bits, row, col, idx, array_name,
							combo.order.unwrap_or(3)
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} with params: order={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            array_name,
                            combo.order.unwrap_or(3)
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_minmax_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic high/low price data with various orders
        let strat = (1usize..=50).prop_flat_map(|order| {
            (
                // Generate data length from order to 400
                (order..400).prop_flat_map(move |len| {
                    // Generate pairs of (low, spread) to ensure high >= low
                    prop::collection::vec(
                        (0.1f64..1000.0f64, 0.0f64..=0.2)
                            .prop_filter("finite", |(x, _)| x.is_finite()),
                        len,
                    )
                    .prop_map(move |pairs| {
                        let mut low = Vec::with_capacity(len);
                        let mut high = Vec::with_capacity(len);

                        for (l, spread) in pairs {
                            low.push(l);
                            high.push(l * (1.0 + spread)); // Ensure high[i] >= low[i]
                        }

                        (high, low)
                    })
                }),
                Just(order),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |((high, low), order)| {
                let params = MinmaxParams { order: Some(order) };
                let input = MinmaxInput::from_slices(&high, &low, params);

                // Get outputs from different kernels
                let output = minmax_with_kernel(&input, kernel)?;
                let ref_output = minmax_with_kernel(&input, Kernel::Scalar)?;

                // Property 1: Output length matches input
                prop_assert_eq!(output.is_min.len(), high.len());
                prop_assert_eq!(output.is_max.len(), high.len());
                prop_assert_eq!(output.last_min.len(), high.len());
                prop_assert_eq!(output.last_max.len(), high.len());

                // Property 2: Warmup period handling
                // First 'order' values should be NaN for is_min/is_max
                for i in 0..order.min(high.len()) {
                    prop_assert!(
                        output.is_min[i].is_nan(),
                        "is_min[{}] should be NaN during warmup",
                        i
                    );
                    prop_assert!(
                        output.is_max[i].is_nan(),
                        "is_max[{}] should be NaN during warmup",
                        i
                    );
                }

                // Property 3: Local extrema validity
                // When a minimum is detected, it should be a valid local minimum
                for i in order..high.len().saturating_sub(order) {
                    if !output.is_min[i].is_nan() {
                        // This is a detected minimum - verify it's actually a local min
                        prop_assert_eq!(
                            output.is_min[i],
                            low[i],
                            "is_min[{}] should equal low[{}]",
                            i,
                            i
                        );

                        // Check it's less than or equal to all neighbors (implementation uses >=)
                        for o in 1..=order {
                            if i >= o && i + o < low.len() {
                                prop_assert!(
                                    low[i] <= low[i - o] && low[i] <= low[i + o],
                                    "Detected min at {} not <= neighbors at {} and {}",
                                    i,
                                    i - o,
                                    i + o
                                );
                            }
                        }
                    }

                    if !output.is_max[i].is_nan() {
                        // This is a detected maximum - verify it's actually a local max
                        prop_assert_eq!(
                            output.is_max[i],
                            high[i],
                            "is_max[{}] should equal high[{}]",
                            i,
                            i
                        );

                        // Check it's greater than or equal to all neighbors (implementation uses <=)
                        for o in 1..=order {
                            if i >= o && i + o < high.len() {
                                prop_assert!(
                                    high[i] >= high[i - o] && high[i] >= high[i + o],
                                    "Detected max at {} not >= neighbors at {} and {}",
                                    i,
                                    i - o,
                                    i + o
                                );
                            }
                        }
                    }
                }

                // Property 4: Forward-filling behavior
                // last_min and last_max should forward-fill from is_min/is_max
                let first_valid_idx = high
                    .iter()
                    .zip(low.iter())
                    .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
                    .unwrap_or(0);

                for i in first_valid_idx..high.len() {
                    // Check forward-filling logic
                    if i > first_valid_idx {
                        // If no new extrema detected, should maintain previous value
                        if output.is_min[i].is_nan() && !output.last_min[i - 1].is_nan() {
                            prop_assert_eq!(
                                output.last_min[i],
                                output.last_min[i - 1],
                                "last_min[{}] should equal last_min[{}]",
                                i,
                                i - 1
                            );
                        }
                        if output.is_max[i].is_nan() && !output.last_max[i - 1].is_nan() {
                            prop_assert_eq!(
                                output.last_max[i],
                                output.last_max[i - 1],
                                "last_max[{}] should equal last_max[{}]",
                                i,
                                i - 1
                            );
                        }

                        // If new extrema detected, should update to new value
                        if !output.is_min[i].is_nan() {
                            prop_assert_eq!(
                                output.last_min[i],
                                output.is_min[i],
                                "last_min[{}] should update to new minimum",
                                i
                            );
                        }
                        if !output.is_max[i].is_nan() {
                            prop_assert_eq!(
                                output.last_max[i],
                                output.is_max[i],
                                "last_max[{}] should update to new maximum",
                                i
                            );
                        }
                    }
                }

                // Property 5: Kernel consistency
                // Different kernels should produce identical results within ULP tolerance
                for i in 0..high.len() {
                    // Check is_min consistency
                    if output.is_min[i].is_finite() && ref_output.is_min[i].is_finite() {
                        let ulp_diff = output.is_min[i]
                            .to_bits()
                            .abs_diff(ref_output.is_min[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "is_min[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.is_min[i],
                            ref_output.is_min[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.is_min[i].to_bits(),
                            ref_output.is_min[i].to_bits(),
                            "is_min[{}] NaN mismatch",
                            i
                        );
                    }

                    // Check is_max consistency
                    if output.is_max[i].is_finite() && ref_output.is_max[i].is_finite() {
                        let ulp_diff = output.is_max[i]
                            .to_bits()
                            .abs_diff(ref_output.is_max[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "is_max[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.is_max[i],
                            ref_output.is_max[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.is_max[i].to_bits(),
                            ref_output.is_max[i].to_bits(),
                            "is_max[{}] NaN mismatch",
                            i
                        );
                    }

                    // Check last_min consistency
                    if output.last_min[i].is_finite() && ref_output.last_min[i].is_finite() {
                        let ulp_diff = output.last_min[i]
                            .to_bits()
                            .abs_diff(ref_output.last_min[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "last_min[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.last_min[i],
                            ref_output.last_min[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.last_min[i].to_bits(),
                            ref_output.last_min[i].to_bits(),
                            "last_min[{}] NaN mismatch",
                            i
                        );
                    }

                    // Check last_max consistency
                    if output.last_max[i].is_finite() && ref_output.last_max[i].is_finite() {
                        let ulp_diff = output.last_max[i]
                            .to_bits()
                            .abs_diff(ref_output.last_max[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "last_max[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.last_max[i],
                            ref_output.last_max[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.last_max[i].to_bits(),
                            ref_output.last_max[i].to_bits(),
                            "last_max[{}] NaN mismatch",
                            i
                        );
                    }
                }

                // Property 6: Boundary values
                // All detected extrema should be within data range
                let min_low = low.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_high = high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                for i in 0..high.len() {
                    if !output.is_min[i].is_nan() {
                        prop_assert!(
                            output.is_min[i] >= min_low && output.is_min[i] <= max_high,
                            "is_min[{}]={} outside data range [{}, {}]",
                            i,
                            output.is_min[i],
                            min_low,
                            max_high
                        );
                    }
                    if !output.is_max[i].is_nan() {
                        prop_assert!(
                            output.is_max[i] >= min_low && output.is_max[i] <= max_high,
                            "is_max[{}]={} outside data range [{}, {}]",
                            i,
                            output.is_max[i],
                            min_low,
                            max_high
                        );
                    }
                    if !output.last_min[i].is_nan() {
                        prop_assert!(
                            output.last_min[i] >= min_low && output.last_min[i] <= max_high,
                            "last_min[{}]={} outside data range [{}, {}]",
                            i,
                            output.last_min[i],
                            min_low,
                            max_high
                        );
                    }
                    if !output.last_max[i].is_nan() {
                        prop_assert!(
                            output.last_max[i] >= min_low && output.last_max[i] <= max_high,
                            "last_max[{}]={} outside data range [{}, {}]",
                            i,
                            output.last_max[i],
                            min_low,
                            max_high
                        );
                    }
                }

                // Property 7: Order = 1 special case
                // With order=1, extrema detection looks at immediate neighbors only
                if order == 1 && high.len() >= 3 {
                    for i in 1..high.len() - 1 {
                        // Check if this is a valid local minimum (strict inequality)
                        if low[i] < low[i - 1] && low[i] < low[i + 1] {
                            prop_assert!(
                                !output.is_min[i].is_nan(),
                                "Expected minimum at {} not detected",
                                i
                            );
                        }
                        // Check if this is a valid local maximum (strict inequality)
                        if high[i] > high[i - 1] && high[i] > high[i + 1] {
                            prop_assert!(
                                !output.is_max[i].is_nan(),
                                "Expected maximum at {} not detected",
                                i
                            );
                        }
                    }
                }

                // Property 8: High >= Low constraint
                // Verify our generated data maintains the constraint
                for i in 0..high.len() {
                    prop_assert!(
                        high[i] >= low[i],
                        "Invalid data: high[{}]={} < low[{}]={}",
                        i,
                        high[i],
                        i,
                        low[i]
                    );
                }

                Ok(())
            })
            .unwrap();

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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
