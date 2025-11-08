//! # Midprice
//!
//! The midpoint price over a specified period, calculated as `(highest high + lowest low) / 2`.
//! Useful for identifying average price levels in a range and potential support/resistance.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 14.
//!
//! ## Returns
//! - **`Ok(MidpriceOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the window is filled.
//! - **`Err(MidpriceError)`** on failure
//!
//! ## Developer Notes
//! - **AVX2**: Stub implementation - calls scalar function
//! - **AVX512**: Multiple stub functions (midprice_avx512_short, midprice_avx512_long) - all call scalar
//! - **Streaming**: O(1) amortized per update via two monotonic deques (max of highs, min of lows);
//!   warmup matches batch semantics (first_valid_idx + period - 1). Accuracy unchanged.
//! - **Memory**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// --- Input Data Abstractions ---
#[derive(Debug, Clone)]
pub enum MidpriceData<'a> {
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
pub struct MidpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MidpriceParams {
    pub period: Option<usize>,
}

impl Default for MidpriceParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct MidpriceInput<'a> {
    pub data: MidpriceData<'a>,
    pub params: MidpriceParams,
}

impl<'a> MidpriceInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        params: MidpriceParams,
    ) -> Self {
        Self {
            data: MidpriceData::Candles {
                candles,
                high_src,
                low_src,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MidpriceParams) -> Self {
        Self {
            data: MidpriceData::Slices { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MidpriceData::Candles {
                candles,
                high_src: "high",
                low_src: "low",
            },
            params: MidpriceParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// --- Builder/Stream/Batch Structs for Parity ---
#[derive(Copy, Clone, Debug)]
pub struct MidpriceBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MidpriceBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MidpriceBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MidpriceOutput, MidpriceError> {
        let p = MidpriceParams {
            period: self.period,
        };
        let i = MidpriceInput::from_candles(c, "high", "low", p);
        midprice_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MidpriceOutput, MidpriceError> {
        let p = MidpriceParams {
            period: self.period,
        };
        let i = MidpriceInput::from_slices(high, low, p);
        midprice_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MidpriceStream, MidpriceError> {
        let p = MidpriceParams {
            period: self.period,
        };
        MidpriceStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MidpriceError {
    #[error("midprice: Empty data provided.")]
    EmptyData,
    #[error("midprice: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("midprice: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("midprice: All values are NaN.")]
    AllValuesNaN,
    #[error("midprice: Mismatched data length: high_len = {high_len}, low_len = {low_len}")]
    MismatchedDataLength { high_len: usize, low_len: usize },
    #[error("midprice: Invalid output length: expected = {expected}, actual = {actual}")]
    InvalidLength { expected: usize, actual: usize },
}

// --- Kernel/Dispatch API ---
#[inline]
pub fn midprice(input: &MidpriceInput) -> Result<MidpriceOutput, MidpriceError> {
    midprice_with_kernel(input, Kernel::Auto)
}

pub fn midprice_with_kernel(
    input: &MidpriceInput,
    kernel: Kernel,
) -> Result<MidpriceOutput, MidpriceError> {
    let (high, low) = match &input.data {
        MidpriceData::Candles {
            candles,
            high_src,
            low_src,
        } => {
            let h = source_type(candles, high_src);
            let l = source_type(candles, low_src);
            (h, l)
        }
        MidpriceData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MidpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MidpriceError::MismatchedDataLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(MidpriceError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }
    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MidpriceError::AllValuesNaN),
    };
    if (high.len() - first_valid_idx) < period {
        return Err(MidpriceError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }
    let mut out = alloc_with_nan_prefix(high.len(), first_valid_idx + period - 1);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                midprice_scalar(high, low, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                midprice_avx2(high, low, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                midprice_avx512(high, low, period, first_valid_idx, &mut out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                midprice_scalar(high, low, period, first_valid_idx, &mut out)
            }
            Kernel::Auto => midprice_scalar(high, low, period, first_valid_idx, &mut out), // Shouldn't happen but handle it
        }
    }
    Ok(MidpriceOutput { values: out })
}

/// Write midprice values into the provided buffer without allocating.
///
/// - Preserves NaN warmups identically to the Vec-returning API.
/// - The output slice length must match the input length.
#[cfg(not(feature = "wasm"))]
pub fn midprice_into(input: &MidpriceInput, out: &mut [f64]) -> Result<(), MidpriceError> {
    let (high, low) = match &input.data {
        MidpriceData::Candles {
            candles,
            high_src,
            low_src,
        } => {
            let h = source_type(candles, high_src);
            let l = source_type(candles, low_src);
            (h, l)
        }
        MidpriceData::Slices { high, low } => (*high, *low),
    };

    if out.len() != high.len() || high.len() != low.len() {
        return Err(MidpriceError::InvalidLength {
            expected: high.len(),
            actual: out.len(),
        });
    }

    // Reuse the module's into-slice implementation with Kernel::Auto
    midprice_into_slice(out, high, low, &input.params, Kernel::Auto)
}

#[inline]
pub fn midprice_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    for i in (first_valid_idx + period - 1)..high.len() {
        let window_start = i + 1 - period;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        for j in window_start..=i {
            if high[j] > highest {
                highest = high[j];
            }
            if low[j] < lowest {
                lowest = low[j];
            }
        }
        out[i] = (highest + lowest) / 2.0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midprice_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    midprice_scalar(high, low, period, first_valid_idx, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midprice_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { midprice_avx512_short(high, low, period, first_valid_idx, out) }
    } else {
        unsafe { midprice_avx512_long(high, low, period, first_valid_idx, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn midprice_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    midprice_scalar(high, low, period, first_valid_idx, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn midprice_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    midprice_scalar(high, low, period, first_valid_idx, out);
}

// --- Streaming Struct ---
// Decision: Switch streaming to O(1) amortized updates using two
// monotonic deques; warmup matches batch (first_valid_idx + period - 1).
#[derive(Debug, Clone)]
pub struct MidpriceStream {
    period: usize,
    // Have we observed the first fully valid (high, low)?
    started: bool,
    // Number of valid bars seen since `started` (also the next index to use).
    seen: usize,
    // Monotonic deques; store (index_since_start, value)
    dq_high: std::collections::VecDeque<(usize, f64)>, // max queue (descending values)
    dq_low: std::collections::VecDeque<(usize, f64)>,  // min queue (ascending values)
}

impl MidpriceStream {
    pub fn try_new(params: MidpriceParams) -> Result<Self, MidpriceError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(MidpriceError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let cap = period + 1;
        Ok(Self {
            period,
            started: false,
            seen: 0,
            dq_high: std::collections::VecDeque::with_capacity(cap),
            dq_low: std::collections::VecDeque::with_capacity(cap),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        // Delay start until both high & low are finite, mirroring batch warmup
        if !self.started {
            if !(high.is_finite() && low.is_finite()) {
                return None;
            }
            self.started = true;
            self.seen = 0; // first valid sample uses index 0
        }

        let i = self.seen;

        // Push into max-deque for highs (keep descending values)
        if high.is_finite() {
            while let Some(&(_, v)) = self.dq_high.back() {
                if v <= high {
                    self.dq_high.pop_back();
                } else {
                    break;
                }
            }
            self.dq_high.push_back((i, high));
        }

        // Push into min-deque for lows (keep ascending values)
        if low.is_finite() {
            while let Some(&(_, v)) = self.dq_low.back() {
                if v >= low {
                    self.dq_low.pop_back();
                } else {
                    break;
                }
            }
            self.dq_low.push_back((i, low));
        }

        // Evict items that fell out of the window [i - period + 1, i]
        let start = i.saturating_add(1).saturating_sub(self.period);
        while let Some(&(idx, _)) = self.dq_high.front() {
            if idx < start {
                self.dq_high.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(idx, _)) = self.dq_low.front() {
            if idx < start {
                self.dq_low.pop_front();
            } else {
                break;
            }
        }

        // Advance stream index
        self.seen = i + 1;

        // Not enough bars since start → still warming
        if self.seen < self.period {
            return None;
        }

        Some(self.calc())
    }

    #[inline(always)]
    fn calc(&self) -> f64 {
        let max_h = self
            .dq_high
            .front()
            .map(|&(_, v)| v)
            .unwrap_or(f64::NEG_INFINITY);
        let min_l = self
            .dq_low
            .front()
            .map(|&(_, v)| v)
            .unwrap_or(f64::INFINITY);
        (max_h + min_l) / 2.0
    }
}

// --- Batch/Range/Builder ---

#[derive(Clone, Debug)]
pub struct MidpriceBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MidpriceBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MidpriceBatchBuilder {
    range: MidpriceBatchRange,
    kernel: Kernel,
}

impl MidpriceBatchBuilder {
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<MidpriceBatchOutput, MidpriceError> {
        midprice_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        high_src: &str,
        low_src: &str,
    ) -> Result<MidpriceBatchOutput, MidpriceError> {
        let high = source_type(c, high_src);
        let low = source_type(c, low_src);
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MidpriceBatchOutput, MidpriceError> {
        MidpriceBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "high", "low")
    }
}

pub fn midprice_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &MidpriceBatchRange,
    k: Kernel,
) -> Result<MidpriceBatchOutput, MidpriceError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MidpriceError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Fallback to scalar
    };
    midprice_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MidpriceBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MidpriceParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MidpriceBatchOutput {
    pub fn row_for_params(&self, p: &MidpriceParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &MidpriceParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MidpriceBatchRange) -> Vec<MidpriceParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(MidpriceParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn midprice_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MidpriceBatchRange,
    kern: Kernel,
) -> Result<MidpriceBatchOutput, MidpriceError> {
    midprice_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn midprice_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MidpriceBatchRange,
    kern: Kernel,
) -> Result<MidpriceBatchOutput, MidpriceError> {
    midprice_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn midprice_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &MidpriceBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MidpriceParams>, MidpriceError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MidpriceError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    if high.is_empty() || low.is_empty() {
        return Err(MidpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MidpriceError::EmptyData);
    }
    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MidpriceError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(MidpriceError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();

    // Note: output is already initialized with proper NaN prefixes by the caller

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                midprice_row_scalar(high, low, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                midprice_row_avx2(high, low, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                midprice_row_avx512(high, low, first, period, out_row)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                midprice_row_scalar(high, low, first, period, out_row)
            }
            Kernel::Auto => midprice_row_scalar(high, low, first, period, out_row), // Shouldn't happen but handle it
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

    Ok(combos)
}

#[inline(always)]
fn midprice_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &MidpriceBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MidpriceBatchOutput, MidpriceError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MidpriceError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    if high.is_empty() || low.is_empty() {
        return Err(MidpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MidpriceError::MismatchedDataLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MidpriceError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(MidpriceError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();

    // 1) allocate uninit
    let mut buf_mu = make_uninit_matrix(rows, cols);
    // 2) set only warm prefixes to NaN
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // 3) compute directly into the same allocation
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let combos = midprice_batch_inner_into(high, low, sweep, kern, parallel, out)?;

    // 4) return the very same buffer as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(MidpriceBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn midprice_row_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    for i in (first + period - 1)..high.len() {
        let window_start = i + 1 - period;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        for j in window_start..=i {
            if high[j] > highest {
                highest = high[j];
            }
            if low[j] < lowest {
                lowest = low[j];
            }
        }
        out[i] = (highest + lowest) / 2.0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn midprice_row_avx2(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midprice_row_scalar(high, low, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn midprice_row_avx512(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        midprice_row_avx512_short(high, low, first, period, out);
    } else {
        midprice_row_avx512_long(high, low, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn midprice_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midprice_row_scalar(high, low, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn midprice_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midprice_row_scalar(high, low, first, period, out);
}

#[cfg(feature = "python")]
#[pyfunction(name = "midprice")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn midprice_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MidpriceParams {
        period: Some(period),
    };
    let input = MidpriceInput::from_slices(high_slice, low_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| midprice_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "midprice_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn midprice_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = MidpriceBatchRange {
        period: period_range,
    };

    let cols = high_slice.len();
    if cols == 0 {
        return Err(PyValueError::new_err("midprice: empty data"));
    }
    if cols != low_slice.len() {
        return Err(PyValueError::new_err(format!(
            "midprice: length mismatch: high={}, low={}",
            cols,
            low_slice.len()
        )));
    }
    let first = (0..cols)
        .find(|&i| !high_slice[i].is_nan() && !low_slice[i].is_nan())
        .ok_or_else(|| PyValueError::new_err("midprice: All values are NaN"))?;

    let combos = expand_grid(&sweep);
    let rows = combos.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // warm NaN prefixes in-place (no extra buffer)
    let warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.period.unwrap() - 1)
        .collect();
    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(
            slice_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            rows * cols,
        )
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    // compute
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
                _ => Kernel::Scalar, // Fallback to scalar for any other kernel type
            };
            midprice_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
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
#[pyclass(name = "MidpriceStream")]
pub struct MidpriceStreamPy {
    stream: MidpriceStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MidpriceStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = MidpriceParams {
            period: Some(period),
        };
        let stream =
            MidpriceStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MidpriceStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

// --- WASM Bindings ---

/// Write midprice values directly to output slice - no allocations
pub fn midprice_into_slice(
    dst: &mut [f64],
    high: &[f64],
    low: &[f64],
    params: &MidpriceParams,
    kern: Kernel,
) -> Result<(), MidpriceError> {
    // Validate inputs
    if high.is_empty() || low.is_empty() {
        return Err(MidpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MidpriceError::MismatchedDataLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    if dst.len() != high.len() {
        return Err(MidpriceError::InvalidLength {
            expected: high.len(),
            actual: dst.len(),
        });
    }

    let period = params.period.unwrap_or(14);
    if period == 0 || period > high.len() {
        return Err(MidpriceError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    // Find first valid index
    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MidpriceError::AllValuesNaN),
    };

    if high.len() - first_valid_idx < period {
        return Err(MidpriceError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    // Select kernel
    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Initialize output with NaN up to warmup period
    let warmup_end = first_valid_idx + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    // Compute midprice directly into output
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            midprice_scalar(high, low, period, first_valid_idx, dst)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => midprice_avx2(high, low, period, first_valid_idx, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            midprice_avx512(high, low, period, first_valid_idx, dst)
        }
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            midprice_scalar(high, low, period, first_valid_idx, dst)
        }
        Kernel::Auto => midprice_scalar(high, low, period, first_valid_idx, dst), // Shouldn't happen but handle it
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midprice_js(high: &[f64], low: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = MidpriceParams {
        period: Some(period),
    };

    let mut output = vec![0.0; high.len()];

    midprice_into_slice(&mut output, high, low, &params, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midprice_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midprice_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midprice_into(
    in_high_ptr: *const f64,
    in_low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_high_ptr.is_null() || in_low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(in_high_ptr, len);
        let low = std::slice::from_raw_parts(in_low_ptr, len);
        let params = MidpriceParams {
            period: Some(period),
        };

        // Check for aliasing - if either input pointer equals output pointer
        if in_high_ptr == out_ptr || in_low_ptr == out_ptr {
            // Use temporary buffer
            let mut temp = vec![0.0; len];
            midprice_into_slice(&mut temp, high, low, &params, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            midprice_into_slice(out, high, low, &params, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MidpriceBatchConfig {
    pub period_range: (usize, usize, usize), // (start, end, step)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MidpriceBatchJsOutput {
    pub values: Vec<f64>,
    pub periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = midprice_batch)]
pub fn midprice_batch_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MidpriceBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let range = MidpriceBatchRange {
        period: config.period_range,
    };

    let output = midprice_batch_inner(high, low, &range, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert output to flat array with period values
    let mut periods = Vec::new();
    let (start, end, step) = config.period_range;
    if step == 0 || start == end {
        periods.push(start);
    } else {
        let mut current = start;
        while current <= end {
            periods.push(current);
            current += step;
        }
    }

    let js_output = MidpriceBatchJsOutput {
        values: output.values,
        periods,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midprice_batch_into(
    in_high_ptr: *const f64,
    in_low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    unsafe {
        let high = std::slice::from_raw_parts(in_high_ptr, len);
        let low = std::slice::from_raw_parts(in_low_ptr, len);

        let range = MidpriceBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&range);
        let rows = combos.len();
        let total = rows * len;

        let first = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .ok_or_else(|| JsValue::from_str("All values are NaN"))?;
        let warm: Vec<usize> = combos
            .iter()
            .map(|p| first + p.period.unwrap() - 1)
            .collect();

        if in_high_ptr == out_ptr || in_low_ptr == out_ptr {
            let mut temp: Vec<f64> = Vec::with_capacity(total);
            temp.set_len(total);
            let mu = std::slice::from_raw_parts_mut(
                temp.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                total,
            );
            init_matrix_prefixes(mu, len, &warm);
            midprice_batch_inner_into(high, low, &range, Kernel::Auto, false, &mut temp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, total).copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, total);
            let mu = std::slice::from_raw_parts_mut(
                out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                total,
            );
            init_matrix_prefixes(mu, len, &warm);
            midprice_batch_inner_into(high, low, &range, Kernel::Auto, false, out)
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

    fn check_midprice_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = MidpriceParams { period: None };
        let input = MidpriceInput::with_default_candles(&candles);
        let output = midprice_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_midprice_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = MidpriceInput::with_default_candles(&candles);
        let result = midprice_with_kernel(&input, kernel)?;
        let expected_last_five = [59583.0, 59583.0, 59583.0, 59486.0, 58989.0];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MIDPRICE {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_midprice_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MidpriceInput::with_default_candles(&candles);
        let output = midprice_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_midprice_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let highs = [10.0, 14.0, 12.0];
        let lows = [5.0, 6.0, 7.0];
        let params = MidpriceParams { period: Some(0) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let res = midprice_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MIDPRICE should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_midprice_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let highs = [10.0, 14.0, 12.0];
        let lows = [5.0, 6.0, 7.0];
        let params = MidpriceParams { period: Some(10) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let res = midprice_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MIDPRICE should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_midprice_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let highs = [42.0];
        let lows = [36.0];
        let params = MidpriceParams { period: Some(14) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let res = midprice_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MIDPRICE should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_midprice_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = MidpriceParams { period: Some(10) };
        let input = MidpriceInput::with_default_candles(&candles);
        let first_result = midprice_with_kernel(&input, kernel)?;

        let second_input = MidpriceInput::from_slices(
            &first_result.values,
            &first_result.values,
            MidpriceParams { period: Some(10) },
        );
        let second_result = midprice_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_midprice_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MidpriceInput::with_default_candles(&candles);
        let res = midprice_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_midprice_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = MidpriceInput::with_default_candles(&candles);
        let batch_output = midprice_with_kernel(&input, kernel)?.values;

        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let mut stream = MidpriceStream::try_new(MidpriceParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(high.len());
        for (&h, &l) in high.iter().zip(low.iter()) {
            match stream.update(h, l) {
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
                "[{}] MIDPRICE streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_midprice_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let highs = [f64::NAN, f64::NAN, f64::NAN];
        let lows = [f64::NAN, f64::NAN, f64::NAN];
        let params = MidpriceParams { period: Some(2) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let result = midprice_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for all NaN values",
            test_name
        );
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_midprice_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            MidpriceParams::default(),            // period: 14
            MidpriceParams { period: Some(2) },   // minimum viable
            MidpriceParams { period: Some(5) },   // small
            MidpriceParams { period: Some(7) },   // small
            MidpriceParams { period: Some(20) },  // medium
            MidpriceParams { period: Some(30) },  // medium-large
            MidpriceParams { period: Some(50) },  // large
            MidpriceParams { period: Some(100) }, // very large
            MidpriceParams { period: Some(200) }, // extreme
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MidpriceInput::from_candles(&candles, "high", "low", params.clone());
            let output = midprice_with_kernel(&input, kernel)?;

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
                        params.period.unwrap_or(14),
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
                        params.period.unwrap_or(14),
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
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_midprice_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(test)]
    fn check_midprice_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test strategy with realistic price data
        let strat = (2usize..=100)
            .prop_flat_map(|period| {
                (
                    // Generate base prices - include edge cases with very small and large values
                    prop::collection::vec(
                        prop::strategy::Union::new(vec![
                            (0.001f64..0.1f64).boxed(), // Very small prices
                            (10f64..10000f64).boxed(),  // Normal prices
                            (1e6f64..1e8f64).boxed(),   // Very large prices
                        ])
                        .prop_filter("finite", |x| x.is_finite()),
                        period..=500,
                    ),
                    Just(period),
                    // Spread factor - realistic market spreads
                    prop::strategy::Union::new(vec![
                        (0.0001f64..0.01f64).boxed(), // Tight spreads (0.01% - 1%)
                        (0.01f64..0.1f64).boxed(),    // Normal spreads (1% - 10%)
                        (0.1f64..0.3f64).boxed(),     // Wide spreads (10% - 30%)
                    ]),
                    // Scenario selector - expanded to 10 scenarios
                    0usize..=9,
                )
            })
            .prop_map(|(base_prices, period, spread_factor, scenario)| {
                let len = base_prices.len();
                let mut highs = Vec::with_capacity(len);
                let mut lows = Vec::with_capacity(len);

                match scenario {
                    0 => {
                        // Random realistic price data with variable spread
                        for &base in &base_prices {
                            let spread = base * spread_factor;
                            highs.push(base + spread / 2.0);
                            lows.push(base - spread / 2.0);
                        }
                    }
                    1 => {
                        // Constant prices
                        let price = base_prices[0];
                        let spread = price * spread_factor;
                        for _ in 0..len {
                            highs.push(price + spread / 2.0);
                            lows.push(price - spread / 2.0);
                        }
                    }
                    2 => {
                        // Monotonically increasing prices
                        let start_price = base_prices[0];
                        let increment = 10.0;
                        for i in 0..len {
                            let price = start_price + (i as f64) * increment;
                            let spread = price * spread_factor;
                            highs.push(price + spread / 2.0);
                            lows.push(price - spread / 2.0);
                        }
                    }
                    3 => {
                        // Monotonically decreasing prices
                        let start_price = base_prices[0];
                        let decrement = 10.0;
                        for i in 0..len {
                            let price = (start_price - (i as f64) * decrement).max(10.0);
                            let spread = price * spread_factor;
                            highs.push(price + spread / 2.0);
                            lows.push(price - spread / 2.0);
                        }
                    }
                    4 => {
                        // Period=1 special case (exact midpoint test)
                        for &base in &base_prices {
                            let spread = base * 0.01; // Small fixed spread
                            highs.push(base + spread);
                            lows.push(base);
                        }
                    }
                    5 => {
                        // Oscillating prices (sine wave)
                        let amplitude = 100.0;
                        let offset = base_prices[0];
                        for i in 0..len {
                            let phase = (i as f64) * 0.1;
                            let price = offset + amplitude * phase.sin();
                            let spread = price.abs() * spread_factor;
                            highs.push(price + spread / 2.0);
                            lows.push(price - spread / 2.0);
                        }
                    }
                    6 => {
                        // Step function (sudden jumps)
                        let step_size = 100.0;
                        let steps = 5;
                        for i in 0..len {
                            let step = (i * steps / len) as f64;
                            let price = base_prices[0] + step * step_size;
                            let spread = price * spread_factor;
                            highs.push(price + spread / 2.0);
                            lows.push(price - spread / 2.0);
                        }
                    }
                    7 => {
                        // Volatile market (use base prices with added volatility)
                        for (i, &base) in base_prices.iter().enumerate() {
                            // Add volatility based on position (creates pseudo-random walk)
                            let volatility = ((i as f64 * 0.1).sin() + 1.0) * 0.5; // 0 to 1
                            let price = base * (1.0 + spread_factor * volatility);
                            let spread = price * spread_factor;
                            highs.push(price + spread / 2.0);
                            lows.push(price - spread / 2.0);
                        }
                    }
                    8 => {
                        // Edge case: period equals data length (test full window)
                        // Generate prices that change to verify window includes all data
                        for i in 0..len {
                            let price = base_prices[0] + (i as f64) * 5.0;
                            let spread = price * spread_factor.min(0.1); // Cap spread for this test
                            highs.push(price + spread / 2.0);
                            lows.push(price - spread / 2.0);
                        }
                    }
                    _ => {
                        // Extreme monotonic test: prices that strictly increase then strictly decrease
                        // This tests the indicator's ability to track turning points
                        let mid_point = len / 2;
                        for i in 0..len {
                            let base = if i < mid_point {
                                // Strictly increasing in first half
                                base_prices[0] + (i as f64) * 20.0
                            } else {
                                // Strictly decreasing in second half
                                base_prices[0] + (mid_point as f64) * 20.0
                                    - ((i - mid_point) as f64) * 20.0
                            };
                            let spread = base * spread_factor.min(0.05); // Cap spread at 5% for this test
                            highs.push(base + spread / 2.0);
                            lows.push(base - spread / 2.0);
                        }
                    }
                }

                (highs, lows, period)
            });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(highs, lows, period)| {
				let params = MidpriceParams { period: Some(period) };
				let input = MidpriceInput::from_slices(&highs, &lows, params);

				// Get outputs from both test kernel and scalar reference
				let MidpriceOutput { values: out } = midprice_with_kernel(&input, kernel)?;
				let MidpriceOutput { values: ref_out } = midprice_with_kernel(&input, Kernel::Scalar)?;

				// Find first valid index
				let first_valid = (0..highs.len())
					.find(|&i| !highs[i].is_nan() && !lows[i].is_nan())
					.unwrap_or(0);

				// Property 1: Warmup period - first (first_valid + period - 1) values should be NaN
				let warmup_end = first_valid + period - 1;
				for i in 0..warmup_end.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i, out[i]
					);
				}

				// Properties for valid output values
				for i in warmup_end..out.len() {
					let y = out[i];
					let r = ref_out[i];
					let window_start = i + 1 - period;

					// Property 2: Output bounds - midprice should be between min(low) and max(high) in window
					let window_high = &highs[window_start..=i];
					let window_low = &lows[window_start..=i];
					let max_high = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
					let min_low = window_low.iter().cloned().fold(f64::INFINITY, f64::min);

					// Adaptive tolerance based on magnitude
					let magnitude = y.abs().max(1.0);
					let tolerance = (magnitude * f64::EPSILON * 10.0).max(1e-9);

					prop_assert!(
						y.is_nan() || (y >= min_low - tolerance && y <= max_high + tolerance),
						"Midprice {} at index {} outside bounds [{}, {}]",
						y, i, min_low, max_high
					);

					// Property 3: Kernel consistency
					if y.is_finite() && r.is_finite() {
						let ulp_diff = y.to_bits().abs_diff(r.to_bits());
						prop_assert!(
							(y - r).abs() <= tolerance || ulp_diff <= 8,
							"Kernel mismatch at index {}: {} vs {} (ULP={})",
							i, y, r, ulp_diff
						);
					} else {
						prop_assert_eq!(
							y.to_bits(), r.to_bits(),
							"NaN/finite mismatch at index {}: {} vs {}",
							i, y, r
						);
					}

					// Property 4: Period=1 special case
					if period == 1 {
						let expected = (highs[i] + lows[i]) / 2.0;
						prop_assert!(
							(y - expected).abs() <= tolerance,
							"Period=1 midprice {} != expected {} at index {}",
							y, expected, i
						);
					}

					// Property 5: Constant data
					if window_high.windows(2).all(|w| (w[0] - w[1]).abs() < tolerance) &&
					   window_low.windows(2).all(|w| (w[0] - w[1]).abs() < tolerance) {
						let expected = (window_high[0] + window_low[0]) / 2.0;
						prop_assert!(
							(y - expected).abs() <= tolerance,
							"Constant data midprice {} != expected {} at index {}",
							y, expected, i
						);
					}

					// Property 6: Mathematical correctness
					let actual_max_high = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
					let actual_min_low = window_low.iter().cloned().fold(f64::INFINITY, f64::min);
					let expected = (actual_max_high + actual_min_low) / 2.0;
					prop_assert!(
						(y - expected).abs() <= tolerance,
						"Midprice {} != expected {} at index {}",
						y, expected, i
					);

					// Property 7: Midprice always within actual high and low at current index
					// The midprice of the window should be between the lowest low and highest high
					// but also should never exceed the current bar's high or go below current bar's low
					// when period = 1
					if period == 1 {
						prop_assert!(
							y >= lows[i] - tolerance && y <= highs[i] + tolerance,
							"When period=1, midprice {} should be within current bar's range [{}, {}] at index {}",
							y, lows[i], highs[i], i
						);
					}

					// Property 8: Monotonicity check for strictly monotonic data
					// If both high and low arrays are strictly increasing/decreasing in the window,
					// the midprice should follow the trend
					if i > warmup_end && window_high.len() > 1 && window_low.len() > 1 {
						let high_increasing = window_high.windows(2).all(|w| w[1] >= w[0] - tolerance);
						let high_decreasing = window_high.windows(2).all(|w| w[1] <= w[0] + tolerance);
						let low_increasing = window_low.windows(2).all(|w| w[1] >= w[0] - tolerance);
						let low_decreasing = window_low.windows(2).all(|w| w[1] <= w[0] + tolerance);

						// If both arrays are monotonic in the same direction
						if high_increasing && low_increasing {
							// Previous midprice should be less than or equal to current
							if i > warmup_end {
								let prev_y = out[i - 1];
								if prev_y.is_finite() && y.is_finite() {
									// Allow for some tolerance due to the sliding window
									// The midprice might decrease slightly if a high value exits the window
									let allowed_decrease = max_high * 0.1; // Allow 10% decrease
									prop_assert!(
										y >= prev_y - allowed_decrease,
										"Midprice should generally increase with monotonic increasing data: {} < {} - {} at index {}",
										y, prev_y, allowed_decrease, i
									);
								}
							}
						}
					}
				}

				Ok(())
			})?;

        Ok(())
    }

    macro_rules! generate_all_midprice_tests {
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

    generate_all_midprice_tests!(
        check_midprice_partial_params,
        check_midprice_accuracy,
        check_midprice_default_candles,
        check_midprice_zero_period,
        check_midprice_period_exceeds_length,
        check_midprice_very_small_dataset,
        check_midprice_reinput,
        check_midprice_nan_handling,
        check_midprice_streaming,
        check_midprice_all_nan,
        check_midprice_no_poison
    );

    #[cfg(test)]
    generate_all_midprice_tests!(check_midprice_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MidpriceBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "high", "low")?;

        let def = MidpriceParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [59583.0, 59583.0, 59583.0, 59486.0, 58989.0];
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
            (2, 10, 2),    // Small periods
            (5, 25, 5),    // Medium periods
            (30, 60, 15),  // Large periods
            (2, 5, 1),     // Dense small range
            (10, 10, 0),   // Single period (no step)
            (14, 14, 0),   // Default period
            (50, 100, 25), // Very large periods
        ];

        for (cfg_idx, &(period_start, period_end, period_step)) in test_configs.iter().enumerate() {
            let output = MidpriceBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_start, period_end, period_step)
                .apply_candles(&c, "high", "low")?;

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

    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_midprice_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Deterministic synthetic data (no file IO)
        let n = 256usize;
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            let h = 100.0 + 0.1 * t + (0.03 * t).sin();
            let l = h - (2.0 + ((i % 7) as f64));
            high.push(h);
            low.push(l);
        }

        let params = MidpriceParams::default(); // period = 14
        let input = MidpriceInput::from_slices(&high, &low, params.clone());

        // Baseline via existing Vec-returning API
        let baseline = midprice(&input)?.values;

        // Preallocate destination and call into API
        let mut out = vec![0.0; n];
        midprice_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }
        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }
}

#[cfg(feature = "python")]
pub fn register_midprice_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(midprice_py, m)?)?;
    m.add_function(wrap_pyfunction!(midprice_batch_py, m)?)?;
    Ok(())
}
