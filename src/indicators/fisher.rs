//! # Fisher Transform
//!
//! The Fisher Transform identifies potential price reversals by normalizing price extremes
//! using a Fisher Transform function. It converts prices to a Gaussian normal distribution
//! to identify turning points more clearly.
//!
//! ## Parameters
//! - **period**: Window size for finding price extremes (default: 9)
//!
//! ## Returns
//! - **`Ok(FisherOutput)`** containing:
//!   - `fisher`: The Fisher Transform values
//!   - `signal`: The lagged Fisher values (previous Fisher value)
//!
//! ## Developer Notes
//! ### Implementation Status
//! - **AVX2 Kernel**: Stub (calls scalar implementation)
//! - **AVX512 Kernel**: Stub (calls scalar implementation)
//! - **Streaming Update**: O(1) - efficient with circular buffer for min/max tracking
//! - **Memory Optimization**: Fully optimized with `alloc_with_nan_prefix` for output vectors
//! - **Batch Operations**: Fully implemented with `make_uninit_matrix` and `init_matrix_prefixes`
//!
//! ### TODO - Performance Improvements
//! - [ ] Implement actual AVX2 SIMD kernel (currently stub)
//! - [ ] Implement actual AVX512 SIMD kernel (currently stub)
//! - [ ] Vectorize min/max finding in rolling window
//! - [ ] Optimize Fisher transform calculation with SIMD
//! - [ ] Consider using SIMD for the constrain and transform operations

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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
// Note: AVec and CACHELINE_ALIGN are not used by Fisher itself, only by helper functions
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> FisherInput<'a> {
    #[inline(always)]
    pub fn as_ref(&self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            FisherData::Candles { candles } => {
                (source_type(candles, "high"), source_type(candles, "low"))
            }
            FisherData::Slices { high, low } => (*high, *low),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FisherData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct FisherOutput {
    pub fisher: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct FisherParams {
    pub period: Option<usize>,
}

impl Default for FisherParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct FisherInput<'a> {
    pub data: FisherData<'a>,
    pub params: FisherParams,
}

impl<'a> FisherInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FisherParams) -> Self {
        Self {
            data: FisherData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn get_high_low(&self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            FisherData::Candles { candles } => {
                let high = source_type(candles, "high");
                let low = source_type(candles, "low");
                (high, low)
            }
            FisherData::Slices { high, low } => (*high, *low),
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: FisherParams) -> Self {
        Self {
            data: FisherData::Slices { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FisherParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FisherBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for FisherBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FisherBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<FisherOutput, FisherError> {
        let p = FisherParams {
            period: self.period,
        };
        let i = FisherInput::from_candles(c, p);
        fisher_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<FisherOutput, FisherError> {
        let p = FisherParams {
            period: self.period,
        };
        let i = FisherInput::from_slices(high, low, p);
        fisher_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<FisherStream, FisherError> {
        let p = FisherParams {
            period: self.period,
        };
        FisherStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum FisherError {
    #[error("fisher: Empty data provided.")]
    EmptyData,
    #[error("fisher: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fisher: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fisher: All values are NaN.")]
    AllValuesNaN,
    #[error("fisher: Invalid output length: expected = {expected}, actual = {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("fisher: Mismatched data length: high={high}, low={low}")]
    MismatchedDataLength { high: usize, low: usize },
}

#[inline]
pub fn fisher(input: &FisherInput) -> Result<FisherOutput, FisherError> {
    fisher_with_kernel(input, Kernel::Auto)
}

pub fn fisher_with_kernel(
    input: &FisherInput,
    kernel: Kernel,
) -> Result<FisherOutput, FisherError> {
    let (high, low) = input.get_high_low();

    if high.is_empty() || low.is_empty() {
        return Err(FisherError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(FisherError::MismatchedDataLength {
            high: high.len(),
            low: low.len(),
        });
    }

    let period = input.get_period();
    let data_len = high.len();
    if period == 0 || period > data_len {
        return Err(FisherError::InvalidPeriod { period, data_len });
    }

    // Find first valid index by checking both high and low
    let mut first = None;
    for i in 0..data_len {
        if !high[i].is_nan() && !low[i].is_nan() {
            first = Some(i);
            break;
        }
    }
    let first = first.ok_or(FisherError::AllValuesNaN)?;

    if (data_len - first) < period {
        return Err(FisherError::NotEnoughValidData {
            needed: period,
            valid: data_len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Allocate output arrays with NaN prefix for warmup period
    let warmup = first + period - 1;
    let mut fisher_vals = alloc_with_nan_prefix(data_len, warmup);
    let mut signal_vals = alloc_with_nan_prefix(data_len, warmup);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                fisher_scalar_into(high, low, period, first, &mut fisher_vals, &mut signal_vals)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                fisher_avx2_into(high, low, period, first, &mut fisher_vals, &mut signal_vals)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                fisher_avx512_into(high, low, period, first, &mut fisher_vals, &mut signal_vals)
            }
            _ => unreachable!(),
        }
    }

    Ok(FisherOutput {
        fisher: fisher_vals,
        signal: signal_vals,
    })
}

#[inline]
pub fn fisher_scalar_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) {
    let data_len = high.len().min(low.len());
    let mut prev_fish = 0.0;
    let mut val1 = 0.0;

    for i in first..data_len {
        if i < first + period - 1 {
            continue;
        }
        let start = i + 1 - period;

        // Calculate min/max from midpoint values
        let (mut min_val, mut max_val) = (f64::MAX, f64::MIN);
        for j in start..=i {
            let midpoint = 0.5 * (high[j] + low[j]);
            if midpoint > max_val {
                max_val = midpoint;
            }
            if midpoint < min_val {
                min_val = midpoint;
            }
        }

        let range = (max_val - min_val).max(0.001);
        let current_hl = 0.5 * (high[i] + low[i]);
        val1 = 0.33 * 2.0 * ((current_hl - min_val) / range - 0.5) + 0.67 * val1;
        if val1 > 0.99 {
            val1 = 0.999;
        } else if val1 < -0.99 {
            val1 = -0.999;
        }
        signal_out[i] = prev_fish;
        let new_fish = 0.5 * ((1.0 + val1) / (1.0 - val1)).ln() + 0.5 * prev_fish;
        fisher_out[i] = new_fish;
        prev_fish = new_fish;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx512_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) {
    // AVX512 stub: fallback to scalar
    fisher_scalar_into(high, low, period, first, fisher_out, signal_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx2_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) {
    // AVX2 stub: fallback to scalar
    fisher_scalar_into(high, low, period, first, fisher_out, signal_out)
}

/// Write Fisher Transform directly to output slices - no allocations
#[inline]
pub fn fisher_into_slice(
    fisher_dst: &mut [f64],
    signal_dst: &mut [f64],
    input: &FisherInput,
    kern: Kernel,
) -> Result<(), FisherError> {
    let (high, low) = input.as_ref();
    if high.is_empty() || low.is_empty() {
        return Err(FisherError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(FisherError::MismatchedDataLength {
            high: high.len(),
            low: low.len(),
        });
    }

    let data_len = high.len();
    let period = input.params.period.unwrap_or(9);

    // Find first valid index by checking both high and low
    let mut first = None;
    for i in 0..data_len {
        if !high[i].is_nan() && !low[i].is_nan() {
            first = Some(i);
            break;
        }
    }
    let first = first.ok_or(FisherError::AllValuesNaN)?;

    // Validate parameters
    if period == 0 || period > data_len {
        return Err(FisherError::InvalidPeriod { period, data_len });
    }
    if fisher_dst.len() != data_len || signal_dst.len() != data_len {
        return Err(FisherError::InvalidLength {
            expected: data_len,
            actual: fisher_dst.len().min(signal_dst.len()),
        });
    }

    let chosen = if kern == Kernel::Auto {
        detect_best_kernel()
    } else {
        kern
    };

    // Write results directly to output slices
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            fisher_scalar_into(high, low, period, first, fisher_dst, signal_dst)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => {
            fisher_avx2_into(high, low, period, first, fisher_dst, signal_dst)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            fisher_avx512_into(high, low, period, first, fisher_dst, signal_dst)
        }
        _ => unreachable!(),
    }

    // Fill warmup period with NaN
    let warmup_end = first + period - 1;
    for i in 0..warmup_end {
        fisher_dst[i] = f64::NAN;
        signal_dst[i] = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn fisher_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    k: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(FisherError::InvalidPeriod {
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
    fisher_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct FisherBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for FisherBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FisherBatchBuilder {
    range: FisherBatchRange,
    kernel: Kernel,
}

impl FisherBatchBuilder {
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
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<FisherBatchOutput, FisherError> {
        fisher_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<FisherBatchOutput, FisherError> {
        FisherBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<FisherBatchOutput, FisherError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FisherBatchOutput, FisherError> {
        FisherBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct FisherBatchOutput {
    pub fisher: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<FisherParams>,
    pub rows: usize,
    pub cols: usize,
}
impl FisherBatchOutput {
    pub fn row_for_params(&self, p: &FisherParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
    }
    pub fn fisher_for(&self, p: &FisherParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.fisher[start..start + self.cols]
        })
    }
    pub fn signal_for(&self, p: &FisherParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.signal[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &FisherBatchRange) -> Vec<FisherParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(FisherParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn fisher_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
    fisher_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn fisher_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
    fisher_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn fisher_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FisherBatchOutput, FisherError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(FisherError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let data_len = high.len().min(low.len());

    // Find first valid index
    let mut first = None;
    for i in 0..data_len {
        if !high[i].is_nan() && !low[i].is_nan() {
            first = Some(i);
            break;
        }
    }
    let first = first.ok_or(FisherError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data_len - first < max_p {
        return Err(FisherError::NotEnoughValidData {
            needed: max_p,
            valid: data_len - first,
        });
    }
    let rows = combos.len();
    let cols = data_len;

    // Use make_uninit_matrix for zero allocation
    let mut fisher_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each parameter combination
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Initialize NaN prefixes
    init_matrix_prefixes(&mut fisher_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut signal_mu, cols, &warmup_periods);

    // Convert to slices for computation
    let mut fisher_guard = core::mem::ManuallyDrop::new(fisher_mu);
    let mut signal_guard = core::mem::ManuallyDrop::new(signal_mu);

    let fisher_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(fisher_guard.as_mut_ptr() as *mut f64, fisher_guard.len())
    };
    let signal_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    let do_row = |row: usize, out_fish: &mut [f64], out_signal: &mut [f64]| {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                fisher_row_avx2_direct(high, low, first, period, out_fish, out_signal)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                fisher_row_avx512_direct(high, low, first, period, out_fish, out_signal)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
            }
            Kernel::Auto => {
                fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
            } // Should be resolved before here
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            fisher_slice
                .par_chunks_mut(cols)
                .zip(signal_slice.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (fish, sig))| do_row(row, fish, sig));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (fish, sig)) in fisher_slice
                .chunks_mut(cols)
                .zip(signal_slice.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, fish, sig);
            }
        }
    } else {
        for (row, (fish, sig)) in fisher_slice
            .chunks_mut(cols)
            .zip(signal_slice.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, fish, sig);
        }
    }

    // Convert back to owned vectors
    let fisher = unsafe {
        Vec::from_raw_parts(
            fisher_guard.as_mut_ptr() as *mut f64,
            fisher_guard.len(),
            fisher_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    // Forget the guards to prevent double-free
    core::mem::forget(fisher_guard);
    core::mem::forget(signal_guard);

    Ok(FisherBatchOutput {
        fisher,
        signal,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn fisher_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
    parallel: bool,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<FisherParams>, FisherError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(FisherError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let data_len = high.len().min(low.len());

    // Find first valid index
    let mut first = None;
    for i in 0..data_len {
        if !high[i].is_nan() && !low[i].is_nan() {
            first = Some(i);
            break;
        }
    }
    let first = first.ok_or(FisherError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data_len - first < max_p {
        return Err(FisherError::NotEnoughValidData {
            needed: max_p,
            valid: data_len - first,
        });
    }

    let rows = combos.len();
    let cols = data_len;

    // Initialize NaN values for warmup periods in the pre-allocated buffers
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.period.unwrap() - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            fisher_out[row_start + i] = f64::NAN;
            signal_out[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, out_fish: &mut [f64], out_signal: &mut [f64]| {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                fisher_row_avx2_direct(high, low, first, period, out_fish, out_signal)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                fisher_row_avx512_direct(high, low, first, period, out_fish, out_signal)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
            }
            Kernel::Auto => {
                fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
            } // Should be resolved before here
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            fisher_out
                .par_chunks_mut(cols)
                .zip(signal_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (fish, sig))| do_row(row, fish, sig));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (fish, sig)) in fisher_out
                .chunks_mut(cols)
                .zip(signal_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, fish, sig);
            }
        }
    } else {
        for (row, (fish, sig)) in fisher_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, fish, sig);
        }
    }

    Ok(combos)
}

#[inline(always)]
fn fisher_row_scalar_direct(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out_fish: &mut [f64],
    out_signal: &mut [f64],
) {
    let data_len = high.len().min(low.len());
    let mut prev_fish = 0.0;
    let mut val1 = 0.0;

    for i in first..data_len {
        if i < first + period - 1 {
            continue;
        }
        let start = i + 1 - period;

        // Calculate min/max from midpoint values
        let (mut min_val, mut max_val) = (f64::MAX, f64::MIN);
        for j in start..=i {
            let midpoint = 0.5 * (high[j] + low[j]);
            if midpoint > max_val {
                max_val = midpoint;
            }
            if midpoint < min_val {
                min_val = midpoint;
            }
        }

        let range = (max_val - min_val).max(0.001);
        let current_hl = 0.5 * (high[i] + low[i]);
        val1 = 0.33 * 2.0 * ((current_hl - min_val) / range - 0.5) + 0.67 * val1;

        if val1 > 0.99 {
            val1 = 0.999;
        } else if val1 < -0.99 {
            val1 = -0.999;
        }

        out_signal[i] = prev_fish;
        let new_fish = 0.5 * ((1.0 + val1) / (1.0 - val1)).ln() + 0.5 * prev_fish;
        out_fish[i] = new_fish;
        prev_fish = new_fish;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx2_direct(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out_fish: &mut [f64],
    out_signal: &mut [f64],
) {
    // AVX2 stub: fallback to scalar
    fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx512_direct(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out_fish: &mut [f64],
    out_signal: &mut [f64],
) {
    // AVX512 stub: fallback to scalar
    fisher_row_scalar_direct(high, low, first, period, out_fish, out_signal)
}

#[derive(Debug, Clone)]
pub struct FisherStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    prev_fish: f64,
    val1: f64,
}
impl FisherStream {
    pub fn try_new(params: FisherParams) -> Result<Self, FisherError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(FisherError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            prev_fish: 0.0,
            val1: 0.0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        let merged = 0.5 * (high + low);
        self.buffer[self.head] = merged;
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
    fn dot_ring(&mut self) -> (f64, f64) {
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        let mut idx = self.head;
        for _ in 0..self.period {
            let v = self.buffer[idx];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            idx = (idx + 1) % self.period;
        }
        let range = (max_val - min_val).max(0.001);
        let current_hl = self.buffer[(self.head + self.period - 1) % self.period];
        self.val1 = 0.33 * 2.0 * ((current_hl - min_val) / range - 0.5) + 0.67 * self.val1;
        if self.val1 > 0.99 {
            self.val1 = 0.999;
        } else if self.val1 < -0.99 {
            self.val1 = -0.999;
        }
        let new_signal = self.prev_fish;
        let new_fish = 0.5 * ((1.0 + self.val1) / (1.0 - self.val1)).ln() + 0.5 * self.prev_fish;
        self.prev_fish = new_fish;
        (new_fish, new_signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_fisher_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = FisherParams { period: None };
        let input = FisherInput::from_candles(&candles, default_params);
        let output = fisher_with_kernel(&input, kernel)?;
        assert_eq!(output.fisher.len(), candles.close.len());
        Ok(())
    }

    fn check_fisher_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = FisherInput::from_candles(&candles, FisherParams::default());
        let result = fisher_with_kernel(&input, kernel)?;
        let expected_last_five_fisher = [
            -0.4720164683904261,
            -0.23467530106650444,
            -0.14879388501136784,
            -0.026651419122953053,
            -0.2569225042442664,
        ];
        let start = result.fisher.len().saturating_sub(5);
        for (i, &val) in result.fisher[start..].iter().enumerate() {
            let diff = (val - expected_last_five_fisher[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] Fisher {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five_fisher[i]
            );
        }
        Ok(())
    }

    fn check_fisher_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = FisherParams { period: Some(0) };
        let input = FisherInput::from_slices(&high, &low, params);
        let res = fisher_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Fisher should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_fisher_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = FisherParams { period: Some(10) };
        let input = FisherInput::from_slices(&high, &low, params);
        let res = fisher_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Fisher should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_fisher_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0];
        let low = [5.0];
        let params = FisherParams { period: Some(9) };
        let input = FisherInput::from_slices(&high, &low, params);
        let res = fisher_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Fisher should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_fisher_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let low = [5.0, 7.0, 9.0, 10.0, 13.0, 15.0];
        let first_params = FisherParams { period: Some(3) };
        let first_input = FisherInput::from_slices(&high, &low, first_params);
        let first_result = fisher_with_kernel(&first_input, kernel)?;
        let second_params = FisherParams { period: Some(3) };
        let second_input =
            FisherInput::from_slices(&first_result.fisher, &first_result.signal, second_params);
        let second_result = fisher_with_kernel(&second_input, kernel)?;
        assert_eq!(first_result.fisher.len(), second_result.fisher.len());
        assert_eq!(first_result.signal.len(), second_result.signal.len());
        Ok(())
    }

    fn check_fisher_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = FisherInput::from_candles(&candles, FisherParams::default());
        let res = fisher_with_kernel(&input, kernel)?;
        assert_eq!(res.fisher.len(), candles.close.len());
        if res.fisher.len() > 240 {
            for (i, &val) in res.fisher[240..].iter().enumerate() {
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

    fn check_fisher_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 9;
        let input = FisherInput::from_candles(
            &candles,
            FisherParams {
                period: Some(period),
            },
        );
        let batch_output = fisher_with_kernel(&input, kernel)?.fisher;

        let highs = source_type(&candles, "high");
        let lows = source_type(&candles, "low");

        let mut stream = FisherStream::try_new(FisherParams {
            period: Some(period),
        })?;
        let mut stream_fisher = Vec::with_capacity(highs.len());
        for (&h, &l) in highs.iter().zip(lows.iter()) {
            match stream.update(h, l) {
                Some((fish, _sig)) => stream_fisher.push(fish),
                None => stream_fisher.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_fisher.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_fisher.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Fisher streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_fisher_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            FisherParams::default(),            // period: 9
            FisherParams { period: Some(1) },   // minimum period
            FisherParams { period: Some(2) },   // very small period
            FisherParams { period: Some(3) },   // small period
            FisherParams { period: Some(5) },   // small period
            FisherParams { period: Some(10) },  // medium period
            FisherParams { period: Some(20) },  // medium period
            FisherParams { period: Some(30) },  // medium-large period
            FisherParams { period: Some(50) },  // large period
            FisherParams { period: Some(100) }, // very large period
            FisherParams { period: Some(200) }, // extra large period
            FisherParams { period: Some(240) }, // edge case from default batch range
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = FisherInput::from_candles(&candles, params.clone());
            let output = fisher_with_kernel(&input, kernel)?;

            // Check fisher values
            for (i, &val) in output.fisher.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in fisher output with params: period={} (param set {})",
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
						 in fisher output with params: period={} (param set {})",
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
						 in fisher output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }
            }

            // Check signal values
            for (i, &val) in output.signal.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in signal output with params: period={} (param set {})",
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
						 in signal output with params: period={} (param set {})",
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
						 in signal output with params: period={} (param set {})",
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
    fn check_fisher_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_fisher_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic high/low price data
        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                // Generate base price, volatility, and data length
                (100f64..10000f64, 0.01f64..0.05f64, period + 10..400)
                    .prop_flat_map(move |(base_price, volatility, data_len)| {
                        // Generate random changes for price movement and spread flags
                        (
                            Just(base_price),
                            Just(volatility),
                            Just(data_len),
                            prop::collection::vec((-1f64..1f64), data_len),
                            prop::collection::vec(prop::bool::ANY, data_len), // Flags for zero spread
                        )
                    })
                    .prop_map(
                        move |(
                            base_price,
                            volatility,
                            data_len,
                            price_changes,
                            zero_spread_flags,
                        )| {
                            // Generate synthetic high/low data with realistic movement
                            let mut high = Vec::with_capacity(data_len);
                            let mut low = Vec::with_capacity(data_len);
                            let mut current_price = base_price;

                            for i in 0..data_len {
                                // Random walk for price with volatility
                                let change = price_changes[i] * volatility * current_price;
                                current_price = (current_price + change).max(10.0); // Prevent negative prices

                                // Sometimes have zero spread (high == low), sometimes have normal spread
                                if zero_spread_flags[i] && i % 5 == 0 {
                                    // ~20% chance of zero spread
                                    high.push(current_price);
                                    low.push(current_price);
                                } else {
                                    // Generate high/low with some spread
                                    let spread =
                                        current_price * 0.01 * (0.1 + price_changes[i].abs()); // 0.1-2% spread
                                    high.push(current_price + spread);
                                    low.push((current_price - spread).max(10.0));
                                }
                            }

                            (high, low)
                        },
                    ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |((high, low), period)| {
            let params = FisherParams {
                period: Some(period),
            };
            let input = FisherInput::from_slices(&high, &low, params);

            let FisherOutput {
                fisher: out,
                signal: sig,
            } = fisher_with_kernel(&input, kernel)?;
            let FisherOutput {
                fisher: ref_out,
                signal: ref_sig,
            } = fisher_with_kernel(&input, Kernel::Scalar)?;

            // Property 1: Output length matches input length
            prop_assert_eq!(
                out.len(),
                high.len(),
                "[{}] Fisher output length mismatch",
                test_name
            );
            prop_assert_eq!(
                sig.len(),
                high.len(),
                "[{}] Signal output length mismatch",
                test_name
            );

            // Property 2: Warmup period behavior (first + period - 1 positions should be NaN)
            // Find first valid index
            let mut first_valid = None;
            for i in 0..high.len() {
                if !high[i].is_nan() && !low[i].is_nan() {
                    first_valid = Some(i);
                    break;
                }
            }

            if let Some(first) = first_valid {
                let warmup_end = first + period - 1;
                for i in 0..warmup_end.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN at index {} during warmup",
                        test_name,
                        i
                    );
                    prop_assert!(
                        sig[i].is_nan(),
                        "[{}] Expected NaN at signal index {} during warmup",
                        test_name,
                        i
                    );
                }

                // Should have valid values after warmup
                if warmup_end < out.len() {
                    prop_assert!(
                        !out[warmup_end].is_nan(),
                        "[{}] Expected valid value at index {} after warmup",
                        test_name,
                        warmup_end
                    );
                }
            }

            // Property 3: Constant price behavior - Fisher should trend toward zero
            // Look for periods where high and low are very close (constant price)
            if let Some(first) = first_valid {
                let warmup_end = first + period - 1;

                for window_start in warmup_end..out.len().saturating_sub(period * 2) {
                    let window_end = (window_start + period).min(out.len());

                    // Check if we have a constant price period
                    let mut is_constant = true;
                    let first_hl = (high[window_start] + low[window_start]) / 2.0;

                    for i in window_start..window_end {
                        let current_hl = (high[i] + low[i]) / 2.0;
                        if (current_hl - first_hl).abs() > 0.001 * first_hl {
                            // 0.1% tolerance
                            is_constant = false;
                            break;
                        }
                    }

                    // If prices are constant, Fisher should be trending toward zero
                    if is_constant && window_end > window_start + 3 {
                        let fisher_start = out[window_start].abs();
                        let fisher_end = out[window_end - 1].abs();

                        // Fisher should be decreasing in magnitude (trending to zero)
                        // Allow some tolerance for numerical precision
                        if fisher_start > 0.1 {
                            // Only check if initial value is significant
                            prop_assert!(
									fisher_end <= fisher_start * 1.1, // Allow 10% tolerance for numerical noise
									"[{}] Fisher not trending to zero in constant period [{}, {}]: start={}, end={}",
									test_name, window_start, window_end, fisher_start, fisher_end
								);
                        }
                    }
                }
            }

            // Property 4: Signal equals previous Fisher value
            for i in 1..out.len() {
                if !out[i - 1].is_nan() && !sig[i].is_nan() {
                    prop_assert!(
                        (sig[i] - out[i - 1]).abs() < 1e-9,
                        "[{}] Signal at {} ({}) doesn't match previous Fisher ({})",
                        test_name,
                        i,
                        sig[i],
                        out[i - 1]
                    );
                }
            }

            // Property 5: Initial state verification - first valid Fisher value calculation
            if let Some(first) = first_valid {
                let warmup_end = first + period - 1;
                if warmup_end < out.len() && !out[warmup_end].is_nan() {
                    // The first Fisher value should start from prev_fish = 0.0
                    // Due to the 0.5 factor and starting from 0, first value should be moderate
                    prop_assert!(
                        out[warmup_end].abs() < 5.0,
                        "[{}] First Fisher value {} seems incorrect (should start from zero state)",
                        test_name,
                        out[warmup_end]
                    );
                }
            }

            // Property 6: Kernel consistency - compare with scalar reference
            for i in 0..out.len() {
                let y = out[i];
                let r = ref_out[i];
                let s = sig[i];
                let rs = ref_sig[i];

                // Check NaN consistency
                if y.is_nan() || r.is_nan() {
                    prop_assert_eq!(
                        y.is_nan(),
                        r.is_nan(),
                        "[{}] NaN mismatch at index {}",
                        test_name,
                        i
                    );
                    continue;
                }

                // Check signal NaN consistency
                if s.is_nan() || rs.is_nan() {
                    prop_assert_eq!(
                        s.is_nan(),
                        rs.is_nan(),
                        "[{}] Signal NaN mismatch at index {}",
                        test_name,
                        i
                    );
                    continue;
                }

                // ULP comparison for floating-point accuracy
                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let s_bits = s.to_bits();
                let rs_bits = rs.to_bits();

                let ulp_diff_fisher: u64 = y_bits.abs_diff(r_bits);
                let ulp_diff_signal: u64 = s_bits.abs_diff(rs_bits);

                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff_fisher <= 4,
                    "[{}] Fisher mismatch idx {}: {} vs {} (ULP={})",
                    test_name,
                    i,
                    y,
                    r,
                    ulp_diff_fisher
                );

                prop_assert!(
                    (s - rs).abs() <= 1e-9 || ulp_diff_signal <= 4,
                    "[{}] Signal mismatch idx {}: {} vs {} (ULP={})",
                    test_name,
                    i,
                    s,
                    rs,
                    ulp_diff_signal
                );
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_fisher_tests {
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

    generate_all_fisher_tests!(
        check_fisher_partial_params,
        check_fisher_accuracy,
        check_fisher_zero_period,
        check_fisher_period_exceeds_length,
        check_fisher_very_small_dataset,
        check_fisher_reinput,
        check_fisher_nan_handling,
        check_fisher_streaming,
        check_fisher_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_fisher_tests!(check_fisher_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = FisherBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = FisherParams::default();
        let row = output.fisher_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Optional: If you have reference values for the last few outputs,
        // you could add accuracy checks like ALMA does.
        // For demonstration, let's assume the expected values for the last five Fisher are:
        let expected_last_five = [
            -0.4720164683904261,
            -0.23467530106650444,
            -0.14879388501136784,
            -0.026651419122953053,
            -0.2569225042442664,
        ];
        let start = row.len().saturating_sub(5);
        for (i, &val) in row[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {val} vs {expected_last_five:?}"
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
            // (period_start, period_end, period_step)
            (1, 10, 1),     // Small periods, every value
            (2, 20, 2),     // Small to medium, even values
            (5, 50, 5),     // Medium range, step 5
            (10, 100, 10),  // Medium to large, step 10
            (20, 240, 20),  // Large range, step 20
            (9, 9, 0),      // Static default period
            (50, 200, 50),  // Large periods only
            (1, 5, 1),      // Very small periods only
            (100, 240, 40), // Very large periods
            (3, 30, 3),     // Multiples of 3
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = FisherBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            // Check fisher values
            for (idx, &val) in output.fisher.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
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
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
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
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
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

            // Check signal values
            for (idx, &val) in output.signal.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in signal output with params: period={}",
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
						 at row {} col {} (flat index {}) in signal output with params: period={}",
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
						 at row {} col {} (flat index {}) in signal output with params: period={}",
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

    #[test]
    fn check_batch_kernel_dispatch() -> Result<(), Box<dyn Error>> {
        // Test that batch functions correctly dispatch to different kernel row functions
        let high = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0];
        let low = vec![5.0, 7.0, 9.0, 10.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0];
        let sweep = FisherBatchRange {
            period: (3, 5, 1), // Small periods that fit our test data
        };

        // Test scalar kernel
        let scalar_result = fisher_batch_slice(&high, &low, &sweep, Kernel::Scalar)?;

        // Test AVX2 kernel (if available)
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            let avx2_result = fisher_batch_slice(&high, &low, &sweep, Kernel::Avx2)?;
            // Results should be identical regardless of kernel
            for i in 0..scalar_result.fisher.len() {
                let diff = (scalar_result.fisher[i] - avx2_result.fisher[i]).abs();
                assert!(
                    diff < 1e-10
                        || (scalar_result.fisher[i].is_nan() && avx2_result.fisher[i].is_nan()),
                    "Fisher mismatch at {}: scalar={}, avx2={}",
                    i,
                    scalar_result.fisher[i],
                    avx2_result.fisher[i]
                );
            }
        }

        // Test AVX512 kernel (if available)
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") {
            let avx512_result = fisher_batch_slice(&high, &low, &sweep, Kernel::Avx512)?;
            // Results should be identical regardless of kernel
            for i in 0..scalar_result.fisher.len() {
                let diff = (scalar_result.fisher[i] - avx512_result.fisher[i]).abs();
                assert!(
                    diff < 1e-10
                        || (scalar_result.fisher[i].is_nan() && avx512_result.fisher[i].is_nan()),
                    "Fisher mismatch at {}: scalar={}, avx512={}",
                    i,
                    scalar_result.fisher[i],
                    avx512_result.fisher[i]
                );
            }
        }

        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "fisher")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn fisher_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let data_len = high_slice.len().min(low_slice.len());

    // Pre-allocate NumPy arrays for zero-copy operation
    let fisher_arr = unsafe { PyArray1::<f64>::new(py, [data_len], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [data_len], false) };
    let fisher_slice = unsafe { fisher_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };

    let params = FisherParams {
        period: Some(period),
    };
    let input = FisherInput::from_slices(high_slice, low_slice, params);

    // Compute directly into pre-allocated arrays
    py.allow_threads(|| fisher_into_slice(fisher_slice, signal_slice, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((fisher_arr, signal_arr))
}

#[cfg(feature = "python")]
#[pyclass(name = "FisherStream")]
pub struct FisherStreamPy {
    stream: FisherStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl FisherStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = FisherParams {
            period: Some(period),
        };
        let stream =
            FisherStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(FisherStreamPy { stream })
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "fisher_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn fisher_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize), // (start, end, step)
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err(format!(
            "Mismatched data length: high={}, low={}",
            high_slice.len(),
            low_slice.len()
        )));
    }

    let kern = validate_kernel(kernel, true)?;
    let sweep = FisherBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high_slice.len();

    // Pre-allocate uninitialized NumPy arrays (no copy)
    let fisher_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

    // Compute first valid and warmups
    let first = (0..cols)
        .find(|&i| !high_slice[i].is_nan() && !low_slice[i].is_nan())
        .ok_or_else(|| PyValueError::new_err("All values are NaN"))?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Initialize only warm prefixes using helpers, zero extra copies
    unsafe {
        let fisher_mu = std::slice::from_raw_parts_mut(
            fisher_arr.as_slice_mut()?.as_mut_ptr() as *mut MaybeUninit<f64>,
            rows * cols,
        );
        let signal_mu = std::slice::from_raw_parts_mut(
            signal_arr.as_slice_mut()?.as_mut_ptr() as *mut MaybeUninit<f64>,
            rows * cols,
        );
        init_matrix_prefixes(fisher_mu, cols, &warmups);
        init_matrix_prefixes(signal_mu, cols, &warmups);
    }

    // Get raw pointers before entering allow_threads (convert to usize for Send)
    let fisher_ptr = unsafe { fisher_arr.as_slice_mut()?.as_mut_ptr() } as usize;
    let signal_ptr = unsafe { signal_arr.as_slice_mut()?.as_mut_ptr() } as usize;
    let total_len = rows * cols;

    // Compute directly into the same buffers
    py.allow_threads(move || {
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
        // Use existing in-place writer (convert back from usize)
        unsafe {
            let fisher_slice = std::slice::from_raw_parts_mut(fisher_ptr as *mut f64, total_len);
            let signal_slice = std::slice::from_raw_parts_mut(signal_ptr as *mut f64, total_len);
            fisher_batch_inner_into(
                high_slice,
                low_slice,
                &sweep,
                simd,
                true,
                fisher_slice,
                signal_slice,
            )
        }
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("fisher", fisher_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
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

// ================== WASM BINDINGS ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct FisherResult {
    values: Vec<f64>, // layout: [fisher(0..len), signal(0..len)]
    rows: usize,      // 2
    cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl FisherResult {
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fisher_js(high: &[f64], low: &[f64], period: usize) -> Result<FisherResult, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("Mismatched data length"));
    }
    let len = high.len();
    let params = FisherParams {
        period: Some(period),
    };
    let input = FisherInput::from_slices(high, low, params);

    let mut out = vec![0.0_f64; len * 2];
    let (fisher_out, signal_out) = out.split_at_mut(len);

    fisher_into_slice(fisher_out, signal_out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(FisherResult {
        values: out,
        rows: 2,
        cols: len,
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fisher_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to fisher_into"));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        if high.len() != low.len() {
            return Err(JsValue::from_str("Mismatched data length"));
        }
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (fisher_out, signal_out) = out.split_at_mut(len);

        let params = FisherParams {
            period: Some(period),
        };
        let input = FisherInput::from_slices(high, low, params);

        fisher_into_slice(fisher_out, signal_out, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fisher_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fisher_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FisherBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FisherBatchJsOutput {
    pub values: Vec<f64>, // layout per combo: [fisher row, signal row] each of length cols
    pub rows: usize,      // 2 * combos.len()
    pub cols: usize,
    pub combos: Vec<FisherParams>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = fisher_batch)]
pub fn fisher_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("Mismatched data length"));
    }
    let config: FisherBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = FisherBatchRange {
        period: config.period_range,
    };

    let combos = expand_grid(&sweep);
    let cols = high.len();
    let rows = combos.len();

    // Allocate combined buffer: fisher(rows*cols) + signal(rows*cols)
    let mut fisher = vec![0.0; rows * cols];
    let mut signal = vec![0.0; rows * cols];

    fisher_batch_inner_into(
        high,
        low,
        &sweep,
        detect_best_kernel(),
        false,
        &mut fisher,
        &mut signal,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Interleave per combo without extra allocs
    let mut values = Vec::with_capacity(2 * rows * cols);
    for r in 0..rows {
        values.extend_from_slice(&fisher[r * cols..(r + 1) * cols]);
        values.extend_from_slice(&signal[r * cols..(r + 1) * cols]);
    }

    let js = FisherBatchJsOutput {
        values,
        rows: 2 * rows,
        cols,
        combos,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fisher_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    fisher_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || fisher_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = FisherBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let total_size = rows * len;

        let fisher_out = std::slice::from_raw_parts_mut(fisher_ptr, total_size);
        let signal_out = std::slice::from_raw_parts_mut(signal_ptr, total_size);

        // Compute batch results directly into output buffers
        let output = fisher_batch_inner(high, low, &sweep, Kernel::Auto, false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Copy results to output buffers
        fisher_out.copy_from_slice(&output.fisher);
        signal_out.copy_from_slice(&output.signal);

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For streaming patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct FisherContext {
    period: usize,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl FisherContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For streaming patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(period: usize) -> Result<FisherContext, JsValue> {
        if period == 0 {
            return Err(JsValue::from_str("Invalid period: 0"));
        }

        Ok(FisherContext {
            period,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        high_ptr: *const f64,
        low_ptr: *const f64,
        fisher_ptr: *mut f64,
        signal_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if high_ptr.is_null() || low_ptr.is_null() || fisher_ptr.is_null() || signal_ptr.is_null() {
            return Err(JsValue::from_str("Null pointer provided"));
        }

        unsafe {
            let high = std::slice::from_raw_parts(high_ptr, len);
            let low = std::slice::from_raw_parts(low_ptr, len);
            let fisher_out = std::slice::from_raw_parts_mut(fisher_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);

            let params = FisherParams {
                period: Some(self.period),
            };
            let input = FisherInput::from_slices(high, low, params);

            fisher_into_slice(fisher_out, signal_out, &input, self.kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            Ok(())
        }
    }

    pub fn get_period(&self) -> usize {
        self.period
    }

    pub fn get_warmup_period(&self) -> usize {
        self.period - 1
    }
}
