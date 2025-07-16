//! # Center of Gravity (CG)
//!
//! The Center of Gravity (CG) indicator attempts to measure the "center" of prices
//! over a given window, sometimes used for smoothing or cycle analysis.  
//!
//! ## Parameters
//! - **period**: The window size. Defaults to 10.
//!
//! ## Errors
//! - **EmptyData**: cg: Input data slice is empty.
//! - **InvalidPeriod**: cg: `period` is zero or exceeds the data length.
//! - **AllValuesNaN**: cg: All input data values are `NaN`.
//! - **NotEnoughValidData**: cg: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
//!
//! ## Returns
//! - **`Ok(CgOutput)`** on success, containing a `Vec<f64>` matching input length,
//!   with leading `NaN` until the warm-up period is reached.
//! - **`Err(CgError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for CgInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CgData::Slice(slice) => slice,
            CgData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CgData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CgOutput {
    pub values: Vec<f64>,
}

impl std::ops::Deref for CgOutput {
    type Target = [f64];
    
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl std::ops::DerefMut for CgOutput {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CgParams {
    pub period: Option<usize>,
}

impl Default for CgParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct CgInput<'a> {
    pub data: CgData<'a>,
    pub params: CgParams,
}

impl<'a> CgInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CgParams) -> Self {
        Self {
            data: CgData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CgParams) -> Self {
        Self {
            data: CgData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CgParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CgBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CgBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CgBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<CgOutput, CgError> {
        let p = CgParams {
            period: self.period,
        };
        let i = CgInput::from_candles(c, "close", p);
        cg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CgOutput, CgError> {
        let p = CgParams {
            period: self.period,
        };
        let i = CgInput::from_slice(d, p);
        cg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CgStream, CgError> {
        let p = CgParams {
            period: self.period,
        };
        CgStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CgError {
    #[error("CG: Empty data provided for CG.")]
    EmptyData,
    #[error("CG: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("CG: All values are NaN.")]
    AllValuesNaN,
    #[error("CG: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn cg(input: &CgInput) -> Result<CgOutput, CgError> {
    cg_with_kernel(input, Kernel::Auto)
}

pub fn cg_with_kernel(input: &CgInput, kernel: Kernel) -> Result<CgOutput, CgError> {
    let data: &[f64] = match &input.data {
        CgData::Candles { candles, source } => source_type(candles, source),
        CgData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(CgError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CgError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CgError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // ==== Revert to requiring period + 1 valid points =====
    if (len - first) < (period + 1) {
        return Err(CgError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    // Use helper function to allocate with NaN prefix only where needed
    let mut out = alloc_with_nan_prefix(len, first + period);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => cg_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cg_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cg_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(CgOutput { values: out })
}

// Pre-computed weights for common periods (1.0, 2.0, 3.0, ..., 64.0)
const CG_WEIGHTS: [f64; 64] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
    41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
    51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
    61.0, 62.0, 63.0, 64.0,
];

#[inline(always)]
pub fn cg_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // Start writing at i = first + period
    for i in (first + period)..data.len() {
        let mut num = 0.0;
        let mut denom = 0.0;

        // Sum exactly (period - 1) bars
        if period <= 65 {
            // Use pre-computed weights for common periods
            for count in 0..(period - 1) {
                let price = data[i - count];
                let weight = unsafe { *CG_WEIGHTS.get_unchecked(count) };
                num += weight * price;
                denom += price;
            }
        } else {
            // Fall back to computing weights for large periods
            for count in 0..(period - 1) {
                let price = data[i - count];
                num += (1.0 + count as f64) * price;
                denom += price;
            }
        }

        out[i] = if denom.abs() > f64::EPSILON {
            -num / denom
        } else {
            0.0
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cg_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { cg_avx512_short(data, period, first, out) }
    } else {
        unsafe { cg_avx512_long(data, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cg_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cg_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cg_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct CgStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl CgStream {
    pub fn try_new(params: CgParams) -> Result<Self, CgError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(CgError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // 1) Insert the new value into the ring buffer at `head`
        self.buffer[self.head] = value;

        // 2) Advance head; if we have just filled exactly `period` slots, mark `filled = true` but return None now
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            // We have just completed the first `period` writes—do not emit CG yet
            self.filled = true;
            return None;
        }

        // 3) If still not filled, return None
        if !self.filled {
            return None;
        }

        // 4) Otherwise, compute and return the CG over the last (period - 1) bars
        Some(self.dot_ring())
    }

    #[inline(always)]
    fn dot_ring(&self) -> f64 {
        let mut num = 0.0;
        let mut denom = 0.0;
        let mut idx = self.head;
        // Sum exactly (period - 1) bars, matching cg_scalar’s inner loop
        for k in 0..(self.period - 1) {
            idx = if idx == 0 { self.period - 1 } else { idx - 1 };
            let price = self.buffer[idx];
            num += (1.0 + k as f64) * price;
            denom += price;
        }
        if denom.abs() > f64::EPSILON {
            -num / denom
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
pub struct CgBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CgBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 10, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CgBatchBuilder {
    range: CgBatchRange,
    kernel: Kernel,
}

impl CgBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<CgBatchOutput, CgError> {
        cg_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CgBatchOutput, CgError> {
        CgBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CgBatchOutput, CgError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CgBatchOutput, CgError> {
        CgBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn cg_batch_with_kernel(
    data: &[f64],
    sweep: &CgBatchRange,
    k: Kernel,
) -> Result<CgBatchOutput, CgError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(CgError::InvalidPeriod {
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
    cg_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CgBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CgParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CgBatchOutput {
    pub fn row_for_params(&self, p: &CgParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
    }
    pub fn values_for(&self, p: &CgParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CgBatchRange) -> Vec<CgParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CgParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn cg_batch_slice(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
) -> Result<CgBatchOutput, CgError> {
    cg_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cg_batch_par_slice(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
) -> Result<CgBatchOutput, CgError> {
    cg_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cg_batch_inner(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CgBatchOutput, CgError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CgError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CgError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p + 1 {
        return Err(CgError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    
    // Use helper to allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);
    
    // Calculate warm-up prefixes for each row
    let warm_prefixes: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();
    
    // Initialize only the NaN prefixes
    init_matrix_prefixes(&mut buf_mu, cols, &warm_prefixes);
    
    // Convert to mutable slice for computation
    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
        )
    };
    
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => cg_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cg_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cg_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out
                .par_chunks_mut(cols)
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
    
    // Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity()
        )
    };
    
    Ok(CgBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn cg_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        cg_row_avx512_short(data, first, period, out)
    } else {
        cg_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_cg_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = CgParams { period: Some(12) };
        let input = CgInput::from_candles(&candles, "close", partial_params);
        let output = cg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cg_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -4.99905186931943,
            -4.998559827254377,
            -4.9970065675119555,
            -4.9928483984587295,
            -5.004210799262688,
        ];
        assert!(
            result.values.len() >= 5,
            "Not enough data for final 5-values check"
        );
        let start = result.values.len() - 5;
        for (i, &exp) in expected_last_five.iter().enumerate() {
            let got = result.values[start + i];
            assert!(
                (got - exp).abs() < 1e-4,
                "Mismatch in CG at idx {}: expected={}, got={}",
                start + i,
                exp,
                got
            );
        }
        Ok(())
    }

    fn check_cg_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CgInput::with_default_candles(&candles);
        match input.data {
            CgData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CgData::Candles"),
        }
        let output = cg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cg_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = CgParams { period: Some(0) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for zero period");
        Ok(())
    }

    fn check_cg_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for period > data.len()");
        Ok(())
    }

    fn check_cg_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period=10"
        );
        Ok(())
    }

    fn check_cg_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg_with_kernel(&input, kernel)?;
        let check_idx = 240;
        if result.values.len() > check_idx {
            for i in check_idx..result.values.len() {
                if !result.values[i].is_nan() {
                    break;
                }
                if i == result.values.len() - 1 {
                    panic!("All CG values from index {} onward are NaN.", check_idx);
                }
            }
        }
        Ok(())
    }

    fn check_cg_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 10;
        let input = CgInput::from_candles(
            &candles,
            "close",
            CgParams {
                period: Some(period),
            },
        );
        let batch_output = cg_with_kernel(&input, kernel)?.values;
        let mut stream = CgStream::try_new(CgParams {
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
                "[{}] CG streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_cg_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // Test with multiple parameter combinations to increase coverage
        let test_periods = vec![5, 10, 20, 50];
        
        for period in test_periods {
            let params = CgParams { period: Some(period) };
            let input = CgInput::from_candles(&candles, "close", params);
            let output = cg_with_kernel(&input, kernel)?;
            
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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }
                
                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }
                
                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }
            }
        }
        
        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_cg_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cg_tests {
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

    generate_all_cg_tests!(
        check_cg_partial_params,
        check_cg_accuracy,
        check_cg_default_candles,
        check_cg_zero_period,
        check_cg_period_exceeds_length,
        check_cg_very_small_dataset,
        check_cg_nan_handling,
        check_cg_streaming,
        check_cg_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CgBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = CgParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        
        // Test batch with multiple parameter combinations
        let output = CgBatchBuilder::new()
            .kernel(kernel)
            .period_range(5, 50, 5)  // Test periods from 5 to 50 in steps of 5
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
            let period = output.combos[row].period.unwrap_or(10);
            
            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {}",
                    test, val, bits, row, col, idx, period
                );
            }
            
            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {}",
                    test, val, bits, row, col, idx, period
                );
            }
            
            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {}",
                    test, val, bits, row, col, idx, period
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "cg")]
#[pyo3(signature = (data, period=None, *, kernel=None))]
pub fn cg_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?; // zero-copy, read-only view

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, false)?;

    // ---------- build input struct -------------------------------------------------
    let params = CgParams {
        period: period,  // Now correctly passes the optional period
    };
    let cg_in = CgInput::from_slice(slice_in, params);

    // ---------- allocate uninitialized NumPy output buffer -------------------------
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // ---------- heavy lifting without the GIL --------------------------------------
    py.allow_threads(|| -> Result<(), CgError> {
        // First, validate input and find first valid index
        if slice_in.is_empty() {
            return Err(CgError::EmptyData);
        }
        let first = slice_in
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(CgError::AllValuesNaN)?;
        let len = slice_in.len();
        let period = cg_in.get_period();

        if period == 0 || period > len {
            return Err(CgError::InvalidPeriod {
                period,
                data_len: len,
            });
        }

        // CG requires period + 1 valid points
        if (len - first) < (period + 1) {
            return Err(CgError::NotEnoughValidData {
                needed: period + 1,
                valid: len - first,
            });
        }

        // SAFETY: We must write to ALL elements before returning to Python
        // 1. Write NaN prefix to the first (first + period) elements
        if first + period > 0 {
            slice_out[..first + period].fill(f64::NAN);
        }

        // 2. Compute CG values for the rest
        let chosen = match kern {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        };

        // CG computation writes directly to slice_out
        unsafe {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => cg_scalar(slice_in, period, first, slice_out),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => cg_avx2(slice_in, period, first, slice_out),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => cg_avx512(slice_in, period, first, slice_out),
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                    // Fallback to scalar when AVX is not available
                    cg_scalar(slice_in, period, first, slice_out)
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "CgStream")]
pub struct CgStreamPy {
    stream: CgStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CgStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = CgParams {
            period: Some(period),
        };
        let stream =
            CgStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CgStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated CG value.
    /// Returns `None` if the buffer is not yet full (needs period + 1 values).
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cg_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn cg_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = CgBatchRange {
        period: period_range,
    };

    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // 2. Pre-allocate uninitialized NumPy array (1-D, will reshape later)
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, true)?;

    // 3. Heavy work without the GIL
    let combos = py.allow_threads(|| {
        // Resolve Kernel::Auto to a specific kernel
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

        // Check for empty data and all NaN first
        if slice_in.is_empty() {
            return Err(CgError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
        let first = slice_in
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(CgError::AllValuesNaN)?;
        
        // Check max period requirement
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if slice_in.len() - first < max_p + 1 {
            return Err(CgError::NotEnoughValidData {
                needed: max_p + 1,
                valid: slice_in.len() - first,
            });
        }

        // Fill NaN prefixes for each row
        for (row, combo) in combos.iter().enumerate() {
            let period = combo.period.unwrap();
            let warm = first + period;
            let row_start = row * cols;
            if warm > 0 {
                slice_out[row_start..row_start + warm].fill(f64::NAN);
            }
        }

        // Compute each row
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            match simd {
                Kernel::Scalar => cg_row_scalar(slice_in, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => cg_row_avx2(slice_in, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => cg_row_avx512(slice_in, first, period, out_row),
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx512 => {
                    // Fall back to scalar on non-AVX systems
                    cg_row_scalar(slice_in, first, period, out_row)
                }
                _ => unreachable!(),
            }
        };

        // Use parallel processing
        #[cfg(not(target_arch = "wasm32"))]
        {
            slice_out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in slice_out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }

        Ok(combos)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 4. Build dict with the GIL
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cg_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = CgParams {
        period: Some(period),
    };
    let input = CgInput::from_slice(data, params);

    cg_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cg_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = CgBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    cg_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cg_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = CgBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CgBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CgBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CgParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cg_batch)]
pub fn cg_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: CgBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = CgBatchRange {
        period: config.period_range,
    };

    // 2. Run the existing core logic
    let output = cg_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = CgBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
