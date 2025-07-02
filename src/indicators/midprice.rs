//! # Midprice
//!
//! The midpoint price over a specified period, calculated as `(highest high + lowest low) / 2`.
//! Useful for identifying average price levels in a range and potential support/resistance.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 14.
//!
//! ## Errors
//! - **EmptyData**: midprice: Input data slice is empty.
//! - **InvalidPeriod**: midprice: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: midprice: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: midprice: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(MidpriceOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the window is filled.
//! - **`Err(MidpriceError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;
use std::convert::AsRef;

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
        let i = MidpriceInput::with_default_candles(c);
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
}

// --- Kernel/Dispatch API ---
#[inline]
pub fn midprice(input: &MidpriceInput) -> Result<MidpriceOutput, MidpriceError> {
    midprice_with_kernel(input, Kernel::Auto)
}

pub fn midprice_with_kernel(input: &MidpriceInput, kernel: Kernel) -> Result<MidpriceOutput, MidpriceError> {
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
        return Err(MidpriceError::EmptyData);
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
    let mut out = vec![f64::NAN; high.len()];
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
            _ => unreachable!(),
        }
    }
    Ok(MidpriceOutput { values: out })
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
#[derive(Debug, Clone)]
pub struct MidpriceStream {
    period: usize,
    high_buffer: Vec<f64>,
    low_buffer: Vec<f64>,
    head: usize,
    filled: bool,
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
        Ok(Self {
            period,
            high_buffer: vec![f64::NAN; period],
            low_buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.high_buffer[self.head] = high;
        self.low_buffer[self.head] = low;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(self.calc())
    }
    #[inline(always)]
    fn calc(&self) -> f64 {
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        let mut idx = self.head;
        for _ in 0..self.period {
            let h = self.high_buffer[idx];
            let l = self.low_buffer[idx];
            if h > highest {
                highest = h;
            }
            if l < lowest {
                lowest = l;
            }
            idx = (idx + 1) % self.period;
        }
        (highest + lowest) / 2.0
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
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MidpriceBatchOutput, MidpriceError> {
        midprice_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles, high_src: &str, low_src: &str) -> Result<MidpriceBatchOutput, MidpriceError> {
        let high = source_type(c, high_src);
        let low = source_type(c, low_src);
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MidpriceBatchOutput, MidpriceError> {
        MidpriceBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "high", "low")
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
        _ => unreachable!(),
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
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
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
        return Err(MidpriceError::EmptyData);
    }
    let first = (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan())
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
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => midprice_row_scalar(high, low, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => midprice_row_avx2(high, low, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => midprice_row_avx512(high, low, first, period, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        values

                    .par_chunks_mut(cols)

                    .enumerate()

                    .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in values.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }

        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_midprice_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    fn check_midprice_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    fn check_midprice_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    fn check_midprice_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
        let mut stream = MidpriceStream::try_new(MidpriceParams { period: Some(period) })?;
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
        assert!(result.is_err(), "[{}] Expected error for all NaN values", test_name);
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
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
        check_midprice_all_nan
    );

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
}
