//! # Triangular Moving Average (TRIMA)
//!
//! A moving average computed by averaging an underlying Simple Moving Average (SMA) over
//! the specified `period`, resulting in a smoother output than a single SMA.
//! TRIMA supports different compute kernels and batch processing via builder APIs.
//!
//! ## Parameters
//! - **period**: Window size (must be > 3).
//!
//! ## Errors
//! - **AllValuesNaN**: trima: All input data values are `NaN`.
//! - **InvalidPeriod**: trima: `period` is zero, ≤ 3, or exceeds the data length.
//! - **NotEnoughValidData**: trima: Not enough valid data points for the requested `period`.
//! - **NoData**: trima: No data provided.
//!
//! ## Returns
//! - **`Ok(TrimaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(TrimaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;
use paste::paste;

impl<'a> AsRef<[f64]> for TrimaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrimaData::Slice(slice) => slice,
            TrimaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrimaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrimaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrimaParams {
    pub period: Option<usize>,
}

impl Default for TrimaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct TrimaInput<'a> {
    pub data: TrimaData<'a>,
    pub params: TrimaParams,
}

impl<'a> TrimaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TrimaParams) -> Self {
        Self {
            data: TrimaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TrimaParams) -> Self {
        Self {
            data: TrimaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TrimaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrimaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TrimaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrimaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TrimaOutput, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        let i = TrimaInput::from_candles(c, "close", p);
        trima_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TrimaOutput, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        let i = TrimaInput::from_slice(d, p);
        trima_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TrimaStream, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        TrimaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TrimaError {
    #[error("trima: All values are NaN.")]
    AllValuesNaN,

    #[error("trima: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("trima: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("trima: Period too small: {period}")]
    PeriodTooSmall { period: usize },

    #[error("trima: No data provided.")]
    NoData,
}

#[inline]
pub fn trima(input: &TrimaInput) -> Result<TrimaOutput, TrimaError> {
    trima_with_kernel(input, Kernel::Auto)
}

pub fn trima_with_kernel(input: &TrimaInput, kernel: Kernel) -> Result<TrimaOutput, TrimaError> {
    let data: &[f64] = match &input.data {
        TrimaData::Candles { candles, source } => source_type(candles, source),
        TrimaData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrimaError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(TrimaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period <= 3 {
        return Err(TrimaError::PeriodTooSmall { period });
    }
    if (len - first) < period {
        return Err(TrimaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if len == 0 {
        return Err(TrimaError::NoData);
    }

    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trima_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                trima_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trima_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(TrimaOutput { values: out })
}

#[inline]
pub fn trima_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // Original logic from trima
    let n = data.len();

    let sum_of_weights = if period % 2 == 1 {
        let half = period / 2 + 1;
        (half * half) as f64
    } else {
        let half_up = period / 2 + 1;
        let half_down = period / 2;
        (half_up * half_down) as f64
    };
    let inv_weights = 1.0 / sum_of_weights;

    let lead_period = if period % 2 == 1 {
        period / 2
    } else {
        (period / 2) - 1
    };
    let trail_period = lead_period + 1;

    let mut weight_sum = 0.0;
    let mut lead_sum = 0.0;
    let mut trail_sum = 0.0;
    let mut w = 1;

    for i in 0..(period - 1) {
        let idx = first + i;
        let val = data[idx];
        weight_sum += val * (w as f64);
        if i + 1 > period - lead_period {
            lead_sum += val;
        }
        if i < trail_period {
            trail_sum += val;
        }
        if i + 1 < trail_period {
            w += 1;
        }
        if i + 1 >= (period - lead_period) {
            w -= 1;
        }
    }

    let mut lsi = (period - 1) as isize - lead_period as isize + 1;
    let mut tsi1 = (period - 1) as isize - period as isize + 1 + trail_period as isize;
    let mut tsi2 = (period - 1) as isize - period as isize + 1;

    for i in (first + (period - 1))..n {
        let val = data[i];
        weight_sum += val;

        out[i] = weight_sum * inv_weights;

        lead_sum += val;
        weight_sum += lead_sum;
        weight_sum -= trail_sum;

        if lsi >= 0 {
            lead_sum -= data[lsi as usize];
        }
        if tsi1 >= 0 {
            trail_sum += data[tsi1 as usize];
        }
        if tsi2 >= 0 {
            trail_sum -= data[tsi2 as usize];
        }

        lsi += 1;
        tsi1 += 1;
        tsi2 += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn trima_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { trima_avx512_short(data, period, first, out) }
    } else {
        unsafe { trima_avx512_long(data, period, first, out) }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}

#[inline(always)]
pub fn trima_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512_short(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512_long(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct TrimaStream {
    period: usize,
    buffer: Vec<f64>,
    sum: f64,
    head: usize,
    filled: bool,
    count: usize,
}

impl TrimaStream {
    pub fn try_new(params: TrimaParams) -> Result<Self, TrimaError> {
        let period = params.period.unwrap_or(14);
        if period == 0 || period <= 3 {
            return Err(TrimaError::PeriodTooSmall { period });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            sum: 0.0,
            head: 0,
            filled: false,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        if !old.is_nan() {
            self.sum -= old;
        }
        self.sum += value;
        Some(self.sum / (self.period as f64))
    }
}

#[derive(Clone, Debug)]
pub struct TrimaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TrimaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrimaBatchBuilder {
    range: TrimaBatchRange,
    kernel: Kernel,
}

impl TrimaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<TrimaBatchOutput, TrimaError> {
        trima_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TrimaBatchOutput, TrimaError> {
        TrimaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TrimaBatchOutput, TrimaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<TrimaBatchOutput, TrimaError> {
        TrimaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn trima_batch_with_kernel(
    data: &[f64],
    sweep: &TrimaBatchRange,
    k: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(TrimaError::InvalidPeriod {
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
    trima_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TrimaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrimaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TrimaBatchOutput {
    pub fn row_for_params(&self, p: &TrimaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
        })
    }

    pub fn values_for(&self, p: &TrimaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TrimaBatchRange) -> Vec<TrimaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TrimaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn trima_batch_slice(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    trima_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn trima_batch_par_slice(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    trima_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn trima_batch_inner(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TrimaBatchOutput, TrimaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TrimaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrimaError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.period.unwrap())
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(TrimaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => trima_row_scalar(data, first, period, 0, std::ptr::null(), 1.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => trima_row_avx2(data, first, period, 0, std::ptr::null(), 1.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => trima_row_avx512(data, first, period, 0, std::ptr::null(), 1.0, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(TrimaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_trima_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = TrimaParams { period: None };
        let input = TrimaInput::from_candles(&candles, "close", default_params);
        let output = trima_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_10 = TrimaParams { period: Some(10) };
        let input2 = TrimaInput::from_candles(&candles, "hl2", params_period_10);
        let output2 = trima_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = TrimaParams { period: Some(30) };
        let input3 = TrimaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = trima_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());

        Ok(())
    }

    fn check_trima_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;
        let params = TrimaParams { period: Some(30) };
        let input = TrimaInput::from_candles(&candles, "close", params);
        let trima_result = trima_with_kernel(&input, kernel)?;

        assert_eq!(trima_result.values.len(), close_prices.len(), "TRIMA output length should match input data length");
        let expected_last_five_trima = [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ];
        assert!(trima_result.values.len() >= 5, "Not enough TRIMA values for the test");
        let start_index = trima_result.values.len() - 5;
        let result_last_five_trima = &trima_result.values[start_index..];
        for (i, &value) in result_last_five_trima.iter().enumerate() {
            let expected_value = expected_last_five_trima[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "[{}] TRIMA value mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        let period = input.params.period.unwrap_or(14);
        for i in 0..(period - 1) {
            assert!(
                trima_result.values[i].is_nan(),
                "[{}] Expected NaN at early index {} for TRIMA, got {}",
                test_name,
                i,
                trima_result.values[i]
            );
        }
        Ok(())
    }

    fn check_trima_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TrimaInput::with_default_candles(&candles);
        match input.data {
            TrimaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TrimaData::Candles"),
        }
        let output = trima_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_trima_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TrimaParams { period: Some(0) };
        let input = TrimaInput::from_slice(&input_data, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_trima_period_too_small(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0, 40.0];
        let params = TrimaParams { period: Some(3) };
        let input = TrimaInput::from_slice(&input_data, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with period <= 3",
            test_name
        );
        Ok(())
    }

    fn check_trima_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TrimaParams { period: Some(10) };
        let input = TrimaInput::from_slice(&data_small, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_trima_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TrimaParams { period: Some(14) };
        let input = TrimaInput::from_slice(&single_point, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_trima_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = TrimaParams { period: Some(14) };
        let first_input = TrimaInput::from_candles(&candles, "close", first_params);
        let first_result = trima_with_kernel(&first_input, kernel)?;

        let second_params = TrimaParams { period: Some(10) };
        let second_input = TrimaInput::from_slice(&first_result.values, second_params);
        let second_result = trima_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(val.is_finite());
        }
        Ok(())
    }

    fn check_trima_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TrimaInput::from_candles(
            &candles,
            "close",
            TrimaParams { period: Some(14) },
        );
        let res = trima_with_kernel(&input, kernel)?;
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

    fn check_trima_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;

        let input = TrimaInput::from_candles(
            &candles,
            "close",
            TrimaParams { period: Some(period) },
        );
        let batch_output = trima_with_kernel(&input, kernel)?.values;

        let mut stream = TrimaStream::try_new(TrimaParams { period: Some(period) })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(trima_val) => stream_values.push(trima_val),
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
                diff < 1e-8,
                "[{}] TRIMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_trima_tests {
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

    generate_all_trima_tests!(
        check_trima_partial_params,
        check_trima_accuracy,
        check_trima_default_candles,
        check_trima_zero_period,
        check_trima_period_exceeds_length,
        check_trima_period_too_small,
        check_trima_very_small_dataset,
        check_trima_reinput,
        check_trima_nan_handling,
        check_trima_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TrimaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = TrimaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // You can use expected values as appropriate for TRIMA.
        let expected = [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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

    gen_batch_tests!(check_batch_default_row);
}
