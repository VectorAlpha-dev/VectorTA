//! # Hull Moving Average (HMA)
//!
//! The Hull Moving Average (HMA) is a moving average technique that aims to
//! minimize lag while providing smooth output. It combines Weighted Moving
//! Averages of different lengths—namely `period/2` and `period`—to form an
//! intermediate difference. A final Weighted Moving Average is then applied
//! using the integer part of `sqrt(period)`, yielding a responsive trend
//! indication with reduced lag.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). (defaults to 5)
//!
//! ## Errors
//! - **NoData**: hma: No data provided.
//! - **AllValuesNaN**: hma: All input data values are `NaN`.
//! - **InvalidPeriod**: hma: `period` is zero or exceeds the data length.
//! - **ZeroHalf**: hma: Cannot calculate half of period.
//! - **ZeroSqrtPeriod**: hma: Cannot calculate sqrt of period.
//!
//! ## Returns
//! - **`Ok(HmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(HmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for HmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HmaData::Slice(slice) => slice,
            HmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HmaParams {
    pub period: Option<usize>,
}

impl Default for HmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct HmaInput<'a> {
    pub data: HmaData<'a>,
    pub params: HmaParams,
}

impl<'a> HmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HmaParams) -> Self {
        Self {
            data: HmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HmaParams) -> Self {
        Self {
            data: HmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for HmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<HmaOutput, HmaError> {
        let p = HmaParams { period: self.period };
        let i = HmaInput::from_candles(c, "close", p);
        hma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HmaOutput, HmaError> {
        let p = HmaParams { period: self.period };
        let i = HmaInput::from_slice(d, p);
        hma_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum HmaError {
    #[error("hma: No data provided.")]
    NoData,

    #[error("hma: All values are NaN.")]
    AllValuesNaN,

    #[error("hma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("hma: Cannot calculate half of period: period = {period}")]
    ZeroHalf { period: usize },

    #[error("hma: Cannot calculate sqrt of period: period = {period}")]
    ZeroSqrtPeriod { period: usize },
}

#[inline]
pub fn hma(input: &HmaInput) -> Result<HmaOutput, HmaError> {
    hma_with_kernel(input, Kernel::Auto)
}

pub fn hma_with_kernel(input: &HmaInput, kernel: Kernel) -> Result<HmaOutput, HmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HmaError::NoData);
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(HmaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(HmaError::InvalidPeriod { period, data_len: len });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                hma_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                hma_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                hma_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(HmaOutput { values: out })
}

#[inline]
pub fn hma_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // --- main HMA scalar logic (unchanged from your code) ---
    let len = data.len();
    let half = period / 2;
    if half == 0 || period > len {
        return;
    }
    let sqrtp = (period as f64).sqrt().floor() as usize;
    if sqrtp == 0 {
        return;
    }
    let m = len - first;
    if period > m {
        return;
    }

    let sum_w_half = (half * (half + 1)) >> 1;
    let denom_half = sum_w_half as f64;

    let sum_w_full = (period * (period + 1)) >> 1;
    let denom_full = sum_w_full as f64;

    let sum_w_sqrt = (sqrtp * (sqrtp + 1)) >> 1;
    let denom_sqrt = sum_w_sqrt as f64;

    let lookback_half = half - 1;
    let lookback_full = period - 1;

    let half_f = half as f64;
    let period_f = period as f64;
    let sqrtp_f = sqrtp as f64;

    let mut wma_half = vec![f64::NAN; len];
    let mut wma_full = vec![f64::NAN; len];

    let mut period_sub_half = 0.0;
    let mut period_sum_half = 0.0;
    let mut in_idx = 0;
    let mut i_half = 1;

    while in_idx < lookback_half {
        let val = data[first + in_idx];
        period_sub_half += val;
        period_sum_half += val * (i_half as f64);
        in_idx += 1;
        i_half += 1;
    }

    let mut period_sub_full = 0.0;
    let mut period_sum_full = 0.0;
    let mut in_idx_full = 0;
    let mut i_full = 1;

    while in_idx_full < lookback_full {
        let val = data[first + in_idx_full];
        period_sub_full += val;
        period_sum_full += val * (i_full as f64);
        in_idx_full += 1;
        i_full += 1;
    }

    if in_idx < m {
        let val = data[first + in_idx];
        in_idx += 1;
        period_sub_half += val;
        period_sum_half += val * half_f;

        wma_half[first + lookback_half] = period_sum_half / denom_half;
        period_sum_half -= period_sub_half;

        let mut trailing_idx_half = 1;
        let mut trailing_value_half = data[first];

        if in_idx_full < m {
            let valf = data[first + in_idx_full];
            in_idx_full += 1;
            period_sub_full += valf;
            period_sum_full += valf * period_f;

            wma_full[first + lookback_full] = period_sum_full / denom_full;
            period_sum_full -= period_sub_full;

            let mut trailing_idx_full = 1;
            let mut trailing_value_full = data[first];

            while in_idx < m || in_idx_full < m {
                if in_idx < m {
                    let new_val = data[first + in_idx];
                    in_idx += 1;

                    period_sub_half += new_val;
                    period_sub_half -= trailing_value_half;
                    period_sum_half += new_val * half_f;

                    trailing_value_half = data[first + trailing_idx_half];
                    trailing_idx_half += 1;

                    wma_half[first + (in_idx - 1)] = period_sum_half / denom_half;
                    period_sum_half -= period_sub_half;
                }

                if in_idx_full < m {
                    let new_valf = data[first + in_idx_full];
                    in_idx_full += 1;

                    period_sub_full += new_valf;
                    period_sub_full -= trailing_value_full;
                    period_sum_full += new_valf * period_f;

                    trailing_value_full = data[first + trailing_idx_full];
                    trailing_idx_full += 1;

                    wma_full[first + (in_idx_full - 1)] = period_sum_full / denom_full;
                    period_sum_full -= period_sub_full;
                }
            }
        }
    }

    let mut diff = vec![f64::NAN; len];
    for i in 0..len {
        let a = wma_half[i];
        let b = wma_full[i];
        if a.is_finite() && b.is_finite() {
            diff[i] = 2.0 * a - b;
        }
    }

    let mut wma_sqrt = vec![f64::NAN; len];
    {
        let lookback_sqrt = sqrtp - 1;
        let mut period_sub_sqrt = 0.0;
        let mut period_sum_sqrt = 0.0;
        let mut in_idx_sqrt = 0;
        let mut i_s = 1;

        while in_idx_sqrt < lookback_sqrt {
            let val = diff[first + in_idx_sqrt];
            if val.is_finite() {
                period_sub_sqrt += val;
                period_sum_sqrt += val * (i_s as f64);
            }
            in_idx_sqrt += 1;
            i_s += 1;
        }

        if in_idx_sqrt < m {
            let val = diff[first + in_idx_sqrt];
            in_idx_sqrt += 1;
            if val.is_finite() {
                period_sub_sqrt += val;
                period_sum_sqrt += val * sqrtp_f;
            }
            let mut trailing_idx_sqrt = 1;
            let mut trailing_value_sqrt = diff[first];

            wma_sqrt[first + lookback_sqrt] = if trailing_value_sqrt.is_finite() {
                period_sum_sqrt / denom_sqrt
            } else {
                f64::NAN
            };
            period_sum_sqrt -= period_sub_sqrt;

            while in_idx_sqrt < m {
                let new_val = diff[first + in_idx_sqrt];
                in_idx_sqrt += 1;

                if new_val.is_finite() {
                    period_sub_sqrt += new_val;
                }
                if trailing_value_sqrt.is_finite() {
                    period_sub_sqrt -= trailing_value_sqrt;
                }
                if new_val.is_finite() {
                    period_sum_sqrt += new_val * sqrtp_f;
                }

                trailing_value_sqrt = diff[first + trailing_idx_sqrt];
                trailing_idx_sqrt += 1;

                wma_sqrt[first + (in_idx_sqrt - 1)] = if period_sub_sqrt != 0.0 {
                    period_sum_sqrt / denom_sqrt
                } else {
                    f64::NAN
                };
                period_sum_sqrt -= period_sub_sqrt;
            }
        }
    }
    out.copy_from_slice(&wma_sqrt);
}

// --- AVX2/AVX512 stubs (redirect to scalar) ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn hma_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    hma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn hma_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    hma_scalar(data, period, first, out)
}

// --- Batch API, batch builder, and helpers ---

#[derive(Clone, Debug)]
pub struct HmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for HmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 120, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HmaBatchBuilder {
    range: HmaBatchRange,
    kernel: Kernel,
}

impl HmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<HmaBatchOutput, HmaError> {
        hma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<HmaBatchOutput, HmaError> {
        HmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<HmaBatchOutput, HmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HmaBatchOutput, HmaError> {
        HmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn hma_batch_with_kernel(
    data: &[f64],
    sweep: &HmaBatchRange,
    k: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(HmaError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    hma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HmaBatchOutput {
    pub fn row_for_params(&self, p: &HmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &HmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HmaBatchRange) -> Vec<HmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(HmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn hma_batch_slice(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    hma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn hma_batch_par_slice(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    hma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn hma_batch_inner(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HmaBatchOutput, HmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HmaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(HmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HmaError::InvalidPeriod { period: max_p, data_len: data.len() });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => hma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => hma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => hma_row_avx512(data, first, period, out_row),
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
    Ok(HmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// --- row variants (all AVX point to scalar, as per your pattern) ---

#[inline(always)]
pub unsafe fn hma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    hma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    hma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        hma_row_avx512_short(data, first, period, out)
    } else {
        hma_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    hma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    hma_row_scalar(data, first, period, out)
}

#[inline(always)]
fn expand_grid_hma(r: &HmaBatchRange) -> Vec<HmaParams> {
    expand_grid(r)
}

// --- tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_hma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HmaParams { period: None };
        let input_default = HmaInput::from_candles(&candles, "close", default_params);
        let output_default = hma_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }

    fn check_hma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HmaInput::with_default_candles(&candles);
        let result = hma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        assert!(result.values.len() >= 5);
        assert_eq!(result.values.len(), candles.close.len());
        let start = result.values.len() - 5;
        let last_five = &result.values[start..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!((val - exp).abs() < 1e-3, "[{}] idx {}: got {}, expected {}", test_name, i, val, exp);
        }
        Ok(())
    }

    fn check_hma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(0) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] HMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_hma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(10) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] HMA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_hma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = HmaParams { period: Some(5) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] HMA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_hma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = HmaParams { period: Some(5) };
        let first_input = HmaInput::from_candles(&candles, "close", first_params);
        let first_result = hma_with_kernel(&first_input, kernel)?;
        let second_params = HmaParams { period: Some(3) };
        let second_input = HmaInput::from_slice(&first_result.values, second_params);
        let second_result = hma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_hma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = HmaParams::default();
        let input = HmaInput::from_candles(&candles, "close", params);
        let result = hma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_hma_tests {
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

    generate_all_hma_tests!(
        check_hma_partial_params,
        check_hma_accuracy,
        check_hma_zero_period,
        check_hma_period_exceeds_length,
        check_hma_very_small_dataset,
        check_hma_reinput,
        check_hma_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = HmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = HmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-3,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
