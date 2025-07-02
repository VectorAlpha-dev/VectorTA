//! # Williams' %R (WILLR)
//!
//! Williams' %R is a momentum oscillator that measures overbought/oversold levels.
//! This implementation supports both scalar and AVX2/AVX512 batch processing with API/features/tests
//! matching the `alma.rs` template, including parameter builders, batch processing, error handling,
//! and AVX stubs.
//!
//! ## Formula
//! \[ \text{%R} = \frac{\text{Highest\_High} - \text{Close}}{\text{Highest\_High} - \text{Lowest\_Low}} \times (-100) \]
//!
//! ## Parameters
//! - **period**: Window size (number of bars, default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: willr: All input values are NaN.
//! - **InvalidPeriod**: willr: period is zero or exceeds data length.
//! - **NotEnoughValidData**: willr: Not enough valid data points for the requested period.
//!
//! ## Returns
//! - **`Ok(WillrOutput)`**: Vec<f64> output, NaN padded.
//! - **`Err(WillrError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// --- Data types ---

#[derive(Debug, Clone)]
pub enum WillrData<'a> {
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
pub struct WillrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WillrParams {
    pub period: Option<usize>,
}

impl Default for WillrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct WillrInput<'a> {
    pub data: WillrData<'a>,
    pub params: WillrParams,
}

impl<'a> WillrInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: WillrParams) -> Self {
        Self {
            data: WillrData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: WillrParams,
    ) -> Self {
        Self {
            data: WillrData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, WillrParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WillrBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for WillrBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WillrBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply(self, c: &Candles) -> Result<WillrOutput, WillrError> {
        let p = WillrParams { period: self.period };
        let i = WillrInput::from_candles(c, p);
        willr_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WillrOutput, WillrError> {
        let p = WillrParams { period: self.period };
        let i = WillrInput::from_slices(high, low, close, p);
        willr_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum WillrError {
    #[error("willr: All values are NaN.")]
    AllValuesNaN,
    #[error("willr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("willr: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("willr: Data slices are empty or mismatched.")]
    EmptyOrMismatched,
}

// --- Main entrypoints ---

#[inline]
pub fn willr(input: &WillrInput) -> Result<WillrOutput, WillrError> {
    willr_with_kernel(input, Kernel::Auto)
}

pub fn willr_with_kernel(input: &WillrInput, kernel: Kernel) -> Result<WillrOutput, WillrError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        WillrData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        WillrData::Slices { high, low, close } => (high, low, close),
    };

    let len = high.len();
    if low.len() != len || close.len() != len || len == 0 {
        return Err(WillrError::EmptyOrMismatched);
    }
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(WillrError::InvalidPeriod { period, data_len: len });
    }

    let first_valid = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(WillrError::AllValuesNaN)?;

    if (len - first_valid) < period {
        return Err(WillrError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                willr_scalar(high, low, close, period, first_valid, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                willr_avx2(high, low, close, period, first_valid, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                willr_avx512(high, low, close, period, first_valid, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(WillrOutput { values: out })
}

// --- Scalar/AVX kernels ---

#[inline]
pub unsafe fn willr_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    for i in (first_valid + period - 1)..high.len() {
        let start = i + 1 - period;
        let (mut h, mut l) = (f64::NEG_INFINITY, f64::INFINITY);
        let mut has_nan = false;
        for j in start..=i {
            if high[j].is_nan() || low[j].is_nan() || close[i].is_nan() {
                has_nan = true;
                break;
            }
            if high[j] > h {
                h = high[j];
            }
            if low[j] < l {
                l = low[j];
            }
        }
        if has_nan || h.is_infinite() || l.is_infinite() {
            out[i] = f64::NAN;
        
            } else {
            let denom = h - l;
            if denom == 0.0 {
                out[i] = 0.0;
            
                } else {
                out[i] = (h - close[i]) / denom * -100.0;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        willr_avx512_short(high, low, close, period, first_valid, out);
    
        } else {
        willr_avx512_long(high, low, close, period, first_valid, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

// --- Batch grid/params/output for batch evaluation ---

#[derive(Clone, Debug)]
pub struct WillrBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for WillrBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WillrBatchBuilder {
    range: WillrBatchRange,
    kernel: Kernel,
}

impl WillrBatchBuilder {
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
        close: &[f64],
    ) -> Result<WillrBatchOutput, WillrError> {
        willr_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<WillrBatchOutput, WillrError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

pub fn willr_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    k: Kernel,
) -> Result<WillrBatchOutput, WillrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    willr_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct WillrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WillrParams>,
    pub rows: usize,
    pub cols: usize,
}

impl WillrBatchOutput {
    pub fn row_for_params(&self, p: &WillrParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &WillrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &WillrBatchRange) -> Vec<WillrParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods.into_iter().map(|p| WillrParams { period: Some(p) }).collect()
}

#[inline(always)]
pub fn willr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    kern: Kernel,
) -> Result<WillrBatchOutput, WillrError> {
    willr_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn willr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    kern: Kernel,
) -> Result<WillrBatchOutput, WillrError> {
    willr_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn willr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &WillrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WillrBatchOutput, WillrError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(WillrError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let len = high.len();
    if low.len() != len || close.len() != len || len == 0 {
        return Err(WillrError::EmptyOrMismatched);
    }

    let first_valid = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(WillrError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first_valid < max_p {
        return Err(WillrError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid,
        });
    }
    let rows = combos.len();
    let cols = len;
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => willr_row_scalar(high, low, close, first_valid, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => willr_row_avx2(high, low, close, first_valid, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => willr_row_avx512(high, low, close, first_valid, period, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        values.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in values.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }

    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(WillrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn willr_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        willr_row_avx512_short(high, low, close, first_valid, period, out);
    
        } else {
        willr_row_avx512_long(high, low, close, first_valid, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn willr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    period: usize,
    out: &mut [f64],
) {
    willr_scalar(high, low, close, period, first_valid, out)
}

#[inline(always)]
fn expand_grid_willr(r: &WillrBatchRange) -> Vec<WillrParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_willr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = WillrParams { period: None };
        let input = WillrInput::from_candles(&candles, params);
        let output = willr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_willr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = WillrParams { period: Some(14) };
        let input = WillrInput::from_candles(&candles, params);
        let output = willr_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -58.72876391329818,
            -61.77504393673111,
            -65.93438781487991,
            -60.27950310559006,
            -65.00449236298293,
        ];
        let start = output.values.len() - 5;
        for (i, &val) in output.values[start..].iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-8,
                "[{}] WILLR {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_willr_with_slice_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 2.0, 3.0, 4.0];
        let low = [0.5, 1.5, 2.5, 3.5];
        let close = [0.75, 1.75, 2.75, 3.75];
        let params = WillrParams { period: Some(2) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let output = willr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), 4);
        Ok(())
    }

    fn check_willr_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 2.0];
        let low = [0.8, 1.8];
        let close = [1.0, 2.0];
        let params = WillrParams { period: Some(0) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] WILLR should fail with zero period", test_name);
        Ok(())
    }

    fn check_willr_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 2.0, 3.0];
        let low = [0.5, 1.5, 2.5];
        let close = [1.0, 2.0, 3.0];
        let params = WillrParams { period: Some(10) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] WILLR should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_willr_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let params = WillrParams::default();
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] WILLR should fail with all NaN", test_name);
        Ok(())
    }

    fn check_willr_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 2.0];
        let low = [f64::NAN, 1.0];
        let close = [f64::NAN, 1.5];
        let params = WillrParams { period: Some(3) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let res = willr_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] WILLR should fail with not enough valid data", test_name);
        Ok(())
    }

    macro_rules! generate_all_willr_tests {
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

    generate_all_willr_tests!(
        check_willr_partial_params,
        check_willr_accuracy,
        check_willr_with_slice_data,
        check_willr_zero_period,
        check_willr_period_exceeds_length,
        check_willr_all_nan,
        check_willr_not_enough_valid_data
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = WillrBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let def = WillrParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -58.72876391329818,
            -61.77504393673111,
            -65.93438781487991,
            -60.27950310559006,
            -65.00449236298293,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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
