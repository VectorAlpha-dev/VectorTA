//! # Ultimate Oscillator (ULTOSC)
//!
//! The Ultimate Oscillator (ULTOSC) combines short, medium, and long time periods into a single oscillator value
//! (0-100), blending market momentum over multiple horizons. Three periods are summed with weights 4:2:1.
//!
//! ## Parameters
//! - **timeperiod1**: Short window (default = 7)
//! - **timeperiod2**: Medium window (default = 14)
//! - **timeperiod3**: Long window (default = 28)
//!
//! ## Errors
//! - **EmptyData**: All input slices are empty.
//! - **InvalidPeriods**: Any period is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data for the largest period.
//! - **AllValuesNaN**: All input values are `NaN`.
//!
//! ## Returns
//! - **`Ok(UltOscOutput)`**: Contains a `Vec<f64>` matching the input length.
//! - **`Err(UltOscError)`** otherwise.

use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// --- DATA STRUCTS ---
#[derive(Debug, Clone)]
pub enum UltOscData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct UltOscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct UltOscParams {
    pub timeperiod1: Option<usize>,
    pub timeperiod2: Option<usize>,
    pub timeperiod3: Option<usize>,
}

impl Default for UltOscParams {
    fn default() -> Self {
        Self {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UltOscInput<'a> {
    pub data: UltOscData<'a>,
    pub params: UltOscParams,
}

impl<'a> UltOscInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src,
                low_src,
                close_src,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src: "high",
                low_src: "low",
                close_src: "close",
            },
            params: UltOscParams::default(),
        }
    }
    #[inline]
    pub fn get_timeperiod1(&self) -> usize {
        self.params.timeperiod1.unwrap_or(7)
    }
    #[inline]
    pub fn get_timeperiod2(&self) -> usize {
        self.params.timeperiod2.unwrap_or(14)
    }
    #[inline]
    pub fn get_timeperiod3(&self) -> usize {
        self.params.timeperiod3.unwrap_or(28)
    }
}

// --- BUILDER ---
#[derive(Copy, Clone, Debug)]
pub struct UltOscBuilder {
    timeperiod1: Option<usize>,
    timeperiod2: Option<usize>,
    timeperiod3: Option<usize>,
    kernel: Kernel,
}

impl Default for UltOscBuilder {
    fn default() -> Self {
        Self {
            timeperiod1: None,
            timeperiod2: None,
            timeperiod3: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UltOscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn timeperiod1(mut self, p: usize) -> Self {
        self.timeperiod1 = Some(p);
        self
    }
    #[inline(always)]
    pub fn timeperiod2(mut self, p: usize) -> Self {
        self.timeperiod2 = Some(p);
        self
    }
    #[inline(always)]
    pub fn timeperiod3(mut self, p: usize) -> Self {
        self.timeperiod3 = Some(p);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<UltOscOutput, UltOscError> {
        let params = UltOscParams {
            timeperiod1: self.timeperiod1,
            timeperiod2: self.timeperiod2,
            timeperiod3: self.timeperiod3,
        };
        let input = UltOscInput::with_default_candles(candles);
        ultosc_with_kernel(&UltOscInput {
            params,
            ..input
        }, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<UltOscOutput, UltOscError> {
        let params = UltOscParams {
            timeperiod1: self.timeperiod1,
            timeperiod2: self.timeperiod2,
            timeperiod3: self.timeperiod3,
        };
        let input = UltOscInput::from_slices(high, low, close, params);
        ultosc_with_kernel(&input, self.kernel)
    }
}

// --- ERROR ---
#[derive(Debug, Error)]
pub enum UltOscError {
    #[error("ultosc: Empty data provided.")]
    EmptyData,
    #[error("ultosc: Invalid periods: p1 = {p1}, p2 = {p2}, p3 = {p3}, data length = {data_len}")]
    InvalidPeriods { p1: usize, p2: usize, p3: usize, data_len: usize },
    #[error("ultosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ultosc: All values are NaN (or their preceding data is NaN).")]
    AllValuesNaN,
}

// --- KERNEL ENTRYPOINTS ---
#[inline]
pub fn ultosc(input: &UltOscInput) -> Result<UltOscOutput, UltOscError> {
    ultosc_with_kernel(input, Kernel::Auto)
}

pub fn ultosc_with_kernel(input: &UltOscInput, kernel: Kernel) -> Result<UltOscOutput, UltOscError> {
    let (high, low, close) = match &input.data {
        UltOscData::Candles { candles, high_src, low_src, close_src } => {
            let high = candles.select_candle_field(high_src).unwrap();
            let low = candles.select_candle_field(low_src).unwrap();
            let close = candles.select_candle_field(close_src).unwrap();
            (high, low, close)
        }
        UltOscData::Slices { high, low, close } => (*high, *low, *close),
    };
    let len = high.len();
    if len == 0 || low.len() == 0 || close.len() == 0 {
        return Err(UltOscError::EmptyData);
    }
    let p1 = input.get_timeperiod1();
    let p2 = input.get_timeperiod2();
    let p3 = input.get_timeperiod3();
    if p1 == 0 || p2 == 0 || p3 == 0 || p1 > len || p2 > len || p3 > len {
        return Err(UltOscError::InvalidPeriods { p1, p2, p3, data_len: len });
    }
    let largest_period = p1.max(p2.max(p3));
    let first_valid = match (1..len).find(|&i| {
        !high[i-1].is_nan() && !low[i-1].is_nan() && !close[i-1].is_nan() &&
        !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan()
    }) {
        Some(i) => i,
        None => return Err(UltOscError::AllValuesNaN),
    };
    let start_idx = first_valid + (largest_period - 1);
    if start_idx >= len {
        return Err(UltOscError::NotEnoughValidData {
            needed: largest_period,
            valid: len.saturating_sub(first_valid),
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ultosc_scalar(high, low, close, p1, p2, p3, first_valid, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ultosc_avx2(high, low, close, p1, p2, p3, first_valid, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ultosc_avx512(high, low, close, p1, p2, p3, first_valid, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(UltOscOutput { values: out })
}

// --- KERNEL IMPL ---
#[inline(always)]
pub unsafe fn ultosc_scalar(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    let len = high.len();
    let mut cmtl = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
    let mut tr   = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
    cmtl.resize(len, f64::NAN);
    tr.resize(len, f64::NAN);
    for i in 1..len {
        if high[i].is_nan() || low[i].is_nan() || close[i].is_nan() || close[i - 1].is_nan() {
            continue;
        }
        let true_low = low[i].min(close[i - 1]);
        let mut true_range = high[i] - low[i];
        let diff1 = (high[i] - close[i - 1]).abs();
        if diff1 > true_range {
            true_range = diff1;
        }
        let diff2 = (low[i] - close[i - 1]).abs();
        if diff2 > true_range {
            true_range = diff2;
        }
        cmtl[i] = close[i] - true_low;
        tr[i] = true_range;
    }
    let mut sum1_a = 0.0;
    let mut sum1_b = 0.0;
    let mut sum2_a = 0.0;
    let mut sum2_b = 0.0;
    let mut sum3_a = 0.0;
    let mut sum3_b = 0.0;
    let prime_range_1 = (first_valid + p1 - 1).saturating_sub(p1 - 1)..first_valid + p1 - 1;
    let prime_range_2 = (first_valid + p2 - 1).saturating_sub(p2 - 1)..first_valid + p2 - 1;
    let prime_range_3 = (first_valid + p3 - 1).saturating_sub(p3 - 1)..first_valid + p3 - 1;
    for i in prime_range_1 {
        if i < len && !cmtl[i].is_nan() && !tr[i].is_nan() {
            sum1_a += cmtl[i];
            sum1_b += tr[i];
        }
    }
    for i in prime_range_2 {
        if i < len && !cmtl[i].is_nan() && !tr[i].is_nan() {
            sum2_a += cmtl[i];
            sum2_b += tr[i];
        }
    }
    for i in prime_range_3 {
        if i < len && !cmtl[i].is_nan() && !tr[i].is_nan() {
            sum3_a += cmtl[i];
            sum3_b += tr[i];
        }
    }
    let mut today = first_valid + p1.max(p2).max(p3) - 1;
    while today < len {
        if !cmtl[today].is_nan() && !tr[today].is_nan() {
            sum1_a += cmtl[today];
            sum1_b += tr[today];
            sum2_a += cmtl[today];
            sum2_b += tr[today];
            sum3_a += cmtl[today];
            sum3_b += tr[today];
        }
        let v1 = if sum1_b != 0.0 { 4.0 * (sum1_a / sum1_b) } else { 0.0 };
        let v2 = if sum2_b != 0.0 { 2.0 * (sum2_a / sum2_b) } else { 0.0 };
        let v3 = if sum3_b != 0.0 { sum3_a / sum3_b } else { 0.0 };
        out[today] = 100.0 * (v1 + v2 + v3) / 7.0;
        let trailing_1 = today as isize - (p1 as isize) + 1;
        if trailing_1 >= 0 && (trailing_1 as usize) < len {
            let idx = trailing_1 as usize;
            if !cmtl[idx].is_nan() { sum1_a -= cmtl[idx]; }
            if !tr[idx].is_nan() { sum1_b -= tr[idx]; }
        }
        let trailing_2 = today as isize - (p2 as isize) + 1;
        if trailing_2 >= 0 && (trailing_2 as usize) < len {
            let idx = trailing_2 as usize;
            if !cmtl[idx].is_nan() { sum2_a -= cmtl[idx]; }
            if !tr[idx].is_nan() { sum2_b -= tr[idx]; }
        }
        let trailing_3 = today as isize - (p3 as isize) + 1;
        if trailing_3 >= 0 && (trailing_3 as usize) < len {
            let idx = trailing_3 as usize;
            if !cmtl[idx].is_nan() { sum3_a -= cmtl[idx]; }
            if !tr[idx].is_nan() { sum3_b -= tr[idx]; }
        }
        today += 1;
    }
}

// --- AVX STUBS ---
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx2(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    if p1.max(p2).max(p3) <= 32 {
        ultosc_avx512_short(high, low, close, p1, p2, p3, first_valid, out)
    } else {
        ultosc_avx512_long(high, low, close, p1, p2, p3, first_valid, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512_short(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512_long(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}

// --- ROW/BATCH/BATCHBUILDER (no sweep for ultosc, but stubs for parity) ---
#[inline(always)]
pub fn ultosc_row_scalar(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx2(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx2(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512_short(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64],
    p1: usize, p2: usize, p3: usize, first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512_long(high, low, close, p1, p2, p3, first_valid, out) }
}

// --- Batch APIs (No grid sweep for ultosc, but maintain struct/trait parity) ---
#[derive(Clone, Debug, Default)]
pub struct UltOscBatchRange;
#[derive(Clone, Debug, Default)]
pub struct UltOscBatchBuilder;
#[derive(Clone, Debug)]
pub struct UltOscBatchOutput {
    pub values: Vec<f64>,
}

impl UltOscBatchBuilder {
    pub fn new() -> Self { Self }
    pub fn apply_slice(
        self,
        high: &[f64], low: &[f64], close: &[f64], params: UltOscParams, kernel: Kernel
    ) -> Result<UltOscBatchOutput, UltOscError> {
        let input = UltOscInput::from_slices(high, low, close, params);
        let res = ultosc_with_kernel(&input, kernel)?;
        Ok(UltOscBatchOutput { values: res.values })
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_ultosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UltOscParams { timeperiod1: None, timeperiod2: None, timeperiod3: None };
        let input = UltOscInput::from_candles(&candles, "high", "low", "close", params);
        let output = ultosc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ultosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UltOscParams { timeperiod1: Some(7), timeperiod2: Some(14), timeperiod3: Some(28) };
        let input = UltOscInput::from_candles(&candles, "high", "low", "close", params);
        let result = ultosc_with_kernel(&input, kernel)?;
        let expected_last_five = [
            41.25546890298435,
            40.83865967175865,
            48.910324164909625,
            45.43113094857947,
            42.163165136766295,
        ];
        assert!(result.values.len() >= 5);
        let start_idx = result.values.len() - 5;
        for (i, &val) in result.values[start_idx..].iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "[{}] ULTOSC mismatch at last five index {}: expected {}, got {}",
                test_name,
                i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_ultosc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UltOscInput::with_default_candles(&candles);
        let result = ultosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ultosc_zero_periods(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams { timeperiod1: Some(0), timeperiod2: Some(14), timeperiod3: Some(28) };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Expected error for zero period", test_name);
        Ok(())
    }

    fn check_ultosc_period_exceeds_data_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams { timeperiod1: Some(7), timeperiod2: Some(14), timeperiod3: Some(28) };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Expected error for period exceeding data length", test_name);
        Ok(())
    }

    macro_rules! generate_all_ultosc_tests {
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

    generate_all_ultosc_tests!(
        check_ultosc_partial_params,
        check_ultosc_accuracy,
        check_ultosc_default_candles,
        check_ultosc_zero_periods,
        check_ultosc_period_exceeds_data_length
    );
        fn check_ultosc_batch_default(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = UltOscParams::default();
        let batch_builder = UltOscBatchBuilder::new();

        let output = batch_builder.apply_slice(
            &candles.high,
            &candles.low,
            &candles.close,
            params,
            kernel,
        )?;

        // Output length must match input length
        assert_eq!(output.values.len(), candles.close.len());

        // Spot check last 5 values against expected (if you have reference values)
        // Here I use the same expected values as in the accuracy test for simplicity
        let expected_last_five = [
            41.25546890298435,
            40.83865967175865,
            48.910324164909625,
            45.43113094857947,
            42.163165136766295,
        ];

        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] Batch ULTOSC mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
            }
        };
    }

    gen_batch_tests!(check_ultosc_batch_default);
}
