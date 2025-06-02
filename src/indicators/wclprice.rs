//! # Weighted Close Price (WCLPRICE)
//!
//! Computes `(high + low + 2*close) / 4` for each index. NaN if any input field is NaN at index.
//!
//! ## Parameters
//! - None (uses all of high, low, close)
//!
//! ## Errors
//! - **EmptyData**: Input is empty
//! - **AllValuesNaN**: All values are NaN in any required field
//!
//! ## Returns
//! - **Ok(WclpriceOutput)** on success, contains a `Vec<f64>`
//! - **Err(WclpriceError)** otherwise

use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum WclpriceData<'a> {
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
pub struct WclpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WclpriceParams;

impl Default for WclpriceParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct WclpriceInput<'a> {
    pub data: WclpriceData<'a>,
    pub params: WclpriceParams,
}

impl<'a> WclpriceInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: WclpriceData::Candles { candles },
            params: WclpriceParams::default(),
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: WclpriceData::Slices { high, low, close },
            params: WclpriceParams::default(),
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WclpriceBuilder {
    kernel: Kernel,
}
impl Default for WclpriceBuilder {
    fn default() -> Self {
        Self { kernel: Kernel::Auto }
    }
}
impl WclpriceBuilder {
    #[inline]
    pub fn new() -> Self { Self::default() }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<WclpriceOutput, WclpriceError> {
        let i = WclpriceInput::from_candles(candles);
        wclprice_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<WclpriceOutput, WclpriceError> {
        let i = WclpriceInput::from_slices(high, low, close);
        wclprice_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn into_stream(self) -> WclpriceStream {
        WclpriceStream::default()
    }
}

#[derive(Debug, Error)]
pub enum WclpriceError {
    #[error("wclprice: Empty data provided.")]
    EmptyData,
    #[error("wclprice: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn wclprice(input: &WclpriceInput) -> Result<WclpriceOutput, WclpriceError> {
    wclprice_with_kernel(input, Kernel::Auto)
}

pub fn wclprice_with_kernel(input: &WclpriceInput, kernel: Kernel) -> Result<WclpriceOutput, WclpriceError> {
    let (high, low, close) = match &input.data {
        WclpriceData::Candles { candles } => {
            let high = candles.select_candle_field("high").unwrap();
            let low = candles.select_candle_field("low").unwrap();
            let close = candles.select_candle_field("close").unwrap();
            (high, low, close)
        }
        WclpriceData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyData);
    }
    let len = high.len().min(low.len()).min(close.len());
    if len == 0 {
        return Err(WclpriceError::EmptyData);
    }
    let first = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;

    let mut out = vec![f64::NAN; len];
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                wclprice_scalar(high, low, close, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                wclprice_avx2(high, low, close, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wclprice_avx512(high, low, close, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(WclpriceOutput { values: out })
}

#[inline]
pub fn wclprice_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    let len = high.len().min(low.len()).min(close.len());
    for i in first_valid..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if h.is_nan() || l.is_nan() || c.is_nan() {
            out[i] = f64::NAN;
        } else {
            out[i] = (h + l + 2.0 * c) / 4.0;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { wclprice_avx512_short(high, low, close, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[inline]
pub fn wclprice_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx2(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx512(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx512_short(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx512_long(high, low, close, first_valid, out)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchRange; // No parameters

impl Default for WclpriceBatchRange {
    fn default() -> Self { Self }
}

#[derive(Clone, Debug, Default)]
pub struct WclpriceBatchBuilder {
    kernel: Kernel,
}
impl WclpriceBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<WclpriceBatchOutput, WclpriceError> {
        wclprice_batch_with_kernel(high, low, close, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
        let high = c.select_candle_field("high").unwrap();
        let low = c.select_candle_field("low").unwrap();
        let close = c.select_candle_field("close").unwrap();
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
        WclpriceBatchBuilder::new().apply_candles(c)
    }
}

pub fn wclprice_batch_with_kernel(
    high: &[f64], low: &[f64], close: &[f64], k: Kernel
) -> Result<WclpriceBatchOutput, WclpriceError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    wclprice_batch_par_slice(high, low, close, kernel)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
impl WclpriceBatchOutput {
    pub fn values_for(&self, _params: &WclpriceParams) -> Option<&[f64]> {
        if self.rows == 1 { Some(&self.values[..self.cols]) } else { None }
    }
}

#[inline(always)]
pub fn wclprice_batch_slice(
    high: &[f64], low: &[f64], close: &[f64], kern: Kernel
) -> Result<WclpriceBatchOutput, WclpriceError> {
    wclprice_batch_inner(high, low, close, kern, false)
}
#[inline(always)]
pub fn wclprice_batch_par_slice(
    high: &[f64], low: &[f64], close: &[f64], kern: Kernel
) -> Result<WclpriceBatchOutput, WclpriceError> {
    wclprice_batch_inner(high, low, close, kern, true)
}
#[inline(always)]
fn wclprice_batch_inner(
    high: &[f64], low: &[f64], close: &[f64], kern: Kernel, _parallel: bool
) -> Result<WclpriceBatchOutput, WclpriceError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyData);
    }
    let len = high.len().min(low.len()).min(close.len());
    let mut values = vec![f64::NAN; len];
    let first = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;
    unsafe {
        match kern {
            Kernel::ScalarBatch | Kernel::Scalar => wclprice_row_scalar(high, low, close, first, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => wclprice_row_avx2(high, low, close, first, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => wclprice_row_avx512(high, low, close, first, &mut values),
            _ => unreachable!(),
        }
    }
    Ok(WclpriceBatchOutput { values, rows: 1, cols: len })
}

#[inline(always)]
fn expand_grid(_r: &WclpriceBatchRange) -> Vec<WclpriceParams> {
    vec![WclpriceParams]
}

#[derive(Debug, Clone)]
pub struct WclpriceStream;
impl Default for WclpriceStream {
    fn default() -> Self { Self }
}
impl WclpriceStream {
    #[inline(always)]
    pub fn update(&mut self, h: f64, l: f64, c: f64) -> Option<f64> {
        if h.is_nan() || l.is_nan() || c.is_nan() { None } else { Some((h + l + 2.0 * c) / 4.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_wclprice_slices(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![59230.0, 59220.0, 59077.0, 59160.0, 58717.0];
        let low = vec![59222.0, 59211.0, 59077.0, 59143.0, 58708.0];
        let close = vec![59225.0, 59210.0, 59080.0, 59150.0, 58710.0];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice_with_kernel(&input, kernel)?;
        let expected = vec![59225.5, 59212.75, 59078.5, 59150.75, 58711.25];
        for (i, &v) in output.values.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-2, "[{test}] mismatch at {i}: {v} vs {expected:?}");
        }
        Ok(())
    }
    fn check_wclprice_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let input = WclpriceInput::from_candles(&candles);
        let output = wclprice_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_wclprice_empty_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let res = wclprice_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with empty data", test);
        Ok(())
    }
    fn check_wclprice_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![f64::NAN, f64::NAN];
        let low = vec![f64::NAN, f64::NAN];
        let close = vec![f64::NAN, f64::NAN];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let res = wclprice_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with all NaN", test);
        Ok(())
    }
    fn check_wclprice_partial_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![f64::NAN, 59000.0];
        let low = vec![f64::NAN, 58950.0];
        let close = vec![f64::NAN, 58975.0];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice_with_kernel(&input, kernel)?;
        assert!(output.values[0].is_nan());
        assert!((output.values[1] - (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0).abs() < 1e-8);
        Ok(())
    }
    macro_rules! generate_all_wclprice_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                   #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }
    generate_all_wclprice_tests!(
        check_wclprice_slices,
        check_wclprice_candles,
        check_wclprice_empty_data,
        check_wclprice_all_nan,
        check_wclprice_partial_nan
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = WclpriceBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let row = output.values_for(&WclpriceParams).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        }
    }
    gen_batch_tests!(check_batch_default_row);
}
