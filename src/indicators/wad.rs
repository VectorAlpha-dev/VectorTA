//! # Williams Accumulation/Distribution (WAD)
//!
//! Williams Accumulation/Distribution (WAD) is a cumulative measure of buying and selling pressure
//! based on the relationship between the current close, previous close, and high and low price ranges.
//! This implementation is API and feature-parity with alma.rs, including AVX stub functions,
//! kernel selection, batch parameter sweep support, input validation, builder and stream APIs,
//! and extensive test coverage.
//!
//! ## Parameters
//! - None (WAD does not use a period).
//!
//! ## Errors
//! - **EmptyData**: wad: Input data slice is empty.
//! - **AllValuesNaN**: wad: All input data values for high, low, or close are `NaN`.
//!
//! ## Returns
//! - **`Ok(WadOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(WadError)`** otherwise.

use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum WadData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct WadOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WadParams;

#[derive(Debug, Clone)]
pub struct WadInput<'a> {
    pub data: WadData<'a>,
    pub params: WadParams,
}

impl<'a> WadInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: WadData::Candles { candles },
            params: WadParams::default(),
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: WadData::Slices { high, low, close },
            params: WadParams::default(),
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct WadBuilder {
    kernel: Kernel,
}
impl WadBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<WadOutput, WadError> {
        let i = WadInput::from_candles(candles);
        wad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<WadOutput, WadError> {
        let i = WadInput::from_slices(high, low, close);
        wad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<WadStream, WadError> {
        WadStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum WadError {
    #[error("wad: Empty data provided.")]
    EmptyData,
    #[error("wad: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn wad(input: &WadInput) -> Result<WadOutput, WadError> {
    wad_with_kernel(input, Kernel::Auto)
}

pub fn wad_with_kernel(input: &WadInput, kernel: Kernel) -> Result<WadOutput, WadError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        WadData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        WadData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WadError::EmptyData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(WadError::EmptyData);
    }
    if high.iter().all(|x| x.is_nan()) || low.iter().all(|x| x.is_nan()) || close.iter().all(|x| x.is_nan()) {
        return Err(WadError::AllValuesNaN);
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut out = vec![0.0; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wad_scalar(high, low, close, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wad_avx2(high, low, close, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wad_avx512(high, low, close, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(WadOutput { values: out })
}

#[inline(always)]
pub fn wad_scalar(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    let n = close.len();
    if n == 0 {
        return;
    }


    out[0] = 0.0;
    let mut running_sum = 0.0;
    let mut prev_close = close[0];

    for i in 1..n {
        let trh = if prev_close > high[i] { prev_close } else { high[i] };
        let trl = if prev_close < low[i]  { prev_close } else { low[i]  };
        let ad = if close[i] > prev_close {
            close[i] - trl
        } else if close[i] < prev_close {
            close[i] - trh
        } else {
            0.0
        };

        running_sum += ad;
        out[i] = running_sum;
        prev_close = close[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx2(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    // stub, API parity
    wad_scalar(high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx512(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    // stub, API parity, route by len
    if high.len() <= 32 {
        wad_avx512_short(high, low, close, out);
    } else {
        wad_avx512_long(high, low, close, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx512_short(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    wad_scalar(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx512_long(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    wad_scalar(high, low, close, out)
}

// Per-row APIs for batch processing
#[inline(always)]
pub unsafe fn wad_row_scalar(
    high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]
) {
    wad_scalar(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx2(
    high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]
) {
    wad_scalar(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx512(
    high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]
) {
    wad_scalar(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]
) {
    wad_scalar(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]
) {
    wad_scalar(high, low, close, out)
}

#[derive(Debug, Clone)]
pub struct WadStream {
    sum: f64,
    prev_close: Option<f64>,
}
impl WadStream {
    pub fn try_new() -> Result<Self, WadError> {
        Ok(Self { sum: 0.0, prev_close: None })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let ad = if let Some(pc) = self.prev_close {
            if close > pc {
                close - low.min(pc)
            } else if close < pc {
                close - high.max(pc)
            } else { 0.0 }
        } else { 0.0 };
        self.prev_close = Some(close);
        self.sum += ad;
        self.sum
    }
}

#[derive(Clone, Debug)]
pub struct WadBatchRange {
    pub dummy: (usize, usize, usize),
}
impl Default for WadBatchRange {
    fn default() -> Self {
        Self { dummy: (0, 0, 0) }
    }
}
#[derive(Clone, Debug, Default)]
pub struct WadBatchBuilder {
    range: WadBatchRange,
    kernel: Kernel,
}
impl WadBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<WadBatchOutput, WadError> {
        wad_batch_with_kernel(high, low, close, self.kernel)
    }
    pub fn with_default_slices(high: &[f64], low: &[f64], close: &[f64], k: Kernel) -> Result<WadBatchOutput, WadError> {
        WadBatchBuilder::new().kernel(k).apply_slices(high, low, close)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<WadBatchOutput, WadError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WadBatchOutput, WadError> {
        WadBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

pub fn wad_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k: Kernel,
) -> Result<WadBatchOutput, WadError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    wad_batch_par_slice(high, low, close, kernel)
}

#[derive(Clone, Debug)]
pub struct WadBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
impl WadBatchOutput {
    pub fn row_for_params(&self, _: &WadParams) -> Option<usize> {
        Some(0)
    }
    pub fn values_for(&self, _: &WadParams) -> Option<&[f64]> {
        Some(&self.values)
    }
}

#[inline(always)]
pub fn expand_grid(_r: &WadBatchRange) -> Vec<WadParams> {
    vec![WadParams]
}

#[inline(always)]
pub fn wad_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WadBatchOutput, WadError> {
    wad_batch_inner(high, low, close, kern, false)
}
#[inline(always)]
pub fn wad_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WadBatchOutput, WadError> {
    wad_batch_inner(high, low, close, kern, true)
}

#[inline(always)]
fn wad_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<WadBatchOutput, WadError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WadError::EmptyData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(WadError::EmptyData);
    }
    if high.iter().all(|x| x.is_nan()) || low.iter().all(|x| x.is_nan()) || close.iter().all(|x| x.is_nan()) {
        return Err(WadError::AllValuesNaN);
    }
    let mut values = vec![0.0; len];
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => wad_row_scalar(high, low, close, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wad_row_avx2(high, low, close, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wad_row_avx512(high, low, close, &mut values),
            _ => unreachable!(),
        }
    }
    Ok(WadBatchOutput { values, rows: 1, cols: len })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use std::error::Error;

    fn check_wad_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WadInput::from_candles(&candles);
        let output = wad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_wad_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WadInput::from_candles(&candles);
        let output = wad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let expected_last_five_wad = [
            158503.46790000016,
            158279.46790000016,
            158014.46790000016,
            158186.46790000016,
            157605.46790000016,
        ];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let exp = expected_last_five_wad[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "[{}] WAD {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                exp
            );
        }
        Ok(())
    }

    fn check_wad_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input = WadInput::from_slices(&[], &[], &[]);
        let result = wad_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_wad_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_slice = [f64::NAN, f64::NAN, f64::NAN];
        let input = WadInput::from_slices(&nan_slice, &nan_slice, &nan_slice);
        let result = wad_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_wad_basic_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 11.0, 12.0];
        let low = [9.0, 9.0, 10.0, 10.0];
        let close = [9.5, 10.5, 10.5, 11.5];
        let input = WadInput::from_slices(&high, &low, &close);
        let output = wad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), 4);
        assert!((output.values[0] - 0.0).abs() < 1e-10);
        assert!((output.values[1] - 1.5).abs() < 1e-10);
        assert!((output.values[2] - 1.5).abs() < 1e-10);
        assert!((output.values[3] - 3.0).abs() < 1e-10);
        Ok(())
    }

    fn check_wad_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let close = source_type(&candles, "close");
        let batch_output = wad_with_kernel(&WadInput::from_slices(high, low, close), kernel)?.values;
        let mut stream = WadStream::try_new()?;
        let mut stream_values = Vec::with_capacity(close.len());
        for ((&h, &l), &c) in high.iter().zip(low).zip(close) {
            stream_values.push(stream.update(h, l, c));
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] WAD streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    // New: small fixed‐input test for wad_scalar via wad_with_kernel
    fn check_wad_small_example(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        // 5‐bar example from documentation:
        let high = [10.0, 11.0, 12.0, 11.5, 12.5];
        let low = [ 9.0,  9.5, 11.0, 10.5, 11.0];
        let close = [ 9.5, 10.5, 11.5, 11.0, 12.0];
        let expected = [0.0, 1.0, 2.0, 1.5, 2.5];

        let input = WadInput::from_slices(&high, &low, &close);
        let output = wad_with_kernel(&input, kernel)?;
        // output.values should be length 5
        assert_eq!(output.values.len(), 5);

        for i in 0..5 {
            let got = output.values[i];
            let exp = expected[i];
            assert!(
                (got - exp).abs() < 1e-10,
                "[{}] WAD {:?} small example mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                got,
                exp
            );
        }

        Ok(())
    }

    macro_rules! generate_all_wad_tests {
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

    generate_all_wad_tests!(
        check_wad_partial_params,
        check_wad_accuracy,
        check_wad_empty_data,
        check_wad_all_values_nan,
        check_wad_basic_slice,
        check_wad_streaming,
        check_wad_small_example
    );
}
