//! # Market Facilitation Index (marketefi)
//!
//! Market Facilitation Index (marketefi) measures price movement efficiency relative to trading volume.
//!
//! ## Parameters
//! - No adjustable parameters; calculation is direct.
//!
//! ## Errors
//! - **EmptyData**: marketefi: Input data slice is empty.
//! - **MismatchedDataLength**: marketefi: `high`, `low`, and `volume` slices do not have the same length.
//! - **AllValuesNaN**: marketefi: All input data values are `NaN`.
//! - **NotEnoughValidData**: marketefi: No calculable values remain after the first valid index.
//! - **ZeroOrNaNVolume**: marketefi: Volume is zero or NaN at a valid index.
//!
//! ## Returns
//! - **`Ok(MarketefiOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the first valid index.
//! - **`Err(MarketefiError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum MarketefiData<'a> {
    Candles {
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
        source_volume: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MarketefiParams;

impl Default for MarketefiParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct MarketefiInput<'a> {
    pub data: MarketefiData<'a>,
    pub params: MarketefiParams,
}

impl<'a> MarketefiInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
        source_volume: &'a str,
        params: MarketefiParams,
    ) -> Self {
        Self {
            data: MarketefiData::Candles {
                candles,
                source_high,
                source_low,
                source_volume,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        volume: &'a [f64],
        params: MarketefiParams,
    ) -> Self {
        Self {
            data: MarketefiData::Slices { high, low, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "high",
            "low",
            "volume",
            MarketefiParams::default(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct MarketefiOutput {
    pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MarketefiBuilder {
    kernel: Kernel,
}

impl MarketefiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(
        self,
        c: &Candles,
    ) -> Result<MarketefiOutput, MarketefiError> {
        let i = MarketefiInput::with_default_candles(c);
        marketefi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        volume: &[f64],
    ) -> Result<MarketefiOutput, MarketefiError> {
        let i = MarketefiInput::from_slices(high, low, volume, MarketefiParams::default());
        marketefi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> MarketefiStream {
        MarketefiStream::new()
    }
}

#[derive(Debug, Error)]
pub enum MarketefiError {
    #[error("marketefi: Empty data provided.")]
    EmptyData,
    #[error("marketefi: Mismatched data length among high, low, and volume.")]
    MismatchedDataLength,
    #[error("marketefi: All values are NaN.")]
    AllValuesNaN,
    #[error("marketefi: Not enough valid data to calculate.")]
    NotEnoughValidData,
    #[error("marketefi: Zero or NaN volume at a valid index.")]
    ZeroOrNaNVolume,
}

#[inline]
pub fn marketefi(input: &MarketefiInput) -> Result<MarketefiOutput, MarketefiError> {
    marketefi_with_kernel(input, Kernel::Auto)
}

pub fn marketefi_with_kernel(
    input: &MarketefiInput,
    kernel: Kernel,
) -> Result<MarketefiOutput, MarketefiError> {
    let (high, low, volume) = match &input.data {
        MarketefiData::Candles {
            candles,
            source_high,
            source_low,
            source_volume,
        } => (
            source_type(candles, source_high),
            source_type(candles, source_low),
            source_type(candles, source_volume),
        ),
        MarketefiData::Slices { high, low, volume } => (*high, *low, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(MarketefiError::EmptyData);
    }
    if high.len() != low.len() || low.len() != volume.len() {
        return Err(MarketefiError::MismatchedDataLength);
    }
    let len = high.len();
    let first = (0..len).find(|&i| {
        let h = high[i];
        let l = low[i];
        let v = volume[i];
        !(h.is_nan() || l.is_nan() || v.is_nan())
    }).ok_or(MarketefiError::AllValuesNaN)?;
    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                marketefi_scalar(high, low, volume, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                marketefi_avx2(high, low, volume, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                marketefi_avx512(high, low, volume, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    let valid_count = out[first..].iter().filter(|v| !v.is_nan()).count();
    if valid_count == 0 {
        return Err(MarketefiError::NotEnoughValidData);
    }
    if out[first..].iter().all(|&val| val.is_nan()) {
        return Err(MarketefiError::ZeroOrNaNVolume);
    }
    Ok(MarketefiOutput { values: out })
}

#[inline]
pub fn marketefi_scalar(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    for i in first_valid..high.len() {
        let h = high[i];
        let l = low[i];
        let v = volume[i];
        if h.is_nan() || l.is_nan() || v.is_nan() || v == 0.0 {
            out[i] = f64::NAN;
        } else {
            out[i] = (h - l) / v;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx512(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { marketefi_avx512_short(high, low, volume, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx2(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx512_short(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx512_long(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first_valid, out)
}

// Row/batch interface

#[inline(always)]
pub fn marketefi_row_scalar(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx2(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx512(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx512_short(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx512_long(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    marketefi_scalar(high, low, volume, first, out)
}

#[derive(Clone, Debug)]
pub struct MarketefiBatchRange; // No params, just 1 row.

impl Default for MarketefiBatchRange {
    fn default() -> Self { Self }
}

#[derive(Clone, Debug, Default)]
pub struct MarketefiBatchBuilder {
    kernel: Kernel,
}

impl MarketefiBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        volume: &[f64],
    ) -> Result<MarketefiBatchOutput, MarketefiError> {
        marketefi_batch_with_kernel(high, low, volume, self.kernel)
    }
    pub fn with_default_candles(
        c: &Candles
    ) -> Result<MarketefiBatchOutput, MarketefiError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let volume = source_type(c, "volume");
        MarketefiBatchBuilder::new().kernel(Kernel::Auto)
            .apply_slices(high, low, volume)
    }
}

pub fn marketefi_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<MarketefiBatchOutput, MarketefiError> {
    let k = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        x if x.is_batch() => x,
        _ => Kernel::ScalarBatch,
    };
    marketefi_batch_par_slice(high, low, volume, k)
}

#[derive(Clone, Debug)]
pub struct MarketefiBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
pub fn marketefi_batch_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<MarketefiBatchOutput, MarketefiError> {
    marketefi_batch_inner(high, low, volume, kernel, false)
}

#[inline(always)]
pub fn marketefi_batch_par_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<MarketefiBatchOutput, MarketefiError> {
    marketefi_batch_inner(high, low, volume, kernel, true)
}

#[inline(always)]
fn marketefi_batch_inner(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kernel: Kernel,
    _parallel: bool,
) -> Result<MarketefiBatchOutput, MarketefiError> {
    let len = high.len();
    let mut out = vec![f64::NAN; len];
    let first = (0..len).find(|&i| {
        let h = high[i];
        let l = low[i];
        let v = volume[i];
        !(h.is_nan() || l.is_nan() || v.is_nan())
    }).ok_or(MarketefiError::AllValuesNaN)?;
    unsafe {
        match kernel {
            Kernel::ScalarBatch | Kernel::Scalar => {
                marketefi_row_scalar(high, low, volume, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => {
                marketefi_row_avx2(high, low, volume, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => {
                marketefi_row_avx512(high, low, volume, first, &mut out)
            }
            _ => marketefi_row_scalar(high, low, volume, first, &mut out),
        }
    }
    Ok(MarketefiBatchOutput {
        values: out,
        rows: 1,
        cols: len,
    })
}

#[inline(always)]
pub fn expand_grid(_: &MarketefiBatchRange) -> Vec<MarketefiParams> {
    vec![MarketefiParams]
}

// Streaming (single-point rolling)
#[derive(Debug, Clone)]
pub struct MarketefiStream;

impl MarketefiStream {
    pub fn new() -> Self { Self }
    pub fn update(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() || volume.is_nan() || volume == 0.0 {
            None
        } else {
            Some((high - low) / volume)
        }
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_marketefi_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MarketefiInput::with_default_candles(&candles);
        let res = marketefi_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        let expected_last_five = [
            2.8460112192104607,
            3.020938522420525,
            3.0474861329079292,
            3.691017115591989,
            2.247810963176202,
        ];
        let start = res.values.len() - 5;
        for (i, &v) in res.values[start..].iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (v - exp).abs() < 1e-6,
                "[{}] marketefi mismatch at {}: got {}, exp {}",
                test,
                start + i,
                v,
                exp
            );
        }
        Ok(())
    }

    fn check_marketefi_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [f64::NAN, 2.0, 3.0];
        let low = [f64::NAN, 1.0, 2.0];
        let vol = [f64::NAN, 1.0, 1.0];
        let input = MarketefiInput::from_slices(&high, &low, &vol, MarketefiParams::default());
        let res = marketefi_with_kernel(&input, kernel)?;
        assert!(res.values[0].is_nan());
        assert_eq!(res.values[1], 1.0 / 1.0);
        Ok(())
    }

    fn check_marketefi_empty_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let input = MarketefiInput::from_slices(&[], &[], &[], MarketefiParams::default());
        let res = marketefi_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_marketefi_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [3.0, 4.0, 5.0];
        let low = [2.0, 3.0, 3.0];
        let vol = [1.0, 2.0, 2.0];
        let mut stream = MarketefiStream::new();
        let mut vals = Vec::new();
        for i in 0..high.len() {
            vals.push(stream.update(high[i], low[i], vol[i]).unwrap_or(f64::NAN));
        }
        let input = MarketefiInput::from_slices(&high, &low, &vol, MarketefiParams::default());
        let res = marketefi_with_kernel(&input, kernel)?;
        for (a, b) in vals.iter().zip(res.values.iter()) {
            if a.is_nan() && b.is_nan() { continue; }
            assert!((a - b).abs() < 1e-8);
        }
        Ok(())
    }

    macro_rules! generate_all_marketefi_tests {
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
    generate_all_marketefi_tests!(
        check_marketefi_accuracy,
        check_marketefi_nan_handling,
        check_marketefi_empty_data,
        check_marketefi_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let out = MarketefiBatchBuilder::new().kernel(kernel)
            .apply_slices(
                source_type(&candles, "high"),
                source_type(&candles, "low"),
                source_type(&candles, "volume"),
            )?;
        let expected_last_five = [
            2.8460112192104607,
            3.020938522420525,
            3.0474861329079292,
            3.691017115591989,
            2.247810963176202,
        ];
        let row = &out.values;
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (v - exp).abs() < 1e-8,
                "[{test}] batch row mismatch at {i}: {v} vs {exp}"
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
