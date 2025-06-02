//! # Negative Volume Index (NVI)
//!
//! The NVI (Negative Volume Index) focuses on days when the volume decreases from the previous day.
//! This implementation follows Tulip Indicators and does not take any parameters.
//!
//! ## Errors
//! - **EmptyData**: nvi: Input data slice(s) is empty.
//! - **AllCloseValuesNaN**: nvi: All close input values are `NaN`.
//! - **AllVolumeValuesNaN**: nvi: All volume input values are `NaN`.
//! - **NotEnoughValidData**: nvi: Fewer than 2 valid (non-`NaN`) data points after the first valid index.
//!
//! ## Returns
//! - **`Ok(NviOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(NviError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum NviData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct NviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct NviParams; // no params

#[derive(Debug, Clone)]
pub struct NviInput<'a> {
    pub data: NviData<'a>,
    pub params: NviParams,
}

impl<'a> NviInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, close_source: &'a str, params: NviParams) -> Self {
        Self { data: NviData::Candles { candles, close_source }, params }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: NviParams) -> Self {
        Self { data: NviData::Slices { close, volume }, params }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", NviParams)
    }
}

#[derive(Debug, Error)]
pub enum NviError {
    #[error("nvi: Empty data provided.")]
    EmptyData,
    #[error("nvi: All close values are NaN.")]
    AllCloseValuesNaN,
    #[error("nvi: All volume values are NaN.")]
    AllVolumeValuesNaN,
    #[error("nvi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[derive(Copy, Clone, Debug, Default)]
pub struct NviBuilder {
    kernel: Kernel,
}
impl NviBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self { kernel: Kernel::Auto } }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<NviOutput, NviError> {
        let i = NviInput::with_default_candles(c);
        nvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, close: &[f64], volume: &[f64]) -> Result<NviOutput, NviError> {
        let i = NviInput::from_slices(close, volume, NviParams);
        nvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NviStream, NviError> {
        NviStream::try_new()
    }
}

#[derive(Debug, Clone)]
pub struct NviStream {
    prev_close: Option<f64>,
    prev_volume: Option<f64>,
    nvi_val: f64,
    started: bool,
}
impl NviStream {
    pub fn try_new() -> Result<Self, NviError> {
        Ok(Self {
            prev_close: None,
            prev_volume: None,
            nvi_val: 1000.0,
            started: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        if !self.started && !close.is_nan() && !volume.is_nan() {
            self.prev_close = Some(close);
            self.prev_volume = Some(volume);
            self.started = true;
            return Some(self.nvi_val);
        }
        if !self.started { return None; }
        let prev_c = self.prev_close?;
        let prev_v = self.prev_volume?;
        let mut new_nvi = self.nvi_val;
        if volume < prev_v {
            let pct = (close - prev_c) / prev_c;
            new_nvi += new_nvi * pct;
        }
        self.nvi_val = new_nvi;
        self.prev_close = Some(close);
        self.prev_volume = Some(volume);
        Some(self.nvi_val)
    }
}

#[derive(Clone, Debug, Default)]
pub struct NviBatchRange; // NVI has no sweep

#[derive(Clone, Debug, Default)]
pub struct NviBatchBuilder {
    kernel: Kernel,
}
impl NviBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn apply_slice(self, close: &[f64], volume: &[f64]) -> Result<NviBatchOutput, NviError> {
        nvi_batch_with_kernel(close, volume, self.kernel)
    }
    pub fn with_default_slice(close: &[f64], volume: &[f64], k: Kernel) -> Result<NviBatchOutput, NviError> {
        NviBatchBuilder::new().kernel(k).apply_slice(close, volume)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<NviBatchOutput, NviError> {
        let close = source_type(c, src);
        let volume = c.select_candle_field("volume").map_err(|_| NviError::EmptyData)?;
        self.apply_slice(close, volume)
    }
    pub fn with_default_candles(c: &Candles) -> Result<NviBatchOutput, NviError> {
        NviBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct NviBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
impl NviBatchOutput {
    pub fn row_for_params(&self, _p: &NviParams) -> Option<usize> { Some(0) }
    pub fn values_for(&self, _p: &NviParams) -> Option<&[f64]> {
        Some(&self.values)
    }
}

#[inline]
pub fn nvi(input: &NviInput) -> Result<NviOutput, NviError> {
    nvi_with_kernel(input, Kernel::Auto)
}
pub fn nvi_with_kernel(input: &NviInput, kernel: Kernel) -> Result<NviOutput, NviError> {
    let (close, volume): (&[f64], &[f64]) = match &input.data {
        NviData::Candles { candles, close_source } => {
            let close = source_type(candles, close_source);
            let volume = candles.select_candle_field("volume").map_err(|_| NviError::EmptyData)?;
            (close, volume)
        }
        NviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() { return Err(NviError::EmptyData); }
    let first_valid_idx = close.iter().zip(volume.iter()).position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| if close.iter().all(|&c| c.is_nan()) { NviError::AllCloseValuesNaN } else { NviError::AllVolumeValuesNaN })?;
    if (close.len() - first_valid_idx) < 2 {
        return Err(NviError::NotEnoughValidData { needed: 2, valid: close.len() - first_valid_idx });
    }
    let mut out = vec![f64::NAN; close.len()];
    let chosen = match kernel { Kernel::Auto => detect_best_kernel(), other => other };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => nvi_scalar(close, volume, first_valid_idx, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => nvi_avx2(close, volume, first_valid_idx, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => nvi_avx512(close, volume, first_valid_idx, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(NviOutput { values: out })
}

#[inline]
pub unsafe fn nvi_scalar(
    close: &[f64], volume: &[f64], first: usize, out: &mut [f64]
) {
    let mut nvi_val = 1000.0;
    out[first] = nvi_val;
    let mut prev_close = close[first];
    let mut prev_volume = volume[first];
    for i in (first+1)..close.len() {
        if volume[i] < prev_volume {
            let pct_change = (close[i] - prev_close) / prev_close;
            nvi_val += nvi_val * pct_change;
        }
        out[i] = nvi_val;
        prev_close = close[i];
        prev_volume = volume[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx2(
    close: &[f64], volume: &[f64], first: usize, out: &mut [f64]
) {
    nvi_scalar(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512(
    close: &[f64], volume: &[f64], first: usize, out: &mut [f64]
) {
    nvi_scalar(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline] pub unsafe fn nvi_avx512_short(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) { nvi_scalar(close, volume, first, out) }
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline] pub unsafe fn nvi_avx512_long(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) { nvi_scalar(close, volume, first, out) }

#[inline(always)]
pub fn nvi_batch_with_kernel(
    close: &[f64], volume: &[f64], k: Kernel
) -> Result<NviBatchOutput, NviError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(NviError::EmptyData),
    };
    nvi_batch_par_slice(close, volume, kernel)
}
#[inline(always)]
pub fn nvi_batch_slice(close: &[f64], volume: &[f64], kern: Kernel) -> Result<NviBatchOutput, NviError> {
    nvi_batch_inner(close, volume, kern, false)
}
#[inline(always)]
pub fn nvi_batch_par_slice(close: &[f64], volume: &[f64], kern: Kernel) -> Result<NviBatchOutput, NviError> {
    nvi_batch_inner(close, volume, kern, true)
}

fn nvi_batch_inner(close: &[f64], volume: &[f64], kern: Kernel, _parallel: bool) -> Result<NviBatchOutput, NviError> {
    let mut values = vec![f64::NAN; close.len()];
    let first = close.iter().zip(volume.iter()).position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| if close.iter().all(|&c| c.is_nan()) { NviError::AllCloseValuesNaN } else { NviError::AllVolumeValuesNaN })?;
    unsafe {
        match kern {
            Kernel::ScalarBatch | Kernel::Scalar => nvi_row_scalar(close, volume, first, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => nvi_row_avx2(close, volume, first, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => nvi_row_avx512(close, volume, first, &mut values),
            _ => unreachable!(),
        }
    }
    Ok(NviBatchOutput { values, rows: 1, cols: close.len() })
}

#[inline(always)]
pub unsafe fn nvi_row_scalar(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_scalar(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nvi_row_avx2(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_avx2(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nvi_row_avx512(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_avx512(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nvi_row_avx512_short(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_avx512_short(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nvi_row_avx512_long(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_avx512_long(close, volume, first, out)
}

#[inline(always)]
fn expand_grid(_r: &NviBatchRange) -> Vec<NviParams> { vec![NviParams] }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_nvi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NviInput::with_default_candles(&candles);
        let output = nvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_nvi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NviInput::with_default_candles(&candles);
        let result = nvi_with_kernel(&input, kernel)?;
        let expected_last_five = [
            17555.49871646325,
            17524.70219345554,
            17524.70219345554,
            17559.13477961792,
            17559.13477961792,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-5, "[{}] NVI {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }

    fn check_nvi_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data: [f64; 0] = [];
        let volume_data: [f64; 0] = [];
        let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
        let res = nvi_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] NVI should fail with empty data", test_name);
        Ok(())
    }

    fn check_nvi_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [f64::NAN, 100.0];
        let volume_data = [f64::NAN, 120.0];
        let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
        let res = nvi_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] NVI should fail with not enough valid data", test_name);
        Ok(())
    }

    fn check_nvi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close = candles.select_candle_field("close")?;
        let volume = candles.select_candle_field("volume")?;
        let input = NviInput::from_slices(close, volume, NviParams);
        let batch_output = nvi_with_kernel(&input, kernel)?.values;
        let mut stream = NviStream::try_new()?;
        let mut stream_values = Vec::with_capacity(close.len());
        for (&c, &v) in close.iter().zip(volume.iter()) {
            match stream.update(c, v) {
                Some(nvi_val) => stream_values.push(nvi_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(diff < 1e-9, "[{}] NVI streaming mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_nvi_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }

    generate_all_nvi_tests!(
        check_nvi_partial_params,
        check_nvi_accuracy,
        check_nvi_empty_data,
        check_nvi_not_enough_valid_data,
        check_nvi_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = NviBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let row = output.values_for(&NviParams).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            17555.49871646325,
            17524.70219345554,
            17524.70219345554,
            17559.13477961792,
            17559.13477961792,
        ];
        let start = row.len().saturating_sub(5);
        for (i, &v) in row[start..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-5, "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}");
        }
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
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
