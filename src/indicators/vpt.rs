//! # Volume Price Trend (VPT)
//!
//! Exact match for Jesse's implementation (shifted array approach).
//!
//! ## Parameters
//! None (uses price/volume arrays).
//!
//! ## Errors
//! - **EmptyData**: vpt: Input price or volume data is empty or mismatched.
//! - **AllValuesNaN**: vpt: All input price or volume values are NaN.
//! - **NotEnoughValidData**: vpt: Fewer than 2 valid price/volume points.
//!
//! ## Returns
//! - **Ok(VptOutput)** with output array.
//! - **Err(VptError)** otherwise.

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

#[derive(Debug, Clone)]
pub enum VptData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VptOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct VptParams;

#[derive(Debug, Clone)]
pub struct VptInput<'a> {
    pub data: VptData<'a>,
    pub params: VptParams,
}

impl<'a> VptInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
        Self {
            data: VptData::Candles { candles, source },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn from_slices(price: &'a [f64], volume: &'a [f64]) -> Self {
        Self {
            data: VptData::Slices { price, volume },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VptData::Candles {
                candles,
                source: "close",
            },
            params: VptParams::default(),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VptBuilder {
    kernel: Kernel,
}

impl VptBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self { kernel: Kernel::Auto }
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VptOutput, VptError> {
        let i = VptInput::with_default_candles(c);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
        let i = VptInput::from_slices(price, volume);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> VptStream {
        VptStream::default()
    }
}

#[derive(Debug, Error)]
pub enum VptError {
    #[error("vpt: Empty data provided.")]
    EmptyData,
    #[error("vpt: All price/volume values are NaN.")]
    AllValuesNaN,
    #[error("vpt: Not enough valid data (fewer than 2 valid points).")]
    NotEnoughValidData,
}

#[inline]
pub fn vpt(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, Kernel::Auto)
}

pub fn vpt_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    let (price, volume) = match &input.data {
        VptData::Candles { candles, source } => {
            let price = source_type(candles, source);
            let vol = candles
                .select_candle_field("volume")
                .map_err(|_| VptError::EmptyData)?;
            (price, vol)
        }
        VptData::Slices { price, volume } => (*price, *volume),
    };

    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyData);
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData);
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vpt_scalar(price, volume)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vpt_avx2(price, volume)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpt_avx512(price, volume)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn vpt_scalar(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    let n = price.len();
    let mut vpt_val = vec![f64::NAN; n];
    for i in 1..n {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
            vpt_val[i] = f64::NAN;
        } else {
            vpt_val[i] = v1 * ((p1 - p0) / p0);
        }
    }
    let mut res = vec![f64::NAN; n];
    for i in 1..n {
        let shifted = vpt_val[i - 1];
        if vpt_val[i].is_nan() || shifted.is_nan() {
            res[i] = f64::NAN;
        } else {
            res[i] = vpt_val[i] + shifted;
        }
    }
    Ok(VptOutput { values: res })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx2(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    // For API parity only; reuses scalar logic.
    vpt_scalar(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    // For API parity only; reuses scalar logic.
    vpt_scalar(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_short(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_long(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[inline]
pub fn vpt_indicator(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt(input)
}

#[inline]
pub fn vpt_indicator_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, kernel)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx2(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx2(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_short(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_short(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_long(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_long(price, volume)
    }
}

#[inline]
pub fn vpt_indicator_scalar(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_scalar(price, volume)
    }
}

#[inline]
pub fn vpt_expand_grid() -> Vec<VptParams> {
    vec![VptParams]
}

#[derive(Clone, Debug, Default)]
pub struct VptStream {
    last_price: f64,
    last_vpt: f64,
    is_initialized: bool,
}

impl VptStream {
    #[inline]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        if !self.is_initialized {
            self.last_price = price;
            self.last_vpt = f64::NAN;
            self.is_initialized = true;
            return None;
        }
        if self.last_price.is_nan() || self.last_price == 0.0 || price.is_nan() || volume.is_nan() {
            self.last_price = price;
            self.last_vpt = f64::NAN;
            return Some(f64::NAN);
        }
        let vpt_val = volume * ((price - self.last_price) / self.last_price);
        let out = if self.last_vpt.is_nan() {
            f64::NAN
        } else {
            vpt_val + self.last_vpt
        };
        self.last_price = price;
        self.last_vpt = vpt_val;
        Some(out)
    }
}

#[derive(Clone, Debug, Default)]
pub struct VptBatchRange;

#[derive(Clone, Debug, Default)]
pub struct VptBatchBuilder {
    kernel: Kernel,
}

impl VptBatchBuilder {
    pub fn new() -> Self {
        Self { kernel: Kernel::Auto }
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptBatchOutput, VptError> {
        vpt_batch_with_kernel(price, volume, self.kernel)
    }

    pub fn with_default_slices(price: &[f64], volume: &[f64], k: Kernel) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new().kernel(k).apply_slices(price, volume)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VptBatchOutput, VptError> {
        let price = source_type(c, src);
        let volume = c.select_candle_field("volume").map_err(|_| VptError::EmptyData)?;
        self.apply_slices(price, volume)
    }

    pub fn with_default_candles(c: &Candles) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn vpt_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    k: Kernel,
) -> Result<VptBatchOutput, VptError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    vpt_batch_par_slice(price, volume, kernel)
}

#[derive(Clone, Debug)]
pub struct VptBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VptParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VptBatchOutput {
    pub fn row_for_params(&self, _p: &VptParams) -> Option<usize> {
        Some(0)
    }

    pub fn values_for(&self, _p: &VptParams) -> Option<&[f64]> {
        Some(&self.values[..])
    }
}

#[inline(always)]
pub fn vpt_batch_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, false)
}

#[inline(always)]
pub fn vpt_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, true)
}

#[inline(always)]
fn vpt_batch_inner(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<VptBatchOutput, VptError> {
    let combos = vpt_expand_grid();
    let rows = 1;
    let cols = price.len();

    let output = match kern {
        Kernel::Scalar | Kernel::ScalarBatch => unsafe { vpt_scalar(price, volume)? },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => unsafe { vpt_avx2(price, volume)? },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => unsafe { vpt_avx512(price, volume)? },
        _ => unreachable!(),
    };

    Ok(VptBatchOutput {
        values: output.values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn vpt_row_scalar(
    price: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    let n = price.len();
    let mut vpt_val = vec![f64::NAN; n];
    for i in 1..n {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
            vpt_val[i] = f64::NAN;
        } else {
            vpt_val[i] = v1 * ((p1 - p0) / p0);
        }
    }
    for i in 1..n {
        let shifted = vpt_val[i - 1];
        if vpt_val[i].is_nan() || shifted.is_nan() {
            out[i] = f64::NAN;
        } else {
            out[i] = vpt_val[i] + shifted;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_short(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_long(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_vpt_basic_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vpt_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [1.0, 1.1, 1.05, 1.2, 1.3];
        let volume = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
        let input = VptInput::from_slices(&price, &volume);
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), price.len());
        Ok(())
    }

    fn check_vpt_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [100.0];
        let volume = [500.0];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN, f64::NAN];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_accuracy_from_csv(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -0.40358334248536065,
            -0.16292768139917702,
            -0.4792942916867958,
            -0.1188231211518107,
            -3.3492674990910025,
        ];

        assert!(output.values.len() >= 5);
        let start_index = output.values.len() - 5;
        for (i, &value) in output.values[start_index..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "VPT mismatch at final bars, index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    macro_rules! generate_all_vpt_tests {
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

    generate_all_vpt_tests!(
        check_vpt_basic_candles,
        check_vpt_basic_slices,
        check_vpt_not_enough_data,
        check_vpt_empty_data,
        check_vpt_all_nan,
        check_vpt_accuracy_from_csv
    );
}