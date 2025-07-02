//! # Absolute Price Oscillator (APO)
//!
//! Calculates the difference between two exponential moving averages (EMAs) of
//! different lengths (`short_period` and `long_period`), measuring momentum and
//! trend shifts. The interface and performance are structured similar to ALMA.
//!
//! ## Parameters
//! - **short_period**: EMA window size for the short period (defaults to 10).
//! - **long_period**: EMA window size for the long period (defaults to 20).
//!
//! ## Errors
//! - **AllValuesNaN**: apo: All input data values are `NaN`.
//! - **InvalidPeriod**: apo: Periods are zero, or invalid.
//! - **ShortPeriodNotLessThanLong**: apo: `short_period` is not less than `long_period`.
//! - **NotEnoughValidData**: apo: Not enough valid data for the requested `long_period`.
//!
//! ## Returns
//! - **`Ok(ApoOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(ApoError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

// --- Data Representation

#[derive(Debug, Clone)]
pub enum ApoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for ApoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ApoData::Slice(slice) => slice,
            ApoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// --- Structs and Params

#[derive(Debug, Clone)]
pub struct ApoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ApoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}
impl Default for ApoParams {
    fn default() -> Self {
        Self {
            short_period: Some(10),
            long_period: Some(20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApoInput<'a> {
    pub data: ApoData<'a>,
    pub params: ApoParams,
}
impl<'a> ApoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ApoParams) -> Self {
        Self {
            data: ApoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ApoParams) -> Self {
        Self {
            data: ApoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ApoParams::default())
    }
    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(10)
    }
    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(20)
    }
}

// --- Error Types

#[derive(Debug, Error)]
pub enum ApoError {
    #[error("apo: All values are NaN.")]
    AllValuesNaN,
    #[error("apo: Invalid period: short={short}, long={long}")]
    InvalidPeriod { short: usize, long: usize },
    #[error("apo: short_period not less than long_period: short={short}, long={long}")]
    ShortPeriodNotLessThanLong { short: usize, long: usize },
    #[error("apo: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// --- Builder API

#[derive(Copy, Clone, Debug)]
pub struct ApoBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}
impl Default for ApoBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}
impl ApoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<ApoOutput, ApoError> {
        let p = ApoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = ApoInput::from_candles(c, "close", p);
        apo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ApoOutput, ApoError> {
        let p = ApoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = ApoInput::from_slice(d, p);
        apo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ApoStream, ApoError> {
        let p = ApoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        ApoStream::try_new(p)
    }
}

// --- Main Indicator Function

#[inline]
pub fn apo(input: &ApoInput) -> Result<ApoOutput, ApoError> {
    apo_with_kernel(input, Kernel::Auto)
}

pub fn apo_with_kernel(input: &ApoInput, kernel: Kernel) -> Result<ApoOutput, ApoError> {
    let data: &[f64] = input.as_ref();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ApoError::AllValuesNaN)?;
    let len = data.len();
    let short = input.get_short_period();
    let long = input.get_long_period();

    if short == 0 || long == 0 {
        return Err(ApoError::InvalidPeriod { short, long });
    }
    if short >= long {
        return Err(ApoError::ShortPeriodNotLessThanLong { short, long });
    }
    if (len - first) < long {
        return Err(ApoError::NotEnoughValidData {
            needed: long,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => apo_scalar(data, short, long, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => apo_avx2(data, short, long, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => apo_avx512(data, short, long, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(ApoOutput { values: out })
}

// --- Scalar Kernel

#[inline]
pub fn apo_scalar(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);

    let mut short_ema = data[first];
    let mut long_ema = data[first];

    for i in 0..data.len() {
        let price = data[i];
        if i < first {
            out[i] = f64::NAN;
            continue;
        }
        if i == first {
            short_ema = price;
            long_ema = price;
            out[i] = short_ema - long_ema;
            continue;
        }
        short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
        long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;
        out[i] = short_ema - long_ema;
    }
}

// --- AVX2/AVX512 Kernels: Stubs

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn apo_avx2(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    // Stub: Use scalar fallback.
    apo_scalar(data, short, long, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn apo_avx512(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    // Choose between short/long, stub both to scalar
    if long <= 32 {
        apo_avx512_short(data, short, long, first, out);
    } else {
        apo_avx512_long(data, short, long, first, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn apo_avx512_short(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    apo_scalar(data, short, long, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn apo_avx512_long(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    apo_scalar(data, short, long, first, out);
}

// --- Batch, Streaming, and Builder APIs

#[derive(Clone, Debug)]
pub struct ApoStream {
    short: usize,
    long: usize,
    alpha_short: f64,
    alpha_long: f64,
    short_ema: f64,
    long_ema: f64,
    filled: bool,
    nan_leading: usize,
    seen: usize,
}

impl ApoStream {
    pub fn try_new(params: ApoParams) -> Result<Self, ApoError> {
        let short = params.short_period.unwrap_or(10);
        let long = params.long_period.unwrap_or(20);
        if short == 0 || long == 0 {
            return Err(ApoError::InvalidPeriod { short, long });
        }
        if short >= long {
            return Err(ApoError::ShortPeriodNotLessThanLong { short, long });
        }
        Ok(Self {
            short,
            long,
            alpha_short: 2.0 / (short as f64 + 1.0),
            alpha_long: 2.0 / (long as f64 + 1.0),
            short_ema: f64::NAN,
            long_ema: f64::NAN,
            filled: false,
            nan_leading: 0,
            seen: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<f64> {
        if !self.filled && price.is_nan() {
            self.nan_leading += 1;
            return None;
        }
        if !self.filled {
            self.short_ema = price;
            self.long_ema = price;
            self.filled = true;
            self.seen = 1;
            return Some(0.0);
        }
        self.seen += 1;
        self.short_ema = self.alpha_short * price + (1.0 - self.alpha_short) * self.short_ema;
        self.long_ema = self.alpha_long * price + (1.0 - self.alpha_long) * self.long_ema;
        Some(self.short_ema - self.long_ema)
    }
}

// --- Batch Sweeping API

#[derive(Clone, Debug)]
pub struct ApoBatchRange {
    pub short: (usize, usize, usize),
    pub long: (usize, usize, usize),
}
impl Default for ApoBatchRange {
    fn default() -> Self {
        Self {
            short: (5, 20, 5),
            long: (15, 50, 5),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ApoBatchBuilder {
    range: ApoBatchRange,
    kernel: Kernel,
}
impl ApoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short = (start, end, step);
        self
    }
    pub fn short_static(mut self, s: usize) -> Self {
        self.range.short = (s, s, 0);
        self
    }
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long = (start, end, step);
        self
    }
    pub fn long_static(mut self, s: usize) -> Self {
        self.range.long = (s, s, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<ApoBatchOutput, ApoError> {
        apo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ApoBatchOutput, ApoError> {
        ApoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ApoBatchOutput, ApoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ApoBatchOutput, ApoError> {
        ApoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct ApoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ApoParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ApoBatchOutput {
    pub fn row_for_params(&self, p: &ApoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(10) == p.short_period.unwrap_or(10)
                && c.long_period.unwrap_or(20) == p.long_period.unwrap_or(20)
        })
    }
    pub fn values_for(&self, p: &ApoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// --- Grid Expansion

#[inline(always)]
fn expand_grid(r: &ApoBatchRange) -> Vec<ApoParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis(r.short);
    let longs = axis(r.long);
    let mut out = Vec::with_capacity(shorts.len() * longs.len());
    for &s in &shorts {
        for &l in &longs {
            if s < l && s > 0 && l > 0 {
                out.push(ApoParams {
                    short_period: Some(s),
                    long_period: Some(l),
                });
            }
        }
    }
    out
}

// --- Batch Slice API

#[inline(always)]
pub fn apo_batch_with_kernel(
    data: &[f64],
    sweep: &ApoBatchRange,
    k: Kernel,
) -> Result<ApoBatchOutput, ApoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(ApoError::InvalidPeriod { short: 0, long: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    apo_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn apo_batch_slice(
    data: &[f64],
    sweep: &ApoBatchRange,
    kern: Kernel,
) -> Result<ApoBatchOutput, ApoError> {
    apo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn apo_batch_par_slice(
    data: &[f64],
    sweep: &ApoBatchRange,
    kern: Kernel,
) -> Result<ApoBatchOutput, ApoError> {
    apo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn apo_batch_inner(
    data: &[f64],
    sweep: &ApoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ApoBatchOutput, ApoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ApoError::InvalidPeriod { short: 0, long: 0 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ApoError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(ApoError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let s = combos[row].short_period.unwrap();
        let l = combos[row].long_period.unwrap();
        match kern {
            Kernel::Scalar => apo_row_scalar(data, first, s, l, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => apo_row_avx2(data, first, s, l, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => apo_row_avx512(data, first, s, l, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(ApoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// --- Row Kernels

#[inline(always)]
pub unsafe fn apo_row_scalar(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    apo_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn apo_row_avx2(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
    apo_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn apo_row_avx512(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    if long <= 32 {
        apo_row_avx512_short(data, first, short, long, out)
    } else {
        apo_row_avx512_long(data, first, short, long, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn apo_row_avx512_short(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    apo_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn apo_row_avx512_long(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    apo_scalar(data, short, long, first, out)
}

// --- Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_apo_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ApoParams {
            short_period: None,
            long_period: None,
        };
        let input = ApoInput::from_candles(&candles, "close", default_params);
        let output = apo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_apo_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ApoInput::with_default_candles(&candles);
        let result = apo_with_kernel(&input, kernel)?;
        let expected_last_five = [-429.8, -401.6, -386.1, -357.9, -374.1];
        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "[{}] APO value mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_last_five[i],
                value
            );
        }
        Ok(())
    }

    fn check_apo_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ApoInput::with_default_candles(&candles);
        match input.data {
            ApoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ApoData::Candles"),
        }
        let output = apo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_apo_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ApoParams {
            short_period: Some(0),
            long_period: Some(20),
        };
        let input = ApoInput::from_slice(&input_data, params);
        let res = apo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] APO should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_apo_period_invalid(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ApoParams {
            short_period: Some(20),
            long_period: Some(10),
        };
        let input = ApoInput::from_slice(&data_small, params);
        let res = apo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] APO should fail with invalid period",
            test_name
        );
        Ok(())
    }

    fn check_apo_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ApoParams {
            short_period: Some(9),
            long_period: Some(10),
        };
        let input = ApoInput::from_slice(&single_point, params);
        let res = apo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] APO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_apo_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ApoParams {
            short_period: Some(10),
            long_period: Some(20),
        };
        let first_input = ApoInput::from_candles(&candles, "close", first_params);
        let first_result = apo_with_kernel(&first_input, kernel)?;
        let second_params = ApoParams {
            short_period: Some(5),
            long_period: Some(15),
        };
        let second_input = ApoInput::from_slice(&first_result.values, second_params);
        let second_result = apo_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_apo_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ApoInput::from_candles(
            &candles,
            "close",
            ApoParams {
                short_period: Some(10),
                long_period: Some(20),
            },
        );
        let res = apo_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 30 {
            for (i, &val) in res.values[30..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    30 + i
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_apo_tests {
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

    generate_all_apo_tests!(
        check_apo_partial_params,
        check_apo_accuracy,
        check_apo_default_candles,
        check_apo_zero_period,
        check_apo_period_invalid,
        check_apo_very_small_dataset,
        check_apo_reinput,
        check_apo_nan_handling
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ApoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = ApoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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
