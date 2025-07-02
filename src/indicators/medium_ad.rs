//! # Median Absolute Deviation (MEDIUM_AD)
//!
//! A robust measure of dispersion that calculates the median of the absolute
//! deviations from the median for a specified `period`.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 5.
//!
//! ## Errors
//! - **AllValuesNaN**: medium_ad: All input data values are `NaN`.
//! - **EmptyData**: medium_ad: Input data slice is empty.
//! - **InvalidPeriod**: medium_ad: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: medium_ad: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(MediumAdOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(MediumAdError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;

impl<'a> AsRef<[f64]> for MediumAdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MediumAdData::Slice(slice) => slice,
            MediumAdData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MediumAdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MediumAdOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MediumAdParams {
    pub period: Option<usize>,
}

impl Default for MediumAdParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MediumAdInput<'a> {
    pub data: MediumAdData<'a>,
    pub params: MediumAdParams,
}

impl<'a> MediumAdInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MediumAdParams) -> Self {
        Self {
            data: MediumAdData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MediumAdParams) -> Self {
        Self {
            data: MediumAdData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MediumAdParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MediumAdBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MediumAdBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MediumAdBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MediumAdOutput, MediumAdError> {
        let p = MediumAdParams { period: self.period };
        let i = MediumAdInput::from_candles(c, "close", p);
        medium_ad_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MediumAdOutput, MediumAdError> {
        let p = MediumAdParams { period: self.period };
        let i = MediumAdInput::from_slice(d, p);
        medium_ad_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<MediumAdStream, MediumAdError> {
        let p = MediumAdParams { period: self.period };
        MediumAdStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MediumAdError {
    #[error("medium_ad: All values are NaN.")]
    AllValuesNaN,
    #[error("medium_ad: Empty data provided for MEDIUM_AD.")]
    EmptyData,
    #[error("medium_ad: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("medium_ad: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn medium_ad(input: &MediumAdInput) -> Result<MediumAdOutput, MediumAdError> {
    medium_ad_with_kernel(input, Kernel::Auto)
}

pub fn medium_ad_with_kernel(input: &MediumAdInput, kernel: Kernel) -> Result<MediumAdOutput, MediumAdError> {
    let data: &[f64] = match &input.data {
        MediumAdData::Candles { candles, source } => source_type(candles, source),
        MediumAdData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(MediumAdError::EmptyData);
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(MediumAdError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(MediumAdError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(MediumAdError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                medium_ad_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                medium_ad_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                medium_ad_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(MediumAdOutput { values: out })
}

#[inline]
pub fn medium_ad_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    for i in (first_valid + period - 1)..data.len() {
        let window = &data[i + 1 - period..=i];
        if window.iter().any(|&v| v.is_nan()) {
            out[i] = f64::NAN;
            continue;
        }
        let mut sorted_window: Vec<f64> = window.to_vec();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_val = if period % 2 == 1 {
            sorted_window[period / 2]
        
            } else {
            0.5 * (sorted_window[period / 2 - 1] + sorted_window[period / 2])
        };
        let mut abs_devs: Vec<f64> = sorted_window.iter().map(|&v| (v - median_val).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if period % 2 == 1 {
            abs_devs[period / 2]
        
            } else {
            0.5 * (abs_devs[period / 2 - 1] + abs_devs[period / 2])
        };
        out[i] = mad;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[inline(always)]
pub fn medium_ad_batch_with_kernel(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    k: Kernel,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(MediumAdError::InvalidPeriod { period: 0, data_len: 0 }),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    medium_ad_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MediumAdBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MediumAdBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 50, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MediumAdBatchBuilder {
    range: MediumAdBatchRange,
    kernel: Kernel,
}

impl MediumAdBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<MediumAdBatchOutput, MediumAdError> {
        medium_ad_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MediumAdBatchOutput, MediumAdError> {
        MediumAdBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MediumAdBatchOutput, MediumAdError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<MediumAdBatchOutput, MediumAdError> {
        MediumAdBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MediumAdBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MediumAdParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MediumAdBatchOutput {
    pub fn row_for_params(&self, p: &MediumAdParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
        })
    }

    pub fn values_for(&self, p: &MediumAdParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MediumAdBatchRange) -> Vec<MediumAdParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(MediumAdParams {
            period: Some(p),
        });
    }
    out
}

#[inline(always)]
pub fn medium_ad_batch_slice(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    kern: Kernel,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    medium_ad_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn medium_ad_batch_par_slice(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    kern: Kernel,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    medium_ad_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn medium_ad_batch_inner(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MediumAdError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(MediumAdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MediumAdError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => medium_ad_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => medium_ad_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => medium_ad_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {


        #[cfg(not(target_arch = "wasm32"))] {


        values


                    .par_chunks_mut(cols)


                    .enumerate()


                    .for_each(|(row, slice)| do_row(row, slice));


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
    Ok(MediumAdBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn medium_ad_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    for i in (first + period - 1)..data.len() {
        let window = &data[i + 1 - period..=i];
        if window.iter().any(|&v| v.is_nan()) {
            out[i] = f64::NAN;
            continue;
        }
        let mut sorted_window: Vec<f64> = window.to_vec();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_val = if period % 2 == 1 {
            sorted_window[period / 2]
        
            } else {
            0.5 * (sorted_window[period / 2 - 1] + sorted_window[period / 2])
        };
        let mut abs_devs: Vec<f64> = sorted_window.iter().map(|&v| (v - median_val).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if period % 2 == 1 {
            abs_devs[period / 2]
        
            } else {
            0.5 * (abs_devs[period / 2 - 1] + abs_devs[period / 2])
        };
        out[i] = mad;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    medium_ad_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        medium_ad_row_avx512_short(data, first, period, out)
    
        } else {
        medium_ad_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    medium_ad_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    medium_ad_row_scalar(data, first, period, out)
}

#[derive(Debug, Clone)]
pub struct MediumAdStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl MediumAdStream {
    pub fn try_new(params: MediumAdParams) -> Result<Self, MediumAdError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(MediumAdError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled || self.buffer.iter().any(|v| v.is_nan()) {
            return None;
        }
        Some(self.compute())
    }

    #[inline(always)]
    fn compute(&self) -> f64 {
        let mut sorted = self.buffer.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if self.period % 2 == 1 {
            sorted[self.period / 2]
        
            } else {
            0.5 * (sorted[self.period / 2 - 1] + sorted[self.period / 2])
        };
        let mut absdevs: Vec<f64> = sorted.iter().map(|&x| (x - median).abs()).collect();
        absdevs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if self.period % 2 == 1 {
            absdevs[self.period / 2]
        
            } else {
            0.5 * (absdevs[self.period / 2 - 1] + absdevs[self.period / 2])
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_medium_ad_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = MediumAdParams { period: None };
        let input = MediumAdInput::from_candles(&candles, "close", default_params);
        let output = medium_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_medium_ad_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = MediumAdParams { period: Some(5) };
        let input = MediumAdInput::from_candles(&candles, "hl2", params);
        let result = medium_ad_with_kernel(&input, kernel)?;
        let expected_last_five = [220.0, 78.5, 126.5, 48.0, 28.5];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MEDIUM_AD {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_medium_ad_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MediumAdInput::with_default_candles(&candles);
        match input.data {
            MediumAdData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MediumAdData::Candles"),
        }
        let output = medium_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_medium_ad_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MediumAdParams { period: Some(0) };
        let input = MediumAdInput::from_slice(&input_data, params);
        let res = medium_ad_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MEDIUM_AD should fail with zero period", test_name);
        Ok(())
    }

    fn check_medium_ad_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MediumAdParams { period: Some(10) };
        let input = MediumAdInput::from_slice(&data_small, params);
        let res = medium_ad_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MEDIUM_AD should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_medium_ad_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MediumAdParams { period: Some(5) };
        let input = MediumAdInput::from_slice(&single_point, params);
        let res = medium_ad_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MEDIUM_AD should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_medium_ad_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = MediumAdParams { period: Some(5) };
        let first_input = MediumAdInput::from_candles(&candles, "close", first_params);
        let first_result = medium_ad_with_kernel(&first_input, kernel)?;

        let second_params = MediumAdParams { period: Some(5) };
        let second_input = MediumAdInput::from_slice(&first_result.values, second_params);
        let second_result = medium_ad_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_medium_ad_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MediumAdInput::from_candles(&candles, "close", MediumAdParams { period: Some(5) });
        let res = medium_ad_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 60 {
            for (i, &val) in res.values[60..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    60 + i
                );
            }
        }
        Ok(())
    }

    fn check_medium_ad_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;
        let input = MediumAdInput::from_candles(
            &candles,
            "close",
            MediumAdParams { period: Some(period) },
        );
        let batch_output = medium_ad_with_kernel(&input, kernel)?.values;

        let mut stream = MediumAdStream::try_new(MediumAdParams { period: Some(period) })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(mad_val) => stream_values.push(mad_val),
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
                diff < 1e-9,
                "[{}] MEDIUM_AD streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_medium_ad_tests {
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

    generate_all_medium_ad_tests!(
        check_medium_ad_partial_params,
        check_medium_ad_accuracy,
        check_medium_ad_default_candles,
        check_medium_ad_zero_period,
        check_medium_ad_period_exceeds_length,
        check_medium_ad_very_small_dataset,
        check_medium_ad_reinput,
        check_medium_ad_nan_handling,
        check_medium_ad_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MediumAdBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = MediumAdParams::default();
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
