//! # Rate of Change (ROC)
//!
//! Measures the percentage change in price between the current value and the value `period` bars ago.
//! Implements kernel auto-detection and AVX2/AVX512 stubs for SIMD compatibility, with a fully
//! featured builder/batch/stream API and input validation parity with alma.rs.
//!
//! ## Parameters
//! - **period**: Lookback window (defaults to 9)
//!
//! ## Errors
//! - **EmptyData**: roc: Input data slice is empty.
//! - **InvalidPeriod**: roc: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: roc: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
//! - **AllValuesNaN**: roc: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(RocOutput)`** on success, containing a `Vec<f64>` matching the input length
//! - **`Err(RocError)`** otherwise

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
use std::convert::AsRef;

// --- Data Types ---

#[derive(Debug, Clone)]
pub enum RocData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RocParams {
    pub period: Option<usize>,
}

impl Default for RocParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct RocInput<'a> {
    pub data: RocData<'a>,
    pub params: RocParams,
}

impl<'a> AsRef<[f64]> for RocInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RocData::Slice(slice) => slice,
            RocData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> RocInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RocParams) -> Self {
        Self {
            data: RocData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slice(slice: &'a [f64], params: RocParams) -> Self {
        Self {
            data: RocData::Slice(slice),
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", RocParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

// --- Builder/Stream/Batch Structs ---

#[derive(Copy, Clone, Debug)]
pub struct RocBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for RocBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RocBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<RocOutput, RocError> {
        let p = RocParams { period: self.period };
        let i = RocInput::from_candles(c, "close", p);
        roc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<RocOutput, RocError> {
        let p = RocParams { period: self.period };
        let i = RocInput::from_slice(d, p);
        roc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<RocStream, RocError> {
        let p = RocParams { period: self.period };
        RocStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum RocError {
    #[error("roc: Empty data provided.")]
    EmptyData,
    #[error("roc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("roc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("roc: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn roc(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Auto)
}

pub fn roc_with_kernel(input: &RocInput, kernel: Kernel) -> Result<RocOutput, RocError> {
    let data: &[f64] = match &input.data {
        RocData::Candles { candles, source } => source_type(candles, source),
        RocData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(RocError::EmptyData);
    }

    let period = input.get_period();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(RocError::AllValuesNaN)?;
    if period == 0 || period > data.len() {
        return Err(RocError::InvalidPeriod { period, data_len: data.len() });
    }
    if (data.len() - first) < period {
        return Err(RocError::NotEnoughValidData { needed: period, valid: data.len() - first });
    }

    let mut out = vec![f64::NAN; data.len()];
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                roc_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                roc_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                roc_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(RocOutput { values: out })
}

// --- Indicator Functions ---

#[inline(always)]
pub unsafe fn roc_indicator(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Auto)
}
#[inline(always)]
pub unsafe fn roc_indicator_with_kernel(input: &RocInput, k: Kernel) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, k)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx512(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx512)
}
#[inline(always)]
pub unsafe fn roc_indicator_scalar(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Scalar)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx2(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx2)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx512_short(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx512)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_indicator_avx512_long(input: &RocInput) -> Result<RocOutput, RocError> {
    roc_with_kernel(input, Kernel::Avx512)
}

// --- Core Scalar & SIMD ---

#[inline]
pub fn roc_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let start = first + period;
    for i in start..len {
        let curr = data[i];
        let prev = data[i - period];
        if prev == 0.0 || prev.is_nan() {
            out[i] = 0.0;
        
            } else {
            out[i] = ((curr / prev) - 1.0) * 100.0;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn roc_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn roc_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn roc_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn roc_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    roc_scalar(data, period, first, out)
}

// --- Row/Batch Parity ---

#[inline(always)]
pub unsafe fn roc_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn roc_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _weights: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    roc_scalar(data, period, first, out)
}

// --- Stream API ---

#[derive(Debug, Clone)]
pub struct RocStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl RocStream {
    pub fn try_new(params: RocParams) -> Result<Self, RocError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(RocError::InvalidPeriod { period, data_len: 0 });
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
        let prev = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled || prev.is_nan() {
            None
        } else if prev == 0.0 {
            Some(0.0)
        
            } else {
            Some(((value / prev) - 1.0) * 100.0)
        }
    }
}

// --- Batch API ---

#[derive(Clone, Debug)]
pub struct RocBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for RocBatchRange {
    fn default() -> Self {
        Self { period: (9, 240, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RocBatchBuilder {
    range: RocBatchRange,
    kernel: Kernel,
}

impl RocBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<RocBatchOutput, RocError> {
        roc_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RocBatchOutput, RocError> {
        RocBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RocBatchOutput, RocError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<RocBatchOutput, RocError> {
        RocBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn roc_batch_with_kernel(
    data: &[f64],
    sweep: &RocBatchRange,
    k: Kernel,
) -> Result<RocBatchOutput, RocError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(RocError::InvalidPeriod { period: 0, data_len: 0 }),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    roc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RocBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<RocParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RocBatchOutput {
    pub fn row_for_params(&self, p: &RocParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
    }
    pub fn values_for(&self, p: &RocParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &RocBatchRange) -> Vec<RocParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![RocParams { period: Some(start) }];
    }
    (start..=end)
        .step_by(step)
        .map(|p| RocParams { period: Some(p) })
        .collect()
}

#[inline(always)]
pub fn roc_batch_slice(
    data: &[f64],
    sweep: &RocBatchRange,
    kern: Kernel,
) -> Result<RocBatchOutput, RocError> {
    roc_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn roc_batch_par_slice(
    data: &[f64],
    sweep: &RocBatchRange,
    kern: Kernel,
) -> Result<RocBatchOutput, RocError> {
    roc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn roc_batch_inner(
    data: &[f64],
    sweep: &RocBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RocBatchOutput, RocError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RocError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(RocError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(RocError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => roc_row_scalar(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => roc_row_avx2(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => roc_row_avx512(data, first, period, 0, std::ptr::null(), 0.0, out_row),
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
    Ok(RocBatchOutput { values, combos, rows, cols })
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_roc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = RocParams { period: None };
        let input_default = RocInput::from_candles(&candles, "close", default_params);
        let output_default = roc_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = RocParams { period: Some(14) };
        let input_period_14 = RocInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = roc_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = RocParams { period: Some(20) };
        let input_custom = RocInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = roc_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());

        Ok(())
    }

    fn check_roc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let close_prices = &candles.close;
        let params = RocParams { period: Some(10) };
        let input = RocInput::from_candles(&candles, "close", params);
        let roc_result = roc_with_kernel(&input, kernel)?;

        assert_eq!(roc_result.values.len(), close_prices.len());

        let expected_last_five_roc = [
            -0.22551709049294377,
            -0.5561903481650754,
            -0.32752013235864963,
            -0.49454153980722504,
            -1.5045927020536976,
        ];
        assert!(roc_result.values.len() >= 5);
        let start_index = roc_result.values.len() - 5;
        let result_last_five_roc = &roc_result.values[start_index..];
        for (i, &value) in result_last_five_roc.iter().enumerate() {
            let expected_value = expected_last_five_roc[i];
            assert!(
                (value - expected_value).abs() < 1e-7,
                "[{}] ROC mismatch at index {}: expected {}, got {}",
                test_name, i, expected_value, value
            );
        }
        let period = input.get_period();
        for i in 0..(period - 1) {
            assert!(roc_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_roc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = RocInput::with_default_candles(&candles);
        match input.data {
            RocData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected RocData::Candles"),
        }
        let output = roc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_roc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = RocParams { period: Some(0) };
        let input = RocInput::from_slice(&input_data, params);
        let res = roc_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_roc_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = RocParams { period: Some(10) };
        let input = RocInput::from_slice(&data_small, params);
        let res = roc_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_roc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = RocParams { period: Some(9) };
        let input = RocInput::from_slice(&single_point, params);
        let res = roc_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_roc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = RocParams { period: Some(14) };
        let first_input = RocInput::from_candles(&candles, "close", first_params);
        let first_result = roc_with_kernel(&first_input, kernel)?;

        let second_params = RocParams { period: Some(14) };
        let second_input = RocInput::from_slice(&first_result.values, second_params);
        let second_result = roc_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index 28, found NaN at {}",
                test_name, i
            );
        }
        Ok(())
    }

    fn check_roc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = RocInput::from_candles(&candles, "close", RocParams { period: Some(9) });
        let res = roc_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_roc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 9;

        let input = RocInput::from_candles(&candles, "close", RocParams { period: Some(period) });
        let batch_output = roc_with_kernel(&input, kernel)?.values;

        let mut stream = RocStream::try_new(RocParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(diff < 1e-9,
                "[{}] ROC streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_roc_tests {
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
    generate_all_roc_tests!(
        check_roc_partial_params,
        check_roc_accuracy,
        check_roc_default_candles,
        check_roc_zero_period,
        check_roc_period_exceeds_length,
        check_roc_very_small_dataset,
        check_roc_reinput,
        check_roc_nan_handling,
        check_roc_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = RocBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = RocParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -0.22551709049294377,
            -0.5561903481650754,
            -0.32752013235864963,
            -0.49454153980722504,
            -1.5045927020536976,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
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
