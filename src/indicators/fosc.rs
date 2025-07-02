//! # Forecast Oscillator (FOSC)
//!
//! The Forecast Oscillator (FOSC) measures the percentage difference between the current price
//! and a one-step-ahead linear regression forecast. Positive values suggest the price is above
//! its trend, while negative values indicate it's below. Parameters and errors match ALMA's API
//! conventions for batch processing, builder usage, SIMD kernel dispatch, and streaming updates.
//!
//! ## Parameters
//! - **period**: Regression window (default: 5).
//!
//! ## Errors
//! - **AllValuesNaN**: fosc: All input data values are `NaN`.
//! - **InvalidPeriod**: fosc: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: fosc: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(FoscOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(FoscError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

// --------- Input Data & Param Structs ---------

impl<'a> AsRef<[f64]> for FoscInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            FoscData::Slice(slice) => slice,
            FoscData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FoscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FoscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FoscParams {
    pub period: Option<usize>,
}

impl Default for FoscParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct FoscInput<'a> {
    pub data: FoscData<'a>,
    pub params: FoscParams,
}

impl<'a> FoscInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: FoscParams) -> Self {
        Self {
            data: FoscData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: FoscParams) -> Self {
        Self {
            data: FoscData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", FoscParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

// --------- Builder Struct ---------

#[derive(Copy, Clone, Debug)]
pub struct FoscBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for FoscBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FoscBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
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
    pub fn apply(self, c: &Candles) -> Result<FoscOutput, FoscError> {
        let p = FoscParams { period: self.period };
        let i = FoscInput::from_candles(c, "close", p);
        fosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<FoscOutput, FoscError> {
        let p = FoscParams { period: self.period };
        let i = FoscInput::from_slice(d, p);
        fosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<FoscStream, FoscError> {
        let p = FoscParams { period: self.period };
        FoscStream::try_new(p)
    }
}

// --------- Error ---------

#[derive(Debug, Error)]
pub enum FoscError {
    #[error("fosc: All values are NaN.")]
    AllValuesNaN,
    #[error("fosc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// --------- Main Dispatchers ---------

#[inline]
pub fn fosc(input: &FoscInput) -> Result<FoscOutput, FoscError> {
    fosc_with_kernel(input, Kernel::Auto)
}

pub fn fosc_with_kernel(input: &FoscInput, kernel: Kernel) -> Result<FoscOutput, FoscError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let period = input.get_period();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(FoscError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(FoscError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(FoscError::NotEnoughValidData { needed: period, valid: len - first });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                fosc_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                fosc_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                fosc_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(FoscOutput { values: out })
}

// --------- Scalar Core ---------

#[inline]
pub fn fosc_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let n = data.len();
    if n < period { return; }
    let mut x = 0.0;
    let mut x2 = 0.0;
    let mut y = 0.0;
    let mut xy = 0.0;
    let p = 1.0 / (period as f64);
    let mut tsf = 0.0;

    for i in 0..(period - 1) {
        x += (i + 1) as f64;
        x2 += ((i + 1) as f64) * ((i + 1) as f64);
        xy += data[i] * ((i + 1) as f64);
        y += data[i];
    }
    x += period as f64;
    x2 += (period as f64) * (period as f64);

    let denom = (period as f64) * x2 - x * x;
    let bd = if denom.abs() < f64::EPSILON { 0.0 } else { 1.0 / denom };

    for i in (first + period - 1)..n {
        xy += data[i] * (period as f64);
        y += data[i];
        let b = (period as f64 * xy - x * y) * bd;
        let a = (y - b * x) * p;
        if !data[i].is_nan() && data[i] != 0.0 {
            out[i] = 100.0 * (data[i] - tsf) / data[i];
        
            } else {
            out[i] = f64::NAN;
        }
        tsf = a + b * ((period + 1) as f64);
        xy -= y;
        let old_idx = i as isize - (period as isize) + 1;
        y -= data[old_idx as usize];
    }
}

// --------- AVX2/AVX512 Routing (Stubs) ---------

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // AVX2 not implemented, fallback to scalar
    fosc_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // AVX512 not implemented, fallback to scalar
    if period <= 32 {
        fosc_avx512_short(data, period, first, out)
    
        } else {
        fosc_avx512_long(data, period, first, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    fosc_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    fosc_scalar(data, period, first, out)
}

// --------- Streaming (Stateful) ---------

#[derive(Debug, Clone)]
pub struct FoscStream {
    period: usize,
    buffer: Vec<f64>,
    idx: usize,
    filled: bool,
    x: f64,
    x2: f64,
    y: f64,
    xy: f64,
    tsf: f64,
}

impl FoscStream {
    pub fn try_new(params: FoscParams) -> Result<Self, FoscError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(FoscError::InvalidPeriod { period, data_len: 0 });
        }
        let mut x = 0.0;
        let mut x2 = 0.0;
        for i in 0..period {
            let xi = (i + 1) as f64;
            x += xi;
            x2 += xi * xi;
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            idx: 0,
            filled: false,
            x,
            x2,
            y: 0.0,
            xy: 0.0,
            tsf: 0.0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.idx < self.period {
            self.buffer[self.idx] = value;
            self.idx += 1;
            if self.idx == self.period {
                self.filled = true;
            }
            return None;
        }
        // Simple O(period) recompute (can be optimized, but match scalar logic)
        for i in 0..self.period {
            self.buffer[i] = if i == self.period - 1 { value } else { self.buffer[i + 1] };
        }
        if !self.filled { return None; }
        let mut x = 0.0;
        let mut x2 = 0.0;
        let mut y = 0.0;
        let mut xy = 0.0;
        for i in 0..self.period {
            let xi = (i + 1) as f64;
            let v = self.buffer[i];
            x += xi;
            x2 += xi * xi;
            y += v;
            xy += v * xi;
        }
        let p = 1.0 / (self.period as f64);
        let denom = (self.period as f64) * x2 - x * x;
        let bd = if denom.abs() < f64::EPSILON { 0.0 } else { 1.0 / denom };
        let b = (self.period as f64 * xy - x * y) * bd;
        let a = (y - b * x) * p;
        let tsf = a + b * ((self.period + 1) as f64);
        let v = self.buffer[self.period - 1];
        let out = if !v.is_nan() && v != 0.0 { 100.0 * (v - tsf) / v } else { f64::NAN };
        Some(out)
    }
}

// --------- Batch/Parameter Grid ---------

#[derive(Clone, Debug)]
pub struct FoscBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for FoscBatchRange {
    fn default() -> Self {
        Self { period: (5, 50, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FoscBatchBuilder {
    range: FoscBatchRange,
    kernel: Kernel,
}

impl FoscBatchBuilder {
    pub fn new() -> Self { Self::default() }
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
    pub fn apply_slice(self, data: &[f64]) -> Result<FoscBatchOutput, FoscError> {
        fosc_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<FoscBatchOutput, FoscError> {
        FoscBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<FoscBatchOutput, FoscError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FoscBatchOutput, FoscError> {
        FoscBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct FoscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FoscParams>,
    pub rows: usize,
    pub cols: usize,
}

impl FoscBatchOutput {
    pub fn row_for_params(&self, p: &FoscParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &FoscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &FoscBatchRange) -> Vec<FoscParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(FoscParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn fosc_batch_slice(
    data: &[f64],
    sweep: &FoscBatchRange,
    kern: Kernel,
) -> Result<FoscBatchOutput, FoscError> {
    fosc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn fosc_batch_par_slice(
    data: &[f64],
    sweep: &FoscBatchRange,
    kern: Kernel,
) -> Result<FoscBatchOutput, FoscError> {
    fosc_batch_inner(data, sweep, kern, true)
}

pub fn fosc_batch_with_kernel(
    data: &[f64],
    sweep: &FoscBatchRange,
    k: Kernel,
) -> Result<FoscBatchOutput, FoscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(FoscError::InvalidPeriod { period: 0, data_len: 0 })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    fosc_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn fosc_batch_inner(
    data: &[f64],
    sweep: &FoscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FoscBatchOutput, FoscError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(FoscError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(FoscError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(FoscError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => fosc_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => fosc_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => fosc_row_avx512(data, first, period, out_row),
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


        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(FoscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn fosc_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    fosc_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    fosc_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        fosc_row_avx512_short(data, first, period, out)
    
        } else {
        fosc_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    fosc_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    fosc_avx512_long(data, period, first, out)
}

#[inline(always)]
fn expand_grid_fosc(r: &FoscBatchRange) -> Vec<FoscParams> {
    expand_grid(r)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_fosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = FoscParams { period: None };
        let input = FoscInput::from_candles(&candles, "close", default_params);
        let output = fosc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_fosc_basic_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let test_data = [
            81.59, 81.06, 82.87, 83.00, 83.61, 83.15, 82.84, 82.84, 83.99, 84.55, 84.36, 85.53,
        ];
        let period = 5;
        let input = FoscInput::from_slice(
            &test_data,
            FoscParams { period: Some(period) },
        );
        let result = fosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), test_data.len());
        for i in 0..(period - 1) {
            assert!(result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_fosc_with_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        let params = FoscParams { period: Some(3) };
        let input = FoscInput::from_slice(&input_data, params);
        let result = fosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), input_data.len());
        Ok(())
    }

    fn check_fosc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = FoscParams { period: Some(0) };
        let input = FoscInput::from_slice(&input_data, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] FOSC should fail with zero period", test_name);
        Ok(())
    }

    fn check_fosc_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = FoscParams { period: Some(10) };
        let input = FoscInput::from_slice(&data_small, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] FOSC should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_fosc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = FoscParams { period: Some(5) };
        let input = FoscInput::from_slice(&single_point, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] FOSC should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_fosc_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = FoscParams { period: Some(2) };
        let input = FoscInput::from_slice(&input_data, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] FOSC should fail with all NaN", test_name);
        Ok(())
    }

    fn check_fosc_expected_values_reference(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let expected_last_five = [
            -0.8904444627923475,
            -0.4763353099245297,
            -0.2379782851444668,
            0.292790128025632,
            -0.6597902988090389,
        ];
        let params = FoscParams { period: Some(5) };
        let input = FoscInput::from_candles(&candles, "close", params);
        let result = fosc_with_kernel(&input, kernel)?;
        let valid_len = result.values.len();
        assert!(valid_len >= 5);
        let output_slice = &result.values[valid_len - 5..valid_len];
        for (i, &val) in output_slice.iter().enumerate() {
            let exp: f64 = expected_last_five[i];
            if exp.is_nan() {
                assert!(val.is_nan());
            
                } else {
                assert!(
                    (val - exp).abs() < 1e-7,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    exp,
                    val
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_fosc_tests {
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

    generate_all_fosc_tests!(
        check_fosc_partial_params,
        check_fosc_basic_accuracy,
        check_fosc_with_nan_data,
        check_fosc_zero_period,
        check_fosc_period_exceeds_length,
        check_fosc_very_small_dataset,
        check_fosc_all_values_nan,
        check_fosc_expected_values_reference
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = FoscBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = FoscParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -0.8904444627923475,
            -0.4763353099245297,
            -0.2379782851444668,
            0.292790128025632,
            -0.6597902988090389,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
