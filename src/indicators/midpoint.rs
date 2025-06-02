//! # Midpoint Indicator
//!
//! Calculates the midpoint of the highest and lowest value over a given window (`period`).
//! Returns a vector matching the input size, with leading NaNs for incomplete windows.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: midpoint: All input data values are `NaN`.
//! - **InvalidPeriod**: midpoint: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: midpoint: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(MidpointOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(MidpointError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// --- INPUT/OUTPUT TYPES ---

#[derive(Debug, Clone)]
pub enum MidpointData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for MidpointInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MidpointData::Slice(slice) => slice,
            MidpointData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MidpointOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MidpointParams {
    pub period: Option<usize>,
}

impl Default for MidpointParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct MidpointInput<'a> {
    pub data: MidpointData<'a>,
    pub params: MidpointParams,
}

impl<'a> MidpointInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MidpointParams) -> Self {
        Self {
            data: MidpointData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MidpointParams) -> Self {
        Self {
            data: MidpointData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MidpointParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// --- BUILDER ---

#[derive(Copy, Clone, Debug)]
pub struct MidpointBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MidpointBuilder {
    fn default() -> Self {
        Self { period: None, kernel: Kernel::Auto }
    }
}

impl MidpointBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MidpointOutput, MidpointError> {
        let p = MidpointParams { period: self.period };
        let i = MidpointInput::from_candles(c, "close", p);
        midpoint_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MidpointOutput, MidpointError> {
        let p = MidpointParams { period: self.period };
        let i = MidpointInput::from_slice(d, p);
        midpoint_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MidpointStream, MidpointError> {
        let p = MidpointParams { period: self.period };
        MidpointStream::try_new(p)
    }
}

// --- ERROR ---

#[derive(Debug, Error)]
pub enum MidpointError {
    #[error("midpoint: All values are NaN.")]
    AllValuesNaN,
    #[error("midpoint: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("midpoint: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// --- INDICATOR API ---

#[inline]
pub fn midpoint(input: &MidpointInput) -> Result<MidpointOutput, MidpointError> {
    midpoint_with_kernel(input, Kernel::Auto)
}

pub fn midpoint_with_kernel(input: &MidpointInput, kernel: Kernel) -> Result<MidpointOutput, MidpointError> {
    let data: &[f64] = input.as_ref();

    let first = data.iter().position(|x| !x.is_nan()).ok_or(MidpointError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(MidpointError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(MidpointError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => midpoint_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => midpoint_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => midpoint_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(MidpointOutput { values: out })
}

// --- SIMD STUBS ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe { midpoint_avx512_long(data, period, first, out) }
}

#[inline]
pub fn midpoint_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    for i in (first + period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let mut highest = f64::MIN;
        let mut lowest = f64::MAX;
        for &val in window {
            if val > highest { highest = val; }
            if val < lowest { lowest = val; }
        }
        out[i] = (highest + lowest) / 2.0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}

// --- BATCH / STREAMING ---

#[derive(Debug, Clone)]
pub struct MidpointStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl MidpointStream {
    pub fn try_new(params: MidpointParams) -> Result<Self, MidpointError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(MidpointError::InvalidPeriod { period, data_len: 0 });
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
        if !self.filled && self.head == 0 { self.filled = true; }
        if !self.filled { return None; }
        Some(self.calc_midpoint())
    }
    #[inline(always)]
    fn calc_midpoint(&self) -> f64 {
        let mut highest = f64::MIN;
        let mut lowest = f64::MAX;
        let mut idx = self.head;
        for _ in 0..self.period {
            let v = self.buffer[idx];
            if v > highest { highest = v; }
            if v < lowest { lowest = v; }
            idx = (idx + 1) % self.period;
        }
        (highest + lowest) / 2.0
    }
}

// --- BATCH API (Parameter Sweep) ---

#[derive(Clone, Debug)]
pub struct MidpointBatchRange {
    pub period: (usize, usize, usize),
}
impl Default for MidpointBatchRange {
    fn default() -> Self {
        Self { period: (14, 14, 0) }
    }
}
#[derive(Clone, Debug, Default)]
pub struct MidpointBatchBuilder {
    range: MidpointBatchRange,
    kernel: Kernel,
}
impl MidpointBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<MidpointBatchOutput, MidpointError> {
        midpoint_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MidpointBatchOutput, MidpointError> {
        MidpointBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MidpointBatchOutput, MidpointError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MidpointBatchOutput, MidpointError> {
        MidpointBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn midpoint_batch_with_kernel(
    data: &[f64],
    sweep: &MidpointBatchRange,
    k: Kernel,
) -> Result<MidpointBatchOutput, MidpointError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(MidpointError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    midpoint_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MidpointBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MidpointParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MidpointBatchOutput {
    pub fn row_for_params(&self, p: &MidpointParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &MidpointParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// --- BATCH INNER ---

#[inline(always)]
fn expand_grid(r: &MidpointBatchRange) -> Vec<MidpointParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(MidpointParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn midpoint_batch_slice(
    data: &[f64],
    sweep: &MidpointBatchRange,
    kern: Kernel,
) -> Result<MidpointBatchOutput, MidpointError> {
    midpoint_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn midpoint_batch_par_slice(
    data: &[f64],
    sweep: &MidpointBatchRange,
    kern: Kernel,
) -> Result<MidpointBatchOutput, MidpointError> {
    midpoint_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn midpoint_batch_inner(
    data: &[f64],
    sweep: &MidpointBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MidpointBatchOutput, MidpointError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MidpointError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(MidpointError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MidpointError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => midpoint_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => midpoint_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => midpoint_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        values.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() { do_row(row, slice); }
    }
    Ok(MidpointBatchOutput { values, combos, rows, cols })
}

// --- ROW KERNELS ---

#[inline(always)]
unsafe fn midpoint_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    midpoint_scalar(data, period, first, out)
}

// --- TESTS ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_midpoint_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MidpointParams { period: None };
        let input = MidpointInput::from_candles(&candles, "close", default_params);
        let output = midpoint_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_midpoint_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MidpointInput::from_candles(&candles, "close", MidpointParams::default());
        let result = midpoint_with_kernel(&input, kernel)?;
        let expected_last_five = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MIDPOINT {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_midpoint_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MidpointInput::with_default_candles(&candles);
        match input.data {
            MidpointData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MidpointData::Candles"),
        }
        let output = midpoint_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_midpoint_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MidpointParams { period: Some(0) };
        let input = MidpointInput::from_slice(&input_data, params);
        let res = midpoint_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MIDPOINT should fail with zero period", test_name);
        Ok(())
    }
    fn check_midpoint_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MidpointParams { period: Some(10) };
        let input = MidpointInput::from_slice(&data_small, params);
        let res = midpoint_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MIDPOINT should fail with period exceeding length", test_name);
        Ok(())
    }
    fn check_midpoint_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MidpointParams { period: Some(9) };
        let input = MidpointInput::from_slice(&single_point, params);
        let res = midpoint_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MIDPOINT should fail with insufficient data", test_name);
        Ok(())
    }
    fn check_midpoint_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = MidpointParams { period: Some(14) };
        let first_input = MidpointInput::from_candles(&candles, "close", first_params);
        let first_result = midpoint_with_kernel(&first_input, kernel)?;
        let second_params = MidpointParams { period: Some(14) };
        let second_input = MidpointInput::from_slice(&first_result.values, second_params);
        let second_result = midpoint_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_midpoint_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MidpointInput::from_candles(&candles, "close", MidpointParams::default());
        let res = midpoint_with_kernel(&input, kernel)?;
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
    fn check_midpoint_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let input = MidpointInput::from_candles(&candles, "close", MidpointParams { period: Some(period) });
        let batch_output = midpoint_with_kernel(&input, kernel)?.values;
        let mut stream = MidpointStream::try_new(MidpointParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(mid_val) => stream_values.push(mid_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] MIDPOINT streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_midpoint_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                   #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }
    generate_all_midpoint_tests!(
        check_midpoint_partial_params,
        check_midpoint_accuracy,
        check_midpoint_default_candles,
        check_midpoint_zero_period,
        check_midpoint_period_exceeds_length,
        check_midpoint_very_small_dataset,
        check_midpoint_reinput,
        check_midpoint_nan_handling,
        check_midpoint_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MidpointBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = MidpointParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-1, "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}");
        }
        Ok(())
    }
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
