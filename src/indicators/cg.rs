//! # Center of Gravity (CG)
//!
//! The Center of Gravity (CG) indicator attempts to measure the "center" of prices
//! over a given window, sometimes used for smoothing or cycle analysis.  
//!
//! ## Parameters
//! - **period**: The window size. Defaults to 10.
//!
//! ## Errors
//! - **EmptyData**: cg: Input data slice is empty.
//! - **InvalidPeriod**: cg: `period` is zero or exceeds the data length.
//! - **AllValuesNaN**: cg: All input data values are `NaN`.
//! - **NotEnoughValidData**: cg: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
//!
//! ## Returns
//! - **`Ok(CgOutput)`** on success, containing a `Vec<f64>` matching input length,
//!   with leading `NaN` until the warm-up period is reached.
//! - **`Err(CgError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for CgInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CgData::Slice(slice) => slice,
            CgData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CgData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CgOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CgParams {
    pub period: Option<usize>,
}

impl Default for CgParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct CgInput<'a> {
    pub data: CgData<'a>,
    pub params: CgParams,
}

impl<'a> CgInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CgParams) -> Self {
        Self {
            data: CgData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CgParams) -> Self {
        Self {
            data: CgData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CgParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CgBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CgBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CgBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<CgOutput, CgError> {
        let p = CgParams {
            period: self.period,
        };
        let i = CgInput::from_candles(c, "close", p);
        cg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CgOutput, CgError> {
        let p = CgParams {
            period: self.period,
        };
        let i = CgInput::from_slice(d, p);
        cg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CgStream, CgError> {
        let p = CgParams {
            period: self.period,
        };
        CgStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CgError {
    #[error("cg: Empty data provided for CG.")]
    EmptyData,
    #[error("cg: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cg: All values are NaN.")]
    AllValuesNaN,
    #[error("cg: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn cg(input: &CgInput) -> Result<CgOutput, CgError> {
    cg_with_kernel(input, Kernel::Auto)
}

pub fn cg_with_kernel(input: &CgInput, kernel: Kernel) -> Result<CgOutput, CgError> {
    let data: &[f64] = match &input.data {
        CgData::Candles { candles, source } => source_type(candles, source),
        CgData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(CgError::EmptyData);
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(CgError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CgError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period + 1 {
        return Err(CgError::NotEnoughValidData { needed: period + 1, valid: len - first });
    }

    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cg_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                cg_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cg_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(CgOutput { values: out })
}

#[inline]
pub fn cg_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    for i in (first + period)..data.len() {
        let mut num = 0.0;
        let mut denom = 0.0;
        for count in 0..period {
            let price = data[i - count];
            num += (1.0 + count as f64) * price;
            denom += price;
        }
        out[i] = if denom.abs() > f64::EPSILON { -num / denom } else { 0.0 };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cg_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { cg_avx512_short(data, period, first, out) }
    } else {
        unsafe { cg_avx512_long(data, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cg_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cg_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cg_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    cg_scalar(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct CgStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl CgStream {
    pub fn try_new(params: CgParams) -> Result<Self, CgError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(CgError::InvalidPeriod { period, data_len: 0 });
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
        if !self.filled {
            return None;
        }
        Some(self.dot_ring())
    }
    #[inline(always)]
    fn dot_ring(&self) -> f64 {
        let mut num = 0.0;
        let mut denom = 0.0;
        let mut idx = self.head;
        for k in 0..self.period {
            idx = if idx == 0 { self.period - 1 } else { idx - 1 };
            let price = self.buffer[idx];
            num += (1.0 + k as f64) * price;
            denom += price;
        }
        if denom.abs() > f64::EPSILON {
            -num / denom
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
pub struct CgBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CgBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 10, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CgBatchBuilder {
    range: CgBatchRange,
    kernel: Kernel,
}

impl CgBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<CgBatchOutput, CgError> {
        cg_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CgBatchOutput, CgError> {
        CgBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CgBatchOutput, CgError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CgBatchOutput, CgError> {
        CgBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn cg_batch_with_kernel(
    data: &[f64],
    sweep: &CgBatchRange,
    k: Kernel,
) -> Result<CgBatchOutput, CgError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(CgError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cg_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CgBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CgParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CgBatchOutput {
    pub fn row_for_params(&self, p: &CgParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(10) == p.period.unwrap_or(10)
        })
    }
    pub fn values_for(&self, p: &CgParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CgBatchRange) -> Vec<CgParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CgParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn cg_batch_slice(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
) -> Result<CgBatchOutput, CgError> {
    cg_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cg_batch_par_slice(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
) -> Result<CgBatchOutput, CgError> {
    cg_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cg_batch_inner(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CgBatchOutput, CgError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CgError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(CgError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p + 1 {
        return Err(CgError::NotEnoughValidData { needed: max_p + 1, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => cg_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cg_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cg_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(CgBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn cg_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        cg_row_avx512_short(data, first, period, out)
    } else {
        cg_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cg_scalar(data, period, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_cg_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = CgParams { period: Some(12) };
        let input = CgInput::from_candles(&candles, "close", partial_params);
        let output = cg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cg_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -4.99905186931943,
            -4.998559827254377,
            -4.9970065675119555,
            -4.9928483984587295,
            -5.004210799262688,
        ];
        assert!(result.values.len() >= 5, "Not enough data for final 5-values check");
        let start = result.values.len() - 5;
        for (i, &exp) in expected_last_five.iter().enumerate() {
            let got = result.values[start + i];
            assert!(
                (got - exp).abs() < 1e-4,
                "Mismatch in CG at idx {}: expected={}, got={}", start + i, exp, got
            );
        }
        Ok(())
    }

    fn check_cg_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CgInput::with_default_candles(&candles);
        match input.data {
            CgData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CgData::Candles"),
        }
        let output = cg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cg_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = CgParams { period: Some(0) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for zero period");
        Ok(())
    }

    fn check_cg_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for period > data.len()");
        Ok(())
    }

    fn check_cg_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for data smaller than period=10");
        Ok(())
    }

    fn check_cg_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg_with_kernel(&input, kernel)?;
        let check_idx = 240;
        if result.values.len() > check_idx {
            for i in check_idx..result.values.len() {
                if !result.values[i].is_nan() {
                    break;
                }
                if i == result.values.len() - 1 {
                    panic!("All CG values from index {} onward are NaN.", check_idx);
                }
            }
        }
        Ok(())
    }

    fn check_cg_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 10;
        let input = CgInput::from_candles(&candles, "close", CgParams { period: Some(period) });
        let batch_output = cg_with_kernel(&input, kernel)?.values;
        let mut stream = CgStream::try_new(CgParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(diff < 1e-9,
                "[{}] CG streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_cg_tests {
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

    generate_all_cg_tests!(
        check_cg_partial_params,
        check_cg_accuracy,
        check_cg_default_candles,
        check_cg_zero_period,
        check_cg_period_exceeds_length,
        check_cg_very_small_dataset,
        check_cg_nan_handling,
        check_cg_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CgBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = CgParams::default();
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
