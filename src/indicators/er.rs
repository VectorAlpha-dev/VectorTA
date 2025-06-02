//! # Kaufman Efficiency Ratio (ER)
//!
//! The Kaufman Efficiency Ratio (ER) compares the absolute price change over a specified
//! period to the sum of the incremental absolute changes within that same window.
//! Returns a value between 0.0 and 1.0 (high = efficient move, low = choppy).
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **AllValuesNaN**: er: All input data values are `NaN`.
//! - **InvalidPeriod**: er: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: er: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(ErOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(ErError)`** otherwise.

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

impl<'a> AsRef<[f64]> for ErInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ErData::Slice(slice) => slice,
            ErData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ErData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ErOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ErParams {
    pub period: Option<usize>,
}

impl Default for ErParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct ErInput<'a> {
    pub data: ErData<'a>,
    pub params: ErParams,
}

impl<'a> ErInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ErParams) -> Self {
        Self {
            data: ErData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ErParams) -> Self {
        Self {
            data: ErData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ErParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ErBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ErBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ErBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ErOutput, ErError> {
        let p = ErParams { period: self.period };
        let i = ErInput::from_candles(c, "close", p);
        er_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ErOutput, ErError> {
        let p = ErParams { period: self.period };
        let i = ErInput::from_slice(d, p);
        er_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ErStream, ErError> {
        let p = ErParams { period: self.period };
        ErStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ErError {
    #[error("er: All values are NaN.")]
    AllValuesNaN,
    #[error("er: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("er: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn er(input: &ErInput) -> Result<ErOutput, ErError> {
    er_with_kernel(input, Kernel::Auto)
}

pub fn er_with_kernel(input: &ErInput, kernel: Kernel) -> Result<ErOutput, ErError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(ErError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ErError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(ErError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                er_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                er_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                er_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(ErOutput { values: out })
}

#[inline]
pub fn er_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let n = data.len();
    for i in (first + period - 1)..n {
        let start = i + 1 - period;
        let delta = (data[i] - data[start]).abs();
        let mut sum = 0.0;
        for j in start..i {
            sum += (data[j + 1] - data[j]).abs();
        }
        if sum > 0.0 {
            out[i] = delta / sum;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn er_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe {
        if period <= 32 {
            er_avx512_short(data, period, first, out);
        } else {
            er_avx512_long(data, period, first, out);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn er_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn er_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn er_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    er_scalar(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct ErStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl ErStream {
    pub fn try_new(params: ErParams) -> Result<Self, ErError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(ErError::InvalidPeriod { period, data_len: 0 });
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
        let mut sum = 0.0;
        let mut last = self.head;
        let mut prev = last;
        for _ in 1..self.period {
            prev = (prev + 1) % self.period;
            let a = self.buffer[last];
            let b = self.buffer[prev];
            sum += (b - a).abs();
            last = prev;
        }
        let start = self.head;
        let end = (self.head + self.period - 1) % self.period;
        let delta = (self.buffer[end] - self.buffer[start]).abs();
        if sum > 0.0 {
            Some(delta / sum)
        } else {
            Some(f64::NAN)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ErBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ErBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 60, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ErBatchBuilder {
    range: ErBatchRange,
    kernel: Kernel,
}

impl ErBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<ErBatchOutput, ErError> {
        er_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ErBatchOutput, ErError> {
        ErBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ErBatchOutput, ErError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ErBatchOutput, ErError> {
        ErBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn er_batch_with_kernel(
    data: &[f64],
    sweep: &ErBatchRange,
    k: Kernel,
) -> Result<ErBatchOutput, ErError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(ErError::InvalidPeriod {
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
    er_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ErBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ErParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ErBatchOutput {
    pub fn row_for_params(&self, p: &ErParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &ErParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ErBatchRange) -> Vec<ErParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ErParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn er_batch_slice(
    data: &[f64],
    sweep: &ErBatchRange,
    kern: Kernel,
) -> Result<ErBatchOutput, ErError> {
    er_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn er_batch_par_slice(
    data: &[f64],
    sweep: &ErBatchRange,
    kern: Kernel,
) -> Result<ErBatchOutput, ErError> {
    er_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn er_batch_inner(
    data: &[f64],
    sweep: &ErBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ErBatchOutput, ErError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ErError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(ErError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ErError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => er_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => er_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => er_row_avx512(data, first, period, out_row),
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
    Ok(ErBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn er_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        er_row_avx512_short(data, first, period, out);
    } else {
        er_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    er_scalar(data, period, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_er_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = ErParams { period: None };
        let input = ErInput::from_candles(&candles, "close", default_params);
        let output = er_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_er_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = ErInput::with_default_candles(&candles);
        match input.data {
            ErData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ErData::Candles"),
        }
        let output = er_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_er_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ErParams { period: Some(0) };
        let input = ErInput::from_slice(&input_data, params);
        let res = er_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ER should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_er_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ErParams { period: Some(10) };
        let input = ErInput::from_slice(&data_small, params);
        let res = er_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ER should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_er_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ErParams { period: Some(5) };
        let input = ErInput::from_slice(&single_point, params);
        let res = er_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ER should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_er_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = ErParams { period: Some(5) };
        let first_input = ErInput::from_candles(&candles, "close", first_params);
        let first_result = er_with_kernel(&first_input, kernel)?;

        let second_params = ErParams { period: Some(5) };
        let second_input = ErInput::from_slice(&first_result.values, second_params);
        let second_result = er_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_er_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = ErInput::from_candles(
            &candles,
            "close",
            ErParams {
                period: Some(5),
            },
        );
        let res = er_with_kernel(&input, kernel)?;
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

    fn check_er_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;

        let input = ErInput::from_candles(
            &candles,
            "close",
            ErParams {
                period: Some(period),
            },
        );
        let batch_output = er_with_kernel(&input, kernel)?.values;

        let mut stream = ErStream::try_new(ErParams { period: Some(period) })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(er_val) => stream_values.push(er_val),
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
                "[{}] ER streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_er_tests {
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

    generate_all_er_tests!(
        check_er_partial_params,
        check_er_default_candles,
        check_er_zero_period,
        check_er_period_exceeds_length,
        check_er_very_small_dataset,
        check_er_reinput,
        check_er_nan_handling,
        check_er_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = ErBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = ErParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        // Not a strict accuracy test, just batch output row length check.
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
