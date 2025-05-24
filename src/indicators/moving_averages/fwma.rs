//! # Fibonacci Weighted Moving Average (FWMA)
//!
//! A weighted moving average that applies Fibonacci coefficients to each data
//! point within the specified `period`. Fibonacci numbers grow in a way that
//! places slightly greater emphasis on more recent data points, while still
//! smoothing out short-term noise. The sum of the weights (Fibonacci coefficients)
//! is normalized to 1.0, ensuring the resulting average remains correctly scaled.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **AllValuesNaN**: fwma: All input data values are `NaN`.
//! - **InvalidPeriod**: fwma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: fwma: Not enough valid data points for the requested `period`.
//! - **ZeroFibonacciSum**: fwma: The sum of Fibonacci weights was zero, preventing normalization.
//!
//! ## Returns
//! - **`Ok(FwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(FwmaError)`** otherwise.

use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for FwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            FwmaData::Slice(slice) => slice,
            FwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FwmaParams {
    pub period: Option<usize>,
}

impl Default for FwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct FwmaInput<'a> {
    pub data: FwmaData<'a>,
    pub params: FwmaParams,
}

impl<'a> FwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: FwmaParams) -> Self {
        Self {
            data: FwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: FwmaParams) -> Self {
        Self {
            data: FwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", FwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for FwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FwmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<FwmaOutput, FwmaError> {
        let p = FwmaParams {
            period: self.period,
        };
        let i = FwmaInput::from_candles(c, "close", p);
        fwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<FwmaOutput, FwmaError> {
        let p = FwmaParams {
            period: self.period,
        };
        let i = FwmaInput::from_slice(d, p);
        fwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<FwmaStream, FwmaError> {
        let p = FwmaParams {
            period: self.period,
        };
        FwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum FwmaError {
    #[error("fwma: All values are NaN.")]
    AllValuesNaN,
    #[error("fwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fwma: Fibonacci sum is zero. Cannot normalize weights.")]
    ZeroFibonacciSum,
}

#[inline]
pub fn fwma(input: &FwmaInput) -> Result<FwmaOutput, FwmaError> {
    fwma_with_kernel(input, Kernel::Auto)
}

pub fn fwma_with_kernel(input: &FwmaInput, kernel: Kernel) -> Result<FwmaOutput, FwmaError> {
    let data: &[f64] = match &input.data {
        FwmaData::Candles { candles, source } => source_type(candles, source),
        FwmaData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(FwmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(FwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(FwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let mut fib = vec![1.0; period];
    for i in 2..period {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    let fib_sum: f64 = fib.iter().sum();
    if fib_sum == 0.0 {
        return Err(FwmaError::ZeroFibonacciSum);
    }
    for w in &mut fib {
        *w /= fib_sum;
    }
    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                fwma_scalar(data, &fib, period, first, &mut out)
            }
            Kernel::Avx2 | Kernel::Avx2Batch => fwma_avx2(data, &fib, period, first, &mut out),
            Kernel::Avx512 | Kernel::Avx512Batch => {
                fwma_avx512(data, &fib, period, first, &mut out)
            }
            Kernel::Auto => unreachable!(),
        }
    }
    Ok(FwmaOutput { values: out })
}

#[inline(always)]
pub fn fwma_scalar(data: &[f64], fib: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    assert_eq!(fib.len(), period, "fib.len() must equal period");
    assert!(
        out.len() >= data.len(),
        "out must be at least as long as data"
    );

    let p4 = period & !3;
    for i in (first_val + period - 1)..data.len() {
        let start = i + 1 - period;
        let window = &data[start..start + period];
        let mut sum = 0.0;
        for (d4, w4) in window[..p4].chunks_exact(4).zip(fib[..p4].chunks_exact(4)) {
            sum += d4[0] * w4[0] + d4[1] * w4[1] + d4[2] * w4[2] + d4[3] * w4[3];
        }
        for (d, w) in window[p4..].iter().zip(&fib[p4..]) {
            sum += d * w;
        }
        out[i] = sum;
    }
}

#[inline(always)]
pub fn fwma_avx512(data: &[f64], fib: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    fwma_scalar(data, fib, period, first_valid, out)
}
#[inline(always)]
pub fn fwma_avx2(data: &[f64], fib: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    fwma_scalar(data, fib, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct FwmaStream {
    period: usize,
    fib: Vec<f64>,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl FwmaStream {
    pub fn try_new(params: FwmaParams) -> Result<Self, FwmaError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(FwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let mut fib = vec![1.0; period];
        for i in 2..period {
            fib[i] = fib[i - 1] + fib[i - 2];
        }
        let fib_sum: f64 = fib.iter().sum();
        if fib_sum == 0.0 {
            return Err(FwmaError::ZeroFibonacciSum);
        }
        for w in &mut fib {
            *w /= fib_sum;
        }
        Ok(Self {
            period,
            fib,
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
        let mut sum = 0.0;
        let mut idx = self.head;
        for &w in &self.fib {
            sum += w * self.buffer[idx];
            idx = (idx + 1) % self.period;
        }
        sum
    }
}

#[derive(Clone, Debug)]
pub struct FwmaBatchRange {
    pub period: (usize, usize, usize),
}
impl Default for FwmaBatchRange {
    fn default() -> Self {
        Self { period: (5, 40, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FwmaBatchBuilder {
    range: FwmaBatchRange,
    kernel: Kernel,
}

impl FwmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<FwmaBatchOutput, FwmaError> {
        fwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<FwmaBatchOutput, FwmaError> {
        FwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<FwmaBatchOutput, FwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FwmaBatchOutput, FwmaError> {
        FwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn fwma_batch_with_kernel(
    data: &[f64],
    sweep: &FwmaBatchRange,
    k: Kernel,
) -> Result<FwmaBatchOutput, FwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(FwmaError::InvalidPeriod {
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
    fwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct FwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl FwmaBatchOutput {
    pub fn row_for_params(&self, p: &FwmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &FwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &FwmaBatchRange) -> Vec<FwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(FwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn fwma_batch_slice(
    data: &[f64],
    sweep: &FwmaBatchRange,
    kern: Kernel,
) -> Result<FwmaBatchOutput, FwmaError> {
    fwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn fwma_batch_par_slice(
    data: &[f64],
    sweep: &FwmaBatchRange,
    kern: Kernel,
) -> Result<FwmaBatchOutput, FwmaError> {
    fwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn fwma_batch_inner(
    data: &[f64],
    sweep: &FwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FwmaBatchOutput, FwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(FwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(FwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(FwmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let cap = rows * max_p;
    let mut aligned = AlignedVec::with_capacity(cap);
    let flat_fib = aligned.as_mut_slice();

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let base = row * max_p;
        let slice = &mut flat_fib[base..base + period];
        slice[0] = 1.0;
        if period > 1 {
            slice[1] = 1.0;
        }
        for i in 2..period {
            slice[i] = slice[i - 1] + slice[i - 2];
        }
        let sum: f64 = slice[..period].iter().sum();
        for w in &mut slice[..period] {
            *w /= sum;
        }
    }

    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let fib_ptr = flat_fib.as_ptr().add(row * max_p);

        match kern {
            Kernel::Avx512 => fwma_row_avx512(data, first, period, max_p, fib_ptr, out_row),
            Kernel::Avx2 => fwma_row_avx2(data, first, period, max_p, fib_ptr, out_row),
            _ => fwma_row_scalar(data, first, period, max_p, fib_ptr, out_row),
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

    Ok(FwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn fwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    fib_ptr: *const f64,
    out: &mut [f64],
) {
    let p4 = period & !3;
    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for k in (0..p4).step_by(4) {
            let fib = std::slice::from_raw_parts(fib_ptr.add(k), 4);
            let d = &data[start + k..start + k + 4];
            sum += d[0] * fib[0] + d[1] * fib[1] + d[2] * fib[2] + d[3] * fib[3];
        }
        for k in p4..period {
            sum += *data.get_unchecked(start + k) * *fib_ptr.add(k);
        }
        out[i] = sum;
    }
}

#[inline(always)]
unsafe fn fwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    fib_ptr: *const f64,
    out: &mut [f64],
) {
    fwma_row_scalar(data, first, period, stride, fib_ptr, out)
}
#[inline(always)]
unsafe fn fwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    fib_ptr: *const f64,
    out: &mut [f64],
) {
    fwma_row_scalar(data, first, period, stride, fib_ptr, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::helpers::skip_if_unsupported;

    fn check_fwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = FwmaParams { period: None };
        let input = FwmaInput::from_candles(&candles, "close", default_params);
        let output = fwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_fwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = FwmaInput::with_default_candles(&candles);
        let result = fwma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] FWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_fwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = FwmaInput::with_default_candles(&candles);
        match input.data {
            FwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected FwmaData::Candles"),
        }
        let output = fwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_fwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = FwmaParams { period: Some(0) };
        let input = FwmaInput::from_slice(&input_data, params);
        let res = fwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_fwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = FwmaParams { period: Some(10) };
        let input = FwmaInput::from_slice(&data_small, params);
        let res = fwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_fwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let single_point = [42.0];
        let params = FwmaParams { period: Some(5) };
        let input = FwmaInput::from_slice(&single_point, params);
        let res = fwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_fwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = FwmaParams { period: Some(5) };
        let first_input = FwmaInput::from_candles(&candles, "close", first_params);
        let first_result = fwma_with_kernel(&first_input, kernel)?;

        let second_params = FwmaParams { period: Some(3) };
        let second_input = FwmaInput::from_slice(&first_result.values, second_params);
        let second_result = fwma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] NaN found at idx {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_fwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = FwmaInput::from_candles(&candles, "close", FwmaParams { period: Some(5) });
        let res = fwma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 50 {
            for (i, &val) in res.values[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_fwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;

        let input = FwmaInput::from_candles(
            &candles,
            "close",
            FwmaParams {
                period: Some(period),
            },
        );
        let batch_output = fwma_with_kernel(&input, kernel)?.values;

        let mut stream = FwmaStream::try_new(FwmaParams {
            period: Some(period),
        })?;

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
            assert!(
                diff < 1e-9,
                "[{}] FWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_fwma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }

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

    generate_all_fwma_tests!(
        check_fwma_partial_params,
        check_fwma_accuracy,
        check_fwma_default_candles,
        check_fwma_zero_period,
        check_fwma_period_exceeds_length,
        check_fwma_very_small_dataset,
        check_fwma_reinput,
        check_fwma_nan_handling,
        check_fwma_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = FwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = FwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
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
