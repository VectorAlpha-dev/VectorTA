//! # Rolling Standard Deviation (STDDEV)
//!
//! The STDDEV indicator measures the rolling standard deviation of a window over input data,
//! scaled by `nbdev`. Parameters `period` and `nbdev` control the window size and deviation scale.
//! Features include AVX2/AVX512 API parity, streaming computation, and parameter grid batching.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **nbdev**: Multiplier for standard deviations (default: 1.0).
//!
//! ## Errors
//! - **AllValuesNaN**: stddev: All input values are `NaN`.
//! - **InvalidPeriod**: stddev: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: stddev: Not enough valid data points for requested `period`.
//!
//! ## Returns
//! - **Ok(StdDevOutput)** on success (output vector of length == input).
//! - **Err(StdDevError)** otherwise.

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

impl<'a> AsRef<[f64]> for StdDevInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            StdDevData::Slice(slice) => slice,
            StdDevData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum StdDevData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct StdDevOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StdDevParams {
    pub period: Option<usize>,
    pub nbdev: Option<f64>,
}

impl Default for StdDevParams {
    fn default() -> Self {
        Self {
            period: Some(5),
            nbdev: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StdDevInput<'a> {
    pub data: StdDevData<'a>,
    pub params: StdDevParams,
}

impl<'a> StdDevInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: StdDevParams) -> Self {
        Self {
            data: StdDevData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: StdDevParams) -> Self {
        Self {
            data: StdDevData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", StdDevParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
    #[inline]
    pub fn get_nbdev(&self) -> f64 {
        self.params.nbdev.unwrap_or(1.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StdDevBuilder {
    period: Option<usize>,
    nbdev: Option<f64>,
    kernel: Kernel,
}

impl Default for StdDevBuilder {
    fn default() -> Self {
        Self {
            period: None,
            nbdev: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StdDevBuilder {
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
    pub fn nbdev(mut self, x: f64) -> Self {
        self.nbdev = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<StdDevOutput, StdDevError> {
        let p = StdDevParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        let i = StdDevInput::from_candles(c, "close", p);
        stddev_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<StdDevOutput, StdDevError> {
        let p = StdDevParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        let i = StdDevInput::from_slice(d, p);
        stddev_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<StdDevStream, StdDevError> {
        let p = StdDevParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        StdDevStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum StdDevError {
    #[error("stddev: All values are NaN.")]
    AllValuesNaN,
    #[error("stddev: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("stddev: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn stddev(input: &StdDevInput) -> Result<StdDevOutput, StdDevError> {
    stddev_with_kernel(input, Kernel::Auto)
}

pub fn stddev_with_kernel(
    input: &StdDevInput,
    kernel: Kernel,
) -> Result<StdDevOutput, StdDevError> {
    let data: &[f64] = match &input.data {
        StdDevData::Candles { candles, source } => source_type(candles, source),
        StdDevData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StdDevError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();
    let nbdev = input.get_nbdev();

    if period == 0 || period > len {
        return Err(StdDevError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(StdDevError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                stddev_scalar(data, period, first, nbdev, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stddev_avx2(data, period, first, nbdev, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                stddev_avx512(data, period, first, nbdev, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(StdDevOutput { values: out })
}

#[inline]
pub fn stddev_scalar(data: &[f64], period: usize, first: usize, nbdev: f64, out: &mut [f64]) {
    let mut sum = 0.0;
    let mut sum_sqr = 0.0;
    for &val in &data[first..first + period] {
        sum += val;
        sum_sqr += val * val;
    }
    let mut compute = |sum: f64, sum_sqr: f64| {
        let mean = sum / period as f64;
        let var = (sum_sqr / period as f64) - (mean * mean);
        if var <= 0.0 {
            0.0
        } else {
            var.sqrt() * nbdev
        }
    };
    out[first + period - 1] = compute(sum, sum_sqr);
    for i in (first + period)..data.len() {
        let old = data[i - period];
        let new = data[i];
        sum += new - old;
        sum_sqr += new * new - old * old;
        out[i] = compute(sum, sum_sqr);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stddev_avx2(data: &[f64], period: usize, first: usize, nbdev: f64, out: &mut [f64]) {
    stddev_scalar(data, period, first, nbdev, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stddev_avx512(data: &[f64], period: usize, first: usize, nbdev: f64, out: &mut [f64]) {
    if period <= 32 {
        unsafe { stddev_avx512_short(data, period, first, nbdev, out) }
    } else {
        unsafe { stddev_avx512_long(data, period, first, nbdev, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn stddev_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn stddev_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out);
}

#[derive(Debug, Clone)]
pub struct StdDevStream {
    period: usize,
    nbdev: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    sum: f64,
    sum_sqr: f64,
}

impl StdDevStream {
    pub fn try_new(params: StdDevParams) -> Result<Self, StdDevError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(StdDevError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let nbdev = params.nbdev.unwrap_or(1.0);
        Ok(Self {
            period,
            nbdev,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            sum: 0.0,
            sum_sqr: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.filled {
            let old = self.buffer[self.head];
            self.sum += value - old;
            self.sum_sqr += value * value - old * old;
        } else {
            self.sum += value;
            self.sum_sqr += value * value;
            if self.head + 1 == self.period {
                self.filled = true;
            }
        }

        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        if !self.filled {
            return None;
        }
        let mean = self.sum / self.period as f64;
        let var = (self.sum_sqr / self.period as f64) - (mean * mean);
        Some(if var <= 0.0 {
            0.0
        } else {
            var.sqrt() * self.nbdev
        })
    }
}

#[derive(Clone, Debug)]
pub struct StdDevBatchRange {
    pub period: (usize, usize, usize),
    pub nbdev: (f64, f64, f64),
}

impl Default for StdDevBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 50, 1),
            nbdev: (1.0, 1.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StdDevBatchBuilder {
    range: StdDevBatchRange,
    kernel: Kernel,
}

impl StdDevBatchBuilder {
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
    #[inline]
    pub fn nbdev_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nbdev = (start, end, step);
        self
    }
    #[inline]
    pub fn nbdev_static(mut self, x: f64) -> Self {
        self.range.nbdev = (x, x, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<StdDevBatchOutput, StdDevError> {
        stddev_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<StdDevBatchOutput, StdDevError> {
        StdDevBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<StdDevBatchOutput, StdDevError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<StdDevBatchOutput, StdDevError> {
        StdDevBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn stddev_batch_with_kernel(
    data: &[f64],
    sweep: &StdDevBatchRange,
    k: Kernel,
) -> Result<StdDevBatchOutput, StdDevError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(StdDevError::InvalidPeriod {
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
    stddev_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StdDevBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<StdDevParams>,
    pub rows: usize,
    pub cols: usize,
}
impl StdDevBatchOutput {
    pub fn row_for_params(&self, p: &StdDevParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
                && (c.nbdev.unwrap_or(1.0) - p.nbdev.unwrap_or(1.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &StdDevParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &StdDevBatchRange) -> Vec<StdDevParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let periods = axis_usize(r.period);
    let nbdevs = axis_f64(r.nbdev);

    let mut out = Vec::with_capacity(periods.len() * nbdevs.len());
    for &p in &periods {
        for &n in &nbdevs {
            out.push(StdDevParams {
                period: Some(p),
                nbdev: Some(n),
            });
        }
    }
    out
}

#[inline(always)]
pub fn stddev_batch_slice(
    data: &[f64],
    sweep: &StdDevBatchRange,
    kern: Kernel,
) -> Result<StdDevBatchOutput, StdDevError> {
    stddev_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn stddev_batch_par_slice(
    data: &[f64],
    sweep: &StdDevBatchRange,
    kern: Kernel,
) -> Result<StdDevBatchOutput, StdDevError> {
    stddev_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn stddev_batch_inner(
    data: &[f64],
    sweep: &StdDevBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<StdDevBatchOutput, StdDevError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(StdDevError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StdDevError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(StdDevError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let nbdev = combos[row].nbdev.unwrap();
        match kern {
            Kernel::Scalar => stddev_row_scalar(data, first, period, nbdev, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => stddev_row_avx2(data, first, period, nbdev, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => stddev_row_avx512(data, first, period, nbdev, out_row),
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

    Ok(StdDevBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn stddev_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx2(data: &[f64], first: usize, period: usize, nbdev: f64, out: &mut [f64]) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        stddev_row_avx512_short(data, first, period, nbdev, out)
    } else {
        stddev_row_avx512_long(data, first, period, nbdev, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[inline(always)]
pub fn expand_grid_stddev(r: &StdDevBatchRange) -> Vec<StdDevParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_stddev_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = StdDevParams {
            period: None,
            nbdev: None,
        };
        let input = StdDevInput::from_candles(&candles, "close", default_params);
        let output = stddev_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_stddev_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = StdDevInput::from_candles(&candles, "close", StdDevParams::default());
        let result = stddev_with_kernel(&input, kernel)?;
        let expected_last_five = [
            180.12506767314034,
            77.7395652441455,
            127.16225857341935,
            89.40156600773197,
            218.50034325919697,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] STDDEV {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_stddev_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = StdDevInput::with_default_candles(&candles);
        match input.data {
            StdDevData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected StdDevData::Candles"),
        }
        let output = stddev_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_stddev_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = StdDevParams {
            period: Some(0),
            nbdev: None,
        };
        let input = StdDevInput::from_slice(&input_data, params);
        let res = stddev_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] STDDEV should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_stddev_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = StdDevParams {
            period: Some(10),
            nbdev: None,
        };
        let input = StdDevInput::from_slice(&data_small, params);
        let res = stddev_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] STDDEV should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_stddev_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = StdDevParams {
            period: Some(5),
            nbdev: None,
        };
        let input = StdDevInput::from_slice(&single_point, params);
        let res = stddev_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] STDDEV should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_stddev_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = StdDevParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let first_input = StdDevInput::from_candles(&candles, "close", first_params);
        let first_result = stddev_with_kernel(&first_input, kernel)?;

        let second_params = StdDevParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let second_input = StdDevInput::from_slice(&first_result.values, second_params);
        let second_result = stddev_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 19..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "STDDEV slice reinput: Expected no NaN after index 19, but found NaN at index {}",
                i
            );
        }
        Ok(())
    }

    fn check_stddev_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = StdDevInput::from_candles(
            &candles,
            "close",
            StdDevParams {
                period: Some(5),
                nbdev: None,
            },
        );
        let res = stddev_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_stddev_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;
        let nbdev = 1.0;

        let input = StdDevInput::from_candles(
            &candles,
            "close",
            StdDevParams {
                period: Some(period),
                nbdev: Some(nbdev),
            },
        );
        let batch_output = stddev_with_kernel(&input, kernel)?.values;

        let mut stream = StdDevStream::try_new(StdDevParams {
            period: Some(period),
            nbdev: Some(nbdev),
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
                "[{}] STDDEV streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_stddev_tests {
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
    generate_all_stddev_tests!(
        check_stddev_partial_params,
        check_stddev_accuracy,
        check_stddev_default_candles,
        check_stddev_zero_period,
        check_stddev_period_exceeds_length,
        check_stddev_very_small_dataset,
        check_stddev_reinput,
        check_stddev_nan_handling,
        check_stddev_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = StdDevBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = StdDevParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            180.12506767314034,
            77.7395652441455,
            127.16225857341935,
            89.40156600773197,
            218.50034325919697,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
