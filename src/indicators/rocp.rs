//! # Rate of Change Percentage (ROCP)
//!
//! The Rate of Change Percentage (ROCP) calculates the relative change in value
//! between the current price and the price `period` bars ago, without the
//! extra `* 100` factor used by ROC:
//!
//! \[ ROCP[i] = (price[i] - price[i - period]) / price[i - period] \]
//!
//! This indicator is centered around 0 and can be positive or negative.
//!
//! ## Parameters
//! - **period**: The lookback window (number of data points). Defaults to 9.
//!
//! ## Errors
//! - **AllValuesNaN**: rocp: All input data values are `NaN`.
//! - **InvalidPeriod**: rocp: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: rocp: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//!
//! ## Returns
//! - **`Ok(RocpOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the moving window is filled.
//! - **`Err(RocpError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for RocpInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RocpData::Slice(slice) => slice,
            RocpData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RocpData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocpOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RocpParams {
    pub period: Option<usize>,
}

impl Default for RocpParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct RocpInput<'a> {
    pub data: RocpData<'a>,
    pub params: RocpParams,
}

impl<'a> RocpInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: RocpParams) -> Self {
        Self {
            data: RocpData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: RocpParams) -> Self {
        Self {
            data: RocpData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", RocpParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RocpBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for RocpBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RocpBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<RocpOutput, RocpError> {
        let p = RocpParams {
            period: self.period,
        };
        let i = RocpInput::from_candles(c, "close", p);
        rocp_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<RocpOutput, RocpError> {
        let p = RocpParams {
            period: self.period,
        };
        let i = RocpInput::from_slice(d, p);
        rocp_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RocpStream, RocpError> {
        let p = RocpParams {
            period: self.period,
        };
        RocpStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum RocpError {
    #[error("rocp: All values are NaN.")]
    AllValuesNaN,
    #[error("rocp: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rocp: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn rocp(input: &RocpInput) -> Result<RocpOutput, RocpError> {
    rocp_with_kernel(input, Kernel::Auto)
}

pub fn rocp_with_kernel(input: &RocpInput, kernel: Kernel) -> Result<RocpOutput, RocpError> {
    let data: &[f64] = match &input.data {
        RocpData::Candles { candles, source } => source_type(candles, source),
        RocpData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocpError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(RocpError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(RocpError::NotEnoughValidData {
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
                rocp_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                rocp_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                rocp_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(RocpOutput { values: out })
}

#[inline]
pub fn rocp_scalar(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    for i in (first_val + period)..data.len() {
        let prev = data[i - period];
        out[i] = (data[i] - prev) / prev;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocp_avx512(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { rocp_avx512_short(data, period, first_val, out) }
    } else {
        unsafe { rocp_avx512_long(data, period, first_val, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocp_avx2(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // AVX2 stub (uses scalar for now, but keeps API parity)
    rocp_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocp_avx512_short(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // AVX512 short stub (uses scalar for now)
    rocp_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocp_avx512_long(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // AVX512 long stub (uses scalar for now)
    rocp_scalar(data, period, first_val, out)
}

#[derive(Debug, Clone)]
pub struct RocpStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl RocpStream {
    pub fn try_new(params: RocpParams) -> Result<Self, RocpError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(RocpError::InvalidPeriod {
                period,
                data_len: 0,
            });
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
        if !self.filled {
            return None;
        }
        Some((value - prev) / prev)
    }
}

#[derive(Clone, Debug)]
pub struct RocpBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for RocpBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RocpBatchBuilder {
    range: RocpBatchRange,
    kernel: Kernel,
}

impl RocpBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<RocpBatchOutput, RocpError> {
        rocp_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RocpBatchOutput, RocpError> {
        RocpBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RocpBatchOutput, RocpError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<RocpBatchOutput, RocpError> {
        RocpBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn rocp_batch_with_kernel(
    data: &[f64],
    sweep: &RocpBatchRange,
    k: Kernel,
) -> Result<RocpBatchOutput, RocpError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(RocpError::InvalidPeriod {
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
    rocp_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RocpBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<RocpParams>,
    pub rows: usize,
    pub cols: usize,
}
impl RocpBatchOutput {
    pub fn row_for_params(&self, p: &RocpParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(9) == p.period.unwrap_or(9)
        })
    }

    pub fn values_for(&self, p: &RocpParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &RocpBatchRange) -> Vec<RocpParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(RocpParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn rocp_batch_slice(
    data: &[f64],
    sweep: &RocpBatchRange,
    kern: Kernel,
) -> Result<RocpBatchOutput, RocpError> {
    rocp_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn rocp_batch_par_slice(
    data: &[f64],
    sweep: &RocpBatchRange,
    kern: Kernel,
) -> Result<RocpBatchOutput, RocpError> {
    rocp_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rocp_batch_inner(
    data: &[f64],
    sweep: &RocpBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RocpBatchOutput, RocpError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RocpError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocpError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.period.unwrap())
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(RocpError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => rocp_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => rocp_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => rocp_row_avx512(data, first, period, out_row),
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

    Ok(RocpBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn rocp_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    for i in (first + period)..data.len() {
        let prev = data[i - period];
        out[i] = (data[i] - prev) / prev;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rocp_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    rocp_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        rocp_row_avx512_short(data, first, period, out);
    } else {
        rocp_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    rocp_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    rocp_row_scalar(data, first, period, out)
}

#[inline(always)]
fn expand_grid_rocp(r: &RocpBatchRange) -> Vec<RocpParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    axis_usize(r.period)
        .into_iter()
        .map(|p| RocpParams { period: Some(p) })
        .collect()
}

// No changes to tests required; batch and scalar logic are unified.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_rocp_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = RocpParams { period: None };
        let input = RocpInput::from_candles(&candles, "close", default_params);
        let output = rocp_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_rocp_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = RocpInput::from_candles(&candles, "close", RocpParams { period: Some(10) });
        let result = rocp_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -0.0022551709049293996,
            -0.005561903481650759,
            -0.003275201323586514,
            -0.004945415398072297,
            -0.015045927020537019,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-9,
                "[{}] ROCP {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_rocp_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = RocpInput::with_default_candles(&candles);
        match input.data {
            RocpData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected RocpData::Candles"),
        }
        let output = rocp_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_rocp_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = RocpParams { period: Some(0) };
        let input = RocpInput::from_slice(&input_data, params);
        let res = rocp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ROCP should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_rocp_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = RocpParams { period: Some(10) };
        let input = RocpInput::from_slice(&data_small, params);
        let res = rocp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ROCP should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_rocp_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = RocpParams { period: Some(9) };
        let input = RocpInput::from_slice(&single_point, params);
        let res = rocp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ROCP should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_rocp_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = RocpParams { period: Some(14) };
        let first_input = RocpInput::from_candles(&candles, "close", first_params);
        let first_result = rocp_with_kernel(&first_input, kernel)?;

        let second_params = RocpParams { period: Some(14) };
        let second_input = RocpInput::from_slice(&first_result.values, second_params);
        let second_result = rocp_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] ROCP Slice Reinput {:?} mismatch at idx {}: got NaN",
                test_name,
                kernel,
                i
            );
        }
        Ok(())
    }

    fn check_rocp_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = RocpInput::from_candles(
            &candles,
            "close",
            RocpParams {
                period: Some(9),
            },
        );
        let res = rocp_with_kernel(&input, kernel)?;
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

    fn check_rocp_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 9;

        let input = RocpInput::from_candles(
            &candles,
            "close",
            RocpParams {
                period: Some(period),
            },
        );
        let batch_output = rocp_with_kernel(&input, kernel)?.values;

        let mut stream = RocpStream::try_new(RocpParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(rocp_val) => stream_values.push(rocp_val),
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
                "[{}] ROCP streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_rocp_tests {
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

    generate_all_rocp_tests!(
        check_rocp_partial_params,
        check_rocp_accuracy,
        check_rocp_default_candles,
        check_rocp_zero_period,
        check_rocp_period_exceeds_length,
        check_rocp_very_small_dataset,
        check_rocp_reinput,
        check_rocp_nan_handling,
        check_rocp_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = RocpBatchBuilder::new()
            .period_static(10)
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = RocpParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -0.0022551709049293996,
            -0.005561903481650759,
            -0.003275201323586514,
            -0.004945415398072297,
            -0.015045927020537019,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-9,
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
