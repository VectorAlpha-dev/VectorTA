//! # Linear Regression Angle (LRA)
//!
//! Computes the angle (in degrees) of the linear regression line for a given period.
//! Follows ALMA-style API for compatibility, SIMD-stubbed, with streaming and batch mode.
//!
//! ## Parameters
//! - **period**: Window size (number of data points), defaults to 14.
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are `NaN`.
//! - **InvalidPeriod**: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data for `period`.
//!
//! ## Returns
//! - **`Ok(Linearreg_angleOutput)`** on success with `.values` field.
//! - **`Err(Linearreg_angleError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum Linearreg_angleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for Linearreg_angleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            Linearreg_angleData::Slice(slice) => slice,
            Linearreg_angleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleParams {
    pub period: Option<usize>,
}

impl Default for Linearreg_angleParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleInput<'a> {
    pub data: Linearreg_angleData<'a>,
    pub params: Linearreg_angleParams,
}

impl<'a> Linearreg_angleInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: Linearreg_angleParams) -> Self {
        Self {
            data: Linearreg_angleData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: Linearreg_angleParams) -> Self {
        Self {
            data: Linearreg_angleData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", Linearreg_angleParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Linearreg_angleBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for Linearreg_angleBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl Linearreg_angleBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
        let p = Linearreg_angleParams { period: self.period };
        let i = Linearreg_angleInput::from_candles(c, "close", p);
        linearreg_angle_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
        let p = Linearreg_angleParams { period: self.period };
        let i = Linearreg_angleInput::from_slice(d, p);
        linearreg_angle_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<Linearreg_angleStream, Linearreg_angleError> {
        let p = Linearreg_angleParams { period: self.period };
        Linearreg_angleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum Linearreg_angleError {
    #[error("linearreg_angle: All values are NaN.")]
    AllValuesNaN,
    #[error("linearreg_angle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("linearreg_angle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("linearreg_angle: Empty data slice.")]
    EmptyData,
}

#[inline]
pub fn linearreg_angle(input: &Linearreg_angleInput) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
    linearreg_angle_with_kernel(input, Kernel::Auto)
}

pub fn linearreg_angle_with_kernel(input: &Linearreg_angleInput, kernel: Kernel) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(Linearreg_angleError::EmptyData);
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(Linearreg_angleError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(Linearreg_angleError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(Linearreg_angleError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_angle_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                linearreg_angle_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_angle_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(Linearreg_angleOutput { values: out })
}

#[inline]
pub fn linearreg_angle_scalar(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let sum_x = (period * (period - 1)) as f64 * 0.5;
    let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
    let divisor = sum_x * sum_x - (period as f64) * sum_x_sqr;
    let n = data.len();
    let mut prefix_data = vec![0.0; n + 1];
    let mut prefix_id = vec![0.0; n + 1];

    for i in 0..n {
        prefix_data[i + 1] = prefix_data[i] + data[i];
        prefix_id[i + 1] = prefix_id[i] + (i as f64) * data[i];
    }
    for i in (first_valid + period - 1)..n {
        let sum_y = prefix_data[i + 1] - prefix_data[i + 1 - period];
        let sum_kd = prefix_id[i + 1] - prefix_id[i + 1 - period];
        let sum_xy = (i as f64) * sum_y - sum_kd;
        let slope = ((period as f64) * sum_xy - sum_x * sum_y) / divisor;
        out[i] = slope.atan() * (180.0 / PI);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { linearreg_angle_avx512_short(data, period, first_valid, out) }
    } else {
        unsafe { linearreg_angle_avx512_long(data, period, first_valid, out) }
    }
}

#[inline]
pub fn linearreg_angle_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    linearreg_angle_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    linearreg_angle_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    linearreg_angle_scalar(data, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    sum_x: f64,
    sum_x_sqr: f64,
    divisor: f64,
    prefix_sum: f64,
    prefix_id: f64,
}

impl Linearreg_angleStream {
    pub fn try_new(params: Linearreg_angleParams) -> Result<Self, Linearreg_angleError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(Linearreg_angleError::InvalidPeriod { period, data_len: 0 });
        }

        let sum_x = (period * (period - 1)) as f64 * 0.5;
        let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
        let divisor = sum_x * sum_x - (period as f64) * sum_x_sqr;

        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            sum_x,
            sum_x_sqr,
            divisor,
            prefix_sum: 0.0,
            prefix_id: 0.0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let idx = self.head;
        let prev = self.buffer[idx];
        self.buffer[idx] = value;
        self.prefix_sum = self.prefix_sum + value - prev;
        self.prefix_id = self.prefix_id + (idx as f64) * value - (idx as f64) * prev;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        let period = self.period as f64;
        let i = (self.head + self.period - 1) % self.period;
        let sum_y = self.prefix_sum;
        let sum_kd = self.prefix_id;
        let sum_xy = (i as f64) * sum_y - sum_kd;
        let slope = (period * sum_xy - self.sum_x * sum_y) / self.divisor;
        Some(slope.atan() * (180.0 / PI))
    }
}

#[derive(Clone, Debug)]
pub struct Linearreg_angleBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for Linearreg_angleBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 60, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Linearreg_angleBatchBuilder {
    range: Linearreg_angleBatchRange,
    kernel: Kernel,
}

impl Linearreg_angleBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        linearreg_angle_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        Linearreg_angleBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        Linearreg_angleBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn linearreg_angle_batch_with_kernel(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    k: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(Linearreg_angleError::InvalidPeriod {
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
    linearreg_angle_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct Linearreg_angleBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<Linearreg_angleParams>,
    pub rows: usize,
    pub cols: usize,
}

impl Linearreg_angleBatchOutput {
    pub fn row_for_params(&self, p: &Linearreg_angleParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
        })
    }
    pub fn values_for(&self, p: &Linearreg_angleParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &Linearreg_angleBatchRange) -> Vec<Linearreg_angleParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(Linearreg_angleParams {
            period: Some(p),
        });
    }
    out
}

#[inline(always)]
pub fn linearreg_angle_batch_slice(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    kern: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    linearreg_angle_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linearreg_angle_batch_par_slice(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    kern: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    linearreg_angle_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linearreg_angle_batch_inner(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(Linearreg_angleError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(Linearreg_angleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(Linearreg_angleError::NotEnoughValidData {
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
            Kernel::Scalar => linearreg_angle_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => linearreg_angle_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => linearreg_angle_row_avx512(data, first, period, out_row),
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

        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(Linearreg_angleBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn linearreg_angle_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_angle_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_angle_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        linearreg_angle_row_avx512_short(data, first, period, out)
    
        } else {
        linearreg_angle_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_angle_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_angle_scalar(data, period, first, out)
}

#[inline(always)]
fn expand_grid_lra(r: &Linearreg_angleBatchRange) -> Vec<Linearreg_angleParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    axis_usize(r.period)
        .into_iter()
        .map(|p| Linearreg_angleParams { period: Some(p) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_lra_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = Linearreg_angleParams { period: None };
        let input = Linearreg_angleInput::from_candles(&candles, "close", default_params);
        let output = linearreg_angle_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_lra_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = Linearreg_angleParams { period: Some(14) };
        let input = Linearreg_angleInput::from_candles(&candles, "close", params);
        let result = linearreg_angle_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -89.30491945492733,
            -89.28911257342405,
            -89.1088041965075,
            -86.58419429159467,
            -87.77085937059316,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-5,
                "[{}] LRA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_lra_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = Linearreg_angleParams { period: Some(0) };
        let input = Linearreg_angleInput::from_slice(&input_data, params);
        let res = linearreg_angle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LRA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_lra_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = Linearreg_angleParams { period: Some(10) };
        let input = Linearreg_angleInput::from_slice(&data_small, params);
        let res = linearreg_angle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LRA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_lra_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = Linearreg_angleParams { period: Some(14) };
        let input = Linearreg_angleInput::from_slice(&single_point, params);
        let res = linearreg_angle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LRA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_lra_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = Linearreg_angleParams { period: Some(14) };
        let first_input = Linearreg_angleInput::from_candles(&candles, "close", first_params);
        let first_result = linearreg_angle_with_kernel(&first_input, kernel)?;

        let second_params = Linearreg_angleParams { period: Some(14) };
        let second_input = Linearreg_angleInput::from_slice(&first_result.values, second_params);
        let second_result = linearreg_angle_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    macro_rules! generate_all_lra_tests {
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
    generate_all_lra_tests!(
        check_lra_partial_params,
        check_lra_accuracy,
        check_lra_zero_period,
        check_lra_period_exceeds_length,
        check_lra_very_small_dataset,
        check_lra_reinput
    );
        fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = Linearreg_angleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = Linearreg_angleParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -89.30491945492733,
            -89.28911257342405,
            -89.1088041965075,
            -86.58419429159467,
            -87.77085937059316,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    fn check_batch_grid_search(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let batch = Linearreg_angleBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 16, 2)
            .apply_candles(&c, "close")?;

        // Should have periods: 10, 12, 14, 16
        let periods = [10, 12, 14, 16];
        assert_eq!(batch.rows, 4);

        for (ix, p) in periods.iter().enumerate() {
            let param = Linearreg_angleParams { period: Some(*p) };
            let row_idx = batch.row_for_params(&param);
            assert_eq!(row_idx, Some(ix), "Batch grid missing period {p}");
            let row = batch.values_for(&param).expect("Missing row for period");
            assert_eq!(row.len(), batch.cols, "Row len mismatch for period {p}");
        }
        Ok(())
    }

    fn check_batch_period_static(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let batch = Linearreg_angleBatchBuilder::new()
            .kernel(kernel)
            .period_static(14)
            .apply_candles(&c, "close")?;

        assert_eq!(batch.rows, 1);
        let param = Linearreg_angleParams { period: Some(14) };
        let row = batch.values_for(&param).expect("Missing static row");
        assert_eq!(row.len(), batch.cols);

        // Check a value
        let last = *row.last().unwrap();
        let expected = -87.77085937059316;
        assert!((last - expected).abs() < 1e-5, "Static period row last val mismatch: got {last}, want {expected}");

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
    gen_batch_tests!(check_batch_grid_search);
    gen_batch_tests!(check_batch_period_static);

}
