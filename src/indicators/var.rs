//! # Rolling Variance (VAR)
//!
//! Computes the rolling variance over a specified window (`period`), with optional standard deviation factor (`nbdev`).
//! API mirrors alma.rs, including builder, streaming, batch/grid sweep, and AVX stubs for full parity.
//!
//! ## Parameters
//! - **period**: Window size (defaults to 14).
//! - **nbdev**: Stddev factor (`VAR = var * nbdev^2`, defaults to 1.0).
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are `NaN`.
//! - **InvalidPeriod**: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data points for `period`.
//!
//! ## Returns
//! - `Ok(VarOutput)` on success, with a `Vec<f64>` matching the input.
//! - `Err(VarError)` otherwise.

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

// -- Data Structures --

#[derive(Debug, Clone)]
pub enum VarData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for VarInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VarData::Slice(slice) => slice,
            VarData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VarParams {
    pub period: Option<usize>,
    pub nbdev: Option<f64>,
}

impl Default for VarParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            nbdev: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarInput<'a> {
    pub data: VarData<'a>,
    pub params: VarParams,
}

impl<'a> VarInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VarParams) -> Self {
        Self {
            data: VarData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VarParams) -> Self {
        Self {
            data: VarData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VarParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_nbdev(&self) -> f64 {
        self.params.nbdev.unwrap_or(1.0)
    }
}

// -- Builder --

#[derive(Copy, Clone, Debug)]
pub struct VarBuilder {
    period: Option<usize>,
    nbdev: Option<f64>,
    kernel: Kernel,
}

impl Default for VarBuilder {
    fn default() -> Self {
        Self {
            period: None,
            nbdev: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VarBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<VarOutput, VarError> {
        let p = VarParams { period: self.period, nbdev: self.nbdev };
        let i = VarInput::from_candles(c, "close", p);
        var_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VarOutput, VarError> {
        let p = VarParams { period: self.period, nbdev: self.nbdev };
        let i = VarInput::from_slice(d, p);
        var_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VarStream, VarError> {
        let p = VarParams { period: self.period, nbdev: self.nbdev };
        VarStream::try_new(p)
    }
}

// -- Errors --

#[derive(Debug, Error)]
pub enum VarError {
    #[error("var: All values are NaN.")]
    AllValuesNaN,
    #[error("var: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("var: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("var: nbdev is NaN or infinite: {nbdev}")]
    InvalidNbdev { nbdev: f64 },
}

// -- Indicator functions --

#[inline]
pub fn var(input: &VarInput) -> Result<VarOutput, VarError> {
    var_with_kernel(input, Kernel::Auto)
}

pub fn var_with_kernel(input: &VarInput, kernel: Kernel) -> Result<VarOutput, VarError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(VarError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    let nbdev = input.get_nbdev();

    if period == 0 || period > len {
        return Err(VarError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(VarError::NotEnoughValidData { needed: period, valid: len - first });
    }
    if nbdev.is_nan() || nbdev.is_infinite() {
        return Err(VarError::InvalidNbdev { nbdev });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                var_scalar(data, period, first, nbdev, &mut vec![f64::NAN; len])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                var_avx2(data, period, first, nbdev, &mut vec![f64::NAN; len])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                var_avx512(data, period, first, nbdev, &mut vec![f64::NAN; len])
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub fn var_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) -> Result<VarOutput, VarError> {
    let len = data.len();
    let nbdev2 = nbdev * nbdev;
    let inv_p = 1.0 / (period as f64);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for &v in &data[first..first+period] {
        sum += v;
        sum_sq += v*v;
    }

    out[first + period - 1] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;

    for i in (first + period)..len {
        let old = data[i - period];
        let new = data[i];
        sum += new - old;
        sum_sq += new * new - old * old;
        out[i] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;
    }

    Ok(VarOutput { values: out.to_vec() })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn var_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) -> Result<VarOutput, VarError> {
    // Stub: points to scalar logic for API parity
    var_scalar(data, period, first, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn var_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) -> Result<VarOutput, VarError> {
    // Stub: points to scalar logic for API parity
    var_scalar(data, period, first, nbdev, out)
}

// --- Row variants for batch ---

#[inline(always)]
pub unsafe fn var_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let nbdev2 = nbdev * nbdev;
    let inv_p = 1.0 / (period as f64);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &v in &data[first..first+period] {
        sum += v;
        sum_sq += v*v;
    }
    out[first + period - 1] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;

    for i in (first + period)..len {
        let old = data[i - period];
        let new = data[i];
        sum += new - old;
        sum_sq += new * new - old * old;
        out[i] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    var_row_scalar(data, first, period, stride, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        var_row_avx512_short(data, first, period, stride, nbdev, out);
    } else {
        var_row_avx512_long(data, first, period, stride, nbdev, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    var_row_scalar(data, first, period, stride, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    var_row_scalar(data, first, period, stride, nbdev, out)
}

// --- Batch support ---

#[derive(Clone, Debug)]
pub struct VarBatchRange {
    pub period: (usize, usize, usize),
    pub nbdev: (f64, f64, f64),
}

impl Default for VarBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 60, 1),
            nbdev: (1.0, 1.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VarBatchBuilder {
    range: VarBatchRange,
    kernel: Kernel,
}

impl VarBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<VarBatchOutput, VarError> {
        var_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VarBatchOutput, VarError> {
        VarBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VarBatchOutput, VarError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VarBatchOutput, VarError> {
        VarBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct VarBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VarParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VarBatchOutput {
    pub fn row_for_params(&self, p: &VarParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.nbdev.unwrap_or(1.0) - p.nbdev.unwrap_or(1.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &VarParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// -- Grid

#[inline(always)]
fn expand_grid(r: &VarBatchRange) -> Vec<VarParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 { v.push(x); x += step; }
        v
    }
    let periods = axis_usize(r.period);
    let nbdevs = axis_f64(r.nbdev);
    let mut out = Vec::with_capacity(periods.len() * nbdevs.len());
    for &p in &periods {
        for &n in &nbdevs {
            out.push(VarParams { period: Some(p), nbdev: Some(n) });
        }
    }
    out
}

// -- Batch Inner

#[inline(always)]
pub fn var_batch_slice(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
) -> Result<VarBatchOutput, VarError> {
    var_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn var_batch_par_slice(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
) -> Result<VarBatchOutput, VarError> {
    var_batch_inner(data, sweep, kern, true)
}

fn round_up8(x: usize) -> usize { (x + 7) & !7 }

#[inline(always)]
fn var_batch_inner(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VarBatchOutput, VarError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VarError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(VarError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();
    if data.len() - first < max_p {
        return Err(VarError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let nbdev = combos[row].nbdev.unwrap();
        match kern {
            Kernel::Scalar => var_row_scalar(data, first, period, max_p, nbdev, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => var_row_avx2(data, first, period, max_p, nbdev, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => var_row_avx512(data, first, period, max_p, nbdev, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        values.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() { do_row(row, slice); }
    }
    Ok(VarBatchOutput { values, combos, rows, cols })
}

pub fn var_batch_with_kernel(
    data: &[f64],
    sweep: &VarBatchRange,
    k: Kernel,
) -> Result<VarBatchOutput, VarError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(VarError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    var_batch_par_slice(data, sweep, simd)
}

// --- Streaming ---

#[derive(Debug, Clone)]
pub struct VarStream {
    period: usize,
    nbdev: f64,
    buffer: Vec<f64>,
    sum: f64,
    sum_sq: f64,
    head: usize,
    filled: bool,
}
impl VarStream {
    pub fn try_new(params: VarParams) -> Result<Self, VarError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(VarError::InvalidPeriod { period, data_len: 0 });
        }
        let nbdev = params.nbdev.unwrap_or(1.0);
        if nbdev.is_nan() || nbdev.is_infinite() {
            return Err(VarError::InvalidNbdev { nbdev });
        }
        Ok(Self {
            period,
            nbdev,
            buffer: vec![f64::NAN; period],
            sum: 0.0,
            sum_sq: 0.0,
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled {
            self.sum += value;
            self.sum_sq += value * value;
            if self.head == 0 {
                self.filled = true;
            } else {
                return None;
            }
        } else {
            self.sum += value - old;
            self.sum_sq += value * value - old * old;
        }
        let inv_p = 1.0 / self.period as f64;
        let mean = self.sum * inv_p;
        let mean_sq = self.sum_sq * inv_p;
        Some((mean_sq - mean * mean) * self.nbdev * self.nbdev)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_var_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VarParams { period: None, nbdev: None };
        let input = VarInput::from_candles(&candles, "close", default_params);
        let output = var_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_var_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VarInput::from_candles(&candles, "close", VarParams::default());
        let var_result = var_with_kernel(&input, kernel)?;
        assert_eq!(var_result.values.len(), candles.close.len());
        let expected_last_five = [
            350987.4081501961,
            348493.9183540344,
            302611.06121110916,
            106092.2499871254,
            121941.35202789307,
        ];
        let start_index = var_result.values.len() - 5;
        let result_last_five = &var_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "[{}] VAR mismatch at idx {}: got {}, expected {}",
                test_name, i, value, expected_value
            );
        }
        Ok(())
    }

    fn check_var_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VarInput::with_default_candles(&candles);
        match input.data {
            VarData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected VarData::Candles"),
        }
        let output = var_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_var_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = VarParams { period: Some(0), nbdev: None };
        let input = VarInput::from_slice(&input_data, params);
        let res = var_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VAR should fail with zero period", test_name);
        Ok(())
    }

    fn check_var_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = VarParams { period: Some(10), nbdev: None };
        let input = VarInput::from_slice(&data_small, params);
        let res = var_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VAR should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_var_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = VarParams { period: Some(14), nbdev: None };
        let input = VarInput::from_slice(&single_point, params);
        let res = var_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VAR should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_var_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = VarParams { period: Some(14), nbdev: Some(1.0) };
        let first_input = VarInput::from_candles(&candles, "close", first_params);
        let first_result = var_with_kernel(&first_input, kernel)?;
        let second_params = VarParams { period: Some(14), nbdev: Some(1.0) };
        let second_input = VarInput::from_slice(&first_result.values, second_params);
        let second_result = var_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_var_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VarInput::from_candles(&candles, "close", VarParams { period: Some(14), nbdev: None });
        let res = var_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 30 {
            for (i, &val) in res.values[30..].iter().enumerate() {
                assert!(!val.is_nan(), "[{}] Found unexpected NaN at out-index {}", test_name, 30 + i);
            }
        }
        Ok(())
    }

    fn check_var_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let nbdev = 1.0;
        let input = VarInput::from_candles(
            &candles,
            "close",
            VarParams { period: Some(period), nbdev: Some(nbdev) }
        );
        let batch_output = var_with_kernel(&input, kernel)?.values;
        let mut stream = VarStream::try_new(VarParams { period: Some(period), nbdev: Some(nbdev) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(var_val) => stream_values.push(var_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(diff < 1e-6, "[{}] VAR streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_var_tests {
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
    generate_all_var_tests!(
        check_var_partial_params,
        check_var_accuracy,
        check_var_default_candles,
        check_var_zero_period,
        check_var_period_exceeds_length,
        check_var_very_small_dataset,
        check_var_reinput,
        check_var_nan_handling,
        check_var_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = VarBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = VarParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            350987.4081501961,
            348493.9183540344,
            302611.06121110916,
            106092.2499871254,
            121941.35202789307,
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
