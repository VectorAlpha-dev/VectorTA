//! # Reflex
//!
//! An indicator (attributed to John Ehlers) designed to detect turning points in a time
//! series by comparing a 2-pole filtered version of the data to a projected slope over
//! a specified window (`period`). It then adjusts its output (`Reflex`) based on the
//! difference between predicted and past values, normalized by a rolling measure of
//! variance. Includes batch/grid operation, builder APIs, and supports AVX2/AVX512 (stubbed).
//!
//! ## Parameters
//! - **period**: The window size used for measuring and predicting the slope (must be ≥ 2).
//!
//! ## Errors
//! - **NoData**: reflex: No data provided (empty slice).
//! - **InvalidPeriod**: reflex: `period` < 2.
//! - **NotEnoughData**: reflex: The available data is shorter than `period`.
//! - **AllValuesNaN**: reflex: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(ReflexOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(ReflexError)`** otherwise.

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

impl<'a> AsRef<[f64]> for ReflexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ReflexData::Slice(slice) => slice,
            ReflexData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ReflexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ReflexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ReflexParams {
    pub period: Option<usize>,
}

impl Default for ReflexParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct ReflexInput<'a> {
    pub data: ReflexData<'a>,
    pub params: ReflexParams,
}

impl<'a> ReflexInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ReflexParams) -> Self {
        Self {
            data: ReflexData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ReflexParams) -> Self {
        Self {
            data: ReflexData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ReflexParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ReflexBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ReflexBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ReflexBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ReflexOutput, ReflexError> {
        let p = ReflexParams { period: self.period };
        let i = ReflexInput::from_candles(c, "close", p);
        reflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ReflexOutput, ReflexError> {
        let p = ReflexParams { period: self.period };
        let i = ReflexInput::from_slice(d, p);
        reflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ReflexStream, ReflexError> {
        let p = ReflexParams { period: self.period };
        ReflexStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ReflexError {
    #[error("reflex: No data available for Reflex.")]
    NoData,
    #[error("reflex: Reflex period must be >=2. Provided period was {period}")]
    InvalidPeriod { period: usize },
    #[error("reflex: Not enough data: needed {needed}, found {found}")]
    NotEnoughData { needed: usize, found: usize },
    #[error("reflex: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn reflex(input: &ReflexInput) -> Result<ReflexOutput, ReflexError> {
    reflex_with_kernel(input, Kernel::Auto)
}

pub fn reflex_with_kernel(input: &ReflexInput, kernel: Kernel) -> Result<ReflexOutput, ReflexError> {
    let data: &[f64] = match &input.data {
        ReflexData::Candles { candles, source } => source_type(candles, source),
        ReflexData::Slice(sl) => sl,
    };
    let first = data.iter().position(|x| !x.is_nan()).ok_or(ReflexError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if len == 0 {
        return Err(ReflexError::NoData);
    }
    if period < 2 {
        return Err(ReflexError::InvalidPeriod { period });
    }
    if period > len {
        return Err(ReflexError::NotEnoughData {
            needed: period,
            found: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![0.0; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                reflex_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                reflex_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                reflex_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(ReflexOutput { values: out })
}

#[inline]
pub fn reflex_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let half_period = (period / 2).max(1);
    let a = (-1.414_f64 * std::f64::consts::PI / half_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * std::f64::consts::PI / half_period as f64).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    let mut ssf = vec![0.0; len];
    let mut ms = vec![0.0; len];
    let mut sums = vec![0.0; len];

    if len > 0 {
        ssf[0] = data[0];
    }
    if len > 1 {
        ssf[1] = data[1];
    }
    let period_f = period as f64;

    for i in 2..len {
        let d_i = data[i];
        let d_im1 = data[i - 1];
        let prev_ssf1 = ssf[i - 1];
        let prev_ssf2 = ssf[i - 2];
        let ssf_i = c * (d_i + d_im1) + b * prev_ssf1 - a_sq * prev_ssf2;
        ssf[i] = ssf_i;

        if i >= period {
            let slope = (ssf[i - period] - ssf_i) / period_f;
            let mut my_sum = 0.0;
            for t in 1..=period {
                let pred = ssf_i + slope * (t as f64);
                let past = ssf[i - t];
                my_sum += pred - past;
            }
            my_sum /= period_f;
            sums[i] = my_sum;
            let ms_im1 = ms[i - 1];
            let my_sum_sq = my_sum * my_sum;
            let ms_i = 0.04 * my_sum_sq + 0.96 * ms_im1;
            ms[i] = ms_i;

            out[i] = if ms_i > 0.0 {
                my_sum / ms_i.sqrt()
            } else {
                0.0
            };
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn reflex_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn reflex_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { reflex_avx512_short(data, period, first, out) }
    } else {
        unsafe { reflex_avx512_long(data, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn reflex_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn reflex_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

// --- Streaming API ---

#[derive(Debug, Clone)]
pub struct ReflexStream {
    period: usize,
    buffer: Vec<f64>,
    ssf: Vec<f64>,
    ms: Vec<f64>,
    sums: Vec<f64>,
    pos: usize,
    filled: bool,
}

impl ReflexStream {
    pub fn try_new(params: ReflexParams) -> Result<Self, ReflexError> {
        let period = params.period.unwrap_or(20);
        if period < 2 {
            return Err(ReflexError::InvalidPeriod { period });
        }
        Ok(Self {
            period,
            buffer: vec![0.0; period],
            ssf: Vec::new(),
            ms: vec![0.0; period],
            sums: vec![0.0; period],
            pos: 0,
            filled: false,
        })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let len = self.buffer.len();
        self.buffer[self.pos] = value;
        self.pos = (self.pos + 1) % len;
        if !self.filled && self.pos == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        let mut tmp_buf;
        let buf: &[f64] = if self.pos == 0 {
            &self.buffer[..]
        } else {
            tmp_buf = self.buffer[self.pos..].to_vec();
            tmp_buf.extend_from_slice(&self.buffer[..self.pos]);
            &tmp_buf
        };
        let mut out = vec![0.0; len];
        reflex_scalar(buf, self.period, 0, &mut out);
        Some(out[len - 1])
    }
}

// --- Batch/grid API ---

#[derive(Clone, Debug)]
pub struct ReflexBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ReflexBatchRange {
    fn default() -> Self {
        Self { period: (20, 20, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReflexBatchBuilder {
    range: ReflexBatchRange,
    kernel: Kernel,
}

impl ReflexBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<ReflexBatchOutput, ReflexError> {
        reflex_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ReflexBatchOutput, ReflexError> {
        ReflexBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ReflexBatchOutput, ReflexError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ReflexBatchOutput, ReflexError> {
        ReflexBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn reflex_batch_with_kernel(
    data: &[f64],
    sweep: &ReflexBatchRange,
    k: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(ReflexError::InvalidPeriod { period: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    reflex_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ReflexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ReflexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ReflexBatchOutput {
    pub fn row_for_params(&self, p: &ReflexParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
        })
    }
    pub fn values_for(&self, p: &ReflexParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ReflexBatchRange) -> Vec<ReflexParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ReflexParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn reflex_batch_slice(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    reflex_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn reflex_batch_par_slice(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    reflex_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn reflex_batch_inner(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ReflexBatchOutput, ReflexError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ReflexError::InvalidPeriod { period: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(ReflexError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ReflexError::NotEnoughData { needed: max_p, found: data.len() - first });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![0.0; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => reflex_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => reflex_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => reflex_row_avx512(data, first, period, out_row),
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

    Ok(ReflexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn reflex_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        reflex_row_avx512_short(data, first, period, out);
    } else {
        reflex_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

// -- Test coverage macros: ALMA parity --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_reflex_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ReflexParams { period: None };
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let output = reflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = ReflexParams { period: Some(14) };
        let input2 = ReflexInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = reflex_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = ReflexParams { period: Some(30) };
        let input3 = ReflexInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = reflex_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());
        Ok(())
    }

    fn check_reflex_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ReflexParams::default();
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let result = reflex_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let len = result.values.len();
        let expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
        ];
        let start_idx = len - 5;
        let last_five = &result.values[start_idx..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-7,
                "[{}] Reflex mismatch at idx {}: got {}, expected {}",
                test_name, i, val, exp
            );
        }
        Ok(())
    }

    fn check_reflex_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ReflexInput::with_default_candles(&candles);
        match input.data {
            ReflexData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ReflexData::Candles"),
        }
        let output = reflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_reflex_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(0) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Reflex should fail with zero period", test_name);
        Ok(())
    }

    fn check_reflex_period_less_than_two(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(1) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Reflex should fail with period<2", test_name);
        Ok(())
    }

    fn check_reflex_very_small_data_set(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = ReflexParams { period: Some(2) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Reflex should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_reflex_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ReflexParams { period: Some(14) };
        let first_input = ReflexInput::from_candles(&candles, "close", first_params);
        let first_result = reflex_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = ReflexParams { period: Some(10) };
        let second_input = ReflexInput::from_slice(&first_result.values, second_params);
        let second_result = reflex_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 14..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
        Ok(())
    }

    fn check_reflex_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let params = ReflexParams { period: Some(period) };
        let input = ReflexInput::from_candles(&candles, "close", params);
        let result = reflex_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > period {
            for i in period..result.values.len() {
                assert!(
                    result.values[i].is_finite(),
                    "[{}] Unexpected NaN at index {}",
                    test_name, i
                );
            }
        }
        Ok(())
    }

    fn check_reflex_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let params = ReflexParams { period: Some(period) };
        let input = ReflexInput::from_candles(&candles, "close", params.clone());
        let batch_output = reflex_with_kernel(&input, kernel)?.values;
        let mut stream = ReflexStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(v) => stream_values.push(v),
                None => stream_values.push(0.0),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Reflex streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_reflex_tests {
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

    generate_all_reflex_tests!(
        check_reflex_partial_params,
        check_reflex_accuracy,
        check_reflex_default_candles,
        check_reflex_zero_period,
        check_reflex_period_less_than_two,
        check_reflex_very_small_data_set,
        check_reflex_reinput,
        check_reflex_nan_handling,
        check_reflex_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = ReflexBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = ReflexParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
