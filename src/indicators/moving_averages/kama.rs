//! # Kaufman Adaptive Moving Average (KAMA)
//!
//! An adaptive moving average that dynamically adjusts its smoothing factor
//! based on price noise or volatility. When price movements are relatively stable,
//! KAMA becomes smoother, filtering out minor fluctuations. Conversely, in more
//! volatile or trending periods, KAMA becomes more reactive, aiming to catch the
//! prevailing trend sooner.
//!
//! ## Parameters
//! - **period**: Core lookback length for the KAMA calculation (defaults to 30).
//!
//! ## Errors
//! - **NoData**: kama: No input data was provided.
//! - **AllValuesNaN**: kama: All input data is `NaN`.
//! - **InvalidPeriod**: kama: `period` is zero or exceeds the data length.
//! - **NotEnoughData**: kama: Not enough data to calculate KAMA for the requested `period`.
//!
//! ## Returns
//! - **`Ok(KamaOutput)`** on success, containing a `Vec<f64>` with length matching the input.
//! - **`Err(KamaError)`** otherwise.

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

#[derive(Debug, Clone)]
pub enum KamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KamaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KamaParams {
    pub period: Option<usize>,
}

impl Default for KamaParams {
    fn default() -> Self {
        KamaParams { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct KamaInput<'a> {
    pub data: KamaData<'a>,
    pub params: KamaParams,
}

impl<'a> AsRef<[f64]> for KamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            KamaData::Slice(slice) => slice,
            KamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> KamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: KamaParams) -> Self {
        Self {
            data: KamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: KamaParams) -> Self {
        Self {
            data: KamaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", KamaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KamaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for KamaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KamaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<KamaOutput, KamaError> {
        let p = KamaParams { period: self.period };
        let i = KamaInput::from_candles(c, "close", p);
        kama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<KamaOutput, KamaError> {
        let p = KamaParams { period: self.period };
        let i = KamaInput::from_slice(d, p);
        kama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<KamaStream, KamaError> {
        let p = KamaParams { period: self.period };
        KamaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum KamaError {
    #[error("kama: No data provided for KAMA.")]
    NoData,
    #[error("kama: All values are NaN.")]
    AllValuesNaN,
    #[error("kama: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughData { needed: usize, valid: usize },
}

#[inline]
pub fn kama(input: &KamaInput) -> Result<KamaOutput, KamaError> {
    kama_with_kernel(input, Kernel::Auto)
}

pub fn kama_with_kernel(input: &KamaInput, kernel: Kernel) -> Result<KamaOutput, KamaError> {
    let data: &[f64] = match &input.data {
        KamaData::Candles { candles, source } => source_type(candles, source),
        KamaData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(KamaError::NoData);
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(KamaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(KamaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(KamaError::NotEnoughData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kama_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kama_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kama_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(KamaOutput { values: out })
}

#[inline]
pub fn kama_scalar(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    let len = data.len();
    let lookback = period.saturating_sub(1);

    let const_max = 2.0 / (30.0 + 1.0);
    let const_diff = (2.0 / (2.0 + 1.0)) - const_max;

    let mut sum_roc1 = 0.0;
    let mut today = first_valid;
    for i in 0..=lookback {
        sum_roc1 += (data[today + i + 1] - data[today + i]).abs();
    }

    let initial_idx = today + lookback + 1;
    let mut prev_kama = data[initial_idx];
    out[initial_idx] = prev_kama;

    let mut trailing_idx = today;
    let mut trailing_value = data[trailing_idx];

    for i in (initial_idx + 1)..len {
        let price = data[i];

        sum_roc1 -= (data[trailing_idx + 1] - trailing_value).abs();

        sum_roc1 += (price - data[i - 1]).abs();

        trailing_value = data[trailing_idx + 1];
        trailing_idx += 1;

        let direction = (price - data[trailing_idx]).abs();
        let er = if sum_roc1 == 0.0 { 0.0 } else { direction / sum_roc1 };
        let sc = (er * const_diff + const_max).powi(2);
        prev_kama += (price - prev_kama) * sc;
        out[i] = prev_kama;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kama_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kama_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kama_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kama_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first_valid, out)
}

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct KamaStream {
    period: usize,
    buffer: VecDeque<f64>,
    prev_kama: f64,
    sum_roc1: f64,
    const_max: f64,
    const_diff: f64,
}

impl KamaStream {
    pub fn try_new(params: KamaParams) -> Result<Self, KamaError> {
        let period = params.period.unwrap_or(30);
        if period == 0 {
            return Err(KamaError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: VecDeque::with_capacity(period + 1),
            prev_kama: 0.0,
            sum_roc1: 0.0,
            const_max: 2.0 / (30.0 + 1.0),
            const_diff: (2.0 / (2.0 + 1.0)) - (2.0 / (30.0 + 1.0)),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.buffer.len() < self.period {
            self.buffer.push_back(value);
            return None;
        }

        if self.buffer.len() == self.period {
            self.sum_roc1 = 0.0;
            for i in 0..(self.period - 1) {
                let a = self.buffer[i];
                let b = self.buffer[i + 1];
                self.sum_roc1 += (b - a).abs();
            }
            if let Some(&last) = self.buffer.back() {
                self.sum_roc1 += (value - last).abs();
            }

            self.prev_kama = value;
            self.buffer.push_back(value);
            return Some(self.prev_kama);
        }

        let old_front = self.buffer.pop_front().unwrap();
        let new_front = *self.buffer.front().unwrap();

        self.sum_roc1 -= (new_front - old_front).abs();

        let last = *self.buffer.back().unwrap();
        self.sum_roc1 += (value - last).abs();

        let direction = (value - new_front).abs();

        let er = if self.sum_roc1 == 0.0 {
            0.0
        } else {
            direction / self.sum_roc1
        };
        let sc = (er * self.const_diff + self.const_max).powi(2);
        self.prev_kama += (value - self.prev_kama) * sc;

        self.buffer.push_back(value);
        Some(self.prev_kama)
    }
}


#[derive(Clone, Debug)]
pub struct KamaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for KamaBatchRange {
    fn default() -> Self {
        Self { period: (30, 100, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KamaBatchBuilder {
    range: KamaBatchRange,
    kernel: Kernel,
}

impl KamaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<KamaBatchOutput, KamaError> {
        kama_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<KamaBatchOutput, KamaError> {
        KamaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KamaBatchOutput, KamaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<KamaBatchOutput, KamaError> {
        KamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn kama_batch_with_kernel(
    data: &[f64],
    sweep: &KamaBatchRange,
    k: Kernel,
) -> Result<KamaBatchOutput, KamaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(KamaError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    kama_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KamaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<KamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl KamaBatchOutput {
    pub fn row_for_params(&self, p: &KamaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
    }
    pub fn values_for(&self, p: &KamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &KamaBatchRange) -> Vec<KamaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods.into_iter().map(|p| KamaParams { period: Some(p) }).collect()
}

#[inline(always)]
pub fn kama_batch_slice(
    data: &[f64],
    sweep: &KamaBatchRange,
    kern: Kernel,
) -> Result<KamaBatchOutput, KamaError> {
    kama_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn kama_batch_par_slice(
    data: &[f64],
    sweep: &KamaBatchRange,
    kern: Kernel,
) -> Result<KamaBatchOutput, KamaError> {
    kama_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn kama_batch_inner(
    data: &[f64],
    sweep: &KamaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<KamaBatchOutput, KamaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KamaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(KamaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(KamaError::NotEnoughData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => kama_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => kama_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => kama_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        values.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(KamaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn kama_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first, out)
}

#[inline(always)]
pub fn expand_grid_kama(r: &KamaBatchRange) -> Vec<KamaParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_kama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = KamaParams { period: None };
        let input = KamaInput::from_candles(&candles, "close", default_params);
        let output = kama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_kama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KamaInput::with_default_candles(&candles);
        let result = kama_with_kernel(&input, kernel)?;
        let expected_last_five = [
            60234.925553804125,
            60176.838757545665,
            60115.177367962766,
            60071.37070833558,
            59992.79386218023,
        ];
        assert!(
            result.values.len() >= 5,
            "Expected at least 5 values to compare"
        );
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "KAMA output length does not match input length"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "KAMA mismatch at last-five index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_kama_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KamaInput::with_default_candles(&candles);
        match input.data {
            KamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected KamaData::Candles"),
        }
        let output = kama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_kama_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = KamaParams { period: Some(0) };
        let input = KamaInput::from_slice(&input_data, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] KAMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_kama_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = KamaParams { period: Some(10) };
        let input = KamaInput::from_slice(&data_small, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] KAMA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_kama_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = KamaParams { period: Some(30) };
        let input = KamaInput::from_slice(&single_point, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] KAMA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_kama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = KamaParams { period: Some(30) };
        let first_input = KamaInput::from_candles(&candles, "close", first_params);
        let first_result = kama_with_kernel(&first_input, kernel)?;
        let second_params = KamaParams { period: Some(10) };
        let second_input = KamaInput::from_slice(&first_result.values, second_params);
        let second_result = kama_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_kama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KamaInput::from_candles(
            &candles,
            "close",
            KamaParams { period: Some(30) },
        );
        let res = kama_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        for val in res.values.iter().skip(30) {
            assert!(val.is_finite());
        }
        Ok(())
    }

    fn check_kama_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 30;
        let input = KamaInput::from_candles(
            &candles,
            "close",
            KamaParams { period: Some(period) },
        );
        let batch_output = kama_with_kernel(&input, kernel)?.values;
        let mut stream = KamaStream::try_new(KamaParams { period: Some(period) })?;
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
                "[{}] KAMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_kama_tests {
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
    generate_all_kama_tests!(
        check_kama_partial_params,
        check_kama_accuracy,
        check_kama_default_candles,
        check_kama_zero_period,
        check_kama_period_exceeds_length,
        check_kama_very_small_dataset,
        check_kama_reinput,
        check_kama_nan_handling,
        check_kama_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KamaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = KamaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            60234.925553804125,
            60176.838757545665,
            60115.177367962766,
            60071.37070833558,
            59992.79386218023,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
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
