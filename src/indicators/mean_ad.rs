//! # Mean Absolute Deviation (MeanAd)
//!
//! Computes the mean absolute deviation as a rolling statistic. The indicator is implemented
//! with a two-pass method: first, a rolling mean is computed, then a rolling mean of the
//! absolute deviations from that mean. The `period` parameter controls the window size.
//!
//! ## Parameters
//! - **period**: The window size (number of data points, default: 5).
//!
//! ## Errors
//! - **EmptyData**: mean_ad: Input data slice is empty.
//! - **InvalidPeriod**: mean_ad: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: mean_ad: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
//! - **AllValuesNaN**: mean_ad: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(MeanAdOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(MeanAdError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MeanAdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MeanAdData::Slice(slice) => slice,
            MeanAdData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MeanAdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MeanAdOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MeanAdParams {
    pub period: Option<usize>,
}

impl Default for MeanAdParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MeanAdInput<'a> {
    pub data: MeanAdData<'a>,
    pub params: MeanAdParams,
}

impl<'a> MeanAdInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MeanAdParams) -> Self {
        Self {
            data: MeanAdData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MeanAdParams) -> Self {
        Self {
            data: MeanAdData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MeanAdParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MeanAdBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MeanAdBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MeanAdBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MeanAdOutput, MeanAdError> {
        let p = MeanAdParams { period: self.period };
        let i = MeanAdInput::from_candles(c, "close", p);
        mean_ad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MeanAdOutput, MeanAdError> {
        let p = MeanAdParams { period: self.period };
        let i = MeanAdInput::from_slice(d, p);
        mean_ad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MeanAdStream, MeanAdError> {
        let p = MeanAdParams { period: self.period };
        MeanAdStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MeanAdError {
    #[error("mean_ad: Empty data provided.")]
    EmptyData,
    #[error("mean_ad: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mean_ad: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mean_ad: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn mean_ad(input: &MeanAdInput) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_with_kernel(input, Kernel::Auto)
}

pub fn mean_ad_with_kernel(input: &MeanAdInput, kernel: Kernel) -> Result<MeanAdOutput, MeanAdError> {
    let data: &[f64] = match &input.data {
        MeanAdData::Candles { candles, source } => source_type(candles, source),
        MeanAdData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(MeanAdError::EmptyData);
    }

    let period = input.get_period();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(MeanAdError::AllValuesNaN)?;

    if period == 0 || period > data.len() {
        return Err(MeanAdError::InvalidPeriod { period, data_len: data.len() });
    }
    if (data.len() - first) < period {
        return Err(MeanAdError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                mean_ad_scalar(data, period, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mean_ad_avx2(data, period, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mean_ad_avx512(data, period, first)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn mean_ad_scalar(data: &[f64], period: usize, first: usize) -> Result<MeanAdOutput, MeanAdError> {
    let mut out = vec![f64::NAN; data.len()];
    let mut rolling_mean = vec![f64::NAN; data.len()];
    if first + period > data.len() {
        return Ok(MeanAdOutput { values: out });
    }
    let mut sum = 0.0;
    for &v in &data[first..(first + period)] {
        sum += v;
    }
    rolling_mean[first + period - 1] = sum / (period as f64);

    for i in (first + period)..data.len() {
        sum += data[i] - data[i - period];
        rolling_mean[i] = sum / (period as f64);
    }

    let mut abs_diff = vec![f64::NAN; data.len()];
    for i in (first + period - 1)..data.len() {
        if !rolling_mean[i].is_nan() {
            abs_diff[i] = (data[i] - rolling_mean[i]).abs();
        }
    }

    let mut mad_sum = 0.0;
    for &v in &abs_diff[first + period - 1..first + period - 1 + period] {
        mad_sum += v;
    }
    out[first + 2 * period - 2] = mad_sum / (period as f64);

    for i in (first + 2 * period - 1)..data.len() {
        mad_sum += abs_diff[i] - abs_diff[i - period];
        out[i] = mad_sum / (period as f64);
    }

    Ok(MeanAdOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mean_ad_avx2(data: &[f64], period: usize, first: usize) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mean_ad_avx512(data: &[f64], period: usize, first: usize) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mean_ad_avx512_short(data: &[f64], period: usize, first: usize) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mean_ad_avx512_long(data: &[f64], period: usize, first: usize) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[inline(always)]
pub fn mean_ad_batch_with_kernel(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    k: Kernel,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MeanAdError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    mean_ad_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MeanAdBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MeanAdBatchRange {
    fn default() -> Self {
        Self { period: (5, 50, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeanAdBatchBuilder {
    range: MeanAdBatchRange,
    kernel: Kernel,
}

impl MeanAdBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<MeanAdBatchOutput, MeanAdError> {
        mean_ad_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MeanAdBatchOutput, MeanAdError> {
        MeanAdBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MeanAdBatchOutput, MeanAdError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MeanAdBatchOutput, MeanAdError> {
        MeanAdBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MeanAdBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MeanAdParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MeanAdBatchOutput {
    pub fn row_for_params(&self, p: &MeanAdParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &MeanAdParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MeanAdBatchRange) -> Vec<MeanAdParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods.into_iter().map(|p| MeanAdParams { period: Some(p) }).collect()
}

#[inline(always)]
pub fn mean_ad_batch_slice(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    kern: Kernel,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    mean_ad_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mean_ad_batch_par_slice(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    kern: Kernel,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    mean_ad_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn mean_ad_batch_inner(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MeanAdError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(MeanAdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MeanAdError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => mean_ad_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mean_ad_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mean_ad_row_avx512(data, first, period, out_row),
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

    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(MeanAdBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn mean_ad_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let mut rolling_mean = vec![f64::NAN; data.len()];
    if first + period > data.len() {
        return;
    }
    let mut sum = 0.0;
    for &v in &data[first..(first + period)] {
        sum += v;
    }
    rolling_mean[first + period - 1] = sum / (period as f64);

    for i in (first + period)..data.len() {
        sum += data[i] - data[i - period];
        rolling_mean[i] = sum / (period as f64);
    }

    let mut abs_diff = vec![f64::NAN; data.len()];
    for i in (first + period - 1)..data.len() {
        if !rolling_mean[i].is_nan() {
            abs_diff[i] = (data[i] - rolling_mean[i]).abs();
        }
    }

    let mut mad_sum = 0.0;
    for &v in &abs_diff[first + period - 1..first + period - 1 + period] {
        mad_sum += v;
    }
    out[first + 2 * period - 2] = mad_sum / (period as f64);

    for i in (first + 2 * period - 1)..data.len() {
        mad_sum += abs_diff[i] - abs_diff[i - period];
        out[i] = mad_sum / (period as f64);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mean_ad_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    mean_ad_row_scalar(data, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mean_ad_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        mean_ad_row_avx512_short(data, first, period, out);
    
        } else {
        mean_ad_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mean_ad_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    mean_ad_row_scalar(data, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mean_ad_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    mean_ad_row_scalar(data, first, period, out);
}

#[derive(Debug, Clone)]
pub struct MeanAdStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    mean_buffer: Vec<f64>,
    mean_head: usize,
    mean_filled: bool,
    mean: f64,
    mad: f64,
}

impl MeanAdStream {
    pub fn try_new(params: MeanAdParams) -> Result<Self, MeanAdError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(MeanAdError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            mean_buffer: vec![f64::NAN; period],
            mean_head: 0,
            mean_filled: false,
            mean: 0.0,
            mad: 0.0,
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
        self.mean = self.buffer.iter().copied().sum::<f64>() / (self.period as f64);
        let deviation = (value - self.mean).abs();
        self.mean_buffer[self.mean_head] = deviation;
        self.mean_head = (self.mean_head + 1) % self.period;
        if !self.mean_filled && self.mean_head == 0 {
            self.mean_filled = true;
        }
        if !self.mean_filled {
            return None;
        }
        self.mad = self.mean_buffer.iter().copied().sum::<f64>() / (self.period as f64);
        Some(self.mad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_mean_ad_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MeanAdParams { period: None };
        let input = MeanAdInput::from_candles(&candles, "close", default_params);
        let output = mean_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mean_ad_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MeanAdInput::from_candles(&candles, "hl2", MeanAdParams { period: Some(5) });
        let result = mean_ad_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let expected_last_five = [
            199.71999999999971,
            104.14000000000087,
            133.4,
            100.54000000000087,
            117.98000000000029,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MeanAd {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_mean_ad_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MeanAdInput::with_default_candles(&candles);
        match input.data {
            MeanAdData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MeanAdData::Candles"),
        }
        let output = mean_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mean_ad_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MeanAdParams { period: Some(0) };
        let input = MeanAdInput::from_slice(&input_data, params);
        let res = mean_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MeanAd should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_mean_ad_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MeanAdParams { period: Some(10) };
        let input = MeanAdInput::from_slice(&data_small, params);
        let res = mean_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MeanAd should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_mean_ad_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MeanAdParams { period: Some(5) };
        let input = MeanAdInput::from_slice(&single_point, params);
        let res = mean_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MeanAd should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_mean_ad_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = MeanAdParams { period: Some(5) };
        let first_input = MeanAdInput::from_candles(&candles, "close", first_params);
        let first_result = mean_ad_with_kernel(&first_input, kernel)?;
        let params2 = MeanAdParams { period: Some(3) };
        let second_input = MeanAdInput::from_slice(&first_result.values, params2);
        let second_result = mean_ad_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_mean_ad_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MeanAdInput::from_candles(&candles, "close", MeanAdParams { period: Some(5) });
        let res = mean_ad_with_kernel(&input, kernel)?;
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

    fn check_mean_ad_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = MeanAdInput::from_candles(&candles, "close", MeanAdParams { period: Some(period) });
        let batch_output = mean_ad_with_kernel(&input, kernel)?.values;
        let mut stream = MeanAdStream::try_new(MeanAdParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(mean_ad_val) => stream_values.push(mean_ad_val),
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
                "[{}] MeanAd streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_mean_ad_tests {
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
    generate_all_mean_ad_tests!(
        check_mean_ad_partial_params,
        check_mean_ad_accuracy,
        check_mean_ad_default_candles,
        check_mean_ad_zero_period,
        check_mean_ad_period_exceeds_length,
        check_mean_ad_very_small_dataset,
        check_mean_ad_reinput,
        check_mean_ad_nan_handling,
        check_mean_ad_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MeanAdBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hl2")?;

        let def = MeanAdParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            199.71999999999971,
            104.14000000000087,
            133.4,
            100.54000000000087,
            117.98000000000029,
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
        }
    }
    gen_batch_tests!(check_batch_default_row);
}
