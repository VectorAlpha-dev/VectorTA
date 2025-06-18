//! # Weighted Moving Average (WMA)
//!
//! A moving average where each data point in the window is assigned a linearly increasing weight. The most recent values carry the highest weights, making the WMA more responsive to new data than a simple moving average.
//!
//! ## Parameters
//! - **period**: Window size (must be >= 2).
//!
//! ## Errors
//! - **AllValuesNaN**: wma: All input data values are `NaN`.
//! - **InvalidPeriod**: wma: `period` is less than 2 or exceeds data length.
//! - **NotEnoughValidData**: wma: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(WmaOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(WmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for WmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WmaData::Slice(slice) => slice,
            WmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WmaParams {
    pub period: Option<usize>,
}

impl Default for WmaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct WmaInput<'a> {
    pub data: WmaData<'a>,
    pub params: WmaParams,
}

impl<'a> WmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: WmaParams) -> Self {
        Self {
            data: WmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: WmaParams) -> Self {
        Self {
            data: WmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", WmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for WmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<WmaOutput, WmaError> {
        let p = WmaParams {
            period: self.period,
        };
        let i = WmaInput::from_candles(c, "close", p);
        wma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<WmaOutput, WmaError> {
        let p = WmaParams {
            period: self.period,
        };
        let i = WmaInput::from_slice(d, p);
        wma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<WmaStream, WmaError> {
        let p = WmaParams {
            period: self.period,
        };
        WmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum WmaError {
    #[error("wma: All values are NaN.")]
    AllValuesNaN,

    #[error("wma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("wma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn wma(input: &WmaInput) -> Result<WmaOutput, WmaError> {
    wma_with_kernel(input, Kernel::Auto)
}

pub fn wma_with_kernel(input: &WmaInput, kernel: Kernel) -> Result<WmaOutput, WmaError> {
    let data: &[f64] = match &input.data {
        WmaData::Candles { candles, source } => source_type(candles, source),
        WmaData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WmaError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();

    if period < 2 || period > len {
        return Err(WmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(WmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                wma_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                wma_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wma_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(WmaOutput { values: out })
}

#[inline]
pub fn wma_scalar(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    let lookback = period - 1;
    let sum_of_weights = (period * (period + 1)) >> 1;
    let divider = sum_of_weights as f64;

    let mut weighted_sum = 0.0;
    let mut plain_sum = 0.0;

    for i in 0..lookback {
        let val = data[first_val + i];
        weighted_sum += (i as f64 + 1.0) * val;
        plain_sum += val;
    }

    let first_wma_idx = first_val + lookback;
    let val = data[first_wma_idx];
    weighted_sum += (period as f64) * val;
    plain_sum += val;

    out[first_wma_idx] = weighted_sum / divider;

    weighted_sum -= plain_sum;
    plain_sum -= data[first_val];

    for i in (first_wma_idx + 1)..data.len() {
        let val = data[i];
        weighted_sum += (period as f64) * val;
        plain_sum += val;

        out[i] = weighted_sum / divider;

        weighted_sum -= plain_sum;
        let old_val = data[i - lookback];
        plain_sum -= old_val;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wma_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    wma_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wma_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    wma_scalar(data, period, first_valid, out)
}

#[inline]
pub fn wma_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    wma_scalar(data, period, first_valid, out)
}

#[inline]
pub fn wma_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    wma_scalar(data, period, first_valid, out)
}

#[inline(always)]
pub fn wma_with_kernel_batch(
    data: &[f64],
    sweep: &WmaBatchRange,
    k: Kernel,
) -> Result<WmaBatchOutput, WmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(WmaError::InvalidPeriod {
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
    wma_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct WmaStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl WmaStream {
    pub fn try_new(params: WmaParams) -> Result<Self, WmaError> {
        let period = params.period.unwrap_or(30);
        if period < 2 {
            return Err(WmaError::InvalidPeriod {
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
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(self.weighted_average())
    }

    #[inline(always)]
    fn weighted_average(&self) -> f64 {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        let mut idx = self.head;
        for i in 0..self.period {
            let weight = (i + 1) as f64;
            let val = self.buffer[idx];
            sum += val * weight;
            weight_sum += weight;
            idx = (idx + 1) % self.period;
        }
        sum / weight_sum
    }
}

#[derive(Clone, Debug)]
pub struct WmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for WmaBatchRange {
    fn default() -> Self {
        Self {
            period: (2, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WmaBatchBuilder {
    range: WmaBatchRange,
    kernel: Kernel,
}

impl WmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<WmaBatchOutput, WmaError> {
        wma_with_kernel_batch(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<WmaBatchOutput, WmaError> {
        WmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<WmaBatchOutput, WmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<WmaBatchOutput, WmaError> {
        WmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct WmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WmaBatchOutput {
    pub fn row_for_params(&self, p: &WmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(30) == p.period.unwrap_or(30)
        })
    }

    pub fn values_for(&self, p: &WmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &WmaBatchRange) -> Vec<WmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(WmaParams {
            period: Some(p),
        });
    }
    out
}

#[inline(always)]
pub fn wma_batch_slice(
    data: &[f64],
    sweep: &WmaBatchRange,
    kern: Kernel,
) -> Result<WmaBatchOutput, WmaError> {
    wma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn wma_batch_par_slice(
    data: &[f64],
    sweep: &WmaBatchRange,
    kern: Kernel,
) -> Result<WmaBatchOutput, WmaError> {
    wma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn wma_batch_inner(
    data: &[f64],
    sweep: &WmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WmaBatchOutput, WmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(WmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(WmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // ---------- 1.  How many leading NaNs each row needs ----------
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    // ---------- 2.  Allocate as MaybeUninit and fill NaN prefixes ----------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 3.  Closure that fills one row ----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Re-interpret this row as &mut [f64] so the kernel can write directly.
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => wma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => wma_row_avx2  (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => wma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- 4.  Run every row ----------
    if parallel {
        raw.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 5.  All rows initialised → transmute to Vec<f64> ----------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(WmaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn wma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        wma_row_avx512_short(data, first, period, out);
    } else {
        wma_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wma_row_scalar(data, first, period, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_wma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = WmaParams { period: None };
        let input = WmaInput::from_candles(&candles, "close", default_params);
        let output = wma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = WmaParams { period: Some(14) };
        let input2 = WmaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = wma_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = WmaParams { period: Some(20) };
        let input3 = WmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = wma_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());
        Ok(())
    }

    fn check_wma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let data = &candles.close;
        let default_params = WmaParams::default();
        let input = WmaInput::from_candles(&candles, "close", default_params);
        let result = wma_with_kernel(&input, kernel)?;

        let expected_last_five = [
            59638.52903225806,
            59563.7376344086,
            59489.4064516129,
            59432.02580645162,
            59350.58279569892,
        ];
        assert!(result.values.len() >= 5, "Not enough WMA values");
        assert_eq!(
            result.values.len(),
            data.len(),
            "WMA output length should match input length"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-6,
                "WMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }
        let period = input.params.period.unwrap_or(30);
        for val in result.values.iter().skip(period - 1) {
            if !val.is_nan() {
                assert!(val.is_finite(), "WMA output should be finite");
            }
        }
        Ok(())
    }

    fn check_wma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WmaInput::with_default_candles(&candles);
        match input.data {
            WmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected WmaData::Candles"),
        }
        let output = wma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_wma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = WmaParams { period: Some(0) };
        let input = WmaInput::from_slice(&input_data, params);
        let res = wma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] WMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_wma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = WmaParams { period: Some(10) };
        let input = WmaInput::from_slice(&data_small, params);
        let res = wma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_wma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = WmaParams { period: Some(9) };
        let input = WmaInput::from_slice(&single_point, params);
        let res = wma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_wma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = WmaParams { period: Some(14) };
        let first_input = WmaInput::from_candles(&candles, "close", first_params);
        let first_result = wma_with_kernel(&first_input, kernel)?;
        let second_params = WmaParams { period: Some(5) };
        let second_input = WmaInput::from_slice(&first_result.values, second_params);
        let second_result = wma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[50..] {
            assert!(!val.is_nan());
        }
        Ok(())
    }

    fn check_wma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = WmaParams { period: Some(14) };
        let input = WmaInput::from_candles(&candles, "close", params);
        let result = wma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_wma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 30;
        let input = WmaInput::from_candles(
            &candles,
            "close",
            WmaParams { period: Some(period) },
        );
        let batch_output = wma_with_kernel(&input, kernel)?.values;

        let mut stream = WmaStream::try_new(WmaParams { period: Some(period) })?;
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
                diff < 1e-8,
                "[{}] WMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_wma_tests {
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

    generate_all_wma_tests!(
        check_wma_partial_params,
        check_wma_accuracy,
        check_wma_default_candles,
        check_wma_zero_period,
        check_wma_period_exceeds_length,
        check_wma_very_small_dataset,
        check_wma_reinput,
        check_wma_nan_handling,
        check_wma_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = WmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = WmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59638.52903225806,
            59563.7376344086,
            59489.4064516129,
            59432.02580645162,
            59350.58279569892,
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
