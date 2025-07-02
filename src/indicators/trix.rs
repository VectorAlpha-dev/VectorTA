//! # TRIX (Triple Exponential Average Oscillator)
//!
//! TRIX is a momentum oscillator derived from a triple-smoothed Exponential Moving Average (EMA),
//! then taking the 1-day Rate-Of-Change (ROC) of that triple EMA (multiplied by 100).
//!
//! ## Parameters
//! - **period**: The EMA window size. Defaults to 18.
//!
//! ## Errors
//! - **EmptyData**: trix: Input data slice is empty.
//! - **InvalidPeriod**: trix: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: trix: Fewer than `3*(period - 1) + 1` valid data points remain
//!   after the first valid index for triple-EMA + 1-bar ROC.
//! - **AllValuesNaN**: trix: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(TrixOutput)`** on success, matching the input length,
//!   with `NaN` until triple-EMA is fully initialized plus 1 bar for the ROC.
//! - **`Err(TrixError)`** otherwise.

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

impl<'a> AsRef<[f64]> for TrixInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrixData::Slice(slice) => slice,
            TrixData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrixData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrixOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrixParams {
    pub period: Option<usize>,
}

impl Default for TrixParams {
    fn default() -> Self {
        Self { period: Some(18) }
    }
}

#[derive(Debug, Clone)]
pub struct TrixInput<'a> {
    pub data: TrixData<'a>,
    pub params: TrixParams,
}

impl<'a> TrixInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TrixParams) -> Self {
        Self {
            data: TrixData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TrixParams) -> Self {
        Self {
            data: TrixData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TrixParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(18)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrixBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TrixBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrixBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TrixOutput, TrixError> {
        let p = TrixParams {
            period: self.period,
        };
        let i = TrixInput::from_candles(c, "close", p);
        trix_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TrixOutput, TrixError> {
        let p = TrixParams {
            period: self.period,
        };
        let i = TrixInput::from_slice(d, p);
        trix_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TrixStream, TrixError> {
        let p = TrixParams {
            period: self.period,
        };
        TrixStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TrixError {
    #[error("trix: Empty data provided.")]
    EmptyData,
    #[error("trix: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("trix: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("trix: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn trix(input: &TrixInput) -> Result<TrixOutput, TrixError> {
    trix_with_kernel(input, Kernel::Auto)
}

pub fn trix_with_kernel(input: &TrixInput, kernel: Kernel) -> Result<TrixOutput, TrixError> {
    let data: &[f64] = match &input.data {
        TrixData::Candles { candles, source } => source_type(candles, source),
        TrixData::Slice(sl) => sl,
    };
    if data.is_empty() {
        return Err(TrixError::EmptyData);
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(TrixError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(TrixError::AllValuesNaN)?;
    let needed = 3 * (period - 1) + 1;
    let valid_len = data.len() - first;
    if valid_len < needed {
        return Err(TrixError::NotEnoughValidData { needed, valid: valid_len });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trix_scalar(data, period, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                trix_avx2(data, period, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trix_avx512(data, period, first)
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn trix_scalar(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<TrixOutput, TrixError> {
    // 1) Build a log‐series of `data`; NaNs propagate automatically.
    let len = data.len();
    let mut log_data = Vec::with_capacity(len);
    for &x in data.iter() {
        log_data.push(if x.is_nan() { f64::NAN } else { x.ln() });
    }

    // 2) triple‐EMA on log_data:
    let ema1 = compute_standard_ema(&log_data, period, first);
    let ema2 = compute_standard_ema(&ema1, period, first + period - 1);
    let ema3 = compute_standard_ema(&ema2, period, first + 2 * (period - 1));

    // 3) Build output array (NaN until (first + 3*(period−1) + 1)), then
    //    out[i] = (ema3[i] − ema3[i−1]) * 10000.0.
    let mut out = vec![f64::NAN; len];
    let triple_ema_start = first + 3 * (period - 1);

    for i in (triple_ema_start + 1)..len {
        let prev = ema3[i - 1];
        let curr = ema3[i];
        if !prev.is_nan() && !curr.is_nan() {
            out[i] = (curr - prev) * 10000.0;
        }
    }

    Ok(TrixOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_avx2(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<TrixOutput, TrixError> {
    trix_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_avx512(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<TrixOutput, TrixError> {
    if period <= 32 {
        trix_avx512_short(data, period, first)
    
        } else {
        trix_avx512_long(data, period, first)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<TrixOutput, TrixError> {
    trix_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<TrixOutput, TrixError> {
    trix_scalar(data, period, first)
}

#[inline]
fn compute_standard_ema(data: &[f64], period: usize, first_valid_idx: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut sum = 0.0;
    for &val in &data[first_valid_idx..(first_valid_idx + period)] {
        sum += val;
    }
    let initial_ema = sum / (period as f64);
    out[first_valid_idx + period - 1] = initial_ema;
    for i in (first_valid_idx + period)..data.len() {
        let prev = out[i - 1];
        if !prev.is_nan() && !data[i].is_nan() {
            out[i] = alpha * data[i] + (1.0 - alpha) * prev;
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct TrixStream {
    period: usize,
    stage: u8,
    buffer1: Vec<f64>,
    buffer2: Vec<f64>,
    buffer3: Vec<f64>,
    head: usize,
    filled: bool,
    prev_ema3: f64,
    initialized: bool,
}

impl TrixStream {
    pub fn try_new(params: TrixParams) -> Result<Self, TrixError> {
        let period = params.period.unwrap_or(18);
        if period == 0 {
            return Err(TrixError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            stage: 0,
            buffer1: vec![f64::NAN; period],
            buffer2: vec![f64::NAN; period],
            buffer3: vec![f64::NAN; period],
            head: 0,
            filled: false,
            prev_ema3: f64::NAN,
            initialized: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // 1) Take ln(value) (or NaN if value was NaN)
        let log_val = if value.is_nan() {
            f64::NAN
        
            } else {
            value.ln()
        };

        // 2) Feed log_val into the first EMA buffer
        self.buffer1[self.head] = log_val;

        // Compute EMA1 stage:
        if self.stage < 1 && self.head == self.period - 1 {
            // We have exactly 'period' logs in buffer1 ⇒ initialize EMA1 with simple average
            let sum1: f64 = self.buffer1.iter().sum();
            let ema1 = sum1 / (self.period as f64);
            self.buffer2[self.head] = ema1;
            self.stage = 1;
        } else if self.stage >= 1 {
            // Ongoing EMA1 update: EMA1[i] = α * log_val + (1−α) * prev_ema1
            let prev_ema1 = self.buffer2[(self.head + self.period - 1) % self.period];
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let ema1 = alpha * log_val + (1.0 - alpha) * prev_ema1;
            self.buffer2[self.head] = ema1;
        }

        // Compute EMA2 stage:
        if self.stage >= 1 {
            if self.stage < 2 && self.head == self.period - 1 {
                // Exactly 'period' EMAs in buffer2 ⇒ initialize EMA2 with simple average
                let sum2: f64 = self.buffer2.iter().sum();
                let ema2 = sum2 / (self.period as f64);
                self.buffer3[self.head] = ema2;
                self.stage = 2;
            } else if self.stage >= 2 {
                // Ongoing EMA2 update: EMA2[i] = α * EMA1[i] + (1−α) * prev_ema2
                let prev_ema2 = self.buffer3[(self.head + self.period - 1) % self.period];
                let alpha = 2.0 / (self.period as f64 + 1.0);
                let ema2 = alpha * self.buffer2[self.head] + (1.0 - alpha) * prev_ema2;
                self.buffer3[self.head] = ema2;
            }
        }

        // Compute EMA3 stage:
        let mut output = None;
        if self.stage >= 2 && self.head == self.period - 1 {
            // Exactly 'period' EMAs in buffer3 ⇒ initialize EMA3
            let sum3: f64 = self.buffer3.iter().sum();
            self.prev_ema3 = sum3 / (self.period as f64);
            self.initialized = true;
        } else if self.stage >= 2 && self.initialized {
            // Ongoing EMA3 update: EMA3[i] = α * EMA2[i] + (1−α) * prev_ema3
            let prev_ema3 = self.prev_ema3;
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let ema3 = alpha * self.buffer3[self.head] + (1.0 - alpha) * prev_ema3;

            // 3) If prev_ema3 and ema3 are both valid, out = (ema3 − prev_ema3)*10000
            if !prev_ema3.is_nan() && !ema3.is_nan() {
                let trix_val = (ema3 - prev_ema3) * 10000.0;
                output = Some(trix_val);
            }
            self.prev_ema3 = ema3;
        }

        // advance head
        self.head = (self.head + 1) % self.period;
        output
    }
}

#[derive(Clone, Debug)]
pub struct TrixBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TrixBatchRange {
    fn default() -> Self {
        Self {
            period: (18, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrixBatchBuilder {
    range: TrixBatchRange,
    kernel: Kernel,
}

impl TrixBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<TrixBatchOutput, TrixError> {
        trix_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TrixBatchOutput, TrixError> {
        TrixBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TrixBatchOutput, TrixError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<TrixBatchOutput, TrixError> {
        TrixBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn trix_batch_with_kernel(
    data: &[f64],
    sweep: &TrixBatchRange,
    k: Kernel,
) -> Result<TrixBatchOutput, TrixError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(TrixError::InvalidPeriod {
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
    trix_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TrixBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrixParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TrixBatchOutput {
    pub fn row_for_params(&self, p: &TrixParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(18) == p.period.unwrap_or(18))
    }
    pub fn values_for(&self, p: &TrixParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TrixBatchRange) -> Vec<TrixParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TrixParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn trix_batch_slice(
    data: &[f64],
    sweep: &TrixBatchRange,
    kern: Kernel,
) -> Result<TrixBatchOutput, TrixError> {
    trix_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn trix_batch_par_slice(
    data: &[f64],
    sweep: &TrixBatchRange,
    kern: Kernel,
) -> Result<TrixBatchOutput, TrixError> {
    trix_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn trix_batch_inner(
    data: &[f64],
    sweep: &TrixBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TrixBatchOutput, TrixError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TrixError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(TrixError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = 3 * (max_p - 1) + 1;
    if data.len() - first < needed {
        return Err(TrixError::NotEnoughValidData {
            needed,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => trix_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => trix_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => trix_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        values.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));

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
    Ok(TrixBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn trix_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    // 1) Build log‐series of data
    let len = data.len();
    let mut log_data = Vec::with_capacity(len);
    for &x in data.iter() {
        log_data.push(if x.is_nan() { f64::NAN } else { x.ln() });
    }

    // 2) triple‐EMA on the log series
    let ema1 = compute_standard_ema(&log_data, period, first);
    let ema2 = compute_standard_ema(&ema1, period, first + period - 1);
    let ema3 = compute_standard_ema(&ema2, period, first + 2 * (period - 1));

    // 3) Δ(ema3) * 10000.0
    let triple_ema_start = first + 3 * (period - 1);
    for i in (triple_ema_start + 1)..len {
        let prev = ema3[i - 1];
        let curr = ema3[i];
        if !prev.is_nan() && !curr.is_nan() {
            out[i] = (curr - prev) * 10000.0;
        }
    }
}



#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    trix_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        trix_row_avx512_short(data, first, period, out)
    
        } else {
        trix_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    trix_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trix_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    trix_row_scalar(data, first, period, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_trix_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = TrixParams { period: None };
        let input_default = TrixInput::from_candles(&candles, "close", default_params);
        let output_default = trix_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period_14 = TrixParams { period: Some(14) };
        let input_period_14 = TrixInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = trix_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.values.len(), candles.close.len());
        let params_custom = TrixParams { period: Some(20) };
        let input_custom = TrixInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = trix_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_trix_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close")?;
        let params = TrixParams { period: Some(18) };
        let input = TrixInput::from_candles(&candles, "close", params);
        let trix_result = trix_with_kernel(&input, kernel)?;
        assert_eq!(trix_result.values.len(), close_prices.len(), "TRIX length mismatch");
        let expected_last_five = [
            -16.03736447,
            -15.92084231,
            -15.76171478,
            -15.53571033,
            -15.34967155,
        ];
        assert!(trix_result.values.len() >= 5, "TRIX length too short");
        let start_index = trix_result.values.len() - 5;
        let result_last_five = &trix_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "TRIX mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_trix_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TrixInput::with_default_candles(&candles);
        match input.data {
            TrixData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TrixData::Candles"),
        }
        Ok(())
    }

    fn check_trix_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TrixParams { period: Some(0) };
        let input = TrixInput::from_slice(&input_data, params);
        let res = trix_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] TRIX should fail with zero period", test_name);
        Ok(())
    }

    fn check_trix_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TrixParams { period: Some(10) };
        let input = TrixInput::from_slice(&data_small, params);
        let res = trix_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] TRIX should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_trix_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TrixParams { period: Some(18) };
        let input = TrixInput::from_slice(&single_point, params);
        let res = trix_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] TRIX should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_trix_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = TrixParams { period: Some(10) };
        let input = TrixInput::from_candles(&candles, "close", params);
        let first_result = trix_with_kernel(&input, kernel)?;
        let second_input = TrixInput::from_slice(&first_result.values, TrixParams { period: Some(10) });
        let second_result = trix_with_kernel(&second_input, kernel)?;
        assert_eq!(first_result.values.len(), second_result.values.len());
        Ok(())
    }

    macro_rules! generate_all_trix_tests {
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
    generate_all_trix_tests!(
        check_trix_partial_params,
        check_trix_accuracy,
        check_trix_default_candles,
        check_trix_zero_period,
        check_trix_period_exceeds_length,
        check_trix_very_small_dataset,
        check_trix_reinput
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = TrixBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = TrixParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -16.03736447,
            -15.92084231,
            -15.76171478,
            -15.53571033,
            -15.34967155,
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
