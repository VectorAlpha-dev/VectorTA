//! # Relative Strength Index (RSI)
//!
//! A momentum oscillator measuring recent price changes’ speed and magnitude.
//! RSI oscillates between 0 and 100. Typical period: 14 bars.
//!
//! ## Parameters
//! - **period**: Window size (number of bars, default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: rsi: All input values are `NaN`.
//! - **InvalidPeriod**: rsi: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: rsi: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(RsiOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(RsiError)`** otherwise.

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
use paste::paste;

impl<'a> AsRef<[f64]> for RsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RsiData::Slice(slice) => slice,
            RsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RsiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RsiParams {
    pub period: Option<usize>,
}

impl Default for RsiParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct RsiInput<'a> {
    pub data: RsiData<'a>,
    pub params: RsiParams,
}

impl<'a> RsiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: RsiParams) -> Self {
        Self {
            data: RsiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: RsiParams) -> Self {
        Self {
            data: RsiData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", RsiParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RsiBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for RsiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RsiBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<RsiOutput, RsiError> {
        let p = RsiParams {
            period: self.period,
        };
        let i = RsiInput::from_candles(c, "close", p);
        rsi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<RsiOutput, RsiError> {
        let p = RsiParams {
            period: self.period,
        };
        let i = RsiInput::from_slice(d, p);
        rsi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<RsiStream, RsiError> {
        let p = RsiParams {
            period: self.period,
        };
        RsiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum RsiError {
    #[error("rsi: All values are NaN.")]
    AllValuesNaN,
    #[error("rsi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn rsi(input: &RsiInput) -> Result<RsiOutput, RsiError> {
    rsi_with_kernel(input, Kernel::Auto)
}

pub fn rsi_with_kernel(input: &RsiInput, kernel: Kernel) -> Result<RsiOutput, RsiError> {
    let data: &[f64] = match &input.data {
        RsiData::Candles { candles, source } => source_type(candles, source),
        RsiData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RsiError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(RsiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(RsiError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                rsi_scalar(data, period, first, &mut vec![f64::NAN; len])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                rsi_avx2(data, period, first, &mut vec![f64::NAN; len])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                rsi_avx512(data, period, first, &mut vec![f64::NAN; len])
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub unsafe fn rsi_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut Vec<f64>,
) -> Result<RsiOutput, RsiError> {
    let len = data.len();
    let mut rsi_values = vec![f64::NAN; len];

    let inv_period = 1.0 / period as f64;
    let beta = 1.0 - inv_period;

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in (first + 1)..=(first + period) {
        let delta = data[i] - data[i - 1];
        if delta > 0.0 {
            avg_gain += delta;
        
            } else {
            avg_loss += -delta;
        }
    }
    avg_gain *= inv_period;
    avg_loss *= inv_period;

    let initial_rsi = if avg_gain + avg_loss == 0.0 {
        50.0
    
        } else {
        100.0 * avg_gain / (avg_gain + avg_loss)
    };
    rsi_values[first + period] = initial_rsi;

    for i in (first + period + 1)..len {
        let delta = data[i] - data[i - 1];
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };
        avg_gain = inv_period * gain + beta * avg_gain;
        avg_loss = inv_period * loss + beta * avg_loss;
        let rsi = if avg_gain + avg_loss == 0.0 {
            50.0
        
            } else {
            100.0 * avg_gain / (avg_gain + avg_loss)
        };
        rsi_values[i] = rsi;
    }
    Ok(RsiOutput { values: rsi_values })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rsi_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut Vec<f64>,
) -> Result<RsiOutput, RsiError> {
    rsi_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rsi_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut Vec<f64>,
) -> Result<RsiOutput, RsiError> {
    if period <= 32 {
        rsi_avx512_short(data, period, first, out)
    
        } else {
        rsi_avx512_long(data, period, first, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rsi_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut Vec<f64>,
) -> Result<RsiOutput, RsiError> {
    rsi_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rsi_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut Vec<f64>,
) -> Result<RsiOutput, RsiError> {
    rsi_scalar(data, period, first, out)
}

// --- Batch grid/range support ---

#[derive(Clone, Debug)]
pub struct RsiBatchRange {
    pub period: (usize, usize, usize),
}
impl Default for RsiBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct RsiBatchBuilder {
    range: RsiBatchRange,
    kernel: Kernel,
}
impl RsiBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<RsiBatchOutput, RsiError> {
        rsi_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RsiBatchOutput, RsiError> {
        RsiBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RsiBatchOutput, RsiError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<RsiBatchOutput, RsiError> {
        RsiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn rsi_batch_with_kernel(
    data: &[f64],
    sweep: &RsiBatchRange,
    k: Kernel,
) -> Result<RsiBatchOutput, RsiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(RsiError::InvalidPeriod {
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
    rsi_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RsiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<RsiParams>,
    pub rows: usize,
    pub cols: usize,
}
impl RsiBatchOutput {
    pub fn row_for_params(&self, p: &RsiParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &RsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &RsiBatchRange) -> Vec<RsiParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(RsiParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn rsi_batch_slice(
    data: &[f64],
    sweep: &RsiBatchRange,
    kern: Kernel,
) -> Result<RsiBatchOutput, RsiError> {
    rsi_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn rsi_batch_par_slice(
    data: &[f64],
    sweep: &RsiBatchRange,
    kern: Kernel,
) -> Result<RsiBatchOutput, RsiError> {
    rsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rsi_batch_inner(
    data: &[f64],
    sweep: &RsiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RsiBatchOutput, RsiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RsiError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RsiError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(RsiError::NotEnoughValidData {
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
            Kernel::Scalar => rsi_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => rsi_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => rsi_row_avx512(data, first, period, out_row),
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

    Ok(RsiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn rsi_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let inv_period = 1.0 / period as f64;
    let beta = 1.0 - inv_period;
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in (first + 1)..=(first + period) {
        let delta = data[i] - data[i - 1];
        if delta > 0.0 {
            avg_gain += delta;
        
            } else {
            avg_loss += -delta;
        }
    }
    avg_gain *= inv_period;
    avg_loss *= inv_period;

    let initial_rsi = if avg_gain + avg_loss == 0.0 {
        50.0
    
        } else {
        100.0 * avg_gain / (avg_gain + avg_loss)
    };
    out[first + period] = initial_rsi;

    for i in (first + period + 1)..len {
        let delta = data[i] - data[i - 1];
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };
        avg_gain = inv_period * gain + beta * avg_gain;
        avg_loss = inv_period * loss + beta * avg_loss;
        let rsi = if avg_gain + avg_loss == 0.0 {
            50.0
        
            } else {
            100.0 * avg_gain / (avg_gain + avg_loss)
        };
        out[i] = rsi;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    rsi_row_scalar(data, first, period, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        rsi_row_avx512_short(data, first, period, out)
    
        } else {
        rsi_row_avx512_long(data, first, period, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    rsi_row_scalar(data, first, period, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    rsi_row_scalar(data, first, period, out)
}

// --- Streaming RSI (for parity with alma.rs) ---

#[derive(Debug, Clone)]
pub struct RsiStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    avg_gain: f64,
    avg_loss: f64,
    prev: f64,
    first: bool,
}
impl RsiStream {
    pub fn try_new(params: RsiParams) -> Result<Self, RsiError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(RsiError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev: f64::NAN,
            first: true,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.first {
            self.prev = value;
            self.first = false;
            return None;
        }
        let delta = value - self.prev;
        self.prev = value;
        self.buffer[self.head] = delta;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
            let mut gain = 0.0;
            let mut loss = 0.0;
            for &d in &self.buffer[..self.period] {
                if d > 0.0 {
                    gain += d;
                
                    } else {
                    loss += -d;
                }
            }
            self.avg_gain = gain / self.period as f64;
            self.avg_loss = loss / self.period as f64;
            if self.avg_gain + self.avg_loss == 0.0 {
                return Some(50.0);
            
                } else {
                return Some(100.0 * self.avg_gain / (self.avg_gain + self.avg_loss));
            }
        }
        if !self.filled {
            return None;
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };
        let inv_period = 1.0 / self.period as f64;
        let beta = 1.0 - inv_period;
        self.avg_gain = inv_period * gain + beta * self.avg_gain;
        self.avg_loss = inv_period * loss + beta * self.avg_loss;
        if self.avg_gain + self.avg_loss == 0.0 {
            Some(50.0)
        
            } else {
            Some(100.0 * self.avg_gain / (self.avg_gain + self.avg_loss))
        }
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_rsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = RsiParams { period: None };
        let input = RsiInput::from_candles(&candles, "close", partial_params);
        let result = rsi_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }
    fn check_rsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = RsiInput::from_candles(&candles, "close", RsiParams { period: Some(14) });
        let result = rsi_with_kernel(&input, kernel)?;
        let expected_last_five = [43.42, 42.68, 41.62, 42.86, 39.01];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-2,
                "[{}] RSI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_rsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = RsiInput::with_default_candles(&candles);
        match input.data {
            RsiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected RsiData::Candles"),
        }
        let output = rsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_rsi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = RsiParams { period: Some(0) };
        let input = RsiInput::from_slice(&input_data, params);
        let res = rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] RSI should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_rsi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = RsiParams { period: Some(10) };
        let input = RsiInput::from_slice(&data_small, params);
        let res = rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] RSI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_rsi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = RsiParams { period: Some(14) };
        let input = RsiInput::from_slice(&single_point, params);
        let res = rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] RSI should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_rsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = RsiParams { period: Some(14) };
        let first_input = RsiInput::from_candles(&candles, "close", first_params);
        let first_result = rsi_with_kernel(&first_input, kernel)?;
        let second_params = RsiParams { period: Some(5) };
        let second_input = RsiInput::from_slice(&first_result.values, second_params);
        let second_result = rsi_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Found NaN in RSI at {}",
                    i
                );
            }
        }
        Ok(())
    }
    fn check_rsi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = RsiInput::from_candles(
            &candles,
            "close",
            RsiParams { period: Some(14) },
        );
        let res = rsi_with_kernel(&input, kernel)?;
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
    fn check_rsi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let input = RsiInput::from_candles(
            &candles,
            "close",
            RsiParams { period: Some(period) },
        );
        let batch_output = rsi_with_kernel(&input, kernel)?.values;

        let mut stream = RsiStream::try_new(RsiParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(rsi_val) => stream_values.push(rsi_val),
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
                diff < 1e-6,
                "[{}] RSI streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_rsi_tests {
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

    generate_all_rsi_tests!(
        check_rsi_partial_params,
        check_rsi_accuracy,
        check_rsi_default_candles,
        check_rsi_zero_period,
        check_rsi_period_exceeds_length,
        check_rsi_very_small_dataset,
        check_rsi_reinput,
        check_rsi_nan_handling,
        check_rsi_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = RsiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = RsiParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [43.42, 42.68, 41.62, 42.86, 39.01];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
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
