//! # DX (Directional Movement Index)
//!
//! Measures trend strength by comparing smoothed +DI and -DI, using Welles Wilder’s logic.
//! - `period`: window size (typically 14).
//!
//! ## Errors
//! - **EmptyData**: All input slices empty.
//! - **SelectCandleFieldError**: Failed to select field from `Candles`.
//! - **InvalidPeriod**: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data after first valid index.
//! - **AllValuesNaN**: All high, low, and close values are NaN.
//!
//! ## Returns
//! - **Ok(DxOutput)** on success, with output length matching input.
//! - **Err(DxError)** otherwise.

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

#[derive(Debug, Clone)]
pub enum DxData<'a> {
    Candles {
        candles: &'a Candles,
    },
    HlcSlices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct DxOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DxParams {
    pub period: Option<usize>,
}

impl Default for DxParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct DxInput<'a> {
    pub data: DxData<'a>,
    pub params: DxParams,
}

impl<'a> DxInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DxParams) -> Self {
        Self {
            data: DxData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_hlc_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: DxParams,
    ) -> Self {
        Self {
            data: DxData::HlcSlices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DxData::Candles { candles },
            params: DxParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DxBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DxBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DxBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<DxOutput, DxError> {
        let p = DxParams { period: self.period };
        let i = DxInput::from_candles(c, p);
        dx_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_hlc(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<DxOutput, DxError> {
        let p = DxParams { period: self.period };
        let i = DxInput::from_hlc_slices(high, low, close, p);
        dx_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DxStream, DxError> {
        let p = DxParams { period: self.period };
        DxStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DxError {
    #[error("dx: Empty data provided for DX.")]
    EmptyData,
    #[error("dx: Could not select candle field: {0}")]
    SelectCandleFieldError(String),
    #[error("dx: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dx: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dx: All high, low, and close values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn dx(input: &DxInput) -> Result<DxOutput, DxError> {
    dx_with_kernel(input, Kernel::Auto)
}

pub fn dx_with_kernel(input: &DxInput, kernel: Kernel) -> Result<DxOutput, DxError> {
    let (high, low, close) = match &input.data {
        DxData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
            (high, low, close)
        }
        DxData::HlcSlices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(DxError::EmptyData);
    }
    let len = high.len().min(low.len()).min(close.len());
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(DxError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let first_valid_idx =
        (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(DxError::AllValuesNaN),
    };
    if (len - first_valid_idx) < period {
        return Err(DxError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
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
                dx_scalar(high, low, close, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                dx_avx2(high, low, close, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                dx_avx512(high, low, close, period, first_valid_idx, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(DxOutput { values: out })
}

// Scalar implementation (original logic preserved)
#[inline]
pub fn dx_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    let len = high.len().min(low.len()).min(close.len());
    let mut prev_high = high[first_valid_idx];
    let mut prev_low = low[first_valid_idx];
    let mut prev_close = close[first_valid_idx];
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;
    let mut initial_count = 0;
    for i in (first_valid_idx + 1)..len {
        if high[i].is_nan() || low[i].is_nan() || close[i].is_nan() {
            out[i] = if i > 0 { out[i - 1] } else { f64::NAN };
            prev_high = high[i];
            prev_low = low[i];
            prev_close = close[i];
            continue;
        }
        let up_move = high[i] - prev_high;
        let down_move = prev_low - low[i];
        let mut plus_dm = 0.0;
        let mut minus_dm = 0.0;
        if up_move > 0.0 && up_move > down_move {
            plus_dm = up_move;
        } else if down_move > 0.0 && down_move > up_move {
            minus_dm = down_move;
        }
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - prev_close).abs();
        let tr3 = (low[i] - prev_close).abs();
        let tr = tr1.max(tr2).max(tr3);
        if initial_count < (period - 1) {
            plus_dm_sum += plus_dm;
            minus_dm_sum += minus_dm;
            tr_sum += tr;
            initial_count += 1;
            if initial_count == (period - 1) {
                let plus_di = (plus_dm_sum / tr_sum) * 100.0;
                let minus_di = (minus_dm_sum / tr_sum) * 100.0;
                let sum_di = plus_di + minus_di;
                out[i] = if sum_di != 0.0 {
                    100.0 * ((plus_di - minus_di).abs() / sum_di)
                
                    } else {
                    0.0
                };
        } else {
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period as f64) + plus_dm;
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period as f64) + minus_dm;
            tr_sum = tr_sum - (tr_sum / period as f64) + tr;
            let plus_di = if tr_sum != 0.0 {
                (plus_dm_sum / tr_sum) * 100.0
            
                } else {
                0.0
            };
            let minus_di = if tr_sum != 0.0 {
                (minus_dm_sum / tr_sum) * 100.0
            
                } else {
                0.0
            };
            let sum_di = plus_di + minus_di;
            out[i] = if sum_di != 0.0 {
                100.0 * ((plus_di - minus_di).abs() / sum_di)
            
                } else {
                out[i - 1]
            };
        }
        prev_high = high[i];
        prev_low = low[i];
        prev_close = close[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dx_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { dx_avx512_short(high, low, close, period, first_valid, out) }
    } else {
        unsafe { dx_avx512_long(high, low, close, period, first_valid, out) }
    }
}

#[inline]
pub fn dx_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // Stub points to scalar for API parity
    dx_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dx_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    dx_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dx_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    dx_scalar(high, low, close, period, first_valid, out)
}

// Stream implementation (emulates alma.rs streaming)
#[derive(Debug, Clone)]
pub struct DxStream {
    period: usize,
    plus_dm_sum: f64,
    minus_dm_sum: f64,
    tr_sum: f64,
    prev_high: f64,
    prev_low: f64,
    prev_close: f64,
    initial_count: usize,
    filled: bool,
}

impl DxStream {
    pub fn try_new(params: DxParams) -> Result<Self, DxError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(DxError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            plus_dm_sum: 0.0,
            minus_dm_sum: 0.0,
            tr_sum: 0.0,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            prev_close: f64::NAN,
            initial_count: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if self.prev_high.is_nan() || self.prev_low.is_nan() || self.prev_close.is_nan() {
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            return None;
        }
        let up_move = high - self.prev_high;
        let down_move = self.prev_low - low;
        let plus_dm = if up_move > 0.0 && up_move > down_move {
            up_move
        
            } else {
            0.0
        };
        let minus_dm = if down_move > 0.0 && down_move > up_move {
            down_move
        
            } else {
            0.0
        };
        let tr1 = high - low;
        let tr2 = (high - self.prev_close).abs();
        let tr3 = (low - self.prev_close).abs();
        let tr = tr1.max(tr2).max(tr3);

        if self.initial_count < (self.period - 1) {
            self.plus_dm_sum += plus_dm;
            self.minus_dm_sum += minus_dm;
            self.tr_sum += tr;
            self.initial_count += 1;
            if self.initial_count == (self.period - 1) {
                let plus_di = (self.plus_dm_sum / self.tr_sum) * 100.0;
                let minus_di = (self.minus_dm_sum / self.tr_sum) * 100.0;
                let sum_di = plus_di + minus_di;
                self.filled = true;
                self.prev_high = high;
                self.prev_low = low;
                self.prev_close = close;
                return Some(if sum_di != 0.0 {
                    100.0 * ((plus_di - minus_di).abs() / sum_di)
                
                    } else {
                    0.0
                });
            } else {
                self.prev_high = high;
                self.prev_low = low;
                self.prev_close = close;
                return None;
        } else {
            self.plus_dm_sum = self.plus_dm_sum - (self.plus_dm_sum / self.period as f64) + plus_dm;
            self.minus_dm_sum = self.minus_dm_sum - (self.minus_dm_sum / self.period as f64) + minus_dm;
            self.tr_sum = self.tr_sum - (self.tr_sum / self.period as f64) + tr;
            let plus_di = if self.tr_sum != 0.0 {
                (self.plus_dm_sum / self.tr_sum) * 100.0
            
                } else {
                0.0
            };
            let minus_di = if self.tr_sum != 0.0 {
                (self.minus_dm_sum / self.tr_sum) * 100.0
            
                } else {
                0.0
            };
            let sum_di = plus_di + minus_di;
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            return Some(if sum_di != 0.0 {
                100.0 * ((plus_di - minus_di).abs() / sum_di)
            } else {
                0.0
            });
        }
    }
}

// Batch/grid sweep types
#[derive(Clone, Debug)]
pub struct DxBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DxBatchRange {
    fn default() -> Self {
        Self { period: (14, 14, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DxBatchBuilder {
    range: DxBatchRange,
    kernel: Kernel,
}

impl DxBatchBuilder {
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

    pub fn apply_hlc(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<DxBatchOutput, DxError> {
        dx_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<DxBatchOutput, DxError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_hlc(high, low, close)
    }
}

pub struct DxBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DxParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DxBatchOutput {
    pub fn row_for_params(&self, p: &DxParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
        })
    }
    pub fn values_for(&self, p: &DxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DxBatchRange) -> Vec<DxParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![DxParams { period: Some(start) }];
    }
    (start..=end).step_by(step).map(|p| DxParams { period: Some(p) }).collect()
}

pub fn dx_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DxBatchRange,
    k: Kernel,
) -> Result<DxBatchOutput, DxError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(DxError::InvalidPeriod {
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
    dx_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn dx_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DxBatchRange,
    kern: Kernel,
) -> Result<DxBatchOutput, DxError> {
    dx_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn dx_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DxBatchRange,
    kern: Kernel,
) -> Result<DxBatchOutput, DxError> {
    dx_batch_inner(high, low, close, sweep, kern, true)
}

fn dx_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DxBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DxBatchOutput, DxError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DxError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let len = high.len().min(low.len()).min(close.len());
    let first = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(DxError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(DxError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => dx_row_scalar(high, low, close, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dx_row_avx2(high, low, close, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dx_row_avx512(high, low, close, first, period, out_row),
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
    Ok(DxBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn dx_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    dx_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dx_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    dx_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dx_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        dx_row_avx512_short(high, low, close, first, period, out);
    
        } else {
        dx_row_avx512_long(high, low, close, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dx_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    dx_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dx_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    dx_scalar(high, low, close, period, first, out)
}

#[inline(always)]
pub fn expand_grid_dx(r: &DxBatchRange) -> Vec<DxParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_dx_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = DxParams { period: None };
        let input = DxInput::from_candles(&candles, default_params);
        let output = dx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dx_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = DxInput::from_candles(&candles, DxParams::default());
        let result = dx_with_kernel(&input, kernel)?;
        let expected_last_five = [
            43.72121533411883,
            41.47251493226443,
            43.43041386436222,
            43.22673458811955,
            51.65514026197179,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-4,
                "[{}] DX {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_dx_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = DxInput::with_default_candles(&candles);
        match input.data {
            DxData::Candles { .. } => {}
            _ => panic!("Expected DxData::Candles"),
        }
        let output = dx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dx_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [2.0, 2.5, 3.0];
        let low = [1.0, 1.2, 2.1];
        let close = [1.5, 2.3, 2.2];
        let params = DxParams { period: Some(0) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let res = dx_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] DX should fail with zero period", test_name);
        Ok(())
    }

    fn check_dx_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [3.0, 4.0];
        let low = [2.0, 3.0];
        let close = [2.5, 3.5];
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let res = dx_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DX should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_dx_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [3.0];
        let low = [2.0];
        let close = [2.5];
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let res = dx_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DX should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_dx_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = DxParams { period: Some(14) };
        let first_input = DxInput::from_candles(&candles, first_params);
        let first_result = dx_with_kernel(&first_input, kernel)?;

        let second_params = DxParams { period: Some(14) };
        let second_input = DxInput::from_hlc_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = dx_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index 28, found NaN at idx {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_dx_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DxInput::from_candles(
            &candles,
            DxParams { period: Some(14) },
        );
        let res = dx_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 50 {
            for (i, &val) in res.values[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_dx_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let close = source_type(&candles, "close");
        let period = 14;

        let input = DxInput::from_candles(&candles, DxParams { period: Some(period) });
        let batch_output = dx_with_kernel(&input, kernel)?.values;

        let mut stream = DxStream::try_new(DxParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), &c) in high.iter().zip(low).zip(close) {
            match stream.update(h, l, c) {
                Some(dx_val) => stream_values.push(dx_val),
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
                "[{}] DX streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_dx_tests {
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
    generate_all_dx_tests!(
        check_dx_partial_params,
        check_dx_accuracy,
        check_dx_default_candles,
        check_dx_zero_period,
        check_dx_period_exceeds_length,
        check_dx_very_small_dataset,
        check_dx_reinput,
        check_dx_nan_handling,
        check_dx_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = DxBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = DxParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            43.72121533411883,
            41.47251493226443,
            43.43041386436222,
            43.22673458811955,
            51.65514026197179,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
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
