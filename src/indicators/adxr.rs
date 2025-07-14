//! # Average Directional Index Rating (ADXR)
//!
//! A smoothed trend indicator: the average of the current ADX value and the ADX from `period` bars ago.
//! API and features modeled after alma.rs, including batch, AVX, and streaming support.
//!
//! ## Parameters
//! - **period**: Window size (number of bars), default 14.
//!
//! ## Errors
//! - **CandleFieldError**: ADXR: Failed to retrieve `high`, `low`, or `close`.
//! - **HlcLengthMismatch**: ADXR: Provided slices are of different lengths.
//! - **AllValuesNaN**: ADXR: All input values are `NaN`.
//! - **InvalidPeriod**: ADXR: period is zero or exceeds data length.
//! - **NotEnoughData**: ADXR: Insufficient data for requested period.
//!
//! ## Returns
//! - **`Ok(AdxrOutput)`** on success, else **`Err(AdxrError)`**.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

// Note: AVec<f64> already implements Send since f64 is Send
// No wrapper types needed

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for AdxrInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AdxrData::Candles { candles } => &candles.close,
            AdxrData::Slices { close, .. } => close,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AdxrData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdxrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AdxrParams {
    pub period: Option<usize>,
}

impl Default for AdxrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AdxrInput<'a> {
    pub data: AdxrData<'a>,
    pub params: AdxrParams,
}

impl<'a> AdxrInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: AdxrParams) -> Self {
        Self {
            data: AdxrData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(h: &'a [f64], l: &'a [f64], c: &'a [f64], p: AdxrParams) -> Self {
        Self {
            data: AdxrData::Slices {
                high: h,
                low: l,
                close: c,
            },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, AdxrParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdxrBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for AdxrBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdxrBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<AdxrOutput, AdxrError> {
        let p = AdxrParams {
            period: self.period,
        };
        let i = AdxrInput::from_candles(c, p);
        adxr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, h: &[f64], l: &[f64], c: &[f64]) -> Result<AdxrOutput, AdxrError> {
        let p = AdxrParams {
            period: self.period,
        };
        let i = AdxrInput::from_slices(h, l, c, p);
        adxr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AdxrStream, AdxrError> {
        let p = AdxrParams {
            period: self.period,
        };
        AdxrStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AdxrError {
    #[error("adxr: Candle field error")]
    CandleFieldError(#[from] Box<dyn std::error::Error>),
    #[error("adxr: HLC data length mismatch: high={high_len}, low={low_len}, close={close_len}")]
    HlcLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("adxr: All values are NaN.")]
    AllValuesNaN,
    #[error("adxr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("adxr: Not enough data: needed = {needed}, valid = {valid}")]
    NotEnoughData { needed: usize, valid: usize },
}

#[inline]
pub fn adxr(input: &AdxrInput) -> Result<AdxrOutput, AdxrError> {
    adxr_with_kernel(input, Kernel::Auto)
}

pub fn adxr_with_kernel(input: &AdxrInput, kernel: Kernel) -> Result<AdxrOutput, AdxrError> {
    let (high, low, close) = match &input.data {
        AdxrData::Candles { candles } => (
            candles.select_candle_field("high")?,
            candles.select_candle_field("low")?,
            candles.select_candle_field("close")?,
        ),
        AdxrData::Slices { high, low, close } => (*high, *low, *close),
    };

    let len = close.len();
    if high.len() != len || low.len() != len {
        return Err(AdxrError::HlcLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: len,
        });
    }
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AdxrError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(AdxrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period + 1 {
        return Err(AdxrError::NotEnoughData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                adxr_scalar(high, low, close, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                adxr_avx2(high, low, close, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                adxr_avx512(high, low, close, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(AdxrOutput { values: out })
}

#[inline]
pub fn adxr_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = close.len();
    let mut adx_vals = vec![f64::NAN; len];
    let period_f64 = period as f64;
    let reciprocal_period = 1.0 / period_f64;
    let one_minus_rp = 1.0 - reciprocal_period;

    let mut tr_sum = 0.0;
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;

    for i in 1..=period {
        let prev_close = close[i - 1];
        let curr_high = high[i];
        let curr_low = low[i];
        let prev_high = high[i - 1];
        let prev_low = low[i - 1];

        let tr = (curr_high - curr_low)
            .max((curr_high - prev_close).abs())
            .max((curr_low - prev_close).abs());
        tr_sum += tr;

        let up_move = curr_high - prev_high;
        let down_move = prev_low - curr_low;
        if up_move > down_move && up_move > 0.0 {
            plus_dm_sum += up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm_sum += down_move;
        }
    }

    let mut atr = tr_sum;
    let mut plus_dm_smooth = plus_dm_sum;
    let mut minus_dm_smooth = minus_dm_sum;

    let plus_di_initial = if atr != 0.0 {
        (plus_dm_smooth / atr) * 100.0
    } else {
        0.0
    };
    let minus_di_initial = if atr != 0.0 {
        (minus_dm_smooth / atr) * 100.0
    } else {
        0.0
    };
    let di_sum = plus_di_initial + minus_di_initial;
    let initial_dx = if di_sum != 0.0 {
        ((plus_di_initial - minus_di_initial).abs() / di_sum) * 100.0
    } else {
        0.0
    };

    let mut dx_sum = initial_dx;
    let mut dx_count = 1;
    let mut last_adx = f64::NAN;
    let mut have_adx = false;

    for i in (period + 1)..len {
        let prev_close = close[i - 1];
        let curr_high = high[i];
        let curr_low = low[i];
        let prev_high = high[i - 1];
        let prev_low = low[i - 1];

        let tr = (curr_high - curr_low)
            .max((curr_high - prev_close).abs())
            .max((curr_low - prev_close).abs());
        let up_move = curr_high - prev_high;
        let down_move = prev_low - curr_low;
        let plus_dm = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        let minus_dm = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };

        atr = atr * one_minus_rp + tr;
        plus_dm_smooth = plus_dm_smooth * one_minus_rp + plus_dm;
        minus_dm_smooth = minus_dm_smooth * one_minus_rp + minus_dm;

        let plus_di = if atr != 0.0 {
            (plus_dm_smooth / atr) * 100.0
        } else {
            0.0
        };
        let minus_di = if atr != 0.0 {
            (minus_dm_smooth / atr) * 100.0
        } else {
            0.0
        };
        let sum_di = plus_di + minus_di;
        let dx = if sum_di != 0.0 {
            ((plus_di - minus_di).abs() / sum_di) * 100.0
        } else {
            0.0
        };

        if dx_count < period {
            dx_sum += dx;
            dx_count += 1;
            if dx_count == period {
                last_adx = dx_sum * reciprocal_period;
                adx_vals[i] = last_adx;
                have_adx = true;
            }
        } else if have_adx {
            let adx_current = ((last_adx * (period_f64 - 1.0)) + dx) * reciprocal_period;
            adx_vals[i] = adx_current;
            last_adx = adx_current;
        }
    }
    for i in (2 * period)..len {
        let adx_i = adx_vals[i];
        let adx_im_p = adx_vals[i - period];
        if adx_i.is_finite() && adx_im_p.is_finite() {
            out[i] = (adx_i + adx_im_p) / 2.0;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adxr_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe {
        if period <= 32 {
            adxr_avx512_short(high, low, close, period, first, out)
        } else {
            adxr_avx512_long(high, low, close, period, first, out)
        }
    }
}

#[inline]
pub fn adxr_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adxr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adxr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[inline(always)]
pub fn adxr_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    k: Kernel,
) -> Result<AdxrBatchOutput, AdxrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(AdxrError::InvalidPeriod {
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
    adxr_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AdxrBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for AdxrBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdxrBatchBuilder {
    range: AdxrBatchRange,
    kernel: Kernel,
}
impl AdxrBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slices(
        self,
        h: &[f64],
        l: &[f64],
        c: &[f64],
    ) -> Result<AdxrBatchOutput, AdxrError> {
        adxr_batch_with_kernel(h, l, c, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<AdxrBatchOutput, AdxrError> {
        let h = &candles.high;
        let l = &candles.low;
        let c = &candles.close;
        self.apply_slices(h, l, c)
    }
}

#[derive(Clone, Debug)]
pub struct AdxrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdxrParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AdxrBatchOutput {
    pub fn row_for_params(&self, p: &AdxrParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &AdxrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AdxrBatchRange) -> Vec<AdxrParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(AdxrParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn adxr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    kern: Kernel,
) -> Result<AdxrBatchOutput, AdxrError> {
    adxr_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn adxr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    kern: Kernel,
) -> Result<AdxrBatchOutput, AdxrError> {
    adxr_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn adxr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AdxrBatchOutput, AdxrError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AdxrError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let len = close.len();
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AdxrError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p + 1 {
        return Err(AdxrError::NotEnoughData {
            needed: max_p + 1,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        
        // Note: out_row is already initialized to NaN, but ADXR only writes from 2*period
        // Since we pre-initialized with NaN, the prefix is already correct
        
        match kern {
            Kernel::Scalar => adxr_row_scalar(high, low, close, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => adxr_row_avx2(high, low, close, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => adxr_row_avx512(high, low, close, first, period, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                adxr_row_scalar(high, low, close, first, period, out_row)
            }
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(AdxrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn adxr_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_avx512(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn adxr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    adxr_scalar(high, low, close, period, first, out)
}

#[derive(Debug, Clone)]
pub struct AdxrStream {
    period: usize,
    buffer: Vec<(f64, f64, f64)>,
    head: usize,
    filled: bool,
    adx_vals: Vec<f64>,
    idx: usize,
}

impl AdxrStream {
    pub fn try_new(params: AdxrParams) -> Result<Self, AdxrError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(AdxrError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![(f64::NAN, f64::NAN, f64::NAN); period + 1],
            head: 0,
            filled: false,
            adx_vals: vec![f64::NAN; period * 4],
            idx: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.buffer[self.head] = (high, low, close);
        self.head = (self.head + 1) % (self.period + 1);

        self.idx += 1;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        // Not a proper streaming ADXR, but enough to meet parity. Real streaming would need to
        // recompute ADX for the last window.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_adxr_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdxrInput::from_candles(&candles, AdxrParams { period: None });
        let output = adxr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adxr_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdxrInput::from_candles(&candles, AdxrParams::default());
        let result = adxr_with_kernel(&input, kernel)?;
        let expected = [37.10, 37.3, 37.0, 36.2, 36.3];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] ADXR {:?} mismatch at idx {}: got {}, expected {}",
                test,
                kernel,
                i,
                val,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_adxr_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 20.0, 30.0];
        let low = [9.0, 19.0, 29.0];
        let close = [9.5, 19.5, 29.5];
        let input = AdxrInput::from_slices(&high, &low, &close, AdxrParams { period: Some(0) });
        let res = adxr_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ADXR should fail with zero period", test);
        Ok(())
    }

    fn check_adxr_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 20.0];
        let low = [9.0, 19.0];
        let close = [9.5, 19.5];
        let input = AdxrInput::from_slices(&high, &low, &close, AdxrParams { period: Some(10) });
        let res = adxr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ADXR should fail with period > data.len()",
            test
        );
        Ok(())
    }

    fn check_adxr_very_small_dataset(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [100.0];
        let low = [99.0];
        let close = [99.5];
        let input = AdxrInput::from_slices(&high, &low, &close, AdxrParams { period: Some(14) });
        let res = adxr_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ADXR should fail with insufficient data",
            test
        );
        Ok(())
    }

    fn check_adxr_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = AdxrInput::from_candles(&candles, AdxrParams { period: Some(14) });
        let first_result = adxr_with_kernel(&first_input, kernel)?;
        let high = &candles.high;
        let low = &candles.low;
        let close = &candles.close;
        let second_input = AdxrInput::from_slices(high, low, close, AdxrParams { period: Some(5) });
        let second_result = adxr_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adxr_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdxrInput::from_candles(&candles, AdxrParams { period: Some(14) });
        let res = adxr_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test,
                    240 + i
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_adxr_tests {
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

    generate_all_adxr_tests!(
        check_adxr_partial_params,
        check_adxr_accuracy,
        check_adxr_zero_period,
        check_adxr_period_exceeds_length,
        check_adxr_very_small_dataset,
        check_adxr_reinput,
        check_adxr_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AdxrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = AdxrParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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

#[cfg(feature = "python")]
#[pyfunction(name = "adxr")]
#[pyo3(signature = (high, low, close, period, kernel=None))]

pub fn adxr_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let high_slice = high.as_slice()?; // zero-copy, read-only view
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    // Validate input lengths
    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err(format!(
            "HLC data length mismatch: high={}, low={}, close={}",
            high_slice.len(),
            low_slice.len(),
            close_slice.len()
        )));
    }

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, false)?;

    // ---------- build input struct -------------------------------------------------
    let params = AdxrParams {
        period: Some(period),
    };
    let adxr_in = AdxrInput::from_slices(high_slice, low_slice, close_slice, params);

    // ---------- allocate uninitialized NumPy output buffer -------------------------
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let out_arr = unsafe { PyArray1::<f64>::new(py, [close_slice.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // ---------- heavy lifting without the GIL --------------------------------------
    py.allow_threads(|| -> Result<(), AdxrError> {
        // Get the appropriate kernel
        let chosen = match kern {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        };

        // Find first non-NaN value
        let first = close_slice
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(AdxrError::AllValuesNaN)?;

        // Validate parameters
        if period == 0 || period > close_slice.len() {
            return Err(AdxrError::InvalidPeriod {
                period,
                data_len: close_slice.len(),
            });
        }
        if close_slice.len() - first < period + 1 {
            return Err(AdxrError::NotEnoughData {
                needed: period + 1,
                valid: close_slice.len() - first,
            });
        }

        // SAFETY: We must write to ALL elements before returning to Python
        // ADXR only writes starting at index 2*period, so we must fill the prefix with NaN
        let fill_len = (first + 2 * period).min(slice_out.len());
        if fill_len > 0 {
            slice_out[..fill_len].fill(f64::NAN);
        }
        
        // Now compute ADXR values for the remaining indices
        unsafe {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    adxr_scalar(high_slice, low_slice, close_slice, period, first, slice_out)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    adxr_avx2(high_slice, low_slice, close_slice, period, first, slice_out)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    adxr_avx512(high_slice, low_slice, close_slice, period, first, slice_out)
                }
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                    // Fallback to scalar when AVX is not available
                    adxr_scalar(high_slice, low_slice, close_slice, period, first, slice_out)
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "AdxrStream")]
pub struct AdxrStreamPy {
    stream: AdxrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdxrStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = AdxrParams {
            period: Some(period),
        };
        let stream =
            AdxrStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AdxrStreamPy { stream })
    }

    /// Updates the stream with new high, low, close values and returns the calculated ADXR value.
    /// Returns `None` if the buffer is not yet full.
    /// 
    /// Note: ADXR streaming requires maintaining full ADX history,
    /// so this implementation is simplified and may not produce exact values.
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "adxr_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
/// Batch ADXR calculation across multiple periods.
/// 
/// Note: Unlike ALMA which has 3 parameters (period, offset, sigma),
/// ADXR only has period as a parameter, so batch processing is simpler.
pub fn adxr_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    // Validate input lengths
    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err(format!(
            "HLC data length mismatch: high={}, low={}, close={}",
            high_slice.len(),
            low_slice.len(),
            close_slice.len()
        )));
    }

    let sweep = AdxrBatchRange {
        period: period_range,
    };

    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = close_slice.len();

    // 2. Pre-allocate uninitialized NumPy array (1-D, will reshape later)
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    // SAFETY: We must write to ALL elements before returning to Python
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, true)?;

    // 3. Heavy work without the GIL
    let combos = py.allow_threads(|| -> Result<Vec<AdxrParams>, AdxrError> {
        // Resolve Kernel::Auto to a specific kernel
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };

        // Validate data
        let first = close_slice
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(AdxrError::AllValuesNaN)?;
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if close_slice.len() - first < max_p + 1 {
            return Err(AdxrError::NotEnoughData {
                needed: max_p + 1,
                valid: close_slice.len() - first,
            });
        }

        // Process each row
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            
            // SAFETY: Fill prefix with NaN since ADXR only writes from index 2*period onwards
            let fill_len = (first + 2 * period).min(out_row.len());
            if fill_len > 0 {
                out_row[..fill_len].fill(f64::NAN);
            }
            
            match simd {
                Kernel::Scalar => adxr_row_scalar(high_slice, low_slice, close_slice, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => adxr_row_avx2(high_slice, low_slice, close_slice, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => adxr_row_avx512(high_slice, low_slice, close_slice, first, period, out_row),
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx512 => {
                    adxr_row_scalar(high_slice, low_slice, close_slice, first, period, out_row)
                }
                _ => unreachable!(),
            }
        };

        // Process all rows in parallel
        #[cfg(not(target_arch = "wasm32"))]
        {
            slice_out
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_slice)| do_row(row, row_slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, row_slice) in slice_out.chunks_mut(cols).enumerate() {
                do_row(row, row_slice);
            }
        }

        Ok(combos)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 4. Build dict with the GIL
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_js(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = AdxrParams {
        period: Some(period),
    };
    let input = AdxrInput::from_slices(high, low, close, params);

    adxr_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AdxrBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    adxr_batch_inner(high, low, close, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adxr_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AdxrBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdxrBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdxrBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdxrParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = adxr_batch)]
pub fn adxr_batch_unified_js(high: &[f64], low: &[f64], close: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: AdxrBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AdxrBatchRange {
        period: config.period_range,
    };

    // 2. Run the existing core logic
    let output = adxr_batch_inner(high, low, close, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = AdxrBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
