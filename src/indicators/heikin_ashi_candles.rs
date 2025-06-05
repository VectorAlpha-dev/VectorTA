//! # Heikin Ashi Candles
//!
//! Heikin Ashi Candles reduce noise from standard candlestick charts by applying
//! an averaging formula to both the current candle and the previous Heikin Ashi
//! candle. This helps in identifying and visualizing trends more clearly.
//!
//! ## Parameters
//! - *(None)*: No user-configurable parameters. The transformation applies to
//!   the full open, high, low, and close data sets.
//!
//! ## Errors
//! - **EmptyData**: heikin_ashi_candles: Input data slice(s) are empty.
//! - **AllValuesNaN**: heikin_ashi_candles: All input data values are `NaN`.
//! - **NotEnoughValidData**: heikin_ashi_candles: Fewer than 2 valid (non-`NaN`) data points remain
//!   after the first valid index.
//!
//! ## Returns
//! - **`Ok(HeikinAshiOutput)`** on success, containing `Vec<f64>` for open,
//!   high, low, and close matching the input length, with leading `NaN`s
//!   until the first valid index.
//! - **`Err(HeikinAshiError)`** otherwise.

use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum HeikinAshiData<'a> {
    Candles { candles: &'a Candles },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct HeikinAshiOutput {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HeikinAshiParams;

impl Default for HeikinAshiParams {
    fn default() -> Self { Self }
}

#[derive(Debug, Clone)]
pub struct HeikinAshiInput<'a> {
    pub data: HeikinAshiData<'a>,
    pub params: HeikinAshiParams,
}

impl<'a> HeikinAshiInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: HeikinAshiData::Candles { candles },
            params: HeikinAshiParams::default(),
        }
    }
    #[inline]
    pub fn from_slices(open: &'a [f64], high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: HeikinAshiData::Slices { open, high, low, close },
            params: HeikinAshiParams::default(),
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HeikinAshiBuilder {
    kernel: Kernel,
}

impl Default for HeikinAshiBuilder {
    fn default() -> Self {
        Self { kernel: Kernel::Auto }
    }
}

impl HeikinAshiBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply_candles(self, candles: &Candles) -> Result<HeikinAshiOutput, HeikinAshiError> {
        let input = HeikinAshiInput::from_candles(candles);
        heikin_ashi_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Result<HeikinAshiOutput, HeikinAshiError> {
        let input = HeikinAshiInput::from_slices(open, high, low, close);
        heikin_ashi_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<HeikinAshiStream, HeikinAshiError> {
        HeikinAshiStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum HeikinAshiError {
    #[error("heikin_ashi_candles: Empty data provided.")]
    EmptyData,
    #[error("heikin_ashi_candles: All values are NaN.")]
    AllValuesNaN,
    #[error("heikin_ashi_candles: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn heikin_ashi_candles(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_with_kernel(input, Kernel::Auto)
}

pub fn heikin_ashi_with_kernel(input: &HeikinAshiInput, kernel: Kernel) -> Result<HeikinAshiOutput, HeikinAshiError> {
    let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        HeikinAshiData::Candles { candles } => (
            source_type(candles, "open"),
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        HeikinAshiData::Slices { open, high, low, close } => (*open, *high, *low, *close),
    };
    let len = open.len();
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(HeikinAshiError::EmptyData);
    }
    if len != high.len() || len != low.len() || len != close.len() {
        return Err(HeikinAshiError::EmptyData);
    }
    let first_valid_idx = (0..len).find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(HeikinAshiError::AllValuesNaN)?;
    if (len - first_valid_idx) < 2 {
        return Err(HeikinAshiError::NotEnoughValidData { needed: 2, valid: len - first_valid_idx });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                heikin_ashi_scalar(open, high, low, close, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                heikin_ashi_avx2(open, high, low, close, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                heikin_ashi_avx512(open, high, low, close, first_valid_idx)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn heikin_ashi_scalar(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize) -> Result<HeikinAshiOutput, HeikinAshiError> {
    let len = open.len();
    let mut ha_open = vec![f64::NAN; len];
    let mut ha_high = vec![f64::NAN; len];
    let mut ha_low = vec![f64::NAN; len];
    let mut ha_close = vec![f64::NAN; len];

    ha_open[first] = open[first];
    ha_close[first] = (open[first] + high[first] + low[first] + close[first]) / 4.0;
    ha_high[first] = high[first].max(ha_open[first]).max(ha_close[first]);
    ha_low[first] = low[first].min(ha_open[first]).min(ha_close[first]);
    for i in (first + 1)..len {
        let prev = i - 1;
        ha_open[i] = (open[prev] + close[prev]) / 2.0;
        ha_close[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;
        ha_high[i] = high[i].max(ha_open[i]).max(ha_close[i]);
        ha_low[i] = low[i].min(ha_open[i]).min(ha_close[i]);
    }
    Ok(HeikinAshiOutput {
        open: ha_open,
        high: ha_high,
        low: ha_low,
        close: ha_close,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn heikin_ashi_avx2(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_scalar(open, high, low, close, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn heikin_ashi_avx512(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_scalar(open, high, low, close, first)
}

#[inline]
pub fn heikin_ashi_indicator(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_candles(input)
}

#[inline]
pub fn heikin_ashi_indicator_with_kernel(input: &HeikinAshiInput, kernel: Kernel) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_with_kernel(input, kernel)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn heikin_ashi_indicator_avx512(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_indicator_with_kernel(input, Kernel::Avx512)
}

#[inline]
pub fn heikin_ashi_indicator_scalar(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_indicator_with_kernel(input, Kernel::Scalar)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn heikin_ashi_indicator_avx2(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_indicator_with_kernel(input, Kernel::Avx2)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn heikin_ashi_indicator_avx512_short(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_indicator_avx512(input)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn heikin_ashi_indicator_avx512_long(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    heikin_ashi_indicator_avx512(input)
}

#[derive(Debug, Clone)]
pub struct HeikinAshiStream {
    prev_open: Option<f64>,
    prev_close: Option<f64>,
    started: bool,
}

impl HeikinAshiStream {
    pub fn try_new() -> Result<Self, HeikinAshiError> {
        Ok(Self { prev_open: None, prev_close: None, started: false })
    }
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        if open.is_nan() || high.is_nan() || low.is_nan() || close.is_nan() { return None; }
        let ha_close = (open + high + low + close) / 4.0;
        let ha_open = if !self.started {
            self.started = true;
            open
        } else {
            (self.prev_open.unwrap() + self.prev_close.unwrap()) / 2.0
        };
        let ha_high = high.max(ha_open).max(ha_close);
        let ha_low = low.min(ha_open).min(ha_close);
        self.prev_open = Some(ha_open);
        self.prev_close = Some(ha_close);
        Some((ha_open, ha_high, ha_low, ha_close))
    }
}

#[derive(Clone, Debug)]
pub struct HeikinAshiBatchRange;

impl Default for HeikinAshiBatchRange {
    fn default() -> Self { Self }
}

#[derive(Clone, Debug, Default)]
pub struct HeikinAshiBatchBuilder {
    kernel: Kernel,
}

impl HeikinAshiBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn apply_slices(self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Result<HeikinAshiBatchOutput, HeikinAshiError> {
        heikin_ashi_batch_with_kernel(open, high, low, close, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<HeikinAshiBatchOutput, HeikinAshiError> {
        let open = source_type(c, "open");
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(open, high, low, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HeikinAshiBatchOutput, HeikinAshiError> {
        HeikinAshiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct HeikinAshiBatchOutput {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

pub fn heikin_ashi_batch_with_kernel(open: &[f64], high: &[f64], low: &[f64], close: &[f64], k: Kernel) -> Result<HeikinAshiBatchOutput, HeikinAshiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(HeikinAshiError::EmptyData),
    };
    heikin_ashi_batch_par_slice(open, high, low, close, kernel)
}

#[inline(always)]
pub fn heikin_ashi_batch_slice(open: &[f64], high: &[f64], low: &[f64], close: &[f64], kern: Kernel) -> Result<HeikinAshiBatchOutput, HeikinAshiError> {
    heikin_ashi_batch_inner(open, high, low, close, kern, false)
}

#[inline(always)]
pub fn heikin_ashi_batch_par_slice(open: &[f64], high: &[f64], low: &[f64], close: &[f64], kern: Kernel) -> Result<HeikinAshiBatchOutput, HeikinAshiError> {
    heikin_ashi_batch_inner(open, high, low, close, kern, true)
}

#[inline(always)]
fn heikin_ashi_batch_inner(open: &[f64], high: &[f64], low: &[f64], close: &[f64], kern: Kernel, _parallel: bool) -> Result<HeikinAshiBatchOutput, HeikinAshiError> {
    // No param sweep for HA, but batch API exists for parity
    let len = open.len();
    if len == 0 || high.len() != len || low.len() != len || close.len() != len {
        return Err(HeikinAshiError::EmptyData);
    }
    let first = (0..len).find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(HeikinAshiError::AllValuesNaN)?;
    if (len - first) < 2 {
        return Err(HeikinAshiError::NotEnoughValidData { needed: 2, valid: len - first });
    }
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                let o = heikin_ashi_scalar(open, high, low, close, first)?;
                Ok(HeikinAshiBatchOutput {
                    open: o.open,
                    high: o.high,
                    low: o.low,
                    close: o.close,
                    rows: 1,
                    cols: len,
                })
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                let o = heikin_ashi_avx2(open, high, low, close, first)?;
                Ok(HeikinAshiBatchOutput {
                    open: o.open,
                    high: o.high,
                    low: o.low,
                    close: o.close,
                    rows: 1,
                    cols: len,
                })
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                let o = heikin_ashi_avx512(open, high, low, close, first)?;
                Ok(HeikinAshiBatchOutput {
                    open: o.open,
                    high: o.high,
                    low: o.low,
                    close: o.close,
                    rows: 1,
                    cols: len,
                })
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
fn expand_grid(_: &HeikinAshiBatchRange) -> Vec<HeikinAshiParams> {
    vec![HeikinAshiParams]
}

#[inline]
pub unsafe fn heikin_ashi_row_scalar(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize, out_open: &mut [f64], out_high: &mut [f64], out_low: &mut [f64], out_close: &mut [f64]) {
    let len = open.len();
    out_open[first] = open[first];
    out_close[first] = (open[first] + high[first] + low[first] + close[first]) / 4.0;
    out_high[first] = high[first].max(out_open[first]).max(out_close[first]);
    out_low[first] = low[first].min(out_open[first]).min(out_close[first]);
    for i in (first + 1)..len {
        let prev = i - 1;
        out_open[i] = (open[prev] + close[prev]) / 2.0;
        out_close[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;
        out_high[i] = high[i].max(out_open[i]).max(out_close[i]);
        out_low[i] = low[i].min(out_open[i]).min(out_close[i]);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn heikin_ashi_row_avx2(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize, out_open: &mut [f64], out_high: &mut [f64], out_low: &mut [f64], out_close: &mut [f64]) {
    heikin_ashi_row_scalar(open, high, low, close, first, out_open, out_high, out_low, out_close)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn heikin_ashi_row_avx512(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize, out_open: &mut [f64], out_high: &mut [f64], out_low: &mut [f64], out_close: &mut [f64]) {
    heikin_ashi_row_scalar(open, high, low, close, first, out_open, out_high, out_low, out_close)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn heikin_ashi_row_avx512_short(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize, out_open: &mut [f64], out_high: &mut [f64], out_low: &mut [f64], out_close: &mut [f64]) {
    heikin_ashi_row_avx512(open, high, low, close, first, out_open, out_high, out_low, out_close)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn heikin_ashi_row_avx512_long(open: &[f64], high: &[f64], low: &[f64], close: &[f64], first: usize, out_open: &mut [f64], out_high: &mut [f64], out_low: &mut [f64], out_close: &mut [f64]) {
    heikin_ashi_row_avx512(open, high, low, close, first, out_open, out_high, out_low, out_close)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_ha_empty(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input = HeikinAshiInput::from_slices(&[], &[], &[], &[]);
        let result = heikin_ashi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_ha_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let open = [f64::NAN, f64::NAN];
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let input = HeikinAshiInput::from_slices(&open, &high, &low, &close);
        let result = heikin_ashi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_ha_not_enough(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let open = [60000.0];
        let high = [60100.0];
        let low = [59900.0];
        let close = [60050.0];
        let input = HeikinAshiInput::from_slices(&open, &high, &low, &close);
        let result = heikin_ashi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_ha_from_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HeikinAshiInput::from_candles(&candles);
        let output = heikin_ashi_with_kernel(&input, kernel)?;
        assert_eq!(output.open.len(), candles.close.len());
        assert_eq!(output.high.len(), candles.close.len());
        assert_eq!(output.low.len(), candles.close.len());
        assert_eq!(output.close.len(), candles.close.len());
        Ok(())
    }

    fn check_ha_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
    skip_if_unsupported!(kernel, test_name);

    // 1) Load candles from CSV
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;
    let input = HeikinAshiInput::from_candles(&candles);

    // 2) Run Heikin‐Ashi
    let output = heikin_ashi_with_kernel(&input, kernel)?;
    let len = output.open.len();
    assert_eq!(len, candles.close.len(), "Length mismatch between HA output and CSV");

    // 3) Grab the last five indices
    let start = len.saturating_sub(5);

    // 4) “Ground truth” arrays (replace these with your actual expected values):
    let expected_last_five_open: [f64; 5] = [
        59348.5,
        59277.5,
        59233.0,
        59110.5,
        59097.0,
    ];
    let expected_last_five_high: [f64; 5] = [
        59348.5,
        59405.0,
        59304.0,
        59310.0,
        59236.0,
    ];
    let expected_last_five_low: [f64; 5] = [
        59001.0,
        59084.0,
        58932.0,
        58983.0,
        58299.0,
    ];
    let expected_last_five_close: [f64; 5] = [
        59221.75,
        59238.75,
        59114.25,
        59121.75,
        58836.25,
    ];

    // 5) Compare each of the last five data‐points, with a tight tolerance
    for i in 0..5 {
        let idx = start + i;
        let got_open = output.open[idx];
        let got_high = output.high[idx];
        let got_low = output.low[idx];
        let got_close = output.close[idx];

        let diff_open = (got_open - expected_last_five_open[i]).abs();
        let diff_high = (got_high - expected_last_five_high[i]).abs();
        let diff_low = (got_low - expected_last_five_low[i]).abs();
        let diff_close = (got_close - expected_last_five_close[i]).abs();

        assert!(
            diff_open < 1e-8,
            "[{}] HA {:?} open mismatch at idx {}: got {}, expected {}",
            test_name,
            kernel,
            idx,
            got_open,
            expected_last_five_open[i]
        );
        assert!(
            diff_high < 1e-8,
            "[{}] HA {:?} high mismatch at idx {}: got {}, expected {}",
            test_name,
            kernel,
            idx,
            got_high,
            expected_last_five_high[i]
        );
        assert!(
            diff_low < 1e-8,
            "[{}] HA {:?} low mismatch at idx {}: got {}, expected {}",
            test_name,
            kernel,
            idx,
            got_low,
            expected_last_five_low[i]
        );
        assert!(
            diff_close < 1e-8,
            "[{}] HA {:?} close mismatch at idx {}: got {}, expected {}",
            test_name,
            kernel,
            idx,
            got_close,
            expected_last_five_close[i]
        );
    }

        Ok(())
    }

    macro_rules! generate_all_ha_tests {
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
    generate_all_ha_tests!(
        check_ha_accuracy,
        check_ha_empty,
        check_ha_all_nan,
        check_ha_not_enough,
        check_ha_from_candles
    );
    fn check_ha_batch_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Use the batch builder (Auto‐detect, ScalarBatch, Avx2Batch, Avx512Batch)
        let output = HeikinAshiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;
        let len = output.open.len();
        assert_eq!(len, candles.close.len());

        let start = len.saturating_sub(5);

        // Replace these with the same expected numbers as above:
        let expected_last_five_open: [f64; 5] = [
            59348.5,
            59277.5,
            59233.0,
            59110.5,
            59097.0,
        ];
        let expected_last_five_high: [f64; 5] = [
            59348.5,
            59405.0,
            59304.0,
            59310.0,
            59236.0,
        ];
        let expected_last_five_low: [f64; 5] = [
            59001.0,
            59084.0,
            58932.0,
            58983.0,
            58299.0,
        ];
        let expected_last_five_close: [f64; 5] = [
            59221.75,
            59238.75,
            59114.25,
            59121.75,
            58836.25,
        ];

        for i in 0..5 {
            let idx = start + i;
            let got_open = output.open[idx];
            let got_high = output.high[idx];
            let got_low = output.low[idx];
            let got_close = output.close[idx];

            assert!(
                (got_open - expected_last_five_open[i]).abs() < 1e-8,
                "[{}] HA Batch {:?} open mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                idx,
                got_open,
                expected_last_five_open[i]
            );
            assert!(
                (got_high - expected_last_five_high[i]).abs() < 1e-8,
                "[{}] HA Batch {:?} high mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                idx,
                got_high,
                expected_last_five_high[i]
            );
            assert!(
                (got_low - expected_last_five_low[i]).abs() < 1e-8,
                "[{}] HA Batch {:?} low mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                idx,
                got_low,
                expected_last_five_low[i]
            );
            assert!(
                (got_close - expected_last_five_close[i]).abs() < 1e-8,
                "[{}] HA Batch {:?} close mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                idx,
                got_close,
                expected_last_five_close[i]
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

    gen_batch_tests!(check_ha_batch_accuracy);
}
