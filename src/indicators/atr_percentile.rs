#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AtrPercentileData<'a> {
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
pub struct AtrPercentileOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AtrPercentileParams {
    pub atr_length: Option<usize>,
    pub percentile_length: Option<usize>,
}

impl Default for AtrPercentileParams {
    fn default() -> Self {
        Self {
            atr_length: Some(10),
            percentile_length: Some(50),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AtrPercentileInput<'a> {
    pub data: AtrPercentileData<'a>,
    pub params: AtrPercentileParams,
}

impl<'a> AtrPercentileInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AtrPercentileParams) -> Self {
        Self {
            data: AtrPercentileData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AtrPercentileParams,
    ) -> Self {
        Self {
            data: AtrPercentileData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AtrPercentileParams::default())
    }

    #[inline]
    pub fn get_atr_length(&self) -> usize {
        self.params.atr_length.unwrap_or(10)
    }

    #[inline]
    pub fn get_percentile_length(&self) -> usize {
        self.params.percentile_length.unwrap_or(50)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AtrPercentileBuilder {
    atr_length: Option<usize>,
    percentile_length: Option<usize>,
    kernel: Kernel,
}

impl Default for AtrPercentileBuilder {
    fn default() -> Self {
        Self {
            atr_length: None,
            percentile_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AtrPercentileBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn atr_length(mut self, atr_length: usize) -> Self {
        self.atr_length = Some(atr_length);
        self
    }

    #[inline(always)]
    pub fn percentile_length(mut self, percentile_length: usize) -> Self {
        self.percentile_length = Some(percentile_length);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<AtrPercentileOutput, AtrPercentileError> {
        let input = AtrPercentileInput::from_candles(
            candles,
            AtrPercentileParams {
                atr_length: self.atr_length,
                percentile_length: self.percentile_length,
            },
        );
        atr_percentile_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AtrPercentileOutput, AtrPercentileError> {
        let input = AtrPercentileInput::from_slices(
            high,
            low,
            close,
            AtrPercentileParams {
                atr_length: self.atr_length,
                percentile_length: self.percentile_length,
            },
        );
        atr_percentile_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AtrPercentileStream, AtrPercentileError> {
        AtrPercentileStream::try_new(AtrPercentileParams {
            atr_length: self.atr_length,
            percentile_length: self.percentile_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum AtrPercentileError {
    #[error("atr_percentile: Input data slice is empty.")]
    EmptyInputData,
    #[error("atr_percentile: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "atr_percentile: Invalid ATR length: atr_length = {atr_length}, data length = {data_len}"
    )]
    InvalidAtrLength { atr_length: usize, data_len: usize },
    #[error(
        "atr_percentile: Invalid percentile length: percentile_length = {percentile_length}, data length = {data_len}"
    )]
    InvalidPercentileLength {
        percentile_length: usize,
        data_len: usize,
    },
    #[error("atr_percentile: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "atr_percentile: Inconsistent slice lengths: high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("atr_percentile: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("atr_percentile: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("atr_percentile: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct AtrPercentileStream {
    atr_length: usize,
    percentile_length: usize,
    prev_close: f64,
    has_prev_close: bool,
    tr_values: Vec<f64>,
    tr_valid: Vec<u8>,
    tr_idx: usize,
    tr_count: usize,
    tr_valid_count: usize,
    tr_sum: f64,
    atr_values: Vec<f64>,
    atr_valid: Vec<u8>,
    atr_idx: usize,
    atr_count: usize,
    atr_valid_count: usize,
}

impl AtrPercentileStream {
    pub fn try_new(params: AtrPercentileParams) -> Result<AtrPercentileStream, AtrPercentileError> {
        let atr_length = params.atr_length.unwrap_or(10);
        let percentile_length = params.percentile_length.unwrap_or(50);
        if atr_length == 0 {
            return Err(AtrPercentileError::InvalidAtrLength {
                atr_length,
                data_len: 0,
            });
        }
        if percentile_length == 0 {
            return Err(AtrPercentileError::InvalidPercentileLength {
                percentile_length,
                data_len: 0,
            });
        }
        Ok(Self {
            atr_length,
            percentile_length,
            prev_close: f64::NAN,
            has_prev_close: false,
            tr_values: vec![0.0; atr_length],
            tr_valid: vec![0u8; atr_length],
            tr_idx: 0,
            tr_count: 0,
            tr_valid_count: 0,
            tr_sum: 0.0,
            atr_values: vec![0.0; percentile_length],
            atr_valid: vec![0u8; percentile_length],
            atr_idx: 0,
            atr_count: 0,
            atr_valid_count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if self.tr_count >= self.atr_length {
            let old_idx = self.tr_idx;
            if self.tr_valid[old_idx] != 0 {
                self.tr_valid_count = self.tr_valid_count.saturating_sub(1);
                self.tr_sum -= self.tr_values[old_idx];
            }
        } else {
            self.tr_count += 1;
        }

        let tr = atr_percentile_true_range(high, low, close, self.prev_close, self.has_prev_close);
        if let Some(value) = tr {
            self.tr_values[self.tr_idx] = value;
            self.tr_valid[self.tr_idx] = 1;
            self.tr_valid_count += 1;
            self.tr_sum += value;
        } else {
            self.tr_values[self.tr_idx] = 0.0;
            self.tr_valid[self.tr_idx] = 0;
        }
        self.tr_idx += 1;
        if self.tr_idx == self.atr_length {
            self.tr_idx = 0;
        }

        let result = if self.tr_count < self.atr_length {
            None
        } else {
            let atr_valid_now = self.tr_valid_count == self.atr_length;
            let atr_now = if atr_valid_now {
                self.tr_sum / self.atr_length as f64
            } else {
                0.0
            };

            let out = if self.atr_count < self.percentile_length {
                None
            } else if atr_valid_now && self.atr_valid_count == self.percentile_length {
                let mut below = 0usize;
                for &prev in &self.atr_values {
                    if atr_now > prev {
                        below += 1;
                    }
                }
                Some(100.0 * below as f64 / self.percentile_length as f64)
            } else {
                Some(f64::NAN)
            };

            if self.atr_count >= self.percentile_length {
                let old_idx = self.atr_idx;
                if self.atr_valid[old_idx] != 0 {
                    self.atr_valid_count = self.atr_valid_count.saturating_sub(1);
                }
            } else {
                self.atr_count += 1;
            }

            if atr_valid_now {
                self.atr_values[self.atr_idx] = atr_now;
                self.atr_valid[self.atr_idx] = 1;
                self.atr_valid_count += 1;
            } else {
                self.atr_values[self.atr_idx] = 0.0;
                self.atr_valid[self.atr_idx] = 0;
            }
            self.atr_idx += 1;
            if self.atr_idx == self.percentile_length {
                self.atr_idx = 0;
            }

            out
        };

        if valid_hlc_bar(high, low, close) {
            self.prev_close = close;
            self.has_prev_close = true;
        } else {
            self.prev_close = f64::NAN;
            self.has_prev_close = false;
        }

        result
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.atr_length + self.percentile_length - 1
    }
}

#[inline]
pub fn atr_percentile(
    input: &AtrPercentileInput,
) -> Result<AtrPercentileOutput, AtrPercentileError> {
    atr_percentile_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_hlc_bar(high: f64, low: f64, close: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn atr_percentile_true_range(
    high: f64,
    low: f64,
    close: f64,
    prev_close: f64,
    has_prev_close: bool,
) -> Option<f64> {
    if !valid_hlc_bar(high, low, close) {
        return None;
    }
    let hl = high - low;
    if !has_prev_close || !prev_close.is_finite() {
        return Some(hl);
    }
    let hc = (high - prev_close).abs();
    let lc = (low - prev_close).abs();
    Some(hl.max(hc).max(lc))
}

#[inline(always)]
fn first_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let len = close.len();
    let mut i = 0usize;
    while i < len {
        if valid_hlc_bar(high[i], low[i], close[i]) {
            break;
        }
        i += 1;
    }
    i.min(len)
}

#[inline(always)]
fn count_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut count = 0usize;
    for i in 0..close.len() {
        if valid_hlc_bar(high[i], low[i], close[i]) {
            count += 1;
        }
    }
    count
}

#[inline(always)]
fn first_and_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> (usize, usize) {
    let mut first = close.len();
    let mut count = 0usize;
    for i in 0..close.len() {
        if valid_hlc_bar(high[i], low[i], close[i]) {
            if first == close.len() {
                first = i;
            }
            count += 1;
        }
    }
    (first, count)
}

#[inline(always)]
fn atr_percentile_row_from_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    percentile_length: usize,
    out: &mut [f64],
) {
    let mut prev_close = f64::NAN;
    let mut has_prev_close = false;

    let mut tr_values = vec![0.0f64; atr_length];
    let mut tr_valid = vec![0u8; atr_length];
    let mut tr_idx = 0usize;
    let mut tr_count = 0usize;
    let mut tr_valid_count = 0usize;
    let mut tr_sum = 0.0f64;

    let mut atr_values = vec![0.0f64; percentile_length];
    let mut atr_valid = vec![0u8; percentile_length];
    let mut atr_idx = 0usize;
    let mut atr_count = 0usize;
    let mut atr_valid_count = 0usize;

    for i in 0..close.len() {
        if tr_count >= atr_length {
            let old_idx = tr_idx;
            if tr_valid[old_idx] != 0 {
                tr_valid_count = tr_valid_count.saturating_sub(1);
                tr_sum -= tr_values[old_idx];
            }
        } else {
            tr_count += 1;
        }

        let tr = atr_percentile_true_range(high[i], low[i], close[i], prev_close, has_prev_close);
        if let Some(value) = tr {
            tr_values[tr_idx] = value;
            tr_valid[tr_idx] = 1;
            tr_valid_count += 1;
            tr_sum += value;
        } else {
            tr_values[tr_idx] = 0.0;
            tr_valid[tr_idx] = 0;
        }
        tr_idx += 1;
        if tr_idx == atr_length {
            tr_idx = 0;
        }

        if tr_count >= atr_length {
            let atr_valid_now = tr_valid_count == atr_length;
            let atr_now = if atr_valid_now {
                tr_sum / atr_length as f64
            } else {
                0.0
            };

            if atr_count >= percentile_length {
                if atr_valid_now && atr_valid_count == percentile_length {
                    let mut below = 0usize;
                    for &prev in &atr_values {
                        if atr_now > prev {
                            below += 1;
                        }
                    }
                    out[i] = 100.0 * below as f64 / percentile_length as f64;
                } else {
                    out[i] = f64::NAN;
                }
            }

            if atr_count >= percentile_length {
                let old_idx = atr_idx;
                if atr_valid[old_idx] != 0 {
                    atr_valid_count = atr_valid_count.saturating_sub(1);
                }
            } else {
                atr_count += 1;
            }

            if atr_valid_now {
                atr_values[atr_idx] = atr_now;
                atr_valid[atr_idx] = 1;
                atr_valid_count += 1;
            } else {
                atr_values[atr_idx] = 0.0;
                atr_valid[atr_idx] = 0;
            }
            atr_idx += 1;
            if atr_idx == percentile_length {
                atr_idx = 0;
            }
        }

        if valid_hlc_bar(high[i], low[i], close[i]) {
            prev_close = close[i];
            has_prev_close = true;
        } else {
            prev_close = f64::NAN;
            has_prev_close = false;
        }
    }
}

#[inline(always)]
fn atr_percentile_prepare<'a>(
    input: &'a AtrPercentileInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, usize, Kernel), AtrPercentileError> {
    let (high, low, close) = match &input.data {
        AtrPercentileData::Candles { candles } => {
            (&candles.high[..], &candles.low[..], &candles.close[..])
        }
        AtrPercentileData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(AtrPercentileError::InconsistentSliceLengths {
                    high_len: high.len(),
                    low_len: low.len(),
                    close_len: close.len(),
                });
            }
            (*high, *low, *close)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(AtrPercentileError::EmptyInputData);
    }

    let (first, valid) = first_and_valid_hlc(high, low, close);
    if first >= len {
        return Err(AtrPercentileError::AllValuesNaN);
    }

    let atr_length = input.get_atr_length();
    if atr_length == 0 || atr_length > len {
        return Err(AtrPercentileError::InvalidAtrLength {
            atr_length,
            data_len: len,
        });
    }

    let percentile_length = input.get_percentile_length();
    if percentile_length == 0 || percentile_length > len {
        return Err(AtrPercentileError::InvalidPercentileLength {
            percentile_length,
            data_len: len,
        });
    }

    let needed = atr_length.saturating_add(percentile_length);
    if valid < needed {
        return Err(AtrPercentileError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((
        high,
        low,
        close,
        atr_length,
        percentile_length,
        first,
        chosen,
    ))
}

pub fn atr_percentile_with_kernel(
    input: &AtrPercentileInput,
    kernel: Kernel,
) -> Result<AtrPercentileOutput, AtrPercentileError> {
    let (high, low, close, atr_length, percentile_length, first, _chosen) =
        atr_percentile_prepare(input, kernel)?;
    let warmup = first
        .saturating_add(atr_length)
        .saturating_add(percentile_length)
        .saturating_sub(1);
    let mut values = alloc_with_nan_prefix(high.len(), warmup);
    atr_percentile_row_from_slices(high, low, close, atr_length, percentile_length, &mut values);
    Ok(AtrPercentileOutput { values })
}

#[inline]
pub fn atr_percentile_into_slice(
    dst: &mut [f64],
    input: &AtrPercentileInput,
    kernel: Kernel,
) -> Result<(), AtrPercentileError> {
    let (high, low, close, atr_length, percentile_length, _first, _chosen) =
        atr_percentile_prepare(input, kernel)?;
    if dst.len() != high.len() {
        return Err(AtrPercentileError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);
    atr_percentile_row_from_slices(high, low, close, atr_length, percentile_length, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn atr_percentile_into(
    input: &AtrPercentileInput,
    out: &mut [f64],
) -> Result<(), AtrPercentileError> {
    atr_percentile_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct AtrPercentileBatchRange {
    pub atr_length: (usize, usize, usize),
    pub percentile_length: (usize, usize, usize),
}

impl Default for AtrPercentileBatchRange {
    fn default() -> Self {
        Self {
            atr_length: (10, 252, 1),
            percentile_length: (50, 252, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AtrPercentileBatchBuilder {
    range: AtrPercentileBatchRange,
    kernel: Kernel,
}

impl AtrPercentileBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn atr_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.atr_length = (start, end, step);
        self
    }

    #[inline]
    pub fn atr_length_static(mut self, atr_length: usize) -> Self {
        self.range.atr_length = (atr_length, atr_length, 0);
        self
    }

    #[inline]
    pub fn percentile_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.percentile_length = (start, end, step);
        self
    }

    #[inline]
    pub fn percentile_length_static(mut self, percentile_length: usize) -> Self {
        self.range.percentile_length = (percentile_length, percentile_length, 0);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AtrPercentileBatchOutput, AtrPercentileError> {
        atr_percentile_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<AtrPercentileBatchOutput, AtrPercentileError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }

    #[inline]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<AtrPercentileBatchOutput, AtrPercentileError> {
        AtrPercentileBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles)
    }
}

#[derive(Clone, Debug)]
pub struct AtrPercentileBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AtrPercentileParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AtrPercentileBatchOutput {
    pub fn row_for_params(&self, params: &AtrPercentileParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.atr_length.unwrap_or(10) == params.atr_length.unwrap_or(10)
                && combo.percentile_length.unwrap_or(50) == params.percentile_length.unwrap_or(50)
        })
    }

    pub fn values_for(&self, params: &AtrPercentileParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, AtrPercentileError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            let next = x.saturating_add(step);
            if next == x {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(AtrPercentileError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_atr_percentile(
    range: &AtrPercentileBatchRange,
) -> Result<Vec<AtrPercentileParams>, AtrPercentileError> {
    let atr_lengths = expand_axis_usize(range.atr_length)?;
    let percentile_lengths = expand_axis_usize(range.percentile_length)?;

    if let Some(&bad) = atr_lengths.iter().find(|&&x| x == 0) {
        return Err(AtrPercentileError::InvalidAtrLength {
            atr_length: bad,
            data_len: 0,
        });
    }
    if let Some(&bad) = percentile_lengths.iter().find(|&&x| x == 0) {
        return Err(AtrPercentileError::InvalidPercentileLength {
            percentile_length: bad,
            data_len: 0,
        });
    }

    let mut out = Vec::with_capacity(atr_lengths.len() * percentile_lengths.len());
    for &atr_length in &atr_lengths {
        for &percentile_length in &percentile_lengths {
            out.push(AtrPercentileParams {
                atr_length: Some(atr_length),
                percentile_length: Some(percentile_length),
            });
        }
    }
    Ok(out)
}

#[inline]
pub fn atr_percentile_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrPercentileBatchRange,
    kernel: Kernel,
) -> Result<AtrPercentileBatchOutput, AtrPercentileError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AtrPercentileError::InvalidKernelForBatch(other)),
    };
    atr_percentile_batch_par_slice(high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn atr_percentile_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrPercentileBatchRange,
    kernel: Kernel,
) -> Result<AtrPercentileBatchOutput, AtrPercentileError> {
    atr_percentile_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline]
pub fn atr_percentile_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrPercentileBatchRange,
    kernel: Kernel,
) -> Result<AtrPercentileBatchOutput, AtrPercentileError> {
    atr_percentile_batch_inner(high, low, close, sweep, kernel, true)
}

#[inline(always)]
fn atr_percentile_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrPercentileBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<AtrPercentileBatchOutput, AtrPercentileError> {
    let combos = expand_grid_atr_percentile(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(AtrPercentileError::EmptyInputData);
    }
    if high.len() != cols || low.len() != cols {
        return Err(AtrPercentileError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let first = first_valid_hlc(high, low, close);
    if first >= cols {
        return Err(AtrPercentileError::AllValuesNaN);
    }

    let valid = count_valid_hlc(high, low, close);
    let max_needed = combos
        .iter()
        .map(|combo| combo.atr_length.unwrap_or(10) + combo.percentile_length.unwrap_or(50))
        .max()
        .unwrap_or(0);
    if valid < max_needed {
        return Err(AtrPercentileError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first
                .saturating_add(combo.atr_length.unwrap_or(10))
                .saturating_add(combo.percentile_length.unwrap_or(50))
                .saturating_sub(1)
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                atr_percentile_row_from_slices(
                    high,
                    low,
                    close,
                    combo.atr_length.unwrap_or(10),
                    combo.percentile_length.unwrap_or(50),
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            atr_percentile_row_from_slices(
                high,
                low,
                close,
                combo.atr_length.unwrap_or(10),
                combo.percentile_length.unwrap_or(50),
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            atr_percentile_row_from_slices(
                high,
                low,
                close,
                combo.atr_length.unwrap_or(10),
                combo.percentile_length.unwrap_or(50),
                out_row,
            );
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(AtrPercentileBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn atr_percentile_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrPercentileBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AtrPercentileParams>, AtrPercentileError> {
    let combos = expand_grid_atr_percentile(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(AtrPercentileError::EmptyInputData);
    }
    if high.len() != cols || low.len() != cols {
        return Err(AtrPercentileError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| AtrPercentileError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        })?;
    if out.len() != total {
        return Err(AtrPercentileError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }

    let first = first_valid_hlc(high, low, close);
    if first >= cols {
        return Err(AtrPercentileError::AllValuesNaN);
    }

    let valid = count_valid_hlc(high, low, close);
    let max_needed = combos
        .iter()
        .map(|combo| combo.atr_length.unwrap_or(10) + combo.percentile_length.unwrap_or(50))
        .max()
        .unwrap_or(0);
    if valid < max_needed {
        return Err(AtrPercentileError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    out.fill(f64::NAN);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                atr_percentile_row_from_slices(
                    high,
                    low,
                    close,
                    combo.atr_length.unwrap_or(10),
                    combo.percentile_length.unwrap_or(50),
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            atr_percentile_row_from_slices(
                high,
                low,
                close,
                combo.atr_length.unwrap_or(10),
                combo.percentile_length.unwrap_or(50),
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            atr_percentile_row_from_slices(
                high,
                low,
                close,
                combo.atr_length.unwrap_or(10),
                combo.percentile_length.unwrap_or(50),
                out_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "atr_percentile")]
#[pyo3(signature = (high, low, close, atr_length=10, percentile_length=50, kernel=None))]
pub fn atr_percentile_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    atr_length: usize,
    percentile_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if high.len() != low.len() || low.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kernel = validate_kernel(kernel, false)?;
    let input = AtrPercentileInput::from_slices(
        high,
        low,
        close,
        AtrPercentileParams {
            atr_length: Some(atr_length),
            percentile_length: Some(percentile_length),
        },
    );
    let output = py
        .allow_threads(|| atr_percentile_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AtrPercentileStream")]
pub struct AtrPercentileStreamPy {
    stream: AtrPercentileStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AtrPercentileStreamPy {
    #[new]
    fn new(atr_length: usize, percentile_length: usize) -> PyResult<Self> {
        let stream = AtrPercentileStream::try_new(AtrPercentileParams {
            atr_length: Some(atr_length),
            percentile_length: Some(percentile_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "atr_percentile_batch")]
#[pyo3(signature = (high, low, close, atr_length_range, percentile_length_range=(50, 50, 0), kernel=None))]
pub fn atr_percentile_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    atr_length_range: (usize, usize, usize),
    percentile_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if high.len() != low.len() || low.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kernel = validate_kernel(kernel, true)?;
    let sweep = AtrPercentileBatchRange {
        atr_length: atr_length_range,
        percentile_length: percentile_length_range,
    };
    let combos =
        expand_grid_atr_percentile(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            atr_percentile_batch_inner_into(
                high,
                low,
                close,
                &sweep,
                batch.to_non_batch(),
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "atr_lengths",
        combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(10) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "percentile_lengths",
        combos
            .iter()
            .map(|combo| combo.percentile_length.unwrap_or(50) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_atr_percentile_module(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(atr_percentile_py, module)?)?;
    module.add_function(wrap_pyfunction!(atr_percentile_batch_py, module)?)?;
    module.add_class::<AtrPercentileStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "atr_percentile_js")]
pub fn atr_percentile_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    percentile_length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = AtrPercentileInput::from_slices(
        high,
        low,
        close,
        AtrPercentileParams {
            atr_length: Some(atr_length),
            percentile_length: Some(percentile_length),
        },
    );
    let mut output = vec![0.0; close.len()];
    atr_percentile_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn atr_percentile_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn atr_percentile_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn atr_percentile_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    atr_length: usize,
    percentile_length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = AtrPercentileInput::from_slices(
            high,
            low,
            close,
            AtrPercentileParams {
                atr_length: Some(atr_length),
                percentile_length: Some(percentile_length),
            },
        );
        if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            atr_percentile_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            atr_percentile_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AtrPercentileBatchConfig {
    pub atr_length_range: (usize, usize, usize),
    pub percentile_length_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AtrPercentileBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AtrPercentileParams>,
    pub atr_lengths: Vec<usize>,
    pub percentile_lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "atr_percentile_batch_js")]
pub fn atr_percentile_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AtrPercentileBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = AtrPercentileBatchRange {
        atr_length: config.atr_length_range,
        percentile_length: config.percentile_length_range.unwrap_or((50, 50, 0)),
    };
    let output = atr_percentile_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&AtrPercentileBatchJsOutput {
        atr_lengths: output
            .combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(10))
            .collect(),
        percentile_lengths: output
            .combos
            .iter()
            .map(|combo| combo.percentile_length.unwrap_or(50))
            .collect(),
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn atr_percentile_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    percentile_length_start: usize,
    percentile_length_end: usize,
    percentile_length_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = AtrPercentileBatchRange {
        atr_length: (atr_length_start, atr_length_end, atr_length_step),
        percentile_length: (
            percentile_length_start,
            percentile_length_end,
            percentile_length_step,
        ),
    };
    let combos =
        expand_grid_atr_percentile(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        atr_percentile_batch_inner_into(high, low, close, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn atr_percentile_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    percentile_length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = atr_percentile_js(high, low, close, atr_length, percentile_length)?;
    crate::write_wasm_f64_output("atr_percentile_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn atr_percentile_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = atr_percentile_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "atr_percentile_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn load_ohlc() -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok((candles.high, candles.low, candles.close))
    }

    #[test]
    fn atr_percentile_output_contract() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = load_ohlc()?;
        let input = AtrPercentileInput::from_slices(
            &high,
            &low,
            &close,
            AtrPercentileParams {
                atr_length: Some(10),
                percentile_length: Some(50),
            },
        );
        let out = atr_percentile_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.values.len(), close.len());
        let first_valid = out.values.iter().position(|v| !v.is_nan()).unwrap();
        assert!(first_valid >= 59);
        assert!(out.values[first_valid..].iter().any(|v| v.is_finite()));
        for &value in out.values[first_valid + 16..].iter().take(64) {
            if value.is_finite() {
                assert!((0.0..=100.0).contains(&value));
            }
        }
        Ok(())
    }

    #[test]
    fn atr_percentile_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = load_ohlc()?;
        let input = AtrPercentileInput::from_slices(
            &high,
            &low,
            &close,
            AtrPercentileParams {
                atr_length: Some(7),
                percentile_length: Some(21),
            },
        );
        let auto = atr_percentile_with_kernel(&input, Kernel::Auto)?;
        let scalar = atr_percentile_with_kernel(&input, Kernel::Scalar)?;
        for (a, b) in auto.values.iter().zip(scalar.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn atr_percentile_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = load_ohlc()?;
        let input = AtrPercentileInput::from_slices(
            &high,
            &low,
            &close,
            AtrPercentileParams {
                atr_length: Some(8),
                percentile_length: Some(16),
            },
        );
        let api = atr_percentile_with_kernel(&input, Kernel::Auto)?;
        let mut out = vec![0.0; close.len()];
        atr_percentile_into(&input, &mut out)?;
        for (a, b) in api.values.iter().zip(out.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() <= 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn atr_percentile_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = load_ohlc()?;
        let params = AtrPercentileParams {
            atr_length: Some(6),
            percentile_length: Some(12),
        };
        let input = AtrPercentileInput::from_slices(&high, &low, &close, params.clone());
        let batch = atr_percentile_with_kernel(&input, Kernel::Scalar)?;
        let mut stream = AtrPercentileStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            streamed.push(stream.update(high[i], low[i], close[i]).unwrap_or(f64::NAN));
        }
        for (a, b) in streamed.iter().zip(batch.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn atr_percentile_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = load_ohlc()?;
        let single = atr_percentile_with_kernel(
            &AtrPercentileInput::from_slices(
                &high,
                &low,
                &close,
                AtrPercentileParams {
                    atr_length: Some(10),
                    percentile_length: Some(20),
                },
            ),
            Kernel::Scalar,
        )?;
        let batch = atr_percentile_batch_with_kernel(
            &high,
            &low,
            &close,
            &AtrPercentileBatchRange {
                atr_length: (10, 10, 0),
                percentile_length: (20, 20, 0),
            },
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn atr_percentile_invalid_window_recovers() -> Result<(), Box<dyn Error>> {
        let (mut high, mut low, mut close) = load_ohlc()?;
        high.truncate(96);
        low.truncate(96);
        close.truncate(96);
        high[30] = f64::NAN;
        low[30] = f64::NAN;
        close[30] = f64::NAN;

        let out = atr_percentile_with_kernel(
            &AtrPercentileInput::from_slices(
                &high,
                &low,
                &close,
                AtrPercentileParams {
                    atr_length: Some(5),
                    percentile_length: Some(5),
                },
            ),
            Kernel::Scalar,
        )?;
        assert!(out.values[30].is_nan());
        assert!(out.values[38].is_nan());
        assert!(out.values[39].is_nan());
        assert!(out.values[40].is_finite());
        Ok(())
    }

    #[test]
    fn atr_percentile_rejects_invalid_lengths() {
        let high = [2.0, 3.0, 4.0];
        let low = [1.0, 2.0, 3.0];
        let close = [1.5, 2.5, 3.5];

        let err = atr_percentile_with_kernel(
            &AtrPercentileInput::from_slices(
                &high,
                &low,
                &close,
                AtrPercentileParams {
                    atr_length: Some(0),
                    percentile_length: Some(2),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, AtrPercentileError::InvalidAtrLength { .. }));

        let err = atr_percentile_with_kernel(
            &AtrPercentileInput::from_slices(
                &high,
                &low,
                &close,
                AtrPercentileParams {
                    atr_length: Some(2),
                    percentile_length: Some(0),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AtrPercentileError::InvalidPercentileLength { .. }
        ));
    }
}
