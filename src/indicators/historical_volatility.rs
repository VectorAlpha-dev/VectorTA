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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

impl<'a> AsRef<[f64]> for HistoricalVolatilityInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HistoricalVolatilityData::Slice(slice) => slice,
            HistoricalVolatilityData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HistoricalVolatilityData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HistoricalVolatilityParams {
    pub lookback: Option<usize>,
    pub annualization_days: Option<f64>,
}

impl Default for HistoricalVolatilityParams {
    fn default() -> Self {
        Self {
            lookback: Some(20),
            annualization_days: Some(250.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityInput<'a> {
    pub data: HistoricalVolatilityData<'a>,
    pub params: HistoricalVolatilityParams,
}

impl<'a> HistoricalVolatilityInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: HistoricalVolatilityParams,
    ) -> Self {
        Self {
            data: HistoricalVolatilityData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: HistoricalVolatilityParams) -> Self {
        Self {
            data: HistoricalVolatilityData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", HistoricalVolatilityParams::default())
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(20)
    }

    #[inline]
    pub fn get_annualization_days(&self) -> f64 {
        self.params.annualization_days.unwrap_or(250.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HistoricalVolatilityBuilder {
    lookback: Option<usize>,
    annualization_days: Option<f64>,
    kernel: Kernel,
}

impl Default for HistoricalVolatilityBuilder {
    fn default() -> Self {
        Self {
            lookback: None,
            annualization_days: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HistoricalVolatilityBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback(mut self, lookback: usize) -> Self {
        self.lookback = Some(lookback);
        self
    }

    #[inline(always)]
    pub fn annualization_days(mut self, annualization_days: f64) -> Self {
        self.annualization_days = Some(annualization_days);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<HistoricalVolatilityOutput, HistoricalVolatilityError> {
        let input = HistoricalVolatilityInput::from_candles(
            candles,
            source,
            HistoricalVolatilityParams {
                lookback: self.lookback,
                annualization_days: self.annualization_days,
            },
        );
        historical_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HistoricalVolatilityOutput, HistoricalVolatilityError> {
        let input = HistoricalVolatilityInput::from_slice(
            data,
            HistoricalVolatilityParams {
                lookback: self.lookback,
                annualization_days: self.annualization_days,
            },
        );
        historical_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<HistoricalVolatilityStream, HistoricalVolatilityError> {
        HistoricalVolatilityStream::try_new(HistoricalVolatilityParams {
            lookback: self.lookback,
            annualization_days: self.annualization_days,
        })
    }
}

#[derive(Debug, Error)]
pub enum HistoricalVolatilityError {
    #[error("historical_volatility: Input data slice is empty.")]
    EmptyInputData,
    #[error("historical_volatility: All values are NaN or do not produce valid returns.")]
    AllValuesNaN,
    #[error(
        "historical_volatility: Invalid lookback: lookback = {lookback}, data length = {data_len}"
    )]
    InvalidLookback { lookback: usize, data_len: usize },
    #[error("historical_volatility: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "historical_volatility: Invalid annualization_days: {annualization_days}. Must be finite and > 0."
    )]
    InvalidAnnualizationDays { annualization_days: f64 },
    #[error("historical_volatility: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("historical_volatility: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("historical_volatility: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityStream {
    lookback: usize,
    annualization_scale: f64,
    prev: f64,
    has_prev: bool,
    returns: Vec<f64>,
    valid: Vec<u8>,
    idx: usize,
    cnt: usize,
    valid_count: usize,
    sum: f64,
    sumsq: f64,
}

impl HistoricalVolatilityStream {
    pub fn try_new(
        params: HistoricalVolatilityParams,
    ) -> Result<HistoricalVolatilityStream, HistoricalVolatilityError> {
        let lookback = params.lookback.unwrap_or(20);
        if lookback == 0 {
            return Err(HistoricalVolatilityError::InvalidLookback {
                lookback,
                data_len: 0,
            });
        }
        let annualization_days = params.annualization_days.unwrap_or(250.0);
        if !annualization_days.is_finite() || annualization_days <= 0.0 {
            return Err(HistoricalVolatilityError::InvalidAnnualizationDays { annualization_days });
        }
        Ok(Self {
            lookback,
            annualization_scale: annualization_days.sqrt(),
            prev: f64::NAN,
            has_prev: false,
            returns: vec![0.0; lookback],
            valid: vec![0u8; lookback],
            idx: 0,
            cnt: 0,
            valid_count: 0,
            sum: 0.0,
            sumsq: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.has_prev {
            self.prev = value;
            self.has_prev = true;
            return None;
        }

        if self.cnt >= self.lookback {
            let old_idx = self.idx;
            if self.valid[old_idx] != 0 {
                let old = self.returns[old_idx];
                self.valid_count = self.valid_count.saturating_sub(1);
                self.sum -= old;
                self.sumsq -= old * old;
            }
        } else {
            self.cnt += 1;
        }

        if valid_return_pair(self.prev, value) {
            let ret = pct_return(self.prev, value);
            self.returns[self.idx] = ret;
            self.valid[self.idx] = 1;
            self.valid_count += 1;
            self.sum += ret;
            self.sumsq += ret * ret;
        } else {
            self.returns[self.idx] = 0.0;
            self.valid[self.idx] = 0;
        }

        self.prev = value;
        self.idx += 1;
        if self.idx == self.lookback {
            self.idx = 0;
        }

        if self.cnt < self.lookback {
            return None;
        }
        if self.valid_count != self.lookback {
            return Some(f64::NAN);
        }

        let mean = self.sum / self.lookback as f64;
        let variance = ((self.sumsq / self.lookback as f64) - mean * mean).max(0.0);
        Some(variance.sqrt() * self.annualization_scale)
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.lookback
    }
}

#[inline]
pub fn historical_volatility(
    input: &HistoricalVolatilityInput,
) -> Result<HistoricalVolatilityOutput, HistoricalVolatilityError> {
    historical_volatility_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_return_pair(prev: f64, curr: f64) -> bool {
    prev.is_finite() && curr.is_finite() && prev != 0.0
}

#[inline(always)]
fn pct_return(prev: f64, curr: f64) -> f64 {
    ((curr / prev) - 1.0) * 100.0
}

#[inline(always)]
fn first_valid_return(data: &[f64]) -> usize {
    let len = data.len();
    let mut i = 1usize;
    while i < len {
        if valid_return_pair(data[i - 1], data[i]) {
            return i;
        }
        i += 1;
    }
    len
}

#[inline(always)]
fn count_valid_returns(data: &[f64]) -> usize {
    let mut count = 0usize;
    for i in 1..data.len() {
        if valid_return_pair(data[i - 1], data[i]) {
            count += 1;
        }
    }
    count
}

#[inline(always)]
fn build_return_prefixes(data: &[f64]) -> (Vec<u32>, Vec<f64>, Vec<f64>) {
    let len = data.len();
    let mut prefix_valid = vec![0u32; len + 1];
    let mut prefix_sum = vec![0.0f64; len + 1];
    let mut prefix_sumsq = vec![0.0f64; len + 1];

    for i in 0..len {
        prefix_valid[i + 1] = prefix_valid[i];
        prefix_sum[i + 1] = prefix_sum[i];
        prefix_sumsq[i + 1] = prefix_sumsq[i];

        if i == 0 || !valid_return_pair(data[i - 1], data[i]) {
            continue;
        }

        let ret = pct_return(data[i - 1], data[i]);
        prefix_valid[i + 1] += 1;
        prefix_sum[i + 1] += ret;
        prefix_sumsq[i + 1] += ret * ret;
    }

    (prefix_valid, prefix_sum, prefix_sumsq)
}

#[inline(always)]
fn hv_row_from_prefix(
    prefix_valid: &[u32],
    prefix_sum: &[f64],
    prefix_sumsq: &[f64],
    lookback: usize,
    annualization_scale: f64,
    first: usize,
    out: &mut [f64],
) {
    let warmup = first.saturating_add(lookback.saturating_sub(1));
    let lookback_u32 = lookback as u32;
    let inv_lb = 1.0 / lookback as f64;

    for (t, slot) in out.iter_mut().enumerate() {
        if t < warmup {
            *slot = f64::NAN;
            continue;
        }

        let window_start = t + 1 - lookback;
        let valid_count = prefix_valid[t + 1] - prefix_valid[window_start];
        if valid_count != lookback_u32 {
            *slot = f64::NAN;
            continue;
        }

        let sum = prefix_sum[t + 1] - prefix_sum[window_start];
        let sumsq = prefix_sumsq[t + 1] - prefix_sumsq[window_start];
        let mean = sum * inv_lb;
        let variance = (sumsq * inv_lb - mean * mean).max(0.0);
        *slot = variance.sqrt() * annualization_scale;
    }
}

#[inline(always)]
fn historical_volatility_prepare<'a>(
    input: &'a HistoricalVolatilityInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, f64, Kernel), HistoricalVolatilityError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HistoricalVolatilityError::EmptyInputData);
    }

    let first = first_valid_return(data);
    if first >= len {
        return Err(HistoricalVolatilityError::AllValuesNaN);
    }

    let lookback = input.get_lookback();
    if lookback == 0 || lookback > len {
        return Err(HistoricalVolatilityError::InvalidLookback {
            lookback,
            data_len: len,
        });
    }

    let annualization_days = input.get_annualization_days();
    if !annualization_days.is_finite() || annualization_days <= 0.0 {
        return Err(HistoricalVolatilityError::InvalidAnnualizationDays { annualization_days });
    }

    let valid = count_valid_returns(data);
    if valid < lookback {
        return Err(HistoricalVolatilityError::NotEnoughValidData {
            needed: lookback,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };

    Ok((data, lookback, first, annualization_days.sqrt(), chosen))
}

#[inline]
pub fn historical_volatility_with_kernel(
    input: &HistoricalVolatilityInput,
    kernel: Kernel,
) -> Result<HistoricalVolatilityOutput, HistoricalVolatilityError> {
    let (data, lookback, first, annualization_scale, _chosen) =
        historical_volatility_prepare(input, kernel)?;
    let mut values =
        alloc_with_nan_prefix(data.len(), first.saturating_add(lookback.saturating_sub(1)));
    let (prefix_valid, prefix_sum, prefix_sumsq) = build_return_prefixes(data);
    hv_row_from_prefix(
        &prefix_valid,
        &prefix_sum,
        &prefix_sumsq,
        lookback,
        annualization_scale,
        first,
        &mut values,
    );
    Ok(HistoricalVolatilityOutput { values })
}

#[inline]
pub fn historical_volatility_into_slice(
    dst: &mut [f64],
    input: &HistoricalVolatilityInput,
    kernel: Kernel,
) -> Result<(), HistoricalVolatilityError> {
    let (data, lookback, first, annualization_scale, _chosen) =
        historical_volatility_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(HistoricalVolatilityError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    let (prefix_valid, prefix_sum, prefix_sumsq) = build_return_prefixes(data);
    hv_row_from_prefix(
        &prefix_valid,
        &prefix_sum,
        &prefix_sumsq,
        lookback,
        annualization_scale,
        first,
        dst,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn historical_volatility_into(
    input: &HistoricalVolatilityInput,
    out: &mut [f64],
) -> Result<(), HistoricalVolatilityError> {
    historical_volatility_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct HistoricalVolatilityBatchRange {
    pub lookback: (usize, usize, usize),
    pub annualization_days: (f64, f64, f64),
}

impl Default for HistoricalVolatilityBatchRange {
    fn default() -> Self {
        Self {
            lookback: (20, 252, 1),
            annualization_days: (250.0, 250.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HistoricalVolatilityBatchBuilder {
    range: HistoricalVolatilityBatchRange,
    kernel: Kernel,
}

impl HistoricalVolatilityBatchBuilder {
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
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }

    #[inline]
    pub fn lookback_static(mut self, lookback: usize) -> Self {
        self.range.lookback = (lookback, lookback, 0);
        self
    }

    #[inline]
    pub fn annualization_days_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.annualization_days = (start, end, step);
        self
    }

    #[inline]
    pub fn annualization_days_static(mut self, annualization_days: f64) -> Self {
        self.range.annualization_days = (annualization_days, annualization_days, 0.0);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HistoricalVolatilityBatchOutput, HistoricalVolatilityError> {
        historical_volatility_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<HistoricalVolatilityBatchOutput, HistoricalVolatilityError> {
        self.apply_slice(source_type(candles, source))
    }

    #[inline]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<HistoricalVolatilityBatchOutput, HistoricalVolatilityError> {
        HistoricalVolatilityBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles, "close")
    }
}

#[derive(Clone, Debug)]
pub struct HistoricalVolatilityBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HistoricalVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HistoricalVolatilityBatchOutput {
    pub fn row_for_params(&self, params: &HistoricalVolatilityParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.lookback.unwrap_or(20) == params.lookback.unwrap_or(20)
                && (combo.annualization_days.unwrap_or(250.0)
                    - params.annualization_days.unwrap_or(250.0))
                .abs()
                    < 1e-12
        })
    }

    pub fn values_for(&self, params: &HistoricalVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_grid_historical_volatility(
    range: &HistoricalVolatilityBatchRange,
) -> Result<Vec<HistoricalVolatilityParams>, HistoricalVolatilityError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, HistoricalVolatilityError> {
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
            return Err(HistoricalVolatilityError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, HistoricalVolatilityError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(HistoricalVolatilityError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let st = step.abs();
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += st;
            }
        } else {
            let st = -step.abs();
            let mut x = start;
            while x >= end - 1e-12 {
                out.push(x);
                x += st;
            }
        }

        if out.is_empty() {
            return Err(HistoricalVolatilityError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let lookbacks = axis_usize(range.lookback)?;
    if lookbacks.iter().any(|&lookback| lookback == 0) {
        return Err(HistoricalVolatilityError::InvalidLookback {
            lookback: 0,
            data_len: 0,
        });
    }

    let annualization_days = axis_f64(range.annualization_days)?;
    if let Some(&bad) = annualization_days
        .iter()
        .find(|&&annualization_days| !annualization_days.is_finite() || annualization_days <= 0.0)
    {
        return Err(HistoricalVolatilityError::InvalidAnnualizationDays {
            annualization_days: bad,
        });
    }

    let mut out = Vec::with_capacity(lookbacks.len() * annualization_days.len());
    for &lookback in &lookbacks {
        for &annualization_days in &annualization_days {
            out.push(HistoricalVolatilityParams {
                lookback: Some(lookback),
                annualization_days: Some(annualization_days),
            });
        }
    }
    Ok(out)
}

#[inline]
pub fn historical_volatility_batch_with_kernel(
    data: &[f64],
    sweep: &HistoricalVolatilityBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityBatchOutput, HistoricalVolatilityError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HistoricalVolatilityError::InvalidKernelForBatch(other)),
    };
    historical_volatility_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn historical_volatility_batch_slice(
    data: &[f64],
    sweep: &HistoricalVolatilityBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityBatchOutput, HistoricalVolatilityError> {
    historical_volatility_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn historical_volatility_batch_par_slice(
    data: &[f64],
    sweep: &HistoricalVolatilityBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityBatchOutput, HistoricalVolatilityError> {
    historical_volatility_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn historical_volatility_batch_inner(
    data: &[f64],
    sweep: &HistoricalVolatilityBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<HistoricalVolatilityBatchOutput, HistoricalVolatilityError> {
    let combos = expand_grid_historical_volatility(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(HistoricalVolatilityError::EmptyInputData);
    }
    let first = first_valid_return(data);
    if first >= cols {
        return Err(HistoricalVolatilityError::AllValuesNaN);
    }
    let valid = count_valid_returns(data);
    let max_lookback = combos
        .iter()
        .map(|combo| combo.lookback.unwrap_or(20))
        .max()
        .unwrap_or(0);
    if max_lookback == 0 || valid < max_lookback {
        return Err(HistoricalVolatilityError::NotEnoughValidData {
            needed: max_lookback,
            valid,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first.saturating_add(combo.lookback.unwrap_or(20).saturating_sub(1)))
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let (prefix_valid, prefix_sum, prefix_sumsq) = build_return_prefixes(data);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                hv_row_from_prefix(
                    &prefix_valid,
                    &prefix_sum,
                    &prefix_sumsq,
                    combo.lookback.unwrap_or(20),
                    combo.annualization_days.unwrap_or(250.0).sqrt(),
                    first,
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            hv_row_from_prefix(
                &prefix_valid,
                &prefix_sum,
                &prefix_sumsq,
                combo.lookback.unwrap_or(20),
                combo.annualization_days.unwrap_or(250.0).sqrt(),
                first,
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            hv_row_from_prefix(
                &prefix_valid,
                &prefix_sum,
                &prefix_sumsq,
                combo.lookback.unwrap_or(20),
                combo.annualization_days.unwrap_or(250.0).sqrt(),
                first,
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

    Ok(HistoricalVolatilityBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn historical_volatility_batch_inner_into(
    data: &[f64],
    sweep: &HistoricalVolatilityBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<HistoricalVolatilityParams>, HistoricalVolatilityError> {
    let combos = expand_grid_historical_volatility(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(HistoricalVolatilityError::EmptyInputData);
    }
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| HistoricalVolatilityError::OutputLengthMismatch {
                expected: usize::MAX,
                got: out.len(),
            })?;
    if out.len() != total {
        return Err(HistoricalVolatilityError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }
    let first = first_valid_return(data);
    if first >= cols {
        return Err(HistoricalVolatilityError::AllValuesNaN);
    }
    let valid = count_valid_returns(data);
    let max_lookback = combos
        .iter()
        .map(|combo| combo.lookback.unwrap_or(20))
        .max()
        .unwrap_or(0);
    if max_lookback == 0 || valid < max_lookback {
        return Err(HistoricalVolatilityError::NotEnoughValidData {
            needed: max_lookback,
            valid,
        });
    }

    let (prefix_valid, prefix_sum, prefix_sumsq) = build_return_prefixes(data);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                hv_row_from_prefix(
                    &prefix_valid,
                    &prefix_sum,
                    &prefix_sumsq,
                    combo.lookback.unwrap_or(20),
                    combo.annualization_days.unwrap_or(250.0).sqrt(),
                    first,
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            hv_row_from_prefix(
                &prefix_valid,
                &prefix_sum,
                &prefix_sumsq,
                combo.lookback.unwrap_or(20),
                combo.annualization_days.unwrap_or(250.0).sqrt(),
                first,
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            hv_row_from_prefix(
                &prefix_valid,
                &prefix_sum,
                &prefix_sumsq,
                combo.lookback.unwrap_or(20),
                combo.annualization_days.unwrap_or(250.0).sqrt(),
                first,
                out_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "historical_volatility")]
#[pyo3(signature = (data, lookback=20, annualization_days=250.0, kernel=None))]
pub fn historical_volatility_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    annualization_days: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = HistoricalVolatilityInput::from_slice(
        slice,
        HistoricalVolatilityParams {
            lookback: Some(lookback),
            annualization_days: Some(annualization_days),
        },
    );
    let output = py
        .allow_threads(|| historical_volatility_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "HistoricalVolatilityStream")]
pub struct HistoricalVolatilityStreamPy {
    stream: HistoricalVolatilityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HistoricalVolatilityStreamPy {
    #[new]
    fn new(lookback: usize, annualization_days: f64) -> PyResult<Self> {
        let stream = HistoricalVolatilityStream::try_new(HistoricalVolatilityParams {
            lookback: Some(lookback),
            annualization_days: Some(annualization_days),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "historical_volatility_batch")]
#[pyo3(signature = (data, lookback_range, annualization_days_range=(250.0, 250.0, 0.0), kernel=None))]
pub fn historical_volatility_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    annualization_days_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = HistoricalVolatilityBatchRange {
        lookback: lookback_range,
        annualization_days: annualization_days_range,
    };

    let combos = expand_grid_historical_volatility(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice.len();
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
            historical_volatility_batch_inner_into(
                slice,
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
        "lookbacks",
        combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "annualization_days",
        combos
            .iter()
            .map(|combo| combo.annualization_days.unwrap_or(250.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_historical_volatility_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(historical_volatility_py, module)?)?;
    module.add_function(wrap_pyfunction!(historical_volatility_batch_py, module)?)?;
    module.add_class::<HistoricalVolatilityStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "historical_volatility_js")]
pub fn historical_volatility_js(
    data: &[f64],
    lookback: usize,
    annualization_days: f64,
) -> Result<Vec<f64>, JsValue> {
    let input = HistoricalVolatilityInput::from_slice(
        data,
        HistoricalVolatilityParams {
            lookback: Some(lookback),
            annualization_days: Some(annualization_days),
        },
    );
    let mut output = vec![0.0; data.len()];
    historical_volatility_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback: usize,
    annualization_days: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = HistoricalVolatilityInput::from_slice(
            data,
            HistoricalVolatilityParams {
                lookback: Some(lookback),
                annualization_days: Some(annualization_days),
            },
        );

        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            historical_volatility_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            historical_volatility_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HistoricalVolatilityBatchConfig {
    pub lookback_range: (usize, usize, usize),
    pub annualization_days_range: Option<(f64, f64, f64)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HistoricalVolatilityBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HistoricalVolatilityParams>,
    pub lookbacks: Vec<usize>,
    pub annualization_days: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "historical_volatility_batch_js")]
pub fn historical_volatility_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: HistoricalVolatilityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = HistoricalVolatilityBatchRange {
        lookback: config.lookback_range,
        annualization_days: config
            .annualization_days_range
            .unwrap_or((250.0, 250.0, 0.0)),
    };
    let output = historical_volatility_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&HistoricalVolatilityBatchJsOutput {
        lookbacks: output
            .combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(20))
            .collect(),
        annualization_days: output
            .combos
            .iter()
            .map(|combo| combo.annualization_days.unwrap_or(250.0))
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
pub fn historical_volatility_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    annualization_days_start: f64,
    annualization_days_end: f64,
    annualization_days_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = HistoricalVolatilityBatchRange {
        lookback: (lookback_start, lookback_end, lookback_step),
        annualization_days: (
            annualization_days_start,
            annualization_days_end,
            annualization_days_step,
        ),
    };
    let combos =
        expand_grid_historical_volatility(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        historical_volatility_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_output_into_js(
    data: &[f64],
    lookback: usize,
    annualization_days: f64,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = historical_volatility_js(data, lookback, annualization_days)?;
    crate::write_wasm_f64_output("historical_volatility_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = historical_volatility_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "historical_volatility_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn load_close() -> Result<Vec<f64>, Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok(candles.close)
    }

    #[test]
    fn historical_volatility_output_contract() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = HistoricalVolatilityInput::from_slice(
            &close,
            HistoricalVolatilityParams {
                lookback: Some(20),
                annualization_days: Some(250.0),
            },
        );
        let out = historical_volatility_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.values.len(), close.len());
        let first_valid = out.values.iter().position(|v| !v.is_nan()).unwrap();
        assert!(first_valid >= 20);
        assert!(out.values[first_valid..].iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn historical_volatility_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = HistoricalVolatilityInput::from_slice(
            &close,
            HistoricalVolatilityParams {
                lookback: Some(30),
                annualization_days: Some(252.0),
            },
        );
        let auto = historical_volatility_with_kernel(&input, Kernel::Auto)?;
        let scalar = historical_volatility_with_kernel(&input, Kernel::Scalar)?;
        for (a, b) in auto.values.iter().zip(scalar.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn historical_volatility_rejects_invalid_annualization_days() {
        let data = [100.0, 101.0, 102.0, 103.0];
        let input = HistoricalVolatilityInput::from_slice(
            &data,
            HistoricalVolatilityParams {
                lookback: Some(2),
                annualization_days: Some(0.0),
            },
        );
        let err = historical_volatility_with_kernel(&input, Kernel::Scalar).unwrap_err();
        assert!(matches!(
            err,
            HistoricalVolatilityError::InvalidAnnualizationDays { .. }
        ));
    }

    #[test]
    fn historical_volatility_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let params = HistoricalVolatilityParams {
            lookback: Some(20),
            annualization_days: Some(250.0),
        };
        let input = HistoricalVolatilityInput::from_slice(&close, params.clone());
        let batch = historical_volatility_with_kernel(&input, Kernel::Scalar)?;
        let mut stream = HistoricalVolatilityStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(close.len());
        for &value in &close {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }
        for (a, b) in streamed.iter().zip(batch.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-10);
        }
        Ok(())
    }

    #[test]
    fn historical_volatility_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let sweep = HistoricalVolatilityBatchRange {
            lookback: (20, 20, 0),
            annualization_days: (250.0, 250.0, 0.0),
        };
        let batch = historical_volatility_batch_with_kernel(&close, &sweep, Kernel::ScalarBatch)?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        let single = historical_volatility_with_kernel(
            &HistoricalVolatilityInput::from_slice(
                &close,
                HistoricalVolatilityParams {
                    lookback: Some(20),
                    annualization_days: Some(250.0),
                },
            ),
            Kernel::Scalar,
        )?;
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn historical_volatility_nan_window_recovers() -> Result<(), Box<dyn Error>> {
        let mut close = load_close()?;
        close[40] = f64::NAN;
        let out = historical_volatility_with_kernel(
            &HistoricalVolatilityInput::from_slice(
                &close,
                HistoricalVolatilityParams {
                    lookback: Some(10),
                    annualization_days: Some(250.0),
                },
            ),
            Kernel::Scalar,
        )?;
        assert!(out.values[40].is_nan());
        assert!(out.values[49].is_nan());
        assert!(out.values[50].is_nan());
        assert!(out.values[51].is_finite());
        Ok(())
    }
}
