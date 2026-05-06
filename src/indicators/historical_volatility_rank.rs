#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for HistoricalVolatilityRankInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HistoricalVolatilityRankData::Slice(slice) => slice,
            HistoricalVolatilityRankData::Candles { candles } => candles.close.as_slice(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HistoricalVolatilityRankData<'a> {
    Candles { candles: &'a Candles },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityRankOutput {
    pub hvr: Vec<f64>,
    pub hv: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalVolatilityRankOutputField {
    Hvr,
    Hv,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HistoricalVolatilityRankParams {
    pub hv_length: Option<usize>,
    pub rank_length: Option<usize>,
    pub annualization_days: Option<f64>,
    pub bar_days: Option<f64>,
}

impl Default for HistoricalVolatilityRankParams {
    fn default() -> Self {
        Self {
            hv_length: Some(10),
            rank_length: Some(52 * 7),
            annualization_days: Some(365.0),
            bar_days: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityRankInput<'a> {
    pub data: HistoricalVolatilityRankData<'a>,
    pub params: HistoricalVolatilityRankParams,
}

impl<'a> HistoricalVolatilityRankInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: HistoricalVolatilityRankParams) -> Self {
        Self {
            data: HistoricalVolatilityRankData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: HistoricalVolatilityRankParams) -> Self {
        Self {
            data: HistoricalVolatilityRankData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, HistoricalVolatilityRankParams::default())
    }

    #[inline]
    pub fn get_hv_length(&self) -> usize {
        self.params.hv_length.unwrap_or(10)
    }

    #[inline]
    pub fn get_rank_length(&self) -> usize {
        self.params.rank_length.unwrap_or(52 * 7)
    }

    #[inline]
    pub fn get_annualization_days(&self) -> f64 {
        self.params.annualization_days.unwrap_or(365.0)
    }

    #[inline]
    pub fn get_bar_days(&self) -> f64 {
        self.params.bar_days.unwrap_or(1.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HistoricalVolatilityRankBuilder {
    hv_length: Option<usize>,
    rank_length: Option<usize>,
    annualization_days: Option<f64>,
    bar_days: Option<f64>,
    kernel: Kernel,
}

impl Default for HistoricalVolatilityRankBuilder {
    fn default() -> Self {
        Self {
            hv_length: None,
            rank_length: None,
            annualization_days: None,
            bar_days: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HistoricalVolatilityRankBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn hv_length(mut self, value: usize) -> Self {
        self.hv_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn rank_length(mut self, value: usize) -> Self {
        self.rank_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn annualization_days(mut self, value: f64) -> Self {
        self.annualization_days = Some(value);
        self
    }

    #[inline(always)]
    pub fn bar_days(mut self, value: f64) -> Self {
        self.bar_days = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<HistoricalVolatilityRankOutput, HistoricalVolatilityRankError> {
        let params = HistoricalVolatilityRankParams {
            hv_length: self.hv_length,
            rank_length: self.rank_length,
            annualization_days: self.annualization_days,
            bar_days: self.bar_days,
        };
        historical_volatility_rank_with_kernel(
            &HistoricalVolatilityRankInput::from_candles(candles, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HistoricalVolatilityRankOutput, HistoricalVolatilityRankError> {
        let params = HistoricalVolatilityRankParams {
            hv_length: self.hv_length,
            rank_length: self.rank_length,
            annualization_days: self.annualization_days,
            bar_days: self.bar_days,
        };
        historical_volatility_rank_with_kernel(
            &HistoricalVolatilityRankInput::from_slice(data, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<HistoricalVolatilityRankStream, HistoricalVolatilityRankError> {
        HistoricalVolatilityRankStream::try_new(HistoricalVolatilityRankParams {
            hv_length: self.hv_length,
            rank_length: self.rank_length,
            annualization_days: self.annualization_days,
            bar_days: self.bar_days,
        })
    }
}

#[derive(Debug, Error)]
pub enum HistoricalVolatilityRankError {
    #[error("historical_volatility_rank: Input data slice is empty.")]
    EmptyInputData,
    #[error("historical_volatility_rank: All values are NaN or non-positive.")]
    AllValuesNaN,
    #[error(
        "historical_volatility_rank: Invalid hv_length: hv_length = {hv_length}, data length = {data_len}"
    )]
    InvalidHvLength { hv_length: usize, data_len: usize },
    #[error("historical_volatility_rank: Invalid rank_length: rank_length = {rank_length}")]
    InvalidRankLength { rank_length: usize },
    #[error(
        "historical_volatility_rank: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "historical_volatility_rank: Invalid annualization_days: {annualization_days}. Must be positive and finite."
    )]
    InvalidAnnualizationDays { annualization_days: f64 },
    #[error(
        "historical_volatility_rank: Invalid bar_days: {bar_days}. Must be positive and finite."
    )]
    InvalidBarDays { bar_days: f64 },
    #[error(
        "historical_volatility_rank: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("historical_volatility_rank: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("historical_volatility_rank: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "historical_volatility_rank: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("historical_volatility_rank: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityRankStream {
    hv_length: usize,
    rank_length: usize,
    annualization_scale: f64,
    prev_close: Option<f64>,
    returns: Vec<Option<f64>>,
    returns_sum: f64,
    returns_sumsq: f64,
    returns_valid: usize,
    returns_idx: usize,
    returns_count: usize,
    hv_ring: Vec<Option<f64>>,
    hv_valid: usize,
    hv_idx: usize,
    hv_count: usize,
    min_q: VecDeque<(usize, f64)>,
    max_q: VecDeque<(usize, f64)>,
    tick: usize,
}

impl HistoricalVolatilityRankStream {
    #[inline(always)]
    pub fn try_new(
        params: HistoricalVolatilityRankParams,
    ) -> Result<Self, HistoricalVolatilityRankError> {
        let hv_length = params.hv_length.unwrap_or(10);
        if hv_length == 0 {
            return Err(HistoricalVolatilityRankError::InvalidHvLength {
                hv_length,
                data_len: 0,
            });
        }
        let rank_length = params.rank_length.unwrap_or(52 * 7);
        if rank_length == 0 {
            return Err(HistoricalVolatilityRankError::InvalidRankLength { rank_length });
        }
        let annualization_days = params.annualization_days.unwrap_or(365.0);
        if !annualization_days.is_finite() || annualization_days <= 0.0 {
            return Err(HistoricalVolatilityRankError::InvalidAnnualizationDays {
                annualization_days,
            });
        }
        let bar_days = params.bar_days.unwrap_or(1.0);
        if !bar_days.is_finite() || bar_days <= 0.0 {
            return Err(HistoricalVolatilityRankError::InvalidBarDays { bar_days });
        }

        Ok(Self {
            hv_length,
            rank_length,
            annualization_scale: (annualization_days / bar_days).sqrt(),
            prev_close: None,
            returns: vec![None; hv_length],
            returns_sum: 0.0,
            returns_sumsq: 0.0,
            returns_valid: 0,
            returns_idx: 0,
            returns_count: 0,
            hv_ring: vec![None; rank_length],
            hv_valid: 0,
            hv_idx: 0,
            hv_count: 0,
            min_q: VecDeque::with_capacity(rank_length),
            max_q: VecDeque::with_capacity(rank_length),
            tick: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64) -> Option<(f64, f64)> {
        let valid_close = close.is_finite() && close > 0.0;
        let ret = match (self.prev_close, valid_close) {
            (Some(prev), true) => Some((close / prev).ln()),
            _ => None,
        };

        self.prev_close = if valid_close { Some(close) } else { None };

        if self.returns_count == self.hv_length {
            if let Some(old) = self.returns[self.returns_idx] {
                self.returns_sum -= old;
                self.returns_sumsq -= old * old;
                self.returns_valid -= 1;
            }
        } else {
            self.returns_count += 1;
        }

        self.returns[self.returns_idx] = ret;
        if let Some(value) = ret {
            self.returns_sum += value;
            self.returns_sumsq += value * value;
            self.returns_valid += 1;
        }
        self.returns_idx += 1;
        if self.returns_idx == self.hv_length {
            self.returns_idx = 0;
        }

        let hv = if self.returns_count == self.hv_length && self.returns_valid == self.hv_length {
            let n = self.hv_length as f64;
            let mean = self.returns_sum / n;
            let mut var = (self.returns_sumsq / n) - mean * mean;
            if var < 0.0 {
                var = 0.0;
            }
            Some(100.0 * var.sqrt() * self.annualization_scale)
        } else {
            None
        };

        let current_tick = self.tick;
        self.tick += 1;

        if self.hv_count == self.rank_length {
            if self.hv_ring[self.hv_idx].is_some() {
                self.hv_valid -= 1;
            }
        } else {
            self.hv_count += 1;
        }

        self.hv_ring[self.hv_idx] = hv;
        if let Some(value) = hv {
            self.hv_valid += 1;
            while let Some((_, tail)) = self.min_q.back() {
                if *tail <= value {
                    break;
                }
                self.min_q.pop_back();
            }
            self.min_q.push_back((current_tick, value));
            while let Some((_, tail)) = self.max_q.back() {
                if *tail >= value {
                    break;
                }
                self.max_q.pop_back();
            }
            self.max_q.push_back((current_tick, value));
        }
        self.hv_idx += 1;
        if self.hv_idx == self.rank_length {
            self.hv_idx = 0;
        }

        let window_start = (current_tick + 1).saturating_sub(self.rank_length);
        while let Some((idx, _)) = self.min_q.front() {
            if *idx >= window_start {
                break;
            }
            self.min_q.pop_front();
        }
        while let Some((idx, _)) = self.max_q.front() {
            if *idx >= window_start {
                break;
            }
            self.max_q.pop_front();
        }

        hv.map(|hv_value| {
            let hvr = if self.hv_count == self.rank_length && self.hv_valid == self.rank_length {
                let min_v = self.min_q.front().map(|(_, v)| *v).unwrap_or(hv_value);
                let max_v = self.max_q.front().map(|(_, v)| *v).unwrap_or(hv_value);
                let range = max_v - min_v;
                if !range.is_finite() || range <= 0.0 {
                    0.0
                } else {
                    100.0 * (hv_value - min_v) / range
                }
            } else {
                f64::NAN
            };
            (hvr, hv_value)
        })
    }

    #[inline(always)]
    pub fn get_hv_warmup_period(&self) -> usize {
        self.hv_length
    }

    #[inline(always)]
    pub fn get_hvr_warmup_period(&self) -> usize {
        self.hv_length + self.rank_length - 1
    }
}

#[derive(Clone)]
struct ReturnPrefixes {
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    invalid: Vec<u32>,
}

#[inline(always)]
fn is_valid_price(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if is_valid_price(value) {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn build_return_prefixes(close: &[f64]) -> ReturnPrefixes {
    let len = close.len();
    let mut sum = vec![0.0; len + 1];
    let mut sumsq = vec![0.0; len + 1];
    let mut invalid = vec![0u32; len + 1];

    for i in 0..len {
        let ret = if i > 0 && is_valid_price(close[i]) && is_valid_price(close[i - 1]) {
            Some((close[i] / close[i - 1]).ln())
        } else {
            None
        };

        if let Some(value) = ret {
            sum[i + 1] = sum[i] + value;
            sumsq[i + 1] = sumsq[i] + value * value;
            invalid[i + 1] = invalid[i];
        } else {
            sum[i + 1] = sum[i];
            sumsq[i + 1] = sumsq[i];
            invalid[i + 1] = invalid[i] + 1;
        }
    }

    ReturnPrefixes {
        sum,
        sumsq,
        invalid,
    }
}

#[inline(always)]
fn compute_hv_row_from_prefixes(
    prefixes: &ReturnPrefixes,
    len: usize,
    hv_length: usize,
    annualization_scale: f64,
    out_hv: &mut [f64],
) {
    if hv_length >= len {
        return;
    }

    let n = hv_length as f64;
    for i in hv_length..len {
        let start = i + 1 - hv_length;
        if prefixes.invalid[i + 1] - prefixes.invalid[start] != 0 {
            continue;
        }

        let sum = prefixes.sum[i + 1] - prefixes.sum[start];
        let sumsq = prefixes.sumsq[i + 1] - prefixes.sumsq[start];
        let mean = sum / n;
        let mut var = (sumsq / n) - mean * mean;
        if var < 0.0 {
            var = 0.0;
        }
        out_hv[i] = 100.0 * var.sqrt() * annualization_scale;
    }
}

#[inline(always)]
fn compute_hvr_row_from_hv(hv: &[f64], rank_length: usize, out_hvr: &mut [f64]) {
    let len = hv.len();
    if rank_length == 0 || rank_length > len {
        return;
    }

    let mut invalid = vec![0u32; len + 1];
    for i in 0..len {
        invalid[i + 1] = invalid[i] + u32::from(!hv[i].is_finite());
    }

    let mut min_q: VecDeque<usize> = VecDeque::with_capacity(rank_length);
    let mut max_q: VecDeque<usize> = VecDeque::with_capacity(rank_length);

    for i in 0..len {
        let value = hv[i];
        if value.is_finite() {
            while let Some(&idx) = min_q.back() {
                if hv[idx] <= value {
                    break;
                }
                min_q.pop_back();
            }
            min_q.push_back(i);

            while let Some(&idx) = max_q.back() {
                if hv[idx] >= value {
                    break;
                }
                max_q.pop_back();
            }
            max_q.push_back(i);
        }

        if i + 1 < rank_length {
            continue;
        }

        let start = i + 1 - rank_length;
        while let Some(&idx) = min_q.front() {
            if idx >= start {
                break;
            }
            min_q.pop_front();
        }
        while let Some(&idx) = max_q.front() {
            if idx >= start {
                break;
            }
            max_q.pop_front();
        }

        if invalid[i + 1] - invalid[start] != 0 {
            continue;
        }

        let min_v = hv[*min_q.front().unwrap()];
        let max_v = hv[*max_q.front().unwrap()];
        let range = max_v - min_v;
        out_hvr[i] = if !range.is_finite() || range <= 0.0 {
            0.0
        } else {
            100.0 * (value - min_v) / range
        };
    }
}

#[inline(always)]
fn validate_common(
    data: &[f64],
    hv_length: usize,
    rank_length: usize,
    annualization_days: f64,
    bar_days: f64,
) -> Result<(), HistoricalVolatilityRankError> {
    let len = data.len();
    if len == 0 {
        return Err(HistoricalVolatilityRankError::EmptyInputData);
    }
    if hv_length == 0 || hv_length >= len {
        return Err(HistoricalVolatilityRankError::InvalidHvLength {
            hv_length,
            data_len: len,
        });
    }
    if rank_length == 0 {
        return Err(HistoricalVolatilityRankError::InvalidRankLength { rank_length });
    }
    if !annualization_days.is_finite() || annualization_days <= 0.0 {
        return Err(HistoricalVolatilityRankError::InvalidAnnualizationDays { annualization_days });
    }
    if !bar_days.is_finite() || bar_days <= 0.0 {
        return Err(HistoricalVolatilityRankError::InvalidBarDays { bar_days });
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(HistoricalVolatilityRankError::AllValuesNaN);
    }
    if max_run <= hv_length {
        return Err(HistoricalVolatilityRankError::NotEnoughValidData {
            needed: hv_length + 1,
            valid: max_run,
        });
    }
    Ok(())
}

#[inline]
pub fn historical_volatility_rank(
    input: &HistoricalVolatilityRankInput,
) -> Result<HistoricalVolatilityRankOutput, HistoricalVolatilityRankError> {
    historical_volatility_rank_with_kernel(input, Kernel::Auto)
}

pub fn historical_volatility_rank_with_kernel(
    input: &HistoricalVolatilityRankInput,
    kernel: Kernel,
) -> Result<HistoricalVolatilityRankOutput, HistoricalVolatilityRankError> {
    let data: &[f64] = input.as_ref();
    let hv_length = input.get_hv_length();
    let rank_length = input.get_rank_length();
    let annualization_days = input.get_annualization_days();
    let bar_days = input.get_bar_days();
    validate_common(data, hv_length, rank_length, annualization_days, bar_days)?;

    let len = data.len();
    let mut hvr = alloc_with_nan_prefix(len, hv_length + rank_length - 1);
    let mut hv = alloc_with_nan_prefix(len, hv_length);
    historical_volatility_rank_into_slice(&mut hvr, &mut hv, input, kernel)?;
    Ok(HistoricalVolatilityRankOutput { hvr, hv })
}

pub fn historical_volatility_rank_into_slice(
    dst_hvr: &mut [f64],
    dst_hv: &mut [f64],
    input: &HistoricalVolatilityRankInput,
    kernel: Kernel,
) -> Result<(), HistoricalVolatilityRankError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if dst_hvr.len() != len {
        return Err(HistoricalVolatilityRankError::MismatchedOutputLen {
            dst_len: dst_hvr.len(),
            expected_len: len,
        });
    }
    if dst_hv.len() != len {
        return Err(HistoricalVolatilityRankError::MismatchedOutputLen {
            dst_len: dst_hv.len(),
            expected_len: len,
        });
    }

    let hv_length = input.get_hv_length();
    let rank_length = input.get_rank_length();
    let annualization_days = input.get_annualization_days();
    let bar_days = input.get_bar_days();
    validate_common(data, hv_length, rank_length, annualization_days, bar_days)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    dst_hvr.fill(f64::NAN);
    dst_hv.fill(f64::NAN);

    let prefixes = build_return_prefixes(data);
    let scale = (annualization_days / bar_days).sqrt();
    compute_hv_row_from_prefixes(&prefixes, len, hv_length, scale, dst_hv);
    compute_hvr_row_from_hv(dst_hv, rank_length, dst_hvr);
    Ok(())
}

pub fn historical_volatility_rank_output_into_slice(
    out: &mut [f64],
    input: &HistoricalVolatilityRankInput,
    kernel: Kernel,
    field: HistoricalVolatilityRankOutputField,
) -> Result<(), HistoricalVolatilityRankError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if out.len() != len {
        return Err(HistoricalVolatilityRankError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: len,
        });
    }

    let hv_length = input.get_hv_length();
    let rank_length = input.get_rank_length();
    let annualization_days = input.get_annualization_days();
    let bar_days = input.get_bar_days();
    validate_common(data, hv_length, rank_length, annualization_days, bar_days)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    out.fill(f64::NAN);
    let prefixes = build_return_prefixes(data);
    let scale = (annualization_days / bar_days).sqrt();
    match field {
        HistoricalVolatilityRankOutputField::Hv => {
            compute_hv_row_from_prefixes(&prefixes, len, hv_length, scale, out);
        }
        HistoricalVolatilityRankOutputField::Hvr => {
            let mut hv = alloc_with_nan_prefix(len, hv_length);
            compute_hv_row_from_prefixes(&prefixes, len, hv_length, scale, &mut hv);
            compute_hvr_row_from_hv(&hv, rank_length, out);
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn historical_volatility_rank_into(
    input: &HistoricalVolatilityRankInput,
    out_hvr: &mut [f64],
    out_hv: &mut [f64],
) -> Result<(), HistoricalVolatilityRankError> {
    historical_volatility_rank_into_slice(out_hvr, out_hv, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HistoricalVolatilityRankBatchRange {
    pub hv_length: (usize, usize, usize),
    pub rank_length: (usize, usize, usize),
    pub annualization_days: (f64, f64, f64),
    pub bar_days: (f64, f64, f64),
}

impl Default for HistoricalVolatilityRankBatchRange {
    fn default() -> Self {
        Self {
            hv_length: (10, 252, 1),
            rank_length: (52 * 7, 52 * 7, 0),
            annualization_days: (365.0, 365.0, 0.0),
            bar_days: (1.0, 1.0, 0.0),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HistoricalVolatilityRankBatchBuilder {
    range: HistoricalVolatilityRankBatchRange,
    kernel: Kernel,
}

impl HistoricalVolatilityRankBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn hv_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.hv_length = (start, end, step);
        self
    }

    #[inline]
    pub fn hv_length_static(mut self, value: usize) -> Self {
        self.range.hv_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn rank_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rank_length = (start, end, step);
        self
    }

    #[inline]
    pub fn rank_length_static(mut self, value: usize) -> Self {
        self.range.rank_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn annualization_days_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.annualization_days = (start, end, step);
        self
    }

    #[inline]
    pub fn annualization_days_static(mut self, value: f64) -> Self {
        self.range.annualization_days = (value, value, 0.0);
        self
    }

    #[inline]
    pub fn bar_days_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.bar_days = (start, end, step);
        self
    }

    #[inline]
    pub fn bar_days_static(mut self, value: f64) -> Self {
        self.range.bar_days = (value, value, 0.0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HistoricalVolatilityRankBatchOutput, HistoricalVolatilityRankError> {
        historical_volatility_rank_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<HistoricalVolatilityRankBatchOutput, HistoricalVolatilityRankError> {
        self.apply_slice(&candles.close)
    }
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityRankBatchOutput {
    pub hvr: Vec<f64>,
    pub hv: Vec<f64>,
    pub combos: Vec<HistoricalVolatilityRankParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HistoricalVolatilityRankBatchOutput {
    pub fn row_for_params(&self, params: &HistoricalVolatilityRankParams) -> Option<usize> {
        let hv_length = params.hv_length.unwrap_or(10);
        let rank_length = params.rank_length.unwrap_or(52 * 7);
        let annualization_days = params.annualization_days.unwrap_or(365.0);
        let bar_days = params.bar_days.unwrap_or(1.0);
        self.combos.iter().position(|combo| {
            combo.hv_length.unwrap_or(10) == hv_length
                && combo.rank_length.unwrap_or(52 * 7) == rank_length
                && (combo.annualization_days.unwrap_or(365.0) - annualization_days).abs() < 1e-12
                && (combo.bar_days.unwrap_or(1.0) - bar_days).abs() < 1e-12
        })
    }

    pub fn hvr_for(&self, params: &HistoricalVolatilityRankParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.hvr.get(start..start + self.cols)
        })
    }

    pub fn hv_for(&self, params: &HistoricalVolatilityRankParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.hv.get(start..start + self.cols)
        })
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &HistoricalVolatilityRankBatchRange,
) -> Result<Vec<HistoricalVolatilityRankParams>, HistoricalVolatilityRankError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, HistoricalVolatilityRankError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                out.push(cur);
                let next = cur.saturating_add(step.max(1));
                if next == cur {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.saturating_sub(step.max(1));
                if next == cur || next < end {
                    break;
                }
                cur = next;
            }
        }

        if out.is_empty() {
            return Err(HistoricalVolatilityRankError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, HistoricalVolatilityRankError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let step = step.abs();
            let mut cur = start;
            while cur <= end + 1e-12 {
                out.push(cur);
                cur += step;
            }
        } else {
            let step = step.abs();
            let mut cur = start;
            while cur >= end - 1e-12 {
                out.push(cur);
                cur -= step;
            }
        }

        if out.is_empty() {
            return Err(HistoricalVolatilityRankError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let hv_lengths = axis_usize(range.hv_length)?;
    if hv_lengths.iter().any(|&value| value == 0) {
        return Err(HistoricalVolatilityRankError::InvalidHvLength {
            hv_length: 0,
            data_len: 0,
        });
    }
    let rank_lengths = axis_usize(range.rank_length)?;
    if rank_lengths.iter().any(|&value| value == 0) {
        return Err(HistoricalVolatilityRankError::InvalidRankLength { rank_length: 0 });
    }
    let annualization_days = axis_f64(range.annualization_days)?;
    let bar_days = axis_f64(range.bar_days)?;

    let cap = hv_lengths
        .len()
        .checked_mul(rank_lengths.len())
        .and_then(|v| v.checked_mul(annualization_days.len()))
        .and_then(|v| v.checked_mul(bar_days.len()))
        .ok_or_else(|| HistoricalVolatilityRankError::InvalidInput {
            msg: "historical_volatility_rank: parameter grid size overflow".to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &hv_length in &hv_lengths {
        for &rank_length in &rank_lengths {
            for &annualization_day in &annualization_days {
                for &bar_day in &bar_days {
                    out.push(HistoricalVolatilityRankParams {
                        hv_length: Some(hv_length),
                        rank_length: Some(rank_length),
                        annualization_days: Some(annualization_day),
                        bar_days: Some(bar_day),
                    });
                }
            }
        }
    }
    Ok(out)
}

pub fn historical_volatility_rank_batch_with_kernel(
    data: &[f64],
    sweep: &HistoricalVolatilityRankBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityRankBatchOutput, HistoricalVolatilityRankError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HistoricalVolatilityRankError::InvalidKernelForBatch(other)),
    };
    historical_volatility_rank_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn historical_volatility_rank_batch_slice(
    data: &[f64],
    sweep: &HistoricalVolatilityRankBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityRankBatchOutput, HistoricalVolatilityRankError> {
    historical_volatility_rank_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn historical_volatility_rank_batch_par_slice(
    data: &[f64],
    sweep: &HistoricalVolatilityRankBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityRankBatchOutput, HistoricalVolatilityRankError> {
    historical_volatility_rank_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn historical_volatility_rank_batch_inner(
    data: &[f64],
    sweep: &HistoricalVolatilityRankBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<HistoricalVolatilityRankBatchOutput, HistoricalVolatilityRankError> {
    let combos = expand_grid_checked(sweep)?;
    if data.is_empty() {
        return Err(HistoricalVolatilityRankError::EmptyInputData);
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(HistoricalVolatilityRankError::AllValuesNaN);
    }

    let max_hv_length = combos
        .iter()
        .map(|params| params.hv_length.unwrap_or(10))
        .max()
        .unwrap_or(0);
    if max_hv_length >= data.len() {
        return Err(HistoricalVolatilityRankError::InvalidHvLength {
            hv_length: max_hv_length,
            data_len: data.len(),
        });
    }
    if max_run <= max_hv_length {
        return Err(HistoricalVolatilityRankError::NotEnoughValidData {
            needed: max_hv_length + 1,
            valid: max_run,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| HistoricalVolatilityRankError::InvalidInput {
                msg: "historical_volatility_rank: rows*cols overflow in batch".to_string(),
            })?;

    let mut hvr_mu = make_uninit_matrix(rows, cols);
    let mut hv_mu = make_uninit_matrix(rows, cols);
    let hvr_warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.hv_length.unwrap_or(10) + params.rank_length.unwrap_or(52 * 7) - 1)
        .collect();
    let hv_warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.hv_length.unwrap_or(10))
        .collect();

    init_matrix_prefixes(&mut hvr_mu, cols, &hvr_warmups);
    init_matrix_prefixes(&mut hv_mu, cols, &hv_warmups);

    let mut hvr = unsafe {
        Vec::from_raw_parts(
            hvr_mu.as_mut_ptr() as *mut f64,
            hvr_mu.len(),
            hvr_mu.capacity(),
        )
    };
    let mut hv = unsafe {
        Vec::from_raw_parts(
            hv_mu.as_mut_ptr() as *mut f64,
            hv_mu.len(),
            hv_mu.capacity(),
        )
    };
    std::mem::forget(hvr_mu);
    std::mem::forget(hv_mu);

    debug_assert_eq!(hvr.len(), total);
    debug_assert_eq!(hv.len(), total);

    historical_volatility_rank_batch_inner_into(data, sweep, kernel, parallel, &mut hvr, &mut hv)?;

    Ok(HistoricalVolatilityRankBatchOutput {
        hvr,
        hv,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn historical_volatility_rank_batch_inner_into(
    data: &[f64],
    sweep: &HistoricalVolatilityRankBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_hvr: &mut [f64],
    out_hv: &mut [f64],
) -> Result<Vec<HistoricalVolatilityRankParams>, HistoricalVolatilityRankError> {
    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(HistoricalVolatilityRankError::EmptyInputData);
    }

    let total = combos.len().checked_mul(len).ok_or_else(|| {
        HistoricalVolatilityRankError::InvalidInput {
            msg: "historical_volatility_rank: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    if out_hvr.len() != total {
        return Err(HistoricalVolatilityRankError::MismatchedOutputLen {
            dst_len: out_hvr.len(),
            expected_len: total,
        });
    }
    if out_hv.len() != total {
        return Err(HistoricalVolatilityRankError::MismatchedOutputLen {
            dst_len: out_hv.len(),
            expected_len: total,
        });
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(HistoricalVolatilityRankError::AllValuesNaN);
    }
    let max_hv_length = combos
        .iter()
        .map(|params| params.hv_length.unwrap_or(10))
        .max()
        .unwrap_or(0);
    if max_hv_length >= len {
        return Err(HistoricalVolatilityRankError::InvalidHvLength {
            hv_length: max_hv_length,
            data_len: len,
        });
    }
    if max_run <= max_hv_length {
        return Err(HistoricalVolatilityRankError::NotEnoughValidData {
            needed: max_hv_length + 1,
            valid: max_run,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let prefixes = build_return_prefixes(data);
    let worker = |row: usize, dst_hvr: &mut [f64], dst_hv: &mut [f64]| {
        dst_hvr.fill(f64::NAN);
        dst_hv.fill(f64::NAN);
        let params = &combos[row];
        let scale =
            (params.annualization_days.unwrap_or(365.0) / params.bar_days.unwrap_or(1.0)).sqrt();
        compute_hv_row_from_prefixes(
            &prefixes,
            len,
            params.hv_length.unwrap_or(10),
            scale,
            dst_hv,
        );
        compute_hvr_row_from_hv(dst_hv, params.rank_length.unwrap_or(52 * 7), dst_hvr);
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out_hvr
            .par_chunks_mut(len)
            .zip(out_hv.par_chunks_mut(len))
            .enumerate()
            .for_each(|(row, (dst_hvr, dst_hv))| worker(row, dst_hvr, dst_hv));
    } else {
        for (row, (dst_hvr, dst_hv)) in out_hvr
            .chunks_mut(len)
            .zip(out_hv.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_hvr, dst_hv);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, (dst_hvr, dst_hv)) in out_hvr
            .chunks_mut(len)
            .zip(out_hv.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_hvr, dst_hv);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn expand_grid_historical_volatility_rank(
    range: &HistoricalVolatilityRankBatchRange,
) -> Vec<HistoricalVolatilityRankParams> {
    expand_grid_checked(range).unwrap_or_default()
}

#[cfg(feature = "python")]
#[pyfunction(name = "historical_volatility_rank")]
#[pyo3(signature = (data, hv_length=10, rank_length=52*7, annualization_days=365.0, bar_days=1.0, kernel=None))]
pub fn historical_volatility_rank_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    hv_length: usize,
    rank_length: usize,
    annualization_days: f64,
    bar_days: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = HistoricalVolatilityRankInput::from_slice(
        slice_in,
        HistoricalVolatilityRankParams {
            hv_length: Some(hv_length),
            rank_length: Some(rank_length),
            annualization_days: Some(annualization_days),
            bar_days: Some(bar_days),
        },
    );
    let out = py
        .allow_threads(|| historical_volatility_rank_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.hvr.into_pyarray(py), out.hv.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "HistoricalVolatilityRankStream")]
pub struct HistoricalVolatilityRankStreamPy {
    stream: HistoricalVolatilityRankStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HistoricalVolatilityRankStreamPy {
    #[new]
    fn new(
        hv_length: usize,
        rank_length: usize,
        annualization_days: f64,
        bar_days: f64,
    ) -> PyResult<Self> {
        let stream = HistoricalVolatilityRankStream::try_new(HistoricalVolatilityRankParams {
            hv_length: Some(hv_length),
            rank_length: Some(rank_length),
            annualization_days: Some(annualization_days),
            bar_days: Some(bar_days),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, close: f64) -> Option<(f64, f64)> {
        self.stream.update(close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "historical_volatility_rank_batch")]
#[pyo3(signature = (data, hv_length_range=(10,10,0), rank_length_range=(52*7,52*7,0), annualization_days_range=(365.0,365.0,0.0), bar_days_range=(1.0,1.0,0.0), kernel=None))]
pub fn historical_volatility_rank_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    hv_length_range: (usize, usize, usize),
    rank_length_range: (usize, usize, usize),
    annualization_days_range: (f64, f64, f64),
    bar_days_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = HistoricalVolatilityRankBatchRange {
        hv_length: hv_length_range,
        rank_length: rank_length_range,
        annualization_days: annualization_days_range,
        bar_days: bar_days_range,
    };

    let output = py
        .allow_threads(|| historical_volatility_rank_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = output.rows;
    let cols = output.cols;
    let dict = PyDict::new(py);
    dict.set_item("hvr", output.hvr.into_pyarray(py).reshape((rows, cols))?)?;
    dict.set_item("hv", output.hv.into_pyarray(py).reshape((rows, cols))?)?;
    dict.set_item(
        "hv_lengths",
        output
            .combos
            .iter()
            .map(|params| params.hv_length.unwrap_or(10) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "rank_lengths",
        output
            .combos
            .iter()
            .map(|params| params.rank_length.unwrap_or(52 * 7) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "annualization_days",
        output
            .combos
            .iter()
            .map(|params| params.annualization_days.unwrap_or(365.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "bar_days",
        output
            .combos
            .iter()
            .map(|params| params.bar_days.unwrap_or(1.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_historical_volatility_rank_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(historical_volatility_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(historical_volatility_rank_batch_py, m)?)?;
    m.add_class::<HistoricalVolatilityRankStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = historical_volatility_rank_js)]
pub fn historical_volatility_rank_js(
    data: &[f64],
    hv_length: usize,
    rank_length: usize,
    annualization_days: f64,
    bar_days: f64,
) -> Result<JsValue, JsValue> {
    let input = HistoricalVolatilityRankInput::from_slice(
        data,
        HistoricalVolatilityRankParams {
            hv_length: Some(hv_length),
            rank_length: Some(rank_length),
            annualization_days: Some(annualization_days),
            bar_days: Some(bar_days),
        },
    );
    let out = historical_volatility_rank_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hvr"),
        &serde_wasm_bindgen::to_value(&out.hvr).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hv"),
        &serde_wasm_bindgen::to_value(&out.hv).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalVolatilityRankBatchConfig {
    pub hv_length_range: Vec<usize>,
    pub rank_length_range: Vec<usize>,
    pub annualization_days_range: Vec<f64>,
    pub bar_days_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = historical_volatility_rank_batch_js)]
pub fn historical_volatility_rank_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: HistoricalVolatilityRankBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;

    if config.hv_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: hv_length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.rank_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: rank_length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.annualization_days_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: annualization_days_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.bar_days_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: bar_days_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = HistoricalVolatilityRankBatchRange {
        hv_length: (
            config.hv_length_range[0],
            config.hv_length_range[1],
            config.hv_length_range[2],
        ),
        rank_length: (
            config.rank_length_range[0],
            config.rank_length_range[1],
            config.rank_length_range[2],
        ),
        annualization_days: (
            config.annualization_days_range[0],
            config.annualization_days_range[1],
            config.annualization_days_range[2],
        ),
        bar_days: (
            config.bar_days_range[0],
            config.bar_days_range[1],
            config.bar_days_range[2],
        ),
    };

    let out = historical_volatility_rank_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hvr"),
        &serde_wasm_bindgen::to_value(&out.hvr).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hv"),
        &serde_wasm_bindgen::to_value(&out.hv).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_rank_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(2 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_rank_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_rank_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    hv_length: usize,
    rank_length: usize,
    annualization_days: f64,
    bar_days: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to historical_volatility_rank_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (dst_hvr, dst_hv) = out.split_at_mut(len);
        let input = HistoricalVolatilityRankInput::from_slice(
            data,
            HistoricalVolatilityRankParams {
                hv_length: Some(hv_length),
                rank_length: Some(rank_length),
                annualization_days: Some(annualization_days),
                bar_days: Some(bar_days),
            },
        );
        historical_volatility_rank_into_slice(dst_hvr, dst_hv, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_rank_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    hv_length_start: usize,
    hv_length_end: usize,
    hv_length_step: usize,
    rank_length_start: usize,
    rank_length_end: usize,
    rank_length_step: usize,
    annualization_days_start: f64,
    annualization_days_end: f64,
    annualization_days_step: f64,
    bar_days_start: f64,
    bar_days_end: f64,
    bar_days_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to historical_volatility_rank_batch_into",
        ));
    }

    let sweep = HistoricalVolatilityRankBatchRange {
        hv_length: (hv_length_start, hv_length_end, hv_length_step),
        rank_length: (rank_length_start, rank_length_end, rank_length_step),
        annualization_days: (
            annualization_days_start,
            annualization_days_end,
            annualization_days_step,
        ),
        bar_days: (bar_days_start, bar_days_end, bar_days_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in historical_volatility_rank_batch_into")
        })?;

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_hvr, dst_hv) = out.split_at_mut(split);
        historical_volatility_rank_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            dst_hvr,
            dst_hv,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_rank_output_into_js(
    data: &[f64],
    hv_length: usize,
    rank_length: usize,
    annualization_days: f64,
    bar_days: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        historical_volatility_rank_js(data, hv_length, rank_length, annualization_days, bar_days)?;
    crate::write_wasm_object_f64_outputs("historical_volatility_rank_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_rank_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = historical_volatility_rank_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "historical_volatility_rank_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, ParamKV, ParamValue,
    };

    fn sample_close(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + 0.3 * x + 1.5 * (x * 0.07).sin() + 0.5 * (x * 0.03).cos()
            })
            .collect()
    }

    fn naive_hvr(
        close: &[f64],
        hv_length: usize,
        rank_length: usize,
        annualization_days: f64,
        bar_days: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let len = close.len();
        let mut hvr = vec![f64::NAN; len];
        let mut hv = vec![f64::NAN; len];
        let scale = (annualization_days / bar_days).sqrt();

        for i in hv_length..len {
            let start = i + 1 - hv_length;
            let mut returns = Vec::with_capacity(hv_length);
            for j in start..=i {
                if !is_valid_price(close[j]) || !is_valid_price(close[j - 1]) {
                    returns.clear();
                    break;
                }
                returns.push((close[j] / close[j - 1]).ln());
            }
            if returns.len() == hv_length {
                let mean = returns.iter().sum::<f64>() / hv_length as f64;
                let var = returns
                    .iter()
                    .map(|v| {
                        let d = *v - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / hv_length as f64;
                hv[i] = 100.0 * var.sqrt() * scale;
            }
        }

        for i in (hv_length + rank_length - 1)..len {
            let start = i + 1 - rank_length;
            let window = &hv[start..=i];
            if window.iter().all(|v| v.is_finite()) {
                let min_v = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_v = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let range = max_v - min_v;
                hvr[i] = if range <= 0.0 {
                    0.0
                } else {
                    100.0 * (hv[i] - min_v) / range
                };
            }
        }

        (hvr, hv)
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn historical_volatility_rank_matches_naive() -> Result<(), Box<dyn Error>> {
        let close = sample_close(256);
        let input = HistoricalVolatilityRankInput::from_slice(
            &close,
            HistoricalVolatilityRankParams {
                hv_length: Some(10),
                rank_length: Some(20),
                annualization_days: Some(365.0),
                bar_days: Some(1.0),
            },
        );
        let out = historical_volatility_rank_with_kernel(&input, Kernel::Scalar)?;
        let (expected_hvr, expected_hv) = naive_hvr(&close, 10, 20, 365.0, 1.0);

        assert_series_close(&out.hvr, &expected_hvr, 1e-8);
        assert_series_close(&out.hv, &expected_hv, 1e-10);
        Ok(())
    }

    #[test]
    fn historical_volatility_rank_into_matches_api() -> Result<(), Box<dyn Error>> {
        let close = sample_close(192);
        let input = HistoricalVolatilityRankInput::from_slice(
            &close,
            HistoricalVolatilityRankParams {
                hv_length: Some(12),
                rank_length: Some(30),
                annualization_days: Some(252.0),
                bar_days: Some(1.0),
            },
        );
        let baseline = historical_volatility_rank_with_kernel(&input, Kernel::Auto)?;
        let mut hvr = vec![0.0; close.len()];
        let mut hv = vec![0.0; close.len()];
        historical_volatility_rank_into_slice(&mut hvr, &mut hv, &input, Kernel::Auto)?;
        assert_series_close(&baseline.hvr, &hvr, 1e-10);
        assert_series_close(&baseline.hv, &hv, 1e-10);
        Ok(())
    }

    #[test]
    fn historical_volatility_rank_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = sample_close(300);
        let params = HistoricalVolatilityRankParams {
            hv_length: Some(10),
            rank_length: Some(28),
            annualization_days: Some(365.0),
            bar_days: Some(1.0),
        };
        let batch = historical_volatility_rank(&HistoricalVolatilityRankInput::from_slice(
            &close,
            params.clone(),
        ))?;

        let mut stream = HistoricalVolatilityRankStream::try_new(params)?;
        let mut hvr = Vec::with_capacity(close.len());
        let mut hv = Vec::with_capacity(close.len());
        for &value in &close {
            if let Some((hvr_value, hv_value)) = stream.update(value) {
                hvr.push(hvr_value);
                hv.push(hv_value);
            } else {
                hvr.push(f64::NAN);
                hv.push(f64::NAN);
            }
        }

        assert_series_close(&batch.hvr, &hvr, 1e-8);
        assert_series_close(&batch.hv, &hv, 1e-8);
        Ok(())
    }

    #[test]
    fn historical_volatility_rank_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let close = sample_close(220);
        let batch = historical_volatility_rank_batch_with_kernel(
            &close,
            &HistoricalVolatilityRankBatchRange {
                hv_length: (10, 10, 0),
                rank_length: (20, 20, 0),
                annualization_days: (365.0, 365.0, 0.0),
                bar_days: (1.0, 1.0, 0.0),
            },
            Kernel::ScalarBatch,
        )?;
        let single = historical_volatility_rank(&HistoricalVolatilityRankInput::from_slice(
            &close,
            HistoricalVolatilityRankParams {
                hv_length: Some(10),
                rank_length: Some(20),
                annualization_days: Some(365.0),
                bar_days: Some(1.0),
            },
        ))?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_close(&batch.hvr, &single.hvr, 1e-8);
        assert_series_close(&batch.hv, &single.hv, 1e-8);
        Ok(())
    }

    #[test]
    fn historical_volatility_rank_rejects_invalid_params() {
        let close = sample_close(32);
        let bad_hv = HistoricalVolatilityRankInput::from_slice(
            &close,
            HistoricalVolatilityRankParams {
                hv_length: Some(0),
                ..HistoricalVolatilityRankParams::default()
            },
        );
        assert!(matches!(
            historical_volatility_rank(&bad_hv),
            Err(HistoricalVolatilityRankError::InvalidHvLength { .. })
        ));

        let bad_rank = HistoricalVolatilityRankInput::from_slice(
            &close,
            HistoricalVolatilityRankParams {
                hv_length: Some(10),
                rank_length: Some(0),
                annualization_days: Some(365.0),
                bar_days: Some(1.0),
            },
        );
        assert!(matches!(
            historical_volatility_rank(&bad_rank),
            Err(HistoricalVolatilityRankError::InvalidRankLength { .. })
        ));
    }

    #[test]
    fn historical_volatility_rank_dispatch_compute_returns_hvr() -> Result<(), Box<dyn Error>> {
        let close = sample_close(180);
        let params = [
            ParamKV {
                key: "hv_length",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "rank_length",
                value: ParamValue::Int(20),
            },
        ];
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "historical_volatility_rank",
            output_id: Some("hvr"),
            data: IndicatorDataRef::Slice { values: &close },
            params: &params,
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out.output_id, "hvr");
        Ok(())
    }
}
