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

impl<'a> AsRef<[f64]> for DualUlcerIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DualUlcerIndexData::Slice(slice) => slice,
            DualUlcerIndexData::Candles { candles } => candles.close.as_slice(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DualUlcerIndexData<'a> {
    Candles { candles: &'a Candles },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DualUlcerIndexOutput {
    pub long_ulcer: Vec<f64>,
    pub short_ulcer: Vec<f64>,
    pub threshold: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualUlcerIndexOutputField {
    LongUlcer,
    ShortUlcer,
    Threshold,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DualUlcerIndexParams {
    pub period: Option<usize>,
    pub auto_threshold: Option<bool>,
    pub threshold: Option<f64>,
}

impl Default for DualUlcerIndexParams {
    fn default() -> Self {
        Self {
            period: Some(5),
            auto_threshold: Some(true),
            threshold: Some(0.1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DualUlcerIndexInput<'a> {
    pub data: DualUlcerIndexData<'a>,
    pub params: DualUlcerIndexParams,
}

impl<'a> DualUlcerIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DualUlcerIndexParams) -> Self {
        Self {
            data: DualUlcerIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: DualUlcerIndexParams) -> Self {
        Self {
            data: DualUlcerIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DualUlcerIndexParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }

    #[inline]
    pub fn get_auto_threshold(&self) -> bool {
        self.params.auto_threshold.unwrap_or(true)
    }

    #[inline]
    pub fn get_threshold(&self) -> f64 {
        self.params.threshold.unwrap_or(0.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DualUlcerIndexBuilder {
    period: Option<usize>,
    auto_threshold: Option<bool>,
    threshold: Option<f64>,
    kernel: Kernel,
}

impl Default for DualUlcerIndexBuilder {
    fn default() -> Self {
        Self {
            period: None,
            auto_threshold: None,
            threshold: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DualUlcerIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn auto_threshold(mut self, value: bool) -> Self {
        self.auto_threshold = Some(value);
        self
    }

    #[inline(always)]
    pub fn threshold(mut self, value: f64) -> Self {
        self.threshold = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<DualUlcerIndexOutput, DualUlcerIndexError> {
        let params = DualUlcerIndexParams {
            period: self.period,
            auto_threshold: self.auto_threshold,
            threshold: self.threshold,
        };
        dual_ulcer_index_with_kernel(
            &DualUlcerIndexInput::from_candles(candles, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<DualUlcerIndexOutput, DualUlcerIndexError> {
        let params = DualUlcerIndexParams {
            period: self.period,
            auto_threshold: self.auto_threshold,
            threshold: self.threshold,
        };
        dual_ulcer_index_with_kernel(&DualUlcerIndexInput::from_slice(data, params), self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DualUlcerIndexStream, DualUlcerIndexError> {
        DualUlcerIndexStream::try_new(DualUlcerIndexParams {
            period: self.period,
            auto_threshold: self.auto_threshold,
            threshold: self.threshold,
        })
    }
}

#[derive(Debug, Error)]
pub enum DualUlcerIndexError {
    #[error("dual_ulcer_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("dual_ulcer_index: All values are NaN or non-positive.")]
    AllValuesNaN,
    #[error("dual_ulcer_index: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dual_ulcer_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dual_ulcer_index: Invalid threshold: {threshold}")]
    InvalidThreshold { threshold: f64 },
    #[error("dual_ulcer_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("dual_ulcer_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("dual_ulcer_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "dual_ulcer_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("dual_ulcer_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct DualUlcerIndexStream {
    period: usize,
    auto_threshold: bool,
    custom_threshold: f64,
    close_count: usize,
    long_sq_ring: Vec<f64>,
    short_sq_ring: Vec<f64>,
    sq_idx: usize,
    sq_count: usize,
    long_sq_sum: f64,
    short_sq_sum: f64,
    min_q: VecDeque<(usize, f64)>,
    max_q: VecDeque<(usize, f64)>,
    diff_sum: f64,
    diff_count: usize,
    tick: usize,
}

impl DualUlcerIndexStream {
    #[inline(always)]
    pub fn try_new(params: DualUlcerIndexParams) -> Result<Self, DualUlcerIndexError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(DualUlcerIndexError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let threshold = params.threshold.unwrap_or(0.1);
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(DualUlcerIndexError::InvalidThreshold { threshold });
        }

        Ok(Self {
            period,
            auto_threshold: params.auto_threshold.unwrap_or(true),
            custom_threshold: threshold,
            close_count: 0,
            long_sq_ring: vec![0.0; period],
            short_sq_ring: vec![0.0; period],
            sq_idx: 0,
            sq_count: 0,
            long_sq_sum: 0.0,
            short_sq_sum: 0.0,
            min_q: VecDeque::with_capacity(period),
            max_q: VecDeque::with_capacity(period),
            diff_sum: 0.0,
            diff_count: 0,
            tick: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64) -> Option<(f64, f64, f64)> {
        let current_tick = self.tick;
        self.tick += 1;

        if !is_valid_price(close) {
            self.close_count = 0;
            self.sq_idx = 0;
            self.sq_count = 0;
            self.long_sq_sum = 0.0;
            self.short_sq_sum = 0.0;
            self.min_q.clear();
            self.max_q.clear();
            return None;
        }

        while let Some((_, value)) = self.max_q.back() {
            if *value > close {
                break;
            }
            self.max_q.pop_back();
        }
        self.max_q.push_back((current_tick, close));
        while let Some((_, value)) = self.min_q.back() {
            if *value < close {
                break;
            }
            self.min_q.pop_back();
        }
        self.min_q.push_back((current_tick, close));

        let window_start = current_tick + 1 - self.period.min(current_tick + 1);
        while let Some((idx, _)) = self.max_q.front() {
            if *idx >= window_start {
                break;
            }
            self.max_q.pop_front();
        }
        while let Some((idx, _)) = self.min_q.front() {
            if *idx >= window_start {
                break;
            }
            self.min_q.pop_front();
        }

        if self.close_count < self.period {
            self.close_count += 1;
        }
        if self.close_count < self.period {
            return None;
        }

        let highest = self.max_q.front().map(|(_, v)| *v).unwrap_or(close);
        let lowest = self.min_q.front().map(|(_, v)| *v).unwrap_or(close);
        let long_ret = 100.0 * (close - highest) / highest;
        let short_ret = 100.0 * (close - lowest) / lowest;
        let long_sq = long_ret * long_ret;
        let short_sq = short_ret * short_ret;

        if self.sq_count == self.period {
            self.long_sq_sum -= self.long_sq_ring[self.sq_idx];
            self.short_sq_sum -= self.short_sq_ring[self.sq_idx];
        } else {
            self.sq_count += 1;
        }
        self.long_sq_ring[self.sq_idx] = long_sq;
        self.short_sq_ring[self.sq_idx] = short_sq;
        self.long_sq_sum += long_sq;
        self.short_sq_sum += short_sq;
        self.sq_idx += 1;
        if self.sq_idx == self.period {
            self.sq_idx = 0;
        }

        if self.sq_count < self.period {
            return None;
        }

        let denom = self.period as f64;
        let long_ulcer = self.long_sq_sum.sqrt() / denom;
        let short_ulcer = self.short_sq_sum.sqrt() / denom;
        let diff = (long_ulcer - short_ulcer).abs();
        let threshold = if self.auto_threshold {
            self.diff_sum += diff;
            self.diff_count += 1;
            self.diff_sum / self.diff_count as f64
        } else {
            self.custom_threshold
        };
        Some((long_ulcer, short_ulcer, threshold))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period.saturating_mul(2).saturating_sub(2)
    }
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
fn validate_common(data: &[f64], period: usize, threshold: f64) -> Result<(), DualUlcerIndexError> {
    let len = data.len();
    if len == 0 {
        return Err(DualUlcerIndexError::EmptyInputData);
    }
    if period == 0 || period > len {
        return Err(DualUlcerIndexError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(DualUlcerIndexError::InvalidThreshold { threshold });
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(DualUlcerIndexError::AllValuesNaN);
    }
    let needed = period
        .checked_mul(2)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| DualUlcerIndexError::InvalidInput {
            msg: "dual_ulcer_index: period overflow".to_string(),
        })?;
    if max_run < needed {
        return Err(DualUlcerIndexError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(())
}

#[inline(always)]
fn compute_dual_ulcer_index_row(
    data: &[f64],
    period: usize,
    auto_threshold: bool,
    custom_threshold: f64,
    out_long_ulcer: &mut [f64],
    out_short_ulcer: &mut [f64],
    out_threshold: &mut [f64],
) {
    out_long_ulcer.fill(f64::NAN);
    out_short_ulcer.fill(f64::NAN);
    out_threshold.fill(f64::NAN);

    let len = data.len();
    let mut max_q: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
    let mut min_q: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
    let mut close_count = 0usize;
    let mut long_sq_ring = vec![0.0; period];
    let mut short_sq_ring = vec![0.0; period];
    let mut sq_idx = 0usize;
    let mut sq_count = 0usize;
    let mut long_sq_sum = 0.0;
    let mut short_sq_sum = 0.0;
    let mut diff_sum = 0.0;
    let mut diff_count = 0usize;

    for i in 0..len {
        let close = data[i];
        if !is_valid_price(close) {
            close_count = 0;
            sq_idx = 0;
            sq_count = 0;
            long_sq_sum = 0.0;
            short_sq_sum = 0.0;
            max_q.clear();
            min_q.clear();
            continue;
        }

        while let Some((_, value)) = max_q.back() {
            if *value > close {
                break;
            }
            max_q.pop_back();
        }
        max_q.push_back((i, close));
        while let Some((_, value)) = min_q.back() {
            if *value < close {
                break;
            }
            min_q.pop_back();
        }
        min_q.push_back((i, close));

        let window_start = i + 1 - period.min(i + 1);
        while let Some((idx, _)) = max_q.front() {
            if *idx >= window_start {
                break;
            }
            max_q.pop_front();
        }
        while let Some((idx, _)) = min_q.front() {
            if *idx >= window_start {
                break;
            }
            min_q.pop_front();
        }

        if close_count < period {
            close_count += 1;
        }
        if close_count < period {
            continue;
        }

        let highest = max_q.front().map(|(_, v)| *v).unwrap_or(close);
        let lowest = min_q.front().map(|(_, v)| *v).unwrap_or(close);
        let long_ret = 100.0 * (close - highest) / highest;
        let short_ret = 100.0 * (close - lowest) / lowest;
        let long_sq = long_ret * long_ret;
        let short_sq = short_ret * short_ret;

        if sq_count == period {
            long_sq_sum -= long_sq_ring[sq_idx];
            short_sq_sum -= short_sq_ring[sq_idx];
        } else {
            sq_count += 1;
        }
        long_sq_ring[sq_idx] = long_sq;
        short_sq_ring[sq_idx] = short_sq;
        long_sq_sum += long_sq;
        short_sq_sum += short_sq;
        sq_idx += 1;
        if sq_idx == period {
            sq_idx = 0;
        }

        if sq_count < period {
            continue;
        }

        let denom = period as f64;
        let long_ulcer = long_sq_sum.sqrt() / denom;
        let short_ulcer = short_sq_sum.sqrt() / denom;
        let diff = (long_ulcer - short_ulcer).abs();
        let threshold = if auto_threshold {
            diff_sum += diff;
            diff_count += 1;
            diff_sum / diff_count as f64
        } else {
            custom_threshold
        };

        out_long_ulcer[i] = long_ulcer;
        out_short_ulcer[i] = short_ulcer;
        out_threshold[i] = threshold;
    }
}

#[inline(always)]
fn compute_dual_ulcer_index_selected_row(
    data: &[f64],
    period: usize,
    auto_threshold: bool,
    custom_threshold: f64,
    field: DualUlcerIndexOutputField,
    out: &mut [f64],
) {
    out.fill(f64::NAN);

    let len = data.len();
    let mut max_q: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
    let mut min_q: VecDeque<(usize, f64)> = VecDeque::with_capacity(period);
    let mut close_count = 0usize;
    let mut long_sq_ring = vec![0.0; period];
    let mut short_sq_ring = vec![0.0; period];
    let mut sq_idx = 0usize;
    let mut sq_count = 0usize;
    let mut long_sq_sum = 0.0;
    let mut short_sq_sum = 0.0;
    let mut diff_sum = 0.0;
    let mut diff_count = 0usize;

    for i in 0..len {
        let close = data[i];
        if !is_valid_price(close) {
            close_count = 0;
            sq_idx = 0;
            sq_count = 0;
            long_sq_sum = 0.0;
            short_sq_sum = 0.0;
            max_q.clear();
            min_q.clear();
            continue;
        }

        while let Some((_, value)) = max_q.back() {
            if *value > close {
                break;
            }
            max_q.pop_back();
        }
        max_q.push_back((i, close));
        while let Some((_, value)) = min_q.back() {
            if *value < close {
                break;
            }
            min_q.pop_back();
        }
        min_q.push_back((i, close));

        let window_start = i + 1 - period.min(i + 1);
        while let Some((idx, _)) = max_q.front() {
            if *idx >= window_start {
                break;
            }
            max_q.pop_front();
        }
        while let Some((idx, _)) = min_q.front() {
            if *idx >= window_start {
                break;
            }
            min_q.pop_front();
        }

        if close_count < period {
            close_count += 1;
        }
        if close_count < period {
            continue;
        }

        let highest = max_q.front().map(|(_, v)| *v).unwrap_or(close);
        let lowest = min_q.front().map(|(_, v)| *v).unwrap_or(close);
        let long_ret = 100.0 * (close - highest) / highest;
        let short_ret = 100.0 * (close - lowest) / lowest;
        let long_sq = long_ret * long_ret;
        let short_sq = short_ret * short_ret;

        if sq_count == period {
            long_sq_sum -= long_sq_ring[sq_idx];
            short_sq_sum -= short_sq_ring[sq_idx];
        } else {
            sq_count += 1;
        }
        long_sq_ring[sq_idx] = long_sq;
        short_sq_ring[sq_idx] = short_sq;
        long_sq_sum += long_sq;
        short_sq_sum += short_sq;
        sq_idx += 1;
        if sq_idx == period {
            sq_idx = 0;
        }

        if sq_count < period {
            continue;
        }

        let denom = period as f64;
        let long_ulcer = long_sq_sum.sqrt() / denom;
        let short_ulcer = short_sq_sum.sqrt() / denom;
        let diff = (long_ulcer - short_ulcer).abs();
        let threshold = if auto_threshold {
            diff_sum += diff;
            diff_count += 1;
            diff_sum / diff_count as f64
        } else {
            custom_threshold
        };

        out[i] = match field {
            DualUlcerIndexOutputField::LongUlcer => long_ulcer,
            DualUlcerIndexOutputField::ShortUlcer => short_ulcer,
            DualUlcerIndexOutputField::Threshold => threshold,
        };
    }
}

#[inline]
pub fn dual_ulcer_index(
    input: &DualUlcerIndexInput,
) -> Result<DualUlcerIndexOutput, DualUlcerIndexError> {
    dual_ulcer_index_with_kernel(input, Kernel::Auto)
}

pub fn dual_ulcer_index_with_kernel(
    input: &DualUlcerIndexInput,
    kernel: Kernel,
) -> Result<DualUlcerIndexOutput, DualUlcerIndexError> {
    let data: &[f64] = input.as_ref();
    let period = input.get_period();
    let threshold = input.get_threshold();
    validate_common(data, period, threshold)?;

    let len = data.len();
    let warmup = period.saturating_mul(2).saturating_sub(2);
    let mut long_ulcer = alloc_with_nan_prefix(len, warmup.min(len));
    let mut short_ulcer = alloc_with_nan_prefix(len, warmup.min(len));
    let mut threshold_out = alloc_with_nan_prefix(len, warmup.min(len));
    dual_ulcer_index_into_slice(
        &mut long_ulcer,
        &mut short_ulcer,
        &mut threshold_out,
        input,
        kernel,
    )?;
    Ok(DualUlcerIndexOutput {
        long_ulcer,
        short_ulcer,
        threshold: threshold_out,
    })
}

pub fn dual_ulcer_index_into_slice(
    dst_long_ulcer: &mut [f64],
    dst_short_ulcer: &mut [f64],
    dst_threshold: &mut [f64],
    input: &DualUlcerIndexInput,
    kernel: Kernel,
) -> Result<(), DualUlcerIndexError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if dst_long_ulcer.len() != len {
        return Err(DualUlcerIndexError::MismatchedOutputLen {
            dst_len: dst_long_ulcer.len(),
            expected_len: len,
        });
    }
    if dst_short_ulcer.len() != len {
        return Err(DualUlcerIndexError::MismatchedOutputLen {
            dst_len: dst_short_ulcer.len(),
            expected_len: len,
        });
    }
    if dst_threshold.len() != len {
        return Err(DualUlcerIndexError::MismatchedOutputLen {
            dst_len: dst_threshold.len(),
            expected_len: len,
        });
    }

    let period = input.get_period();
    let auto_threshold = input.get_auto_threshold();
    let threshold = input.get_threshold();
    validate_common(data, period, threshold)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_dual_ulcer_index_row(
        data,
        period,
        auto_threshold,
        threshold,
        dst_long_ulcer,
        dst_short_ulcer,
        dst_threshold,
    );
    Ok(())
}

pub fn dual_ulcer_index_output_into_slice(
    out: &mut [f64],
    input: &DualUlcerIndexInput,
    kernel: Kernel,
    field: DualUlcerIndexOutputField,
) -> Result<(), DualUlcerIndexError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if out.len() != len {
        return Err(DualUlcerIndexError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: len,
        });
    }

    let period = input.get_period();
    let auto_threshold = input.get_auto_threshold();
    let threshold = input.get_threshold();
    validate_common(data, period, threshold)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_dual_ulcer_index_selected_row(data, period, auto_threshold, threshold, field, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn dual_ulcer_index_into(
    input: &DualUlcerIndexInput,
    out_long_ulcer: &mut [f64],
    out_short_ulcer: &mut [f64],
    out_threshold: &mut [f64],
) -> Result<(), DualUlcerIndexError> {
    dual_ulcer_index_into_slice(
        out_long_ulcer,
        out_short_ulcer,
        out_threshold,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DualUlcerIndexBatchRange {
    pub period: (usize, usize, usize),
    pub threshold: (f64, f64, f64),
    pub auto_threshold: bool,
}

impl Default for DualUlcerIndexBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 252, 1),
            threshold: (0.1, 0.1, 0.0),
            auto_threshold: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DualUlcerIndexBatchBuilder {
    range: DualUlcerIndexBatchRange,
    kernel: Kernel,
}

impl DualUlcerIndexBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline]
    pub fn period_static(mut self, value: usize) -> Self {
        self.range.period = (value, value, 0);
        self
    }

    #[inline]
    pub fn threshold_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.threshold = (start, end, step);
        self
    }

    #[inline]
    pub fn threshold_static(mut self, value: f64) -> Self {
        self.range.threshold = (value, value, 0.0);
        self
    }

    #[inline]
    pub fn auto_threshold(mut self, value: bool) -> Self {
        self.range.auto_threshold = value;
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<DualUlcerIndexBatchOutput, DualUlcerIndexError> {
        dual_ulcer_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<DualUlcerIndexBatchOutput, DualUlcerIndexError> {
        self.apply_slice(&candles.close)
    }
}

#[derive(Debug, Clone)]
pub struct DualUlcerIndexBatchOutput {
    pub long_ulcer: Vec<f64>,
    pub short_ulcer: Vec<f64>,
    pub threshold: Vec<f64>,
    pub combos: Vec<DualUlcerIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DualUlcerIndexBatchOutput {
    pub fn row_for_params(&self, params: &DualUlcerIndexParams) -> Option<usize> {
        let period = params.period.unwrap_or(5);
        let auto_threshold = params.auto_threshold.unwrap_or(true);
        let threshold = params.threshold.unwrap_or(0.1);
        self.combos.iter().position(|combo| {
            combo.period.unwrap_or(5) == period
                && combo.auto_threshold.unwrap_or(true) == auto_threshold
                && (combo.threshold.unwrap_or(0.1) - threshold).abs() < 1e-12
        })
    }

    pub fn long_ulcer_for(&self, params: &DualUlcerIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.long_ulcer.get(start..start + self.cols)
        })
    }

    pub fn short_ulcer_for(&self, params: &DualUlcerIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.short_ulcer.get(start..start + self.cols)
        })
    }

    pub fn threshold_for(&self, params: &DualUlcerIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.threshold.get(start..start + self.cols)
        })
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &DualUlcerIndexBatchRange,
) -> Result<Vec<DualUlcerIndexParams>, DualUlcerIndexError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, DualUlcerIndexError> {
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
            return Err(DualUlcerIndexError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, DualUlcerIndexError> {
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
            return Err(DualUlcerIndexError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let periods = axis_usize(range.period)?;
    if periods.iter().any(|&value| value == 0) {
        return Err(DualUlcerIndexError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let thresholds = axis_f64(range.threshold)?;
    if let Some(&bad_threshold) = thresholds
        .iter()
        .find(|&&value| !value.is_finite() || value < 0.0)
    {
        return Err(DualUlcerIndexError::InvalidThreshold {
            threshold: bad_threshold,
        });
    }

    let cap = periods.len().checked_mul(thresholds.len()).ok_or_else(|| {
        DualUlcerIndexError::InvalidInput {
            msg: "dual_ulcer_index: parameter grid size overflow".to_string(),
        }
    })?;

    let mut out = Vec::with_capacity(cap);
    for &period in &periods {
        for &threshold in &thresholds {
            out.push(DualUlcerIndexParams {
                period: Some(period),
                auto_threshold: Some(range.auto_threshold),
                threshold: Some(threshold),
            });
        }
    }
    Ok(out)
}

pub fn dual_ulcer_index_batch_with_kernel(
    data: &[f64],
    sweep: &DualUlcerIndexBatchRange,
    kernel: Kernel,
) -> Result<DualUlcerIndexBatchOutput, DualUlcerIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DualUlcerIndexError::InvalidKernelForBatch(other)),
    };
    dual_ulcer_index_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn dual_ulcer_index_batch_slice(
    data: &[f64],
    sweep: &DualUlcerIndexBatchRange,
    kernel: Kernel,
) -> Result<DualUlcerIndexBatchOutput, DualUlcerIndexError> {
    dual_ulcer_index_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn dual_ulcer_index_batch_par_slice(
    data: &[f64],
    sweep: &DualUlcerIndexBatchRange,
    kernel: Kernel,
) -> Result<DualUlcerIndexBatchOutput, DualUlcerIndexError> {
    dual_ulcer_index_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn dual_ulcer_index_batch_inner(
    data: &[f64],
    sweep: &DualUlcerIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DualUlcerIndexBatchOutput, DualUlcerIndexError> {
    let combos = expand_grid_checked(sweep)?;
    if data.is_empty() {
        return Err(DualUlcerIndexError::EmptyInputData);
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(DualUlcerIndexError::AllValuesNaN);
    }

    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap_or(5))
        .max()
        .unwrap_or(0);
    if max_period > data.len() {
        return Err(DualUlcerIndexError::InvalidPeriod {
            period: max_period,
            data_len: data.len(),
        });
    }
    let needed = max_period
        .checked_mul(2)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| DualUlcerIndexError::InvalidInput {
            msg: "dual_ulcer_index: period overflow in batch".to_string(),
        })?;
    if max_run < needed {
        return Err(DualUlcerIndexError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| DualUlcerIndexError::InvalidInput {
            msg: "dual_ulcer_index: rows*cols overflow in batch".to_string(),
        })?;

    let mut long_mu = make_uninit_matrix(rows, cols);
    let mut short_mu = make_uninit_matrix(rows, cols);
    let mut threshold_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            params
                .period
                .unwrap_or(5)
                .saturating_mul(2)
                .saturating_sub(2)
        })
        .collect();

    init_matrix_prefixes(&mut long_mu, cols, &warmups);
    init_matrix_prefixes(&mut short_mu, cols, &warmups);
    init_matrix_prefixes(&mut threshold_mu, cols, &warmups);

    let mut long_ulcer = unsafe {
        Vec::from_raw_parts(
            long_mu.as_mut_ptr() as *mut f64,
            long_mu.len(),
            long_mu.capacity(),
        )
    };
    let mut short_ulcer = unsafe {
        Vec::from_raw_parts(
            short_mu.as_mut_ptr() as *mut f64,
            short_mu.len(),
            short_mu.capacity(),
        )
    };
    let mut threshold = unsafe {
        Vec::from_raw_parts(
            threshold_mu.as_mut_ptr() as *mut f64,
            threshold_mu.len(),
            threshold_mu.capacity(),
        )
    };
    std::mem::forget(long_mu);
    std::mem::forget(short_mu);
    std::mem::forget(threshold_mu);

    debug_assert_eq!(long_ulcer.len(), total);
    debug_assert_eq!(short_ulcer.len(), total);
    debug_assert_eq!(threshold.len(), total);

    dual_ulcer_index_batch_inner_into(
        data,
        sweep,
        kernel,
        parallel,
        &mut long_ulcer,
        &mut short_ulcer,
        &mut threshold,
    )?;

    Ok(DualUlcerIndexBatchOutput {
        long_ulcer,
        short_ulcer,
        threshold,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn dual_ulcer_index_batch_inner_into(
    data: &[f64],
    sweep: &DualUlcerIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_long_ulcer: &mut [f64],
    out_short_ulcer: &mut [f64],
    out_threshold: &mut [f64],
) -> Result<Vec<DualUlcerIndexParams>, DualUlcerIndexError> {
    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(DualUlcerIndexError::EmptyInputData);
    }

    let total = combos
        .len()
        .checked_mul(len)
        .ok_or_else(|| DualUlcerIndexError::InvalidInput {
            msg: "dual_ulcer_index: rows*cols overflow in batch_into".to_string(),
        })?;
    if out_long_ulcer.len() != total {
        return Err(DualUlcerIndexError::MismatchedOutputLen {
            dst_len: out_long_ulcer.len(),
            expected_len: total,
        });
    }
    if out_short_ulcer.len() != total {
        return Err(DualUlcerIndexError::MismatchedOutputLen {
            dst_len: out_short_ulcer.len(),
            expected_len: total,
        });
    }
    if out_threshold.len() != total {
        return Err(DualUlcerIndexError::MismatchedOutputLen {
            dst_len: out_threshold.len(),
            expected_len: total,
        });
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(DualUlcerIndexError::AllValuesNaN);
    }
    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap_or(5))
        .max()
        .unwrap_or(0);
    if max_period > len {
        return Err(DualUlcerIndexError::InvalidPeriod {
            period: max_period,
            data_len: len,
        });
    }
    let needed = max_period
        .checked_mul(2)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| DualUlcerIndexError::InvalidInput {
            msg: "dual_ulcer_index: period overflow in batch_into".to_string(),
        })?;
    if max_run < needed {
        return Err(DualUlcerIndexError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let worker = |row: usize,
                  dst_long_ulcer: &mut [f64],
                  dst_short_ulcer: &mut [f64],
                  dst_threshold: &mut [f64]| {
        let params = &combos[row];
        compute_dual_ulcer_index_row(
            data,
            params.period.unwrap_or(5),
            params.auto_threshold.unwrap_or(true),
            params.threshold.unwrap_or(0.1),
            dst_long_ulcer,
            dst_short_ulcer,
            dst_threshold,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out_long_ulcer
            .par_chunks_mut(len)
            .zip(out_short_ulcer.par_chunks_mut(len))
            .zip(out_threshold.par_chunks_mut(len))
            .enumerate()
            .for_each(
                |(row, ((dst_long_ulcer, dst_short_ulcer), dst_threshold))| {
                    worker(row, dst_long_ulcer, dst_short_ulcer, dst_threshold)
                },
            );
    } else {
        for (row, ((dst_long_ulcer, dst_short_ulcer), dst_threshold)) in out_long_ulcer
            .chunks_mut(len)
            .zip(out_short_ulcer.chunks_mut(len))
            .zip(out_threshold.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_long_ulcer, dst_short_ulcer, dst_threshold);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, ((dst_long_ulcer, dst_short_ulcer), dst_threshold)) in out_long_ulcer
            .chunks_mut(len)
            .zip(out_short_ulcer.chunks_mut(len))
            .zip(out_threshold.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_long_ulcer, dst_short_ulcer, dst_threshold);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn expand_grid_dual_ulcer_index(range: &DualUlcerIndexBatchRange) -> Vec<DualUlcerIndexParams> {
    expand_grid_checked(range).unwrap_or_default()
}

#[cfg(feature = "python")]
#[pyfunction(name = "dual_ulcer_index")]
#[pyo3(signature = (data, period=5, auto_threshold=true, threshold=0.1, kernel=None))]
pub fn dual_ulcer_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    auto_threshold: bool,
    threshold: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = DualUlcerIndexInput::from_slice(
        slice_in,
        DualUlcerIndexParams {
            period: Some(period),
            auto_threshold: Some(auto_threshold),
            threshold: Some(threshold),
        },
    );
    let out = py
        .allow_threads(|| dual_ulcer_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.long_ulcer.into_pyarray(py),
        out.short_ulcer.into_pyarray(py),
        out.threshold.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "DualUlcerIndexStream")]
pub struct DualUlcerIndexStreamPy {
    stream: DualUlcerIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DualUlcerIndexStreamPy {
    #[new]
    #[pyo3(signature = (period=5, auto_threshold=true, threshold=0.1))]
    fn new(period: usize, auto_threshold: bool, threshold: f64) -> PyResult<Self> {
        let stream = DualUlcerIndexStream::try_new(DualUlcerIndexParams {
            period: Some(period),
            auto_threshold: Some(auto_threshold),
            threshold: Some(threshold),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, close: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dual_ulcer_index_batch")]
#[pyo3(signature = (data, period_range=(5,5,0), threshold_range=(0.1,0.1,0.0), auto_threshold=true, kernel=None))]
pub fn dual_ulcer_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    threshold_range: (f64, f64, f64),
    auto_threshold: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = DualUlcerIndexBatchRange {
        period: period_range,
        threshold: threshold_range,
        auto_threshold,
    };

    let output = py
        .allow_threads(|| dual_ulcer_index_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = output.rows;
    let cols = output.cols;
    let dict = PyDict::new(py);
    dict.set_item(
        "long_ulcer",
        output.long_ulcer.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "short_ulcer",
        output.short_ulcer.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "threshold",
        output.threshold.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|params| params.period.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "threshold_values",
        output
            .combos
            .iter()
            .map(|params| params.threshold.unwrap_or(0.1))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "auto_threshold",
        output
            .combos
            .iter()
            .map(|params| params.auto_threshold.unwrap_or(true))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_dual_ulcer_index_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dual_ulcer_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(dual_ulcer_index_batch_py, m)?)?;
    m.add_class::<DualUlcerIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dual_ulcer_index_js)]
pub fn dual_ulcer_index_js(
    data: &[f64],
    period: usize,
    auto_threshold: bool,
    threshold: f64,
) -> Result<JsValue, JsValue> {
    let input = DualUlcerIndexInput::from_slice(
        data,
        DualUlcerIndexParams {
            period: Some(period),
            auto_threshold: Some(auto_threshold),
            threshold: Some(threshold),
        },
    );
    let out = dual_ulcer_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("long_ulcer"),
        &serde_wasm_bindgen::to_value(&out.long_ulcer).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("short_ulcer"),
        &serde_wasm_bindgen::to_value(&out.short_ulcer).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("threshold"),
        &serde_wasm_bindgen::to_value(&out.threshold).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualUlcerIndexBatchConfig {
    pub period_range: Vec<usize>,
    pub threshold_range: Vec<f64>,
    pub auto_threshold: bool,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dual_ulcer_index_batch_js)]
pub fn dual_ulcer_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DualUlcerIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;

    if config.period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: period_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.threshold_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: threshold_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = DualUlcerIndexBatchRange {
        period: (
            config.period_range[0],
            config.period_range[1],
            config.period_range[2],
        ),
        threshold: (
            config.threshold_range[0],
            config.threshold_range[1],
            config.threshold_range[2],
        ),
        auto_threshold: config.auto_threshold,
    };

    let out = dual_ulcer_index_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("long_ulcer"),
        &serde_wasm_bindgen::to_value(&out.long_ulcer).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("short_ulcer"),
        &serde_wasm_bindgen::to_value(&out.short_ulcer).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("threshold"),
        &serde_wasm_bindgen::to_value(&out.threshold).unwrap(),
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
pub fn dual_ulcer_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(3 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dual_ulcer_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 3 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dual_ulcer_index_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    auto_threshold: bool,
    threshold: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to dual_ulcer_index_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 3 * len);
        let (dst_long_ulcer, rem) = out.split_at_mut(len);
        let (dst_short_ulcer, dst_threshold) = rem.split_at_mut(len);
        let input = DualUlcerIndexInput::from_slice(
            data,
            DualUlcerIndexParams {
                period: Some(period),
                auto_threshold: Some(auto_threshold),
                threshold: Some(threshold),
            },
        );
        dual_ulcer_index_into_slice(
            dst_long_ulcer,
            dst_short_ulcer,
            dst_threshold,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dual_ulcer_index_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    threshold_start: f64,
    threshold_end: f64,
    threshold_step: f64,
    auto_threshold: bool,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to dual_ulcer_index_batch_into",
        ));
    }

    let sweep = DualUlcerIndexBatchRange {
        period: (period_start, period_end, period_step),
        threshold: (threshold_start, threshold_end, threshold_step),
        auto_threshold,
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in dual_ulcer_index_batch_into"))?;

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_long_ulcer, rem) = out.split_at_mut(split);
        let (dst_short_ulcer, dst_threshold) = rem.split_at_mut(split);
        dual_ulcer_index_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            dst_long_ulcer,
            dst_short_ulcer,
            dst_threshold,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dual_ulcer_index_output_into_js(
    data: &[f64],
    period: usize,
    auto_threshold: bool,
    threshold: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = dual_ulcer_index_js(data, period, auto_threshold, threshold)?;
    crate::write_wasm_object_f64_outputs("dual_ulcer_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dual_ulcer_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = dual_ulcer_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "dual_ulcer_index_batch_output_into_js",
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

    fn naive_dui(
        close: &[f64],
        period: usize,
        auto_threshold: bool,
        custom_threshold: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let len = close.len();
        let mut long_ulcer = vec![f64::NAN; len];
        let mut short_ulcer = vec![f64::NAN; len];
        let mut threshold = vec![f64::NAN; len];
        let mut diff_sum = 0.0;
        let mut diff_count = 0usize;

        for i in 0..len {
            if i + 1 < period * 2 - 1 {
                continue;
            }
            let first = i + 1 - (period * 2 - 1);
            let segment = &close[first..=i];
            if segment.iter().any(|&v| !is_valid_price(v)) {
                continue;
            }

            let mut long_sq_sum = 0.0;
            let mut short_sq_sum = 0.0;
            for j in (i + 1 - period)..=i {
                let window = &close[j + 1 - period..=j];
                let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let long_ret = 100.0 * (close[j] - highest) / highest;
                let short_ret = 100.0 * (close[j] - lowest) / lowest;
                long_sq_sum += long_ret * long_ret;
                short_sq_sum += short_ret * short_ret;
            }

            let denom = period as f64;
            let long_value = long_sq_sum.sqrt() / denom;
            let short_value = short_sq_sum.sqrt() / denom;
            let diff = (long_value - short_value).abs();
            let threshold_value = if auto_threshold {
                diff_sum += diff;
                diff_count += 1;
                diff_sum / diff_count as f64
            } else {
                custom_threshold
            };

            long_ulcer[i] = long_value;
            short_ulcer[i] = short_value;
            threshold[i] = threshold_value;
        }

        (long_ulcer, short_ulcer, threshold)
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
    fn dual_ulcer_index_matches_naive() -> Result<(), Box<dyn Error>> {
        let close = sample_close(256);
        let input = DualUlcerIndexInput::from_slice(
            &close,
            DualUlcerIndexParams {
                period: Some(5),
                auto_threshold: Some(true),
                threshold: Some(0.1),
            },
        );
        let out = dual_ulcer_index_with_kernel(&input, Kernel::Scalar)?;
        let (expected_long, expected_short, expected_threshold) = naive_dui(&close, 5, true, 0.1);

        assert_series_close(&out.long_ulcer, &expected_long, 1e-10);
        assert_series_close(&out.short_ulcer, &expected_short, 1e-10);
        assert_series_close(&out.threshold, &expected_threshold, 1e-10);
        Ok(())
    }

    #[test]
    fn dual_ulcer_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let close = sample_close(192);
        let input = DualUlcerIndexInput::from_slice(
            &close,
            DualUlcerIndexParams {
                period: Some(7),
                auto_threshold: Some(false),
                threshold: Some(0.25),
            },
        );
        let baseline = dual_ulcer_index_with_kernel(&input, Kernel::Auto)?;
        let mut long_ulcer = vec![0.0; close.len()];
        let mut short_ulcer = vec![0.0; close.len()];
        let mut threshold = vec![0.0; close.len()];
        dual_ulcer_index_into_slice(
            &mut long_ulcer,
            &mut short_ulcer,
            &mut threshold,
            &input,
            Kernel::Auto,
        )?;
        assert_series_close(&baseline.long_ulcer, &long_ulcer, 1e-10);
        assert_series_close(&baseline.short_ulcer, &short_ulcer, 1e-10);
        assert_series_close(&baseline.threshold, &threshold, 1e-10);
        Ok(())
    }

    #[test]
    fn dual_ulcer_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = sample_close(300);
        let params = DualUlcerIndexParams {
            period: Some(6),
            auto_threshold: Some(true),
            threshold: Some(0.1),
        };
        let batch = dual_ulcer_index(&DualUlcerIndexInput::from_slice(&close, params.clone()))?;

        let mut stream = DualUlcerIndexStream::try_new(params)?;
        let mut long_ulcer = Vec::with_capacity(close.len());
        let mut short_ulcer = Vec::with_capacity(close.len());
        let mut threshold = Vec::with_capacity(close.len());
        for &value in &close {
            if let Some((long_value, short_value, threshold_value)) = stream.update(value) {
                long_ulcer.push(long_value);
                short_ulcer.push(short_value);
                threshold.push(threshold_value);
            } else {
                long_ulcer.push(f64::NAN);
                short_ulcer.push(f64::NAN);
                threshold.push(f64::NAN);
            }
        }

        assert_series_close(&batch.long_ulcer, &long_ulcer, 1e-10);
        assert_series_close(&batch.short_ulcer, &short_ulcer, 1e-10);
        assert_series_close(&batch.threshold, &threshold, 1e-10);
        Ok(())
    }

    #[test]
    fn dual_ulcer_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let close = sample_close(220);
        let batch = dual_ulcer_index_batch_with_kernel(
            &close,
            &DualUlcerIndexBatchRange {
                period: (5, 5, 0),
                threshold: (0.1, 0.1, 0.0),
                auto_threshold: true,
            },
            Kernel::ScalarBatch,
        )?;
        let single = dual_ulcer_index(&DualUlcerIndexInput::from_slice(
            &close,
            DualUlcerIndexParams {
                period: Some(5),
                auto_threshold: Some(true),
                threshold: Some(0.1),
            },
        ))?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_close(&batch.long_ulcer, &single.long_ulcer, 1e-10);
        assert_series_close(&batch.short_ulcer, &single.short_ulcer, 1e-10);
        assert_series_close(&batch.threshold, &single.threshold, 1e-10);
        Ok(())
    }

    #[test]
    fn dual_ulcer_index_rejects_invalid_params() {
        let close = sample_close(32);
        let bad_period = DualUlcerIndexInput::from_slice(
            &close,
            DualUlcerIndexParams {
                period: Some(0),
                ..DualUlcerIndexParams::default()
            },
        );
        assert!(matches!(
            dual_ulcer_index(&bad_period),
            Err(DualUlcerIndexError::InvalidPeriod { .. })
        ));

        let bad_threshold = DualUlcerIndexInput::from_slice(
            &close,
            DualUlcerIndexParams {
                period: Some(5),
                auto_threshold: Some(false),
                threshold: Some(-0.1),
            },
        );
        assert!(matches!(
            dual_ulcer_index(&bad_threshold),
            Err(DualUlcerIndexError::InvalidThreshold { .. })
        ));
    }

    #[test]
    fn dual_ulcer_index_dispatch_compute_returns_long_ulcer() -> Result<(), Box<dyn Error>> {
        let close = sample_close(180);
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "dual_ulcer_index",
            output_id: Some("long_ulcer"),
            data: IndicatorDataRef::Slice { values: &close },
            params: &params,
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out.output_id, "long_ulcer");
        Ok(())
    }
}
