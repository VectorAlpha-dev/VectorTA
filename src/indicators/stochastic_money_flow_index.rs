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
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StochasticMoneyFlowIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        source: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct StochasticMoneyFlowIndexOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StochasticMoneyFlowIndexParams {
    pub stoch_k_length: Option<usize>,
    pub stoch_k_smooth: Option<usize>,
    pub stoch_d_smooth: Option<usize>,
    pub mfi_length: Option<usize>,
}

impl Default for StochasticMoneyFlowIndexParams {
    fn default() -> Self {
        Self {
            stoch_k_length: Some(14),
            stoch_k_smooth: Some(3),
            stoch_d_smooth: Some(3),
            mfi_length: Some(14),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochasticMoneyFlowIndexInput<'a> {
    pub data: StochasticMoneyFlowIndexData<'a>,
    pub params: StochasticMoneyFlowIndexParams,
}

impl<'a> StochasticMoneyFlowIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: StochasticMoneyFlowIndexParams,
    ) -> Self {
        Self {
            data: StochasticMoneyFlowIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        volume: &'a [f64],
        params: StochasticMoneyFlowIndexParams,
    ) -> Self {
        Self {
            data: StochasticMoneyFlowIndexData::Slices { source, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", StochasticMoneyFlowIndexParams::default())
    }

    #[inline]
    pub fn get_stoch_k_length(&self) -> usize {
        self.params.stoch_k_length.unwrap_or(14)
    }

    #[inline]
    pub fn get_stoch_k_smooth(&self) -> usize {
        self.params.stoch_k_smooth.unwrap_or(3)
    }

    #[inline]
    pub fn get_stoch_d_smooth(&self) -> usize {
        self.params.stoch_d_smooth.unwrap_or(3)
    }

    #[inline]
    pub fn get_mfi_length(&self) -> usize {
        self.params.mfi_length.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StochasticMoneyFlowIndexBuilder {
    stoch_k_length: Option<usize>,
    stoch_k_smooth: Option<usize>,
    stoch_d_smooth: Option<usize>,
    mfi_length: Option<usize>,
    kernel: Kernel,
}

impl Default for StochasticMoneyFlowIndexBuilder {
    fn default() -> Self {
        Self {
            stoch_k_length: None,
            stoch_k_smooth: None,
            stoch_d_smooth: None,
            mfi_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochasticMoneyFlowIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn stoch_k_length(mut self, value: usize) -> Self {
        self.stoch_k_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn stoch_k_smooth(mut self, value: usize) -> Self {
        self.stoch_k_smooth = Some(value);
        self
    }

    #[inline(always)]
    pub fn stoch_d_smooth(mut self, value: usize) -> Self {
        self.stoch_d_smooth = Some(value);
        self
    }

    #[inline(always)]
    pub fn mfi_length(mut self, value: usize) -> Self {
        self.mfi_length = Some(value);
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
    ) -> Result<StochasticMoneyFlowIndexOutput, StochasticMoneyFlowIndexError> {
        let params = StochasticMoneyFlowIndexParams {
            stoch_k_length: self.stoch_k_length,
            stoch_k_smooth: self.stoch_k_smooth,
            stoch_d_smooth: self.stoch_d_smooth,
            mfi_length: self.mfi_length,
        };
        stochastic_money_flow_index_with_kernel(
            &StochasticMoneyFlowIndexInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<StochasticMoneyFlowIndexOutput, StochasticMoneyFlowIndexError> {
        let params = StochasticMoneyFlowIndexParams {
            stoch_k_length: self.stoch_k_length,
            stoch_k_smooth: self.stoch_k_smooth,
            stoch_d_smooth: self.stoch_d_smooth,
            mfi_length: self.mfi_length,
        };
        stochastic_money_flow_index_with_kernel(
            &StochasticMoneyFlowIndexInput::from_slices(source, volume, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<StochasticMoneyFlowIndexStream, StochasticMoneyFlowIndexError> {
        StochasticMoneyFlowIndexStream::try_new(StochasticMoneyFlowIndexParams {
            stoch_k_length: self.stoch_k_length,
            stoch_k_smooth: self.stoch_k_smooth,
            stoch_d_smooth: self.stoch_d_smooth,
            mfi_length: self.mfi_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum StochasticMoneyFlowIndexError {
    #[error("stochastic_money_flow_index: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "stochastic_money_flow_index: Input length mismatch: source = {source_len}, volume = {volume_len}"
    )]
    InputLengthMismatch {
        source_len: usize,
        volume_len: usize,
    },
    #[error("stochastic_money_flow_index: All values are NaN.")]
    AllValuesNaN,
    #[error("stochastic_money_flow_index: Invalid {name}: {value} (data length = {data_len})")]
    InvalidPeriod {
        name: &'static str,
        value: usize,
        data_len: usize,
    },
    #[error(
        "stochastic_money_flow_index: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "stochastic_money_flow_index: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("stochastic_money_flow_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("stochastic_money_flow_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "stochastic_money_flow_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("stochastic_money_flow_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[inline(always)]
fn longest_valid_run(source: &[f64], volume: &[f64]) -> usize {
    valid_run_stats(source, volume).0
}

#[inline(always)]
fn valid_run_stats(source: &[f64], volume: &[f64]) -> (usize, bool) {
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut all_valid = true;
    for (&src, &vol) in source.iter().zip(volume.iter()) {
        if src.is_finite() && vol.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            all_valid = false;
            cur = 0;
        }
    }
    (best, all_valid)
}

#[inline(always)]
fn required_bars_for_k(mfi_length: usize, stoch_k_length: usize, stoch_k_smooth: usize) -> usize {
    mfi_length
        .saturating_add(stoch_k_length)
        .saturating_add(stoch_k_smooth)
        .saturating_sub(2)
}

#[inline(always)]
fn k_warmup_prefix(mfi_length: usize, stoch_k_length: usize, stoch_k_smooth: usize) -> usize {
    required_bars_for_k(mfi_length, stoch_k_length, stoch_k_smooth).saturating_sub(1)
}

#[inline(always)]
fn d_warmup_prefix(
    mfi_length: usize,
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
) -> usize {
    required_bars_for_k(mfi_length, stoch_k_length, stoch_k_smooth)
        .saturating_add(stoch_d_smooth)
        .saturating_sub(2)
}

#[inline(always)]
fn validate_period(
    name: &'static str,
    value: usize,
    data_len: usize,
) -> Result<(), StochasticMoneyFlowIndexError> {
    if value == 0 || (data_len != usize::MAX && value > data_len) {
        return Err(StochasticMoneyFlowIndexError::InvalidPeriod {
            name,
            value,
            data_len,
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_common(
    source: &[f64],
    volume: &[f64],
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
    mfi_length: usize,
) -> Result<bool, StochasticMoneyFlowIndexError> {
    if source.is_empty() || volume.is_empty() {
        return Err(StochasticMoneyFlowIndexError::EmptyInputData);
    }
    if source.len() != volume.len() {
        return Err(StochasticMoneyFlowIndexError::InputLengthMismatch {
            source_len: source.len(),
            volume_len: volume.len(),
        });
    }
    validate_period("stoch_k_length", stoch_k_length, source.len())?;
    validate_period("stoch_k_smooth", stoch_k_smooth, source.len())?;
    validate_period("stoch_d_smooth", stoch_d_smooth, source.len())?;
    validate_period("mfi_length", mfi_length, source.len())?;

    let (max_run, all_valid) = valid_run_stats(source, volume);
    if max_run == 0 {
        return Err(StochasticMoneyFlowIndexError::AllValuesNaN);
    }
    let needed = required_bars_for_k(mfi_length, stoch_k_length, stoch_k_smooth);
    if max_run < needed {
        return Err(StochasticMoneyFlowIndexError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(all_valid)
}

#[derive(Debug, Clone)]
struct SmaState {
    period: usize,
    sum: f64,
    window: Vec<f64>,
    head: usize,
    len: usize,
}

impl SmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            sum: 0.0,
            window: vec![0.0; period.max(1)],
            head: 0,
            len: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.sum = 0.0;
        self.head = 0;
        self.len = 0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.update_finite(value)
    }

    #[inline(always)]
    fn update_finite(&mut self, value: f64) -> Option<f64> {
        if self.period == 1 {
            return Some(value);
        }
        if self.len < self.period {
            self.window[self.head] = value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            self.len += 1;
            self.sum += value;
            return (self.len == self.period).then_some(self.sum / self.period as f64);
        }
        self.sum += value;
        self.sum -= self.window[self.head];
        self.window[self.head] = value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        Some(self.sum / self.period as f64)
    }
}

#[derive(Debug, Clone)]
struct MoneyFlowState {
    period: usize,
    flow_len: usize,
    pos_buf: Vec<f64>,
    neg_buf: Vec<f64>,
    head: usize,
    count: usize,
    pos_sum: f64,
    neg_sum: f64,
    prev_source: f64,
    has_prev: bool,
}

impl MoneyFlowState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let flow_len = period.saturating_sub(1);
        Self {
            period,
            flow_len,
            pos_buf: vec![0.0; flow_len.max(1)],
            neg_buf: vec![0.0; flow_len.max(1)],
            head: 0,
            count: 0,
            pos_sum: 0.0,
            neg_sum: 0.0,
            prev_source: f64::NAN,
            has_prev: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.pos_sum = 0.0;
        self.neg_sum = 0.0;
        self.prev_source = f64::NAN;
        self.has_prev = false;
    }

    #[inline(always)]
    fn update(&mut self, source: f64, volume: f64) -> Option<f64> {
        if !self.has_prev {
            self.prev_source = source;
            self.has_prev = true;
            return if self.period == 1 { Some(0.0) } else { None };
        }

        let diff = source - self.prev_source;
        self.prev_source = source;

        if self.flow_len == 0 {
            return Some(0.0);
        }

        let flow = source * volume;
        let pos_new = if diff > 0.0 { flow } else { 0.0 };
        let neg_new = if diff < 0.0 { flow } else { 0.0 };

        if self.count == self.flow_len {
            self.pos_sum -= self.pos_buf[self.head];
            self.neg_sum -= self.neg_buf[self.head];
        } else {
            self.count += 1;
        }

        self.pos_buf[self.head] = pos_new;
        self.neg_buf[self.head] = neg_new;
        self.pos_sum += pos_new;
        self.neg_sum += neg_new;
        self.head += 1;
        if self.head == self.flow_len {
            self.head = 0;
        }

        if self.count < self.flow_len {
            return None;
        }

        let total = self.pos_sum + self.neg_sum;
        Some(if total <= 1e-14 {
            0.0
        } else {
            100.0 * self.pos_sum / total
        })
    }
}

#[derive(Debug, Clone)]
pub struct StochasticMoneyFlowIndexStream {
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
    mfi: MoneyFlowState,
    maxdq: VecDeque<(usize, f64)>,
    mindq: VecDeque<(usize, f64)>,
    mfi_index: usize,
    k_sma: SmaState,
    d_sma: SmaState,
}

impl StochasticMoneyFlowIndexStream {
    #[inline(always)]
    pub fn try_new(
        params: StochasticMoneyFlowIndexParams,
    ) -> Result<Self, StochasticMoneyFlowIndexError> {
        let stoch_k_length = params.stoch_k_length.unwrap_or(14);
        let stoch_k_smooth = params.stoch_k_smooth.unwrap_or(3);
        let stoch_d_smooth = params.stoch_d_smooth.unwrap_or(3);
        let mfi_length = params.mfi_length.unwrap_or(14);
        validate_period("stoch_k_length", stoch_k_length, usize::MAX)?;
        validate_period("stoch_k_smooth", stoch_k_smooth, usize::MAX)?;
        validate_period("stoch_d_smooth", stoch_d_smooth, usize::MAX)?;
        validate_period("mfi_length", mfi_length, usize::MAX)?;
        Ok(Self {
            stoch_k_length,
            stoch_k_smooth,
            stoch_d_smooth,
            mfi: MoneyFlowState::new(mfi_length),
            maxdq: VecDeque::with_capacity(stoch_k_length.max(1)),
            mindq: VecDeque::with_capacity(stoch_k_length.max(1)),
            mfi_index: 0,
            k_sma: SmaState::new(stoch_k_smooth),
            d_sma: SmaState::new(stoch_d_smooth),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.mfi.reset();
        self.maxdq.clear();
        self.mindq.clear();
        self.mfi_index = 0;
        self.k_sma.reset();
        self.d_sma.reset();
    }

    #[inline(always)]
    fn push_mfi(&mut self, value: f64) {
        while self
            .maxdq
            .back()
            .map(|(_, existing)| *existing <= value)
            .unwrap_or(false)
        {
            self.maxdq.pop_back();
        }
        self.maxdq.push_back((self.mfi_index, value));

        while self
            .mindq
            .back()
            .map(|(_, existing)| *existing >= value)
            .unwrap_or(false)
        {
            self.mindq.pop_back();
        }
        self.mindq.push_back((self.mfi_index, value));

        let window_start = self
            .mfi_index
            .saturating_add(1)
            .saturating_sub(self.stoch_k_length);
        while self
            .maxdq
            .front()
            .map(|(idx, _)| *idx < window_start)
            .unwrap_or(false)
        {
            self.maxdq.pop_front();
        }
        while self
            .mindq
            .front()
            .map(|(idx, _)| *idx < window_start)
            .unwrap_or(false)
        {
            self.mindq.pop_front();
        }

        self.mfi_index += 1;
    }

    #[inline(always)]
    pub fn update(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        if !source.is_finite() || !volume.is_finite() {
            self.reset();
            return None;
        }
        self.update_finite(source, volume)
    }

    #[inline(always)]
    fn update_finite(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        let mfi = self.mfi.update(source, volume)?;
        self.push_mfi(mfi);
        if self.mfi_index < self.stoch_k_length {
            return None;
        }

        let highest = self.maxdq.front().map(|(_, value)| *value).unwrap_or(mfi);
        let lowest = self.mindq.front().map(|(_, value)| *value).unwrap_or(mfi);
        let raw_k = if highest - lowest > f64::EPSILON {
            100.0 * (mfi - lowest) / (highest - lowest)
        } else {
            0.0
        };

        let k = self.k_sma.update_finite(raw_k)?;
        let d = match self.d_sma.update(k) {
            Some(value) => value,
            None => f64::NAN,
        };
        Some((k, d))
    }

    #[inline(always)]
    pub fn get_k_warmup_period(&self) -> usize {
        k_warmup_prefix(self.mfi.period, self.stoch_k_length, self.stoch_k_smooth)
    }

    #[inline(always)]
    pub fn get_d_warmup_period(&self) -> usize {
        d_warmup_prefix(
            self.mfi.period,
            self.stoch_k_length,
            self.stoch_k_smooth,
            self.stoch_d_smooth,
        )
    }
}

#[inline(always)]
fn compute_row_default_14_3_3_14<const CHECK_FINITE: bool>(
    source: &[f64],
    volume: &[f64],
    out_k: &mut [f64],
    out_d: &mut [f64],
) {
    let mut pos_buf = [0.0; 13];
    let mut neg_buf = [0.0; 13];
    let mut flow_head = 0usize;
    let mut flow_count = 0usize;
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;
    let mut prev_source = f64::NAN;
    let mut has_prev = false;

    let mut max_idx = [0usize; 15];
    let mut max_val = [0.0; 15];
    let mut max_head = 0usize;
    let mut max_len = 0usize;
    let mut min_idx = [0usize; 15];
    let mut min_val = [0.0; 15];
    let mut min_head = 0usize;
    let mut min_len = 0usize;
    let mut mfi_index = 0usize;

    let mut k_buf = [0.0; 3];
    let mut k_head = 0usize;
    let mut k_len = 0usize;
    let mut k_sum = 0.0;
    let mut d_buf = [0.0; 3];
    let mut d_head = 0usize;
    let mut d_len = 0usize;
    let mut d_sum = 0.0;

    for i in 0..source.len() {
        let src = source[i];
        let vol = volume[i];
        if CHECK_FINITE && (!src.is_finite() || !vol.is_finite()) {
            flow_head = 0;
            flow_count = 0;
            pos_sum = 0.0;
            neg_sum = 0.0;
            prev_source = f64::NAN;
            has_prev = false;
            max_head = 0;
            max_len = 0;
            min_head = 0;
            min_len = 0;
            mfi_index = 0;
            k_head = 0;
            k_len = 0;
            k_sum = 0.0;
            d_head = 0;
            d_len = 0;
            d_sum = 0.0;
            out_k[i] = f64::NAN;
            out_d[i] = f64::NAN;
            continue;
        }

        if !has_prev {
            prev_source = src;
            has_prev = true;
            out_k[i] = f64::NAN;
            out_d[i] = f64::NAN;
            continue;
        }

        let diff = src - prev_source;
        prev_source = src;
        let flow = src * vol;
        let pos_new = if diff > 0.0 { flow } else { 0.0 };
        let neg_new = if diff < 0.0 { flow } else { 0.0 };

        if flow_count == 13 {
            pos_sum -= pos_buf[flow_head];
            neg_sum -= neg_buf[flow_head];
        } else {
            flow_count += 1;
        }

        pos_buf[flow_head] = pos_new;
        neg_buf[flow_head] = neg_new;
        pos_sum += pos_new;
        neg_sum += neg_new;
        flow_head += 1;
        if flow_head == 13 {
            flow_head = 0;
        }

        if flow_count < 13 {
            out_k[i] = f64::NAN;
            out_d[i] = f64::NAN;
            continue;
        }

        let total = pos_sum + neg_sum;
        let mfi = if total <= 1e-14 {
            0.0
        } else {
            100.0 * pos_sum / total
        };

        while max_len > 0 {
            let back = (max_head + max_len - 1) % 15;
            if max_val[back] > mfi {
                break;
            }
            max_len -= 1;
        }
        let max_tail = (max_head + max_len) % 15;
        max_idx[max_tail] = mfi_index;
        max_val[max_tail] = mfi;
        max_len += 1;

        while min_len > 0 {
            let back = (min_head + min_len - 1) % 15;
            if min_val[back] < mfi {
                break;
            }
            min_len -= 1;
        }
        let min_tail = (min_head + min_len) % 15;
        min_idx[min_tail] = mfi_index;
        min_val[min_tail] = mfi;
        min_len += 1;

        let window_start = mfi_index.saturating_add(1).saturating_sub(14);
        while max_len > 0 && max_idx[max_head] < window_start {
            max_head = (max_head + 1) % 15;
            max_len -= 1;
        }
        while min_len > 0 && min_idx[min_head] < window_start {
            min_head = (min_head + 1) % 15;
            min_len -= 1;
        }
        mfi_index += 1;

        if mfi_index < 14 {
            out_k[i] = f64::NAN;
            out_d[i] = f64::NAN;
            continue;
        }

        let highest = max_val[max_head];
        let lowest = min_val[min_head];
        let raw_k = if highest - lowest > f64::EPSILON {
            100.0 * (mfi - lowest) / (highest - lowest)
        } else {
            0.0
        };

        let k = if k_len < 3 {
            k_buf[k_head] = raw_k;
            k_head += 1;
            if k_head == 3 {
                k_head = 0;
            }
            k_len += 1;
            k_sum += raw_k;
            if k_len < 3 {
                out_k[i] = f64::NAN;
                out_d[i] = f64::NAN;
                continue;
            }
            k_sum / 3.0
        } else {
            k_sum += raw_k;
            k_sum -= k_buf[k_head];
            k_buf[k_head] = raw_k;
            k_head += 1;
            if k_head == 3 {
                k_head = 0;
            }
            k_sum / 3.0
        };

        out_k[i] = k;
        let d = if d_len < 3 {
            d_buf[d_head] = k;
            d_head += 1;
            if d_head == 3 {
                d_head = 0;
            }
            d_len += 1;
            d_sum += k;
            if d_len < 3 {
                f64::NAN
            } else {
                d_sum / 3.0
            }
        } else {
            d_sum += k;
            d_sum -= d_buf[d_head];
            d_buf[d_head] = k;
            d_head += 1;
            if d_head == 3 {
                d_head = 0;
            }
            d_sum / 3.0
        };
        out_d[i] = d;
    }
}

#[inline(always)]
fn compute_row(
    source: &[f64],
    volume: &[f64],
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
    mfi_length: usize,
    all_finite: bool,
    out_k: &mut [f64],
    out_d: &mut [f64],
) {
    if stoch_k_length == 14 && stoch_k_smooth == 3 && stoch_d_smooth == 3 && mfi_length == 14 {
        if all_finite {
            compute_row_default_14_3_3_14::<false>(source, volume, out_k, out_d);
        } else {
            compute_row_default_14_3_3_14::<true>(source, volume, out_k, out_d);
        }
        return;
    }

    let mut stream = StochasticMoneyFlowIndexStream::try_new(StochasticMoneyFlowIndexParams {
        stoch_k_length: Some(stoch_k_length),
        stoch_k_smooth: Some(stoch_k_smooth),
        stoch_d_smooth: Some(stoch_d_smooth),
        mfi_length: Some(mfi_length),
    })
    .expect("validated params");

    for i in 0..source.len() {
        let value = if all_finite {
            stream.update_finite(source[i], volume[i])
        } else {
            stream.update(source[i], volume[i])
        };
        match value {
            Some((k, d)) => {
                out_k[i] = k;
                out_d[i] = d;
            }
            None => {
                out_k[i] = f64::NAN;
                out_d[i] = f64::NAN;
            }
        }
    }
}

#[inline(always)]
fn slices_from_input<'a>(input: &'a StochasticMoneyFlowIndexInput<'a>) -> (&'a [f64], &'a [f64]) {
    match &input.data {
        StochasticMoneyFlowIndexData::Candles { candles, source } => {
            let source_slice = match *source {
                "close" => candles.close.as_slice(),
                "open" => candles.open.as_slice(),
                "high" => candles.high.as_slice(),
                "low" => candles.low.as_slice(),
                "hl2" => candles.hl2.as_slice(),
                "hlc3" => candles.hlc3.as_slice(),
                "ohlc4" => candles.ohlc4.as_slice(),
                "hlcc4" => candles.hlcc4.as_slice(),
                _ => source_type(candles, source),
            };
            (source_slice, candles.volume.as_slice())
        }
        StochasticMoneyFlowIndexData::Slices { source, volume } => (*source, *volume),
    }
}

pub fn stochastic_money_flow_index(
    input: &StochasticMoneyFlowIndexInput,
) -> Result<StochasticMoneyFlowIndexOutput, StochasticMoneyFlowIndexError> {
    stochastic_money_flow_index_with_kernel(input, Kernel::Auto)
}

pub fn stochastic_money_flow_index_with_kernel(
    input: &StochasticMoneyFlowIndexInput,
    kernel: Kernel,
) -> Result<StochasticMoneyFlowIndexOutput, StochasticMoneyFlowIndexError> {
    let (source, volume) = slices_from_input(input);
    let stoch_k_length = input.get_stoch_k_length();
    let stoch_k_smooth = input.get_stoch_k_smooth();
    let stoch_d_smooth = input.get_stoch_d_smooth();
    let mfi_length = input.get_mfi_length();
    let all_finite = validate_common(
        source,
        volume,
        stoch_k_length,
        stoch_k_smooth,
        stoch_d_smooth,
        mfi_length,
    )?;

    let _chosen = match kernel {
        Kernel::Auto | Kernel::Avx2 | Kernel::Avx512 => Kernel::Scalar,
        other => other,
    };

    let mut k = alloc_uninit_f64(source.len());
    let mut d = alloc_uninit_f64(source.len());
    compute_row(
        source,
        volume,
        stoch_k_length,
        stoch_k_smooth,
        stoch_d_smooth,
        mfi_length,
        all_finite,
        &mut k,
        &mut d,
    );
    Ok(StochasticMoneyFlowIndexOutput { k, d })
}

pub fn stochastic_money_flow_index_into_slice(
    dst_k: &mut [f64],
    dst_d: &mut [f64],
    input: &StochasticMoneyFlowIndexInput,
    kernel: Kernel,
) -> Result<(), StochasticMoneyFlowIndexError> {
    let (source, volume) = slices_from_input(input);
    let stoch_k_length = input.get_stoch_k_length();
    let stoch_k_smooth = input.get_stoch_k_smooth();
    let stoch_d_smooth = input.get_stoch_d_smooth();
    let mfi_length = input.get_mfi_length();
    let all_finite = validate_common(
        source,
        volume,
        stoch_k_length,
        stoch_k_smooth,
        stoch_d_smooth,
        mfi_length,
    )?;
    if dst_k.len() != source.len() {
        return Err(StochasticMoneyFlowIndexError::OutputLengthMismatch {
            expected: source.len(),
            got: dst_k.len(),
        });
    }
    if dst_d.len() != source.len() {
        return Err(StochasticMoneyFlowIndexError::OutputLengthMismatch {
            expected: source.len(),
            got: dst_d.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto | Kernel::Avx2 | Kernel::Avx512 => Kernel::Scalar,
        other => other,
    };

    compute_row(
        source,
        volume,
        stoch_k_length,
        stoch_k_smooth,
        stoch_d_smooth,
        mfi_length,
        all_finite,
        dst_k,
        dst_d,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn stochastic_money_flow_index_into(
    input: &StochasticMoneyFlowIndexInput,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<(), StochasticMoneyFlowIndexError> {
    stochastic_money_flow_index_into_slice(out_k, out_d, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct StochasticMoneyFlowIndexBatchRange {
    pub stoch_k_length: (usize, usize, usize),
    pub stoch_k_smooth: (usize, usize, usize),
    pub stoch_d_smooth: (usize, usize, usize),
    pub mfi_length: (usize, usize, usize),
}

impl Default for StochasticMoneyFlowIndexBatchRange {
    fn default() -> Self {
        Self {
            stoch_k_length: (14, 14, 0),
            stoch_k_smooth: (3, 3, 0),
            stoch_d_smooth: (3, 3, 0),
            mfi_length: (14, 14, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochasticMoneyFlowIndexBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub combos: Vec<StochasticMoneyFlowIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl StochasticMoneyFlowIndexBatchOutput {
    pub fn row_for_params(&self, params: &StochasticMoneyFlowIndexParams) -> Option<usize> {
        let stoch_k_length = params.stoch_k_length.unwrap_or(14);
        let stoch_k_smooth = params.stoch_k_smooth.unwrap_or(3);
        let stoch_d_smooth = params.stoch_d_smooth.unwrap_or(3);
        let mfi_length = params.mfi_length.unwrap_or(14);
        self.combos.iter().position(|combo| {
            combo.stoch_k_length.unwrap_or(14) == stoch_k_length
                && combo.stoch_k_smooth.unwrap_or(3) == stoch_k_smooth
                && combo.stoch_d_smooth.unwrap_or(3) == stoch_d_smooth
                && combo.mfi_length.unwrap_or(14) == mfi_length
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StochasticMoneyFlowIndexBatchBuilder {
    range: StochasticMoneyFlowIndexBatchRange,
    kernel: Kernel,
}

impl Default for StochasticMoneyFlowIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: StochasticMoneyFlowIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl StochasticMoneyFlowIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn stoch_k_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.stoch_k_length = value;
        self
    }

    #[inline(always)]
    pub fn stoch_k_smooth_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.stoch_k_smooth = value;
        self
    }

    #[inline(always)]
    pub fn stoch_d_smooth_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.stoch_d_smooth = value;
        self
    }

    #[inline(always)]
    pub fn mfi_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.mfi_length = value;
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<StochasticMoneyFlowIndexBatchOutput, StochasticMoneyFlowIndexError> {
        stochastic_money_flow_index_batch_with_kernel(source, volume, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<StochasticMoneyFlowIndexBatchOutput, StochasticMoneyFlowIndexError> {
        stochastic_money_flow_index_batch_with_kernel(
            source_type(candles, source),
            candles.volume.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_axis(range: (usize, usize, usize)) -> Result<Vec<usize>, StochasticMoneyFlowIndexError> {
    let (start, end, step) = range;
    if start == 0 {
        return Err(StochasticMoneyFlowIndexError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(StochasticMoneyFlowIndexError::InvalidRange { start, end, step });
    }
    let mut values = Vec::new();
    let mut cur = start;
    loop {
        values.push(cur);
        if cur >= end {
            break;
        }
        let next =
            cur.checked_add(step)
                .ok_or_else(|| StochasticMoneyFlowIndexError::InvalidInput {
                    msg: "stochastic_money_flow_index: range step overflow".to_string(),
                })?;
        if next <= cur {
            return Err(StochasticMoneyFlowIndexError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
    }
    Ok(values)
}

#[inline(always)]
fn expand_grid_checked(
    range: &StochasticMoneyFlowIndexBatchRange,
) -> Result<Vec<StochasticMoneyFlowIndexParams>, StochasticMoneyFlowIndexError> {
    let stoch_k_lengths = expand_axis(range.stoch_k_length)?;
    let stoch_k_smooths = expand_axis(range.stoch_k_smooth)?;
    let stoch_d_smooths = expand_axis(range.stoch_d_smooth)?;
    let mfi_lengths = expand_axis(range.mfi_length)?;

    let mut combos = Vec::new();
    for stoch_k_length in stoch_k_lengths {
        for &stoch_k_smooth in &stoch_k_smooths {
            for &stoch_d_smooth in &stoch_d_smooths {
                for &mfi_length in &mfi_lengths {
                    combos.push(StochasticMoneyFlowIndexParams {
                        stoch_k_length: Some(stoch_k_length),
                        stoch_k_smooth: Some(stoch_k_smooth),
                        stoch_d_smooth: Some(stoch_d_smooth),
                        mfi_length: Some(mfi_length),
                    });
                }
            }
        }
    }
    Ok(combos)
}

pub fn expand_grid_stochastic_money_flow_index(
    range: &StochasticMoneyFlowIndexBatchRange,
) -> Vec<StochasticMoneyFlowIndexParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn stochastic_money_flow_index_batch_with_kernel(
    source: &[f64],
    volume: &[f64],
    sweep: &StochasticMoneyFlowIndexBatchRange,
    kernel: Kernel,
) -> Result<StochasticMoneyFlowIndexBatchOutput, StochasticMoneyFlowIndexError> {
    stochastic_money_flow_index_batch_inner(source, volume, sweep, kernel, true)
}

pub fn stochastic_money_flow_index_batch_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &StochasticMoneyFlowIndexBatchRange,
    kernel: Kernel,
) -> Result<StochasticMoneyFlowIndexBatchOutput, StochasticMoneyFlowIndexError> {
    stochastic_money_flow_index_batch_inner(source, volume, sweep, kernel, false)
}

pub fn stochastic_money_flow_index_batch_par_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &StochasticMoneyFlowIndexBatchRange,
    kernel: Kernel,
) -> Result<StochasticMoneyFlowIndexBatchOutput, StochasticMoneyFlowIndexError> {
    stochastic_money_flow_index_batch_inner(source, volume, sweep, kernel, true)
}

fn stochastic_money_flow_index_batch_inner(
    source: &[f64],
    volume: &[f64],
    sweep: &StochasticMoneyFlowIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<StochasticMoneyFlowIndexBatchOutput, StochasticMoneyFlowIndexError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| StochasticMoneyFlowIndexError::InvalidInput {
                msg: "stochastic_money_flow_index: rows*cols overflow in batch".to_string(),
            })?;

    if source.is_empty() || volume.is_empty() {
        return Err(StochasticMoneyFlowIndexError::EmptyInputData);
    }
    if source.len() != volume.len() {
        return Err(StochasticMoneyFlowIndexError::InputLengthMismatch {
            source_len: source.len(),
            volume_len: volume.len(),
        });
    }

    let (max_run, all_finite) = valid_run_stats(source, volume);
    if max_run == 0 {
        return Err(StochasticMoneyFlowIndexError::AllValuesNaN);
    }

    let mut max_needed = 0usize;
    let k_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let warmup = k_warmup_prefix(
                params.mfi_length.unwrap_or(14),
                params.stoch_k_length.unwrap_or(14),
                params.stoch_k_smooth.unwrap_or(3),
            );
            max_needed = max_needed.max(required_bars_for_k(
                params.mfi_length.unwrap_or(14),
                params.stoch_k_length.unwrap_or(14),
                params.stoch_k_smooth.unwrap_or(3),
            ));
            warmup
        })
        .collect();
    let d_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            d_warmup_prefix(
                params.mfi_length.unwrap_or(14),
                params.stoch_k_length.unwrap_or(14),
                params.stoch_k_smooth.unwrap_or(3),
                params.stoch_d_smooth.unwrap_or(3),
            )
        })
        .collect();
    if max_run < max_needed {
        return Err(StochasticMoneyFlowIndexError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let mut k_mu = make_uninit_matrix(rows, cols);
    let mut d_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut k_mu, cols, &k_warmups);
    init_matrix_prefixes(&mut d_mu, cols, &d_warmups);

    let mut k =
        unsafe { Vec::from_raw_parts(k_mu.as_mut_ptr() as *mut f64, k_mu.len(), k_mu.capacity()) };
    let mut d =
        unsafe { Vec::from_raw_parts(d_mu.as_mut_ptr() as *mut f64, d_mu.len(), d_mu.capacity()) };
    std::mem::forget(k_mu);
    std::mem::forget(d_mu);
    debug_assert_eq!(k.len(), total);
    debug_assert_eq!(d.len(), total);

    stochastic_money_flow_index_batch_inner_into(
        source, volume, sweep, kernel, parallel, &mut k, &mut d,
    )?;

    Ok(StochasticMoneyFlowIndexBatchOutput {
        k,
        d,
        combos,
        rows,
        cols,
    })
}

fn stochastic_money_flow_index_batch_inner_into(
    source: &[f64],
    volume: &[f64],
    sweep: &StochasticMoneyFlowIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<Vec<StochasticMoneyFlowIndexParams>, StochasticMoneyFlowIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(StochasticMoneyFlowIndexError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = source.len();
    if len == 0 || volume.is_empty() {
        return Err(StochasticMoneyFlowIndexError::EmptyInputData);
    }
    if len != volume.len() {
        return Err(StochasticMoneyFlowIndexError::InputLengthMismatch {
            source_len: len,
            volume_len: volume.len(),
        });
    }
    let total = combos.len().checked_mul(len).ok_or_else(|| {
        StochasticMoneyFlowIndexError::InvalidInput {
            msg: "stochastic_money_flow_index: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    if out_k.len() != total {
        return Err(StochasticMoneyFlowIndexError::MismatchedOutputLen {
            dst_len: out_k.len(),
            expected_len: total,
        });
    }
    if out_d.len() != total {
        return Err(StochasticMoneyFlowIndexError::MismatchedOutputLen {
            dst_len: out_d.len(),
            expected_len: total,
        });
    }

    let (max_run, all_finite) = valid_run_stats(source, volume);
    if max_run == 0 {
        return Err(StochasticMoneyFlowIndexError::AllValuesNaN);
    }
    let max_needed = combos
        .iter()
        .map(|params| {
            required_bars_for_k(
                params.mfi_length.unwrap_or(14),
                params.stoch_k_length.unwrap_or(14),
                params.stoch_k_smooth.unwrap_or(3),
            )
        })
        .max()
        .unwrap_or(0);
    if max_run < max_needed {
        return Err(StochasticMoneyFlowIndexError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_k: &mut [f64], dst_d: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            source,
            volume,
            params.stoch_k_length.unwrap_or(14),
            params.stoch_k_smooth.unwrap_or(3),
            params.stoch_d_smooth.unwrap_or(3),
            params.mfi_length.unwrap_or(14),
            all_finite,
            dst_k,
            dst_d,
        );
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_k
                .par_chunks_mut(len)
                .zip(out_d.par_chunks_mut(len))
                .enumerate()
                .for_each(|(row, (dst_k, dst_d))| worker(row, dst_k, dst_d));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (dst_k, dst_d)) in
                out_k.chunks_mut(len).zip(out_d.chunks_mut(len)).enumerate()
            {
                worker(row, dst_k, dst_d);
            }
        }
    } else {
        for (row, (dst_k, dst_d)) in out_k.chunks_mut(len).zip(out_d.chunks_mut(len)).enumerate() {
            worker(row, dst_k, dst_d);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochastic_money_flow_index")]
#[pyo3(signature = (source, volume, stoch_k_length=14, stoch_k_smooth=3, stoch_d_smooth=3, mfi_length=14, kernel=None))]
pub fn stochastic_money_flow_index_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
    mfi_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = StochasticMoneyFlowIndexInput::from_slices(
        source,
        volume,
        StochasticMoneyFlowIndexParams {
            stoch_k_length: Some(stoch_k_length),
            stoch_k_smooth: Some(stoch_k_smooth),
            stoch_d_smooth: Some(stoch_d_smooth),
            mfi_length: Some(mfi_length),
        },
    );
    let out = py
        .allow_threads(|| stochastic_money_flow_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.k.into_pyarray(py), out.d.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "StochasticMoneyFlowIndexStream")]
pub struct StochasticMoneyFlowIndexStreamPy {
    stream: StochasticMoneyFlowIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StochasticMoneyFlowIndexStreamPy {
    #[new]
    #[pyo3(signature = (stoch_k_length=14, stoch_k_smooth=3, stoch_d_smooth=3, mfi_length=14))]
    fn new(
        stoch_k_length: usize,
        stoch_k_smooth: usize,
        stoch_d_smooth: usize,
        mfi_length: usize,
    ) -> PyResult<Self> {
        let stream = StochasticMoneyFlowIndexStream::try_new(StochasticMoneyFlowIndexParams {
            stoch_k_length: Some(stoch_k_length),
            stoch_k_smooth: Some(stoch_k_smooth),
            stoch_d_smooth: Some(stoch_d_smooth),
            mfi_length: Some(mfi_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(source, volume)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochastic_money_flow_index_batch")]
#[pyo3(signature = (source, volume, stoch_k_length_range=(14,14,0), stoch_k_smooth_range=(3,3,0), stoch_d_smooth_range=(3,3,0), mfi_length_range=(14,14,0), kernel=None))]
pub fn stochastic_money_flow_index_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    stoch_k_length_range: (usize, usize, usize),
    stoch_k_smooth_range: (usize, usize, usize),
    stoch_d_smooth_range: (usize, usize, usize),
    mfi_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            stochastic_money_flow_index_batch_with_kernel(
                source,
                volume,
                &StochasticMoneyFlowIndexBatchRange {
                    stoch_k_length: stoch_k_length_range,
                    stoch_k_smooth: stoch_k_smooth_range,
                    stoch_d_smooth: stoch_d_smooth_range,
                    mfi_length: mfi_length_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "k",
        output
            .k
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "d",
        output
            .d
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "stoch_k_lengths",
        output
            .combos
            .iter()
            .map(|params| params.stoch_k_length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_k_smooths",
        output
            .combos
            .iter()
            .map(|params| params.stoch_k_smooth.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_d_smooths",
        output
            .combos
            .iter()
            .map(|params| params.stoch_d_smooth.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "mfi_lengths",
        output
            .combos
            .iter()
            .map(|params| params.mfi_length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_stochastic_money_flow_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stochastic_money_flow_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic_money_flow_index_batch_py, m)?)?;
    m.add_class::<StochasticMoneyFlowIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticMoneyFlowIndexBatchConfig {
    pub stoch_k_length_range: Vec<usize>,
    pub stoch_k_smooth_range: Vec<usize>,
    pub stoch_d_smooth_range: Vec<usize>,
    pub mfi_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = stochastic_money_flow_index_js)]
pub fn stochastic_money_flow_index_js(
    source: &[f64],
    volume: &[f64],
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
    mfi_length: usize,
) -> Result<JsValue, JsValue> {
    let input = StochasticMoneyFlowIndexInput::from_slices(
        source,
        volume,
        StochasticMoneyFlowIndexParams {
            stoch_k_length: Some(stoch_k_length),
            stoch_k_smooth: Some(stoch_k_smooth),
            stoch_d_smooth: Some(stoch_d_smooth),
            mfi_length: Some(mfi_length),
        },
    );
    let out = stochastic_money_flow_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("k"),
        &serde_wasm_bindgen::to_value(&out.k).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("d"),
        &serde_wasm_bindgen::to_value(&out.d).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = stochastic_money_flow_index_batch_js)]
pub fn stochastic_money_flow_index_batch_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: StochasticMoneyFlowIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.stoch_k_length_range.len() != 3
        || config.stoch_k_smooth_range.len() != 3
        || config.stoch_d_smooth_range.len() != 3
        || config.mfi_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = stochastic_money_flow_index_batch_with_kernel(
        source,
        volume,
        &StochasticMoneyFlowIndexBatchRange {
            stoch_k_length: (
                config.stoch_k_length_range[0],
                config.stoch_k_length_range[1],
                config.stoch_k_length_range[2],
            ),
            stoch_k_smooth: (
                config.stoch_k_smooth_range[0],
                config.stoch_k_smooth_range[1],
                config.stoch_k_smooth_range[2],
            ),
            stoch_d_smooth: (
                config.stoch_d_smooth_range[0],
                config.stoch_d_smooth_range[1],
                config.stoch_d_smooth_range[2],
            ),
            mfi_length: (
                config.mfi_length_range[0],
                config.mfi_length_range[1],
                config.mfi_length_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("k"),
        &serde_wasm_bindgen::to_value(&out.k).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("d"),
        &serde_wasm_bindgen::to_value(&out.d).unwrap(),
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
pub fn stochastic_money_flow_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(2 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_money_flow_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_money_flow_index_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
    mfi_length: usize,
) -> Result<(), JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to stochastic_money_flow_index_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (dst_k, dst_d) = out.split_at_mut(len);
        let input = StochasticMoneyFlowIndexInput::from_slices(
            source,
            volume,
            StochasticMoneyFlowIndexParams {
                stoch_k_length: Some(stoch_k_length),
                stoch_k_smooth: Some(stoch_k_smooth),
                stoch_d_smooth: Some(stoch_d_smooth),
                mfi_length: Some(mfi_length),
            },
        );
        stochastic_money_flow_index_into_slice(dst_k, dst_d, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_money_flow_index_batch_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    stoch_k_length_start: usize,
    stoch_k_length_end: usize,
    stoch_k_length_step: usize,
    stoch_k_smooth_start: usize,
    stoch_k_smooth_end: usize,
    stoch_k_smooth_step: usize,
    stoch_d_smooth_start: usize,
    stoch_d_smooth_end: usize,
    stoch_d_smooth_step: usize,
    mfi_length_start: usize,
    mfi_length_end: usize,
    mfi_length_step: usize,
) -> Result<usize, JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to stochastic_money_flow_index_batch_into",
        ));
    }
    let sweep = StochasticMoneyFlowIndexBatchRange {
        stoch_k_length: (
            stoch_k_length_start,
            stoch_k_length_end,
            stoch_k_length_step,
        ),
        stoch_k_smooth: (
            stoch_k_smooth_start,
            stoch_k_smooth_end,
            stoch_k_smooth_step,
        ),
        stoch_d_smooth: (
            stoch_d_smooth_start,
            stoch_d_smooth_end,
            stoch_d_smooth_step,
        ),
        mfi_length: (mfi_length_start, mfi_length_end, mfi_length_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in stochastic_money_flow_index_batch_into")
        })?;
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_k, dst_d) = out.split_at_mut(split);
        stochastic_money_flow_index_batch_inner_into(
            source,
            volume,
            &sweep,
            Kernel::Auto,
            false,
            dst_k,
            dst_d,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_money_flow_index_output_into_js(
    source: &[f64],
    volume: &[f64],
    stoch_k_length: usize,
    stoch_k_smooth: usize,
    stoch_d_smooth: usize,
    mfi_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = stochastic_money_flow_index_js(
        source,
        volume,
        stoch_k_length,
        stoch_k_smooth,
        stoch_d_smooth,
        mfi_length,
    )?;
    crate::write_wasm_object_f64_outputs("stochastic_money_flow_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_money_flow_index_batch_output_into_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = stochastic_money_flow_index_batch_js(source, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "stochastic_money_flow_index_batch_output_into_js",
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

    fn sample_source_volume(len: usize) -> (Vec<f64>, Vec<f64>) {
        let source = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.18 + (x * 0.23).sin() * 2.7 + (x * 0.071).cos() * 0.9
            })
            .collect::<Vec<_>>();
        let volume = (0..len)
            .map(|i| 1_000.0 + ((i * 37) % 113) as f64 * 17.0 + (i % 5) as f64 * 5.0)
            .collect::<Vec<_>>();
        (source, volume)
    }

    fn naive_indicator(
        source: &[f64],
        volume: &[f64],
        stoch_k_length: usize,
        stoch_k_smooth: usize,
        stoch_d_smooth: usize,
        mfi_length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let len = source.len();
        let mut mfi = vec![f64::NAN; len];
        if mfi_length == 1 {
            for value in &mut mfi {
                *value = 0.0;
            }
        } else if len >= mfi_length {
            for i in (mfi_length - 1)..len {
                let start = i + 1 - mfi_length;
                let mut pos = 0.0;
                let mut neg = 0.0;
                for j in (start + 1)..=i {
                    let flow = source[j] * volume[j];
                    if source[j] > source[j - 1] {
                        pos += flow;
                    } else if source[j] < source[j - 1] {
                        neg += flow;
                    }
                }
                let total = pos + neg;
                mfi[i] = if total <= 1e-14 {
                    0.0
                } else {
                    100.0 * pos / total
                };
            }
        }

        let mut raw_k = vec![f64::NAN; len];
        for i in 0..len {
            if i + 1 < stoch_k_length {
                continue;
            }
            let window = &mfi[i + 1 - stoch_k_length..=i];
            if window.iter().any(|value| !value.is_finite()) {
                continue;
            }
            let mut lowest = f64::INFINITY;
            let mut highest = f64::NEG_INFINITY;
            for &value in window {
                lowest = lowest.min(value);
                highest = highest.max(value);
            }
            raw_k[i] = if highest - lowest > f64::EPSILON {
                100.0 * (mfi[i] - lowest) / (highest - lowest)
            } else {
                0.0
            };
        }

        let mut k = vec![f64::NAN; len];
        for i in 0..len {
            if i + 1 < stoch_k_smooth {
                continue;
            }
            let window = &raw_k[i + 1 - stoch_k_smooth..=i];
            if window.iter().any(|value| !value.is_finite()) {
                continue;
            }
            k[i] = window.iter().sum::<f64>() / stoch_k_smooth as f64;
        }

        let mut d = vec![f64::NAN; len];
        for i in 0..len {
            if i + 1 < stoch_d_smooth {
                continue;
            }
            let window = &k[i + 1 - stoch_d_smooth..=i];
            if window.iter().any(|value| !value.is_finite()) {
                continue;
            }
            d[i] = window.iter().sum::<f64>() / stoch_d_smooth as f64;
        }

        (k, d)
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "left={a} right={b}");
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn stochastic_money_flow_index_matches_naive() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(256);
        let input = StochasticMoneyFlowIndexInput::from_slices(
            &source,
            &volume,
            StochasticMoneyFlowIndexParams::default(),
        );
        let out = stochastic_money_flow_index_with_kernel(&input, Kernel::Scalar)?;
        let (expected_k, expected_d) = naive_indicator(&source, &volume, 14, 3, 3, 14);
        assert_series_close(&out.k, &expected_k, 1e-10);
        assert_series_close(&out.d, &expected_d, 1e-10);
        Ok(())
    }

    #[test]
    fn stochastic_money_flow_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(220);
        let input = StochasticMoneyFlowIndexInput::from_slices(
            &source,
            &volume,
            StochasticMoneyFlowIndexParams {
                stoch_k_length: Some(10),
                stoch_k_smooth: Some(4),
                stoch_d_smooth: Some(3),
                mfi_length: Some(12),
            },
        );
        let base = stochastic_money_flow_index(&input)?;
        let mut k = vec![0.0; source.len()];
        let mut d = vec![0.0; source.len()];
        stochastic_money_flow_index_into_slice(&mut k, &mut d, &input, Kernel::Auto)?;
        assert_series_close(&base.k, &k, 1e-12);
        assert_series_close(&base.d, &d, 1e-12);
        Ok(())
    }

    #[test]
    fn stochastic_money_flow_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(256);
        let batch = stochastic_money_flow_index(&StochasticMoneyFlowIndexInput::from_slices(
            &source,
            &volume,
            StochasticMoneyFlowIndexParams::default(),
        ))?;

        let mut stream =
            StochasticMoneyFlowIndexStream::try_new(StochasticMoneyFlowIndexParams::default())?;
        let mut k = vec![f64::NAN; source.len()];
        let mut d = vec![f64::NAN; source.len()];
        for i in 0..source.len() {
            if let Some((k_val, d_val)) = stream.update(source[i], volume[i]) {
                k[i] = k_val;
                d[i] = d_val;
            }
        }

        assert_series_close(&batch.k, &k, 1e-12);
        assert_series_close(&batch.d, &d, 1e-12);
        Ok(())
    }

    #[test]
    fn stochastic_money_flow_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(192);
        let single = stochastic_money_flow_index(&StochasticMoneyFlowIndexInput::from_slices(
            &source,
            &volume,
            StochasticMoneyFlowIndexParams::default(),
        ))?;

        let batch = stochastic_money_flow_index_batch_with_kernel(
            &source,
            &volume,
            &StochasticMoneyFlowIndexBatchRange::default(),
            Kernel::Auto,
        )?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        assert_series_close(&batch.k[..source.len()], &single.k, 1e-12);
        assert_series_close(&batch.d[..source.len()], &single.d, 1e-12);
        Ok(())
    }

    #[test]
    fn stochastic_money_flow_index_rejects_invalid_params() {
        let (source, volume) = sample_source_volume(64);
        let input = StochasticMoneyFlowIndexInput::from_slices(
            &source,
            &volume,
            StochasticMoneyFlowIndexParams {
                stoch_k_length: Some(0),
                stoch_k_smooth: Some(3),
                stoch_d_smooth: Some(3),
                mfi_length: Some(14),
            },
        );
        assert!(matches!(
            stochastic_money_flow_index(&input),
            Err(StochasticMoneyFlowIndexError::InvalidPeriod {
                name: "stoch_k_length",
                ..
            })
        ));
    }

    #[test]
    fn stochastic_money_flow_index_dispatch_compute_returns_outputs() -> Result<(), Box<dyn Error>>
    {
        let (source, volume) = sample_source_volume(160);
        let params = [
            ParamKV {
                key: "stoch_k_length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "stoch_k_smooth",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "stoch_d_smooth",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "mfi_length",
                value: ParamValue::Int(14),
            },
        ];

        let out_k = compute_cpu(IndicatorComputeRequest {
            indicator_id: "stochastic_money_flow_index",
            data: IndicatorDataRef::CloseVolume {
                close: &source,
                volume: &volume,
            },
            params: &params,
            output_id: Some("k"),
            kernel: Kernel::Auto,
        })?;
        let out_d = compute_cpu(IndicatorComputeRequest {
            indicator_id: "stochastic_money_flow_index",
            data: IndicatorDataRef::CloseVolume {
                close: &source,
                volume: &volume,
            },
            params: &params,
            output_id: Some("d"),
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out_k.output_id, "k");
        assert_eq!(out_d.output_id, "d");
        assert_eq!(out_k.rows, 1);
        assert_eq!(out_d.rows, 1);
        assert_eq!(out_k.cols, source.len());
        assert_eq!(out_d.cols, source.len());
        Ok(())
    }
}
