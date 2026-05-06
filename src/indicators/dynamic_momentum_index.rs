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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DynamicMomentumIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DynamicMomentumIndexData::Slice(slice) => slice,
            DynamicMomentumIndexData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DynamicMomentumIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DynamicMomentumIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DynamicMomentumIndexParams {
    pub rsi_period: Option<usize>,
    pub volatility_period: Option<usize>,
    pub volatility_sma_period: Option<usize>,
    pub upper_limit: Option<usize>,
    pub lower_limit: Option<usize>,
}

impl Default for DynamicMomentumIndexParams {
    fn default() -> Self {
        Self {
            rsi_period: Some(14),
            volatility_period: Some(5),
            volatility_sma_period: Some(10),
            upper_limit: Some(30),
            lower_limit: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicMomentumIndexInput<'a> {
    pub data: DynamicMomentumIndexData<'a>,
    pub params: DynamicMomentumIndexParams,
}

impl<'a> DynamicMomentumIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: DynamicMomentumIndexParams,
    ) -> Self {
        Self {
            data: DynamicMomentumIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: DynamicMomentumIndexParams) -> Self {
        Self {
            data: DynamicMomentumIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", DynamicMomentumIndexParams::default())
    }

    #[inline]
    pub fn get_rsi_period(&self) -> usize {
        self.params.rsi_period.unwrap_or(14)
    }

    #[inline]
    pub fn get_volatility_period(&self) -> usize {
        self.params.volatility_period.unwrap_or(5)
    }

    #[inline]
    pub fn get_volatility_sma_period(&self) -> usize {
        self.params.volatility_sma_period.unwrap_or(10)
    }

    #[inline]
    pub fn get_upper_limit(&self) -> usize {
        self.params.upper_limit.unwrap_or(30)
    }

    #[inline]
    pub fn get_lower_limit(&self) -> usize {
        self.params.lower_limit.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DynamicMomentumIndexBuilder {
    rsi_period: Option<usize>,
    volatility_period: Option<usize>,
    volatility_sma_period: Option<usize>,
    upper_limit: Option<usize>,
    lower_limit: Option<usize>,
    kernel: Kernel,
}

impl Default for DynamicMomentumIndexBuilder {
    fn default() -> Self {
        Self {
            rsi_period: None,
            volatility_period: None,
            volatility_sma_period: None,
            upper_limit: None,
            lower_limit: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DynamicMomentumIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn rsi_period(mut self, value: usize) -> Self {
        self.rsi_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn volatility_period(mut self, value: usize) -> Self {
        self.volatility_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn volatility_sma_period(mut self, value: usize) -> Self {
        self.volatility_sma_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn upper_limit(mut self, value: usize) -> Self {
        self.upper_limit = Some(value);
        self
    }

    #[inline(always)]
    pub fn lower_limit(mut self, value: usize) -> Self {
        self.lower_limit = Some(value);
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
    ) -> Result<DynamicMomentumIndexOutput, DynamicMomentumIndexError> {
        let params = DynamicMomentumIndexParams {
            rsi_period: self.rsi_period,
            volatility_period: self.volatility_period,
            volatility_sma_period: self.volatility_sma_period,
            upper_limit: self.upper_limit,
            lower_limit: self.lower_limit,
        };
        dynamic_momentum_index_with_kernel(
            &DynamicMomentumIndexInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<DynamicMomentumIndexOutput, DynamicMomentumIndexError> {
        let params = DynamicMomentumIndexParams {
            rsi_period: self.rsi_period,
            volatility_period: self.volatility_period,
            volatility_sma_period: self.volatility_sma_period,
            upper_limit: self.upper_limit,
            lower_limit: self.lower_limit,
        };
        dynamic_momentum_index_with_kernel(
            &DynamicMomentumIndexInput::from_slice(data, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DynamicMomentumIndexStream, DynamicMomentumIndexError> {
        DynamicMomentumIndexStream::try_new(DynamicMomentumIndexParams {
            rsi_period: self.rsi_period,
            volatility_period: self.volatility_period,
            volatility_sma_period: self.volatility_sma_period,
            upper_limit: self.upper_limit,
            lower_limit: self.lower_limit,
        })
    }
}

#[derive(Debug, Error)]
pub enum DynamicMomentumIndexError {
    #[error("dynamic_momentum_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("dynamic_momentum_index: All values are NaN.")]
    AllValuesNaN,
    #[error("dynamic_momentum_index: Invalid RSI period: {rsi_period}")]
    InvalidRsiPeriod { rsi_period: usize },
    #[error("dynamic_momentum_index: Invalid volatility_period: {volatility_period}")]
    InvalidVolatilityPeriod { volatility_period: usize },
    #[error("dynamic_momentum_index: Invalid volatility_sma_period: {volatility_sma_period}")]
    InvalidVolatilitySmaPeriod { volatility_sma_period: usize },
    #[error("dynamic_momentum_index: Invalid upper_limit: {upper_limit}")]
    InvalidUpperLimit { upper_limit: usize },
    #[error("dynamic_momentum_index: Invalid lower_limit: {lower_limit}")]
    InvalidLowerLimit { lower_limit: usize },
    #[error(
        "dynamic_momentum_index: Invalid limits: lower_limit = {lower_limit}, upper_limit = {upper_limit}"
    )]
    InvalidLimits {
        lower_limit: usize,
        upper_limit: usize,
    },
    #[error("dynamic_momentum_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dynamic_momentum_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("dynamic_momentum_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("dynamic_momentum_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "dynamic_momentum_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("dynamic_momentum_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct DynamicMomentumIndexStream {
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
    prev_close: f64,
    has_prev: bool,
    close_ring: Vec<f64>,
    close_sum: f64,
    close_sumsq: f64,
    close_idx: usize,
    close_count: usize,
    std_ring: Vec<f64>,
    std_sum: f64,
    std_idx: usize,
    std_count: usize,
    gain_ring: Vec<f64>,
    loss_ring: Vec<f64>,
    gl_idx: usize,
    gl_count: usize,
}

impl DynamicMomentumIndexStream {
    #[inline(always)]
    pub fn try_new(params: DynamicMomentumIndexParams) -> Result<Self, DynamicMomentumIndexError> {
        let rsi_period = params.rsi_period.unwrap_or(14);
        let volatility_period = params.volatility_period.unwrap_or(5);
        let volatility_sma_period = params.volatility_sma_period.unwrap_or(10);
        let upper_limit = params.upper_limit.unwrap_or(30);
        let lower_limit = params.lower_limit.unwrap_or(5);
        validate_params_raw(
            rsi_period,
            volatility_period,
            volatility_sma_period,
            upper_limit,
            lower_limit,
        )?;

        Ok(Self {
            rsi_period,
            volatility_period,
            volatility_sma_period,
            upper_limit,
            lower_limit,
            prev_close: f64::NAN,
            has_prev: false,
            close_ring: vec![0.0; volatility_period],
            close_sum: 0.0,
            close_sumsq: 0.0,
            close_idx: 0,
            close_count: 0,
            std_ring: vec![0.0; volatility_sma_period],
            std_sum: 0.0,
            std_idx: 0,
            std_count: 0,
            gain_ring: vec![0.0; upper_limit],
            loss_ring: vec![0.0; upper_limit],
            gl_idx: 0,
            gl_count: 0,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.has_prev = false;
        self.close_sum = 0.0;
        self.close_sumsq = 0.0;
        self.close_idx = 0;
        self.close_count = 0;
        self.std_sum = 0.0;
        self.std_idx = 0;
        self.std_count = 0;
        self.gl_idx = 0;
        self.gl_count = 0;
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64) -> Option<f64> {
        if !close.is_finite() {
            self.reset();
            return None;
        }

        let std_now = push_close_window(
            close,
            &mut self.close_ring,
            &mut self.close_sum,
            &mut self.close_sumsq,
            &mut self.close_idx,
            &mut self.close_count,
        );

        if self.has_prev {
            let delta = close - self.prev_close;
            push_gain_loss(
                delta,
                &mut self.gain_ring,
                &mut self.loss_ring,
                &mut self.gl_idx,
                &mut self.gl_count,
            );
        }
        self.prev_close = close;
        self.has_prev = true;

        let Some(std_value) = std_now else {
            return None;
        };
        let avg_std = push_std_window(
            std_value,
            &mut self.std_ring,
            &mut self.std_sum,
            &mut self.std_idx,
            &mut self.std_count,
        )?;

        let period = dynamic_period(
            self.rsi_period,
            std_value,
            avg_std,
            self.lower_limit,
            self.upper_limit,
        );
        if self.gl_count < period {
            return None;
        }

        let (sum_gain, sum_loss) =
            sum_last_period(&self.gain_ring, &self.loss_ring, self.gl_idx, period);
        Some(rsi_from_sums(sum_gain, sum_loss))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        warmup_bars(
            self.volatility_period,
            self.volatility_sma_period,
            self.lower_limit,
        )
    }
}

#[inline(always)]
fn input_slice<'a>(input: &'a DynamicMomentumIndexInput<'a>) -> &'a [f64] {
    match &input.data {
        DynamicMomentumIndexData::Candles { candles, source } => source_type(candles, source),
        DynamicMomentumIndexData::Slice(slice) => slice,
    }
}

#[inline(always)]
fn validate_params_raw(
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
) -> Result<(), DynamicMomentumIndexError> {
    if rsi_period == 0 {
        return Err(DynamicMomentumIndexError::InvalidRsiPeriod { rsi_period });
    }
    if volatility_period == 0 {
        return Err(DynamicMomentumIndexError::InvalidVolatilityPeriod { volatility_period });
    }
    if volatility_sma_period == 0 {
        return Err(DynamicMomentumIndexError::InvalidVolatilitySmaPeriod {
            volatility_sma_period,
        });
    }
    if upper_limit == 0 {
        return Err(DynamicMomentumIndexError::InvalidUpperLimit { upper_limit });
    }
    if lower_limit == 0 {
        return Err(DynamicMomentumIndexError::InvalidLowerLimit { lower_limit });
    }
    if lower_limit > upper_limit {
        return Err(DynamicMomentumIndexError::InvalidLimits {
            lower_limit,
            upper_limit,
        });
    }
    Ok(())
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
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
fn warmup_bars(
    volatility_period: usize,
    volatility_sma_period: usize,
    lower_limit: usize,
) -> usize {
    (volatility_period + volatility_sma_period - 2).max(lower_limit)
}

#[inline(always)]
fn validate_common(
    data: &[f64],
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
) -> Result<(), DynamicMomentumIndexError> {
    if data.is_empty() {
        return Err(DynamicMomentumIndexError::EmptyInputData);
    }
    if !data.iter().any(|x| x.is_finite()) {
        return Err(DynamicMomentumIndexError::AllValuesNaN);
    }
    validate_params_raw(
        rsi_period,
        volatility_period,
        volatility_sma_period,
        upper_limit,
        lower_limit,
    )?;

    let needed = warmup_bars(volatility_period, volatility_sma_period, lower_limit) + 1;
    let valid = longest_valid_run(data);
    if valid < needed {
        return Err(DynamicMomentumIndexError::NotEnoughValidData { needed, valid });
    }
    Ok(())
}

#[inline(always)]
fn push_close_window(
    value: f64,
    ring: &mut [f64],
    sum: &mut f64,
    sumsq: &mut f64,
    idx: &mut usize,
    count: &mut usize,
) -> Option<f64> {
    if *count == ring.len() {
        let old = ring[*idx];
        *sum -= old;
        *sumsq -= old * old;
    } else {
        *count += 1;
    }
    ring[*idx] = value;
    *sum += value;
    *sumsq += value * value;
    *idx += 1;
    if *idx == ring.len() {
        *idx = 0;
    }
    if *count == ring.len() {
        let n = ring.len() as f64;
        let mean = *sum / n;
        let mut var = (*sumsq / n) - mean * mean;
        if var < 0.0 {
            var = 0.0;
        }
        Some(var.sqrt())
    } else {
        None
    }
}

#[inline(always)]
fn push_std_window(
    value: f64,
    ring: &mut [f64],
    sum: &mut f64,
    idx: &mut usize,
    count: &mut usize,
) -> Option<f64> {
    if *count == ring.len() {
        *sum -= ring[*idx];
    } else {
        *count += 1;
    }
    ring[*idx] = value;
    *sum += value;
    *idx += 1;
    if *idx == ring.len() {
        *idx = 0;
    }
    if *count == ring.len() {
        Some(*sum / ring.len() as f64)
    } else {
        None
    }
}

#[inline(always)]
fn push_gain_loss(
    delta: f64,
    gains: &mut [f64],
    losses: &mut [f64],
    idx: &mut usize,
    count: &mut usize,
) {
    gains[*idx] = if delta > 0.0 { delta } else { 0.0 };
    losses[*idx] = if delta < 0.0 { -delta } else { 0.0 };
    *idx += 1;
    if *idx == gains.len() {
        *idx = 0;
    }
    if *count < gains.len() {
        *count += 1;
    }
}

#[inline(always)]
fn sum_last_period(gains: &[f64], losses: &[f64], next_idx: usize, period: usize) -> (f64, f64) {
    let cap = gains.len();
    let mut sum_gain = 0.0;
    let mut sum_loss = 0.0;
    unsafe {
        if next_idx >= period {
            let end = next_idx - period;
            let mut idx = next_idx;
            while idx > end {
                idx -= 1;
                sum_gain += *gains.get_unchecked(idx);
                sum_loss += *losses.get_unchecked(idx);
            }
        } else {
            let mut idx = next_idx;
            while idx > 0 {
                idx -= 1;
                sum_gain += *gains.get_unchecked(idx);
                sum_loss += *losses.get_unchecked(idx);
            }
            let mut remaining = period - next_idx;
            idx = cap;
            while remaining > 0 {
                idx -= 1;
                sum_gain += *gains.get_unchecked(idx);
                sum_loss += *losses.get_unchecked(idx);
                remaining -= 1;
            }
        }
    }
    (sum_gain, sum_loss)
}

#[inline(always)]
fn rsi_from_sums(sum_gain: f64, sum_loss: f64) -> f64 {
    let denom = sum_gain + sum_loss;
    if denom == 0.0 {
        50.0
    } else {
        100.0 * sum_gain / denom
    }
}

#[inline(always)]
fn dynamic_period(
    rsi_period: usize,
    std_value: f64,
    avg_std: f64,
    lower_limit: usize,
    upper_limit: usize,
) -> usize {
    if !std_value.is_finite() || !avg_std.is_finite() || std_value <= 0.0 || avg_std <= 0.0 {
        return upper_limit;
    }
    let raw = ((rsi_period as f64) * avg_std / std_value).floor();
    let period = if raw.is_finite() && raw > 0.0 {
        raw as usize
    } else {
        upper_limit
    };
    period.clamp(lower_limit, upper_limit)
}

#[inline(always)]
fn compute_row(
    data: &[f64],
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
    out: &mut [f64],
) {
    let mut prev_close = f64::NAN;
    let mut has_prev = false;
    let mut close_ring = vec![0.0; volatility_period];
    let mut close_sum = 0.0;
    let mut close_sumsq = 0.0;
    let mut close_idx = 0usize;
    let mut close_count = 0usize;
    let mut std_ring = vec![0.0; volatility_sma_period];
    let mut std_sum = 0.0;
    let mut std_idx = 0usize;
    let mut std_count = 0usize;
    let mut gain_ring = vec![0.0; upper_limit];
    let mut loss_ring = vec![0.0; upper_limit];
    let mut gl_idx = 0usize;
    let mut gl_count = 0usize;

    for (i, &close) in data.iter().enumerate() {
        if !close.is_finite() {
            prev_close = f64::NAN;
            has_prev = false;
            close_sum = 0.0;
            close_sumsq = 0.0;
            close_idx = 0;
            close_count = 0;
            std_sum = 0.0;
            std_idx = 0;
            std_count = 0;
            gl_idx = 0;
            gl_count = 0;
            continue;
        }

        let std_value = push_close_window(
            close,
            &mut close_ring,
            &mut close_sum,
            &mut close_sumsq,
            &mut close_idx,
            &mut close_count,
        );

        if has_prev {
            let delta = close - prev_close;
            push_gain_loss(
                delta,
                &mut gain_ring,
                &mut loss_ring,
                &mut gl_idx,
                &mut gl_count,
            );
        }
        prev_close = close;
        has_prev = true;

        let Some(std_now) = std_value else {
            continue;
        };
        let Some(avg_std) = push_std_window(
            std_now,
            &mut std_ring,
            &mut std_sum,
            &mut std_idx,
            &mut std_count,
        ) else {
            continue;
        };

        let period = dynamic_period(rsi_period, std_now, avg_std, lower_limit, upper_limit);
        if gl_count < period {
            continue;
        }
        let (sum_gain, sum_loss) = sum_last_period(&gain_ring, &loss_ring, gl_idx, period);
        out[i] = rsi_from_sums(sum_gain, sum_loss);
    }
}

#[inline]
pub fn dynamic_momentum_index(
    input: &DynamicMomentumIndexInput,
) -> Result<DynamicMomentumIndexOutput, DynamicMomentumIndexError> {
    dynamic_momentum_index_with_kernel(input, Kernel::Auto)
}

pub fn dynamic_momentum_index_with_kernel(
    input: &DynamicMomentumIndexInput,
    _kernel: Kernel,
) -> Result<DynamicMomentumIndexOutput, DynamicMomentumIndexError> {
    let data = input_slice(input);
    let rsi_period = input.get_rsi_period();
    let volatility_period = input.get_volatility_period();
    let volatility_sma_period = input.get_volatility_sma_period();
    let upper_limit = input.get_upper_limit();
    let lower_limit = input.get_lower_limit();
    validate_common(
        data,
        rsi_period,
        volatility_period,
        volatility_sma_period,
        upper_limit,
        lower_limit,
    )?;

    let mut out = alloc_with_nan_prefix(data.len(), data.len());
    compute_row(
        data,
        rsi_period,
        volatility_period,
        volatility_sma_period,
        upper_limit,
        lower_limit,
        &mut out,
    );
    Ok(DynamicMomentumIndexOutput { values: out })
}

#[inline]
pub fn dynamic_momentum_index_into_slice(
    dst: &mut [f64],
    input: &DynamicMomentumIndexInput,
    _kernel: Kernel,
) -> Result<(), DynamicMomentumIndexError> {
    let data = input_slice(input);
    let rsi_period = input.get_rsi_period();
    let volatility_period = input.get_volatility_period();
    let volatility_sma_period = input.get_volatility_sma_period();
    let upper_limit = input.get_upper_limit();
    let lower_limit = input.get_lower_limit();
    validate_common(
        data,
        rsi_period,
        volatility_period,
        volatility_sma_period,
        upper_limit,
        lower_limit,
    )?;
    if dst.len() != data.len() {
        return Err(DynamicMomentumIndexError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);
    compute_row(
        data,
        rsi_period,
        volatility_period,
        volatility_sma_period,
        upper_limit,
        lower_limit,
        dst,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn dynamic_momentum_index_into(
    input: &DynamicMomentumIndexInput,
    out: &mut [f64],
) -> Result<(), DynamicMomentumIndexError> {
    dynamic_momentum_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicMomentumIndexBatchRange {
    pub rsi_period: (usize, usize, usize),
    pub volatility_period: (usize, usize, usize),
    pub volatility_sma_period: (usize, usize, usize),
    pub upper_limit: (usize, usize, usize),
    pub lower_limit: (usize, usize, usize),
}

impl Default for DynamicMomentumIndexBatchRange {
    fn default() -> Self {
        Self {
            rsi_period: (14, 14, 0),
            volatility_period: (5, 5, 0),
            volatility_sma_period: (10, 10, 0),
            upper_limit: (30, 30, 0),
            lower_limit: (5, 5, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicMomentumIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DynamicMomentumIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicMomentumIndexBatchBuilder {
    range: DynamicMomentumIndexBatchRange,
    kernel: Kernel,
}

impl Default for DynamicMomentumIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: DynamicMomentumIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl DynamicMomentumIndexBatchBuilder {
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
    pub fn rsi_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn volatility_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.volatility_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn volatility_sma_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.volatility_sma_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn upper_limit_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.upper_limit = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn lower_limit_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lower_limit = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<DynamicMomentumIndexBatchOutput, DynamicMomentumIndexError> {
        dynamic_momentum_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<DynamicMomentumIndexBatchOutput, DynamicMomentumIndexError> {
        dynamic_momentum_index_batch_with_kernel(candles.close.as_slice(), &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_one(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, DynamicMomentumIndexError> {
    if start == 0 || end == 0 {
        return Err(DynamicMomentumIndexError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(DynamicMomentumIndexError::InvalidRange { start, end, step });
    }

    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(cur);
        if cur >= end {
            break;
        }
        let next = cur.saturating_add(step);
        if next <= cur {
            return Err(DynamicMomentumIndexError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
        if cur == *out.last().unwrap() {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_checked(
    range: &DynamicMomentumIndexBatchRange,
) -> Result<Vec<DynamicMomentumIndexParams>, DynamicMomentumIndexError> {
    let rsi_periods = expand_one(range.rsi_period.0, range.rsi_period.1, range.rsi_period.2)?;
    let volatility_periods = expand_one(
        range.volatility_period.0,
        range.volatility_period.1,
        range.volatility_period.2,
    )?;
    let volatility_sma_periods = expand_one(
        range.volatility_sma_period.0,
        range.volatility_sma_period.1,
        range.volatility_sma_period.2,
    )?;
    let upper_limits = expand_one(
        range.upper_limit.0,
        range.upper_limit.1,
        range.upper_limit.2,
    )?;
    let lower_limits = expand_one(
        range.lower_limit.0,
        range.lower_limit.1,
        range.lower_limit.2,
    )?;

    let mut total = rsi_periods.len();
    total = total
        .checked_mul(volatility_periods.len())
        .and_then(|x| x.checked_mul(volatility_sma_periods.len()))
        .and_then(|x| x.checked_mul(upper_limits.len()))
        .and_then(|x| x.checked_mul(lower_limits.len()))
        .ok_or_else(|| DynamicMomentumIndexError::InvalidInput {
            msg: "dynamic_momentum_index: parameter grid size overflow".to_string(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &rsi_period in &rsi_periods {
        for &volatility_period in &volatility_periods {
            for &volatility_sma_period in &volatility_sma_periods {
                for &upper_limit in &upper_limits {
                    for &lower_limit in &lower_limits {
                        out.push(DynamicMomentumIndexParams {
                            rsi_period: Some(rsi_period),
                            volatility_period: Some(volatility_period),
                            volatility_sma_period: Some(volatility_sma_period),
                            upper_limit: Some(upper_limit),
                            lower_limit: Some(lower_limit),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_dynamic_momentum_index(
    range: &DynamicMomentumIndexBatchRange,
) -> Vec<DynamicMomentumIndexParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn dynamic_momentum_index_batch_with_kernel(
    data: &[f64],
    sweep: &DynamicMomentumIndexBatchRange,
    kernel: Kernel,
) -> Result<DynamicMomentumIndexBatchOutput, DynamicMomentumIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(DynamicMomentumIndexError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    for combo in &combos {
        validate_common(
            data,
            combo.rsi_period.unwrap_or(14),
            combo.volatility_period.unwrap_or(5),
            combo.volatility_sma_period.unwrap_or(10),
            combo.upper_limit.unwrap_or(30),
            combo.lower_limit.unwrap_or(5),
        )?;
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            warmup_bars(
                combo.volatility_period.unwrap_or(5),
                combo.volatility_sma_period.unwrap_or(10),
                combo.lower_limit.unwrap_or(5),
            )
        })
        .collect();
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);

    dynamic_momentum_index_batch_inner_into(data, sweep, kernel, true, &mut values)?;

    Ok(DynamicMomentumIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn dynamic_momentum_index_batch_slice(
    data: &[f64],
    sweep: &DynamicMomentumIndexBatchRange,
    kernel: Kernel,
) -> Result<DynamicMomentumIndexBatchOutput, DynamicMomentumIndexError> {
    dynamic_momentum_index_batch_inner(data, sweep, kernel, false)
}

pub fn dynamic_momentum_index_batch_par_slice(
    data: &[f64],
    sweep: &DynamicMomentumIndexBatchRange,
    kernel: Kernel,
) -> Result<DynamicMomentumIndexBatchOutput, DynamicMomentumIndexError> {
    dynamic_momentum_index_batch_inner(data, sweep, kernel, true)
}

fn dynamic_momentum_index_batch_inner(
    data: &[f64],
    sweep: &DynamicMomentumIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DynamicMomentumIndexBatchOutput, DynamicMomentumIndexError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| DynamicMomentumIndexError::InvalidInput {
            msg: "dynamic_momentum_index: rows*cols overflow in batch".to_string(),
        })?;

    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            warmup_bars(
                combo.volatility_period.unwrap_or(5),
                combo.volatility_sma_period.unwrap_or(10),
                combo.lower_limit.unwrap_or(5),
            )
        })
        .collect();
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);

    debug_assert_eq!(values.len(), total);
    dynamic_momentum_index_batch_inner_into(data, sweep, kernel, parallel, &mut values)?;

    Ok(DynamicMomentumIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn dynamic_momentum_index_batch_inner_into(
    data: &[f64],
    sweep: &DynamicMomentumIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DynamicMomentumIndexParams>, DynamicMomentumIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(DynamicMomentumIndexError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(DynamicMomentumIndexError::EmptyInputData);
    }

    let total =
        combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| DynamicMomentumIndexError::InvalidInput {
                msg: "dynamic_momentum_index: rows*cols overflow in batch_into".to_string(),
            })?;
    if out.len() != total {
        return Err(DynamicMomentumIndexError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: total,
        });
    }

    for combo in &combos {
        validate_common(
            data,
            combo.rsi_period.unwrap_or(14),
            combo.volatility_period.unwrap_or(5),
            combo.volatility_sma_period.unwrap_or(10),
            combo.upper_limit.unwrap_or(30),
            combo.lower_limit.unwrap_or(5),
        )?;
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst: &mut [f64]| {
        dst.fill(f64::NAN);
        let combo = &combos[row];
        compute_row(
            data,
            combo.rsi_period.unwrap_or(14),
            combo.volatility_period.unwrap_or(5),
            combo.volatility_sma_period.unwrap_or(10),
            combo.upper_limit.unwrap_or(30),
            combo.lower_limit.unwrap_or(5),
            dst,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out.par_chunks_mut(len)
            .enumerate()
            .for_each(|(row, dst)| worker(row, dst));
    } else {
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "dynamic_momentum_index")]
#[pyo3(signature = (
    data,
    rsi_period=14,
    volatility_period=5,
    volatility_sma_period=10,
    upper_limit=30,
    lower_limit=5,
    kernel=None
))]
pub fn dynamic_momentum_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = DynamicMomentumIndexInput::from_slice(
        data,
        DynamicMomentumIndexParams {
            rsi_period: Some(rsi_period),
            volatility_period: Some(volatility_period),
            volatility_sma_period: Some(volatility_sma_period),
            upper_limit: Some(upper_limit),
            lower_limit: Some(lower_limit),
        },
    );
    let out = py
        .allow_threads(|| dynamic_momentum_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DynamicMomentumIndexStream")]
pub struct DynamicMomentumIndexStreamPy {
    stream: DynamicMomentumIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DynamicMomentumIndexStreamPy {
    #[new]
    #[pyo3(signature = (
        rsi_period=14,
        volatility_period=5,
        volatility_sma_period=10,
        upper_limit=30,
        lower_limit=5
    ))]
    fn new(
        rsi_period: usize,
        volatility_period: usize,
        volatility_sma_period: usize,
        upper_limit: usize,
        lower_limit: usize,
    ) -> PyResult<Self> {
        let stream = DynamicMomentumIndexStream::try_new(DynamicMomentumIndexParams {
            rsi_period: Some(rsi_period),
            volatility_period: Some(volatility_period),
            volatility_sma_period: Some(volatility_sma_period),
            upper_limit: Some(upper_limit),
            lower_limit: Some(lower_limit),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dynamic_momentum_index_batch")]
#[pyo3(signature = (
    data,
    rsi_period_range=(14, 14, 0),
    volatility_period_range=(5, 5, 0),
    volatility_sma_period_range=(10, 10, 0),
    upper_limit_range=(30, 30, 0),
    lower_limit_range=(5, 5, 0),
    kernel=None
))]
pub fn dynamic_momentum_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period_range: (usize, usize, usize),
    volatility_period_range: (usize, usize, usize),
    volatility_sma_period_range: (usize, usize, usize),
    upper_limit_range: (usize, usize, usize),
    lower_limit_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<PyObject> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let sweep = DynamicMomentumIndexBatchRange {
        rsi_period: rsi_period_range,
        volatility_period: volatility_period_range,
        volatility_sma_period: volatility_sma_period_range,
        upper_limit: upper_limit_range,
        lower_limit: lower_limit_range,
    };
    let out = py
        .allow_threads(|| dynamic_momentum_index_batch_with_kernel(data, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let values = out
        .values
        .into_pyarray(py)
        .reshape([out.rows, out.cols])?
        .into_pyobject(py)?;
    let rsi_periods: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.rsi_period.unwrap_or(14) as u64)
        .collect();
    let volatility_periods: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.volatility_period.unwrap_or(5) as u64)
        .collect();
    let volatility_sma_periods: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.volatility_sma_period.unwrap_or(10) as u64)
        .collect();
    let upper_limits: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.upper_limit.unwrap_or(30) as u64)
        .collect();
    let lower_limits: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.lower_limit.unwrap_or(5) as u64)
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("values", values)?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    dict.set_item("rsi_periods", rsi_periods.into_pyarray(py))?;
    dict.set_item("volatility_periods", volatility_periods.into_pyarray(py))?;
    dict.set_item(
        "volatility_sma_periods",
        volatility_sma_periods.into_pyarray(py),
    )?;
    dict.set_item("upper_limits", upper_limits.into_pyarray(py))?;
    dict.set_item("lower_limits", lower_limits.into_pyarray(py))?;
    Ok(dict.into_any().unbind())
}

#[cfg(feature = "python")]
pub fn register_dynamic_momentum_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dynamic_momentum_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_momentum_index_batch_py, m)?)?;
    m.add_class::<DynamicMomentumIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicMomentumIndexBatchConfig {
    pub rsi_period_range: Vec<usize>,
    pub volatility_period_range: Vec<usize>,
    pub volatility_sma_period_range: Vec<usize>,
    pub upper_limit_range: Vec<usize>,
    pub lower_limit_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dynamic_momentum_index_js)]
pub fn dynamic_momentum_index_js(
    data: &[f64],
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
) -> Result<JsValue, JsValue> {
    let input = DynamicMomentumIndexInput::from_slice(
        data,
        DynamicMomentumIndexParams {
            rsi_period: Some(rsi_period),
            volatility_period: Some(volatility_period),
            volatility_sma_period: Some(volatility_sma_period),
            upper_limit: Some(upper_limit),
            lower_limit: Some(lower_limit),
        },
    );
    let out = dynamic_momentum_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out.values).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dynamic_momentum_index_batch_js)]
pub fn dynamic_momentum_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DynamicMomentumIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.rsi_period_range.len() != 3
        || config.volatility_period_range.len() != 3
        || config.volatility_sma_period_range.len() != 3
        || config.upper_limit_range.len() != 3
        || config.lower_limit_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = dynamic_momentum_index_batch_with_kernel(
        data,
        &DynamicMomentumIndexBatchRange {
            rsi_period: (
                config.rsi_period_range[0],
                config.rsi_period_range[1],
                config.rsi_period_range[2],
            ),
            volatility_period: (
                config.volatility_period_range[0],
                config.volatility_period_range[1],
                config.volatility_period_range[2],
            ),
            volatility_sma_period: (
                config.volatility_sma_period_range[0],
                config.volatility_sma_period_range[1],
                config.volatility_sma_period_range[2],
            ),
            upper_limit: (
                config.upper_limit_range[0],
                config.upper_limit_range[1],
                config.upper_limit_range[2],
            ),
            lower_limit: (
                config.lower_limit_range[0],
                config.lower_limit_range[1],
                config.lower_limit_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("values"),
        &serde_wasm_bindgen::to_value(&out.values).unwrap(),
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
pub fn dynamic_momentum_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dynamic_momentum_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dynamic_momentum_index_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to dynamic_momentum_index_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = DynamicMomentumIndexInput::from_slice(
            data,
            DynamicMomentumIndexParams {
                rsi_period: Some(rsi_period),
                volatility_period: Some(volatility_period),
                volatility_sma_period: Some(volatility_sma_period),
                upper_limit: Some(upper_limit),
                lower_limit: Some(lower_limit),
            },
        );
        dynamic_momentum_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dynamic_momentum_index_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_period_start: usize,
    rsi_period_end: usize,
    rsi_period_step: usize,
    volatility_period_start: usize,
    volatility_period_end: usize,
    volatility_period_step: usize,
    volatility_sma_period_start: usize,
    volatility_sma_period_end: usize,
    volatility_sma_period_step: usize,
    upper_limit_start: usize,
    upper_limit_end: usize,
    upper_limit_step: usize,
    lower_limit_start: usize,
    lower_limit_end: usize,
    lower_limit_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to dynamic_momentum_index_batch_into",
        ));
    }

    let sweep = DynamicMomentumIndexBatchRange {
        rsi_period: (rsi_period_start, rsi_period_end, rsi_period_step),
        volatility_period: (
            volatility_period_start,
            volatility_period_end,
            volatility_period_step,
        ),
        volatility_sma_period: (
            volatility_sma_period_start,
            volatility_sma_period_end,
            volatility_sma_period_step,
        ),
        upper_limit: (upper_limit_start, upper_limit_end, upper_limit_step),
        lower_limit: (lower_limit_start, lower_limit_end, lower_limit_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in dynamic_momentum_index_batch_into")
    })?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        dynamic_momentum_index_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dynamic_momentum_index_output_into_js(
    data: &[f64],
    rsi_period: usize,
    volatility_period: usize,
    volatility_sma_period: usize,
    upper_limit: usize,
    lower_limit: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = dynamic_momentum_index_js(
        data,
        rsi_period,
        volatility_period,
        volatility_sma_period,
        upper_limit,
        lower_limit,
    )?;
    crate::write_wasm_object_f64_outputs("dynamic_momentum_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dynamic_momentum_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = dynamic_momentum_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "dynamic_momentum_index_batch_output_into_js",
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
                100.0
                    + ((i as f64) * 0.11).sin() * 2.3
                    + ((i as f64) * 0.037).cos() * 0.9
                    + (i as f64) * 0.015
            })
            .collect()
    }

    fn naive_dynamic_momentum_index(
        data: &[f64],
        rsi_period: usize,
        volatility_period: usize,
        volatility_sma_period: usize,
        upper_limit: usize,
        lower_limit: usize,
    ) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        compute_row(
            data,
            rsi_period,
            volatility_period,
            volatility_sma_period,
            upper_limit,
            lower_limit,
            &mut out,
        );
        out
    }

    #[test]
    fn dynamic_momentum_index_matches_naive() -> Result<(), Box<dyn Error>> {
        let close = sample_close(256);
        let input =
            DynamicMomentumIndexInput::from_slice(&close, DynamicMomentumIndexParams::default());
        let out = dynamic_momentum_index(&input)?;
        let expected = naive_dynamic_momentum_index(&close, 14, 5, 10, 30, 5);
        assert_eq!(out.values.len(), expected.len());
        for (a, b) in out.values.iter().zip(expected.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn dynamic_momentum_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let close = sample_close(220);
        let input =
            DynamicMomentumIndexInput::from_slice(&close, DynamicMomentumIndexParams::default());
        let base = dynamic_momentum_index(&input)?;
        let mut out = vec![0.0; close.len()];
        dynamic_momentum_index_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in out.iter().zip(base.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn dynamic_momentum_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = sample_close(240);
        let batch = dynamic_momentum_index(&DynamicMomentumIndexInput::from_slice(
            &close,
            DynamicMomentumIndexParams::default(),
        ))?;

        let mut stream =
            DynamicMomentumIndexStream::try_new(DynamicMomentumIndexParams::default())?;
        let mut got = Vec::with_capacity(close.len());
        for &value in &close {
            got.push(stream.update(value).unwrap_or(f64::NAN));
        }
        for (a, b) in got.iter().zip(batch.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn dynamic_momentum_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let close = sample_close(180);
        let single = dynamic_momentum_index(&DynamicMomentumIndexInput::from_slice(
            &close,
            DynamicMomentumIndexParams::default(),
        ))?;
        let batch = dynamic_momentum_index_batch_with_kernel(
            &close,
            &DynamicMomentumIndexBatchRange::default(),
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn dynamic_momentum_index_rejects_invalid_params() {
        let close = sample_close(64);
        let err = dynamic_momentum_index(&DynamicMomentumIndexInput::from_slice(
            &close,
            DynamicMomentumIndexParams {
                rsi_period: Some(0),
                ..DynamicMomentumIndexParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            DynamicMomentumIndexError::InvalidRsiPeriod { .. }
        ));

        let err = dynamic_momentum_index(&DynamicMomentumIndexInput::from_slice(
            &close,
            DynamicMomentumIndexParams {
                upper_limit: Some(4),
                lower_limit: Some(5),
                ..DynamicMomentumIndexParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            DynamicMomentumIndexError::InvalidLimits { .. }
        ));
    }

    #[test]
    fn dynamic_momentum_index_dispatch_compute_returns_value() -> Result<(), Box<dyn Error>> {
        let close = sample_close(160);
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "dynamic_momentum_index",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            params: &[
                ParamKV {
                    key: "rsi_period",
                    value: ParamValue::Int(14),
                },
                ParamKV {
                    key: "volatility_period",
                    value: ParamValue::Int(5),
                },
                ParamKV {
                    key: "volatility_sma_period",
                    value: ParamValue::Int(10),
                },
                ParamKV {
                    key: "upper_limit",
                    value: ParamValue::Int(30),
                },
                ParamKV {
                    key: "lower_limit",
                    value: ParamValue::Int(5),
                },
            ],
            kernel: Kernel::Auto,
        })?;
        let values = match out.series {
            crate::indicators::dispatch::IndicatorSeries::F64(values) => values,
            _ => panic!("expected F64 output"),
        };
        assert_eq!(values.len(), close.len());
        assert!(values.iter().any(|v| v.is_finite()));
        Ok(())
    }
}
