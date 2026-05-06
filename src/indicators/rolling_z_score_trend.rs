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

impl<'a> AsRef<[f64]> for RollingZScoreTrendInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RollingZScoreTrendData::Slice(slice) => slice,
            RollingZScoreTrendData::Candles { candles, source } => {
                if *source == "close" {
                    &candles.close
                } else {
                    source_type(candles, source)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum RollingZScoreTrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RollingZScoreTrendOutput {
    pub zscore: Vec<f64>,
    pub momentum: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RollingZScoreTrendParams {
    pub lookback_period: Option<usize>,
}

impl Default for RollingZScoreTrendParams {
    fn default() -> Self {
        Self {
            lookback_period: Some(20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingZScoreTrendInput<'a> {
    pub data: RollingZScoreTrendData<'a>,
    pub params: RollingZScoreTrendParams,
}

impl<'a> RollingZScoreTrendInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: RollingZScoreTrendParams,
    ) -> Self {
        Self {
            data: RollingZScoreTrendData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: RollingZScoreTrendParams) -> Self {
        Self {
            data: RollingZScoreTrendData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", RollingZScoreTrendParams::default())
    }

    #[inline]
    pub fn get_lookback_period(&self) -> usize {
        self.params.lookback_period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RollingZScoreTrendBuilder {
    lookback_period: Option<usize>,
    kernel: Kernel,
}

impl Default for RollingZScoreTrendBuilder {
    fn default() -> Self {
        Self {
            lookback_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RollingZScoreTrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback_period(mut self, value: usize) -> Self {
        self.lookback_period = Some(value);
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
    ) -> Result<RollingZScoreTrendOutput, RollingZScoreTrendError> {
        let params = RollingZScoreTrendParams {
            lookback_period: self.lookback_period,
        };
        rolling_z_score_trend_with_kernel(
            &RollingZScoreTrendInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<RollingZScoreTrendOutput, RollingZScoreTrendError> {
        let params = RollingZScoreTrendParams {
            lookback_period: self.lookback_period,
        };
        rolling_z_score_trend_with_kernel(
            &RollingZScoreTrendInput::from_candles(candles, source, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<RollingZScoreTrendOutput, RollingZScoreTrendError> {
        let params = RollingZScoreTrendParams {
            lookback_period: self.lookback_period,
        };
        rolling_z_score_trend_with_kernel(
            &RollingZScoreTrendInput::from_slice(data, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RollingZScoreTrendStream, RollingZScoreTrendError> {
        RollingZScoreTrendStream::try_new(RollingZScoreTrendParams {
            lookback_period: self.lookback_period,
        })
    }
}

#[derive(Debug, Error)]
pub enum RollingZScoreTrendError {
    #[error("rolling_z_score_trend: Input data slice is empty.")]
    EmptyInputData,
    #[error("rolling_z_score_trend: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "rolling_z_score_trend: Invalid lookback_period: lookback_period = {lookback_period}, data length = {data_len}"
    )]
    InvalidLookbackPeriod {
        lookback_period: usize,
        data_len: usize,
    },
    #[error("rolling_z_score_trend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rolling_z_score_trend: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("rolling_z_score_trend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("rolling_z_score_trend: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "rolling_z_score_trend: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("rolling_z_score_trend: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_params(lookback_period: usize, data_len: usize) -> Result<(), RollingZScoreTrendError> {
    if lookback_period == 0 || (data_len != usize::MAX && lookback_period > data_len) {
        return Err(RollingZScoreTrendError::InvalidLookbackPeriod {
            lookback_period,
            data_len,
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_common(data: &[f64], lookback_period: usize) -> Result<usize, RollingZScoreTrendError> {
    if data.is_empty() {
        return Err(RollingZScoreTrendError::EmptyInputData);
    }
    validate_params(lookback_period, data.len())?;
    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(RollingZScoreTrendError::AllValuesNaN);
    }
    if max_run < lookback_period {
        return Err(RollingZScoreTrendError::NotEnoughValidData {
            needed: lookback_period,
            valid: max_run,
        });
    }
    Ok(max_run)
}

#[inline(always)]
fn zscore_warmup_prefix(lookback_period: usize) -> usize {
    lookback_period.saturating_sub(1)
}

#[inline(always)]
fn momentum_warmup_prefix(lookback_period: usize) -> usize {
    lookback_period
}

#[inline(always)]
fn compute_row_all_finite(
    data: &[f64],
    lookback_period: usize,
    out_zscore: &mut [f64],
    out_momentum: &mut [f64],
) {
    out_zscore[..zscore_warmup_prefix(lookback_period).min(data.len())].fill(f64::NAN);
    out_momentum[..momentum_warmup_prefix(lookback_period).min(data.len())].fill(f64::NAN);

    let mut window = vec![0.0; lookback_period];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut smoothed = 0.0;
    let mut has_smoothed = false;
    let n = lookback_period as f64;

    for i in 0..data.len() {
        let value = data[i];
        if count < lookback_period {
            window[head] = value;
            head += 1;
            if head == lookback_period {
                head = 0;
            }
            count += 1;
            sum += value;
            sumsq += value * value;
        } else {
            let old = window[head];
            window[head] = value;
            head += 1;
            if head == lookback_period {
                head = 0;
            }
            sum += value - old;
            sumsq += value * value - old * old;
        }

        if count < lookback_period {
            continue;
        }

        let mean = sum / n;
        let variance = (sumsq / n - mean * mean).max(0.0);
        let stddev = variance.sqrt();
        let raw_zscore = if stddev > f64::EPSILON {
            (value - mean) / stddev
        } else {
            0.0
        };

        if !has_smoothed {
            smoothed = raw_zscore;
            has_smoothed = true;
            out_zscore[i] = smoothed;
        } else {
            let prev_smoothed = smoothed;
            smoothed = 0.5 * raw_zscore + 0.5 * prev_smoothed;
            out_zscore[i] = smoothed;
            out_momentum[i] = smoothed - prev_smoothed;
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingZScoreTrendStream {
    lookback_period: usize,
    window: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
    sumsq: f64,
    smoothed: f64,
    has_smoothed: bool,
}

impl RollingZScoreTrendStream {
    #[inline(always)]
    pub fn try_new(params: RollingZScoreTrendParams) -> Result<Self, RollingZScoreTrendError> {
        let lookback_period = params.lookback_period.unwrap_or(20);
        validate_params(lookback_period, usize::MAX)?;
        Ok(Self {
            lookback_period,
            window: vec![0.0; lookback_period.max(1)],
            head: 0,
            count: 0,
            sum: 0.0,
            sumsq: 0.0,
            smoothed: f64::NAN,
            has_smoothed: false,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sumsq = 0.0;
        self.smoothed = f64::NAN;
        self.has_smoothed = false;
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        if self.count < self.lookback_period {
            self.window[self.head] = value;
            self.head = (self.head + 1) % self.lookback_period;
            self.count += 1;
            self.sum += value;
            self.sumsq += value * value;
        } else {
            let old = self.window[self.head];
            self.window[self.head] = value;
            self.head = (self.head + 1) % self.lookback_period;
            self.sum += value - old;
            self.sumsq += value * value - old * old;
        }

        if self.count < self.lookback_period {
            return None;
        }

        let n = self.lookback_period as f64;
        let mean = self.sum / n;
        let variance = (self.sumsq / n - mean * mean).max(0.0);
        let stddev = variance.sqrt();
        let raw_zscore = if stddev > f64::EPSILON {
            (value - mean) / stddev
        } else {
            0.0
        };

        if !self.has_smoothed {
            self.smoothed = raw_zscore;
            self.has_smoothed = true;
            return Some((self.smoothed, f64::NAN));
        }

        let prev_smoothed = self.smoothed;
        self.smoothed = 0.5 * raw_zscore + 0.5 * prev_smoothed;
        Some((self.smoothed, self.smoothed - prev_smoothed))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        momentum_warmup_prefix(self.lookback_period)
    }
}

#[inline(always)]
fn compute_row(
    data: &[f64],
    lookback_period: usize,
    all_finite: bool,
    out_zscore: &mut [f64],
    out_momentum: &mut [f64],
) {
    if all_finite {
        compute_row_all_finite(data, lookback_period, out_zscore, out_momentum);
        return;
    } else {
        out_zscore.fill(f64::NAN);
        out_momentum.fill(f64::NAN);
    }
    let mut stream = RollingZScoreTrendStream::try_new(RollingZScoreTrendParams {
        lookback_period: Some(lookback_period),
    })
    .expect("validated params");

    for (i, &value) in data.iter().enumerate() {
        if let Some((zscore, momentum)) = stream.update(value) {
            out_zscore[i] = zscore;
            out_momentum[i] = momentum;
        }
    }
}

pub fn rolling_z_score_trend(
    input: &RollingZScoreTrendInput,
) -> Result<RollingZScoreTrendOutput, RollingZScoreTrendError> {
    rolling_z_score_trend_with_kernel(input, Kernel::Auto)
}

pub fn rolling_z_score_trend_with_kernel(
    input: &RollingZScoreTrendInput,
    kernel: Kernel,
) -> Result<RollingZScoreTrendOutput, RollingZScoreTrendError> {
    let data = input.as_ref();
    let lookback_period = input.get_lookback_period();
    let max_run = validate_common(data, lookback_period)?;

    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let mut zscore = alloc_with_nan_prefix(data.len(), zscore_warmup_prefix(lookback_period));
    let mut momentum = alloc_with_nan_prefix(data.len(), momentum_warmup_prefix(lookback_period));
    compute_row(
        data,
        lookback_period,
        max_run == data.len(),
        &mut zscore,
        &mut momentum,
    );
    Ok(RollingZScoreTrendOutput { zscore, momentum })
}

pub fn rolling_z_score_trend_into_slice(
    dst_zscore: &mut [f64],
    dst_momentum: &mut [f64],
    input: &RollingZScoreTrendInput,
    kernel: Kernel,
) -> Result<(), RollingZScoreTrendError> {
    let data = input.as_ref();
    let lookback_period = input.get_lookback_period();
    let max_run = validate_common(data, lookback_period)?;
    if dst_zscore.len() != data.len() {
        return Err(RollingZScoreTrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_zscore.len(),
        });
    }
    if dst_momentum.len() != data.len() {
        return Err(RollingZScoreTrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_momentum.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    compute_row(
        data,
        lookback_period,
        max_run == data.len(),
        dst_zscore,
        dst_momentum,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn rolling_z_score_trend_into(
    input: &RollingZScoreTrendInput,
    out_zscore: &mut [f64],
    out_momentum: &mut [f64],
) -> Result<(), RollingZScoreTrendError> {
    rolling_z_score_trend_into_slice(out_zscore, out_momentum, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct RollingZScoreTrendBatchRange {
    pub lookback_period: (usize, usize, usize),
}

impl Default for RollingZScoreTrendBatchRange {
    fn default() -> Self {
        Self {
            lookback_period: (20, 20, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingZScoreTrendBatchOutput {
    pub zscore: Vec<f64>,
    pub momentum: Vec<f64>,
    pub combos: Vec<RollingZScoreTrendParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RollingZScoreTrendBatchOutput {
    pub fn row_for_params(&self, params: &RollingZScoreTrendParams) -> Option<usize> {
        let lookback_period = params.lookback_period.unwrap_or(20);
        self.combos
            .iter()
            .position(|combo| combo.lookback_period.unwrap_or(20) == lookback_period)
    }

    pub fn zscore_for(&self, params: &RollingZScoreTrendParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.zscore.get(start..start + self.cols)
        })
    }

    pub fn momentum_for(&self, params: &RollingZScoreTrendParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.momentum.get(start..start + self.cols)
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RollingZScoreTrendBatchBuilder {
    range: RollingZScoreTrendBatchRange,
    kernel: Kernel,
}

impl Default for RollingZScoreTrendBatchBuilder {
    fn default() -> Self {
        Self {
            range: RollingZScoreTrendBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl RollingZScoreTrendBatchBuilder {
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
    pub fn lookback_period_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.lookback_period = value;
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<RollingZScoreTrendBatchOutput, RollingZScoreTrendError> {
        rolling_z_score_trend_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<RollingZScoreTrendBatchOutput, RollingZScoreTrendError> {
        rolling_z_score_trend_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_axis(range: (usize, usize, usize)) -> Result<Vec<usize>, RollingZScoreTrendError> {
    let (start, end, step) = range;
    if start == 0 {
        return Err(RollingZScoreTrendError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(RollingZScoreTrendError::InvalidRange { start, end, step });
    }
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(cur);
        if cur >= end {
            break;
        }
        let next = cur
            .checked_add(step)
            .ok_or_else(|| RollingZScoreTrendError::InvalidInput {
                msg: "rolling_z_score_trend: range step overflow".to_string(),
            })?;
        if next <= cur {
            return Err(RollingZScoreTrendError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_checked(
    range: &RollingZScoreTrendBatchRange,
) -> Result<Vec<RollingZScoreTrendParams>, RollingZScoreTrendError> {
    let lookback_periods = expand_axis(range.lookback_period)?;
    Ok(lookback_periods
        .into_iter()
        .map(|lookback_period| RollingZScoreTrendParams {
            lookback_period: Some(lookback_period),
        })
        .collect())
}

pub fn expand_grid_rolling_z_score_trend(
    range: &RollingZScoreTrendBatchRange,
) -> Vec<RollingZScoreTrendParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn rolling_z_score_trend_batch_with_kernel(
    data: &[f64],
    sweep: &RollingZScoreTrendBatchRange,
    kernel: Kernel,
) -> Result<RollingZScoreTrendBatchOutput, RollingZScoreTrendError> {
    rolling_z_score_trend_batch_inner(data, sweep, kernel, true)
}

pub fn rolling_z_score_trend_batch_slice(
    data: &[f64],
    sweep: &RollingZScoreTrendBatchRange,
    kernel: Kernel,
) -> Result<RollingZScoreTrendBatchOutput, RollingZScoreTrendError> {
    rolling_z_score_trend_batch_inner(data, sweep, kernel, false)
}

pub fn rolling_z_score_trend_batch_par_slice(
    data: &[f64],
    sweep: &RollingZScoreTrendBatchRange,
    kernel: Kernel,
) -> Result<RollingZScoreTrendBatchOutput, RollingZScoreTrendError> {
    rolling_z_score_trend_batch_inner(data, sweep, kernel, true)
}

fn rolling_z_score_trend_batch_inner(
    data: &[f64],
    sweep: &RollingZScoreTrendBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<RollingZScoreTrendBatchOutput, RollingZScoreTrendError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| RollingZScoreTrendError::InvalidInput {
            msg: "rolling_z_score_trend: rows*cols overflow in batch".to_string(),
        })?;

    if data.is_empty() {
        return Err(RollingZScoreTrendError::EmptyInputData);
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(RollingZScoreTrendError::AllValuesNaN);
    }

    let mut max_needed = 0usize;
    let zscore_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let lookback_period = params.lookback_period.unwrap_or(20);
            max_needed = max_needed.max(lookback_period);
            zscore_warmup_prefix(lookback_period)
        })
        .collect();
    let momentum_warmups: Vec<usize> = combos
        .iter()
        .map(|params| momentum_warmup_prefix(params.lookback_period.unwrap_or(20)))
        .collect();
    if max_run < max_needed {
        return Err(RollingZScoreTrendError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let mut zscore_mu = make_uninit_matrix(rows, cols);
    let mut momentum_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut zscore_mu, cols, &zscore_warmups);
    init_matrix_prefixes(&mut momentum_mu, cols, &momentum_warmups);

    let mut zscore = unsafe {
        Vec::from_raw_parts(
            zscore_mu.as_mut_ptr() as *mut f64,
            zscore_mu.len(),
            zscore_mu.capacity(),
        )
    };
    let mut momentum = unsafe {
        Vec::from_raw_parts(
            momentum_mu.as_mut_ptr() as *mut f64,
            momentum_mu.len(),
            momentum_mu.capacity(),
        )
    };
    std::mem::forget(zscore_mu);
    std::mem::forget(momentum_mu);
    debug_assert_eq!(zscore.len(), total);
    debug_assert_eq!(momentum.len(), total);

    rolling_z_score_trend_batch_inner_into(
        data,
        sweep,
        kernel,
        parallel,
        &mut zscore,
        &mut momentum,
    )?;

    Ok(RollingZScoreTrendBatchOutput {
        zscore,
        momentum,
        combos,
        rows,
        cols,
    })
}

fn rolling_z_score_trend_batch_inner_into(
    data: &[f64],
    sweep: &RollingZScoreTrendBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_zscore: &mut [f64],
    out_momentum: &mut [f64],
) -> Result<Vec<RollingZScoreTrendParams>, RollingZScoreTrendError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(RollingZScoreTrendError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(RollingZScoreTrendError::EmptyInputData);
    }
    let total =
        combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| RollingZScoreTrendError::InvalidInput {
                msg: "rolling_z_score_trend: rows*cols overflow in batch_into".to_string(),
            })?;
    if out_zscore.len() != total {
        return Err(RollingZScoreTrendError::MismatchedOutputLen {
            dst_len: out_zscore.len(),
            expected_len: total,
        });
    }
    if out_momentum.len() != total {
        return Err(RollingZScoreTrendError::MismatchedOutputLen {
            dst_len: out_momentum.len(),
            expected_len: total,
        });
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(RollingZScoreTrendError::AllValuesNaN);
    }
    let max_needed = combos
        .iter()
        .map(|params| params.lookback_period.unwrap_or(20))
        .max()
        .unwrap_or(0);
    if max_run < max_needed {
        return Err(RollingZScoreTrendError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_zscore: &mut [f64], dst_momentum: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            data,
            params.lookback_period.unwrap_or(20),
            max_run == data.len(),
            dst_zscore,
            dst_momentum,
        );
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_zscore
                .par_chunks_mut(len)
                .zip(out_momentum.par_chunks_mut(len))
                .enumerate()
                .for_each(|(row, (dst_zscore, dst_momentum))| {
                    worker(row, dst_zscore, dst_momentum)
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (dst_zscore, dst_momentum)) in out_zscore
                .chunks_mut(len)
                .zip(out_momentum.chunks_mut(len))
                .enumerate()
            {
                worker(row, dst_zscore, dst_momentum);
            }
        }
    } else {
        for (row, (dst_zscore, dst_momentum)) in out_zscore
            .chunks_mut(len)
            .zip(out_momentum.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_zscore, dst_momentum);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "rolling_z_score_trend")]
#[pyo3(signature = (data, lookback_period=20, kernel=None))]
pub fn rolling_z_score_trend_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback_period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = RollingZScoreTrendInput::from_slice(
        data,
        RollingZScoreTrendParams {
            lookback_period: Some(lookback_period),
        },
    );
    let out = py
        .allow_threads(|| rolling_z_score_trend_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.zscore.into_pyarray(py), out.momentum.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "RollingZScoreTrendStream")]
pub struct RollingZScoreTrendStreamPy {
    stream: RollingZScoreTrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RollingZScoreTrendStreamPy {
    #[new]
    #[pyo3(signature = (lookback_period=20))]
    fn new(lookback_period: usize) -> PyResult<Self> {
        let stream = RollingZScoreTrendStream::try_new(RollingZScoreTrendParams {
            lookback_period: Some(lookback_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "rolling_z_score_trend_batch")]
#[pyo3(signature = (data, lookback_period_range=(20,20,0), kernel=None))]
pub fn rolling_z_score_trend_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            rolling_z_score_trend_batch_with_kernel(
                data,
                &RollingZScoreTrendBatchRange {
                    lookback_period: lookback_period_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "zscore",
        output
            .zscore
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "momentum",
        output
            .momentum
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lookback_periods",
        output
            .combos
            .iter()
            .map(|params| params.lookback_period.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_rolling_z_score_trend_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_z_score_trend_py, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_z_score_trend_batch_py, m)?)?;
    m.add_class::<RollingZScoreTrendStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingZScoreTrendBatchConfig {
    pub lookback_period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = rolling_z_score_trend_js)]
pub fn rolling_z_score_trend_js(data: &[f64], lookback_period: usize) -> Result<JsValue, JsValue> {
    let input = RollingZScoreTrendInput::from_slice(
        data,
        RollingZScoreTrendParams {
            lookback_period: Some(lookback_period),
        },
    );
    let out = rolling_z_score_trend_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("zscore"),
        &serde_wasm_bindgen::to_value(&out.zscore).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("momentum"),
        &serde_wasm_bindgen::to_value(&out.momentum).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = rolling_z_score_trend_batch_js)]
pub fn rolling_z_score_trend_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: RollingZScoreTrendBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.lookback_period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = rolling_z_score_trend_batch_with_kernel(
        data,
        &RollingZScoreTrendBatchRange {
            lookback_period: (
                config.lookback_period_range[0],
                config.lookback_period_range[1],
                config.lookback_period_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("zscore"),
        &serde_wasm_bindgen::to_value(&out.zscore).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("momentum"),
        &serde_wasm_bindgen::to_value(&out.momentum).unwrap(),
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
pub fn rolling_z_score_trend_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(2 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_z_score_trend_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_z_score_trend_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_period: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to rolling_z_score_trend_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (dst_zscore, dst_momentum) = out.split_at_mut(len);
        let input = RollingZScoreTrendInput::from_slice(
            data,
            RollingZScoreTrendParams {
                lookback_period: Some(lookback_period),
            },
        );
        rolling_z_score_trend_into_slice(dst_zscore, dst_momentum, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_z_score_trend_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_period_start: usize,
    lookback_period_end: usize,
    lookback_period_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to rolling_z_score_trend_batch_into",
        ));
    }
    let sweep = RollingZScoreTrendBatchRange {
        lookback_period: (
            lookback_period_start,
            lookback_period_end,
            lookback_period_step,
        ),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in rolling_z_score_trend_batch_into")
        })?;
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_zscore, dst_momentum) = out.split_at_mut(split);
        rolling_z_score_trend_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            dst_zscore,
            dst_momentum,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_z_score_trend_output_into_js(
    data: &[f64],
    lookback_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = rolling_z_score_trend_js(data, lookback_period)?;
    crate::write_wasm_object_f64_outputs("rolling_z_score_trend_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_z_score_trend_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = rolling_z_score_trend_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "rolling_z_score_trend_batch_output_into_js",
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

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.12 + (x * 0.17).sin() * 3.0 + (x * 0.047).cos() * 1.1
            })
            .collect()
    }

    fn naive_indicator(data: &[f64], lookback_period: usize) -> (Vec<f64>, Vec<f64>) {
        let mut zscore = vec![f64::NAN; data.len()];
        let mut momentum = vec![f64::NAN; data.len()];
        let mut prev_smoothed = f64::NAN;
        for i in (lookback_period - 1)..data.len() {
            let window = &data[i + 1 - lookback_period..=i];
            if window.iter().any(|v| !v.is_finite()) {
                prev_smoothed = f64::NAN;
                continue;
            }
            let n = lookback_period as f64;
            let mean = window.iter().sum::<f64>() / n;
            let variance = (window.iter().map(|v| v * v).sum::<f64>() / n - mean * mean).max(0.0);
            let stddev = variance.sqrt();
            let raw = if stddev > f64::EPSILON {
                (data[i] - mean) / stddev
            } else {
                0.0
            };
            let smooth = if prev_smoothed.is_nan() {
                raw
            } else {
                0.5 * raw + 0.5 * prev_smoothed
            };
            zscore[i] = smooth;
            if !prev_smoothed.is_nan() {
                momentum[i] = smooth - prev_smoothed;
            }
            prev_smoothed = smooth;
        }
        (zscore, momentum)
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
    fn rolling_z_score_trend_matches_naive() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let input = RollingZScoreTrendInput::from_slice(
            &data,
            RollingZScoreTrendParams {
                lookback_period: Some(20),
            },
        );
        let out = rolling_z_score_trend_with_kernel(&input, Kernel::Scalar)?;
        let (expected_zscore, expected_momentum) = naive_indicator(&data, 20);
        assert_series_close(&out.zscore, &expected_zscore, 1e-10);
        assert_series_close(&out.momentum, &expected_momentum, 1e-10);
        Ok(())
    }

    #[test]
    fn rolling_z_score_trend_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data(220);
        let input = RollingZScoreTrendInput::from_slice(
            &data,
            RollingZScoreTrendParams {
                lookback_period: Some(18),
            },
        );
        let base = rolling_z_score_trend(&input)?;
        let mut zscore = vec![0.0; data.len()];
        let mut momentum = vec![0.0; data.len()];
        rolling_z_score_trend_into_slice(&mut zscore, &mut momentum, &input, Kernel::Auto)?;
        assert_series_close(&base.zscore, &zscore, 1e-12);
        assert_series_close(&base.momentum, &momentum, 1e-12);
        Ok(())
    }

    #[test]
    fn rolling_z_score_trend_into_overwrites_stale_constant_data() -> Result<(), Box<dyn Error>> {
        let data = vec![42.0; 64];
        let input = RollingZScoreTrendInput::from_slice(
            &data,
            RollingZScoreTrendParams {
                lookback_period: Some(20),
            },
        );
        let mut zscore = vec![123.0; data.len()];
        let mut momentum = vec![456.0; data.len()];
        rolling_z_score_trend_into_slice(&mut zscore, &mut momentum, &input, Kernel::Auto)?;

        assert!(zscore[..19].iter().all(|v| v.is_nan()));
        assert!(zscore[19..].iter().all(|v| *v == 0.0));
        assert!(momentum[..20].iter().all(|v| v.is_nan()));
        assert!(momentum[20..].iter().all(|v| *v == 0.0));
        Ok(())
    }

    #[test]
    fn rolling_z_score_trend_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let batch = rolling_z_score_trend(&RollingZScoreTrendInput::from_slice(
            &data,
            RollingZScoreTrendParams {
                lookback_period: Some(20),
            },
        ))?;

        let mut stream = RollingZScoreTrendStream::try_new(RollingZScoreTrendParams {
            lookback_period: Some(20),
        })?;
        let mut zscore = vec![f64::NAN; data.len()];
        let mut momentum = vec![f64::NAN; data.len()];
        for (i, &value) in data.iter().enumerate() {
            if let Some((z, m)) = stream.update(value) {
                zscore[i] = z;
                momentum[i] = m;
            }
        }

        assert_series_close(&batch.zscore, &zscore, 1e-12);
        assert_series_close(&batch.momentum, &momentum, 1e-12);
        Ok(())
    }

    #[test]
    fn rolling_z_score_trend_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let data = sample_data(192);
        let single = rolling_z_score_trend(&RollingZScoreTrendInput::from_slice(
            &data,
            RollingZScoreTrendParams {
                lookback_period: Some(15),
            },
        ))?;

        let batch = rolling_z_score_trend_batch_with_kernel(
            &data,
            &RollingZScoreTrendBatchRange {
                lookback_period: (15, 15, 0),
            },
            Kernel::Auto,
        )?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_series_close(&batch.zscore[..data.len()], &single.zscore, 1e-12);
        assert_series_close(&batch.momentum[..data.len()], &single.momentum, 1e-12);
        Ok(())
    }

    #[test]
    fn rolling_z_score_trend_rejects_invalid_params() {
        let data = sample_data(64);

        let err = rolling_z_score_trend(&RollingZScoreTrendInput::from_slice(
            &data,
            RollingZScoreTrendParams {
                lookback_period: Some(0),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            RollingZScoreTrendError::InvalidLookbackPeriod { .. }
        ));
    }

    #[test]
    fn rolling_z_score_trend_dispatch_compute_returns_outputs() -> Result<(), Box<dyn Error>> {
        let data = sample_data(128);
        let params = [ParamKV {
            key: "lookback_period",
            value: ParamValue::Int(20),
        }];

        let zscore = compute_cpu(IndicatorComputeRequest {
            indicator_id: "rolling_z_score_trend",
            data: IndicatorDataRef::Slice { values: &data },
            params: &params,
            output_id: Some("zscore"),
            kernel: Kernel::Auto,
        })?;
        let momentum = compute_cpu(IndicatorComputeRequest {
            indicator_id: "rolling_z_score_trend",
            data: IndicatorDataRef::Slice { values: &data },
            params: &params,
            output_id: Some("momentum"),
            kernel: Kernel::Auto,
        })?;

        assert_eq!(zscore.output_id, "zscore");
        assert_eq!(momentum.output_id, "momentum");
        assert_eq!(zscore.rows, 1);
        assert_eq!(momentum.rows, 1);
        assert_eq!(zscore.cols, data.len());
        assert_eq!(momentum.cols, data.len());
        Ok(())
    }
}
