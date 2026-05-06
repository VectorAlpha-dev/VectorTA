#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(feature = "python")]
use numpy::{
    IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::{ManuallyDrop, MaybeUninit};
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
use thiserror::Error;

const FOUR_LN_2: f64 = 4.0 * std::f64::consts::LN_2;

#[derive(Debug, Clone)]
pub enum ParkinsonVolatilityData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct ParkinsonVolatilityOutput {
    pub volatility: Vec<f64>,
    pub variance: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ParkinsonVolatilityParams {
    pub period: Option<usize>,
}

impl Default for ParkinsonVolatilityParams {
    fn default() -> Self {
        Self { period: Some(8) }
    }
}

#[derive(Debug, Clone)]
pub struct ParkinsonVolatilityInput<'a> {
    pub data: ParkinsonVolatilityData<'a>,
    pub params: ParkinsonVolatilityParams,
}

impl<'a> ParkinsonVolatilityInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ParkinsonVolatilityParams) -> Self {
        Self {
            data: ParkinsonVolatilityData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: ParkinsonVolatilityParams) -> Self {
        Self {
            data: ParkinsonVolatilityData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, ParkinsonVolatilityParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(8)
    }

    #[inline]
    pub fn as_refs(&'a self) -> Result<(&'a [f64], &'a [f64]), ParkinsonVolatilityError> {
        match &self.data {
            ParkinsonVolatilityData::Candles { candles } => {
                Ok((candles.high.as_slice(), candles.low.as_slice()))
            }
            ParkinsonVolatilityData::Slices { high, low } => Ok((*high, *low)),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ParkinsonVolatilityBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ParkinsonVolatilityBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ParkinsonVolatilityBuilder {
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
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<ParkinsonVolatilityOutput, ParkinsonVolatilityError> {
        let params = ParkinsonVolatilityParams {
            period: self.period,
        };
        let input = ParkinsonVolatilityInput::from_candles(candles, params);
        parkinson_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<ParkinsonVolatilityOutput, ParkinsonVolatilityError> {
        let params = ParkinsonVolatilityParams {
            period: self.period,
        };
        let input = ParkinsonVolatilityInput::from_slices(high, low, params);
        parkinson_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ParkinsonVolatilityStream, ParkinsonVolatilityError> {
        let params = ParkinsonVolatilityParams {
            period: self.period,
        };
        ParkinsonVolatilityStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum ParkinsonVolatilityError {
    #[error("parkinson_volatility: Empty input data.")]
    EmptyInputData,
    #[error("parkinson_volatility: Data length mismatch between high and low.")]
    DataLengthMismatch,
    #[error("parkinson_volatility: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("parkinson_volatility: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("parkinson_volatility: All values are invalid in high or low.")]
    AllValuesNaN,
    #[error("parkinson_volatility: Candle field error: {field}")]
    CandleFieldError { field: &'static str },
    #[error("parkinson_volatility: Output length mismatch (expected {expected}, got {got})")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("parkinson_volatility: invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("parkinson_volatility: invalid range: start={start} end={end} step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("parkinson_volatility: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn parkinson_volatility(
    input: &ParkinsonVolatilityInput,
) -> Result<ParkinsonVolatilityOutput, ParkinsonVolatilityError> {
    parkinson_volatility_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn is_valid_high_low(high: f64, low: f64) -> bool {
    high.is_finite() && low.is_finite() && high > 0.0 && low > 0.0
}

#[inline(always)]
fn first_valid_high_low(high: &[f64], low: &[f64]) -> Option<usize> {
    high.iter()
        .zip(low.iter())
        .position(|(&h, &l)| is_valid_high_low(h, l))
}

#[inline(always)]
fn log_range_sq(high: f64, low: f64) -> f64 {
    let x = (high / low).ln();
    x * x
}

#[inline(always)]
fn outputs_from_sum(sum_log_sq: f64, period: usize) -> (f64, f64) {
    let variance = ((sum_log_sq / (period as f64)) / FOUR_LN_2).max(0.0);
    (variance.sqrt(), variance)
}

#[inline(always)]
fn parkinson_prepare<'a>(
    input: &'a ParkinsonVolatilityInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, Kernel), ParkinsonVolatilityError> {
    let (high, low) = input.as_refs()?;
    if high.is_empty() || low.is_empty() {
        return Err(ParkinsonVolatilityError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(ParkinsonVolatilityError::DataLengthMismatch);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(ParkinsonVolatilityError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first = first_valid_high_low(high, low).ok_or(ParkinsonVolatilityError::AllValuesNaN)?;
    if high.len() - first < period {
        return Err(ParkinsonVolatilityError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok((high, low, period, first, chosen))
}

#[inline(always)]
fn parkinson_compute_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    out_volatility: &mut [f64],
    out_variance: &mut [f64],
) {
    let warm = first + period - 1;
    if warm >= high.len() {
        return;
    }

    let mut invalid = 0usize;
    let mut sum_log_sq = 0.0f64;

    if period == 8 {
        let mut ring = [f64::NAN; 8];
        for j in 0..8 {
            let i = first + j;
            if is_valid_high_low(high[i], low[i]) {
                let value = log_range_sq(high[i], low[i]);
                ring[j] = value;
                sum_log_sq += value;
            } else {
                invalid += 1;
            }
        }

        if invalid == 0 {
            let (vol, var) = outputs_from_sum(sum_log_sq, period);
            out_volatility[warm] = vol;
            out_variance[warm] = var;
        } else {
            out_volatility[warm] = f64::NAN;
            out_variance[warm] = f64::NAN;
        }

        let mut head = 0usize;
        for i in (warm + 1)..high.len() {
            let old = ring[head];
            if old.is_nan() {
                invalid -= 1;
            } else {
                sum_log_sq -= old;
            }

            if is_valid_high_low(high[i], low[i]) {
                let value = log_range_sq(high[i], low[i]);
                ring[head] = value;
                sum_log_sq += value;
            } else {
                ring[head] = f64::NAN;
                invalid += 1;
            }

            head += 1;
            if head == 8 {
                head = 0;
            }

            if invalid == 0 {
                let (vol, var) = outputs_from_sum(sum_log_sq, period);
                out_volatility[i] = vol;
                out_variance[i] = var;
            } else {
                out_volatility[i] = f64::NAN;
                out_variance[i] = f64::NAN;
            }
        }
        return;
    }

    let mut ring = vec![f64::NAN; period];
    for j in 0..period {
        let i = first + j;
        if is_valid_high_low(high[i], low[i]) {
            let value = log_range_sq(high[i], low[i]);
            ring[j] = value;
            sum_log_sq += value;
        } else {
            invalid += 1;
        }
    }

    if invalid == 0 {
        let (vol, var) = outputs_from_sum(sum_log_sq, period);
        out_volatility[warm] = vol;
        out_variance[warm] = var;
    } else {
        out_volatility[warm] = f64::NAN;
        out_variance[warm] = f64::NAN;
    }

    let mut head = 0usize;
    for i in (warm + 1)..high.len() {
        let old = ring[head];
        if old.is_nan() {
            invalid -= 1;
        } else {
            sum_log_sq -= old;
        }

        if is_valid_high_low(high[i], low[i]) {
            let value = log_range_sq(high[i], low[i]);
            ring[head] = value;
            sum_log_sq += value;
        } else {
            ring[head] = f64::NAN;
            invalid += 1;
        }

        head += 1;
        if head == period {
            head = 0;
        }

        if invalid == 0 {
            let (vol, var) = outputs_from_sum(sum_log_sq, period);
            out_volatility[i] = vol;
            out_variance[i] = var;
        } else {
            out_volatility[i] = f64::NAN;
            out_variance[i] = f64::NAN;
        }
    }
}

#[inline]
pub fn parkinson_volatility_with_kernel(
    input: &ParkinsonVolatilityInput,
    kernel: Kernel,
) -> Result<ParkinsonVolatilityOutput, ParkinsonVolatilityError> {
    let (high, low, period, first, _chosen) = parkinson_prepare(input, kernel)?;
    let warm = first + period - 1;
    let mut volatility = alloc_with_nan_prefix(high.len(), warm);
    let mut variance = alloc_with_nan_prefix(high.len(), warm);
    parkinson_compute_into(high, low, period, first, &mut volatility, &mut variance);
    Ok(ParkinsonVolatilityOutput {
        volatility,
        variance,
    })
}

#[inline]
pub fn parkinson_volatility_into_slice(
    dst_volatility: &mut [f64],
    dst_variance: &mut [f64],
    input: &ParkinsonVolatilityInput,
    kernel: Kernel,
) -> Result<(), ParkinsonVolatilityError> {
    let (high, low, period, first, _chosen) = parkinson_prepare(input, kernel)?;
    let expected = high.len();
    if dst_volatility.len() != expected || dst_variance.len() != expected {
        return Err(ParkinsonVolatilityError::OutputLengthMismatch {
            expected,
            got: dst_volatility.len().max(dst_variance.len()),
        });
    }

    let warm = first + period - 1;
    for v in &mut dst_volatility[..warm] {
        *v = f64::NAN;
    }
    for v in &mut dst_variance[..warm] {
        *v = f64::NAN;
    }
    parkinson_compute_into(high, low, period, first, dst_volatility, dst_variance);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn parkinson_volatility_into(
    input: &ParkinsonVolatilityInput,
    out_volatility: &mut [f64],
    out_variance: &mut [f64],
) -> Result<(), ParkinsonVolatilityError> {
    parkinson_volatility_into_slice(out_volatility, out_variance, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct ParkinsonVolatilityStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    len: usize,
    invalid: usize,
    sum_log_sq: f64,
}

impl ParkinsonVolatilityStream {
    #[inline]
    pub fn try_new(params: ParkinsonVolatilityParams) -> Result<Self, ParkinsonVolatilityError> {
        let period = params.period.unwrap_or(8);
        if period == 0 {
            return Err(ParkinsonVolatilityError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            len: 0,
            invalid: 0,
            sum_log_sq: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        if self.len == self.period {
            let old = self.buffer[self.head];
            if old.is_nan() {
                self.invalid -= 1;
            } else {
                self.sum_log_sq -= old;
            }
        }

        let contrib = if is_valid_high_low(high, low) {
            log_range_sq(high, low)
        } else {
            f64::NAN
        };
        self.buffer[self.head] = contrib;
        if contrib.is_nan() {
            self.invalid += 1;
        } else {
            self.sum_log_sq += contrib;
        }

        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        if self.len < self.period {
            self.len += 1;
        }

        if self.len < self.period {
            return None;
        }
        if self.invalid != 0 {
            return Some((f64::NAN, f64::NAN));
        }
        Some(outputs_from_sum(self.sum_log_sq, self.period))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period
    }
}

#[derive(Clone, Debug)]
pub struct ParkinsonVolatilityBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ParkinsonVolatilityBatchRange {
    fn default() -> Self {
        Self {
            period: (8, 256, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ParkinsonVolatilityBatchBuilder {
    range: ParkinsonVolatilityBatchRange,
    kernel: Kernel,
}

impl ParkinsonVolatilityBatchBuilder {
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

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<ParkinsonVolatilityBatchOutput, ParkinsonVolatilityError> {
        parkinson_volatility_batch_with_kernel(high, low, &self.range, self.kernel)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct ParkinsonVolatilityBatchConfig {
    pub period_range: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ParkinsonVolatilityBatchOutput {
    pub volatility: Vec<f64>,
    pub variance: Vec<f64>,
    pub combos: Vec<ParkinsonVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ParkinsonVolatilityBatchOutput {
    pub fn row_for_params(&self, params: &ParkinsonVolatilityParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(8) == params.period.unwrap_or(8))
    }

    pub fn volatility_for(&self, params: &ParkinsonVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.volatility.get(start..start + self.cols)
        })
    }

    pub fn variance_for(&self, params: &ParkinsonVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.variance.get(start..start + self.cols)
        })
    }
}

#[inline]
pub fn expand_grid_parkinson(
    range: &ParkinsonVolatilityBatchRange,
) -> Result<Vec<ParkinsonVolatilityParams>, ParkinsonVolatilityError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, ParkinsonVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut values = Vec::new();
            let mut x = start;
            while x <= end {
                values.push(x);
                match x.checked_add(step) {
                    Some(next) if next > x => x = next,
                    _ => break,
                }
            }
            if values.is_empty() {
                return Err(ParkinsonVolatilityError::InvalidRange { start, end, step });
            }
            Ok(values)
        } else {
            let mut values = Vec::new();
            let st = step.max(1);
            let mut x = start;
            while x >= end {
                values.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(st);
                if next == x || next < end {
                    break;
                }
                x = next;
            }
            if values.is_empty() {
                return Err(ParkinsonVolatilityError::InvalidRange { start, end, step });
            }
            Ok(values)
        }
    }

    Ok(axis_usize(range.period)?
        .into_iter()
        .map(|period| ParkinsonVolatilityParams {
            period: Some(period),
        })
        .collect())
}

#[inline]
pub fn parkinson_volatility_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &ParkinsonVolatilityBatchRange,
    kernel: Kernel,
) -> Result<ParkinsonVolatilityBatchOutput, ParkinsonVolatilityError> {
    let batch = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(ParkinsonVolatilityError::InvalidKernelForBatch(other)),
    };
    parkinson_volatility_batch_par_slice(high, low, sweep, batch.to_non_batch())
}

#[inline(always)]
pub fn parkinson_volatility_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &ParkinsonVolatilityBatchRange,
    kernel: Kernel,
) -> Result<ParkinsonVolatilityBatchOutput, ParkinsonVolatilityError> {
    parkinson_volatility_batch_inner(high, low, sweep, kernel, false)
}

#[inline(always)]
pub fn parkinson_volatility_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &ParkinsonVolatilityBatchRange,
    kernel: Kernel,
) -> Result<ParkinsonVolatilityBatchOutput, ParkinsonVolatilityError> {
    parkinson_volatility_batch_inner(high, low, sweep, kernel, true)
}

#[inline(always)]
fn parkinson_volatility_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &ParkinsonVolatilityBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<ParkinsonVolatilityBatchOutput, ParkinsonVolatilityError> {
    let combos = expand_grid_parkinson(sweep)?;
    if high.is_empty() || low.is_empty() {
        return Err(ParkinsonVolatilityError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(ParkinsonVolatilityError::DataLengthMismatch);
    }

    let first = first_valid_high_low(high, low).ok_or(ParkinsonVolatilityError::AllValuesNaN)?;
    let max_period = combos
        .iter()
        .map(|c| c.period.unwrap_or(8))
        .max()
        .unwrap_or(0);
    if max_period == 0 || high.len() - first < max_period {
        return Err(ParkinsonVolatilityError::NotEnoughValidData {
            needed: max_period,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    let mut volatility_mu = make_uninit_matrix(rows, cols);
    let mut variance_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap_or(8) - 1)
        .collect();
    init_matrix_prefixes(&mut volatility_mu, cols, &warmups);
    init_matrix_prefixes(&mut variance_mu, cols, &warmups);

    let mut volatility_guard = ManuallyDrop::new(volatility_mu);
    let mut variance_guard = ManuallyDrop::new(variance_mu);
    let volatility = unsafe {
        core::slice::from_raw_parts_mut(
            volatility_guard.as_mut_ptr() as *mut f64,
            volatility_guard.len(),
        )
    };
    let variance = unsafe {
        core::slice::from_raw_parts_mut(
            variance_guard.as_mut_ptr() as *mut f64,
            variance_guard.len(),
        )
    };

    parkinson_volatility_batch_inner_into(
        high,
        low,
        sweep,
        Kernel::Scalar,
        parallel,
        volatility,
        variance,
    )?;

    let volatility_values = unsafe {
        Vec::from_raw_parts(
            volatility_guard.as_mut_ptr() as *mut f64,
            volatility_guard.len(),
            volatility_guard.capacity(),
        )
    };
    let variance_values = unsafe {
        Vec::from_raw_parts(
            variance_guard.as_mut_ptr() as *mut f64,
            variance_guard.len(),
            variance_guard.capacity(),
        )
    };

    Ok(ParkinsonVolatilityBatchOutput {
        volatility: volatility_values,
        variance: variance_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn parkinson_volatility_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &ParkinsonVolatilityBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_volatility: &mut [f64],
    out_variance: &mut [f64],
) -> Result<Vec<ParkinsonVolatilityParams>, ParkinsonVolatilityError> {
    let combos = expand_grid_parkinson(sweep)?;
    if high.is_empty() || low.is_empty() {
        return Err(ParkinsonVolatilityError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(ParkinsonVolatilityError::DataLengthMismatch);
    }

    let first = first_valid_high_low(high, low).ok_or(ParkinsonVolatilityError::AllValuesNaN)?;
    let max_period = combos
        .iter()
        .map(|c| c.period.unwrap_or(8))
        .max()
        .unwrap_or(0);
    if max_period == 0 || high.len() - first < max_period {
        return Err(ParkinsonVolatilityError::NotEnoughValidData {
            needed: max_period,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(ParkinsonVolatilityError::InvalidInput("rows*cols overflow"))?;
    if out_volatility.len() != total || out_variance.len() != total {
        return Err(ParkinsonVolatilityError::OutputLengthMismatch {
            expected: total,
            got: out_volatility.len().max(out_variance.len()),
        });
    }

    let vol_mu = unsafe {
        core::slice::from_raw_parts_mut(out_volatility.as_mut_ptr() as *mut MaybeUninit<f64>, total)
    };
    let var_mu = unsafe {
        core::slice::from_raw_parts_mut(out_variance.as_mut_ptr() as *mut MaybeUninit<f64>, total)
    };
    let warmups: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap_or(8) - 1)
        .collect();
    init_matrix_prefixes(vol_mu, cols, &warmups);
    init_matrix_prefixes(var_mu, cols, &warmups);

    let n = high.len();
    let mut prefix_sum = vec![0.0f64; n + 1];
    let mut prefix_invalid = vec![0i32; n + 1];
    for i in 0..n {
        if is_valid_high_low(high[i], low[i]) {
            prefix_sum[i + 1] = prefix_sum[i] + log_range_sq(high[i], low[i]);
            prefix_invalid[i + 1] = prefix_invalid[i];
        } else {
            prefix_sum[i + 1] = prefix_sum[i];
            prefix_invalid[i + 1] = prefix_invalid[i] + 1;
        }
    }

    let do_row = |row: usize, vol_row: &mut [f64], var_row: &mut [f64]| {
        let period = combos[row].period.unwrap_or(8);
        let warm = first + period - 1;
        for i in warm..n {
            let end = i + 1;
            let start = end - period;
            if prefix_invalid[end] - prefix_invalid[start] != 0 {
                vol_row[i] = f64::NAN;
                var_row[i] = f64::NAN;
            } else {
                let sum = prefix_sum[end] - prefix_sum[start];
                let (vol, var) = outputs_from_sum(sum, period);
                vol_row[i] = vol;
                var_row[i] = var;
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_volatility
                .par_chunks_mut(cols)
                .zip(out_variance.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (vol, var))| do_row(row, vol, var));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (vol, var)) in out_volatility
                .chunks_mut(cols)
                .zip(out_variance.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, vol, var);
            }
        }
    } else {
        for (row, (vol, var)) in out_volatility
            .chunks_mut(cols)
            .zip(out_variance.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, vol, var);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "parkinson_volatility")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn parkinson_volatility_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let params = ParkinsonVolatilityParams {
        period: Some(period),
    };
    let input = ParkinsonVolatilityInput::from_slices(high, low, params);
    let output = py
        .allow_threads(|| parkinson_volatility_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.volatility.into_pyarray(py),
        output.variance.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "ParkinsonVolatilityStream")]
pub struct ParkinsonVolatilityStreamPy {
    stream: ParkinsonVolatilityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ParkinsonVolatilityStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = ParkinsonVolatilityParams {
            period: Some(period),
        };
        let stream = ParkinsonVolatilityStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "parkinson_volatility_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn parkinson_volatility_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let sweep = ParkinsonVolatilityBatchRange {
        period: period_range,
    };
    let combos = expand_grid_parkinson(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let volatility_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let variance_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let volatility_out = unsafe { volatility_arr.as_slice_mut()? };
    let variance_out = unsafe { variance_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        parkinson_volatility_batch_inner_into(
            high,
            low,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            volatility_out,
            variance_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("volatility", volatility_arr.reshape((rows, cols))?)?;
    dict.set_item("variance", variance_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap_or(8) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(
    module = "vector_ta",
    name = "ParkinsonVolatilityDeviceArrayF32",
    unsendable
)]
pub struct ParkinsonVolatilityDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl ParkinsonVolatilityDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        let row_stride = self
            .cols
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| PyValueError::new_err("stride overflow in __cuda_array_interface__"))?;
        d.set_item("strides", (row_stride, std::mem::size_of::<f32>()))?;
        let buf = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let ptr = buf.as_device_ptr().as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<(u8, u8)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        let _ = stream;
        let _ = max_version;
        let _ = &self.ctx;
        if let Some((_ty, dev)) = dl_device {
            if dev != self.device_id as i32 {
                return Err(PyValueError::new_err("dlpack device mismatch"));
            }
        }
        if matches!(copy, Some(true)) {
            return Err(PyValueError::new_err(
                "copy=True not supported for ParkinsonVolatilityDeviceArrayF32",
            ));
        }

        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;
        export_f32_cuda_dlpack_2d(py, buf, self.rows, self.cols, self.device_id as i32, None)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "parkinson_volatility_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, period_range, device_id=0))]
pub fn parkinson_volatility_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: PyReadonlyArray1<'py, f32>,
    low_f32: PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !crate::cuda::cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let sweep = ParkinsonVolatilityBatchRange {
        period: period_range,
    };
    let (result, ctx, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda = crate::cuda::CudaParkinsonVolatility::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = cuda
            .parkinson_volatility_batch_dev(high, low, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((result, cuda.context_arc(), cuda.device_id()))
    })?;

    let rows = result.outputs.rows();
    let cols = result.outputs.cols();
    let dict = PyDict::new(py);
    dict.set_item(
        "volatility",
        Py::new(
            py,
            ParkinsonVolatilityDeviceArrayF32Py {
                buf: Some(result.outputs.volatility.buf),
                rows,
                cols,
                ctx: ctx.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "variance",
        Py::new(
            py,
            ParkinsonVolatilityDeviceArrayF32Py {
                buf: Some(result.outputs.variance.buf),
                rows,
                cols,
                ctx,
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "periods",
        result
            .combos
            .iter()
            .map(|p| p.period.unwrap_or(8) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "parkinson_volatility_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, period, device_id=0))]
pub fn parkinson_volatility_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: PyReadonlyArray2<'py, f32>,
    low_tm_f32: PyReadonlyArray2<'py, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !crate::cuda::cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let sh = high_tm_f32.shape();
    let sl = low_tm_f32.shape();
    if sh.len() != 2 || sl.len() != 2 || sh != sl {
        return Err(PyValueError::new_err(
            "expected 2D arrays with identical shape",
        ));
    }
    let rows = sh[0];
    let cols = sh[1];
    let high = high_tm_f32.as_slice()?;
    let low = low_tm_f32.as_slice()?;
    let (outputs, ctx, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda = crate::cuda::CudaParkinsonVolatility::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let outputs = cuda
            .parkinson_volatility_many_series_one_param_time_major_dev(
                high, low, cols, rows, period,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((outputs, cuda.context_arc(), cuda.device_id()))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "volatility",
        Py::new(
            py,
            ParkinsonVolatilityDeviceArrayF32Py {
                buf: Some(outputs.volatility.buf),
                rows,
                cols,
                ctx: ctx.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "variance",
        Py::new(
            py,
            ParkinsonVolatilityDeviceArrayF32Py {
                buf: Some(outputs.variance.buf),
                rows,
                cols,
                ctx,
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_parkinson_volatility_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parkinson_volatility_py, m)?)?;
    m.add_function(wrap_pyfunction!(parkinson_volatility_batch_py, m)?)?;
    m.add_class::<ParkinsonVolatilityStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<ParkinsonVolatilityDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(parkinson_volatility_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            parkinson_volatility_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "parkinson_volatility_js")]
pub fn parkinson_volatility_js(
    high: &[f64],
    low: &[f64],
    period: usize,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("high/low slice length mismatch"));
    }

    let params = ParkinsonVolatilityParams {
        period: Some(period),
    };
    let input = ParkinsonVolatilityInput::from_slices(high, low, params);
    let mut volatility = vec![0.0; high.len()];
    let mut variance = vec![0.0; high.len()];
    parkinson_volatility_into_slice(&mut volatility, &mut variance, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("volatility"),
        &serde_wasm_bindgen::to_value(&volatility).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("variance"),
        &serde_wasm_bindgen::to_value(&variance).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "parkinson_volatility_batch_js")]
pub fn parkinson_volatility_batch_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("high/low slice length mismatch"));
    }
    let config: ParkinsonVolatilityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: period_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = ParkinsonVolatilityBatchRange {
        period: (
            config.period_range[0],
            config.period_range[1],
            config.period_range[2],
        ),
    };
    let combos = expand_grid_parkinson(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let mut volatility = vec![0.0; total];
    let mut variance = vec![0.0; total];
    parkinson_volatility_batch_inner_into(
        high,
        low,
        &sweep,
        Kernel::Scalar,
        false,
        &mut volatility,
        &mut variance,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("volatility"),
        &serde_wasm_bindgen::to_value(&volatility).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("variance"),
        &serde_wasm_bindgen::to_value(&variance).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn parkinson_volatility_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn parkinson_volatility_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn parkinson_volatility_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to parkinson_volatility_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (volatility, variance) = out.split_at_mut(len);
        let params = ParkinsonVolatilityParams {
            period: Some(period),
        };
        let input = ParkinsonVolatilityInput::from_slices(high, low, params);
        parkinson_volatility_into_slice(volatility, variance, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "parkinson_volatility_into_host")]
pub fn parkinson_volatility_into_host(
    high: &[f64],
    low: &[f64],
    out_ptr: *mut f64,
    period: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to parkinson_volatility_into_host",
        ));
    }
    if high.len() != low.len() {
        return Err(JsValue::from_str("high/low slice length mismatch"));
    }

    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * high.len());
        let (volatility, variance) = out.split_at_mut(high.len());
        let params = ParkinsonVolatilityParams {
            period: Some(period),
        };
        let input = ParkinsonVolatilityInput::from_slices(high, low, params);
        parkinson_volatility_into_slice(volatility, variance, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn parkinson_volatility_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    volatility_ptr: *mut f64,
    variance_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || volatility_ptr.is_null() || variance_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to parkinson_volatility_batch_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let sweep = ParkinsonVolatilityBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos =
            expand_grid_parkinson(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let volatility = std::slice::from_raw_parts_mut(volatility_ptr, total);
        let variance = std::slice::from_raw_parts_mut(variance_ptr, total);
        parkinson_volatility_batch_inner_into(
            high,
            low,
            &sweep,
            Kernel::Scalar,
            false,
            volatility,
            variance,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn parkinson_volatility_output_into_js(
    high: &[f64],
    low: &[f64],
    period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = parkinson_volatility_js(high, low, period)?;
    crate::write_wasm_object_f64_outputs("parkinson_volatility_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn parkinson_volatility_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = parkinson_volatility_batch_js(high, low, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "parkinson_volatility_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vecs_match(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(&x, &y)| {
                (x.is_nan() && y.is_nan()) || (!x.is_nan() && !y.is_nan() && (x - y).abs() < 1e-12)
            })
    }

    fn sample_high_low() -> (Vec<f64>, Vec<f64>) {
        let high = vec![10.0, 10.4, 10.6, 10.8, 10.7, 11.0, 11.2, 11.4];
        let low = vec![9.6, 10.0, 10.1, 10.2, 10.1, 10.5, 10.8, 11.0];
        (high, low)
    }

    #[test]
    fn parkinson_output_contract() {
        let (high, low) = sample_high_low();
        let input = ParkinsonVolatilityInput::from_slices(
            &high,
            &low,
            ParkinsonVolatilityParams { period: Some(3) },
        );
        let out = parkinson_volatility(&input).expect("parkinson output");
        assert_eq!(out.volatility.len(), high.len());
        assert_eq!(out.variance.len(), high.len());
        assert!(out.volatility[..2].iter().all(|v| v.is_nan()));
        assert!(out.variance[..2].iter().all(|v| v.is_nan()));
        assert!(out.volatility[2].is_finite());
        assert!(out.variance[2].is_finite());
        assert!((out.volatility[2] * out.volatility[2] - out.variance[2]).abs() < 1e-12);
    }

    #[test]
    fn parkinson_into_matches_api() {
        let (high, low) = sample_high_low();
        let input = ParkinsonVolatilityInput::from_slices(
            &high,
            &low,
            ParkinsonVolatilityParams { period: Some(4) },
        );
        let direct = parkinson_volatility(&input).expect("direct output");
        let mut volatility = vec![0.0; high.len()];
        let mut variance = vec![0.0; high.len()];
        parkinson_volatility_into(&input, &mut volatility, &mut variance).expect("into output");
        assert!(vecs_match(&direct.volatility, &volatility));
        assert!(vecs_match(&direct.variance, &variance));
    }

    #[test]
    fn parkinson_stream_matches_batch() {
        let (high, low) = sample_high_low();
        let input = ParkinsonVolatilityInput::from_slices(
            &high,
            &low,
            ParkinsonVolatilityParams { period: Some(3) },
        );
        let batch = parkinson_volatility(&input).expect("batch output");
        let mut stream =
            ParkinsonVolatilityStream::try_new(ParkinsonVolatilityParams { period: Some(3) })
                .expect("stream");
        let mut stream_volatility = Vec::new();
        let mut stream_variance = Vec::new();
        for (&h, &l) in high.iter().zip(low.iter()) {
            match stream.update(h, l) {
                Some((vol, var)) => {
                    stream_volatility.push(vol);
                    stream_variance.push(var);
                }
                None => {
                    stream_volatility.push(f64::NAN);
                    stream_variance.push(f64::NAN);
                }
            }
        }
        assert!(vecs_match(&stream_volatility, &batch.volatility));
        assert!(vecs_match(&stream_variance, &batch.variance));
    }

    #[test]
    fn parkinson_batch_single_param_matches_single() {
        let (high, low) = sample_high_low();
        let sweep = ParkinsonVolatilityBatchRange { period: (3, 3, 0) };
        let batch = parkinson_volatility_batch_with_kernel(&high, &low, &sweep, Kernel::Auto)
            .expect("batch output");
        let input = ParkinsonVolatilityInput::from_slices(
            &high,
            &low,
            ParkinsonVolatilityParams { period: Some(3) },
        );
        let single = parkinson_volatility(&input).expect("single output");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, high.len());
        assert!(vecs_match(&batch.volatility, &single.volatility));
        assert!(vecs_match(&batch.variance, &single.variance));
    }

    #[test]
    fn parkinson_rejects_invalid_period() {
        let (high, low) = sample_high_low();
        let input = ParkinsonVolatilityInput::from_slices(
            &high,
            &low,
            ParkinsonVolatilityParams { period: Some(0) },
        );
        let err = parkinson_volatility(&input).expect_err("invalid period should fail");
        assert!(matches!(
            err,
            ParkinsonVolatilityError::InvalidPeriod { .. }
        ));
    }
}
