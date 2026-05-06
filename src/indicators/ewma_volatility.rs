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
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EwmaVolatilityInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EwmaVolatilityData::Slice(slice) => slice,
            EwmaVolatilityData::Candles { candles, source } => {
                ewma_volatility_source(candles, source)
            }
        }
    }
}

#[inline(always)]
fn ewma_volatility_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "close" => &candles.close,
        "volume" => &candles.volume,
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum EwmaVolatilityData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EwmaVolatilityOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EwmaVolatilityParams {
    pub lambda: Option<f64>,
}

impl Default for EwmaVolatilityParams {
    fn default() -> Self {
        Self { lambda: Some(0.94) }
    }
}

#[derive(Debug, Clone)]
pub struct EwmaVolatilityInput<'a> {
    pub data: EwmaVolatilityData<'a>,
    pub params: EwmaVolatilityParams,
}

impl<'a> EwmaVolatilityInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EwmaVolatilityParams,
    ) -> Self {
        Self {
            data: EwmaVolatilityData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: EwmaVolatilityParams) -> Self {
        Self {
            data: EwmaVolatilityData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", EwmaVolatilityParams::default())
    }

    #[inline]
    pub fn get_lambda(&self) -> f64 {
        self.params.lambda.unwrap_or(0.94)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EwmaVolatilityBuilder {
    lambda: Option<f64>,
    kernel: Kernel,
}

impl Default for EwmaVolatilityBuilder {
    fn default() -> Self {
        Self {
            lambda: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EwmaVolatilityBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lambda(mut self, value: f64) -> Self {
        self.lambda = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<EwmaVolatilityOutput, EwmaVolatilityError> {
        let input = EwmaVolatilityInput::from_candles(
            candles,
            "close",
            EwmaVolatilityParams {
                lambda: self.lambda,
            },
        );
        ewma_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<EwmaVolatilityOutput, EwmaVolatilityError> {
        let input = EwmaVolatilityInput::from_slice(
            data,
            EwmaVolatilityParams {
                lambda: self.lambda,
            },
        );
        ewma_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EwmaVolatilityStream, EwmaVolatilityError> {
        EwmaVolatilityStream::try_new(EwmaVolatilityParams {
            lambda: self.lambda,
        })
    }
}

#[derive(Debug, Error)]
pub enum EwmaVolatilityError {
    #[error("ewma_volatility: Input data slice is empty.")]
    EmptyInputData,
    #[error("ewma_volatility: All values are NaN.")]
    AllValuesNaN,
    #[error("ewma_volatility: Invalid lambda: {lambda}. Expected finite value in [0, 1).")]
    InvalidLambda { lambda: f64 },
    #[error("ewma_volatility: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ewma_volatility: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ewma_volatility: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ewma_volatility: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct EwmaPrepared {
    sq_returns: Vec<f64>,
    valid_indices: Vec<usize>,
    valid_values: Vec<f64>,
}

const EWMA_SCALE: f64 = 100.0;

#[inline(always)]
fn period_from_lambda(lambda: f64) -> Result<usize, EwmaVolatilityError> {
    if !lambda.is_finite() || !(0.0..1.0).contains(&lambda) {
        return Err(EwmaVolatilityError::InvalidLambda { lambda });
    }
    let raw = (2.0 / (1.0 - lambda) - 1.0).round();
    let period = raw.max(1.0) as usize;
    Ok(period)
}

#[inline(always)]
fn alpha_from_period(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

#[inline(always)]
fn prepare_returns(data: &[f64]) -> Result<EwmaPrepared, EwmaVolatilityError> {
    if data.is_empty() {
        return Err(EwmaVolatilityError::EmptyInputData);
    }
    if !data.iter().any(|v| !v.is_nan()) {
        return Err(EwmaVolatilityError::AllValuesNaN);
    }

    let len = data.len();
    let mut sq_returns = vec![f64::NAN; len];
    let mut valid_indices = Vec::with_capacity(len.saturating_sub(1));
    let mut valid_values = Vec::with_capacity(len.saturating_sub(1));

    for i in 1..len {
        let prev = data[i - 1];
        let curr = data[i];
        if prev.is_finite() && curr.is_finite() && prev > 0.0 && curr > 0.0 {
            let ret = (curr / prev).ln();
            let sq = ret * ret;
            sq_returns[i] = sq;
            valid_indices.push(i);
            valid_values.push(sq);
        }
    }

    Ok(EwmaPrepared {
        sq_returns,
        valid_indices,
        valid_values,
    })
}

#[inline(always)]
fn validate_ewma_data(data: &[f64]) -> Result<(), EwmaVolatilityError> {
    if data.is_empty() {
        return Err(EwmaVolatilityError::EmptyInputData);
    }
    if !data.iter().any(|v| !v.is_nan()) {
        return Err(EwmaVolatilityError::AllValuesNaN);
    }
    Ok(())
}

#[inline(always)]
fn valid_sq_return(prev: f64, curr: f64) -> Option<f64> {
    if prev.is_finite() && curr.is_finite() && prev > 0.0 && curr > 0.0 {
        let ret = (curr / prev).ln();
        Some(ret * ret)
    } else {
        None
    }
}

#[inline(always)]
fn seed_ewma_single_pass(data: &[f64], period: usize) -> Result<(usize, f64), EwmaVolatilityError> {
    let mut valid = 0usize;
    let mut sum = 0.0;

    for i in 1..data.len() {
        if let Some(sq) = valid_sq_return(data[i - 1], data[i]) {
            valid += 1;
            sum += sq;
            if valid == period {
                return Ok((i, sum / period as f64));
            }
        }
    }

    Err(EwmaVolatilityError::NotEnoughValidData {
        needed: period,
        valid,
    })
}

#[inline(always)]
fn fill_row_single_pass(data: &[f64], seed_idx: usize, mut ema: f64, alpha: f64, out: &mut [f64]) {
    let beta = 1.0 - alpha;

    out[seed_idx] = ema.max(0.0).sqrt() * EWMA_SCALE;
    for i in (seed_idx + 1)..out.len() {
        if let Some(sq) = valid_sq_return(data[i - 1], data[i]) {
            ema = beta.mul_add(ema, alpha * sq);
        }
        out[i] = ema.max(0.0).sqrt() * EWMA_SCALE;
    }
}

#[inline(always)]
fn fill_row_from_precomputed(
    prep: &EwmaPrepared,
    period: usize,
    alpha: f64,
    out: &mut [f64],
) -> Result<usize, EwmaVolatilityError> {
    if prep.valid_values.len() < period {
        return Err(EwmaVolatilityError::NotEnoughValidData {
            needed: period,
            valid: prep.valid_values.len(),
        });
    }

    let seed_idx = prep.valid_indices[period - 1];
    let mut ema = prep.valid_values[..period].iter().copied().sum::<f64>() / period as f64;
    let beta = 1.0 - alpha;

    out[seed_idx] = ema.max(0.0).sqrt() * EWMA_SCALE;
    for i in (seed_idx + 1)..out.len() {
        let sq = prep.sq_returns[i];
        if sq.is_finite() {
            ema = beta.mul_add(ema, alpha * sq);
        }
        out[i] = ema.max(0.0).sqrt() * EWMA_SCALE;
    }

    Ok(seed_idx)
}

#[inline]
pub fn ewma_volatility(
    input: &EwmaVolatilityInput,
) -> Result<EwmaVolatilityOutput, EwmaVolatilityError> {
    ewma_volatility_with_kernel(input, Kernel::Auto)
}

pub fn ewma_volatility_with_kernel(
    input: &EwmaVolatilityInput,
    _kernel: Kernel,
) -> Result<EwmaVolatilityOutput, EwmaVolatilityError> {
    let data = input.as_ref();
    let period = period_from_lambda(input.get_lambda())?;
    validate_ewma_data(data)?;
    let alpha = alpha_from_period(period);
    let (seed_idx, ema) = seed_ewma_single_pass(data, period)?;
    let mut out = alloc_with_nan_prefix(data.len(), seed_idx);
    fill_row_single_pass(data, seed_idx, ema, alpha, &mut out);
    Ok(EwmaVolatilityOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn ewma_volatility_into(
    input: &EwmaVolatilityInput,
    out: &mut [f64],
) -> Result<(), EwmaVolatilityError> {
    ewma_volatility_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn ewma_volatility_into_slice(
    dst: &mut [f64],
    input: &EwmaVolatilityInput,
    _kernel: Kernel,
) -> Result<(), EwmaVolatilityError> {
    let data = input.as_ref();
    let period = period_from_lambda(input.get_lambda())?;
    validate_ewma_data(data)?;
    if dst.len() != data.len() {
        return Err(EwmaVolatilityError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);
    let alpha = alpha_from_period(period);
    let (seed_idx, ema) = seed_ewma_single_pass(data, period)?;
    fill_row_single_pass(data, seed_idx, ema, alpha, dst);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct EwmaVolatilityStream {
    period: usize,
    alpha: f64,
    prev_close: f64,
    seed_window: Vec<f64>,
    seed_sum: f64,
    ema: f64,
    seeded: bool,
}

impl EwmaVolatilityStream {
    #[inline(always)]
    pub fn try_new(params: EwmaVolatilityParams) -> Result<Self, EwmaVolatilityError> {
        let lambda = params.lambda.unwrap_or(0.94);
        let period = period_from_lambda(lambda)?;
        Ok(Self {
            period,
            alpha: alpha_from_period(period),
            prev_close: f64::NAN,
            seed_window: Vec::with_capacity(period),
            seed_sum: 0.0,
            ema: f64::NAN,
            seeded: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64) -> Option<f64> {
        let ret_valid = self.prev_close.is_finite()
            && self.prev_close > 0.0
            && close.is_finite()
            && close > 0.0;

        if ret_valid {
            let ret = (close / self.prev_close).ln();
            let sq = ret * ret;
            if !self.seeded {
                self.seed_window.push(sq);
                self.seed_sum += sq;
                if self.seed_window.len() == self.period {
                    self.ema = self.seed_sum / self.period as f64;
                    self.seeded = true;
                }
            } else {
                self.ema = (1.0 - self.alpha).mul_add(self.ema, self.alpha * sq);
            }
        }

        self.prev_close = close;

        if self.seeded {
            Some(self.ema.max(0.0).sqrt() * EWMA_SCALE)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period
    }
}

#[derive(Clone, Debug)]
pub struct EwmaVolatilityBatchRange {
    pub lambda: (f64, f64, f64),
}

impl Default for EwmaVolatilityBatchRange {
    fn default() -> Self {
        Self {
            lambda: (0.94, 0.94, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EwmaVolatilityBatchBuilder {
    range: EwmaVolatilityBatchRange,
    kernel: Kernel,
}

impl EwmaVolatilityBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn lambda_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.lambda = (start, end, step);
        self
    }

    pub fn lambda_static(mut self, value: f64) -> Self {
        self.range.lambda = (value, value, 0.0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EwmaVolatilityBatchOutput, EwmaVolatilityError> {
        ewma_volatility_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<EwmaVolatilityBatchOutput, EwmaVolatilityError> {
        self.apply_slice(&candles.close)
    }
}

#[derive(Clone, Debug)]
pub struct EwmaVolatilityBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EwmaVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EwmaVolatilityBatchOutput {
    pub fn row_for_params(&self, params: &EwmaVolatilityParams) -> Option<usize> {
        let lambda = params.lambda.unwrap_or(0.94);
        self.combos
            .iter()
            .position(|combo| (combo.lambda.unwrap_or(0.94) - lambda).abs() <= 1e-12)
    }

    pub fn values_for(&self, params: &EwmaVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.values.get(start..start + self.cols)
        })
    }
}

#[inline(always)]
fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, EwmaVolatilityError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(EwmaVolatilityError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }
    let step_abs = step.abs();
    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end + 1e-12 {
            out.push(x);
            x += step_abs;
        }
    } else {
        let mut x = start;
        while x >= end - 1e-12 {
            out.push(x);
            x -= step_abs;
        }
    }
    if out.is_empty() {
        return Err(EwmaVolatilityError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid(
    range: &EwmaVolatilityBatchRange,
) -> Result<Vec<EwmaVolatilityParams>, EwmaVolatilityError> {
    Ok(axis_f64(range.lambda)?
        .into_iter()
        .map(|lambda| EwmaVolatilityParams {
            lambda: Some(lambda),
        })
        .collect())
}

pub fn ewma_volatility_batch_with_kernel(
    data: &[f64],
    sweep: &EwmaVolatilityBatchRange,
    kernel: Kernel,
) -> Result<EwmaVolatilityBatchOutput, EwmaVolatilityError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(EwmaVolatilityError::InvalidKernelForBatch(kernel)),
    };
    ewma_volatility_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn ewma_volatility_batch_slice(
    data: &[f64],
    sweep: &EwmaVolatilityBatchRange,
    kernel: Kernel,
) -> Result<EwmaVolatilityBatchOutput, EwmaVolatilityError> {
    ewma_volatility_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn ewma_volatility_batch_par_slice(
    data: &[f64],
    sweep: &EwmaVolatilityBatchRange,
    kernel: Kernel,
) -> Result<EwmaVolatilityBatchOutput, EwmaVolatilityError> {
    ewma_volatility_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn ewma_volatility_batch_inner(
    data: &[f64],
    sweep: &EwmaVolatilityBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EwmaVolatilityBatchOutput, EwmaVolatilityError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(EwmaVolatilityError::EmptyInputData);
    }
    let prep = prepare_returns(data)?;
    let rows = combos.len();
    let cols = data.len();

    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            let period = period_from_lambda(combo.lambda.unwrap_or(0.94))?;
            prep.valid_indices
                .get(period.saturating_sub(1))
                .copied()
                .ok_or(EwmaVolatilityError::NotEnoughValidData {
                    needed: period,
                    valid: prep.valid_values.len(),
                })
        })
        .collect::<Result<_, _>>()?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);
    {
        let out: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(buf_mu.as_mut_ptr() as *mut f64, buf_mu.len())
        };
        ewma_volatility_batch_inner_into(data, sweep, kernel, parallel, out)?;
    }

    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(EwmaVolatilityBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn ewma_volatility_batch_into_slice(
    dst: &mut [f64],
    data: &[f64],
    sweep: &EwmaVolatilityBatchRange,
    kernel: Kernel,
) -> Result<(), EwmaVolatilityError> {
    ewma_volatility_batch_inner_into(data, sweep, kernel, false, dst)?;
    Ok(())
}

fn ewma_volatility_batch_inner_into(
    data: &[f64],
    sweep: &EwmaVolatilityBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EwmaVolatilityParams>, EwmaVolatilityError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(EwmaVolatilityError::EmptyInputData);
    }
    let prep = prepare_returns(data)?;
    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| EwmaVolatilityError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;
    if out.len() != expected {
        return Err(EwmaVolatilityError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };
    let _ = chosen;

    let periods: Vec<usize> = combos
        .iter()
        .map(|combo| period_from_lambda(combo.lambda.unwrap_or(0.94)))
        .collect::<Result<_, _>>()?;

    let warmups: Vec<usize> = periods
        .iter()
        .map(|&period| {
            prep.valid_indices
                .get(period.saturating_sub(1))
                .copied()
                .ok_or(EwmaVolatilityError::NotEnoughValidData {
                    needed: period,
                    valid: prep.valid_values.len(),
                })
        })
        .collect::<Result<_, _>>()?;

    for (row, &warm) in warmups.iter().enumerate() {
        let row_start = row * cols;
        out[row_start..row_start + warm.min(cols)].fill(f64::NAN);
    }

    let do_row = |row: usize, dst: &mut [f64]| -> Result<(), EwmaVolatilityError> {
        let period = periods[row];
        let alpha = alpha_from_period(period);
        fill_row_from_precomputed(&prep, period, alpha, dst)?;
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, dst)| do_row(row, dst))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                do_row(row, dst)?;
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            do_row(row, dst)?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ewma_volatility")]
#[pyo3(signature = (data, lambda_=0.94, kernel=None))]
pub fn ewma_volatility_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lambda_: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = EwmaVolatilityInput::from_slice(
        slice,
        EwmaVolatilityParams {
            lambda: Some(lambda_),
        },
    );
    let out = py
        .allow_threads(|| ewma_volatility_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EwmaVolatilityStream")]
pub struct EwmaVolatilityStreamPy {
    stream: EwmaVolatilityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EwmaVolatilityStreamPy {
    #[new]
    fn new(lambda_: Option<f64>) -> PyResult<Self> {
        let stream = EwmaVolatilityStream::try_new(EwmaVolatilityParams { lambda: lambda_ })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ewma_volatility_batch")]
#[pyo3(signature = (data, lambda_range=(0.94, 0.94, 0.0), kernel=None))]
pub fn ewma_volatility_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lambda_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = EwmaVolatilityBatchRange {
        lambda: lambda_range,
    };
    let out = py
        .allow_threads(|| ewma_volatility_batch_with_kernel(slice, &sweep, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        out.values
            .clone()
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "lambdas",
        out.combos
            .iter()
            .map(|combo| combo.lambda.unwrap_or(0.94))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ewma_volatility_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ewma_volatility_py, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_volatility_batch_py, m)?)?;
    m.add_class::<EwmaVolatilityStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ewma_volatility_js")]
pub fn ewma_volatility_js(data: &[f64], lambda_: f64) -> Result<Vec<f64>, JsValue> {
    let input = EwmaVolatilityInput::from_slice(
        data,
        EwmaVolatilityParams {
            lambda: Some(lambda_),
        },
    );
    let mut out = vec![0.0; data.len()];
    ewma_volatility_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EwmaVolatilityBatchConfig {
    pub lambda_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EwmaVolatilityBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EwmaVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ewma_volatility_batch_js")]
pub fn ewma_volatility_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: EwmaVolatilityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.lambda_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: lambda_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = ewma_volatility_batch_with_kernel(
        data,
        &EwmaVolatilityBatchRange {
            lambda: (
                config.lambda_range[0],
                config.lambda_range[1],
                config.lambda_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EwmaVolatilityBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ewma_volatility_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ewma_volatility_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ewma_volatility_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lambda_: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = EwmaVolatilityInput::from_slice(
            data,
            EwmaVolatilityParams {
                lambda: Some(lambda_),
            },
        );
        ewma_volatility_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ewma_volatility_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lambda_start: f64,
    lambda_end: f64,
    lambda_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ewma_volatility_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = EwmaVolatilityBatchRange {
            lambda: (lambda_start, lambda_end, lambda_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        ewma_volatility_batch_into_slice(out, data, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ewma_volatility_output_into_js(
    data: &[f64],
    lambda_: f64,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = ewma_volatility_js(data, lambda_)?;
    crate::write_wasm_f64_output("ewma_volatility_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ewma_volatility_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ewma_volatility_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ewma_volatility_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometric_series(len: usize, start: f64, ratio: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(len);
        let mut v = start;
        for _ in 0..len {
            out.push(v);
            v *= ratio;
        }
        out
    }

    fn assert_close_series(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            let a = lhs[i];
            let b = rhs[i];
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol,
                "mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn ewma_volatility_constant_return_series_converges_exactly() {
        let data = geometric_series(128, 100.0, 1.01);
        let input =
            EwmaVolatilityInput::from_slice(&data, EwmaVolatilityParams { lambda: Some(0.94) });
        let out = ewma_volatility(&input).unwrap();
        let expected = 1.01f64.ln().abs() * 100.0;
        let period = period_from_lambda(0.94).unwrap();
        for i in 0..period {
            assert!(out.values[i].is_nan(), "expected warmup NaN at {i}");
        }
        for v in &out.values[period..] {
            assert!((*v - expected).abs() <= 1e-12, "unexpected value {v}");
        }
    }

    #[test]
    fn ewma_volatility_stream_matches_batch() {
        let data = geometric_series(96, 50.0, 1.005);
        let batch = EwmaVolatilityBuilder::new()
            .lambda(0.90)
            .apply_slice(&data)
            .unwrap();
        let mut stream = EwmaVolatilityBuilder::new()
            .lambda(0.90)
            .into_stream()
            .unwrap();
        let stream_values: Vec<f64> = data
            .iter()
            .map(|&v| stream.update(v).unwrap_or(f64::NAN))
            .collect();
        assert_close_series(&batch.values, &stream_values, 1e-12);
    }

    #[test]
    fn ewma_volatility_batch_rows_match_single() {
        let data = geometric_series(128, 100.0, 1.002);
        let sweep = EwmaVolatilityBatchRange {
            lambda: (0.90, 0.94, 0.02),
        };
        let batch = ewma_volatility_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, data.len());

        for (row, &lambda) in [0.90, 0.92, 0.94].iter().enumerate() {
            let single = EwmaVolatilityBuilder::new()
                .lambda(lambda)
                .apply_slice(&data)
                .unwrap();
            let start = row * data.len();
            assert_close_series(
                &batch.values[start..start + data.len()],
                &single.values,
                1e-12,
            );
        }
    }

    #[test]
    fn ewma_volatility_into_slice_matches_single() {
        let data = geometric_series(80, 25.0, 1.003);
        let input =
            EwmaVolatilityInput::from_slice(&data, EwmaVolatilityParams { lambda: Some(0.94) });
        let direct = ewma_volatility(&input).unwrap();
        let mut out = vec![0.0; data.len()];
        ewma_volatility_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_close_series(&direct.values, &out, 1e-12);
    }

    #[test]
    fn ewma_volatility_invalid_lambda_errors() {
        let data = geometric_series(40, 10.0, 1.01);
        let err = EwmaVolatilityBuilder::new()
            .lambda(1.0)
            .apply_slice(&data)
            .unwrap_err();
        assert!(matches!(err, EwmaVolatilityError::InvalidLambda { .. }));
    }
}
