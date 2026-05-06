#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

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
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EhlersFmDemodulatorData<'a> {
    Candles {
        candles: &'a Candles,
        open_source: &'a str,
        close_source: &'a str,
    },
    Slices {
        open: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EhlersFmDemodulatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersFmDemodulatorParams {
    pub period: Option<usize>,
}

impl Default for EhlersFmDemodulatorParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersFmDemodulatorInput<'a> {
    pub data: EhlersFmDemodulatorData<'a>,
    pub params: EhlersFmDemodulatorParams,
}

impl<'a> EhlersFmDemodulatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        open_source: &'a str,
        close_source: &'a str,
        params: EhlersFmDemodulatorParams,
    ) -> Self {
        Self {
            data: EhlersFmDemodulatorData::Candles {
                candles,
                open_source,
                close_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        close: &'a [f64],
        params: EhlersFmDemodulatorParams,
    ) -> Self {
        Self {
            data: EhlersFmDemodulatorData::Slices { open, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "open",
            "close",
            EhlersFmDemodulatorParams::default(),
        )
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersFmDemodulatorBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EhlersFmDemodulatorBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersFmDemodulatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, period: usize) -> Self {
        self.period = Some(period);
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
    ) -> Result<EhlersFmDemodulatorOutput, EhlersFmDemodulatorError> {
        let input = EhlersFmDemodulatorInput::from_candles(
            candles,
            "open",
            "close",
            EhlersFmDemodulatorParams {
                period: self.period,
            },
        );
        ehlers_fm_demodulator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<EhlersFmDemodulatorOutput, EhlersFmDemodulatorError> {
        let input = EhlersFmDemodulatorInput::from_slices(
            open,
            close,
            EhlersFmDemodulatorParams {
                period: self.period,
            },
        );
        ehlers_fm_demodulator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersFmDemodulatorStream, EhlersFmDemodulatorError> {
        EhlersFmDemodulatorStream::try_new(EhlersFmDemodulatorParams {
            period: self.period,
        })
    }
}

#[derive(Debug, Error)]
pub enum EhlersFmDemodulatorError {
    #[error("ehlers_fm_demodulator: input data slice is empty.")]
    EmptyInputData,
    #[error(
        "ehlers_fm_demodulator: open/close length mismatch: open = {open_len}, close = {close_len}"
    )]
    DataLengthMismatch { open_len: usize, close_len: usize },
    #[error("ehlers_fm_demodulator: all values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_fm_demodulator: invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ehlers_fm_demodulator: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ehlers_fm_demodulator: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ehlers_fm_demodulator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("ehlers_fm_demodulator: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
}

#[derive(Copy, Clone, Debug)]
struct Coefficients {
    c1: f64,
    c2: f64,
    c3: f64,
}

#[derive(Copy, Clone, Debug)]
struct PreparedInput<'a> {
    open: &'a [f64],
    close: &'a [f64],
    len: usize,
    first_valid: usize,
    period: usize,
}

fn normalize_single_kernel(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn coefficients(period: usize) -> Coefficients {
    let period_f = period as f64;
    let a1 = (-1.414 * PI / period_f).exp();
    let b1 = 2.0 * a1 * (1.414 * PI / period_f).cos();
    let c2 = b1;
    let c3 = -(a1 * a1);
    let c1 = 1.0 - c2 - c3;
    Coefficients { c1, c2, c3 }
}

#[inline(always)]
fn warmup_prefix_len(first_valid: usize, period: usize) -> usize {
    first_valid.saturating_add(period.saturating_sub(3))
}

#[inline(always)]
fn minimum_valid_length(period: usize) -> usize {
    period.saturating_sub(2).max(1)
}

#[inline(always)]
fn split_input<'a>(data: &'a EhlersFmDemodulatorData<'a>) -> (&'a [f64], &'a [f64]) {
    match data {
        EhlersFmDemodulatorData::Candles {
            candles,
            open_source,
            close_source,
        } => (
            source_type(candles, open_source),
            source_type(candles, close_source),
        ),
        EhlersFmDemodulatorData::Slices { open, close } => (open, close),
    }
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a EhlersFmDemodulatorInput<'a>,
    kernel: Kernel,
) -> Result<(PreparedInput<'a>, Kernel), EhlersFmDemodulatorError> {
    let (open, close) = split_input(&input.data);
    if open.is_empty() || close.is_empty() {
        return Err(EhlersFmDemodulatorError::EmptyInputData);
    }
    if open.len() != close.len() {
        return Err(EhlersFmDemodulatorError::DataLengthMismatch {
            open_len: open.len(),
            close_len: close.len(),
        });
    }

    let len = open.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EhlersFmDemodulatorError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid = (0..len)
        .find(|&i| !open[i].is_nan() && !close[i].is_nan())
        .ok_or(EhlersFmDemodulatorError::AllValuesNaN)?;

    let valid = len - first_valid;
    let needed = minimum_valid_length(period);
    if valid < needed {
        return Err(EhlersFmDemodulatorError::NotEnoughValidData { needed, valid });
    }

    Ok((
        PreparedInput {
            open,
            close,
            len,
            first_valid,
            period,
        },
        normalize_single_kernel(kernel),
    ))
}

#[inline(always)]
fn compute_scalar_into(prepared: PreparedInput<'_>, out: &mut [f64]) {
    let coeffs = coefficients(prepared.period);
    let warmup_bars = prepared.period.saturating_sub(3);
    let mut prev_hl = 0.0;
    let mut ss1 = 0.0;
    let mut ss2 = 0.0;
    let mut valid_count = 0usize;

    for i in prepared.first_valid..prepared.len {
        let open = prepared.open[i];
        let close = prepared.close[i];
        if open.is_nan() || close.is_nan() {
            out[i] = f64::NAN;
            prev_hl = 0.0;
            ss1 = 0.0;
            ss2 = 0.0;
            valid_count = 0;
            continue;
        }

        let derivative = close - open;
        let hl = (10.0 * derivative).clamp(-1.0, 1.0);
        let value = if valid_count < 3 {
            derivative
        } else {
            coeffs.c1 * (hl + prev_hl) * 0.5 + coeffs.c2 * ss1 + coeffs.c3 * ss2
        };

        prev_hl = hl;
        ss2 = ss1;
        ss1 = value;
        valid_count += 1;

        out[i] = if valid_count > warmup_bars {
            value
        } else {
            f64::NAN
        };
    }
}

#[inline]
pub fn ehlers_fm_demodulator(
    input: &EhlersFmDemodulatorInput<'_>,
) -> Result<EhlersFmDemodulatorOutput, EhlersFmDemodulatorError> {
    ehlers_fm_demodulator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn ehlers_fm_demodulator_with_kernel(
    input: &EhlersFmDemodulatorInput<'_>,
    kernel: Kernel,
) -> Result<EhlersFmDemodulatorOutput, EhlersFmDemodulatorError> {
    let (prepared, _) = prepare_input(input, kernel)?;
    let mut out = alloc_with_nan_prefix(prepared.len, prepared.first_valid);
    compute_scalar_into(prepared, &mut out);
    Ok(EhlersFmDemodulatorOutput { values: out })
}

#[inline]
pub fn ehlers_fm_demodulator_into_slice(
    dst: &mut [f64],
    input: &EhlersFmDemodulatorInput<'_>,
    kernel: Kernel,
) -> Result<(), EhlersFmDemodulatorError> {
    let (prepared, _) = prepare_input(input, kernel)?;
    if dst.len() != prepared.len {
        return Err(EhlersFmDemodulatorError::OutputLengthMismatch {
            expected: prepared.len,
            got: dst.len(),
        });
    }

    for value in &mut dst[..prepared.first_valid] {
        *value = f64::NAN;
    }
    compute_scalar_into(prepared, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ehlers_fm_demodulator_into(
    input: &EhlersFmDemodulatorInput<'_>,
    out: &mut [f64],
) -> Result<(), EhlersFmDemodulatorError> {
    ehlers_fm_demodulator_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct EhlersFmDemodulatorBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EhlersFmDemodulatorBatchRange {
    fn default() -> Self {
        Self {
            period: (30, 254, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EhlersFmDemodulatorBatchBuilder {
    range: EhlersFmDemodulatorBatchRange,
    kernel: Kernel,
}

impl EhlersFmDemodulatorBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    pub fn period_static(mut self, period: usize) -> Self {
        self.range.period = (period, period, 0);
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorError> {
        ehlers_fm_demodulator_batch_with_kernel(open, close, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        open_source: &str,
        close_source: &str,
    ) -> Result<EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorError> {
        let open = source_type(candles, open_source);
        let close = source_type(candles, close_source);
        self.apply_slices(open, close)
    }

    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorError> {
        Self::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles, "open", "close")
    }
}

#[derive(Clone, Debug)]
pub struct EhlersFmDemodulatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersFmDemodulatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersFmDemodulatorBatchOutput {
    pub fn row_for_params(&self, params: &EhlersFmDemodulatorParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.period.unwrap_or(30) == params.period.unwrap_or(30))
    }

    pub fn values_for(&self, params: &EhlersFmDemodulatorParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(
    range: &EhlersFmDemodulatorBatchRange,
) -> Result<Vec<EhlersFmDemodulatorParams>, EhlersFmDemodulatorError> {
    fn axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, EhlersFmDemodulatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut values = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                values.push(cur);
                let next = cur.saturating_add(step);
                if next == cur {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            while cur >= end {
                values.push(cur);
                let next = cur.saturating_sub(step);
                if next == cur {
                    break;
                }
                cur = next;
                if cur == 0 && end > 0 {
                    break;
                }
            }
        }

        if values.is_empty() {
            return Err(EhlersFmDemodulatorError::InvalidRange { start, end, step });
        }
        Ok(values)
    }

    Ok(axis(range.period)?
        .into_iter()
        .map(|period| EhlersFmDemodulatorParams {
            period: Some(period),
        })
        .collect())
}

#[inline(always)]
fn batch_prepare(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersFmDemodulatorBatchRange,
) -> Result<(Vec<EhlersFmDemodulatorParams>, usize, usize), EhlersFmDemodulatorError> {
    if open.is_empty() || close.is_empty() {
        return Err(EhlersFmDemodulatorError::EmptyInputData);
    }
    if open.len() != close.len() {
        return Err(EhlersFmDemodulatorError::DataLengthMismatch {
            open_len: open.len(),
            close_len: close.len(),
        });
    }

    let combos = expand_grid(sweep)?;
    let len = open.len();
    let first_valid = (0..len)
        .find(|&i| !open[i].is_nan() && !close[i].is_nan())
        .ok_or(EhlersFmDemodulatorError::AllValuesNaN)?;
    let valid = len - first_valid;
    let max_period = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(30))
        .max()
        .unwrap_or(30);
    let needed = minimum_valid_length(max_period);
    if valid < needed {
        return Err(EhlersFmDemodulatorError::NotEnoughValidData { needed, valid });
    }

    Ok((combos, len, first_valid))
}

#[inline]
pub fn ehlers_fm_demodulator_batch_with_kernel(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersFmDemodulatorBatchRange,
    kernel: Kernel,
) -> Result<EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        other if other.is_batch() => {
            let _ = other;
        }
        _ => return Err(EhlersFmDemodulatorError::InvalidKernelForBatch(kernel)),
    }
    ehlers_fm_demodulator_batch_par_slice(open, close, sweep, Kernel::Scalar)
}

#[inline(always)]
pub fn ehlers_fm_demodulator_batch_slice(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersFmDemodulatorBatchRange,
    kernel: Kernel,
) -> Result<EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorError> {
    ehlers_fm_demodulator_batch_inner(open, close, sweep, kernel, false)
}

#[inline(always)]
pub fn ehlers_fm_demodulator_batch_par_slice(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersFmDemodulatorBatchRange,
    kernel: Kernel,
) -> Result<EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorError> {
    ehlers_fm_demodulator_batch_inner(open, close, sweep, kernel, true)
}

#[inline(always)]
fn ehlers_fm_demodulator_batch_inner(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersFmDemodulatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorError> {
    let _ = kernel;
    let (combos, cols, first_valid) = batch_prepare(open, close, sweep)?;
    let rows = combos.len();
    let total = rows * cols;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| warmup_prefix_len(first_valid, combo.period.unwrap_or(30)))
        .collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    ehlers_fm_demodulator_batch_inner_into(open, close, sweep, Kernel::Scalar, parallel, out)?;

    let values =
        unsafe { Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, total, guard.capacity()) };

    Ok(EhlersFmDemodulatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ehlers_fm_demodulator_batch_inner_into(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersFmDemodulatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EhlersFmDemodulatorParams>, EhlersFmDemodulatorError> {
    let (combos, cols, first_valid) = batch_prepare(open, close, sweep)?;
    let expected = combos.len() * cols;
    if out.len() != expected {
        return Err(EhlersFmDemodulatorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let dst: &mut [f64] = unsafe {
            std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };
        dst.fill(f64::NAN);
        let combo = &combos[row];
        let prepared = PreparedInput {
            open,
            close,
            len: cols,
            first_valid,
            period: combo.period.unwrap_or(30),
        };
        compute_scalar_into(prepared, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[derive(Debug, Clone)]
pub struct EhlersFmDemodulatorStream {
    period: usize,
    warmup_bars: usize,
    coeffs: Coefficients,
    valid_count: usize,
    prev_hl: f64,
    ss1: f64,
    ss2: f64,
}

impl EhlersFmDemodulatorStream {
    #[inline(always)]
    pub fn try_new(params: EhlersFmDemodulatorParams) -> Result<Self, EhlersFmDemodulatorError> {
        let period = params.period.unwrap_or(30);
        if period == 0 {
            return Err(EhlersFmDemodulatorError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            warmup_bars: period.saturating_sub(3),
            coeffs: coefficients(period),
            valid_count: 0,
            prev_hl: 0.0,
            ss1: 0.0,
            ss2: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, close: f64) -> Option<f64> {
        if open.is_nan() || close.is_nan() {
            self.reset();
            return None;
        }

        let derivative = close - open;
        let hl = (10.0 * derivative).clamp(-1.0, 1.0);
        let value = if self.valid_count < 3 {
            derivative
        } else {
            self.coeffs.c1 * (hl + self.prev_hl) * 0.5
                + self.coeffs.c2 * self.ss1
                + self.coeffs.c3 * self.ss2
        };

        self.prev_hl = hl;
        self.ss2 = self.ss1;
        self.ss1 = value;
        self.valid_count += 1;

        if self.valid_count > self.warmup_bars {
            Some(value)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.valid_count = 0;
        self.prev_hl = 0.0;
        self.ss1 = 0.0;
        self.ss2 = 0.0;
    }

    #[inline(always)]
    pub fn period(&self) -> usize {
        self.period
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_fm_demodulator")]
#[pyo3(signature = (open, close, period=30, kernel=None))]
pub fn ehlers_fm_demodulator_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let open_slice = open.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = EhlersFmDemodulatorInput::from_slices(
        open_slice,
        close_slice,
        EhlersFmDemodulatorParams {
            period: Some(period),
        },
    );

    let result = py
        .allow_threads(|| ehlers_fm_demodulator_with_kernel(&input, kernel).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersFmDemodulatorStream")]
pub struct EhlersFmDemodulatorStreamPy {
    stream: EhlersFmDemodulatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersFmDemodulatorStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let stream = EhlersFmDemodulatorStream::try_new(EhlersFmDemodulatorParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    pub fn update(&mut self, open: f64, close: f64) -> Option<f64> {
        self.stream.update(open, close)
    }

    pub fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_fm_demodulator_batch")]
#[pyo3(signature = (open, close, period_range, kernel=None))]
pub fn ehlers_fm_demodulator_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open_slice = open.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = EhlersFmDemodulatorBatchRange {
        period: period_range,
    };
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let output = py
        .allow_threads(|| {
            ehlers_fm_demodulator_batch_with_kernel(open_slice, close_slice, &sweep, kernel)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let values = output.values.into_pyarray(py);
    let dict = PyDict::new(py);
    dict.set_item("values", values.reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(30) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_fm_demodulator_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ehlers_fm_demodulator_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_fm_demodulator_batch_py, m)?)?;
    m.add_class::<EhlersFmDemodulatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn ehlers_fm_demodulator_into_native(
    dst: &mut [f64],
    open: &[f64],
    close: &[f64],
    period: usize,
) -> Result<(), EhlersFmDemodulatorError> {
    let input = EhlersFmDemodulatorInput::from_slices(
        open,
        close,
        EhlersFmDemodulatorParams {
            period: Some(period),
        },
    );
    ehlers_fm_demodulator_into_slice(dst, &input, Kernel::Auto)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_fm_demodulator_js(
    open: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut out = vec![0.0; open.len()];
    ehlers_fm_demodulator_into_native(&mut out, open, close, period)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_fm_demodulator_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        if open_ptr == out_ptr || close_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            ehlers_fm_demodulator_into_native(&mut temp, open, close, period)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ehlers_fm_demodulator_into_native(out, open, close, period)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_fm_demodulator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_fm_demodulator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersFmDemodulatorBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersFmDemodulatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersFmDemodulatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_fm_demodulator_batch)]
pub fn ehlers_fm_demodulator_batch_unified_js(
    open: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersFmDemodulatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = EhlersFmDemodulatorBatchRange {
        period: config.period_range,
    };
    let output = ehlers_fm_demodulator_batch_with_kernel(open, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&EhlersFmDemodulatorBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_fm_demodulator_batch_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = EhlersFmDemodulatorBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("size overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        ehlers_fm_demodulator_batch_inner_into(open, close, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_fm_demodulator_output_into_js(
    open: &[f64],
    close: &[f64],
    period: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = ehlers_fm_demodulator_js(open, close, period)?;
    crate::write_wasm_f64_output("ehlers_fm_demodulator_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_fm_demodulator_batch_unified_output_into_js(
    open: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_fm_demodulator_batch_unified_js(open, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_fm_demodulator_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_open() -> Vec<f64> {
        vec![1.0; 8]
    }

    fn sample_close() -> Vec<f64> {
        vec![1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85]
    }

    #[test]
    fn ehlers_fm_demodulator_small_reference_case() {
        let open = vec![1.0; 6];
        let close = vec![1.1, 0.9, 1.2, 0.8, 1.05, 0.95];
        let input = EhlersFmDemodulatorInput::from_slices(
            &open,
            &close,
            EhlersFmDemodulatorParams { period: Some(3) },
        );
        let out = ehlers_fm_demodulator(&input).unwrap().values;
        let expected = [
            0.10000000000000009,
            -0.09999999999999998,
            0.19999999999999996,
            0.013357467332142794,
            -0.26250860145491506,
            -0.011431966667873657,
        ];

        for (actual, expected) in out.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
        }
    }

    #[test]
    fn ehlers_fm_demodulator_uses_period_minus_three_warmup() {
        let open = sample_open();
        let close = sample_close();
        let input = EhlersFmDemodulatorInput::from_slices(
            &open,
            &close,
            EhlersFmDemodulatorParams { period: Some(5) },
        );
        let out = ehlers_fm_demodulator(&input).unwrap().values;
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!(!out[2].is_nan());
    }

    #[test]
    fn ehlers_fm_demodulator_into_matches_owned_api() {
        let open = sample_open();
        let close = sample_close();
        let input = EhlersFmDemodulatorInput::from_slices(
            &open,
            &close,
            EhlersFmDemodulatorParams { period: Some(5) },
        );
        let baseline = ehlers_fm_demodulator(&input).unwrap().values;
        let mut out = vec![0.0; baseline.len()];
        ehlers_fm_demodulator_into(&input, &mut out).unwrap();

        for (a, b) in baseline.iter().zip(out.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (a == b), "{a} != {b}");
        }
    }

    #[test]
    fn ehlers_fm_demodulator_stream_matches_batch() {
        let open = sample_open();
        let close = sample_close();
        let input = EhlersFmDemodulatorInput::from_slices(
            &open,
            &close,
            EhlersFmDemodulatorParams { period: Some(5) },
        );
        let batch = ehlers_fm_demodulator(&input).unwrap().values;

        let mut stream =
            EhlersFmDemodulatorStream::try_new(EhlersFmDemodulatorParams { period: Some(5) })
                .unwrap();
        let streamed: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| stream.update(o, c).unwrap_or(f64::NAN))
            .collect();

        for (a, b) in batch.iter().zip(streamed.iter()) {
            assert!((a.is_nan() && b.is_nan()) || ((a - b).abs() < 1e-12));
        }
    }

    #[test]
    fn ehlers_fm_demodulator_batch_first_row_matches_single() {
        let open = sample_open();
        let close = sample_close();
        let single = ehlers_fm_demodulator(&EhlersFmDemodulatorInput::from_slices(
            &open,
            &close,
            EhlersFmDemodulatorParams { period: Some(5) },
        ))
        .unwrap()
        .values;

        let batch = ehlers_fm_demodulator_batch_with_kernel(
            &open,
            &close,
            &EhlersFmDemodulatorBatchRange { period: (5, 7, 1) },
            Kernel::Auto,
        )
        .unwrap();

        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, open.len());
        let first_row = &batch.values[..batch.cols];
        for (a, b) in first_row.iter().zip(single.iter()) {
            assert!((a.is_nan() && b.is_nan()) || ((a - b).abs() < 1e-12));
        }
    }

    #[test]
    fn ehlers_fm_demodulator_rejects_mismatched_lengths() {
        let err = ehlers_fm_demodulator(&EhlersFmDemodulatorInput::from_slices(
            &[1.0, 1.0, 1.0],
            &[1.1, 1.1],
            EhlersFmDemodulatorParams { period: Some(3) },
        ))
        .unwrap_err();

        assert!(matches!(
            err,
            EhlersFmDemodulatorError::DataLengthMismatch { .. }
        ));
    }

    #[test]
    fn ehlers_fm_demodulator_handles_leading_nans() {
        let open = vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0, 1.0];
        let close = vec![f64::NAN, f64::NAN, 1.1, 0.9, 1.2, 0.8];
        let out = ehlers_fm_demodulator(&EhlersFmDemodulatorInput::from_slices(
            &open,
            &close,
            EhlersFmDemodulatorParams { period: Some(3) },
        ))
        .unwrap()
        .values;

        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!(!out[2].is_nan());
        assert!(!out[5].is_nan());
    }
}
