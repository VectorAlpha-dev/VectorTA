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
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_correlation_oscillator_output_into_js(
    data: &[f64],
    period: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = linear_correlation_oscillator_js(data, period)?;
    crate::write_wasm_f64_output("linear_correlation_oscillator_output_into_js", &values, out)
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for LinearCorrelationOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            LinearCorrelationOscillatorData::Slice(slice) => slice,
            LinearCorrelationOscillatorData::Candles { candles, source } => {
                linear_correlation_oscillator_source_type(candles, source)
            }
        }
    }
}

#[inline(always)]
fn linear_correlation_oscillator_source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "close" => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        "hl2" => &candles.hl2,
        "hlc3" => &candles.hlc3,
        "ohlc4" => &candles.ohlc4,
        "hlcc4" | "hlcc" => &candles.hlcc4,
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum LinearCorrelationOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinearCorrelationOscillatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct LinearCorrelationOscillatorParams {
    pub period: Option<usize>,
}

impl Default for LinearCorrelationOscillatorParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct LinearCorrelationOscillatorInput<'a> {
    pub data: LinearCorrelationOscillatorData<'a>,
    pub params: LinearCorrelationOscillatorParams,
}

impl<'a> LinearCorrelationOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: LinearCorrelationOscillatorParams,
    ) -> Self {
        Self {
            data: LinearCorrelationOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: LinearCorrelationOscillatorParams) -> Self {
        Self {
            data: LinearCorrelationOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            LinearCorrelationOscillatorParams::default(),
        )
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LinearCorrelationOscillatorBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for LinearCorrelationOscillatorBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LinearCorrelationOscillatorBuilder {
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
    ) -> Result<LinearCorrelationOscillatorOutput, LinearCorrelationOscillatorError> {
        let params = LinearCorrelationOscillatorParams {
            period: self.period,
        };
        let input = LinearCorrelationOscillatorInput::from_candles(candles, "close", params);
        linear_correlation_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LinearCorrelationOscillatorOutput, LinearCorrelationOscillatorError> {
        let params = LinearCorrelationOscillatorParams {
            period: self.period,
        };
        let input = LinearCorrelationOscillatorInput::from_slice(data, params);
        linear_correlation_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<LinearCorrelationOscillatorStream, LinearCorrelationOscillatorError> {
        let params = LinearCorrelationOscillatorParams {
            period: self.period,
        };
        LinearCorrelationOscillatorStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum LinearCorrelationOscillatorError {
    #[error("linear_correlation_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("linear_correlation_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "linear_correlation_oscillator: Invalid period: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },
    #[error(
        "linear_correlation_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "linear_correlation_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "linear_correlation_oscillator: Invalid range: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("linear_correlation_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn linear_correlation_oscillator(
    input: &LinearCorrelationOscillatorInput,
) -> Result<LinearCorrelationOscillatorOutput, LinearCorrelationOscillatorError> {
    linear_correlation_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => Kernel::Scalar,
    }
}

#[inline(always)]
fn linear_correlation_oscillator_prepare<'a>(
    input: &'a LinearCorrelationOscillatorInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), LinearCorrelationOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(LinearCorrelationOscillatorError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|v| !v.is_nan())
        .ok_or(LinearCorrelationOscillatorError::AllValuesNaN)?;

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(LinearCorrelationOscillatorError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let valid = data.len() - first;
    if valid <= period + 1 {
        return Err(LinearCorrelationOscillatorError::NotEnoughValidData {
            needed: period + 2,
            valid,
        });
    }

    Ok((data, period, first, normalize_single_kernel(kernel)))
}

#[inline(always)]
fn compute_correlation_from_sums(sum_y: f64, sum_y2: f64, weighted_sum: f64, period: usize) -> f64 {
    let period_f = period as f64;
    let inv_period = 1.0 / period_f;
    let mean_x = 0.5 * (period_f + 1.0);
    let var_x = ((period * period - 1) as f64) / 12.0;
    compute_correlation_from_precomputed(sum_y, sum_y2, weighted_sum, inv_period, mean_x, var_x)
}

#[inline(always)]
fn compute_correlation_from_precomputed(
    sum_y: f64,
    sum_y2: f64,
    weighted_sum: f64,
    inv_period: f64,
    mean_x: f64,
    var_x: f64,
) -> f64 {
    if var_x <= 0.0 {
        return f64::NAN;
    }

    let centered = weighted_sum - mean_x * sum_y;
    let mut var_y = sum_y2 * inv_period - (sum_y * inv_period).powi(2);
    if var_y < 0.0 && var_y > -1e-12 {
        var_y = 0.0;
    }
    if var_y <= 0.0 || !var_y.is_finite() {
        return f64::NAN;
    }

    let denom = (var_y * var_x).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        return f64::NAN;
    }

    let corr = centered * inv_period / denom;
    if corr.is_finite() {
        corr.clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn recompute_lco_window(data: &[f64], start: usize, end: usize) -> (f64, f64, f64, usize) {
    let mut sum_y = 0.0;
    let mut sum_y2 = 0.0;
    let mut weighted_sum = 0.0;
    let mut nan_count = 0usize;
    for (offset, &value) in data[start..=end].iter().enumerate() {
        if value.is_nan() {
            nan_count += 1;
        } else {
            let weight = (offset + 1) as f64;
            sum_y += value;
            sum_y2 += value * value;
            weighted_sum += weight * value;
        }
    }
    (sum_y, sum_y2, weighted_sum, nan_count)
}

#[inline(always)]
pub fn linear_correlation_oscillator_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let start = first + 2;
    let mut end = first + period + 1;
    if end >= data.len() {
        return;
    }

    let period_f = period as f64;
    let inv_period = 1.0 / period_f;
    let mean_x = 0.5 * (period_f + 1.0);
    let var_x = ((period * period - 1) as f64) / 12.0;
    let (mut sum_y, mut sum_y2, mut weighted_sum, mut nan_count) =
        recompute_lco_window(data, start, end);
    let mut window_start = start;
    loop {
        out[end] = if nan_count == 0 {
            compute_correlation_from_precomputed(
                sum_y,
                sum_y2,
                weighted_sum,
                inv_period,
                mean_x,
                var_x,
            )
        } else {
            f64::NAN
        };
        if end + 1 == data.len() {
            break;
        }

        let old = data[window_start];
        let new = data[end + 1];
        if nan_count == 0 && !new.is_nan() {
            weighted_sum = weighted_sum - sum_y + period_f * new;
            sum_y += new - old;
            sum_y2 += new * new - old * old;
        } else {
            if old.is_nan() {
                nan_count -= 1;
            }
            if new.is_nan() {
                nan_count += 1;
            }
            if nan_count == 0 {
                (sum_y, sum_y2, weighted_sum, _) =
                    recompute_lco_window(data, window_start + 1, end + 1);
            }
        }
        window_start += 1;
        end += 1;
    }
}

#[inline(always)]
fn linear_correlation_oscillator_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    match normalize_single_kernel(kernel) {
        Kernel::Scalar => linear_correlation_oscillator_scalar(data, period, first, out),
        _ => unreachable!(),
    }
}

pub fn linear_correlation_oscillator_with_kernel(
    input: &LinearCorrelationOscillatorInput,
    kernel: Kernel,
) -> Result<LinearCorrelationOscillatorOutput, LinearCorrelationOscillatorError> {
    let (data, period, first, chosen) = linear_correlation_oscillator_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + period + 1);
    linear_correlation_oscillator_compute_into(data, period, first, chosen, &mut out);
    Ok(LinearCorrelationOscillatorOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn linear_correlation_oscillator_into(
    input: &LinearCorrelationOscillatorInput,
    out: &mut [f64],
) -> Result<(), LinearCorrelationOscillatorError> {
    linear_correlation_oscillator_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn linear_correlation_oscillator_into_slice(
    dst: &mut [f64],
    input: &LinearCorrelationOscillatorInput,
    kernel: Kernel,
) -> Result<(), LinearCorrelationOscillatorError> {
    let (data, period, first, chosen) = linear_correlation_oscillator_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(LinearCorrelationOscillatorError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    for value in &mut dst[..(first + period + 1)] {
        *value = f64::NAN;
    }
    linear_correlation_oscillator_compute_into(data, period, first, chosen, dst);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct LinearCorrelationOscillatorStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    count: usize,
}

impl LinearCorrelationOscillatorStream {
    pub fn try_new(
        params: LinearCorrelationOscillatorParams,
    ) -> Result<Self, LinearCorrelationOscillatorError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(LinearCorrelationOscillatorError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let cap = period + 2;
        Ok(Self {
            period,
            buffer: vec![f64::NAN; cap],
            head: 0,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let cap = self.buffer.len();
        self.buffer[self.head] = value;
        self.head += 1;
        if self.head == cap {
            self.head = 0;
        }
        if self.count < cap {
            self.count += 1;
        }
        if self.count < cap {
            return None;
        }

        let mut sum_y = 0.0;
        let mut sum_y2 = 0.0;
        let mut weighted_sum = 0.0;
        for offset in 0..self.period {
            let idx = (self.head + 2 + offset) % cap;
            let current = self.buffer[idx];
            if current.is_nan() {
                return None;
            }
            let weight = (offset + 1) as f64;
            sum_y += current;
            sum_y2 += current * current;
            weighted_sum += weight * current;
        }

        Some(compute_correlation_from_sums(
            sum_y,
            sum_y2,
            weighted_sum,
            self.period,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct LinearCorrelationOscillatorBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for LinearCorrelationOscillatorBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 258, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinearCorrelationOscillatorBatchBuilder {
    range: LinearCorrelationOscillatorBatchRange,
    kernel: Kernel,
}

impl LinearCorrelationOscillatorBatchBuilder {
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

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LinearCorrelationOscillatorBatchOutput, LinearCorrelationOscillatorError> {
        linear_correlation_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<LinearCorrelationOscillatorBatchOutput, LinearCorrelationOscillatorError> {
        self.apply_slice(linear_correlation_oscillator_source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct LinearCorrelationOscillatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LinearCorrelationOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl LinearCorrelationOscillatorBatchOutput {
    pub fn row_for_params(&self, params: &LinearCorrelationOscillatorParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.period.unwrap_or(14) == params.period.unwrap_or(14))
    }

    pub fn values_for(&self, params: &LinearCorrelationOscillatorParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            let end = start.checked_add(self.cols)?;
            self.values.get(start..end)
        })
    }
}

#[inline(always)]
pub(crate) fn expand_grid(
    range: &LinearCorrelationOscillatorBatchRange,
) -> Result<Vec<LinearCorrelationOscillatorParams>, LinearCorrelationOscillatorError> {
    fn axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, LinearCorrelationOscillatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end {
                out.push(current);
                match current.checked_add(step) {
                    Some(next) if next > current => current = next,
                    _ => break,
                }
            }
        } else {
            let mut current = start;
            while current >= end {
                out.push(current);
                if current < end + step {
                    break;
                }
                current -= step;
                if current == 0 {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(LinearCorrelationOscillatorError::InvalidRange { start, end, step });
        }
        Ok(out)
    }

    Ok(axis(range.period)?
        .into_iter()
        .map(|period| LinearCorrelationOscillatorParams {
            period: Some(period),
        })
        .collect())
}

pub fn linear_correlation_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &LinearCorrelationOscillatorBatchRange,
    kernel: Kernel,
) -> Result<LinearCorrelationOscillatorBatchOutput, LinearCorrelationOscillatorError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(LinearCorrelationOscillatorError::InvalidKernelForBatch(
                other,
            ))
        }
    };

    let simd = match kernel {
        Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::Scalar,
        _ => unreachable!(),
    };
    linear_correlation_oscillator_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn linear_correlation_oscillator_batch_slice(
    data: &[f64],
    sweep: &LinearCorrelationOscillatorBatchRange,
    kernel: Kernel,
) -> Result<LinearCorrelationOscillatorBatchOutput, LinearCorrelationOscillatorError> {
    linear_correlation_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn linear_correlation_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &LinearCorrelationOscillatorBatchRange,
    kernel: Kernel,
) -> Result<LinearCorrelationOscillatorBatchOutput, LinearCorrelationOscillatorError> {
    linear_correlation_oscillator_batch_inner(data, sweep, kernel, true)
}

fn linear_correlation_oscillator_batch_inner(
    data: &[f64],
    sweep: &LinearCorrelationOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<LinearCorrelationOscillatorBatchOutput, LinearCorrelationOscillatorError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(LinearCorrelationOscillatorError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|v| !v.is_nan())
        .ok_or(LinearCorrelationOscillatorError::AllValuesNaN)?;
    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap())
        .max()
        .unwrap();
    let valid = data.len() - first;
    if valid <= max_period + 1 {
        return Err(LinearCorrelationOscillatorError::NotEnoughValidData {
            needed: max_period + 2,
            valid,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm_prefixes: Vec<usize> = combos
        .iter()
        .map(|params| first + params.period.unwrap() + 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm_prefixes);

    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        linear_correlation_oscillator_scalar(data, period, first, out_row);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, chunk)| do_row(row, chunk));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, chunk) in out.chunks_mut(cols).enumerate() {
                do_row(row, chunk);
            }
        }
    } else {
        for (row, chunk) in out.chunks_mut(cols).enumerate() {
            do_row(row, chunk);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(LinearCorrelationOscillatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn linear_correlation_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &LinearCorrelationOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<LinearCorrelationOscillatorParams>, LinearCorrelationOscillatorError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(LinearCorrelationOscillatorError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|v| !v.is_nan())
        .ok_or(LinearCorrelationOscillatorError::AllValuesNaN)?;
    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap())
        .max()
        .unwrap();
    let valid = data.len() - first;
    if valid <= max_period + 1 {
        return Err(LinearCorrelationOscillatorError::NotEnoughValidData {
            needed: max_period + 2,
            valid,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or(LinearCorrelationOscillatorError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            })?;
    if out.len() != expected {
        return Err(LinearCorrelationOscillatorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm_prefixes: Vec<usize> = combos
        .iter()
        .map(|params| first + params.period.unwrap() + 1)
        .collect();
    init_matrix_prefixes(out_mu, cols, &warm_prefixes);

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let period = combos[row].period.unwrap();
        let dst: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        match kernel {
            Kernel::Scalar => linear_correlation_oscillator_scalar(data, period, first, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_mu)| do_row(row, row_mu));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, row_mu);
            }
        }
    } else {
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "linear_correlation_oscillator")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn linear_correlation_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = LinearCorrelationOscillatorInput::from_slice(
        slice_in,
        LinearCorrelationOscillatorParams {
            period: Some(period),
        },
    );
    let values = py
        .allow_threads(|| linear_correlation_oscillator_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "linear_correlation_oscillator_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn linear_correlation_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = LinearCorrelationOscillatorBatchRange {
        period: period_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            let simd = match kernel {
                Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::Scalar,
                _ => unreachable!(),
            };
            linear_correlation_oscillator_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|params| params.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "LinearCorrelationOscillatorStream")]
pub struct LinearCorrelationOscillatorStreamPy {
    inner: LinearCorrelationOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LinearCorrelationOscillatorStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let inner = LinearCorrelationOscillatorStream::try_new(LinearCorrelationOscillatorParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct LinearCorrelationOscillatorBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct LinearCorrelationOscillatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LinearCorrelationOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_correlation_oscillator_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let input = LinearCorrelationOscillatorInput::from_slice(
        data,
        LinearCorrelationOscillatorParams {
            period: Some(period),
        },
    );
    let mut out = vec![0.0; data.len()];
    linear_correlation_oscillator_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_correlation_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_correlation_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_correlation_oscillator_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = LinearCorrelationOscillatorInput::from_slice(
            data,
            LinearCorrelationOscillatorParams {
                period: Some(period),
            },
        );
        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            linear_correlation_oscillator_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            linear_correlation_oscillator_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_correlation_oscillator_batch(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: LinearCorrelationOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = LinearCorrelationOscillatorBatchRange {
        period: config.period_range,
    };
    let result = linear_correlation_oscillator_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = LinearCorrelationOscillatorBatchJsOutput {
        values: result.values,
        combos: result.combos,
        rows: result.rows,
        cols: result.cols,
    };
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn pine_literal_reference(data: &[f64], period: usize) -> Vec<f64> {
        let mut cmla = vec![f64::NAN; data.len()];
        let mut cmlb = vec![f64::NAN; data.len()];
        let mut cmlc = vec![f64::NAN; data.len()];

        for (idx, &value) in data.iter().enumerate() {
            let prev_a = if idx > 0 && !cmla[idx - 1].is_nan() {
                cmla[idx - 1]
            } else {
                0.0
            };
            let prev_b = if idx > 0 && !cmlb[idx - 1].is_nan() {
                cmlb[idx - 1]
            } else {
                0.0
            };
            let prev_c = if idx > 0 && !cmlc[idx - 1].is_nan() {
                cmlc[idx - 1]
            } else {
                0.0
            };
            cmla[idx] = if value.is_nan() {
                f64::NAN
            } else {
                prev_a + value
            };
            cmlb[idx] = if cmla[idx].is_nan() {
                f64::NAN
            } else {
                prev_b + cmla[idx]
            };
            cmlc[idx] = if value.is_nan() {
                f64::NAN
            } else {
                prev_c + value * value
            };
        }

        let mut sum = vec![f64::NAN; data.len()];
        let mut out = vec![f64::NAN; data.len()];
        let var_x = ((period * period - 1) as f64) / 12.0;
        for idx in 0..data.len() {
            if idx >= period && !cmlb[idx].is_nan() && !cmlb[idx - period].is_nan() {
                sum[idx] = cmlb[idx] - cmlb[idx - period];
            }
            if idx > period
                && !cmla[idx].is_nan()
                && !cmla[idx - period].is_nan()
                && !cmlc[idx].is_nan()
                && !cmlc[idx - period].is_nan()
                && !sum[idx - 1].is_nan()
            {
                let a = (period as f64) * cmla[idx] - sum[idx - 1];
                let b = cmla[idx] - cmla[idx - period];
                let c = cmlc[idx] - cmlc[idx - period];
                let num = (a - b * ((period + 1) as f64) * 0.5) / (period as f64);
                let vary = c / (period as f64) - (b / (period as f64)).powi(2);
                out[idx] = if vary <= 0.0 {
                    f64::NAN
                } else {
                    num / (vary * var_x).sqrt()
                };
            }
        }
        out
    }

    fn assert_series_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() || e.is_nan() {
                assert!(
                    a.is_nan() && e.is_nan(),
                    "NaN mismatch at idx {}: actual={} expected={}",
                    idx,
                    a,
                    e
                );
            } else {
                assert!(
                    (a - e).abs() <= tol,
                    "value mismatch at idx {}: actual={} expected={} tol={}",
                    idx,
                    a,
                    e,
                    tol
                );
            }
        }
    }

    fn sample_data() -> Vec<f64> {
        vec![
            12.0, 14.0, 13.0, 15.0, 16.5, 18.0, 17.0, 19.5, 21.0, 20.5, 22.5, 24.0, 23.0, 25.5,
            26.0, 28.0,
        ]
    }

    fn check_literal_parity(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(5) },
        );
        let result = linear_correlation_oscillator_with_kernel(&input, kernel)?;
        let expected = pine_literal_reference(&data, 5);
        assert_series_close(&result.values, &expected, 1e-12);
        Ok(())
    }

    fn check_increasing_series(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data: Vec<f64> = (0..32).map(|idx| idx as f64).collect();
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(6) },
        );
        let result = linear_correlation_oscillator_with_kernel(&input, kernel)?;
        assert!(result.values[..7].iter().all(|v| v.is_nan()));
        for &value in &result.values[7..] {
            assert!((value - 1.0).abs() <= 1e-12, "expected +1, got {}", value);
        }
        Ok(())
    }

    fn check_decreasing_series(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data: Vec<f64> = (0..32).map(|idx| -(idx as f64)).collect();
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(6) },
        );
        let result = linear_correlation_oscillator_with_kernel(&input, kernel)?;
        assert!(result.values[..7].iter().all(|v| v.is_nan()));
        for &value in &result.values[7..] {
            assert!((value + 1.0).abs() <= 1e-12, "expected -1, got {}", value);
        }
        Ok(())
    }

    fn check_constant_series_returns_nan(
        _name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn StdError>> {
        let data = vec![5.0; 32];
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(5) },
        );
        let result = linear_correlation_oscillator_with_kernel(&input, kernel)?;
        assert!(result.values[..6].iter().all(|v| v.is_nan()));
        assert!(result.values[6..].iter().all(|v| v.is_nan()));
        Ok(())
    }

    fn check_nan_prefix_semantics(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = vec![f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(3) },
        );
        let result = linear_correlation_oscillator_with_kernel(&input, kernel)?;
        assert!(result.values[..6].iter().all(|v| v.is_nan()));
        assert!((result.values[6] - 1.0).abs() <= 1e-12);
        Ok(())
    }

    fn check_into_matches_api(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(5) },
        );
        let baseline = linear_correlation_oscillator_with_kernel(&input, kernel)?.values;
        let mut out = vec![0.0; data.len()];
        linear_correlation_oscillator_into_slice(&mut out, &input, kernel)?;
        assert_series_close(&out, &baseline, 1e-12);
        Ok(())
    }

    fn check_stream_matches_batch(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(5) },
        );
        let batch = linear_correlation_oscillator_with_kernel(&input, kernel)?.values;
        let mut stream =
            LinearCorrelationOscillatorStream::try_new(LinearCorrelationOscillatorParams {
                period: Some(5),
            })?;
        let streamed: Vec<f64> = data
            .iter()
            .map(|&value| stream.update(value).unwrap_or(f64::NAN))
            .collect();
        assert_series_close(&streamed, &batch, 1e-12);
        Ok(())
    }

    fn check_batch_matches_single(_name: &str, _kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let batch = LinearCorrelationOscillatorBatchBuilder::new()
            .period_range(5, 7, 1)
            .kernel(Kernel::ScalarBatch)
            .apply_slice(&data)?;
        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, data.len());

        for period in [5usize, 6, 7] {
            let input = LinearCorrelationOscillatorInput::from_slice(
                &data,
                LinearCorrelationOscillatorParams {
                    period: Some(period),
                },
            );
            let single = linear_correlation_oscillator(&input)?.values;
            let row = batch
                .values_for(&LinearCorrelationOscillatorParams {
                    period: Some(period),
                })
                .unwrap();
            assert_series_close(row, &single, 1e-12);
        }
        Ok(())
    }

    fn check_invalid_period(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(0) },
        );
        let err = linear_correlation_oscillator_with_kernel(&input, kernel).unwrap_err();
        assert!(matches!(
            err,
            LinearCorrelationOscillatorError::InvalidPeriod { .. }
        ));
        Ok(())
    }

    fn check_all_nan(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = vec![f64::NAN; 16];
        let input = LinearCorrelationOscillatorInput::from_slice(
            &data,
            LinearCorrelationOscillatorParams { period: Some(5) },
        );
        let err = linear_correlation_oscillator_with_kernel(&input, kernel).unwrap_err();
        assert!(matches!(
            err,
            LinearCorrelationOscillatorError::AllValuesNaN
        ));
        Ok(())
    }

    macro_rules! generate_lco_tests {
        ($($name:ident),* $(,)?) => {
            $(
                paste! {
                    #[test]
                    fn [<linear_correlation_oscillator_ $name _scalar>]() -> Result<(), Box<dyn StdError>> {
                        $name("scalar", Kernel::Scalar)
                    }

                    #[test]
                    fn [<linear_correlation_oscillator_ $name _auto>]() -> Result<(), Box<dyn StdError>> {
                        $name("auto", Kernel::Auto)
                    }
                }
            )*
        };
    }

    generate_lco_tests!(
        check_literal_parity,
        check_increasing_series,
        check_decreasing_series,
        check_constant_series_returns_nan,
        check_nan_prefix_semantics,
        check_into_matches_api,
        check_stream_matches_batch,
        check_batch_matches_single,
        check_invalid_period,
        check_all_nan,
    );

    #[test]
    fn linear_correlation_oscillator_default_candles_smoke() -> Result<(), Box<dyn StdError>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let input = LinearCorrelationOscillatorInput::with_default_candles(&candles);
        let output = linear_correlation_oscillator(&input)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
}
