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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for TrendDirectionForceIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrendDirectionForceIndexData::Candles { candles, source } => {
                source_type(candles, source)
            }
            TrendDirectionForceIndexData::Slice(slice) => slice,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrendDirectionForceIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrendDirectionForceIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TrendDirectionForceIndexParams {
    pub length: Option<usize>,
}

impl Default for TrendDirectionForceIndexParams {
    fn default() -> Self {
        Self { length: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct TrendDirectionForceIndexInput<'a> {
    pub data: TrendDirectionForceIndexData<'a>,
    pub params: TrendDirectionForceIndexParams,
}

impl<'a> TrendDirectionForceIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: TrendDirectionForceIndexParams,
    ) -> Self {
        Self {
            data: TrendDirectionForceIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: TrendDirectionForceIndexParams) -> Self {
        Self {
            data: TrendDirectionForceIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", TrendDirectionForceIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrendDirectionForceIndexBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for TrendDirectionForceIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrendDirectionForceIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
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
    ) -> Result<TrendDirectionForceIndexOutput, TrendDirectionForceIndexError> {
        let input = TrendDirectionForceIndexInput::from_candles(
            candles,
            "close",
            TrendDirectionForceIndexParams {
                length: self.length,
            },
        );
        trend_direction_force_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<TrendDirectionForceIndexOutput, TrendDirectionForceIndexError> {
        let input = TrendDirectionForceIndexInput::from_slice(
            data,
            TrendDirectionForceIndexParams {
                length: self.length,
            },
        );
        trend_direction_force_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<TrendDirectionForceIndexStream, TrendDirectionForceIndexError> {
        TrendDirectionForceIndexStream::try_new(TrendDirectionForceIndexParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum TrendDirectionForceIndexError {
    #[error("trend_direction_force_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("trend_direction_force_index: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "trend_direction_force_index: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "trend_direction_force_index: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "trend_direction_force_index: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("trend_direction_force_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("trend_direction_force_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn first_valid_index(data: &[f64]) -> Option<usize> {
    data.iter().position(|x| x.is_finite())
}

#[inline(always)]
fn is_fast_path_clean(data: &[f64], first: usize) -> bool {
    data[first..].iter().all(|x| x.is_finite())
}

#[inline(always)]
fn half_length(length: usize) -> usize {
    (length / 2).max(1)
}

#[inline(always)]
fn alpha_from_half_length(half_len: usize) -> f64 {
    2.0 / (half_len as f64 + 1.0)
}

#[inline(always)]
fn normalization_window(length: usize) -> usize {
    length.saturating_mul(3).max(1)
}

#[inline(always)]
fn trend_direction_force_index_prepare<'a>(
    input: &'a TrendDirectionForceIndexInput,
) -> Result<(&'a [f64], usize, usize), TrendDirectionForceIndexError> {
    let data = input.as_ref();
    let data_len = data.len();
    if data_len == 0 {
        return Err(TrendDirectionForceIndexError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(TrendDirectionForceIndexError::AllValuesNaN)?;
    let length = input.get_length();
    if length == 0 {
        return Err(TrendDirectionForceIndexError::InvalidLength { length, data_len });
    }

    let valid = data_len - first;
    if valid < 2 {
        return Err(TrendDirectionForceIndexError::NotEnoughValidData { needed: 2, valid });
    }

    Ok((data, length, first))
}

#[inline(always)]
fn push_max_q(max_q: &mut VecDeque<(f64, usize)>, idx: usize, value: f64, window: usize) {
    let min_idx = idx.saturating_add(1).saturating_sub(window);
    while let Some(&(_, old_idx)) = max_q.front() {
        if old_idx < min_idx {
            max_q.pop_front();
        } else {
            break;
        }
    }
    while let Some(&(back, _)) = max_q.back() {
        if back <= value {
            max_q.pop_back();
        } else {
            break;
        }
    }
    max_q.push_back((value, idx));
}

#[inline(always)]
fn trend_direction_force_index_compute_fast(
    data: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    let half_len = half_length(length);
    let alpha = alpha_from_half_length(half_len);
    let one_minus_alpha = 1.0 - alpha;
    let window = normalization_window(length);
    let mut ema1 = data[first] * 1000.0;
    let mut ema2 = ema1;
    let mut max_q = VecDeque::with_capacity(window.min(data.len().saturating_sub(first)));
    let mut tdf_idx = 0usize;

    for i in (first + 1)..data.len() {
        let prev_ema1 = ema1;
        let prev_ema2 = ema2;
        let scaled = data[i] * 1000.0;
        ema1 = alpha * scaled + one_minus_alpha * prev_ema1;
        ema2 = alpha * ema1 + one_minus_alpha * prev_ema2;
        let ema_diff_avg = ((ema1 - prev_ema1) + (ema2 - prev_ema2)) * 0.5;
        let tdf = (ema1 - ema2).abs() * ema_diff_avg.powi(3);
        push_max_q(&mut max_q, tdf_idx, tdf.abs(), window);
        let denom = max_q.front().map(|entry| entry.0).unwrap_or(0.0);
        out[i] = if denom != 0.0 { tdf / denom } else { 0.0 };
        tdf_idx += 1;
    }
}

#[inline(always)]
fn trend_direction_force_index_compute_fallback(
    data: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    let mut stream = TrendDirectionForceIndexStream::from_length(length);
    for i in first..data.len() {
        out[i] = stream.update_reset_on_nan(data[i]).unwrap_or(f64::NAN);
    }
}

#[inline(always)]
fn trend_direction_force_index_compute_into(
    data: &[f64],
    length: usize,
    first: usize,
    _kernel: Kernel,
    out: &mut [f64],
) {
    if is_fast_path_clean(data, first) {
        trend_direction_force_index_compute_fast(data, length, first, out);
    } else {
        trend_direction_force_index_compute_fallback(data, length, first, out);
    }
}

#[inline]
pub fn trend_direction_force_index(
    input: &TrendDirectionForceIndexInput,
) -> Result<TrendDirectionForceIndexOutput, TrendDirectionForceIndexError> {
    trend_direction_force_index_with_kernel(input, Kernel::Auto)
}

pub fn trend_direction_force_index_with_kernel(
    input: &TrendDirectionForceIndexInput,
    kernel: Kernel,
) -> Result<TrendDirectionForceIndexOutput, TrendDirectionForceIndexError> {
    let (data, length, first) = trend_direction_force_index_prepare(input)?;
    let mut out = alloc_with_nan_prefix(data.len(), first.saturating_add(1));
    trend_direction_force_index_compute_into(data, length, first, kernel, &mut out);
    Ok(TrendDirectionForceIndexOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn trend_direction_force_index_into(
    input: &TrendDirectionForceIndexInput,
    out: &mut [f64],
) -> Result<(), TrendDirectionForceIndexError> {
    trend_direction_force_index_into_slice(out, input, Kernel::Auto)
}

pub fn trend_direction_force_index_into_slice(
    out: &mut [f64],
    input: &TrendDirectionForceIndexInput,
    kernel: Kernel,
) -> Result<(), TrendDirectionForceIndexError> {
    let (data, length, first) = trend_direction_force_index_prepare(input)?;
    if out.len() != data.len() {
        return Err(TrendDirectionForceIndexError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    out.fill(f64::NAN);
    trend_direction_force_index_compute_into(data, length, first, kernel, out);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct TrendDirectionForceIndexStream {
    length: usize,
    alpha: f64,
    one_minus_alpha: f64,
    window: usize,
    seeded: bool,
    ema1: f64,
    ema2: f64,
    tdf_index: usize,
    max_q: VecDeque<(f64, usize)>,
}

impl TrendDirectionForceIndexStream {
    #[inline]
    fn from_length(length: usize) -> Self {
        let half_len = half_length(length);
        let window = normalization_window(length);
        let alpha = alpha_from_half_length(half_len);
        Self {
            length,
            alpha,
            one_minus_alpha: 1.0 - alpha,
            window,
            seeded: false,
            ema1: 0.0,
            ema2: 0.0,
            tdf_index: 0,
            max_q: VecDeque::with_capacity(window),
        }
    }

    #[inline]
    pub fn try_new(
        params: TrendDirectionForceIndexParams,
    ) -> Result<Self, TrendDirectionForceIndexError> {
        let length = params.length.unwrap_or(10);
        if length == 0 {
            return Err(TrendDirectionForceIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self::from_length(length))
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.seeded = false;
        self.ema1 = 0.0;
        self.ema2 = 0.0;
        self.tdf_index = 0;
        self.max_q.clear();
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }

        let scaled = value * 1000.0;
        if !self.seeded {
            self.seeded = true;
            self.ema1 = scaled;
            self.ema2 = scaled;
            return None;
        }

        let prev_ema1 = self.ema1;
        let prev_ema2 = self.ema2;
        self.ema1 = self.alpha * scaled + self.one_minus_alpha * prev_ema1;
        self.ema2 = self.alpha * self.ema1 + self.one_minus_alpha * prev_ema2;

        let ema_diff_avg = ((self.ema1 - prev_ema1) + (self.ema2 - prev_ema2)) * 0.5;
        let tdf = (self.ema1 - self.ema2).abs() * ema_diff_avg.powi(3);
        push_max_q(&mut self.max_q, self.tdf_index, tdf.abs(), self.window);
        self.tdf_index += 1;
        let denom = self.max_q.front().map(|entry| entry.0).unwrap_or(0.0);
        Some(if denom != 0.0 { tdf / denom } else { 0.0 })
    }

    #[inline(always)]
    pub fn update_reset_on_nan(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.update(value)
    }
}

#[derive(Clone, Debug)]
pub struct TrendDirectionForceIndexBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for TrendDirectionForceIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (10, 200, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrendDirectionForceIndexBatchBuilder {
    range: TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
}

impl TrendDirectionForceIndexBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
        trend_direction_force_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct TrendDirectionForceIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrendDirectionForceIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendDirectionForceIndexBatchOutput {
    pub fn row_for_params(&self, params: &TrendDirectionForceIndexParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.length.unwrap_or(10) == params.length.unwrap_or(10))
    }

    pub fn values_for(&self, params: &TrendDirectionForceIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, TrendDirectionForceIndexError> {
    let (start, end, step) = range;
    if start < 1 || end < 1 {
        return Err(TrendDirectionForceIndexError::InvalidRange { start, end, step });
    }
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) if next > value => value = next,
                _ => break,
            }
        }
    } else {
        let mut value = start;
        while value >= end {
            out.push(value);
            if value < end + step {
                break;
            }
            value = value.saturating_sub(step);
            if value == 0 {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(TrendDirectionForceIndexError::InvalidRange { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_trend_direction_force_index(
    sweep: &TrendDirectionForceIndexBatchRange,
) -> Result<Vec<TrendDirectionForceIndexParams>, TrendDirectionForceIndexError> {
    Ok(axis_usize(sweep.length)?
        .into_iter()
        .map(|length| TrendDirectionForceIndexParams {
            length: Some(length),
        })
        .collect())
}

pub fn trend_direction_force_index_batch_with_kernel(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(TrendDirectionForceIndexError::InvalidKernelForBatch(other)),
    };
    trend_direction_force_index_batch_impl(data, sweep, batch_kernel.to_non_batch(), true)
}

pub fn trend_direction_force_index_batch_slice(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    trend_direction_force_index_batch_impl(data, sweep, Kernel::Scalar, false)
}

pub fn trend_direction_force_index_batch_par_slice(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    trend_direction_force_index_batch_impl(data, sweep, Kernel::Scalar, true)
}

fn trend_direction_force_index_batch_impl(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    let combos = expand_grid_trend_direction_force_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(TrendDirectionForceIndexError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(TrendDirectionForceIndexError::AllValuesNaN)?;
    let valid = cols - first;
    if valid < 2 {
        return Err(TrendDirectionForceIndexError::NotEnoughValidData { needed: 2, valid });
    }

    let mut matrix = make_uninit_matrix(rows, cols);
    let warmups = vec![first.saturating_add(1); rows];
    init_matrix_prefixes(&mut matrix, cols, &warmups);

    let mut guard = ManuallyDrop::new(matrix);
    let out_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let length = combos[row].length.unwrap_or(10);
        let dst = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        trend_direction_force_index_compute_into(data, length, first, kernel, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_mu)| do_row(row, row_mu));
        #[cfg(target_arch = "wasm32")]
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    } else {
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(TrendDirectionForceIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn trend_direction_force_index_batch_inner_into(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), TrendDirectionForceIndexError> {
    let combos = expand_grid_trend_direction_force_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if rows.checked_mul(cols) != Some(out.len()) {
        return Err(TrendDirectionForceIndexError::OutputLengthMismatch {
            expected: rows * cols,
            got: out.len(),
        });
    }

    let first = first_valid_index(data).ok_or(TrendDirectionForceIndexError::AllValuesNaN)?;
    let valid = cols - first;
    if valid < 2 {
        return Err(TrendDirectionForceIndexError::NotEnoughValidData { needed: 2, valid });
    }

    for row_out in out.chunks_mut(cols) {
        row_out.fill(f64::NAN);
    }

    let do_row = |row: usize, row_out: &mut [f64]| {
        let length = combos[row].length.unwrap_or(10);
        trend_direction_force_index_compute_into(data, length, first, kernel, row_out);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_out)| do_row(row, row_out));
        #[cfg(target_arch = "wasm32")]
        for (row, row_out) in out.chunks_mut(cols).enumerate() {
            do_row(row, row_out);
        }
    } else {
        for (row, row_out) in out.chunks_mut(cols).enumerate() {
            do_row(row, row_out);
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_direction_force_index")]
#[pyo3(signature = (data, length=10, kernel=None))]
pub fn trend_direction_force_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = TrendDirectionForceIndexInput::from_slice(
        data,
        TrendDirectionForceIndexParams {
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| trend_direction_force_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrendDirectionForceIndexStream")]
pub struct TrendDirectionForceIndexStreamPy {
    stream: TrendDirectionForceIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrendDirectionForceIndexStreamPy {
    #[new]
    #[pyo3(signature = (length=10))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = TrendDirectionForceIndexStream::try_new(TrendDirectionForceIndexParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update_reset_on_nan(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_direction_force_index_batch")]
#[pyo3(signature = (data, length_range, kernel=None))]
pub fn trend_direction_force_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = TrendDirectionForceIndexBatchRange {
        length: length_range,
    };
    let combos = expand_grid_trend_direction_force_index(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out = unsafe { arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        trend_direction_force_index_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(10) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_trend_direction_force_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trend_direction_force_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(trend_direction_force_index_batch_py, m)?)?;
    m.add_class::<TrendDirectionForceIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendDirectionForceIndexBatchConfig {
    length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendDirectionForceIndexBatchJsOutput {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<TrendDirectionForceIndexParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_direction_force_index_js")]
pub fn trend_direction_force_index_js(data: &[f64], length: usize) -> Result<Vec<f64>, JsValue> {
    let input = TrendDirectionForceIndexInput::from_slice(
        data,
        TrendDirectionForceIndexParams {
            length: Some(length),
        },
    );
    let mut out = vec![0.0; data.len()];
    trend_direction_force_index_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_direction_force_index_batch_js")]
pub fn trend_direction_force_index_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: TrendDirectionForceIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = TrendDirectionForceIndexBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let batch = trend_direction_force_index_batch_slice(data, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&TrendDirectionForceIndexBatchJsOutput {
        values: batch.values,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_direction_force_index_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = TrendDirectionForceIndexInput::from_slice(
            data,
            TrendDirectionForceIndexParams {
                length: Some(length),
            },
        );
        trend_direction_force_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_direction_force_index_into_host")]
pub fn trend_direction_force_index_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_direction_force_index_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len());
        let input = TrendDirectionForceIndexInput::from_slice(
            data,
            TrendDirectionForceIndexParams {
                length: Some(length),
            },
        );
        trend_direction_force_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_direction_force_index_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = TrendDirectionForceIndexBatchRange {
            length: (length_start, length_end, length_step),
        };
        let combos = expand_grid_trend_direction_force_index(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        trend_direction_force_index_batch_inner_into(data, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };

    fn sample_data(len: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let trend = 100.0 + i as f64 * 0.07;
            let wave = (i as f64 * 0.19).sin() * 2.8 + (i as f64 * 0.041).cos() * 1.3;
            out.push(trend + wave);
        }
        out
    }

    fn naive_tdfi(data: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        if data.is_empty() || length == 0 {
            return out;
        }
        let Some(first) = first_valid_index(data) else {
            return out;
        };
        if data.len() - first < 2 {
            return out;
        }

        let half_len = half_length(length);
        let alpha = alpha_from_half_length(half_len);
        let one_minus_alpha = 1.0 - alpha;
        let window = normalization_window(length);
        let mut ema1 = data[first] * 1000.0;
        let mut ema2 = ema1;
        let mut tdfs = Vec::with_capacity(data.len().saturating_sub(first + 1));

        for i in (first + 1)..data.len() {
            let prev_ema1 = ema1;
            let prev_ema2 = ema2;
            let scaled = data[i] * 1000.0;
            ema1 = alpha * scaled + one_minus_alpha * prev_ema1;
            ema2 = alpha * ema1 + one_minus_alpha * prev_ema2;
            let ema_diff_avg = ((ema1 - prev_ema1) + (ema2 - prev_ema2)) * 0.5;
            let tdf = (ema1 - ema2).abs() * ema_diff_avg.powi(3);
            tdfs.push(tdf);
            let start = tdfs.len().saturating_sub(window);
            let tdfh = tdfs[start..]
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            out[i] = if tdfh != 0.0 { tdf / tdfh } else { 0.0 };
        }
        out
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(
                    a[i].is_nan() && b[i].is_nan(),
                    "nan mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            } else {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-10,
                    "mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn trend_direction_force_index_matches_naive() {
        let data = sample_data(256);
        let input = TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(10) },
        );
        let out = trend_direction_force_index(&input).expect("indicator");
        let reference = naive_tdfi(&data, 10);
        assert_close(&out.values, &reference);
    }

    #[test]
    fn trend_direction_force_index_into_matches_api() {
        let data = sample_data(192);
        let input = TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(13) },
        );
        let baseline = trend_direction_force_index(&input).expect("baseline");
        let mut out = vec![0.0; data.len()];
        trend_direction_force_index_into(&input, &mut out).expect("into");
        assert_close(&baseline.values, &out);
    }

    #[test]
    fn trend_direction_force_index_stream_matches_batch() {
        let data = sample_data(192);
        let batch = trend_direction_force_index(&TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(10) },
        ))
        .expect("batch");
        let mut stream = TrendDirectionForceIndexStream::try_new(TrendDirectionForceIndexParams {
            length: Some(10),
        })
        .expect("stream");
        let mut values = Vec::with_capacity(data.len());
        for &value in &data {
            values.push(stream.update(value).unwrap_or(f64::NAN));
        }
        assert_close(&batch.values, &values);
    }

    #[test]
    fn trend_direction_force_index_batch_single_param_matches_single() {
        let data = sample_data(192);
        let sweep = TrendDirectionForceIndexBatchRange {
            length: (10, 10, 0),
        };
        let batch =
            trend_direction_force_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
                .expect("batch");
        let single = trend_direction_force_index(&TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(10) },
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.values, &single.values);
    }

    #[test]
    fn trend_direction_force_index_rejects_invalid_length() {
        let data = sample_data(32);
        let err = trend_direction_force_index(&TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(0) },
        ))
        .expect_err("invalid length");
        assert!(matches!(
            err,
            TrendDirectionForceIndexError::InvalidLength { .. }
        ));
    }

    #[test]
    fn trend_direction_force_index_dispatch_matches_direct() {
        let data = sample_data(192);
        let params = [ParamKV {
            key: "length",
            value: ParamValue::Int(10),
        }];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "trend_direction_force_index",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = trend_direction_force_index(&TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(10) },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert_close(out.values_f64.as_ref().expect("values"), &direct.values);
    }
}
