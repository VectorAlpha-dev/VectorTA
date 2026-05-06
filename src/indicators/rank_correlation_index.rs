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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_LENGTH: usize = 12;

impl<'a> AsRef<[f64]> for RankCorrelationIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RankCorrelationIndexData::Candles { candles, source } => match *source {
                "open" => candles.open.as_slice(),
                "high" => candles.high.as_slice(),
                "low" => candles.low.as_slice(),
                "close" => candles.close.as_slice(),
                "volume" => candles.volume.as_slice(),
                "hl2" => candles.hl2.as_slice(),
                "hlc3" => candles.hlc3.as_slice(),
                "ohlc4" => candles.ohlc4.as_slice(),
                "hlcc4" => candles.hlcc4.as_slice(),
                _ => source_type(candles, source),
            },
            RankCorrelationIndexData::Slice(slice) => slice,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RankCorrelationIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RankCorrelationIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RankCorrelationIndexParams {
    pub length: Option<usize>,
}

impl Default for RankCorrelationIndexParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RankCorrelationIndexInput<'a> {
    pub data: RankCorrelationIndexData<'a>,
    pub params: RankCorrelationIndexParams,
}

impl<'a> RankCorrelationIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: RankCorrelationIndexParams,
    ) -> Self {
        Self {
            data: RankCorrelationIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: RankCorrelationIndexParams) -> Self {
        Self {
            data: RankCorrelationIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", RankCorrelationIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RankCorrelationIndexBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for RankCorrelationIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RankCorrelationIndexBuilder {
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
    ) -> Result<RankCorrelationIndexOutput, RankCorrelationIndexError> {
        let input = RankCorrelationIndexInput::from_candles(
            candles,
            "close",
            RankCorrelationIndexParams {
                length: self.length,
            },
        );
        rank_correlation_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<RankCorrelationIndexOutput, RankCorrelationIndexError> {
        let input = RankCorrelationIndexInput::from_slice(
            data,
            RankCorrelationIndexParams {
                length: self.length,
            },
        );
        rank_correlation_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RankCorrelationIndexStream, RankCorrelationIndexError> {
        RankCorrelationIndexStream::try_new(RankCorrelationIndexParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum RankCorrelationIndexError {
    #[error("rank_correlation_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("rank_correlation_index: All values are NaN.")]
    AllValuesNaN,
    #[error("rank_correlation_index: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("rank_correlation_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rank_correlation_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("rank_correlation_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("rank_correlation_index: Invalid kernel for batch: {0:?}")]
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
fn rank_correlation_index_prepare<'a>(
    input: &'a RankCorrelationIndexInput,
) -> Result<(&'a [f64], usize, usize, bool), RankCorrelationIndexError> {
    let data = input.as_ref();
    let data_len = data.len();
    if data_len == 0 {
        return Err(RankCorrelationIndexError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(RankCorrelationIndexError::AllValuesNaN)?;
    let length = input.get_length();
    if length < 2 || length > data_len {
        return Err(RankCorrelationIndexError::InvalidLength { length, data_len });
    }

    let valid = data_len - first;
    if valid < length {
        return Err(RankCorrelationIndexError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    Ok((data, length, first, is_fast_path_clean(data, first)))
}

#[inline(always)]
fn compute_window_rci(window: &[f64]) -> f64 {
    let len = window.len();
    let len_f = len as f64;
    let denom = len_f * ((len * len - 1) as f64);
    let mut sum = 0.0;

    for c in 0..len {
        let p = window[c];
        let mut o = 1.0;
        let mut s = 0.0;
        for &other in window {
            if p < other {
                o += 1.0;
            } else if p == other {
                s += 1.0;
            }
        }
        let ord = o + (s - 1.0) * 0.5;
        let time_rank = (len - c) as f64;
        let diff = time_rank - ord;
        sum += diff * diff;
    }

    (1.0 - 6.0 * sum / denom) * 100.0
}

#[inline(always)]
fn compute_window_rci_12(data: &[f64], start: usize) -> f64 {
    let values = [
        data[start],
        data[start + 1],
        data[start + 2],
        data[start + 3],
        data[start + 4],
        data[start + 5],
        data[start + 6],
        data[start + 7],
        data[start + 8],
        data[start + 9],
        data[start + 10],
        data[start + 11],
    ];
    let mut ranks = [1.0; DEFAULT_LENGTH];

    for i in 0..DEFAULT_LENGTH {
        let a = values[i];
        for j in i + 1..DEFAULT_LENGTH {
            let b = values[j];
            if a < b {
                ranks[i] += 1.0;
            } else if a > b {
                ranks[j] += 1.0;
            } else {
                ranks[i] += 0.5;
                ranks[j] += 0.5;
            }
        }
    }

    let mut sum = 0.0;
    for c in 0..DEFAULT_LENGTH {
        let time_rank = (DEFAULT_LENGTH - c) as f64;
        let diff = time_rank - ranks[c];
        sum += diff * diff;
    }

    (1.0 - 6.0 * sum / 1716.0) * 100.0
}

#[inline(always)]
fn rank_correlation_index_compute_fast_default(data: &[f64], first: usize, out: &mut [f64]) {
    let warmup = first + DEFAULT_LENGTH - 1;
    for i in warmup..data.len() {
        out[i] = compute_window_rci_12(data, i + 1 - DEFAULT_LENGTH);
    }
}

#[inline(always)]
fn rank_correlation_index_compute_fast(data: &[f64], length: usize, first: usize, out: &mut [f64]) {
    if length == DEFAULT_LENGTH {
        rank_correlation_index_compute_fast_default(data, first, out);
        return;
    }

    let warmup = first + length - 1;
    for i in warmup..data.len() {
        out[i] = compute_window_rci(&data[i + 1 - length..=i]);
    }
}

#[inline(always)]
fn rank_correlation_index_compute_fallback(
    data: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    let mut stream = RankCorrelationIndexStream::from_length(length);
    for i in first..data.len() {
        out[i] = stream.update_reset_on_nan(data[i]).unwrap_or(f64::NAN);
    }
}

#[inline(always)]
fn rank_correlation_index_compute_into(
    data: &[f64],
    length: usize,
    first: usize,
    clean: bool,
    _kernel: Kernel,
    out: &mut [f64],
) {
    if clean {
        rank_correlation_index_compute_fast(data, length, first, out);
    } else {
        rank_correlation_index_compute_fallback(data, length, first, out);
    }
}

#[inline]
pub fn rank_correlation_index(
    input: &RankCorrelationIndexInput,
) -> Result<RankCorrelationIndexOutput, RankCorrelationIndexError> {
    rank_correlation_index_with_kernel(input, Kernel::Auto)
}

pub fn rank_correlation_index_with_kernel(
    input: &RankCorrelationIndexInput,
    kernel: Kernel,
) -> Result<RankCorrelationIndexOutput, RankCorrelationIndexError> {
    let (data, length, first, clean) = rank_correlation_index_prepare(input)?;
    let warmup = first + length - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warmup);
    rank_correlation_index_compute_into(data, length, first, clean, kernel, &mut out);
    Ok(RankCorrelationIndexOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn rank_correlation_index_into(
    input: &RankCorrelationIndexInput,
    out: &mut [f64],
) -> Result<(), RankCorrelationIndexError> {
    rank_correlation_index_into_slice(out, input, Kernel::Auto)
}

pub fn rank_correlation_index_into_slice(
    out: &mut [f64],
    input: &RankCorrelationIndexInput,
    kernel: Kernel,
) -> Result<(), RankCorrelationIndexError> {
    let (data, length, first, clean) = rank_correlation_index_prepare(input)?;
    if out.len() != data.len() {
        return Err(RankCorrelationIndexError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    if clean {
        out[..first + length - 1].fill(f64::NAN);
    } else {
        out.fill(f64::NAN);
    }
    rank_correlation_index_compute_into(data, length, first, clean, kernel, out);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct RankCorrelationIndexStream {
    length: usize,
    head: usize,
    filled: usize,
    buffer: Vec<f64>,
}

impl RankCorrelationIndexStream {
    #[inline]
    fn from_length(length: usize) -> Self {
        Self {
            length,
            head: 0,
            filled: 0,
            buffer: vec![0.0; length.max(1)],
        }
    }

    #[inline]
    pub fn try_new(params: RankCorrelationIndexParams) -> Result<Self, RankCorrelationIndexError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length < 2 {
            return Err(RankCorrelationIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self::from_length(length))
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        self.buffer.fill(0.0);
    }

    #[inline(always)]
    fn compute_from_ring(&self) -> f64 {
        let len = self.length;
        let len_f = len as f64;
        let denom = len_f * ((len * len - 1) as f64);
        let mut sum = 0.0;

        for c in 0..len {
            let p = self.buffer[(self.head + c) % len];
            let mut o = 1.0;
            let mut s = 0.0;
            for j in 0..len {
                let other = self.buffer[(self.head + j) % len];
                if p < other {
                    o += 1.0;
                } else if p == other {
                    s += 1.0;
                }
            }
            let ord = o + (s - 1.0) * 0.5;
            let time_rank = (len - c) as f64;
            let diff = time_rank - ord;
            sum += diff * diff;
        }

        (1.0 - 6.0 * sum / denom) * 100.0
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }

        if self.filled < self.length {
            self.buffer[self.filled] = value;
            self.filled += 1;
            if self.filled < self.length {
                return None;
            }
            self.head = 0;
            return Some(self.compute_from_ring());
        }

        self.buffer[self.head] = value;
        self.head += 1;
        if self.head == self.length {
            self.head = 0;
        }
        Some(self.compute_from_ring())
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
pub struct RankCorrelationIndexBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for RankCorrelationIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, 200, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RankCorrelationIndexBatchBuilder {
    range: RankCorrelationIndexBatchRange,
    kernel: Kernel,
}

impl RankCorrelationIndexBatchBuilder {
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
    ) -> Result<RankCorrelationIndexBatchOutput, RankCorrelationIndexError> {
        rank_correlation_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<RankCorrelationIndexBatchOutput, RankCorrelationIndexError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct RankCorrelationIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<RankCorrelationIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RankCorrelationIndexBatchOutput {
    pub fn row_for_params(&self, params: &RankCorrelationIndexParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == params.length.unwrap_or(DEFAULT_LENGTH)
        })
    }

    pub fn values_for(&self, params: &RankCorrelationIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, RankCorrelationIndexError> {
    let (start, end, step) = range;
    if start < 2 || end < 2 {
        return Err(RankCorrelationIndexError::InvalidRange { start, end, step });
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
        return Err(RankCorrelationIndexError::InvalidRange { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_rank_correlation_index(
    sweep: &RankCorrelationIndexBatchRange,
) -> Result<Vec<RankCorrelationIndexParams>, RankCorrelationIndexError> {
    Ok(axis_usize(sweep.length)?
        .into_iter()
        .map(|length| RankCorrelationIndexParams {
            length: Some(length),
        })
        .collect())
}

pub fn rank_correlation_index_batch_with_kernel(
    data: &[f64],
    sweep: &RankCorrelationIndexBatchRange,
    kernel: Kernel,
) -> Result<RankCorrelationIndexBatchOutput, RankCorrelationIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(RankCorrelationIndexError::InvalidKernelForBatch(other)),
    };
    rank_correlation_index_batch_impl(data, sweep, batch_kernel.to_non_batch(), true)
}

pub fn rank_correlation_index_batch_slice(
    data: &[f64],
    sweep: &RankCorrelationIndexBatchRange,
) -> Result<RankCorrelationIndexBatchOutput, RankCorrelationIndexError> {
    rank_correlation_index_batch_impl(data, sweep, Kernel::Scalar, false)
}

pub fn rank_correlation_index_batch_par_slice(
    data: &[f64],
    sweep: &RankCorrelationIndexBatchRange,
) -> Result<RankCorrelationIndexBatchOutput, RankCorrelationIndexError> {
    rank_correlation_index_batch_impl(data, sweep, Kernel::Scalar, true)
}

fn rank_correlation_index_batch_impl(
    data: &[f64],
    sweep: &RankCorrelationIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<RankCorrelationIndexBatchOutput, RankCorrelationIndexError> {
    let combos = expand_grid_rank_correlation_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    if cols == 0 {
        return Err(RankCorrelationIndexError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(RankCorrelationIndexError::AllValuesNaN)?;
    let clean = is_fast_path_clean(data, first);
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(DEFAULT_LENGTH);
    let valid = cols - first;
    if valid < max_length {
        return Err(RankCorrelationIndexError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let mut matrix = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| first + params.length.unwrap_or(DEFAULT_LENGTH) - 1)
        .collect();
    init_matrix_prefixes(&mut matrix, cols, &warmups);

    let mut guard = ManuallyDrop::new(matrix);
    let out_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let length = combos[row].length.unwrap_or(DEFAULT_LENGTH);
        let dst = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        rank_correlation_index_compute_into(data, length, first, clean, kernel, dst);
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

    Ok(RankCorrelationIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn rank_correlation_index_batch_inner_into(
    data: &[f64],
    sweep: &RankCorrelationIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), RankCorrelationIndexError> {
    let combos = expand_grid_rank_correlation_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if rows.checked_mul(cols) != Some(out.len()) {
        return Err(RankCorrelationIndexError::OutputLengthMismatch {
            expected: rows * cols,
            got: out.len(),
        });
    }

    let first = first_valid_index(data).ok_or(RankCorrelationIndexError::AllValuesNaN)?;
    let clean = is_fast_path_clean(data, first);
    for (row, params) in combos.iter().enumerate() {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let row_out = &mut out[row * cols..(row + 1) * cols];
        if clean {
            row_out[..first + length - 1].fill(f64::NAN);
        } else {
            row_out.fill(f64::NAN);
        }
        if cols - first < length {
            return Err(RankCorrelationIndexError::NotEnoughValidData {
                needed: length,
                valid: cols - first,
            });
        }
    }

    let do_row = |row: usize, row_out: &mut [f64]| {
        let length = combos[row].length.unwrap_or(DEFAULT_LENGTH);
        rank_correlation_index_compute_into(data, length, first, clean, kernel, row_out);
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
#[pyfunction(name = "rank_correlation_index")]
#[pyo3(signature = (data, length=12, kernel=None))]
pub fn rank_correlation_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = RankCorrelationIndexInput::from_slice(
        data,
        RankCorrelationIndexParams {
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| rank_correlation_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "RankCorrelationIndexStream")]
pub struct RankCorrelationIndexStreamPy {
    stream: RankCorrelationIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RankCorrelationIndexStreamPy {
    #[new]
    #[pyo3(signature = (length=12))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = RankCorrelationIndexStream::try_new(RankCorrelationIndexParams {
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
#[pyfunction(name = "rank_correlation_index_batch")]
#[pyo3(signature = (data, length_range, kernel=None))]
pub fn rank_correlation_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = RankCorrelationIndexBatchRange {
        length: length_range,
    };
    let combos = expand_grid_rank_correlation_index(&sweep)
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
        rank_correlation_index_batch_inner_into(
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
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_rank_correlation_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank_correlation_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(rank_correlation_index_batch_py, m)?)?;
    m.add_class::<RankCorrelationIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RankCorrelationIndexBatchConfig {
    length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RankCorrelationIndexBatchJsOutput {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<RankCorrelationIndexParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "rank_correlation_index_js")]
pub fn rank_correlation_index_js(data: &[f64], length: usize) -> Result<Vec<f64>, JsValue> {
    let input = RankCorrelationIndexInput::from_slice(
        data,
        RankCorrelationIndexParams {
            length: Some(length),
        },
    );
    let mut out = vec![0.0; data.len()];
    rank_correlation_index_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "rank_correlation_index_batch_js")]
pub fn rank_correlation_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: RankCorrelationIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = RankCorrelationIndexBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let batch = rank_correlation_index_batch_slice(data, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&RankCorrelationIndexBatchJsOutput {
        values: batch.values,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rank_correlation_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rank_correlation_index_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rank_correlation_index_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to rank_correlation_index_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = RankCorrelationIndexInput::from_slice(
            data,
            RankCorrelationIndexParams {
                length: Some(length),
            },
        );
        rank_correlation_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "rank_correlation_index_into_host")]
pub fn rank_correlation_index_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to rank_correlation_index_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len());
        let input = RankCorrelationIndexInput::from_slice(
            data,
            RankCorrelationIndexParams {
                length: Some(length),
            },
        );
        rank_correlation_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rank_correlation_index_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to rank_correlation_index_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = RankCorrelationIndexBatchRange {
            length: (length_start, length_end, length_step),
        };
        let combos = expand_grid_rank_correlation_index(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        rank_correlation_index_batch_inner_into(data, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rank_correlation_index_output_into_js(
    data: &[f64],
    length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = rank_correlation_index_js(data, length)?;
    crate::write_wasm_f64_output("rank_correlation_index_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rank_correlation_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = rank_correlation_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "rank_correlation_index_batch_output_into_js",
        &value,
        out,
    )
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
            let base = 100.0 + (i as f64 * 0.11).sin() * 3.0 + i as f64 * 0.03;
            let tied = ((i % 5) as f64) * 0.25;
            out.push((base + tied).round());
        }
        out
    }

    fn naive_rci(data: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        if length < 2 || data.len() < length {
            return out;
        }
        for i in (length - 1)..data.len() {
            out[i] = compute_window_rci(&data[i + 1 - length..=i]);
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
    fn rank_correlation_index_matches_naive() {
        let data = sample_data(256);
        let input = RankCorrelationIndexInput::from_slice(
            &data,
            RankCorrelationIndexParams { length: Some(12) },
        );
        let out = rank_correlation_index(&input).expect("indicator");
        let reference = naive_rci(&data, 12);
        assert_close(&out.values, &reference);
    }

    #[test]
    fn rank_correlation_index_into_matches_api() {
        let data = sample_data(192);
        let input = RankCorrelationIndexInput::from_slice(
            &data,
            RankCorrelationIndexParams { length: Some(9) },
        );
        let baseline = rank_correlation_index(&input).expect("baseline");
        let mut out = vec![0.0; data.len()];
        rank_correlation_index_into(&input, &mut out).expect("into");
        assert_close(&baseline.values, &out);
    }

    #[test]
    fn rank_correlation_index_stream_matches_batch() {
        let data = sample_data(192);
        let batch = rank_correlation_index(&RankCorrelationIndexInput::from_slice(
            &data,
            RankCorrelationIndexParams { length: Some(12) },
        ))
        .expect("batch");
        let mut stream =
            RankCorrelationIndexStream::try_new(RankCorrelationIndexParams { length: Some(12) })
                .expect("stream");
        let mut values = Vec::with_capacity(data.len());
        for &value in &data {
            values.push(stream.update(value).unwrap_or(f64::NAN));
        }
        assert_close(&batch.values, &values);
    }

    #[test]
    fn rank_correlation_index_batch_single_param_matches_single() {
        let data = sample_data(192);
        let sweep = RankCorrelationIndexBatchRange {
            length: (12, 12, 0),
        };
        let batch = rank_correlation_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
            .expect("batch");
        let single = rank_correlation_index(&RankCorrelationIndexInput::from_slice(
            &data,
            RankCorrelationIndexParams { length: Some(12) },
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.values, &single.values);
    }

    #[test]
    fn rank_correlation_index_rejects_invalid_length() {
        let data = sample_data(32);
        let err = rank_correlation_index(&RankCorrelationIndexInput::from_slice(
            &data,
            RankCorrelationIndexParams { length: Some(1) },
        ))
        .expect_err("invalid length");
        assert!(matches!(
            err,
            RankCorrelationIndexError::InvalidLength { .. }
        ));
    }

    #[test]
    fn rank_correlation_index_dispatch_matches_direct() {
        let data = sample_data(192);
        let params = [ParamKV {
            key: "length",
            value: ParamValue::Int(12),
        }];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "rank_correlation_index",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = rank_correlation_index(&RankCorrelationIndexInput::from_slice(
            &data,
            RankCorrelationIndexParams { length: Some(12) },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert_close(out.values_f64.as_ref().expect("values"), &direct.values);
    }
}
