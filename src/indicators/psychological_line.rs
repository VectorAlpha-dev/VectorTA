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

impl<'a> AsRef<[f64]> for PsychologicalLineInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PsychologicalLineData::Candles { candles, source } => {
                psychological_line_source(candles, source)
            }
            PsychologicalLineData::Slice(slice) => slice,
        }
    }
}

#[inline(always)]
fn psychological_line_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => candles.open.as_slice(),
        "high" => candles.high.as_slice(),
        "low" => candles.low.as_slice(),
        "close" => candles.close.as_slice(),
        "volume" => candles.volume.as_slice(),
        "hl2" => candles.hl2.as_slice(),
        "hlc3" => candles.hlc3.as_slice(),
        "ohlc4" => candles.ohlc4.as_slice(),
        "hlcc4" | "hlcc" => candles.hlcc4.as_slice(),
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum PsychologicalLineData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PsychologicalLineOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PsychologicalLineParams {
    pub length: Option<usize>,
}

impl Default for PsychologicalLineParams {
    fn default() -> Self {
        Self { length: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct PsychologicalLineInput<'a> {
    pub data: PsychologicalLineData<'a>,
    pub params: PsychologicalLineParams,
}

impl<'a> PsychologicalLineInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: PsychologicalLineParams,
    ) -> Self {
        Self {
            data: PsychologicalLineData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: PsychologicalLineParams) -> Self {
        Self {
            data: PsychologicalLineData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", PsychologicalLineParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PsychologicalLineBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for PsychologicalLineBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PsychologicalLineBuilder {
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
    ) -> Result<PsychologicalLineOutput, PsychologicalLineError> {
        let input = PsychologicalLineInput::from_candles(
            candles,
            "close",
            PsychologicalLineParams {
                length: self.length,
            },
        );
        psychological_line_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<PsychologicalLineOutput, PsychologicalLineError> {
        let input = PsychologicalLineInput::from_slice(
            data,
            PsychologicalLineParams {
                length: self.length,
            },
        );
        psychological_line_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PsychologicalLineStream, PsychologicalLineError> {
        PsychologicalLineStream::try_new(PsychologicalLineParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum PsychologicalLineError {
    #[error("psychological_line: Input data slice is empty.")]
    EmptyInputData,
    #[error("psychological_line: All values are NaN.")]
    AllValuesNaN,
    #[error("psychological_line: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("psychological_line: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("psychological_line: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("psychological_line: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("psychological_line: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn first_valid_index(data: &[f64]) -> Option<usize> {
    data.iter().position(|x| x.is_finite())
}

#[inline(always)]
fn psychological_line_prepare<'a>(
    input: &'a PsychologicalLineInput,
) -> Result<(&'a [f64], usize, usize), PsychologicalLineError> {
    let data = input.as_ref();
    let data_len = data.len();
    if data_len == 0 {
        return Err(PsychologicalLineError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(PsychologicalLineError::AllValuesNaN)?;
    let length = input.get_length();
    if length == 0 || length > data_len {
        return Err(PsychologicalLineError::InvalidLength { length, data_len });
    }

    let valid = data_len - first;
    if valid <= length {
        return Err(PsychologicalLineError::NotEnoughValidData {
            needed: length + 1,
            valid,
        });
    }

    Ok((data, length, first))
}

#[inline(always)]
fn psychological_line_compute_fast_checked(
    data: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) -> bool {
    let warmup = first + length;
    let scale = 100.0 / length as f64;
    let mut count = 0usize;
    let ptr = data.as_ptr();

    unsafe {
        for i in (first + 1)..=warmup {
            let current = *ptr.add(i);
            if !current.is_finite() {
                return false;
            }
            count += usize::from(current > *ptr.add(i - 1));
        }
        *out.get_unchecked_mut(warmup) = count as f64 * scale;

        for i in (warmup + 1)..data.len() {
            let current = *ptr.add(i);
            if !current.is_finite() {
                return false;
            }
            count -= usize::from(*ptr.add(i - length) > *ptr.add(i - length - 1));
            count += usize::from(current > *ptr.add(i - 1));
            *out.get_unchecked_mut(i) = count as f64 * scale;
        }
    }
    true
}

#[inline(always)]
fn psychological_line_compute_fallback(data: &[f64], length: usize, first: usize, out: &mut [f64]) {
    let mut stream = PsychologicalLineStream::from_length(length);
    for i in first..data.len() {
        out[i] = stream.update_reset_on_nan(data[i]).unwrap_or(f64::NAN);
    }
}

#[inline(always)]
fn psychological_line_compute_into(
    data: &[f64],
    length: usize,
    first: usize,
    _kernel: Kernel,
    out: &mut [f64],
) {
    if !psychological_line_compute_fast_checked(data, length, first, out) {
        out.fill(f64::NAN);
        psychological_line_compute_fallback(data, length, first, out);
    }
}

#[inline]
pub fn psychological_line(
    input: &PsychologicalLineInput,
) -> Result<PsychologicalLineOutput, PsychologicalLineError> {
    psychological_line_with_kernel(input, Kernel::Auto)
}

pub fn psychological_line_with_kernel(
    input: &PsychologicalLineInput,
    kernel: Kernel,
) -> Result<PsychologicalLineOutput, PsychologicalLineError> {
    let (data, length, first) = psychological_line_prepare(input)?;
    let warmup = first + length;
    let mut out = alloc_with_nan_prefix(data.len(), warmup);
    psychological_line_compute_into(data, length, first, kernel, &mut out);
    Ok(PsychologicalLineOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn psychological_line_into(
    input: &PsychologicalLineInput,
    out: &mut [f64],
) -> Result<(), PsychologicalLineError> {
    psychological_line_into_slice(out, input, Kernel::Auto)
}

pub fn psychological_line_into_slice(
    out: &mut [f64],
    input: &PsychologicalLineInput,
    kernel: Kernel,
) -> Result<(), PsychologicalLineError> {
    let (data, length, first) = psychological_line_prepare(input)?;
    if out.len() != data.len() {
        return Err(PsychologicalLineError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    out[..(first + length)].fill(f64::NAN);
    psychological_line_compute_into(data, length, first, kernel, out);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct PsychologicalLineStream {
    length: usize,
    prev: Option<f64>,
    comparisons_seen: usize,
    head: usize,
    rolling_sum: usize,
    buffer: Vec<u8>,
}

impl PsychologicalLineStream {
    #[inline]
    fn from_length(length: usize) -> Self {
        Self {
            length,
            prev: None,
            comparisons_seen: 0,
            head: 0,
            rolling_sum: 0,
            buffer: vec![0; length.max(1)],
        }
    }

    #[inline]
    pub fn try_new(params: PsychologicalLineParams) -> Result<Self, PsychologicalLineError> {
        let length = params.length.unwrap_or(20);
        if length == 0 {
            return Err(PsychologicalLineError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self::from_length(length))
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev = None;
        self.comparisons_seen = 0;
        self.head = 0;
        self.rolling_sum = 0;
        self.buffer.fill(0);
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }

        let prev = match self.prev.replace(value) {
            Some(prev) => prev,
            None => return None,
        };

        let up = u8::from(value > prev);
        if self.comparisons_seen < self.length {
            self.buffer[self.comparisons_seen] = up;
            self.rolling_sum += up as usize;
            self.comparisons_seen += 1;
            if self.comparisons_seen < self.length {
                return None;
            }
            return Some(self.rolling_sum as f64 * (100.0 / self.length as f64));
        }

        let old = self.buffer[self.head] as usize;
        self.buffer[self.head] = up;
        self.rolling_sum = self.rolling_sum + up as usize - old;
        self.head += 1;
        if self.head == self.length {
            self.head = 0;
        }

        Some(self.rolling_sum as f64 * (100.0 / self.length as f64))
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
pub struct PsychologicalLineBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for PsychologicalLineBatchRange {
    fn default() -> Self {
        Self {
            length: (20, 200, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PsychologicalLineBatchBuilder {
    range: PsychologicalLineBatchRange,
    kernel: Kernel,
}

impl PsychologicalLineBatchBuilder {
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
    ) -> Result<PsychologicalLineBatchOutput, PsychologicalLineError> {
        psychological_line_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PsychologicalLineBatchOutput, PsychologicalLineError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct PsychologicalLineBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PsychologicalLineParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PsychologicalLineBatchOutput {
    pub fn row_for_params(&self, params: &PsychologicalLineParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.length.unwrap_or(20) == params.length.unwrap_or(20))
    }

    pub fn values_for(&self, params: &PsychologicalLineParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, PsychologicalLineError> {
    let (start, end, step) = range;
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
        return Err(PsychologicalLineError::InvalidRange { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_psychological_line(
    sweep: &PsychologicalLineBatchRange,
) -> Result<Vec<PsychologicalLineParams>, PsychologicalLineError> {
    Ok(axis_usize(sweep.length)?
        .into_iter()
        .map(|length| PsychologicalLineParams {
            length: Some(length),
        })
        .collect())
}

pub fn psychological_line_batch_with_kernel(
    data: &[f64],
    sweep: &PsychologicalLineBatchRange,
    kernel: Kernel,
) -> Result<PsychologicalLineBatchOutput, PsychologicalLineError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(PsychologicalLineError::InvalidKernelForBatch(other)),
    };
    psychological_line_batch_impl(data, sweep, batch_kernel.to_non_batch(), true)
}

pub fn psychological_line_batch_slice(
    data: &[f64],
    sweep: &PsychologicalLineBatchRange,
) -> Result<PsychologicalLineBatchOutput, PsychologicalLineError> {
    psychological_line_batch_impl(data, sweep, Kernel::Scalar, false)
}

pub fn psychological_line_batch_par_slice(
    data: &[f64],
    sweep: &PsychologicalLineBatchRange,
) -> Result<PsychologicalLineBatchOutput, PsychologicalLineError> {
    psychological_line_batch_impl(data, sweep, Kernel::Scalar, true)
}

fn psychological_line_batch_impl(
    data: &[f64],
    sweep: &PsychologicalLineBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<PsychologicalLineBatchOutput, PsychologicalLineError> {
    let combos = expand_grid_psychological_line(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    if cols == 0 {
        return Err(PsychologicalLineError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(PsychologicalLineError::AllValuesNaN)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(20))
        .max()
        .unwrap_or(20);
    let valid = cols - first;
    if valid <= max_length {
        return Err(PsychologicalLineError::NotEnoughValidData {
            needed: max_length + 1,
            valid,
        });
    }

    let mut matrix = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| first + params.length.unwrap_or(20))
        .collect();
    init_matrix_prefixes(&mut matrix, cols, &warmups);

    let mut guard = ManuallyDrop::new(matrix);
    let out_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let length = combos[row].length.unwrap_or(20);
        let dst = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        psychological_line_compute_into(data, length, first, kernel, dst);
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

    Ok(PsychologicalLineBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn psychological_line_batch_inner_into(
    data: &[f64],
    sweep: &PsychologicalLineBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), PsychologicalLineError> {
    let combos = expand_grid_psychological_line(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if rows.checked_mul(cols) != Some(out.len()) {
        return Err(PsychologicalLineError::OutputLengthMismatch {
            expected: rows * cols,
            got: out.len(),
        });
    }

    let first = first_valid_index(data).ok_or(PsychologicalLineError::AllValuesNaN)?;
    for (row, params) in combos.iter().enumerate() {
        let length = params.length.unwrap_or(20);
        let row_out = &mut out[row * cols..(row + 1) * cols];
        row_out.fill(f64::NAN);
        if cols - first <= length {
            return Err(PsychologicalLineError::NotEnoughValidData {
                needed: length + 1,
                valid: cols - first,
            });
        }
    }

    let do_row = |row: usize, row_out: &mut [f64]| {
        let length = combos[row].length.unwrap_or(20);
        psychological_line_compute_into(data, length, first, kernel, row_out);
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
#[pyfunction(name = "psychological_line")]
#[pyo3(signature = (data, length=20, kernel=None))]
pub fn psychological_line_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = PsychologicalLineInput::from_slice(
        data,
        PsychologicalLineParams {
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| psychological_line_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "PsychologicalLineStream")]
pub struct PsychologicalLineStreamPy {
    stream: PsychologicalLineStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PsychologicalLineStreamPy {
    #[new]
    #[pyo3(signature = (length=20))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = PsychologicalLineStream::try_new(PsychologicalLineParams {
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
#[pyfunction(name = "psychological_line_batch")]
#[pyo3(signature = (data, length_range, kernel=None))]
pub fn psychological_line_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = PsychologicalLineBatchRange {
        length: length_range,
    };
    let combos =
        expand_grid_psychological_line(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        psychological_line_batch_inner_into(data, &sweep, batch_kernel.to_non_batch(), true, out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_psychological_line_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(psychological_line_py, m)?)?;
    m.add_function(wrap_pyfunction!(psychological_line_batch_py, m)?)?;
    m.add_class::<PsychologicalLineStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PsychologicalLineBatchConfig {
    length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PsychologicalLineBatchJsOutput {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<PsychologicalLineParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "psychological_line_js")]
pub fn psychological_line_js(data: &[f64], length: usize) -> Result<Vec<f64>, JsValue> {
    let input = PsychologicalLineInput::from_slice(
        data,
        PsychologicalLineParams {
            length: Some(length),
        },
    );
    let mut out = vec![0.0; data.len()];
    psychological_line_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "psychological_line_batch_js")]
pub fn psychological_line_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: PsychologicalLineBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = PsychologicalLineBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let batch = psychological_line_batch_slice(data, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&PsychologicalLineBatchJsOutput {
        values: batch.values,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn psychological_line_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn psychological_line_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn psychological_line_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to psychological_line_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = PsychologicalLineInput::from_slice(
            data,
            PsychologicalLineParams {
                length: Some(length),
            },
        );
        psychological_line_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "psychological_line_into_host")]
pub fn psychological_line_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to psychological_line_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len());
        let input = PsychologicalLineInput::from_slice(
            data,
            PsychologicalLineParams {
                length: Some(length),
            },
        );
        psychological_line_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn psychological_line_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to psychological_line_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = PsychologicalLineBatchRange {
            length: (length_start, length_end, length_step),
        };
        let combos = expand_grid_psychological_line(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        psychological_line_batch_inner_into(data, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn psychological_line_output_into_js(
    data: &[f64],
    length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = psychological_line_js(data, length)?;
    crate::write_wasm_f64_output("psychological_line_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn psychological_line_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = psychological_line_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "psychological_line_batch_output_into_js",
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
            out.push(100.0 + i as f64 * 0.1 + (i as f64 * 0.37).sin() * 2.0);
        }
        out
    }

    fn naive_psy(data: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        if data.len() <= length {
            return out;
        }
        let scale = 100.0 / length as f64;
        let mut count = 0usize;
        for i in 1..=length {
            count += usize::from(data[i] > data[i - 1]);
        }
        out[length] = count as f64 * scale;
        for i in (length + 1)..data.len() {
            count -= usize::from(data[i - length] > data[i - length - 1]);
            count += usize::from(data[i] > data[i - 1]);
            out[i] = count as f64 * scale;
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
    fn psychological_line_matches_naive() {
        let data = sample_data(256);
        let input =
            PsychologicalLineInput::from_slice(&data, PsychologicalLineParams { length: Some(20) });
        let out = psychological_line(&input).expect("indicator");
        let reference = naive_psy(&data, 20);
        assert_close(&out.values, &reference);
    }

    #[test]
    fn psychological_line_into_matches_api() {
        let data = sample_data(192);
        let input =
            PsychologicalLineInput::from_slice(&data, PsychologicalLineParams { length: Some(14) });
        let baseline = psychological_line(&input).expect("baseline");
        let mut out = vec![0.0; data.len()];
        psychological_line_into(&input, &mut out).expect("into");
        assert_close(&baseline.values, &out);
    }

    #[test]
    fn psychological_line_stream_matches_batch() {
        let data = sample_data(192);
        let batch = psychological_line(&PsychologicalLineInput::from_slice(
            &data,
            PsychologicalLineParams { length: Some(20) },
        ))
        .expect("batch");
        let mut stream =
            PsychologicalLineStream::try_new(PsychologicalLineParams { length: Some(20) })
                .expect("stream");
        let mut values = Vec::with_capacity(data.len());
        for &value in &data {
            values.push(stream.update(value).unwrap_or(f64::NAN));
        }
        assert_close(&batch.values, &values);
    }

    #[test]
    fn psychological_line_batch_single_param_matches_single() {
        let data = sample_data(192);
        let sweep = PsychologicalLineBatchRange {
            length: (20, 20, 0),
        };
        let batch = psychological_line_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
            .expect("batch");
        let single = psychological_line(&PsychologicalLineInput::from_slice(
            &data,
            PsychologicalLineParams { length: Some(20) },
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.values, &single.values);
    }

    #[test]
    fn psychological_line_rejects_invalid_length() {
        let data = sample_data(32);
        let err = psychological_line(&PsychologicalLineInput::from_slice(
            &data,
            PsychologicalLineParams { length: Some(0) },
        ))
        .expect_err("invalid length");
        assert!(matches!(err, PsychologicalLineError::InvalidLength { .. }));
    }

    #[test]
    fn psychological_line_dispatch_matches_direct() {
        let data = sample_data(192);
        let params = [ParamKV {
            key: "length",
            value: ParamValue::Int(20),
        }];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "psychological_line",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = psychological_line(&PsychologicalLineInput::from_slice(
            &data,
            PsychologicalLineParams { length: Some(20) },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert_close(out.values_f64.as_ref().expect("values"), &direct.values);
    }
}
