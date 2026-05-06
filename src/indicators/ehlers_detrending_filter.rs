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
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 10;
const EDF_ALPHA: f64 = 0.95;
const EDF_FEEDBACK: f64 = 0.9;

impl<'a> AsRef<[f64]> for EhlersDetrendingFilterInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersDetrendingFilterData::Slice(slice) => slice,
            EhlersDetrendingFilterData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersDetrendingFilterData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersDetrendingFilterOutput {
    pub edf: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersDetrendingFilterParams {
    pub length: Option<usize>,
}

impl Default for EhlersDetrendingFilterParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersDetrendingFilterInput<'a> {
    pub data: EhlersDetrendingFilterData<'a>,
    pub params: EhlersDetrendingFilterParams,
}

impl<'a> EhlersDetrendingFilterInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersDetrendingFilterParams,
    ) -> Self {
        Self {
            data: EhlersDetrendingFilterData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: EhlersDetrendingFilterParams) -> Self {
        Self {
            data: EhlersDetrendingFilterData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "hlcc4", EhlersDetrendingFilterParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }
}

#[derive(Clone, Debug)]
pub struct EhlersDetrendingFilterBuilder {
    length: Option<usize>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for EhlersDetrendingFilterBuilder {
    fn default() -> Self {
        Self {
            length: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersDetrendingFilterBuilder {
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
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
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
    ) -> Result<EhlersDetrendingFilterOutput, EhlersDetrendingFilterError> {
        let input = EhlersDetrendingFilterInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("hlcc4"),
            EhlersDetrendingFilterParams {
                length: self.length,
            },
        );
        ehlers_detrending_filter_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersDetrendingFilterOutput, EhlersDetrendingFilterError> {
        let input = EhlersDetrendingFilterInput::from_slice(
            data,
            EhlersDetrendingFilterParams {
                length: self.length,
            },
        );
        ehlers_detrending_filter_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersDetrendingFilterStream, EhlersDetrendingFilterError> {
        EhlersDetrendingFilterStream::try_new(EhlersDetrendingFilterParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum EhlersDetrendingFilterError {
    #[error("ehlers_detrending_filter: Input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_detrending_filter: All source values are invalid.")]
    AllValuesNaN,
    #[error(
        "ehlers_detrending_filter: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("ehlers_detrending_filter: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "ehlers_detrending_filter: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ehlers_detrending_filter: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ehlers_detrending_filter: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn first_valid_source(data: &[f64]) -> Option<usize> {
    data.iter().position(|x| x.is_finite())
}

#[inline(always)]
fn max_consecutive_finite_from(data: &[f64], start: usize) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    for &value in &data[start..] {
        if value.is_finite() {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

#[inline(always)]
fn output_warmup(first: usize, length: usize) -> usize {
    first + length - 1
}

#[inline(always)]
fn normalized_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

fn cosine_weights(length: usize) -> (Vec<f64>, f64) {
    let denom = (length + 1) as f64;
    let mut weights = Vec::with_capacity(length);
    let mut sum = 0.0;
    for i in 1..=length {
        let weight = 1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
        weights.push(weight);
        sum += weight;
    }
    (weights, sum)
}

#[derive(Debug, Clone)]
pub struct EhlersDetrendingFilterStream {
    length: usize,
    weights: Vec<f64>,
    weight_sum: f64,
    ring: Vec<f64>,
    head: usize,
    count: usize,
    initialized: bool,
    prev_src: f64,
    prev_edf: f64,
    prev_filt: f64,
    prev_slo: f64,
}

impl EhlersDetrendingFilterStream {
    pub fn try_new(
        params: EhlersDetrendingFilterParams,
    ) -> Result<Self, EhlersDetrendingFilterError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length == 0 {
            return Err(EhlersDetrendingFilterError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        let (weights, weight_sum) = cosine_weights(length);
        Ok(Self {
            length,
            weights,
            weight_sum,
            ring: vec![0.0; length],
            head: 0,
            count: 0,
            initialized: false,
            prev_src: 0.0,
            prev_edf: 0.0,
            prev_filt: 0.0,
            prev_slo: 0.0,
        })
    }

    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.head = 0;
        self.count = 0;
        self.initialized = false;
        self.prev_src = 0.0;
        self.prev_edf = 0.0;
        self.prev_filt = 0.0;
        self.prev_slo = 0.0;
    }

    #[inline(always)]
    fn push_edf(&mut self, value: f64) {
        self.ring[self.head] = value;
        self.head = (self.head + 1) % self.length;
        if self.count < self.length {
            self.count += 1;
        }
    }

    #[inline(always)]
    fn weighted_filter(&self) -> f64 {
        let mut sum = 0.0;
        let mut offset = 0usize;
        let mut idx = self.head;
        while offset < self.count && idx > 0 {
            idx -= 1;
            sum += self.weights[offset] * self.ring[idx];
            offset += 1;
        }
        idx = self.length;
        while offset < self.count {
            idx -= 1;
            sum += self.weights[offset] * self.ring[idx];
            offset += 1;
        }
        if self.weight_sum != 0.0 {
            sum / self.weight_sum
        } else {
            0.0
        }
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let prev_src = if self.initialized { self.prev_src } else { 0.0 };
        let edf = (EDF_ALPHA * value) - (EDF_ALPHA * prev_src) + (EDF_FEEDBACK * self.prev_edf);
        self.push_edf(edf);

        let filt = self.weighted_filter();
        let slo = filt - self.prev_filt;
        let signal = if slo > 0.0 {
            if slo > self.prev_slo {
                2.0
            } else {
                1.0
            }
        } else if slo < 0.0 {
            if slo < self.prev_slo {
                -2.0
            } else {
                -1.0
            }
        } else {
            0.0
        };

        self.prev_src = value;
        self.prev_edf = edf;
        self.prev_filt = filt;
        self.prev_slo = slo;
        self.initialized = true;

        if self.count >= self.length {
            Some((filt, signal))
        } else {
            None
        }
    }
}

fn ehlers_detrending_filter_prepare<'a>(
    input: &'a EhlersDetrendingFilterInput<'a>,
) -> Result<(&'a [f64], usize, usize, usize), EhlersDetrendingFilterError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(EhlersDetrendingFilterError::EmptyInputData);
    }
    let first = first_valid_source(data).ok_or(EhlersDetrendingFilterError::AllValuesNaN)?;
    let length = input.get_length();
    if length == 0 || length > data.len() {
        return Err(EhlersDetrendingFilterError::InvalidLength {
            length,
            data_len: data.len(),
        });
    }
    let valid = max_consecutive_finite_from(data, first);
    if valid < length {
        return Err(EhlersDetrendingFilterError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }
    Ok((
        data,
        length,
        first,
        output_warmup(first, length).min(data.len()),
    ))
}

fn ehlers_detrending_filter_compute_into(
    data: &[f64],
    length: usize,
    _kernel: Kernel,
    out_edf: &mut [f64],
    out_signal: &mut [f64],
) {
    out_edf.fill(f64::NAN);
    out_signal.fill(f64::NAN);
    let mut stream = EhlersDetrendingFilterStream::try_new(EhlersDetrendingFilterParams {
        length: Some(length),
    })
    .expect("validated length");
    for (idx, value) in data.iter().copied().enumerate() {
        if let Some((edf, signal)) = stream.update(value) {
            out_edf[idx] = edf;
            out_signal[idx] = signal;
        }
    }
}

pub fn ehlers_detrending_filter(
    input: &EhlersDetrendingFilterInput,
) -> Result<EhlersDetrendingFilterOutput, EhlersDetrendingFilterError> {
    ehlers_detrending_filter_with_kernel(input, Kernel::Auto)
}

pub fn ehlers_detrending_filter_with_kernel(
    input: &EhlersDetrendingFilterInput,
    kernel: Kernel,
) -> Result<EhlersDetrendingFilterOutput, EhlersDetrendingFilterError> {
    let (data, length, _, warmup) = ehlers_detrending_filter_prepare(input)?;
    let _ = warmup;
    let mut edf = alloc_uninit_f64(data.len());
    let mut signal = alloc_uninit_f64(data.len());
    ehlers_detrending_filter_compute_into(
        data,
        length,
        normalized_kernel(kernel),
        &mut edf,
        &mut signal,
    );
    Ok(EhlersDetrendingFilterOutput { edf, signal })
}

pub fn ehlers_detrending_filter_into_slice(
    out_edf: &mut [f64],
    out_signal: &mut [f64],
    input: &EhlersDetrendingFilterInput,
    kernel: Kernel,
) -> Result<(), EhlersDetrendingFilterError> {
    let (data, length, _, _) = ehlers_detrending_filter_prepare(input)?;
    if out_edf.len() != data.len() || out_signal.len() != data.len() {
        return Err(EhlersDetrendingFilterError::OutputLengthMismatch {
            expected: data.len(),
            got: out_edf.len().max(out_signal.len()),
        });
    }
    ehlers_detrending_filter_compute_into(
        data,
        length,
        normalized_kernel(kernel),
        out_edf,
        out_signal,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn ehlers_detrending_filter_into(
    input: &EhlersDetrendingFilterInput,
    out_edf: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), EhlersDetrendingFilterError> {
    ehlers_detrending_filter_into_slice(out_edf, out_signal, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct EhlersDetrendingFilterBatchOutput {
    pub edf: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<EhlersDetrendingFilterParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersDetrendingFilterBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &EhlersDetrendingFilterParams) -> Option<usize> {
        self.combos.iter().position(|p| p.length == params.length)
    }
}

#[derive(Debug, Clone)]
pub struct EhlersDetrendingFilterBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for EhlersDetrendingFilterBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EhlersDetrendingFilterBatchBuilder {
    range: EhlersDetrendingFilterBatchRange,
    kernel: Kernel,
}

impl Default for EhlersDetrendingFilterBatchBuilder {
    fn default() -> Self {
        Self {
            range: EhlersDetrendingFilterBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersDetrendingFilterBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersDetrendingFilterBatchOutput, EhlersDetrendingFilterError> {
        ehlers_detrending_filter_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<EhlersDetrendingFilterBatchOutput, EhlersDetrendingFilterError> {
        self.apply_slice(source_type(candles, source))
    }
}

fn expand_usize_axis(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, EhlersDetrendingFilterError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    if start <= end {
        let mut out = Vec::new();
        let mut x = start;
        while x <= end {
            out.push(x);
            match x.checked_add(step.max(1)) {
                Some(next) if next > x => x = next,
                _ => break,
            }
        }
        if out.is_empty() {
            return Err(EhlersDetrendingFilterError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    } else {
        let mut out = Vec::new();
        let mut x = start;
        while x >= end {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step.max(1));
            if next == x || next < end {
                break;
            }
            x = next;
        }
        if out.is_empty() {
            return Err(EhlersDetrendingFilterError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }
}

pub fn expand_grid_ehlers_detrending_filter(
    range: &EhlersDetrendingFilterBatchRange,
) -> Result<Vec<EhlersDetrendingFilterParams>, EhlersDetrendingFilterError> {
    let lengths = expand_usize_axis(range.length)?;
    Ok(lengths
        .into_iter()
        .map(|length| EhlersDetrendingFilterParams {
            length: Some(length),
        })
        .collect())
}

pub fn ehlers_detrending_filter_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersDetrendingFilterBatchRange,
    kernel: Kernel,
) -> Result<EhlersDetrendingFilterBatchOutput, EhlersDetrendingFilterError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EhlersDetrendingFilterError::InvalidKernelForBatch(other)),
    };
    ehlers_detrending_filter_batch_inner(data, sweep, batch_kernel.to_non_batch(), true)
}

pub fn ehlers_detrending_filter_batch_slice(
    data: &[f64],
    sweep: &EhlersDetrendingFilterBatchRange,
    kernel: Kernel,
) -> Result<EhlersDetrendingFilterBatchOutput, EhlersDetrendingFilterError> {
    ehlers_detrending_filter_batch_inner(data, sweep, kernel, false)
}

pub fn ehlers_detrending_filter_batch_par_slice(
    data: &[f64],
    sweep: &EhlersDetrendingFilterBatchRange,
    kernel: Kernel,
) -> Result<EhlersDetrendingFilterBatchOutput, EhlersDetrendingFilterError> {
    ehlers_detrending_filter_batch_inner(data, sweep, kernel, true)
}

fn ehlers_detrending_filter_batch_inner(
    data: &[f64],
    sweep: &EhlersDetrendingFilterBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EhlersDetrendingFilterBatchOutput, EhlersDetrendingFilterError> {
    if data.is_empty() {
        return Err(EhlersDetrendingFilterError::EmptyInputData);
    }
    let first = first_valid_source(data).ok_or(EhlersDetrendingFilterError::AllValuesNaN)?;
    let combos = expand_grid_ehlers_detrending_filter(sweep)?;
    let valid = max_consecutive_finite_from(data, first);
    let max_length = combos
        .iter()
        .map(|p| p.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(DEFAULT_LENGTH);
    if max_length == 0 || max_length > data.len() {
        return Err(EhlersDetrendingFilterError::InvalidLength {
            length: max_length,
            data_len: data.len(),
        });
    }
    if valid < max_length {
        return Err(EhlersDetrendingFilterError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(EhlersDetrendingFilterError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    let mut edf = vec![f64::NAN; total];
    let mut signal = vec![f64::NAN; total];

    let do_row = |row: usize, dst_edf: &mut [f64], dst_signal: &mut [f64]| {
        let params = &combos[row];
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        ehlers_detrending_filter_compute_into(data, length, kernel, dst_edf, dst_signal);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        edf.par_chunks_mut(cols)
            .zip(signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (dst_edf, dst_signal))| do_row(row, dst_edf, dst_signal));
        #[cfg(target_arch = "wasm32")]
        for (row, (dst_edf, dst_signal)) in edf
            .chunks_mut(cols)
            .zip(signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_edf, dst_signal);
        }
    } else {
        for (row, (dst_edf, dst_signal)) in edf
            .chunks_mut(cols)
            .zip(signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_edf, dst_signal);
        }
    }

    Ok(EhlersDetrendingFilterBatchOutput {
        edf,
        signal,
        combos,
        rows,
        cols,
    })
}

fn ehlers_detrending_filter_batch_inner_into(
    data: &[f64],
    sweep: &EhlersDetrendingFilterBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_edf: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<EhlersDetrendingFilterParams>, EhlersDetrendingFilterError> {
    let combos = expand_grid_ehlers_detrending_filter(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(EhlersDetrendingFilterError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    if out_edf.len() != total || out_signal.len() != total {
        return Err(EhlersDetrendingFilterError::OutputLengthMismatch {
            expected: total,
            got: out_edf.len().max(out_signal.len()),
        });
    }
    out_edf.fill(f64::NAN);
    out_signal.fill(f64::NAN);

    let do_row = |row: usize, dst_edf: &mut [f64], dst_signal: &mut [f64]| {
        let params = &combos[row];
        ehlers_detrending_filter_compute_into(
            data,
            params.length.unwrap_or(DEFAULT_LENGTH),
            kernel,
            dst_edf,
            dst_signal,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_edf
            .par_chunks_mut(cols)
            .zip(out_signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (dst_edf, dst_signal))| do_row(row, dst_edf, dst_signal));
        #[cfg(target_arch = "wasm32")]
        for (row, (dst_edf, dst_signal)) in out_edf
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_edf, dst_signal);
        }
    } else {
        for (row, (dst_edf, dst_signal)) in out_edf
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_edf, dst_signal);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_detrending_filter")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, kernel=None))]
pub fn ehlers_detrending_filter_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = EhlersDetrendingFilterInput::from_slice(
        data,
        EhlersDetrendingFilterParams {
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| ehlers_detrending_filter_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((output.edf.into_pyarray(py), output.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersDetrendingFilterStream")]
pub struct EhlersDetrendingFilterStreamPy {
    stream: EhlersDetrendingFilterStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersDetrendingFilterStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = EhlersDetrendingFilterStream::try_new(EhlersDetrendingFilterParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_detrending_filter_batch")]
#[pyo3(signature = (data, length_range, kernel=None))]
pub fn ehlers_detrending_filter_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = EhlersDetrendingFilterBatchRange {
        length: length_range,
    };
    let combos = expand_grid_ehlers_detrending_filter(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let edf_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_edf = unsafe { edf_arr.as_slice_mut()? };
    let out_signal = unsafe { signal_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        ehlers_detrending_filter_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_edf,
            out_signal,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let lengths: Vec<usize> = combos
        .iter()
        .map(|c| c.length.unwrap_or(DEFAULT_LENGTH))
        .collect();
    let dict = PyDict::new(py);
    dict.set_item("edf", edf_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("lengths", lengths.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_detrending_filter_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ehlers_detrending_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_detrending_filter_batch_py, m)?)?;
    m.add_class::<EhlersDetrendingFilterStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EhlersDetrendingFilterJsOutput {
    edf: Vec<f64>,
    signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EhlersDetrendingFilterBatchConfig {
    length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EhlersDetrendingFilterBatchJsOutput {
    edf: Vec<f64>,
    signal: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<EhlersDetrendingFilterParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_detrending_filter_js")]
pub fn ehlers_detrending_filter_js(data: &[f64], length: usize) -> Result<JsValue, JsValue> {
    let input = EhlersDetrendingFilterInput::from_slice(
        data,
        EhlersDetrendingFilterParams {
            length: Some(length),
        },
    );
    let output = ehlers_detrending_filter_with_kernel(&input, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersDetrendingFilterJsOutput {
        edf: output.edf,
        signal: output.signal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_detrending_filter_batch_js")]
pub fn ehlers_detrending_filter_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersDetrendingFilterBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = EhlersDetrendingFilterBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let batch = ehlers_detrending_filter_batch_slice(data, &sweep, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersDetrendingFilterBatchJsOutput {
        edf: batch.edf,
        signal: batch.signal,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_detrending_filter_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 2];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_detrending_filter_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_detrending_filter_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_detrending_filter_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_edf, out_signal) = out.split_at_mut(len);
        let input = EhlersDetrendingFilterInput::from_slice(
            data,
            EhlersDetrendingFilterParams {
                length: Some(length),
            },
        );
        ehlers_detrending_filter_into_slice(out_edf, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_detrending_filter_into_host")]
pub fn ehlers_detrending_filter_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_detrending_filter_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len() * 2);
        let (out_edf, out_signal) = out.split_at_mut(data.len());
        let input = EhlersDetrendingFilterInput::from_slice(
            data,
            EhlersDetrendingFilterParams {
                length: Some(length),
            },
        );
        ehlers_detrending_filter_into_slice(out_edf, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_detrending_filter_batch_into(
    data: &[f64],
    out_ptr: *mut f64,
    config: JsValue,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_detrending_filter_batch_into",
        ));
    }
    let config: EhlersDetrendingFilterBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = EhlersDetrendingFilterBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let combos = expand_grid_ehlers_detrending_filter(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, total) };
    let (out_edf, out_signal) = out.split_at_mut(rows * cols);
    ehlers_detrending_filter_batch_inner_into(
        data,
        &sweep,
        Kernel::Scalar,
        false,
        out_edf,
        out_signal,
    )
    .map(|_| ())
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_detrending_filter_output_into_js(
    data: &[f64],
    length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_detrending_filter_js(data, length)?;
    crate::write_wasm_object_f64_outputs("ehlers_detrending_filter_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_detrending_filter_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_detrending_filter_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_detrending_filter_batch_output_into_js",
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
            let x = i as f64;
            out.push(100.0 + x * 0.05 + (x * 0.17).sin() * 1.9 + (x * 0.09).cos() * 0.7);
        }
        out
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            if x.is_nan() || y.is_nan() {
                assert!(x.is_nan() && y.is_nan(), "nan mismatch at {i}: {x} vs {y}");
            } else {
                assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn ehlers_detrending_filter_output_contract() {
        let data = sample_data(160);
        let input = EhlersDetrendingFilterInput::from_slice(
            &data,
            EhlersDetrendingFilterParams { length: Some(10) },
        );
        let out = ehlers_detrending_filter(&input).expect("indicator");
        assert_eq!(out.edf.len(), data.len());
        assert_eq!(out.signal.len(), data.len());
        assert!(out.edf[..9].iter().all(|v| v.is_nan()));
        assert!(out.signal[..9].iter().all(|v| v.is_nan()));
        assert!(out.edf[9..].iter().any(|v| v.is_finite()));
        assert!(out.signal[9..].iter().any(|v| v.is_finite()));
    }

    #[test]
    fn ehlers_detrending_filter_into_matches_api() {
        let data = sample_data(160);
        let input = EhlersDetrendingFilterInput::from_slice(
            &data,
            EhlersDetrendingFilterParams { length: Some(12) },
        );
        let baseline = ehlers_detrending_filter(&input).expect("baseline");
        let mut edf = vec![0.0; data.len()];
        let mut signal = vec![0.0; data.len()];
        ehlers_detrending_filter_into_slice(&mut edf, &mut signal, &input, Kernel::Auto)
            .expect("into");
        assert_close(&baseline.edf, &edf, 1e-12);
        assert_close(&baseline.signal, &signal, 1e-12);
    }

    #[test]
    fn ehlers_detrending_filter_stream_matches_batch() {
        let data = sample_data(160);
        let input = EhlersDetrendingFilterInput::from_slice(
            &data,
            EhlersDetrendingFilterParams { length: Some(14) },
        );
        let batch = ehlers_detrending_filter(&input).expect("batch");
        let mut stream = EhlersDetrendingFilterStream::try_new(EhlersDetrendingFilterParams {
            length: Some(14),
        })
        .expect("stream");
        let mut edf = vec![f64::NAN; data.len()];
        let mut signal = vec![f64::NAN; data.len()];
        for (i, value) in data.iter().copied().enumerate() {
            if let Some((a, b)) = stream.update(value) {
                edf[i] = a;
                signal[i] = b;
            }
        }
        assert_close(&batch.edf, &edf, 1e-12);
        assert_close(&batch.signal, &signal, 1e-12);
    }

    #[test]
    fn ehlers_detrending_filter_batch_single_param_matches_single() {
        let data = sample_data(160);
        let batch = ehlers_detrending_filter_batch_with_kernel(
            &data,
            &EhlersDetrendingFilterBatchRange {
                length: (11, 11, 0),
            },
            Kernel::Auto,
        )
        .expect("batch");
        let single = ehlers_detrending_filter(&EhlersDetrendingFilterInput::from_slice(
            &data,
            EhlersDetrendingFilterParams { length: Some(11) },
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.edf[..data.len()], &single.edf, 1e-12);
        assert_close(&batch.signal[..data.len()], &single.signal, 1e-12);
    }

    #[test]
    fn ehlers_detrending_filter_rejects_invalid_length() {
        let data = sample_data(32);
        let input = EhlersDetrendingFilterInput::from_slice(
            &data,
            EhlersDetrendingFilterParams { length: Some(0) },
        );
        let err = ehlers_detrending_filter(&input).expect_err("invalid");
        assert!(matches!(
            err,
            EhlersDetrendingFilterError::InvalidLength { .. }
        ));
    }

    #[test]
    fn ehlers_detrending_filter_dispatch_matches_direct() {
        let data = sample_data(160);
        let combo = [ParamKV {
            key: "length",
            value: ParamValue::Int(13),
        }];
        let combos = [IndicatorParamSet { params: &combo }];

        let req_edf = IndicatorBatchRequest {
            indicator_id: "ehlers_detrending_filter",
            output_id: Some("edf"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let req_signal = IndicatorBatchRequest {
            indicator_id: "ehlers_detrending_filter",
            output_id: Some("signal"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };

        let out_edf = compute_cpu_batch(req_edf).expect("dispatch edf");
        let out_signal = compute_cpu_batch(req_signal).expect("dispatch signal");
        let direct = ehlers_detrending_filter(&EhlersDetrendingFilterInput::from_slice(
            &data,
            EhlersDetrendingFilterParams { length: Some(13) },
        ))
        .expect("direct");

        assert_eq!(out_edf.rows, 1);
        assert_eq!(out_edf.cols, data.len());
        assert_eq!(out_signal.rows, 1);
        assert_eq!(out_signal.cols, data.len());
        assert_close(
            out_edf.values_f64.as_ref().expect("edf values"),
            &direct.edf,
            1e-12,
        );
        assert_close(
            out_signal.values_f64.as_ref().expect("signal values"),
            &direct.signal,
            1e-12,
        );
    }
}
