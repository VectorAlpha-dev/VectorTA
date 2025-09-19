//! # Percentile Nearest Rank (PNR)
//!
//! A statistical indicator that calculates the value at a specified percentile
//! within a rolling window of data points.
//!
//! ## Parameters
//! - **length**: Window size for the rolling calculation (default: 15)
//! - **percentage**: Percentile to calculate (0-100, default: 50)
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **values**: Vector of percentile values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Not implemented (kernel parameter reserved but unused)
//! - **Streaming**: Implemented with O(n log n) update performance (sorts window on each update)
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

impl<'a> AsRef<[f64]> for PercentileNearestRankInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PercentileNearestRankData::Slice(slice) => slice,
            PercentileNearestRankData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PercentileNearestRankData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PercentileNearestRankOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct PercentileNearestRankParams {
    pub length: Option<usize>,
    pub percentage: Option<f64>,
}

impl Default for PercentileNearestRankParams {
    fn default() -> Self {
        Self {
            length: Some(15),
            percentage: Some(50.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PercentileNearestRankInput<'a> {
    pub data: PercentileNearestRankData<'a>,
    pub params: PercentileNearestRankParams,
}

impl<'a> PercentileNearestRankInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PercentileNearestRankParams) -> Self {
        Self {
            data: PercentileNearestRankData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PercentileNearestRankParams) -> Self {
        Self {
            data: PercentileNearestRankData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PercentileNearestRankParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(15)
    }

    #[inline]
    pub fn get_percentage(&self) -> f64 {
        self.params.percentage.unwrap_or(50.0)
    }
}

#[derive(Debug, Error)]
pub enum PercentileNearestRankError {
    #[error("percentile_nearest_rank: Input data is empty")]
    EmptyInputData,

    #[error("percentile_nearest_rank: All values are NaN")]
    AllValuesNaN,

    #[error(
        "percentile_nearest_rank: Invalid period: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("percentile_nearest_rank: Percentage must be between 0 and 100, got {percentage}")]
    InvalidPercentage { percentage: f64 },

    #[error("percentile_nearest_rank: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline(always)]
fn pnr_prepare<'a>(
    input: &'a PercentileNearestRankInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, usize, Kernel), PercentileNearestRankError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(PercentileNearestRankError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PercentileNearestRankError::AllValuesNaN)?;

    let length = input.get_length();
    let percentage = input.get_percentage();

    if length == 0 || length > len {
        return Err(PercentileNearestRankError::InvalidPeriod {
            period: length,
            data_len: len,
        });
    }
    if !(0.0..=100.0).contains(&percentage) || percentage.is_nan() || percentage.is_infinite() {
        return Err(PercentileNearestRankError::InvalidPercentage { percentage });
    }
    if len - first < length {
        return Err(PercentileNearestRankError::NotEnoughValidData {
            needed: length,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((data, length, percentage, first, chosen))
}

#[inline(always)]
fn pnr_compute_into(
    data: &[f64],
    length: usize,
    percentage: f64,
    first: usize,
    _kernel: Kernel, // reserved for parity; no SIMD needed
    out: &mut [f64],
) {
    // warmup prefix already set by caller when needed
    let mut window = Vec::with_capacity(length);
    let start_i = first + length - 1;

    for i in start_i..data.len() {
        window.clear();
        // collect last `length` samples ending at i (Pine semantics, newest first)
        for j in 0..length {
            let idx = i - j;
            let v = data[idx];
            if !v.is_nan() {
                window.push(v);
            }
        }

        if window.is_empty() {
            out[i] = f64::NAN;
            continue;
        }
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // nearest-rank index based on the actual window size
        let wl = window.len() as f64;
        let raw = (percentage / 100.0 * wl).round() as isize - 1;
        let idx = raw.max(0) as usize;
        out[i] = window[idx.min(window.len() - 1)];
    }
}

#[inline]
pub fn percentile_nearest_rank(
    input: &PercentileNearestRankInput,
) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
    percentile_nearest_rank_with_kernel(input, Kernel::Auto)
}

pub fn percentile_nearest_rank_with_kernel(
    input: &PercentileNearestRankInput,
    kernel: Kernel,
) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
    let (data, length, percentage, first, chosen) = pnr_prepare(input, kernel)?;
    let warmup_end = first + length - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_end);
    pnr_compute_into(data, length, percentage, first, chosen, &mut out);
    Ok(PercentileNearestRankOutput { values: out })
}

/// Parity with `alma_into_slice`
#[inline]
pub fn percentile_nearest_rank_into_slice(
    dst: &mut [f64],
    input: &PercentileNearestRankInput,
    kernel: Kernel,
) -> Result<(), PercentileNearestRankError> {
    let (data, length, percentage, first, chosen) = pnr_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(PercentileNearestRankError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    // compute into `dst`
    pnr_compute_into(data, length, percentage, first, chosen, dst);

    // set warmup NaNs exactly like alma.rs
    let warmup_end = first + length - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    Ok(())
}

// ==================== BUILDER PATTERN ====================

#[derive(Copy, Clone, Debug)]
pub struct PercentileNearestRankBuilder {
    length: Option<usize>,
    percentage: Option<f64>,
    kernel: Kernel,
}

impl Default for PercentileNearestRankBuilder {
    fn default() -> Self {
        Self {
            length: None,
            percentage: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PercentileNearestRankBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }

    #[inline(always)]
    pub fn percentage(mut self, p: f64) -> Self {
        self.percentage = Some(p);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn build(
        self,
        data: &[f64],
    ) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
        let params = PercentileNearestRankParams {
            length: self.length,
            percentage: self.percentage,
        };
        let input = PercentileNearestRankInput::from_slice(data, params);
        percentile_nearest_rank_with_kernel(&input, self.kernel)
    }

    pub fn build_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
        let params = PercentileNearestRankParams {
            length: self.length,
            percentage: self.percentage,
        };
        let input = PercentileNearestRankInput::from_candles(candles, source, params);
        percentile_nearest_rank_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        c: &Candles,
    ) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
        let p = PercentileNearestRankParams {
            length: self.length,
            percentage: self.percentage,
        };
        let i = PercentileNearestRankInput::from_candles(c, "close", p);
        percentile_nearest_rank_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        d: &[f64],
    ) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
        let p = PercentileNearestRankParams {
            length: self.length,
            percentage: self.percentage,
        };
        let i = PercentileNearestRankInput::from_slice(d, p);
        percentile_nearest_rank_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PercentileNearestRankStream, PercentileNearestRankError> {
        let p = PercentileNearestRankParams {
            length: self.length,
            percentage: self.percentage,
        };
        PercentileNearestRankStream::try_new(p)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
        Self::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<PercentileNearestRankOutput, PercentileNearestRankError> {
        Self::new().kernel(Kernel::Auto).apply(c)
    }
}

// ==================== STREAMING SUPPORT ====================

#[derive(Debug, Clone)]
pub struct PercentileNearestRankStream {
    length: usize,
    percentage: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl PercentileNearestRankStream {
    pub fn try_new(
        params: PercentileNearestRankParams,
    ) -> Result<Self, PercentileNearestRankError> {
        let length = params.length.unwrap_or(15);
        if length == 0 {
            return Err(PercentileNearestRankError::InvalidPeriod {
                period: length,
                data_len: 0,
            });
        }

        let percentage = params.percentage.unwrap_or(50.0);
        if !(0.0..=100.0).contains(&percentage) || percentage.is_nan() || percentage.is_infinite() {
            return Err(PercentileNearestRankError::InvalidPercentage { percentage });
        }

        Ok(Self {
            length,
            percentage,
            buffer: vec![f64::NAN; length],
            head: 0,
            filled: false,
        })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.length;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }

        let mut window: Vec<f64> = self
            .buffer
            .iter()
            .copied()
            .filter(|x| !x.is_nan())
            .collect();
        if window.is_empty() {
            return Some(f64::NAN);
        }
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // changed: use window.len() like pnr_compute_into
        let wl = window.len() as f64;
        let raw = (self.percentage / 100.0 * wl).round() as isize - 1;
        let idx = raw.max(0) as usize;
        Some(window[idx.min(window.len() - 1)])
    }
}

// ==================== PYTHON BINDINGS ====================

#[cfg(feature = "python")]
#[pyfunction(name = "percentile_nearest_rank")]
#[pyo3(signature = (data, length=15, percentage=50.0, kernel=None))]
pub fn percentile_nearest_rank_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    percentage: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kern = validate_kernel(kernel, false)?;
    let data_slice = data.as_slice()?;

    let params = PercentileNearestRankParams {
        length: Some(length),
        percentage: Some(percentage),
    };
    let input = PercentileNearestRankInput::from_slice(data_slice, params);

    let result = py
        .allow_threads(|| percentile_nearest_rank_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "PercentileNearestRankStream")]
pub struct PercentileNearestRankStreamPy {
    stream: PercentileNearestRankStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PercentileNearestRankStreamPy {
    #[new]
    fn new(length: usize, percentage: f64) -> PyResult<Self> {
        let params = PercentileNearestRankParams {
            length: Some(length),
            percentage: Some(percentage),
        };
        let stream = PercentileNearestRankStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PercentileNearestRankStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "percentile_nearest_rank_batch")]
#[pyo3(signature = (data, length_range, percentage_range, kernel=None))]
pub fn percentile_nearest_rank_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    percentage_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let sweep = PercentileNearestRankBatchRange {
        length: length_range,
        percentage: percentage_range,
    };

    // expand once to know rows/cols and reuse later for metadata
    let combos = expand_grid_pnr(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // flat output buffer, no extra copy
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Initialize warmup values with NaN
    for (row_idx, combo) in combos.iter().enumerate() {
        let length = combo.length.unwrap_or(15);
        let warmup = length - 1;
        let row_start = row_idx * cols;
        for i in 0..warmup.min(cols) {
            slice_out[row_start + i] = f64::NAN;
        }
    }

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let k = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        // compute directly into NumPy memory
        pnr_batch_inner_into(slice_in, &combos, k, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(15) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "percentages",
        combos
            .iter()
            .map(|p| p.percentage.unwrap_or(50.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict.into())
}

// ==================== BATCH PROCESSING ====================

#[derive(Clone, Debug)]
pub struct PercentileNearestRankBatchRange {
    pub length: (usize, usize, usize), // (start, end, step)
    pub percentage: (f64, f64, f64),   // (start, end, step)
}

impl Default for PercentileNearestRankBatchRange {
    fn default() -> Self {
        Self {
            length: (15, 100, 5),
            percentage: (50.0, 50.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PercentileNearestRankBatchBuilder {
    range: PercentileNearestRankBatchRange,
    kernel: Kernel,
}

impl PercentileNearestRankBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    pub fn percentage_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.percentage = (start, end, step);
        self
    }

    pub fn apply(
        self,
        data: &[f64],
    ) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
        pnr_batch_slice(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
        let data = source_type(candles, source);
        pnr_batch_slice(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
        PercentileNearestRankBatchBuilder::new()
            .kernel(k)
            .apply(data)
    }

    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
        PercentileNearestRankBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct PercentileNearestRankBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PercentileNearestRankParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PercentileNearestRankBatchOutput {
    pub fn row_for_params(&self, p: &PercentileNearestRankParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(15) == p.length.unwrap_or(15)
                && (c.percentage.unwrap_or(50.0) - p.percentage.unwrap_or(50.0)).abs() < 1e-12
        })
    }

    pub fn values_for(&self, p: &PercentileNearestRankParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_pnr(r: &PercentileNearestRankBatchRange) -> Vec<PercentileNearestRankParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step == 0.0 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut val = start;
        while val <= end + 1e-12 {
            v.push(val);
            val += step;
        }
        v
    }

    let lengths = axis_usize(r.length);
    let percentages = axis_f64(r.percentage);

    let mut combos = Vec::with_capacity(lengths.len() * percentages.len());
    for &length in &lengths {
        for &percentage in &percentages {
            combos.push(PercentileNearestRankParams {
                length: Some(length),
                percentage: Some(percentage),
            });
        }
    }
    combos
}

#[inline(always)]
pub fn pnr_batch_with_kernel(
    data: &[f64],
    sweep: &PercentileNearestRankBatchRange,
    k: Kernel,
) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PercentileNearestRankError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    pnr_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
pub fn pnr_batch_slice(
    data: &[f64],
    sweep: &PercentileNearestRankBatchRange,
    k: Kernel,
) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
    pnr_batch_inner(data, sweep, k, false)
}

#[inline(always)]
pub fn pnr_batch_par_slice(
    data: &[f64],
    sweep: &PercentileNearestRankBatchRange,
    k: Kernel,
) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
    pnr_batch_inner(data, sweep, k, true)
}

#[inline(always)]
fn pnr_batch_inner(
    data: &[f64],
    sweep: &PercentileNearestRankBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PercentileNearestRankBatchOutput, PercentileNearestRankError> {
    if data.is_empty() {
        return Err(PercentileNearestRankError::EmptyInputData);
    }
    let combos = expand_grid_pnr(sweep);
    if combos.is_empty() {
        return Err(PercentileNearestRankError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // first valid and availability check mirrors alma.rs
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PercentileNearestRankError::AllValuesNaN)?;
    let max_len = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
    if data.len() - first < max_len {
        return Err(PercentileNearestRankError::NotEnoughValidData {
            needed: max_len,
            valid: data.len() - first,
        });
    }

    // allocate uninit matrix and set NaN prefixes per row
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.length.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // write into the uninit matrix in place
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    pnr_batch_inner_into(data, &combos, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(PercentileNearestRankBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn pnr_batch_inner_into(
    data: &[f64],
    combos: &[PercentileNearestRankParams],
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), PercentileNearestRankError> {
    let cols = data.len();
    if cols == 0 {
        return Err(PercentileNearestRankError::EmptyInputData);
    }

    // single pass like alma.rs
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PercentileNearestRankError::AllValuesNaN)?;
    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_row: &mut [f64]| {
        let length = combos[row].length.unwrap_or(15);
        let percentage = combos[row].percentage.unwrap_or(50.0);
        // warmup NaNs were already written by init_matrix_prefixes
        pnr_compute_into(data, length, percentage, first, chosen, dst_row);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, s)| do_row(row, s));
        }
        #[cfg(target_arch = "wasm32")]
        for (row, s) in out.chunks_mut(cols).enumerate() {
            do_row(row, s);
        }
    } else {
        for (row, s) in out.chunks_mut(cols).enumerate() {
            do_row(row, s);
        }
    }
    Ok(())
}

// ==================== WASM BINDINGS ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn percentile_nearest_rank_js(
    data: &[f64],
    length: usize,
    percentage: f64,
) -> Result<Vec<f64>, JsValue> {
    let params = PercentileNearestRankParams {
        length: Some(length),
        percentage: Some(percentage),
    };
    let input = PercentileNearestRankInput::from_slice(data, params);
    percentile_nearest_rank(&input)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn percentile_nearest_rank_alloc(n: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(n);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn percentile_nearest_rank_free(ptr: *mut f64, n: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, n, n);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn percentile_nearest_rank_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    percentage: f64,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let params = PercentileNearestRankParams {
            length: Some(length),
            percentage: Some(percentage),
        };
        let input = PercentileNearestRankInput::from_slice(data, params);

        if data_ptr == out_ptr {
            // avoid overwriting in-place
            let mut temp = vec![0.0; len];
            percentile_nearest_rank_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            percentile_nearest_rank_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PercentileNearestRankBatchConfig {
    pub length_range: (usize, usize, usize),
    pub percentage_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PercentileNearestRankBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PercentileNearestRankParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = percentile_nearest_rank_batch)]
pub fn percentile_nearest_rank_batch_unified_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: PercentileNearestRankBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = PercentileNearestRankBatchRange {
        length: cfg.length_range,
        percentage: cfg.percentage_range,
    };
    let out = pnr_batch_inner(data, &sweep, detect_best_batch_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = PercentileNearestRankBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    macro_rules! skip_if_unsupported {
        ($kernel:expr, $test_name:expr) => {
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            match $kernel {
                Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch => {
                    println!("[{}] Skipping: AVX not supported", $test_name);
                    return Ok(());
                }
                _ => {}
            }
        };
    }

    fn check_pnr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = PercentileNearestRankParams {
            length: Some(15),
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_candles(&candles, "close", params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel)?;

        // Verify result structure
        assert_eq!(result.values.len(), candles.close.len());

        // Check that warmup period is NaN
        for i in 0..14 {
            assert!(
                result.values[i].is_nan(),
                "[{}] Expected NaN at index {}",
                test_name,
                i
            );
        }

        // Check that we have valid values after warmup
        assert!(
            !result.values[14].is_nan(),
            "[{}] Expected valid value at index 14",
            test_name
        );

        // Verify last 5 values match expected reference values
        let expected_last_5 = vec![59419.0, 59419.0, 59300.0, 59285.0, 59273.0];
        let len = result.values.len();
        let actual_last_5 = &result.values[len - 5..];

        for (i, (&actual, &expected)) in
            actual_last_5.iter().zip(expected_last_5.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-6,
                "[{}] Value mismatch at last_5[{}]: expected {}, got {}, diff {}",
                test_name,
                i,
                expected,
                actual,
                diff
            );
        }

        Ok(())
    }

    fn check_pnr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test with partial params (only length)
        let params = PercentileNearestRankParams {
            length: Some(5),
            percentage: None,
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[4], 3.0); // Default 50th percentile

        Ok(())
    }

    fn check_pnr_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = PercentileNearestRankInput::with_default_candles(&candles);
        let result = percentile_nearest_rank_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pnr_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![1.0; 10];
        let params = PercentileNearestRankParams {
            length: Some(0),
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel);

        assert!(matches!(
            result,
            Err(PercentileNearestRankError::InvalidPeriod { .. })
        ));
        Ok(())
    }

    fn check_pnr_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![1.0; 5];
        let params = PercentileNearestRankParams {
            length: Some(10),
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel);

        assert!(matches!(
            result,
            Err(PercentileNearestRankError::InvalidPeriod { .. })
        ));
        Ok(())
    }

    fn check_pnr_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![5.0];
        let params = PercentileNearestRankParams {
            length: Some(1),
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), 1);
        assert_eq!(result.values[0], 5.0);
        Ok(())
    }

    fn check_pnr_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data: Vec<f64> = vec![];
        let params = PercentileNearestRankParams::default();
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel);

        assert!(matches!(
            result,
            Err(PercentileNearestRankError::EmptyInputData)
        ));
        Ok(())
    }

    fn check_pnr_invalid_percentage(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![1.0; 20];

        // Test percentage > 100
        let params = PercentileNearestRankParams {
            length: Some(5),
            percentage: Some(150.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel);
        assert!(matches!(
            result,
            Err(PercentileNearestRankError::InvalidPercentage { .. })
        ));

        // Test negative percentage
        let params = PercentileNearestRankParams {
            length: Some(5),
            percentage: Some(-10.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel);
        assert!(matches!(
            result,
            Err(PercentileNearestRankError::InvalidPercentage { .. })
        ));

        Ok(())
    }

    fn check_pnr_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = PercentileNearestRankParams {
            length: Some(15),
            percentage: Some(50.0),
        };
        let first_input = PercentileNearestRankInput::from_candles(&candles, "close", first_params);
        let first_result = percentile_nearest_rank_with_kernel(&first_input, kernel)?;

        let second_params = PercentileNearestRankParams {
            length: Some(15),
            percentage: Some(50.0),
        };
        let second_input =
            PercentileNearestRankInput::from_slice(&first_result.values, second_params);
        let second_result = percentile_nearest_rank_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_pnr_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![
            1.0,
            2.0,
            f64::NAN,
            4.0,
            5.0,
            f64::NAN,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            f64::NAN,
            15.0,
        ];

        let params = PercentileNearestRankParams {
            length: Some(5),
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), data.len());
        // Should handle NaN values in window
        assert!(!result.values[6].is_nan());
        Ok(())
    }

    fn check_pnr_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = PercentileNearestRankParams {
            length: Some(15),
            percentage: Some(50.0),
        };

        let input = PercentileNearestRankInput::from_candles(&candles, "close", params.clone());
        let batch_output = percentile_nearest_rank_with_kernel(&input, kernel)?.values;

        let mut stream = PercentileNearestRankStream::try_new(params)?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] PNR streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_pnr_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            PercentileNearestRankParams::default(),
            PercentileNearestRankParams {
                length: Some(5),
                percentage: Some(25.0),
            },
            PercentileNearestRankParams {
                length: Some(10),
                percentage: Some(75.0),
            },
            PercentileNearestRankParams {
                length: Some(20),
                percentage: Some(50.0),
            },
            PercentileNearestRankParams {
                length: Some(50),
                percentage: Some(90.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = PercentileNearestRankInput::from_candles(&candles, "close", params.clone());
            let output = percentile_nearest_rank_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: length={}, percentage={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(15),
                        params.percentage.unwrap_or(50.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_pnr_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    // Macro to generate tests for all kernel variants
    macro_rules! generate_all_pnr_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    // Generate all kernel-specific tests
    generate_all_pnr_tests!(
        check_pnr_accuracy,
        check_pnr_partial_params,
        check_pnr_default_candles,
        check_pnr_zero_period,
        check_pnr_period_exceeds_length,
        check_pnr_very_small_dataset,
        check_pnr_empty_input,
        check_pnr_invalid_percentage,
        check_pnr_reinput,
        check_pnr_nan_handling,
        check_pnr_streaming,
        check_pnr_no_poison
    );

    // Batch testing functions
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = PercentileNearestRankBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = PercentileNearestRankParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Check warmup period
        for i in 0..14 {
            assert!(row[i].is_nan());
        }
        assert!(!row[14].is_nan());

        Ok(())
    }

    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = PercentileNearestRankBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)
            .percentage_range(25.0, 75.0, 25.0)
            .apply_candles(&c, "close")?;

        assert_eq!(output.rows, 9); // 3 periods * 3 percentages
        assert_eq!(output.cols, c.close.len());
        assert_eq!(output.combos.len(), 9);

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = PercentileNearestRankBatchBuilder::new()
            .kernel(kernel)
            .period_range(5, 20, 5)
            .percentage_range(10.0, 90.0, 20.0)
            .apply_candles(&c, "close")?;

        for &val in &output.values {
            if val.is_nan() {
                continue;
            }
            let bits = val.to_bits();
            if bits == 0x11111111_11111111
                || bits == 0x22222222_22222222
                || bits == 0x33333333_33333333
            {
                panic!(
                    "[{}] Found poison value {} (0x{:016X})",
                    test_name, val, bits
                );
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    // Macro for batch tests
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    // Keep old simple tests for compatibility
    #[test]
    fn test_percentile_nearest_rank_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let params = PercentileNearestRankParams {
            length: Some(5),
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank(&input).unwrap();

        assert_eq!(result.values.len(), data.len());
        // First 4 values should be NaN (warmup period)
        for i in 0..4 {
            assert!(result.values[i].is_nan());
        }
        // 50th percentile of [1,2,3,4,5] = 3.0
        assert_eq!(result.values[4], 3.0);
    }

    #[test]
    fn test_percentile_nearest_rank_empty_data() {
        let data = vec![];
        let params = PercentileNearestRankParams::default();
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank(&input);

        assert!(matches!(
            result,
            Err(PercentileNearestRankError::EmptyInputData)
        ));
    }

    #[test]
    fn test_percentile_nearest_rank_all_nan() {
        let data = vec![f64::NAN; 10];
        let params = PercentileNearestRankParams::default();
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank(&input);

        assert!(matches!(
            result,
            Err(PercentileNearestRankError::AllValuesNaN)
        ));
    }

    #[test]
    fn test_percentile_nearest_rank_invalid_percentage() {
        let data = vec![1.0; 20];
        let params = PercentileNearestRankParams {
            length: Some(5),
            percentage: Some(150.0), // Invalid
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank(&input);

        assert!(matches!(
            result,
            Err(PercentileNearestRankError::InvalidPercentage { .. })
        ));
    }

    #[test]
    fn test_percentile_nearest_rank_period_too_large() {
        let data = vec![1.0; 10];
        let params = PercentileNearestRankParams {
            length: Some(20), // Larger than data
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_slice(&data, params);
        let result = percentile_nearest_rank(&input);

        assert!(matches!(
            result,
            Err(PercentileNearestRankError::InvalidPeriod { .. })
        ));
    }

    #[test]
    fn test_percentile_nearest_rank_with_candles() {
        let close_data = vec![
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5,
            17.5, 18.5, 19.5, 20.5,
        ];
        let open_data = vec![1.0; 20];
        let high_data = vec![2.0; 20];
        let low_data = vec![0.5; 20];
        let volume_data = vec![100.0; 20];

        // Calculate derived fields
        let mut hl2 = Vec::with_capacity(20);
        let mut hlc3 = Vec::with_capacity(20);
        let mut ohlc4 = Vec::with_capacity(20);
        let mut hlcc4 = Vec::with_capacity(20);

        for i in 0..20 {
            hl2.push((high_data[i] + low_data[i]) / 2.0);
            hlc3.push((high_data[i] + low_data[i] + close_data[i]) / 3.0);
            ohlc4.push((open_data[i] + high_data[i] + low_data[i] + close_data[i]) / 4.0);
            hlcc4.push((high_data[i] + low_data[i] + 2.0 * close_data[i]) / 4.0);
        }

        let candles = Candles {
            timestamp: vec![0; 20],
            open: open_data,
            high: high_data,
            low: low_data,
            close: close_data,
            volume: volume_data,
            hl2,
            hlc3,
            ohlc4,
            hlcc4,
        };

        let params = PercentileNearestRankParams {
            length: Some(5),
            percentage: Some(50.0),
        };
        let input = PercentileNearestRankInput::from_candles(&candles, "close", params);
        let result = percentile_nearest_rank(&input).unwrap();

        assert_eq!(result.values.len(), 20);
        // First 4 values should be NaN
        for i in 0..4 {
            assert!(result.values[i].is_nan());
        }
        // 50th percentile of [1.5,2.5,3.5,4.5,5.5] = 3.5
        assert_eq!(result.values[4], 3.5);
    }
}

#[cfg(feature = "python")]
pub fn register_percentile_nearest_rank_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(percentile_nearest_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_nearest_rank_batch_py, m)?)?;
    Ok(())
}
