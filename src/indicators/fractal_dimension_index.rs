#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for FractalDimensionIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            FractalDimensionIndexData::Slice(slice) => slice,
            FractalDimensionIndexData::Candles { candles } => candles.close.as_slice(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FractalDimensionIndexData<'a> {
    Candles { candles: &'a Candles },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FractalDimensionIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct FractalDimensionIndexParams {
    pub length: Option<usize>,
}

impl Default for FractalDimensionIndexParams {
    fn default() -> Self {
        Self { length: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct FractalDimensionIndexInput<'a> {
    pub data: FractalDimensionIndexData<'a>,
    pub params: FractalDimensionIndexParams,
}

impl<'a> FractalDimensionIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FractalDimensionIndexParams) -> Self {
        Self {
            data: FractalDimensionIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: FractalDimensionIndexParams) -> Self {
        Self {
            data: FractalDimensionIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FractalDimensionIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FractalDimensionIndexBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for FractalDimensionIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FractalDimensionIndexBuilder {
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
    ) -> Result<FractalDimensionIndexOutput, FractalDimensionIndexError> {
        let params = FractalDimensionIndexParams {
            length: self.length,
        };
        fractal_dimension_index_with_kernel(
            &FractalDimensionIndexInput::from_candles(candles, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<FractalDimensionIndexOutput, FractalDimensionIndexError> {
        let params = FractalDimensionIndexParams {
            length: self.length,
        };
        fractal_dimension_index_with_kernel(
            &FractalDimensionIndexInput::from_slice(data, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<FractalDimensionIndexStream, FractalDimensionIndexError> {
        FractalDimensionIndexStream::try_new(FractalDimensionIndexParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum FractalDimensionIndexError {
    #[error("fractal_dimension_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("fractal_dimension_index: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "fractal_dimension_index: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("fractal_dimension_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fractal_dimension_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("fractal_dimension_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("fractal_dimension_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "fractal_dimension_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("fractal_dimension_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct FractalDimensionIndexStream {
    length: usize,
    window: Vec<f64>,
    pos: usize,
    count: usize,
    tick: usize,
    min_q: VecDeque<(usize, f64)>,
    max_q: VecDeque<(usize, f64)>,
}

impl FractalDimensionIndexStream {
    #[inline(always)]
    pub fn try_new(
        params: FractalDimensionIndexParams,
    ) -> Result<Self, FractalDimensionIndexError> {
        let length = params.length.unwrap_or(30);
        if length < 2 {
            return Err(FractalDimensionIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }

        Ok(Self {
            length,
            window: vec![0.0; length],
            pos: 0,
            count: 0,
            tick: 0,
            min_q: VecDeque::with_capacity(length),
            max_q: VecDeque::with_capacity(length),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !is_valid_value(value) {
            self.pos = 0;
            self.count = 0;
            self.tick = 0;
            self.min_q.clear();
            self.max_q.clear();
            return None;
        }

        let idx = self.tick;
        self.tick += 1;

        self.window[self.pos] = value;
        self.pos += 1;
        if self.pos == self.length {
            self.pos = 0;
        }
        if self.count < self.length {
            self.count += 1;
        }

        while let Some((_, tail)) = self.min_q.back() {
            if *tail <= value {
                break;
            }
            self.min_q.pop_back();
        }
        self.min_q.push_back((idx, value));

        while let Some((_, tail)) = self.max_q.back() {
            if *tail >= value {
                break;
            }
            self.max_q.pop_back();
        }
        self.max_q.push_back((idx, value));

        if self.count < self.length {
            return None;
        }

        let start = idx + 1 - self.length;
        while let Some((old_idx, _)) = self.min_q.front() {
            if *old_idx >= start {
                break;
            }
            self.min_q.pop_front();
        }
        while let Some((old_idx, _)) = self.max_q.front() {
            if *old_idx >= start {
                break;
            }
            self.max_q.pop_front();
        }

        let low = self.min_q.front().map(|(_, v)| *v).unwrap_or(value);
        let high = self.max_q.front().map(|(_, v)| *v).unwrap_or(value);
        let length_sum = path_length_from_ring(&self.window, self.pos, self.length, low, high);
        Some(fdi_from_length(length_sum, self.length))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.length - 1
    }
}

#[inline(always)]
fn is_valid_value(value: f64) -> bool {
    value.is_finite()
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if is_valid_value(value) {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn build_invalid_prefix(data: &[f64]) -> Vec<u32> {
    let mut invalid = vec![0u32; data.len() + 1];
    for i in 0..data.len() {
        invalid[i + 1] = invalid[i] + u32::from(!is_valid_value(data[i]));
    }
    invalid
}

#[inline(always)]
fn fdi_from_length(length_sum: f64, length: usize) -> f64 {
    1.0 + (length_sum.ln() + std::f64::consts::LN_2) / ((2 * length) as f64).ln()
}

#[inline(always)]
fn fdi_from_length_with_den(length_sum: f64, ln_2_len: f64) -> f64 {
    1.0 + (length_sum.ln() + std::f64::consts::LN_2) / ln_2_len
}

#[inline(always)]
fn path_length_from_window(data: &[f64], start: usize, end: usize, low: f64, high: f64) -> f64 {
    let length = end - start + 1;
    let inv_n_sq = 1.0 / ((length * length) as f64);
    let flat_length = (length - 1) as f64 / length as f64;
    path_length_from_window_precomputed(data, start, end, low, high, inv_n_sq, flat_length)
}

#[inline(always)]
fn path_length_from_window_precomputed(
    data: &[f64],
    start: usize,
    end: usize,
    low: f64,
    high: f64,
    inv_n_sq: f64,
    flat_length: f64,
) -> f64 {
    let range = high - low;
    if !range.is_finite() || range <= 0.0 {
        return flat_length;
    }

    let mut prev = (data[start] - low) / range;
    let mut acc = 0.0;
    for &value in &data[(start + 1)..=end] {
        let cur = (value - low) / range;
        let delta = cur - prev;
        acc += (delta * delta + inv_n_sq).sqrt();
        prev = cur;
    }
    acc
}

#[inline(always)]
fn path_length_from_ring(
    window: &[f64],
    start_pos: usize,
    length: usize,
    low: f64,
    high: f64,
) -> f64 {
    let inv_n_sq = 1.0 / ((length * length) as f64);
    let range = high - low;
    if !range.is_finite() || range <= 0.0 {
        return (length - 1) as f64 / length as f64;
    }

    let first = window[start_pos];
    let mut prev = (first - low) / range;
    let mut acc = 0.0;
    for step in 1..length {
        let value = window[(start_pos + step) % length];
        let cur = (value - low) / range;
        let delta = cur - prev;
        acc += (delta * delta + inv_n_sq).sqrt();
        prev = cur;
    }
    acc
}

#[inline(always)]
fn compute_fdi_row(data: &[f64], length: usize, out: &mut [f64]) {
    let len = data.len();
    if len < length {
        return;
    }

    let inv_n_sq = 1.0 / ((length * length) as f64);
    let flat_length = (length - 1) as f64 / length as f64;
    let ln_2_len = ((2 * length) as f64).ln();
    let invalid = build_invalid_prefix(data);
    let mut min_q: VecDeque<usize> = VecDeque::with_capacity(length);
    let mut max_q: VecDeque<usize> = VecDeque::with_capacity(length);

    for i in 0..len {
        let value = data[i];
        if is_valid_value(value) {
            while let Some(&idx) = min_q.back() {
                if data[idx] <= value {
                    break;
                }
                min_q.pop_back();
            }
            min_q.push_back(i);

            while let Some(&idx) = max_q.back() {
                if data[idx] >= value {
                    break;
                }
                max_q.pop_back();
            }
            max_q.push_back(i);
        }

        if i + 1 < length {
            continue;
        }

        let start = i + 1 - length;
        while let Some(&idx) = min_q.front() {
            if idx >= start {
                break;
            }
            min_q.pop_front();
        }
        while let Some(&idx) = max_q.front() {
            if idx >= start {
                break;
            }
            max_q.pop_front();
        }

        if invalid[i + 1] - invalid[start] != 0 {
            continue;
        }

        let low = data[*min_q.front().unwrap()];
        let high = data[*max_q.front().unwrap()];
        let length_sum =
            path_length_from_window_precomputed(data, start, i, low, high, inv_n_sq, flat_length);
        out[i] = fdi_from_length_with_den(length_sum, ln_2_len);
    }
}

#[inline(always)]
fn compute_fdi_row_all_valid(data: &[f64], length: usize, out: &mut [f64]) {
    let len = data.len();
    if len < length {
        return;
    }

    let inv_n_sq = 1.0 / ((length * length) as f64);
    let flat_length = (length - 1) as f64 / length as f64;
    let ln_2_len = ((2 * length) as f64).ln();
    let mut min_q: VecDeque<usize> = VecDeque::with_capacity(length);
    let mut max_q: VecDeque<usize> = VecDeque::with_capacity(length);

    for i in 0..len {
        let value = data[i];
        while let Some(&idx) = min_q.back() {
            if data[idx] <= value {
                break;
            }
            min_q.pop_back();
        }
        min_q.push_back(i);

        while let Some(&idx) = max_q.back() {
            if data[idx] >= value {
                break;
            }
            max_q.pop_back();
        }
        max_q.push_back(i);

        if i + 1 < length {
            continue;
        }

        let start = i + 1 - length;
        while let Some(&idx) = min_q.front() {
            if idx >= start {
                break;
            }
            min_q.pop_front();
        }
        while let Some(&idx) = max_q.front() {
            if idx >= start {
                break;
            }
            max_q.pop_front();
        }

        let low = data[*min_q.front().unwrap()];
        let high = data[*max_q.front().unwrap()];
        let length_sum =
            path_length_from_window_precomputed(data, start, i, low, high, inv_n_sq, flat_length);
        out[i] = fdi_from_length_with_den(length_sum, ln_2_len);
    }
}

#[inline(always)]
fn validate_common(data: &[f64], length: usize) -> Result<bool, FractalDimensionIndexError> {
    let len = data.len();
    if len == 0 {
        return Err(FractalDimensionIndexError::EmptyInputData);
    }
    if length < 2 || length > len {
        return Err(FractalDimensionIndexError::InvalidLength {
            length,
            data_len: len,
        });
    }

    let mut max_run = 0usize;
    let mut current_run = 0usize;
    let mut all_valid = true;
    for &value in data {
        if is_valid_value(value) {
            current_run += 1;
            if current_run > max_run {
                max_run = current_run;
            }
        } else {
            all_valid = false;
            current_run = 0;
        }
    }
    if max_run == 0 {
        return Err(FractalDimensionIndexError::AllValuesNaN);
    }
    if max_run < length {
        return Err(FractalDimensionIndexError::NotEnoughValidData {
            needed: length,
            valid: max_run,
        });
    }
    Ok(all_valid)
}

#[inline]
pub fn fractal_dimension_index(
    input: &FractalDimensionIndexInput,
) -> Result<FractalDimensionIndexOutput, FractalDimensionIndexError> {
    fractal_dimension_index_with_kernel(input, Kernel::Auto)
}

pub fn fractal_dimension_index_with_kernel(
    input: &FractalDimensionIndexInput,
    kernel: Kernel,
) -> Result<FractalDimensionIndexOutput, FractalDimensionIndexError> {
    let data: &[f64] = input.as_ref();
    let length = input.get_length();
    let all_valid = validate_common(data, length)?;

    let mut values = alloc_with_nan_prefix(data.len(), length - 1);
    let _ = kernel;
    if all_valid {
        compute_fdi_row_all_valid(data, length, &mut values);
    } else {
        values.fill(f64::NAN);
        compute_fdi_row(data, length, &mut values);
    }
    Ok(FractalDimensionIndexOutput { values })
}

pub fn fractal_dimension_index_into_slice(
    dst: &mut [f64],
    input: &FractalDimensionIndexInput,
    kernel: Kernel,
) -> Result<(), FractalDimensionIndexError> {
    let data: &[f64] = input.as_ref();
    if dst.len() != data.len() {
        return Err(FractalDimensionIndexError::MismatchedOutputLen {
            dst_len: dst.len(),
            expected_len: data.len(),
        });
    }

    let length = input.get_length();
    let all_valid = validate_common(data, length)?;

    let _ = kernel;

    if all_valid {
        for value in &mut dst[..length - 1] {
            *value = f64::NAN;
        }
        compute_fdi_row_all_valid(data, length, dst);
    } else {
        dst.fill(f64::NAN);
        compute_fdi_row(data, length, dst);
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn fractal_dimension_index_into(
    input: &FractalDimensionIndexInput,
    out: &mut [f64],
) -> Result<(), FractalDimensionIndexError> {
    fractal_dimension_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct FractalDimensionIndexBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for FractalDimensionIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (30, 30, 0),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FractalDimensionIndexBatchBuilder {
    range: FractalDimensionIndexBatchRange,
    kernel: Kernel,
}

impl FractalDimensionIndexBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<FractalDimensionIndexBatchOutput, FractalDimensionIndexError> {
        fractal_dimension_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<FractalDimensionIndexBatchOutput, FractalDimensionIndexError> {
        self.apply_slice(&candles.close)
    }
}

#[derive(Debug, Clone)]
pub struct FractalDimensionIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FractalDimensionIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl FractalDimensionIndexBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &FractalDimensionIndexParams) -> Option<&[f64]> {
        self.combos
            .iter()
            .position(|p| p.length == params.length)
            .map(|row| {
                let start = row * self.cols;
                &self.values[start..start + self.cols]
            })
    }
}

fn expand_grid_checked(
    range: &FractalDimensionIndexBatchRange,
) -> Result<Vec<FractalDimensionIndexParams>, FractalDimensionIndexError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, FractalDimensionIndexError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.saturating_add(step.max(1));
                if next == cur || next > end {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.saturating_sub(step.max(1));
                if next == cur || next < end {
                    break;
                }
                cur = next;
            }
        }

        if out.is_empty() {
            return Err(FractalDimensionIndexError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let lengths = axis_usize(range.length)?;
    if lengths.iter().any(|&value| value < 2) {
        return Err(FractalDimensionIndexError::InvalidLength {
            length: 0,
            data_len: 0,
        });
    }

    let mut out = Vec::with_capacity(lengths.len());
    for &length in &lengths {
        out.push(FractalDimensionIndexParams {
            length: Some(length),
        });
    }
    Ok(out)
}

pub fn fractal_dimension_index_batch_with_kernel(
    data: &[f64],
    sweep: &FractalDimensionIndexBatchRange,
    kernel: Kernel,
) -> Result<FractalDimensionIndexBatchOutput, FractalDimensionIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(FractalDimensionIndexError::InvalidKernelForBatch(other)),
    };
    fractal_dimension_index_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn fractal_dimension_index_batch_slice(
    data: &[f64],
    sweep: &FractalDimensionIndexBatchRange,
    kernel: Kernel,
) -> Result<FractalDimensionIndexBatchOutput, FractalDimensionIndexError> {
    fractal_dimension_index_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn fractal_dimension_index_batch_par_slice(
    data: &[f64],
    sweep: &FractalDimensionIndexBatchRange,
    kernel: Kernel,
) -> Result<FractalDimensionIndexBatchOutput, FractalDimensionIndexError> {
    fractal_dimension_index_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn fractal_dimension_index_batch_inner(
    data: &[f64],
    sweep: &FractalDimensionIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<FractalDimensionIndexBatchOutput, FractalDimensionIndexError> {
    let combos = expand_grid_checked(sweep)?;
    if data.is_empty() {
        return Err(FractalDimensionIndexError::EmptyInputData);
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(FractalDimensionIndexError::AllValuesNaN);
    }

    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(30))
        .max()
        .unwrap_or(0);
    if max_length > data.len() {
        return Err(FractalDimensionIndexError::InvalidLength {
            length: max_length,
            data_len: data.len(),
        });
    }
    if max_run < max_length {
        return Err(FractalDimensionIndexError::NotEnoughValidData {
            needed: max_length,
            valid: max_run,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| FractalDimensionIndexError::InvalidInput {
            msg: "fractal_dimension_index: rows*cols overflow in batch".to_string(),
        })?;

    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.length.unwrap_or(30).saturating_sub(1))
        .collect();
    init_matrix_prefixes(&mut values_mu, cols, &warmups);

    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);

    debug_assert_eq!(values.len(), total);

    fractal_dimension_index_batch_inner_into(data, sweep, kernel, parallel, &mut values)?;

    Ok(FractalDimensionIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn fractal_dimension_index_batch_inner_into(
    data: &[f64],
    sweep: &FractalDimensionIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<FractalDimensionIndexParams>, FractalDimensionIndexError> {
    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(FractalDimensionIndexError::EmptyInputData);
    }

    let total =
        combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| FractalDimensionIndexError::InvalidInput {
                msg: "fractal_dimension_index: rows*cols overflow in batch_into".to_string(),
            })?;
    if out.len() != total {
        return Err(FractalDimensionIndexError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: total,
        });
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(FractalDimensionIndexError::AllValuesNaN);
    }
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(30))
        .max()
        .unwrap_or(0);
    if max_length > len {
        return Err(FractalDimensionIndexError::InvalidLength {
            length: max_length,
            data_len: len,
        });
    }
    if max_run < max_length {
        return Err(FractalDimensionIndexError::NotEnoughValidData {
            needed: max_length,
            valid: max_run,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let worker = |row: usize, dst: &mut [f64]| {
        dst.fill(f64::NAN);
        let length = combos[row].length.unwrap_or(30);
        compute_fdi_row(data, length, dst);
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out.par_chunks_mut(len)
            .enumerate()
            .for_each(|(row, dst)| worker(row, dst));
    } else {
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn expand_grid_fractal_dimension_index(
    range: &FractalDimensionIndexBatchRange,
) -> Vec<FractalDimensionIndexParams> {
    expand_grid_checked(range).unwrap_or_default()
}

#[cfg(feature = "python")]
#[pyfunction(name = "fractal_dimension_index")]
#[pyo3(signature = (data, length=30, kernel=None))]
pub fn fractal_dimension_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = FractalDimensionIndexInput::from_slice(
        slice_in,
        FractalDimensionIndexParams {
            length: Some(length),
        },
    );
    let out = py
        .allow_threads(|| fractal_dimension_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "FractalDimensionIndexStream")]
pub struct FractalDimensionIndexStreamPy {
    stream: FractalDimensionIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl FractalDimensionIndexStreamPy {
    #[new]
    fn new(length: usize) -> PyResult<Self> {
        let stream = FractalDimensionIndexStream::try_new(FractalDimensionIndexParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "fractal_dimension_index_batch")]
#[pyo3(signature = (data, length_range=(30,30,0), kernel=None))]
pub fn fractal_dimension_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = FractalDimensionIndexBatchRange {
        length: length_range,
    };

    let output = py
        .allow_threads(|| fractal_dimension_index_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = output.rows;
    let cols = output.cols;
    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        output.values.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(30) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_fractal_dimension_index_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fractal_dimension_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(fractal_dimension_index_batch_py, m)?)?;
    m.add_class::<FractalDimensionIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = fractal_dimension_index_js)]
pub fn fractal_dimension_index_js(data: &[f64], length: usize) -> Result<JsValue, JsValue> {
    let input = FractalDimensionIndexInput::from_slice(
        data,
        FractalDimensionIndexParams {
            length: Some(length),
        },
    );
    let out = fractal_dimension_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out.values).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalDimensionIndexBatchConfig {
    pub length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = fractal_dimension_index_batch_js)]
pub fn fractal_dimension_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: FractalDimensionIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;

    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = FractalDimensionIndexBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let out = fractal_dimension_index_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("values"),
        &serde_wasm_bindgen::to_value(&out.values).unwrap(),
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
pub fn fractal_dimension_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fractal_dimension_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fractal_dimension_index_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to fractal_dimension_index_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = FractalDimensionIndexInput::from_slice(
            data,
            FractalDimensionIndexParams {
                length: Some(length),
            },
        );
        fractal_dimension_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fractal_dimension_index_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to fractal_dimension_index_batch_into",
        ));
    }

    let sweep = FractalDimensionIndexBatchRange {
        length: (length_start, length_end, length_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in fractal_dimension_index_batch_into")
    })?;

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        fractal_dimension_index_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fractal_dimension_index_output_into_js(
    data: &[f64],
    length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = fractal_dimension_index_js(data, length)?;
    crate::write_wasm_object_f64_outputs("fractal_dimension_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fractal_dimension_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = fractal_dimension_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "fractal_dimension_index_batch_output_into_js",
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

    fn sample_close(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + 0.18 * x + 2.0 * (x * 0.11).sin() + 0.7 * (x * 0.037).cos()
            })
            .collect()
    }

    fn naive_fdi(close: &[f64], length: usize) -> Vec<f64> {
        let len = close.len();
        let mut out = vec![f64::NAN; len];
        if length > len {
            return out;
        }

        for i in (length - 1)..len {
            let start = i + 1 - length;
            let window = &close[start..=i];
            if window.iter().any(|v| !is_valid_value(*v)) {
                continue;
            }
            let low = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let high = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let path = path_length_from_window(close, start, i, low, high);
            out[i] = fdi_from_length(path, length);
        }
        out
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn fractal_dimension_index_matches_naive() -> Result<(), Box<dyn Error>> {
        let close = sample_close(256);
        let input = FractalDimensionIndexInput::from_slice(
            &close,
            FractalDimensionIndexParams { length: Some(30) },
        );
        let out = fractal_dimension_index_with_kernel(&input, Kernel::Scalar)?;
        let expected = naive_fdi(&close, 30);
        assert_series_close(&out.values, &expected, 1e-12);
        Ok(())
    }

    #[test]
    fn fractal_dimension_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let close = sample_close(192);
        let input = FractalDimensionIndexInput::from_slice(
            &close,
            FractalDimensionIndexParams { length: Some(24) },
        );
        let baseline = fractal_dimension_index_with_kernel(&input, Kernel::Auto)?;
        let mut out = vec![0.0; close.len()];
        fractal_dimension_index_into_slice(&mut out, &input, Kernel::Auto)?;
        assert_series_close(&baseline.values, &out, 1e-12);
        Ok(())
    }

    #[test]
    fn fractal_dimension_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = sample_close(300);
        let params = FractalDimensionIndexParams { length: Some(21) };
        let batch = fractal_dimension_index(&FractalDimensionIndexInput::from_slice(
            &close,
            params.clone(),
        ))?;
        let mut stream = FractalDimensionIndexStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(close.len());

        for &value in &close {
            if let Some(out) = stream.update(value) {
                streamed.push(out);
            } else {
                streamed.push(f64::NAN);
            }
        }

        assert_series_close(&batch.values, &streamed, 1e-12);
        Ok(())
    }

    #[test]
    fn fractal_dimension_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let close = sample_close(220);
        let batch = fractal_dimension_index_batch_with_kernel(
            &close,
            &FractalDimensionIndexBatchRange {
                length: (30, 30, 0),
            },
            Kernel::ScalarBatch,
        )?;
        let single = fractal_dimension_index(&FractalDimensionIndexInput::from_slice(
            &close,
            FractalDimensionIndexParams { length: Some(30) },
        ))?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_close(&batch.values, &single.values, 1e-12);
        Ok(())
    }

    #[test]
    fn fractal_dimension_index_rejects_invalid_params() {
        let close = sample_close(32);
        let bad = FractalDimensionIndexInput::from_slice(
            &close,
            FractalDimensionIndexParams { length: Some(1) },
        );
        assert!(matches!(
            fractal_dimension_index(&bad),
            Err(FractalDimensionIndexError::InvalidLength { .. })
        ));
    }

    #[test]
    fn fractal_dimension_index_dispatch_compute_returns_value() -> Result<(), Box<dyn Error>> {
        let close = sample_close(180);
        let params = [ParamKV {
            key: "length",
            value: ParamValue::Int(30),
        }];
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "fractal_dimension_index",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            params: &params,
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out.output_id, "value");
        Ok(())
    }
}
