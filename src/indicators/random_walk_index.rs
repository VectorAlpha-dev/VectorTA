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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 14;

#[derive(Debug, Clone)]
pub enum RandomWalkIndexData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct RandomWalkIndexOutput {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RandomWalkIndexParams {
    pub length: Option<usize>,
}

impl Default for RandomWalkIndexParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RandomWalkIndexInput<'a> {
    pub data: RandomWalkIndexData<'a>,
    pub params: RandomWalkIndexParams,
}

impl<'a> RandomWalkIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: RandomWalkIndexParams) -> Self {
        Self {
            data: RandomWalkIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: RandomWalkIndexParams,
    ) -> Self {
        Self {
            data: RandomWalkIndexData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, RandomWalkIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RandomWalkIndexBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for RandomWalkIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RandomWalkIndexBuilder {
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
    pub fn apply(self, candles: &Candles) -> Result<RandomWalkIndexOutput, RandomWalkIndexError> {
        let input = RandomWalkIndexInput::from_candles(
            candles,
            RandomWalkIndexParams {
                length: self.length,
            },
        );
        random_walk_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<RandomWalkIndexOutput, RandomWalkIndexError> {
        let input = RandomWalkIndexInput::from_slices(
            high,
            low,
            close,
            RandomWalkIndexParams {
                length: self.length,
            },
        );
        random_walk_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RandomWalkIndexStream, RandomWalkIndexError> {
        RandomWalkIndexStream::try_new(RandomWalkIndexParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum RandomWalkIndexError {
    #[error("random_walk_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("random_walk_index: All values are NaN.")]
    AllValuesNaN,
    #[error("random_walk_index: Inconsistent slice lengths: high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("random_walk_index: Invalid length: length={length}, data length={data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("random_walk_index: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("random_walk_index: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("random_walk_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("random_walk_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn extract_hlc<'a>(
    input: &'a RandomWalkIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), RandomWalkIndexError> {
    let (high, low, close) = match &input.data {
        RandomWalkIndexData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        RandomWalkIndexData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(RandomWalkIndexError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(RandomWalkIndexError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    Ok((high, low, close))
}

#[inline(always)]
fn first_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..high.len()).find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
}

#[inline(always)]
fn prepare<'a>(
    input: &'a RandomWalkIndexInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), RandomWalkIndexError> {
    let (high, low, close) = extract_hlc(input)?;
    let len = close.len();
    let length = input.get_length();
    if length == 0 || length > len {
        return Err(RandomWalkIndexError::InvalidLength {
            length,
            data_len: len,
        });
    }
    let first = first_valid_hlc(high, low, close).ok_or(RandomWalkIndexError::AllValuesNaN)?;
    let valid = len.saturating_sub(first);
    if valid < length {
        return Err(RandomWalkIndexError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }
    Ok((high, low, close, length, first, kernel.to_non_batch()))
}

#[inline(always)]
fn nz_history(src: &[f64], idx: usize, offset: usize) -> f64 {
    if idx >= offset {
        let value = src[idx - offset];
        if value.is_finite() {
            value
        } else {
            0.0
        }
    } else {
        0.0
    }
}

#[inline(always)]
unsafe fn nz_history_14_ptr(src: *const f64, idx: usize) -> f64 {
    if idx >= DEFAULT_LENGTH {
        let value = unsafe { *src.add(idx - DEFAULT_LENGTH) };
        if value.is_finite() {
            value
        } else {
            0.0
        }
    } else {
        0.0
    }
}

#[inline(always)]
fn compute_random_walk_index_14_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    out_high: &mut [f64],
    out_low: &mut [f64],
) {
    let n = close.len();
    let warm = first + DEFAULT_LENGTH - 1;
    let sqrt_length = (DEFAULT_LENGTH as f64).sqrt();
    let alpha = 1.0 / DEFAULT_LENGTH as f64;
    unsafe {
        let high_ptr = high.as_ptr();
        let low_ptr = low.as_ptr();
        let close_ptr = close.as_ptr();
        let out_high_ptr = out_high.as_mut_ptr();
        let out_low_ptr = out_low.as_mut_ptr();
        let mut prev_close = *close_ptr.add(first);
        let mut sum_tr = *high_ptr.add(first) - *low_ptr.add(first);
        let mut i = first + 1;

        while i < warm {
            let h = *high_ptr.add(i);
            let l = *low_ptr.add(i);
            let tr = (h - l)
                .max((h - prev_close).abs())
                .max((l - prev_close).abs());
            sum_tr += tr;
            prev_close = *close_ptr.add(i);
            i += 1;
        }

        let h = *high_ptr.add(warm);
        let l = *low_ptr.add(warm);
        let tr = (h - l)
            .max((h - prev_close).abs())
            .max((l - prev_close).abs());
        sum_tr += tr;
        let mut atr = sum_tr / DEFAULT_LENGTH as f64;
        let denom = atr * sqrt_length;
        if denom.is_finite() && denom != 0.0 {
            *out_high_ptr.add(warm) = (h - nz_history_14_ptr(low_ptr, warm)) / denom;
            *out_low_ptr.add(warm) = (nz_history_14_ptr(high_ptr, warm) - l) / denom;
        } else {
            *out_high_ptr.add(warm) = f64::NAN;
            *out_low_ptr.add(warm) = f64::NAN;
        }
        prev_close = *close_ptr.add(warm);
        i = warm + 1;

        while i < n {
            let h = *high_ptr.add(i);
            let l = *low_ptr.add(i);
            let tr = (h - l)
                .max((h - prev_close).abs())
                .max((l - prev_close).abs());
            atr = alpha.mul_add(tr - atr, atr);
            let denom = atr * sqrt_length;
            if denom.is_finite() && denom != 0.0 {
                *out_high_ptr.add(i) = (h - nz_history_14_ptr(low_ptr, i)) / denom;
                *out_low_ptr.add(i) = (nz_history_14_ptr(high_ptr, i) - l) / denom;
            } else {
                *out_high_ptr.add(i) = f64::NAN;
                *out_low_ptr.add(i) = f64::NAN;
            }

            prev_close = *close_ptr.add(i);
            i += 1;
        }
    }
}

#[inline(always)]
fn compute_random_walk_index_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    first: usize,
    out_high: &mut [f64],
    out_low: &mut [f64],
) {
    if length == DEFAULT_LENGTH {
        compute_random_walk_index_14_into(high, low, close, first, out_high, out_low);
        return;
    }

    let n = close.len();
    let warm = first + length - 1;
    let sqrt_length = (length as f64).sqrt();
    let alpha = 1.0 / length as f64;

    let mut prev_close = close[first];
    let mut sum_tr = high[first] - low[first];
    let mut atr = f64::NAN;

    if length == 1 {
        atr = sum_tr;
        let denom = atr * sqrt_length;
        if denom.is_finite() && denom != 0.0 {
            out_high[first] = (high[first] - nz_history(low, first, length)) / denom;
            out_low[first] = (nz_history(high, first, length) - low[first]) / denom;
        }
        let mut i = first + 1;
        while i < n {
            let tr = (high[i] - low[i])
                .max((high[i] - prev_close).abs())
                .max((low[i] - prev_close).abs());
            atr = alpha.mul_add(tr - atr, atr);
            let denom = atr * sqrt_length;
            if denom.is_finite() && denom != 0.0 {
                out_high[i] = (high[i] - nz_history(low, i, length)) / denom;
                out_low[i] = (nz_history(high, i, length) - low[i]) / denom;
            } else {
                out_high[i] = f64::NAN;
                out_low[i] = f64::NAN;
            }

            prev_close = close[i];
            i += 1;
        }
        return;
    }

    let mut i = first + 1;
    while i < warm {
        let tr = (high[i] - low[i])
            .max((high[i] - prev_close).abs())
            .max((low[i] - prev_close).abs());
        sum_tr += tr;
        prev_close = close[i];
        i += 1;
    }

    let tr = (high[warm] - low[warm])
        .max((high[warm] - prev_close).abs())
        .max((low[warm] - prev_close).abs());
    sum_tr += tr;
    atr = sum_tr / length as f64;
    let denom = atr * sqrt_length;
    if denom.is_finite() && denom != 0.0 {
        out_high[warm] = (high[warm] - nz_history(low, warm, length)) / denom;
        out_low[warm] = (nz_history(high, warm, length) - low[warm]) / denom;
    } else {
        out_high[warm] = f64::NAN;
        out_low[warm] = f64::NAN;
    }
    prev_close = close[warm];
    i = warm + 1;

    while i < n {
        let tr = (high[i] - low[i])
            .max((high[i] - prev_close).abs())
            .max((low[i] - prev_close).abs());
        atr = alpha.mul_add(tr - atr, atr);
        let denom = atr * sqrt_length;
        if denom.is_finite() && denom != 0.0 {
            out_high[i] = (high[i] - nz_history(low, i, length)) / denom;
            out_low[i] = (nz_history(high, i, length) - low[i]) / denom;
        } else {
            out_high[i] = f64::NAN;
            out_low[i] = f64::NAN;
        }

        prev_close = close[i];
        i += 1;
    }
}

#[inline]
pub fn random_walk_index(
    input: &RandomWalkIndexInput,
) -> Result<RandomWalkIndexOutput, RandomWalkIndexError> {
    random_walk_index_with_kernel(input, Kernel::Auto)
}

pub fn random_walk_index_with_kernel(
    input: &RandomWalkIndexInput,
    kernel: Kernel,
) -> Result<RandomWalkIndexOutput, RandomWalkIndexError> {
    let (high, low, close, length, first, _) = prepare(input, kernel)?;
    let warm = first + length - 1;
    let mut out_high = alloc_with_nan_prefix(close.len(), warm);
    let mut out_low = alloc_with_nan_prefix(close.len(), warm);
    compute_random_walk_index_into(high, low, close, length, first, &mut out_high, &mut out_low);
    Ok(RandomWalkIndexOutput {
        high: out_high,
        low: out_low,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn random_walk_index_into(
    out_high: &mut [f64],
    out_low: &mut [f64],
    input: &RandomWalkIndexInput,
    kernel: Kernel,
) -> Result<(), RandomWalkIndexError> {
    random_walk_index_into_slice(out_high, out_low, input, kernel)
}

pub fn random_walk_index_into_slice(
    out_high: &mut [f64],
    out_low: &mut [f64],
    input: &RandomWalkIndexInput,
    kernel: Kernel,
) -> Result<(), RandomWalkIndexError> {
    let (high, low, close, length, first, _) = prepare(input, kernel)?;
    let expected = close.len();
    if out_high.len() != expected || out_low.len() != expected {
        return Err(RandomWalkIndexError::OutputLengthMismatch {
            expected,
            got: out_high.len().max(out_low.len()),
        });
    }
    let warm = first + length - 1;
    out_high[..warm.min(expected)].fill(f64::NAN);
    out_low[..warm.min(expected)].fill(f64::NAN);
    compute_random_walk_index_into(high, low, close, length, first, out_high, out_low);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct RandomWalkIndexStream {
    length: usize,
    sqrt_length: f64,
    count: usize,
    warm_sum: f64,
    atr: f64,
    prev_close: f64,
    history_high: VecDeque<f64>,
    history_low: VecDeque<f64>,
}

impl RandomWalkIndexStream {
    pub fn try_new(params: RandomWalkIndexParams) -> Result<Self, RandomWalkIndexError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length == 0 {
            return Err(RandomWalkIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self {
            length,
            sqrt_length: (length as f64).sqrt(),
            count: 0,
            warm_sum: 0.0,
            atr: f64::NAN,
            prev_close: f64::NAN,
            history_high: VecDeque::with_capacity(length),
            history_low: VecDeque::with_capacity(length),
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            return (f64::NAN, f64::NAN);
        }

        let tr = if self.count == 0 {
            high - low
        } else {
            (high - low)
                .max((high - self.prev_close).abs())
                .max((low - self.prev_close).abs())
        };

        if self.count < self.length {
            self.warm_sum += tr;
            self.count += 1;
            if self.count == self.length {
                self.atr = self.warm_sum / self.length as f64;
            }
        } else {
            let alpha = 1.0 / self.length as f64;
            self.atr = alpha.mul_add(tr - self.atr, self.atr);
            self.count += 1;
        }

        let hist_high = if self.history_high.len() == self.length {
            self.history_high.front().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let hist_low = if self.history_low.len() == self.length {
            self.history_low.front().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let denom = self.atr * self.sqrt_length;
        let out = if self.count >= self.length && denom.is_finite() && denom != 0.0 {
            ((high - hist_low) / denom, (hist_high - low) / denom)
        } else {
            (f64::NAN, f64::NAN)
        };

        self.history_high.push_back(high);
        self.history_low.push_back(low);
        if self.history_high.len() > self.length {
            self.history_high.pop_front();
        }
        if self.history_low.len() > self.length {
            self.history_low.pop_front();
        }
        self.prev_close = close;

        out
    }
}

#[derive(Debug, Clone)]
pub struct RandomWalkIndexBatchRange {
    pub length: (usize, usize, usize),
}

#[derive(Debug, Clone)]
pub struct RandomWalkIndexBatchOutput {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub combos: Vec<RandomWalkIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct RandomWalkIndexBatchBuilder {
    length: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for RandomWalkIndexBatchBuilder {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl RandomWalkIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.length = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<RandomWalkIndexBatchOutput, RandomWalkIndexError> {
        random_walk_index_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &RandomWalkIndexBatchRange {
                length: self.length,
            },
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<RandomWalkIndexBatchOutput, RandomWalkIndexError> {
        random_walk_index_batch_with_kernel(
            high,
            low,
            close,
            &RandomWalkIndexBatchRange {
                length: self.length,
            },
            self.kernel,
        )
    }
}

pub fn expand_grid(
    sweep: &RandomWalkIndexBatchRange,
) -> Result<Vec<RandomWalkIndexParams>, RandomWalkIndexError> {
    let (start, end, step) = sweep.length;
    if start == 0 {
        return Err(RandomWalkIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut lengths = Vec::new();
    if step == 0 {
        if start != end {
            return Err(RandomWalkIndexError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        lengths.push(start);
    } else {
        if start > end {
            return Err(RandomWalkIndexError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut current = start;
        while current <= end {
            lengths.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    }

    Ok(lengths
        .into_iter()
        .map(|length| RandomWalkIndexParams {
            length: Some(length),
        })
        .collect())
}

pub fn random_walk_index_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RandomWalkIndexBatchRange,
    kernel: Kernel,
) -> Result<RandomWalkIndexBatchOutput, RandomWalkIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(RandomWalkIndexError::InvalidKernelForBatch(kernel)),
    };
    random_walk_index_batch_par_slice(high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn random_walk_index_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RandomWalkIndexBatchRange,
    kernel: Kernel,
) -> Result<RandomWalkIndexBatchOutput, RandomWalkIndexError> {
    random_walk_index_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn random_walk_index_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RandomWalkIndexBatchRange,
    kernel: Kernel,
) -> Result<RandomWalkIndexBatchOutput, RandomWalkIndexError> {
    random_walk_index_batch_inner(high, low, close, sweep, kernel, true)
}

fn validate_raw_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<usize, RandomWalkIndexError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(RandomWalkIndexError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(RandomWalkIndexError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    first_valid_hlc(high, low, close).ok_or(RandomWalkIndexError::AllValuesNaN)
}

fn random_walk_index_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RandomWalkIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<RandomWalkIndexBatchOutput, RandomWalkIndexError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(high, low, close)?;
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap())
        .max()
        .unwrap();
    let valid = close.len().saturating_sub(first);
    if valid < max_length {
        return Err(RandomWalkIndexError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let rows = combos.len();
    let cols = close.len();
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.length.unwrap() - 1)
        .collect();

    let mut high_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut high_buf, cols, &warmups);
    let mut high_guard = ManuallyDrop::new(high_buf);
    let out_high: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(high_guard.as_mut_ptr() as *mut f64, high_guard.len())
    };

    let mut low_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut low_buf, cols, &warmups);
    let mut low_guard = ManuallyDrop::new(low_buf);
    let out_low: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(low_guard.as_mut_ptr() as *mut f64, low_guard.len())
    };

    random_walk_index_batch_inner_into(
        high, low, close, sweep, kernel, parallel, out_high, out_low,
    )?;

    let high_values = unsafe {
        Vec::from_raw_parts(
            high_guard.as_mut_ptr() as *mut f64,
            high_guard.len(),
            high_guard.capacity(),
        )
    };
    let low_values = unsafe {
        Vec::from_raw_parts(
            low_guard.as_mut_ptr() as *mut f64,
            low_guard.len(),
            low_guard.capacity(),
        )
    };

    Ok(RandomWalkIndexBatchOutput {
        high: high_values,
        low: low_values,
        combos,
        rows,
        cols,
    })
}

pub fn random_walk_index_batch_into_slice(
    out_high: &mut [f64],
    out_low: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RandomWalkIndexBatchRange,
    kernel: Kernel,
) -> Result<(), RandomWalkIndexError> {
    random_walk_index_batch_inner_into(high, low, close, sweep, kernel, false, out_high, out_low)?;
    Ok(())
}

fn random_walk_index_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RandomWalkIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_high: &mut [f64],
    out_low: &mut [f64],
) -> Result<Vec<RandomWalkIndexParams>, RandomWalkIndexError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(high, low, close)?;
    let rows = combos.len();
    let cols = close.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| RandomWalkIndexError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;
    if out_high.len() != expected || out_low.len() != expected {
        return Err(RandomWalkIndexError::OutputLengthMismatch {
            expected,
            got: out_high.len().max(out_low.len()),
        });
    }
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap())
        .max()
        .unwrap();
    let valid = cols.saturating_sub(first);
    if valid < max_length {
        return Err(RandomWalkIndexError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let do_row = |row: usize, dst_high: &mut [f64], dst_low: &mut [f64]| {
        let length = combos[row].length.unwrap();
        let warm = first + length - 1;
        dst_high[..warm.min(cols)].fill(f64::NAN);
        dst_low[..warm.min(cols)].fill(f64::NAN);
        compute_random_walk_index_into(high, low, close, length, first, dst_high, dst_low);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_high
                .par_chunks_mut(cols)
                .zip(out_low.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (dst_high, dst_low))| do_row(row, dst_high, dst_low));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for ((row, dst_high), dst_low) in out_high
                .chunks_mut(cols)
                .enumerate()
                .zip(out_low.chunks_mut(cols))
            {
                do_row(row, dst_high, dst_low);
            }
        }
    } else {
        for ((row, dst_high), dst_low) in out_high
            .chunks_mut(cols)
            .enumerate()
            .zip(out_low.chunks_mut(cols))
        {
            do_row(row, dst_high, dst_low);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "random_walk_index")]
#[pyo3(signature = (high, low, close, length=14, kernel=None))]
pub fn random_walk_index_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = RandomWalkIndexInput::from_slices(
        high,
        low,
        close,
        RandomWalkIndexParams {
            length: Some(length),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| random_walk_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("high", out.high.into_pyarray(py))?;
    dict.set_item("low", out.low.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "RandomWalkIndexStream")]
pub struct RandomWalkIndexStreamPy {
    stream: RandomWalkIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RandomWalkIndexStreamPy {
    #[new]
    #[pyo3(signature = (length=14))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = RandomWalkIndexStream::try_new(RandomWalkIndexParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "random_walk_index_batch")]
#[pyo3(signature = (high, low, close, length_range=(14,14,0), kernel=None))]
pub fn random_walk_index_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = RandomWalkIndexBatchRange {
        length: length_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_high = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_low = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let high_slice = unsafe { out_high.as_slice_mut()? };
    let low_slice = unsafe { out_low.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        random_walk_index_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            high_slice,
            low_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("high", out_high.reshape((rows, cols))?)?;
    dict.set_item("low", out_low.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_random_walk_index_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(random_walk_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(random_walk_index_batch_py, m)?)?;
    m.add_class::<RandomWalkIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RandomWalkIndexJsOutput {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "random_walk_index_js")]
pub fn random_walk_index_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
) -> Result<JsValue, JsValue> {
    let input = RandomWalkIndexInput::from_slices(
        high,
        low,
        close,
        RandomWalkIndexParams {
            length: Some(length),
        },
    );
    let out = random_walk_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&RandomWalkIndexJsOutput {
        high: out.high,
        low: out.low,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RandomWalkIndexBatchConfig {
    pub length_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RandomWalkIndexBatchJsOutput {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (i, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a finite non-negative whole number"
            )));
        }
        let rounded = value.round();
        if (value - rounded).abs() > 1e-9 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a whole number"
            )));
        }
        out[i] = rounded as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "random_walk_index_batch_js")]
pub fn random_walk_index_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: RandomWalkIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = RandomWalkIndexBatchRange {
        length: js_vec3_to_usize("length_range", &config.length_range)?,
    };
    let out = random_walk_index_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let lengths = out
        .combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
        .collect();
    serde_wasm_bindgen::to_value(&RandomWalkIndexBatchJsOutput {
        high: out.high,
        low: out.low,
        lengths,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn random_walk_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn random_walk_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn random_walk_index_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_high_ptr: *mut f64,
    out_low_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_high_ptr.is_null()
        || out_low_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out_high = std::slice::from_raw_parts_mut(out_high_ptr, len);
        let out_low = std::slice::from_raw_parts_mut(out_low_ptr, len);
        let input = RandomWalkIndexInput::from_slices(
            high,
            low,
            close,
            RandomWalkIndexParams {
                length: Some(length),
            },
        );
        random_walk_index_into_slice(out_high, out_low, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn random_walk_index_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_high_ptr: *mut f64,
    out_low_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_high_ptr.is_null()
        || out_low_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to random_walk_index_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = RandomWalkIndexBatchRange {
            length: (length_start, length_end, length_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in random_walk_index_batch_into")
        })?;
        let out_high = std::slice::from_raw_parts_mut(out_high_ptr, total);
        let out_low = std::slice::from_raw_parts_mut(out_low_ptr, total);
        random_walk_index_batch_into_slice(
            out_high,
            out_low,
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn random_walk_index_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = random_walk_index_js(high, low, close, length)?;
    crate::write_wasm_object_f64_outputs("random_walk_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn random_walk_index_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = random_walk_index_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "random_walk_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manual_random_walk_index(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut out_high = vec![f64::NAN; n];
        let mut out_low = vec![f64::NAN; n];
        let first = first_valid_hlc(high, low, close).unwrap();
        compute_random_walk_index_into(
            high,
            low,
            close,
            length,
            first,
            &mut out_high,
            &mut out_low,
        );
        (out_high, out_low)
    }

    fn sample_hlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.19).sin() * 2.0 + (i as f64) * 0.03)
            .collect();
        let high: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 1.5 + ((i as f64) * 0.11).cos().abs())
            .collect();
        let low: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - 1.3 - ((i as f64) * 0.07).sin().abs())
            .collect();
        (high, low, close)
    }

    fn assert_close(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            let diff = (a - b).abs();
            assert!(diff <= 1e-12, "mismatch at {idx}: {a} vs {b}");
        }
    }

    #[test]
    fn manual_reference_matches_api() {
        let (high, low, close) = sample_hlc(128);
        let input = RandomWalkIndexInput::from_slices(
            &high,
            &low,
            &close,
            RandomWalkIndexParams { length: Some(14) },
        );
        let out = random_walk_index(&input).unwrap();
        let (want_high, want_low) = manual_random_walk_index(&high, &low, &close, 14);
        assert_close(&out.high, &want_high);
        assert_close(&out.low, &want_low);
    }

    #[test]
    fn stream_matches_batch() {
        let (high, low, close) = sample_hlc(96);
        let input = RandomWalkIndexInput::from_slices(
            &high,
            &low,
            &close,
            RandomWalkIndexParams { length: Some(14) },
        );
        let out = random_walk_index(&input).unwrap();
        let mut stream =
            RandomWalkIndexStream::try_new(RandomWalkIndexParams { length: Some(14) }).unwrap();
        let mut got_high = Vec::with_capacity(high.len());
        let mut got_low = Vec::with_capacity(high.len());
        for i in 0..high.len() {
            let (h, l) = stream.update(high[i], low[i], close[i]);
            got_high.push(h);
            got_low.push(l);
        }
        assert_close(&out.high, &got_high);
        assert_close(&out.low, &got_low);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (high, low, close) = sample_hlc(80);
        let batch = random_walk_index_batch_with_kernel(
            &high,
            &low,
            &close,
            &RandomWalkIndexBatchRange {
                length: (14, 16, 2),
            },
            Kernel::Auto,
        )
        .unwrap();
        let input = RandomWalkIndexInput::from_slices(
            &high,
            &low,
            &close,
            RandomWalkIndexParams { length: Some(14) },
        );
        let single = random_walk_index(&input).unwrap();
        assert_eq!(batch.rows, 2);
        assert_close(&batch.high[..80], single.high.as_slice());
        assert_close(&batch.low[..80], single.low.as_slice());
    }

    #[test]
    fn into_slice_matches_single() {
        let (high, low, close) = sample_hlc(72);
        let input = RandomWalkIndexInput::from_slices(
            &high,
            &low,
            &close,
            RandomWalkIndexParams { length: Some(14) },
        );
        let single = random_walk_index(&input).unwrap();
        let mut out_high = vec![0.0; close.len()];
        let mut out_low = vec![0.0; close.len()];
        random_walk_index_into_slice(&mut out_high, &mut out_low, &input, Kernel::Auto).unwrap();
        assert_close(&single.high, &out_high);
        assert_close(&single.low, &out_low);
    }

    #[test]
    fn invalid_length_is_rejected() {
        let (high, low, close) = sample_hlc(8);
        let input = RandomWalkIndexInput::from_slices(
            &high,
            &low,
            &close,
            RandomWalkIndexParams { length: Some(0) },
        );
        assert!(matches!(
            random_walk_index(&input),
            Err(RandomWalkIndexError::InvalidLength { .. })
        ));
    }
}
