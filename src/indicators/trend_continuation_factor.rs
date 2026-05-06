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

impl<'a> AsRef<[f64]> for TrendContinuationFactorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrendContinuationFactorData::Candles { candles, source } => match *source {
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                "hlcc4" | "hlcc" => &candles.hlcc4,
                _ => source_type(candles, source),
            },
            TrendContinuationFactorData::Slice(slice) => slice,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrendContinuationFactorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrendContinuationFactorOutput {
    pub plus_tcf: Vec<f64>,
    pub minus_tcf: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TrendContinuationFactorParams {
    pub length: Option<usize>,
}

impl Default for TrendContinuationFactorParams {
    fn default() -> Self {
        Self { length: Some(35) }
    }
}

#[derive(Debug, Clone)]
pub struct TrendContinuationFactorInput<'a> {
    pub data: TrendContinuationFactorData<'a>,
    pub params: TrendContinuationFactorParams,
}

impl<'a> TrendContinuationFactorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: TrendContinuationFactorParams,
    ) -> Self {
        Self {
            data: TrendContinuationFactorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: TrendContinuationFactorParams) -> Self {
        Self {
            data: TrendContinuationFactorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", TrendContinuationFactorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(35)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrendContinuationFactorBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for TrendContinuationFactorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrendContinuationFactorBuilder {
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
    ) -> Result<TrendContinuationFactorOutput, TrendContinuationFactorError> {
        let input = TrendContinuationFactorInput::from_candles(
            candles,
            "close",
            TrendContinuationFactorParams {
                length: self.length,
            },
        );
        trend_continuation_factor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<TrendContinuationFactorOutput, TrendContinuationFactorError> {
        let input = TrendContinuationFactorInput::from_slice(
            data,
            TrendContinuationFactorParams {
                length: self.length,
            },
        );
        trend_continuation_factor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<TrendContinuationFactorStream, TrendContinuationFactorError> {
        TrendContinuationFactorStream::try_new(TrendContinuationFactorParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum TrendContinuationFactorError {
    #[error("trend_continuation_factor: Input data slice is empty.")]
    EmptyInputData,
    #[error("trend_continuation_factor: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "trend_continuation_factor: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "trend_continuation_factor: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "trend_continuation_factor: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("trend_continuation_factor: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("trend_continuation_factor: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn first_valid_index(data: &[f64]) -> Option<usize> {
    data.iter().position(|x| x.is_finite())
}

#[inline(always)]
fn trend_continuation_factor_prepare<'a>(
    input: &'a TrendContinuationFactorInput,
) -> Result<(&'a [f64], usize, usize), TrendContinuationFactorError> {
    let data = input.as_ref();
    let data_len = data.len();
    if data_len == 0 {
        return Err(TrendContinuationFactorError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(TrendContinuationFactorError::AllValuesNaN)?;
    let length = input.get_length();
    if length == 0 || length > data_len {
        return Err(TrendContinuationFactorError::InvalidLength { length, data_len });
    }

    let valid = data_len - first;
    if valid <= length {
        return Err(TrendContinuationFactorError::NotEnoughValidData {
            needed: length + 1,
            valid,
        });
    }

    Ok((data, length, first))
}

#[derive(Clone, Debug)]
pub struct TrendContinuationFactorStream {
    length: usize,
    prev: Option<f64>,
    plus_cf: Option<f64>,
    minus_cf: Option<f64>,
    comparisons_seen: usize,
    head: usize,
    sum_plus: f64,
    sum_minus: f64,
    plus_buffer: Vec<f64>,
    minus_buffer: Vec<f64>,
}

impl TrendContinuationFactorStream {
    #[inline]
    fn from_length(length: usize) -> Self {
        Self {
            length,
            prev: None,
            plus_cf: None,
            minus_cf: None,
            comparisons_seen: 0,
            head: 0,
            sum_plus: 0.0,
            sum_minus: 0.0,
            plus_buffer: vec![0.0; length.max(1)],
            minus_buffer: vec![0.0; length.max(1)],
        }
    }

    #[inline]
    pub fn try_new(
        params: TrendContinuationFactorParams,
    ) -> Result<Self, TrendContinuationFactorError> {
        let length = params.length.unwrap_or(35);
        if length == 0 {
            return Err(TrendContinuationFactorError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self::from_length(length))
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev = None;
        self.plus_cf = None;
        self.minus_cf = None;
        self.comparisons_seen = 0;
        self.head = 0;
        self.sum_plus = 0.0;
        self.sum_minus = 0.0;
        self.plus_buffer.fill(0.0);
        self.minus_buffer.fill(0.0);
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            return None;
        }

        let prev = match self.prev.replace(value) {
            Some(prev) => prev,
            None => return None,
        };

        let change = value - prev;
        let plus_change = if change > 0.0 { change } else { 0.0 };
        let minus_change = if change < 0.0 { -change } else { 0.0 };

        let next_plus_cf = if plus_change == 0.0 {
            0.0
        } else {
            plus_change + self.plus_cf.unwrap_or(1.0)
        };
        let next_minus_cf = if minus_change == 0.0 {
            0.0
        } else {
            minus_change + self.minus_cf.unwrap_or(1.0)
        };

        self.plus_cf = Some(next_plus_cf);
        self.minus_cf = Some(next_minus_cf);

        let plus = plus_change - next_minus_cf;
        let minus = minus_change - next_plus_cf;

        if self.comparisons_seen < self.length {
            self.plus_buffer[self.comparisons_seen] = plus;
            self.minus_buffer[self.comparisons_seen] = minus;
            self.sum_plus += plus;
            self.sum_minus += minus;
            self.comparisons_seen += 1;
            if self.comparisons_seen < self.length {
                return None;
            }
            return Some((self.sum_plus, self.sum_minus));
        }

        let old_plus = self.plus_buffer[self.head];
        let old_minus = self.minus_buffer[self.head];
        self.plus_buffer[self.head] = plus;
        self.minus_buffer[self.head] = minus;
        self.sum_plus += plus - old_plus;
        self.sum_minus += minus - old_minus;
        self.head += 1;
        if self.head == self.length {
            self.head = 0;
        }

        Some((self.sum_plus, self.sum_minus))
    }

    #[inline(always)]
    pub fn update_reset_on_nan(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.update(value)
    }
}

#[inline(always)]
fn trend_continuation_factor_compute_into(
    data: &[f64],
    length: usize,
    _first: usize,
    _kernel: Kernel,
    out_plus: &mut [f64],
    out_minus: &mut [f64],
) {
    if length == 35 {
        let mut plus_buffer = [0.0f64; 35];
        let mut minus_buffer = [0.0f64; 35];
        trend_continuation_factor_compute_with_buffers(
            data,
            length,
            out_plus,
            out_minus,
            &mut plus_buffer,
            &mut minus_buffer,
        );
        return;
    }

    let mut plus_buffer = vec![0.0f64; length.max(1)];
    let mut minus_buffer = vec![0.0f64; length.max(1)];
    trend_continuation_factor_compute_with_buffers(
        data,
        length,
        out_plus,
        out_minus,
        &mut plus_buffer,
        &mut minus_buffer,
    );
}

#[inline(always)]
fn trend_continuation_factor_compute_with_buffers(
    data: &[f64],
    length: usize,
    out_plus: &mut [f64],
    out_minus: &mut [f64],
    plus_buffer: &mut [f64],
    minus_buffer: &mut [f64],
) {
    let mut prev = 0.0;
    let mut has_prev = false;
    let mut plus_cf = 0.0;
    let mut minus_cf = 0.0;
    let mut has_cf = false;
    let mut comparisons_seen = 0usize;
    let mut head = 0usize;
    let mut sum_plus = 0.0;
    let mut sum_minus = 0.0;

    for i in 0..data.len() {
        let value = data[i];
        if !value.is_finite() {
            has_prev = false;
            has_cf = false;
            comparisons_seen = 0;
            head = 0;
            sum_plus = 0.0;
            sum_minus = 0.0;
            out_plus[i] = f64::NAN;
            out_minus[i] = f64::NAN;
            continue;
        }

        if !has_prev {
            prev = value;
            has_prev = true;
            out_plus[i] = f64::NAN;
            out_minus[i] = f64::NAN;
            continue;
        }

        let change = value - prev;
        prev = value;

        let plus_change = if change > 0.0 { change } else { 0.0 };
        let minus_change = if change < 0.0 { -change } else { 0.0 };
        let cf_seed_plus = if has_cf { plus_cf } else { 1.0 };
        let cf_seed_minus = if has_cf { minus_cf } else { 1.0 };

        let next_plus_cf = if plus_change == 0.0 {
            0.0
        } else {
            plus_change + cf_seed_plus
        };
        let next_minus_cf = if minus_change == 0.0 {
            0.0
        } else {
            minus_change + cf_seed_minus
        };

        plus_cf = next_plus_cf;
        minus_cf = next_minus_cf;
        has_cf = true;

        let plus = plus_change - next_minus_cf;
        let minus = minus_change - next_plus_cf;

        if comparisons_seen < length {
            plus_buffer[comparisons_seen] = plus;
            minus_buffer[comparisons_seen] = minus;
            sum_plus += plus;
            sum_minus += minus;
            comparisons_seen += 1;
            if comparisons_seen < length {
                out_plus[i] = f64::NAN;
                out_minus[i] = f64::NAN;
            } else {
                out_plus[i] = sum_plus;
                out_minus[i] = sum_minus;
            }
            continue;
        }

        let old_plus = plus_buffer[head];
        let old_minus = minus_buffer[head];
        plus_buffer[head] = plus;
        minus_buffer[head] = minus;
        sum_plus += plus - old_plus;
        sum_minus += minus - old_minus;
        head += 1;
        if head == length {
            head = 0;
        }

        out_plus[i] = sum_plus;
        out_minus[i] = sum_minus;
    }
}

#[inline]
pub fn trend_continuation_factor(
    input: &TrendContinuationFactorInput,
) -> Result<TrendContinuationFactorOutput, TrendContinuationFactorError> {
    trend_continuation_factor_with_kernel(input, Kernel::Auto)
}

pub fn trend_continuation_factor_with_kernel(
    input: &TrendContinuationFactorInput,
    kernel: Kernel,
) -> Result<TrendContinuationFactorOutput, TrendContinuationFactorError> {
    let (data, length, first) = trend_continuation_factor_prepare(input)?;
    let warmup = first + length;
    let mut plus_tcf = alloc_with_nan_prefix(data.len(), warmup);
    let mut minus_tcf = alloc_with_nan_prefix(data.len(), warmup);
    trend_continuation_factor_compute_into(
        data,
        length,
        first,
        kernel,
        &mut plus_tcf,
        &mut minus_tcf,
    );
    Ok(TrendContinuationFactorOutput {
        plus_tcf,
        minus_tcf,
    })
}

pub fn trend_continuation_factor_into_slice(
    dst_plus_tcf: &mut [f64],
    dst_minus_tcf: &mut [f64],
    input: &TrendContinuationFactorInput,
    kernel: Kernel,
) -> Result<(), TrendContinuationFactorError> {
    let (data, length, first) = trend_continuation_factor_prepare(input)?;
    if dst_plus_tcf.len() != data.len() || dst_minus_tcf.len() != data.len() {
        return Err(TrendContinuationFactorError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_plus_tcf.len().max(dst_minus_tcf.len()),
        });
    }
    trend_continuation_factor_compute_into(
        data,
        length,
        first,
        kernel,
        dst_plus_tcf,
        dst_minus_tcf,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn trend_continuation_factor_into(
    input: &TrendContinuationFactorInput,
    out_plus_tcf: &mut [f64],
    out_minus_tcf: &mut [f64],
) -> Result<(), TrendContinuationFactorError> {
    trend_continuation_factor_into_slice(out_plus_tcf, out_minus_tcf, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct TrendContinuationFactorBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for TrendContinuationFactorBatchRange {
    fn default() -> Self {
        Self {
            length: (35, 200, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrendContinuationFactorBatchBuilder {
    range: TrendContinuationFactorBatchRange,
    kernel: Kernel,
}

impl TrendContinuationFactorBatchBuilder {
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
    ) -> Result<TrendContinuationFactorBatchOutput, TrendContinuationFactorError> {
        trend_continuation_factor_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<TrendContinuationFactorBatchOutput, TrendContinuationFactorError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct TrendContinuationFactorBatchOutput {
    pub plus_tcf: Vec<f64>,
    pub minus_tcf: Vec<f64>,
    pub combos: Vec<TrendContinuationFactorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendContinuationFactorBatchOutput {
    pub fn row_for_params(&self, params: &TrendContinuationFactorParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.length.unwrap_or(35) == params.length.unwrap_or(35))
    }

    pub fn plus_tcf_for(&self, params: &TrendContinuationFactorParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.plus_tcf[start..start + self.cols]
        })
    }

    pub fn minus_tcf_for(&self, params: &TrendContinuationFactorParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.minus_tcf[start..start + self.cols]
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, TrendContinuationFactorError> {
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
        return Err(TrendContinuationFactorError::InvalidRange { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_trend_continuation_factor(
    sweep: &TrendContinuationFactorBatchRange,
) -> Result<Vec<TrendContinuationFactorParams>, TrendContinuationFactorError> {
    Ok(axis_usize(sweep.length)?
        .into_iter()
        .map(|length| TrendContinuationFactorParams {
            length: Some(length),
        })
        .collect())
}

pub fn trend_continuation_factor_batch_with_kernel(
    data: &[f64],
    sweep: &TrendContinuationFactorBatchRange,
    kernel: Kernel,
) -> Result<TrendContinuationFactorBatchOutput, TrendContinuationFactorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(TrendContinuationFactorError::InvalidKernelForBatch(other)),
    };
    trend_continuation_factor_batch_impl(data, sweep, batch_kernel.to_non_batch(), true)
}

pub fn trend_continuation_factor_batch_slice(
    data: &[f64],
    sweep: &TrendContinuationFactorBatchRange,
) -> Result<TrendContinuationFactorBatchOutput, TrendContinuationFactorError> {
    trend_continuation_factor_batch_impl(data, sweep, Kernel::Scalar, false)
}

pub fn trend_continuation_factor_batch_par_slice(
    data: &[f64],
    sweep: &TrendContinuationFactorBatchRange,
) -> Result<TrendContinuationFactorBatchOutput, TrendContinuationFactorError> {
    trend_continuation_factor_batch_impl(data, sweep, Kernel::Scalar, true)
}

fn trend_continuation_factor_batch_impl(
    data: &[f64],
    sweep: &TrendContinuationFactorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<TrendContinuationFactorBatchOutput, TrendContinuationFactorError> {
    let combos = expand_grid_trend_continuation_factor(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    if cols == 0 {
        return Err(TrendContinuationFactorError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(TrendContinuationFactorError::AllValuesNaN)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(35))
        .max()
        .unwrap_or(35);
    let valid = cols - first;
    if valid <= max_length {
        return Err(TrendContinuationFactorError::NotEnoughValidData {
            needed: max_length + 1,
            valid,
        });
    }

    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| first + params.length.unwrap_or(35))
        .collect();

    let mut plus_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut plus_matrix, cols, &warmups);
    let mut minus_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut minus_matrix, cols, &warmups);

    let mut plus_guard = ManuallyDrop::new(plus_matrix);
    let mut minus_guard = ManuallyDrop::new(minus_matrix);

    let plus_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(plus_guard.as_mut_ptr(), plus_guard.len()) };
    let minus_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(minus_guard.as_mut_ptr(), minus_guard.len()) };

    let do_row = |row: usize,
                  row_plus_mu: &mut [MaybeUninit<f64>],
                  row_minus_mu: &mut [MaybeUninit<f64>]| {
        let length = combos[row].length.unwrap_or(35);
        let dst_plus = unsafe {
            std::slice::from_raw_parts_mut(row_plus_mu.as_mut_ptr() as *mut f64, row_plus_mu.len())
        };
        let dst_minus = unsafe {
            std::slice::from_raw_parts_mut(
                row_minus_mu.as_mut_ptr() as *mut f64,
                row_minus_mu.len(),
            )
        };
        trend_continuation_factor_compute_into(data, length, first, kernel, dst_plus, dst_minus);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        plus_mu
            .par_chunks_mut(cols)
            .zip(minus_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_plus_mu, row_minus_mu))| do_row(row, row_plus_mu, row_minus_mu));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_plus_mu, row_minus_mu)) in plus_mu
            .chunks_mut(cols)
            .zip(minus_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_plus_mu, row_minus_mu);
        }
    } else {
        for (row, (row_plus_mu, row_minus_mu)) in plus_mu
            .chunks_mut(cols)
            .zip(minus_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_plus_mu, row_minus_mu);
        }
    }

    let plus_tcf = unsafe {
        Vec::from_raw_parts(
            plus_guard.as_mut_ptr() as *mut f64,
            plus_guard.len(),
            plus_guard.capacity(),
        )
    };
    let minus_tcf = unsafe {
        Vec::from_raw_parts(
            minus_guard.as_mut_ptr() as *mut f64,
            minus_guard.len(),
            minus_guard.capacity(),
        )
    };

    Ok(TrendContinuationFactorBatchOutput {
        plus_tcf,
        minus_tcf,
        combos,
        rows,
        cols,
    })
}

fn trend_continuation_factor_batch_inner_into(
    data: &[f64],
    sweep: &TrendContinuationFactorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_plus: &mut [f64],
    out_minus: &mut [f64],
) -> Result<(), TrendContinuationFactorError> {
    let combos = expand_grid_trend_continuation_factor(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if rows.checked_mul(cols) != Some(out_plus.len()) || out_minus.len() != out_plus.len() {
        return Err(TrendContinuationFactorError::OutputLengthMismatch {
            expected: rows * cols,
            got: out_plus.len().max(out_minus.len()),
        });
    }

    let first = first_valid_index(data).ok_or(TrendContinuationFactorError::AllValuesNaN)?;
    for (row, params) in combos.iter().enumerate() {
        let length = params.length.unwrap_or(35);
        let row_plus = &mut out_plus[row * cols..(row + 1) * cols];
        let row_minus = &mut out_minus[row * cols..(row + 1) * cols];
        row_plus.fill(f64::NAN);
        row_minus.fill(f64::NAN);
        if cols - first <= length {
            return Err(TrendContinuationFactorError::NotEnoughValidData {
                needed: length + 1,
                valid: cols - first,
            });
        }
    }

    let do_row = |row: usize, row_plus: &mut [f64], row_minus: &mut [f64]| {
        let length = combos[row].length.unwrap_or(35);
        trend_continuation_factor_compute_into(data, length, first, kernel, row_plus, row_minus);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_plus
            .par_chunks_mut(cols)
            .zip(out_minus.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_plus, row_minus))| do_row(row, row_plus, row_minus));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_plus, row_minus)) in out_plus
            .chunks_mut(cols)
            .zip(out_minus.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_plus, row_minus);
        }
    } else {
        for (row, (row_plus, row_minus)) in out_plus
            .chunks_mut(cols)
            .zip(out_minus.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_plus, row_minus);
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_continuation_factor")]
#[pyo3(signature = (data, length=35, kernel=None))]
pub fn trend_continuation_factor_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = TrendContinuationFactorInput::from_slice(
        data,
        TrendContinuationFactorParams {
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| trend_continuation_factor_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.plus_tcf.into_pyarray(py),
        output.minus_tcf.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrendContinuationFactorStream")]
pub struct TrendContinuationFactorStreamPy {
    stream: TrendContinuationFactorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrendContinuationFactorStreamPy {
    #[new]
    #[pyo3(signature = (length=35))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = TrendContinuationFactorStream::try_new(TrendContinuationFactorParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update_reset_on_nan(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_continuation_factor_batch")]
#[pyo3(signature = (data, length_range, kernel=None))]
pub fn trend_continuation_factor_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = TrendContinuationFactorBatchRange {
        length: length_range,
    };
    let combos = expand_grid_trend_continuation_factor(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let plus_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let minus_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_plus = unsafe { plus_arr.as_slice_mut()? };
    let out_minus = unsafe { minus_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        trend_continuation_factor_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_plus,
            out_minus,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("plus_tcf", plus_arr.reshape((rows, cols))?)?;
    dict.set_item("minus_tcf", minus_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(35) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_trend_continuation_factor_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trend_continuation_factor_py, m)?)?;
    m.add_function(wrap_pyfunction!(trend_continuation_factor_batch_py, m)?)?;
    m.add_class::<TrendContinuationFactorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendContinuationFactorBatchConfig {
    length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendContinuationFactorBatchJsOutput {
    plus_tcf: Vec<f64>,
    minus_tcf: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<TrendContinuationFactorParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendContinuationFactorJsOutput {
    plus_tcf: Vec<f64>,
    minus_tcf: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_continuation_factor_js")]
pub fn trend_continuation_factor_js(data: &[f64], length: usize) -> Result<JsValue, JsValue> {
    let input = TrendContinuationFactorInput::from_slice(
        data,
        TrendContinuationFactorParams {
            length: Some(length),
        },
    );
    let output =
        trend_continuation_factor(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&TrendContinuationFactorJsOutput {
        plus_tcf: output.plus_tcf,
        minus_tcf: output.minus_tcf,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_continuation_factor_batch_js")]
pub fn trend_continuation_factor_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: TrendContinuationFactorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = TrendContinuationFactorBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let batch = trend_continuation_factor_batch_slice(data, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&TrendContinuationFactorBatchJsOutput {
        plus_tcf: batch.plus_tcf,
        minus_tcf: batch.minus_tcf,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_continuation_factor_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len * 2);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_continuation_factor_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_continuation_factor_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_continuation_factor_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_plus, out_minus) = out.split_at_mut(len);
        let input = TrendContinuationFactorInput::from_slice(
            data,
            TrendContinuationFactorParams {
                length: Some(length),
            },
        );
        trend_continuation_factor_into_slice(out_plus, out_minus, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_continuation_factor_into_host")]
pub fn trend_continuation_factor_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_continuation_factor_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len() * 2);
        let (out_plus, out_minus) = out.split_at_mut(data.len());
        let input = TrendContinuationFactorInput::from_slice(
            data,
            TrendContinuationFactorParams {
                length: Some(length),
            },
        );
        trend_continuation_factor_into_slice(out_plus, out_minus, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_continuation_factor_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_continuation_factor_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = TrendContinuationFactorBatchRange {
            length: (length_start, length_end, length_step),
        };
        let combos = expand_grid_trend_continuation_factor(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len * 2);
        let (out_plus, out_minus) = out.split_at_mut(rows * len);
        trend_continuation_factor_batch_inner_into(
            data,
            &sweep,
            Kernel::Scalar,
            false,
            out_plus,
            out_minus,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_continuation_factor_output_into_js(
    data: &[f64],
    length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_continuation_factor_js(data, length)?;
    crate::write_wasm_object_f64_outputs("trend_continuation_factor_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_continuation_factor_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_continuation_factor_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "trend_continuation_factor_batch_output_into_js",
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
            out.push(100.0 + (i as f64 * 0.17).sin() * 2.5 + i as f64 * 0.03);
        }
        out
    }

    fn naive_tcf(data: &[f64], length: usize) -> (Vec<f64>, Vec<f64>) {
        let mut plus_out = vec![f64::NAN; data.len()];
        let mut minus_out = vec![f64::NAN; data.len()];
        if data.len() <= length {
            return (plus_out, minus_out);
        }

        let mut plus_cf: Option<f64> = None;
        let mut minus_cf: Option<f64> = None;
        let mut plus_terms = vec![0.0; length];
        let mut minus_terms = vec![0.0; length];
        let mut head = 0usize;
        let mut seen = 0usize;
        let mut sum_plus = 0.0;
        let mut sum_minus = 0.0;

        for i in 1..data.len() {
            let change = data[i] - data[i - 1];
            let plus_change = if change > 0.0 { change } else { 0.0 };
            let minus_change = if change < 0.0 { -change } else { 0.0 };
            let next_plus_cf = if plus_change == 0.0 {
                0.0
            } else {
                plus_change + plus_cf.unwrap_or(1.0)
            };
            let next_minus_cf = if minus_change == 0.0 {
                0.0
            } else {
                minus_change + minus_cf.unwrap_or(1.0)
            };
            plus_cf = Some(next_plus_cf);
            minus_cf = Some(next_minus_cf);

            let plus = plus_change - next_minus_cf;
            let minus = minus_change - next_plus_cf;

            if seen < length {
                plus_terms[seen] = plus;
                minus_terms[seen] = minus;
                sum_plus += plus;
                sum_minus += minus;
                seen += 1;
                if seen == length {
                    plus_out[i] = sum_plus;
                    minus_out[i] = sum_minus;
                }
            } else {
                sum_plus += plus - plus_terms[head];
                sum_minus += minus - minus_terms[head];
                plus_terms[head] = plus;
                minus_terms[head] = minus;
                head += 1;
                if head == length {
                    head = 0;
                }
                plus_out[i] = sum_plus;
                minus_out[i] = sum_minus;
            }
        }

        (plus_out, minus_out)
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
    fn trend_continuation_factor_matches_naive() {
        let data = sample_data(256);
        let input = TrendContinuationFactorInput::from_slice(
            &data,
            TrendContinuationFactorParams { length: Some(35) },
        );
        let out = trend_continuation_factor(&input).expect("indicator");
        let (plus_ref, minus_ref) = naive_tcf(&data, 35);
        assert_close(&out.plus_tcf, &plus_ref);
        assert_close(&out.minus_tcf, &minus_ref);
    }

    #[test]
    fn trend_continuation_factor_into_matches_api() {
        let data = sample_data(192);
        let input = TrendContinuationFactorInput::from_slice(
            &data,
            TrendContinuationFactorParams { length: Some(20) },
        );
        let baseline = trend_continuation_factor(&input).expect("baseline");
        let mut plus_out = vec![0.0; data.len()];
        let mut minus_out = vec![0.0; data.len()];
        trend_continuation_factor_into(&input, &mut plus_out, &mut minus_out).expect("into");
        assert_close(&baseline.plus_tcf, &plus_out);
        assert_close(&baseline.minus_tcf, &minus_out);
    }

    #[test]
    fn trend_continuation_factor_stream_matches_batch() {
        let data = sample_data(192);
        let batch = trend_continuation_factor(&TrendContinuationFactorInput::from_slice(
            &data,
            TrendContinuationFactorParams { length: Some(18) },
        ))
        .expect("batch");
        let mut stream = TrendContinuationFactorStream::try_new(TrendContinuationFactorParams {
            length: Some(18),
        })
        .expect("stream");
        let mut plus = vec![f64::NAN; data.len()];
        let mut minus = vec![f64::NAN; data.len()];
        for (i, value) in data.iter().enumerate() {
            if let Some((p, m)) = stream.update_reset_on_nan(*value) {
                plus[i] = p;
                minus[i] = m;
            }
        }
        assert_close(&batch.plus_tcf, &plus);
        assert_close(&batch.minus_tcf, &minus);
    }

    #[test]
    fn trend_continuation_factor_batch_single_param_matches_single() {
        let data = sample_data(192);
        let batch = trend_continuation_factor_batch_with_kernel(
            &data,
            &TrendContinuationFactorBatchRange {
                length: (12, 12, 0),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        let input = TrendContinuationFactorInput::from_slice(
            &data,
            TrendContinuationFactorParams { length: Some(12) },
        );
        let direct = trend_continuation_factor_with_kernel(&input, Kernel::Scalar).expect("direct");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.plus_tcf[..data.len()], &direct.plus_tcf);
        assert_close(&batch.minus_tcf[..data.len()], &direct.minus_tcf);
    }

    #[test]
    fn trend_continuation_factor_rejects_invalid_length() {
        let data = sample_data(32);
        let input = TrendContinuationFactorInput::from_slice(
            &data,
            TrendContinuationFactorParams { length: Some(0) },
        );
        let err = trend_continuation_factor(&input).unwrap_err();
        assert!(matches!(
            err,
            TrendContinuationFactorError::InvalidLength { .. }
        ));
    }

    #[test]
    fn trend_continuation_factor_dispatch_matches_direct() {
        let data = sample_data(160);
        let combo = [ParamKV {
            key: "length",
            value: ParamValue::Int(16),
        }];
        let combos = [IndicatorParamSet { params: &combo }];

        let req = IndicatorBatchRequest {
            indicator_id: "trend_continuation_factor",
            output_id: Some("plus_tcf"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).expect("dispatch");

        let input = TrendContinuationFactorInput::from_slice(
            &data,
            TrendContinuationFactorParams { length: Some(16) },
        );
        let direct = trend_continuation_factor(&input).expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        let values = out.values_f64.expect("values");
        assert_close(&values, &direct.plus_tcf);
    }
}
