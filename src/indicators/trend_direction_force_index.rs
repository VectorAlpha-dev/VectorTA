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
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for TrendDirectionForceIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrendDirectionForceIndexData::Slice(slice) => slice,
            TrendDirectionForceIndexData::Candles { candles, source } => match *source {
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
        let params = TrendDirectionForceIndexParams {
            length: self.length,
        };
        trend_direction_force_index_with_kernel(
            &TrendDirectionForceIndexInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<TrendDirectionForceIndexOutput, TrendDirectionForceIndexError> {
        let params = TrendDirectionForceIndexParams {
            length: self.length,
        };
        trend_direction_force_index_with_kernel(
            &TrendDirectionForceIndexInput::from_candles(candles, source, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<TrendDirectionForceIndexOutput, TrendDirectionForceIndexError> {
        let params = TrendDirectionForceIndexParams {
            length: self.length,
        };
        trend_direction_force_index_with_kernel(
            &TrendDirectionForceIndexInput::from_slice(data, params),
            self.kernel,
        )
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
    #[error(
        "trend_direction_force_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("trend_direction_force_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
struct EmaSeededStream {
    period: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    mean: f64,
    filled: bool,
}

impl EmaSeededStream {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            alpha,
            beta: 1.0 - alpha,
            count: 0,
            mean: f64::NAN,
            filled: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.mean = f64::NAN;
        self.filled = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        self.count += 1;
        let count = self.count;
        if count == 1 {
            self.mean = value;
        } else if count <= self.period {
            let inv = 1.0 / count as f64;
            self.mean = (value - self.mean).mul_add(inv, self.mean);
        } else {
            self.mean = self.beta.mul_add(self.mean, self.alpha * value);
        }
        if !self.filled && count >= self.period {
            self.filled = true;
        }
        if self.filled {
            Some(self.mean)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrendDirectionForceIndexStream {
    length: usize,
    norm_window: usize,
    next_index: usize,
    ema1: EmaSeededStream,
    ema2: EmaSeededStream,
    prev_ema1: f64,
    prev_ema2: f64,
    have_prev_emas: bool,
    max_abs: VecDeque<(usize, f64)>,
}

impl TrendDirectionForceIndexStream {
    #[inline(always)]
    pub fn try_new(
        params: TrendDirectionForceIndexParams,
    ) -> Result<Self, TrendDirectionForceIndexError> {
        let length = params.length.unwrap_or(10);
        validate_length(length, usize::MAX)?;
        let half = half_length(length);
        Ok(Self {
            length,
            norm_window: normalization_window(length),
            next_index: 0,
            ema1: EmaSeededStream::new(half),
            ema2: EmaSeededStream::new(half),
            prev_ema1: f64::NAN,
            prev_ema2: f64::NAN,
            have_prev_emas: false,
            max_abs: VecDeque::with_capacity(normalization_window(length).max(1)),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.ema1.reset();
        self.ema2.reset();
        self.prev_ema1 = f64::NAN;
        self.prev_ema2 = f64::NAN;
        self.have_prev_emas = false;
        self.max_abs.clear();
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let idx = self.next_index;
        self.next_index += 1;

        let ema1 = self.ema1.update(value * 1000.0)?;
        let ema2 = self.ema2.update(ema1)?;

        if !self.have_prev_emas {
            self.prev_ema1 = ema1;
            self.prev_ema2 = ema2;
            self.have_prev_emas = true;
            return None;
        }

        let ema_diff_avg = ((ema1 - self.prev_ema1) + (ema2 - self.prev_ema2)) * 0.5;
        let tdf = (ema1 - ema2).abs() * ema_diff_avg.powi(3);
        self.prev_ema1 = ema1;
        self.prev_ema2 = ema2;

        let abs_tdf = tdf.abs();
        while let Some((_, back)) = self.max_abs.back() {
            if *back <= abs_tdf {
                self.max_abs.pop_back();
            } else {
                break;
            }
        }
        self.max_abs.push_back((idx, abs_tdf));

        let window_start = idx.saturating_add(1).saturating_sub(self.norm_window);
        while let Some((front_idx, _)) = self.max_abs.front() {
            if *front_idx < window_start {
                self.max_abs.pop_front();
            } else {
                break;
            }
        }

        let max_abs = self.max_abs.front().map(|(_, v)| *v).unwrap_or(0.0);
        if max_abs == 0.0 {
            Some(0.0)
        } else {
            Some(tdf / max_abs)
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        required_samples(self.length).saturating_sub(1)
    }
}

#[inline(always)]
fn half_length(length: usize) -> usize {
    (length / 2).max(1)
}

#[inline(always)]
fn required_samples(length: usize) -> usize {
    half_length(length).saturating_mul(2)
}

#[inline(always)]
fn normalization_window(length: usize) -> usize {
    length.saturating_mul(3).max(1)
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
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
fn validate_length(length: usize, data_len: usize) -> Result<(), TrendDirectionForceIndexError> {
    if length == 0 || (data_len != usize::MAX && length > data_len) {
        return Err(TrendDirectionForceIndexError::InvalidLength { length, data_len });
    }
    Ok(())
}

#[inline(always)]
fn validate_common(data: &[f64], length: usize) -> Result<(), TrendDirectionForceIndexError> {
    if data.is_empty() {
        return Err(TrendDirectionForceIndexError::EmptyInputData);
    }
    validate_length(length, data.len())?;
    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(TrendDirectionForceIndexError::AllValuesNaN);
    }
    let needed = required_samples(length);
    if max_run < needed {
        return Err(TrendDirectionForceIndexError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(())
}

#[inline(always)]
fn compute_row(data: &[f64], length: usize, out: &mut [f64]) {
    let norm_window = normalization_window(length);
    if length == 10 {
        let mut max_idx = [0usize; 31];
        let mut max_vals = [0.0f64; 31];
        compute_row_with_buffers(data, length, norm_window, out, &mut max_idx, &mut max_vals);
        return;
    }

    let cap = norm_window + 1;
    let mut max_idx = vec![0usize; cap];
    let mut max_vals = vec![0.0f64; cap];
    compute_row_with_buffers(data, length, norm_window, out, &mut max_idx, &mut max_vals);
}

#[inline(always)]
fn compute_row_with_buffers(
    data: &[f64],
    length: usize,
    norm_window: usize,
    out: &mut [f64],
    max_idx: &mut [usize],
    max_vals: &mut [f64],
) {
    let half = half_length(length);
    let mut ema1_stream = EmaSeededStream::new(half);
    let mut ema2_stream = EmaSeededStream::new(half);
    let mut next_index = 0usize;
    let mut prev_ema1 = f64::NAN;
    let mut prev_ema2 = f64::NAN;
    let mut have_prev_emas = false;
    let mut head = 0usize;
    let mut tail = 0usize;
    let mut count = 0usize;
    let cap = max_idx.len();

    for i in 0..data.len() {
        let value = data[i];
        if !value.is_finite() {
            ema1_stream.reset();
            ema2_stream.reset();
            next_index = 0;
            prev_ema1 = f64::NAN;
            prev_ema2 = f64::NAN;
            have_prev_emas = false;
            head = 0;
            tail = 0;
            count = 0;
            out[i] = f64::NAN;
            continue;
        }

        let idx = next_index;
        next_index += 1;

        let ema1 = match ema1_stream.update(value * 1000.0) {
            Some(value) => value,
            None => {
                out[i] = f64::NAN;
                continue;
            }
        };
        let ema2 = match ema2_stream.update(ema1) {
            Some(value) => value,
            None => {
                out[i] = f64::NAN;
                continue;
            }
        };

        if !have_prev_emas {
            prev_ema1 = ema1;
            prev_ema2 = ema2;
            have_prev_emas = true;
            out[i] = f64::NAN;
            continue;
        }

        let ema_diff_avg = ((ema1 - prev_ema1) + (ema2 - prev_ema2)) * 0.5;
        let tdf = (ema1 - ema2).abs() * ema_diff_avg.powi(3);
        prev_ema1 = ema1;
        prev_ema2 = ema2;

        let abs_tdf = tdf.abs();
        while count > 0 {
            let back = if tail == 0 { cap - 1 } else { tail - 1 };
            if max_vals[back] <= abs_tdf {
                tail = back;
                count -= 1;
            } else {
                break;
            }
        }

        max_idx[tail] = idx;
        max_vals[tail] = abs_tdf;
        tail += 1;
        if tail == cap {
            tail = 0;
        }
        count += 1;

        let window_start = idx.saturating_add(1).saturating_sub(norm_window);
        while count > 0 && max_idx[head] < window_start {
            head += 1;
            if head == cap {
                head = 0;
            }
            count -= 1;
        }

        let max_abs = if count == 0 { 0.0 } else { max_vals[head] };
        out[i] = if max_abs == 0.0 { 0.0 } else { tdf / max_abs };
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
    _kernel: Kernel,
) -> Result<TrendDirectionForceIndexOutput, TrendDirectionForceIndexError> {
    let data = input.as_ref();
    let length = input.get_length();
    validate_common(data, length)?;

    let warmup = required_samples(length).saturating_sub(1);
    let mut out = alloc_with_nan_prefix(data.len(), warmup);
    compute_row(data, length, &mut out);
    Ok(TrendDirectionForceIndexOutput { values: out })
}

pub fn trend_direction_force_index_into_slice(
    dst: &mut [f64],
    input: &TrendDirectionForceIndexInput,
    _kernel: Kernel,
) -> Result<(), TrendDirectionForceIndexError> {
    let data = input.as_ref();
    let length = input.get_length();
    validate_common(data, length)?;
    if dst.len() != data.len() {
        return Err(TrendDirectionForceIndexError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    compute_row(data, length, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn trend_direction_force_index_into(
    input: &TrendDirectionForceIndexInput,
    out: &mut [f64],
) -> Result<(), TrendDirectionForceIndexError> {
    trend_direction_force_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct TrendDirectionForceIndexBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for TrendDirectionForceIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (10, 10, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrendDirectionForceIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrendDirectionForceIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TrendDirectionForceIndexBatchBuilder {
    range: TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
}

impl Default for TrendDirectionForceIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: TrendDirectionForceIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl TrendDirectionForceIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
        trend_direction_force_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
        trend_direction_force_index_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &TrendDirectionForceIndexBatchRange,
) -> Result<Vec<TrendDirectionForceIndexParams>, TrendDirectionForceIndexError> {
    let (start, end, step) = range.length;
    if start == 0 || end == 0 {
        return Err(TrendDirectionForceIndexError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![TrendDirectionForceIndexParams {
            length: Some(start),
        }]);
    }
    if start > end {
        return Err(TrendDirectionForceIndexError::InvalidRange { start, end, step });
    }

    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(TrendDirectionForceIndexParams { length: Some(cur) });
        if cur >= end {
            break;
        }
        let next = cur.saturating_add(step);
        if next <= cur {
            return Err(TrendDirectionForceIndexError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
        if cur == out.last().and_then(|p| p.length).unwrap_or(cur) {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_trend_direction_force_index(
    range: &TrendDirectionForceIndexBatchRange,
) -> Vec<TrendDirectionForceIndexParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn trend_direction_force_index_batch_with_kernel(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(TrendDirectionForceIndexError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(10))
        .max()
        .unwrap_or(0);
    validate_common(data, max_length)?;

    let rows = combos.len();
    let cols = data.len();
    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| required_samples(params.length.unwrap_or(10)).saturating_sub(1))
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

    trend_direction_force_index_batch_inner_into(data, sweep, kernel, true, &mut values)?;

    Ok(TrendDirectionForceIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn trend_direction_force_index_batch_slice(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    trend_direction_force_index_batch_inner(data, sweep, kernel, false)
}

pub fn trend_direction_force_index_batch_par_slice(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    trend_direction_force_index_batch_inner(data, sweep, kernel, true)
}

fn trend_direction_force_index_batch_inner(
    data: &[f64],
    sweep: &TrendDirectionForceIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| TrendDirectionForceIndexError::InvalidInput {
                msg: "trend_direction_force_index: rows*cols overflow in batch".to_string(),
            })?;

    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| required_samples(params.length.unwrap_or(10)).saturating_sub(1))
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

    trend_direction_force_index_batch_inner_into(data, sweep, kernel, parallel, &mut values)?;

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
) -> Result<Vec<TrendDirectionForceIndexParams>, TrendDirectionForceIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(TrendDirectionForceIndexError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(TrendDirectionForceIndexError::EmptyInputData);
    }

    let total = combos.len().checked_mul(len).ok_or_else(|| {
        TrendDirectionForceIndexError::InvalidInput {
            msg: "trend_direction_force_index: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    if out.len() != total {
        return Err(TrendDirectionForceIndexError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: total,
        });
    }

    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(10))
        .max()
        .unwrap_or(0);
    validate_common(data, max_length)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst: &mut [f64]| {
        dst.fill(f64::NAN);
        let length = combos[row].length.unwrap_or(10);
        compute_row(data, length, dst);
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
    let kern = validate_kernel(kernel, true)?;
    let input = TrendDirectionForceIndexInput::from_slice(
        data,
        TrendDirectionForceIndexParams {
            length: Some(length),
        },
    );
    let out = py
        .allow_threads(|| trend_direction_force_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
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
    fn new(length: usize) -> PyResult<Self> {
        let stream = TrendDirectionForceIndexStream::try_new(TrendDirectionForceIndexParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_direction_force_index_batch")]
#[pyo3(signature = (data, length_range=(10,10,0), kernel=None))]
pub fn trend_direction_force_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            trend_direction_force_index_batch_with_kernel(
                data,
                &TrendDirectionForceIndexBatchRange {
                    length: length_range,
                },
                kern,
            )
        })
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
pub struct TrendDirectionForceIndexBatchConfig {
    pub length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = trend_direction_force_index_js)]
pub fn trend_direction_force_index_js(data: &[f64], length: usize) -> Result<JsValue, JsValue> {
    let input = TrendDirectionForceIndexInput::from_slice(
        data,
        TrendDirectionForceIndexParams {
            length: Some(length),
        },
    );
    let out = trend_direction_force_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out.values).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = trend_direction_force_index_batch_js)]
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
    let out = trend_direction_force_index_batch_with_kernel(
        data,
        &TrendDirectionForceIndexBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
        },
        Kernel::Auto,
    )
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
pub fn trend_direction_force_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_direction_force_index_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
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
#[wasm_bindgen]
pub fn trend_direction_force_index_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_direction_force_index_batch_into",
        ));
    }
    let sweep = TrendDirectionForceIndexBatchRange {
        length: (length_start, length_end, length_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in trend_direction_force_index_batch_into")
    })?;
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        trend_direction_force_index_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_output_into_js(
    data: &[f64],
    length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_direction_force_index_js(data, length)?;
    crate::write_wasm_object_f64_outputs("trend_direction_force_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_direction_force_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_direction_force_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "trend_direction_force_index_batch_output_into_js",
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

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.04 + (x * 0.13).sin() * 2.2 + (x * 0.017).cos() * 0.7
            })
            .collect()
    }

    fn ema_series(values: &[f64], period: usize) -> Vec<Option<f64>> {
        let mut out = vec![None; values.len()];
        let mut mean = f64::NAN;
        for (i, &value) in values.iter().enumerate() {
            if i == 0 {
                mean = value;
            } else if i + 1 <= period {
                let inv = 1.0 / (i + 1) as f64;
                mean = (value - mean).mul_add(inv, mean);
            } else {
                let alpha = 2.0 / (period as f64 + 1.0);
                mean = (1.0 - alpha).mul_add(mean, alpha * value);
            }
            if i + 1 >= period {
                out[i] = Some(mean);
            }
        }
        out
    }

    fn naive_tdfi(data: &[f64], length: usize) -> Vec<f64> {
        let half = half_length(length);
        let norm = normalization_window(length);
        let scaled: Vec<f64> = data.iter().map(|v| v * 1000.0).collect();
        let ema1 = ema_series(&scaled, half);
        let ema1_vals: Vec<f64> = ema1.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
        let ema2 = ema_series(&ema1_vals[half.saturating_sub(1)..], half);
        let mut ema2_full = vec![None; data.len()];
        for (offset, value) in ema2.into_iter().enumerate() {
            ema2_full[offset + half.saturating_sub(1)] = value;
        }

        let mut out = vec![f64::NAN; data.len()];
        for i in 1..data.len() {
            let (Some(e1), Some(e1_prev), Some(e2), Some(e2_prev)) =
                (ema1[i], ema1[i - 1], ema2_full[i], ema2_full[i - 1])
            else {
                continue;
            };
            let ema_diff_avg = ((e1 - e1_prev) + (e2 - e2_prev)) * 0.5;
            let tdf = (e1 - e2).abs() * ema_diff_avg.powi(3);
            let start = i.saturating_add(1).saturating_sub(norm);
            let mut max_abs: f64 = 0.0;
            for j in start..=i {
                let (Some(a1), Some(a1_prev), Some(a2), Some(a2_prev)) = (
                    ema1[j],
                    ema1.get(j.wrapping_sub(1)).copied().flatten(),
                    ema2_full[j],
                    ema2_full.get(j.wrapping_sub(1)).copied().flatten(),
                ) else {
                    continue;
                };
                let avg = ((a1 - a1_prev) + (a2 - a2_prev)) * 0.5;
                let val = (a1 - a2).abs() * avg.powi(3);
                max_abs = max_abs.max(val.abs());
            }
            out[i] = if max_abs == 0.0 { 0.0 } else { tdf / max_abs };
        }
        out
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn trend_direction_force_index_matches_naive() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let input = TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(10) },
        );
        let out = trend_direction_force_index_with_kernel(&input, Kernel::Scalar)?;
        let expected = naive_tdfi(&data, 10);
        assert_series_close(&out.values, &expected, 1e-9);
        Ok(())
    }

    #[test]
    fn trend_direction_force_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data(200);
        let input = TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(12) },
        );
        let base = trend_direction_force_index(&input)?;
        let mut out = vec![f64::NAN; data.len()];
        trend_direction_force_index_into_slice(&mut out, &input, Kernel::Auto)?;
        assert_series_close(&base.values, &out, 1e-12);
        Ok(())
    }

    #[test]
    fn trend_direction_force_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let batch = trend_direction_force_index(&TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(10) },
        ))?;

        let mut stream = TrendDirectionForceIndexStream::try_new(TrendDirectionForceIndexParams {
            length: Some(10),
        })?;
        let mut streamed = Vec::with_capacity(data.len());
        for &value in &data {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }
        assert_series_close(&batch.values, &streamed, 1e-12);
        Ok(())
    }

    #[test]
    fn trend_direction_force_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let single = trend_direction_force_index(&TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(10) },
        ))?;
        let batch = trend_direction_force_index_batch_with_kernel(
            &data,
            &TrendDirectionForceIndexBatchRange::default(),
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_series_close(&single.values, &batch.values, 1e-12);
        Ok(())
    }

    #[test]
    fn trend_direction_force_index_rejects_invalid_params() {
        let data = sample_data(32);
        let err = trend_direction_force_index(&TrendDirectionForceIndexInput::from_slice(
            &data,
            TrendDirectionForceIndexParams { length: Some(0) },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            TrendDirectionForceIndexError::InvalidLength { .. }
        ));

        let err = trend_direction_force_index_batch_with_kernel(
            &data,
            &TrendDirectionForceIndexBatchRange { length: (10, 5, 1) },
            Kernel::Auto,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            TrendDirectionForceIndexError::InvalidRange { .. }
        ));
    }

    #[test]
    fn trend_direction_force_index_dispatch_compute_returns_value() -> Result<(), Box<dyn Error>> {
        let data = sample_data(192);
        let req = IndicatorComputeRequest {
            indicator_id: "trend_direction_force_index",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            params: &[ParamKV {
                key: "length",
                value: ParamValue::Int(10),
            }],
            kernel: Kernel::Auto,
        };
        let out = compute_cpu(req)?;
        assert_eq!(out.output_id, "value");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        Ok(())
    }
}
