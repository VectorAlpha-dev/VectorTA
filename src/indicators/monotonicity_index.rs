#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 20;
const DEFAULT_INDEX_SMOOTH: usize = 5;
const DEFAULT_SOURCE: &str = "close";

impl<'a> AsRef<[f64]> for MonotonicityIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MonotonicityIndexData::Slice(slice) => slice,
            MonotonicityIndexData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MonotonicityIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MonotonicityIndexOutput {
    pub index: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub upper_bound: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    serde(rename_all = "snake_case")
)]
pub enum MonotonicityIndexMode {
    Complexity,
    #[default]
    Efficiency,
}

impl MonotonicityIndexMode {
    #[inline]
    pub fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("complexity") {
            Some(Self::Complexity)
        } else if value.eq_ignore_ascii_case("efficiency") {
            Some(Self::Efficiency)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complexity => "complexity",
            Self::Efficiency => "efficiency",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MonotonicityIndexParams {
    pub length: Option<usize>,
    pub mode: Option<MonotonicityIndexMode>,
    pub index_smooth: Option<usize>,
}

impl Default for MonotonicityIndexParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            mode: Some(MonotonicityIndexMode::Efficiency),
            index_smooth: Some(DEFAULT_INDEX_SMOOTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonotonicityIndexInput<'a> {
    pub data: MonotonicityIndexData<'a>,
    pub params: MonotonicityIndexParams,
}

impl<'a> MonotonicityIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: MonotonicityIndexParams,
    ) -> Self {
        Self {
            data: MonotonicityIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: MonotonicityIndexParams) -> Self {
        Self {
            data: MonotonicityIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DEFAULT_SOURCE, MonotonicityIndexParams::default())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MonotonicityIndexBuilder {
    length: Option<usize>,
    mode: Option<MonotonicityIndexMode>,
    index_smooth: Option<usize>,
    kernel: Kernel,
}

impl MonotonicityIndexBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline]
    pub fn mode(mut self, mode: MonotonicityIndexMode) -> Self {
        self.mode = Some(mode);
        self
    }

    #[inline]
    pub fn index_smooth(mut self, index_smooth: usize) -> Self {
        self.index_smooth = Some(index_smooth);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<MonotonicityIndexOutput, MonotonicityIndexError> {
        let input = MonotonicityIndexInput::from_candles(
            candles,
            source,
            MonotonicityIndexParams {
                length: self.length,
                mode: self.mode,
                index_smooth: self.index_smooth,
            },
        );
        monotonicity_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<MonotonicityIndexOutput, MonotonicityIndexError> {
        let input = MonotonicityIndexInput::from_slice(
            data,
            MonotonicityIndexParams {
                length: self.length,
                mode: self.mode,
                index_smooth: self.index_smooth,
            },
        );
        monotonicity_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<MonotonicityIndexStream, MonotonicityIndexError> {
        MonotonicityIndexStream::try_new(MonotonicityIndexParams {
            length: self.length,
            mode: self.mode,
            index_smooth: self.index_smooth,
        })
    }
}

#[derive(Debug, Error)]
pub enum MonotonicityIndexError {
    #[error("monotonicity_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("monotonicity_index: All values are NaN.")]
    AllValuesNaN,
    #[error("monotonicity_index: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("monotonicity_index: Invalid index_smooth: {index_smooth}")]
    InvalidIndexSmooth { index_smooth: usize },
    #[error("monotonicity_index: Invalid mode: {mode}")]
    InvalidMode { mode: String },
    #[error("monotonicity_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "monotonicity_index: Output length mismatch: expected = {expected}, index = {index_got}, cumulative_mean = {cumulative_mean_got}, upper_bound = {upper_bound_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        index_got: usize,
        cumulative_mean_got: usize,
        upper_bound_got: usize,
    },
    #[error("monotonicity_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("monotonicity_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    length: usize,
    mode: MonotonicityIndexMode,
    index_smooth: usize,
    warmup_period: usize,
    needed_valid: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct PavaFitSummary {
    mse: f64,
    pools: usize,
    start_value: f64,
    end_value: f64,
}

#[derive(Clone, Debug, Default)]
struct PavaScratch {
    inc_pool_vals: Vec<f64>,
    inc_pool_weights: Vec<usize>,
    dec_pool_vals: Vec<f64>,
    dec_pool_weights: Vec<usize>,
}

impl PavaScratch {
    #[inline]
    fn fit(&mut self, data: &[f64], non_decreasing: bool) -> PavaFitSummary {
        let (pool_vals, pool_weights) = if non_decreasing {
            (&mut self.inc_pool_vals, &mut self.inc_pool_weights)
        } else {
            (&mut self.dec_pool_vals, &mut self.dec_pool_weights)
        };
        pool_vals.clear();
        pool_weights.clear();

        for &value in data {
            let mut current_pool = value;
            let mut current_weight = 1usize;
            while let Some(&prev_pool) = pool_vals.last() {
                let violation = if non_decreasing {
                    prev_pool > current_pool
                } else {
                    prev_pool < current_pool
                };
                if !violation {
                    break;
                }

                let prev_weight = pool_weights.pop().unwrap();
                let last_pool = pool_vals.pop().unwrap();
                let combined_weight = prev_weight + current_weight;
                current_pool = (last_pool * prev_weight as f64
                    + current_pool * current_weight as f64)
                    / combined_weight as f64;
                current_weight = combined_weight;
            }

            pool_vals.push(current_pool);
            pool_weights.push(current_weight);
        }

        let mut total_error = 0.0;
        let mut idx = 0usize;
        for (&pool_value, &pool_weight) in pool_vals.iter().zip(pool_weights.iter()) {
            for _ in 0..pool_weight {
                let delta = data[idx] - pool_value;
                total_error += delta * delta;
                idx += 1;
            }
        }

        PavaFitSummary {
            mse: total_error / data.len() as f64,
            pools: pool_vals.len(),
            start_value: pool_vals.first().copied().unwrap_or(0.0),
            end_value: pool_vals.last().copied().unwrap_or(0.0),
        }
    }
}

#[derive(Clone, Debug)]
struct RollingWindow {
    buf: Vec<f64>,
    next: usize,
    len: usize,
}

impl RollingWindow {
    #[inline]
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            next: 0,
            len: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.next = 0;
        self.len = 0;
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn push(&mut self, value: f64) {
        self.buf[self.next] = value;
        self.next += 1;
        if self.next == self.buf.len() {
            self.next = 0;
        }
        if self.len < self.buf.len() {
            self.len += 1;
        }
    }

    #[inline]
    fn copy_to_vec(&self, out: &mut Vec<f64>) {
        out.clear();
        if self.len == 0 {
            return;
        }

        if self.len < self.buf.len() {
            out.extend_from_slice(&self.buf[..self.len]);
            return;
        }

        out.extend_from_slice(&self.buf[self.next..]);
        out.extend_from_slice(&self.buf[..self.next]);
    }
}

#[derive(Clone, Debug)]
struct RollingSma {
    buf: Vec<f64>,
    next: usize,
    len: usize,
    sum: f64,
}

impl RollingSma {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            buf: vec![0.0; period],
            next: 0,
            len: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.next = 0;
        self.len = 0;
        self.sum = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.len == self.buf.len() {
            self.sum -= self.buf[self.next];
        } else {
            self.len += 1;
        }
        self.buf[self.next] = value;
        self.sum += value;
        self.next += 1;
        if self.next == self.buf.len() {
            self.next = 0;
        }

        if self.len == self.buf.len() {
            Some(self.sum / self.buf.len() as f64)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct MonotonicityIndexStream {
    params: ResolvedParams,
    price_window: RollingWindow,
    raw_sma: RollingSma,
    cumulative_sum: f64,
    cumulative_count: usize,
    scratch: PavaScratch,
    window_data: Vec<f64>,
}

impl MonotonicityIndexStream {
    #[inline]
    pub fn try_new(params: MonotonicityIndexParams) -> Result<Self, MonotonicityIndexError> {
        let params = resolve_params(&params)?;
        Ok(Self {
            price_window: RollingWindow::new(params.length),
            raw_sma: RollingSma::new(params.index_smooth),
            cumulative_sum: 0.0,
            cumulative_count: 0,
            scratch: PavaScratch::default(),
            window_data: Vec::with_capacity(params.length),
            params,
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.price_window.reset();
        self.raw_sma.reset();
        self.cumulative_sum = 0.0;
        self.cumulative_count = 0;
        self.window_data.clear();
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.params.warmup_period
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        self.price_window.push(value);
        if self.price_window.len() < self.price_window.capacity() {
            return None;
        }

        self.price_window.copy_to_vec(&mut self.window_data);
        let raw_index = compute_raw_index(&self.window_data, self.params.mode, &mut self.scratch);
        let smoothed = self.raw_sma.update(raw_index)?;
        self.cumulative_sum += smoothed;
        self.cumulative_count += 1;
        let cumulative_mean = self.cumulative_sum / self.cumulative_count as f64;
        Some((smoothed, cumulative_mean, cumulative_mean * 2.0))
    }
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if data[i].is_finite() {
            return i;
        }
        i += 1;
    }
    data.len()
}

#[inline(always)]
fn max_consecutive_valid_values(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    for &value in data {
        if value.is_finite() {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
    }
    best
}

#[inline(always)]
fn resolve_params(
    params: &MonotonicityIndexParams,
) -> Result<ResolvedParams, MonotonicityIndexError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    if length < 2 {
        return Err(MonotonicityIndexError::InvalidLength { length });
    }

    let index_smooth = params.index_smooth.unwrap_or(DEFAULT_INDEX_SMOOTH);
    if index_smooth == 0 {
        return Err(MonotonicityIndexError::InvalidIndexSmooth { index_smooth });
    }

    let mode = params.mode.unwrap_or_default();
    let needed_valid = length
        .checked_add(index_smooth)
        .and_then(|x| x.checked_sub(1))
        .ok_or(MonotonicityIndexError::InvalidLength { length })?;

    Ok(ResolvedParams {
        length,
        mode,
        index_smooth,
        warmup_period: needed_valid - 1,
        needed_valid,
    })
}

#[inline(always)]
fn compute_raw_index(data: &[f64], mode: MonotonicityIndexMode, scratch: &mut PavaScratch) -> f64 {
    let inc_fit = scratch.fit(data, true);
    let dec_fit = scratch.fit(data, false);
    let best_fit = if inc_fit.mse < dec_fit.mse {
        inc_fit
    } else {
        dec_fit
    };

    match mode {
        MonotonicityIndexMode::Efficiency => {
            let mut price_path = 0.0;
            let mut i = 1usize;
            while i < data.len() {
                price_path += (data[i] - data[i - 1]).abs();
                i += 1;
            }
            if price_path > 0.0 {
                (best_fit.end_value - best_fit.start_value).abs() / price_path * 100.0
            } else {
                0.0
            }
        }
        MonotonicityIndexMode::Complexity => {
            (best_fit.pools.saturating_sub(1) as f64 / (data.len() - 1) as f64) * 100.0
        }
    }
}

#[inline(always)]
fn monotonicity_index_prepare<'a>(
    input: &'a MonotonicityIndexInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, ResolvedParams, Kernel), MonotonicityIndexError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(MonotonicityIndexError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(MonotonicityIndexError::AllValuesNaN);
    }

    let params = resolve_params(&input.params)?;
    let valid = max_consecutive_valid_values(data);
    if valid < params.needed_valid {
        return Err(MonotonicityIndexError::NotEnoughValidData {
            needed: params.needed_valid,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, first, params, chosen))
}

#[inline(always)]
fn monotonicity_index_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    index_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    upper_bound_out: &mut [f64],
) {
    let mut stream = MonotonicityIndexStream::try_new(MonotonicityIndexParams {
        length: Some(params.length),
        mode: Some(params.mode),
        index_smooth: Some(params.index_smooth),
    })
    .unwrap();

    for (((index_slot, cumulative_mean_slot), upper_bound_slot), &value) in index_out
        .iter_mut()
        .zip(cumulative_mean_out.iter_mut())
        .zip(upper_bound_out.iter_mut())
        .zip(data.iter())
    {
        if let Some((index, cumulative_mean, upper_bound)) = stream.update(value) {
            *index_slot = index;
            *cumulative_mean_slot = cumulative_mean;
            *upper_bound_slot = upper_bound;
        } else {
            *index_slot = f64::NAN;
            *cumulative_mean_slot = f64::NAN;
            *upper_bound_slot = f64::NAN;
        }
    }
}

#[inline]
pub fn monotonicity_index(
    input: &MonotonicityIndexInput,
) -> Result<MonotonicityIndexOutput, MonotonicityIndexError> {
    monotonicity_index_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn monotonicity_index_with_kernel(
    input: &MonotonicityIndexInput,
    kernel: Kernel,
) -> Result<MonotonicityIndexOutput, MonotonicityIndexError> {
    let (data, first, params, _chosen) = monotonicity_index_prepare(input, kernel)?;
    let warmup = first.saturating_add(params.warmup_period).min(data.len());
    let mut index = alloc_with_nan_prefix(data.len(), warmup);
    let mut cumulative_mean = alloc_with_nan_prefix(data.len(), warmup);
    let mut upper_bound = alloc_with_nan_prefix(data.len(), warmup);
    monotonicity_index_row_from_slice(
        data,
        params,
        &mut index,
        &mut cumulative_mean,
        &mut upper_bound,
    );
    Ok(MonotonicityIndexOutput {
        index,
        cumulative_mean,
        upper_bound,
    })
}

#[inline]
pub fn monotonicity_index_into_slices(
    index_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    upper_bound_out: &mut [f64],
    input: &MonotonicityIndexInput,
    kernel: Kernel,
) -> Result<(), MonotonicityIndexError> {
    let expected = input.as_ref().len();
    if index_out.len() != expected
        || cumulative_mean_out.len() != expected
        || upper_bound_out.len() != expected
    {
        return Err(MonotonicityIndexError::OutputLengthMismatch {
            expected,
            index_got: index_out.len(),
            cumulative_mean_got: cumulative_mean_out.len(),
            upper_bound_got: upper_bound_out.len(),
        });
    }

    let (data, _first, params, _chosen) = monotonicity_index_prepare(input, kernel)?;
    monotonicity_index_row_from_slice(
        data,
        params,
        index_out,
        cumulative_mean_out,
        upper_bound_out,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn monotonicity_index_into(
    input: &MonotonicityIndexInput,
    index_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    upper_bound_out: &mut [f64],
) -> Result<(), MonotonicityIndexError> {
    monotonicity_index_into_slices(
        index_out,
        cumulative_mean_out,
        upper_bound_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MonotonicityIndexBatchRange {
    pub length: (usize, usize, usize),
    pub index_smooth: (usize, usize, usize),
    pub mode: MonotonicityIndexMode,
}

impl Default for MonotonicityIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            index_smooth: (DEFAULT_INDEX_SMOOTH, DEFAULT_INDEX_SMOOTH, 0),
            mode: MonotonicityIndexMode::Efficiency,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonotonicityIndexBatchOutput {
    pub index: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub combos: Vec<MonotonicityIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MonotonicityIndexBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &MonotonicityIndexParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == params.length.unwrap_or(DEFAULT_LENGTH)
                && combo.mode.unwrap_or_default() == params.mode.unwrap_or_default()
                && combo.index_smooth.unwrap_or(DEFAULT_INDEX_SMOOTH)
                    == params.index_smooth.unwrap_or(DEFAULT_INDEX_SMOOTH)
        })
    }

    #[inline]
    pub fn row_slices(&self, row: usize) -> Option<(&[f64], &[f64], &[f64])> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some((
            &self.index[start..end],
            &self.cumulative_mean[start..end],
            &self.upper_bound[start..end],
        ))
    }
}

#[derive(Clone, Debug, Default)]
pub struct MonotonicityIndexBatchBuilder {
    range: MonotonicityIndexBatchRange,
    kernel: Kernel,
}

impl MonotonicityIndexBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
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
    pub fn index_smooth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.index_smooth = (start, end, step);
        self
    }

    #[inline]
    pub fn mode(mut self, mode: MonotonicityIndexMode) -> Self {
        self.range.mode = mode;
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<MonotonicityIndexBatchOutput, MonotonicityIndexError> {
        monotonicity_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<MonotonicityIndexBatchOutput, MonotonicityIndexError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, MonotonicityIndexError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            let next = x.saturating_add(step);
            if next == x {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(MonotonicityIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_monotonicity_index(
    sweep: &MonotonicityIndexBatchRange,
) -> Result<Vec<MonotonicityIndexParams>, MonotonicityIndexError> {
    let lengths = expand_axis_usize(sweep.length)?;
    let index_smooths = expand_axis_usize(sweep.index_smooth)?;

    let mut combos = Vec::with_capacity(lengths.len() * index_smooths.len());
    for length in lengths {
        for index_smooth in index_smooths.iter().copied() {
            let combo = MonotonicityIndexParams {
                length: Some(length),
                mode: Some(sweep.mode),
                index_smooth: Some(index_smooth),
            };
            let _ = resolve_params(&combo)?;
            combos.push(combo);
        }
    }
    Ok(combos)
}

#[inline]
pub fn monotonicity_index_batch_with_kernel(
    data: &[f64],
    sweep: &MonotonicityIndexBatchRange,
    kernel: Kernel,
) -> Result<MonotonicityIndexBatchOutput, MonotonicityIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(MonotonicityIndexError::InvalidKernelForBatch(other)),
    };
    monotonicity_index_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn monotonicity_index_batch_slice(
    data: &[f64],
    sweep: &MonotonicityIndexBatchRange,
    kernel: Kernel,
) -> Result<MonotonicityIndexBatchOutput, MonotonicityIndexError> {
    monotonicity_index_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn monotonicity_index_batch_par_slice(
    data: &[f64],
    sweep: &MonotonicityIndexBatchRange,
    kernel: Kernel,
) -> Result<MonotonicityIndexBatchOutput, MonotonicityIndexError> {
    monotonicity_index_batch_inner(data, sweep, kernel, true)
}

#[inline]
pub fn monotonicity_index_batch_inner(
    data: &[f64],
    sweep: &MonotonicityIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<MonotonicityIndexBatchOutput, MonotonicityIndexError> {
    let combos = expand_grid_monotonicity_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(MonotonicityIndexError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(MonotonicityIndexError::AllValuesNaN);
    }

    let max_needed = combos
        .iter()
        .map(|combo| resolve_params(combo).unwrap().needed_valid)
        .max()
        .unwrap_or(0);
    let max_warmup = combos
        .iter()
        .map(|combo| resolve_params(combo).unwrap().warmup_period)
        .max()
        .unwrap_or(0);
    let valid = max_consecutive_valid_values(data);
    if valid < max_needed {
        return Err(MonotonicityIndexError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let mut index_mu = make_uninit_matrix(rows, cols);
    let mut cumulative_mean_mu = make_uninit_matrix(rows, cols);
    let mut upper_bound_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(
        &mut index_mu,
        cols,
        &vec![first.saturating_add(max_warmup).min(cols); rows],
    );
    init_matrix_prefixes(
        &mut cumulative_mean_mu,
        cols,
        &vec![first.saturating_add(max_warmup).min(cols); rows],
    );
    init_matrix_prefixes(
        &mut upper_bound_mu,
        cols,
        &vec![first.saturating_add(max_warmup).min(cols); rows],
    );

    let mut index_guard = ManuallyDrop::new(index_mu);
    let mut cumulative_mean_guard = ManuallyDrop::new(cumulative_mean_mu);
    let mut upper_bound_guard = ManuallyDrop::new(upper_bound_mu);
    let index_out = unsafe {
        std::slice::from_raw_parts_mut(index_guard.as_mut_ptr() as *mut f64, index_guard.len())
    };
    let cumulative_mean_out = unsafe {
        std::slice::from_raw_parts_mut(
            cumulative_mean_guard.as_mut_ptr() as *mut f64,
            cumulative_mean_guard.len(),
        )
    };
    let upper_bound_out = unsafe {
        std::slice::from_raw_parts_mut(
            upper_bound_guard.as_mut_ptr() as *mut f64,
            upper_bound_guard.len(),
        )
    };

    let combos = monotonicity_index_batch_inner_into(
        data,
        sweep,
        _kernel,
        parallel,
        index_out,
        cumulative_mean_out,
        upper_bound_out,
    )?;

    let index = unsafe {
        Vec::from_raw_parts(
            index_guard.as_mut_ptr() as *mut f64,
            index_guard.len(),
            index_guard.capacity(),
        )
    };
    let cumulative_mean = unsafe {
        Vec::from_raw_parts(
            cumulative_mean_guard.as_mut_ptr() as *mut f64,
            cumulative_mean_guard.len(),
            cumulative_mean_guard.capacity(),
        )
    };
    let upper_bound = unsafe {
        Vec::from_raw_parts(
            upper_bound_guard.as_mut_ptr() as *mut f64,
            upper_bound_guard.len(),
            upper_bound_guard.capacity(),
        )
    };

    Ok(MonotonicityIndexBatchOutput {
        index,
        cumulative_mean,
        upper_bound,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn monotonicity_index_batch_inner_into(
    data: &[f64],
    sweep: &MonotonicityIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
    index_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    upper_bound_out: &mut [f64],
) -> Result<Vec<MonotonicityIndexParams>, MonotonicityIndexError> {
    let combos = expand_grid_monotonicity_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(MonotonicityIndexError::EmptyInputData);
    }

    let total = rows
        .checked_mul(cols)
        .ok_or(MonotonicityIndexError::OutputLengthMismatch {
            expected: usize::MAX,
            index_got: index_out.len(),
            cumulative_mean_got: cumulative_mean_out.len(),
            upper_bound_got: upper_bound_out.len(),
        })?;
    if index_out.len() != total
        || cumulative_mean_out.len() != total
        || upper_bound_out.len() != total
    {
        return Err(MonotonicityIndexError::OutputLengthMismatch {
            expected: total,
            index_got: index_out.len(),
            cumulative_mean_got: cumulative_mean_out.len(),
            upper_bound_got: upper_bound_out.len(),
        });
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(MonotonicityIndexError::AllValuesNaN);
    }

    let max_needed = combos
        .iter()
        .map(|combo| resolve_params(combo).unwrap().needed_valid)
        .max()
        .unwrap_or(0);
    let valid = max_consecutive_valid_values(data);
    if valid < max_needed {
        return Err(MonotonicityIndexError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        index_out
            .par_chunks_mut(cols)
            .zip(cumulative_mean_out.par_chunks_mut(cols))
            .zip(upper_bound_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(row, ((index_row, cumulative_mean_row), upper_bound_row))| {
                    let params = resolve_params(&combos[row]).unwrap();
                    monotonicity_index_row_from_slice(
                        data,
                        params,
                        index_row,
                        cumulative_mean_row,
                        upper_bound_row,
                    );
                },
            );

        #[cfg(target_arch = "wasm32")]
        for (row, ((index_row, cumulative_mean_row), upper_bound_row)) in index_out
            .chunks_mut(cols)
            .zip(cumulative_mean_out.chunks_mut(cols))
            .zip(upper_bound_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            monotonicity_index_row_from_slice(
                data,
                params,
                index_row,
                cumulative_mean_row,
                upper_bound_row,
            );
        }
    } else {
        for (row, ((index_row, cumulative_mean_row), upper_bound_row)) in index_out
            .chunks_mut(cols)
            .zip(cumulative_mean_out.chunks_mut(cols))
            .zip(upper_bound_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            monotonicity_index_row_from_slice(
                data,
                params,
                index_row,
                cumulative_mean_row,
                upper_bound_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
fn parse_mode_py(value: &str) -> PyResult<MonotonicityIndexMode> {
    MonotonicityIndexMode::parse(value)
        .ok_or_else(|| PyValueError::new_err(format!("Invalid mode: {value}")))
}

#[cfg(feature = "python")]
#[pyfunction(name = "monotonicity_index")]
#[pyo3(signature = (
    data,
    length=DEFAULT_LENGTH,
    mode="efficiency",
    index_smooth=DEFAULT_INDEX_SMOOTH,
    kernel=None
))]
pub fn monotonicity_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    mode: &str,
    index_smooth: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = MonotonicityIndexInput::from_slice(
        data,
        MonotonicityIndexParams {
            length: Some(length),
            mode: Some(parse_mode_py(mode)?),
            index_smooth: Some(index_smooth),
        },
    );
    let output = py
        .allow_threads(|| monotonicity_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.index.into_pyarray(py),
        output.cumulative_mean.into_pyarray(py),
        output.upper_bound.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "MonotonicityIndexStream")]
pub struct MonotonicityIndexStreamPy {
    stream: MonotonicityIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MonotonicityIndexStreamPy {
    #[new]
    #[pyo3(signature = (
        length=DEFAULT_LENGTH,
        mode="efficiency",
        index_smooth=DEFAULT_INDEX_SMOOTH
    ))]
    fn new(length: usize, mode: &str, index_smooth: usize) -> PyResult<Self> {
        let stream = MonotonicityIndexStream::try_new(MonotonicityIndexParams {
            length: Some(length),
            mode: Some(parse_mode_py(mode)?),
            index_smooth: Some(index_smooth),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "monotonicity_index_batch")]
#[pyo3(signature = (
    data,
    length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
    index_smooth_range=(DEFAULT_INDEX_SMOOTH, DEFAULT_INDEX_SMOOTH, 0),
    mode="efficiency",
    kernel=None
))]
pub fn monotonicity_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    index_smooth_range: (usize, usize, usize),
    mode: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = MonotonicityIndexBatchRange {
        length: length_range,
        index_smooth: index_smooth_range,
        mode: parse_mode_py(mode)?,
    };
    let combos =
        expand_grid_monotonicity_index(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let index_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let cumulative_mean_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let upper_bound_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let index_slice = unsafe { index_arr.as_slice_mut()? };
    let cumulative_mean_slice = unsafe { cumulative_mean_arr.as_slice_mut()? };
    let upper_bound_slice = unsafe { upper_bound_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            monotonicity_index_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                true,
                index_slice,
                cumulative_mean_slice,
                upper_bound_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("index", index_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "cumulative_mean",
        cumulative_mean_arr.reshape((rows, cols))?,
    )?;
    dict.set_item("upper_bound", upper_bound_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "modes",
        PyList::new(
            py,
            combos
                .iter()
                .map(|combo| combo.mode.unwrap_or_default().as_str())
                .collect::<Vec<_>>(),
        )?,
    )?;
    dict.set_item(
        "index_smooths",
        combos
            .iter()
            .map(|combo| combo.index_smooth.unwrap_or(DEFAULT_INDEX_SMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_monotonicity_index_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(monotonicity_index_py, module)?)?;
    module.add_function(wrap_pyfunction!(monotonicity_index_batch_py, module)?)?;
    module.add_class::<MonotonicityIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn parse_mode_js(value: &str) -> Result<MonotonicityIndexMode, JsValue> {
    MonotonicityIndexMode::parse(value)
        .ok_or_else(|| JsValue::from_str(&format!("Invalid mode: {value}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MonotonicityIndexJsOutput {
    pub index: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub upper_bound: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "monotonicity_index_js")]
pub fn monotonicity_index_js(
    data: &[f64],
    length: usize,
    mode: &str,
    index_smooth: usize,
) -> Result<JsValue, JsValue> {
    let input = MonotonicityIndexInput::from_slice(
        data,
        MonotonicityIndexParams {
            length: Some(length),
            mode: Some(parse_mode_js(mode)?),
            index_smooth: Some(index_smooth),
        },
    );
    let output = monotonicity_index(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MonotonicityIndexJsOutput {
        index: output.index,
        cumulative_mean: output.cumulative_mean,
        upper_bound: output.upper_bound,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn monotonicity_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn monotonicity_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn monotonicity_index_into(
    in_ptr: *const f64,
    index_out_ptr: *mut f64,
    cumulative_mean_out_ptr: *mut f64,
    upper_bound_out_ptr: *mut f64,
    len: usize,
    length: usize,
    mode: &str,
    index_smooth: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || index_out_ptr.is_null()
        || cumulative_mean_out_ptr.is_null()
        || upper_bound_out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = MonotonicityIndexInput::from_slice(
            data,
            MonotonicityIndexParams {
                length: Some(length),
                mode: Some(parse_mode_js(mode)?),
                index_smooth: Some(index_smooth),
            },
        );
        let index_out = std::slice::from_raw_parts_mut(index_out_ptr, len);
        let cumulative_mean_out = std::slice::from_raw_parts_mut(cumulative_mean_out_ptr, len);
        let upper_bound_out = std::slice::from_raw_parts_mut(upper_bound_out_ptr, len);
        monotonicity_index_into_slices(
            index_out,
            cumulative_mean_out,
            upper_bound_out,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MonotonicityIndexBatchJsConfig {
    pub length_range: Option<(usize, usize, usize)>,
    pub index_smooth_range: Option<(usize, usize, usize)>,
    pub mode: Option<MonotonicityIndexMode>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MonotonicityIndexBatchJsOutput {
    pub index: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub combos: Vec<MonotonicityIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "monotonicity_index_batch_js")]
pub fn monotonicity_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MonotonicityIndexBatchJsConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let sweep = MonotonicityIndexBatchRange {
        length: config
            .length_range
            .unwrap_or((DEFAULT_LENGTH, DEFAULT_LENGTH, 0)),
        index_smooth: config.index_smooth_range.unwrap_or((
            DEFAULT_INDEX_SMOOTH,
            DEFAULT_INDEX_SMOOTH,
            0,
        )),
        mode: config.mode.unwrap_or_default(),
    };
    let output = monotonicity_index_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MonotonicityIndexBatchJsOutput {
        index: output.index,
        cumulative_mean: output.cumulative_mean,
        upper_bound: output.upper_bound,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn monotonicity_index_batch_into(
    in_ptr: *const f64,
    index_out_ptr: *mut f64,
    cumulative_mean_out_ptr: *mut f64,
    upper_bound_out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    index_smooth_start: usize,
    index_smooth_end: usize,
    index_smooth_step: usize,
    mode: &str,
) -> Result<usize, JsValue> {
    if in_ptr.is_null()
        || index_out_ptr.is_null()
        || cumulative_mean_out_ptr.is_null()
        || upper_bound_out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = MonotonicityIndexBatchRange {
        length: (length_start, length_end, length_step),
        index_smooth: (index_smooth_start, index_smooth_end, index_smooth_step),
        mode: parse_mode_js(mode)?,
    };

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let combos = expand_grid_monotonicity_index(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let index_out = std::slice::from_raw_parts_mut(index_out_ptr, total);
        let cumulative_mean_out = std::slice::from_raw_parts_mut(cumulative_mean_out_ptr, total);
        let upper_bound_out = std::slice::from_raw_parts_mut(upper_bound_out_ptr, total);
        let rows = monotonicity_index_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            index_out,
            cumulative_mean_out,
            upper_bound_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn monotonicity_index_output_into_js(
    data: &[f64],
    length: usize,
    mode: &str,
    index_smooth: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = monotonicity_index_js(data, length, mode, index_smooth)?;
    crate::write_wasm_object_f64_outputs("monotonicity_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn monotonicity_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = monotonicity_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "monotonicity_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::Candles;

    fn sample_source(length: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(length);
        for i in 0..length {
            let x = i as f64;
            out.push(100.0 + x * 0.05 + (x * 0.17).sin() * 2.4 + (x * 0.04).cos() * 0.8);
        }
        out
    }

    fn sample_candles(length: usize) -> Candles {
        let open: Vec<f64> = (0..length)
            .map(|i| 100.0 + i as f64 * 0.04 + (i as f64 * 0.08).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.13).cos() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.6 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.6 - (i as f64 * 0.03).cos().abs() * 0.2)
            .collect();
        Candles::new(
            (0..length as i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; length],
        )
    }

    fn assert_series_eq(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&lhs, &rhs) in left.iter().zip(right.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= tol, "lhs={lhs}, rhs={rhs}");
        }
    }

    #[test]
    fn monotonicity_index_output_contract() {
        let data = sample_source(256);
        let out = monotonicity_index(&MonotonicityIndexInput::from_slice(
            &data,
            MonotonicityIndexParams::default(),
        ))
        .unwrap();

        assert_eq!(out.index.len(), data.len());
        assert_eq!(out.cumulative_mean.len(), data.len());
        assert_eq!(out.upper_bound.len(), data.len());
        assert_eq!(out.index.iter().position(|v| v.is_finite()), Some(23));
        assert_eq!(
            out.cumulative_mean.iter().position(|v| v.is_finite()),
            Some(23)
        );
        assert_eq!(out.upper_bound.iter().position(|v| v.is_finite()), Some(23));
        assert!(out.index.last().copied().unwrap().is_finite());
        assert!(out.cumulative_mean.last().copied().unwrap().is_finite());
        assert!(out.upper_bound.last().copied().unwrap().is_finite());
    }

    #[test]
    fn monotonicity_index_rejects_invalid_parameters() {
        let data = sample_source(64);

        let err = monotonicity_index(&MonotonicityIndexInput::from_slice(
            &data,
            MonotonicityIndexParams {
                length: Some(1),
                ..MonotonicityIndexParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(err, MonotonicityIndexError::InvalidLength { .. }));

        let err = monotonicity_index(&MonotonicityIndexInput::from_slice(
            &data,
            MonotonicityIndexParams {
                index_smooth: Some(0),
                ..MonotonicityIndexParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            MonotonicityIndexError::InvalidIndexSmooth { .. }
        ));
    }

    #[test]
    fn monotonicity_index_builder_supports_candles() {
        let candles = sample_candles(220);
        let out = MonotonicityIndexBuilder::new()
            .mode(MonotonicityIndexMode::Complexity)
            .apply(&candles, "close")
            .unwrap();
        assert_eq!(out.index.len(), candles.close.len());
        assert_eq!(out.cumulative_mean.len(), candles.close.len());
        assert_eq!(out.upper_bound.len(), candles.close.len());
        assert!(out.index.last().copied().unwrap().is_finite());
    }

    #[test]
    fn monotonicity_index_stream_matches_batch_with_reset() {
        let mut data = sample_source(240);
        data[120] = f64::NAN;

        let batch = monotonicity_index(&MonotonicityIndexInput::from_slice(
            &data,
            MonotonicityIndexParams::default(),
        ))
        .unwrap();
        let mut stream =
            MonotonicityIndexStream::try_new(MonotonicityIndexParams::default()).unwrap();

        let mut index = Vec::with_capacity(data.len());
        let mut cumulative_mean = Vec::with_capacity(data.len());
        let mut upper_bound = Vec::with_capacity(data.len());
        for &value in &data {
            if let Some((idx, mean, upper)) = stream.update(value) {
                index.push(idx);
                cumulative_mean.push(mean);
                upper_bound.push(upper);
            } else {
                index.push(f64::NAN);
                cumulative_mean.push(f64::NAN);
                upper_bound.push(f64::NAN);
            }
        }

        assert_series_eq(&index, &batch.index, 1e-12);
        assert_series_eq(&cumulative_mean, &batch.cumulative_mean, 1e-12);
        assert_series_eq(&upper_bound, &batch.upper_bound, 1e-12);
    }

    #[test]
    fn monotonicity_index_batch_single_param_matches_single() {
        let data = sample_source(192);
        let sweep = MonotonicityIndexBatchRange::default();
        let batch = monotonicity_index_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        let single = monotonicity_index(&MonotonicityIndexInput::from_slice(
            &data,
            MonotonicityIndexParams::default(),
        ))
        .unwrap();

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        let (index_row, cumulative_mean_row, upper_bound_row) = batch.row_slices(0).unwrap();
        assert_series_eq(index_row, &single.index, 1e-12);
        assert_series_eq(cumulative_mean_row, &single.cumulative_mean, 1e-12);
        assert_series_eq(upper_bound_row, &single.upper_bound, 1e-12);
    }

    #[test]
    fn monotonicity_index_batch_metadata() {
        let data = sample_source(160);
        let sweep = MonotonicityIndexBatchRange {
            length: (18, 20, 2),
            index_smooth: (4, 5, 1),
            mode: MonotonicityIndexMode::Complexity,
        };
        let batch = monotonicity_index_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();

        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.index.len(), 4 * data.len());
        assert_eq!(batch.cumulative_mean.len(), 4 * data.len());
        assert_eq!(batch.upper_bound.len(), 4 * data.len());
        assert_eq!(
            batch.row_for_params(&MonotonicityIndexParams {
                length: Some(20),
                mode: Some(MonotonicityIndexMode::Complexity),
                index_smooth: Some(5),
            }),
            Some(3)
        );
    }
}
