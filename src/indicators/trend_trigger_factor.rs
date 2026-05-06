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

const DEFAULT_LENGTH: usize = 15;

#[derive(Debug, Clone)]
pub enum TrendTriggerFactorData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct TrendTriggerFactorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TrendTriggerFactorParams {
    pub length: Option<usize>,
}

impl Default for TrendTriggerFactorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrendTriggerFactorInput<'a> {
    pub data: TrendTriggerFactorData<'a>,
    pub params: TrendTriggerFactorParams,
}

impl<'a> TrendTriggerFactorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: TrendTriggerFactorParams) -> Self {
        Self {
            data: TrendTriggerFactorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: TrendTriggerFactorParams) -> Self {
        Self {
            data: TrendTriggerFactorData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, TrendTriggerFactorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrendTriggerFactorBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for TrendTriggerFactorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrendTriggerFactorBuilder {
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
    ) -> Result<TrendTriggerFactorOutput, TrendTriggerFactorError> {
        let input = TrendTriggerFactorInput::from_candles(
            candles,
            TrendTriggerFactorParams {
                length: self.length,
            },
        );
        trend_trigger_factor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<TrendTriggerFactorOutput, TrendTriggerFactorError> {
        let input = TrendTriggerFactorInput::from_slices(
            high,
            low,
            TrendTriggerFactorParams {
                length: self.length,
            },
        );
        trend_trigger_factor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<TrendTriggerFactorStream, TrendTriggerFactorError> {
        TrendTriggerFactorStream::try_new(TrendTriggerFactorParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum TrendTriggerFactorError {
    #[error("trend_trigger_factor: Input data slice is empty.")]
    EmptyInputData,
    #[error("trend_trigger_factor: All values are NaN.")]
    AllValuesNaN,
    #[error("trend_trigger_factor: Inconsistent slice lengths: high={high_len}, low={low_len}")]
    InconsistentSliceLengths { high_len: usize, low_len: usize },
    #[error("trend_trigger_factor: Invalid length: length={length}, data length={data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("trend_trigger_factor: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("trend_trigger_factor: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("trend_trigger_factor: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("trend_trigger_factor: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn extract_high_low<'a>(
    input: &'a TrendTriggerFactorInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), TrendTriggerFactorError> {
    let (high, low) = match &input.data {
        TrendTriggerFactorData::Candles { candles } => {
            (candles.high.as_slice(), candles.low.as_slice())
        }
        TrendTriggerFactorData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(TrendTriggerFactorError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(TrendTriggerFactorError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    Ok((high, low))
}

#[inline(always)]
fn first_valid_high_low(high: &[f64], low: &[f64]) -> Option<usize> {
    (0..high.len()).find(|&i| high[i].is_finite() && low[i].is_finite())
}

#[inline(always)]
fn prepare<'a>(
    input: &'a TrendTriggerFactorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, Kernel), TrendTriggerFactorError> {
    let (high, low) = extract_high_low(input)?;
    let len = high.len();
    let length = input.get_length();
    if length == 0 || length > len {
        return Err(TrendTriggerFactorError::InvalidLength {
            length,
            data_len: len,
        });
    }
    let first = first_valid_high_low(high, low).ok_or(TrendTriggerFactorError::AllValuesNaN)?;
    let valid = len.saturating_sub(first);
    if valid < length {
        return Err(TrendTriggerFactorError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }
    Ok((high, low, length, first, kernel.to_non_batch()))
}

#[inline(always)]
fn calc_ttf(hh: f64, ll: f64, hist_hh: f64, hist_ll: f64) -> f64 {
    let buy_power = hh - hist_ll;
    let sell_power = hist_hh - ll;
    let denom = buy_power + sell_power;
    if denom.is_finite() && denom != 0.0 {
        200.0 * (buy_power - sell_power) / denom
    } else {
        f64::NAN
    }
}

#[derive(Clone, Debug)]
struct IndexMonoQueue {
    idx: Vec<usize>,
    head: usize,
    tail: usize,
    count: usize,
}

impl IndexMonoQueue {
    fn new(window: usize) -> Self {
        Self {
            idx: vec![0; window.max(1) + 1],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn cap(&self) -> usize {
        self.idx.len()
    }

    #[inline(always)]
    fn back_pos(&self) -> usize {
        if self.tail == 0 {
            self.cap() - 1
        } else {
            self.tail - 1
        }
    }

    #[inline(always)]
    fn evict(&mut self, window_start: usize) {
        while self.count > 0 && self.idx[self.head] < window_start {
            self.head += 1;
            if self.head == self.cap() {
                self.head = 0;
            }
            self.count -= 1;
        }
    }

    #[inline(always)]
    fn push_max(&mut self, idx: usize, values: &[f64]) {
        let value = values[idx];
        while self.count > 0 {
            let back = self.back_pos();
            if values[self.idx[back]] <= value {
                self.tail = back;
                self.count -= 1;
            } else {
                break;
            }
        }
        self.idx[self.tail] = idx;
        self.tail += 1;
        if self.tail == self.cap() {
            self.tail = 0;
        }
        self.count += 1;
    }

    #[inline(always)]
    fn push_min(&mut self, idx: usize, values: &[f64]) {
        let value = values[idx];
        while self.count > 0 {
            let back = self.back_pos();
            if values[self.idx[back]] >= value {
                self.tail = back;
                self.count -= 1;
            } else {
                break;
            }
        }
        self.idx[self.tail] = idx;
        self.tail += 1;
        if self.tail == self.cap() {
            self.tail = 0;
        }
        self.count += 1;
    }

    #[inline(always)]
    fn front(&self) -> Option<usize> {
        if self.count == 0 {
            None
        } else {
            Some(self.idx[self.head])
        }
    }
}

#[inline(always)]
fn compute_trend_trigger_factor_into(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    let warm = first + length - 1;
    let mut maxq = IndexMonoQueue::new(length);
    let mut minq = IndexMonoQueue::new(length);
    let mut hh_history = vec![0.0f64; length];
    let mut ll_history = vec![0.0f64; length];
    let mut hist_head = 0usize;
    let mut hist_len = 0usize;

    for i in first..high.len() {
        let h = high[i];
        let l = low[i];
        if !h.is_finite() || !l.is_finite() {
            if i >= warm {
                out[i] = f64::NAN;
            }
            continue;
        }

        let window_start = i.saturating_add(1).saturating_sub(length).max(first);

        maxq.evict(window_start);
        minq.evict(window_start);
        maxq.push_max(i, high);
        minq.push_min(i, low);

        if i >= warm {
            let hh = high[maxq.front().unwrap()];
            let ll = low[minq.front().unwrap()];
            let hist_hh = if hist_len == length {
                hh_history[hist_head]
            } else {
                0.0
            };
            let hist_ll = if hist_len == length {
                ll_history[hist_head]
            } else {
                0.0
            };
            out[i] = calc_ttf(hh, ll, hist_hh, hist_ll);

            if hist_len < length {
                let pos = (hist_head + hist_len) % length;
                hh_history[pos] = hh;
                ll_history[pos] = ll;
                hist_len += 1;
            } else {
                hh_history[hist_head] = hh;
                ll_history[hist_head] = ll;
                hist_head += 1;
                if hist_head == length {
                    hist_head = 0;
                }
            }
        }
    }
}

#[inline]
pub fn trend_trigger_factor(
    input: &TrendTriggerFactorInput,
) -> Result<TrendTriggerFactorOutput, TrendTriggerFactorError> {
    trend_trigger_factor_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn trend_trigger_factor_with_kernel(
    input: &TrendTriggerFactorInput,
    kernel: Kernel,
) -> Result<TrendTriggerFactorOutput, TrendTriggerFactorError> {
    let (high, low, length, first, chosen) = prepare(input, kernel)?;
    let _ = chosen;
    let warm = first + length - 1;
    let mut out = alloc_with_nan_prefix(high.len(), warm);
    compute_trend_trigger_factor_into(high, low, length, first, &mut out);
    Ok(TrendTriggerFactorOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn trend_trigger_factor_into(
    input: &TrendTriggerFactorInput,
    out: &mut [f64],
) -> Result<(), TrendTriggerFactorError> {
    trend_trigger_factor_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn trend_trigger_factor_into_slice(
    out: &mut [f64],
    input: &TrendTriggerFactorInput,
    kernel: Kernel,
) -> Result<(), TrendTriggerFactorError> {
    let (high, low, length, first, chosen) = prepare(input, kernel)?;
    let _ = chosen;
    if out.len() != high.len() {
        return Err(TrendTriggerFactorError::OutputLengthMismatch {
            expected: high.len(),
            got: out.len(),
        });
    }
    out.fill(f64::NAN);
    compute_trend_trigger_factor_into(high, low, length, first, out);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct TrendTriggerFactorStream {
    length: usize,
    index: usize,
    maxq: VecDeque<(usize, f64)>,
    minq: VecDeque<(usize, f64)>,
    hh_history: VecDeque<f64>,
    ll_history: VecDeque<f64>,
}

impl TrendTriggerFactorStream {
    pub fn try_new(params: TrendTriggerFactorParams) -> Result<Self, TrendTriggerFactorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length == 0 {
            return Err(TrendTriggerFactorError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self {
            length,
            index: 0,
            maxq: VecDeque::with_capacity(length + 1),
            minq: VecDeque::with_capacity(length + 1),
            hh_history: VecDeque::with_capacity(length + 1),
            ll_history: VecDeque::with_capacity(length + 1),
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        let idx = self.index;
        self.index = self.index.saturating_add(1);

        if !high.is_finite() || !low.is_finite() {
            return f64::NAN;
        }

        let window_start = idx.saturating_add(1).saturating_sub(self.length);

        while let Some(&(front_idx, _)) = self.maxq.front() {
            if front_idx < window_start {
                self.maxq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(front_idx, _)) = self.minq.front() {
            if front_idx < window_start {
                self.minq.pop_front();
            } else {
                break;
            }
        }

        while let Some(&(_, back_val)) = self.maxq.back() {
            if back_val <= high {
                self.maxq.pop_back();
            } else {
                break;
            }
        }
        self.maxq.push_back((idx, high));

        while let Some(&(_, back_val)) = self.minq.back() {
            if back_val >= low {
                self.minq.pop_back();
            } else {
                break;
            }
        }
        self.minq.push_back((idx, low));

        if idx + 1 < self.length {
            return f64::NAN;
        }

        let hh = self.maxq.front().map(|(_, v)| *v).unwrap_or(high);
        let ll = self.minq.front().map(|(_, v)| *v).unwrap_or(low);
        let hist_hh = if self.hh_history.len() == self.length {
            self.hh_history.front().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let hist_ll = if self.ll_history.len() == self.length {
            self.ll_history.front().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let out = calc_ttf(hh, ll, hist_hh, hist_ll);

        self.hh_history.push_back(hh);
        self.ll_history.push_back(ll);
        if self.hh_history.len() > self.length {
            self.hh_history.pop_front();
        }
        if self.ll_history.len() > self.length {
            self.ll_history.pop_front();
        }

        out
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.length.saturating_sub(1)
    }
}

#[derive(Debug, Clone)]
pub struct TrendTriggerFactorBatchRange {
    pub length: (usize, usize, usize),
}

#[derive(Debug, Clone)]
pub struct TrendTriggerFactorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrendTriggerFactorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct TrendTriggerFactorBatchBuilder {
    length: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for TrendTriggerFactorBatchBuilder {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl TrendTriggerFactorBatchBuilder {
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
    ) -> Result<TrendTriggerFactorBatchOutput, TrendTriggerFactorError> {
        trend_trigger_factor_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            &TrendTriggerFactorBatchRange {
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
    ) -> Result<TrendTriggerFactorBatchOutput, TrendTriggerFactorError> {
        trend_trigger_factor_batch_with_kernel(
            high,
            low,
            &TrendTriggerFactorBatchRange {
                length: self.length,
            },
            self.kernel,
        )
    }
}

pub fn expand_grid(
    sweep: &TrendTriggerFactorBatchRange,
) -> Result<Vec<TrendTriggerFactorParams>, TrendTriggerFactorError> {
    let (start, end, step) = sweep.length;
    if start == 0 {
        return Err(TrendTriggerFactorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut lengths = Vec::new();
    if step == 0 {
        if start != end {
            return Err(TrendTriggerFactorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        lengths.push(start);
    } else {
        if start > end {
            return Err(TrendTriggerFactorError::InvalidRange {
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
        .map(|length| TrendTriggerFactorParams {
            length: Some(length),
        })
        .collect())
}

pub fn trend_trigger_factor_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &TrendTriggerFactorBatchRange,
    kernel: Kernel,
) -> Result<TrendTriggerFactorBatchOutput, TrendTriggerFactorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(TrendTriggerFactorError::InvalidKernelForBatch(kernel)),
    };
    trend_trigger_factor_batch_par_slice(high, low, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn trend_trigger_factor_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &TrendTriggerFactorBatchRange,
    kernel: Kernel,
) -> Result<TrendTriggerFactorBatchOutput, TrendTriggerFactorError> {
    trend_trigger_factor_batch_inner(high, low, sweep, kernel, false)
}

#[inline(always)]
pub fn trend_trigger_factor_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &TrendTriggerFactorBatchRange,
    kernel: Kernel,
) -> Result<TrendTriggerFactorBatchOutput, TrendTriggerFactorError> {
    trend_trigger_factor_batch_inner(high, low, sweep, kernel, true)
}

fn validate_raw_slices(high: &[f64], low: &[f64]) -> Result<usize, TrendTriggerFactorError> {
    if high.is_empty() || low.is_empty() {
        return Err(TrendTriggerFactorError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(TrendTriggerFactorError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    first_valid_high_low(high, low).ok_or(TrendTriggerFactorError::AllValuesNaN)
}

fn trend_trigger_factor_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &TrendTriggerFactorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<TrendTriggerFactorBatchOutput, TrendTriggerFactorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(high, low)?;
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap())
        .max()
        .unwrap();
    let valid = high.len().saturating_sub(first);
    if valid < max_length {
        return Err(TrendTriggerFactorError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.length.unwrap() - 1)
        .collect();

    let mut buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf, cols, &warmups);
    let mut guard = ManuallyDrop::new(buf);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    trend_trigger_factor_batch_inner_into(high, low, sweep, kernel, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(TrendTriggerFactorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn trend_trigger_factor_batch_into_slice(
    out: &mut [f64],
    high: &[f64],
    low: &[f64],
    sweep: &TrendTriggerFactorBatchRange,
    kernel: Kernel,
) -> Result<(), TrendTriggerFactorError> {
    trend_trigger_factor_batch_inner_into(high, low, sweep, kernel, false, out)?;
    Ok(())
}

fn trend_trigger_factor_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &TrendTriggerFactorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TrendTriggerFactorParams>, TrendTriggerFactorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(high, low)?;
    let rows = combos.len();
    let cols = high.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| TrendTriggerFactorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;
    if out.len() != expected {
        return Err(TrendTriggerFactorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap())
        .max()
        .unwrap();
    let valid = cols.saturating_sub(first);
    if valid < max_length {
        return Err(TrendTriggerFactorError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let do_row = |row: usize, dst: &mut [f64]| {
        dst.fill(f64::NAN);
        compute_trend_trigger_factor_into(high, low, combos[row].length.unwrap(), first, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, dst)| do_row(row, dst));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                do_row(row, dst);
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            do_row(row, dst);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_trigger_factor")]
#[pyo3(signature = (high, low, length=15, kernel=None))]
pub fn trend_trigger_factor_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let input = TrendTriggerFactorInput::from_slices(
        high,
        low,
        TrendTriggerFactorParams {
            length: Some(length),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| trend_trigger_factor_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrendTriggerFactorStream")]
pub struct TrendTriggerFactorStreamPy {
    stream: TrendTriggerFactorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrendTriggerFactorStreamPy {
    #[new]
    #[pyo3(signature = (length=15))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = TrendTriggerFactorStream::try_new(TrendTriggerFactorParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> f64 {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_trigger_factor_batch")]
#[pyo3(signature = (high, low, length_range=(15,15,0), kernel=None))]
pub fn trend_trigger_factor_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let sweep = TrendTriggerFactorBatchRange {
        length: length_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        trend_trigger_factor_batch_inner_into(
            high,
            low,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out.reshape((rows, cols))?)?;
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
pub fn register_trend_trigger_factor_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trend_trigger_factor_py, m)?)?;
    m.add_function(wrap_pyfunction!(trend_trigger_factor_batch_py, m)?)?;
    m.add_class::<TrendTriggerFactorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TrendTriggerFactorBatchConfig {
    pub length_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TrendTriggerFactorBatchJsOutput {
    pub values: Vec<f64>,
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
#[wasm_bindgen(js_name = "trend_trigger_factor_js")]
pub fn trend_trigger_factor_js(
    high: &[f64],
    low: &[f64],
    length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = TrendTriggerFactorInput::from_slices(
        high,
        low,
        TrendTriggerFactorParams {
            length: Some(length),
        },
    );
    trend_trigger_factor_with_kernel(&input, Kernel::Auto)
        .map(|out| out.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_trigger_factor_batch_js")]
pub fn trend_trigger_factor_batch_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: TrendTriggerFactorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = TrendTriggerFactorBatchRange {
        length: js_vec3_to_usize("length_range", &config.length_range)?,
    };
    let out = trend_trigger_factor_batch_with_kernel(high, low, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let lengths = out
        .combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
        .collect();
    serde_wasm_bindgen::to_value(&TrendTriggerFactorBatchJsOutput {
        values: out.values,
        lengths,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_trigger_factor_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_trigger_factor_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_trigger_factor_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = TrendTriggerFactorInput::from_slices(
            high,
            low,
            TrendTriggerFactorParams {
                length: Some(length),
            },
        );
        trend_trigger_factor_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_trigger_factor_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    start: usize,
    end: usize,
    step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let sweep = TrendTriggerFactorBatchRange {
            length: (start, end, step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        trend_trigger_factor_batch_into_slice(out, high, low, &sweep, Kernel::Scalar)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_trigger_factor_output_into_js(
    high: &[f64],
    low: &[f64],
    length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = trend_trigger_factor_js(high, low, length)?;
    crate::write_wasm_f64_output("trend_trigger_factor_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_trigger_factor_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_trigger_factor_batch_js(high, low, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "trend_trigger_factor_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::enums::Kernel;

    fn sample_high_low(len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + (i as f64 * 0.17).sin() * 2.0 + (i as f64) * 0.05;
            high.push(base + 1.5 + (i as f64 * 0.11).cos() * 0.2);
            low.push(base - 1.5 - (i as f64 * 0.07).sin() * 0.2);
        }
        (high, low)
    }

    fn manual_ttf(high: &[f64], low: &[f64], length: usize) -> Vec<f64> {
        let n = high.len();
        let mut out = vec![f64::NAN; n];
        let first = first_valid_high_low(high, low).unwrap();
        let warm = first + length - 1;
        let mut hh_series = vec![f64::NAN; n];
        let mut ll_series = vec![f64::NAN; n];

        for i in warm..n {
            let start = i + 1 - length;
            let mut hh = f64::NEG_INFINITY;
            let mut ll = f64::INFINITY;
            for j in start..=i {
                hh = hh.max(high[j]);
                ll = ll.min(low[j]);
            }
            hh_series[i] = hh;
            ll_series[i] = ll;
            let hist_hh = if i >= length && hh_series[i - length].is_finite() {
                hh_series[i - length]
            } else {
                0.0
            };
            let hist_ll = if i >= length && ll_series[i - length].is_finite() {
                ll_series[i - length]
            } else {
                0.0
            };
            out[i] = calc_ttf(hh, ll, hist_hh, hist_ll);
        }

        out
    }

    fn assert_vec_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g.is_nan() || w.is_nan() {
                assert!(g.is_nan() && w.is_nan(), "index={idx} got={g} want={w}");
            } else {
                assert!((g - w).abs() <= 1e-12, "index={idx} got={g} want={w}");
            }
        }
    }

    #[test]
    fn manual_reference_matches_api() {
        let (high, low) = sample_high_low(96);
        let expected = manual_ttf(&high, &low, 15);
        let input = TrendTriggerFactorInput::from_slices(
            &high,
            &low,
            TrendTriggerFactorParams { length: Some(15) },
        );
        let out = trend_trigger_factor(&input).unwrap();
        for (got, want) in out.values.iter().zip(expected.iter()) {
            if got.is_nan() || want.is_nan() {
                assert!(got.is_nan() && want.is_nan());
            } else {
                assert!((got - want).abs() <= 1e-12, "got={got} want={want}");
            }
        }
    }

    #[test]
    fn stream_matches_batch() {
        let (high, low) = sample_high_low(128);
        let input = TrendTriggerFactorInput::from_slices(
            &high,
            &low,
            TrendTriggerFactorParams { length: Some(15) },
        );
        let batch = trend_trigger_factor(&input).unwrap();
        let mut stream =
            TrendTriggerFactorStream::try_new(TrendTriggerFactorParams { length: Some(15) })
                .unwrap();
        for i in 0..high.len() {
            let got = stream.update(high[i], low[i]);
            let want = batch.values[i];
            if got.is_nan() || want.is_nan() {
                assert!(got.is_nan() && want.is_nan());
            } else {
                assert!(
                    (got - want).abs() <= 1e-12,
                    "index={i} got={got} want={want}"
                );
            }
        }
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (high, low) = sample_high_low(144);
        let batch = trend_trigger_factor_batch_with_kernel(
            &high,
            &low,
            &TrendTriggerFactorBatchRange {
                length: (15, 17, 2),
            },
            Kernel::Auto,
        )
        .unwrap();
        let single = trend_trigger_factor(&TrendTriggerFactorInput::from_slices(
            &high,
            &low,
            TrendTriggerFactorParams { length: Some(15) },
        ))
        .unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, high.len());
        assert_vec_close(&batch.values[..high.len()], single.values.as_slice());
    }

    #[test]
    fn into_slice_matches_single() {
        let (high, low) = sample_high_low(120);
        let input = TrendTriggerFactorInput::from_slices(
            &high,
            &low,
            TrendTriggerFactorParams { length: Some(15) },
        );
        let single = trend_trigger_factor(&input).unwrap();
        let mut out = vec![0.0; high.len()];
        trend_trigger_factor_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_vec_close(&out, &single.values);
    }

    #[test]
    fn invalid_length_is_rejected() {
        let (high, low) = sample_high_low(32);
        let input = TrendTriggerFactorInput::from_slices(
            &high,
            &low,
            TrendTriggerFactorParams { length: Some(0) },
        );
        let err = trend_trigger_factor(&input).unwrap_err();
        assert!(err.to_string().contains("Invalid length"));
    }
}
