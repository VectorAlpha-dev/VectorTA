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
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_DAILY_LIMIT: f64 = 10_000.0;

#[derive(Debug, Clone)]
pub enum AccumulationSwingIndexData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AccumulationSwingIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AccumulationSwingIndexParams {
    pub daily_limit: Option<f64>,
}

impl Default for AccumulationSwingIndexParams {
    fn default() -> Self {
        Self {
            daily_limit: Some(DEFAULT_DAILY_LIMIT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccumulationSwingIndexInput<'a> {
    pub data: AccumulationSwingIndexData<'a>,
    pub params: AccumulationSwingIndexParams,
}

impl<'a> AccumulationSwingIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AccumulationSwingIndexParams) -> Self {
        Self {
            data: AccumulationSwingIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AccumulationSwingIndexParams,
    ) -> Self {
        Self {
            data: AccumulationSwingIndexData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AccumulationSwingIndexParams::default())
    }

    #[inline]
    pub fn get_daily_limit(&self) -> f64 {
        self.params.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AccumulationSwingIndexBuilder {
    daily_limit: Option<f64>,
    kernel: Kernel,
}

impl Default for AccumulationSwingIndexBuilder {
    fn default() -> Self {
        Self {
            daily_limit: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AccumulationSwingIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn daily_limit(mut self, value: f64) -> Self {
        self.daily_limit = Some(value);
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
    ) -> Result<AccumulationSwingIndexOutput, AccumulationSwingIndexError> {
        let input = AccumulationSwingIndexInput::from_candles(
            candles,
            AccumulationSwingIndexParams {
                daily_limit: self.daily_limit,
            },
        );
        accumulation_swing_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AccumulationSwingIndexOutput, AccumulationSwingIndexError> {
        let input = AccumulationSwingIndexInput::from_slices(
            open,
            high,
            low,
            close,
            AccumulationSwingIndexParams {
                daily_limit: self.daily_limit,
            },
        );
        accumulation_swing_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AccumulationSwingIndexStream, AccumulationSwingIndexError> {
        AccumulationSwingIndexStream::try_new(AccumulationSwingIndexParams {
            daily_limit: self.daily_limit,
        })
    }
}

#[derive(Debug, Error)]
pub enum AccumulationSwingIndexError {
    #[error("accumulation_swing_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("accumulation_swing_index: All values are NaN.")]
    AllValuesNaN,
    #[error("accumulation_swing_index: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("accumulation_swing_index: Invalid daily_limit: {daily_limit}")]
    InvalidDailyLimit { daily_limit: f64 },
    #[error("accumulation_swing_index: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("accumulation_swing_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("accumulation_swing_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn extract_ohlc<'a>(
    input: &'a AccumulationSwingIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), AccumulationSwingIndexError> {
    let (open, high, low, close) = match &input.data {
        AccumulationSwingIndexData::Candles { candles } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        AccumulationSwingIndexData::Slices {
            open,
            high,
            low,
            close,
        } => (*open, *high, *low, *close),
    };

    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(AccumulationSwingIndexError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(AccumulationSwingIndexError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    Ok((open, high, low, close))
}

#[inline(always)]
fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| {
        open[i].is_finite() && high[i].is_finite() && low[i].is_finite() && close[i].is_finite()
    })
}

#[inline(always)]
fn prepare<'a>(
    input: &'a AccumulationSwingIndexInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        f64,
        usize,
        Kernel,
    ),
    AccumulationSwingIndexError,
> {
    let (open, high, low, close) = extract_ohlc(input)?;
    let daily_limit = input.get_daily_limit();
    if !daily_limit.is_finite() || daily_limit <= 0.0 {
        return Err(AccumulationSwingIndexError::InvalidDailyLimit { daily_limit });
    }
    let first = first_valid_ohlc(open, high, low, close)
        .ok_or(AccumulationSwingIndexError::AllValuesNaN)?;
    Ok((
        open,
        high,
        low,
        close,
        daily_limit,
        first,
        kernel.to_non_batch(),
    ))
}

#[inline(always)]
fn compute_increment(
    prev_open: f64,
    prev_close: f64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    daily_limit: f64,
) -> f64 {
    let abs_high_close = (high - prev_close).abs();
    let abs_low_close = (low - prev_close).abs();
    let abs_close_open = (prev_close - prev_open).abs();
    let k = if abs_high_close >= abs_low_close {
        abs_high_close
    } else {
        abs_low_close
    };
    let range = high - low;
    let r = if abs_high_close >= abs_low_close {
        if abs_high_close >= range {
            abs_high_close - 0.5 * abs_low_close + 0.25 * abs_close_open
        } else {
            range + 0.25 * abs_close_open
        }
    } else if abs_low_close >= range {
        abs_low_close - 0.5 * abs_high_close + 0.25 * abs_close_open
    } else {
        range + 0.25 * abs_close_open
    };

    if r != 0.0 {
        50.0 * (((close - prev_close) + 0.5 * (close - open) + 0.25 * (prev_close - prev_open)) / r)
            * k
            / daily_limit
    } else {
        0.0
    }
}

#[inline(always)]
fn compute_accumulation_swing_index_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    daily_limit: f64,
    first: usize,
    out: &mut [f64],
) {
    let n = close.len();
    if first >= n {
        return;
    }

    let mut accum = 0.0;
    out[first] = 0.0;
    let mut prev_open = open[first];
    let mut prev_close = close[first];

    let mut i = first + 1;
    while i < n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if o.is_finite()
            && h.is_finite()
            && l.is_finite()
            && c.is_finite()
            && prev_open.is_finite()
            && prev_close.is_finite()
        {
            let delta = compute_increment(prev_open, prev_close, o, h, l, c, daily_limit);
            if delta.is_finite() {
                accum += delta;
            }
        }
        out[i] = accum;
        prev_open = o;
        prev_close = c;
        i += 1;
    }
}

#[inline]
pub fn accumulation_swing_index(
    input: &AccumulationSwingIndexInput,
) -> Result<AccumulationSwingIndexOutput, AccumulationSwingIndexError> {
    accumulation_swing_index_with_kernel(input, Kernel::Auto)
}

pub fn accumulation_swing_index_with_kernel(
    input: &AccumulationSwingIndexInput,
    kernel: Kernel,
) -> Result<AccumulationSwingIndexOutput, AccumulationSwingIndexError> {
    let (open, high, low, close, daily_limit, first, _) = prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(close.len(), first);
    compute_accumulation_swing_index_into(open, high, low, close, daily_limit, first, &mut out);
    Ok(AccumulationSwingIndexOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn accumulation_swing_index_into(
    out: &mut [f64],
    input: &AccumulationSwingIndexInput,
    kernel: Kernel,
) -> Result<(), AccumulationSwingIndexError> {
    accumulation_swing_index_into_slice(out, input, kernel)
}

pub fn accumulation_swing_index_into_slice(
    out: &mut [f64],
    input: &AccumulationSwingIndexInput,
    kernel: Kernel,
) -> Result<(), AccumulationSwingIndexError> {
    let (open, high, low, close, daily_limit, first, _) = prepare(input, kernel)?;
    let expected = close.len();
    if out.len() != expected {
        return Err(AccumulationSwingIndexError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    out[..first.min(expected)].fill(f64::NAN);
    compute_accumulation_swing_index_into(open, high, low, close, daily_limit, first, out);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct AccumulationSwingIndexStream {
    daily_limit: f64,
    started: bool,
    prev_open: f64,
    prev_close: f64,
    accum: f64,
}

impl AccumulationSwingIndexStream {
    pub fn try_new(
        params: AccumulationSwingIndexParams,
    ) -> Result<Self, AccumulationSwingIndexError> {
        let daily_limit = params.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT);
        if !daily_limit.is_finite() || daily_limit <= 0.0 {
            return Err(AccumulationSwingIndexError::InvalidDailyLimit { daily_limit });
        }
        Ok(Self {
            daily_limit,
            started: false,
            prev_open: f64::NAN,
            prev_close: f64::NAN,
            accum: 0.0,
        })
    }

    #[inline]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        if !self.started {
            if open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite() {
                self.started = true;
                self.prev_open = open;
                self.prev_close = close;
                self.accum = 0.0;
                return 0.0;
            }
            return f64::NAN;
        }

        if open.is_finite()
            && high.is_finite()
            && low.is_finite()
            && close.is_finite()
            && self.prev_open.is_finite()
            && self.prev_close.is_finite()
        {
            let delta = compute_increment(
                self.prev_open,
                self.prev_close,
                open,
                high,
                low,
                close,
                self.daily_limit,
            );
            if delta.is_finite() {
                self.accum += delta;
            }
        }

        self.prev_open = open;
        self.prev_close = close;
        self.accum
    }
}

#[derive(Debug, Clone)]
pub struct AccumulationSwingIndexBatchRange {
    pub daily_limit: (f64, f64, f64),
}

#[derive(Debug, Clone)]
pub struct AccumulationSwingIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AccumulationSwingIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AccumulationSwingIndexBatchOutput {
    pub fn row_for_params(&self, params: &AccumulationSwingIndexParams) -> Option<usize> {
        let daily_limit = params.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT);
        self.combos.iter().position(|combo| {
            (combo.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT) - daily_limit).abs() <= 1e-12
        })
    }

    pub fn values_for(&self, params: &AccumulationSwingIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.values.get(start..start + self.cols)
        })
    }
}

#[derive(Clone, Debug)]
pub struct AccumulationSwingIndexBatchBuilder {
    range: AccumulationSwingIndexBatchRange,
    kernel: Kernel,
}

impl Default for AccumulationSwingIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: AccumulationSwingIndexBatchRange {
                daily_limit: (DEFAULT_DAILY_LIMIT, DEFAULT_DAILY_LIMIT, 0.0),
            },
            kernel: Kernel::Auto,
        }
    }
}

impl AccumulationSwingIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn daily_limit_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.daily_limit = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn daily_limit_static(mut self, value: f64) -> Self {
        self.range.daily_limit = (value, value, 0.0);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AccumulationSwingIndexBatchOutput, AccumulationSwingIndexError> {
        accumulation_swing_index_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<AccumulationSwingIndexBatchOutput, AccumulationSwingIndexError> {
        self.apply_slices(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }
}

#[inline(always)]
fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, AccumulationSwingIndexError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(AccumulationSwingIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }
    let step_abs = step.abs();
    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end + 1e-12 {
            out.push(x);
            x += step_abs;
        }
    } else {
        let mut x = start;
        while x >= end - 1e-12 {
            out.push(x);
            x -= step_abs;
        }
    }
    if out.is_empty() {
        return Err(AccumulationSwingIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid(
    range: &AccumulationSwingIndexBatchRange,
) -> Result<Vec<AccumulationSwingIndexParams>, AccumulationSwingIndexError> {
    Ok(axis_f64(range.daily_limit)?
        .into_iter()
        .map(|daily_limit| AccumulationSwingIndexParams {
            daily_limit: Some(daily_limit),
        })
        .collect())
}

pub fn accumulation_swing_index_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AccumulationSwingIndexBatchRange,
    kernel: Kernel,
) -> Result<AccumulationSwingIndexBatchOutput, AccumulationSwingIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AccumulationSwingIndexError::InvalidKernelForBatch(kernel)),
    };
    accumulation_swing_index_batch_par_slice(
        open,
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn accumulation_swing_index_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AccumulationSwingIndexBatchRange,
    kernel: Kernel,
) -> Result<AccumulationSwingIndexBatchOutput, AccumulationSwingIndexError> {
    accumulation_swing_index_batch_inner(open, high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn accumulation_swing_index_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AccumulationSwingIndexBatchRange,
    kernel: Kernel,
) -> Result<AccumulationSwingIndexBatchOutput, AccumulationSwingIndexError> {
    accumulation_swing_index_batch_inner(open, high, low, close, sweep, kernel, true)
}

fn validate_raw_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<usize, AccumulationSwingIndexError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(AccumulationSwingIndexError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(AccumulationSwingIndexError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    first_valid_ohlc(open, high, low, close).ok_or(AccumulationSwingIndexError::AllValuesNaN)
}

fn accumulation_swing_index_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AccumulationSwingIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<AccumulationSwingIndexBatchOutput, AccumulationSwingIndexError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(open, high, low, close)?;
    let rows = combos.len();
    let cols = close.len();
    let warmups = vec![first; rows];

    let mut buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf, cols, &warmups);
    let mut guard = ManuallyDrop::new(buf);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    accumulation_swing_index_batch_inner_into(
        open, high, low, close, sweep, kernel, parallel, out,
    )?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(AccumulationSwingIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn accumulation_swing_index_batch_into_slice(
    out: &mut [f64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AccumulationSwingIndexBatchRange,
    kernel: Kernel,
) -> Result<(), AccumulationSwingIndexError> {
    accumulation_swing_index_batch_inner_into(open, high, low, close, sweep, kernel, false, out)?;
    Ok(())
}

fn accumulation_swing_index_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AccumulationSwingIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AccumulationSwingIndexParams>, AccumulationSwingIndexError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(open, high, low, close)?;
    let rows = combos.len();
    let cols = close.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| AccumulationSwingIndexError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    if out.len() != expected {
        return Err(AccumulationSwingIndexError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let daily_limits: Vec<f64> = combos
        .iter()
        .map(|combo| combo.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT))
        .collect();
    for &daily_limit in &daily_limits {
        if !daily_limit.is_finite() || daily_limit <= 0.0 {
            return Err(AccumulationSwingIndexError::InvalidDailyLimit { daily_limit });
        }
    }

    let do_row = |row: usize, dst: &mut [f64]| {
        dst[..first.min(cols)].fill(f64::NAN);
        compute_accumulation_swing_index_into(
            open,
            high,
            low,
            close,
            daily_limits[row],
            first,
            dst,
        );
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
#[pyfunction(name = "accumulation_swing_index")]
#[pyo3(signature = (open, high, low, close, daily_limit=10000.0, kernel=None))]
pub fn accumulation_swing_index_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    daily_limit: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = AccumulationSwingIndexInput::from_slices(
        open,
        high,
        low,
        close,
        AccumulationSwingIndexParams {
            daily_limit: Some(daily_limit),
        },
    );
    let out = py
        .allow_threads(|| accumulation_swing_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AccumulationSwingIndexStream")]
pub struct AccumulationSwingIndexStreamPy {
    stream: AccumulationSwingIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AccumulationSwingIndexStreamPy {
    #[new]
    #[pyo3(signature = (daily_limit=10000.0))]
    fn new(daily_limit: f64) -> PyResult<Self> {
        let stream = AccumulationSwingIndexStream::try_new(AccumulationSwingIndexParams {
            daily_limit: Some(daily_limit),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "accumulation_swing_index_batch")]
#[pyo3(signature = (open, high, low, close, daily_limit_range=(10000.0,10000.0,0.0), kernel=None))]
pub fn accumulation_swing_index_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    daily_limit_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = AccumulationSwingIndexBatchRange {
        daily_limit: daily_limit_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        accumulation_swing_index_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "daily_limits",
        combos
            .iter()
            .map(|combo| combo.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_accumulation_swing_index_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accumulation_swing_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(accumulation_swing_index_batch_py, m)?)?;
    m.add_class::<AccumulationSwingIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "accumulation_swing_index_js")]
pub fn accumulation_swing_index_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    daily_limit: f64,
) -> Result<Vec<f64>, JsValue> {
    let input = AccumulationSwingIndexInput::from_slices(
        open,
        high,
        low,
        close,
        AccumulationSwingIndexParams {
            daily_limit: Some(daily_limit),
        },
    );
    let out = accumulation_swing_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out.values)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AccumulationSwingIndexBatchConfig {
    pub daily_limit_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AccumulationSwingIndexBatchJsOutput {
    pub values: Vec<f64>,
    pub daily_limits: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_f64(name: &str, values: &[f64]) -> Result<(f64, f64, f64), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} entries must be finite numbers"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "accumulation_swing_index_batch_js")]
pub fn accumulation_swing_index_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AccumulationSwingIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = AccumulationSwingIndexBatchRange {
        daily_limit: js_vec3_to_f64("daily_limit_range", &config.daily_limit_range)?,
    };
    let out =
        accumulation_swing_index_batch_with_kernel(open, high, low, close, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let daily_limits = out
        .combos
        .iter()
        .map(|combo| combo.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT))
        .collect();
    serde_wasm_bindgen::to_value(&AccumulationSwingIndexBatchJsOutput {
        values: out.values,
        daily_limits,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn accumulation_swing_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn accumulation_swing_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn accumulation_swing_index_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    daily_limit: f64,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = AccumulationSwingIndexInput::from_slices(
            open,
            high,
            low,
            close,
            AccumulationSwingIndexParams {
                daily_limit: Some(daily_limit),
            },
        );
        accumulation_swing_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn accumulation_swing_index_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    daily_limit_start: f64,
    daily_limit_end: f64,
    daily_limit_step: f64,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to accumulation_swing_index_batch_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = AccumulationSwingIndexBatchRange {
            daily_limit: (daily_limit_start, daily_limit_end, daily_limit_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in accumulation_swing_index_batch_into")
        })?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        accumulation_swing_index_batch_into_slice(
            out,
            open,
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
pub fn accumulation_swing_index_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    daily_limit: f64,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = accumulation_swing_index_js(open, high, low, close, daily_limit)?;
    crate::write_wasm_f64_output("accumulation_swing_index_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn accumulation_swing_index_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = accumulation_swing_index_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "accumulation_swing_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manual_accumulation_swing_index(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        daily_limit: f64,
    ) -> Vec<f64> {
        let n = close.len();
        let mut out = vec![f64::NAN; n];
        let first = first_valid_ohlc(open, high, low, close).unwrap();
        compute_accumulation_swing_index_into(open, high, low, close, daily_limit, first, &mut out);
        out
    }

    fn sample_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.17).sin() * 1.4 + (i as f64) * 0.02)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + ((i as f64) * 0.23).cos() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.8 + ((i as f64) * 0.07).sin().abs())
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.7 - ((i as f64) * 0.11).cos().abs())
            .collect();
        (open, high, low, close)
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
        let (open, high, low, close) = sample_ohlc(128);
        let input = AccumulationSwingIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            AccumulationSwingIndexParams {
                daily_limit: Some(10_000.0),
            },
        );
        let out = accumulation_swing_index(&input).unwrap();
        let want = manual_accumulation_swing_index(&open, &high, &low, &close, 10_000.0);
        assert_close(&out.values, &want);
    }

    #[test]
    fn stream_matches_batch() {
        let (open, high, low, close) = sample_ohlc(96);
        let input = AccumulationSwingIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            AccumulationSwingIndexParams {
                daily_limit: Some(10_000.0),
            },
        );
        let out = accumulation_swing_index(&input).unwrap();
        let mut stream = AccumulationSwingIndexStream::try_new(AccumulationSwingIndexParams {
            daily_limit: Some(10_000.0),
        })
        .unwrap();
        let mut got = Vec::with_capacity(open.len());
        for i in 0..open.len() {
            got.push(stream.update(open[i], high[i], low[i], close[i]));
        }
        assert_close(&out.values, &got);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (open, high, low, close) = sample_ohlc(80);
        let batch = accumulation_swing_index_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &AccumulationSwingIndexBatchRange {
                daily_limit: (10_000.0, 12_000.0, 2_000.0),
            },
            Kernel::Auto,
        )
        .unwrap();
        let input = AccumulationSwingIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            AccumulationSwingIndexParams {
                daily_limit: Some(10_000.0),
            },
        );
        let single = accumulation_swing_index(&input).unwrap();
        assert_eq!(batch.rows, 2);
        assert_close(&batch.values[..80], single.values.as_slice());
    }

    #[test]
    fn into_slice_matches_single() {
        let (open, high, low, close) = sample_ohlc(72);
        let input = AccumulationSwingIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            AccumulationSwingIndexParams {
                daily_limit: Some(10_000.0),
            },
        );
        let single = accumulation_swing_index(&input).unwrap();
        let mut out = vec![0.0; close.len()];
        accumulation_swing_index_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_close(&single.values, &out);
    }

    #[test]
    fn invalid_daily_limit_is_rejected() {
        let (open, high, low, close) = sample_ohlc(32);
        let input = AccumulationSwingIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            AccumulationSwingIndexParams {
                daily_limit: Some(0.0),
            },
        );
        assert!(matches!(
            accumulation_swing_index(&input),
            Err(AccumulationSwingIndexError::InvalidDailyLimit { .. })
        ));
    }
}
