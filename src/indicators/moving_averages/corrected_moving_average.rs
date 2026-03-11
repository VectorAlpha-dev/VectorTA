#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde_wasm_bindgen;
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
use std::mem::MaybeUninit;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 35;
const TOLERANCE: f64 = 1e-5;
const MAX_GAIN_ITERS: usize = 5000;

impl<'a> AsRef<[f64]> for CorrectedMovingAverageInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CorrectedMovingAverageData::Slice(slice) => slice,
            CorrectedMovingAverageData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CorrectedMovingAverageData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CorrectedMovingAverageOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct CorrectedMovingAverageParams {
    pub period: Option<usize>,
}

impl Default for CorrectedMovingAverageParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorrectedMovingAverageInput<'a> {
    pub data: CorrectedMovingAverageData<'a>,
    pub params: CorrectedMovingAverageParams,
}

impl<'a> CorrectedMovingAverageInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: CorrectedMovingAverageParams,
    ) -> Self {
        Self {
            data: CorrectedMovingAverageData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: CorrectedMovingAverageParams) -> Self {
        Self {
            data: CorrectedMovingAverageData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", CorrectedMovingAverageParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CorrectedMovingAverageBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CorrectedMovingAverageBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CorrectedMovingAverageBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
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
    ) -> Result<CorrectedMovingAverageOutput, CorrectedMovingAverageError> {
        let input = CorrectedMovingAverageInput::from_candles(
            candles,
            "close",
            CorrectedMovingAverageParams {
                period: self.period,
            },
        );
        corrected_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<CorrectedMovingAverageOutput, CorrectedMovingAverageError> {
        let input = CorrectedMovingAverageInput::from_slice(
            data,
            CorrectedMovingAverageParams {
                period: self.period,
            },
        );
        corrected_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CorrectedMovingAverageStream, CorrectedMovingAverageError> {
        CorrectedMovingAverageStream::try_new(CorrectedMovingAverageParams {
            period: self.period,
        })
    }
}

#[derive(Debug, Error)]
pub enum CorrectedMovingAverageError {
    #[error("corrected_moving_average: Input data slice is empty.")]
    EmptyInputData,
    #[error("corrected_moving_average: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "corrected_moving_average: Invalid period: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("corrected_moving_average: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "corrected_moving_average: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("corrected_moving_average: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("corrected_moving_average: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn corrected_moving_average(
    input: &CorrectedMovingAverageInput,
) -> Result<CorrectedMovingAverageOutput, CorrectedMovingAverageError> {
    corrected_moving_average_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn longest_finite_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a CorrectedMovingAverageInput<'a>,
) -> Result<(&'a [f64], usize, usize), CorrectedMovingAverageError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CorrectedMovingAverageError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(CorrectedMovingAverageError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CorrectedMovingAverageError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let longest = longest_finite_run(data);
    if longest < period {
        return Err(CorrectedMovingAverageError::NotEnoughValidData {
            needed: period,
            valid: longest,
        });
    }

    Ok((data, period, first))
}

#[inline(always)]
fn solve_gain(v3: f64) -> f64 {
    let mut err = 1.0;
    let mut k_prev = 1.0;
    let mut k = 1.0;

    for _ in 0..MAX_GAIN_ITERS {
        if err <= TOLERANCE {
            break;
        }
        k = v3 * k_prev * (2.0 - k_prev);
        err = k_prev - k;
        k_prev = k;
    }

    k.clamp(0.0, 1.0)
}

#[inline(always)]
fn compute_corrected_moving_average(data: &[f64], period: usize, out: &mut [f64]) {
    let mut ring = vec![0.0; period];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut prev_cma: Option<f64> = None;
    let inv_period = 1.0 / period as f64;

    for (i, &x) in data.iter().enumerate() {
        if !x.is_finite() {
            out[i] = f64::NAN;
            head = 0;
            count = 0;
            sum = 0.0;
            sum_sq = 0.0;
            prev_cma = None;
            continue;
        }

        if count < period {
            ring[count] = x;
            count += 1;
            sum += x;
            sum_sq += x * x;

            if count < period {
                out[i] = f64::NAN;
                continue;
            }

            head = 0;
        } else {
            let old = ring[head];
            sum -= old;
            sum_sq -= old * old;
            ring[head] = x;
            sum += x;
            sum_sq += x * x;
            head += 1;
            if head == period {
                head = 0;
            }
        }

        let sma = sum * inv_period;
        let mut variance = sum_sq * inv_period - sma * sma;
        if variance < 0.0 && variance > -1e-12 {
            variance = 0.0;
        }

        let prev = prev_cma.unwrap_or(x);
        let diff = prev - sma;
        let v2 = diff * diff;
        let v3 = if variance == 0.0 || v2 == 0.0 {
            1.0
        } else {
            (v2 / (variance + v2)).clamp(0.0, 1.0)
        };
        let k = solve_gain(v3);
        let cma = prev + k * (sma - prev);
        out[i] = cma;
        prev_cma = Some(cma);
    }
}

#[inline]
pub fn corrected_moving_average_into_slice(
    out: &mut [f64],
    input: &CorrectedMovingAverageInput,
    _kernel: Kernel,
) -> Result<(), CorrectedMovingAverageError> {
    let (data, period, _) = prepare_input(input)?;
    if out.len() != data.len() {
        return Err(CorrectedMovingAverageError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    compute_corrected_moving_average(data, period, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn corrected_moving_average_into(
    input: &CorrectedMovingAverageInput,
    out: &mut [f64],
) -> Result<(), CorrectedMovingAverageError> {
    corrected_moving_average_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn corrected_moving_average_with_kernel(
    input: &CorrectedMovingAverageInput,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageOutput, CorrectedMovingAverageError> {
    let (data, period, first) = prepare_input(input)?;
    let mut values = alloc_with_nan_prefix(data.len(), (first + period - 1).min(data.len()));
    corrected_moving_average_into_slice(
        &mut values,
        &CorrectedMovingAverageInput::from_slice(
            data,
            CorrectedMovingAverageParams {
                period: Some(period),
            },
        ),
        kernel,
    )?;
    Ok(CorrectedMovingAverageOutput { values })
}

#[derive(Debug, Clone)]
pub struct CorrectedMovingAverageStream {
    period: usize,
    ring: Box<[f64]>,
    head: usize,
    count: usize,
    sum: f64,
    sum_sq: f64,
    prev_cma: Option<f64>,
}

impl CorrectedMovingAverageStream {
    pub fn try_new(
        params: CorrectedMovingAverageParams,
    ) -> Result<Self, CorrectedMovingAverageError> {
        let period = params.period.unwrap_or(DEFAULT_PERIOD);
        if period == 0 {
            return Err(CorrectedMovingAverageError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            ring: vec![0.0; period].into_boxed_slice(),
            head: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            prev_cma: None,
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.prev_cma = None;
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        if self.count < self.period {
            self.ring[self.count] = value;
            self.count += 1;
            self.sum += value;
            self.sum_sq += value * value;

            if self.count < self.period {
                return None;
            }

            self.head = 0;
        } else {
            let old = self.ring[self.head];
            self.sum -= old;
            self.sum_sq -= old * old;
            self.ring[self.head] = value;
            self.sum += value;
            self.sum_sq += value * value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
        }

        let inv_period = 1.0 / self.period as f64;
        let sma = self.sum * inv_period;
        let mut variance = self.sum_sq * inv_period - sma * sma;
        if variance < 0.0 && variance > -1e-12 {
            variance = 0.0;
        }
        let prev = self.prev_cma.unwrap_or(value);
        let diff = prev - sma;
        let v2 = diff * diff;
        let v3 = if variance == 0.0 || v2 == 0.0 {
            1.0
        } else {
            (v2 / (variance + v2)).clamp(0.0, 1.0)
        };
        let k = solve_gain(v3);
        let cma = prev + k * (sma - prev);
        self.prev_cma = Some(cma);
        Some(cma)
    }
}

#[derive(Clone, Debug)]
pub struct CorrectedMovingAverageBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CorrectedMovingAverageBatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CorrectedMovingAverageBatchBuilder {
    range: CorrectedMovingAverageBatchRange,
    kernel: Kernel,
}

impl Default for CorrectedMovingAverageBatchBuilder {
    fn default() -> Self {
        Self {
            range: CorrectedMovingAverageBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl CorrectedMovingAverageBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    pub fn period_static(mut self, period: usize) -> Self {
        self.range.period = (period, period, 0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
        corrected_moving_average_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct CorrectedMovingAverageBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CorrectedMovingAverageParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CorrectedMovingAverageBatchOutput {
    pub fn row_for_params(&self, params: &CorrectedMovingAverageParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.period.unwrap_or(DEFAULT_PERIOD) == params.period.unwrap_or(DEFAULT_PERIOD)
        })
    }

    pub fn values_for(&self, params: &CorrectedMovingAverageParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, CorrectedMovingAverageError> {
    let (start, end, step) = range;
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut out = Vec::new();
    let mut cur = lo;
    while cur <= hi {
        out.push(cur);
        cur = cur
            .checked_add(step)
            .ok_or(CorrectedMovingAverageError::InvalidRange { start, end, step })?;
        if cur == *out.last().unwrap() {
            break;
        }
    }
    if out.is_empty() {
        return Err(CorrectedMovingAverageError::InvalidRange { start, end, step });
    }
    if out.iter().any(|&period| period == 0) {
        return Err(CorrectedMovingAverageError::InvalidRange { start, end, step });
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_corrected_moving_average(
    range: &CorrectedMovingAverageBatchRange,
) -> Vec<CorrectedMovingAverageParams> {
    let periods = match axis_usize(range.period) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    periods
        .into_iter()
        .map(|period| CorrectedMovingAverageParams {
            period: Some(period),
        })
        .collect()
}

#[inline(always)]
pub fn corrected_moving_average_batch_slice(
    data: &[f64],
    range: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    corrected_moving_average_batch_inner(data, range, kernel, false)
}

#[inline(always)]
pub fn corrected_moving_average_batch_par_slice(
    data: &[f64],
    range: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    corrected_moving_average_batch_inner(data, range, kernel, true)
}

#[inline(always)]
fn corrected_moving_average_batch_inner(
    data: &[f64],
    range: &CorrectedMovingAverageBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    if data.is_empty() {
        return Err(CorrectedMovingAverageError::EmptyInputData);
    }
    let combos = expand_grid_corrected_moving_average(range);
    if combos.is_empty() {
        return Err(CorrectedMovingAverageError::InvalidRange {
            start: range.period.0,
            end: range.period.1,
            step: range.period.2,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let longest = longest_finite_run(data);
    let max_period = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD))
        .max()
        .unwrap_or(DEFAULT_PERIOD);
    if longest < max_period {
        return Err(CorrectedMovingAverageError::NotEnoughValidData {
            needed: max_period,
            valid: longest,
        });
    }
    let mut matrix = make_uninit_matrix(rows, cols);
    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(CorrectedMovingAverageError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.period.unwrap_or(DEFAULT_PERIOD).saturating_sub(1))
        .collect();
    init_matrix_prefixes(&mut matrix, cols, &warmups);

    let row_fn = |row: usize, dst: &mut [MaybeUninit<f64>]| {
        let period = combos[row].period.unwrap_or(DEFAULT_PERIOD);
        let out =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len()) };
        compute_corrected_moving_average(data, period, out);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            matrix
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| row_fn(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in matrix.chunks_mut(cols).enumerate() {
                row_fn(row, slice);
            }
        }
    } else {
        for (row, slice) in matrix.chunks_mut(cols).enumerate() {
            row_fn(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            matrix.as_mut_ptr() as *mut f64,
            matrix.len(),
            matrix.capacity(),
        )
    };
    std::mem::forget(matrix);

    Ok(CorrectedMovingAverageBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn corrected_moving_average_batch_inner_into(
    data: &[f64],
    range: &CorrectedMovingAverageBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CorrectedMovingAverageParams>, CorrectedMovingAverageError> {
    if data.is_empty() {
        return Err(CorrectedMovingAverageError::EmptyInputData);
    }
    let combos = expand_grid_corrected_moving_average(range);
    if combos.is_empty() {
        return Err(CorrectedMovingAverageError::InvalidRange {
            start: range.period.0,
            end: range.period.1,
            step: range.period.2,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let longest = longest_finite_run(data);
    let max_period = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD))
        .max()
        .unwrap_or(DEFAULT_PERIOD);
    if longest < max_period {
        return Err(CorrectedMovingAverageError::NotEnoughValidData {
            needed: max_period,
            valid: longest,
        });
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or(CorrectedMovingAverageError::InvalidRange {
            start: range.period.0,
            end: range.period.1,
            step: range.period.2,
        })?;
    if out.len() != expected {
        return Err(CorrectedMovingAverageError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(CorrectedMovingAverageError::AllValuesNaN)?;
    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.period.unwrap_or(DEFAULT_PERIOD).saturating_sub(1))
        .collect();
    init_matrix_prefixes(out_mu, cols, &warmups);

    let row_fn = |row: usize, dst: &mut [MaybeUninit<f64>]| {
        let period = combos[row].period.unwrap_or(DEFAULT_PERIOD);
        let out =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len()) };
        compute_corrected_moving_average(data, period, out);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| row_fn(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
                row_fn(row, slice);
            }
        }
    } else {
        for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
            row_fn(row, slice);
        }
    }

    Ok(combos)
}

pub fn corrected_moving_average_batch_with_kernel(
    data: &[f64],
    range: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(CorrectedMovingAverageError::InvalidKernelForBatch(other)),
    };
    corrected_moving_average_batch_par_slice(data, range, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "corrected_moving_average")]
#[pyo3(signature = (data, period=DEFAULT_PERIOD, kernel=None))]
pub fn corrected_moving_average_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = CorrectedMovingAverageInput::from_slice(
        slice,
        CorrectedMovingAverageParams {
            period: Some(period),
        },
    );
    let values = py
        .allow_threads(|| corrected_moving_average_with_kernel(&input, kernel).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CorrectedMovingAverageStream")]
pub struct CorrectedMovingAverageStreamPy {
    stream: CorrectedMovingAverageStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CorrectedMovingAverageStreamPy {
    #[new]
    fn new(period: Option<usize>) -> PyResult<Self> {
        let stream = CorrectedMovingAverageStream::try_new(CorrectedMovingAverageParams { period })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "corrected_moving_average_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn corrected_moving_average_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;

    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let range = CorrectedMovingAverageBatchRange {
        period: period_range,
    };

    let combos = expand_grid_corrected_moving_average(&range);
    let rows = combos.len();
    let cols = slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("size overflow: rows*cols exceeds usize"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let combos = py
        .allow_threads(|| {
            corrected_moving_average_batch_inner_into(slice, &range, batch_kernel, true, out_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_corrected_moving_average_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(corrected_moving_average_py, m)?)?;
    m.add_function(wrap_pyfunction!(corrected_moving_average_batch_py, m)?)?;
    m.add_class::<CorrectedMovingAverageStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let input = CorrectedMovingAverageInput::from_slice(
        data,
        CorrectedMovingAverageParams {
            period: Some(period),
        },
    );
    corrected_moving_average(&input)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct CorrectedMovingAverageBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct CorrectedMovingAverageBatchJsOutput {
    pub values: Vec<f64>,
    pub periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: CorrectedMovingAverageBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = CorrectedMovingAverageBatchRange {
        period: config.period_range,
    };
    let output = corrected_moving_average_batch_with_kernel(data, &range, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = CorrectedMovingAverageBatchJsOutput {
        periods: output
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD))
            .collect(),
        values: output.values,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    period: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to corrected_moving_average_into_host",
        ));
    }
    let input = CorrectedMovingAverageInput::from_slice(
        data,
        CorrectedMovingAverageParams {
            period: Some(period),
        },
    );
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, data.len()) };
    corrected_moving_average_into_slice(out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to corrected_moving_average_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = CorrectedMovingAverageInput::from_slice(
            data,
            CorrectedMovingAverageParams {
                period: Some(period),
            },
        );
        corrected_moving_average_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_batch_into(
    data: &[f64],
    out_ptr: *mut f64,
    config: JsValue,
) -> Result<usize, JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to corrected_moving_average_batch_into",
        ));
    }
    let config: CorrectedMovingAverageBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = CorrectedMovingAverageBatchRange {
        period: config.period_range,
    };
    let combos = expand_grid_corrected_moving_average(&range);
    let rows = combos.len();
    let cols = data.len();
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, rows * cols) };
    corrected_moving_average_batch_inner_into(data, &range, Kernel::Auto, false, out)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };

    fn naive_cma(data: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        let mut prev_cma: Option<f64> = None;

        for i in 0..data.len() {
            if i + 1 < period {
                continue;
            }
            let window = &data[i + 1 - period..=i];
            if window.iter().any(|x| !x.is_finite()) {
                prev_cma = None;
                continue;
            }
            let sma = window.iter().sum::<f64>() / period as f64;
            let variance = window
                .iter()
                .map(|v| {
                    let d = *v - sma;
                    d * d
                })
                .sum::<f64>()
                / period as f64;
            let prev = prev_cma.unwrap_or(data[i]);
            let v2 = (prev - sma) * (prev - sma);
            let v3 = if variance == 0.0 || v2 == 0.0 {
                1.0
            } else {
                (v2 / (variance + v2)).clamp(0.0, 1.0)
            };
            let k = solve_gain(v3);
            let cma = prev + k * (sma - prev);
            out[i] = cma;
            prev_cma = Some(cma);
        }

        out
    }

    #[test]
    fn corrected_moving_average_matches_naive() -> Result<(), Box<dyn std::error::Error>> {
        let mut data = vec![f64::NAN; 4];
        data.extend((0..128).map(|i| (i as f64 * 0.17).sin() * 5.0 + i as f64 * 0.03));
        let input = CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(35) },
        );
        let output = corrected_moving_average(&input)?;
        let expected = naive_cma(&data, 35);

        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            let both_nan = actual.is_nan() && expected.is_nan();
            assert!(both_nan || (actual - expected).abs() < 1e-10);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let data: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.09).cos() * 3.0 + i as f64 * 0.02)
            .collect();
        let input = CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(20) },
        );
        let baseline = corrected_moving_average(&input)?;
        let mut out = vec![0.0; data.len()];
        corrected_moving_average_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in baseline.values.iter().zip(out.iter()) {
            let both_nan = a.is_nan() && b.is_nan();
            assert!(both_nan || (a - b).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_stream_matches_batch() -> Result<(), Box<dyn std::error::Error>> {
        let data: Vec<f64> = (0..220)
            .map(|i| (i as f64 * 0.11).sin() * 4.0 + i as f64 * 0.01)
            .collect();
        let input = CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(17) },
        );
        let batch = corrected_moving_average(&input)?;
        let mut stream = CorrectedMovingAverageStream::try_new(CorrectedMovingAverageParams {
            period: Some(17),
        })?;
        let mut streamed = Vec::with_capacity(data.len());
        for &value in &data {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }
        for (a, b) in batch.values.iter().zip(streamed.iter()) {
            let both_nan = a.is_nan() && b.is_nan();
            assert!(both_nan || (a - b).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_batch_single_param_matches_single(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let data: Vec<f64> = (0..180)
            .map(|i| (i as f64 * 0.07).cos() * 6.0 + i as f64 * 0.015)
            .collect();
        let single = corrected_moving_average(&CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(35) },
        ))?;
        let batch = CorrectedMovingAverageBatchBuilder::new()
            .period_static(35)
            .apply_slice(&data)?;
        assert_eq!(batch.rows, 1);
        let row = batch
            .values_for(&CorrectedMovingAverageParams { period: Some(35) })
            .unwrap();
        for (a, b) in single.values.iter().zip(row.iter()) {
            let both_nan = a.is_nan() && b.is_nan();
            assert!(both_nan || (a - b).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_dispatch_matches_direct() -> Result<(), Box<dyn std::error::Error>>
    {
        let data: Vec<f64> = (0..192)
            .map(|i| (i as f64 * 0.05).sin() * 2.5 + i as f64 * 0.04)
            .collect();

        let direct = corrected_moving_average(&CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(21) },
        ))?;

        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(21),
        }];
        let combos = [IndicatorParamSet { params: &params }];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "corrected_moving_average",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })?;

        assert_eq!(dispatch.rows, 1);
        assert_eq!(dispatch.cols, data.len());
        for (a, b) in direct
            .values
            .iter()
            .zip(dispatch.values_f64.as_ref().expect("values").iter())
        {
            let both_nan = a.is_nan() && b.is_nan();
            assert!(both_nan || (a - b).abs() < 1e-12);
        }
        Ok(())
    }
}
