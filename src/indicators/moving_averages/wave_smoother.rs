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
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 20;
const DEFAULT_PHASE: f64 = 70.0;
const MIN_PHASE: f64 = 0.0;
const MAX_PHASE: f64 = 119.0;

impl<'a> AsRef<[f64]> for WaveSmootherInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WaveSmootherData::Slice(slice) => slice,
            WaveSmootherData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WaveSmootherData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WaveSmootherOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct WaveSmootherParams {
    pub period: Option<usize>,
    pub phase: Option<f64>,
}

impl Default for WaveSmootherParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            phase: Some(DEFAULT_PHASE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaveSmootherInput<'a> {
    pub data: WaveSmootherData<'a>,
    pub params: WaveSmootherParams,
}

impl<'a> WaveSmootherInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: WaveSmootherParams) -> Self {
        Self {
            data: WaveSmootherData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: WaveSmootherParams) -> Self {
        Self {
            data: WaveSmootherData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", WaveSmootherParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }

    #[inline]
    pub fn get_phase(&self) -> f64 {
        self.params.phase.unwrap_or(DEFAULT_PHASE)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WaveSmootherBuilder {
    period: Option<usize>,
    phase: Option<f64>,
    kernel: Kernel,
}

impl Default for WaveSmootherBuilder {
    fn default() -> Self {
        Self {
            period: None,
            phase: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WaveSmootherBuilder {
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
    pub fn phase(mut self, value: f64) -> Self {
        self.phase = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<WaveSmootherOutput, WaveSmootherError> {
        let input = WaveSmootherInput::from_candles(
            candles,
            "close",
            WaveSmootherParams {
                period: self.period,
                phase: self.phase,
            },
        );
        wave_smoother_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<WaveSmootherOutput, WaveSmootherError> {
        let input = WaveSmootherInput::from_slice(
            data,
            WaveSmootherParams {
                period: self.period,
                phase: self.phase,
            },
        );
        wave_smoother_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<WaveSmootherStream, WaveSmootherError> {
        WaveSmootherStream::try_new(WaveSmootherParams {
            period: self.period,
            phase: self.phase,
        })
    }
}

#[derive(Debug, Error)]
pub enum WaveSmootherError {
    #[error("wave_smoother: Input data slice is empty.")]
    EmptyInputData,

    #[error("wave_smoother: All values are NaN.")]
    AllValuesNaN,

    #[error("wave_smoother: Invalid period: {period}")]
    InvalidPeriod { period: usize },

    #[error("wave_smoother: Invalid phase: {phase}")]
    InvalidPhase { phase: f64 },

    #[error("wave_smoother: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("wave_smoother: Invalid integer range: start={start}, end={end}, step={step}")]
    InvalidRangeUsize {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("wave_smoother: Invalid float range: start={start}, end={end}, step={step}")]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },

    #[error("wave_smoother: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn validate_phase(phase: f64) -> Result<(), WaveSmootherError> {
    if !phase.is_finite() || !(MIN_PHASE..=MAX_PHASE).contains(&phase) {
        return Err(WaveSmootherError::InvalidPhase { phase });
    }
    Ok(())
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a WaveSmootherInput<'a>,
) -> Result<(&'a [f64], usize, usize, f64), WaveSmootherError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(WaveSmootherError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(WaveSmootherError::AllValuesNaN)?;

    let period = input.get_period();
    if period == 0 {
        return Err(WaveSmootherError::InvalidPeriod { period });
    }

    let phase = input.get_phase();
    validate_phase(phase)?;

    Ok((data, first, period, phase))
}

#[inline(always)]
fn build_normalized_weights(period: usize, phase: f64) -> Option<Box<[f64]>> {
    let phase_rad = phase.to_radians();
    let window = period + 1;
    let period_f = period as f64;
    let mut weights = vec![0.0; window];
    let mut sum = 0.0;

    for (i, weight) in weights.iter_mut().enumerate() {
        let idx = i as f64;
        let value = (idx * PI / period_f + phase_rad).sin() * (idx * PI / (2.0 * period_f)).cos();
        *weight = value;
        sum += value;
    }

    if !sum.is_finite() || sum.abs() <= f64::EPSILON {
        return None;
    }

    let inv = 1.0 / sum;
    for weight in &mut weights {
        *weight *= inv;
    }
    Some(weights.into_boxed_slice())
}

#[inline(always)]
fn smooth_value(value: f64, prev_raw: f64) -> f64 {
    if value.is_finite() {
        0.5 * (value + if prev_raw.is_finite() { prev_raw } else { 0.0 })
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn compute_wave_smoother(
    data: &[f64],
    first: usize,
    period: usize,
    weights: Option<&[f64]>,
    out: &mut [f64],
) {
    let window = period + 1;
    let mut ring = vec![f64::NAN; window].into_boxed_slice();
    let mut head = 0usize;
    let mut count = 0usize;
    let mut prev_raw = f64::NAN;
    let mut first_nz = None;

    for value in &mut out[..first] {
        *value = f64::NAN;
    }

    for (idx, &value) in data.iter().enumerate().skip(first) {
        let smooth = smooth_value(value, prev_raw);
        prev_raw = value;

        if first_nz.is_none() && smooth.is_finite() {
            first_nz = Some(smooth);
        }

        ring[head] = smooth;
        head += 1;
        if head == window {
            head = 0;
        }
        if count < window {
            count += 1;
        }

        let Some(first_fill) = first_nz else {
            out[idx] = f64::NAN;
            continue;
        };

        let Some(weights) = weights else {
            out[idx] = f64::NAN;
            continue;
        };

        let mut acc = 0.0;
        let latest = if head == 0 { window - 1 } else { head - 1 };
        for lag in 0..window {
            let hist = if lag < count {
                let pos = (latest + window - lag) % window;
                ring[pos]
            } else {
                first_fill
            };
            acc += if hist.is_finite() { hist } else { first_fill } * weights[lag];
        }
        out[idx] = acc;
    }
}

#[inline]
pub fn wave_smoother(input: &WaveSmootherInput) -> Result<WaveSmootherOutput, WaveSmootherError> {
    wave_smoother_with_kernel(input, Kernel::Auto)
}

pub fn wave_smoother_with_kernel(
    input: &WaveSmootherInput,
    _kernel: Kernel,
) -> Result<WaveSmootherOutput, WaveSmootherError> {
    let (data, first, period, phase) = prepare_input(input)?;
    let mut values = alloc_with_nan_prefix(data.len(), first);
    let weights = build_normalized_weights(period, phase);
    compute_wave_smoother(data, first, period, weights.as_deref(), &mut values);
    Ok(WaveSmootherOutput { values })
}

#[inline]
pub fn wave_smoother_into_slice(
    out: &mut [f64],
    input: &WaveSmootherInput,
    _kernel: Kernel,
) -> Result<(), WaveSmootherError> {
    let (data, first, period, phase) = prepare_input(input)?;
    if out.len() != data.len() {
        return Err(WaveSmootherError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    let weights = build_normalized_weights(period, phase);
    out.fill(f64::NAN);
    compute_wave_smoother(data, first, period, weights.as_deref(), out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn wave_smoother_into(
    input: &WaveSmootherInput,
    out: &mut [f64],
) -> Result<(), WaveSmootherError> {
    wave_smoother_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct WaveSmootherStream {
    weights: Option<Box<[f64]>>,
    ring: Box<[f64]>,
    window: usize,
    head: usize,
    count: usize,
    prev_raw: f64,
    first_nz: Option<f64>,
}

impl WaveSmootherStream {
    pub fn try_new(params: WaveSmootherParams) -> Result<Self, WaveSmootherError> {
        let period = params.period.unwrap_or(DEFAULT_PERIOD);
        if period == 0 {
            return Err(WaveSmootherError::InvalidPeriod { period });
        }
        let phase = params.phase.unwrap_or(DEFAULT_PHASE);
        validate_phase(phase)?;
        let window = period + 1;

        Ok(Self {
            weights: build_normalized_weights(period, phase),
            ring: vec![f64::NAN; window].into_boxed_slice(),
            window,
            head: 0,
            count: 0,
            prev_raw: f64::NAN,
            first_nz: None,
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.ring.fill(f64::NAN);
        self.head = 0;
        self.count = 0;
        self.prev_raw = f64::NAN;
        self.first_nz = None;
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let smooth = smooth_value(value, self.prev_raw);
        self.prev_raw = value;

        if self.first_nz.is_none() && smooth.is_finite() {
            self.first_nz = Some(smooth);
        }

        self.ring[self.head] = smooth;
        self.head += 1;
        if self.head == self.window {
            self.head = 0;
        }
        if self.count < self.window {
            self.count += 1;
        }

        let first_fill = self.first_nz?;
        let weights = self.weights.as_deref()?;
        let latest = if self.head == 0 {
            self.window - 1
        } else {
            self.head - 1
        };

        let mut acc = 0.0;
        for lag in 0..self.window {
            let hist = if lag < self.count {
                let pos = (latest + self.window - lag) % self.window;
                self.ring[pos]
            } else {
                first_fill
            };
            acc += if hist.is_finite() { hist } else { first_fill } * weights[lag];
        }
        Some(acc)
    }
}

#[derive(Clone, Debug)]
pub struct WaveSmootherBatchRange {
    pub period: (usize, usize, usize),
    pub phase: (f64, f64, f64),
}

impl Default for WaveSmootherBatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
            phase: (DEFAULT_PHASE, DEFAULT_PHASE, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WaveSmootherBatchBuilder {
    range: WaveSmootherBatchRange,
    kernel: Kernel,
}

impl Default for WaveSmootherBatchBuilder {
    fn default() -> Self {
        Self {
            range: WaveSmootherBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl WaveSmootherBatchBuilder {
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

    pub fn phase_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.phase = (start, end, step);
        self
    }

    pub fn period_static(mut self, period: usize) -> Self {
        self.range.period = (period, period, 0);
        self
    }

    pub fn phase_static(mut self, phase: f64) -> Self {
        self.range.phase = (phase, phase, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<WaveSmootherBatchOutput, WaveSmootherError> {
        wave_smoother_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<WaveSmootherBatchOutput, WaveSmootherError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct WaveSmootherBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WaveSmootherParams>,
    pub rows: usize,
    pub cols: usize,
}

impl WaveSmootherBatchOutput {
    pub fn row_for_params(&self, params: &WaveSmootherParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.period.unwrap_or(DEFAULT_PERIOD) == params.period.unwrap_or(DEFAULT_PERIOD)
                && combo.phase.unwrap_or(DEFAULT_PHASE).to_bits()
                    == params.phase.unwrap_or(DEFAULT_PHASE).to_bits()
        })
    }

    pub fn values_for(&self, params: &WaveSmootherParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, WaveSmootherError> {
    let (start, end, step) = range;
    if step == 0 || start == end {
        if start == 0 {
            return Err(WaveSmootherError::InvalidRangeUsize { start, end, step });
        }
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
        if cur == 0 {
            return Err(WaveSmootherError::InvalidRangeUsize { start, end, step });
        }
        out.push(cur);
        cur = cur
            .checked_add(step)
            .ok_or(WaveSmootherError::InvalidRangeUsize { start, end, step })?;
        if cur == *out.last().unwrap() {
            break;
        }
    }
    if out.is_empty() {
        return Err(WaveSmootherError::InvalidRangeUsize { start, end, step });
    }
    Ok(out)
}

#[inline(always)]
fn axis_f64(range: (f64, f64, f64)) -> Result<Vec<f64>, WaveSmootherError> {
    let (start, end, step) = range;
    validate_phase(start)?;
    validate_phase(end)?;
    if !step.is_finite() {
        return Err(WaveSmootherError::InvalidRangeF64 { start, end, step });
    }
    if step == 0.0 || start.to_bits() == end.to_bits() {
        return Ok(vec![start]);
    }
    let step = step.abs();
    if step <= f64::EPSILON {
        return Err(WaveSmootherError::InvalidRangeF64 { start, end, step });
    }
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut out = Vec::new();
    let mut cur = lo;
    let limit = hi + step * 1e-9;
    while cur <= limit {
        validate_phase(cur)?;
        out.push(cur);
        cur += step;
        if out.len() > 1_000_000 {
            return Err(WaveSmootherError::InvalidRangeF64 { start, end, step });
        }
    }
    if out.is_empty() {
        return Err(WaveSmootherError::InvalidRangeF64 { start, end, step });
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_wave_smoother(range: &WaveSmootherBatchRange) -> Vec<WaveSmootherParams> {
    let periods = match axis_usize(range.period) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let phases = match axis_f64(range.phase) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut combos = Vec::with_capacity(periods.len() * phases.len());
    for period in periods {
        for &phase in &phases {
            combos.push(WaveSmootherParams {
                period: Some(period),
                phase: Some(phase),
            });
        }
    }
    combos
}

#[inline(always)]
pub fn wave_smoother_batch_slice(
    data: &[f64],
    range: &WaveSmootherBatchRange,
    kernel: Kernel,
) -> Result<WaveSmootherBatchOutput, WaveSmootherError> {
    wave_smoother_batch_inner(data, range, kernel, false)
}

#[inline(always)]
pub fn wave_smoother_batch_par_slice(
    data: &[f64],
    range: &WaveSmootherBatchRange,
    kernel: Kernel,
) -> Result<WaveSmootherBatchOutput, WaveSmootherError> {
    wave_smoother_batch_inner(data, range, kernel, true)
}

#[inline(always)]
fn wave_smoother_batch_inner(
    data: &[f64],
    range: &WaveSmootherBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<WaveSmootherBatchOutput, WaveSmootherError> {
    if data.is_empty() {
        return Err(WaveSmootherError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(WaveSmootherError::AllValuesNaN)?;
    let combos = expand_grid_wave_smoother(range);
    if combos.is_empty() {
        return Err(WaveSmootherError::InvalidRangeUsize {
            start: range.period.0,
            end: range.period.1,
            step: range.period.2,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut matrix = make_uninit_matrix(rows, cols);
    let warmups = vec![first; rows];
    init_matrix_prefixes(&mut matrix, cols, &warmups);

    let row_fn = |row: usize, dst: &mut [MaybeUninit<f64>]| {
        let params = combos[row].clone();
        let out =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len()) };
        let _ = wave_smoother_into_slice(
            out,
            &WaveSmootherInput::from_slice(data, params),
            Kernel::Scalar,
        );
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

    Ok(WaveSmootherBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn wave_smoother_batch_inner_into(
    data: &[f64],
    range: &WaveSmootherBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<WaveSmootherParams>, WaveSmootherError> {
    if data.is_empty() {
        return Err(WaveSmootherError::EmptyInputData);
    }
    let combos = expand_grid_wave_smoother(range);
    if combos.is_empty() {
        return Err(WaveSmootherError::InvalidRangeUsize {
            start: range.period.0,
            end: range.period.1,
            step: range.period.2,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    if out.len() != rows * cols {
        return Err(WaveSmootherError::OutputLengthMismatch {
            expected: rows * cols,
            got: out.len(),
        });
    }

    let row_fn = |row: usize, dst: &mut [f64]| {
        let _ = wave_smoother_into_slice(
            dst,
            &WaveSmootherInput::from_slice(data, combos[row].clone()),
            Kernel::Scalar,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, dst)| row_fn(row, dst));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                row_fn(row, dst);
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            row_fn(row, dst);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn wave_smoother_batch_with_kernel(
    data: &[f64],
    range: &WaveSmootherBatchRange,
    kernel: Kernel,
) -> Result<WaveSmootherBatchOutput, WaveSmootherError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(WaveSmootherError::InvalidKernelForBatch(other)),
    };
    wave_smoother_batch_par_slice(data, range, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "wave_smoother")]
#[pyo3(signature = (data, period=DEFAULT_PERIOD, phase=DEFAULT_PHASE, kernel=None))]
pub fn wave_smoother_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    phase: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = WaveSmootherInput::from_slice(
        slice,
        WaveSmootherParams {
            period: Some(period),
            phase: Some(phase),
        },
    );
    let values = py
        .allow_threads(|| wave_smoother_with_kernel(&input, kernel).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "WaveSmootherStream")]
pub struct WaveSmootherStreamPy {
    stream: WaveSmootherStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WaveSmootherStreamPy {
    #[new]
    fn new(period: Option<usize>, phase: Option<f64>) -> PyResult<Self> {
        let stream = WaveSmootherStream::try_new(WaveSmootherParams { period, phase })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "wave_smoother_batch")]
#[pyo3(signature = (data, period_range, phase_range=(DEFAULT_PHASE, DEFAULT_PHASE, 0.0), kernel=None))]
pub fn wave_smoother_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    phase_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;

    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let range = WaveSmootherBatchRange {
        period: period_range,
        phase: phase_range,
    };

    let combos = expand_grid_wave_smoother(&range);
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
            wave_smoother_batch_inner_into(slice, &range, batch_kernel, true, out_slice)
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
    dict.set_item(
        "phases",
        combos
            .iter()
            .map(|combo| combo.phase.unwrap_or(DEFAULT_PHASE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_wave_smoother_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wave_smoother_py, m)?)?;
    m.add_function(wrap_pyfunction!(wave_smoother_batch_py, m)?)?;
    m.add_class::<WaveSmootherStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_js(data: &[f64], period: usize, phase: f64) -> Result<Vec<f64>, JsValue> {
    let input = WaveSmootherInput::from_slice(
        data,
        WaveSmootherParams {
            period: Some(period),
            phase: Some(phase),
        },
    );
    wave_smoother(&input)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct WaveSmootherBatchConfig {
    pub period_range: (usize, usize, usize),
    pub phase_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct WaveSmootherBatchJsOutput {
    pub values: Vec<f64>,
    pub periods: Vec<usize>,
    pub phases: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: WaveSmootherBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = WaveSmootherBatchRange {
        period: config.period_range,
        phase: config.phase_range,
    };
    let output = wave_smoother_batch_with_kernel(data, &range, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = WaveSmootherBatchJsOutput {
        periods: output
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD))
            .collect(),
        phases: output
            .combos
            .iter()
            .map(|combo| combo.phase.unwrap_or(DEFAULT_PHASE))
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
pub fn wave_smoother_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    period: usize,
    phase: f64,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to wave_smoother_into_host",
        ));
    }
    let input = WaveSmootherInput::from_slice(
        data,
        WaveSmootherParams {
            period: Some(period),
            phase: Some(phase),
        },
    );
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, data.len()) };
    wave_smoother_into_slice(out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    phase: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to wave_smoother_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = WaveSmootherInput::from_slice(
            data,
            WaveSmootherParams {
                period: Some(period),
                phase: Some(phase),
            },
        );
        wave_smoother_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_batch_into(
    data: &[f64],
    out_ptr: *mut f64,
    config: JsValue,
) -> Result<usize, JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to wave_smoother_batch_into",
        ));
    }
    let config: WaveSmootherBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = WaveSmootherBatchRange {
        period: config.period_range,
        phase: config.phase_range,
    };
    let combos = expand_grid_wave_smoother(&range);
    let rows = combos.len();
    let cols = data.len();
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, rows * cols) };
    wave_smoother_batch_inner_into(data, &range, Kernel::Auto, false, out)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_output_into_js(
    data: &[f64],
    period: usize,
    phase: f64,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = wave_smoother_js(data, period, phase)?;
    crate::write_wasm_f64_output("wave_smoother_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wave_smoother_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = wave_smoother_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs("wave_smoother_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_wave_smoother(data: &[f64], period: usize, phase: f64) -> Vec<f64> {
        let weights = build_normalized_weights(period, phase);
        let mut src = vec![f64::NAN; data.len()];
        let mut prev_raw = f64::NAN;
        let mut first_nz = None;
        for (i, &value) in data.iter().enumerate() {
            let smooth = smooth_value(value, prev_raw);
            prev_raw = value;
            src[i] = smooth;
            if first_nz.is_none() && smooth.is_finite() {
                first_nz = Some(smooth);
            }
        }

        let mut out = vec![f64::NAN; data.len()];
        let Some(first_fill) = first_nz else {
            return out;
        };
        let Some(weights) = weights else {
            return out;
        };
        let first = data.iter().position(|x| x.is_finite()).unwrap();
        for i in first..data.len() {
            let mut acc = 0.0;
            for lag in 0..=period {
                let hist = if i >= lag { src[i - lag] } else { first_fill };
                acc += if hist.is_finite() { hist } else { first_fill } * weights[lag];
            }
            out[i] = acc;
        }
        out
    }

    #[test]
    fn wave_smoother_matches_naive_small_sample() {
        let data = [f64::NAN, 10.0, 12.0, 13.0, f64::NAN, 15.0, 16.0, 18.0];
        let input = WaveSmootherInput::from_slice(
            &data,
            WaveSmootherParams {
                period: Some(4),
                phase: Some(70.0),
            },
        );

        let direct = wave_smoother(&input).unwrap();
        let expected = naive_wave_smoother(&data, 4, 70.0);
        assert_eq!(direct.values.len(), expected.len());
        for (got, want) in direct.values.iter().zip(expected.iter()) {
            if got.is_nan() && want.is_nan() {
                continue;
            }
            assert!((got - want).abs() <= 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    fn wave_smoother_period_larger_than_length_is_supported() {
        let data = [10.0, 12.0, 14.0];
        let input = WaveSmootherInput::from_slice(
            &data,
            WaveSmootherParams {
                period: Some(20),
                phase: Some(70.0),
            },
        );

        let out = wave_smoother(&input).unwrap();
        assert_eq!(out.values.len(), data.len());
        assert!(out.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn wave_smoother_into_matches_api() {
        let data = [11.0, 12.0, 13.0, 14.0, 15.0];
        let input = WaveSmootherInput::from_slice(
            &data,
            WaveSmootherParams {
                period: Some(3),
                phase: Some(70.0),
            },
        );
        let direct = wave_smoother(&input).unwrap();
        let mut out = vec![f64::NAN; data.len()];
        wave_smoother_into_slice(&mut out, &input, Kernel::Auto).unwrap();

        for (got, want) in out.iter().zip(direct.values.iter()) {
            if got.is_nan() && want.is_nan() {
                continue;
            }
            assert!((got - want).abs() <= 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    fn wave_smoother_stream_matches_batch_with_nan_input() {
        let data = [f64::NAN, 10.0, 12.0, 14.0, f64::NAN, 16.0, 18.0, 20.0];
        let input = WaveSmootherInput::from_slice(
            &data,
            WaveSmootherParams {
                period: Some(4),
                phase: Some(70.0),
            },
        );
        let batch = wave_smoother(&input).unwrap();
        let mut stream = WaveSmootherStream::try_new(WaveSmootherParams {
            period: Some(4),
            phase: Some(70.0),
        })
        .unwrap();

        for (idx, &value) in data.iter().enumerate() {
            let got = stream.update(value).unwrap_or(f64::NAN);
            let want = batch.values[idx];
            if got.is_nan() && want.is_nan() {
                continue;
            }
            assert!(
                (got - want).abs() <= 1e-12,
                "idx={idx} got={got} want={want}"
            );
        }
    }

    #[test]
    fn wave_smoother_batch_single_param_matches_single() {
        let data = [10.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0];
        let single = wave_smoother(&WaveSmootherInput::from_slice(
            &data,
            WaveSmootherParams {
                period: Some(5),
                phase: Some(70.0),
            },
        ))
        .unwrap();

        let batch = wave_smoother_batch_with_kernel(
            &data,
            &WaveSmootherBatchRange {
                period: (5, 5, 0),
                phase: (70.0, 70.0, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for (got, want) in batch.values.iter().zip(single.values.iter()) {
            if got.is_nan() && want.is_nan() {
                continue;
            }
            assert!((got - want).abs() <= 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    fn wave_smoother_invalid_phase_rejected() {
        let data = [10.0, 11.0, 12.0];
        let input = WaveSmootherInput::from_slice(
            &data,
            WaveSmootherParams {
                period: Some(3),
                phase: Some(120.0),
            },
        );
        let err = wave_smoother(&input).unwrap_err();
        assert!(matches!(err, WaveSmootherError::InvalidPhase { .. }));
    }
}
