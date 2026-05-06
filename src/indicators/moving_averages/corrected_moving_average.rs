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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct CorrectedMovingAverageParams {
    pub period: Option<usize>,
}

impl Default for CorrectedMovingAverageParams {
    fn default() -> Self {
        Self { period: Some(35) }
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
        self.params.period.unwrap_or(35)
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
        corrected_moving_average_with_kernel(
            &CorrectedMovingAverageInput::from_candles(
                candles,
                "close",
                CorrectedMovingAverageParams {
                    period: self.period,
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<CorrectedMovingAverageOutput, CorrectedMovingAverageError> {
        corrected_moving_average_with_kernel(
            &CorrectedMovingAverageInput::from_slice(
                data,
                CorrectedMovingAverageParams {
                    period: self.period,
                },
            ),
            self.kernel,
        )
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
    #[error(
        "corrected_moving_average: Invalid range: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("corrected_moving_average: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct RollingStats {
    period: usize,
    window: VecDeque<f64>,
    sum: f64,
    sumsq: f64,
}

impl RollingStats {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            window: VecDeque::with_capacity(period.max(1)),
            sum: 0.0,
            sumsq: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sumsq = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.window.push_back(value);
        self.sum += value;
        self.sumsq += value * value;
        if self.window.len() > self.period {
            if let Some(oldest) = self.window.pop_front() {
                self.sum -= oldest;
                self.sumsq -= oldest * oldest;
            }
        }
        if self.window.len() < self.period {
            return None;
        }
        let denom = self.period as f64;
        let mean = self.sum / denom;
        let variance = (self.sumsq / denom - mean * mean).max(0.0);
        Some((mean, variance))
    }
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &v in data {
        if v.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_input(data: &[f64], period: usize) -> Result<(), CorrectedMovingAverageError> {
    if data.is_empty() {
        return Err(CorrectedMovingAverageError::EmptyInputData);
    }
    if !data.iter().any(|v| v.is_finite()) {
        return Err(CorrectedMovingAverageError::AllValuesNaN);
    }
    if period == 0 || period > data.len() {
        return Err(CorrectedMovingAverageError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let valid = longest_valid_run(data);
    if valid < period {
        return Err(CorrectedMovingAverageError::NotEnoughValidData {
            needed: period,
            valid,
        });
    }
    Ok(())
}

#[inline(always)]
fn gain_factor(v3: f64) -> f64 {
    if !v3.is_finite() {
        return 0.0;
    }
    if (v3 - 1.0).abs() <= f64::EPSILON {
        return 1.0;
    }
    let mut k_prev = 1.0f64;
    let mut k = 1.0f64;
    for _ in 0..64 {
        k = v3 * k_prev * (2.0 - k_prev);
        let err = k_prev - k;
        k_prev = k;
        if err <= 1e-5 {
            break;
        }
    }
    if k.is_finite() {
        k.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct CorrectedMovingAverageStream {
    period: usize,
    stats: RollingStats,
    prev_cma: Option<f64>,
}

impl CorrectedMovingAverageStream {
    #[inline(always)]
    pub fn try_new(
        params: CorrectedMovingAverageParams,
    ) -> Result<Self, CorrectedMovingAverageError> {
        let period = params.period.unwrap_or(35);
        if period == 0 {
            return Err(CorrectedMovingAverageError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            stats: RollingStats::new(period),
            prev_cma: None,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.stats.reset();
        self.prev_cma = None;
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let (sma, variance) = match self.stats.update(value) {
            Some(v) => v,
            None => {
                if !value.is_finite() {
                    self.prev_cma = None;
                }
                return None;
            }
        };

        let cma = match self.prev_cma {
            None => sma,
            Some(prev) => {
                let v2 = (prev - sma).powi(2);
                let v3 = if variance <= f64::EPSILON || v2 <= f64::EPSILON {
                    1.0
                } else {
                    v2 / (variance + v2)
                };
                let k = gain_factor(v3);
                prev + k * (sma - prev)
            }
        };
        self.prev_cma = Some(cma);
        Some(cma)
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period.saturating_sub(1)
    }
}

#[inline(always)]
pub fn corrected_moving_average(
    input: &CorrectedMovingAverageInput,
) -> Result<CorrectedMovingAverageOutput, CorrectedMovingAverageError> {
    corrected_moving_average_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn corrected_moving_average_with_kernel(
    input: &CorrectedMovingAverageInput,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageOutput, CorrectedMovingAverageError> {
    let data = input.as_ref();
    let period = input.get_period();
    validate_input(data, period)?;
    let mut values = alloc_with_nan_prefix(data.len(), 0);
    corrected_moving_average_into_slice(&mut values, input, kernel)?;
    Ok(CorrectedMovingAverageOutput { values })
}

pub fn corrected_moving_average_into_slice(
    dst: &mut [f64],
    input: &CorrectedMovingAverageInput,
    kernel: Kernel,
) -> Result<(), CorrectedMovingAverageError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(CorrectedMovingAverageError::InvalidKernelForBatch(other)),
    }

    let data = input.as_ref();
    let period = input.get_period();
    validate_input(data, period)?;
    if dst.len() != data.len() {
        return Err(CorrectedMovingAverageError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::from_bits(0x7ff8_0000_0000_0000));
    let mut stream = CorrectedMovingAverageStream::try_new(CorrectedMovingAverageParams {
        period: Some(period),
    })?;
    for (i, &value) in data.iter().enumerate() {
        if let Some(out) = stream.update(value) {
            dst[i] = out;
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn corrected_moving_average_into(
    input: &CorrectedMovingAverageInput,
    out: &mut [f64],
) -> Result<(), CorrectedMovingAverageError> {
    corrected_moving_average_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct CorrectedMovingAverageBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CorrectedMovingAverageBatchRange {
    fn default() -> Self {
        let period = CorrectedMovingAverageParams::default().period.unwrap_or(35);
        Self {
            period: (period, period, 0),
        }
    }
}

#[derive(Copy, Clone, Debug)]
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

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
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
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(35) == params.period.unwrap_or(35))
    }

    pub fn values_for(&self, params: &CorrectedMovingAverageParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

pub fn expand_grid_corrected_moving_average(
    sweep: &CorrectedMovingAverageBatchRange,
) -> Result<Vec<CorrectedMovingAverageParams>, CorrectedMovingAverageError> {
    let (start, end, step) = sweep.period;
    let periods = if step == 0 || start == end {
        vec![start]
    } else if start < end && step > 0 {
        let mut out = Vec::new();
        let mut cur = start;
        while cur <= end {
            out.push(cur);
            cur = cur
                .checked_add(step)
                .ok_or(CorrectedMovingAverageError::InvalidRange { start, end, step })?;
        }
        out
    } else {
        return Err(CorrectedMovingAverageError::InvalidRange { start, end, step });
    };
    Ok(periods
        .into_iter()
        .map(|period| CorrectedMovingAverageParams {
            period: Some(period),
        })
        .collect())
}

pub fn corrected_moving_average_batch_with_kernel(
    data: &[f64],
    sweep: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    corrected_moving_average_batch_inner(data, sweep, kernel, false)
}

pub fn corrected_moving_average_batch_slice(
    data: &[f64],
    sweep: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    corrected_moving_average_batch_inner(data, sweep, kernel, false)
}

pub fn corrected_moving_average_batch_par_slice(
    data: &[f64],
    sweep: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    corrected_moving_average_batch_inner(data, sweep, kernel, true)
}

fn corrected_moving_average_batch_inner(
    data: &[f64],
    sweep: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<CorrectedMovingAverageBatchOutput, CorrectedMovingAverageError> {
    let combos = expand_grid_corrected_moving_average(sweep)?;
    let max_period = combos
        .iter()
        .map(|c| c.period.unwrap_or(35))
        .max()
        .unwrap_or(0);
    validate_input(data, max_period)?;

    let rows = combos.len();
    let cols = data.len();
    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.period.unwrap_or(35).saturating_sub(1))
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

    corrected_moving_average_batch_inner_into(data, sweep, kernel, parallel, &mut values)?;

    Ok(CorrectedMovingAverageBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn corrected_moving_average_batch_inner_into(
    data: &[f64],
    sweep: &CorrectedMovingAverageBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CorrectedMovingAverageParams>, CorrectedMovingAverageError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(CorrectedMovingAverageError::InvalidKernelForBatch(other)),
    }
    let combos = expand_grid_corrected_moving_average(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if out.len() != rows * cols {
        return Err(CorrectedMovingAverageError::OutputLengthMismatch {
            expected: rows * cols,
            got: out.len(),
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .zip(combos.par_iter())
                .try_for_each(|(dst, params)| {
                    let input = CorrectedMovingAverageInput::from_slice(data, *params);
                    corrected_moving_average_into_slice(dst, &input, kernel)
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, params) in combos.iter().enumerate() {
                let input = CorrectedMovingAverageInput::from_slice(data, *params);
                corrected_moving_average_into_slice(
                    &mut out[row * cols..(row + 1) * cols],
                    &input,
                    kernel,
                )?;
            }
        }
    } else {
        for (row, params) in combos.iter().enumerate() {
            let input = CorrectedMovingAverageInput::from_slice(data, *params);
            corrected_moving_average_into_slice(
                &mut out[row * cols..(row + 1) * cols],
                &input,
                kernel,
            )?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "corrected_moving_average")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn corrected_moving_average_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = CorrectedMovingAverageInput::from_slice(
        slice,
        CorrectedMovingAverageParams {
            period: Some(period),
        },
    );
    let values = py
        .allow_threads(|| corrected_moving_average_with_kernel(&input, kern).map(|o| o.values))
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
    fn new(period: usize) -> PyResult<Self> {
        let stream = CorrectedMovingAverageStream::try_new(CorrectedMovingAverageParams {
            period: Some(period),
        })
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
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let batch_kernel = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let output = py
        .allow_threads(|| {
            corrected_moving_average_batch_with_kernel(
                slice,
                &CorrectedMovingAverageBatchRange {
                    period: period_range,
                },
                batch_kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    let values = output
        .values
        .into_pyarray(py)
        .reshape((output.rows, output.cols))?;
    dict.set_item("values", values)?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|p| p.period.unwrap_or(35) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
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
    let mut output = alloc_with_nan_prefix(data.len(), 0);
    corrected_moving_average_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
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
    pub combos: Vec<CorrectedMovingAverageParams>,
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
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let output = corrected_moving_average_batch_with_kernel(
        data,
        &CorrectedMovingAverageBatchRange {
            period: config.period_range,
        },
        Kernel::ScalarBatch,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&CorrectedMovingAverageBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
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
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
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
        let input = CorrectedMovingAverageInput::from_slice(
            data,
            CorrectedMovingAverageParams {
                period: Some(period),
            },
        );
        if in_ptr == out_ptr {
            let mut tmp = alloc_with_nan_prefix(len, 0);
            corrected_moving_average_into_slice(&mut tmp, &input, Kernel::Scalar)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            corrected_moving_average_into_slice(
                std::slice::from_raw_parts_mut(out_ptr, len),
                &input,
                Kernel::Scalar,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to corrected_moving_average_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = CorrectedMovingAverageBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid_corrected_moving_average(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        corrected_moving_average_batch_inner_into(
            data,
            &sweep,
            Kernel::ScalarBatch,
            false,
            std::slice::from_raw_parts_mut(out_ptr, rows * len),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_output_into_js(
    data: &[f64],
    period: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = corrected_moving_average_js(data, period)?;
    crate::write_wasm_f64_output("corrected_moving_average_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn corrected_moving_average_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = corrected_moving_average_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "corrected_moving_average_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::moving_averages::ma::{ma, ma_with_kernel, MaData};

    #[test]
    fn corrected_moving_average_constant_series_stays_constant() -> Result<(), Box<dyn Error>> {
        let data = vec![100.0; 64];
        let input = CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(5) },
        );
        let out = corrected_moving_average(&input)?.values;
        for value in out.iter().skip(4) {
            assert!((*value - 100.0).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_first_valid_equals_sma() -> Result<(), Box<dyn Error>> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(3) },
        );
        let out = corrected_moving_average(&input)?.values;
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!((out[2] - 2.0).abs() <= 1e-12);
        Ok(())
    }

    #[test]
    fn corrected_moving_average_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = (0..256)
            .map(|i| 100.0 + i as f64 * 0.25 + (i as f64 * 0.1).sin())
            .collect::<Vec<_>>();
        let input = CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(17) },
        );
        let baseline = corrected_moving_average(&input)?.values;
        let mut out = vec![0.0; data.len()];
        corrected_moving_average_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in baseline.iter().zip(out.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = (0..128)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0 + i as f64 * 0.1)
            .collect::<Vec<_>>();
        let batch = corrected_moving_average(&CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(11) },
        ))?
        .values;

        let mut stream = CorrectedMovingAverageStream::try_new(CorrectedMovingAverageParams {
            period: Some(11),
        })?;
        let mut stream_values = Vec::with_capacity(data.len());
        for value in data {
            stream_values.push(stream.update(value).unwrap_or(f64::NAN));
        }
        for (a, b) in batch.iter().zip(stream_values.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let data = (0..128)
            .map(|i| 100.0 + i as f64 * 0.2 + (i as f64 * 0.05).cos())
            .collect::<Vec<_>>();
        let batch = corrected_moving_average_batch_with_kernel(
            &data,
            &CorrectedMovingAverageBatchRange { period: (9, 9, 0) },
            Kernel::Auto,
        )?;
        let single = corrected_moving_average(&CorrectedMovingAverageInput::from_slice(
            &data,
            CorrectedMovingAverageParams { period: Some(9) },
        ))?
        .values;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        let row = batch
            .values_for(&CorrectedMovingAverageParams { period: Some(9) })
            .unwrap();
        for (a, b) in row.iter().zip(single.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn corrected_moving_average_ma_dispatch_matches_direct() -> Result<(), Box<dyn Error>> {
        let data = (0..96)
            .map(|i| 100.0 + (i as f64 * 0.2).sin())
            .collect::<Vec<_>>();
        let direct = corrected_moving_average_with_kernel(
            &CorrectedMovingAverageInput::from_slice(
                &data,
                CorrectedMovingAverageParams { period: Some(7) },
            ),
            Kernel::Scalar,
        )?
        .values;
        let via_ma = ma_with_kernel(
            "corrected_moving_average",
            MaData::Slice(&data),
            7,
            Kernel::Scalar,
        )?;
        let via_alias = ma("cma", MaData::Slice(&data), 7)?;
        assert_eq!(direct.len(), via_ma.len());
        assert_eq!(direct.len(), via_alias.len());
        Ok(())
    }
}
