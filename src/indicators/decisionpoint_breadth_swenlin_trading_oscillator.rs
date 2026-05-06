#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use std::mem::ManuallyDrop;
use thiserror::Error;

const EMA_LENGTH: usize = 4;
const SMA_LENGTH: usize = 5;
const MULTIPLIER: f64 = 1000.0;
const EMA_ALPHA: f64 = 2.0 / (EMA_LENGTH as f64 + 1.0);
const WARMUP: usize = SMA_LENGTH - 1;
const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone)]
pub enum DecisionPointBreadthSwenlinTradingOscillatorData<'a> {
    Slices {
        advancing: &'a [f64],
        declining: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorParams;

#[derive(Debug, Clone)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorInput<'a> {
    pub data: DecisionPointBreadthSwenlinTradingOscillatorData<'a>,
    pub params: DecisionPointBreadthSwenlinTradingOscillatorParams,
}

impl<'a> DecisionPointBreadthSwenlinTradingOscillatorInput<'a> {
    #[inline]
    pub fn from_slices(
        advancing: &'a [f64],
        declining: &'a [f64],
        params: DecisionPointBreadthSwenlinTradingOscillatorParams,
    ) -> Self {
        Self {
            data: DecisionPointBreadthSwenlinTradingOscillatorData::Slices {
                advancing,
                declining,
            },
            params,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorBuilder {
    kernel: Kernel,
}

impl DecisionPointBreadthSwenlinTradingOscillatorBuilder {
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
    pub fn apply_slices(
        self,
        advancing: &[f64],
        declining: &[f64],
    ) -> Result<
        DecisionPointBreadthSwenlinTradingOscillatorOutput,
        DecisionPointBreadthSwenlinTradingOscillatorError,
    > {
        let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
            advancing,
            declining,
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        );
        decisionpoint_breadth_swenlin_trading_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<
        DecisionPointBreadthSwenlinTradingOscillatorStream,
        DecisionPointBreadthSwenlinTradingOscillatorError,
    > {
        let _ = self.kernel;
        DecisionPointBreadthSwenlinTradingOscillatorStream::try_new(
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        )
    }
}

#[derive(Debug, Error)]
pub enum DecisionPointBreadthSwenlinTradingOscillatorError {
    #[error("decisionpoint_breadth_swenlin_trading_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("decisionpoint_breadth_swenlin_trading_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "decisionpoint_breadth_swenlin_trading_oscillator: Inconsistent slice lengths: advancing={advancing_len}, declining={declining_len}"
    )]
    InconsistentSliceLengths {
        advancing_len: usize,
        declining_len: usize,
    },
    #[error(
        "decisionpoint_breadth_swenlin_trading_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "decisionpoint_breadth_swenlin_trading_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("decisionpoint_breadth_swenlin_trading_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorStream {
    ema_started: bool,
    ema_value: f64,
    sma_values: [f64; SMA_LENGTH],
    sma_idx: usize,
    sma_count: usize,
    sma_sum: f64,
}

impl DecisionPointBreadthSwenlinTradingOscillatorStream {
    #[inline]
    pub fn try_new(
        _params: DecisionPointBreadthSwenlinTradingOscillatorParams,
    ) -> Result<Self, DecisionPointBreadthSwenlinTradingOscillatorError> {
        Ok(Self::new_inner())
    }

    #[inline]
    fn new_inner() -> Self {
        Self {
            ema_started: false,
            ema_value: f64::NAN,
            sma_values: [0.0; SMA_LENGTH],
            sma_idx: 0,
            sma_count: 0,
            sma_sum: 0.0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new_inner();
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        WARMUP
    }

    #[inline]
    pub fn update(&mut self, advancing: f64, declining: f64) -> Option<f64> {
        if !valid_breadth_pair(advancing, declining) {
            self.reset();
            return None;
        }

        let breadth = ((advancing - declining) / (advancing + declining)) * MULTIPLIER;
        if !self.ema_started {
            self.ema_started = true;
            self.ema_value = breadth;
        } else {
            self.ema_value += EMA_ALPHA * (breadth - self.ema_value);
        }

        if self.sma_count == SMA_LENGTH {
            self.sma_sum -= self.sma_values[self.sma_idx];
        } else {
            self.sma_count += 1;
        }
        self.sma_values[self.sma_idx] = self.ema_value;
        self.sma_sum += self.ema_value;
        self.sma_idx += 1;
        if self.sma_idx == SMA_LENGTH {
            self.sma_idx = 0;
        }

        if self.sma_count < SMA_LENGTH {
            return None;
        }

        Some(self.sma_sum / SMA_LENGTH as f64)
    }
}

#[inline]
pub fn decisionpoint_breadth_swenlin_trading_oscillator(
    input: &DecisionPointBreadthSwenlinTradingOscillatorInput,
) -> Result<
    DecisionPointBreadthSwenlinTradingOscillatorOutput,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    decisionpoint_breadth_swenlin_trading_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_breadth_pair(advancing: f64, declining: f64) -> bool {
    if !advancing.is_finite() || !declining.is_finite() {
        return false;
    }
    let total = advancing + declining;
    total.is_finite() && total.abs() > EPSILON
}

#[inline(always)]
fn first_valid_pair(advancing: &[f64], declining: &[f64]) -> usize {
    let mut i = 0usize;
    while i < advancing.len() {
        if valid_breadth_pair(advancing[i], declining[i]) {
            return i;
        }
        i += 1;
    }
    advancing.len()
}

#[inline(always)]
fn count_valid_pairs(advancing: &[f64], declining: &[f64]) -> usize {
    let mut count = 0usize;
    for i in 0..advancing.len() {
        if valid_breadth_pair(advancing[i], declining[i]) {
            count += 1;
        }
    }
    count
}

#[inline(always)]
fn first_and_valid_pairs(advancing: &[f64], declining: &[f64]) -> (usize, usize) {
    let mut first = advancing.len();
    let mut count = 0usize;
    for i in 0..advancing.len() {
        if valid_breadth_pair(advancing[i], declining[i]) {
            if first == advancing.len() {
                first = i;
            }
            count += 1;
        }
    }
    (first, count)
}

#[inline(always)]
fn decisionpoint_breadth_swenlin_trading_oscillator_prepare<'a>(
    input: &'a DecisionPointBreadthSwenlinTradingOscillatorInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, Kernel), DecisionPointBreadthSwenlinTradingOscillatorError>
{
    let (advancing, declining) = match &input.data {
        DecisionPointBreadthSwenlinTradingOscillatorData::Slices {
            advancing,
            declining,
        } => (*advancing, *declining),
    };

    if advancing.len() != declining.len() {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::InconsistentSliceLengths {
                advancing_len: advancing.len(),
                declining_len: declining.len(),
            },
        );
    }
    if advancing.is_empty() {
        return Err(DecisionPointBreadthSwenlinTradingOscillatorError::EmptyInputData);
    }

    let (first, valid) = first_and_valid_pairs(advancing, declining);
    if first >= advancing.len() {
        return Err(DecisionPointBreadthSwenlinTradingOscillatorError::AllValuesNaN);
    }

    if valid < SMA_LENGTH {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::NotEnoughValidData {
                needed: SMA_LENGTH,
                valid,
            },
        );
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((advancing, declining, first, chosen))
}

#[inline(always)]
fn decisionpoint_breadth_swenlin_trading_oscillator_row(
    advancing: &[f64],
    declining: &[f64],
    out: &mut [f64],
) {
    let mut stream = DecisionPointBreadthSwenlinTradingOscillatorStream::new_inner();
    for i in 0..advancing.len() {
        out[i] = match stream.update(advancing[i], declining[i]) {
            Some(value) => value,
            None => f64::NAN,
        };
    }
}

#[inline]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_with_kernel(
    input: &DecisionPointBreadthSwenlinTradingOscillatorInput,
    kernel: Kernel,
) -> Result<
    DecisionPointBreadthSwenlinTradingOscillatorOutput,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    let (advancing, declining, _first, _chosen) =
        decisionpoint_breadth_swenlin_trading_oscillator_prepare(input, kernel)?;
    let mut values = alloc_uninit_f64(advancing.len());
    decisionpoint_breadth_swenlin_trading_oscillator_row(advancing, declining, &mut values);
    Ok(DecisionPointBreadthSwenlinTradingOscillatorOutput { values })
}

#[inline]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_into_slice(
    out: &mut [f64],
    input: &DecisionPointBreadthSwenlinTradingOscillatorInput,
    kernel: Kernel,
) -> Result<(), DecisionPointBreadthSwenlinTradingOscillatorError> {
    let expected = match &input.data {
        DecisionPointBreadthSwenlinTradingOscillatorData::Slices { advancing, .. } => {
            advancing.len()
        }
    };
    if out.len() != expected {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::OutputLengthMismatch {
                expected,
                got: out.len(),
            },
        );
    }
    let (advancing, declining, _first, _chosen) =
        decisionpoint_breadth_swenlin_trading_oscillator_prepare(input, kernel)?;
    decisionpoint_breadth_swenlin_trading_oscillator_row(advancing, declining, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_into(
    input: &DecisionPointBreadthSwenlinTradingOscillatorInput,
    out: &mut [f64],
) -> Result<(), DecisionPointBreadthSwenlinTradingOscillatorError> {
    decisionpoint_breadth_swenlin_trading_oscillator_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug, Default)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorBatchRange;

#[derive(Clone, Debug, Default)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorBatchBuilder {
    kernel: Kernel,
}

impl DecisionPointBreadthSwenlinTradingOscillatorBatchBuilder {
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
    pub fn apply_slices(
        self,
        advancing: &[f64],
        declining: &[f64],
    ) -> Result<
        DecisionPointBreadthSwenlinTradingOscillatorBatchOutput,
        DecisionPointBreadthSwenlinTradingOscillatorError,
    > {
        decisionpoint_breadth_swenlin_trading_oscillator_batch_with_kernel(
            advancing,
            declining,
            &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
            self.kernel,
        )
    }
}

#[derive(Clone, Debug)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DecisionPointBreadthSwenlinTradingOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DecisionPointBreadthSwenlinTradingOscillatorBatchOutput {
    #[inline]
    pub fn row_for_params(
        &self,
        params: &DecisionPointBreadthSwenlinTradingOscillatorParams,
    ) -> Option<usize> {
        self.combos.iter().position(|combo| combo == params)
    }

    #[inline]
    pub fn values_for(
        &self,
        params: &DecisionPointBreadthSwenlinTradingOscillatorParams,
    ) -> Option<&[f64]> {
        let row = self.row_for_params(params)?;
        let start = row * self.cols;
        self.values.get(start..start + self.cols)
    }
}

#[inline(always)]
fn expand_grid_decisionpoint_breadth_swenlin_trading_oscillator(
    _range: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
) -> Result<
    Vec<DecisionPointBreadthSwenlinTradingOscillatorParams>,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    Ok(vec![DecisionPointBreadthSwenlinTradingOscillatorParams])
}

#[inline]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_with_kernel(
    advancing: &[f64],
    declining: &[f64],
    sweep: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    kernel: Kernel,
) -> Result<
    DecisionPointBreadthSwenlinTradingOscillatorBatchOutput,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(
                DecisionPointBreadthSwenlinTradingOscillatorError::InvalidKernelForBatch(other),
            )
        }
    };
    decisionpoint_breadth_swenlin_trading_oscillator_batch_par_slices(
        advancing,
        declining,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_slice(
    advancing: &[f64],
    declining: &[f64],
    sweep: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    kernel: Kernel,
) -> Result<
    DecisionPointBreadthSwenlinTradingOscillatorBatchOutput,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    decisionpoint_breadth_swenlin_trading_oscillator_batch_inner(
        advancing, declining, sweep, kernel, false,
    )
}

#[inline]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_par_slices(
    advancing: &[f64],
    declining: &[f64],
    sweep: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    kernel: Kernel,
) -> Result<
    DecisionPointBreadthSwenlinTradingOscillatorBatchOutput,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    decisionpoint_breadth_swenlin_trading_oscillator_batch_inner(
        advancing, declining, sweep, kernel, true,
    )
}

#[inline(always)]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_inner(
    advancing: &[f64],
    declining: &[f64],
    sweep: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    _kernel: Kernel,
    _parallel: bool,
) -> Result<
    DecisionPointBreadthSwenlinTradingOscillatorBatchOutput,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    if advancing.len() != declining.len() {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::InconsistentSliceLengths {
                advancing_len: advancing.len(),
                declining_len: declining.len(),
            },
        );
    }
    if advancing.is_empty() {
        return Err(DecisionPointBreadthSwenlinTradingOscillatorError::EmptyInputData);
    }
    let first = first_valid_pair(advancing, declining);
    if first >= advancing.len() {
        return Err(DecisionPointBreadthSwenlinTradingOscillatorError::AllValuesNaN);
    }
    let valid = count_valid_pairs(advancing, declining);
    if valid < SMA_LENGTH {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::NotEnoughValidData {
                needed: SMA_LENGTH,
                valid,
            },
        );
    }

    let combos = expand_grid_decisionpoint_breadth_swenlin_trading_oscillator(sweep)?;
    let rows = combos.len();
    let cols = advancing.len();

    let mut matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut matrix, cols, &[(first + WARMUP).min(cols)]);
    let mut guard = ManuallyDrop::new(matrix);
    let out =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    decisionpoint_breadth_swenlin_trading_oscillator_row(advancing, declining, &mut out[..cols]);

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(DecisionPointBreadthSwenlinTradingOscillatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_inner_into(
    advancing: &[f64],
    declining: &[f64],
    sweep: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    _kernel: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<
    Vec<DecisionPointBreadthSwenlinTradingOscillatorParams>,
    DecisionPointBreadthSwenlinTradingOscillatorError,
> {
    if advancing.len() != declining.len() {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::InconsistentSliceLengths {
                advancing_len: advancing.len(),
                declining_len: declining.len(),
            },
        );
    }
    if advancing.is_empty() {
        return Err(DecisionPointBreadthSwenlinTradingOscillatorError::EmptyInputData);
    }
    let first = first_valid_pair(advancing, declining);
    if first >= advancing.len() {
        return Err(DecisionPointBreadthSwenlinTradingOscillatorError::AllValuesNaN);
    }
    let valid = count_valid_pairs(advancing, declining);
    if valid < SMA_LENGTH {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::NotEnoughValidData {
                needed: SMA_LENGTH,
                valid,
            },
        );
    }

    let combos = expand_grid_decisionpoint_breadth_swenlin_trading_oscillator(sweep)?;
    let rows = combos.len();
    let cols = advancing.len();
    let total = rows.checked_mul(cols).ok_or(
        DecisionPointBreadthSwenlinTradingOscillatorError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        },
    )?;
    if out.len() != total {
        return Err(
            DecisionPointBreadthSwenlinTradingOscillatorError::OutputLengthMismatch {
                expected: total,
                got: out.len(),
            },
        );
    }

    out[..cols].fill(f64::NAN);
    decisionpoint_breadth_swenlin_trading_oscillator_row(advancing, declining, &mut out[..cols]);
    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "decisionpoint_breadth_swenlin_trading_oscillator")]
#[pyo3(signature = (advancing, declining, kernel=None))]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_py<'py>(
    py: Python<'py>,
    advancing: PyReadonlyArray1<'py, f64>,
    declining: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let advancing = advancing.as_slice()?;
    let declining = declining.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
        advancing,
        declining,
        DecisionPointBreadthSwenlinTradingOscillatorParams,
    );
    let values = py
        .allow_threads(|| {
            decisionpoint_breadth_swenlin_trading_oscillator_with_kernel(&input, kern)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .values;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DecisionPointBreadthSwenlinTradingOscillatorStream")]
pub struct DecisionPointBreadthSwenlinTradingOscillatorStreamPy {
    inner: DecisionPointBreadthSwenlinTradingOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DecisionPointBreadthSwenlinTradingOscillatorStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: DecisionPointBreadthSwenlinTradingOscillatorStream::try_new(
                DecisionPointBreadthSwenlinTradingOscillatorParams,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn update(&mut self, advancing: f64, declining: f64) -> Option<f64> {
        self.inner.update(advancing, declining)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "decisionpoint_breadth_swenlin_trading_oscillator_batch")]
#[pyo3(signature = (advancing, declining, kernel=None))]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_py<'py>(
    py: Python<'py>,
    advancing: PyReadonlyArray1<'py, f64>,
    declining: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let advancing = advancing.as_slice()?;
    let declining = declining.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    if advancing.len() != declining.len() {
        return Err(PyValueError::new_err(
            "Advancing/declining slice length mismatch",
        ));
    }

    let rows = 1usize;
    let cols = advancing.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let batch = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        decisionpoint_breadth_swenlin_trading_oscillator_batch_inner_into(
            advancing,
            declining,
            &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
            batch,
            true,
            out_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("params", Vec::<f64>::new().into_pyarray(py))?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_decisionpoint_breadth_swenlin_trading_oscillator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        decisionpoint_breadth_swenlin_trading_oscillator_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        decisionpoint_breadth_swenlin_trading_oscillator_batch_py,
        module
    )?)?;
    module.add_class::<DecisionPointBreadthSwenlinTradingOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "decisionpoint_breadth_swenlin_trading_oscillator_js")]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_js(
    advancing: &[f64],
    declining: &[f64],
) -> Result<Vec<f64>, JsValue> {
    let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
        advancing,
        declining,
        DecisionPointBreadthSwenlinTradingOscillatorParams,
    );
    decisionpoint_breadth_swenlin_trading_oscillator(&input)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_into(
    advancing_ptr: *const f64,
    declining_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if advancing_ptr.is_null() || declining_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let advancing = std::slice::from_raw_parts(advancing_ptr, len);
        let declining = std::slice::from_raw_parts(declining_ptr, len);
        let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
            advancing,
            declining,
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        );
        if advancing_ptr == out_ptr || declining_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            decisionpoint_breadth_swenlin_trading_oscillator_into_slice(
                &mut temp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            decisionpoint_breadth_swenlin_trading_oscillator_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorBatchConfig {}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DecisionPointBreadthSwenlinTradingOscillatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DecisionPointBreadthSwenlinTradingOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "decisionpoint_breadth_swenlin_trading_oscillator_batch_js")]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_js(
    advancing: &[f64],
    declining: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let _: DecisionPointBreadthSwenlinTradingOscillatorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let output = decisionpoint_breadth_swenlin_trading_oscillator_batch_with_kernel(
        advancing,
        declining,
        &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&DecisionPointBreadthSwenlinTradingOscillatorBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_into(
    advancing_ptr: *const f64,
    declining_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if advancing_ptr.is_null() || declining_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let advancing = std::slice::from_raw_parts(advancing_ptr, len);
        let declining = std::slice::from_raw_parts(declining_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        decisionpoint_breadth_swenlin_trading_oscillator_batch_inner_into(
            advancing,
            declining,
            &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
            Kernel::Auto,
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(1)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_output_into_js(
    advancing: &[f64],
    declining: &[f64],
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = decisionpoint_breadth_swenlin_trading_oscillator_js(advancing, declining)?;
    crate::write_wasm_f64_output(
        "decisionpoint_breadth_swenlin_trading_oscillator_output_into_js",
        &values,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn decisionpoint_breadth_swenlin_trading_oscillator_batch_output_into_js(
    advancing: &[f64],
    declining: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        decisionpoint_breadth_swenlin_trading_oscillator_batch_js(advancing, declining, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "decisionpoint_breadth_swenlin_trading_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_breadth(length: usize) -> (Vec<f64>, Vec<f64>) {
        let mut advancing = Vec::with_capacity(length);
        let mut declining = Vec::with_capacity(length);
        for i in 0..length {
            let x = i as f64;
            advancing.push(1500.0 + x * 0.8 + (x * 0.07).sin() * 120.0 + 40.0);
            declining.push(1300.0 + x * 0.5 + (x * 0.05).cos() * 95.0 + 30.0);
        }
        (advancing, declining)
    }

    fn assert_series_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a.is_nan() && e.is_nan()) || (a - e).abs() <= 1e-12,
                "expected {e:?}, got {a:?}"
            );
        }
    }

    #[test]
    fn decisionpoint_breadth_swenlin_trading_oscillator_known_sequence() {
        let advancing = [60.0, 70.0, 80.0, 90.0, 100.0];
        let declining = [40.0, 30.0, 20.0, 10.0, 0.0];
        let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
            &advancing,
            &declining,
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        );
        let out =
            decisionpoint_breadth_swenlin_trading_oscillator_with_kernel(&input, Kernel::Scalar)
                .unwrap();
        assert!(out.values[..4].iter().all(|value| value.is_nan()));
        assert!((out.values[4] - 438.336).abs() <= 1e-9);
    }

    #[test]
    fn decisionpoint_breadth_swenlin_trading_oscillator_output_contract() {
        let (advancing, declining) = sample_breadth(256);
        let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
            &advancing,
            &declining,
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        );
        let out = decisionpoint_breadth_swenlin_trading_oscillator(&input).unwrap();
        assert_eq!(out.values.len(), advancing.len());
        assert_eq!(out.values.iter().position(|v| v.is_finite()).unwrap(), 4);
        assert!(out.values.last().unwrap().is_finite());
    }

    #[test]
    fn decisionpoint_breadth_swenlin_trading_oscillator_rejects_invalid_input() {
        let advancing = [1.0, 2.0, 3.0];
        let declining = [1.0, 2.0];
        let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
            &advancing,
            &declining,
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        );
        let err = decisionpoint_breadth_swenlin_trading_oscillator(&input).unwrap_err();
        assert!(matches!(
            err,
            DecisionPointBreadthSwenlinTradingOscillatorError::InconsistentSliceLengths { .. }
        ));
    }

    #[test]
    fn decisionpoint_breadth_swenlin_trading_oscillator_stream_matches_batch_with_reset() {
        let (mut advancing, mut declining) = sample_breadth(180);
        advancing[90] = 0.0;
        declining[90] = 0.0;

        let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
            &advancing,
            &declining,
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        );
        let batch = decisionpoint_breadth_swenlin_trading_oscillator(&input).unwrap();
        let mut stream = DecisionPointBreadthSwenlinTradingOscillatorStream::try_new(
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        )
        .unwrap();
        let mut streamed = Vec::with_capacity(advancing.len());
        for (&adv, &dec) in advancing.iter().zip(declining.iter()) {
            streamed.push(stream.update(adv, dec).unwrap_or(f64::NAN));
        }
        assert_series_eq(&batch.values, &streamed);
    }

    #[test]
    fn decisionpoint_breadth_swenlin_trading_oscillator_into_matches_api() {
        let (advancing, declining) = sample_breadth(192);
        let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
            &advancing,
            &declining,
            DecisionPointBreadthSwenlinTradingOscillatorParams,
        );
        let direct = decisionpoint_breadth_swenlin_trading_oscillator(&input).unwrap();
        let mut out = vec![0.0; advancing.len()];
        decisionpoint_breadth_swenlin_trading_oscillator_into(&input, &mut out).unwrap();
        assert_series_eq(&direct.values, &out);
    }

    #[test]
    fn decisionpoint_breadth_swenlin_trading_oscillator_batch_matches_single() {
        let (advancing, declining) = sample_breadth(192);
        let batch = decisionpoint_breadth_swenlin_trading_oscillator_batch_with_kernel(
            &advancing,
            &declining,
            &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
            Kernel::ScalarBatch,
        )
        .unwrap();
        let single = decisionpoint_breadth_swenlin_trading_oscillator(
            &DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
                &advancing,
                &declining,
                DecisionPointBreadthSwenlinTradingOscillatorParams,
            ),
        )
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, advancing.len());
        assert_series_eq(&batch.values[..advancing.len()], &single.values);
    }
}
