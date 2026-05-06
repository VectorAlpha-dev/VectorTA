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
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn polynomial_regression_extrapolation_output_into_js(
    data: &[f64],
    length: usize,
    extrapolate: usize,
    degree: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = polynomial_regression_extrapolation_js(data, length, extrapolate, degree)?;
    crate::write_wasm_f64_output(
        "polynomial_regression_extrapolation_output_into_js",
        &values,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn polynomial_regression_extrapolation_batch_unified_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = polynomial_regression_extrapolation_batch_unified_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "polynomial_regression_extrapolation_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::OnceLock;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 100;
const DEFAULT_EXTRAPOLATE: usize = 10;
const DEFAULT_DEGREE: usize = 3;
const MAX_DEGREE: usize = 8;
const SINGULAR_EPSILON: f64 = 1e-12;
static DEFAULT_WEIGHTS: OnceLock<Vec<f64>> = OnceLock::new();

impl<'a> AsRef<[f64]> for PolynomialRegressionExtrapolationInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        polynomial_regression_extrapolation_data(self)
    }
}

#[derive(Debug, Clone)]
pub enum PolynomialRegressionExtrapolationData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PolynomialRegressionExtrapolationOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PolynomialRegressionExtrapolationParams {
    pub length: Option<usize>,
    pub extrapolate: Option<usize>,
    pub degree: Option<usize>,
}

impl Default for PolynomialRegressionExtrapolationParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            extrapolate: Some(DEFAULT_EXTRAPOLATE),
            degree: Some(DEFAULT_DEGREE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PolynomialRegressionExtrapolationInput<'a> {
    pub data: PolynomialRegressionExtrapolationData<'a>,
    pub params: PolynomialRegressionExtrapolationParams,
}

impl<'a> PolynomialRegressionExtrapolationInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: PolynomialRegressionExtrapolationParams,
    ) -> Self {
        Self {
            data: PolynomialRegressionExtrapolationData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: PolynomialRegressionExtrapolationParams) -> Self {
        Self {
            data: PolynomialRegressionExtrapolationData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            PolynomialRegressionExtrapolationParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_extrapolate(&self) -> usize {
        self.params.extrapolate.unwrap_or(DEFAULT_EXTRAPOLATE)
    }

    #[inline]
    pub fn get_degree(&self) -> usize {
        self.params.degree.unwrap_or(DEFAULT_DEGREE)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PolynomialRegressionExtrapolationBuilder {
    length: Option<usize>,
    extrapolate: Option<usize>,
    degree: Option<usize>,
    kernel: Kernel,
}

impl Default for PolynomialRegressionExtrapolationBuilder {
    fn default() -> Self {
        Self {
            length: None,
            extrapolate: None,
            degree: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PolynomialRegressionExtrapolationBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline(always)]
    pub fn extrapolate(mut self, extrapolate: usize) -> Self {
        self.extrapolate = Some(extrapolate);
        self
    }

    #[inline(always)]
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = Some(degree);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<PolynomialRegressionExtrapolationOutput, PolynomialRegressionExtrapolationError>
    {
        let params = PolynomialRegressionExtrapolationParams {
            length: self.length,
            extrapolate: self.extrapolate,
            degree: self.degree,
        };
        let input = PolynomialRegressionExtrapolationInput::from_candles(candles, "close", params);
        polynomial_regression_extrapolation_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<PolynomialRegressionExtrapolationOutput, PolynomialRegressionExtrapolationError>
    {
        let params = PolynomialRegressionExtrapolationParams {
            length: self.length,
            extrapolate: self.extrapolate,
            degree: self.degree,
        };
        let input = PolynomialRegressionExtrapolationInput::from_slice(data, params);
        polynomial_regression_extrapolation_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<PolynomialRegressionExtrapolationStream, PolynomialRegressionExtrapolationError>
    {
        let params = PolynomialRegressionExtrapolationParams {
            length: self.length,
            extrapolate: self.extrapolate,
            degree: self.degree,
        };
        PolynomialRegressionExtrapolationStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum PolynomialRegressionExtrapolationError {
    #[error("polynomial_regression_extrapolation: Input data slice is empty.")]
    EmptyInputData,
    #[error("polynomial_regression_extrapolation: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "polynomial_regression_extrapolation: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "polynomial_regression_extrapolation: Invalid degree: degree = {degree}, max = {max_degree}"
    )]
    InvalidDegree { degree: usize, max_degree: usize },
    #[error(
        "polynomial_regression_extrapolation: Degree exceeds length: degree = {degree}, length = {length}"
    )]
    DegreeExceedsLength { degree: usize, length: usize },
    #[error(
        "polynomial_regression_extrapolation: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "polynomial_regression_extrapolation: Singular polynomial fit for length = {length}, degree = {degree}"
    )]
    SingularFit { length: usize, degree: usize },
    #[error(
        "polynomial_regression_extrapolation: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "polynomial_regression_extrapolation: Invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("polynomial_regression_extrapolation: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone)]
struct PreparedPolynomialRegressionExtrapolation<'a> {
    data: &'a [f64],
    first: usize,
    length: usize,
    weights: Vec<f64>,
    kernel: Kernel,
}

#[derive(Clone)]
struct BatchRowSpec {
    params: PolynomialRegressionExtrapolationParams,
    length: usize,
    weights: Vec<f64>,
}

#[inline]
pub fn polynomial_regression_extrapolation(
    input: &PolynomialRegressionExtrapolationInput,
) -> Result<PolynomialRegressionExtrapolationOutput, PolynomialRegressionExtrapolationError> {
    polynomial_regression_extrapolation_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn polynomial_regression_extrapolation_data<'a>(
    input: &'a PolynomialRegressionExtrapolationInput<'a>,
) -> &'a [f64] {
    match &input.data {
        PolynomialRegressionExtrapolationData::Slice(slice) => slice,
        PolynomialRegressionExtrapolationData::Candles { candles, source } => match *source {
            "open" => candles.open.as_slice(),
            "high" => candles.high.as_slice(),
            "low" => candles.low.as_slice(),
            "close" => candles.close.as_slice(),
            "volume" => candles.volume.as_slice(),
            _ => source_type(candles, source),
        },
    }
}

#[inline(always)]
fn normalize_single_kernel(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

fn solve_dense_system_in_place(matrix: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
    for pivot_col in 0..n {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col * n + pivot_col].abs();
        for row in (pivot_col + 1)..n {
            let candidate = matrix[row * n + pivot_col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= SINGULAR_EPSILON {
            return Err(());
        }
        if pivot_row != pivot_col {
            for col in pivot_col..n {
                matrix.swap(pivot_col * n + col, pivot_row * n + col);
            }
            rhs.swap(pivot_col, pivot_row);
        }
        let pivot = matrix[pivot_col * n + pivot_col];
        for row in (pivot_col + 1)..n {
            let factor = matrix[row * n + pivot_col] / pivot;
            if factor == 0.0 {
                continue;
            }
            matrix[row * n + pivot_col] = 0.0;
            for col in (pivot_col + 1)..n {
                matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
            }
            rhs[row] -= factor * rhs[pivot_col];
        }
    }

    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for col in (row + 1)..n {
            acc -= matrix[row * n + col] * rhs[col];
        }
        let pivot = matrix[row * n + row];
        if pivot.abs() <= SINGULAR_EPSILON {
            return Err(());
        }
        rhs[row] = acc / pivot;
    }
    Ok(())
}

fn build_forecast_weights(
    length: usize,
    extrapolate: usize,
    degree: usize,
) -> Result<Vec<f64>, PolynomialRegressionExtrapolationError> {
    if length == DEFAULT_LENGTH && extrapolate == DEFAULT_EXTRAPOLATE && degree == DEFAULT_DEGREE {
        return Ok(DEFAULT_WEIGHTS
            .get_or_init(|| {
                build_forecast_weights_uncached(length, extrapolate, degree)
                    .expect("default polynomial regression weights must be valid")
            })
            .clone());
    }

    build_forecast_weights_uncached(length, extrapolate, degree)
}

fn build_forecast_weights_uncached(
    length: usize,
    extrapolate: usize,
    degree: usize,
) -> Result<Vec<f64>, PolynomialRegressionExtrapolationError> {
    if degree > MAX_DEGREE {
        return Err(PolynomialRegressionExtrapolationError::InvalidDegree {
            degree,
            max_degree: MAX_DEGREE,
        });
    }
    if length == 0 {
        return Err(PolynomialRegressionExtrapolationError::InvalidLength {
            length,
            data_len: 0,
        });
    }
    if degree + 1 > length {
        return Err(PolynomialRegressionExtrapolationError::DegreeExceedsLength { degree, length });
    }

    let order_count = degree + 1;
    let mut normal = vec![0.0; order_count * order_count];
    for row in 0..order_count {
        for col in 0..order_count {
            let power = row + col;
            let mut sum = 0.0;
            for x in 0..length {
                sum += (x as f64).powi(power as i32);
            }
            normal[row * order_count + col] = sum;
        }
    }

    let x_eval = -(extrapolate as f64);
    let mut rhs = vec![0.0; order_count];
    for (power, value) in rhs.iter_mut().enumerate() {
        *value = x_eval.powi(power as i32);
    }
    solve_dense_system_in_place(&mut normal, &mut rhs, order_count)
        .map_err(|_| PolynomialRegressionExtrapolationError::SingularFit { length, degree })?;

    let mut weights = vec![0.0; length];
    for (x, weight) in weights.iter_mut().enumerate() {
        let xf = x as f64;
        let mut acc = 0.0f64;
        for power in (0..order_count).rev() {
            acc = acc.mul_add(xf, rhs[power]);
        }
        *weight = acc;
    }
    Ok(weights)
}

fn polynomial_regression_extrapolation_prepare<'a>(
    input: &'a PolynomialRegressionExtrapolationInput,
    kernel: Kernel,
) -> Result<PreparedPolynomialRegressionExtrapolation<'a>, PolynomialRegressionExtrapolationError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(PolynomialRegressionExtrapolationError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(PolynomialRegressionExtrapolationError::AllValuesNaN)?;

    let length = input.get_length();
    if length == 0 || length > data.len() {
        return Err(PolynomialRegressionExtrapolationError::InvalidLength {
            length,
            data_len: data.len(),
        });
    }

    let valid = data.len() - first;
    if valid < length {
        return Err(PolynomialRegressionExtrapolationError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    let weights = build_forecast_weights(length, input.get_extrapolate(), input.get_degree())?;
    Ok(PreparedPolynomialRegressionExtrapolation {
        data,
        first,
        length,
        weights,
        kernel: normalize_single_kernel(kernel),
    })
}

#[inline(always)]
fn polynomial_regression_extrapolation_all_finite(data: &[f64], first: usize) -> bool {
    let mut idx = first;
    while idx < data.len() {
        if data[idx].is_nan() {
            return false;
        }
        idx += 1;
    }
    true
}

#[inline(always)]
fn polynomial_regression_extrapolation_scalar_len_100_finite(
    data: &[f64],
    first: usize,
    weights: &[f64],
    out: &mut [f64],
) {
    let start = first + DEFAULT_LENGTH - 1;
    let len = data.len();
    unsafe {
        let dp = data.as_ptr();
        let wp = weights.as_ptr();
        let op = out.as_mut_ptr();
        let mut idx = start;
        while idx < len {
            let mut acc = 0.0;
            let mut offset = 0usize;
            while offset < DEFAULT_LENGTH {
                acc += *wp.add(offset) * *dp.add(idx - offset);
                offset += 1;
            }
            *op.add(idx) = acc;
            idx += 1;
        }
    }
}

#[inline(always)]
fn polynomial_regression_extrapolation_scalar(
    data: &[f64],
    first: usize,
    length: usize,
    weights: &[f64],
    out: &mut [f64],
) {
    if length == DEFAULT_LENGTH && polynomial_regression_extrapolation_all_finite(data, first) {
        polynomial_regression_extrapolation_scalar_len_100_finite(data, first, weights, out);
        return;
    }

    let mut valid_run = 0usize;
    for idx in first..data.len() {
        let value = data[idx];
        if value.is_nan() {
            valid_run = 0;
            out[idx] = f64::NAN;
            continue;
        }

        valid_run += 1;
        if valid_run < length {
            out[idx] = f64::NAN;
            continue;
        }

        let mut acc = 0.0;
        for offset in 0..length {
            acc += weights[offset] * data[idx - offset];
        }
        out[idx] = acc;
    }
}

#[inline(always)]
fn polynomial_regression_extrapolation_compute_into(
    prepared: &PreparedPolynomialRegressionExtrapolation,
    out: &mut [f64],
) {
    match prepared.kernel {
        Kernel::Scalar => polynomial_regression_extrapolation_scalar(
            prepared.data,
            prepared.first,
            prepared.length,
            &prepared.weights,
            out,
        ),
        _ => unreachable!(),
    }
}

pub fn polynomial_regression_extrapolation_with_kernel(
    input: &PolynomialRegressionExtrapolationInput,
    kernel: Kernel,
) -> Result<PolynomialRegressionExtrapolationOutput, PolynomialRegressionExtrapolationError> {
    let prepared = polynomial_regression_extrapolation_prepare(input, kernel)?;
    let warmup = prepared.first + prepared.length - 1;
    let mut out = alloc_with_nan_prefix(prepared.data.len(), warmup);
    polynomial_regression_extrapolation_compute_into(&prepared, &mut out);
    Ok(PolynomialRegressionExtrapolationOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn polynomial_regression_extrapolation_into(
    input: &PolynomialRegressionExtrapolationInput,
    out: &mut [f64],
) -> Result<(), PolynomialRegressionExtrapolationError> {
    polynomial_regression_extrapolation_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn polynomial_regression_extrapolation_into_slice(
    dst: &mut [f64],
    input: &PolynomialRegressionExtrapolationInput,
    kernel: Kernel,
) -> Result<(), PolynomialRegressionExtrapolationError> {
    let prepared = polynomial_regression_extrapolation_prepare(input, kernel)?;
    if dst.len() != prepared.data.len() {
        return Err(
            PolynomialRegressionExtrapolationError::OutputLengthMismatch {
                expected: prepared.data.len(),
                got: dst.len(),
            },
        );
    }
    let warmup = prepared.first + prepared.length - 1;
    for value in &mut dst[..warmup] {
        *value = f64::NAN;
    }
    polynomial_regression_extrapolation_compute_into(&prepared, dst);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct PolynomialRegressionExtrapolationStream {
    weights: Vec<f64>,
    buffer: Vec<f64>,
    head: usize,
    count: usize,
    valid_count: usize,
}

impl PolynomialRegressionExtrapolationStream {
    pub fn try_new(
        params: PolynomialRegressionExtrapolationParams,
    ) -> Result<Self, PolynomialRegressionExtrapolationError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let extrapolate = params.extrapolate.unwrap_or(DEFAULT_EXTRAPOLATE);
        let degree = params.degree.unwrap_or(DEFAULT_DEGREE);
        let weights = build_forecast_weights(length, extrapolate, degree)?;
        Ok(Self {
            weights,
            buffer: vec![f64::NAN; length],
            head: 0,
            count: 0,
            valid_count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let len = self.buffer.len();
        if self.count == len {
            let old = self.buffer[self.head];
            if !old.is_nan() {
                self.valid_count = self.valid_count.saturating_sub(1);
            }
        } else {
            self.count += 1;
        }

        self.buffer[self.head] = value;
        if !value.is_nan() {
            self.valid_count += 1;
        }
        self.head += 1;
        if self.head == len {
            self.head = 0;
        }

        if self.count < len {
            return None;
        }
        if self.valid_count < len {
            return Some(f64::NAN);
        }

        let mut idx = if self.head == 0 {
            len - 1
        } else {
            self.head - 1
        };
        let mut acc = 0.0;
        for weight in &self.weights {
            acc += *weight * self.buffer[idx];
            if idx == 0 {
                idx = len - 1;
            } else {
                idx -= 1;
            }
        }
        Some(acc)
    }
}

#[derive(Clone, Debug)]
pub struct PolynomialRegressionExtrapolationBatchRange {
    pub length: (usize, usize, usize),
    pub extrapolate: (usize, usize, usize),
    pub degree: (usize, usize, usize),
}

impl Default for PolynomialRegressionExtrapolationBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            extrapolate: (DEFAULT_EXTRAPOLATE, DEFAULT_EXTRAPOLATE, 0),
            degree: (DEFAULT_DEGREE, DEFAULT_DEGREE, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PolynomialRegressionExtrapolationBatchBuilder {
    range: PolynomialRegressionExtrapolationBatchRange,
    kernel: Kernel,
}

impl PolynomialRegressionExtrapolationBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    pub fn extrapolate_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.extrapolate = (start, end, step);
        self
    }

    pub fn degree_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.degree = (start, end, step);
        self
    }

    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    pub fn extrapolate_static(mut self, value: usize) -> Self {
        self.range.extrapolate = (value, value, 0);
        self
    }

    pub fn degree_static(mut self, value: usize) -> Self {
        self.range.degree = (value, value, 0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<PolynomialRegressionExtrapolationBatchOutput, PolynomialRegressionExtrapolationError>
    {
        polynomial_regression_extrapolation_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PolynomialRegressionExtrapolationBatchOutput, PolynomialRegressionExtrapolationError>
    {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct PolynomialRegressionExtrapolationBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PolynomialRegressionExtrapolationParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PolynomialRegressionExtrapolationBatchOutput {
    pub fn row_for_params(
        &self,
        params: &PolynomialRegressionExtrapolationParams,
    ) -> Option<usize> {
        self.combos.iter().position(|combo| combo == params)
    }

    pub fn values_for(&self, params: &PolynomialRegressionExtrapolationParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            let end = start.checked_add(self.cols)?;
            self.values.get(start..end)
        })
    }
}

fn axis_values(
    axis: &'static str,
    range: (usize, usize, usize),
) -> Result<Vec<usize>, PolynomialRegressionExtrapolationError> {
    let (start, end, step) = range;
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) if next > current => current = next,
                _ => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            if current == end {
                break;
            }
            match current.checked_sub(step) {
                Some(next) if next < current => current = next,
                _ => break,
            }
        }
    }

    if out.is_empty() || !out.last().is_some_and(|value| *value == end) {
        return Err(PolynomialRegressionExtrapolationError::InvalidRange {
            axis,
            start,
            end,
            step,
        });
    }
    Ok(out)
}

pub(crate) fn expand_grid(
    range: &PolynomialRegressionExtrapolationBatchRange,
) -> Result<Vec<PolynomialRegressionExtrapolationParams>, PolynomialRegressionExtrapolationError> {
    let lengths = axis_values("length", range.length)?;
    let extrapolates = axis_values("extrapolate", range.extrapolate)?;
    let degrees = axis_values("degree", range.degree)?;

    let total = lengths
        .len()
        .checked_mul(extrapolates.len())
        .and_then(|value| value.checked_mul(degrees.len()))
        .ok_or(PolynomialRegressionExtrapolationError::InvalidRange {
            axis: "grid",
            start: lengths.len(),
            end: extrapolates.len(),
            step: degrees.len(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &extrapolate in &extrapolates {
            for &degree in &degrees {
                out.push(PolynomialRegressionExtrapolationParams {
                    length: Some(length),
                    extrapolate: Some(extrapolate),
                    degree: Some(degree),
                });
            }
        }
    }
    Ok(out)
}

fn prepare_batch_specs(
    data: &[f64],
    sweep: &PolynomialRegressionExtrapolationBatchRange,
) -> Result<(usize, Vec<BatchRowSpec>), PolynomialRegressionExtrapolationError> {
    if data.is_empty() {
        return Err(PolynomialRegressionExtrapolationError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(PolynomialRegressionExtrapolationError::AllValuesNaN)?;
    let valid = data.len() - first;

    let combos = expand_grid(sweep)?;
    let mut max_length = 0usize;
    let mut specs = Vec::with_capacity(combos.len());
    for params in combos {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length == 0 || length > data.len() {
            return Err(PolynomialRegressionExtrapolationError::InvalidLength {
                length,
                data_len: data.len(),
            });
        }
        if valid < length {
            return Err(PolynomialRegressionExtrapolationError::NotEnoughValidData {
                needed: length,
                valid,
            });
        }
        let weights = build_forecast_weights(
            length,
            params.extrapolate.unwrap_or(DEFAULT_EXTRAPOLATE),
            params.degree.unwrap_or(DEFAULT_DEGREE),
        )?;
        max_length = max_length.max(length);
        specs.push(BatchRowSpec {
            params,
            length,
            weights,
        });
    }

    if valid < max_length {
        return Err(PolynomialRegressionExtrapolationError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    Ok((first, specs))
}

pub fn polynomial_regression_extrapolation_batch_with_kernel(
    data: &[f64],
    sweep: &PolynomialRegressionExtrapolationBatchRange,
    kernel: Kernel,
) -> Result<PolynomialRegressionExtrapolationBatchOutput, PolynomialRegressionExtrapolationError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(PolynomialRegressionExtrapolationError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::Scalar,
        _ => unreachable!(),
    };
    polynomial_regression_extrapolation_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn polynomial_regression_extrapolation_batch_slice(
    data: &[f64],
    sweep: &PolynomialRegressionExtrapolationBatchRange,
    kernel: Kernel,
) -> Result<PolynomialRegressionExtrapolationBatchOutput, PolynomialRegressionExtrapolationError> {
    polynomial_regression_extrapolation_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn polynomial_regression_extrapolation_batch_par_slice(
    data: &[f64],
    sweep: &PolynomialRegressionExtrapolationBatchRange,
    kernel: Kernel,
) -> Result<PolynomialRegressionExtrapolationBatchOutput, PolynomialRegressionExtrapolationError> {
    polynomial_regression_extrapolation_batch_inner(data, sweep, kernel, true)
}

fn polynomial_regression_extrapolation_batch_inner(
    data: &[f64],
    sweep: &PolynomialRegressionExtrapolationBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<PolynomialRegressionExtrapolationBatchOutput, PolynomialRegressionExtrapolationError> {
    let (first, specs) = prepare_batch_specs(data, sweep)?;
    let rows = specs.len();
    let cols = data.len();
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm_prefixes: Vec<usize> = specs.iter().map(|spec| first + spec.length - 1).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm_prefixes);

    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let spec = &specs[row];
        polynomial_regression_extrapolation_scalar(
            data,
            first,
            spec.length,
            &spec.weights,
            out_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, chunk)| do_row(row, chunk));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, chunk) in out.chunks_mut(cols).enumerate() {
                do_row(row, chunk);
            }
        }
    } else {
        for (row, chunk) in out.chunks_mut(cols).enumerate() {
            do_row(row, chunk);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };
    let combos = specs.into_iter().map(|spec| spec.params).collect();
    Ok(PolynomialRegressionExtrapolationBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn polynomial_regression_extrapolation_batch_inner_into(
    data: &[f64],
    sweep: &PolynomialRegressionExtrapolationBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PolynomialRegressionExtrapolationParams>, PolynomialRegressionExtrapolationError> {
    let (first, specs) = prepare_batch_specs(data, sweep)?;
    let rows = specs.len();
    let cols = data.len();
    let expected = rows.checked_mul(cols).ok_or(
        PolynomialRegressionExtrapolationError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        },
    )?;
    if out.len() != expected {
        return Err(
            PolynomialRegressionExtrapolationError::OutputLengthMismatch {
                expected,
                got: out.len(),
            },
        );
    }

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm_prefixes: Vec<usize> = specs.iter().map(|spec| first + spec.length - 1).collect();
    init_matrix_prefixes(out_mu, cols, &warm_prefixes);

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let spec = &specs[row];
        let dst: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        match kernel {
            Kernel::Scalar => polynomial_regression_extrapolation_scalar(
                data,
                first,
                spec.length,
                &spec.weights,
                dst,
            ),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_mu)| do_row(row, row_mu));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, row_mu);
            }
        }
    } else {
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    }

    Ok(specs.into_iter().map(|spec| spec.params).collect())
}

#[cfg(feature = "python")]
#[pyfunction(name = "polynomial_regression_extrapolation")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, extrapolate=DEFAULT_EXTRAPOLATE, degree=DEFAULT_DEGREE, kernel=None))]
pub fn polynomial_regression_extrapolation_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    extrapolate: usize,
    degree: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = PolynomialRegressionExtrapolationInput::from_slice(
        slice_in,
        PolynomialRegressionExtrapolationParams {
            length: Some(length),
            extrapolate: Some(extrapolate),
            degree: Some(degree),
        },
    );
    let values = py
        .allow_threads(|| {
            polynomial_regression_extrapolation_with_kernel(&input, kern)
                .map(|output| output.values)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "polynomial_regression_extrapolation_batch")]
#[pyo3(signature = (data, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), extrapolate_range=(DEFAULT_EXTRAPOLATE, DEFAULT_EXTRAPOLATE, 0), degree_range=(DEFAULT_DEGREE, DEFAULT_DEGREE, 0), kernel=None))]
pub fn polynomial_regression_extrapolation_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    extrapolate_range: (usize, usize, usize),
    degree_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = PolynomialRegressionExtrapolationBatchRange {
        length: length_range,
        extrapolate: extrapolate_range,
        degree: degree_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            let simd = match batch_kernel {
                Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::Scalar,
                _ => unreachable!(),
            };
            polynomial_regression_extrapolation_batch_inner_into(
                slice_in, &sweep, simd, true, slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "extrapolates",
        combos
            .iter()
            .map(|params| params.extrapolate.unwrap_or(DEFAULT_EXTRAPOLATE) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "degrees",
        combos
            .iter()
            .map(|params| params.degree.unwrap_or(DEFAULT_DEGREE) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "PolynomialRegressionExtrapolationStream")]
pub struct PolynomialRegressionExtrapolationStreamPy {
    inner: PolynomialRegressionExtrapolationStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PolynomialRegressionExtrapolationStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, extrapolate=DEFAULT_EXTRAPOLATE, degree=DEFAULT_DEGREE))]
    pub fn new(length: usize, extrapolate: usize, degree: usize) -> PyResult<Self> {
        let inner = PolynomialRegressionExtrapolationStream::try_new(
            PolynomialRegressionExtrapolationParams {
                length: Some(length),
                extrapolate: Some(extrapolate),
                degree: Some(degree),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct PolynomialRegressionExtrapolationBatchConfig {
    pub length_range: (usize, usize, usize),
    pub extrapolate_range: (usize, usize, usize),
    pub degree_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct PolynomialRegressionExtrapolationBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PolynomialRegressionExtrapolationParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn polynomial_regression_extrapolation_js(
    data: &[f64],
    length: usize,
    extrapolate: usize,
    degree: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = PolynomialRegressionExtrapolationInput::from_slice(
        data,
        PolynomialRegressionExtrapolationParams {
            length: Some(length),
            extrapolate: Some(extrapolate),
            degree: Some(degree),
        },
    );
    let mut out = vec![0.0; data.len()];
    polynomial_regression_extrapolation_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn polynomial_regression_extrapolation_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn polynomial_regression_extrapolation_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn polynomial_regression_extrapolation_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    extrapolate: usize,
    degree: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            data,
            PolynomialRegressionExtrapolationParams {
                length: Some(length),
                extrapolate: Some(extrapolate),
                degree: Some(degree),
            },
        );
        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            polynomial_regression_extrapolation_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            polynomial_regression_extrapolation_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = polynomial_regression_extrapolation_batch)]
pub fn polynomial_regression_extrapolation_batch_unified_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: PolynomialRegressionExtrapolationBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = PolynomialRegressionExtrapolationBatchRange {
        length: config.length_range,
        extrapolate: config.extrapolate_range,
        degree: config.degree_range,
    };
    let output = polynomial_regression_extrapolation_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = PolynomialRegressionExtrapolationBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn polynomial_regression_extrapolation_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    extrapolate_start: usize,
    extrapolate_end: usize,
    extrapolate_step: usize,
    degree_start: usize,
    degree_end: usize,
    degree_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = PolynomialRegressionExtrapolationBatchRange {
        length: (length_start, length_end, length_step),
        extrapolate: (extrapolate_start, extrapolate_end, extrapolate_step),
        degree: (degree_start, degree_end, degree_step),
    };
    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        polynomial_regression_extrapolation_batch_inner_into(
            data,
            &sweep,
            Kernel::Scalar,
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn assert_series_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() || e.is_nan() {
                assert!(
                    a.is_nan() && e.is_nan(),
                    "NaN mismatch at idx {}: actual={} expected={}",
                    idx,
                    a,
                    e
                );
            } else {
                assert!(
                    (a - e).abs() <= tol,
                    "value mismatch at idx {}: actual={} expected={} tol={}",
                    idx,
                    a,
                    e,
                    tol
                );
            }
        }
    }

    fn quadratic_data(size: usize) -> Vec<f64> {
        (0..size).map(|idx| (idx * idx) as f64).collect()
    }

    fn cubic_data(size: usize) -> Vec<f64> {
        (0..size)
            .map(|idx| {
                let x = idx as f64;
                x * x * x - 2.0 * x * x + 3.0 * x + 1.0
            })
            .collect()
    }

    fn check_quadratic_exactness(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = quadratic_data(24);
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(5),
                extrapolate: Some(2),
                degree: Some(2),
            },
        );
        let output = polynomial_regression_extrapolation_with_kernel(&input, kernel)?;
        let mut expected = vec![f64::NAN; data.len()];
        for idx in 4..data.len() {
            let x = idx as f64 + 2.0;
            expected[idx] = x * x;
        }
        assert_series_close(&output.values, &expected, 1e-9);
        Ok(())
    }

    fn check_cubic_exactness(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = cubic_data(30);
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(6),
                extrapolate: Some(3),
                degree: Some(3),
            },
        );
        let output = polynomial_regression_extrapolation_with_kernel(&input, kernel)?;
        let mut expected = vec![f64::NAN; data.len()];
        for idx in 5..data.len() {
            let x = idx as f64 + 3.0;
            expected[idx] = x * x * x - 2.0 * x * x + 3.0 * x + 1.0;
        }
        assert_series_close(&output.values, &expected, 1e-8);
        Ok(())
    }

    fn check_constant_degree_zero(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = vec![42.0; 18];
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(4),
                extrapolate: Some(7),
                degree: Some(0),
            },
        );
        let output = polynomial_regression_extrapolation_with_kernel(&input, kernel)?;
        assert!(output.values[..3].iter().all(|value| value.is_nan()));
        for value in &output.values[3..] {
            assert!((*value - 42.0).abs() <= 1e-12);
        }
        Ok(())
    }

    fn check_nan_gap_semantics(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = vec![
            f64::NAN,
            1.0,
            4.0,
            9.0,
            16.0,
            25.0,
            f64::NAN,
            64.0,
            81.0,
            100.0,
            121.0,
            144.0,
        ];
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(3),
                extrapolate: Some(1),
                degree: Some(2),
            },
        );
        let output = polynomial_regression_extrapolation_with_kernel(&input, kernel)?;
        assert!(output.values[..3].iter().all(|value| value.is_nan()));
        assert!((output.values[3] - 16.0).abs() <= 1e-9);
        assert!((output.values[4] - 25.0).abs() <= 1e-9);
        assert!((output.values[5] - 36.0).abs() <= 1e-9);
        assert!(output.values[6].is_nan());
        assert!(output.values[7].is_nan());
        assert!(output.values[8].is_nan());
        assert!((output.values[9] - 121.0).abs() <= 1e-9);
        assert!((output.values[10] - 144.0).abs() <= 1e-9);
        assert!((output.values[11] - 169.0).abs() <= 1e-9);
        Ok(())
    }

    fn check_into_matches_api(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = quadratic_data(20);
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(5),
                extrapolate: Some(2),
                degree: Some(2),
            },
        );
        let baseline = polynomial_regression_extrapolation_with_kernel(&input, kernel)?.values;
        let mut out = vec![0.0; data.len()];
        polynomial_regression_extrapolation_into_slice(&mut out, &input, kernel)?;
        assert_series_close(&out, &baseline, 1e-12);
        Ok(())
    }

    fn check_stream_matches_batch(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = cubic_data(24);
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(6),
                extrapolate: Some(2),
                degree: Some(3),
            },
        );
        let batch = polynomial_regression_extrapolation_with_kernel(&input, kernel)?.values;
        let mut stream = PolynomialRegressionExtrapolationStream::try_new(
            PolynomialRegressionExtrapolationParams {
                length: Some(6),
                extrapolate: Some(2),
                degree: Some(3),
            },
        )?;
        let streamed: Vec<f64> = data
            .iter()
            .map(|&value| stream.update(value).unwrap_or(f64::NAN))
            .collect();
        assert_series_close(&streamed, &batch, 1e-12);
        Ok(())
    }

    fn check_batch_matches_single(_name: &str, _kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = quadratic_data(22);
        let batch = PolynomialRegressionExtrapolationBatchBuilder::new()
            .length_range(4, 5, 1)
            .extrapolate_range(1, 2, 1)
            .degree_range(1, 2, 1)
            .kernel(Kernel::ScalarBatch)
            .apply_slice(&data)?;
        assert_eq!(batch.rows, 8);
        assert_eq!(batch.cols, data.len());

        for length in [4usize, 5] {
            for extrapolate in [1usize, 2] {
                for degree in [1usize, 2] {
                    let params = PolynomialRegressionExtrapolationParams {
                        length: Some(length),
                        extrapolate: Some(extrapolate),
                        degree: Some(degree),
                    };
                    let input =
                        PolynomialRegressionExtrapolationInput::from_slice(&data, params.clone());
                    let single = polynomial_regression_extrapolation(&input)?.values;
                    let row = batch.values_for(&params).unwrap();
                    assert_series_close(row, &single, 1e-12);
                }
            }
        }
        Ok(())
    }

    fn check_invalid_degree(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = quadratic_data(12);
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(4),
                extrapolate: Some(1),
                degree: Some(9),
            },
        );
        let err = polynomial_regression_extrapolation_with_kernel(&input, kernel).unwrap_err();
        assert!(matches!(
            err,
            PolynomialRegressionExtrapolationError::InvalidDegree { .. }
        ));
        Ok(())
    }

    fn check_degree_exceeds_length(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = quadratic_data(12);
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(3),
                extrapolate: Some(1),
                degree: Some(3),
            },
        );
        let err = polynomial_regression_extrapolation_with_kernel(&input, kernel).unwrap_err();
        assert!(matches!(
            err,
            PolynomialRegressionExtrapolationError::DegreeExceedsLength { .. }
        ));
        Ok(())
    }

    fn check_all_nan(_name: &str, kernel: Kernel) -> Result<(), Box<dyn StdError>> {
        let data = vec![f64::NAN; 16];
        let input = PolynomialRegressionExtrapolationInput::from_slice(
            &data,
            PolynomialRegressionExtrapolationParams {
                length: Some(4),
                extrapolate: Some(1),
                degree: Some(2),
            },
        );
        let err = polynomial_regression_extrapolation_with_kernel(&input, kernel).unwrap_err();
        assert!(matches!(
            err,
            PolynomialRegressionExtrapolationError::AllValuesNaN
        ));
        Ok(())
    }

    macro_rules! generate_pre_tests {
        ($($name:ident),* $(,)?) => {
            $(
                paste! {
                    #[test]
                    fn [<polynomial_regression_extrapolation_ $name _scalar>]() -> Result<(), Box<dyn StdError>> {
                        $name("scalar", Kernel::Scalar)
                    }

                    #[test]
                    fn [<polynomial_regression_extrapolation_ $name _auto>]() -> Result<(), Box<dyn StdError>> {
                        $name("auto", Kernel::Auto)
                    }
                }
            )*
        };
    }

    generate_pre_tests!(
        check_quadratic_exactness,
        check_cubic_exactness,
        check_constant_degree_zero,
        check_nan_gap_semantics,
        check_into_matches_api,
        check_stream_matches_batch,
        check_batch_matches_single,
        check_invalid_degree,
        check_degree_exceeds_length,
        check_all_nan,
    );

    #[test]
    fn polynomial_regression_extrapolation_default_candles_smoke() -> Result<(), Box<dyn StdError>>
    {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let input = PolynomialRegressionExtrapolationInput::with_default_candles(&candles);
        let output = polynomial_regression_extrapolation(&input)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
}
