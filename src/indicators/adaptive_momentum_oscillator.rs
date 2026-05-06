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

use crate::indicators::moving_averages::linreg::{LinRegParams, LinRegStream};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_momentum_oscillator_output_into_js(
    data: &[f64],
    length: Option<usize>,
    smoothing_length: Option<usize>,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_momentum_oscillator_js(data, length, smoothing_length)?;
    crate::write_wasm_object_f64_outputs("adaptive_momentum_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_momentum_oscillator_batch_unified_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_momentum_oscillator_batch_unified_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "adaptive_momentum_oscillator_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 14;
const DEFAULT_SMOOTHING_LENGTH: usize = 9;
const MIN_LENGTH: usize = 1;
const MIN_SMOOTHING_LENGTH: usize = 1;

impl<'a> AsRef<[f64]> for AdaptiveMomentumOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AdaptiveMomentumOscillatorData::Slice(slice) => slice,
            AdaptiveMomentumOscillatorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AdaptiveMomentumOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AdaptiveMomentumOscillatorOutput {
    pub amo: Vec<f64>,
    pub ama: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMomentumOscillatorOutputField {
    Amo,
    Ama,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdaptiveMomentumOscillatorParams {
    pub length: Option<usize>,
    pub smoothing_length: Option<usize>,
}

impl Default for AdaptiveMomentumOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            smoothing_length: Some(DEFAULT_SMOOTHING_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveMomentumOscillatorInput<'a> {
    pub data: AdaptiveMomentumOscillatorData<'a>,
    pub params: AdaptiveMomentumOscillatorParams,
}

impl<'a> AdaptiveMomentumOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: AdaptiveMomentumOscillatorParams,
    ) -> Self {
        Self {
            data: AdaptiveMomentumOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AdaptiveMomentumOscillatorParams) -> Self {
        Self {
            data: AdaptiveMomentumOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            AdaptiveMomentumOscillatorParams::default(),
        )
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline(always)]
    pub fn get_smoothing_length(&self) -> usize {
        self.params
            .smoothing_length
            .unwrap_or(DEFAULT_SMOOTHING_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdaptiveMomentumOscillatorBuilder {
    length: Option<usize>,
    smoothing_length: Option<usize>,
    kernel: Kernel,
}

impl Default for AdaptiveMomentumOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smoothing_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdaptiveMomentumOscillatorBuilder {
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
    pub fn smoothing_length(mut self, smoothing_length: usize) -> Self {
        self.smoothing_length = Some(smoothing_length);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> AdaptiveMomentumOscillatorParams {
        AdaptiveMomentumOscillatorParams {
            length: self.length,
            smoothing_length: self.smoothing_length,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<AdaptiveMomentumOscillatorOutput, AdaptiveMomentumOscillatorError> {
        adaptive_momentum_oscillator_with_kernel(
            &AdaptiveMomentumOscillatorInput::from_candles(candles, "close", self.params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdaptiveMomentumOscillatorOutput, AdaptiveMomentumOscillatorError> {
        adaptive_momentum_oscillator_with_kernel(
            &AdaptiveMomentumOscillatorInput::from_slice(data, self.params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<AdaptiveMomentumOscillatorStream, AdaptiveMomentumOscillatorError> {
        AdaptiveMomentumOscillatorStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum AdaptiveMomentumOscillatorError {
    #[error("adaptive_momentum_oscillator: input data slice is empty.")]
    EmptyInputData,
    #[error("adaptive_momentum_oscillator: all values are NaN.")]
    AllValuesNaN,
    #[error("adaptive_momentum_oscillator: invalid length: {length}. Expected >= 1.")]
    InvalidLength { length: usize },
    #[error(
        "adaptive_momentum_oscillator: invalid smoothing_length: {smoothing_length}. Expected >= 1."
    )]
    InvalidSmoothingLength { smoothing_length: usize },
    #[error(
        "adaptive_momentum_oscillator: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "adaptive_momentum_oscillator: output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "adaptive_momentum_oscillator: invalid length range: start={start}, end={end}, step={step}"
    )]
    InvalidLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error(
        "adaptive_momentum_oscillator: invalid smoothing length range: start={start}, end={end}, step={step}"
    )]
    InvalidSmoothingLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("adaptive_momentum_oscillator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Copy, Clone, Debug)]
struct PreparedInput<'a> {
    data: &'a [f64],
    len: usize,
    length: usize,
    smoothing_length: usize,
}

#[inline(always)]
fn validate_params(
    length: usize,
    smoothing_length: usize,
) -> Result<(), AdaptiveMomentumOscillatorError> {
    if length < MIN_LENGTH {
        return Err(AdaptiveMomentumOscillatorError::InvalidLength { length });
    }
    if smoothing_length < MIN_SMOOTHING_LENGTH {
        return Err(AdaptiveMomentumOscillatorError::InvalidSmoothingLength { smoothing_length });
    }
    Ok(())
}

#[inline(always)]
fn normalize_single_kernel_to_scalar(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Auto
            | Kernel::Scalar
            | Kernel::ScalarBatch
            | Kernel::Avx2
            | Kernel::Avx2Batch
            | Kernel::Avx512
            | Kernel::Avx512Batch => Kernel::Scalar,
        },
        _ => Kernel::Scalar,
    }
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a AdaptiveMomentumOscillatorInput<'a>,
) -> Result<PreparedInput<'a>, AdaptiveMomentumOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(AdaptiveMomentumOscillatorError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(AdaptiveMomentumOscillatorError::AllValuesNaN)?;

    let length = input.get_length();
    let smoothing_length = input.get_smoothing_length();
    validate_params(length, smoothing_length)?;

    let valid = data.len() - first_valid;
    let needed = length + smoothing_length;
    if valid < needed {
        return Err(AdaptiveMomentumOscillatorError::NotEnoughValidData { needed, valid });
    }

    Ok(PreparedInput {
        data,
        len: data.len(),
        length,
        smoothing_length,
    })
}

#[derive(Debug, Clone)]
struct AmoRawState {
    length: usize,
    ring: Vec<f64>,
    head: usize,
    count: usize,
}

impl AmoRawState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            length,
            ring: vec![f64::NAN; length],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn history_value(&self, lag: usize) -> f64 {
        let idx = (self.head + self.length - lag) % self.length;
        self.ring[idx]
    }

    #[inline(always)]
    fn push(&mut self, value: f64) {
        self.ring[self.head] = value;
        self.head += 1;
        if self.head == self.length {
            self.head = 0;
        }
        if self.count < self.length {
            self.count += 1;
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        let out = if value.is_finite() && self.count >= self.length {
            let mut best_abs = -1.0;
            let mut best_delta = f64::NAN;
            let mut valid = true;
            for lag in 1..=self.length {
                let past = self.history_value(lag);
                if !past.is_finite() {
                    valid = false;
                    break;
                }
                let delta = value - past;
                let abs_delta = delta.abs();
                if abs_delta >= best_abs {
                    best_abs = abs_delta;
                    best_delta = delta;
                }
            }
            if valid {
                best_delta
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        self.push(value);
        out
    }
}

#[derive(Debug, Clone)]
struct AdaptiveAverageState {
    length: usize,
    change_ring: Vec<f64>,
    head: usize,
    count: usize,
    change_sum: f64,
    prev: f64,
    have_prev: bool,
    value: f64,
}

impl AdaptiveAverageState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            length,
            change_ring: vec![0.0; length],
            head: 0,
            count: 0,
            change_sum: 0.0,
            prev: f64::NAN,
            have_prev: false,
            value: 0.0,
        }
    }

    #[inline(always)]
    fn push_change(&mut self, change: f64) {
        let normalized = if change.is_finite() { change } else { 0.0 };
        if self.count < self.length {
            self.change_ring[self.head] = normalized;
            self.change_sum += normalized;
            self.count += 1;
        } else {
            let old = self.change_ring[self.head];
            self.change_ring[self.head] = normalized;
            self.change_sum += normalized - old;
        }
        self.head += 1;
        if self.head == self.length {
            self.head = 0;
        }
    }

    #[inline(always)]
    fn update(&mut self, input: f64) -> Option<f64> {
        let change = if self.have_prev && input.is_finite() && self.prev.is_finite() {
            (input - self.prev).abs()
        } else {
            0.0
        };
        self.push_change(change);

        if input.is_finite() && self.change_sum > 0.0 {
            let efficiency_ratio = input.abs() / self.change_sum;
            let delta = efficiency_ratio * (input - self.value);
            if delta.is_finite() {
                self.value += delta;
            }
        }

        self.prev = input;
        self.have_prev = true;

        if input.is_finite() {
            Some(self.value)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct AdaptiveMomentumOscillatorCore {
    raw: AmoRawState,
    smoothing: LinRegStream,
    average: AdaptiveAverageState,
}

impl AdaptiveMomentumOscillatorCore {
    #[inline(always)]
    fn try_new(
        params: &AdaptiveMomentumOscillatorParams,
    ) -> Result<Self, AdaptiveMomentumOscillatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let smoothing_length = params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
        validate_params(length, smoothing_length)?;
        Ok(Self {
            raw: AmoRawState::new(length),
            smoothing: LinRegStream::try_new(LinRegParams {
                period: Some(smoothing_length),
            })
            .map_err(
                |_| AdaptiveMomentumOscillatorError::InvalidSmoothingLength { smoothing_length },
            )?,
            average: AdaptiveAverageState::new(length),
        })
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let raw = self.raw.update(value);
        let amo = self.smoothing.update(raw).unwrap_or(f64::NAN);
        let ama = self.average.update(amo);
        if amo.is_finite() {
            Some((amo, ama.unwrap_or(f64::NAN)))
        } else {
            None
        }
    }
}

#[inline(always)]
fn ensure_output_len(expected: usize, got: usize) -> Result<(), AdaptiveMomentumOscillatorError> {
    if expected == got {
        Ok(())
    } else {
        Err(AdaptiveMomentumOscillatorError::OutputLengthMismatch { expected, got })
    }
}

#[inline(always)]
fn compute_into_slices(
    prepared: PreparedInput<'_>,
    amo_out: &mut [f64],
    ama_out: &mut [f64],
) -> Result<(), AdaptiveMomentumOscillatorError> {
    ensure_output_len(prepared.len, amo_out.len())?;
    ensure_output_len(prepared.len, ama_out.len())?;

    amo_out.fill(f64::NAN);
    ama_out.fill(f64::NAN);

    let mut core = AdaptiveMomentumOscillatorCore::try_new(&AdaptiveMomentumOscillatorParams {
        length: Some(prepared.length),
        smoothing_length: Some(prepared.smoothing_length),
    })?;

    for idx in 0..prepared.len {
        if let Some((amo, ama)) = core.update(prepared.data[idx]) {
            amo_out[idx] = amo;
            ama_out[idx] = ama;
        }
    }

    Ok(())
}

#[inline(always)]
fn compute_output_into_slice(
    prepared: PreparedInput<'_>,
    field: AdaptiveMomentumOscillatorOutputField,
    out: &mut [f64],
) -> Result<(), AdaptiveMomentumOscillatorError> {
    ensure_output_len(prepared.len, out.len())?;
    out.fill(f64::NAN);

    let mut core = AdaptiveMomentumOscillatorCore::try_new(&AdaptiveMomentumOscillatorParams {
        length: Some(prepared.length),
        smoothing_length: Some(prepared.smoothing_length),
    })?;

    for idx in 0..prepared.len {
        if let Some((amo, ama)) = core.update(prepared.data[idx]) {
            out[idx] = match field {
                AdaptiveMomentumOscillatorOutputField::Amo => amo,
                AdaptiveMomentumOscillatorOutputField::Ama => ama,
            };
        }
    }

    Ok(())
}

#[inline]
pub fn adaptive_momentum_oscillator(
    input: &AdaptiveMomentumOscillatorInput<'_>,
) -> Result<AdaptiveMomentumOscillatorOutput, AdaptiveMomentumOscillatorError> {
    adaptive_momentum_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn adaptive_momentum_oscillator_with_kernel(
    input: &AdaptiveMomentumOscillatorInput<'_>,
    kernel: Kernel,
) -> Result<AdaptiveMomentumOscillatorOutput, AdaptiveMomentumOscillatorError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let prepared = prepare_input(input)?;
    let mut amo = vec![0.0; prepared.len];
    let mut ama = vec![0.0; prepared.len];
    compute_into_slices(prepared, &mut amo, &mut ama)?;
    Ok(AdaptiveMomentumOscillatorOutput { amo, ama })
}

#[inline]
pub fn adaptive_momentum_oscillator_into_slice(
    amo_out: &mut [f64],
    ama_out: &mut [f64],
    input: &AdaptiveMomentumOscillatorInput<'_>,
    kernel: Kernel,
) -> Result<(), AdaptiveMomentumOscillatorError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let prepared = prepare_input(input)?;
    compute_into_slices(prepared, amo_out, ama_out)
}

pub fn adaptive_momentum_oscillator_output_into_slice(
    out: &mut [f64],
    input: &AdaptiveMomentumOscillatorInput<'_>,
    kernel: Kernel,
    field: AdaptiveMomentumOscillatorOutputField,
) -> Result<(), AdaptiveMomentumOscillatorError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let prepared = prepare_input(input)?;
    compute_output_into_slice(prepared, field, out)
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn adaptive_momentum_oscillator_into(
    input: &AdaptiveMomentumOscillatorInput<'_>,
    amo_out: &mut [f64],
    ama_out: &mut [f64],
) -> Result<(), AdaptiveMomentumOscillatorError> {
    adaptive_momentum_oscillator_into_slice(amo_out, ama_out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct AdaptiveMomentumOscillatorStream {
    params: AdaptiveMomentumOscillatorParams,
    core: AdaptiveMomentumOscillatorCore,
}

impl AdaptiveMomentumOscillatorStream {
    pub fn try_new(
        params: AdaptiveMomentumOscillatorParams,
    ) -> Result<Self, AdaptiveMomentumOscillatorError> {
        let core = AdaptiveMomentumOscillatorCore::try_new(&params)?;
        Ok(Self { params, core })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.core.update(value)
    }

    pub fn reset(&mut self) {
        self.core = AdaptiveMomentumOscillatorCore::try_new(&self.params)
            .expect("adaptive_momentum_oscillator stream reset should revalidate existing params");
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveMomentumOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub smoothing_length: (usize, usize, usize),
}

impl Default for AdaptiveMomentumOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            smoothing_length: (DEFAULT_SMOOTHING_LENGTH, DEFAULT_SMOOTHING_LENGTH, 0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveMomentumOscillatorBatchBuilder {
    range: AdaptiveMomentumOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for AdaptiveMomentumOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: AdaptiveMomentumOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl AdaptiveMomentumOscillatorBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    pub fn smoothing_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_length = (start, end, step);
        self
    }

    pub fn smoothing_length_static(mut self, smoothing_length: usize) -> Self {
        self.range.smoothing_length = (smoothing_length, smoothing_length, 0);
        self
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorError> {
        adaptive_momentum_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorError> {
        self.apply_slice(source_type(candles, "close"))
    }

    pub fn apply_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveMomentumOscillatorBatchOutput {
    pub amo: Vec<f64>,
    pub ama: Vec<f64>,
    pub combos: Vec<AdaptiveMomentumOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AdaptiveMomentumOscillatorBatchOutput {
    pub fn row_for_params(&self, params: &AdaptiveMomentumOscillatorParams) -> Option<usize> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let smoothing_length = params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == length
                && combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH) == smoothing_length
        })
    }

    pub fn amo_for(&self, params: &AdaptiveMomentumOscillatorParams) -> Option<&[f64]> {
        let row = self.row_for_params(params)?;
        let start = row.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.amo.get(start..end)
    }

    pub fn ama_for(&self, params: &AdaptiveMomentumOscillatorParams) -> Option<&[f64]> {
        let row = self.row_for_params(params)?;
        let start = row.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.ama.get(start..end)
    }
}

#[inline(always)]
fn expand_axis(
    axis: (usize, usize, usize),
    smoothing: bool,
) -> Result<Vec<usize>, AdaptiveMomentumOscillatorError> {
    let (start, end, step) = axis;
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut values = Vec::new();
    if start < end {
        let mut current = start;
        while current <= end {
            values.push(current);
            match current.checked_add(step) {
                Some(next) if next > current => current = next,
                _ => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            values.push(current);
            if current < end.saturating_add(step) {
                break;
            }
            current = current.saturating_sub(step);
        }
    }

    if values.is_empty() {
        return Err(if smoothing {
            AdaptiveMomentumOscillatorError::InvalidSmoothingLengthRange { start, end, step }
        } else {
            AdaptiveMomentumOscillatorError::InvalidLengthRange { start, end, step }
        });
    }

    Ok(values)
}

fn expand_grid(
    range: &AdaptiveMomentumOscillatorBatchRange,
) -> Result<Vec<AdaptiveMomentumOscillatorParams>, AdaptiveMomentumOscillatorError> {
    let lengths = expand_axis(range.length, false)?;
    let smoothing_lengths = expand_axis(range.smoothing_length, true)?;
    let mut combos = Vec::with_capacity(lengths.len() * smoothing_lengths.len());
    for &length in &lengths {
        for &smoothing_length in &smoothing_lengths {
            validate_params(length, smoothing_length)?;
            combos.push(AdaptiveMomentumOscillatorParams {
                length: Some(length),
                smoothing_length: Some(smoothing_length),
            });
        }
    }
    Ok(combos)
}

pub fn expand_grid_adaptive_momentum_oscillator(
    range: &AdaptiveMomentumOscillatorBatchRange,
) -> Result<Vec<AdaptiveMomentumOscillatorParams>, AdaptiveMomentumOscillatorError> {
    expand_grid(range)
}

#[inline(always)]
pub fn adaptive_momentum_oscillator_batch_slice(
    data: &[f64],
    sweep: &AdaptiveMomentumOscillatorBatchRange,
) -> Result<AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorError> {
    adaptive_momentum_oscillator_batch_inner(data, sweep, Kernel::Scalar, false)
}

#[inline(always)]
pub fn adaptive_momentum_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &AdaptiveMomentumOscillatorBatchRange,
) -> Result<AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorError> {
    adaptive_momentum_oscillator_batch_inner(data, sweep, Kernel::Scalar, true)
}

pub fn adaptive_momentum_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &AdaptiveMomentumOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(AdaptiveMomentumOscillatorError::InvalidKernelForBatch(
                other,
            ))
        }
    };

    let scalar_kernel = match batch_kernel {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => unreachable!(),
    };

    adaptive_momentum_oscillator_batch_inner(
        data,
        sweep,
        scalar_kernel,
        !matches!(batch_kernel, Kernel::ScalarBatch),
    )
}

fn adaptive_momentum_oscillator_batch_inner(
    data: &[f64],
    sweep: &AdaptiveMomentumOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let input = AdaptiveMomentumOscillatorInput::from_slice(
        data,
        AdaptiveMomentumOscillatorParams::default(),
    );
    let prepared = prepare_input(&input)?;
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    let amo_mu = make_uninit_matrix(rows, cols);
    let ama_mu = make_uninit_matrix(rows, cols);
    let mut amo_guard = ManuallyDrop::new(amo_mu);
    let mut ama_guard = ManuallyDrop::new(ama_mu);

    let amo_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(amo_guard.as_mut_ptr() as *mut f64, amo_guard.len())
    };
    let ama_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(ama_guard.as_mut_ptr() as *mut f64, ama_guard.len())
    };

    let do_row = |row: usize, amo_row: &mut [f64], ama_row: &mut [f64]| {
        let combo = &combos[row];
        compute_into_slices(
            PreparedInput {
                data: prepared.data,
                len: cols,
                length: combo.length.unwrap_or(DEFAULT_LENGTH),
                smoothing_length: combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH),
            },
            amo_row,
            ama_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            amo_out
                .par_chunks_mut(cols)
                .zip(ama_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, (amo_row, ama_row))| do_row(row, amo_row, ama_row))?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (amo_row, ama_row)) in amo_out
                .chunks_mut(cols)
                .zip(ama_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, amo_row, ama_row)?;
            }
        }
    } else {
        for (row, (amo_row, ama_row)) in amo_out
            .chunks_mut(cols)
            .zip(ama_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, amo_row, ama_row)?;
        }
    }

    let amo = unsafe {
        Vec::from_raw_parts(
            amo_guard.as_mut_ptr() as *mut f64,
            amo_guard.len(),
            amo_guard.capacity(),
        )
    };
    let ama = unsafe {
        Vec::from_raw_parts(
            ama_guard.as_mut_ptr() as *mut f64,
            ama_guard.len(),
            ama_guard.capacity(),
        )
    };

    Ok(AdaptiveMomentumOscillatorBatchOutput {
        amo,
        ama,
        combos,
        rows,
        cols,
    })
}

fn adaptive_momentum_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &AdaptiveMomentumOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    amo_out: &mut [f64],
    ama_out: &mut [f64],
) -> Result<Vec<AdaptiveMomentumOscillatorParams>, AdaptiveMomentumOscillatorError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let input = AdaptiveMomentumOscillatorInput::from_slice(
        data,
        AdaptiveMomentumOscillatorParams::default(),
    );
    let prepared = prepare_input(&input)?;
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or(AdaptiveMomentumOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                got: amo_out.len(),
            })?;
    ensure_output_len(expected, amo_out.len())?;
    ensure_output_len(expected, ama_out.len())?;

    let do_row = |row: usize, amo_row: &mut [f64], ama_row: &mut [f64]| {
        let combo = &combos[row];
        compute_into_slices(
            PreparedInput {
                data: prepared.data,
                len: cols,
                length: combo.length.unwrap_or(DEFAULT_LENGTH),
                smoothing_length: combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH),
            },
            amo_row,
            ama_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            amo_out
                .par_chunks_mut(cols)
                .zip(ama_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, (amo_row, ama_row))| do_row(row, amo_row, ama_row))?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (amo_row, ama_row)) in amo_out
                .chunks_mut(cols)
                .zip(ama_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, amo_row, ama_row)?;
            }
        }
    } else {
        for (row, (amo_row, ama_row)) in amo_out
            .chunks_mut(cols)
            .zip(ama_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, amo_row, ama_row)?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_momentum_oscillator")]
#[pyo3(signature = (data, length=None, smoothing_length=None, *, kernel=None))]
pub fn adaptive_momentum_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: Option<usize>,
    smoothing_length: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = AdaptiveMomentumOscillatorInput::from_slice(
        slice,
        AdaptiveMomentumOscillatorParams {
            length,
            smoothing_length,
        },
    );
    let out = py
        .allow_threads(|| adaptive_momentum_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.amo.into_pyarray(py), out.ama.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_momentum_oscillator_batch")]
#[pyo3(signature = (data, length_range, smoothing_length_range, kernel=None))]
pub fn adaptive_momentum_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    smoothing_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = AdaptiveMomentumOscillatorBatchRange {
        length: length_range,
        smoothing_length: smoothing_length_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let amo_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let ama_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let amo_slice = unsafe { amo_arr.as_slice_mut()? };
    let ama_slice = unsafe { ama_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            adaptive_momentum_oscillator_batch_inner_into(
                slice,
                &sweep,
                batch_kernel,
                !matches!(batch_kernel, Kernel::ScalarBatch),
                amo_slice,
                ama_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("amo", amo_arr.reshape((rows, cols))?)?;
    dict.set_item("ama", ama_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smoothing_lengths",
        combos
            .iter()
            .map(|combo| combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AdaptiveMomentumOscillatorStream")]
pub struct AdaptiveMomentumOscillatorStreamPy {
    inner: AdaptiveMomentumOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdaptiveMomentumOscillatorStreamPy {
    #[new]
    pub fn new(length: Option<usize>, smoothing_length: Option<usize>) -> PyResult<Self> {
        let inner = AdaptiveMomentumOscillatorStream::try_new(AdaptiveMomentumOscillatorParams {
            length,
            smoothing_length,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
pub fn register_adaptive_momentum_oscillator_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adaptive_momentum_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_momentum_oscillator_batch_py, m)?)?;
    m.add_class::<AdaptiveMomentumOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct AdaptiveMomentumOscillatorJsOutput {
    amo: Vec<f64>,
    ama: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct AdaptiveMomentumOscillatorStreamJsOutput {
    amo: f64,
    ama: f64,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveMomentumOscillatorBatchConfig {
    pub length_range: (usize, usize, usize),
    pub smoothing_length_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveMomentumOscillatorBatchJsOutput {
    pub amo: Vec<f64>,
    pub ama: Vec<f64>,
    pub combos: Vec<AdaptiveMomentumOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_momentum_oscillator_js)]
pub fn adaptive_momentum_oscillator_js(
    data: &[f64],
    length: Option<usize>,
    smoothing_length: Option<usize>,
) -> Result<JsValue, JsValue> {
    let out = adaptive_momentum_oscillator(&AdaptiveMomentumOscillatorInput::from_slice(
        data,
        AdaptiveMomentumOscillatorParams {
            length,
            smoothing_length,
        },
    ))
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&AdaptiveMomentumOscillatorJsOutput {
        amo: out.amo,
        ama: out.ama,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_momentum_oscillator_batch)]
pub fn adaptive_momentum_oscillator_batch_unified_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AdaptiveMomentumOscillatorBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = adaptive_momentum_oscillator_batch_with_kernel(
        data,
        &AdaptiveMomentumOscillatorBatchRange {
            length: config.length_range,
            smoothing_length: config.smoothing_length_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&AdaptiveMomentumOscillatorBatchJsOutput {
        amo: out.amo,
        ama: out.ama,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_momentum_oscillator_alloc)]
pub fn adaptive_momentum_oscillator_alloc(len: usize) -> *mut f64 {
    let mut values = vec![0.0; len];
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_momentum_oscillator_free)]
pub fn adaptive_momentum_oscillator_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_momentum_oscillator_into)]
pub fn adaptive_momentum_oscillator_into_js(
    data_ptr: *const f64,
    amo_ptr: *mut f64,
    ama_ptr: *mut f64,
    len: usize,
    length: Option<usize>,
    smoothing_length: Option<usize>,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || amo_ptr.is_null() || ama_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to adaptive_momentum_oscillator_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = AdaptiveMomentumOscillatorInput::from_slice(
            data,
            AdaptiveMomentumOscillatorParams {
                length,
                smoothing_length,
            },
        );

        let alias_input = data_ptr == amo_ptr as *const f64 || data_ptr == ama_ptr as *const f64;
        let alias_outputs = amo_ptr == ama_ptr;
        if alias_input || alias_outputs {
            let mut amo = vec![0.0; len];
            let mut ama = vec![0.0; len];
            adaptive_momentum_oscillator_into_slice(&mut amo, &mut ama, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(amo_ptr, len).copy_from_slice(&amo);
            std::slice::from_raw_parts_mut(ama_ptr, len).copy_from_slice(&ama);
            return Ok(());
        }

        let amo = std::slice::from_raw_parts_mut(amo_ptr, len);
        let ama = std::slice::from_raw_parts_mut(ama_ptr, len);
        adaptive_momentum_oscillator_into_slice(amo, ama, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_momentum_oscillator_batch_into)]
pub fn adaptive_momentum_oscillator_batch_into(
    data_ptr: *const f64,
    amo_ptr: *mut f64,
    ama_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    smoothing_length_start: usize,
    smoothing_length_end: usize,
    smoothing_length_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || amo_ptr.is_null() || ama_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to adaptive_momentum_oscillator_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = AdaptiveMomentumOscillatorBatchRange {
            length: (length_start, length_end, length_step),
            smoothing_length: (
                smoothing_length_start,
                smoothing_length_end,
                smoothing_length_step,
            ),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

        let alias_input = data_ptr == amo_ptr as *const f64 || data_ptr == ama_ptr as *const f64;
        let alias_outputs = amo_ptr == ama_ptr;
        if alias_input || alias_outputs {
            let mut amo = vec![0.0; total];
            let mut ama = vec![0.0; total];
            let batch_kernel = detect_best_batch_kernel();
            adaptive_momentum_oscillator_batch_inner_into(
                data,
                &sweep,
                batch_kernel,
                !matches!(batch_kernel, Kernel::ScalarBatch),
                &mut amo,
                &mut ama,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(amo_ptr, total).copy_from_slice(&amo);
            std::slice::from_raw_parts_mut(ama_ptr, total).copy_from_slice(&ama);
            return Ok(rows);
        }

        let amo = std::slice::from_raw_parts_mut(amo_ptr, total);
        let ama = std::slice::from_raw_parts_mut(ama_ptr, total);
        let batch_kernel = detect_best_batch_kernel();
        adaptive_momentum_oscillator_batch_inner_into(
            data,
            &sweep,
            batch_kernel,
            !matches!(batch_kernel, Kernel::ScalarBatch),
            amo,
            ama,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct AdaptiveMomentumOscillatorStreamWasm {
    inner: AdaptiveMomentumOscillatorStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl AdaptiveMomentumOscillatorStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        length: Option<usize>,
        smoothing_length: Option<usize>,
    ) -> Result<AdaptiveMomentumOscillatorStreamWasm, JsValue> {
        Ok(Self {
            inner: AdaptiveMomentumOscillatorStream::try_new(AdaptiveMomentumOscillatorParams {
                length,
                smoothing_length,
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        })
    }

    pub fn update(&mut self, value: f64) -> Result<JsValue, JsValue> {
        match self.inner.update(value) {
            Some((amo, ama)) => {
                serde_wasm_bindgen::to_value(&AdaptiveMomentumOscillatorStreamJsOutput { amo, ama })
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Ok(JsValue::NULL),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        (0..256)
            .map(|idx| {
                let x = idx as f64;
                100.0 + (x * 0.07).sin() * 3.0 + (x * 0.023).cos() * 1.4 + x * 0.01
            })
            .collect()
    }

    fn naive_adaptive_momentum_oscillator(
        data: &[f64],
        length: usize,
        smoothing_length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut amo = vec![f64::NAN; data.len()];
        let mut ama = vec![f64::NAN; data.len()];

        let mut raw = vec![f64::NAN; data.len()];
        for idx in 0..data.len() {
            let value = data[idx];
            if !value.is_finite() || idx < length {
                continue;
            }
            let mut best_abs = -1.0;
            let mut best_delta = f64::NAN;
            let mut valid = true;
            for lag in 1..=length {
                let past = data[idx - lag];
                if !past.is_finite() {
                    valid = false;
                    break;
                }
                let delta = value - past;
                let abs_delta = delta.abs();
                if abs_delta >= best_abs {
                    best_abs = abs_delta;
                    best_delta = delta;
                }
            }
            if valid {
                raw[idx] = best_delta;
            }
        }

        if smoothing_length > 0 {
            let pf = smoothing_length as f64;
            let x_sum = (pf * (pf + 1.0)) * 0.5;
            let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
            let denom = pf * x2_sum - x_sum * x_sum;
            for idx in (smoothing_length - 1)..data.len() {
                let mut window_valid = true;
                let mut y_sum = 0.0;
                let mut xy_sum = 0.0;
                for offset in 0..smoothing_length {
                    let y = raw[idx + 1 - smoothing_length + offset];
                    if !y.is_finite() {
                        window_valid = false;
                        break;
                    }
                    let x = (offset + 1) as f64;
                    y_sum += y;
                    xy_sum += y * x;
                }
                if !window_valid {
                    continue;
                }
                let b = (pf * xy_sum - x_sum * y_sum) / denom;
                let a = (y_sum - b * x_sum) / pf;
                amo[idx] = a + b * pf;
            }
        }

        let mut ama_state = 0.0;
        let mut prev = f64::NAN;
        let mut have_prev = false;
        let mut change_ring = vec![0.0; length];
        let mut head = 0usize;
        let mut count = 0usize;
        let mut change_sum = 0.0;

        for idx in 0..data.len() {
            let current = amo[idx];
            let change = if have_prev && current.is_finite() && prev.is_finite() {
                (current - prev).abs()
            } else {
                0.0
            };

            if count < length {
                change_ring[head] = change;
                change_sum += change;
                count += 1;
            } else {
                let old = change_ring[head];
                change_ring[head] = change;
                change_sum += change - old;
            }
            head = (head + 1) % length;

            if current.is_finite() {
                if change_sum > 0.0 {
                    let efficiency_ratio = current.abs() / change_sum;
                    let delta = efficiency_ratio * (current - ama_state);
                    if delta.is_finite() {
                        ama_state += delta;
                    }
                }
                ama[idx] = ama_state;
            }

            prev = current;
            have_prev = true;
        }

        (amo, ama)
    }

    #[test]
    fn adaptive_momentum_oscillator_into_matches_api() -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let input = AdaptiveMomentumOscillatorInput::from_slice(
            &data,
            AdaptiveMomentumOscillatorParams::default(),
        );
        let out = adaptive_momentum_oscillator(&input)?;

        let mut amo = vec![0.0; data.len()];
        let mut ama = vec![0.0; data.len()];
        adaptive_momentum_oscillator_into(&input, &mut amo, &mut ama)?;

        for idx in 0..data.len() {
            let a = out.amo[idx];
            let b = amo[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);

            let a = out.ama[idx];
            let b = ama[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn adaptive_momentum_oscillator_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let batch = adaptive_momentum_oscillator(&AdaptiveMomentumOscillatorInput::from_slice(
            &data,
            AdaptiveMomentumOscillatorParams::default(),
        ))?;

        let mut stream =
            AdaptiveMomentumOscillatorStream::try_new(AdaptiveMomentumOscillatorParams::default())?;
        let mut amo = vec![f64::NAN; data.len()];
        let mut ama = vec![f64::NAN; data.len()];
        for (idx, value) in data.iter().copied().enumerate() {
            if let Some((amo_value, ama_value)) = stream.update(value) {
                amo[idx] = amo_value;
                ama[idx] = ama_value;
            }
        }

        for idx in 0..data.len() {
            let a = batch.amo[idx];
            let b = amo[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);

            let a = batch.ama[idx];
            let b = ama[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn adaptive_momentum_oscillator_matches_naive_reference() -> Result<(), Box<dyn StdError>> {
        let data = vec![
            10.0, 10.4, 10.2, 10.9, 11.3, 11.0, 11.8, 11.5, 12.0, 11.7, 12.4, 12.1,
        ];
        let input = AdaptiveMomentumOscillatorInput::from_slice(
            &data,
            AdaptiveMomentumOscillatorParams {
                length: Some(3),
                smoothing_length: Some(2),
            },
        );
        let out = adaptive_momentum_oscillator(&input)?;
        let (expected_amo, expected_ama) = naive_adaptive_momentum_oscillator(&data, 3, 2);

        for idx in 0..data.len() {
            let a = out.amo[idx];
            let b = expected_amo[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);

            let a = out.ama[idx];
            let b = expected_ama[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9);
        }
        Ok(())
    }

    #[test]
    fn adaptive_momentum_oscillator_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let single = adaptive_momentum_oscillator(&AdaptiveMomentumOscillatorInput::from_slice(
            &data,
            AdaptiveMomentumOscillatorParams {
                length: Some(14),
                smoothing_length: Some(9),
            },
        ))?;
        let batch = adaptive_momentum_oscillator_batch_with_kernel(
            &data,
            &AdaptiveMomentumOscillatorBatchRange {
                length: (14, 14, 0),
                smoothing_length: (9, 9, 0),
            },
            Kernel::Auto,
        )?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        let amo = &batch.amo[0..data.len()];
        let ama = &batch.ama[0..data.len()];
        for idx in 0..data.len() {
            let a = single.amo[idx];
            let b = amo[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);

            let a = single.ama[idx];
            let b = ama[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
        }
        Ok(())
    }
}
