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

const DEFAULT_LENGTH: usize = 14;
const DEFAULT_MULT: f64 = 2.0;
const DEFAULT_SOURCE: &str = "close";
const FLOAT_TOL: f64 = 1e-12;

impl<'a> AsRef<[f64]> for HullButterflyOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HullButterflyOscillatorData::Slice(slice) => slice,
            HullButterflyOscillatorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum HullButterflyOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HullButterflyOscillatorOutput {
    pub oscillator: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullButterflyOscillatorOutputField {
    Oscillator,
    CumulativeMean,
    Signal,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HullButterflyOscillatorParams {
    pub length: Option<usize>,
    pub mult: Option<f64>,
}

impl Default for HullButterflyOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            mult: Some(DEFAULT_MULT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HullButterflyOscillatorInput<'a> {
    pub data: HullButterflyOscillatorData<'a>,
    pub params: HullButterflyOscillatorParams,
}

impl<'a> HullButterflyOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: HullButterflyOscillatorParams,
    ) -> Self {
        Self {
            data: HullButterflyOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: HullButterflyOscillatorParams) -> Self {
        Self {
            data: HullButterflyOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            HullButterflyOscillatorParams::default(),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HullButterflyOscillatorBuilder {
    length: Option<usize>,
    mult: Option<f64>,
    kernel: Kernel,
}

impl Default for HullButterflyOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            mult: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HullButterflyOscillatorBuilder {
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
    pub fn mult(mut self, mult: f64) -> Self {
        self.mult = Some(mult);
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
    ) -> Result<HullButterflyOscillatorOutput, HullButterflyOscillatorError> {
        let input = HullButterflyOscillatorInput::from_candles(
            candles,
            source,
            HullButterflyOscillatorParams {
                length: self.length,
                mult: self.mult,
            },
        );
        hull_butterfly_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HullButterflyOscillatorOutput, HullButterflyOscillatorError> {
        let input = HullButterflyOscillatorInput::from_slice(
            data,
            HullButterflyOscillatorParams {
                length: self.length,
                mult: self.mult,
            },
        );
        hull_butterfly_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<HullButterflyOscillatorStream, HullButterflyOscillatorError> {
        HullButterflyOscillatorStream::try_new(HullButterflyOscillatorParams {
            length: self.length,
            mult: self.mult,
        })
    }
}

#[derive(Debug, Error)]
pub enum HullButterflyOscillatorError {
    #[error("hull_butterfly_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("hull_butterfly_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "hull_butterfly_oscillator: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("hull_butterfly_oscillator: Invalid multiplier: {mult}")]
    InvalidMultiplier { mult: f64 },
    #[error(
        "hull_butterfly_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "hull_butterfly_oscillator: Output length mismatch: expected = {expected}, oscillator = {oscillator_got}, cumulative_mean = {cumulative_mean_got}, signal = {signal_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        oscillator_got: usize,
        cumulative_mean_got: usize,
        signal_got: usize,
    },
    #[error("hull_butterfly_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("hull_butterfly_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct ResolvedParams {
    mult: f64,
    coeffs: Vec<f64>,
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
fn compute_hull_coeffs(length: usize) -> Vec<f64> {
    let short_len = length / 2;
    let hull_len = ((length as f64).sqrt().floor() as usize).max(1);
    let den1 = (short_len * (short_len + 1) / 2) as f64;
    let den2 = (length * (length + 1) / 2) as f64;
    let den3 = (hull_len * (hull_len + 1) / 2) as f64;

    let mut lcwa_coeffs = vec![0.0; hull_len];
    for i in 0..length {
        let sum1 = short_len.saturating_sub(i) as f64;
        let sum2 = (length - i) as f64;
        lcwa_coeffs.insert(0, 2.0 * (sum1 / den1) - (sum2 / den2));
    }
    for _ in 0..hull_len.saturating_sub(1) {
        lcwa_coeffs.insert(0, 0.0);
    }

    let size = lcwa_coeffs.len();
    let mut hull_coeffs = Vec::with_capacity(size.saturating_sub(hull_len));
    for i in hull_len..size {
        let mut sum3 = 0.0;
        for j in (i - hull_len)..i {
            sum3 += lcwa_coeffs[j] * (i - j) as f64;
        }
        hull_coeffs.insert(0, sum3 / den3);
    }
    hull_coeffs
}

#[inline(always)]
fn warmup_period(params: &ResolvedParams) -> usize {
    params.coeffs.len().saturating_sub(1)
}

#[inline]
fn resolve_params(
    params: &HullButterflyOscillatorParams,
    data_len: Option<usize>,
) -> Result<ResolvedParams, HullButterflyOscillatorError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    let mult = params.mult.unwrap_or(DEFAULT_MULT);

    if length < 2 {
        return Err(HullButterflyOscillatorError::InvalidLength {
            length,
            data_len: data_len.unwrap_or(0),
        });
    }
    if !mult.is_finite() {
        return Err(HullButterflyOscillatorError::InvalidMultiplier { mult });
    }
    if let Some(data_len) = data_len {
        if length > data_len {
            return Err(HullButterflyOscillatorError::InvalidLength { length, data_len });
        }
    }

    Ok(ResolvedParams {
        mult,
        coeffs: compute_hull_coeffs(length),
    })
}

#[inline(always)]
fn crossed(prev_a: f64, curr_a: f64, prev_b: f64, curr_b: f64) -> bool {
    (curr_a > curr_b && prev_a <= prev_b) || (curr_a < curr_b && prev_a >= prev_b)
}

#[derive(Debug, Clone)]
pub struct HullButterflyOscillatorStream {
    params: ResolvedParams,
    ring: Vec<f64>,
    head: usize,
    count: usize,
    cumulative_abs: f64,
    segment_index: usize,
    prev_hso: Option<f64>,
    prev_cmean: Option<f64>,
    signal_state: f64,
}

impl HullButterflyOscillatorStream {
    pub fn try_new(
        params: HullButterflyOscillatorParams,
    ) -> Result<Self, HullButterflyOscillatorError> {
        let params = resolve_params(&params, None)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        let coeff_len = params.coeffs.len().max(1);
        Self {
            params,
            ring: vec![0.0; coeff_len],
            head: 0,
            count: 0,
            cumulative_abs: 0.0,
            segment_index: 0,
            prev_hso: None,
            prev_cmean: None,
            signal_state: 0.0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.cumulative_abs = 0.0;
        self.segment_index = 0;
        self.prev_hso = None;
        self.prev_cmean = None;
        self.signal_state = 0.0;
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        warmup_period(&self.params)
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let coeff_len = self.params.coeffs.len();
        self.ring[self.head] = value;
        self.head += 1;
        if self.head == coeff_len {
            self.head = 0;
        }
        if self.count < coeff_len {
            self.count += 1;
        }
        let current_index = self.segment_index;
        self.segment_index = self.segment_index.saturating_add(1);

        if self.count < coeff_len {
            return None;
        }

        let mut hma = 0.0;
        let mut inv_hma = 0.0;
        for (i, &coeff) in self.params.coeffs.iter().enumerate() {
            let recent_idx = (self.head + coeff_len - 1 - i) % coeff_len;
            let inverse_idx = (self.head + i) % coeff_len;
            hma += self.ring[recent_idx] * coeff;
            inv_hma += self.ring[inverse_idx] * coeff;
        }

        let hso = hma - inv_hma;
        self.cumulative_abs += hso.abs();
        if current_index == 0 {
            return None;
        }
        let cmean = self.cumulative_abs / current_index as f64 * self.params.mult;

        if let (Some(prev_hso), Some(prev_cmean)) = (self.prev_hso, self.prev_cmean) {
            if crossed(prev_hso, hso, prev_cmean, cmean)
                || crossed(prev_hso, hso, -prev_cmean, -cmean)
            {
                self.signal_state = 0.0;
            } else if hso < prev_hso && hso > cmean {
                self.signal_state = -1.0;
            } else if hso > prev_hso && hso < -cmean {
                self.signal_state = 1.0;
            }
        }

        self.prev_hso = Some(hso);
        self.prev_cmean = Some(cmean);
        Some((hso, cmean, self.signal_state))
    }
}

#[inline(always)]
fn hull_butterfly_oscillator_prepare<'a>(
    input: &'a HullButterflyOscillatorInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, ResolvedParams, Kernel), HullButterflyOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(HullButterflyOscillatorError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(HullButterflyOscillatorError::AllValuesNaN);
    }

    let params = resolve_params(&input.params, Some(data.len()))?;
    let needed = warmup_period(&params) + 1;
    let valid = max_consecutive_valid_values(data);
    if valid < needed {
        return Err(HullButterflyOscillatorError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, first, params, chosen))
}

#[inline(always)]
fn hull_butterfly_oscillator_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    oscillator_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    signal_out: &mut [f64],
) {
    oscillator_out.fill(f64::NAN);
    cumulative_mean_out.fill(f64::NAN);
    signal_out.fill(f64::NAN);

    let mut stream = HullButterflyOscillatorStream::new_resolved(params);
    for (((osc_slot, mean_slot), sig_slot), &value) in oscillator_out
        .iter_mut()
        .zip(cumulative_mean_out.iter_mut())
        .zip(signal_out.iter_mut())
        .zip(data.iter())
    {
        if let Some((oscillator, cumulative_mean, signal)) = stream.update(value) {
            *osc_slot = oscillator;
            *mean_slot = cumulative_mean;
            *sig_slot = signal;
        }
    }
}

#[inline(always)]
fn hull_butterfly_oscillator_selected_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    field: HullButterflyOscillatorOutputField,
    out: &mut [f64],
) {
    out.fill(f64::NAN);

    let mut stream = HullButterflyOscillatorStream::new_resolved(params);
    for (slot, &value) in out.iter_mut().zip(data.iter()) {
        if let Some((oscillator, cumulative_mean, signal)) = stream.update(value) {
            *slot = match field {
                HullButterflyOscillatorOutputField::Oscillator => oscillator,
                HullButterflyOscillatorOutputField::CumulativeMean => cumulative_mean,
                HullButterflyOscillatorOutputField::Signal => signal,
            };
        }
    }
}

#[inline]
pub fn hull_butterfly_oscillator(
    input: &HullButterflyOscillatorInput,
) -> Result<HullButterflyOscillatorOutput, HullButterflyOscillatorError> {
    hull_butterfly_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn hull_butterfly_oscillator_with_kernel(
    input: &HullButterflyOscillatorInput,
    kernel: Kernel,
) -> Result<HullButterflyOscillatorOutput, HullButterflyOscillatorError> {
    let (data, first, params, _chosen) = hull_butterfly_oscillator_prepare(input, kernel)?;
    let warmup = first.saturating_add(warmup_period(&params)).min(data.len());
    let mut oscillator = alloc_with_nan_prefix(data.len(), warmup);
    let mut cumulative_mean = alloc_with_nan_prefix(data.len(), warmup);
    let mut signal = alloc_with_nan_prefix(data.len(), warmup);
    hull_butterfly_oscillator_row_from_slice(
        data,
        params,
        &mut oscillator,
        &mut cumulative_mean,
        &mut signal,
    );
    Ok(HullButterflyOscillatorOutput {
        oscillator,
        cumulative_mean,
        signal,
    })
}

#[inline]
pub fn hull_butterfly_oscillator_into_slices(
    oscillator_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    signal_out: &mut [f64],
    input: &HullButterflyOscillatorInput,
    kernel: Kernel,
) -> Result<(), HullButterflyOscillatorError> {
    let expected = input.as_ref().len();
    if oscillator_out.len() != expected
        || cumulative_mean_out.len() != expected
        || signal_out.len() != expected
    {
        return Err(HullButterflyOscillatorError::OutputLengthMismatch {
            expected,
            oscillator_got: oscillator_out.len(),
            cumulative_mean_got: cumulative_mean_out.len(),
            signal_got: signal_out.len(),
        });
    }
    let (data, _first, params, _chosen) = hull_butterfly_oscillator_prepare(input, kernel)?;
    hull_butterfly_oscillator_row_from_slice(
        data,
        params,
        oscillator_out,
        cumulative_mean_out,
        signal_out,
    );
    Ok(())
}

#[inline]
pub fn hull_butterfly_oscillator_output_into_slice(
    out: &mut [f64],
    input: &HullButterflyOscillatorInput,
    kernel: Kernel,
    field: HullButterflyOscillatorOutputField,
) -> Result<(), HullButterflyOscillatorError> {
    let expected = input.as_ref().len();
    if out.len() != expected {
        return Err(HullButterflyOscillatorError::OutputLengthMismatch {
            expected,
            oscillator_got: out.len(),
            cumulative_mean_got: out.len(),
            signal_got: out.len(),
        });
    }
    let (data, _first, params, _chosen) = hull_butterfly_oscillator_prepare(input, kernel)?;
    hull_butterfly_oscillator_selected_row_from_slice(data, params, field, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn hull_butterfly_oscillator_into(
    input: &HullButterflyOscillatorInput,
    oscillator_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<(), HullButterflyOscillatorError> {
    hull_butterfly_oscillator_into_slices(
        oscillator_out,
        cumulative_mean_out,
        signal_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HullButterflyOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub mult: (f64, f64, f64),
}

impl Default for HullButterflyOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            mult: (DEFAULT_MULT, DEFAULT_MULT, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HullButterflyOscillatorBatchOutput {
    pub oscillator: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<HullButterflyOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HullButterflyOscillatorBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &HullButterflyOscillatorParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == params.length.unwrap_or(DEFAULT_LENGTH)
                && (combo.mult.unwrap_or(DEFAULT_MULT) - params.mult.unwrap_or(DEFAULT_MULT)).abs()
                    < FLOAT_TOL
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
            &self.oscillator[start..end],
            &self.cumulative_mean[start..end],
            &self.signal[start..end],
        ))
    }
}

#[derive(Clone, Debug, Default)]
pub struct HullButterflyOscillatorBatchBuilder {
    range: HullButterflyOscillatorBatchRange,
    kernel: Kernel,
}

impl HullButterflyOscillatorBatchBuilder {
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
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HullButterflyOscillatorBatchOutput, HullButterflyOscillatorError> {
        hull_butterfly_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<HullButterflyOscillatorBatchOutput, HullButterflyOscillatorError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, HullButterflyOscillatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            let next = value.saturating_add(step);
            if next == value {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }

    if out.is_empty() {
        return Err(HullButterflyOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, HullButterflyOscillatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(HullButterflyOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(HullButterflyOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(HullButterflyOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }

    let mut values = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        values.push(value.min(end));
        value += step;
    }
    if (values.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
        return Err(HullButterflyOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(values)
}

#[inline(always)]
fn expand_grid_hull_butterfly_oscillator(
    sweep: &HullButterflyOscillatorBatchRange,
) -> Result<Vec<HullButterflyOscillatorParams>, HullButterflyOscillatorError> {
    let lengths = expand_axis_usize(sweep.length)?;
    let mults = expand_axis_f64(sweep.mult.0, sweep.mult.1, sweep.mult.2)?;
    let mut combos = Vec::with_capacity(lengths.len() * mults.len());
    for length in lengths {
        for &mult in &mults {
            let combo = HullButterflyOscillatorParams {
                length: Some(length),
                mult: Some(mult),
            };
            let _ = resolve_params(&combo, None)?;
            combos.push(combo);
        }
    }
    Ok(combos)
}

#[inline]
pub fn hull_butterfly_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &HullButterflyOscillatorBatchRange,
    kernel: Kernel,
) -> Result<HullButterflyOscillatorBatchOutput, HullButterflyOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HullButterflyOscillatorError::InvalidKernelForBatch(other)),
    };
    hull_butterfly_oscillator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn hull_butterfly_oscillator_batch_slice(
    data: &[f64],
    sweep: &HullButterflyOscillatorBatchRange,
    kernel: Kernel,
) -> Result<HullButterflyOscillatorBatchOutput, HullButterflyOscillatorError> {
    hull_butterfly_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn hull_butterfly_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &HullButterflyOscillatorBatchRange,
    kernel: Kernel,
) -> Result<HullButterflyOscillatorBatchOutput, HullButterflyOscillatorError> {
    hull_butterfly_oscillator_batch_inner(data, sweep, kernel, true)
}

pub fn hull_butterfly_oscillator_batch_inner(
    data: &[f64],
    sweep: &HullButterflyOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<HullButterflyOscillatorBatchOutput, HullButterflyOscillatorError> {
    if data.is_empty() {
        return Err(HullButterflyOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(HullButterflyOscillatorError::AllValuesNaN);
    }

    let combos = expand_grid_hull_butterfly_oscillator(sweep)?;
    let resolved = combos
        .iter()
        .map(|params| resolve_params(params, Some(data.len())))
        .collect::<Result<Vec<_>, _>>()?;
    let max_valid = max_consecutive_valid_values(data);
    for params in &resolved {
        let needed = warmup_period(params) + 1;
        if max_valid < needed {
            return Err(HullButterflyOscillatorError::NotEnoughValidData {
                needed,
                valid: max_valid,
            });
        }
    }

    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(HullButterflyOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                oscillator_got: 0,
                cumulative_mean_got: 0,
                signal_got: 0,
            })?;

    let warmups = resolved
        .iter()
        .map(|params| first.saturating_add(warmup_period(params)).min(cols))
        .collect::<Vec<_>>();

    let mut oscillator_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut oscillator_mu, cols, &warmups);
    let mut oscillator_guard = ManuallyDrop::new(oscillator_mu);
    let oscillator_out =
        unsafe { std::slice::from_raw_parts_mut(oscillator_guard.as_mut_ptr() as *mut f64, total) };

    let mut cumulative_mean_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut cumulative_mean_mu, cols, &warmups);
    let mut cumulative_mean_guard = ManuallyDrop::new(cumulative_mean_mu);
    let cumulative_mean_out = unsafe {
        std::slice::from_raw_parts_mut(cumulative_mean_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut signal_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut signal_mu, cols, &warmups);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let signal_out =
        unsafe { std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            oscillator_out
                .par_chunks_mut(cols)
                .zip(cumulative_mean_out.par_chunks_mut(cols))
                .zip(signal_out.par_chunks_mut(cols))
                .zip(resolved.par_iter())
                .for_each(|(((osc_row, mean_row), sig_row), params)| {
                    hull_butterfly_oscillator_row_from_slice(
                        data,
                        params.clone(),
                        osc_row,
                        mean_row,
                        sig_row,
                    );
                });
        }

        #[cfg(target_arch = "wasm32")]
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            hull_butterfly_oscillator_row_from_slice(
                data,
                params.clone(),
                &mut oscillator_out[start..end],
                &mut cumulative_mean_out[start..end],
                &mut signal_out[start..end],
            );
        }
    } else {
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            hull_butterfly_oscillator_row_from_slice(
                data,
                params.clone(),
                &mut oscillator_out[start..end],
                &mut cumulative_mean_out[start..end],
                &mut signal_out[start..end],
            );
        }
    }

    let oscillator = unsafe {
        Vec::from_raw_parts(
            oscillator_guard.as_mut_ptr() as *mut f64,
            oscillator_guard.len(),
            oscillator_guard.capacity(),
        )
    };
    let cumulative_mean = unsafe {
        Vec::from_raw_parts(
            cumulative_mean_guard.as_mut_ptr() as *mut f64,
            cumulative_mean_guard.len(),
            cumulative_mean_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };
    core::mem::forget(oscillator_guard);
    core::mem::forget(cumulative_mean_guard);
    core::mem::forget(signal_guard);

    Ok(HullButterflyOscillatorBatchOutput {
        oscillator,
        cumulative_mean,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn hull_butterfly_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &HullButterflyOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    oscillator_out: &mut [f64],
    cumulative_mean_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<HullButterflyOscillatorParams>, HullButterflyOscillatorError> {
    let out = hull_butterfly_oscillator_batch_inner(data, sweep, kernel, parallel)?;
    let total = out.rows * out.cols;
    if oscillator_out.len() != total
        || cumulative_mean_out.len() != total
        || signal_out.len() != total
    {
        return Err(HullButterflyOscillatorError::OutputLengthMismatch {
            expected: total,
            oscillator_got: oscillator_out.len(),
            cumulative_mean_got: cumulative_mean_out.len(),
            signal_got: signal_out.len(),
        });
    }
    oscillator_out.copy_from_slice(&out.oscillator);
    cumulative_mean_out.copy_from_slice(&out.cumulative_mean);
    signal_out.copy_from_slice(&out.signal);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "hull_butterfly_oscillator")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, mult=DEFAULT_MULT, kernel=None))]
pub fn hull_butterfly_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    mult: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = HullButterflyOscillatorInput::from_slice(
        data,
        HullButterflyOscillatorParams {
            length: Some(length),
            mult: Some(mult),
        },
    );
    let out = py
        .allow_threads(|| hull_butterfly_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.oscillator.into_pyarray(py),
        out.cumulative_mean.into_pyarray(py),
        out.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "HullButterflyOscillatorStream")]
pub struct HullButterflyOscillatorStreamPy {
    stream: HullButterflyOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HullButterflyOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, mult=DEFAULT_MULT))]
    fn new(length: usize, mult: f64) -> PyResult<Self> {
        let stream = HullButterflyOscillatorStream::try_new(HullButterflyOscillatorParams {
            length: Some(length),
            mult: Some(mult),
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
#[pyfunction(name = "hull_butterfly_oscillator_batch")]
#[pyo3(signature = (
    data,
    length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
    mult_range=(DEFAULT_MULT, DEFAULT_MULT, 0.0),
    kernel=None
))]
pub fn hull_butterfly_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = HullButterflyOscillatorBatchRange {
        length: length_range,
        mult: mult_range,
    };
    let combos = expand_grid_hull_butterfly_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let oscillator_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let cumulative_mean_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let oscillator_slice = unsafe { oscillator_arr.as_slice_mut()? };
    let cumulative_mean_slice = unsafe { cumulative_mean_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            hull_butterfly_oscillator_batch_inner_into(
                data,
                &sweep,
                batch_kernel.to_non_batch(),
                true,
                oscillator_slice,
                cumulative_mean_slice,
                signal_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("oscillator", oscillator_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "cumulative_mean",
        cumulative_mean_arr.reshape((rows, cols))?,
    )?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        combos
            .iter()
            .map(|combo| combo.mult.unwrap_or(DEFAULT_MULT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_hull_butterfly_oscillator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hull_butterfly_oscillator_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        hull_butterfly_oscillator_batch_py,
        module
    )?)?;
    module.add_class::<HullButterflyOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HullButterflyOscillatorJsOutput {
    pub oscillator: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "hull_butterfly_oscillator_js")]
pub fn hull_butterfly_oscillator_js(
    data: &[f64],
    length: usize,
    mult: f64,
) -> Result<JsValue, JsValue> {
    let input = HullButterflyOscillatorInput::from_slice(
        data,
        HullButterflyOscillatorParams {
            length: Some(length),
            mult: Some(mult),
        },
    );
    let output =
        hull_butterfly_oscillator(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&HullButterflyOscillatorJsOutput {
        oscillator: output.oscillator,
        cumulative_mean: output.cumulative_mean,
        signal: output.signal,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hull_butterfly_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hull_butterfly_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hull_butterfly_oscillator_into(
    in_ptr: *const f64,
    oscillator_out_ptr: *mut f64,
    cumulative_mean_out_ptr: *mut f64,
    signal_out_ptr: *mut f64,
    len: usize,
    length: usize,
    mult: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || oscillator_out_ptr.is_null()
        || cumulative_mean_out_ptr.is_null()
        || signal_out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = HullButterflyOscillatorInput::from_slice(
            data,
            HullButterflyOscillatorParams {
                length: Some(length),
                mult: Some(mult),
            },
        );
        let need_temp = in_ptr == oscillator_out_ptr as *const f64
            || in_ptr == cumulative_mean_out_ptr as *const f64
            || in_ptr == signal_out_ptr as *const f64
            || oscillator_out_ptr == cumulative_mean_out_ptr
            || oscillator_out_ptr == signal_out_ptr
            || cumulative_mean_out_ptr == signal_out_ptr;

        if need_temp {
            let mut oscillator = vec![0.0; len];
            let mut cumulative_mean = vec![0.0; len];
            let mut signal = vec![0.0; len];
            hull_butterfly_oscillator_into_slices(
                &mut oscillator,
                &mut cumulative_mean,
                &mut signal,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(oscillator_out_ptr, len).copy_from_slice(&oscillator);
            std::slice::from_raw_parts_mut(cumulative_mean_out_ptr, len)
                .copy_from_slice(&cumulative_mean);
            std::slice::from_raw_parts_mut(signal_out_ptr, len).copy_from_slice(&signal);
        } else {
            hull_butterfly_oscillator_into_slices(
                std::slice::from_raw_parts_mut(oscillator_out_ptr, len),
                std::slice::from_raw_parts_mut(cumulative_mean_out_ptr, len),
                std::slice::from_raw_parts_mut(signal_out_ptr, len),
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HullButterflyOscillatorBatchJsConfig {
    pub length_range: Option<(usize, usize, usize)>,
    pub mult_range: Option<(f64, f64, f64)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HullButterflyOscillatorBatchJsOutput {
    pub oscillator: Vec<f64>,
    pub cumulative_mean: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<HullButterflyOscillatorParams>,
    pub lengths: Vec<usize>,
    pub multipliers: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "hull_butterfly_oscillator_batch_js")]
pub fn hull_butterfly_oscillator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: HullButterflyOscillatorBatchJsConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = HullButterflyOscillatorBatchRange {
        length: config
            .length_range
            .unwrap_or((DEFAULT_LENGTH, DEFAULT_LENGTH, 0)),
        mult: config
            .mult_range
            .unwrap_or((DEFAULT_MULT, DEFAULT_MULT, 0.0)),
    };
    let output = hull_butterfly_oscillator_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&HullButterflyOscillatorBatchJsOutput {
        lengths: output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        multipliers: output
            .combos
            .iter()
            .map(|combo| combo.mult.unwrap_or(DEFAULT_MULT))
            .collect(),
        oscillator: output.oscillator,
        cumulative_mean: output.cumulative_mean,
        signal: output.signal,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hull_butterfly_oscillator_batch_into(
    in_ptr: *const f64,
    oscillator_out_ptr: *mut f64,
    cumulative_mean_out_ptr: *mut f64,
    signal_out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null()
        || oscillator_out_ptr.is_null()
        || cumulative_mean_out_ptr.is_null()
        || signal_out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = HullButterflyOscillatorBatchRange {
        length: (length_start, length_end, length_step),
        mult: (mult_start, mult_end, mult_step),
    };

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let combos = expand_grid_hull_butterfly_oscillator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let need_temp = in_ptr == oscillator_out_ptr as *const f64
            || in_ptr == cumulative_mean_out_ptr as *const f64
            || in_ptr == signal_out_ptr as *const f64
            || oscillator_out_ptr == cumulative_mean_out_ptr
            || oscillator_out_ptr == signal_out_ptr
            || cumulative_mean_out_ptr == signal_out_ptr;

        if need_temp {
            let mut oscillator = vec![0.0; total];
            let mut cumulative_mean = vec![0.0; total];
            let mut signal = vec![0.0; total];
            let rows = hull_butterfly_oscillator_batch_inner_into(
                data,
                &sweep,
                Kernel::Auto,
                false,
                &mut oscillator,
                &mut cumulative_mean,
                &mut signal,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            std::slice::from_raw_parts_mut(oscillator_out_ptr, total).copy_from_slice(&oscillator);
            std::slice::from_raw_parts_mut(cumulative_mean_out_ptr, total)
                .copy_from_slice(&cumulative_mean);
            std::slice::from_raw_parts_mut(signal_out_ptr, total).copy_from_slice(&signal);
            Ok(rows)
        } else {
            let rows = hull_butterfly_oscillator_batch_inner_into(
                data,
                &sweep,
                Kernel::Auto,
                false,
                std::slice::from_raw_parts_mut(oscillator_out_ptr, total),
                std::slice::from_raw_parts_mut(cumulative_mean_out_ptr, total),
                std::slice::from_raw_parts_mut(signal_out_ptr, total),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            Ok(rows)
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hull_butterfly_oscillator_output_into_js(
    data: &[f64],
    length: usize,
    mult: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = hull_butterfly_oscillator_js(data, length, mult)?;
    crate::write_wasm_object_f64_outputs("hull_butterfly_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hull_butterfly_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = hull_butterfly_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "hull_butterfly_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::Candles;

    fn sample_source(length: usize) -> Vec<f64> {
        (0..length)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.05 + (x * 0.17).sin() * 2.4 + (x * 0.03).cos() * 0.7
            })
            .collect()
    }

    fn sample_candles(length: usize) -> Candles {
        let close = sample_source(length);
        let open = close.iter().map(|v| v - 0.3).collect::<Vec<_>>();
        let high = close.iter().map(|v| v + 0.8).collect::<Vec<_>>();
        let low = close.iter().map(|v| v - 0.9).collect::<Vec<_>>();
        let volume = vec![1_000.0; length];
        Candles::new((0..length as i64).collect(), open, high, low, close, volume)
    }

    fn assert_series_eq(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&lhs, &rhs) in left.iter().zip(right.iter()) {
            assert!(
                (lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() <= tol,
                "series mismatch: left={lhs:?}, right={rhs:?}"
            );
        }
    }

    #[test]
    fn hull_butterfly_oscillator_output_contract() {
        let close = sample_source(320);
        let input = HullButterflyOscillatorInput::from_slice(
            &close,
            HullButterflyOscillatorParams::default(),
        );
        let out = hull_butterfly_oscillator(&input).unwrap();
        assert_eq!(out.oscillator.len(), close.len());
        assert_eq!(out.cumulative_mean.len(), close.len());
        assert_eq!(out.signal.len(), close.len());
        let first_finite = out
            .oscillator
            .iter()
            .position(|value| value.is_finite())
            .unwrap();
        assert_eq!(first_finite, 15);
        assert!(out.cumulative_mean[first_finite].is_finite());
        assert!(out.signal[first_finite].is_finite());
        assert!(out.signal.iter().filter(|v| v.is_finite()).all(|v| {
            (*v - 1.0).abs() <= FLOAT_TOL || v.abs() <= FLOAT_TOL || (*v + 1.0).abs() <= FLOAT_TOL
        }));
    }

    #[test]
    fn hull_butterfly_oscillator_rejects_invalid_parameters() {
        let data = sample_source(32);
        let err = hull_butterfly_oscillator(&HullButterflyOscillatorInput::from_slice(
            &data,
            HullButterflyOscillatorParams {
                length: Some(1),
                mult: Some(2.0),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            HullButterflyOscillatorError::InvalidLength { .. }
        ));

        let err = hull_butterfly_oscillator(&HullButterflyOscillatorInput::from_slice(
            &data,
            HullButterflyOscillatorParams {
                length: Some(14),
                mult: Some(f64::NAN),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            HullButterflyOscillatorError::InvalidMultiplier { .. }
        ));
    }

    #[test]
    fn hull_butterfly_oscillator_builder_supports_candles() {
        let candles = sample_candles(256);
        let out = HullButterflyOscillatorBuilder::new()
            .length(18)
            .mult(1.5)
            .apply(&candles, "hlc3")
            .unwrap();
        assert_eq!(out.oscillator.len(), candles.close.len());
        assert!(out.oscillator.iter().any(|value| value.is_finite()));
    }

    #[test]
    fn hull_butterfly_oscillator_stream_matches_batch_with_reset() {
        let mut data = sample_source(260);
        data[130] = f64::NAN;
        let input = HullButterflyOscillatorInput::from_slice(
            &data,
            HullButterflyOscillatorParams {
                length: Some(12),
                mult: Some(1.75),
            },
        );
        let batch = hull_butterfly_oscillator(&input).unwrap();
        let mut stream = HullButterflyOscillatorStream::try_new(HullButterflyOscillatorParams {
            length: Some(12),
            mult: Some(1.75),
        })
        .unwrap();
        let mut oscillator = Vec::with_capacity(data.len());
        let mut cumulative_mean = Vec::with_capacity(data.len());
        let mut signal = Vec::with_capacity(data.len());
        for &value in &data {
            if let Some((osc, mean, sig)) = stream.update(value) {
                oscillator.push(osc);
                cumulative_mean.push(mean);
                signal.push(sig);
            } else {
                oscillator.push(f64::NAN);
                cumulative_mean.push(f64::NAN);
                signal.push(f64::NAN);
            }
        }
        assert_eq!(stream.get_warmup_period(), 13);
        assert_series_eq(&oscillator, &batch.oscillator, 1e-12);
        assert_series_eq(&cumulative_mean, &batch.cumulative_mean, 1e-12);
        assert_series_eq(&signal, &batch.signal, 1e-12);
    }

    #[test]
    fn hull_butterfly_oscillator_into_matches_api() {
        let data = sample_source(192);
        let input = HullButterflyOscillatorInput::from_slice(
            &data,
            HullButterflyOscillatorParams {
                length: Some(16),
                mult: Some(2.25),
            },
        );
        let direct = hull_butterfly_oscillator(&input).unwrap();
        let mut oscillator = vec![0.0; data.len()];
        let mut cumulative_mean = vec![0.0; data.len()];
        let mut signal = vec![0.0; data.len()];
        hull_butterfly_oscillator_into(&input, &mut oscillator, &mut cumulative_mean, &mut signal)
            .unwrap();
        assert_series_eq(&oscillator, &direct.oscillator, 1e-12);
        assert_series_eq(&cumulative_mean, &direct.cumulative_mean, 1e-12);
        assert_series_eq(&signal, &direct.signal, 1e-12);
    }

    #[test]
    fn hull_butterfly_oscillator_batch_single_param_matches_single() {
        let data = sample_source(160);
        let sweep = HullButterflyOscillatorBatchRange {
            length: (12, 12, 0),
            mult: (1.5, 1.5, 0.0),
        };
        let batch =
            hull_butterfly_oscillator_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        let single = hull_butterfly_oscillator(&HullButterflyOscillatorInput::from_slice(
            &data,
            HullButterflyOscillatorParams {
                length: Some(12),
                mult: Some(1.5),
            },
        ))
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        let (oscillator, cumulative_mean, signal) = batch.row_slices(0).unwrap();
        assert_series_eq(oscillator, &single.oscillator, 1e-12);
        assert_series_eq(cumulative_mean, &single.cumulative_mean, 1e-12);
        assert_series_eq(signal, &single.signal, 1e-12);
    }

    #[test]
    fn hull_butterfly_oscillator_batch_metadata() {
        let data = sample_source(120);
        let sweep = HullButterflyOscillatorBatchRange {
            length: (10, 14, 2),
            mult: (1.0, 2.0, 0.5),
        };
        let batch =
            hull_butterfly_oscillator_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        assert_eq!(batch.rows, 9);
        assert_eq!(batch.cols, data.len());
        assert_eq!(
            batch
                .combos
                .iter()
                .map(|combo| combo.length.unwrap())
                .collect::<Vec<_>>(),
            vec![10, 10, 10, 12, 12, 12, 14, 14, 14]
        );
        assert_eq!(
            batch
                .combos
                .iter()
                .map(|combo| combo.mult.unwrap())
                .collect::<Vec<_>>(),
            vec![1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0]
        );
    }
}
