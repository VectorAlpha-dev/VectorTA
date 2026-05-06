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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use std::sync::OnceLock;
use thiserror::Error;

const DEFAULT_DOMESTIC_CYCLE_LENGTH: usize = 15;

#[derive(Debug, Clone)]
pub enum L1EhlersPhasorData<'a> {
    Candles { candles: &'a Candles },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct L1EhlersPhasorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct L1EhlersPhasorParams {
    pub domestic_cycle_length: Option<usize>,
}

impl Default for L1EhlersPhasorParams {
    fn default() -> Self {
        Self {
            domestic_cycle_length: Some(DEFAULT_DOMESTIC_CYCLE_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct L1EhlersPhasorInput<'a> {
    pub data: L1EhlersPhasorData<'a>,
    pub params: L1EhlersPhasorParams,
}

impl<'a> L1EhlersPhasorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: L1EhlersPhasorParams) -> Self {
        Self {
            data: L1EhlersPhasorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: L1EhlersPhasorParams) -> Self {
        Self {
            data: L1EhlersPhasorData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, L1EhlersPhasorParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct L1EhlersPhasorBuilder {
    domestic_cycle_length: Option<usize>,
    kernel: Kernel,
}

impl Default for L1EhlersPhasorBuilder {
    fn default() -> Self {
        Self {
            domestic_cycle_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl L1EhlersPhasorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn domestic_cycle_length(mut self, value: usize) -> Self {
        self.domestic_cycle_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<L1EhlersPhasorOutput, L1EhlersPhasorError> {
        let input = L1EhlersPhasorInput::from_candles(
            candles,
            L1EhlersPhasorParams {
                domestic_cycle_length: self.domestic_cycle_length,
            },
        );
        l1_ehlers_phasor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<L1EhlersPhasorOutput, L1EhlersPhasorError> {
        let input = L1EhlersPhasorInput::from_slice(
            data,
            L1EhlersPhasorParams {
                domestic_cycle_length: self.domestic_cycle_length,
            },
        );
        l1_ehlers_phasor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<L1EhlersPhasorStream, L1EhlersPhasorError> {
        L1EhlersPhasorStream::try_new(L1EhlersPhasorParams {
            domestic_cycle_length: self.domestic_cycle_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum L1EhlersPhasorError {
    #[error("l1_ehlers_phasor: Input data slice is empty.")]
    EmptyInputData,
    #[error("l1_ehlers_phasor: All values are NaN.")]
    AllValuesNaN,
    #[error("l1_ehlers_phasor: Invalid domestic_cycle_length: {domestic_cycle_length}")]
    InvalidDomesticCycleLength { domestic_cycle_length: usize },
    #[error("l1_ehlers_phasor: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("l1_ehlers_phasor: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("l1_ehlers_phasor: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("l1_ehlers_phasor: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct ResolvedParams {
    domestic_cycle_length: usize,
    cos_angle: f64,
    sin_angle: f64,
    cos_weights: WeightSlice,
    sin_weights: WeightSlice,
}

#[derive(Clone, Debug)]
enum WeightSlice {
    Static(&'static [f64]),
    Owned(Vec<f64>),
}

impl WeightSlice {
    #[inline(always)]
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::Static(values) => values,
            Self::Owned(values) => values.as_slice(),
        }
    }
}

static DEFAULT_WEIGHTS: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();

#[inline(always)]
fn default_weight_slices() -> (&'static [f64], &'static [f64]) {
    let weights = DEFAULT_WEIGHTS.get_or_init(|| {
        let angle = 2.0 * std::f64::consts::PI / DEFAULT_DOMESTIC_CYCLE_LENGTH as f64;
        let mut cos_weights = Vec::with_capacity(DEFAULT_DOMESTIC_CYCLE_LENGTH);
        let mut sin_weights = Vec::with_capacity(DEFAULT_DOMESTIC_CYCLE_LENGTH);
        for j in 0..DEFAULT_DOMESTIC_CYCLE_LENGTH {
            let theta = angle * j as f64;
            cos_weights.push(theta.cos());
            sin_weights.push(theta.sin());
        }
        (cos_weights, sin_weights)
    });
    (weights.0.as_slice(), weights.1.as_slice())
}

#[inline(always)]
fn extract_slice<'a>(input: &'a L1EhlersPhasorInput<'a>) -> Result<&'a [f64], L1EhlersPhasorError> {
    let data = match &input.data {
        L1EhlersPhasorData::Candles { candles } => candles.close.as_slice(),
        L1EhlersPhasorData::Slice(values) => *values,
    };
    if data.is_empty() {
        return Err(L1EhlersPhasorError::EmptyInputData);
    }
    Ok(data)
}

#[inline(always)]
fn first_valid(data: &[f64]) -> Option<usize> {
    data.iter().position(|value| value.is_finite())
}

#[inline(always)]
fn resolve_params(params: &L1EhlersPhasorParams) -> Result<ResolvedParams, L1EhlersPhasorError> {
    let domestic_cycle_length = params
        .domestic_cycle_length
        .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH);
    if domestic_cycle_length == 0 {
        return Err(L1EhlersPhasorError::InvalidDomesticCycleLength {
            domestic_cycle_length,
        });
    }
    let angle = 2.0 * std::f64::consts::PI / domestic_cycle_length as f64;
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    let (cos_weights, sin_weights) = if domestic_cycle_length == DEFAULT_DOMESTIC_CYCLE_LENGTH {
        let (cos_weights, sin_weights) = default_weight_slices();
        (
            WeightSlice::Static(cos_weights),
            WeightSlice::Static(sin_weights),
        )
    } else {
        let mut cos_weights = Vec::with_capacity(domestic_cycle_length);
        let mut sin_weights = Vec::with_capacity(domestic_cycle_length);
        for j in 0..domestic_cycle_length {
            let theta = angle * j as f64;
            cos_weights.push(theta.cos());
            sin_weights.push(theta.sin());
        }
        (
            WeightSlice::Owned(cos_weights),
            WeightSlice::Owned(sin_weights),
        )
    };
    Ok(ResolvedParams {
        domestic_cycle_length,
        cos_angle,
        sin_angle,
        cos_weights,
        sin_weights,
    })
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a L1EhlersPhasorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, Kernel), L1EhlersPhasorError> {
    let data = extract_slice(input)?;
    let resolved = resolve_params(&input.params)?;
    let first = first_valid(data).ok_or(L1EhlersPhasorError::AllValuesNaN)?;
    let valid = data.len().saturating_sub(first);
    if valid < resolved.domestic_cycle_length {
        return Err(L1EhlersPhasorError::NotEnoughValidData {
            needed: resolved.domestic_cycle_length,
            valid,
        });
    }
    Ok((data, resolved, first, kernel.to_non_batch()))
}

#[inline(always)]
fn compute_phase_angle(real: f64, imaginary: f64) -> f64 {
    let mut phase_angle = if real.abs() > 0.001 {
        (imaginary / real).atan().to_degrees()
    } else if imaginary > 0.0 {
        90.0
    } else if imaginary < 0.0 {
        -90.0
    } else {
        0.0
    };
    if real < 0.0 {
        phase_angle += 180.0;
    }
    phase_angle += 90.0;
    if phase_angle < 0.0 {
        phase_angle += 360.0;
    }
    if phase_angle > 360.0 {
        phase_angle -= 360.0;
    }
    phase_angle
}

#[derive(Clone, Debug)]
struct L1EhlersPhasorCore {
    params: ResolvedParams,
    ring: Vec<f64>,
    idx: usize,
    count: usize,
    invalid_count: usize,
    real: f64,
    imaginary: f64,
    phasor_valid: bool,
}

impl L1EhlersPhasorCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        let domestic_cycle_length = params.domestic_cycle_length;
        Self {
            params,
            ring: vec![f64::NAN; domestic_cycle_length],
            idx: 0,
            count: 0,
            invalid_count: 0,
            real: f64::NAN,
            imaginary: f64::NAN,
            phasor_valid: false,
        }
    }

    #[inline(always)]
    fn ring_get_lag(&self, current_idx: usize, lag: usize) -> f64 {
        let len = self.ring.len();
        let idx = (current_idx + len - (lag % len)) % len;
        self.ring[idx]
    }

    #[inline(always)]
    fn recompute_window(&mut self, current_idx: usize) {
        let len = self.params.domestic_cycle_length;
        let mut real_component = 0.0;
        let mut imaginary_component = 0.0;
        let cos_weights = self.params.cos_weights.as_slice();
        let sin_weights = self.params.sin_weights.as_slice();
        for j in 0..len {
            let value = self.ring_get_lag(current_idx, j);
            real_component += cos_weights[j] * value;
            imaginary_component += sin_weights[j] * value;
        }
        self.real = real_component;
        self.imaginary = imaginary_component;
        self.phasor_valid = true;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        let len = self.params.domestic_cycle_length;
        let full_before = self.count == len;
        let removed = if full_before {
            self.ring[self.idx]
        } else {
            f64::NAN
        };
        if full_before && !removed.is_finite() && self.invalid_count > 0 {
            self.invalid_count -= 1;
        }

        self.ring[self.idx] = value;
        if !value.is_finite() {
            self.invalid_count += 1;
        }
        if !full_before {
            self.count += 1;
        }

        let output = if self.count < len || self.invalid_count > 0 {
            self.phasor_valid = false;
            f64::NAN
        } else {
            if self.phasor_valid && full_before && value.is_finite() && removed.is_finite() {
                let prev_real = self.real;
                let prev_imaginary = self.imaginary;
                self.real = self.params.cos_angle * prev_real
                    - self.params.sin_angle * prev_imaginary
                    + value
                    - removed;
                self.imaginary =
                    self.params.sin_angle * prev_real + self.params.cos_angle * prev_imaginary;
            } else {
                self.recompute_window(self.idx);
            }
            compute_phase_angle(self.real, self.imaginary)
        };

        self.idx += 1;
        if self.idx == len {
            self.idx = 0;
        }
        output
    }
}

#[inline(always)]
fn compute_l1_ehlers_phasor_into(
    data: &[f64],
    params: ResolvedParams,
    first: usize,
    out: &mut [f64],
) -> Result<(), L1EhlersPhasorError> {
    if out.len() != data.len() {
        return Err(L1EhlersPhasorError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    if compute_l1_ehlers_phasor_clean(data, &params, first, out) {
        return Ok(());
    }
    let mut core = L1EhlersPhasorCore::new(params);
    for (dst, &value) in out.iter_mut().zip(data.iter()) {
        *dst = core.update(value);
    }
    Ok(())
}

#[inline(always)]
fn compute_l1_ehlers_phasor_clean(
    data: &[f64],
    params: &ResolvedParams,
    first: usize,
    out: &mut [f64],
) -> bool {
    let len = params.domestic_cycle_length;
    let warm = first + len - 1;
    for value in &mut out[..warm] {
        *value = f64::NAN;
    }

    let mut real = 0.0;
    let mut imaginary = 0.0;
    let cos_weights = params.cos_weights.as_slice();
    let sin_weights = params.sin_weights.as_slice();
    for j in 0..len {
        let value = data[warm - j];
        if !value.is_finite() {
            return false;
        }
        real += cos_weights[j] * value;
        imaginary += sin_weights[j] * value;
    }
    out[warm] = compute_phase_angle(real, imaginary);

    for i in (warm + 1)..data.len() {
        let value = data[i];
        let removed = data[i - len];
        if !value.is_finite() || !removed.is_finite() {
            return false;
        }
        let prev_real = real;
        let prev_imaginary = imaginary;
        real = params.cos_angle * prev_real - params.sin_angle * prev_imaginary + value - removed;
        imaginary = params.sin_angle * prev_real + params.cos_angle * prev_imaginary;
        out[i] = compute_phase_angle(real, imaginary);
    }
    true
}

#[inline(always)]
fn alloc_l1_output(len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
    }
    out
}

#[inline]
pub fn l1_ehlers_phasor(
    input: &L1EhlersPhasorInput,
) -> Result<L1EhlersPhasorOutput, L1EhlersPhasorError> {
    l1_ehlers_phasor_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn l1_ehlers_phasor_with_kernel(
    input: &L1EhlersPhasorInput,
    kernel: Kernel,
) -> Result<L1EhlersPhasorOutput, L1EhlersPhasorError> {
    let (data, params, first, _kernel) = validate_input(input, kernel)?;
    let mut out = alloc_l1_output(data.len());
    compute_l1_ehlers_phasor_into(data, params, first, &mut out)?;
    Ok(L1EhlersPhasorOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn l1_ehlers_phasor_into(
    out: &mut [f64],
    input: &L1EhlersPhasorInput,
    kernel: Kernel,
) -> Result<(), L1EhlersPhasorError> {
    l1_ehlers_phasor_into_slice(out, input, kernel)
}

#[inline]
pub fn l1_ehlers_phasor_into_slice(
    out: &mut [f64],
    input: &L1EhlersPhasorInput,
    kernel: Kernel,
) -> Result<(), L1EhlersPhasorError> {
    let (data, params, first, _kernel) = validate_input(input, kernel)?;
    compute_l1_ehlers_phasor_into(data, params, first, out)
}

#[derive(Clone, Debug)]
pub struct L1EhlersPhasorStream {
    core: L1EhlersPhasorCore,
}

impl L1EhlersPhasorStream {
    pub fn try_new(params: L1EhlersPhasorParams) -> Result<Self, L1EhlersPhasorError> {
        Ok(Self {
            core: L1EhlersPhasorCore::new(resolve_params(&params)?),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        self.core.update(value)
    }
}

#[derive(Clone, Debug)]
pub struct L1EhlersPhasorBatchRange {
    pub domestic_cycle_length: (usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct L1EhlersPhasorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<L1EhlersPhasorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct L1EhlersPhasorBatchBuilder {
    domestic_cycle_length: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for L1EhlersPhasorBatchBuilder {
    fn default() -> Self {
        Self {
            domestic_cycle_length: (
                DEFAULT_DOMESTIC_CYCLE_LENGTH,
                DEFAULT_DOMESTIC_CYCLE_LENGTH,
                0,
            ),
            kernel: Kernel::Auto,
        }
    }
}

impl L1EhlersPhasorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn domestic_cycle_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.domestic_cycle_length = value;
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
    ) -> Result<L1EhlersPhasorBatchOutput, L1EhlersPhasorError> {
        l1_ehlers_phasor_batch_with_kernel(
            candles.close.as_slice(),
            &L1EhlersPhasorBatchRange {
                domestic_cycle_length: self.domestic_cycle_length,
            },
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<L1EhlersPhasorBatchOutput, L1EhlersPhasorError> {
        l1_ehlers_phasor_batch_with_kernel(
            data,
            &L1EhlersPhasorBatchRange {
                domestic_cycle_length: self.domestic_cycle_length,
            },
            self.kernel,
        )
    }
}

pub fn expand_grid(
    sweep: &L1EhlersPhasorBatchRange,
) -> Result<Vec<L1EhlersPhasorParams>, L1EhlersPhasorError> {
    let (start, end, step) = sweep.domestic_cycle_length;
    if start == 0 {
        return Err(L1EhlersPhasorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut values = Vec::new();
    if step == 0 {
        if start != end {
            return Err(L1EhlersPhasorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        values.push(start);
    } else {
        if start > end {
            return Err(L1EhlersPhasorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut current = start;
        while current <= end {
            values.push(current);
            current = match current.checked_add(step) {
                Some(next) => next,
                None => break,
            };
        }
    }
    Ok(values
        .into_iter()
        .map(|domestic_cycle_length| L1EhlersPhasorParams {
            domestic_cycle_length: Some(domestic_cycle_length),
        })
        .collect())
}

#[inline(always)]
fn validate_raw_slices(data: &[f64]) -> Result<usize, L1EhlersPhasorError> {
    if data.is_empty() {
        return Err(L1EhlersPhasorError::EmptyInputData);
    }
    first_valid(data).ok_or(L1EhlersPhasorError::AllValuesNaN)
}

pub fn l1_ehlers_phasor_batch_with_kernel(
    data: &[f64],
    sweep: &L1EhlersPhasorBatchRange,
    kernel: Kernel,
) -> Result<L1EhlersPhasorBatchOutput, L1EhlersPhasorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(L1EhlersPhasorError::InvalidKernelForBatch(kernel)),
    };
    l1_ehlers_phasor_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn l1_ehlers_phasor_batch_slice(
    data: &[f64],
    sweep: &L1EhlersPhasorBatchRange,
    kernel: Kernel,
) -> Result<L1EhlersPhasorBatchOutput, L1EhlersPhasorError> {
    l1_ehlers_phasor_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn l1_ehlers_phasor_batch_par_slice(
    data: &[f64],
    sweep: &L1EhlersPhasorBatchRange,
    kernel: Kernel,
) -> Result<L1EhlersPhasorBatchOutput, L1EhlersPhasorError> {
    l1_ehlers_phasor_batch_inner(data, sweep, kernel, true)
}

fn l1_ehlers_phasor_batch_inner(
    data: &[f64],
    sweep: &L1EhlersPhasorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<L1EhlersPhasorBatchOutput, L1EhlersPhasorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(data)?;
    let rows = combos.len();
    let cols = data.len();
    let warmups = combos
        .iter()
        .map(|combo| {
            let domestic_cycle_length = combo
                .domestic_cycle_length
                .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH);
            (first + domestic_cycle_length.saturating_sub(1)).min(cols)
        })
        .collect::<Vec<_>>();

    let mut buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf, cols, &warmups);
    let mut guard = ManuallyDrop::new(buf);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    l1_ehlers_phasor_batch_inner_into(data, sweep, kernel, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(L1EhlersPhasorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn l1_ehlers_phasor_batch_into_slice(
    out: &mut [f64],
    data: &[f64],
    sweep: &L1EhlersPhasorBatchRange,
    kernel: Kernel,
) -> Result<(), L1EhlersPhasorError> {
    l1_ehlers_phasor_batch_inner_into(data, sweep, kernel, false, out)?;
    Ok(())
}

fn l1_ehlers_phasor_batch_inner_into(
    data: &[f64],
    sweep: &L1EhlersPhasorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<L1EhlersPhasorParams>, L1EhlersPhasorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(data)?;
    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| L1EhlersPhasorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;
    if out.len() != expected {
        return Err(L1EhlersPhasorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let min_needed = combos
        .iter()
        .map(|combo| {
            combo
                .domestic_cycle_length
                .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH)
        })
        .max()
        .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH);
    let valid = cols.saturating_sub(first);
    if valid < min_needed {
        return Err(L1EhlersPhasorError::NotEnoughValidData {
            needed: min_needed,
            valid,
        });
    }

    let do_row = |row: usize, dst: &mut [f64]| -> Result<(), L1EhlersPhasorError> {
        let params = resolve_params(&combos[row])?;
        compute_l1_ehlers_phasor_into(data, params, first, dst)
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, dst)| do_row(row, dst))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                do_row(row, dst)?;
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            do_row(row, dst)?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "l1_ehlers_phasor")]
#[pyo3(signature = (data, domestic_cycle_length=15, kernel=None))]
pub fn l1_ehlers_phasor_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    domestic_cycle_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = L1EhlersPhasorInput::from_slice(
        data,
        L1EhlersPhasorParams {
            domestic_cycle_length: Some(domestic_cycle_length),
        },
    );
    let out = py
        .allow_threads(|| l1_ehlers_phasor_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "L1EhlersPhasorStream")]
pub struct L1EhlersPhasorStreamPy {
    stream: L1EhlersPhasorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl L1EhlersPhasorStreamPy {
    #[new]
    #[pyo3(signature = (domestic_cycle_length=15))]
    fn new(domestic_cycle_length: usize) -> PyResult<Self> {
        let stream = L1EhlersPhasorStream::try_new(L1EhlersPhasorParams {
            domestic_cycle_length: Some(domestic_cycle_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> f64 {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "l1_ehlers_phasor_batch")]
#[pyo3(signature = (data, domestic_cycle_length_range=(15,15,0), kernel=None))]
pub fn l1_ehlers_phasor_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    domestic_cycle_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = L1EhlersPhasorBatchRange {
        domestic_cycle_length: domestic_cycle_length_range,
    };
    let out = py
        .allow_threads(|| l1_ehlers_phasor_batch_with_kernel(data, &sweep, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        out.values
            .clone()
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "domestic_cycle_lengths",
        out.combos
            .iter()
            .map(|combo| {
                combo
                    .domestic_cycle_length
                    .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH) as u64
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_l1_ehlers_phasor_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(l1_ehlers_phasor_py, m)?)?;
    m.add_function(wrap_pyfunction!(l1_ehlers_phasor_batch_py, m)?)?;
    m.add_class::<L1EhlersPhasorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l1_ehlers_phasor_js")]
pub fn l1_ehlers_phasor_js(
    data: &[f64],
    domestic_cycle_length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = L1EhlersPhasorInput::from_slice(
        data,
        L1EhlersPhasorParams {
            domestic_cycle_length: Some(domestic_cycle_length),
        },
    );
    let mut out = vec![0.0; data.len()];
    l1_ehlers_phasor_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct L1EhlersPhasorBatchConfig {
    pub domestic_cycle_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct L1EhlersPhasorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<L1EhlersPhasorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l1_ehlers_phasor_batch_js")]
pub fn l1_ehlers_phasor_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: L1EhlersPhasorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.domestic_cycle_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: domestic_cycle_length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = l1_ehlers_phasor_batch_with_kernel(
        data,
        &L1EhlersPhasorBatchRange {
            domestic_cycle_length: (
                config.domestic_cycle_length_range[0],
                config.domestic_cycle_length_range[1],
                config.domestic_cycle_length_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&L1EhlersPhasorBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l1_ehlers_phasor_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l1_ehlers_phasor_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l1_ehlers_phasor_into")]
pub fn l1_ehlers_phasor_into_wasm(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    domestic_cycle_length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = L1EhlersPhasorInput::from_slice(
            data,
            L1EhlersPhasorParams {
                domestic_cycle_length: Some(domestic_cycle_length),
            },
        );
        l1_ehlers_phasor_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l1_ehlers_phasor_batch_into")]
pub fn l1_ehlers_phasor_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    domestic_cycle_length_start: usize,
    domestic_cycle_length_end: usize,
    domestic_cycle_length_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to l1_ehlers_phasor_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = L1EhlersPhasorBatchRange {
            domestic_cycle_length: (
                domestic_cycle_length_start,
                domestic_cycle_length_end,
                domestic_cycle_length_step,
            ),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in l1_ehlers_phasor_batch_into")
        })?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        l1_ehlers_phasor_batch_into_slice(out, data, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
mod tests {
    use super::*;

    fn build_series(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                100.0
                    + i as f64 * 0.23
                    + (i as f64 * 0.17).sin() * 1.7
                    + (i as f64 * 0.043).cos() * 0.6
            })
            .collect()
    }

    fn manual_reference(data: &[f64], domestic_cycle_length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        if domestic_cycle_length == 0 {
            return out;
        }
        let pi = std::f64::consts::PI;
        for i in 0..data.len() {
            if i + 1 < domestic_cycle_length {
                continue;
            }
            let mut real_component = 0.0;
            let mut imaginary_component = 0.0;
            let mut valid = true;
            for j in 0..domestic_cycle_length {
                let value = data[i - j];
                if !value.is_finite() {
                    valid = false;
                    break;
                }
                let angle = 2.0 * pi * j as f64 / domestic_cycle_length as f64;
                real_component += angle.cos() * value;
                imaginary_component += angle.sin() * value;
            }
            if !valid {
                continue;
            }
            out[i] = compute_phase_angle(real_component, imaginary_component);
        }
        out
    }

    fn assert_close_series(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            let a = lhs[i];
            let b = rhs[i];
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol,
                "mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn manual_reference_matches_api() {
        let data = build_series(128);
        let expected = manual_reference(&data, 15);
        let input = L1EhlersPhasorInput::from_slice(
            &data,
            L1EhlersPhasorParams {
                domestic_cycle_length: Some(15),
            },
        );
        let out = l1_ehlers_phasor(&input).unwrap();
        assert_close_series(&out.values, &expected, 5e-11);
    }

    #[test]
    fn stream_matches_batch() {
        let data = build_series(144);
        let batch = L1EhlersPhasorBuilder::new()
            .domestic_cycle_length(15)
            .apply_slice(&data)
            .unwrap();
        let mut stream = L1EhlersPhasorBuilder::new()
            .domestic_cycle_length(15)
            .into_stream()
            .unwrap();
        let streamed: Vec<f64> = data.iter().map(|&value| stream.update(value)).collect();
        assert_close_series(&batch.values, &streamed, 5e-11);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let data = build_series(160);
        let sweep = L1EhlersPhasorBatchRange {
            domestic_cycle_length: (15, 17, 2),
        };
        let batch = l1_ehlers_phasor_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, data.len());

        let input = L1EhlersPhasorInput::from_slice(
            &data,
            L1EhlersPhasorParams {
                domestic_cycle_length: Some(15),
            },
        );
        let single = l1_ehlers_phasor(&input).unwrap();
        assert_close_series(&batch.values[..data.len()], &single.values, 5e-11);
    }

    #[test]
    fn into_slice_matches_single() {
        let data = build_series(96);
        let input = L1EhlersPhasorInput::from_slice(
            &data,
            L1EhlersPhasorParams {
                domestic_cycle_length: Some(15),
            },
        );
        let single = l1_ehlers_phasor(&input).unwrap();
        let mut out = vec![f64::NAN; data.len()];
        l1_ehlers_phasor_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_close_series(&single.values, &out, 5e-11);
    }

    #[test]
    fn rejects_invalid_length() {
        let data = build_series(32);
        let input = L1EhlersPhasorInput::from_slice(
            &data,
            L1EhlersPhasorParams {
                domestic_cycle_length: Some(0),
            },
        );
        let err = l1_ehlers_phasor(&input).unwrap_err();
        assert!(matches!(
            err,
            L1EhlersPhasorError::InvalidDomesticCycleLength { .. }
        ));
    }
}
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l1_ehlers_phasor_output_into_js(
    data: &[f64],
    domestic_cycle_length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = l1_ehlers_phasor_js(data, domestic_cycle_length)?;
    crate::write_wasm_f64_output("l1_ehlers_phasor_output_into_js", &values, out)
}
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l1_ehlers_phasor_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = l1_ehlers_phasor_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "l1_ehlers_phasor_batch_output_into_js",
        &value,
        out,
    )
}
