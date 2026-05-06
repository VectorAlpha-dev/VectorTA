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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_LENGTH: usize = 21;
const DEFAULT_SMOOTH_LENGTH: usize = 5;
const MIN_LENGTH: usize = 2;
const MAX_LENGTH: usize = 60;
const MIN_SMOOTH_LENGTH: usize = 1;
const MAX_SMOOTH_LENGTH: usize = 9;

impl<'a> AsRef<[f64]> for VelocityInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VelocityData::Slice(slice) => slice,
            VelocityData::Candles { candles, source } => match *source {
                "hlcc4" => candles.hlcc4.as_slice(),
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum VelocityData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VelocityOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VelocityParams {
    pub length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for VelocityParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            smooth_length: Some(DEFAULT_SMOOTH_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VelocityInput<'a> {
    pub data: VelocityData<'a>,
    pub params: VelocityParams,
}

impl<'a> VelocityInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VelocityParams) -> Self {
        Self {
            data: VelocityData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: VelocityParams) -> Self {
        Self {
            data: VelocityData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "hlcc4", VelocityParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VelocityBuilder {
    length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for VelocityBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VelocityBuilder {
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
    pub fn smooth_length(mut self, smooth_length: usize) -> Self {
        self.smooth_length = Some(smooth_length);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<VelocityOutput, VelocityError> {
        let input = VelocityInput::from_candles(
            candles,
            "hlcc4",
            VelocityParams {
                length: self.length,
                smooth_length: self.smooth_length,
            },
        );
        velocity_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<VelocityOutput, VelocityError> {
        let input = VelocityInput::from_slice(
            data,
            VelocityParams {
                length: self.length,
                smooth_length: self.smooth_length,
            },
        );
        velocity_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VelocityStream, VelocityError> {
        VelocityStream::try_new(VelocityParams {
            length: self.length,
            smooth_length: self.smooth_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum VelocityError {
    #[error("velocity: input data slice is empty.")]
    EmptyInputData,
    #[error("velocity: all values are NaN.")]
    AllValuesNaN,
    #[error("velocity: invalid length: {length}. Expected 2..=60.")]
    InvalidLength { length: usize },
    #[error("velocity: invalid smoothing length: {smooth_length}. Expected 1..=9.")]
    InvalidSmoothLength { smooth_length: usize },
    #[error("velocity: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("velocity: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("velocity: invalid length range: start={start}, end={end}, step={step}")]
    InvalidLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("velocity: invalid smoothing length range: start={start}, end={end}, step={step}")]
    InvalidSmoothLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("velocity: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Copy, Clone, Debug)]
struct PreparedVelocity<'a> {
    data: &'a [f64],
    first_valid: usize,
    length: usize,
    smooth_length: usize,
}

#[derive(Debug, Clone)]
struct VelocityCore {
    length: usize,
    smooth_length: usize,
    harmonic_over_length: f64,
    history: Vec<f64>,
    history_head: usize,
    history_count: usize,
    raw_ring: Vec<f64>,
    raw_head: usize,
    raw_count: usize,
}

impl VelocityCore {
    #[inline]
    fn new(length: usize, smooth_length: usize) -> Self {
        let mut harmonic = 0.0;
        for lag in 1..=length {
            harmonic += 1.0 / lag as f64;
        }
        Self {
            length,
            smooth_length,
            harmonic_over_length: harmonic / length as f64,
            history: vec![f64::NAN; length],
            history_head: 0,
            history_count: 0,
            raw_ring: vec![f64::NAN; smooth_length],
            raw_head: 0,
            raw_count: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.history.fill(f64::NAN);
        self.history_head = 0;
        self.history_count = 0;
        self.raw_ring.fill(f64::NAN);
        self.raw_head = 0;
        self.raw_count = 0;
    }

    #[inline(always)]
    fn history_value(&self, lag: usize) -> f64 {
        if lag == 0 || lag > self.history_count {
            return 0.0;
        }
        let idx = (self.history_head + self.length - lag) % self.length;
        let value = self.history[idx];
        if value.is_finite() {
            value
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn push_history(&mut self, value: f64) {
        self.history[self.history_head] = value;
        self.history_head += 1;
        if self.history_head == self.length {
            self.history_head = 0;
        }
        if self.history_count < self.length {
            self.history_count += 1;
        }
    }

    #[inline(always)]
    fn push_raw(&mut self, raw: f64) -> Option<f64> {
        self.raw_ring[self.raw_head] = raw;
        self.raw_head += 1;
        if self.raw_head == self.smooth_length {
            self.raw_head = 0;
        }
        if self.raw_count < self.smooth_length {
            self.raw_count += 1;
        }
        if self.raw_count < self.smooth_length {
            return None;
        }

        let mut weighted = 0.0;
        for offset in 0..self.smooth_length {
            let idx = (self.raw_head + offset) % self.smooth_length;
            let value = self.raw_ring[idx];
            if !value.is_finite() {
                return Some(f64::NAN);
            }
            weighted += (offset + 1) as f64 * value;
        }

        let denom = (self.smooth_length * (self.smooth_length + 1) / 2) as f64;
        Some(weighted / denom)
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let raw = if value.is_finite() {
            let mut weighted_past = 0.0;
            for lag in 1..=self.length {
                weighted_past += self.history_value(lag) / lag as f64;
            }
            value * self.harmonic_over_length - weighted_past / self.length as f64
        } else {
            f64::NAN
        };

        self.push_history(value);
        self.push_raw(raw)
    }
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => Kernel::Scalar,
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        other => other,
    }
}

#[inline(always)]
fn validate_params(length: usize, smooth_length: usize) -> Result<(), VelocityError> {
    if !(MIN_LENGTH..=MAX_LENGTH).contains(&length) {
        return Err(VelocityError::InvalidLength { length });
    }
    if !(MIN_SMOOTH_LENGTH..=MAX_SMOOTH_LENGTH).contains(&smooth_length) {
        return Err(VelocityError::InvalidSmoothLength { smooth_length });
    }
    Ok(())
}

#[inline(always)]
fn velocity_prepare<'a>(
    input: &'a VelocityInput<'a>,
    kernel: Kernel,
) -> Result<(PreparedVelocity<'a>, Kernel), VelocityError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(VelocityError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(VelocityError::AllValuesNaN)?;
    let length = input.get_length();
    let smooth_length = input.get_smooth_length();
    validate_params(length, smooth_length)?;

    let valid = data.len() - first_valid;
    if valid < smooth_length {
        return Err(VelocityError::NotEnoughValidData {
            needed: smooth_length,
            valid,
        });
    }

    Ok((
        PreparedVelocity {
            data,
            first_valid,
            length,
            smooth_length,
        },
        normalize_single_kernel(kernel),
    ))
}

#[inline(always)]
fn compute_velocity_default_into(data: &[f64], first_valid: usize, out: &mut [f64]) {
    let mut harmonic = 0.0;
    for lag in 1..=DEFAULT_LENGTH {
        harmonic += 1.0 / lag as f64;
    }
    let harmonic_over_length = harmonic / DEFAULT_LENGTH as f64;
    let mut history = [f64::NAN; DEFAULT_LENGTH];
    let mut history_head = 0usize;
    let mut history_count = 0usize;
    let mut raw_ring = [f64::NAN; DEFAULT_SMOOTH_LENGTH];
    let mut raw_head = 0usize;
    let mut raw_count = 0usize;
    let raw_denom = (DEFAULT_SMOOTH_LENGTH * (DEFAULT_SMOOTH_LENGTH + 1) / 2) as f64;

    for idx in first_valid..data.len() {
        let value = data[idx];
        let raw = if value.is_finite() {
            let mut weighted_past = 0.0;
            for lag in 1..=DEFAULT_LENGTH {
                let hist = if lag <= history_count {
                    let mut hist_idx = history_head + DEFAULT_LENGTH - lag;
                    if hist_idx >= DEFAULT_LENGTH {
                        hist_idx -= DEFAULT_LENGTH;
                    }
                    let past = history[hist_idx];
                    if past.is_finite() {
                        past
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                weighted_past += hist / lag as f64;
            }
            value * harmonic_over_length - weighted_past / DEFAULT_LENGTH as f64
        } else {
            f64::NAN
        };

        history[history_head] = value;
        history_head += 1;
        if history_head == DEFAULT_LENGTH {
            history_head = 0;
        }
        if history_count < DEFAULT_LENGTH {
            history_count += 1;
        }

        raw_ring[raw_head] = raw;
        raw_head += 1;
        if raw_head == DEFAULT_SMOOTH_LENGTH {
            raw_head = 0;
        }
        if raw_count < DEFAULT_SMOOTH_LENGTH {
            raw_count += 1;
        }
        if raw_count < DEFAULT_SMOOTH_LENGTH {
            out[idx] = f64::NAN;
            continue;
        }

        let mut weighted = 0.0;
        let mut valid = true;
        for offset in 0..DEFAULT_SMOOTH_LENGTH {
            let mut raw_idx = raw_head + offset;
            if raw_idx >= DEFAULT_SMOOTH_LENGTH {
                raw_idx -= DEFAULT_SMOOTH_LENGTH;
            }
            let raw_value = raw_ring[raw_idx];
            if !raw_value.is_finite() {
                valid = false;
                break;
            }
            weighted += (offset + 1) as f64 * raw_value;
        }
        out[idx] = if valid {
            weighted / raw_denom
        } else {
            f64::NAN
        };
    }
}

#[inline(always)]
fn compute_velocity_into(prepared: PreparedVelocity<'_>, out: &mut [f64]) {
    if prepared.length == DEFAULT_LENGTH && prepared.smooth_length == DEFAULT_SMOOTH_LENGTH {
        compute_velocity_default_into(prepared.data, prepared.first_valid, out);
        return;
    }

    let mut core = VelocityCore::new(prepared.length, prepared.smooth_length);
    for idx in prepared.first_valid..prepared.data.len() {
        out[idx] = match core.update(prepared.data[idx]) {
            Some(value) => value,
            None => f64::NAN,
        };
    }
}

#[inline]
pub fn velocity(input: &VelocityInput) -> Result<VelocityOutput, VelocityError> {
    velocity_with_kernel(input, Kernel::Auto)
}

pub fn velocity_with_kernel(
    input: &VelocityInput,
    kernel: Kernel,
) -> Result<VelocityOutput, VelocityError> {
    let (prepared, _) = velocity_prepare(input, kernel)?;
    let warm = prepared.first_valid + prepared.smooth_length - 1;
    let mut out = alloc_with_nan_prefix(prepared.data.len(), warm);
    compute_velocity_into(prepared, &mut out);
    Ok(VelocityOutput { values: out })
}

#[inline]
pub fn velocity_into_slice(
    out: &mut [f64],
    input: &VelocityInput,
    kernel: Kernel,
) -> Result<(), VelocityError> {
    let (prepared, _) = velocity_prepare(input, kernel)?;
    if out.len() != prepared.data.len() {
        return Err(VelocityError::OutputLengthMismatch {
            expected: prepared.data.len(),
            got: out.len(),
        });
    }

    let warm = (prepared.first_valid + prepared.smooth_length - 1).min(out.len());
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for value in &mut out[..warm] {
        *value = qnan;
    }
    compute_velocity_into(prepared, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn velocity_into(input: &VelocityInput, out: &mut [f64]) -> Result<(), VelocityError> {
    velocity_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct VelocityStream {
    core: VelocityCore,
    started: bool,
}

impl VelocityStream {
    pub fn try_new(params: VelocityParams) -> Result<Self, VelocityError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let smooth_length = params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH);
        validate_params(length, smooth_length)?;
        Ok(Self {
            core: VelocityCore::new(length, smooth_length),
            started: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.started {
            if value.is_nan() {
                return None;
            }
            self.started = true;
        }
        self.core.update(value)
    }

    #[inline]
    pub fn reset(&mut self) {
        self.started = false;
        self.core.reset();
    }
}

#[derive(Clone, Debug)]
pub struct VelocityBatchRange {
    pub length: (usize, usize, usize),
    pub smooth_length: (usize, usize, usize),
}

impl Default for VelocityBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            smooth_length: (DEFAULT_SMOOTH_LENGTH, DEFAULT_SMOOTH_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VelocityBatchBuilder {
    range: VelocityBatchRange,
    kernel: Kernel,
}

impl VelocityBatchBuilder {
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

    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    pub fn smooth_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth_length = (start, end, step);
        self
    }

    pub fn smooth_length_static(mut self, smooth_length: usize) -> Self {
        self.range.smooth_length = (smooth_length, smooth_length, 0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<VelocityBatchOutput, VelocityError> {
        velocity_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(self, candles: &Candles) -> Result<VelocityBatchOutput, VelocityError> {
        self.apply_slice(source_type(candles, "hlcc4"))
    }

    pub fn apply_candles_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<VelocityBatchOutput, VelocityError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct VelocityBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VelocityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VelocityBatchOutput {
    pub fn row_for_params(&self, params: &VelocityParams) -> Option<usize> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let smooth_length = params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH);
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == length
                && combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH) == smooth_length
        })
    }

    pub fn values_for(&self, params: &VelocityParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_axis(axis: (usize, usize, usize), is_smooth: bool) -> Result<Vec<usize>, VelocityError> {
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
        return Err(if is_smooth {
            VelocityError::InvalidSmoothLengthRange { start, end, step }
        } else {
            VelocityError::InvalidLengthRange { start, end, step }
        });
    }

    Ok(values)
}

#[inline(always)]
fn expand_grid(range: &VelocityBatchRange) -> Result<Vec<VelocityParams>, VelocityError> {
    let lengths = expand_axis(range.length, false)?;
    let smooth_lengths = expand_axis(range.smooth_length, true)?;
    let mut combos = Vec::with_capacity(lengths.len() * smooth_lengths.len());
    for &length in &lengths {
        for &smooth_length in &smooth_lengths {
            validate_params(length, smooth_length)?;
            combos.push(VelocityParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            });
        }
    }
    Ok(combos)
}

pub fn velocity_batch_with_kernel(
    data: &[f64],
    sweep: &VelocityBatchRange,
    kernel: Kernel,
) -> Result<VelocityBatchOutput, VelocityError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VelocityError::InvalidKernelForBatch(other)),
    };

    let scalar_kernel = match batch_kernel {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => unreachable!(),
    };

    velocity_batch_inner(
        data,
        sweep,
        scalar_kernel,
        !matches!(batch_kernel, Kernel::ScalarBatch),
    )
}

#[inline(always)]
pub fn velocity_batch_slice(
    data: &[f64],
    sweep: &VelocityBatchRange,
    kernel: Kernel,
) -> Result<VelocityBatchOutput, VelocityError> {
    velocity_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn velocity_batch_par_slice(
    data: &[f64],
    sweep: &VelocityBatchRange,
    kernel: Kernel,
) -> Result<VelocityBatchOutput, VelocityError> {
    velocity_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn velocity_batch_inner(
    data: &[f64],
    sweep: &VelocityBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<VelocityBatchOutput, VelocityError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(VelocityError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(VelocityError::AllValuesNaN)?;
    let valid = data.len() - first_valid;
    let max_smooth = combos
        .iter()
        .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH))
        .max()
        .unwrap_or(DEFAULT_SMOOTH_LENGTH);
    if valid < max_smooth {
        return Err(VelocityError::NotEnoughValidData {
            needed: max_smooth,
            valid,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut buf = make_uninit_matrix(rows, cols);
    let warm_prefixes: Vec<usize> = combos
        .iter()
        .map(|combo| first_valid + combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH) - 1)
        .collect();
    init_matrix_prefixes(&mut buf, cols, &warm_prefixes);

    let mut guard = ManuallyDrop::new(buf);
    let out_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let _ = normalize_single_kernel(kernel);
    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let out = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        compute_velocity_into(
            PreparedVelocity {
                data,
                first_valid,
                length: combos[row].length.unwrap_or(DEFAULT_LENGTH),
                smooth_length: combos[row].smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
            },
            out,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_mu)| do_row(row, row_mu));
        #[cfg(target_arch = "wasm32")]
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    } else {
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VelocityBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn velocity_batch_inner_into(
    data: &[f64],
    sweep: &VelocityBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VelocityParams>, VelocityError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(VelocityError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(VelocityError::AllValuesNaN)?;
    let valid = data.len() - first_valid;
    let max_smooth = combos
        .iter()
        .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH))
        .max()
        .unwrap_or(DEFAULT_SMOOTH_LENGTH);
    if valid < max_smooth {
        return Err(VelocityError::NotEnoughValidData {
            needed: max_smooth,
            valid,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(VelocityError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        })?;
    if out.len() != expected {
        return Err(VelocityError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let _ = normalize_single_kernel(kernel);
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_out)| {
                compute_velocity_into(
                    PreparedVelocity {
                        data,
                        first_valid,
                        length: combos[row].length.unwrap_or(DEFAULT_LENGTH),
                        smooth_length: combos[row].smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                    },
                    row_out,
                );
            });
        #[cfg(target_arch = "wasm32")]
        for (row, row_out) in out.chunks_mut(cols).enumerate() {
            compute_velocity_into(
                PreparedVelocity {
                    data,
                    first_valid,
                    length: combos[row].length.unwrap_or(DEFAULT_LENGTH),
                    smooth_length: combos[row].smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                },
                row_out,
            );
        }
    } else {
        for (row, row_out) in out.chunks_mut(cols).enumerate() {
            compute_velocity_into(
                PreparedVelocity {
                    data,
                    first_valid,
                    length: combos[row].length.unwrap_or(DEFAULT_LENGTH),
                    smooth_length: combos[row].smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                },
                row_out,
            );
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "velocity")]
#[pyo3(signature = (data, length=21, smooth_length=5, kernel=None))]
pub fn velocity_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    smooth_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = VelocityInput::from_slice(
        slice,
        VelocityParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let out = py
        .allow_threads(|| velocity_with_kernel(&input, kernel).map(|output| output.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "velocity_batch")]
#[pyo3(signature = (data, length_range, smooth_length_range, kernel=None))]
pub fn velocity_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = VelocityBatchRange {
        length: length_range,
        smooth_length: smooth_length_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            velocity_batch_inner_into(
                slice,
                &sweep,
                batch_kernel,
                !matches!(batch_kernel, Kernel::ScalarBatch),
                out_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        combos
            .iter()
            .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "VelocityStream")]
pub struct VelocityStreamPy {
    inner: VelocityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VelocityStreamPy {
    #[new]
    pub fn new(length: usize, smooth_length: usize) -> PyResult<Self> {
        let inner = VelocityStream::try_new(VelocityParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
pub fn register_velocity_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(velocity_py, m)?)?;
    m.add_function(wrap_pyfunction!(velocity_batch_py, m)?)?;
    m.add_class::<VelocityStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VelocityBatchConfig {
    pub length_range: (usize, usize, usize),
    pub smooth_length_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VelocityBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VelocityParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_js(data: &[f64], length: usize, smooth_length: usize) -> Result<Vec<f64>, JsValue> {
    let input = VelocityInput::from_slice(
        data,
        VelocityParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let mut out = vec![0.0; data.len()];
    velocity_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = velocity_batch)]
pub fn velocity_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: VelocityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = VelocityBatchRange {
        length: config.length_range,
        smooth_length: config.smooth_length_range,
    };
    let output = velocity_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VelocityBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_alloc(len: usize) -> *mut f64 {
    let mut values = Vec::<f64>::with_capacity(len);
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to velocity_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = VelocityInput::from_slice(
            data,
            VelocityParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            },
        );
        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            velocity_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            velocity_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    smooth_length_start: usize,
    smooth_length_end: usize,
    smooth_length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to velocity_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = VelocityBatchRange {
            length: (length_start, length_end, length_step),
            smooth_length: (smooth_length_start, smooth_length_end, smooth_length_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let batch_kernel = detect_best_batch_kernel();
        velocity_batch_inner_into(
            data,
            &sweep,
            batch_kernel,
            !matches!(batch_kernel, Kernel::ScalarBatch),
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct VelocityStreamWasm {
    inner: VelocityStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl VelocityStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(length: usize, smooth_length: usize) -> Result<VelocityStreamWasm, JsValue> {
        Ok(Self {
            inner: VelocityStream::try_new(VelocityParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        })
    }

    pub fn update(&mut self, value: f64) -> Result<JsValue, JsValue> {
        match self.inner.update(value) {
            Some(output) => {
                serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Ok(JsValue::NULL),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_output_into_js(
    data: &[f64],
    length: usize,
    smooth_length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = velocity_js(data, length, smooth_length)?;
    crate::write_wasm_f64_output("velocity_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = velocity_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs("velocity_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn naive_velocity(data: &[f64], length: usize, smooth_length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        let Some(first_valid) = data.iter().position(|value| !value.is_nan()) else {
            return out;
        };

        let denom = (smooth_length * (smooth_length + 1) / 2) as f64;
        let mut raw = vec![f64::NAN; data.len()];
        for idx in first_valid..data.len() {
            let value = data[idx];
            if !value.is_finite() {
                continue;
            }
            let mut acc = 0.0;
            for lag in 1..=length {
                let hist = if idx >= lag {
                    let prev = data[idx - lag];
                    if prev.is_finite() {
                        prev
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                acc += (value - hist) / lag as f64;
            }
            raw[idx] = acc / length as f64;
        }

        for idx in (first_valid + smooth_length - 1)..data.len() {
            let mut weighted = 0.0;
            let mut valid = true;
            for offset in 0..smooth_length {
                let raw_value = raw[idx - smooth_length + 1 + offset];
                if !raw_value.is_finite() {
                    valid = false;
                    break;
                }
                weighted += (offset + 1) as f64 * raw_value;
            }
            if valid {
                out[idx] = weighted / denom;
            }
        }
        out
    }

    fn sample_data() -> Vec<f64> {
        (0..256)
            .map(|idx| {
                let x = idx as f64;
                100.0 + (x * 0.07).sin() * 3.0 + (x * 0.033).cos() * 1.75 + x * 0.015
            })
            .collect()
    }

    #[test]
    fn velocity_matches_naive_reference() -> Result<(), Box<dyn Error>> {
        let data = vec![f64::NAN, f64::NAN, 10.0, 11.0, 12.0, 13.0, 12.0, 14.0];
        let input = VelocityInput::from_slice(
            &data,
            VelocityParams {
                length: Some(3),
                smooth_length: Some(2),
            },
        );
        let output = velocity(&input)?;
        let expected = naive_velocity(&data, 3, 2);
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert!((actual.is_nan() && expected.is_nan()) || (actual - expected).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn velocity_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data();
        let input = VelocityInput::from_slice(&data, VelocityParams::default());
        let baseline = velocity(&input)?.values;
        let mut out = vec![0.0; data.len()];
        velocity_into(&input, &mut out)?;
        for (actual, expected) in out.iter().zip(baseline.iter()) {
            assert!((actual.is_nan() && expected.is_nan()) || (actual - expected).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn velocity_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_data();
        let batch = velocity(&VelocityInput::from_slice(&data, VelocityParams::default()))?;
        let mut stream = VelocityStream::try_new(VelocityParams::default())?;
        let mut values = vec![f64::NAN; data.len()];
        for (idx, value) in data.iter().copied().enumerate() {
            if let Some(output) = stream.update(value) {
                values[idx] = output;
            }
        }
        for (actual, expected) in values.iter().zip(batch.values.iter()) {
            assert!((actual.is_nan() && expected.is_nan()) || (actual - expected).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn velocity_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let data = sample_data();
        let batch = velocity_batch_with_kernel(
            &data,
            &VelocityBatchRange {
                length: (10, 12, 2),
                smooth_length: (3, 5, 2),
            },
            Kernel::ScalarBatch,
        )?;

        for (row, combo) in batch.combos.iter().enumerate() {
            let single = velocity(&VelocityInput::from_slice(&data, combo.clone()))?;
            let start = row * batch.cols;
            let row_values = &batch.values[start..start + batch.cols];
            for (actual, expected) in row_values.iter().zip(single.values.iter()) {
                assert!(
                    (actual.is_nan() && expected.is_nan()) || (actual - expected).abs() <= 1e-12
                );
            }
        }
        Ok(())
    }

    #[test]
    fn velocity_fixture_has_values() -> Result<(), Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let out = velocity(&VelocityInput::with_default_candles(&candles))?;
        assert_eq!(out.values.len(), candles.close.len());
        assert!(out.values.iter().skip(64).any(|value| value.is_finite()));
        Ok(())
    }
}
