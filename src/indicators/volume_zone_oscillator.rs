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

const DEFAULT_LENGTH: usize = 14;
const DEFAULT_INTRADAY_SMOOTHING: bool = true;
const DEFAULT_NOISE_FILTER: usize = 4;

#[derive(Debug, Clone)]
pub enum VolumeZoneOscillatorData<'a> {
    Candles { candles: &'a Candles },
    Slices { close: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct VolumeZoneOscillatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VolumeZoneOscillatorParams {
    pub length: Option<usize>,
    pub intraday_smoothing: Option<bool>,
    pub noise_filter: Option<usize>,
}

impl Default for VolumeZoneOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            intraday_smoothing: Some(DEFAULT_INTRADAY_SMOOTHING),
            noise_filter: Some(DEFAULT_NOISE_FILTER),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeZoneOscillatorInput<'a> {
    pub data: VolumeZoneOscillatorData<'a>,
    pub params: VolumeZoneOscillatorParams,
}

impl<'a> VolumeZoneOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: VolumeZoneOscillatorParams) -> Self {
        Self {
            data: VolumeZoneOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        close: &'a [f64],
        volume: &'a [f64],
        params: VolumeZoneOscillatorParams,
    ) -> Self {
        Self {
            data: VolumeZoneOscillatorData::Slices { close, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, VolumeZoneOscillatorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_intraday_smoothing(&self) -> bool {
        self.params
            .intraday_smoothing
            .unwrap_or(DEFAULT_INTRADAY_SMOOTHING)
    }

    #[inline]
    pub fn get_noise_filter(&self) -> usize {
        self.params.noise_filter.unwrap_or(DEFAULT_NOISE_FILTER)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VolumeZoneOscillatorBuilder {
    length: Option<usize>,
    intraday_smoothing: Option<bool>,
    noise_filter: Option<usize>,
    kernel: Kernel,
}

impl Default for VolumeZoneOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            intraday_smoothing: None,
            noise_filter: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeZoneOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline(always)]
    pub fn intraday_smoothing(mut self, value: bool) -> Self {
        self.intraday_smoothing = Some(value);
        self
    }

    #[inline(always)]
    pub fn noise_filter(mut self, value: usize) -> Self {
        self.noise_filter = Some(value);
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
    ) -> Result<VolumeZoneOscillatorOutput, VolumeZoneOscillatorError> {
        let input = VolumeZoneOscillatorInput::from_candles(
            candles,
            VolumeZoneOscillatorParams {
                length: self.length,
                intraday_smoothing: self.intraday_smoothing,
                noise_filter: self.noise_filter,
            },
        );
        volume_zone_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        close: &[f64],
        volume: &[f64],
    ) -> Result<VolumeZoneOscillatorOutput, VolumeZoneOscillatorError> {
        let input = VolumeZoneOscillatorInput::from_slices(
            close,
            volume,
            VolumeZoneOscillatorParams {
                length: self.length,
                intraday_smoothing: self.intraday_smoothing,
                noise_filter: self.noise_filter,
            },
        );
        volume_zone_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VolumeZoneOscillatorStream, VolumeZoneOscillatorError> {
        VolumeZoneOscillatorStream::try_new(VolumeZoneOscillatorParams {
            length: self.length,
            intraday_smoothing: self.intraday_smoothing,
            noise_filter: self.noise_filter,
        })
    }
}

#[derive(Debug, Error)]
pub enum VolumeZoneOscillatorError {
    #[error("volume_zone_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("volume_zone_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("volume_zone_oscillator: Inconsistent slice lengths: close={close_len}, volume={volume_len}")]
    InconsistentSliceLengths { close_len: usize, volume_len: usize },
    #[error("volume_zone_oscillator: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("volume_zone_oscillator: Invalid noise_filter: {noise_filter}")]
    InvalidNoiseFilter { noise_filter: usize },
    #[error("volume_zone_oscillator: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("volume_zone_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("volume_zone_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn validate_length(length: usize) -> Result<usize, VolumeZoneOscillatorError> {
    if length < 2 {
        return Err(VolumeZoneOscillatorError::InvalidLength { length });
    }
    Ok(length)
}

#[inline(always)]
fn validate_noise_filter(noise_filter: usize) -> Result<usize, VolumeZoneOscillatorError> {
    if noise_filter < 2 {
        return Err(VolumeZoneOscillatorError::InvalidNoiseFilter { noise_filter });
    }
    Ok(noise_filter)
}

#[inline(always)]
fn ema_alpha(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

#[inline(always)]
fn extract_close_volume<'a>(
    input: &'a VolumeZoneOscillatorInput<'a>,
) -> Result<(&'a [f64], &'a [f64], usize), VolumeZoneOscillatorError> {
    let (close, volume) = match &input.data {
        VolumeZoneOscillatorData::Candles { candles } => {
            (candles.close.as_slice(), candles.volume.as_slice())
        }
        VolumeZoneOscillatorData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(VolumeZoneOscillatorError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(VolumeZoneOscillatorError::InconsistentSliceLengths {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first_valid = volume
        .iter()
        .position(|v| v.is_finite())
        .ok_or(VolumeZoneOscillatorError::AllValuesNaN)?;
    Ok((close, volume, first_valid))
}

#[inline(always)]
fn compute_vzo_value(
    current_close: f64,
    prev_close: f64,
    volume: f64,
    ema_direction: &mut f64,
    ema_total: &mut f64,
    alpha: f64,
    beta: f64,
) -> Option<f64> {
    if !volume.is_finite() {
        return if *ema_total != 0.0 {
            Some(100.0 * *ema_direction / *ema_total)
        } else {
            None
        };
    }

    let directed =
        if current_close.is_finite() && prev_close.is_finite() && current_close > prev_close {
            volume
        } else {
            -volume
        };

    *ema_direction = beta.mul_add(*ema_direction, alpha * directed);
    *ema_total = beta.mul_add(*ema_total, alpha * volume);

    if *ema_total != 0.0 {
        Some(100.0 * *ema_direction / *ema_total)
    } else {
        None
    }
}

#[inline(always)]
fn compute_volume_zone_oscillator_into(
    close: &[f64],
    volume: &[f64],
    length: usize,
    intraday_smoothing: bool,
    noise_filter: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let alpha = ema_alpha(length);
    let beta = 1.0 - alpha;
    let smooth_alpha = ema_alpha(noise_filter);
    let smooth_beta = 1.0 - smooth_alpha;

    let mut prev_close = f64::NAN;
    let mut ema_direction = 0.0;
    let mut ema_total = 0.0;
    let mut smooth = 0.0;
    let mut smooth_valid = false;

    let warm = first_valid.min(out.len());
    for value in &mut out[..warm] {
        *value = f64::NAN;
    }

    for i in first_valid..close.len() {
        let raw = compute_vzo_value(
            close[i],
            prev_close,
            volume[i],
            &mut ema_direction,
            &mut ema_total,
            alpha,
            beta,
        );

        if close[i].is_finite() {
            prev_close = close[i];
        }

        if intraday_smoothing {
            if let Some(value) = raw {
                smooth = smooth_beta.mul_add(smooth, smooth_alpha * value);
                smooth_valid = true;
                out[i] = smooth;
            } else if smooth_valid {
                out[i] = smooth;
            } else {
                out[i] = f64::NAN;
            }
        } else {
            out[i] = raw.unwrap_or(f64::NAN);
        }
    }
}

#[inline]
pub fn volume_zone_oscillator(
    input: &VolumeZoneOscillatorInput,
) -> Result<VolumeZoneOscillatorOutput, VolumeZoneOscillatorError> {
    volume_zone_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn volume_zone_oscillator_with_kernel(
    input: &VolumeZoneOscillatorInput,
    kernel: Kernel,
) -> Result<VolumeZoneOscillatorOutput, VolumeZoneOscillatorError> {
    let (close, volume, first_valid) = extract_close_volume(input)?;
    let length = validate_length(input.get_length())?;
    let intraday_smoothing = input.get_intraday_smoothing();
    let noise_filter = validate_noise_filter(input.get_noise_filter())?;
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let _ = chosen;

    let mut out = alloc_with_nan_prefix(close.len(), first_valid);
    compute_volume_zone_oscillator_into(
        close,
        volume,
        length,
        intraday_smoothing,
        noise_filter,
        first_valid,
        &mut out,
    );
    Ok(VolumeZoneOscillatorOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn volume_zone_oscillator_into(
    input: &VolumeZoneOscillatorInput,
    out: &mut [f64],
) -> Result<(), VolumeZoneOscillatorError> {
    volume_zone_oscillator_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn volume_zone_oscillator_into_slice(
    out: &mut [f64],
    input: &VolumeZoneOscillatorInput,
    kernel: Kernel,
) -> Result<(), VolumeZoneOscillatorError> {
    let (close, volume, first_valid) = extract_close_volume(input)?;
    if out.len() != close.len() {
        return Err(VolumeZoneOscillatorError::OutputLengthMismatch {
            expected: close.len(),
            got: out.len(),
        });
    }
    let length = validate_length(input.get_length())?;
    let intraday_smoothing = input.get_intraday_smoothing();
    let noise_filter = validate_noise_filter(input.get_noise_filter())?;
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let _ = chosen;

    compute_volume_zone_oscillator_into(
        close,
        volume,
        length,
        intraday_smoothing,
        noise_filter,
        first_valid,
        out,
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub struct VolumeZoneOscillatorStream {
    alpha: f64,
    beta: f64,
    intraday_smoothing: bool,
    smooth_alpha: f64,
    smooth_beta: f64,
    prev_close: f64,
    ema_direction: f64,
    ema_total: f64,
    smooth: f64,
    smooth_valid: bool,
    seen_volume: bool,
}

impl VolumeZoneOscillatorStream {
    pub fn try_new(params: VolumeZoneOscillatorParams) -> Result<Self, VolumeZoneOscillatorError> {
        let length = validate_length(params.length.unwrap_or(DEFAULT_LENGTH))?;
        let intraday_smoothing = params
            .intraday_smoothing
            .unwrap_or(DEFAULT_INTRADAY_SMOOTHING);
        let noise_filter =
            validate_noise_filter(params.noise_filter.unwrap_or(DEFAULT_NOISE_FILTER))?;
        let alpha = ema_alpha(length);
        let smooth_alpha = ema_alpha(noise_filter);
        Ok(Self {
            alpha,
            beta: 1.0 - alpha,
            intraday_smoothing,
            smooth_alpha,
            smooth_beta: 1.0 - smooth_alpha,
            prev_close: f64::NAN,
            ema_direction: 0.0,
            ema_total: 0.0,
            smooth: 0.0,
            smooth_valid: false,
            seen_volume: false,
        })
    }

    #[inline]
    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        let raw = compute_vzo_value(
            close,
            self.prev_close,
            volume,
            &mut self.ema_direction,
            &mut self.ema_total,
            self.alpha,
            self.beta,
        );

        if volume.is_finite() {
            self.seen_volume = true;
        }
        if close.is_finite() {
            self.prev_close = close;
        }

        if self.intraday_smoothing {
            if let Some(value) = raw {
                self.smooth = self
                    .smooth_beta
                    .mul_add(self.smooth, self.smooth_alpha * value);
                self.smooth_valid = true;
                self.smooth
            } else if self.smooth_valid {
                self.smooth
            } else {
                f64::NAN
            }
        } else if self.seen_volume {
            raw.unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        0
    }
}

#[derive(Clone, Debug)]
pub struct VolumeZoneOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub noise_filter: (usize, usize, usize),
    pub intraday_smoothing: Option<bool>,
}

impl Default for VolumeZoneOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            noise_filter: (DEFAULT_NOISE_FILTER, DEFAULT_NOISE_FILTER, 0),
            intraday_smoothing: Some(DEFAULT_INTRADAY_SMOOTHING),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VolumeZoneOscillatorBatchBuilder {
    range: VolumeZoneOscillatorBatchRange,
    kernel: Kernel,
}

impl VolumeZoneOscillatorBatchBuilder {
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

    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    pub fn noise_filter_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.noise_filter = (start, end, step);
        self
    }

    pub fn noise_filter_static(mut self, value: usize) -> Self {
        self.range.noise_filter = (value, value, 0);
        self
    }

    pub fn intraday_smoothing(mut self, value: bool) -> Self {
        self.range.intraday_smoothing = Some(value);
        self
    }

    pub fn apply_slice(
        self,
        close: &[f64],
        volume: &[f64],
    ) -> Result<VolumeZoneOscillatorBatchOutput, VolumeZoneOscillatorError> {
        volume_zone_oscillator_batch_with_kernel(close, volume, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<VolumeZoneOscillatorBatchOutput, VolumeZoneOscillatorError> {
        self.apply_slice(&candles.close, &candles.volume)
    }
}

#[derive(Clone, Debug)]
pub struct VolumeZoneOscillatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VolumeZoneOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeZoneOscillatorBatchOutput {
    pub fn row_for_params(&self, params: &VolumeZoneOscillatorParams) -> Option<usize> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let intraday_smoothing = params
            .intraday_smoothing
            .unwrap_or(DEFAULT_INTRADAY_SMOOTHING);
        let noise_filter = params.noise_filter.unwrap_or(DEFAULT_NOISE_FILTER);
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == length
                && combo
                    .intraday_smoothing
                    .unwrap_or(DEFAULT_INTRADAY_SMOOTHING)
                    == intraday_smoothing
                && combo.noise_filter.unwrap_or(DEFAULT_NOISE_FILTER) == noise_filter
        })
    }

    pub fn values_for(&self, params: &VolumeZoneOscillatorParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.values.get(start..start + self.cols)
        })
    }
}

#[inline(always)]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, VolumeZoneOscillatorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(VolumeZoneOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            match x.checked_add(step) {
                Some(next) => x = next,
                None => break,
            }
        }
    } else {
        let mut x = start;
        while x >= end {
            out.push(x);
            match x.checked_sub(step) {
                Some(next) => x = next,
                None => break,
            }
            if x > start {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(VolumeZoneOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid(
    range: &VolumeZoneOscillatorBatchRange,
) -> Result<Vec<VolumeZoneOscillatorParams>, VolumeZoneOscillatorError> {
    let intraday_smoothing = range
        .intraday_smoothing
        .unwrap_or(DEFAULT_INTRADAY_SMOOTHING);
    let lengths = axis_usize(range.length)?;
    let noise_filters = axis_usize(range.noise_filter)?;
    let mut out = Vec::with_capacity(lengths.len() * noise_filters.len());
    for length in lengths {
        for &noise_filter in &noise_filters {
            out.push(VolumeZoneOscillatorParams {
                length: Some(length),
                intraday_smoothing: Some(intraday_smoothing),
                noise_filter: Some(noise_filter),
            });
        }
    }
    Ok(out)
}

pub fn volume_zone_oscillator_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeZoneOscillatorBatchRange,
    kernel: Kernel,
) -> Result<VolumeZoneOscillatorBatchOutput, VolumeZoneOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(VolumeZoneOscillatorError::InvalidKernelForBatch(kernel)),
    };
    volume_zone_oscillator_batch_par_slice(close, volume, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn volume_zone_oscillator_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeZoneOscillatorBatchRange,
    kernel: Kernel,
) -> Result<VolumeZoneOscillatorBatchOutput, VolumeZoneOscillatorError> {
    volume_zone_oscillator_batch_inner(close, volume, sweep, kernel, false)
}

#[inline(always)]
pub fn volume_zone_oscillator_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeZoneOscillatorBatchRange,
    kernel: Kernel,
) -> Result<VolumeZoneOscillatorBatchOutput, VolumeZoneOscillatorError> {
    volume_zone_oscillator_batch_inner(close, volume, sweep, kernel, true)
}

fn volume_zone_oscillator_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeZoneOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<VolumeZoneOscillatorBatchOutput, VolumeZoneOscillatorError> {
    let combos = expand_grid(sweep)?;
    if close.is_empty() || volume.is_empty() {
        return Err(VolumeZoneOscillatorError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(VolumeZoneOscillatorError::InconsistentSliceLengths {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first_valid = volume
        .iter()
        .position(|v| v.is_finite())
        .ok_or(VolumeZoneOscillatorError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = close.len();

    let warmups = vec![first_valid; rows];
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);
    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    volume_zone_oscillator_batch_inner_into(close, volume, sweep, kernel, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(VolumeZoneOscillatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn volume_zone_oscillator_batch_into_slice(
    out: &mut [f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeZoneOscillatorBatchRange,
    kernel: Kernel,
) -> Result<(), VolumeZoneOscillatorError> {
    volume_zone_oscillator_batch_inner_into(close, volume, sweep, kernel, false, out)?;
    Ok(())
}

fn volume_zone_oscillator_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeZoneOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VolumeZoneOscillatorParams>, VolumeZoneOscillatorError> {
    let combos = expand_grid(sweep)?;
    if close.is_empty() || volume.is_empty() {
        return Err(VolumeZoneOscillatorError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(VolumeZoneOscillatorError::InconsistentSliceLengths {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first_valid = volume
        .iter()
        .position(|v| v.is_finite())
        .ok_or(VolumeZoneOscillatorError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = close.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| VolumeZoneOscillatorError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    if out.len() != expected {
        return Err(VolumeZoneOscillatorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let _ = chosen;

    let lengths: Vec<usize> = combos
        .iter()
        .map(|combo| validate_length(combo.length.unwrap_or(DEFAULT_LENGTH)))
        .collect::<Result<_, _>>()?;
    let intraday_flags: Vec<bool> = combos
        .iter()
        .map(|combo| {
            combo
                .intraday_smoothing
                .unwrap_or(DEFAULT_INTRADAY_SMOOTHING)
        })
        .collect();
    let noise_filters: Vec<usize> = combos
        .iter()
        .map(|combo| validate_noise_filter(combo.noise_filter.unwrap_or(DEFAULT_NOISE_FILTER)))
        .collect::<Result<_, _>>()?;

    for row in 0..rows {
        out[row * cols..row * cols + first_valid.min(cols)].fill(f64::NAN);
    }

    let do_row = |row: usize, dst: &mut [f64]| -> Result<(), VolumeZoneOscillatorError> {
        compute_volume_zone_oscillator_into(
            close,
            volume,
            lengths[row],
            intraday_flags[row],
            noise_filters[row],
            first_valid,
            dst,
        );
        Ok(())
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
#[pyfunction(name = "volume_zone_oscillator")]
#[pyo3(signature = (close, volume, length=14, intraday_smoothing=true, noise_filter=4, kernel=None))]
pub fn volume_zone_oscillator_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length: usize,
    intraday_smoothing: bool,
    noise_filter: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = VolumeZoneOscillatorInput::from_slices(
        close,
        volume,
        VolumeZoneOscillatorParams {
            length: Some(length),
            intraday_smoothing: Some(intraday_smoothing),
            noise_filter: Some(noise_filter),
        },
    );
    let out = py
        .allow_threads(|| volume_zone_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VolumeZoneOscillatorStream")]
pub struct VolumeZoneOscillatorStreamPy {
    stream: VolumeZoneOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolumeZoneOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=14, intraday_smoothing=true, noise_filter=4))]
    fn new(length: usize, intraday_smoothing: bool, noise_filter: usize) -> PyResult<Self> {
        let stream = VolumeZoneOscillatorStream::try_new(VolumeZoneOscillatorParams {
            length: Some(length),
            intraday_smoothing: Some(intraday_smoothing),
            noise_filter: Some(noise_filter),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, close: f64, volume: f64) -> f64 {
        self.stream.update(close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_zone_oscillator_batch")]
#[pyo3(signature = (close, volume, length_range=(14,14,0), intraday_smoothing=true, noise_filter_range=(4,4,0), kernel=None))]
pub fn volume_zone_oscillator_batch_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    intraday_smoothing: bool,
    noise_filter_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let sweep = VolumeZoneOscillatorBatchRange {
        length: length_range,
        noise_filter: noise_filter_range,
        intraday_smoothing: Some(intraday_smoothing),
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let values_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let values_out = unsafe { values_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        volume_zone_oscillator_batch_inner_into(
            close,
            volume,
            &sweep,
            batch.to_non_batch(),
            true,
            values_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", values_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "intraday_smoothing",
        combos
            .iter()
            .map(|p| p.intraday_smoothing.unwrap_or(DEFAULT_INTRADAY_SMOOTHING))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "noise_filters",
        combos
            .iter()
            .map(|p| p.noise_filter.unwrap_or(DEFAULT_NOISE_FILTER) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_volume_zone_oscillator_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volume_zone_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(volume_zone_oscillator_batch_py, m)?)?;
    m.add_class::<VolumeZoneOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volume_zone_oscillator_js")]
pub fn volume_zone_oscillator_js(
    close: &[f64],
    volume: &[f64],
    length: usize,
    intraday_smoothing: bool,
    noise_filter: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = VolumeZoneOscillatorInput::from_slices(
        close,
        volume,
        VolumeZoneOscillatorParams {
            length: Some(length),
            intraday_smoothing: Some(intraday_smoothing),
            noise_filter: Some(noise_filter),
        },
    );
    let mut out = vec![0.0; close.len()];
    volume_zone_oscillator_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeZoneOscillatorBatchConfig {
    pub length_range: Vec<f64>,
    pub intraday_smoothing: bool,
    pub noise_filter_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeZoneOscillatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VolumeZoneOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (i, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a finite non-negative whole number"
            )));
        }
        let rounded = value.round();
        if (value - rounded).abs() > 1e-9 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a whole number"
            )));
        }
        out[i] = rounded as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volume_zone_oscillator_batch_js")]
pub fn volume_zone_oscillator_batch_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VolumeZoneOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let length = vec3_to_usize("length_range", &config.length_range)?;
    let noise_filter = vec3_to_usize("noise_filter_range", &config.noise_filter_range)?;
    let out = volume_zone_oscillator_batch_with_kernel(
        close,
        volume,
        &VolumeZoneOscillatorBatchRange {
            length,
            noise_filter,
            intraday_smoothing: Some(config.intraday_smoothing),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VolumeZoneOscillatorBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_zone_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_zone_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_zone_oscillator_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    intraday_smoothing: bool,
    noise_filter: usize,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = VolumeZoneOscillatorInput::from_slices(
            close,
            volume,
            VolumeZoneOscillatorParams {
                length: Some(length),
                intraday_smoothing: Some(intraday_smoothing),
                noise_filter: Some(noise_filter),
            },
        );
        volume_zone_oscillator_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_zone_oscillator_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    intraday_smoothing: bool,
    noise_filter_start: usize,
    noise_filter_end: usize,
    noise_filter_step: usize,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volume_zone_oscillator_batch_into",
        ));
    }
    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let sweep = VolumeZoneOscillatorBatchRange {
            length: (length_start, length_end, length_step),
            noise_filter: (noise_filter_start, noise_filter_end, noise_filter_step),
            intraday_smoothing: Some(intraday_smoothing),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        volume_zone_oscillator_batch_into_slice(out, close, volume, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_zone_oscillator_output_into_js(
    close: &[f64],
    volume: &[f64],
    length: usize,
    intraday_smoothing: bool,
    noise_filter: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values =
        volume_zone_oscillator_js(close, volume, length, intraday_smoothing, noise_filter)?;
    crate::write_wasm_f64_output("volume_zone_oscillator_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_zone_oscillator_batch_output_into_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_zone_oscillator_batch_js(close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "volume_zone_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_series(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol,
                "mismatch at {i}: {a} vs {b}"
            );
        }
    }

    fn manual_vzo(
        close: &[f64],
        volume: &[f64],
        length: usize,
        intraday_smoothing: bool,
        noise_filter: usize,
    ) -> Vec<f64> {
        let mut out = vec![f64::NAN; close.len()];
        let alpha = ema_alpha(length);
        let beta = 1.0 - alpha;
        let smooth_alpha = ema_alpha(noise_filter);
        let smooth_beta = 1.0 - smooth_alpha;
        let mut prev_close = f64::NAN;
        let mut ema_dir = 0.0;
        let mut ema_total = 0.0;
        let mut smooth = 0.0;
        let mut smooth_valid = false;
        let first_valid = volume.iter().position(|v| v.is_finite()).unwrap();
        for i in first_valid..close.len() {
            let raw = compute_vzo_value(
                close[i],
                prev_close,
                volume[i],
                &mut ema_dir,
                &mut ema_total,
                alpha,
                beta,
            );
            if close[i].is_finite() {
                prev_close = close[i];
            }
            if intraday_smoothing {
                if let Some(value) = raw {
                    smooth = smooth_beta.mul_add(smooth, smooth_alpha * value);
                    smooth_valid = true;
                    out[i] = smooth;
                } else if smooth_valid {
                    out[i] = smooth;
                }
            } else {
                out[i] = raw.unwrap_or(f64::NAN);
            }
        }
        out
    }

    #[test]
    fn volume_zone_oscillator_matches_manual_reference() {
        let close = [10.0, 10.5, 10.25, 10.75, 10.2, 10.9, 10.95];
        let volume = [100.0, 120.0, 80.0, 140.0, 90.0, 160.0, 110.0];
        let input = VolumeZoneOscillatorInput::from_slices(
            &close,
            &volume,
            VolumeZoneOscillatorParams {
                length: Some(4),
                intraday_smoothing: Some(true),
                noise_filter: Some(3),
            },
        );
        let out = volume_zone_oscillator(&input).unwrap();
        let expected = manual_vzo(&close, &volume, 4, true, 3);
        assert_close_series(&out.values, &expected, 1e-12);
    }

    #[test]
    fn volume_zone_oscillator_stream_matches_batch() {
        let close = [10.0, 10.5, 10.25, 10.75, 10.2, 10.9, 10.95, 11.1];
        let volume = [100.0, 120.0, 80.0, 140.0, 90.0, 160.0, 110.0, 170.0];
        let input = VolumeZoneOscillatorInput::from_slices(
            &close,
            &volume,
            VolumeZoneOscillatorParams {
                length: Some(4),
                intraday_smoothing: Some(false),
                noise_filter: Some(3),
            },
        );
        let batch = volume_zone_oscillator(&input).unwrap();
        let mut stream = VolumeZoneOscillatorStream::try_new(input.params.clone()).unwrap();
        let streamed: Vec<f64> = close
            .iter()
            .zip(volume.iter())
            .map(|(&c, &v)| stream.update(c, v))
            .collect();
        assert_close_series(&streamed, &batch.values, 1e-12);
    }

    #[test]
    fn volume_zone_oscillator_batch_rows_match_single() {
        let close = [10.0, 10.5, 10.25, 10.75, 10.2, 10.9, 10.95, 11.1];
        let volume = [100.0, 120.0, 80.0, 140.0, 90.0, 160.0, 110.0, 170.0];
        let sweep = VolumeZoneOscillatorBatchRange {
            length: (4, 6, 2),
            noise_filter: (3, 3, 0),
            intraday_smoothing: Some(true),
        };
        let batch = volume_zone_oscillator_batch_with_kernel(&close, &volume, &sweep, Kernel::Auto)
            .unwrap();
        let single = volume_zone_oscillator(&VolumeZoneOscillatorInput::from_slices(
            &close,
            &volume,
            VolumeZoneOscillatorParams {
                length: Some(4),
                intraday_smoothing: Some(true),
                noise_filter: Some(3),
            },
        ))
        .unwrap();
        assert_close_series(&batch.values[..close.len()], &single.values, 1e-12);
    }

    #[test]
    fn volume_zone_oscillator_into_slice_matches_single() {
        let close = [10.0, 10.5, 10.25, 10.75, 10.2, 10.9];
        let volume = [100.0, 120.0, 80.0, 140.0, 90.0, 160.0];
        let input = VolumeZoneOscillatorInput::from_slices(
            &close,
            &volume,
            VolumeZoneOscillatorParams::default(),
        );
        let direct = volume_zone_oscillator(&input).unwrap();
        let mut out = vec![0.0; close.len()];
        volume_zone_oscillator_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_close_series(&out, &direct.values, 1e-12);
    }

    #[test]
    fn volume_zone_oscillator_invalid_length_errors() {
        let close = [10.0, 10.5, 10.25];
        let volume = [100.0, 120.0, 80.0];
        let input = VolumeZoneOscillatorInput::from_slices(
            &close,
            &volume,
            VolumeZoneOscillatorParams {
                length: Some(1),
                intraday_smoothing: Some(true),
                noise_filter: Some(4),
            },
        );
        let err = volume_zone_oscillator(&input).unwrap_err();
        match err {
            VolumeZoneOscillatorError::InvalidLength { length } => assert_eq!(length, 1),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
