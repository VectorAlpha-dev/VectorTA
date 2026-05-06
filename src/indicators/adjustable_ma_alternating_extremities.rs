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
use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_LENGTH: usize = 50;
const DEFAULT_MULT: f64 = 2.0;
const DEFAULT_ALPHA: f64 = 1.0;
const DEFAULT_BETA: f64 = 0.5;
const TWO_PI: f64 = core::f64::consts::PI * 2.0;
const WEIGHT_SUM_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub enum AdjustableMaAlternatingExtremitiesData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdjustableMaAlternatingExtremitiesOutput {
    pub ma: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub extremity: Vec<f64>,
    pub state: Vec<f64>,
    pub changed: Vec<f64>,
    pub smoothed_open: Vec<f64>,
    pub smoothed_high: Vec<f64>,
    pub smoothed_low: Vec<f64>,
    pub smoothed_close: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub enum AdjustableMaAlternatingExtremitiesOutputField {
    Ma,
    Upper,
    Lower,
    Extremity,
    State,
    Changed,
    SmoothedOpen,
    SmoothedHigh,
    SmoothedLow,
    SmoothedClose,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdjustableMaAlternatingExtremitiesParams {
    pub length: Option<usize>,
    pub mult: Option<f64>,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
}

impl Default for AdjustableMaAlternatingExtremitiesParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            mult: Some(DEFAULT_MULT),
            alpha: Some(DEFAULT_ALPHA),
            beta: Some(DEFAULT_BETA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdjustableMaAlternatingExtremitiesInput<'a> {
    pub data: AdjustableMaAlternatingExtremitiesData<'a>,
    pub params: AdjustableMaAlternatingExtremitiesParams,
}

impl<'a> AdjustableMaAlternatingExtremitiesInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        params: AdjustableMaAlternatingExtremitiesParams,
    ) -> Self {
        Self {
            data: AdjustableMaAlternatingExtremitiesData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AdjustableMaAlternatingExtremitiesParams,
    ) -> Self {
        Self {
            data: AdjustableMaAlternatingExtremitiesData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AdjustableMaAlternatingExtremitiesParams::default())
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }
    #[inline(always)]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(DEFAULT_MULT)
    }
    #[inline(always)]
    pub fn get_alpha(&self) -> f64 {
        self.params.alpha.unwrap_or(DEFAULT_ALPHA)
    }
    #[inline(always)]
    pub fn get_beta(&self) -> f64 {
        self.params.beta.unwrap_or(DEFAULT_BETA)
    }
}

#[derive(Clone, Debug)]
pub struct AdjustableMaAlternatingExtremitiesBuilder {
    length: Option<usize>,
    mult: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    kernel: Kernel,
}

impl Default for AdjustableMaAlternatingExtremitiesBuilder {
    fn default() -> Self {
        Self {
            length: None,
            mult: None,
            alpha: None,
            beta: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdjustableMaAlternatingExtremitiesBuilder {
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
    pub fn mult(mut self, value: f64) -> Self {
        self.mult = Some(value);
        self
    }
    #[inline(always)]
    pub fn alpha(mut self, value: f64) -> Self {
        self.alpha = Some(value);
        self
    }
    #[inline(always)]
    pub fn beta(mut self, value: f64) -> Self {
        self.beta = Some(value);
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
    ) -> Result<AdjustableMaAlternatingExtremitiesOutput, AdjustableMaAlternatingExtremitiesError>
    {
        let input = AdjustableMaAlternatingExtremitiesInput::from_candles(
            candles,
            AdjustableMaAlternatingExtremitiesParams {
                length: self.length,
                mult: self.mult,
                alpha: self.alpha,
                beta: self.beta,
            },
        );
        adjustable_ma_alternating_extremities_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AdjustableMaAlternatingExtremitiesOutput, AdjustableMaAlternatingExtremitiesError>
    {
        let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
            high,
            low,
            close,
            AdjustableMaAlternatingExtremitiesParams {
                length: self.length,
                mult: self.mult,
                alpha: self.alpha,
                beta: self.beta,
            },
        );
        adjustable_ma_alternating_extremities_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<AdjustableMaAlternatingExtremitiesStream, AdjustableMaAlternatingExtremitiesError>
    {
        AdjustableMaAlternatingExtremitiesStream::try_new(
            AdjustableMaAlternatingExtremitiesParams {
                length: self.length,
                mult: self.mult,
                alpha: self.alpha,
                beta: self.beta,
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum AdjustableMaAlternatingExtremitiesError {
    #[error("adjustable_ma_alternating_extremities: input data slice is empty")]
    EmptyInputData,
    #[error("adjustable_ma_alternating_extremities: data length mismatch: high={high}, low={low}, close={close}")]
    DataLengthMismatch {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("adjustable_ma_alternating_extremities: all values are NaN")]
    AllValuesNaN,
    #[error("adjustable_ma_alternating_extremities: invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("adjustable_ma_alternating_extremities: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("adjustable_ma_alternating_extremities: invalid mult: {mult}")]
    InvalidMult { mult: f64 },
    #[error("adjustable_ma_alternating_extremities: invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error("adjustable_ma_alternating_extremities: invalid beta: {beta}")]
    InvalidBeta { beta: f64 },
    #[error("adjustable_ma_alternating_extremities: degenerate kernel weights for alpha={alpha}, beta={beta}")]
    DegenerateKernel { alpha: f64, beta: f64 },
    #[error("adjustable_ma_alternating_extremities: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("adjustable_ma_alternating_extremities: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("adjustable_ma_alternating_extremities: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct OutputWarmups {
    ma: usize,
    open: usize,
    bands: usize,
}

#[derive(Clone)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    len: usize,
    length: usize,
    mult: f64,
    weights: Vec<f64>,
    first: usize,
    warmups: OutputWarmups,
    kernel: Kernel,
}

#[inline]
pub fn adjustable_ma_alternating_extremities(
    input: &AdjustableMaAlternatingExtremitiesInput<'_>,
) -> Result<AdjustableMaAlternatingExtremitiesOutput, AdjustableMaAlternatingExtremitiesError> {
    adjustable_ma_alternating_extremities_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn adjustable_ma_alternating_extremities_with_kernel(
    input: &AdjustableMaAlternatingExtremitiesInput<'_>,
    kernel: Kernel,
) -> Result<AdjustableMaAlternatingExtremitiesOutput, AdjustableMaAlternatingExtremitiesError> {
    let prepared = prepare_input(input, kernel)?;

    let mut ma = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);
    let mut upper = alloc_with_nan_prefix(prepared.len, prepared.warmups.bands);
    let mut lower = alloc_with_nan_prefix(prepared.len, prepared.warmups.bands);
    let mut extremity = alloc_with_nan_prefix(prepared.len, prepared.warmups.bands);
    let mut state = alloc_with_nan_prefix(prepared.len, prepared.warmups.bands);
    let mut changed = alloc_with_nan_prefix(prepared.len, prepared.warmups.bands);
    let mut smoothed_open = alloc_with_nan_prefix(prepared.len, prepared.warmups.open);
    let mut smoothed_high = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);
    let mut smoothed_low = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);
    let mut smoothed_close = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);

    compute_into_slices(
        &prepared,
        &mut ma,
        &mut upper,
        &mut lower,
        &mut extremity,
        &mut state,
        &mut changed,
        &mut smoothed_open,
        &mut smoothed_high,
        &mut smoothed_low,
        &mut smoothed_close,
    );

    Ok(AdjustableMaAlternatingExtremitiesOutput {
        ma,
        upper,
        lower,
        extremity,
        state,
        changed,
        smoothed_open,
        smoothed_high,
        smoothed_low,
        smoothed_close,
    })
}

#[inline]
pub fn adjustable_ma_alternating_extremities_into(
    input: &AdjustableMaAlternatingExtremitiesInput<'_>,
    ma: &mut [f64],
    upper: &mut [f64],
    lower: &mut [f64],
    extremity: &mut [f64],
    state: &mut [f64],
    changed: &mut [f64],
    smoothed_open: &mut [f64],
    smoothed_high: &mut [f64],
    smoothed_low: &mut [f64],
    smoothed_close: &mut [f64],
) -> Result<(), AdjustableMaAlternatingExtremitiesError> {
    adjustable_ma_alternating_extremities_into_slices(
        input,
        Kernel::Auto,
        ma,
        upper,
        lower,
        extremity,
        state,
        changed,
        smoothed_open,
        smoothed_high,
        smoothed_low,
        smoothed_close,
    )
}

#[inline]
pub fn adjustable_ma_alternating_extremities_into_slices(
    input: &AdjustableMaAlternatingExtremitiesInput<'_>,
    kernel: Kernel,
    ma: &mut [f64],
    upper: &mut [f64],
    lower: &mut [f64],
    extremity: &mut [f64],
    state: &mut [f64],
    changed: &mut [f64],
    smoothed_open: &mut [f64],
    smoothed_high: &mut [f64],
    smoothed_low: &mut [f64],
    smoothed_close: &mut [f64],
) -> Result<(), AdjustableMaAlternatingExtremitiesError> {
    let prepared = prepare_input(input, kernel)?;
    let expected = prepared.len;
    for out in [
        ma.len(),
        upper.len(),
        lower.len(),
        extremity.len(),
        state.len(),
        changed.len(),
        smoothed_open.len(),
        smoothed_high.len(),
        smoothed_low.len(),
        smoothed_close.len(),
    ] {
        if out != expected {
            return Err(
                AdjustableMaAlternatingExtremitiesError::OutputLengthMismatch {
                    expected,
                    got: out,
                },
            );
        }
    }
    ma.fill(f64::NAN);
    upper.fill(f64::NAN);
    lower.fill(f64::NAN);
    extremity.fill(f64::NAN);
    state.fill(f64::NAN);
    changed.fill(f64::NAN);
    smoothed_open.fill(f64::NAN);
    smoothed_high.fill(f64::NAN);
    smoothed_low.fill(f64::NAN);
    smoothed_close.fill(f64::NAN);
    compute_into_slices(
        &prepared,
        ma,
        upper,
        lower,
        extremity,
        state,
        changed,
        smoothed_open,
        smoothed_high,
        smoothed_low,
        smoothed_close,
    );
    Ok(())
}

#[inline]
pub fn adjustable_ma_alternating_extremities_output_into_slice(
    dst: &mut [f64],
    input: &AdjustableMaAlternatingExtremitiesInput<'_>,
    kernel: Kernel,
    field: AdjustableMaAlternatingExtremitiesOutputField,
) -> Result<(), AdjustableMaAlternatingExtremitiesError> {
    let prepared = prepare_input(input, kernel)?;
    if dst.len() != prepared.len {
        return Err(
            AdjustableMaAlternatingExtremitiesError::OutputLengthMismatch {
                expected: prepared.len,
                got: dst.len(),
            },
        );
    }

    dst.fill(f64::NAN);
    let _ = prepared.kernel;
    match field {
        AdjustableMaAlternatingExtremitiesOutputField::Ma
        | AdjustableMaAlternatingExtremitiesOutputField::SmoothedClose => {
            weighted_filter_into(
                prepared.close,
                prepared.first,
                prepared.length,
                &prepared.weights,
                dst,
            );
        }
        AdjustableMaAlternatingExtremitiesOutputField::SmoothedHigh => {
            weighted_filter_into(
                prepared.high,
                prepared.first,
                prepared.length,
                &prepared.weights,
                dst,
            );
        }
        AdjustableMaAlternatingExtremitiesOutputField::SmoothedLow => {
            weighted_filter_into(
                prepared.low,
                prepared.first,
                prepared.length,
                &prepared.weights,
                dst,
            );
        }
        AdjustableMaAlternatingExtremitiesOutputField::SmoothedOpen => {
            let mut ma = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);
            weighted_filter_into(
                prepared.close,
                prepared.first,
                prepared.length,
                &prepared.weights,
                &mut ma,
            );
            compute_smoothed_open(&ma, prepared.warmups.ma, dst);
        }
        AdjustableMaAlternatingExtremitiesOutputField::Upper => {
            let mut ma = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);
            weighted_filter_into(
                prepared.close,
                prepared.first,
                prepared.length,
                &prepared.weights,
                &mut ma,
            );
            compute_selected_deviation_band(&prepared, &ma, dst, true);
        }
        AdjustableMaAlternatingExtremitiesOutputField::Lower => {
            let mut ma = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);
            weighted_filter_into(
                prepared.close,
                prepared.first,
                prepared.length,
                &prepared.weights,
                &mut ma,
            );
            compute_selected_deviation_band(&prepared, &ma, dst, false);
        }
        AdjustableMaAlternatingExtremitiesOutputField::Extremity
        | AdjustableMaAlternatingExtremitiesOutputField::State
        | AdjustableMaAlternatingExtremitiesOutputField::Changed => {
            let mut ma = alloc_with_nan_prefix(prepared.len, prepared.warmups.ma);
            let mut upper = alloc_with_nan_prefix(prepared.len, prepared.warmups.bands);
            let mut lower = alloc_with_nan_prefix(prepared.len, prepared.warmups.bands);
            weighted_filter_into(
                prepared.close,
                prepared.first,
                prepared.length,
                &prepared.weights,
                &mut ma,
            );
            compute_deviation_bands(&prepared, &ma, &mut upper, &mut lower);
            compute_selected_state_output(&prepared, &upper, &lower, dst, field);
        }
    }
    Ok(())
}

#[inline]
fn resolve_data<'a>(
    input: &'a AdjustableMaAlternatingExtremitiesInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), AdjustableMaAlternatingExtremitiesError> {
    match &input.data {
        AdjustableMaAlternatingExtremitiesData::Candles { candles } => Ok((
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        AdjustableMaAlternatingExtremitiesData::Slices { high, low, close } => {
            if high.len() != low.len() || high.len() != close.len() {
                return Err(
                    AdjustableMaAlternatingExtremitiesError::DataLengthMismatch {
                        high: high.len(),
                        low: low.len(),
                        close: close.len(),
                    },
                );
            }
            Ok((high, low, close))
        }
    }
}

#[inline]
fn prepare_input<'a>(
    input: &'a AdjustableMaAlternatingExtremitiesInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, AdjustableMaAlternatingExtremitiesError> {
    let (high, low, close) = resolve_data(input)?;
    let len = close.len();
    if len == 0 {
        return Err(AdjustableMaAlternatingExtremitiesError::EmptyInputData);
    }
    let first = (0..len)
        .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .ok_or(AdjustableMaAlternatingExtremitiesError::AllValuesNaN)?;

    let length = input.get_length();
    let mult = input.get_mult();
    let alpha = input.get_alpha();
    let beta = input.get_beta();

    if length < 2 || length > len {
        return Err(AdjustableMaAlternatingExtremitiesError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if !mult.is_finite() || mult < 1.0 {
        return Err(AdjustableMaAlternatingExtremitiesError::InvalidMult { mult });
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(AdjustableMaAlternatingExtremitiesError::InvalidAlpha { alpha });
    }
    if !beta.is_finite() || beta < 0.0 {
        return Err(AdjustableMaAlternatingExtremitiesError::InvalidBeta { beta });
    }
    let needed = (length * 2) - 1;
    if len - first < needed {
        return Err(
            AdjustableMaAlternatingExtremitiesError::NotEnoughValidData {
                needed,
                valid: len - first,
            },
        );
    }

    let weights = build_weights(length, alpha, beta)?;
    let ma_warm = first + length - 1;
    Ok(PreparedInput {
        high,
        low,
        close,
        len,
        length,
        mult,
        weights,
        first,
        warmups: OutputWarmups {
            ma: ma_warm,
            open: ma_warm + 2,
            bands: first + (length * 2) - 2,
        },
        kernel: kernel.to_non_batch(),
    })
}

#[inline]
fn build_weights(
    length: usize,
    alpha: f64,
    beta: f64,
) -> Result<Vec<f64>, AdjustableMaAlternatingExtremitiesError> {
    let denom = (length - 1) as f64;
    let mut weights = Vec::with_capacity(length);
    let mut sum = 0.0;
    for i in 0..length {
        let x = i as f64 / denom;
        let w = (TWO_PI * x.powf(alpha)).sin() * (1.0 - x.powf(beta));
        weights.push(w);
        sum += w;
    }
    if !sum.is_finite() || sum.abs() <= WEIGHT_SUM_EPS {
        return Err(AdjustableMaAlternatingExtremitiesError::DegenerateKernel { alpha, beta });
    }
    let inv_sum = 1.0 / sum;
    for weight in &mut weights {
        *weight *= inv_sum;
    }
    Ok(weights)
}

#[inline]
fn compute_into_slices(
    prepared: &PreparedInput<'_>,
    ma: &mut [f64],
    upper: &mut [f64],
    lower: &mut [f64],
    extremity: &mut [f64],
    state: &mut [f64],
    changed: &mut [f64],
    smoothed_open: &mut [f64],
    smoothed_high: &mut [f64],
    smoothed_low: &mut [f64],
    smoothed_close: &mut [f64],
) {
    let _ = prepared.kernel;
    weighted_filter_into(
        prepared.close,
        prepared.first,
        prepared.length,
        &prepared.weights,
        ma,
    );
    weighted_filter_into(
        prepared.high,
        prepared.first,
        prepared.length,
        &prepared.weights,
        smoothed_high,
    );
    weighted_filter_into(
        prepared.low,
        prepared.first,
        prepared.length,
        &prepared.weights,
        smoothed_low,
    );
    smoothed_close.copy_from_slice(ma);
    compute_smoothed_open(ma, prepared.warmups.ma, smoothed_open);
    compute_deviation_bands(prepared, ma, upper, lower);
    compute_state_and_extremity(prepared, upper, lower, extremity, state, changed);
}

#[inline]
fn weighted_filter_into(
    source: &[f64],
    first: usize,
    length: usize,
    weights: &[f64],
    out: &mut [f64],
) {
    let start = first + length - 1;
    for i in start..source.len() {
        let mut acc = 0.0;
        for j in 0..length {
            acc += source[i - j] * weights[j];
        }
        out[i] = acc;
    }
}

#[inline]
fn compute_smoothed_open(smoothed_close: &[f64], ma_warm: usize, out: &mut [f64]) {
    let start = ma_warm + 2;
    if start >= smoothed_close.len() {
        return;
    }
    for i in start..smoothed_close.len() {
        out[i] = 0.5 * (smoothed_close[i - 1] + smoothed_close[i - 2]);
    }
}

#[inline]
fn compute_deviation_bands(
    prepared: &PreparedInput<'_>,
    ma: &[f64],
    upper: &mut [f64],
    lower: &mut [f64],
) {
    let ma_start = prepared.warmups.ma;
    let band_start = prepared.warmups.bands;
    let mut rolling = 0.0;
    for i in ma_start..=band_start {
        rolling += (prepared.close[i] - ma[i]).abs();
    }
    let first_dev = (rolling / prepared.length as f64) * prepared.mult;
    upper[band_start] = ma[band_start] + first_dev;
    lower[band_start] = ma[band_start] - first_dev;
    for i in (band_start + 1)..prepared.len {
        rolling += (prepared.close[i] - ma[i]).abs();
        rolling -= (prepared.close[i - prepared.length] - ma[i - prepared.length]).abs();
        let dev = (rolling / prepared.length as f64) * prepared.mult;
        upper[i] = ma[i] + dev;
        lower[i] = ma[i] - dev;
    }
}

#[inline]
fn compute_selected_deviation_band(
    prepared: &PreparedInput<'_>,
    ma: &[f64],
    out: &mut [f64],
    upper: bool,
) {
    let ma_start = prepared.warmups.ma;
    let band_start = prepared.warmups.bands;
    let mut rolling = 0.0;
    for i in ma_start..=band_start {
        rolling += (prepared.close[i] - ma[i]).abs();
    }
    let first_dev = (rolling / prepared.length as f64) * prepared.mult;
    out[band_start] = if upper {
        ma[band_start] + first_dev
    } else {
        ma[band_start] - first_dev
    };
    for i in (band_start + 1)..prepared.len {
        rolling += (prepared.close[i] - ma[i]).abs();
        rolling -= (prepared.close[i - prepared.length] - ma[i - prepared.length]).abs();
        let dev = (rolling / prepared.length as f64) * prepared.mult;
        out[i] = if upper { ma[i] + dev } else { ma[i] - dev };
    }
}

#[inline]
fn pine_cross(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    if !(prev_a.is_finite() && prev_b.is_finite() && curr_a.is_finite() && curr_b.is_finite()) {
        return false;
    }
    (curr_a > curr_b && prev_a <= prev_b) || (curr_a < curr_b && prev_a >= prev_b)
}

#[inline]
fn compute_state_and_extremity(
    prepared: &PreparedInput<'_>,
    upper: &[f64],
    lower: &[f64],
    extremity: &mut [f64],
    state: &mut [f64],
    changed: &mut [f64],
) {
    let start = prepared.warmups.bands;
    state[start] = 0.0;
    changed[start] = 0.0;
    extremity[start] = lower[start];
    for i in (start + 1)..prepared.len {
        let prev_state = state[i - 1];
        let cross_high = pine_cross(
            prepared.high[i - 1],
            upper[i - 1],
            prepared.high[i],
            upper[i],
        );
        let cross_low = pine_cross(prepared.low[i - 1], lower[i - 1], prepared.low[i], lower[i]);
        let next_state = if cross_high {
            1.0
        } else if cross_low {
            0.0
        } else {
            prev_state
        };
        state[i] = next_state;
        changed[i] = if (next_state - prev_state).abs() > 0.0 {
            1.0
        } else {
            0.0
        };
        extremity[i] = if next_state >= 0.5 {
            upper[i]
        } else {
            lower[i]
        };
    }
}

#[inline]
fn compute_selected_state_output(
    prepared: &PreparedInput<'_>,
    upper: &[f64],
    lower: &[f64],
    out: &mut [f64],
    field: AdjustableMaAlternatingExtremitiesOutputField,
) {
    let start = prepared.warmups.bands;
    let mut prev_state = 0.0;
    out[start] = match field {
        AdjustableMaAlternatingExtremitiesOutputField::Extremity => lower[start],
        AdjustableMaAlternatingExtremitiesOutputField::State
        | AdjustableMaAlternatingExtremitiesOutputField::Changed => 0.0,
        _ => unreachable!(),
    };
    for i in (start + 1)..prepared.len {
        let cross_high = pine_cross(
            prepared.high[i - 1],
            upper[i - 1],
            prepared.high[i],
            upper[i],
        );
        let cross_low = pine_cross(prepared.low[i - 1], lower[i - 1], prepared.low[i], lower[i]);
        let next_state = if cross_high {
            1.0
        } else if cross_low {
            0.0
        } else {
            prev_state
        };
        out[i] = match field {
            AdjustableMaAlternatingExtremitiesOutputField::Extremity => {
                if next_state >= 0.5 {
                    upper[i]
                } else {
                    lower[i]
                }
            }
            AdjustableMaAlternatingExtremitiesOutputField::State => next_state,
            AdjustableMaAlternatingExtremitiesOutputField::Changed => {
                if (next_state - prev_state).abs() > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => unreachable!(),
        };
        prev_state = next_state;
    }
}

#[derive(Clone, Debug)]
pub struct AdjustableMaAlternatingExtremitiesBatchRange {
    pub length: (usize, usize, usize),
    pub mult: (f64, f64, f64),
    pub alpha: (f64, f64, f64),
    pub beta: (f64, f64, f64),
}

impl Default for AdjustableMaAlternatingExtremitiesBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            mult: (DEFAULT_MULT, DEFAULT_MULT, 0.0),
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
            beta: (DEFAULT_BETA, DEFAULT_BETA, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdjustableMaAlternatingExtremitiesBatchOutput {
    pub ma: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub extremity: Vec<f64>,
    pub state: Vec<f64>,
    pub changed: Vec<f64>,
    pub smoothed_open: Vec<f64>,
    pub smoothed_high: Vec<f64>,
    pub smoothed_low: Vec<f64>,
    pub smoothed_close: Vec<f64>,
    pub combos: Vec<AdjustableMaAlternatingExtremitiesParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AdjustableMaAlternatingExtremitiesBatchOutput {
    pub fn row_for_params(
        &self,
        params: &AdjustableMaAlternatingExtremitiesParams,
    ) -> Option<usize> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
        let beta = params.beta.unwrap_or(DEFAULT_BETA);
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == length
                && (combo.mult.unwrap_or(DEFAULT_MULT) - mult).abs() <= 1e-12
                && (combo.alpha.unwrap_or(DEFAULT_ALPHA) - alpha).abs() <= 1e-12
                && (combo.beta.unwrap_or(DEFAULT_BETA) - beta).abs() <= 1e-12
        })
    }
}

#[derive(Clone, Debug)]
pub struct AdjustableMaAlternatingExtremitiesBatchBuilder {
    range: AdjustableMaAlternatingExtremitiesBatchRange,
    kernel: Kernel,
}

impl Default for AdjustableMaAlternatingExtremitiesBatchBuilder {
    fn default() -> Self {
        Self {
            range: AdjustableMaAlternatingExtremitiesBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl AdjustableMaAlternatingExtremitiesBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn range(mut self, value: AdjustableMaAlternatingExtremitiesBatchRange) -> Self {
        self.range = value;
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<
        AdjustableMaAlternatingExtremitiesBatchOutput,
        AdjustableMaAlternatingExtremitiesError,
    > {
        adjustable_ma_alternating_extremities_batch_with_kernel(
            high,
            low,
            close,
            &self.range,
            self.kernel,
        )
    }
    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<
        AdjustableMaAlternatingExtremitiesBatchOutput,
        AdjustableMaAlternatingExtremitiesError,
    > {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, AdjustableMaAlternatingExtremitiesError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            match current.checked_sub(step) {
                Some(next) => current = next,
                None => break,
            }
            if current < end {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(AdjustableMaAlternatingExtremitiesError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, AdjustableMaAlternatingExtremitiesError> {
    let eps = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(AdjustableMaAlternatingExtremitiesError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step.abs() < eps || (start - end).abs() < eps {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    let dir = if end >= start { 1.0 } else { -1.0 };
    let step_eff = dir * step.abs();
    let mut current = start;
    if dir > 0.0 {
        while current <= end + eps {
            out.push(current);
            current += step_eff;
        }
    } else {
        while current >= end - eps {
            out.push(current);
            current += step_eff;
        }
    }
    if out.is_empty() {
        return Err(AdjustableMaAlternatingExtremitiesError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid(
    range: &AdjustableMaAlternatingExtremitiesBatchRange,
) -> Result<Vec<AdjustableMaAlternatingExtremitiesParams>, AdjustableMaAlternatingExtremitiesError>
{
    let lengths = axis_usize(range.length)?;
    let mults = axis_f64(range.mult)?;
    let alphas = axis_f64(range.alpha)?;
    let betas = axis_f64(range.beta)?;
    let total = lengths
        .len()
        .checked_mul(mults.len())
        .and_then(|v| v.checked_mul(alphas.len()))
        .and_then(|v| v.checked_mul(betas.len()))
        .ok_or_else(|| AdjustableMaAlternatingExtremitiesError::InvalidRange {
            start: range.length.0.to_string(),
            end: range.length.1.to_string(),
            step: range.length.2.to_string(),
        })?;
    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &mult in &mults {
            for &alpha in &alphas {
                for &beta in &betas {
                    out.push(AdjustableMaAlternatingExtremitiesParams {
                        length: Some(length),
                        mult: Some(mult),
                        alpha: Some(alpha),
                        beta: Some(beta),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn adjustable_ma_alternating_extremities_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    range: &AdjustableMaAlternatingExtremitiesBatchRange,
    kernel: Kernel,
) -> Result<AdjustableMaAlternatingExtremitiesBatchOutput, AdjustableMaAlternatingExtremitiesError>
{
    if high.len() != low.len() || high.len() != close.len() {
        return Err(
            AdjustableMaAlternatingExtremitiesError::DataLengthMismatch {
                high: high.len(),
                low: low.len(),
                close: close.len(),
            },
        );
    }
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AdjustableMaAlternatingExtremitiesError::InvalidKernelForBatch(kernel)),
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(AdjustableMaAlternatingExtremitiesError::EmptyInputData);
    }
    let _ = rows.checked_mul(cols).ok_or_else(|| {
        AdjustableMaAlternatingExtremitiesError::InvalidRange {
            start: range.length.0.to_string(),
            end: range.length.1.to_string(),
            step: range.length.2.to_string(),
        }
    })?;

    let first = (0..cols)
        .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .ok_or(AdjustableMaAlternatingExtremitiesError::AllValuesNaN)?;
    let ma_warm: Vec<usize> = combos
        .iter()
        .map(|params| first + params.length.unwrap_or(DEFAULT_LENGTH) - 1)
        .collect();
    let band_warm: Vec<usize> = combos
        .iter()
        .map(|params| first + (params.length.unwrap_or(DEFAULT_LENGTH) * 2) - 2)
        .collect();
    let open_warm: Vec<usize> = ma_warm.iter().map(|warm| warm + 2).collect();

    let mut ma_mu = make_uninit_matrix(rows, cols);
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);
    let mut extremity_mu = make_uninit_matrix(rows, cols);
    let mut state_mu = make_uninit_matrix(rows, cols);
    let mut changed_mu = make_uninit_matrix(rows, cols);
    let mut smoothed_open_mu = make_uninit_matrix(rows, cols);
    let mut smoothed_high_mu = make_uninit_matrix(rows, cols);
    let mut smoothed_low_mu = make_uninit_matrix(rows, cols);
    let mut smoothed_close_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut ma_mu, cols, &ma_warm);
    init_matrix_prefixes(&mut upper_mu, cols, &band_warm);
    init_matrix_prefixes(&mut lower_mu, cols, &band_warm);
    init_matrix_prefixes(&mut extremity_mu, cols, &band_warm);
    init_matrix_prefixes(&mut state_mu, cols, &band_warm);
    init_matrix_prefixes(&mut changed_mu, cols, &band_warm);
    init_matrix_prefixes(&mut smoothed_open_mu, cols, &open_warm);
    init_matrix_prefixes(&mut smoothed_high_mu, cols, &ma_warm);
    init_matrix_prefixes(&mut smoothed_low_mu, cols, &ma_warm);
    init_matrix_prefixes(&mut smoothed_close_mu, cols, &ma_warm);

    let mut ma_guard = ManuallyDrop::new(ma_mu);
    let mut upper_guard = ManuallyDrop::new(upper_mu);
    let mut lower_guard = ManuallyDrop::new(lower_mu);
    let mut extremity_guard = ManuallyDrop::new(extremity_mu);
    let mut state_guard = ManuallyDrop::new(state_mu);
    let mut changed_guard = ManuallyDrop::new(changed_mu);
    let mut smoothed_open_guard = ManuallyDrop::new(smoothed_open_mu);
    let mut smoothed_high_guard = ManuallyDrop::new(smoothed_high_mu);
    let mut smoothed_low_guard = ManuallyDrop::new(smoothed_low_mu);
    let mut smoothed_close_guard = ManuallyDrop::new(smoothed_close_mu);

    let ma = unsafe { mu_slice_as_f64_slice_mut(&mut ma_guard) };
    let upper = unsafe { mu_slice_as_f64_slice_mut(&mut upper_guard) };
    let lower = unsafe { mu_slice_as_f64_slice_mut(&mut lower_guard) };
    let extremity = unsafe { mu_slice_as_f64_slice_mut(&mut extremity_guard) };
    let state = unsafe { mu_slice_as_f64_slice_mut(&mut state_guard) };
    let changed = unsafe { mu_slice_as_f64_slice_mut(&mut changed_guard) };
    let smoothed_open = unsafe { mu_slice_as_f64_slice_mut(&mut smoothed_open_guard) };
    let smoothed_high = unsafe { mu_slice_as_f64_slice_mut(&mut smoothed_high_guard) };
    let smoothed_low = unsafe { mu_slice_as_f64_slice_mut(&mut smoothed_low_guard) };
    let smoothed_close = unsafe { mu_slice_as_f64_slice_mut(&mut smoothed_close_guard) };

    let run_row = |row: usize,
                   ma_row: &mut [f64],
                   upper_row: &mut [f64],
                   lower_row: &mut [f64],
                   extremity_row: &mut [f64],
                   state_row: &mut [f64],
                   changed_row: &mut [f64],
                   open_row: &mut [f64],
                   sh_row: &mut [f64],
                   sl_row: &mut [f64],
                   sc_row: &mut [f64]|
     -> Result<(), AdjustableMaAlternatingExtremitiesError> {
        let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
            high,
            low,
            close,
            combos[row].clone(),
        );
        adjustable_ma_alternating_extremities_into_slices(
            &input,
            single_kernel,
            ma_row,
            upper_row,
            lower_row,
            extremity_row,
            state_row,
            changed_row,
            open_row,
            sh_row,
            sl_row,
            sc_row,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        ma.par_chunks_mut(cols)
            .zip(upper.par_chunks_mut(cols))
            .zip(lower.par_chunks_mut(cols))
            .zip(extremity.par_chunks_mut(cols))
            .zip(state.par_chunks_mut(cols))
            .zip(changed.par_chunks_mut(cols))
            .zip(smoothed_open.par_chunks_mut(cols))
            .zip(smoothed_high.par_chunks_mut(cols))
            .zip(smoothed_low.par_chunks_mut(cols))
            .zip(smoothed_close.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(
                |(
                    row,
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (((ma_row, upper_row), lower_row), extremity_row),
                                            state_row,
                                        ),
                                        changed_row,
                                    ),
                                    open_row,
                                ),
                                sh_row,
                            ),
                            sl_row,
                        ),
                        sc_row,
                    ),
                )| {
                    run_row(
                        row,
                        ma_row,
                        upper_row,
                        lower_row,
                        extremity_row,
                        state_row,
                        changed_row,
                        open_row,
                        sh_row,
                        sl_row,
                        sc_row,
                    )
                },
            )?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            run_row(
                row,
                &mut ma[start..end],
                &mut upper[start..end],
                &mut lower[start..end],
                &mut extremity[start..end],
                &mut state[start..end],
                &mut changed[start..end],
                &mut smoothed_open[start..end],
                &mut smoothed_high[start..end],
                &mut smoothed_low[start..end],
                &mut smoothed_close[start..end],
            )?;
        }
    }

    Ok(AdjustableMaAlternatingExtremitiesBatchOutput {
        ma: unsafe { vec_f64_from_mu_guard(ma_guard) },
        upper: unsafe { vec_f64_from_mu_guard(upper_guard) },
        lower: unsafe { vec_f64_from_mu_guard(lower_guard) },
        extremity: unsafe { vec_f64_from_mu_guard(extremity_guard) },
        state: unsafe { vec_f64_from_mu_guard(state_guard) },
        changed: unsafe { vec_f64_from_mu_guard(changed_guard) },
        smoothed_open: unsafe { vec_f64_from_mu_guard(smoothed_open_guard) },
        smoothed_high: unsafe { vec_f64_from_mu_guard(smoothed_high_guard) },
        smoothed_low: unsafe { vec_f64_from_mu_guard(smoothed_low_guard) },
        smoothed_close: unsafe { vec_f64_from_mu_guard(smoothed_close_guard) },
        combos,
        rows,
        cols,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjustableMaAlternatingExtremitiesStreamOutput {
    pub ma: f64,
    pub upper: f64,
    pub lower: f64,
    pub extremity: f64,
    pub state: f64,
    pub changed: f64,
    pub smoothed_open: f64,
    pub smoothed_high: f64,
    pub smoothed_low: f64,
    pub smoothed_close: f64,
}

#[derive(Debug, Clone)]
pub struct AdjustableMaAlternatingExtremitiesStream {
    length: usize,
    mult: f64,
    weights: Vec<f64>,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    closes: VecDeque<f64>,
    abs_diffs: VecDeque<f64>,
    rolling_abs_sum: f64,
    prev_high: Option<f64>,
    prev_low: Option<f64>,
    prev_upper: Option<f64>,
    prev_lower: Option<f64>,
    prev_state: f64,
    last_close_1: Option<f64>,
    last_close_2: Option<f64>,
}

impl AdjustableMaAlternatingExtremitiesStream {
    pub fn try_new(
        params: AdjustableMaAlternatingExtremitiesParams,
    ) -> Result<Self, AdjustableMaAlternatingExtremitiesError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
        let beta = params.beta.unwrap_or(DEFAULT_BETA);
        if length < 2 {
            return Err(AdjustableMaAlternatingExtremitiesError::InvalidLength {
                length,
                data_len: length,
            });
        }
        if !mult.is_finite() || mult < 1.0 {
            return Err(AdjustableMaAlternatingExtremitiesError::InvalidMult { mult });
        }
        let weights = build_weights(length, alpha, beta)?;
        Ok(Self {
            length,
            mult,
            weights,
            highs: VecDeque::with_capacity(length),
            lows: VecDeque::with_capacity(length),
            closes: VecDeque::with_capacity(length),
            abs_diffs: VecDeque::with_capacity(length),
            rolling_abs_sum: 0.0,
            prev_high: None,
            prev_low: None,
            prev_upper: None,
            prev_lower: None,
            prev_state: 0.0,
            last_close_1: None,
            last_close_2: None,
        })
    }

    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<AdjustableMaAlternatingExtremitiesStreamOutput> {
        if !(high.is_finite() && low.is_finite() && close.is_finite()) {
            return None;
        }
        push_ring(&mut self.highs, self.length, high);
        push_ring(&mut self.lows, self.length, low);
        push_ring(&mut self.closes, self.length, close);
        if self.closes.len() < self.length {
            return None;
        }
        let ma = dot_recent(&self.closes, &self.weights);
        let smoothed_high = dot_recent(&self.highs, &self.weights);
        let smoothed_low = dot_recent(&self.lows, &self.weights);
        let smoothed_open = self
            .last_close_1
            .zip(self.last_close_2)
            .map(|(a, b)| 0.5 * (a + b))
            .unwrap_or(f64::NAN);
        let abs_diff = (close - ma).abs();
        self.abs_diffs.push_back(abs_diff);
        self.rolling_abs_sum += abs_diff;
        if self.abs_diffs.len() > self.length {
            if let Some(removed) = self.abs_diffs.pop_front() {
                self.rolling_abs_sum -= removed;
            }
        }
        self.last_close_2 = self.last_close_1;
        self.last_close_1 = Some(ma);
        if self.abs_diffs.len() < self.length {
            return None;
        }
        let dev = (self.rolling_abs_sum / self.length as f64) * self.mult;
        let upper = ma + dev;
        let lower = ma - dev;
        let cross_high = self
            .prev_high
            .zip(self.prev_upper)
            .map(|(ph, pu)| pine_cross(ph, pu, high, upper))
            .unwrap_or(false);
        let cross_low = self
            .prev_low
            .zip(self.prev_lower)
            .map(|(pl, plow)| pine_cross(pl, plow, low, lower))
            .unwrap_or(false);
        let next_state = if cross_high {
            1.0
        } else if cross_low {
            0.0
        } else {
            self.prev_state
        };
        let changed = if (next_state - self.prev_state).abs() > 0.0 {
            1.0
        } else {
            0.0
        };
        let extremity = if next_state >= 0.5 { upper } else { lower };
        self.prev_high = Some(high);
        self.prev_low = Some(low);
        self.prev_upper = Some(upper);
        self.prev_lower = Some(lower);
        self.prev_state = next_state;
        Some(AdjustableMaAlternatingExtremitiesStreamOutput {
            ma,
            upper,
            lower,
            extremity,
            state: next_state,
            changed,
            smoothed_open,
            smoothed_high,
            smoothed_low,
            smoothed_close: ma,
        })
    }
}

#[inline]
fn push_ring(queue: &mut VecDeque<f64>, len: usize, value: f64) {
    if queue.len() == len {
        queue.pop_front();
    }
    queue.push_back(value);
}

#[inline]
fn dot_recent(queue: &VecDeque<f64>, weights: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (i, value) in queue.iter().rev().enumerate() {
        acc += value * weights[i];
    }
    acc
}

#[inline(always)]
unsafe fn mu_slice_as_f64_slice_mut(buf: &mut ManuallyDrop<Vec<MaybeUninit<f64>>>) -> &mut [f64] {
    core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f64, buf.len())
}

#[inline(always)]
unsafe fn vec_f64_from_mu_guard(buf: ManuallyDrop<Vec<MaybeUninit<f64>>>) -> Vec<f64> {
    let mut buf = buf;
    Vec::from_raw_parts(buf.as_mut_ptr() as *mut f64, buf.len(), buf.capacity())
}

#[cfg(feature = "python")]
#[pyfunction(name = "adjustable_ma_alternating_extremities")]
#[pyo3(signature = (high, low, close, length=DEFAULT_LENGTH, mult=DEFAULT_MULT, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, kernel=None))]
pub fn adjustable_ma_alternating_extremities_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    mult: f64,
    alpha: f64,
    beta: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
        high,
        low,
        close,
        AdjustableMaAlternatingExtremitiesParams {
            length: Some(length),
            mult: Some(mult),
            alpha: Some(alpha),
            beta: Some(beta),
        },
    );
    let output = py
        .allow_threads(|| adjustable_ma_alternating_extremities_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("ma", output.ma.into_pyarray(py))?;
    dict.set_item("upper", output.upper.into_pyarray(py))?;
    dict.set_item("lower", output.lower.into_pyarray(py))?;
    dict.set_item("extremity", output.extremity.into_pyarray(py))?;
    dict.set_item("state", output.state.into_pyarray(py))?;
    dict.set_item("changed", output.changed.into_pyarray(py))?;
    dict.set_item("smoothed_open", output.smoothed_open.into_pyarray(py))?;
    dict.set_item("smoothed_high", output.smoothed_high.into_pyarray(py))?;
    dict.set_item("smoothed_low", output.smoothed_low.into_pyarray(py))?;
    dict.set_item("smoothed_close", output.smoothed_close.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "adjustable_ma_alternating_extremities_batch")]
#[pyo3(signature = (high, low, close, length_range, mult_range, alpha_range, beta_range, kernel=None))]
pub fn adjustable_ma_alternating_extremities_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    alpha_range: (f64, f64, f64),
    beta_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            adjustable_ma_alternating_extremities_batch_with_kernel(
                high,
                low,
                close,
                &AdjustableMaAlternatingExtremitiesBatchRange {
                    length: length_range,
                    mult: mult_range,
                    alpha: alpha_range,
                    beta: beta_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let total = output.rows * output.cols;
    let arrays = [
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
    ];
    unsafe { arrays[0].as_slice_mut()? }.copy_from_slice(&output.ma);
    unsafe { arrays[1].as_slice_mut()? }.copy_from_slice(&output.upper);
    unsafe { arrays[2].as_slice_mut()? }.copy_from_slice(&output.lower);
    unsafe { arrays[3].as_slice_mut()? }.copy_from_slice(&output.extremity);
    unsafe { arrays[4].as_slice_mut()? }.copy_from_slice(&output.state);
    unsafe { arrays[5].as_slice_mut()? }.copy_from_slice(&output.changed);
    unsafe { arrays[6].as_slice_mut()? }.copy_from_slice(&output.smoothed_open);
    unsafe { arrays[7].as_slice_mut()? }.copy_from_slice(&output.smoothed_high);
    unsafe { arrays[8].as_slice_mut()? }.copy_from_slice(&output.smoothed_low);
    unsafe { arrays[9].as_slice_mut()? }.copy_from_slice(&output.smoothed_close);

    let dict = PyDict::new(py);
    dict.set_item("ma", arrays[0].reshape((output.rows, output.cols))?)?;
    dict.set_item("upper", arrays[1].reshape((output.rows, output.cols))?)?;
    dict.set_item("lower", arrays[2].reshape((output.rows, output.cols))?)?;
    dict.set_item("extremity", arrays[3].reshape((output.rows, output.cols))?)?;
    dict.set_item("state", arrays[4].reshape((output.rows, output.cols))?)?;
    dict.set_item("changed", arrays[5].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "smoothed_open",
        arrays[6].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "smoothed_high",
        arrays[7].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "smoothed_low",
        arrays[8].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "smoothed_close",
        arrays[9].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "mults",
        output
            .combos
            .iter()
            .map(|combo| combo.mult.unwrap_or(DEFAULT_MULT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "alphas",
        output
            .combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "betas",
        output
            .combos
            .iter()
            .map(|combo| combo.beta.unwrap_or(DEFAULT_BETA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AdjustableMaAlternatingExtremitiesStream")]
pub struct AdjustableMaAlternatingExtremitiesStreamPy {
    stream: AdjustableMaAlternatingExtremitiesStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdjustableMaAlternatingExtremitiesStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, mult=DEFAULT_MULT, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA))]
    fn new(length: usize, mult: f64, alpha: f64, beta: f64) -> PyResult<Self> {
        let stream = AdjustableMaAlternatingExtremitiesStream::try_new(
            AdjustableMaAlternatingExtremitiesParams {
                length: Some(length),
                mult: Some(mult),
                alpha: Some(alpha),
                beta: Some(beta),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.stream.update(high, low, close).map(|out| {
            (
                out.ma,
                out.upper,
                out.lower,
                out.extremity,
                out.state,
                out.changed,
                out.smoothed_open,
                out.smoothed_high,
                out.smoothed_low,
                out.smoothed_close,
            )
        })
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdjustableMaAlternatingExtremitiesJsOutput {
    pub ma: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub extremity: Vec<f64>,
    pub state: Vec<f64>,
    pub changed: Vec<f64>,
    pub smoothed_open: Vec<f64>,
    pub smoothed_high: Vec<f64>,
    pub smoothed_low: Vec<f64>,
    pub smoothed_close: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adjustable_ma_alternating_extremities_js)]
pub fn adjustable_ma_alternating_extremities_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
    alpha: f64,
    beta: f64,
) -> Result<JsValue, JsValue> {
    let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
        high,
        low,
        close,
        AdjustableMaAlternatingExtremitiesParams {
            length: Some(length),
            mult: Some(mult),
            alpha: Some(alpha),
            beta: Some(beta),
        },
    );
    let output = adjustable_ma_alternating_extremities_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AdjustableMaAlternatingExtremitiesJsOutput {
        ma: output.ma,
        upper: output.upper,
        lower: output.lower,
        extremity: output.extremity,
        state: output.state,
        changed: output.changed,
        smoothed_open: output.smoothed_open,
        smoothed_high: output.smoothed_high,
        smoothed_low: output.smoothed_low,
        smoothed_close: output.smoothed_close,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdjustableMaAlternatingExtremitiesBatchConfig {
    pub length_range: (usize, usize, usize),
    pub mult_range: (f64, f64, f64),
    pub alpha_range: (f64, f64, f64),
    pub beta_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdjustableMaAlternatingExtremitiesBatchJsOutput {
    pub ma: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub extremity: Vec<f64>,
    pub state: Vec<f64>,
    pub changed: Vec<f64>,
    pub smoothed_open: Vec<f64>,
    pub smoothed_high: Vec<f64>,
    pub smoothed_low: Vec<f64>,
    pub smoothed_close: Vec<f64>,
    pub lengths: Vec<usize>,
    pub mults: Vec<f64>,
    pub alphas: Vec<f64>,
    pub betas: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adjustable_ma_alternating_extremities_batch)]
pub fn adjustable_ma_alternating_extremities_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: AdjustableMaAlternatingExtremitiesBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = adjustable_ma_alternating_extremities_batch_with_kernel(
        high,
        low,
        close,
        &AdjustableMaAlternatingExtremitiesBatchRange {
            length: cfg.length_range,
            mult: cfg.mult_range,
            alpha: cfg.alpha_range,
            beta: cfg.beta_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AdjustableMaAlternatingExtremitiesBatchJsOutput {
        ma: output.ma,
        upper: output.upper,
        lower: output.lower,
        extremity: output.extremity,
        state: output.state,
        changed: output.changed,
        smoothed_open: output.smoothed_open,
        smoothed_high: output.smoothed_high,
        smoothed_low: output.smoothed_low,
        smoothed_close: output.smoothed_close,
        lengths: output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        mults: output
            .combos
            .iter()
            .map(|combo| combo.mult.unwrap_or(DEFAULT_MULT))
            .collect(),
        alphas: output
            .combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect(),
        betas: output
            .combos
            .iter()
            .map(|combo| combo.beta.unwrap_or(DEFAULT_BETA))
            .collect(),
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adjustable_ma_alternating_extremities_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
    alpha: f64,
    beta: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        adjustable_ma_alternating_extremities_js(high, low, close, length, mult, alpha, beta)?;
    crate::write_wasm_object_f64_outputs(
        "adjustable_ma_alternating_extremities_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adjustable_ma_alternating_extremities_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adjustable_ma_alternating_extremities_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "adjustable_ma_alternating_extremities_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq_or_both_nan(lhs: f64, rhs: f64) -> bool {
        (lhs.is_nan() && rhs.is_nan()) || lhs == rhs
    }

    fn assert_series_eq(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            assert!(
                eq_or_both_nan(lhs[i], rhs[i]),
                "mismatch at index {i}: lhs={} rhs={}",
                lhs[i],
                rhs[i]
            );
        }
    }

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + (i as f64 * 0.17).sin() * 4.0 + i as f64 * 0.03;
            close.push(base);
            high.push(base + 1.5 + (i as f64 * 0.11).cos().abs());
            low.push(base - 1.5 - (i as f64 * 0.07).sin().abs());
        }
        (high, low, close)
    }

    #[test]
    fn constant_series_produces_flat_outputs() -> Result<(), Box<dyn std::error::Error>> {
        let n = 180;
        let high = vec![101.0; n];
        let low = vec![99.0; n];
        let close = vec![100.0; n];
        let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
            &high,
            &low,
            &close,
            AdjustableMaAlternatingExtremitiesParams::default(),
        );
        let out = adjustable_ma_alternating_extremities(&input)?;
        let start = DEFAULT_LENGTH * 2 - 2;
        for i in start..n {
            assert!((out.ma[i] - 100.0).abs() <= 1e-9);
            assert!((out.upper[i] - 100.0).abs() <= 1e-9);
            assert!((out.lower[i] - 100.0).abs() <= 1e-9);
            assert!((out.extremity[i] - 100.0).abs() <= 1e-9);
            assert_eq!(out.state[i], 0.0);
            assert_eq!(out.changed[i], 0.0);
        }
        Ok(())
    }

    #[test]
    fn into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(256);
        let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
            &high,
            &low,
            &close,
            AdjustableMaAlternatingExtremitiesParams::default(),
        );
        let baseline = adjustable_ma_alternating_extremities(&input)?;
        let n = close.len();
        let mut ma = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];
        let mut extremity = vec![0.0; n];
        let mut state = vec![0.0; n];
        let mut changed = vec![0.0; n];
        let mut smoothed_open = vec![0.0; n];
        let mut smoothed_high = vec![0.0; n];
        let mut smoothed_low = vec![0.0; n];
        let mut smoothed_close = vec![0.0; n];
        adjustable_ma_alternating_extremities_into(
            &input,
            &mut ma,
            &mut upper,
            &mut lower,
            &mut extremity,
            &mut state,
            &mut changed,
            &mut smoothed_open,
            &mut smoothed_high,
            &mut smoothed_low,
            &mut smoothed_close,
        )?;
        assert_series_eq(&baseline.ma, &ma);
        assert_series_eq(&baseline.upper, &upper);
        assert_series_eq(&baseline.lower, &lower);
        assert_series_eq(&baseline.extremity, &extremity);
        assert_series_eq(&baseline.state, &state);
        assert_series_eq(&baseline.changed, &changed);
        assert_series_eq(&baseline.smoothed_open, &smoothed_open);
        assert_series_eq(&baseline.smoothed_high, &smoothed_high);
        assert_series_eq(&baseline.smoothed_low, &smoothed_low);
        assert_series_eq(&baseline.smoothed_close, &smoothed_close);
        Ok(())
    }

    #[test]
    fn stream_matches_batch() -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(240);
        let params = AdjustableMaAlternatingExtremitiesParams::default();
        let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
            &high,
            &low,
            &close,
            params.clone(),
        );
        let batch = adjustable_ma_alternating_extremities(&input)?;
        let mut stream = AdjustableMaAlternatingExtremitiesStream::try_new(params)?;
        for i in 0..close.len() {
            match stream.update(high[i], low[i], close[i]) {
                Some(out) => {
                    assert!((out.ma - batch.ma[i]).abs() <= 1e-9);
                    assert!((out.upper - batch.upper[i]).abs() <= 1e-9);
                    assert!((out.lower - batch.lower[i]).abs() <= 1e-9);
                    assert!((out.extremity - batch.extremity[i]).abs() <= 1e-9);
                    assert!((out.state - batch.state[i]).abs() <= 1e-9);
                    assert!((out.changed - batch.changed[i]).abs() <= 1e-9);
                }
                None => {
                    assert!(batch.upper[i].is_nan());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn batch_default_row_matches_single() -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(180);
        let params = AdjustableMaAlternatingExtremitiesParams::default();
        let single = adjustable_ma_alternating_extremities(
            &AdjustableMaAlternatingExtremitiesInput::from_slices(
                &high,
                &low,
                &close,
                params.clone(),
            ),
        )?;
        let batch = adjustable_ma_alternating_extremities_batch_with_kernel(
            &high,
            &low,
            &close,
            &AdjustableMaAlternatingExtremitiesBatchRange::default(),
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_series_eq(&batch.ma[..close.len()], single.ma.as_slice());
        assert_series_eq(&batch.extremity[..close.len()], single.extremity.as_slice());
        Ok(())
    }

    #[test]
    fn state_and_extremity_invariants_hold() -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(320);
        let out = adjustable_ma_alternating_extremities(
            &AdjustableMaAlternatingExtremitiesInput::from_slices(
                &high,
                &low,
                &close,
                AdjustableMaAlternatingExtremitiesParams::default(),
            ),
        )?;
        let start = DEFAULT_LENGTH * 2 - 2;
        for i in start..close.len() {
            assert!(out.state[i] == 0.0 || out.state[i] == 1.0);
            if out.state[i] == 1.0 {
                assert!((out.extremity[i] - out.upper[i]).abs() <= 1e-12);
            } else {
                assert!((out.extremity[i] - out.lower[i]).abs() <= 1e-12);
            }
            if i > start {
                let expected_changed = if (out.state[i] - out.state[i - 1]).abs() > 0.0 {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(out.changed[i], expected_changed);
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_params_are_rejected() {
        let (high, low, close) = sample_ohlc(160);
        let err = adjustable_ma_alternating_extremities(
            &AdjustableMaAlternatingExtremitiesInput::from_slices(
                &high,
                &low,
                &close,
                AdjustableMaAlternatingExtremitiesParams {
                    length: Some(1),
                    ..Default::default()
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AdjustableMaAlternatingExtremitiesError::InvalidLength { .. }
        ));
        let err = adjustable_ma_alternating_extremities(
            &AdjustableMaAlternatingExtremitiesInput::from_slices(
                &high,
                &low,
                &close,
                AdjustableMaAlternatingExtremitiesParams {
                    mult: Some(0.5),
                    ..Default::default()
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AdjustableMaAlternatingExtremitiesError::InvalidMult { .. }
        ));
    }
}
