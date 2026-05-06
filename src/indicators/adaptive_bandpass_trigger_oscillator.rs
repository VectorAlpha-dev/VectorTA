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
use std::f64::consts::PI;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_DELTA: f64 = 0.1;
const DEFAULT_ALPHA: f64 = 0.07;
const MIN_VALID_SAMPLES: usize = 12;
const IN_PHASE_WARMUP: usize = MIN_VALID_SAMPLES - 1;
const LEAD_WARMUP: usize = MIN_VALID_SAMPLES;
const FLOAT_TOL: f64 = 1e-12;

impl<'a> AsRef<[f64]> for AdaptiveBandpassTriggerOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AdaptiveBandpassTriggerOscillatorData::Slice(slice) => slice,
            AdaptiveBandpassTriggerOscillatorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AdaptiveBandpassTriggerOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorOutput {
    pub in_phase: Vec<f64>,
    pub lead: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveBandpassTriggerOscillatorOutputField {
    InPhase,
    Lead,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdaptiveBandpassTriggerOscillatorParams {
    pub delta: Option<f64>,
    pub alpha: Option<f64>,
}

impl Default for AdaptiveBandpassTriggerOscillatorParams {
    fn default() -> Self {
        Self {
            delta: Some(DEFAULT_DELTA),
            alpha: Some(DEFAULT_ALPHA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorInput<'a> {
    pub data: AdaptiveBandpassTriggerOscillatorData<'a>,
    pub params: AdaptiveBandpassTriggerOscillatorParams,
}

impl<'a> AdaptiveBandpassTriggerOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: AdaptiveBandpassTriggerOscillatorParams,
    ) -> Self {
        Self {
            data: AdaptiveBandpassTriggerOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AdaptiveBandpassTriggerOscillatorParams) -> Self {
        Self {
            data: AdaptiveBandpassTriggerOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            AdaptiveBandpassTriggerOscillatorParams::default(),
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AdaptiveBandpassTriggerOscillatorBuilder {
    delta: Option<f64>,
    alpha: Option<f64>,
    kernel: Kernel,
}

impl AdaptiveBandpassTriggerOscillatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = Some(delta);
        self
    }

    #[inline]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
}

#[derive(Debug, Error)]
pub enum AdaptiveBandpassTriggerOscillatorError {
    #[error("adaptive_bandpass_trigger_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("adaptive_bandpass_trigger_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("adaptive_bandpass_trigger_oscillator: Invalid delta: {delta}")]
    InvalidDelta { delta: f64 },
    #[error("adaptive_bandpass_trigger_oscillator: Invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error(
        "adaptive_bandpass_trigger_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "adaptive_bandpass_trigger_oscillator: Output length mismatch: expected = {expected}, in_phase = {in_phase_got}, lead = {lead_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        in_phase_got: usize,
        lead_got: usize,
    },
    #[error(
        "adaptive_bandpass_trigger_oscillator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("adaptive_bandpass_trigger_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    delta: f64,
    alpha: f64,
}

#[inline(always)]
fn resolve_params(
    params: &AdaptiveBandpassTriggerOscillatorParams,
) -> Result<ResolvedParams, AdaptiveBandpassTriggerOscillatorError> {
    let delta = params.delta.unwrap_or(DEFAULT_DELTA);
    let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
    if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidDelta { delta });
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidAlpha { alpha });
    }
    Ok(ResolvedParams { delta, alpha })
}

#[inline(always)]
fn median3(x: f64, y: f64, z: f64) -> f64 {
    (x + y + z) - x.min(y.min(z)) - x.max(y.max(z))
}

#[inline(always)]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|value| value.is_finite()).count()
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    data.iter()
        .position(|value| value.is_finite())
        .unwrap_or(data.len())
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorStream {
    params: ResolvedParams,
    price: [f64; 4],
    smooth_hist: [f64; 2],
    c_hist: [f64; 6],
    dp_hist: [f64; 4],
    q1_prev: f64,
    i1_prev: f64,
    ip_prev: f64,
    p_prev: f64,
    bp_prev1: f64,
    bp_prev2: f64,
    valid_count: usize,
}

impl AdaptiveBandpassTriggerOscillatorStream {
    pub fn try_new(
        params: AdaptiveBandpassTriggerOscillatorParams,
    ) -> Result<Self, AdaptiveBandpassTriggerOscillatorError> {
        let params = resolve_params(&params)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            price: [0.0; 4],
            smooth_hist: [0.0; 2],
            c_hist: [0.0; 6],
            dp_hist: [0.0; 4],
            q1_prev: 0.0,
            i1_prev: 0.0,
            ip_prev: 0.0,
            p_prev: 0.0,
            bp_prev1: 0.0,
            bp_prev2: 0.0,
            valid_count: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        IN_PHASE_WARMUP
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        self.price[3] = self.price[2];
        self.price[2] = self.price[1];
        self.price[1] = self.price[0];
        self.price[0] = value;

        let index = self.valid_count;
        self.valid_count += 1;

        let smooth = if index >= 3 {
            (self.price[0] + 2.0 * self.price[1] + 2.0 * self.price[2] + self.price[3]) / 6.0
        } else {
            0.0
        };

        let alpha = self.params.alpha;
        let c = if index < 2 {
            0.0
        } else if index < 7 {
            (self.price[0] - 2.0 * self.price[1] + self.price[2]) * 0.25
        } else {
            let smooth_gain = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
            smooth_gain * (smooth - 2.0 * self.smooth_hist[0] + self.smooth_hist[1])
                + 2.0 * (1.0 - alpha) * self.c_hist[0]
                - (1.0 - alpha) * (1.0 - alpha) * self.c_hist[1]
        };

        let q1 = if index >= 6 {
            (0.0962 * c + 0.5769 * self.c_hist[1]
                - 0.5769 * self.c_hist[3]
                - 0.0962 * self.c_hist[5])
                * (0.5 + 0.08 * self.ip_prev)
        } else {
            0.0
        };
        let i1 = if index >= 3 { self.c_hist[2] } else { 0.0 };

        let dp_raw = if q1.abs() > FLOAT_TOL && self.q1_prev.abs() > FLOAT_TOL {
            let denominator = 1.0 + (i1 * self.i1_prev) / (q1 * self.q1_prev);
            if denominator.abs() > FLOAT_TOL {
                ((i1 / q1) - (self.i1_prev / self.q1_prev)) / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };
        let dp = dp_raw.clamp(0.1, 1.1);

        let md = if index >= 10 {
            median3(
                dp,
                self.dp_hist[0],
                median3(self.dp_hist[1], self.dp_hist[2], self.dp_hist[3]),
            )
        } else {
            0.0
        };
        let dc = if md.abs() <= FLOAT_TOL {
            15.0
        } else {
            (2.0 * PI) / md + 0.5
        };

        let ip = 0.33 * dc + 0.67 * self.ip_prev;
        let p = 0.15 * ip + 0.85 * self.p_prev;

        let mut in_phase = f64::NAN;
        let mut lead = f64::NAN;
        if index >= IN_PHASE_WARMUP {
            let length = p.max(6.0);
            let beta = (2.0 * PI / length).cos();
            let cos_angle = (4.0 * PI * self.params.delta / length).cos();
            let denom = if cos_angle.abs() < FLOAT_TOL {
                if cos_angle.is_sign_negative() {
                    -FLOAT_TOL
                } else {
                    FLOAT_TOL
                }
            } else {
                cos_angle
            };
            let gamma = 1.0 / denom;
            let alpha_bp = gamma - (gamma * gamma - 1.0).max(0.0).sqrt();

            in_phase = 0.5 * (1.0 - alpha_bp) * (self.price[0] - self.price[2])
                + beta * (1.0 + alpha_bp) * self.bp_prev1
                - alpha_bp * self.bp_prev2;
            if index >= LEAD_WARMUP && self.bp_prev1.is_finite() {
                let quadrature = (in_phase - self.bp_prev1) * length / (2.0 * PI);
                lead = 0.5 * in_phase + 0.866 * quadrature;
            }
        }

        self.smooth_hist[1] = self.smooth_hist[0];
        self.smooth_hist[0] = smooth;

        self.c_hist[5] = self.c_hist[4];
        self.c_hist[4] = self.c_hist[3];
        self.c_hist[3] = self.c_hist[2];
        self.c_hist[2] = self.c_hist[1];
        self.c_hist[1] = self.c_hist[0];
        self.c_hist[0] = c;

        self.dp_hist[3] = self.dp_hist[2];
        self.dp_hist[2] = self.dp_hist[1];
        self.dp_hist[1] = self.dp_hist[0];
        self.dp_hist[0] = dp;

        self.q1_prev = q1;
        self.i1_prev = i1;
        self.ip_prev = ip;
        self.p_prev = p;

        if in_phase.is_finite() {
            self.bp_prev2 = self.bp_prev1;
            self.bp_prev1 = in_phase;
            Some((in_phase, lead))
        } else {
            None
        }
    }
}

impl AdaptiveBandpassTriggerOscillatorBuilder {
    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError>
    {
        let input = AdaptiveBandpassTriggerOscillatorInput::from_candles(
            candles,
            "close",
            AdaptiveBandpassTriggerOscillatorParams {
                delta: self.delta,
                alpha: self.alpha,
            },
        );
        adaptive_bandpass_trigger_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError>
    {
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            data,
            AdaptiveBandpassTriggerOscillatorParams {
                delta: self.delta,
                alpha: self.alpha,
            },
        );
        adaptive_bandpass_trigger_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<AdaptiveBandpassTriggerOscillatorStream, AdaptiveBandpassTriggerOscillatorError>
    {
        AdaptiveBandpassTriggerOscillatorStream::try_new(AdaptiveBandpassTriggerOscillatorParams {
            delta: self.delta,
            alpha: self.alpha,
        })
    }
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator(
    input: &AdaptiveBandpassTriggerOscillatorInput,
) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn adaptive_bandpass_trigger_oscillator_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
) {
    let mut stream = AdaptiveBandpassTriggerOscillatorStream::new_resolved(params);
    for i in 0..data.len() {
        match stream.update(data[i]) {
            Some((in_phase, lead)) => {
                in_phase_out[i] = in_phase;
                lead_out[i] = lead;
            }
            None => {
                in_phase_out[i] = f64::NAN;
                lead_out[i] = f64::NAN;
            }
        }
    }
}

#[inline(always)]
fn adaptive_bandpass_trigger_oscillator_output_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    field: AdaptiveBandpassTriggerOscillatorOutputField,
    out: &mut [f64],
) {
    let mut stream = AdaptiveBandpassTriggerOscillatorStream::new_resolved(params);
    match field {
        AdaptiveBandpassTriggerOscillatorOutputField::InPhase => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((in_phase, _)) => in_phase,
                    None => f64::NAN,
                };
            }
        }
        AdaptiveBandpassTriggerOscillatorOutputField::Lead => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((_, lead)) => lead,
                    None => f64::NAN,
                };
            }
        }
    }
}

#[inline(always)]
fn adaptive_bandpass_trigger_oscillator_prepare<'a>(
    input: &'a AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, ResolvedParams, Kernel), AdaptiveBandpassTriggerOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(AdaptiveBandpassTriggerOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(AdaptiveBandpassTriggerOscillatorError::AllValuesNaN);
    }
    let params = resolve_params(&input.params)?;
    let valid = count_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(AdaptiveBandpassTriggerOscillatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, first, params, chosen))
}

pub fn adaptive_bandpass_trigger_oscillator_with_kernel(
    input: &AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError> {
    let (data, first, params, _chosen) =
        adaptive_bandpass_trigger_oscillator_prepare(input, kernel)?;
    let mut in_phase = alloc_with_nan_prefix(data.len(), (first + IN_PHASE_WARMUP).min(data.len()));
    let mut lead = alloc_with_nan_prefix(data.len(), (first + LEAD_WARMUP).min(data.len()));
    adaptive_bandpass_trigger_oscillator_row_from_slice(data, params, &mut in_phase, &mut lead);
    Ok(AdaptiveBandpassTriggerOscillatorOutput { in_phase, lead })
}

pub fn adaptive_bandpass_trigger_oscillator_into_slices(
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
    input: &AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
) -> Result<(), AdaptiveBandpassTriggerOscillatorError> {
    let expected = input.as_ref().len();
    if in_phase_out.len() != expected || lead_out.len() != expected {
        return Err(
            AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
                expected,
                in_phase_got: in_phase_out.len(),
                lead_got: lead_out.len(),
            },
        );
    }
    let (data, _first, params, _chosen) =
        adaptive_bandpass_trigger_oscillator_prepare(input, kernel)?;
    adaptive_bandpass_trigger_oscillator_row_from_slice(data, params, in_phase_out, lead_out);
    Ok(())
}

pub fn adaptive_bandpass_trigger_oscillator_output_into_slice(
    out: &mut [f64],
    input: &AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
    field: AdaptiveBandpassTriggerOscillatorOutputField,
) -> Result<(), AdaptiveBandpassTriggerOscillatorError> {
    let expected = input.as_ref().len();
    if out.len() != expected {
        return Err(
            AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
                expected,
                in_phase_got: out.len(),
                lead_got: out.len(),
            },
        );
    }
    let (data, _first, params, _chosen) =
        adaptive_bandpass_trigger_oscillator_prepare(input, kernel)?;
    adaptive_bandpass_trigger_oscillator_output_row_from_slice(data, params, field, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn adaptive_bandpass_trigger_oscillator_into(
    input: &AdaptiveBandpassTriggerOscillatorInput,
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
) -> Result<(), AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_into_slices(in_phase_out, lead_out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdaptiveBandpassTriggerOscillatorBatchRange {
    pub delta: (f64, f64, f64),
    pub alpha: (f64, f64, f64),
}

impl Default for AdaptiveBandpassTriggerOscillatorBatchRange {
    fn default() -> Self {
        Self {
            delta: (DEFAULT_DELTA, DEFAULT_DELTA, 0.0),
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorBatchOutput {
    pub in_phase: Vec<f64>,
    pub lead: Vec<f64>,
    pub combos: Vec<AdaptiveBandpassTriggerOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AdaptiveBandpassTriggerOscillatorBatchBuilder {
    sweep: AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
}

impl AdaptiveBandpassTriggerOscillatorBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn delta(mut self, start: f64, end: f64, step: f64) -> Self {
        self.sweep.delta = (start, end, step);
        self
    }

    #[inline]
    pub fn alpha(mut self, start: f64, end: f64, step: f64) -> Self {
        self.sweep.alpha = (start, end, step);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError>
    {
        adaptive_bandpass_trigger_oscillator_batch_with_kernel(data, &self.sweep, self.kernel)
    }
}

#[inline]
fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, AdaptiveBandpassTriggerOscillatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
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
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(values)
}

#[inline]
fn expand_grid_adaptive_bandpass_trigger_oscillator(
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
) -> Result<Vec<AdaptiveBandpassTriggerOscillatorParams>, AdaptiveBandpassTriggerOscillatorError> {
    let deltas = expand_axis_f64(sweep.delta.0, sweep.delta.1, sweep.delta.2)?;
    let alphas = expand_axis_f64(sweep.alpha.0, sweep.alpha.1, sweep.alpha.2)?;
    let mut combos = Vec::with_capacity(deltas.len() * alphas.len());
    for &delta in &deltas {
        for &alpha in &alphas {
            let combo = AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(delta),
                alpha: Some(alpha),
            };
            let _ = resolve_params(&combo)?;
            combos.push(combo);
        }
    }
    Ok(combos)
}

impl AdaptiveBandpassTriggerOscillatorBatchOutput {
    #[inline]
    pub fn row_for_params(
        &self,
        params: &AdaptiveBandpassTriggerOscillatorParams,
    ) -> Option<usize> {
        self.combos.iter().position(|combo| {
            (combo.delta.unwrap_or(DEFAULT_DELTA) - params.delta.unwrap_or(DEFAULT_DELTA)).abs()
                < FLOAT_TOL
                && (combo.alpha.unwrap_or(DEFAULT_ALPHA) - params.alpha.unwrap_or(DEFAULT_ALPHA))
                    .abs()
                    < FLOAT_TOL
        })
    }

    #[inline]
    pub fn row_slices(&self, row: usize) -> Option<(&[f64], &[f64])> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some((&self.in_phase[start..end], &self.lead[start..end]))
    }
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdaptiveBandpassTriggerOscillatorError::InvalidKernelForBatch(other)),
    };
    adaptive_bandpass_trigger_oscillator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator_batch_slice(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_batch_inner(data, sweep, kernel, true)
}

pub fn adaptive_bandpass_trigger_oscillator_batch_inner(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    let combos = expand_grid_adaptive_bandpass_trigger_oscillator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= cols {
        return Err(AdaptiveBandpassTriggerOscillatorError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(AdaptiveBandpassTriggerOscillatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }

    let mut in_phase_mu = make_uninit_matrix(rows, cols);
    let mut lead_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(
        &mut in_phase_mu,
        cols,
        &vec![(first + IN_PHASE_WARMUP).min(cols); rows],
    );
    init_matrix_prefixes(
        &mut lead_mu,
        cols,
        &vec![(first + LEAD_WARMUP).min(cols); rows],
    );

    let mut in_phase_guard = ManuallyDrop::new(in_phase_mu);
    let mut lead_guard = ManuallyDrop::new(lead_mu);
    let in_phase_out = unsafe {
        std::slice::from_raw_parts_mut(
            in_phase_guard.as_mut_ptr() as *mut f64,
            in_phase_guard.len(),
        )
    };
    let lead_out = unsafe {
        std::slice::from_raw_parts_mut(lead_guard.as_mut_ptr() as *mut f64, lead_guard.len())
    };

    let combos = adaptive_bandpass_trigger_oscillator_batch_inner_into(
        data,
        sweep,
        _kernel,
        parallel,
        in_phase_out,
        lead_out,
    )?;

    let in_phase = unsafe {
        Vec::from_raw_parts(
            in_phase_guard.as_mut_ptr() as *mut f64,
            in_phase_guard.len(),
            in_phase_guard.capacity(),
        )
    };
    let lead = unsafe {
        Vec::from_raw_parts(
            lead_guard.as_mut_ptr() as *mut f64,
            lead_guard.len(),
            lead_guard.capacity(),
        )
    };

    Ok(AdaptiveBandpassTriggerOscillatorBatchOutput {
        in_phase,
        lead,
        combos,
        rows,
        cols,
    })
}

pub fn adaptive_bandpass_trigger_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
) -> Result<Vec<AdaptiveBandpassTriggerOscillatorParams>, AdaptiveBandpassTriggerOscillatorError> {
    let combos = expand_grid_adaptive_bandpass_trigger_oscillator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::EmptyInputData);
    }
    let total = rows.checked_mul(cols).ok_or(
        AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
            expected: usize::MAX,
            in_phase_got: in_phase_out.len(),
            lead_got: lead_out.len(),
        },
    )?;
    if in_phase_out.len() != total || lead_out.len() != total {
        return Err(
            AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
                expected: total,
                in_phase_got: in_phase_out.len(),
                lead_got: lead_out.len(),
            },
        );
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(AdaptiveBandpassTriggerOscillatorError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(AdaptiveBandpassTriggerOscillatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        in_phase_out
            .par_chunks_mut(cols)
            .zip(lead_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (in_phase_row, lead_row))| {
                let params = resolve_params(&combos[row]).unwrap();
                adaptive_bandpass_trigger_oscillator_row_from_slice(
                    data,
                    params,
                    in_phase_row,
                    lead_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, (in_phase_row, lead_row)) in in_phase_out
            .chunks_mut(cols)
            .zip(lead_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            adaptive_bandpass_trigger_oscillator_row_from_slice(
                data,
                params,
                in_phase_row,
                lead_row,
            );
        }
    } else {
        for (row, (in_phase_row, lead_row)) in in_phase_out
            .chunks_mut(cols)
            .zip(lead_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            adaptive_bandpass_trigger_oscillator_row_from_slice(
                data,
                params,
                in_phase_row,
                lead_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_bandpass_trigger_oscillator")]
#[pyo3(signature = (data, delta=DEFAULT_DELTA, alpha=DEFAULT_ALPHA, kernel=None))]
pub fn adaptive_bandpass_trigger_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    delta: f64,
    alpha: f64,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
        data,
        AdaptiveBandpassTriggerOscillatorParams {
            delta: Some(delta),
            alpha: Some(alpha),
        },
    );
    let output = py
        .allow_threads(|| adaptive_bandpass_trigger_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.in_phase.into_pyarray(py),
        output.lead.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdaptiveBandpassTriggerOscillatorStream")]
pub struct AdaptiveBandpassTriggerOscillatorStreamPy {
    stream: AdaptiveBandpassTriggerOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdaptiveBandpassTriggerOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (delta=DEFAULT_DELTA, alpha=DEFAULT_ALPHA))]
    fn new(delta: f64, alpha: f64) -> PyResult<Self> {
        let stream = AdaptiveBandpassTriggerOscillatorStream::try_new(
            AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(delta),
                alpha: Some(alpha),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_bandpass_trigger_oscillator_batch")]
#[pyo3(signature = (data, delta_range=(DEFAULT_DELTA, DEFAULT_DELTA, 0.0), alpha_range=(DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0), kernel=None))]
pub fn adaptive_bandpass_trigger_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    delta_range: (f64, f64, f64),
    alpha_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
        delta: delta_range,
        alpha: alpha_range,
    };
    let combos = expand_grid_adaptive_bandpass_trigger_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let in_phase_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let lead_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let in_phase_slice = unsafe { in_phase_arr.as_slice_mut()? };
    let lead_slice = unsafe { lead_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            adaptive_bandpass_trigger_oscillator_batch_inner_into(
                data,
                &sweep,
                batch_kernel.to_non_batch(),
                true,
                in_phase_slice,
                lead_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("in_phase", in_phase_arr.reshape((rows, cols))?)?;
    dict.set_item("lead", lead_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "deltas",
        combos
            .iter()
            .map(|combo| combo.delta.unwrap_or(DEFAULT_DELTA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "alphas",
        combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_adaptive_bandpass_trigger_oscillator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        adaptive_bandpass_trigger_oscillator_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        adaptive_bandpass_trigger_oscillator_batch_py,
        module
    )?)?;
    module.add_class::<AdaptiveBandpassTriggerOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "adaptive_bandpass_trigger_oscillator_js")]
pub fn adaptive_bandpass_trigger_oscillator_js(
    data: &[f64],
    delta: f64,
    alpha: f64,
) -> Result<JsValue, JsValue> {
    let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
        data,
        AdaptiveBandpassTriggerOscillatorParams {
            delta: Some(delta),
            alpha: Some(alpha),
        },
    );
    let output = adaptive_bandpass_trigger_oscillator(&input)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();

    let in_phase = js_sys::Float64Array::new_with_length(output.in_phase.len() as u32);
    in_phase.copy_from(&output.in_phase);
    js_sys::Reflect::set(&result, &JsValue::from_str("in_phase"), &in_phase)?;

    let lead = js_sys::Float64Array::new_with_length(output.lead.len() as u32);
    lead.copy_from(&output.lead);
    js_sys::Reflect::set(&result, &JsValue::from_str("lead"), &lead)?;

    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bandpass_trigger_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bandpass_trigger_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bandpass_trigger_oscillator_into(
    data_ptr: *const f64,
    in_phase_ptr: *mut f64,
    lead_ptr: *mut f64,
    len: usize,
    delta: f64,
    alpha: f64,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || in_phase_ptr.is_null() || lead_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            data,
            AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(delta),
                alpha: Some(alpha),
            },
        );
        let alias = data_ptr == in_phase_ptr || data_ptr == lead_ptr;
        if alias {
            let mut in_phase_tmp = vec![0.0; len];
            let mut lead_tmp = vec![0.0; len];
            adaptive_bandpass_trigger_oscillator_into_slices(
                &mut in_phase_tmp,
                &mut lead_tmp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(in_phase_ptr, len).copy_from_slice(&in_phase_tmp);
            std::slice::from_raw_parts_mut(lead_ptr, len).copy_from_slice(&lead_tmp);
        } else {
            adaptive_bandpass_trigger_oscillator_into_slices(
                std::slice::from_raw_parts_mut(in_phase_ptr, len),
                std::slice::from_raw_parts_mut(lead_ptr, len),
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
pub struct AdaptiveBandpassTriggerOscillatorBatchConfig {
    pub delta_range: (f64, f64, f64),
    pub alpha_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveBandpassTriggerOscillatorBatchJsOutput {
    pub in_phase: Vec<f64>,
    pub lead: Vec<f64>,
    pub combos: Vec<AdaptiveBandpassTriggerOscillatorParams>,
    pub deltas: Vec<f64>,
    pub alphas: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "adaptive_bandpass_trigger_oscillator_batch_js")]
pub fn adaptive_bandpass_trigger_oscillator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AdaptiveBandpassTriggerOscillatorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
        delta: config.delta_range,
        alpha: config.alpha_range,
    };
    let output =
        adaptive_bandpass_trigger_oscillator_batch_inner(data, &sweep, detect_best_kernel(), false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AdaptiveBandpassTriggerOscillatorBatchJsOutput {
        deltas: output
            .combos
            .iter()
            .map(|combo| combo.delta.unwrap_or(DEFAULT_DELTA))
            .collect(),
        alphas: output
            .combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect(),
        in_phase: output.in_phase,
        lead: output.lead,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bandpass_trigger_oscillator_batch_into(
    data_ptr: *const f64,
    in_phase_ptr: *mut f64,
    lead_ptr: *mut f64,
    len: usize,
    delta_start: f64,
    delta_end: f64,
    delta_step: f64,
    alpha_start: f64,
    alpha_end: f64,
    alpha_step: f64,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || in_phase_ptr.is_null() || lead_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
        delta: (delta_start, delta_end, delta_step),
        alpha: (alpha_start, alpha_end, alpha_step),
    };
    let combos = expand_grid_adaptive_bandpass_trigger_oscillator(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        adaptive_bandpass_trigger_oscillator_batch_inner_into(
            data,
            &sweep,
            detect_best_kernel(),
            false,
            std::slice::from_raw_parts_mut(in_phase_ptr, total),
            std::slice::from_raw_parts_mut(lead_ptr, total),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bandpass_trigger_oscillator_output_into_js(
    data: &[f64],
    delta: f64,
    alpha: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_bandpass_trigger_oscillator_js(data, delta, alpha)?;
    crate::write_wasm_object_f64_outputs(
        "adaptive_bandpass_trigger_oscillator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bandpass_trigger_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_bandpass_trigger_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "adaptive_bandpass_trigger_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn sample_close(length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; length];
        let mut prev = 100.0;
        for i in 2..length {
            let x = i as f64;
            let value = prev + (x * 0.07).sin() * 1.3 + (x * 0.03).cos() * 0.6 + x * 0.02;
            out[i] = value;
            prev = value;
        }
        out
    }

    fn assert_series_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            assert!(
                (lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() < 1e-12,
                "series mismatch: left={lhs:?}, right={rhs:?}"
            );
        }
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_output_contract() {
        let close = sample_close(512);
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            &close,
            AdaptiveBandpassTriggerOscillatorParams::default(),
        );
        let out = adaptive_bandpass_trigger_oscillator(&input).unwrap();
        assert_eq!(out.in_phase.len(), close.len());
        assert_eq!(out.lead.len(), close.len());
        assert!(out.in_phase.iter().position(|v| v.is_finite()).unwrap() >= 13);
        assert!(out.lead.iter().position(|v| v.is_finite()).unwrap() >= 14);
        assert!(out.in_phase.last().unwrap().is_finite());
        assert!(out.lead.last().unwrap().is_finite());
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_rejects_invalid_parameters() {
        let close = sample_close(64);
        let err = adaptive_bandpass_trigger_oscillator(
            &AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(0.0),
                    alpha: Some(0.07),
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AdaptiveBandpassTriggerOscillatorError::InvalidDelta { .. }
        ));

        let err = adaptive_bandpass_trigger_oscillator(
            &AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(0.1),
                    alpha: Some(1.0),
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AdaptiveBandpassTriggerOscillatorError::InvalidAlpha { .. }
        ));
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_builder_supports_candles() {
        let candles =
            read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let out = AdaptiveBandpassTriggerOscillatorBuilder::new()
            .delta(0.1)
            .alpha(0.07)
            .apply(&candles)
            .unwrap();
        assert_eq!(out.in_phase.len(), candles.close.len());
        assert_eq!(out.lead.len(), candles.close.len());
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_stream_matches_batch_with_reset() {
        let mut close = sample_close(256);
        close[120] = f64::NAN;
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            &close,
            AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(0.12),
                alpha: Some(0.08),
            },
        );
        let batch = adaptive_bandpass_trigger_oscillator(&input).unwrap();
        let mut stream = AdaptiveBandpassTriggerOscillatorStream::try_new(
            AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(0.12),
                alpha: Some(0.08),
            },
        )
        .unwrap();
        let mut in_phase = Vec::with_capacity(close.len());
        let mut lead = Vec::with_capacity(close.len());
        for value in close {
            match stream.update(value) {
                Some((bp, ld)) => {
                    in_phase.push(bp);
                    lead.push(ld);
                }
                None => {
                    in_phase.push(f64::NAN);
                    lead.push(f64::NAN);
                }
            }
        }
        assert_series_eq(&batch.in_phase, &in_phase);
        assert_series_eq(&batch.lead, &lead);
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_into_matches_api() {
        let close = sample_close(192);
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            &close,
            AdaptiveBandpassTriggerOscillatorParams::default(),
        );
        let direct = adaptive_bandpass_trigger_oscillator(&input).unwrap();
        let mut in_phase = vec![0.0; close.len()];
        let mut lead = vec![0.0; close.len()];
        adaptive_bandpass_trigger_oscillator_into(&input, &mut in_phase, &mut lead).unwrap();
        assert_series_eq(&direct.in_phase, &in_phase);
        assert_series_eq(&direct.lead, &lead);
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_batch_single_param_matches_single() {
        let close = sample_close(192);
        let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
            delta: (0.1, 0.1, 0.0),
            alpha: (0.07, 0.07, 0.0),
        };
        let batch =
            adaptive_bandpass_trigger_oscillator_batch_with_kernel(&close, &sweep, Kernel::Auto)
                .unwrap();
        let single = adaptive_bandpass_trigger_oscillator(
            &AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams::default(),
            ),
        )
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_eq(&batch.in_phase[..close.len()], single.in_phase.as_slice());
        assert_series_eq(&batch.lead[..close.len()], single.lead.as_slice());
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_batch_metadata() {
        let close = sample_close(160);
        let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
            delta: (0.08, 0.12, 0.04),
            alpha: (0.05, 0.09, 0.02),
        };
        let batch =
            adaptive_bandpass_trigger_oscillator_batch_with_kernel(&close, &sweep, Kernel::Auto)
                .unwrap();
        assert_eq!(batch.rows, 6);
        assert_eq!(batch.cols, close.len());
        assert_eq!(batch.in_phase.len(), 6 * close.len());
        assert_eq!(batch.lead.len(), 6 * close.len());
    }
}
