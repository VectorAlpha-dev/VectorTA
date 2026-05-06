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

use crate::indicators::dispatch::{
    compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
    ParamValue,
};
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
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_MIN_RANGE: usize = 10;
const DEFAULT_MAX_RANGE: usize = 100;
const DEFAULT_STEP: usize = 5;
const DEFAULT_SIGNAL_LINE: usize = 7;

#[derive(Debug, Clone)]
pub enum RegressionSlopeOscillatorData<'a> {
    Candles { candles: &'a Candles },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RegressionSlopeOscillatorOutput {
    pub value: Vec<f64>,
    pub signal: Vec<f64>,
    pub bullish_reversal: Vec<f64>,
    pub bearish_reversal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RegressionSlopeOscillatorParams {
    pub min_range: Option<usize>,
    pub max_range: Option<usize>,
    pub step: Option<usize>,
    pub signal_line: Option<usize>,
}

impl Default for RegressionSlopeOscillatorParams {
    fn default() -> Self {
        Self {
            min_range: Some(DEFAULT_MIN_RANGE),
            max_range: Some(DEFAULT_MAX_RANGE),
            step: Some(DEFAULT_STEP),
            signal_line: Some(DEFAULT_SIGNAL_LINE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegressionSlopeOscillatorInput<'a> {
    pub data: RegressionSlopeOscillatorData<'a>,
    pub params: RegressionSlopeOscillatorParams,
}

impl<'a> RegressionSlopeOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: RegressionSlopeOscillatorParams) -> Self {
        Self {
            data: RegressionSlopeOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: RegressionSlopeOscillatorParams) -> Self {
        Self {
            data: RegressionSlopeOscillatorData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, RegressionSlopeOscillatorParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RegressionSlopeOscillatorBuilder {
    min_range: Option<usize>,
    max_range: Option<usize>,
    step: Option<usize>,
    signal_line: Option<usize>,
    kernel: Kernel,
}

impl Default for RegressionSlopeOscillatorBuilder {
    fn default() -> Self {
        Self {
            min_range: None,
            max_range: None,
            step: None,
            signal_line: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RegressionSlopeOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn min_range(mut self, value: usize) -> Self {
        self.min_range = Some(value);
        self
    }

    #[inline(always)]
    pub fn max_range(mut self, value: usize) -> Self {
        self.max_range = Some(value);
        self
    }

    #[inline(always)]
    pub fn step(mut self, value: usize) -> Self {
        self.step = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_line(mut self, value: usize) -> Self {
        self.signal_line = Some(value);
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
    ) -> Result<RegressionSlopeOscillatorOutput, RegressionSlopeOscillatorError> {
        let input = RegressionSlopeOscillatorInput::from_candles(
            candles,
            RegressionSlopeOscillatorParams {
                min_range: self.min_range,
                max_range: self.max_range,
                step: self.step,
                signal_line: self.signal_line,
            },
        );
        regression_slope_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<RegressionSlopeOscillatorOutput, RegressionSlopeOscillatorError> {
        let input = RegressionSlopeOscillatorInput::from_slice(
            data,
            RegressionSlopeOscillatorParams {
                min_range: self.min_range,
                max_range: self.max_range,
                step: self.step,
                signal_line: self.signal_line,
            },
        );
        regression_slope_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<RegressionSlopeOscillatorStream, RegressionSlopeOscillatorError> {
        RegressionSlopeOscillatorStream::try_new(RegressionSlopeOscillatorParams {
            min_range: self.min_range,
            max_range: self.max_range,
            step: self.step,
            signal_line: self.signal_line,
        })
    }
}

#[derive(Debug, Error)]
pub enum RegressionSlopeOscillatorError {
    #[error("regression_slope_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("regression_slope_oscillator: All values are NaN, non-finite, or non-positive.")]
    AllValuesInvalidForLog,
    #[error("regression_slope_oscillator: Invalid min_range: {min_range}")]
    InvalidMinRange { min_range: usize },
    #[error("regression_slope_oscillator: Invalid max_range: {max_range}")]
    InvalidMaxRange { max_range: usize },
    #[error("regression_slope_oscillator: Invalid step: {step}")]
    InvalidStep { step: usize },
    #[error("regression_slope_oscillator: Invalid signal_line: {signal_line}")]
    InvalidSignalLine { signal_line: usize },
    #[error("regression_slope_oscillator: Invalid range config: min_range={min_range}, max_range={max_range}, step={step}")]
    InvalidRangeConfig {
        min_range: usize,
        max_range: usize,
        step: usize,
    },
    #[error("regression_slope_oscillator: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("regression_slope_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("regression_slope_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct LengthSpec {
    length: usize,
    length_f64: f64,
    sum_x: f64,
    denominator: f64,
}

#[derive(Debug, Clone)]
struct ResolvedParams {
    min_range: usize,
    max_range: usize,
    step: usize,
    signal_line: usize,
    specs: Vec<LengthSpec>,
    value_warmup: usize,
    signal_warmup: usize,
}

#[inline(always)]
fn build_length_spec(length: usize) -> LengthSpec {
    let length_f64 = length as f64;
    let sum_x = length_f64 * (length_f64 + 1.0) * 0.5;
    let sum_x_sqr = length_f64 * (length_f64 + 1.0) * (2.0 * length_f64 + 1.0) / 6.0;
    LengthSpec {
        length,
        length_f64,
        sum_x,
        denominator: length_f64 * sum_x_sqr - sum_x * sum_x,
    }
}

#[inline(always)]
fn expand_specs(min_range: usize, max_range: usize, step: usize) -> Vec<LengthSpec> {
    let mut specs = Vec::new();
    let mut current = min_range;
    loop {
        specs.push(build_length_spec(current));
        if current >= max_range {
            break;
        }
        let next = current.saturating_add(step);
        if next > max_range {
            break;
        }
        current = next;
    }
    specs
}

#[inline(always)]
fn extract_slice<'a>(
    input: &'a RegressionSlopeOscillatorInput<'a>,
) -> Result<&'a [f64], RegressionSlopeOscillatorError> {
    let data = match &input.data {
        RegressionSlopeOscillatorData::Candles { candles } => candles.close.as_slice(),
        RegressionSlopeOscillatorData::Slice(values) => *values,
    };
    if data.is_empty() {
        return Err(RegressionSlopeOscillatorError::EmptyInputData);
    }
    Ok(data)
}

#[inline(always)]
fn first_valid_positive(data: &[f64]) -> Option<usize> {
    data.iter().position(|v| v.is_finite() && *v > 0.0)
}

#[inline(always)]
fn resolve_params(
    params: &RegressionSlopeOscillatorParams,
) -> Result<ResolvedParams, RegressionSlopeOscillatorError> {
    let min_range = params.min_range.unwrap_or(DEFAULT_MIN_RANGE);
    let max_range = params.max_range.unwrap_or(DEFAULT_MAX_RANGE);
    let step = params.step.unwrap_or(DEFAULT_STEP);
    let signal_line = params.signal_line.unwrap_or(DEFAULT_SIGNAL_LINE);

    if min_range < 2 {
        return Err(RegressionSlopeOscillatorError::InvalidMinRange { min_range });
    }
    if max_range < 2 {
        return Err(RegressionSlopeOscillatorError::InvalidMaxRange { max_range });
    }
    if step == 0 {
        return Err(RegressionSlopeOscillatorError::InvalidStep { step });
    }
    if signal_line == 0 {
        return Err(RegressionSlopeOscillatorError::InvalidSignalLine { signal_line });
    }
    if min_range > max_range {
        return Err(RegressionSlopeOscillatorError::InvalidRangeConfig {
            min_range,
            max_range,
            step,
        });
    }

    let specs = expand_specs(min_range, max_range, step);
    if specs.is_empty() {
        return Err(RegressionSlopeOscillatorError::InvalidRangeConfig {
            min_range,
            max_range,
            step,
        });
    }

    Ok(ResolvedParams {
        min_range,
        max_range,
        step,
        signal_line,
        specs,
        value_warmup: max_range.saturating_sub(1),
        signal_warmup: max_range
            .saturating_sub(1)
            .saturating_add(signal_line.saturating_sub(1)),
    })
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a RegressionSlopeOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, Kernel), RegressionSlopeOscillatorError> {
    let data = extract_slice(input)?;
    let params = resolve_params(&input.params)?;
    let first =
        first_valid_positive(data).ok_or(RegressionSlopeOscillatorError::AllValuesInvalidForLog)?;
    Ok((data, params, first, kernel.to_non_batch()))
}

#[inline(always)]
fn check_output_len(out: &[f64], expected: usize) -> Result<(), RegressionSlopeOscillatorError> {
    if out.len() != expected {
        return Err(RegressionSlopeOscillatorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn build_prefixes(data: &[f64]) -> (Option<Vec<usize>>, Vec<f64>, Vec<f64>) {
    let len = data.len();
    let mut sum_prefix = vec![0.0; len + 1];
    let mut weighted_prefix = vec![0.0; len + 1];
    let mut invalid_prefix = Vec::new();
    let mut invalid_count = 0usize;

    for i in 0..len {
        sum_prefix[i + 1] = sum_prefix[i];
        weighted_prefix[i + 1] = weighted_prefix[i];
        let value = data[i];
        if value.is_finite() && value > 0.0 {
            let logged = value.ln();
            sum_prefix[i + 1] += logged;
            weighted_prefix[i + 1] += logged * i as f64;
            if !invalid_prefix.is_empty() {
                invalid_prefix.push(invalid_count);
            }
        } else {
            if invalid_prefix.is_empty() {
                invalid_prefix.resize(i + 1, 0);
            }
            invalid_count += 1;
            invalid_prefix.push(invalid_count);
        }
    }

    let invalid_prefix = if invalid_prefix.is_empty() {
        None
    } else {
        Some(invalid_prefix)
    };
    (invalid_prefix, sum_prefix, weighted_prefix)
}

#[inline(always)]
fn window_has_invalid(
    invalid_prefix: Option<&[usize]>,
    start: usize,
    end_exclusive: usize,
) -> bool {
    match invalid_prefix {
        Some(prefix) => prefix[end_exclusive] != prefix[start],
        None => false,
    }
}

#[inline(always)]
fn slope_from_prefix(
    sum_prefix: &[f64],
    weighted_prefix: &[f64],
    start: usize,
    spec: LengthSpec,
) -> f64 {
    let end = start + spec.length;
    let sum_y = sum_prefix[end] - sum_prefix[start];
    let weighted_abs = weighted_prefix[end] - weighted_prefix[start];
    let weighted_rel = weighted_abs + (1.0 - start as f64) * sum_y;
    (spec.length_f64 * weighted_rel - spec.sum_x * sum_y) / spec.denominator
}

#[inline(always)]
fn fill_signal_and_reversals(
    params: &ResolvedParams,
    out_value: &[f64],
    out_signal: &mut [f64],
    out_bullish_reversal: &mut [f64],
    out_bearish_reversal: &mut [f64],
) -> Result<(), RegressionSlopeOscillatorError> {
    let len = out_value.len();
    check_output_len(out_signal, len)?;
    check_output_len(out_bullish_reversal, len)?;
    check_output_len(out_bearish_reversal, len)?;

    let mut signal_ring = vec![0.0; params.signal_line];
    let mut signal_head = 0usize;
    let mut signal_count = 0usize;
    let mut sum = 0.0;

    for i in 0..len {
        let value = out_value[i];
        if value.is_finite() {
            if signal_count < params.signal_line {
                signal_ring[signal_count] = value;
                signal_count += 1;
                sum += value;
            } else {
                let old = signal_ring[signal_head];
                signal_ring[signal_head] = value;
                sum += value - old;
                signal_head += 1;
                if signal_head == params.signal_line {
                    signal_head = 0;
                }
            }
        }

        out_signal[i] = if signal_count == params.signal_line {
            sum / params.signal_line as f64
        } else {
            f64::NAN
        };

        if value.is_finite() && out_signal[i].is_finite() {
            let prev_value = if i > 0 { out_value[i - 1] } else { f64::NAN };
            let prev_signal = if i > 0 { out_signal[i - 1] } else { f64::NAN };
            out_bearish_reversal[i] = if prev_value.is_finite()
                && prev_signal.is_finite()
                && value < out_signal[i]
                && prev_value >= prev_signal
                && value > 0.0
            {
                1.0
            } else {
                0.0
            };
            out_bullish_reversal[i] = if prev_value.is_finite()
                && prev_signal.is_finite()
                && value > out_signal[i]
                && prev_value <= prev_signal
                && value < 0.0
            {
                1.0
            } else {
                0.0
            };
        } else {
            out_bearish_reversal[i] = f64::NAN;
            out_bullish_reversal[i] = f64::NAN;
        }
    }

    Ok(())
}

fn regression_slope_oscillator_compute_into(
    data: &[f64],
    params: &ResolvedParams,
    out_value: &mut [f64],
    out_signal: &mut [f64],
    out_bullish_reversal: &mut [f64],
    out_bearish_reversal: &mut [f64],
) -> Result<(), RegressionSlopeOscillatorError> {
    let len = data.len();
    check_output_len(out_value, len)?;
    check_output_len(out_signal, len)?;
    check_output_len(out_bullish_reversal, len)?;
    check_output_len(out_bearish_reversal, len)?;
    out_value[..params.value_warmup.min(len)].fill(f64::NAN);

    let (invalid_prefix, sum_prefix, weighted_prefix) = build_prefixes(data);
    let invalid_prefix = invalid_prefix.as_deref();
    let spec_count = params.specs.len() as f64;

    for i in params.value_warmup..len {
        let max_start = i + 1 - params.max_range;
        if window_has_invalid(invalid_prefix, max_start, i + 1) {
            out_value[i] = f64::NAN;
            continue;
        }

        let mut sum_slopes = 0.0;
        for spec in &params.specs {
            let start = i + 1 - spec.length;
            sum_slopes += slope_from_prefix(&sum_prefix, &weighted_prefix, start, *spec);
        }
        out_value[i] = sum_slopes / spec_count;
    }

    fill_signal_and_reversals(
        params,
        out_value,
        out_signal,
        out_bullish_reversal,
        out_bearish_reversal,
    )
}

#[inline]
pub fn regression_slope_oscillator(
    input: &RegressionSlopeOscillatorInput,
) -> Result<RegressionSlopeOscillatorOutput, RegressionSlopeOscillatorError> {
    regression_slope_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn regression_slope_oscillator_with_kernel(
    input: &RegressionSlopeOscillatorInput,
    kernel: Kernel,
) -> Result<RegressionSlopeOscillatorOutput, RegressionSlopeOscillatorError> {
    let (data, params, _, _) = validate_input(input, kernel)?;
    let len = data.len();
    let mut value = alloc_with_nan_prefix(len, 0);
    let mut signal = alloc_with_nan_prefix(len, 0);
    let mut bullish_reversal = alloc_with_nan_prefix(len, 0);
    let mut bearish_reversal = alloc_with_nan_prefix(len, 0);
    regression_slope_oscillator_compute_into(
        data,
        &params,
        &mut value,
        &mut signal,
        &mut bullish_reversal,
        &mut bearish_reversal,
    )?;
    Ok(RegressionSlopeOscillatorOutput {
        value,
        signal,
        bullish_reversal,
        bearish_reversal,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[allow(clippy::too_many_arguments)]
pub fn regression_slope_oscillator_into(
    out_value: &mut [f64],
    out_signal: &mut [f64],
    out_bullish_reversal: &mut [f64],
    out_bearish_reversal: &mut [f64],
    input: &RegressionSlopeOscillatorInput,
    kernel: Kernel,
) -> Result<(), RegressionSlopeOscillatorError> {
    regression_slope_oscillator_into_slice(
        out_value,
        out_signal,
        out_bullish_reversal,
        out_bearish_reversal,
        input,
        kernel,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn regression_slope_oscillator_into_slice(
    out_value: &mut [f64],
    out_signal: &mut [f64],
    out_bullish_reversal: &mut [f64],
    out_bearish_reversal: &mut [f64],
    input: &RegressionSlopeOscillatorInput,
    kernel: Kernel,
) -> Result<(), RegressionSlopeOscillatorError> {
    let (data, params, _, _) = validate_input(input, kernel)?;
    regression_slope_oscillator_compute_into(
        data,
        &params,
        out_value,
        out_signal,
        out_bullish_reversal,
        out_bearish_reversal,
    )
}

#[derive(Debug)]
struct RegressionSlopeLengthState {
    spec: LengthSpec,
    window: VecDeque<f64>,
    sum: f64,
    weighted: f64,
    valid_count: usize,
    prev_full_valid: bool,
}

impl RegressionSlopeLengthState {
    fn new(spec: LengthSpec) -> Self {
        Self {
            spec,
            window: VecDeque::with_capacity(spec.length),
            sum: 0.0,
            weighted: 0.0,
            valid_count: 0,
            prev_full_valid: false,
        }
    }

    fn update(&mut self, value: f64) -> f64 {
        let old_sum = self.sum;
        let popped = if self.window.len() == self.spec.length {
            let old = self.window.pop_front().unwrap();
            if old.is_finite() {
                self.sum -= old;
                self.valid_count -= 1;
            }
            Some(old)
        } else {
            None
        };

        self.window.push_back(value);
        if value.is_finite() {
            self.sum += value;
            self.valid_count += 1;
        }

        let full_valid =
            self.window.len() == self.spec.length && self.valid_count == self.spec.length;
        if full_valid {
            if self.prev_full_valid && popped.is_some() && value.is_finite() {
                self.weighted = self.weighted - old_sum + self.spec.length_f64 * value;
            } else {
                self.weighted = 0.0;
                for (idx, entry) in self.window.iter().enumerate() {
                    self.weighted += *entry * (idx + 1) as f64;
                }
            }
            self.prev_full_valid = true;
            (self.spec.length_f64 * self.weighted - self.spec.sum_x * self.sum)
                / self.spec.denominator
        } else {
            self.prev_full_valid = false;
            self.weighted = 0.0;
            f64::NAN
        }
    }
}

#[derive(Debug)]
pub struct RegressionSlopeOscillatorStream {
    params: ResolvedParams,
    states: Vec<RegressionSlopeLengthState>,
    signal_values: VecDeque<f64>,
    signal_sum: f64,
    prev_value: f64,
    prev_signal: f64,
}

impl RegressionSlopeOscillatorStream {
    pub fn try_new(
        params: RegressionSlopeOscillatorParams,
    ) -> Result<Self, RegressionSlopeOscillatorError> {
        let params = resolve_params(&params)?;
        let states = params
            .specs
            .iter()
            .copied()
            .map(RegressionSlopeLengthState::new)
            .collect();
        Ok(Self {
            params,
            states,
            signal_values: VecDeque::new(),
            signal_sum: 0.0,
            prev_value: f64::NAN,
            prev_signal: f64::NAN,
        })
    }

    pub fn update(&mut self, value: f64) -> (f64, f64, f64, f64) {
        let logged = if value.is_finite() && value > 0.0 {
            value.ln()
        } else {
            f64::NAN
        };
        let mut sum_slopes = 0.0;
        let mut full_valid = true;
        for state in &mut self.states {
            let slope = state.update(logged);
            if slope.is_finite() {
                sum_slopes += slope;
            } else {
                full_valid = false;
            }
        }

        let oscillator = if full_valid {
            sum_slopes / self.states.len() as f64
        } else {
            f64::NAN
        };

        if oscillator.is_finite() {
            self.signal_values.push_back(oscillator);
            self.signal_sum += oscillator;
            if self.signal_values.len() > self.params.signal_line {
                if let Some(old) = self.signal_values.pop_front() {
                    self.signal_sum -= old;
                }
            }
        }

        let signal = if self.signal_values.len() == self.params.signal_line {
            self.signal_sum / self.params.signal_line as f64
        } else {
            f64::NAN
        };

        let bullish_reversal = if oscillator.is_finite()
            && signal.is_finite()
            && self.prev_value.is_finite()
            && self.prev_signal.is_finite()
            && oscillator > signal
            && self.prev_value <= self.prev_signal
            && oscillator < 0.0
        {
            1.0
        } else if oscillator.is_finite() && signal.is_finite() {
            0.0
        } else {
            f64::NAN
        };

        let bearish_reversal = if oscillator.is_finite()
            && signal.is_finite()
            && self.prev_value.is_finite()
            && self.prev_signal.is_finite()
            && oscillator < signal
            && self.prev_value >= self.prev_signal
            && oscillator > 0.0
        {
            1.0
        } else if oscillator.is_finite() && signal.is_finite() {
            0.0
        } else {
            f64::NAN
        };

        self.prev_value = oscillator;
        self.prev_signal = signal;
        (oscillator, signal, bullish_reversal, bearish_reversal)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RegressionSlopeOscillatorBatchRange {
    pub min_range: (usize, usize, usize),
    pub max_range: (usize, usize, usize),
    pub step: (usize, usize, usize),
    pub signal_line: (usize, usize, usize),
}

impl Default for RegressionSlopeOscillatorBatchRange {
    fn default() -> Self {
        Self {
            min_range: (DEFAULT_MIN_RANGE, DEFAULT_MIN_RANGE, 0),
            max_range: (DEFAULT_MAX_RANGE, DEFAULT_MAX_RANGE, 0),
            step: (DEFAULT_STEP, DEFAULT_STEP, 0),
            signal_line: (DEFAULT_SIGNAL_LINE, DEFAULT_SIGNAL_LINE, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegressionSlopeOscillatorBatchOutput {
    pub value: Vec<f64>,
    pub signal: Vec<f64>,
    pub bullish_reversal: Vec<f64>,
    pub bearish_reversal: Vec<f64>,
    pub combos: Vec<RegressionSlopeOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RegressionSlopeOscillatorBatchBuilder {
    range: RegressionSlopeOscillatorBatchRange,
    kernel: Kernel,
}

impl RegressionSlopeOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: RegressionSlopeOscillatorBatchRange) -> Self {
        self.range = value;
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
    ) -> Result<RegressionSlopeOscillatorBatchOutput, RegressionSlopeOscillatorError> {
        regression_slope_oscillator_batch_with_kernel(
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<RegressionSlopeOscillatorBatchOutput, RegressionSlopeOscillatorError> {
        regression_slope_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }
}

fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, RegressionSlopeOscillatorError> {
    if start > end || (start != end && step == 0) {
        return Err(RegressionSlopeOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        current = current.checked_add(step).ok_or_else(|| {
            RegressionSlopeOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            }
        })?;
        if current > end && out.last().copied() != Some(end) {
            break;
        }
        if out.len() > 1_000_000 {
            return Err(RegressionSlopeOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
    }
    Ok(out)
}

pub fn regression_slope_oscillator_expand_grid(
    sweep: &RegressionSlopeOscillatorBatchRange,
) -> Result<Vec<RegressionSlopeOscillatorParams>, RegressionSlopeOscillatorError> {
    let mins = expand_usize_range(sweep.min_range.0, sweep.min_range.1, sweep.min_range.2)?;
    let maxes = expand_usize_range(sweep.max_range.0, sweep.max_range.1, sweep.max_range.2)?;
    let steps = expand_usize_range(sweep.step.0, sweep.step.1, sweep.step.2)?;
    let signals = expand_usize_range(
        sweep.signal_line.0,
        sweep.signal_line.1,
        sweep.signal_line.2,
    )?;
    let mut out = Vec::new();
    for min_range in mins {
        for &max_range in &maxes {
            for &step in &steps {
                for &signal_line in &signals {
                    out.push(RegressionSlopeOscillatorParams {
                        min_range: Some(min_range),
                        max_range: Some(max_range),
                        step: Some(step),
                        signal_line: Some(signal_line),
                    });
                }
            }
        }
    }
    Ok(out)
}

fn validate_raw_slice(data: &[f64]) -> Result<usize, RegressionSlopeOscillatorError> {
    if data.is_empty() {
        return Err(RegressionSlopeOscillatorError::EmptyInputData);
    }
    first_valid_positive(data).ok_or(RegressionSlopeOscillatorError::AllValuesInvalidForLog)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, RegressionSlopeOscillatorError> {
    rows.checked_mul(cols)
        .ok_or_else(|| RegressionSlopeOscillatorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

pub fn regression_slope_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &RegressionSlopeOscillatorBatchRange,
    kernel: Kernel,
) -> Result<RegressionSlopeOscillatorBatchOutput, RegressionSlopeOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(RegressionSlopeOscillatorError::InvalidKernelForBatch(
                kernel,
            ))
        }
    };
    regression_slope_oscillator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn regression_slope_oscillator_batch_slice(
    data: &[f64],
    sweep: &RegressionSlopeOscillatorBatchRange,
    kernel: Kernel,
) -> Result<RegressionSlopeOscillatorBatchOutput, RegressionSlopeOscillatorError> {
    regression_slope_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn regression_slope_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &RegressionSlopeOscillatorBatchRange,
    kernel: Kernel,
) -> Result<RegressionSlopeOscillatorBatchOutput, RegressionSlopeOscillatorError> {
    regression_slope_oscillator_batch_inner(data, sweep, kernel, true)
}

fn regression_slope_oscillator_batch_inner(
    data: &[f64],
    sweep: &RegressionSlopeOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<RegressionSlopeOscillatorBatchOutput, RegressionSlopeOscillatorError> {
    let combos = regression_slope_oscillator_expand_grid(sweep)?;
    validate_raw_slice(data)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;

    let resolved = combos
        .iter()
        .map(resolve_params)
        .collect::<Result<Vec<_>, _>>()?;
    let value_warmups = resolved.iter().map(|p| p.value_warmup).collect::<Vec<_>>();
    let signal_warmups = resolved.iter().map(|p| p.signal_warmup).collect::<Vec<_>>();

    let mut value_buf = make_uninit_matrix(rows, cols);
    let mut signal_buf = make_uninit_matrix(rows, cols);
    let mut bullish_buf = make_uninit_matrix(rows, cols);
    let mut bearish_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut value_buf, cols, &value_warmups);
    init_matrix_prefixes(&mut signal_buf, cols, &signal_warmups);
    init_matrix_prefixes(&mut bullish_buf, cols, &signal_warmups);
    init_matrix_prefixes(&mut bearish_buf, cols, &signal_warmups);

    let mut value_guard = ManuallyDrop::new(value_buf);
    let mut signal_guard = ManuallyDrop::new(signal_buf);
    let mut bullish_guard = ManuallyDrop::new(bullish_buf);
    let mut bearish_guard = ManuallyDrop::new(bearish_buf);

    let out_value = unsafe {
        core::slice::from_raw_parts_mut(value_guard.as_mut_ptr() as *mut f64, value_guard.len())
    };
    let out_signal = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };
    let out_bullish = unsafe {
        core::slice::from_raw_parts_mut(bullish_guard.as_mut_ptr() as *mut f64, bullish_guard.len())
    };
    let out_bearish = unsafe {
        core::slice::from_raw_parts_mut(bearish_guard.as_mut_ptr() as *mut f64, bearish_guard.len())
    };

    regression_slope_oscillator_batch_inner_into(
        data,
        sweep,
        parallel,
        out_value,
        out_signal,
        out_bullish,
        out_bearish,
    )?;

    let value = unsafe {
        Vec::from_raw_parts(
            value_guard.as_mut_ptr() as *mut f64,
            total,
            value_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            total,
            signal_guard.capacity(),
        )
    };
    let bullish_reversal = unsafe {
        Vec::from_raw_parts(
            bullish_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_guard.capacity(),
        )
    };
    let bearish_reversal = unsafe {
        Vec::from_raw_parts(
            bearish_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_guard.capacity(),
        )
    };

    Ok(RegressionSlopeOscillatorBatchOutput {
        value,
        signal,
        bullish_reversal,
        bearish_reversal,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn regression_slope_oscillator_batch_into_slice(
    out_value: &mut [f64],
    out_signal: &mut [f64],
    out_bullish_reversal: &mut [f64],
    out_bearish_reversal: &mut [f64],
    data: &[f64],
    sweep: &RegressionSlopeOscillatorBatchRange,
    _kernel: Kernel,
) -> Result<(), RegressionSlopeOscillatorError> {
    regression_slope_oscillator_batch_inner_into(
        data,
        sweep,
        false,
        out_value,
        out_signal,
        out_bullish_reversal,
        out_bearish_reversal,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn regression_slope_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &RegressionSlopeOscillatorBatchRange,
    parallel: bool,
    out_value: &mut [f64],
    out_signal: &mut [f64],
    out_bullish_reversal: &mut [f64],
    out_bearish_reversal: &mut [f64],
) -> Result<Vec<RegressionSlopeOscillatorParams>, RegressionSlopeOscillatorError> {
    let combos = regression_slope_oscillator_expand_grid(sweep)?;
    validate_raw_slice(data)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;
    check_output_len(out_value, total)?;
    check_output_len(out_signal, total)?;
    check_output_len(out_bullish_reversal, total)?;
    check_output_len(out_bearish_reversal, total)?;

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        let results: Vec<Result<(), RegressionSlopeOscillatorError>> = out_value
            .par_chunks_mut(cols)
            .zip(out_signal.par_chunks_mut(cols))
            .zip(out_bullish_reversal.par_chunks_mut(cols))
            .zip(out_bearish_reversal.par_chunks_mut(cols))
            .zip(combos.par_iter())
            .map(
                |((((value_row, signal_row), bullish_row), bearish_row), combo)| {
                    let params = resolve_params(combo)?;
                    regression_slope_oscillator_compute_into(
                        data,
                        &params,
                        value_row,
                        signal_row,
                        bullish_row,
                        bearish_row,
                    )
                },
            )
            .collect();
        for result in results {
            result?;
        }
    }

    if !parallel || cfg!(target_arch = "wasm32") {
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(combo)?;
            regression_slope_oscillator_compute_into(
                data,
                &params,
                &mut out_value[start..end],
                &mut out_signal[start..end],
                &mut out_bullish_reversal[start..end],
                &mut out_bearish_reversal[start..end],
            )?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "regression_slope_oscillator")]
#[pyo3(signature = (
    data,
    min_range=10,
    max_range=100,
    step=5,
    signal_line=7,
    kernel=None
))]
pub fn regression_slope_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_range: usize,
    max_range: usize,
    step: usize,
    signal_line: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let input = RegressionSlopeOscillatorInput::from_slice(
        data,
        RegressionSlopeOscillatorParams {
            min_range: Some(min_range),
            max_range: Some(max_range),
            step: Some(step),
            signal_line: Some(signal_line),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| regression_slope_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("value", out.value.into_pyarray(py))?;
    dict.set_item("signal", out.signal.into_pyarray(py))?;
    dict.set_item("bullish_reversal", out.bullish_reversal.into_pyarray(py))?;
    dict.set_item("bearish_reversal", out.bearish_reversal.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "RegressionSlopeOscillatorStream")]
pub struct RegressionSlopeOscillatorStreamPy {
    stream: RegressionSlopeOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RegressionSlopeOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (min_range=10, max_range=100, step=5, signal_line=7))]
    fn new(min_range: usize, max_range: usize, step: usize, signal_line: usize) -> PyResult<Self> {
        let stream = RegressionSlopeOscillatorStream::try_new(RegressionSlopeOscillatorParams {
            min_range: Some(min_range),
            max_range: Some(max_range),
            step: Some(step),
            signal_line: Some(signal_line),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> (f64, f64, f64, f64) {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "regression_slope_oscillator_batch")]
#[pyo3(signature = (
    data,
    min_range_range=(10,10,0),
    max_range_range=(100,100,0),
    step_range=(5,5,0),
    signal_line_range=(7,7,0),
    kernel=None
))]
pub fn regression_slope_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_range_range: (usize, usize, usize),
    max_range_range: (usize, usize, usize),
    step_range: (usize, usize, usize),
    signal_line_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = RegressionSlopeOscillatorBatchRange {
        min_range: min_range_range,
        max_range: max_range_range,
        step: step_range,
        signal_line: signal_line_range,
    };
    let combos = regression_slope_oscillator_expand_grid(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_value = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_signal = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_bullish = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_bearish = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let value_slice = unsafe { out_value.as_slice_mut()? };
    let signal_slice = unsafe { out_signal.as_slice_mut()? };
    let bullish_slice = unsafe { out_bullish.as_slice_mut()? };
    let bearish_slice = unsafe { out_bearish.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        regression_slope_oscillator_batch_inner_into(
            data,
            &sweep,
            batch_kernel.is_batch(),
            value_slice,
            signal_slice,
            bullish_slice,
            bearish_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("value", out_value.reshape((rows, cols))?)?;
    dict.set_item("signal", out_signal.reshape((rows, cols))?)?;
    dict.set_item("bullish_reversal", out_bullish.reshape((rows, cols))?)?;
    dict.set_item("bearish_reversal", out_bearish.reshape((rows, cols))?)?;
    dict.set_item(
        "min_ranges",
        combos
            .iter()
            .map(|combo| combo.min_range.unwrap_or(DEFAULT_MIN_RANGE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "max_ranges",
        combos
            .iter()
            .map(|combo| combo.max_range.unwrap_or(DEFAULT_MAX_RANGE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "steps",
        combos
            .iter()
            .map(|combo| combo.step.unwrap_or(DEFAULT_STEP))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_lines",
        combos
            .iter()
            .map(|combo| combo.signal_line.unwrap_or(DEFAULT_SIGNAL_LINE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_regression_slope_oscillator_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regression_slope_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(regression_slope_oscillator_batch_py, m)?)?;
    m.add_class::<RegressionSlopeOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RegressionSlopeOscillatorJsOutput {
    pub value: Vec<f64>,
    pub signal: Vec<f64>,
    pub bullish_reversal: Vec<f64>,
    pub bearish_reversal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "regression_slope_oscillator_js")]
pub fn regression_slope_oscillator_js(
    data: &[f64],
    min_range: usize,
    max_range: usize,
    step: usize,
    signal_line: usize,
) -> Result<JsValue, JsValue> {
    let input = RegressionSlopeOscillatorInput::from_slice(
        data,
        RegressionSlopeOscillatorParams {
            min_range: Some(min_range),
            max_range: Some(max_range),
            step: Some(step),
            signal_line: Some(signal_line),
        },
    );
    let out = regression_slope_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&RegressionSlopeOscillatorJsOutput {
        value: out.value,
        signal: out.signal,
        bullish_reversal: out.bullish_reversal,
        bearish_reversal: out.bearish_reversal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RegressionSlopeOscillatorBatchConfig {
    pub min_range_range: Vec<f64>,
    pub max_range_range: Vec<f64>,
    pub step_range: Vec<f64>,
    pub signal_line_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RegressionSlopeOscillatorBatchJsOutput {
    pub value: Vec<f64>,
    pub signal: Vec<f64>,
    pub bullish_reversal: Vec<f64>,
    pub bearish_reversal: Vec<f64>,
    pub min_ranges: Vec<usize>,
    pub max_ranges: Vec<usize>,
    pub steps: Vec<usize>,
    pub signal_lines: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name} values must be non-negative integers"
            )));
        }
        out[idx] = *value as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "regression_slope_oscillator_batch_js")]
pub fn regression_slope_oscillator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: RegressionSlopeOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = RegressionSlopeOscillatorBatchRange {
        min_range: js_vec3_to_usize("min_range_range", &config.min_range_range)?,
        max_range: js_vec3_to_usize("max_range_range", &config.max_range_range)?,
        step: js_vec3_to_usize("step_range", &config.step_range)?,
        signal_line: js_vec3_to_usize("signal_line_range", &config.signal_line_range)?,
    };
    let out = regression_slope_oscillator_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&RegressionSlopeOscillatorBatchJsOutput {
        value: out.value,
        signal: out.signal,
        bullish_reversal: out.bullish_reversal,
        bearish_reversal: out.bearish_reversal,
        min_ranges: out
            .combos
            .iter()
            .map(|combo| combo.min_range.unwrap_or(DEFAULT_MIN_RANGE))
            .collect(),
        max_ranges: out
            .combos
            .iter()
            .map(|combo| combo.max_range.unwrap_or(DEFAULT_MAX_RANGE))
            .collect(),
        steps: out
            .combos
            .iter()
            .map(|combo| combo.step.unwrap_or(DEFAULT_STEP))
            .collect(),
        signal_lines: out
            .combos
            .iter()
            .map(|combo| combo.signal_line.unwrap_or(DEFAULT_SIGNAL_LINE))
            .collect(),
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn regression_slope_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn regression_slope_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn regression_slope_oscillator_into(
    in_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_signal_ptr: *mut f64,
    out_bullish_reversal_ptr: *mut f64,
    out_bearish_reversal_ptr: *mut f64,
    len: usize,
    min_range: usize,
    max_range: usize,
    step: usize,
    signal_line: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || out_value_ptr.is_null()
        || out_signal_ptr.is_null()
        || out_bullish_reversal_ptr.is_null()
        || out_bearish_reversal_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to regression_slope_oscillator_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, len);
        let out_signal = std::slice::from_raw_parts_mut(out_signal_ptr, len);
        let out_bullish = std::slice::from_raw_parts_mut(out_bullish_reversal_ptr, len);
        let out_bearish = std::slice::from_raw_parts_mut(out_bearish_reversal_ptr, len);
        let input = RegressionSlopeOscillatorInput::from_slice(
            data,
            RegressionSlopeOscillatorParams {
                min_range: Some(min_range),
                max_range: Some(max_range),
                step: Some(step),
                signal_line: Some(signal_line),
            },
        );
        regression_slope_oscillator_into_slice(
            out_value,
            out_signal,
            out_bullish,
            out_bearish,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn regression_slope_oscillator_batch_into(
    in_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_signal_ptr: *mut f64,
    out_bullish_reversal_ptr: *mut f64,
    out_bearish_reversal_ptr: *mut f64,
    len: usize,
    min_range_start: usize,
    min_range_end: usize,
    min_range_step: usize,
    max_range_start: usize,
    max_range_end: usize,
    max_range_step: usize,
    step_start: usize,
    step_end: usize,
    step_step: usize,
    signal_line_start: usize,
    signal_line_end: usize,
    signal_line_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null()
        || out_value_ptr.is_null()
        || out_signal_ptr.is_null()
        || out_bullish_reversal_ptr.is_null()
        || out_bearish_reversal_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to regression_slope_oscillator_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = RegressionSlopeOscillatorBatchRange {
            min_range: (min_range_start, min_range_end, min_range_step),
            max_range: (max_range_start, max_range_end, max_range_step),
            step: (step_start, step_end, step_step),
            signal_line: (signal_line_start, signal_line_end, signal_line_step),
        };
        let combos = regression_slope_oscillator_expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in regression_slope_oscillator_batch_into")
        })?;
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, total);
        let out_signal = std::slice::from_raw_parts_mut(out_signal_ptr, total);
        let out_bullish = std::slice::from_raw_parts_mut(out_bullish_reversal_ptr, total);
        let out_bearish = std::slice::from_raw_parts_mut(out_bearish_reversal_ptr, total);
        regression_slope_oscillator_batch_into_slice(
            out_value,
            out_signal,
            out_bullish,
            out_bearish,
            data,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn regression_slope_oscillator_output_into_js(
    data: &[f64],
    min_range: usize,
    max_range: usize,
    step: usize,
    signal_line: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = regression_slope_oscillator_js(data, min_range, max_range, step, signal_line)?;
    crate::write_wasm_object_f64_outputs("regression_slope_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn regression_slope_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = regression_slope_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "regression_slope_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.11 + (x * 0.17).sin() * 1.9 + (x * 0.03).cos() * 0.7
            })
            .collect()
    }

    fn assert_vec_close(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            let diff = (a - b).abs();
            assert!(diff <= 1e-10, "mismatch at {idx}: {a} vs {b}");
        }
    }

    fn manual_reference(
        data: &[f64],
        min_range: usize,
        max_range: usize,
        step: usize,
        signal_line: usize,
    ) -> RegressionSlopeOscillatorOutput {
        let len = data.len();
        let specs = expand_specs(min_range, max_range, step);
        let mut value = vec![f64::NAN; len];
        let mut signal = vec![f64::NAN; len];
        let mut bullish_reversal = vec![f64::NAN; len];
        let mut bearish_reversal = vec![f64::NAN; len];

        for i in max_range.saturating_sub(1)..len {
            let mut slopes = Vec::with_capacity(specs.len());
            for spec in &specs {
                let start = i + 1 - spec.length;
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_x_sqr = 0.0;
                let mut sum_xy = 0.0;
                let mut valid = true;
                for j in 0..spec.length {
                    let x = (j + 1) as f64;
                    let sample = data[start + j];
                    if !(sample.is_finite() && sample > 0.0) {
                        valid = false;
                        break;
                    }
                    let y = sample.ln();
                    sum_x += x;
                    sum_y += y;
                    sum_x_sqr += x * x;
                    sum_xy += x * y;
                }
                if !valid {
                    slopes.clear();
                    break;
                }
                slopes.push(
                    (spec.length as f64 * sum_xy - sum_x * sum_y)
                        / (spec.length as f64 * sum_x_sqr - sum_x * sum_x),
                );
            }
            if slopes.len() == specs.len() {
                value[i] = slopes.iter().sum::<f64>() / slopes.len() as f64;
            }
        }

        let mut queue = VecDeque::new();
        let mut sum = 0.0;
        for i in 0..len {
            if value[i].is_finite() {
                queue.push_back(value[i]);
                sum += value[i];
                if queue.len() > signal_line {
                    sum -= queue.pop_front().unwrap();
                }
            }
            if queue.len() == signal_line {
                signal[i] = sum / signal_line as f64;
            }
            if value[i].is_finite() && signal[i].is_finite() {
                let prev_value = if i > 0 { value[i - 1] } else { f64::NAN };
                let prev_signal = if i > 0 { signal[i - 1] } else { f64::NAN };
                bullish_reversal[i] = if prev_value.is_finite()
                    && prev_signal.is_finite()
                    && value[i] > signal[i]
                    && prev_value <= prev_signal
                    && value[i] < 0.0
                {
                    1.0
                } else {
                    0.0
                };
                bearish_reversal[i] = if prev_value.is_finite()
                    && prev_signal.is_finite()
                    && value[i] < signal[i]
                    && prev_value >= prev_signal
                    && value[i] > 0.0
                {
                    1.0
                } else {
                    0.0
                };
            }
        }

        RegressionSlopeOscillatorOutput {
            value,
            signal,
            bullish_reversal,
            bearish_reversal,
        }
    }

    #[test]
    fn manual_reference_matches_single() {
        let data = sample_data(192);
        let input = RegressionSlopeOscillatorInput::from_slice(
            &data,
            RegressionSlopeOscillatorParams {
                min_range: Some(10),
                max_range: Some(30),
                step: Some(5),
                signal_line: Some(7),
            },
        );
        let expected = manual_reference(&data, 10, 30, 5, 7);
        let actual = regression_slope_oscillator(&input).unwrap();
        assert_vec_close(&expected.value, &actual.value);
        assert_vec_close(&expected.signal, &actual.signal);
        assert_vec_close(&expected.bullish_reversal, &actual.bullish_reversal);
        assert_vec_close(&expected.bearish_reversal, &actual.bearish_reversal);
    }

    #[test]
    fn stream_matches_batch() {
        let data = sample_data(224);
        let params = RegressionSlopeOscillatorParams {
            min_range: Some(10),
            max_range: Some(40),
            step: Some(5),
            signal_line: Some(6),
        };
        let batch = regression_slope_oscillator(&RegressionSlopeOscillatorInput::from_slice(
            &data,
            params.clone(),
        ))
        .unwrap();
        let mut stream = RegressionSlopeOscillatorStream::try_new(params).unwrap();
        let mut value = Vec::with_capacity(data.len());
        let mut signal = Vec::with_capacity(data.len());
        let mut bullish = Vec::with_capacity(data.len());
        let mut bearish = Vec::with_capacity(data.len());
        for &item in &data {
            let out = stream.update(item);
            value.push(out.0);
            signal.push(out.1);
            bullish.push(out.2);
            bearish.push(out.3);
        }
        assert_vec_close(&value, &batch.value);
        assert_vec_close(&signal, &batch.signal);
        assert_vec_close(&bullish, &batch.bullish_reversal);
        assert_vec_close(&bearish, &batch.bearish_reversal);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let data = sample_data(160);
        let single = regression_slope_oscillator(&RegressionSlopeOscillatorInput::from_slice(
            &data,
            RegressionSlopeOscillatorParams {
                min_range: Some(10),
                max_range: Some(30),
                step: Some(5),
                signal_line: Some(7),
            },
        ))
        .unwrap();
        let batch = regression_slope_oscillator_batch_slice(
            &data,
            &RegressionSlopeOscillatorBatchRange {
                min_range: (10, 15, 5),
                max_range: (30, 30, 0),
                step: (5, 5, 0),
                signal_line: (7, 7, 0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, data.len());
        assert_vec_close(&batch.value[..data.len()], &single.value);
        assert_vec_close(&batch.signal[..data.len()], &single.signal);
        assert_vec_close(
            &batch.bullish_reversal[..data.len()],
            &single.bullish_reversal,
        );
        assert_vec_close(
            &batch.bearish_reversal[..data.len()],
            &single.bearish_reversal,
        );
    }

    #[test]
    fn invalid_range_config_fails() {
        let data = sample_data(96);
        let input = RegressionSlopeOscillatorInput::from_slice(
            &data,
            RegressionSlopeOscillatorParams {
                min_range: Some(20),
                max_range: Some(10),
                step: Some(5),
                signal_line: Some(7),
            },
        );
        let err = regression_slope_oscillator(&input).unwrap_err();
        assert!(err.to_string().contains("Invalid range config"));
    }

    #[test]
    fn cpu_dispatch_matches_direct() {
        let data = sample_data(144);
        let request = IndicatorBatchRequest {
            indicator_id: "regression_slope_oscillator",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &[IndicatorParamSet {
                params: &[
                    ParamKV {
                        key: "min_range",
                        value: ParamValue::Int(10),
                    },
                    ParamKV {
                        key: "max_range",
                        value: ParamValue::Int(30),
                    },
                    ParamKV {
                        key: "step",
                        value: ParamValue::Int(5),
                    },
                    ParamKV {
                        key: "signal_line",
                        value: ParamValue::Int(7),
                    },
                ],
            }],
            kernel: Kernel::Auto,
        };
        let output = compute_cpu_batch(request).unwrap();
        let values = output.values_f64.unwrap();
        let direct = regression_slope_oscillator(&RegressionSlopeOscillatorInput::from_slice(
            &data,
            RegressionSlopeOscillatorParams {
                min_range: Some(10),
                max_range: Some(30),
                step: Some(5),
                signal_line: Some(7),
            },
        ))
        .unwrap();
        assert_vec_close(&values[..data.len()], &direct.value);
    }
}
