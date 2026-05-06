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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::f64::consts::PI;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 100;
const DEFAULT_DELTA: f64 = 0.5;
const DEFAULT_LOOKBACK_MULT: f64 = 1.0;
const DEFAULT_SIGNAL_LENGTH: usize = 9;
const DEFAULT_SOURCE: &str = "hl2";
const MIN_VALID_SAMPLES: usize = 3;
const WARMUP: usize = MIN_VALID_SAMPLES - 1;
const FLOAT_TOL: f64 = 1e-12;

impl<'a> AsRef<[f64]> for NormalizedResonatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NormalizedResonatorData::Slice(slice) => slice,
            NormalizedResonatorData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NormalizedResonatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NormalizedResonatorOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct NormalizedResonatorParams {
    pub period: Option<usize>,
    pub delta: Option<f64>,
    pub lookback_mult: Option<f64>,
    pub signal_length: Option<usize>,
}

impl Default for NormalizedResonatorParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            delta: Some(DEFAULT_DELTA),
            lookback_mult: Some(DEFAULT_LOOKBACK_MULT),
            signal_length: Some(DEFAULT_SIGNAL_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalizedResonatorInput<'a> {
    pub data: NormalizedResonatorData<'a>,
    pub params: NormalizedResonatorParams,
}

impl<'a> NormalizedResonatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: NormalizedResonatorParams,
    ) -> Self {
        Self {
            data: NormalizedResonatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: NormalizedResonatorParams) -> Self {
        Self {
            data: NormalizedResonatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            NormalizedResonatorParams::default(),
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NormalizedResonatorBuilder {
    period: Option<usize>,
    delta: Option<f64>,
    lookback_mult: Option<f64>,
    signal_length: Option<usize>,
    kernel: Kernel,
}

impl NormalizedResonatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn period(mut self, period: usize) -> Self {
        self.period = Some(period);
        self
    }

    #[inline]
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = Some(delta);
        self
    }

    #[inline]
    pub fn lookback_mult(mut self, lookback_mult: f64) -> Self {
        self.lookback_mult = Some(lookback_mult);
        self
    }

    #[inline]
    pub fn signal_length(mut self, signal_length: usize) -> Self {
        self.signal_length = Some(signal_length);
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
    ) -> Result<NormalizedResonatorOutput, NormalizedResonatorError> {
        let input = NormalizedResonatorInput::from_candles(
            candles,
            source,
            NormalizedResonatorParams {
                period: self.period,
                delta: self.delta,
                lookback_mult: self.lookback_mult,
                signal_length: self.signal_length,
            },
        );
        normalized_resonator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<NormalizedResonatorOutput, NormalizedResonatorError> {
        let input = NormalizedResonatorInput::from_slice(
            data,
            NormalizedResonatorParams {
                period: self.period,
                delta: self.delta,
                lookback_mult: self.lookback_mult,
                signal_length: self.signal_length,
            },
        );
        normalized_resonator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<NormalizedResonatorStream, NormalizedResonatorError> {
        NormalizedResonatorStream::try_new(NormalizedResonatorParams {
            period: self.period,
            delta: self.delta,
            lookback_mult: self.lookback_mult,
            signal_length: self.signal_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum NormalizedResonatorError {
    #[error("normalized_resonator: Input data slice is empty.")]
    EmptyInputData,
    #[error("normalized_resonator: All values are NaN.")]
    AllValuesNaN,
    #[error("normalized_resonator: Invalid period: {period}")]
    InvalidPeriod { period: usize },
    #[error("normalized_resonator: Invalid delta: {delta}")]
    InvalidDelta { delta: f64 },
    #[error("normalized_resonator: Invalid lookback_mult: {lookback_mult}")]
    InvalidLookbackMult { lookback_mult: f64 },
    #[error("normalized_resonator: Invalid signal_length: {signal_length}")]
    InvalidSignalLength { signal_length: usize },
    #[error("normalized_resonator: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "normalized_resonator: Output length mismatch: expected = {expected}, oscillator = {oscillator_got}, signal = {signal_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        oscillator_got: usize,
        signal_got: usize,
    },
    #[error("normalized_resonator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("normalized_resonator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    period: usize,
    delta: f64,
    lookback_mult: f64,
    signal_length: usize,
    peak_lookback: usize,
    gain: f64,
    c1: f64,
    c2: f64,
    ema_alpha: f64,
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
fn valid_run_until(data: &[f64], needed: usize) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    for &value in data {
        if value.is_finite() {
            run += 1;
            if run > best {
                best = run;
                if best >= needed {
                    return best;
                }
            }
        } else {
            run = 0;
        }
    }
    best
}

#[inline(always)]
fn resolve_params(
    params: &NormalizedResonatorParams,
) -> Result<ResolvedParams, NormalizedResonatorError> {
    let period = params.period.unwrap_or(DEFAULT_PERIOD);
    if period < 2 {
        return Err(NormalizedResonatorError::InvalidPeriod { period });
    }

    let delta = params.delta.unwrap_or(DEFAULT_DELTA);
    if !delta.is_finite() || delta <= 0.0 || delta > 1.0 {
        return Err(NormalizedResonatorError::InvalidDelta { delta });
    }

    let lookback_mult = params.lookback_mult.unwrap_or(DEFAULT_LOOKBACK_MULT);
    if !lookback_mult.is_finite() || lookback_mult <= 0.0 {
        return Err(NormalizedResonatorError::InvalidLookbackMult { lookback_mult });
    }

    let signal_length = params.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH);
    if signal_length == 0 {
        return Err(NormalizedResonatorError::InvalidSignalLength { signal_length });
    }

    let alpha = (PI * delta / period as f64).tan();
    if !alpha.is_finite() {
        return Err(NormalizedResonatorError::InvalidDelta { delta });
    }
    let beta = (2.0 * PI / period as f64).cos();
    let r = 1.0 / (1.0 + alpha);
    let c1 = 2.0 * r * beta;
    let c2 = -(2.0 * r - 1.0);
    let gain = alpha * r;
    let peak_lookback_raw = period as f64 * lookback_mult;
    if !peak_lookback_raw.is_finite() || peak_lookback_raw > usize::MAX as f64 {
        return Err(NormalizedResonatorError::InvalidLookbackMult { lookback_mult });
    }
    let peak_lookback = peak_lookback_raw.floor().max(1.0) as usize;
    let ema_alpha = 2.0 / (signal_length as f64 + 1.0);

    Ok(ResolvedParams {
        period,
        delta,
        lookback_mult,
        signal_length,
        peak_lookback,
        gain,
        c1,
        c2,
        ema_alpha,
    })
}

#[derive(Clone, Debug)]
struct RollingAbsMax {
    window: usize,
    next_index: usize,
    deque: VecDeque<(usize, f64)>,
}

impl RollingAbsMax {
    #[inline]
    fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            next_index: 0,
            deque: VecDeque::new(),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.next_index = 0;
        self.deque.clear();
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        let index = self.next_index;
        self.next_index = self.next_index.wrapping_add(1);

        while let Some(&(_, back_value)) = self.deque.back() {
            if back_value <= value {
                self.deque.pop_back();
            } else {
                break;
            }
        }
        self.deque.push_back((index, value));

        let min_index = index.saturating_add(1).saturating_sub(self.window);
        while let Some(&(front_index, _)) = self.deque.front() {
            if front_index < min_index {
                self.deque.pop_front();
            } else {
                break;
            }
        }

        self.deque
            .front()
            .map(|&(_, max_value)| max_value)
            .unwrap_or(0.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct RollingAbsMax100 {
    indices: [usize; 101],
    values: [f64; 101],
    head: usize,
    len: usize,
    next_index: usize,
}

impl RollingAbsMax100 {
    #[inline]
    fn new() -> Self {
        Self {
            indices: [0; 101],
            values: [0.0; 101],
            head: 0,
            len: 0,
            next_index: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
        self.next_index = 0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        let index = self.next_index;
        self.next_index = self.next_index.wrapping_add(1);

        while self.len > 0 {
            let back = (self.head + self.len - 1) % 101;
            if self.values[back] <= value {
                self.len -= 1;
            } else {
                break;
            }
        }

        let tail = (self.head + self.len) % 101;
        self.indices[tail] = index;
        self.values[tail] = value;
        self.len += 1;

        let min_index = index.saturating_add(1).saturating_sub(DEFAULT_PERIOD);
        while self.len > 0 && self.indices[self.head] < min_index {
            self.head += 1;
            if self.head == 101 {
                self.head = 0;
            }
            self.len -= 1;
        }

        if self.len > 0 {
            self.values[self.head]
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
pub struct NormalizedResonatorStream {
    params: ResolvedParams,
    src_prev1: f64,
    src_prev2: f64,
    src_count: usize,
    bp_prev1: f64,
    bp_prev2: f64,
    peak_window: RollingAbsMax,
    ema_value: f64,
    ema_seeded: bool,
}

impl NormalizedResonatorStream {
    #[inline]
    pub fn try_new(params: NormalizedResonatorParams) -> Result<Self, NormalizedResonatorError> {
        let params = resolve_params(&params)?;
        Ok(Self::from_resolved(params))
    }

    #[inline]
    fn from_resolved(params: ResolvedParams) -> Self {
        Self {
            src_prev1: 0.0,
            src_prev2: 0.0,
            src_count: 0,
            bp_prev1: 0.0,
            bp_prev2: 0.0,
            peak_window: RollingAbsMax::new(params.peak_lookback),
            ema_value: 0.0,
            ema_seeded: false,
            params,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.src_prev1 = 0.0;
        self.src_prev2 = 0.0;
        self.src_count = 0;
        self.bp_prev1 = 0.0;
        self.bp_prev2 = 0.0;
        self.peak_window.reset();
        self.ema_value = 0.0;
        self.ema_seeded = false;
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        WARMUP
    }

    #[inline]
    fn advance_source_history(&mut self, value: f64) {
        match self.src_count {
            0 => {
                self.src_prev1 = value;
                self.src_count = 1;
            }
            1 => {
                self.src_prev2 = self.src_prev1;
                self.src_prev1 = value;
                self.src_count = 2;
            }
            _ => {
                self.src_prev2 = self.src_prev1;
                self.src_prev1 = value;
            }
        }
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let out = if self.src_count >= 2 {
            let bp = self.params.gain * (value - self.src_prev2)
                + self.params.c1 * self.bp_prev1
                + self.params.c2 * self.bp_prev2;
            let peak = self.peak_window.update(bp.abs());
            let oscillator = if peak > 0.0 { bp / peak } else { 0.0 };
            let signal = if self.ema_seeded {
                self.ema_value += self.params.ema_alpha * (oscillator - self.ema_value);
                self.ema_value
            } else {
                self.ema_value = oscillator;
                self.ema_seeded = true;
                oscillator
            };
            self.bp_prev2 = self.bp_prev1;
            self.bp_prev1 = bp;
            Some((oscillator, signal))
        } else {
            None
        };

        self.advance_source_history(value);
        out
    }
}

#[inline(always)]
fn normalized_resonator_prepare<'a>(
    input: &'a NormalizedResonatorInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, ResolvedParams, Kernel), NormalizedResonatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(NormalizedResonatorError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(NormalizedResonatorError::AllValuesNaN);
    }

    let params = resolve_params(&input.params)?;
    let valid = valid_run_until(data, MIN_VALID_SAMPLES);
    if valid < MIN_VALID_SAMPLES {
        return Err(NormalizedResonatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok((data, first, params, chosen))
}

#[inline(always)]
fn normalized_resonator_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
) {
    if params.period == DEFAULT_PERIOD
        && params.delta == DEFAULT_DELTA
        && params.lookback_mult == DEFAULT_LOOKBACK_MULT
        && params.signal_length == DEFAULT_SIGNAL_LENGTH
        && params.peak_lookback == DEFAULT_PERIOD
    {
        normalized_resonator_default_row(data, params, oscillator_out, signal_out);
        return;
    }

    let mut stream = NormalizedResonatorStream::from_resolved(params);

    for ((oscillator_slot, signal_slot), &value) in oscillator_out
        .iter_mut()
        .zip(signal_out.iter_mut())
        .zip(data.iter())
    {
        if let Some((oscillator, signal)) = stream.update(value) {
            *oscillator_slot = oscillator;
            *signal_slot = signal;
        } else {
            *oscillator_slot = f64::NAN;
            *signal_slot = f64::NAN;
        }
    }
}

#[inline(always)]
fn normalized_resonator_default_row(
    data: &[f64],
    params: ResolvedParams,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
) {
    let mut src_prev1 = 0.0;
    let mut src_prev2 = 0.0;
    let mut src_count = 0usize;
    let mut bp_prev1 = 0.0;
    let mut bp_prev2 = 0.0;
    let mut peak_window = RollingAbsMax100::new();
    let mut ema_value = 0.0;
    let mut ema_seeded = false;

    let mut i = 0usize;
    while i < data.len() {
        let value = data[i];
        if !value.is_finite() {
            src_prev1 = 0.0;
            src_prev2 = 0.0;
            src_count = 0;
            bp_prev1 = 0.0;
            bp_prev2 = 0.0;
            peak_window.reset();
            ema_value = 0.0;
            ema_seeded = false;
            oscillator_out[i] = f64::NAN;
            signal_out[i] = f64::NAN;
            i += 1;
            continue;
        }

        if src_count >= 2 {
            let bp =
                params.gain * (value - src_prev2) + params.c1 * bp_prev1 + params.c2 * bp_prev2;
            let peak = peak_window.update(bp.abs());
            let oscillator = if peak > 0.0 { bp / peak } else { 0.0 };
            let signal = if ema_seeded {
                ema_value += params.ema_alpha * (oscillator - ema_value);
                ema_value
            } else {
                ema_value = oscillator;
                ema_seeded = true;
                oscillator
            };
            bp_prev2 = bp_prev1;
            bp_prev1 = bp;
            oscillator_out[i] = oscillator;
            signal_out[i] = signal;
        } else {
            oscillator_out[i] = f64::NAN;
            signal_out[i] = f64::NAN;
        }

        match src_count {
            0 => {
                src_prev1 = value;
                src_count = 1;
            }
            1 => {
                src_prev2 = src_prev1;
                src_prev1 = value;
                src_count = 2;
            }
            _ => {
                src_prev2 = src_prev1;
                src_prev1 = value;
            }
        }

        i += 1;
    }
}

#[inline]
pub fn normalized_resonator(
    input: &NormalizedResonatorInput,
) -> Result<NormalizedResonatorOutput, NormalizedResonatorError> {
    normalized_resonator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn normalized_resonator_with_kernel(
    input: &NormalizedResonatorInput,
    kernel: Kernel,
) -> Result<NormalizedResonatorOutput, NormalizedResonatorError> {
    let (data, first, params, _chosen) = normalized_resonator_prepare(input, kernel)?;
    let warmup = first.saturating_add(WARMUP).min(data.len());
    let mut oscillator = alloc_with_nan_prefix(data.len(), warmup);
    let mut signal = alloc_with_nan_prefix(data.len(), warmup);
    normalized_resonator_row_from_slice(data, params, &mut oscillator, &mut signal);
    Ok(NormalizedResonatorOutput { oscillator, signal })
}

#[inline]
pub fn normalized_resonator_into_slices(
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
    input: &NormalizedResonatorInput,
    kernel: Kernel,
) -> Result<(), NormalizedResonatorError> {
    let expected = input.as_ref().len();
    if oscillator_out.len() != expected || signal_out.len() != expected {
        return Err(NormalizedResonatorError::OutputLengthMismatch {
            expected,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
        });
    }
    let (data, _first, params, _chosen) = normalized_resonator_prepare(input, kernel)?;
    normalized_resonator_row_from_slice(data, params, oscillator_out, signal_out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn normalized_resonator_into(
    input: &NormalizedResonatorInput,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<(), NormalizedResonatorError> {
    normalized_resonator_into_slices(oscillator_out, signal_out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct NormalizedResonatorBatchRange {
    pub period: (usize, usize, usize),
    pub delta: (f64, f64, f64),
    pub lookback_mult: (f64, f64, f64),
    pub signal_length: (usize, usize, usize),
}

impl Default for NormalizedResonatorBatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
            delta: (DEFAULT_DELTA, DEFAULT_DELTA, 0.0),
            lookback_mult: (DEFAULT_LOOKBACK_MULT, DEFAULT_LOOKBACK_MULT, 0.0),
            signal_length: (DEFAULT_SIGNAL_LENGTH, DEFAULT_SIGNAL_LENGTH, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalizedResonatorBatchOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<NormalizedResonatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NormalizedResonatorBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &NormalizedResonatorParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.period.unwrap_or(DEFAULT_PERIOD) == params.period.unwrap_or(DEFAULT_PERIOD)
                && (combo.delta.unwrap_or(DEFAULT_DELTA) - params.delta.unwrap_or(DEFAULT_DELTA))
                    .abs()
                    < FLOAT_TOL
                && (combo.lookback_mult.unwrap_or(DEFAULT_LOOKBACK_MULT)
                    - params.lookback_mult.unwrap_or(DEFAULT_LOOKBACK_MULT))
                .abs()
                    < FLOAT_TOL
                && combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH)
                    == params.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH)
        })
    }

    #[inline]
    pub fn row_slices(&self, row: usize) -> Option<(&[f64], &[f64])> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some((&self.oscillator[start..end], &self.signal[start..end]))
    }
}

#[derive(Clone, Debug, Default)]
pub struct NormalizedResonatorBatchBuilder {
    range: NormalizedResonatorBatchRange,
    kernel: Kernel,
}

impl NormalizedResonatorBatchBuilder {
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
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline]
    pub fn delta_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.delta = (start, end, step);
        self
    }

    #[inline]
    pub fn lookback_mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.lookback_mult = (start, end, step);
        self
    }

    #[inline]
    pub fn signal_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_length = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<NormalizedResonatorBatchOutput, NormalizedResonatorError> {
        normalized_resonator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<NormalizedResonatorBatchOutput, NormalizedResonatorError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, NormalizedResonatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            let next = x.saturating_add(step);
            if next == x {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(NormalizedResonatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_axis_f64(start: f64, end: f64, step: f64) -> Result<Vec<f64>, NormalizedResonatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(NormalizedResonatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(NormalizedResonatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(NormalizedResonatorError::InvalidRange {
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
        return Err(NormalizedResonatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(values)
}

#[inline(always)]
fn expand_grid_normalized_resonator(
    sweep: &NormalizedResonatorBatchRange,
) -> Result<Vec<NormalizedResonatorParams>, NormalizedResonatorError> {
    let periods = expand_axis_usize(sweep.period)?;
    let deltas = expand_axis_f64(sweep.delta.0, sweep.delta.1, sweep.delta.2)?;
    let lookback_mults = expand_axis_f64(
        sweep.lookback_mult.0,
        sweep.lookback_mult.1,
        sweep.lookback_mult.2,
    )?;
    let signal_lengths = expand_axis_usize(sweep.signal_length)?;

    let mut combos = Vec::with_capacity(
        periods.len() * deltas.len() * lookback_mults.len() * signal_lengths.len(),
    );
    for period in periods {
        for &delta in &deltas {
            for &lookback_mult in &lookback_mults {
                for signal_length in signal_lengths.iter().copied() {
                    let combo = NormalizedResonatorParams {
                        period: Some(period),
                        delta: Some(delta),
                        lookback_mult: Some(lookback_mult),
                        signal_length: Some(signal_length),
                    };
                    let _ = resolve_params(&combo)?;
                    combos.push(combo);
                }
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn normalized_resonator_batch_with_kernel(
    data: &[f64],
    sweep: &NormalizedResonatorBatchRange,
    kernel: Kernel,
) -> Result<NormalizedResonatorBatchOutput, NormalizedResonatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(NormalizedResonatorError::InvalidKernelForBatch(other)),
    };
    normalized_resonator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn normalized_resonator_batch_slice(
    data: &[f64],
    sweep: &NormalizedResonatorBatchRange,
    kernel: Kernel,
) -> Result<NormalizedResonatorBatchOutput, NormalizedResonatorError> {
    normalized_resonator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn normalized_resonator_batch_par_slice(
    data: &[f64],
    sweep: &NormalizedResonatorBatchRange,
    kernel: Kernel,
) -> Result<NormalizedResonatorBatchOutput, NormalizedResonatorError> {
    normalized_resonator_batch_inner(data, sweep, kernel, true)
}

#[inline]
pub fn normalized_resonator_batch_inner(
    data: &[f64],
    sweep: &NormalizedResonatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<NormalizedResonatorBatchOutput, NormalizedResonatorError> {
    let combos = expand_grid_normalized_resonator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(NormalizedResonatorError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(NormalizedResonatorError::AllValuesNaN);
    }

    let valid = max_consecutive_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(NormalizedResonatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }

    let mut oscillator_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(
        &mut oscillator_mu,
        cols,
        &vec![first.saturating_add(WARMUP).min(cols); rows],
    );
    init_matrix_prefixes(
        &mut signal_mu,
        cols,
        &vec![first.saturating_add(WARMUP).min(cols); rows],
    );

    let mut oscillator_guard = ManuallyDrop::new(oscillator_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let oscillator_out = unsafe {
        std::slice::from_raw_parts_mut(
            oscillator_guard.as_mut_ptr() as *mut f64,
            oscillator_guard.len(),
        )
    };
    let signal_out = unsafe {
        std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    let combos = normalized_resonator_batch_inner_into(
        data,
        sweep,
        _kernel,
        parallel,
        oscillator_out,
        signal_out,
    )?;

    let oscillator = unsafe {
        Vec::from_raw_parts(
            oscillator_guard.as_mut_ptr() as *mut f64,
            oscillator_guard.len(),
            oscillator_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    Ok(NormalizedResonatorBatchOutput {
        oscillator,
        signal,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn normalized_resonator_batch_inner_into(
    data: &[f64],
    sweep: &NormalizedResonatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<NormalizedResonatorParams>, NormalizedResonatorError> {
    let combos = expand_grid_normalized_resonator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(NormalizedResonatorError::EmptyInputData);
    }

    let total = rows
        .checked_mul(cols)
        .ok_or(NormalizedResonatorError::OutputLengthMismatch {
            expected: usize::MAX,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
        })?;
    if oscillator_out.len() != total || signal_out.len() != total {
        return Err(NormalizedResonatorError::OutputLengthMismatch {
            expected: total,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
        });
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(NormalizedResonatorError::AllValuesNaN);
    }

    let valid = max_consecutive_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(NormalizedResonatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        oscillator_out
            .par_chunks_mut(cols)
            .zip(signal_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (oscillator_row, signal_row))| {
                let params = resolve_params(&combos[row]).unwrap();
                normalized_resonator_row_from_slice(data, params, oscillator_row, signal_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, (oscillator_row, signal_row)) in oscillator_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            normalized_resonator_row_from_slice(data, params, oscillator_row, signal_row);
        }
    } else {
        for (row, (oscillator_row, signal_row)) in oscillator_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            normalized_resonator_row_from_slice(data, params, oscillator_row, signal_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "normalized_resonator")]
#[pyo3(signature = (
    data,
    period=DEFAULT_PERIOD,
    delta=DEFAULT_DELTA,
    lookback_mult=DEFAULT_LOOKBACK_MULT,
    signal_length=DEFAULT_SIGNAL_LENGTH,
    kernel=None
))]
pub fn normalized_resonator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    delta: f64,
    lookback_mult: f64,
    signal_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = NormalizedResonatorInput::from_slice(
        data,
        NormalizedResonatorParams {
            period: Some(period),
            delta: Some(delta),
            lookback_mult: Some(lookback_mult),
            signal_length: Some(signal_length),
        },
    );
    let output = py
        .allow_threads(|| normalized_resonator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.oscillator.into_pyarray(py),
        output.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "NormalizedResonatorStream")]
pub struct NormalizedResonatorStreamPy {
    stream: NormalizedResonatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NormalizedResonatorStreamPy {
    #[new]
    #[pyo3(signature = (
        period=DEFAULT_PERIOD,
        delta=DEFAULT_DELTA,
        lookback_mult=DEFAULT_LOOKBACK_MULT,
        signal_length=DEFAULT_SIGNAL_LENGTH
    ))]
    fn new(period: usize, delta: f64, lookback_mult: f64, signal_length: usize) -> PyResult<Self> {
        let stream = NormalizedResonatorStream::try_new(NormalizedResonatorParams {
            period: Some(period),
            delta: Some(delta),
            lookback_mult: Some(lookback_mult),
            signal_length: Some(signal_length),
        })
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
#[pyfunction(name = "normalized_resonator_batch")]
#[pyo3(signature = (
    data,
    period_range=(DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
    delta_range=(DEFAULT_DELTA, DEFAULT_DELTA, 0.0),
    lookback_mult_range=(DEFAULT_LOOKBACK_MULT, DEFAULT_LOOKBACK_MULT, 0.0),
    signal_length_range=(DEFAULT_SIGNAL_LENGTH, DEFAULT_SIGNAL_LENGTH, 0),
    kernel=None
))]
pub fn normalized_resonator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    delta_range: (f64, f64, f64),
    lookback_mult_range: (f64, f64, f64),
    signal_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = NormalizedResonatorBatchRange {
        period: period_range,
        delta: delta_range,
        lookback_mult: lookback_mult_range,
        signal_length: signal_length_range,
    };
    let combos = expand_grid_normalized_resonator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let oscillator_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let oscillator_slice = unsafe { oscillator_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            normalized_resonator_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                true,
                oscillator_slice,
                signal_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("oscillator", oscillator_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "deltas",
        combos
            .iter()
            .map(|combo| combo.delta.unwrap_or(DEFAULT_DELTA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lookback_multipliers",
        combos
            .iter()
            .map(|combo| combo.lookback_mult.unwrap_or(DEFAULT_LOOKBACK_MULT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_lengths",
        combos
            .iter()
            .map(|combo| combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_normalized_resonator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalized_resonator_py, module)?)?;
    module.add_function(wrap_pyfunction!(normalized_resonator_batch_py, module)?)?;
    module.add_class::<NormalizedResonatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NormalizedResonatorJsOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "normalized_resonator_js")]
pub fn normalized_resonator_js(
    data: &[f64],
    period: usize,
    delta: f64,
    lookback_mult: f64,
    signal_length: usize,
) -> Result<JsValue, JsValue> {
    let input = NormalizedResonatorInput::from_slice(
        data,
        NormalizedResonatorParams {
            period: Some(period),
            delta: Some(delta),
            lookback_mult: Some(lookback_mult),
            signal_length: Some(signal_length),
        },
    );
    let output = normalized_resonator(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&NormalizedResonatorJsOutput {
        oscillator: output.oscillator,
        signal: output.signal,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_resonator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_resonator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_resonator_into(
    in_ptr: *const f64,
    oscillator_out_ptr: *mut f64,
    signal_out_ptr: *mut f64,
    len: usize,
    period: usize,
    delta: f64,
    lookback_mult: f64,
    signal_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || oscillator_out_ptr.is_null() || signal_out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = NormalizedResonatorInput::from_slice(
            data,
            NormalizedResonatorParams {
                period: Some(period),
                delta: Some(delta),
                lookback_mult: Some(lookback_mult),
                signal_length: Some(signal_length),
            },
        );
        let oscillator_out = std::slice::from_raw_parts_mut(oscillator_out_ptr, len);
        let signal_out = std::slice::from_raw_parts_mut(signal_out_ptr, len);
        normalized_resonator_into_slices(oscillator_out, signal_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NormalizedResonatorBatchJsConfig {
    pub period_range: Option<(usize, usize, usize)>,
    pub delta_range: Option<(f64, f64, f64)>,
    pub lookback_mult_range: Option<(f64, f64, f64)>,
    pub signal_length_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NormalizedResonatorBatchJsOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<NormalizedResonatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "normalized_resonator_batch_js")]
pub fn normalized_resonator_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: NormalizedResonatorBatchJsConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let sweep = NormalizedResonatorBatchRange {
        period: config
            .period_range
            .unwrap_or((DEFAULT_PERIOD, DEFAULT_PERIOD, 0)),
        delta: config
            .delta_range
            .unwrap_or((DEFAULT_DELTA, DEFAULT_DELTA, 0.0)),
        lookback_mult: config.lookback_mult_range.unwrap_or((
            DEFAULT_LOOKBACK_MULT,
            DEFAULT_LOOKBACK_MULT,
            0.0,
        )),
        signal_length: config.signal_length_range.unwrap_or((
            DEFAULT_SIGNAL_LENGTH,
            DEFAULT_SIGNAL_LENGTH,
            0,
        )),
    };
    let output = normalized_resonator_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&NormalizedResonatorBatchJsOutput {
        oscillator: output.oscillator,
        signal: output.signal,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_resonator_batch_into(
    in_ptr: *const f64,
    oscillator_out_ptr: *mut f64,
    signal_out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    delta_start: f64,
    delta_end: f64,
    delta_step: f64,
    lookback_mult_start: f64,
    lookback_mult_end: f64,
    lookback_mult_step: f64,
    signal_length_start: usize,
    signal_length_end: usize,
    signal_length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || oscillator_out_ptr.is_null() || signal_out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = NormalizedResonatorBatchRange {
        period: (period_start, period_end, period_step),
        delta: (delta_start, delta_end, delta_step),
        lookback_mult: (lookback_mult_start, lookback_mult_end, lookback_mult_step),
        signal_length: (signal_length_start, signal_length_end, signal_length_step),
    };

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let combos = expand_grid_normalized_resonator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let oscillator_out = std::slice::from_raw_parts_mut(oscillator_out_ptr, total);
        let signal_out = std::slice::from_raw_parts_mut(signal_out_ptr, total);
        let rows = normalized_resonator_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            oscillator_out,
            signal_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_resonator_output_into_js(
    data: &[f64],
    period: usize,
    delta: f64,
    lookback_mult: f64,
    signal_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = normalized_resonator_js(data, period, delta, lookback_mult, signal_length)?;
    crate::write_wasm_object_f64_outputs("normalized_resonator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_resonator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = normalized_resonator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "normalized_resonator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::Candles;

    fn sample_source(length: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(length);
        for i in 0..length {
            let x = i as f64;
            out.push(100.0 + x * 0.03 + (x * 0.09).sin() * 2.1 + (x * 0.02).cos() * 0.7);
        }
        out
    }

    fn sample_candles(length: usize) -> Candles {
        let open: Vec<f64> = (0..length)
            .map(|i| 100.0 + i as f64 * 0.03 + (i as f64 * 0.07).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.11).cos() * 0.8)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.6 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.6 - (i as f64 * 0.03).cos().abs() * 0.2)
            .collect();
        Candles::new(
            (0..length as i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; length],
        )
    }

    fn assert_series_eq(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&lhs, &rhs) in left.iter().zip(right.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= tol, "lhs={lhs}, rhs={rhs}");
        }
    }

    #[test]
    fn normalized_resonator_output_contract() {
        let data = sample_source(256);
        let out = normalized_resonator(&NormalizedResonatorInput::from_slice(
            &data,
            NormalizedResonatorParams::default(),
        ))
        .unwrap();

        assert_eq!(out.oscillator.len(), data.len());
        assert_eq!(out.signal.len(), data.len());
        assert_eq!(
            out.oscillator.iter().position(|v| v.is_finite()),
            Some(WARMUP)
        );
        assert_eq!(out.signal.iter().position(|v| v.is_finite()), Some(WARMUP));
        assert!(out.oscillator.last().copied().unwrap().is_finite());
        assert!(out.signal.last().copied().unwrap().is_finite());
    }

    #[test]
    fn normalized_resonator_rejects_invalid_parameters() {
        let data = sample_source(32);

        let err = normalized_resonator(&NormalizedResonatorInput::from_slice(
            &data,
            NormalizedResonatorParams {
                period: Some(1),
                ..NormalizedResonatorParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            NormalizedResonatorError::InvalidPeriod { .. }
        ));

        let err = normalized_resonator(&NormalizedResonatorInput::from_slice(
            &data,
            NormalizedResonatorParams {
                delta: Some(0.0),
                ..NormalizedResonatorParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(err, NormalizedResonatorError::InvalidDelta { .. }));

        let err = normalized_resonator(&NormalizedResonatorInput::from_slice(
            &data,
            NormalizedResonatorParams {
                signal_length: Some(0),
                ..NormalizedResonatorParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            NormalizedResonatorError::InvalidSignalLength { .. }
        ));
    }

    #[test]
    fn normalized_resonator_builder_supports_candles() {
        let candles = sample_candles(180);
        let out = NormalizedResonatorBuilder::new()
            .apply(&candles, "hl2")
            .unwrap();
        assert_eq!(out.oscillator.len(), candles.close.len());
        assert_eq!(out.signal.len(), candles.close.len());
        assert!(out.oscillator.last().copied().unwrap().is_finite());
        assert!(out.signal.last().copied().unwrap().is_finite());
    }

    #[test]
    fn normalized_resonator_stream_matches_batch_with_reset() {
        let mut data = sample_source(220);
        data[110] = f64::NAN;

        let input = NormalizedResonatorInput::from_slice(
            &data,
            NormalizedResonatorParams {
                period: Some(48),
                delta: Some(0.4),
                lookback_mult: Some(1.2),
                signal_length: Some(7),
            },
        );
        let batch = normalized_resonator(&input).unwrap();
        let mut stream = NormalizedResonatorStream::try_new(input.params.clone()).unwrap();

        let mut oscillator = Vec::with_capacity(data.len());
        let mut signal = Vec::with_capacity(data.len());
        for &value in &data {
            if let Some((osc, sig)) = stream.update(value) {
                oscillator.push(osc);
                signal.push(sig);
            } else {
                oscillator.push(f64::NAN);
                signal.push(f64::NAN);
            }
        }

        assert_series_eq(&oscillator, &batch.oscillator, 1e-12);
        assert_series_eq(&signal, &batch.signal, 1e-12);
    }

    #[test]
    fn normalized_resonator_into_matches_api() {
        let data = sample_source(128);
        let input =
            NormalizedResonatorInput::from_slice(&data, NormalizedResonatorParams::default());
        let direct = normalized_resonator(&input).unwrap();
        let mut oscillator = vec![f64::NAN; data.len()];
        let mut signal = vec![f64::NAN; data.len()];
        normalized_resonator_into(&input, &mut oscillator, &mut signal).unwrap();
        assert_series_eq(&oscillator, &direct.oscillator, 1e-12);
        assert_series_eq(&signal, &direct.signal, 1e-12);
    }

    #[test]
    fn normalized_resonator_batch_single_param_matches_single() {
        let data = sample_source(192);
        let batch = normalized_resonator_batch_with_kernel(
            &data,
            &NormalizedResonatorBatchRange {
                period: (48, 48, 0),
                delta: (0.4, 0.4, 0.0),
                lookback_mult: (1.2, 1.2, 0.0),
                signal_length: (7, 7, 0),
            },
            Kernel::Auto,
        )
        .unwrap();
        let direct = normalized_resonator(&NormalizedResonatorInput::from_slice(
            &data,
            NormalizedResonatorParams {
                period: Some(48),
                delta: Some(0.4),
                lookback_mult: Some(1.2),
                signal_length: Some(7),
            },
        ))
        .unwrap();

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        let (oscillator, signal) = batch.row_slices(0).unwrap();
        assert_series_eq(oscillator, &direct.oscillator, 1e-12);
        assert_series_eq(signal, &direct.signal, 1e-12);
    }

    #[test]
    fn normalized_resonator_batch_metadata() {
        let data = sample_source(160);
        let batch = normalized_resonator_batch_with_kernel(
            &data,
            &NormalizedResonatorBatchRange {
                period: (40, 44, 4),
                delta: (0.3, 0.5, 0.2),
                lookback_mult: (1.0, 1.2, 0.2),
                signal_length: (5, 6, 1),
            },
            Kernel::Auto,
        )
        .unwrap();

        assert_eq!(batch.rows, 16);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.oscillator.len(), 16 * data.len());
        assert_eq!(batch.signal.len(), 16 * data.len());
    }
}
