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
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_SHORT_CYCLE_LENGTH: usize = 10;
const DEFAULT_MEDIUM_CYCLE_LENGTH: usize = 30;
const DEFAULT_SHORT_MULTIPLIER: f64 = 1.0;
const DEFAULT_MEDIUM_MULTIPLIER: f64 = 3.0;

#[derive(Debug, Clone)]
pub enum CycleChannelOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        source: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct CycleChannelOscillatorOutput {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleChannelOscillatorOutputField {
    Fast,
    Slow,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct CycleChannelOscillatorParams {
    pub short_cycle_length: Option<usize>,
    pub medium_cycle_length: Option<usize>,
    pub short_multiplier: Option<f64>,
    pub medium_multiplier: Option<f64>,
}

impl Default for CycleChannelOscillatorParams {
    fn default() -> Self {
        Self {
            short_cycle_length: Some(DEFAULT_SHORT_CYCLE_LENGTH),
            medium_cycle_length: Some(DEFAULT_MEDIUM_CYCLE_LENGTH),
            short_multiplier: Some(DEFAULT_SHORT_MULTIPLIER),
            medium_multiplier: Some(DEFAULT_MEDIUM_MULTIPLIER),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CycleChannelOscillatorInput<'a> {
    pub data: CycleChannelOscillatorData<'a>,
    pub params: CycleChannelOscillatorParams,
}

impl<'a> CycleChannelOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: CycleChannelOscillatorParams,
    ) -> Self {
        Self {
            data: CycleChannelOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: CycleChannelOscillatorParams,
    ) -> Self {
        Self {
            data: CycleChannelOscillatorData::Slices {
                source,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            CycleChannelOscillatorParams::default(),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CycleChannelOscillatorBuilder {
    source: Option<&'static str>,
    short_cycle_length: Option<usize>,
    medium_cycle_length: Option<usize>,
    short_multiplier: Option<f64>,
    medium_multiplier: Option<f64>,
    kernel: Kernel,
}

impl Default for CycleChannelOscillatorBuilder {
    fn default() -> Self {
        Self {
            source: None,
            short_cycle_length: None,
            medium_cycle_length: None,
            short_multiplier: None,
            medium_multiplier: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CycleChannelOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn short_cycle_length(mut self, value: usize) -> Self {
        self.short_cycle_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn medium_cycle_length(mut self, value: usize) -> Self {
        self.medium_cycle_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn short_multiplier(mut self, value: f64) -> Self {
        self.short_multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn medium_multiplier(mut self, value: f64) -> Self {
        self.medium_multiplier = Some(value);
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
    ) -> Result<CycleChannelOscillatorOutput, CycleChannelOscillatorError> {
        let input = CycleChannelOscillatorInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            CycleChannelOscillatorParams {
                short_cycle_length: self.short_cycle_length,
                medium_cycle_length: self.medium_cycle_length,
                short_multiplier: self.short_multiplier,
                medium_multiplier: self.medium_multiplier,
            },
        );
        cycle_channel_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<CycleChannelOscillatorOutput, CycleChannelOscillatorError> {
        let input = CycleChannelOscillatorInput::from_slices(
            source,
            high,
            low,
            close,
            CycleChannelOscillatorParams {
                short_cycle_length: self.short_cycle_length,
                medium_cycle_length: self.medium_cycle_length,
                short_multiplier: self.short_multiplier,
                medium_multiplier: self.medium_multiplier,
            },
        );
        cycle_channel_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CycleChannelOscillatorStream, CycleChannelOscillatorError> {
        CycleChannelOscillatorStream::try_new(CycleChannelOscillatorParams {
            short_cycle_length: self.short_cycle_length,
            medium_cycle_length: self.medium_cycle_length,
            short_multiplier: self.short_multiplier,
            medium_multiplier: self.medium_multiplier,
        })
    }
}

#[derive(Debug, Error)]
pub enum CycleChannelOscillatorError {
    #[error("cycle_channel_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("cycle_channel_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("cycle_channel_oscillator: Inconsistent slice lengths: source={source_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        source_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("cycle_channel_oscillator: Invalid cycle length `{name}`: {value}")]
    InvalidCycleLength { name: &'static str, value: usize },
    #[error("cycle_channel_oscillator: Invalid multiplier `{name}`: {value}")]
    InvalidMultiplier { name: &'static str, value: f64 },
    #[error("cycle_channel_oscillator: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("cycle_channel_oscillator: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("cycle_channel_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("cycle_channel_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    short_cycle_length: usize,
    medium_cycle_length: usize,
    short_multiplier: f64,
    medium_multiplier: f64,
    short_period: usize,
    medium_period: usize,
    short_delay: usize,
    medium_delay: usize,
}

#[inline(always)]
fn extract_slices<'a>(
    input: &'a CycleChannelOscillatorInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), CycleChannelOscillatorError> {
    let (source, high, low, close) = match &input.data {
        CycleChannelOscillatorData::Candles { candles, source } => (
            source_type(candles, source),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        CycleChannelOscillatorData::Slices {
            source,
            high,
            low,
            close,
        } => (*source, *high, *low, *close),
    };

    if source.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(CycleChannelOscillatorError::EmptyInputData);
    }
    if source.len() != high.len() || source.len() != low.len() || source.len() != close.len() {
        return Err(CycleChannelOscillatorError::InconsistentSliceLengths {
            source_len: source.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    Ok((source, high, low, close))
}

#[inline(always)]
fn first_valid_quad(source: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| {
        source[i].is_finite() && high[i].is_finite() && low[i].is_finite() && close[i].is_finite()
    })
}

#[inline(always)]
fn resolve_params(
    params: &CycleChannelOscillatorParams,
) -> Result<ResolvedParams, CycleChannelOscillatorError> {
    let short_cycle_length = params
        .short_cycle_length
        .unwrap_or(DEFAULT_SHORT_CYCLE_LENGTH);
    let medium_cycle_length = params
        .medium_cycle_length
        .unwrap_or(DEFAULT_MEDIUM_CYCLE_LENGTH);
    let short_multiplier = params.short_multiplier.unwrap_or(DEFAULT_SHORT_MULTIPLIER);
    let medium_multiplier = params
        .medium_multiplier
        .unwrap_or(DEFAULT_MEDIUM_MULTIPLIER);

    if short_cycle_length < 2 {
        return Err(CycleChannelOscillatorError::InvalidCycleLength {
            name: "short_cycle_length",
            value: short_cycle_length,
        });
    }
    if medium_cycle_length < 2 {
        return Err(CycleChannelOscillatorError::InvalidCycleLength {
            name: "medium_cycle_length",
            value: medium_cycle_length,
        });
    }
    if !short_multiplier.is_finite() || short_multiplier < 0.0 {
        return Err(CycleChannelOscillatorError::InvalidMultiplier {
            name: "short_multiplier",
            value: short_multiplier,
        });
    }
    if !medium_multiplier.is_finite() || medium_multiplier < 0.0 {
        return Err(CycleChannelOscillatorError::InvalidMultiplier {
            name: "medium_multiplier",
            value: medium_multiplier,
        });
    }

    let short_period = short_cycle_length / 2;
    let medium_period = medium_cycle_length / 2;
    let short_delay = short_period / 2;
    let medium_delay = medium_period / 2;

    Ok(ResolvedParams {
        short_cycle_length,
        medium_cycle_length,
        short_multiplier,
        medium_multiplier,
        short_period,
        medium_period,
        short_delay,
        medium_delay,
    })
}

#[derive(Debug, Clone)]
struct RmaState {
    length: usize,
    count: usize,
    sum: f64,
    value: f64,
}

impl RmaState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            length,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
        }
    }

    #[inline(always)]
    fn update(&mut self, input: f64) -> f64 {
        if self.count < self.length {
            self.sum += input;
            self.count += 1;
            if self.count == self.length {
                self.value = self.sum / self.length as f64;
            }
        } else {
            self.value = self.value + (input - self.value) / self.length as f64;
            self.count += 1;
        }
        self.value
    }
}

#[derive(Debug, Clone)]
struct AtrState {
    rma: RmaState,
    prev_close: Option<f64>,
}

impl AtrState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            rma: RmaState::new(length),
            prev_close: None,
        }
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = match self.prev_close {
            Some(prev_close) => (high - low)
                .max((high - prev_close).abs())
                .max((low - prev_close).abs()),
            None => high - low,
        };
        self.prev_close = Some(close);
        self.rma.update(tr)
    }
}

#[inline(always)]
fn delayed_or_source(history: &VecDeque<f64>, delay: usize, source: f64) -> f64 {
    if history.len() > delay {
        let idx = history.len() - 1 - delay;
        match history.get(idx).copied() {
            Some(value) if value.is_finite() => value,
            _ => source,
        }
    } else {
        source
    }
}

#[derive(Debug, Clone)]
struct CycleChannelOscillatorCore {
    short_rma: RmaState,
    medium_rma: RmaState,
    medium_atr: AtrState,
    short_delay: usize,
    medium_delay: usize,
    medium_multiplier: f64,
    short_history: VecDeque<f64>,
    medium_history: VecDeque<f64>,
}

impl CycleChannelOscillatorCore {
    #[inline(always)]
    fn new(resolved: ResolvedParams) -> Self {
        let _ = resolved.short_cycle_length;
        let _ = resolved.medium_cycle_length;
        let _ = resolved.short_multiplier;
        Self {
            short_rma: RmaState::new(resolved.short_period),
            medium_rma: RmaState::new(resolved.medium_period),
            medium_atr: AtrState::new(resolved.medium_period),
            short_delay: resolved.short_delay,
            medium_delay: resolved.medium_delay,
            medium_multiplier: resolved.medium_multiplier,
            short_history: VecDeque::with_capacity(resolved.short_delay + 2),
            medium_history: VecDeque::with_capacity(resolved.medium_delay + 2),
        }
    }

    #[inline(always)]
    fn update(&mut self, source: f64, high: f64, low: f64, close: f64) -> (f64, f64) {
        if !(source.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()) {
            return (f64::NAN, f64::NAN);
        }

        let short_ma = self.short_rma.update(source);
        let medium_ma = self.medium_rma.update(source);
        let medium_atr = self.medium_atr.update(high, low, close);

        self.short_history.push_back(short_ma);
        if self.short_history.len() > self.short_delay + 1 {
            self.short_history.pop_front();
        }
        self.medium_history.push_back(medium_ma);
        if self.medium_history.len() > self.medium_delay + 1 {
            self.medium_history.pop_front();
        }

        let short_center = delayed_or_source(&self.short_history, self.short_delay, source);
        let medium_center = delayed_or_source(&self.medium_history, self.medium_delay, source);
        let offset = self.medium_multiplier * medium_atr;
        let denom = 2.0 * offset;
        if !denom.is_finite() || denom == 0.0 {
            return (f64::NAN, f64::NAN);
        }

        let medium_bottom = medium_center - offset;
        (
            (source - medium_bottom) / denom,
            (short_center - medium_bottom) / denom,
        )
    }
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a CycleChannelOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        ResolvedParams,
        usize,
        Kernel,
    ),
    CycleChannelOscillatorError,
> {
    let (source, high, low, close) = extract_slices(input)?;
    let resolved = resolve_params(&input.params)?;
    let first = first_valid_quad(source, high, low, close)
        .ok_or(CycleChannelOscillatorError::AllValuesNaN)?;
    let valid = source.len().saturating_sub(first);
    if valid < resolved.medium_period {
        return Err(CycleChannelOscillatorError::NotEnoughValidData {
            needed: resolved.medium_period,
            valid,
        });
    }
    Ok((
        source,
        high,
        low,
        close,
        resolved,
        first,
        kernel.to_non_batch(),
    ))
}

#[inline(always)]
fn compute_cycle_channel_oscillator_into(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    resolved: ResolvedParams,
    out_fast: &mut [f64],
    out_slow: &mut [f64],
) -> Result<(), CycleChannelOscillatorError> {
    let n = source.len();
    if out_fast.len() != n || out_slow.len() != n {
        return Err(CycleChannelOscillatorError::OutputLengthMismatch {
            expected: n,
            got: out_fast.len().max(out_slow.len()),
        });
    }
    let mut core = CycleChannelOscillatorCore::new(resolved);
    for i in 0..n {
        let (fast, slow) = core.update(source[i], high[i], low[i], close[i]);
        out_fast[i] = fast;
        out_slow[i] = slow;
    }
    Ok(())
}

#[inline]
pub fn cycle_channel_oscillator(
    input: &CycleChannelOscillatorInput,
) -> Result<CycleChannelOscillatorOutput, CycleChannelOscillatorError> {
    cycle_channel_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn cycle_channel_oscillator_with_kernel(
    input: &CycleChannelOscillatorInput,
    kernel: Kernel,
) -> Result<CycleChannelOscillatorOutput, CycleChannelOscillatorError> {
    let (source, high, low, close, resolved, first, chosen) = validate_input(input, kernel)?;
    let _ = chosen;
    let warm = first + resolved.medium_period - 1;
    let mut fast = alloc_with_nan_prefix(source.len(), warm.min(source.len()));
    let mut slow = alloc_with_nan_prefix(source.len(), warm.min(source.len()));
    compute_cycle_channel_oscillator_into(
        source, high, low, close, resolved, &mut fast, &mut slow,
    )?;
    Ok(CycleChannelOscillatorOutput { fast, slow })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn cycle_channel_oscillator_into(
    out_fast: &mut [f64],
    out_slow: &mut [f64],
    input: &CycleChannelOscillatorInput,
    kernel: Kernel,
) -> Result<(), CycleChannelOscillatorError> {
    cycle_channel_oscillator_into_slice(out_fast, out_slow, input, kernel)
}

#[inline]
pub fn cycle_channel_oscillator_into_slice(
    out_fast: &mut [f64],
    out_slow: &mut [f64],
    input: &CycleChannelOscillatorInput,
    kernel: Kernel,
) -> Result<(), CycleChannelOscillatorError> {
    let (source, high, low, close, resolved, _first, chosen) = validate_input(input, kernel)?;
    let _ = chosen;
    out_fast.fill(f64::NAN);
    out_slow.fill(f64::NAN);
    compute_cycle_channel_oscillator_into(source, high, low, close, resolved, out_fast, out_slow)
}

#[inline]
pub fn cycle_channel_oscillator_output_into_slice(
    dst: &mut [f64],
    input: &CycleChannelOscillatorInput,
    kernel: Kernel,
    field: CycleChannelOscillatorOutputField,
) -> Result<(), CycleChannelOscillatorError> {
    let (source, high, low, close, resolved, _first, chosen) = validate_input(input, kernel)?;
    let _ = chosen;
    if dst.len() != source.len() {
        return Err(CycleChannelOscillatorError::OutputLengthMismatch {
            expected: source.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);
    let mut core = CycleChannelOscillatorCore::new(resolved);
    for i in 0..source.len() {
        let (fast, slow) = core.update(source[i], high[i], low[i], close[i]);
        dst[i] = match field {
            CycleChannelOscillatorOutputField::Fast => fast,
            CycleChannelOscillatorOutputField::Slow => slow,
        };
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct CycleChannelOscillatorStream {
    core: CycleChannelOscillatorCore,
}

impl CycleChannelOscillatorStream {
    pub fn try_new(
        params: CycleChannelOscillatorParams,
    ) -> Result<Self, CycleChannelOscillatorError> {
        Ok(Self {
            core: CycleChannelOscillatorCore::new(resolve_params(&params)?),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, source: f64, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.core.update(source, high, low, close)
    }
}

#[derive(Debug, Clone)]
pub struct CycleChannelOscillatorBatchRange {
    pub short_cycle_length: (usize, usize, usize),
    pub medium_cycle_length: (usize, usize, usize),
    pub short_multiplier: (f64, f64, f64),
    pub medium_multiplier: (f64, f64, f64),
}

impl Default for CycleChannelOscillatorBatchRange {
    fn default() -> Self {
        Self {
            short_cycle_length: (DEFAULT_SHORT_CYCLE_LENGTH, DEFAULT_SHORT_CYCLE_LENGTH, 0),
            medium_cycle_length: (DEFAULT_MEDIUM_CYCLE_LENGTH, DEFAULT_MEDIUM_CYCLE_LENGTH, 0),
            short_multiplier: (DEFAULT_SHORT_MULTIPLIER, DEFAULT_SHORT_MULTIPLIER, 0.0),
            medium_multiplier: (DEFAULT_MEDIUM_MULTIPLIER, DEFAULT_MEDIUM_MULTIPLIER, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CycleChannelOscillatorBatchOutput {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
    pub combos: Vec<CycleChannelOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct CycleChannelOscillatorBatchBuilder {
    source: Option<&'static str>,
    range: CycleChannelOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for CycleChannelOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            range: CycleChannelOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl CycleChannelOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn short_cycle_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.short_cycle_length = value;
        self
    }

    #[inline(always)]
    pub fn medium_cycle_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.medium_cycle_length = value;
        self
    }

    #[inline(always)]
    pub fn short_multiplier_range(mut self, value: (f64, f64, f64)) -> Self {
        self.range.short_multiplier = value;
        self
    }

    #[inline(always)]
    pub fn medium_multiplier_range(mut self, value: (f64, f64, f64)) -> Self {
        self.range.medium_multiplier = value;
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
    ) -> Result<CycleChannelOscillatorBatchOutput, CycleChannelOscillatorError> {
        cycle_channel_oscillator_batch_with_kernel(
            source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE)),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<CycleChannelOscillatorBatchOutput, CycleChannelOscillatorError> {
        cycle_channel_oscillator_batch_with_kernel(
            source,
            high,
            low,
            close,
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CycleChannelOscillatorError> {
    if start == 0 {
        return Err(CycleChannelOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut values = Vec::new();
    if step == 0 {
        if start != end {
            return Err(CycleChannelOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        values.push(start);
    } else {
        if start > end {
            return Err(CycleChannelOscillatorError::InvalidRange {
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
    Ok(values)
}

#[inline(always)]
fn expand_f64_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CycleChannelOscillatorError> {
    if !start.is_finite()
        || !end.is_finite()
        || !step.is_finite()
        || start < 0.0
        || end < 0.0
        || step < 0.0
    {
        return Err(CycleChannelOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut values = Vec::new();
    if step == 0.0 {
        if (start - end).abs() > 1e-12 {
            return Err(CycleChannelOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        values.push(start);
    } else {
        if start > end {
            return Err(CycleChannelOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut current = start;
        while current <= end + 1e-12 {
            values.push(current);
            current += step;
        }
    }
    Ok(values)
}

pub fn expand_grid(
    sweep: &CycleChannelOscillatorBatchRange,
) -> Result<Vec<CycleChannelOscillatorParams>, CycleChannelOscillatorError> {
    let short_lengths = expand_usize_range(
        sweep.short_cycle_length.0,
        sweep.short_cycle_length.1,
        sweep.short_cycle_length.2,
    )?;
    let medium_lengths = expand_usize_range(
        sweep.medium_cycle_length.0,
        sweep.medium_cycle_length.1,
        sweep.medium_cycle_length.2,
    )?;
    let short_multipliers = expand_f64_range(
        sweep.short_multiplier.0,
        sweep.short_multiplier.1,
        sweep.short_multiplier.2,
    )?;
    let medium_multipliers = expand_f64_range(
        sweep.medium_multiplier.0,
        sweep.medium_multiplier.1,
        sweep.medium_multiplier.2,
    )?;

    let mut combos = Vec::new();
    for short_cycle_length in short_lengths.iter().copied() {
        for medium_cycle_length in medium_lengths.iter().copied() {
            for short_multiplier in short_multipliers.iter().copied() {
                for medium_multiplier in medium_multipliers.iter().copied() {
                    combos.push(CycleChannelOscillatorParams {
                        short_cycle_length: Some(short_cycle_length),
                        medium_cycle_length: Some(medium_cycle_length),
                        short_multiplier: Some(short_multiplier),
                        medium_multiplier: Some(medium_multiplier),
                    });
                }
            }
        }
    }
    Ok(combos)
}

#[inline(always)]
fn validate_raw_slices(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<usize, CycleChannelOscillatorError> {
    if source.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(CycleChannelOscillatorError::EmptyInputData);
    }
    if source.len() != high.len() || source.len() != low.len() || source.len() != close.len() {
        return Err(CycleChannelOscillatorError::InconsistentSliceLengths {
            source_len: source.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    first_valid_quad(source, high, low, close).ok_or(CycleChannelOscillatorError::AllValuesNaN)
}

pub fn cycle_channel_oscillator_batch_with_kernel(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CycleChannelOscillatorBatchRange,
    kernel: Kernel,
) -> Result<CycleChannelOscillatorBatchOutput, CycleChannelOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CycleChannelOscillatorError::InvalidKernelForBatch(kernel)),
    };
    cycle_channel_oscillator_batch_par_slice(
        source,
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn cycle_channel_oscillator_batch_slice(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CycleChannelOscillatorBatchRange,
    kernel: Kernel,
) -> Result<CycleChannelOscillatorBatchOutput, CycleChannelOscillatorError> {
    cycle_channel_oscillator_batch_inner(source, high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn cycle_channel_oscillator_batch_par_slice(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CycleChannelOscillatorBatchRange,
    kernel: Kernel,
) -> Result<CycleChannelOscillatorBatchOutput, CycleChannelOscillatorError> {
    cycle_channel_oscillator_batch_inner(source, high, low, close, sweep, kernel, true)
}

fn cycle_channel_oscillator_batch_inner(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CycleChannelOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<CycleChannelOscillatorBatchOutput, CycleChannelOscillatorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(source, high, low, close)?;
    let resolveds = combos
        .iter()
        .map(resolve_params)
        .collect::<Result<Vec<_>, _>>()?;
    let max_needed = resolveds
        .iter()
        .map(|resolved| resolved.medium_period)
        .max()
        .unwrap_or(DEFAULT_MEDIUM_CYCLE_LENGTH / 2);
    let valid = source.len().saturating_sub(first);
    if valid < max_needed {
        return Err(CycleChannelOscillatorError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let rows = combos.len();
    let cols = source.len();
    let warmups: Vec<usize> = resolveds
        .iter()
        .map(|resolved| (first + resolved.medium_period - 1).min(cols))
        .collect();

    let mut fast_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut fast_buf, cols, &warmups);
    let mut fast_guard = ManuallyDrop::new(fast_buf);
    let out_fast: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(fast_guard.as_mut_ptr() as *mut f64, fast_guard.len())
    };

    let mut slow_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut slow_buf, cols, &warmups);
    let mut slow_guard = ManuallyDrop::new(slow_buf);
    let out_slow: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(slow_guard.as_mut_ptr() as *mut f64, slow_guard.len())
    };

    cycle_channel_oscillator_batch_inner_into(
        source, high, low, close, sweep, kernel, parallel, out_fast, out_slow,
    )?;

    let fast = unsafe {
        Vec::from_raw_parts(
            fast_guard.as_mut_ptr() as *mut f64,
            fast_guard.len(),
            fast_guard.capacity(),
        )
    };
    let slow = unsafe {
        Vec::from_raw_parts(
            slow_guard.as_mut_ptr() as *mut f64,
            slow_guard.len(),
            slow_guard.capacity(),
        )
    };

    Ok(CycleChannelOscillatorBatchOutput {
        fast,
        slow,
        combos,
        rows,
        cols,
    })
}

pub fn cycle_channel_oscillator_batch_into_slice(
    out_fast: &mut [f64],
    out_slow: &mut [f64],
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CycleChannelOscillatorBatchRange,
    kernel: Kernel,
) -> Result<(), CycleChannelOscillatorError> {
    cycle_channel_oscillator_batch_inner_into(
        source, high, low, close, sweep, kernel, false, out_fast, out_slow,
    )?;
    Ok(())
}

fn cycle_channel_oscillator_batch_inner_into(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CycleChannelOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_fast: &mut [f64],
    out_slow: &mut [f64],
) -> Result<Vec<CycleChannelOscillatorParams>, CycleChannelOscillatorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(source, high, low, close)?;
    let resolveds = combos
        .iter()
        .map(resolve_params)
        .collect::<Result<Vec<_>, _>>()?;
    let rows = combos.len();
    let cols = source.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| CycleChannelOscillatorError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    if out_fast.len() != expected || out_slow.len() != expected {
        return Err(CycleChannelOscillatorError::OutputLengthMismatch {
            expected,
            got: out_fast.len().max(out_slow.len()),
        });
    }
    let max_needed = resolveds
        .iter()
        .map(|resolved| resolved.medium_period)
        .max()
        .unwrap_or(DEFAULT_MEDIUM_CYCLE_LENGTH / 2);
    let valid = cols.saturating_sub(first);
    if valid < max_needed {
        return Err(CycleChannelOscillatorError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let do_row = |row: usize, dst_fast: &mut [f64], dst_slow: &mut [f64]| {
        let resolved = resolveds[row];
        let warm = (first + resolved.medium_period - 1).min(cols);
        dst_fast[..warm].fill(f64::NAN);
        dst_slow[..warm].fill(f64::NAN);
        compute_cycle_channel_oscillator_into(
            source, high, low, close, resolved, dst_fast, dst_slow,
        )
        .unwrap();
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_fast
                .par_chunks_mut(cols)
                .zip(out_slow.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (dst_fast, dst_slow))| do_row(row, dst_fast, dst_slow));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for ((row, dst_fast), dst_slow) in out_fast
                .chunks_mut(cols)
                .enumerate()
                .zip(out_slow.chunks_mut(cols))
            {
                do_row(row, dst_fast, dst_slow);
            }
        }
    } else {
        for ((row, dst_fast), dst_slow) in out_fast
            .chunks_mut(cols)
            .enumerate()
            .zip(out_slow.chunks_mut(cols))
        {
            do_row(row, dst_fast, dst_slow);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "cycle_channel_oscillator")]
#[pyo3(signature = (source, high, low, close, short_cycle_length=10, medium_cycle_length=30, short_multiplier=1.0, medium_multiplier=3.0, kernel=None))]
pub fn cycle_channel_oscillator_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    short_cycle_length: usize,
    medium_cycle_length: usize,
    short_multiplier: f64,
    medium_multiplier: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = CycleChannelOscillatorInput::from_slices(
        source,
        high,
        low,
        close,
        CycleChannelOscillatorParams {
            short_cycle_length: Some(short_cycle_length),
            medium_cycle_length: Some(medium_cycle_length),
            short_multiplier: Some(short_multiplier),
            medium_multiplier: Some(medium_multiplier),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| cycle_channel_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("fast", out.fast.into_pyarray(py))?;
    dict.set_item("slow", out.slow.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "CycleChannelOscillatorStream")]
pub struct CycleChannelOscillatorStreamPy {
    stream: CycleChannelOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CycleChannelOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (short_cycle_length=10, medium_cycle_length=30, short_multiplier=1.0, medium_multiplier=3.0))]
    fn new(
        short_cycle_length: usize,
        medium_cycle_length: usize,
        short_multiplier: f64,
        medium_multiplier: f64,
    ) -> PyResult<Self> {
        let stream = CycleChannelOscillatorStream::try_new(CycleChannelOscillatorParams {
            short_cycle_length: Some(short_cycle_length),
            medium_cycle_length: Some(medium_cycle_length),
            short_multiplier: Some(short_multiplier),
            medium_multiplier: Some(medium_multiplier),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, source: f64, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.stream.update(source, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cycle_channel_oscillator_batch")]
#[pyo3(signature = (source, high, low, close, short_cycle_length_range=(10,10,0), medium_cycle_length_range=(30,30,0), short_multiplier_range=(1.0,1.0,0.0), medium_multiplier_range=(3.0,3.0,0.0), kernel=None))]
pub fn cycle_channel_oscillator_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    short_cycle_length_range: (usize, usize, usize),
    medium_cycle_length_range: (usize, usize, usize),
    short_multiplier_range: (f64, f64, f64),
    medium_multiplier_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = CycleChannelOscillatorBatchRange {
        short_cycle_length: short_cycle_length_range,
        medium_cycle_length: medium_cycle_length_range,
        short_multiplier: short_multiplier_range,
        medium_multiplier: medium_multiplier_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_fast = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slow = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let fast_slice = unsafe { out_fast.as_slice_mut()? };
    let slow_slice = unsafe { out_slow.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        cycle_channel_oscillator_batch_inner_into(
            source,
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            fast_slice,
            slow_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("fast", out_fast.reshape((rows, cols))?)?;
    dict.set_item("slow", out_slow.reshape((rows, cols))?)?;
    dict.set_item(
        "short_cycle_lengths",
        combos
            .iter()
            .map(|combo| {
                combo
                    .short_cycle_length
                    .unwrap_or(DEFAULT_SHORT_CYCLE_LENGTH) as u64
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "medium_cycle_lengths",
        combos
            .iter()
            .map(|combo| {
                combo
                    .medium_cycle_length
                    .unwrap_or(DEFAULT_MEDIUM_CYCLE_LENGTH) as u64
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "short_multipliers",
        combos
            .iter()
            .map(|combo| combo.short_multiplier.unwrap_or(DEFAULT_SHORT_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "medium_multipliers",
        combos
            .iter()
            .map(|combo| combo.medium_multiplier.unwrap_or(DEFAULT_MEDIUM_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_cycle_channel_oscillator_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cycle_channel_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(cycle_channel_oscillator_batch_py, m)?)?;
    m.add_class::<CycleChannelOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct CycleChannelOscillatorJsOutput {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "cycle_channel_oscillator_js")]
pub fn cycle_channel_oscillator_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    short_cycle_length: usize,
    medium_cycle_length: usize,
    short_multiplier: f64,
    medium_multiplier: f64,
) -> Result<JsValue, JsValue> {
    let input = CycleChannelOscillatorInput::from_slices(
        source,
        high,
        low,
        close,
        CycleChannelOscillatorParams {
            short_cycle_length: Some(short_cycle_length),
            medium_cycle_length: Some(medium_cycle_length),
            short_multiplier: Some(short_multiplier),
            medium_multiplier: Some(medium_multiplier),
        },
    );
    let out = cycle_channel_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&CycleChannelOscillatorJsOutput {
        fast: out.fast,
        slow: out.slow,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct CycleChannelOscillatorBatchConfig {
    pub short_cycle_length_range: Vec<f64>,
    pub medium_cycle_length_range: Vec<f64>,
    pub short_multiplier_range: Vec<f64>,
    pub medium_multiplier_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct CycleChannelOscillatorBatchJsOutput {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
    pub short_cycle_lengths: Vec<usize>,
    pub medium_cycle_lengths: Vec<usize>,
    pub short_multipliers: Vec<f64>,
    pub medium_multipliers: Vec<f64>,
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
fn js_vec3_to_f64(name: &str, values: &[f64]) -> Result<(f64, f64, f64), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} values must be finite non-negative numbers"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "cycle_channel_oscillator_batch_js")]
pub fn cycle_channel_oscillator_batch_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: CycleChannelOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = CycleChannelOscillatorBatchRange {
        short_cycle_length: js_vec3_to_usize(
            "short_cycle_length_range",
            &config.short_cycle_length_range,
        )?,
        medium_cycle_length: js_vec3_to_usize(
            "medium_cycle_length_range",
            &config.medium_cycle_length_range,
        )?,
        short_multiplier: js_vec3_to_f64("short_multiplier_range", &config.short_multiplier_range)?,
        medium_multiplier: js_vec3_to_f64(
            "medium_multiplier_range",
            &config.medium_multiplier_range,
        )?,
    };
    let out =
        cycle_channel_oscillator_batch_with_kernel(source, high, low, close, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&CycleChannelOscillatorBatchJsOutput {
        short_cycle_lengths: out
            .combos
            .iter()
            .map(|combo| {
                combo
                    .short_cycle_length
                    .unwrap_or(DEFAULT_SHORT_CYCLE_LENGTH)
            })
            .collect(),
        medium_cycle_lengths: out
            .combos
            .iter()
            .map(|combo| {
                combo
                    .medium_cycle_length
                    .unwrap_or(DEFAULT_MEDIUM_CYCLE_LENGTH)
            })
            .collect(),
        short_multipliers: out
            .combos
            .iter()
            .map(|combo| combo.short_multiplier.unwrap_or(DEFAULT_SHORT_MULTIPLIER))
            .collect(),
        medium_multipliers: out
            .combos
            .iter()
            .map(|combo| combo.medium_multiplier.unwrap_or(DEFAULT_MEDIUM_MULTIPLIER))
            .collect(),
        fast: out.fast,
        slow: out.slow,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cycle_channel_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cycle_channel_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cycle_channel_oscillator_into(
    source_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_fast_ptr: *mut f64,
    out_slow_ptr: *mut f64,
    len: usize,
    short_cycle_length: usize,
    medium_cycle_length: usize,
    short_multiplier: f64,
    medium_multiplier: f64,
) -> Result<(), JsValue> {
    if source_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_fast_ptr.is_null()
        || out_slow_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to cycle_channel_oscillator_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out_fast = std::slice::from_raw_parts_mut(out_fast_ptr, len);
        let out_slow = std::slice::from_raw_parts_mut(out_slow_ptr, len);
        let input = CycleChannelOscillatorInput::from_slices(
            source,
            high,
            low,
            close,
            CycleChannelOscillatorParams {
                short_cycle_length: Some(short_cycle_length),
                medium_cycle_length: Some(medium_cycle_length),
                short_multiplier: Some(short_multiplier),
                medium_multiplier: Some(medium_multiplier),
            },
        );
        cycle_channel_oscillator_into_slice(out_fast, out_slow, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cycle_channel_oscillator_batch_into(
    source_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_fast_ptr: *mut f64,
    out_slow_ptr: *mut f64,
    len: usize,
    short_cycle_length_start: usize,
    short_cycle_length_end: usize,
    short_cycle_length_step: usize,
    medium_cycle_length_start: usize,
    medium_cycle_length_end: usize,
    medium_cycle_length_step: usize,
    short_multiplier_start: f64,
    short_multiplier_end: f64,
    short_multiplier_step: f64,
    medium_multiplier_start: f64,
    medium_multiplier_end: f64,
    medium_multiplier_step: f64,
) -> Result<usize, JsValue> {
    if source_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_fast_ptr.is_null()
        || out_slow_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to cycle_channel_oscillator_batch_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = CycleChannelOscillatorBatchRange {
            short_cycle_length: (
                short_cycle_length_start,
                short_cycle_length_end,
                short_cycle_length_step,
            ),
            medium_cycle_length: (
                medium_cycle_length_start,
                medium_cycle_length_end,
                medium_cycle_length_step,
            ),
            short_multiplier: (
                short_multiplier_start,
                short_multiplier_end,
                short_multiplier_step,
            ),
            medium_multiplier: (
                medium_multiplier_start,
                medium_multiplier_end,
                medium_multiplier_step,
            ),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in cycle_channel_oscillator_batch_into")
        })?;
        let out_fast = std::slice::from_raw_parts_mut(out_fast_ptr, total);
        let out_slow = std::slice::from_raw_parts_mut(out_slow_ptr, total);
        cycle_channel_oscillator_batch_into_slice(
            out_fast,
            out_slow,
            source,
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cycle_channel_oscillator_output_into_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    short_cycle_length: usize,
    medium_cycle_length: usize,
    short_multiplier: f64,
    medium_multiplier: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = cycle_channel_oscillator_js(
        source,
        high,
        low,
        close,
        short_cycle_length,
        medium_cycle_length,
        short_multiplier,
        medium_multiplier,
    )?;
    crate::write_wasm_object_f64_outputs("cycle_channel_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cycle_channel_oscillator_batch_output_into_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = cycle_channel_oscillator_batch_js(source, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "cycle_channel_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.11).sin() * 2.4 + (i as f64) * 0.03)
            .collect();
        let high: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 1.1 + ((i as f64) * 0.17).cos().abs() * 0.4)
            .collect();
        let low: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - 1.0 - ((i as f64) * 0.13).sin().abs() * 0.35)
            .collect();
        let source = close.clone();
        (source, high, low, close)
    }

    fn manual_rma(src: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; src.len()];
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut value = f64::NAN;
        for (i, &x) in src.iter().enumerate() {
            if !x.is_finite() {
                continue;
            }
            if count < length {
                sum += x;
                count += 1;
                if count == length {
                    value = sum / length as f64;
                    out[i] = value;
                }
            } else {
                value = value + (x - value) / length as f64;
                count += 1;
                out[i] = value;
            }
        }
        out
    }

    fn manual_atr(high: &[f64], low: &[f64], close: &[f64], length: usize) -> Vec<f64> {
        let mut tr = vec![f64::NAN; close.len()];
        let mut prev_close: Option<f64> = None;
        for i in 0..close.len() {
            if !(high[i].is_finite() && low[i].is_finite() && close[i].is_finite()) {
                continue;
            }
            tr[i] = match prev_close {
                Some(prev) => (high[i] - low[i])
                    .max((high[i] - prev).abs())
                    .max((low[i] - prev).abs()),
                None => high[i] - low[i],
            };
            prev_close = Some(close[i]);
        }
        manual_rma(&tr, length)
    }

    fn delayed_or_current(values: &[f64], idx: usize, delay: usize, current: f64) -> f64 {
        if idx >= delay {
            let value = values[idx - delay];
            if value.is_finite() {
                value
            } else {
                current
            }
        } else {
            current
        }
    }

    fn manual_cycle_channel_oscillator(
        source: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        params: CycleChannelOscillatorParams,
    ) -> (Vec<f64>, Vec<f64>) {
        let resolved = resolve_params(&params).unwrap();
        let short_ma = manual_rma(source, resolved.short_period);
        let medium_ma = manual_rma(source, resolved.medium_period);
        let medium_atr = manual_atr(high, low, close, resolved.medium_period);
        let mut fast = vec![f64::NAN; source.len()];
        let mut slow = vec![f64::NAN; source.len()];
        for i in 0..source.len() {
            if !(source[i].is_finite()
                && high[i].is_finite()
                && low[i].is_finite()
                && close[i].is_finite())
            {
                continue;
            }
            let medium_center = delayed_or_current(&medium_ma, i, resolved.medium_delay, source[i]);
            let short_center = delayed_or_current(&short_ma, i, resolved.short_delay, source[i]);
            let offset = resolved.medium_multiplier * medium_atr[i];
            let denom = 2.0 * offset;
            if denom.is_finite() && denom != 0.0 {
                let medium_bottom = medium_center - offset;
                fast[i] = (source[i] - medium_bottom) / denom;
                slow[i] = (short_center - medium_bottom) / denom;
            }
        }
        (fast, slow)
    }

    fn assert_close(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12, "mismatch at {idx}: {a} vs {b}");
        }
    }

    #[test]
    fn manual_reference_matches_api() {
        let (source, high, low, close) = sample_ohlc(160);
        let params = CycleChannelOscillatorParams::default();
        let input =
            CycleChannelOscillatorInput::from_slices(&source, &high, &low, &close, params.clone());
        let out = cycle_channel_oscillator(&input).unwrap();
        let (want_fast, want_slow) =
            manual_cycle_channel_oscillator(&source, &high, &low, &close, params);
        assert_close(&out.fast, &want_fast);
        assert_close(&out.slow, &want_slow);
    }

    #[test]
    fn stream_matches_batch() {
        let (source, high, low, close) = sample_ohlc(128);
        let params = CycleChannelOscillatorParams::default();
        let input =
            CycleChannelOscillatorInput::from_slices(&source, &high, &low, &close, params.clone());
        let batch = cycle_channel_oscillator(&input).unwrap();
        let mut stream = CycleChannelOscillatorStream::try_new(params).unwrap();
        let mut got_fast = Vec::with_capacity(source.len());
        let mut got_slow = Vec::with_capacity(source.len());
        for i in 0..source.len() {
            let (fast, slow) = stream.update(source[i], high[i], low[i], close[i]);
            got_fast.push(fast);
            got_slow.push(slow);
        }
        assert_close(&batch.fast, &got_fast);
        assert_close(&batch.slow, &got_slow);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (source, high, low, close) = sample_ohlc(96);
        let batch = cycle_channel_oscillator_batch_with_kernel(
            &source,
            &high,
            &low,
            &close,
            &CycleChannelOscillatorBatchRange {
                short_cycle_length: (10, 10, 0),
                medium_cycle_length: (30, 32, 2),
                short_multiplier: (1.0, 1.0, 0.0),
                medium_multiplier: (3.0, 3.0, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();
        let input = CycleChannelOscillatorInput::from_slices(
            &source,
            &high,
            &low,
            &close,
            CycleChannelOscillatorParams::default(),
        );
        let single = cycle_channel_oscillator(&input).unwrap();
        assert_eq!(batch.rows, 2);
        assert_close(&batch.fast[..96], single.fast.as_slice());
        assert_close(&batch.slow[..96], single.slow.as_slice());
    }

    #[test]
    fn into_slice_matches_single() {
        let (source, high, low, close) = sample_ohlc(72);
        let input = CycleChannelOscillatorInput::from_slices(
            &source,
            &high,
            &low,
            &close,
            CycleChannelOscillatorParams::default(),
        );
        let single = cycle_channel_oscillator(&input).unwrap();
        let mut out_fast = vec![0.0; source.len()];
        let mut out_slow = vec![0.0; source.len()];
        cycle_channel_oscillator_into_slice(&mut out_fast, &mut out_slow, &input, Kernel::Auto)
            .unwrap();
        assert_close(&single.fast, &out_fast);
        assert_close(&single.slow, &out_slow);
    }

    #[test]
    fn invalid_length_is_rejected() {
        let (source, high, low, close) = sample_ohlc(32);
        let input = CycleChannelOscillatorInput::from_slices(
            &source,
            &high,
            &low,
            &close,
            CycleChannelOscillatorParams {
                short_cycle_length: Some(1),
                ..CycleChannelOscillatorParams::default()
            },
        );
        let err = cycle_channel_oscillator(&input).unwrap_err();
        assert!(matches!(
            err,
            CycleChannelOscillatorError::InvalidCycleLength {
                name: "short_cycle_length",
                ..
            }
        ));
    }
}
