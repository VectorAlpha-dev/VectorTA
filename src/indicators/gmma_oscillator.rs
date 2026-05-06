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
    alloc_uninit_f64, alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

const GUPPY_FAST_PERIODS: [usize; 6] = [3, 5, 8, 10, 12, 15];
const GUPPY_SLOW_PERIODS: [usize; 6] = [30, 35, 40, 45, 50, 60];
const SUPER_GUPPY_FAST_PERIODS: [usize; 11] = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
const SUPER_GUPPY_SLOW_PERIODS: [usize; 16] = [
    25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70,
];

#[inline(always)]
fn gmma_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "close" => candles.close.as_slice(),
        "open" => candles.open.as_slice(),
        "high" => candles.high.as_slice(),
        "low" => candles.low.as_slice(),
        "volume" => candles.volume.as_slice(),
        "hl2" => candles.hl2.as_slice(),
        "hlc3" => candles.hlc3.as_slice(),
        "ohlc4" => candles.ohlc4.as_slice(),
        "hlcc4" => candles.hlcc4.as_slice(),
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GmmaOscillatorMode {
    Guppy,
    SuperGuppy,
}

impl GmmaOscillatorMode {
    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Guppy => "guppy",
            Self::SuperGuppy => "super_guppy",
        }
    }

    #[inline(always)]
    fn from_str(value: &str) -> Result<Self, GmmaOscillatorError> {
        if value.eq_ignore_ascii_case("guppy") {
            return Ok(Self::Guppy);
        }
        if value.eq_ignore_ascii_case("super_guppy")
            || value.eq_ignore_ascii_case("superguppy")
            || value.eq_ignore_ascii_case("super-guppy")
        {
            return Ok(Self::SuperGuppy);
        }
        Err(GmmaOscillatorError::InvalidGmmaType {
            gmma_type: value.to_string(),
        })
    }

    #[inline(always)]
    fn fast_periods(self) -> &'static [usize] {
        match self {
            Self::Guppy => &GUPPY_FAST_PERIODS,
            Self::SuperGuppy => &SUPER_GUPPY_FAST_PERIODS,
        }
    }

    #[inline(always)]
    fn slow_periods(self) -> &'static [usize] {
        match self {
            Self::Guppy => &GUPPY_SLOW_PERIODS,
            Self::SuperGuppy => &SUPER_GUPPY_SLOW_PERIODS,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GmmaOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct GmmaOscillatorOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct GmmaOscillatorParams {
    pub gmma_type: Option<String>,
    pub smooth_length: Option<usize>,
    pub signal_length: Option<usize>,
    pub anchor_minutes: Option<usize>,
    pub interval_minutes: Option<usize>,
}

impl Default for GmmaOscillatorParams {
    fn default() -> Self {
        Self {
            gmma_type: Some("guppy".to_string()),
            smooth_length: Some(1),
            signal_length: Some(13),
            anchor_minutes: Some(0),
            interval_minutes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GmmaOscillatorInput<'a> {
    pub data: GmmaOscillatorData<'a>,
    pub params: GmmaOscillatorParams,
}

impl<'a> GmmaOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: GmmaOscillatorParams,
    ) -> Self {
        Self {
            data: GmmaOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: GmmaOscillatorParams) -> Self {
        Self {
            data: GmmaOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", GmmaOscillatorParams::default())
    }

    #[inline]
    pub fn get_gmma_type(&self) -> &str {
        self.params.gmma_type.as_deref().unwrap_or("guppy")
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(1)
    }

    #[inline]
    pub fn get_signal_length(&self) -> usize {
        self.params.signal_length.unwrap_or(13)
    }

    #[inline]
    pub fn get_anchor_minutes(&self) -> usize {
        self.params.anchor_minutes.unwrap_or(0)
    }

    #[inline]
    pub fn get_interval_minutes(&self) -> Option<usize> {
        self.params.interval_minutes
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GmmaOscillatorBuilder {
    gmma_type: Option<&'static str>,
    smooth_length: Option<usize>,
    signal_length: Option<usize>,
    anchor_minutes: Option<usize>,
    interval_minutes: Option<usize>,
    kernel: Kernel,
}

impl Default for GmmaOscillatorBuilder {
    fn default() -> Self {
        Self {
            gmma_type: None,
            smooth_length: None,
            signal_length: None,
            anchor_minutes: None,
            interval_minutes: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GmmaOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn gmma_type(mut self, value: &'static str) -> Self {
        self.gmma_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth_length(mut self, value: usize) -> Self {
        self.smooth_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_length(mut self, value: usize) -> Self {
        self.signal_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn anchor_minutes(mut self, value: usize) -> Self {
        self.anchor_minutes = Some(value);
        self
    }

    #[inline(always)]
    pub fn interval_minutes(mut self, value: usize) -> Self {
        self.interval_minutes = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn build_params(self) -> GmmaOscillatorParams {
        GmmaOscillatorParams {
            gmma_type: self.gmma_type.map(str::to_string),
            smooth_length: self.smooth_length,
            signal_length: self.signal_length,
            anchor_minutes: self.anchor_minutes,
            interval_minutes: self.interval_minutes,
        }
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<GmmaOscillatorOutput, GmmaOscillatorError> {
        gmma_oscillator_with_kernel(
            &GmmaOscillatorInput::from_candles(candles, "close", self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<GmmaOscillatorOutput, GmmaOscillatorError> {
        gmma_oscillator_with_kernel(
            &GmmaOscillatorInput::from_candles(candles, source, self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<GmmaOscillatorOutput, GmmaOscillatorError> {
        gmma_oscillator_with_kernel(
            &GmmaOscillatorInput::from_slice(data, self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<GmmaOscillatorStream, GmmaOscillatorError> {
        GmmaOscillatorStream::try_new(self.build_params())
    }
}

#[derive(Debug, Error)]
pub enum GmmaOscillatorError {
    #[error("gmma_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("gmma_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("gmma_oscillator: Invalid GMMA type: {gmma_type}")]
    InvalidGmmaType { gmma_type: String },
    #[error("gmma_oscillator: Invalid smooth_length: {smooth_length}")]
    InvalidSmoothLength { smooth_length: usize },
    #[error("gmma_oscillator: Invalid signal_length: {signal_length}")]
    InvalidSignalLength { signal_length: usize },
    #[error("gmma_oscillator: Invalid anchor_minutes: {anchor_minutes}")]
    InvalidAnchorMinutes { anchor_minutes: usize },
    #[error("gmma_oscillator: Invalid interval_minutes: {interval_minutes}")]
    InvalidIntervalMinutes { interval_minutes: usize },
    #[error("gmma_oscillator: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("gmma_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("gmma_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("gmma_oscillator: Output length mismatch: dst = {dst_len}, expected = {expected_len}")]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("gmma_oscillator: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Clone, Copy)]
struct ResolvedInput<'a> {
    data: &'a [f64],
    multiplier: usize,
    mode: GmmaOscillatorMode,
    smooth_length: usize,
    signal_length: usize,
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in data {
        if value.is_finite() {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_params(
    params: &GmmaOscillatorParams,
) -> Result<GmmaOscillatorMode, GmmaOscillatorError> {
    let mode = GmmaOscillatorMode::from_str(params.gmma_type.as_deref().unwrap_or("guppy"))?;
    let smooth_length = params.smooth_length.unwrap_or(1);
    if smooth_length == 0 {
        return Err(GmmaOscillatorError::InvalidSmoothLength { smooth_length });
    }
    let signal_length = params.signal_length.unwrap_or(13);
    if signal_length == 0 {
        return Err(GmmaOscillatorError::InvalidSignalLength { signal_length });
    }
    let anchor_minutes = params.anchor_minutes.unwrap_or(0);
    if anchor_minutes > 1440 {
        return Err(GmmaOscillatorError::InvalidAnchorMinutes { anchor_minutes });
    }
    if let Some(interval_minutes) = params.interval_minutes {
        if interval_minutes == 0 {
            return Err(GmmaOscillatorError::InvalidIntervalMinutes { interval_minutes });
        }
    }
    Ok(mode)
}

#[inline(always)]
fn infer_interval_minutes(timestamps: &[i64]) -> Option<usize> {
    timestamps
        .windows(2)
        .filter_map(|window| {
            let delta = window[1].saturating_sub(window[0]);
            if delta <= 0 {
                return None;
            }
            let minutes = ((delta as f64) / 60_000.0).round() as usize;
            Some(minutes.max(1))
        })
        .find(|&value| value > 0)
}

#[inline(always)]
fn resolve_multiplier(
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
) -> Result<usize, GmmaOscillatorError> {
    if anchor_minutes == 0 {
        return Ok(1);
    }
    let interval_minutes = interval_minutes.ok_or_else(|| GmmaOscillatorError::InvalidInput {
        msg: "anchor_minutes requires interval_minutes or candle timestamps".to_string(),
    })?;
    if interval_minutes == 0 {
        return Err(GmmaOscillatorError::InvalidIntervalMinutes { interval_minutes });
    }
    if interval_minutes >= anchor_minutes {
        return Ok(1);
    }
    let ratio = (anchor_minutes as f64 / interval_minutes as f64).round();
    Ok(ratio.max(1.0) as usize)
}

fn resolve_input<'a>(
    input: &'a GmmaOscillatorInput<'a>,
) -> Result<ResolvedInput<'a>, GmmaOscillatorError> {
    let mode = validate_params(&input.params)?;
    let smooth_length = input.get_smooth_length();
    let signal_length = input.get_signal_length();
    let anchor_minutes = input.get_anchor_minutes();
    match &input.data {
        GmmaOscillatorData::Slice(data) => {
            if data.is_empty() {
                return Err(GmmaOscillatorError::EmptyInputData);
            }
            if longest_valid_run(data) == 0 {
                return Err(GmmaOscillatorError::AllValuesNaN);
            }
            let multiplier = resolve_multiplier(anchor_minutes, input.get_interval_minutes())?;
            Ok(ResolvedInput {
                data,
                multiplier,
                mode,
                smooth_length,
                signal_length,
            })
        }
        GmmaOscillatorData::Candles { candles, source } => {
            let data = gmma_source(candles, source);
            if data.is_empty() {
                return Err(GmmaOscillatorError::EmptyInputData);
            }
            if longest_valid_run(data) == 0 {
                return Err(GmmaOscillatorError::AllValuesNaN);
            }
            let interval_minutes = input
                .get_interval_minutes()
                .or_else(|| infer_interval_minutes(&candles.timestamp));
            let multiplier = resolve_multiplier(anchor_minutes, interval_minutes)?;
            Ok(ResolvedInput {
                data,
                multiplier,
                mode,
                smooth_length,
                signal_length,
            })
        }
    }
}

#[derive(Debug, Clone)]
struct GmmaCore {
    fast_count: usize,
    slow_count: usize,
    fast_state: [f64; 16],
    slow_state: [f64; 16],
    fast_alpha: [f64; 16],
    slow_alpha: [f64; 16],
    smooth_length: usize,
    smooth_window: Vec<f64>,
    smooth_sum: f64,
    smooth_count: usize,
    smooth_index: usize,
    signal_alpha: f64,
    signal_state: Option<f64>,
    initialized: bool,
}

impl GmmaCore {
    fn new(
        mode: GmmaOscillatorMode,
        smooth_length: usize,
        signal_length: usize,
        multiplier: usize,
    ) -> Self {
        let mut fast_alpha = [0.0; 16];
        let mut slow_alpha = [0.0; 16];
        let fast_periods = mode.fast_periods();
        let slow_periods = mode.slow_periods();
        for (i, &period) in fast_periods.iter().enumerate() {
            let effective = period.saturating_mul(multiplier).max(1);
            fast_alpha[i] = 2.0 / (effective as f64 + 1.0);
        }
        for (i, &period) in slow_periods.iter().enumerate() {
            let effective = period.saturating_mul(multiplier).max(1);
            slow_alpha[i] = 2.0 / (effective as f64 + 1.0);
        }
        Self {
            fast_count: fast_periods.len(),
            slow_count: slow_periods.len(),
            fast_state: [0.0; 16],
            slow_state: [0.0; 16],
            fast_alpha,
            slow_alpha,
            smooth_length,
            smooth_window: vec![0.0; smooth_length.max(1)],
            smooth_sum: 0.0,
            smooth_count: 0,
            smooth_index: 0,
            signal_alpha: 2.0 / (signal_length as f64 + 1.0),
            signal_state: None,
            initialized: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.fast_state.fill(0.0);
        self.slow_state.fill(0.0);
        self.smooth_window.fill(0.0);
        self.smooth_sum = 0.0;
        self.smooth_count = 0;
        self.smooth_index = 0;
        self.signal_state = None;
        self.initialized = false;
    }

    #[inline(always)]
    fn update_sma(&mut self, value: f64) -> Option<f64> {
        if self.smooth_length == 1 {
            return Some(value);
        }
        if self.smooth_count < self.smooth_length {
            self.smooth_window[self.smooth_index] = value;
            self.smooth_sum += value;
            self.smooth_count += 1;
            self.smooth_index = (self.smooth_index + 1) % self.smooth_length;
            if self.smooth_count < self.smooth_length {
                return None;
            }
            return Some(self.smooth_sum / self.smooth_length as f64);
        }
        let old = self.smooth_window[self.smooth_index];
        self.smooth_window[self.smooth_index] = value;
        self.smooth_sum += value - old;
        self.smooth_index = (self.smooth_index + 1) % self.smooth_length;
        Some(self.smooth_sum / self.smooth_length as f64)
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        if !self.initialized {
            for i in 0..self.fast_count {
                self.fast_state[i] = value;
            }
            for i in 0..self.slow_count {
                self.slow_state[i] = value;
            }
            self.initialized = true;
        } else {
            for i in 0..self.fast_count {
                let prev = self.fast_state[i];
                self.fast_state[i] = prev + self.fast_alpha[i] * (value - prev);
            }
            for i in 0..self.slow_count {
                let prev = self.slow_state[i];
                self.slow_state[i] = prev + self.slow_alpha[i] * (value - prev);
            }
        }

        let mut fast_sum = 0.0;
        for i in 0..self.fast_count {
            fast_sum += self.fast_state[i];
        }
        let mut slow_sum = 0.0;
        for i in 0..self.slow_count {
            slow_sum += self.slow_state[i];
        }
        let fast_avg = fast_sum / self.fast_count as f64;
        let slow_avg = slow_sum / self.slow_count as f64;
        if !slow_avg.is_finite() || slow_avg == 0.0 {
            self.reset();
            return None;
        }

        let raw = ((fast_avg - slow_avg) / slow_avg) * 100.0;
        let signal = match self.signal_state {
            Some(prev) => {
                let next = prev + self.signal_alpha * (raw - prev);
                self.signal_state = Some(next);
                next
            }
            None => {
                self.signal_state = Some(raw);
                raw
            }
        };
        let oscillator = self.update_sma(raw).unwrap_or(f64::NAN);
        Some((oscillator, signal))
    }
}

#[derive(Debug, Clone)]
pub struct GmmaOscillatorStream {
    smooth_length: usize,
    core: GmmaCore,
}

impl GmmaOscillatorStream {
    #[inline(always)]
    pub fn try_new(params: GmmaOscillatorParams) -> Result<Self, GmmaOscillatorError> {
        let mode = validate_params(&params)?;
        let smooth_length = params.smooth_length.unwrap_or(1);
        let signal_length = params.signal_length.unwrap_or(13);
        let multiplier =
            resolve_multiplier(params.anchor_minutes.unwrap_or(0), params.interval_minutes)?;
        Ok(Self {
            smooth_length,
            core: GmmaCore::new(mode, smooth_length, signal_length, multiplier),
        })
    }

    #[inline(always)]
    fn from_resolved(resolved: ResolvedInput<'_>) -> Self {
        Self {
            smooth_length: resolved.smooth_length,
            core: GmmaCore::new(
                resolved.mode,
                resolved.smooth_length,
                resolved.signal_length,
                resolved.multiplier,
            ),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.core.update(value)
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.core.reset();
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.smooth_length.saturating_sub(1)
    }
}

#[inline(always)]
fn compute_row(
    data: &[f64],
    resolved: ResolvedInput<'_>,
    oscillator: &mut [f64],
    signal: &mut [f64],
) {
    oscillator.fill(f64::NAN);
    signal.fill(f64::NAN);
    let mut stream = GmmaOscillatorStream::from_resolved(resolved);
    for (idx, &value) in data.iter().enumerate() {
        if let Some((osc_value, sig_value)) = stream.update(value) {
            oscillator[idx] = osc_value;
            signal[idx] = sig_value;
        }
    }
}

pub fn gmma_oscillator(
    input: &GmmaOscillatorInput,
) -> Result<GmmaOscillatorOutput, GmmaOscillatorError> {
    gmma_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn gmma_oscillator_with_kernel(
    input: &GmmaOscillatorInput,
    kernel: Kernel,
) -> Result<GmmaOscillatorOutput, GmmaOscillatorError> {
    let resolved = resolve_input(input)?;
    let _ = kernel;

    let mut oscillator = alloc_uninit_f64(resolved.data.len());
    let mut signal = alloc_uninit_f64(resolved.data.len());
    compute_row(resolved.data, resolved, &mut oscillator, &mut signal);
    Ok(GmmaOscillatorOutput { oscillator, signal })
}

pub fn gmma_oscillator_into_slice(
    dst_oscillator: &mut [f64],
    dst_signal: &mut [f64],
    input: &GmmaOscillatorInput,
    kernel: Kernel,
) -> Result<(), GmmaOscillatorError> {
    let resolved = resolve_input(input)?;
    if dst_oscillator.len() != resolved.data.len() {
        return Err(GmmaOscillatorError::OutputLengthMismatch {
            expected: resolved.data.len(),
            got: dst_oscillator.len(),
        });
    }
    if dst_signal.len() != resolved.data.len() {
        return Err(GmmaOscillatorError::OutputLengthMismatch {
            expected: resolved.data.len(),
            got: dst_signal.len(),
        });
    }
    let _ = kernel;
    compute_row(resolved.data, resolved, dst_oscillator, dst_signal);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn gmma_oscillator_into(
    input: &GmmaOscillatorInput,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), GmmaOscillatorError> {
    gmma_oscillator_into_slice(out_oscillator, out_signal, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct GmmaOscillatorBatchRange {
    pub smooth_length: (usize, usize, usize),
    pub signal_length: (usize, usize, usize),
}

impl Default for GmmaOscillatorBatchRange {
    fn default() -> Self {
        Self {
            smooth_length: (1, 1, 0),
            signal_length: (13, 13, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GmmaOscillatorBatchOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<GmmaOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl GmmaOscillatorBatchOutput {
    pub fn row_for_params(&self, params: &GmmaOscillatorParams) -> Option<usize> {
        let gmma_type =
            GmmaOscillatorMode::from_str(params.gmma_type.as_deref().unwrap_or("guppy"))
                .ok()?
                .as_str();
        let smooth_length = params.smooth_length.unwrap_or(1);
        let signal_length = params.signal_length.unwrap_or(13);
        let anchor_minutes = params.anchor_minutes.unwrap_or(0);
        let interval_minutes = params.interval_minutes;
        self.combos.iter().position(|combo| {
            GmmaOscillatorMode::from_str(combo.gmma_type.as_deref().unwrap_or("guppy"))
                .map(|mode| mode.as_str() == gmma_type)
                .unwrap_or(false)
                && combo.smooth_length.unwrap_or(1) == smooth_length
                && combo.signal_length.unwrap_or(13) == signal_length
                && combo.anchor_minutes.unwrap_or(0) == anchor_minutes
                && combo.interval_minutes == interval_minutes
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GmmaOscillatorBatchBuilder {
    range: GmmaOscillatorBatchRange,
    gmma_type: &'static str,
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
    kernel: Kernel,
}

impl Default for GmmaOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: GmmaOscillatorBatchRange::default(),
            gmma_type: "guppy",
            anchor_minutes: 0,
            interval_minutes: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GmmaOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn gmma_type(mut self, value: &'static str) -> Self {
        self.gmma_type = value;
        self
    }

    #[inline(always)]
    pub fn anchor_minutes(mut self, value: usize) -> Self {
        self.anchor_minutes = value;
        self
    }

    #[inline(always)]
    pub fn interval_minutes(mut self, value: usize) -> Self {
        self.interval_minutes = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.smooth_length = value;
        self
    }

    #[inline(always)]
    pub fn signal_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.signal_length = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn fixed_params(self) -> GmmaOscillatorParams {
        GmmaOscillatorParams {
            gmma_type: Some(self.gmma_type.to_string()),
            smooth_length: None,
            signal_length: None,
            anchor_minutes: Some(self.anchor_minutes),
            interval_minutes: self.interval_minutes,
        }
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<GmmaOscillatorBatchOutput, GmmaOscillatorError> {
        gmma_oscillator_batch_with_kernel(data, &self.range, &self.fixed_params(), self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<GmmaOscillatorBatchOutput, GmmaOscillatorError> {
        let input = GmmaOscillatorInput::from_candles(candles, source, self.fixed_params());
        gmma_oscillator_batch_from_input_with_kernel(&input, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_axis(range: (usize, usize, usize)) -> Result<Vec<usize>, GmmaOscillatorError> {
    let (start, end, step) = range;
    if start == 0 {
        return Err(GmmaOscillatorError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(GmmaOscillatorError::InvalidRange { start, end, step });
    }
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(cur);
        if cur >= end {
            break;
        }
        let next = cur
            .checked_add(step)
            .ok_or_else(|| GmmaOscillatorError::InvalidInput {
                msg: "gmma_oscillator: range step overflow".to_string(),
            })?;
        if next <= cur {
            return Err(GmmaOscillatorError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
    }
    Ok(out)
}

fn expand_grid_checked(
    sweep: &GmmaOscillatorBatchRange,
    fixed: &GmmaOscillatorParams,
) -> Result<Vec<GmmaOscillatorParams>, GmmaOscillatorError> {
    let mode = GmmaOscillatorMode::from_str(fixed.gmma_type.as_deref().unwrap_or("guppy"))?;
    let smooth_lengths = expand_axis(sweep.smooth_length)?;
    let signal_lengths = expand_axis(sweep.signal_length)?;
    let total = smooth_lengths
        .len()
        .checked_mul(signal_lengths.len())
        .ok_or_else(|| GmmaOscillatorError::InvalidInput {
            msg: "gmma_oscillator: parameter grid size overflow".to_string(),
        })?;
    let mut combos = Vec::with_capacity(total);
    for &smooth_length in &smooth_lengths {
        for &signal_length in &signal_lengths {
            combos.push(GmmaOscillatorParams {
                gmma_type: Some(mode.as_str().to_string()),
                smooth_length: Some(smooth_length),
                signal_length: Some(signal_length),
                anchor_minutes: Some(fixed.anchor_minutes.unwrap_or(0)),
                interval_minutes: fixed.interval_minutes,
            });
        }
    }
    Ok(combos)
}

pub fn expand_grid_gmma_oscillator(
    sweep: &GmmaOscillatorBatchRange,
    fixed: &GmmaOscillatorParams,
) -> Result<Vec<GmmaOscillatorParams>, GmmaOscillatorError> {
    expand_grid_checked(sweep, fixed)
}

pub fn gmma_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &GmmaOscillatorBatchRange,
    fixed: &GmmaOscillatorParams,
    kernel: Kernel,
) -> Result<GmmaOscillatorBatchOutput, GmmaOscillatorError> {
    let input = GmmaOscillatorInput::from_slice(data, fixed.clone());
    gmma_oscillator_batch_from_input_with_kernel(&input, sweep, kernel)
}

pub fn gmma_oscillator_batch_from_input_with_kernel(
    input: &GmmaOscillatorInput,
    sweep: &GmmaOscillatorBatchRange,
    kernel: Kernel,
) -> Result<GmmaOscillatorBatchOutput, GmmaOscillatorError> {
    gmma_oscillator_batch_inner(input, sweep, kernel, true)
}

pub fn gmma_oscillator_batch_slice(
    data: &[f64],
    sweep: &GmmaOscillatorBatchRange,
    fixed: &GmmaOscillatorParams,
    kernel: Kernel,
) -> Result<GmmaOscillatorBatchOutput, GmmaOscillatorError> {
    let input = GmmaOscillatorInput::from_slice(data, fixed.clone());
    gmma_oscillator_batch_inner(&input, sweep, kernel, false)
}

pub fn gmma_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &GmmaOscillatorBatchRange,
    fixed: &GmmaOscillatorParams,
    kernel: Kernel,
) -> Result<GmmaOscillatorBatchOutput, GmmaOscillatorError> {
    let input = GmmaOscillatorInput::from_slice(data, fixed.clone());
    gmma_oscillator_batch_inner(&input, sweep, kernel, true)
}

fn gmma_oscillator_batch_inner(
    input: &GmmaOscillatorInput,
    sweep: &GmmaOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<GmmaOscillatorBatchOutput, GmmaOscillatorError> {
    let resolved = resolve_input(input)?;
    let fixed = GmmaOscillatorParams {
        gmma_type: Some(resolved.mode.as_str().to_string()),
        smooth_length: None,
        signal_length: None,
        anchor_minutes: Some(input.get_anchor_minutes()),
        interval_minutes: input.get_interval_minutes(),
    };
    let combos = expand_grid_checked(sweep, &fixed)?;
    let rows = combos.len();
    let cols = resolved.data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| GmmaOscillatorError::InvalidInput {
            msg: "gmma_oscillator: rows*cols overflow in batch".to_string(),
        })?;

    let osc_warmups = combos
        .iter()
        .map(|params| params.smooth_length.unwrap_or(1).saturating_sub(1))
        .collect::<Vec<_>>();
    let signal_warmups = vec![0usize; rows];

    let mut oscillator_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut oscillator_mu, cols, &osc_warmups);
    init_matrix_prefixes(&mut signal_mu, cols, &signal_warmups);

    let mut oscillator = unsafe {
        Vec::from_raw_parts(
            oscillator_mu.as_mut_ptr() as *mut f64,
            oscillator_mu.len(),
            oscillator_mu.capacity(),
        )
    };
    let mut signal = unsafe {
        Vec::from_raw_parts(
            signal_mu.as_mut_ptr() as *mut f64,
            signal_mu.len(),
            signal_mu.capacity(),
        )
    };
    std::mem::forget(oscillator_mu);
    std::mem::forget(signal_mu);
    debug_assert_eq!(oscillator.len(), total);
    debug_assert_eq!(signal.len(), total);

    gmma_oscillator_batch_inner_into(
        resolved.data,
        &combos,
        resolved.mode,
        resolved.multiplier,
        kernel,
        parallel,
        &mut oscillator,
        &mut signal,
    )?;

    Ok(GmmaOscillatorBatchOutput {
        oscillator,
        signal,
        combos,
        rows,
        cols,
    })
}

fn gmma_oscillator_batch_inner_into(
    data: &[f64],
    combos: &[GmmaOscillatorParams],
    default_mode: GmmaOscillatorMode,
    default_multiplier: usize,
    kernel: Kernel,
    parallel: bool,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), GmmaOscillatorError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(GmmaOscillatorError::InvalidKernelForBatch(other)),
    }

    let len = data.len();
    let total = combos
        .len()
        .checked_mul(len)
        .ok_or_else(|| GmmaOscillatorError::InvalidInput {
            msg: "gmma_oscillator: rows*cols overflow in batch_into".to_string(),
        })?;
    if out_oscillator.len() != total {
        return Err(GmmaOscillatorError::MismatchedOutputLen {
            dst_len: out_oscillator.len(),
            expected_len: total,
        });
    }
    if out_signal.len() != total {
        return Err(GmmaOscillatorError::MismatchedOutputLen {
            dst_len: out_signal.len(),
            expected_len: total,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_oscillator: &mut [f64], dst_signal: &mut [f64]| {
        let combo = &combos[row];
        let mode = GmmaOscillatorMode::from_str(
            combo.gmma_type.as_deref().unwrap_or(default_mode.as_str()),
        )
        .unwrap_or(default_mode);
        let resolved = ResolvedInput {
            data,
            multiplier: default_multiplier,
            mode,
            smooth_length: combo.smooth_length.unwrap_or(1),
            signal_length: combo.signal_length.unwrap_or(13),
        };
        compute_row(data, resolved, dst_oscillator, dst_signal);
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_oscillator
                .par_chunks_mut(len)
                .zip(out_signal.par_chunks_mut(len))
                .enumerate()
                .for_each(|(row, (dst_oscillator, dst_signal))| {
                    worker(row, dst_oscillator, dst_signal)
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (dst_oscillator, dst_signal)) in out_oscillator
                .chunks_mut(len)
                .zip(out_signal.chunks_mut(len))
                .enumerate()
            {
                worker(row, dst_oscillator, dst_signal);
            }
        }
    } else {
        for (row, (dst_oscillator, dst_signal)) in out_oscillator
            .chunks_mut(len)
            .zip(out_signal.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_oscillator, dst_signal);
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "gmma_oscillator")]
#[pyo3(signature = (data, gmma_type="guppy", smooth_length=1, signal_length=13, anchor_minutes=0, interval_minutes=None, kernel=None))]
pub fn gmma_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    gmma_type: &str,
    smooth_length: usize,
    signal_length: usize,
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = GmmaOscillatorInput::from_slice(
        data,
        GmmaOscillatorParams {
            gmma_type: Some(gmma_type.to_string()),
            smooth_length: Some(smooth_length),
            signal_length: Some(signal_length),
            anchor_minutes: Some(anchor_minutes),
            interval_minutes,
        },
    );
    let out = py
        .allow_threads(|| gmma_oscillator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.oscillator.into_pyarray(py), out.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "GmmaOscillatorStream")]
pub struct GmmaOscillatorStreamPy {
    stream: GmmaOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GmmaOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (gmma_type="guppy", smooth_length=1, signal_length=13, anchor_minutes=0, interval_minutes=None))]
    fn new(
        gmma_type: &str,
        smooth_length: usize,
        signal_length: usize,
        anchor_minutes: usize,
        interval_minutes: Option<usize>,
    ) -> PyResult<Self> {
        let stream = GmmaOscillatorStream::try_new(GmmaOscillatorParams {
            gmma_type: Some(gmma_type.to_string()),
            smooth_length: Some(smooth_length),
            signal_length: Some(signal_length),
            anchor_minutes: Some(anchor_minutes),
            interval_minutes,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "gmma_oscillator_batch")]
#[pyo3(signature = (data, gmma_type="guppy", smooth_length_range=(1, 1, 0), signal_length_range=(13, 13, 0), anchor_minutes=0, interval_minutes=None, kernel=None))]
pub fn gmma_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    gmma_type: &str,
    smooth_length_range: (usize, usize, usize),
    signal_length_range: (usize, usize, usize),
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            gmma_oscillator_batch_with_kernel(
                data,
                &GmmaOscillatorBatchRange {
                    smooth_length: smooth_length_range,
                    signal_length: signal_length_range,
                },
                &GmmaOscillatorParams {
                    gmma_type: Some(gmma_type.to_string()),
                    smooth_length: None,
                    signal_length: None,
                    anchor_minutes: Some(anchor_minutes),
                    interval_minutes,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "oscillator",
        output
            .oscillator
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "signal",
        output
            .signal
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "smooth_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.smooth_length.unwrap_or(1) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.signal_length.unwrap_or(13) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "gmma_types",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .gmma_type
                    .clone()
                    .unwrap_or_else(|| "guppy".to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_gmma_oscillator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gmma_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(gmma_oscillator_batch_py, m)?)?;
    m.add_class::<GmmaOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmaOscillatorBatchConfig {
    pub gmma_type: Option<String>,
    pub smooth_length_range: Vec<usize>,
    pub signal_length_range: Vec<usize>,
    pub anchor_minutes: Option<usize>,
    pub interval_minutes: Option<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = gmma_oscillator_js)]
pub fn gmma_oscillator_js(
    data: &[f64],
    gmma_type: &str,
    smooth_length: usize,
    signal_length: usize,
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
) -> Result<JsValue, JsValue> {
    let input = GmmaOscillatorInput::from_slice(
        data,
        GmmaOscillatorParams {
            gmma_type: Some(gmma_type.to_string()),
            smooth_length: Some(smooth_length),
            signal_length: Some(signal_length),
            anchor_minutes: Some(anchor_minutes),
            interval_minutes,
        },
    );
    let out = gmma_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("oscillator"),
        &serde_wasm_bindgen::to_value(&out.oscillator).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = gmma_oscillator_batch_js)]
pub fn gmma_oscillator_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: GmmaOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.smooth_length_range.len() != 3 || config.signal_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = gmma_oscillator_batch_with_kernel(
        data,
        &GmmaOscillatorBatchRange {
            smooth_length: (
                config.smooth_length_range[0],
                config.smooth_length_range[1],
                config.smooth_length_range[2],
            ),
            signal_length: (
                config.signal_length_range[0],
                config.signal_length_range[1],
                config.signal_length_range[2],
            ),
        },
        &GmmaOscillatorParams {
            gmma_type: Some(config.gmma_type.unwrap_or_else(|| "guppy".to_string())),
            smooth_length: None,
            signal_length: None,
            anchor_minutes: Some(config.anchor_minutes.unwrap_or(0)),
            interval_minutes: config.interval_minutes,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("oscillator"),
        &serde_wasm_bindgen::to_value(&out.oscillator).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gmma_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(2 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gmma_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gmma_oscillator_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    gmma_type: &str,
    smooth_length: usize,
    signal_length: usize,
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to gmma_oscillator_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (dst_oscillator, dst_signal) = out.split_at_mut(len);
        let input = GmmaOscillatorInput::from_slice(
            data,
            GmmaOscillatorParams {
                gmma_type: Some(gmma_type.to_string()),
                smooth_length: Some(smooth_length),
                signal_length: Some(signal_length),
                anchor_minutes: Some(anchor_minutes),
                interval_minutes,
            },
        );
        gmma_oscillator_into_slice(dst_oscillator, dst_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gmma_oscillator_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    gmma_type: &str,
    smooth_length_start: usize,
    smooth_length_end: usize,
    smooth_length_step: usize,
    signal_length_start: usize,
    signal_length_end: usize,
    signal_length_step: usize,
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to gmma_oscillator_batch_into",
        ));
    }
    let batch = unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        gmma_oscillator_batch_with_kernel(
            data,
            &GmmaOscillatorBatchRange {
                smooth_length: (smooth_length_start, smooth_length_end, smooth_length_step),
                signal_length: (signal_length_start, signal_length_end, signal_length_step),
            },
            &GmmaOscillatorParams {
                gmma_type: Some(gmma_type.to_string()),
                smooth_length: None,
                signal_length: None,
                anchor_minutes: Some(anchor_minutes),
                interval_minutes,
            },
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?
    };
    let rows = batch.rows;
    let total = rows
        .checked_mul(len)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in gmma_oscillator_batch_into"))?;
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_oscillator, dst_signal) = out.split_at_mut(split);
        dst_oscillator.copy_from_slice(&batch.oscillator);
        dst_signal.copy_from_slice(&batch.signal);
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gmma_oscillator_output_into_js(
    data: &[f64],
    gmma_type: &str,
    smooth_length: usize,
    signal_length: usize,
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = gmma_oscillator_js(
        data,
        gmma_type,
        smooth_length,
        signal_length,
        anchor_minutes,
        interval_minutes,
    )?;
    crate::write_wasm_object_f64_outputs("gmma_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gmma_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = gmma_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "gmma_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV,
        ParamValue,
    };

    fn sample_close(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.08 + (x * 0.17).sin() * 1.6 + (x * 0.043).cos() * 0.8
            })
            .collect()
    }

    fn sample_candles(len: usize) -> Candles {
        let close = sample_close(len);
        let open = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - ((i as f64) * 0.11).sin() * 0.4)
            .collect::<Vec<_>>();
        let high = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.max(c) + 0.5)
            .collect::<Vec<_>>();
        let low = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.min(c) - 0.5)
            .collect::<Vec<_>>();
        let volume = vec![1_000.0; len];
        let timestamp = (0..len)
            .map(|i| 1_700_000_000_000i64 + (i as i64) * 300_000i64)
            .collect::<Vec<_>>();
        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn ema_series(data: &[f64], alpha: f64) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        let mut state = None;
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                state = None;
                continue;
            }
            let next = match state {
                Some(prev) => prev + alpha * (value - prev),
                None => value,
            };
            out[i] = next;
            state = Some(next);
        }
        out
    }

    fn sma_series(data: &[f64], period: usize) -> Vec<f64> {
        if period == 1 {
            return data.to_vec();
        }
        let mut out = vec![f64::NAN; data.len()];
        let mut window = vec![0.0; period];
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut index = 0usize;
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                window.fill(0.0);
                sum = 0.0;
                count = 0;
                index = 0;
                continue;
            }
            if count < period {
                window[index] = value;
                sum += value;
                count += 1;
                index = (index + 1) % period;
                if count == period {
                    out[i] = sum / period as f64;
                }
            } else {
                let old = window[index];
                window[index] = value;
                sum += value - old;
                index = (index + 1) % period;
                out[i] = sum / period as f64;
            }
        }
        out
    }

    fn naive_gmma(
        data: &[f64],
        mode: GmmaOscillatorMode,
        multiplier: usize,
        smooth_length: usize,
        signal_length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let fast_periods = mode.fast_periods();
        let slow_periods = mode.slow_periods();
        let fast_ema = fast_periods
            .iter()
            .map(|&period| ema_series(data, 2.0 / ((period * multiplier) as f64 + 1.0)))
            .collect::<Vec<_>>();
        let slow_ema = slow_periods
            .iter()
            .map(|&period| ema_series(data, 2.0 / ((period * multiplier) as f64 + 1.0)))
            .collect::<Vec<_>>();
        let mut raw = vec![f64::NAN; data.len()];
        for i in 0..data.len() {
            let mut fast_sum = 0.0;
            let mut slow_sum = 0.0;
            let mut valid = true;
            for series in &fast_ema {
                if !series[i].is_finite() {
                    valid = false;
                    break;
                }
                fast_sum += series[i];
            }
            if valid {
                for series in &slow_ema {
                    if !series[i].is_finite() {
                        valid = false;
                        break;
                    }
                    slow_sum += series[i];
                }
            }
            if valid {
                let fast_avg = fast_sum / fast_periods.len() as f64;
                let slow_avg = slow_sum / slow_periods.len() as f64;
                raw[i] = ((fast_avg - slow_avg) / slow_avg) * 100.0;
            }
        }
        let oscillator = sma_series(&raw, smooth_length);
        let signal = ema_series(&raw, 2.0 / (signal_length as f64 + 1.0));
        (oscillator, signal)
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn gmma_oscillator_matches_naive_guppy() -> Result<(), Box<dyn Error>> {
        let data = sample_close(256);
        let input = GmmaOscillatorInput::from_slice(
            &data,
            GmmaOscillatorParams {
                gmma_type: Some("guppy".to_string()),
                smooth_length: Some(3),
                signal_length: Some(13),
                anchor_minutes: Some(0),
                interval_minutes: None,
            },
        );
        let out = gmma_oscillator_with_kernel(&input, Kernel::Scalar)?;
        let (expected_oscillator, expected_signal) =
            naive_gmma(&data, GmmaOscillatorMode::Guppy, 1, 3, 13);
        assert_series_close(&out.oscillator, &expected_oscillator, 1e-12);
        assert_series_close(&out.signal, &expected_signal, 1e-12);
        Ok(())
    }

    #[test]
    fn gmma_oscillator_matches_naive_super_guppy_with_anchor() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(220);
        let input = GmmaOscillatorInput::from_candles(
            &candles,
            "close",
            GmmaOscillatorParams {
                gmma_type: Some("super_guppy".to_string()),
                smooth_length: Some(2),
                signal_length: Some(7),
                anchor_minutes: Some(60),
                interval_minutes: None,
            },
        );
        let out = gmma_oscillator_with_kernel(&input, Kernel::Scalar)?;
        let (expected_oscillator, expected_signal) =
            naive_gmma(&candles.close, GmmaOscillatorMode::SuperGuppy, 12, 2, 7);
        assert_series_close(&out.oscillator, &expected_oscillator, 1e-12);
        assert_series_close(&out.signal, &expected_signal, 1e-12);
        Ok(())
    }

    #[test]
    fn gmma_oscillator_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_close(192);
        let input = GmmaOscillatorInput::from_slice(
            &data,
            GmmaOscillatorParams {
                gmma_type: Some("guppy".to_string()),
                smooth_length: Some(4),
                signal_length: Some(9),
                anchor_minutes: Some(0),
                interval_minutes: None,
            },
        );
        let batch = gmma_oscillator(&input)?;
        let mut stream = GmmaOscillatorStream::try_new(input.params.clone())?;
        let mut oscillator = Vec::with_capacity(data.len());
        let mut signal = Vec::with_capacity(data.len());
        for &value in &data {
            match stream.update(value) {
                Some((osc_value, sig_value)) => {
                    oscillator.push(osc_value);
                    signal.push(sig_value);
                }
                None => {
                    oscillator.push(f64::NAN);
                    signal.push(f64::NAN);
                }
            }
        }
        assert_series_close(&oscillator, &batch.oscillator, 1e-12);
        assert_series_close(&signal, &batch.signal, 1e-12);
        Ok(())
    }

    #[test]
    fn gmma_oscillator_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let data = sample_close(180);
        let batch = gmma_oscillator_batch_with_kernel(
            &data,
            &GmmaOscillatorBatchRange {
                smooth_length: (3, 3, 0),
                signal_length: (13, 13, 0),
            },
            &GmmaOscillatorParams {
                gmma_type: Some("guppy".to_string()),
                smooth_length: None,
                signal_length: None,
                anchor_minutes: Some(0),
                interval_minutes: None,
            },
            Kernel::ScalarBatch,
        )?;
        let single = gmma_oscillator(&GmmaOscillatorInput::from_slice(
            &data,
            GmmaOscillatorParams {
                gmma_type: Some("guppy".to_string()),
                smooth_length: Some(3),
                signal_length: Some(13),
                anchor_minutes: Some(0),
                interval_minutes: None,
            },
        ))?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_series_close(&batch.oscillator[0..data.len()], &single.oscillator, 1e-12);
        assert_series_close(&batch.signal[0..data.len()], &single.signal, 1e-12);
        Ok(())
    }

    #[test]
    fn gmma_oscillator_dispatch_outputs_match_direct() -> Result<(), Box<dyn Error>> {
        let data = sample_close(160);
        let osc = compute_cpu(IndicatorComputeRequest {
            indicator_id: "gmma_oscillator",
            output_id: Some("oscillator"),
            data: IndicatorDataRef::Slice { values: &data },
            params: &[
                ParamKV {
                    key: "smooth_length",
                    value: ParamValue::Int(3),
                },
                ParamKV {
                    key: "signal_length",
                    value: ParamValue::Int(13),
                },
                ParamKV {
                    key: "gmma_type",
                    value: ParamValue::EnumString("guppy"),
                },
            ],
            kernel: Kernel::Auto,
        })?;
        let sig = compute_cpu(IndicatorComputeRequest {
            indicator_id: "gmma_oscillator",
            output_id: Some("signal"),
            data: IndicatorDataRef::Slice { values: &data },
            params: &[
                ParamKV {
                    key: "smooth_length",
                    value: ParamValue::Int(3),
                },
                ParamKV {
                    key: "signal_length",
                    value: ParamValue::Int(13),
                },
                ParamKV {
                    key: "gmma_type",
                    value: ParamValue::EnumString("guppy"),
                },
            ],
            kernel: Kernel::Auto,
        })?;
        let direct = gmma_oscillator(&GmmaOscillatorInput::from_slice(
            &data,
            GmmaOscillatorParams {
                gmma_type: Some("guppy".to_string()),
                smooth_length: Some(3),
                signal_length: Some(13),
                anchor_minutes: Some(0),
                interval_minutes: None,
            },
        ))?;

        let osc_values = match osc.series {
            IndicatorSeries::F64(values) => values,
            _ => panic!("expected f64 series"),
        };
        let sig_values = match sig.series {
            IndicatorSeries::F64(values) => values,
            _ => panic!("expected f64 series"),
        };
        assert_series_close(&osc_values, &direct.oscillator, 1e-12);
        assert_series_close(&sig_values, &direct.signal, 1e-12);
        Ok(())
    }
}
