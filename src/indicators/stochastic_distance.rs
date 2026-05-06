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
    alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LOOKBACK_LENGTH: usize = 200;
const DEFAULT_LENGTH1: usize = 12;
const DEFAULT_LENGTH2: usize = 3;
const DEFAULT_OB_LEVEL: i32 = 40;
const DEFAULT_OS_LEVEL: i32 = -40;
const FLOAT_TOL: f64 = 1e-12;

impl<'a> AsRef<[f64]> for StochasticDistanceInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            StochasticDistanceData::Slice(slice) => slice,
            StochasticDistanceData::Candles { candles } => &candles.close,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StochasticDistanceData<'a> {
    Candles { candles: &'a Candles },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct StochasticDistanceOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StochasticDistanceParams {
    pub lookback_length: Option<usize>,
    pub length1: Option<usize>,
    pub length2: Option<usize>,
    pub ob_level: Option<i32>,
    pub os_level: Option<i32>,
}

impl Default for StochasticDistanceParams {
    fn default() -> Self {
        Self {
            lookback_length: Some(DEFAULT_LOOKBACK_LENGTH),
            length1: Some(DEFAULT_LENGTH1),
            length2: Some(DEFAULT_LENGTH2),
            ob_level: Some(DEFAULT_OB_LEVEL),
            os_level: Some(DEFAULT_OS_LEVEL),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochasticDistanceInput<'a> {
    pub data: StochasticDistanceData<'a>,
    pub params: StochasticDistanceParams,
}

impl<'a> StochasticDistanceInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: StochasticDistanceParams) -> Self {
        Self {
            data: StochasticDistanceData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: StochasticDistanceParams) -> Self {
        Self {
            data: StochasticDistanceData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, StochasticDistanceParams::default())
    }
}

#[derive(Clone, Debug)]
pub struct StochasticDistanceBuilder {
    lookback_length: Option<usize>,
    length1: Option<usize>,
    length2: Option<usize>,
    ob_level: Option<i32>,
    os_level: Option<i32>,
    kernel: Kernel,
}

impl Default for StochasticDistanceBuilder {
    fn default() -> Self {
        Self {
            lookback_length: None,
            length1: None,
            length2: None,
            ob_level: None,
            os_level: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochasticDistanceBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn lookback_length(mut self, lookback_length: usize) -> Self {
        self.lookback_length = Some(lookback_length);
        self
    }

    #[inline]
    pub fn length1(mut self, length1: usize) -> Self {
        self.length1 = Some(length1);
        self
    }

    #[inline]
    pub fn length2(mut self, length2: usize) -> Self {
        self.length2 = Some(length2);
        self
    }

    #[inline]
    pub fn ob_level(mut self, ob_level: i32) -> Self {
        self.ob_level = Some(ob_level);
        self
    }

    #[inline]
    pub fn os_level(mut self, os_level: i32) -> Self {
        self.os_level = Some(os_level);
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
    ) -> Result<StochasticDistanceOutput, StochasticDistanceError> {
        let input = StochasticDistanceInput::from_candles(
            candles,
            StochasticDistanceParams {
                lookback_length: self.lookback_length,
                length1: self.length1,
                length2: self.length2,
                ob_level: self.ob_level,
                os_level: self.os_level,
            },
        );
        stochastic_distance_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<StochasticDistanceOutput, StochasticDistanceError> {
        let input = StochasticDistanceInput::from_slice(
            data,
            StochasticDistanceParams {
                lookback_length: self.lookback_length,
                length1: self.length1,
                length2: self.length2,
                ob_level: self.ob_level,
                os_level: self.os_level,
            },
        );
        stochastic_distance_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<StochasticDistanceStream, StochasticDistanceError> {
        StochasticDistanceStream::try_new(StochasticDistanceParams {
            lookback_length: self.lookback_length,
            length1: self.length1,
            length2: self.length2,
            ob_level: self.ob_level,
            os_level: self.os_level,
        })
    }
}

#[derive(Debug, Error)]
pub enum StochasticDistanceError {
    #[error("stochastic_distance: Input data slice is empty.")]
    EmptyInputData,
    #[error("stochastic_distance: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "stochastic_distance: Invalid lookback_length: lookback_length = {lookback_length}, data length = {data_len}"
    )]
    InvalidLookbackLength {
        lookback_length: usize,
        data_len: usize,
    },
    #[error("stochastic_distance: Invalid length1: length1 = {length1}, data length = {data_len}")]
    InvalidLength1 { length1: usize, data_len: usize },
    #[error("stochastic_distance: Invalid length2: {length2}")]
    InvalidLength2 { length2: usize },
    #[error("stochastic_distance: Invalid ob_level: {ob_level}")]
    InvalidObLevel { ob_level: i32 },
    #[error("stochastic_distance: Invalid os_level: {os_level}")]
    InvalidOsLevel { os_level: i32 },
    #[error(
        "stochastic_distance: Invalid thresholds: os_level ({os_level}) must be less than ob_level ({ob_level})"
    )]
    InvalidThresholdOrder { ob_level: i32, os_level: i32 },
    #[error("stochastic_distance: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stochastic_distance: Output length mismatch: expected = {expected}, oscillator = {oscillator_got}, signal = {signal_got}")]
    OutputLengthMismatch {
        expected: usize,
        oscillator_got: usize,
        signal_got: usize,
    },
    #[error("stochastic_distance: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("stochastic_distance: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    lookback_length: usize,
    length1: usize,
    length2: usize,
    ob_level: f64,
    os_level: f64,
    alpha: f64,
    beta: f64,
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if data[i].is_finite() {
            break;
        }
        i += 1;
    }
    i.min(data.len())
}

#[inline(always)]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|v| v.is_finite()).count()
}

#[inline(always)]
fn warmup_period(params: ResolvedParams) -> usize {
    params.length1 + params.lookback_length - 1
}

#[inline]
fn resolve_params(
    params: &StochasticDistanceParams,
    data_len: Option<usize>,
) -> Result<ResolvedParams, StochasticDistanceError> {
    let lookback_length = params.lookback_length.unwrap_or(DEFAULT_LOOKBACK_LENGTH);
    let length1 = params.length1.unwrap_or(DEFAULT_LENGTH1);
    let length2 = params.length2.unwrap_or(DEFAULT_LENGTH2);
    let ob_level = params.ob_level.unwrap_or(DEFAULT_OB_LEVEL);
    let os_level = params.os_level.unwrap_or(DEFAULT_OS_LEVEL);

    if lookback_length == 0 {
        return Err(StochasticDistanceError::InvalidLookbackLength {
            lookback_length,
            data_len: data_len.unwrap_or(0),
        });
    }
    if length1 == 0 {
        return Err(StochasticDistanceError::InvalidLength1 {
            length1,
            data_len: data_len.unwrap_or(0),
        });
    }
    if length2 == 0 {
        return Err(StochasticDistanceError::InvalidLength2 { length2 });
    }
    if !(0..=100).contains(&ob_level) {
        return Err(StochasticDistanceError::InvalidObLevel { ob_level });
    }
    if !(-100..=0).contains(&os_level) {
        return Err(StochasticDistanceError::InvalidOsLevel { os_level });
    }
    if os_level >= ob_level {
        return Err(StochasticDistanceError::InvalidThresholdOrder { ob_level, os_level });
    }

    if let Some(data_len) = data_len {
        if lookback_length > data_len {
            return Err(StochasticDistanceError::InvalidLookbackLength {
                lookback_length,
                data_len,
            });
        }
        if length1 > data_len {
            return Err(StochasticDistanceError::InvalidLength1 { length1, data_len });
        }
    }

    let alpha = 2.0 / (length2 as f64 + 1.0);
    Ok(ResolvedParams {
        lookback_length,
        length1,
        length2,
        ob_level: ob_level as f64,
        os_level: os_level as f64,
        alpha,
        beta: 1.0 - alpha,
    })
}

#[derive(Debug, Clone)]
pub struct StochasticDistanceStream {
    params: ResolvedParams,
    close_ring: Vec<f64>,
    close_head: usize,
    close_count: usize,
    dist_index: usize,
    max_deque: VecDeque<(usize, f64)>,
    min_deque: VecDeque<(usize, f64)>,
    ema: f64,
    have_ema: bool,
    prev_sdo: f64,
}

impl StochasticDistanceStream {
    pub fn try_new(params: StochasticDistanceParams) -> Result<Self, StochasticDistanceError> {
        let params = resolve_params(&params, None)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            close_ring: vec![0.0; params.length1.max(1)],
            close_head: 0,
            close_count: 0,
            dist_index: 0,
            max_deque: VecDeque::with_capacity(params.lookback_length),
            min_deque: VecDeque::with_capacity(params.lookback_length),
            ema: 0.0,
            have_ema: false,
            prev_sdo: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        warmup_period(self.params)
    }

    #[inline]
    pub fn update(&mut self, close: f64) -> Option<(f64, f64)> {
        if !close.is_finite() {
            self.reset();
            return None;
        }

        let lag_close = if self.close_count >= self.params.length1 {
            Some(self.close_ring[self.close_head])
        } else {
            None
        };

        self.close_ring[self.close_head] = close;
        self.close_head += 1;
        if self.close_head == self.params.length1 {
            self.close_head = 0;
        }
        if self.close_count < self.params.length1 {
            self.close_count += 1;
        }

        let lag_close = lag_close?;
        let distance = (close - lag_close).abs();
        self.push_distance(distance, close, lag_close)
    }

    #[inline]
    fn push_distance(&mut self, distance: f64, close: f64, lag_close: f64) -> Option<(f64, f64)> {
        let idx = self.dist_index;
        self.dist_index += 1;

        while matches!(self.max_deque.back(), Some((_, v)) if *v <= distance) {
            self.max_deque.pop_back();
        }
        self.max_deque.push_back((idx, distance));
        while matches!(self.min_deque.back(), Some((_, v)) if *v >= distance) {
            self.min_deque.pop_back();
        }
        self.min_deque.push_back((idx, distance));

        let window = self.params.lookback_length;
        let cutoff = idx.saturating_sub(window.saturating_sub(1));
        while matches!(self.max_deque.front(), Some((front_idx, _)) if *front_idx < cutoff) {
            self.max_deque.pop_front();
        }
        while matches!(self.min_deque.front(), Some((front_idx, _)) if *front_idx < cutoff) {
            self.min_deque.pop_front();
        }

        if idx + 1 < window {
            return None;
        }

        let hh = self.max_deque.front().map(|(_, v)| *v).unwrap_or(distance);
        let ll = self.min_deque.front().map(|(_, v)| *v).unwrap_or(distance);
        let spread = hh - ll;
        let distance_sto = if spread.abs() > FLOAT_TOL {
            (distance - ll) / spread * 100.0
        } else {
            0.0
        };
        let distance_d = if close > lag_close + FLOAT_TOL {
            distance_sto
        } else if close + FLOAT_TOL < lag_close {
            -distance_sto
        } else {
            0.0
        };

        if self.have_ema {
            self.ema = self.params.alpha * distance_d + self.params.beta * self.ema;
        } else {
            self.ema = distance_d;
            self.have_ema = true;
        }

        let signal = if distance_d > self.ema
            || (self.prev_sdo < self.params.os_level && self.ema > self.params.os_level)
        {
            1.0
        } else if distance_d < self.ema
            || (self.prev_sdo > self.params.ob_level && self.ema < self.params.ob_level)
        {
            -1.0
        } else {
            0.0
        };
        self.prev_sdo = self.ema;

        Some((self.ema, signal))
    }
}

#[inline]
pub fn stochastic_distance(
    input: &StochasticDistanceInput,
) -> Result<StochasticDistanceOutput, StochasticDistanceError> {
    stochastic_distance_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn stochastic_distance_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
) {
    let mut stream = StochasticDistanceStream::new_resolved(params);
    for i in 0..data.len() {
        match stream.update(data[i]) {
            Some((oscillator, signal)) => {
                oscillator_out[i] = oscillator;
                signal_out[i] = signal;
            }
            None => {
                oscillator_out[i] = f64::NAN;
                signal_out[i] = f64::NAN;
            }
        }
    }
}

#[inline(always)]
fn stochastic_distance_prepare<'a>(
    input: &'a StochasticDistanceInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, ResolvedParams, Kernel), StochasticDistanceError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(StochasticDistanceError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(StochasticDistanceError::AllValuesNaN);
    }

    let params = resolve_params(&input.params, Some(data.len()))?;
    let valid = count_valid_values(data);
    let needed = params.lookback_length + params.length1;
    if valid < needed {
        return Err(StochasticDistanceError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok((data, first, valid, params, chosen))
}

pub fn stochastic_distance_with_kernel(
    input: &StochasticDistanceInput,
    kernel: Kernel,
) -> Result<StochasticDistanceOutput, StochasticDistanceError> {
    let (data, first, _valid, params, _chosen) = stochastic_distance_prepare(input, kernel)?;
    let mut oscillator = alloc_uninit_f64(data.len());
    let mut signal = alloc_uninit_f64(data.len());
    stochastic_distance_row_from_slice(data, params, &mut oscillator, &mut signal);
    Ok(StochasticDistanceOutput { oscillator, signal })
}

pub fn stochastic_distance_into_slices(
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
    input: &StochasticDistanceInput,
    kernel: Kernel,
) -> Result<(), StochasticDistanceError> {
    let expected = input.as_ref().len();
    if oscillator_out.len() != expected || signal_out.len() != expected {
        return Err(StochasticDistanceError::OutputLengthMismatch {
            expected,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
        });
    }
    let (data, _first, _valid, params, _chosen) = stochastic_distance_prepare(input, kernel)?;
    stochastic_distance_row_from_slice(data, params, oscillator_out, signal_out);
    Ok(())
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StochasticDistanceBatchRange {
    pub lookback_length: (usize, usize, usize),
    pub length1: (usize, usize, usize),
    pub length2: (usize, usize, usize),
    pub ob_level: (i32, i32, i32),
    pub os_level: (i32, i32, i32),
}

impl Default for StochasticDistanceBatchRange {
    fn default() -> Self {
        Self {
            lookback_length: (DEFAULT_LOOKBACK_LENGTH, DEFAULT_LOOKBACK_LENGTH, 0),
            length1: (DEFAULT_LENGTH1, DEFAULT_LENGTH1, 0),
            length2: (DEFAULT_LENGTH2, DEFAULT_LENGTH2, 0),
            ob_level: (DEFAULT_OB_LEVEL, DEFAULT_OB_LEVEL, 0),
            os_level: (DEFAULT_OS_LEVEL, DEFAULT_OS_LEVEL, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochasticDistanceBatchOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<StochasticDistanceParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct StochasticDistanceBatchBuilder {
    sweep: StochasticDistanceBatchRange,
    kernel: Kernel,
}

impl StochasticDistanceBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn lookback_length(mut self, start: usize, end: usize, step: usize) -> Self {
        self.sweep.lookback_length = (start, end, step);
        self
    }

    #[inline]
    pub fn length1(mut self, start: usize, end: usize, step: usize) -> Self {
        self.sweep.length1 = (start, end, step);
        self
    }

    #[inline]
    pub fn length2(mut self, start: usize, end: usize, step: usize) -> Self {
        self.sweep.length2 = (start, end, step);
        self
    }

    #[inline]
    pub fn ob_level(mut self, start: i32, end: i32, step: i32) -> Self {
        self.sweep.ob_level = (start, end, step);
        self
    }

    #[inline]
    pub fn os_level(mut self, start: i32, end: i32, step: i32) -> Self {
        self.sweep.os_level = (start, end, step);
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
    ) -> Result<StochasticDistanceBatchOutput, StochasticDistanceError> {
        stochastic_distance_batch_with_kernel(data, &self.sweep, self.kernel)
    }
}

#[inline]
fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, StochasticDistanceError> {
    if start > end {
        return Err(StochasticDistanceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if start == end {
        if step != 0 {
            return Err(StochasticDistanceError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(StochasticDistanceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(value);
        match value.checked_add(step) {
            Some(next) => value = next,
            None => break,
        }
    }
    if *out.last().unwrap_or(&start) != end {
        return Err(StochasticDistanceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
fn expand_axis_i32(start: i32, end: i32, step: i32) -> Result<Vec<i32>, StochasticDistanceError> {
    if start > end {
        return Err(StochasticDistanceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if start == end {
        if step != 0 {
            return Err(StochasticDistanceError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0 {
        return Err(StochasticDistanceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(value);
        match value.checked_add(step) {
            Some(next) => value = next,
            None => break,
        }
    }
    if *out.last().unwrap_or(&start) != end {
        return Err(StochasticDistanceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
fn expand_grid_stochastic_distance(
    range: &StochasticDistanceBatchRange,
) -> Result<Vec<StochasticDistanceParams>, StochasticDistanceError> {
    let lookbacks = expand_axis_usize(
        range.lookback_length.0,
        range.lookback_length.1,
        range.lookback_length.2,
    )?;
    let length1s = expand_axis_usize(range.length1.0, range.length1.1, range.length1.2)?;
    let length2s = expand_axis_usize(range.length2.0, range.length2.1, range.length2.2)?;
    let ob_levels = expand_axis_i32(range.ob_level.0, range.ob_level.1, range.ob_level.2)?;
    let os_levels = expand_axis_i32(range.os_level.0, range.os_level.1, range.os_level.2)?;

    let mut combos = Vec::with_capacity(
        lookbacks.len() * length1s.len() * length2s.len() * ob_levels.len() * os_levels.len(),
    );
    for &lookback_length in &lookbacks {
        for &length1 in &length1s {
            for &length2 in &length2s {
                for &ob_level in &ob_levels {
                    for &os_level in &os_levels {
                        let combo = StochasticDistanceParams {
                            lookback_length: Some(lookback_length),
                            length1: Some(length1),
                            length2: Some(length2),
                            ob_level: Some(ob_level),
                            os_level: Some(os_level),
                        };
                        let _ = resolve_params(&combo, None)?;
                        combos.push(combo);
                    }
                }
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn stochastic_distance_batch_with_kernel(
    data: &[f64],
    sweep: &StochasticDistanceBatchRange,
    kernel: Kernel,
) -> Result<StochasticDistanceBatchOutput, StochasticDistanceError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(StochasticDistanceError::InvalidKernelForBatch(other)),
    };
    stochastic_distance_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn stochastic_distance_batch_slice(
    data: &[f64],
    sweep: &StochasticDistanceBatchRange,
    kernel: Kernel,
) -> Result<StochasticDistanceBatchOutput, StochasticDistanceError> {
    stochastic_distance_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn stochastic_distance_batch_par_slice(
    data: &[f64],
    sweep: &StochasticDistanceBatchRange,
    kernel: Kernel,
) -> Result<StochasticDistanceBatchOutput, StochasticDistanceError> {
    stochastic_distance_batch_inner(data, sweep, kernel, true)
}

pub fn stochastic_distance_batch_inner(
    data: &[f64],
    sweep: &StochasticDistanceBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<StochasticDistanceBatchOutput, StochasticDistanceError> {
    let combos = expand_grid_stochastic_distance(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(StochasticDistanceError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(StochasticDistanceError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    let max_needed = combos
        .iter()
        .map(|combo| {
            combo.lookback_length.unwrap_or(DEFAULT_LOOKBACK_LENGTH)
                + combo.length1.unwrap_or(DEFAULT_LENGTH1)
        })
        .max()
        .unwrap_or(0);
    if valid < max_needed {
        return Err(StochasticDistanceError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let mut oscillator_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first
                + combo.length1.unwrap_or(DEFAULT_LENGTH1)
                + combo.lookback_length.unwrap_or(DEFAULT_LOOKBACK_LENGTH)
                - 1
        })
        .collect();
    init_matrix_prefixes(&mut oscillator_mu, cols, &warmups);
    init_matrix_prefixes(&mut signal_mu, cols, &warmups);

    let mut oscillator_guard = ManuallyDrop::new(oscillator_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let oscillator_out: &mut [f64] = unsafe {
        std::slice::from_raw_parts_mut(
            oscillator_guard.as_mut_ptr() as *mut f64,
            oscillator_guard.len(),
        )
    };
    let signal_out: &mut [f64] = unsafe {
        std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        oscillator_out
            .par_chunks_mut(cols)
            .zip(signal_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (osc_row, sig_row))| {
                let params = resolve_params(&combos[row], Some(cols)).unwrap();
                stochastic_distance_row_from_slice(data, params, osc_row, sig_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, (osc_row, sig_row)) in oscillator_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row], Some(cols)).unwrap();
            stochastic_distance_row_from_slice(data, params, osc_row, sig_row);
        }
    } else {
        for (row, (osc_row, sig_row)) in oscillator_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row], Some(cols)).unwrap();
            stochastic_distance_row_from_slice(data, params, osc_row, sig_row);
        }
    }

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

    Ok(StochasticDistanceBatchOutput {
        oscillator,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn stochastic_distance_batch_inner_into(
    data: &[f64],
    sweep: &StochasticDistanceBatchRange,
    _kernel: Kernel,
    parallel: bool,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<StochasticDistanceParams>, StochasticDistanceError> {
    let combos = expand_grid_stochastic_distance(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(StochasticDistanceError::EmptyInputData);
    }
    let total = rows
        .checked_mul(cols)
        .ok_or(StochasticDistanceError::OutputLengthMismatch {
            expected: usize::MAX,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
        })?;
    if oscillator_out.len() != total || signal_out.len() != total {
        return Err(StochasticDistanceError::OutputLengthMismatch {
            expected: total,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
        });
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(StochasticDistanceError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    let max_needed = combos
        .iter()
        .map(|combo| {
            combo.lookback_length.unwrap_or(DEFAULT_LOOKBACK_LENGTH)
                + combo.length1.unwrap_or(DEFAULT_LENGTH1)
        })
        .max()
        .unwrap_or(0);
    if valid < max_needed {
        return Err(StochasticDistanceError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        oscillator_out
            .par_chunks_mut(cols)
            .zip(signal_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (osc_row, sig_row))| {
                let params = resolve_params(&combos[row], Some(cols)).unwrap();
                stochastic_distance_row_from_slice(data, params, osc_row, sig_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, (osc_row, sig_row)) in oscillator_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row], Some(cols)).unwrap();
            stochastic_distance_row_from_slice(data, params, osc_row, sig_row);
        }
    } else {
        for (row, (osc_row, sig_row)) in oscillator_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row], Some(cols)).unwrap();
            stochastic_distance_row_from_slice(data, params, osc_row, sig_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochastic_distance")]
#[pyo3(signature = (data, lookback_length=DEFAULT_LOOKBACK_LENGTH, length1=DEFAULT_LENGTH1, length2=DEFAULT_LENGTH2, ob_level=DEFAULT_OB_LEVEL, os_level=DEFAULT_OS_LEVEL, kernel=None))]
pub fn stochastic_distance_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback_length: usize,
    length1: usize,
    length2: usize,
    ob_level: i32,
    os_level: i32,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = StochasticDistanceInput::from_slice(
        data,
        StochasticDistanceParams {
            lookback_length: Some(lookback_length),
            length1: Some(length1),
            length2: Some(length2),
            ob_level: Some(ob_level),
            os_level: Some(os_level),
        },
    );
    let output = py
        .allow_threads(|| stochastic_distance_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.oscillator.into_pyarray(py),
        output.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "StochasticDistanceStream")]
pub struct StochasticDistanceStreamPy {
    stream: StochasticDistanceStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StochasticDistanceStreamPy {
    #[new]
    #[pyo3(signature = (lookback_length=DEFAULT_LOOKBACK_LENGTH, length1=DEFAULT_LENGTH1, length2=DEFAULT_LENGTH2, ob_level=DEFAULT_OB_LEVEL, os_level=DEFAULT_OS_LEVEL))]
    fn new(
        lookback_length: usize,
        length1: usize,
        length2: usize,
        ob_level: i32,
        os_level: i32,
    ) -> PyResult<Self> {
        let stream = StochasticDistanceStream::try_new(StochasticDistanceParams {
            lookback_length: Some(lookback_length),
            length1: Some(length1),
            length2: Some(length2),
            ob_level: Some(ob_level),
            os_level: Some(os_level),
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
#[pyfunction(name = "stochastic_distance_batch")]
#[pyo3(signature = (data, lookback_length_range=(DEFAULT_LOOKBACK_LENGTH, DEFAULT_LOOKBACK_LENGTH, 0), length1_range=(DEFAULT_LENGTH1, DEFAULT_LENGTH1, 0), length2_range=(DEFAULT_LENGTH2, DEFAULT_LENGTH2, 0), ob_level_range=(DEFAULT_OB_LEVEL, DEFAULT_OB_LEVEL, 0), os_level_range=(DEFAULT_OS_LEVEL, DEFAULT_OS_LEVEL, 0), kernel=None))]
pub fn stochastic_distance_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback_length_range: (usize, usize, usize),
    length1_range: (usize, usize, usize),
    length2_range: (usize, usize, usize),
    ob_level_range: (i32, i32, i32),
    os_level_range: (i32, i32, i32),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = StochasticDistanceBatchRange {
        lookback_length: lookback_length_range,
        length1: length1_range,
        length2: length2_range,
        ob_level: ob_level_range,
        os_level: os_level_range,
    };
    let combos = expand_grid_stochastic_distance(&sweep)
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
            stochastic_distance_batch_inner_into(
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
        "lookback_lengths",
        combos
            .iter()
            .map(|combo| combo.lookback_length.unwrap_or(DEFAULT_LOOKBACK_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "length1s",
        combos
            .iter()
            .map(|combo| combo.length1.unwrap_or(DEFAULT_LENGTH1) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "length2s",
        combos
            .iter()
            .map(|combo| combo.length2.unwrap_or(DEFAULT_LENGTH2) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ob_levels",
        combos
            .iter()
            .map(|combo| combo.ob_level.unwrap_or(DEFAULT_OB_LEVEL))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "os_levels",
        combos
            .iter()
            .map(|combo| combo.os_level.unwrap_or(DEFAULT_OS_LEVEL))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_stochastic_distance_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stochastic_distance_py, module)?)?;
    module.add_function(wrap_pyfunction!(stochastic_distance_batch_py, module)?)?;
    module.add_class::<StochasticDistanceStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_distance_js")]
pub fn stochastic_distance_js(
    data: &[f64],
    lookback_length: usize,
    length1: usize,
    length2: usize,
    ob_level: i32,
    os_level: i32,
) -> Result<JsValue, JsValue> {
    let input = StochasticDistanceInput::from_slice(
        data,
        StochasticDistanceParams {
            lookback_length: Some(lookback_length),
            length1: Some(length1),
            length2: Some(length2),
            ob_level: Some(ob_level),
            os_level: Some(os_level),
        },
    );
    let out = stochastic_distance(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();

    let oscillator = js_sys::Float64Array::new_with_length(out.oscillator.len() as u32);
    oscillator.copy_from(&out.oscillator);
    js_sys::Reflect::set(&result, &JsValue::from_str("oscillator"), &oscillator)?;

    let signal = js_sys::Float64Array::new_with_length(out.signal.len() as u32);
    signal.copy_from(&out.signal);
    js_sys::Reflect::set(&result, &JsValue::from_str("signal"), &signal)?;

    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_distance_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_distance_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_distance_into(
    data_ptr: *const f64,
    oscillator_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    lookback_length: usize,
    length1: usize,
    length2: usize,
    ob_level: i32,
    os_level: i32,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || oscillator_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = StochasticDistanceInput::from_slice(
            data,
            StochasticDistanceParams {
                lookback_length: Some(lookback_length),
                length1: Some(length1),
                length2: Some(length2),
                ob_level: Some(ob_level),
                os_level: Some(os_level),
            },
        );
        let alias = data_ptr == oscillator_ptr || data_ptr == signal_ptr;
        if alias {
            let mut oscillator_tmp = vec![0.0; len];
            let mut signal_tmp = vec![0.0; len];
            stochastic_distance_into_slices(
                &mut oscillator_tmp,
                &mut signal_tmp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(oscillator_ptr, len).copy_from_slice(&oscillator_tmp);
            std::slice::from_raw_parts_mut(signal_ptr, len).copy_from_slice(&signal_tmp);
        } else {
            let oscillator_out = std::slice::from_raw_parts_mut(oscillator_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
            stochastic_distance_into_slices(oscillator_out, signal_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StochasticDistanceBatchConfig {
    pub lookback_length_range: (usize, usize, usize),
    pub length1_range: (usize, usize, usize),
    pub length2_range: (usize, usize, usize),
    pub ob_level_range: (i32, i32, i32),
    pub os_level_range: (i32, i32, i32),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StochasticDistanceBatchJsOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<StochasticDistanceParams>,
    pub lookback_lengths: Vec<usize>,
    pub length1s: Vec<usize>,
    pub length2s: Vec<usize>,
    pub ob_levels: Vec<i32>,
    pub os_levels: Vec<i32>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_distance_batch_js")]
pub fn stochastic_distance_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: StochasticDistanceBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = StochasticDistanceBatchRange {
        lookback_length: config.lookback_length_range,
        length1: config.length1_range,
        length2: config.length2_range,
        ob_level: config.ob_level_range,
        os_level: config.os_level_range,
    };
    let output = stochastic_distance_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&StochasticDistanceBatchJsOutput {
        lookback_lengths: output
            .combos
            .iter()
            .map(|combo| combo.lookback_length.unwrap_or(DEFAULT_LOOKBACK_LENGTH))
            .collect(),
        length1s: output
            .combos
            .iter()
            .map(|combo| combo.length1.unwrap_or(DEFAULT_LENGTH1))
            .collect(),
        length2s: output
            .combos
            .iter()
            .map(|combo| combo.length2.unwrap_or(DEFAULT_LENGTH2))
            .collect(),
        ob_levels: output
            .combos
            .iter()
            .map(|combo| combo.ob_level.unwrap_or(DEFAULT_OB_LEVEL))
            .collect(),
        os_levels: output
            .combos
            .iter()
            .map(|combo| combo.os_level.unwrap_or(DEFAULT_OS_LEVEL))
            .collect(),
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
pub fn stochastic_distance_batch_into(
    data_ptr: *const f64,
    oscillator_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    lookback_length_start: usize,
    lookback_length_end: usize,
    lookback_length_step: usize,
    length1_start: usize,
    length1_end: usize,
    length1_step: usize,
    length2_start: usize,
    length2_end: usize,
    length2_step: usize,
    ob_level_start: i32,
    ob_level_end: i32,
    ob_level_step: i32,
    os_level_start: i32,
    os_level_end: i32,
    os_level_step: i32,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || oscillator_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let sweep = StochasticDistanceBatchRange {
        lookback_length: (
            lookback_length_start,
            lookback_length_end,
            lookback_length_step,
        ),
        length1: (length1_start, length1_end, length1_step),
        length2: (length2_start, length2_end, length2_step),
        ob_level: (ob_level_start, ob_level_end, ob_level_step),
        os_level: (os_level_start, os_level_end, os_level_step),
    };
    let combos =
        expand_grid_stochastic_distance(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let oscillator_out = std::slice::from_raw_parts_mut(oscillator_ptr, total);
        let signal_out = std::slice::from_raw_parts_mut(signal_ptr, total);
        stochastic_distance_batch_inner_into(
            data,
            &sweep,
            detect_best_kernel(),
            false,
            oscillator_out,
            signal_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_distance_output_into_js(
    data: &[f64],
    lookback_length: usize,
    length1: usize,
    length2: usize,
    ob_level: i32,
    os_level: i32,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        stochastic_distance_js(data, lookback_length, length1, length2, ob_level, os_level)?;
    crate::write_wasm_object_f64_outputs("stochastic_distance_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_distance_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = stochastic_distance_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "stochastic_distance_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_close(length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; length];
        let mut prev = 100.0;
        for (i, slot) in out.iter_mut().enumerate().skip(2) {
            let x = i as f64;
            let value = prev + x.sin() * 0.75 + (x * 0.11).cos() * 1.25 + x * 0.03;
            *slot = value;
            prev = value;
        }
        out
    }

    #[test]
    fn stochastic_distance_output_contract() {
        let data = sample_close(512);
        let input = StochasticDistanceInput::from_slice(
            &data,
            StochasticDistanceParams {
                lookback_length: Some(80),
                length1: Some(12),
                length2: Some(3),
                ob_level: Some(40),
                os_level: Some(-40),
            },
        );
        let out = stochastic_distance(&input).unwrap();

        assert_eq!(out.oscillator.len(), data.len());
        assert_eq!(out.signal.len(), data.len());
        let first_valid = out.oscillator.iter().position(|v| v.is_finite()).unwrap();
        assert!(first_valid >= 91);
        for &v in out.signal.iter().skip(first_valid + 16) {
            assert!(v.is_nan() || v == -1.0 || v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn stochastic_distance_rejects_invalid_parameters() {
        let data = sample_close(64);

        let err = stochastic_distance(&StochasticDistanceInput::from_slice(
            &data,
            StochasticDistanceParams {
                lookback_length: Some(0),
                ..StochasticDistanceParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            StochasticDistanceError::InvalidLookbackLength { .. }
        ));

        let err = stochastic_distance(&StochasticDistanceInput::from_slice(
            &data,
            StochasticDistanceParams {
                os_level: Some(10),
                ..StochasticDistanceParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            StochasticDistanceError::InvalidOsLevel { .. }
        ));
    }

    #[test]
    fn stochastic_distance_stream_matches_batch_with_reset() {
        let mut data = sample_close(256);
        data[120] = f64::NAN;

        let params = StochasticDistanceParams {
            lookback_length: Some(60),
            length1: Some(10),
            length2: Some(4),
            ob_level: Some(35),
            os_level: Some(-35),
        };
        let batch =
            stochastic_distance(&StochasticDistanceInput::from_slice(&data, params.clone()))
                .unwrap();
        let mut stream = StochasticDistanceStream::try_new(params).unwrap();

        let mut osc = Vec::with_capacity(data.len());
        let mut sig = Vec::with_capacity(data.len());
        for &value in &data {
            match stream.update(value) {
                Some((o, s)) => {
                    osc.push(o);
                    sig.push(s);
                }
                None => {
                    osc.push(f64::NAN);
                    sig.push(f64::NAN);
                }
            }
        }

        for i in 0..osc.len() {
            let a = osc[i];
            let b = batch.oscillator[i];
            assert!(
                a.is_nan() && b.is_nan() || (a - b).abs() <= 1e-12,
                "osc mismatch at {i}"
            );
            let sa = sig[i];
            let sb = batch.signal[i];
            assert!(
                sa.is_nan() && sb.is_nan() || (sa - sb).abs() <= 1e-12,
                "signal mismatch at {i}"
            );
        }
    }

    #[test]
    fn stochastic_distance_batch_single_param_matches_single() {
        let data = sample_close(192);
        let sweep = StochasticDistanceBatchRange {
            lookback_length: (50, 50, 0),
            length1: (8, 8, 0),
            length2: (4, 4, 0),
            ob_level: (40, 40, 0),
            os_level: (-40, -40, 0),
        };
        let batch = stochastic_distance_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        let direct = stochastic_distance(&StochasticDistanceInput::from_slice(
            &data,
            StochasticDistanceParams {
                lookback_length: Some(50),
                length1: Some(8),
                length2: Some(4),
                ob_level: Some(40),
                os_level: Some(-40),
            },
        ))
        .unwrap();

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for i in 0..data.len() {
            let a = batch.oscillator[i];
            let b = direct.oscillator[i];
            assert!(
                a.is_nan() && b.is_nan() || (a - b).abs() <= 1e-12,
                "osc mismatch at {i}"
            );
            let sa = batch.signal[i];
            let sb = direct.signal[i];
            assert!(
                sa.is_nan() && sb.is_nan() || (sa - sb).abs() <= 1e-12,
                "signal mismatch at {i}"
            );
        }
    }
}
