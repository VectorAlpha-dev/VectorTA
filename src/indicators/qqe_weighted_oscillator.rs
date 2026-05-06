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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_LENGTH: usize = 14;
const DEFAULT_FACTOR: f64 = 4.236;
const DEFAULT_SMOOTH: usize = 5;
const DEFAULT_WEIGHT: f64 = 2.0;

impl<'a> AsRef<[f64]> for QqeWeightedOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            QqeWeightedOscillatorData::Slice(slice) => slice,
            QqeWeightedOscillatorData::Candles { candles, source } => {
                qqe_weighted_source(candles, source)
            }
        }
    }
}

#[inline(always)]
fn qqe_weighted_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => candles.open.as_slice(),
        "high" => candles.high.as_slice(),
        "low" => candles.low.as_slice(),
        "close" => candles.close.as_slice(),
        "volume" => candles.volume.as_slice(),
        "hl2" => candles.hl2.as_slice(),
        "hlc3" => candles.hlc3.as_slice(),
        "ohlc4" => candles.ohlc4.as_slice(),
        "hlcc4" | "hlcc" => candles.hlcc4.as_slice(),
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum QqeWeightedOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct QqeWeightedOscillatorOutput {
    pub rsi: Vec<f64>,
    pub trailing_stop: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct QqeWeightedOscillatorParams {
    pub length: Option<usize>,
    pub factor: Option<f64>,
    pub smooth: Option<usize>,
    pub weight: Option<f64>,
}

impl Default for QqeWeightedOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            factor: Some(DEFAULT_FACTOR),
            smooth: Some(DEFAULT_SMOOTH),
            weight: Some(DEFAULT_WEIGHT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QqeWeightedOscillatorInput<'a> {
    pub data: QqeWeightedOscillatorData<'a>,
    pub params: QqeWeightedOscillatorParams,
}

impl<'a> QqeWeightedOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: QqeWeightedOscillatorParams,
    ) -> Self {
        Self {
            data: QqeWeightedOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: QqeWeightedOscillatorParams) -> Self {
        Self {
            data: QqeWeightedOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", QqeWeightedOscillatorParams::default())
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline(always)]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(DEFAULT_FACTOR)
    }

    #[inline(always)]
    pub fn get_smooth(&self) -> usize {
        self.params.smooth.unwrap_or(DEFAULT_SMOOTH)
    }

    #[inline(always)]
    pub fn get_weight(&self) -> f64 {
        self.params.weight.unwrap_or(DEFAULT_WEIGHT)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct QqeWeightedOscillatorBuilder {
    length: Option<usize>,
    factor: Option<f64>,
    smooth: Option<usize>,
    weight: Option<f64>,
    kernel: Kernel,
}

impl Default for QqeWeightedOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            factor: None,
            smooth: None,
            weight: None,
            kernel: Kernel::Auto,
        }
    }
}

impl QqeWeightedOscillatorBuilder {
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
    pub fn factor(mut self, value: f64) -> Self {
        self.factor = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth(mut self, value: usize) -> Self {
        self.smooth = Some(value);
        self
    }

    #[inline(always)]
    pub fn weight(mut self, value: f64) -> Self {
        self.weight = Some(value);
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
    ) -> Result<QqeWeightedOscillatorOutput, QqeWeightedOscillatorError> {
        let input = QqeWeightedOscillatorInput::from_candles(
            candles,
            "close",
            QqeWeightedOscillatorParams {
                length: self.length,
                factor: self.factor,
                smooth: self.smooth,
                weight: self.weight,
            },
        );
        qqe_weighted_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<QqeWeightedOscillatorOutput, QqeWeightedOscillatorError> {
        let input = QqeWeightedOscillatorInput::from_slice(
            data,
            QqeWeightedOscillatorParams {
                length: self.length,
                factor: self.factor,
                smooth: self.smooth,
                weight: self.weight,
            },
        );
        qqe_weighted_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<QqeWeightedOscillatorStream, QqeWeightedOscillatorError> {
        QqeWeightedOscillatorStream::try_new(QqeWeightedOscillatorParams {
            length: self.length,
            factor: self.factor,
            smooth: self.smooth,
            weight: self.weight,
        })
    }
}

#[derive(Debug, Error)]
pub enum QqeWeightedOscillatorError {
    #[error("qqe_weighted_oscillator: input data slice is empty")]
    EmptyInputData,
    #[error("qqe_weighted_oscillator: all values are NaN")]
    AllValuesNaN,
    #[error(
        "qqe_weighted_oscillator: invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "qqe_weighted_oscillator: invalid smooth: smooth = {smooth}, data length = {data_len}"
    )]
    InvalidSmooth { smooth: usize, data_len: usize },
    #[error("qqe_weighted_oscillator: invalid factor: {factor}")]
    InvalidFactor { factor: f64 },
    #[error("qqe_weighted_oscillator: invalid weight: {weight}")]
    InvalidWeight { weight: f64 },
    #[error("qqe_weighted_oscillator: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("qqe_weighted_oscillator: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("qqe_weighted_oscillator: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("qqe_weighted_oscillator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct PreparedInput<'a> {
    data: &'a [f64],
    len: usize,
    length: usize,
    factor: f64,
    smooth: usize,
    weight: f64,
    first: usize,
    warmup: usize,
    clean: bool,
}

#[derive(Clone, Debug)]
struct RmaState {
    period: usize,
    count: usize,
    sum: f64,
    value: Option<f64>,
}

impl RmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            sum: 0.0,
            value: None,
        }
    }

    #[inline(always)]
    fn update(&mut self, input: f64) -> Option<f64> {
        if !input.is_finite() {
            return None;
        }
        if let Some(prev) = self.value {
            let next = (prev * (self.period as f64 - 1.0) + input) / self.period as f64;
            self.value = Some(next);
            return Some(next);
        }
        self.count += 1;
        self.sum += input;
        if self.count == self.period {
            let seeded = self.sum / self.period as f64;
            self.value = Some(seeded);
            Some(seeded)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct EmaState {
    alpha: f64,
    value: Option<f64>,
}

impl EmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            value: None,
        }
    }

    #[inline(always)]
    fn update(&mut self, input: f64) -> Option<f64> {
        if !input.is_finite() {
            return None;
        }
        let next = match self.value {
            Some(prev) => self.alpha * input + (1.0 - self.alpha) * prev,
            None => input,
        };
        self.value = Some(next);
        Some(next)
    }
}

#[inline]
pub fn qqe_weighted_oscillator(
    input: &QqeWeightedOscillatorInput<'_>,
) -> Result<QqeWeightedOscillatorOutput, QqeWeightedOscillatorError> {
    qqe_weighted_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn qqe_weighted_oscillator_with_kernel(
    input: &QqeWeightedOscillatorInput<'_>,
    kernel: Kernel,
) -> Result<QqeWeightedOscillatorOutput, QqeWeightedOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    let mut rsi = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut trailing_stop = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    compute_into_slices(&prepared, &mut rsi, &mut trailing_stop, true)?;
    Ok(QqeWeightedOscillatorOutput { rsi, trailing_stop })
}

#[inline]
pub fn qqe_weighted_oscillator_into(
    input: &QqeWeightedOscillatorInput<'_>,
    rsi: &mut [f64],
    trailing_stop: &mut [f64],
) -> Result<(), QqeWeightedOscillatorError> {
    qqe_weighted_oscillator_into_slices(input, Kernel::Auto, rsi, trailing_stop)
}

#[inline]
pub fn qqe_weighted_oscillator_into_slices(
    input: &QqeWeightedOscillatorInput<'_>,
    kernel: Kernel,
    rsi: &mut [f64],
    trailing_stop: &mut [f64],
) -> Result<(), QqeWeightedOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    if rsi.len() != prepared.len || trailing_stop.len() != prepared.len {
        return Err(QqeWeightedOscillatorError::OutputLengthMismatch {
            expected: prepared.len,
            got: core::cmp::min(rsi.len(), trailing_stop.len()),
        });
    }
    compute_into_slices(&prepared, rsi, trailing_stop, false)
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a QqeWeightedOscillatorInput<'_>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, QqeWeightedOscillatorError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(QqeWeightedOscillatorError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(QqeWeightedOscillatorError::AllValuesNaN)?;

    let length = input.get_length();
    let factor = input.get_factor();
    let smooth = input.get_smooth();
    let weight = input.get_weight();

    if length == 0 || length > len {
        return Err(QqeWeightedOscillatorError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if smooth == 0 {
        return Err(QqeWeightedOscillatorError::InvalidSmooth {
            smooth,
            data_len: len,
        });
    }
    if !factor.is_finite() || factor < 0.0 {
        return Err(QqeWeightedOscillatorError::InvalidFactor { factor });
    }
    if !weight.is_finite() {
        return Err(QqeWeightedOscillatorError::InvalidWeight { weight });
    }

    let mut valid = 0usize;
    let mut clean = true;
    for value in &data[first..] {
        if value.is_finite() {
            valid += 1;
        } else {
            clean = false;
        }
    }
    let needed = length + 1;
    if valid < needed {
        return Err(QqeWeightedOscillatorError::NotEnoughValidData { needed, valid });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        value => value,
    };

    Ok(PreparedInput {
        data,
        len,
        length,
        factor,
        smooth,
        weight,
        first,
        warmup: first + length,
        clean,
    })
}

#[inline(always)]
fn compute_into_slices(
    prepared: &PreparedInput<'_>,
    dst_rsi: &mut [f64],
    dst_trailing_stop: &mut [f64],
    allow_prefix_init: bool,
) -> Result<(), QqeWeightedOscillatorError> {
    if dst_rsi.len() != prepared.len || dst_trailing_stop.len() != prepared.len {
        return Err(QqeWeightedOscillatorError::OutputLengthMismatch {
            expected: prepared.len,
            got: core::cmp::min(dst_rsi.len(), dst_trailing_stop.len()),
        });
    }

    let prefix_init = allow_prefix_init && prepared.clean;
    if prefix_init {
        let prefix = prepared.warmup.min(prepared.len);
        dst_rsi[..prefix].fill(f64::NAN);
        dst_trailing_stop[..prefix].fill(f64::NAN);
    } else {
        dst_rsi.fill(f64::NAN);
        dst_trailing_stop.fill(f64::NAN);
    }

    let mut num_state = RmaState::new(prepared.length);
    let mut den_state = RmaState::new(prepared.length);
    let mut ratio_state = EmaState::new(prepared.smooth);
    let mut diff_state = RmaState::new(prepared.length);

    let mut prev_src = Some(prepared.data[prepared.first]);
    let mut prev_rsi: Option<f64> = None;
    let mut prev_ts: Option<f64> = None;

    for i in (prepared.first + 1)..prepared.len {
        let current = prepared.data[i];
        if !current.is_finite() {
            prev_src = None;
            continue;
        }
        let Some(prev_price) = prev_src else {
            prev_src = Some(current);
            continue;
        };

        let delta = current - prev_price;
        let w = match (prev_rsi, prev_ts) {
            (Some(rsi_prev), Some(ts_prev)) if delta * (rsi_prev - ts_prev) > 0.0 => {
                prepared.weight
            }
            _ => 1.0,
        };
        let weighted_delta = delta * w;

        let num = num_state.update(weighted_delta);
        let den = den_state.update(weighted_delta.abs());

        if let (Some(num), Some(den)) = (num, den) {
            if den != 0.0 {
                if let Some(smoothed_ratio) = ratio_state.update(num / den) {
                    let rsi = 50.0 * smoothed_ratio + 50.0;
                    dst_rsi[i] = rsi;

                    let diff = match prev_rsi {
                        Some(last_rsi) => diff_state.update((rsi - last_rsi).abs()),
                        None => None,
                    };

                    let trailing_stop = if let Some(diff) = diff {
                        let crossover = matches!(
                            (prev_rsi, prev_ts),
                            (Some(rsi_prev), Some(ts_prev)) if rsi > ts_prev && rsi_prev <= ts_prev
                        );
                        let crossunder = matches!(
                            (prev_rsi, prev_ts),
                            (Some(rsi_prev), Some(ts_prev)) if rsi < ts_prev && rsi_prev >= ts_prev
                        );
                        if crossover {
                            rsi - diff * prepared.factor
                        } else if crossunder {
                            rsi + diff * prepared.factor
                        } else if let Some(last_ts) = prev_ts {
                            if rsi > last_ts {
                                (rsi - diff * prepared.factor).max(last_ts)
                            } else {
                                (rsi + diff * prepared.factor).min(last_ts)
                            }
                        } else {
                            rsi
                        }
                    } else {
                        rsi
                    };

                    dst_trailing_stop[i] = trailing_stop;
                    prev_rsi = Some(rsi);
                    prev_ts = Some(trailing_stop);
                }
            } else if prefix_init {
                dst_rsi[i] = f64::NAN;
                dst_trailing_stop[i] = f64::NAN;
            }
        }

        prev_src = Some(current);
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct QqeWeightedOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub factor: (f64, f64, f64),
    pub smooth: (usize, usize, usize),
    pub weight: (f64, f64, f64),
}

impl Default for QqeWeightedOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            factor: (DEFAULT_FACTOR, DEFAULT_FACTOR, 0.0),
            smooth: (DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0),
            weight: (DEFAULT_WEIGHT, DEFAULT_WEIGHT, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QqeWeightedOscillatorBatchOutput {
    pub rsi: Vec<f64>,
    pub trailing_stop: Vec<f64>,
    pub combos: Vec<QqeWeightedOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct QqeWeightedOscillatorBatchBuilder {
    range: QqeWeightedOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for QqeWeightedOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: QqeWeightedOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl QqeWeightedOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: QqeWeightedOscillatorBatchRange) -> Self {
        self.range = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<QqeWeightedOscillatorBatchOutput, QqeWeightedOscillatorError> {
        qqe_weighted_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<QqeWeightedOscillatorBatchOutput, QqeWeightedOscillatorError> {
        self.apply_slice(candles.close.as_slice())
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, QqeWeightedOscillatorError> {
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
        return Err(QqeWeightedOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, QqeWeightedOscillatorError> {
    let eps = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(QqeWeightedOscillatorError::InvalidRange {
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
        return Err(QqeWeightedOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid(
    range: &QqeWeightedOscillatorBatchRange,
) -> Result<Vec<QqeWeightedOscillatorParams>, QqeWeightedOscillatorError> {
    let lengths = axis_usize(range.length)?;
    let factors = axis_f64(range.factor)?;
    let smooths = axis_usize(range.smooth)?;
    let weights = axis_f64(range.weight)?;
    let total = lengths
        .len()
        .checked_mul(factors.len())
        .and_then(|value| value.checked_mul(smooths.len()))
        .and_then(|value| value.checked_mul(weights.len()))
        .ok_or_else(|| QqeWeightedOscillatorError::InvalidRange {
            start: range.length.0.to_string(),
            end: range.length.1.to_string(),
            step: range.length.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &factor in &factors {
            for &smooth in &smooths {
                for &weight in &weights {
                    out.push(QqeWeightedOscillatorParams {
                        length: Some(length),
                        factor: Some(factor),
                        smooth: Some(smooth),
                        weight: Some(weight),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn qqe_weighted_oscillator_batch_with_kernel(
    data: &[f64],
    range: &QqeWeightedOscillatorBatchRange,
    kernel: Kernel,
) -> Result<QqeWeightedOscillatorBatchOutput, QqeWeightedOscillatorError> {
    if data.is_empty() {
        return Err(QqeWeightedOscillatorError::EmptyInputData);
    }
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        value if value.is_batch() => value,
        _ => return Err(QqeWeightedOscillatorError::InvalidKernelForBatch(kernel)),
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = data.len();

    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(QqeWeightedOscillatorError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.length.unwrap_or(DEFAULT_LENGTH))
        .collect();

    let mut rsi_mu = make_uninit_matrix(rows, cols);
    let mut ts_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut rsi_mu, cols, &warmups);
    init_matrix_prefixes(&mut ts_mu, cols, &warmups);

    let mut rsi_guard = ManuallyDrop::new(rsi_mu);
    let mut ts_guard = ManuallyDrop::new(ts_mu);
    let rsi_all = unsafe { mu_slice_as_f64_slice_mut(&mut rsi_guard) };
    let ts_all = unsafe { mu_slice_as_f64_slice_mut(&mut ts_guard) };

    let run_row = |row: usize,
                   rsi_row: &mut [f64],
                   ts_row: &mut [f64]|
     -> Result<(), QqeWeightedOscillatorError> {
        let input = QqeWeightedOscillatorInput::from_slice(data, combos[row].clone());
        qqe_weighted_oscillator_into_slices(&input, single_kernel, rsi_row, ts_row)
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        rsi_all
            .par_chunks_mut(cols)
            .zip(ts_all.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(|(row, (rsi_row, ts_row))| run_row(row, rsi_row, ts_row))?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            run_row(row, &mut rsi_all[start..end], &mut ts_all[start..end])?;
        }
    }

    Ok(QqeWeightedOscillatorBatchOutput {
        rsi: unsafe { vec_f64_from_mu_guard(rsi_guard) },
        trailing_stop: unsafe { vec_f64_from_mu_guard(ts_guard) },
        combos,
        rows,
        cols,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QqeWeightedOscillatorStreamOutput {
    pub rsi: f64,
    pub trailing_stop: f64,
}

#[derive(Debug, Clone)]
pub struct QqeWeightedOscillatorStream {
    factor: f64,
    weight: f64,
    prev_src: Option<f64>,
    prev_rsi: Option<f64>,
    prev_ts: Option<f64>,
    num_state: RmaState,
    den_state: RmaState,
    ratio_state: EmaState,
    diff_state: RmaState,
}

impl QqeWeightedOscillatorStream {
    pub fn try_new(
        params: QqeWeightedOscillatorParams,
    ) -> Result<Self, QqeWeightedOscillatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let factor = params.factor.unwrap_or(DEFAULT_FACTOR);
        let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
        let weight = params.weight.unwrap_or(DEFAULT_WEIGHT);

        if length == 0 {
            return Err(QqeWeightedOscillatorError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        if smooth == 0 {
            return Err(QqeWeightedOscillatorError::InvalidSmooth {
                smooth,
                data_len: 0,
            });
        }
        if !factor.is_finite() || factor < 0.0 {
            return Err(QqeWeightedOscillatorError::InvalidFactor { factor });
        }
        if !weight.is_finite() {
            return Err(QqeWeightedOscillatorError::InvalidWeight { weight });
        }

        Ok(Self {
            factor,
            weight,
            prev_src: None,
            prev_rsi: None,
            prev_ts: None,
            num_state: RmaState::new(length),
            den_state: RmaState::new(length),
            ratio_state: EmaState::new(smooth),
            diff_state: RmaState::new(length),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<QqeWeightedOscillatorStreamOutput> {
        if !value.is_finite() {
            self.prev_src = None;
            return None;
        }
        let Some(prev_src) = self.prev_src else {
            self.prev_src = Some(value);
            return None;
        };

        let delta = value - prev_src;
        let w = match (self.prev_rsi, self.prev_ts) {
            (Some(rsi_prev), Some(ts_prev)) if delta * (rsi_prev - ts_prev) > 0.0 => self.weight,
            _ => 1.0,
        };
        let weighted_delta = delta * w;
        let num = self.num_state.update(weighted_delta);
        let den = self.den_state.update(weighted_delta.abs());
        self.prev_src = Some(value);

        let (Some(num), Some(den)) = (num, den) else {
            return None;
        };
        if den == 0.0 {
            return None;
        }
        let smoothed_ratio = self.ratio_state.update(num / den)?;
        let rsi = 50.0 * smoothed_ratio + 50.0;
        let diff = match self.prev_rsi {
            Some(last_rsi) => self.diff_state.update((rsi - last_rsi).abs()),
            None => None,
        };

        let trailing_stop = if let Some(diff) = diff {
            let crossover = matches!(
                (self.prev_rsi, self.prev_ts),
                (Some(rsi_prev), Some(ts_prev)) if rsi > ts_prev && rsi_prev <= ts_prev
            );
            let crossunder = matches!(
                (self.prev_rsi, self.prev_ts),
                (Some(rsi_prev), Some(ts_prev)) if rsi < ts_prev && rsi_prev >= ts_prev
            );
            if crossover {
                rsi - diff * self.factor
            } else if crossunder {
                rsi + diff * self.factor
            } else if let Some(last_ts) = self.prev_ts {
                if rsi > last_ts {
                    (rsi - diff * self.factor).max(last_ts)
                } else {
                    (rsi + diff * self.factor).min(last_ts)
                }
            } else {
                rsi
            }
        } else {
            rsi
        };

        self.prev_rsi = Some(rsi);
        self.prev_ts = Some(trailing_stop);
        Some(QqeWeightedOscillatorStreamOutput { rsi, trailing_stop })
    }
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
#[pyfunction(name = "qqe_weighted_oscillator")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, factor=DEFAULT_FACTOR, smooth=DEFAULT_SMOOTH, weight=DEFAULT_WEIGHT, kernel=None))]
pub fn qqe_weighted_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    factor: f64,
    smooth: usize,
    weight: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = QqeWeightedOscillatorInput::from_slice(
        data,
        QqeWeightedOscillatorParams {
            length: Some(length),
            factor: Some(factor),
            smooth: Some(smooth),
            weight: Some(weight),
        },
    );
    let output = py
        .allow_threads(|| qqe_weighted_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("rsi", output.rsi.into_pyarray(py))?;
    dict.set_item("trailing_stop", output.trailing_stop.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "qqe_weighted_oscillator_batch")]
#[pyo3(signature = (data, length_range, factor_range, smooth_range, weight_range, kernel=None))]
pub fn qqe_weighted_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    factor_range: (f64, f64, f64),
    smooth_range: (usize, usize, usize),
    weight_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            qqe_weighted_oscillator_batch_with_kernel(
                data,
                &QqeWeightedOscillatorBatchRange {
                    length: length_range,
                    factor: factor_range,
                    smooth: smooth_range,
                    weight: weight_range,
                },
                kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let total = output.rows * output.cols;
    let arrays = [
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
    ];
    unsafe { arrays[0].as_slice_mut()? }.copy_from_slice(&output.rsi);
    unsafe { arrays[1].as_slice_mut()? }.copy_from_slice(&output.trailing_stop);

    let dict = PyDict::new(py);
    dict.set_item("rsi", arrays[0].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "trailing_stop",
        arrays[1].reshape((output.rows, output.cols))?,
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
        "factors",
        output
            .combos
            .iter()
            .map(|combo| combo.factor.unwrap_or(DEFAULT_FACTOR))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooths",
        output
            .combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "weights",
        output
            .combos
            .iter()
            .map(|combo| combo.weight.unwrap_or(DEFAULT_WEIGHT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "QqeWeightedOscillatorStream")]
pub struct QqeWeightedOscillatorStreamPy {
    stream: QqeWeightedOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl QqeWeightedOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, factor=DEFAULT_FACTOR, smooth=DEFAULT_SMOOTH, weight=DEFAULT_WEIGHT))]
    fn new(length: usize, factor: f64, smooth: usize, weight: f64) -> PyResult<Self> {
        let stream = QqeWeightedOscillatorStream::try_new(QqeWeightedOscillatorParams {
            length: Some(length),
            factor: Some(factor),
            smooth: Some(smooth),
            weight: Some(weight),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream
            .update(value)
            .map(|output| (output.rsi, output.trailing_stop))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct QqeWeightedOscillatorJsOutput {
    pub rsi: Vec<f64>,
    pub trailing_stop: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = qqe_weighted_oscillator_js)]
pub fn qqe_weighted_oscillator_js(
    data: &[f64],
    length: usize,
    factor: f64,
    smooth: usize,
    weight: f64,
) -> Result<JsValue, JsValue> {
    let input = QqeWeightedOscillatorInput::from_slice(
        data,
        QqeWeightedOscillatorParams {
            length: Some(length),
            factor: Some(factor),
            smooth: Some(smooth),
            weight: Some(weight),
        },
    );
    let output = qqe_weighted_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&QqeWeightedOscillatorJsOutput {
        rsi: output.rsi,
        trailing_stop: output.trailing_stop,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct QqeWeightedOscillatorBatchConfig {
    pub length_range: (usize, usize, usize),
    pub factor_range: (f64, f64, f64),
    pub smooth_range: (usize, usize, usize),
    pub weight_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct QqeWeightedOscillatorBatchJsOutput {
    pub rsi: Vec<f64>,
    pub trailing_stop: Vec<f64>,
    pub lengths: Vec<usize>,
    pub factors: Vec<f64>,
    pub smooths: Vec<usize>,
    pub weights: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = qqe_weighted_oscillator_batch)]
pub fn qqe_weighted_oscillator_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: QqeWeightedOscillatorBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = qqe_weighted_oscillator_batch_with_kernel(
        data,
        &QqeWeightedOscillatorBatchRange {
            length: cfg.length_range,
            factor: cfg.factor_range,
            smooth: cfg.smooth_range,
            weight: cfg.weight_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&QqeWeightedOscillatorBatchJsOutput {
        rsi: output.rsi,
        trailing_stop: output.trailing_stop,
        lengths: output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        factors: output
            .combos
            .iter()
            .map(|combo| combo.factor.unwrap_or(DEFAULT_FACTOR))
            .collect(),
        smooths: output
            .combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH))
            .collect(),
        weights: output
            .combos
            .iter()
            .map(|combo| combo.weight.unwrap_or(DEFAULT_WEIGHT))
            .collect(),
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn qqe_weighted_oscillator_output_into_js(
    data: &[f64],
    length: usize,
    factor: f64,
    smooth: usize,
    weight: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = qqe_weighted_oscillator_js(data, length, factor, smooth, weight)?;
    crate::write_wasm_object_f64_outputs("qqe_weighted_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn qqe_weighted_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = qqe_weighted_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "qqe_weighted_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![
            100.0, 101.0, 102.5, 101.8, 103.2, 104.1, 103.7, 105.4, 106.2, 105.8, 107.1, 108.0,
            107.6, 108.8, 109.4, 108.9, 110.3, 111.2, 110.7, 112.0, 112.6, 112.1, 113.4, 114.0,
            113.5, 114.7, 115.1, 114.8, 116.0, 116.4, 116.1, 117.0, 117.8, 117.2, 118.4, 119.1,
            118.6, 119.7, 120.2, 119.8,
        ]
    }

    #[test]
    fn qqe_weighted_oscillator_into_matches_single() {
        let data = sample_data();
        let input = QqeWeightedOscillatorInput::from_slice(
            &data,
            QqeWeightedOscillatorParams {
                length: Some(14),
                factor: Some(4.236),
                smooth: Some(5),
                weight: Some(2.0),
            },
        );
        let out = qqe_weighted_oscillator_with_kernel(&input, Kernel::Scalar).expect("single");
        let mut rsi = vec![0.0; data.len()];
        let mut ts = vec![0.0; data.len()];
        qqe_weighted_oscillator_into_slices(&input, Kernel::Scalar, &mut rsi, &mut ts)
            .expect("into");

        for i in 0..data.len() {
            if out.rsi[i].is_nan() {
                assert!(rsi[i].is_nan());
            } else {
                assert!((out.rsi[i] - rsi[i]).abs() <= 1e-12);
            }
            if out.trailing_stop[i].is_nan() {
                assert!(ts[i].is_nan());
            } else {
                assert!((out.trailing_stop[i] - ts[i]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn qqe_weighted_oscillator_stream_matches_batch_points() {
        let data = sample_data();
        let params = QqeWeightedOscillatorParams {
            length: Some(14),
            factor: Some(4.236),
            smooth: Some(5),
            weight: Some(2.0),
        };
        let input = QqeWeightedOscillatorInput::from_slice(&data, params.clone());
        let batch = qqe_weighted_oscillator(&input).expect("batch");
        let mut stream = QqeWeightedOscillatorStream::try_new(params).expect("stream");

        for i in 0..data.len() {
            let point = stream.update(data[i]);
            if let Some(point) = point {
                assert!((point.rsi - batch.rsi[i]).abs() <= 1e-12);
                assert!((point.trailing_stop - batch.trailing_stop[i]).abs() <= 1e-12);
            } else {
                assert!(batch.rsi[i].is_nan());
                assert!(batch.trailing_stop[i].is_nan());
            }
        }
    }

    #[test]
    fn qqe_weighted_oscillator_batch_first_row_matches_single() {
        let data = sample_data();
        let batch = qqe_weighted_oscillator_batch_with_kernel(
            &data,
            &QqeWeightedOscillatorBatchRange {
                length: (14, 16, 2),
                factor: (4.236, 4.736, 0.5),
                smooth: (5, 5, 0),
                weight: (2.0, 2.0, 0.0),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, data.len());

        let single = qqe_weighted_oscillator(&QqeWeightedOscillatorInput::from_slice(
            &data,
            QqeWeightedOscillatorParams {
                length: Some(14),
                factor: Some(4.236),
                smooth: Some(5),
                weight: Some(2.0),
            },
        ))
        .expect("single");

        for i in 0..data.len() {
            let batch_rsi = batch.rsi[i];
            let batch_ts = batch.trailing_stop[i];
            if single.rsi[i].is_nan() {
                assert!(batch_rsi.is_nan());
            } else {
                assert!((single.rsi[i] - batch_rsi).abs() <= 1e-12);
            }
            if single.trailing_stop[i].is_nan() {
                assert!(batch_ts.is_nan());
            } else {
                assert!((single.trailing_stop[i] - batch_ts).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn qqe_weighted_oscillator_rejects_invalid_inputs() {
        let data = sample_data();
        let err = qqe_weighted_oscillator(&QqeWeightedOscillatorInput::from_slice(
            &data,
            QqeWeightedOscillatorParams {
                length: Some(0),
                factor: Some(4.236),
                smooth: Some(5),
                weight: Some(2.0),
            },
        ))
        .expect_err("invalid length");
        assert!(matches!(
            err,
            QqeWeightedOscillatorError::InvalidLength { .. }
        ));

        let err = qqe_weighted_oscillator(&QqeWeightedOscillatorInput::from_slice(
            &data,
            QqeWeightedOscillatorParams {
                length: Some(14),
                factor: Some(-1.0),
                smooth: Some(5),
                weight: Some(2.0),
            },
        ))
        .expect_err("invalid factor");
        assert!(matches!(
            err,
            QqeWeightedOscillatorError::InvalidFactor { .. }
        ));

        let err = qqe_weighted_oscillator(&QqeWeightedOscillatorInput::from_slice(
            &data,
            QqeWeightedOscillatorParams {
                length: Some(14),
                factor: Some(4.236),
                smooth: Some(0),
                weight: Some(2.0),
            },
        ))
        .expect_err("invalid smooth");
        assert!(matches!(
            err,
            QqeWeightedOscillatorError::InvalidSmooth { .. }
        ));
    }
}
