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

const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_ALPHA: f64 = 0.1;
const MIN_ALPHA: f64 = 0.001;
const MAX_ALPHA: f64 = 1.0;

impl<'a> AsRef<[f64]> for AdaptiveBoundsRsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AdaptiveBoundsRsiData::Slice(slice) => slice,
            AdaptiveBoundsRsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AdaptiveBoundsRsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AdaptiveBoundsRsiOutput {
    pub rsi: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub lower_mid: Vec<f64>,
    pub mid: Vec<f64>,
    pub upper_mid: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub regime: Vec<f64>,
    pub regime_flip: Vec<f64>,
    pub lower_signal: Vec<f64>,
    pub upper_signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveBoundsRsiOutputField {
    Rsi,
    LowerBound,
    LowerMid,
    Mid,
    UpperMid,
    UpperBound,
    Regime,
    RegimeFlip,
    LowerSignal,
    UpperSignal,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdaptiveBoundsRsiParams {
    pub rsi_length: Option<usize>,
    pub alpha: Option<f64>,
}

impl Default for AdaptiveBoundsRsiParams {
    fn default() -> Self {
        Self {
            rsi_length: Some(DEFAULT_RSI_LENGTH),
            alpha: Some(DEFAULT_ALPHA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBoundsRsiInput<'a> {
    pub data: AdaptiveBoundsRsiData<'a>,
    pub params: AdaptiveBoundsRsiParams,
}

impl<'a> AdaptiveBoundsRsiInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: AdaptiveBoundsRsiParams,
    ) -> Self {
        Self {
            data: AdaptiveBoundsRsiData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AdaptiveBoundsRsiParams) -> Self {
        Self {
            data: AdaptiveBoundsRsiData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", AdaptiveBoundsRsiParams::default())
    }

    #[inline(always)]
    pub fn get_rsi_length(&self) -> usize {
        self.params.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH)
    }

    #[inline(always)]
    pub fn get_alpha(&self) -> f64 {
        self.params.alpha.unwrap_or(DEFAULT_ALPHA)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdaptiveBoundsRsiBuilder {
    rsi_length: Option<usize>,
    alpha: Option<f64>,
    kernel: Kernel,
}

impl Default for AdaptiveBoundsRsiBuilder {
    fn default() -> Self {
        Self {
            rsi_length: None,
            alpha: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdaptiveBoundsRsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn rsi_length(mut self, value: usize) -> Self {
        self.rsi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn alpha(mut self, value: f64) -> Self {
        self.alpha = Some(value);
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
    ) -> Result<AdaptiveBoundsRsiOutput, AdaptiveBoundsRsiError> {
        let input = AdaptiveBoundsRsiInput::from_candles(
            candles,
            "close",
            AdaptiveBoundsRsiParams {
                rsi_length: self.rsi_length,
                alpha: self.alpha,
            },
        );
        adaptive_bounds_rsi_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdaptiveBoundsRsiOutput, AdaptiveBoundsRsiError> {
        let input = AdaptiveBoundsRsiInput::from_slice(
            data,
            AdaptiveBoundsRsiParams {
                rsi_length: self.rsi_length,
                alpha: self.alpha,
            },
        );
        adaptive_bounds_rsi_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AdaptiveBoundsRsiStream, AdaptiveBoundsRsiError> {
        AdaptiveBoundsRsiStream::try_new(AdaptiveBoundsRsiParams {
            rsi_length: self.rsi_length,
            alpha: self.alpha,
        })
    }
}

#[derive(Debug, Error)]
pub enum AdaptiveBoundsRsiError {
    #[error("adaptive_bounds_rsi: input data slice is empty")]
    EmptyInputData,
    #[error("adaptive_bounds_rsi: all values are NaN")]
    AllValuesNaN,
    #[error("adaptive_bounds_rsi: invalid rsi_length: rsi_length = {rsi_length}, data length = {data_len}")]
    InvalidRsiLength { rsi_length: usize, data_len: usize },
    #[error("adaptive_bounds_rsi: invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error("adaptive_bounds_rsi: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("adaptive_bounds_rsi: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("adaptive_bounds_rsi: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("adaptive_bounds_rsi: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct PreparedInput<'a> {
    data: &'a [f64],
    len: usize,
    rsi_length: usize,
    alpha: f64,
    warmup: usize,
}

#[derive(Clone, Debug)]
struct RsiState {
    period: usize,
    inv_p: f64,
    beta: f64,
    prev_price: Option<f64>,
    seed_count: usize,
    sum_gain: f64,
    sum_loss: f64,
    avg_gain: f64,
    avg_loss: f64,
    seeded: bool,
}

impl RsiState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let inv_p = 1.0 / period as f64;
        Self {
            period,
            inv_p,
            beta: 1.0 - inv_p,
            prev_price: None,
            seed_count: 0,
            sum_gain: 0.0,
            sum_loss: 0.0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            seeded: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_price = None;
        self.seed_count = 0;
        self.sum_gain = 0.0;
        self.sum_loss = 0.0;
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.seeded = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let Some(prev) = self.prev_price else {
            self.prev_price = Some(value);
            return None;
        };
        let delta = value - prev;
        self.prev_price = Some(value);
        let gain = delta.max(0.0);
        let loss = (-delta).max(0.0);

        if !self.seeded {
            self.sum_gain += gain;
            self.sum_loss += loss;
            self.seed_count += 1;
            if self.seed_count == self.period {
                self.seeded = true;
                self.avg_gain = self.sum_gain * self.inv_p;
                self.avg_loss = self.sum_loss * self.inv_p;
                return Some(rsi_value(self.avg_gain, self.avg_loss));
            }
            return None;
        }

        self.avg_gain = self.avg_gain.mul_add(self.beta, self.inv_p * gain);
        self.avg_loss = self.avg_loss.mul_add(self.beta, self.inv_p * loss);
        Some(rsi_value(self.avg_gain, self.avg_loss))
    }
}

#[inline(always)]
fn rsi_value(avg_gain: f64, avg_loss: f64) -> f64 {
    let denom = avg_gain + avg_loss;
    if denom == 0.0 {
        50.0
    } else {
        100.0 * avg_gain / denom
    }
}

#[inline(always)]
fn pine_cross(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    (prev_a <= prev_b && curr_a > curr_b) || (prev_a >= prev_b && curr_a < curr_b)
}

#[inline(always)]
fn pine_crossover(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a <= prev_b && curr_a > curr_b
}

#[inline(always)]
fn pine_crossunder(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a >= prev_b && curr_a < curr_b
}

#[inline]
pub fn adaptive_bounds_rsi(
    input: &AdaptiveBoundsRsiInput<'_>,
) -> Result<AdaptiveBoundsRsiOutput, AdaptiveBoundsRsiError> {
    adaptive_bounds_rsi_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn adaptive_bounds_rsi_with_kernel(
    input: &AdaptiveBoundsRsiInput<'_>,
    kernel: Kernel,
) -> Result<AdaptiveBoundsRsiOutput, AdaptiveBoundsRsiError> {
    let prepared = prepare_input(input, kernel)?;
    let mut rsi = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut lower_bound = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut lower_mid = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut mid = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut upper_mid = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut upper_bound = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut regime = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut regime_flip = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut lower_signal = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    let mut upper_signal = alloc_with_nan_prefix(prepared.len, prepared.warmup.min(prepared.len));
    adaptive_bounds_rsi_into_slices(
        input,
        kernel,
        &mut rsi,
        &mut lower_bound,
        &mut lower_mid,
        &mut mid,
        &mut upper_mid,
        &mut upper_bound,
        &mut regime,
        &mut regime_flip,
        &mut lower_signal,
        &mut upper_signal,
    )?;
    Ok(AdaptiveBoundsRsiOutput {
        rsi,
        lower_bound,
        lower_mid,
        mid,
        upper_mid,
        upper_bound,
        regime,
        regime_flip,
        lower_signal,
        upper_signal,
    })
}

#[inline]
pub fn adaptive_bounds_rsi_into(
    input: &AdaptiveBoundsRsiInput<'_>,
    rsi: &mut [f64],
    lower_bound: &mut [f64],
    lower_mid: &mut [f64],
    mid: &mut [f64],
    upper_mid: &mut [f64],
    upper_bound: &mut [f64],
    regime: &mut [f64],
    regime_flip: &mut [f64],
    lower_signal: &mut [f64],
    upper_signal: &mut [f64],
) -> Result<(), AdaptiveBoundsRsiError> {
    adaptive_bounds_rsi_into_slices(
        input,
        Kernel::Auto,
        rsi,
        lower_bound,
        lower_mid,
        mid,
        upper_mid,
        upper_bound,
        regime,
        regime_flip,
        lower_signal,
        upper_signal,
    )
}

#[inline]
pub fn adaptive_bounds_rsi_into_slices(
    input: &AdaptiveBoundsRsiInput<'_>,
    kernel: Kernel,
    rsi: &mut [f64],
    lower_bound: &mut [f64],
    lower_mid: &mut [f64],
    mid: &mut [f64],
    upper_mid: &mut [f64],
    upper_bound: &mut [f64],
    regime: &mut [f64],
    regime_flip: &mut [f64],
    lower_signal: &mut [f64],
    upper_signal: &mut [f64],
) -> Result<(), AdaptiveBoundsRsiError> {
    let prepared = prepare_input(input, kernel)?;
    compute_into_slices(
        &prepared,
        rsi,
        lower_bound,
        lower_mid,
        mid,
        upper_mid,
        upper_bound,
        regime,
        regime_flip,
        lower_signal,
        upper_signal,
    )
}

#[inline]
fn prepare_input<'a>(
    input: &'a AdaptiveBoundsRsiInput<'_>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, AdaptiveBoundsRsiError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(AdaptiveBoundsRsiError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(AdaptiveBoundsRsiError::AllValuesNaN)?;

    let rsi_length = input.get_rsi_length();
    if rsi_length == 0 || rsi_length > len {
        return Err(AdaptiveBoundsRsiError::InvalidRsiLength {
            rsi_length,
            data_len: len,
        });
    }

    let alpha = input.get_alpha();
    if !alpha.is_finite() || !(MIN_ALPHA..=MAX_ALPHA).contains(&alpha) {
        return Err(AdaptiveBoundsRsiError::InvalidAlpha { alpha });
    }

    let valid = data[first..]
        .iter()
        .filter(|value| value.is_finite())
        .count();
    let needed = rsi_length + 1;
    if valid < needed {
        return Err(AdaptiveBoundsRsiError::NotEnoughValidData { needed, valid });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        value => value,
    };

    Ok(PreparedInput {
        data,
        len,
        rsi_length,
        alpha,
        warmup: first + rsi_length,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_into_slices(
    prepared: &PreparedInput<'_>,
    dst_rsi: &mut [f64],
    dst_lower_bound: &mut [f64],
    dst_lower_mid: &mut [f64],
    dst_mid: &mut [f64],
    dst_upper_mid: &mut [f64],
    dst_upper_bound: &mut [f64],
    dst_regime: &mut [f64],
    dst_regime_flip: &mut [f64],
    dst_lower_signal: &mut [f64],
    dst_upper_signal: &mut [f64],
) -> Result<(), AdaptiveBoundsRsiError> {
    let expected = prepared.len;
    let lens = [
        dst_rsi.len(),
        dst_lower_bound.len(),
        dst_lower_mid.len(),
        dst_mid.len(),
        dst_upper_mid.len(),
        dst_upper_bound.len(),
        dst_regime.len(),
        dst_regime_flip.len(),
        dst_lower_signal.len(),
        dst_upper_signal.len(),
    ];
    if lens.iter().any(|&len| len != expected) {
        let got = lens.into_iter().min().unwrap_or(0);
        return Err(AdaptiveBoundsRsiError::OutputLengthMismatch { expected, got });
    }

    dst_rsi.fill(f64::NAN);
    dst_lower_bound.fill(f64::NAN);
    dst_lower_mid.fill(f64::NAN);
    dst_mid.fill(f64::NAN);
    dst_upper_mid.fill(f64::NAN);
    dst_upper_bound.fill(f64::NAN);
    dst_regime.fill(f64::NAN);
    dst_regime_flip.fill(f64::NAN);
    dst_lower_signal.fill(f64::NAN);
    dst_upper_signal.fill(f64::NAN);

    let mut rsi_state = RsiState::new(prepared.rsi_length);
    let mut c1 = 20.0;
    let mut c2 = 40.0;
    let mut c3 = 50.0;
    let mut c4 = 60.0;
    let mut c5 = 80.0;

    let mut prev_rsi: Option<f64> = None;
    let mut prev_c1: Option<f64> = None;
    let mut prev_c5: Option<f64> = None;
    let mut prev_regime: Option<i32> = None;
    let mut can_show_lower = true;
    let mut can_show_upper = true;

    for i in 0..prepared.len {
        let value = prepared.data[i];
        if !value.is_finite() {
            rsi_state.reset();
            prev_rsi = None;
            prev_c1 = None;
            prev_c5 = None;
            prev_regime = None;
            continue;
        }

        let Some(rsi) = rsi_state.update(value) else {
            continue;
        };

        let d1 = (rsi - c1).abs();
        let d2 = (rsi - c2).abs();
        let d3 = (rsi - c3).abs();
        let d4 = (rsi - c4).abs();
        let d5 = (rsi - c5).abs();
        let min_dist = d1.min(d2.min(d3.min(d4.min(d5))));
        if min_dist == d1 {
            c1 += (rsi - c1) * prepared.alpha;
        } else if min_dist == d2 {
            c2 += (rsi - c2) * prepared.alpha;
        } else if min_dist == d3 {
            c3 += (rsi - c3) * prepared.alpha;
        } else if min_dist == d4 {
            c4 += (rsi - c4) * prepared.alpha;
        } else {
            c5 += (rsi - c5) * prepared.alpha;
        }

        let regime = if rsi <= c1 {
            -2
        } else if rsi <= c2 {
            -1
        } else if rsi <= c3 {
            0
        } else if rsi <= c4 {
            1
        } else {
            2
        };

        let crossed_mid = match prev_rsi {
            Some(prev) => pine_cross(prev, 50.0, rsi, 50.0),
            None => false,
        };
        if crossed_mid {
            can_show_lower = true;
            can_show_upper = true;
        }

        let lower_signal = match (prev_rsi, prev_c1) {
            (Some(prev_value), Some(prev_bound)) => {
                pine_crossunder(prev_value, prev_bound, rsi, c1) && can_show_lower
            }
            _ => false,
        };
        let upper_signal = match (prev_rsi, prev_c5) {
            (Some(prev_value), Some(prev_bound)) => {
                pine_crossover(prev_value, prev_bound, rsi, c5) && can_show_upper
            }
            _ => false,
        };

        if lower_signal {
            can_show_lower = false;
        }
        if upper_signal {
            can_show_upper = false;
        }

        let regime_flip = matches!(prev_regime, Some(0)) && regime != 0;

        dst_rsi[i] = rsi;
        dst_lower_bound[i] = c1;
        dst_lower_mid[i] = c2;
        dst_mid[i] = c3;
        dst_upper_mid[i] = c4;
        dst_upper_bound[i] = c5;
        dst_regime[i] = regime as f64;
        dst_regime_flip[i] = if regime_flip { 1.0 } else { 0.0 };
        dst_lower_signal[i] = if lower_signal { 1.0 } else { 0.0 };
        dst_upper_signal[i] = if upper_signal { 1.0 } else { 0.0 };

        prev_rsi = Some(rsi);
        prev_c1 = Some(c1);
        prev_c5 = Some(c5);
        prev_regime = Some(regime);
    }

    Ok(())
}

pub fn adaptive_bounds_rsi_output_into_slice(
    dst: &mut [f64],
    input: &AdaptiveBoundsRsiInput<'_>,
    kernel: Kernel,
    field: AdaptiveBoundsRsiOutputField,
) -> Result<(), AdaptiveBoundsRsiError> {
    let prepared = prepare_input(input, kernel)?;
    if dst.len() != prepared.len {
        return Err(AdaptiveBoundsRsiError::OutputLengthMismatch {
            expected: prepared.len,
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);

    let mut stream = AdaptiveBoundsRsiStream::try_new(AdaptiveBoundsRsiParams {
        rsi_length: Some(prepared.rsi_length),
        alpha: Some(prepared.alpha),
    })?;

    for i in 0..prepared.len {
        if let Some(point) = stream.update(prepared.data[i]) {
            dst[i] = match field {
                AdaptiveBoundsRsiOutputField::Rsi => point.rsi,
                AdaptiveBoundsRsiOutputField::LowerBound => point.lower_bound,
                AdaptiveBoundsRsiOutputField::LowerMid => point.lower_mid,
                AdaptiveBoundsRsiOutputField::Mid => point.mid,
                AdaptiveBoundsRsiOutputField::UpperMid => point.upper_mid,
                AdaptiveBoundsRsiOutputField::UpperBound => point.upper_bound,
                AdaptiveBoundsRsiOutputField::Regime => point.regime,
                AdaptiveBoundsRsiOutputField::RegimeFlip => point.regime_flip,
                AdaptiveBoundsRsiOutputField::LowerSignal => point.lower_signal,
                AdaptiveBoundsRsiOutputField::UpperSignal => point.upper_signal,
            };
        }
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct AdaptiveBoundsRsiBatchRange {
    pub rsi_length: (usize, usize, usize),
    pub alpha: (f64, f64, f64),
}

impl Default for AdaptiveBoundsRsiBatchRange {
    fn default() -> Self {
        Self {
            rsi_length: (DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0),
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveBoundsRsiBatchOutput {
    pub rsi: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub lower_mid: Vec<f64>,
    pub mid: Vec<f64>,
    pub upper_mid: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub regime: Vec<f64>,
    pub regime_flip: Vec<f64>,
    pub lower_signal: Vec<f64>,
    pub upper_signal: Vec<f64>,
    pub combos: Vec<AdaptiveBoundsRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct AdaptiveBoundsRsiBatchBuilder {
    range: AdaptiveBoundsRsiBatchRange,
    kernel: Kernel,
}

impl Default for AdaptiveBoundsRsiBatchBuilder {
    fn default() -> Self {
        Self {
            range: AdaptiveBoundsRsiBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl AdaptiveBoundsRsiBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: AdaptiveBoundsRsiBatchRange) -> Self {
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
    ) -> Result<AdaptiveBoundsRsiBatchOutput, AdaptiveBoundsRsiError> {
        adaptive_bounds_rsi_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<AdaptiveBoundsRsiBatchOutput, AdaptiveBoundsRsiError> {
        self.apply_slice(candles.close.as_slice())
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, AdaptiveBoundsRsiError> {
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
        return Err(AdaptiveBoundsRsiError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, AdaptiveBoundsRsiError> {
    let eps = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(AdaptiveBoundsRsiError::InvalidRange {
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
        return Err(AdaptiveBoundsRsiError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid(
    range: &AdaptiveBoundsRsiBatchRange,
) -> Result<Vec<AdaptiveBoundsRsiParams>, AdaptiveBoundsRsiError> {
    let rsi_lengths = axis_usize(range.rsi_length)?;
    let alphas = axis_f64(range.alpha)?;
    let total = rsi_lengths.len().checked_mul(alphas.len()).ok_or_else(|| {
        AdaptiveBoundsRsiError::InvalidRange {
            start: range.rsi_length.0.to_string(),
            end: range.rsi_length.1.to_string(),
            step: range.rsi_length.2.to_string(),
        }
    })?;

    let mut out = Vec::with_capacity(total);
    for &rsi_length in &rsi_lengths {
        for &alpha in &alphas {
            out.push(AdaptiveBoundsRsiParams {
                rsi_length: Some(rsi_length),
                alpha: Some(alpha),
            });
        }
    }
    Ok(out)
}

pub fn adaptive_bounds_rsi_batch_with_kernel(
    data: &[f64],
    range: &AdaptiveBoundsRsiBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveBoundsRsiBatchOutput, AdaptiveBoundsRsiError> {
    if data.is_empty() {
        return Err(AdaptiveBoundsRsiError::EmptyInputData);
    }
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        value if value.is_batch() => value,
        _ => return Err(AdaptiveBoundsRsiError::InvalidKernelForBatch(kernel)),
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = data.len();

    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(AdaptiveBoundsRsiError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH))
        .collect();

    let mut rsi_mu = make_uninit_matrix(rows, cols);
    let mut lower_bound_mu = make_uninit_matrix(rows, cols);
    let mut lower_mid_mu = make_uninit_matrix(rows, cols);
    let mut mid_mu = make_uninit_matrix(rows, cols);
    let mut upper_mid_mu = make_uninit_matrix(rows, cols);
    let mut upper_bound_mu = make_uninit_matrix(rows, cols);
    let mut regime_mu = make_uninit_matrix(rows, cols);
    let mut regime_flip_mu = make_uninit_matrix(rows, cols);
    let mut lower_signal_mu = make_uninit_matrix(rows, cols);
    let mut upper_signal_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut rsi_mu, cols, &warmups);
    init_matrix_prefixes(&mut lower_bound_mu, cols, &warmups);
    init_matrix_prefixes(&mut lower_mid_mu, cols, &warmups);
    init_matrix_prefixes(&mut mid_mu, cols, &warmups);
    init_matrix_prefixes(&mut upper_mid_mu, cols, &warmups);
    init_matrix_prefixes(&mut upper_bound_mu, cols, &warmups);
    init_matrix_prefixes(&mut regime_mu, cols, &warmups);
    init_matrix_prefixes(&mut regime_flip_mu, cols, &warmups);
    init_matrix_prefixes(&mut lower_signal_mu, cols, &warmups);
    init_matrix_prefixes(&mut upper_signal_mu, cols, &warmups);

    let mut rsi_guard = ManuallyDrop::new(rsi_mu);
    let mut lower_bound_guard = ManuallyDrop::new(lower_bound_mu);
    let mut lower_mid_guard = ManuallyDrop::new(lower_mid_mu);
    let mut mid_guard = ManuallyDrop::new(mid_mu);
    let mut upper_mid_guard = ManuallyDrop::new(upper_mid_mu);
    let mut upper_bound_guard = ManuallyDrop::new(upper_bound_mu);
    let mut regime_guard = ManuallyDrop::new(regime_mu);
    let mut regime_flip_guard = ManuallyDrop::new(regime_flip_mu);
    let mut lower_signal_guard = ManuallyDrop::new(lower_signal_mu);
    let mut upper_signal_guard = ManuallyDrop::new(upper_signal_mu);

    let rsi_all = unsafe { mu_slice_as_f64_slice_mut(&mut rsi_guard) };
    let lower_bound_all = unsafe { mu_slice_as_f64_slice_mut(&mut lower_bound_guard) };
    let lower_mid_all = unsafe { mu_slice_as_f64_slice_mut(&mut lower_mid_guard) };
    let mid_all = unsafe { mu_slice_as_f64_slice_mut(&mut mid_guard) };
    let upper_mid_all = unsafe { mu_slice_as_f64_slice_mut(&mut upper_mid_guard) };
    let upper_bound_all = unsafe { mu_slice_as_f64_slice_mut(&mut upper_bound_guard) };
    let regime_all = unsafe { mu_slice_as_f64_slice_mut(&mut regime_guard) };
    let regime_flip_all = unsafe { mu_slice_as_f64_slice_mut(&mut regime_flip_guard) };
    let lower_signal_all = unsafe { mu_slice_as_f64_slice_mut(&mut lower_signal_guard) };
    let upper_signal_all = unsafe { mu_slice_as_f64_slice_mut(&mut upper_signal_guard) };

    let run_row = |row: usize,
                   rsi_row: &mut [f64],
                   lower_bound_row: &mut [f64],
                   lower_mid_row: &mut [f64],
                   mid_row: &mut [f64],
                   upper_mid_row: &mut [f64],
                   upper_bound_row: &mut [f64],
                   regime_row: &mut [f64],
                   regime_flip_row: &mut [f64],
                   lower_signal_row: &mut [f64],
                   upper_signal_row: &mut [f64]|
     -> Result<(), AdaptiveBoundsRsiError> {
        let input = AdaptiveBoundsRsiInput::from_slice(data, combos[row].clone());
        adaptive_bounds_rsi_into_slices(
            &input,
            single_kernel,
            rsi_row,
            lower_bound_row,
            lower_mid_row,
            mid_row,
            upper_mid_row,
            upper_bound_row,
            regime_row,
            regime_flip_row,
            lower_signal_row,
            upper_signal_row,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        rsi_all
            .par_chunks_mut(cols)
            .zip(lower_bound_all.par_chunks_mut(cols))
            .zip(lower_mid_all.par_chunks_mut(cols))
            .zip(mid_all.par_chunks_mut(cols))
            .zip(upper_mid_all.par_chunks_mut(cols))
            .zip(upper_bound_all.par_chunks_mut(cols))
            .zip(regime_all.par_chunks_mut(cols))
            .zip(regime_flip_all.par_chunks_mut(cols))
            .zip(lower_signal_all.par_chunks_mut(cols))
            .zip(upper_signal_all.par_chunks_mut(cols))
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
                                            (((rsi_row, lower_bound_row), lower_mid_row), mid_row),
                                            upper_mid_row,
                                        ),
                                        upper_bound_row,
                                    ),
                                    regime_row,
                                ),
                                regime_flip_row,
                            ),
                            lower_signal_row,
                        ),
                        upper_signal_row,
                    ),
                )| {
                    run_row(
                        row,
                        rsi_row,
                        lower_bound_row,
                        lower_mid_row,
                        mid_row,
                        upper_mid_row,
                        upper_bound_row,
                        regime_row,
                        regime_flip_row,
                        lower_signal_row,
                        upper_signal_row,
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
                &mut rsi_all[start..end],
                &mut lower_bound_all[start..end],
                &mut lower_mid_all[start..end],
                &mut mid_all[start..end],
                &mut upper_mid_all[start..end],
                &mut upper_bound_all[start..end],
                &mut regime_all[start..end],
                &mut regime_flip_all[start..end],
                &mut lower_signal_all[start..end],
                &mut upper_signal_all[start..end],
            )?;
        }
    }

    Ok(AdaptiveBoundsRsiBatchOutput {
        rsi: unsafe { vec_f64_from_mu_guard(rsi_guard) },
        lower_bound: unsafe { vec_f64_from_mu_guard(lower_bound_guard) },
        lower_mid: unsafe { vec_f64_from_mu_guard(lower_mid_guard) },
        mid: unsafe { vec_f64_from_mu_guard(mid_guard) },
        upper_mid: unsafe { vec_f64_from_mu_guard(upper_mid_guard) },
        upper_bound: unsafe { vec_f64_from_mu_guard(upper_bound_guard) },
        regime: unsafe { vec_f64_from_mu_guard(regime_guard) },
        regime_flip: unsafe { vec_f64_from_mu_guard(regime_flip_guard) },
        lower_signal: unsafe { vec_f64_from_mu_guard(lower_signal_guard) },
        upper_signal: unsafe { vec_f64_from_mu_guard(upper_signal_guard) },
        combos,
        rows,
        cols,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveBoundsRsiStreamOutput {
    pub rsi: f64,
    pub lower_bound: f64,
    pub lower_mid: f64,
    pub mid: f64,
    pub upper_mid: f64,
    pub upper_bound: f64,
    pub regime: f64,
    pub regime_flip: f64,
    pub lower_signal: f64,
    pub upper_signal: f64,
}

#[derive(Debug, Clone)]
pub struct AdaptiveBoundsRsiStream {
    alpha: f64,
    rsi_state: RsiState,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    c5: f64,
    prev_rsi: Option<f64>,
    prev_c1: Option<f64>,
    prev_c5: Option<f64>,
    prev_regime: Option<i32>,
    can_show_lower: bool,
    can_show_upper: bool,
}

impl AdaptiveBoundsRsiStream {
    pub fn try_new(params: AdaptiveBoundsRsiParams) -> Result<Self, AdaptiveBoundsRsiError> {
        let rsi_length = params.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
        let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
        if rsi_length == 0 {
            return Err(AdaptiveBoundsRsiError::InvalidRsiLength {
                rsi_length,
                data_len: 0,
            });
        }
        if !alpha.is_finite() || !(MIN_ALPHA..=MAX_ALPHA).contains(&alpha) {
            return Err(AdaptiveBoundsRsiError::InvalidAlpha { alpha });
        }
        Ok(Self {
            alpha,
            rsi_state: RsiState::new(rsi_length),
            c1: 20.0,
            c2: 40.0,
            c3: 50.0,
            c4: 60.0,
            c5: 80.0,
            prev_rsi: None,
            prev_c1: None,
            prev_c5: None,
            prev_regime: None,
            can_show_lower: true,
            can_show_upper: true,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<AdaptiveBoundsRsiStreamOutput> {
        if !value.is_finite() {
            self.rsi_state.reset();
            self.prev_rsi = None;
            self.prev_c1 = None;
            self.prev_c5 = None;
            self.prev_regime = None;
            return None;
        }

        let rsi = self.rsi_state.update(value)?;
        let d1 = (rsi - self.c1).abs();
        let d2 = (rsi - self.c2).abs();
        let d3 = (rsi - self.c3).abs();
        let d4 = (rsi - self.c4).abs();
        let d5 = (rsi - self.c5).abs();
        let min_dist = d1.min(d2.min(d3.min(d4.min(d5))));
        if min_dist == d1 {
            self.c1 += (rsi - self.c1) * self.alpha;
        } else if min_dist == d2 {
            self.c2 += (rsi - self.c2) * self.alpha;
        } else if min_dist == d3 {
            self.c3 += (rsi - self.c3) * self.alpha;
        } else if min_dist == d4 {
            self.c4 += (rsi - self.c4) * self.alpha;
        } else {
            self.c5 += (rsi - self.c5) * self.alpha;
        }

        let regime_i32 = if rsi <= self.c1 {
            -2
        } else if rsi <= self.c2 {
            -1
        } else if rsi <= self.c3 {
            0
        } else if rsi <= self.c4 {
            1
        } else {
            2
        };

        let crossed_mid = match self.prev_rsi {
            Some(prev) => pine_cross(prev, 50.0, rsi, 50.0),
            None => false,
        };
        if crossed_mid {
            self.can_show_lower = true;
            self.can_show_upper = true;
        }

        let lower_signal = match (self.prev_rsi, self.prev_c1) {
            (Some(prev_value), Some(prev_bound)) => {
                pine_crossunder(prev_value, prev_bound, rsi, self.c1) && self.can_show_lower
            }
            _ => false,
        };
        let upper_signal = match (self.prev_rsi, self.prev_c5) {
            (Some(prev_value), Some(prev_bound)) => {
                pine_crossover(prev_value, prev_bound, rsi, self.c5) && self.can_show_upper
            }
            _ => false,
        };

        if lower_signal {
            self.can_show_lower = false;
        }
        if upper_signal {
            self.can_show_upper = false;
        }

        let regime_flip = matches!(self.prev_regime, Some(0)) && regime_i32 != 0;

        self.prev_rsi = Some(rsi);
        self.prev_c1 = Some(self.c1);
        self.prev_c5 = Some(self.c5);
        self.prev_regime = Some(regime_i32);

        Some(AdaptiveBoundsRsiStreamOutput {
            rsi,
            lower_bound: self.c1,
            lower_mid: self.c2,
            mid: self.c3,
            upper_mid: self.c4,
            upper_bound: self.c5,
            regime: regime_i32 as f64,
            regime_flip: if regime_flip { 1.0 } else { 0.0 },
            lower_signal: if lower_signal { 1.0 } else { 0.0 },
            upper_signal: if upper_signal { 1.0 } else { 0.0 },
        })
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
#[pyfunction(name = "adaptive_bounds_rsi")]
#[pyo3(signature = (data, rsi_length=DEFAULT_RSI_LENGTH, alpha=DEFAULT_ALPHA, kernel=None))]
pub fn adaptive_bounds_rsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_length: usize,
    alpha: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = AdaptiveBoundsRsiInput::from_slice(
        data,
        AdaptiveBoundsRsiParams {
            rsi_length: Some(rsi_length),
            alpha: Some(alpha),
        },
    );
    let output = py
        .allow_threads(|| adaptive_bounds_rsi_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("rsi", output.rsi.into_pyarray(py))?;
    dict.set_item("lower_bound", output.lower_bound.into_pyarray(py))?;
    dict.set_item("lower_mid", output.lower_mid.into_pyarray(py))?;
    dict.set_item("mid", output.mid.into_pyarray(py))?;
    dict.set_item("upper_mid", output.upper_mid.into_pyarray(py))?;
    dict.set_item("upper_bound", output.upper_bound.into_pyarray(py))?;
    dict.set_item("regime", output.regime.into_pyarray(py))?;
    dict.set_item("regime_flip", output.regime_flip.into_pyarray(py))?;
    dict.set_item("lower_signal", output.lower_signal.into_pyarray(py))?;
    dict.set_item("upper_signal", output.upper_signal.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_bounds_rsi_batch")]
#[pyo3(signature = (data, rsi_length_range, alpha_range, kernel=None))]
pub fn adaptive_bounds_rsi_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_length_range: (usize, usize, usize),
    alpha_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            adaptive_bounds_rsi_batch_with_kernel(
                data,
                &AdaptiveBoundsRsiBatchRange {
                    rsi_length: rsi_length_range,
                    alpha: alpha_range,
                },
                kernel,
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
    unsafe { arrays[0].as_slice_mut()? }.copy_from_slice(&output.rsi);
    unsafe { arrays[1].as_slice_mut()? }.copy_from_slice(&output.lower_bound);
    unsafe { arrays[2].as_slice_mut()? }.copy_from_slice(&output.lower_mid);
    unsafe { arrays[3].as_slice_mut()? }.copy_from_slice(&output.mid);
    unsafe { arrays[4].as_slice_mut()? }.copy_from_slice(&output.upper_mid);
    unsafe { arrays[5].as_slice_mut()? }.copy_from_slice(&output.upper_bound);
    unsafe { arrays[6].as_slice_mut()? }.copy_from_slice(&output.regime);
    unsafe { arrays[7].as_slice_mut()? }.copy_from_slice(&output.regime_flip);
    unsafe { arrays[8].as_slice_mut()? }.copy_from_slice(&output.lower_signal);
    unsafe { arrays[9].as_slice_mut()? }.copy_from_slice(&output.upper_signal);

    let dict = PyDict::new(py);
    dict.set_item("rsi", arrays[0].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "lower_bound",
        arrays[1].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("lower_mid", arrays[2].reshape((output.rows, output.cols))?)?;
    dict.set_item("mid", arrays[3].reshape((output.rows, output.cols))?)?;
    dict.set_item("upper_mid", arrays[4].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "upper_bound",
        arrays[5].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("regime", arrays[6].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "regime_flip",
        arrays[7].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lower_signal",
        arrays[8].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "upper_signal",
        arrays[9].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "rsi_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH) as u64)
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
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AdaptiveBoundsRsiStream")]
pub struct AdaptiveBoundsRsiStreamPy {
    stream: AdaptiveBoundsRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdaptiveBoundsRsiStreamPy {
    #[new]
    #[pyo3(signature = (rsi_length=DEFAULT_RSI_LENGTH, alpha=DEFAULT_ALPHA))]
    fn new(rsi_length: usize, alpha: f64) -> PyResult<Self> {
        let stream = AdaptiveBoundsRsiStream::try_new(AdaptiveBoundsRsiParams {
            rsi_length: Some(rsi_length),
            alpha: Some(alpha),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.stream.update(value).map(|output| {
            (
                output.rsi,
                output.lower_bound,
                output.lower_mid,
                output.mid,
                output.upper_mid,
                output.upper_bound,
                output.regime,
                output.regime_flip,
                output.lower_signal,
                output.upper_signal,
            )
        })
    }
}

#[cfg(feature = "python")]
pub fn register_adaptive_bounds_rsi_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adaptive_bounds_rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_bounds_rsi_batch_py, m)?)?;
    m.add_class::<AdaptiveBoundsRsiStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveBoundsRsiJsOutput {
    pub rsi: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub lower_mid: Vec<f64>,
    pub mid: Vec<f64>,
    pub upper_mid: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub regime: Vec<f64>,
    pub regime_flip: Vec<f64>,
    pub lower_signal: Vec<f64>,
    pub upper_signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_bounds_rsi_js)]
pub fn adaptive_bounds_rsi_js(
    data: &[f64],
    rsi_length: usize,
    alpha: f64,
) -> Result<JsValue, JsValue> {
    let input = AdaptiveBoundsRsiInput::from_slice(
        data,
        AdaptiveBoundsRsiParams {
            rsi_length: Some(rsi_length),
            alpha: Some(alpha),
        },
    );
    let output = adaptive_bounds_rsi_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AdaptiveBoundsRsiJsOutput {
        rsi: output.rsi,
        lower_bound: output.lower_bound,
        lower_mid: output.lower_mid,
        mid: output.mid,
        upper_mid: output.upper_mid,
        upper_bound: output.upper_bound,
        regime: output.regime,
        regime_flip: output.regime_flip,
        lower_signal: output.lower_signal,
        upper_signal: output.upper_signal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveBoundsRsiBatchConfig {
    pub rsi_length_range: (usize, usize, usize),
    pub alpha_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveBoundsRsiBatchJsOutput {
    pub rsi: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub lower_mid: Vec<f64>,
    pub mid: Vec<f64>,
    pub upper_mid: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub regime: Vec<f64>,
    pub regime_flip: Vec<f64>,
    pub lower_signal: Vec<f64>,
    pub upper_signal: Vec<f64>,
    pub rsi_lengths: Vec<usize>,
    pub alphas: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_bounds_rsi_batch)]
pub fn adaptive_bounds_rsi_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: AdaptiveBoundsRsiBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = adaptive_bounds_rsi_batch_with_kernel(
        data,
        &AdaptiveBoundsRsiBatchRange {
            rsi_length: cfg.rsi_length_range,
            alpha: cfg.alpha_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&AdaptiveBoundsRsiBatchJsOutput {
        rsi: output.rsi,
        lower_bound: output.lower_bound,
        lower_mid: output.lower_mid,
        mid: output.mid,
        upper_mid: output.upper_mid,
        upper_bound: output.upper_bound,
        regime: output.regime,
        regime_flip: output.regime_flip,
        lower_signal: output.lower_signal,
        upper_signal: output.upper_signal,
        rsi_lengths: output
            .combos
            .iter()
            .map(|combo| combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH))
            .collect(),
        alphas: output
            .combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect(),
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bounds_rsi_output_into_js(
    data: &[f64],
    rsi_length: usize,
    alpha: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_bounds_rsi_js(data, rsi_length, alpha)?;
    crate::write_wasm_object_f64_outputs("adaptive_bounds_rsi_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_bounds_rsi_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_bounds_rsi_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "adaptive_bounds_rsi_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![
            100.0, 101.2, 102.0, 101.5, 102.9, 103.7, 103.0, 102.4, 101.8, 102.5, 103.6, 104.3,
            105.1, 104.6, 103.8, 104.2, 105.0, 106.4, 107.1, 106.7, 105.9, 106.8, 107.4, 108.0,
            107.6, 106.9, 107.8, 108.9, 109.7, 109.1, 108.4, 109.3, 110.1, 110.7, 111.5, 110.8,
            111.4, 112.2, 113.0, 112.4,
        ]
    }

    fn assert_series_eq(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (index, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= 1e-12,
                "series mismatch at {index}: left={a}, right={b}"
            );
        }
    }

    #[test]
    fn adaptive_bounds_rsi_into_matches_single() {
        let data = sample_data();
        let input = AdaptiveBoundsRsiInput::from_slice(
            &data,
            AdaptiveBoundsRsiParams {
                rsi_length: Some(14),
                alpha: Some(0.1),
            },
        );
        let output = adaptive_bounds_rsi(&input).expect("single");

        let mut rsi = vec![f64::NAN; data.len()];
        let mut lower_bound = vec![f64::NAN; data.len()];
        let mut lower_mid = vec![f64::NAN; data.len()];
        let mut mid = vec![f64::NAN; data.len()];
        let mut upper_mid = vec![f64::NAN; data.len()];
        let mut upper_bound = vec![f64::NAN; data.len()];
        let mut regime = vec![f64::NAN; data.len()];
        let mut regime_flip = vec![f64::NAN; data.len()];
        let mut lower_signal = vec![f64::NAN; data.len()];
        let mut upper_signal = vec![f64::NAN; data.len()];

        adaptive_bounds_rsi_into_slices(
            &input,
            Kernel::Scalar,
            &mut rsi,
            &mut lower_bound,
            &mut lower_mid,
            &mut mid,
            &mut upper_mid,
            &mut upper_bound,
            &mut regime,
            &mut regime_flip,
            &mut lower_signal,
            &mut upper_signal,
        )
        .expect("into");

        assert_series_eq(&output.rsi, &rsi);
        assert_series_eq(&output.lower_bound, &lower_bound);
        assert_series_eq(&output.lower_mid, &lower_mid);
        assert_series_eq(&output.mid, &mid);
        assert_series_eq(&output.upper_mid, &upper_mid);
        assert_series_eq(&output.upper_bound, &upper_bound);
        assert_series_eq(&output.regime, &regime);
        assert_series_eq(&output.regime_flip, &regime_flip);
        assert_series_eq(&output.lower_signal, &lower_signal);
        assert_series_eq(&output.upper_signal, &upper_signal);
    }

    #[test]
    fn adaptive_bounds_rsi_stream_matches_batch_points() {
        let data = sample_data();
        let params = AdaptiveBoundsRsiParams {
            rsi_length: Some(14),
            alpha: Some(0.1),
        };
        let input = AdaptiveBoundsRsiInput::from_slice(&data, params.clone());
        let batch = adaptive_bounds_rsi(&input).expect("batch");
        let mut stream = AdaptiveBoundsRsiStream::try_new(params).expect("stream");

        let mut out_rsi = Vec::with_capacity(data.len());
        let mut out_lower_bound = Vec::with_capacity(data.len());
        let mut out_lower_mid = Vec::with_capacity(data.len());
        let mut out_mid = Vec::with_capacity(data.len());
        let mut out_upper_mid = Vec::with_capacity(data.len());
        let mut out_upper_bound = Vec::with_capacity(data.len());
        let mut out_regime = Vec::with_capacity(data.len());
        let mut out_regime_flip = Vec::with_capacity(data.len());
        let mut out_lower_signal = Vec::with_capacity(data.len());
        let mut out_upper_signal = Vec::with_capacity(data.len());

        for value in data {
            if let Some(point) = stream.update(value) {
                out_rsi.push(point.rsi);
                out_lower_bound.push(point.lower_bound);
                out_lower_mid.push(point.lower_mid);
                out_mid.push(point.mid);
                out_upper_mid.push(point.upper_mid);
                out_upper_bound.push(point.upper_bound);
                out_regime.push(point.regime);
                out_regime_flip.push(point.regime_flip);
                out_lower_signal.push(point.lower_signal);
                out_upper_signal.push(point.upper_signal);
            } else {
                out_rsi.push(f64::NAN);
                out_lower_bound.push(f64::NAN);
                out_lower_mid.push(f64::NAN);
                out_mid.push(f64::NAN);
                out_upper_mid.push(f64::NAN);
                out_upper_bound.push(f64::NAN);
                out_regime.push(f64::NAN);
                out_regime_flip.push(f64::NAN);
                out_lower_signal.push(f64::NAN);
                out_upper_signal.push(f64::NAN);
            }
        }

        assert_series_eq(&batch.rsi, &out_rsi);
        assert_series_eq(&batch.lower_bound, &out_lower_bound);
        assert_series_eq(&batch.lower_mid, &out_lower_mid);
        assert_series_eq(&batch.mid, &out_mid);
        assert_series_eq(&batch.upper_mid, &out_upper_mid);
        assert_series_eq(&batch.upper_bound, &out_upper_bound);
        assert_series_eq(&batch.regime, &out_regime);
        assert_series_eq(&batch.regime_flip, &out_regime_flip);
        assert_series_eq(&batch.lower_signal, &out_lower_signal);
        assert_series_eq(&batch.upper_signal, &out_upper_signal);
    }

    #[test]
    fn adaptive_bounds_rsi_batch_first_row_matches_single() {
        let data = sample_data();
        let batch = adaptive_bounds_rsi_batch_with_kernel(
            &data,
            &AdaptiveBoundsRsiBatchRange {
                rsi_length: (14, 16, 2),
                alpha: (0.1, 0.2, 0.1),
            },
            Kernel::Auto,
        )
        .expect("batch");
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, data.len());

        let single = adaptive_bounds_rsi(&AdaptiveBoundsRsiInput::from_slice(
            &data,
            AdaptiveBoundsRsiParams {
                rsi_length: Some(14),
                alpha: Some(0.1),
            },
        ))
        .expect("single");

        assert_series_eq(&batch.rsi[..data.len()], &single.rsi);
        assert_series_eq(&batch.lower_bound[..data.len()], &single.lower_bound);
        assert_series_eq(&batch.upper_bound[..data.len()], &single.upper_bound);
        assert_series_eq(&batch.regime[..data.len()], &single.regime);
    }

    #[test]
    fn adaptive_bounds_rsi_rejects_invalid_inputs() {
        let data = sample_data();
        let err = adaptive_bounds_rsi(&AdaptiveBoundsRsiInput::from_slice(
            &data,
            AdaptiveBoundsRsiParams {
                rsi_length: Some(0),
                alpha: Some(0.1),
            },
        ))
        .expect_err("invalid rsi_length");
        assert!(matches!(
            err,
            AdaptiveBoundsRsiError::InvalidRsiLength { .. }
        ));

        let err = adaptive_bounds_rsi(&AdaptiveBoundsRsiInput::from_slice(
            &data,
            AdaptiveBoundsRsiParams {
                rsi_length: Some(14),
                alpha: Some(0.0),
            },
        ))
        .expect_err("invalid alpha");
        assert!(matches!(err, AdaptiveBoundsRsiError::InvalidAlpha { .. }));
    }
}
