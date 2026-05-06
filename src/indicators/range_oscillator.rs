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
const ATR_FALLBACK_PERIOD: usize = 200;
const ATR_PRIMARY_PERIOD: usize = 2000;
const ZERO_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub enum RangeOscillatorData<'a> {
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
pub struct RangeOscillatorOutput {
    pub oscillator: Vec<f64>,
    pub ma: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub range_width: Vec<f64>,
    pub in_range: Vec<f64>,
    pub trend: Vec<f64>,
    pub break_up: Vec<f64>,
    pub break_down: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RangeOscillatorParams {
    pub length: Option<usize>,
    pub mult: Option<f64>,
}

impl Default for RangeOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            mult: Some(DEFAULT_MULT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeOscillatorInput<'a> {
    pub data: RangeOscillatorData<'a>,
    pub params: RangeOscillatorParams,
}

impl<'a> RangeOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: RangeOscillatorParams) -> Self {
        Self {
            data: RangeOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: RangeOscillatorParams,
    ) -> Self {
        Self {
            data: RangeOscillatorData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, RangeOscillatorParams::default())
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline(always)]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(DEFAULT_MULT)
    }
}

#[derive(Clone, Debug)]
pub struct RangeOscillatorBuilder {
    length: Option<usize>,
    mult: Option<f64>,
    kernel: Kernel,
}

impl Default for RangeOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            mult: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RangeOscillatorBuilder {
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
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<RangeOscillatorOutput, RangeOscillatorError> {
        let input = RangeOscillatorInput::from_candles(
            candles,
            RangeOscillatorParams {
                length: self.length,
                mult: self.mult,
            },
        );
        range_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<RangeOscillatorOutput, RangeOscillatorError> {
        let input = RangeOscillatorInput::from_slices(
            high,
            low,
            close,
            RangeOscillatorParams {
                length: self.length,
                mult: self.mult,
            },
        );
        range_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RangeOscillatorStream, RangeOscillatorError> {
        RangeOscillatorStream::try_new(RangeOscillatorParams {
            length: self.length,
            mult: self.mult,
        })
    }
}

#[derive(Debug, Error)]
pub enum RangeOscillatorError {
    #[error("range_oscillator: input data slice is empty")]
    EmptyInputData,
    #[error("range_oscillator: data length mismatch: high={high}, low={low}, close={close}")]
    DataLengthMismatch {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("range_oscillator: all values are NaN")]
    AllValuesNaN,
    #[error("range_oscillator: invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("range_oscillator: invalid mult: {mult}")]
    InvalidMult { mult: f64 },
    #[error("range_oscillator: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("range_oscillator: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("range_oscillator: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("range_oscillator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    len: usize,
    first: usize,
    length: usize,
    mult: f64,
    warmup: usize,
    all_valid_from_first: bool,
}

#[derive(Clone, Debug)]
struct AtrState {
    period: usize,
    count: usize,
    sum: f64,
    value: Option<f64>,
    prev_close: Option<f64>,
}

impl AtrState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            sum: 0.0,
            value: None,
            prev_close: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = None;
        self.prev_close = None;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if let Some(prev_close) = self.prev_close {
            let hl = high - low;
            let hc = (high - prev_close).abs();
            let lc = (low - prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };
        self.prev_close = Some(close);

        if let Some(prev) = self.value {
            let next = (prev * (self.period as f64 - 1.0) + tr) / self.period as f64;
            self.value = Some(next);
            Some(next)
        } else {
            self.count += 1;
            self.sum += tr;
            if self.count == self.period {
                let seeded = self.sum / self.period as f64;
                self.value = Some(seeded);
                Some(seeded)
            } else {
                None
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeOscillatorStreamOutput {
    pub oscillator: f64,
    pub ma: f64,
    pub upper_band: f64,
    pub lower_band: f64,
    pub range_width: f64,
    pub in_range: f64,
    pub trend: f64,
    pub break_up: f64,
    pub break_down: f64,
}

#[derive(Debug, Clone)]
pub struct RangeOscillatorStream {
    length: usize,
    mult: f64,
    atr_fallback: AtrState,
    atr_primary: AtrState,
    closes: VecDeque<f64>,
    trend: f64,
}

#[inline]
pub fn range_oscillator(
    input: &RangeOscillatorInput<'_>,
) -> Result<RangeOscillatorOutput, RangeOscillatorError> {
    range_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn range_oscillator_with_kernel(
    input: &RangeOscillatorInput<'_>,
    kernel: Kernel,
) -> Result<RangeOscillatorOutput, RangeOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    let mut oscillator = alloc_with_nan_prefix(prepared.len, 0);
    let mut ma = alloc_with_nan_prefix(prepared.len, 0);
    let mut upper_band = alloc_with_nan_prefix(prepared.len, 0);
    let mut lower_band = alloc_with_nan_prefix(prepared.len, 0);
    let mut range_width = alloc_with_nan_prefix(prepared.len, 0);
    let mut in_range = alloc_with_nan_prefix(prepared.len, 0);
    let mut trend = alloc_with_nan_prefix(prepared.len, 0);
    let mut break_up = alloc_with_nan_prefix(prepared.len, 0);
    let mut break_down = alloc_with_nan_prefix(prepared.len, 0);

    compute_into_slices(
        &prepared,
        &mut oscillator,
        &mut ma,
        &mut upper_band,
        &mut lower_band,
        &mut range_width,
        &mut in_range,
        &mut trend,
        &mut break_up,
        &mut break_down,
    )?;

    Ok(RangeOscillatorOutput {
        oscillator,
        ma,
        upper_band,
        lower_band,
        range_width,
        in_range,
        trend,
        break_up,
        break_down,
    })
}

#[inline]
pub fn range_oscillator_into(
    input: &RangeOscillatorInput<'_>,
    oscillator: &mut [f64],
    ma: &mut [f64],
    upper_band: &mut [f64],
    lower_band: &mut [f64],
    range_width: &mut [f64],
    in_range: &mut [f64],
    trend: &mut [f64],
    break_up: &mut [f64],
    break_down: &mut [f64],
) -> Result<(), RangeOscillatorError> {
    range_oscillator_into_slices(
        input,
        Kernel::Auto,
        oscillator,
        ma,
        upper_band,
        lower_band,
        range_width,
        in_range,
        trend,
        break_up,
        break_down,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn range_oscillator_into_slices(
    input: &RangeOscillatorInput<'_>,
    kernel: Kernel,
    oscillator: &mut [f64],
    ma: &mut [f64],
    upper_band: &mut [f64],
    lower_band: &mut [f64],
    range_width: &mut [f64],
    in_range: &mut [f64],
    trend: &mut [f64],
    break_up: &mut [f64],
    break_down: &mut [f64],
) -> Result<(), RangeOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    let got = *[
        oscillator.len(),
        ma.len(),
        upper_band.len(),
        lower_band.len(),
        range_width.len(),
        in_range.len(),
        trend.len(),
        break_up.len(),
        break_down.len(),
    ]
    .iter()
    .min()
    .unwrap_or(&0);
    if oscillator.len() != prepared.len
        || ma.len() != prepared.len
        || upper_band.len() != prepared.len
        || lower_band.len() != prepared.len
        || range_width.len() != prepared.len
        || in_range.len() != prepared.len
        || trend.len() != prepared.len
        || break_up.len() != prepared.len
        || break_down.len() != prepared.len
    {
        return Err(RangeOscillatorError::OutputLengthMismatch {
            expected: prepared.len,
            got,
        });
    }

    compute_into_slices(
        &prepared,
        oscillator,
        ma,
        upper_band,
        lower_band,
        range_width,
        in_range,
        trend,
        break_up,
        break_down,
    )
}

#[inline]
fn resolve_data<'a>(
    input: &'a RangeOscillatorInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), RangeOscillatorError> {
    match &input.data {
        RangeOscillatorData::Candles { candles } => Ok((
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        RangeOscillatorData::Slices { high, low, close } => {
            if high.len() != low.len() || high.len() != close.len() {
                return Err(RangeOscillatorError::DataLengthMismatch {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                });
            }
            Ok((high, low, close))
        }
    }
}

#[inline]
fn prepare_input<'a>(
    input: &'a RangeOscillatorInput<'a>,
    _kernel: Kernel,
) -> Result<PreparedInput<'a>, RangeOscillatorError> {
    let (high, low, close) = resolve_data(input)?;
    let len = close.len();
    if len == 0 {
        return Err(RangeOscillatorError::EmptyInputData);
    }
    let first = (0..len)
        .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .ok_or(RangeOscillatorError::AllValuesNaN)?;

    let length = input.get_length();
    let mult = input.get_mult();

    if length == 0 || length >= len {
        return Err(RangeOscillatorError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if !mult.is_finite() || mult < 0.1 {
        return Err(RangeOscillatorError::InvalidMult { mult });
    }

    let valid = (first..len)
        .filter(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .count();
    let needed = (length + 1).max(ATR_FALLBACK_PERIOD);
    if valid < needed {
        return Err(RangeOscillatorError::NotEnoughValidData { needed, valid });
    }

    Ok(PreparedInput {
        high,
        low,
        close,
        len,
        first,
        length,
        mult,
        warmup: first + length.max(ATR_FALLBACK_PERIOD - 1),
        all_valid_from_first: valid == len - first,
    })
}

#[inline(always)]
fn compute_weighted_ma(closes: &VecDeque<f64>, length: usize) -> Option<f64> {
    if closes.len() < length + 1 {
        return None;
    }
    let mut sum_weighted = 0.0;
    let mut sum_weights = 0.0;
    let last = closes.len() - 1;
    for i in 0..length {
        let curr = closes[last - i];
        let prev = closes[last - i - 1];
        if prev.abs() <= ZERO_EPS {
            continue;
        }
        let delta = (curr - prev).abs();
        let w = delta / prev;
        sum_weighted += curr * w;
        sum_weights += w;
    }
    if sum_weights.abs() <= ZERO_EPS {
        None
    } else {
        Some(sum_weighted / sum_weights)
    }
}

#[inline(always)]
fn compute_point(
    closes: &VecDeque<f64>,
    current_close: f64,
    range_width: f64,
    trend_state: &mut f64,
) -> Option<RangeOscillatorStreamOutput> {
    let length = closes.len().saturating_sub(1);
    let ma = compute_weighted_ma(closes, length)?;
    let mut max_dist = 0.0;
    let last = closes.len() - 1;
    for i in 0..length {
        let value = closes[last - i];
        let dist = (value - ma).abs();
        if dist > max_dist {
            max_dist = dist;
        }
    }

    if current_close > ma {
        *trend_state = 1.0;
    } else if current_close < ma {
        *trend_state = -1.0;
    }

    let upper_band = ma + range_width;
    let lower_band = ma - range_width;
    let break_up = if current_close > upper_band { 1.0 } else { 0.0 };
    let break_down = if current_close < lower_band { 1.0 } else { 0.0 };
    let oscillator = if range_width.abs() <= ZERO_EPS {
        f64::NAN
    } else {
        100.0 * (current_close - ma) / range_width
    };

    Some(RangeOscillatorStreamOutput {
        oscillator,
        ma,
        upper_band,
        lower_band,
        range_width,
        in_range: if max_dist <= range_width { 1.0 } else { 0.0 },
        trend: *trend_state,
        break_up,
        break_down,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn write_nan_row(
    i: usize,
    dst_oscillator: &mut [f64],
    dst_ma: &mut [f64],
    dst_upper_band: &mut [f64],
    dst_lower_band: &mut [f64],
    dst_range_width: &mut [f64],
    dst_in_range: &mut [f64],
    dst_trend: &mut [f64],
    dst_break_up: &mut [f64],
    dst_break_down: &mut [f64],
) {
    dst_oscillator[i] = f64::NAN;
    dst_ma[i] = f64::NAN;
    dst_upper_band[i] = f64::NAN;
    dst_lower_band[i] = f64::NAN;
    dst_range_width[i] = f64::NAN;
    dst_in_range[i] = f64::NAN;
    dst_trend[i] = f64::NAN;
    dst_break_up[i] = f64::NAN;
    dst_break_down[i] = f64::NAN;
}

#[inline(always)]
fn default_ring_index(head: usize, offset_from_latest: usize) -> usize {
    let idx = head + DEFAULT_LENGTH - offset_from_latest;
    if idx > DEFAULT_LENGTH {
        idx - (DEFAULT_LENGTH + 1)
    } else {
        idx
    }
}

#[inline(always)]
fn compute_default_point(
    closes: &[f64; DEFAULT_LENGTH + 1],
    head: usize,
    current_close: f64,
    range_width: f64,
    trend_state: &mut f64,
) -> Option<RangeOscillatorStreamOutput> {
    let mut sum_weighted = 0.0;
    let mut sum_weights = 0.0;
    for i in 0..DEFAULT_LENGTH {
        let curr = closes[default_ring_index(head, i)];
        let prev = closes[default_ring_index(head, i + 1)];
        if prev.abs() <= ZERO_EPS {
            continue;
        }
        let delta = (curr - prev).abs();
        let w = delta / prev;
        sum_weighted += curr * w;
        sum_weights += w;
    }
    if sum_weights.abs() <= ZERO_EPS {
        return None;
    }

    let ma = sum_weighted / sum_weights;
    let mut max_dist = 0.0;
    for i in 0..DEFAULT_LENGTH {
        let value = closes[default_ring_index(head, i)];
        let dist = (value - ma).abs();
        if dist > max_dist {
            max_dist = dist;
        }
    }

    if current_close > ma {
        *trend_state = 1.0;
    } else if current_close < ma {
        *trend_state = -1.0;
    }

    let upper_band = ma + range_width;
    let lower_band = ma - range_width;
    let break_up = if current_close > upper_band { 1.0 } else { 0.0 };
    let break_down = if current_close < lower_band { 1.0 } else { 0.0 };
    let oscillator = if range_width.abs() <= ZERO_EPS {
        f64::NAN
    } else {
        100.0 * (current_close - ma) / range_width
    };

    Some(RangeOscillatorStreamOutput {
        oscillator,
        ma,
        upper_band,
        lower_band,
        range_width,
        in_range: if max_dist <= range_width { 1.0 } else { 0.0 },
        trend: *trend_state,
        break_up,
        break_down,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_default_into_slices(
    prepared: &PreparedInput<'_>,
    dst_oscillator: &mut [f64],
    dst_ma: &mut [f64],
    dst_upper_band: &mut [f64],
    dst_lower_band: &mut [f64],
    dst_range_width: &mut [f64],
    dst_in_range: &mut [f64],
    dst_trend: &mut [f64],
    dst_break_up: &mut [f64],
    dst_break_down: &mut [f64],
) {
    if prepared.all_valid_from_first {
        compute_default_clean_into_slices(
            prepared,
            dst_oscillator,
            dst_ma,
            dst_upper_band,
            dst_lower_band,
            dst_range_width,
            dst_in_range,
            dst_trend,
            dst_break_up,
            dst_break_down,
        );
        return;
    }

    let mut atr_fallback = AtrState::new(ATR_FALLBACK_PERIOD);
    let mut atr_primary = AtrState::new(ATR_PRIMARY_PERIOD);
    let mut closes = [0.0; DEFAULT_LENGTH + 1];
    let mut close_count = 0usize;
    let mut close_head = 0usize;
    let mut trend_state = 0.0;

    for i in 0..prepared.len {
        let high = prepared.high[i];
        let low = prepared.low[i];
        let close = prepared.close[i];
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            atr_fallback.reset();
            atr_primary.reset();
            close_count = 0;
            close_head = 0;
            trend_state = 0.0;
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        }

        let atr200 = atr_fallback.update(high, low, close);
        let atr2000 = atr_primary.update(high, low, close);
        let atr_raw = atr2000.or(atr200);

        if close_count < DEFAULT_LENGTH + 1 {
            closes[close_count] = close;
            close_count += 1;
            if close_count == DEFAULT_LENGTH + 1 {
                close_head = 0;
            }
        } else {
            closes[close_head] = close;
            close_head += 1;
            if close_head == DEFAULT_LENGTH + 1 {
                close_head = 0;
            }
        }

        let Some(atr_raw) = atr_raw else {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        };
        if close_count < DEFAULT_LENGTH + 1 {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        }

        let range_width = atr_raw * DEFAULT_MULT;
        let Some(point) =
            compute_default_point(&closes, close_head, close, range_width, &mut trend_state)
        else {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        };

        dst_oscillator[i] = point.oscillator;
        dst_ma[i] = point.ma;
        dst_upper_band[i] = point.upper_band;
        dst_lower_band[i] = point.lower_band;
        dst_range_width[i] = point.range_width;
        dst_in_range[i] = point.in_range;
        dst_trend[i] = point.trend;
        dst_break_up[i] = point.break_up;
        dst_break_down[i] = point.break_down;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_default_clean_into_slices(
    prepared: &PreparedInput<'_>,
    dst_oscillator: &mut [f64],
    dst_ma: &mut [f64],
    dst_upper_band: &mut [f64],
    dst_lower_band: &mut [f64],
    dst_range_width: &mut [f64],
    dst_in_range: &mut [f64],
    dst_trend: &mut [f64],
    dst_break_up: &mut [f64],
    dst_break_down: &mut [f64],
) {
    for i in 0..prepared.first {
        write_nan_row(
            i,
            dst_oscillator,
            dst_ma,
            dst_upper_band,
            dst_lower_band,
            dst_range_width,
            dst_in_range,
            dst_trend,
            dst_break_up,
            dst_break_down,
        );
    }

    let mut atr_fallback = AtrState::new(ATR_FALLBACK_PERIOD);
    let mut atr_primary = AtrState::new(ATR_PRIMARY_PERIOD);
    let mut closes = [0.0; DEFAULT_LENGTH + 1];
    let mut close_count = 0usize;
    let mut close_head = 0usize;
    let mut trend_state = 0.0;

    for i in prepared.first..prepared.len {
        let high = prepared.high[i];
        let low = prepared.low[i];
        let close = prepared.close[i];

        let atr200 = atr_fallback.update(high, low, close);
        let atr2000 = atr_primary.update(high, low, close);
        let atr_raw = atr2000.or(atr200);

        if close_count < DEFAULT_LENGTH + 1 {
            closes[close_count] = close;
            close_count += 1;
            if close_count == DEFAULT_LENGTH + 1 {
                close_head = 0;
            }
        } else {
            closes[close_head] = close;
            close_head += 1;
            if close_head == DEFAULT_LENGTH + 1 {
                close_head = 0;
            }
        }

        let Some(atr_raw) = atr_raw else {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        };
        if close_count < DEFAULT_LENGTH + 1 {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        }

        let range_width = atr_raw * DEFAULT_MULT;
        let Some(point) =
            compute_default_point(&closes, close_head, close, range_width, &mut trend_state)
        else {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        };

        dst_oscillator[i] = point.oscillator;
        dst_ma[i] = point.ma;
        dst_upper_band[i] = point.upper_band;
        dst_lower_band[i] = point.lower_band;
        dst_range_width[i] = point.range_width;
        dst_in_range[i] = point.in_range;
        dst_trend[i] = point.trend;
        dst_break_up[i] = point.break_up;
        dst_break_down[i] = point.break_down;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_into_slices(
    prepared: &PreparedInput<'_>,
    dst_oscillator: &mut [f64],
    dst_ma: &mut [f64],
    dst_upper_band: &mut [f64],
    dst_lower_band: &mut [f64],
    dst_range_width: &mut [f64],
    dst_in_range: &mut [f64],
    dst_trend: &mut [f64],
    dst_break_up: &mut [f64],
    dst_break_down: &mut [f64],
) -> Result<(), RangeOscillatorError> {
    let got = *[
        dst_oscillator.len(),
        dst_ma.len(),
        dst_upper_band.len(),
        dst_lower_band.len(),
        dst_range_width.len(),
        dst_in_range.len(),
        dst_trend.len(),
        dst_break_up.len(),
        dst_break_down.len(),
    ]
    .iter()
    .min()
    .unwrap_or(&0);
    if dst_oscillator.len() != prepared.len
        || dst_ma.len() != prepared.len
        || dst_upper_band.len() != prepared.len
        || dst_lower_band.len() != prepared.len
        || dst_range_width.len() != prepared.len
        || dst_in_range.len() != prepared.len
        || dst_trend.len() != prepared.len
        || dst_break_up.len() != prepared.len
        || dst_break_down.len() != prepared.len
    {
        return Err(RangeOscillatorError::OutputLengthMismatch {
            expected: prepared.len,
            got,
        });
    }

    if prepared.length == DEFAULT_LENGTH && prepared.mult == DEFAULT_MULT {
        compute_default_into_slices(
            prepared,
            dst_oscillator,
            dst_ma,
            dst_upper_band,
            dst_lower_band,
            dst_range_width,
            dst_in_range,
            dst_trend,
            dst_break_up,
            dst_break_down,
        );
        return Ok(());
    }

    let mut atr_fallback = AtrState::new(ATR_FALLBACK_PERIOD);
    let mut atr_primary = AtrState::new(ATR_PRIMARY_PERIOD);
    let mut closes = VecDeque::with_capacity(prepared.length + 1);
    let mut trend_state = 0.0;

    for i in 0..prepared.len {
        let high = prepared.high[i];
        let low = prepared.low[i];
        let close = prepared.close[i];
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            atr_fallback.reset();
            atr_primary.reset();
            closes.clear();
            trend_state = 0.0;
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        }

        let atr200 = atr_fallback.update(high, low, close);
        let atr2000 = atr_primary.update(high, low, close);
        let atr_raw = atr2000.or(atr200);

        if closes.len() == prepared.length + 1 {
            closes.pop_front();
        }
        closes.push_back(close);

        let Some(atr_raw) = atr_raw else {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        };
        if closes.len() < prepared.length + 1 {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        }

        let range_width = atr_raw * prepared.mult;
        let Some(point) = compute_point(&closes, close, range_width, &mut trend_state) else {
            write_nan_row(
                i,
                dst_oscillator,
                dst_ma,
                dst_upper_band,
                dst_lower_band,
                dst_range_width,
                dst_in_range,
                dst_trend,
                dst_break_up,
                dst_break_down,
            );
            continue;
        };

        dst_oscillator[i] = point.oscillator;
        dst_ma[i] = point.ma;
        dst_upper_band[i] = point.upper_band;
        dst_lower_band[i] = point.lower_band;
        dst_range_width[i] = point.range_width;
        dst_in_range[i] = point.in_range;
        dst_trend[i] = point.trend;
        dst_break_up[i] = point.break_up;
        dst_break_down[i] = point.break_down;
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct RangeOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub mult: (f64, f64, f64),
}

impl Default for RangeOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            mult: (DEFAULT_MULT, DEFAULT_MULT, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RangeOscillatorBatchOutput {
    pub oscillator: Vec<f64>,
    pub ma: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub range_width: Vec<f64>,
    pub in_range: Vec<f64>,
    pub trend: Vec<f64>,
    pub break_up: Vec<f64>,
    pub break_down: Vec<f64>,
    pub combos: Vec<RangeOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct RangeOscillatorBatchBuilder {
    range: RangeOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for RangeOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: RangeOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl RangeOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: RangeOscillatorBatchRange) -> Self {
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
    ) -> Result<RangeOscillatorBatchOutput, RangeOscillatorError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<RangeOscillatorBatchOutput, RangeOscillatorError> {
        range_oscillator_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, RangeOscillatorError> {
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
        return Err(RangeOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, RangeOscillatorError> {
    let eps = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(RangeOscillatorError::InvalidRange {
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
        return Err(RangeOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid(
    range: &RangeOscillatorBatchRange,
) -> Result<Vec<RangeOscillatorParams>, RangeOscillatorError> {
    let lengths = axis_usize(range.length)?;
    let mults = axis_f64(range.mult)?;
    let total = lengths.len().checked_mul(mults.len()).ok_or_else(|| {
        RangeOscillatorError::InvalidRange {
            start: range.length.0.to_string(),
            end: range.length.1.to_string(),
            step: range.length.2.to_string(),
        }
    })?;

    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &mult in &mults {
            out.push(RangeOscillatorParams {
                length: Some(length),
                mult: Some(mult),
            });
        }
    }
    Ok(out)
}

#[inline]
pub fn range_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    range: &RangeOscillatorBatchRange,
    kernel: Kernel,
) -> Result<RangeOscillatorBatchOutput, RangeOscillatorError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(RangeOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(RangeOscillatorError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }

    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        value if value.is_batch() => value,
        _ => return Err(RangeOscillatorError::InvalidKernelForBatch(kernel)),
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = close.len();

    let first = (0..cols)
        .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .ok_or(RangeOscillatorError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first
                + combo
                    .length
                    .unwrap_or(DEFAULT_LENGTH)
                    .max(ATR_FALLBACK_PERIOD - 1)
        })
        .collect();

    let mut osc_mu = make_uninit_matrix(rows, cols);
    let mut ma_mu = make_uninit_matrix(rows, cols);
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);
    let mut width_mu = make_uninit_matrix(rows, cols);
    let mut in_range_mu = make_uninit_matrix(rows, cols);
    let mut trend_mu = make_uninit_matrix(rows, cols);
    let mut break_up_mu = make_uninit_matrix(rows, cols);
    let mut break_down_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut osc_mu, cols, &warmups);
    init_matrix_prefixes(&mut ma_mu, cols, &warmups);
    init_matrix_prefixes(&mut upper_mu, cols, &warmups);
    init_matrix_prefixes(&mut lower_mu, cols, &warmups);
    init_matrix_prefixes(&mut width_mu, cols, &warmups);
    init_matrix_prefixes(&mut in_range_mu, cols, &warmups);
    init_matrix_prefixes(&mut trend_mu, cols, &warmups);
    init_matrix_prefixes(&mut break_up_mu, cols, &warmups);
    init_matrix_prefixes(&mut break_down_mu, cols, &warmups);

    let mut osc_guard = ManuallyDrop::new(osc_mu);
    let mut ma_guard = ManuallyDrop::new(ma_mu);
    let mut upper_guard = ManuallyDrop::new(upper_mu);
    let mut lower_guard = ManuallyDrop::new(lower_mu);
    let mut width_guard = ManuallyDrop::new(width_mu);
    let mut in_range_guard = ManuallyDrop::new(in_range_mu);
    let mut trend_guard = ManuallyDrop::new(trend_mu);
    let mut break_up_guard = ManuallyDrop::new(break_up_mu);
    let mut break_down_guard = ManuallyDrop::new(break_down_mu);

    let osc_all = unsafe { mu_slice_as_f64_slice_mut(&mut osc_guard) };
    let ma_all = unsafe { mu_slice_as_f64_slice_mut(&mut ma_guard) };
    let upper_all = unsafe { mu_slice_as_f64_slice_mut(&mut upper_guard) };
    let lower_all = unsafe { mu_slice_as_f64_slice_mut(&mut lower_guard) };
    let width_all = unsafe { mu_slice_as_f64_slice_mut(&mut width_guard) };
    let in_range_all = unsafe { mu_slice_as_f64_slice_mut(&mut in_range_guard) };
    let trend_all = unsafe { mu_slice_as_f64_slice_mut(&mut trend_guard) };
    let break_up_all = unsafe { mu_slice_as_f64_slice_mut(&mut break_up_guard) };
    let break_down_all = unsafe { mu_slice_as_f64_slice_mut(&mut break_down_guard) };

    let run_row = |row: usize,
                   osc_row: &mut [f64],
                   ma_row: &mut [f64],
                   upper_row: &mut [f64],
                   lower_row: &mut [f64],
                   width_row: &mut [f64],
                   in_range_row: &mut [f64],
                   trend_row: &mut [f64],
                   break_up_row: &mut [f64],
                   break_down_row: &mut [f64]|
     -> Result<(), RangeOscillatorError> {
        let input = RangeOscillatorInput::from_slices(high, low, close, combos[row].clone());
        range_oscillator_into_slices(
            &input,
            single_kernel,
            osc_row,
            ma_row,
            upper_row,
            lower_row,
            width_row,
            in_range_row,
            trend_row,
            break_up_row,
            break_down_row,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        osc_all
            .par_chunks_mut(cols)
            .zip(ma_all.par_chunks_mut(cols))
            .zip(upper_all.par_chunks_mut(cols))
            .zip(lower_all.par_chunks_mut(cols))
            .zip(width_all.par_chunks_mut(cols))
            .zip(in_range_all.par_chunks_mut(cols))
            .zip(trend_all.par_chunks_mut(cols))
            .zip(break_up_all.par_chunks_mut(cols))
            .zip(break_down_all.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(
                |(
                    row,
                    (
                        (
                            (
                                (
                                    ((((osc_row, ma_row), upper_row), lower_row), width_row),
                                    in_range_row,
                                ),
                                trend_row,
                            ),
                            break_up_row,
                        ),
                        break_down_row,
                    ),
                )| {
                    run_row(
                        row,
                        osc_row,
                        ma_row,
                        upper_row,
                        lower_row,
                        width_row,
                        in_range_row,
                        trend_row,
                        break_up_row,
                        break_down_row,
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
                &mut osc_all[start..end],
                &mut ma_all[start..end],
                &mut upper_all[start..end],
                &mut lower_all[start..end],
                &mut width_all[start..end],
                &mut in_range_all[start..end],
                &mut trend_all[start..end],
                &mut break_up_all[start..end],
                &mut break_down_all[start..end],
            )?;
        }
    }

    Ok(RangeOscillatorBatchOutput {
        oscillator: unsafe { vec_f64_from_mu_guard(osc_guard) },
        ma: unsafe { vec_f64_from_mu_guard(ma_guard) },
        upper_band: unsafe { vec_f64_from_mu_guard(upper_guard) },
        lower_band: unsafe { vec_f64_from_mu_guard(lower_guard) },
        range_width: unsafe { vec_f64_from_mu_guard(width_guard) },
        in_range: unsafe { vec_f64_from_mu_guard(in_range_guard) },
        trend: unsafe { vec_f64_from_mu_guard(trend_guard) },
        break_up: unsafe { vec_f64_from_mu_guard(break_up_guard) },
        break_down: unsafe { vec_f64_from_mu_guard(break_down_guard) },
        combos,
        rows,
        cols,
    })
}

impl RangeOscillatorStream {
    pub fn try_new(params: RangeOscillatorParams) -> Result<Self, RangeOscillatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        if length == 0 {
            return Err(RangeOscillatorError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        if !mult.is_finite() || mult < 0.1 {
            return Err(RangeOscillatorError::InvalidMult { mult });
        }
        Ok(Self {
            length,
            mult,
            atr_fallback: AtrState::new(ATR_FALLBACK_PERIOD),
            atr_primary: AtrState::new(ATR_PRIMARY_PERIOD),
            closes: VecDeque::with_capacity(length + 1),
            trend: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<RangeOscillatorStreamOutput> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.atr_fallback.reset();
            self.atr_primary.reset();
            self.closes.clear();
            self.trend = 0.0;
            return None;
        }

        let atr200 = self.atr_fallback.update(high, low, close);
        let atr2000 = self.atr_primary.update(high, low, close);
        if self.closes.len() == self.length + 1 {
            self.closes.pop_front();
        }
        self.closes.push_back(close);

        let atr_raw = atr2000.or(atr200)?;
        if self.closes.len() < self.length + 1 {
            return None;
        }
        let range_width = atr_raw * self.mult;
        compute_point(&self.closes, close, range_width, &mut self.trend)
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
#[pyfunction(name = "range_oscillator")]
#[pyo3(signature = (high, low, close, length=DEFAULT_LENGTH, mult=DEFAULT_MULT, kernel=None))]
pub fn range_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    mult: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = RangeOscillatorInput::from_slices(
        high,
        low,
        close,
        RangeOscillatorParams {
            length: Some(length),
            mult: Some(mult),
        },
    );
    let output = py
        .allow_threads(|| range_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("oscillator", output.oscillator.into_pyarray(py))?;
    dict.set_item("ma", output.ma.into_pyarray(py))?;
    dict.set_item("upper_band", output.upper_band.into_pyarray(py))?;
    dict.set_item("lower_band", output.lower_band.into_pyarray(py))?;
    dict.set_item("range_width", output.range_width.into_pyarray(py))?;
    dict.set_item("in_range", output.in_range.into_pyarray(py))?;
    dict.set_item("trend", output.trend.into_pyarray(py))?;
    dict.set_item("break_up", output.break_up.into_pyarray(py))?;
    dict.set_item("break_down", output.break_down.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "range_oscillator_batch")]
#[pyo3(signature = (high, low, close, length_range, mult_range, kernel=None))]
pub fn range_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            range_oscillator_batch_with_kernel(
                high,
                low,
                close,
                &RangeOscillatorBatchRange {
                    length: length_range,
                    mult: mult_range,
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
    ];
    unsafe { arrays[0].as_slice_mut()? }.copy_from_slice(&output.oscillator);
    unsafe { arrays[1].as_slice_mut()? }.copy_from_slice(&output.ma);
    unsafe { arrays[2].as_slice_mut()? }.copy_from_slice(&output.upper_band);
    unsafe { arrays[3].as_slice_mut()? }.copy_from_slice(&output.lower_band);
    unsafe { arrays[4].as_slice_mut()? }.copy_from_slice(&output.range_width);
    unsafe { arrays[5].as_slice_mut()? }.copy_from_slice(&output.in_range);
    unsafe { arrays[6].as_slice_mut()? }.copy_from_slice(&output.trend);
    unsafe { arrays[7].as_slice_mut()? }.copy_from_slice(&output.break_up);
    unsafe { arrays[8].as_slice_mut()? }.copy_from_slice(&output.break_down);

    let dict = PyDict::new(py);
    dict.set_item("oscillator", arrays[0].reshape((output.rows, output.cols))?)?;
    dict.set_item("ma", arrays[1].reshape((output.rows, output.cols))?)?;
    dict.set_item("upper_band", arrays[2].reshape((output.rows, output.cols))?)?;
    dict.set_item("lower_band", arrays[3].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "range_width",
        arrays[4].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("in_range", arrays[5].reshape((output.rows, output.cols))?)?;
    dict.set_item("trend", arrays[6].reshape((output.rows, output.cols))?)?;
    dict.set_item("break_up", arrays[7].reshape((output.rows, output.cols))?)?;
    dict.set_item("break_down", arrays[8].reshape((output.rows, output.cols))?)?;
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
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "RangeOscillatorStream")]
pub struct RangeOscillatorStreamPy {
    stream: RangeOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RangeOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, mult=DEFAULT_MULT))]
    fn new(length: usize, mult: f64) -> PyResult<Self> {
        let stream = RangeOscillatorStream::try_new(RangeOscillatorParams {
            length: Some(length),
            mult: Some(mult),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.stream.update(high, low, close).map(|output| {
            (
                output.oscillator,
                output.ma,
                output.upper_band,
                output.lower_band,
                output.range_width,
                output.in_range,
                output.trend,
                output.break_up,
                output.break_down,
            )
        })
    }
}

#[cfg(feature = "python")]
pub fn register_range_oscillator_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(range_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(range_oscillator_batch_py, m)?)?;
    m.add_class::<RangeOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RangeOscillatorJsOutput {
    pub oscillator: Vec<f64>,
    pub ma: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub range_width: Vec<f64>,
    pub in_range: Vec<f64>,
    pub trend: Vec<f64>,
    pub break_up: Vec<f64>,
    pub break_down: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = range_oscillator_js)]
pub fn range_oscillator_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
) -> Result<JsValue, JsValue> {
    let input = RangeOscillatorInput::from_slices(
        high,
        low,
        close,
        RangeOscillatorParams {
            length: Some(length),
            mult: Some(mult),
        },
    );
    let output = range_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&RangeOscillatorJsOutput {
        oscillator: output.oscillator,
        ma: output.ma,
        upper_band: output.upper_band,
        lower_band: output.lower_band,
        range_width: output.range_width,
        in_range: output.in_range,
        trend: output.trend,
        break_up: output.break_up,
        break_down: output.break_down,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RangeOscillatorBatchConfig {
    pub length_range: (usize, usize, usize),
    pub mult_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RangeOscillatorBatchJsOutput {
    pub oscillator: Vec<f64>,
    pub ma: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub range_width: Vec<f64>,
    pub in_range: Vec<f64>,
    pub trend: Vec<f64>,
    pub break_up: Vec<f64>,
    pub break_down: Vec<f64>,
    pub lengths: Vec<usize>,
    pub mults: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = range_oscillator_batch)]
pub fn range_oscillator_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: RangeOscillatorBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = range_oscillator_batch_with_kernel(
        high,
        low,
        close,
        &RangeOscillatorBatchRange {
            length: cfg.length_range,
            mult: cfg.mult_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&RangeOscillatorBatchJsOutput {
        oscillator: output.oscillator,
        ma: output.ma,
        upper_band: output.upper_band,
        lower_band: output.lower_band,
        range_width: output.range_width,
        in_range: output.in_range,
        trend: output.trend,
        break_up: output.break_up,
        break_down: output.break_down,
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
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = range_oscillator_js(high, low, close, length, mult)?;
    crate::write_wasm_object_f64_outputs("range_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_oscillator_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = range_oscillator_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "range_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(320);
        let mut low = Vec::with_capacity(320);
        let mut close = Vec::with_capacity(320);
        for i in 0..320 {
            let base = 100.0 + i as f64 * 0.18 + (i as f64 * 0.17).sin() * 1.7;
            let c = base + (i as f64 * 0.11).cos() * 0.45;
            let h = c + 0.9 + (i as f64 * 0.07).sin().abs() * 0.35;
            let l = c - 0.9 - (i as f64 * 0.05).cos().abs() * 0.30;
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (high, low, close)
    }

    #[test]
    fn range_oscillator_into_matches_single() {
        let (high, low, close) = sample_ohlc();
        let input = RangeOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            RangeOscillatorParams {
                length: Some(50),
                mult: Some(2.0),
            },
        );
        let out = range_oscillator_with_kernel(&input, Kernel::Scalar).expect("single");
        let mut oscillator = vec![0.0; close.len()];
        let mut ma = vec![0.0; close.len()];
        let mut upper = vec![0.0; close.len()];
        let mut lower = vec![0.0; close.len()];
        let mut width = vec![0.0; close.len()];
        let mut in_range = vec![0.0; close.len()];
        let mut trend = vec![0.0; close.len()];
        let mut break_up = vec![0.0; close.len()];
        let mut break_down = vec![0.0; close.len()];

        range_oscillator_into_slices(
            &input,
            Kernel::Scalar,
            &mut oscillator,
            &mut ma,
            &mut upper,
            &mut lower,
            &mut width,
            &mut in_range,
            &mut trend,
            &mut break_up,
            &mut break_down,
        )
        .expect("into");

        for i in 0..close.len() {
            for (lhs, rhs) in [
                (out.oscillator[i], oscillator[i]),
                (out.ma[i], ma[i]),
                (out.upper_band[i], upper[i]),
                (out.lower_band[i], lower[i]),
                (out.range_width[i], width[i]),
                (out.in_range[i], in_range[i]),
                (out.trend[i], trend[i]),
                (out.break_up[i], break_up[i]),
                (out.break_down[i], break_down[i]),
            ] {
                if lhs.is_nan() {
                    assert!(rhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn range_oscillator_stream_matches_batch() {
        let (high, low, close) = sample_ohlc();
        let input = RangeOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            RangeOscillatorParams::default(),
        );
        let out = range_oscillator(&input).expect("batch");
        let mut stream =
            RangeOscillatorStream::try_new(RangeOscillatorParams::default()).expect("stream");
        let mut collected = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            collected.push(stream.update(high[i], low[i], close[i]));
        }
        for i in 0..close.len() {
            let Some(point) = collected[i] else {
                assert!(out.oscillator[i].is_nan());
                continue;
            };
            assert!((point.oscillator - out.oscillator[i]).abs() <= 1e-12);
            assert!((point.ma - out.ma[i]).abs() <= 1e-12);
            assert!((point.upper_band - out.upper_band[i]).abs() <= 1e-12);
            assert!((point.lower_band - out.lower_band[i]).abs() <= 1e-12);
            assert!((point.range_width - out.range_width[i]).abs() <= 1e-12);
            assert!((point.in_range - out.in_range[i]).abs() <= 1e-12);
            assert!((point.trend - out.trend[i]).abs() <= 1e-12);
            assert!((point.break_up - out.break_up[i]).abs() <= 1e-12);
            assert!((point.break_down - out.break_down[i]).abs() <= 1e-12);
        }
    }

    #[test]
    fn range_oscillator_into_overwrites_flat_rows_with_nan() {
        let len = 260;
        let high = vec![101.0; len];
        let low = vec![99.0; len];
        let close = vec![100.0; len];
        let input = RangeOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            RangeOscillatorParams::default(),
        );
        let mut oscillator = vec![42.0; len];
        let mut ma = vec![42.0; len];
        let mut upper = vec![42.0; len];
        let mut lower = vec![42.0; len];
        let mut width = vec![42.0; len];
        let mut in_range = vec![42.0; len];
        let mut trend = vec![42.0; len];
        let mut break_up = vec![42.0; len];
        let mut break_down = vec![42.0; len];

        range_oscillator_into_slices(
            &input,
            Kernel::Scalar,
            &mut oscillator,
            &mut ma,
            &mut upper,
            &mut lower,
            &mut width,
            &mut in_range,
            &mut trend,
            &mut break_up,
            &mut break_down,
        )
        .expect("into");

        for i in 0..len {
            assert!(oscillator[i].is_nan());
            assert!(ma[i].is_nan());
            assert!(upper[i].is_nan());
            assert!(lower[i].is_nan());
            assert!(width[i].is_nan());
            assert!(in_range[i].is_nan());
            assert!(trend[i].is_nan());
            assert!(break_up[i].is_nan());
            assert!(break_down[i].is_nan());
        }
    }

    #[test]
    fn range_oscillator_batch_first_row_matches_single() {
        let (high, low, close) = sample_ohlc();
        let single = range_oscillator(&RangeOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            RangeOscillatorParams {
                length: Some(50),
                mult: Some(2.0),
            },
        ))
        .expect("single");
        let batch = range_oscillator_batch_with_kernel(
            &high,
            &low,
            &close,
            &RangeOscillatorBatchRange {
                length: (50, 52, 2),
                mult: (2.0, 2.5, 0.5),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");

        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, close.len());
        for i in 0..close.len() {
            let idx = i;
            for (lhs, rhs) in [
                (single.oscillator[i], batch.oscillator[idx]),
                (single.ma[i], batch.ma[idx]),
                (single.upper_band[i], batch.upper_band[idx]),
                (single.lower_band[i], batch.lower_band[idx]),
                (single.range_width[i], batch.range_width[idx]),
                (single.in_range[i], batch.in_range[idx]),
                (single.trend[i], batch.trend[idx]),
                (single.break_up[i], batch.break_up[idx]),
                (single.break_down[i], batch.break_down[idx]),
            ] {
                if lhs.is_nan() {
                    assert!(rhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn range_oscillator_rejects_invalid_params() {
        let (high, low, close) = sample_ohlc();
        let err = range_oscillator(&RangeOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            RangeOscillatorParams {
                length: Some(0),
                mult: Some(2.0),
            },
        ))
        .expect_err("invalid length");
        assert!(err.to_string().contains("invalid length"));

        let err = RangeOscillatorStream::try_new(RangeOscillatorParams {
            length: Some(50),
            mult: Some(0.0),
        })
        .expect_err("invalid mult");
        assert!(err.to_string().contains("invalid mult"));
    }
}
