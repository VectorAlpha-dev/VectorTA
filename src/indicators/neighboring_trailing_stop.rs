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

const DEFAULT_BUFFER_SIZE: usize = 200;
const DEFAULT_K: usize = 50;
const DEFAULT_PERCENTILE: f64 = 90.0;
const DEFAULT_SMOOTH: usize = 5;
const MIN_BUFFER_SIZE: usize = 100;
const MIN_K: usize = 5;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Debug, Clone)]
pub enum NeighboringTrailingStopData<'a> {
    Candles(&'a Candles),
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct NeighboringTrailingStopOutput {
    pub trailing_stop: Vec<f64>,
    pub bullish_band: Vec<f64>,
    pub bearish_band: Vec<f64>,
    pub direction: Vec<f64>,
    pub discovery_bull: Vec<f64>,
    pub discovery_bear: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct NeighboringTrailingStopPoint {
    pub trailing_stop: f64,
    pub bullish_band: f64,
    pub bearish_band: f64,
    pub direction: f64,
    pub discovery_bull: f64,
    pub discovery_bear: f64,
}

impl NeighboringTrailingStopPoint {
    #[inline(always)]
    fn nan() -> Self {
        Self {
            trailing_stop: f64::NAN,
            bullish_band: f64::NAN,
            bearish_band: f64::NAN,
            direction: f64::NAN,
            discovery_bull: f64::NAN,
            discovery_bear: f64::NAN,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct NeighboringTrailingStopParams {
    pub buffer_size: Option<usize>,
    pub k: Option<usize>,
    pub percentile: Option<f64>,
    pub smooth: Option<usize>,
}

impl Default for NeighboringTrailingStopParams {
    fn default() -> Self {
        Self {
            buffer_size: Some(DEFAULT_BUFFER_SIZE),
            k: Some(DEFAULT_K),
            percentile: Some(DEFAULT_PERCENTILE),
            smooth: Some(DEFAULT_SMOOTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeighboringTrailingStopInput<'a> {
    pub data: NeighboringTrailingStopData<'a>,
    pub params: NeighboringTrailingStopParams,
}

impl<'a> NeighboringTrailingStopInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: NeighboringTrailingStopParams) -> Self {
        Self {
            data: NeighboringTrailingStopData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: NeighboringTrailingStopParams,
    ) -> Self {
        Self {
            data: NeighboringTrailingStopData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, NeighboringTrailingStopParams::default())
    }

    #[inline]
    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            NeighboringTrailingStopData::Candles(candles) => {
                (&candles.high, &candles.low, &candles.close)
            }
            NeighboringTrailingStopData::Slices { high, low, close } => (high, low, close),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeighboringTrailingStopBuilder {
    buffer_size: Option<usize>,
    k: Option<usize>,
    percentile: Option<f64>,
    smooth: Option<usize>,
    kernel: Kernel,
}

impl NeighboringTrailingStopBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn buffer_size(mut self, value: usize) -> Self {
        self.buffer_size = Some(value);
        self
    }

    #[inline]
    pub fn k(mut self, value: usize) -> Self {
        self.k = Some(value);
        self
    }

    #[inline]
    pub fn percentile(mut self, value: f64) -> Self {
        self.percentile = Some(value);
        self
    }

    #[inline]
    pub fn smooth(mut self, value: usize) -> Self {
        self.smooth = Some(value);
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
    ) -> Result<NeighboringTrailingStopOutput, NeighboringTrailingStopError> {
        let input = NeighboringTrailingStopInput::from_candles(
            candles,
            NeighboringTrailingStopParams {
                buffer_size: self.buffer_size,
                k: self.k,
                percentile: self.percentile,
                smooth: self.smooth,
            },
        );
        neighboring_trailing_stop_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<NeighboringTrailingStopOutput, NeighboringTrailingStopError> {
        let input = NeighboringTrailingStopInput::from_slices(
            high,
            low,
            close,
            NeighboringTrailingStopParams {
                buffer_size: self.buffer_size,
                k: self.k,
                percentile: self.percentile,
                smooth: self.smooth,
            },
        );
        neighboring_trailing_stop_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<NeighboringTrailingStopStream, NeighboringTrailingStopError> {
        NeighboringTrailingStopStream::try_new(NeighboringTrailingStopParams {
            buffer_size: self.buffer_size,
            k: self.k,
            percentile: self.percentile,
            smooth: self.smooth,
        })
    }
}

#[derive(Debug, Error)]
pub enum NeighboringTrailingStopError {
    #[error("neighboring_trailing_stop: Input data slice is empty.")]
    EmptyInputData,
    #[error("neighboring_trailing_stop: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "neighboring_trailing_stop: Inconsistent slice lengths - high={high_len}, low={low_len}, close={close_len}"
    )]
    MismatchedInputLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "neighboring_trailing_stop: Invalid buffer_size: buffer_size = {buffer_size}, min = {min}"
    )]
    InvalidBufferSize { buffer_size: usize, min: usize },
    #[error("neighboring_trailing_stop: Invalid k: k = {k}, min = {min}")]
    InvalidK { k: usize, min: usize },
    #[error("neighboring_trailing_stop: Invalid percentile: {percentile}")]
    InvalidPercentile { percentile: f64 },
    #[error("neighboring_trailing_stop: Invalid smooth: {smooth}")]
    InvalidSmooth { smooth: usize },
    #[error("neighboring_trailing_stop: Output length mismatch: expected = {expected}")]
    OutputLengthMismatch { expected: usize },
    #[error("neighboring_trailing_stop: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("neighboring_trailing_stop: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    buffer_size: usize,
    k: usize,
    percentile: f64,
    smooth: usize,
}

#[inline(always)]
fn first_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let len = high.len();
    let mut i = 0usize;
    while i < len {
        if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
            return i;
        }
        i += 1;
    }
    len
}

#[inline(always)]
fn resolve_params(
    params: &NeighboringTrailingStopParams,
) -> Result<ResolvedParams, NeighboringTrailingStopError> {
    let buffer_size = params.buffer_size.unwrap_or(DEFAULT_BUFFER_SIZE);
    let k = params.k.unwrap_or(DEFAULT_K);
    let percentile = params.percentile.unwrap_or(DEFAULT_PERCENTILE);
    let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);

    if buffer_size < MIN_BUFFER_SIZE {
        return Err(NeighboringTrailingStopError::InvalidBufferSize {
            buffer_size,
            min: MIN_BUFFER_SIZE,
        });
    }
    if k < MIN_K {
        return Err(NeighboringTrailingStopError::InvalidK { k, min: MIN_K });
    }
    if !percentile.is_finite() || !(1.0..=99.0).contains(&percentile) {
        return Err(NeighboringTrailingStopError::InvalidPercentile { percentile });
    }
    if smooth == 0 {
        return Err(NeighboringTrailingStopError::InvalidSmooth { smooth });
    }

    Ok(ResolvedParams {
        buffer_size,
        k,
        percentile,
        smooth,
    })
}

#[inline(always)]
fn lower_bound(sorted: &[f64], value: f64) -> usize {
    let mut left = 0usize;
    let mut right = sorted.len();
    while left < right {
        let mid = left + ((right - left) >> 1);
        if sorted[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[inline(always)]
fn insert_sorted(sorted: &mut Vec<f64>, value: f64) {
    let idx = lower_bound(sorted, value);
    sorted.insert(idx, value);
}

#[inline(always)]
fn remove_sorted_once(sorted: &mut Vec<f64>, value: f64) {
    let idx = lower_bound(sorted, value);
    if idx < sorted.len() && sorted[idx] == value {
        sorted.remove(idx);
    }
}

#[inline(always)]
fn percentile_sorted_slice(sorted: &[f64], percentile: f64) -> f64 {
    let len = sorted.len();
    if len == 0 {
        return f64::NAN;
    }
    if len == 1 {
        return sorted[0];
    }

    let idx = (len.saturating_sub(1)) as f64 * percentile / 100.0;
    let i1 = idx.floor() as usize;
    let i2 = idx.ceil() as usize;
    if i1 == i2 {
        sorted[i1]
    } else {
        let v1 = sorted[i1];
        let v2 = sorted[i2];
        v1 + (v2 - v1) * (idx - i1 as f64)
    }
}

#[derive(Clone, Debug)]
struct SmaIgnoreNa {
    period: usize,
    values: VecDeque<f64>,
    sum: f64,
}

impl SmaIgnoreNa {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: VecDeque::with_capacity(period.max(1)),
            sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        if value.is_finite() {
            self.values.push_back(value);
            self.sum += value;
            if self.values.len() > self.period {
                if let Some(old) = self.values.pop_front() {
                    self.sum -= old;
                }
            }
        }

        if self.values.len() == self.period {
            self.sum / self.period as f64
        } else {
            f64::NAN
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }
}

#[derive(Clone, Debug)]
struct CoreState {
    params: ResolvedParams,
    price_buffer: VecDeque<f64>,
    sorted: Vec<f64>,
    bull_sma: SmaIgnoreNa,
    bear_sma: SmaIgnoreNa,
    direction: i8,
    trailing_stop: f64,
}

impl CoreState {
    #[inline]
    fn new(params: ResolvedParams) -> Self {
        Self {
            price_buffer: VecDeque::with_capacity(params.buffer_size.max(params.smooth)),
            sorted: Vec::with_capacity(params.buffer_size.max(params.k)),
            bull_sma: SmaIgnoreNa::new(params.smooth),
            bear_sma: SmaIgnoreNa::new(params.smooth),
            params,
            direction: 0,
            trailing_stop: f64::NAN,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.price_buffer.clear();
        self.sorted.clear();
        self.bull_sma.reset();
        self.bear_sma.reset();
        self.direction = 0;
        self.trailing_stop = f64::NAN;
    }

    #[inline]
    fn update(&mut self, high: f64, low: f64, close: f64) -> NeighboringTrailingStopPoint {
        let mut bear_val = f64::NAN;
        let mut bull_val = f64::NAN;
        let size = self.sorted.len();

        if size > 5 {
            let idx = lower_bound(&self.sorted, close);
            let bear_start = idx.saturating_sub(self.params.k);
            if idx > bear_start {
                bear_val = percentile_sorted_slice(
                    &self.sorted[bear_start..idx],
                    100.0 - self.params.percentile,
                );
            }

            if size > 0 {
                let bull_end = (idx + self.params.k).min(size - 1);
                if bull_end > idx {
                    bull_val = percentile_sorted_slice(
                        &self.sorted[idx..(bull_end + 1)],
                        self.params.percentile,
                    );
                }
            }
        }

        if self.price_buffer.len() >= self.params.buffer_size {
            if let Some(old) = self.price_buffer.pop_front() {
                remove_sorted_once(&mut self.sorted, old);
            }
        }
        self.price_buffer.push_back(close);
        insert_sorted(&mut self.sorted, close);

        let final_bull = self.bull_sma.update(bull_val);
        let final_bear = self.bear_sma.update(bear_val);
        let discovery_bull = bull_val.is_nan() && bear_val.is_finite();
        let discovery_bear = bear_val.is_nan() && bull_val.is_finite();

        let prev_direction = self.direction;
        if discovery_bull {
            self.direction = 1;
        } else if discovery_bear {
            self.direction = -1;
        }

        if self.direction > prev_direction {
            self.trailing_stop = if final_bear.is_finite() {
                final_bear
            } else {
                low
            };
        } else if self.direction < prev_direction {
            self.trailing_stop = if final_bull.is_finite() {
                final_bull
            } else {
                high
            };
        }

        if self.direction == 1 {
            let candidate = if final_bear.is_finite() {
                final_bear
            } else {
                self.trailing_stop
            };
            self.trailing_stop = if self.trailing_stop.is_finite() {
                self.trailing_stop.max(candidate)
            } else {
                candidate
            };
        } else if self.direction == -1 {
            let candidate = if final_bull.is_finite() {
                final_bull
            } else {
                self.trailing_stop
            };
            self.trailing_stop = if self.trailing_stop.is_finite() {
                self.trailing_stop.min(candidate)
            } else {
                candidate
            };
        }

        NeighboringTrailingStopPoint {
            trailing_stop: self.trailing_stop,
            bullish_band: final_bull,
            bearish_band: final_bear,
            direction: self.direction as f64,
            discovery_bull: if discovery_bull { 1.0 } else { 0.0 },
            discovery_bear: if discovery_bear { 1.0 } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeighboringTrailingStopStream {
    state: CoreState,
}

impl NeighboringTrailingStopStream {
    #[inline]
    pub fn try_new(
        params: NeighboringTrailingStopParams,
    ) -> Result<Self, NeighboringTrailingStopError> {
        let params = resolve_params(&params)?;
        Ok(Self {
            state: CoreState::new(params),
        })
    }

    #[inline]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<NeighboringTrailingStopPoint> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.state.reset();
            return None;
        }
        Some(self.state.update(high, low, close))
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state.reset();
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        0
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn neighboring_trailing_stop_row_from_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    trailing_stop: &mut [f64],
    bullish_band: &mut [f64],
    bearish_band: &mut [f64],
    direction: &mut [f64],
    discovery_bull: &mut [f64],
    discovery_bear: &mut [f64],
) {
    let len = high.len();
    debug_assert_eq!(low.len(), len);
    debug_assert_eq!(close.len(), len);
    debug_assert_eq!(trailing_stop.len(), len);
    debug_assert_eq!(bullish_band.len(), len);
    debug_assert_eq!(bearish_band.len(), len);
    debug_assert_eq!(direction.len(), len);
    debug_assert_eq!(discovery_bull.len(), len);
    debug_assert_eq!(discovery_bear.len(), len);

    if params.buffer_size == DEFAULT_BUFFER_SIZE
        && params.k == DEFAULT_K
        && params.percentile == DEFAULT_PERCENTILE
        && params.smooth == DEFAULT_SMOOTH
    {
        neighboring_trailing_stop_default_row(
            high,
            low,
            close,
            trailing_stop,
            bullish_band,
            bearish_band,
            direction,
            discovery_bull,
            discovery_bear,
        );
        return;
    }

    let mut state = CoreState::new(params);
    let mut i = 0usize;
    while i < len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let point = if h.is_finite() && l.is_finite() && c.is_finite() {
            state.update(h, l, c)
        } else {
            state.reset();
            NeighboringTrailingStopPoint::nan()
        };
        trailing_stop[i] = point.trailing_stop;
        bullish_band[i] = point.bullish_band;
        bearish_band[i] = point.bearish_band;
        direction[i] = point.direction;
        discovery_bull[i] = point.discovery_bull;
        discovery_bear[i] = point.discovery_bear;
        i += 1;
    }
}

#[derive(Clone, Copy)]
struct NeighborSma5 {
    ring: [f64; DEFAULT_SMOOTH],
    head: usize,
    count: usize,
    sum: f64,
}

impl NeighborSma5 {
    #[inline(always)]
    fn new() -> Self {
        Self {
            ring: [0.0; DEFAULT_SMOOTH],
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        if value.is_finite() {
            if self.count == DEFAULT_SMOOTH {
                let old = self.ring[self.head];
                self.ring[self.head] = value;
                self.sum += value;
                self.sum -= old;
            } else {
                self.count += 1;
                self.ring[self.head] = value;
                self.sum += value;
            }
            self.head += 1;
            if self.head == DEFAULT_SMOOTH {
                self.head = 0;
            }
        }

        if self.count == DEFAULT_SMOOTH {
            self.sum / DEFAULT_SMOOTH as f64
        } else {
            f64::NAN
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn neighboring_trailing_stop_default_row(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    trailing_stop: &mut [f64],
    bullish_band: &mut [f64],
    bearish_band: &mut [f64],
    direction_out: &mut [f64],
    discovery_bull_out: &mut [f64],
    discovery_bear_out: &mut [f64],
) {
    let len = close.len();
    let mut price_ring = [0.0f64; DEFAULT_BUFFER_SIZE];
    let mut price_head = 0usize;
    let mut price_count = 0usize;
    let mut sorted = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
    let mut bull_sma = NeighborSma5::new();
    let mut bear_sma = NeighborSma5::new();
    let mut direction = 0i8;
    let mut stop = f64::NAN;

    let mut i = 0usize;
    while i < len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !h.is_finite() || !l.is_finite() || !c.is_finite() {
            price_head = 0;
            price_count = 0;
            sorted.clear();
            bull_sma.reset();
            bear_sma.reset();
            direction = 0;
            stop = f64::NAN;
            trailing_stop[i] = f64::NAN;
            bullish_band[i] = f64::NAN;
            bearish_band[i] = f64::NAN;
            direction_out[i] = f64::NAN;
            discovery_bull_out[i] = f64::NAN;
            discovery_bear_out[i] = f64::NAN;
            i += 1;
            continue;
        }

        let mut bear_val = f64::NAN;
        let mut bull_val = f64::NAN;
        let size = sorted.len();
        if size > 5 {
            let idx = lower_bound(&sorted, c);
            let bear_start = idx.saturating_sub(DEFAULT_K);
            if idx > bear_start {
                bear_val = percentile_sorted_slice(&sorted[bear_start..idx], 10.0);
            }

            let bull_end = (idx + DEFAULT_K).min(size - 1);
            if bull_end > idx {
                bull_val = percentile_sorted_slice(&sorted[idx..(bull_end + 1)], 90.0);
            }
        }

        if price_count == DEFAULT_BUFFER_SIZE {
            let idx = lower_bound(&sorted, price_ring[price_head]);
            if idx < sorted.len() && sorted[idx] == price_ring[price_head] {
                sorted.remove(idx);
            }
        } else {
            price_count += 1;
        }
        price_ring[price_head] = c;
        price_head += 1;
        if price_head == DEFAULT_BUFFER_SIZE {
            price_head = 0;
        }
        let idx = lower_bound(&sorted, c);
        sorted.insert(idx, c);

        let final_bull = bull_sma.update(bull_val);
        let final_bear = bear_sma.update(bear_val);
        let discovery_bull = bull_val.is_nan() && bear_val.is_finite();
        let discovery_bear = bear_val.is_nan() && bull_val.is_finite();

        let prev_direction = direction;
        if discovery_bull {
            direction = 1;
        } else if discovery_bear {
            direction = -1;
        }

        if direction > prev_direction {
            stop = if final_bear.is_finite() {
                final_bear
            } else {
                l
            };
        } else if direction < prev_direction {
            stop = if final_bull.is_finite() {
                final_bull
            } else {
                h
            };
        }

        if direction == 1 {
            let candidate = if final_bear.is_finite() {
                final_bear
            } else {
                stop
            };
            stop = if stop.is_finite() {
                stop.max(candidate)
            } else {
                candidate
            };
        } else if direction == -1 {
            let candidate = if final_bull.is_finite() {
                final_bull
            } else {
                stop
            };
            stop = if stop.is_finite() {
                stop.min(candidate)
            } else {
                candidate
            };
        }

        trailing_stop[i] = stop;
        bullish_band[i] = final_bull;
        bearish_band[i] = final_bear;
        direction_out[i] = direction as f64;
        discovery_bull_out[i] = if discovery_bull { 1.0 } else { 0.0 };
        discovery_bear_out[i] = if discovery_bear { 1.0 } else { 0.0 };
        i += 1;
    }
}

#[inline]
pub fn neighboring_trailing_stop(
    input: &NeighboringTrailingStopInput,
) -> Result<NeighboringTrailingStopOutput, NeighboringTrailingStopError> {
    neighboring_trailing_stop_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn neighboring_trailing_stop_with_kernel(
    input: &NeighboringTrailingStopInput,
    _kernel: Kernel,
) -> Result<NeighboringTrailingStopOutput, NeighboringTrailingStopError> {
    let (high, low, close) = input.as_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(NeighboringTrailingStopError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(NeighboringTrailingStopError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    if first_valid_ohlc(high, low, close) >= high.len() {
        return Err(NeighboringTrailingStopError::AllValuesNaN);
    }

    let params = resolve_params(&input.params)?;
    let len = close.len();
    let mut trailing_stop = alloc_with_nan_prefix(len, 0);
    let mut bullish_band = alloc_with_nan_prefix(len, 0);
    let mut bearish_band = alloc_with_nan_prefix(len, 0);
    let mut direction = alloc_with_nan_prefix(len, 0);
    let mut discovery_bull = alloc_with_nan_prefix(len, 0);
    let mut discovery_bear = alloc_with_nan_prefix(len, 0);

    neighboring_trailing_stop_row_from_slices(
        high,
        low,
        close,
        params,
        &mut trailing_stop,
        &mut bullish_band,
        &mut bearish_band,
        &mut direction,
        &mut discovery_bull,
        &mut discovery_bear,
    );

    Ok(NeighboringTrailingStopOutput {
        trailing_stop,
        bullish_band,
        bearish_band,
        direction,
        discovery_bull,
        discovery_bear,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn neighboring_trailing_stop_into_slices(
    trailing_stop_out: &mut [f64],
    bullish_band_out: &mut [f64],
    bearish_band_out: &mut [f64],
    direction_out: &mut [f64],
    discovery_bull_out: &mut [f64],
    discovery_bear_out: &mut [f64],
    input: &NeighboringTrailingStopInput,
    _kernel: Kernel,
) -> Result<(), NeighboringTrailingStopError> {
    let (high, low, close) = input.as_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(NeighboringTrailingStopError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(NeighboringTrailingStopError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let expected = high.len();
    if trailing_stop_out.len() != expected
        || bullish_band_out.len() != expected
        || bearish_band_out.len() != expected
        || direction_out.len() != expected
        || discovery_bull_out.len() != expected
        || discovery_bear_out.len() != expected
    {
        return Err(NeighboringTrailingStopError::OutputLengthMismatch { expected });
    }
    if first_valid_ohlc(high, low, close) >= high.len() {
        return Err(NeighboringTrailingStopError::AllValuesNaN);
    }

    let params = resolve_params(&input.params)?;
    neighboring_trailing_stop_row_from_slices(
        high,
        low,
        close,
        params,
        trailing_stop_out,
        bullish_band_out,
        bearish_band_out,
        direction_out,
        discovery_bull_out,
        discovery_bear_out,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn neighboring_trailing_stop_into(
    trailing_stop_out: &mut [f64],
    bullish_band_out: &mut [f64],
    bearish_band_out: &mut [f64],
    direction_out: &mut [f64],
    discovery_bull_out: &mut [f64],
    discovery_bear_out: &mut [f64],
    input: &NeighboringTrailingStopInput,
) -> Result<(), NeighboringTrailingStopError> {
    neighboring_trailing_stop_into_slices(
        trailing_stop_out,
        bullish_band_out,
        bearish_band_out,
        direction_out,
        discovery_bull_out,
        discovery_bear_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeighboringTrailingStopBatchRange {
    pub buffer_size: (usize, usize, usize),
    pub k: (usize, usize, usize),
    pub percentile: (f64, f64, f64),
    pub smooth: (usize, usize, usize),
}

impl Default for NeighboringTrailingStopBatchRange {
    fn default() -> Self {
        Self {
            buffer_size: (DEFAULT_BUFFER_SIZE, DEFAULT_BUFFER_SIZE, 0),
            k: (DEFAULT_K, DEFAULT_K, 0),
            percentile: (DEFAULT_PERCENTILE, DEFAULT_PERCENTILE, 0.0),
            smooth: (DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeighboringTrailingStopBatchOutput {
    pub trailing_stop: Vec<f64>,
    pub bullish_band: Vec<f64>,
    pub bearish_band: Vec<f64>,
    pub direction: Vec<f64>,
    pub discovery_bull: Vec<f64>,
    pub discovery_bear: Vec<f64>,
    pub combos: Vec<NeighboringTrailingStopParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NeighboringTrailingStopBatchOutput {
    #[inline]
    pub fn params_for(&self, row: usize) -> Option<&NeighboringTrailingStopParams> {
        self.combos.get(row)
    }

    #[inline]
    pub fn row_slices(
        &self,
        row: usize,
    ) -> Option<(&[f64], &[f64], &[f64], &[f64], &[f64], &[f64])> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some((
            &self.trailing_stop[start..end],
            &self.bullish_band[start..end],
            &self.bearish_band[start..end],
            &self.direction[start..end],
            &self.discovery_bull[start..end],
            &self.discovery_bear[start..end],
        ))
    }
}

#[derive(Clone, Debug, Default)]
pub struct NeighboringTrailingStopBatchBuilder {
    range: NeighboringTrailingStopBatchRange,
    kernel: Kernel,
}

impl NeighboringTrailingStopBatchBuilder {
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
    pub fn buffer_size_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.buffer_size = (start, end, step);
        self
    }

    #[inline]
    pub fn k_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.k = (start, end, step);
        self
    }

    #[inline]
    pub fn percentile_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.percentile = (start, end, step);
        self
    }

    #[inline]
    pub fn smooth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<NeighboringTrailingStopBatchOutput, NeighboringTrailingStopError> {
        neighboring_trailing_stop_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<NeighboringTrailingStopBatchOutput, NeighboringTrailingStopError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, NeighboringTrailingStopError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            let next = value.saturating_add(step);
            if next == value {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }
    if out.is_empty() {
        return Err(NeighboringTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, NeighboringTrailingStopError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(NeighboringTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(NeighboringTrailingStopError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(NeighboringTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        out.push(value.min(end));
        value += step;
    }
    if (out.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
        return Err(NeighboringTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_neighboring_trailing_stop(
    sweep: &NeighboringTrailingStopBatchRange,
) -> Result<Vec<NeighboringTrailingStopParams>, NeighboringTrailingStopError> {
    let buffer_sizes = expand_axis_usize(sweep.buffer_size)?;
    let ks = expand_axis_usize(sweep.k)?;
    let percentiles = expand_axis_f64(sweep.percentile.0, sweep.percentile.1, sweep.percentile.2)?;
    let smooths = expand_axis_usize(sweep.smooth)?;

    let capacity = buffer_sizes
        .len()
        .saturating_mul(ks.len())
        .saturating_mul(percentiles.len())
        .saturating_mul(smooths.len());
    let mut combos = Vec::with_capacity(capacity);
    for buffer_size in buffer_sizes {
        for &k in &ks {
            for &percentile in &percentiles {
                for &smooth in &smooths {
                    let params = NeighboringTrailingStopParams {
                        buffer_size: Some(buffer_size),
                        k: Some(k),
                        percentile: Some(percentile),
                        smooth: Some(smooth),
                    };
                    let _ = resolve_params(&params)?;
                    combos.push(params);
                }
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn neighboring_trailing_stop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NeighboringTrailingStopBatchRange,
    kernel: Kernel,
) -> Result<NeighboringTrailingStopBatchOutput, NeighboringTrailingStopError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(NeighboringTrailingStopError::InvalidKernelForBatch(other)),
    };
    neighboring_trailing_stop_batch_par_slices(high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn neighboring_trailing_stop_batch_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NeighboringTrailingStopBatchRange,
    kernel: Kernel,
) -> Result<NeighboringTrailingStopBatchOutput, NeighboringTrailingStopError> {
    neighboring_trailing_stop_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline]
pub fn neighboring_trailing_stop_batch_par_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NeighboringTrailingStopBatchRange,
    kernel: Kernel,
) -> Result<NeighboringTrailingStopBatchOutput, NeighboringTrailingStopError> {
    neighboring_trailing_stop_batch_inner(high, low, close, sweep, kernel, true)
}

#[allow(clippy::too_many_lines)]
pub fn neighboring_trailing_stop_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NeighboringTrailingStopBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<NeighboringTrailingStopBatchOutput, NeighboringTrailingStopError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(NeighboringTrailingStopError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(NeighboringTrailingStopError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    if first_valid_ohlc(high, low, close) >= high.len() {
        return Err(NeighboringTrailingStopError::AllValuesNaN);
    }

    let combos = expand_grid_neighboring_trailing_stop(sweep)?;
    let resolved = combos
        .iter()
        .map(resolve_params)
        .collect::<Result<Vec<_>, _>>()?;
    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(NeighboringTrailingStopError::OutputLengthMismatch {
                expected: usize::MAX,
            })?;
    let zero_prefixes = vec![0usize; rows];

    let mut trailing_stop_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut trailing_stop_mu, cols, &zero_prefixes);
    let mut trailing_stop_guard = ManuallyDrop::new(trailing_stop_mu);
    let trailing_stop_out = unsafe {
        std::slice::from_raw_parts_mut(trailing_stop_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut bullish_band_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut bullish_band_mu, cols, &zero_prefixes);
    let mut bullish_band_guard = ManuallyDrop::new(bullish_band_mu);
    let bullish_band_out = unsafe {
        std::slice::from_raw_parts_mut(bullish_band_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut bearish_band_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut bearish_band_mu, cols, &zero_prefixes);
    let mut bearish_band_guard = ManuallyDrop::new(bearish_band_mu);
    let bearish_band_out = unsafe {
        std::slice::from_raw_parts_mut(bearish_band_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut direction_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut direction_mu, cols, &zero_prefixes);
    let mut direction_guard = ManuallyDrop::new(direction_mu);
    let direction_out =
        unsafe { std::slice::from_raw_parts_mut(direction_guard.as_mut_ptr() as *mut f64, total) };

    let mut discovery_bull_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut discovery_bull_mu, cols, &zero_prefixes);
    let mut discovery_bull_guard = ManuallyDrop::new(discovery_bull_mu);
    let discovery_bull_out = unsafe {
        std::slice::from_raw_parts_mut(discovery_bull_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut discovery_bear_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut discovery_bear_mu, cols, &zero_prefixes);
    let mut discovery_bear_guard = ManuallyDrop::new(discovery_bear_mu);
    let discovery_bear_out = unsafe {
        std::slice::from_raw_parts_mut(discovery_bear_guard.as_mut_ptr() as *mut f64, total)
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let trailing_stop_ptr = trailing_stop_out.as_mut_ptr() as usize;
            let bullish_band_ptr = bullish_band_out.as_mut_ptr() as usize;
            let bearish_band_ptr = bearish_band_out.as_mut_ptr() as usize;
            let direction_ptr = direction_out.as_mut_ptr() as usize;
            let discovery_bull_ptr = discovery_bull_out.as_mut_ptr() as usize;
            let discovery_bear_ptr = discovery_bear_out.as_mut_ptr() as usize;

            resolved
                .par_iter()
                .enumerate()
                .for_each(|(row, params)| unsafe {
                    let start = row * cols;
                    neighboring_trailing_stop_row_from_slices(
                        high,
                        low,
                        close,
                        *params,
                        std::slice::from_raw_parts_mut(
                            (trailing_stop_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (bullish_band_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (bearish_band_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (direction_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (discovery_bull_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (discovery_bear_ptr as *mut f64).add(start),
                            cols,
                        ),
                    );
                });
        }

        #[cfg(target_arch = "wasm32")]
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            neighboring_trailing_stop_row_from_slices(
                high,
                low,
                close,
                *params,
                &mut trailing_stop_out[start..end],
                &mut bullish_band_out[start..end],
                &mut bearish_band_out[start..end],
                &mut direction_out[start..end],
                &mut discovery_bull_out[start..end],
                &mut discovery_bear_out[start..end],
            );
        }
    } else {
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            neighboring_trailing_stop_row_from_slices(
                high,
                low,
                close,
                *params,
                &mut trailing_stop_out[start..end],
                &mut bullish_band_out[start..end],
                &mut bearish_band_out[start..end],
                &mut direction_out[start..end],
                &mut discovery_bull_out[start..end],
                &mut discovery_bear_out[start..end],
            );
        }
    }

    let trailing_stop = unsafe {
        Vec::from_raw_parts(
            trailing_stop_guard.as_mut_ptr() as *mut f64,
            trailing_stop_guard.len(),
            trailing_stop_guard.capacity(),
        )
    };
    let bullish_band = unsafe {
        Vec::from_raw_parts(
            bullish_band_guard.as_mut_ptr() as *mut f64,
            bullish_band_guard.len(),
            bullish_band_guard.capacity(),
        )
    };
    let bearish_band = unsafe {
        Vec::from_raw_parts(
            bearish_band_guard.as_mut_ptr() as *mut f64,
            bearish_band_guard.len(),
            bearish_band_guard.capacity(),
        )
    };
    let direction = unsafe {
        Vec::from_raw_parts(
            direction_guard.as_mut_ptr() as *mut f64,
            direction_guard.len(),
            direction_guard.capacity(),
        )
    };
    let discovery_bull = unsafe {
        Vec::from_raw_parts(
            discovery_bull_guard.as_mut_ptr() as *mut f64,
            discovery_bull_guard.len(),
            discovery_bull_guard.capacity(),
        )
    };
    let discovery_bear = unsafe {
        Vec::from_raw_parts(
            discovery_bear_guard.as_mut_ptr() as *mut f64,
            discovery_bear_guard.len(),
            discovery_bear_guard.capacity(),
        )
    };
    core::mem::forget(trailing_stop_guard);
    core::mem::forget(bullish_band_guard);
    core::mem::forget(bearish_band_guard);
    core::mem::forget(direction_guard);
    core::mem::forget(discovery_bull_guard);
    core::mem::forget(discovery_bear_guard);

    Ok(NeighboringTrailingStopBatchOutput {
        trailing_stop,
        bullish_band,
        bearish_band,
        direction,
        discovery_bull,
        discovery_bear,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn neighboring_trailing_stop_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NeighboringTrailingStopBatchRange,
    kernel: Kernel,
    parallel: bool,
    trailing_stop: &mut [f64],
    bullish_band: &mut [f64],
    bearish_band: &mut [f64],
    direction: &mut [f64],
    discovery_bull: &mut [f64],
    discovery_bear: &mut [f64],
) -> Result<Vec<NeighboringTrailingStopParams>, NeighboringTrailingStopError> {
    let out = neighboring_trailing_stop_batch_inner(high, low, close, sweep, kernel, parallel)?;
    let total = out.rows * out.cols;
    if trailing_stop.len() != total
        || bullish_band.len() != total
        || bearish_band.len() != total
        || direction.len() != total
        || discovery_bull.len() != total
        || discovery_bear.len() != total
    {
        return Err(NeighboringTrailingStopError::OutputLengthMismatch { expected: total });
    }
    trailing_stop.copy_from_slice(&out.trailing_stop);
    bullish_band.copy_from_slice(&out.bullish_band);
    bearish_band.copy_from_slice(&out.bearish_band);
    direction.copy_from_slice(&out.direction);
    discovery_bull.copy_from_slice(&out.discovery_bull);
    discovery_bear.copy_from_slice(&out.discovery_bear);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "neighboring_trailing_stop")]
#[pyo3(signature = (
    high,
    low,
    close,
    buffer_size=DEFAULT_BUFFER_SIZE,
    k=DEFAULT_K,
    percentile=DEFAULT_PERCENTILE,
    smooth=DEFAULT_SMOOTH,
    kernel=None
))]
pub fn neighboring_trailing_stop_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    buffer_size: usize,
    k: usize,
    percentile: f64,
    smooth: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = NeighboringTrailingStopInput::from_slices(
        high,
        low,
        close,
        NeighboringTrailingStopParams {
            buffer_size: Some(buffer_size),
            k: Some(k),
            percentile: Some(percentile),
            smooth: Some(smooth),
        },
    );
    let out = py
        .allow_threads(|| neighboring_trailing_stop_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.trailing_stop.into_pyarray(py),
        out.bullish_band.into_pyarray(py),
        out.bearish_band.into_pyarray(py),
        out.direction.into_pyarray(py),
        out.discovery_bull.into_pyarray(py),
        out.discovery_bear.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "NeighboringTrailingStopStream")]
pub struct NeighboringTrailingStopStreamPy {
    stream: NeighboringTrailingStopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NeighboringTrailingStopStreamPy {
    #[new]
    #[pyo3(signature = (
        buffer_size=DEFAULT_BUFFER_SIZE,
        k=DEFAULT_K,
        percentile=DEFAULT_PERCENTILE,
        smooth=DEFAULT_SMOOTH
    ))]
    fn new(buffer_size: usize, k: usize, percentile: f64, smooth: usize) -> PyResult<Self> {
        let stream = NeighboringTrailingStopStream::try_new(NeighboringTrailingStopParams {
            buffer_size: Some(buffer_size),
            k: Some(k),
            percentile: Some(percentile),
            smooth: Some(smooth),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.stream.update(high, low, close).map(|point| {
            (
                point.trailing_stop,
                point.bullish_band,
                point.bearish_band,
                point.direction,
                point.discovery_bull,
                point.discovery_bear,
            )
        })
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
#[pyfunction(name = "neighboring_trailing_stop_batch")]
#[pyo3(signature = (
    high,
    low,
    close,
    buffer_size_range=(DEFAULT_BUFFER_SIZE, DEFAULT_BUFFER_SIZE, 0),
    k_range=(DEFAULT_K, DEFAULT_K, 0),
    percentile_range=(DEFAULT_PERCENTILE, DEFAULT_PERCENTILE, 0.0),
    smooth_range=(DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0),
    kernel=None
))]
pub fn neighboring_trailing_stop_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    buffer_size_range: (usize, usize, usize),
    k_range: (usize, usize, usize),
    percentile_range: (f64, f64, f64),
    smooth_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = NeighboringTrailingStopBatchRange {
        buffer_size: buffer_size_range,
        k: k_range,
        percentile: percentile_range,
        smooth: smooth_range,
    };
    let combos = expand_grid_neighboring_trailing_stop(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let trailing_stop_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bullish_band_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bearish_band_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let direction_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let discovery_bull_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let discovery_bear_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let trailing_stop_slice = unsafe { trailing_stop_arr.as_slice_mut()? };
    let bullish_band_slice = unsafe { bullish_band_arr.as_slice_mut()? };
    let bearish_band_slice = unsafe { bearish_band_arr.as_slice_mut()? };
    let direction_slice = unsafe { direction_arr.as_slice_mut()? };
    let discovery_bull_slice = unsafe { discovery_bull_arr.as_slice_mut()? };
    let discovery_bear_slice = unsafe { discovery_bear_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            neighboring_trailing_stop_batch_inner_into(
                high,
                low,
                close,
                &sweep,
                batch_kernel.to_non_batch(),
                true,
                trailing_stop_slice,
                bullish_band_slice,
                bearish_band_slice,
                direction_slice,
                discovery_bull_slice,
                discovery_bear_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("trailing_stop", trailing_stop_arr.reshape((rows, cols))?)?;
    dict.set_item("bullish_band", bullish_band_arr.reshape((rows, cols))?)?;
    dict.set_item("bearish_band", bearish_band_arr.reshape((rows, cols))?)?;
    dict.set_item("direction", direction_arr.reshape((rows, cols))?)?;
    dict.set_item("discovery_bull", discovery_bull_arr.reshape((rows, cols))?)?;
    dict.set_item("discovery_bear", discovery_bear_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "buffer_sizes",
        combos
            .iter()
            .map(|combo| combo.buffer_size.unwrap_or(DEFAULT_BUFFER_SIZE) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ks",
        combos
            .iter()
            .map(|combo| combo.k.unwrap_or(DEFAULT_K) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "percentiles",
        combos
            .iter()
            .map(|combo| combo.percentile.unwrap_or(DEFAULT_PERCENTILE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smoothings",
        combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_neighboring_trailing_stop_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(neighboring_trailing_stop_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        neighboring_trailing_stop_batch_py,
        module
    )?)?;
    module.add_class::<NeighboringTrailingStopStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NeighboringTrailingStopJsOutput {
    pub trailing_stop: Vec<f64>,
    pub bullish_band: Vec<f64>,
    pub bearish_band: Vec<f64>,
    pub direction: Vec<f64>,
    pub discovery_bull: Vec<f64>,
    pub discovery_bear: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "neighboring_trailing_stop_js")]
pub fn neighboring_trailing_stop_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buffer_size: usize,
    k: usize,
    percentile: f64,
    smooth: usize,
) -> Result<JsValue, JsValue> {
    let input = NeighboringTrailingStopInput::from_slices(
        high,
        low,
        close,
        NeighboringTrailingStopParams {
            buffer_size: Some(buffer_size),
            k: Some(k),
            percentile: Some(percentile),
            smooth: Some(smooth),
        },
    );
    let out = neighboring_trailing_stop(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&NeighboringTrailingStopJsOutput {
        trailing_stop: out.trailing_stop,
        bullish_band: out.bullish_band,
        bearish_band: out.bearish_band,
        direction: out.direction,
        discovery_bull: out.discovery_bull,
        discovery_bear: out.discovery_bear,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn neighboring_trailing_stop_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn neighboring_trailing_stop_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn has_duplicate_ptrs(ptrs: &[usize]) -> bool {
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            if ptrs[i] == ptrs[j] {
                return true;
            }
        }
    }
    false
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn neighboring_trailing_stop_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    trailing_stop_ptr: *mut f64,
    bullish_band_ptr: *mut f64,
    bearish_band_ptr: *mut f64,
    direction_ptr: *mut f64,
    discovery_bull_ptr: *mut f64,
    discovery_bear_ptr: *mut f64,
    len: usize,
    buffer_size: usize,
    k: usize,
    percentile: f64,
    smooth: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || trailing_stop_ptr.is_null()
        || bullish_band_ptr.is_null()
        || bearish_band_ptr.is_null()
        || direction_ptr.is_null()
        || discovery_bull_ptr.is_null()
        || discovery_bear_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = NeighboringTrailingStopInput::from_slices(
            high,
            low,
            close,
            NeighboringTrailingStopParams {
                buffer_size: Some(buffer_size),
                k: Some(k),
                percentile: Some(percentile),
                smooth: Some(smooth),
            },
        );

        let output_ptrs = [
            trailing_stop_ptr as usize,
            bullish_band_ptr as usize,
            bearish_band_ptr as usize,
            direction_ptr as usize,
            discovery_bull_ptr as usize,
            discovery_bear_ptr as usize,
        ];
        let need_temp = output_ptrs.iter().any(|&ptr| {
            ptr == high_ptr as usize || ptr == low_ptr as usize || ptr == close_ptr as usize
        }) || has_duplicate_ptrs(&output_ptrs);

        if need_temp {
            let mut trailing_stop = vec![0.0; len];
            let mut bullish_band = vec![0.0; len];
            let mut bearish_band = vec![0.0; len];
            let mut direction = vec![0.0; len];
            let mut discovery_bull = vec![0.0; len];
            let mut discovery_bear = vec![0.0; len];
            neighboring_trailing_stop_into_slices(
                &mut trailing_stop,
                &mut bullish_band,
                &mut bearish_band,
                &mut direction,
                &mut discovery_bull,
                &mut discovery_bear,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(trailing_stop_ptr, len).copy_from_slice(&trailing_stop);
            std::slice::from_raw_parts_mut(bullish_band_ptr, len).copy_from_slice(&bullish_band);
            std::slice::from_raw_parts_mut(bearish_band_ptr, len).copy_from_slice(&bearish_band);
            std::slice::from_raw_parts_mut(direction_ptr, len).copy_from_slice(&direction);
            std::slice::from_raw_parts_mut(discovery_bull_ptr, len)
                .copy_from_slice(&discovery_bull);
            std::slice::from_raw_parts_mut(discovery_bear_ptr, len)
                .copy_from_slice(&discovery_bear);
        } else {
            neighboring_trailing_stop_into_slices(
                std::slice::from_raw_parts_mut(trailing_stop_ptr, len),
                std::slice::from_raw_parts_mut(bullish_band_ptr, len),
                std::slice::from_raw_parts_mut(bearish_band_ptr, len),
                std::slice::from_raw_parts_mut(direction_ptr, len),
                std::slice::from_raw_parts_mut(discovery_bull_ptr, len),
                std::slice::from_raw_parts_mut(discovery_bear_ptr, len),
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
pub struct NeighboringTrailingStopBatchJsConfig {
    pub buffer_size_range: Option<(usize, usize, usize)>,
    pub k_range: Option<(usize, usize, usize)>,
    pub percentile_range: Option<(f64, f64, f64)>,
    pub smooth_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NeighboringTrailingStopBatchJsOutput {
    pub trailing_stop: Vec<f64>,
    pub bullish_band: Vec<f64>,
    pub bearish_band: Vec<f64>,
    pub direction: Vec<f64>,
    pub discovery_bull: Vec<f64>,
    pub discovery_bear: Vec<f64>,
    pub combos: Vec<NeighboringTrailingStopParams>,
    pub buffer_sizes: Vec<usize>,
    pub ks: Vec<usize>,
    pub percentiles: Vec<f64>,
    pub smoothings: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "neighboring_trailing_stop_batch_js")]
pub fn neighboring_trailing_stop_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: NeighboringTrailingStopBatchJsConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = NeighboringTrailingStopBatchRange {
        buffer_size: config.buffer_size_range.unwrap_or((
            DEFAULT_BUFFER_SIZE,
            DEFAULT_BUFFER_SIZE,
            0,
        )),
        k: config.k_range.unwrap_or((DEFAULT_K, DEFAULT_K, 0)),
        percentile: config.percentile_range.unwrap_or((
            DEFAULT_PERCENTILE,
            DEFAULT_PERCENTILE,
            0.0,
        )),
        smooth: config
            .smooth_range
            .unwrap_or((DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0)),
    };
    let output =
        neighboring_trailing_stop_batch_inner(high, low, close, &sweep, Kernel::Auto, false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&NeighboringTrailingStopBatchJsOutput {
        trailing_stop: output.trailing_stop,
        bullish_band: output.bullish_band,
        bearish_band: output.bearish_band,
        direction: output.direction,
        discovery_bull: output.discovery_bull,
        discovery_bear: output.discovery_bear,
        buffer_sizes: output
            .combos
            .iter()
            .map(|combo| combo.buffer_size.unwrap_or(DEFAULT_BUFFER_SIZE))
            .collect(),
        ks: output
            .combos
            .iter()
            .map(|combo| combo.k.unwrap_or(DEFAULT_K))
            .collect(),
        percentiles: output
            .combos
            .iter()
            .map(|combo| combo.percentile.unwrap_or(DEFAULT_PERCENTILE))
            .collect(),
        smoothings: output
            .combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH))
            .collect(),
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn neighboring_trailing_stop_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    trailing_stop_ptr: *mut f64,
    bullish_band_ptr: *mut f64,
    bearish_band_ptr: *mut f64,
    direction_ptr: *mut f64,
    discovery_bull_ptr: *mut f64,
    discovery_bear_ptr: *mut f64,
    len: usize,
    buffer_size_start: usize,
    buffer_size_end: usize,
    buffer_size_step: usize,
    k_start: usize,
    k_end: usize,
    k_step: usize,
    percentile_start: f64,
    percentile_end: f64,
    percentile_step: f64,
    smooth_start: usize,
    smooth_end: usize,
    smooth_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || trailing_stop_ptr.is_null()
        || bullish_band_ptr.is_null()
        || bearish_band_ptr.is_null()
        || direction_ptr.is_null()
        || discovery_bull_ptr.is_null()
        || discovery_bear_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = NeighboringTrailingStopBatchRange {
        buffer_size: (buffer_size_start, buffer_size_end, buffer_size_step),
        k: (k_start, k_end, k_step),
        percentile: (percentile_start, percentile_end, percentile_step),
        smooth: (smooth_start, smooth_end, smooth_step),
    };
    let combos = expand_grid_neighboring_trailing_stop(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        neighboring_trailing_stop_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
            false,
            std::slice::from_raw_parts_mut(trailing_stop_ptr, total),
            std::slice::from_raw_parts_mut(bullish_band_ptr, total),
            std::slice::from_raw_parts_mut(bearish_band_ptr, total),
            std::slice::from_raw_parts_mut(direction_ptr, total),
            std::slice::from_raw_parts_mut(discovery_bull_ptr, total),
            std::slice::from_raw_parts_mut(discovery_bear_ptr, total),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn neighboring_trailing_stop_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buffer_size: usize,
    k: usize,
    percentile: f64,
    smooth: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = neighboring_trailing_stop_js(high, low, close, buffer_size, k, percentile, smooth)?;
    crate::write_wasm_object_f64_outputs("neighboring_trailing_stop_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn neighboring_trailing_stop_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = neighboring_trailing_stop_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "neighboring_trailing_stop_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(length);
        let mut low = Vec::with_capacity(length);
        let mut close = Vec::with_capacity(length);
        for i in 0..length {
            let x = i as f64;
            let open = 100.0 + x * 0.04 + (x * 0.07).sin();
            let c = open + (x * 0.11).cos() * 0.85;
            high.push(open.max(c) + 0.55 + (x * 0.03).sin().abs() * 0.2);
            low.push(open.min(c) - 0.55 - (x * 0.05).cos().abs() * 0.2);
            close.push(c);
        }
        (high, low, close)
    }

    fn assert_series_eq(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((*a - *b).abs() <= tol, "left={a}, right={b}");
        }
    }

    #[test]
    fn neighboring_trailing_stop_output_contract() -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(256);
        let input = NeighboringTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            NeighboringTrailingStopParams::default(),
        );
        let out = neighboring_trailing_stop_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.trailing_stop.len(), close.len());
        assert_eq!(out.bullish_band.len(), close.len());
        assert_eq!(out.bearish_band.len(), close.len());
        assert_eq!(out.direction.len(), close.len());
        assert_eq!(out.discovery_bull.len(), close.len());
        assert_eq!(out.discovery_bear.len(), close.len());
        assert!(out.direction.iter().any(|v| *v == 1.0 || *v == -1.0));
        Ok(())
    }

    #[test]
    fn neighboring_trailing_stop_rejects_invalid_params() {
        let (high, low, close) = sample_ohlc(64);
        let err = neighboring_trailing_stop_with_kernel(
            &NeighboringTrailingStopInput::from_slices(
                &high,
                &low,
                &close,
                NeighboringTrailingStopParams {
                    buffer_size: Some(50),
                    k: Some(DEFAULT_K),
                    percentile: Some(DEFAULT_PERCENTILE),
                    smooth: Some(DEFAULT_SMOOTH),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            NeighboringTrailingStopError::InvalidBufferSize { .. }
        ));

        let err = neighboring_trailing_stop_with_kernel(
            &NeighboringTrailingStopInput::from_slices(
                &high,
                &low,
                &close,
                NeighboringTrailingStopParams {
                    buffer_size: Some(DEFAULT_BUFFER_SIZE),
                    k: Some(2),
                    percentile: Some(DEFAULT_PERCENTILE),
                    smooth: Some(DEFAULT_SMOOTH),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(err, NeighboringTrailingStopError::InvalidK { .. }));
    }

    #[test]
    fn neighboring_trailing_stop_builder_matches_direct() -> Result<(), Box<dyn std::error::Error>>
    {
        let (high, low, close) = sample_ohlc(220);
        let direct = neighboring_trailing_stop_with_kernel(
            &NeighboringTrailingStopInput::from_slices(
                &high,
                &low,
                &close,
                NeighboringTrailingStopParams {
                    buffer_size: Some(180),
                    k: Some(30),
                    percentile: Some(87.5),
                    smooth: Some(4),
                },
            ),
            Kernel::Scalar,
        )?;
        let built = NeighboringTrailingStopBuilder::new()
            .buffer_size(180)
            .k(30)
            .percentile(87.5)
            .smooth(4)
            .kernel(Kernel::Scalar)
            .apply_slices(&high, &low, &close)?;
        assert_series_eq(&direct.trailing_stop, &built.trailing_stop, 1e-12);
        assert_series_eq(&direct.bullish_band, &built.bullish_band, 1e-12);
        assert_series_eq(&direct.bearish_band, &built.bearish_band, 1e-12);
        Ok(())
    }

    #[test]
    fn neighboring_trailing_stop_stream_matches_batch_with_reset(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (mut high, mut low, mut close) = sample_ohlc(240);
        high[120] = f64::NAN;
        low[120] = f64::NAN;
        close[120] = f64::NAN;

        let params = NeighboringTrailingStopParams {
            buffer_size: Some(180),
            k: Some(25),
            percentile: Some(92.0),
            smooth: Some(4),
        };
        let batch = neighboring_trailing_stop_with_kernel(
            &NeighboringTrailingStopInput::from_slices(&high, &low, &close, params.clone()),
            Kernel::Scalar,
        )?;
        let mut stream = NeighboringTrailingStopStream::try_new(params)?;

        let mut trailing_stop = Vec::with_capacity(close.len());
        let mut bullish_band = Vec::with_capacity(close.len());
        let mut bearish_band = Vec::with_capacity(close.len());
        let mut direction = Vec::with_capacity(close.len());
        let mut discovery_bull = Vec::with_capacity(close.len());
        let mut discovery_bear = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            if let Some(point) = stream.update(high[i], low[i], close[i]) {
                trailing_stop.push(point.trailing_stop);
                bullish_band.push(point.bullish_band);
                bearish_band.push(point.bearish_band);
                direction.push(point.direction);
                discovery_bull.push(point.discovery_bull);
                discovery_bear.push(point.discovery_bear);
            } else {
                trailing_stop.push(f64::NAN);
                bullish_band.push(f64::NAN);
                bearish_band.push(f64::NAN);
                direction.push(f64::NAN);
                discovery_bull.push(f64::NAN);
                discovery_bear.push(f64::NAN);
            }
        }

        assert_eq!(stream.get_warmup_period(), 0);
        assert_series_eq(&trailing_stop, &batch.trailing_stop, 1e-12);
        assert_series_eq(&bullish_band, &batch.bullish_band, 1e-12);
        assert_series_eq(&bearish_band, &batch.bearish_band, 1e-12);
        assert_series_eq(&direction, &batch.direction, 1e-12);
        assert_series_eq(&discovery_bull, &batch.discovery_bull, 1e-12);
        assert_series_eq(&discovery_bear, &batch.discovery_bear, 1e-12);
        Ok(())
    }

    #[test]
    fn neighboring_trailing_stop_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(180);
        let input = NeighboringTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            NeighboringTrailingStopParams {
                buffer_size: Some(160),
                k: Some(20),
                percentile: Some(88.0),
                smooth: Some(3),
            },
        );
        let api = neighboring_trailing_stop_with_kernel(&input, Kernel::Scalar)?;
        let mut trailing_stop = vec![0.0; close.len()];
        let mut bullish_band = vec![0.0; close.len()];
        let mut bearish_band = vec![0.0; close.len()];
        let mut direction = vec![0.0; close.len()];
        let mut discovery_bull = vec![0.0; close.len()];
        let mut discovery_bear = vec![0.0; close.len()];
        neighboring_trailing_stop_into_slices(
            &mut trailing_stop,
            &mut bullish_band,
            &mut bearish_band,
            &mut direction,
            &mut discovery_bull,
            &mut discovery_bear,
            &input,
            Kernel::Scalar,
        )?;
        assert_series_eq(&trailing_stop, &api.trailing_stop, 1e-12);
        assert_series_eq(&bullish_band, &api.bullish_band, 1e-12);
        assert_series_eq(&bearish_band, &api.bearish_band, 1e-12);
        assert_series_eq(&direction, &api.direction, 1e-12);
        assert_series_eq(&discovery_bull, &api.discovery_bull, 1e-12);
        assert_series_eq(&discovery_bear, &api.discovery_bear, 1e-12);
        Ok(())
    }

    #[test]
    fn neighboring_trailing_stop_batch_single_param_matches_single(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(160);
        let single = neighboring_trailing_stop_with_kernel(
            &NeighboringTrailingStopInput::from_slices(
                &high,
                &low,
                &close,
                NeighboringTrailingStopParams {
                    buffer_size: Some(180),
                    k: Some(35),
                    percentile: Some(91.0),
                    smooth: Some(4),
                },
            ),
            Kernel::Scalar,
        )?;
        let batch = neighboring_trailing_stop_batch_with_kernel(
            &high,
            &low,
            &close,
            &NeighboringTrailingStopBatchRange {
                buffer_size: (180, 180, 0),
                k: (35, 35, 0),
                percentile: (91.0, 91.0, 0.0),
                smooth: (4, 4, 0),
            },
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        let row = batch.row_slices(0).unwrap();
        assert_series_eq(row.0, single.trailing_stop.as_slice(), 1e-12);
        assert_series_eq(row.1, single.bullish_band.as_slice(), 1e-12);
        assert_series_eq(row.2, single.bearish_band.as_slice(), 1e-12);
        Ok(())
    }

    #[test]
    fn neighboring_trailing_stop_batch_metadata() -> Result<(), Box<dyn std::error::Error>> {
        let (high, low, close) = sample_ohlc(96);
        let batch = neighboring_trailing_stop_batch_with_kernel(
            &high,
            &low,
            &close,
            &NeighboringTrailingStopBatchRange {
                buffer_size: (150, 150, 0),
                k: (20, 24, 4),
                percentile: (85.0, 90.0, 5.0),
                smooth: (3, 3, 0),
            },
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, close.len());
        assert_eq!(batch.combos[0].buffer_size, Some(150));
        assert_eq!(batch.combos[0].k, Some(20));
        assert_eq!(batch.combos[1].percentile, Some(90.0));
        Ok(())
    }
}
