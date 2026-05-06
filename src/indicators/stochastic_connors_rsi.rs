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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const SCALE_100: f64 = 100.0;
const EPS: f64 = 1.0e-14;

impl<'a> AsRef<[f64]> for StochasticConnorsRsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            StochasticConnorsRsiData::Slice(slice) => slice,
            StochasticConnorsRsiData::Candles { candles, source } if *source == "close" => {
                &candles.close
            }
            StochasticConnorsRsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum StochasticConnorsRsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct StochasticConnorsRsiOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StochasticConnorsRsiParams {
    pub stoch_length: Option<usize>,
    pub smooth_k: Option<usize>,
    pub smooth_d: Option<usize>,
    pub rsi_length: Option<usize>,
    pub updown_length: Option<usize>,
    pub roc_length: Option<usize>,
}

impl Default for StochasticConnorsRsiParams {
    fn default() -> Self {
        Self {
            stoch_length: Some(3),
            smooth_k: Some(3),
            smooth_d: Some(3),
            rsi_length: Some(3),
            updown_length: Some(2),
            roc_length: Some(100),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochasticConnorsRsiInput<'a> {
    pub data: StochasticConnorsRsiData<'a>,
    pub params: StochasticConnorsRsiParams,
}

impl<'a> StochasticConnorsRsiInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: StochasticConnorsRsiParams,
    ) -> Self {
        Self {
            data: StochasticConnorsRsiData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: StochasticConnorsRsiParams) -> Self {
        Self {
            data: StochasticConnorsRsiData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", StochasticConnorsRsiParams::default())
    }

    #[inline]
    pub fn get_stoch_length(&self) -> usize {
        self.params.stoch_length.unwrap_or(3)
    }

    #[inline]
    pub fn get_smooth_k(&self) -> usize {
        self.params.smooth_k.unwrap_or(3)
    }

    #[inline]
    pub fn get_smooth_d(&self) -> usize {
        self.params.smooth_d.unwrap_or(3)
    }

    #[inline]
    pub fn get_rsi_length(&self) -> usize {
        self.params.rsi_length.unwrap_or(3)
    }

    #[inline]
    pub fn get_updown_length(&self) -> usize {
        self.params.updown_length.unwrap_or(2)
    }

    #[inline]
    pub fn get_roc_length(&self) -> usize {
        self.params.roc_length.unwrap_or(100)
    }
}

#[derive(Clone, Debug)]
pub struct StochasticConnorsRsiBuilder {
    stoch_length: Option<usize>,
    smooth_k: Option<usize>,
    smooth_d: Option<usize>,
    rsi_length: Option<usize>,
    updown_length: Option<usize>,
    roc_length: Option<usize>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for StochasticConnorsRsiBuilder {
    fn default() -> Self {
        Self {
            stoch_length: None,
            smooth_k: None,
            smooth_d: None,
            rsi_length: None,
            updown_length: None,
            roc_length: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochasticConnorsRsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn stoch_length(mut self, value: usize) -> Self {
        self.stoch_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth_k(mut self, value: usize) -> Self {
        self.smooth_k = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth_d(mut self, value: usize) -> Self {
        self.smooth_d = Some(value);
        self
    }

    #[inline(always)]
    pub fn rsi_length(mut self, value: usize) -> Self {
        self.rsi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn updown_length(mut self, value: usize) -> Self {
        self.updown_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn roc_length(mut self, value: usize) -> Self {
        self.roc_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
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
    ) -> Result<StochasticConnorsRsiOutput, StochasticConnorsRsiError> {
        let input = StochasticConnorsRsiInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("close"),
            StochasticConnorsRsiParams {
                stoch_length: self.stoch_length,
                smooth_k: self.smooth_k,
                smooth_d: self.smooth_d,
                rsi_length: self.rsi_length,
                updown_length: self.updown_length,
                roc_length: self.roc_length,
            },
        );
        stochastic_connors_rsi_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<StochasticConnorsRsiOutput, StochasticConnorsRsiError> {
        let input = StochasticConnorsRsiInput::from_slice(
            data,
            StochasticConnorsRsiParams {
                stoch_length: self.stoch_length,
                smooth_k: self.smooth_k,
                smooth_d: self.smooth_d,
                rsi_length: self.rsi_length,
                updown_length: self.updown_length,
                roc_length: self.roc_length,
            },
        );
        stochastic_connors_rsi_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<StochasticConnorsRsiStream, StochasticConnorsRsiError> {
        StochasticConnorsRsiStream::try_new(StochasticConnorsRsiParams {
            stoch_length: self.stoch_length,
            smooth_k: self.smooth_k,
            smooth_d: self.smooth_d,
            rsi_length: self.rsi_length,
            updown_length: self.updown_length,
            roc_length: self.roc_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum StochasticConnorsRsiError {
    #[error("stochastic_connors_rsi: Empty input data.")]
    EmptyInputData,
    #[error("stochastic_connors_rsi: All input data values are NaN.")]
    AllValuesNaN,
    #[error("stochastic_connors_rsi: Invalid stoch_length: stoch_length = {stoch_length}, data length = {data_len}")]
    InvalidStochLength {
        stoch_length: usize,
        data_len: usize,
    },
    #[error(
        "stochastic_connors_rsi: Invalid smooth_k: smooth_k = {smooth_k}, data length = {data_len}"
    )]
    InvalidSmoothK { smooth_k: usize, data_len: usize },
    #[error(
        "stochastic_connors_rsi: Invalid smooth_d: smooth_d = {smooth_d}, data length = {data_len}"
    )]
    InvalidSmoothD { smooth_d: usize, data_len: usize },
    #[error("stochastic_connors_rsi: Invalid rsi_length: rsi_length = {rsi_length}, data length = {data_len}")]
    InvalidRsiLength { rsi_length: usize, data_len: usize },
    #[error("stochastic_connors_rsi: Invalid updown_length: updown_length = {updown_length}, data length = {data_len}")]
    InvalidUpdownLength {
        updown_length: usize,
        data_len: usize,
    },
    #[error("stochastic_connors_rsi: Invalid roc_length: roc_length = {roc_length}, data length = {data_len}")]
    InvalidRocLength { roc_length: usize, data_len: usize },
    #[error("stochastic_connors_rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stochastic_connors_rsi: Output length mismatch: expected {expected}, got k={k_len}, d={d_len}")]
    OutputLengthMismatch {
        expected: usize,
        k_len: usize,
        d_len: usize,
    },
    #[error("stochastic_connors_rsi: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("stochastic_connors_rsi: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct WilderRsiState {
    period: usize,
    inv_p: f64,
    beta: f64,
    has_prev: bool,
    prev: f64,
    seed_count: usize,
    sum_gain: f64,
    sum_loss: f64,
    avg_gain: f64,
    avg_loss: f64,
    seeded: bool,
}

impl WilderRsiState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let inv_p = 1.0 / period as f64;
        Self {
            period,
            inv_p,
            beta: 1.0 - inv_p,
            has_prev: false,
            prev: f64::NAN,
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
        self.has_prev = false;
        self.prev = f64::NAN;
        self.seed_count = 0;
        self.sum_gain = 0.0;
        self.sum_loss = 0.0;
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.seeded = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !self.has_prev {
            self.prev = value;
            self.has_prev = true;
            return None;
        }

        let delta = value - self.prev;
        self.prev = value;

        if !self.seeded {
            self.sum_gain += delta.max(0.0);
            self.sum_loss += (-delta).max(0.0);
            self.seed_count += 1;
            if self.seed_count == self.period {
                self.seeded = true;
                self.avg_gain = self.sum_gain * self.inv_p;
                self.avg_loss = self.sum_loss * self.inv_p;
                let denom = self.avg_gain + self.avg_loss;
                return Some(if denom == 0.0 {
                    50.0
                } else {
                    SCALE_100 * self.avg_gain / denom
                });
            }
            return None;
        }

        let gain = delta.max(0.0);
        let loss = (-delta).max(0.0);
        self.avg_gain = self.avg_gain.mul_add(self.beta, self.inv_p * gain);
        self.avg_loss = self.avg_loss.mul_add(self.beta, self.inv_p * loss);
        let denom = self.avg_gain + self.avg_loss;
        Some(if denom == 0.0 {
            50.0
        } else {
            SCALE_100 * self.avg_gain / denom
        })
    }
}

#[derive(Clone, Debug)]
struct SmaState {
    period: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
}

impl SmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buf: vec![0.0; period],
            head: 0,
            len: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.len < self.period {
            self.buf[self.head] = value;
            self.sum += value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            self.len += 1;
            if self.len == self.period {
                return Some(self.sum / self.period as f64);
            }
            return None;
        }

        let old = self.buf[self.head];
        self.buf[self.head] = value;
        self.sum += value - old;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        Some(self.sum / self.period as f64)
    }
}

#[derive(Clone, Debug)]
struct StochasticConnorsRsiState {
    stoch_length: usize,
    roc_length: usize,
    prev_source: Option<f64>,
    streak: i64,
    src_rsi: WilderRsiState,
    streak_rsi: WilderRsiState,
    roc_window: VecDeque<f64>,
    crsi_seen: usize,
    minq: VecDeque<(usize, f64)>,
    maxq: VecDeque<(usize, f64)>,
    k_sma: SmaState,
    d_sma: SmaState,
}

impl StochasticConnorsRsiState {
    #[inline(always)]
    fn new(params: &StochasticConnorsRsiParams) -> Self {
        Self {
            stoch_length: params.stoch_length.unwrap_or(3),
            roc_length: params.roc_length.unwrap_or(100),
            prev_source: None,
            streak: 0,
            src_rsi: WilderRsiState::new(params.rsi_length.unwrap_or(3)),
            streak_rsi: WilderRsiState::new(params.updown_length.unwrap_or(2)),
            roc_window: VecDeque::with_capacity(params.roc_length.unwrap_or(100)),
            crsi_seen: 0,
            minq: VecDeque::with_capacity(params.stoch_length.unwrap_or(3)),
            maxq: VecDeque::with_capacity(params.stoch_length.unwrap_or(3)),
            k_sma: SmaState::new(params.smooth_k.unwrap_or(3)),
            d_sma: SmaState::new(params.smooth_d.unwrap_or(3)),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_source = None;
        self.streak = 0;
        self.src_rsi.reset();
        self.streak_rsi.reset();
        self.roc_window.clear();
        self.crsi_seen = 0;
        self.minq.clear();
        self.maxq.clear();
        self.k_sma.reset();
        self.d_sma.reset();
    }

    #[inline(always)]
    fn update(&mut self, source: f64) -> (f64, f64) {
        let prev = self.prev_source;
        self.streak = match prev {
            Some(prev_value) if source > prev_value => {
                if self.streak >= 0 {
                    self.streak + 1
                } else {
                    1
                }
            }
            Some(prev_value) if source < prev_value => {
                if self.streak <= 0 {
                    self.streak - 1
                } else {
                    -1
                }
            }
            Some(_) | None => 0,
        };
        self.prev_source = Some(source);

        let src_rsi = self.src_rsi.update(source);
        let streak_rsi = self.streak_rsi.update(self.streak as f64);
        let percent_rank = match prev {
            Some(prev_value) => {
                let roc = if prev_value == 0.0 || !prev_value.is_finite() {
                    0.0
                } else {
                    (source / prev_value).mul_add(SCALE_100, -SCALE_100)
                };
                self.roc_window.push_back(roc);
                if self.roc_window.len() > self.roc_length {
                    self.roc_window.pop_front();
                }
                if self.roc_window.len() == self.roc_length {
                    let count = self
                        .roc_window
                        .iter()
                        .filter(|&&value| value <= roc)
                        .count();
                    Some(SCALE_100 * count as f64 / self.roc_length as f64)
                } else {
                    None
                }
            }
            None => None,
        };

        let crsi = match (src_rsi, streak_rsi, percent_rank) {
            (Some(a), Some(b), Some(c)) => (a + b + c) / 3.0,
            _ => return (f64::NAN, f64::NAN),
        };

        let idx = self.crsi_seen;
        self.crsi_seen += 1;
        while let Some(&(_, value)) = self.minq.back() {
            if value >= crsi {
                self.minq.pop_back();
            } else {
                break;
            }
        }
        self.minq.push_back((idx, crsi));
        while let Some(&(_, value)) = self.maxq.back() {
            if value <= crsi {
                self.maxq.pop_back();
            } else {
                break;
            }
        }
        self.maxq.push_back((idx, crsi));

        if self.crsi_seen < self.stoch_length {
            return (f64::NAN, f64::NAN);
        }

        let window_start = self.crsi_seen - self.stoch_length;
        while let Some(&(old_idx, _)) = self.minq.front() {
            if old_idx < window_start {
                self.minq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(old_idx, _)) = self.maxq.front() {
            if old_idx < window_start {
                self.maxq.pop_front();
            } else {
                break;
            }
        }

        let ll = self.minq.front().map(|&(_, value)| value).unwrap_or(crsi);
        let hh = self.maxq.front().map(|&(_, value)| value).unwrap_or(crsi);
        let denom = hh - ll;
        let raw_k = if denom.abs() < EPS {
            0.0
        } else {
            (crsi - ll).mul_add(SCALE_100 / denom, 0.0)
        };

        let k = match self.k_sma.update(raw_k) {
            Some(value) => value,
            None => return (f64::NAN, f64::NAN),
        };
        let d = self.d_sma.update(k).unwrap_or(f64::NAN);
        (k, d)
    }

    #[inline(always)]
    fn update_reset_on_nan(&mut self, source: f64) -> (f64, f64) {
        if !source.is_finite() {
            self.reset();
            return (f64::NAN, f64::NAN);
        }
        self.update(source)
    }
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> Option<usize> {
    data.iter().position(|value| value.is_finite())
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in data {
        if value.is_finite() {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 0;
        }
    }
    best
}

#[inline(always)]
fn max_component_len(input: &StochasticConnorsRsiInput<'_>) -> usize {
    input
        .get_rsi_length()
        .max(input.get_updown_length())
        .max(input.get_roc_length())
}

#[inline(always)]
fn k_warmup(first: usize, input: &StochasticConnorsRsiInput<'_>) -> usize {
    first + max_component_len(input) + input.get_stoch_length() + input.get_smooth_k() - 2
}

#[inline(always)]
fn d_warmup(first: usize, input: &StochasticConnorsRsiInput<'_>) -> usize {
    k_warmup(first, input) + input.get_smooth_d() - 1
}

fn stochastic_connors_rsi_prepare(
    input: &StochasticConnorsRsiInput<'_>,
) -> Result<usize, StochasticConnorsRsiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(StochasticConnorsRsiError::EmptyInputData);
    }

    let stoch_length = input.get_stoch_length();
    let smooth_k = input.get_smooth_k();
    let smooth_d = input.get_smooth_d();
    let rsi_length = input.get_rsi_length();
    let updown_length = input.get_updown_length();
    let roc_length = input.get_roc_length();

    if stoch_length == 0 || stoch_length > len {
        return Err(StochasticConnorsRsiError::InvalidStochLength {
            stoch_length,
            data_len: len,
        });
    }
    if smooth_k == 0 || smooth_k > len {
        return Err(StochasticConnorsRsiError::InvalidSmoothK {
            smooth_k,
            data_len: len,
        });
    }
    if smooth_d == 0 || smooth_d > len {
        return Err(StochasticConnorsRsiError::InvalidSmoothD {
            smooth_d,
            data_len: len,
        });
    }
    if rsi_length == 0 || rsi_length > len {
        return Err(StochasticConnorsRsiError::InvalidRsiLength {
            rsi_length,
            data_len: len,
        });
    }
    if updown_length == 0 || updown_length > len {
        return Err(StochasticConnorsRsiError::InvalidUpdownLength {
            updown_length,
            data_len: len,
        });
    }
    if roc_length == 0 || roc_length > len {
        return Err(StochasticConnorsRsiError::InvalidRocLength {
            roc_length,
            data_len: len,
        });
    }

    let first = first_valid_value(data).ok_or(StochasticConnorsRsiError::AllValuesNaN)?;
    let needed = max_component_len(input) + stoch_length + smooth_k + smooth_d - 1;
    let valid = longest_valid_run(&data[first..]);
    if valid < needed {
        return Err(StochasticConnorsRsiError::NotEnoughValidData { needed, valid });
    }
    Ok(first)
}

fn stochastic_connors_rsi_compute_into(
    data: &[f64],
    params: &StochasticConnorsRsiParams,
    out_k: &mut [f64],
    out_d: &mut [f64],
) {
    let mut state = StochasticConnorsRsiState::new(params);
    for (idx, &value) in data.iter().enumerate() {
        let (k, d) = state.update_reset_on_nan(value);
        out_k[idx] = k;
        out_d[idx] = d;
    }
}

#[inline]
pub fn stochastic_connors_rsi(
    input: &StochasticConnorsRsiInput<'_>,
) -> Result<StochasticConnorsRsiOutput, StochasticConnorsRsiError> {
    stochastic_connors_rsi_with_kernel(input, Kernel::Auto)
}

pub fn stochastic_connors_rsi_with_kernel(
    input: &StochasticConnorsRsiInput<'_>,
    _kernel: Kernel,
) -> Result<StochasticConnorsRsiOutput, StochasticConnorsRsiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    stochastic_connors_rsi_prepare(input)?;
    let mut k = alloc_uninit_f64(len);
    let mut d = alloc_uninit_f64(len);
    stochastic_connors_rsi_compute_into(data, &input.params, &mut k, &mut d);
    Ok(StochasticConnorsRsiOutput { k, d })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn stochastic_connors_rsi_into(
    input: &StochasticConnorsRsiInput<'_>,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<(), StochasticConnorsRsiError> {
    stochastic_connors_rsi_into_slice(out_k, out_d, input, Kernel::Auto)
}

pub fn stochastic_connors_rsi_into_slice(
    out_k: &mut [f64],
    out_d: &mut [f64],
    input: &StochasticConnorsRsiInput<'_>,
    _kernel: Kernel,
) -> Result<(), StochasticConnorsRsiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if out_k.len() != len || out_d.len() != len {
        return Err(StochasticConnorsRsiError::OutputLengthMismatch {
            expected: len,
            k_len: out_k.len(),
            d_len: out_d.len(),
        });
    }
    stochastic_connors_rsi_prepare(input)?;
    stochastic_connors_rsi_compute_into(data, &input.params, out_k, out_d);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct StochasticConnorsRsiStream {
    state: StochasticConnorsRsiState,
}

impl StochasticConnorsRsiStream {
    pub fn try_new(params: StochasticConnorsRsiParams) -> Result<Self, StochasticConnorsRsiError> {
        let stoch_length = params.stoch_length.unwrap_or(3);
        let smooth_k = params.smooth_k.unwrap_or(3);
        let smooth_d = params.smooth_d.unwrap_or(3);
        let rsi_length = params.rsi_length.unwrap_or(3);
        let updown_length = params.updown_length.unwrap_or(2);
        let roc_length = params.roc_length.unwrap_or(100);
        if stoch_length == 0 {
            return Err(StochasticConnorsRsiError::InvalidStochLength {
                stoch_length,
                data_len: 0,
            });
        }
        if smooth_k == 0 {
            return Err(StochasticConnorsRsiError::InvalidSmoothK {
                smooth_k,
                data_len: 0,
            });
        }
        if smooth_d == 0 {
            return Err(StochasticConnorsRsiError::InvalidSmoothD {
                smooth_d,
                data_len: 0,
            });
        }
        if rsi_length == 0 {
            return Err(StochasticConnorsRsiError::InvalidRsiLength {
                rsi_length,
                data_len: 0,
            });
        }
        if updown_length == 0 {
            return Err(StochasticConnorsRsiError::InvalidUpdownLength {
                updown_length,
                data_len: 0,
            });
        }
        if roc_length == 0 {
            return Err(StochasticConnorsRsiError::InvalidRocLength {
                roc_length,
                data_len: 0,
            });
        }
        Ok(Self {
            state: StochasticConnorsRsiState::new(&params),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let out = self.state.update(value);
        if out.0.is_finite() || out.1.is_finite() {
            Some(out)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn update_reset_on_nan(&mut self, value: f64) -> Option<(f64, f64)> {
        let out = self.state.update_reset_on_nan(value);
        if out.0.is_finite() || out.1.is_finite() {
            Some(out)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct StochasticConnorsRsiBatchRange {
    pub stoch_length: (usize, usize, usize),
    pub smooth_k: (usize, usize, usize),
    pub smooth_d: (usize, usize, usize),
    pub rsi_length: (usize, usize, usize),
    pub updown_length: (usize, usize, usize),
    pub roc_length: (usize, usize, usize),
}

impl Default for StochasticConnorsRsiBatchRange {
    fn default() -> Self {
        Self {
            stoch_length: (3, 3, 0),
            smooth_k: (3, 3, 0),
            smooth_d: (3, 3, 0),
            rsi_length: (3, 3, 0),
            updown_length: (2, 2, 0),
            roc_length: (100, 100, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StochasticConnorsRsiBatchBuilder {
    range: StochasticConnorsRsiBatchRange,
    source: String,
    kernel: Kernel,
}

impl Default for StochasticConnorsRsiBatchBuilder {
    fn default() -> Self {
        Self {
            range: StochasticConnorsRsiBatchRange::default(),
            source: "close".to_string(),
            kernel: Kernel::Auto,
        }
    }
}

impl StochasticConnorsRsiBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = value.into();
        self
    }

    #[inline(always)]
    pub fn stoch_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.stoch_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smooth_k_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth_k = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smooth_d_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth_d = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn rsi_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn updown_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.updown_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn roc_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.roc_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn stoch_length_static(mut self, value: usize) -> Self {
        self.range.stoch_length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn smooth_k_static(mut self, value: usize) -> Self {
        self.range.smooth_k = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn smooth_d_static(mut self, value: usize) -> Self {
        self.range.smooth_d = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn rsi_length_static(mut self, value: usize) -> Self {
        self.range.rsi_length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn updown_length_static(mut self, value: usize) -> Self {
        self.range.updown_length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn roc_length_static(mut self, value: usize) -> Self {
        self.range.roc_length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<StochasticConnorsRsiBatchOutput, StochasticConnorsRsiError> {
        stochastic_connors_rsi_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<StochasticConnorsRsiBatchOutput, StochasticConnorsRsiError> {
        let data = source_type(candles, &self.source);
        self.apply_slice(data)
    }
}

#[derive(Clone, Debug)]
pub struct StochasticConnorsRsiBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub combos: Vec<StochasticConnorsRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl StochasticConnorsRsiBatchOutput {
    pub fn row_for_params(&self, params: &StochasticConnorsRsiParams) -> Option<usize> {
        let stoch_length = params.stoch_length.unwrap_or(3);
        let smooth_k = params.smooth_k.unwrap_or(3);
        let smooth_d = params.smooth_d.unwrap_or(3);
        let rsi_length = params.rsi_length.unwrap_or(3);
        let updown_length = params.updown_length.unwrap_or(2);
        let roc_length = params.roc_length.unwrap_or(100);
        self.combos.iter().position(|combo| {
            combo.stoch_length.unwrap_or(3) == stoch_length
                && combo.smooth_k.unwrap_or(3) == smooth_k
                && combo.smooth_d.unwrap_or(3) == smooth_d
                && combo.rsi_length.unwrap_or(3) == rsi_length
                && combo.updown_length.unwrap_or(2) == updown_length
                && combo.roc_length.unwrap_or(100) == roc_length
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, StochasticConnorsRsiError> {
    let (start, end, step) = range;
    if start == 0 || end == 0 {
        return Err(StochasticConnorsRsiError::InvalidRange { start, end, step });
    }
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) if next > value => value = next,
                _ => break,
            }
        }
    } else {
        let mut value = start;
        while value >= end {
            out.push(value);
            if value < end.saturating_add(step) {
                break;
            }
            value = value.saturating_sub(step);
            if value == 0 {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(StochasticConnorsRsiError::InvalidRange { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_stochastic_connors_rsi(
    sweep: &StochasticConnorsRsiBatchRange,
) -> Result<Vec<StochasticConnorsRsiParams>, StochasticConnorsRsiError> {
    let stoch_lengths = axis_usize(sweep.stoch_length)?;
    let smooth_ks = axis_usize(sweep.smooth_k)?;
    let smooth_ds = axis_usize(sweep.smooth_d)?;
    let rsi_lengths = axis_usize(sweep.rsi_length)?;
    let updown_lengths = axis_usize(sweep.updown_length)?;
    let roc_lengths = axis_usize(sweep.roc_length)?;
    let mut out = Vec::with_capacity(
        stoch_lengths.len()
            * smooth_ks.len()
            * smooth_ds.len()
            * rsi_lengths.len()
            * updown_lengths.len()
            * roc_lengths.len(),
    );
    for stoch_length in stoch_lengths {
        for &smooth_k in &smooth_ks {
            for &smooth_d in &smooth_ds {
                for &rsi_length in &rsi_lengths {
                    for &updown_length in &updown_lengths {
                        for &roc_length in &roc_lengths {
                            out.push(StochasticConnorsRsiParams {
                                stoch_length: Some(stoch_length),
                                smooth_k: Some(smooth_k),
                                smooth_d: Some(smooth_d),
                                rsi_length: Some(rsi_length),
                                updown_length: Some(updown_length),
                                roc_length: Some(roc_length),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn stochastic_connors_rsi_batch_with_kernel(
    data: &[f64],
    sweep: &StochasticConnorsRsiBatchRange,
    kernel: Kernel,
) -> Result<StochasticConnorsRsiBatchOutput, StochasticConnorsRsiError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(StochasticConnorsRsiError::InvalidKernelForBatch(other)),
    };
    stochastic_connors_rsi_batch_impl(data, sweep, batch_kernel.to_non_batch(), true)
}

pub fn stochastic_connors_rsi_batch_slice(
    data: &[f64],
    sweep: &StochasticConnorsRsiBatchRange,
) -> Result<StochasticConnorsRsiBatchOutput, StochasticConnorsRsiError> {
    stochastic_connors_rsi_batch_impl(data, sweep, Kernel::Scalar, false)
}

pub fn stochastic_connors_rsi_batch_par_slice(
    data: &[f64],
    sweep: &StochasticConnorsRsiBatchRange,
) -> Result<StochasticConnorsRsiBatchOutput, StochasticConnorsRsiError> {
    stochastic_connors_rsi_batch_impl(data, sweep, Kernel::Scalar, true)
}

fn stochastic_connors_rsi_batch_impl(
    data: &[f64],
    sweep: &StochasticConnorsRsiBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<StochasticConnorsRsiBatchOutput, StochasticConnorsRsiError> {
    let combos = expand_grid_stochastic_connors_rsi(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(StochasticConnorsRsiError::EmptyInputData);
    }

    let first = first_valid_value(data).ok_or(StochasticConnorsRsiError::AllValuesNaN)?;
    let k_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let input = StochasticConnorsRsiInput::from_slice(data, params.clone());
            stochastic_connors_rsi_prepare(&input)?;
            Ok(k_warmup(first, &input).min(cols))
        })
        .collect::<Result<_, StochasticConnorsRsiError>>()?;
    let d_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let input = StochasticConnorsRsiInput::from_slice(data, params.clone());
            Ok(d_warmup(first, &input).min(cols))
        })
        .collect::<Result<_, StochasticConnorsRsiError>>()?;

    let mut k_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut k_matrix, cols, &k_warmups);
    let mut d_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut d_matrix, cols, &d_warmups);

    let mut k_guard = ManuallyDrop::new(k_matrix);
    let mut d_guard = ManuallyDrop::new(d_matrix);
    let k_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(k_guard.as_mut_ptr(), k_guard.len()) };
    let d_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(d_guard.as_mut_ptr(), d_guard.len()) };

    let do_row =
        |row: usize, row_k_mu: &mut [MaybeUninit<f64>], row_d_mu: &mut [MaybeUninit<f64>]| {
            let params = &combos[row];
            let dst_k = unsafe {
                std::slice::from_raw_parts_mut(row_k_mu.as_mut_ptr() as *mut f64, row_k_mu.len())
            };
            let dst_d = unsafe {
                std::slice::from_raw_parts_mut(row_d_mu.as_mut_ptr() as *mut f64, row_d_mu.len())
            };
            let input = StochasticConnorsRsiInput::from_slice(data, params.clone());
            let _ = stochastic_connors_rsi_into_slice(dst_k, dst_d, &input, kernel);
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        k_mu.par_chunks_mut(cols)
            .zip(d_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_k, row_d))| do_row(row, row_k, row_d));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_k, row_d)) in k_mu.chunks_mut(cols).zip(d_mu.chunks_mut(cols)).enumerate() {
            do_row(row, row_k, row_d);
        }
    } else {
        for (row, (row_k, row_d)) in k_mu.chunks_mut(cols).zip(d_mu.chunks_mut(cols)).enumerate() {
            do_row(row, row_k, row_d);
        }
    }

    let k = unsafe {
        Vec::from_raw_parts(
            k_guard.as_mut_ptr() as *mut f64,
            k_guard.len(),
            k_guard.capacity(),
        )
    };
    let d = unsafe {
        Vec::from_raw_parts(
            d_guard.as_mut_ptr() as *mut f64,
            d_guard.len(),
            d_guard.capacity(),
        )
    };

    Ok(StochasticConnorsRsiBatchOutput {
        k,
        d,
        combos,
        rows,
        cols,
    })
}

fn stochastic_connors_rsi_batch_inner_into(
    data: &[f64],
    sweep: &StochasticConnorsRsiBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<(), StochasticConnorsRsiError> {
    let combos = expand_grid_stochastic_connors_rsi(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(StochasticConnorsRsiError::EmptyInputData);
    }
    let expected =
        rows.checked_mul(cols)
            .ok_or(StochasticConnorsRsiError::OutputLengthMismatch {
                expected: usize::MAX,
                k_len: out_k.len(),
                d_len: out_d.len(),
            })?;
    if out_k.len() != expected || out_d.len() != expected {
        return Err(StochasticConnorsRsiError::OutputLengthMismatch {
            expected,
            k_len: out_k.len(),
            d_len: out_d.len(),
        });
    }

    for params in &combos {
        let input = StochasticConnorsRsiInput::from_slice(data, params.clone());
        stochastic_connors_rsi_prepare(&input)?;
    }

    let do_row = |row: usize, dst_k: &mut [f64], dst_d: &mut [f64]| {
        let params = &combos[row];
        let input = StochasticConnorsRsiInput::from_slice(data, params.clone());
        stochastic_connors_rsi_into_slice(dst_k, dst_d, &input, kernel)
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_k
            .par_chunks_mut(cols)
            .zip(out_d.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(|(row, (dst_k, dst_d))| do_row(row, dst_k, dst_d))?;
        #[cfg(target_arch = "wasm32")]
        for (row, (dst_k, dst_d)) in out_k
            .chunks_mut(cols)
            .zip(out_d.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_k, dst_d)?;
        }
    } else {
        for (row, (dst_k, dst_d)) in out_k
            .chunks_mut(cols)
            .zip(out_d.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_k, dst_d)?;
        }
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochastic_connors_rsi")]
#[pyo3(signature = (source, stoch_length=3, smooth_k=3, smooth_d=3, rsi_length=3, updown_length=2, roc_length=100, kernel=None))]
pub fn stochastic_connors_rsi_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    stoch_length: usize,
    smooth_k: usize,
    smooth_d: usize,
    rsi_length: usize,
    updown_length: usize,
    roc_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let source = source.as_slice()?;
    let input = StochasticConnorsRsiInput::from_slice(
        source,
        StochasticConnorsRsiParams {
            stoch_length: Some(stoch_length),
            smooth_k: Some(smooth_k),
            smooth_d: Some(smooth_d),
            rsi_length: Some(rsi_length),
            updown_length: Some(updown_length),
            roc_length: Some(roc_length),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| stochastic_connors_rsi_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.k.into_pyarray(py), out.d.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "StochasticConnorsRsiStream")]
pub struct StochasticConnorsRsiStreamPy {
    stream: StochasticConnorsRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StochasticConnorsRsiStreamPy {
    #[new]
    #[pyo3(signature = (stoch_length=3, smooth_k=3, smooth_d=3, rsi_length=3, updown_length=2, roc_length=100))]
    fn new(
        stoch_length: usize,
        smooth_k: usize,
        smooth_d: usize,
        rsi_length: usize,
        updown_length: usize,
        roc_length: usize,
    ) -> PyResult<Self> {
        let stream = StochasticConnorsRsiStream::try_new(StochasticConnorsRsiParams {
            stoch_length: Some(stoch_length),
            smooth_k: Some(smooth_k),
            smooth_d: Some(smooth_d),
            rsi_length: Some(rsi_length),
            updown_length: Some(updown_length),
            roc_length: Some(roc_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update_reset_on_nan(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochastic_connors_rsi_batch")]
#[pyo3(signature = (source, stoch_length_range, smooth_k_range, smooth_d_range, rsi_length_range, updown_length_range, roc_length_range, kernel=None))]
pub fn stochastic_connors_rsi_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    stoch_length_range: (usize, usize, usize),
    smooth_k_range: (usize, usize, usize),
    smooth_d_range: (usize, usize, usize),
    rsi_length_range: (usize, usize, usize),
    updown_length_range: (usize, usize, usize),
    roc_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let sweep = StochasticConnorsRsiBatchRange {
        stoch_length: stoch_length_range,
        smooth_k: smooth_k_range,
        smooth_d: smooth_d_range,
        rsi_length: rsi_length_range,
        updown_length: updown_length_range,
        roc_length: roc_length_range,
    };
    let combos = expand_grid_stochastic_connors_rsi(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let arr_k = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let arr_d = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_k = unsafe { arr_k.as_slice_mut()? };
    let out_d = unsafe { arr_d.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        stochastic_connors_rsi_batch_inner_into(
            source,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_k,
            out_d,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("k", arr_k.reshape((rows, cols))?)?;
    dict.set_item("d", arr_d.reshape((rows, cols))?)?;
    dict.set_item(
        "stoch_lengths",
        combos
            .iter()
            .map(|params| params.stoch_length.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_ks",
        combos
            .iter()
            .map(|params| params.smooth_k.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_ds",
        combos
            .iter()
            .map(|params| params.smooth_d.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "rsi_lengths",
        combos
            .iter()
            .map(|params| params.rsi_length.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "updown_lengths",
        combos
            .iter()
            .map(|params| params.updown_length.unwrap_or(2) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "roc_lengths",
        combos
            .iter()
            .map(|params| params.roc_length.unwrap_or(100) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_stochastic_connors_rsi_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stochastic_connors_rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic_connors_rsi_batch_py, m)?)?;
    m.add_class::<StochasticConnorsRsiStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StochasticConnorsRsiJsOutput {
    k: Vec<f64>,
    d: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StochasticConnorsRsiBatchConfig {
    stoch_length_range: Vec<usize>,
    smooth_k_range: Vec<usize>,
    smooth_d_range: Vec<usize>,
    rsi_length_range: Vec<usize>,
    updown_length_range: Vec<usize>,
    roc_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StochasticConnorsRsiBatchJsOutput {
    k: Vec<f64>,
    d: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<StochasticConnorsRsiParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_connors_rsi_js")]
pub fn stochastic_connors_rsi_js(
    source: &[f64],
    stoch_length: usize,
    smooth_k: usize,
    smooth_d: usize,
    rsi_length: usize,
    updown_length: usize,
    roc_length: usize,
) -> Result<JsValue, JsValue> {
    let input = StochasticConnorsRsiInput::from_slice(
        source,
        StochasticConnorsRsiParams {
            stoch_length: Some(stoch_length),
            smooth_k: Some(smooth_k),
            smooth_d: Some(smooth_d),
            rsi_length: Some(rsi_length),
            updown_length: Some(updown_length),
            roc_length: Some(roc_length),
        },
    );
    let out = stochastic_connors_rsi_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&StochasticConnorsRsiJsOutput { k: out.k, d: out.d })
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_connors_rsi_batch_js")]
pub fn stochastic_connors_rsi_batch_js(
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: StochasticConnorsRsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.stoch_length_range.len() != 3
        || config.smooth_k_range.len() != 3
        || config.smooth_d_range.len() != 3
        || config.rsi_length_range.len() != 3
        || config.updown_length_range.len() != 3
        || config.roc_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: each range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = StochasticConnorsRsiBatchRange {
        stoch_length: (
            config.stoch_length_range[0],
            config.stoch_length_range[1],
            config.stoch_length_range[2],
        ),
        smooth_k: (
            config.smooth_k_range[0],
            config.smooth_k_range[1],
            config.smooth_k_range[2],
        ),
        smooth_d: (
            config.smooth_d_range[0],
            config.smooth_d_range[1],
            config.smooth_d_range[2],
        ),
        rsi_length: (
            config.rsi_length_range[0],
            config.rsi_length_range[1],
            config.rsi_length_range[2],
        ),
        updown_length: (
            config.updown_length_range[0],
            config.updown_length_range[1],
            config.updown_length_range[2],
        ),
        roc_length: (
            config.roc_length_range[0],
            config.roc_length_range[1],
            config.roc_length_range[2],
        ),
    };
    let out = stochastic_connors_rsi_batch_with_kernel(source, &sweep, Kernel::ScalarBatch)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&StochasticConnorsRsiBatchJsOutput {
        k: out.k,
        d: out.d,
        rows: out.rows,
        cols: out.cols,
        combos: out.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_connors_rsi_alloc(len: usize) -> *mut f64 {
    let mut buf = Vec::<f64>::with_capacity(len * 2);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_connors_rsi_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::<f64>::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_connors_rsi_into(
    source: &[f64],
    out_ptr: *mut f64,
    stoch_length: usize,
    smooth_k: usize,
    smooth_d: usize,
    rsi_length: usize,
    updown_length: usize,
    roc_length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to stochastic_connors_rsi_into",
        ));
    }
    let len = source.len();
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len * 2) };
    let (out_k, out_d) = out.split_at_mut(len);
    let input = StochasticConnorsRsiInput::from_slice(
        source,
        StochasticConnorsRsiParams {
            stoch_length: Some(stoch_length),
            smooth_k: Some(smooth_k),
            smooth_d: Some(smooth_d),
            rsi_length: Some(rsi_length),
            updown_length: Some(updown_length),
            roc_length: Some(roc_length),
        },
    );
    stochastic_connors_rsi_into_slice(out_k, out_d, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_connors_rsi_into_host")]
pub fn stochastic_connors_rsi_into_host(
    source: &[f64],
    out_ptr: *mut f64,
    stoch_length: usize,
    smooth_k: usize,
    smooth_d: usize,
    rsi_length: usize,
    updown_length: usize,
    roc_length: usize,
) -> Result<(), JsValue> {
    stochastic_connors_rsi_into(
        source,
        out_ptr,
        stoch_length,
        smooth_k,
        smooth_d,
        rsi_length,
        updown_length,
        roc_length,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_connors_rsi_batch_into(
    source: &[f64],
    out_ptr: *mut f64,
    config: JsValue,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to stochastic_connors_rsi_batch_into",
        ));
    }
    let config: StochasticConnorsRsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.stoch_length_range.len() != 3
        || config.smooth_k_range.len() != 3
        || config.smooth_d_range.len() != 3
        || config.rsi_length_range.len() != 3
        || config.updown_length_range.len() != 3
        || config.roc_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: each range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = StochasticConnorsRsiBatchRange {
        stoch_length: (
            config.stoch_length_range[0],
            config.stoch_length_range[1],
            config.stoch_length_range[2],
        ),
        smooth_k: (
            config.smooth_k_range[0],
            config.smooth_k_range[1],
            config.smooth_k_range[2],
        ),
        smooth_d: (
            config.smooth_d_range[0],
            config.smooth_d_range[1],
            config.smooth_d_range[2],
        ),
        rsi_length: (
            config.rsi_length_range[0],
            config.rsi_length_range[1],
            config.rsi_length_range[2],
        ),
        updown_length: (
            config.updown_length_range[0],
            config.updown_length_range[1],
            config.updown_length_range[2],
        ),
        roc_length: (
            config.roc_length_range[0],
            config.roc_length_range[1],
            config.roc_length_range[2],
        ),
    };
    let combos = expand_grid_stochastic_connors_rsi(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let expected = rows * cols * 2;
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, expected) };
    let (out_k, out_d) = out.split_at_mut(rows * cols);
    stochastic_connors_rsi_batch_inner_into(source, &sweep, Kernel::Scalar, false, out_k, out_d)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_connors_rsi_output_into_js(
    source: &[f64],
    stoch_length: usize,
    smooth_k: usize,
    smooth_d: usize,
    rsi_length: usize,
    updown_length: usize,
    roc_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = stochastic_connors_rsi_js(
        source,
        stoch_length,
        smooth_k,
        smooth_d,
        rsi_length,
        updown_length,
        roc_length,
    )?;
    crate::write_wasm_object_f64_outputs("stochastic_connors_rsi_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_connors_rsi_batch_output_into_js(
    source: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = stochastic_connors_rsi_batch_js(source, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "stochastic_connors_rsi_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::cpu_batch::compute_cpu_batch;
    use crate::indicators::dispatch::{
        IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV, ParamValue,
    };

    fn sample_data(len: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            out.push(
                100.0
                    + (x * 0.17).sin() * 3.0
                    + (x * 0.03)
                    + (x * 0.11).cos() * 1.4
                    + (x % 7.0) * 0.05,
            );
        }
        out
    }

    fn assert_close_nan(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (idx, (&lhs, &rhs)) in a.iter().zip(b.iter()).enumerate() {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "nan mismatch at {idx}: {lhs} vs {rhs}"
                );
            } else {
                assert!(
                    (lhs - rhs).abs() <= 1e-10,
                    "value mismatch at {idx}: {lhs} vs {rhs}"
                );
            }
        }
    }

    fn naive_expected(
        data: &[f64],
        stoch_length: usize,
        smooth_k: usize,
        smooth_d: usize,
        rsi_length: usize,
        updown_length: usize,
        roc_length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut state = StochasticConnorsRsiState::new(&StochasticConnorsRsiParams {
            stoch_length: Some(stoch_length),
            smooth_k: Some(smooth_k),
            smooth_d: Some(smooth_d),
            rsi_length: Some(rsi_length),
            updown_length: Some(updown_length),
            roc_length: Some(roc_length),
        });
        let mut k = vec![f64::NAN; data.len()];
        let mut d = vec![f64::NAN; data.len()];
        for (i, &value) in data.iter().enumerate() {
            let (kv, dv) = state.update_reset_on_nan(value);
            k[i] = kv;
            d[i] = dv;
        }
        (k, d)
    }

    #[test]
    fn stochastic_connors_rsi_matches_naive() {
        let data = sample_data(320);
        let input = StochasticConnorsRsiInput::from_slice(
            &data,
            StochasticConnorsRsiParams {
                stoch_length: Some(3),
                smooth_k: Some(3),
                smooth_d: Some(3),
                rsi_length: Some(3),
                updown_length: Some(2),
                roc_length: Some(100),
            },
        );
        let out = stochastic_connors_rsi(&input).expect("indicator");
        let (expected_k, expected_d) = naive_expected(&data, 3, 3, 3, 3, 2, 100);
        assert_close_nan(&out.k, &expected_k);
        assert_close_nan(&out.d, &expected_d);
    }

    #[test]
    fn stochastic_connors_rsi_into_matches_api() {
        let data = sample_data(240);
        let input = StochasticConnorsRsiInput::from_slice(
            &data,
            StochasticConnorsRsiParams {
                stoch_length: Some(5),
                smooth_k: Some(4),
                smooth_d: Some(3),
                rsi_length: Some(4),
                updown_length: Some(3),
                roc_length: Some(30),
            },
        );
        let out = stochastic_connors_rsi(&input).expect("baseline");
        let mut k = vec![0.0; data.len()];
        let mut d = vec![0.0; data.len()];
        stochastic_connors_rsi_into(&input, &mut k, &mut d).expect("into");
        assert_close_nan(&k, &out.k);
        assert_close_nan(&d, &out.d);
    }

    #[test]
    fn stochastic_connors_rsi_stream_matches_batch() {
        let data = sample_data(240);
        let input = StochasticConnorsRsiInput::from_slice(
            &data,
            StochasticConnorsRsiParams {
                stoch_length: Some(3),
                smooth_k: Some(3),
                smooth_d: Some(3),
                rsi_length: Some(3),
                updown_length: Some(2),
                roc_length: Some(50),
            },
        );
        let batch = stochastic_connors_rsi(&input).expect("batch");
        let mut stream = StochasticConnorsRsiStream::try_new(StochasticConnorsRsiParams {
            stoch_length: Some(3),
            smooth_k: Some(3),
            smooth_d: Some(3),
            rsi_length: Some(3),
            updown_length: Some(2),
            roc_length: Some(50),
        })
        .expect("stream");
        let mut k = Vec::with_capacity(data.len());
        let mut d = Vec::with_capacity(data.len());
        for &value in &data {
            match stream.update_reset_on_nan(value) {
                Some((kv, dv)) => {
                    k.push(kv);
                    d.push(dv);
                }
                None => {
                    k.push(f64::NAN);
                    d.push(f64::NAN);
                }
            }
        }
        assert_close_nan(&k, &batch.k);
        assert_close_nan(&d, &batch.d);
    }

    #[test]
    fn stochastic_connors_rsi_batch_single_param_matches_single() {
        let data = sample_data(220);
        let sweep = StochasticConnorsRsiBatchRange {
            stoch_length: (3, 3, 0),
            smooth_k: (3, 3, 0),
            smooth_d: (3, 3, 0),
            rsi_length: (3, 3, 0),
            updown_length: (2, 2, 0),
            roc_length: (20, 20, 0),
        };
        let batch = stochastic_connors_rsi_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
            .expect("batch");
        let input = StochasticConnorsRsiInput::from_slice(
            &data,
            StochasticConnorsRsiParams {
                stoch_length: Some(3),
                smooth_k: Some(3),
                smooth_d: Some(3),
                rsi_length: Some(3),
                updown_length: Some(2),
                roc_length: Some(20),
            },
        );
        let out = stochastic_connors_rsi(&input).expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close_nan(&batch.k[..data.len()], &out.k);
        assert_close_nan(&batch.d[..data.len()], &out.d);
    }

    #[test]
    fn stochastic_connors_rsi_rejects_invalid_roc_length() {
        let data = sample_data(32);
        let input = StochasticConnorsRsiInput::from_slice(
            &data,
            StochasticConnorsRsiParams {
                stoch_length: Some(3),
                smooth_k: Some(3),
                smooth_d: Some(3),
                rsi_length: Some(3),
                updown_length: Some(2),
                roc_length: Some(0),
            },
        );
        let err = stochastic_connors_rsi(&input).expect_err("invalid");
        assert!(matches!(
            err,
            StochasticConnorsRsiError::InvalidRocLength { .. }
        ));
    }

    #[test]
    fn stochastic_connors_rsi_dispatch_matches_direct() {
        let data = sample_data(220);
        let params = [
            ParamKV {
                key: "stoch_length",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "smooth_k",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "smooth_d",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "rsi_length",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "updown_length",
                value: ParamValue::Int(2),
            },
            ParamKV {
                key: "roc_length",
                value: ParamValue::Int(20),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "stochastic_connors_rsi",
            output_id: Some("k"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = stochastic_connors_rsi(&StochasticConnorsRsiInput::from_slice(
            &data,
            StochasticConnorsRsiParams {
                stoch_length: Some(3),
                smooth_k: Some(3),
                smooth_d: Some(3),
                rsi_length: Some(3),
                updown_length: Some(2),
                roc_length: Some(20),
            },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert_close_nan(out.values_f64.as_ref().expect("values"), &direct.k);
    }
}
