#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
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
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 14;
const DEFAULT_SOURCE: &str = "close";
const DEFAULT_PRESMOOTH: usize = 10;
const DEFAULT_POSTSMOOTH: usize = 10;
const DEFAULT_SMOOTHING_METHOD: &str = "sma";
const MIN_STOCH_LENGTH: usize = 4;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SmoothingMethod {
    None,
    Sma,
    Tma,
    Lsma,
}

impl SmoothingMethod {
    #[inline]
    fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("none") {
            Some(Self::None)
        } else if name.eq_ignore_ascii_case("sma") {
            Some(Self::Sma)
        } else if name.eq_ignore_ascii_case("tma") {
            Some(Self::Tma)
        } else if name.eq_ignore_ascii_case("lsma") {
            Some(Self::Lsma)
        } else {
            None
        }
    }

    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sma => "sma",
            Self::Tma => "tma",
            Self::Lsma => "lsma",
        }
    }
}

impl<'a> AsRef<[f64]> for MultiLengthStochasticAverageInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MultiLengthStochasticAverageData::Slice(slice) => slice,
            MultiLengthStochasticAverageData::Candles { candles, source } => match *source {
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                "hlcc4" | "hlcc" => &candles.hlcc4,
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum MultiLengthStochasticAverageData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MultiLengthStochasticAverageOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MultiLengthStochasticAverageParams {
    pub length: Option<usize>,
    pub presmooth: Option<usize>,
    pub premethod: Option<String>,
    pub postsmooth: Option<usize>,
    pub postmethod: Option<String>,
}

impl Default for MultiLengthStochasticAverageParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            presmooth: Some(DEFAULT_PRESMOOTH),
            premethod: Some(DEFAULT_SMOOTHING_METHOD.to_string()),
            postsmooth: Some(DEFAULT_POSTSMOOTH),
            postmethod: Some(DEFAULT_SMOOTHING_METHOD.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiLengthStochasticAverageInput<'a> {
    pub data: MultiLengthStochasticAverageData<'a>,
    pub params: MultiLengthStochasticAverageParams,
}

impl<'a> MultiLengthStochasticAverageInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: MultiLengthStochasticAverageParams,
    ) -> Self {
        Self {
            data: MultiLengthStochasticAverageData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: MultiLengthStochasticAverageParams) -> Self {
        Self {
            data: MultiLengthStochasticAverageData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            MultiLengthStochasticAverageParams::default(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct MultiLengthStochasticAverageBuilder {
    length: Option<usize>,
    presmooth: Option<usize>,
    premethod: Option<String>,
    postsmooth: Option<usize>,
    postmethod: Option<String>,
    kernel: Kernel,
}

impl Default for MultiLengthStochasticAverageBuilder {
    fn default() -> Self {
        Self {
            length: None,
            presmooth: None,
            premethod: None,
            postsmooth: None,
            postmethod: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MultiLengthStochasticAverageBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline]
    pub fn presmooth(mut self, presmooth: usize) -> Self {
        self.presmooth = Some(presmooth);
        self
    }

    #[inline]
    pub fn premethod<T: Into<String>>(mut self, premethod: T) -> Self {
        self.premethod = Some(premethod.into());
        self
    }

    #[inline]
    pub fn postsmooth(mut self, postsmooth: usize) -> Self {
        self.postsmooth = Some(postsmooth);
        self
    }

    #[inline]
    pub fn postmethod<T: Into<String>>(mut self, postmethod: T) -> Self {
        self.postmethod = Some(postmethod.into());
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
    ) -> Result<MultiLengthStochasticAverageOutput, MultiLengthStochasticAverageError> {
        let input = MultiLengthStochasticAverageInput::from_candles(
            candles,
            source,
            MultiLengthStochasticAverageParams {
                length: self.length,
                presmooth: self.presmooth,
                premethod: self.premethod,
                postsmooth: self.postsmooth,
                postmethod: self.postmethod,
            },
        );
        multi_length_stochastic_average_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<MultiLengthStochasticAverageOutput, MultiLengthStochasticAverageError> {
        let input = MultiLengthStochasticAverageInput::from_slice(
            data,
            MultiLengthStochasticAverageParams {
                length: self.length,
                presmooth: self.presmooth,
                premethod: self.premethod,
                postsmooth: self.postsmooth,
                postmethod: self.postmethod,
            },
        );
        multi_length_stochastic_average_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<MultiLengthStochasticAverageStream, MultiLengthStochasticAverageError> {
        MultiLengthStochasticAverageStream::try_new(MultiLengthStochasticAverageParams {
            length: self.length,
            presmooth: self.presmooth,
            premethod: self.premethod,
            postsmooth: self.postsmooth,
            postmethod: self.postmethod,
        })
    }
}

#[derive(Debug, Error)]
pub enum MultiLengthStochasticAverageError {
    #[error("multi_length_stochastic_average: Input data slice is empty.")]
    EmptyInputData,
    #[error("multi_length_stochastic_average: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "multi_length_stochastic_average: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("multi_length_stochastic_average: Invalid presmooth: {presmooth}")]
    InvalidPresmooth { presmooth: usize },
    #[error("multi_length_stochastic_average: Invalid postsmooth: {postsmooth}")]
    InvalidPostsmooth { postsmooth: usize },
    #[error("multi_length_stochastic_average: Invalid premethod: {premethod}")]
    InvalidPreMethod { premethod: String },
    #[error("multi_length_stochastic_average: Invalid postmethod: {postmethod}")]
    InvalidPostMethod { postmethod: String },
    #[error(
        "multi_length_stochastic_average: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "multi_length_stochastic_average: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "multi_length_stochastic_average: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("multi_length_stochastic_average: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    length: usize,
    presmooth: usize,
    premethod: SmoothingMethod,
    postsmooth: usize,
    postmethod: SmoothingMethod,
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
fn smoothing_warmup(method: SmoothingMethod, length: usize) -> usize {
    match method {
        SmoothingMethod::None => 0,
        SmoothingMethod::Sma | SmoothingMethod::Lsma => length.saturating_sub(1),
        SmoothingMethod::Tma => length.saturating_sub(1).saturating_mul(2),
    }
}

#[inline(always)]
fn total_warmup(params: ResolvedParams) -> usize {
    smoothing_warmup(params.premethod, params.presmooth)
        + params.length.saturating_sub(1)
        + smoothing_warmup(params.postmethod, params.postsmooth)
}

#[inline(always)]
fn canonical_method_name(name: Option<&str>, default: &str) -> String {
    name.unwrap_or(default).to_ascii_lowercase()
}

#[inline]
fn resolve_params(
    params: &MultiLengthStochasticAverageParams,
    data_len: Option<usize>,
) -> Result<ResolvedParams, MultiLengthStochasticAverageError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    let presmooth = params.presmooth.unwrap_or(DEFAULT_PRESMOOTH);
    let postsmooth = params.postsmooth.unwrap_or(DEFAULT_POSTSMOOTH);
    let premethod_name =
        canonical_method_name(params.premethod.as_deref(), DEFAULT_SMOOTHING_METHOD);
    let postmethod_name =
        canonical_method_name(params.postmethod.as_deref(), DEFAULT_SMOOTHING_METHOD);
    let premethod = SmoothingMethod::parse(&premethod_name).ok_or_else(|| {
        MultiLengthStochasticAverageError::InvalidPreMethod {
            premethod: premethod_name.clone(),
        }
    })?;
    let postmethod = SmoothingMethod::parse(&postmethod_name).ok_or_else(|| {
        MultiLengthStochasticAverageError::InvalidPostMethod {
            postmethod: postmethod_name.clone(),
        }
    })?;

    if length < MIN_STOCH_LENGTH {
        return Err(MultiLengthStochasticAverageError::InvalidLength {
            length,
            data_len: data_len.unwrap_or(0),
        });
    }
    if presmooth == 0 {
        return Err(MultiLengthStochasticAverageError::InvalidPresmooth { presmooth });
    }
    if postsmooth == 0 {
        return Err(MultiLengthStochasticAverageError::InvalidPostsmooth { postsmooth });
    }
    if let Some(data_len) = data_len {
        if length > data_len {
            return Err(MultiLengthStochasticAverageError::InvalidLength { length, data_len });
        }
    }

    Ok(ResolvedParams {
        length,
        presmooth,
        premethod,
        postsmooth,
        postmethod,
    })
}

#[derive(Clone, Debug)]
struct SmaState {
    ring: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
}

impl SmaState {
    #[inline]
    fn new(length: usize) -> Self {
        Self {
            ring: vec![0.0; length.max(1)],
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        let len = self.ring.len();
        if self.count == len {
            self.sum -= self.ring[self.head];
        } else {
            self.count += 1;
        }
        self.ring[self.head] = value;
        self.sum += value;
        self.head += 1;
        if self.head == len {
            self.head = 0;
        }
        if self.count == len {
            Some(self.sum / len as f64)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct LsmaState {
    ring: Vec<f64>,
    head: usize,
    count: usize,
    sum_y: f64,
    sum_xy: f64,
    x_sum: f64,
    denom: f64,
}

impl LsmaState {
    #[inline]
    fn new(length: usize) -> Self {
        let n = length.max(1);
        let n_f = n as f64;
        let x_sum = ((n * (n - 1)) / 2) as f64;
        let x2_sum = ((n * (n - 1) * (2 * n - 1)) / 6) as f64;
        Self {
            ring: vec![0.0; n],
            head: 0,
            count: 0,
            sum_y: 0.0,
            sum_xy: 0.0,
            x_sum,
            denom: n_f * x2_sum - x_sum * x_sum,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum_y = 0.0;
        self.sum_xy = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        let n = self.ring.len();
        if self.count < n {
            let idx = self.count;
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == n {
                self.head = 0;
            }
            self.count += 1;
            self.sum_y += value;
            self.sum_xy += idx as f64 * value;
            if self.count == n {
                Some(self.endpoint())
            } else {
                None
            }
        } else {
            let old = self.ring[self.head];
            let old_sum_y = self.sum_y;
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == n {
                self.head = 0;
            }
            self.sum_y = old_sum_y - old + value;
            self.sum_xy = self.sum_xy - (old_sum_y - old) + (n - 1) as f64 * value;
            Some(self.endpoint())
        }
    }

    #[inline]
    fn endpoint(&self) -> f64 {
        let n = self.ring.len() as f64;
        let slope = (n * self.sum_xy - self.x_sum * self.sum_y) / self.denom;
        let intercept = (self.sum_y - slope * self.x_sum) / n;
        intercept + slope * (self.ring.len() - 1) as f64
    }
}

#[derive(Clone, Debug)]
enum SmoothingState {
    None,
    Sma(SmaState),
    Tma { inner: SmaState, outer: SmaState },
    Lsma(LsmaState),
}

impl SmoothingState {
    #[inline]
    fn new(method: SmoothingMethod, length: usize) -> Self {
        match method {
            SmoothingMethod::None => Self::None,
            SmoothingMethod::Sma => Self::Sma(SmaState::new(length)),
            SmoothingMethod::Tma => Self::Tma {
                inner: SmaState::new(length),
                outer: SmaState::new(length),
            },
            SmoothingMethod::Lsma => Self::Lsma(LsmaState::new(length)),
        }
    }

    #[inline]
    fn reset(&mut self) {
        match self {
            Self::None => {}
            Self::Sma(state) => state.reset(),
            Self::Tma { inner, outer } => {
                inner.reset();
                outer.reset();
            }
            Self::Lsma(state) => state.reset(),
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        match self {
            Self::None => Some(value),
            Self::Sma(state) => state.update(value),
            Self::Tma { inner, outer } => inner.update(value).and_then(|x| outer.update(x)),
            Self::Lsma(state) => state.update(value),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiLengthStochasticAverageStream {
    params: ResolvedParams,
    pre_smoother: SmoothingState,
    post_smoother: SmoothingState,
    ring: Vec<f64>,
    head: usize,
    count: usize,
}

impl MultiLengthStochasticAverageStream {
    #[inline]
    pub fn try_new(
        params: MultiLengthStochasticAverageParams,
    ) -> Result<Self, MultiLengthStochasticAverageError> {
        let params = resolve_params(&params, None)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            pre_smoother: SmoothingState::new(params.premethod, params.presmooth),
            post_smoother: SmoothingState::new(params.postmethod, params.postsmooth),
            ring: vec![0.0; params.length],
            head: 0,
            count: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.pre_smoother.reset();
        self.post_smoother.reset();
        self.head = 0;
        self.count = 0;
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        total_warmup(self.params)
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let pre = self.pre_smoother.update(value)?;
        self.push_pre_value(pre);
        if self.count < self.params.length {
            return None;
        }

        let norm = self.current_norm()?;
        self.post_smoother.update(norm)
    }

    #[inline]
    fn push_pre_value(&mut self, value: f64) {
        self.ring[self.head] = value;
        self.head += 1;
        if self.head == self.params.length {
            self.head = 0;
        }
        if self.count < self.params.length {
            self.count += 1;
        }
    }

    #[inline]
    fn current_norm(&mut self) -> Option<f64> {
        let len = self.params.length;
        let newest = (self.head + len - 1) % len;
        let current = self.ring[newest];
        let mut min_value = current;
        let mut max_value = current;
        let mut idx = newest;
        let mut sum = 0.0;

        for window in 1..=len {
            let value = self.ring[idx];
            if value < min_value {
                min_value = value;
            }
            if value > max_value {
                max_value = value;
            }
            if window >= MIN_STOCH_LENGTH {
                let denom = max_value - min_value;
                if denom.abs() <= FLOAT_TOL {
                    self.post_smoother.reset();
                    return None;
                }
                sum += (current - min_value) / denom;
            }
            idx = if idx == 0 { len - 1 } else { idx - 1 };
        }

        Some(sum / (len - (MIN_STOCH_LENGTH - 1)) as f64 * 100.0)
    }
}

#[inline(always)]
fn multi_length_stochastic_average_prepare<'a>(
    input: &'a MultiLengthStochasticAverageInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, ResolvedParams, Kernel), MultiLengthStochasticAverageError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(MultiLengthStochasticAverageError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(MultiLengthStochasticAverageError::AllValuesNaN);
    }

    let params = resolve_params(&input.params, Some(data.len()))?;
    let needed = total_warmup(params) + 1;
    let valid = max_consecutive_valid_values(data);
    if valid < needed {
        return Err(MultiLengthStochasticAverageError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };

    Ok((data, first, params, chosen))
}

#[inline(always)]
fn multi_length_stochastic_average_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    out: &mut [f64],
) {
    if params.length == DEFAULT_LENGTH
        && params.presmooth == DEFAULT_PRESMOOTH
        && params.postsmooth == DEFAULT_POSTSMOOTH
        && params.premethod == SmoothingMethod::Sma
        && params.postmethod == SmoothingMethod::Sma
        && data.iter().all(|value| value.is_finite())
    {
        multi_length_stochastic_average_default_sma_finite(data, out);
        return;
    }

    out.fill(f64::NAN);
    let mut stream = MultiLengthStochasticAverageStream::new_resolved(params);
    for (slot, &value) in out.iter_mut().zip(data.iter()) {
        if let Some(result) = stream.update(value) {
            *slot = result;
        }
    }
}

#[inline(always)]
fn multi_length_stochastic_average_default_sma_finite(data: &[f64], out: &mut [f64]) {
    out.fill(f64::NAN);

    let mut pre_ring = [0.0f64; DEFAULT_PRESMOOTH];
    let mut pre_head = 0usize;
    let mut pre_count = 0usize;
    let mut pre_sum = 0.0;

    let mut stoch_ring = [0.0f64; DEFAULT_LENGTH];
    let mut stoch_head = 0usize;
    let mut stoch_count = 0usize;

    let mut post_ring = [0.0f64; DEFAULT_POSTSMOOTH];
    let mut post_head = 0usize;
    let mut post_count = 0usize;
    let mut post_sum = 0.0;

    for (out_slot, &value) in out.iter_mut().zip(data.iter()) {
        if pre_count == DEFAULT_PRESMOOTH {
            pre_sum -= pre_ring[pre_head];
        } else {
            pre_count += 1;
        }
        pre_ring[pre_head] = value;
        pre_sum += value;
        pre_head += 1;
        if pre_head == DEFAULT_PRESMOOTH {
            pre_head = 0;
        }
        if pre_count < DEFAULT_PRESMOOTH {
            continue;
        }

        let pre = pre_sum / DEFAULT_PRESMOOTH as f64;
        stoch_ring[stoch_head] = pre;
        stoch_head += 1;
        if stoch_head == DEFAULT_LENGTH {
            stoch_head = 0;
        }
        if stoch_count < DEFAULT_LENGTH {
            stoch_count += 1;
            if stoch_count < DEFAULT_LENGTH {
                continue;
            }
        }

        let newest = if stoch_head == 0 {
            DEFAULT_LENGTH - 1
        } else {
            stoch_head - 1
        };
        let current = stoch_ring[newest];
        let mut min_value = current;
        let mut max_value = current;
        let mut idx = newest;
        let mut sum = 0.0;
        let mut valid_norm = true;

        for window in 1..=DEFAULT_LENGTH {
            let sample = stoch_ring[idx];
            if sample < min_value {
                min_value = sample;
            }
            if sample > max_value {
                max_value = sample;
            }
            if window >= MIN_STOCH_LENGTH {
                let denom = max_value - min_value;
                if denom.abs() <= FLOAT_TOL {
                    post_head = 0;
                    post_count = 0;
                    post_sum = 0.0;
                    valid_norm = false;
                    break;
                }
                sum += (current - min_value) / denom;
            }
            idx = if idx == 0 {
                DEFAULT_LENGTH - 1
            } else {
                idx - 1
            };
        }
        if !valid_norm {
            continue;
        }

        let norm = sum / (DEFAULT_LENGTH - (MIN_STOCH_LENGTH - 1)) as f64 * 100.0;
        if post_count == DEFAULT_POSTSMOOTH {
            post_sum -= post_ring[post_head];
        } else {
            post_count += 1;
        }
        post_ring[post_head] = norm;
        post_sum += norm;
        post_head += 1;
        if post_head == DEFAULT_POSTSMOOTH {
            post_head = 0;
        }
        if post_count == DEFAULT_POSTSMOOTH {
            *out_slot = post_sum / DEFAULT_POSTSMOOTH as f64;
        }
    }
}

#[inline]
pub fn multi_length_stochastic_average(
    input: &MultiLengthStochasticAverageInput,
) -> Result<MultiLengthStochasticAverageOutput, MultiLengthStochasticAverageError> {
    multi_length_stochastic_average_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn multi_length_stochastic_average_with_kernel(
    input: &MultiLengthStochasticAverageInput,
    kernel: Kernel,
) -> Result<MultiLengthStochasticAverageOutput, MultiLengthStochasticAverageError> {
    let (data, first, params, _chosen) = multi_length_stochastic_average_prepare(input, kernel)?;
    let warmup = first.saturating_add(total_warmup(params)).min(data.len());
    let mut values = alloc_with_nan_prefix(data.len(), warmup);
    multi_length_stochastic_average_row_from_slice(data, params, &mut values);
    Ok(MultiLengthStochasticAverageOutput { values })
}

#[inline]
pub fn multi_length_stochastic_average_into_slice(
    dst: &mut [f64],
    input: &MultiLengthStochasticAverageInput,
    kernel: Kernel,
) -> Result<(), MultiLengthStochasticAverageError> {
    let expected = input.as_ref().len();
    if dst.len() != expected {
        return Err(MultiLengthStochasticAverageError::OutputLengthMismatch {
            expected,
            got: dst.len(),
        });
    }
    let (data, _first, params, _chosen) = multi_length_stochastic_average_prepare(input, kernel)?;
    multi_length_stochastic_average_row_from_slice(data, params, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn multi_length_stochastic_average_into(
    input: &MultiLengthStochasticAverageInput,
    out: &mut [f64],
) -> Result<(), MultiLengthStochasticAverageError> {
    multi_length_stochastic_average_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MultiLengthStochasticAverageBatchRange {
    pub length: (usize, usize, usize),
    pub presmooth: (usize, usize, usize),
    pub postsmooth: (usize, usize, usize),
    pub premethod: Option<String>,
    pub postmethod: Option<String>,
}

impl Default for MultiLengthStochasticAverageBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            presmooth: (DEFAULT_PRESMOOTH, DEFAULT_PRESMOOTH, 0),
            postsmooth: (DEFAULT_POSTSMOOTH, DEFAULT_POSTSMOOTH, 0),
            premethod: Some(DEFAULT_SMOOTHING_METHOD.to_string()),
            postmethod: Some(DEFAULT_SMOOTHING_METHOD.to_string()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiLengthStochasticAverageBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MultiLengthStochasticAverageParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MultiLengthStochasticAverageBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &MultiLengthStochasticAverageParams) -> Option<usize> {
        let target = MultiLengthStochasticAverageParams {
            length: Some(params.length.unwrap_or(DEFAULT_LENGTH)),
            presmooth: Some(params.presmooth.unwrap_or(DEFAULT_PRESMOOTH)),
            premethod: Some(canonical_method_name(
                params.premethod.as_deref(),
                DEFAULT_SMOOTHING_METHOD,
            )),
            postsmooth: Some(params.postsmooth.unwrap_or(DEFAULT_POSTSMOOTH)),
            postmethod: Some(canonical_method_name(
                params.postmethod.as_deref(),
                DEFAULT_SMOOTHING_METHOD,
            )),
        };
        self.combos.iter().position(|combo| combo == &target)
    }

    #[inline]
    pub fn values_for(&self, params: &MultiLengthStochasticAverageParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[derive(Clone, Debug)]
pub struct MultiLengthStochasticAverageBatchBuilder {
    range: MultiLengthStochasticAverageBatchRange,
    kernel: Kernel,
}

impl Default for MultiLengthStochasticAverageBatchBuilder {
    fn default() -> Self {
        Self {
            range: MultiLengthStochasticAverageBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl MultiLengthStochasticAverageBatchBuilder {
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
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn presmooth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.presmooth = (start, end, step);
        self
    }

    #[inline]
    pub fn postsmooth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.postsmooth = (start, end, step);
        self
    }

    #[inline]
    pub fn premethod<T: Into<String>>(mut self, premethod: T) -> Self {
        self.range.premethod = Some(premethod.into());
        self
    }

    #[inline]
    pub fn postmethod<T: Into<String>>(mut self, postmethod: T) -> Self {
        self.range.postmethod = Some(postmethod.into());
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<MultiLengthStochasticAverageBatchOutput, MultiLengthStochasticAverageError> {
        multi_length_stochastic_average_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<MultiLengthStochasticAverageBatchOutput, MultiLengthStochasticAverageError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[inline]
fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, MultiLengthStochasticAverageError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(MultiLengthStochasticAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
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
        while value >= end {
            out.push(value);
            let next = value.saturating_sub(step);
            if next == value {
                break;
            }
            value = next;
        }
    }

    if out.is_empty() {
        return Err(MultiLengthStochasticAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }

    Ok(out)
}

fn expand_grid_multi_length_stochastic_average(
    range: &MultiLengthStochasticAverageBatchRange,
) -> Result<Vec<MultiLengthStochasticAverageParams>, MultiLengthStochasticAverageError> {
    let lengths = expand_axis_usize(range.length.0, range.length.1, range.length.2)?;
    let presmooths = expand_axis_usize(range.presmooth.0, range.presmooth.1, range.presmooth.2)?;
    let postsmooths =
        expand_axis_usize(range.postsmooth.0, range.postsmooth.1, range.postsmooth.2)?;
    let premethod = canonical_method_name(range.premethod.as_deref(), DEFAULT_SMOOTHING_METHOD);
    let postmethod = canonical_method_name(range.postmethod.as_deref(), DEFAULT_SMOOTHING_METHOD);

    let mut combos = Vec::with_capacity(
        lengths
            .len()
            .saturating_mul(presmooths.len())
            .saturating_mul(postsmooths.len()),
    );

    for &length in &lengths {
        for &presmooth in &presmooths {
            for &postsmooth in &postsmooths {
                combos.push(MultiLengthStochasticAverageParams {
                    length: Some(length),
                    presmooth: Some(presmooth),
                    premethod: Some(premethod.clone()),
                    postsmooth: Some(postsmooth),
                    postmethod: Some(postmethod.clone()),
                });
            }
        }
    }

    if combos.is_empty() {
        return Err(MultiLengthStochasticAverageError::InvalidRange {
            start: range.length.0.to_string(),
            end: range.length.1.to_string(),
            step: range.length.2.to_string(),
        });
    }

    Ok(combos)
}

#[inline]
pub fn multi_length_stochastic_average_batch_with_kernel(
    data: &[f64],
    sweep: &MultiLengthStochasticAverageBatchRange,
    kernel: Kernel,
) -> Result<MultiLengthStochasticAverageBatchOutput, MultiLengthStochasticAverageError> {
    let batch = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(MultiLengthStochasticAverageError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    multi_length_stochastic_average_batch_par_slice(data, sweep, batch.to_non_batch())
}

#[inline]
pub fn multi_length_stochastic_average_batch_slice(
    data: &[f64],
    sweep: &MultiLengthStochasticAverageBatchRange,
    kernel: Kernel,
) -> Result<MultiLengthStochasticAverageBatchOutput, MultiLengthStochasticAverageError> {
    multi_length_stochastic_average_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn multi_length_stochastic_average_batch_par_slice(
    data: &[f64],
    sweep: &MultiLengthStochasticAverageBatchRange,
    kernel: Kernel,
) -> Result<MultiLengthStochasticAverageBatchOutput, MultiLengthStochasticAverageError> {
    multi_length_stochastic_average_batch_inner(data, sweep, kernel, true)
}

pub fn multi_length_stochastic_average_batch_inner(
    data: &[f64],
    sweep: &MultiLengthStochasticAverageBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<MultiLengthStochasticAverageBatchOutput, MultiLengthStochasticAverageError> {
    if data.is_empty() {
        return Err(MultiLengthStochasticAverageError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(MultiLengthStochasticAverageError::AllValuesNaN);
    }

    let combos = expand_grid_multi_length_stochastic_average(sweep)?;
    let max_valid = max_consecutive_valid_values(data);
    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(MultiLengthStochasticAverageError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;

    let resolved = combos
        .iter()
        .map(|params| resolve_params(params, Some(cols)))
        .collect::<Result<Vec<_>, _>>()?;

    for params in &resolved {
        let needed = total_warmup(*params) + 1;
        if max_valid < needed {
            return Err(MultiLengthStochasticAverageError::NotEnoughValidData {
                needed,
                valid: max_valid,
            });
        }
    }

    let warmups = resolved
        .iter()
        .map(|params| first.saturating_add(total_warmup(*params)).min(cols))
        .collect::<Vec<_>>();

    let mut values_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values_guard = ManuallyDrop::new(values_mu);
    let values_out =
        unsafe { std::slice::from_raw_parts_mut(values_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_out
                .par_chunks_mut(cols)
                .zip(resolved.par_iter())
                .for_each(|(row, params)| {
                    multi_length_stochastic_average_row_from_slice(data, *params, row);
                });
        }

        #[cfg(target_arch = "wasm32")]
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            multi_length_stochastic_average_row_from_slice(
                data,
                *params,
                &mut values_out[start..end],
            );
        }
    } else {
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            multi_length_stochastic_average_row_from_slice(
                data,
                *params,
                &mut values_out[start..end],
            );
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            values_guard.as_mut_ptr() as *mut f64,
            values_guard.len(),
            values_guard.capacity(),
        )
    };
    core::mem::forget(values_guard);

    Ok(MultiLengthStochasticAverageBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn multi_length_stochastic_average_batch_inner_into(
    data: &[f64],
    sweep: &MultiLengthStochasticAverageBatchRange,
    kernel: Kernel,
    values_out: &mut [f64],
) -> Result<Vec<MultiLengthStochasticAverageParams>, MultiLengthStochasticAverageError> {
    let out = multi_length_stochastic_average_batch_inner(data, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if values_out.len() != total {
        return Err(MultiLengthStochasticAverageError::OutputLengthMismatch {
            expected: total,
            got: values_out.len(),
        });
    }
    values_out.copy_from_slice(&out.values);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "multi_length_stochastic_average")]
#[pyo3(signature = (
    data,
    length=DEFAULT_LENGTH,
    presmooth=DEFAULT_PRESMOOTH,
    premethod=DEFAULT_SMOOTHING_METHOD,
    postsmooth=DEFAULT_POSTSMOOTH,
    postmethod=DEFAULT_SMOOTHING_METHOD,
    kernel=None
))]
pub fn multi_length_stochastic_average_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    presmooth: usize,
    premethod: &str,
    postsmooth: usize,
    postmethod: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = MultiLengthStochasticAverageInput::from_slice(
        data,
        MultiLengthStochasticAverageParams {
            length: Some(length),
            presmooth: Some(presmooth),
            premethod: Some(premethod.to_string()),
            postsmooth: Some(postsmooth),
            postmethod: Some(postmethod.to_string()),
        },
    );
    let out = py
        .allow_threads(|| multi_length_stochastic_average_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MultiLengthStochasticAverageStream")]
pub struct MultiLengthStochasticAverageStreamPy {
    stream: MultiLengthStochasticAverageStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MultiLengthStochasticAverageStreamPy {
    #[new]
    #[pyo3(signature = (
        length=DEFAULT_LENGTH,
        presmooth=DEFAULT_PRESMOOTH,
        premethod=DEFAULT_SMOOTHING_METHOD,
        postsmooth=DEFAULT_POSTSMOOTH,
        postmethod=DEFAULT_SMOOTHING_METHOD
    ))]
    fn new(
        length: usize,
        presmooth: usize,
        premethod: &str,
        postsmooth: usize,
        postmethod: &str,
    ) -> PyResult<Self> {
        let stream =
            MultiLengthStochasticAverageStream::try_new(MultiLengthStochasticAverageParams {
                length: Some(length),
                presmooth: Some(presmooth),
                premethod: Some(premethod.to_string()),
                postsmooth: Some(postsmooth),
                postmethod: Some(postmethod.to_string()),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "multi_length_stochastic_average_batch")]
#[pyo3(signature = (
    data,
    length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
    presmooth_range=(DEFAULT_PRESMOOTH, DEFAULT_PRESMOOTH, 0),
    premethod=DEFAULT_SMOOTHING_METHOD,
    postsmooth_range=(DEFAULT_POSTSMOOTH, DEFAULT_POSTSMOOTH, 0),
    postmethod=DEFAULT_SMOOTHING_METHOD,
    kernel=None
))]
pub fn multi_length_stochastic_average_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    presmooth_range: (usize, usize, usize),
    premethod: &str,
    postsmooth_range: (usize, usize, usize),
    postmethod: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = MultiLengthStochasticAverageBatchRange {
        length: length_range,
        presmooth: presmooth_range,
        postsmooth: postsmooth_range,
        premethod: Some(premethod.to_string()),
        postmethod: Some(postmethod.to_string()),
    };
    let combos = expand_grid_multi_length_stochastic_average(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let values_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let values_slice = unsafe { values_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            multi_length_stochastic_average_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                values_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", values_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "presmooths",
        combos
            .iter()
            .map(|combo| combo.presmooth.unwrap_or(DEFAULT_PRESMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "postsmooths",
        combos
            .iter()
            .map(|combo| combo.postsmooth.unwrap_or(DEFAULT_POSTSMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "premethods",
        PyList::new(
            py,
            combos.iter().map(|combo| {
                combo
                    .premethod
                    .as_deref()
                    .unwrap_or(DEFAULT_SMOOTHING_METHOD)
            }),
        )?,
    )?;
    dict.set_item(
        "postmethods",
        PyList::new(
            py,
            combos.iter().map(|combo| {
                combo
                    .postmethod
                    .as_deref()
                    .unwrap_or(DEFAULT_SMOOTHING_METHOD)
            }),
        )?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_multi_length_stochastic_average_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        multi_length_stochastic_average_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        multi_length_stochastic_average_batch_py,
        module
    )?)?;
    module.add_class::<MultiLengthStochasticAverageStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MultiLengthStochasticAverageJsOutput {
    pub values: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "multi_length_stochastic_average_js")]
pub fn multi_length_stochastic_average_js(
    data: &[f64],
    length: usize,
    presmooth: usize,
    premethod: String,
    postsmooth: usize,
    postmethod: String,
) -> Result<JsValue, JsValue> {
    let input = MultiLengthStochasticAverageInput::from_slice(
        data,
        MultiLengthStochasticAverageParams {
            length: Some(length),
            presmooth: Some(presmooth),
            premethod: Some(premethod),
            postsmooth: Some(postsmooth),
            postmethod: Some(postmethod),
        },
    );
    let out =
        multi_length_stochastic_average(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MultiLengthStochasticAverageJsOutput { values: out.values })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn multi_length_stochastic_average_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn multi_length_stochastic_average_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn multi_length_stochastic_average_into(
    data_ptr: *const f64,
    values_ptr: *mut f64,
    len: usize,
    length: usize,
    presmooth: usize,
    premethod: String,
    postsmooth: usize,
    postmethod: String,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || values_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = MultiLengthStochasticAverageInput::from_slice(
            data,
            MultiLengthStochasticAverageParams {
                length: Some(length),
                presmooth: Some(presmooth),
                premethod: Some(premethod),
                postsmooth: Some(postsmooth),
                postmethod: Some(postmethod),
            },
        );
        if data_ptr == values_ptr {
            let mut tmp = vec![0.0; len];
            multi_length_stochastic_average_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(values_ptr, len).copy_from_slice(&tmp);
        } else {
            multi_length_stochastic_average_into_slice(
                std::slice::from_raw_parts_mut(values_ptr, len),
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
pub struct MultiLengthStochasticAverageBatchJsConfig {
    pub length_range: (usize, usize, usize),
    pub presmooth_range: Option<(usize, usize, usize)>,
    pub premethod: Option<String>,
    pub postsmooth_range: Option<(usize, usize, usize)>,
    pub postmethod: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MultiLengthStochasticAverageBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MultiLengthStochasticAverageParams>,
    pub lengths: Vec<usize>,
    pub presmooths: Vec<usize>,
    pub postsmooths: Vec<usize>,
    pub premethods: Vec<String>,
    pub postmethods: Vec<String>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "multi_length_stochastic_average_batch_js")]
pub fn multi_length_stochastic_average_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MultiLengthStochasticAverageBatchJsConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = MultiLengthStochasticAverageBatchRange {
        length: config.length_range,
        presmooth: config
            .presmooth_range
            .unwrap_or((DEFAULT_PRESMOOTH, DEFAULT_PRESMOOTH, 0)),
        premethod: config
            .premethod
            .or_else(|| Some(DEFAULT_SMOOTHING_METHOD.to_string())),
        postsmooth: config
            .postsmooth_range
            .unwrap_or((DEFAULT_POSTSMOOTH, DEFAULT_POSTSMOOTH, 0)),
        postmethod: config
            .postmethod
            .or_else(|| Some(DEFAULT_SMOOTHING_METHOD.to_string())),
    };
    let out = multi_length_stochastic_average_batch_inner(
        data,
        &sweep,
        detect_best_batch_kernel().to_non_batch(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&MultiLengthStochasticAverageBatchJsOutput {
        lengths: out
            .combos
            .iter()
            .map(|p| p.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        presmooths: out
            .combos
            .iter()
            .map(|p| p.presmooth.unwrap_or(DEFAULT_PRESMOOTH))
            .collect(),
        postsmooths: out
            .combos
            .iter()
            .map(|p| p.postsmooth.unwrap_or(DEFAULT_POSTSMOOTH))
            .collect(),
        premethods: out
            .combos
            .iter()
            .map(|p| canonical_method_name(p.premethod.as_deref(), DEFAULT_SMOOTHING_METHOD))
            .collect(),
        postmethods: out
            .combos
            .iter()
            .map(|p| canonical_method_name(p.postmethod.as_deref(), DEFAULT_SMOOTHING_METHOD))
            .collect(),
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn multi_length_stochastic_average_batch_into(
    data_ptr: *const f64,
    values_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    presmooth_start: usize,
    presmooth_end: usize,
    presmooth_step: usize,
    premethod: String,
    postsmooth_start: usize,
    postsmooth_end: usize,
    postsmooth_step: usize,
    postmethod: String,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || values_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = MultiLengthStochasticAverageBatchRange {
        length: (length_start, length_end, length_step),
        presmooth: (presmooth_start, presmooth_end, presmooth_step),
        premethod: Some(premethod),
        postsmooth: (postsmooth_start, postsmooth_end, postsmooth_step),
        postmethod: Some(postmethod),
    };
    let combos = expand_grid_multi_length_stochastic_average(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let values_out = std::slice::from_raw_parts_mut(values_ptr, total);
        multi_length_stochastic_average_batch_inner_into(
            data,
            &sweep,
            detect_best_batch_kernel().to_non_batch(),
            values_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn multi_length_stochastic_average_output_into_js(
    data: &[f64],
    length: usize,
    presmooth: usize,
    premethod: String,
    postsmooth: usize,
    postmethod: String,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = multi_length_stochastic_average_js(
        data, length, presmooth, premethod, postsmooth, postmethod,
    )?;
    crate::write_wasm_object_f64_outputs(
        "multi_length_stochastic_average_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn multi_length_stochastic_average_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = multi_length_stochastic_average_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "multi_length_stochastic_average_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn sample_source(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                100.0
                    + i as f64 * 0.04
                    + (i as f64 * 0.17).sin() * 1.8
                    + (i as f64 * 0.03).cos() * 0.4
            })
            .collect()
    }

    fn sample_candles(len: usize) -> Candles {
        let open: Vec<f64> = (0..len)
            .map(|i| 100.0 + i as f64 * 0.03 + (i as f64 * 0.11).sin() * 1.2)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.19).cos() * 0.7)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.4 + (i as f64 * 0.07).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.4 - (i as f64 * 0.05).cos().abs() * 0.2)
            .collect();
        Candles::new(
            (0..len as i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; len],
        )
    }

    fn assert_series_eq(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (i, (&left, &right)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if left.is_nan() && right.is_nan() {
                continue;
            }
            assert!(
                (left - right).abs() <= tol,
                "mismatch at index {i}: left={left}, right={right}, tol={tol}"
            );
        }
    }

    #[test]
    fn multi_length_stochastic_average_output_contract() -> Result<(), Box<dyn Error>> {
        let data = sample_source(256);
        let out = multi_length_stochastic_average(&MultiLengthStochasticAverageInput::from_slice(
            &data,
            MultiLengthStochasticAverageParams::default(),
        ))?;

        assert_eq!(out.values.len(), data.len());
        let first_finite = out
            .values
            .iter()
            .position(|value| value.is_finite())
            .unwrap();
        assert!(first_finite >= 22);
        for &value in out.values.iter().filter(|value| value.is_finite()) {
            assert!((-1e-9..=100.0 + 1e-9).contains(&value));
        }
        Ok(())
    }

    #[test]
    fn multi_length_stochastic_average_rejects_invalid_parameters() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        let err = multi_length_stochastic_average(&MultiLengthStochasticAverageInput::from_slice(
            &data,
            MultiLengthStochasticAverageParams {
                length: Some(3),
                ..MultiLengthStochasticAverageParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            MultiLengthStochasticAverageError::InvalidLength { length: 3, .. }
        ));

        let err = multi_length_stochastic_average(&MultiLengthStochasticAverageInput::from_slice(
            &data,
            MultiLengthStochasticAverageParams {
                premethod: Some("ema".to_string()),
                ..MultiLengthStochasticAverageParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            MultiLengthStochasticAverageError::InvalidPreMethod { .. }
        ));
    }

    #[test]
    fn multi_length_stochastic_average_builder_supports_candles() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(220);
        let built = MultiLengthStochasticAverageBuilder::new()
            .length(12)
            .presmooth(5)
            .premethod("lsma")
            .postsmooth(4)
            .postmethod("sma")
            .apply(&candles, "hlc3")?;

        let direct =
            multi_length_stochastic_average(&MultiLengthStochasticAverageInput::from_candles(
                &candles,
                "hlc3",
                MultiLengthStochasticAverageParams {
                    length: Some(12),
                    presmooth: Some(5),
                    premethod: Some("lsma".to_string()),
                    postsmooth: Some(4),
                    postmethod: Some("sma".to_string()),
                },
            ))?;

        assert_series_eq(&built.values, &direct.values, 1e-12);
        Ok(())
    }

    #[test]
    fn multi_length_stochastic_average_stream_matches_batch_with_reset(
    ) -> Result<(), Box<dyn Error>> {
        let mut data = sample_source(220);
        data[110] = f64::NAN;
        let params = MultiLengthStochasticAverageParams {
            length: Some(12),
            presmooth: Some(5),
            premethod: Some("lsma".to_string()),
            postsmooth: Some(4),
            postmethod: Some("sma".to_string()),
        };

        let batch = multi_length_stochastic_average(
            &MultiLengthStochasticAverageInput::from_slice(&data, params.clone()),
        )?;
        let mut stream = MultiLengthStochasticAverageStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(data.len());

        for &value in &data {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }

        assert_eq!(stream.get_warmup_period(), 18);
        assert_series_eq(&batch.values, &streamed, 1e-12);
        Ok(())
    }

    #[test]
    fn multi_length_stochastic_average_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_source(192);
        let input = MultiLengthStochasticAverageInput::from_slice(
            &data,
            MultiLengthStochasticAverageParams {
                length: Some(16),
                presmooth: Some(6),
                premethod: Some("tma".to_string()),
                postsmooth: Some(5),
                postmethod: Some("lsma".to_string()),
            },
        );
        let api = multi_length_stochastic_average(&input)?;
        let mut out = vec![f64::NAN; data.len()];
        multi_length_stochastic_average_into_slice(&mut out, &input, Kernel::Auto)?;
        assert_series_eq(&api.values, &out, 1e-12);
        Ok(())
    }

    #[test]
    fn multi_length_stochastic_average_batch_single_param_matches_single(
    ) -> Result<(), Box<dyn Error>> {
        let data = sample_source(128);
        let sweep = MultiLengthStochasticAverageBatchRange {
            length: (12, 12, 0),
            presmooth: (5, 5, 0),
            premethod: Some("lsma".to_string()),
            postsmooth: (4, 4, 0),
            postmethod: Some("sma".to_string()),
        };
        let batch = multi_length_stochastic_average_batch_with_kernel(&data, &sweep, Kernel::Auto)?;
        let single =
            multi_length_stochastic_average(&MultiLengthStochasticAverageInput::from_slice(
                &data,
                MultiLengthStochasticAverageParams {
                    length: Some(12),
                    presmooth: Some(5),
                    premethod: Some("lsma".to_string()),
                    postsmooth: Some(4),
                    postmethod: Some("sma".to_string()),
                },
            ))?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.combos[0].length, Some(12));
        assert_eq!(batch.combos[0].presmooth, Some(5));
        assert_eq!(batch.combos[0].postsmooth, Some(4));
        assert_eq!(batch.combos[0].premethod.as_deref(), Some("lsma"));
        assert_eq!(batch.combos[0].postmethod.as_deref(), Some("sma"));
        assert_series_eq(&batch.values[..data.len()], &single.values, 1e-12);
        Ok(())
    }

    #[test]
    fn multi_length_stochastic_average_batch_metadata() -> Result<(), Box<dyn Error>> {
        let data = sample_source(96);
        let sweep = MultiLengthStochasticAverageBatchRange {
            length: (10, 14, 2),
            presmooth: (4, 6, 2),
            premethod: Some("sma".to_string()),
            postsmooth: (3, 5, 2),
            postmethod: Some("lsma".to_string()),
        };
        let batch = multi_length_stochastic_average_batch_with_kernel(&data, &sweep, Kernel::Auto)?;

        assert_eq!(batch.rows, 12);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.combos.len(), 12);
        assert_eq!(batch.values.len(), 12 * data.len());
        for combo in &batch.combos {
            assert_eq!(combo.premethod.as_deref(), Some("sma"));
            assert_eq!(combo.postmethod.as_deref(), Some("lsma"));
        }
        Ok(())
    }
}
