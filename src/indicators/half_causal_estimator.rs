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
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use chrono::{NaiveDateTime, Timelike};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_DATA_PERIOD: usize = 5;
const DEFAULT_FILTER_LENGTH: usize = 20;
const DEFAULT_KERNEL_WIDTH: f64 = 20.0;
const DEFAULT_MAXIMUM_CONFIDENCE_ADJUST: f64 = 100.0;
const DEFAULT_ENABLE_EXPECTED_VALUE: bool = false;
const DEFAULT_EXTRA_SMOOTHING: usize = 0;
const DEFAULT_SOURCE: &str = "volume";
const DAY_MS: i64 = 86_400_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    serde(rename_all = "snake_case")
)]
pub enum HalfCausalEstimatorKernelType {
    Gaussian,
    Epanechnikov,
    Triangular,
    Sinc,
}

impl Default for HalfCausalEstimatorKernelType {
    fn default() -> Self {
        Self::Epanechnikov
    }
}

impl HalfCausalEstimatorKernelType {
    #[inline]
    fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "gaussian" => Some(Self::Gaussian),
            "epanechnikov" => Some(Self::Epanechnikov),
            "triangular" => Some(Self::Triangular),
            "sinc" | "blackman_windowed_sinc" | "blackman-windowed-sinc" => Some(Self::Sinc),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    serde(rename_all = "snake_case")
)]
pub enum HalfCausalEstimatorConfidenceAdjust {
    Symmetric,
    Linear,
    None,
}

impl Default for HalfCausalEstimatorConfidenceAdjust {
    fn default() -> Self {
        Self::Symmetric
    }
}

impl HalfCausalEstimatorConfidenceAdjust {
    #[inline]
    fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "symmetric" => Some(Self::Symmetric),
            "linear" => Some(Self::Linear),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HalfCausalEstimatorParams {
    pub slots_per_day: Option<usize>,
    pub data_period: Option<usize>,
    pub filter_length: Option<usize>,
    pub kernel_width: Option<f64>,
    pub kernel_type: Option<HalfCausalEstimatorKernelType>,
    pub confidence_adjust: Option<HalfCausalEstimatorConfidenceAdjust>,
    pub maximum_confidence_adjust: Option<f64>,
    pub enable_expected_value: Option<bool>,
    pub extra_smoothing: Option<usize>,
}

impl Default for HalfCausalEstimatorParams {
    fn default() -> Self {
        Self {
            slots_per_day: None,
            data_period: Some(DEFAULT_DATA_PERIOD),
            filter_length: Some(DEFAULT_FILTER_LENGTH),
            kernel_width: Some(DEFAULT_KERNEL_WIDTH),
            kernel_type: Some(HalfCausalEstimatorKernelType::Epanechnikov),
            confidence_adjust: Some(HalfCausalEstimatorConfidenceAdjust::Symmetric),
            maximum_confidence_adjust: Some(DEFAULT_MAXIMUM_CONFIDENCE_ADJUST),
            enable_expected_value: Some(DEFAULT_ENABLE_EXPECTED_VALUE),
            extra_smoothing: Some(DEFAULT_EXTRA_SMOOTHING),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HalfCausalEstimatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorInput<'a> {
    pub data: HalfCausalEstimatorData<'a>,
    pub params: HalfCausalEstimatorParams,
}

impl<'a> HalfCausalEstimatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: HalfCausalEstimatorParams,
    ) -> Self {
        Self {
            data: HalfCausalEstimatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: HalfCausalEstimatorParams) -> Self {
        Self {
            data: HalfCausalEstimatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            HalfCausalEstimatorParams::default(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorOutput {
    pub estimate: Vec<f64>,
    pub expected_value: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBatchOutput {
    pub estimate_values: Vec<f64>,
    pub expected_value_values: Vec<f64>,
    pub combos: Vec<HalfCausalEstimatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HalfCausalEstimatorBatchOutput {
    #[inline]
    pub fn estimate_for(&self, row: usize) -> Option<&[f64]> {
        row.checked_mul(self.cols)
            .and_then(|start| self.estimate_values.get(start..start + self.cols))
    }

    #[inline]
    pub fn expected_value_for(&self, row: usize) -> Option<&[f64]> {
        row.checked_mul(self.cols)
            .and_then(|start| self.expected_value_values.get(start..start + self.cols))
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBatchRange {
    pub slots_per_day: Option<usize>,
    pub data_period: (usize, usize, usize),
    pub filter_length: (usize, usize, usize),
    pub kernel_width: (f64, f64, f64),
    pub maximum_confidence_adjust: (f64, f64, f64),
    pub extra_smoothing: (usize, usize, usize),
    pub kernel_type: HalfCausalEstimatorKernelType,
    pub confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    pub enable_expected_value: bool,
}

impl Default for HalfCausalEstimatorBatchRange {
    fn default() -> Self {
        Self {
            slots_per_day: None,
            data_period: (DEFAULT_DATA_PERIOD, DEFAULT_DATA_PERIOD, 0),
            filter_length: (DEFAULT_FILTER_LENGTH, DEFAULT_FILTER_LENGTH, 0),
            kernel_width: (DEFAULT_KERNEL_WIDTH, DEFAULT_KERNEL_WIDTH, 0.0),
            maximum_confidence_adjust: (
                DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
                DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
                0.0,
            ),
            extra_smoothing: (DEFAULT_EXTRA_SMOOTHING, DEFAULT_EXTRA_SMOOTHING, 0),
            kernel_type: HalfCausalEstimatorKernelType::Epanechnikov,
            confidence_adjust: HalfCausalEstimatorConfidenceAdjust::Symmetric,
            enable_expected_value: DEFAULT_ENABLE_EXPECTED_VALUE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBuilder {
    slots_per_day: Option<usize>,
    data_period: Option<usize>,
    filter_length: Option<usize>,
    kernel_width: Option<f64>,
    kernel_type: Option<HalfCausalEstimatorKernelType>,
    confidence_adjust: Option<HalfCausalEstimatorConfidenceAdjust>,
    maximum_confidence_adjust: Option<f64>,
    enable_expected_value: Option<bool>,
    extra_smoothing: Option<usize>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for HalfCausalEstimatorBuilder {
    fn default() -> Self {
        Self {
            slots_per_day: None,
            data_period: None,
            filter_length: None,
            kernel_width: None,
            kernel_type: None,
            confidence_adjust: None,
            maximum_confidence_adjust: None,
            enable_expected_value: None,
            extra_smoothing: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HalfCausalEstimatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn slots_per_day(mut self, slots_per_day: usize) -> Self {
        self.slots_per_day = Some(slots_per_day);
        self
    }

    #[inline]
    pub fn data_period(mut self, data_period: usize) -> Self {
        self.data_period = Some(data_period);
        self
    }

    #[inline]
    pub fn filter_length(mut self, filter_length: usize) -> Self {
        self.filter_length = Some(filter_length);
        self
    }

    #[inline]
    pub fn kernel_width(mut self, kernel_width: f64) -> Self {
        self.kernel_width = Some(kernel_width);
        self
    }

    #[inline]
    pub fn kernel_type(mut self, kernel_type: HalfCausalEstimatorKernelType) -> Self {
        self.kernel_type = Some(kernel_type);
        self
    }

    #[inline]
    pub fn confidence_adjust(
        mut self,
        confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    ) -> Self {
        self.confidence_adjust = Some(confidence_adjust);
        self
    }

    #[inline]
    pub fn maximum_confidence_adjust(mut self, maximum_confidence_adjust: f64) -> Self {
        self.maximum_confidence_adjust = Some(maximum_confidence_adjust);
        self
    }

    #[inline]
    pub fn enable_expected_value(mut self, enable_expected_value: bool) -> Self {
        self.enable_expected_value = Some(enable_expected_value);
        self
    }

    #[inline]
    pub fn extra_smoothing(mut self, extra_smoothing: usize) -> Self {
        self.extra_smoothing = Some(extra_smoothing);
        self
    }

    #[inline]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    fn build_params(&self) -> HalfCausalEstimatorParams {
        HalfCausalEstimatorParams {
            slots_per_day: self.slots_per_day,
            data_period: self.data_period,
            filter_length: self.filter_length,
            kernel_width: self.kernel_width,
            kernel_type: self.kernel_type,
            confidence_adjust: self.confidence_adjust,
            maximum_confidence_adjust: self.maximum_confidence_adjust,
            enable_expected_value: self.enable_expected_value,
            extra_smoothing: self.extra_smoothing,
        }
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
        let input = HalfCausalEstimatorInput::from_slice(data, self.build_params());
        half_causal_estimator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
        let source = self.source.as_deref().unwrap_or(DEFAULT_SOURCE);
        let input = HalfCausalEstimatorInput::from_candles(candles, source, self.build_params());
        half_causal_estimator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<HalfCausalEstimatorStream, HalfCausalEstimatorError> {
        HalfCausalEstimatorStream::try_new(self.build_params())
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBatchBuilder {
    range: HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    source: Option<String>,
}

impl Default for HalfCausalEstimatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: HalfCausalEstimatorBatchRange::default(),
            kernel: Kernel::Auto,
            source: None,
        }
    }
}

impl HalfCausalEstimatorBatchBuilder {
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
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    #[inline]
    pub fn slots_per_day(mut self, slots_per_day: usize) -> Self {
        self.range.slots_per_day = Some(slots_per_day);
        self
    }

    #[inline]
    pub fn data_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.data_period = (start, end, step);
        self
    }

    #[inline]
    pub fn filter_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.filter_length = (start, end, step);
        self
    }

    #[inline]
    pub fn kernel_width_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.kernel_width = (start, end, step);
        self
    }

    #[inline]
    pub fn maximum_confidence_adjust_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.maximum_confidence_adjust = (start, end, step);
        self
    }

    #[inline]
    pub fn extra_smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.extra_smoothing = (start, end, step);
        self
    }

    #[inline]
    pub fn kernel_type(mut self, kernel_type: HalfCausalEstimatorKernelType) -> Self {
        self.range.kernel_type = kernel_type;
        self
    }

    #[inline]
    pub fn confidence_adjust(
        mut self,
        confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    ) -> Self {
        self.range.confidence_adjust = confidence_adjust;
        self
    }

    #[inline]
    pub fn enable_expected_value(mut self, enable_expected_value: bool) -> Self {
        self.range.enable_expected_value = enable_expected_value;
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
        half_causal_estimator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
        let source = self.source.as_deref().unwrap_or(DEFAULT_SOURCE);
        let params = HalfCausalEstimatorParams {
            slots_per_day: self.range.slots_per_day,
            data_period: Some(self.range.data_period.0),
            filter_length: Some(self.range.filter_length.0),
            kernel_width: Some(self.range.kernel_width.0),
            kernel_type: Some(self.range.kernel_type),
            confidence_adjust: Some(self.range.confidence_adjust),
            maximum_confidence_adjust: Some(self.range.maximum_confidence_adjust.0),
            enable_expected_value: Some(self.range.enable_expected_value),
            extra_smoothing: Some(self.range.extra_smoothing.0),
        };
        let prepared = prepare_source_and_slots(&HalfCausalEstimatorInput::from_candles(
            candles, source, params,
        ))?;
        let mut sweep = self.range.clone();
        if sweep.slots_per_day.is_none() {
            sweep.slots_per_day = Some(prepared.slots_per_day);
        }
        half_causal_estimator_batch_with_kernel(&prepared.values, &sweep, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum HalfCausalEstimatorError {
    #[error("half_causal_estimator: Input data slice is empty.")]
    EmptyInputData,
    #[error("half_causal_estimator: All values are NaN.")]
    AllValuesNaN,
    #[error("half_causal_estimator: Missing slots_per_day for slice input.")]
    MissingSlotsPerDay,
    #[error("half_causal_estimator: Invalid slots_per_day: {slots_per_day}")]
    InvalidSlotsPerDay { slots_per_day: usize },
    #[error("half_causal_estimator: Invalid data_period: {data_period}")]
    InvalidDataPeriod { data_period: usize },
    #[error("half_causal_estimator: Invalid filter_length: {filter_length}")]
    InvalidFilterLength { filter_length: usize },
    #[error("half_causal_estimator: Invalid kernel_width: {kernel_width}")]
    InvalidKernelWidth { kernel_width: f64 },
    #[error(
        "half_causal_estimator: Invalid maximum_confidence_adjust: {maximum_confidence_adjust}"
    )]
    InvalidMaximumConfidenceAdjust { maximum_confidence_adjust: f64 },
    #[error("half_causal_estimator: Invalid source: {source_name}")]
    InvalidSource { source_name: String },
    #[error("half_causal_estimator: Unable to infer minute timeframe from timestamps.")]
    UnableToInferMinuteTimeframe,
    #[error("half_causal_estimator: Invalid timestamp: {timestamp}")]
    InvalidTimestamp { timestamp: i64 },
    #[error(
        "half_causal_estimator: Output length mismatch: expected = {expected}, estimate = {estimate_got}, expected_value = {expected_value_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        estimate_got: usize,
        expected_value_got: usize,
    },
    #[error("half_causal_estimator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("half_causal_estimator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    slots_per_day: usize,
    data_period: usize,
    filter_length: usize,
    real_filter_length: usize,
    window_size: usize,
    kernel_width: f64,
    kernel_type: HalfCausalEstimatorKernelType,
    confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    maximum_confidence_adjust_factor: f64,
    enable_expected_value: bool,
    extra_smoothing: usize,
}

#[derive(Debug, Clone)]
struct PreparedInput<'a> {
    values: Cow<'a, [f64]>,
    slots: PreparedSlots,
    slots_per_day: usize,
}

#[derive(Debug, Clone)]
enum PreparedSlots {
    Sequential { slots_per_day: usize },
    Explicit(Vec<usize>),
}

#[derive(Debug, Clone)]
struct TimeOfDayBucket {
    values: Vec<f64>,
    next: usize,
    count: usize,
    sum: f64,
    sum_sq: f64,
    bounded: bool,
}

impl TimeOfDayBucket {
    #[inline]
    fn new(capacity: usize) -> Self {
        let bounded = capacity > 0;
        Self {
            values: if bounded {
                vec![0.0; capacity]
            } else {
                Vec::new()
            },
            next: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            bounded,
        }
    }

    #[inline]
    fn add(&mut self, value: f64) {
        if self.bounded {
            if self.values.is_empty() {
                return;
            }
            if self.count < self.values.len() {
                self.values[self.next] = value;
                self.count += 1;
            } else {
                let old = self.values[self.next];
                self.sum -= old;
                self.sum_sq -= old * old;
                self.values[self.next] = value;
            }
            self.next += 1;
            if self.next == self.values.len() {
                self.next = 0;
            }
        } else {
            self.values.push(value);
            self.count += 1;
        }
        self.sum += value;
        self.sum_sq += value * value;
    }

    #[inline]
    fn has_values(&self) -> bool {
        self.count > 0
    }

    #[inline]
    fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f64)
        }
    }

    #[inline]
    fn stdev(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mean = self.sum / self.count as f64;
        let variance = (self.sum_sq / self.count as f64) - mean * mean;
        Some(variance.max(0.0).sqrt())
    }
}

#[derive(Debug, Clone)]
struct TimeOfDayStore {
    buckets: Vec<TimeOfDayBucket>,
}

impl TimeOfDayStore {
    #[inline]
    fn new(slots_per_day: usize, data_period: usize) -> Self {
        let mut buckets = Vec::with_capacity(slots_per_day);
        for _ in 0..slots_per_day {
            buckets.push(TimeOfDayBucket::new(data_period));
        }
        Self { buckets }
    }

    #[inline]
    fn add(&mut self, slot: usize, value: f64) {
        self.buckets[slot].add(value);
    }

    #[inline]
    fn has_values(&self, slot: usize) -> bool {
        self.buckets[slot].has_values()
    }

    #[inline]
    fn mean(&self, slot: usize) -> Option<f64> {
        self.buckets[slot].mean()
    }

    #[inline]
    fn icv(&self, slot: usize, maximum_confidence_adjust_factor: f64) -> f64 {
        let bucket = &self.buckets[slot];
        if bucket.count == 0 {
            return 1.0;
        }
        let avg = bucket.sum / bucket.count as f64;
        if avg.abs() <= f64::EPSILON {
            return 1.0;
        }
        let variance = (bucket.sum_sq / bucket.count as f64) - avg * avg;
        let stdev = variance.max(0.0).sqrt();
        let ratio = (stdev / avg).clamp(0.0, 1.0);
        1.0 - ratio * maximum_confidence_adjust_factor
    }
}

#[derive(Debug, Clone)]
struct FixedFrontBuffer {
    values: Vec<f64>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl FixedFrontBuffer {
    #[inline]
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![0.0; capacity],
            capacity,
            head: 0,
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, value: f64) {
        if self.capacity == 0 {
            return;
        }
        if self.len == 0 {
            self.values[0] = value;
            self.len = 1;
            self.head = 0;
            return;
        }
        self.head = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };
        self.values[self.head] = value;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.len).map(move |offset| {
            let mut index = self.head + offset;
            if index >= self.capacity {
                index -= self.capacity;
            }
            self.values[index]
        })
    }
}

#[derive(Debug, Clone)]
struct FillWmaState {
    length: usize,
    first: Option<f64>,
    values: VecDeque<f64>,
    denominator: f64,
}

impl FillWmaState {
    #[inline]
    fn new(extra_smoothing: usize) -> Self {
        let length = extra_smoothing + 1;
        Self {
            length,
            first: None,
            values: VecDeque::with_capacity(length),
            denominator: (length * (length + 1) / 2) as f64,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }
        let first = *self.first.get_or_insert(value);
        if self.length == 1 {
            return Some(value);
        }

        self.values.push_front(value);
        if self.values.len() > self.length {
            let _ = self.values.pop_back();
        }

        let mut sum = 0.0;
        for i in 0..self.length {
            let sample = self.values.get(i).copied().unwrap_or(first);
            sum += sample * (self.length - i) as f64;
        }
        Some(sum / self.denominator)
    }
}

#[derive(Debug, Clone)]
struct HalfCausalEstimatorContext {
    params: ResolvedParams,
    store: TimeOfDayStore,
    source_buffer: FixedFrontBuffer,
    average_buffer: FixedFrontBuffer,
    wma: FillWmaState,
    kernel: Vec<f64>,
    future_values: Vec<f64>,
    future_weights: Vec<f64>,
    ready: bool,
    prev_slot: Option<usize>,
    index: usize,
}

impl HalfCausalEstimatorContext {
    #[inline]
    fn new(params: ResolvedParams) -> Self {
        Self {
            store: TimeOfDayStore::new(params.slots_per_day, params.data_period),
            source_buffer: FixedFrontBuffer::new(params.real_filter_length),
            average_buffer: FixedFrontBuffer::new(params.real_filter_length),
            wma: FillWmaState::new(params.extra_smoothing),
            kernel: build_kernel(params),
            future_values: Vec::with_capacity(params.real_filter_length.saturating_sub(1)),
            future_weights: Vec::with_capacity(params.real_filter_length.saturating_sub(1)),
            ready: false,
            prev_slot: None,
            index: 0,
            params,
        }
    }

    #[inline]
    fn update(&mut self, value: f64, slot: usize) -> (Option<f64>, Option<f64>) {
        let session_start = self.prev_slot.map(|prev| slot <= prev).unwrap_or(true);
        self.prev_slot = Some(slot);

        if !self.ready && self.index > self.params.window_size && session_start {
            self.ready = true;
        }

        self.source_buffer.push(value);
        self.average_buffer
            .push(self.store.mean(slot).unwrap_or(f64::NAN));

        let estimate_raw = if self.ready && self.source_buffer.is_full() {
            self.compute_window(slot, true)
        } else {
            None
        };

        let estimate = estimate_raw.and_then(|x| self.wma.update(x));
        let expected_value =
            if self.params.enable_expected_value && self.ready && self.average_buffer.is_full() {
                self.compute_window(slot, false)
            } else {
                None
            };

        if value.is_finite() {
            self.store.add(slot, value);
        }

        self.index += 1;
        (estimate, expected_value)
    }

    #[inline]
    fn compute_window(&mut self, slot: usize, apply_confidence_adjust: bool) -> Option<f64> {
        let future_len = self.params.real_filter_length.saturating_sub(1);
        let uses_confidence = apply_confidence_adjust
            && !matches!(
                self.params.confidence_adjust,
                HalfCausalEstimatorConfidenceAdjust::None
            );
        collect_future_into(
            &self.store,
            slot,
            future_len,
            self.params.maximum_confidence_adjust_factor,
            uses_confidence,
            &mut self.future_values,
            &mut self.future_weights,
        )?;

        let causal_values = if apply_confidence_adjust {
            &self.source_buffer
        } else {
            &self.average_buffer
        };
        if causal_values.len != self.params.real_filter_length {
            return None;
        }
        if self.future_values.len() + causal_values.len != self.params.window_size {
            return None;
        }
        if uses_confidence && self.future_weights.len() != future_len {
            return None;
        }

        let mut sum = 0.0;
        let mut kernel_index = 0usize;
        let linear_fill = if uses_confidence
            && matches!(
                self.params.confidence_adjust,
                HalfCausalEstimatorConfidenceAdjust::Linear
            ) {
            let weight_sum: f64 = self.future_weights.iter().copied().sum();
            if self.params.real_filter_length > 1 {
                2.0 - weight_sum / future_len as f64
            } else {
                1.0
            }
        } else {
            1.0
        };

        for i in 0..future_len {
            let value = self.future_values[i];
            if !value.is_finite() {
                return None;
            }
            let confidence = if uses_confidence {
                self.future_weights[i]
            } else {
                1.0
            };
            sum += value * confidence * self.kernel[kernel_index];
            kernel_index += 1;
        }

        for (i, value) in causal_values.iter().enumerate() {
            if !value.is_finite() {
                return None;
            }
            let confidence = match self.params.confidence_adjust {
                HalfCausalEstimatorConfidenceAdjust::None => 1.0,
                HalfCausalEstimatorConfidenceAdjust::Symmetric if apply_confidence_adjust => {
                    if i == 0 {
                        1.0
                    } else {
                        2.0 - self.future_weights[future_len - i]
                    }
                }
                HalfCausalEstimatorConfidenceAdjust::Linear if apply_confidence_adjust => {
                    linear_fill
                }
                _ => 1.0,
            };
            sum += value * confidence * self.kernel[kernel_index];
            kernel_index += 1;
        }

        Some(sum)
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorStream {
    ctx: HalfCausalEstimatorContext,
    next_slot: usize,
}

impl HalfCausalEstimatorStream {
    #[inline]
    pub fn try_new(params: HalfCausalEstimatorParams) -> Result<Self, HalfCausalEstimatorError> {
        let slots_per_day = params
            .slots_per_day
            .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
        let resolved = resolve_params(&params, slots_per_day)?;
        Ok(Self {
            ctx: HalfCausalEstimatorContext::new(resolved),
            next_slot: 0,
        })
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.ctx.params.slots_per_day + self.ctx.params.window_size
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> (Option<f64>, Option<f64>) {
        let out = self.ctx.update(value, self.next_slot);
        self.next_slot += 1;
        if self.next_slot == self.ctx.params.slots_per_day {
            self.next_slot = 0;
        }
        out
    }
}

#[inline(always)]
fn resolve_params(
    params: &HalfCausalEstimatorParams,
    slots_per_day: usize,
) -> Result<ResolvedParams, HalfCausalEstimatorError> {
    if slots_per_day < 2 {
        return Err(HalfCausalEstimatorError::InvalidSlotsPerDay { slots_per_day });
    }

    let data_period = params.data_period.unwrap_or(DEFAULT_DATA_PERIOD);
    if data_period == usize::MAX {
        return Err(HalfCausalEstimatorError::InvalidDataPeriod { data_period });
    }

    let filter_length = params.filter_length.unwrap_or(DEFAULT_FILTER_LENGTH);
    if filter_length < 2 {
        return Err(HalfCausalEstimatorError::InvalidFilterLength { filter_length });
    }

    let kernel_width = params.kernel_width.unwrap_or(DEFAULT_KERNEL_WIDTH);
    if !kernel_width.is_finite() || kernel_width <= 0.0 {
        return Err(HalfCausalEstimatorError::InvalidKernelWidth { kernel_width });
    }

    let maximum_confidence_adjust = params
        .maximum_confidence_adjust
        .unwrap_or(DEFAULT_MAXIMUM_CONFIDENCE_ADJUST);
    if !maximum_confidence_adjust.is_finite() || !(0.0..=100.0).contains(&maximum_confidence_adjust)
    {
        return Err(HalfCausalEstimatorError::InvalidMaximumConfidenceAdjust {
            maximum_confidence_adjust,
        });
    }

    let kernel_type = params.kernel_type.unwrap_or_default();
    let confidence_adjust = params.confidence_adjust.unwrap_or_default();
    let extra_smoothing = params.extra_smoothing.unwrap_or(DEFAULT_EXTRA_SMOOTHING);
    let real_filter_length = if matches!(kernel_type, HalfCausalEstimatorKernelType::Sinc) {
        filter_length.saturating_mul(2)
    } else {
        filter_length
    };

    Ok(ResolvedParams {
        slots_per_day,
        data_period,
        filter_length,
        real_filter_length,
        window_size: real_filter_length.saturating_mul(2).saturating_sub(1),
        kernel_width,
        kernel_type,
        confidence_adjust,
        maximum_confidence_adjust_factor: maximum_confidence_adjust * 0.01,
        enable_expected_value: params
            .enable_expected_value
            .unwrap_or(DEFAULT_ENABLE_EXPECTED_VALUE),
        extra_smoothing,
    })
}

#[inline(always)]
fn gaussian_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    (-ratio * ratio * 0.25).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[inline(always)]
fn epanechnikov_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    if ratio.abs() <= 1.0 {
        0.75 * (1.0 - ratio * ratio)
    } else {
        0.0
    }
}

#[inline(always)]
fn triangular_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    if ratio.abs() <= 1.0 {
        1.0 - ratio.abs()
    } else {
        0.0
    }
}

#[inline(always)]
fn blackman(index: f64, length: f64) -> f64 {
    0.42 - 0.5 * ((2.0 * std::f64::consts::PI * index) / (length - 1.0)).cos()
        + 0.08 * ((4.0 * std::f64::consts::PI * index) / (length - 1.0)).cos()
}

#[inline(always)]
fn sinc(centered_index: f64, width: f64) -> f64 {
    let fc = 0.5 / width;
    if centered_index.abs() <= f64::EPSILON {
        1.0
    } else {
        let x = std::f64::consts::PI * fc * centered_index;
        x.sin() / x
    }
}

#[inline(always)]
fn build_kernel(params: ResolvedParams) -> Vec<f64> {
    let mut kernel = Vec::with_capacity(params.window_size);
    let center = (params.window_size - 1) as f64 * 0.5;
    let length = params.window_size as f64;
    let mut normalization = 0.0;

    for i in 0..params.window_size {
        let index = i as f64;
        let centered = index - center;
        let weight = match params.kernel_type {
            HalfCausalEstimatorKernelType::Gaussian => {
                gaussian_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Epanechnikov => {
                epanechnikov_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Triangular => {
                triangular_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Sinc => {
                sinc(centered, params.kernel_width) * blackman(index, length)
            }
        };
        normalization += weight;
        kernel.push(weight);
    }

    if normalization != 0.0 {
        for weight in &mut kernel {
            *weight /= normalization;
        }
    }

    kernel
}

#[inline(always)]
fn collect_future_into(
    store: &TimeOfDayStore,
    slot: usize,
    needed: usize,
    maximum_confidence_adjust_factor: f64,
    collect_weights: bool,
    values: &mut Vec<f64>,
    weights: &mut Vec<f64>,
) -> Option<()> {
    values.clear();
    weights.clear();
    if needed == 0 {
        return Some(());
    }

    let slots_per_day = store.buckets.len();
    if slots_per_day == 0 {
        return None;
    }

    let mut offset = 1usize;
    let mut saw_valid = false;
    while values.len() < needed {
        let next_slot = (slot + offset) % slots_per_day;
        if store.has_values(next_slot) {
            saw_valid = true;
            values.push(store.mean(next_slot).unwrap_or(f64::NAN));
            if collect_weights {
                weights.push(
                    store
                        .icv(next_slot, maximum_confidence_adjust_factor)
                        .max(0.0),
                );
            }
        }
        offset += 1;
        if offset > slots_per_day.saturating_mul(4) && !saw_valid {
            return None;
        }
    }
    values.reverse();
    if collect_weights {
        weights.reverse();
    }
    Some(())
}

#[inline]
fn infer_slots_per_day(timestamps: &[i64]) -> Result<usize, HalfCausalEstimatorError> {
    let mut min_positive = i64::MAX;
    for pair in timestamps.windows(2) {
        let delta = pair[1] - pair[0];
        if delta > 0 && delta < DAY_MS && delta < min_positive {
            min_positive = delta;
        }
    }

    if min_positive == i64::MAX || min_positive % 60_000 != 0 {
        return Err(HalfCausalEstimatorError::UnableToInferMinuteTimeframe);
    }

    let minutes = (min_positive / 60_000) as usize;
    if minutes == 0 || 1440 % minutes != 0 {
        return Err(HalfCausalEstimatorError::UnableToInferMinuteTimeframe);
    }
    Ok(1440 / minutes)
}

#[inline]
fn slot_from_timestamp(
    timestamp: i64,
    slots_per_day: usize,
) -> Result<usize, HalfCausalEstimatorError> {
    let seconds = timestamp / 1000;
    let nanos = ((timestamp % 1000) * 1_000_000) as u32;
    let dt = NaiveDateTime::from_timestamp_opt(seconds, nanos)
        .ok_or(HalfCausalEstimatorError::InvalidTimestamp { timestamp })?;
    let minutes = dt.hour() as usize * 60 + dt.minute() as usize;
    let minutes_per_slot = 1440 / slots_per_day;
    Ok(minutes / minutes_per_slot)
}

#[inline]
fn source_from_candles<'a>(
    candles: &'a Candles,
    source: &str,
    slots_per_day: usize,
) -> Result<Cow<'a, [f64]>, HalfCausalEstimatorError> {
    match source.to_ascii_lowercase().as_str() {
        "volume" => Ok(Cow::Borrowed(&candles.volume)),
        "tr" => Ok(Cow::Owned(
            candles
                .high
                .iter()
                .zip(candles.low.iter())
                .map(|(&high, &low)| {
                    if low.is_finite() && low != 0.0 {
                        (high - low) / low * 100.0
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        )),
        "change" => {
            let mut out = Vec::with_capacity(candles.close.len());
            let mut prev = None;
            for &close in &candles.close {
                let prior = prev.unwrap_or(close);
                let denom = close.min(prior);
                if denom.is_finite() && denom != 0.0 {
                    out.push((close - prior).abs() / denom * 100.0);
                } else if prev.is_none() {
                    out.push(0.0);
                } else {
                    out.push(f64::NAN);
                }
                prev = Some(close);
            }
            Ok(Cow::Owned(out))
        }
        "test" => {
            let mut out = Vec::with_capacity(candles.timestamp.len());
            for &timestamp in &candles.timestamp {
                let slot = slot_from_timestamp(timestamp, slots_per_day)?;
                let cycle = slots_per_day as f64;
                let value = ((std::f64::consts::PI / cycle) * slot as f64).sin();
                out.push((value * value).max(0.0) * 100.0);
            }
            Ok(Cow::Owned(out))
        }
        _ => Err(HalfCausalEstimatorError::InvalidSource {
            source_name: source.to_string(),
        }),
    }
}

#[inline]
fn prepare_source_and_slots<'a>(
    input: &HalfCausalEstimatorInput<'a>,
) -> Result<PreparedInput<'a>, HalfCausalEstimatorError> {
    match &input.data {
        HalfCausalEstimatorData::Slice(values) => {
            if values.is_empty() {
                return Err(HalfCausalEstimatorError::EmptyInputData);
            }
            let slots_per_day = input
                .params
                .slots_per_day
                .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
            Ok(PreparedInput {
                values: Cow::Borrowed(values),
                slots: PreparedSlots::Sequential { slots_per_day },
                slots_per_day,
            })
        }
        HalfCausalEstimatorData::Candles { candles, source } => {
            if candles.close.is_empty() {
                return Err(HalfCausalEstimatorError::EmptyInputData);
            }
            let slots_per_day = match input.params.slots_per_day {
                Some(slots) => slots,
                None => infer_slots_per_day(&candles.timestamp)?,
            };
            let mut slots = Vec::with_capacity(candles.timestamp.len());
            for &timestamp in &candles.timestamp {
                slots.push(slot_from_timestamp(timestamp, slots_per_day)?);
            }
            let values = source_from_candles(candles, source, slots_per_day)?;
            Ok(PreparedInput {
                values,
                slots: PreparedSlots::Explicit(slots),
                slots_per_day,
            })
        }
    }
}

#[inline]
fn first_finite(values: &[f64]) -> usize {
    values
        .iter()
        .position(|value| value.is_finite())
        .unwrap_or(values.len())
}

#[inline]
fn resolve_and_prepare<'a>(
    input: &HalfCausalEstimatorInput<'a>,
) -> Result<(PreparedInput<'a>, ResolvedParams), HalfCausalEstimatorError> {
    let prepared = prepare_source_and_slots(input)?;
    let first = first_finite(&prepared.values);
    if first >= prepared.values.len() {
        return Err(HalfCausalEstimatorError::AllValuesNaN);
    }
    let resolved = resolve_params(&input.params, prepared.slots_per_day)?;
    Ok((prepared, resolved))
}

#[inline]
fn compute_row(
    values: &[f64],
    slots: &PreparedSlots,
    params: ResolvedParams,
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
) {
    let mut ctx = HalfCausalEstimatorContext::new(params);
    match slots {
        PreparedSlots::Sequential { slots_per_day } => {
            let mut slot = 0usize;
            for i in 0..values.len() {
                let (estimate, expected_value) = ctx.update(values[i], slot);
                estimate_out[i] = estimate.unwrap_or(f64::NAN);
                expected_value_out[i] = expected_value.unwrap_or(f64::NAN);
                slot += 1;
                if slot == *slots_per_day {
                    slot = 0;
                }
            }
        }
        PreparedSlots::Explicit(slots) => {
            for i in 0..values.len() {
                let (estimate, expected_value) = ctx.update(values[i], slots[i]);
                estimate_out[i] = estimate.unwrap_or(f64::NAN);
                expected_value_out[i] = expected_value.unwrap_or(f64::NAN);
            }
        }
    }
}

#[inline]
pub fn half_causal_estimator(
    input: &HalfCausalEstimatorInput<'_>,
) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
    half_causal_estimator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn half_causal_estimator_with_kernel(
    input: &HalfCausalEstimatorInput<'_>,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
    let _ = kernel;
    let (prepared, params) = resolve_and_prepare(input)?;
    let mut estimate = alloc_uninit_f64(prepared.values.len());
    let mut expected_value = alloc_uninit_f64(prepared.values.len());
    compute_row(
        &prepared.values,
        &prepared.slots,
        params,
        &mut estimate,
        &mut expected_value,
    );
    Ok(HalfCausalEstimatorOutput {
        estimate,
        expected_value,
    })
}

#[inline]
pub fn half_causal_estimator_into_slices(
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
    input: &HalfCausalEstimatorInput<'_>,
    kernel: Kernel,
) -> Result<(), HalfCausalEstimatorError> {
    let _ = kernel;
    let (prepared, params) = resolve_and_prepare(input)?;
    let expected = prepared.values.len();
    if estimate_out.len() != expected || expected_value_out.len() != expected {
        return Err(HalfCausalEstimatorError::OutputLengthMismatch {
            expected,
            estimate_got: estimate_out.len(),
            expected_value_got: expected_value_out.len(),
        });
    }
    compute_row(
        &prepared.values,
        &prepared.slots,
        params,
        estimate_out,
        expected_value_out,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn half_causal_estimator_into(
    input: &HalfCausalEstimatorInput<'_>,
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
) -> Result<(), HalfCausalEstimatorError> {
    half_causal_estimator_into_slices(estimate_out, expected_value_out, input, Kernel::Auto)
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, HalfCausalEstimatorError> {
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
        return Err(HalfCausalEstimatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, HalfCausalEstimatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(HalfCausalEstimatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 || (start - end).abs() <= f64::EPSILON {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end + 1e-12 {
            out.push(value);
            value += step;
        }
    } else {
        let mut value = start;
        while value >= end - 1e-12 {
            out.push(value);
            value -= step.abs();
        }
    }
    if out.is_empty() {
        return Err(HalfCausalEstimatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
fn expand_grid_half_causal_estimator(
    sweep: &HalfCausalEstimatorBatchRange,
) -> Result<Vec<HalfCausalEstimatorParams>, HalfCausalEstimatorError> {
    let data_periods = expand_axis_usize(sweep.data_period)?;
    let filter_lengths = expand_axis_usize(sweep.filter_length)?;
    let kernel_widths = expand_axis_f64(sweep.kernel_width)?;
    let maximum_confidence_adjusts = expand_axis_f64(sweep.maximum_confidence_adjust)?;
    let extra_smoothings = expand_axis_usize(sweep.extra_smoothing)?;

    let mut combos = Vec::new();
    for data_period in data_periods {
        for filter_length in filter_lengths.iter().copied() {
            for kernel_width in kernel_widths.iter().copied() {
                for maximum_confidence_adjust in maximum_confidence_adjusts.iter().copied() {
                    for extra_smoothing in extra_smoothings.iter().copied() {
                        let params = HalfCausalEstimatorParams {
                            slots_per_day: sweep.slots_per_day,
                            data_period: Some(data_period),
                            filter_length: Some(filter_length),
                            kernel_width: Some(kernel_width),
                            kernel_type: Some(sweep.kernel_type),
                            confidence_adjust: Some(sweep.confidence_adjust),
                            maximum_confidence_adjust: Some(maximum_confidence_adjust),
                            enable_expected_value: Some(sweep.enable_expected_value),
                            extra_smoothing: Some(extra_smoothing),
                        };
                        let slots_per_day = params
                            .slots_per_day
                            .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
                        let _ = resolve_params(&params, slots_per_day)?;
                        combos.push(params);
                    }
                }
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn half_causal_estimator_batch_with_kernel(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HalfCausalEstimatorError::InvalidKernelForBatch(other)),
    };
    half_causal_estimator_batch_inner(data, sweep, batch_kernel.to_non_batch(), true)
}

#[inline]
pub fn half_causal_estimator_batch_slice(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    half_causal_estimator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn half_causal_estimator_batch_par_slice(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    half_causal_estimator_batch_inner(data, sweep, kernel, true)
}

#[inline]
pub fn half_causal_estimator_batch_inner(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    if data.is_empty() {
        return Err(HalfCausalEstimatorError::EmptyInputData);
    }
    if first_finite(data) >= data.len() {
        return Err(HalfCausalEstimatorError::AllValuesNaN);
    }
    let combos = expand_grid_half_causal_estimator(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    let est_mu = make_uninit_matrix(rows, cols);
    let exp_mu = make_uninit_matrix(rows, cols);
    let mut est_guard = ManuallyDrop::new(est_mu);
    let mut exp_guard = ManuallyDrop::new(exp_mu);

    let estimate_values = unsafe {
        std::slice::from_raw_parts_mut(est_guard.as_mut_ptr() as *mut f64, est_guard.len())
    };
    let expected_value_values = unsafe {
        std::slice::from_raw_parts_mut(exp_guard.as_mut_ptr() as *mut f64, exp_guard.len())
    };

    estimate_values.fill(f64::NAN);
    expected_value_values.fill(f64::NAN);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        estimate_values
            .par_chunks_mut(cols)
            .zip(expected_value_values.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (estimate_row, expected_row))| {
                let params = resolve_params(
                    &combos[row],
                    combos[row]
                        .slots_per_day
                        .unwrap_or(sweep.slots_per_day.unwrap()),
                )
                .unwrap();
                let slots = PreparedSlots::Sequential {
                    slots_per_day: params.slots_per_day,
                };
                let _ = kernel;
                compute_row(data, &slots, params, estimate_row, expected_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, (estimate_row, expected_row)) in estimate_values
            .chunks_mut(cols)
            .zip(expected_value_values.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(
                &combos[row],
                combos[row]
                    .slots_per_day
                    .unwrap_or(sweep.slots_per_day.unwrap()),
            )?;
            let slots = PreparedSlots::Sequential {
                slots_per_day: params.slots_per_day,
            };
            let _ = kernel;
            compute_row(data, &slots, params, estimate_row, expected_row);
        }
    } else {
        for (row, (estimate_row, expected_row)) in estimate_values
            .chunks_mut(cols)
            .zip(expected_value_values.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(
                &combos[row],
                combos[row]
                    .slots_per_day
                    .unwrap_or(sweep.slots_per_day.unwrap()),
            )?;
            let slots = PreparedSlots::Sequential {
                slots_per_day: params.slots_per_day,
            };
            let _ = kernel;
            compute_row(data, &slots, params, estimate_row, expected_row);
        }
    }

    let estimate_values = unsafe {
        Vec::from_raw_parts(
            est_guard.as_mut_ptr() as *mut f64,
            est_guard.len(),
            est_guard.capacity(),
        )
    };
    let expected_value_values = unsafe {
        Vec::from_raw_parts(
            exp_guard.as_mut_ptr() as *mut f64,
            exp_guard.len(),
            exp_guard.capacity(),
        )
    };

    Ok(HalfCausalEstimatorBatchOutput {
        estimate_values,
        expected_value_values,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn half_causal_estimator_batch_inner_into(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
) -> Result<Vec<HalfCausalEstimatorParams>, HalfCausalEstimatorError> {
    if data.is_empty() {
        return Err(HalfCausalEstimatorError::EmptyInputData);
    }
    if first_finite(data) >= data.len() {
        return Err(HalfCausalEstimatorError::AllValuesNaN);
    }
    let combos = expand_grid_half_causal_estimator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(HalfCausalEstimatorError::OutputLengthMismatch {
            expected: usize::MAX,
            estimate_got: estimate_out.len(),
            expected_value_got: expected_value_out.len(),
        })?;
    if estimate_out.len() != total || expected_value_out.len() != total {
        return Err(HalfCausalEstimatorError::OutputLengthMismatch {
            expected: total,
            estimate_got: estimate_out.len(),
            expected_value_got: expected_value_out.len(),
        });
    }

    estimate_out.fill(f64::NAN);
    expected_value_out.fill(f64::NAN);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        estimate_out
            .par_chunks_mut(cols)
            .zip(expected_value_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (estimate_row, expected_row))| {
                let params = resolve_params(
                    &combos[row],
                    combos[row]
                        .slots_per_day
                        .unwrap_or(sweep.slots_per_day.unwrap()),
                )
                .unwrap();
                let slots = PreparedSlots::Sequential {
                    slots_per_day: params.slots_per_day,
                };
                let _ = kernel;
                compute_row(data, &slots, params, estimate_row, expected_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, (estimate_row, expected_row)) in estimate_out
            .chunks_mut(cols)
            .zip(expected_value_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(
                &combos[row],
                combos[row]
                    .slots_per_day
                    .unwrap_or(sweep.slots_per_day.unwrap()),
            )?;
            let slots = PreparedSlots::Sequential {
                slots_per_day: params.slots_per_day,
            };
            let _ = kernel;
            compute_row(data, &slots, params, estimate_row, expected_row);
        }
    } else {
        for (row, (estimate_row, expected_row)) in estimate_out
            .chunks_mut(cols)
            .zip(expected_value_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(
                &combos[row],
                combos[row]
                    .slots_per_day
                    .unwrap_or(sweep.slots_per_day.unwrap()),
            )?;
            let slots = PreparedSlots::Sequential {
                slots_per_day: params.slots_per_day,
            };
            let _ = kernel;
            compute_row(data, &slots, params, estimate_row, expected_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "half_causal_estimator")]
#[pyo3(signature = (
    data,
    slots_per_day,
    data_period=DEFAULT_DATA_PERIOD,
    filter_length=DEFAULT_FILTER_LENGTH,
    kernel_width=DEFAULT_KERNEL_WIDTH,
    kernel_type="epanechnikov",
    confidence_adjust="symmetric",
    maximum_confidence_adjust=DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
    enable_expected_value=DEFAULT_ENABLE_EXPECTED_VALUE,
    extra_smoothing=DEFAULT_EXTRA_SMOOTHING,
    kernel=None
))]
pub fn half_causal_estimator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    slots_per_day: usize,
    data_period: usize,
    filter_length: usize,
    kernel_width: f64,
    kernel_type: &str,
    confidence_adjust: &str,
    maximum_confidence_adjust: f64,
    enable_expected_value: bool,
    extra_smoothing: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let params = HalfCausalEstimatorParams {
        slots_per_day: Some(slots_per_day),
        data_period: Some(data_period),
        filter_length: Some(filter_length),
        kernel_width: Some(kernel_width),
        kernel_type: Some(parse_kernel_type_py(kernel_type)?),
        confidence_adjust: Some(parse_confidence_adjust_py(confidence_adjust)?),
        maximum_confidence_adjust: Some(maximum_confidence_adjust),
        enable_expected_value: Some(enable_expected_value),
        extra_smoothing: Some(extra_smoothing),
    };
    let input = HalfCausalEstimatorInput::from_slice(data, params);
    let out = py
        .allow_threads(|| half_causal_estimator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("estimate", out.estimate.into_pyarray(py))?;
    dict.set_item("expected_value", out.expected_value.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "HalfCausalEstimatorStream")]
pub struct HalfCausalEstimatorStreamPy {
    inner: HalfCausalEstimatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HalfCausalEstimatorStreamPy {
    #[new]
    #[pyo3(signature = (
        slots_per_day,
        data_period=DEFAULT_DATA_PERIOD,
        filter_length=DEFAULT_FILTER_LENGTH,
        kernel_width=DEFAULT_KERNEL_WIDTH,
        kernel_type="epanechnikov",
        confidence_adjust="symmetric",
        maximum_confidence_adjust=DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
        enable_expected_value=DEFAULT_ENABLE_EXPECTED_VALUE,
        extra_smoothing=DEFAULT_EXTRA_SMOOTHING
    ))]
    fn new(
        slots_per_day: usize,
        data_period: usize,
        filter_length: usize,
        kernel_width: f64,
        kernel_type: &str,
        confidence_adjust: &str,
        maximum_confidence_adjust: f64,
        enable_expected_value: bool,
        extra_smoothing: usize,
    ) -> PyResult<Self> {
        let inner = HalfCausalEstimatorStream::try_new(HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            data_period: Some(data_period),
            filter_length: Some(filter_length),
            kernel_width: Some(kernel_width),
            kernel_type: Some(parse_kernel_type_py(kernel_type)?),
            confidence_adjust: Some(parse_confidence_adjust_py(confidence_adjust)?),
            maximum_confidence_adjust: Some(maximum_confidence_adjust),
            enable_expected_value: Some(enable_expected_value),
            extra_smoothing: Some(extra_smoothing),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> (Option<f64>, Option<f64>) {
        self.inner.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "half_causal_estimator_batch")]
#[pyo3(signature = (
    data,
    slots_per_day,
    data_period_range=(DEFAULT_DATA_PERIOD, DEFAULT_DATA_PERIOD, 0),
    filter_length_range=(DEFAULT_FILTER_LENGTH, DEFAULT_FILTER_LENGTH, 0),
    kernel_width_range=(DEFAULT_KERNEL_WIDTH, DEFAULT_KERNEL_WIDTH, 0.0),
    maximum_confidence_adjust_range=(DEFAULT_MAXIMUM_CONFIDENCE_ADJUST, DEFAULT_MAXIMUM_CONFIDENCE_ADJUST, 0.0),
    extra_smoothing_range=(DEFAULT_EXTRA_SMOOTHING, DEFAULT_EXTRA_SMOOTHING, 0),
    kernel_type="epanechnikov",
    confidence_adjust="symmetric",
    enable_expected_value=DEFAULT_ENABLE_EXPECTED_VALUE,
    kernel=None
))]
pub fn half_causal_estimator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    slots_per_day: usize,
    data_period_range: (usize, usize, usize),
    filter_length_range: (usize, usize, usize),
    kernel_width_range: (f64, f64, f64),
    maximum_confidence_adjust_range: (f64, f64, f64),
    extra_smoothing_range: (usize, usize, usize),
    kernel_type: &str,
    confidence_adjust: &str,
    enable_expected_value: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = HalfCausalEstimatorBatchRange {
        slots_per_day: Some(slots_per_day),
        data_period: data_period_range,
        filter_length: filter_length_range,
        kernel_width: kernel_width_range,
        maximum_confidence_adjust: maximum_confidence_adjust_range,
        extra_smoothing: extra_smoothing_range,
        kernel_type: parse_kernel_type_py(kernel_type)?,
        confidence_adjust: parse_confidence_adjust_py(confidence_adjust)?,
        enable_expected_value,
    };
    let combos = expand_grid_half_causal_estimator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let estimate_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let expected_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let estimate_slice = unsafe { estimate_arr.as_slice_mut()? };
    let expected_slice = unsafe { expected_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            half_causal_estimator_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                true,
                estimate_slice,
                expected_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("estimate", estimate_arr.reshape((rows, cols))?)?;
    dict.set_item("expected_value", expected_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "data_periods",
        combos
            .iter()
            .map(|combo| combo.data_period.unwrap_or(DEFAULT_DATA_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "filter_lengths",
        combos
            .iter()
            .map(|combo| combo.filter_length.unwrap_or(DEFAULT_FILTER_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "kernel_widths",
        combos
            .iter()
            .map(|combo| combo.kernel_width.unwrap_or(DEFAULT_KERNEL_WIDTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "maximum_confidence_adjusts",
        combos
            .iter()
            .map(|combo| {
                combo
                    .maximum_confidence_adjust
                    .unwrap_or(DEFAULT_MAXIMUM_CONFIDENCE_ADJUST)
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "extra_smoothings",
        combos
            .iter()
            .map(|combo| combo.extra_smoothing.unwrap_or(DEFAULT_EXTRA_SMOOTHING) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
fn parse_kernel_type_py(value: &str) -> PyResult<HalfCausalEstimatorKernelType> {
    HalfCausalEstimatorKernelType::from_str(value)
        .ok_or_else(|| PyValueError::new_err(format!("Invalid kernel_type: {value}")))
}

#[cfg(feature = "python")]
fn parse_confidence_adjust_py(value: &str) -> PyResult<HalfCausalEstimatorConfidenceAdjust> {
    HalfCausalEstimatorConfidenceAdjust::from_str(value)
        .ok_or_else(|| PyValueError::new_err(format!("Invalid confidence_adjust: {value}")))
}

#[cfg(feature = "python")]
pub fn register_half_causal_estimator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(half_causal_estimator_py, module)?)?;
    module.add_function(wrap_pyfunction!(half_causal_estimator_batch_py, module)?)?;
    module.add_class::<HalfCausalEstimatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HalfCausalEstimatorJsConfig {
    pub slots_per_day: usize,
    pub data_period: Option<usize>,
    pub filter_length: Option<usize>,
    pub kernel_width: Option<f64>,
    pub kernel_type: Option<HalfCausalEstimatorKernelType>,
    pub confidence_adjust: Option<HalfCausalEstimatorConfidenceAdjust>,
    pub maximum_confidence_adjust: Option<f64>,
    pub enable_expected_value: Option<bool>,
    pub extra_smoothing: Option<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl From<HalfCausalEstimatorJsConfig> for HalfCausalEstimatorParams {
    fn from(value: HalfCausalEstimatorJsConfig) -> Self {
        Self {
            slots_per_day: Some(value.slots_per_day),
            data_period: value.data_period,
            filter_length: value.filter_length,
            kernel_width: value.kernel_width,
            kernel_type: value.kernel_type,
            confidence_adjust: value.confidence_adjust,
            maximum_confidence_adjust: value.maximum_confidence_adjust,
            enable_expected_value: value.enable_expected_value,
            extra_smoothing: value.extra_smoothing,
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HalfCausalEstimatorJsOutput {
    pub estimate: Vec<f64>,
    pub expected_value: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "half_causal_estimator_js")]
pub fn half_causal_estimator_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: HalfCausalEstimatorJsConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let input = HalfCausalEstimatorInput::from_slice(data, config.into());
    let out = half_causal_estimator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&HalfCausalEstimatorJsOutput {
        estimate: out.estimate,
        expected_value: out.expected_value,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn half_causal_estimator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn half_causal_estimator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn half_causal_estimator_into(
    in_ptr: *const f64,
    estimate_out_ptr: *mut f64,
    expected_value_out_ptr: *mut f64,
    len: usize,
    config: JsValue,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || estimate_out_ptr.is_null() || expected_value_out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let config: HalfCausalEstimatorJsConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = HalfCausalEstimatorInput::from_slice(data, config.into());
        let estimate_out = std::slice::from_raw_parts_mut(estimate_out_ptr, len);
        let expected_value_out = std::slice::from_raw_parts_mut(expected_value_out_ptr, len);
        half_causal_estimator_into_slices(estimate_out, expected_value_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HalfCausalEstimatorBatchJsConfig {
    pub slots_per_day: usize,
    pub data_period_range: Option<(usize, usize, usize)>,
    pub filter_length_range: Option<(usize, usize, usize)>,
    pub kernel_width_range: Option<(f64, f64, f64)>,
    pub maximum_confidence_adjust_range: Option<(f64, f64, f64)>,
    pub extra_smoothing_range: Option<(usize, usize, usize)>,
    pub kernel_type: Option<HalfCausalEstimatorKernelType>,
    pub confidence_adjust: Option<HalfCausalEstimatorConfidenceAdjust>,
    pub enable_expected_value: Option<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HalfCausalEstimatorBatchJsOutput {
    pub estimate: Vec<f64>,
    pub expected_value: Vec<f64>,
    pub combos: Vec<HalfCausalEstimatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "half_causal_estimator_batch_js")]
pub fn half_causal_estimator_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: HalfCausalEstimatorBatchJsConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let sweep = HalfCausalEstimatorBatchRange {
        slots_per_day: Some(config.slots_per_day),
        data_period: config.data_period_range.unwrap_or((
            DEFAULT_DATA_PERIOD,
            DEFAULT_DATA_PERIOD,
            0,
        )),
        filter_length: config.filter_length_range.unwrap_or((
            DEFAULT_FILTER_LENGTH,
            DEFAULT_FILTER_LENGTH,
            0,
        )),
        kernel_width: config.kernel_width_range.unwrap_or((
            DEFAULT_KERNEL_WIDTH,
            DEFAULT_KERNEL_WIDTH,
            0.0,
        )),
        maximum_confidence_adjust: config.maximum_confidence_adjust_range.unwrap_or((
            DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
            DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
            0.0,
        )),
        extra_smoothing: config.extra_smoothing_range.unwrap_or((
            DEFAULT_EXTRA_SMOOTHING,
            DEFAULT_EXTRA_SMOOTHING,
            0,
        )),
        kernel_type: config.kernel_type.unwrap_or_default(),
        confidence_adjust: config.confidence_adjust.unwrap_or_default(),
        enable_expected_value: config
            .enable_expected_value
            .unwrap_or(DEFAULT_ENABLE_EXPECTED_VALUE),
    };
    let out = half_causal_estimator_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&HalfCausalEstimatorBatchJsOutput {
        estimate: out.estimate_values,
        expected_value: out.expected_value_values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn half_causal_estimator_batch_into(
    in_ptr: *const f64,
    estimate_out_ptr: *mut f64,
    expected_value_out_ptr: *mut f64,
    len: usize,
    config: JsValue,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || estimate_out_ptr.is_null() || expected_value_out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let config: HalfCausalEstimatorBatchJsConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let sweep = HalfCausalEstimatorBatchRange {
        slots_per_day: Some(config.slots_per_day),
        data_period: config.data_period_range.unwrap_or((
            DEFAULT_DATA_PERIOD,
            DEFAULT_DATA_PERIOD,
            0,
        )),
        filter_length: config.filter_length_range.unwrap_or((
            DEFAULT_FILTER_LENGTH,
            DEFAULT_FILTER_LENGTH,
            0,
        )),
        kernel_width: config.kernel_width_range.unwrap_or((
            DEFAULT_KERNEL_WIDTH,
            DEFAULT_KERNEL_WIDTH,
            0.0,
        )),
        maximum_confidence_adjust: config.maximum_confidence_adjust_range.unwrap_or((
            DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
            DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
            0.0,
        )),
        extra_smoothing: config.extra_smoothing_range.unwrap_or((
            DEFAULT_EXTRA_SMOOTHING,
            DEFAULT_EXTRA_SMOOTHING,
            0,
        )),
        kernel_type: config.kernel_type.unwrap_or_default(),
        confidence_adjust: config.confidence_adjust.unwrap_or_default(),
        enable_expected_value: config
            .enable_expected_value
            .unwrap_or(DEFAULT_ENABLE_EXPECTED_VALUE),
    };
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let combos = expand_grid_half_causal_estimator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let estimate_out = std::slice::from_raw_parts_mut(estimate_out_ptr, total);
        let expected_value_out = std::slice::from_raw_parts_mut(expected_value_out_ptr, total);
        let rows = half_causal_estimator_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            estimate_out,
            expected_value_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn half_causal_estimator_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = half_causal_estimator_js(data, config)?;
    crate::write_wasm_object_f64_outputs("half_causal_estimator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn half_causal_estimator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = half_causal_estimator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "half_causal_estimator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::Candles;

    fn sample_source(length: usize, slots_per_day: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(length);
        for i in 0..length {
            let slot = (i % slots_per_day) as f64;
            let day = (i / slots_per_day) as f64;
            out.push(
                1000.0
                    + day * 5.0
                    + (slot * 0.11).sin() * 30.0
                    + (slot * 0.03).cos() * 12.0
                    + (slot / slots_per_day as f64) * 25.0,
            );
        }
        out
    }

    fn sample_candles(days: usize, minutes_per_bar: usize) -> Candles {
        let slots_per_day = 1440 / minutes_per_bar;
        let len = days * slots_per_day;
        let mut timestamp = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        let start_ms = 1_700_000_000_000i64;
        let interval_ms = minutes_per_bar as i64 * 60_000;
        for i in 0..len {
            let x = i as f64;
            timestamp.push(start_ms + i as i64 * interval_ms);
            let c = 100.0 + (x * 0.05).sin() * 2.0 + x * 0.001;
            open.push(c - 0.2);
            high.push(c + 0.6);
            low.push(c - 0.7);
            close.push(c);
            volume.push(1000.0 + (x * 0.07).cos() * 80.0 + x * 0.5);
        }
        Candles::new(timestamp, open, high, low, close, volume)
    }

    #[test]
    fn half_causal_estimator_output_contract() {
        let slots_per_day = 60;
        let data = sample_source(slots_per_day * 4, slots_per_day);
        let input = HalfCausalEstimatorInput::from_slice(
            &data,
            HalfCausalEstimatorParams {
                slots_per_day: Some(slots_per_day),
                ..HalfCausalEstimatorParams::default()
            },
        );
        let out = half_causal_estimator(&input).unwrap();
        assert_eq!(out.estimate.len(), data.len());
        assert_eq!(out.expected_value.len(), data.len());
        assert!(out.estimate.iter().any(|value| value.is_finite()));
        assert!(out.expected_value.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn half_causal_estimator_stream_matches_batch() {
        let slots_per_day = 48;
        let data = sample_source(slots_per_day * 5, slots_per_day);
        let params = HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            enable_expected_value: Some(true),
            extra_smoothing: Some(2),
            ..HalfCausalEstimatorParams::default()
        };
        let input = HalfCausalEstimatorInput::from_slice(&data, params.clone());
        let out = half_causal_estimator(&input).unwrap();
        let mut stream = HalfCausalEstimatorStream::try_new(params).unwrap();
        let mut est = Vec::with_capacity(data.len());
        let mut exp = Vec::with_capacity(data.len());
        for value in data {
            let (estimate, expected_value) = stream.update(value);
            est.push(estimate.unwrap_or(f64::NAN));
            exp.push(expected_value.unwrap_or(f64::NAN));
        }
        for i in 0..est.len() {
            if out.estimate[i].is_nan() {
                assert!(est[i].is_nan());
            } else {
                assert!((est[i] - out.estimate[i]).abs() < 1e-12);
            }
            if out.expected_value[i].is_nan() {
                assert!(exp[i].is_nan());
            } else {
                assert!((exp[i] - out.expected_value[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn half_causal_estimator_batch_single_matches_direct() {
        let slots_per_day = 60;
        let data = sample_source(slots_per_day * 4, slots_per_day);
        let params = HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            enable_expected_value: Some(true),
            ..HalfCausalEstimatorParams::default()
        };
        let input = HalfCausalEstimatorInput::from_slice(&data, params.clone());
        let direct = half_causal_estimator(&input).unwrap();
        let batch = half_causal_estimator_batch_with_kernel(
            &data,
            &HalfCausalEstimatorBatchRange {
                slots_per_day: Some(slots_per_day),
                enable_expected_value: true,
                ..HalfCausalEstimatorBatchRange::default()
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for (&lhs, &rhs) in batch
            .estimate_for(0)
            .unwrap()
            .iter()
            .zip(direct.estimate.iter())
        {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert!((lhs - rhs).abs() < 1e-12);
            }
        }
        for (&lhs, &rhs) in batch
            .expected_value_for(0)
            .unwrap()
            .iter()
            .zip(direct.expected_value.iter())
        {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert!((lhs - rhs).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn half_causal_estimator_candle_test_source_runs() {
        let candles = sample_candles(4, 30);
        let input = HalfCausalEstimatorInput::from_candles(
            &candles,
            "test",
            HalfCausalEstimatorParams {
                enable_expected_value: Some(true),
                ..HalfCausalEstimatorParams::default()
            },
        );
        let out = half_causal_estimator(&input).unwrap();
        assert_eq!(out.estimate.len(), candles.close.len());
        assert!(out.estimate.iter().any(|value| value.is_finite()));
        assert!(out.expected_value.iter().any(|value| value.is_finite()));
    }

    #[test]
    fn half_causal_estimator_rejects_invalid_params() {
        let data = sample_source(128, 32);
        let input = HalfCausalEstimatorInput::from_slice(
            &data,
            HalfCausalEstimatorParams {
                slots_per_day: Some(1),
                ..HalfCausalEstimatorParams::default()
            },
        );
        let err = half_causal_estimator(&input).unwrap_err();
        assert!(matches!(
            err,
            HalfCausalEstimatorError::InvalidSlotsPerDay { .. }
        ));
    }
}
