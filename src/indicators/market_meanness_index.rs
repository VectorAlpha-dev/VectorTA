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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MarketMeannessIndexData<'a> {
    Candles { candles: &'a Candles },
    Slices { open: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct MarketMeannessIndexOutput {
    pub mmi: Vec<f64>,
    pub mmi_smoothed: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MarketMeannessIndexParams {
    pub length: Option<usize>,
    pub source_mode: Option<String>,
}

impl Default for MarketMeannessIndexParams {
    fn default() -> Self {
        Self {
            length: Some(300),
            source_mode: Some("Price".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketMeannessIndexInput<'a> {
    pub data: MarketMeannessIndexData<'a>,
    pub params: MarketMeannessIndexParams,
}

impl<'a> MarketMeannessIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: MarketMeannessIndexParams) -> Self {
        Self {
            data: MarketMeannessIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        close: &'a [f64],
        params: MarketMeannessIndexParams,
    ) -> Self {
        Self {
            data: MarketMeannessIndexData::Slices { open, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, MarketMeannessIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(300)
    }

    #[inline]
    pub fn source_mode_str(&self) -> &str {
        self.params.source_mode.as_deref().unwrap_or("Price")
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            MarketMeannessIndexData::Candles { candles } => {
                (candles.open.as_slice(), candles.close.as_slice())
            }
            MarketMeannessIndexData::Slices { open, close } => (*open, *close),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MarketMeannessSourceMode {
    Price,
    Change,
}

impl MarketMeannessSourceMode {
    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Price => "Price",
            Self::Change => "Change",
        }
    }
}

#[inline(always)]
fn parse_source_mode(value: &str) -> Result<MarketMeannessSourceMode, MarketMeannessIndexError> {
    match value.trim().to_ascii_uppercase().as_str() {
        "PRICE" => Ok(MarketMeannessSourceMode::Price),
        "CHANGE" => Ok(MarketMeannessSourceMode::Change),
        _ => Err(MarketMeannessIndexError::InvalidSourceMode {
            source_mode: value.to_string(),
        }),
    }
}

#[inline(always)]
fn normalize_source_mode(value: &str) -> Result<String, MarketMeannessIndexError> {
    Ok(parse_source_mode(value)?.as_str().to_string())
}

#[derive(Copy, Clone, Debug)]
pub struct MarketMeannessIndexBuilder {
    length: Option<usize>,
    source_mode: Option<MarketMeannessSourceMode>,
    kernel: Kernel,
}

impl Default for MarketMeannessIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            source_mode: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MarketMeannessIndexBuilder {
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
    pub fn source_mode(mut self, value: &str) -> Result<Self, MarketMeannessIndexError> {
        self.source_mode = Some(parse_source_mode(value)?);
        Ok(self)
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
    ) -> Result<MarketMeannessIndexOutput, MarketMeannessIndexError> {
        let input = MarketMeannessIndexInput::from_candles(
            candles,
            MarketMeannessIndexParams {
                length: self.length,
                source_mode: Some(
                    self.source_mode
                        .unwrap_or(MarketMeannessSourceMode::Price)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        market_meanness_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<MarketMeannessIndexOutput, MarketMeannessIndexError> {
        let input = MarketMeannessIndexInput::from_slices(
            open,
            close,
            MarketMeannessIndexParams {
                length: self.length,
                source_mode: Some(
                    self.source_mode
                        .unwrap_or(MarketMeannessSourceMode::Price)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        market_meanness_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<MarketMeannessIndexStream, MarketMeannessIndexError> {
        MarketMeannessIndexStream::try_new(MarketMeannessIndexParams {
            length: self.length,
            source_mode: Some(
                self.source_mode
                    .unwrap_or(MarketMeannessSourceMode::Price)
                    .as_str()
                    .to_string(),
            ),
        })
    }
}

#[derive(Debug, Error)]
pub enum MarketMeannessIndexError {
    #[error("market_meanness_index: Empty input data.")]
    EmptyInputData,
    #[error("market_meanness_index: Data length mismatch across open and close.")]
    DataLengthMismatch,
    #[error("market_meanness_index: All values are NaN.")]
    AllValuesNaN,
    #[error("market_meanness_index: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("market_meanness_index: Invalid source mode: {source_mode}")]
    InvalidSourceMode { source_mode: String },
    #[error("market_meanness_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("market_meanness_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("market_meanness_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("market_meanness_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn minimum_length() -> usize {
    6
}

#[inline(always)]
fn is_valid_bar(open: f64, close: f64, mode: MarketMeannessSourceMode) -> bool {
    match mode {
        MarketMeannessSourceMode::Price => close.is_finite(),
        MarketMeannessSourceMode::Change => open.is_finite() && close.is_finite(),
    }
}

#[inline(always)]
fn source_value(open: f64, close: f64, mode: MarketMeannessSourceMode) -> f64 {
    match mode {
        MarketMeannessSourceMode::Price => close,
        MarketMeannessSourceMode::Change => close - open,
    }
}

#[inline(always)]
fn first_valid_bar(open: &[f64], close: &[f64], mode: MarketMeannessSourceMode) -> Option<usize> {
    (0..close.len()).find(|&i| is_valid_bar(open[i], close[i], mode))
}

#[inline(always)]
fn count_valid_from(
    open: &[f64],
    close: &[f64],
    start: usize,
    mode: MarketMeannessSourceMode,
) -> usize {
    (start..close.len())
        .filter(|&i| is_valid_bar(open[i], close[i], mode))
        .count()
}

#[inline(always)]
fn first_and_valid_from(
    open: &[f64],
    close: &[f64],
    mode: MarketMeannessSourceMode,
) -> Option<(usize, usize)> {
    let mut first = None;
    let mut valid = 0usize;
    for i in 0..close.len() {
        if is_valid_bar(open[i], close[i], mode) {
            if first.is_none() {
                first = Some(i);
            }
            valid += 1;
        }
    }
    first.map(|idx| (idx, valid))
}

#[inline(always)]
fn ordered_window_from_ring(dst: &mut [f64], ring: &[f64], len: usize, head: usize) {
    if head == 0 {
        dst[..len].copy_from_slice(&ring[..len]);
        return;
    }
    let tail = len - head;
    dst[..tail].copy_from_slice(&ring[head..len]);
    dst[tail..len].copy_from_slice(&ring[..head]);
}

#[inline(always)]
fn median_from(buf: &mut [f64]) -> f64 {
    let len = buf.len();
    let mid = len / 2;
    if (len & 1) == 1 {
        *buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
    } else {
        let (left, upper, _) = buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let upper = *upper;
        let lower = left
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .copied()
            .unwrap_or(upper);
        (lower + upper) * 0.5
    }
}

#[derive(Clone, Debug)]
struct RollingSmaState {
    period: usize,
    count: usize,
    head: usize,
    sum: f64,
    buffer: Vec<f64>,
}

impl RollingSmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            head: 0,
            sum: 0.0,
            buffer: vec![0.0; period.max(1)],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.head = 0;
        self.sum = 0.0;
        self.buffer.fill(0.0);
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count < self.period {
            self.buffer[self.count] = value;
            self.sum += value;
            self.count += 1;
            if self.count == self.period {
                return Some(self.sum / self.period as f64);
            }
            return None;
        }

        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.sum += value - old;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        Some(self.sum / self.period as f64)
    }
}

#[derive(Clone, Debug)]
pub struct MarketMeannessIndexStream {
    length: usize,
    mode: MarketMeannessSourceMode,
    source_count: usize,
    source_head: usize,
    source_ring: Vec<f64>,
    window_buf: Vec<f64>,
    median_buf: Vec<f64>,
    smoothing: RollingSmaState,
}

impl MarketMeannessIndexStream {
    #[inline]
    fn from_parts(length: usize, mode: MarketMeannessSourceMode) -> Self {
        Self {
            length,
            mode,
            source_count: 0,
            source_head: 0,
            source_ring: vec![0.0; length.max(1)],
            window_buf: vec![0.0; length.max(1)],
            median_buf: vec![0.0; length.max(1)],
            smoothing: RollingSmaState::new(length),
        }
    }

    #[inline]
    pub fn try_new(params: MarketMeannessIndexParams) -> Result<Self, MarketMeannessIndexError> {
        let length = params.length.unwrap_or(300);
        if length < minimum_length() {
            return Err(MarketMeannessIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        let mode = parse_source_mode(params.source_mode.as_deref().unwrap_or("Price"))?;
        Ok(Self::from_parts(length, mode))
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.source_count = 0;
        self.source_head = 0;
        self.source_ring.fill(0.0);
        self.window_buf.fill(0.0);
        self.median_buf.fill(0.0);
        self.smoothing.reset();
    }

    #[inline]
    pub fn update(&mut self, open: f64, close: f64) -> Option<(f64, f64)> {
        if !is_valid_bar(open, close, self.mode) {
            return None;
        }

        let value = source_value(open, close, self.mode);
        if self.source_count < self.length {
            self.source_ring[self.source_count] = value;
            self.source_count += 1;
            if self.source_count < self.length {
                return None;
            }
        } else {
            self.source_ring[self.source_head] = value;
            self.source_head += 1;
            if self.source_head == self.length {
                self.source_head = 0;
            }
        }

        ordered_window_from_ring(
            &mut self.window_buf,
            &self.source_ring,
            self.length,
            self.source_head,
        );
        self.median_buf[..self.length].copy_from_slice(&self.window_buf[..self.length]);
        let median = median_from(&mut self.median_buf[..self.length]);
        let mut count = 0usize;
        for i in 1..self.length {
            let prev = self.window_buf[i - 1];
            let curr = self.window_buf[i];
            if (curr > median && curr > prev) || (curr < median && curr < prev) {
                count += 1;
            }
        }
        let mmi = count as f64 * (100.0 / (self.length - 1) as f64);
        let mmi_smoothed = self.smoothing.update(mmi).unwrap_or(f64::NAN);
        Some((mmi, mmi_smoothed))
    }

    #[inline(always)]
    pub fn update_reset_on_nan(&mut self, open: f64, close: f64) -> Option<(f64, f64)> {
        if !is_valid_bar(open, close, self.mode) {
            self.reset();
            return None;
        }
        self.update(open, close)
    }
}

#[inline(always)]
fn mmi_warmup(length: usize, first: usize) -> usize {
    first + length - 1
}

#[inline(always)]
fn mmi_smoothed_warmup(length: usize, first: usize) -> usize {
    mmi_warmup(length, first) + length - 1
}

#[inline(always)]
fn market_meanness_index_prepare<'a>(
    input: &'a MarketMeannessIndexInput,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        usize,
        MarketMeannessSourceMode,
        usize,
        usize,
    ),
    MarketMeannessIndexError,
> {
    let (open, close) = input.as_refs();
    let data_len = close.len();
    if data_len == 0 {
        return Err(MarketMeannessIndexError::EmptyInputData);
    }
    if open.len() != data_len {
        return Err(MarketMeannessIndexError::DataLengthMismatch);
    }

    let length = input.get_length();
    if length < minimum_length() || length > data_len {
        return Err(MarketMeannessIndexError::InvalidLength { length, data_len });
    }

    let mode = parse_source_mode(input.source_mode_str())?;
    let (first, total_valid) =
        first_and_valid_from(open, close, mode).ok_or(MarketMeannessIndexError::AllValuesNaN)?;
    let valid = total_valid;
    if valid < length {
        return Err(MarketMeannessIndexError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    Ok((open, close, length, mode, first, valid))
}

#[inline(always)]
fn market_meanness_index_compute_into(
    open: &[f64],
    close: &[f64],
    length: usize,
    mode: MarketMeannessSourceMode,
    _kernel: Kernel,
    out_mmi: &mut [f64],
    out_mmi_smoothed: &mut [f64],
) {
    let mut stream = MarketMeannessIndexStream::from_parts(length, mode);
    for i in 0..close.len() {
        if let Some((mmi, mmi_smoothed)) = stream.update_reset_on_nan(open[i], close[i]) {
            out_mmi[i] = mmi;
            out_mmi_smoothed[i] = mmi_smoothed;
        }
    }
}

#[inline(always)]
fn sorted_insert(values: &mut Vec<f64>, value: f64) {
    let pos = values
        .binary_search_by(|probe| probe.total_cmp(&value))
        .unwrap_or_else(|pos| pos);
    values.insert(pos, value);
}

#[inline(always)]
fn sorted_remove(values: &mut Vec<f64>, value: f64) {
    if let Ok(pos) = values.binary_search_by(|probe| probe.total_cmp(&value)) {
        values.remove(pos);
    }
}

#[inline(always)]
fn count_meanness_from_ring(ring: &[f64], length: usize, head: usize, median: f64) -> usize {
    let mut count = 0usize;
    let mut prev = ring[head];

    let mut i = head + 1;
    while i < length {
        let curr = ring[i];
        if (curr > median && curr > prev) || (curr < median && curr < prev) {
            count += 1;
        }
        prev = curr;
        i += 1;
    }

    i = 0;
    while i < head {
        let curr = ring[i];
        if (curr > median && curr > prev) || (curr < median && curr < prev) {
            count += 1;
        }
        prev = curr;
        i += 1;
    }

    count
}

#[inline(always)]
fn market_meanness_index_compute_clean_sorted(
    open: &[f64],
    close: &[f64],
    length: usize,
    mode: MarketMeannessSourceMode,
    first: usize,
    out_mmi: &mut [f64],
    out_mmi_smoothed: &mut [f64],
) {
    let mut ring = vec![0.0; length];
    let mut sorted = Vec::with_capacity(length);
    let mut smoothing = RollingSmaState::new(length);
    let mut count = 0usize;
    let mut head = 0usize;
    let scale = 100.0 / (length - 1) as f64;

    let mut i = first;
    while i < close.len() {
        let value = source_value(open[i], close[i], mode);
        if count < length {
            ring[count] = value;
            sorted_insert(&mut sorted, value);
            count += 1;
            if count < length {
                i += 1;
                continue;
            }
        } else {
            let old = ring[head];
            sorted_remove(&mut sorted, old);
            ring[head] = value;
            sorted_insert(&mut sorted, value);
            head += 1;
            if head == length {
                head = 0;
            }
        }

        let mid = length / 2;
        let median = if (length & 1) == 1 {
            sorted[mid]
        } else {
            (sorted[mid - 1] + sorted[mid]) * 0.5
        };
        let mmi = count_meanness_from_ring(&ring, length, head, median) as f64 * scale;
        out_mmi[i] = mmi;
        out_mmi_smoothed[i] = smoothing.update(mmi).unwrap_or(f64::NAN);
        i += 1;
    }
}

#[inline]
pub fn market_meanness_index(
    input: &MarketMeannessIndexInput,
) -> Result<MarketMeannessIndexOutput, MarketMeannessIndexError> {
    market_meanness_index_with_kernel(input, Kernel::Auto)
}

pub fn market_meanness_index_with_kernel(
    input: &MarketMeannessIndexInput,
    kernel: Kernel,
) -> Result<MarketMeannessIndexOutput, MarketMeannessIndexError> {
    let (open, close, length, mode, first, valid) = market_meanness_index_prepare(input)?;
    let mut mmi = alloc_with_nan_prefix(close.len(), mmi_warmup(length, first).min(close.len()));
    let mut mmi_smoothed = alloc_with_nan_prefix(
        close.len(),
        mmi_smoothed_warmup(length, first).min(close.len()),
    );
    if valid == close.len() - first {
        market_meanness_index_compute_clean_sorted(
            open,
            close,
            length,
            mode,
            first,
            &mut mmi,
            &mut mmi_smoothed,
        );
    } else {
        market_meanness_index_compute_into(
            open,
            close,
            length,
            mode,
            kernel,
            &mut mmi,
            &mut mmi_smoothed,
        );
    }
    Ok(MarketMeannessIndexOutput { mmi, mmi_smoothed })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn market_meanness_index_into(
    input: &MarketMeannessIndexInput,
    out_mmi: &mut [f64],
    out_mmi_smoothed: &mut [f64],
) -> Result<(), MarketMeannessIndexError> {
    market_meanness_index_into_slice(out_mmi, out_mmi_smoothed, input, Kernel::Auto)
}

pub fn market_meanness_index_into_slice(
    out_mmi: &mut [f64],
    out_mmi_smoothed: &mut [f64],
    input: &MarketMeannessIndexInput,
    kernel: Kernel,
) -> Result<(), MarketMeannessIndexError> {
    let (open, close, length, mode, first, valid) = market_meanness_index_prepare(input)?;
    if out_mmi.len() != close.len() || out_mmi_smoothed.len() != close.len() {
        return Err(MarketMeannessIndexError::OutputLengthMismatch {
            expected: close.len(),
            got: out_mmi.len().max(out_mmi_smoothed.len()),
        });
    }

    out_mmi.fill(f64::NAN);
    out_mmi_smoothed.fill(f64::NAN);
    if valid == close.len() - first {
        market_meanness_index_compute_clean_sorted(
            open,
            close,
            length,
            mode,
            first,
            out_mmi,
            out_mmi_smoothed,
        );
    } else {
        market_meanness_index_compute_into(
            open,
            close,
            length,
            mode,
            kernel,
            out_mmi,
            out_mmi_smoothed,
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct MarketMeannessIndexBatchRange {
    pub length: (usize, usize, usize),
    pub source_mode: Option<String>,
}

impl Default for MarketMeannessIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (300, 300, 0),
            source_mode: Some("Price".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MarketMeannessIndexBatchBuilder {
    range: MarketMeannessIndexBatchRange,
    kernel: Kernel,
}

impl MarketMeannessIndexBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

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
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline]
    pub fn source_mode(mut self, value: &str) -> Result<Self, MarketMeannessIndexError> {
        self.range.source_mode = Some(normalize_source_mode(value)?);
        Ok(self)
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<MarketMeannessIndexBatchOutput, MarketMeannessIndexError> {
        market_meanness_index_batch_with_kernel(open, close, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<MarketMeannessIndexBatchOutput, MarketMeannessIndexError> {
        self.apply_slices(candles.open.as_slice(), candles.close.as_slice())
    }
}

#[derive(Clone, Debug)]
pub struct MarketMeannessIndexBatchOutput {
    pub mmi: Vec<f64>,
    pub mmi_smoothed: Vec<f64>,
    pub combos: Vec<MarketMeannessIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MarketMeannessIndexBatchOutput {
    pub fn row_for_params(&self, params: &MarketMeannessIndexParams) -> Option<usize> {
        let target_length = params.length.unwrap_or(300);
        let target_source_mode =
            normalize_source_mode(params.source_mode.as_deref().unwrap_or("Price")).ok()?;
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(300) == target_length
                && combo.source_mode.as_deref().unwrap_or("Price") == target_source_mode
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, MarketMeannessIndexError> {
    let (start, end, step) = range;
    if start < minimum_length() || end < minimum_length() {
        return Err(MarketMeannessIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
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
        return Err(MarketMeannessIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_market_meanness_index(
    sweep: &MarketMeannessIndexBatchRange,
) -> Result<Vec<MarketMeannessIndexParams>, MarketMeannessIndexError> {
    let lengths = axis_usize(sweep.length)?;
    let source_mode = normalize_source_mode(sweep.source_mode.as_deref().unwrap_or("Price"))?;
    Ok(lengths
        .into_iter()
        .map(|length| MarketMeannessIndexParams {
            length: Some(length),
            source_mode: Some(source_mode.clone()),
        })
        .collect())
}

pub fn market_meanness_index_batch_with_kernel(
    open: &[f64],
    close: &[f64],
    sweep: &MarketMeannessIndexBatchRange,
    kernel: Kernel,
) -> Result<MarketMeannessIndexBatchOutput, MarketMeannessIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(MarketMeannessIndexError::InvalidKernelForBatch(other)),
    };
    market_meanness_index_batch_impl(open, close, sweep, batch_kernel.to_non_batch(), true)
}

pub fn market_meanness_index_batch_slice(
    open: &[f64],
    close: &[f64],
    sweep: &MarketMeannessIndexBatchRange,
) -> Result<MarketMeannessIndexBatchOutput, MarketMeannessIndexError> {
    market_meanness_index_batch_impl(open, close, sweep, Kernel::Scalar, false)
}

pub fn market_meanness_index_batch_par_slice(
    open: &[f64],
    close: &[f64],
    sweep: &MarketMeannessIndexBatchRange,
) -> Result<MarketMeannessIndexBatchOutput, MarketMeannessIndexError> {
    market_meanness_index_batch_impl(open, close, sweep, Kernel::Scalar, true)
}

fn market_meanness_index_batch_impl(
    open: &[f64],
    close: &[f64],
    sweep: &MarketMeannessIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<MarketMeannessIndexBatchOutput, MarketMeannessIndexError> {
    let combos = expand_grid_market_meanness_index(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(MarketMeannessIndexError::EmptyInputData);
    }
    if open.len() != cols {
        return Err(MarketMeannessIndexError::DataLengthMismatch);
    }

    for params in &combos {
        let input = MarketMeannessIndexInput::from_slices(open, close, params.clone());
        market_meanness_index_prepare(&input)?;
    }

    let mmi_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let mode = parse_source_mode(params.source_mode.as_deref().unwrap_or("Price"))
                .unwrap_or(MarketMeannessSourceMode::Price);
            let first = first_valid_bar(open, close, mode).unwrap_or(cols);
            mmi_warmup(params.length.unwrap_or(300), first).min(cols)
        })
        .collect();
    let smoothed_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let mode = parse_source_mode(params.source_mode.as_deref().unwrap_or("Price"))
                .unwrap_or(MarketMeannessSourceMode::Price);
            let first = first_valid_bar(open, close, mode).unwrap_or(cols);
            mmi_smoothed_warmup(params.length.unwrap_or(300), first).min(cols)
        })
        .collect();

    let mut mmi_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut mmi_matrix, cols, &mmi_warmups);
    let mut smoothed_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut smoothed_matrix, cols, &smoothed_warmups);

    let mut mmi_guard = ManuallyDrop::new(mmi_matrix);
    let mut smoothed_guard = ManuallyDrop::new(smoothed_matrix);
    let mmi_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(mmi_guard.as_mut_ptr(), mmi_guard.len()) };
    let smoothed_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(smoothed_guard.as_mut_ptr(), smoothed_guard.len())
    };

    let do_row = |row: usize,
                  row_mmi_mu: &mut [MaybeUninit<f64>],
                  row_smoothed_mu: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let mode = parse_source_mode(params.source_mode.as_deref().unwrap_or("Price"))
            .unwrap_or(MarketMeannessSourceMode::Price);
        let dst_mmi = unsafe {
            std::slice::from_raw_parts_mut(row_mmi_mu.as_mut_ptr() as *mut f64, row_mmi_mu.len())
        };
        let dst_smoothed = unsafe {
            std::slice::from_raw_parts_mut(
                row_smoothed_mu.as_mut_ptr() as *mut f64,
                row_smoothed_mu.len(),
            )
        };
        market_meanness_index_compute_into(
            open,
            close,
            params.length.unwrap_or(300),
            mode,
            kernel,
            dst_mmi,
            dst_smoothed,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        mmi_mu
            .par_chunks_mut(cols)
            .zip(smoothed_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_mmi, row_smoothed))| do_row(row, row_mmi, row_smoothed));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_mmi, row_smoothed)) in mmi_mu
            .chunks_mut(cols)
            .zip(smoothed_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_mmi, row_smoothed);
        }
    } else {
        for (row, (row_mmi, row_smoothed)) in mmi_mu
            .chunks_mut(cols)
            .zip(smoothed_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_mmi, row_smoothed);
        }
    }

    let mmi = unsafe {
        Vec::from_raw_parts(
            mmi_guard.as_mut_ptr() as *mut f64,
            mmi_guard.len(),
            mmi_guard.capacity(),
        )
    };
    let mmi_smoothed = unsafe {
        Vec::from_raw_parts(
            smoothed_guard.as_mut_ptr() as *mut f64,
            smoothed_guard.len(),
            smoothed_guard.capacity(),
        )
    };

    Ok(MarketMeannessIndexBatchOutput {
        mmi,
        mmi_smoothed,
        combos,
        rows,
        cols,
    })
}

fn market_meanness_index_batch_inner_into(
    open: &[f64],
    close: &[f64],
    sweep: &MarketMeannessIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_mmi: &mut [f64],
    out_mmi_smoothed: &mut [f64],
) -> Result<(), MarketMeannessIndexError> {
    let combos = expand_grid_market_meanness_index(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if out_mmi.len() != rows * cols || out_mmi_smoothed.len() != rows * cols {
        return Err(MarketMeannessIndexError::OutputLengthMismatch {
            expected: rows * cols,
            got: out_mmi.len().max(out_mmi_smoothed.len()),
        });
    }

    for params in &combos {
        let input = MarketMeannessIndexInput::from_slices(open, close, params.clone());
        market_meanness_index_prepare(&input)?;
    }

    for row in 0..rows {
        let row_mmi = &mut out_mmi[row * cols..(row + 1) * cols];
        let row_smoothed = &mut out_mmi_smoothed[row * cols..(row + 1) * cols];
        row_mmi.fill(f64::NAN);
        row_smoothed.fill(f64::NAN);
    }

    let do_row = |row: usize, row_mmi: &mut [f64], row_smoothed: &mut [f64]| {
        let params = &combos[row];
        let mode = parse_source_mode(params.source_mode.as_deref().unwrap_or("Price"))
            .unwrap_or(MarketMeannessSourceMode::Price);
        market_meanness_index_compute_into(
            open,
            close,
            params.length.unwrap_or(300),
            mode,
            kernel,
            row_mmi,
            row_smoothed,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mmi
            .par_chunks_mut(cols)
            .zip(out_mmi_smoothed.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_mmi, row_smoothed))| do_row(row, row_mmi, row_smoothed));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_mmi, row_smoothed)) in out_mmi
            .chunks_mut(cols)
            .zip(out_mmi_smoothed.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_mmi, row_smoothed);
        }
    } else {
        for (row, (row_mmi, row_smoothed)) in out_mmi
            .chunks_mut(cols)
            .zip(out_mmi_smoothed.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_mmi, row_smoothed);
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "market_meanness_index")]
#[pyo3(signature = (open, close, length=300, source_mode="Price", kernel=None))]
pub fn market_meanness_index_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    source_mode: &str,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let open = open.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = MarketMeannessIndexInput::from_slices(
        open,
        close,
        MarketMeannessIndexParams {
            length: Some(length),
            source_mode: Some(
                normalize_source_mode(source_mode)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        },
    );
    let output = py
        .allow_threads(|| market_meanness_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.mmi.into_pyarray(py),
        output.mmi_smoothed.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "MarketMeannessIndexStream")]
pub struct MarketMeannessIndexStreamPy {
    stream: MarketMeannessIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MarketMeannessIndexStreamPy {
    #[new]
    #[pyo3(signature = (length=300, source_mode="Price"))]
    fn new(length: usize, source_mode: &str) -> PyResult<Self> {
        let stream = MarketMeannessIndexStream::try_new(MarketMeannessIndexParams {
            length: Some(length),
            source_mode: Some(
                normalize_source_mode(source_mode)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update_reset_on_nan(open, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "market_meanness_index_batch")]
#[pyo3(signature = (open, close, length_range, source_mode="Price", kernel=None))]
pub fn market_meanness_index_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    source_mode: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let close = close.as_slice()?;
    let sweep = MarketMeannessIndexBatchRange {
        length: length_range,
        source_mode: Some(
            normalize_source_mode(source_mode).map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
    };
    let combos = expand_grid_market_meanness_index(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let mmi_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let smoothed_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_mmi = unsafe { mmi_arr.as_slice_mut()? };
    let out_smoothed = unsafe { smoothed_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        market_meanness_index_batch_inner_into(
            open,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_mmi,
            out_smoothed,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("mmi", mmi_arr.reshape((rows, cols))?)?;
    dict.set_item("mmi_smoothed", smoothed_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(300) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "source_modes",
        combos
            .iter()
            .map(|params| {
                params
                    .source_mode
                    .clone()
                    .unwrap_or_else(|| "Price".to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_market_meanness_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(market_meanness_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(market_meanness_index_batch_py, m)?)?;
    m.add_class::<MarketMeannessIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketMeannessIndexJsOutput {
    mmi: Vec<f64>,
    mmi_smoothed: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketMeannessIndexBatchConfig {
    length_range: Vec<usize>,
    source_mode: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketMeannessIndexBatchJsOutput {
    mmi: Vec<f64>,
    mmi_smoothed: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<MarketMeannessIndexParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "market_meanness_index_js")]
pub fn market_meanness_index_js(
    open: &[f64],
    close: &[f64],
    length: usize,
    source_mode: &str,
) -> Result<JsValue, JsValue> {
    let input = MarketMeannessIndexInput::from_slices(
        open,
        close,
        MarketMeannessIndexParams {
            length: Some(length),
            source_mode: Some(
                normalize_source_mode(source_mode)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
        },
    );
    let output = market_meanness_index(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MarketMeannessIndexJsOutput {
        mmi: output.mmi,
        mmi_smoothed: output.mmi_smoothed,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "market_meanness_index_batch_js")]
pub fn market_meanness_index_batch_js(
    open: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MarketMeannessIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = MarketMeannessIndexBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        source_mode: Some(
            normalize_source_mode(config.source_mode.as_deref().unwrap_or("Price"))
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        ),
    };
    let batch = market_meanness_index_batch_slice(open, close, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MarketMeannessIndexBatchJsOutput {
        mmi: batch.mmi,
        mmi_smoothed: batch.mmi_smoothed,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_meanness_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len * 2);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_meanness_index_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_meanness_index_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    source_mode: String,
) -> Result<(), JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to market_meanness_index_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_mmi, out_smoothed) = out.split_at_mut(len);
        let input = MarketMeannessIndexInput::from_slices(
            open,
            close,
            MarketMeannessIndexParams {
                length: Some(length),
                source_mode: Some(source_mode),
            },
        );
        market_meanness_index_into_slice(out_mmi, out_smoothed, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "market_meanness_index_into_host")]
pub fn market_meanness_index_into_host(
    open: &[f64],
    close: &[f64],
    out_ptr: *mut f64,
    length: usize,
    source_mode: &str,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to market_meanness_index_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len() * 2);
        let (out_mmi, out_smoothed) = out.split_at_mut(close.len());
        let input = MarketMeannessIndexInput::from_slices(
            open,
            close,
            MarketMeannessIndexParams {
                length: Some(length),
                source_mode: Some(
                    normalize_source_mode(source_mode)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?,
                ),
            },
        );
        market_meanness_index_into_slice(out_mmi, out_smoothed, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_meanness_index_batch_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    source_mode: String,
) -> Result<usize, JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to market_meanness_index_batch_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = MarketMeannessIndexBatchRange {
            length: (length_start, length_end, length_step),
            source_mode: Some(source_mode),
        };
        let combos = expand_grid_market_meanness_index(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len * 2);
        let (out_mmi, out_smoothed) = out.split_at_mut(rows * len);
        market_meanness_index_batch_inner_into(
            open,
            close,
            &sweep,
            Kernel::Scalar,
            false,
            out_mmi,
            out_smoothed,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_meanness_index_output_into_js(
    open: &[f64],
    close: &[f64],
    length: usize,
    source_mode: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = market_meanness_index_js(open, close, length, source_mode)?;
    crate::write_wasm_object_f64_outputs("market_meanness_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_meanness_index_batch_output_into_js(
    open: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = market_meanness_index_batch_js(open, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "market_meanness_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };

    fn assert_vec_close_nan(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            let a = actual[i];
            let e = expected[i];
            if a.is_nan() && e.is_nan() {
                continue;
            }
            assert!(
                (a - e).abs() < 1e-10,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
    }

    fn sample_open_close(len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.09 + (i as f64 * 0.17).sin() * 1.4;
            let op = base + (i as f64 * 0.11).cos() * 0.25;
            let cl = base + (i as f64 * 0.07).sin() * 0.55;
            open.push(op);
            close.push(cl);
        }
        (open, close)
    }

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (open, close) = sample_open_close(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        for i in 0..len {
            let hi = open[i].max(close[i]) + 0.6 + (i as f64 * 0.03).sin().abs();
            let lo = open[i].min(close[i]) - 0.6 - (i as f64 * 0.05).cos().abs();
            high.push(hi);
            low.push(lo);
        }
        (open, high, low, close)
    }

    fn naive_expected(
        open: &[f64],
        close: &[f64],
        length: usize,
        mode: MarketMeannessSourceMode,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let source: Vec<f64> = (0..n)
            .map(|i| source_value(open[i], close[i], mode))
            .collect();
        let mut mmi = vec![f64::NAN; n];
        let mut smoothed = vec![f64::NAN; n];

        for i in (length - 1)..n {
            let window = &source[i + 1 - length..=i];
            let mut med_buf = window.to_vec();
            let median = median_from(&mut med_buf);
            let mut count = 0usize;
            for j in 1..length {
                let prev = window[j - 1];
                let curr = window[j];
                if (curr > median && curr > prev) || (curr < median && curr < prev) {
                    count += 1;
                }
            }
            mmi[i] = count as f64 * (100.0 / (length - 1) as f64);
        }

        let mut sum = 0.0;
        for i in 0..n {
            if mmi[i].is_finite() {
                sum += mmi[i];
            }
            if i >= length && mmi[i - length].is_finite() {
                sum -= mmi[i - length];
            }
            if i + 1 >= (length * 2 - 1) {
                smoothed[i] = sum / length as f64;
            }
        }

        (mmi, smoothed)
    }

    #[test]
    fn market_meanness_index_matches_naive_price_and_change() {
        let (open, close) = sample_open_close(128);
        for mode in [
            MarketMeannessSourceMode::Price,
            MarketMeannessSourceMode::Change,
        ] {
            let params = MarketMeannessIndexParams {
                length: Some(12),
                source_mode: Some(mode.as_str().to_string()),
            };
            let input = MarketMeannessIndexInput::from_slices(&open, &close, params);
            let out = market_meanness_index(&input).unwrap();
            let (expected_mmi, expected_smoothed) = naive_expected(&open, &close, 12, mode);
            assert_vec_close_nan(&out.mmi, &expected_mmi);
            assert_vec_close_nan(&out.mmi_smoothed, &expected_smoothed);
        }
    }

    #[test]
    fn market_meanness_index_into_matches_api() {
        let (open, close) = sample_open_close(96);
        let input = MarketMeannessIndexInput::from_slices(
            &open,
            &close,
            MarketMeannessIndexParams {
                length: Some(10),
                source_mode: Some("Price".to_string()),
            },
        );
        let out = market_meanness_index(&input).unwrap();
        let mut mmi = vec![0.0; close.len()];
        let mut smoothed = vec![0.0; close.len()];
        market_meanness_index_into_slice(&mut mmi, &mut smoothed, &input, Kernel::Auto).unwrap();
        assert_vec_close_nan(&mmi, &out.mmi);
        assert_vec_close_nan(&smoothed, &out.mmi_smoothed);
    }

    #[test]
    fn market_meanness_index_stream_matches_batch() {
        let (open, close) = sample_open_close(144);
        let params = MarketMeannessIndexParams {
            length: Some(14),
            source_mode: Some("Change".to_string()),
        };
        let input = MarketMeannessIndexInput::from_slices(&open, &close, params.clone());
        let batch = market_meanness_index(&input).unwrap();
        let mut stream = MarketMeannessIndexStream::try_new(params).unwrap();
        let mut mmi = Vec::with_capacity(close.len());
        let mut smoothed = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            match stream.update_reset_on_nan(open[i], close[i]) {
                Some((a, b)) => {
                    mmi.push(a);
                    smoothed.push(b);
                }
                None => {
                    mmi.push(f64::NAN);
                    smoothed.push(f64::NAN);
                }
            }
        }
        assert_vec_close_nan(&mmi, &batch.mmi);
        assert_vec_close_nan(&smoothed, &batch.mmi_smoothed);
    }

    #[test]
    fn market_meanness_index_batch_single_param_matches_single() {
        let (open, close) = sample_open_close(160);
        let sweep = MarketMeannessIndexBatchRange {
            length: (12, 12, 0),
            source_mode: Some("Price".to_string()),
        };
        let batch = market_meanness_index_batch_slice(&open, &close, &sweep).unwrap();
        let input = MarketMeannessIndexInput::from_slices(
            &open,
            &close,
            MarketMeannessIndexParams {
                length: Some(12),
                source_mode: Some("Price".to_string()),
            },
        );
        let out = market_meanness_index(&input).unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_vec_close_nan(&batch.mmi[..close.len()], out.mmi.as_slice());
        assert_vec_close_nan(
            &batch.mmi_smoothed[..close.len()],
            out.mmi_smoothed.as_slice(),
        );
    }

    #[test]
    fn market_meanness_index_rejects_invalid_source_mode() {
        let (open, close) = sample_open_close(32);
        let input = MarketMeannessIndexInput::from_slices(
            &open,
            &close,
            MarketMeannessIndexParams {
                length: Some(12),
                source_mode: Some("bogus".to_string()),
            },
        );
        match market_meanness_index(&input) {
            Err(MarketMeannessIndexError::InvalidSourceMode { .. }) => {}
            other => panic!("expected InvalidSourceMode, got {other:?}"),
        }
    }

    #[test]
    fn market_meanness_index_dispatch_matches_direct() {
        let (open, high, low, close) = sample_ohlc(120);
        let combos = [IndicatorParamSet {
            params: &[
                ParamKV {
                    key: "length",
                    value: ParamValue::Int(12),
                },
                ParamKV {
                    key: "source_mode",
                    value: ParamValue::EnumString("Price"),
                },
            ],
        }];
        let req = IndicatorBatchRequest {
            indicator_id: "market_meanness_index",
            output_id: Some("mmi"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let direct = market_meanness_index(&MarketMeannessIndexInput::from_slices(
            &open,
            &close,
            MarketMeannessIndexParams {
                length: Some(12),
                source_mode: Some("Price".to_string()),
            },
        ))
        .unwrap();
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, close.len());
        assert_vec_close_nan(&out.values_f64.unwrap(), &direct.mmi);
    }
}
