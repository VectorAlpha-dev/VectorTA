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
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_COMPARISON_SOURCE: &str = "open";
const DEFAULT_LOOKBACK: usize = 30;
const DEFAULT_SMOOTHING_LENGTH: usize = 3;

#[derive(Debug, Clone)]
pub enum SpearmanCorrelationData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
        comparison_source: &'a str,
    },
    Slices {
        main: &'a [f64],
        compare: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct SpearmanCorrelationOutput {
    pub raw: Vec<f64>,
    pub smoothed: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SpearmanCorrelationParams {
    pub lookback: Option<usize>,
    pub smoothing_length: Option<usize>,
}

impl Default for SpearmanCorrelationParams {
    fn default() -> Self {
        Self {
            lookback: Some(DEFAULT_LOOKBACK),
            smoothing_length: Some(DEFAULT_SMOOTHING_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpearmanCorrelationInput<'a> {
    pub data: SpearmanCorrelationData<'a>,
    pub params: SpearmanCorrelationParams,
}

impl<'a> SpearmanCorrelationInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        comparison_source: &'a str,
        params: SpearmanCorrelationParams,
    ) -> Self {
        Self {
            data: SpearmanCorrelationData::Candles {
                candles,
                source,
                comparison_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        main: &'a [f64],
        compare: &'a [f64],
        params: SpearmanCorrelationParams,
    ) -> Self {
        Self {
            data: SpearmanCorrelationData::Slices { main, compare },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            DEFAULT_COMPARISON_SOURCE,
            SpearmanCorrelationParams::default(),
        )
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(DEFAULT_LOOKBACK)
    }

    #[inline]
    pub fn get_smoothing_length(&self) -> usize {
        self.params
            .smoothing_length
            .unwrap_or(DEFAULT_SMOOTHING_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SpearmanCorrelationBuilder {
    source: Option<&'static str>,
    comparison_source: Option<&'static str>,
    lookback: Option<usize>,
    smoothing_length: Option<usize>,
    kernel: Kernel,
}

impl Default for SpearmanCorrelationBuilder {
    fn default() -> Self {
        Self {
            source: None,
            comparison_source: None,
            lookback: None,
            smoothing_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SpearmanCorrelationBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn comparison_source(mut self, value: &'static str) -> Self {
        self.comparison_source = Some(value);
        self
    }

    #[inline(always)]
    pub fn lookback(mut self, value: usize) -> Self {
        self.lookback = Some(value);
        self
    }

    #[inline(always)]
    pub fn smoothing_length(mut self, value: usize) -> Self {
        self.smoothing_length = Some(value);
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
    ) -> Result<SpearmanCorrelationOutput, SpearmanCorrelationError> {
        let input = SpearmanCorrelationInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            self.comparison_source.unwrap_or(DEFAULT_COMPARISON_SOURCE),
            SpearmanCorrelationParams {
                lookback: self.lookback,
                smoothing_length: self.smoothing_length,
            },
        );
        spearman_correlation_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        main: &[f64],
        compare: &[f64],
    ) -> Result<SpearmanCorrelationOutput, SpearmanCorrelationError> {
        let input = SpearmanCorrelationInput::from_slices(
            main,
            compare,
            SpearmanCorrelationParams {
                lookback: self.lookback,
                smoothing_length: self.smoothing_length,
            },
        );
        spearman_correlation_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SpearmanCorrelationStream, SpearmanCorrelationError> {
        SpearmanCorrelationStream::try_new(SpearmanCorrelationParams {
            lookback: self.lookback,
            smoothing_length: self.smoothing_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum SpearmanCorrelationError {
    #[error("spearman_correlation: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "spearman_correlation: Inconsistent slice lengths: main={main_len}, compare={compare_len}"
    )]
    InconsistentSliceLengths { main_len: usize, compare_len: usize },
    #[error("spearman_correlation: All values are NaN.")]
    AllValuesNaN,
    #[error("spearman_correlation: Invalid lookback: lookback={lookback}, data length={data_len}")]
    InvalidLookback { lookback: usize, data_len: usize },
    #[error("spearman_correlation: Invalid smoothing length: {smoothing_length}")]
    InvalidSmoothingLength { smoothing_length: usize },
    #[error("spearman_correlation: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("spearman_correlation: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("spearman_correlation: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("spearman_correlation: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn extract_pair<'a>(
    input: &'a SpearmanCorrelationInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), SpearmanCorrelationError> {
    let (main, compare) = match &input.data {
        SpearmanCorrelationData::Candles {
            candles,
            source,
            comparison_source,
        } => (
            spearman_source_type(candles, source),
            spearman_source_type(candles, comparison_source),
        ),
        SpearmanCorrelationData::Slices { main, compare } => (*main, *compare),
    };
    if main.is_empty() || compare.is_empty() {
        return Err(SpearmanCorrelationError::EmptyInputData);
    }
    if main.len() != compare.len() {
        return Err(SpearmanCorrelationError::InconsistentSliceLengths {
            main_len: main.len(),
            compare_len: compare.len(),
        });
    }
    Ok((main, compare))
}

#[inline(always)]
fn spearman_source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "close" => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        "hl2" => &candles.hl2,
        "hlc3" => &candles.hlc3,
        "ohlc4" => &candles.ohlc4,
        "hlcc4" | "hlcc" => &candles.hlcc4,
        _ => source_type(candles, source),
    }
}

#[inline(always)]
fn first_valid_pair(main: &[f64], compare: &[f64]) -> Option<usize> {
    (0..main.len()).find(|&i| main[i].is_finite() && compare[i].is_finite())
}

#[inline(always)]
fn first_valid_return_idx(main: &[f64], compare: &[f64]) -> Option<usize> {
    (1..main.len()).find(|&i| {
        main[i].is_finite()
            && main[i - 1].is_finite()
            && compare[i].is_finite()
            && compare[i - 1].is_finite()
    })
}

#[inline(always)]
fn prepare<'a>(
    input: &'a SpearmanCorrelationInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, usize, Kernel), SpearmanCorrelationError> {
    let (main, compare) = extract_pair(input)?;
    let len = main.len();
    let lookback = input.get_lookback();
    let smoothing_length = input.get_smoothing_length();
    if lookback == 0 || lookback >= len {
        return Err(SpearmanCorrelationError::InvalidLookback {
            lookback,
            data_len: len,
        });
    }
    if smoothing_length == 0 {
        return Err(SpearmanCorrelationError::InvalidSmoothingLength { smoothing_length });
    }
    let _ = first_valid_pair(main, compare).ok_or(SpearmanCorrelationError::AllValuesNaN)?;
    let first_return = first_valid_return_idx(main, compare).ok_or(
        SpearmanCorrelationError::NotEnoughValidData {
            needed: lookback + 1,
            valid: 0,
        },
    )?;
    let valid = len.saturating_sub(first_return);
    if valid < lookback {
        return Err(SpearmanCorrelationError::NotEnoughValidData {
            needed: lookback,
            valid,
        });
    }
    Ok((
        main,
        compare,
        lookback,
        smoothing_length,
        first_return,
        kernel.to_non_batch(),
    ))
}

#[inline(always)]
fn rank_average(values: &[f64], indices: &mut [usize], ranks: &mut [f64]) {
    let n = values.len();
    for (i, slot) in indices.iter_mut().enumerate() {
        *slot = i;
    }
    indices.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));

    let mut start = 0usize;
    while start < n {
        let value = values[indices[start]];
        let mut end = start + 1;
        while end < n && values[indices[end]] == value {
            end += 1;
        }
        let avg_rank = (start as f64 + 1.0 + end as f64) * 0.5;
        for pos in start..end {
            ranks[indices[pos]] = avg_rank;
        }
        start = end;
    }
}

#[inline(always)]
fn rank_pearson_correlation(x: &[f64], y: &[f64], mean: f64) -> f64 {
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean;
        let dy = y[i] - mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        f64::NAN
    } else {
        cov / denom
    }
}

#[inline(always)]
#[inline(always)]
fn valid_return_pair(main_returns: &[f64], compare_returns: &[f64], idx: usize) -> bool {
    main_returns[idx].is_finite() && compare_returns[idx].is_finite()
}

#[inline(always)]
fn all_pairs_finite(main: &[f64], compare: &[f64]) -> bool {
    main.iter()
        .zip(compare)
        .all(|(main, compare)| main.is_finite() && compare.is_finite())
}

#[inline(always)]
fn compute_raw_all_finite_into(
    main: &[f64],
    compare: &[f64],
    lookback: usize,
    first_return: usize,
    raw_out: &mut [f64],
) {
    let n = main.len();
    let warm = first_return + lookback - 1;
    raw_out[..warm.min(n)].fill(f64::NAN);

    let mut main_returns = alloc_uninit_f64(n);
    let mut compare_returns = alloc_uninit_f64(n);
    main_returns[0] = f64::NAN;
    compare_returns[0] = f64::NAN;
    for i in 1..n {
        main_returns[i] = main[i] - main[i - 1];
        compare_returns[i] = compare[i] - compare[i - 1];
    }

    let mut main_indices = vec![0usize; lookback];
    let mut compare_indices = vec![0usize; lookback];
    let mut main_ranks = vec![0.0; lookback];
    let mut compare_ranks = vec![0.0; lookback];
    let rank_mean = (lookback as f64 + 1.0) * 0.5;

    for i in warm..n {
        let start = i + 1 - lookback;
        rank_average(&main_returns[start..=i], &mut main_indices, &mut main_ranks);
        rank_average(
            &compare_returns[start..=i],
            &mut compare_indices,
            &mut compare_ranks,
        );
        raw_out[i] = rank_pearson_correlation(&main_ranks, &compare_ranks, rank_mean);
    }
}

#[inline(always)]
fn compute_raw_into(
    main: &[f64],
    compare: &[f64],
    lookback: usize,
    first_return: usize,
    raw_out: &mut [f64],
) {
    if all_pairs_finite(main, compare) {
        compute_raw_all_finite_into(main, compare, lookback, first_return, raw_out);
        return;
    }

    let n = main.len();
    let warm = first_return + lookback - 1;
    raw_out[..warm.min(n)].fill(f64::NAN);

    let mut main_returns = alloc_uninit_f64(n);
    let mut compare_returns = alloc_uninit_f64(n);
    main_returns[0] = f64::NAN;
    compare_returns[0] = f64::NAN;
    for i in 1..n {
        let m0 = main[i - 1];
        let m1 = main[i];
        let c0 = compare[i - 1];
        let c1 = compare[i];
        if m0.is_finite() && m1.is_finite() && c0.is_finite() && c1.is_finite() {
            main_returns[i] = m1 - m0;
            compare_returns[i] = c1 - c0;
        } else {
            main_returns[i] = f64::NAN;
            compare_returns[i] = f64::NAN;
        }
    }

    let mut main_indices = vec![0usize; lookback];
    let mut compare_indices = vec![0usize; lookback];
    let mut main_ranks = vec![0.0; lookback];
    let mut compare_ranks = vec![0.0; lookback];
    let rank_mean = (lookback as f64 + 1.0) * 0.5;
    let mut valid_pairs = 0usize;
    for idx in first_return..=warm {
        if valid_return_pair(&main_returns, &compare_returns, idx) {
            valid_pairs += 1;
        }
    }

    for i in warm..n {
        if i != warm {
            let old = i - lookback;
            if valid_return_pair(&main_returns, &compare_returns, old) {
                valid_pairs -= 1;
            }
            if valid_return_pair(&main_returns, &compare_returns, i) {
                valid_pairs += 1;
            }
        }
        let start = i + 1 - lookback;
        let main_window = &main_returns[start..=i];
        let compare_window = &compare_returns[start..=i];
        if valid_pairs != lookback {
            raw_out[i] = f64::NAN;
            continue;
        }
        rank_average(main_window, &mut main_indices, &mut main_ranks);
        rank_average(compare_window, &mut compare_indices, &mut compare_ranks);
        raw_out[i] = rank_pearson_correlation(&main_ranks, &compare_ranks, rank_mean);
    }
}

#[inline(always)]
fn compute_smoothed_into(raw: &[f64], smoothing_length: usize, smoothed_out: &mut [f64]) {
    let mut sum = 0.0;
    let mut finite_count = 0usize;
    let denom = smoothing_length as f64;
    for i in 0..raw.len() {
        let value = raw[i];
        if value.is_finite() {
            sum += value;
            finite_count += 1;
        }
        if i >= smoothing_length {
            let old = raw[i - smoothing_length];
            if old.is_finite() {
                sum -= old;
                finite_count -= 1;
            }
        }
        if i + 1 >= smoothing_length && finite_count == smoothing_length {
            smoothed_out[i] = sum / denom;
        } else {
            smoothed_out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
fn compute_spearman_correlation_into(
    main: &[f64],
    compare: &[f64],
    lookback: usize,
    smoothing_length: usize,
    first_return: usize,
    raw_out: &mut [f64],
    smoothed_out: &mut [f64],
) {
    compute_raw_into(main, compare, lookback, first_return, raw_out);
    compute_smoothed_into(raw_out, smoothing_length, smoothed_out);
}

#[inline]
pub fn spearman_correlation(
    input: &SpearmanCorrelationInput,
) -> Result<SpearmanCorrelationOutput, SpearmanCorrelationError> {
    spearman_correlation_with_kernel(input, Kernel::Auto)
}

pub fn spearman_correlation_with_kernel(
    input: &SpearmanCorrelationInput,
    kernel: Kernel,
) -> Result<SpearmanCorrelationOutput, SpearmanCorrelationError> {
    let (main, compare, lookback, smoothing_length, first_return, _) = prepare(input, kernel)?;
    let raw_warm = first_return + lookback - 1;
    let smoothed_warm = raw_warm
        .saturating_add(smoothing_length.saturating_sub(1))
        .min(main.len());
    let mut out = SpearmanCorrelationOutput {
        raw: alloc_with_nan_prefix(main.len(), raw_warm),
        smoothed: alloc_with_nan_prefix(main.len(), smoothed_warm),
    };
    compute_spearman_correlation_into(
        main,
        compare,
        lookback,
        smoothing_length,
        first_return,
        &mut out.raw,
        &mut out.smoothed,
    );
    Ok(out)
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn spearman_correlation_into(
    raw_out: &mut [f64],
    smoothed_out: &mut [f64],
    input: &SpearmanCorrelationInput,
    kernel: Kernel,
) -> Result<(), SpearmanCorrelationError> {
    spearman_correlation_into_slice(raw_out, smoothed_out, input, kernel)
}

pub fn spearman_correlation_into_slice(
    raw_out: &mut [f64],
    smoothed_out: &mut [f64],
    input: &SpearmanCorrelationInput,
    kernel: Kernel,
) -> Result<(), SpearmanCorrelationError> {
    let (main, compare, lookback, smoothing_length, first_return, _) = prepare(input, kernel)?;
    let expected = main.len();
    if raw_out.len() != expected || smoothed_out.len() != expected {
        return Err(SpearmanCorrelationError::OutputLengthMismatch {
            expected,
            got: raw_out.len().max(smoothed_out.len()),
        });
    }
    compute_spearman_correlation_into(
        main,
        compare,
        lookback,
        smoothing_length,
        first_return,
        raw_out,
        smoothed_out,
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SpearmanCorrelationStream {
    lookback: usize,
    smoothing_length: usize,
    prev_main: f64,
    prev_compare: f64,
    has_prev: bool,
    main_returns: VecDeque<f64>,
    compare_returns: VecDeque<f64>,
    valid_return_pairs: usize,
    smoothing_window: VecDeque<f64>,
    smoothing_sum: f64,
    smoothing_finite_count: usize,
    main_window_buf: Vec<f64>,
    compare_window_buf: Vec<f64>,
    main_indices: Vec<usize>,
    compare_indices: Vec<usize>,
    main_ranks: Vec<f64>,
    compare_ranks: Vec<f64>,
}

impl SpearmanCorrelationStream {
    pub fn try_new(params: SpearmanCorrelationParams) -> Result<Self, SpearmanCorrelationError> {
        let lookback = params.lookback.unwrap_or(DEFAULT_LOOKBACK);
        let smoothing_length = params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
        if lookback == 0 {
            return Err(SpearmanCorrelationError::InvalidLookback {
                lookback,
                data_len: 0,
            });
        }
        if smoothing_length == 0 {
            return Err(SpearmanCorrelationError::InvalidSmoothingLength { smoothing_length });
        }
        Ok(Self {
            lookback,
            smoothing_length,
            prev_main: f64::NAN,
            prev_compare: f64::NAN,
            has_prev: false,
            main_returns: VecDeque::with_capacity(lookback + 1),
            compare_returns: VecDeque::with_capacity(lookback + 1),
            valid_return_pairs: 0,
            smoothing_window: VecDeque::with_capacity(smoothing_length + 1),
            smoothing_sum: 0.0,
            smoothing_finite_count: 0,
            main_window_buf: vec![f64::NAN; lookback],
            compare_window_buf: vec![f64::NAN; lookback],
            main_indices: vec![0usize; lookback],
            compare_indices: vec![0usize; lookback],
            main_ranks: vec![0.0; lookback],
            compare_ranks: vec![0.0; lookback],
        })
    }

    #[inline]
    fn update_smoothing(&mut self, value: f64) -> f64 {
        self.smoothing_window.push_back(value);
        if value.is_finite() {
            self.smoothing_sum += value;
            self.smoothing_finite_count += 1;
        }
        if self.smoothing_window.len() > self.smoothing_length {
            if let Some(old) = self.smoothing_window.pop_front() {
                if old.is_finite() {
                    self.smoothing_sum -= old;
                    self.smoothing_finite_count -= 1;
                }
            }
        }
        if self.smoothing_window.len() == self.smoothing_length
            && self.smoothing_finite_count == self.smoothing_length
        {
            self.smoothing_sum / self.smoothing_length as f64
        } else {
            f64::NAN
        }
    }

    #[inline]
    pub fn update(&mut self, main: f64, compare: f64) -> (f64, f64) {
        let (main_ret, compare_ret) = if self.has_prev
            && self.prev_main.is_finite()
            && self.prev_compare.is_finite()
            && main.is_finite()
            && compare.is_finite()
        {
            (main - self.prev_main, compare - self.prev_compare)
        } else {
            (f64::NAN, f64::NAN)
        };

        self.prev_main = main;
        self.prev_compare = compare;
        self.has_prev = true;

        self.main_returns.push_back(main_ret);
        self.compare_returns.push_back(compare_ret);
        if main_ret.is_finite() && compare_ret.is_finite() {
            self.valid_return_pairs += 1;
        }
        if self.main_returns.len() > self.lookback {
            let old_main = self.main_returns.pop_front().unwrap_or(f64::NAN);
            let old_compare = self.compare_returns.pop_front().unwrap_or(f64::NAN);
            if old_main.is_finite() && old_compare.is_finite() {
                self.valid_return_pairs -= 1;
            }
        }

        let raw = if self.main_returns.len() == self.lookback
            && self.valid_return_pairs == self.lookback
        {
            for (dst, value) in self
                .main_window_buf
                .iter_mut()
                .zip(self.main_returns.iter().copied())
            {
                *dst = value;
            }
            for (dst, value) in self
                .compare_window_buf
                .iter_mut()
                .zip(self.compare_returns.iter().copied())
            {
                *dst = value;
            }
            rank_average(
                &self.main_window_buf,
                &mut self.main_indices,
                &mut self.main_ranks,
            );
            rank_average(
                &self.compare_window_buf,
                &mut self.compare_indices,
                &mut self.compare_ranks,
            );
            rank_pearson_correlation(
                &self.main_ranks,
                &self.compare_ranks,
                (self.lookback as f64 + 1.0) * 0.5,
            )
        } else {
            f64::NAN
        };

        let smoothed = self.update_smoothing(raw);
        (raw, smoothed)
    }
}

#[derive(Debug, Clone)]
pub struct SpearmanCorrelationBatchRange {
    pub lookback: (usize, usize, usize),
    pub smoothing_length: (usize, usize, usize),
}

#[derive(Debug, Clone)]
pub struct SpearmanCorrelationBatchOutput {
    pub raw: Vec<f64>,
    pub smoothed: Vec<f64>,
    pub combos: Vec<SpearmanCorrelationParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct SpearmanCorrelationBatchBuilder {
    source: Option<&'static str>,
    comparison_source: Option<&'static str>,
    lookback: (usize, usize, usize),
    smoothing_length: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for SpearmanCorrelationBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            comparison_source: None,
            lookback: (DEFAULT_LOOKBACK, DEFAULT_LOOKBACK, 0),
            smoothing_length: (DEFAULT_SMOOTHING_LENGTH, DEFAULT_SMOOTHING_LENGTH, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl SpearmanCorrelationBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn comparison_source(mut self, value: &'static str) -> Self {
        self.comparison_source = Some(value);
        self
    }

    #[inline(always)]
    pub fn lookback_range(mut self, value: (usize, usize, usize)) -> Self {
        self.lookback = value;
        self
    }

    #[inline(always)]
    pub fn smoothing_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.smoothing_length = value;
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
    ) -> Result<SpearmanCorrelationBatchOutput, SpearmanCorrelationError> {
        spearman_correlation_batch_with_kernel(
            source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE)),
            source_type(
                candles,
                self.comparison_source.unwrap_or(DEFAULT_COMPARISON_SOURCE),
            ),
            &SpearmanCorrelationBatchRange {
                lookback: self.lookback,
                smoothing_length: self.smoothing_length,
            },
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        main: &[f64],
        compare: &[f64],
    ) -> Result<SpearmanCorrelationBatchOutput, SpearmanCorrelationError> {
        spearman_correlation_batch_with_kernel(
            main,
            compare,
            &SpearmanCorrelationBatchRange {
                lookback: self.lookback,
                smoothing_length: self.smoothing_length,
            },
            self.kernel,
        )
    }
}

fn expand_one_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, SpearmanCorrelationError> {
    if start == 0 {
        return Err(SpearmanCorrelationError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        if start != end {
            return Err(SpearmanCorrelationError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end {
        return Err(SpearmanCorrelationError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end {
        out.push(current);
        current = match current.checked_add(step) {
            Some(next) => next,
            None => break,
        };
    }
    Ok(out)
}

pub fn expand_grid(
    sweep: &SpearmanCorrelationBatchRange,
) -> Result<Vec<SpearmanCorrelationParams>, SpearmanCorrelationError> {
    let lookbacks = expand_one_range(sweep.lookback.0, sweep.lookback.1, sweep.lookback.2)?;
    let smoothing_lengths = expand_one_range(
        sweep.smoothing_length.0,
        sweep.smoothing_length.1,
        sweep.smoothing_length.2,
    )?;
    let mut out = Vec::new();
    for lookback in lookbacks {
        for &smoothing_length in &smoothing_lengths {
            out.push(SpearmanCorrelationParams {
                lookback: Some(lookback),
                smoothing_length: Some(smoothing_length),
            });
        }
    }
    Ok(out)
}

fn validate_raw_slices(main: &[f64], compare: &[f64]) -> Result<usize, SpearmanCorrelationError> {
    if main.is_empty() || compare.is_empty() {
        return Err(SpearmanCorrelationError::EmptyInputData);
    }
    if main.len() != compare.len() {
        return Err(SpearmanCorrelationError::InconsistentSliceLengths {
            main_len: main.len(),
            compare_len: compare.len(),
        });
    }
    first_valid_pair(main, compare).ok_or(SpearmanCorrelationError::AllValuesNaN)
}

pub fn spearman_correlation_batch_with_kernel(
    main: &[f64],
    compare: &[f64],
    sweep: &SpearmanCorrelationBatchRange,
    kernel: Kernel,
) -> Result<SpearmanCorrelationBatchOutput, SpearmanCorrelationError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SpearmanCorrelationError::InvalidKernelForBatch(kernel)),
    };
    spearman_correlation_batch_par_slice(main, compare, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn spearman_correlation_batch_slice(
    main: &[f64],
    compare: &[f64],
    sweep: &SpearmanCorrelationBatchRange,
    kernel: Kernel,
) -> Result<SpearmanCorrelationBatchOutput, SpearmanCorrelationError> {
    spearman_correlation_batch_inner(main, compare, sweep, kernel, false)
}

#[inline(always)]
pub fn spearman_correlation_batch_par_slice(
    main: &[f64],
    compare: &[f64],
    sweep: &SpearmanCorrelationBatchRange,
    kernel: Kernel,
) -> Result<SpearmanCorrelationBatchOutput, SpearmanCorrelationError> {
    spearman_correlation_batch_inner(main, compare, sweep, kernel, true)
}

fn spearman_correlation_batch_inner(
    main: &[f64],
    compare: &[f64],
    sweep: &SpearmanCorrelationBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<SpearmanCorrelationBatchOutput, SpearmanCorrelationError> {
    let combos = expand_grid(sweep)?;
    let _ = validate_raw_slices(main, compare)?;
    let rows = combos.len();
    let cols = main.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| SpearmanCorrelationError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;

    let first_return = first_valid_return_idx(main, compare).unwrap_or(1);
    let raw_warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            let lookback = combo.lookback.unwrap_or(DEFAULT_LOOKBACK);
            first_return.saturating_add(lookback.saturating_sub(1))
        })
        .collect();
    let smoothed_warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            let lookback = combo.lookback.unwrap_or(DEFAULT_LOOKBACK);
            let smoothing_length = combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
            first_return
                .saturating_add(lookback.saturating_sub(1))
                .saturating_add(smoothing_length.saturating_sub(1))
        })
        .collect();

    let mut raw_buf = make_uninit_matrix(rows, cols);
    let mut smoothed_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut raw_buf, cols, &raw_warmups);
    init_matrix_prefixes(&mut smoothed_buf, cols, &smoothed_warmups);

    let mut raw_guard = ManuallyDrop::new(raw_buf);
    let raw_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(raw_guard.as_mut_ptr() as *mut f64, raw_guard.len())
    };
    let mut smoothed_guard = ManuallyDrop::new(smoothed_buf);
    let smoothed_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            smoothed_guard.as_mut_ptr() as *mut f64,
            smoothed_guard.len(),
        )
    };

    spearman_correlation_batch_inner_into(
        main,
        compare,
        sweep,
        kernel,
        parallel,
        raw_out,
        smoothed_out,
    )?;

    let raw_values = unsafe {
        Vec::from_raw_parts(
            raw_guard.as_mut_ptr() as *mut f64,
            expected,
            raw_guard.capacity(),
        )
    };
    let smoothed_values = unsafe {
        Vec::from_raw_parts(
            smoothed_guard.as_mut_ptr() as *mut f64,
            expected,
            smoothed_guard.capacity(),
        )
    };

    Ok(SpearmanCorrelationBatchOutput {
        raw: raw_values,
        smoothed: smoothed_values,
        combos,
        rows,
        cols,
    })
}

pub fn spearman_correlation_batch_into_slice(
    raw_out: &mut [f64],
    smoothed_out: &mut [f64],
    main: &[f64],
    compare: &[f64],
    sweep: &SpearmanCorrelationBatchRange,
    kernel: Kernel,
) -> Result<(), SpearmanCorrelationError> {
    spearman_correlation_batch_inner_into(
        main,
        compare,
        sweep,
        kernel,
        false,
        raw_out,
        smoothed_out,
    )?;
    Ok(())
}

fn spearman_correlation_batch_inner_into(
    main: &[f64],
    compare: &[f64],
    sweep: &SpearmanCorrelationBatchRange,
    _kernel: Kernel,
    parallel: bool,
    raw_out: &mut [f64],
    smoothed_out: &mut [f64],
) -> Result<Vec<SpearmanCorrelationParams>, SpearmanCorrelationError> {
    let combos = expand_grid(sweep)?;
    let _ = validate_raw_slices(main, compare)?;
    let first_return = first_valid_return_idx(main, compare).ok_or(
        SpearmanCorrelationError::NotEnoughValidData {
            needed: 2,
            valid: 0,
        },
    )?;
    let rows = combos.len();
    let cols = main.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| SpearmanCorrelationError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    if raw_out.len() != expected || smoothed_out.len() != expected {
        return Err(SpearmanCorrelationError::OutputLengthMismatch {
            expected,
            got: raw_out.len().max(smoothed_out.len()),
        });
    }
    let max_lookback = combos
        .iter()
        .map(|combo| combo.lookback.unwrap_or(DEFAULT_LOOKBACK))
        .max()
        .unwrap_or(DEFAULT_LOOKBACK);
    let valid = cols.saturating_sub(first_return);
    if valid < max_lookback {
        return Err(SpearmanCorrelationError::NotEnoughValidData {
            needed: max_lookback,
            valid,
        });
    }

    let do_row = |row: usize, dst_raw: &mut [f64], dst_smoothed: &mut [f64]| {
        let params = &combos[row];
        let lookback = params.lookback.unwrap_or(DEFAULT_LOOKBACK);
        let smoothing_length = params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
        compute_spearman_correlation_into(
            main,
            compare,
            lookback,
            smoothing_length,
            first_return,
            dst_raw,
            dst_smoothed,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw_out
                .par_chunks_mut(cols)
                .zip(smoothed_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (dst_raw, dst_smoothed))| do_row(row, dst_raw, dst_smoothed));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for ((row, dst_raw), dst_smoothed) in raw_out
                .chunks_mut(cols)
                .enumerate()
                .zip(smoothed_out.chunks_mut(cols))
            {
                do_row(row, dst_raw, dst_smoothed);
            }
        }
    } else {
        for ((row, dst_raw), dst_smoothed) in raw_out
            .chunks_mut(cols)
            .enumerate()
            .zip(smoothed_out.chunks_mut(cols))
        {
            do_row(row, dst_raw, dst_smoothed);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "spearman_correlation")]
#[pyo3(signature = (main, compare, lookback=30, smoothing_length=3, kernel=None))]
pub fn spearman_correlation_py<'py>(
    py: Python<'py>,
    main: PyReadonlyArray1<'py, f64>,
    compare: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    smoothing_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let main = main.as_slice()?;
    let compare = compare.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = SpearmanCorrelationInput::from_slices(
        main,
        compare,
        SpearmanCorrelationParams {
            lookback: Some(lookback),
            smoothing_length: Some(smoothing_length),
        },
    );
    let out = py
        .allow_threads(|| spearman_correlation_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("raw", out.raw.into_pyarray(py))?;
    dict.set_item("smoothed", out.smoothed.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "SpearmanCorrelationStream")]
pub struct SpearmanCorrelationStreamPy {
    stream: SpearmanCorrelationStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SpearmanCorrelationStreamPy {
    #[new]
    #[pyo3(signature = (lookback=30, smoothing_length=3))]
    fn new(lookback: usize, smoothing_length: usize) -> PyResult<Self> {
        let stream = SpearmanCorrelationStream::try_new(SpearmanCorrelationParams {
            lookback: Some(lookback),
            smoothing_length: Some(smoothing_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        main: f64,
        compare: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (raw, smoothed) = self.stream.update(main, compare);
        let dict = PyDict::new(py);
        dict.set_item("raw", raw)?;
        dict.set_item("smoothed", smoothed)?;
        Ok(dict)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "spearman_correlation_batch")]
#[pyo3(signature = (
    main,
    compare,
    lookback_range=(30,30,0),
    smoothing_length_range=(3,3,0),
    kernel=None
))]
pub fn spearman_correlation_batch_py<'py>(
    py: Python<'py>,
    main: PyReadonlyArray1<'py, f64>,
    compare: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    smoothing_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let main = main.as_slice()?;
    let compare = compare.as_slice()?;
    let sweep = SpearmanCorrelationBatchRange {
        lookback: lookback_range,
        smoothing_length: smoothing_length_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = main.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_raw = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_smoothed = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let raw_slice = unsafe { out_raw.as_slice_mut()? };
    let smoothed_slice = unsafe { out_smoothed.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        spearman_correlation_batch_inner_into(
            main,
            compare,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            raw_slice,
            smoothed_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("raw", out_raw.reshape((rows, cols))?)?;
    dict.set_item("smoothed", out_smoothed.reshape((rows, cols))?)?;
    dict.set_item(
        "lookback",
        combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(DEFAULT_LOOKBACK) as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smoothing_length",
        combos
            .iter()
            .map(|combo| combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH) as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_spearman_correlation_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spearman_correlation_py, m)?)?;
    m.add_function(wrap_pyfunction!(spearman_correlation_batch_py, m)?)?;
    m.add_class::<SpearmanCorrelationStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SpearmanCorrelationJsOutput {
    pub raw: Vec<f64>,
    pub smoothed: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "spearman_correlation_js")]
pub fn spearman_correlation_js(
    main: &[f64],
    compare: &[f64],
    lookback: usize,
    smoothing_length: usize,
) -> Result<JsValue, JsValue> {
    let input = SpearmanCorrelationInput::from_slices(
        main,
        compare,
        SpearmanCorrelationParams {
            lookback: Some(lookback),
            smoothing_length: Some(smoothing_length),
        },
    );
    let out = spearman_correlation_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&SpearmanCorrelationJsOutput {
        raw: out.raw,
        smoothed: out.smoothed,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SpearmanCorrelationBatchConfig {
    pub lookback_range: Vec<f64>,
    pub smoothing_length_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SpearmanCorrelationBatchJsOutput {
    pub raw: Vec<f64>,
    pub smoothed: Vec<f64>,
    pub lookback: Vec<usize>,
    pub smoothing_length: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (i, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a finite non-negative whole number"
            )));
        }
        let rounded = value.round();
        if (value - rounded).abs() > 1e-9 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a whole number"
            )));
        }
        out[i] = rounded as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "spearman_correlation_batch_js")]
pub fn spearman_correlation_batch_js(
    main: &[f64],
    compare: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: SpearmanCorrelationBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = SpearmanCorrelationBatchRange {
        lookback: js_vec3_to_usize("lookback_range", &config.lookback_range)?,
        smoothing_length: js_vec3_to_usize(
            "smoothing_length_range",
            &config.smoothing_length_range,
        )?,
    };
    let out = spearman_correlation_batch_with_kernel(main, compare, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let lookback = out
        .combos
        .iter()
        .map(|combo| combo.lookback.unwrap_or(DEFAULT_LOOKBACK))
        .collect();
    let smoothing_length = out
        .combos
        .iter()
        .map(|combo| combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH))
        .collect();
    serde_wasm_bindgen::to_value(&SpearmanCorrelationBatchJsOutput {
        raw: out.raw,
        smoothed: out.smoothed,
        lookback,
        smoothing_length,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn spearman_correlation_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn spearman_correlation_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn spearman_correlation_into(
    main_ptr: *const f64,
    compare_ptr: *const f64,
    raw_out_ptr: *mut f64,
    smoothed_out_ptr: *mut f64,
    len: usize,
    lookback: usize,
    smoothing_length: usize,
) -> Result<(), JsValue> {
    if main_ptr.is_null()
        || compare_ptr.is_null()
        || raw_out_ptr.is_null()
        || smoothed_out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let main = std::slice::from_raw_parts(main_ptr, len);
        let compare = std::slice::from_raw_parts(compare_ptr, len);
        let raw_out = std::slice::from_raw_parts_mut(raw_out_ptr, len);
        let smoothed_out = std::slice::from_raw_parts_mut(smoothed_out_ptr, len);
        let input = SpearmanCorrelationInput::from_slices(
            main,
            compare,
            SpearmanCorrelationParams {
                lookback: Some(lookback),
                smoothing_length: Some(smoothing_length),
            },
        );
        spearman_correlation_into_slice(raw_out, smoothed_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn spearman_correlation_batch_into(
    main_ptr: *const f64,
    compare_ptr: *const f64,
    raw_out_ptr: *mut f64,
    smoothed_out_ptr: *mut f64,
    len: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    smoothing_length_start: usize,
    smoothing_length_end: usize,
    smoothing_length_step: usize,
) -> Result<usize, JsValue> {
    if main_ptr.is_null()
        || compare_ptr.is_null()
        || raw_out_ptr.is_null()
        || smoothed_out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to spearman_correlation_batch_into",
        ));
    }
    unsafe {
        let main = std::slice::from_raw_parts(main_ptr, len);
        let compare = std::slice::from_raw_parts(compare_ptr, len);
        let sweep = SpearmanCorrelationBatchRange {
            lookback: (lookback_start, lookback_end, lookback_step),
            smoothing_length: (
                smoothing_length_start,
                smoothing_length_end,
                smoothing_length_step,
            ),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in spearman_correlation_batch_into")
        })?;
        let raw_out = std::slice::from_raw_parts_mut(raw_out_ptr, total);
        let smoothed_out = std::slice::from_raw_parts_mut(smoothed_out_ptr, total);
        spearman_correlation_batch_into_slice(
            raw_out,
            smoothed_out,
            main,
            compare,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn spearman_correlation_output_into_js(
    main: &[f64],
    compare: &[f64],
    lookback: usize,
    smoothing_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = spearman_correlation_js(main, compare, lookback, smoothing_length)?;
    crate::write_wasm_object_f64_outputs("spearman_correlation_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn spearman_correlation_batch_output_into_js(
    main: &[f64],
    compare: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = spearman_correlation_batch_js(main, compare, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "spearman_correlation_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pair(n: usize) -> (Vec<f64>, Vec<f64>) {
        let main: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.17).sin() * 2.5 + i as f64 * 0.03)
            .collect();
        let compare: Vec<f64> = (0..n)
            .map(|i| 75.0 + ((i as f64) * 0.11).cos() * 1.8 + i as f64 * 0.015)
            .collect();
        (main, compare)
    }

    fn manual_reference(
        main: &[f64],
        compare: &[f64],
        lookback: usize,
        smoothing_length: usize,
    ) -> SpearmanCorrelationOutput {
        let first_return = first_valid_return_idx(main, compare).unwrap();
        let mut raw = vec![f64::NAN; main.len()];
        let mut smoothed = vec![f64::NAN; main.len()];
        compute_spearman_correlation_into(
            main,
            compare,
            lookback,
            smoothing_length,
            first_return,
            &mut raw,
            &mut smoothed,
        );
        SpearmanCorrelationOutput { raw, smoothed }
    }

    #[test]
    fn manual_reference_matches_binding() {
        let (main, compare) = sample_pair(160);
        let input = SpearmanCorrelationInput::from_slices(
            &main,
            &compare,
            SpearmanCorrelationParams {
                lookback: Some(30),
                smoothing_length: Some(3),
            },
        );
        let out = spearman_correlation(&input).unwrap();
        let want = manual_reference(&main, &compare, 30, 3);
        for i in 0..main.len() {
            assert!(
                (out.raw[i].is_nan() && want.raw[i].is_nan())
                    || (out.raw[i] - want.raw[i]).abs() < 1e-12
            );
            assert!(
                (out.smoothed[i].is_nan() && want.smoothed[i].is_nan())
                    || (out.smoothed[i] - want.smoothed[i]).abs() < 1e-12
            );
        }
    }

    #[test]
    fn stream_matches_batch_last_value() {
        let (main, compare) = sample_pair(180);
        let input = SpearmanCorrelationInput::from_slices(
            &main,
            &compare,
            SpearmanCorrelationParams::default(),
        );
        let batch = spearman_correlation(&input).unwrap();
        let mut stream =
            SpearmanCorrelationStream::try_new(SpearmanCorrelationParams::default()).unwrap();
        let mut last = (f64::NAN, f64::NAN);
        for i in 0..main.len() {
            last = stream.update(main[i], compare[i]);
        }
        assert!(
            (last.0.is_nan() && batch.raw.last().unwrap().is_nan())
                || (last.0 - batch.raw.last().unwrap()).abs() < 1e-12
        );
        assert!(
            (last.1.is_nan() && batch.smoothed.last().unwrap().is_nan())
                || (last.1 - batch.smoothed.last().unwrap()).abs() < 1e-12
        );
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (main, compare) = sample_pair(128);
        let sweep = SpearmanCorrelationBatchRange {
            lookback: (30, 32, 2),
            smoothing_length: (3, 3, 0),
        };
        let batch =
            spearman_correlation_batch_with_kernel(&main, &compare, &sweep, Kernel::Auto).unwrap();
        let single = spearman_correlation(&SpearmanCorrelationInput::from_slices(
            &main,
            &compare,
            SpearmanCorrelationParams::default(),
        ))
        .unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, main.len());
        assert_eq!(batch.combos[0].lookback, Some(30));
        assert_eq!(batch.combos[1].lookback, Some(32));
        for i in 0..main.len() {
            assert!(
                (batch.raw[i].is_nan() && single.raw[i].is_nan())
                    || (batch.raw[i] - single.raw[i]).abs() < 1e-12
            );
            assert!(
                (batch.smoothed[i].is_nan() && single.smoothed[i].is_nan())
                    || (batch.smoothed[i] - single.smoothed[i]).abs() < 1e-12
            );
        }
    }

    #[test]
    fn into_slice_matches_single() {
        let (main, compare) = sample_pair(144);
        let input = SpearmanCorrelationInput::from_slices(
            &main,
            &compare,
            SpearmanCorrelationParams::default(),
        );
        let single = spearman_correlation(&input).unwrap();
        let mut raw = vec![f64::NAN; main.len()];
        let mut smoothed = vec![f64::NAN; main.len()];
        spearman_correlation_into_slice(&mut raw, &mut smoothed, &input, Kernel::Auto).unwrap();
        for i in 0..main.len() {
            assert!(
                (raw[i].is_nan() && single.raw[i].is_nan())
                    || (raw[i] - single.raw[i]).abs() < 1e-12
            );
            assert!(
                (smoothed[i].is_nan() && single.smoothed[i].is_nan())
                    || (smoothed[i] - single.smoothed[i]).abs() < 1e-12
            );
        }
    }

    #[test]
    fn rejects_invalid_lookback() {
        let (main, compare) = sample_pair(32);
        let input = SpearmanCorrelationInput::from_slices(
            &main,
            &compare,
            SpearmanCorrelationParams {
                lookback: Some(0),
                smoothing_length: Some(3),
            },
        );
        let err = spearman_correlation(&input).unwrap_err();
        assert!(matches!(
            err,
            SpearmanCorrelationError::InvalidLookback { .. }
        ));
    }
}
