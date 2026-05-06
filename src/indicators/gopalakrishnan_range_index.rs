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
    alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum GopalakrishnanRangeIndexData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct GopalakrishnanRangeIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct GopalakrishnanRangeIndexParams {
    pub length: Option<usize>,
}

impl Default for GopalakrishnanRangeIndexParams {
    fn default() -> Self {
        Self { length: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct GopalakrishnanRangeIndexInput<'a> {
    pub data: GopalakrishnanRangeIndexData<'a>,
    pub params: GopalakrishnanRangeIndexParams,
}

impl<'a> GopalakrishnanRangeIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: GopalakrishnanRangeIndexParams) -> Self {
        Self {
            data: GopalakrishnanRangeIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        params: GopalakrishnanRangeIndexParams,
    ) -> Self {
        Self {
            data: GopalakrishnanRangeIndexData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, GopalakrishnanRangeIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GopalakrishnanRangeIndexBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for GopalakrishnanRangeIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GopalakrishnanRangeIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<GopalakrishnanRangeIndexOutput, GopalakrishnanRangeIndexError> {
        let input = GopalakrishnanRangeIndexInput::from_candles(
            candles,
            GopalakrishnanRangeIndexParams {
                length: self.length,
            },
        );
        gopalakrishnan_range_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<GopalakrishnanRangeIndexOutput, GopalakrishnanRangeIndexError> {
        let input = GopalakrishnanRangeIndexInput::from_slices(
            high,
            low,
            GopalakrishnanRangeIndexParams {
                length: self.length,
            },
        );
        gopalakrishnan_range_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<GopalakrishnanRangeIndexStream, GopalakrishnanRangeIndexError> {
        GopalakrishnanRangeIndexStream::try_new(GopalakrishnanRangeIndexParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum GopalakrishnanRangeIndexError {
    #[error("gopalakrishnan_range_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("gopalakrishnan_range_index: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "gopalakrishnan_range_index: Invalid length: length = {length}, data length = {data_len}. Length must be > 1."
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "gopalakrishnan_range_index: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "gopalakrishnan_range_index: Inconsistent slice lengths: high={high_len}, low={low_len}"
    )]
    InconsistentSliceLengths { high_len: usize, low_len: usize },
    #[error(
        "gopalakrishnan_range_index: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("gopalakrishnan_range_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("gopalakrishnan_range_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct GopalakrishnanRangeIndexStream {
    length: usize,
    seen: usize,
    started: bool,
    dq_high: VecDeque<(usize, f64)>,
    dq_low: VecDeque<(usize, f64)>,
    valid: Vec<u8>,
    valid_idx: usize,
    valid_cnt: usize,
    total_cnt: usize,
    log_length: f64,
}

impl GopalakrishnanRangeIndexStream {
    pub fn try_new(
        params: GopalakrishnanRangeIndexParams,
    ) -> Result<GopalakrishnanRangeIndexStream, GopalakrishnanRangeIndexError> {
        let length = params.length.unwrap_or(5);
        if length <= 1 {
            return Err(GopalakrishnanRangeIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self {
            length,
            seen: 0,
            started: false,
            dq_high: VecDeque::with_capacity(length + 1),
            dq_low: VecDeque::with_capacity(length + 1),
            valid: vec![0u8; length],
            valid_idx: 0,
            valid_cnt: 0,
            total_cnt: 0,
            log_length: (length as f64).ln(),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        if !self.started {
            if !valid_high_low_bar(high, low) {
                return None;
            }
            self.started = true;
        }

        if self.total_cnt >= self.length {
            let old_idx = self.valid_idx;
            if self.valid[old_idx] != 0 {
                self.valid_cnt = self.valid_cnt.saturating_sub(1);
            }
        } else {
            self.total_cnt += 1;
        }

        let i = self.seen;
        if valid_high_low_bar(high, low) {
            while let Some(&(_, back)) = self.dq_high.back() {
                if back <= high {
                    self.dq_high.pop_back();
                } else {
                    break;
                }
            }
            self.dq_high.push_back((i, high));

            while let Some(&(_, back)) = self.dq_low.back() {
                if back >= low {
                    self.dq_low.pop_back();
                } else {
                    break;
                }
            }
            self.dq_low.push_back((i, low));

            self.valid[self.valid_idx] = 1;
            self.valid_cnt += 1;
        } else {
            self.valid[self.valid_idx] = 0;
        }

        self.valid_idx += 1;
        if self.valid_idx == self.length {
            self.valid_idx = 0;
        }

        let start = i.saturating_add(1).saturating_sub(self.length);
        while let Some(&(idx, _)) = self.dq_high.front() {
            if idx < start {
                self.dq_high.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(idx, _)) = self.dq_low.front() {
            if idx < start {
                self.dq_low.pop_front();
            } else {
                break;
            }
        }

        self.seen = i + 1;
        if self.total_cnt < self.length {
            return None;
        }
        if self.valid_cnt != self.length {
            return Some(f64::NAN);
        }

        let highest = self
            .dq_high
            .front()
            .map(|&(_, v)| v)
            .unwrap_or(f64::NEG_INFINITY);
        let lowest = self
            .dq_low
            .front()
            .map(|&(_, v)| v)
            .unwrap_or(f64::INFINITY);
        Some(gapo_value(highest, lowest, self.log_length))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.length.saturating_sub(1)
    }
}

#[inline]
pub fn gopalakrishnan_range_index(
    input: &GopalakrishnanRangeIndexInput,
) -> Result<GopalakrishnanRangeIndexOutput, GopalakrishnanRangeIndexError> {
    gopalakrishnan_range_index_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_high_low_bar(high: f64, low: f64) -> bool {
    high.is_finite() && low.is_finite()
}

#[inline(always)]
fn gapo_value(highest: f64, lowest: f64, log_length: f64) -> f64 {
    let range = highest - lowest;
    if range.is_finite() && range > 0.0 {
        range.ln() / log_length
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn scan_valid_high_low(high: &[f64], low: &[f64]) -> (usize, usize) {
    let len = high.len();
    let mut first = len;
    let mut count = 0usize;
    for i in 0..high.len() {
        if valid_high_low_bar(high[i], low[i]) {
            if first == len {
                first = i;
            }
            count += 1;
        }
    }
    (first, count)
}

#[inline(always)]
fn build_prefix_valid(high: &[f64], low: &[f64]) -> Vec<u32> {
    let len = high.len();
    let mut prefix_valid = vec![0u32; len + 1];
    for i in 0..len {
        prefix_valid[i + 1] = prefix_valid[i]
            + if valid_high_low_bar(high[i], low[i]) {
                1
            } else {
                0
            };
    }
    prefix_valid
}

#[inline(always)]
fn gapo_row_from_slices(
    high: &[f64],
    low: &[f64],
    prefix_valid: &[u32],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    out.fill(f64::NAN);

    let warmup = first.saturating_add(length.saturating_sub(1));
    let length_u32 = length as u32;
    let log_length = (length as f64).ln();
    let mut dq_high = VecDeque::<usize>::with_capacity(length + 1);
    let mut dq_low = VecDeque::<usize>::with_capacity(length + 1);

    for i in first..high.len() {
        if valid_high_low_bar(high[i], low[i]) {
            while let Some(&j) = dq_high.back() {
                if high[j] <= high[i] {
                    dq_high.pop_back();
                } else {
                    break;
                }
            }
            dq_high.push_back(i);

            while let Some(&j) = dq_low.back() {
                if low[j] >= low[i] {
                    dq_low.pop_back();
                } else {
                    break;
                }
            }
            dq_low.push_back(i);
        }

        if i < warmup {
            continue;
        }

        let start = i + 1 - length;
        while let Some(&j) = dq_high.front() {
            if j < start {
                dq_high.pop_front();
            } else {
                break;
            }
        }
        while let Some(&j) = dq_low.front() {
            if j < start {
                dq_low.pop_front();
            } else {
                break;
            }
        }

        let valid_count = prefix_valid[i + 1] - prefix_valid[start];
        if valid_count != length_u32 {
            continue;
        }

        let highest = dq_high
            .front()
            .map(|&j| high[j])
            .unwrap_or(f64::NEG_INFINITY);
        let lowest = dq_low.front().map(|&j| low[j]).unwrap_or(f64::INFINITY);
        out[i] = gapo_value(highest, lowest, log_length);
    }
}

#[inline(always)]
fn gapo_row_len5_all_valid(high: &[f64], low: &[f64], out: &mut [f64]) {
    let len = high.len();
    let warmup = 4.min(len);
    out[..warmup].fill(f64::NAN);
    let log_length = 5.0f64.ln();

    for i in 4..len {
        let h0 = high[i - 4];
        let h1 = high[i - 3];
        let h2 = high[i - 2];
        let h3 = high[i - 1];
        let h4 = high[i];
        let l0 = low[i - 4];
        let l1 = low[i - 3];
        let l2 = low[i - 2];
        let l3 = low[i - 1];
        let l4 = low[i];
        let highest = h0.max(h1).max(h2).max(h3).max(h4);
        let lowest = l0.min(l1).min(l2).min(l3).min(l4);
        out[i] = gapo_value(highest, lowest, log_length);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn gapo_row_len5_all_valid_avx2(high: &[f64], low: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_max_pd, _mm256_min_pd, _mm256_storeu_pd};

    let len = high.len();
    out[..4.min(len)].fill(f64::NAN);
    let log_length = 5.0f64.ln();
    let mut i = 4usize;
    let mut ranges = [0.0; 4];

    while i + 3 < len {
        let base = i - 4;
        let h0 = _mm256_loadu_pd(high.as_ptr().add(base));
        let h1 = _mm256_loadu_pd(high.as_ptr().add(base + 1));
        let h2 = _mm256_loadu_pd(high.as_ptr().add(base + 2));
        let h3 = _mm256_loadu_pd(high.as_ptr().add(base + 3));
        let h4 = _mm256_loadu_pd(high.as_ptr().add(base + 4));
        let highest = _mm256_max_pd(
            _mm256_max_pd(_mm256_max_pd(h0, h1), _mm256_max_pd(h2, h3)),
            h4,
        );

        let l0 = _mm256_loadu_pd(low.as_ptr().add(base));
        let l1 = _mm256_loadu_pd(low.as_ptr().add(base + 1));
        let l2 = _mm256_loadu_pd(low.as_ptr().add(base + 2));
        let l3 = _mm256_loadu_pd(low.as_ptr().add(base + 3));
        let l4 = _mm256_loadu_pd(low.as_ptr().add(base + 4));
        let lowest = _mm256_min_pd(
            _mm256_min_pd(_mm256_min_pd(l0, l1), _mm256_min_pd(l2, l3)),
            l4,
        );

        let range = std::arch::x86_64::_mm256_sub_pd(highest, lowest);
        _mm256_storeu_pd(ranges.as_mut_ptr(), range);
        out[i] = gapo_value_from_range(ranges[0], log_length);
        out[i + 1] = gapo_value_from_range(ranges[1], log_length);
        out[i + 2] = gapo_value_from_range(ranges[2], log_length);
        out[i + 3] = gapo_value_from_range(ranges[3], log_length);
        i += 4;
    }

    while i < len {
        let h0 = high[i - 4];
        let h1 = high[i - 3];
        let h2 = high[i - 2];
        let h3 = high[i - 1];
        let h4 = high[i];
        let l0 = low[i - 4];
        let l1 = low[i - 3];
        let l2 = low[i - 2];
        let l3 = low[i - 1];
        let l4 = low[i];
        let highest = h0.max(h1).max(h2).max(h3).max(h4);
        let lowest = l0.min(l1).min(l2).min(l3).min(l4);
        out[i] = gapo_value(highest, lowest, log_length);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn gapo_row_len5_all_valid_avx512(high: &[f64], low: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::{_mm512_loadu_pd, _mm512_max_pd, _mm512_min_pd, _mm512_storeu_pd};

    let len = high.len();
    out[..4.min(len)].fill(f64::NAN);
    let log_length = 5.0f64.ln();
    let mut i = 4usize;
    let mut ranges = [0.0; 8];

    while i + 7 < len {
        let base = i - 4;
        let h0 = _mm512_loadu_pd(high.as_ptr().add(base));
        let h1 = _mm512_loadu_pd(high.as_ptr().add(base + 1));
        let h2 = _mm512_loadu_pd(high.as_ptr().add(base + 2));
        let h3 = _mm512_loadu_pd(high.as_ptr().add(base + 3));
        let h4 = _mm512_loadu_pd(high.as_ptr().add(base + 4));
        let highest = _mm512_max_pd(
            _mm512_max_pd(_mm512_max_pd(h0, h1), _mm512_max_pd(h2, h3)),
            h4,
        );

        let l0 = _mm512_loadu_pd(low.as_ptr().add(base));
        let l1 = _mm512_loadu_pd(low.as_ptr().add(base + 1));
        let l2 = _mm512_loadu_pd(low.as_ptr().add(base + 2));
        let l3 = _mm512_loadu_pd(low.as_ptr().add(base + 3));
        let l4 = _mm512_loadu_pd(low.as_ptr().add(base + 4));
        let lowest = _mm512_min_pd(
            _mm512_min_pd(_mm512_min_pd(l0, l1), _mm512_min_pd(l2, l3)),
            l4,
        );

        let range = std::arch::x86_64::_mm512_sub_pd(highest, lowest);
        _mm512_storeu_pd(ranges.as_mut_ptr(), range);
        out[i] = gapo_value_from_range(ranges[0], log_length);
        out[i + 1] = gapo_value_from_range(ranges[1], log_length);
        out[i + 2] = gapo_value_from_range(ranges[2], log_length);
        out[i + 3] = gapo_value_from_range(ranges[3], log_length);
        out[i + 4] = gapo_value_from_range(ranges[4], log_length);
        out[i + 5] = gapo_value_from_range(ranges[5], log_length);
        out[i + 6] = gapo_value_from_range(ranges[6], log_length);
        out[i + 7] = gapo_value_from_range(ranges[7], log_length);
        i += 8;
    }

    while i < len {
        let h0 = high[i - 4];
        let h1 = high[i - 3];
        let h2 = high[i - 2];
        let h3 = high[i - 1];
        let h4 = high[i];
        let l0 = low[i - 4];
        let l1 = low[i - 3];
        let l2 = low[i - 2];
        let l3 = low[i - 1];
        let l4 = low[i];
        let highest = h0.max(h1).max(h2).max(h3).max(h4);
        let lowest = l0.min(l1).min(l2).min(l3).min(l4);
        out[i] = gapo_value(highest, lowest, log_length);
        i += 1;
    }
}

#[inline(always)]
fn gapo_value_from_range(range: f64, log_length: f64) -> f64 {
    if range.is_finite() && range > 0.0 {
        range.ln() / log_length
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn gapo_row_len5_all_valid_kernel(high: &[f64], low: &[f64], kernel: Kernel, out: &mut [f64]) {
    let chosen = match kernel {
        Kernel::Auto => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    Kernel::Avx2
                } else if is_x86_feature_detected!("avx512f") {
                    Kernel::Avx512
                } else {
                    Kernel::Scalar
                }
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                Kernel::Scalar
            }
        }
        other => other.to_non_batch(),
    };
    match chosen {
        Kernel::Avx2 => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        gapo_row_len5_all_valid_avx2(high, low, out);
                    }
                    return;
                }
            }
            gapo_row_len5_all_valid(high, low, out);
        }
        Kernel::Avx512 => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    unsafe {
                        gapo_row_len5_all_valid_avx512(high, low, out);
                    }
                    return;
                }
            }
            gapo_row_len5_all_valid(high, low, out);
        }
        _ => gapo_row_len5_all_valid(high, low, out),
    }
}

#[inline(always)]
fn gopalakrishnan_range_index_prepare<'a>(
    input: &'a GopalakrishnanRangeIndexInput,
    _kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, bool), GopalakrishnanRangeIndexError> {
    let (high, low): (&[f64], &[f64]) = match &input.data {
        GopalakrishnanRangeIndexData::Candles { candles } => (&candles.high, &candles.low),
        GopalakrishnanRangeIndexData::Slices { high, low } => (high, low),
    };

    let len = high.len();
    if len == 0 {
        return Err(GopalakrishnanRangeIndexError::EmptyInputData);
    }
    if low.len() != len {
        return Err(GopalakrishnanRangeIndexError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let (first, valid) = scan_valid_high_low(high, low);
    if first >= len {
        return Err(GopalakrishnanRangeIndexError::AllValuesNaN);
    }

    let length = input.get_length();
    if length <= 1 || length > len {
        return Err(GopalakrishnanRangeIndexError::InvalidLength {
            length,
            data_len: len,
        });
    }

    if valid < length {
        return Err(GopalakrishnanRangeIndexError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    Ok((high, low, length, first, valid == len))
}

#[inline]
pub fn gopalakrishnan_range_index_with_kernel(
    input: &GopalakrishnanRangeIndexInput,
    kernel: Kernel,
) -> Result<GopalakrishnanRangeIndexOutput, GopalakrishnanRangeIndexError> {
    let (high, low, length, first, all_valid) = gopalakrishnan_range_index_prepare(input, kernel)?;
    let mut values = alloc_uninit_f64(high.len());
    if all_valid && length == 5 {
        gapo_row_len5_all_valid_kernel(high, low, kernel, &mut values);
    } else {
        let prefix_valid = build_prefix_valid(high, low);
        gapo_row_from_slices(high, low, &prefix_valid, length, first, &mut values);
    }
    Ok(GopalakrishnanRangeIndexOutput { values })
}

#[inline]
pub fn gopalakrishnan_range_index_into_slice(
    dst: &mut [f64],
    input: &GopalakrishnanRangeIndexInput,
    kernel: Kernel,
) -> Result<(), GopalakrishnanRangeIndexError> {
    let (high, low, length, first, all_valid) = gopalakrishnan_range_index_prepare(input, kernel)?;
    if dst.len() != high.len() {
        return Err(GopalakrishnanRangeIndexError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }
    if all_valid && length == 5 {
        gapo_row_len5_all_valid_kernel(high, low, kernel, dst);
    } else {
        let prefix_valid = build_prefix_valid(high, low);
        gapo_row_from_slices(high, low, &prefix_valid, length, first, dst);
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn gopalakrishnan_range_index_into(
    input: &GopalakrishnanRangeIndexInput,
    out: &mut [f64],
) -> Result<(), GopalakrishnanRangeIndexError> {
    gopalakrishnan_range_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct GopalakrishnanRangeIndexBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for GopalakrishnanRangeIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (5, 252, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GopalakrishnanRangeIndexBatchBuilder {
    range: GopalakrishnanRangeIndexBatchRange,
    kernel: Kernel,
}

impl GopalakrishnanRangeIndexBatchBuilder {
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
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexError> {
        gopalakrishnan_range_index_batch_with_kernel(high, low, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexError> {
        self.apply_slices(&candles.high, &candles.low)
    }

    #[inline]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexError> {
        GopalakrishnanRangeIndexBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles)
    }
}

#[derive(Clone, Debug)]
pub struct GopalakrishnanRangeIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GopalakrishnanRangeIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl GopalakrishnanRangeIndexBatchOutput {
    pub fn row_for_params(&self, params: &GopalakrishnanRangeIndexParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.length.unwrap_or(5) == params.length.unwrap_or(5))
    }

    pub fn values_for(&self, params: &GopalakrishnanRangeIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_grid_gopalakrishnan_range_index(
    range: &GopalakrishnanRangeIndexBatchRange,
) -> Result<Vec<GopalakrishnanRangeIndexParams>, GopalakrishnanRangeIndexError> {
    let (start, end, step) = range.length;
    let lengths = if step == 0 || start == end {
        vec![start]
    } else if start < end {
        let mut out = Vec::new();
        let mut x = start;
        let st = step.max(1);
        while x <= end {
            out.push(x);
            let next = x.saturating_add(st);
            if next == x {
                break;
            }
            x = next;
        }
        out
    } else {
        let mut out = Vec::new();
        let mut x = start;
        let st = step.max(1);
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(st);
            if next == x || next < end {
                break;
            }
            x = next;
        }
        out
    };

    if lengths.is_empty() {
        return Err(GopalakrishnanRangeIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if let Some(&bad) = lengths.iter().find(|&&length| length <= 1) {
        return Err(GopalakrishnanRangeIndexError::InvalidLength {
            length: bad,
            data_len: 0,
        });
    }

    Ok(lengths
        .into_iter()
        .map(|length| GopalakrishnanRangeIndexParams {
            length: Some(length),
        })
        .collect())
}

#[inline]
pub fn gopalakrishnan_range_index_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &GopalakrishnanRangeIndexBatchRange,
    kernel: Kernel,
) -> Result<GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(GopalakrishnanRangeIndexError::InvalidKernelForBatch(other)),
    };
    gopalakrishnan_range_index_batch_par_slice(high, low, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn gopalakrishnan_range_index_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &GopalakrishnanRangeIndexBatchRange,
    kernel: Kernel,
) -> Result<GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexError> {
    gopalakrishnan_range_index_batch_inner(high, low, sweep, kernel, false)
}

#[inline]
pub fn gopalakrishnan_range_index_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &GopalakrishnanRangeIndexBatchRange,
    kernel: Kernel,
) -> Result<GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexError> {
    gopalakrishnan_range_index_batch_inner(high, low, sweep, kernel, true)
}

#[inline(always)]
fn gopalakrishnan_range_index_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &GopalakrishnanRangeIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexError> {
    let combos = expand_grid_gopalakrishnan_range_index(sweep)?;
    let rows = combos.len();
    let cols = high.len();
    if cols == 0 {
        return Err(GopalakrishnanRangeIndexError::EmptyInputData);
    }
    if low.len() != cols {
        return Err(GopalakrishnanRangeIndexError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let (first, valid) = scan_valid_high_low(high, low);
    if first >= cols {
        return Err(GopalakrishnanRangeIndexError::AllValuesNaN);
    }

    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap_or(5))
        .max()
        .unwrap_or(0);
    if valid < max_length {
        return Err(GopalakrishnanRangeIndexError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first.saturating_add(combo.length.unwrap_or(5).saturating_sub(1)))
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let prefix_valid = build_prefix_valid(high, low);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let length = combos[row].length.unwrap_or(5);
                gapo_row_from_slices(high, low, &prefix_valid, length, first, out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let length = combos[row].length.unwrap_or(5);
            gapo_row_from_slices(high, low, &prefix_valid, length, first, out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let length = combos[row].length.unwrap_or(5);
            gapo_row_from_slices(high, low, &prefix_valid, length, first, out_row);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(GopalakrishnanRangeIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn gopalakrishnan_range_index_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &GopalakrishnanRangeIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<GopalakrishnanRangeIndexParams>, GopalakrishnanRangeIndexError> {
    let combos = expand_grid_gopalakrishnan_range_index(sweep)?;
    let rows = combos.len();
    let cols = high.len();
    if cols == 0 {
        return Err(GopalakrishnanRangeIndexError::EmptyInputData);
    }
    if low.len() != cols {
        return Err(GopalakrishnanRangeIndexError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let total = rows.checked_mul(cols).ok_or_else(|| {
        GopalakrishnanRangeIndexError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        }
    })?;
    if out.len() != total {
        return Err(GopalakrishnanRangeIndexError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }

    let (first, valid) = scan_valid_high_low(high, low);
    if first >= cols {
        return Err(GopalakrishnanRangeIndexError::AllValuesNaN);
    }

    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap_or(5))
        .max()
        .unwrap_or(0);
    if valid < max_length {
        return Err(GopalakrishnanRangeIndexError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let prefix_valid = build_prefix_valid(high, low);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let length = combos[row].length.unwrap_or(5);
                gapo_row_from_slices(high, low, &prefix_valid, length, first, out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let length = combos[row].length.unwrap_or(5);
            gapo_row_from_slices(high, low, &prefix_valid, length, first, out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let length = combos[row].length.unwrap_or(5);
            gapo_row_from_slices(high, low, &prefix_valid, length, first, out_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "gopalakrishnan_range_index")]
#[pyo3(signature = (high, low, length=5, kernel=None))]
pub fn gopalakrishnan_range_index_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    if high.len() != low.len() {
        return Err(PyValueError::new_err("High/low slice length mismatch"));
    }

    let kernel = validate_kernel(kernel, false)?;
    let input = GopalakrishnanRangeIndexInput::from_slices(
        high,
        low,
        GopalakrishnanRangeIndexParams {
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| gopalakrishnan_range_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "GopalakrishnanRangeIndexStream")]
pub struct GopalakrishnanRangeIndexStreamPy {
    stream: GopalakrishnanRangeIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GopalakrishnanRangeIndexStreamPy {
    #[new]
    fn new(length: usize) -> PyResult<Self> {
        let stream = GopalakrishnanRangeIndexStream::try_new(GopalakrishnanRangeIndexParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "gopalakrishnan_range_index_batch")]
#[pyo3(signature = (high, low, length_range, kernel=None))]
pub fn gopalakrishnan_range_index_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    if high.len() != low.len() {
        return Err(PyValueError::new_err("High/low slice length mismatch"));
    }

    let kernel = validate_kernel(kernel, true)?;
    let sweep = GopalakrishnanRangeIndexBatchRange {
        length: length_range,
    };
    let combos = expand_grid_gopalakrishnan_range_index(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            gopalakrishnan_range_index_batch_inner_into(
                high,
                low,
                &sweep,
                batch.to_non_batch(),
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_gopalakrishnan_range_index_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gopalakrishnan_range_index_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        gopalakrishnan_range_index_batch_py,
        module
    )?)?;
    module.add_class::<GopalakrishnanRangeIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "gopalakrishnan_range_index_js")]
pub fn gopalakrishnan_range_index_js(
    high: &[f64],
    low: &[f64],
    length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = GopalakrishnanRangeIndexInput::from_slices(
        high,
        low,
        GopalakrishnanRangeIndexParams {
            length: Some(length),
        },
    );
    let mut output = vec![0.0; high.len()];
    gopalakrishnan_range_index_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gopalakrishnan_range_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gopalakrishnan_range_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gopalakrishnan_range_index_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let input = GopalakrishnanRangeIndexInput::from_slices(
            high,
            low,
            GopalakrishnanRangeIndexParams {
                length: Some(length),
            },
        );
        if high_ptr == out_ptr || low_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            gopalakrishnan_range_index_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            gopalakrishnan_range_index_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GopalakrishnanRangeIndexBatchConfig {
    pub length_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GopalakrishnanRangeIndexBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GopalakrishnanRangeIndexParams>,
    pub lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "gopalakrishnan_range_index_batch_js")]
pub fn gopalakrishnan_range_index_batch_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: GopalakrishnanRangeIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = GopalakrishnanRangeIndexBatchRange {
        length: config.length_range,
    };
    let output =
        gopalakrishnan_range_index_batch_inner(high, low, &sweep, detect_best_kernel(), false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&GopalakrishnanRangeIndexBatchJsOutput {
        lengths: output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(5))
            .collect(),
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gopalakrishnan_range_index_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = GopalakrishnanRangeIndexBatchRange {
        length: (length_start, length_end, length_step),
    };
    let combos = expand_grid_gopalakrishnan_range_index(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        gopalakrishnan_range_index_batch_inner_into(
            high,
            low,
            &sweep,
            detect_best_kernel(),
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gopalakrishnan_range_index_output_into_js(
    high: &[f64],
    low: &[f64],
    length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = gopalakrishnan_range_index_js(high, low, length)?;
    crate::write_wasm_f64_output("gopalakrishnan_range_index_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn gopalakrishnan_range_index_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = gopalakrishnan_range_index_batch_js(high, low, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "gopalakrishnan_range_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn load_high_low() -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok((candles.high, candles.low))
    }

    #[test]
    fn gopalakrishnan_range_index_output_contract() -> Result<(), Box<dyn Error>> {
        let (high, low) = load_high_low()?;
        let input = GopalakrishnanRangeIndexInput::from_slices(
            &high,
            &low,
            GopalakrishnanRangeIndexParams { length: Some(5) },
        );
        let out = gopalakrishnan_range_index_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.values.len(), high.len());
        let first_valid = out.values.iter().position(|v| !v.is_nan()).unwrap();
        assert!(first_valid >= 4);
        assert!(out.values[first_valid..].iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn gopalakrishnan_range_index_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let (high, low) = load_high_low()?;
        let input = GopalakrishnanRangeIndexInput::from_slices(
            &high,
            &low,
            GopalakrishnanRangeIndexParams { length: Some(9) },
        );
        let auto = gopalakrishnan_range_index_with_kernel(&input, Kernel::Auto)?;
        let scalar = gopalakrishnan_range_index_with_kernel(&input, Kernel::Scalar)?;
        for (a, b) in auto.values.iter().zip(scalar.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn gopalakrishnan_range_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (high, low) = load_high_low()?;
        let input = GopalakrishnanRangeIndexInput::from_slices(
            &high,
            &low,
            GopalakrishnanRangeIndexParams { length: Some(7) },
        );
        let api = gopalakrishnan_range_index_with_kernel(&input, Kernel::Auto)?;
        let mut out = vec![0.0; high.len()];
        gopalakrishnan_range_index_into(&input, &mut out)?;
        for (a, b) in api.values.iter().zip(out.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() <= 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn gopalakrishnan_range_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (high, low) = load_high_low()?;
        let params = GopalakrishnanRangeIndexParams { length: Some(8) };
        let input = GopalakrishnanRangeIndexInput::from_slices(&high, &low, params.clone());
        let batch = gopalakrishnan_range_index_with_kernel(&input, Kernel::Scalar)?;
        let mut stream = GopalakrishnanRangeIndexStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(high.len());
        for i in 0..high.len() {
            streamed.push(stream.update(high[i], low[i]).unwrap_or(f64::NAN));
        }
        for (a, b) in streamed.iter().zip(batch.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn gopalakrishnan_range_index_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let (high, low) = load_high_low()?;
        let single = gopalakrishnan_range_index_with_kernel(
            &GopalakrishnanRangeIndexInput::from_slices(
                &high,
                &low,
                GopalakrishnanRangeIndexParams { length: Some(10) },
            ),
            Kernel::Scalar,
        )?;
        let batch = gopalakrishnan_range_index_batch_with_kernel(
            &high,
            &low,
            &GopalakrishnanRangeIndexBatchRange {
                length: (10, 10, 0),
            },
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, high.len());
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn gopalakrishnan_range_index_invalid_window_recovers() -> Result<(), Box<dyn Error>> {
        let (mut high, mut low) = load_high_low()?;
        high.truncate(80);
        low.truncate(80);
        high[30] = f64::NAN;
        low[30] = f64::NAN;
        let out = gopalakrishnan_range_index_with_kernel(
            &GopalakrishnanRangeIndexInput::from_slices(
                &high,
                &low,
                GopalakrishnanRangeIndexParams { length: Some(10) },
            ),
            Kernel::Scalar,
        )?;
        assert!(out.values[30].is_nan());
        assert!(out.values[39].is_nan());
        assert!(out.values[40].is_finite());
        Ok(())
    }

    #[test]
    fn gopalakrishnan_range_index_rejects_invalid_length() {
        let high = [1.0, 2.0, 3.0];
        let low = [0.5, 1.5, 2.5];
        let err = gopalakrishnan_range_index_with_kernel(
            &GopalakrishnanRangeIndexInput::from_slices(
                &high,
                &low,
                GopalakrishnanRangeIndexParams { length: Some(1) },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            GopalakrishnanRangeIndexError::InvalidLength { .. }
        ));
    }
}
