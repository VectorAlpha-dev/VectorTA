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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ZigZagChannelsData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ZigZagChannelsOutput {
    pub middle: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ZigZagChannelsParams {
    pub length: Option<usize>,
    pub extend: Option<bool>,
}

impl Default for ZigZagChannelsParams {
    fn default() -> Self {
        Self {
            length: Some(100),
            extend: Some(true),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZigZagChannelsInput<'a> {
    pub data: ZigZagChannelsData<'a>,
    pub params: ZigZagChannelsParams,
}

impl<'a> ZigZagChannelsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ZigZagChannelsParams) -> Self {
        Self {
            data: ZigZagChannelsData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: ZigZagChannelsParams,
    ) -> Self {
        Self {
            data: ZigZagChannelsData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, ZigZagChannelsParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(100)
    }

    #[inline]
    pub fn get_extend(&self) -> bool {
        self.params.extend.unwrap_or(true)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ZigZagChannelsBuilder {
    length: Option<usize>,
    extend: Option<bool>,
    kernel: Kernel,
}

impl Default for ZigZagChannelsBuilder {
    fn default() -> Self {
        Self {
            length: None,
            extend: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ZigZagChannelsBuilder {
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
    pub fn extend(mut self, value: bool) -> Self {
        self.extend = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<ZigZagChannelsOutput, ZigZagChannelsError> {
        zig_zag_channels_with_kernel(
            &ZigZagChannelsInput::from_candles(
                candles,
                ZigZagChannelsParams {
                    length: self.length,
                    extend: self.extend,
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ZigZagChannelsOutput, ZigZagChannelsError> {
        zig_zag_channels_with_kernel(
            &ZigZagChannelsInput::from_slices(
                open,
                high,
                low,
                close,
                ZigZagChannelsParams {
                    length: self.length,
                    extend: self.extend,
                },
            ),
            self.kernel,
        )
    }
}

#[derive(Debug, Error)]
pub enum ZigZagChannelsError {
    #[error("zig_zag_channels: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "zig_zag_channels: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("zig_zag_channels: All values are NaN.")]
    AllValuesNaN,
    #[error("zig_zag_channels: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("zig_zag_channels: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("zig_zag_channels: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("zig_zag_channels: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("zig_zag_channels: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "zig_zag_channels: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("zig_zag_channels: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy)]
struct PivotState {
    confirm_idx: usize,
    value: f64,
}

#[inline(always)]
fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for (((&o, &h), &l), &c) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if is_valid_ohlc(o, h, l, c) {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a ZigZagChannelsInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), ZigZagChannelsError> {
    match &input.data {
        ZigZagChannelsData::Candles { candles } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        ZigZagChannelsData::Slices {
            open,
            high,
            low,
            close,
        } => Ok((open, high, low, close)),
    }
}

#[inline(always)]
fn validate_common(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
) -> Result<(), ZigZagChannelsError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ZigZagChannelsError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(ZigZagChannelsError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    if length == 0 {
        return Err(ZigZagChannelsError::InvalidLength { length });
    }

    let longest = longest_valid_run(open, high, low, close);
    if longest == 0 {
        return Err(ZigZagChannelsError::AllValuesNaN);
    }

    let needed = length
        .checked_add(1)
        .ok_or_else(|| ZigZagChannelsError::InvalidInput {
            msg: "zig_zag_channels: length overflow".to_string(),
        })?;
    if longest < needed {
        return Err(ZigZagChannelsError::NotEnoughValidData {
            needed,
            valid: longest,
        });
    }
    Ok(())
}

#[inline(always)]
fn compute_segment_offsets(
    open: &[f64],
    close: &[f64],
    start_idx: usize,
    end_idx: usize,
    start_value: f64,
    end_value: f64,
) -> (f64, f64) {
    if end_idx <= start_idx {
        return (0.0, 0.0);
    }

    if end_idx == start_idx + 1 {
        let top = open[end_idx].max(close[end_idx]);
        let bottom = open[end_idx].min(close[end_idx]);
        return ((top - end_value).max(0.0), (end_value - bottom).max(0.0));
    }

    let mut max_diff_up = 0.0f64;
    let mut max_diff_dn = 0.0f64;
    let denom = (end_idx - start_idx - 1) as f64;
    let span = end_value - start_value;

    for idx in (start_idx + 1)..=end_idx {
        let j = (idx - start_idx - 1) as f64;
        let point = start_value + (j / denom) * span;
        let top = open[idx].max(close[idx]);
        let bottom = open[idx].min(close[idx]);
        max_diff_up = max_diff_up.max(top - point);
        max_diff_dn = max_diff_dn.max(point - bottom);
    }

    (max_diff_up.max(0.0), max_diff_dn.max(0.0))
}

#[inline(always)]
fn fill_segment(
    middle: &mut [f64],
    upper: &mut [f64],
    lower: &mut [f64],
    start_idx: usize,
    end_idx: usize,
    start_value: f64,
    end_value: f64,
    up_offset: f64,
    dn_offset: f64,
) {
    if end_idx < start_idx {
        return;
    }

    if start_idx == end_idx {
        middle[start_idx] = start_value;
        upper[start_idx] = start_value + up_offset;
        lower[start_idx] = start_value - dn_offset;
        return;
    }

    let denom = (end_idx - start_idx) as f64;
    let span = end_value - start_value;
    for idx in start_idx..=end_idx {
        let t = (idx - start_idx) as f64 / denom;
        let value = start_value + t * span;
        middle[idx] = value;
        upper[idx] = value + up_offset;
        lower[idx] = value - dn_offset;
    }
}

fn compute_run(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    extend: bool,
    middle: &mut [f64],
    upper: &mut [f64],
    lower: &mut [f64],
) {
    let n = close.len();
    if n <= length {
        return;
    }

    let mut max_deque: VecDeque<usize> = VecDeque::with_capacity(length);
    let mut min_deque: VecDeque<usize> = VecDeque::with_capacity(length);
    let mut os = 0usize;
    let mut last_top: Option<PivotState> = None;
    let mut last_bottom: Option<PivotState> = None;

    for idx in 0..n {
        let current_close = close[idx];
        while let Some(&back) = max_deque.back() {
            if close[back] <= current_close {
                max_deque.pop_back();
            } else {
                break;
            }
        }
        max_deque.push_back(idx);

        while let Some(&back) = min_deque.back() {
            if close[back] >= current_close {
                min_deque.pop_back();
            } else {
                break;
            }
        }
        min_deque.push_back(idx);

        if idx < length {
            continue;
        }

        let window_start = idx + 1 - length;
        while let Some(&front) = max_deque.front() {
            if front < window_start {
                max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some(&front) = min_deque.front() {
            if front < window_start {
                min_deque.pop_front();
            } else {
                break;
            }
        }

        let candidate = idx - length;
        let upper_close = close[*max_deque.front().expect("window max present")];
        let lower_close = close[*min_deque.front().expect("window min present")];
        let prev_os = os;
        let candidate_close = close[candidate];

        if candidate_close > upper_close {
            os = 0;
        } else if candidate_close < lower_close {
            os = 1;
        }

        if os == 1 && prev_os != 1 {
            let end_idx = candidate;
            let end_value = low[end_idx];
            if let Some(prev_top) = last_top {
                let start_idx = prev_top.confirm_idx - length;
                let start_value = prev_top.value;
                let (up_offset, dn_offset) = compute_segment_offsets(
                    open,
                    close,
                    start_idx,
                    end_idx,
                    start_value,
                    end_value,
                );
                fill_segment(
                    middle,
                    upper,
                    lower,
                    start_idx,
                    end_idx,
                    start_value,
                    end_value,
                    up_offset,
                    dn_offset,
                );
            }
            last_bottom = Some(PivotState {
                confirm_idx: idx,
                value: end_value,
            });
        }

        if os == 0 && prev_os != 0 {
            let end_idx = candidate;
            let end_value = high[end_idx];
            if let Some(prev_bottom) = last_bottom {
                let start_idx = prev_bottom.confirm_idx - length;
                let start_value = prev_bottom.value;
                let (up_offset, dn_offset) = compute_segment_offsets(
                    open,
                    close,
                    start_idx,
                    end_idx,
                    start_value,
                    end_value,
                );
                fill_segment(
                    middle,
                    upper,
                    lower,
                    start_idx,
                    end_idx,
                    start_value,
                    end_value,
                    up_offset,
                    dn_offset,
                );
            }
            last_top = Some(PivotState {
                confirm_idx: idx,
                value: end_value,
            });
        }
    }

    if !extend {
        return;
    }

    let end_idx = n - 1;
    let end_value = close[end_idx];
    if os == 1 {
        if let Some(prev_bottom) = last_bottom {
            let start_idx = prev_bottom.confirm_idx - length;
            let start_value = prev_bottom.value;
            let (up_offset, dn_offset) =
                compute_segment_offsets(open, close, start_idx, end_idx, start_value, end_value);
            fill_segment(
                middle,
                upper,
                lower,
                start_idx,
                end_idx,
                start_value,
                end_value,
                up_offset,
                dn_offset,
            );
        }
    } else if let Some(prev_top) = last_top {
        let start_idx = prev_top.confirm_idx - length;
        let start_value = prev_top.value;
        let (up_offset, dn_offset) =
            compute_segment_offsets(open, close, start_idx, end_idx, start_value, end_value);
        fill_segment(
            middle,
            upper,
            lower,
            start_idx,
            end_idx,
            start_value,
            end_value,
            up_offset,
            dn_offset,
        );
    }
}

fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    extend: bool,
    middle: &mut [f64],
    upper: &mut [f64],
    lower: &mut [f64],
) {
    let mut idx = 0usize;
    while idx < close.len() {
        while idx < close.len() && !is_valid_ohlc(open[idx], high[idx], low[idx], close[idx]) {
            idx += 1;
        }
        if idx >= close.len() {
            break;
        }
        let seg_start = idx;
        idx += 1;
        while idx < close.len() && is_valid_ohlc(open[idx], high[idx], low[idx], close[idx]) {
            idx += 1;
        }
        let seg_end = idx;
        if seg_end - seg_start >= length + 1 {
            compute_run(
                &open[seg_start..seg_end],
                &high[seg_start..seg_end],
                &low[seg_start..seg_end],
                &close[seg_start..seg_end],
                length,
                extend,
                &mut middle[seg_start..seg_end],
                &mut upper[seg_start..seg_end],
                &mut lower[seg_start..seg_end],
            );
        }
    }
}

#[inline]
pub fn zig_zag_channels(
    input: &ZigZagChannelsInput,
) -> Result<ZigZagChannelsOutput, ZigZagChannelsError> {
    zig_zag_channels_with_kernel(input, Kernel::Auto)
}

pub fn zig_zag_channels_with_kernel(
    input: &ZigZagChannelsInput,
    kernel: Kernel,
) -> Result<ZigZagChannelsOutput, ZigZagChannelsError> {
    let (open, high, low, close) = input_slices(input)?;
    let length = input.get_length();
    let extend = input.get_extend();
    validate_common(open, high, low, close, length)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut middle = alloc_with_nan_prefix(close.len(), 0);
    let mut upper = alloc_with_nan_prefix(close.len(), 0);
    let mut lower = alloc_with_nan_prefix(close.len(), 0);
    middle.fill(f64::NAN);
    upper.fill(f64::NAN);
    lower.fill(f64::NAN);

    compute_row(
        open,
        high,
        low,
        close,
        length,
        extend,
        &mut middle,
        &mut upper,
        &mut lower,
    );

    Ok(ZigZagChannelsOutput {
        middle,
        upper,
        lower,
    })
}

pub fn zig_zag_channels_into_slice(
    out_middle: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    input: &ZigZagChannelsInput,
    kernel: Kernel,
) -> Result<(), ZigZagChannelsError> {
    let (open, high, low, close) = input_slices(input)?;
    let length = input.get_length();
    let extend = input.get_extend();
    validate_common(open, high, low, close, length)?;

    if out_middle.len() != close.len() {
        return Err(ZigZagChannelsError::OutputLengthMismatch {
            expected: close.len(),
            got: out_middle.len(),
        });
    }
    if out_upper.len() != close.len() {
        return Err(ZigZagChannelsError::OutputLengthMismatch {
            expected: close.len(),
            got: out_upper.len(),
        });
    }
    if out_lower.len() != close.len() {
        return Err(ZigZagChannelsError::OutputLengthMismatch {
            expected: close.len(),
            got: out_lower.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    out_middle.fill(f64::NAN);
    out_upper.fill(f64::NAN);
    out_lower.fill(f64::NAN);
    compute_row(
        open, high, low, close, length, extend, out_middle, out_upper, out_lower,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn zig_zag_channels_into(
    input: &ZigZagChannelsInput,
    out_middle: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), ZigZagChannelsError> {
    zig_zag_channels_into_slice(out_middle, out_upper, out_lower, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct ZigZagChannelsBatchRange {
    pub length: (usize, usize, usize),
    pub extend: bool,
}

impl Default for ZigZagChannelsBatchRange {
    fn default() -> Self {
        Self {
            length: (100, 100, 0),
            extend: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZigZagChannelsBatchOutput {
    pub middle: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<ZigZagChannelsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ZigZagChannelsBatchBuilder {
    range: ZigZagChannelsBatchRange,
    kernel: Kernel,
}

impl Default for ZigZagChannelsBatchBuilder {
    fn default() -> Self {
        Self {
            range: ZigZagChannelsBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl ZigZagChannelsBatchBuilder {
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
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn extend(mut self, value: bool) -> Self {
        self.range.extend = value;
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ZigZagChannelsBatchOutput, ZigZagChannelsError> {
        zig_zag_channels_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<ZigZagChannelsBatchOutput, ZigZagChannelsError> {
        zig_zag_channels_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &ZigZagChannelsBatchRange,
) -> Result<Vec<ZigZagChannelsParams>, ZigZagChannelsError> {
    let (start, end, step) = range.length;
    if start == 0 || end == 0 {
        return Err(ZigZagChannelsError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![ZigZagChannelsParams {
            length: Some(start),
            extend: Some(range.extend),
        }]);
    }
    if start > end {
        return Err(ZigZagChannelsError::InvalidRange { start, end, step });
    }

    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(ZigZagChannelsParams {
            length: Some(current),
            extend: Some(range.extend),
        });
        if current >= end {
            break;
        }
        let next = current.saturating_add(step);
        if next <= current {
            return Err(ZigZagChannelsError::InvalidRange { start, end, step });
        }
        current = next.min(end);
        if current == out.last().and_then(|item| item.length).unwrap_or(0) {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_zig_zag_channels(range: &ZigZagChannelsBatchRange) -> Vec<ZigZagChannelsParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn zig_zag_channels_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ZigZagChannelsBatchRange,
    kernel: Kernel,
) -> Result<ZigZagChannelsBatchOutput, ZigZagChannelsError> {
    zig_zag_channels_batch_inner(open, high, low, close, sweep, kernel, true)
}

pub fn zig_zag_channels_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ZigZagChannelsBatchRange,
    kernel: Kernel,
) -> Result<ZigZagChannelsBatchOutput, ZigZagChannelsError> {
    zig_zag_channels_batch_inner(open, high, low, close, sweep, kernel, false)
}

pub fn zig_zag_channels_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ZigZagChannelsBatchRange,
    kernel: Kernel,
) -> Result<ZigZagChannelsBatchOutput, ZigZagChannelsError> {
    zig_zag_channels_batch_inner(open, high, low, close, sweep, kernel, true)
}

fn zig_zag_channels_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ZigZagChannelsBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<ZigZagChannelsBatchOutput, ZigZagChannelsError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(ZigZagChannelsError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(100))
        .max()
        .unwrap_or(0);
    validate_common(open, high, low, close, max_length)?;

    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| ZigZagChannelsError::InvalidInput {
            msg: "zig_zag_channels: rows*cols overflow in batch".to_string(),
        })?;

    let mut middle = vec![f64::NAN; total];
    let mut upper = vec![f64::NAN; total];
    let mut lower = vec![f64::NAN; total];
    zig_zag_channels_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        kernel,
        parallel,
        &mut middle,
        &mut upper,
        &mut lower,
    )?;

    Ok(ZigZagChannelsBatchOutput {
        middle,
        upper,
        lower,
        combos,
        rows,
        cols,
    })
}

fn zig_zag_channels_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ZigZagChannelsBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_middle: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<Vec<ZigZagChannelsParams>, ZigZagChannelsError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(ZigZagChannelsError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(100))
        .max()
        .unwrap_or(0);
    validate_common(open, high, low, close, max_length)?;

    let cols = close.len();
    let total =
        combos
            .len()
            .checked_mul(cols)
            .ok_or_else(|| ZigZagChannelsError::InvalidInput {
                msg: "zig_zag_channels: rows*cols overflow in batch_into".to_string(),
            })?;
    if out_middle.len() != total {
        return Err(ZigZagChannelsError::MismatchedOutputLen {
            dst_len: out_middle.len(),
            expected_len: total,
        });
    }
    if out_upper.len() != total {
        return Err(ZigZagChannelsError::MismatchedOutputLen {
            dst_len: out_upper.len(),
            expected_len: total,
        });
    }
    if out_lower.len() != total {
        return Err(ZigZagChannelsError::MismatchedOutputLen {
            dst_len: out_lower.len(),
            expected_len: total,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker =
        |row: usize, middle_row: &mut [f64], upper_row: &mut [f64], lower_row: &mut [f64]| {
            middle_row.fill(f64::NAN);
            upper_row.fill(f64::NAN);
            lower_row.fill(f64::NAN);
            let params = &combos[row];
            compute_row(
                open,
                high,
                low,
                close,
                params.length.unwrap_or(100),
                params.extend.unwrap_or(true),
                middle_row,
                upper_row,
                lower_row,
            );
        };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && combos.len() > 1 {
        out_middle
            .par_chunks_mut(cols)
            .zip(out_upper.par_chunks_mut(cols))
            .zip(out_lower.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((middle_row, upper_row), lower_row))| {
                worker(row, middle_row, upper_row, lower_row);
            });
    } else {
        for (row, ((middle_row, upper_row), lower_row)) in out_middle
            .chunks_mut(cols)
            .zip(out_upper.chunks_mut(cols))
            .zip(out_lower.chunks_mut(cols))
            .enumerate()
        {
            worker(row, middle_row, upper_row, lower_row);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, ((middle_row, upper_row), lower_row)) in out_middle
            .chunks_mut(cols)
            .zip(out_upper.chunks_mut(cols))
            .zip(out_lower.chunks_mut(cols))
            .enumerate()
        {
            worker(row, middle_row, upper_row, lower_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "zig_zag_channels", signature = (open, high, low, close, length=100, extend=true, kernel=None))]
pub fn zig_zag_channels_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    extend: bool,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = ZigZagChannelsInput::from_slices(
        open,
        high,
        low,
        close,
        ZigZagChannelsParams {
            length: Some(length),
            extend: Some(extend),
        },
    );
    let out = py
        .allow_threads(|| zig_zag_channels_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.middle.into_pyarray(py),
        out.upper.into_pyarray(py),
        out.lower.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "zig_zag_channels_batch", signature = (open, high, low, close, length_range=(100, 100, 0), extend=true, kernel=None))]
pub fn zig_zag_channels_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    extend: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| {
            zig_zag_channels_batch_with_kernel(
                open,
                high,
                low,
                close,
                &ZigZagChannelsBatchRange {
                    length: length_range,
                    extend,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "middle",
        output
            .middle
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "upper",
        output
            .upper
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lower",
        output
            .lower
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(100) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "extends",
        output
            .combos
            .iter()
            .map(|params| params.extend.unwrap_or(true))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_zig_zag_channels_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zig_zag_channels_py, m)?)?;
    m.add_function(wrap_pyfunction!(zig_zag_channels_batch_py, m)?)?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZigZagChannelsBatchConfig {
    pub length_range: Vec<usize>,
    pub extend: Option<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = zig_zag_channels_js)]
pub fn zig_zag_channels_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    extend: bool,
) -> Result<JsValue, JsValue> {
    let input = ZigZagChannelsInput::from_slices(
        open,
        high,
        low,
        close,
        ZigZagChannelsParams {
            length: Some(length),
            extend: Some(extend),
        },
    );
    let out = zig_zag_channels_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("middle"),
        &serde_wasm_bindgen::to_value(&out.middle).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("upper"),
        &serde_wasm_bindgen::to_value(&out.upper).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lower"),
        &serde_wasm_bindgen::to_value(&out.lower).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = zig_zag_channels_batch_js)]
pub fn zig_zag_channels_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: ZigZagChannelsBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = zig_zag_channels_batch_with_kernel(
        open,
        high,
        low,
        close,
        &ZigZagChannelsBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            extend: config.extend.unwrap_or(true),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("middle"),
        &serde_wasm_bindgen::to_value(&out.middle).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("upper"),
        &serde_wasm_bindgen::to_value(&out.upper).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lower"),
        &serde_wasm_bindgen::to_value(&out.lower).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn zig_zag_channels_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(3 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn zig_zag_channels_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 3 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn zig_zag_channels_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    extend: bool,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to zig_zag_channels_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 3 * len);
        let (middle, tail) = out.split_at_mut(len);
        let (upper, lower) = tail.split_at_mut(len);
        let input = ZigZagChannelsInput::from_slices(
            open,
            high,
            low,
            close,
            ZigZagChannelsParams {
                length: Some(length),
                extend: Some(extend),
            },
        );
        zig_zag_channels_into_slice(middle, upper, lower, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn zig_zag_channels_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    extend: bool,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to zig_zag_channels_batch_into",
        ));
    }

    let sweep = ZigZagChannelsBatchRange {
        length: (length_start, length_end, length_step),
        extend,
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let split = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in zig_zag_channels_batch_into"))?;
    let total = split
        .checked_mul(3)
        .ok_or_else(|| JsValue::from_str("3*rows*cols overflow in zig_zag_channels_batch_into"))?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (middle, tail) = out.split_at_mut(split);
        let (upper, lower) = tail.split_at_mut(split);
        zig_zag_channels_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
            false,
            middle,
            upper,
            lower,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn zig_zag_channels_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    extend: bool,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = zig_zag_channels_js(open, high, low, close, length, extend)?;
    crate::write_wasm_object_f64_outputs("zig_zag_channels_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn zig_zag_channels_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = zig_zag_channels_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "zig_zag_channels_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, ParamKV, ParamValue,
    };

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + (x * 0.21).sin() * 7.0 + (x * 0.037).cos() * 2.5 + (x * 0.02)
            })
            .collect();
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c + ((i as f64) * 0.41).sin() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.max(c) + 0.75)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.min(c) - 0.75)
            .collect();
        (open, high, low, close)
    }

    fn naive_offsets(
        open: &[f64],
        close: &[f64],
        start_idx: usize,
        end_idx: usize,
        start_value: f64,
        end_value: f64,
    ) -> (f64, f64) {
        if end_idx <= start_idx {
            return (0.0, 0.0);
        }
        if end_idx == start_idx + 1 {
            let top = open[end_idx].max(close[end_idx]);
            let bottom = open[end_idx].min(close[end_idx]);
            return ((top - end_value).max(0.0), (end_value - bottom).max(0.0));
        }
        let mut up = 0.0f64;
        let mut dn = 0.0f64;
        let denom = (end_idx - start_idx - 1) as f64;
        for idx in (start_idx + 1)..=end_idx {
            let j = (idx - start_idx - 1) as f64;
            let point = start_value + (j / denom) * (end_value - start_value);
            up = up.max(open[idx].max(close[idx]) - point);
            dn = dn.max(point - open[idx].min(close[idx]));
        }
        (up.max(0.0), dn.max(0.0))
    }

    fn naive_fill(
        middle: &mut [f64],
        upper: &mut [f64],
        lower: &mut [f64],
        start_idx: usize,
        end_idx: usize,
        start_value: f64,
        end_value: f64,
        up: f64,
        dn: f64,
    ) {
        if end_idx < start_idx {
            return;
        }
        if start_idx == end_idx {
            middle[start_idx] = start_value;
            upper[start_idx] = start_value + up;
            lower[start_idx] = start_value - dn;
            return;
        }
        let denom = (end_idx - start_idx) as f64;
        for idx in start_idx..=end_idx {
            let t = (idx - start_idx) as f64 / denom;
            let value = start_value + t * (end_value - start_value);
            middle[idx] = value;
            upper[idx] = value + up;
            lower[idx] = value - dn;
        }
    }

    fn naive_run(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        length: usize,
        extend: bool,
        middle: &mut [f64],
        upper: &mut [f64],
        lower: &mut [f64],
    ) {
        let mut os = 0usize;
        let mut last_top: Option<(usize, f64)> = None;
        let mut last_bottom: Option<(usize, f64)> = None;

        for current in length..close.len() {
            let candidate = current - length;
            let mut hi = f64::NEG_INFINITY;
            let mut lo = f64::INFINITY;
            for idx in (candidate + 1)..=current {
                hi = hi.max(close[idx]);
                lo = lo.min(close[idx]);
            }

            let prev_os = os;
            if close[candidate] > hi {
                os = 0;
            } else if close[candidate] < lo {
                os = 1;
            }

            if os == 1 && prev_os != 1 {
                let end_idx = candidate;
                let end_value = low[end_idx];
                if let Some((confirm_idx, start_value)) = last_top {
                    let start_idx = confirm_idx - length;
                    let (up, dn) =
                        naive_offsets(open, close, start_idx, end_idx, start_value, end_value);
                    naive_fill(
                        middle,
                        upper,
                        lower,
                        start_idx,
                        end_idx,
                        start_value,
                        end_value,
                        up,
                        dn,
                    );
                }
                last_bottom = Some((current, end_value));
            }

            if os == 0 && prev_os != 0 {
                let end_idx = candidate;
                let end_value = high[end_idx];
                if let Some((confirm_idx, start_value)) = last_bottom {
                    let start_idx = confirm_idx - length;
                    let (up, dn) =
                        naive_offsets(open, close, start_idx, end_idx, start_value, end_value);
                    naive_fill(
                        middle,
                        upper,
                        lower,
                        start_idx,
                        end_idx,
                        start_value,
                        end_value,
                        up,
                        dn,
                    );
                }
                last_top = Some((current, end_value));
            }
        }

        if !extend || close.is_empty() {
            return;
        }
        let end_idx = close.len() - 1;
        let end_value = close[end_idx];
        if os == 1 {
            if let Some((confirm_idx, start_value)) = last_bottom {
                let start_idx = confirm_idx - length;
                let (up, dn) =
                    naive_offsets(open, close, start_idx, end_idx, start_value, end_value);
                naive_fill(
                    middle,
                    upper,
                    lower,
                    start_idx,
                    end_idx,
                    start_value,
                    end_value,
                    up,
                    dn,
                );
            }
        } else if let Some((confirm_idx, start_value)) = last_top {
            let start_idx = confirm_idx - length;
            let (up, dn) = naive_offsets(open, close, start_idx, end_idx, start_value, end_value);
            naive_fill(
                middle,
                upper,
                lower,
                start_idx,
                end_idx,
                start_value,
                end_value,
                up,
                dn,
            );
        }
    }

    fn naive_zig_zag_channels(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        length: usize,
        extend: bool,
    ) -> ZigZagChannelsOutput {
        let mut middle = vec![f64::NAN; close.len()];
        let mut upper = vec![f64::NAN; close.len()];
        let mut lower = vec![f64::NAN; close.len()];
        let mut idx = 0usize;
        while idx < close.len() {
            while idx < close.len() && !is_valid_ohlc(open[idx], high[idx], low[idx], close[idx]) {
                idx += 1;
            }
            if idx >= close.len() {
                break;
            }
            let start = idx;
            idx += 1;
            while idx < close.len() && is_valid_ohlc(open[idx], high[idx], low[idx], close[idx]) {
                idx += 1;
            }
            let end = idx;
            if end - start >= length + 1 {
                naive_run(
                    &open[start..end],
                    &high[start..end],
                    &low[start..end],
                    &close[start..end],
                    length,
                    extend,
                    &mut middle[start..end],
                    &mut upper[start..end],
                    &mut lower[start..end],
                );
            }
        }
        ZigZagChannelsOutput {
            middle,
            upper,
            lower,
        }
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn zig_zag_channels_matches_naive_reference() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(256);
        let input = ZigZagChannelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            ZigZagChannelsParams {
                length: Some(7),
                extend: Some(true),
            },
        );
        let out = zig_zag_channels_with_kernel(&input, Kernel::Scalar)?;
        let expected = naive_zig_zag_channels(&open, &high, &low, &close, 7, true);
        assert_series_close(&out.middle, &expected.middle, 1e-12);
        assert_series_close(&out.upper, &expected.upper, 1e-12);
        assert_series_close(&out.lower, &expected.lower, 1e-12);
        Ok(())
    }

    #[test]
    fn zig_zag_channels_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(220);
        let input = ZigZagChannelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            ZigZagChannelsParams {
                length: Some(6),
                extend: Some(true),
            },
        );
        let base = zig_zag_channels(&input)?;
        let mut middle = vec![0.0; close.len()];
        let mut upper = vec![0.0; close.len()];
        let mut lower = vec![0.0; close.len()];
        zig_zag_channels_into_slice(&mut middle, &mut upper, &mut lower, &input, Kernel::Auto)?;
        assert_series_close(&base.middle, &middle, 1e-12);
        assert_series_close(&base.upper, &upper, 1e-12);
        assert_series_close(&base.lower, &lower, 1e-12);
        Ok(())
    }

    #[test]
    fn zig_zag_channels_extend_changes_tail_only() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(180);
        let extend_true = zig_zag_channels(&ZigZagChannelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            ZigZagChannelsParams {
                length: Some(8),
                extend: Some(true),
            },
        ))?;
        let extend_false = zig_zag_channels(&ZigZagChannelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            ZigZagChannelsParams {
                length: Some(8),
                extend: Some(false),
            },
        ))?;

        let finite_true = extend_true.middle.iter().filter(|v| v.is_finite()).count();
        let finite_false = extend_false.middle.iter().filter(|v| v.is_finite()).count();
        assert!(finite_true >= finite_false);

        for i in 0..close.len() {
            if extend_false.middle[i].is_finite() {
                assert!(extend_true.middle[i].is_finite());
            }
        }
        Ok(())
    }

    #[test]
    fn zig_zag_channels_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(192);
        let single = zig_zag_channels(&ZigZagChannelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            ZigZagChannelsParams {
                length: Some(9),
                extend: Some(true),
            },
        ))?;
        let batch = zig_zag_channels_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &ZigZagChannelsBatchRange {
                length: (9, 9, 0),
                extend: true,
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_close(&batch.middle, &single.middle, 1e-12);
        assert_series_close(&batch.upper, &single.upper, 1e-12);
        assert_series_close(&batch.lower, &single.lower, 1e-12);
        Ok(())
    }

    #[test]
    fn zig_zag_channels_rejects_invalid_params() {
        let (open, high, low, close) = sample_ohlc(32);
        let err = zig_zag_channels(&ZigZagChannelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            ZigZagChannelsParams {
                length: Some(0),
                extend: Some(true),
            },
        ))
        .unwrap_err();
        assert!(matches!(err, ZigZagChannelsError::InvalidLength { .. }));
    }

    #[test]
    fn zig_zag_channels_dispatch_compute_returns_middle() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(160);
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "zig_zag_channels",
            output_id: Some("middle"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &[
                ParamKV {
                    key: "length",
                    value: ParamValue::Int(7),
                },
                ParamKV {
                    key: "extend",
                    value: ParamValue::Bool(true),
                },
            ],
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out.output_id, "middle");
        assert_eq!(out.cols, close.len());
        Ok(())
    }
}
