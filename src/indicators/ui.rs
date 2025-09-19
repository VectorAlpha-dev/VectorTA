//! # Ulcer Index (UI)
//!
//! The Ulcer Index (UI) is a volatility indicator that measures price drawdown from recent highs
//! and focuses on downside risk. It is calculated as the square root of the average of the squared
//! percentage drawdowns from the rolling maximum price within a specified window.
//!
//! ## Parameters
//! - **period**: Window size (number of data points), default 14.
//! - **scalar**: Multiplier applied to drawdown, default 100.0.
//!
//! ## Returns
//! - **Ok(UiOutput)** on success, containing a Vec<f64> matching the input length.
//! - **Err(UiError)** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Stub implementations that delegate to scalar. SIMD optimization opportunity for rolling max calculations and squared drawdown computations.
//! - **Streaming Performance**: O(n) implementation - recalculates rolling max by iterating through entire buffer on each update. Could be optimized with a deque-based max tracker for O(1) amortized.
//! - **Memory Optimization**: Uses `alloc_with_nan_prefix` and batch helpers properly. Uses AVec for cache-aligned buffers but SIMD kernels not yet implemented to take advantage.

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for UiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            UiData::Slice(slice) => slice,
            UiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct UiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct UiParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
}

impl Default for UiParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UiInput<'a> {
    pub data: UiData<'a>,
    pub params: UiParams,
}

impl<'a> UiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: UiParams) -> Self {
        Self {
            data: UiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: UiParams) -> Self {
        Self {
            data: UiData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", UiParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_scalar(&self) -> f64 {
        self.params.scalar.unwrap_or(100.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UiBuilder {
    period: Option<usize>,
    scalar: Option<f64>,
    kernel: Kernel,
}

impl Default for UiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            scalar: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn scalar(mut self, s: f64) -> Self {
        self.scalar = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<UiOutput, UiError> {
        let p = UiParams {
            period: self.period,
            scalar: self.scalar,
        };
        let i = UiInput::from_candles(c, "close", p);
        ui_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<UiOutput, UiError> {
        let p = UiParams {
            period: self.period,
            scalar: self.scalar,
        };
        let i = UiInput::from_slice(d, p);
        ui_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<UiStream, UiError> {
        let p = UiParams {
            period: self.period,
            scalar: self.scalar,
        };
        UiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum UiError {
    #[error("ui: Empty input")]
    EmptyInput,
    #[error("ui: All values are NaN.")]
    AllValuesNaN,
    #[error("ui: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ui: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ui: Invalid length: expected = {expected}, actual = {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("ui: Invalid scalar: {scalar}")]
    InvalidScalar { scalar: f64 },
}

#[inline]
pub fn ui(input: &UiInput) -> Result<UiOutput, UiError> {
    ui_with_kernel(input, Kernel::Auto)
}

pub fn ui_with_kernel(input: &UiInput, kernel: Kernel) -> Result<UiOutput, UiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(UiError::EmptyInput);
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    let period = input.get_period();
    let scalar = input.get_scalar();

    if period == 0 || period > len {
        return Err(UiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(UiError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if !scalar.is_finite() {
        return Err(UiError::InvalidScalar { scalar });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warmup = first + (period * 2 - 2);
    let mut out = alloc_with_nan_prefix(len, warmup.min(len));

    // no extra clearing needed; prefix already set
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => ui_scalar(data, period, scalar, first, &mut out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => ui_avx2(data, period, scalar, first, &mut out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => ui_avx512(data, period, scalar, first, &mut out),
        _ => ui_scalar(data, period, scalar, first, &mut out),
    }

    Ok(UiOutput { values: out })
}

pub fn ui_scalar(data: &[f64], period: usize, scalar: f64, first: usize, out: &mut [f64]) {
    use std::collections::VecDeque;

    let len = data.len();
    debug_assert_eq!(out.len(), len);

    // Monotonic deque of indices for rolling max over last `period`
    let mut deq: VecDeque<usize> = VecDeque::with_capacity(period);

    // Sliding window over last `period` squared drawdowns
    let mut sq_ring = vec![f64::NAN; period];
    let mut ring_idx = 0usize;
    let mut sum = 0.0f64;
    let mut count = 0usize;

    for i in first..len {
        // expire indices older than window start
        let start = i.saturating_add(1).saturating_sub(period);
        while let Some(&j) = deq.front() {
            if j < start {
                deq.pop_front();
            } else {
                break;
            }
        }
        // push current if finite
        let xi = data[i];
        if xi.is_finite() {
            while let Some(&j) = deq.back() {
                let xj = data[j];
                if !xj.is_finite() || xj <= xi {
                    deq.pop_back();
                } else {
                    break;
                }
            }
            deq.push_back(i);
        }

        // squared drawdown only once we have at least `period` samples since `first`
        let dd_sq = if i + 1 >= first + period {
            if let Some(&jmax) = deq.front() {
                let m = data[jmax];
                if m.abs() > f64::EPSILON && m.is_finite() && xi.is_finite() {
                    let dd = scalar * (xi - m) / m;
                    dd * dd
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        // update ring and running sum
        let old = sq_ring[ring_idx];
        sq_ring[ring_idx] = dd_sq;
        ring_idx = (ring_idx + 1) % period;

        if old.is_finite() {
            sum -= old;
            count -= 1;
        }
        if dd_sq.is_finite() {
            sum += dd_sq;
            count += 1;
        }

        // emit once past warmup
        let warmup_end = first + (period * 2 - 2);
        if i >= warmup_end {
            out[i] = if count == period {
                (sum / period as f64).sqrt()
            } else {
                f64::NAN
            };
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ui_avx512(data: &[f64], period: usize, scalar: f64, first: usize, out: &mut [f64]) {
    unsafe { ui_avx512_short(data, period, scalar, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ui_avx2(data: &[f64], period: usize, scalar: f64, first: usize, out: &mut [f64]) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ui_avx512_short(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ui_avx512_long(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[derive(Debug, Clone)]
pub struct UiStream {
    period: usize,
    scalar: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    squared_dd_window: Vec<f64>,
    window_idx: usize,
    warmup_counter: usize,
}

impl UiStream {
    pub fn try_new(params: UiParams) -> Result<Self, UiError> {
        let period = params.period.unwrap_or(14);
        let scalar = params.scalar.unwrap_or(100.0);
        if period == 0 {
            return Err(UiError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if !scalar.is_finite() {
            return Err(UiError::InvalidScalar { scalar });
        }
        Ok(Self {
            period,
            scalar,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            squared_dd_window: vec![f64::NAN; period],
            window_idx: 0,
            warmup_counter: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Store value in circular buffer
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        // Track total values seen
        self.warmup_counter += 1;

        // Mark when buffer is first filled
        if !self.filled && self.head == 0 {
            self.filled = true;
        }

        // Need at least one full buffer before we can start calculating
        if !self.filled {
            return None;
        }

        // Calculate rolling max over the buffer
        let mut max = f64::NAN;
        for &v in &self.buffer {
            if !v.is_nan() && (max.is_nan() || v > max) {
                max = v;
            }
        }

        // Calculate squared drawdown for current value
        let squared_dd = if !value.is_nan() && !max.is_nan() && max != 0.0 {
            let dd = self.scalar * (value - max) / max;
            dd * dd
        } else {
            f64::NAN
        };

        // Store in circular window
        self.squared_dd_window[self.window_idx] = squared_dd;
        self.window_idx = (self.window_idx + 1) % self.period;

        // We need period*2-1 total values before we can produce first UI output
        // With period=5: need 9 total values, first output at index 8
        if self.warmup_counter < self.period * 2 - 1 {
            return None;
        }

        // Calculate sum of squared drawdowns
        let mut sum = 0.0;
        let mut valid = 0usize;
        for &sq_dd in &self.squared_dd_window {
            if !sq_dd.is_nan() {
                sum += sq_dd;
                valid += 1;
            }
        }

        if valid == self.period {
            Some((sum / self.period as f64).sqrt())
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct UiBatchRange {
    pub period: (usize, usize, usize),
    pub scalar: (f64, f64, f64),
}

impl Default for UiBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 60, 1),
            scalar: (100.0, 100.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiBatchBuilder {
    range: UiBatchRange,
    kernel: Kernel,
}

impl UiBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    #[inline]
    pub fn scalar_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.scalar = (start, end, step);
        self
    }
    #[inline]
    pub fn scalar_static(mut self, s: f64) -> Self {
        self.range.scalar = (s, s, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<UiBatchOutput, UiError> {
        ui_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<UiBatchOutput, UiError> {
        UiBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<UiBatchOutput, UiError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<UiBatchOutput, UiError> {
        UiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn ui_batch_with_kernel(
    data: &[f64],
    sweep: &UiBatchRange,
    k: Kernel,
) -> Result<UiBatchOutput, UiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(UiError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Default to scalar for any other kernel
    };
    ui_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct UiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl UiBatchOutput {
    pub fn row_for_params(&self, p: &UiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.scalar.unwrap_or(100.0) - p.scalar.unwrap_or(100.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &UiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &UiBatchRange) -> Vec<UiParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        // Guard against invalid steps that could cause infinite loops
        if (start < end && step <= 0.0) || (start > end && step >= 0.0) {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        let max_iterations = 10000; // Safety limit
        let mut iterations = 0;
        while x <= end + 1e-12 && iterations < max_iterations {
            v.push(x);
            x += step;
            iterations += 1;
        }
        v
    }
    let periods = axis_usize(r.period);
    let scalars = axis_f64(r.scalar);
    let mut out = Vec::with_capacity(periods.len() * scalars.len());
    for &p in &periods {
        for &s in &scalars {
            out.push(UiParams {
                period: Some(p),
                scalar: Some(s),
            });
        }
    }
    out
}

#[inline(always)]
pub fn ui_batch_slice(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
) -> Result<UiBatchOutput, UiError> {
    ui_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ui_batch_par_slice(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
) -> Result<UiBatchOutput, UiError> {
    ui_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ui_batch_inner(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<UiBatchOutput, UiError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(UiError::EmptyInput);
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(UiError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // Resolve Kernel::Auto to a concrete kernel
    let kern = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let max_warmup = first + (max_p * 2 - 2);
    if data.len() <= max_warmup {
        return Err(UiError::NotEnoughValidData {
            needed: max_warmup + 1,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // per-row warmups include `first`
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + (c.period.unwrap() * 2 - 2))
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let scalar = combos[row].scalar.unwrap();
        match kern {
            Kernel::Scalar => ui_row_scalar(data, first, period, scalar, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ui_row_avx2(data, first, period, scalar, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ui_row_avx512(data, first, period, scalar, out_row),
            _ => ui_row_scalar(data, first, period, scalar, out_row),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Convert back to Vec<f64> from ManuallyDrop
    let values = unsafe {
        let ptr = buf_guard.as_mut_ptr() as *mut f64;
        let len = buf_guard.len();
        let cap = buf_guard.capacity();
        core::mem::forget(buf_guard);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(UiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ui_row_scalar(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // `out` already has NaN prefix from init_matrix_prefixes
    ui_scalar(data, period, scalar, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx2(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX2 optimizations
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx512(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX512 optimizations
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx512_short(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX512 optimizations for short periods
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx512_long(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX512 optimizations for long periods
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

// Python bindings

#[cfg(feature = "python")]
#[pyfunction(name = "ui")]
#[pyo3(signature = (data, period, scalar=100.0, kernel=None))]
pub fn ui_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    scalar: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = UiParams {
        period: Some(period),
        scalar: Some(scalar),
    };
    let input = UiInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ui_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ui_batch")]
#[pyo3(signature = (data, period_range, scalar_range=(100.0, 100.0, 0.0), kernel=None))]
pub fn ui_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    scalar_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = UiBatchRange {
        period: period_range,
        scalar: scalar_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => Kernel::Scalar, // Default to scalar for any other kernel
            };
            ui_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "scalars",
        combos
            .iter()
            .map(|p| p.scalar.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "UiStream")]
pub struct UiStreamPy {
    inner: UiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl UiStreamPy {
    #[new]
    pub fn new(period: usize, scalar: f64) -> PyResult<Self> {
        let params = UiParams {
            period: Some(period),
            scalar: Some(scalar),
        };
        let inner = UiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(UiStreamPy { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// Helper function for batch operations with direct output writing
#[inline(always)]
fn ui_batch_inner_into(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<UiParams>, UiError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(UiError::EmptyInput);
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(UiError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let max_warmup = first + (max_p * 2 - 2);
    if data.len() <= max_warmup {
        return Err(UiError::NotEnoughValidData {
            needed: max_warmup + 1,
            valid: data.len() - first,
        });
    }

    let cols = data.len();
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + (combo.period.unwrap() * 2 - 2);
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let scalar = combos[row].scalar.unwrap();
        match kern {
            Kernel::Scalar => ui_row_scalar(data, first, period, scalar, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ui_row_avx2(data, first, period, scalar, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ui_row_avx512(data, first, period, scalar, out_row),
            _ => ui_row_scalar(data, first, period, scalar, out_row),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

// WASM bindings

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Core helper for writing directly to output slice - no allocations
#[cfg(feature = "wasm")]
pub fn ui_into_slice(dst: &mut [f64], input: &UiInput, kern: Kernel) -> Result<(), UiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(UiError::EmptyInput);
    }

    if dst.len() != len {
        return Err(UiError::InvalidLength {
            expected: len,
            actual: dst.len(),
        });
    }

    let period = input.get_period();
    let scalar = input.get_scalar();
    if period == 0 || period > len {
        return Err(UiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if !scalar.is_finite() {
        return Err(UiError::InvalidScalar { scalar });
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    if (len - first) < period {
        return Err(UiError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // correct prefix
    let warmup = first + (period * 2 - 2);
    for v in &mut dst[..warmup.min(len)] {
        *v = f64::NAN;
    }

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => ui_scalar(data, period, scalar, first, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => ui_avx2(data, period, scalar, first, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => ui_avx512(data, period, scalar, first, dst),
        _ => ui_scalar(data, period, scalar, first, dst),
    }
    Ok(())
}

/// Safe API - allocates and returns Vec<f64>
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_js(data: &[f64], period: usize, scalar: f64) -> Result<Vec<f64>, JsValue> {
    if !scalar.is_finite() {
        return Err(JsValue::from_str(&format!("Invalid scalar: {}", scalar)));
    }
    let params = UiParams {
        period: Some(period),
        scalar: Some(scalar),
    };
    let input = UiInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    ui_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

/// Fast API with aliasing detection - zero allocations unless aliased
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    scalar: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    if !scalar.is_finite() {
        return Err(JsValue::from_str(&format!("Invalid scalar: {}", scalar)));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = UiParams {
            period: Some(period),
            scalar: Some(scalar),
        };
        let input = UiInput::from_slice(data, params);

        if in_ptr == out_ptr.cast_const() {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            ui_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ui_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

/// Memory allocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

/// Memory deallocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

/// Batch configuration for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UiBatchConfig {
    pub period_range: (usize, usize, usize),
    pub scalar_range: (f64, f64, f64),
}

/// Batch output for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UiParams>,
    pub rows: usize,
    pub cols: usize,
}

/// Batch processing API
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ui_batch)]
pub fn ui_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: UiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = UiBatchRange {
        period: config.period_range,
        scalar: config.scalar_range,
    };

    let output = ui_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = UiBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Fast batch API with raw pointers
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    scalar_start: f64,
    scalar_end: f64,
    scalar_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ui_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = UiBatchRange {
            period: (period_start, period_end, period_step),
            scalar: (scalar_start, scalar_end, scalar_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        ui_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_ui_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = UiParams {
            period: None,
            scalar: None,
        };
        let input = UiInput::from_candles(&candles, "close", default_params);
        let output = ui_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ui_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UiParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = UiInput::from_candles(&candles, "close", params);
        let ui_result = ui_with_kernel(&input, kernel)?;
        let expected_last_five_ui = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
        ];
        assert!(ui_result.values.len() >= 5);
        let start_index = ui_result.values.len() - 5;
        let result_last_five_ui = &ui_result.values[start_index..];
        for (i, &value) in result_last_five_ui.iter().enumerate() {
            let expected_value = expected_last_five_ui[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "[{}] UI mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        let period = 14;
        for i in 0..(period - 1) {
            assert!(ui_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_ui_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UiInput::with_default_candles(&candles);
        match input.data {
            UiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected UiData::Candles"),
        }
        let output = ui_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ui_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = UiParams {
            period: Some(0),
            scalar: None,
        };
        let input = UiInput::from_slice(&input_data, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_ui_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = UiParams {
            period: Some(10),
            scalar: None,
        };
        let input = UiInput::from_slice(&data_small, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_ui_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = UiParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = UiInput::from_slice(&single_point, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ui_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            UiParams::default(), // period: 14, scalar: 100.0
            UiParams {
                period: Some(2), // minimum viable
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(5), // small
                scalar: Some(50.0),
            },
            UiParams {
                period: Some(10), // medium
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(20), // large
                scalar: Some(200.0),
            },
            UiParams {
                period: Some(50), // very large
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(100), // extreme
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(14), // default period with different scalars
                scalar: Some(1.0),
            },
            UiParams {
                period: Some(14),
                scalar: Some(500.0),
            },
            UiParams {
                period: Some(14),
                scalar: Some(1000.0),
            },
            UiParams {
                period: Some(7), // edge case combinations
                scalar: Some(75.0),
            },
            UiParams {
                period: Some(30),
                scalar: Some(150.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = UiInput::from_candles(&candles, "close", params.clone());
            let output = ui_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, scalar={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.scalar.unwrap_or(100.0),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, scalar={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.scalar.unwrap_or(100.0),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, scalar={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.scalar.unwrap_or(100.0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ui_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_ui_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ui_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=20, 1.0f64..200.0f64).prop_flat_map(|(period, scalar)| {
            let min_data_needed = period * 2 - 2 + 20; // warmup + some data for testing
            (
                prop::collection::vec(
                    (0.001f64..1e6f64)
                        .prop_filter("positive finite", |x| x.is_finite() && *x > 0.0),
                    min_data_needed..400,
                ),
                Just(period),
                Just(scalar),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, scalar)| {
                let params = UiParams {
                    period: Some(period),
                    scalar: Some(scalar),
                };
                let input = UiInput::from_slice(&data, params);

                let UiOutput { values: out } = ui_with_kernel(&input, kernel).unwrap();
                let UiOutput { values: ref_out } = ui_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Warmup period should be NaN
                let warmup_period = period * 2 - 2;
                for i in 0..warmup_period.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                // Property 2: Non-negativity - UI must always be >= 0
                for (i, &value) in out.iter().enumerate() {
                    if !value.is_nan() {
                        prop_assert!(
                            value >= 0.0,
                            "[{}] UI must be non-negative at index {}: got {}",
                            test_name,
                            i,
                            value
                        );
                    }
                }

                // Property 3: Zero when prices monotonically increase
                let is_monotonic_increase = data.windows(2).all(|w| w[1] >= w[0]);
                if is_monotonic_increase && data.len() > warmup_period {
                    for i in warmup_period..data.len() {
                        prop_assert!(
                            out[i].abs() < 1e-9,
                            "[{}] UI should be ~0 for monotonic increase at index {}: got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                // Property 4: Period=1 edge case - UI should always be 0
                if period == 1 {
                    // With period=1, warmup is 0, so all values should be valid
                    for (i, &value) in out.iter().enumerate() {
                        prop_assert!(
                            value.abs() < 1e-9,
                            "[{}] UI with period=1 should be 0 at index {}: got {}",
                            test_name,
                            i,
                            value
                        );
                    }
                }

                // Property 5: Flat data should give UI = 0
                let is_flat = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                if is_flat && data.len() > warmup_period {
                    for i in warmup_period..data.len() {
                        prop_assert!(
                            out[i].abs() < 1e-9,
                            "[{}] UI should be 0 for flat data at index {}: got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                // Property 6: Bounded by theoretical maximum
                // UI = sqrt(avg(squared_percentage_drawdowns)) * scalar
                // Maximum theoretical: all prices drop to near-zero = 100% drawdown
                // UI_max = sqrt(1.0) * scalar = scalar
                for i in warmup_period..data.len() {
                    if !out[i].is_nan() {
                        // UI theoretical max is scalar * 1.0, allow 10% margin for numerical precision
                        prop_assert!(
                            out[i] <= scalar * 1.1,
                            "[{}] UI exceeds theoretical maximum at index {}: UI={}, max={}",
                            test_name,
                            i,
                            out[i],
                            scalar * 1.1
                        );

                        // Also check that UI is finite
                        prop_assert!(
                            out[i].is_finite(),
                            "[{}] UI is not finite at index {}: {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                // Property 7: Kernel consistency - all kernels should produce identical results
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] finite/NaN mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "[{}] kernel mismatch at index {}: {} vs {} (ULP={})",
                        test_name,
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 8: Determinism - running twice should give identical results
                let UiOutput { values: out2 } = ui_with_kernel(&input, kernel).unwrap();
                for i in 0..data.len() {
                    if out[i].is_finite() && out2[i].is_finite() {
                        prop_assert!(
                            (out[i] - out2[i]).abs() < 1e-12,
                            "[{}] Non-deterministic result at index {}: {} vs {}",
                            test_name,
                            i,
                            out[i],
                            out2[i]
                        );
                    } else {
                        prop_assert!(
                            out[i].to_bits() == out2[i].to_bits(),
                            "[{}] Non-deterministic NaN at index {}",
                            test_name,
                            i
                        );
                    }
                }

                // Property 9: Scalar proportionality
                // Test that doubling scalar doubles the output
                if scalar > 1.0 && scalar < 100.0 {
                    let params2 = UiParams {
                        period: Some(period),
                        scalar: Some(scalar * 2.0),
                    };
                    let input2 = UiInput::from_slice(&data, params2);
                    let UiOutput { values: out_scaled } = ui_with_kernel(&input2, kernel).unwrap();

                    for i in warmup_period..data.len() {
                        if out[i].is_finite() && out_scaled[i].is_finite() && out[i] > 1e-9 {
                            let ratio = out_scaled[i] / out[i];
                            prop_assert!(
                                (ratio - 2.0).abs() < 1e-6,
                                "[{}] Scalar not proportional at index {}: ratio={} (expected 2.0)",
                                test_name,
                                i,
                                ratio
                            );
                        }
                    }
                }

                // Property 10: When data is well-behaved, outputs should stabilize
                // For sufficiently large, stable data, we should get valid UI values
                let has_large_stable_region =
                    data.len() > period * 4 && data.iter().all(|&x| x > 0.1 && x < 1e5);
                if has_large_stable_region {
                    // Count valid outputs after warmup
                    let valid_count = out[warmup_period..]
                        .iter()
                        .filter(|&&x| !x.is_nan())
                        .count();
                    let expected_valid = data.len() - warmup_period;

                    // We should have mostly valid outputs (allow some edge cases)
                    prop_assert!(
                        valid_count as f64 >= expected_valid as f64 * 0.8,
                        "[{}] Too few valid outputs: {} out of {} expected",
                        test_name,
                        valid_count,
                        expected_valid
                    );
                }

                // Property 11: Volatility relationship - UI should increase with volatility
                // Create synthetic high and low volatility periods if we have enough data
                if data.len() > period * 4 {
                    // Find a stable period (low volatility)
                    let mut min_volatility_ui = f64::INFINITY;
                    let mut max_volatility_ui = 0.0;

                    for i in warmup_period..data.len() {
                        if !out[i].is_nan() {
                            // Look at the price range in the window
                            let window_start = i.saturating_sub(period - 1);
                            let window = &data[window_start..=i];
                            let max_price =
                                window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let min_price = window.iter().cloned().fold(f64::INFINITY, f64::min);
                            let price_range = (max_price - min_price) / max_price;

                            // Track UI values for different volatility levels
                            if price_range < 0.01 && out[i] < min_volatility_ui {
                                min_volatility_ui = out[i];
                            }
                            if price_range > 0.1 && out[i] > max_volatility_ui {
                                max_volatility_ui = out[i];
                            }
                        }
                    }

                    // If we found both low and high volatility periods, high should have higher UI
                    if min_volatility_ui != f64::INFINITY && max_volatility_ui > 0.0 {
                        prop_assert!(
							max_volatility_ui >= min_volatility_ui,
							"[{}] UI should be higher for volatile periods: low_vol_UI={}, high_vol_UI={}",
							test_name, min_volatility_ui, max_volatility_ui
						);
                    }
                }

                // Property 12: Direct formula verification for simple cases
                // For a window with sufficient drawdown, verify the calculation
                if period <= 5 && data.len() > warmup_period + period {
                    // Find a window where we can manually calculate
                    for i in (warmup_period + period)..data.len().min(warmup_period + period * 2) {
                        if !out[i].is_nan() && out[i] > scalar * 0.01 {
                            // Only verify when UI is meaningful
                            // Calculate the window of interest for the last 'period' UI calculations
                            // UI at position i uses a sliding window approach
                            let mut sum_squared_dd = 0.0;
                            let mut valid_count = 0;

                            // For UI at position i, we need to look at the last 'period' drawdowns
                            for j in 0..period {
                                let pos = i - j;
                                if pos >= period - 1 {
                                    // Find the rolling max for this position
                                    let max_start = pos + 1 - period;
                                    let max_end = pos + 1;
                                    let rolling_max = data[max_start..max_end]
                                        .iter()
                                        .cloned()
                                        .fold(f64::NEG_INFINITY, f64::max);

                                    if rolling_max > 0.0 && !data[pos].is_nan() {
                                        let dd = scalar * (data[pos] - rolling_max) / rolling_max;
                                        sum_squared_dd += dd * dd;
                                        valid_count += 1;
                                    }
                                }
                            }

                            if valid_count == period {
                                let manual_ui = (sum_squared_dd / period as f64).sqrt();
                                // Allow 5% tolerance or small absolute difference for floating point
                                let tolerance = manual_ui * 0.05 + 1e-6;
                                prop_assert!(
									(out[i] - manual_ui).abs() <= tolerance,
									"[{}] Direct formula verification failed at index {}: calculated={}, expected={}, diff={}",
									test_name, i, out[i], manual_ui, (out[i] - manual_ui).abs()
								);
                                break; // Only need to verify once
                            }
                        }
                    }
                }

                // Property 13: Near-zero volatility test
                // When price movements are minimal, UI should approach zero
                let has_low_volatility =
                    data.windows(2).all(|w| (w[1] - w[0]).abs() / w[0] < 0.0001); // Less than 0.01% change
                if has_low_volatility && data.len() > warmup_period {
                    for i in warmup_period..data.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                out[i] < scalar * 0.01, // UI should be less than 1% of scalar
                                "[{}] UI too high for near-zero volatility at index {}: UI={}",
                                test_name,
                                i,
                                out[i]
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_ui_tests!(
        check_ui_partial_params,
        check_ui_accuracy,
        check_ui_default_candles,
        check_ui_zero_period,
        check_ui_period_exceeds_length,
        check_ui_very_small_dataset,
        check_ui_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ui_tests!(check_ui_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = UiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = UiParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (period_start, period_end, period_step, scalar_start, scalar_end, scalar_step)
            (2, 10, 2, 100.0, 100.0, 0.0), // Small periods, static scalar
            (5, 25, 5, 50.0, 150.0, 50.0), // Medium periods with scalar sweep
            (30, 60, 15, 100.0, 100.0, 0.0), // Large periods
            (2, 5, 1, 1.0, 100.0, 33.0),   // Dense small range with scalar sweep
            (10, 20, 2, 200.0, 200.0, 0.0), // Medium range, high scalar
            (14, 14, 0, 1.0, 1000.0, 199.0), // Static period with scalar sweep
            (3, 12, 3, 75.0, 125.0, 25.0), // Small to medium with scalar variations
            (50, 100, 25, 100.0, 500.0, 200.0), // Very large periods with scalar sweep
            (7, 21, 7, 50.0, 50.0, 0.0),   // Specific periods, static scalar
        ];

        for (cfg_idx, &(p_start, p_end, p_step, s_start, s_end, s_step)) in
            test_configs.iter().enumerate()
        {
            let output = UiBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .scalar_range(s_start, s_end, s_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, scalar={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.scalar.unwrap_or(100.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, scalar={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.scalar.unwrap_or(100.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, scalar={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.scalar.unwrap_or(100.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
