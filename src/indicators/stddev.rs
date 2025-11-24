//! # Rolling Standard Deviation (STDDEV)
//!
//! The STDDEV indicator measures the rolling standard deviation of a window over input data,
//! scaled by `nbdev`. Parameters `period` and `nbdev` control the window size and deviation scale.
//! Features include AVX2/AVX512 API parity, streaming computation, and parameter grid batching.
//!
//! ## Variance Calculation
//! This implementation uses **population variance** (dividing by n) rather than sample variance
//! (dividing by n-1). This is standard for technical indicators where the window is considered
//! the entire population of interest.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **nbdev**: Non-negative multiplier for standard deviations (default: 1.0). Must be ≥ 0.
//!
//! ## Errors
//! - **AllValuesNaN**: stddev: All input values are `NaN`.
//! - **InvalidPeriod**: stddev: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: stddev: Not enough valid data points for requested `period`.
//!
//! ## Errors
//! - **EmptyInputData**: stddev: Input data slice is empty.
//! - **AllValuesNaN**: stddev: All input values are `NaN`.
//! - **InvalidPeriod**: stddev: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: stddev: Not enough valid data points for requested `period`.
//! - **InvalidNbdev**: stddev: `nbdev` is negative or non-finite.
//! - **OutputLengthMismatch**: stddev: Output slice length does not match input length.
//! - **InvalidRange**: stddev batch: Invalid sweep configuration (period/nbdev ranges).
//! - **InvalidKernelForBatch**: stddev batch: Non-batch kernel passed to batch API.
//!
//! ## Returns
//! - **Ok(StdDevOutput)** on success (output vector of length == input).
//! - **Err(StdDevError)** otherwise.
//!
//! ## Developer Notes
//! - Single-series SIMD: kept disabled (delegates to scalar). The rolling update is memory-bound and per-step dependent, and did not show consistent >5% wins at realistic sizes (10k/100k).
//! - Batch SIMD: row-specific AVX2/AVX512 from prefix sums is enabled and faster than scalar by ~5–10% at 100k on a modern x86_64 CPU.
//! - CUDA: PTX wrapper provides batch and many-series kernels with VRAM-checked launches; Python uses a DeviceArray handle with CAI v3 byte-strides and DLPack v1.x negotiation.
//! - Streaming parity: scalar path computes variance with the same operation order as the stream (`mean = sum/den; var = sum2/den - mean*mean`) to ensure bitwise consistency in tests.
//! - Allocation: follows alma.rs patterns (warmup prefix via `alloc_with_nan_prefix`; batch via `make_uninit_matrix`/`init_matrix_prefixes`).
//! - Streaming kernel: O(1) modulo-free ring buffer; NaN-robust using a `nan_count` window tracker. Emits NaN only while a NaN is inside the window and recovers as soon as it slides out.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaStddev;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
// Note: AVec and CACHELINE_ALIGN not needed for stddev as we don't use cache-aligned buffers
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for StdDevInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            StdDevData::Slice(slice) => slice,
            StdDevData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum StdDevData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct StdDevOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct StdDevParams {
    pub period: Option<usize>,
    pub nbdev: Option<f64>,
}

impl Default for StdDevParams {
    fn default() -> Self {
        Self {
            period: Some(5),
            nbdev: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StdDevInput<'a> {
    pub data: StdDevData<'a>,
    pub params: StdDevParams,
}

impl<'a> StdDevInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: StdDevParams) -> Self {
        Self {
            data: StdDevData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: StdDevParams) -> Self {
        Self {
            data: StdDevData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", StdDevParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
    #[inline]
    pub fn get_nbdev(&self) -> f64 {
        self.params.nbdev.unwrap_or(1.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StdDevBuilder {
    period: Option<usize>,
    nbdev: Option<f64>,
    kernel: Kernel,
}

impl Default for StdDevBuilder {
    fn default() -> Self {
        Self {
            period: None,
            nbdev: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StdDevBuilder {
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
    pub fn nbdev(mut self, x: f64) -> Self {
        self.nbdev = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<StdDevOutput, StdDevError> {
        let p = StdDevParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        let i = StdDevInput::from_candles(c, "close", p);
        stddev_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<StdDevOutput, StdDevError> {
        let p = StdDevParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        let i = StdDevInput::from_slice(d, p);
        stddev_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<StdDevStream, StdDevError> {
        let p = StdDevParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        StdDevStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum StdDevError {
    #[error("stddev: Input data slice is empty.")]
    EmptyInputData,
    #[error("stddev: All values are NaN.")]
    AllValuesNaN,
    #[error("stddev: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("stddev: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stddev: Invalid nbdev: {nbdev}. Must be non-negative and finite.")]
    InvalidNbdev { nbdev: f64 },
    #[error("stddev: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("stddev: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },
    #[error("stddev: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    // Kept for backward compatibility; new code should use `OutputLengthMismatch`.
    #[error("stddev: Output length mismatch: dst = {dst_len}, expected = {expected_len}")]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("stddev: Invalid kernel type: {msg}")]
    InvalidKernel { msg: String },
    #[error("stddev: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[inline]
pub fn stddev(input: &StdDevInput) -> Result<StdDevOutput, StdDevError> {
    stddev_with_kernel(input, Kernel::Auto)
}

pub fn stddev_with_kernel(
    input: &StdDevInput,
    kernel: Kernel,
) -> Result<StdDevOutput, StdDevError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(StdDevError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StdDevError::AllValuesNaN)?;
    let period = input.get_period();
    let nbdev = input.get_nbdev();

    if !nbdev.is_finite() || nbdev < 0.0 {
        return Err(StdDevError::InvalidNbdev { nbdev });
    }
    if period == 0 || period > len {
        return Err(StdDevError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(StdDevError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let warmup = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warmup);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                stddev_scalar(data, period, first, nbdev, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stddev_avx2(data, period, first, nbdev, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                stddev_avx512(data, period, first, nbdev, &mut out)
            }
            // fallback parity with alma.rs
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                stddev_scalar(data, period, first, nbdev, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(StdDevOutput { values: out })
}

/// Write StdDev directly to output slice - no allocations (WASM optimization)
#[inline]
pub fn stddev_into_slice(
    dst: &mut [f64],
    input: &StdDevInput,
    kern: Kernel,
) -> Result<(), StdDevError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(StdDevError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StdDevError::AllValuesNaN)?;
    let period = input.get_period();
    let nbdev = input.get_nbdev();

    if !nbdev.is_finite() || nbdev < 0.0 {
        return Err(StdDevError::InvalidNbdev { nbdev });
    }
    if period == 0 || period > len {
        return Err(StdDevError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(StdDevError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if dst.len() != len {
        return Err(StdDevError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    // warmup fill
    let warmup = first + period - 1;
    for v in &mut dst[..warmup] {
        *v = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => stddev_scalar(data, period, first, nbdev, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stddev_avx2(data, period, first, nbdev, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => stddev_avx512(data, period, first, nbdev, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                stddev_scalar(data, period, first, nbdev, dst)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// Write STDDEV into a caller-provided buffer without allocating.
///
/// - Preserves NaN warmups exactly as the Vec-returning API does.
/// - The output slice length must equal the input length.
/// - Selects `Kernel::Auto` at runtime (same as `stddev()`).
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn stddev_into(input: &StdDevInput, out: &mut [f64]) -> Result<(), StdDevError> {
    // Delegate to the existing no-allocation entry which validates lengths,
    // computes identical warmup, and dispatches to the selected kernel.
    stddev_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn stddev_scalar(data: &[f64], period: usize, first: usize, nbdev: f64, out: &mut [f64]) {
    // Rolling O(1) update of sum and sum of squares.
    // Compute variance with the same operation order as the stream path
    // to preserve bitwise-equivalent results in tests:
    //   mean = sum / den; var = (sum_sqr / den) - (mean * mean)
    // Using division by a constant `den` matches the streaming update logic.
    let den = period as f64;

    let mut sum = 0.0;
    let mut sum_sqr = 0.0;
    for &val in &data[first..first + period] {
        sum += val;
        sum_sqr += val * val;
    }

    // First output at warmup index
    let mut mean = sum / den;
    let mut var = (sum_sqr / den) - (mean * mean);
    out[first + period - 1] = if var <= 0.0 { 0.0 } else { var.sqrt() * nbdev };

    // Subsequent outputs
    for i in (first + period)..data.len() {
        let old = data[i - period];
        let new = data[i];
        sum += new - old;
        sum_sqr += new * new - old * old;
        mean = sum / den;
        var = (sum_sqr / den) - (mean * mean);
        out[i] = if var <= 0.0 { 0.0 } else { var.sqrt() * nbdev };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stddev_avx2(data: &[f64], period: usize, first: usize, nbdev: f64, out: &mut [f64]) {
    stddev_scalar(data, period, first, nbdev, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stddev_avx512(data: &[f64], period: usize, first: usize, nbdev: f64, out: &mut [f64]) {
    if period <= 32 {
        unsafe { stddev_avx512_short(data, period, first, nbdev, out) }
    } else {
        unsafe { stddev_avx512_long(data, period, first, nbdev, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn stddev_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn stddev_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out);
}

#[derive(Debug, Clone)]
pub struct StdDevStream {
    period: usize,
    nbdev: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    sum: f64,
    sum_sqr: f64,
    nan_count: usize, // Track NaNs currently inside the window
}

impl StdDevStream {
    pub fn try_new(params: StdDevParams) -> Result<Self, StdDevError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(StdDevError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let nbdev = params.nbdev.unwrap_or(1.0);
        if !nbdev.is_finite() || nbdev < 0.0 {
            return Err(StdDevError::InvalidNbdev { nbdev });
        }
        Ok(Self {
            period,
            nbdev,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            sum: 0.0,
            sum_sqr: 0.0,
            nan_count: 0,
        })
    }

    /// O(1) update with NaN-robustness and modulo-free ring advance.
    ///
    /// - Maintains rolling sum and sum of squares ignoring NaNs (tracked via `nan_count`).
    /// - Emits `None` until warmup completes (first `period` values seen).
    /// - Emits `NaN` if any NaN is inside the current window; recovers once it slides out.
    /// - Computes variance with the same operation order as the scalar path for parity:
    ///   mean = sum/den; var = sum2/den - mean*mean.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Warmup phase: fill buffer, build sums, first emit at window-full.
        if !self.filled {
            if value.is_nan() {
                self.nan_count += 1;
            } else {
                self.sum += value;
                self.sum_sqr += value * value;
            }
            self.buffer[self.head] = value;

            let next = self.head + 1;
            if next == self.period {
                // Window becomes full on this tick; emit first value
                self.head = 0;
                self.filled = true;

                if self.nan_count > 0 {
                    return Some(f64::NAN);
                }
                let den = self.period as f64;
                let mean = self.sum / den;
                let var = (self.sum_sqr / den) - (mean * mean);
                return Some(if var <= 0.0 {
                    0.0
                } else {
                    var.sqrt() * self.nbdev
                });
            } else {
                self.head = next;
                return None;
            }
        }

        // Steady state: drop oldest, add newest.
        let old = self.buffer[self.head];
        let new_is_nan = value.is_nan();
        let old_is_nan = old.is_nan();

        // Update sums with operation order parity when both are finite.
        match (old_is_nan, new_is_nan) {
            (false, false) => {
                // Match original rounding: single add of (new - old)
                self.sum += value - old;
                self.sum_sqr += (value * value) - (old * old);
            }
            (false, true) => {
                // Remove only the finite old; add no contribution for NaN new
                self.sum -= old;
                self.sum_sqr -= old * old;
                self.nan_count += 1;
            }
            (true, false) => {
                // Old was NaN; just add the finite new
                if self.nan_count > 0 {
                    self.nan_count -= 1;
                }
                self.sum += value;
                self.sum_sqr += value * value;
            }
            (true, true) => {
                // NaN slid out and NaN slid in: nan_count unchanged; sums unchanged
                // Maintain nan_count explicitly
                // old NaN out
                if self.nan_count > 0 {
                    self.nan_count -= 1;
                }
                // new NaN in
                self.nan_count += 1;
            }
        }

        // Write new value into ring
        self.buffer[self.head] = value;

        // Advance ring head without modulo
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }

        // Emit NaN if any NaN in the window
        if self.nan_count > 0 {
            return Some(f64::NAN);
        }

        let den = self.period as f64;
        let mean = self.sum / den;
        let var = (self.sum_sqr / den) - (mean * mean);
        Some(if var <= 0.0 {
            0.0
        } else {
            var.sqrt() * self.nbdev
        })
    }
}

#[derive(Clone, Debug)]
pub struct StdDevBatchRange {
    pub period: (usize, usize, usize),
    pub nbdev: (f64, f64, f64),
}

impl Default for StdDevBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 50, 1),
            nbdev: (1.0, 1.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StdDevBatchBuilder {
    range: StdDevBatchRange,
    kernel: Kernel,
}

impl StdDevBatchBuilder {
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
    pub fn nbdev_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nbdev = (start, end, step);
        self
    }
    #[inline]
    pub fn nbdev_static(mut self, x: f64) -> Self {
        self.range.nbdev = (x, x, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<StdDevBatchOutput, StdDevError> {
        stddev_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<StdDevBatchOutput, StdDevError> {
        StdDevBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<StdDevBatchOutput, StdDevError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<StdDevBatchOutput, StdDevError> {
        StdDevBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn stddev_batch_with_kernel(
    data: &[f64],
    sweep: &StdDevBatchRange,
    k: Kernel,
) -> Result<StdDevBatchOutput, StdDevError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(StdDevError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    stddev_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StdDevBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<StdDevParams>,
    pub rows: usize,
    pub cols: usize,
}
impl StdDevBatchOutput {
    pub fn row_for_params(&self, p: &StdDevParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
                && (c.nbdev.unwrap_or(1.0) - p.nbdev.unwrap_or(1.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &StdDevParams) -> Option<&[f64]> {
        self.row_for_params(p).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            let end = start.checked_add(self.cols)?;
            self.values.get(start..end)
        })
    }
}

#[inline(always)]
fn expand_grid_checked(r: &StdDevBatchRange) -> Result<Vec<StdDevParams>, StdDevError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, StdDevError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                let next = cur.saturating_add(step);
                if next == cur {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                let next = cur.saturating_sub(step);
                if next == cur {
                    break;
                }
                cur = next;
                if cur == 0 && end > 0 {
                    break;
                }
            }
        }
        if v.is_empty() {
            return Err(StdDevError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, StdDevError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let st = if step > 0.0 { step } else { -step };
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += st;
            }
        } else {
            let st = if step > 0.0 { -step } else { step };
            if st.abs() < 1e-12 {
                return Ok(vec![start]);
            }
            let mut x = start;
            while x >= end - 1e-12 {
                out.push(x);
                x += st;
            }
        }
        if out.is_empty() {
            return Err(StdDevError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let periods = axis_usize(r.period)?;
    let nbdevs = axis_f64(r.nbdev)?;
    let cap = periods
        .len()
        .checked_mul(nbdevs.len())
        .ok_or_else(|| StdDevError::InvalidInput {
            msg: "stddev: parameter grid size overflow".to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &n in &nbdevs {
            out.push(StdDevParams {
                period: Some(p),
                nbdev: Some(n),
            });
        }
    }
    if out.is_empty() {
        return Err(StdDevError::InvalidRange {
            start: r.period.0.to_string(),
            end: r.period.1.to_string(),
            step: r.period.2.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
pub fn stddev_batch_slice(
    data: &[f64],
    sweep: &StdDevBatchRange,
    kern: Kernel,
) -> Result<StdDevBatchOutput, StdDevError> {
    stddev_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn stddev_batch_par_slice(
    data: &[f64],
    sweep: &StdDevBatchRange,
    kern: Kernel,
) -> Result<StdDevBatchOutput, StdDevError> {
    stddev_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn stddev_batch_inner(
    data: &[f64],
    sweep: &StdDevBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<StdDevBatchOutput, StdDevError> {
    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(StdDevError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StdDevError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(StdDevError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Allocate uninitialized matrix and set NaN prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to initialized slice
    let mut values = unsafe {
        Vec::from_raw_parts(
            buf_mu.as_mut_ptr() as *mut f64,
            buf_mu.len(),
            buf_mu.capacity(),
        )
    };
    std::mem::forget(buf_mu);

    // Build prefix sums once (n+1 to simplify window differences); track NaNs like zscore
    #[derive(Clone)]
    struct StdPrefixes {
        ps: Vec<f64>,
        ps2: Vec<f64>,
        pnan: Vec<i32>,
    }
    #[inline]
    fn build_std_prefixes(data: &[f64]) -> StdPrefixes {
        let n = data.len();
        let mut ps = vec![0.0f64; n + 1];
        let mut ps2 = vec![0.0f64; n + 1];
        let mut pnan = vec![0i32; n + 1];
        for i in 0..n {
            let v = data[i];
            if v.is_nan() {
                ps[i + 1] = ps[i];
                ps2[i + 1] = ps2[i];
                pnan[i + 1] = pnan[i] + 1;
            } else {
                ps[i + 1] = ps[i] + v;
                ps2[i + 1] = ps2[i] + v * v;
                pnan[i + 1] = pnan[i];
            }
        }
        StdPrefixes { ps, ps2, pnan }
    }

    #[inline]
    fn stddev_from_prefix_scalar(
        warmup_end: usize,
        period: usize,
        nbdev: f64,
        pre: &StdPrefixes,
        out_row: &mut [f64],
    ) {
        let n = out_row.len();
        if n <= warmup_end {
            return;
        }

        // Precompute reciprocals to avoid per-element divisions
        let inv_den = 1.0 / (period as f64);
        let inv_den2 = inv_den * inv_den;

        // Fast path: no NaNs anywhere in the slice ⇒ skip per-element NaN checks
        let no_nans = pre.pnan[n] == 0;
        if no_nans {
            for i in warmup_end..n {
                let sum = pre.ps[i + 1] - pre.ps[i + 1 - period];
                let sum2 = pre.ps2[i + 1] - pre.ps2[i + 1 - period];
                // var = sum2 * inv_den - (sum * sum) * inv_den2
                let var = sum2.mul_add(inv_den, -(sum * sum) * inv_den2);
                out_row[i] = if var <= 0.0 { 0.0 } else { var.sqrt() * nbdev };
            }
            return;
        }

        // General path: handle NaNs per-window
        for i in warmup_end..n {
            if pre.pnan[i + 1] - pre.pnan[i + 1 - period] > 0 {
                out_row[i] = f64::NAN;
                continue;
            }
            let sum = pre.ps[i + 1] - pre.ps[i + 1 - period];
            let sum2 = pre.ps2[i + 1] - pre.ps2[i + 1 - period];
            let var = sum2.mul_add(inv_den, -(sum * sum) * inv_den2);
            out_row[i] = if var <= 0.0 { 0.0 } else { var.sqrt() * nbdev };
        }
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[inline]
    unsafe fn stddev_from_prefix_avx2(
        warmup_end: usize,
        period: usize,
        nbdev: f64,
        pre: &StdPrefixes,
        out_row: &mut [f64],
    ) {
        use core::arch::x86_64::*;
        let n = out_row.len();
        if n <= warmup_end {
            return;
        }
        let no_nans = pre.pnan[n] == 0;
        if !no_nans {
            // Fallback to scalar with NaN handling
            stddev_from_prefix_scalar(warmup_end, period, nbdev, pre, out_row);
            return;
        }

        let inv_den = 1.0 / (period as f64);
        let inv_den2 = inv_den * inv_den;
        let v_inv_den = _mm256_set1_pd(inv_den);
        let v_inv_den2 = _mm256_set1_pd(inv_den2);
        let v_nbdev = _mm256_set1_pd(nbdev);
        let v_zero = _mm256_set1_pd(0.0);

        let mut i = warmup_end;
        while i + 4 <= n {
            // Sums
            let s_hi = _mm256_loadu_pd(pre.ps.as_ptr().add(i + 1));
            let s_lo = _mm256_loadu_pd(pre.ps.as_ptr().add(i + 1 - period));
            let sum = _mm256_sub_pd(s_hi, s_lo);

            let q_hi = _mm256_loadu_pd(pre.ps2.as_ptr().add(i + 1));
            let q_lo = _mm256_loadu_pd(pre.ps2.as_ptr().add(i + 1 - period));
            let sum2 = _mm256_sub_pd(q_hi, q_lo);

            // var = sum2*inv_den - (sum*sum)*inv_den2
            let sum_sq = _mm256_mul_pd(sum, sum);
            let term = _mm256_mul_pd(sum_sq, v_inv_den2);
            let var = _mm256_sub_pd(_mm256_mul_pd(sum2, v_inv_den), term);
            let var_pos = _mm256_max_pd(var, v_zero);
            let stdv = _mm256_sqrt_pd(var_pos);
            let outv = _mm256_mul_pd(stdv, v_nbdev);
            _mm256_storeu_pd(out_row.as_mut_ptr().add(i), outv);
            i += 4;
        }
        // Tail
        if i < n {
            stddev_from_prefix_scalar(i, period, nbdev, pre, out_row);
        }
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[inline]
    unsafe fn stddev_from_prefix_avx512(
        warmup_end: usize,
        period: usize,
        nbdev: f64,
        pre: &StdPrefixes,
        out_row: &mut [f64],
    ) {
        use core::arch::x86_64::*;
        let n = out_row.len();
        if n <= warmup_end {
            return;
        }
        let no_nans = pre.pnan[n] == 0;
        if !no_nans {
            stddev_from_prefix_scalar(warmup_end, period, nbdev, pre, out_row);
            return;
        }

        let inv_den = 1.0 / (period as f64);
        let inv_den2 = inv_den * inv_den;
        let v_inv_den = _mm512_set1_pd(inv_den);
        let v_inv_den2 = _mm512_set1_pd(inv_den2);
        let v_nbdev = _mm512_set1_pd(nbdev);
        let v_zero = _mm512_set1_pd(0.0);

        let mut i = warmup_end;
        while i + 8 <= n {
            let s_hi = _mm512_loadu_pd(pre.ps.as_ptr().add(i + 1));
            let s_lo = _mm512_loadu_pd(pre.ps.as_ptr().add(i + 1 - period));
            let sum = _mm512_sub_pd(s_hi, s_lo);

            let q_hi = _mm512_loadu_pd(pre.ps2.as_ptr().add(i + 1));
            let q_lo = _mm512_loadu_pd(pre.ps2.as_ptr().add(i + 1 - period));
            let sum2 = _mm512_sub_pd(q_hi, q_lo);

            let sum_sq = _mm512_mul_pd(sum, sum);
            let term = _mm512_mul_pd(sum_sq, v_inv_den2);
            let var = _mm512_sub_pd(_mm512_mul_pd(sum2, v_inv_den), term);
            let var_pos = _mm512_max_pd(var, v_zero);
            let stdv = _mm512_sqrt_pd(var_pos);
            let outv = _mm512_mul_pd(stdv, v_nbdev);
            _mm512_storeu_pd(out_row.as_mut_ptr().add(i), outv);
            i += 8;
        }
        if i < n {
            stddev_from_prefix_scalar(i, period, nbdev, pre, out_row);
        }
    }

    let prefixes = build_std_prefixes(data);

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let nbdev = combos[row].nbdev.unwrap();
        let warmup_end = first + period - 1;
        match kern {
            Kernel::Scalar => {
                stddev_from_prefix_scalar(warmup_end, period, nbdev, &prefixes, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => unsafe {
                stddev_from_prefix_avx2(warmup_end, period, nbdev, &prefixes, out_row)
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => unsafe {
                stddev_from_prefix_avx512(warmup_end, period, nbdev, &prefixes, out_row)
            },
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                stddev_from_prefix_scalar(warmup_end, period, nbdev, &prefixes, out_row)
            }
            _ => unreachable!(),
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

    Ok(StdDevBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn stddev_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx2(data: &[f64], first: usize, period: usize, nbdev: f64, out: &mut [f64]) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        stddev_row_avx512_short(data, first, period, nbdev, out)
    } else {
        stddev_row_avx512_long(data, first, period, nbdev, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stddev_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    stddev_scalar(data, period, first, nbdev, out)
}

#[inline(always)]
pub fn stddev_batch_inner_into(
    data: &[f64],
    sweep: &StdDevBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<StdDevParams>, StdDevError> {
    let combos = expand_grid_checked(sweep)?;

    let len = data.len();
    if len == 0 {
        return Err(StdDevError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StdDevError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(StdDevError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| StdDevError::InvalidInput {
            msg: "stddev: rows*cols overflow in batch_into".to_string(),
        })?;
    if out.len() != expected {
        return Err(StdDevError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // warmup vector and helper init
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let nbdev = combos[row].nbdev.unwrap();
        match kern {
            Kernel::Scalar => stddev_row_scalar(data, first, period, nbdev, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => stddev_row_avx2(data, first, period, nbdev, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => stddev_row_avx512(data, first, period, nbdev, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => stddev_row_scalar(data, first, period, nbdev, out_row),
            _ => unreachable!(),
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

#[inline(always)]
pub fn expand_grid_stddev(r: &StdDevBatchRange) -> Vec<StdDevParams> {
    expand_grid_checked(r).unwrap_or_else(|_| Vec::new())
}

#[cfg(feature = "python")]
#[pyfunction(name = "stddev")]
#[pyo3(signature = (data, period, nbdev, kernel=None))]
pub fn stddev_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    nbdev: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = StdDevParams {
        period: Some(period),
        nbdev: Some(nbdev),
    };
    let input = StdDevInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| stddev_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "StdDevStream")]
pub struct StdDevStreamPy {
    stream: StdDevStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StdDevStreamPy {
    #[new]
    fn new(period: usize, nbdev: f64) -> PyResult<Self> {
        let params = StdDevParams {
            period: Some(period),
            nbdev: Some(nbdev),
        };
        let stream =
            StdDevStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(StdDevStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "stddev_batch")]
#[pyo3(signature = (data, period_range, nbdev_range, kernel=None))]
pub fn stddev_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    nbdev_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = StdDevBatchRange {
        period: period_range,
        nbdev: nbdev_range,
    };

    let combos = expand_grid_checked(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

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
                _ => unreachable!(),
            };
            stddev_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
        "nbdevs",
        combos
            .iter()
            .map(|p| p.nbdev.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ============================
// Python CUDA bindings
// ============================

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "stddev_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, nbdev_range=(1.0, 1.0, 0.0), device_id=0))]
pub fn stddev_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    nbdev_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = StdDevBatchRange {
        period: period_range,
        nbdev: nbdev_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaStddev::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.stddev_batch_dev(slice_in, &sweep)
            .map(|(dev, _)| dev)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "stddev_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, cols, rows, period, nbdev, device_id=0))]
pub fn stddev_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    nbdev: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_tm_f32.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda = CudaStddev::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.stddev_many_series_one_param_time_major_dev(slice_in, cols, rows, period, nbdev as f32)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

// ============================
// WASM Bindings
// ============================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stddev_js(data: &[f64], period: usize, nbdev: f64) -> Result<Vec<f64>, JsValue> {
    let params = StdDevParams {
        period: Some(period),
        nbdev: Some(nbdev),
    };
    let input = StdDevInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()]; // Single allocation
    stddev_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stddev_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    nbdev: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = StdDevParams {
            period: Some(period),
            nbdev: Some(nbdev),
        };
        let input = StdDevInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            stddev_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            stddev_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stddev_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stddev_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StdDevBatchConfig {
    pub period_range: (usize, usize, usize),
    pub nbdev_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StdDevBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<StdDevParams>,
    // Legacy fields for backward compatibility
    pub periods: Vec<usize>,
    pub nbdevs: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = stddev_batch)]
pub fn stddev_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: StdDevBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = StdDevBatchRange {
        period: config.period_range,
        nbdev: config.nbdev_range,
    };

    let output = stddev_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = StdDevBatchJsOutput {
        values: output.values,
        periods: output.combos.iter().map(|p| p.period.unwrap()).collect(),
        nbdevs: output.combos.iter().map(|p| p.nbdev.unwrap()).collect(),
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// add ALMA-parity batch_into
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stddev_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    nbdev_start: f64,
    nbdev_end: f64,
    nbdev_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to stddev_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = StdDevBatchRange {
            period: (period_start, period_end, period_step),
            nbdev: (nbdev_start, nbdev_end, nbdev_step),
        };

        let combos = expand_grid_checked(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow in stddev_batch_into"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        stddev_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

// keep your existing config-based entrypoint but rename
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stddev_batch_into_cfg(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let config: StdDevBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let sweep = StdDevBatchRange {
        period: config.period_range,
        nbdev: config.nbdev_range,
    };

    let combos = expand_grid_checked(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    if combos.is_empty() {
        return Err(JsValue::from_str("No parameter combinations generated"));
    }

    let rows = combos.len();
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in stddev_batch_into_cfg"))?;

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        let params = stddev_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = StdDevBatchJsOutput {
            values: vec![], // Empty, data is already in out_ptr
            periods: params.iter().map(|p| p.period.unwrap()).collect(),
            nbdevs: params.iter().map(|p| p.nbdev.unwrap()).collect(),
            combos: params,
            rows,
            cols,
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_stddev_empty_input(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let empty: [f64; 0] = [];
        let input = StdDevInput::from_slice(&empty, StdDevParams::default());
        let res = stddev_with_kernel(&input, kernel);
        assert!(matches!(res, Err(StdDevError::EmptyInputData)));
        Ok(())
    }

    fn check_stddev_invalid_batch_kernel(
        test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        // Test that non-batch kernels are rejected in batch functions
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sweep = StdDevBatchRange::default();

        // Try using a non-batch kernel
        let res = stddev_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(matches!(res, Err(StdDevError::InvalidKernelForBatch(_))));

        // Also test with other non-batch kernels
        let res2 = stddev_batch_with_kernel(&data, &sweep, Kernel::Avx2);
        assert!(matches!(res2, Err(StdDevError::InvalidKernelForBatch(_))));
        Ok(())
    }

    fn check_stddev_mismatched_output_len(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = StdDevParams::default();
        let input = StdDevInput::from_slice(&data, params);

        // Create a mismatched output buffer
        let mut wrong_size_output = vec![0.0; 10]; // Wrong size: data is 5, output is 10
        let res = stddev_into_slice(&mut wrong_size_output, &input, kernel);
        assert!(matches!(res, Err(StdDevError::OutputLengthMismatch { .. })));

        // Also test with smaller buffer
        let mut small_output = vec![0.0; 3]; // Too small
        let res2 = stddev_into_slice(&mut small_output, &input, kernel);
        assert!(matches!(res2, Err(StdDevError::OutputLengthMismatch { .. })));
        Ok(())
    }

    fn check_stddev_negative_nbdev(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = StdDevParams {
            period: Some(3),
            nbdev: Some(-1.0), // Negative nbdev should be rejected
        };
        let input = StdDevInput::from_slice(&data, params.clone());
        let res = stddev_with_kernel(&input, kernel);
        assert!(matches!(res, Err(StdDevError::InvalidNbdev { .. })));

        // Also test with stream
        let stream_res = StdDevStream::try_new(params);
        assert!(matches!(stream_res, Err(StdDevError::InvalidNbdev { .. })));

        // Test infinite nbdev
        let inf_params = StdDevParams {
            period: Some(3),
            nbdev: Some(f64::INFINITY),
        };
        let inf_input = StdDevInput::from_slice(&data, inf_params);
        let inf_res = stddev_with_kernel(&inf_input, kernel);
        assert!(matches!(inf_res, Err(StdDevError::InvalidNbdev { .. })));

        // Test NaN nbdev
        let nan_params = StdDevParams {
            period: Some(3),
            nbdev: Some(f64::NAN),
        };
        let nan_input = StdDevInput::from_slice(&data, nan_params);
        let nan_res = stddev_with_kernel(&nan_input, kernel);
        assert!(matches!(nan_res, Err(StdDevError::InvalidNbdev { .. })));
        Ok(())
    }

    fn check_stddev_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = StdDevParams {
            period: None,
            nbdev: None,
        };
        let input = StdDevInput::from_candles(&candles, "close", default_params);
        let output = stddev_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_stddev_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = StdDevInput::from_candles(&candles, "close", StdDevParams::default());
        let result = stddev_with_kernel(&input, kernel)?;
        let expected_last_five = [
            180.12506767314034,
            77.7395652441455,
            127.16225857341935,
            89.40156600773197,
            218.50034325919697,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] STDDEV {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_stddev_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = StdDevInput::with_default_candles(&candles);
        match input.data {
            StdDevData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected StdDevData::Candles"),
        }
        let output = stddev_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_stddev_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = StdDevParams {
            period: Some(0),
            nbdev: None,
        };
        let input = StdDevInput::from_slice(&input_data, params);
        let res = stddev_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] STDDEV should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_stddev_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = StdDevParams {
            period: Some(10),
            nbdev: None,
        };
        let input = StdDevInput::from_slice(&data_small, params);
        let res = stddev_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] STDDEV should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_stddev_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = StdDevParams {
            period: Some(5),
            nbdev: None,
        };
        let input = StdDevInput::from_slice(&single_point, params);
        let res = stddev_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] STDDEV should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_stddev_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = StdDevParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let first_input = StdDevInput::from_candles(&candles, "close", first_params);
        let first_result = stddev_with_kernel(&first_input, kernel)?;

        let second_params = StdDevParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let second_input = StdDevInput::from_slice(&first_result.values, second_params);
        let second_result = stddev_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 19..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "STDDEV slice reinput: Expected no NaN after index 19, but found NaN at index {}",
                i
            );
        }
        Ok(())
    }

    #[test]
    fn test_stddev_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Use existing CSV data to match other tests and exercise realistic paths.
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = StdDevInput::from_candles(&candles, "close", StdDevParams::default());

        // Baseline via Vec-returning API
        let baseline = stddev(&input)?.values;

        // Preallocate output buffer and run the into variant
        let mut out = vec![0.0; baseline.len()];
        #[cfg(not(feature = "wasm"))]
        {
            stddev_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, the native symbol is not present; use the slice variant.
            stddev_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), out.len());

        // Helper to treat NaN == NaN, otherwise require exact equality (or tight epsilon)
        let eq_or_both_nan = |a: f64, b: f64| -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        };

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }

    fn check_stddev_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = StdDevInput::from_candles(
            &candles,
            "close",
            StdDevParams {
                period: Some(5),
                nbdev: None,
            },
        );
        let res = stddev_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_stddev_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;
        let nbdev = 1.0;

        let input = StdDevInput::from_candles(
            &candles,
            "close",
            StdDevParams {
                period: Some(period),
                nbdev: Some(nbdev),
            },
        );
        let batch_output = stddev_with_kernel(&input, kernel)?.values;

        let mut stream = StdDevStream::try_new(StdDevParams {
            period: Some(period),
            nbdev: Some(nbdev),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] STDDEV streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_stddev_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            StdDevParams::default(),
            StdDevParams {
                period: Some(2),
                nbdev: Some(1.0),
            },
            StdDevParams {
                period: Some(3),
                nbdev: Some(0.5),
            },
            StdDevParams {
                period: Some(5),
                nbdev: Some(1.0),
            },
            StdDevParams {
                period: Some(5),
                nbdev: Some(2.0),
            },
            StdDevParams {
                period: Some(7),
                nbdev: Some(1.5),
            },
            StdDevParams {
                period: Some(10),
                nbdev: Some(1.0),
            },
            StdDevParams {
                period: Some(10),
                nbdev: Some(3.0),
            },
            StdDevParams {
                period: Some(20),
                nbdev: Some(1.0),
            },
            StdDevParams {
                period: Some(20),
                nbdev: Some(2.5),
            },
            StdDevParams {
                period: Some(30),
                nbdev: Some(1.0),
            },
            StdDevParams {
                period: Some(50),
                nbdev: Some(2.0),
            },
            StdDevParams {
                period: Some(100),
                nbdev: Some(1.0),
            },
            StdDevParams {
                period: Some(100),
                nbdev: Some(3.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = StdDevInput::from_candles(&candles, "close", params.clone());
            let output = stddev_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_stddev_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_stddev_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate random parameter combinations and corresponding data vectors
        // Use positive values to simulate realistic financial data (prices >= 0)
        let strat = (2usize..=30, 0.5f64..=3.0f64).prop_flat_map(|(period, nbdev)| {
            (
                prop::collection::vec(
                    (0.01f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
                Just(nbdev),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period, nbdev)| {
            let params = StdDevParams {
                period: Some(period),
                nbdev: Some(nbdev),
            };
            let input = StdDevInput::from_slice(&data, params);

            let StdDevOutput { values: out } = stddev_with_kernel(&input, kernel).unwrap();
            let StdDevOutput { values: ref_out } =
                stddev_with_kernel(&input, Kernel::Scalar).unwrap();

            // Calculate expected warmup period
            let warmup_period = period - 1;

            // Verify warmup period values are NaN
            for i in 0..warmup_period {
                prop_assert!(
                    out[i].is_nan(),
                    "Expected NaN during warmup at index {}, got {}",
                    i,
                    out[i]
                );
            }

            // Test properties for valid output values
            for i in warmup_period..data.len() {
                let y = out[i];
                let r = ref_out[i];

                // Property 1: Standard deviation is always non-negative
                prop_assert!(
                    y.is_nan() || y >= 0.0,
                    "StdDev at index {} is negative: {}",
                    i,
                    y
                );

                // Property 2: Check kernel consistency (all kernels should produce same result)
                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let ulp_diff = if y_bits > r_bits {
                    y_bits - r_bits
                } else {
                    r_bits - y_bits
                };
                prop_assert!(
                    ulp_diff <= 3 || (y.is_nan() && r.is_nan()),
                    "Kernel mismatch at index {}: {} vs {} (ULP diff: {})",
                    i,
                    y,
                    r,
                    ulp_diff
                );

                // Property 3: For constant window, stddev should be 0
                let window = &data[i + 1 - period..=i];
                let is_constant = window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                if is_constant {
                    prop_assert!(
                        y.abs() < 1e-9,
                        "StdDev should be ~0 for constant data at index {}, got {}",
                        i,
                        y
                    );
                }

                // Property 4: Bounded by theoretical maximum (for nbdev=1)
                // Maximum stddev occurs with bimodal distribution at extremes
                // Max stddev ≈ range/2 * sqrt(n/(n-1)) for n points
                if (nbdev - 1.0).abs() < 1e-9 {
                    let window_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let window_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = window_max - window_min;

                    // Theoretical maximum with correction factor for finite samples
                    let max_stddev =
                        (range / 2.0) * ((period as f64) / ((period - 1) as f64)).sqrt();

                    prop_assert!(
                        y <= max_stddev + 1e-9,
                        "StdDev at index {} ({}) exceeds theoretical maximum ({})",
                        i,
                        y,
                        max_stddev
                    );
                }

                // Property 5: Scaling with nbdev
                // Run another calculation with nbdev=1.0 to verify scaling
                if (nbdev - 1.0).abs() > 1e-9 {
                    let params_unit = StdDevParams {
                        period: Some(period),
                        nbdev: Some(1.0),
                    };
                    let input_unit = StdDevInput::from_slice(&data, params_unit);
                    let StdDevOutput { values: out_unit } =
                        stddev_with_kernel(&input_unit, kernel).unwrap();
                    let y_unit = out_unit[i];

                    // Verify that output scales linearly with nbdev
                    let expected = y_unit * nbdev;
                    let diff = (y - expected).abs();
                    prop_assert!(
                        diff < 1e-9 || (y.is_nan() && y_unit.is_nan()),
                        "Scaling mismatch at index {}: {} != {} * {} = {}",
                        i,
                        y,
                        y_unit,
                        nbdev,
                        expected
                    );
                }
            }

            // Special case: period=2 with two identical values should give 0
            if period == 2 && data.len() >= 2 {
                let identical_data = vec![42.0; 10];
                let params2 = StdDevParams {
                    period: Some(2),
                    nbdev: Some(nbdev),
                };
                let input2 = StdDevInput::from_slice(&identical_data, params2);
                let StdDevOutput { values: out2 } = stddev_with_kernel(&input2, kernel).unwrap();

                for i in 1..out2.len() {
                    prop_assert!(
                        out2[i].abs() < 1e-9,
                        "StdDev for identical pairs should be 0, got {} at index {}",
                        out2[i],
                        i
                    );
                }
            }

            // Test monotonic increasing pattern
            // For a linear sequence, stddev = range * sqrt((n²-1)/12) / n
            if data.len() >= period * 2 {
                let monotonic_data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 10.0).collect();
                let mono_params = StdDevParams {
                    period: Some(period),
                    nbdev: Some(1.0),
                };
                let mono_input = StdDevInput::from_slice(&monotonic_data, mono_params);
                let StdDevOutput { values: mono_out } =
                    stddev_with_kernel(&mono_input, kernel).unwrap();

                // For linear sequence with step size d, theoretical stddev = d * sqrt((n²-1)/12)
                let step_size = 10.0;
                let expected_stddev = step_size * ((period * period - 1) as f64 / 12.0).sqrt();

                for i in (period - 1)..mono_out.len().min(period * 3) {
                    let deviation = (mono_out[i] - expected_stddev).abs();
                    prop_assert!(
                        deviation < 1.0, // Allow some tolerance for numerical precision
                        "Monotonic pattern stddev mismatch at index {}: got {}, expected ~{}",
                        i,
                        mono_out[i],
                        expected_stddev
                    );
                }
            }

            // Test alternating pattern (high-low-high-low)
            // This should produce high variance
            if data.len() >= period * 2 && period >= 4 {
                let alternating_data: Vec<f64> = (0..100)
                    .map(|i| if i % 2 == 0 { 1000.0 } else { 100.0 })
                    .collect();
                let alt_params = StdDevParams {
                    period: Some(period),
                    nbdev: Some(1.0),
                };
                let alt_input = StdDevInput::from_slice(&alternating_data, alt_params);
                let StdDevOutput { values: alt_out } =
                    stddev_with_kernel(&alt_input, kernel).unwrap();

                // For alternating pattern, stddev should be close to range/2
                let alt_range = 900.0; // 1000 - 100
                let expected_alt_stddev = alt_range / 2.0; // Approximately 450

                for i in (period - 1)..alt_out.len().min(period * 3) {
                    // Should be substantial (at least 40% of range)
                    prop_assert!(
                        alt_out[i] > alt_range * 0.4,
                        "Alternating pattern should produce high stddev at index {}: got {}",
                        i,
                        alt_out[i]
                    );
                    // But not exceed theoretical maximum
                    let max_possible =
                        (alt_range / 2.0) * ((period as f64) / ((period - 1) as f64)).sqrt();
                    prop_assert!(
                        alt_out[i] <= max_possible + 1e-9,
                        "Alternating pattern stddev exceeds maximum at index {}: got {}, max {}",
                        i,
                        alt_out[i],
                        max_possible
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_stddev_tests {
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
    generate_all_stddev_tests!(
        check_stddev_empty_input,
        check_stddev_invalid_batch_kernel,
        check_stddev_mismatched_output_len,
        check_stddev_negative_nbdev,
        check_stddev_partial_params,
        check_stddev_accuracy,
        check_stddev_default_candles,
        check_stddev_zero_period,
        check_stddev_period_exceeds_length,
        check_stddev_very_small_dataset,
        check_stddev_reinput,
        check_stddev_nan_handling,
        check_stddev_streaming,
        check_stddev_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_stddev_tests!(check_stddev_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = StdDevBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = StdDevParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            180.12506767314034,
            77.7395652441455,
            127.16225857341935,
            89.40156600773197,
            218.50034325919697,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            // (period_start, period_end, period_step, nbdev_start, nbdev_end, nbdev_step)
            (2, 10, 2, 1.0, 1.0, 0.0),    // Small periods, static nbdev
            (5, 25, 5, 0.5, 2.5, 0.5),    // Medium periods, varying nbdev
            (30, 60, 15, 1.0, 1.0, 0.0),  // Large periods, static nbdev
            (2, 5, 1, 1.0, 3.0, 1.0),     // Dense small range, varying nbdev
            (10, 10, 0, 0.5, 3.0, 0.5),   // Static period, varying nbdev
            (20, 50, 10, 2.0, 2.0, 0.0),  // Medium to large periods, static nbdev=2
            (100, 100, 0, 1.0, 3.0, 1.0), // Large static period, varying nbdev
        ];

        for (cfg_idx, &(p_start, p_end, p_step, n_start, n_end, n_step)) in
            test_configs.iter().enumerate()
        {
            let output = StdDevBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .nbdev_range(n_start, n_end, n_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
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
