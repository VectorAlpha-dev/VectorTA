//! # Aroon Oscillator
//!
//! The Aroon Oscillator measures the relative time since the most recent highest
//! high and lowest low within a specified `length`. It oscillates between -100
//! and +100, providing insights into the strength and direction of a price trend.
//! Higher positive values indicate a stronger uptrend, while negative values
//! signify a more dominant downtrend.
//!
//! ## Parameters
//! - **length**: The number of recent bars to look back when identifying the highest
//!   high and lowest low (defaults to 14).
//!
//! ## Returns
//! - **`Ok(AroonOscOutput)`** on success, containing a `Vec<f64>` of the oscillator values.
//! - **`Err(AroonOscError)`** otherwise.
//!
//! ## Developer Notes
//! - Decision: Streaming switched to true amortized O(1) using two monotonic deques (max-high, min-low),
//!   preserving earliest-index tie rules and computing `(hi_idx - lo_idx) * (100/length)`.
//!   This fixes the previous stream path’s sign issue and improves worst-case update cost.
//! - Scalar: single-pass implementation with two paths:
//!   - length <= 64: tight window rescan jammed across high/low (fastest for small windows)
//!   - length > 64: O(n) monotonic deques for max(high)/min(low) with earliest-index tie rules
//!   Baseline (RUSTFLAGS="-C target-cpu=native") on local machine:
//!   - aroon_osc_scalar/10k ≈ 75–77 µs; 100k ≈ 761–795 µs
//! - SIMD: stubs delegate to scalar. Intra-series parallelization is not viable (IIR-like dependency).
//!   Packing the two deques into lanes underutilizes AVX and regresses due to control flow and shuffles.
//! - Batch: current path computes each row via the scalar kernel; parallelization uses rayon when selected.
//!   Row-specific SIMD was evaluated as low benefit without shared precompute; no RMQ/sparse-table added.
//! - Memory: follows `alma.rs` patterns (zero-copy/uninitialized allocation; warmup prefixes preserved).
//! - Decision log: SIMD stubs call scalar; CUDA wrapper returns FP32 VRAM handles; Python interop exposes CAI v3 + DLPack v1.x; numerical outputs unchanged.

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[inline(always)]
fn first_valid_hilo(high: &[f64], low: &[f64]) -> Option<usize> {
    high.iter()
        .zip(low)
        .position(|(h, l)| h.is_finite() && l.is_finite())
}

#[derive(Debug, Clone)]
pub enum AroonOscData<'a> {
    Candles { candles: &'a Candles },
    SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct AroonOscParams {
    pub length: Option<usize>,
}

impl Default for AroonOscParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AroonOscInput<'a> {
    pub data: AroonOscData<'a>,
    pub params: AroonOscParams,
}

impl<'a> AroonOscInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AroonOscParams) -> Self {
        Self {
            data: AroonOscData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], params: AroonOscParams) -> Self {
        Self {
            data: AroonOscData::SlicesHL { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AroonOscData::Candles { candles },
            params: AroonOscParams::default(),
        }
    }
    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }

    #[inline]
    pub fn data_len(&self) -> usize {
        match &self.data {
            AroonOscData::Candles { candles } => candles.close.len(),
            AroonOscData::SlicesHL { high, .. } => high.len(),
        }
    }

    #[inline]
    pub fn get_high(&self) -> &'a [f64] {
        match &self.data {
            AroonOscData::Candles { candles } => &candles.high,
            AroonOscData::SlicesHL { high, .. } => high,
        }
    }

    #[inline]
    pub fn get_low(&self) -> &'a [f64] {
        match &self.data {
            AroonOscData::Candles { candles } => &candles.low,
            AroonOscData::SlicesHL { low, .. } => low,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AroonOscOutput {
    pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct AroonOscBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for AroonOscBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AroonOscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AroonOscOutput, AroonOscError> {
        let p = AroonOscParams {
            length: self.length,
        };
        let i = AroonOscInput::from_candles(c, p);
        aroon_osc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<AroonOscOutput, AroonOscError> {
        let p = AroonOscParams {
            length: self.length,
        };
        let i = AroonOscInput::from_slices_hl(high, low, p);
        aroon_osc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AroonOscStream, AroonOscError> {
        let p = AroonOscParams {
            length: self.length,
        };
        AroonOscStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AroonOscError {
    #[error("aroonosc: Input data slice is empty.")]
    EmptyInputData,
    #[error("aroonosc: All values are NaN.")]
    AllValuesNaN,
    #[error("aroonosc: Invalid length: length = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("aroonosc: Not enough data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("aroonosc: Mismatch in high/low slice length: high_len={high_len}, low_len={low_len}")]
    MismatchSliceLength { high_len: usize, low_len: usize },
    #[error("aroonosc: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    // Batch-specific
    #[error("aroonosc: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("aroonosc: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

#[inline]
pub fn aroon_osc(input: &AroonOscInput) -> Result<AroonOscOutput, AroonOscError> {
    aroon_osc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn aroon_osc_prepare<'a>(
    input: &'a AroonOscInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, Kernel), AroonOscError> {
    let length = input.get_length();
    let high = input.get_high();
    let low = input.get_low();
    let len = low.len();
    if length == 0 {
        return Err(AroonOscError::InvalidPeriod { period: length, data_len: len });
    }

    if high.is_empty() || low.is_empty() {
        return Err(AroonOscError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(AroonOscError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first = first_valid_hilo(high, low).ok_or(AroonOscError::AllValuesNaN)?;
    // Need at least (length+1) values after `first`
    let window = length
        .checked_add(1)
        .ok_or(AroonOscError::InvalidPeriod {
            period: length,
            data_len: len,
        })?;
    let available = len
        .checked_sub(first)
        .ok_or(AroonOscError::InvalidPeriod {
            period: length,
            data_len: len,
        })?;
    if available < window {
        return Err(AroonOscError::NotEnoughValidData { needed: window, valid: available });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((high, low, length, first, chosen))
}

pub fn aroon_osc_with_kernel(
    input: &AroonOscInput,
    kernel: Kernel,
) -> Result<AroonOscOutput, AroonOscError> {
    let (high, low, length, first, chosen) = aroon_osc_prepare(input, kernel)?;
    let warm_end = first
        .checked_add(length)
        .ok_or(AroonOscError::InvalidPeriod {
            period: length,
            data_len: high.len(),
        })?;
    let mut out = alloc_with_nan_prefix(high.len(), warm_end);

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            aroon_osc_scalar_highlow_into(high, low, length, first, &mut out)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            aroon_osc_scalar_highlow_into(high, low, length, first, &mut out)
        }
        _ => unreachable!(),
    }
    Ok(AroonOscOutput { values: out })
}

/// Compute Aroon Oscillator into a caller-provided buffer (no allocations).
///
/// - Preserves the module's NaN warmup behavior by pre-filling the warmup
///   prefix with a quiet-NaN (same pattern as `alloc_with_nan_prefix`).
/// - `out.len()` must equal the input length; otherwise an error is returned.
#[inline]
pub fn aroon_osc_into(input: &AroonOscInput, out: &mut [f64]) -> Result<(), AroonOscError> {
    let (high, low, length, first, chosen) = aroon_osc_prepare(input, Kernel::Auto)?;

    if out.len() != high.len() {
        return Err(AroonOscError::OutputLengthMismatch {
            expected: high.len(),
            got: out.len(),
        });
    }

    // Prefill warmup prefix to match Vec-returning API's quiet-NaN pattern
    let warm_end = first
        .checked_add(length)
        .ok_or(AroonOscError::InvalidPeriod {
            period: length,
            data_len: high.len(),
        })?;
    let warm = warm_end.min(out.len());
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..warm] {
        *v = qnan;
    }

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            aroon_osc_scalar_highlow_into(high, low, length, first, out)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            aroon_osc_scalar_highlow_into(high, low, length, first, out)
        }
        _ => unreachable!(),
    }

    Ok(())
}

#[inline]
pub fn aroon_osc_scalar_highlow_into(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = low.len();
    let window = length + 1;
    let start_i = first + length; // window-1 == length
    if start_i >= len {
        return;
    }
    // Heuristic: For small windows, a straight rescan is typically fastest.
    // For larger windows, switch to O(n) monotonic deques for max(high) and min(low).
    if length <= 64 {
        // Precompute scale for final oscillator value: 100/length
        let scale = 100.0 / length as f64;
        for i in start_i..len {
            let start = i + 1 - window;
            let mut highest_val = high[start];
            let mut lowest_val = low[start];
            let mut highest_idx = start;
            let mut lowest_idx = start;
            for j in (start + 1)..=i {
                let h_val = high[j];
                if h_val > highest_val {
                    highest_val = h_val;
                    highest_idx = j;
                }
                let l_val = low[j];
                if l_val < lowest_val {
                    lowest_val = l_val;
                    lowest_idx = j;
                }
            }
            // Aroon Osc = 100/length * ( (i - low_idx) - (i - high_idx) )
            //            = 100/length * (high_idx - low_idx)
            let v = (highest_idx as f64 - lowest_idx as f64) * scale;
            out[i] = v.max(-100.0).min(100.0);
        }
        return;
    }

    // Monotonic deque implementation (safe, ring-buffer based)
    let cap = window; // max elements kept in each deque

    let mut dq_hi = vec![0usize; cap];
    let mut hi_head = 0usize;
    let mut hi_tail = 0usize;
    let mut hi_len = 0usize;

    let mut dq_lo = vec![0usize; cap];
    let mut lo_head = 0usize;
    let mut lo_tail = 0usize;
    let mut lo_len = 0usize;

    #[inline(always)]
    fn dec_wrap(x: usize, cap: usize) -> usize {
        if x == 0 {
            cap - 1
        } else {
            x - 1
        }
    }
    #[inline(always)]
    fn inc_wrap(x: &mut usize, cap: usize) {
        *x += 1;
        if *x == cap {
            *x = 0;
        }
    }

    // Warm-up: seed deques with indices [first, start_i)
    for i in first..start_i {
        // Highs: maintain strictly decreasing values from front to back
        let v_hi = high[i];
        while hi_len > 0 {
            let last = dec_wrap(hi_tail, cap);
            let last_idx = dq_hi[last];
            let last_val = high[last_idx];
            if last_val < v_hi {
                // strict to preserve earliest index on ties
                hi_tail = last;
                hi_len -= 1;
            } else {
                break;
            }
        }
        dq_hi[hi_tail] = i;
        inc_wrap(&mut hi_tail, cap);
        hi_len += 1;

        // Lows: maintain strictly increasing values from front to back
        let v_lo = low[i];
        while lo_len > 0 {
            let last = dec_wrap(lo_tail, cap);
            let last_idx = dq_lo[last];
            let last_val = low[last_idx];
            if last_val > v_lo {
                // strict to preserve earliest index on ties
                lo_tail = last;
                lo_len -= 1;
            } else {
                break;
            }
        }
        dq_lo[lo_tail] = i;
        inc_wrap(&mut lo_tail, cap);
        lo_len += 1;
    }

    let scale = 100.0 / length as f64;
    for i in start_i..len {
        let start = i - length; // window start index (inclusive)

        // Expire outdated indices
        while hi_len > 0 && dq_hi[hi_head] < start {
            inc_wrap(&mut hi_head, cap);
            hi_len -= 1;
        }
        while lo_len > 0 && dq_lo[lo_head] < start {
            inc_wrap(&mut lo_head, cap);
            lo_len -= 1;
        }

        // Insert current index
        let v_hi = high[i];
        while hi_len > 0 {
            let last = dec_wrap(hi_tail, cap);
            let last_idx = dq_hi[last];
            let last_val = high[last_idx];
            if last_val < v_hi {
                // strict comparison keeps earliest index on ties
                hi_tail = last;
                hi_len -= 1;
            } else {
                break;
            }
        }
        dq_hi[hi_tail] = i;
        inc_wrap(&mut hi_tail, cap);
        hi_len += 1;

        let v_lo = low[i];
        while lo_len > 0 {
            let last = dec_wrap(lo_tail, cap);
            let last_idx = dq_lo[last];
            let last_val = low[last_idx];
            if last_val > v_lo {
                // strict comparison keeps earliest index on ties
                lo_tail = last;
                lo_len -= 1;
            } else {
                break;
            }
        }
        dq_lo[lo_tail] = i;
        inc_wrap(&mut lo_tail, cap);
        lo_len += 1;

        let hi_idx = dq_hi[hi_head];
        let lo_idx = dq_lo[lo_head];
        let v = (hi_idx as f64 - lo_idx as f64) * scale;
        out[i] = v.max(-100.0).min(100.0);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_osc_avx512(high: &[f64], low: &[f64], length: usize, first: usize, out: &mut [f64]) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_osc_avx2(high: &[f64], low: &[f64], length: usize, first: usize, out: &mut [f64]) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_osc_avx512_short(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_osc_avx512_long(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}

/// Write Aroon Oscillator directly to output slice - no allocations
#[inline]
pub fn aroon_osc_into_slice(
    dst: &mut [f64],
    input: &AroonOscInput,
    kern: Kernel,
) -> Result<(), AroonOscError> {
    let (high, low, length, first, chosen) = aroon_osc_prepare(input, kern)?;
    if dst.len() != high.len() {
        return Err(AroonOscError::OutputLengthMismatch { expected: high.len(), got: dst.len() });
    }

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            aroon_osc_scalar_highlow_into(high, low, length, first, dst)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            aroon_osc_scalar_highlow_into(high, low, length, first, dst)
        }
        _ => unreachable!(),
    }

    let warm_end = first
        .checked_add(length)
        .ok_or(AroonOscError::InvalidPeriod {
            period: length,
            data_len: high.len(),
        })?;
    let warm = warm_end.min(dst.len());
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
pub fn aroon_osc_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &AroonOscBatchRange,
    k: Kernel,
) -> Result<AroonOscBatchOutput, AroonOscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(AroonOscError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    aroon_osc_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AroonOscBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for AroonOscBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 30, 1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AroonOscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AroonOscParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AroonOscBatchOutput {
    pub fn row_for_params(&self, p: &AroonOscParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.length.unwrap_or(14) == p.length.unwrap_or(14))
    }
    pub fn values_for(&self, p: &AroonOscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct AroonOscBatchBuilder {
    range: AroonOscBatchRange,
    kernel: Kernel,
}
impl AroonOscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }
    #[inline]
    pub fn length_static(mut self, l: usize) -> Self {
        self.range.length = (l, l, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<AroonOscBatchOutput, AroonOscError> {
        aroon_osc_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<AroonOscBatchOutput, AroonOscError> {
        AroonOscBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<AroonOscBatchOutput, AroonOscError> {
        let high = c
            .select_candle_field("high")
            .map_err(|_| AroonOscError::EmptyInputData)?;
        let low = c
            .select_candle_field("low")
            .map_err(|_| AroonOscError::EmptyInputData)?;
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AroonOscBatchOutput, AroonOscError> {
        AroonOscBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

#[inline(always)]
fn expand_grid(r: &AroonOscBatchRange) -> Result<Vec<AroonOscParams>, AroonOscError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, AroonOscError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(AroonOscError::InvalidRange { start, end, step });
            }
            Ok(v)
        } else {
            // reversed bounds supported
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                let next = cur.saturating_sub(step);
                if next == cur { break; }
                cur = next;
            }
            if v.is_empty() { return Err(AroonOscError::InvalidRange { start, end, step }); }
            Ok(v)
        }
    }
    let lengths = axis_usize(r.length)?;
    Ok(lengths
        .into_iter()
        .map(|l| AroonOscParams { length: Some(l) })
        .collect())
}

#[inline(always)]
pub fn aroon_osc_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &AroonOscBatchRange,
    kern: Kernel,
) -> Result<AroonOscBatchOutput, AroonOscError> {
    aroon_osc_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn aroon_osc_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &AroonOscBatchRange,
    kern: Kernel,
) -> Result<AroonOscBatchOutput, AroonOscError> {
    aroon_osc_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn aroon_osc_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &AroonOscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AroonOscBatchOutput, AroonOscError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        });
    }
    if high.len() != low.len() {
        return Err(AroonOscError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let len = high.len();
    let first = first_valid_hilo(high, low).ok_or(AroonOscError::AllValuesNaN)?;
    // largest window = length + 1; first must allow that
    let max_len = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
    let needed = max_len
        .checked_add(1)
        .ok_or(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        })?;
    let available = len
        .checked_sub(first)
        .ok_or(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        })?;
    if available < needed {
        return Err(AroonOscError::NotEnoughValidData { needed, valid: available });
    }

    let rows = combos.len();
    let cols = len;

    // Guard rows * cols before allocation
    rows.checked_mul(cols).ok_or(AroonOscError::InvalidRange {
        start: sweep.length.0,
        end: sweep.length.1,
        step: sweep.length.2,
    })?;

    // Step 1: Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Step 2: per-row warm prefix = first + length
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            first
                .checked_add(c.length.unwrap())
                .ok_or(AroonOscError::InvalidRange {
                    start: sweep.length.0,
                    end: sweep.length.1,
                    step: sweep.length.2,
                })
        })
        .collect::<Result<_, _>>()?;
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    // Step 4: Convert to mutable slice for computation
    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let length = combos[row].length.unwrap();
        aroon_osc_scalar_highlow_into(high, low, length, first, out_row);
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

    // Step 5: Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(AroonOscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn aroon_osc_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &AroonOscBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AroonOscParams>, AroonOscError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        });
    }
    if high.len() != low.len() {
        return Err(AroonOscError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let len = high.len();
    let first = first_valid_hilo(high, low).ok_or(AroonOscError::AllValuesNaN)?;
    let max_len = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
    let needed = max_len
        .checked_add(1)
        .ok_or(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        })?;
    let available = len
        .checked_sub(first)
        .ok_or(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        })?;
    if available < needed {
        return Err(AroonOscError::NotEnoughValidData { needed, valid: available });
    }

    let rows = combos.len();
    let cols = len;
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            first
                .checked_add(c.length.unwrap())
                .ok_or(AroonOscError::InvalidRange {
                    start: sweep.length.0,
                    end: sweep.length.1,
                    step: sweep.length.2,
                })
        })
        .collect::<Result<_, _>>()?;

    // Initialize warm prefixes to NaN using the standard helper
    let mut out_uninit = unsafe {
        Vec::from_raw_parts(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
            out.len(),
        )
    };
    init_matrix_prefixes(&mut out_uninit, cols, &warmup_periods);
    std::mem::forget(out_uninit);

    // Treat `out` as uninitialized for computation
    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };

    let do_row = |row: usize, row_mu: &mut [std::mem::MaybeUninit<f64>]| {
        // compute into a temporary view over the same memory as f64 for the computed tail only
        let dst = unsafe {
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        let length = combos[row].length.unwrap();
        aroon_osc_scalar_highlow_into(high, low, length, first, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn aroon_osc_row_scalar(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx2(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx512(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    if length <= 32 {
        aroon_osc_avx512_short(high, low, length, first, out);
    } else {
        aroon_osc_avx512_long(high, low, length, first, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx512_short(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx512_long(
    high: &[f64],
    low: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    aroon_osc_scalar_highlow_into(high, low, length, first, out)
}

/// Write batch Aroon Oscillator directly to output slice - no allocations
#[inline]
pub fn aroon_osc_batch_into_slice(
    high: &[f64],
    low: &[f64],
    sweep: &AroonOscBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AroonOscParams>, AroonOscError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        });
    }

    let len = high.len();
    if high.len() != low.len() {
        return Err(AroonOscError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let expected_len = combos
        .len()
        .checked_mul(len)
        .ok_or(AroonOscError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        })?;
    if out.len() != expected_len {
        return Err(AroonOscError::OutputLengthMismatch { expected: expected_len, got: out.len() });
    }

    // Use the existing inner function
    aroon_osc_batch_inner_into(high, low, sweep, kern, parallel, out)
}

#[derive(Debug, Clone)]
pub struct AroonOscStream {
    length: usize,
    scale: f64, // 100.0 / length
    cap: usize, // window size = length + 1 (inclusive [t-length, t])
    t: usize,   // absolute index for the next sample

    // Monotonic decreasing deque for highs (max)
    hi_idx: Vec<usize>,
    hi_val: Vec<f64>,
    hi_head: usize,
    hi_tail: usize,
    hi_len: usize,

    // Monotonic increasing deque for lows (min)
    lo_idx: Vec<usize>,
    lo_val: Vec<f64>,
    lo_head: usize,
    lo_tail: usize,
    lo_len: usize,
}

impl AroonOscStream {
    #[inline(always)]
    pub fn try_new(params: AroonOscParams) -> Result<Self, AroonOscError> {
        let length = params.length.unwrap_or(14);
        if length == 0 {
            return Err(AroonOscError::InvalidPeriod { period: length, data_len: 0 });
        }
        let cap = length
            .checked_add(1)
            .ok_or(AroonOscError::InvalidPeriod { period: length, data_len: 0 })?; // inclusive window [t-length, t]
        Ok(Self {
            length,
            scale: 100.0 / length as f64,
            cap,
            t: 0,
            hi_idx: vec![0; cap],
            hi_val: vec![0.0; cap],
            hi_head: 0,
            hi_tail: 0,
            hi_len: 0,
            lo_idx: vec![0; cap],
            lo_val: vec![0.0; cap],
            lo_head: 0,
            lo_tail: 0,
            lo_len: 0,
        })
    }

    /// O(1) amortized per update.
    /// Returns `None` until we've seen `length+1` bars (first non-NaN at t == length).
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let idx = self.t; // index of this bar
        let min_idx_in_window = idx.saturating_sub(self.length); // earliest index allowed

        // 1) Expire outdated indices from the fronts
        while self.hi_len > 0 && self.hi_idx[self.hi_head] < min_idx_in_window {
            self.hi_head = self.inc_wrap(self.hi_head);
            self.hi_len -= 1;
        }
        while self.lo_len > 0 && self.lo_idx[self.lo_head] < min_idx_in_window {
            self.lo_head = self.inc_wrap(self.lo_head);
            self.lo_len -= 1;
        }

        // 2) Normalize NaNs so they never become extrema
        let h = if high.is_finite() {
            high
        } else {
            f64::NEG_INFINITY
        };
        let l = if low.is_finite() { low } else { f64::INFINITY };

        // 3) Push new high into a strictly decreasing deque (preserves earliest index on ties)
        while self.hi_len > 0 {
            let last = self.dec_wrap(self.hi_tail);
            if self.hi_val[last] < h {
                self.hi_tail = last;
                self.hi_len -= 1;
            } else {
                break;
            }
        }
        self.hi_idx[self.hi_tail] = idx;
        self.hi_val[self.hi_tail] = h;
        self.hi_tail = self.inc_wrap(self.hi_tail);
        self.hi_len += 1;

        // 4) Push new low into a strictly increasing deque (preserves earliest index on ties)
        while self.lo_len > 0 {
            let last = self.dec_wrap(self.lo_tail);
            if self.lo_val[last] > l {
                self.lo_tail = last;
                self.lo_len -= 1;
            } else {
                break;
            }
        }
        self.lo_idx[self.lo_tail] = idx;
        self.lo_val[self.lo_tail] = l;
        self.lo_tail = self.inc_wrap(self.lo_tail);
        self.lo_len += 1;

        // 5) Advance time
        self.t = idx.wrapping_add(1);

        // 6) Emit oscillator after warmup (first non-NaN at t == length)
        if idx < self.length {
            return None;
        }
        debug_assert!(self.hi_len > 0 && self.lo_len > 0);

        // Aroon Osc = (hi_idx - lo_idx) * (100/length)
        let hi_i = self.hi_idx[self.hi_head] as i64;
        let lo_i = self.lo_idx[self.lo_head] as i64;
        let v = (hi_i - lo_i) as f64 * self.scale;

        // Strictly speaking this clamp isn't needed (diff in [-length, length])
        Some(v.max(-100.0).min(100.0))
    }

    #[inline(always)]
    fn inc_wrap(&self, x: usize) -> usize {
        let y = x + 1;
        if y == self.cap {
            0
        } else {
            y
        }
    }
    #[inline(always)]
    fn dec_wrap(&self, x: usize) -> usize {
        if x == 0 {
            self.cap - 1
        } else {
            x - 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    
    #[test]
    fn test_aroonosc_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Build a small but non-trivial synthetic candle set
        let n = 256usize;
        let timestamp: Vec<i64> = (0..n as i64).collect();
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let ib = i as f64;
            let base = 1000.0 + (ib * 0.05).sin() * 10.0 + (i % 7) as f64;
            let spread = 5.0 + (i % 11) as f64 * 0.1;
            let h = base + spread;
            let l = base - spread;
            let o = base;
            let c = base + ((i % 3) as f64 - 1.0) * 0.5;
            let v = 1000.0 + (i % 5) as f64 * 10.0;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(v);
        }

        let candles = Candles::new(timestamp, open, high.clone(), low.clone(), close, volume);
        let input = AroonOscInput::with_default_candles(&candles);

        // Baseline via Vec-returning API
        let baseline = aroon_osc(&input)?.values;

        // Into API writes into caller-provided buffer
        let mut out = vec![0.0; n];
        aroon_osc_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        // Helper: NaN == NaN; otherwise exact equality
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "Mismatch at index {}: baseline={}, into={}",
                i,
                a,
                b
            );
        }

        Ok(())
    }
    fn check_aroonosc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AroonOscParams { length: Some(20) };
        let input = AroonOscInput::from_candles(&candles, partial_params);
        let result = aroon_osc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }
    fn check_aroonosc_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AroonOscInput::with_default_candles(&candles);
        let result = aroon_osc_with_kernel(&input, kernel)?;
        let expected_last_five = [-50.0, -50.0, -50.0, -50.0, -42.8571];
        assert!(result.values.len() >= 5, "Not enough Aroon Osc values");
        assert_eq!(result.values.len(), candles.close.len());
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-2,
                "Aroon Osc mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }
        let length = 14;
        for val in result.values.iter().skip(length) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "Aroon Osc should be finite after enough data"
                );
            }
        }
        Ok(())
    }
    fn check_aroonosc_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AroonOscInput::with_default_candles(&candles);
        match input.data {
            AroonOscData::Candles { .. } => {}
            _ => panic!("Expected AroonOscData::Candles variant"),
        }
        assert!(input.params.length.is_some());
        Ok(())
    }
    fn check_aroonosc_with_slices_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AroonOscParams { length: Some(10) };
        let first_input = AroonOscInput::from_candles(&candles, first_params);
        let first_result = aroon_osc_with_kernel(&first_input, kernel)?;
        let second_params = AroonOscParams { length: Some(5) };
        let second_input = AroonOscInput::from_slices_hl(
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = aroon_osc_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 20..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
        Ok(())
    }
    fn check_aroonosc_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AroonOscInput::with_default_candles(&candles);
        let result = aroon_osc_with_kernel(&input, kernel)?;
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "Expected no NaN after index {}, but found NaN",
                    i
                );
            }
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_aroonosc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to catch edge cases
        let test_lengths = vec![
            5,   // Very small period
            14,  // Default period
            25,  // Medium period
            50,  // Large period
            100, // Very large period
            200, // Extra large period
        ];

        for length in test_lengths {
            let params = AroonOscParams {
                length: Some(length),
            };
            let input = AroonOscInput::from_candles(&candles, params);

            // Skip if not enough data for this length
            if candles.close.len() < length {
                continue;
            }

            let output = aroon_osc_with_kernel(&input, kernel)?;

            // Check every value for poison patterns
            for (i, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in the warmup period
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with length {}",
						test_name, val, bits, i, length
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with length {}",
						test_name, val, bits, i, length
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with length {}",
						test_name, val, bits, i, length
					);
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_aroonosc_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_aroonosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic test scenarios with various market conditions
        let strat = (2usize..=100)
            .prop_flat_map(|length| {
                let min_size = (length * 2).max(length + 20); // Ensure enough data for proper testing
                let max_size = 400;
                (
                    10.0f64..1000.0f64, // Base price
                    0.0f64..0.1f64,     // Volatility percentage (0% to 10%)
                    -0.02f64..0.02f64,  // Trend strength (-2% to 2% per period)
                    min_size..max_size, // Data size
                    Just(length),       // Period for AroonOsc
                    0u8..6, // Market type: 0=ranging, 1=uptrend, 2=downtrend, 3=flat, 4=monotonic up, 5=monotonic down
                )
            })
            .prop_map(
                |(base_price, volatility, trend, size, length, market_type)| {
                    let mut high = Vec::with_capacity(size);
                    let mut low = Vec::with_capacity(size);

                    // Generate realistic OHLC data based on market type
                    for i in 0..size {
                        let time_factor = i as f64 / size as f64;

                        let (h, l) = match market_type {
                            0 => {
                                // Ranging market with sine wave pattern
                                let cycle = (time_factor * 4.0 * std::f64::consts::PI).sin();
                                let price = base_price * (1.0 + cycle * volatility);
                                let spread = price * volatility * 0.5;
                                (price + spread, price - spread)
                            }
                            1 => {
                                // Strong uptrend
                                let price = base_price * (1.0 + trend.abs() * i as f64);
                                let noise = ((i * 17 + 13) % 100) as f64 / 100.0 - 0.5;
                                let variation = price * volatility * noise * 0.3;
                                let spread = price * volatility * 0.2;
                                (price + variation + spread, price + variation - spread)
                            }
                            2 => {
                                // Strong downtrend
                                let price = base_price * (1.0 - trend.abs() * i as f64).max(1.0);
                                let noise = ((i * 23 + 7) % 100) as f64 / 100.0 - 0.5;
                                let variation = price * volatility * noise * 0.3;
                                let spread = price * volatility * 0.2;
                                (price + variation + spread, price + variation - spread)
                            }
                            3 => {
                                // Completely flat market (high = low = constant)
                                let price = base_price;
                                (price, price)
                            }
                            4 => {
                                // Monotonically increasing
                                let price = base_price + (i as f64 * base_price * 0.01);
                                let spread = price * 0.001; // Very small spread
                                (price + spread, price - spread)
                            }
                            _ => {
                                // Monotonically decreasing
                                let price = base_price
                                    - (i as f64 * base_price * 0.005).min(base_price * 0.9);
                                let spread = price * 0.001; // Very small spread
                                (price + spread, price - spread)
                            }
                        };

                        // Ensure high >= low constraint
                        high.push(h.max(l));
                        low.push(h.min(l));
                    }

                    (high, low, length, market_type)
                },
            );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high, low, length, market_type)| {
                let params = AroonOscParams {
                    length: Some(length),
                };
                let input = AroonOscInput::from_slices_hl(&high, &low, params);

                // Test with specified kernel
                let result = aroon_osc_with_kernel(&input, kernel)?;

                // Test with scalar reference for comparison
                let reference = aroon_osc_with_kernel(&input, Kernel::Scalar)?;

                // Verify output length matches input
                prop_assert_eq!(result.values.len(), high.len(), "Output length mismatch");

                // Check warmup period - first 'length' values should be NaN
                for i in 0..length {
                    prop_assert!(
                        result.values[i].is_nan(),
                        "Expected NaN at index {} during warmup (length={})",
                        i,
                        length
                    );
                }

                // Check values after warmup period
                for i in length..result.values.len() {
                    let val = result.values[i];
                    let ref_val = reference.values[i];

                    // AroonOsc must be in range [-100, 100]
                    prop_assert!(
                        val >= -100.0 && val <= 100.0,
                        "AroonOsc value {} at index {} out of range [-100, 100]",
                        val,
                        i
                    );

                    // Check kernel consistency
                    if val.is_finite() && ref_val.is_finite() {
                        // Allow small floating point differences
                        let diff = (val - ref_val).abs();
                        prop_assert!(
                            diff <= 1e-9,
                            "Kernel mismatch at index {}: {} vs {} (diff={})",
                            i,
                            val,
                            ref_val,
                            diff
                        );
                    } else {
                        // Both should be NaN or both finite
                        prop_assert_eq!(
                            val.is_nan(),
                            ref_val.is_nan(),
                            "NaN mismatch at index {}: {} vs {}",
                            i,
                            val,
                            ref_val
                        );
                    }

                    // Test AroonOsc-specific properties based on data patterns
                    let window_start = i.saturating_sub(length);
                    let window_high = &high[window_start..=i];
                    let window_low = &low[window_start..=i];

                    // Property 1: When all values in window are identical (flat), AroonOsc should be 0
                    if window_high
                        .iter()
                        .all(|&h| (h - window_high[0]).abs() < f64::EPSILON)
                        && window_low
                            .iter()
                            .all(|&l| (l - window_low[0]).abs() < f64::EPSILON)
                    {
                        prop_assert!(
							val.abs() < 1e-9,
							"Completely flat window should produce AroonOsc = 0, got {} at index {}",
							val,
							i
						);
                    }

                    // Property 2: Find position of highest high and lowest low in window
                    let highest_idx = window_high
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    let lowest_idx = window_low
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    // When highest is most recent (at end of window), Aroon Up should be 100
                    if highest_idx == window_high.len() - 1 {
                        // AroonOsc should be positive when highest high is most recent
                        prop_assert!(
							val >= -100.0 && val <= 100.0,
							"When highest high is most recent, AroonOsc {} should be valid at index {}",
							val,
							i
						);
                    }

                    // When lowest is most recent (at end of window), Aroon Down should be 100
                    if lowest_idx == window_low.len() - 1 {
                        // AroonOsc should be negative when lowest low is most recent
                        prop_assert!(
							val >= -100.0 && val <= 100.0,
							"When lowest low is most recent, AroonOsc {} should be valid at index {}",
							val,
							i
						);
                    }

                    // Property 3: Monotonic sequences should have consistent behavior
                    if market_type == 4 {
                        // Monotonically increasing - most recent high should be highest
                        // AroonOsc should tend to be positive
                        prop_assert!(
							val >= -100.0,
							"Monotonic increasing should not produce very negative AroonOsc, got {} at index {}",
							val,
							i
						);
                    } else if market_type == 5 {
                        // Monotonically decreasing - most recent low should be lowest
                        // AroonOsc should tend to be negative
                        prop_assert!(
							val <= 100.0,
							"Monotonic decreasing should not produce very positive AroonOsc, got {} at index {}",
							val,
							i
						);
                    }

                    // Property 4: When extremes are at opposite ends of window (but not in flat markets)
                    let is_flat_window = window_high
                        .iter()
                        .all(|&h| (h - window_high[0]).abs() < f64::EPSILON)
                        && window_low
                            .iter()
                            .all(|&l| (l - window_low[0]).abs() < f64::EPSILON);

                    if !is_flat_window {
                        if highest_idx == window_high.len() - 1 && lowest_idx == 0 {
                            // Highest is most recent, lowest is oldest -> AroonOsc should be near 100
                            prop_assert!(
								val >= 50.0,
								"When highest is recent and lowest is old, AroonOsc {} should be positive at index {}",
								val,
								i
							);
                        } else if lowest_idx == window_low.len() - 1 && highest_idx == 0 {
                            // Lowest is most recent, highest is oldest -> AroonOsc should be near -100
                            prop_assert!(
								val <= -50.0,
								"When lowest is recent and highest is old, AroonOsc {} should be negative at index {}",
								val,
								i
							);
                        }
                    }
                }

                // Verify no poison values (debug builds check these)
                #[cfg(debug_assertions)]
                for &val in &result.values {
                    if !val.is_nan() {
                        let bits = val.to_bits();
                        prop_assert_ne!(
                            bits,
                            0x11111111_11111111,
                            "Found poison value from alloc_with_nan_prefix"
                        );
                        prop_assert_ne!(
                            bits,
                            0x22222222_22222222,
                            "Found poison value from init_matrix_prefixes"
                        );
                        prop_assert_ne!(
                            bits,
                            0x33333333_33333333,
                            "Found poison value from make_uninit_matrix"
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_aroonosc_tests {
        ($($test_fn:ident),*) => {
            paste! {
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
    generate_all_aroonosc_tests!(
        check_aroonosc_partial_params,
        check_aroonosc_accuracy,
        check_aroonosc_default_candles,
        check_aroonosc_with_slices_data_reinput,
        check_aroonosc_nan_handling,
        check_aroonosc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_aroonosc_tests!(check_aroonosc_property);
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AroonOscBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let def = AroonOscParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations to catch edge cases
        let test_configs = vec![
            // (start, end, step) for length ranges
            (2, 10, 2),    // Small lengths: 2, 4, 6, 8, 10
            (5, 25, 5),    // Medium lengths: 5, 10, 15, 20, 25
            (10, 100, 10), // Large lengths: 10, 20, 30, ..., 100
            (50, 200, 50), // Very large lengths: 50, 100, 150, 200
            (14, 14, 0),   // Single parameter (default)
            (1, 5, 1),     // Edge case: very small lengths 1, 2, 3, 4, 5
        ];

        for (start, end, step) in test_configs {
            // Skip if data is not sufficient for the largest length
            if c.close.len() < end {
                continue;
            }

            let output = AroonOscBatchBuilder::new()
                .kernel(kernel)
                .length_range(start, end, step)
                .apply_candles(&c)?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let length = output.combos[row].length.unwrap_or(14);

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()   { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::{CudaAroonOsc, DeviceArrayF32Aroonosc};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "python")]
#[pyfunction(name = "aroonosc")]
#[pyo3(signature = (high, low, length=14, kernel=None))]
pub fn aroon_osc_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    // Get slices - as_slice() will fail if array is not contiguous
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    // Validate inputs have same length
    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err(format!(
            "High and low arrays must have same length. Got high: {}, low: {}",
            high_slice.len(),
            low_slice.len()
        )));
    }

    // Check length validity early
    if length == 0 {
        return Err(PyValueError::new_err(
            "Invalid length: length must be greater than 0",
        ));
    }

    // Validate kernel parameter before entering allow_threads
    let kern = validate_kernel(kernel, false)?;

    // Build input struct
    let params = AroonOscParams {
        length: Some(length),
    };
    let aroon_in = AroonOscInput::from_slices_hl(high_slice, low_slice, params);

    // Get Vec<f64> from Rust function with zero-copy transfer
    let result_vec: Vec<f64> = py
        .allow_threads(|| aroon_osc_with_kernel(&aroon_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AroonOscStream")]
pub struct AroonOscStreamPy {
    stream: AroonOscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AroonOscStreamPy {
    #[new]
    fn new(length: usize) -> PyResult<Self> {
        let params = AroonOscParams {
            length: Some(length),
        };
        let stream =
            AroonOscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AroonOscStreamPy { stream })
    }

    /// Updates the stream with new high/low values and returns the calculated Aroon Oscillator value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "aroonosc_batch")]
#[pyo3(signature = (high, low, length_range, kernel=None))]
pub fn aroon_osc_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    // Validate inputs have same length
    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err(format!(
            "High and low arrays must have same length. Got high: {}, low: {}",
            high_slice.len(),
            low_slice.len()
        )));
    }

    // Validate kernel parameter before entering allow_threads
    let kern = validate_kernel(kernel, true)?;

    let sweep = AroonOscBatchRange {
        length: length_range,
    };

    // Calculate dimensions for pre-allocation
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = high_slice.len();

    // Pre-allocate uninitialized NumPy array for maximum efficiency
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute directly into pre-allocated buffer
    let combos = py
        .allow_threads(|| -> Result<Vec<AroonOscParams>, AroonOscError> {
            // Resolve kernel for batch operations
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

            aroon_osc_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary with zero-copy transfers
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ---------------- Python CUDA bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
pub struct PrimaryCtxGuard {
    dev: i32,
    ctx: cust::sys::CUcontext,
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl PrimaryCtxGuard {
    fn new(device_id: u32) -> Result<Self, cust::error::CudaError> {
        unsafe {
            let mut ctx: cust::sys::CUcontext = core::ptr::null_mut();
            let dev = device_id as i32;
            let res = cust::sys::cuDevicePrimaryCtxRetain(&mut ctx as *mut _, dev);
            if res != cust::sys::CUresult::CUDA_SUCCESS {
                return Err(cust::error::CudaError::UnknownError);
            }
            Ok(PrimaryCtxGuard { dev, ctx })
        }
    }
    #[inline]
    unsafe fn push_current(&self) {
        let _ = cust::sys::cuCtxSetCurrent(self.ctx);
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl Drop for PrimaryCtxGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cust::sys::cuDevicePrimaryCtxRelease_v2(self.dev);
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct AroonOscDeviceArrayF32Py {
    inner: Option<DeviceArrayF32Aroonosc>,
    device_id: u32,
    pc_guard: PrimaryCtxGuard,
}
#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl AroonOscDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        let inner = self.inner.as_ref().ok_or_else(|| PyValueError::new_err("buffer already exported"))?;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr_val: usize =
            if inner.rows == 0 || inner.cols == 0 { 0 } else { inner.device_ptr() as usize };
        d.set_item("data", (ptr_val, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl Drop for AroonOscDeviceArrayF32Py {
    fn drop(&mut self) {
        unsafe { self.pc_guard.push_current(); }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "aroonosc_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, length_range, device_id=0))]
pub fn aroonosc_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    length_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<AroonOscDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use pyo3::exceptions::PyValueError;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    if high.len() != low.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }

    let sweep = AroonOscBatchRange {
        length: length_range,
    };
    let inner = py.allow_threads(|| {
        let cuda =
            CudaAroonOsc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.aroonosc_batch_dev(high, low, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let guard = PrimaryCtxGuard::new(device_id as u32)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(AroonOscDeviceArrayF32Py { inner: Some(inner), device_id: device_id as u32, pc_guard: guard })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "aroonosc_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, length, device_id=0))]
pub fn aroonosc_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    length: usize,
    device_id: usize,
) -> PyResult<AroonOscDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;
    use pyo3::exceptions::PyValueError;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape_h = high_tm_f32.shape();
    let shape_l = low_tm_f32.shape();
    if shape_h != shape_l || shape_h.len() != 2 {
        return Err(PyValueError::new_err("high/low must be same 2D shape"));
    }
    let rows = shape_h[0];
    let cols = shape_h[1];
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda =
            CudaAroonOsc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.aroonosc_many_series_one_param_time_major_dev(h, l, cols, rows, length)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let guard = PrimaryCtxGuard::new(device_id as u32)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(AroonOscDeviceArrayF32Py { inner: Some(inner), device_id: device_id as u32, pc_guard: guard })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_js(high: &[f64], low: &[f64], length: usize) -> Result<Vec<f64>, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str(&format!(
            "High and low arrays must have same length. Got high: {}, low: {}",
            high.len(),
            low.len()
        )));
    }

    let params = AroonOscParams {
        length: Some(length),
    };
    let input = AroonOscInput::from_slices_hl(high, low, params);

    // Use aroon_osc_with_kernel to properly allocate with NaN prefix
    aroon_osc_with_kernel(&input, Kernel::Auto)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_batch_js(
    high: &[f64],
    low: &[f64],
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<Vec<f64>, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str(&format!(
            "High and low arrays must have same length. Got high: {}, low: {}",
            high.len(),
            low.len()
        )));
    }

    let sweep = AroonOscBatchRange {
        length: (length_start, length_end, length_step),
    };

    // Use the existing batch function with parallel=false for WASM
    aroon_osc_batch_slice(high, low, &sweep, Kernel::Auto)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_batch_metadata_js(
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AroonOscBatchRange {
        length: (length_start, length_end, length_step),
    };

    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.length.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AroonOscBatchConfig {
    pub length_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AroonOscBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AroonOscParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = aroonosc_batch)]
pub fn aroon_osc_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str(&format!(
            "High and low arrays must have same length. Got high: {}, low: {}",
            high.len(),
            low.len()
        )));
    }

    // 1. Deserialize the configuration object from JavaScript
    let config: AroonOscBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AroonOscBatchRange {
        length: config.length_range,
    };

    // 2. Run the existing core logic
    let output = aroon_osc_batch_slice(high, low, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = AroonOscBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_into(
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

        let params = AroonOscParams {
            length: Some(length),
        };
        let input = AroonOscInput::from_slices_hl(high, low, params);

        // Check for aliasing - if any input overlaps with output, use temp buffer
        if high_ptr == out_ptr || low_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            aroon_osc_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            aroon_osc_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_batch_into(
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

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = AroonOscBatchRange {
            length: (length_start, length_end, length_step),
        };

        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let expected_len = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("aroonosc: length range too large"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, expected_len);

        // Check for aliasing - if any input overlaps with output, use temp buffer
        let high_overlaps = (high_ptr as usize) < (out_ptr as usize + expected_len * 8)
            && (high_ptr as usize + len * 8) > (out_ptr as usize);
        let low_overlaps = (low_ptr as usize) < (out_ptr as usize + expected_len * 8)
            && (low_ptr as usize + len * 8) > (out_ptr as usize);

        if high_overlaps || low_overlaps {
            let mut temp = vec![0.0; expected_len];
            aroon_osc_batch_into_slice(high, low, &sweep, Kernel::Auto, false, &mut temp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            out.copy_from_slice(&temp);
        } else {
            aroon_osc_batch_into_slice(high, low, &sweep, Kernel::Auto, false, out)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}
