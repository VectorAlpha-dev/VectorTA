//! # Square Weighted Moving Average (SQWMA)
//!
//! A specialized moving average that applies squared weights to recent data
//! points. Uses `period - 1` data points with weights from `period^2` down to `2^2`.
//! The most recent value receives `(period)^2` weight, and each preceding value's
//! weight decreases quadratically. This approach enhances sensitivity to current
//! price changes while still smoothing out older noise.
//!
//! ## Parameters
//! - **period**: Defines the weight range; uses `period - 1` data points (must be ≥ 2).
//!
//! ## Returns
//! - **`Ok(SqwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SqwmaError)`** otherwise.
//!
//! ## Developer Status
//! - Scalar kernel: Reference O(p) weighted sum for batch path (strict tests preserved).
//! - Batch row kernel: O(1) per-step via exact sliding recurrence; periodic rebase every 64 steps.
//! - AVX2/AVX512: Stubs delegate to scalar/batch (memory-bound; minimal wins expected).
//! - Streaming update: O(p) per update to match the scalar dot-product path exactly (tests require
//!   near bit-for-bit equality). The O(1) recurrence state is retained, but we rebase from the ring
//!   every step to produce identical outputs.
//! - Memory: GOOD — constant ring buffer; no per-tick allocations; no per-tick Vec growth.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaSqwma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SqwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SqwmaData::Slice(slice) => slice,
            SqwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SqwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SqwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SqwmaParams {
    pub period: Option<usize>,
}

impl Default for SqwmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SqwmaInput<'a> {
    pub data: SqwmaData<'a>,
    pub params: SqwmaParams,
}

impl<'a> SqwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SqwmaParams) -> Self {
        Self {
            data: SqwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SqwmaParams) -> Self {
        Self {
            data: SqwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SqwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SqwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SqwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SqwmaBuilder {
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
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SqwmaOutput, SqwmaError> {
        let p = SqwmaParams {
            period: self.period,
        };
        let i = SqwmaInput::from_candles(c, "close", p);
        sqwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SqwmaOutput, SqwmaError> {
        let p = SqwmaParams {
            period: self.period,
        };
        let i = SqwmaInput::from_slice(d, p);
        sqwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SqwmaStream, SqwmaError> {
        let p = SqwmaParams {
            period: self.period,
        };
        SqwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SqwmaError {
    #[error("sqwma: Input data slice is empty.")]
    EmptyInputData,
    #[error("sqwma: All values are NaN.")]
    AllValuesNaN,
    #[error("sqwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("sqwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn sqwma(input: &SqwmaInput) -> Result<SqwmaOutput, SqwmaError> {
    sqwma_with_kernel(input, Kernel::Auto)
}

pub fn sqwma_with_kernel(input: &SqwmaInput, kernel: Kernel) -> Result<SqwmaOutput, SqwmaError> {
    let data: &[f64] = match &input.data {
        SqwmaData::Candles { candles, source } => source_type(candles, source),
        SqwmaData::Slice(sl) => sl,
    };
    let len = data.len();
    if len == 0 {
        return Err(SqwmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SqwmaError::AllValuesNaN)?;
    let period = input.get_period();
    if period < 2 || period > len {
        return Err(SqwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(SqwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    // ─── BUILD EXACTLY (period - 1) WEIGHTS ───
    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period - 1);
    for i in 0..(period - 1) {
        weights.push((period as f64 - i as f64).powi(2));
    }
    let weight_sum: f64 = weights.iter().sum();
    // ──────────────────────────────────────────

    let warm = first + period + 1;
    let mut out = alloc_with_nan_prefix(len, warm);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                sqwma_scalar(data, &weights, period, first, weight_sum, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                sqwma_avx2(data, &weights, period, first, weight_sum, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                sqwma_avx512(data, &weights, period, first, weight_sum, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(SqwmaOutput { values: out })
}

#[inline]
pub fn sqwma_scalar(
    data: &[f64],
    weights: &[f64], // length = period - 1
    period: usize,
    first: usize,
    weight_sum: f64,
    out: &mut [f64],
) {
    let p_minus_1 = period - 1;
    let p4 = p_minus_1 & !3;
    let n = data.len();

    let inv_ws = 1.0 / weight_sum;
    for j in (first + period + 1)..n {
        let mut sum = 0.0;
        let mut k = 0;
        while k < p4 {
            let d0 = data[j - k];
            let d1 = data[j - (k + 1)];
            let d2 = data[j - (k + 2)];
            let d3 = data[j - (k + 3)];
            sum = d0.mul_add(weights[k], sum);
            sum = d1.mul_add(weights[k + 1], sum);
            sum = d2.mul_add(weights[k + 2], sum);
            sum = d3.mul_add(weights[k + 3], sum);
            k += 4;
        }
        while k < p_minus_1 {
            let d = data[j - k];
            sum = d.mul_add(weights[k], sum);
            k += 1;
        }
        out[j] = sum * inv_ws;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sqwma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    weight_sum: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { sqwma_avx512_short(data, weights, period, first, weight_sum, out) }
    } else {
        unsafe { sqwma_avx512_long(data, weights, period, first, weight_sum, out) }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sqwma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    weight_sum: f64,
    out: &mut [f64],
) {
    unsafe { sqwma_scalar(data, weights, period, first, weight_sum, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sqwma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    weight_sum: f64,
    out: &mut [f64],
) {
    sqwma_scalar(data, weights, period, first, weight_sum, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sqwma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    weight_sum: f64,
    out: &mut [f64],
) {
    sqwma_scalar(data, weights, period, first, weight_sum, out)
}

// Decision: O(1) recurrence state is implemented, but to exactly match the
// scalar dot-product path used in tests, we rebase from the ring every step
// (O(p) per update). This preserves strict equality while keeping the O(1)
// logic available for future enablement.
#[derive(Clone, Debug)]
pub struct SqwmaStream {
    // --- config / constants ---
    period: usize,
    weights: Vec<f64>,
    weight_sum: f64,
    inv_weight_sum: f64,
    p_f: f64,
    p2: f64,
    pm1f: f64,
    c1: f64,

    // --- ring buffer of last p values ---
    ring: Vec<f64>,
    tail: usize,
    len: usize,

    // --- running state for O(1) updates over last (p-1) values ---
    a_sum: f64,
    b_sum: f64,
    r_acc: f64,

    // --- control ---
    count: usize,
    seeded: bool,
    rebase_ctr: usize,
}

impl SqwmaStream {
    pub fn try_new(params: SqwmaParams) -> Result<Self, SqwmaError> {
        let period = params.period.unwrap_or(14);
        if period < 2 {
            return Err(SqwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        // Build (p-1) weights: p^2, (p-1)^2, ..., 2^2
        let mut weights = Vec::with_capacity(period - 1);
        for i in 0..(period - 1) {
            weights.push((period as f64 - i as f64).powi(2));
        }
        let weight_sum: f64 = weights.iter().sum();
        let inv_weight_sum = 1.0 / weight_sum;

        Ok(Self {
            period,
            p_f: period as f64,
            p2: (period as f64) * (period as f64),
            pm1f: (period - 1) as f64,
            c1: 1.0 - 2.0 * (period as f64),

            weights,
            weight_sum,
            inv_weight_sum,

            ring: vec![0.0; period],
            tail: 0,
            len: 0,

            a_sum: 0.0,
            b_sum: 0.0,
            r_acc: 0.0,

            count: 0,
            seeded: false,
            rebase_ctr: 0,
        })
    }

    #[inline(always)]
    fn ring_push(&mut self, x: f64) -> Option<f64> {
        // Returns Some(x_out) if the ring was full and an element fell off; otherwise None.
        if self.len < self.period {
            let pos = (self.tail + self.len) % self.period;
            self.ring[pos] = x;
            self.len += 1;
            None
        } else {
            let x_out = self.ring[self.tail];
            self.ring[self.tail] = x;
            self.tail = if self.tail + 1 == self.period {
                0
            } else {
                self.tail + 1
            };
            Some(x_out)
        }
    }

    #[inline(always)]
    fn rebase_from_ring(&mut self) {
        // Recompute A, B, R exactly from the ring for the current j.
        debug_assert!(self.len == self.period);

        let p = self.period;
        let m_len = p - 1; // number of points included in A/B/R
        let mut a = 0.0;
        let mut b = 0.0;
        let mut r = 0.0;

        let mut i = 0usize;
        let u4 = m_len & !3;
        while i < u4 {
            let pos0 = (self.tail + p - 1 - i) % p;
            let pos1 = (self.tail + p - 2 - i) % p;
            let pos2 = (self.tail + p - 3 - i) % p;
            let pos3 = (self.tail + p - 4 - i) % p;

            let x0 = self.ring[pos0];
            let x1 = self.ring[pos1];
            let x2 = self.ring[pos2];
            let x3 = self.ring[pos3];

            a += x0 + x1 + x2 + x3;
            b += (i as f64) * x0
                + ((i + 1) as f64) * x1
                + ((i + 2) as f64) * x2
                + ((i + 3) as f64) * x3;

            r = x0.mul_add(self.weights[i], r);
            r = x1.mul_add(self.weights[i + 1], r);
            r = x2.mul_add(self.weights[i + 2], r);
            r = x3.mul_add(self.weights[i + 3], r);

            i += 4;
        }
        while i < m_len {
            let pos = (self.tail + p - 1 - i) % p;
            let x = self.ring[pos];
            a += x;
            b += (i as f64) * x;
            r = x.mul_add(self.weights[i], r);
            i += 1;
        }

        self.a_sum = a;
        self.b_sum = b;
        self.r_acc = r;
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.count = self.count.wrapping_add(1);

        // Write into the ring buffer first; capture x_out if we overwrote oldest
        let x_out_opt = self.ring_push(value);

        // Match scalar/batch warm-up exactly: first output at index j = p + 1
        // -> in terms of call count, first Some(...) when count >= p + 2.
        if self.count < (self.period + 2) {
            return None;
        }

        // Initial seeding (compute A,B,R exactly from ring at j0)
        if !self.seeded {
            debug_assert!(self.len == self.period);
            self.rebase_from_ring();
            self.seeded = true;
            self.rebase_ctr = 0;
            return Some(self.r_acc * self.inv_weight_sum);
        }

        // O(1) recurrence using previous (A,B,R) and x_out = x_{j - p + 1}.
        // After ring_push(value), tail points at the oldest retained sample,
        // which is exactly x_out needed by the recurrence.
        let x_out = self.ring[self.tail];
        let mut r = self.r_acc;
        r = self.p2.mul_add(value, r); // + p^2 * x_in
        r = (2.0_f64).mul_add(self.b_sum, r); // + 2 * B_prev
        r = self.c1.mul_add(self.a_sum, r); // + (1 - 2p) * A_prev
        r -= x_out; // - x_out
        self.r_acc = r;

        // Update A and B
        let a_prev = self.a_sum;
        self.a_sum = a_prev + value - x_out;
        self.b_sum = self.b_sum + a_prev - self.pm1f * x_out;

        // Rebase every step to ensure exact agreement with the scalar O(p)
        // path used in tests (keeps outputs bit-close to dot-product).
        const REBASE_MASK: usize = 0; // always rebase
        self.rebase_ctr = self.rebase_ctr.wrapping_add(1);
        if (self.rebase_ctr & REBASE_MASK) == 0 {
            self.rebase_from_ring();
        }

        Some(self.r_acc * self.inv_weight_sum)
    }
}

#[derive(Clone, Debug)]
pub struct SqwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SqwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SqwmaBatchBuilder {
    range: SqwmaBatchRange,
    kernel: Kernel,
}

impl SqwmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<SqwmaBatchOutput, SqwmaError> {
        sqwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SqwmaBatchOutput, SqwmaError> {
        SqwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SqwmaBatchOutput, SqwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<SqwmaBatchOutput, SqwmaError> {
        SqwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn sqwma_batch_with_kernel(
    data: &[f64],
    sweep: &SqwmaBatchRange,
    k: Kernel,
) -> Result<SqwmaBatchOutput, SqwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SqwmaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    sqwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SqwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SqwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SqwmaBatchOutput {
    pub fn row_for_params(&self, p: &SqwmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &SqwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SqwmaBatchRange) -> Vec<SqwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SqwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
fn sqwma_build_flat_weights(combos: &[SqwmaParams], max_p: usize) -> (AVec<f64>, Vec<f64>) {
    let rows = combos.len();
    let cap = rows * max_p;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);
    let mut sums = vec![0.0; rows];

    for (row, prm) in combos.iter().enumerate() {
        let p = prm.period.unwrap();
        for i in 0..(p - 1) {
            flat_w[row * max_p + i] = (p as f64 - i as f64).powi(2);
        }
        let s = &flat_w[row * max_p..row * max_p + (p - 1)];
        sums[row] = s.iter().sum();
    }
    (flat_w, sums)
}

#[inline(always)]
fn sqwma_batch_inner_into(
    data: &[f64],
    combos: &[SqwmaParams],
    kern: Kernel,
    first: usize,
    max_p: usize,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), SqwmaError> {
    let rows = combos.len();
    let cols = data.len();
    let (flat_w, sums) = sqwma_build_flat_weights(combos, max_p);

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => actual,
    };

    let work = |row: usize, dst: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let w_sum = *sums.get_unchecked(row);
        match simd {
            Kernel::Scalar => sqwma_row_scalar(data, first, p, p - 1, w_ptr, w_sum, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => sqwma_row_avx2(data, first, p, p - 1, w_ptr, w_sum, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => sqwma_row_avx512(data, first, p, p - 1, w_ptr, w_sum, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| work(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out.chunks_mut(cols).enumerate() {
            work(r, s);
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            work(r, s);
        }
    }
    Ok(())
}

#[inline(always)]
pub fn sqwma_batch_slice(
    data: &[f64],
    sweep: &SqwmaBatchRange,
    kern: Kernel,
) -> Result<SqwmaBatchOutput, SqwmaError> {
    sqwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn sqwma_batch_par_slice(
    data: &[f64],
    sweep: &SqwmaBatchRange,
    kern: Kernel,
) -> Result<SqwmaBatchOutput, SqwmaError> {
    sqwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn sqwma_batch_inner(
    data: &[f64],
    sweep: &SqwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SqwmaBatchOutput, SqwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SqwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let cols = data.len();
    if cols == 0 {
        return Err(SqwmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SqwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(SqwmaError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }
    let rows = combos.len();

    // Allocate uninit matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Warm prefixes per row (SQWMA warm = first + period + 1)
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| (first + c.period.unwrap() + 1).min(cols))
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // View as &mut [f64] for compute
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Fill rows
    sqwma_batch_inner_into(data, &combos, kern, first, max_p, parallel, out)?;

    // Take ownership of initialized f64 buffer
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(SqwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn sqwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    p_minus_1: usize,  // = period - 1
    w_ptr: *const f64, // pointer to the first of the (period - 1) weights
    w_sum: f64,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let p = period;
    let j0 = first + p + 1;
    if j0 >= n {
        return;
    }

    debug_assert_eq!(p_minus_1, p - 1);
    let m = p - 2;
    let m_len = m + 1;

    let inv_ws = 1.0 / w_sum;
    let p_f = p as f64;
    let p2 = p_f * p_f;
    let c1 = 1.0 - 2.0 * p_f; // (-2p + 1)
    let pm1f = (p - 1) as f64;

    // Initialize A, B, R at j0 (loop-jammed, unrolled x4)
    let mut a_sum = 0.0;
    let mut b_sum = 0.0;
    let mut r_acc = 0.0;

    let base = j0;
    let u4 = m_len & !3;
    let mut i = 0usize;
    while i < u4 {
        let idx0 = base - i;
        let idx1 = base - (i + 1);
        let idx2 = base - (i + 2);
        let idx3 = base - (i + 3);

        let d0 = *data.get_unchecked(idx0);
        let d1 = *data.get_unchecked(idx1);
        let d2 = *data.get_unchecked(idx2);
        let d3 = *data.get_unchecked(idx3);

        a_sum = a_sum + d0 + d1 + d2 + d3;

        b_sum = b_sum
            + (i as f64) * d0
            + ((i + 1) as f64) * d1
            + ((i + 2) as f64) * d2
            + ((i + 3) as f64) * d3;

        r_acc = d0.mul_add(*w_ptr.add(i), r_acc);
        r_acc = d1.mul_add(*w_ptr.add(i + 1), r_acc);
        r_acc = d2.mul_add(*w_ptr.add(i + 2), r_acc);
        r_acc = d3.mul_add(*w_ptr.add(i + 3), r_acc);

        i += 4;
    }
    while i < m_len {
        let idx = base - i;
        let d = *data.get_unchecked(idx);
        a_sum += d;
        b_sum += (i as f64) * d;
        r_acc = d.mul_add(*w_ptr.add(i), r_acc);
        i += 1;
    }

    *out.get_unchecked_mut(j0) = r_acc * inv_ws;

    // O(1) recurrence for the rest of the row, with periodic rebase
    let mut j = j0 + 1;
    let mut iter_since_rebase = 0usize;
    const REBASE_MASK: usize = (1usize << 6) - 1;

    while j < n {
        let x_in = *data.get_unchecked(j);
        let x_out = *data.get_unchecked(j - p + 1);

        r_acc = p2.mul_add(x_in, r_acc);
        r_acc = 2.0_f64.mul_add(b_sum, r_acc);
        r_acc = c1.mul_add(a_sum, r_acc) - x_out;
        *out.get_unchecked_mut(j) = r_acc * inv_ws;

        let a_prev = a_sum;
        a_sum = a_prev + x_in - x_out;
        b_sum = b_sum + a_prev - pm1f * x_out;

        iter_since_rebase = iter_since_rebase.wrapping_add(1);
        if (iter_since_rebase & REBASE_MASK) == 0 {
            let base = j;
            let mut a2 = 0.0;
            let mut b2 = 0.0;
            let mut r2 = 0.0;
            let mut i = 0usize;
            let u4 = m_len & !3;
            while i < u4 {
                let idx0 = base - i;
                let idx1 = base - (i + 1);
                let idx2 = base - (i + 2);
                let idx3 = base - (i + 3);

                let d0 = *data.get_unchecked(idx0);
                let d1 = *data.get_unchecked(idx1);
                let d2 = *data.get_unchecked(idx2);
                let d3 = *data.get_unchecked(idx3);

                a2 = a2 + d0 + d1 + d2 + d3;
                b2 = b2
                    + (i as f64) * d0
                    + ((i + 1) as f64) * d1
                    + ((i + 2) as f64) * d2
                    + ((i + 3) as f64) * d3;
                r2 = d0.mul_add(*w_ptr.add(i), r2);
                r2 = d1.mul_add(*w_ptr.add(i + 1), r2);
                r2 = d2.mul_add(*w_ptr.add(i + 2), r2);
                r2 = d3.mul_add(*w_ptr.add(i + 3), r2);
                i += 4;
            }
            while i < m_len {
                let idx = base - i;
                let d = *data.get_unchecked(idx);
                a2 += d;
                b2 += (i as f64) * d;
                r2 = d.mul_add(*w_ptr.add(i), r2);
                i += 1;
            }
            a_sum = a2;
            b_sum = b2;
            r_acc = r2;
            *out.get_unchecked_mut(j) = r_acc * inv_ws;
        }

        j += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn sqwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    w_sum: f64,
    out: &mut [f64],
) {
    sqwma_row_scalar(data, first, period, stride, w_ptr, w_sum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sqwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    w_sum: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        sqwma_row_avx512_short(data, first, period, stride, w_ptr, w_sum, out);
    } else {
        sqwma_row_avx512_long(data, first, period, stride, w_ptr, w_sum, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn sqwma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    w_sum: f64,
    out: &mut [f64],
) {
    sqwma_row_scalar(data, first, period, _stride, w_ptr, w_sum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn sqwma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    w_sum: f64,
    out: &mut [f64],
) {
    sqwma_row_scalar(data, first, period, _stride, w_ptr, w_sum, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_sqwma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SqwmaParams { period: None };
        let input = SqwmaInput::from_candles(&candles, "close", default_params);
        let output = sqwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_sqwma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five = [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083,
        ];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SqwmaParams::default();
        let input = SqwmaInput::from_candles(&candles, "close", default_params);
        let result = sqwma_with_kernel(&input, kernel)?;
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp_val = expected_last_five[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "[{}] SQWMA mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp_val
            );
        }
        Ok(())
    }

    fn check_sqwma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SqwmaParams { period: Some(0) };
        let input = SqwmaInput::from_slice(&input_data, params);
        let res = sqwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SQWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_sqwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SqwmaParams { period: Some(10) };
        let input = SqwmaInput::from_slice(&data_small, params);
        let res = sqwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SQWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_sqwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SqwmaParams { period: Some(9) };
        let input = SqwmaInput::from_slice(&single_point, params);
        let res = sqwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SQWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_sqwma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SqwmaInput::from_candles(&candles, "close", SqwmaParams { period: Some(14) });
        let res = sqwma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        for (i, &val) in res.values[240..].iter().enumerate() {
            assert!(
                !val.is_nan(),
                "[{}] Found unexpected NaN at out-index {}",
                test_name,
                240 + i
            );
        }
        Ok(())
    }

    fn check_sqwma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let input = SqwmaInput::from_candles(
            &candles,
            "close",
            SqwmaParams {
                period: Some(period),
            },
        );
        let batch_output = sqwma_with_kernel(&input, kernel)?.values;
        let mut stream = SqwmaStream::try_new(SqwmaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(v) => stream_values.push(v),
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
                "[{}] SQWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_sqwma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_sqwma_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase chance of catching bugs
        let test_periods = vec![
            2,   // Minimum valid period
            5,   // Small period
            14,  // Default period
            30,  // Medium period
            50,  // Large period
            100, // Very large period
            200, // Extra large period
        ];

        for &period in &test_periods {
            // Skip if period would be too large for the data
            if period > candles.close.len() {
                continue;
            }

            let input = SqwmaInput::from_candles(
                &candles,
                "close",
                SqwmaParams {
                    period: Some(period),
                },
            );
            let output = sqwma_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_sqwma_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_sqwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Load real market data for realistic testing
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_data = &candles.close;

        // Strategy: test various parameter combinations with real data slices
        // SQWMA uses squared weights, typically with moderate periods
        let strat = (
            2usize..=50,                                  // period (SQWMA typical range)
            0usize..close_data.len().saturating_sub(200), // starting index
            100usize..=200,                               // length of data slice to use
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                // Ensure we have valid slice bounds
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period + 10 {
                    return Ok(()); // Skip invalid combinations
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = SqwmaParams {
                    period: Some(period),
                };
                let input = SqwmaInput::from_slice(data_slice, params.clone());

                // Test the specified kernel
                let result = sqwma_with_kernel(&input, kernel);

                // Also compute with scalar kernel for reference
                let scalar_result = sqwma_with_kernel(&input, Kernel::Scalar);

                // Both should succeed or fail together
                match (result, scalar_result) {
                    (Ok(SqwmaOutput { values: out }), Ok(SqwmaOutput { values: ref_out })) => {
                        // Verify output length
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        // Find first non-NaN value
                        let first = data_slice.iter().position(|x| !x.is_nan()).unwrap_or(0);
                        let expected_warmup = first + period + 1; // SQWMA warmup: first + period + 1

                        // Check NaN pattern during warmup
                        for i in 0..expected_warmup.min(out.len()) {
                            prop_assert!(
                                out[i].is_nan(),
                                "Expected NaN at index {} during warmup, got {}",
                                i,
                                out[i]
                            );
                        }

                        // Test square weight properties
                        // SQWMA weights: (period)^2, (period-1)^2, ..., 2^2
                        let mut weights = Vec::with_capacity(period - 1);
                        for i in 0..(period - 1) {
                            weights.push((period as f64 - i as f64).powi(2));
                        }
                        let weight_sum: f64 = weights.iter().sum();

                        // Verify weights follow quadratic pattern
                        for i in 0..(period - 1) {
                            let expected_weight = (period as f64 - i as f64).powi(2);
                            prop_assert!(
                                (weights[i] - expected_weight).abs() < 1e-10,
                                "Weight {} doesn't match quadratic pattern: {} vs {}",
                                i,
                                weights[i],
                                expected_weight
                            );
                        }

                        // Verify weight sum is positive and reasonable
                        prop_assert!(
                            weight_sum > 0.0,
                            "Weight sum should be positive: {}",
                            weight_sum
                        );

                        // Test specific properties for valid outputs
                        for i in expected_warmup..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            // Both should be valid
                            prop_assert!(!y.is_nan(), "Unexpected NaN at index {}", i);
                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

                            // Kernel consistency check
                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();

                            if !y.is_finite() || !r.is_finite() {
                                prop_assert_eq!(
                                    y_bits,
                                    r_bits,
                                    "NaN/Inf mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                                continue;
                            }

                            // ULP difference check for floating-point precision
                            let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                "Kernel mismatch at {}: {} vs {} (ULP={})",
                                i,
                                y,
                                r,
                                ulp_diff
                            );

                            // Output bounds check - SQWMA output should be within window bounds
                            // SQWMA uses values from i down to i-(period-2), which is period-1 values
                            if i >= period - 1 {
                                let window_start = i - (period - 2); // i - period + 2
                                let window_end = i + 1; // inclusive of i
                                let window = &data_slice[window_start..window_end];
                                let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max_val =
                                    window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                                // SQWMA is a weighted average, so it must be within min/max
                                prop_assert!(
                                    y >= min_val - 1e-9 && y <= max_val + 1e-9,
                                    "SQWMA value {} outside window bounds [{}, {}] at index {}",
                                    y,
                                    min_val,
                                    max_val,
                                    i
                                );
                            }
                        }

                        // Test constant data property
                        let const_data = vec![42.0; period + 10];
                        let const_input = SqwmaInput::from_slice(&const_data, params.clone());
                        if let Ok(SqwmaOutput { values: const_out }) =
                            sqwma_with_kernel(&const_input, kernel)
                        {
                            // Find where outputs should start being valid
                            let const_warmup = period + 1; // No NaN in input, so warmup is just period + 1
                            for (i, &val) in const_out.iter().enumerate() {
                                if i >= const_warmup && !val.is_nan() {
                                    prop_assert!(
										(val - 42.0).abs() < 1e-9,
										"SQWMA of constant data should equal the constant at {}: got {}",
										i, val
									);
                                }
                            }
                        }

                        // Test that weights decrease quadratically
                        if period > 2 {
                            for i in 1..(period - 1) {
                                let ratio = weights[i] / weights[i - 1];
                                let expected_ratio = ((period as f64 - i as f64)
                                    / (period as f64 - (i - 1) as f64))
                                    .powi(2);
                                prop_assert!(
                                    (ratio - expected_ratio).abs() < 1e-10,
                                    "Weight ratio doesn't follow quadratic pattern at {}: {} vs {}",
                                    i,
                                    ratio,
                                    expected_ratio
                                );
                            }
                        }
                    }
                    (Err(e1), Err(e2)) => {
                        // Both kernels should fail with similar errors
                        prop_assert_eq!(
                            std::mem::discriminant(&e1),
                            std::mem::discriminant(&e2),
                            "Different error types: {:?} vs {:?}",
                            e1,
                            e2
                        );
                    }
                    _ => {
                        prop_assert!(
                            false,
                            "Kernel consistency failure: one succeeded, one failed"
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_sqwma_tests!(
        check_sqwma_partial_params,
        check_sqwma_accuracy,
        check_sqwma_zero_period,
        check_sqwma_period_exceeds_length,
        check_sqwma_very_small_dataset,
        check_sqwma_nan_handling,
        check_sqwma_streaming,
        check_sqwma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_sqwma_tests!(check_sqwma_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = SqwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = SqwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations to increase detection coverage
        let batch_configs = vec![
            (2, 10, 1),    // Edge case: starting from minimum period
            (5, 20, 5),    // Small range with step 5
            (10, 30, 10),  // Medium range with larger step
            (14, 100, 7),  // Default start with step 7
            (50, 200, 50), // Very large periods
            (2, 5, 1),     // Very small range to test edge cases
        ];

        for (start, end, step) in batch_configs {
            // Skip configurations that would exceed data length
            if start > c.close.len() {
                continue;
            }

            let output = SqwmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let period = output.combos[row].period.unwrap_or(0);

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
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

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "sqwma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Square Weighted Moving Average (SQWMA) of the input data.
///
/// SQWMA applies squared weights to recent data points, with the most recent
/// value receiving (period)^2 weight, and each preceding value's weight
/// decreasing quadratically.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Window size (number of data points, must be >= 2).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SQWMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period < 2, period > data length, etc).
pub fn sqwma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use crate::utilities::kernel_validation::validate_kernel;
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?; // Validate before allow_threads

    let params = SqwmaParams {
        period: Some(period),
    };
    let sqwma_in = SqwmaInput::from_slice(slice_in, params);

    // GOOD: Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| sqwma_with_kernel(&sqwma_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // GOOD: Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SqwmaStream")]
pub struct SqwmaStreamPy {
    stream: SqwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SqwmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SqwmaParams {
            period: Some(period),
        };
        let stream =
            SqwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SqwmaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated SQWMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "sqwma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SQWMA for multiple parameter combinations in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' arrays.
pub fn sqwma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use crate::utilities::kernel_validation::validate_kernel;
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let sweep = SqwmaBatchRange {
        period: period_range,
    };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    if rows == 0 {
        return Err(PyValueError::new_err("empty period grid"));
    }

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        // Warm prefixes in the Python buffer
        let first = slice_in
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(SqwmaError::AllValuesNaN)?;
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if slice_in.len() - first < max_p {
            return Err(SqwmaError::NotEnoughValidData {
                needed: max_p,
                valid: slice_in.len() - first,
            });
        }
        // Init NaN prefixes
        let warm: Vec<usize> = combos
            .iter()
            .map(|c| (first + c.period.unwrap() + 1).min(cols))
            .collect();
        let mu = unsafe {
            std::slice::from_raw_parts_mut(
                slice_out.as_mut_ptr() as *mut MaybeUninit<f64>,
                slice_out.len(),
            )
        };
        init_matrix_prefixes(mu, cols, &warm);

        sqwma_batch_inner_into(slice_in, &combos, kern, first, max_p, true, slice_out)
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
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "sqwma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn sqwma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = SqwmaBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSqwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.sqwma_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "sqwma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn sqwma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let flat = data_tm_f32.as_slice()?;

    let inner = py.allow_threads(|| {
        let cuda = CudaSqwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.sqwma_many_series_one_param_time_major_dev(flat, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sqwma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SqwmaParams {
        period: Some(period),
    };
    let input = SqwmaInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    sqwma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sqwma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SqwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    sqwma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sqwma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SqwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let periods: Vec<f64> = combos.iter().map(|c| c.period.unwrap() as f64).collect();

    Ok(periods)
}

// ================== Optimized WASM Helper Functions ==================

/// Write directly to output slice - no allocations
pub fn sqwma_into_slice(
    dst: &mut [f64],
    input: &SqwmaInput,
    kernel: Kernel,
) -> Result<(), SqwmaError> {
    let data: &[f64] = match &input.data {
        SqwmaData::Candles { candles, source } => source_type(candles, source),
        SqwmaData::Slice(sl) => sl,
    };
    if data.is_empty() {
        return Err(SqwmaError::EmptyInputData);
    }
    let period = input.params.period.unwrap_or(14);

    if dst.len() != data.len() {
        return Err(SqwmaError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    if period < 2 || period > data.len() {
        return Err(SqwmaError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first = data
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(SqwmaError::AllValuesNaN)?;

    if data.len() - first < period {
        return Err(SqwmaError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    // Build weights
    let mut weights = Vec::with_capacity(period - 1);
    for i in 0..(period - 1) {
        weights.push((period as f64 - i as f64).powi(2));
    }
    let weight_sum: f64 = weights.iter().sum();

    // Fill warmup with NaN
    let warmup = first + period + 1;
    let warmup_end = warmup.min(dst.len());
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    // Compute SQWMA
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                sqwma_scalar(data, &weights, period, first, weight_sum, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                sqwma_avx2(data, &weights, period, first, weight_sum, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                sqwma_avx512(data, &weights, period, first, weight_sum, dst)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sqwma_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sqwma_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sqwma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period < 2 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Calculate SQWMA
        let params = SqwmaParams {
            period: Some(period),
        };
        let input = SqwmaInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            sqwma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            sqwma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SqwmaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SqwmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SqwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = sqwma_batch)]
pub fn sqwma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: SqwmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = SqwmaBatchRange {
        period: config.period_range,
    };

    let output = sqwma_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = SqwmaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sqwma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to sqwma_batch_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = SqwmaBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        if rows == 0 {
            return Err(JsValue::from_str("empty period grid"));
        }
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| JsValue::from_str("All values are NaN"))?;
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if data.len() - first < max_p {
            return Err(JsValue::from_str("Not enough valid data"));
        }

        // Init NaN prefixes in-place
        let warm: Vec<usize> = combos
            .iter()
            .map(|c| (first + c.period.unwrap() + 1).min(cols))
            .collect();
        let mu =
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len());
        init_matrix_prefixes(mu, cols, &warm);

        // Compute into out
        sqwma_batch_inner_into(data, &combos, Kernel::Auto, first, max_p, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
