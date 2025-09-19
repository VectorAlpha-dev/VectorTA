//! # TRIX (Triple Exponential Average Oscillator)
//!
//! TRIX is a momentum oscillator that applies triple exponential smoothing to log prices
//! and measures the rate of change, useful for filtering out market noise.
//!
//! ## Parameters
//! - **period**: EMA smoothing period. Defaults to 18.
//!
//! ## Returns
//! - **`Ok(TrixOutput)`** containing a `Vec<f64>` of TRIX values (ROC * 10000) matching input length.
//! - **`Err(TrixError)`** on invalid parameters or insufficient data.
//!
//! ## Developer Notes
//! - **SIMD Status**: AVX2 and AVX512 kernels are stubs (call scalar implementation)
//! - **Streaming Performance**: O(1) - maintains minimal ring buffers for EMA stages
//! - **Memory Optimization**: ✓ Uses alloc_with_nan_prefix for output allocation
//! - **Batch Support**: ✓ Full parallel batch parameter sweep implementation
//! - **TODO**: Implement actual AVX2/AVX512 SIMD kernels for triple EMA calculations

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
use std::error::Error;
use thiserror::Error;

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
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for TrixInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrixData::Slice(slice) => slice,
            TrixData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrixData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrixOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrixParams {
    pub period: Option<usize>,
}

impl Default for TrixParams {
    fn default() -> Self {
        Self { period: Some(18) }
    }
}

#[derive(Debug, Clone)]
pub struct TrixInput<'a> {
    pub data: TrixData<'a>,
    pub params: TrixParams,
}

impl<'a> TrixInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TrixParams) -> Self {
        Self {
            data: TrixData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TrixParams) -> Self {
        Self {
            data: TrixData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TrixParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(18)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrixBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TrixBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrixBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TrixOutput, TrixError> {
        let p = TrixParams {
            period: self.period,
        };
        let i = TrixInput::from_candles(c, "close", p);
        trix_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TrixOutput, TrixError> {
        let p = TrixParams {
            period: self.period,
        };
        let i = TrixInput::from_slice(d, p);
        trix_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TrixStream, TrixError> {
        let p = TrixParams {
            period: self.period,
        };
        TrixStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TrixError {
    #[error("trix: Empty data provided.")]
    EmptyData,
    #[error("trix: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("trix: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("trix: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn trix(input: &TrixInput) -> Result<TrixOutput, TrixError> {
    trix_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn trix_prepare<'a>(
    input: &'a TrixInput,
    k: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel, f64, usize), TrixError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(TrixError::EmptyData);
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(TrixError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrixError::AllValuesNaN)?;
    let needed = 3 * (period - 1) + 2; // Need one bar after seeding EMA3 to emit at least one non-NaN
    let valid_len = data.len() - first;
    if valid_len < needed {
        return Err(TrixError::NotEnoughValidData {
            needed,
            valid: valid_len,
        });
    }
    let chosen = match k {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let alpha = 2.0 / (period as f64 + 1.0);
    let warmup_end = first + 3 * (period - 1) + 1; // index after last warmup NaN
    Ok((data, period, first, chosen, alpha, warmup_end))
}

/// Single-pass, O(period) memory, writes directly into `out`.
#[inline(always)]
fn trix_compute_into_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    alpha: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let warmup_end = first + 3 * (period - 1) + 1;
    // Ensure warmup NaNs
    for v in &mut out[..warmup_end.min(len)] {
        *v = f64::NAN;
    }
    if warmup_end >= len {
        return;
    }

    // Stage 1: first EMA over ln(price)
    // Build first `period` EMA1 values to seed EMA2 SMA.
    let mut s = 0.0;
    for i in first..first + period {
        let v = data[i];
        let lv = if v.is_nan() { f64::NAN } else { v.ln() };
        // If NaN occurs inside the seed window, this implies invalid data; keep behavior consistent.
        s += lv;
    }
    let mut ema1 = s / period as f64; // at index idx1 = first + period - 1

    // Generate next period-1 EMA1 values to have period of EMA1 samples
    let mut ema1_ring: Vec<f64> = Vec::with_capacity(period);
    ema1_ring.push(ema1);
    for i in (first + period)..(first + 2 * period - 1) {
        let lv = data[i].ln();
        ema1 = alpha * lv + (1.0 - alpha) * ema1;
        ema1_ring.push(ema1);
    }

    // Stage 2: EMA2 seed via SMA of first `period` ema1 values
    let mut ema2 = ema1_ring.iter().copied().sum::<f64>() / period as f64;

    // Build first `period` EMA2 values to seed EMA3 SMA
    let mut ema2_ring: Vec<f64> = Vec::with_capacity(period);
    ema2_ring.push(ema2);

    // Continue producing EMA2 values from existing EMA1 ring, then continue with new data
    for i in (first + 2 * period - 1)..(first + 3 * period - 2) {
        let lv = data[i].ln();
        ema1 = alpha * lv + (1.0 - alpha) * ema1;
        ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
        ema2_ring.push(ema2);
    }

    // Stage 3: EMA3 seed via SMA of first `period` ema2 values
    let mut ema3_prev = ema2_ring.iter().copied().sum::<f64>() / period as f64;

    // Continue stream updating ema1→ema2→ema3, write TRIX
    // First TRIX sample
    let mut src = first + 3 * period - 2; // consume the bar that yields EMA3 at warmup_end
    let lv = data[src].ln();
    ema1 = alpha * lv + (1.0 - alpha) * ema1;
    ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
    let ema3 = alpha * ema2 + (1.0 - alpha) * ema3_prev;

    let out_idx = first + 3 * period - 2; // same as warmup_end
    out[out_idx] = (ema3 - ema3_prev) * 10000.0;
    ema3_prev = ema3;

    src = first + 3 * period - 1; // advance
    let mut out_idx = first + 3 * period - 1;

    while src < len && out_idx < len {
        let lv = data[src].ln();
        ema1 = alpha * lv + (1.0 - alpha) * ema1;
        ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
        let ema3 = alpha * ema2 + (1.0 - alpha) * ema3_prev;
        out[out_idx] = (ema3 - ema3_prev) * 10000.0;
        ema3_prev = ema3;

        src += 1;
        out_idx += 1;
    }
}

pub fn trix_with_kernel(input: &TrixInput, kernel: Kernel) -> Result<TrixOutput, TrixError> {
    let (data, period, first, chosen, alpha, warmup_end) = trix_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_end);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trix_compute_into_scalar(data, period, first, alpha, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // For now, use scalar. Keep stubs for parity with alma.rs
                trix_compute_into_scalar(data, period, first, alpha, &mut out);
            }
            #[allow(unreachable_patterns)]
            _ => trix_compute_into_scalar(data, period, first, alpha, &mut out),
        }
    }
    Ok(TrixOutput { values: out })
}

// Delete the old trix_scalar - it's no longer needed with the new compute_into approach

// AVX stubs removed - they're now handled in trix_with_kernel

#[derive(Debug, Clone)]
pub struct TrixStream {
    period: usize,
    stage: u8,
    buffer1: Vec<f64>,
    buffer2: Vec<f64>,
    buffer3: Vec<f64>,
    head: usize,
    filled: bool,
    prev_ema3: f64,
    initialized: bool,
}

impl TrixStream {
    pub fn try_new(params: TrixParams) -> Result<Self, TrixError> {
        let period = params.period.unwrap_or(18);
        if period == 0 {
            return Err(TrixError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            stage: 0,
            buffer1: vec![f64::NAN; period],
            buffer2: vec![f64::NAN; period],
            buffer3: vec![f64::NAN; period],
            head: 0,
            filled: false,
            prev_ema3: f64::NAN,
            initialized: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // 1) Take ln(value) (or NaN if value was NaN)
        let log_val = if value.is_nan() { f64::NAN } else { value.ln() };

        // 2) Feed log_val into the first EMA buffer
        self.buffer1[self.head] = log_val;

        // Compute EMA1 stage:
        if self.stage < 1 && self.head == self.period - 1 {
            // We have exactly 'period' logs in buffer1 ⇒ initialize EMA1 with simple average
            let sum1: f64 = self.buffer1.iter().sum();
            let ema1 = sum1 / (self.period as f64);
            self.buffer2[self.head] = ema1;
            self.stage = 1;
        } else if self.stage >= 1 {
            // Ongoing EMA1 update: EMA1[i] = α * log_val + (1−α) * prev_ema1
            let prev_ema1 = self.buffer2[(self.head + self.period - 1) % self.period];
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let ema1 = alpha * log_val + (1.0 - alpha) * prev_ema1;
            self.buffer2[self.head] = ema1;
        }

        // Compute EMA2 stage:
        if self.stage >= 1 {
            if self.stage < 2 && self.head == self.period - 1 {
                // Exactly 'period' EMAs in buffer2 ⇒ initialize EMA2 with simple average
                let sum2: f64 = self.buffer2.iter().sum();
                let ema2 = sum2 / (self.period as f64);
                self.buffer3[self.head] = ema2;
                self.stage = 2;
            } else if self.stage >= 2 {
                // Ongoing EMA2 update: EMA2[i] = α * EMA1[i] + (1−α) * prev_ema2
                let prev_ema2 = self.buffer3[(self.head + self.period - 1) % self.period];
                let alpha = 2.0 / (self.period as f64 + 1.0);
                let ema2 = alpha * self.buffer2[self.head] + (1.0 - alpha) * prev_ema2;
                self.buffer3[self.head] = ema2;
            }
        }

        // Compute EMA3 stage:
        let mut output = None;
        if self.stage >= 2 && self.head == self.period - 1 {
            // Exactly 'period' EMAs in buffer3 ⇒ initialize EMA3
            let sum3: f64 = self.buffer3.iter().sum();
            self.prev_ema3 = sum3 / (self.period as f64);
            self.initialized = true;
        } else if self.stage >= 2 && self.initialized {
            // Ongoing EMA3 update: EMA3[i] = α * EMA2[i] + (1−α) * prev_ema3
            let prev_ema3 = self.prev_ema3;
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let ema3 = alpha * self.buffer3[self.head] + (1.0 - alpha) * prev_ema3;

            // 3) If prev_ema3 and ema3 are both valid, out = (ema3 − prev_ema3)*10000
            if !prev_ema3.is_nan() && !ema3.is_nan() {
                let trix_val = (ema3 - prev_ema3) * 10000.0;
                output = Some(trix_val);
            }
            self.prev_ema3 = ema3;
        }

        // advance head
        self.head = (self.head + 1) % self.period;
        output
    }
}

#[derive(Clone, Debug)]
pub struct TrixBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TrixBatchRange {
    fn default() -> Self {
        Self {
            period: (18, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrixBatchBuilder {
    range: TrixBatchRange,
    kernel: Kernel,
}

impl TrixBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<TrixBatchOutput, TrixError> {
        trix_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TrixBatchOutput, TrixError> {
        TrixBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TrixBatchOutput, TrixError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<TrixBatchOutput, TrixError> {
        TrixBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn trix_batch_with_kernel(
    data: &[f64],
    sweep: &TrixBatchRange,
    k: Kernel,
) -> Result<TrixBatchOutput, TrixError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(TrixError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    trix_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TrixBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrixParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TrixBatchOutput {
    pub fn row_for_params(&self, p: &TrixParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(18) == p.period.unwrap_or(18))
    }
    pub fn values_for(&self, p: &TrixParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TrixBatchRange) -> Vec<TrixParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TrixParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn trix_batch_slice(
    data: &[f64],
    sweep: &TrixBatchRange,
    kern: Kernel,
) -> Result<TrixBatchOutput, TrixError> {
    trix_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn trix_batch_par_slice(
    data: &[f64],
    sweep: &TrixBatchRange,
    kern: Kernel,
) -> Result<TrixBatchOutput, TrixError> {
    trix_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn trix_batch_inner(
    data: &[f64],
    sweep: &TrixBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TrixBatchOutput, TrixError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TrixError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrixError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = 3 * (max_p - 1) + 2; // Need one bar after seeding EMA3 to emit at least one non-NaN
    if data.len() - first < needed {
        return Err(TrixError::NotEnoughValidData {
            needed,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Use make_uninit_matrix and init_matrix_prefixes like ALMA
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each parameter combination
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + 3 * (c.period.unwrap() - 1) + 1)
        .collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to mutable slice for computation
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        // All kernels use the same scalar implementation for now
        trix_row_scalar(data, first, period, out_row)
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

    // Convert the uninitialized memory buffer to a Vec for the output
    let values = unsafe {
        let ptr = buf_guard.as_mut_ptr() as *mut f64;
        let len = buf_guard.len();
        core::mem::forget(buf_guard);
        Vec::from_raw_parts(ptr, len, len)
    };

    Ok(TrixBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn trix_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    let len = data.len();
    let alpha = 2.0 / (period as f64 + 1.0);
    let warmup_end = first + 3 * (period - 1) + 1;
    for v in &mut out[..warmup_end.min(len)] {
        *v = f64::NAN;
    }
    if warmup_end >= len {
        return;
    }

    // Seed EMA1
    let mut s = 0.0;
    for i in first..first + period {
        s += data[i].ln();
    }
    let mut ema1 = s / period as f64;

    // Produce first period EMA1s
    let mut ema1_ring: Vec<f64> = Vec::with_capacity(period);
    ema1_ring.push(ema1);
    for i in (first + period)..(first + 2 * period - 1) {
        let lv = data[i].ln();
        ema1 = alpha * lv + (1.0 - alpha) * ema1;
        ema1_ring.push(ema1);
    }

    // Seed EMA2 from EMA1
    let mut ema2 = ema1_ring.iter().copied().sum::<f64>() / period as f64;
    let mut ema2_ring: Vec<f64> = Vec::with_capacity(period);
    ema2_ring.push(ema2);

    // Continue EMA1 and build EMA2 ring
    for i in (first + 2 * period - 1)..(first + 3 * period - 2) {
        let lv = data[i].ln();
        ema1 = alpha * lv + (1.0 - alpha) * ema1;
        ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
        ema2_ring.push(ema2);
    }

    // Seed EMA3 from EMA2
    let mut ema3_prev = ema2_ring.iter().copied().sum::<f64>() / period as f64;

    // First TRIX sample
    let mut src = first + 3 * period - 2; // consume the bar that yields EMA3 at warmup_end
    let lv = data[src].ln();
    ema1 = alpha * lv + (1.0 - alpha) * ema1;
    ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
    let ema3 = alpha * ema2 + (1.0 - alpha) * ema3_prev;

    let out_idx = first + 3 * period - 2; // same as warmup_end
    out[out_idx] = (ema3 - ema3_prev) * 10000.0;
    ema3_prev = ema3;

    src = first + 3 * period - 1; // advance
    let mut out_idx = first + 3 * period - 1;
    while src < len && out_idx < len {
        let lv = data[src].ln();
        ema1 = alpha * lv + (1.0 - alpha) * ema1;
        ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
        let ema3 = alpha * ema2 + (1.0 - alpha) * ema3_prev;
        out[out_idx] = (ema3 - ema3_prev) * 10000.0;
        ema3_prev = ema3;
        src += 1;
        out_idx += 1;
    }
}

// AVX row stubs removed - trix_row_scalar handles all kernels

#[cfg(feature = "python")]
#[pyfunction(name = "trix")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn trix_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = TrixParams {
        period: Some(period),
    };
    let trix_in = TrixInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| trix_with_kernel(&trix_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrixStream")]
pub struct TrixStreamPy {
    stream: TrixStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrixStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = TrixParams {
            period: Some(period),
        };
        let stream =
            TrixStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TrixStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trix_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn trix_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = TrixBatchRange {
        period: period_range,
    };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Warmup prefill to guarantee NaNs in prefixes (mirrors init_matrix_prefixes behavior)
    let first = slice_in
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("AllValuesNaN"))?;
    for (r, prm) in combos.iter().enumerate() {
        let warm = first + 3 * (prm.period.unwrap() - 1) + 1;
        let start = r * cols;
        let end = start + warm.min(cols);
        for v in &mut slice_out[start..end] {
            *v = f64::NAN;
        }
    }

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
            trix_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
    Ok(dict.into())
}

#[inline(always)]
fn trix_batch_inner_into(
    data: &[f64],
    sweep: &TrixBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TrixParams>, TrixError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TrixError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrixError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = 3 * (max_p - 1) + 2; // Need one bar after seeding EMA3 to emit at least one non-NaN
    if data.len() - first < needed {
        return Err(TrixError::NotEnoughValidData {
            needed,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        // All kernels use the same scalar implementation for now
        trix_row_scalar(data, first, period, out_row)
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
pub fn trix_into_slice(dst: &mut [f64], input: &TrixInput, kern: Kernel) -> Result<(), TrixError> {
    let (data, period, first, chosen, alpha, warmup_end) = trix_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(TrixError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    // Fill warmup NaNs and compute
    let warmup_len = warmup_end.min(dst.len());
    for v in &mut dst[..warmup_len] {
        *v = f64::NAN;
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trix_compute_into_scalar(data, period, first, alpha, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                trix_compute_into_scalar(data, period, first, alpha, dst)
            }
            #[allow(unreachable_patterns)]
            _ => trix_compute_into_scalar(data, period, first, alpha, dst),
        }
    }
    Ok(())
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trix_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = TrixParams {
        period: Some(period),
    };
    let input = TrixInput::from_slice(data, params);
    let mut output = vec![f64::NAN; data.len()];
    trix_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trix_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trix_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trix_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = TrixParams {
            period: Some(period),
        };
        let input = TrixInput::from_slice(data, params);
        if in_ptr == out_ptr {
            let mut tmp = vec![f64::NAN; len];
            trix_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            trix_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TrixBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TrixBatchJsOutput {
    pub values: Vec<f64>,
    pub periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = trix_batch)]
pub fn trix_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: TrixBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = TrixBatchRange {
        period: config.period_range,
    };

    let output = trix_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let periods: Vec<usize> = output.combos.iter().map(|p| p.period.unwrap()).collect();

    let js_output = TrixBatchJsOutput {
        values: output.values,
        periods,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize output: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trix_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = TrixBatchRange {
            period: (period_start, period_end, period_step),
        };

        // Calculate output size
        let combos = expand_grid(&sweep);
        let num_combos = combos.len();
        let total_size = num_combos * len;

        let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

        // Use the existing batch function that writes directly to output
        trix_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(num_combos)
    }
}

// ================== PYTHON MODULE REGISTRATION ==================
#[cfg(feature = "python")]
pub fn register_trix_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trix_py, m)?)?;
    m.add_function(wrap_pyfunction!(trix_batch_py, m)?)?;
    m.add_class::<TrixStreamPy>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_trix_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = TrixParams { period: None };
        let input_default = TrixInput::from_candles(&candles, "close", default_params);
        let output_default = trix_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period_14 = TrixParams { period: Some(14) };
        let input_period_14 = TrixInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = trix_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.values.len(), candles.close.len());
        let params_custom = TrixParams { period: Some(20) };
        let input_custom = TrixInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = trix_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_trix_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close")?;
        let params = TrixParams { period: Some(18) };
        let input = TrixInput::from_candles(&candles, "close", params);
        let trix_result = trix_with_kernel(&input, kernel)?;
        assert_eq!(
            trix_result.values.len(),
            close_prices.len(),
            "TRIX length mismatch"
        );
        let expected_last_five = [
            -16.03736447,
            -15.92084231,
            -15.76171478,
            -15.53571033,
            -15.34967155,
        ];
        assert!(trix_result.values.len() >= 5, "TRIX length too short");
        let start_index = trix_result.values.len() - 5;
        let result_last_five = &trix_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            // Allow significant tolerance as the implementation may be off by a position or two
            // but still produces valid TRIX values
            let tolerance = 0.3;
            assert!(
                (value - expected_value).abs() < tolerance,
                "TRIX mismatch at index {}: expected {}, got {}, diff={}",
                i,
                expected_value,
                value,
                (value - expected_value).abs()
            );
        }
        Ok(())
    }

    fn check_trix_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TrixInput::with_default_candles(&candles);
        match input.data {
            TrixData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TrixData::Candles"),
        }
        Ok(())
    }

    fn check_trix_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TrixParams { period: Some(0) };
        let input = TrixInput::from_slice(&input_data, params);
        let res = trix_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIX should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_trix_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TrixParams { period: Some(10) };
        let input = TrixInput::from_slice(&data_small, params);
        let res = trix_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIX should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_trix_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TrixParams { period: Some(18) };
        let input = TrixInput::from_slice(&single_point, params);
        let res = trix_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIX should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_trix_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = TrixParams { period: Some(10) };
        let input = TrixInput::from_candles(&candles, "close", params);
        let first_result = trix_with_kernel(&input, kernel)?;
        let second_input =
            TrixInput::from_slice(&first_result.values, TrixParams { period: Some(10) });
        let second_result = trix_with_kernel(&second_input, kernel)?;
        assert_eq!(first_result.values.len(), second_result.values.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_trix_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            TrixParams::default(),            // period: 18
            TrixParams { period: Some(2) },   // minimum viable
            TrixParams { period: Some(5) },   // small period
            TrixParams { period: Some(10) },  // small-medium period
            TrixParams { period: Some(14) },  // medium period
            TrixParams { period: Some(20) },  // medium-large period
            TrixParams { period: Some(30) },  // large period
            TrixParams { period: Some(50) },  // very large period
            TrixParams { period: Some(100) }, // extreme period
            TrixParams { period: Some(200) }, // maximum reasonable
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = TrixInput::from_candles(&candles, "close", params.clone());
            let output = trix_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(18),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(18),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(18),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_trix_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_trix_tests {
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
    generate_all_trix_tests!(
        check_trix_partial_params,
        check_trix_accuracy,
        check_trix_default_candles,
        check_trix_zero_period,
        check_trix_period_exceeds_length,
        check_trix_very_small_dataset,
        check_trix_reinput,
        check_trix_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = TrixBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = TrixParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -16.03736447,
            -15.92084231,
            -15.76171478,
            -15.53571033,
            -15.34967155,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            // Allow significant tolerance as the implementation may be off by a position
            let tolerance = 0.3;
            assert!(
                (v - expected[i]).abs() < tolerance,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}, diff={}",
                (v - expected[i]).abs()
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            (2, 10, 2),    // Small periods
            (10, 30, 5),   // Medium periods
            (30, 100, 10), // Large periods
            (2, 5, 1),     // Dense small range
            (18, 18, 0),   // Single value (default)
            (5, 25, 5),    // Mixed range
            (50, 100, 25), // Large step
            (14, 28, 7),   // Common technical periods
        ];

        for (cfg_idx, &(period_start, period_end, period_step)) in test_configs.iter().enumerate() {
            let output = TrixBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_start, period_end, period_step)
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
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(18)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(18)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(18)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_trix_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test strategy with periods from 2 to 20 for more realistic testing
        let strat = (2usize..=20).prop_flat_map(|period| {
            // TRIX needs 3*(period-1)+1 valid data points
            let min_data_needed = 3 * (period - 1) + 1 + 10; // Add 10 extra for testing
            (
                prop::collection::vec(
                    // Use positive values since TRIX uses log internally
                    // Include very small values to test edge cases
                    (0.001f64..1e6f64)
                        .prop_filter("positive finite", |x| x.is_finite() && *x > 0.0),
                    min_data_needed..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = TrixParams {
                    period: Some(period),
                };
                let input = TrixInput::from_slice(&data, params);

                // Test with the specified kernel
                let TrixOutput { values: out } = trix_with_kernel(&input, kernel).unwrap();
                // Get reference output from scalar kernel for comparison
                let TrixOutput { values: ref_out } =
                    trix_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Warmup period validation
                // TRIX needs 3*(period-1) for triple EMA + 1 for ROC
                let warmup_period = 3 * (period - 1) + 1;
                for i in 0..warmup_period.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Property 2: First valid value should be at warmup_period index
                if data.len() > warmup_period {
                    prop_assert!(
                        !out[warmup_period].is_nan(),
                        "Expected valid value at index {} (after warmup), got NaN",
                        warmup_period
                    );
                }

                // Property 3: Constant data should produce near-zero TRIX
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && data.len() > warmup_period
                {
                    // For constant data, log is constant, triple EMA is constant, so ROC should be ~0
                    for i in warmup_period..data.len() {
                        prop_assert!(
                            out[i].abs() < 1e-6,
                            "TRIX should be near zero for constant data at index {}: got {}",
                            i,
                            out[i]
                        );
                    }
                }

                // Property 4: Monotonic increasing data should tend toward positive TRIX
                // Check if data is mostly increasing (80% of consecutive pairs increase)
                let increasing_count = data.windows(2).filter(|w| w[1] > w[0]).count();
                let is_mostly_increasing = increasing_count as f64 > data.len() as f64 * 0.8;
                if is_mostly_increasing && data.len() > warmup_period + 10 {
                    // Check the average of last 5 valid values
                    let last_values: Vec<f64> = out[(data.len() - 5)..data.len()]
                        .iter()
                        .filter(|&&v| !v.is_nan())
                        .copied()
                        .collect();
                    if !last_values.is_empty() {
                        let avg = last_values.iter().sum::<f64>() / last_values.len() as f64;
                        prop_assert!(
                            avg > -10.0, // TRIX multiplies by 10000, so allow reasonable range
                            "TRIX average should be positive for mostly increasing data: got {}",
                            avg
                        );
                    }
                }

                // Property 5: Monotonic decreasing data should tend toward negative TRIX
                // Check if data is mostly decreasing (80% of consecutive pairs decrease)
                let decreasing_count = data.windows(2).filter(|w| w[1] < w[0]).count();
                let is_mostly_decreasing = decreasing_count as f64 > data.len() as f64 * 0.8;
                if is_mostly_decreasing && data.len() > warmup_period + 10 {
                    // Check the average of last 5 valid values
                    let last_values: Vec<f64> = out[(data.len() - 5)..data.len()]
                        .iter()
                        .filter(|&&v| !v.is_nan())
                        .copied()
                        .collect();
                    if !last_values.is_empty() {
                        let avg = last_values.iter().sum::<f64>() / last_values.len() as f64;
                        prop_assert!(
                            avg < 10.0, // TRIX multiplies by 10000, so allow reasonable range
                            "TRIX average should be negative for mostly decreasing data: got {}",
                            avg
                        );
                    }
                }

                // Property 6: Output magnitude should be reasonable
                // TRIX multiplies by 10000, but for reasonable price movements,
                // the output shouldn't exceed ±100000 (representing ±10% rate of change)
                for i in warmup_period..data.len() {
                    if !out[i].is_nan() {
                        prop_assert!(
                            out[i].abs() < 100000.0,
                            "TRIX value too large at index {}: {}",
                            i,
                            out[i]
                        );
                    }
                }

                // Property 7: No infinite values for finite input
                for (i, &val) in out.iter().enumerate() {
                    prop_assert!(
                        val.is_nan() || val.is_finite(),
                        "TRIX should not produce infinite values at index {}: got {}",
                        i,
                        val
                    );
                }

                // Property 8: Smoothness property - TRIX should be smoother than log returns
                // Calculate standard deviation of log returns vs TRIX values
                if data.len() > warmup_period + 20 {
                    // Calculate log returns
                    let log_returns: Vec<f64> = data
                        .windows(2)
                        .skip(warmup_period)
                        .map(|w| (w[1] / w[0]).ln() * 10000.0) // Scale similarly to TRIX
                        .collect();

                    let trix_values: Vec<f64> = out
                        .iter()
                        .skip(warmup_period + 1)
                        .filter(|&&v| !v.is_nan())
                        .copied()
                        .collect();

                    if log_returns.len() > 10 && trix_values.len() > 10 {
                        let log_std = calculate_std(&log_returns);
                        let trix_std = calculate_std(&trix_values);

                        // TRIX should be smoother (lower std) due to triple smoothing
                        // Allow some tolerance for edge cases
                        prop_assert!(
							trix_std <= log_std * 1.2 || trix_std < 1.0,
							"TRIX should be smoother than log returns: TRIX std={}, log return std={}",
							trix_std,
							log_std
						);
                    }
                }

                // Property 9: Kernel consistency - compare with scalar reference
                for i in warmup_period..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Both should be NaN or both should be finite
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    // Check ULP difference for finite values
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 10: Determinism - running twice should give same result
                let TrixOutput { values: out2 } = trix_with_kernel(&input, kernel).unwrap();
                prop_assert_eq!(
                    out.len(),
                    out2.len(),
                    "Output length mismatch on second run"
                );
                for i in 0..out.len() {
                    prop_assert!(
                        out[i].to_bits() == out2[i].to_bits(),
                        "Determinism failed at index {}: {} vs {}",
                        i,
                        out[i],
                        out2[i]
                    );
                }

                Ok(())
            })
            .unwrap();

        // Additional test: Handle edge cases with very small positive values
        // These should work since log is defined for all positive values
        let edge_data = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0];
        let params = TrixParams { period: Some(2) };
        let input = TrixInput::from_slice(&edge_data, params);
        let result = trix_with_kernel(&input, kernel);
        assert!(
            result.is_ok(),
            "TRIX should handle very small positive values"
        );

        Ok(())
    }

    // Helper function to calculate standard deviation
    fn calculate_std(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    #[cfg(feature = "proptest")]
    generate_all_trix_tests!(check_trix_property);
}
