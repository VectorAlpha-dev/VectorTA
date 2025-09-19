//! # Zero Lag Exponential Moving Average (ZLEMA)
//!
//! ZLEMA is a moving average designed to reduce lag by de-lagging the input before EMA calculation.
//! Supports kernel (SIMD) selection and batch/grid computation with streaming support.
//!
//! ## Parameters
//! - **period**: Lookback window (>= 1, defaults to 14).
//!
//! ## Returns
//! - **`Ok(ZlemaOutput)`** on success, containing a `Vec<f64>`.
//! - **`Err(ZlemaError)`** otherwise.
//!
//! ## Developer Status
//! - **AVX2 kernel**: STUB - Falls back to scalar implementation
//! - **AVX512 kernel**: STUB - Falls back to scalar implementation
//! - **Streaming update**: O(1) - Efficient EMA approach with lag compensation
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) ✓
//! - **Optimization needed**: Implement SIMD kernels for vectorized EMA computation
//! - **Note**: De-lagging uses lookback offset calculation

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDictMethods;
#[cfg(feature = "python")]
use pyo3::{pyclass, pyfunction, pymethods, Bound, PyResult, Python};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::JsValue;

impl<'a> AsRef<[f64]> for ZlemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ZlemaData::Slice(slice) => slice,
            ZlemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ZlemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ZlemaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ZlemaParams {
    pub period: Option<usize>,
}

impl Default for ZlemaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ZlemaInput<'a> {
    pub data: ZlemaData<'a>,
    pub params: ZlemaParams,
}

impl<'a> ZlemaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ZlemaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ZlemaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ZlemaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ZlemaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ZlemaOutput, ZlemaError> {
        let p = ZlemaParams {
            period: self.period,
        };
        let i = ZlemaInput::from_candles(c, "close", p);
        zlema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ZlemaOutput, ZlemaError> {
        let p = ZlemaParams {
            period: self.period,
        };
        let i = ZlemaInput::from_slice(d, p);
        zlema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ZlemaStream, ZlemaError> {
        let p = ZlemaParams {
            period: self.period,
        };
        ZlemaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ZlemaError {
    #[error("zlema: Input data slice is empty.")]
    EmptyInputData,
    #[error("zlema: All values are NaN.")]
    AllValuesNaN,
    #[error("zlema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("zlema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline(always)]
fn zlema_validate<'a>(
    input: &'a ZlemaInput,
) -> Result<(&'a [f64], usize, usize, usize), ZlemaError> {
    let data: &'a [f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(ZlemaError::EmptyInputData);
    }
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ZlemaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZlemaError::AllValuesNaN)?;
    if len - first < period {
        return Err(ZlemaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let warm = first + period - 1;
    Ok((data, first, period, warm))
}

#[inline]
pub fn zlema(input: &ZlemaInput) -> Result<ZlemaOutput, ZlemaError> {
    zlema_with_kernel(input, Kernel::Auto)
}

pub fn zlema_with_kernel(input: &ZlemaInput, kernel: Kernel) -> Result<ZlemaOutput, ZlemaError> {
    let (data, first, period, warm) = zlema_validate(input)?;
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => zlema_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => zlema_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => zlema_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(ZlemaOutput { values: out })
}

#[inline]
pub fn zlema_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);

    let warm = first + period - 1;

    // Initialize EMA with first valid value
    let mut last_ema = data[first];

    // Process all values starting from first
    for i in first..len {
        if i > first {
            // For de-lagging, we need to ensure we're not accessing NaN values
            // We can only de-lag if we have enough valid history
            let val = if i < first + lag {
                // Not enough valid history for de-lagging, use regular value
                data[i]
            } else {
                // Apply de-lagging formula
                2.0 * data[i] - data[i - lag]
            };
            last_ema = alpha * val + (1.0 - alpha) * last_ema;
        }
        if i >= warm {
            out[i] = last_ema;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx2(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    zlema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx512(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    zlema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx512_short(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    zlema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx512_long(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    zlema_scalar(data, period, first_val, out)
}

#[inline]
pub fn zlema_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);
    let warm = first + period - 1;

    let mut last_ema = data[first];

    // Process all values starting from first
    for i in first..len {
        if i > first {
            // For de-lagging, we need to ensure we're not accessing NaN values
            // We can only de-lag if we have enough valid history
            let val = if i < first + lag {
                // Not enough valid history for de-lagging, use regular value
                data[i]
            } else {
                // Apply de-lagging formula
                2.0 * data[i] - data[i - lag]
            };
            last_ema = alpha * val + (1.0 - alpha) * last_ema;
        }
        if i >= warm {
            out[i] = last_ema;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[derive(Debug, Clone)]
pub struct ZlemaStream {
    period: usize,
    lag: usize,
    alpha: f64,
    last_ema: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl ZlemaStream {
    pub fn try_new(params: ZlemaParams) -> Result<Self, ZlemaError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(ZlemaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            lag: (period - 1) / 2,
            alpha: 2.0 / (period as f64 + 1.0),
            last_ema: f64::NAN,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Match batch behavior: NaN taints the EMA state
        if value.is_nan() {
            // Store NaN in buffer to propagate through de-lagging
            self.buffer[self.head] = value;

            // Advance head pointer
            self.head = (self.head + 1) % self.period;
            if !self.filled && self.head == 0 {
                self.filled = true;
            }

            // Taint the EMA state
            self.last_ema = f64::NAN;

            // Count this as a processed sample
            let samples_seen = if !self.filled {
                self.head
            } else {
                self.period + self.head
            };

            // Return NaN after warmup, None during warmup
            if samples_seen < self.period {
                None
            } else {
                Some(self.last_ema)
            }
        } else {
            // 1. store the new price in the circular buffer
            self.buffer[self.head] = value;

            // 2. how many *valid* samples have we processed so far?
            let samples_seen = if !self.filled {
                self.head + 1 // 0-based → count
            } else {
                self.period + self.head + 1 // already wrapped at least once
            };

            // 3. advance the head pointer & check wrap-around
            self.head = (self.head + 1) % self.period;
            if !self.filled && self.head == 0 {
                self.filled = true;
            }

            // 4. choose the correct input for the EMA core
            let val = if samples_seen <= self.lag {
                // still in the warm-up zone: no de-lagging yet
                value
            } else {
                let lag_idx = (self.head + self.period - self.lag - 1) % self.period;
                2.0 * value - self.buffer[lag_idx]
            };

            // 5. standard EMA recurrence
            // Initialize on first value, but once tainted with NaN, it stays NaN (matches batch behavior)
            if self.last_ema.is_nan() && samples_seen == 1 {
                // First value initialization (not tainted case)
                self.last_ema = val;
            } else {
                // Standard recurrence (preserves NaN if tainted)
                self.last_ema = self.alpha * val + (1.0 - self.alpha) * self.last_ema;
            }

            // 6. Only return values after warmup period (period - 1 samples)
            if samples_seen < self.period {
                None
            } else {
                Some(self.last_ema)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZlemaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ZlemaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 40, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ZlemaBatchBuilder {
    range: ZlemaBatchRange,
    kernel: Kernel,
}

impl ZlemaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<ZlemaBatchOutput, ZlemaError> {
        zlema_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ZlemaBatchOutput, ZlemaError> {
        ZlemaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ZlemaBatchOutput, ZlemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ZlemaBatchOutput, ZlemaError> {
        ZlemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn zlema_batch_with_kernel(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    k: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    // Auto-map non-batch kernels to their batch equivalents for better UX
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::Scalar => Kernel::ScalarBatch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        Kernel::Avx512 => Kernel::Avx512Batch,
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch, // Fallback for any unknown kernels
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    zlema_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ZlemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ZlemaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ZlemaBatchOutput {
    pub fn row_for_params(&self, p: &ZlemaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &ZlemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ZlemaBatchRange) -> Vec<ZlemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ZlemaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn zlema_batch_slice(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    zlema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn zlema_batch_par_slice(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    zlema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn zlema_batch_inner(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    // grid
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ZlemaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    if data.is_empty() {
        return Err(ZlemaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZlemaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ZlemaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // allocate uninit and stamp only warm prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // reinterpret once
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // write rows without touching the warm prefix
    let do_row = |row: usize, dst: &mut [f64]| {
        let p = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => zlema_row_scalar(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => zlema_row_avx2(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => zlema_row_avx512(data, first, p, 0, core::ptr::null(), 0.0, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    // move out the flat f64 buffer safely
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(ZlemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn expand_grid_zlema(r: &ZlemaBatchRange) -> Vec<ZlemaParams> {
    expand_grid(r)
}

/// Direct buffer write version for Python bindings
#[inline(always)]
pub fn zlema_batch_inner_into(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ZlemaParams>, ZlemaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ZlemaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    if data.is_empty() {
        return Err(ZlemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZlemaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ZlemaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Reinterpret the target as MaybeUninit and stamp ONLY warm prefixes once.
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    // Now write rows beyond warm
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let dst = unsafe {
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };
        match kern {
            Kernel::Scalar => zlema_row_scalar(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => zlema_row_avx2(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => zlema_row_avx512(data, first, p, 0, core::ptr::null(), 0.0, dst),
            _ => unreachable!(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_zlema_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ZlemaParams { period: None };
        let input = ZlemaInput::from_candles(&candles, "close", default_params);
        let output = zlema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_zlema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ZlemaInput::from_candles(&candles, "close", ZlemaParams::default());
        let result = zlema_with_kernel(&input, kernel)?;
        let expected_last_five = [59015.1, 59165.2, 59168.1, 59147.0, 58978.9];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] ZLEMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_zlema_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ZlemaParams { period: Some(0) };
        let input = ZlemaInput::from_slice(&input_data, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ZLEMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_zlema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ZlemaParams { period: Some(10) };
        let input = ZlemaInput::from_slice(&data_small, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ZLEMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_zlema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ZlemaParams { period: Some(14) };
        let input = ZlemaInput::from_slice(&single_point, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ZLEMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_zlema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ZlemaParams { period: Some(21) };
        let first_input = ZlemaInput::from_candles(&candles, "close", first_params);
        let first_result = zlema_with_kernel(&first_input, kernel)?;
        let second_params = ZlemaParams { period: Some(14) };
        let second_input = ZlemaInput::from_slice(&first_result.values, second_params);
        let second_result = zlema_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        // First result has warmup at index 20 (period 21 - 1)
        // So second calculation starts from index 20 and has warmup at 20 + 14 - 1 = 33
        // Values should be valid from index 33 onwards
        for (idx, &val) in second_result.values.iter().enumerate().skip(34) {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
        Ok(())
    }

    fn check_zlema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ZlemaInput::from_candles(&candles, "close", ZlemaParams::default());
        let res = zlema_with_kernel(&input, kernel)?;
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

    fn check_zlema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = ZlemaInput::from_candles(
            &candles,
            "close",
            ZlemaParams {
                period: Some(period),
            },
        );
        let batch_output = zlema_with_kernel(&input, kernel)?.values;

        let mut stream = ZlemaStream::try_new(ZlemaParams {
            period: Some(period),
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
                "[{}] ZLEMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_zlema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to catch uninitialized memory reads
        // Include edge cases like odd/even periods, small and large values
        let test_periods = vec![1, 2, 3, 5, 7, 10, 14, 20, 21, 30, 50, 100, 200];

        for period in test_periods {
            let params = ZlemaParams {
                period: Some(period),
            };
            let input = ZlemaInput::from_candles(&candles, "close", params);

            // Skip if period is too large for the data
            if period > candles.close.len() {
                continue;
            }

            let output = zlema_with_kernel(&input, kernel)?;

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
    fn check_zlema_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_zlema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // ZLEMA only has period parameter, no offset/sigma like ALMA
        // Strategy: period from 1..=100, then generate matching data
        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period.max(2)..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = ZlemaParams {
                    period: Some(period),
                };
                let input = ZlemaInput::from_slice(&data, params);

                let ZlemaOutput { values: out } = zlema_with_kernel(&input, kernel).unwrap();
                let ZlemaOutput { values: ref_out } =
                    zlema_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length should match input
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // Property 2: Warmup period check
                // ZLEMA starts calculating from the first non-NaN value, but uses a warmup period
                // The actual warmup is first + period - 1, where first is the first non-NaN index
                let first_non_nan = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup = first_non_nan + period - 1;

                // Essential: Values before first_non_nan input must be NaN
                for i in 0..first_non_nan.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN at index {} before first non-NaN input",
                        i
                    );
                }

                // Essential: After warmup period, must have valid calculated values
                for i in warmup..data.len() {
                    prop_assert!(
                        !out[i].is_nan(),
                        "Expected valid value after warmup at index {}",
                        i
                    );
                }

                // Implementation detail: During warmup (first_non_nan..warmup),
                // the implementation may choose to output NaN or calculated values.
                // The current implementation produces values, but this is not a requirement.

                // Property 3: De-lagging behavior verification
                // ZLEMA uses: lag = (period - 1) / 2
                // de-lagged value = 2.0 * data[i] - data[i - lag]
                let lag = (period - 1) / 2;
                let alpha = 2.0 / (period as f64 + 1.0);

                // Property 4: Values should be within reasonable bounds
                // After warmup, check that values are bounded
                for i in warmup..data.len() {
                    // For ZLEMA, we need to consider a wider window because of de-lagging
                    // The de-lagging looks back by 'lag' positions
                    let window_start = i.saturating_sub(period + lag);
                    let window_end = i.min(data.len() - 1);
                    let window = &data[window_start..=window_end];

                    // Get min/max of the extended window for bounds checking
                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];

                    // ZLEMA can overshoot due to de-lagging formula: 2*current - lagged
                    // This can theoretically push values to 2*hi - lo or 2*lo - hi
                    let extended_lo = 2.0 * lo - hi;
                    let extended_hi = 2.0 * hi - lo;

                    prop_assert!(
                        y.is_nan() || (y >= extended_lo - 1e-9 && y <= extended_hi + 1e-9),
                        "idx {}: {} ∉ [{}, {}] (extended bounds for de-lagging)",
                        i,
                        y,
                        extended_lo,
                        extended_hi
                    );
                }

                // Property 5: Period=1 edge case
                // With period=1, lag=0, so de-lagged value = 2*data[i] - data[i] = data[i]
                // And alpha = 2/2 = 1, so EMA = data[i]
                if period == 1 && data.len() > 0 {
                    for i in 1..data.len() {
                        let expected = data[i];
                        let actual = out[i];
                        prop_assert!(
                            (actual - expected).abs() <= 1e-9,
                            "Period=1 mismatch at {}: expected {}, got {}",
                            i,
                            expected,
                            actual
                        );
                    }
                }

                // Property 6: Constant data should converge to that constant
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && data.len() > warmup {
                    // After sufficient iterations, ZLEMA should converge to the constant value
                    let constant_val = data[first_non_nan];
                    for i in (warmup + period * 2)..data.len() {
                        prop_assert!(
                            (out[i] - constant_val).abs() <= 1e-6,
                            "Constant data convergence failed at {}: expected {}, got {}",
                            i,
                            constant_val,
                            out[i]
                        );
                    }
                }

                // Property 7: Cross-kernel validation
                // Compare against scalar reference implementation
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Handle NaN/infinity cases
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "NaN/Inf mismatch at idx {}: {} vs {}",
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

                    // ZLEMA is relatively simple, so use tighter ULP tolerance
                    let max_ulp = if matches!(kernel, Kernel::Avx512) {
                        10
                    } else {
                        5
                    };

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= max_ulp,
                        "Cross-kernel mismatch at idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_zlema_tests {
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

    generate_all_zlema_tests!(
        check_zlema_partial_params,
        check_zlema_accuracy,
        check_zlema_zero_period,
        check_zlema_period_exceeds_length,
        check_zlema_very_small_dataset,
        check_zlema_reinput,
        check_zlema_nan_handling,
        check_zlema_streaming,
        check_zlema_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_zlema_tests!(check_zlema_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ZlemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = ZlemaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple different batch configurations to catch edge cases
        // ZLEMA uses lag = (period - 1) / 2, so test odd/even periods
        let batch_configs = vec![
            (1, 10, 1),   // All small periods including edge cases
            (3, 21, 3),   // Odd periods with gaps
            (2, 20, 2),   // Even periods
            (10, 50, 10), // Larger periods
            (7, 7, 1),    // Single odd period (edge case)
            (8, 8, 1),    // Single even period (edge case)
            (5, 100, 5),  // Wide range with step
        ];

        for (start, end, step) in batch_configs {
            // Skip if the largest period exceeds data length
            if end > c.close.len() {
                continue;
            }

            let output = ZlemaBatchBuilder::new()
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
                let period = if row < output.combos.len() {
                    output.combos[row].period.unwrap_or(0)
                } else {
                    0
                };

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
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

/// Zero-copy version that writes directly into a provided buffer
#[inline]
pub fn zlema_compute_into(
    input: &ZlemaInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), ZlemaError> {
    let (data, first, period, warm) = zlema_validate(input)?;
    if out.len() != data.len() {
        return Err(ZlemaError::InvalidPeriod {
            period: out.len(),
            data_len: data.len(),
        });
    }

    // Initialize the warmup period with NaN
    out[..warm].fill(f64::NAN);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Write directly into the provided buffer
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                zlema_scalar(data, period, first, out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                zlema_avx2(data, period, first, out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                zlema_avx512(data, period, first, out);
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// WASM-optimized helper function that writes directly to output slice - no allocations
#[inline]
pub fn zlema_into_slice(
    dst: &mut [f64],
    input: &ZlemaInput,
    kern: Kernel,
) -> Result<(), ZlemaError> {
    let (data, first, period, warm) = zlema_validate(input)?;
    if dst.len() != data.len() {
        return Err(ZlemaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    // Initialize only the prefix, zero-copy style
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => zlema_scalar(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => zlema_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => zlema_avx512(data, period, first, dst),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "zlema")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Zero Lag Exponential Moving Average (ZLEMA) of the input data.
///
/// ZLEMA reduces lag by de-lagging the input before EMA calculation.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of ZLEMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period is 0 or exceeds data length).
pub fn zlema_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = ZlemaParams {
        period: Some(period),
    };
    let zlema_in = ZlemaInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| zlema_with_kernel(&zlema_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ZlemaStream")]
pub struct ZlemaStreamPy {
    stream: ZlemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ZlemaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = ZlemaParams {
            period: Some(period),
        };
        let stream =
            ZlemaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(ZlemaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated ZLEMA value.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "zlema_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute ZLEMA for multiple period values in a single pass.
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
pub fn zlema_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = ZlemaBatchRange {
        period: period_range,
    };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (OK for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py
        .allow_threads(|| {
            // Handle kernel selection for batch operations
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // Map batch kernels to regular kernels for ZLEMA
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };

            zlema_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = ZlemaParams {
        period: Some(period),
    };
    let input = ZlemaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    zlema_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = ZlemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    zlema_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = ZlemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ZlemaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ZlemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ZlemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = zlema_batch)]
pub fn zlema_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: ZlemaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = ZlemaBatchRange {
        period: config.period_range,
    };

    // Use batch kernel detection and conversion like alma does
    let kernel = match Kernel::Auto {
        Kernel::Auto => {
            let batch_kernel = detect_best_batch_kernel();
            match batch_kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => Kernel::Scalar,
            }
        }
        k => k,
    };

    let output = zlema_batch_inner(data, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = ZlemaBatchJsOutput {
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
pub fn zlema_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to zlema_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = ZlemaParams {
            period: Some(period),
        };
        let input = ZlemaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // CRITICAL: Aliasing check - handle in-place operations
            let mut temp = vec![0.0; len];
            zlema_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            zlema_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to zlema_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = ZlemaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);

        // Use batch kernel detection and conversion
        let kernel = match Kernel::Auto {
            Kernel::Auto => {
                let batch_kernel = detect_best_batch_kernel();
                match batch_kernel {
                    Kernel::Avx512Batch => Kernel::Avx512,
                    Kernel::Avx2Batch => Kernel::Avx2,
                    Kernel::ScalarBatch => Kernel::Scalar,
                    _ => Kernel::Scalar,
                }
            }
            k => k,
        };

        zlema_batch_inner_into(data, &sweep, kernel, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
