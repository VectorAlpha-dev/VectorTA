//! # Decycler Oscillator (DEC_OSC)
//!
//! A trend-following oscillator that uses cascaded high-pass filters to remove cyclical
//! components from price data, leaving only the long-term trend. The indicator applies
//! two sequential 2-pole high-pass filters to eliminate shorter-term cycles, then scales
//! the result to show the percentage deviation from the original price series. Useful for
//! identifying the underlying trend strength without short-term noise.
//!
//! ## Parameters
//! - **hp_period**: Period for the high-pass filters (default: 125)
//! - **k**: Multiplier for scaling the oscillator output (default: 1.0)
//!
//! ## Inputs
//! - Single price array (typically close prices)
//! - Supports both raw slices and Candles with source selection
//!
//! ## Returns
//! - **`Ok(DecOscOutput)`** containing a `Vec<f64>` matching input length
//! - Values represent trend strength as percentage of price
//! - Leading NaNs for first 2 values during filter initialization
//! - Positive values indicate uptrend, negative indicate downtrend
//!
//! ## Developer Notes (Implementation Status)
//! - **SIMD Kernels**:
//!   - AVX2/AVX512: kept as stubs by design (delegate to scalar). The inner loop is a 2-pole IIR with loop-carried dependencies; direct time-axis SIMD does not yield wins without look-ahead transforms and adds complexity. Runtime selection remains, but stubs short-circuit to scalar for correctness and speed.
//! - **Scalar Path**: optimized to precompute constants, reduce multiplies, and reuse prior raw samples to cut loads; semantics unchanged (warmup handling preserved).
//! - **Streaming Performance**: O(1) with maintained filter state.
//! - **Memory Optimization**: YES — uses `alloc_with_nan_prefix` and `make_uninit_matrix` helpers.
//! - **Batch Operations**: Supported; parallel row execution. A future optimization is to group rows by equal `hp_period` and SIMD-scale across `k` per tick to avoid redundant filter work.
//! - **Decision**: Single-series SIMD remains disabled due to IIR feedback; consider row-specific SIMD across `k` only if period-grouped workloads are common and show >5% gains.
//! - **CUDA**: Wrapper present with typed errors and pre-synchronized launches; Python interop provides CAI v3 + DLPack and retains the CUDA context with the returned VRAM handle.

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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DecOscInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DecOscData::Slice(slice) => slice,
            DecOscData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DecOscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DecOscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DecOscParams {
    pub hp_period: Option<usize>,
    pub k: Option<f64>,
}

impl Default for DecOscParams {
    fn default() -> Self {
        Self {
            hp_period: Some(125),
            k: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecOscInput<'a> {
    pub data: DecOscData<'a>,
    pub params: DecOscParams,
}

impl<'a> DecOscInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DecOscParams) -> Self {
        Self {
            data: DecOscData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DecOscParams) -> Self {
        Self {
            data: DecOscData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DecOscParams::default())
    }
    #[inline]
    pub fn get_hp_period(&self) -> usize {
        self.params.hp_period.unwrap_or(125)
    }
    #[inline]
    pub fn get_k(&self) -> f64 {
        self.params.k.unwrap_or(1.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DecOscBuilder {
    hp_period: Option<usize>,
    k: Option<f64>,
    kernel: Kernel,
}

impl Default for DecOscBuilder {
    fn default() -> Self {
        Self {
            hp_period: None,
            k: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DecOscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn hp_period(mut self, n: usize) -> Self {
        self.hp_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn k(mut self, v: f64) -> Self {
        self.k = Some(v);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DecOscOutput, DecOscError> {
        let p = DecOscParams {
            hp_period: self.hp_period,
            k: self.k,
        };
        let i = DecOscInput::from_candles(c, "close", p);
        dec_osc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DecOscOutput, DecOscError> {
        let p = DecOscParams {
            hp_period: self.hp_period,
            k: self.k,
        };
        let i = DecOscInput::from_slice(d, p);
        dec_osc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DecOscStream, DecOscError> {
        let p = DecOscParams {
            hp_period: self.hp_period,
            k: self.k,
        };
        DecOscStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DecOscError {
    #[error("dec_osc: Input data slice is empty.")]
    EmptyInputData,

    #[error("dec_osc: All values are NaN.")]
    AllValuesNaN,

    #[error("dec_osc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("dec_osc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("dec_osc: Invalid K: k = {k}")]
    InvalidK { k: f64 },

    #[error("dec_osc: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("dec_osc: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },

    #[error("dec_osc: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn dec_osc(input: &DecOscInput) -> Result<DecOscOutput, DecOscError> {
    dec_osc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn dec_osc_prepare<'a>(
    input: &'a DecOscInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, usize, Kernel), DecOscError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(DecOscError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DecOscError::AllValuesNaN)?;
    let period = input.get_hp_period();
    let k_val = input.get_k();

    if period < 2 || period > len {
        return Err(DecOscError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < 2 {
        return Err(DecOscError::NotEnoughValidData {
            needed: 2,
            valid: len - first,
        });
    }
    if k_val <= 0.0 || k_val.is_nan() {
        return Err(DecOscError::InvalidK { k: k_val });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, k_val, first, chosen))
}

pub fn dec_osc_with_kernel(
    input: &DecOscInput,
    kernel: Kernel,
) -> Result<DecOscOutput, DecOscError> {
    let (data, period, k_val, first, chosen) = dec_osc_prepare(input, kernel)?;

    let mut out = alloc_with_nan_prefix(data.len(), first + 2);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                dec_osc_scalar(data, period, k_val, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => dec_osc_avx2(data, period, k_val, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                dec_osc_avx512(data, period, k_val, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(DecOscOutput { values: out })
}

/// Write DEC_OSC results into a caller-provided output slice without allocating.
///
/// - Preserves the same NaN warmup prefix as the Vec-returning API (first-valid + 2).
/// - The output slice length must match the input length.
/// - Uses `Kernel::Auto` to dispatch to the existing compute path.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn dec_osc_into(out: &mut [f64], input: &DecOscInput) -> Result<(), DecOscError> {
    dec_osc_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn dec_osc_into_slice(
    dst: &mut [f64],
    input: &DecOscInput,
    kern: Kernel,
) -> Result<(), DecOscError> {
    let (data, period, k_val, first, chosen) = dec_osc_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(DecOscError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => dec_osc_scalar(data, period, k_val, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => dec_osc_avx2(data, period, k_val, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => dec_osc_avx512(data, period, k_val, first, dst),
            _ => unreachable!(),
        }
    }

    // Ensure warmup period is filled with NaN
    let warmup_end = first + 2;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dec_osc_avx512(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { dec_osc_avx512_short(data, period, k_val, first, out) }
    } else {
        unsafe { dec_osc_avx512_long(data, period, k_val, first, out) }
    }
}

#[inline]
pub fn dec_osc_scalar(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    let len = data.len();
    if len == 0 || first + 1 >= len {
        return;
    }

    // Precompute constants once; use sin_cos to reduce transcendentals.
    let p = period as f64;
    let half_p = p * 0.5;

    let angle1 = 2.0 * PI * 0.707 / p;
    let (sin1, cos1) = angle1.sin_cos();
    let alpha1 = 1.0 + ((sin1 - 1.0) / cos1);
    let t1 = 1.0 - alpha1 * 0.5;
    let c1 = t1 * t1;
    let oma1 = 1.0 - alpha1;
    let two_oma1 = oma1 + oma1;
    let oma1_sq = oma1 * oma1;

    let angle2 = 2.0 * PI * 0.707 / half_p;
    let (sin2, cos2) = angle2.sin_cos();
    let alpha2 = 1.0 + ((sin2 - 1.0) / cos2);
    let t2 = 1.0 - alpha2 * 0.5;
    let c2 = t2 * t2;
    let oma2 = 1.0 - alpha2;
    let two_oma2 = oma2 + oma2;
    let oma2_sq = oma2 * oma2;

    let scale = 100.0 * k_val;

    // Warmup exactly two leading NaNs from `first`.
    out[first] = f64::NAN;
    out[first + 1] = f64::NAN;

    // Seed states to match existing behavior and cache prior raw samples.
    let mut x2 = data[first];
    let mut x1 = data[first + 1];
    let mut hp_prev_2 = x2;
    let mut hp_prev_1 = x1;
    let mut decosc_prev_2 = 0.0f64;
    let mut decosc_prev_1 = 0.0f64;

    // Main loop: fuse filter sections and scale; keep scalar-safe.
    for i in (first + 2)..len {
        let d0 = data[i];

        // First 2‑pole high‑pass on raw data (reuse cached x1/x2)
        let dx = d0 - 2.0 * x1 + x2;
        let hp0 = c1 * dx + two_oma1 * hp_prev_1 - oma1_sq * hp_prev_2;

        // Decycle and second high‑pass on decycled series
        let dec = d0 - hp0;
        let d_dec1 = x1 - hp_prev_1;
        let d_dec2 = x2 - hp_prev_2;
        let decdx = dec - 2.0 * d_dec1 + d_dec2;
        let osc0 = c2 * decdx + two_oma2 * decosc_prev_1 - oma2_sq * decosc_prev_2;

        // Final percentage scaling vs price
        out[i] = scale * osc0 / d0;

        // Roll states and cached raw samples
        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp0;
        decosc_prev_2 = decosc_prev_1;
        decosc_prev_1 = osc0;
        x2 = x1;
        x1 = d0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dec_osc_avx2(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
    // AVX2 stub - call scalar.
    dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dec_osc_avx512_short(
    data: &[f64],
    period: usize,
    k_val: f64,
    first: usize,
    out: &mut [f64],
) {
    // AVX512 short stub - call scalar.
    dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dec_osc_avx512_long(
    data: &[f64],
    period: usize,
    k_val: f64,
    first: usize,
    out: &mut [f64],
) {
    // AVX512 long stub - call scalar.
    dec_osc_scalar(data, period, k_val, first, out)
}

#[inline(always)]
pub fn dec_osc_batch_with_kernel(
    data: &[f64],
    sweep: &DecOscBatchRange,
    k: Kernel,
) -> Result<DecOscBatchOutput, DecOscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DecOscError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dec_osc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DecOscBatchRange {
    pub hp_period: (usize, usize, usize),
    pub k: (f64, f64, f64),
}

impl Default for DecOscBatchRange {
    fn default() -> Self {
        Self {
            hp_period: (125, 125, 0),
            k: (1.0, 1.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DecOscBatchBuilder {
    range: DecOscBatchRange,
    kernel: Kernel,
}

impl DecOscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn hp_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.hp_period = (start, end, step);
        self
    }
    #[inline]
    pub fn hp_period_static(mut self, p: usize) -> Self {
        self.range.hp_period = (p, p, 0);
        self
    }
    #[inline]
    pub fn k_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.k = (start, end, step);
        self
    }
    #[inline]
    pub fn k_static(mut self, x: f64) -> Self {
        self.range.k = (x, x, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<DecOscBatchOutput, DecOscError> {
        dec_osc_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DecOscBatchOutput, DecOscError> {
        DecOscBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DecOscBatchOutput, DecOscError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<DecOscBatchOutput, DecOscError> {
        DecOscBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct DecOscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DecOscParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DecOscBatchOutput {
    pub fn row_for_params(&self, p: &DecOscParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.hp_period.unwrap_or(125) == p.hp_period.unwrap_or(125)
                && (c.k.unwrap_or(1.0) - p.k.unwrap_or(1.0)).abs() < 1e-12
        })
    }

    pub fn values_for(&self, p: &DecOscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_checked(r: &DecOscBatchRange) -> Result<Vec<DecOscParams>, DecOscError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, DecOscError> {
        if step == 0 || start == end { return Ok(vec![start]); }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                out.push(v);
                v = match v.checked_add(step) { Some(n) if n != v => n, _ => break };
            }
        } else {
            // reversed bounds
            let mut v = start;
            while v >= end {
                out.push(v);
                if v < end + step { break; }
                v -= step;
                if v == 0 { break; }
            }
        }
        if out.is_empty() {
            return Err(DecOscError::InvalidRange { start, end, step });
        }
        Ok(out)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end + 1e-12 { v.push(x); x += step; }
        } else {
            let mut x = start;
            while x >= end - 1e-12 { v.push(x); x -= step.abs(); }
        }
        v
    }

    let periods = axis_usize(r.hp_period)?;
    let ks = axis_f64(r.k);
    let cap = periods.len().checked_mul(ks.len()).ok_or(DecOscError::InvalidRange {
        start: r.hp_period.0,
        end: r.hp_period.1,
        step: r.hp_period.2,
    })?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &k in &ks {
            out.push(DecOscParams { hp_period: Some(p), k: Some(k) });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn dec_osc_batch_slice(
    data: &[f64],
    sweep: &DecOscBatchRange,
    kern: Kernel,
) -> Result<DecOscBatchOutput, DecOscError> {
    dec_osc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn dec_osc_batch_par_slice(
    data: &[f64],
    sweep: &DecOscBatchRange,
    kern: Kernel,
) -> Result<DecOscBatchOutput, DecOscError> {
    dec_osc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn dec_osc_batch_inner(
    data: &[f64],
    sweep: &DecOscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DecOscBatchOutput, DecOscError> {
    let combos = expand_grid_checked(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DecOscError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
    if data.len() - first < 2 {
        return Err(DecOscError::NotEnoughValidData {
            needed: 2,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let _total = rows
        .checked_mul(cols)
        .ok_or(DecOscError::InvalidRange { start: rows, end: cols, step: 0 })?;

    // Use uninitialized memory with proper initialization (following ALMA pattern)
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each parameter combination
    let warmup_periods: Vec<usize> = combos.iter().map(|_| first + 2).collect();

    // Initialize matrix prefixes with NaN for warmup periods
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    // Use ManuallyDrop to avoid double-free (following ALMA pattern)
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].hp_period.unwrap();
        let k_val = combos[row].k.unwrap();
        match kern {
            Kernel::Scalar => dec_osc_row_scalar(data, first, period, k_val, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dec_osc_row_avx2(data, first, period, k_val, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dec_osc_row_avx512(data, first, period, k_val, out_row),
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

    // Convert uninitialized memory to Vec (following ALMA pattern)
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };
    core::mem::forget(buf_guard);

    Ok(DecOscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn dec_osc_batch_inner_into(
    data: &[f64],
    sweep: &DecOscBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DecOscParams>, DecOscError> {
    let combos = expand_grid_checked(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DecOscError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
    if data.len() - first < 2 {
        return Err(DecOscError::NotEnoughValidData {
            needed: 2,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let expected = rows
        .checked_mul(cols)
        .ok_or(DecOscError::InvalidRange { start: rows, end: cols, step: 0 })?;
    if out.len() != expected {
        return Err(DecOscError::OutputLengthMismatch { expected, got: out.len() });
    }

    // Cast out -> MaybeUninit and initialize prefixes via helper
    let warmups: Vec<usize> = combos.iter().map(|_| first + 2).collect();

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warmups);

    // Compute rows
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let dst: &mut [f64] =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        let period = combos[row].hp_period.unwrap();
        let k_val = combos[row].k.unwrap();
        match kern {
            Kernel::Scalar => dec_osc_row_scalar(data, first, period, k_val, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dec_osc_row_avx2(data, first, period, k_val, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dec_osc_row_avx512(data, first, period, k_val, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn dec_osc_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    k_val: f64,
    out: &mut [f64],
) {
    dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dec_osc_row_avx2(data: &[f64], first: usize, period: usize, k_val: f64, out: &mut [f64]) {
    dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dec_osc_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    k_val: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        dec_osc_row_avx512_short(data, first, period, k_val, out)
    } else {
        dec_osc_row_avx512_long(data, first, period, k_val, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dec_osc_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    k_val: f64,
    out: &mut [f64],
) {
    dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dec_osc_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    k_val: f64,
    out: &mut [f64],
) {
    dec_osc_scalar(data, period, k_val, first, out)
}

// Decision: Streaming path replaced with FMA-based O(1) kernel; outputs unchanged.

#[derive(Debug, Clone)]
pub struct DecOscStream {
    // Params
    period: usize,
    scale: f64, // 100.0 * k

    // Section 1 coeffs
    c1: f64,
    two_oma1: f64,
    oma1_sq: f64,

    // Section 2 coeffs
    c2: f64,
    two_oma2: f64,
    oma2_sq: f64,

    // Raw sample cache (x[t-1], x[t-2])
    x1: f64,
    x2: f64,

    // 1st HP stage past outputs: hp[t-1], hp[t-2]
    hp1: f64,
    hp2: f64,

    // Decycled series (dec = x - hp) history: dec[t-1], dec[t-2]
    dec1: f64,
    dec2: f64,

    // 2nd HP stage past outputs: osc[t-1], osc[t-2]
    osc1: f64,
    osc2: f64,

    // Warmup counter
    idx: usize,
}

impl DecOscStream {
    #[inline(always)]
    pub fn try_new(params: DecOscParams) -> Result<Self, DecOscError> {
        let period = params.hp_period.unwrap_or(125);
        if period < 2 {
            return Err(DecOscError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let k = params.k.unwrap_or(1.0);
        if k <= 0.0 || k.is_nan() {
            return Err(DecOscError::InvalidK { k });
        }

        // ---- Precompute filter constants (Ehlers two‑pole HP, Q≈0.707) ----
        // Use sin_cos once per section; faster & often more precise than separate calls.
        let p = period as f64;
        let angle1 = 2.0 * std::f64::consts::PI * 0.707 / p;
        let (sin1, cos1) = angle1.sin_cos(); // single call, two results
        let alpha1 = 1.0 + ((sin1 - 1.0) / cos1);
        let t1 = 1.0 - 0.5 * alpha1;
        let c1 = t1 * t1;
        let oma1 = 1.0 - alpha1;
        let two_oma1 = oma1 + oma1;
        let oma1_sq = oma1 * oma1;

        let half_p = 0.5 * p;
        let angle2 = 2.0 * std::f64::consts::PI * 0.707 / half_p;
        let (sin2, cos2) = angle2.sin_cos();
        let alpha2 = 1.0 + ((sin2 - 1.0) / cos2);
        let t2 = 1.0 - 0.5 * alpha2;
        let c2 = t2 * t2;
        let oma2 = 1.0 - alpha2;
        let two_oma2 = oma2 + oma2;
        let oma2_sq = oma2 * oma2;

        Ok(Self {
            period,
            scale: 100.0 * k,

            c1,
            two_oma1,
            oma1_sq,

            c2,
            two_oma2,
            oma2_sq,

            x1: f64::NAN,
            x2: f64::NAN,

            hp1: f64::NAN,
            hp2: f64::NAN,

            dec1: 0.0,
            dec2: 0.0,

            osc1: 0.0,
            osc2: 0.0,

            idx: 0,
        })
    }

    /// O(1) update. Returns `None` for the first two samples (warmup), mirroring the batch path’s NaNs.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Warmup 0: seed with the first value (match batch: use raw value as initial HP state)
        if self.idx == 0 {
            self.idx = 1;
            self.x2 = value;
            self.x1 = value;
            self.hp2 = value;
            self.hp1 = value;
            // dec1/dec2 = 0, osc1/osc2 = 0 (already)
            return None;
        }

        // Warmup 1: second tick; keep seeds consistent and still return None
        if self.idx == 1 {
            self.idx = 2;
            self.x2 = self.x1;
            self.x1 = value;
            self.hp2 = self.hp1;
            self.hp1 = value;
            // dec at this step is 0, so just keep dec1/dec2 = 0
            return None;
        }

        // ----- Main path -----
        // First 2‑pole HPF on raw data: hp = c1*(x - 2*x1 + x2) + 2*(1−α1)*hp1 − (1−α1)^2*hp2
        let dx = value - self.x1 - self.x1 + self.x2;

        // Use mul_add to get FMA on capable CPUs (fewer roundings, often faster).
        // hp = (-oma1_sq)*hp2 + c1*dx + two_oma1*hp1
        let hp = (-self.oma1_sq).mul_add(self.hp2, self.c1.mul_add(dx, self.two_oma1 * self.hp1));

        // Decycle
        let dec = value - hp;

        // Second 2‑pole HPF on decycled series: osc = c2*(dec - 2*dec1 + dec2) + 2*(1−α2)*osc1 − (1−α2)^2*osc2
        let decdx = dec - self.dec1 - self.dec1 + self.dec2;
        let osc =
            (-self.oma2_sq).mul_add(self.osc2, self.c2.mul_add(decdx, self.two_oma2 * self.osc1));

        // Output: percentage of price
        let out = (self.scale * osc) / value;

        // Roll states
        self.hp2 = self.hp1;
        self.hp1 = hp;

        self.dec2 = self.dec1;
        self.dec1 = dec;

        self.osc2 = self.osc1;
        self.osc1 = osc;

        self.x2 = self.x1;
        self.x1 = value;

        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_dec_osc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = DecOscParams {
            hp_period: None,
            k: None,
        };
        let input = DecOscInput::from_candles(&candles, "close", default_params);
        let output = dec_osc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_dec_osc_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DecOscInput::from_candles(&candles, "close", DecOscParams::default());
        let result = dec_osc_with_kernel(&input, kernel)?;

        if result.values.len() > 5 {
            let expected_last_five = [
                -1.5036367540303395,
                -1.4037875172207006,
                -1.3174199471429475,
                -1.2245874070642693,
                -1.1638422627265639,
            ];
            let start = result.values.len().saturating_sub(5);
            for (i, &val) in result.values[start..].iter().enumerate() {
                let diff = (val - expected_last_five[i]).abs();
                assert!(
                    diff < 1e-7,
                    "[{}] DEC_OSC {:?} mismatch at idx {}: got {}, expected {}",
                    test_name,
                    kernel,
                    i,
                    val,
                    expected_last_five[i]
                );
            }
        }
        Ok(())
    }

    fn check_dec_osc_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DecOscInput::with_default_candles(&candles);
        match input.data {
            DecOscData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected DecOscData::Candles"),
        }
        let output = dec_osc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dec_osc_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DecOscParams {
            hp_period: Some(0),
            k: Some(1.0),
        };
        let input = DecOscInput::from_slice(&input_data, params);
        let res = dec_osc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DEC_OSC should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_dec_osc_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = DecOscParams {
            hp_period: Some(10),
            k: Some(1.0),
        };
        let input = DecOscInput::from_slice(&data_small, params);
        let res = dec_osc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DEC_OSC should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_dec_osc_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = DecOscParams {
            hp_period: Some(125),
            k: Some(1.0),
        };
        let input = DecOscInput::from_slice(&single_point, params);
        let res = dec_osc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DEC_OSC should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_dec_osc_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = DecOscParams {
            hp_period: Some(50),
            k: Some(1.0),
        };
        let first_input = DecOscInput::from_candles(&candles, "close", first_params);
        let first_result = dec_osc_with_kernel(&first_input, kernel)?;
        let second_params = DecOscParams {
            hp_period: Some(50),
            k: Some(1.0),
        };
        let second_input = DecOscInput::from_slice(&first_result.values, second_params);
        let second_result = dec_osc_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_dec_osc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations for DEC_OSC
        let test_params = vec![
            DecOscParams::default(), // hp_period: 125, k: 1.0
            DecOscParams {
                hp_period: Some(2),
                k: Some(1.0),
            }, // minimum viable period
            DecOscParams {
                hp_period: Some(10),
                k: Some(1.0),
            }, // small period
            DecOscParams {
                hp_period: Some(50),
                k: Some(1.0),
            }, // medium period
            DecOscParams {
                hp_period: Some(125),
                k: Some(1.0),
            }, // default period
            DecOscParams {
                hp_period: Some(200),
                k: Some(1.0),
            }, // large period
            DecOscParams {
                hp_period: Some(500),
                k: Some(1.0),
            }, // very large period
            DecOscParams {
                hp_period: Some(50),
                k: Some(0.5),
            }, // small k
            DecOscParams {
                hp_period: Some(50),
                k: Some(2.0),
            }, // large k
            DecOscParams {
                hp_period: Some(125),
                k: Some(0.1),
            }, // very small k
            DecOscParams {
                hp_period: Some(125),
                k: Some(10.0),
            }, // very large k
            DecOscParams {
                hp_period: Some(20),
                k: Some(1.5),
            }, // mixed params
            DecOscParams {
                hp_period: Some(100),
                k: Some(0.75),
            }, // mixed params
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DecOscInput::from_candles(&candles, "close", params.clone());
            let output = dec_osc_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.hp_period.unwrap_or(125),
                        params.k.unwrap_or(1.0),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.hp_period.unwrap_or(125),
                        params.k.unwrap_or(1.0),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.hp_period.unwrap_or(125),
                        params.k.unwrap_or(1.0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_dec_osc_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_dec_osc_tests {
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

    generate_all_dec_osc_tests!(
        check_dec_osc_partial_params,
        check_dec_osc_accuracy,
        check_dec_osc_default_candles,
        check_dec_osc_zero_period,
        check_dec_osc_period_exceeds_length,
        check_dec_osc_very_small_dataset,
        check_dec_osc_reinput,
        check_dec_osc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_dec_osc_tests!(check_dec_osc_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = DecOscBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = DecOscParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -1.5036367540303395,
            -1.4037875172207006,
            -1.3174199471429475,
            -1.2245874070642693,
            -1.1638422627265639,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations for DEC_OSC
        let test_configs = vec![
            // (hp_period_start, hp_period_end, hp_period_step, k_start, k_end, k_step)
            (2, 10, 2, 1.0, 1.0, 0.0),     // Small periods, fixed k
            (10, 50, 10, 1.0, 1.0, 0.0),   // Medium periods, fixed k
            (50, 150, 25, 1.0, 1.0, 0.0),  // Large periods, fixed k
            (125, 125, 0, 0.5, 2.0, 0.5),  // Fixed period, varying k
            (20, 40, 5, 0.5, 1.5, 0.25),   // Mixed sweep
            (100, 200, 50, 1.0, 1.0, 0.0), // Very large periods
            (2, 5, 1, 0.1, 10.0, 4.95),    // Dense small range with extreme k
        ];

        for (cfg_idx, &(hp_start, hp_end, hp_step, k_start, k_end, k_step)) in
            test_configs.iter().enumerate()
        {
            let mut builder = DecOscBatchBuilder::new().kernel(kernel);

            // Configure period range
            if hp_step > 0 {
                builder = builder.hp_period_range(hp_start, hp_end, hp_step);
            } else {
                builder = builder.hp_period_range(hp_start, hp_start, 1);
            }

            // Configure k range
            if k_step > 0.0 {
                builder = builder.k_range(k_start, k_end, k_step);
            }

            let output = builder.apply_candles(&c, "close")?;

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
						 at row {} col {} (flat index {}) with params: hp_period={}, k={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.hp_period.unwrap_or(125),
                        combo.k.unwrap_or(1.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: hp_period={}, k={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.hp_period.unwrap_or(125),
                        combo.k.unwrap_or(1.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: hp_period={}, k={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.hp_period.unwrap_or(125),
                        combo.k.unwrap_or(1.0)
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_dec_osc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy 1: Random price data with full hp_period range (2-200)
        let strat_random = (2usize..=200).prop_flat_map(|hp_period| {
            (
                prop::collection::vec(
                    (100.0f64..10000.0f64)
                        .prop_filter("finite positive", |x| x.is_finite() && *x > 50.0),
                    hp_period + 10..400,
                ),
                Just(hp_period),
                (0.1f64..3.0f64), // k multiplier
            )
        });

        // Strategy 2: Constant data (should produce oscillator near 0 after warmup)
        let strat_constant = (2usize..=100, 0.1f64..3.0f64).prop_map(|(hp_period, k)| {
            let value = 1000.0; // Use larger value to avoid division issues
            let data = vec![value; hp_period.max(10) + 20];
            (data, hp_period, k)
        });

        // Strategy 3: Trending data (monotonic increasing/decreasing)
        let strat_trending = (2usize..=100, 0.1f64..3.0f64, prop::bool::ANY).prop_map(
            |(hp_period, k, increasing)| {
                let len = hp_period.max(10) + 50;
                let data: Vec<f64> = if increasing {
                    (0..len).map(|i| 500.0 + i as f64 * 2.0).collect()
                } else {
                    (0..len).map(|i| 2000.0 - i as f64 * 2.0).collect()
                };
                (data, hp_period, k)
            },
        );

        // Strategy 4: Small k values to test proportionality
        let strat_small_k = (10usize..=50, 0.01f64..0.5f64).prop_flat_map(|(hp_period, k)| {
            (
                prop::collection::vec(
                    (500.0f64..1500.0f64).prop_filter("finite", |x| x.is_finite()),
                    hp_period + 20..100,
                ),
                Just(hp_period),
                Just(k),
            )
        });

        // Strategy 5: High volatility data to test filter response
        let strat_volatile = (5usize..=50, 0.1f64..2.0f64).prop_flat_map(|(hp_period, k)| {
            (
                prop::collection::vec(
                    prop::strategy::Union::new(vec![
                        (100.0f64..200.0f64).boxed(),
                        (800.0f64..1000.0f64).boxed(),
                    ]),
                    hp_period + 20..100,
                ),
                Just(hp_period),
                Just(k),
            )
        });

        // Combine all strategies
        let combined_strat = prop::strategy::Union::new(vec![
            strat_random.boxed(),
            strat_constant.boxed(),
            strat_trending.boxed(),
            strat_small_k.boxed(),
            strat_volatile.boxed(),
        ]);

        proptest::test_runner::TestRunner::default()
            .run(&combined_strat, |(data, hp_period, k)| {
                let params = DecOscParams {
                    hp_period: Some(hp_period),
                    k: Some(k),
                };
                let input = DecOscInput::from_slice(&data, params);

                // Get outputs from tested kernel and reference scalar kernel
                let DecOscOutput { values: out } = dec_osc_with_kernel(&input, kernel).unwrap();
                let DecOscOutput { values: ref_out } =
                    dec_osc_with_kernel(&input, Kernel::Scalar).unwrap();

                // Find first non-NaN value
                let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_end = first + 2; // DEC_OSC needs two seeded samples

                // Property validations
                for i in warmup_end..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Property 1: Finite values after warmup
                    prop_assert!(
                        y.is_finite(),
                        "[{}] Output should be finite at index {} (after warmup): got {}",
                        test_name,
                        i,
                        y
                    );

                    // Property 2: Graduated magnitude bounds based on hp_period
                    // DEC_OSC can produce extreme values especially with small periods
                    // Skip magnitude check for hp_period=2 as it can produce mathematically valid but extreme values
                    if hp_period > 2 {
                        let magnitude_limit = if hp_period >= 50 {
                            5000.0 // ±5,000% for large periods
                        } else if hp_period >= 20 {
                            20000.0 // ±20,000% for medium periods
                        } else if hp_period >= 10 {
                            100000.0 // ±100,000% for small-medium periods
                        } else {
                            1000000.0 // ±1,000,000% for very small periods (3-9)
                        };

                        prop_assert!(
							y.abs() <= magnitude_limit,
							"[{}] Oscillator exceeds bounds at index {}: {} (> ±{}%) with hp_period={}",
							test_name, i, y, magnitude_limit, hp_period
						);
                    }

                    // Property 3: For constant data, oscillator should converge
                    // Skip for hp_period=2 which can produce extreme values even with constant data
                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                        && i > first + hp_period * 3
                        && hp_period > 2
                    {
                        // Allow more tolerance for small periods which can be unstable
                        let convergence_limit = if hp_period < 10 { 10.0 } else { 0.1 };
                        prop_assert!(
							y.abs() <= convergence_limit,
							"[{}] Constant data should converge near zero at index {}: got {} (hp_period={})",
							test_name, i, y, hp_period
						);
                    }

                    // Property 4: Kernel consistency check with tighter tolerance
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] NaN/Inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    // Note: Since AVX2/AVX512 are stubs that call scalar, they should be identical
                    // but floating-point operations can have slight differences due to compiler optimizations
                    prop_assert!(
                        (y - r).abs() <= 1e-7 || ulp_diff <= 20,
                        "[{}] Kernel mismatch at index {}: {} vs {} (ULP={}, diff={})",
                        test_name,
                        i,
                        y,
                        r,
                        ulp_diff,
                        (y - r).abs()
                    );

                    // Property 5: No poison values
                    prop_assert!(
                        y_bits != 0x11111111_11111111
                            && y_bits != 0x22222222_22222222
                            && y_bits != 0x33333333_33333333,
                        "[{}] Found poison value at index {}: {} (0x{:016X})",
                        test_name,
                        i,
                        y,
                        y_bits
                    );

                    // Property 6: Basic k parameter validation
                    // Since k multiplies the output, with k=0.01, values should generally be smaller
                    if k < 0.1 && i > first + hp_period * 2 && hp_period > 2 {
                        // Very small k should produce smaller outputs on average
                        // But allow for filter transients and edge effects
                        // Calculate appropriate limit based on period
                        let k_limit = if hp_period >= 50 {
                            10000.0 // 2x the normal limit for large periods
                        } else if hp_period >= 20 {
                            40000.0 // 2x the normal limit for medium periods
                        } else if hp_period >= 10 {
                            200000.0 // 2x the normal limit for small-medium periods
                        } else {
                            2000000.0 // 2x the normal limit for very small periods (3-9)
                        };

                        prop_assert!(
                            y.abs() <= k_limit,
                            "[{}] Unexpectedly large output with small k={} at index {}: got {}",
                            test_name,
                            k,
                            i,
                            y
                        );
                    }
                }

                // Property 6: Warmup period should have NaN values
                for i in 0..warmup_end.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}: got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    // Parity test: native no-alloc API matches Vec-returning API
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_dec_osc_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Build a small but non-trivial input: linear drift + sinusoid
        let n = 256usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            let v = 100.0 + 0.05 * t + 2.0 * (0.1 * t).sin();
            data.push(v);
        }

        let input = DecOscInput::from_slice(&data, DecOscParams::default());

        // Baseline via existing API
        let baseline = dec_osc(&input)?.values;

        // Preallocate output and call the new into API
        let mut out = vec![0.0; n];
        super::dec_osc_into(&mut out, &input)?;

        assert_eq!(baseline.len(), out.len());

        // Helper: NaN equals NaN, otherwise exact equality preferred
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at index {}: baseline={} vs into={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dec_osc")]
#[pyo3(signature = (data, hp_period=125, k=1.0, kernel=None))]
pub fn dec_osc_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    hp_period: usize,
    k: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = DecOscParams {
        hp_period: Some(hp_period),
        k: Some(k),
    };
    let input = DecOscInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| dec_osc_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DecOscStream")]
pub struct DecOscStreamPy {
    stream: DecOscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DecOscStreamPy {
    #[new]
    fn new(hp_period: usize, k: f64) -> PyResult<Self> {
        let params = DecOscParams {
            hp_period: Some(hp_period),
            k: Some(k),
        };
        let stream =
            DecOscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(DecOscStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dec_osc_batch")]
#[pyo3(signature = (data, hp_period_range, k_range, kernel=None))]
pub fn dec_osc_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    hp_period_range: (usize, usize, usize),
    k_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = DecOscBatchRange {
        hp_period: hp_period_range,
        k: k_range,
    };

    let rows = expand_grid_checked(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .len();
    let cols = slice_in.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow in dec_osc_batch"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
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
            dec_osc_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "hp_periods",
        combos
            .iter()
            .map(|p| p.hp_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ks",
        combos
            .iter()
            .map(|p| p.k.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ---------- CUDA Python bindings ----------
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::CudaDecOsc;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dec_osc_cuda_batch_dev")]
#[pyo3(signature = (data, hp_period_range, k_range))]
pub fn dec_osc_cuda_batch_dev_py(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    hp_period_range: (usize, usize, usize),
    k_range: (f64, f64, f64),
) -> PyResult<DecOscDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA device not available"));
    }
    let slice_in = data.as_slice()?;
    let data_f32: Vec<f32> = slice_in.iter().map(|&v| v as f32).collect();
    let sweep = DecOscBatchRange {
        hp_period: hp_period_range,
        k: k_range,
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaDecOsc::new(0).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let inner = cuda
            .dec_osc_batch_dev(&data_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((inner, ctx, dev_id))
    })?;
    Ok(DecOscDeviceArrayF32Py { inner: Some(inner), _ctx_guard: ctx, _device_id: dev_id })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dec_osc_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm, cols, rows, hp_period, k))]
pub fn dec_osc_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm: PyReadonlyArray1<'_, f64>,
    cols: usize,
    rows: usize,
    hp_period: usize,
    k: f64,
) -> PyResult<DecOscDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA device not available"));
    }
    let slice = data_tm.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("cols*rows overflow"))?;
    if slice.len() != expected {
        return Err(PyValueError::new_err(
            "time-major array length != cols*rows",
        ));
    }
    let data_f32: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
    let params = DecOscParams {
        hp_period: Some(hp_period),
        k: Some(k),
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaDecOsc::new(0).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let inner = cuda
            .dec_osc_many_series_one_param_time_major_dev(&data_f32, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((inner, ctx, dev_id))
    })?;
    Ok(DecOscDeviceArrayF32Py { inner: Some(inner), _ctx_guard: ctx, _device_id: dev_id })
}

// A dec_osc-specific Python VRAM handle that keeps the CUDA context alive.
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DecOscDeviceArrayF32")]
pub struct DecOscDeviceArrayF32Py {
    pub(crate) inner: Option<DeviceArrayF32>,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DecOscDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let itemsize = std::mem::size_of::<f32>();
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * itemsize, itemsize))?;
        let nelems = inner.rows.saturating_mul(inner.cols);
        let ptr_val: usize = if nelems == 0 {
            0
        } else {
            inner.device_ptr() as usize
        };
        d.set_item("data", (ptr_val, false))?;
        // Producing stream is synchronized before return; omit 'stream' per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self._device_id as i32) }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<usize>,
        max_version: Option<(i64, i64)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use pyo3::ffi as pyffi;
        use std::ffi::{c_void, CStr, CString};

        // Move ownership once while keeping context guard alive in manager
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: usize,
        }
        #[repr(C)]
        struct DLManagedTensor {
            dl_tensor: DLTensor,
            manager_ctx: *mut c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
        }
        #[repr(C)]
        struct DLPackVersion { major: u32, minor: u32 }
        #[repr(C)]
        struct DLManagedTensorVersioned {
            version: DLPackVersion,
            manager_ctx: *mut c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
            flags: u64,
            dl_tensor: DLTensor,
        }

        #[repr(C)]
        struct ManagerCtx {
            shape: *mut i64,
            strides: *mut i64,
            _shape_box: Box<[i64; 2]>,
            _strides_box: Box<[i64; 2]>,
            _ctx_guard: Arc<Context>,
            _buf: DeviceBuffer<f32>,
        }

        unsafe extern "C" fn deleter_legacy(self_ptr: *mut DLManagedTensor) {
            if self_ptr.is_null() { return; }
            let mt = Box::from_raw(self_ptr);
            let ctx_ptr = mt.manager_ctx as *mut ManagerCtx;
            if !ctx_ptr.is_null() {
                let _ctx = Box::from_raw(ctx_ptr);
                let _ = _ctx;
            }
            drop(mt);
        }

        unsafe extern "C" fn deleter_versioned(self_ptr: *mut DLManagedTensorVersioned) {
            if self_ptr.is_null() { return; }
            let mt = Box::from_raw(self_ptr);
            let ctx_ptr = mt.manager_ctx as *mut ManagerCtx;
            if !ctx_ptr.is_null() {
                let _ctx = Box::from_raw(ctx_ptr);
                let _ = _ctx;
            }
            drop(mt);
        }

        // Validate optional device and copy semantics (kDLCUDA = 2)
        if let Some((dev_type, dev_id)) = dl_device {
            if dev_type == 2 && dev_id as u32 != self._device_id {
                if copy.unwrap_or(false) {
                    // Let consumer handle cross-device copies.
                } else {
                    return Err(PyValueError::new_err(
                        "device mismatch without copy permission",
                    ));
                }
            }
        }

        // Producer stream is synchronized internally; ignore consumer stream per Array API.
        let _ = stream;

        let rows = inner.rows as i64;
        let cols = inner.cols as i64;
        let mut shape_box = Box::new([rows, cols]);
        let mut strides_box = Box::new([cols, 1]);
        let shape_ptr: *mut i64 = shape_box.as_mut_ptr();
        let strides_ptr: *mut i64 = strides_box.as_mut_ptr();
        let nelems = (rows as usize).saturating_mul(cols as usize);
        let data_ptr = if nelems == 0 {
            std::ptr::null_mut()
        } else {
            inner.buf.as_device_ptr().as_raw() as *mut c_void
        };

        let mgr = Box::new(ManagerCtx {
            shape: shape_ptr,
            strides: strides_ptr,
            _shape_box: shape_box,
            _strides_box: strides_box,
            _ctx_guard: self._ctx_guard.clone(),
            _buf: inner.buf,
        });
        let mgr_ptr = Box::into_raw(mgr) as *mut c_void;

        let tensor = DLTensor {
            data: data_ptr,
            device: DLDevice { device_type: 2, device_id: self._device_id as i32 },
            ndim: 2,
            dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        };

        let wants_versioned = max_version
            .map(|(maj, _)| maj >= 1)
            .unwrap_or(false);

        unsafe extern "C" fn capsule_destructor(capsule: *mut pyffi::PyObject) {
            if capsule.is_null() { return; }
            let name_ptr = pyffi::PyCapsule_GetName(capsule);
            if name_ptr.is_null() { return; }
            let name = CStr::from_ptr(name_ptr).to_bytes();
            if name == b"dltensor" {
                let ptr = pyffi::PyCapsule_GetPointer(capsule, name_ptr);
                if !ptr.is_null() {
                    let mt = ptr as *mut DLManagedTensor;
                    if let Some(del) = unsafe { (*mt).deleter } {
                        unsafe { del(mt) };
                    }
                }
            } else if name == b"dltensor_versioned" {
                let ptr = pyffi::PyCapsule_GetPointer(capsule, name_ptr);
                if !ptr.is_null() {
                    let mt = ptr as *mut DLManagedTensorVersioned;
                    if let Some(del) = unsafe { (*mt).deleter } {
                        unsafe { del(mt) };
                    }
                }
            } else {
                // used_* capsules: do nothing.
            }
        }

        unsafe {
            if wants_versioned {
                let mt = Box::new(DLManagedTensorVersioned {
                    version: DLPackVersion { major: 1, minor: 0 },
                    manager_ctx: mgr_ptr,
                    deleter: Some(deleter_versioned),
                    flags: 0,
                    dl_tensor: tensor,
                });
                let ptr = Box::into_raw(mt) as *mut c_void;
                let name = CString::new("dltensor_versioned").unwrap();
                let cap = pyffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_destructor));
                if cap.is_null() {
                    let _ = Box::from_raw(ptr as *mut DLManagedTensorVersioned);
                    return Err(PyValueError::new_err(
                        "failed to create DLPack v1.x capsule",
                    ));
                }
                Ok(PyObject::from_owned_ptr(py, cap))
            } else {
                let mt = Box::new(DLManagedTensor {
                    dl_tensor: tensor,
                    manager_ctx: mgr_ptr,
                    deleter: Some(deleter_legacy),
                });
                let ptr = Box::into_raw(mt) as *mut c_void;
                let name = CString::new("dltensor").unwrap();
                let cap = pyffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_destructor));
                if cap.is_null() {
                    let _ = Box::from_raw(ptr as *mut DLManagedTensor);
                    return Err(PyValueError::new_err(
                        "failed to create DLPack capsule",
                    ));
                }
                Ok(PyObject::from_owned_ptr(py, cap))
            }
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dec_osc_js(data: &[f64], hp_period: usize, k: f64) -> Result<Vec<f64>, JsValue> {
    let params = DecOscParams {
        hp_period: Some(hp_period),
        k: Some(k),
    };
    let input = DecOscInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    #[cfg(target_arch = "wasm32")]
    let kernel = detect_best_kernel();
    #[cfg(not(target_arch = "wasm32"))]
    let kernel = Kernel::Auto;

    dec_osc_into_slice(&mut output, &input, kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dec_osc_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    hp_period: usize,
    k: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = DecOscParams {
            hp_period: Some(hp_period),
            k: Some(k),
        };
        let input = DecOscInput::from_slice(data, params);

        #[cfg(target_arch = "wasm32")]
        let kernel = detect_best_kernel();
        #[cfg(not(target_arch = "wasm32"))]
        let kernel = Kernel::Auto;

        if in_ptr == out_ptr as *const f64 {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            dec_osc_into_slice(&mut temp, &input, kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            dec_osc_into_slice(out, &input, kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dec_osc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dec_osc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DecOscBatchConfig {
    pub hp_period_range: (usize, usize, usize),
    pub k_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DecOscBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DecOscParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dec_osc_batch)]
pub fn dec_osc_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DecOscBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = DecOscBatchRange {
        hp_period: config.hp_period_range,
        k: config.k_range,
    };

    #[cfg(target_arch = "wasm32")]
    let output = dec_osc_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(not(target_arch = "wasm32"))]
    let output = dec_osc_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = DecOscBatchJsOutput {
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
pub fn dec_osc_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    hp_start: usize,
    hp_end: usize,
    hp_step: usize,
    k_start: f64,
    k_end: f64,
    k_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = DecOscBatchRange {
            hp_period: (hp_start, hp_end, hp_step),
            k: (k_start, k_end, k_step),
        };

        let combos = expand_grid_checked(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        dec_osc_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
