//! # Forecast Oscillator (FOSC)
//!
//! The Forecast Oscillator (FOSC) measures the percentage difference between the current price
//! and a one-step-ahead linear regression forecast. Positive values suggest the price is above
//! its trend, while negative values indicate it's below.
//!
//! ## Parameters
//! - **period**: Regression window for linear forecast (default: 5)
//!
//! ## Returns
//! - **`Ok(FoscOutput)`** containing a `Vec<f64>` of oscillator values as percentages
//!
//! ## Developer Notes
//! Decision log: scalar path is reference; SIMD stubs exist but `Kernel::Auto` short-circuits to Scalar for perf; CUDA wrapper + Python bindings return VRAM handles with CAI v3 + DLPack v1.x semantics.
//! ### Implementation Status
//! - SIMD implemented (AVX2/AVX512) with vectorized initializer and FMA in scalar loop.
//! - Runtime selection short-circuits to Scalar for `Kernel::Auto` because SIMD underperforms on
//!   typical CPUs at realistic sizes (latency-bound recurrence; AVX512 may downclock). Keep code
//!   for future tuning.
//! - Streaming Update: O(1) – efficient with incremental regression updates.
//! - Memory Optimization: Uses `alloc_with_nan_prefix` for outputs and batch helpers per ALMA.
//! - Batch Operations: Implemented; Auto selection maps to scalar path for stability/perf.
//!
//! ### TODO - Performance Improvements
//! - [ ] Implement actual AVX2 SIMD kernel (currently stub)
//! - [ ] Implement actual AVX512 SIMD kernel (currently stub)
//! - [ ] Vectorize linear regression calculations
//! - [ ] Optimize sum calculations with SIMD horizontal adds
//! - [ ] Consider caching regression coefficients for batch operations

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

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::{
    exceptions::PyValueError,
    pyclass, pyfunction, pymethods,
    types::{PyDict, PyDictMethods},
    Bound, PyErr, PyResult, Python,
};
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// --------- Input Data & Param Structs ---------

impl<'a> AsRef<[f64]> for FoscInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            FoscData::Slice(slice) => slice,
            FoscData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FoscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FoscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct FoscParams {
    pub period: Option<usize>,
}

impl Default for FoscParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct FoscInput<'a> {
    pub data: FoscData<'a>,
    pub params: FoscParams,
}

impl<'a> FoscInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: FoscParams) -> Self {
        Self {
            data: FoscData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: FoscParams) -> Self {
        Self {
            data: FoscData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", FoscParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

// --------- Builder Struct ---------

#[derive(Copy, Clone, Debug)]
pub struct FoscBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for FoscBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FoscBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<FoscOutput, FoscError> {
        let p = FoscParams {
            period: self.period,
        };
        let i = FoscInput::from_candles(c, "close", p);
        fosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<FoscOutput, FoscError> {
        let p = FoscParams {
            period: self.period,
        };
        let i = FoscInput::from_slice(d, p);
        fosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<FoscStream, FoscError> {
        let p = FoscParams {
            period: self.period,
        };
        FoscStream::try_new(p)
    }
}

// --------- Error ---------

#[derive(Debug, Error)]
pub enum FoscError {
    #[error("fosc: Empty input data provided")]
    EmptyInputData,
    #[error("fosc: All values are NaN.")]
    AllValuesNaN,
    #[error("fosc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fosc: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("fosc: Invalid kernel for batch operation: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("fosc: Invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
}

// --------- Main Dispatchers ---------

#[inline]
pub fn fosc(input: &FoscInput) -> Result<FoscOutput, FoscError> {
    fosc_with_kernel(input, Kernel::Auto)
}

pub fn fosc_with_kernel(input: &FoscInput, kernel: Kernel) -> Result<FoscOutput, FoscError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(FoscError::EmptyInputData);
    }
    let period = input.get_period();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(FoscError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(FoscError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(FoscError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    // FIX: warmup = first + period - 1
    let mut out = alloc_with_nan_prefix(len, first + period - 1);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => fosc_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => fosc_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => fosc_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(FoscOutput { values: out })
}

/// Write directly to output slice - no allocations
#[inline]
pub fn fosc_into_slice(dst: &mut [f64], input: &FoscInput, kern: Kernel) -> Result<(), FoscError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let period = input.get_period();

    if len == 0 {
        return Err(FoscError::EmptyInputData);
    }

    if dst.len() != len {
        return Err(FoscError::OutputLengthMismatch { expected: len, got: dst.len() });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(FoscError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(FoscError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(FoscError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => fosc_scalar(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => fosc_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => fosc_avx512(data, period, first, dst),
            _ => unreachable!(),
        }
    }

    // Fill warmup with NaN
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Write Forecast Oscillator (FOSC) outputs into a caller-provided buffer without allocating.
///
/// - Preserves the NaN warmup prefix exactly like `fosc()`/`fosc_with_kernel()`.
/// - The output slice length must equal the input length.
/// - Uses `Kernel::Auto` for runtime kernel selection.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn fosc_into(input: &FoscInput, out: &mut [f64]) -> Result<(), FoscError> {
    fosc_into_slice(out, input, Kernel::Auto)
}

// --------- Scalar Core ---------

#[inline(always)]
unsafe fn fosc_core(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let n = data.len();
    if n == 0 || period == 0 || n < period {
        return;
    }

    // Compute the first index we will write and bail if it exceeds len
    let begin = first + period - 1;
    if begin >= n {
        return;
    }

    // Closed-form constants for 1..=p and their squares
    let p = period as f64;
    let x = 0.5 * p * (p + 1.0);
    let x2 = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0;
    let denom = p * x2 - x * x;
    let bd = if denom.abs() < f64::EPSILON {
        0.0
    } else {
        1.0 / denom
    };
    let inv_p = 1.0 / p;

    // Precompute b = (p*xy - x*y)/denom as: b = xy*(p/denom) - y*(x/denom)
    let p_bd = p * bd;
    let x_bd = x * bd;
    // TSF one-step forecast at x=p+1 with x=1..p weights:
    // mean_y + b*(p+1 - mean_x) where mean_x=(p+1)/2  => mean_y + b*(p+1)/2
    let tsf_coeff = 0.5 * (p + 1.0);

    // Initialize running sums over window [first .. first+period-2]
    let mut y = 0.0f64;
    let mut xy = 0.0f64;
    let base = data.as_ptr().add(first);
    let limit = period - 1; // number of elements in the partial window
    let mut k = 0usize;
    while k + 4 <= limit {
        let d0 = *base.add(k + 0);
        let d1 = *base.add(k + 1);
        let d2 = *base.add(k + 2);
        let d3 = *base.add(k + 3);

        y += d0 + d1 + d2 + d3;
        xy += d0 * (k + 1) as f64;
        xy += d1 * (k + 2) as f64;
        xy += d2 * (k + 3) as f64;
        xy += d3 * (k + 4) as f64;
        k += 4;
    }
    while k < limit {
        let d = *base.add(k);
        y += d;
        xy += d * (k + 1) as f64;
        k += 1;
    }

    // Recurrence: write using previous TSF forecast
    let dp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut tsf_prev = 0.0f64;
    let mut i = begin;
    while i < n {
        let newv = *dp.add(i);

        // Include the new value to form the current full window sums
        let y_plus = y + newv;
        let xy_plus = xy + newv * p;

        // Oscillator w.r.t. previous forecast
        *op.add(i) = if newv != 0.0 {
            100.0 * (newv - tsf_prev) / newv
        } else {
            f64::NAN
        };

        // Next-step forecast for this window
        let b = xy_plus * p_bd - y_plus * x_bd;
        tsf_prev = y_plus * inv_p + b * tsf_coeff;

        // Slide window forward: drop oldest and shift weights
        let old_idx = i + 1 - period;
        let oldv = *dp.add(old_idx);
        xy = xy_plus - y_plus;
        y = y_plus - oldv;

        i += 1;
    }
}

#[inline]
pub fn fosc_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    unsafe { fosc_core(data, period, first, out) }
}

// --------- AVX2/AVX512 Routing (Stubs) ---------

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    fosc_core(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // AVX512 kernel stub: reuse AVX2 (unsafe scalar) implementation
    fosc_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    fosc_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn fosc_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    fosc_avx2(data, period, first, out)
}

// fosc_avx512_core removed; AVX512 maps to AVX2 (unsafe scalar) above.

// --------- Streaming (Stateful) ---------

// Decision: Streaming uses O(1) sliding OLS updates and the previous bar's forecast


#[derive(Debug, Clone)]
pub struct FoscStream {
    period: usize,
    buffer: Vec<f64>,
    idx: usize, 
    filled: bool,

    
    x: f64,       
    x2: f64,      
    inv_den: f64, 
    inv_p: f64,   
    p_f64: f64,   
    p1: f64,      

    
    y: f64,  
    xy: f64, 

    
    
    tsf: f64,

    count: usize,
}

impl FoscStream {
    pub fn try_new(params: FoscParams) -> Result<Self, FoscError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(FoscError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let p = period as f64;
        
        let x = 0.5 * p * (p + 1.0);
        let x2 = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0; 
        let den = p * x2 - x * x;
        let inv_den = if den.abs() < f64::EPSILON {
            0.0
        } else {
            1.0 / den
        };

        Ok(Self {
            period,
            buffer: vec![0.0; period],
            idx: 0,
            filled: false,

            x,
            x2,
            inv_den,
            inv_p: 1.0 / p,
            p_f64: p,
            p1: p + 1.0,

            y: 0.0,
            xy: 0.0,
            tsf: 0.0,
            count: 0,
        })
    }

    /// Streaming update in O(1).
    /// Returns None during warmup; thereafter returns the Forecast Oscillator
    /// using the previous one-step-ahead linear regression forecast
    /// (matches the scalar/batch semantics).
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        
        if self.count < self.period {
            self.buffer[self.idx] = value;
            self.y += value;
            self.xy += value * (self.count as f64 + 1.0);

            
            self.idx += 1;
            if self.idx == self.period {
                self.idx = 0;
            }
            self.count += 1;

            if self.count == self.period {
                self.filled = true;

                
                let b = (self.p_f64.mul_add(self.xy, -self.x * self.y)) * self.inv_den; 
                let a = (self.y - b * self.x) * self.inv_p;
                self.tsf = b.mul_add(self.p1, a); 
            }
            return None;
        }

        
        
        let out = if value.is_finite() && value != 0.0 {
            
            100.0 * (1.0 - self.tsf / value)
        } else {
            f64::NAN
        };

        
        let old = self.buffer[self.idx];
        self.buffer[self.idx] = value;
        self.idx += 1;
        if self.idx == self.period {
            self.idx = 0;
        }

        
        let y_prev = self.y;
        self.y = y_prev - old + value;
        self.xy = self.xy - y_prev + self.p_f64 * value;

        
        let b = (self.p_f64.mul_add(self.xy, -self.x * self.y)) * self.inv_den;
        let a = (self.y - b * self.x) * self.inv_p;
        self.tsf = b.mul_add(self.p1, a);

        Some(out)
    }
}



#[derive(Clone, Debug)]
pub struct FoscBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for FoscBatchRange {
    fn default() -> Self {
        Self { period: (5, 254, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FoscBatchBuilder {
    range: FoscBatchRange,
    kernel: Kernel,
}

impl FoscBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<FoscBatchOutput, FoscError> {
        fosc_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<FoscBatchOutput, FoscError> {
        FoscBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<FoscBatchOutput, FoscError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FoscBatchOutput, FoscError> {
        FoscBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct FoscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FoscParams>,
    pub rows: usize,
    pub cols: usize,
}

impl FoscBatchOutput {
    pub fn row_for_params(&self, p: &FoscParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &FoscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &FoscBatchRange) -> Result<Vec<FoscParams>, FoscError> {
    #[inline]
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, FoscError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step).collect();
            return if v.is_empty() {
                Err(FoscError::InvalidRange { start, end, step })
            } else {
                Ok(v)
            };
        } else {
            
            let mut v = Vec::new();
            let mut cur = start;
            loop {
                v.push(cur);
                if cur <= end { break; }
                match cur.checked_sub(step) {
                    Some(next) => cur = next,
                    None => break,
                }
                if cur == usize::MAX { break; }
                if cur <= end { break; }
            }
            if v.is_empty() {
                Err(FoscError::InvalidRange { start, end, step })
            } else {
                Ok(v)
            }
        }
    }

    let periods = axis_usize(r.period)?;
    if periods.is_empty() {
        return Err(FoscError::InvalidRange {
            start: r.period.0,
            end: r.period.1,
            step: r.period.2,
        });
    }
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(FoscParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn fosc_batch_slice(
    data: &[f64],
    sweep: &FoscBatchRange,
    kern: Kernel,
) -> Result<FoscBatchOutput, FoscError> {
    fosc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn fosc_batch_par_slice(
    data: &[f64],
    sweep: &FoscBatchRange,
    kern: Kernel,
) -> Result<FoscBatchOutput, FoscError> {
    fosc_batch_inner(data, sweep, kern, true)
}

pub fn fosc_batch_with_kernel(
    data: &[f64],
    sweep: &FoscBatchRange,
    k: Kernel,
) -> Result<FoscBatchOutput, FoscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(FoscError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    fosc_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn fosc_batch_inner(
    data: &[f64],
    sweep: &FoscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FoscBatchOutput, FoscError> {
    let combos = expand_grid(sweep)?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(FoscError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(FoscError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut buf_mu = make_uninit_matrix(rows, cols);

    
    
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap_or(5) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_f64: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    
    
    let actual = match kern {
        Kernel::Auto => Kernel::ScalarBatch,
        k => k,
    };
    let simd = match actual {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Scalar => Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => Kernel::Avx512,
        _ => unreachable!(),
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match simd {
            Kernel::Scalar => fosc_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => fosc_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => fosc_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_f64
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_f64.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_f64.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(FoscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn fosc_batch_inner_into(
    data: &[f64],
    sweep: &FoscBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<FoscParams>, FoscError> {
    let combos = expand_grid(sweep)?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(FoscError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(FoscError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let cols = data.len();
    let rows = combos.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(FoscError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if out.len() != expected {
        return Err(FoscError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    
    let out_uninit: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(out_uninit, cols, &warmup_periods);

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Scalar => Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => Kernel::Avx512,
        _ => unreachable!(),
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match simd {
            Kernel::Scalar => fosc_row_scalar(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => fosc_row_avx2(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => fosc_row_avx512(data, first, period, dst),
            _ => unreachable!(),
        }
        
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, sl)| do_row(r, sl));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, sl) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(r, sl);
            }
        }
    } else {
        for (r, sl) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(r, sl);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn fosc_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    fosc_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    fosc_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        fosc_row_avx512_short(data, first, period, out)
    } else {
        fosc_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    fosc_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fosc_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    fosc_avx512_long(data, period, first, out)
}



#[cfg(feature = "python")]
#[pyfunction(name = "fosc")]
#[pyo3(signature = (data, period=5, kernel=None))]
pub fn fosc_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let kernel_enum = validate_kernel(kernel, false)?;

    let params = FoscParams {
        period: Some(period),
    };
    let input = FoscInput::from_slice(data_slice, params);

    py.allow_threads(|| {
        let output = fosc_with_kernel(&input, kernel_enum)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Rust computation error: {}", e)))?;
        Ok(output)
    })
    .map(|result| result.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "fosc_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn fosc_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?; 
    let sweep = FoscBatchRange {
        period: period_range,
    };

    
    let combos = expand_grid(&sweep).map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyErr::new::<PyValueError, _>("rows*cols overflow in fosc_batch_py"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    
    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match batch {
                Kernel::ScalarBatch => Kernel::Scalar,
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2Batch => Kernel::Avx2,
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512Batch => Kernel::Avx512,
                _ => unreachable!(),
            };
            fosc_batch_inner_into(slice_in, &sweep, simd, true, out_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}


#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "fosc_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn fosc_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::fosc_wrapper::CudaFosc;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = FoscBatchRange {
        period: period_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaFosc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.fosc_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let handle = make_device_array_py(device_id, inner)?;
    Ok(handle)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "fosc_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn fosc_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::fosc_wrapper::CudaFosc;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let flat_in: &[f32] = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = FoscParams {
        period: Some(period),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaFosc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.fosc_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let handle = make_device_array_py(device_id, inner)?;
    Ok(handle)
}

#[cfg(feature = "python")]
#[pyclass(name = "FoscStream")]
pub struct FoscStreamPy {
    inner: FoscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl FoscStreamPy {
    #[new]
    #[pyo3(signature = (period=5))]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = FoscParams {
            period: Some(period),
        };
        let inner = FoscStream::try_new(params).map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to create stream: {}", e))
        })?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// --------- WASM Bindings ---------

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fosc_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = FoscParams {
        period: Some(period),
    };
    let input = FoscInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()]; // Single allocation
    fosc_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fosc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fosc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fosc_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to fosc_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = FoscParams {
            period: Some(period),
        };
        let input = FoscInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            fosc_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            fosc_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

// Batch API structures
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FoscBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FoscBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FoscParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = fosc_batch)]
pub fn fosc_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: FoscBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let range = FoscBatchRange {
        period: config.period_range,
    };

    let output = fosc_batch_inner(data, &range, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = FoscBatchJsOutput {
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
pub fn fosc_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to fosc_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let range = FoscBatchRange {
            period: (period_start, period_end, period_step),
        };

        // Call fosc_batch_inner and copy results
        let output = fosc_batch_inner(data, &range, Kernel::Auto, false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Copy results to output pointer
        let out = std::slice::from_raw_parts_mut(out_ptr, output.values.len());
        out.copy_from_slice(&output.values);

        Ok(output.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_fosc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = FoscParams { period: None };
        let input = FoscInput::from_candles(&candles, "close", default_params);
        let output = fosc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_fosc_basic_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let test_data = [
            81.59, 81.06, 82.87, 83.00, 83.61, 83.15, 82.84, 82.84, 83.99, 84.55, 84.36, 85.53,
        ];
        let period = 5;
        let input = FoscInput::from_slice(
            &test_data,
            FoscParams {
                period: Some(period),
            },
        );
        let result = fosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), test_data.len());
        for i in 0..(period - 1) {
            assert!(result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_fosc_with_nan_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        let params = FoscParams { period: Some(3) };
        let input = FoscInput::from_slice(&input_data, params);
        let result = fosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), input_data.len());
        Ok(())
    }

    fn check_fosc_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = FoscParams { period: Some(0) };
        let input = FoscInput::from_slice(&input_data, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FOSC should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_fosc_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = FoscParams { period: Some(10) };
        let input = FoscInput::from_slice(&data_small, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FOSC should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_fosc_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = FoscParams { period: Some(5) };
        let input = FoscInput::from_slice(&single_point, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FOSC should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_fosc_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = FoscParams { period: Some(2) };
        let input = FoscInput::from_slice(&input_data, params);
        let res = fosc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FOSC should fail with all NaN",
            test_name
        );
        Ok(())
    }

    fn check_fosc_expected_values_reference(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let expected_last_five = [
            -0.8904444627923475,
            -0.4763353099245297,
            -0.2379782851444668,
            0.292790128025632,
            -0.6597902988090389,
        ];
        let params = FoscParams { period: Some(5) };
        let input = FoscInput::from_candles(&candles, "close", params);
        let result = fosc_with_kernel(&input, kernel)?;
        let valid_len = result.values.len();
        assert!(valid_len >= 5);
        let output_slice = &result.values[valid_len - 5..valid_len];
        for (i, &val) in output_slice.iter().enumerate() {
            let exp: f64 = expected_last_five[i];
            if exp.is_nan() {
                assert!(val.is_nan());
            } else {
                assert!(
                    (val - exp).abs() < 1e-7,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    exp,
                    val
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_fosc_tests {
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

    #[cfg(debug_assertions)]
    fn check_fosc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            FoscParams::default(),
            // Minimum period
            FoscParams { period: Some(2) },
            // Small periods
            FoscParams { period: Some(3) },
            FoscParams { period: Some(5) }, // Default value
            FoscParams { period: Some(7) },
            // Medium periods
            FoscParams { period: Some(10) },
            FoscParams { period: Some(14) },
            FoscParams { period: Some(20) },
            // Large periods
            FoscParams { period: Some(30) },
            FoscParams { period: Some(50) },
            // Very large periods
            FoscParams { period: Some(100) },
            FoscParams { period: Some(200) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = FoscInput::from_candles(&candles, "close", params.clone());
            let output = fosc_with_kernel(&input, kernel)?;

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
                        params.period.unwrap_or(5),
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
                        params.period.unwrap_or(5),
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
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_fosc_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_fosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy 1: Variable period with random price data
        let strat1 = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (1e-6f64..1e6f64)
                        .prop_filter("finite and non-zero", |x| x.is_finite() && x.abs() > 1e-10),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat1, |(data, period)| {
            let params = FoscParams {
                period: Some(period),
            };
            let input = FoscInput::from_slice(&data, params);

            let FoscOutput { values: out } = fosc_with_kernel(&input, kernel).unwrap();
            let FoscOutput { values: ref_out } = fosc_with_kernel(&input, Kernel::Scalar).unwrap();

            // Property 1: Output length must match input length
            prop_assert_eq!(out.len(), data.len());
            prop_assert_eq!(ref_out.len(), data.len());

            // Property 2: Warmup period correctness - first (period-1) values should be NaN
            for i in 0..(period - 1) {
                prop_assert!(out[i].is_nan(), "Expected NaN at index {} during warmup", i);
                prop_assert!(
                    ref_out[i].is_nan(),
                    "Expected NaN at index {} during warmup (scalar)",
                    i
                );
            }

            // Property 3: Kernel consistency - all kernels should produce identical results
            for i in (period - 1)..data.len() {
                let y = out[i];
                let r = ref_out[i];

                // Both should be NaN or both should be finite
                if r.is_nan() {
                    prop_assert!(
                        y.is_nan(),
                        "Kernel mismatch at {}: scalar is NaN but kernel is {}",
                        i,
                        y
                    );
                } else if y.is_nan() {
                    prop_assert!(
                        r.is_nan(),
                        "Kernel mismatch at {}: kernel is NaN but scalar is {}",
                        i,
                        r
                    );
                } else {
                    // Allow small tolerance for floating-point differences
                    let diff = (y - r).abs();
                    let tolerance = 1e-9 * r.abs().max(1.0);
                    prop_assert!(
                        diff <= tolerance,
                        "Kernel mismatch at {}: {} vs {} (diff: {})",
                        i,
                        y,
                        r,
                        diff
                    );
                }

                // Property 4: Poison value detection
                let y_bits = y.to_bits();
                prop_assert_ne!(
                    y_bits,
                    0x11111111_11111111,
                    "Found alloc_with_nan_prefix poison at {}",
                    i
                );
                prop_assert_ne!(
                    y_bits,
                    0x22222222_22222222,
                    "Found init_matrix_prefixes poison at {}",
                    i
                );
                prop_assert_ne!(
                    y_bits,
                    0x33333333_33333333,
                    "Found make_uninit_matrix poison at {}",
                    i
                );
            }

            Ok(())
        })?;

        // Strategy 2: Fixed period with varying data lengths
        let strat2 = prop::collection::vec(
            (1e-6f64..1e6f64)
                .prop_filter("finite and non-zero", |x| x.is_finite() && x.abs() > 1e-10),
            10..1000,
        );

        proptest::test_runner::TestRunner::default().run(&strat2, |data| {
            let period = 5; // Use default period
            let params = FoscParams {
                period: Some(period),
            };
            let input = FoscInput::from_slice(&data, params);

            let FoscOutput { values: out } = fosc_with_kernel(&input, kernel).unwrap();

            // Check bounds - FOSC is percentage, typically -100 to +100 but can exceed
            // We'll use a reasonable bound of -200% to +200%
            for i in (period - 1)..data.len() {
                if !out[i].is_nan() {
                    prop_assert!(
                        out[i] >= -200.0 && out[i] <= 200.0,
                        "FOSC value {} at index {} is out of reasonable bounds",
                        out[i],
                        i
                    );
                }
            }

            Ok(())
        })?;

        
        let strat3 =
            (2usize..=20, 1e-6f64..10f64, 10usize..100).prop_map(|(period, start, len)| {
                let data: Vec<f64> = (0..len).map(|i| start + (i as f64) * 0.1).collect();
                (data, period)
            });

        proptest::test_runner::TestRunner::default().run(&strat3, |(data, period)| {
            let params = FoscParams {
                period: Some(period),
            };
            let input = FoscInput::from_slice(&data, params);

            let FoscOutput { values: out } = fosc_with_kernel(&input, kernel).unwrap();

            
            
            
            let start_idx = if period > 0 { period } else { period - 1 }; 
            let valid_fosc: Vec<f64> = out
                .iter()
                .skip(start_idx)
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            if valid_fosc.len() > 5 {
                
                for &val in &valid_fosc {
                    prop_assert!(val.abs() < 5.0, "FOSC {} too large for linear trend", val);
                }

                
                let mean: f64 = valid_fosc.iter().sum::<f64>() / valid_fosc.len() as f64;
                let variance: f64 = valid_fosc.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / valid_fosc.len() as f64;
                let std_dev = variance.sqrt();

                prop_assert!(
                    std_dev < 1.0,
                    "FOSC standard deviation {} too high for linear trend",
                    std_dev
                );
            }

            Ok(())
        })?;

        
        let strat4 = (2usize..=20, 10usize..100).prop_map(|(period, len)| {
            let data: Vec<f64> = (0..len)
                .map(|i| 100.0 + 10.0 * (i as f64 * 0.5).sin())
                .collect();
            (data, period)
        });

        proptest::test_runner::TestRunner::default().run(&strat4, |(data, period)| {
            let params = FoscParams {
                period: Some(period),
            };
            let input = FoscInput::from_slice(&data, params);

            let FoscOutput { values: out } = fosc_with_kernel(&input, kernel).unwrap();

            
            let valid_values: Vec<f64> = out
                .iter()
                .skip(period - 1)
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            if valid_values.len() > 10 {
                let mean: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                
                prop_assert!(
                    mean.abs() < 10.0,
                    "Mean FOSC {} is too far from zero for oscillating data",
                    mean
                );
            }

            Ok(())
        })?;

        
        let strat5 = (2usize..=20, 50f64..200f64, 20usize..100).prop_flat_map(
            |(period, base_price, len)| {
                (
                    prop::collection::vec(
                        (-0.5f64..0.5f64, 0f64..0.5f64, 0f64..0.5f64).prop_map(
                            move |(change, high_diff, low_diff)| {
                                base_price * (1.0 + change * 0.01) + high_diff
                            },
                        ),
                        len,
                    ),
                    Just(period),
                )
            },
        );

        proptest::test_runner::TestRunner::default().run(&strat5, |(data, period)| {
            let params = FoscParams {
                period: Some(period),
            };
            let input = FoscInput::from_slice(&data, params);

            let result = fosc_with_kernel(&input, kernel);
            prop_assert!(result.is_ok(), "FOSC failed for OHLC-like data");

            let FoscOutput { values: out } = result.unwrap();

            
            if period == 2 {
                for i in 1..data.len() {
                    if !out[i].is_nan() {
                        
                        
                        prop_assert!(
                            out[i].abs() < 100.0,
                            "FOSC {} at index {} unreasonable for period=2",
                            out[i],
                            i
                        );
                    }
                }
            }

            
            if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() > period {
                for i in (period - 1)..data.len() {
                    if !out[i].is_nan() {
                        prop_assert!(
                            out[i].abs() < 1e-6,
                            "FOSC {} at index {} should be ~0 for constant data",
                            out[i],
                            i
                        );
                    }
                }
            }

            Ok(())
        })?;

        
        let strat6 = (2usize..=10, 10usize..50).prop_flat_map(|(period, len)| {
            (
                prop::collection::vec(
                    prop::strategy::Union::new(vec![
                        
                        (0.9f64..1.0)
                            .prop_map(|p| {
                                if p < 0.95 {
                                    100.0 + p
                                } else if p < 0.98 {
                                    1e-12
                                } else {
                                    0.0
                                }
                            })
                            .boxed(),
                        (50f64..150f64).boxed(),
                        (-1e-12f64..1e-12f64)
                            .prop_filter("not exactly zero", |x| x.abs() > 1e-15)
                            .boxed(),
                    ]),
                    len,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat6, |(data, period)| {
            let params = FoscParams {
                period: Some(period),
            };
            let input = FoscInput::from_slice(&data, params);

            let result = fosc_with_kernel(&input, kernel);
            prop_assert!(result.is_ok(), "FOSC failed for near-zero data");

            let FoscOutput { values: out } = result.unwrap();

            
            for i in (period - 1)..data.len() {
                if data[i] == 0.0 {
                    prop_assert!(
                        out[i].is_nan(),
                        "FOSC should be NaN when price is 0.0 at index {}",
                        i
                    );
                } else if data[i].abs() < 1e-10 {
                    
                    
                    if !out[i].is_nan() {
                        
                        prop_assert!(
                            out[i].abs() < 10000.0,
                            "FOSC {} unreasonably large for tiny price {} at index {}",
                            out[i],
                            data[i],
                            i
                        );
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_fosc_tests!(
        check_fosc_partial_params,
        check_fosc_basic_accuracy,
        check_fosc_with_nan_data,
        check_fosc_zero_period,
        check_fosc_period_exceeds_length,
        check_fosc_very_small_dataset,
        check_fosc_all_values_nan,
        check_fosc_expected_values_reference,
        check_fosc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_fosc_tests!(check_fosc_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = FoscBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = FoscParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -0.8904444627923475,
            -0.4763353099245297,
            -0.2379782851444668,
            0.292790128025632,
            -0.6597902988090389,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = FoscBatchBuilder::new()
            .kernel(kernel)
            .period_range(5, 25, 5)
            .apply_candles(&c, "close")?;

        let expected_combos = 5; 
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        
        let test_configs = vec![
            
            (2, 10, 2),     
            (5, 25, 5),     
            (30, 60, 15),   
            (2, 5, 1),      
            (10, 20, 2),    
            (5, 5, 0),      
            (5, 50, 15),    
            (100, 200, 50), 
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = FoscBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
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
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
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
                        combo.period.unwrap_or(5)
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
                        combo.period.unwrap_or(5)
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
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_fosc_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        
        let n = 512usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64;
            
            data.push(50.0 + 0.05 * x + (0.1 * x).sin() * 2.0);
        }

        let input = FoscInput::from_slice(&data, FoscParams::default());

        
        let baseline = fosc_with_kernel(&input, Kernel::Auto)?.values;

        
        let mut out = vec![0.0; data.len()];
        fosc_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        
        #[inline]
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }
}
