//! # Chande Exits (Chandelier Exits)
//!
//! A volatility-based trailing stop indicator that combines Average True Range (ATR) with rolling
//! highest high or lowest low values to create adaptive stop-loss levels. Designed to protect
//! profits by trailing price movements while allowing room for normal volatility. The indicator
//! follows price movements more closely in trending markets and provides wider stops in volatile conditions.
//!
//! ## Parameters
//! - **period**: Window size for both ATR calculation and rolling max/min (default: 22)
//! - **mult**: ATR multiplier for stop distance (default: 3.0)
//! - **direction**: Trading direction - "long" or "short" (default: "long")
//!
//! ## Inputs
//! - Requires high, low, and close price arrays
//! - Supports both raw slices and Candles data structure
//!
//! ## Returns
//! - **`Ok(ChandeOutput)`** containing a `Vec<f64>` of stop levels matching input length
//! - For long: Highest High[period] - ATR[period] * multiplier
//! - For short: Lowest Low[period] + ATR[period] * multiplier
//!
//! ## Developer Notes (Implementation Status)
//! - **Scalar path**: Single-pass O(n) using Wilder ATR (RMA) + monotonic deques for rolling
//!   max/min. Warmup uses NaN prefix identical to alma.rs patterns.
//! - **SIMD Kernels**: AVX2/AVX512 present as stubs delegating to the scalar implementation.
//!   Rationale: the hot loop is sequential (ATR recurrence) and deque-bound (rolling max/min),
//!   so end-to-end AVX wins are typically within noise; we short-circuit to scalar for
//!   performance stability and identical numerics.
//! - **Streaming Performance**: O(1) with deques in the streaming API.
//! - **Memory Optimization**: Uses `alloc_with_nan_prefix`/matrix helpers; no O(N) temps per output.
//! - **Batch**: Parallel per-row supported. Potential future optimization: precompute the TR stream
//!   once across rows; current design favors simplicity and clarity.
//! - **Decision log**: SIMD kept as scalar delegate; CUDA wrapper returns VRAM handles with CAI v3 + DLPack v1.x and syncs streams before hand-off; scalar/CPU outputs remain the numerical reference.

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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::{PyTypeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum ChandeData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ChandeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ChandeParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub direction: Option<String>,
}

impl Default for ChandeParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChandeInput<'a> {
    pub data: ChandeData<'a>,
    pub params: ChandeParams,
}

impl<'a> ChandeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: ChandeParams) -> Self {
        Self {
            data: ChandeData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], p: ChandeParams) -> Self {
        Self {
            data: ChandeData::Slices { high, low, close },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, ChandeParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(22)
    }
    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(3.0)
    }
    #[inline]
    pub fn get_direction(&self) -> &str {
        self.params.direction.as_deref().unwrap_or("long")
    }
    #[inline]
    pub fn borrow_slices(&self) -> (&[f64], &[f64], &[f64]) {
        match &self.data {
            ChandeData::Candles { candles } => (
                source_type(candles, "high"),
                source_type(candles, "low"),
                source_type(candles, "close"),
            ),
            ChandeData::Slices { high, low, close } => (high, low, close),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChandeBuilder {
    period: Option<usize>,
    mult: Option<f64>,
    direction: Option<String>,
    kernel: Kernel,
}

impl Default for ChandeBuilder {
    fn default() -> Self {
        Self {
            period: None,
            mult: None,
            direction: None,
            kernel: Kernel::Auto,
        }
    }
}
impl ChandeBuilder {
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
    pub fn mult(mut self, m: f64) -> Self {
        self.mult = Some(m);
        self
    }
    #[inline(always)]
    pub fn direction<S: Into<String>>(mut self, d: S) -> Self {
        self.direction = Some(d.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<ChandeOutput, ChandeError> {
        let p = ChandeParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
        };
        let i = ChandeInput::from_candles(c, p);
        chande_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ChandeOutput, ChandeError> {
        let p = ChandeParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
        };
        let i = ChandeInput::from_slices(high, low, close, p);
        chande_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ChandeStream, ChandeError> {
        let p = ChandeParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
        };
        ChandeStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ChandeError {
    #[error("chande: input series are empty")] 
    EmptyInputData,
    #[error("chande: all values are NaN")] 
    AllValuesNaN,
    #[error("chande: invalid period: period={period}, data_len={data_len}")] 
    InvalidPeriod { period: usize, data_len: usize },
    #[error("chande: not enough valid data: needed={needed}, valid={valid}")] 
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("chande: input length mismatch: high={h}, low={l}, close={c}")] 
    DataLengthMismatch { h: usize, l: usize, c: usize },
    #[error("chande: invalid direction: {direction}")] 
    InvalidDirection { direction: String },
    #[error("chande: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("chande: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: isize, end: isize, step: isize },
    #[error("chande: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("chande: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
fn first_valid3(h: &[f64], l: &[f64], c: &[f64]) -> Option<usize> {
    let n = h.len().min(l.len()).min(c.len());
    (0..n).find(|&i| !h[i].is_nan() && !l[i].is_nan() && !c[i].is_nan())
}

#[inline]
pub fn chande(input: &ChandeInput) -> Result<ChandeOutput, ChandeError> {
    chande_with_kernel(input, Kernel::Auto)
}

pub fn chande_with_kernel(
    input: &ChandeInput,
    kernel: Kernel,
) -> Result<ChandeOutput, ChandeError> {
    let (high, low, close) = input.borrow_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }

    let len = high.len();
    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;
    let period = input.get_period();
    let mult = input.get_mult();
    let dir = input.get_direction().to_lowercase();
    if dir != "long" && dir != "short" {
        return Err(ChandeError::InvalidDirection { direction: dir });
    }
    if period == 0 || period > len {
        return Err(ChandeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(ChandeError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Safe fallback when AVX isn't available
    let chosen = match (
        chosen,
        cfg!(all(feature = "nightly-avx", target_arch = "x86_64")),
    ) {
        (Kernel::Avx512 | Kernel::Avx512Batch, false)
        | (Kernel::Avx2 | Kernel::Avx2Batch, false) => Kernel::Scalar,
        (k, _) => k,
    };

    let warmup = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warmup);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                chande_scalar(high, low, close, period, mult, &dir, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                chande_avx2(high, low, close, period, mult, &dir, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_avx512(high, low, close, period, mult, &dir, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(ChandeOutput { values: out })
}

/// Writes Chande Exits values into a caller-provided output slice without allocating.
///
/// - Preserves NaN warmup semantics identical to the Vec-returning API.
/// - `out.len()` must equal the input length; otherwise an error is returned.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn chande_into(input: &ChandeInput, out: &mut [f64]) -> Result<(), ChandeError> {
    chande_into_slice(out, input, Kernel::Auto)
}

/// Helper function to compute chande directly into a pre-allocated slice
#[inline]
pub fn chande_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    direction: &str,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), ChandeError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }
    if out.len() != high.len() {
        return Err(ChandeError::OutputLengthMismatch { expected: high.len(), got: out.len() });
    }
    let len = high.len();
    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(ChandeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(ChandeError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let dir = direction.to_lowercase();
    if dir != "long" && dir != "short" {
        return Err(ChandeError::InvalidDirection { direction: dir });
    }

    let warmup = first + period - 1;
    let warmup_end = warmup.min(out.len());
    for v in &mut out[..warmup_end] {
        *v = f64::NAN;
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Safe fallback when AVX isn't available
    let chosen = match (
        chosen,
        cfg!(all(feature = "nightly-avx", target_arch = "x86_64")),
    ) {
        (Kernel::Avx512 | Kernel::Avx512Batch, false)
        | (Kernel::Avx2 | Kernel::Avx2Batch, false) => Kernel::Scalar,
        (k, _) => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                chande_scalar(high, low, close, period, mult, &dir, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                chande_avx2(high, low, close, period, mult, &dir, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_avx512(high, low, close, period, mult, &dir, first, out)
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Helper function for WASM to compute chande directly into a pre-allocated slice
/// This follows the pattern from alma_into_slice for zero-copy operations
#[inline]
pub fn chande_into_slice(
    dst: &mut [f64],
    input: &ChandeInput,
    kern: Kernel,
) -> Result<(), ChandeError> {
    let (high, low, close) = input.borrow_slices();
    let p = input.get_period();
    let m = input.get_mult();
    let d = input.get_direction();
    chande_compute_into(high, low, close, p, m, d, kern, dst)
}

#[inline]
pub fn chande_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    let len = high.len();
    if first >= len {
        return;
    }

    let alpha = 1.0 / period as f64;
    let warmup = first + period - 1;

    // Wilder's ATR via RMA over True Range, single pass
    let mut sum_tr = 0.0f64;
    let mut rma = 0.0f64;
    let mut prev_close = close[first];

    // Monotonic deques: store indices, access values from slices
    use std::collections::VecDeque;

    if dir == "long" {
        // Max-deque over highs
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            // True Range
            let hi = high[i];
            let lo = low[i];
            let tr = if i == first {
                hi - lo
            } else {
                let hl = hi - lo;
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                hl.max(hc).max(lc)
            };

            // Maintain deque window (remove out-of-window indices once full)
            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            // Push current index, maintain decreasing values
            while let Some(&j) = dq.back() {
                if high[j] <= hi {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            // ATR update
            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;
                // Output: HighestHigh - ATR * mult using FMA
                let max_h = high[*dq.front().expect("deque nonempty at warmup")];
                out[i] = (-rma).mul_add(mult, max_h);
            } else {
                // Steady-state RMA update with FMA
                rma = alpha.mul_add(tr - rma, rma);
                let max_h = high[*dq.front().expect("deque nonempty in steady state")];
                out[i] = (-rma).mul_add(mult, max_h);
            }

            prev_close = close[i];
        }
    } else {
        // dir == "short": Min-deque over lows
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            // True Range
            let hi = high[i];
            let lo = low[i];
            let tr = if i == first {
                hi - lo
            } else {
                let hl = hi - lo;
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                hl.max(hc).max(lc)
            };

            // Maintain deque window (remove out-of-window indices once full)
            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            // Push current index, maintain increasing values
            while let Some(&j) = dq.back() {
                if low[j] >= lo {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            // ATR update
            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;
                // Output: LowestLow + ATR * mult using FMA
                let min_l = low[*dq.front().expect("deque nonempty at warmup")];
                out[i] = rma.mul_add(mult, min_l);
            } else {
                // Steady-state RMA update with FMA
                rma = alpha.mul_add(tr - rma, rma);
                let min_l = low[*dq.front().expect("deque nonempty in steady state")];
                out[i] = rma.mul_add(mult, min_l);
            }

            prev_close = close[i];
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    unsafe { chande_fast_unchecked(high, low, close, period, mult, dir, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    // Reuse the same fast scalar-optimized kernel; AVX512 not beneficial end-to-end.
    unsafe { chande_fast_unchecked(high, low, close, period, mult, dir, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_fast_unchecked(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    use std::collections::VecDeque;
    let len = high.len();
    if first >= len {
        return;
    }
    let alpha = 1.0 / period as f64;
    let warmup = first + period - 1;

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let op = out.as_mut_ptr();

    let mut prev_close = *cp.add(first);
    let mut sum_tr = 0.0f64;
    let mut rma = 0.0f64;

    if dir == "long" {
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            let hi = *hp.add(i);
            let lo = *lp.add(i);
            let hl = hi - lo;
            let tr = if i == first {
                hl
            } else {
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                let t = if hl >= hc { hl } else { hc };
                if t >= lc {
                    t
                } else {
                    lc
                }
            };

            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            while let Some(&j) = dq.back() {
                if *hp.add(j) <= hi {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;
                let max_h = *hp.add(*dq.front().unwrap());
                *op.add(i) = (-rma).mul_add(mult, max_h);
            } else {
                rma = alpha.mul_add(tr - rma, rma);
                let max_h = *hp.add(*dq.front().unwrap());
                *op.add(i) = (-rma).mul_add(mult, max_h);
            }
            prev_close = *cp.add(i);
        }
    } else {
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            let hi = *hp.add(i);
            let lo = *lp.add(i);
            let hl = hi - lo;
            let tr = if i == first {
                hl
            } else {
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                let t = if hl >= hc { hl } else { hc };
                if t >= lc {
                    t
                } else {
                    lc
                }
            };

            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            while let Some(&j) = dq.back() {
                if *lp.add(j) >= lo {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;
                let min_l = *lp.add(*dq.front().unwrap());
                *op.add(i) = rma.mul_add(mult, min_l);
            } else {
                rma = alpha.mul_add(tr - rma, rma);
                let min_l = *lp.add(*dq.front().unwrap());
                *op.add(i) = rma.mul_add(mult, min_l);
            }
            prev_close = *cp.add(i);
        }
    }
}

// Decision note: Streaming kernel uses O(1) monotonic deque + Wilder ATR with TR identity; FMA used for tail.
#[derive(Debug, Clone)]
pub struct ChandeStream {
    // Parameters
    period: usize,
    mult: f64,
    direction: String,
    is_long: bool,

    // Precomputed constant
    alpha: f64, // 1.0 / period

    // State
    atr: f64,
    close_prev: f64,
    t: usize,     // logical time (0-based)
    warm: usize,  // number of samples accumulated (<= period)
    filled: bool, // window is “full” -> outputs are valid

    // Monotonic queues (store (value, time))
    max_deque: std::collections::VecDeque<(f64, usize)>,
    min_deque: std::collections::VecDeque<(f64, usize)>,
}

impl ChandeStream {
    pub fn try_new(params: ChandeParams) -> Result<Self, ChandeError> {
        let period = params.period.unwrap_or(22);
        let mult = params.mult.unwrap_or(3.0);
        let direction = params
            .direction
            .unwrap_or_else(|| "long".into())
            .to_lowercase();

        if period == 0 {
            return Err(ChandeError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if direction != "long" && direction != "short" {
            return Err(ChandeError::InvalidDirection { direction });
        }

        let is_long = direction == "long";
        Ok(Self {
            period,
            mult,
            direction,
            is_long,
            alpha: 1.0 / period as f64,
            atr: 0.0,
            close_prev: f64::NAN,
            t: 0,
            warm: 0,
            filled: false,
            max_deque: std::collections::VecDeque::with_capacity(period),
            min_deque: std::collections::VecDeque::with_capacity(period),
        })
    }

    #[inline(always)]
    fn evict_old(&mut self) {
        // keep window [t - (period - 1), t]
        let window_start = self.t.saturating_sub(self.period - 1);
        if self.is_long {
            while let Some(&(_, idx)) = self.max_deque.front() {
                if idx < window_start {
                    self.max_deque.pop_front();
                } else {
                    break;
                }
            }
        } else {
            while let Some(&(_, idx)) = self.min_deque.front() {
                if idx < window_start {
                    self.min_deque.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    #[inline(always)]
    fn push_max(&mut self, v: f64) {
        // maintain non-increasing deque for highs
        while let Some(&(back, _)) = self.max_deque.back() {
            if back <= v {
                self.max_deque.pop_back();
            } else {
                break;
            }
        }
        self.max_deque.push_back((v, self.t));
    }

    #[inline(always)]
    fn push_min(&mut self, v: f64) {
        // maintain non-decreasing deque for lows
        while let Some(&(back, _)) = self.min_deque.back() {
            if back >= v {
                self.min_deque.pop_back();
            } else {
                break;
            }
        }
        self.min_deque.push_back((v, self.t));
    }

    #[inline(always)]
    fn tr(&self, high: f64, low: f64) -> f64 {
        // Wilder’s TR identity:
        // TR = max(high, prev_close) - min(low, prev_close)
        // First observation falls back to high - low
        if self.warm == 0 {
            high - low
        } else {
            let max_h = if high > self.close_prev {
                high
            } else {
                self.close_prev
            };
            let min_l = if low < self.close_prev {
                low
            } else {
                self.close_prev
            };
            max_h - min_l
        }
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        // --- compute TR
        let tr = self.tr(high, low);

        if !self.filled {
            // Warmup: build initial window + accumulate TR
            if self.is_long {
                self.push_max(high);
            } else {
                self.push_min(low);
            }
            self.atr += tr;
            self.warm += 1;

            let now_ready = self.warm == self.period;
            if now_ready {
                self.atr *= self.alpha; // initial ATR = mean(TR[0..period-1])
                self.filled = true;
            }

            self.close_prev = close;
            self.t = self.t.wrapping_add(1);

            if !now_ready {
                return None;
            }
            // Emit the first value at the instant the window fills
            if self.is_long {
                let m = self.max_deque.front().unwrap().0;
                Some((-self.atr).mul_add(self.mult, m)) // m - ATR*mult with FMA
            } else {
                let m = self.min_deque.front().unwrap().0;
                Some(self.atr.mul_add(self.mult, m)) // m + ATR*mult with FMA
            }
        } else {
            // Steady-state: O(1) maintenance
            self.evict_old();
            if self.is_long {
                self.push_max(high);
            } else {
                self.push_min(low);
            }
            // Wilder RMA: atr += alpha * (tr - atr)
            self.atr = self.alpha.mul_add(tr - self.atr, self.atr);

            self.close_prev = close;
            self.t = self.t.wrapping_add(1);

            if self.is_long {
                let m = self.max_deque.front().unwrap().0;
                Some((-self.atr).mul_add(self.mult, m))
            } else {
                let m = self.min_deque.front().unwrap().0;
                Some(self.atr.mul_add(self.mult, m))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChandeBatchRange {
    pub period: (usize, usize, usize),
    pub mult: (f64, f64, f64),
}

impl Default for ChandeBatchRange {
    fn default() -> Self {
        Self {
            period: (22, 22, 0),
            mult: (3.0, 3.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChandeBatchBuilder {
    range: ChandeBatchRange,
    direction: String,
    kernel: Kernel,
}

impl ChandeBatchBuilder {
    pub fn new() -> Self {
        Self {
            range: ChandeBatchRange::default(),
            direction: "long".into(),
            kernel: Kernel::Auto,
        }
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn direction<S: Into<String>>(mut self, d: S) -> Self {
        self.direction = d.into();
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }
    pub fn mult_static(mut self, m: f64) -> Self {
        self.range.mult = (m, m, 0.0);
        self
    }

    pub fn apply_candles(self, c: &Candles) -> Result<ChandeBatchOutput, ChandeError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        chande_batch_with_kernel(high, low, close, &self.range, &self.direction, self.kernel)
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ChandeBatchOutput, ChandeError> {
        chande_batch_with_kernel(high, low, close, &self.range, &self.direction, self.kernel)
    }
}

pub fn chande_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    direction: &str,
    k: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(ChandeError::InvalidKernelForBatch(other));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    chande_batch_par_slice(high, low, close, sweep, direction, simd)
}

#[derive(Clone, Debug)]
pub struct ChandeBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ChandeParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ChandeBatchOutput {
    pub fn row_for_params(&self, p: &ChandeParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(22) == p.period.unwrap_or(22)
                && (c.mult.unwrap_or(3.0) - p.mult.unwrap_or(3.0)).abs() < 1e-12
                && c.direction.as_deref().unwrap_or("long")
                    == p.direction.as_deref().unwrap_or("long")
        })
    }
    pub fn values_for(&self, p: &ChandeParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ChandeBatchRange, dir: &str) -> Result<Vec<ChandeParams>, ChandeError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, ChandeError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        // support reversed bounds
        if start < end {
            if step == 0 { return Ok(vec![start]); }
            Ok((start..=end).step_by(step).collect())
        } else {
            // reversed: start >= end
            let step_i = step as isize;
            if step_i == 0 { return Ok(vec![start]); }
            let mut vals = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            while x >= end_i {
                vals.push(x as usize);
                x = x.saturating_sub(step_i);
                if step_i <= 0 { break; }
            }
            if vals.is_empty() {
                return Err(ChandeError::InvalidRange { start: start as isize, end: end as isize, step: step as isize });
            }
            Ok(vals)
        }
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, ChandeError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += step;
            }
        } else {
            let mut x = start;
            let st = -step.abs();
            while x >= end - 1e-12 {
                v.push(x);
                x += st;
            }
        }
        if v.is_empty() {
            return Err(ChandeError::InvalidRange { start: start as isize, end: end as isize, step: step as isize });
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let mults = axis_f64(r.mult)?;
    // checked capacity to avoid overflow
    let cap = periods
        .len()
        .checked_mul(mults.len())
        .ok_or(ChandeError::InvalidRange { start: 0, end: 0, step: 0 })?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &m in &mults {
            out.push(ChandeParams {
                period: Some(p),
                mult: Some(m),
                direction: Some(dir.to_string()),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn chande_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
    chande_batch_inner(high, low, close, sweep, dir, kern, false)
}

#[inline(always)]
pub fn chande_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
    chande_batch_inner(high, low, close, sweep, dir, kern, true)
}

#[inline(always)]
fn chande_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
    parallel: bool,
) -> Result<ChandeBatchOutput, ChandeError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }

    let combos = expand_grid(sweep, dir)?;
    if combos.is_empty() {
        return Err(ChandeError::InvalidRange { start: 0, end: 0, step: 0 });
    }
    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ChandeError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();
    // Guard rows * cols overflow
    let _total = rows
        .checked_mul(cols)
        .ok_or(ChandeError::InvalidInput("rows*cols overflow".into()))?;

    // Calculate warmup periods for each row
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Allocate uninitialized matrix and set NaN prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    // Convert to mutable slice for computation
    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let values_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let mult = combos[row].mult.unwrap();
        let direction = combos[row].direction.as_deref().unwrap();
        match kern {
            Kernel::Scalar => {
                chande_row_scalar(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                chande_row_avx2(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                chande_row_avx512(high, low, close, first, period, mult, direction, out_row)
            }
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_slice
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(ChandeBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

/// Computes batch chande directly into pre-allocated output slice
#[inline(always)]
fn chande_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ChandeParams>, ChandeError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }

    let combos = expand_grid(sweep, dir)?;
    if combos.is_empty() {
        return Err(ChandeError::InvalidRange { start: 0, end: 0, step: 0 });
    }

    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ChandeError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }

    let cols = high.len();

    // Validate output slice length
    let expected = combos
        .len()
        .checked_mul(cols)
        .ok_or_else(|| ChandeError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(ChandeError::OutputLengthMismatch { expected, got: out.len() });
    }

    // Resolve Auto kernel to concrete kernel
    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    // Initialize NaN prefixes for each row based on warmup period
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.period.unwrap() - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let mult = combos[row].mult.unwrap();
        let direction = combos[row].direction.as_deref().unwrap();
        match actual_kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                chande_row_scalar(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                chande_row_avx2(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_row_avx512(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_row_scalar(high, low, close, first, period, mult, direction, out_row)
            }
            Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
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
unsafe fn chande_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_scalar(high, low, close, period, mult, dir, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    if period <= 32 {
        chande_row_avx512_short(high, low, close, first, period, mult, dir, out)
    } else {
        chande_row_avx512_long(high, low, close, first, period, mult, dir, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_chande_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Prepare input from candles (non-trivial series)
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = ChandeInput::with_default_candles(&candles);

        // Baseline via Vec-returning API
        let baseline = chande(&input)?;

        // Preallocate output and compute via into-API
        let mut out = vec![0.0f64; candles.close.len()];
        #[cfg(not(feature = "wasm"))]
        {
            chande_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, call the internal slice variant directly
            chande_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.values.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline.values[i], out[i]),
                "Mismatch at index {}: got {}, expected {}",
                i,
                out[i],
                baseline.values[i]
            );
        }
        Ok(())
    }

    fn check_chande_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = ChandeParams {
            period: None,
            mult: None,
            direction: None,
        };
        let input = ChandeInput::from_candles(&candles, default_params);
        let output = chande_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_chande_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let chande_result = chande_with_kernel(&input, kernel)?;

        assert_eq!(chande_result.values.len(), close_prices.len());

        let expected_last_five = [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639,
        ];

        assert!(chande_result.values.len() >= 5);
        let start_idx = chande_result.values.len() - 5;
        let actual_last_five = &chande_result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "[{}] Chande Exits mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                exp,
                val
            );
        }

        let period = 22;
        for i in 0..(period - 1) {
            assert!(
                chande_result.values[i].is_nan(),
                "Expected leading NaN at index {}",
                i
            );
        }

        let default_input = ChandeInput::with_default_candles(&candles);
        let default_output = chande_with_kernel(&default_input, kernel)?;
        assert_eq!(default_output.values.len(), close_prices.len());
        Ok(())
    }

    fn check_chande_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(0),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Chande should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_chande_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(99999),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Chande should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_chande_bad_direction(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("bad".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Chande should fail with bad direction",
            test_name
        );
        Ok(())
    }

    fn check_chande_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let result = chande_with_kernel(&input, kernel)?;

        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "[{}] Unexpected NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_chande_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params.clone());
        let batch_output = chande_with_kernel(&input, kernel)?.values;

        let mut stream = ChandeStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), &c) in candles.high.iter().zip(&candles.low).zip(&candles.close) {
            match stream.update(h, l, c) {
                Some(chande_val) => stream_values.push(chande_val),
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
                diff < 1e-8,
                "[{}] Chande streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_chande_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase chance of catching bugs
        let param_combinations = vec![
            ChandeParams {
                period: Some(10),
                mult: Some(2.0),
                direction: Some("long".into()),
            },
            ChandeParams {
                period: Some(22),
                mult: Some(3.0),
                direction: Some("short".into()),
            },
            ChandeParams {
                period: Some(50),
                mult: Some(5.0),
                direction: Some("long".into()),
            },
        ];

        for params in param_combinations {
            let input = ChandeInput::from_candles(&candles, params.clone());
            let output = chande_with_kernel(&input, kernel)?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name, val, bits, i,
                        params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name, val, bits, i,
                        params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name, val, bits, i,
                        params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_chande_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_chande_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_chande_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Note: This test validates Chande Exits invariants including ATR calculation,
        // rolling max/min windows, and directional consistency.

        // Generate test strategy: period, data length, mult, direction
        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                // Generate high/low/close data with realistic relationships
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                )
                .prop_flat_map(move |close| {
                    // Generate high/low based on close with realistic constraints
                    let len = close.len();
                    (
                        Just(close.clone()),
                        prop::collection::vec(
                            0.0f64..1000.0f64, // spread above close
                            len,
                        ),
                        prop::collection::vec(
                            0.0f64..1000.0f64, // spread below close
                            len,
                        ),
                    )
                        .prop_map(move |(c, high_spread, low_spread)| {
                            let high: Vec<f64> = c
                                .iter()
                                .zip(&high_spread)
                                .map(|(&close_val, &spread)| close_val + spread)
                                .collect();
                            let low: Vec<f64> = c
                                .iter()
                                .zip(&low_spread)
                                .map(|(&close_val, &spread)| close_val - spread)
                                .collect();
                            (high, low, c.clone())
                        })
                }),
                Just(period),
                0.1f64..10.0f64, // mult range
                prop::bool::ANY, // direction (true = long, false = short)
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |((high, low, close), period, mult, is_long)| {
                let direction = if is_long { "long" } else { "short" };

                // Build candles structure
                let candles = Candles {
                    high: high.clone(),
                    low: low.clone(),
                    close: close.clone(),
                    timestamp: vec![],
                    open: vec![],
                    volume: vec![],
                    hl2: vec![],
                    hlc3: vec![],
                    ohlc4: vec![],
                    hlcc4: vec![],
                };

                let params = ChandeParams {
                    period: Some(period),
                    mult: Some(mult),
                    direction: Some(direction.to_string()),
                };

                let input = ChandeInput::from_candles(&candles, params);

                // Test with specified kernel
                let result = chande_with_kernel(&input, kernel);

                // Property 1: Should succeed for valid inputs
                prop_assert!(result.is_ok(), "Chande should succeed for valid inputs");
                let output = result.unwrap();

                // Property 2: Output length matches input length
                prop_assert_eq!(
                    output.values.len(),
                    high.len(),
                    "Output length should match input"
                );

                // Find first non-NaN index
                let first_valid = close.iter().position(|&x| !x.is_nan()).unwrap_or(0);
                let warmup_period = first_valid + period - 1;

                // Property 3: Warmup period correctness - NaN values until warmup complete
                for i in 0..warmup_period.min(output.values.len()) {
                    prop_assert!(
                        output.values[i].is_nan(),
                        "Expected NaN during warmup at index {}",
                        i
                    );
                }

                // Property 4: Values after warmup should be finite (if input is finite)
                if warmup_period < output.values.len() {
                    for i in warmup_period..output.values.len() {
                        let val = output.values[i];
                        prop_assert!(
                            val.is_finite(),
                            "Expected finite value after warmup at index {}, got {}",
                            i,
                            val
                        );
                    }
                }

                // Property 5: Long exit should be below or equal to period max high
                // Short exit should be above or equal to period min low
                for i in warmup_period..output.values.len() {
                    let start_idx = i + 1 - period;
                    let period_high = high[start_idx..=i].iter().cloned().fold(f64::MIN, f64::max);
                    let period_low = low[start_idx..=i].iter().cloned().fold(f64::MAX, f64::min);
                    let val = output.values[i];

                    if is_long {
                        // Long exit should be below the period high
                        prop_assert!(
                            val <= period_high + 1e-6,
                            "Long exit {} should be <= period high {} at index {}",
                            val,
                            period_high,
                            i
                        );
                    } else {
                        // Short exit should be above the period low
                        prop_assert!(
                            val >= period_low - 1e-6,
                            "Short exit {} should be >= period low {} at index {}",
                            val,
                            period_low,
                            i
                        );
                    }
                }

                // Property 6: Cross-kernel consistency
                let ref_output = chande_with_kernel(&input, Kernel::Scalar).unwrap();
                for i in 0..output.values.len() {
                    let val = output.values[i];
                    let ref_val = ref_output.values[i];

                    // Handle NaN/infinite values
                    if !val.is_finite() || !ref_val.is_finite() {
                        prop_assert_eq!(
                            val.to_bits(),
                            ref_val.to_bits(),
                            "NaN/Inf mismatch at index {}: {} vs {}",
                            i,
                            val,
                            ref_val
                        );
                        continue;
                    }

                    // Check ULP difference for finite values
                    let val_bits = val.to_bits();
                    let ref_bits = ref_val.to_bits();
                    let ulp_diff = val_bits.abs_diff(ref_bits);

                    prop_assert!(
                        (val - ref_val).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        val,
                        ref_val,
                        ulp_diff
                    );
                }

                // Property 7: Period=1 edge case
                // With period=1, ATR calculation uses a single TR value
                // Due to the complexity of ATR calculation with previous close values,
                // we just verify the basic invariant that the output is finite
                // and follows the directional constraints
                if period == 1 && warmup_period < output.values.len() {
                    for i in warmup_period..output.values.len() {
                        let val = output.values[i];
                        prop_assert!(
                            val.is_finite(),
                            "Period=1 should produce finite values at index {}",
                            i
                        );

                        // Basic directional check
                        if is_long {
                            // Long exit should be <= current high
                            prop_assert!(
                                val <= high[i] + 1e-6,
                                "Period=1 long exit {} should be <= high {} at index {}",
                                val,
                                high[i],
                                i
                            );
                        } else {
                            // Short exit should be >= current low
                            prop_assert!(
                                val >= low[i] - 1e-6,
                                "Period=1 short exit {} should be >= low {} at index {}",
                                val,
                                low[i],
                                i
                            );
                        }
                    }
                }

                // Property 8: Constant data produces stable output (after warmup)
                let all_same_close = close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                let all_same_high = high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                let all_same_low = low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);

                if all_same_close
                    && all_same_high
                    && all_same_low
                    && warmup_period + 10 < output.values.len()
                {
                    // After sufficient warmup, output should stabilize
                    let stable_start = warmup_period + period; // Extra period for ATR to stabilize
                    if stable_start + 2 < output.values.len() {
                        for i in stable_start..output.values.len() - 1 {
                            prop_assert!(
                                (output.values[i] - output.values[i + 1]).abs() <= 1e-6,
                                "Constant data should produce stable output at index {}: {} vs {}",
                                i,
                                output.values[i],
                                output.values[i + 1]
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_chande_tests!(
        check_chande_partial_params,
        check_chande_accuracy,
        check_chande_zero_period,
        check_chande_period_exceeds_length,
        check_chande_bad_direction,
        check_chande_nan_handling,
        check_chande_streaming,
        check_chande_no_poison
    );

    // Generate property tests only when proptest feature is enabled
    #[cfg(feature = "proptest")]
    generate_all_chande_tests!(check_chande_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ChandeBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = ChandeParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = ChandeBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10) // Tests periods 10, 20, 30
            .mult_range(2.0, 5.0, 1.5) // Tests multipliers 2.0, 3.5, 5.0
            .direction("long")
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
            let params = &output.combos[row];

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
            }
        }

        // Also test with "short" direction
        let output_short = ChandeBatchBuilder::new()
            .kernel(kernel)
            .period_range(15, 45, 15) // Tests periods 15, 30, 45
            .mult_range(1.0, 4.0, 1.5) // Tests multipliers 1.0, 2.5, 4.0
            .direction("short")
            .apply_candles(&c)?;

        for (idx, &val) in output_short.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output_short.cols;
            let col = idx % output_short.cols;
            let params = &output_short.combos[row];

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
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

// ============================
// Python Bindings
// ============================

#[cfg(feature = "python")]
#[pyfunction(name = "chande")]
#[pyo3(signature = (high, low, close, period, mult, direction, kernel=None))]
pub fn chande_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
    mult: f64,
    direction: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = ChandeParams {
        period: Some(period),
        mult: Some(mult),
        direction: Some(direction.to_string()),
    };
    let input = ChandeInput::from_slices(h, l, c, params);

    let result_vec = py
        .allow_threads(|| chande_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ChandeStream")]
pub struct ChandeStreamPy {
    stream: ChandeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ChandeStreamPy {
    #[new]
    fn new(period: usize, mult: f64, direction: &str) -> PyResult<Self> {
        let params = ChandeParams {
            period: Some(period),
            mult: Some(mult),
            direction: Some(direction.to_string()),
        };
        let stream =
            ChandeStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(ChandeStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "chande_batch")]
#[pyo3(signature = (high, low, close, period_range, mult_range, direction, kernel=None))]
pub fn chande_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    direction: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;

    let sweep = ChandeBatchRange {
        period: period_range,
        mult: mult_range,
    };
    let combos = expand_grid(&sweep, direction)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = h.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let simd = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        // map Batch to compute kernel like alma.rs
        let simd = match simd {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => simd,
        };
        chande_batch_inner_into(h, l, c, &sweep, direction, simd, true, slice_out)
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
        "mults",
        combos
            .iter()
            .map(|p| p.mult.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "directions",
        combos
            .iter()
            .map(|p| p.direction.as_deref().unwrap())
            .collect::<Vec<_>>(),
    )?;
    Ok(dict)
}

// ============================
// Python CUDA (zero-copy device)
// ============================

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32Chande", unsendable)]
pub struct DeviceArrayF32ChandePy {
    pub(crate) inner: DeviceArrayF32,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32ChandePy {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "DeviceArrayF32Chande cannot be created directly; use chande_cuda_* factories",
        ))
    }

    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (self.inner.cols * itemsize, itemsize))?;
        let ptr_val: usize = self.inner.buf.as_device_ptr().as_raw() as usize;
        d.set_item("data", (ptr_val, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self._device_id as i32)
    }

    #[pyo3(signature = (_stream=None, max_version=None, _dl_device=None, _copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        _stream: Option<pyo3::PyObject>,
        max_version: Option<(u8, u8)>,
        _dl_device: Option<(i32, i32)>,
        _copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use std::os::raw::c_char;
        use std::ptr::null_mut;

        #[repr(C)]
        struct DLDataType {
            code: u8,
            bits: u8,
            lanes: u16,
        }
        #[repr(C)]
        struct DLDevice {
            device_type: i32,
            device_id: i32,
        }
        #[repr(C)]
        struct DLTensor {
            data: *mut std::ffi::c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLManagedTensor {
            dl_tensor: DLTensor,
            manager_ctx: *mut std::ffi::c_void,
            deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
        }

        struct Holder {
            managed: DLManagedTensor,
            shape: [i64; 2],
            strides: [i64; 2],
            arr: DeviceArrayF32,
            _ctx_guard: Arc<Context>,
            _device_id: u32,
        }

        extern "C" fn dl_managed_deleter(mt: *mut DLManagedTensor) {
            if mt.is_null() {
                return;
            }
            unsafe {
                let holder_ptr = (*mt).manager_ctx as *mut Holder;
                if !holder_ptr.is_null() {
                    drop(Box::from_raw(holder_ptr));
                }
            }
        }

        unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            let versioned = b"dltensor_versioned\0";
            let legacy = b"dltensor\0";
            let mut ptr = pyo3::ffi::PyCapsule_GetPointer(
                capsule,
                versioned.as_ptr() as *const c_char,
            );
            if ptr.is_null() {
                ptr = pyo3::ffi::PyCapsule_GetPointer(
                    capsule,
                    legacy.as_ptr() as *const c_char,
                );
            }
            if !ptr.is_null() {
                let mt = ptr as *mut DLManagedTensor;
                if let Some(del) = (*mt).deleter {
                    del(mt);
                }
                pyo3::ffi::PyCapsule_SetPointer(capsule, null_mut());
            }
        }

        // Move VRAM handle into Holder; keep context alive via Arc<Context>.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );
        let ctx_clone = Arc::clone(&self._ctx_guard);

        let rows = inner.rows as i64;
        let cols = inner.cols as i64;

        let mut holder = Box::new(Holder {
            managed: DLManagedTensor {
                dl_tensor: DLTensor {
                    data: if rows == 0 || cols == 0 {
                        std::ptr::null_mut()
                    } else {
                        inner.buf.as_device_ptr().as_raw() as *mut std::ffi::c_void
                    },
                    device: DLDevice {
                        device_type: 2,
                        device_id: self._device_id as i32,
                    },
                    ndim: 2,
                    dtype: DLDataType {
                        code: 2,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(dl_managed_deleter),
            },
            shape: [rows, cols],
            strides: [cols, 1],
            arr: inner,
            _ctx_guard: ctx_clone,
            _device_id: self._device_id,
        });
        holder.managed.dl_tensor.shape = holder.shape.as_mut_ptr();
        holder.managed.dl_tensor.strides = holder.strides.as_mut_ptr();

        let mt_ptr: *mut DLManagedTensor = &mut holder.managed;
        holder.managed.manager_ctx = &mut *holder as *mut Holder as *mut std::ffi::c_void;
        let _ = Box::into_raw(holder);

        let wants_versioned = matches!(max_version, Some((maj, _)) if maj >= 1);
        let name = if wants_versioned {
            b"dltensor_versioned\0"
        } else {
            b"dltensor\0"
        };
        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(
                mt_ptr as *mut std::ffi::c_void,
                name.as_ptr() as *const c_char,
                Some(capsule_destructor),
            )
        };
        if capsule.is_null() {
            unsafe { dl_managed_deleter(mt_ptr) };
            return Err(PyValueError::new_err("failed to create DLPack capsule"));
        }
        Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32ChandePy {
    pub fn new(inner: DeviceArrayF32, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self {
            inner,
            _ctx_guard: ctx_guard,
            _device_id: device_id,
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "chande_cuda_batch_dev")]
#[pyo3(signature = (high, low, close, period_range, mult_range, direction, device_id=0))]
pub fn chande_cuda_batch_dev_py(
    py: Python<'_>,
    high: PyReadonlyArray1<'_, f32>,
    low: PyReadonlyArray1<'_, f32>,
    close: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    direction: &str,
    device_id: usize,
) -> PyResult<DeviceArrayF32ChandePy> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaChande;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }

    let sweep = ChandeBatchRange {
        period: period_range,
        mult: mult_range,
    };

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let mut cuda = CudaChande::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let dev_arr = cuda
            .chande_batch_dev(high_slice, low_slice, close_slice, &sweep, direction)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((dev_arr, ctx, dev_id))
    })?;

    Ok(DeviceArrayF32ChandePy::new(inner, ctx, dev_id))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "chande_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, close_tm, cols, rows, period, mult, direction, device_id=0))]
pub fn chande_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm: PyReadonlyArray1<'_, f32>,
    low_tm: PyReadonlyArray1<'_, f32>,
    close_tm: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    mult: f32,
    direction: &str,
    device_id: usize,
) -> PyResult<DeviceArrayF32ChandePy> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaChande;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high_slice = high_tm.as_slice()?;
    let low_slice = low_tm.as_slice()?;
    let close_slice = close_tm.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if high_slice.len() != expected || low_slice.len() != expected || close_slice.len() != expected
    {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaChande::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let arr = cuda.chande_many_series_one_param_time_major_dev(
            high_slice,
            low_slice,
            close_slice,
            cols,
            rows,
            period,
            mult,
            direction,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
    })?;

    Ok(DeviceArrayF32ChandePy::new(inner, ctx, dev_id))
}

// ============================
// WASM Bindings
// ============================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    direction: &str,
) -> Result<Vec<f64>, JsValue> {
    let params = ChandeParams {
        period: Some(period),
        mult: Some(mult),
        direction: Some(direction.to_string()),
    };
    let input = ChandeInput::from_slices(high, low, close, params);
    let mut out = vec![0.0; high.len()];
    chande_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ChandeBatchConfig {
    pub period_range: (usize, usize, usize),
    pub mult_range: (f64, f64, f64),
    pub direction: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ChandeBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ChandeParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
    direction: &str,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::prelude::*;

    let sweep = ChandeBatchRange {
        period: (period_start, period_end, period_step),
        mult: (mult_start, mult_end, mult_step),
    };

    let simd = detect_best_batch_kernel().to_non_batch();

    let out = chande_batch_inner(
        high,
        low,
        close,
        &sweep,
        direction,
        simd,
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create JS object with values, periods, mults, directions arrays
    let js_obj = js_sys::Object::new();

    // Convert values to JS array
    let values_arr = js_sys::Float64Array::new_with_length(out.values.len() as u32);
    values_arr.copy_from(&out.values);
    js_sys::Reflect::set(&js_obj, &JsValue::from_str("values"), &values_arr.into())?;

    // Extract periods, mults, directions from combos
    let periods: Vec<f64> = out
        .combos
        .iter()
        .map(|c| c.period.unwrap() as f64)
        .collect();
    let mults: Vec<f64> = out.combos.iter().map(|c| c.mult.unwrap()).collect();
    let directions: Vec<String> = out
        .combos
        .iter()
        .map(|c| c.direction.as_ref().unwrap().clone())
        .collect();

    // Convert to JS arrays
    let periods_arr = js_sys::Float64Array::new_with_length(periods.len() as u32);
    periods_arr.copy_from(&periods);
    js_sys::Reflect::set(&js_obj, &JsValue::from_str("periods"), &periods_arr.into())?;

    let mults_arr = js_sys::Float64Array::new_with_length(mults.len() as u32);
    mults_arr.copy_from(&mults);
    js_sys::Reflect::set(&js_obj, &JsValue::from_str("mults"), &mults_arr.into())?;

    // Convert directions to JS array
    let dirs_arr = js_sys::Array::new();
    for dir in &directions {
        dirs_arr.push(&JsValue::from_str(dir));
    }
    js_sys::Reflect::set(&js_obj, &JsValue::from_str("directions"), &dirs_arr.into())?;

    // Add rows and cols
    js_sys::Reflect::set(
        &js_obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &js_obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;

    Ok(js_obj.into())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = chande_batch)]
pub fn chande_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: ChandeBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = ChandeBatchRange {
        period: cfg.period_range,
        mult: cfg.mult_range,
    };
    let simd = detect_best_batch_kernel().to_non_batch();
    let out = chande_batch_inner(
        high,
        low,
        close,
        &sweep,
        &cfg.direction,
        simd,
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = ChandeBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_alloc(len: usize) -> *mut f64 {
    let mut v: Vec<f64> = Vec::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_into(
    h_ptr: *const f64,
    l_ptr: *const f64,
    c_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    mult: f64,
    direction: &str,
) -> Result<(), JsValue> {
    if [
        h_ptr as usize,
        l_ptr as usize,
        c_ptr as usize,
        out_ptr as usize,
    ]
    .iter()
    .any(|&p| p == 0)
    {
        return Err(JsValue::from_str("null pointer passed to chande_into"));
    }
    unsafe {
        let h = std::slice::from_raw_parts(h_ptr, len);
        let l = std::slice::from_raw_parts(l_ptr, len);
        let c = std::slice::from_raw_parts(c_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);

        // Handle aliasing safely
        if out_ptr as *const f64 == h_ptr
            || out_ptr as *const f64 == l_ptr
            || out_ptr as *const f64 == c_ptr
        {
            let mut tmp = vec![0.0; len];
            let params = ChandeParams {
                period: Some(period),
                mult: Some(mult),
                direction: Some(direction.to_string()),
            };
            let input = ChandeInput::from_slices(h, l, c, params);
            chande_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            out.copy_from_slice(&tmp);
        } else {
            let params = ChandeParams {
                period: Some(period),
                mult: Some(mult),
                direction: Some(direction.to_string()),
            };
            let input = ChandeInput::from_slices(h, l, c, params);
            chande_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_batch_into(
    h_ptr: *const f64,
    l_ptr: *const f64,
    c_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    p_start: usize,
    p_end: usize,
    p_step: usize,
    m_start: f64,
    m_end: f64,
    m_step: f64,
    direction: &str,
) -> Result<usize, JsValue> {
    if [
        h_ptr as usize,
        l_ptr as usize,
        c_ptr as usize,
        out_ptr as usize,
    ]
    .iter()
    .any(|&p| p == 0)
    {
        return Err(JsValue::from_str(
            "null pointer passed to chande_batch_into",
        ));
    }
    unsafe {
        let h = std::slice::from_raw_parts(h_ptr, len);
        let l = std::slice::from_raw_parts(l_ptr, len);
        let c = std::slice::from_raw_parts(c_ptr, len);
        let sweep = ChandeBatchRange {
            period: (p_start, p_end, p_step),
            mult: (m_start, m_end, m_step),
        };
        let combos = expand_grid(&sweep, direction)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        // Map Auto to concrete compute kernel
        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        chande_batch_inner_into(h, l, c, &sweep, direction, simd, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
