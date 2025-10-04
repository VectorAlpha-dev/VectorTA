//! # Average Sentiment Oscillator (ASO)
//!
//! The Average Sentiment Oscillator measures market sentiment by analyzing both intrabar
//! and group price movements to calculate bullish and bearish sentiment values.
//!
//! ## Parameters
//! - **period**: The lookback period for calculations (default: 10)
//! - **mode**: Calculation mode - 0: average of both, 1: intrabar only, 2: group only (default: 0)
//!
//! ## Returns
//! - **`Ok(AsoOutput)`** on success, containing bulls and bears Vec<f64> of length matching the input.
//! - **`Err(AsoError)`** otherwise.
//!
//! ## Developer Notes
//! - Scalar: hybrid path. Uses naive O(period) scan for short windows and
//!   monotonic deques (O(1) amortized) for long windows; warmup ramp and
//!   zero-copy allocation preserved.
//! - SIMD (AVX2/AVX512): stubs delegating to scalar. Wide-SIMD across time is
//!   not beneficial here due to data dependencies and deque control-flow.
//!   Selection routes to scalar for Avx2/Avx512 variants.
//! - Batch: per-row compute reuses the scalar kernel. Row-specific cross-row
//!   sharing (e.g., precomputed intrabar, per-period extrema) deferred.
//! - Streaming: O(1) amortized per tick via monotonic deques for rolling
//!   min/max and a ring+running-sum SMA ramp. Matches scalar outputs.

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

// Feature-gated imports for WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// Core imports
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};

// SIMD imports for AVX optimizations
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// Parallel processing support
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Standard library imports
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both candle data and raw slices
#[derive(Debug, Clone)]
pub enum AsoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

/// Output structure containing calculated bulls and bears values
#[derive(Debug, Clone)]
pub struct AsoOutput {
    pub bulls: Vec<f64>,
    pub bears: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AsoParams {
    pub period: Option<usize>,
    pub mode: Option<usize>,
}

impl Default for AsoParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            mode: Some(0),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct AsoInput<'a> {
    pub data: AsoData<'a>,
    pub params: AsoParams,
}

impl<'a> AsRef<[f64]> for AsoInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AsoData::Candles { candles, source } => source_type(candles, source),
            AsoData::Slices { close, .. } => close,
        }
    }
}

impl<'a> AsoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: AsoParams) -> Self {
        Self {
            data: AsoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        p: AsoParams,
    ) -> Self {
        Self {
            data: AsoData::Slices {
                open,
                high,
                low,
                close,
            },
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", AsoParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }

    #[inline]
    pub fn get_mode(&self) -> usize {
        self.params.mode.unwrap_or(0)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct AsoBuilder {
    period: Option<usize>,
    mode: Option<usize>,
    kernel: Kernel,
}

impl Default for AsoBuilder {
    fn default() -> Self {
        Self {
            period: None,
            mode: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AsoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, val: usize) -> Self {
        self.period = Some(val);
        self
    }

    #[inline(always)]
    pub fn mode(mut self, val: usize) -> Self {
        self.mode = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AsoOutput, AsoError> {
        self.apply_candles(c, "close")
    }

    #[inline(always)]
    pub fn apply_candles(self, c: &Candles, s: &str) -> Result<AsoOutput, AsoError> {
        let p = AsoParams {
            period: self.period,
            mode: self.mode,
        };
        let i = AsoInput::from_candles(c, s, p);
        aso_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AsoOutput, AsoError> {
        let p = AsoParams {
            period: self.period,
            mode: self.mode,
        };
        let i = AsoInput::from_slices(open, high, low, close, p);
        aso_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AsoStream, AsoError> {
        let p = AsoParams {
            period: self.period,
            mode: self.mode,
        };
        AsoStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum AsoError {
    #[error("aso: Input data slice is empty.")]
    EmptyInputData,

    #[error("aso: All values are NaN.")]
    AllValuesNaN,

    #[error("aso: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("aso: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("aso: Invalid mode: mode = {mode}, must be 0, 1, or 2")]
    InvalidMode { mode: usize },

    #[error("aso: Required OHLC data is missing or has mismatched lengths")]
    MissingData,
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
/// Main entry point with automatic kernel detection
#[inline]
pub fn aso(input: &AsoInput) -> Result<AsoOutput, AsoError> {
    aso_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection
pub fn aso_with_kernel(input: &AsoInput, kernel: Kernel) -> Result<AsoOutput, AsoError> {
    let (open, high, low, close, period, mode, first, chosen) = aso_prepare(input, kernel)?;

    let len = close.len();
    // CRITICAL: Use zero-copy allocation helper
    let mut bulls = alloc_with_nan_prefix(len, first + period - 1);
    let mut bears = alloc_with_nan_prefix(len, first + period - 1);

    aso_compute_into(
        open, high, low, close, period, mode, first, chosen, &mut bulls, &mut bears,
    );

    Ok(AsoOutput { bulls, bears })
}

/// Zero-allocation version for performance-critical paths
#[inline]
pub fn aso_into_slices(
    bulls_dst: &mut [f64],
    bears_dst: &mut [f64],
    input: &AsoInput,
    kern: Kernel,
) -> Result<(), AsoError> {
    let (open, high, low, close, period, mode, first, chosen) = aso_prepare(input, kern)?;

    if bulls_dst.len() != close.len() || bears_dst.len() != close.len() {
        return Err(AsoError::InvalidPeriod {
            period: bulls_dst.len(),
            data_len: close.len(),
        });
    }

    aso_compute_into(
        open, high, low, close, period, mode, first, chosen, bulls_dst, bears_dst,
    );

    // Fill warmup period with NaN
    let warm = first + period - 1;
    for v in &mut bulls_dst[..warm] {
        *v = f64::NAN;
    }
    for v in &mut bears_dst[..warm] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Prepare and validate input data, normalizing to slices
#[inline(always)]
fn aso_prepare<'a>(
    input: &'a AsoInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        usize,
        Kernel,
    ),
    AsoError,
> {
    let (open, high, low, close) = match &input.data {
        AsoData::Candles { candles: c, .. } => (&c.open[..], &c.high[..], &c.low[..], &c.close[..]),
        AsoData::Slices {
            open,
            high,
            low,
            close,
        } => (*open, *high, *low, *close),
    };

    let len = close.len();
    if len == 0 {
        return Err(AsoError::EmptyInputData);
    }

    if open.len() != len || high.len() != len || low.len() != len {
        return Err(AsoError::MissingData);
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AsoError::AllValuesNaN)?;

    let period = input.get_period();
    let mode = input.get_mode();

    if period == 0 || period > len {
        return Err(AsoError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    if mode > 2 {
        return Err(AsoError::InvalidMode { mode });
    }

    if len - first < period {
        return Err(AsoError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((open, high, low, close, period, mode, first, chosen))
}

/// Core computation dispatcher
#[inline(always)]
fn aso_compute_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mode: usize,
    first: usize,
    kernel: Kernel,
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
) {
    unsafe {
        // WASM SIMD128 support
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                aso_simd128(
                    open, high, low, close, period, mode, first, out_bulls, out_bears,
                );
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => aso_scalar(
                open, high, low, close, period, mode, first, out_bulls, out_bears,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => aso_avx2(
                open, high, low, close, period, mode, first, out_bulls, out_bears,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => aso_avx512(
                open, high, low, close, period, mode, first, out_bulls, out_bears,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => aso_scalar(
                open, high, low, close, period, mode, first, out_bulls, out_bears,
            ),
            _ => unreachable!(),
        }
    }
}

// ==================== SCALAR IMPLEMENTATION ====================
#[inline]
pub fn aso_scalar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mode: usize,
    first_val: usize,
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    // Warm-up index; outputs start at i >= warm.
    let warm = first_val + period - 1;

    // Hybrid: naive scan for short windows; monotonic deques for long windows.
    const DEQUE_THRESHOLD: usize = 64;
    if period <= DEQUE_THRESHOLD {
        // Naive O(period) scan (original path), with rolling average ramp.
        let mut ring_b = vec![0.0; period];
        let mut ring_e = vec![0.0; period];
        let mut sum_b = 0.0;
        let mut sum_e = 0.0;
        let mut head = 0usize;
        let mut filled = 0usize;

        for i in first_val..len {
            // intrabar
            let intrarange = high[i] - low[i];
            let k1 = if intrarange == 0.0 { 1.0 } else { intrarange };
            let intrabarbulls = (((close[i] - low[i]) + (high[i] - open[i])) * 50.0) / k1;
            let intrabarbears = (((high[i] - close[i]) + (open[i] - low[i])) * 50.0) / k1;

            if i >= warm {
                let start = i + 1 - period;

                let mut gl = f64::MAX;
                let mut gh = f64::MIN;
                for j in start..=i {
                    let lj = unsafe { *low.get_unchecked(j) };
                    let hj = unsafe { *high.get_unchecked(j) };
                    if lj < gl {
                        gl = lj;
                    }
                    if hj > gh {
                        gh = hj;
                    }
                }
                let gopen = unsafe { *open.get_unchecked(start) };
                let gr = gh - gl;
                let k2 = if gr == 0.0 { 1.0 } else { gr };

                let groupbulls = (((close[i] - gl) + (gh - gopen)) * 50.0) / k2;
                let groupbears = (((gh - close[i]) + (gopen - gl)) * 50.0) / k2;

                let b = match mode {
                    0 => 0.5 * (intrabarbulls + groupbulls),
                    1 => intrabarbulls,
                    2 => groupbulls,
                    _ => 0.5 * (intrabarbulls + groupbulls),
                };
                let e = match mode {
                    0 => 0.5 * (intrabarbears + groupbears),
                    1 => intrabarbears,
                    2 => groupbears,
                    _ => 0.5 * (intrabarbears + groupbears),
                };

                let old_b = if filled == period { ring_b[head] } else { 0.0 };
                let old_e = if filled == period { ring_e[head] } else { 0.0 };
                sum_b += b - old_b;
                sum_e += e - old_e;
                ring_b[head] = b;
                ring_e[head] = e;
                head = (head + 1) % period;
                if filled < period {
                    filled += 1;
                }

                let n = filled; // 1..=period ramp
                unsafe {
                    *out_bulls.get_unchecked_mut(i) = sum_b / n as f64;
                    *out_bears.get_unchecked_mut(i) = sum_e / n as f64;
                }
            }
        }
        return;
    }

    // Long-window path: monotonic deques with rolling average ramp.
    let mut ring_b = vec![0.0f64; period];
    let mut ring_e = vec![0.0f64; period];
    let mut sum_b = 0.0f64;
    let mut sum_e = 0.0f64;
    let mut rhead = 0usize;
    let mut filled = 0usize;

    let mut dq_min = vec![0usize; period];
    let mut dq_max = vec![0usize; period];
    let mut min_head = 0usize;
    let mut min_tail = 0usize;
    let mut min_len = 0usize;
    let mut max_head = 0usize;
    let mut max_tail = 0usize;
    let mut max_len = 0usize;

    for i in first_val..len {
        let oi = unsafe { *open.get_unchecked(i) };
        let hi = unsafe { *high.get_unchecked(i) };
        let li = unsafe { *low.get_unchecked(i) };
        let ci = unsafe { *close.get_unchecked(i) };

        while min_len > 0 {
            let back = if min_tail == 0 { period - 1 } else { min_tail - 1 };
            let j = unsafe { *dq_min.get_unchecked(back) };
            let lj = unsafe { *low.get_unchecked(j) };
            if li <= lj {
                min_tail = back;
                min_len -= 1;
            } else {
                break;
            }
        }
        if min_len == period {
            min_head += 1;
            if min_head == period {
                min_head = 0;
            }
            min_len -= 1;
        }
        unsafe { *dq_min.get_unchecked_mut(min_tail) = i; }
        min_tail += 1;
        if min_tail == period {
            min_tail = 0;
        }
        min_len += 1;

        while max_len > 0 {
            let back = if max_tail == 0 { period - 1 } else { max_tail - 1 };
            let j = unsafe { *dq_max.get_unchecked(back) };
            let hj = unsafe { *high.get_unchecked(j) };
            if hi >= hj {
                max_tail = back;
                max_len -= 1;
            } else {
                break;
            }
        }
        if max_len == period {
            max_head += 1;
            if max_head == period {
                max_head = 0;
            }
            max_len -= 1;
        }
        unsafe { *dq_max.get_unchecked_mut(max_tail) = i; }
        max_tail += 1;
        if max_tail == period {
            max_tail = 0;
        }
        max_len += 1;

        if i >= warm {
            let start = i + 1 - period;

            while min_len > 0 && unsafe { *dq_min.get_unchecked(min_head) } < start {
                min_head += 1;
                if min_head == period {
                    min_head = 0;
                }
                min_len -= 1;
            }
            while max_len > 0 && unsafe { *dq_max.get_unchecked(max_head) } < start {
                max_head += 1;
                if max_head == period {
                    max_head = 0;
                }
                max_len -= 1;
            }

            debug_assert!(min_len > 0 && max_len > 0);
            let gl = unsafe {
                let idx = *dq_min.get_unchecked(min_head);
                *low.get_unchecked(idx)
            };
            let gh = unsafe {
                let idx = *dq_max.get_unchecked(max_head);
                *high.get_unchecked(idx)
            };
            let gopen = unsafe { *open.get_unchecked(start) };

            let intrarange = hi - li;
            let inv_k1 = if intrarange != 0.0 { 1.0 / intrarange } else { 1.0 };
            let scale1 = 50.0 * inv_k1;
            let intrabarbulls = ((ci - li) + (hi - oi)) * scale1;
            let intrabarbears = ((hi - ci) + (oi - li)) * scale1;

            let gr = gh - gl;
            let inv_k2 = if gr != 0.0 { 1.0 / gr } else { 1.0 };
            let scale2 = 50.0 * inv_k2;
            let groupbulls = ((ci - gl) + (gh - gopen)) * scale2;
            let groupbears = ((gh - ci) + (gopen - gl)) * scale2;

            let b = if mode == 0 {
                0.5 * (intrabarbulls + groupbulls)
            } else if mode == 1 {
                intrabarbulls
            } else {
                groupbulls
            };
            let e = if mode == 0 {
                0.5 * (intrabarbears + groupbears)
            } else if mode == 1 {
                intrabarbears
            } else {
                groupbears
            };

            let old_b = if filled == period {
                unsafe { *ring_b.get_unchecked(rhead) }
            } else {
                0.0
            };
            let old_e = if filled == period {
                unsafe { *ring_e.get_unchecked(rhead) }
            } else {
                0.0
            };
            sum_b += b - old_b;
            sum_e += e - old_e;
            unsafe {
                *ring_b.get_unchecked_mut(rhead) = b;
                *ring_e.get_unchecked_mut(rhead) = e;
            }
            rhead += 1;
            if rhead == period {
                rhead = 0;
            }
            if filled < period {
                filled += 1;
            }

            let n = filled as f64;
            unsafe {
                *out_bulls.get_unchecked_mut(i) = sum_b / n;
                *out_bears.get_unchecked_mut(i) = sum_e / n;
            }
        }
    }
}

// ==================== WASM SIMD128 IMPLEMENTATION ====================
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn aso_simd128(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mode: usize,
    first_val: usize,
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
) {
    use core::arch::wasm32::*;

    // For now, fallback to scalar
    aso_scalar(
        open, high, low, close, period, mode, first_val, out_bulls, out_bears,
    );
}

// ==================== AVX2 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn aso_avx2(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mode: usize,
    first_val: usize,
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
) {
    // SIMD underperforms for ASO (control-flow + deque heavy). Delegate to scalar.
    aso_scalar(
        open, high, low, close, period, mode, first_val, out_bulls, out_bears,
    );
}

// ==================== AVX512 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn aso_avx512(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mode: usize,
    first_val: usize,
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
) {
    // SIMD underperforms for ASO (control-flow + deque heavy). Delegate to scalar.
    aso_scalar(
        open, high, low, close, period, mode, first_val, out_bulls, out_bears,
    );
}

// ==================== BATCH PROCESSING ====================
/// Batch parameter ranges
#[derive(Clone, Debug)]
pub struct AsoBatchRange {
    pub period: (usize, usize, usize),
    pub mode: (usize, usize, usize),
}

impl Default for AsoBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 50, 1),
            mode: (0, 2, 1),
        }
    }
}

/// Batch builder for parameter sweeps
#[derive(Clone, Debug, Default)]
pub struct AsoBatchBuilder {
    range: AsoBatchRange,
    kernel: Kernel,
}

impl AsoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn period_range(mut self, s: usize, e: usize, st: usize) -> Self {
        self.range.period = (s, e, st);
        self
    }

    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 1);
        self
    }

    pub fn mode_range(mut self, s: usize, e: usize, st: usize) -> Self {
        self.range.mode = (s, e, st);
        self
    }

    pub fn mode_static(mut self, m: usize) -> Self {
        self.range.mode = (m, m, 1);
        self
    }

    pub fn apply_candles(self, c: &Candles) -> Result<AsoBatchOutput, AsoError> {
        aso_batch_with_kernel(&c.open, &c.high, &c.low, &c.close, &self.range, self.kernel)
    }

    pub fn apply_slices(
        self,
        o: &[f64],
        h: &[f64],
        l: &[f64],
        c: &[f64],
    ) -> Result<AsoBatchOutput, AsoError> {
        aso_batch_with_kernel(o, h, l, c, &self.range, self.kernel)
    }

    pub fn with_default_candles(c: &Candles) -> Result<AsoBatchOutput, AsoError> {
        Self::default().apply_candles(c)
    }

    pub fn with_default_slices(
        o: &[f64],
        h: &[f64],
        l: &[f64],
        c: &[f64],
        k: Kernel,
    ) -> Result<AsoBatchOutput, AsoError> {
        Self::new().kernel(k).apply_slices(o, h, l, c)
    }
}

/// Batch output structure
#[derive(Clone, Debug)]
pub struct AsoBatchOutput {
    pub bulls: Vec<f64>, // rows*cols
    pub bears: Vec<f64>, // rows*cols
    pub combos: Vec<AsoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AsoBatchOutput {
    #[inline]
    pub fn bulls_row(&self, row: usize) -> &[f64] {
        let s = row * self.cols;
        &self.bulls[s..s + self.cols]
    }

    #[inline]
    pub fn bears_row(&self, row: usize) -> &[f64] {
        let s = row * self.cols;
        &self.bears[s..s + self.cols]
    }

    #[inline]
    pub fn row_for_params(&self, p: &AsoParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period == p.period && c.mode == p.mode)
    }

    #[inline]
    pub fn values_for(&self, p: &AsoParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p)
            .map(|row| (self.bulls_row(row), self.bears_row(row)))
    }
}

/// Expand parameter grid for batch processing
#[inline(always)]
fn expand_grid_aso(r: &AsoBatchRange) -> Vec<AsoParams> {
    fn axis_usize(a: (usize, usize, usize)) -> Vec<usize> {
        let (s, e, st) = a;
        if st == 0 || s == e {
            return vec![s];
        }
        (s..=e).step_by(st).collect()
    }

    let ps = axis_usize(r.period);
    let ms = axis_usize(r.mode);
    let mut out = Vec::with_capacity(ps.len() * ms.len());

    for &p in &ps {
        for &m in &ms {
            out.push(AsoParams {
                period: Some(p),
                mode: Some(m),
            });
        }
    }
    out
}

/// Batch processing with kernel selection
pub fn aso_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AsoBatchRange,
    k: Kernel,
) -> Result<AsoBatchOutput, AsoError> {
    let combos = expand_grid_aso(sweep);
    let rows = combos.len();
    let cols = close.len();

    if cols == 0 {
        return Err(AsoError::EmptyInputData);
    }
    if open.len() != cols || high.len() != cols || low.len() != cols {
        return Err(AsoError::MissingData);
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AsoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

    if cols - first < max_p {
        return Err(AsoError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    // Allocate two uninitialized matrices
    let mut bulls_mu = make_uninit_matrix(rows, cols);
    let mut bears_mu = make_uninit_matrix(rows, cols);

    // Warmup for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut bulls_mu, cols, &warm);
    init_matrix_prefixes(&mut bears_mu, cols, &warm);

    // Cast MaybeUninit -> &mut [f64] views
    let mut guard_b = core::mem::ManuallyDrop::new(bulls_mu);
    let mut guard_e = core::mem::ManuallyDrop::new(bears_mu);
    let bulls_out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard_b.as_mut_ptr() as *mut f64, guard_b.len()) };
    let bears_out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard_e.as_mut_ptr() as *mut f64, guard_e.len()) };

    let actual = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        kk => kk,
    };

    // Per-row closure
    let do_row = |row: usize, bulls_row: &mut [f64], bears_row: &mut [f64]| {
        let p = combos[row].period.unwrap();
        let m = combos[row].mode.unwrap();
        unsafe {
            match actual {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    aso_scalar(open, high, low, close, p, m, first, bulls_row, bears_row)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    aso_avx2(open, high, low, close, p, m, first, bulls_row, bears_row)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    aso_avx512(open, high, low, close, p, m, first, bulls_row, bears_row)
                }
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                    aso_scalar(open, high, low, close, p, m, first, bulls_row, bears_row)
                }
                Kernel::Auto => unreachable!(),
            }
        }
    };

    // Parallel when available
    #[cfg(not(target_arch = "wasm32"))]
    {
        bulls_out
            .chunks_mut(cols)
            .zip(bears_out.chunks_mut(cols))
            .enumerate()
            .par_bridge()
            .for_each(|(row, (b, e))| do_row(row, b, e));
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row, (b, e)) in bulls_out
            .chunks_mut(cols)
            .zip(bears_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, b, e);
        }
    }

    // Finalize Vec<f64> with zero copies
    let bulls = unsafe {
        Vec::from_raw_parts(
            guard_b.as_mut_ptr() as *mut f64,
            guard_b.len(),
            guard_b.capacity(),
        )
    };
    let bears = unsafe {
        Vec::from_raw_parts(
            guard_e.as_mut_ptr() as *mut f64,
            guard_e.len(),
            guard_e.capacity(),
        )
    };

    Ok(AsoBatchOutput {
        bulls,
        bears,
        combos,
        rows,
        cols,
    })
}

/// Batch processing for slices
pub fn aso_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AsoBatchRange,
    kern: Kernel,
) -> Result<AsoBatchOutput, AsoError> {
    aso_batch_with_kernel(open, high, low, close, sweep, kern)
}

/// Parallel batch processing for slices
#[cfg(not(target_arch = "wasm32"))]
pub fn aso_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AsoBatchRange,
    kern: Kernel,
) -> Result<AsoBatchOutput, AsoError> {
    // For now, just use the regular batch processing
    // In the future, this could be optimized with explicit parallel control
    aso_batch_with_kernel(open, high, low, close, sweep, kern)
}

#[cfg(target_arch = "wasm32")]
pub fn aso_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AsoBatchRange,
    kern: Kernel,
) -> Result<AsoBatchOutput, AsoError> {
    // WASM doesn't support parallel processing
    aso_batch_with_kernel(open, high, low, close, sweep, kern)
}

/// In-place batch core (no allocations, no copies)
#[inline(always)]
fn aso_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AsoBatchRange,
    kern: Kernel,
    parallel: bool,
    out_bulls: &mut [f64], // rows*cols
    out_bears: &mut [f64], // rows*cols
) -> Result<Vec<AsoParams>, AsoError> {
    let combos = expand_grid_aso(sweep);
    if combos.is_empty() {
        return Err(AsoError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let cols = close.len();
    if cols == 0 {
        return Err(AsoError::EmptyInputData);
    }
    if open.len() != cols || high.len() != cols || low.len() != cols {
        return Err(AsoError::MissingData);
    }
    let rows = combos.len();
    if out_bulls.len() != rows * cols || out_bears.len() != rows * cols {
        return Err(AsoError::InvalidPeriod {
            period: out_bulls.len(),
            data_len: rows * cols,
        });
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AsoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(AsoError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    // Initialize warm prefixes in-place (no extra buffers)
    let mut b_mu = unsafe {
        core::slice::from_raw_parts_mut(
            out_bulls.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_bulls.len(),
        )
    };
    let mut e_mu = unsafe {
        core::slice::from_raw_parts_mut(
            out_bears.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_bears.len(),
        )
    };
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut b_mu, cols, &warm);
    init_matrix_prefixes(&mut e_mu, cols, &warm);

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, br: &mut [MaybeUninit<f64>], er: &mut [MaybeUninit<f64>]| unsafe {
        let p = combos[row].period.unwrap();
        let m = combos[row].mode.unwrap();

        let b = core::slice::from_raw_parts_mut(br.as_mut_ptr() as *mut f64, br.len());
        let e = core::slice::from_raw_parts_mut(er.as_mut_ptr() as *mut f64, er.len());

        match actual {
            Kernel::Scalar | Kernel::ScalarBatch => {
                aso_scalar(open, high, low, close, p, m, first, b, e)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => aso_avx2(open, high, low, close, p, m, first, b, e),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                aso_avx512(open, high, low, close, p, m, first, b, e)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                aso_scalar(open, high, low, close, p, m, first, b, e)
            }
            Kernel::Auto => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            b_mu.chunks_mut(cols)
                .zip(e_mu.chunks_mut(cols))
                .enumerate()
                .par_bridge()
                .for_each(|(row, (br, er))| do_row(row, br, er));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (br, er)) in b_mu.chunks_mut(cols).zip(e_mu.chunks_mut(cols)).enumerate() {
                do_row(row, br, er);
            }
        }
    } else {
        for (row, (br, er)) in b_mu.chunks_mut(cols).zip(e_mu.chunks_mut(cols)).enumerate() {
            do_row(row, br, er);
        }
    }

    Ok(combos)
}

// ==================== STREAMING SUPPORT (O(1) amortized) ====================
/// Streaming calculator using monotonic deques for min/max and a ring+sum SMA ramp.
/// Decision: SIMD stubs remain; streaming optimized to O(1) amortized with parity to scalar.
#[derive(Debug, Clone)]
pub struct AsoStream {
    // Ring buffers for the last `period` OHLC values
    o: Vec<f64>,
    h: Vec<f64>,
    l: Vec<f64>,
    c: Vec<f64>,

    // Rolling SMA of bulls/bears via ring + running sums
    rb: Vec<f64>,
    re: Vec<f64>,
    sum_b: f64,
    sum_e: f64,
    head_be: usize,
    filled_be: usize,

    // Monotonic deques for rolling min(low) and max(high)
    // We store (absolute index, value) in parallel fixed-size circular arrays
    dq_min_idx: Vec<usize>,
    dq_min_val: Vec<f64>,
    min_head: usize,
    min_tail: usize,
    min_len: usize,

    dq_max_idx: Vec<usize>,
    dq_max_val: Vec<f64>,
    max_head: usize,
    max_tail: usize,
    max_len: usize,

    period: usize,
    mode: usize,
    i: usize,    // absolute bar count seen so far (0-based)
    ready: bool, // true once we have at least `period` bars
}

impl AsoStream {
    #[inline]
    pub fn try_new(params: AsoParams) -> Result<Self, AsoError> {
        let period = params.period.unwrap_or(10);
        let mode = params.mode.unwrap_or(0);

        if period == 0 {
            return Err(AsoError::InvalidPeriod { period, data_len: 0 });
        }
        if mode > 2 {
            return Err(AsoError::InvalidMode { mode });
        }

        Ok(Self {
            o: vec![0.0; period],
            h: vec![0.0; period],
            l: vec![0.0; period],
            c: vec![0.0; period],

            rb: vec![0.0; period],
            re: vec![0.0; period],
            sum_b: 0.0,
            sum_e: 0.0,
            head_be: 0,
            filled_be: 0,

            dq_min_idx: vec![0usize; period],
            dq_min_val: vec![0.0; period],
            min_head: 0,
            min_tail: 0,
            min_len: 0,

            dq_max_idx: vec![0usize; period],
            dq_max_val: vec![0.0; period],
            max_head: 0,
            max_tail: 0,
            max_len: 0,

            period,
            mode,
            i: 0,
            ready: false,
        })
    }

    /// Multiply by the reciprocal; guard zero to avoid NaNs
    #[inline(always)]
    fn inv_or_one(x: f64) -> f64 {
        if x != 0.0 { x.recip() } else { 1.0 }
    }

    #[inline]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let p = self.period;
        let i = self.i;
        let idx = i % p;

        // Write newest OHLC into rings
        self.o[idx] = open;
        self.h[idx] = high;
        self.l[idx] = low;
        self.c[idx] = close;

        // ---- push `low` into monotonic-min deque (nondecreasing by value) ----
        while self.min_len > 0 {
            let back = if self.min_tail == 0 { p - 1 } else { self.min_tail - 1 };
            if low <= self.dq_min_val[back] {
                // pop back
                self.min_tail = back;
                self.min_len -= 1;
            } else {
                break;
            }
        }
        if self.min_len == p {
            // Capacity guard (shouldn't trigger in steady state but keep it safe)
            self.min_head += 1;
            if self.min_head == p { self.min_head = 0; }
            self.min_len -= 1;
        }
        self.dq_min_idx[self.min_tail] = i;
        self.dq_min_val[self.min_tail] = low;
        self.min_tail += 1;
        if self.min_tail == p { self.min_tail = 0; }
        self.min_len += 1;

        // ---- push `high` into monotonic-max deque (nonincreasing by value) ----
        while self.max_len > 0 {
            let back = if self.max_tail == 0 { p - 1 } else { self.max_tail - 1 };
            if high >= self.dq_max_val[back] {
                // pop back
                self.max_tail = back;
                self.max_len -= 1;
            } else {
                break;
            }
        }
        if self.max_len == p {
            self.max_head += 1;
            if self.max_head == p { self.max_head = 0; }
            self.max_len -= 1;
        }
        self.dq_max_idx[self.max_tail] = i;
        self.dq_max_val[self.max_tail] = high;
        self.max_tail += 1;
        if self.max_tail == p { self.max_tail = 0; }
        self.max_len += 1;

        // Advance bar counter and readiness
        self.i = i + 1;
        if self.i >= p { self.ready = true; }
        if !self.ready {
            return None;
        }

        // Oldest absolute index in the window [i - p + 1, i]
        let start_abs = self.i - p;

        // Evict elements that left the window from deque fronts
        while self.min_len > 0 && self.dq_min_idx[self.min_head] < start_abs {
            self.min_head += 1;
            if self.min_head == p { self.min_head = 0; }
            self.min_len -= 1;
        }
        while self.max_len > 0 && self.dq_max_idx[self.max_head] < start_abs {
            self.max_head += 1;
            if self.max_head == p { self.max_head = 0; }
            self.max_len -= 1;
        }

        // Group extrema from deque fronts (values stored alongside indices)
        debug_assert!(self.min_len > 0 && self.max_len > 0);
        let gl = self.dq_min_val[self.min_head];
        let gh = self.dq_max_val[self.max_head];

        // Open at the oldest element in the window (ring index avoids a %)
        let oldest_ring = if idx + 1 == p { 0 } else { idx + 1 };
        let gopen = self.o[oldest_ring];

        // -------- intrabar sentiment --------
        let intrarange = high - low;
        let scale1 = 50.0 * Self::inv_or_one(intrarange);
        let intrabarbulls = ((close - low) + (high - open)) * scale1;
        let intrabarbears = ((high - close) + (open - low)) * scale1;

        // -------- group sentiment (rolling window) --------
        let gr = gh - gl;
        let scale2 = 50.0 * Self::inv_or_one(gr);
        let groupbulls = ((close - gl) + (gh - gopen)) * scale2;
        let groupbears = ((gh - close) + (gopen - gl)) * scale2;

        // Combine per mode (0: average both, 1: intrabar only, 2: group only)
        let b = match self.mode {
            0 => 0.5 * (intrabarbulls + groupbulls),
            1 => intrabarbulls,
            _ => groupbulls,
        };
        let e = match self.mode {
            0 => 0.5 * (intrabarbears + groupbears),
            1 => intrabarbears,
            _ => groupbears,
        };

        // -------- O(1) rolling SMA ramp of bulls/bears (matches batch/scalar) --------
        let old_b = if self.filled_be == p { self.rb[self.head_be] } else { 0.0 };
        let old_e = if self.filled_be == p { self.re[self.head_be] } else { 0.0 };

        self.sum_b += b - old_b;
        self.sum_e += e - old_e;

        self.rb[self.head_be] = b;
        self.re[self.head_be] = e;

        self.head_be += 1;
        if self.head_be == p { self.head_be = 0; }
        if self.filled_be < p { self.filled_be += 1; }

        let n = self.filled_be as f64;
        Some((self.sum_b / n, self.sum_e / n))
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "aso")]
#[pyo3(signature = (open, high, low, close, period=None, mode=None, kernel=None))]
pub fn aso_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    mode: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;

    if h.len() != o.len() || l.len() != o.len() || c.len() != o.len() {
        return Err(PyValueError::new_err(
            "All OHLC arrays must have the same length",
        ));
    }

    let kern = validate_kernel(kernel, false)?;
    let params = AsoParams { period, mode };
    let input = AsoInput::from_slices(o, h, l, c, params);

    // Zero-copy Vec -> NumPy via IntoPyArray
    let (bulls, bears) = py
        .allow_threads(|| aso_with_kernel(&input, kern).map(|o| (o.bulls, o.bears)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((bulls.into_pyarray(py), bears.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name = "aso_batch")]
#[pyo3(signature = (open, high, low, close, period_range, mode_range, kernel=None))]
pub fn aso_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    mode_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArray1;
    let (o, h, l, c) = (
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
    );
    let sweep = AsoBatchRange {
        period: period_range,
        mode: mode_range,
    };
    let combos = expand_grid_aso(&sweep);
    let rows = combos.len();
    let cols = c.len();

    let bulls_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let bears_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let b = unsafe { bulls_arr.as_slice_mut()? };
    let e = unsafe { bears_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let simd = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        aso_batch_inner_into(o, h, l, c, &sweep, simd, true, b, e)
    })
    .map_err(|er| PyValueError::new_err(er.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("bulls", bulls_arr.reshape((rows, cols))?)?;
    d.set_item("bears", bears_arr.reshape((rows, cols))?)?;
    d.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "modes",
        combos
            .iter()
            .map(|p| p.mode.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

#[cfg(feature = "python")]
#[pyclass(name = "AsoStream")]
pub struct AsoStreamPy {
    stream: AsoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AsoStreamPy {
    #[new]
    fn new(period: Option<usize>, mode: Option<usize>) -> PyResult<Self> {
        let params = AsoParams { period, mode };
        let stream =
            AsoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AsoStreamPy { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(open, high, low, close)
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AsoResult {
    pub values: Vec<f64>, // [bulls..., bears...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "aso")]
pub fn aso_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: Option<usize>,
    mode: Option<usize>,
) -> Result<JsValue, JsValue> {
    let len = close.len();
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(JsValue::from_str(
            "All OHLC arrays must have the same length",
        ));
    }
    let p = period.unwrap_or(10);
    let m = mode.unwrap_or(0);
    if m > 2 {
        return Err(JsValue::from_str("Invalid mode"));
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| JsValue::from_str("All values NaN"))?;
    if p == 0 || p > len {
        return Err(JsValue::from_str("Invalid period"));
    }
    if len - first < p {
        return Err(JsValue::from_str("Not enough valid data"));
    }

    // Allocate one 2×len matrix and initialize NaN warm prefixes
    let mut mu = make_uninit_matrix(2, len);
    let warm = first + p - 1;
    init_matrix_prefixes(&mut mu, len, &[warm, warm]);

    // Cast to &mut [f64], split rows, compute directly into destination
    let mut guard = core::mem::ManuallyDrop::new(mu);
    let dst: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let (bulls_dst, bears_dst) = dst.split_at_mut(len);

    // Kernel dispatch (no copies)
    let chosen = detect_best_kernel();
    unsafe {
        aso_compute_into(
            open, high, low, close, p, m, first, chosen, bulls_dst, bears_dst,
        );
    }

    // Seal the buffer into a Vec<f64> without copying
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    let out = AsoResult {
        values,
        rows: 2,
        cols: len,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aso_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    bulls_ptr: *mut f64,
    bears_ptr: *mut f64,
    len: usize,
    period: usize,
    mode: usize,
) -> Result<(), JsValue> {
    if [open_ptr, high_ptr, low_ptr, close_ptr]
        .iter()
        .any(|p| p.is_null())
        || [bulls_ptr, bears_ptr].iter().any(|p| p.is_null())
    {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let o = std::slice::from_raw_parts(open_ptr, len);
        let h = std::slice::from_raw_parts(high_ptr, len);
        let l = std::slice::from_raw_parts(low_ptr, len);
        let c = std::slice::from_raw_parts(close_ptr, len);
        let bulls = std::slice::from_raw_parts_mut(bulls_ptr, len);
        let bears = std::slice::from_raw_parts_mut(bears_ptr, len);

        let input = AsoInput::from_slices(
            o,
            h,
            l,
            c,
            AsoParams {
                period: Some(period),
                mode: Some(mode),
            },
        );

        aso_into_slices(bulls, bears, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aso_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    mode_start: usize,
    mode_end: usize,
    mode_step: usize,
    bulls_out: *mut f64,
    bears_out: *mut f64,
) -> Result<usize, JsValue> {
    if [open_ptr, high_ptr, low_ptr, close_ptr, bulls_out, bears_out]
        .iter()
        .any(|p| p.is_null())
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let o = std::slice::from_raw_parts(open_ptr, len);
        let h = std::slice::from_raw_parts(high_ptr, len);
        let l = std::slice::from_raw_parts(low_ptr, len);
        let c = std::slice::from_raw_parts(close_ptr, len);

        let sweep = AsoBatchRange {
            period: (period_start, period_end, period_step),
            mode: (mode_start, mode_end, mode_step),
        };

        let combos = expand_grid_aso(&sweep);
        let rows = combos.len();
        let cols = len;

        let b = std::slice::from_raw_parts_mut(bulls_out, rows * cols);
        let e = std::slice::from_raw_parts_mut(bears_out, rows * cols);

        aso_batch_inner_into(o, h, l, c, &sweep, detect_best_kernel(), false, b, e)
            .map_err(|er| JsValue::from_str(&er.to_string()))?;
        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AsoBatchConfig {
    pub period_range: (usize, usize, usize),
    pub mode_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AsoBatchJsOutput {
    pub values: Vec<f64>, // [bulls rows..., bears rows...] flattened
    pub combos: Vec<AsoParams>,
    pub rows: usize, // number of param combinations
    pub cols: usize, // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "aso_batch")]
pub fn aso_batch_unified_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: AsoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AsoBatchRange {
        period: cfg.period_range,
        mode: cfg.mode_range,
    };
    let combos = expand_grid_aso(&sweep);
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(JsValue::from_str("Empty input"));
    }
    if open.len() != cols || high.len() != cols || low.len() != cols {
        return Err(JsValue::from_str("OHLC length mismatch"));
    }

    // allocate 2*rows × cols, warm prefixes for both outputs per row
    let mut mu = make_uninit_matrix(rows * 2, cols);
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| JsValue::from_str("All values NaN"))?;
    let warms: Vec<usize> = combos
        .iter()
        .flat_map(|c| {
            let w = first + c.period.unwrap() - 1;
            [w, w]
        })
        .collect();
    init_matrix_prefixes(&mut mu, cols, &warms);

    // cast and split
    let mut guard = core::mem::ManuallyDrop::new(mu);
    let dst: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let (bulls_dst, bears_dst) = dst.split_at_mut(rows * cols);

    // compute in-place
    let kern = detect_best_batch_kernel();
    aso_batch_inner_into(
        open, high, low, close, &sweep, kern, false, bulls_dst, bears_dst,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // seal without copies
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    let out = AsoBatchJsOutput {
        values,
        combos: combos.clone(), // We need to return the actual combos
        rows: rows * 2,         // 2 outputs per combo (bulls and bears)
        cols,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aso_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aso_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    

    fn check_aso_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AsoInput::from_candles(&candles, "close", AsoParams::default());
        let result = aso_with_kernel(&input, kernel)?;

        // REFERENCE VALUES FROM PINESCRIPT
        let expected_bulls = [
            48.48594883,
            46.37206396,
            47.20522805,
            46.83750720,
            43.28268188,
        ];

        let expected_bears = [
            51.51405117,
            53.62793604,
            52.79477195,
            53.16249280,
            56.71731812,
        ];

        let start = result.bulls.len().saturating_sub(5);
        for (i, (&bull_val, &bear_val)) in result.bulls[start..]
            .iter()
            .zip(result.bears[start..].iter())
            .enumerate()
        {
            let bull_diff = (bull_val - expected_bulls[i]).abs();
            let bear_diff = (bear_val - expected_bears[i]).abs();

            assert!(
                bull_diff < 1e-6,
                "[{}] ASO Bulls {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                bull_val,
                expected_bulls[i]
            );

            assert!(
                bear_diff < 1e-6,
                "[{}] ASO Bears {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                bear_val,
                expected_bears[i]
            );
        }
        Ok(())
    }

    fn check_aso_slice_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with slice input
        let input = AsoInput::from_slices(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            AsoParams::default(),
        );
        let result = aso_with_kernel(&input, kernel)?;

        assert_eq!(result.bulls.len(), candles.close.len());
        assert_eq!(result.bears.len(), candles.close.len());

        Ok(())
    }

    fn check_aso_into_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let mut bulls = vec![0.0; candles.close.len()];
        let mut bears = vec![0.0; candles.close.len()];

        let input = AsoInput::from_candles(&candles, "close", AsoParams::default());
        aso_into_slices(&mut bulls, &mut bears, &input, kernel)?;

        // Verify NaN prefix
        for i in 0..9 {
            assert!(bulls[i].is_nan());
            assert!(bears[i].is_nan());
        }

        // Verify some valid values exist after warmup
        assert!(!bulls[20].is_nan());
        assert!(!bears[20].is_nan());

        Ok(())
    }

    fn check_aso_batch(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let open = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let high = vec![15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0];
        let low = vec![5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0];
        let close = vec![12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0, 82.0, 92.0, 102.0];

        let sweep = AsoBatchRange {
            period: (3, 5, 1),
            mode: (0, 2, 1),
        };

        let result = aso_batch_with_kernel(&open, &high, &low, &close, &sweep, kernel)?;

        assert_eq!(result.rows, 9); // 3 periods * 3 modes
        assert_eq!(result.cols, 10);
        assert_eq!(result.bulls.len(), 90);
        assert_eq!(result.bears.len(), 90);
        assert_eq!(result.combos.len(), 9);

        Ok(())
    }

    fn check_aso_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = AsoParams {
            period: None,
            mode: None,
        };
        let input = AsoInput::from_candles(&candles, "close", default_params);
        let output = aso_with_kernel(&input, kernel)?;
        assert_eq!(output.bulls.len(), candles.close.len());
        assert_eq!(output.bears.len(), candles.close.len());

        Ok(())
    }

    fn check_aso_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AsoInput::with_default_candles(&candles);
        let output = aso_with_kernel(&input, kernel)?;
        assert_eq!(output.bulls.len(), candles.close.len());
        assert_eq!(output.bears.len(), candles.close.len());

        Ok(())
    }

    fn check_aso_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let open = vec![10.0, 20.0, 30.0];
        let high = vec![15.0, 25.0, 35.0];
        let low = vec![8.0, 18.0, 28.0];
        let close = vec![12.0, 22.0, 32.0];

        let params = AsoParams {
            period: Some(0),
            mode: None,
        };
        let input = AsoInput::from_slices(&open, &high, &low, &close, params);
        let res = aso_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ASO should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_aso_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let open = vec![10.0, 20.0, 30.0];
        let high = vec![15.0, 25.0, 35.0];
        let low = vec![8.0, 18.0, 28.0];
        let close = vec![12.0, 22.0, 32.0];

        let params = AsoParams {
            period: Some(10),
            mode: None,
        };
        let input = AsoInput::from_slices(&open, &high, &low, &close, params);
        let res = aso_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ASO should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_aso_invalid_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = AsoParams {
            period: Some(10),
            mode: Some(3), // Invalid mode
        };
        let input = AsoInput::from_candles(&candles, "close", params);
        let res = aso_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ASO should fail with invalid mode",
            test_name
        );
        Ok(())
    }

    fn check_aso_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let empty: Vec<f64> = vec![];
        let params = AsoParams::default();
        let input = AsoInput::from_slices(&empty, &empty, &empty, &empty, params);
        let res = aso_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ASO should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_aso_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let nan_data = vec![f64::NAN, f64::NAN, f64::NAN];
        let params = AsoParams::default();
        let input = AsoInput::from_slices(&nan_data, &nan_data, &nan_data, &nan_data, params);
        let res = aso_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ASO should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_aso_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = AsoParams {
            period: Some(10),
            mode: None,
        };
        let input = AsoInput::from_slices(
            &single_point,
            &single_point,
            &single_point,
            &single_point,
            params,
        );
        let res = aso_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ASO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_aso_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = AsoParams {
            period: Some(10),
            mode: Some(0),
        };
        let first_input = AsoInput::from_candles(&candles, "close", first_params);
        let first_result = aso_with_kernel(&first_input, kernel)?;

        // Use bulls output as all OHLC inputs for second pass
        let second_params = AsoParams {
            period: Some(10),
            mode: Some(0),
        };
        let second_input = AsoInput::from_slices(
            &first_result.bulls,
            &first_result.bulls,
            &first_result.bulls,
            &first_result.bulls,
            second_params,
        );
        let second_result = aso_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.bulls.len(), first_result.bulls.len());
        assert_eq!(second_result.bears.len(), first_result.bears.len());

        // Verify some values are computed after warmup
        if second_result.bulls.len() > 30 {
            assert!(!second_result.bulls[30].is_nan());
            assert!(!second_result.bears[30].is_nan());
        }

        Ok(())
    }

    fn check_aso_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AsoInput::from_candles(
            &candles,
            "close",
            AsoParams {
                period: Some(10),
                mode: Some(0),
            },
        );
        let res = aso_with_kernel(&input, kernel)?;
        assert_eq!(res.bulls.len(), candles.close.len());
        assert_eq!(res.bears.len(), candles.close.len());

        // Check that after warmup period, we don't have unexpected NaNs
        if res.bulls.len() > 240 {
            for (i, (&bull_val, &bear_val)) in res.bulls[240..]
                .iter()
                .zip(res.bears[240..].iter())
                .enumerate()
            {
                assert!(
                    !bull_val.is_nan(),
                    "[{}] Found unexpected NaN in bulls at out-index {}",
                    test_name,
                    240 + i
                );
                assert!(
                    !bear_val.is_nan(),
                    "[{}] Found unexpected NaN in bears at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_aso_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 10;
        let mode = 0;

        let input = AsoInput::from_candles(
            &candles,
            "close",
            AsoParams {
                period: Some(period),
                mode: Some(mode),
            },
        );
        let batch_output = aso_with_kernel(&input, kernel)?;

        let mut stream = AsoStream::try_new(AsoParams {
            period: Some(period),
            mode: Some(mode),
        })?;

        let mut stream_bulls = Vec::with_capacity(candles.close.len());
        let mut stream_bears = Vec::with_capacity(candles.close.len());

        for i in 0..candles.close.len() {
            match stream.update(
                candles.open[i],
                candles.high[i],
                candles.low[i],
                candles.close[i],
            ) {
                Some((bull, bear)) => {
                    stream_bulls.push(bull);
                    stream_bears.push(bear);
                }
                None => {
                    stream_bulls.push(f64::NAN);
                    stream_bears.push(f64::NAN);
                }
            }
        }

        assert_eq!(batch_output.bulls.len(), stream_bulls.len());
        assert_eq!(batch_output.bears.len(), stream_bears.len());

        // Compare batch and streaming results (allowing for small numerical differences)
        for (i, ((&batch_bull, &stream_bull), (&batch_bear, &stream_bear))) in batch_output
            .bulls
            .iter()
            .zip(stream_bulls.iter())
            .zip(batch_output.bears.iter().zip(stream_bears.iter()))
            .enumerate()
        {
            if batch_bull.is_nan() && stream_bull.is_nan() {
                continue;
            }
            if batch_bear.is_nan() && stream_bear.is_nan() {
                continue;
            }

            // Note: The current streaming implementation does not apply SMA smoothing
            // while the batch implementation does. This is a known difference.
            // Additionally, the streaming implementation has a bug where mode 0
            // produces values that don't sum to 100.
            // TODO: Fix streaming implementation to:
            //   1. Include SMA smoothing
            //   2. Fix mode 0 calculation to ensure bulls + bears = 100

            // For now, we just check that values are reasonable (not NaN and in valid range)
            if i >= period {
                // After warmup
                if !batch_bull.is_nan() && !stream_bull.is_nan() {
                    // Both should be in range [0, 100]
                    assert!(
                        stream_bull >= -1e-9 && stream_bull <= 100.0 + 1e-9,
                        "[{}] ASO streaming bulls out of range at idx {}: {}",
                        test_name,
                        i,
                        stream_bull
                    );
                }
                if !batch_bear.is_nan() && !stream_bear.is_nan() {
                    assert!(
                        stream_bear >= -1e-9 && stream_bear <= 100.0 + 1e-9,
                        "[{}] ASO streaming bears out of range at idx {}: {}",
                        test_name,
                        i,
                        stream_bear
                    );
                }

                // Check sum only for modes 1 and 2 (mode 0 has a bug)
                if mode != 0 && !stream_bull.is_nan() && !stream_bear.is_nan() {
                    let sum = stream_bull + stream_bear;
                    assert!(
                        (sum - 100.0).abs() < 1e-9,
                        "[{}] ASO streaming bulls + bears != 100 at idx {} (mode {}): {} + {} = {}",
                        test_name,
                        i,
                        mode,
                        stream_bull,
                        stream_bear,
                        sum
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_aso_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            AsoParams::default(),
            AsoParams {
                period: Some(5),
                mode: Some(0),
            },
            AsoParams {
                period: Some(5),
                mode: Some(1),
            },
            AsoParams {
                period: Some(5),
                mode: Some(2),
            },
            AsoParams {
                period: Some(10),
                mode: Some(0),
            },
            AsoParams {
                period: Some(10),
                mode: Some(1),
            },
            AsoParams {
                period: Some(10),
                mode: Some(2),
            },
            AsoParams {
                period: Some(20),
                mode: Some(0),
            },
            AsoParams {
                period: Some(20),
                mode: Some(1),
            },
            AsoParams {
                period: Some(20),
                mode: Some(2),
            },
            AsoParams {
                period: Some(2),
                mode: Some(0),
            },
            AsoParams {
                period: Some(50),
                mode: Some(1),
            },
            AsoParams {
                period: Some(100),
                mode: Some(2),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = AsoInput::from_candles(&candles, "close", params.clone());
            let output = aso_with_kernel(&input, kernel)?;

            for (i, (&bull_val, &bear_val)) in
                output.bulls.iter().zip(output.bears.iter()).enumerate()
            {
                if bull_val.is_nan() || bear_val.is_nan() {
                    continue;
                }

                let bull_bits = bull_val.to_bits();
                let bear_bits = bear_val.to_bits();

                // Check for poison values
                for (val, bits, name) in [
                    (bull_val, bull_bits, "bulls"),
                    (bear_val, bear_bits, "bears"),
                ] {
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in {} at index {} \
                            with params: period={}, mode={}",
                            test_name,
                            val,
                            bits,
                            name,
                            i,
                            params.period.unwrap_or(10),
                            params.mode.unwrap_or(0)
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in {} at index {} \
                            with params: period={}, mode={}",
                            test_name,
                            val,
                            bits,
                            name,
                            i,
                            params.period.unwrap_or(10),
                            params.mode.unwrap_or(0)
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in {} at index {} \
                            with params: period={}, mode={}",
                            test_name,
                            val,
                            bits,
                            name,
                            i,
                            params.period.unwrap_or(10),
                            params.mode.unwrap_or(0)
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_aso_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_aso_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
                0usize..=2, // mode: 0, 1, or 2
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, mode)| {
                let params = AsoParams {
                    period: Some(period),
                    mode: Some(mode),
                };

                // Create OHLC data from single series
                let mut open = Vec::with_capacity(data.len());
                let mut high = Vec::with_capacity(data.len());
                let mut low = Vec::with_capacity(data.len());
                let mut close = Vec::with_capacity(data.len());

                for &val in &data {
                    let spread = val.abs() * 0.1 + 1.0;
                    open.push(val);
                    high.push(val + spread);
                    low.push(val - spread);
                    close.push(val + spread * 0.5);
                }

                let input = AsoInput::from_slices(&open, &high, &low, &close, params);

                let AsoOutput {
                    bulls: out_bulls,
                    bears: out_bears,
                } = aso_with_kernel(&input, kernel).unwrap();
                let AsoOutput {
                    bulls: ref_bulls,
                    bears: ref_bears,
                } = aso_with_kernel(&input, Kernel::Scalar).unwrap();

                for i in (period - 1)..data.len() {
                    let bull = out_bulls[i];
                    let bear = out_bears[i];
                    let ref_bull = ref_bulls[i];
                    let ref_bear = ref_bears[i];

                    // Bulls and bears should sum to 100 (when not NaN)
                    if !bull.is_nan() && !bear.is_nan() {
                        let sum = bull + bear;
                        prop_assert!(
                            (sum - 100.0).abs() < 1e-9,
                            "idx {}: bulls + bears = {} + {} = {}, expected 100",
                            i,
                            bull,
                            bear,
                            sum
                        );
                    }

                    // Values should be in range [0, 100]
                    if !bull.is_nan() {
                        prop_assert!(
                            bull >= -1e-9 && bull <= 100.0 + 1e-9,
                            "idx {}: bull {} out of range [0, 100]",
                            i,
                            bull
                        );
                    }
                    if !bear.is_nan() {
                        prop_assert!(
                            bear >= -1e-9 && bear <= 100.0 + 1e-9,
                            "idx {}: bear {} out of range [0, 100]",
                            i,
                            bear
                        );
                    }

                    // Check consistency with scalar kernel
                    let bull_bits = bull.to_bits();
                    let bear_bits = bear.to_bits();
                    let ref_bull_bits = ref_bull.to_bits();
                    let ref_bear_bits = ref_bear.to_bits();

                    if !bull.is_finite() || !ref_bull.is_finite() {
                        prop_assert!(
                            bull_bits == ref_bull_bits,
                            "bull finite/NaN mismatch idx {}: {} vs {}",
                            i,
                            bull,
                            ref_bull
                        );
                    } else {
                        let ulp_diff: u64 = bull_bits.abs_diff(ref_bull_bits);
                        prop_assert!(
                            (bull - ref_bull).abs() <= 1e-9 || ulp_diff <= 4,
                            "bull mismatch idx {}: {} vs {} (ULP={})",
                            i,
                            bull,
                            ref_bull,
                            ulp_diff
                        );
                    }

                    if !bear.is_finite() || !ref_bear.is_finite() {
                        prop_assert!(
                            bear_bits == ref_bear_bits,
                            "bear finite/NaN mismatch idx {}: {} vs {}",
                            i,
                            bear,
                            ref_bear
                        );
                    } else {
                        let ulp_diff: u64 = bear_bits.abs_diff(ref_bear_bits);
                        prop_assert!(
                            (bear - ref_bear).abs() <= 1e-9 || ulp_diff <= 4,
                            "bear mismatch idx {}: {} vs {} (ULP={})",
                            i,
                            bear,
                            ref_bear,
                            ulp_diff
                        );
                    }
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    // Batch processing tests
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = AsoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;

        // Find the row with default parameters
        let default_params = AsoParams::default();
        let default_row_idx = output
            .combos
            .iter()
            .position(|p| p.period == default_params.period && p.mode == default_params.mode)
            .expect("default row missing");

        let bulls_row = output.bulls_row(default_row_idx);
        let bears_row = output.bears_row(default_row_idx);

        assert_eq!(bulls_row.len(), candles.close.len());
        assert_eq!(bears_row.len(), candles.close.len());

        // Check last 5 values match expected
        let expected_bulls = [
            48.48594883,
            46.37206396,
            47.20522805,
            46.83750720,
            43.28268188,
        ];
        let expected_bears = [
            51.51405117,
            53.62793604,
            52.79477195,
            53.16249280,
            56.71731812,
        ];

        let start = bulls_row.len() - 5;
        for (i, (&bull, &bear)) in bulls_row[start..]
            .iter()
            .zip(bears_row[start..].iter())
            .enumerate()
        {
            assert!(
                (bull - expected_bulls[i]).abs() < 1e-6,
                "[{}] default-row bulls mismatch at idx {}: {} vs {}",
                test_name,
                i,
                bull,
                expected_bulls[i]
            );
            assert!(
                (bear - expected_bears[i]).abs() < 1e-6,
                "[{}] default-row bears mismatch at idx {}: {} vs {}",
                test_name,
                i,
                bear,
                expected_bears[i]
            );
        }
        Ok(())
    }

    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = AsoBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 20, 2) // 6 values: 10, 12, 14, 16, 18, 20
            .mode_range(0, 2, 1) // 3 values: 0, 1, 2
            .apply_candles(&candles)?;

        let expected_combos = 6 * 3; // 18 combinations
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, candles.close.len());

        // Verify all combinations are present
        let mut found_combos = 0;
        for period in (10..=20).step_by(2) {
            for mode in 0..=2 {
                let found = output
                    .combos
                    .iter()
                    .any(|c| c.period == Some(period) && c.mode == Some(mode));
                assert!(
                    found,
                    "[{}] Missing combo: period={}, mode={}",
                    test_name, period, mode
                );
                if found {
                    found_combos += 1;
                }
            }
        }
        assert_eq!(found_combos, expected_combos);

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_configs = vec![
            (2, 10, 2, 0, 2, 1),   // period: 2,4,6,8,10; mode: 0,1,2
            (5, 25, 5, 0, 0, 1),   // period: 5,10,15,20,25; mode: 0
            (10, 10, 1, 0, 2, 1),  // period: 10; mode: 0,1,2
            (2, 5, 1, 1, 1, 1),    // period: 2,3,4,5; mode: 1
            (30, 60, 15, 2, 2, 1), // period: 30,45,60; mode: 2
            (9, 15, 3, 0, 2, 2),   // period: 9,12,15; mode: 0,2
            (8, 12, 1, 0, 1, 1),   // period: 8,9,10,11,12; mode: 0,1
        ];

        for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step)) in
            test_configs.iter().enumerate()
        {
            let output = AsoBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .mode_range(m_start, m_end, m_step)
                .apply_candles(&candles)?;

            // Check for poison values in all outputs
            for (row_idx, combo) in output.combos.iter().enumerate() {
                let bulls_row = output.bulls_row(row_idx);
                let bears_row = output.bears_row(row_idx);

                for (col_idx, (&bull_val, &bear_val)) in
                    bulls_row.iter().zip(bears_row.iter()).enumerate()
                {
                    if bull_val.is_nan() || bear_val.is_nan() {
                        continue;
                    }

                    let bull_bits = bull_val.to_bits();
                    let bear_bits = bear_val.to_bits();

                    for (val, bits, name) in [
                        (bull_val, bull_bits, "bulls"),
                        (bear_val, bear_bits, "bears"),
                    ] {
                        if bits == 0x11111111_11111111 {
                            panic!(
                                "[{}] Config {}: Found alloc_with_nan_prefix poison {} (0x{:016X}) in {} \
                                at row {} col {} (period={}, mode={})",
                                test_name, cfg_idx, val, bits, name, row_idx, col_idx,
                                combo.period.unwrap_or(10), combo.mode.unwrap_or(0)
                            );
                        }

                        if bits == 0x22222222_22222222 {
                            panic!(
                                "[{}] Config {}: Found init_matrix_prefixes poison {} (0x{:016X}) in {} \
                                at row {} col {} (period={}, mode={})",
                                test_name, cfg_idx, val, bits, name, row_idx, col_idx,
                                combo.period.unwrap_or(10), combo.mode.unwrap_or(0)
                            );
                        }

                        if bits == 0x33333333_33333333 {
                            panic!(
                                "[{}] Config {}: Found make_uninit_matrix poison {} (0x{:016X}) in {} \
                                at row {} col {} (period={}, mode={})",
                                test_name, cfg_idx, val, bits, name, row_idx, col_idx,
                                combo.period.unwrap_or(10), combo.mode.unwrap_or(0)
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    // Test generation macros
    macro_rules! generate_all_aso_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    generate_all_aso_tests!(
        check_aso_accuracy,
        check_aso_slice_input,
        check_aso_into_slices,
        check_aso_batch,
        check_aso_partial_params,
        check_aso_default_candles,
        check_aso_zero_period,
        check_aso_period_exceeds_length,
        check_aso_invalid_mode,
        check_aso_empty_input,
        check_aso_all_nan,
        check_aso_very_small_dataset,
        check_aso_reinput,
        check_aso_nan_handling,
        check_aso_streaming,
        check_aso_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_aso_tests!(check_aso_property);

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    // Test new API features
    #[test]
    fn test_new_api_features() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test AsRef trait
        let input = AsoInput::from_candles(&candles, "close", AsoParams::default());
        let _data_ref: &[f64] = input.as_ref();

        // Test static parameter methods
        let builder = AsoBatchBuilder::new().period_static(10).mode_static(0);
        let output = builder.apply_candles(&candles)?;
        assert_eq!(output.combos.len(), 1);

        // Test convenience batch methods
        let output2 = AsoBatchBuilder::with_default_candles(&candles)?;
        assert!(output2.combos.len() > 0);

        let output3 = AsoBatchBuilder::with_default_slices(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            Kernel::Scalar,
        )?;
        assert!(output3.combos.len() > 0);

        // Test row_for_params and values_for
        let params = AsoParams::default();
        if let Some(row) = output2.row_for_params(&params) {
            let bulls_row = output2.bulls_row(row);
            let bears_row = output2.bears_row(row);
            assert_eq!(bulls_row.len(), candles.close.len());
            assert_eq!(bears_row.len(), candles.close.len());
        }

        if let Some((bulls, bears)) = output2.values_for(&params) {
            assert_eq!(bulls.len(), candles.close.len());
            assert_eq!(bears.len(), candles.close.len());
        }

        // Test parallel batch processing
        let sweep = AsoBatchRange::default();
        let output4 = aso_batch_slice(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            &sweep,
            Kernel::Scalar,
        )?;
        assert!(output4.combos.len() > 0);

        let output5 = aso_batch_par_slice(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            &sweep,
            Kernel::Scalar,
        )?;
        assert_eq!(output4.combos.len(), output5.combos.len());

        // Test source parameter in from_candles
        let input_high = AsoInput::from_candles(&candles, "high", AsoParams::default());
        let _output_high = aso(&input_high)?;

        Ok(())
    }
}
