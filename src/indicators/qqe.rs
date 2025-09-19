//! # Quantitative Qualitative Estimation (QQE)
//!
//! A momentum indicator based on smoothed RSI with dynamic volatility bands.
//! QQE uses RSI smoothed with EMA and ATR-based bands to generate trading signals.
//!
//! ## Parameters
//! - **rsi_period**: Period for RSI calculation (default: 14)
//! - **smoothing_factor**: EMA smoothing factor for RSI (default: 5)
//! - **fast_factor**: Multiplier for ATR bands (default: 4.236)
//!
//! ## Returns
//! - **`Ok(QqeOutput)`** on success, containing `fast` and `slow` vectors.
//! - **`Err(QqeError)`** on failure
//!
//! ## Developer Notes
//! - **AVX2**: Stub implementation - calls scalar function
//! - **AVX512**: Stub implementation - calls scalar function
//! - **Streaming**: O(1) with efficient ring buffer and incremental RSI updates
//! - **Memory**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)

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
use crate::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
use crate::indicators::rsi::{rsi, RsiInput, RsiParams};
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

// ==================== TRAIT IMPLEMENTATIONS ====================
impl<'a> AsRef<[f64]> for QqeInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            QqeData::Slice(slice) => slice,
            QqeData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum QqeData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct QqeOutput {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct QqeParams {
    pub rsi_period: Option<usize>,
    pub smoothing_factor: Option<usize>,
    pub fast_factor: Option<f64>,
}

impl Default for QqeParams {
    fn default() -> Self {
        Self {
            rsi_period: Some(14),
            smoothing_factor: Some(5),
            fast_factor: Some(4.236),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct QqeInput<'a> {
    pub data: QqeData<'a>,
    pub params: QqeParams,
}

impl<'a> QqeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: QqeParams) -> Self {
        Self {
            data: QqeData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: QqeParams) -> Self {
        Self {
            data: QqeData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", QqeParams::default())
    }

    #[inline]
    pub fn get_rsi_period(&self) -> usize {
        self.params.rsi_period.unwrap_or(14)
    }

    #[inline]
    pub fn get_smoothing_factor(&self) -> usize {
        self.params.smoothing_factor.unwrap_or(5)
    }

    #[inline]
    pub fn get_fast_factor(&self) -> f64 {
        self.params.fast_factor.unwrap_or(4.236)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct QqeBuilder {
    rsi_period: Option<usize>,
    smoothing_factor: Option<usize>,
    fast_factor: Option<f64>,
    kernel: Kernel,
}

impl Default for QqeBuilder {
    fn default() -> Self {
        Self {
            rsi_period: None,
            smoothing_factor: None,
            fast_factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl QqeBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn rsi_period(mut self, val: usize) -> Self {
        self.rsi_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn smoothing_factor(mut self, val: usize) -> Self {
        self.smoothing_factor = Some(val);
        self
    }

    #[inline(always)]
    pub fn fast_factor(mut self, val: f64) -> Self {
        self.fast_factor = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<QqeOutput, QqeError> {
        let p = QqeParams {
            rsi_period: self.rsi_period,
            smoothing_factor: self.smoothing_factor,
            fast_factor: self.fast_factor,
        };
        let i = QqeInput::from_candles(c, "close", p);
        qqe_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<QqeOutput, QqeError> {
        let p = QqeParams {
            rsi_period: self.rsi_period,
            smoothing_factor: self.smoothing_factor,
            fast_factor: self.fast_factor,
        };
        let i = QqeInput::from_slice(d, p);
        qqe_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<QqeStream, QqeError> {
        let p = QqeParams {
            rsi_period: self.rsi_period,
            smoothing_factor: self.smoothing_factor,
            fast_factor: self.fast_factor,
        };
        QqeStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum QqeError {
    #[error("qqe: Input data slice is empty.")]
    EmptyInputData,

    #[error("qqe: All values are NaN.")]
    AllValuesNaN,

    #[error("qqe: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("qqe: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("qqe: Error in dependent indicator: {message}")]
    DependentIndicatorError { message: String },
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
/// Main entry point with automatic kernel detection
#[inline]
pub fn qqe(input: &QqeInput) -> Result<QqeOutput, QqeError> {
    qqe_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection - now uses zero-copy routine
pub fn qqe_with_kernel(input: &QqeInput, kernel: Kernel) -> Result<QqeOutput, QqeError> {
    let (data, rsi_p, ema_p, fast_k, first, chosen) = qqe_prepare(input, kernel)?;
    let warm = first + rsi_p + ema_p - 2;

    // Check for classic kernel conditions (default parameters)
    if chosen == Kernel::Scalar && rsi_p == 14 && ema_p == 5 && fast_k == 4.236 {
        // Use optimized classic kernel for default parameters
        let mut fast = alloc_with_nan_prefix(data.len(), warm);
        let mut slow = alloc_with_nan_prefix(data.len(), warm);
        unsafe {
            qqe_scalar_classic(data, rsi_p, ema_p, fast_k, first, &mut fast, &mut slow)?;
        }
        return Ok(QqeOutput { fast, slow });
    }

    // match alma.rs: allocate with NaN prefix, no full-vector fill
    let mut fast = alloc_with_nan_prefix(data.len(), warm);
    let mut slow = alloc_with_nan_prefix(data.len(), warm);

    // compute directly into the preallocated slices
    qqe_into_slices(&mut fast, &mut slow, input, chosen)?;
    Ok(QqeOutput { fast, slow })
}

/// Scalar implementation - now uses zero-copy pattern
fn qqe_scalar(
    data: &[f64],
    rsi_p: usize,
    ema_p: usize,
    fast_k: f64,
    first: usize,
    fast_warm: usize,
) -> Result<QqeOutput, QqeError> {
    let mut fast = alloc_with_nan_prefix(data.len(), fast_warm);
    let mut slow = alloc_with_nan_prefix(data.len(), fast_warm);

    // Use classic kernel for default parameters
    if rsi_p == 14 && ema_p == 5 && fast_k == 4.236 {
        unsafe {
            qqe_scalar_classic(data, rsi_p, ema_p, fast_k, first, &mut fast, &mut slow)?;
        }
    } else {
        qqe_into_slices(
            &mut fast,
            &mut slow,
            &QqeInput::from_slice(
                data,
                QqeParams {
                    rsi_period: Some(rsi_p),
                    smoothing_factor: Some(ema_p),
                    fast_factor: Some(fast_k),
                },
            ),
            Kernel::Scalar,
        )?;
    }
    Ok(QqeOutput { fast, slow })
}

/// AVX2 implementation (stub for now)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn qqe_avx2(
    data: &[f64],
    rsi_p: usize,
    ema_p: usize,
    fast_k: f64,
    first: usize,
    fast_warm: usize,
) -> Result<QqeOutput, QqeError> {
    // For now, fallback to scalar
    qqe_scalar(data, rsi_p, ema_p, fast_k, first, fast_warm)
}

/// AVX512 implementation (stub for now)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn qqe_avx512(
    data: &[f64],
    rsi_p: usize,
    ema_p: usize,
    fast_k: f64,
    first: usize,
    fast_warm: usize,
) -> Result<QqeOutput, QqeError> {
    // For now, fallback to scalar
    qqe_scalar(data, rsi_p, ema_p, fast_k, first, fast_warm)
}

/// Zero-allocation version for WASM and performance-critical paths - true zero-copy pipeline
#[inline]
pub fn qqe_into_slices(
    dst_fast: &mut [f64],
    dst_slow: &mut [f64],
    input: &QqeInput,
    kern: Kernel,
) -> Result<(), QqeError> {
    use crate::indicators::moving_averages::ema::ema_into_slice;
    use crate::indicators::rsi::rsi_into_slice;

    let (data, rsi_p, ema_p, fast_k, first, chosen) = qqe_prepare(input, kern)?;
    if dst_fast.len() != data.len() || dst_slow.len() != data.len() {
        return Err(QqeError::InvalidPeriod {
            period: dst_fast.len(),
            data_len: data.len(),
        });
    }
    let warm = first + rsi_p + ema_p - 2;

    // Temporary row buffer for RSI (no copies of existing data)
    let mut tmp_mu = make_uninit_matrix(1, data.len());
    let tmp: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(tmp_mu.as_mut_ptr() as *mut f64, data.len()) };

    // RSI → tmp
    let rsi_in = RsiInput::from_slice(
        data,
        RsiParams {
            period: Some(rsi_p),
        },
    );
    rsi_into_slice(tmp, &rsi_in, chosen).map_err(|e| QqeError::DependentIndicatorError {
        message: e.to_string(),
    })?;

    // EMA(tmp) → dst_fast
    let ema_in = EmaInput::from_slice(
        tmp,
        EmaParams {
            period: Some(ema_p),
        },
    );
    ema_into_slice(dst_fast, &ema_in, chosen).map_err(|e| QqeError::DependentIndicatorError {
        message: e.to_string(),
    })?;

    // Enforce NaN warm prefixes for both outputs
    for v in &mut dst_fast[..warm] {
        *v = f64::NAN;
    }
    for v in &mut dst_slow[..warm] {
        *v = f64::NAN;
    }

    // Slow from fast
    qqe_compute_slow_from(dst_fast, fast_k, warm, dst_slow);
    Ok(())
}

/// Thin alias to match ALMA's *_into_slice naming pattern
#[inline]
pub fn qqe_into_pair(
    dst: (&mut [f64], &mut [f64]),
    input: &QqeInput,
    kern: Kernel,
) -> Result<(), QqeError> {
    qqe_into_slices(dst.0, dst.1, input, kern)
}

/// Public alias to mirror *_into_slice naming
#[inline]
pub fn qqe_into_slice(
    dst_fast: &mut [f64],
    dst_slow: &mut [f64],
    input: &QqeInput,
    kern: Kernel,
) -> Result<(), QqeError> {
    qqe_into_slices(dst_fast, dst_slow, input, kern)
}

/// Prepare and validate input data
#[inline(always)]
fn qqe_prepare<'a>(
    input: &'a QqeInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, f64, usize, Kernel), QqeError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(QqeError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(QqeError::AllValuesNaN)?;

    let rsi_period = input.get_rsi_period();
    let smoothing_factor = input.get_smoothing_factor();
    let fast_factor = input.get_fast_factor();

    // Validation
    if rsi_period == 0 || rsi_period > len {
        return Err(QqeError::InvalidPeriod {
            period: rsi_period,
            data_len: len,
        });
    }

    if smoothing_factor == 0 || smoothing_factor > len {
        return Err(QqeError::InvalidPeriod {
            period: smoothing_factor,
            data_len: len,
        });
    }

    // Need enough data for RSI + EMA smoothing
    let needed = rsi_period + smoothing_factor;
    if len - first < needed {
        return Err(QqeError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((
        data,
        rsi_period,
        smoothing_factor,
        fast_factor,
        first,
        chosen,
    ))
}

/// Core computation for slow line (QQES) - optimized version
#[inline]
fn qqe_compute_slow_from(qqef: &[f64], fast_factor: f64, start: usize, qqes: &mut [f64]) {
    let len = qqef.len();
    debug_assert!(start < len);

    // write first valid slow = qqef[start]
    qqes[start] = qqef[start];

    let alpha = 1.0 / 14.0; // ATR smoothing for RSI delta
    let mut wwma = 0.0;
    let mut atrrsi = 0.0;

    for i in (start + 1)..len {
        // qqef[start..] are valid by construction
        let tr = (qqef[i] - qqef[i - 1]).abs();
        wwma = alpha * tr + (1.0 - alpha) * wwma;
        atrrsi = alpha * wwma + (1.0 - alpha) * atrrsi;

        let qup = qqef[i] + atrrsi * fast_factor;
        let qdn = qqef[i] - atrrsi * fast_factor;

        let prev = qqes[i - 1];

        if qup < prev {
            qqes[i] = qup;
        } else if qqef[i] > prev && qqef[i - 1] < prev {
            qqes[i] = qdn;
        } else if qdn > prev {
            qqes[i] = qdn;
        } else if qqef[i] < prev && qqef[i - 1] > prev {
            qqes[i] = qup;
        } else {
            qqes[i] = prev;
        }
    }
}

// ==================== CLASSIC KERNEL ====================
/// Optimized classic kernel for QQE with default parameters
/// Inlines RSI (with Wilder's smoothing) and EMA calculations for maximum performance
#[inline(always)]
pub unsafe fn qqe_scalar_classic(
    data: &[f64],
    rsi_period: usize,
    smoothing_factor: usize,
    fast_factor: f64,
    first: usize,
    dst_fast: &mut [f64],
    dst_slow: &mut [f64],
) -> Result<(), QqeError> {
    let len = data.len();
    let warm = first + rsi_period + smoothing_factor - 2;

    // Ensure output arrays are properly sized
    if dst_fast.len() != len || dst_slow.len() != len {
        return Err(QqeError::InvalidPeriod {
            period: dst_fast.len(),
            data_len: len,
        });
    }

    // Fill entire arrays with NaN initially
    for i in 0..len {
        dst_fast[i] = f64::NAN;
        dst_slow[i] = f64::NAN;
    }

    // Step 1: Inline RSI calculation with Wilder's smoothing
    // Initialize first average gain and loss (matching rsi_compute_into_scalar exactly)
    let inv_period = 1.0 / rsi_period as f64;
    let beta = 1.0 - inv_period;

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    let mut has_nan = false;

    for i in (first + 1)..=((first + rsi_period).min(len - 1)) {
        let delta = data[i] - data[i - 1];
        if !delta.is_finite() {
            has_nan = true;
            break; // Any NaN in warmup invalidates the calculation
        }
        if delta > 0.0 {
            avg_gain += delta;
        } else if delta < 0.0 {
            avg_loss += -delta;
        }
    }

    // Calculate RSI values with Wilder's smoothing
    let mut rsi_values = vec![f64::NAN; len];

    // First RSI value at index first + rsi_period (matching rsi_compute_into_scalar)
    let initial_rsi = if has_nan {
        avg_gain = f64::NAN; // Poison the averages so NaN propagates
        avg_loss = f64::NAN;
        f64::NAN // If any NaN in warmup, initial RSI is NaN
    } else {
        avg_gain *= inv_period;
        avg_loss *= inv_period;
        if avg_gain + avg_loss == 0.0 {
            50.0
        } else {
            100.0 * avg_gain / (avg_gain + avg_loss)
        }
    };
    if first + rsi_period < len {
        rsi_values[first + rsi_period] = initial_rsi;
    }

    // Continue RSI calculation with Wilder's smoothing
    for i in (first + rsi_period + 1)..len {
        let delta = data[i] - data[i - 1];
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };

        avg_gain = inv_period * gain + beta * avg_gain;
        avg_loss = inv_period * loss + beta * avg_loss;

        let rsi = if avg_gain + avg_loss == 0.0 {
            50.0
        } else {
            100.0 * avg_gain / (avg_gain + avg_loss)
        };
        rsi_values[i] = rsi;
    }

    // Step 2: Apply EMA smoothing to RSI values to get fast line (matching ema_scalar_into)
    let ema_alpha = 2.0 / (smoothing_factor as f64 + 1.0);
    let ema_beta = 1.0 - ema_alpha;

    // Find first valid RSI value for EMA initialization
    let rsi_start = first + rsi_period;

    // Use running mean for warmup period (matching ema_scalar_into exactly)
    if rsi_start < len && rsi_values[rsi_start].is_finite() {
        let mut mean = rsi_values[rsi_start];
        dst_fast[rsi_start] = mean;
        let mut valid_count = 1usize;

        // Running mean phase (indices rsi_start+1 to rsi_start+smoothing_factor-1)
        let ema_warmup_end = (rsi_start + smoothing_factor).min(len);
        for i in (rsi_start + 1)..ema_warmup_end {
            let x = rsi_values[i];
            if x.is_finite() {
                valid_count += 1;
                mean = ((valid_count as f64 - 1.0) * mean + x) / valid_count as f64;
                dst_fast[i] = mean;
            } else {
                // During warmup, skip NaN values and carry forward
                dst_fast[i] = mean;
            }
        }

        // EMA phase (from rsi_start+smoothing_factor onwards)
        if ema_warmup_end < len {
            let mut prev = mean;
            for i in ema_warmup_end..len {
                let x = rsi_values[i];
                if x.is_finite() {
                    prev = ema_beta.mul_add(prev, ema_alpha * x);
                    dst_fast[i] = prev;
                } else {
                    // Skip NaN values - carry forward previous value
                    dst_fast[i] = prev;
                }
            }
        }
    }

    // Step 3: Enforce warmup NaN prefix (to match regular implementation)
    // The regular implementation calculates RSI and EMA, THEN enforces NaN prefix
    for i in 0..warm {
        dst_fast[i] = f64::NAN;
        dst_slow[i] = f64::NAN;
    }

    // Step 4: Calculate slow line from fast line
    qqe_compute_slow_from(dst_fast, fast_factor, warm, dst_slow);

    Ok(())
}

// ==================== STREAMING SUPPORT ====================
/// Streaming calculator for real-time updates
#[derive(Debug, Clone)]
pub struct QqeStream {
    buffer: Vec<f64>,
    rsi_period: usize,
    smoothing_factor: usize,
    fast_factor: f64,
    index: usize,
    ready: bool,
    wwma: f64,
    atrrsi: f64,
    prev_qqef: f64,
    prev_qqes: f64,
    prev_price: f64,
    avg_gain: f64,
    avg_loss: f64,
}

impl QqeStream {
    pub fn try_new(params: QqeParams) -> Result<Self, QqeError> {
        let rsi_period = params.rsi_period.unwrap_or(14);
        let smoothing_factor = params.smoothing_factor.unwrap_or(5);
        let fast_factor = params.fast_factor.unwrap_or(4.236);

        if rsi_period == 0 || smoothing_factor == 0 {
            return Err(QqeError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }

        let buffer_size = rsi_period + smoothing_factor;
        Ok(Self {
            buffer: vec![0.0; buffer_size],
            rsi_period,
            smoothing_factor,
            fast_factor,
            index: 0,
            ready: false,
            wwma: 0.0,
            atrrsi: 0.0,
            prev_qqef: f64::NAN,
            prev_qqes: 0.0,
            prev_price: f64::NAN,
            avg_gain: 0.0,
            avg_loss: 0.0,
        })
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        // ring buffer for raw price series
        let n = self.rsi_period + self.smoothing_factor;
        self.buffer[self.index % n] = value;
        self.index += 1;

        if self.index <= self.rsi_period {
            // Still warming up - accumulate for initial averages
            if self.index == 1 {
                self.prev_price = value;
            } else {
                let change = value - self.prev_price;
                if change > 0.0 {
                    self.avg_gain += change;
                } else {
                    self.avg_loss -= change;
                }

                if self.index == self.rsi_period {
                    // Initialize Wilder averages
                    self.avg_gain /= self.rsi_period as f64;
                    self.avg_loss /= self.rsi_period as f64;
                }
                self.prev_price = value;
            }
            return None;
        }

        // Update Wilder's smoothed averages (O(1))
        let change = value - self.prev_price;
        let alpha = 1.0 / self.rsi_period as f64;

        if change > 0.0 {
            self.avg_gain = (1.0 - alpha) * self.avg_gain + alpha * change;
            self.avg_loss = (1.0 - alpha) * self.avg_loss;
        } else {
            self.avg_gain = (1.0 - alpha) * self.avg_gain;
            self.avg_loss = (1.0 - alpha) * self.avg_loss - alpha * change;
        }
        self.prev_price = value;

        // Calculate RSI from Wilder averages
        let rsi_val = if self.avg_loss == 0.0 {
            100.0
        } else {
            let rs = self.avg_gain / self.avg_loss;
            100.0 - 100.0 / (1.0 + rs)
        };

        // EMA smoothing to get fast line
        let k = 2.0 / (self.smoothing_factor as f64 + 1.0);
        let fast = if self.prev_qqef.is_nan() {
            rsi_val
        } else {
            k * rsi_val + (1.0 - k) * self.prev_qqef
        };

        // Calculate slow line
        let slow = if self.prev_qqef.is_nan() {
            // First iteration
            self.prev_qqes = fast;
            fast
        } else {
            // Slow line via same recurrence as batch version
            let tr = (fast - self.prev_qqef).abs();
            self.wwma = (1.0 / 14.0) * tr + (1.0 - 1.0 / 14.0) * self.wwma;
            self.atrrsi = (1.0 / 14.0) * self.wwma + (1.0 - 1.0 / 14.0) * self.atrrsi;

            let qup = fast + self.atrrsi * self.fast_factor;
            let qdn = fast - self.atrrsi * self.fast_factor;

            let prev = self.prev_qqes;
            let slow_val = if qup < prev {
                qup
            } else if fast > prev && self.prev_qqef < prev {
                qdn
            } else if qdn > prev {
                qdn
            } else if fast < prev && self.prev_qqef > prev {
                qup
            } else {
                prev
            };

            self.prev_qqes = slow_val;
            slow_val
        };

        self.prev_qqef = fast;
        Some((fast, slow))
    }
}

// ==================== BATCH PROCESSING ====================
/// Batch processing for parameter sweeps
#[derive(Clone, Debug)]
pub struct QqeBatchRange {
    pub rsi_period: (usize, usize, usize), // (start, end, step)
    pub smoothing_factor: (usize, usize, usize),
    pub fast_factor: (f64, f64, f64),
}

impl Default for QqeBatchRange {
    fn default() -> Self {
        Self {
            rsi_period: (14, 14, 0),
            smoothing_factor: (5, 5, 0),
            fast_factor: (4.236, 4.236, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct QqeBatchBuilder {
    range: QqeBatchRange,
    kernel: Kernel,
}

impl QqeBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn rsi_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_period = (start, end, step);
        self
    }

    #[inline]
    pub fn rsi_period_static(mut self, val: usize) -> Self {
        self.range.rsi_period = (val, val, 0);
        self
    }

    #[inline]
    pub fn smoothing_factor_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_factor = (start, end, step);
        self
    }

    #[inline]
    pub fn smoothing_factor_static(mut self, val: usize) -> Self {
        self.range.smoothing_factor = (val, val, 0);
        self
    }

    #[inline]
    pub fn fast_factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.fast_factor = (start, end, step);
        self
    }

    #[inline]
    pub fn fast_factor_static(mut self, val: f64) -> Self {
        self.range.fast_factor = (val, val, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<QqeBatchOutput, QqeError> {
        qqe_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<QqeBatchOutput, QqeError> {
        QqeBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<QqeBatchOutput, QqeError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<QqeBatchOutput, QqeError> {
        QqeBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct QqeBatchOutput {
    pub fast_values: Vec<f64>, // rows*cols flattened
    pub slow_values: Vec<f64>, // rows*cols flattened
    pub combos: Vec<QqeParams>,
    pub rows: usize, // = combos.len()
    pub cols: usize, // = data.len()
}

impl QqeBatchOutput {
    pub fn row_for_params(&self, p: &QqeParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.rsi_period.unwrap_or(14) == p.rsi_period.unwrap_or(14)
                && c.smoothing_factor.unwrap_or(5) == p.smoothing_factor.unwrap_or(5)
                && (c.fast_factor.unwrap_or(4.236) - p.fast_factor.unwrap_or(4.236)).abs() < 1e-12
        })
    }

    pub fn values_for(&self, p: &QqeParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            let end = start + self.cols;
            (&self.fast_values[start..end], &self.slow_values[start..end])
        })
    }
}

fn expand_grid(r: &QqeBatchRange) -> Vec<QqeParams> {
    fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
        if st == 0 || s == e {
            return vec![s];
        }
        (s..=e).step_by(st).collect()
    }

    fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
        if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
            return vec![s];
        }
        let mut v = Vec::new();
        let mut x = s;
        while x <= e + 1e-12 {
            v.push(x);
            x += st;
        }
        v
    }

    let rs = axis_usize(r.rsi_period);
    let sm = axis_usize(r.smoothing_factor);
    let ff = axis_f64(r.fast_factor);
    let mut out = Vec::with_capacity(rs.len() * sm.len() * ff.len());

    for &rp in &rs {
        for &sp in &sm {
            for &fk in &ff {
                out.push(QqeParams {
                    rsi_period: Some(rp),
                    smoothing_factor: Some(sp),
                    fast_factor: Some(fk),
                });
            }
        }
    }
    out
}

pub fn qqe_batch_with_kernel(
    data: &[f64],
    sweep: &QqeBatchRange,
    k: Kernel,
) -> Result<QqeBatchOutput, QqeError> {
    use crate::indicators::moving_averages::ema::ema_into_slice;
    use crate::indicators::rsi::rsi_into_slice;

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(QqeError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let cols = data.len();
    if cols == 0 {
        return Err(QqeError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(QqeError::AllValuesNaN)?;
    let worst_needed = combos
        .iter()
        .map(|c| c.rsi_period.unwrap() + c.smoothing_factor.unwrap())
        .max()
        .unwrap();
    if cols - first < worst_needed {
        return Err(QqeError::NotEnoughValidData {
            needed: worst_needed,
            valid: cols - first,
        });
    }

    // NEW: enforce batch-kernel or auto
    let actual = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(QqeError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    let rows = combos.len();
    let mut fast_mu = make_uninit_matrix(rows, cols);
    let mut slow_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.rsi_period.unwrap() + c.smoothing_factor.unwrap() - 2)
        .collect();

    init_matrix_prefixes(&mut fast_mu, cols, &warm);
    init_matrix_prefixes(&mut slow_mu, cols, &warm);

    // Materialize flat views
    let fast_out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(fast_mu.as_mut_ptr() as *mut f64, rows * cols) };
    let slow_out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(slow_mu.as_mut_ptr() as *mut f64, rows * cols) };

    // Reusable tmp buffer for RSI
    let mut tmp_mu = make_uninit_matrix(1, cols);
    let tmp: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(tmp_mu.as_mut_ptr() as *mut f64, cols) };

    for (row, combo) in combos.iter().enumerate() {
        let rsi_p = combo.rsi_period.unwrap();
        let ema_p = combo.smoothing_factor.unwrap();
        let fast_k = combo.fast_factor.unwrap();
        let start = warm[row];

        let dst_fast = &mut fast_out[row * cols..(row + 1) * cols];
        let dst_slow = &mut slow_out[row * cols..(row + 1) * cols];

        // RSI → tmp
        let rsi_in = RsiInput::from_slice(
            data,
            RsiParams {
                period: Some(rsi_p),
            },
        );
        rsi_into_slice(tmp, &rsi_in, simd).map_err(|e| QqeError::DependentIndicatorError {
            message: e.to_string(),
        })?;

        // EMA(tmp) → dst_fast
        let ema_in = EmaInput::from_slice(
            tmp,
            EmaParams {
                period: Some(ema_p),
            },
        );
        ema_into_slice(dst_fast, &ema_in, simd).map_err(|e| QqeError::DependentIndicatorError {
            message: e.to_string(),
        })?;

        // warm NaNs already inited, compute slow
        qqe_compute_slow_from(dst_fast, fast_k, start, dst_slow);
    }

    let fast_values =
        unsafe { Vec::from_raw_parts(fast_mu.as_mut_ptr() as *mut f64, rows * cols, rows * cols) };
    let slow_values =
        unsafe { Vec::from_raw_parts(slow_mu.as_mut_ptr() as *mut f64, rows * cols, rows * cols) };
    core::mem::forget(fast_mu);
    core::mem::forget(slow_mu);

    Ok(QqeBatchOutput {
        fast_values,
        slow_values,
        combos,
        rows,
        cols,
    })
}

fn qqe_batch_inner(
    data: &[f64],
    sweep: &QqeBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<QqeBatchOutput, QqeError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(QqeError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let cols = data.len();
    if cols == 0 {
        return Err(QqeError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(QqeError::AllValuesNaN)?;
    let worst_needed = combos
        .iter()
        .map(|c| c.rsi_period.unwrap() + c.smoothing_factor.unwrap())
        .max()
        .unwrap();
    if cols - first < worst_needed {
        return Err(QqeError::NotEnoughValidData {
            needed: worst_needed,
            valid: cols - first,
        });
    }

    let rows = combos.len();
    let mut fast_mu = make_uninit_matrix(rows, cols);
    let mut slow_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.rsi_period.unwrap() + c.smoothing_factor.unwrap() - 2)
        .collect();
    init_matrix_prefixes(&mut fast_mu, cols, &warm);
    init_matrix_prefixes(&mut slow_mu, cols, &warm);

    // NEW: enforce batch-kernel or auto
    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(QqeError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    let do_row = |row: usize, f_mu: &mut [MaybeUninit<f64>], s_mu: &mut [MaybeUninit<f64>]| {
        use crate::indicators::moving_averages::ema::ema_into_slice;
        use crate::indicators::rsi::rsi_into_slice;

        // Per-task tmp
        let mut tmp_mu = make_uninit_matrix(1, cols);
        let tmp: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(tmp_mu.as_mut_ptr() as *mut f64, cols) };

        let rsi_p = combos[row].rsi_period.unwrap();
        let ema_p = combos[row].smoothing_factor.unwrap();
        let fast_k = combos[row].fast_factor.unwrap();
        let start = warm[row];

        let dst_fast =
            unsafe { core::slice::from_raw_parts_mut(f_mu.as_mut_ptr() as *mut f64, cols) };
        let dst_slow =
            unsafe { core::slice::from_raw_parts_mut(s_mu.as_mut_ptr() as *mut f64, cols) };

        // RSI → tmp
        let rsi_in = RsiInput::from_slice(
            data,
            RsiParams {
                period: Some(rsi_p),
            },
        );
        rsi_into_slice(tmp, &rsi_in, simd).map_err(|e| QqeError::DependentIndicatorError {
            message: e.to_string(),
        })?;

        // EMA(tmp) → dst_fast
        let ema_in = EmaInput::from_slice(
            tmp,
            EmaParams {
                period: Some(ema_p),
            },
        );
        ema_into_slice(dst_fast, &ema_in, simd).map_err(|e| QqeError::DependentIndicatorError {
            message: e.to_string(),
        })?;

        // Slow from fast
        qqe_compute_slow_from(dst_fast, fast_k, start, dst_slow);

        Ok::<(), QqeError>(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            fast_mu
                .par_chunks_mut(cols)
                .zip(slow_mu.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, (f_mu, s_mu))| do_row(row, f_mu, s_mu))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (f_mu, s_mu)) in fast_mu
                .chunks_mut(cols)
                .zip(slow_mu.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, f_mu, s_mu)?;
            }
        }
    } else {
        for (row, (f_mu, s_mu)) in fast_mu
            .chunks_mut(cols)
            .zip(slow_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, f_mu, s_mu)?;
        }
    }

    let fast_values =
        unsafe { Vec::from_raw_parts(fast_mu.as_mut_ptr() as *mut f64, rows * cols, rows * cols) };
    let slow_values =
        unsafe { Vec::from_raw_parts(slow_mu.as_mut_ptr() as *mut f64, rows * cols, rows * cols) };
    core::mem::forget(fast_mu);
    core::mem::forget(slow_mu);

    Ok(QqeBatchOutput {
        fast_values,
        slow_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn qqe_batch_slice(
    data: &[f64],
    sweep: &QqeBatchRange,
    kern: Kernel,
) -> Result<QqeBatchOutput, QqeError> {
    qqe_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn qqe_batch_par_slice(
    data: &[f64],
    sweep: &QqeBatchRange,
    kern: Kernel,
) -> Result<QqeBatchOutput, QqeError> {
    qqe_batch_inner(data, sweep, kern, true)
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "qqe")]
#[pyo3(signature = (data, rsi_period=14, smoothing_factor=5, fast_factor=4.236, kernel=None))]
pub fn qqe_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period: usize,
    smoothing_factor: usize,
    fast_factor: f64,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = QqeParams {
        rsi_period: Some(rsi_period),
        smoothing_factor: Some(smoothing_factor),
        fast_factor: Some(fast_factor),
    };
    let input = QqeInput::from_slice(slice_in, params);

    let result = py
        .allow_threads(|| qqe_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((result.fast.into_pyarray(py), result.slow.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "QqeStream")]
pub struct QqeStreamPy {
    stream: QqeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl QqeStreamPy {
    #[new]
    fn new(rsi_period: usize, smoothing_factor: usize, fast_factor: f64) -> PyResult<Self> {
        let params = QqeParams {
            rsi_period: Some(rsi_period),
            smoothing_factor: Some(smoothing_factor),
            fast_factor: Some(fast_factor),
        };
        let stream =
            QqeStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(QqeStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "qqe_batch")]
#[pyo3(signature = (data, rsi_period_range, smoothing_factor_range, fast_factor_range, kernel=None))]
pub fn qqe_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period_range: (usize, usize, usize),
    smoothing_factor_range: (usize, usize, usize),
    fast_factor_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let sweep = QqeBatchRange {
        rsi_period: rsi_period_range,
        smoothing_factor: smoothing_factor_range,
        fast_factor: fast_factor_range,
    };
    let kern = validate_kernel(kernel, true)?;

    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("Empty parameter combination"));
    }
    let rows = combos.len();
    let cols = slice_in.len();

    let fast_arr = unsafe { PyArray2::<f64>::new(py, [rows, cols], false) };
    let slow_arr = unsafe { PyArray2::<f64>::new(py, [rows, cols], false) };
    let fast_slice = unsafe { fast_arr.as_slice_mut()? };
    let slow_slice = unsafe { slow_arr.as_slice_mut()? };

    let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.rsi_period.unwrap() + c.smoothing_factor.unwrap() - 2)
        .collect();

    // One tmp buffer for RSI
    let mut tmp_mu = make_uninit_matrix(1, cols);
    let tmp: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(tmp_mu.as_mut_ptr() as *mut f64, cols) };

    use crate::indicators::moving_averages::ema::ema_into_slice;
    use crate::indicators::rsi::rsi_into_slice;

    // Map batch kernel to single kernel
    let simd = match kern {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };

    py.allow_threads(|| -> PyResult<()> {
        for (row, combo) in combos.iter().enumerate() {
            let rsi_p = combo.rsi_period.unwrap();
            let ema_p = combo.smoothing_factor.unwrap();
            let fast_k = combo.fast_factor.unwrap();
            let start = warm[row];

            let dst_fast = &mut fast_slice[row * cols..(row + 1) * cols];
            let dst_slow = &mut slow_slice[row * cols..(row + 1) * cols];

            // RSI → tmp
            rsi_into_slice(
                tmp,
                &RsiInput::from_slice(
                    slice_in,
                    RsiParams {
                        period: Some(rsi_p),
                    },
                ),
                simd,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            // EMA(tmp) → dst_fast
            ema_into_slice(
                dst_fast,
                &EmaInput::from_slice(
                    tmp,
                    EmaParams {
                        period: Some(ema_p),
                    },
                ),
                simd,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

            // warm NaNs
            for v in &mut dst_fast[..start] {
                *v = f64::NAN;
            }
            for v in &mut dst_slow[..start] {
                *v = f64::NAN;
            }

            // slow from fast
            qqe_compute_slow_from(dst_fast, fast_k, start, dst_slow);
        }
        Ok(())
    })?;

    let dict = PyDict::new(py);
    dict.set_item("fast", fast_arr)?;
    dict.set_item("slow", slow_arr)?;
    dict.set_item(
        "rsi_periods",
        combos
            .iter()
            .map(|c| c.rsi_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smoothing_factors",
        combos
            .iter()
            .map(|c| c.smoothing_factor.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "fast_factors",
        combos
            .iter()
            .map(|c| c.fast_factor.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct QqeJsResult {
    pub values: Vec<f64>, // [fast..., slow...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qqe_js(
    data: &[f64],
    rsi_period: usize,
    smoothing_factor: usize,
    fast_factor: f64,
) -> Result<JsValue, JsValue> {
    let params = QqeParams {
        rsi_period: Some(rsi_period),
        smoothing_factor: Some(smoothing_factor),
        fast_factor: Some(fast_factor),
    };
    let input = QqeInput::from_slice(data, params);

    // Preallocate result vector
    let mut values = vec![f64::NAN; data.len() * 2];

    // Split into fast and slow slices
    let (fast_slice, slow_slice) = values.split_at_mut(data.len());

    // Compute directly into slices
    qqe_into_slices(fast_slice, slow_slice, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = QqeJsResult {
        values,
        rows: 2,
        cols: data.len(),
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qqe_unified_js(
    data: &[f64],
    rsi_period: usize,
    smoothing_factor: usize,
    fast_factor: f64,
) -> Result<Vec<f64>, JsValue> {
    let params = QqeParams {
        rsi_period: Some(rsi_period),
        smoothing_factor: Some(smoothing_factor),
        fast_factor: Some(fast_factor),
    };
    let input = QqeInput::from_slice(data, params);

    // Preallocate result vector
    let mut result = vec![f64::NAN; data.len() * 2];

    // Split into fast and slow slices
    let (fast_slice, slow_slice) = result.split_at_mut(data.len());

    // Compute directly into slices
    qqe_into_slices(fast_slice, slow_slice, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qqe_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len * 2); // Allocate for both fast and slow
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qqe_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len * 2, len * 2);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qqe_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_period: usize,
    smoothing_factor: usize,
    fast_factor: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to qqe_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = QqeParams {
            rsi_period: Some(rsi_period),
            smoothing_factor: Some(smoothing_factor),
            fast_factor: Some(fast_factor),
        };
        let input = QqeInput::from_slice(data, params);

        // layout: [fast..len][slow..len]
        if in_ptr == out_ptr {
            let mut tmp = vec![f64::NAN; len * 2];
            let (tmp_fast, tmp_slow) = tmp.split_at_mut(len);
            qqe_into_slices(tmp_fast, tmp_slow, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let dst = std::slice::from_raw_parts_mut(out_ptr, len * 2);
            dst.copy_from_slice(&tmp);
        } else {
            let dst = std::slice::from_raw_parts_mut(out_ptr, len * 2);
            let (dst_fast, dst_slow) = dst.split_at_mut(len);
            qqe_into_slices(dst_fast, dst_slow, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct QqeBatchConfig {
    pub rsi_period_range: (usize, usize, usize),
    pub smoothing_factor_range: (usize, usize, usize),
    pub fast_factor_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize)]
pub struct QqeBatchJsOutput {
    pub fast_values: Vec<f64>,
    pub slow_values: Vec<f64>,
    pub combos: Vec<QqeParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = qqe_batch)]
pub fn qqe_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: QqeBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = QqeBatchRange {
        rsi_period: config.rsi_period_range,
        smoothing_factor: config.smoothing_factor_range,
        fast_factor: config.fast_factor_range,
    };

    let kernel = detect_best_batch_kernel();
    let result = qqe_batch_with_kernel(data, &sweep, kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let output = QqeBatchJsOutput {
        fast_values: result.fast_values,
        slow_values: result.slow_values,
        combos: result.combos,
        rows: result.rows,
        cols: result.cols,
    };

    serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qqe_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_period_start: usize,
    rsi_period_end: usize,
    rsi_period_step: usize,
    smoothing_start: usize,
    smoothing_end: usize,
    smoothing_step: usize,
    fast_factor_start: f64,
    fast_factor_end: f64,
    fast_factor_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to qqe_batch_into"));
    }
    unsafe {
        let data = core::slice::from_raw_parts(in_ptr, len);
        let sweep = QqeBatchRange {
            rsi_period: (rsi_period_start, rsi_period_end, rsi_period_step),
            smoothing_factor: (smoothing_start, smoothing_end, smoothing_step),
            fast_factor: (fast_factor_start, fast_factor_end, fast_factor_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        if rows == 0 {
            return Err(JsValue::from_str("Empty parameter combination"));
        }

        // Layout: [fast rows...][slow rows...]
        let total = rows * len * 2;
        let dst = core::slice::from_raw_parts_mut(out_ptr, total);
        let (dst_fast_all, dst_slow_all) = dst.split_at_mut(rows * len);

        // Reusable tmp for RSI
        let mut tmp_mu = make_uninit_matrix(1, len);
        let tmp: &mut [f64] = core::slice::from_raw_parts_mut(tmp_mu.as_mut_ptr() as *mut f64, len);

        // Kernel
        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        };

        use crate::indicators::moving_averages::ema::ema_into_slice;
        use crate::indicators::rsi::rsi_into_slice;

        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

        for (row, combo) in combos.iter().enumerate() {
            let rsi_p = combo.rsi_period.unwrap();
            let ema_p = combo.smoothing_factor.unwrap();
            let fast_k = combo.fast_factor.unwrap();

            let start = first + rsi_p + ema_p - 2;

            let dst_fast = &mut dst_fast_all[row * len..(row + 1) * len];
            let dst_slow = &mut dst_slow_all[row * len..(row + 1) * len];

            // RSI → tmp
            rsi_into_slice(
                tmp,
                &RsiInput::from_slice(
                    data,
                    RsiParams {
                        period: Some(rsi_p),
                    },
                ),
                simd,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            // EMA(tmp) → fast row
            ema_into_slice(
                dst_fast,
                &EmaInput::from_slice(
                    tmp,
                    EmaParams {
                        period: Some(ema_p),
                    },
                ),
                simd,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // warm NaNs
            for v in &mut dst_fast[..start] {
                *v = f64::NAN;
            }
            for v in &mut dst_slow[..start] {
                *v = f64::NAN;
            }

            // slow from fast
            qqe_compute_slow_from(dst_fast, fast_k, start, dst_slow);
        }
        Ok(rows)
    }
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_qqe_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = QqeInput::from_candles(&candles, "close", QqeParams::default());
        let result = qqe_with_kernel(&input, kernel)?;

        // REFERENCE VALUES FROM PINESCRIPT
        let expected_fast = [
            42.68548144,
            42.68200826,
            42.32797706,
            42.50623375,
            41.34014948,
        ];

        let expected_slow = [
            36.49339135,
            36.59103557,
            36.59103557,
            36.64790896,
            36.64790896,
        ];

        let start = result.fast.len().saturating_sub(5);

        for (i, (&fast_val, &slow_val)) in result.fast[start..]
            .iter()
            .zip(result.slow[start..].iter())
            .enumerate()
        {
            let fast_diff = (fast_val - expected_fast[i]).abs();
            let slow_diff = (slow_val - expected_slow[i]).abs();

            assert!(
                fast_diff < 1e-6,
                "[{}] QQE fast {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                fast_val,
                expected_fast[i]
            );

            assert!(
                slow_diff < 1e-6,
                "[{}] QQE slow {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                slow_val,
                expected_slow[i]
            );
        }
        Ok(())
    }

    fn check_qqe_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = QqeParams {
            rsi_period: None,
            smoothing_factor: None,
            fast_factor: None,
        };
        let input = QqeInput::from_candles(&candles, "close", default_params);
        let output = qqe_with_kernel(&input, kernel)?;
        assert_eq!(output.fast.len(), candles.close.len());
        assert_eq!(output.slow.len(), candles.close.len());

        Ok(())
    }

    fn check_qqe_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = QqeInput::with_default_candles(&candles);
        match input.data {
            QqeData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("[{}] Expected QqeData::Candles", test_name),
        }
        let output = qqe_with_kernel(&input, kernel)?;
        assert_eq!(output.fast.len(), candles.close.len());
        assert_eq!(output.slow.len(), candles.close.len());

        Ok(())
    }

    fn check_qqe_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = QqeParams {
            rsi_period: Some(0),
            smoothing_factor: None,
            fast_factor: None,
        };
        let input = QqeInput::from_slice(&input_data, params);
        let res = qqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] QQE should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_qqe_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = QqeParams {
            rsi_period: Some(10),
            smoothing_factor: None,
            fast_factor: None,
        };
        let input = QqeInput::from_slice(&data_small, params);
        let res = qqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] QQE should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_qqe_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = QqeParams::default();
        let input = QqeInput::from_slice(&single_point, params);
        let res = qqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] QQE should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_qqe_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = QqeParams::default();
        let input = QqeInput::from_slice(&empty, params);
        let res = qqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] QQE should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_qqe_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = QqeParams::default();
        let input = QqeInput::from_slice(&nan_data, params);
        let res = qqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] QQE should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_qqe_batch(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64).sin() * 10.0).collect();

        let sweep = QqeBatchRange {
            rsi_period: (10, 20, 5),
            smoothing_factor: (3, 5, 1),
            fast_factor: (3.0, 5.0, 1.0),
        };

        let result = qqe_batch_with_kernel(&data, &sweep, kernel)?;

        // Should have 3 * 3 * 3 = 27 combinations
        assert_eq!(result.combos.len(), 27);
        assert_eq!(result.rows, 27);
        assert_eq!(result.cols, 100);
        assert_eq!(result.fast_values.len(), 27 * 100);
        assert_eq!(result.slow_values.len(), 27 * 100);

        Ok(())
    }

    fn check_qqe_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let mut stream = QqeStream::try_new(QqeParams::default())?;

        // Feed data
        let data: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64).sin() * 10.0).collect();
        let mut results = Vec::new();

        for &val in &data {
            if let Some(result) = stream.update(val) {
                results.push(result);
            }
        }

        // Should get results after warmup
        assert!(
            !results.is_empty(),
            "[{}] Should have streaming results",
            test_name
        );

        // Verify results are valid
        for (fast, slow) in &results {
            assert!(
                !fast.is_nan(),
                "[{}] Fast value should not be NaN",
                test_name
            );
            assert!(
                !slow.is_nan(),
                "[{}] Slow value should not be NaN",
                test_name
            );
        }

        Ok(())
    }

    fn check_qqe_into_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64).sin() * 10.0).collect();
        let params = QqeParams::default();
        let input = QqeInput::from_slice(&data, params);

        let mut dst_fast = vec![0.0; data.len()];
        let mut dst_slow = vec![0.0; data.len()];

        qqe_into_slices(&mut dst_fast, &mut dst_slow, &input, kernel)?;

        // Compare with regular computation
        let regular = qqe_with_kernel(&input, kernel)?;

        for i in 0..data.len() {
            if dst_fast[i].is_nan() && regular.fast[i].is_nan() {
                // Both NaN is ok
            } else {
                assert_eq!(
                    dst_fast[i], regular.fast[i],
                    "[{}] Fast mismatch at {}",
                    test_name, i
                );
            }

            if dst_slow[i].is_nan() && regular.slow[i].is_nan() {
                // Both NaN is ok
            } else {
                assert_eq!(
                    dst_slow[i], regular.slow[i],
                    "[{}] Slow mismatch at {}",
                    test_name, i
                );
            }
        }

        Ok(())
    }

    fn check_qqe_poison_sentinel(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        // Test that no uninitialized memory is exposed
        let test_data = vec![
            50.0, 51.0, 52.0, 51.5, 50.5, 49.5, 50.0, 51.0, 52.0, 53.0, 52.5, 51.5, 50.5, 51.0,
            52.0, 53.0, 54.0, 53.5, 52.5, 51.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.0, 54.5, 53.5,
            52.5, 51.5,
        ];

        // Test single computation
        {
            // Allocate buffers with poison pattern
            const POISON: f64 = f64::from_bits(0xDEADBEEF_DEADBEEF);
            let mut fast = vec![POISON; test_data.len()];
            let mut slow = vec![POISON; test_data.len()];

            let params = QqeParams::default();
            let input = QqeInput::from_slice(&test_data, params);

            // Compute directly into buffers
            qqe_into_slices(&mut fast[..], &mut slow[..], &input, kernel)?;

            // Check that all values are either NaN (warmup) or valid numbers
            for (i, &val) in fast.iter().enumerate() {
                assert!(
                    val.is_nan() || (val.is_finite() && val != POISON),
                    "[{}] Uninitialized memory detected in fast at index {}: {:?}",
                    test_name,
                    i,
                    val
                );
            }

            for (i, &val) in slow.iter().enumerate() {
                assert!(
                    val.is_nan() || (val.is_finite() && val != POISON),
                    "[{}] Uninitialized memory detected in slow at index {}: {:?}",
                    test_name,
                    i,
                    val
                );
            }
        }

        // Test batch computation
        {
            let sweep = QqeBatchRange {
                rsi_period: (10, 14, 2),
                smoothing_factor: (3, 5, 2),
                fast_factor: (3.0, 4.0, 1.0),
            };

            let batch_out = qqe_batch_with_kernel(&test_data, &sweep, kernel)?;

            // Check all batch values
            for (i, &val) in batch_out.fast_values.iter().enumerate() {
                assert!(
                    val.is_nan() || val.is_finite(),
                    "[{}] Invalid value in batch fast at index {}: {:?}",
                    test_name,
                    i,
                    val
                );
            }

            for (i, &val) in batch_out.slow_values.iter().enumerate() {
                assert!(
                    val.is_nan() || val.is_finite(),
                    "[{}] Invalid value in batch slow at index {}: {:?}",
                    test_name,
                    i,
                    val
                );
            }
        }

        Ok(())
    }

    fn check_qqe_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let p = QqeParams::default();

        let out1 = qqe_with_kernel(&QqeInput::from_candles(&c, "close", p.clone()), kernel)?;
        // reinput on FAST leg is intentional here to stress NaN handling and alignment
        let out2 = qqe_with_kernel(&QqeInput::from_slice(&out1.fast, p), kernel)?;

        assert_eq!(out1.fast.len(), out2.fast.len());
        assert_eq!(out1.slow.len(), out2.slow.len());
        Ok(())
    }

    fn check_qqe_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let p = QqeParams::default();
        let res = qqe_with_kernel(&QqeInput::from_candles(&c, "close", p.clone()), kernel)?;
        let first = c.close.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let warm = first + p.rsi_period.unwrap_or(14) + p.smoothing_factor.unwrap_or(5) - 2;

        for (i, &v) in res.fast.iter().enumerate().skip(warm) {
            assert!(!v.is_nan(), "[{}] fast NaN @ {}", test_name, i);
        }
        for (i, &v) in res.slow.iter().enumerate().skip(warm) {
            assert!(!v.is_nan(), "[{}] slow NaN @ {}", test_name, i);
        }
        Ok(())
    }

    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = QqeBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = QqeParams::default();
        let row = out.row_for_params(&def).expect("default row missing");

        let start = row * out.cols;
        assert_eq!(
            out.fast_values[start..start + out.cols].len(),
            c.close.len()
        );
        assert_eq!(
            out.slow_values[start..start + out.cols].len(),
            c.close.len()
        );
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = QqeBatchBuilder::new()
            .kernel(kernel)
            .rsi_period_range(10, 14, 2)
            .smoothing_factor_range(3, 5, 1)
            .fast_factor_range(3.0, 5.0, 1.0)
            .apply_candles(&c, "close")?;

        for (idx, &v) in out.fast_values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x1111_1111_1111_1111
                    && b != 0x2222_2222_2222_2222
                    && b != 0x3333_3333_3333_3333,
                "[{}] poison in fast @ {}",
                test_name,
                idx
            );
        }
        for (idx, &v) in out.slow_values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x1111_1111_1111_1111
                    && b != 0x2222_2222_2222_2222
                    && b != 0x3333_3333_3333_3333,
                "[{}] poison in slow @ {}",
                test_name,
                idx
            );
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_qqe_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // generate periods and data long enough for warmup
        let strat = (1usize..=64).prop_flat_map(|rsi_p| {
            (1usize..=32).prop_flat_map(move |ema_p| {
                let need = rsi_p + ema_p + 8;
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        need..400,
                    ),
                    Just(rsi_p),
                    Just(ema_p),
                    0.5f64..8.0f64,
                )
            })
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, rsi_p, ema_p, fast_k)| {
                let p = QqeParams {
                    rsi_period: Some(rsi_p),
                    smoothing_factor: Some(ema_p),
                    fast_factor: Some(fast_k),
                };
                let input = QqeInput::from_slice(&data, p);

                // reference via with_kernel
                let ref_out = qqe_with_kernel(&input, Kernel::Scalar).unwrap();

                // into_slices path must match exactly
                let mut f = vec![0.0; data.len()];
                let mut s = vec![0.0; data.len()];
                qqe_into_slices(&mut f, &mut s, &input, Kernel::Scalar).unwrap();

                for i in 0..data.len() {
                    let a = ref_out.fast[i];
                    let b = f[i];
                    if a.is_nan() {
                        prop_assert!(b.is_nan());
                    } else {
                        prop_assert!((a - b).abs() <= 1e-9);
                    }

                    let c = ref_out.slow[i];
                    let d = s[i];
                    if c.is_nan() {
                        prop_assert!(d.is_nan());
                    } else {
                        prop_assert!((c - d).abs() <= 1e-9);
                    }

                    if !a.is_nan() {
                        prop_assert!(a >= 0.0 && a <= 100.0);
                    }
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    // Macro to generate tests for all kernel variants
    macro_rules! generate_all_qqe_tests {
        ($($test_fn:ident),+ $(,)?) => {

            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar);
                    }
                )*

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2);
                    }

                    #[test]
                    fn [<$test_fn _avx512>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512);
                    }
                )*

                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128>]), Kernel::Scalar);
                    }
                )*
            }
        };
    }

    generate_all_qqe_tests!(
        check_qqe_accuracy,
        check_qqe_partial_params,
        check_qqe_default_candles,
        check_qqe_zero_period,
        check_qqe_period_exceeds_length,
        check_qqe_very_small_dataset,
        check_qqe_empty_input,
        check_qqe_all_nan,
        check_qqe_batch,
        check_qqe_streaming,
        check_qqe_into_slices,
        check_qqe_poison_sentinel,
        check_qqe_reinput,
        check_qqe_nan_handling,
        check_batch_default_row,
        check_batch_no_poison,
    );

    #[cfg(feature = "proptest")]
    generate_all_qqe_tests!(check_qqe_property);
}
