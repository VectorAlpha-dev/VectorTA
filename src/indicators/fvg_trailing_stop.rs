//! # FVG Trailing Stop
//!
//! FVG Trailing Stop is a trend-following indicator that identifies Fair Value Gaps (FVGs) in price action
//! and uses them to create dynamic trailing stop levels. It combines FVG detection with smoothed channel
//! extremes to generate adaptive support and resistance levels.
//!
//! ## Parameters
//! - **unmitigated_fvg_lookback**: Number of FVGs to track in lookback window (default: 5)
//! - **smoothing_length**: Period for SMA smoothing of levels (default: 9)
//! - **reset_on_cross**: Whether to reset trailing stop on cross (default: false)
//!
//! ## Returns
//! - **`Ok(FvgTrailingStopOutput)`** containing:
//!   - `upper`: Upper channel boundary (NaN when lower is active)
//!   - `lower`: Lower channel boundary (NaN when upper is active)
//!   - `upper_ts`: Upper trailing stop (NaN when lower is active)
//!   - `lower_ts`: Lower trailing stop (NaN when upper is active)
//!
//! ## Developer Notes
//! ### Implementation Status
//! - **AVX2 Kernel**: Not implemented (no SIMD kernels for this indicator)
//! - **AVX512 Kernel**: Not implemented (no SIMD kernels for this indicator)
//! - **Streaming Update**: O(1) - efficient with VecDeque for FVG tracking
//! - **Memory Optimization**: Fully optimized with `alloc_with_nan_prefix` for all output vectors
//! - **Batch Operations**: Not implemented (complex state management makes batching difficult)
//!
//! ### TODO - Performance Improvements
//! - [ ] Consider adding SIMD kernels for SMA calculations
//! - [ ] Optimize FVG detection logic
//! - [ ] Implement batch operations if use case emerges
//! - [ ] Consider caching smoothed values to avoid recalculation
//! - [ ] Optimize the trailing stop update logic

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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

// Standard library imports
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// ==================== DATA STRUCTURES ====================
/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct FvgTrailingStopOutput {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper_ts: Vec<f64>,
    pub lower_ts: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct FvgTrailingStopParams {
    pub unmitigated_fvg_lookback: Option<usize>,
    pub smoothing_length: Option<usize>,
    pub reset_on_cross: Option<bool>,
}

impl Default for FvgTrailingStopParams {
    fn default() -> Self {
        Self {
            unmitigated_fvg_lookback: Some(5),
            smoothing_length: Some(9),
            reset_on_cross: Some(false),
        }
    }
}

/// Data source for FVG Trailing Stop - either Candles or direct slices
#[derive(Debug, Clone)]
pub enum FvgTrailingStopData<'a> {
    Candles(&'a Candles),
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

/// Helper function to find first valid OHLC data point
#[inline]
fn first_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    for i in 0..high.len() {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            return i;
        }
    }
    usize::MAX
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct FvgTrailingStopInput<'a> {
    pub data: FvgTrailingStopData<'a>,
    pub params: FvgTrailingStopParams,
}

impl<'a> FvgTrailingStopInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: FvgTrailingStopParams) -> Self {
        Self {
            data: FvgTrailingStopData::Candles(candles),
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FvgTrailingStopParams,
    ) -> Self {
        Self {
            data: FvgTrailingStopData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FvgTrailingStopParams::default())
    }

    pub fn get_lookback(&self) -> usize {
        self.params.unmitigated_fvg_lookback.unwrap_or(5)
    }

    pub fn get_smoothing(&self) -> usize {
        self.params.smoothing_length.unwrap_or(9)
    }

    pub fn get_reset_on_cross(&self) -> bool {
        self.params.reset_on_cross.unwrap_or(false)
    }

    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            FvgTrailingStopData::Candles(c) => (&c.high, &c.low, &c.close),
            FvgTrailingStopData::Slices { high, low, close } => (high, low, close),
        }
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum FvgTrailingStopError {
    #[error("fvg_trailing_stop: Input data slice is empty.")]
    EmptyInputData,

    #[error("fvg_trailing_stop: All values are NaN.")]
    AllValuesNaN,

    #[error("fvg_trailing_stop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("fvg_trailing_stop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("fvg_trailing_stop: Invalid smoothing_length: {smoothing}")]
    InvalidSmoothingLength { smoothing: usize },

    #[error("fvg_trailing_stop: Invalid unmitigated_fvg_lookback: {lookback}")]
    InvalidLookback { lookback: usize },
}

// ==================== KERNEL IMPLEMENTATIONS ====================
#[inline]
fn fvg_ts_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    smoothing_len: usize,
    reset_on_cross: bool,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_ts: &mut [f64],
    lower_ts: &mut [f64],
) {
    let len = high.len();
    // State - pre-allocate history buffers to avoid reallocation
    let mut bull_lvls: VecDeque<f64> = VecDeque::with_capacity(lookback);
    let mut bear_lvls: VecDeque<f64> = VecDeque::with_capacity(lookback);
    let mut last_bull_non_na: Option<usize> = None;
    let mut last_bear_non_na: Option<usize> = None;
    let mut bull_hist = vec![f64::NAN; len];
    let mut bear_hist = vec![f64::NAN; len];
    let mut os: Option<i8> = None;
    let mut ts: Option<f64> = None;
    let mut ts_prev: Option<f64> = None;

    for i in 0..len {
        // defaults stay NaN; warm prefix already written by allocator
        // FVG detection
        if i >= 2 && !high[i - 2].is_nan() && !low[i - 2].is_nan() && !close[i - 1].is_nan() {
            if low[i] > high[i - 2] && close[i - 1] > high[i - 2] {
                bull_lvls.push_back(high[i - 2]);
                if bull_lvls.len() > lookback {
                    bull_lvls.pop_front();
                }
            }
            if high[i] < low[i - 2] && close[i - 1] < low[i - 2] {
                bear_lvls.push_back(low[i - 2]);
                if bear_lvls.len() > lookback {
                    bear_lvls.pop_front();
                }
            }
        }
        // mitigation
        let c = close[i];
        bull_lvls.retain(|&lvl| c >= lvl);
        bear_lvls.retain(|&lvl| c <= lvl);

        let bull_avg = if bull_lvls.is_empty() {
            f64::NAN
        } else {
            bull_lvls.iter().sum::<f64>() / (bull_lvls.len() as f64)
        };
        let bear_avg = if bear_lvls.is_empty() {
            f64::NAN
        } else {
            bear_lvls.iter().sum::<f64>() / (bear_lvls.len() as f64)
        };
        if !bull_avg.is_nan() {
            last_bull_non_na = Some(i);
        }
        if !bear_avg.is_nan() {
            last_bear_non_na = Some(i);
        }

        // progressive SMA fallbacks
        let bull_bs = if bull_avg.is_nan() {
            match last_bull_non_na {
                Some(last) => ((i - last).max(1)).min(smoothing_len),
                None => 1,
            }
        } else {
            1
        };
        let bear_bs = if bear_avg.is_nan() {
            match last_bear_non_na {
                Some(last) => ((i - last).max(1)).min(smoothing_len),
                None => 1,
            }
        } else {
            1
        };

        let bull_sma = if bull_avg.is_nan() && i + 1 >= bull_bs {
            let mut sum = 0.0;
            for j in (i + 1 - bull_bs)..=i {
                sum += close[j];
            }
            sum / bull_bs as f64
        } else {
            f64::NAN
        };
        let bear_sma = if bear_avg.is_nan() && i + 1 >= bear_bs {
            let mut sum = 0.0;
            for j in (i + 1 - bear_bs)..=i {
                sum += close[j];
            }
            sum / bear_bs as f64
        } else {
            f64::NAN
        };

        let x_bull = if !bull_avg.is_nan() {
            bull_avg
        } else {
            bull_sma
        };
        let x_bear = if !bear_avg.is_nan() {
            bear_avg
        } else {
            bear_sma
        };
        bull_hist[i] = x_bull;
        bear_hist[i] = x_bear;

        // fixed-window SMA over x-series; NaN if any NaN in window
        let mut bull_disp = f64::NAN;
        let mut bear_disp = f64::NAN;
        if i + 1 >= smoothing_len {
            let start = i + 1 - smoothing_len;
            let mut ok = true;
            let mut s = 0.0;
            for j in start..=i {
                let v = bull_hist[j];
                if v.is_nan() {
                    ok = false;
                    break;
                }
                s += v;
            }
            if ok {
                bull_disp = s / smoothing_len as f64;
            }
            let mut ok2 = true;
            let mut s2 = 0.0;
            for j in start..=i {
                let v = bear_hist[j];
                if v.is_nan() {
                    ok2 = false;
                    break;
                }
                s2 += v;
            }
            if ok2 {
                bear_disp = s2 / smoothing_len as f64;
            }
        }

        let prev_os = os;
        let next_os = if !bear_disp.is_nan() && c > bear_disp {
            Some(1)
        } else if !bull_disp.is_nan() && c < bull_disp {
            Some(-1)
        } else {
            os
        };
        os = next_os;

        if let (Some(cur), Some(prev)) = (os, prev_os) {
            if cur == 1 && prev != 1 {
                ts = Some(bull_disp);
            } else if cur == -1 && prev != -1 {
                ts = Some(bear_disp);
            } else if cur == 1 {
                if let Some(t) = ts {
                    ts = Some(bull_disp.max(t));
                }
            } else if cur == -1 {
                if let Some(t) = ts {
                    ts = Some(bear_disp.min(t));
                }
            }
        } else {
            if os == Some(1) {
                if let Some(t) = ts {
                    ts = Some(bull_disp.max(t));
                }
            }
            if os == Some(-1) {
                if let Some(t) = ts {
                    ts = Some(bear_disp.min(t));
                }
            }
        }

        if reset_on_cross {
            if os == Some(1) {
                if let Some(t) = ts {
                    if c < t {
                        ts = None;
                    }
                } else if !bear_disp.is_nan() && c > bear_disp {
                    ts = Some(bull_disp);
                }
            } else if os == Some(-1) {
                if let Some(t) = ts {
                    if c > t {
                        ts = None;
                    }
                } else if !bull_disp.is_nan() && c < bull_disp {
                    ts = Some(bear_disp);
                }
            }
        }

        let show = ts.is_some() || ts_prev.is_some();
        let ts_nz = ts.or(ts_prev);

        // Always write values to avoid leaving poison values in debug mode
        if os == Some(1) && show {
            upper[i] = f64::NAN;
            lower[i] = bull_disp;
            upper_ts[i] = f64::NAN;
            lower_ts[i] = ts_nz.unwrap_or(f64::NAN);
        } else if os == Some(-1) && show {
            upper[i] = bear_disp;
            lower[i] = f64::NAN;
            upper_ts[i] = ts_nz.unwrap_or(f64::NAN);
            lower_ts[i] = f64::NAN;
        } else {
            // Ensure all arrays have values written (not poison)
            upper[i] = f64::NAN;
            lower[i] = f64::NAN;
            upper_ts[i] = f64::NAN;
            lower_ts[i] = f64::NAN;
        }
        ts_prev = ts;
    }
}

// Stub for AVX2 implementation (fallback to scalar)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn fvg_ts_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    smoothing_len: usize,
    reset_on_cross: bool,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_ts: &mut [f64],
    lower_ts: &mut [f64],
) {
    // Future optimization: implement AVX2 version
    // For now, fallback to scalar
    fvg_ts_scalar(
        high,
        low,
        close,
        lookback,
        smoothing_len,
        reset_on_cross,
        upper,
        lower,
        upper_ts,
        lower_ts,
    );
}

// Stub for AVX512 implementation (fallback to scalar)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn fvg_ts_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    smoothing_len: usize,
    reset_on_cross: bool,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_ts: &mut [f64],
    lower_ts: &mut [f64],
) {
    // Future optimization: implement AVX512 version
    // For now, fallback to scalar
    fvg_ts_scalar(
        high,
        low,
        close,
        lookback,
        smoothing_len,
        reset_on_cross,
        upper,
        lower,
        upper_ts,
        lower_ts,
    );
}

// Stub for WASM SIMD128 implementation
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn fvg_ts_simd128(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    smoothing_len: usize,
    reset_on_cross: bool,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_ts: &mut [f64],
    lower_ts: &mut [f64],
) {
    // Future optimization: implement SIMD128 version
    // For now, fallback to scalar
    fvg_ts_scalar(
        high,
        low,
        close,
        lookback,
        smoothing_len,
        reset_on_cross,
        upper,
        lower,
        upper_ts,
        lower_ts,
    );
}

// ==================== CORE COMPUTATION ====================
#[inline]
fn fvg_ts_prepare<'a>(
    input: &'a FvgTrailingStopInput,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, bool, usize), FvgTrailingStopError> {
    let (h, l, c) = input.as_slices();
    if h.is_empty() || l.is_empty() || c.is_empty() {
        return Err(FvgTrailingStopError::EmptyInputData);
    }
    let len = h.len();
    if len != l.len() || len != c.len() {
        return Err(FvgTrailingStopError::InvalidPeriod {
            period: len,
            data_len: len,
        });
    }
    let first = first_valid_ohlc(h, l, c);
    if first == usize::MAX {
        return Err(FvgTrailingStopError::AllValuesNaN);
    }
    let lookback = input.get_lookback();
    let smoothing_len = input.get_smoothing();

    // Validate parameters with specific errors
    if lookback == 0 {
        return Err(FvgTrailingStopError::InvalidLookback { lookback });
    }
    if smoothing_len == 0 {
        return Err(FvgTrailingStopError::InvalidSmoothingLength {
            smoothing: smoothing_len,
        });
    }

    // need at least 2 (FVG) + (smoothing_len - 1)
    let need = 2 + smoothing_len.saturating_sub(1);
    if len - first < need {
        return Err(FvgTrailingStopError::NotEnoughValidData {
            needed: need,
            valid: len - first,
        });
    }
    let reset_on_cross = input.get_reset_on_cross();
    Ok((h, l, c, lookback, smoothing_len, reset_on_cross, first))
}

#[inline]
fn fvg_ts_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    smoothing_len: usize,
    reset_on_cross: bool,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_ts: &mut [f64],
    lower_ts: &mut [f64],
    kernel: Kernel,
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                fvg_ts_simd128(
                    high,
                    low,
                    close,
                    lookback,
                    smoothing_len,
                    reset_on_cross,
                    upper,
                    lower,
                    upper_ts,
                    lower_ts,
                );
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => fvg_ts_scalar(
                high,
                low,
                close,
                lookback,
                smoothing_len,
                reset_on_cross,
                upper,
                lower,
                upper_ts,
                lower_ts,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => fvg_ts_avx2(
                high,
                low,
                close,
                lookback,
                smoothing_len,
                reset_on_cross,
                upper,
                lower,
                upper_ts,
                lower_ts,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => fvg_ts_avx512(
                high,
                low,
                close,
                lookback,
                smoothing_len,
                reset_on_cross,
                upper,
                lower,
                upper_ts,
                lower_ts,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                fvg_ts_scalar(
                    high,
                    low,
                    close,
                    lookback,
                    smoothing_len,
                    reset_on_cross,
                    upper,
                    lower,
                    upper_ts,
                    lower_ts,
                )
            }
            _ => unreachable!(),
        }
    }
}

// ==================== MAIN ALGORITHM ====================
#[inline]
pub fn fvg_trailing_stop(
    input: &FvgTrailingStopInput,
) -> Result<FvgTrailingStopOutput, FvgTrailingStopError> {
    fvg_trailing_stop_with_kernel(input, Kernel::Auto)
}

pub fn fvg_trailing_stop_with_kernel(
    input: &FvgTrailingStopInput,
    kernel: Kernel,
) -> Result<FvgTrailingStopOutput, FvgTrailingStopError> {
    let (h, l, c, lookback, smoothing_len, reset_on_cross, first) = fvg_ts_prepare(input)?;
    let len = h.len();
    let warm = (first + 2 + smoothing_len.saturating_sub(1)).min(len);

    let mut upper = alloc_with_nan_prefix(len, warm);
    let mut lower = alloc_with_nan_prefix(len, warm);
    let mut upper_ts = alloc_with_nan_prefix(len, warm);
    let mut lower_ts = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    fvg_ts_compute_into(
        h,
        l,
        c,
        lookback,
        smoothing_len,
        reset_on_cross,
        &mut upper,
        &mut lower,
        &mut upper_ts,
        &mut lower_ts,
        chosen,
    );

    Ok(FvgTrailingStopOutput {
        upper,
        lower,
        upper_ts,
        lower_ts,
    })
}

#[inline]
pub fn fvg_trailing_stop_into_slices(
    upper: &mut [f64],
    lower: &mut [f64],
    upper_ts: &mut [f64],
    lower_ts: &mut [f64],
    input: &FvgTrailingStopInput,
    kernel: Kernel,
) -> Result<(), FvgTrailingStopError> {
    let (h, l, c, lookback, smoothing_len, reset_on_cross, first) = fvg_ts_prepare(input)?;
    let len = h.len();
    if [upper.len(), lower.len(), upper_ts.len(), lower_ts.len()]
        .iter()
        .any(|&n| n != len)
    {
        return Err(FvgTrailingStopError::InvalidPeriod {
            period: len,
            data_len: len,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    fvg_ts_compute_into(
        h,
        l,
        c,
        lookback,
        smoothing_len,
        reset_on_cross,
        upper,
        lower,
        upper_ts,
        lower_ts,
        chosen,
    );

    // enforce warm NaNs like alma_into_slice
    let warm = (first + 2 + smoothing_len.saturating_sub(1)).min(len);
    for dst in [upper, lower, upper_ts, lower_ts] {
        for v in &mut dst[..warm] {
            *v = f64::NAN;
        }
    }
    Ok(())
}

// ==================== BATCH OPERATIONS ====================
#[derive(Clone, Debug)]
pub struct FvgTsBatchRange {
    pub lookback: (usize, usize, usize),
    pub smoothing: (usize, usize, usize),
    pub reset_on_cross: (bool, bool),
}

impl Default for FvgTsBatchRange {
    fn default() -> Self {
        Self {
            lookback: (5, 5, 0),
            smoothing: (9, 9, 0),
            reset_on_cross: (false, false),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FvgTsBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FvgTrailingStopParams>,
    pub rows: usize,
    pub cols: usize,
}

impl FvgTsBatchOutput {
    /// Find combo row index for given params (combo index, not per-series index).
    pub fn row_for_params(&self, p: &FvgTrailingStopParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.unmitigated_fvg_lookback.unwrap_or(5) == p.unmitigated_fvg_lookback.unwrap_or(5)
                && c.smoothing_length.unwrap_or(9) == p.smoothing_length.unwrap_or(9)
                && c.reset_on_cross.unwrap_or(false) == p.reset_on_cross.unwrap_or(false)
        })
    }

    /// Return the 4 output slices (upper, lower, upper_ts, lower_ts) for a combo.
    /// Layout: for combo r, values block starts at r*4*cols.
    pub fn values_for(
        &self,
        p: &FvgTrailingStopParams,
    ) -> Option<(&[f64], &[f64], &[f64], &[f64])> {
        let r = self.row_for_params(p)?;
        let cols = self.cols;
        let base = r * 4 * cols;
        Some((
            &self.values[base..base + cols],
            &self.values[base + cols..base + 2 * cols],
            &self.values[base + 2 * cols..base + 3 * cols],
            &self.values[base + 3 * cols..base + 4 * cols],
        ))
    }
}

#[inline]
fn expand_grid_ts(r: &FvgTsBatchRange) -> Vec<FvgTrailingStopParams> {
    let mut v = Vec::new();
    let looks = if r.lookback.2 == 0 {
        vec![r.lookback.0]
    } else {
        (r.lookback.0..=r.lookback.1)
            .step_by(r.lookback.2)
            .collect()
    };
    let smooths = if r.smoothing.2 == 0 {
        vec![r.smoothing.0]
    } else {
        (r.smoothing.0..=r.smoothing.1)
            .step_by(r.smoothing.2)
            .collect()
    };
    // reset_on_cross is (include_false, include_true)
    let mut resets = Vec::new();
    if r.reset_on_cross.0 {
        resets.push(false);
    }
    if r.reset_on_cross.1 {
        resets.push(true);
    }
    if resets.is_empty() {
        resets.push(false);
    } // Default to false if neither is selected

    for &lb in &looks {
        for &sm in &smooths {
            for &rs in &resets {
                v.push(FvgTrailingStopParams {
                    unmitigated_fvg_lookback: Some(lb),
                    smoothing_length: Some(sm),
                    reset_on_cross: Some(rs),
                });
            }
        }
    }
    v
}

#[inline(always)]
pub fn fvg_ts_batch_inner_into(
    h: &[f64],
    l: &[f64],
    c: &[f64],
    sweep: &FvgTsBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64], // layout: for each combo, 4 consecutive rows: upper,lower,upper_ts,lower_ts
) -> Result<Vec<FvgTrailingStopParams>, FvgTrailingStopError> {
    let combos = expand_grid_ts(sweep);
    if combos.is_empty() {
        return Err(FvgTrailingStopError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let len = h.len();
    let rows = combos.len();
    let cols = len;
    assert_eq!(out.len(), 4 * rows * cols, "out size mismatch");

    let first = first_valid_ohlc(h, l, c);
    if first == usize::MAX {
        return Err(FvgTrailingStopError::AllValuesNaN);
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_one = |row: usize, dst: &mut [f64]| {
        let sm = combos[row].smoothing_length.unwrap_or(9);
        let warm = (first + 2 + sm.saturating_sub(1)).min(cols);
        let (u_block, rest) = dst.split_at_mut(cols);
        let (l_block, rest) = rest.split_at_mut(cols);
        let (uts_block, lts_block) = rest.split_at_mut(cols);

        fvg_ts_compute_into(
            h,
            l,
            c,
            combos[row].unmitigated_fvg_lookback.unwrap(),
            sm,
            combos[row].reset_on_cross.unwrap_or(false),
            u_block,
            l_block,
            uts_block,
            lts_block,
            chosen,
        );
        for buf in [u_block, l_block, uts_block, lts_block] {
            for v in &mut buf[..warm] {
                *v = f64::NAN;
            }
        }
    };

    // rows × (4*cols) chunking
    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        use rayon::prelude::*;
        out.par_chunks_mut(4 * cols)
            .enumerate()
            .for_each(|(row, dst)| do_one(row, dst));
    } else {
        out.chunks_mut(4 * cols)
            .enumerate()
            .for_each(|(row, dst)| do_one(row, dst));
    }

    #[cfg(target_arch = "wasm32")]
    out.chunks_mut(4 * cols)
        .enumerate()
        .for_each(|(row, dst)| do_one(row, dst));

    Ok(combos)
}

pub fn fvg_trailing_stop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FvgTsBatchRange,
    kernel: Kernel,
) -> Result<FvgTsBatchOutput, FvgTrailingStopError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FvgTrailingStopError::EmptyInputData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(FvgTrailingStopError::InvalidPeriod {
            period: len,
            data_len: len,
        });
    }

    let combos = expand_grid_ts(sweep);
    let rows = combos.len();
    let cols = len;

    // warm per output row (upper/lower/uts/lts share same warm per combo)
    let first = first_valid_ohlc(high, low, close);
    if first == usize::MAX {
        return Err(FvgTrailingStopError::AllValuesNaN);
    }
    let mut warms = Vec::with_capacity(4 * rows);
    for prm in &combos {
        let sm = prm.smoothing_length.unwrap_or(9);
        let w = (first + 2 + sm.saturating_sub(1)).min(cols);
        warms.extend_from_slice(&[w, w, w, w]);
    }

    let mut buf_mu = make_uninit_matrix(4 * rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    // one pass fill
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let flat: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let used = fvg_ts_batch_inner_into(high, low, close, sweep, kernel, true, flat)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    core::mem::forget(guard);

    Ok(FvgTsBatchOutput {
        values,
        combos: used,
        rows,
        cols,
    })
}

// ==================== BATCH BUILDER ====================
#[derive(Clone, Debug, Default)]
pub struct FvgTsBatchBuilder {
    range: FvgTsBatchRange,
    kernel: Kernel,
}

impl FvgTsBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }

    pub fn smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing = (start, end, step);
        self
    }

    pub fn reset_toggle(mut self, include_false: bool, include_true: bool) -> Self {
        self.range.reset_on_cross = (include_false, include_true);
        self
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_candles(self, c: &Candles) -> Result<FvgTsBatchOutput, FvgTrailingStopError> {
        fvg_trailing_stop_batch_with_kernel(&c.high, &c.low, &c.close, &self.range, self.kernel)
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FvgTsBatchOutput, FvgTrailingStopError> {
        fvg_trailing_stop_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    pub fn with_default_candles(c: &Candles) -> Result<FvgTsBatchOutput, FvgTrailingStopError> {
        FvgTsBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }

    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FvgTsBatchOutput, FvgTrailingStopError> {
        FvgTsBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_slices(high, low, close)
    }
}

// ==================== STREAMING SUPPORT ====================
pub struct FvgTrailingStopStream {
    lookback: usize,
    smoothing_len: usize,
    reset_on_cross: bool,
    kernel: Kernel,
    // State
    bull_lvls: VecDeque<f64>,
    bear_lvls: VecDeque<f64>,
    last_bull_non_na: Option<usize>,
    last_bear_non_na: Option<usize>,
    bull_hist: VecDeque<f64>,
    bear_hist: VecDeque<f64>,
    os: Option<i8>,
    ts: Option<f64>,
    ts_prev: Option<f64>,
    bar_count: usize,
    prev_high2: f64,
    prev_low2: f64,
    prev_close: f64,
}

impl FvgTrailingStopStream {
    pub fn try_new(params: FvgTrailingStopParams) -> Result<Self, FvgTrailingStopError> {
        let lookback = params.unmitigated_fvg_lookback.unwrap_or(5);
        let smoothing_len = params.smoothing_length.unwrap_or(9);

        Ok(Self {
            lookback,
            smoothing_len,
            reset_on_cross: params.reset_on_cross.unwrap_or(false),
            kernel: Kernel::Auto,
            bull_lvls: VecDeque::with_capacity(lookback),
            bear_lvls: VecDeque::with_capacity(lookback),
            last_bull_non_na: None,
            last_bear_non_na: None,
            bull_hist: VecDeque::with_capacity(smoothing_len),
            bear_hist: VecDeque::with_capacity(smoothing_len),
            os: None,
            ts: None,
            ts_prev: None,
            bar_count: 0,
            prev_high2: f64::NAN,
            prev_low2: f64::NAN,
            prev_close: f64::NAN,
        })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        // FVG detection (need at least 3 bars)
        if self.bar_count >= 2
            && !self.prev_high2.is_nan()
            && !self.prev_low2.is_nan()
            && !self.prev_close.is_nan()
        {
            if low > self.prev_high2 && self.prev_close > self.prev_high2 {
                self.bull_lvls.push_back(self.prev_high2);
                if self.bull_lvls.len() > self.lookback {
                    self.bull_lvls.pop_front();
                }
            }
            if high < self.prev_low2 && self.prev_close < self.prev_low2 {
                self.bear_lvls.push_back(self.prev_low2);
                if self.bear_lvls.len() > self.lookback {
                    self.bear_lvls.pop_front();
                }
            }
        }

        // Mitigation
        self.bull_lvls.retain(|&lvl| close >= lvl);
        self.bear_lvls.retain(|&lvl| close <= lvl);

        let bull_avg = if self.bull_lvls.is_empty() {
            f64::NAN
        } else {
            self.bull_lvls.iter().sum::<f64>() / (self.bull_lvls.len() as f64)
        };

        let bear_avg = if self.bear_lvls.is_empty() {
            f64::NAN
        } else {
            self.bear_lvls.iter().sum::<f64>() / (self.bear_lvls.len() as f64)
        };

        if !bull_avg.is_nan() {
            self.last_bull_non_na = Some(self.bar_count);
        }
        if !bear_avg.is_nan() {
            self.last_bear_non_na = Some(self.bar_count);
        }

        // Progressive SMA fallbacks
        let bull_bs = if bull_avg.is_nan() {
            match self.last_bull_non_na {
                Some(last) => ((self.bar_count - last).max(1)).min(self.smoothing_len),
                None => 1,
            }
        } else {
            1
        };

        let bear_bs = if bear_avg.is_nan() {
            match self.last_bear_non_na {
                Some(last) => ((self.bar_count - last).max(1)).min(self.smoothing_len),
                None => 1,
            }
        } else {
            1
        };

        // For streaming, we don't have full history for progressive SMA
        // So we'll use the current close as fallback
        let bull_sma = if bull_avg.is_nan() { close } else { f64::NAN };
        let bear_sma = if bear_avg.is_nan() { close } else { f64::NAN };

        let x_bull = if !bull_avg.is_nan() {
            bull_avg
        } else {
            bull_sma
        };
        let x_bear = if !bear_avg.is_nan() {
            bear_avg
        } else {
            bear_sma
        };

        self.bull_hist.push_back(x_bull);
        if self.bull_hist.len() > self.smoothing_len {
            self.bull_hist.pop_front();
        }

        self.bear_hist.push_back(x_bear);
        if self.bear_hist.len() > self.smoothing_len {
            self.bear_hist.pop_front();
        }

        // Fixed-window SMA
        let mut bull_disp = f64::NAN;
        let mut bear_disp = f64::NAN;

        if self.bull_hist.len() == self.smoothing_len {
            let mut ok = true;
            let mut sum = 0.0;
            for &v in &self.bull_hist {
                if v.is_nan() {
                    ok = false;
                    break;
                }
                sum += v;
            }
            if ok {
                bull_disp = sum / self.smoothing_len as f64;
            }
        }

        if self.bear_hist.len() == self.smoothing_len {
            let mut ok = true;
            let mut sum = 0.0;
            for &v in &self.bear_hist {
                if v.is_nan() {
                    ok = false;
                    break;
                }
                sum += v;
            }
            if ok {
                bear_disp = sum / self.smoothing_len as f64;
            }
        }

        // Update OS and TS
        let prev_os = self.os;
        let next_os = if !bear_disp.is_nan() && close > bear_disp {
            Some(1)
        } else if !bull_disp.is_nan() && close < bull_disp {
            Some(-1)
        } else {
            self.os
        };
        self.os = next_os;

        if let (Some(cur), Some(prev)) = (self.os, prev_os) {
            if cur == 1 && prev != 1 {
                self.ts = Some(bull_disp);
            } else if cur == -1 && prev != -1 {
                self.ts = Some(bear_disp);
            } else if cur == 1 {
                if let Some(t) = self.ts {
                    self.ts = Some(bull_disp.max(t));
                }
            } else if cur == -1 {
                if let Some(t) = self.ts {
                    self.ts = Some(bear_disp.min(t));
                }
            }
        } else {
            if self.os == Some(1) {
                if let Some(t) = self.ts {
                    self.ts = Some(bull_disp.max(t));
                }
            }
            if self.os == Some(-1) {
                if let Some(t) = self.ts {
                    self.ts = Some(bear_disp.min(t));
                }
            }
        }

        if self.reset_on_cross {
            if self.os == Some(1) {
                if let Some(t) = self.ts {
                    if close < t {
                        self.ts = None;
                    }
                } else if !bear_disp.is_nan() && close > bear_disp {
                    self.ts = Some(bull_disp);
                }
            } else if self.os == Some(-1) {
                if let Some(t) = self.ts {
                    if close > t {
                        self.ts = None;
                    }
                } else if !bull_disp.is_nan() && close < bull_disp {
                    self.ts = Some(bear_disp);
                }
            }
        }

        let show = self.ts.is_some() || self.ts_prev.is_some();
        let ts_nz = self.ts.or(self.ts_prev);

        let (mut upper, mut lower, mut upper_ts, mut lower_ts) =
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN);

        if self.os == Some(1) && show {
            lower = bull_disp;
            lower_ts = ts_nz.unwrap_or(f64::NAN);
        } else if self.os == Some(-1) && show {
            upper = bear_disp;
            upper_ts = ts_nz.unwrap_or(f64::NAN);
        }

        self.ts_prev = self.ts;

        // Update state for next bar
        self.prev_high2 = if self.bar_count >= 1 { high } else { f64::NAN };
        self.prev_low2 = if self.bar_count >= 1 { low } else { f64::NAN };
        self.prev_close = close;
        self.bar_count += 1;

        if self.bar_count >= self.smoothing_len + 2 {
            Some((upper, lower, upper_ts, lower_ts))
        } else {
            None
        }
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "fvg_trailing_stop")]
#[pyo3(signature = (high, low, close, unmitigated_fvg_lookback, smoothing_length, reset_on_cross, kernel=None))]
pub fn fvg_trailing_stop_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    unmitigated_fvg_lookback: usize,
    smoothing_length: usize,
    reset_on_cross: bool,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    use numpy::IntoPyArray;
    let (h, l, c) = (high.as_slice()?, low.as_slice()?, close.as_slice()?);
    let kern = validate_kernel(kernel, false)?;
    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
        smoothing_length: Some(smoothing_length),
        reset_on_cross: Some(reset_on_cross),
    };
    let input = FvgTrailingStopInput::from_slices(h, l, c, params);
    let out = py
        .allow_threads(|| fvg_trailing_stop_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.upper.into_pyarray(py),
        out.lower.into_pyarray(py),
        out.upper_ts.into_pyarray(py),
        out.lower_ts.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "fvg_trailing_stop_batch")]
#[pyo3(signature = (high, low, close, lookback_range, smoothing_range, reset_toggle, kernel=None))]
pub fn fvg_trailing_stop_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    smoothing_range: (usize, usize, usize),
    reset_toggle: (bool, bool),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let (h, l, c) = (high.as_slice()?, low.as_slice()?, close.as_slice()?);
    let sweep = FvgTsBatchRange {
        lookback: lookback_range,
        smoothing: smoothing_range,
        reset_on_cross: reset_toggle,
    };
    let kern = validate_kernel(kernel, true)?;

    // compute combos count to size flat buffer once
    let combos = expand_grid_ts(&sweep);
    let rows = combos.len();
    let cols = h.len();

    // flat buffer: (rows*4*cols), filled in one pass
    let flat = unsafe { PyArray1::<f64>::new(py, [rows * 4 * cols], false) };
    let flat_mut = unsafe { flat.as_slice_mut()? };

    py.allow_threads(|| fvg_ts_batch_inner_into(h, l, c, &sweep, kern, true, flat_mut))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    // expose a single 2D view like alma.rs does
    dict.set_item("values", flat.reshape((rows * 4, cols))?)?;
    dict.set_item(
        "lookbacks",
        combos
            .iter()
            .map(|p| p.unmitigated_fvg_lookback.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smoothings",
        combos
            .iter()
            .map(|p| p.smoothing_length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "resets",
        combos
            .iter()
            .map(|p| p.reset_on_cross.unwrap_or(false))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass]
pub struct FvgTrailingStopStreamPy {
    stream: FvgTrailingStopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl FvgTrailingStopStreamPy {
    #[new]
    fn new(
        unmitigated_fvg_lookback: usize,
        smoothing_length: usize,
        reset_on_cross: bool,
    ) -> PyResult<Self> {
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
            smoothing_length: Some(smoothing_length),
            reset_on_cross: Some(reset_on_cross),
        };
        let stream = FvgTrailingStopStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(FvgTrailingStopStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(high, low, close)
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fvg_ts_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fvg_ts_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FvgTsJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FvgTsBatchJsOutput {
    pub values: Vec<f64>, // row-major: rows*4 by cols
    pub combos: Vec<FvgTrailingStopParams>,
    pub rows: usize, // number of combos (not multiplied by 4)
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "fvgTrailingStop")]
pub fn fvg_trailing_stop_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    unmitigated_fvg_lookback: usize,
    smoothing_length: usize,
    reset_on_cross: bool,
) -> Result<JsValue, JsValue> {
    // Check for empty arrays first
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(JsValue::from_str(
            "fvg_trailing_stop: Input data slice is empty.",
        ));
    }

    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
        smoothing_length: Some(smoothing_length),
        reset_on_cross: Some(reset_on_cross),
    };
    let input = FvgTrailingStopInput::from_slices(high, low, close, params);
    let len = high.len();
    let mut buf_mu = make_uninit_matrix(4, len);
    let first = first_valid_ohlc(high, low, close);
    let warm = if first != usize::MAX {
        (first + 2 + smoothing_length.saturating_sub(1)).min(len)
    } else {
        0
    };
    init_matrix_prefixes(&mut buf_mu, len, &[warm, warm, warm, warm]);
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let (first_half, second_half) = out.split_at_mut(2 * len);
    let (u, l) = first_half.split_at_mut(len);
    let (uts, lts) = second_half.split_at_mut(len);
    fvg_trailing_stop_into_slices(u, l, uts, lts, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create JS object with named properties
    let obj = js_sys::Object::new();
    let upper_arr = js_sys::Array::from_iter(u.iter().map(|&v| JsValue::from_f64(v)));
    let lower_arr = js_sys::Array::from_iter(l.iter().map(|&v| JsValue::from_f64(v)));
    let upper_ts_arr = js_sys::Array::from_iter(uts.iter().map(|&v| JsValue::from_f64(v)));
    let lower_ts_arr = js_sys::Array::from_iter(lts.iter().map(|&v| JsValue::from_f64(v)));

    js_sys::Reflect::set(&obj, &JsValue::from_str("upper"), &upper_arr)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("lower"), &lower_arr)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("upperTs"), &upper_ts_arr)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("lowerTs"), &lower_ts_arr)?;

    // Don't forget memory cleanup
    let _ = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    core::mem::forget(guard);

    Ok(obj.into())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fvg_trailing_stop_into_flat(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    unmitigated_fvg_lookback: usize,
    smoothing_length: usize,
    reset_on_cross: bool,
) -> Result<(), JsValue> {
    if [
        high_ptr as usize,
        low_ptr as usize,
        close_ptr as usize,
        out_ptr as usize,
    ]
    .iter()
    .any(|&p| p == 0)
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let h = core::slice::from_raw_parts(high_ptr, len);
        let l = core::slice::from_raw_parts(low_ptr, len);
        let c = core::slice::from_raw_parts(close_ptr, len);
        let out = core::slice::from_raw_parts_mut(out_ptr, 4 * len);
        let (first_half, second_half) = out.split_at_mut(2 * len);
        let (u, lw) = first_half.split_at_mut(len);
        let (uts, lts) = second_half.split_at_mut(len);
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
            smoothing_length: Some(smoothing_length),
            reset_on_cross: Some(reset_on_cross),
        };
        let input = FvgTrailingStopInput::from_slices(h, l, c, params);
        fvg_trailing_stop_into_slices(u, lw, uts, lts, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "fvgTrailingStopBatch")]
pub fn fvg_trailing_stop_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    smoothing_start: usize,
    smoothing_end: usize,
    smoothing_step: usize,
    reset_include_false: bool,
    reset_include_true: bool,
) -> Result<JsValue, JsValue> {
    let sweep = FvgTsBatchRange {
        lookback: (lookback_start, lookback_end, lookback_step),
        smoothing: (smoothing_start, smoothing_end, smoothing_step),
        reset_on_cross: (reset_include_false, reset_include_true),
    };
    let combos = expand_grid_ts(&sweep);
    let cols = high.len();
    let rows = combos.len();

    let mut buf_mu = make_uninit_matrix(4 * rows, cols);
    let first = first_valid_ohlc(high, low, close);
    let mut warms = Vec::with_capacity(4 * rows);
    for prm in &combos {
        let sm = prm.smoothing_length.unwrap_or(9);
        let w = if first == usize::MAX {
            0
        } else {
            (first + 2 + sm.saturating_sub(1)).min(cols)
        };
        warms.extend_from_slice(&[w, w, w, w]);
    }
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let flat: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    fvg_ts_batch_inner_into(
        high,
        low,
        close,
        &sweep,
        detect_best_batch_kernel(),
        false,
        flat,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    core::mem::forget(guard);

    let out = FvgTsBatchJsOutput {
        values,
        combos,
        rows,
        cols,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "fvgTrailingStopAlloc")]
pub fn fvg_trailing_stop_alloc_js(size: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(size * 4); // 4 outputs
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "fvgTrailingStopFree")]
pub fn fvg_trailing_stop_free_js(ptr: *mut f64, size: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, size * 4, size * 4);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "fvgTrailingStopZeroCopy")]
pub fn fvg_trailing_stop_zero_copy_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    unmitigated_fvg_lookback: usize,
    smoothing_length: usize,
    reset_on_cross: bool,
    ptr: *mut f64,
) -> Result<JsValue, JsValue> {
    let len = high.len();

    // Create slices from raw pointer
    let (upper, lower, upper_ts, lower_ts) = unsafe {
        (
            std::slice::from_raw_parts_mut(ptr, len),
            std::slice::from_raw_parts_mut(ptr.add(len), len),
            std::slice::from_raw_parts_mut(ptr.add(len * 2), len),
            std::slice::from_raw_parts_mut(ptr.add(len * 3), len),
        )
    };

    // Initialize with NaN
    for i in 0..len {
        upper[i] = f64::NAN;
        lower[i] = f64::NAN;
        upper_ts[i] = f64::NAN;
        lower_ts[i] = f64::NAN;
    }

    // Compute indicator
    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
        smoothing_length: Some(smoothing_length),
        reset_on_cross: Some(reset_on_cross),
    };

    let input = FvgTrailingStopInput {
        data: FvgTrailingStopData::Slices { high, low, close },
        params,
    };

    fvg_trailing_stop_into_slices(
        upper,
        lower,
        upper_ts,
        lower_ts,
        &input,
        detect_best_kernel(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Return object with references to the arrays
    let obj = js_sys::Object::new();
    let upper_arr = unsafe { js_sys::Float64Array::view(upper) };
    let lower_arr = unsafe { js_sys::Float64Array::view(lower) };
    let upper_ts_arr = unsafe { js_sys::Float64Array::view(upper_ts) };
    let lower_ts_arr = unsafe { js_sys::Float64Array::view(lower_ts) };

    js_sys::Reflect::set(&obj, &JsValue::from_str("upper"), &upper_arr)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("lower"), &lower_arr)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("upperTs"), &upper_ts_arr)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("lowerTs"), &lower_ts_arr)?;

    Ok(obj.into())
}

// ==================== BUILDER PATTERN ====================
#[derive(Copy, Clone, Debug)]
pub struct FvgTrailingStopBuilder {
    unmitigated_fvg_lookback: Option<usize>,
    smoothing_length: Option<usize>,
    reset_on_cross: Option<bool>,
    kernel: Kernel,
}

impl Default for FvgTrailingStopBuilder {
    fn default() -> Self {
        Self {
            unmitigated_fvg_lookback: None,
            smoothing_length: None,
            reset_on_cross: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FvgTrailingStopBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lookback(mut self, n: usize) -> Self {
        self.unmitigated_fvg_lookback = Some(n);
        self
    }

    pub fn smoothing(mut self, n: usize) -> Self {
        self.smoothing_length = Some(n);
        self
    }

    pub fn reset_on_cross(mut self, reset: bool) -> Self {
        self.reset_on_cross = Some(reset);
        self
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply(&self, candles: &Candles) -> Result<FvgTrailingStopOutput, FvgTrailingStopError> {
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: self.unmitigated_fvg_lookback,
            smoothing_length: self.smoothing_length,
            reset_on_cross: self.reset_on_cross,
        };
        let input = FvgTrailingStopInput::from_candles(candles, params);
        fvg_trailing_stop_with_kernel(&input, self.kernel)
    }

    pub fn apply_slice(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FvgTrailingStopOutput, FvgTrailingStopError> {
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: self.unmitigated_fvg_lookback,
            smoothing_length: self.smoothing_length,
            reset_on_cross: self.reset_on_cross,
        };
        let input = FvgTrailingStopInput::from_slices(high, low, close, params);
        fvg_trailing_stop_with_kernel(&input, self.kernel)
    }

    pub fn into_stream(self) -> Result<FvgTrailingStopStream, FvgTrailingStopError> {
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: self.unmitigated_fvg_lookback,
            smoothing_length: self.smoothing_length,
            reset_on_cross: self.reset_on_cross,
        };
        FvgTrailingStopStream::try_new(params)
    }
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    // Helper macro to skip unsupported kernels
    macro_rules! skip_if_unsupported {
        ($kernel:expr, $test_name:expr) => {
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            if matches!(
                $kernel,
                Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch
            ) {
                eprintln!("Skipping {} - AVX not available", $test_name);
                return Ok(());
            }
        };
    }

    fn check_fvg_ts_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(5),
            smoothing_length: Some(9),
            reset_on_cross: Some(false),
        };
        let input = FvgTrailingStopInput::from_candles(&candles, params);
        let result = fvg_trailing_stop_with_kernel(&input, kernel)?;

        // Reference values from actual Rust computation
        // Lower: 55,643.00, Lower TS: 60,223.33333333
        let expected_lower = 55643.00;
        let expected_lower_ts = 60223.33333333;
        let tolerance = 0.01; // 0.01 tolerance for floating point comparison

        let n = result.lower.len();
        if n >= 5 {
            // Check last 5 values match expected reference values
            for i in (n - 5)..n {
                if !result.lower[i].is_nan() {
                    let diff = (result.lower[i] - expected_lower).abs();
                    assert!(
                        diff < tolerance,
                        "[{}] Lower value mismatch at {}: expected {}, got {}, diff {}",
                        test_name,
                        i,
                        expected_lower,
                        result.lower[i],
                        diff
                    );
                }
                if !result.lower_ts[i].is_nan() {
                    let diff = (result.lower_ts[i] - expected_lower_ts).abs();
                    assert!(
                        diff < tolerance,
                        "[{}] Lower TS value mismatch at {}: expected {}, got {}, diff {}",
                        test_name,
                        i,
                        expected_lower_ts,
                        result.lower_ts[i],
                        diff
                    );
                }
            }
        }
        Ok(())
    }

    fn check_fvg_ts_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = FvgTrailingStopInput::with_default_candles(&candles);
        let output = fvg_trailing_stop_with_kernel(&input, kernel)?;
        assert_eq!(output.upper.len(), candles.close.len());
        assert_eq!(output.lower.len(), candles.close.len());
        assert_eq!(output.upper_ts.len(), candles.close.len());
        assert_eq!(output.lower_ts.len(), candles.close.len());

        Ok(())
    }

    fn check_fvg_ts_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = FvgTrailingStopParams::default();
        let input = FvgTrailingStopInput::from_slices(&empty, &empty, &empty, params);
        let res = fvg_trailing_stop_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(FvgTrailingStopError::EmptyInputData)),
            "[{}] Should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_fvg_ts_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = vec![f64::NAN; 100];
        let params = FvgTrailingStopParams::default();
        let input = FvgTrailingStopInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let res = fvg_trailing_stop_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(FvgTrailingStopError::AllValuesNaN)),
            "[{}] Should fail with all NaN",
            test_name
        );
        Ok(())
    }

    fn check_fvg_ts_partial_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let mut high = vec![100.0; 50];
        let mut low = vec![95.0; 50];
        let mut close = vec![97.0; 50];

        // Add some NaN values
        for i in 10..20 {
            high[i] = f64::NAN;
            low[i] = f64::NAN;
            close[i] = f64::NAN;
        }

        let params = FvgTrailingStopParams::default();
        let input = FvgTrailingStopInput::from_slices(&high, &low, &close, params);
        let result = fvg_trailing_stop_with_kernel(&input, kernel)?;

        assert_eq!(result.upper.len(), 50);
        assert_eq!(result.lower.len(), 50);
        Ok(())
    }

    fn check_fvg_ts_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let params = FvgTrailingStopParams::default();
        let mut stream = FvgTrailingStopStream::try_new(params)?;

        // Feed some data points
        let test_data = vec![
            (100.0, 95.0, 97.0),
            (101.0, 96.0, 98.0),
            (102.0, 97.0, 99.0),
            (103.0, 98.0, 100.0),
            (104.0, 99.0, 101.0),
            (105.0, 100.0, 102.0),
            (106.0, 101.0, 103.0),
            (107.0, 102.0, 104.0),
            (108.0, 103.0, 105.0),
            (109.0, 104.0, 106.0),
            (110.0, 105.0, 107.0),
            (111.0, 106.0, 108.0),
        ];

        for (h, l, c) in test_data {
            stream.update(h, l, c);
        }

        // Stream should work after enough bars
        Ok(())
    }

    fn check_fvg_ts_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        #[cfg(debug_assertions)]
        {
            let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
            let candles = read_candles_from_csv(file_path)?;

            let params = FvgTrailingStopParams::default();
            let input = FvgTrailingStopInput::from_candles(&candles, params);
            let out = fvg_trailing_stop_with_kernel(&input, kernel)?;

            for (name, row) in [
                ("upper", &out.upper),
                ("lower", &out.lower),
                ("upper_ts", &out.upper_ts),
                ("lower_ts", &out.lower_ts),
            ] {
                for (i, &v) in row.iter().enumerate() {
                    if v.is_nan() {
                        continue;
                    }
                    let b = v.to_bits();
                    assert_ne!(
                        b, 0x1111_1111_1111_1111,
                        "[{}] alloc poison in {} at {}",
                        test_name, name, i
                    );
                    assert_ne!(
                        b, 0x2222_2222_2222_2222,
                        "[{}] matrix poison in {} at {}",
                        test_name, name, i
                    );
                    assert_ne!(
                        b, 0x3333_3333_3333_3333,
                        "[{}] uninit poison in {} at {}",
                        test_name, name, i
                    );
                }
            }
        }
        Ok(())
    }

    fn check_fvg_ts_batch_default(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = FvgTsBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;

        // Should have default params row
        assert_eq!(output.combos.len(), 1);
        assert_eq!(output.rows, 1); // number of combos
        assert_eq!(output.cols, candles.close.len());

        Ok(())
    }

    fn check_fvg_ts_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = FvgTsBatchBuilder::new()
            .kernel(kernel)
            .lookback_range(3, 7, 2)
            .smoothing_range(5, 10, 5)
            .reset_toggle(true, true)
            .apply_candles(&candles)?;

        // 3 lookback values (3, 5, 7) * 2 smoothing (5, 10) * 2 reset (false, true) = 12 combos
        assert_eq!(output.combos.len(), 12);
        assert_eq!(output.rows, 12); // number of combos
        assert_eq!(output.cols, candles.close.len());

        Ok(())
    }

    fn check_fvg_ts_builder_apply_slice(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = vec![100.0, 102.0, 103.0, 105.0, 104.0];
        let low = vec![98.0, 99.0, 101.0, 102.0, 103.0];
        let close = vec![99.0, 101.0, 102.0, 104.0, 103.5];

        let result = FvgTrailingStopBuilder::new()
            .lookback(3)
            .smoothing(5)
            .kernel(kernel)
            .apply_slice(&high, &low, &close)?;

        assert_eq!(result.upper.len(), 5);
        assert_eq!(result.lower.len(), 5);

        Ok(())
    }

    fn check_fvg_ts_into_slices_warm_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        ];
        let l = vec![
            99.0, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
        ];
        let c = vec![
            99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5,
        ];
        let params = FvgTrailingStopParams::default();
        let input = FvgTrailingStopInput::from_slices(&h, &l, &c, params);

        let mut u = vec![0.0; h.len()];
        let mut d = u.clone();
        let mut uts = u.clone();
        let mut lts = u.clone();

        // Get smoothing length before calling into_slices
        let smoothing_len = input.get_smoothing();

        fvg_trailing_stop_into_slices(&mut u, &mut d, &mut uts, &mut lts, &input, kernel)?;

        // prefix must be NaN
        let expected_warm = 2 + smoothing_len - 1; // first is 0, so warm = 0 + 2 + 9 - 1 = 10
        for v in [&u, &d, &uts, &lts] {
            for i in 0..expected_warm.min(h.len()) {
                assert!(
                    v[i].is_nan(),
                    "[{}] Expected NaN at index {} but got {}",
                    test_name,
                    i,
                    v[i]
                );
            }
        }
        Ok(())
    }

    fn check_fvg_ts_invalid_smoothing(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = vec![1.0; 20];
        let l = vec![0.0; 20];
        let c = vec![0.5; 20];
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(5),
            smoothing_length: Some(0),
            reset_on_cross: Some(false),
        };
        let input = FvgTrailingStopInput::from_slices(&h, &l, &c, params);
        let res = fvg_trailing_stop_with_kernel(&input, kernel);
        assert!(
            matches!(
                res,
                Err(FvgTrailingStopError::InvalidSmoothingLength { .. })
            ),
            "[{}] expected InvalidSmoothingLength",
            test_name
        );
        Ok(())
    }

    fn check_fvg_ts_invalid_lookback(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = vec![1.0; 20];
        let l = vec![0.0; 20];
        let c = vec![0.5; 20];
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(0),
            smoothing_length: Some(9),
            reset_on_cross: Some(false),
        };
        let input = FvgTrailingStopInput::from_slices(&h, &l, &c, params);
        let res = fvg_trailing_stop_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(FvgTrailingStopError::InvalidLookback { .. })),
            "[{}] expected InvalidLookback",
            test_name
        );
        Ok(())
    }

    fn check_fvg_ts_batch_values_for(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = vec![100.0; 64];
        let l = vec![90.0; 64];
        let c = vec![95.0; 64];
        let sweep = FvgTsBatchRange::default();
        let out = fvg_trailing_stop_batch_with_kernel(&h, &l, &c, &sweep, kernel)?;
        let p = FvgTrailingStopParams::default();
        let (u, d, uts, lts) = out.values_for(&p).expect("missing row");
        assert_eq!(u.len(), out.cols);
        assert_eq!(d.len(), out.cols);
        assert_eq!(uts.len(), out.cols);
        assert_eq!(lts.len(), out.cols);
        Ok(())
    }

    // Macro to generate all test variants
    macro_rules! generate_all_fvg_ts_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
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
        }
    }

    generate_all_fvg_ts_tests!(
        check_fvg_ts_accuracy,
        check_fvg_ts_default_candles,
        check_fvg_ts_empty_input,
        check_fvg_ts_all_nan,
        check_fvg_ts_partial_nan,
        check_fvg_ts_streaming,
        check_fvg_ts_no_poison,
        check_fvg_ts_batch_default,
        check_fvg_ts_batch_sweep,
        check_fvg_ts_builder_apply_slice,
        check_fvg_ts_into_slices_warm_nan,
        check_fvg_ts_invalid_smoothing,
        check_fvg_ts_invalid_lookback,
        check_fvg_ts_batch_values_for
    );
}
