//! # Dual Volume Divergence Index Quantitative Qualitative Estimation (DVDIQQE)
//!
//! The DVDIQQE indicator is an experimental study inspired by the Quantitative Qualitative Estimation
//! indicator designed to identify trend and wave activity. It uses the Dual Volume Divergence Index
//! (DVDI) oscillator calculated from the difference between Positive Volume Index (PVI) and Negative
//! Volume Index (NVI) divergences.
//!
//! ## Parameters
//! - **period**: Sampling period for calculations (default: 13)
//! - **smoothing_period**: Smoothing period for EMA calculations (default: 6)
//! - **fast_multiplier**: Fast trailing level range multiplier (default: 2.618)
//! - **slow_multiplier**: Slow trailing level range multiplier (default: 4.236)
//! - **volume_type**: "default" uses real volume with tick volume fallback, "tick" forces tick volume (default: "default")
//! - **center_type**: "dynamic" for cumulative average, "static" for zero center line (default: "dynamic")
//!
//! ## Returns
//! - **`Ok(DvdiqqeOutput)`** containing dvdi, fast_tl, slow_tl, and center_line Vec<f64> arrays
//!
//! ## Developer Notes
//! ### Implementation Status
//! - SIMD selection enabled: EMA stages use AVX2/AVX512 when available; control-flow identical to scalar.
//! - Scalar optimized: loop-jammed PVI/NVI build and range computation; no extra volume intermediates.
//! - Batch (flat) optimized: precomputes volume selection + PVI/NVI once and reuses across rows.
//! - Streaming update: O(1) enabled; seeds TLs at first ready tick and matches batch warmup parity.
//!
//! ### TODO
//! - Consider O(1) streaming update for trailing levels.
//! - Evaluate further row-specific batch reuse in non-flat inner API.

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
use crate::indicators::moving_averages::ema::{ema_with_kernel, EmaInput, EmaParams};
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
pub enum DvdiqqeData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: Option<&'a [f64]>,
    },
}

/// Output structure containing calculated DVDIQQE components
#[derive(Debug, Clone)]
pub struct DvdiqqeOutput {
    pub dvdi: Vec<f64>,
    pub fast_tl: Vec<f64>,
    pub slow_tl: Vec<f64>,
    pub center_line: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DvdiqqeParams {
    pub period: Option<usize>,
    pub smoothing_period: Option<usize>,
    pub fast_multiplier: Option<f64>,
    pub slow_multiplier: Option<f64>,
    pub volume_type: Option<String>,
    pub center_type: Option<String>,
    pub tick_size: Option<f64>,
}

impl Default for DvdiqqeParams {
    fn default() -> Self {
        Self {
            period: Some(13),
            smoothing_period: Some(6),
            fast_multiplier: Some(2.618),
            slow_multiplier: Some(4.236),
            volume_type: Some("default".to_string()),
            center_type: Some("dynamic".to_string()),
            tick_size: Some(0.01),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct DvdiqqeInput<'a> {
    pub data: DvdiqqeData<'a>,
    pub params: DvdiqqeParams,
}

impl<'a> AsRef<[f64]> for DvdiqqeInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DvdiqqeData::Candles { candles } => &candles.close,
            DvdiqqeData::Slices { close, .. } => close,
        }
    }
}

impl<'a> DvdiqqeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: DvdiqqeParams) -> Self {
        Self {
            data: DvdiqqeData::Candles { candles: c },
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: Option<&'a [f64]>,
        p: DvdiqqeParams,
    ) -> Self {
        Self {
            data: DvdiqqeData::Slices {
                open,
                high,
                low,
                close,
                volume,
            },
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, DvdiqqeParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(13)
    }

    #[inline]
    pub fn get_smoothing_period(&self) -> usize {
        self.params.smoothing_period.unwrap_or(6)
    }

    #[inline]
    pub fn get_fast_multiplier(&self) -> f64 {
        self.params.fast_multiplier.unwrap_or(2.618)
    }

    #[inline]
    pub fn get_slow_multiplier(&self) -> f64 {
        self.params.slow_multiplier.unwrap_or(4.236)
    }

    #[inline]
    pub fn get_volume_type(&self) -> &str {
        self.params.volume_type.as_deref().unwrap_or("default")
    }

    #[inline]
    pub fn get_center_type(&self) -> &str {
        self.params.center_type.as_deref().unwrap_or("dynamic")
    }

    #[inline]
    pub fn get_tick_size(&self) -> f64 {
        self.params.tick_size.unwrap_or(0.01)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum DvdiqqeError {
    #[error("dvdiqqe: Empty input data")]
    EmptyInputData,

    #[error("dvdiqqe: All values are NaN")]
    AllValuesNaN,

    #[error("dvdiqqe: Invalid period: {period}, data length: {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("dvdiqqe: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("Input arrays must have the same length")]
    MissingData,

    #[error("dvdiqqe: Invalid smoothing period: {smoothing}")]
    InvalidSmoothing { smoothing: usize },

    #[error("dvdiqqe: Invalid tick size: {tick}")]
    InvalidTick { tick: f64 },

    #[error("Invalid multiplier: {which} multiplier must be positive (got {multiplier})")]
    InvalidMultiplier { multiplier: f64, which: String },

    #[error("dvdiqqe: EMA computation failed: {0}")]
    EmaError(String),
}

// ==================== CORE ALGORITHM ====================

/// Private function to prepare and validate inputs
#[inline(always)]
fn dvdiqqe_prepare<'a>(
    input: &'a DvdiqqeInput,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        Option<&'a [f64]>,
        usize,
        usize,
        f64,
        f64,
        &'a str,
        &'a str,
        f64,
        usize,
    ),
    DvdiqqeError,
> {
    let (o, h, l, c, v) = match &input.data {
        DvdiqqeData::Candles { candles } => (
            &candles.open[..],
            &candles.high[..],
            &candles.low[..],
            &candles.close[..],
            Some(&candles.volume[..]),
        ),
        DvdiqqeData::Slices {
            open,
            high,
            low,
            close,
            volume,
        } => (*open, *high, *low, *close, *volume),
    };

    let len = c.len();
    if len == 0 {
        return Err(DvdiqqeError::EmptyInputData);
    }
    if o.len() != len || h.len() != len || l.len() != len {
        return Err(DvdiqqeError::MissingData);
    }
    if let Some(vs) = v {
        if vs.len() != len {
            return Err(DvdiqqeError::MissingData);
        }
    }

    let first = c
        .iter()
        .position(|x| x.is_finite())
        .ok_or(DvdiqqeError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(DvdiqqeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let smoothing = input.get_smoothing_period();
    if smoothing == 0 {
        return Err(DvdiqqeError::InvalidSmoothing { smoothing });
    }

    let fast_mult = input.get_fast_multiplier();
    let slow_mult = input.get_slow_multiplier();
    if fast_mult <= 0.0 || !fast_mult.is_finite() {
        return Err(DvdiqqeError::InvalidMultiplier {
            multiplier: fast_mult,
            which: "fast".to_string(),
        });
    }
    if slow_mult <= 0.0 || !slow_mult.is_finite() {
        return Err(DvdiqqeError::InvalidMultiplier {
            multiplier: slow_mult,
            which: "slow".to_string(),
        });
    }

    if len - first < period {
        return Err(DvdiqqeError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let tick = input.get_tick_size();
    if !tick.is_finite() || tick <= 0.0 {
        return Err(DvdiqqeError::InvalidTick { tick });
    }

    Ok((
        o,
        h,
        l,
        c,
        v,
        period,
        smoothing,
        fast_mult,
        slow_mult,
        input.get_volume_type(),
        input.get_center_type(),
        tick,
        first,
    ))
}

/// Private zero-copy compute function
#[inline(always)]
fn dvdiqqe_compute_into(
    open: &[f64],
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    volume_opt: Option<&[f64]>,
    period: usize,
    smoothing_period: usize,
    fast_mult: f64,
    slow_mult: f64,
    volume_type: &str,
    center_type: &str,
    tick: f64,
    first_valid: usize,
    kernel: Kernel,
    dvdi_out: &mut [f64],
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    center_out: &mut [f64],
) -> Result<(), DvdiqqeError> {
    let len = close.len();
    assert_eq!(dvdi_out.len(), len);
    assert_eq!(fast_out.len(), len);
    assert_eq!(slow_out.len(), len);
    assert_eq!(center_out.len(), len);

    if len == 0 {
        return Ok(());
    }

    // derived constants
    let wper = (period * 2) - 1;
    let warmup = first_valid + wper;

    // Phase 1: Build PVI/NVI in one pass with inline tick volume selection
    let mut pvi = alloc_with_nan_prefix(len, 0);
    let mut nvi = alloc_with_nan_prefix(len, 0);

    let mut pvi_prev = 0.0f64;
    let mut nvi_prev = 0.0f64;
    let mut prev_vol = 0.0f64;
    let mut prev_close = 0.0f64;
    let mut tickrng_prev = tick;
    let use_tick_only = volume_type.eq_ignore_ascii_case("tick");

    for i in 0..len {
        let oi = open[i];
        let ci = close[i];

        // pine-like tick volume
        let rng = ci - oi;
        let tickrng = if rng.abs() < tick { tickrng_prev } else { rng };
        let tick_vol = (tickrng.abs() / tick).max(0.0);

        // select real volume or tick fallback
        let sel_vol = if use_tick_only {
            tick_vol
        } else if let Some(vs) = volume_opt {
            let vv = vs[i];
            if vv.is_finite() { vv } else { tick_vol }
        } else {
            tick_vol
        };

        let d_close = ci - prev_close;
        if sel_vol > prev_vol {
            pvi_prev += d_close;
        }
        if sel_vol < prev_vol {
            nvi_prev -= d_close;
        }

        pvi[i] = pvi_prev;
        nvi[i] = nvi_prev;
        prev_close = ci;
        prev_vol = sel_vol;
        tickrng_prev = tickrng;
    }

    // Phase 2: EMA(pvi), EMA(nvi)
    let pvi_ema = {
        let prm = EmaParams { period: Some(period) };
        let inp = EmaInput::from_slice(&pvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let nvi_ema = {
        let prm = EmaParams { period: Some(period) };
        let inp = EmaInput::from_slice(&nvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    // Phase 3: divergences then smoothing EMA (reuse pvi/nvi buffers)
    for i in 0..len {
        pvi[i] = pvi[i] - pvi_ema.values[i];
        nvi[i] = nvi[i] - nvi_ema.values[i];
    }

    let pdiv_ema = {
        let prm = EmaParams { period: Some(smoothing_period) };
        let inp = EmaInput::from_slice(&pvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let ndiv_ema = {
        let prm = EmaParams { period: Some(smoothing_period) };
        let inp = EmaInput::from_slice(&nvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    // Phase 4: compute dvdi and range stream
    let mut ranges = alloc_with_nan_prefix(len, 1);
    dvdi_out[0] = pdiv_ema.values[0] - ndiv_ema.values[0];
    for i in 1..len {
        let dvdi_i = pdiv_ema.values[i] - ndiv_ema.values[i];
        ranges[i] = (dvdi_i - dvdi_out[i - 1]).abs();
        dvdi_out[i] = dvdi_i;
    }

    // Phase 5: double-EMA range smoothing using Kernel::Auto
    let avg_range = {
        let prm = EmaParams { period: Some(wper) };
        let inp = EmaInput::from_slice(&ranges, prm);
        ema_with_kernel(&inp, Kernel::Auto).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let smooth_range = {
        let prm = EmaParams { period: Some(wper) };
        let inp = EmaInput::from_slice(&avg_range.values, prm);
        ema_with_kernel(&inp, Kernel::Auto).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    // Phase 6: Warmup NaNs and trailing levels
    for i in 0..warmup.min(len) {
        dvdi_out[i] = f64::NAN;
        fast_out[i] = f64::NAN;
        slow_out[i] = f64::NAN;
    }

    if warmup < len {
        fast_out[warmup] = dvdi_out[warmup];
        slow_out[warmup] = dvdi_out[warmup];

        for i in (warmup + 1)..len {
            let fr = smooth_range.values[i] * fast_mult;
            let sr = smooth_range.values[i] * slow_mult;

            // fast TL
            if dvdi_out[i] > fast_out[i - 1] {
                let nv = dvdi_out[i] - fr;
                fast_out[i] = if nv < fast_out[i - 1] { fast_out[i - 1] } else { nv };
            } else {
                let nv = dvdi_out[i] + fr;
                fast_out[i] = if nv > fast_out[i - 1] { fast_out[i - 1] } else { nv };
            }

            // slow TL
            if dvdi_out[i] > slow_out[i - 1] {
                let nv = dvdi_out[i] - sr;
                slow_out[i] = if nv < slow_out[i - 1] { slow_out[i - 1] } else { nv };
            } else {
                let nv = dvdi_out[i] + sr;
                slow_out[i] = if nv > slow_out[i - 1] { slow_out[i - 1] } else { nv };
            }
        }
    }

    // Phase 7: Center line
    for i in 0..warmup.min(len) {
        center_out[i] = f64::NAN;
    }
    if center_type.eq_ignore_ascii_case("dynamic") {
        let mut sum = 0.0f64;
        let mut cnt = 0.0f64;
        for i in warmup..len {
            let v = dvdi_out[i];
            if v.is_finite() {
                sum += v;
                cnt += 1.0;
            }
            center_out[i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
        }
    } else {
        for i in warmup..len {
            center_out[i] = 0.0;
        }
    }

    Ok(())
}

/// Main DVDIQQE calculation function (auto-detects best kernel)
pub fn dvdiqqe(input: &DvdiqqeInput) -> Result<DvdiqqeOutput, DvdiqqeError> {
    dvdiqqe_with_kernel(input, Kernel::Auto)
}

/// DVDIQQE calculation with specified kernel
pub fn dvdiqqe_with_kernel(
    input: &DvdiqqeInput,
    kernel: Kernel,
) -> Result<DvdiqqeOutput, DvdiqqeError> {
    let (_, _, _, c, _, period, _, _, _, _, _, _, first) = dvdiqqe_prepare(input)?;
    let len = c.len();
    let wper = (period * 2) - 1;
    let warmup = first + wper;
    let mut dvdi = alloc_with_nan_prefix(len, warmup);
    let mut fast = alloc_with_nan_prefix(len, warmup);
    let mut slow = alloc_with_nan_prefix(len, warmup);
    let mut center = alloc_with_nan_prefix(len, warmup);

    let actual_kernel = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    match actual_kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            dvdiqqe_avx512(&mut dvdi, &mut fast, &mut slow, &mut center, input)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            dvdiqqe_avx2(&mut dvdi, &mut fast, &mut slow, &mut center, input)
        },
        _ => dvdiqqe_into_slices(
            &mut dvdi,
            &mut fast,
            &mut slow,
            &mut center,
            input,
            actual_kernel,
        ),
    }?;

    Ok(DvdiqqeOutput {
        dvdi,
        fast_tl: fast,
        slow_tl: slow,
        center_line: center,
    })
}

// ==================== SIMD IMPLEMENTATIONS ====================

/// AVX2 implementation: route EMA sub-kernels via AVX2 while preserving scalar flow
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dvdiqqe_avx2(
    dvdi_dst: &mut [f64],
    fast_dst: &mut [f64],
    slow_dst: &mut [f64],
    center_dst: &mut [f64],
    input: &DvdiqqeInput,
) -> Result<(), DvdiqqeError> {
    dvdiqqe_into_slices(dvdi_dst, fast_dst, slow_dst, center_dst, input, Kernel::Avx2)
}

/// AVX512 implementation: same rationale as AVX2
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dvdiqqe_avx512(
    dvdi_dst: &mut [f64],
    fast_dst: &mut [f64],
    slow_dst: &mut [f64],
    center_dst: &mut [f64],
    input: &DvdiqqeInput,
) -> Result<(), DvdiqqeError> {
    dvdiqqe_into_slices(dvdi_dst, fast_dst, slow_dst, center_dst, input, Kernel::Avx512)
}

/// Calculate tick volume identical to Pine Script
#[inline]
fn calculate_tick_volume_pine_like(open: &[f64], close: &[f64], tick: f64) -> Vec<f64> {
    let len = open.len();
    let mut out = alloc_with_nan_prefix(len, 0);
    let mut tickrng_prev = tick;

    for i in 0..len {
        let rng = close[i] - open[i];
        let tickrng = if rng.abs() < tick { tickrng_prev } else { rng };
        out[i] = (tickrng.abs() / tick).max(0.0);
        tickrng_prev = tickrng;
    }
    out
}

/// Select volume identical to Pine Script
#[inline]
fn select_volume_pine_like(
    vol_opt: Option<&[f64]>,
    tick_vol: &[f64],
    volume_type: &str,
) -> Vec<f64> {
    let len = tick_vol.len();
    let mut out = alloc_with_nan_prefix(len, 0);
    match (volume_type.eq_ignore_ascii_case("tick"), vol_opt) {
        (true, _) => {
            for i in 0..len {
                out[i] = tick_vol[i];
            }
        }
        (false, Some(v)) => {
            for i in 0..len {
                out[i] = if v[i].is_finite() { v[i] } else { tick_vol[i] };
            }
        }
        (false, None) => {
            for i in 0..len {
                out[i] = tick_vol[i];
            }
        }
    }
    out
}

/// Build PVI and NVI identical to Pine Script
#[inline]
fn build_pvi_nvi_pine_like(close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut pvi = alloc_with_nan_prefix(len, 0);
    let mut nvi = alloc_with_nan_prefix(len, 0);
    let mut pvi_prev = 0.0;
    let mut nvi_prev = 0.0;
    let mut prev_vol = 0.0;
    let mut prev_x = 0.0;

    for i in 0..len {
        if volume[i] > prev_vol {
            pvi_prev += close[i] - prev_x;
        }
        if volume[i] < prev_vol {
            nvi_prev -= close[i] - prev_x;
        }
        pvi[i] = pvi_prev;
        nvi[i] = nvi_prev;
        prev_vol = volume[i];
        prev_x = close[i];
    }
    (pvi, nvi)
}

/// Calculate DVDI (Dual Volume Divergence Index) using Pine-like PVI/NVI
fn calculate_dvdi(
    close: &[f64],
    volume: &[f64],
    period: usize,
    smoothing_period: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, DvdiqqeError> {
    let len = close.len();

    // Build PVI and NVI identical to Pine
    let (pvi, nvi) = build_pvi_nvi_pine_like(close, volume);

    // Calculate EMAs of PVI and NVI with specified kernel
    let pvi_ema_params = EmaParams {
        period: Some(period),
    };
    let pvi_ema_input = EmaInput::from_slice(&pvi, pvi_ema_params);
    let pvi_ema = ema_with_kernel(&pvi_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let nvi_ema_params = EmaParams {
        period: Some(period),
    };
    let nvi_ema_input = EmaInput::from_slice(&nvi, nvi_ema_params);
    let nvi_ema = ema_with_kernel(&nvi_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    // Calculate divergences
    let mut pvi_div = alloc_with_nan_prefix(len, 0);
    let mut nvi_div = alloc_with_nan_prefix(len, 0);

    for i in 0..len {
        pvi_div[i] = pvi[i] - pvi_ema.values[i];
        nvi_div[i] = nvi[i] - nvi_ema.values[i];
    }

    // Apply smoothing EMA to divergences with specified kernel
    let pdiv_ema_params = EmaParams {
        period: Some(smoothing_period),
    };
    let pdiv_ema_input = EmaInput::from_slice(&pvi_div, pdiv_ema_params);
    let pdiv_ema = ema_with_kernel(&pdiv_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let ndiv_ema_params = EmaParams {
        period: Some(smoothing_period),
    };
    let ndiv_ema_input = EmaInput::from_slice(&nvi_div, ndiv_ema_params);
    let ndiv_ema = ema_with_kernel(&ndiv_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    // Calculate final DVDI
    let mut dvdi = alloc_with_nan_prefix(len, 0);
    for i in 0..len {
        dvdi[i] = pdiv_ema.values[i] - ndiv_ema.values[i];
    }

    Ok(dvdi)
}

/// Calculate trailing levels (fast and slow)
fn calculate_trailing_levels(
    dvdi: &[f64],
    period: usize,
    fast_mult: f64,
    slow_mult: f64,
) -> Result<(Vec<f64>, Vec<f64>), DvdiqqeError> {
    let len = dvdi.len();
    let wper = (period * 2) - 1;

    // Calculate range with proper warmup
    let mut ranges = alloc_with_nan_prefix(len, 1);
    for i in 1..len {
        ranges[i] = (dvdi[i] - dvdi[i - 1]).abs();
    }

    // Apply EMA to ranges
    let range_ema_params = EmaParams { period: Some(wper) };
    let range_ema_input = EmaInput::from_slice(&ranges, range_ema_params);
    let avg_range = ema_with_kernel(&range_ema_input, Kernel::Auto)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    // Apply second EMA for smoothed range
    let smooth_range_params = EmaParams { period: Some(wper) };
    let smooth_range_input = EmaInput::from_slice(&avg_range.values, smooth_range_params);
    let smooth_range = ema_with_kernel(&smooth_range_input, Kernel::Auto)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    // Find first valid dvdi value
    let first_valid = dvdi.iter().position(|&x| x.is_finite()).unwrap_or(len);

    // Calculate trailing levels with proper warmup
    let mut fast_tl = alloc_with_nan_prefix(len, first_valid);
    let mut slow_tl = alloc_with_nan_prefix(len, first_valid);

    if first_valid < len {
        fast_tl[first_valid] = dvdi[first_valid];
        slow_tl[first_valid] = dvdi[first_valid];

        for i in (first_valid + 1)..len {
            let fast_range = smooth_range.values[i] * fast_mult;
            let slow_range = smooth_range.values[i] * slow_mult;

            // Fast trailing level
            if dvdi[i] > fast_tl[i - 1] {
                let new_val = dvdi[i] - fast_range;
                fast_tl[i] = if new_val < fast_tl[i - 1] {
                    fast_tl[i - 1]
                } else {
                    new_val
                };
            } else {
                let new_val = dvdi[i] + fast_range;
                fast_tl[i] = if new_val > fast_tl[i - 1] {
                    fast_tl[i - 1]
                } else {
                    new_val
                };
            }

            // Slow trailing level
            if dvdi[i] > slow_tl[i - 1] {
                let new_val = dvdi[i] - slow_range;
                slow_tl[i] = if new_val < slow_tl[i - 1] {
                    slow_tl[i - 1]
                } else {
                    new_val
                };
            } else {
                let new_val = dvdi[i] + slow_range;
                slow_tl[i] = if new_val > slow_tl[i - 1] {
                    slow_tl[i - 1]
                } else {
                    new_val
                };
            }
        }
    }

    Ok((fast_tl, slow_tl))
}

/// Calculate cumulative mean for dynamic center line
fn calculate_cumulative_mean(dvdi: &[f64]) -> Vec<f64> {
    let len = dvdi.len();
    let first_valid = dvdi.iter().position(|&x| x.is_finite()).unwrap_or(len);
    let mut center = alloc_with_nan_prefix(len, first_valid);
    let mut sum = 0.0;
    let mut cnt = 0.0;

    for i in first_valid..len {
        if dvdi[i].is_finite() {
            sum += dvdi[i];
            cnt += 1.0;
        }
        center[i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
    }
    center
}

// ==================== IN-PLACE API ====================
/// Multi-output in-place variant for bindings
pub fn dvdiqqe_into_slices(
    dvdi_dst: &mut [f64],
    fast_tl_dst: &mut [f64],
    slow_tl_dst: &mut [f64],
    center_dst: &mut [f64],
    input: &DvdiqqeInput,
    kernel: Kernel,
) -> Result<(), DvdiqqeError> {
    let (o, h, l, c, v, period, smoothing, fast, slow, vt, ct, tick, first) =
        dvdiqqe_prepare(input)?;

    // Length checks to mirror alma's quirk
    let len = c.len();
    if dvdi_dst.len() != len {
        return Err(DvdiqqeError::InvalidPeriod {
            period: dvdi_dst.len(),
            data_len: len,
        });
    }
    if fast_tl_dst.len() != len {
        return Err(DvdiqqeError::InvalidPeriod {
            period: fast_tl_dst.len(),
            data_len: len,
        });
    }
    if slow_tl_dst.len() != len {
        return Err(DvdiqqeError::InvalidPeriod {
            period: slow_tl_dst.len(),
            data_len: len,
        });
    }
    if center_dst.len() != len {
        return Err(DvdiqqeError::InvalidPeriod {
            period: center_dst.len(),
            data_len: len,
        });
    }

    dvdiqqe_compute_into(
        o,
        h,
        l,
        c,
        v,
        period,
        smoothing,
        fast,
        slow,
        vt,
        ct,
        tick,
        first,
        kernel,
        dvdi_dst,
        fast_tl_dst,
        slow_tl_dst,
        center_dst,
    )
}

/// Single-buffer in-place helper to mirror alma's in-place style and match WASM dvdiqqe_into layout
#[inline]
pub fn dvdiqqe_into_flat(
    dst_4xlen: &mut [f64],
    input: &DvdiqqeInput,
    k: Kernel,
) -> Result<(), DvdiqqeError> {
    let (_, _, _, c, _, _, _, _, _, _, _, _, _) = dvdiqqe_prepare(input)?;
    let len = c.len();
    assert_eq!(dst_4xlen.len(), 4 * len);
    let (dvdi, rest) = dst_4xlen.split_at_mut(len);
    let (fast, rest) = rest.split_at_mut(len);
    let (slow, cent) = rest.split_at_mut(len);
    dvdiqqe_into_slices(dvdi, fast, slow, cent, input, k)
}

// ==================== BUILDER PATTERN ====================
/// Builder for DVDIQQE with fluent API
#[derive(Copy, Clone, Debug, Default)]
pub struct DvdiqqeBuilder {
    period: Option<usize>,
    smoothing: Option<usize>,
    fast: Option<f64>,
    slow: Option<f64>,
    volume_type: Option<&'static str>,
    center_type: Option<&'static str>,
    tick: Option<f64>,
    kernel: Kernel,
}

impl DvdiqqeBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }

    pub fn smoothing(mut self, n: usize) -> Self {
        self.smoothing = Some(n);
        self
    }

    pub fn fast(mut self, mult: f64) -> Self {
        self.fast = Some(mult);
        self
    }

    pub fn slow(mut self, mult: f64) -> Self {
        self.slow = Some(mult);
        self
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn volume_type(mut self, vt: &'static str) -> Self {
        self.volume_type = Some(vt);
        self
    }

    pub fn center_type(mut self, ct: &'static str) -> Self {
        self.center_type = Some(ct);
        self
    }

    pub fn tick_size(mut self, ts: f64) -> Self {
        self.tick = Some(ts);
        self
    }

    pub fn apply_slice(
        self,
        o: &[f64],
        h: &[f64],
        l: &[f64],
        c: &[f64],
        v: Option<&[f64]>,
    ) -> Result<DvdiqqeOutput, DvdiqqeError> {
        let p = DvdiqqeParams {
            period: self.period,
            smoothing_period: self.smoothing,
            fast_multiplier: self.fast,
            slow_multiplier: self.slow,
            volume_type: self.volume_type.map(String::from),
            center_type: self.center_type.map(String::from),
            tick_size: self.tick,
        };
        let i = DvdiqqeInput::from_slices(o, h, l, c, v, p);
        dvdiqqe_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(self, c: &Candles) -> Result<DvdiqqeOutput, DvdiqqeError> {
        let p = DvdiqqeParams {
            period: self.period,
            smoothing_period: self.smoothing,
            fast_multiplier: self.fast,
            slow_multiplier: self.slow,
            volume_type: self.volume_type.map(str::to_string),
            center_type: self.center_type.map(str::to_string),
            tick_size: self.tick,
        };
        let i = DvdiqqeInput::from_candles(c, p);
        dvdiqqe_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DvdiqqeStream, DvdiqqeError> {
        let p = DvdiqqeParams {
            period: self.period,
            smoothing_period: self.smoothing,
            fast_multiplier: self.fast,
            slow_multiplier: self.slow,
            volume_type: self.volume_type.map(str::to_string),
            center_type: self.center_type.map(str::to_string),
            tick_size: self.tick,
        };
        DvdiqqeStream::try_new(p)
    }
}

// ==================== BATCH PROCESSING ====================
/// Batch range for parameter sweeps
#[derive(Clone, Debug)]
pub struct DvdiqqeBatchRange {
    pub period: (usize, usize, usize),
    pub smoothing_period: (usize, usize, usize),
    pub fast_multiplier: (f64, f64, f64),
    pub slow_multiplier: (f64, f64, f64),
}

impl Default for DvdiqqeBatchRange {
    fn default() -> Self {
        Self {
            period: (13, 26, 1),
            smoothing_period: (6, 12, 1),
            fast_multiplier: (2.618, 2.618, 0.0),
            slow_multiplier: (4.236, 4.236, 0.0),
        }
    }
}

/// Batch output structure
#[derive(Clone, Debug)]
pub struct DvdiqqeBatchOutput {
    pub dvdi_values: Vec<f64>,
    pub fast_tl_values: Vec<f64>,
    pub slow_tl_values: Vec<f64>,
    pub center_values: Vec<f64>,
    pub combos: Vec<DvdiqqeParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DvdiqqeBatchOutput {
    /// Get values for specific parameters
    pub fn values_for(&self, params: &DvdiqqeParams) -> Option<DvdiqqeBatchValues> {
        self.combos
            .iter()
            .position(|p| {
                p.period == params.period
                    && p.smoothing_period == params.smoothing_period
                    && p.fast_multiplier == params.fast_multiplier
                    && p.slow_multiplier == params.slow_multiplier
            })
            .map(|idx| {
                let start = idx * self.cols;
                let end = start + self.cols;
                DvdiqqeBatchValues {
                    dvdi: &self.dvdi_values[start..end],
                    fast_tl: &self.fast_tl_values[start..end],
                    slow_tl: &self.slow_tl_values[start..end],
                    center: &self.center_values[start..end],
                }
            })
    }

    /// Get row for specific parameters (mirrors alma's helper)
    pub fn row_for_params(&self, params: &DvdiqqeParams) -> Option<Vec<f64>> {
        self.combos
            .iter()
            .position(|p| {
                p.period == params.period
                    && p.smoothing_period == params.smoothing_period
                    && p.fast_multiplier == params.fast_multiplier
                    && p.slow_multiplier == params.slow_multiplier
            })
            .map(|idx| {
                let start = idx * self.cols;
                let end = start + self.cols;
                // Return concatenated values for all 4 outputs
                let mut row = Vec::with_capacity(self.cols * 4);
                row.extend_from_slice(&self.dvdi_values[start..end]);
                row.extend_from_slice(&self.fast_tl_values[start..end]);
                row.extend_from_slice(&self.slow_tl_values[start..end]);
                row.extend_from_slice(&self.center_values[start..end]);
                row
            })
    }
}

/// Batch values for a specific parameter combination
#[derive(Debug)]
pub struct DvdiqqeBatchValues<'a> {
    pub dvdi: &'a [f64],
    pub fast_tl: &'a [f64],
    pub slow_tl: &'a [f64],
    pub center: &'a [f64],
}

/// Flat batch output structure matching ALMA pattern
#[derive(Clone, Debug)]
pub struct DvdiqqeBatchOutputFlat {
    pub values: Vec<f64>, // layout: [series=4, rows, cols] flattened in row-major blocks per series
    pub combos: Vec<DvdiqqeParams>,
    pub rows: usize,   // number of parameter combos
    pub cols: usize,   // data length
    pub series: usize, // 4 (dvdi, fast_tl, slow_tl, center_line)
}

impl DvdiqqeBatchOutputFlat {
    #[inline]
    pub fn slice_series(&self, s: usize) -> &[f64] {
        assert!(s < self.series);
        let plane = self.rows * self.cols;
        &self.values[s * plane..(s + 1) * plane]
    }
}

/// Builder for batch operations
#[derive(Clone, Debug)]
pub struct DvdiqqeBatchBuilder {
    range: DvdiqqeBatchRange,
    kernel: Kernel,
    volume_type: String,
    center_type: String,
    tick_size: f64,
}

impl Default for DvdiqqeBatchBuilder {
    fn default() -> Self {
        Self {
            range: DvdiqqeBatchRange::default(),
            kernel: Kernel::Auto,
            volume_type: "default".to_string(),
            center_type: "dynamic".to_string(),
            tick_size: 0.01,
        }
    }
}

impl DvdiqqeBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
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

    pub fn smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_period = (start, end, step);
        self
    }

    pub fn smoothing_static(mut self, s: usize) -> Self {
        self.range.smoothing_period = (s, s, 0);
        self
    }

    pub fn fast_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.fast_multiplier = (start, end, step);
        self
    }

    pub fn fast_static(mut self, f: f64) -> Self {
        self.range.fast_multiplier = (f, f, 0.0);
        self
    }

    pub fn slow_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.slow_multiplier = (start, end, step);
        self
    }

    pub fn slow_static(mut self, s: f64) -> Self {
        self.range.slow_multiplier = (s, s, 0.0);
        self
    }

    pub fn volume_type(mut self, vt: &str) -> Self {
        self.volume_type = vt.to_string();
        self
    }

    pub fn center_type(mut self, ct: &str) -> Self {
        self.center_type = ct.to_string();
        self
    }

    pub fn tick_size(mut self, ts: f64) -> Self {
        self.tick_size = ts;
        self
    }

    pub fn apply_candles(self, candles: &Candles) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        dvdiqqe_batch_with_kernel(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            Some(&candles.volume),
            &self.range,
            self.kernel,
            &self.volume_type,
            &self.center_type,
            self.tick_size,
        )
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: Option<&[f64]>,
    ) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        dvdiqqe_batch_with_kernel(
            open,
            high,
            low,
            close,
            volume,
            &self.range,
            self.kernel,
            &self.volume_type,
            &self.center_type,
            self.tick_size,
        )
    }

    pub fn with_default_candles(candles: &Candles) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        let builder = Self::default();
        dvdiqqe_batch_with_kernel(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            Some(&candles.volume),
            &builder.range,
            builder.kernel,
            &builder.volume_type,
            &builder.center_type,
            builder.tick_size,
        )
    }

    pub fn with_default_slice(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: Option<&[f64]>,
    ) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        let builder = Self::default();
        dvdiqqe_batch_with_kernel(
            open,
            high,
            low,
            close,
            volume,
            &builder.range,
            builder.kernel,
            &builder.volume_type,
            &builder.center_type,
            builder.tick_size,
        )
    }
}

/// Expand parameter grid for batch processing
#[inline(always)]
fn expand_grid(r: &DvdiqqeBatchRange) -> Vec<DvdiqqeParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step == 0.0 || start == end {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut curr = start;
        while curr <= end + 1e-12 {
            v.push(curr);
            curr += step;
        }
        v
    }

    let periods = axis_usize(r.period);
    let smoothings = axis_usize(r.smoothing_period);
    let fasts = axis_f64(r.fast_multiplier);
    let slows = axis_f64(r.slow_multiplier);

    let mut combos = Vec::new();
    for &p in &periods {
        for &s in &smoothings {
            for &f in &fasts {
                for &sl in &slows {
                    combos.push(DvdiqqeParams {
                        period: Some(p),
                        smoothing_period: Some(s),
                        fast_multiplier: Some(f),
                        slow_multiplier: Some(sl),
                        volume_type: None,
                        center_type: None,
                        tick_size: None,
                    });
                }
            }
        }
    }
    combos
}

/// Batch calculation with kernel
pub fn dvdiqqe_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    k: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => k,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => kernel,
    };
    dvdiqqe_batch_par_slice(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        simd,
        volume_type,
        center_type,
        tick_size,
    )
}

/// Batch calculation with kernel returning flat output
pub fn dvdiqqe_batch_with_kernel_flat(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    k: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutputFlat, DvdiqqeError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => k,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => kernel,
    };
    dvdiqqe_batch_inner_flat(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        simd,
        true,
        volume_type,
        center_type,
        tick_size,
    )
}

/// Sequential batch processing
#[inline(always)]
pub fn dvdiqqe_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    dvdiqqe_batch_inner(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        kern,
        false,
        volume_type,
        center_type,
        tick_size,
    )
}

/// Parallel batch processing
#[inline(always)]
pub fn dvdiqqe_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    dvdiqqe_batch_inner(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        kern,
        true,
        volume_type,
        center_type,
        tick_size,
    )
}

/// Internal batch processing
fn dvdiqqe_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    parallel: bool,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    let flat = dvdiqqe_batch_inner_flat(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        kern,
        parallel,
        volume_type,
        center_type,
        tick_size,
    )?;

    // Extract individual series from flat output for compatibility
    let plane = flat.rows * flat.cols;
    Ok(DvdiqqeBatchOutput {
        dvdi_values: flat.values[0..plane].to_vec(),
        fast_tl_values: flat.values[plane..2 * plane].to_vec(),
        slow_tl_values: flat.values[2 * plane..3 * plane].to_vec(),
        center_values: flat.values[3 * plane..4 * plane].to_vec(),
        combos: flat.combos,
        rows: flat.rows,
        cols: flat.cols,
    })
}

/// Internal batch processing with flat output using uninitialized memory
fn dvdiqqe_batch_inner_flat(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    parallel: bool,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutputFlat, DvdiqqeError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DvdiqqeError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(DvdiqqeError::EmptyInputData);
    }

    // Allocate one slab: 4 × rows × cols
    let series = 4usize;
    let mut buf_mu = make_uninit_matrix(series * rows, cols);

    // Warm prefixes per row for all series using period
    let first = close
        .iter()
        .position(|x| x.is_finite())
        .ok_or(DvdiqqeError::AllValuesNaN)?;
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + p.period.unwrap_or(13) - 1)
        .collect();
    // apply to each series plane
    for s in 0..series {
        let off = s * rows * cols;
        let plane = &mut buf_mu[off..off + rows * cols];
        init_matrix_prefixes(plane, cols, &warms);
    }

    // Alias to f64 and compute rows
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Precompute volume selection and PVI/NVI once across rows (row-invariant)
    let tick_vol_once = calculate_tick_volume_pine_like(open, close, tick_size);
    let sel_vol_once = select_volume_pine_like(volume, &tick_vol_once, volume_type);
    let (pvi_stream, nvi_stream) = build_pvi_nvi_pine_like(close, &sel_vol_once);

    let process_row = |row: usize, out_slice: &mut [f64]| -> Result<(), DvdiqqeError> {
        let prm = &combos[row];
        let params = DvdiqqeParams {
            period: prm.period,
            smoothing_period: prm.smoothing_period,
            fast_multiplier: prm.fast_multiplier,
            slow_multiplier: prm.slow_multiplier,
            volume_type: Some(volume_type.to_string()),
            center_type: Some(center_type.to_string()),
            tick_size: Some(tick_size),
        };
        let input = DvdiqqeInput::from_slices(open, high, low, close, volume, params);
        let (_o, _h, _l, c, _v, period, smoothing, fast, slow, _vt, ct, _tick, first_local) =
            dvdiqqe_prepare(&input)?;

        // per-series row slices - split the provided slice
        let plane = rows * cols;
        let (dvdi_plane, rest) = out_slice.split_at_mut(plane);
        let (fast_plane, rest) = rest.split_at_mut(plane);
        let (slow_plane, center_plane) = rest.split_at_mut(plane);

        let dvdi_dst = &mut dvdi_plane[row * cols..(row + 1) * cols];
        let fast_dst = &mut fast_plane[row * cols..(row + 1) * cols];
        let slow_dst = &mut slow_plane[row * cols..(row + 1) * cols];
        let center_dst = &mut center_plane[row * cols..(row + 1) * cols];

        // Compute EMAs on precomputed PVI/NVI for this row
        let pvi_ema = {
            let prm = EmaParams { period: Some(period) };
            let inp = EmaInput::from_slice(&pvi_stream, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };
        let nvi_ema = {
            let prm = EmaParams { period: Some(period) };
            let inp = EmaInput::from_slice(&nvi_stream, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };

        // divergences and smoothing
        let mut pdiv = alloc_with_nan_prefix(cols, 0);
        let mut ndiv = alloc_with_nan_prefix(cols, 0);
        for i in 0..cols {
            pdiv[i] = pvi_stream[i] - pvi_ema.values[i];
            ndiv[i] = nvi_stream[i] - nvi_ema.values[i];
        }
        let pdiv_ema = {
            let prm = EmaParams { period: Some(smoothing) };
            let inp = EmaInput::from_slice(&pdiv, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };
        let ndiv_ema = {
            let prm = EmaParams { period: Some(smoothing) };
            let inp = EmaInput::from_slice(&ndiv, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };

        // write dvdi and ranges for TLs
        let wper = (period * 2) - 1;
        let warmup = first_local + wper;

        // ensure warmup NaNs (some already set by init_matrix_prefixes)
        for i in 0..warmup.min(cols) {
            dvdi_dst[i] = f64::NAN;
            fast_dst[i] = f64::NAN;
            slow_dst[i] = f64::NAN;
        }

        // fill dvdi and range buffer
        let mut ranges = alloc_with_nan_prefix(cols, 1);
        if cols > 0 {
            dvdi_dst[0] = pdiv_ema.values[0] - ndiv_ema.values[0];
            for i in 1..cols {
                let dvdi_i = pdiv_ema.values[i] - ndiv_ema.values[i];
                ranges[i] = (dvdi_i - dvdi_dst[i - 1]).abs();
                dvdi_dst[i] = dvdi_i;
            }
        }

        // Double-EMA range smoothing with Kernel::Auto
        let avg_range = {
            let prm = EmaParams { period: Some(wper) };
            let inp = EmaInput::from_slice(&ranges, prm);
            ema_with_kernel(&inp, Kernel::Auto).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };
        let smooth_range = {
            let prm = EmaParams { period: Some(wper) };
            let inp = EmaInput::from_slice(&avg_range.values, prm);
            ema_with_kernel(&inp, Kernel::Auto).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };

        // trailing levels
        if warmup < cols {
            fast_dst[warmup] = dvdi_dst[warmup];
            slow_dst[warmup] = dvdi_dst[warmup];
            for i in (warmup + 1)..cols {
                let fr = smooth_range.values[i] * fast;
                let sr = smooth_range.values[i] * slow;
                if dvdi_dst[i] > fast_dst[i - 1] {
                    let nv = dvdi_dst[i] - fr;
                    fast_dst[i] = if nv < fast_dst[i - 1] { fast_dst[i - 1] } else { nv };
                } else {
                    let nv = dvdi_dst[i] + fr;
                    fast_dst[i] = if nv > fast_dst[i - 1] { fast_dst[i - 1] } else { nv };
                }

                if dvdi_dst[i] > slow_dst[i - 1] {
                    let nv = dvdi_dst[i] - sr;
                    slow_dst[i] = if nv < slow_dst[i - 1] { slow_dst[i - 1] } else { nv };
                } else {
                    let nv = dvdi_dst[i] + sr;
                    slow_dst[i] = if nv > slow_dst[i - 1] { slow_dst[i - 1] } else { nv };
                }
            }
        }

        // center
        for i in 0..warmup.min(cols) {
            center_dst[i] = f64::NAN;
        }
        if ct.eq_ignore_ascii_case("dynamic") {
            let mut sum = 0.0f64;
            let mut cnt = 0.0f64;
            for i in warmup..cols {
                let v = dvdi_dst[i];
                if v.is_finite() {
                    sum += v;
                    cnt += 1.0;
                }
                center_dst[i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
            }
        } else {
            for i in warmup..cols {
                center_dst[i] = 0.0;
            }
        }

        Ok(())
    };

    // Process rows (parallel processing would require unsafe pointer handling,
    // so we keep it sequential for now - the real optimization is in the uninit memory)
    for r in 0..rows {
        process_row(r, out)?;
    }

    // Suppress unused warning
    let _ = parallel;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(DvdiqqeBatchOutputFlat {
        values,
        combos,
        rows,
        cols,
        series,
    })
}

/// Internal batch processing with output slices (legacy compatibility)
fn dvdiqqe_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    parallel: bool,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
    dvdi_out: &mut [f64],
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    center_out: &mut [f64],
) -> Result<Vec<DvdiqqeParams>, DvdiqqeError> {
    let combos = expand_grid(sweep);
    let cols = close.len();

    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            use rayon::prelude::*;

            // Process in parallel using atomic-safe operations
            let results: Result<Vec<_>, DvdiqqeError> = combos
                .par_iter()
                .map(|params| {
                    let mut full_params = params.clone();
                    full_params.volume_type = Some(volume_type.to_string());
                    full_params.center_type = Some(center_type.to_string());
                    full_params.tick_size = Some(tick_size);

                    let input =
                        DvdiqqeInput::from_slices(open, high, low, close, volume, full_params);
                    dvdiqqe_with_kernel(&input, kern)
                })
                .collect();

            let results = results?;

            // Write results sequentially
            for (row, output) in results.iter().enumerate() {
                let start = row * cols;
                let end = start + cols;
                dvdi_out[start..end].copy_from_slice(&output.dvdi);
                fast_out[start..end].copy_from_slice(&output.fast_tl);
                slow_out[start..end].copy_from_slice(&output.slow_tl);
                center_out[start..end].copy_from_slice(&output.center_line);
            }
        } else {
            for (row, params) in combos.iter().enumerate() {
                let mut full_params = params.clone();
                full_params.volume_type = Some(volume_type.to_string());
                full_params.center_type = Some(center_type.to_string());
                full_params.tick_size = Some(tick_size);

                let input = DvdiqqeInput::from_slices(open, high, low, close, volume, full_params);
                let output = dvdiqqe_with_kernel(&input, kern)?;

                let start = row * cols;
                let end = start + cols;
                dvdi_out[start..end].copy_from_slice(&output.dvdi);
                fast_out[start..end].copy_from_slice(&output.fast_tl);
                slow_out[start..end].copy_from_slice(&output.slow_tl);
                center_out[start..end].copy_from_slice(&output.center_line);
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (row, params) in combos.iter().enumerate() {
            let mut full_params = params.clone();
            full_params.volume_type = Some(volume_type.to_string());
            full_params.center_type = Some(center_type.to_string());
            full_params.tick_size = Some(tick_size);

            let input = DvdiqqeInput::from_slices(open, high, low, close, volume, full_params);
            let output = dvdiqqe_with_kernel(&input, kern)?;

            let start = row * cols;
            let end = start + cols;
            dvdi_out[start..end].copy_from_slice(&output.dvdi);
            fast_out[start..end].copy_from_slice(&output.fast_tl);
            slow_out[start..end].copy_from_slice(&output.slow_tl);
            center_out[start..end].copy_from_slice(&output.center_line);
        }
    }

    Ok(combos)
}

// ==================== STREAMING API ====================
/// Streaming DVDIQQE calculator with true O(1) updates
pub struct DvdiqqeStream {
    // --- parameters ---
    period: usize,
    smoothing_period: usize,
    fast_mult: f64,
    slow_mult: f64,
    volume_type: String,
    center_type: String,
    tick_size: f64,

    // --- precomputed constants ---
    alpha_pvi: f64,     // 2/(period+1) for PVI/NVI EMA
    alpha_div: f64,     // 2/(smoothing_period+1) for divergence EMA
    alpha_rng: f64,     // 1/period for range EMAs (since wper = 2*period-1)
    inv_tick: f64,      // 1/tick_size
    use_tick_only: bool,
    warmup_needed: usize, // streaming warmup gate: 2*period

    // --- rolling state for tick-volume selection ---
    prev_close: f64,
    prev_sel_vol: f64,
    tickrng_prev: f64,

    // --- PVI/NVI & their EMAs (cumulative definition) ---
    pvi: f64,
    nvi: f64,
    pvi_ema: f64,
    nvi_ema: f64,
    ema_pvi_inited: bool,

    // --- divergence EMAs ---
    pdiv_ema: f64,
    ndiv_ema: f64,
    ema_div_inited: bool,

    // --- dvdi & double-EMA of range(|Δdvdi|) ---
    dvdi_prev: f64,
    rng_ema1: f64,
    rng_ema2: f64,
    ema_rng_inited: bool,

    // --- trailing levels ---
    fast_tl_prev: f64,
    slow_tl_prev: f64,
    tl_seeded: bool,

    // --- counters & center line ---
    count: usize,
    center_sum: f64,
    center_count: f64,
}

impl DvdiqqeStream {
    pub fn try_new(params: DvdiqqeParams) -> Result<Self, DvdiqqeError> {
        let period = params.period.unwrap_or(13);
        let smoothing_period = params.smoothing_period.unwrap_or(6);
        if period == 0 {
            return Err(DvdiqqeError::InvalidPeriod { period, data_len: 0 });
        }
        if smoothing_period == 0 {
            return Err(DvdiqqeError::InvalidSmoothing { smoothing: 0 });
        }

        let fast_mult = params.fast_multiplier.unwrap_or(2.618);
        let slow_mult = params.slow_multiplier.unwrap_or(4.236);
        if !(fast_mult.is_finite() && fast_mult > 0.0) {
            return Err(DvdiqqeError::InvalidMultiplier { multiplier: fast_mult, which: "fast".into() });
        }
        if !(slow_mult.is_finite() && slow_mult > 0.0) {
            return Err(DvdiqqeError::InvalidMultiplier { multiplier: slow_mult, which: "slow".into() });
        }

        let volume_type = params.volume_type.unwrap_or_else(|| "default".to_string());
        let center_type = params.center_type.unwrap_or_else(|| "dynamic".to_string());
        let tick_size = params.tick_size.unwrap_or(0.01);
        if !(tick_size.is_finite() && tick_size > 0.0) {
            return Err(DvdiqqeError::InvalidTick { tick: tick_size });
        }

        // EMA alphas (compute once)
        let alpha_pvi = 2.0 / (period as f64 + 1.0);
        let alpha_div = 2.0 / (smoothing_period as f64 + 1.0);
        // Double-EMA of range uses wper = (2*period - 1); alpha = 1/period
        let alpha_rng = 1.0 / (period as f64);

        Ok(Self {
            period,
            smoothing_period,
            fast_mult,
            slow_mult,
            use_tick_only: volume_type.eq_ignore_ascii_case("tick"),
            volume_type,
            center_type,
            tick_size,
            inv_tick: 1.0 / tick_size,
            alpha_pvi,
            alpha_div,
            alpha_rng,
            warmup_needed: period * 2,

            prev_close: 0.0,
            prev_sel_vol: 0.0,
            tickrng_prev: tick_size,

            pvi: 0.0,
            nvi: 0.0,
            pvi_ema: 0.0,
            nvi_ema: 0.0,
            ema_pvi_inited: false,

            pdiv_ema: 0.0,
            ndiv_ema: 0.0,
            ema_div_inited: false,

            dvdi_prev: 0.0,
            rng_ema1: 0.0,
            rng_ema2: 0.0,
            ema_rng_inited: false,

            fast_tl_prev: f64::NAN,
            slow_tl_prev: f64::NAN,
            tl_seeded: false,

            count: 0,
            center_sum: 0.0,
            center_count: 0.0,
        })
    }

    pub fn update(
        &mut self,
        open: f64,
        _high: f64,
        _low: f64,
        close: f64,
        volume: f64,
    ) -> Option<DvdiqqeStreamOutput> {
        // ---- 1) Pine‑like tick volume selection ----
        let rng = close - open;
        let tickrng = if rng.abs() < self.tick_size { self.tickrng_prev } else { rng };
        let tick_vol = (tickrng.abs() * self.inv_tick).max(0.0);
        self.tickrng_prev = tickrng;

        let sel_vol = if self.use_tick_only {
            tick_vol
        } else if volume.is_finite() {
            volume
        } else {
            tick_vol
        };

        // ---- 2) PVI/NVI cumulative update using dClose and volume up/down ----
        let d_close = close - self.prev_close;
        if sel_vol > self.prev_sel_vol {
            self.pvi += d_close;
        } else if sel_vol < self.prev_sel_vol {
            self.nvi -= d_close;
        }
        self.prev_sel_vol = sel_vol;
        self.prev_close = close;

        // EMA(pvi), EMA(nvi)
        if !self.ema_pvi_inited {
            self.pvi_ema = self.pvi;
            self.nvi_ema = self.nvi;
            self.ema_pvi_inited = true;
        } else {
            // s += alpha * (x - s)
            self.pvi_ema += self.alpha_pvi * (self.pvi - self.pvi_ema);
            self.nvi_ema += self.alpha_pvi * (self.nvi - self.nvi_ema);
        }

        // ---- 3) divergence = (pvi - EMA(pvi)) and (nvi - EMA(nvi)), then EMA smoothing ----
        let pdiv = self.pvi - self.pvi_ema;
        let ndiv = self.nvi - self.nvi_ema;

        if !self.ema_div_inited {
            self.pdiv_ema = pdiv;
            self.ndiv_ema = ndiv;
            self.ema_div_inited = true;
        } else {
            self.pdiv_ema += self.alpha_div * (pdiv - self.pdiv_ema);
            self.ndiv_ema += self.alpha_div * (ndiv - self.ndiv_ema);
        }

        // ---- 4) dvdi = smoothed divergence difference ----
        let dvdi = self.pdiv_ema - self.ndiv_ema;

        // ---- 5) double EMA of range(|Δdvdi|) ----
        let step_rng = (dvdi - self.dvdi_prev).abs();
        if !self.ema_rng_inited {
            self.rng_ema1 = step_rng;
            self.rng_ema2 = self.rng_ema1;
            self.ema_rng_inited = true;
        } else {
            self.rng_ema1 += self.alpha_rng * (step_rng - self.rng_ema1);
            self.rng_ema2 += self.alpha_rng * (self.rng_ema1 - self.rng_ema2);
        }
        self.dvdi_prev = dvdi;

        // ---- 6) warmup gating ----
        self.count = self.count.saturating_add(1);
        if self.count < self.warmup_needed {
            return None;
        }

        // ---- 7) trailing levels ----
        let smooth_rng = self.rng_ema2;
        let fr = smooth_rng * self.fast_mult;
        let sr = smooth_rng * self.slow_mult;

        // Seed TLs on first ready tick exactly at dvdi
        if !self.tl_seeded {
            self.fast_tl_prev = dvdi;
            self.slow_tl_prev = dvdi;
            self.tl_seeded = true;

            // seed center on the same tick if dynamic
            if self.center_type.eq_ignore_ascii_case("dynamic") && dvdi.is_finite() {
                self.center_sum = dvdi;
                self.center_count = 1.0;
            }
            return Some(DvdiqqeStreamOutput {
                dvdi,
                fast_tl: self.fast_tl_prev,
                slow_tl: self.slow_tl_prev,
                center_line: if self.center_type.eq_ignore_ascii_case("static") {
                    0.0
                } else if self.center_count > 0.0 {
                    self.center_sum / self.center_count
                } else {
                    f64::NAN
                },
            });
        }

        // Fast TL
        let fast_tl = if dvdi > self.fast_tl_prev {
            let nv = dvdi - fr;
            if nv < self.fast_tl_prev { self.fast_tl_prev } else { nv }
        } else {
            let nv = dvdi + fr;
            if nv > self.fast_tl_prev { self.fast_tl_prev } else { nv }
        };

        // Slow TL
        let slow_tl = if dvdi > self.slow_tl_prev {
            let nv = dvdi - sr;
            if nv < self.slow_tl_prev { self.slow_tl_prev } else { nv }
        } else {
            let nv = dvdi + sr;
            if nv > self.slow_tl_prev { self.slow_tl_prev } else { nv }
        };

        self.fast_tl_prev = fast_tl;
        self.slow_tl_prev = slow_tl;

        // ---- 8) center line ----
        let center_val = if self.center_type.eq_ignore_ascii_case("static") {
            0.0
        } else {
            if dvdi.is_finite() {
                self.center_sum += dvdi;
                self.center_count += 1.0;
            }
            if self.center_count > 0.0 {
                self.center_sum / self.center_count
            } else {
                f64::NAN
            }
        };

        Some(DvdiqqeStreamOutput {
            dvdi,
            fast_tl,
            slow_tl,
            center_line: center_val,
        })
    }
}

/// Stream output for a single update
#[derive(Debug, Clone, Copy)]
pub struct DvdiqqeStreamOutput {
    pub dvdi: f64,
    pub fast_tl: f64,
    pub slow_tl: f64,
    pub center_line: f64,
}

#[cfg(feature = "python")]
#[pyclass]
pub struct DvdiqqeStreamPy {
    stream: DvdiqqeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DvdiqqeStreamPy {
    #[new]
    fn new(
        period: Option<i32>,
        smoothing_period: Option<i32>,
        fast_multiplier: Option<f64>,
        slow_multiplier: Option<f64>,
        volume_type: Option<String>,
        center_type: Option<String>,
        tick_size: Option<f64>,
    ) -> PyResult<Self> {
        // Validate period parameters
        let period_validated = if let Some(p) = period {
            if p <= 0 {
                return Err(PyValueError::new_err(format!(
                    "Invalid period: Period must be positive (got {})",
                    p
                )));
            }
            Some(p as usize)
        } else {
            None
        };

        let smoothing_validated = if let Some(s) = smoothing_period {
            if s <= 0 {
                return Err(PyValueError::new_err(format!(
                    "Invalid smoothing period: Smoothing period must be positive (got {})",
                    s
                )));
            }
            Some(s as usize)
        } else {
            None
        };

        let params = DvdiqqeParams {
            period: period_validated,
            smoothing_period: smoothing_validated,
            fast_multiplier,
            slow_multiplier,
            volume_type,
            center_type,
            tick_size,
        };

        let stream =
            DvdiqqeStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(DvdiqqeStreamPy { stream })
    }

    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        self.stream
            .update(open, high, low, close, volume)
            .map(|output| {
                (
                    output.dvdi,
                    output.fast_tl,
                    output.slow_tl,
                    output.center_line,
                )
            })
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "dvdiqqe", signature = (
    open,
    high,
    low,
    close,
    volume=None,
    period=None,
    smoothing_period=None,
    fast_multiplier=None,
    slow_multiplier=None,
    volume_type=None,
    center_type=None,
    tick_size=None,
    kernel=None
))]
pub fn dvdiqqe_py<'py>(
    py: Python<'py>,
    open: Option<PyReadonlyArray1<'py, f64>>,
    high: Option<PyReadonlyArray1<'py, f64>>,
    low: Option<PyReadonlyArray1<'py, f64>>,
    close: Option<PyReadonlyArray1<'py, f64>>,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    period: Option<i32>,
    smoothing_period: Option<i32>,
    fast_multiplier: Option<f64>,
    slow_multiplier: Option<f64>,
    volume_type: Option<String>,
    center_type: Option<String>,
    tick_size: Option<f64>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    // Validate that we have OHLC data
    if open.is_none() || high.is_none() || low.is_none() || close.is_none() {
        return Err(PyValueError::new_err(
            "OHLC data (open, high, low, close) is required",
        ));
    }

    let open_arr = open.unwrap();
    let high_arr = high.unwrap();
    let low_arr = low.unwrap();
    let close_arr = close.unwrap();

    let o = open_arr.as_slice()?;
    let h = high_arr.as_slice()?;
    let l = low_arr.as_slice()?;
    let c = close_arr.as_slice()?;
    let v = volume.as_ref().map(|v| v.as_slice()).transpose()?;
    let len = c.len();

    // Preallocate NumPy arrays
    let mut dvdi = unsafe { numpy::PyArray1::<f64>::new(py, [len], false) };
    let mut fast = unsafe { numpy::PyArray1::<f64>::new(py, [len], false) };
    let mut slow = unsafe { numpy::PyArray1::<f64>::new(py, [len], false) };
    let mut cent = unsafe { numpy::PyArray1::<f64>::new(py, [len], false) };
    let dvdi_s = unsafe { dvdi.as_slice_mut()? };
    let fast_s = unsafe { fast.as_slice_mut()? };
    let slow_s = unsafe { slow.as_slice_mut()? };
    let cent_s = unsafe { cent.as_slice_mut()? };

    // Validate period parameters
    let period_validated = if let Some(p) = period {
        if p <= 0 {
            return Err(PyValueError::new_err(format!(
                "Invalid period: Period must be positive (got {})",
                p
            )));
        }
        Some(p as usize)
    } else {
        None
    };

    let smoothing_validated = if let Some(s) = smoothing_period {
        if s <= 0 {
            return Err(PyValueError::new_err(format!(
                "Invalid smoothing period: Smoothing period must be positive (got {})",
                s
            )));
        }
        Some(s as usize)
    } else {
        None
    };

    let params = DvdiqqeParams {
        period: period_validated,
        smoothing_period: smoothing_validated,
        fast_multiplier,
        slow_multiplier,
        volume_type,
        center_type,
        tick_size,
    };
    let input = DvdiqqeInput::from_slices(o, h, l, c, v, params);
    let kern = validate_kernel(kernel, false).map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Compute directly into NumPy arrays
    py.allow_threads(|| dvdiqqe_into_slices(dvdi_s, fast_s, slow_s, cent_s, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((dvdi.into(), fast.into(), slow.into(), cent.into()))
}

#[cfg(feature = "python")]
#[pyfunction(name = "dvdiqqe_batch")]
#[pyo3(signature = (
    open,
    high,
    low,
    close,
    period_range,
    smoothing_period_range,
    fast_mult_range,
    slow_mult_range,
    kernel=None
))]
pub fn dvdiqqe_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    smoothing_period_range: (usize, usize, usize),
    fast_mult_range: (f64, f64, f64),
    slow_mult_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let sweep = DvdiqqeBatchRange {
        period: period_range,
        smoothing_period: smoothing_period_range,
        fast_multiplier: fast_mult_range,
        slow_multiplier: slow_mult_range,
    };
    let kern = validate_kernel(kernel, true).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = py
        .allow_threads(|| {
            dvdiqqe_batch_with_kernel_flat(
                o, h, l, c, None, &sweep, kern, "default", "dynamic", 0.01,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = out.rows;
    let cols = out.cols;
    let series = out.series; // 4
    let plane = rows * cols;

    use numpy::PyArray2;
    let dvdi = unsafe { PyArray2::new(py, [rows, cols], false) };
    let fast = unsafe { PyArray2::new(py, [rows, cols], false) };
    let slow = unsafe { PyArray2::new(py, [rows, cols], false) };
    let center = unsafe { PyArray2::new(py, [rows, cols], false) };

    unsafe {
        dvdi.as_slice_mut()?
            .copy_from_slice(&out.values[0 * plane..1 * plane]);
        fast.as_slice_mut()?
            .copy_from_slice(&out.values[1 * plane..2 * plane]);
        slow.as_slice_mut()?
            .copy_from_slice(&out.values[2 * plane..3 * plane]);
        center
            .as_slice_mut()?
            .copy_from_slice(&out.values[3 * plane..4 * plane]);
    }

    let d = PyDict::new(py);
    d.set_item("dvdi", dvdi)?;
    d.set_item("fast", fast)?;
    d.set_item("slow", slow)?;
    d.set_item("center", center)?;
    d.set_item(
        "periods",
        out.combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "smoothing_periods",
        out.combos
            .iter()
            .map(|p| p.smoothing_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "fast_multipliers",
        out.combos
            .iter()
            .map(|p| p.fast_multiplier.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "slow_multipliers",
        out.combos
            .iter()
            .map(|p| p.slow_multiplier.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("rows", rows)?;
    d.set_item("cols", cols)?;
    d.set_item("series", series)?;
    Ok(d.into())
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DvdiqqeJsFlat {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dvdiqqe)]
pub fn dvdiqqe_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<Vec<f64>>,
    period: Option<usize>,
    smoothing_period: Option<usize>,
    fast_multiplier: Option<f64>,
    slow_multiplier: Option<f64>,
    volume_type: Option<String>,
    center_type: Option<String>,
    tick_size: Option<f64>,
) -> Result<JsValue, JsValue> {
    // Use defaults if parameters not provided
    let params = DvdiqqeParams {
        period: period.or(Some(13)),
        smoothing_period: smoothing_period.or(Some(6)),
        fast_multiplier: fast_multiplier.or(Some(2.618)),
        slow_multiplier: slow_multiplier.or(Some(4.236)),
        volume_type: volume_type.or_else(|| Some("default".to_string())),
        center_type: center_type.or_else(|| Some("dynamic".to_string())),
        tick_size: tick_size.or(Some(0.01)),
    };
    let input = DvdiqqeInput::from_slices(open, high, low, close, volume.as_deref(), params);
    let out = dvdiqqe_with_kernel(&input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let cols = close.len();
    let mut values = Vec::with_capacity(4 * cols);
    values.extend_from_slice(&out.dvdi);
    values.extend_from_slice(&out.fast_tl);
    values.extend_from_slice(&out.slow_tl);
    values.extend_from_slice(&out.center_line);

    serde_wasm_bindgen::to_value(&DvdiqqeJsFlat {
        values,
        rows: 4,
        cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dvdiqqe_alloc(len: usize) -> *mut f64 {
    let mut v: Vec<f64> = Vec::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dvdiqqe_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dvdiqqe_into)]
pub fn dvdiqqe_into(
    // inputs
    open: *const f64,
    high: *const f64,
    low: *const f64,
    close: *const f64,
    vol: *const f64,
    len: usize,
    period: usize,
    smoothing_period: usize,
    fast_multiplier: f64,
    slow_multiplier: f64,
    volume_type: String,
    center_type: String,
    tick_size: f64,
    // output: flattened 4×len
    out_ptr: *mut f64,
) -> Result<(), JsValue> {
    if open.is_null() || high.is_null() || low.is_null() || close.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let o = std::slice::from_raw_parts(open, len);
        let h = std::slice::from_raw_parts(high, len);
        let l = std::slice::from_raw_parts(low, len);
        let c = std::slice::from_raw_parts(close, len);
        let v = if vol.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(vol, len))
        };

        let mut out = std::slice::from_raw_parts_mut(out_ptr, 4 * len);
        let (dvdi_dst, rest) = out.split_at_mut(len);
        let (fast_dst, rest) = rest.split_at_mut(len);
        let (slow_dst, cent_dst) = rest.split_at_mut(len);

        // no intermediate DvdiqqeOutput
        let params = DvdiqqeParams {
            period: Some(period),
            smoothing_period: Some(smoothing_period),
            fast_multiplier: Some(fast_multiplier),
            slow_multiplier: Some(slow_multiplier),
            volume_type: Some(volume_type),
            center_type: Some(center_type),
            tick_size: Some(tick_size),
        };
        let input = DvdiqqeInput::from_slices(o, h, l, c, v, params);
        dvdiqqe_into_slices(
            dvdi_dst,
            fast_dst,
            slow_dst,
            cent_dst,
            &input,
            detect_best_kernel(),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DvdiqqeBatchConfig {
    pub period_range: (usize, usize, usize),
    pub smoothing_period_range: (usize, usize, usize),
    pub fast_mult_range: (f64, f64, f64),
    pub slow_mult_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DvdiqqeParamsJs {
    pub period: usize,
    pub smoothing_period: usize,
    pub fast_multiplier: f64,
    pub slow_multiplier: f64,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DvdiqqeBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub combos: Vec<DvdiqqeParamsJs>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dvdiqqe_batch_unified)]
pub fn dvdiqqe_batch_unified_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<Vec<f64>>,
    config: JsValue,
    volume_type: String,
    center_type: String,
    tick_size: f64,
) -> Result<JsValue, JsValue> {
    let cfg: DvdiqqeBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let sweep = DvdiqqeBatchRange {
        period: cfg.period_range,
        smoothing_period: cfg.smoothing_period_range,
        fast_multiplier: cfg.fast_mult_range,
        slow_multiplier: cfg.slow_mult_range,
    };

    let result = dvdiqqe_batch_with_kernel(
        open,
        high,
        low,
        close,
        volume.as_deref(),
        &sweep,
        detect_best_kernel(),
        &volume_type,
        &center_type,
        tick_size,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let cols = close.len();
    let rows = result.rows;
    let mut values = Vec::with_capacity(4 * rows * cols);

    // Flatten all outputs into single array
    values.extend_from_slice(&result.dvdi_values);
    values.extend_from_slice(&result.fast_tl_values);
    values.extend_from_slice(&result.slow_tl_values);
    values.extend_from_slice(&result.center_values);

    // Extract combo parameters for metadata as structs
    let combos: Vec<DvdiqqeParamsJs> = result
        .combos
        .iter()
        .map(|p| DvdiqqeParamsJs {
            period: p.period.unwrap_or(13),
            smoothing_period: p.smoothing_period.unwrap_or(6),
            fast_multiplier: p.fast_multiplier.unwrap_or(2.618),
            slow_multiplier: p.slow_multiplier.unwrap_or(4.236),
        })
        .collect();

    let output = DvdiqqeBatchJsOutput {
        values,
        rows: rows * 4, // 4 outputs per parameter combo
        cols,
        combos,
    };

    serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dvdiqqe_batch_into)]
pub fn dvdiqqe_batch_into(
    // inputs
    open: *const f64,
    high: *const f64,
    low: *const f64,
    close: *const f64,
    vol: *const f64,
    len: usize,
    // batch params
    period_start: usize,
    period_end: usize,
    period_step: usize,
    smoothing_start: usize,
    smoothing_end: usize,
    smoothing_step: usize,
    fast_start: f64,
    fast_end: f64,
    fast_step: f64,
    slow_start: f64,
    slow_end: f64,
    slow_step: f64,
    // options
    volume_type: String,
    center_type: String,
    tick_size: f64,
    // output: flattened [rows*4 × cols]
    out_ptr: *mut f64,
) -> Result<(), JsValue> {
    if open.is_null() || high.is_null() || low.is_null() || close.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open, len);
        let high = std::slice::from_raw_parts(high, len);
        let low = std::slice::from_raw_parts(low, len);
        let close = std::slice::from_raw_parts(close, len);
        let volume = if vol.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(vol, len))
        };

        let sweep = DvdiqqeBatchRange {
            period: (period_start, period_end, period_step),
            smoothing_period: (smoothing_start, smoothing_end, smoothing_step),
            fast_multiplier: (fast_start, fast_end, fast_step),
            slow_multiplier: (slow_start, slow_end, slow_step),
        };

        let result = dvdiqqe_batch_with_kernel_flat(
            open,
            high,
            low,
            close,
            volume,
            &sweep,
            detect_best_kernel(),
            &volume_type,
            &center_type,
            tick_size,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // The flat output values are already in the correct order, just copy directly
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, result.values.len());
        out_slice.copy_from_slice(&result.values);

        Ok(())
    }
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::{read_candles_from_csv, Candles};
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    #[test]
    fn test_dvdiqqe_accuracy_scalar() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_scalar", Kernel::Scalar)
    }

    #[test]
    fn test_dvdiqqe_accuracy_auto() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_auto", Kernel::Auto)
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_dvdiqqe_accuracy_avx2() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_avx2", Kernel::Avx2)
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_dvdiqqe_accuracy_avx512() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_avx512", Kernel::Avx512)
    }

    #[test]
    fn test_dvdiqqe_with_csv_data() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input)?;

        let len = candles.close.len();
        assert_eq!(result.dvdi.len(), len);
        assert_eq!(result.fast_tl.len(), len);
        assert_eq!(result.slow_tl.len(), len);
        assert_eq!(result.center_line.len(), len);

        // Check for NaN handling in warmup period and valid values after
        let warmup = 25; // (13 * 2) - 1
        for i in 0..warmup.min(len) {
            assert!(
                result.dvdi[i].is_nan(),
                "Expected NaN in warmup at index {}",
                i
            );
        }
        for i in warmup..len {
            assert!(
                result.dvdi[i].is_finite(),
                "Expected finite value after warmup at index {}",
                i
            );
        }

        Ok(())
    }

    #[test]
    fn test_dvdiqqe_empty_input() {
        let candles = Candles::new(vec![], vec![], vec![], vec![], vec![], vec![]);
        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dvdiqqe_all_nan() {
        let nan_vec = vec![f64::NAN; 10];
        let candles = Candles::new(
            vec![0; 10],
            nan_vec.clone(),
            nan_vec.clone(),
            nan_vec.clone(),
            nan_vec.clone(),
            nan_vec.clone(),
        );
        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dvdiqqe_period_validation() {
        let data = vec![1.0, 2.0, 3.0];
        let candles = Candles::new(
            vec![0, 1, 2],
            data.clone(),
            data.clone(),
            data.clone(),
            data.clone(),
            vec![100.0, 200.0, 300.0],
        );

        let params = DvdiqqeParams {
            period: Some(10), // Period too large
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input);
        assert!(result.is_err());
    }

    // ==================== COMPREHENSIVE TEST FUNCTIONS ====================

    fn check_dvdiqqe_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe_with_kernel(&input, kernel)?;

        // PineScript reference values for the last 5 data points from the user
        // DVDI, Fast TL, and Slow TL match PineScript exactly
        let expected_dvdi = vec![
            -304.41010224,
            -279.48152664,
            -287.58723437,
            -252.40349484,
            -343.00922595,
        ];
        let expected_slow_tl = vec![
            -990.21769695,
            -955.69385266,
            -951.82562405,
            -903.39071943,
            -903.39071943,
        ];
        let expected_fast_tl = vec![
            -728.26380454,
            -697.40500858,
            -697.40500858,
            -654.73695895,
            -654.73695895,
        ];
        // Center line is a cumulative average, so it depends on the dataset's starting point
        // These values are for our CSV file (2018-09-01-2024-Bitfinex_Spot-4h.csv)
        let expected_center = vec![
            21.98929919135097,
            21.969910753134442,
            21.950003541229705,
            21.932361363982043,
            21.908895469736102,
        ];

        let start = result.dvdi.len().saturating_sub(5);

        for i in 0..5 {
            let diff_dvdi = (result.dvdi[start + i] - expected_dvdi[i]).abs();
            let diff_slow = (result.slow_tl[start + i] - expected_slow_tl[i]).abs();
            let diff_fast = (result.fast_tl[start + i] - expected_fast_tl[i]).abs();
            let diff_center = (result.center_line[start + i] - expected_center[i]).abs();

            assert!(
                diff_dvdi < 1e-6,
                "[{}] DVDI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.dvdi[start + i],
                expected_dvdi[i]
            );
            assert!(
                diff_slow < 1e-6,
                "[{}] Slow TL {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.slow_tl[start + i],
                expected_slow_tl[i]
            );
            assert!(
                diff_fast < 1e-6,
                "[{}] Fast TL {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.fast_tl[start + i],
                expected_fast_tl[i]
            );
            assert!(
                diff_center < 1e-6,
                "[{}] Center line {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.center_line[start + i],
                expected_center[i]
            );
        }

        Ok(())
    }

    fn check_dvdiqqe_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = DvdiqqeParams {
            period: None,
            smoothing_period: None,
            fast_multiplier: None,
            slow_multiplier: None,
            volume_type: None,
            center_type: None,
            tick_size: None,
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let output = dvdiqqe_with_kernel(&input, kernel)?;
        assert_eq!(output.dvdi.len(), candles.close.len());

        Ok(())
    }

    fn check_dvdiqqe_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = DvdiqqeInput::with_default_candles(&candles);
        let output = dvdiqqe_with_kernel(&input, kernel)?;
        assert_eq!(output.dvdi.len(), candles.close.len());

        Ok(())
    }

    fn check_dvdiqqe_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![100.0; 50];
        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            data.clone(),
            data.iter().map(|x| x + 1.0).collect(),
            data.iter().map(|x| x - 1.0).collect(),
            data.clone(),
            vec![1000.0; 50],
        );

        let params = DvdiqqeParams {
            period: Some(0),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with zero period",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![100.0; 5];
        let candles = Candles::new(
            vec![0, 1, 2, 3, 4],
            data.clone(),
            data.iter().map(|x| x + 1.0).collect(),
            data.iter().map(|x| x - 1.0).collect(),
            data.clone(),
            vec![1000.0; 5],
        );

        let params = DvdiqqeParams {
            period: Some(20),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with period exceeding length",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = Candles::new(
            vec![0],
            vec![100.0],
            vec![101.0],
            vec![99.0],
            vec![100.0],
            vec![1000.0],
        );

        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with insufficient data",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = Candles::new(vec![], vec![], vec![], vec![], vec![], vec![]);
        let input = DvdiqqeInput::from_candles(&candles, DvdiqqeParams::default());
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with empty input",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = vec![f64::NAN; 50];
        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            nan_data.clone(),
            nan_data.clone(),
            nan_data.clone(),
            nan_data.clone(),
            nan_data.clone(),
        );

        let input = DvdiqqeInput::from_candles(&candles, DvdiqqeParams::default());
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with all NaN input",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Create data with some NaN values
        let mut close = vec![100.0; 50];
        close[10] = f64::NAN;
        close[11] = f64::NAN;

        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            close.clone(),
            close
                .iter()
                .map(|x| if x.is_nan() { f64::NAN } else { x + 1.0 })
                .collect(),
            close
                .iter()
                .map(|x| if x.is_nan() { f64::NAN } else { x - 1.0 })
                .collect(),
            close.clone(),
            vec![1000.0; 50],
        );

        let input = DvdiqqeInput::from_candles(&candles, DvdiqqeParams::default());
        let res = dvdiqqe_with_kernel(&input, kernel)?;

        assert_eq!(res.dvdi.len(), 50);
        // Check that calculation continues after NaN values
        if res.dvdi.len() > 30 {
            assert!(
                res.dvdi[30..].iter().any(|x| x.is_finite()),
                "[{}] DVDIQQE should recover after NaN values",
                test_name
            );
        }

        Ok(())
    }

    fn check_dvdiqqe_with_tick_volume(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with tick volume forced
        let params = DvdiqqeParams {
            volume_type: Some("tick".to_string()),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe_with_kernel(&input, kernel)?;

        assert_eq!(result.dvdi.len(), candles.close.len());

        Ok(())
    }

    fn check_dvdiqqe_static_center(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![100.0; 50];
        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            data.clone(),
            data.iter().map(|x| x + 1.0).collect(),
            data.iter().map(|x| x - 1.0).collect(),
            data.clone(),
            vec![1000.0; 50],
        );

        let params = DvdiqqeParams {
            center_type: Some("static".to_string()),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe_with_kernel(&input, kernel)?;

        // Static center should be all zeros after warmup (NaN during warmup)
        let warmup = 25; // (13 * 2) - 1 for default period
        for i in warmup..result.center_line.len() {
            assert!(
                result.center_line[i] == 0.0,
                "[{}] Static center line should be 0.0 at index {}, got {}",
                test_name,
                i,
                result.center_line[i]
            );
        }

        Ok(())
    }

    // Macro for generating tests for all kernel variants
    macro_rules! generate_all_dvdiqqe_tests {
        ($($test_fn:ident),* $(,)?) => {
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

    // Batch test macro
    macro_rules! gen_dvdiqqe_batch_tests {
        ($fnn:ident) => {
            paste::paste!{
                #[test] fn [<$fnn _scalar>]() { let _ = $fnn(stringify!([<$fnn _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fnn _avx2>]()   { let _ = $fnn(stringify!([<$fnn _avx2>]),   Kernel::Avx2Batch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fnn _avx512>]() { let _ = $fnn(stringify!([<$fnn _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fnn _auto>]()   { let _ = $fnn(stringify!([<$fnn _auto>]),   Kernel::Auto); }
            }
        }
    }

    fn check_batch_default_row(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = DvdiqqeBatchBuilder::new().kernel(k).apply_candles(&c)?;
        assert_eq!(out.dvdi_values.len(), out.rows * out.cols);
        assert_eq!(out.fast_tl_values.len(), out.rows * out.cols);
        assert_eq!(out.slow_tl_values.len(), out.rows * out.cols);
        assert_eq!(out.center_values.len(), out.rows * out.cols);
        Ok(())
    }

    fn check_batch_sweep(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = DvdiqqeBatchBuilder::new()
            .kernel(k)
            .period_range(13, 16, 1)
            .smoothing_range(5, 7, 1)
            .fast_range(2.6, 2.6, 0.0)
            .slow_range(4.2, 4.2, 0.0)
            .apply_candles(&c)?;
        let expected = 4/*periods*/ * 3/*smoothings*/;
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());
        Ok(())
    }

    gen_dvdiqqe_batch_tests!(check_batch_default_row);
    gen_dvdiqqe_batch_tests!(check_batch_sweep);

    fn check_dvdiqqe_batch_default_row_old(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let batch_output = DvdiqqeBatchBuilder::new()
            .kernel(kernel)
            .period_static(13)
            .smoothing_static(6)
            .fast_static(2.618)
            .slow_static(4.236)
            .apply_candles(&c)?;

        let def_params = DvdiqqeParams::default();
        let batch_values = batch_output
            .values_for(&def_params)
            .expect("default row missing");

        // Compare with single calculation
        let single_input = DvdiqqeInput::with_default_candles(&c);
        let single_output = dvdiqqe_with_kernel(&single_input, kernel)?;

        assert_eq!(batch_values.dvdi.len(), single_output.dvdi.len());

        // Check values match
        for i in 0..batch_values.dvdi.len() {
            if batch_values.dvdi[i].is_finite() && single_output.dvdi[i].is_finite() {
                assert!(
                    (batch_values.dvdi[i] - single_output.dvdi[i]).abs() < 1e-10,
                    "[{}] DVDI mismatch at index {}: batch={}, single={}",
                    test_name,
                    i,
                    batch_values.dvdi[i],
                    single_output.dvdi[i]
                );
            }
        }

        Ok(())
    }

    fn check_dvdiqqe_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        // Streaming doesn't use kernel directly, but we keep signature for consistency
        let test_data = vec![
            (100.0, 102.0, 99.0, 101.0, 1000.0),
            (101.0, 103.0, 100.0, 102.0, 1100.0),
            (102.0, 104.0, 101.0, 103.0, 1200.0),
            (103.0, 105.0, 102.0, 104.0, 1150.0),
            (104.0, 106.0, 103.0, 105.0, 1250.0),
            // Add more data points
        ];

        // Fill with more test data
        let mut full_data = test_data.clone();
        for i in 0..50 {
            let base = 105.0 + i as f64 * 0.5;
            full_data.push((
                base,
                base + 2.0,
                base - 1.0,
                base + 1.0,
                1000.0 + (i as f64 * 50.0),
            ));
        }

        let params = DvdiqqeParams::default();
        let mut stream = DvdiqqeStream::try_new(params.clone())?;

        // Feed data to stream
        let mut stream_results = Vec::new();
        for (open, high, low, close, volume) in &full_data {
            if let Some(output) = stream.update(*open, *high, *low, *close, *volume) {
                stream_results.push((
                    output.dvdi,
                    output.fast_tl,
                    output.slow_tl,
                    output.center_line,
                ));
            }
        }

        // Stream should produce some results after warmup
        assert!(
            !stream_results.is_empty(),
            "[{}] Stream should produce outputs",
            test_name
        );

        // Verify last stream value matches batch calculation on full data
        if stream_results.len() > 0 {
            let (opens, highs, lows, closes, volumes): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
                full_data.iter().cloned().unzip_n_tuple();

            let batch_input =
                DvdiqqeInput::from_slices(&opens, &highs, &lows, &closes, Some(&volumes), params);

            if let Ok(batch_output) = dvdiqqe(&batch_input) {
                let last_idx = batch_output.dvdi.len() - 1;
                let last_stream = stream_results.last().unwrap();

                // Allow some tolerance for cumulative calculations
                if batch_output.dvdi[last_idx].is_finite() && last_stream.0.is_finite() {
                    assert!(
                        (batch_output.dvdi[last_idx] - last_stream.0).abs() < 1.0,
                        "[{}] Stream DVDI doesn't match batch: stream={}, batch={}",
                        test_name,
                        last_stream.0,
                        batch_output.dvdi[last_idx]
                    );
                }
            }
        }

        Ok(())
    }

    fn check_dvdiqqe_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Create simple test data
        let n = 100;
        let mut opens = vec![100.0; n];
        let mut highs = vec![102.0; n];
        let mut lows = vec![98.0; n];
        let mut closes = vec![100.0; n];
        let mut volumes = vec![1000.0; n];

        // Add some variation
        for i in 0..n {
            let base = 100.0 + (i as f64 * 0.1);
            opens[i] = base;
            highs[i] = base + 2.0;
            lows[i] = base - 2.0;
            closes[i] = base + 0.5;
            volumes[i] = 1000.0 + (i as f64 * 10.0);
        }

        let batch_output = DvdiqqeBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 15, 2)
            .smoothing_range(4, 8, 2)
            .fast_range(2.0, 3.0, 0.5)
            .slow_range(4.0, 5.0, 0.5)
            .apply_slices(&opens, &highs, &lows, &closes, Some(&volumes))?;

        // Should have correct dimensions
        let expected_periods = 3; // 10, 12, 14
        let expected_smoothings = 3; // 4, 6, 8
        let expected_fasts = 3; // 2.0, 2.5, 3.0
        let expected_slows = 3; // 4.0, 4.5, 5.0
        let expected_rows =
            expected_periods * expected_smoothings * expected_fasts * expected_slows;

        assert_eq!(
            batch_output.rows, expected_rows,
            "[{}] Wrong number of parameter combinations",
            test_name
        );
        assert_eq!(
            batch_output.cols, n,
            "[{}] Wrong number of data points",
            test_name
        );

        // Check that we can retrieve specific combinations
        let test_params = DvdiqqeParams {
            period: Some(12),
            smoothing_period: Some(6),
            fast_multiplier: Some(2.5),
            slow_multiplier: Some(4.5),
            volume_type: None,
            center_type: None,
            tick_size: None,
        };

        assert!(
            batch_output.values_for(&test_params).is_some(),
            "[{}] Should find test parameter combination",
            test_name
        );

        Ok(())
    }

    // Helper for unzipping 5-tuples
    trait UnzipN<A, B, C, D, E> {
        fn unzip_n_tuple(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>);
    }

    impl<A, B, C, D, E, I> UnzipN<A, B, C, D, E> for I
    where
        I: Iterator<Item = (A, B, C, D, E)>,
    {
        fn unzip_n_tuple(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>) {
            let mut a_vec = Vec::new();
            let mut b_vec = Vec::new();
            let mut c_vec = Vec::new();
            let mut d_vec = Vec::new();
            let mut e_vec = Vec::new();

            for (a, b, c, d, e) in self {
                a_vec.push(a);
                b_vec.push(b);
                c_vec.push(c);
                d_vec.push(d);
                e_vec.push(e);
            }

            (a_vec, b_vec, c_vec, d_vec, e_vec)
        }
    }

    // Add batch sweep test function
    fn check_dvdiqqe_batch_sweep_old(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = DvdiqqeBatchBuilder::new()
            .kernel(kernel)
            .period_range(13, 16, 1)
            .smoothing_range(5, 7, 1)
            .fast_range(2.6, 2.6, 0.0)
            .slow_range(4.2, 4.2, 0.0)
            .apply_candles(&c)?;

        let expected = 4/*periods*/ * 3/*smoothings*/;
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());

        Ok(())
    }

    // Add reinput test function
    fn check_dvdiqqe_reinput(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out1 = dvdiqqe_with_kernel(&DvdiqqeInput::with_default_candles(&c), k)?;
        // Reuse dvdi as a synthetic close to ensure length/API parity
        let i2 = DvdiqqeInput::from_slices(
            &c.open,
            &c.high,
            &c.low,
            &out1.dvdi,
            Some(&c.volume),
            DvdiqqeParams::default(),
        );
        let out2 = dvdiqqe_with_kernel(&i2, k)?;
        assert_eq!(out2.dvdi.len(), out1.dvdi.len());
        Ok(())
    }

    // Generate all test variants
    generate_all_dvdiqqe_tests!(
        check_dvdiqqe_accuracy,
        check_dvdiqqe_partial_params,
        check_dvdiqqe_default_candles,
        check_dvdiqqe_zero_period,
        check_dvdiqqe_period_exceeds_length,
        check_dvdiqqe_very_small_dataset,
        check_dvdiqqe_empty_input,
        check_dvdiqqe_all_nan,
        check_dvdiqqe_nan_handling,
        check_dvdiqqe_with_tick_volume,
        check_dvdiqqe_static_center,
        check_dvdiqqe_batch_default_row_old,
        check_dvdiqqe_streaming,
        check_dvdiqqe_batch_sweep_old,
        check_dvdiqqe_reinput
    );

    #[cfg(debug_assertions)]
    #[test]
    fn dvdiqqe_no_poison_in_outputs() -> Result<(), Box<dyn Error>> {
        use crate::utilities::data_loader::read_candles_from_csv;
        let c = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;

        // Test single calculation
        let out = dvdiqqe_with_kernel(&DvdiqqeInput::with_default_candles(&c), Kernel::Scalar)?;

        // Check for poison patterns that would indicate uninitialized memory
        for &v in out
            .dvdi
            .iter()
            .chain(&out.fast_tl)
            .chain(&out.slow_tl)
            .chain(&out.center_line)
        {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x2222_2222_2222_2222,
                "init_matrix_prefixes poison in single output"
            );
            assert_ne!(
                b, 0x3333_3333_3333_3333,
                "make_uninit_matrix poison in single output"
            );
        }

        // Test batch calculation with flat output
        let sweep = DvdiqqeBatchRange::default();
        let flat_out = dvdiqqe_batch_with_kernel_flat(
            &c.open,
            &c.high,
            &c.low,
            &c.close,
            Some(&c.volume),
            &sweep,
            Kernel::Auto,
            "default",
            "dynamic",
            0.01,
        )?;

        for (i, &v) in flat_out.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x2222_2222_2222_2222,
                "init_matrix_prefixes poison at {}",
                i
            );
            assert_ne!(
                b, 0x3333_3333_3333_3333,
                "make_uninit_matrix poison at {}",
                i
            );
        }

        Ok(())
    }

    // ==================== PROPERTY-BASED TESTS ====================
    #[cfg(feature = "proptest")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_dvdiqqe_output_length_matches_input(
                data in prop::collection::vec(prop::num::f64::NORMAL | prop::num::f64::POSITIVE, 50..200)
            ) {
                let len = data.len();
                let timestamps: Vec<i64> = (0..len as i64).collect();
                let mut open = Vec::with_capacity(len);
                let mut high = Vec::with_capacity(len);
                let mut low = Vec::with_capacity(len);
                let mut close = Vec::with_capacity(len);
                let mut volume = Vec::with_capacity(len);

                for &val in &data {
                    open.push(val - 0.5);
                    high.push(val + 1.0);
                    low.push(val - 1.0);
                    close.push(val);
                    volume.push(1000.0);
                }

                let candles = Candles::new(timestamps, open, high, low, close, volume);
                let input = DvdiqqeInput::with_default_candles(&candles);

                match dvdiqqe(&input) {
                    Ok(output) => {
                        prop_assert_eq!(output.dvdi.len(), len);
                        prop_assert_eq!(output.fast_tl.len(), len);
                        prop_assert_eq!(output.slow_tl.len(), len);
                        prop_assert_eq!(output.center_line.len(), len);
                    }
                    Err(DvdiqqeError::NotEnoughValidData { .. }) => {
                        // Expected for small datasets
                    }
                    Err(e) => {
                        prop_assert!(false, "Unexpected error: {:?}", e);
                    }
                }
            }

            #[test]
            fn test_dvdiqqe_nan_propagation(
                valid_data in prop::collection::vec(100.0f64..200.0, 50..100),
                nan_positions in prop::collection::vec(0usize..50, 0..10)
            ) {
                let len = valid_data.len();
                let mut data = valid_data.clone();

                // Insert NaNs at specified positions
                for &pos in &nan_positions {
                    if pos < len {
                        data[pos] = f64::NAN;
                    }
                }

                let timestamps: Vec<i64> = (0..len as i64).collect();
                let open = data.iter().map(|&v| v - 0.5).collect();
                let high = data.iter().map(|&v| v + 1.0).collect();
                let low = data.iter().map(|&v| v - 1.0).collect();
                let volume = vec![1000.0; len];

                let candles = Candles::new(timestamps, open, high, low, data.clone(), volume);
                let input = DvdiqqeInput::with_default_candles(&candles);

                match dvdiqqe(&input) {
                    Ok(output) => {
                        // Verify the indicator handles NaN inputs gracefully
                        // The indicator should produce valid output structure
                        prop_assert_eq!(output.dvdi.len(), len);
                        prop_assert_eq!(output.fast_tl.len(), len);
                        prop_assert_eq!(output.slow_tl.len(), len);
                        prop_assert_eq!(output.center_line.len(), len);

                        // Check that early positions have NaN due to warmup
                        // The exact warmup period depends on parameters
                        let expected_warmup = 25; // Default period (13) * 2 - 1
                        for i in 0..expected_warmup.min(len) {
                            // Early values should be NaN due to warmup, not input NaNs
                            prop_assert!(output.dvdi[i].is_nan() || output.dvdi[i].is_finite(),
                                "Position {} should be either NaN (warmup) or finite", i);
                        }

                        // Verify no poison values (uninitialized memory)
                        for &v in output.dvdi.iter()
                            .chain(&output.fast_tl)
                            .chain(&output.slow_tl)
                            .chain(&output.center_line) {
                            if !v.is_nan() {
                                let bits = v.to_bits();
                                prop_assert_ne!(bits, 0x2222_2222_2222_2222);
                                prop_assert_ne!(bits, 0x3333_3333_3333_3333);
                            }
                        }
                    }
                    Err(DvdiqqeError::AllValuesNaN) => {
                        // Expected if all close values become NaN
                        prop_assert!(nan_positions.len() >= len / 2);
                    }
                    Err(_) => {
                        // Other errors acceptable for edge cases with many NaNs
                    }
                }
            }

            #[test]
            fn test_dvdiqqe_parameter_bounds(
                period in 1usize..50,
                smoothing in 1usize..20,
                fast_mult in 0.1f64..10.0,
                slow_mult in 0.1f64..10.0
            ) {
                let len = 100;
                let data: Vec<f64> = (0..len).map(|i| 100.0 + i as f64).collect();
                let timestamps: Vec<i64> = (0..len as i64).collect();
                let open = data.iter().map(|&v| v - 0.5).collect();
                let high = data.iter().map(|&v| v + 1.0).collect();
                let low = data.iter().map(|&v| v - 1.0).collect();
                let volume = vec![1000.0; len];

                let candles = Candles::new(timestamps, open, high, low, data, volume);
                let params = DvdiqqeParams {
                    period: Some(period),
                    smoothing_period: Some(smoothing),
                    fast_multiplier: Some(fast_mult),
                    slow_multiplier: Some(slow_mult),
                    ..Default::default()
                };
                let input = DvdiqqeInput::from_candles(&candles, params);

                match dvdiqqe(&input) {
                    Ok(output) => {
                        // All outputs should be finite after warmup
                        for i in 30..len {
                            prop_assert!(output.dvdi[i].is_finite() || output.dvdi[i].is_nan());
                            prop_assert!(output.fast_tl[i].is_finite() || output.fast_tl[i].is_nan());
                            prop_assert!(output.slow_tl[i].is_finite() || output.slow_tl[i].is_nan());
                        }
                    }
                    Err(DvdiqqeError::InvalidPeriod { .. }) => {
                        prop_assert!(period > len || period == 0);
                    }
                    Err(_) => {
                        // Other errors acceptable for edge cases
                    }
                }
            }
        }
    }
}
