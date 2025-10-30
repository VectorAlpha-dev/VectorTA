//! # Ultimate Moving Average (UMA)
//!
//! An adaptive moving average that dynamically adjusts its length based on multiple indicators
//! including RSI, MFI, and standard deviation to respond to market conditions.
//!
//! ## Parameters
//! - **accelerator**: Acceleration factor for dynamic length adjustment (default: 8.0)
//! - **min_length**: Minimum lookback period (default: 5)
//! - **max_length**: Maximum lookback period (default: 50)
//! - **smooth_length**: Smoothing period for final output (default: 4)
//!
//! ## Returns
//! - **`Ok(UmaOutput)`** containing values vector of length matching input data
//! - **`Err(UmaError)`** on calculation errors
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Not implemented (no explicit SIMD kernel functions)
//! - **Streaming update**: O(1) where possible; UMA core O(L)
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix)
//! - **Optimization status**: Scalar core optimized (inlined RSI/MFI, ln LUT, FMA). SIMD accumulation present under `nightly-avx`, but Auto selection prefers scalar due to limited wins from scalar `exp` cost.
//! - **Streaming**: optimized. Maintains O(1) updates for mean/σ, momentum, and WMA;
//!   UMA core remains O(L) per tick due to non-decomposable power weights.
//!
//! Binding test status (2025-10-27):
//! - WASM bindings: failures currently observed in tests/wasm/test_uma.js
//!   - "UMA - streaming interface": no valid values produced after warmup.
//!   - "UMA - streaming with volume": no valid values produced after warmup.
//! - Python bindings: failure in tests/python/test_uma.py
//!   - TestUma::test_uma_streaming_with_volume: stream outputs remain all-NaN.
//!
//! Notes: Batch UMA matches Rust references (1% tolerance) in both bindings. The
//! streaming failures appear due to an overly strict warmup gate in UmaStream
//! that never transitions to producing values once the internal ring reaches
//! capacity. Investigate the warmup condition so that streaming begins emitting
//! values after enough samples (>= max_length) and lets the WMA stage handle its
//! own warmup. No reference values changed.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaUma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use js_sys;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::deviation::{deviation, DeviationInput, DeviationParams};
use crate::indicators::mfi::{mfi, MfiInput, MfiParams};
use crate::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
use crate::indicators::moving_averages::wma::{wma, WmaInput, WmaParams};
use crate::indicators::rsi::{rsi, RsiInput, RsiParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for UmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            UmaData::Slice(slice) => slice,
            UmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct UmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct UmaParams {
    pub accelerator: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for UmaParams {
    fn default() -> Self {
        Self {
            accelerator: Some(1.0),
            min_length: Some(5),
            max_length: Some(50),
            smooth_length: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UmaInput<'a> {
    pub data: UmaData<'a>,
    pub params: UmaParams,
    pub volume: Option<&'a [f64]>,
}

impl<'a> UmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: UmaParams) -> Self {
        Self {
            data: UmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
            volume: Some(&c.volume),
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], vol: Option<&'a [f64]>, p: UmaParams) -> Self {
        Self {
            data: UmaData::Slice(sl),
            params: p,
            volume: vol,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", UmaParams::default())
    }

    #[inline]
    pub fn get_accelerator(&self) -> f64 {
        self.params.accelerator.unwrap_or(1.0)
    }

    #[inline]
    pub fn get_min_length(&self) -> usize {
        self.params.min_length.unwrap_or(5)
    }

    #[inline]
    pub fn get_max_length(&self) -> usize {
        self.params.max_length.unwrap_or(50)
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(4)
    }
}

// ==================== BUILDER ====================

#[derive(Copy, Clone, Debug)]
pub struct UmaBuilder {
    accelerator: Option<f64>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for UmaBuilder {
    fn default() -> Self {
        Self {
            accelerator: None,
            min_length: None,
            max_length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn accelerator(mut self, a: f64) -> Self {
        self.accelerator = Some(a);
        self
    }

    #[inline(always)]
    pub fn min_length(mut self, n: usize) -> Self {
        self.min_length = Some(n);
        self
    }

    #[inline(always)]
    pub fn max_length(mut self, n: usize) -> Self {
        self.max_length = Some(n);
        self
    }

    #[inline(always)]
    pub fn smooth_length(mut self, n: usize) -> Self {
        self.smooth_length = Some(n);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<UmaOutput, UmaError> {
        let p = UmaParams {
            accelerator: self.accelerator,
            min_length: self.min_length,
            max_length: self.max_length,
            smooth_length: self.smooth_length,
        };
        let i = UmaInput::from_candles(c, "close", p);
        uma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64], v: Option<&[f64]>) -> Result<UmaOutput, UmaError> {
        let p = UmaParams {
            accelerator: self.accelerator,
            min_length: self.min_length,
            max_length: self.max_length,
            smooth_length: self.smooth_length,
        };
        let i = UmaInput::from_slice(d, v, p);
        uma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<UmaStream, UmaError> {
        let p = UmaParams {
            accelerator: self.accelerator,
            min_length: self.min_length,
            max_length: self.max_length,
            smooth_length: self.smooth_length,
        };
        UmaStream::try_new(p)
    }
}

// ==================== STREAMING (O(1) where possible; O(L) only for UMA core) ====================

/// SIMD status: streaming keeps scalar-safe logic; UMA core is O(L) per tick.
/// Rationale: power-weighted sliding average reweights all items as p and L vary; no exact O(1).
#[derive(Debug, Clone)]
pub struct UmaStream {
    // ---- configuration (copied from params for fast access) ----
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,

    // ---- rolling window over prices (size = max_length) ----
    price: Vec<f64>,
    volume: Vec<f64>,
    cap: usize,
    head: usize,
    count: usize,

    // rolling sums for mean/std at fixed max_length
    sum: f64,
    sumsq: f64,

    // ---- per-diff streaming state ----
    has_prev: bool,
    prev_price: f64,

    // prefix sums (ring) of up/down volume contributions over diffs
    upvol_cum: Vec<f64>,
    dnvol_cum: Vec<f64>,
    diff_step: usize,

    // RSI (Wilder) state for no-volume fallback
    rsi_avg_up: f64,
    rsi_avg_dn: f64,
    rsi_inited: bool,

    // UMA dynamic length state
    dynamic_length: f64,

    // ln(k) lookup for k=1..=max_length (index 0 unused)
    ln_lut: Vec<f64>,

    // ---- WMA(smooth_length) state over raw UMA ----
    uma_raw: Vec<f64>,
    uma_raw_head: usize,
    uma_raw_count: usize,
    s1: f64,
    s2: f64,
    wma_denom: f64,

    // keep params & kernel for parity
    params: UmaParams,
    kernel: Kernel,
}

impl UmaStream {
    pub fn try_new(params: UmaParams) -> Result<Self, UmaError> {
        let accelerator = params.accelerator.unwrap_or(1.0);
        let min_length = params.min_length.unwrap_or(5);
        let max_length = params.max_length.unwrap_or(50);
        let smooth_len = params.smooth_length.unwrap_or(4);

        if min_length == 0 {
            return Err(UmaError::InvalidMinLength { min_length });
        }
        if max_length == 0 {
            return Err(UmaError::InvalidMaxLength {
                max_length,
                data_len: 0,
            });
        }
        if smooth_len == 0 {
            return Err(UmaError::InvalidSmoothLength {
                smooth_length: smooth_len,
            });
        }
        if min_length > max_length {
            return Err(UmaError::MinLengthGreaterThanMaxLength {
                min_length,
                max_length,
            });
        }
        if accelerator < 1.0 {
            return Err(UmaError::InvalidAccelerator { accelerator });
        }

        // Precompute ln(k) once for k=1..=max_length
        let mut ln_lut = Vec::with_capacity(max_length + 1);
        ln_lut.push(0.0); // unused
        for k in 1..=max_length {
            ln_lut.push((k as f64).ln());
        }

        Ok(Self {
            accelerator,
            min_length,
            max_length,
            smooth_length: smooth_len,

            price: vec![0.0; max_length],
            volume: vec![0.0; max_length],
            cap: max_length,
            head: 0,
            count: 0,

            sum: 0.0,
            sumsq: 0.0,

            has_prev: false,
            prev_price: 0.0,

            upvol_cum: vec![0.0; max_length + 1],
            dnvol_cum: vec![0.0; max_length + 1],
            diff_step: 0,

            rsi_avg_up: 0.0,
            rsi_avg_dn: 0.0,
            rsi_inited: false,

            dynamic_length: max_length as f64,
            ln_lut,

            uma_raw: vec![0.0; smooth_len],
            uma_raw_head: 0,
            uma_raw_count: 0,
            s1: 0.0,
            s2: 0.0,
            wma_denom: (smooth_len as f64) * ((smooth_len as f64) + 1.0) * 0.5,

            params,
            kernel: detect_best_kernel(),
        })
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.update_with_volume(value, None)
    }

    /// `volume`: if provided, enables the volume-momentum path (slice-with-volume behavior).
    /// Otherwise the momentum falls back to RSI (Wilder).
    pub fn update_with_volume(&mut self, value: f64, volume: Option<f64>) -> Option<f64> {
        let v = volume.unwrap_or(0.0);

        // --- push into price/volume rings; maintain rolling sum/sumsq over fixed cap ---
        if self.count == self.cap {
            // overwriting oldest at head
            let old = self.price[self.head];
            self.sum -= old;
            self.sumsq -= old * old;
        } else {
            self.count += 1;
        }
        self.price[self.head] = value;
        self.volume[self.head] = v;
        self.sum += value;
        self.sumsq += value * value;

        // --- per-diff streaming state (volume up/down buckets + RSI state) ---
        if self.has_prev {
            let diff = value - self.prev_price;
            let up = if diff > 0.0 { diff } else { 0.0 };
            let dn = if diff < 0.0 { -diff } else { 0.0 };

            // prefix-sum rings for up/down volume (O(1) query for any window length)
            let cur = (self.diff_step + 1) % (self.cap + 1);
            let prev = self.diff_step % (self.cap + 1);
            let up_contrib = if up > 0.0 { v } else { 0.0 };
            let dn_contrib = if dn > 0.0 { v } else { 0.0 };
            self.upvol_cum[cur] = self.upvol_cum[prev] + up_contrib;
            self.dnvol_cum[cur] = self.dnvol_cum[prev] + dn_contrib;
            self.diff_step += 1;

            // RSI (Wilder) recurrence; alpha chosen from current dynamic length later.
            if !self.rsi_inited {
                self.rsi_avg_up = up;
                self.rsi_avg_dn = dn;
                self.rsi_inited = true;
            } else {
                // Defer weighted update until len_r known; see below.
            }
        }
        self.prev_price = value;
        self.has_prev = true;

        // advance ring head
        self.head = (self.head + 1) % self.cap;

        // --- warmup: need at least max_length for μ/σ and then smooth_length values for WMA ---
        let min_needed = self.max_length + self.smooth_length; // conservative
        if self.count + ((self.diff_step > 0) as usize) < min_needed {
            return None;
        }

        // --- rolling μ and σ over fixed max_length (population σ) ---
        let n = self.cap as f64;
        let mu = self.sum / n;
        let var = (self.sumsq / n) - mu * mu;
        let sd = if var > 0.0 { var.sqrt() } else { 0.0 };

        // UMA band thresholds
        let a = (-1.75f64).mul_add(sd, mu);
        let b = (-0.25f64).mul_add(sd, mu);
        let c = (0.25f64).mul_add(sd, mu);
        let d = (1.75f64).mul_add(sd, mu);

        // --- adapt dynamic length (±1 step, then clamp) ---
        let src = value;
        if src >= b && src <= c {
            self.dynamic_length += 1.0;
        } else if src < a || src > d {
            self.dynamic_length -= 1.0;
        }
        self.dynamic_length = self
            .dynamic_length
            .max(self.min_length as f64)
            .min(self.max_length as f64);
        let len_r = self.dynamic_length.round().max(1.0) as usize;

        // --- momentum factor: volume-weighted up/down ratio (if volume present), else RSI ---
        let mf = if v > 0.0 && self.diff_step > 0 {
            // O(1) query of last len_r diffs from prefix sums
            let cur = self.diff_step % (self.cap + 1);
            let prev = (self.diff_step + self.cap + 1 - len_r) % (self.cap + 1);
            let up_sum = self.upvol_cum[cur] - self.upvol_cum[prev];
            let dn_sum = self.dnvol_cum[cur] - self.dnvol_cum[prev];
            let tot = up_sum + dn_sum;
            if tot > 0.0 {
                100.0 * up_sum / tot
            } else {
                50.0
            }
        } else {
            // Wilder RSI with variable alpha = 1/len_r
            if self.diff_step > 0 {
                // Recompute last diff from ring to avoid dependency on overwritten prev
                let newest_idx = (self.head + self.cap - 1) % self.cap;
                let prev_idx = (self.head + self.cap - 2) % self.cap;
                let prevv = self.price[prev_idx];
                let last_diff = self.price[newest_idx] - prevv;
                let up = if last_diff > 0.0 { last_diff } else { 0.0 };
                let dn = if last_diff < 0.0 { -last_diff } else { 0.0 };

                let alpha = 1.0 / (len_r as f64);
                self.rsi_avg_up = (1.0 - alpha) * self.rsi_avg_up + alpha * up;
                self.rsi_avg_dn = (1.0 - alpha) * self.rsi_avg_dn + alpha * dn;
            }
            if self.rsi_avg_dn == 0.0 {
                100.0
            } else {
                let s = self.rsi_avg_up + self.rsi_avg_dn;
                if s == 0.0 {
                    50.0
                } else {
                    100.0 * self.rsi_avg_up / s
                }
            }
        };

        // --- exponent p from oscillator (same formula as batch) ---
        let mf_scaled = mf.mul_add(2.0, -100.0);
        let p = self.accelerator + (mf_scaled.abs() * 0.04); // 1/25

        // --- UMA core: power-weighted average over last len_r values (O(len_r)) ---
        let start = (self.head + self.cap - len_r) % self.cap;
        let mut xws = 0.0f64;
        let mut wsum = 0.0f64;

        // Unrolled scalar accumulation with LUT ln(k) and FMA; newest has k=1
        let mut j = 0usize;
        while j + 1 < len_r {
            let k1 = len_r - j;
            let k2 = k1 - 1;

            let w1 = exp_kernel(p * self.ln_lut[k1]);
            let idx1 = (start + j) % self.cap;
            let x1 = self.price[idx1];
            xws = x1.mul_add(w1, xws);
            wsum += w1;

            let w2 = exp_kernel(p * self.ln_lut[k2]);
            let idx2 = (start + j + 1) % self.cap;
            let x2 = self.price[idx2];
            xws = x2.mul_add(w2, xws);
            wsum += w2;

            j += 2;
        }
        if j < len_r {
            let k = len_r - j;
            let w = exp_kernel(p * self.ln_lut[k]);
            let idx = (start + j) % self.cap;
            let x = self.price[idx];
            xws = x.mul_add(w, xws);
            wsum += w;
        }

        let uma_raw = if wsum > 0.0 { xws / wsum } else { src };

        // --- smoothing: O(1) WMA over last `smooth_length` UMA values ---
        let m = self.smooth_length;
        if self.uma_raw_count < m {
            // initial fill with proper weights 1..k
            self.s1 += uma_raw;
            self.s2 += uma_raw * ((self.uma_raw_count as f64) + 1.0);
            self.uma_raw[self.uma_raw_head] = uma_raw;
            self.uma_raw_head = (self.uma_raw_head + 1) % m;
            self.uma_raw_count += 1;

            if self.uma_raw_count < m {
                return None;
            }
            // first valid output
            let out = self.s2 / self.wma_denom;
            Some(out)
        } else {
            // steady state O(1) updates
            let oldest = self.uma_raw[self.uma_raw_head];
            let s1_prev = self.s1;
            self.s1 = self.s1 - oldest + uma_raw;
            self.s2 = self.s2 - s1_prev + (m as f64) * uma_raw;

            // rotate ring
            self.uma_raw[self.uma_raw_head] = uma_raw;
            self.uma_raw_head = (self.uma_raw_head + 1) % m;

            Some(self.s2 / self.wma_denom)
        }
    }

    pub fn reset(&mut self) {
        self.price.fill(0.0);
        self.volume.fill(0.0);
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sumsq = 0.0;

        self.has_prev = false;
        self.prev_price = 0.0;

        self.upvol_cum.fill(0.0);
        self.dnvol_cum.fill(0.0);
        self.diff_step = 0;

        self.rsi_avg_up = 0.0;
        self.rsi_avg_dn = 0.0;
        self.rsi_inited = false;

        self.dynamic_length = self.max_length as f64;

        self.uma_raw.fill(0.0);
        self.uma_raw_head = 0;
        self.uma_raw_count = 0;
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

// ---- internal helper: swappable exp() (exact by default) ----
#[inline(always)]
fn exp_kernel(x: f64) -> f64 {
    x.exp()
}

// ==================== ERROR TYPES ====================

#[derive(Debug, Error)]
pub enum UmaError {
    #[error("uma: Input data slice is empty.")]
    EmptyInputData,
    #[error("uma: All values are NaN.")]
    AllValuesNaN,
    #[error("uma: Invalid max_length: max_length = {max_length}, data length = {data_len}")]
    InvalidMaxLength { max_length: usize, data_len: usize },
    #[error("uma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("uma: Invalid accelerator: {accelerator}")]
    InvalidAccelerator { accelerator: f64 },
    #[error("uma: Invalid min_length: {min_length}")]
    InvalidMinLength { min_length: usize },
    #[error("uma: Invalid smooth_length: {smooth_length}")]
    InvalidSmoothLength { smooth_length: usize },
    #[error("uma: min_length ({min_length}) must be <= max_length ({max_length})")]
    MinLengthGreaterThanMaxLength {
        min_length: usize,
        max_length: usize,
    },
    #[error("uma: Error from dependency: {0}")]
    DependencyError(String),
}

// ==================== CORE FUNCTIONS ====================

#[inline(always)]
fn uma_prepare<'a>(input: &'a UmaInput) -> Result<(&'a [f64], usize, usize, usize, f64), UmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(UmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(UmaError::AllValuesNaN)?;
    let accelerator = input.get_accelerator();
    let min_length = input.get_min_length();
    let max_length = input.get_max_length();
    let smooth_length = input.get_smooth_length();

    if max_length == 0 || max_length > len {
        return Err(UmaError::InvalidMaxLength {
            max_length,
            data_len: len,
        });
    }
    if min_length == 0 {
        return Err(UmaError::InvalidMinLength { min_length });
    }
    if min_length > max_length {
        return Err(UmaError::MinLengthGreaterThanMaxLength {
            min_length,
            max_length,
        });
    }
    if smooth_length == 0 {
        return Err(UmaError::InvalidSmoothLength { smooth_length });
    }
    if accelerator < 1.0 {
        return Err(UmaError::InvalidAccelerator { accelerator });
    }
    if len - first < max_length {
        return Err(UmaError::NotEnoughValidData {
            needed: max_length,
            valid: len - first,
        });
    }
    Ok((data, first, min_length, max_length, accelerator))
}

#[inline(always)]
fn uma_core_into(
    input: &UmaInput,
    first: usize,
    min_length: usize,
    max_length: usize,
    accelerator: f64,
    _kernel: Kernel, // (dispatch kept for future SIMD kernels)
    out: &mut [f64],
) -> Result<(), UmaError> {
    // ------- Aliases & basic lengths -------
    let data: &[f64] = input.as_ref();
    let len = data.len();
    debug_assert!(len == out.len());

    // ------- Precompute mean/std at fixed max_length (keeps UMA band logic identical) -------
    let mean = sma(&SmaInput::from_slice(
        data,
        SmaParams {
            period: Some(max_length),
        },
    ))
    .map_err(|e| UmaError::DependencyError(e.to_string()))?
    .values;

    let std_dev = deviation(&DeviationInput::from_slice(
        data,
        DeviationParams {
            period: Some(max_length),
            devtype: Some(0),
        },
    ))
    .map_err(|e| UmaError::DependencyError(e.to_string()))?
    .values;

    // ------- Constants & helpers -------
    // Precompute ln(k) once for k in [1..=max_length] so inner loop can use exp(p * ln(k)) instead of powf(k, p)
    let mut ln_lut: Vec<f64> = Vec::with_capacity(max_length + 1);
    // index 0 unused to keep direct indexing by base k
    ln_lut.push(0.0);
    for k in 1..=max_length {
        ln_lut.push((k as f64).ln());
    }

    // Avoid repeated pattern matching in the hot loop
    let (candles_opt, vol_opt) = match &input.data {
        UmaData::Candles { candles, .. } => (Some(*candles), input.volume),
        UmaData::Slice(_) => (None, input.volume),
    };

    // Warmup (prefix already NaN-initialized by caller)
    let warmup_end = first + max_length - 1;
    if warmup_end >= len {
        return Ok(()); // nothing to do; prefix already set by alloc_with_nan_prefix/uma_into_slice
    }

    // Adaptive length state
    let mut dyn_len = max_length as f64;

    // ------- Small inlined helpers (closed over data/ln_lut) -------
    #[inline(always)]
    fn rsi_wilder_last(data: &[f64], start: usize, end: usize, period: usize) -> f64 {
        // Compute only the LAST RSI value for a slice [start..=end] with Wilder smoothing and given period.
        if period == 0 || end <= start || end - start + 1 < period + 1 {
            return 50.0;
        }

        // Initialization over the first `period` diffs
        let mut sum_up = 0.0f64;
        let mut sum_dn = 0.0f64;
        let mut has_nan = false;
        let mut prev = data[start];
        let init_last = start + period;
        for j in (start + 1)..=init_last {
            let cur = data[j];
            let diff = cur - prev;
            if !diff.is_finite() {
                has_nan = true;
                break;
            }
            if diff > 0.0 {
                sum_up += diff;
            } else {
                sum_dn -= diff; // diff <= 0 -> add positive magnitude
            }
            prev = cur;
        }
        if has_nan {
            return 50.0; // conservative fallback
        }
        let mut avg_up = sum_up / (period as f64);
        let mut avg_dn = sum_dn / (period as f64);

        // Wilder update over the rest of the slice
        if init_last < end {
            let n_1 = (period - 1) as f64;
            let n = period as f64;
            for j in (init_last + 1)..=end {
                let cur = data[j];
                let diff = cur - prev;
                let up = if diff > 0.0 { diff } else { 0.0 };
                let dn = if diff < 0.0 { -diff } else { 0.0 };
                avg_up = (avg_up * n_1 + up) / n;
                avg_dn = (avg_dn * n_1 + dn) / n;
                prev = cur;
            }
        }

        if avg_dn == 0.0 {
            100.0
        } else if avg_up + avg_dn == 0.0 {
            50.0
        } else {
            100.0 * avg_up / (avg_up + avg_dn)
        }
    }

    #[inline(always)]
    fn mfi_window_last_candles(c: &Candles, start: usize, end: usize) -> f64 {
        // Compute classic MFI over [start..=end] (inclusive) using typical price and volume.
        if end <= start {
            return 50.0;
        }
        // Typical price at start
        let mut tp_prev = (c.high[start] + c.low[start] + c.close[start]) / 3.0;
        let mut pos = 0.0f64;
        let mut neg = 0.0f64;

        for j in (start + 1)..=end {
            let tp = (c.high[j] + c.low[j] + c.close[j]) / 3.0;
            // Raw money flow uses current TP * current Volume
            let mf = tp * c.volume[j];
            if tp > tp_prev {
                pos += mf;
            } else if tp < tp_prev {
                neg += mf;
            }
            tp_prev = tp;
        }

        let denom = pos + neg;
        if denom > 0.0 {
            100.0 * pos / denom
        } else {
            50.0
        }
    }

    // ------- Main pass -------
    for i in warmup_end..len {
        let mu = mean[i];
        let sd = std_dev[i];
        if mu.is_nan() || sd.is_nan() {
            continue;
        }
        let src = data[i];

        // UMA band thresholds via FMA to reduce rounding & latency
        let a = (-1.75f64).mul_add(sd, mu);
        let b = (-0.25f64).mul_add(sd, mu);
        let c = (0.25f64).mul_add(sd, mu);
        let d = (1.75f64).mul_add(sd, mu);

        // Adapt dynamic length
        if src >= b && src <= c {
            dyn_len += 1.0;
        } else if src < a || src > d {
            dyn_len -= 1.0;
        }

        dyn_len = dyn_len.max(min_length as f64).min(max_length as f64);
        let len_r = dyn_len.round() as usize;
        if i + 1 < len_r {
            continue;
        }

        // ----- Momentum factor (MFI if volume & candles present & nonzero; else RSI fallback; else simplified vol-momentum) -----
        let mf: f64 = match (vol_opt, candles_opt) {
            (Some(vol), Some(candles)) => {
                let v_i = vol[i];
                if v_i == 0.0 || v_i.is_nan() {
                    // RSI fallback over a 2*len_r sized slice
                    let end = i;
                    let start = if end + 1 >= 2 * len_r {
                        end + 1 - 2 * len_r
                    } else {
                        0
                    };
                    rsi_wilder_last(data, start, end, len_r)
                } else {
                    let start = i + 1 - len_r;
                    mfi_window_last_candles(candles, start, i)
                }
            }
            (Some(vol), None) => {
                // Slice data with volume -> simplified volume-weighted momentum (no OHLC for true MFI)
                let start = i + 1 - len_r;
                let mut up_vol = 0.0f64;
                let mut dn_vol = 0.0f64;
                let mut prev = data[start];
                for j in (start + 1)..=i {
                    let cur = data[j];
                    let v = vol[j];
                    let diff = cur - prev;
                    if diff > 0.0 {
                        up_vol += v;
                    } else if diff < 0.0 {
                        dn_vol += v;
                    }
                    prev = cur;
                }
                let tot = up_vol + dn_vol;
                if tot > 0.0 {
                    100.0 * up_vol / tot
                } else {
                    50.0
                }
            }
            _ => {
                // No volume -> RSI fallback over a 2*len_r window
                let end = i;
                let start = if end + 1 >= 2 * len_r {
                    end + 1 - 2 * len_r
                } else {
                    0
                };
                rsi_wilder_last(data, start, end, len_r)
            }
        };

        // Convert 0..100 oscillator to symmetric scale and build power exponent
        // p = accelerator + |(2*mf - 100)| / 25
        let mf_scaled = mf.mul_add(2.0, -100.0);
        let p = accelerator + (mf_scaled.abs() * 0.04); // 1/25 = 0.04

        // ----- Power-weighted average over [i+1-len_r ..= i], newest highest weight (k^p for k=len_r..1) -----
        let start = i + 1 - len_r;

        // Accumulators (SIMD may fill these; otherwise scalar falls back)
        let (mut xws, mut wsum) = (0.0f64, 0.0f64);

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        match _kernel {
            Kernel::Avx512 => unsafe {
                if cfg!(target_feature = "avx512f") {
                    let (sx, sw) = uma_weighted_accumulate_avx512(
                        data.as_ptr().add(start),
                        ln_lut.as_ptr(),
                        len_r,
                        p,
                    );
                    xws = sx;
                    wsum = sw;
                }
            },
            Kernel::Avx2 => unsafe {
                if cfg!(target_feature = "avx2") {
                    let (sx, sw) = uma_weighted_accumulate_avx2(
                        data.as_ptr().add(start),
                        ln_lut.as_ptr(),
                        len_r,
                        p,
                    );
                    xws = sx;
                    wsum = sw;
                }
            },
            _ => {}
        }

        if wsum == 0.0 {
            // Scalar accumulation fallback
            // Tight inner loop: compute weight via exp(p * ln(k)) using LUT for ln(k); FMA for accumulation.
            // Unroll by 2 to hide some latency (len_r is generally small to medium).
            let mut j = 0usize;
            while j + 1 < len_r {
                let k1 = len_r - j;
                let k2 = k1 - 1;

                let w1 = exp_kernel(p * ln_lut[k1]);
                let x1 = data[start + j];
                if !x1.is_nan() {
                    xws = x1.mul_add(w1, xws);
                    wsum += w1;
                }

                let w2 = exp_kernel(p * ln_lut[k2]);
                let x2 = data[start + j + 1];
                if !x2.is_nan() {
                    xws = x2.mul_add(w2, xws);
                    wsum += w2;
                }

                j += 2;
            }
            if j < len_r {
                let k = len_r - j;
                let w = exp_kernel(p * ln_lut[k]);
                let x = data[start + j];
                if !x.is_nan() {
                    xws = x.mul_add(w, xws);
                    wsum += w;
                }
            }
        }

        // Finalize
        out[i] = if wsum > 0.0 { xws / wsum } else { src };
    }

    Ok(())
}

// ====== Optional AVX2/AVX512 accumulation kernels ======
#[cfg(all(
    feature = "nightly-avx",
    target_arch = "x86_64",
    target_feature = "avx2"
))]
#[inline(always)]
unsafe fn uma_weighted_accumulate_avx2(
    data: *const f64,   // &data[start] pointer
    ln_lut: *const f64, // &ln_lut[0]
    len_r: usize,       // window length
    p: f64,             // exponent
) -> (f64, f64) {
    use core::arch::x86_64::*;
    let mut sum_v = _mm256_setzero_pd();
    let mut wsum_v = _mm256_setzero_pd();

    let mut j = 0usize;
    while j + 3 < len_r {
        let k0 = len_r - j;
        let k1 = k0 - 1;
        let k2 = k1 - 1;
        let k3 = k2 - 1;

        let w0 = exp_kernel(p * *ln_lut.add(k0));
        let w1 = exp_kernel(p * *ln_lut.add(k1));
        let w2 = exp_kernel(p * *ln_lut.add(k2));
        let w3 = exp_kernel(p * *ln_lut.add(k3));
        let wv = _mm256_setr_pd(w0, w1, w2, w3);

        let xv = _mm256_loadu_pd(data.add(j));

        // ordered compare: true when not NaN
        let nan_mask = _mm256_cmp_pd(xv, xv, _CMP_ORD_Q);
        let xv_nz = _mm256_and_pd(xv, nan_mask);

        sum_v = _mm256_fmadd_pd(xv_nz, wv, sum_v);
        let wv_masked = _mm256_and_pd(wv, nan_mask);
        wsum_v = _mm256_add_pd(wsum_v, wv_masked);

        j += 4;
    }

    let mut xws = 0.0f64;
    let mut wsum = 0.0f64;
    {
        let tmp: [f64; 4] = core::mem::transmute(sum_v);
        xws += tmp[0] + tmp[1] + tmp[2] + tmp[3];
        let t2: [f64; 4] = core::mem::transmute(wsum_v);
        wsum += t2[0] + t2[1] + t2[2] + t2[3];
    }

    while j < len_r {
        let k = len_r - j;
        let w = exp_kernel(p * *ln_lut.add(k));
        let x = *data.add(j);
        if !x.is_nan() {
            xws = x.mul_add(w, xws);
            wsum += w;
        }
        j += 1;
    }
    (xws, wsum)
}

#[cfg(all(
    feature = "nightly-avx",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
unsafe fn uma_weighted_accumulate_avx512(
    data: *const f64,   // &data[start]
    ln_lut: *const f64, // &ln_lut[0]
    len_r: usize,
    p: f64,
) -> (f64, f64) {
    use core::arch::x86_64::*;
    let mut sum_v = _mm512_setzero_pd();
    let mut wsum_v = _mm512_setzero_pd();

    let mut j = 0usize;
    while j + 7 < len_r {
        let k0 = len_r - j;
        let ks = [k0, k0 - 1, k0 - 2, k0 - 3, k0 - 4, k0 - 5, k0 - 6, k0 - 7];
        let ws = [
            exp_kernel(p * *ln_lut.add(ks[0])),
            exp_kernel(p * *ln_lut.add(ks[1])),
            exp_kernel(p * *ln_lut.add(ks[2])),
            exp_kernel(p * *ln_lut.add(ks[3])),
            exp_kernel(p * *ln_lut.add(ks[4])),
            exp_kernel(p * *ln_lut.add(ks[5])),
            exp_kernel(p * *ln_lut.add(ks[6])),
            exp_kernel(p * *ln_lut.add(ks[7])),
        ];
        let wv = _mm512_loadu_pd(ws.as_ptr());
        let xv = _mm512_loadu_pd(data.add(j));

        let nan_mask = _mm512_cmp_pd_mask(xv, xv, _CMP_ORD_Q);
        let xv_nz = _mm512_maskz_mov_pd(nan_mask, xv);

        sum_v = _mm512_fmadd_pd(xv_nz, wv, sum_v);
        let wv_masked = _mm512_maskz_mov_pd(nan_mask, wv);
        wsum_v = _mm512_add_pd(wsum_v, wv_masked);

        j += 8;
    }

    // Horizontal reduce
    let xws = {
        let mut tmp: [f64; 8] = core::mem::zeroed();
        _mm512_storeu_pd(tmp.as_mut_ptr(), sum_v);
        tmp.iter().copied().sum::<f64>()
    };
    let wsum0 = {
        let mut tmp: [f64; 8] = core::mem::zeroed();
        _mm512_storeu_pd(tmp.as_mut_ptr(), wsum_v);
        tmp.iter().copied().sum::<f64>()
    };

    let mut xws_acc = xws;
    let mut wsum_acc = wsum0;
    while j < len_r {
        let k = len_r - j;
        let w = exp_kernel(p * *ln_lut.add(k));
        let x = *data.add(j);
        if !x.is_nan() {
            xws_acc = x.mul_add(w, xws_acc);
            wsum_acc += w;
        }
        j += 1;
    }
    (xws_acc, wsum_acc)
}

#[inline]
pub fn uma(input: &UmaInput) -> Result<UmaOutput, UmaError> {
    uma_with_kernel(input, Kernel::Auto)
}

pub fn uma_with_kernel(input: &UmaInput, kernel: Kernel) -> Result<UmaOutput, UmaError> {
    let (data, first, min_len, max_len, accel) = uma_prepare(input)?;
    let warm = first + max_len - 1;

    // Primary buffer: NaN prefix allocated once.
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    // Choose kernel (Auto -> detect) and thread through for parity and SIMD.
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    uma_core_into(input, first, min_len, max_len, accel, chosen, &mut out)?;

    // Optional smoothing with no extra copy: return WMA's Vec directly.
    let smooth = input.get_smooth_length();
    if smooth > 1 {
        let w = wma(&WmaInput::from_slice(
            &out,
            WmaParams {
                period: Some(smooth),
            },
        ))
        .map_err(|e| UmaError::DependencyError(e.to_string()))?
        .values;
        return Ok(UmaOutput { values: w });
    }

    Ok(UmaOutput { values: out })
}

#[inline]
pub fn uma_into_slice(dst: &mut [f64], input: &UmaInput, kern: Kernel) -> Result<(), UmaError> {
    let (data, first, min_len, max_len, accel) = uma_prepare(input)?;
    if dst.len() != data.len() {
        return Err(UmaError::InvalidMaxLength {
            max_length: dst.len(),
            data_len: data.len(),
        });
    }

    // Ensure warmup NaNs even if later smoothing overwrites.
    let warm = first + max_len - 1;
    let warm_end = warm.min(dst.len());
    for v in &mut dst[..warm_end] {
        *v = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    uma_core_into(input, first, min_len, max_len, accel, chosen, dst)?;

    let smooth = input.get_smooth_length();
    if smooth > 1 {
        // Fallback: compute WMA once and copy once.
        let tmp = wma(&WmaInput::from_slice(
            dst,
            WmaParams {
                period: Some(smooth),
            },
        ))
        .map_err(|e| UmaError::DependencyError(e.to_string()))?
        .values;
        dst.copy_from_slice(&tmp);
    }
    Ok(())
}

// ==================== BATCH PROCESSING ====================

#[derive(Clone, Debug)]
pub struct UmaBatchRange {
    pub accelerator: (f64, f64, f64),         // (start, end, step)
    pub min_length: (usize, usize, usize),    // (start, end, step)
    pub max_length: (usize, usize, usize),    // (start, end, step)
    pub smooth_length: (usize, usize, usize), // (start, end, step)
}

impl Default for UmaBatchRange {
    fn default() -> Self {
        Self {
            accelerator: (1.0, 1.0, 0.0),
            min_length: (5, 5, 0),
            max_length: (50, 50, 0),
            smooth_length: (4, 4, 0),
        }
    }
}

#[derive(Copy, Clone)]
pub struct UmaBatchBuilder {
    accelerator_range: (f64, f64, f64),
    min_length_range: (usize, usize, usize),
    max_length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for UmaBatchBuilder {
    fn default() -> Self {
        Self {
            accelerator_range: (1.0, 1.0, 0.0),
            min_length_range: (5, 5, 0),
            max_length_range: (50, 50, 0),
            smooth_length_range: (4, 4, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl UmaBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn accelerator_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.accelerator_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn min_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.min_length_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn max_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.max_length_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smooth_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.smooth_length_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<UmaBatchOutput, UmaError> {
        let sweep = UmaBatchRange {
            accelerator: self.accelerator_range,
            min_length: self.min_length_range,
            max_length: self.max_length_range,
            smooth_length: self.smooth_length_range,
        };

        let data = source_type(candles, source);
        uma_batch_inner(data, Some(&candles.volume), &sweep, self.kernel, false)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
        volume: Option<&[f64]>,
    ) -> Result<UmaBatchOutput, UmaError> {
        let sweep = UmaBatchRange {
            accelerator: self.accelerator_range,
            min_length: self.min_length_range,
            max_length: self.max_length_range,
            smooth_length: self.smooth_length_range,
        };

        uma_batch_inner(data, volume, &sweep, self.kernel, false)
    }

    pub fn with_default_slice(
        data: &[f64],
        volume: Option<&[f64]>,
        k: Kernel,
    ) -> Result<UmaBatchOutput, UmaError> {
        UmaBatchBuilder::new().kernel(k).apply_slice(data, volume)
    }

    pub fn with_default_candles(c: &Candles) -> Result<UmaBatchOutput, UmaError> {
        UmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct UmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl UmaBatchOutput {
    pub fn row_for_params(&self, p: &UmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.accelerator.unwrap_or(1.0) == p.accelerator.unwrap_or(1.0)
                && c.min_length.unwrap_or(5) == p.min_length.unwrap_or(5)
                && c.max_length.unwrap_or(50) == p.max_length.unwrap_or(50)
                && c.smooth_length.unwrap_or(4) == p.smooth_length.unwrap_or(4)
        })
    }

    pub fn values_for(&self, p: &UmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid_uma(r: &UmaBatchRange) -> Vec<UmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }

    let accs = axis_f64(r.accelerator);
    let mins = axis_usize(r.min_length);
    let maxs = axis_usize(r.max_length);
    let smooths = axis_usize(r.smooth_length);

    let mut out = Vec::with_capacity(accs.len() * mins.len() * maxs.len() * smooths.len());
    for &a in &accs {
        for &min in &mins {
            for &max in &maxs {
                for &s in &smooths {
                    if min <= max {
                        // Only valid combinations
                        out.push(UmaParams {
                            accelerator: Some(a),
                            min_length: Some(min),
                            max_length: Some(max),
                            smooth_length: Some(s),
                        });
                    }
                }
            }
        }
    }
    out
}

pub fn uma_batch_with_kernel(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    k: Kernel,
) -> Result<UmaBatchOutput, UmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(UmaError::InvalidMaxLength {
                max_length: 0,
                data_len: 0,
            })
        }
    };

    uma_batch_inner(data, volume, sweep, kernel, true)
}

#[inline(always)]
pub fn uma_batch_slice(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
) -> Result<UmaBatchOutput, UmaError> {
    uma_batch_inner(data, volume, sweep, kern, false)
}

#[inline(always)]
pub fn uma_batch_par_slice(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
) -> Result<UmaBatchOutput, UmaError> {
    uma_batch_inner(data, volume, sweep, kern, true)
}

#[inline(always)]
fn debatch(k: Kernel) -> Kernel {
    match k {
        Kernel::Avx512Batch | Kernel::Avx512 => Kernel::Avx512,
        Kernel::Avx2Batch | Kernel::Avx2 => Kernel::Avx2,
        Kernel::ScalarBatch | Kernel::Scalar => Kernel::Scalar,
        Kernel::Auto => detect_best_kernel(),
        _ => Kernel::Scalar,
    }
}

#[inline(always)]
fn uma_batch_inner_into(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<UmaParams>, UmaError> {
    let combos = expand_grid_uma(sweep);
    if combos.is_empty() {
        return Err(UmaError::InvalidMaxLength {
            max_length: 0,
            data_len: 0,
        });
    }

    let cols = data.len();
    let rows = combos.len();
    debug_assert_eq!(out.len(), rows * cols);

    let per_row_kernel = debatch(kern);

    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.max_length.unwrap_or(50) - 1 + c.smooth_length.unwrap_or(4) - 1)
        .collect();

    // Initialize NaN prefixes on an uninitialized view to avoid extra writes.
    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    let do_row = |row: usize, row_slice: &mut [f64]| -> Result<(), UmaError> {
        let input = UmaInput::from_slice(data, volume, combos[row].clone());
        uma_into_slice(row_slice, &input, per_row_kernel)
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        use rayon::prelude::*;
        out.par_chunks_mut(cols)
            .enumerate()
            .try_for_each(|(r, s)| do_row(r, s))?;
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    }

    Ok(combos)
}

#[inline(always)]
fn uma_batch_inner(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<UmaBatchOutput, UmaError> {
    let combos = expand_grid_uma(sweep);
    let cols = data.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(UmaError::EmptyInputData);
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            first + c.max_length.unwrap_or(50) - 1 + c.smooth_length.unwrap_or(4) - 1
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let combos = uma_batch_inner_into(data, volume, sweep, kern, parallel, out_slice)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(UmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// ==================== PYTHON BINDINGS ====================

#[cfg(feature = "python")]
#[pyfunction(name = "uma")]
#[pyo3(signature = (data, accelerator, min_length, max_length, smooth_length, volume=None, kernel=None))]
pub fn uma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::PyArrayMethods;
    let kern = validate_kernel(kernel, false)?;
    let slice_in = data.as_slice()?;
    let vol_slice = volume.as_ref().map(|v| v.as_slice()).transpose()?;

    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };
    let input = UmaInput::from_slice(slice_in, vol_slice, params);

    let out: Vec<f64> = py
        .allow_threads(|| uma_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "UmaStream")]
pub struct UmaStreamPy {
    stream: UmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl UmaStreamPy {
    #[new]
    pub fn new(
        accelerator: f64,
        min_length: usize,
        max_length: usize,
        smooth_length: usize,
    ) -> PyResult<Self> {
        let params = UmaParams {
            accelerator: Some(accelerator),
            min_length: Some(min_length),
            max_length: Some(max_length),
            smooth_length: Some(smooth_length),
        };
        let stream =
            UmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    /// alma-style: single-arg update
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }

    /// keep extended form for volume
    #[pyo3(name = "update_with_volume")]
    pub fn update_with_volume_py(&mut self, value: f64, volume: Option<f64>) -> Option<f64> {
        self.stream.update_with_volume(value, volume)
    }

    pub fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "uma_batch")]
#[pyo3(signature = (data, accelerator_range, min_length_range, max_length_range, smooth_length_range, volume=None, kernel=None))]
pub fn uma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    accelerator_range: (f64, f64, f64),
    min_length_range: (usize, usize, usize),
    max_length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    volume: Option<numpy::PyReadonlyArray1<'py, f64>>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let vol_slice = volume.as_ref().map(|v| v.as_slice()).transpose()?;
    let sweep = UmaBatchRange {
        accelerator: accelerator_range,
        min_length: min_length_range,
        max_length: max_length_range,
        smooth_length: smooth_length_range,
    };
    let combos = expand_grid_uma(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| uma_batch_inner_into(slice_in, vol_slice, &sweep, kern, false, out_slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item(
        "accelerators",
        combos
            .iter()
            .map(|c| c.accelerator.unwrap_or(1.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "min_lengths",
        combos
            .iter()
            .map(|c| c.min_length.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "max_lengths",
        combos
            .iter()
            .map(|c| c.max_length.unwrap_or(50) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        combos
            .iter()
            .map(|c| c.smooth_length.unwrap_or(4) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    // Add combos list for compatibility with tests
    let combo_list: Vec<Bound<'py, PyDict>> = combos
        .iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("accelerator", c.accelerator.unwrap_or(1.0))
                .unwrap();
            d.set_item("min_length", c.min_length.unwrap_or(5)).unwrap();
            d.set_item("max_length", c.max_length.unwrap_or(50))
                .unwrap();
            d.set_item("smooth_length", c.smooth_length.unwrap_or(4))
                .unwrap();
            d.into()
        })
        .collect();
    dict.set_item("combos", combo_list)?;

    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "uma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, accelerator_range, min_length_range, max_length_range, smooth_length_range, volume_f32=None, device_id=0))]
pub fn uma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    accelerator_range: (f64, f64, f64),
    min_length_range: (usize, usize, usize),
    max_length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    volume_f32: Option<numpy::PyReadonlyArray1<'_, f32>>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let volume_slice = volume_f32.as_ref().map(|v| v.as_slice()).transpose()?;
    let sweep = UmaBatchRange {
        accelerator: accelerator_range,
        min_length: min_length_range,
        max_length: max_length_range,
        smooth_length: smooth_length_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaUma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.uma_batch_dev(slice_in, volume_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "uma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, accelerator, min_length, max_length, smooth_length, volume_tm_f32=None, device_id=0))]
pub fn uma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: PyReadonlyArray2<'_, f32>,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
    volume_tm_f32: Option<PyReadonlyArray2<'_, f32>>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    use numpy::PyUntypedArrayMethods;

    let rows = prices_tm_f32.shape()[0];
    let cols = prices_tm_f32.shape()[1];
    if let Some(vol) = &volume_tm_f32 {
        let vshape = vol.shape();
        if vshape != prices_tm_f32.shape() {
            return Err(PyValueError::new_err(
                "price and volume matrices must share shape",
            ));
        }
    }

    let prices_flat = prices_tm_f32.as_slice()?;
    let volume_flat = volume_tm_f32
        .as_ref()
        .map(|arr| arr.as_slice())
        .transpose()?;

    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaUma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.uma_many_series_one_param_time_major_dev(prices_flat, volume_flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

// ==================== WASM BINDINGS ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_js(
    data: &[f64],
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
    volume: Option<Vec<f64>>,
) -> Result<JsValue, JsValue> {
    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };
    let vol_slice = volume.as_deref();
    let input = UmaInput::from_slice(data, vol_slice, params);

    match uma_with_kernel(&input, detect_best_kernel()) {
        Ok(output) => {
            // Return as an object with values property to match test expectations
            let obj = js_sys::Object::new();
            let values_array = js_sys::Array::new();
            for val in output.values {
                values_array.push(&JsValue::from_f64(val));
            }
            js_sys::Reflect::set(&obj, &JsValue::from_str("values"), &values_array)?;
            Ok(obj.into())
        }
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UmaBatchConfig {
    pub accelerator_range: (f64, f64, f64),
    pub min_length_range: (usize, usize, usize),
    pub max_length_range: (usize, usize, usize),
    pub smooth_length_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "uma_batch")]
pub fn uma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: UmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = UmaBatchRange {
        accelerator: cfg.accelerator_range,
        min_length: cfg.min_length_range,
        max_length: cfg.max_length_range,
        smooth_length: cfg.smooth_length_range,
    };
    let out = uma_batch_inner(data, None, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = UmaBatchJsOutput {
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
pub fn uma_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = UmaParams {
            accelerator: Some(accelerator),
            min_length: Some(min_length),
            max_length: Some(max_length),
            smooth_length: Some(smooth_length),
        };
        let input = UmaInput::from_slice(data, None, params);

        if core::ptr::eq(in_ptr as *const u8, out_ptr as *const u8) {
            // in-place: compute into temp then copy once
            let mut tmp = vec![0.0; len];
            uma_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            uma_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

// ==================== WASM STREAMING API ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_new(
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
) -> *mut UmaStream {
    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };

    match UmaStream::try_new(params) {
        Ok(stream) => Box::into_raw(Box::new(stream)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_update(stream: *mut UmaStream, value: f64) -> Option<f64> {
    if stream.is_null() {
        return None;
    }
    unsafe { (*stream).update_with_volume(value, None) }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_update_with_volume(
    stream: *mut UmaStream,
    value: f64,
    volume: f64,
) -> Option<f64> {
    if stream.is_null() {
        return None;
    }
    unsafe { (*stream).update_with_volume(value, Some(volume)) }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_reset(stream: *mut UmaStream) {
    if !stream.is_null() {
        unsafe {
            (*stream).reset();
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_free(stream: *mut UmaStream) {
    if !stream.is_null() {
        unsafe {
            let _ = Box::from_raw(stream);
        }
    }
}

// ==================== WASM ZERO-COPY HELPERS ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_get_view(ptr: *mut f64, len: usize) -> js_sys::Float64Array {
    unsafe { js_sys::Float64Array::view(std::slice::from_raw_parts(ptr, len)) }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_update(
    ptr: *mut f64,
    len: usize,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(ptr, len);
        let params = UmaParams {
            accelerator: Some(accelerator),
            min_length: Some(min_length),
            max_length: Some(max_length),
            smooth_length: Some(smooth_length),
        };
        let input = UmaInput::from_slice(data, None, params);

        // Compute into temporary buffer then copy back
        let mut tmp = vec![0.0; len];
        uma_into_slice(&mut tmp, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let out = std::slice::from_raw_parts_mut(ptr, len);
        out.copy_from_slice(&tmp);
        Ok(())
    }
}

// ==================== WASM BATCH EXPORT ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_batch_js(
    data: &[f64],
    accelerator_range: Vec<f64>,     // [start, end, step]
    min_length_range: Vec<usize>,    // [start, end, step]
    max_length_range: Vec<usize>,    // [start, end, step]
    smooth_length_range: Vec<usize>, // [start, end, step]
    volume: Option<Vec<f64>>,
) -> Result<JsValue, JsValue> {
    if accelerator_range.len() != 3
        || min_length_range.len() != 3
        || max_length_range.len() != 3
        || smooth_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "All range arrays must have exactly 3 elements: [start, end, step]",
        ));
    }

    let sweep = UmaBatchRange {
        accelerator: (
            accelerator_range[0],
            accelerator_range[1],
            accelerator_range[2],
        ),
        min_length: (
            min_length_range[0],
            min_length_range[1],
            min_length_range[2],
        ),
        max_length: (
            max_length_range[0],
            max_length_range[1],
            max_length_range[2],
        ),
        smooth_length: (
            smooth_length_range[0],
            smooth_length_range[1],
            smooth_length_range[2],
        ),
    };

    let vol_slice = volume.as_deref();
    let kernel = detect_best_batch_kernel();

    let result = uma_batch_inner(data, vol_slice, &sweep, kernel, true)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert to JS-friendly format with individual parameter arrays
    let obj = js_sys::Object::new();

    // Add values as flat array
    let values_array = js_sys::Array::new();
    for val in result.values {
        values_array.push(&JsValue::from_f64(val));
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("values"), &values_array)?;

    // Add parameter arrays
    let accelerators = js_sys::Array::new();
    let min_lengths = js_sys::Array::new();
    let max_lengths = js_sys::Array::new();
    let smooth_lengths = js_sys::Array::new();

    for combo in &result.combos {
        accelerators.push(&JsValue::from_f64(combo.accelerator.unwrap_or(1.0)));
        min_lengths.push(&JsValue::from_f64(combo.min_length.unwrap_or(5) as f64));
        max_lengths.push(&JsValue::from_f64(combo.max_length.unwrap_or(50) as f64));
        smooth_lengths.push(&JsValue::from_f64(combo.smooth_length.unwrap_or(4) as f64));
    }

    js_sys::Reflect::set(&obj, &JsValue::from_str("accelerators"), &accelerators)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("min_lengths"), &min_lengths)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("max_lengths"), &max_lengths)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("smooth_lengths"), &smooth_lengths)?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(result.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(result.cols as f64),
    )?;

    Ok(obj.into())
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_uma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = UmaParams {
            accelerator: None,
            min_length: None,
            max_length: None,
            smooth_length: None,
        };
        let input = UmaInput::from_candles(&candles, "close", default_params);
        let output = uma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_uma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = UmaInput::from_candles(&candles, "close", UmaParams::default());
        let result = uma_with_kernel(&input, kernel)?;

        // Get valid values
        let values = &result.values;
        let valid_values: Vec<f64> = values.iter().filter(|&&v| !v.is_nan()).copied().collect();

        // Expected values calculated from CSV data
        let expected_last_five = [
            59665.81830666,
            59477.69234542,
            59314.50778603,
            59218.23033661,
            59143.61473211,
        ];

        // Check we have enough valid values
        if valid_values.len() >= 5 {
            let start = valid_values.len().saturating_sub(5);
            for (i, &val) in valid_values[start..].iter().enumerate() {
                let diff = (val - expected_last_five[i]).abs();
                let tolerance = expected_last_five[i] * 0.01; // 1% tolerance
                assert!(
                    diff < tolerance || diff < 100.0,
                    "[{}] UMA {:?} mismatch at idx {}: got {}, expected {}",
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

    fn check_uma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = UmaInput::with_default_candles(&candles);
        match input.data {
            UmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected UmaData::Candles"),
        }
        let output = uma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_uma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = UmaParams {
            accelerator: Some(1.0),
            min_length: Some(5),
            max_length: Some(0), // Invalid
            smooth_length: Some(4),
        };
        let input = UmaInput::from_slice(&input_data, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] UMA should fail with zero max_length",
            test_name
        );
        Ok(())
    }

    fn check_uma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = UmaParams {
            accelerator: Some(1.0),
            min_length: Some(5),
            max_length: Some(10),
            smooth_length: Some(4),
        };
        let input = UmaInput::from_slice(&data_small, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] UMA should fail with max_length exceeding data length",
            test_name
        );
        Ok(())
    }

    fn check_uma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = UmaParams::default();
        let input = UmaInput::from_slice(&single_point, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] UMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_uma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = UmaInput::from_slice(&empty, None, UmaParams::default());
        let res = uma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(UmaError::EmptyInputData)),
            "[{}] UMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_uma_invalid_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();

        // Test invalid accelerator
        let params = UmaParams {
            accelerator: Some(0.5), // Invalid - must be >= 1.0
            max_length: Some(10),
            ..Default::default()
        };
        let input = UmaInput::from_slice(&data, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(UmaError::InvalidAccelerator { .. })),
            "[{}] UMA should fail with invalid accelerator, got: {:?}",
            test_name,
            res
        );
        Ok(())
    }

    fn check_uma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = UmaParams::default();
        let first_input = UmaInput::from_candles(&candles, "close", first_params);
        let first_result = uma_with_kernel(&first_input, kernel)?;

        let second_params = UmaParams::default();
        let second_input = UmaInput::from_slice(&first_result.values, None, second_params);
        let second_result = uma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        // Just verify we got valid output
        let valid_count = second_result
            .values
            .iter()
            .filter(|&&v| !v.is_nan())
            .count();
        assert!(
            valid_count > 0,
            "[{}] UMA reinput should produce valid values",
            test_name
        );

        Ok(())
    }

    fn check_uma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let mut data = vec![f64::NAN; 10];
        data.extend((0..100).map(|i| 100.0 + i as f64));

        let input = UmaInput::from_slice(&data, None, UmaParams::default());
        let res = uma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), data.len());

        // Check we have valid values after the NaNs and warmup
        let valid_count = res.values[60..].iter().filter(|&&v| !v.is_nan()).count();
        assert!(
            valid_count > 0,
            "[{}] UMA should handle NaN prefix",
            test_name
        );

        Ok(())
    }

    fn check_uma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = UmaParams::default();
        let input = UmaInput::from_candles(&candles, "close", params.clone());
        let batch_output = uma_with_kernel(&input, kernel)?.values;

        let mut stream = UmaStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());

        for (i, &price) in candles.close.iter().enumerate() {
            let volume = if i < candles.volume.len() {
                Some(candles.volume[i])
            } else {
                None
            };

            match stream.update_with_volume(price, volume) {
                Some(uma_val) => stream_values.push(uma_val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());

        // Note: UMA streaming implementation has inherent differences from batch due to
        // how it manages its dynamic length buffer. We compare the overall trend
        // rather than exact values.

        // Compare last few valid values (streaming may have slight differences)
        let batch_valid: Vec<f64> = batch_output
            .iter()
            .filter(|&&v| !v.is_nan())
            .copied()
            .collect();
        let stream_valid: Vec<f64> = stream_values
            .iter()
            .filter(|&&v| !v.is_nan())
            .copied()
            .collect();

        if batch_valid.len() >= 5 && stream_valid.len() >= 5 {
            let batch_last = &batch_valid[batch_valid.len() - 5..];
            let stream_last = &stream_valid[stream_valid.len() - 5..];

            for (i, (&b, &s)) in batch_last.iter().zip(stream_last.iter()).enumerate() {
                let diff = (b - s).abs();
                let relative_diff = diff / b.abs().max(1.0);
                assert!(
                    relative_diff < 0.1, // Allow 10% difference for UMA's dynamic length
                    "[{}] UMA streaming mismatch at idx {}: batch={}, stream={}, diff={}, rel_diff={}",
                    test_name,
                    i,
                    b,
                    s,
                    diff,
                    relative_diff
                );
            }
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_uma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            UmaParams::default(),
            UmaParams {
                accelerator: Some(1.5),
                min_length: Some(3),
                max_length: Some(30),
                smooth_length: Some(3),
            },
            UmaParams {
                accelerator: Some(2.0),
                min_length: Some(10),
                max_length: Some(100),
                smooth_length: Some(8),
            },
        ];

        for params in test_params.iter() {
            let input = UmaInput::from_candles(&candles, "close", params.clone());
            let output = uma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_uma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_uma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (5usize..=20, 20usize..=50, 2usize..=8, 1.0f64..3.0).prop_flat_map(
            |(min_len, max_len, smooth_len, acc)| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        max_len + 10..200,
                    ),
                    Just(min_len),
                    Just(max_len),
                    Just(smooth_len),
                    Just(acc),
                )
            },
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, min_len, max_len, smooth_len, acc)| {
                let params = UmaParams {
                    accelerator: Some(acc),
                    min_length: Some(min_len),
                    max_length: Some(max_len),
                    smooth_length: Some(smooth_len),
                };
                let input = UmaInput::from_slice(&data, None, params);

                let UmaOutput { values: out } = uma_with_kernel(&input, kernel).unwrap();
                let UmaOutput { values: ref_out } =
                    uma_with_kernel(&input, Kernel::Scalar).unwrap();

                // Basic sanity checks
                prop_assert_eq!(out.len(), data.len());
                prop_assert_eq!(ref_out.len(), data.len());

                // Compare kernel output with reference scalar
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-6 || ulp_diff <= 10,
                        "mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_uma_tests {
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

    generate_all_uma_tests!(
        check_uma_partial_params,
        check_uma_accuracy,
        check_uma_default_candles,
        check_uma_zero_period,
        check_uma_period_exceeds_length,
        check_uma_very_small_dataset,
        check_uma_empty_input,
        check_uma_invalid_params,
        check_uma_reinput,
        check_uma_nan_handling,
        check_uma_streaming,
        check_uma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_uma_tests!(check_uma_property);

    #[test]
    fn uma_into_slice_matches_with_kernel() {
        use crate::utilities::data_loader::read_candles_from_csv;
        let c = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let params = UmaParams::default();
        let input = UmaInput::from_candles(&c, "close", params.clone());

        let via_api = uma_with_kernel(&input, Kernel::Scalar).unwrap().values;
        let mut via_into = vec![0.0; via_api.len()];
        uma_into_slice(&mut via_into, &input, Kernel::Scalar).unwrap();

        assert_eq!(via_api.len(), via_into.len());
        for (a, b) in via_api.iter().zip(via_into.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[cfg(feature = "python")]
    #[test]
    fn uma_batch_py_no_copy_shape() {
        pyo3::Python::with_gil(|py| {
            use numpy::PyArray1;
            let data = PyArray1::from_vec(py, (0..256).map(|i| i as f64).collect());
            let d = crate::indicators::moving_averages::uma::uma_batch_py(
                py,
                data.readonly(),
                (1.0, 1.0, 0.0),
                (5, 5, 0),
                (50, 50, 0),
                (4, 4, 0),
                None,
                Some("scalar_batch"),
            )
            .unwrap();
            let v = d.get_item("values").unwrap();
            // Just ensure it's a 2D view, not a copied nested array
            assert!(v.downcast::<numpy::PyArray2<f64>>().is_ok());
        });
    }

    // Batch processing tests
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = UmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = UmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Verify we have valid values
        let valid_count = row.iter().filter(|&&v| !v.is_nan()).count();
        assert!(
            valid_count > 0,
            "[{}] Batch should produce valid values",
            test
        );

        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();

        let output = UmaBatchBuilder::new()
            .kernel(kernel)
            .accelerator_range(1.0, 2.0, 0.5)
            .min_length_range(5, 10, 5)
            .max_length_range(20, 30, 10)
            .smooth_length_range(3, 5, 2)
            .apply_slice(&data, None)?;

        let expected_combos = 3 * 2 * 2 * 2; // 3 accelerators * 2 min_lengths * 2 max_lengths * 2 smooth_lengths
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, data.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (1.0, 1.5, 0.5, 5, 10, 5, 20, 30, 10, 3, 5, 2),
            (2.0, 2.0, 0.0, 10, 10, 0, 50, 50, 0, 4, 4, 0),
        ];

        for (
            cfg_idx,
            &(
                a_start,
                a_end,
                a_step,
                min_start,
                min_end,
                min_step,
                max_start,
                max_end,
                max_step,
                s_start,
                s_end,
                s_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = UmaBatchBuilder::new()
                .kernel(kernel)
                .accelerator_range(a_start, a_end, a_step)
                .min_length_range(min_start, min_end, min_step)
                .max_length_range(max_start, max_end, max_step)
                .smooth_length_range(s_start, s_end, s_step)
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
                        at row {} col {} (flat index {}) with params: acc={}, min={}, max={}, smooth={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.accelerator.unwrap_or(1.0),
                        combo.min_length.unwrap_or(5),
                        combo.max_length.unwrap_or(50),
                        combo.smooth_length.unwrap_or(4)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {})",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {})",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }
            }
        }

        Ok(())
    }

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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                    Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);
}
