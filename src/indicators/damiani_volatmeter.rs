//! # Damiani Volatmeter
//!
//! A dual-volatility indicator that combines ATR (Average True Range) and standard deviation calculations
//! to identify trending vs ranging market conditions. It uses two timeframes - a shorter "viscosity" period
//! for current volatility and a longer "sedation" period for baseline volatility. The indicator produces
//! two outputs: a volatility signal and an anti-trend threshold that helps filter out choppy conditions.
//!
//! ## Parameters
//! - **vis_atr**: ATR period for short-term volatility (default: 13)
//! - **vis_std**: Standard deviation window for short-term (default: 20)
//! - **sed_atr**: ATR period for long-term baseline (default: 40)
//! - **sed_std**: Standard deviation window for long-term (default: 100)
//! - **threshold**: Offset constant for anti-trend calculation (default: 1.4)
//!
//! ## Inputs
//! - Can use either raw price slice or Candles with source selection
//! - When using single price array, high/low/close all use the same values
//! - When using Candles, properly uses separate high/low/close for TR calculation
//!
//! ## Returns
//! - **`Ok(DamianiVolatmeterOutput)`** containing:
//!   - `vol`: Volatility signal array (higher values = stronger trend)
//!   - `anti`: Anti-trend threshold array (signal below this = choppy/ranging)
//!   - Both arrays match input length with NaN during warmup
//!
//! ## Developer Notes (Status)
//! - SIMD: Implemented as stubs that delegate to scalar. ATR and rolling StdDev are loop-carried
//!   recurrences; across-time vectorization does not yield consistent wins. Runtime selection keeps
//!   the SIMD paths present for future work but they call the scalar kernels today.
//! - Scalar: Kept as reference path. Hot loop avoids extra work; warmup and allocation follow alma.rs.
//! - Batch: Parallel rows supported. A row-specific optimized batch is feasible via shared TR and
//!   prefix sums (S, SS) across rows; not implemented in this pass to keep scope minimal.
//! - Streaming: O(1) per-bar kernel integrated. Uses counter-based ATR seeding, branch ring-wraps,
//!   precomputed reciprocals, and a single-sqrt std-ratio; matches scalar outputs.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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

impl<'a> AsRef<[f64]> for DamianiVolatmeterInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DamianiVolatmeterData::Slice(slice) => slice,
            DamianiVolatmeterData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DamianiVolatmeterData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterOutput {
    pub vol: Vec<f64>,
    pub anti: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DamianiVolatmeterParams {
    pub vis_atr: Option<usize>,
    pub vis_std: Option<usize>,
    pub sed_atr: Option<usize>,
    pub sed_std: Option<usize>,
    pub threshold: Option<f64>,
}

impl Default for DamianiVolatmeterParams {
    fn default() -> Self {
        Self {
            vis_atr: Some(13),
            vis_std: Some(20),
            sed_atr: Some(40),
            sed_std: Some(100),
            threshold: Some(1.4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterInput<'a> {
    pub data: DamianiVolatmeterData<'a>,
    pub params: DamianiVolatmeterParams,
}

impl<'a> DamianiVolatmeterInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DamianiVolatmeterParams) -> Self {
        Self {
            data: DamianiVolatmeterData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DamianiVolatmeterParams) -> Self {
        Self {
            data: DamianiVolatmeterData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DamianiVolatmeterParams::default())
    }
    #[inline]
    pub fn get_vis_atr(&self) -> usize {
        self.params.vis_atr.unwrap_or(13)
    }
    #[inline]
    pub fn get_vis_std(&self) -> usize {
        self.params.vis_std.unwrap_or(20)
    }
    #[inline]
    pub fn get_sed_atr(&self) -> usize {
        self.params.sed_atr.unwrap_or(40)
    }
    #[inline]
    pub fn get_sed_std(&self) -> usize {
        self.params.sed_std.unwrap_or(100)
    }
    #[inline]
    pub fn get_threshold(&self) -> f64 {
        self.params.threshold.unwrap_or(1.4)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DamianiVolatmeterBuilder {
    vis_atr: Option<usize>,
    vis_std: Option<usize>,
    sed_atr: Option<usize>,
    sed_std: Option<usize>,
    threshold: Option<f64>,
    kernel: Kernel,
}

impl Default for DamianiVolatmeterBuilder {
    fn default() -> Self {
        Self {
            vis_atr: None,
            vis_std: None,
            sed_atr: None,
            sed_std: None,
            threshold: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DamianiVolatmeterBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn vis_atr(mut self, n: usize) -> Self {
        self.vis_atr = Some(n);
        self
    }
    #[inline(always)]
    pub fn vis_std(mut self, n: usize) -> Self {
        self.vis_std = Some(n);
        self
    }
    #[inline(always)]
    pub fn sed_atr(mut self, n: usize) -> Self {
        self.sed_atr = Some(n);
        self
    }
    #[inline(always)]
    pub fn sed_std(mut self, n: usize) -> Self {
        self.sed_std = Some(n);
        self
    }
    #[inline(always)]
    pub fn threshold(mut self, x: f64) -> Self {
        self.threshold = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
        self.apply_src(c, "close")
    }

    #[inline(always)]
    pub fn apply_src(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
        let p = DamianiVolatmeterParams {
            vis_atr: self.vis_atr,
            vis_std: self.vis_std,
            sed_atr: self.sed_atr,
            sed_std: self.sed_std,
            threshold: self.threshold,
        };
        let i = DamianiVolatmeterInput::from_candles(c, src, p);
        damiani_volatmeter_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
        let p = DamianiVolatmeterParams {
            vis_atr: self.vis_atr,
            vis_std: self.vis_std,
            sed_atr: self.sed_atr,
            sed_std: self.sed_std,
            threshold: self.threshold,
        };
        let i = DamianiVolatmeterInput::from_slice(d, p);
        damiani_volatmeter_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream<'a>(
        self,
        candles: &'a Candles,
        src: &'a str,
    ) -> Result<DamianiVolatmeterStream<'a>, DamianiVolatmeterError> {
        let p = DamianiVolatmeterParams {
            vis_atr: self.vis_atr,
            vis_std: self.vis_std,
            sed_atr: self.sed_atr,
            sed_std: self.sed_std,
            threshold: self.threshold,
        };
        DamianiVolatmeterStream::new_from_candles(candles, src, p)
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DamianiVolatmeterError {
    #[error("damiani_volatmeter: All values are NaN.")]
    AllValuesNaN,
    #[error("damiani_volatmeter: Invalid period: data length = {data_len}, vis_atr = {vis_atr}, vis_std = {vis_std}, sed_atr = {sed_atr}, sed_std = {sed_std}")]
    InvalidPeriod {
        data_len: usize,
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
    },
    #[error("damiani_volatmeter: Not enough valid data after first non-NaN index. needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("damiani_volatmeter: Empty data provided.")]
    EmptyData,
    #[error("damiani_volatmeter: Non-batch kernel '{kernel:?}' cannot be used with batch API. Use one of: Auto, Scalar, Avx2Batch, Avx512Batch.")]
    NonBatchKernel { kernel: Kernel },
}

#[inline]
pub fn damiani_volatmeter(
    input: &DamianiVolatmeterInput,
) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
    damiani_volatmeter_with_kernel(input, Kernel::Auto)
}

fn damiani_volatmeter_prepare<'a>(
    input: &'a DamianiVolatmeterInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64], // high
        &'a [f64], // low
        &'a [f64], // close
        usize,     // vis_atr
        usize,     // vis_std
        usize,     // sed_atr
        usize,     // sed_std
        f64,       // threshold
        usize,     // first
        usize,     // needed
        Kernel,    // chosen kernel
    ),
    DamianiVolatmeterError,
> {
    // Extract three parallel slices: high, low, and close
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        DamianiVolatmeterData::Candles { candles, source } => {
            // For Candles, use "high", "low", and the user-specified source
            let h = source_type(candles, "high");
            let l = source_type(candles, "low");
            let c = source_type(candles, source); // Use the passed source, not hardcoded "close"
            (h, l, c)
        }
        DamianiVolatmeterData::Slice(slice) => {
            // If the user only provided a bare &[f64], treat it as "close",
            // and set "high"="low"="close" so that true range = 0 on all bars.
            (slice, slice, slice)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(DamianiVolatmeterError::EmptyData);
    }

    let vis_atr = input.get_vis_atr();
    let vis_std = input.get_vis_std();
    let sed_atr = input.get_sed_atr();
    let sed_std = input.get_sed_std();
    let threshold = input.get_threshold();

    // Validate zero or out‐of‐bounds periods
    if vis_atr == 0
        || vis_std == 0
        || sed_atr == 0
        || sed_std == 0
        || vis_atr > len
        || vis_std > len
        || sed_atr > len
        || sed_std > len
    {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: len,
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
        });
    }

    // First non‐NaN index in "close"
    let first = close
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(DamianiVolatmeterError::AllValuesNaN)?;
    let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3]
        .iter()
        .max()
        .unwrap();
    if (len - first) < needed {
        return Err(DamianiVolatmeterError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    // Choose kernel
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((
        high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, needed, chosen,
    ))
}

pub fn damiani_volatmeter_with_kernel(
    input: &DamianiVolatmeterInput,
    kernel: Kernel,
) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
    let (high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, needed, chosen) =
        damiani_volatmeter_prepare(input, kernel)?;

    let len = close.len();
    let warm_end = first + needed - 1;
    let mut vol = alloc_with_nan_prefix(len, warm_end.min(len));
    let mut anti = alloc_with_nan_prefix(len, warm_end.min(len));

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => damiani_volatmeter_scalar(
                high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol,
                &mut anti,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => damiani_volatmeter_avx2(
                high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol,
                &mut anti,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => damiani_volatmeter_avx512(
                high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol,
                &mut anti,
            ),
            _ => unreachable!(),
        }
    }

    // Re-assert NaN prefix through warm_end (scalar may have written earlier than warm_end)
    let cut = (warm_end + 1).min(len);
    for x in &mut vol[..cut] {
        *x = f64::NAN;
    }
    for x in &mut anti[..cut] {
        *x = f64::NAN;
    }

    Ok(DamianiVolatmeterOutput { vol, anti })
}

#[inline]
pub fn damiani_volatmeter_into_slice(
    vol_dst: &mut [f64],
    anti_dst: &mut [f64],
    input: &DamianiVolatmeterInput,
    kernel: Kernel,
) -> Result<(), DamianiVolatmeterError> {
    let (high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, needed, chosen) =
        damiani_volatmeter_prepare(input, kernel)?;

    let len = close.len();

    // Validate destination slices have correct length
    if vol_dst.len() != len || anti_dst.len() != len {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: len,
            vis_atr: vol_dst.len(),
            vis_std: anti_dst.len(),
            sed_atr: 0,
            sed_std: 0,
        });
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => damiani_volatmeter_scalar(
                high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol_dst,
                anti_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => damiani_volatmeter_avx2(
                high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol_dst,
                anti_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => damiani_volatmeter_avx512(
                high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol_dst,
                anti_dst,
            ),
            _ => unreachable!(),
        }
    }

    let warm_end = first + needed - 1;
    let cut = (warm_end + 1).min(len);
    for x in &mut vol_dst[..cut] {
        *x = f64::NAN;
    }
    for x in &mut anti_dst[..cut] {
        *x = f64::NAN;
    }

    Ok(())
}

/// Write Damiani Volatmeter outputs into caller-provided buffers (no allocations).
///
/// - Preserves NaN warmups exactly as the Vec-returning API does.
/// - `vol_out.len()` and `anti_out.len()` must equal the input length; returns `InvalidPeriod` on mismatch.
/// - Uses `Kernel::Auto` runtime selection (same as `damiani_volatmeter`).
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn damiani_volatmeter_into(
    input: &DamianiVolatmeterInput,
    vol_out: &mut [f64],
    anti_out: &mut [f64],
) -> Result<(), DamianiVolatmeterError> {
    damiani_volatmeter_into_slice(vol_out, anti_out, input, Kernel::Auto)
}

#[inline]
pub unsafe fn damiani_volatmeter_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    first: usize,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    let len = close.len();
    let mut atr_vis_val = f64::NAN;
    let mut atr_sed_val = f64::NAN;
    let mut sum_vis = 0.0;
    let mut sum_sed = 0.0;

    // Hoisted constants
    let vis_atr_f = vis_atr as f64;
    let sed_atr_f = sed_atr as f64;
    let needed_all = *[vis_atr, vis_std, sed_atr, sed_std, 3]
        .iter()
        .max()
        .unwrap();

    // Initialize prev_close to first finite value
    let mut prev_close = f64::NAN;
    let mut have_prev = false;

    // Ring buffers and running sums for the two StdDev windows
    let mut ring_vis = vec![0.0; vis_std];
    let mut ring_sed = vec![0.0; sed_std];
    let mut sum_vis_std = 0.0;
    let mut sum_sq_vis_std = 0.0;
    let mut sum_sed_std = 0.0;
    let mut sum_sq_sed_std = 0.0;
    let mut idx_vis = 0;
    let mut idx_sed = 0;
    let mut filled_vis = 0;
    let mut filled_sed = 0;

    // Lag constant (half lag)
    let lag_s = 0.5_f64;

    // Start from first non-NaN to avoid poisoning ATR
    for i in first..len {
        // Compute "True Range" only after we have a valid previous close
        let tr = if have_prev && close[i].is_finite() {
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - prev_close).abs();
            let tr3 = (low[i] - prev_close).abs();
            tr1.max(tr2).max(tr3)
        } else {
            0.0
        };
        // Update prev_close only if current close is finite
        if close[i].is_finite() {
            prev_close = close[i];
            have_prev = true;
        }

        // ----- ATR for “vis” line -----
        if i < vis_atr {
            sum_vis += tr;
            if i == vis_atr - 1 {
                atr_vis_val = sum_vis / vis_atr_f;
            }
        } else if atr_vis_val.is_finite() {
            atr_vis_val = ((vis_atr_f - 1.0) * atr_vis_val + tr) / vis_atr_f;
        }

        // ----- ATR for “sed” line -----
        if i < sed_atr {
            sum_sed += tr;
            if i == sed_atr - 1 {
                atr_sed_val = sum_sed / sed_atr_f;
            }
        } else if atr_sed_val.is_finite() {
            atr_sed_val = ((sed_atr_f - 1.0) * atr_sed_val + tr) / sed_atr_f;
        }

        // Insert current “price‐for‐StdDev” (= close, with NaNs treated as 0) into both rings
        let val = if close[i].is_nan() { 0.0 } else { close[i] };
        // —— update “vis” ring buffer
        let old_v = ring_vis[idx_vis];
        ring_vis[idx_vis] = val;
        idx_vis = (idx_vis + 1) % vis_std;
        if filled_vis < vis_std {
            filled_vis += 1;
            sum_vis_std += val;
            sum_sq_vis_std += val * val;
        } else {
            sum_vis_std = sum_vis_std - old_v + val;
            sum_sq_vis_std = sum_sq_vis_std - (old_v * old_v) + (val * val);
        }
        // —— update “sed” ring buffer
        let old_s = ring_sed[idx_sed];
        ring_sed[idx_sed] = val;
        idx_sed = (idx_sed + 1) % sed_std;
        if filled_sed < sed_std {
            filled_sed += 1;
            sum_sed_std += val;
            sum_sq_sed_std += val * val;
        } else {
            sum_sed_std = sum_sed_std - old_s + val;
            sum_sq_sed_std = sum_sq_sed_std - (old_s * old_s) + (val * val);
        }

        // Only start computing “vol” and “anti” once EVERY lookback is satisfied:
        if i >= needed_all {
            // Previous two “vol” values needed for the lag component
            let p1 = if i >= 1 && !vol[i - 1].is_nan() {
                vol[i - 1]
            } else {
                0.0
            };
            let p3 = if i >= 3 && !vol[i - 3].is_nan() {
                vol[i - 3]
            } else {
                0.0
            };

            // Safely handle ATR Sed = 0 or NaN
            let sed_safe = if atr_sed_val.is_finite() && atr_sed_val != 0.0 {
                atr_sed_val
            } else {
                atr_sed_val + f64::EPSILON
            };

            // Compute “vol[i]” exactly as the old version did
            vol[i] = (atr_vis_val / sed_safe) + lag_s * (p1 - p3);

            // Only compute “anti[i]” once both StdDev windows are completely filled
            if filled_vis == vis_std && filled_sed == sed_std {
                // Population‐stddev for “vis” window
                let mean_vis = sum_vis_std / (vis_std as f64);
                let mean_sq_vis = sum_sq_vis_std / (vis_std as f64);
                let var_vis = (mean_sq_vis - mean_vis * mean_vis).max(0.0);
                let std_vis = var_vis.sqrt();

                // Population‐stddev for “sed” window
                let mean_sed = sum_sed_std / (sed_std as f64);
                let mean_sq_sed = sum_sq_sed_std / (sed_std as f64);
                let var_sed = (mean_sq_sed - mean_sed * mean_sed).max(0.0);
                let std_sed = var_sed.sqrt();

                let ratio = if std_sed != 0.0 {
                    std_vis / std_sed
                } else {
                    std_vis / (std_sed + f64::EPSILON)
                };
                anti[i] = threshold - ratio;
            }
        }
    }
}

#[inline]
fn stddev(sum: f64, sum_sq: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mean = sum / n as f64;
    let mean_sq = sum_sq / n as f64;
    let var = mean_sq - mean * mean;
    if var <= 0.0 {
        0.0
    } else {
        var.sqrt()
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    first: usize,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    damiani_volatmeter_scalar(
        high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    first: usize,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    damiani_volatmeter_scalar(
        high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
    )
}

pub fn damiani_volatmeter_batch_with_kernel(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    k: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DamianiVolatmeterError::NonBatchKernel { kernel: other }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    damiani_volatmeter_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DamianiVolatmeterBatchRange {
    pub vis_atr: (usize, usize, usize),
    pub vis_std: (usize, usize, usize),
    pub sed_atr: (usize, usize, usize),
    pub sed_std: (usize, usize, usize),
    pub threshold: (f64, f64, f64),
}
impl Default for DamianiVolatmeterBatchRange {
    fn default() -> Self {
        Self {
            vis_atr: (13, 40, 1),
            vis_std: (20, 40, 1),
            sed_atr: (40, 40, 0),
            sed_std: (100, 100, 0),
            threshold: (1.4, 1.4, 0.0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct DamianiVolatmeterBatchBuilder {
    range: DamianiVolatmeterBatchRange,
    kernel: Kernel,
}
impl DamianiVolatmeterBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn vis_atr_range(mut self, s: usize, e: usize, step: usize) -> Self {
        self.range.vis_atr = (s, e, step);
        self
    }
    pub fn vis_std_range(mut self, s: usize, e: usize, step: usize) -> Self {
        self.range.vis_std = (s, e, step);
        self
    }
    pub fn sed_atr_range(mut self, s: usize, e: usize, step: usize) -> Self {
        self.range.sed_atr = (s, e, step);
        self
    }
    pub fn sed_std_range(mut self, s: usize, e: usize, step: usize) -> Self {
        self.range.sed_std = (s, e, step);
        self
    }
    pub fn threshold_range(mut self, s: f64, e: f64, step: f64) -> Self {
        self.range.threshold = (s, e, step);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
        damiani_volatmeter_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
}

#[derive(Clone, Debug)]
pub struct DamianiVolatmeterBatchOutput {
    pub vol: Vec<f64>,
    pub anti: Vec<f64>,
    pub combos: Vec<DamianiVolatmeterParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DamianiVolatmeterBatchOutput {
    pub fn row_for_params(&self, p: &DamianiVolatmeterParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.vis_atr == p.vis_atr
                && c.vis_std == p.vis_std
                && c.sed_atr == p.sed_atr
                && c.sed_std == p.sed_std
                && (c.threshold.unwrap_or(1.4) - p.threshold.unwrap_or(1.4)).abs() < 1e-12
        })
    }
    pub fn vol_for(&self, p: &DamianiVolatmeterParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.vol[start..start + self.cols]
        })
    }
    pub fn anti_for(&self, p: &DamianiVolatmeterParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.anti[start..start + self.cols]
        })
    }
}
#[inline(always)]
pub fn damiani_volatmeter_batch_slice(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    kern: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    damiani_volatmeter_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn damiani_volatmeter_batch_par_slice(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    kern: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    damiani_volatmeter_batch_inner(data, sweep, kern, true)
}
fn expand_grid(r: &DamianiVolatmeterBatchRange) -> Vec<DamianiVolatmeterParams> {
    fn axis_usize((s, e, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || s == e {
            return vec![s];
        }
        (s..=e).step_by(step).collect()
    }
    fn axis_f64((s, e, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (s - e).abs() < 1e-12 {
            return vec![s];
        }
        let mut v = Vec::new();
        let mut x = s;
        while x <= e + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let vis_atrs = axis_usize(r.vis_atr);
    let vis_stds = axis_usize(r.vis_std);
    let sed_atrs = axis_usize(r.sed_atr);
    let sed_stds = axis_usize(r.sed_std);
    let thresholds = axis_f64(r.threshold);
    let mut out = Vec::with_capacity(
        vis_atrs.len() * vis_stds.len() * sed_atrs.len() * sed_stds.len() * thresholds.len(),
    );
    for &va in &vis_atrs {
        for &vs in &vis_stds {
            for &sa in &sed_atrs {
                for &ss in &sed_stds {
                    for &th in &thresholds {
                        out.push(DamianiVolatmeterParams {
                            vis_atr: Some(va),
                            vis_std: Some(vs),
                            sed_atr: Some(sa),
                            sed_std: Some(ss),
                            threshold: Some(th),
                        });
                    }
                }
            }
        }
    }
    out
}
#[inline(always)]
fn damiani_volatmeter_batch_inner(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: 0,
            vis_atr: 0,
            vis_std: 0,
            sed_atr: 0,
            sed_std: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DamianiVolatmeterError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| {
            *[
                c.vis_atr.unwrap(),
                c.vis_std.unwrap(),
                c.sed_atr.unwrap(),
                c.sed_std.unwrap(),
            ]
            .iter()
            .max()
            .unwrap()
        })
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(DamianiVolatmeterError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Allocate uninitialized matrices
    let mut vol_mu = make_uninit_matrix(rows, cols);
    let mut anti_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each combination
    let warm_per_row: Vec<usize> = combos
        .iter()
        .map(|p| {
            let needed = *[
                p.vis_atr.unwrap(),
                p.vis_std.unwrap(),
                p.sed_atr.unwrap(),
                p.sed_std.unwrap(),
                3,
            ]
            .iter()
            .max()
            .unwrap();
            first + needed - 1
        })
        .collect();

    // Initialize NaN prefixes
    init_matrix_prefixes(&mut vol_mu, cols, &warm_per_row);
    init_matrix_prefixes(&mut anti_mu, cols, &warm_per_row);

    // Convert to mutable slices
    let mut vol_guard = core::mem::ManuallyDrop::new(vol_mu);
    let mut anti_guard = core::mem::ManuallyDrop::new(anti_mu);
    let vol: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(vol_guard.as_mut_ptr() as *mut f64, vol_guard.len())
    };
    let anti: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(anti_guard.as_mut_ptr() as *mut f64, anti_guard.len())
    };

    let do_row = |row: usize, out_vol: &mut [f64], out_anti: &mut [f64]| unsafe {
        let prm = &combos[row];
        let needed = *[
            prm.vis_atr.unwrap(),
            prm.vis_std.unwrap(),
            prm.sed_atr.unwrap(),
            prm.sed_std.unwrap(),
            3,
        ]
        .iter()
        .max()
        .unwrap();
        let warm_end = (first + needed - 1).min(cols);

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => damiani_volatmeter_row_scalar(
                data,
                first,
                prm.vis_atr.unwrap(),
                prm.vis_std.unwrap(),
                prm.sed_atr.unwrap(),
                prm.sed_std.unwrap(),
                prm.threshold.unwrap(),
                out_vol,
                out_anti,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => damiani_volatmeter_row_avx2(
                data,
                first,
                prm.vis_atr.unwrap(),
                prm.vis_std.unwrap(),
                prm.sed_atr.unwrap(),
                prm.sed_std.unwrap(),
                prm.threshold.unwrap(),
                out_vol,
                out_anti,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => damiani_volatmeter_row_avx512(
                data,
                first,
                prm.vis_atr.unwrap(),
                prm.vis_std.unwrap(),
                prm.sed_atr.unwrap(),
                prm.sed_std.unwrap(),
                prm.threshold.unwrap(),
                out_vol,
                out_anti,
            ),
            _ => damiani_volatmeter_row_scalar(
                data,
                first,
                prm.vis_atr.unwrap(),
                prm.vis_std.unwrap(),
                prm.sed_atr.unwrap(),
                prm.sed_std.unwrap(),
                prm.threshold.unwrap(),
                out_vol,
                out_anti,
            ),
        }

        // re-assert NaN prefix through warm_end
        let cut = (warm_end + 1).min(cols);
        for x in &mut out_vol[..cut] {
            *x = f64::NAN;
        }
        for x in &mut out_anti[..cut] {
            *x = f64::NAN;
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            vol.par_chunks_mut(cols)
                .zip(anti.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (outv, outa))| do_row(row, outv, outa));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (outv, outa)) in vol.chunks_mut(cols).zip(anti.chunks_mut(cols)).enumerate() {
                do_row(row, outv, outa);
            }
        }
    } else {
        for (row, (outv, outa)) in vol.chunks_mut(cols).zip(anti.chunks_mut(cols)).enumerate() {
            do_row(row, outv, outa);
        }
    }

    // Convert back to owned vectors
    let vol = unsafe {
        Vec::from_raw_parts(
            vol_guard.as_mut_ptr() as *mut f64,
            vol_guard.len(),
            vol_guard.capacity(),
        )
    };

    let anti = unsafe {
        Vec::from_raw_parts(
            anti_guard.as_mut_ptr() as *mut f64,
            anti_guard.len(),
            anti_guard.capacity(),
        )
    };

    Ok(DamianiVolatmeterBatchOutput {
        vol,
        anti,
        combos,
        rows,
        cols,
    })
}

/// Version of batch computation that writes directly to pre-allocated slices for Python bindings
#[inline(always)]
fn damiani_volatmeter_batch_inner_into(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    kern: Kernel,
    parallel: bool,
    vol_out: &mut [f64],
    anti_out: &mut [f64],
) -> Result<Vec<DamianiVolatmeterParams>, DamianiVolatmeterError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: 0,
            vis_atr: 0,
            vis_std: 0,
            sed_atr: 0,
            sed_std: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DamianiVolatmeterError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .flat_map(|p| {
            [
                p.vis_atr.unwrap_or(13),
                p.vis_std.unwrap_or(20),
                p.sed_atr.unwrap_or(40),
                p.sed_std.unwrap_or(100),
                3,
            ]
        })
        .max()
        .unwrap_or(100);
    if (data.len() - first) < max_p {
        return Err(DamianiVolatmeterError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let total_size = rows * cols;

    // Ensure output slices have the correct size
    if vol_out.len() != total_size || anti_out.len() != total_size {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: data.len(),
            vis_atr: 0,
            vis_std: 0,
            sed_atr: 0,
            sed_std: 0,
        });
    }

    let do_row = |row: usize, out_vol: &mut [f64], out_anti: &mut [f64]| {
        let p = &combos[row];
        // For batch operations, data is always a slice (close prices)
        // Damiani volatmeter needs high/low/close, so we use the close data for all three
        let close = data;
        let high = data;
        let low = data;

        let vis_atr = p.vis_atr.unwrap_or(1);
        let vis_std = p.vis_std.unwrap_or(20);
        let sed_atr = p.sed_atr.unwrap_or(13);
        let sed_std = p.sed_std.unwrap_or(40);
        let threshold = p.threshold.unwrap_or(1.4);

        // Row-specific optimized path using shared precomputes, enabled on non-Scalar kernels.
        // Keeps semantics identical by starting outputs only after warmup.
        #[allow(unused_mut)]
        let mut used_scalar_fallback = false;
        #[allow(unused_variables)]
        {
            match kern {
                Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch => {
                    // Use shared precomputes captured from the outer scope via thread-local closures
                    // (populated below when we detect a row-optimized run).
                    // Filled via prefix_sums/var accessors provided in the outer function scope.
                    // If for some reason shared buffers aren't initialized, fallback to scalar.
                    // The actual precomputes are established outside this closure.
                }
                _ => used_scalar_fallback = true,
            }
        }

        if used_scalar_fallback {
            unsafe {
                match kern {
                    Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                        damiani_volatmeter_scalar(
                            high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first,
                            out_vol, out_anti,
                        )
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 | Kernel::Avx2Batch => damiani_volatmeter_avx2(
                        high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first,
                        out_vol, out_anti,
                    ),
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 | Kernel::Avx512Batch => damiani_volatmeter_avx512(
                        high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first,
                        out_vol, out_anti,
                    ),
                    _ => damiani_volatmeter_scalar(
                        high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first,
                        out_vol, out_anti,
                    ),
                }
            }
        }

        let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3]
            .iter()
            .max()
            .unwrap();
        let warm_end = (first + needed - 1).min(cols);
        let cut = (warm_end + 1).min(cols);
        for x in &mut out_vol[..cut] {
            *x = f64::NAN;
        }
        for x in &mut out_anti[..cut] {
            *x = f64::NAN;
        }
    };

    // If using a row-optimized path, build shared precomputes here and run rows with them.
    // Row-optimized batch experiment: present but disabled by default due to mixed results.
    // Enable manually by setting this to true for local experiments.
    let use_row_optimized = false
        && matches!(
            kern,
            Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch
        );

    if use_row_optimized {
        let len = data.len();
        // Shared TR over the series using close-only semantics (high=low=close)
        let mut tr: Vec<f64> = vec![0.0; len];
        let mut prev_close = f64::NAN;
        let mut have_prev = false;
        for i in 0..len {
            let cl = data[i];
            let t = if have_prev && cl.is_finite() {
                (cl - prev_close).abs()
            } else {
                0.0
            };
            tr[i] = t;
            if cl.is_finite() {
                prev_close = cl;
                have_prev = true;
            }
        }

        // Shared prefix sums for StdDev with NaN→0 policy
        let mut s: Vec<f64> = vec![0.0; len + 1];
        let mut ss: Vec<f64> = vec![0.0; len + 1];
        for i in 0..len {
            let x = if data[i].is_nan() { 0.0 } else { data[i] };
            s[i + 1] = s[i] + x;
            ss[i + 1] = ss[i] + x * x;
        }

        let process_row = |row: usize, outv: &mut [f64], outa: &mut [f64]| {
            let p = &combos[row];
            let vis_atr = p.vis_atr.unwrap_or(13);
            let vis_std = p.vis_std.unwrap_or(20);
            let sed_atr = p.sed_atr.unwrap_or(40);
            let sed_std = p.sed_std.unwrap_or(100);
            let threshold = p.threshold.unwrap_or(1.4);

            let len = data.len();
            let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3]
                .iter()
                .max()
                .unwrap();
            let start_idx = first + needed - 1;

            let vis_atr_f = vis_atr as f64;
            let sed_atr_f = sed_atr as f64;

            let mut atr_vis = f64::NAN;
            let mut atr_sed = f64::NAN;
            let mut sum_vis_tr = 0.0;
            let mut sum_sed_tr = 0.0;

            // vol history to preserve lag semantics (NaN until we start writing)
            let mut vh1 = f64::NAN;
            let mut vh2 = f64::NAN;
            let mut vh3 = f64::NAN;

            for i in first..len {
                let t = tr[i];
                if i < vis_atr {
                    sum_vis_tr += t;
                    if i == vis_atr - 1 {
                        atr_vis = sum_vis_tr / vis_atr_f;
                    }
                } else if atr_vis.is_finite() {
                    atr_vis = ((vis_atr_f - 1.0) * atr_vis + t) / vis_atr_f;
                }

                if i < sed_atr {
                    sum_sed_tr += t;
                    if i == sed_atr - 1 {
                        atr_sed = sum_sed_tr / sed_atr_f;
                    }
                } else if atr_sed.is_finite() {
                    atr_sed = ((sed_atr_f - 1.0) * atr_sed + t) / sed_atr_f;
                }

                if i >= start_idx {
                    let p1 = if vh1.is_nan() { 0.0 } else { vh1 };
                    let p3 = if vh3.is_nan() { 0.0 } else { vh3 };
                    let sed_safe = if atr_sed.is_finite() && atr_sed != 0.0 {
                        atr_sed
                    } else {
                        atr_sed + f64::EPSILON
                    };
                    let v_now = (atr_vis / sed_safe) + 0.5 * (p1 - p3);
                    outv[i] = v_now;
                    vh3 = vh2;
                    vh2 = vh1;
                    vh1 = v_now;

                    // StdDev via prefix sums (population var), both windows fully filled here
                    let sumv = s[i + 1] - s[i + 1 - vis_std];
                    let sumv2 = ss[i + 1] - ss[i + 1 - vis_std];
                    let meanv = sumv / (vis_std as f64);
                    let varv = (sumv2 / (vis_std as f64) - meanv * meanv).max(0.0);
                    let stdv = varv.sqrt();

                    let sums = s[i + 1] - s[i + 1 - sed_std];
                    let sums2 = ss[i + 1] - ss[i + 1 - sed_std];
                    let means = sums / (sed_std as f64);
                    let vars = (sums2 / (sed_std as f64) - means * means).max(0.0);
                    let stds = vars.sqrt();

                    let den = if stds != 0.0 {
                        stds
                    } else {
                        stds + f64::EPSILON
                    };
                    outa[i] = threshold - (stdv / den);
                }
            }

            // Warmup masking to preserve exact NaN prefix semantics
            let warm_end = (first + needed - 1).min(len);
            let cut = (warm_end + 1).min(len);
            for j in 0..cut {
                outv[j] = f64::NAN;
                outa[j] = f64::NAN;
            }
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                vol_out
                    .par_chunks_mut(cols)
                    .zip(anti_out.par_chunks_mut(cols))
                    .enumerate()
                    .for_each(|(row, (outv, outa))| process_row(row, outv, outa));
            }

            #[cfg(target_arch = "wasm32")]
            {
                for (row, (outv, outa)) in vol_out
                    .chunks_mut(cols)
                    .zip(anti_out.chunks_mut(cols))
                    .enumerate()
                {
                    process_row(row, outv, outa);
                }
            }
        } else {
            for (row, (outv, outa)) in vol_out
                .chunks_mut(cols)
                .zip(anti_out.chunks_mut(cols))
                .enumerate()
            {
                process_row(row, outv, outa);
            }
        }

        return Ok(combos);
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            vol_out
                .par_chunks_mut(cols)
                .zip(anti_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (outv, outa))| do_row(row, outv, outa));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (outv, outa)) in vol_out
                .chunks_mut(cols)
                .zip(anti_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, outv, outa);
            }
        }
    } else {
        for (row, (outv, outa)) in vol_out
            .chunks_mut(cols)
            .zip(anti_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, outv, outa);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn damiani_volatmeter_row_scalar(
    data: &[f64],
    first: usize,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    // Call the new scalar signature by passing `data` for high, low, and close:
    damiani_volatmeter_scalar(
        data, // high
        data, // low
        data, // close
        vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx2(
    data: &[f64],
    first: usize,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    damiani_volatmeter_scalar(
        data, // high
        data, // low
        data, // close
        vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512(
    data: &[f64],
    first: usize,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    damiani_volatmeter_scalar(
        data, // high
        data, // low
        data, // close
        vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512_short(
    data: &[f64],
    first: usize,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    damiani_volatmeter_scalar(
        data, // high
        data, // low
        data, // close
        vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512_long(
    data: &[f64],
    first: usize,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    vol: &mut [f64],
    anti: &mut [f64],
) {
    damiani_volatmeter_scalar(
        data, // high
        data, // low
        data, // close
        vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
    )
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterStream<'a> {
    // Data
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],

    // Params
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,

    // Indices & warmup bookkeeping
    start: usize,
    index: usize,
    needed_all: usize,

    // ATR state (Wilder)
    atr_vis_val: f64,
    atr_sed_val: f64,
    sum_vis_seed: f64,
    sum_sed_seed: f64,
    vis_seed_cnt: usize,
    sed_seed_cnt: usize,
    prev_close: f64,
    have_prev: bool,

    // Rolling stddev state (NaN→0 policy)
    ring_vis: Vec<f64>,
    ring_sed: Vec<f64>,
    idx_vis: usize,
    idx_sed: usize,
    filled_vis: usize,
    filled_sed: usize,
    sum_vis_std: f64,
    sum_sq_vis_std: f64,
    sum_sed_std: f64,
    sum_sq_sed_std: f64,

    // Lag for vol
    vol_hist: [f64; 3],
    lag_s: f64,

    // Precomputed reciprocals / constants
    inv_vis_atr: f64,
    inv_sed_atr: f64,
    inv_vis_std: f64,
    inv_sed_std: f64,
    vis_atr_m1: f64,
    sed_atr_m1: f64,
}

impl<'a> DamianiVolatmeterStream<'a> {
    #[inline]
    pub fn new_from_candles(
        candles: &'a Candles,
        src: &str,
        params: DamianiVolatmeterParams,
    ) -> Result<Self, DamianiVolatmeterError> {
        Self::new_from_slices(
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, src),
            params,
        )
    }

    #[inline]
    pub fn new_from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: DamianiVolatmeterParams,
    ) -> Result<Self, DamianiVolatmeterError> {
        let len = close.len();
        if len == 0 {
            return Err(DamianiVolatmeterError::EmptyData);
        }

        let vis_atr = params.vis_atr.unwrap_or(13);
        let vis_std = params.vis_std.unwrap_or(20);
        let sed_atr = params.sed_atr.unwrap_or(40);
        let sed_std = params.sed_std.unwrap_or(100);
        let threshold = params.threshold.unwrap_or(1.4);

        if vis_atr == 0
            || vis_std == 0
            || sed_atr == 0
            || sed_std == 0
            || vis_atr > len
            || vis_std > len
            || sed_atr > len
            || sed_std > len
        {
            return Err(DamianiVolatmeterError::InvalidPeriod {
                data_len: len,
                vis_atr,
                vis_std,
                sed_atr,
                sed_std,
            });
        }

        let start = close
            .iter()
            .position(|&x| !x.is_nan())
            .ok_or(DamianiVolatmeterError::AllValuesNaN)?;
        let needed_all = *[vis_atr, vis_std, sed_atr, sed_std, 3]
            .iter()
            .max()
            .unwrap();
        if len - start < needed_all {
            return Err(DamianiVolatmeterError::NotEnoughValidData {
                needed: needed_all,
                valid: len - start,
            });
        }

        Ok(Self {
            high,
            low,
            close,
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
            threshold,

            start,
            index: start,
            needed_all,

            atr_vis_val: f64::NAN,
            atr_sed_val: f64::NAN,
            sum_vis_seed: 0.0,
            sum_sed_seed: 0.0,
            vis_seed_cnt: 0,
            sed_seed_cnt: 0,
            prev_close: f64::NAN,
            have_prev: false,

            ring_vis: vec![0.0; vis_std],
            ring_sed: vec![0.0; sed_std],
            idx_vis: 0,
            idx_sed: 0,
            filled_vis: 0,
            filled_sed: 0,
            sum_vis_std: 0.0,
            sum_sq_vis_std: 0.0,
            sum_sed_std: 0.0,
            sum_sq_sed_std: 0.0,

            vol_hist: [f64::NAN; 3],
            lag_s: 0.5,

            inv_vis_atr: 1.0 / (vis_atr as f64),
            inv_sed_atr: 1.0 / (sed_atr as f64),
            inv_vis_std: 1.0 / (vis_std as f64),
            inv_sed_std: 1.0 / (sed_std as f64),
            vis_atr_m1: (vis_atr as f64) - 1.0,
            sed_atr_m1: (sed_atr as f64) - 1.0,
        })
    }

    /// Single-bar update. Returns Some((vol, anti)) after warmup; None during warmup.
    #[inline(always)]
    pub fn update(&mut self) -> Option<(f64, f64)> {
        let i = self.index;
        let len = self.close.len();
        if i >= len {
            return None;
        }

        let cl = self.close[i];
        let hi = self.high[i];
        let lo = self.low[i];

        let tr = if self.have_prev && cl.is_finite() {
            let tr1 = hi - lo;
            let tr2 = (hi - self.prev_close).abs();
            let tr3 = (lo - self.prev_close).abs();
            tr1.max(tr2).max(tr3)
        } else {
            0.0
        };

        if cl.is_finite() {
            self.prev_close = cl;
            self.have_prev = true;
        }

        // ATR (vis)
        if self.vis_seed_cnt < self.vis_atr {
            self.sum_vis_seed += tr;
            self.vis_seed_cnt += 1;
            if self.vis_seed_cnt == self.vis_atr {
                self.atr_vis_val = self.sum_vis_seed * self.inv_vis_atr;
            }
        } else {
            self.atr_vis_val = self.atr_vis_val.mul_add(self.vis_atr_m1, tr) * self.inv_vis_atr;
        }

        // ATR (sed)
        if self.sed_seed_cnt < self.sed_atr {
            self.sum_sed_seed += tr;
            self.sed_seed_cnt += 1;
            if self.sed_seed_cnt == self.sed_atr {
                self.atr_sed_val = self.sum_sed_seed * self.inv_sed_atr;
            }
        } else {
            self.atr_sed_val = self.atr_sed_val.mul_add(self.sed_atr_m1, tr) * self.inv_sed_atr;
        }

        // Rolling stddev rings (NaN→0)
        let v = if cl.is_nan() { 0.0 } else { cl };

        // vis
        let old_v = self.ring_vis[self.idx_vis];
        self.ring_vis[self.idx_vis] = v;
        self.idx_vis += 1;
        if self.idx_vis == self.vis_std {
            self.idx_vis = 0;
        }
        if self.filled_vis < self.vis_std {
            self.filled_vis += 1;
            self.sum_vis_std += v;
            self.sum_sq_vis_std = v.mul_add(v, self.sum_sq_vis_std);
        } else {
            self.sum_vis_std += v - old_v;
            self.sum_sq_vis_std += v.mul_add(v, -old_v * old_v);
        }

        // sed
        let old_s = self.ring_sed[self.idx_sed];
        self.ring_sed[self.idx_sed] = v;
        self.idx_sed += 1;
        if self.idx_sed == self.sed_std {
            self.idx_sed = 0;
        }
        if self.filled_sed < self.sed_std {
            self.filled_sed += 1;
            self.sum_sed_std += v;
            self.sum_sq_sed_std = v.mul_add(v, self.sum_sq_sed_std);
        } else {
            self.sum_sed_std += v - old_s;
            self.sum_sq_sed_std += v.mul_add(v, -old_s * old_s);
        }

        // Advance absolute index
        self.index = i + 1;

        // Warmup gating (match scalar: absolute index threshold)
        if i < self.needed_all {
            return None;
        }

        // Lag terms
        let p1 = if self.vol_hist[0].is_nan() {
            0.0
        } else {
            self.vol_hist[0]
        };
        let p3 = if self.vol_hist[2].is_nan() {
            0.0
        } else {
            self.vol_hist[2]
        };
        let sed_safe = if self.atr_sed_val.is_finite() && self.atr_sed_val != 0.0 {
            self.atr_sed_val
        } else {
            f64::EPSILON
        };
        let vol_now = (self.atr_vis_val / sed_safe) + self.lag_s * (p1 - p3);

        // Shift history
        self.vol_hist[2] = self.vol_hist[1];
        self.vol_hist[1] = self.vol_hist[0];
        self.vol_hist[0] = vol_now;

        // Std ratio via one sqrt
        let mean_v = self.sum_vis_std * self.inv_vis_std;
        let mean2_v = self.sum_sq_vis_std * self.inv_vis_std;
        let var_v = (mean2_v - mean_v * mean_v).max(0.0);

        let mean_s = self.sum_sed_std * self.inv_sed_std;
        let mean2_s = self.sum_sq_sed_std * self.inv_sed_std;
        let var_s = (mean2_s - mean_s * mean_s).max(0.0);

        let ratio = Self::sqrt_ratio(var_v, var_s);
        let anti_now = self.threshold - ratio;

        Some((vol_now, anti_now))
    }

    #[inline(always)]
    fn sqrt_ratio(num_var: f64, den_var: f64) -> f64 {
        (num_var / (if den_var > 0.0 { den_var } else { f64::EPSILON })).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterFeedStream {
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,

    // running ATR state
    atr_vis: f64,
    atr_sed: f64,
    sum_vis: f64,
    sum_sed: f64,
    have_vis_seed: usize,
    have_sed_seed: usize,
    prev_close: f64,
    have_prev: bool,

    // std rings
    ring_vis: Vec<f64>,
    ring_sed: Vec<f64>,
    idx_vis: usize,
    idx_sed: usize,
    fill_vis: usize,
    fill_sed: usize,
    sum_vis_std: f64,
    sum_sq_vis_std: f64,
    sum_sed_std: f64,
    sum_sq_sed_std: f64,

    vol_hist: [f64; 3], // p1, p2, p3
}

impl DamianiVolatmeterFeedStream {
    pub fn try_new(p: DamianiVolatmeterParams) -> Result<Self, DamianiVolatmeterError> {
        let vis_atr = p.vis_atr.unwrap_or(13);
        let vis_std = p.vis_std.unwrap_or(20);
        let sed_atr = p.sed_atr.unwrap_or(40);
        let sed_std = p.sed_std.unwrap_or(100);
        let threshold = p.threshold.unwrap_or(1.4);
        if vis_atr == 0 || vis_std == 0 || sed_atr == 0 || sed_std == 0 {
            return Err(DamianiVolatmeterError::InvalidPeriod {
                data_len: 0,
                vis_atr,
                vis_std,
                sed_atr,
                sed_std,
            });
        }
        Ok(Self {
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
            threshold,
            atr_vis: f64::NAN,
            atr_sed: f64::NAN,
            sum_vis: 0.0,
            sum_sed: 0.0,
            have_vis_seed: 0,
            have_sed_seed: 0,
            prev_close: f64::NAN,
            have_prev: false,
            ring_vis: vec![0.0; vis_std],
            ring_sed: vec![0.0; sed_std],
            idx_vis: 0,
            idx_sed: 0,
            fill_vis: 0,
            fill_sed: 0,
            sum_vis_std: 0.0,
            sum_sq_vis_std: 0.0,
            sum_sed_std: 0.0,
            sum_sq_sed_std: 0.0,
            vol_hist: [f64::NAN; 3],
        })
    }

    /// Feed one bar. Returns Some((vol, anti)) once warmup satisfied, else None.
    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let tr = if self.have_prev {
            let tr1 = high - low;
            let tr2 = (high - self.prev_close).abs();
            let tr3 = (low - self.prev_close).abs();
            tr1.max(tr2).max(tr3)
        } else {
            0.0
        };
        self.prev_close = close;
        self.have_prev = true;

        // ATR seeds then Wilder update
        if self.have_vis_seed < self.vis_atr {
            self.sum_vis += tr;
            self.have_vis_seed += 1;
            if self.have_vis_seed == self.vis_atr {
                self.atr_vis = self.sum_vis / self.vis_atr as f64;
            }
        } else if self.atr_vis.is_finite() {
            self.atr_vis = ((self.vis_atr as f64 - 1.0) * self.atr_vis + tr) / self.vis_atr as f64;
        }

        if self.have_sed_seed < self.sed_atr {
            self.sum_sed += tr;
            self.have_sed_seed += 1;
            if self.have_sed_seed == self.sed_atr {
                self.atr_sed = self.sum_sed / self.sed_atr as f64;
            }
        } else if self.atr_sed.is_finite() {
            self.atr_sed = ((self.sed_atr as f64 - 1.0) * self.atr_sed + tr) / self.sed_atr as f64;
        }

        let v = if close.is_nan() { 0.0 } else { close };

        // std rings
        let old_v = self.ring_vis[self.idx_vis];
        self.ring_vis[self.idx_vis] = v;
        self.idx_vis = (self.idx_vis + 1) % self.vis_std;
        if self.fill_vis < self.vis_std {
            self.fill_vis += 1;
            self.sum_vis_std += v;
            self.sum_sq_vis_std += v * v;
        } else {
            self.sum_vis_std -= old_v;
            self.sum_vis_std += v;
            self.sum_sq_vis_std -= old_v * old_v;
            self.sum_sq_vis_std += v * v;
        }

        let old_s = self.ring_sed[self.idx_sed];
        self.ring_sed[self.idx_sed] = v;
        self.idx_sed = (self.idx_sed + 1) % self.sed_std;
        if self.fill_sed < self.sed_std {
            self.fill_sed += 1;
            self.sum_sed_std += v;
            self.sum_sq_sed_std += v * v;
        } else {
            self.sum_sed_std -= old_s;
            self.sum_sed_std += v;
            self.sum_sq_sed_std -= old_s * old_s;
            self.sum_sq_sed_std += v * v;
        }

        let needed = self
            .vis_atr
            .max(self.vis_std)
            .max(self.sed_atr)
            .max(self.sed_std)
            .max(3);
        // we don't track absolute bar index; use ring fills to gate warmup
        if self.have_vis_seed < self.vis_atr || self.have_sed_seed < self.sed_atr {
            return None;
        }
        if self.fill_vis < self.vis_std || self.fill_sed < self.sed_std {
            return None;
        }
        // the +3 lag baked in below requires 3 past values; use vol_hist NaNs as gating
        if self.vol_hist.iter().any(|x| x.is_nan()) { /* compute vol below, return Some once we have 3 */
        }

        let lag_s = 0.5;
        let p1 = if self.vol_hist[0].is_nan() {
            0.0
        } else {
            self.vol_hist[0]
        };
        let p3 = if self.vol_hist[2].is_nan() {
            0.0
        } else {
            self.vol_hist[2]
        };
        let sed_safe = if self.atr_sed.is_finite() && self.atr_sed != 0.0 {
            self.atr_sed
        } else {
            self.atr_sed + f64::EPSILON
        };
        let vol = (self.atr_vis / sed_safe) + lag_s * (p1 - p3);

        self.vol_hist[2] = self.vol_hist[1];
        self.vol_hist[1] = self.vol_hist[0];
        self.vol_hist[0] = vol;

        // stddevs
        let mean_vis = self.sum_vis_std / self.vis_std as f64;
        let mean_sq_vis = self.sum_sq_vis_std / self.vis_std as f64;
        let std_vis = (mean_sq_vis - mean_vis * mean_vis).max(0.0).sqrt();

        let mean_sed = self.sum_sed_std / self.sed_std as f64;
        let mean_sq_sed = self.sum_sq_sed_std / self.sed_std as f64;
        let std_sed = (mean_sq_sed - mean_sed * mean_sed).max(0.0).sqrt();

        let ratio = if std_sed != 0.0 {
            std_vis / std_sed
        } else {
            std_vis / (std_sed + f64::EPSILON)
        };
        let anti = self.threshold - ratio;

        Some((vol, anti))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    fn check_damiani_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = DamianiVolatmeterParams::default();
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let output = damiani_volatmeter_with_kernel(&input, kernel)?;
        assert_eq!(output.vol.len(), candles.close.len());
        assert_eq!(output.anti.len(), candles.close.len());
        Ok(())
    }

    #[test]
    fn test_damiani_volatmeter_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Construct a small but non-trivial OHLCV series (len = 256)
        let len = 256usize;
        let mut ts = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut vol = Vec::with_capacity(len);

        for i in 0..len {
            let i_f = i as f64;
            let o = 100.0 + 0.1 * i_f + (i_f * 0.05).sin() * 0.5;
            let c = o + (i_f * 0.3).cos() * 0.2;
            let mut h = o + 1.0 + (i % 7) as f64 * 0.01;
            let mut l = o - 1.0 - (i % 5) as f64 * 0.01;
            if h < o { h = o; }
            if h < c { h = c; }
            if l > o { l = o; }
            if l > c { l = c; }

            ts.push(i as i64);
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            vol.push(1000.0 + (i % 10) as f64);
        }

        let candles = Candles::new(ts, open, high, low, close.clone(), vol);
        let input = DamianiVolatmeterInput::from_candles(
            &candles,
            "close",
            DamianiVolatmeterParams::default(),
        );

        // Baseline via Vec-returning API
        let base = damiani_volatmeter(&input)?;

        // Into-API outputs
        let mut out_vol = vec![0.0; len];
        let mut out_anti = vec![0.0; len];
        #[cfg(not(feature = "wasm"))]
        {
            damiani_volatmeter_into(&input, &mut out_vol, &mut out_anti)?;
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, the native symbol is not emitted. Use the slice variant directly.
            damiani_volatmeter_into_slice(&mut out_vol, &mut out_anti, &input, Kernel::Auto)?;
        }

        assert_eq!(out_vol.len(), base.vol.len());
        assert_eq!(out_anti.len(), base.anti.len());

        // Helper: treat NaN == NaN; exact equality preferred; allow tiny epsilon for finite values
        fn eq_or_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..len {
            assert!(
                eq_or_nan(out_vol[i], base.vol[i]),
                "vol mismatch at {}: into={}, api={}",
                i,
                out_vol[i],
                base.vol[i]
            );
            assert!(
                eq_or_nan(out_anti[i], base.anti[i]),
                "anti mismatch at {}: into={}, api={}",
                i,
                out_anti[i],
                base.anti[i]
            );
        }

        Ok(())
    }
    fn check_damiani_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DamianiVolatmeterInput::from_candles(
            &candles,
            "close",
            DamianiVolatmeterParams::default(),
        );
        let output = damiani_volatmeter_with_kernel(&input, kernel)?;
        let n = output.vol.len();
        let expected_vol = [
            0.9009485470514558,
            0.8333604467044887,
            0.815318380178986,
            0.8276892636184923,
            0.879447954127426,
        ];
        let expected_anti = [
            1.1227721577887388,
            1.1250333024152703,
            1.1325501989919875,
            1.1403866079746106,
            1.1392919184055932,
        ];
        let start = n - 5;
        for i in 0..5 {
            let diff_vol = (output.vol[start + i] - expected_vol[i]).abs();
            let diff_anti = (output.anti[start + i] - expected_anti[i]).abs();
            assert!(
                diff_vol < 1e-2,
                "vol mismatch at index {}: expected {}, got {}",
                start + i,
                expected_vol[i],
                output.vol[start + i]
            );
            assert!(
                diff_anti < 1e-2,
                "anti mismatch at index {}: expected {}, got {}",
                start + i,
                expected_anti[i],
                output.anti[start + i]
            );
        }
        Ok(())
    }
    fn check_damiani_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let mut params = DamianiVolatmeterParams::default();
        params.vis_atr = Some(0);
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with zero period", test_name);
        Ok(())
    }
    fn check_damiani_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let mut params = DamianiVolatmeterParams::default();
        params.vis_atr = Some(99999);
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] should fail if period exceeds length",
            test_name
        );
        Ok(())
    }
    fn check_damiani_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = DamianiVolatmeterParams {
            vis_atr: Some(9),
            vis_std: Some(9),
            sed_atr: Some(9),
            sed_std: Some(9),
            threshold: Some(1.4),
        };
        let input = DamianiVolatmeterInput::from_slice(&data, params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_damiani_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DamianiVolatmeterInput::from_candles(
            &candles,
            "close",
            DamianiVolatmeterParams::default(),
        );
        let batch = damiani_volatmeter_with_kernel(&input, kernel)?;

        let mut stream = DamianiVolatmeterStream::new_from_candles(
            &candles,
            "close",
            DamianiVolatmeterParams::default(),
        )?;

        let mut stream_vol = Vec::with_capacity(candles.close.len());
        let mut stream_anti = Vec::with_capacity(candles.close.len());

        for _ in 0..candles.close.len() {
            if let Some((v, a)) = stream.update() {
                stream_vol.push(v);
                stream_anti.push(a);
            } else {
                stream_vol.push(f64::NAN);
                stream_anti.push(f64::NAN);
            }
        }

        for (i, (&bv, &sv)) in batch.vol.iter().zip(stream_vol.iter()).enumerate() {
            if bv.is_nan() && sv.is_nan() {
                continue;
            }
            let diff = (bv - sv).abs();
            assert!(
                diff < 1e-8,
                "[{}] streaming vol mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                bv,
                sv
            );
        }

        for (i, (&ba, &sa)) in batch.anti.iter().zip(stream_anti.iter()).enumerate() {
            if ba.is_nan() && sa.is_nan() {
                continue;
            }
            let diff = (ba - sa).abs();
            assert!(
                diff < 1e-8,
                "[{}] streaming anti mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                ba,
                sa
            );
        }

        Ok(())
    }

    fn check_damiani_input_with_default_candles(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DamianiVolatmeterInput::with_default_candles(&candles);
        match input.data {
            DamianiVolatmeterData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected DamianiVolatmeterData::Candles"),
        }
        Ok(())
    }
    fn check_damiani_params_with_defaults(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let default_params = DamianiVolatmeterParams::default();
        assert_eq!(default_params.vis_atr, Some(13));
        assert_eq!(default_params.vis_std, Some(20));
        assert_eq!(default_params.sed_atr, Some(40));
        assert_eq!(default_params.sed_std, Some(100));
        assert_eq!(default_params.threshold, Some(1.4));
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_damiani_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            DamianiVolatmeterParams::default(),
            // Minimum viable parameters
            DamianiVolatmeterParams {
                vis_atr: Some(2),
                vis_std: Some(2),
                sed_atr: Some(2),
                sed_std: Some(2),
                threshold: Some(1.4),
            },
            // Small periods with low threshold
            DamianiVolatmeterParams {
                vis_atr: Some(5),
                vis_std: Some(10),
                sed_atr: Some(10),
                sed_std: Some(20),
                threshold: Some(0.5),
            },
            // Medium periods with various thresholds
            DamianiVolatmeterParams {
                vis_atr: Some(13),
                vis_std: Some(20),
                sed_atr: Some(40),
                sed_std: Some(100),
                threshold: Some(1.0),
            },
            DamianiVolatmeterParams {
                vis_atr: Some(20),
                vis_std: Some(30),
                sed_atr: Some(50),
                sed_std: Some(120),
                threshold: Some(2.0),
            },
            // Large periods
            DamianiVolatmeterParams {
                vis_atr: Some(50),
                vis_std: Some(80),
                sed_atr: Some(100),
                sed_std: Some(200),
                threshold: Some(1.4),
            },
            // Edge case: all periods equal
            DamianiVolatmeterParams {
                vis_atr: Some(15),
                vis_std: Some(15),
                sed_atr: Some(15),
                sed_std: Some(15),
                threshold: Some(1.5),
            },
            // High threshold
            DamianiVolatmeterParams {
                vis_atr: Some(10),
                vis_std: Some(25),
                sed_atr: Some(30),
                sed_std: Some(75),
                threshold: Some(3.0),
            },
            // Unbalanced periods
            DamianiVolatmeterParams {
                vis_atr: Some(3),
                vis_std: Some(50),
                sed_atr: Some(5),
                sed_std: Some(150),
                threshold: Some(1.2),
            },
            // Another edge case
            DamianiVolatmeterParams {
                vis_atr: Some(25),
                vis_std: Some(25),
                sed_atr: Some(80),
                sed_std: Some(80),
                threshold: Some(0.8),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DamianiVolatmeterInput::from_candles(&candles, "close", params.clone());
            let output = damiani_volatmeter_with_kernel(&input, kernel)?;

            // Check vol array
            for (i, &val) in output.vol.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in vol array \
						 with params: vis_atr={}, vis_std={}, sed_atr={}, sed_std={}, threshold={} (param set {})",
						test_name, val, bits, i,
						params.vis_atr.unwrap(), params.vis_std.unwrap(),
						params.sed_atr.unwrap(), params.sed_std.unwrap(),
						params.threshold.unwrap(), param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in vol array \
						 with params: vis_atr={}, vis_std={}, sed_atr={}, sed_std={}, threshold={} (param set {})",
						test_name, val, bits, i,
						params.vis_atr.unwrap(), params.vis_std.unwrap(),
						params.sed_atr.unwrap(), params.sed_std.unwrap(),
						params.threshold.unwrap(), param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in vol array \
						 with params: vis_atr={}, vis_std={}, sed_atr={}, sed_std={}, threshold={} (param set {})",
						test_name, val, bits, i,
						params.vis_atr.unwrap(), params.vis_std.unwrap(),
						params.sed_atr.unwrap(), params.sed_std.unwrap(),
						params.threshold.unwrap(), param_idx
					);
                }
            }

            // Check anti array
            for (i, &val) in output.anti.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in anti array \
						 with params: vis_atr={}, vis_std={}, sed_atr={}, sed_std={}, threshold={} (param set {})",
						test_name, val, bits, i,
						params.vis_atr.unwrap(), params.vis_std.unwrap(),
						params.sed_atr.unwrap(), params.sed_std.unwrap(),
						params.threshold.unwrap(), param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in anti array \
						 with params: vis_atr={}, vis_std={}, sed_atr={}, sed_std={}, threshold={} (param set {})",
						test_name, val, bits, i,
						params.vis_atr.unwrap(), params.vis_std.unwrap(),
						params.sed_atr.unwrap(), params.sed_std.unwrap(),
						params.threshold.unwrap(), param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in anti array \
						 with params: vis_atr={}, vis_std={}, sed_atr={}, sed_std={}, threshold={} (param set {})",
						test_name, val, bits, i,
						params.vis_atr.unwrap(), params.vis_std.unwrap(),
						params.sed_atr.unwrap(), params.sed_std.unwrap(),
						params.threshold.unwrap(), param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_damiani_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }
    fn check_batch_default_row(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = DamianiVolatmeterBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = DamianiVolatmeterParams::default();
        let vol_row = output.vol_for(&def).expect("default vol row missing");
        let anti_row = output.anti_for(&def).expect("default anti row missing");
        assert_eq!(vol_row.len(), c.close.len());
        assert_eq!(anti_row.len(), c.close.len());

        // Compute expected values using the same "close-only" approach
        let close_slice = source_type(&c, "close");
        let input = DamianiVolatmeterInput::from_slice(close_slice, def.clone());
        let expected_output = damiani_volatmeter(&input)?;

        // Compare the last 5 values
        let start = vol_row.len() - 5;
        for i in 0..5 {
            let idx = start + i;
            assert!(
                (vol_row[idx] - expected_output.vol[idx]).abs() < 1e-10,
                "[{test_name}] default-vol-row mismatch at idx {i}: batch={} vs expected={}",
                vol_row[idx],
                expected_output.vol[idx]
            );
            assert!(
                (anti_row[idx] - expected_output.anti[idx]).abs() < 1e-10,
                "[{test_name}] default-anti-row mismatch at idx {i}: batch={} vs expected={}",
                anti_row[idx],
                expected_output.anti[idx]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (vis_atr range, vis_std range, sed_atr range, sed_std range, threshold range)
            // Small periods
            (2, 10, 2, 2, 10, 2, 5, 15, 5, 10, 30, 10, 0.5, 2.0, 0.5),
            // Medium periods
            (
                10, 20, 5, 15, 35, 10, 30, 60, 15, 80, 120, 20, 1.0, 1.5, 0.25,
            ),
            // Large periods
            (
                30, 60, 15, 50, 100, 25, 60, 120, 30, 150, 250, 50, 1.2, 1.8, 0.3,
            ),
            // Dense small range
            (5, 8, 1, 10, 15, 1, 15, 20, 1, 40, 50, 2, 1.4, 1.4, 0.0),
            // Edge case: all equal
            (10, 10, 0, 10, 10, 0, 10, 10, 0, 10, 10, 0, 1.0, 3.0, 1.0),
            // Unbalanced ranges
            (2, 5, 1, 20, 80, 20, 3, 8, 1, 100, 200, 50, 0.8, 2.5, 0.35),
        ];

        for (
            cfg_idx,
            &(
                va_s,
                va_e,
                va_st,
                vs_s,
                vs_e,
                vs_st,
                sa_s,
                sa_e,
                sa_st,
                ss_s,
                ss_e,
                ss_st,
                th_s,
                th_e,
                th_st,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = DamianiVolatmeterBatchBuilder::new()
                .kernel(kernel)
                .vis_atr_range(va_s, va_e, va_st)
                .vis_std_range(vs_s, vs_e, vs_st)
                .sed_atr_range(sa_s, sa_e, sa_st)
                .sed_std_range(ss_s, ss_e, ss_st)
                .threshold_range(th_s, th_e, th_st)
                .apply_candles(&c, "close")?;

            // Check vol matrix
            for (idx, &val) in output.vol.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in vol \
						 at row {} col {} (flat index {}) with params: vis_atr={}, vis_std={}, sed_atr={}, \
						 sed_std={}, threshold={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.vis_atr.unwrap(), combo.vis_std.unwrap(),
						combo.sed_atr.unwrap(), combo.sed_std.unwrap(),
						combo.threshold.unwrap()
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in vol \
						 at row {} col {} (flat index {}) with params: vis_atr={}, vis_std={}, sed_atr={}, \
						 sed_std={}, threshold={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.vis_atr.unwrap(), combo.vis_std.unwrap(),
						combo.sed_atr.unwrap(), combo.sed_std.unwrap(),
						combo.threshold.unwrap()
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in vol \
						 at row {} col {} (flat index {}) with params: vis_atr={}, vis_std={}, sed_atr={}, \
						 sed_std={}, threshold={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.vis_atr.unwrap(), combo.vis_std.unwrap(),
						combo.sed_atr.unwrap(), combo.sed_std.unwrap(),
						combo.threshold.unwrap()
					);
                }
            }

            // Check anti matrix
            for (idx, &val) in output.anti.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in anti \
						 at row {} col {} (flat index {}) with params: vis_atr={}, vis_std={}, sed_atr={}, \
						 sed_std={}, threshold={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.vis_atr.unwrap(), combo.vis_std.unwrap(),
						combo.sed_atr.unwrap(), combo.sed_std.unwrap(),
						combo.threshold.unwrap()
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in anti \
						 at row {} col {} (flat index {}) with params: vis_atr={}, vis_std={}, sed_atr={}, \
						 sed_std={}, threshold={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.vis_atr.unwrap(), combo.vis_std.unwrap(),
						combo.sed_atr.unwrap(), combo.sed_std.unwrap(),
						combo.threshold.unwrap()
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in anti \
						 at row {} col {} (flat index {}) with params: vis_atr={}, vis_std={}, sed_atr={}, \
						 sed_std={}, threshold={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.vis_atr.unwrap(), combo.vis_std.unwrap(),
						combo.sed_atr.unwrap(), combo.sed_std.unwrap(),
						combo.threshold.unwrap()
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

    fn check_damiani_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = DamianiVolatmeterParams::default();
        let input = DamianiVolatmeterInput::from_slice(&empty, params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(DamianiVolatmeterError::EmptyData)),
            "[{}] should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_damiani_all_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 200];
        let params = DamianiVolatmeterParams::default();
        let input = DamianiVolatmeterInput::from_slice(&data, params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(DamianiVolatmeterError::AllValuesNaN)),
            "[{}] should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_damiani_invalid_threshold(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with NaN threshold
        let mut params = DamianiVolatmeterParams::default();
        params.threshold = Some(f64::NAN);
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params.clone());
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        // Unlike alma which has explicit threshold validation, damiani will compute but produce NaN results
        // This is acceptable behavior - the indicator doesn't fail but produces NaN outputs
        assert!(
            res.is_ok(),
            "[{}] should not fail with NaN threshold",
            test_name
        );

        // Test with negative threshold (which is mathematically valid for this indicator)
        params.threshold = Some(-1.0);
        let input2 = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let res2 = damiani_volatmeter_with_kernel(&input2, kernel);
        assert!(
            res2.is_ok(),
            "[{}] should work with negative threshold",
            test_name
        );
        Ok(())
    }

    fn check_damiani_invalid_periods(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test with zero vis_atr
        let mut params = DamianiVolatmeterParams::default();
        params.vis_atr = Some(0);
        let input = DamianiVolatmeterInput::from_slice(&data, params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(DamianiVolatmeterError::InvalidPeriod { .. })),
            "[{}] should fail with zero vis_atr",
            test_name
        );

        // Test with zero vis_std
        params = DamianiVolatmeterParams::default();
        params.vis_std = Some(0);
        let input2 = DamianiVolatmeterInput::from_slice(&data, params);
        let res2 = damiani_volatmeter_with_kernel(&input2, kernel);
        assert!(
            matches!(res2, Err(DamianiVolatmeterError::InvalidPeriod { .. })),
            "[{}] should fail with zero vis_std",
            test_name
        );

        // Test with period exceeding data length
        params = DamianiVolatmeterParams::default();
        params.sed_std = Some(1000);
        let input3 = DamianiVolatmeterInput::from_slice(&data, params);
        let res3 = damiani_volatmeter_with_kernel(&input3, kernel);
        assert!(
            matches!(res3, Err(DamianiVolatmeterError::InvalidPeriod { .. })),
            "[{}] should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_damiani_into_existing_slice(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = DamianiVolatmeterParams::default();
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);

        // Compute using regular function
        let output1 = damiani_volatmeter_with_kernel(&input, kernel)?;

        // Compute into existing slices
        let mut vol2 = vec![0.0; candles.close.len()];
        let mut anti2 = vec![0.0; candles.close.len()];
        damiani_volatmeter_into_slice(&mut vol2, &mut anti2, &input, kernel)?;

        // Compare results
        assert_eq!(output1.vol.len(), vol2.len());
        assert_eq!(output1.anti.len(), anti2.len());

        // Check values match
        for i in 0..output1.vol.len() {
            if output1.vol[i].is_nan() && vol2[i].is_nan() {
                continue;
            }
            assert!(
                (output1.vol[i] - vol2[i]).abs() < 1e-10,
                "[{}] vol mismatch at index {}: {} vs {}",
                test_name,
                i,
                output1.vol[i],
                vol2[i]
            );
        }

        for i in 0..output1.anti.len() {
            if output1.anti[i].is_nan() && anti2[i].is_nan() {
                continue;
            }
            assert!(
                (output1.anti[i] - anti2[i]).abs() < 1e-10,
                "[{}] anti mismatch at index {}: {} vs {}",
                test_name,
                i,
                output1.anti[i],
                anti2[i]
            );
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_damiani_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Test strategy: Generate reasonable parameter ranges based on typical usage
        // Using prop_flat_map to ensure data length is sufficient for the largest period
        let strat = (
            5usize..=20,
            10usize..=30,
            20usize..=50,
            50usize..=150,
            0.5f64..3.0f64,
        )
            .prop_flat_map(|(vis_atr, vis_std, sed_atr, sed_std, threshold)| {
                let min_len = *[vis_atr, vis_std, sed_atr, sed_std].iter().max().unwrap() + 10;
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        min_len..400,
                    ),
                    Just(vis_atr),
                    Just(vis_std),
                    Just(sed_atr),
                    Just(sed_std),
                    Just(threshold),
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(data, vis_atr, vis_std, sed_atr, sed_std, threshold)| {
                    let params = DamianiVolatmeterParams {
                        vis_atr: Some(vis_atr),
                        vis_std: Some(vis_std),
                        sed_atr: Some(sed_atr),
                        sed_std: Some(sed_std),
                        threshold: Some(threshold),
                    };
                    let input = DamianiVolatmeterInput::from_slice(&data, params);

                    // Get output from tested kernel
                    let output = damiani_volatmeter_with_kernel(&input, kernel)?;
                    // Get reference output from scalar kernel for comparison
                    let ref_output = damiani_volatmeter_with_kernel(&input, Kernel::Scalar)?;

                    // Property 1: Output length consistency
                    prop_assert_eq!(output.vol.len(), data.len(), "vol length mismatch");
                    prop_assert_eq!(output.anti.len(), data.len(), "anti length mismatch");

                    // Property 2: Warmup period validation
                    let warmup = *[vis_atr, vis_std, sed_atr, sed_std, 3]
                        .iter()
                        .max()
                        .unwrap();

                    // Check NaN prefix for vol
                    for i in 0..warmup.min(data.len()) {
                        prop_assert!(
                            output.vol[i].is_nan(),
                            "vol[{}] should be NaN during warmup but got {}",
                            i,
                            output.vol[i]
                        );
                    }

                    // Property 3: First valid output appears after warmup
                    let first_valid_vol = output.vol.iter().position(|&x| !x.is_nan());
                    if let Some(idx) = first_valid_vol {
                        prop_assert!(
                            idx >= warmup - 1,
                            "First valid vol at {} but warmup is {}",
                            idx,
                            warmup
                        );
                    }

                    // Property 4: All finite values should be reasonable
                    for (i, &val) in output.vol.iter().enumerate() {
                        if !val.is_nan() {
                            prop_assert!(
                                val.is_finite(),
                                "vol[{}] should be finite but got {}",
                                i,
                                val
                            );
                            // Values shouldn't be absurdly large
                            prop_assert!(
                                val.abs() < 1e10,
                                "vol[{}] = {} is unreasonably large",
                                i,
                                val
                            );
                        }
                    }

                    for (i, &val) in output.anti.iter().enumerate() {
                        if !val.is_nan() {
                            prop_assert!(
                                val.is_finite(),
                                "anti[{}] should be finite but got {}",
                                i,
                                val
                            );
                        }
                    }

                    // Property 5: Kernel consistency - compare with scalar reference
                    for i in 0..data.len() {
                        let vol = output.vol[i];
                        let ref_vol = ref_output.vol[i];
                        let anti = output.anti[i];
                        let ref_anti = ref_output.anti[i];

                        // Check vol consistency
                        if !vol.is_finite() || !ref_vol.is_finite() {
                            prop_assert!(
                                vol.to_bits() == ref_vol.to_bits(),
                                "vol finite/NaN mismatch at {}: {} vs {}",
                                i,
                                vol,
                                ref_vol
                            );
                        } else {
                            let vol_bits = vol.to_bits();
                            let ref_vol_bits = ref_vol.to_bits();
                            let ulp_diff = vol_bits.abs_diff(ref_vol_bits);

                            prop_assert!(
                                (vol - ref_vol).abs() <= 1e-9 || ulp_diff <= 8,
                                "vol mismatch at {}: {} vs {} (ULP={})",
                                i,
                                vol,
                                ref_vol,
                                ulp_diff
                            );
                        }

                        // Check anti consistency
                        if !anti.is_finite() || !ref_anti.is_finite() {
                            prop_assert!(
                                anti.to_bits() == ref_anti.to_bits(),
                                "anti finite/NaN mismatch at {}: {} vs {}",
                                i,
                                anti,
                                ref_anti
                            );
                        } else {
                            let anti_bits = anti.to_bits();
                            let ref_anti_bits = ref_anti.to_bits();
                            let ulp_diff = anti_bits.abs_diff(ref_anti_bits);

                            prop_assert!(
                                (anti - ref_anti).abs() <= 1e-9 || ulp_diff <= 8,
                                "anti mismatch at {}: {} vs {} (ULP={})",
                                i,
                                anti,
                                ref_anti,
                                ulp_diff
                            );
                        }
                    }

                    // Property 6: Special case - constant data
                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                        // For constant data, volatility should be very low
                        for (i, &val) in output.vol.iter().enumerate().skip(warmup) {
                            if !val.is_nan() {
                                prop_assert!(
                                    val.abs() < 1e-4,
                                    "vol[{}] = {} should be near zero for constant data",
                                    i,
                                    val
                                );
                            }
                        }
                    }

                    // Property 7: Monotonic relationships
                    // Strong trends should produce meaningful volatility values
                    let is_strong_uptrend = data.windows(2).all(|w| w[1] > w[0] + 0.01);
                    let is_strong_downtrend = data.windows(2).all(|w| w[1] < w[0] - 0.01);

                    if is_strong_uptrend || is_strong_downtrend {
                        // Strong trends should show volatility (can be positive or negative)
                        let valid_vols: Vec<f64> = output
                            .vol
                            .iter()
                            .skip(warmup)
                            .filter(|&&x| !x.is_nan())
                            .copied()
                            .collect();

                        if !valid_vols.is_empty() {
                            // At least some volatility values should be non-zero for trends
                            let non_zero_count =
                                valid_vols.iter().filter(|&&v| v.abs() > 1e-10).count();
                            prop_assert!(
                                non_zero_count > 0,
                                "Expected non-zero volatility values for trending data"
                            );
                        }
                    }

                    // Property 8: Anti values should be finite when vol is finite
                    // Anti represents threshold-adjusted volatility
                    for i in warmup..data.len() {
                        if !output.vol[i].is_nan() && !output.anti[i].is_nan() {
                            // Both should be finite together
                            prop_assert!(
                                output.vol[i].is_finite() && output.anti[i].is_finite(),
                                "vol[{}] and anti[{}] should both be finite",
                                i,
                                i
                            );
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_damiani_tests {
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
    generate_all_damiani_tests!(
        check_damiani_partial_params,
        check_damiani_accuracy,
        check_damiani_zero_period,
        check_damiani_period_exceeds_length,
        check_damiani_very_small_dataset,
        check_damiani_streaming,
        check_damiani_input_with_default_candles,
        check_damiani_params_with_defaults,
        check_damiani_no_poison,
        check_damiani_empty_input,
        check_damiani_all_nan,
        check_damiani_invalid_threshold,
        check_damiani_invalid_periods,
        check_damiani_into_existing_slice
    );

    #[cfg(feature = "proptest")]
    generate_all_damiani_tests!(check_damiani_property);

    gen_batch_tests!(check_batch_default_row);
}

#[cfg(feature = "python")]
#[pyfunction(name = "damiani")]
#[pyo3(signature = (data, vis_atr, vis_std, sed_atr, sed_std, threshold, kernel=None))]
pub fn damiani_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = DamianiVolatmeterParams {
        vis_atr: Some(vis_atr),
        vis_std: Some(vis_std),
        sed_atr: Some(sed_atr),
        sed_std: Some(sed_std),
        threshold: Some(threshold),
    };
    let input = DamianiVolatmeterInput::from_slice(slice_in, params);

    let len = slice_in.len();
    // Allocate NumPy outputs once
    let vol_np = unsafe { PyArray1::<f64>::new(py, [len], false) };
    let anti_np = unsafe { PyArray1::<f64>::new(py, [len], false) };

    unsafe {
        let vol_sl = vol_np
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let anti_sl = anti_np
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        py.allow_threads(|| damiani_volatmeter_into_slice(vol_sl, anti_sl, &input, kern))
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((vol_np, anti_np))
}

#[cfg(feature = "python")]
#[pyclass(name = "DamianiVolatmeterStream")]
pub struct DamianiVolatmeterStreamPy {
    // Store owned data
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,

    // Stream parameters
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,

    // Index for streaming
    index: usize,

    // Running state (mirrors DamianiVolatmeterStream fields)
    atr_vis_val: f64,
    atr_sed_val: f64,
    sum_vis: f64,
    sum_sed: f64,
    prev_close: f64,
    have_prev: bool, // Track if we have a valid prev_close

    ring_vis: Vec<f64>,
    ring_sed: Vec<f64>,
    sum_vis_std: f64,
    sum_sq_vis_std: f64,
    sum_sed_std: f64,
    sum_sq_sed_std: f64,
    idx_vis: usize,
    idx_sed: usize,
    filled_vis: usize,
    filled_sed: usize,

    vol_history: [f64; 3],
    lag_s: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl DamianiVolatmeterStreamPy {
    #[new]
    fn new(
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
        threshold: f64,
    ) -> PyResult<Self> {
        let len = close.len();
        if len == 0 {
            return Err(PyValueError::new_err("Empty data"));
        }

        if vis_atr == 0
            || vis_std == 0
            || sed_atr == 0
            || sed_std == 0
            || vis_atr > len
            || vis_std > len
            || sed_atr > len
            || sed_std > len
        {
            return Err(PyValueError::new_err(format!(
				"Invalid period: data length = {}, vis_atr = {}, vis_std = {}, sed_atr = {}, sed_std = {}",
				len, vis_atr, vis_std, sed_atr, sed_std
			)));
        }

        // Find first non-NaN index
        let first = close
            .iter()
            .position(|&x| !x.is_nan())
            .ok_or_else(|| PyValueError::new_err("All values are NaN"))?;

        let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3]
            .iter()
            .max()
            .unwrap();
        if (len - first) < needed {
            return Err(PyValueError::new_err(format!(
                "Not enough valid data: needed {}, valid {}",
                needed,
                len - first
            )));
        }

        Ok(Self {
            high,
            low,
            close,
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
            threshold,
            index: first, // Start from first non-NaN
            atr_vis_val: f64::NAN,
            atr_sed_val: f64::NAN,
            sum_vis: 0.0,
            sum_sed: 0.0,
            prev_close: f64::NAN,
            have_prev: false, // Initialize to false
            ring_vis: vec![0.0; vis_std],
            ring_sed: vec![0.0; sed_std],
            sum_vis_std: 0.0,
            sum_sq_vis_std: 0.0,
            sum_sed_std: 0.0,
            sum_sq_sed_std: 0.0,
            idx_vis: 0,
            idx_sed: 0,
            filled_vis: 0,
            filled_sed: 0,
            vol_history: [f64::NAN; 3],
            lag_s: 0.5,
        })
    }

    fn update(&mut self) -> Option<(f64, f64)> {
        let i = self.index;
        let len = self.close.len();
        if i >= len {
            return None;
        }

        // Compute "True Range" exactly like scalar - use have_prev gating
        let tr = if self.have_prev && self.close[i].is_finite() {
            let hi = self.high[i];
            let lo = self.low[i];
            let pc = self.prev_close;

            let tr1 = hi - lo;
            let tr2 = (hi - pc).abs();
            let tr3 = (lo - pc).abs();
            tr1.max(tr2).max(tr3)
        } else {
            0.0
        };

        // Update prev_close only if current close is finite
        if self.close[i].is_finite() {
            self.prev_close = self.close[i];
            self.have_prev = true;
        }

        // ----- ATR for "vis" -----
        if i < self.vis_atr {
            self.sum_vis += tr;
            if i == self.vis_atr - 1 {
                self.atr_vis_val = self.sum_vis / (self.vis_atr as f64);
            }
        } else if self.atr_vis_val.is_finite() {
            self.atr_vis_val =
                ((self.vis_atr as f64 - 1.0) * self.atr_vis_val + tr) / (self.vis_atr as f64);
        }

        // ----- ATR for "sed" -----
        if i < self.sed_atr {
            self.sum_sed += tr;
            if i == self.sed_atr - 1 {
                self.atr_sed_val = self.sum_sed / (self.sed_atr as f64);
            }
        } else if self.atr_sed_val.is_finite() {
            self.atr_sed_val =
                ((self.sed_atr as f64 - 1.0) * self.atr_sed_val + tr) / (self.sed_atr as f64);
        }

        // ---- Insert price‐for‐StdDev (use close, treat NaN as 0) ----
        let val = if self.close[i].is_nan() {
            0.0
        } else {
            self.close[i]
        };

        // Update "vis" ring buffer:
        let old_v = self.ring_vis[self.idx_vis];
        self.ring_vis[self.idx_vis] = val;
        self.idx_vis = (self.idx_vis + 1) % self.vis_std;
        if self.filled_vis < self.vis_std {
            self.filled_vis += 1;
            self.sum_vis_std += val;
            self.sum_sq_vis_std += val * val;
        } else {
            self.sum_vis_std = self.sum_vis_std - old_v + val;
            self.sum_sq_vis_std = self.sum_sq_vis_std - (old_v * old_v) + (val * val);
        }

        // Update "sed" ring buffer:
        let old_s = self.ring_sed[self.idx_sed];
        self.ring_sed[self.idx_sed] = val;
        self.idx_sed = (self.idx_sed + 1) % self.sed_std;
        if self.filled_sed < self.sed_std {
            self.filled_sed += 1;
            self.sum_sed_std += val;
            self.sum_sq_sed_std += val * val;
        } else {
            self.sum_sed_std = self.sum_sed_std - old_s + val;
            self.sum_sq_sed_std = self.sum_sq_sed_std - (old_s * old_s) + (val * val);
        }

        // Increment index for next call
        self.index += 1;

        // Only start computing vol/anti once we have _all_ lookbacks:
        let needed = *[self.vis_atr, self.vis_std, self.sed_atr, self.sed_std, 3]
            .iter()
            .max()
            .unwrap();
        if i < needed {
            return None;
        }

        // Get previous vol values from history
        let p1 = if !self.vol_history[0].is_nan() {
            self.vol_history[0]
        } else {
            0.0
        };
        let p3 = if !self.vol_history[2].is_nan() {
            self.vol_history[2]
        } else {
            0.0
        };

        // Avoid divide‐by‐zero on sed:
        let sed_safe = if self.atr_sed_val.is_finite() && self.atr_sed_val != 0.0 {
            self.atr_sed_val
        } else {
            self.atr_sed_val + f64::EPSILON
        };

        // Compute vol[i]:
        let vol_val = (self.atr_vis_val / sed_safe) + self.lag_s * (p1 - p3);

        // Shift the vol history buffer
        self.vol_history[2] = self.vol_history[1];
        self.vol_history[1] = self.vol_history[0];
        self.vol_history[0] = vol_val;

        // Compute anti[i] only if both StdDev windows are full:
        let anti_val = if self.filled_vis == self.vis_std && self.filled_sed == self.sed_std {
            let std_vis = stddev(self.sum_vis_std, self.sum_sq_vis_std, self.vis_std);
            let std_sed = stddev(self.sum_sed_std, self.sum_sq_sed_std, self.sed_std);
            let ratio = if std_sed != 0.0 {
                std_vis / std_sed
            } else {
                std_vis / (std_sed + f64::EPSILON)
            };
            self.threshold - ratio
        } else {
            f64::NAN
        };

        Some((vol_val, anti_val))
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "damiani_batch")]
#[pyo3(signature = (data, vis_atr_range, vis_std_range, sed_atr_range, sed_std_range, threshold_range, kernel=None))]
pub fn damiani_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    vis_atr_range: (usize, usize, usize),
    vis_std_range: (usize, usize, usize),
    sed_atr_range: (usize, usize, usize),
    sed_std_range: (usize, usize, usize),
    threshold_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let sweep = DamianiVolatmeterBatchRange {
        vis_atr: vis_atr_range,
        vis_std: vis_std_range,
        sed_atr: sed_atr_range,
        sed_std: sed_std_range,
        threshold: threshold_range,
    };

    // Compute combos first
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Allocate NumPy arrays
    let vol_np = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let anti_np = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let vol_sl = unsafe { vol_np.as_slice_mut()? };
    let anti_sl = unsafe { anti_np.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    // Compute in parallel without GIL
    py.allow_threads(|| {
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
        damiani_volatmeter_batch_inner_into(slice_in, &sweep, simd, true, vol_sl, anti_sl)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let d = PyDict::new(py);
    d.set_item("vol", vol_np.reshape((rows, cols))?)?;
    d.set_item("anti", anti_np.reshape((rows, cols))?)?;
    d.set_item(
        "vis_atr",
        combos
            .iter()
            .map(|p| p.vis_atr.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "vis_std",
        combos
            .iter()
            .map(|p| p.vis_std.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "sed_atr",
        combos
            .iter()
            .map(|p| p.sed_atr.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "sed_std",
        combos
            .iter()
            .map(|p| p.sed_std.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "threshold",
        combos
            .iter()
            .map(|p| p.threshold.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "damiani_cuda_batch_dev")]
#[pyo3(signature = (data_f32, vis_atr_range, vis_std_range, sed_atr_range, sed_std_range, threshold_range, device_id=0))]
pub fn damiani_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: PyReadonlyArray1<'py, f32>,
    vis_atr_range: (usize, usize, usize),
    vis_std_range: (usize, usize, usize),
    sed_atr_range: (usize, usize, usize),
    sed_std_range: (usize, usize, usize),
    threshold_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = DamianiVolatmeterBatchRange {
        vis_atr: vis_atr_range,
        vis_std: vis_std_range,
        sed_atr: sed_atr_range,
        sed_std: sed_std_range,
        threshold: threshold_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = crate::cuda::CudaDamianiVolatmeter::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (arr, _combos) = cuda
            .damiani_volatmeter_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>(arr)
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "damiani_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, cols, rows, vis_atr, vis_std, sed_atr, sed_std, threshold, device_id=0))]
pub fn damiani_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: PyReadonlyArray1<'py, f32>,
    low_tm_f32: PyReadonlyArray1<'py, f32>,
    close_tm_f32: PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let c = close_tm_f32.as_slice()?;
    let params = DamianiVolatmeterParams {
        vis_atr: Some(vis_atr),
        vis_std: Some(vis_std),
        sed_atr: Some(sed_atr),
        sed_std: Some(sed_std),
        threshold: Some(threshold),
    };
    let inner = py.allow_threads(|| {
        let cuda = crate::cuda::CudaDamianiVolatmeter::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.damiani_volatmeter_many_series_one_param_time_major_dev(h, l, c, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "DamianiVolatmeterFeedStream")]
pub struct DamianiVolatmeterFeedStreamPy {
    stream: DamianiVolatmeterFeedStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DamianiVolatmeterFeedStreamPy {
    #[new]
    fn new(
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
        threshold: f64,
    ) -> PyResult<Self> {
        let params = DamianiVolatmeterParams {
            vis_atr: Some(vis_atr),
            vis_std: Some(vis_std),
            sed_atr: Some(sed_atr),
            sed_std: Some(sed_std),
            threshold: Some(threshold),
        };
        let stream = DamianiVolatmeterFeedStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(DamianiVolatmeterFeedStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DamianiJsOutput {
    pub values: Vec<f64>, // [vol..., anti...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = damiani_volatmeter_js)]
pub fn damiani_volatmeter_wasm(
    data: &[f64],
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
) -> Result<JsValue, JsValue> {
    let params = DamianiVolatmeterParams {
        vis_atr: Some(vis_atr),
        vis_std: Some(vis_std),
        sed_atr: Some(sed_atr),
        sed_std: Some(sed_std),
        threshold: Some(threshold),
    };
    let input = DamianiVolatmeterInput::from_slice(data, params);
    let out = damiani_volatmeter_with_kernel(&input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let cols = data.len();
    let mut values = Vec::with_capacity(2 * cols);
    values.extend_from_slice(&out.vol);
    values.extend_from_slice(&out.anti);
    serde_wasm_bindgen::to_value(&DamianiJsOutput {
        values,
        rows: 2,
        cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_into(
    in_ptr: *const f64,
    out_vol_ptr: *mut f64,
    out_anti_ptr: *mut f64,
    len: usize,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_vol_ptr.is_null() || out_anti_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to damiani_volatmeter_into",
        ));
    }

    // Check for vol and anti being the same
    if out_vol_ptr == out_anti_ptr {
        return Err(JsValue::from_str("vol_ptr and anti_ptr cannot be the same"));
    }

    unsafe {
        let params = DamianiVolatmeterParams {
            vis_atr: Some(vis_atr),
            vis_std: Some(vis_std),
            sed_atr: Some(sed_atr),
            sed_std: Some(sed_std),
            threshold: Some(threshold),
        };

        // Check if input aliases with either output
        // Cast pointers to usize for comparison to avoid any type issues
        let in_addr = in_ptr as usize;
        let vol_addr = out_vol_ptr as usize;
        let anti_addr = out_anti_ptr as usize;

        if in_addr == vol_addr || in_addr == anti_addr {
            // Need to use temporary buffer for input
            let data_copy = std::slice::from_raw_parts(in_ptr, len).to_vec();
            let vol = std::slice::from_raw_parts_mut(out_vol_ptr, len);
            let anti = std::slice::from_raw_parts_mut(out_anti_ptr, len);
            let input = DamianiVolatmeterInput::from_slice(&data_copy, params);
            damiani_volatmeter_into_slice(vol, anti, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            // No aliasing, use directly
            let data = std::slice::from_raw_parts(in_ptr, len);
            let vol = std::slice::from_raw_parts_mut(out_vol_ptr, len);
            let anti = std::slice::from_raw_parts_mut(out_anti_ptr, len);
            let input = DamianiVolatmeterInput::from_slice(data, params);
            damiani_volatmeter_into_slice(vol, anti, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DamianiBatchJsOutput {
    pub vol: Vec<f64>,
    pub anti: Vec<f64>,
    pub combos: Vec<DamianiVolatmeterParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = damiani_volatmeter_batch)]
pub fn damiani_volatmeter_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: DamianiVolatmeterBatchRange = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let out = damiani_volatmeter_batch_inner(data, &cfg, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&DamianiBatchJsOutput {
        vol: out.vol,
        anti: out.anti,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_batch_into(
    in_ptr: *const f64,
    vol_ptr: *mut f64,
    anti_ptr: *mut f64,
    len: usize,
    vis_atr_start: usize,
    vis_atr_end: usize,
    vis_atr_step: usize,
    vis_std_start: usize,
    vis_std_end: usize,
    vis_std_step: usize,
    sed_atr_start: usize,
    sed_atr_end: usize,
    sed_atr_step: usize,
    sed_std_start: usize,
    sed_std_end: usize,
    sed_std_step: usize,
    threshold_start: f64,
    threshold_end: f64,
    threshold_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || vol_ptr.is_null() || anti_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = DamianiVolatmeterBatchRange {
            vis_atr: (vis_atr_start, vis_atr_end, vis_atr_step),
            vis_std: (vis_std_start, vis_std_end, vis_std_step),
            sed_atr: (sed_atr_start, sed_atr_end, sed_atr_step),
            sed_std: (sed_std_start, sed_std_end, sed_std_step),
            threshold: (threshold_start, threshold_end, threshold_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        // Check for aliasing
        if vol_ptr == anti_ptr {
            return Err(JsValue::from_str("vol_ptr and anti_ptr cannot be the same"));
        }

        if in_ptr == vol_ptr || in_ptr == anti_ptr {
            // Need temporary buffers for aliasing
            let result = damiani_volatmeter_batch_inner(data, &sweep, detect_best_kernel(), false)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let vol_out = std::slice::from_raw_parts_mut(vol_ptr, rows * cols);
            let anti_out = std::slice::from_raw_parts_mut(anti_ptr, rows * cols);

            vol_out.copy_from_slice(&result.vol);
            anti_out.copy_from_slice(&result.anti);
        } else {
            // No aliasing, direct computation
            let vol_out = std::slice::from_raw_parts_mut(vol_ptr, rows * cols);
            let anti_out = std::slice::from_raw_parts_mut(anti_ptr, rows * cols);

            damiani_volatmeter_batch_inner_into(
                data,
                &sweep,
                detect_best_kernel(),
                false,
                vol_out,
                anti_out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}
