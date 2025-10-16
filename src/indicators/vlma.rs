//! # Variable Length Moving Average (VLMA)
//!
//! VLMA is an adaptive moving average that adjusts its period dynamically based on price deviation
//! from a reference moving average, becoming faster in trends and slower in consolidations.
//!
//! ## Parameters
//! - **min_period**: Minimum adaptive period. Defaults to 5.
//! - **max_period**: Maximum adaptive period. Defaults to 50.
//! - **matype**: Moving average type for reference. Defaults to "sma".
//! - **devtype**: Deviation type (0=std, 1=mad, 2=median). Defaults to 0.
//!
//! ## Returns
//! - **`Ok(VlmaOutput)`** containing a `Vec<f64>` of adaptive moving average values matching input length.
//! - **`Err(VlmaError)`** on invalid parameters or insufficient data.
//!
//! ## Developer Notes
//! - SIMD: AVX2/AVX512 remain stubs (delegate to scalar). VLMA is sequential and branchy; time-wise SIMD offers no clear win, so selection short-circuits to scalar.
//! - Scalar: hot loop optimized with branch-reduced period updates and a LUT for smoothing constants; preserves warmup/outputs.
//! - Streaming: O(1) for default case (matype="sma", devtype=0) via rolling sums/sumsq and a LUT of smoothing
//!   constants; other matypes/devtypes retain the correct O(n) fallback. Matches batch semantics exactly.
//! - Memory: uses `alloc_with_nan_prefix` and avoids O(N) temporaries beyond required reference series.
//! - Batch: parallel per-row sweep. Added row-optimized fast path for SMA + stddev using shared prefix sums
//!   to compute mean/stddev in O(1) per index; other matypes/devtypes fall back to scalar per row.
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

use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
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
use std::error::Error;
use thiserror::Error;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaVlma};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

impl<'a> AsRef<[f64]> for VlmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VlmaData::Slice(sl) => sl,
            VlmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// Fast math helpers used by the optimized scalar path. Keep tiny and always-inline
// to match ALMA-style patterns and avoid call overhead in hot loops.
#[inline(always)]
fn fast_ema_update(last: f64, x: f64, sc: f64) -> f64 {
    // last + sc * (x - last), fused when available
    (x - last).mul_add(sc, last)
}

#[inline(always)]
fn fast_clamp_period(p: isize, min_p: usize, max_p: usize) -> usize {
    let lo = min_p as isize;
    let hi = max_p as isize;
    if p < lo {
        min_p
    } else if p > hi {
        max_p
    } else {
        p as usize
    }
}

#[inline(always)]
fn fast_std_from_sums(sum: f64, sumsq: f64, inv_n: f64) -> (f64, f64) {
    let m = sum * inv_n;
    // var = E[x^2] - m^2  ==> (sumsq * inv_n) - m*m
    let var = (-m).mul_add(m, sumsq * inv_n);
    let dv = if var <= 0.0 { 0.0 } else { var.sqrt() };
    (m, dv)
}

#[derive(Debug, Clone)]
pub enum VlmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VlmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VlmaParams {
    pub min_period: Option<usize>,
    pub max_period: Option<usize>,
    pub matype: Option<String>,
    pub devtype: Option<usize>,
}

impl Default for VlmaParams {
    fn default() -> Self {
        Self {
            min_period: Some(5),
            max_period: Some(50),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VlmaInput<'a> {
    pub data: VlmaData<'a>,
    pub params: VlmaParams,
}

impl<'a> VlmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VlmaParams) -> Self {
        Self {
            data: VlmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VlmaParams) -> Self {
        Self {
            data: VlmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VlmaParams::default())
    }
    #[inline]
    pub fn get_min_period(&self) -> usize {
        self.params.min_period.unwrap_or(5)
    }
    #[inline]
    pub fn get_max_period(&self) -> usize {
        self.params.max_period.unwrap_or(50)
    }
    #[inline]
    pub fn get_matype(&self) -> String {
        self.params
            .matype
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
    #[inline]
    pub fn get_devtype(&self) -> usize {
        self.params.devtype.unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct VlmaBuilder {
    min_period: Option<usize>,
    max_period: Option<usize>,
    matype: Option<String>,
    devtype: Option<usize>,
    kernel: Kernel,
}

impl Default for VlmaBuilder {
    fn default() -> Self {
        Self {
            min_period: None,
            max_period: None,
            matype: None,
            devtype: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VlmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn min_period(mut self, n: usize) -> Self {
        self.min_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn max_period(mut self, n: usize) -> Self {
        self.max_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn matype<S: Into<String>>(mut self, t: S) -> Self {
        self.matype = Some(t.into());
        self
    }
    #[inline(always)]
    pub fn devtype(mut self, d: usize) -> Self {
        self.devtype = Some(d);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VlmaOutput, VlmaError> {
        let p = VlmaParams {
            min_period: self.min_period,
            max_period: self.max_period,
            matype: self.matype,
            devtype: self.devtype,
        };
        let i = VlmaInput::from_candles(c, "close", p);
        vlma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VlmaOutput, VlmaError> {
        let p = VlmaParams {
            min_period: self.min_period,
            max_period: self.max_period,
            matype: self.matype,
            devtype: self.devtype,
        };
        let i = VlmaInput::from_slice(d, p);
        vlma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VlmaStream, VlmaError> {
        let p = VlmaParams {
            min_period: self.min_period,
            max_period: self.max_period,
            matype: self.matype,
            devtype: self.devtype,
        };
        VlmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum VlmaError {
    #[error("vlma: Empty data provided.")]
    EmptyData,
    #[error("vlma: min_period={min_period} is greater than max_period={max_period}.")]
    InvalidPeriodRange {
        min_period: usize,
        max_period: usize,
    },
    #[error("vlma: Invalid period: min_period={min_period}, max_period={max_period}, data length={data_len}.")]
    InvalidPeriod {
        min_period: usize,
        max_period: usize,
        data_len: usize,
    },
    #[error("vlma: All values are NaN.")]
    AllValuesNaN,
    #[error("vlma: Not enough valid data: needed={needed}, valid={valid}.")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vlma: Error in MA calculation: {0}")]
    MaError(String),
    #[error("vlma: Error in Deviation calculation: {0}")]
    DevError(String),
}

#[inline]
pub fn vlma(input: &VlmaInput) -> Result<VlmaOutput, VlmaError> {
    vlma_with_kernel(input, Kernel::Auto)
}

pub fn vlma_with_kernel(input: &VlmaInput, kernel: Kernel) -> Result<VlmaOutput, VlmaError> {
    let (data, min_p, max_p, matype, devtype, first, chosen) = vlma_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + max_p - 1);
    vlma_compute_into(
        data, min_p, max_p, &matype, devtype, first, chosen, &mut out,
    )?;
    Ok(VlmaOutput { values: out })
}

#[inline]
pub fn vlma_into_slice(dst: &mut [f64], input: &VlmaInput, kern: Kernel) -> Result<(), VlmaError> {
    let (data, min_p, max_p, matype, devtype, first, chosen) = vlma_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(VlmaError::InvalidPeriod {
            min_period: 0,
            max_period: 0,
            data_len: data.len(),
        });
    }
    vlma_compute_into(data, min_p, max_p, &matype, devtype, first, chosen, dst)?;
    // Set warmup NaNs, but preserve the initial value at first_valid (VLMA specific)
    let warm_end = first + max_p - 1;
    for i in 0..warm_end {
        if i != first {
            // Preserve the value at first_valid
            dst[i] = f64::NAN;
        }
    }
    Ok(())
}

#[inline(always)]
fn vlma_prepare<'a>(
    input: &'a VlmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, String, usize, usize, Kernel), VlmaError> {
    let data: &[f64] = input.as_ref();

    if data.is_empty() {
        return Err(VlmaError::EmptyData);
    }

    let min_period = input.get_min_period();
    let max_period = input.get_max_period();
    if min_period > max_period {
        return Err(VlmaError::InvalidPeriodRange {
            min_period,
            max_period,
        });
    }

    if max_period == 0 || max_period > data.len() {
        return Err(VlmaError::InvalidPeriod {
            min_period,
            max_period,
            data_len: data.len(),
        });
    }

    let first = data
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(VlmaError::AllValuesNaN)?;

    if (data.len() - first) < max_period {
        return Err(VlmaError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }

    let matype = input.get_matype();
    let devtype = input.get_devtype();

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, min_period, max_period, matype, devtype, first, chosen))
}

#[inline(always)]
fn vlma_compute_into(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                // Fast path: inline SMA + stddev when requested (matches reference semantics)
                if matype == "sma" && devtype == 0 {
                    vlma_scalar_sma_stddev_into(data, min_period, max_period, first, out)?;
                } else {
                    vlma_scalar_into(data, min_period, max_period, matype, devtype, first, out)?;
                }
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vlma_avx2_into(data, min_period, max_period, matype, devtype, first, out)?;
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vlma_avx512_into(data, min_period, max_period, matype, devtype, first, out)?;
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Classic kernel optimization for VLMA with inline SMA calculation
/// This eliminates the function call overhead for the reference MA calculation
/// Optimized for the default MA type: SMA
pub unsafe fn vlma_scalar_classic(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    // Delegate to fast SMA+StdDev path if applicable; else fallback to generic scalar
    if matype == "sma" && devtype == 0 {
        return vlma_scalar_sma_stddev_into(data, min_period, max_period, first_valid, out);
    }
    vlma_scalar_into(
        data,
        min_period,
        max_period,
        matype,
        devtype,
        first_valid,
        out,
    )
}

/// Optimized VLMA path for matype="sma" and devtype=0 (standard deviation)
/// Computes rolling SMA and stddev in a single O(n) pass with NaN tracking,
/// and applies the same band logic and warmup semantics as the reference path.
#[inline(always)]
pub unsafe fn vlma_scalar_sma_stddev_into(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    debug_assert_eq!(out.len(), data.len());
    let len = data.len();
    if len == 0 {
        return Ok(());
    }

    // Warmup and initial seed
    let warm_end = first_valid + max_period - 1;
    let x0 = *data.get_unchecked(first_valid);
    *out.get_unchecked_mut(first_valid) = x0;

    // Period LUT and state
    let min_pi = if min_period == 0 { 1 } else { min_period };
    let max_pi = core::cmp::max(max_period, min_pi);
    let mut last_p: usize = max_pi;
    let mut sc_lut = Vec::with_capacity(max_pi + 1);
    sc_lut.push(0.0);
    for p in 1..=max_pi {
        sc_lut.push(2.0 / (p as f64 + 1.0));
    }
    let sc_ptr = sc_lut.as_ptr();

    // Bands constants
    const D175: f64 = 1.75;
    const D025: f64 = 0.25;

    // EMA-like state
    let mut last_val = x0;

    // 1) Advance warmup without adaptation (match reference behavior)
    let mut i = first_valid + 1;
    while i < len && i < warm_end {
        let x = *data.get_unchecked(i);
        if !x.is_nan() {
            let sc = *sc_ptr.add(last_p);
            last_val = fast_ema_update(last_val, x, sc);
        }
        i += 1;
    }

    if warm_end >= len {
        return Ok(());
    }

    // 2) Initialize rolling sums for the first full window ending at warm_end
    let mut sum = 0.0_f64;
    let mut sumsq = 0.0_f64;
    let mut nan_count: usize = 0;
    for k in 0..max_period {
        let v = *data.get_unchecked(first_valid + k);
        if v.is_finite() {
            sum += v;
            sumsq += v * v;
        } else {
            nan_count += 1;
        }
    }
    let inv_n = 1.0 / (max_period as f64);

    // 3) Steady-state loop: compute mean/dev on the fly and adapt period
    i = warm_end;
    while i < len {
        let x = *data.get_unchecked(i);

        if x.is_nan() {
            // passthrough NaN after warmup; keep last state
            *out.get_unchecked_mut(i) = f64::NAN;
        } else {
            // Mean and stddev for current window (ending at i)
            let (m, dv) = if nan_count == 0 {
                let m = sum * inv_n;
                let var = (sumsq * inv_n) - m * m;
                let dv = if var < 0.0 { 0.0 } else { var.sqrt() };
                (m, dv)
            } else {
                (f64::NAN, f64::NAN)
            };

            // Adapt period using same band rules
            let prev_p = if last_p == 0 { max_pi } else { last_p };
            let mut next_p = prev_p;
            if m.is_finite() && dv.is_finite() {
                let d175 = dv * D175;
                let d025 = dv * D025;
                let a = m - d175;
                let b = m - d025;
                let c = m + d025;
                let d = m + d175;
                let inc_fast = ((x < a) as i32) | ((x > d) as i32);
                let inc_slow = ((x >= b) as i32) & ((x <= c) as i32);
                let delta = inc_slow - inc_fast;
                let p_tmp = prev_p as isize + delta as isize;
                next_p = if p_tmp < min_pi as isize {
                    min_pi
                } else if p_tmp > max_pi as isize {
                    max_pi
                } else {
                    p_tmp as usize
                };
            }

            // EMA-like update
            let sc = *sc_ptr.add(next_p);
            last_val = fast_ema_update(last_val, x, sc);
            last_p = next_p;
            *out.get_unchecked_mut(i) = last_val;
        }

        // Prepare rolling window for next position (i+1)
        let next = i + 1;
        if next < len {
            let out_idx = next - max_period;
            let v_out = *data.get_unchecked(out_idx);
            if v_out.is_finite() {
                sum -= v_out;
                sumsq -= v_out * v_out;
            } else {
                nan_count = nan_count.saturating_sub(1);
            }
            let v_in = *data.get_unchecked(next);
            if v_in.is_finite() {
                sum += v_in;
                sumsq += v_in * v_in;
            } else {
                nan_count += 1;
            }
        }

        i = next;
    }

    Ok(())
}

#[inline(always)]
unsafe fn vlma_scalar_into(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    debug_assert_eq!(out.len(), data.len());

    // Reference series via existing builders to preserve semantics/accuracy
    let mean = ma(matype, MaData::Slice(data), max_period)
        .map_err(|e| VlmaError::MaError(e.to_string()))?;
    let dev = deviation(&DevInput::from_slice(
        data,
        DevParams {
            period: Some(max_period),
            devtype: Some(devtype),
        },
    ))
    .map_err(|e| VlmaError::DevError(e.to_string()))?;

    let len = data.len();
    if len == 0 {
        return Ok(());
    }

    // Warmup boundary; caller will set warmup NaNs afterwards
    let warm_end = first_valid + max_period - 1;

    // Initial state and seed
    let x0 = *data.get_unchecked(first_valid);
    *out.get_unchecked_mut(first_valid) = x0;

    // Keep periods as integer indices and precompute smoothing constants
    let min_pi = if min_period == 0 { 1 } else { min_period };
    let max_pi = core::cmp::max(max_period, min_pi);
    let mut last_p: usize = max_pi;

    // sc_lut[p] = 2 / (p + 1)
    let mut sc_lut = Vec::with_capacity(max_pi + 1);
    sc_lut.push(0.0); // unused index 0
    for p in 1..=max_pi {
        sc_lut.push(2.0 / (p as f64 + 1.0));
    }
    debug_assert_eq!(sc_lut.len(), max_pi + 1);
    let sc_ptr = sc_lut.as_ptr();

    // Band multipliers
    const D175: f64 = 1.75;
    const D025: f64 = 0.25;

    // Current EMA-like value
    let mut last_val = x0;

    // Phase 1: advance through warmup without writes (except first_valid already written)
    let mut i = first_valid + 1;
    while i < len && i < warm_end {
        let x = *data.get_unchecked(i);
        if x.is_nan() {
            i += 1;
            continue;
        }

        let m = mean[i];
        let dv = dev[i];

        // Previous period and next period selection
        let prev_p = if last_p == 0 { max_pi } else { last_p };
        let mut next_p = prev_p;

        if m.is_finite() && dv.is_finite() {
            let d175 = dv * D175;
            let d025 = dv * D025;

            let a = m - d175;
            let b = m - d025;
            let c = m + d025;
            let d = m + d175;

            // delta = +1 in [b, c], -1 if < a or > d, else 0
            let inc_fast = ((x < a) as i32) | ((x > d) as i32);
            let inc_slow = ((x >= b) as i32) & ((x <= c) as i32);
            let delta = inc_slow - inc_fast; // -1, 0, or +1

            let p_tmp = prev_p as isize + delta as isize;
            next_p = if p_tmp < min_pi as isize {
                min_pi
            } else if p_tmp > max_pi as isize {
                max_pi
            } else {
                p_tmp as usize
            };
        }

        // EMA-like update: y = last + sc * (x - last)
        let sc = *sc_ptr.add(next_p);
        last_val = (x - last_val).mul_add(sc, last_val);
        last_p = next_p;

        i += 1;
    }

    // Phase 2: steady-state with writes
    while i < len {
        let x = *data.get_unchecked(i);

        if x.is_nan() {
            // Explicit NaN passthrough after warmup
            *out.get_unchecked_mut(i) = f64::NAN;
            i += 1;
            continue;
        }

        let m = mean[i];
        let dv = dev[i];

        let prev_p = if last_p == 0 { max_pi } else { last_p };
        let mut next_p = prev_p;

        if m.is_finite() && dv.is_finite() {
            let d175 = dv * D175;
            let d025 = dv * D025;

            let a = m - d175;
            let b = m - d025;
            let c = m + d025;
            let d = m + d175;

            let inc_fast = ((x < a) as i32) | ((x > d) as i32);
            let inc_slow = ((x >= b) as i32) & ((x <= c) as i32);
            let delta = inc_slow - inc_fast;

            let p_tmp = prev_p as isize + delta as isize;
            next_p = if p_tmp < min_pi as isize {
                min_pi
            } else if p_tmp > max_pi as isize {
                max_pi
            } else {
                p_tmp as usize
            };
        }

        let sc = *sc_ptr.add(next_p);
        last_val = (x - last_val).mul_add(sc, last_val);
        last_p = next_p;

        *out.get_unchecked_mut(i) = last_val;
        i += 1;
    }

    Ok(())
}

#[inline(always)]
unsafe fn vlma_row_scalar(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    vlma_scalar_into(
        data,
        min_period,
        max_period,
        matype,
        devtype,
        first_valid,
        out,
    )
}

/// Row-optimized VLMA using shared prefix sums (SMA + stddev only)
#[inline(always)]
unsafe fn vlma_row_fast_sma_std_prefix(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    first_valid: usize,
    ps_sum: &[f64],
    ps_sumsq: &[f64],
    ps_cnt: &[usize],
    out: &mut [f64],
) -> Result<(), VlmaError> {
    debug_assert_eq!(out.len(), data.len());
    let len = data.len();
    if len == 0 {
        return Ok(());
    }

    let warm_end = first_valid + max_period - 1;
    let x0 = *data.get_unchecked(first_valid);
    // Initial value at first_valid is pre-set by caller; keep in sync
    *out.get_unchecked_mut(first_valid) = x0;

    let min_pi = if min_period == 0 { 1 } else { min_period };
    let max_pi = core::cmp::max(max_period, min_pi);
    let mut last_p: usize = max_pi;

    // Smoothing constants LUT
    let mut sc_lut = Vec::with_capacity(max_pi + 1);
    sc_lut.push(0.0);
    for p in 1..=max_pi {
        sc_lut.push(2.0 / (p as f64 + 1.0));
    }
    let sc_ptr = sc_lut.as_ptr();

    const D175: f64 = 1.75;
    const D025: f64 = 0.25;

    let mut last_val = x0;

    // Warmup advance
    let mut i = first_valid + 1;
    while i < len && i < warm_end {
        let x = *data.get_unchecked(i);
        if x.is_finite() {
            let sc = *sc_ptr.add(last_p);
            last_val = (x - last_val).mul_add(sc, last_val);
        }
        i += 1;
    }
    if warm_end >= len {
        return Ok(());
    }

    // Steady-state
    while i < len {
        let x = *data.get_unchecked(i);
        if !x.is_finite() {
            *out.get_unchecked_mut(i) = f64::NAN;
        } else {
            // Window [i-max_period+1, i]
            let start = i + 1 - max_period;
            let cnt = *ps_cnt.get_unchecked(i + 1) - *ps_cnt.get_unchecked(start);
            let (m, dv) = if cnt == max_period {
                let sum = *ps_sum.get_unchecked(i + 1) - *ps_sum.get_unchecked(start);
                let sumsq = *ps_sumsq.get_unchecked(i + 1) - *ps_sumsq.get_unchecked(start);
                let inv = 1.0 / (max_period as f64);
                let m = sum * inv;
                let var = (sumsq * inv) - m * m;
                let dv = if var < 0.0 { 0.0 } else { var.sqrt() };
                (m, dv)
            } else {
                (f64::NAN, f64::NAN)
            };

            let prev_p = if last_p == 0 { max_pi } else { last_p };
            let mut next_p = prev_p;
            if m.is_finite() && dv.is_finite() {
                let d175 = dv * D175;
                let d025 = dv * D025;
                let a = m - d175;
                let b = m - d025;
                let c = m + d025;
                let d = m + d175;
                let inc_fast = ((x < a) as i32) | ((x > d) as i32);
                let inc_slow = ((x >= b) as i32) & ((x <= c) as i32);
                let delta = inc_slow - inc_fast;
                let p_tmp = prev_p as isize + delta as isize;
                next_p = if p_tmp < min_pi as isize {
                    min_pi
                } else if p_tmp > max_pi as isize {
                    max_pi
                } else {
                    p_tmp as usize
                };
            }
            let sc = *sc_ptr.add(next_p);
            last_val = (x - last_val).mul_add(sc, last_val);
            last_p = next_p;
            *out.get_unchecked_mut(i) = last_val;
        }
        i += 1;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx2_into(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    // For now, delegate to scalar implementation
    vlma_scalar_into(
        data,
        min_period,
        max_period,
        matype,
        devtype,
        first_valid,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_row_avx2(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    vlma_avx2_into(
        data,
        min_period,
        max_period,
        matype,
        devtype,
        first_valid,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_into(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    if max_period <= 32 {
        vlma_avx512_short_into(
            data,
            min_period,
            max_period,
            matype,
            devtype,
            first_valid,
            out,
        )
    } else {
        vlma_avx512_long_into(
            data,
            min_period,
            max_period,
            matype,
            devtype,
            first_valid,
            out,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_row_avx512(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    vlma_avx512_into(
        data,
        min_period,
        max_period,
        matype,
        devtype,
        first_valid,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_short_into(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    // For now, delegate to scalar implementation
    vlma_scalar_into(
        data,
        min_period,
        max_period,
        matype,
        devtype,
        first_valid,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_long_into(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), VlmaError> {
    // For now, delegate to scalar implementation
    vlma_scalar_into(
        data,
        min_period,
        max_period,
        matype,
        devtype,
        first_valid,
        out,
    )
}

// Decision: Streaming kernel is O(1) for SMA+stddev using rolling sums; fall back to O(n) otherwise.
#[derive(Debug, Clone)]
pub struct VlmaStream {
    // existing/public-facing fields
    min_period: usize,
    max_period: usize,
    matype: String,
    devtype: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    period: f64, // tracked as f64 for API parity
    last_val: f64,

    // O(1) state for SMA + stddev fast path
    sum: f64,
    sumsq: f64,
    nan_count: usize,
    inv_n: f64,
    last_p: usize,    // integer period mirror for LUT index
    sc_lut: Vec<f64>, // sc_lut[p] = 2/(p+1), index 0 unused
}

impl VlmaStream {
    pub fn try_new(params: VlmaParams) -> Result<Self, VlmaError> {
        let min_period = params.min_period.unwrap_or(5);
        let max_period = params.max_period.unwrap_or(50);
        let matype = params.matype.unwrap_or_else(|| "sma".to_string());
        let devtype = params.devtype.unwrap_or(0);

        if min_period > max_period {
            return Err(VlmaError::InvalidPeriodRange {
                min_period,
                max_period,
            });
        }
        if max_period == 0 {
            return Err(VlmaError::InvalidPeriod {
                min_period,
                max_period,
                data_len: 0,
            });
        }

        // Precompute smoothing constants α = 2/(p+1) for p∈[1..=max_period].
        let mut sc_lut = Vec::with_capacity(max_period + 1);
        sc_lut.push(0.0); // index 0 unused for alignment with period indexing
        for p in 1..=max_period {
            sc_lut.push(2.0 / (p as f64 + 1.0));
        }

        Ok(Self {
            min_period,
            max_period,
            matype,
            devtype,
            buffer: vec![f64::NAN; max_period],
            head: 0,
            filled: false,
            period: max_period as f64, // start at the slowest
            last_val: f64::NAN,

            // O(1) rolling statistics state
            sum: 0.0,
            sumsq: 0.0,
            nan_count: 0,
            inv_n: 1.0 / (max_period as f64),
            last_p: max_period,
            sc_lut,
        })
    }

    /// O(1) streaming update for (matype="sma", devtype=0). Falls back to O(n) build for others.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Ring buffer bookkeeping
        let out_idx = self.head;
        let v_out = self.buffer[out_idx];
        self.buffer[out_idx] = value;
        self.head = (self.head + 1) % self.max_period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }

        // Maintain rolling sums when window full
        if self.filled {
            if v_out.is_finite() {
                self.sum -= v_out;
                self.sumsq -= v_out * v_out;
            } else {
                self.nan_count = self.nan_count.saturating_sub(1);
            }
        }
        if value.is_finite() {
            self.sum += value;
            self.sumsq += value * value;
        } else {
            self.nan_count += 1;
        }

        // Fast path: SMA reference + standard deviation bands
        if self.matype == "sma" && self.devtype == 0 {
            // Warmup: seed on first finite value, then emit None until the first full window.
            if !self.filled {
                if self.last_val.is_nan() {
                    if value.is_finite() {
                        self.last_val = value;
                        return Some(value);
                    } else {
                        return None;
                    }
                }
                if value.is_finite() {
                    let sc = self.sc_lut[self.last_p];
                    self.last_val = fast_ema_update(self.last_val, value, sc);
                }
                return None;
            }

            // Steady-state: handle NaN input by emitting NaN and not changing state
            if !value.is_finite() {
                return Some(f64::NAN);
            }

            // Compute mean and stddev when no NaNs in the window
            let (m, dv) = if self.nan_count == 0 {
                let mean = self.sum * self.inv_n;
                let var = (self.sumsq * self.inv_n) - mean * mean;
                let std = if var <= 0.0 { 0.0 } else { var.sqrt() };
                (mean, std)
            } else {
                (f64::NAN, f64::NAN)
            };

            // Adapt period by ±1 within [min_period, max_period]
            let mut next_p = self.last_p;
            if m.is_finite() && dv.is_finite() {
                let d175 = dv * 1.75;
                let d025 = dv * 0.25;
                let a = m - d175;
                let b = m - d025;
                let c = m + d025;
                let d = m + d175;

                let inc_fast = ((value < a) as i32) | ((value > d) as i32);
                let inc_slow = ((value >= b) as i32) & ((value <= c) as i32);
                let delta = inc_slow - inc_fast; // -1, 0, +1
                let p_tmp = self.last_p as isize + delta as isize;
                next_p = if p_tmp < self.min_period as isize {
                    self.min_period
                } else if p_tmp > self.max_period as isize {
                    self.max_period
                } else {
                    p_tmp as usize
                };
            }

            // EMA-like smoothing with α from LUT
            let sc = self.sc_lut[next_p];
            self.last_val = fast_ema_update(self.last_val, value, sc);
            self.last_p = next_p;
            self.period = next_p as f64;
            return Some(self.last_val);
        }

        // --- Generic fallback (non-SMA or non-stddev): O(n) window rebuild ---
        let mut window: Vec<f64> = Vec::with_capacity(self.max_period);
        for i in 0..self.max_period {
            let idx = (self.head + i) % self.max_period;
            let v = self.buffer[idx];
            if v.is_finite() {
                window.push(v);
            }
        }
        if window.len() < self.max_period {
            // Not enough finite values yet for this path; preserve warmup EMA seeding
            if self.last_val.is_nan() && value.is_finite() {
                self.last_val = value;
                return Some(value);
            }
            if value.is_finite() {
                let sc = 2.0 / (self.period + 1.0);
                self.last_val = fast_ema_update(self.last_val, value, sc);
            }
            return None;
        }

        // Reference mean via selected MA and deviation
        let mean = match ma(&self.matype, MaData::Slice(&window), self.max_period) {
            Ok(v) => *v.last().unwrap_or(&f64::NAN),
            Err(_) => return None,
        };
        let dev_params = DevParams {
            period: Some(self.max_period),
            devtype: Some(self.devtype),
        };
        let dv = match deviation(&DevInput::from_slice(&window, dev_params)) {
            Ok(v) => *v.last().unwrap_or(&f64::NAN),
            Err(_) => return None,
        };

        if value.is_finite() {
            let prev = if self.period == 0.0 {
                self.max_period as f64
            } else {
                self.period
            };
            let mut new_p = prev;
            if mean.is_finite() && dv.is_finite() {
                let a = mean - 1.75 * dv;
                let b = mean - 0.25 * dv;
                let c = mean + 0.25 * dv;
                let d = mean + 1.75 * dv;
                if value < a || value > d {
                    new_p = (prev - 1.0).max(self.min_period as f64);
                } else if value >= b && value <= c {
                    new_p = (prev + 1.0).min(self.max_period as f64);
                }
            }
            let sc = 2.0 / (new_p + 1.0);
            if !self.last_val.is_nan() {
                self.last_val = fast_ema_update(self.last_val, value, sc);
            } else {
                self.last_val = value;
            }
            self.period = new_p;
            return Some(self.last_val);
        }

        Some(f64::NAN)
    }
}

#[derive(Clone, Debug)]
pub struct VlmaBatchRange {
    pub min_period: (usize, usize, usize),
    pub max_period: (usize, usize, usize),
    pub matype: (String, String, String),
    pub devtype: (usize, usize, usize),
}

impl Default for VlmaBatchRange {
    fn default() -> Self {
        Self {
            min_period: (5, 5, 0),
            max_period: (50, 50, 0),
            matype: ("sma".to_string(), "sma".to_string(), "".to_string()),
            devtype: (0, 0, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VlmaBatchBuilder {
    range: VlmaBatchRange,
    kernel: Kernel,
}

impl VlmaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn min_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.min_period = (start, end, step);
        self
    }
    #[inline]
    pub fn max_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.max_period = (start, end, step);
        self
    }
    pub fn matype_static<S: Into<String>>(mut self, v: S) -> Self {
        let s = v.into();
        self.range.matype = (s.clone(), s, "".to_string());
        self
    }
    #[inline]
    pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.devtype = (start, end, step);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<VlmaBatchOutput, VlmaError> {
        vlma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VlmaBatchOutput, VlmaError> {
        VlmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VlmaBatchOutput, VlmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VlmaBatchOutput, VlmaError> {
        VlmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct VlmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VlmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VlmaBatchOutput {
    pub fn row_for_params(&self, p: &VlmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.min_period.unwrap_or(5) == p.min_period.unwrap_or(5)
                && c.max_period.unwrap_or(50) == p.max_period.unwrap_or(50)
                && c.matype.as_ref().unwrap_or(&"sma".to_string())
                    == p.matype.as_ref().unwrap_or(&"sma".to_string())
                && c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
        })
    }
    pub fn values_for(&self, p: &VlmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}
fn axis_string((start, end, _): (String, String, String)) -> Vec<String> {
    if start == end {
        vec![start]
    } else {
        vec![start, end]
    }
}
fn axis_usize_step((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    }
}
fn axis_devtype((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    }
}

fn expand_grid(r: &VlmaBatchRange) -> Vec<VlmaParams> {
    let min_periods = axis_usize(r.min_period);
    let max_periods = axis_usize(r.max_period);
    let matypes = axis_string(r.matype.clone());
    let devtypes = axis_devtype(r.devtype);
    let mut out =
        Vec::with_capacity(min_periods.len() * max_periods.len() * matypes.len() * devtypes.len());
    for &mn in &min_periods {
        for &mx in &max_periods {
            for mt in &matypes {
                for &dt in &devtypes {
                    out.push(VlmaParams {
                        min_period: Some(mn),
                        max_period: Some(mx),
                        matype: Some(mt.clone()),
                        devtype: Some(dt),
                    });
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn vlma_batch_with_kernel(
    data: &[f64],
    sweep: &VlmaBatchRange,
    k: Kernel,
) -> Result<VlmaBatchOutput, VlmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VlmaError::InvalidPeriod {
                min_period: 0,
                max_period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vlma_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn vlma_batch_slice(
    data: &[f64],
    sweep: &VlmaBatchRange,
    kern: Kernel,
) -> Result<VlmaBatchOutput, VlmaError> {
    vlma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vlma_batch_par_slice(
    data: &[f64],
    sweep: &VlmaBatchRange,
    kern: Kernel,
) -> Result<VlmaBatchOutput, VlmaError> {
    vlma_batch_inner(data, sweep, kern, true)
}

fn vlma_batch_inner(
    data: &[f64],
    sweep: &VlmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VlmaBatchOutput, VlmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VlmaError::InvalidPeriod {
            min_period: 0,
            max_period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VlmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.max_period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(VlmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Allocate uninit matrix, then set warmup prefixes per-row.
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warms: Vec<usize> = combos
        .iter()
        .map(|c| first + c.max_period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    // Cast to &mut [f64] like ALMA
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // VLMA specific: Restore the initial value at first_valid for each row
    // (init_matrix_prefixes sets all warmup values to NaN, but VLMA needs first_valid preserved)
    for row in 0..rows {
        let row_start = row * cols;
        out[row_start + first] = data[first];
    }
    // Compute rows
    let simd_kern = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        },
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        k => k,
    };
    vlma_batch_inner_into(data, sweep, simd_kern, parallel, out)?;

    // Reclaim into Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VlmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn vlma_batch_inner_into(
    data: &[f64],
    sweep: &VlmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VlmaParams>, VlmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VlmaError::InvalidPeriod {
            min_period: 0,
            max_period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VlmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.max_period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(VlmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Detect if we can use the shared-prefix fast path (SMA + stddev rows present)
    let any_sma_std = combos
        .iter()
        .any(|c| c.matype.as_deref() == Some("sma") && c.devtype == Some(0));

    // Optional shared prefix sums for mean/stddev across rows
    let (ps_sum, ps_sumsq, ps_cnt);
    let ps_sum_ref: Option<&[f64]>;
    let ps_sumsq_ref: Option<&[f64]>;
    let ps_cnt_ref: Option<&[usize]>;
    if any_sma_std {
        // Build prefix sums once: ps[i+1] = sum of finite(data[0..=i])
        let mut sum = 0.0_f64;
        let mut sumsq = 0.0_f64;
        let mut cnt = 0_usize;
        let mut ps_s = Vec::with_capacity(cols + 1);
        let mut ps_q = Vec::with_capacity(cols + 1);
        let mut ps_c = Vec::with_capacity(cols + 1);
        ps_s.push(0.0);
        ps_q.push(0.0);
        ps_c.push(0);
        for &v in data.iter() {
            if v.is_finite() {
                sum += v;
                sumsq += v * v;
                cnt += 1;
            }
            ps_s.push(sum);
            ps_q.push(sumsq);
            ps_c.push(cnt);
        }
        ps_sum = ps_s;
        ps_sumsq = ps_q;
        ps_cnt = ps_c;
        ps_sum_ref = Some(&ps_sum);
        ps_sumsq_ref = Some(&ps_sumsq);
        ps_cnt_ref = Some(&ps_cnt);
    } else {
        ps_sum = Vec::new();
        ps_sumsq = Vec::new();
        ps_cnt = Vec::new();
        ps_sum_ref = None;
        ps_sumsq_ref = None;
        ps_cnt_ref = None;
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let min_period = combos[row].min_period.unwrap();
        let max_period = combos[row].max_period.unwrap();
        let matype = combos[row].matype.as_ref().unwrap();
        let devtype = combos[row].devtype.unwrap();
        match kern {
            Kernel::Scalar => {
                if matype == "sma" && devtype == 0 {
                    vlma_row_fast_sma_std_prefix(
                        data,
                        min_period,
                        max_period,
                        first,
                        ps_sum_ref.unwrap(),
                        ps_sumsq_ref.unwrap(),
                        ps_cnt_ref.unwrap(),
                        out_row,
                    )
                    .unwrap();
                } else {
                    vlma_row_scalar(
                        data, min_period, max_period, matype, devtype, first, out_row,
                    )
                    .unwrap();
                }
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                vlma_row_avx2(
                    data, min_period, max_period, matype, devtype, first, out_row,
                )
                .unwrap();
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                vlma_row_avx512(
                    data, min_period, max_period, matype, devtype, first, out_row,
                )
                .unwrap();
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                // Fall back to scalar when AVX is not available
                vlma_row_scalar(
                    data, min_period, max_period, matype, devtype, first, out_row,
                )
                .unwrap();
            }
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

    Ok(combos)
}

#[inline(always)]
pub fn expand_grid_vlma(r: &VlmaBatchRange) -> Vec<VlmaParams> {
    expand_grid(r)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vlma")]
#[pyo3(signature = (data, min_period=5, max_period=50, matype="sma", devtype=0, kernel=None))]
pub fn vlma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = VlmaParams {
        min_period: Some(min_period),
        max_period: Some(max_period),
        matype: Some(matype.to_string()),
        devtype: Some(devtype),
    };
    let input = VlmaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| vlma_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VlmaStream")]
pub struct VlmaStreamPy {
    stream: VlmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VlmaStreamPy {
    #[new]
    fn new(min_period: usize, max_period: usize, matype: &str, devtype: usize) -> PyResult<Self> {
        let params = VlmaParams {
            min_period: Some(min_period),
            max_period: Some(max_period),
            matype: Some(matype.to_string()),
            devtype: Some(devtype),
        };
        let stream =
            VlmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VlmaStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vlma_batch")]
#[pyo3(signature = (data, min_period_range=(5, 5, 0), max_period_range=(50, 50, 0), devtype_range=(0, 0, 0), matype="sma", kernel=None))]
pub fn vlma_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_period_range: (usize, usize, usize),
    max_period_range: (usize, usize, usize),
    devtype_range: (usize, usize, usize),
    matype: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;

    let sweep = VlmaBatchRange {
        min_period: min_period_range,
        max_period: max_period_range,
        matype: (matype.to_string(), matype.to_string(), "".to_string()),
        devtype: devtype_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Initialize warmup NaN prefixes for each row
    let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.max_period.unwrap() - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            if i != first {
                // Preserve initial value at first_valid
                slice_out[row_start + i] = f64::NAN;
            }
        }
        // VLMA specific: Set initial value at first_valid
        if first < cols {
            slice_out[row_start + first] = slice_in[first];
        }
    }

    let kern = validate_kernel(kernel, true)?;

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
            vlma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "min_periods",
        combos
            .iter()
            .map(|p| p.min_period.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "max_periods",
        combos
            .iter()
            .map(|p| p.max_period.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "devtypes",
        combos
            .iter()
            .map(|p| p.devtype.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "matypes",
        combos
            .iter()
            .map(|p| p.matype.as_ref().unwrap().clone())
            .collect::<Vec<_>>(),
    )?;

    Ok(dict)
}

// ------------------------ Python CUDA bindings ------------------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vlma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, min_period_range=(5, 5, 0), max_period_range=(50, 50, 0), devtype_range=(0, 0, 0), matype="sma", device_id=0))]
pub fn vlma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    min_period_range: (usize, usize, usize),
    max_period_range: (usize, usize, usize),
    devtype_range: (usize, usize, usize),
    matype: &str,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = VlmaBatchRange {
        min_period: min_period_range,
        max_period: max_period_range,
        matype: (matype.to_string(), matype.to_string(), "".to_string()),
        devtype: devtype_range,
    };

    let inner = py.allow_threads(|| {
        let mut cuda = CudaVlma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vlma_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vlma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, min_period, max_period, devtype=0, matype="sma", device_id=0))]
pub fn vlma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    min_period: usize,
    max_period: usize,
    devtype: usize,
    matype: &str,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let flat: &[f32] = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = VlmaParams {
        min_period: Some(min_period),
        max_period: Some(max_period),
        matype: Some(matype.to_string()),
        devtype: Some(devtype),
    };
    let inner = py.allow_threads(|| {
        let mut cuda = CudaVlma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vlma_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

// WASM bindings following ALMA pattern

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_js(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = VlmaParams {
        min_period: Some(min_period),
        max_period: Some(max_period),
        matype: Some(matype.to_string()),
        devtype: Some(devtype),
    };
    let input = VlmaInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    vlma_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    min_period: usize,
    max_period: usize,
    matype: &str,
    devtype: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = VlmaParams {
            min_period: Some(min_period),
            max_period: Some(max_period),
            matype: Some(matype.to_string()),
            devtype: Some(devtype),
        };
        let input = VlmaInput::from_slice(data, params);

        if in_ptr == out_ptr as *const f64 {
            let mut temp = vec![0.0; len];
            vlma_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vlma_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VlmaBatchConfig {
    pub min_period_range: (usize, usize, usize),
    pub max_period_range: (usize, usize, usize),
    pub devtype_range: (usize, usize, usize),
    pub matype: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VlmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VlmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vlma_batch)]
pub fn vlma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: VlmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = VlmaBatchRange {
        min_period: config.min_period_range,
        max_period: config.max_period_range,
        matype: (config.matype.clone(), config.matype.clone(), "".to_string()),
        devtype: config.devtype_range,
    };

    let output = vlma_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = VlmaBatchJsOutput {
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
pub fn vlma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    min_period_start: usize,
    min_period_end: usize,
    min_period_step: usize,
    max_period_start: usize,
    max_period_end: usize,
    max_period_step: usize,
    devtype_start: usize,
    devtype_end: usize,
    devtype_step: usize,
    matype: &str,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to vlma_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = VlmaBatchRange {
            min_period: (min_period_start, min_period_end, min_period_step),
            max_period: (max_period_start, max_period_end, max_period_step),
            matype: (matype.to_string(), matype.to_string(), "".to_string()),
            devtype: (devtype_start, devtype_end, devtype_step),
        };

        let combos = expand_grid(&sweep);
        let total_len = combos.len() * len;
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, total_len);

        let _ = vlma_batch_inner_into(data, &sweep, Kernel::Scalar, false, out_slice)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(combos.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    fn check_vlma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VlmaParams {
            min_period: None,
            max_period: None,
            matype: None,
            devtype: None,
        };
        let input_default = VlmaInput::from_candles(&candles, "close", default_params);
        let output_default = vlma_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }
    fn check_vlma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close")?;
        let params = VlmaParams {
            min_period: Some(5),
            max_period: Some(50),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_candles(&candles, "close", params);
        let vlma_result = vlma_with_kernel(&input, kernel)?;
        assert_eq!(vlma_result.values.len(), close_prices.len());
        let required_len = 5;
        assert!(
            vlma_result.values.len() >= required_len,
            "VLMA length is too short"
        );
        let test_vals = [
            59376.252799490234,
            59343.71066624187,
            59292.92555520155,
            59269.93796266796,
            59167.4483022233,
        ];
        let start_idx = vlma_result.values.len() - test_vals.len();
        let actual_slice = &vlma_result.values[start_idx..];
        for (i, &val) in actual_slice.iter().enumerate() {
            let expected = test_vals[i];
            if !val.is_nan() {
                assert!(
                    (val - expected).abs() < 1e-1,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    val
                );
            }
        }
        Ok(())
    }
    fn check_vlma_zero_or_inverted_periods(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let input_data = [10.0, 20.0, 30.0, 40.0];
        let params_min_greater = VlmaParams {
            min_period: Some(10),
            max_period: Some(5),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input_min_greater = VlmaInput::from_slice(&input_data, params_min_greater);
        let result = vlma_with_kernel(&input_min_greater, kernel);
        assert!(result.is_err());
        let params_zero_max = VlmaParams {
            min_period: Some(5),
            max_period: Some(0),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input_zero_max = VlmaInput::from_slice(&input_data, params_zero_max);
        let result2 = vlma_with_kernel(&input_zero_max, kernel);
        assert!(result2.is_err());
        Ok(())
    }
    fn check_vlma_not_enough_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let input_data = [10.0, 20.0, 30.0];
        let params = VlmaParams {
            min_period: Some(5),
            max_period: Some(10),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_slice(&input_data, params);
        let result = vlma_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }
    fn check_vlma_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = VlmaParams {
            min_period: Some(2),
            max_period: Some(3),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_slice(&input_data, params);
        let result = vlma_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }
    fn check_vlma_slice_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = VlmaParams {
            min_period: Some(5),
            max_period: Some(20),
            matype: Some("ema".to_string()),
            devtype: Some(1),
        };
        let first_input = VlmaInput::from_candles(&candles, "close", first_params);
        let first_result = vlma_with_kernel(&first_input, kernel)?;
        let second_params = VlmaParams {
            min_period: Some(5),
            max_period: Some(20),
            matype: Some("ema".to_string()),
            devtype: Some(1),
        };
        let second_input = VlmaInput::from_slice(&first_result.values, second_params);
        let second_result = vlma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_vlma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = VlmaParams {
            min_period: Some(5),
            max_period: Some(50),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_candles(&candles, "close", params.clone());
        let batch_output = vlma_with_kernel(&input, kernel)?.values;
        let mut stream = VlmaStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(v) => stream_values.push(v),
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
                "[{}] VLMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vlma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test parameters
        let strat = (2usize..=20, 0.001f64..1e6f64).prop_flat_map(|(min_period, scalar)| {
            // max_period must be > min_period
            let max_period_start = min_period + 1;
            (
                prop::collection::vec(
                    (0.001f64..1e6f64)
                        .prop_filter("positive finite", |x| x.is_finite() && *x > 0.0),
                    max_period_start..400,
                ),
                Just(min_period),
                (max_period_start..=50),
                prop::sample::select(vec!["sma", "ema", "wma"]),
                (0usize..=2), // devtype: 0=std, 1=mad, 2=median
                Just(scalar),
            )
        });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, min_period, max_period, matype, devtype, scalar)| {
				// Ensure max_period is valid for the data length
				if max_period > data.len() {
					return Ok(());
				}

				let params = VlmaParams {
					min_period: Some(min_period),
					max_period: Some(max_period),
					matype: Some(matype.to_string()),
					devtype: Some(devtype),
				};
				let input = VlmaInput::from_slice(&data, params.clone());

				// Test with specified kernel
				let VlmaOutput { values: out } = vlma_with_kernel(&input, kernel).unwrap();

				// Also compute with scalar kernel for comparison
				let VlmaOutput { values: ref_out } = vlma_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Warmup period validation
				// VLMA sets an initial value at first_valid, then NaN until max_period - 1
				let first_valid = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);
				let expected_warmup = first_valid + max_period - 1;

				// Check that first_valid has a value (if it exists)
				if first_valid < out.len() {
					prop_assert!(
						!out[first_valid].is_nan(),
						"Expected initial value at first_valid index {}, got NaN",
						first_valid
					);

					// Property 1b: Initial value should equal first data point
					prop_assert!(
						(out[first_valid] - data[first_valid]).abs() < 1e-9,
						"Initial VLMA value {} should equal first data point {} at index {}",
						out[first_valid],
						data[first_valid],
						first_valid
					);
				}

				// Check NaN values between first_valid+1 and expected_warmup
				for i in (first_valid + 1)..expected_warmup.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Property 2: Regular values should start at expected_warmup
				if expected_warmup < out.len() {
					prop_assert!(
						!out[expected_warmup].is_nan(),
						"Expected valid value at warmup end (index {}), got NaN",
						expected_warmup
					);
				}

				// Property 3: Output bounds - VLMA should be within exact data range
				let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
				let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

				for (i, &val) in out.iter().enumerate() {
					if !val.is_nan() && i != first_valid { // Skip first_valid as it equals the data point
						prop_assert!(
							val >= data_min - 1e-9 && val <= data_max + 1e-9,
							"VLMA at index {} = {} is outside data range [{}, {}]",
							i,
							val,
							data_min,
							data_max
						);
					}
				}

				// Property 4: Convergence for constant data
				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
					// For constant data, VLMA should converge to that constant
					for (i, &val) in out.iter().enumerate() {
						if !val.is_nan() && i >= expected_warmup + 10 {
							prop_assert!(
								(val - data[0]).abs() < 1e-6,
								"VLMA should converge to constant value {} but got {} at index {}",
								data[0],
								val,
								i
							);
						}
					}
				}

				// Property 5: Smoothness - output variance should be <= input variance in stable regions
				if data.len() >= max_period * 2 {
					let stable_start = expected_warmup + max_period;
					let stable_end = data.len();

					if stable_start < stable_end {
						let input_segment = &data[stable_start..stable_end];
						let output_segment = &out[stable_start..stable_end];

						// Calculate variance for both
						let input_mean: f64 = input_segment.iter().sum::<f64>() / input_segment.len() as f64;
						let input_var: f64 = input_segment.iter()
							.map(|x| (x - input_mean).powi(2))
							.sum::<f64>() / input_segment.len() as f64;

						let valid_outputs: Vec<f64> = output_segment.iter()
							.filter(|x| !x.is_nan())
							.cloned()
							.collect();

						if valid_outputs.len() > 1 {
							let output_mean: f64 = valid_outputs.iter().sum::<f64>() / valid_outputs.len() as f64;
							let output_var: f64 = valid_outputs.iter()
								.map(|x| (x - output_mean).powi(2))
								.sum::<f64>() / valid_outputs.len() as f64;

							// VLMA should smooth the data, so variance should be less than or equal to input
							prop_assert!(
								output_var <= input_var * 1.01 || output_var < 1e-10,
								"Output variance {} should not exceed input variance {} (smoothing property)",
								output_var,
								input_var
							);
						}
					}
				}

				// Property 5b: Adaptive period behavior test
				// For volatile data, VLMA should show adaptive behavior
				if data.len() >= max_period * 3 {
					// Calculate volatility in two different regions
					let mid_point = data.len() / 2;
					let region1_start = expected_warmup + max_period;

					// Ensure mid_point is after region1_start to avoid underflow
					if mid_point > region1_start && data.len() > mid_point + max_period {
						let region1_end = region1_start + max_period.min((mid_point - region1_start) / 2);
						let region2_start = mid_point + max_period;
						let region2_end = region2_start + max_period.min((data.len() - region2_start) / 2);

						if region1_end > region1_start && region2_end > region2_start {
							// Calculate standard deviation for each region
							let calc_std = |segment: &[f64]| -> f64 {
								let mean = segment.iter().sum::<f64>() / segment.len() as f64;
								let variance = segment.iter()
									.map(|x| (x - mean).powi(2))
									.sum::<f64>() / segment.len() as f64;
								variance.sqrt()
							};

							let region1_data = &data[region1_start..region1_end.min(data.len())];
							let region2_data = &data[region2_start..region2_end.min(data.len())];

							if region1_data.len() > 2 && region2_data.len() > 2 {
								let std1 = calc_std(region1_data);
								let std2 = calc_std(region2_data);

								// If one region is significantly more volatile than the other
								if (std1 > std2 * 2.0 || std2 > std1 * 2.0) && std1 > 1e-6 && std2 > 1e-6 {
									// VLMA outputs should show some difference between regions
									let out1: Vec<f64> = out[region1_start..region1_end.min(out.len())]
										.iter()
										.filter(|x| !x.is_nan())
										.cloned()
										.collect();
									let out2: Vec<f64> = out[region2_start..region2_end.min(out.len())]
										.iter()
										.filter(|x| !x.is_nan())
										.cloned()
										.collect();

									if out1.len() > 2 && out2.len() > 2 {
										let out_std1 = calc_std(&out1);
										let out_std2 = calc_std(&out2);

										// The region with higher volatility should have different characteristics
										prop_assert!(
											(out_std1 - out_std2).abs() > 1e-10 || (std1 - std2).abs() < 1e-6,
											"VLMA should show adaptive behavior: region1 std={}, region2 std={}, but outputs are too similar",
											std1,
											std2
										);
									}
								}
							}
						}
					}
				}

				// Property 6: Kernel consistency
				for i in expected_warmup..out.len().min(ref_out.len()) {
					let y = out[i];
					let r = ref_out[i];

					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"NaN/Inf mismatch at index {}: {} vs {}",
							i,
							y,
							r
						);
						continue;
					}

					// Check ULP difference for floating point comparison
					let y_bits = y.to_bits();
					let r_bits = r.to_bits();
					let ulp_diff: u64 = y_bits.abs_diff(r_bits);

					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"Kernel mismatch at index {}: {} vs {} (ULP={})",
						i,
						y,
						r,
						ulp_diff
					);
				}

				// Property 7: No poison values (in debug builds)
				#[cfg(debug_assertions)]
				for (i, &val) in out.iter().enumerate() {
					if !val.is_nan() {
						let bits = val.to_bits();
						prop_assert!(
							bits != 0x11111111_11111111 &&
							bits != 0x22222222_22222222 &&
							bits != 0x33333333_33333333,
							"Found poison value {} (0x{:016X}) at index {}",
							val,
							bits,
							i
						);
					}
				}

				// Property 8: Monotonicity trend preservation
				// If data is strictly increasing/decreasing, VLMA should follow the trend
				let is_increasing = data.windows(2).all(|w| w[1] >= w[0]);
				let is_decreasing = data.windows(2).all(|w| w[1] <= w[0]);

				if is_increasing || is_decreasing {
					let valid_outputs: Vec<(usize, f64)> = out.iter()
						.enumerate()
						.filter(|(_, x)| !x.is_nan())
						.map(|(i, &x)| (i, x))
						.collect();

					if valid_outputs.len() >= 10 {
						// Check last 5 values follow the trend
						let last_5 = &valid_outputs[valid_outputs.len() - 5..];
						if is_increasing {
							for w in last_5.windows(2) {
								prop_assert!(
									w[1].1 >= w[0].1 * 0.999, // Tighter tolerance of 0.1%
									"VLMA should be non-decreasing for increasing data at indices {}-{}: {} > {}",
									w[0].0,
									w[1].0,
									w[0].1,
									w[1].1
								);
							}
						} else if is_decreasing {
							for w in last_5.windows(2) {
								prop_assert!(
									w[1].1 <= w[0].1 * 1.001, // Tighter tolerance of 0.1%
									"VLMA should be non-increasing for decreasing data at indices {}-{}: {} < {}",
									w[0].0,
									w[1].0,
									w[0].1,
									w[1].1
								);
							}
						}
					}
				}

				// Property 9: Determinism - same input produces same output
				let input2 = VlmaInput::from_slice(&data, params);
				let VlmaOutput { values: out2 } = vlma_with_kernel(&input2, kernel).unwrap();

				for i in 0..out.len().min(out2.len()) {
					if out[i].is_finite() && out2[i].is_finite() {
						prop_assert!(
							(out[i] - out2[i]).abs() < f64::EPSILON,
							"Non-deterministic output at index {}: {} vs {}",
							i,
							out[i],
							out2[i]
						);
					} else {
						prop_assert!(
							out[i].to_bits() == out2[i].to_bits(),
							"Non-deterministic NaN/Inf at index {}: {:016X} vs {:016X}",
							i,
							out[i].to_bits(),
							out2[i].to_bits()
						);
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_vlma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            VlmaParams::default(),
            // Minimum periods
            VlmaParams {
                min_period: Some(1),
                max_period: Some(2),
                matype: Some("sma".to_string()),
                devtype: Some(0),
            },
            // Small periods
            VlmaParams {
                min_period: Some(2),
                max_period: Some(10),
                matype: Some("sma".to_string()),
                devtype: Some(0),
            },
            // Medium periods with EMA
            VlmaParams {
                min_period: Some(10),
                max_period: Some(30),
                matype: Some("ema".to_string()),
                devtype: Some(0),
            },
            // Large periods
            VlmaParams {
                min_period: Some(20),
                max_period: Some(100),
                matype: Some("sma".to_string()),
                devtype: Some(0),
            },
            // Very large periods with WMA
            VlmaParams {
                min_period: Some(50),
                max_period: Some(200),
                matype: Some("wma".to_string()),
                devtype: Some(0),
            },
            // Different deviation type (MAD)
            VlmaParams {
                min_period: Some(5),
                max_period: Some(25),
                matype: Some("sma".to_string()),
                devtype: Some(1),
            },
            // Different deviation type (Median)
            VlmaParams {
                min_period: Some(5),
                max_period: Some(25),
                matype: Some("ema".to_string()),
                devtype: Some(2),
            },
            // Edge case: min close to max
            VlmaParams {
                min_period: Some(19),
                max_period: Some(20),
                matype: Some("sma".to_string()),
                devtype: Some(0),
            },
            // Another combination
            VlmaParams {
                min_period: Some(3),
                max_period: Some(15),
                matype: Some("wma".to_string()),
                devtype: Some(1),
            },
            // Large range
            VlmaParams {
                min_period: Some(5),
                max_period: Some(100),
                matype: Some("ema".to_string()),
                devtype: Some(2),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = VlmaInput::from_candles(&candles, "close", params.clone());
            let output = vlma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: min_period={}, max_period={}, matype={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.min_period.unwrap_or(5),
                        params.max_period.unwrap_or(50),
                        params.matype.as_deref().unwrap_or("sma"),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: min_period={}, max_period={}, matype={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.min_period.unwrap_or(5),
                        params.max_period.unwrap_or(50),
                        params.matype.as_deref().unwrap_or("sma"),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: min_period={}, max_period={}, matype={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.min_period.unwrap_or(5),
                        params.max_period.unwrap_or(50),
                        params.matype.as_deref().unwrap_or("sma"),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vlma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_vlma_tests {
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
    generate_all_vlma_tests!(
        check_vlma_partial_params,
        check_vlma_accuracy,
        check_vlma_zero_or_inverted_periods,
        check_vlma_not_enough_data,
        check_vlma_all_nan,
        check_vlma_slice_reinput,
        check_vlma_streaming,
        check_vlma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vlma_tests!(check_vlma_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = VlmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = VlmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59376.252799490234,
            59343.71066624187,
            59292.92555520155,
            59269.93796266796,
            59167.4483022233,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (min_start, min_end, min_step, max_start, max_end, max_step, matype, dev_start, dev_end, dev_step)
            (2, 10, 2, 10, 20, 2, "sma", 0, 0, 0), // Small periods
            (5, 25, 5, 25, 50, 5, "sma", 0, 2, 1), // Medium periods with devtype sweep
            (10, 50, 10, 50, 100, 10, "ema", 0, 0, 0), // Large periods with EMA
            (1, 5, 1, 5, 10, 1, "sma", 0, 0, 0),   // Dense small range
            (5, 5, 0, 20, 100, 20, "wma", 0, 2, 2), // Static min, sweep max
            (2, 10, 4, 20, 20, 0, "sma", 1, 1, 0), // Sweep min, static max, MAD
            (3, 15, 3, 15, 30, 3, "ema", 2, 2, 0), // Median deviation
            (20, 50, 15, 60, 150, 30, "sma", 0, 2, 1), // Large ranges
            (5, 5, 0, 50, 50, 0, "sma", 0, 2, 1),  // Default with devtype sweep
        ];

        for (
            cfg_idx,
            &(
                min_start,
                min_end,
                min_step,
                max_start,
                max_end,
                max_step,
                matype,
                dev_start,
                dev_end,
                dev_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let mut builder = VlmaBatchBuilder::new().kernel(kernel);

            // Configure ranges
            if min_step > 0 {
                builder = builder.min_period_range(min_start, min_end, min_step);
            } else {
                builder = builder.min_period_range(min_start, min_start, 0);
            }

            if max_step > 0 {
                builder = builder.max_period_range(max_start, max_end, max_step);
            } else {
                builder = builder.max_period_range(max_start, max_start, 0);
            }

            builder = builder.matype_static(matype);

            if dev_step > 0 {
                builder = builder.devtype_range(dev_start, dev_end, dev_step);
            } else {
                builder = builder.devtype_range(dev_start, dev_start, 0);
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
						 at row {} col {} (flat index {}) with params: \
						 min_period={}, max_period={}, matype={}, devtype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.min_period.unwrap_or(5),
                        combo.max_period.unwrap_or(50),
                        combo.matype.as_deref().unwrap_or("sma"),
                        combo.devtype.unwrap_or(0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: \
						 min_period={}, max_period={}, matype={}, devtype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.min_period.unwrap_or(5),
                        combo.max_period.unwrap_or(50),
                        combo.matype.as_deref().unwrap_or("sma"),
                        combo.devtype.unwrap_or(0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: \
						 min_period={}, max_period={}, matype={}, devtype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.min_period.unwrap_or(5),
                        combo.max_period.unwrap_or(50),
                        combo.matype.as_deref().unwrap_or("sma"),
                        combo.devtype.unwrap_or(0)
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
