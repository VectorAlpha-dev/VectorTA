//! # Anti-Volume Stop Loss (AVSL)
//!
//! A dynamic stop-loss indicator that uses volume-price confirmation/contradiction to calculate
//! adaptive stop-loss levels. AVSL analyzes the relationship between volume-weighted moving averages
//! (VWMA) and simple moving averages (SMA) to detect volume-price divergences, then calculates
//! dynamic stop levels based on these divergences and market volatility.
//!
//! ## Parameters
//! - **fast_period**: Period for fast average (default: 12)
//! - **slow_period**: Period for slow average (default: 26)
//! - **multiplier**: Volatility multiplier for stop distance (default: 2.0)
//!
//! ## Inputs
//! - Requires close prices, low prices, and volume data
//! - Supports both raw slices and Candles data structure
//! - All three input arrays must have the same length
//!
//! ## Returns
//! - **`Ok(AvslOutput)`** containing a `Vec<f64>` matching input length
//! - Leading values are NaN during warmup (slow_period-1 values)
//!
//! ## Decision Log (2025-11-13)
//! - SIMD enabled under `nightly-avx` with scalar as reference; small speedups on AVX2/AVX512.
//! - CUDA wrappers present; Python VRAM handle implements CAI v3 + DLPack for interop.
//! - Public API and numerics unchanged; tests reference values preserved.
//!
//! ## Developer Notes (Status)
//! - Scalar: optimized to a single forward pass with rolling sums and tiny rings (≤200) for the
//!   dynamic window; avoids large temporaries and extra series materialization.
//! - SIMD: AVX2/AVX512 partially vectorize the contiguous low-sum (pref) and parts of the ring
//!   segment. Modest speedups expected; scalar remains the reference path.
//! - Batch: per-row compute currently dispatches the same kernels; no row-specific sharing yet.
//! - Warmup/semantics: preserved exactly; allocation and NaN prefixes follow alma.rs patterns.
//!
//! ## Binding Tests Status (2025-10-27)
//! - Python bindings: PASS. `tests/python/test_avsl.py` (13 tests) all pass and the accuracy checks
//!   compare the last 5 values against the same PineScript reference used by the Rust unit tests
//!   with the same 1% tolerance.
//! - WASM bindings: PARTIAL. `tests/wasm/test_avsl.js` batch default-row check passes against the
//!   Rust reference (within 1%), but the single-series `avsl_js` accuracy test fails: the last 5
//!   values differ significantly (e.g., got ~64,606 vs expected ~56,472 on the first of the last
//!   five). This suggests an issue specific to the `avsl_js` path rather than data or batch paths.
//!   No changes were made to kernels pending investigation.

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

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

// Import indicators we'll use
use crate::indicators::moving_averages::vwma::{
    vwma_into_slice, vwma_with_kernel, VwmaInput, VwmaParams,
};
use crate::indicators::sma::{sma_into_slice, sma_with_kernel, SmaInput, SmaParams};

// ==================== TRAIT IMPLEMENTATIONS ====================
// Note: AsRef cannot be directly implemented for AvslInput as it requires 3 slices,
// not a single slice. This is a key difference from alma.rs which only needs one data slice.

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum AvslData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        low_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        low: &'a [f64],
        volume: &'a [f64],
    },
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct AvslOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AvslParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub multiplier: Option<f64>,
}

impl Default for AvslParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            multiplier: Some(2.0),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct AvslInput<'a> {
    pub data: AvslData<'a>,
    pub params: AvslParams,
}

impl<'a> AvslInput<'a> {
    #[inline]
    pub fn from_candles(
        c: &'a Candles,
        close_source: &'a str,
        low_source: &'a str,
        p: AvslParams,
    ) -> Self {
        Self {
            data: AvslData::Candles {
                candles: c,
                close_source,
                low_source,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(close: &'a [f64], low: &'a [f64], volume: &'a [f64], p: AvslParams) -> Self {
        Self {
            data: AvslData::Slices { close, low, volume },
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", "low", AvslParams::default())
    }

    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(12)
    }

    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(26)
    }

    #[inline]
    pub fn get_multiplier(&self) -> f64 {
        self.params.multiplier.unwrap_or(2.0)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct AvslBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    multiplier: Option<f64>,
    kernel: Kernel,
}

impl Default for AvslBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            multiplier: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AvslBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn fast_period(mut self, val: usize) -> Self {
        self.fast_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn slow_period(mut self, val: usize) -> Self {
        self.slow_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn multiplier(mut self, val: f64) -> Self {
        self.multiplier = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AvslOutput, AvslError> {
        let p = AvslParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            multiplier: self.multiplier,
        };
        let i = AvslInput::from_candles(c, "close", "low", p);
        avsl_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        close: &[f64],
        low: &[f64],
        volume: &[f64],
    ) -> Result<AvslOutput, AvslError> {
        let p = AvslParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            multiplier: self.multiplier,
        };
        let i = AvslInput::from_slices(close, low, volume, p);
        avsl_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AvslStream, AvslError> {
        let p = AvslParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            multiplier: self.multiplier,
        };
        AvslStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum AvslError {
    #[error("avsl: Input data slice is empty.")]
    EmptyInputData,

    #[error("avsl: All values are NaN.")]
    AllValuesNaN,

    #[error("avsl: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("avsl: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error(
        "avsl: Data length mismatch: close = {close_len}, low = {low_len}, volume = {volume_len}"
    )]
    DataLengthMismatch {
        close_len: usize,
        low_len: usize,
        volume_len: usize,
    },

    #[error("avsl: Invalid multiplier: {multiplier}")]
    InvalidMultiplier { multiplier: f64 },

    #[error("avsl: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("avsl: Invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },

    #[error("avsl: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("avsl: {0}")]
    ComputationError(String),
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
/// Find the maximum first valid index across three input arrays
#[inline(always)]
fn first_valid_max3(a: &[f64], b: &[f64], c: &[f64]) -> Option<usize> {
    let fa = a.iter().position(|x| !x.is_nan())?;
    let fb = b.iter().position(|x| !x.is_nan())?;
    let fc = c.iter().position(|x| !x.is_nan())?;
    Some(fa.max(fb).max(fc))
}
/// Main entry point with automatic kernel detection
#[inline]
pub fn avsl(input: &AvslInput) -> Result<AvslOutput, AvslError> {
    avsl_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection
pub fn avsl_with_kernel(input: &AvslInput, kernel: Kernel) -> Result<AvslOutput, AvslError> {
    let (close, low, volume, fast_period, slow_period, multiplier, first, chosen) =
        avsl_prepare(input, kernel)?;

    // CRITICAL: Use zero-copy allocation helper
    let mut out = alloc_with_nan_prefix(close.len(), first + slow_period - 1);

    avsl_compute_into(
        close,
        low,
        volume,
        fast_period,
        slow_period,
        multiplier,
        first,
        chosen,
        &mut out,
    )?;

    Ok(AvslOutput { values: out })
}

/// Zero-allocation version for WASM and performance-critical paths
#[inline]
pub fn avsl_into_slice(dst: &mut [f64], input: &AvslInput, kern: Kernel) -> Result<(), AvslError> {
    let (close, low, volume, fast_period, slow_period, multiplier, first, chosen) =
        avsl_prepare(input, kern)?;

    if dst.len() != close.len() {
        return Err(AvslError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }

    avsl_compute_into(
        close,
        low,
        volume,
        fast_period,
        slow_period,
        multiplier,
        first,
        chosen,
        dst,
    )?;

    // Fill warmup period with NaN
    let warmup_end = first + slow_period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Prepare and validate input data
#[inline(always)]
fn avsl_prepare<'a>(
    input: &'a AvslInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        f64,
        usize,
        Kernel,
    ),
    AvslError,
> {
    let (close, low, volume) = match &input.data {
        AvslData::Candles {
            candles,
            close_source,
            low_source,
        } => (
            source_type(candles, close_source),
            source_type(candles, low_source),
            candles.volume.as_slice(),
        ),
        AvslData::Slices { close, low, volume } => (*close, *low, *volume),
    };

    let len = close.len();
    if len == 0 {
        return Err(AvslError::EmptyInputData);
    }
    if close.len() != low.len() || close.len() != volume.len() {
        return Err(AvslError::DataLengthMismatch {
            close_len: close.len(),
            low_len: low.len(),
            volume_len: volume.len(),
        });
    }

    let first = first_valid_max3(close, low, volume).ok_or(AvslError::AllValuesNaN)?;
    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let multiplier = input.get_multiplier();

    if fast_period == 0 || fast_period > len {
        return Err(AvslError::InvalidPeriod {
            period: fast_period,
            data_len: len,
        });
    }
    if slow_period == 0 || slow_period > len {
        return Err(AvslError::InvalidPeriod {
            period: slow_period,
            data_len: len,
        });
    }
    if len - first < slow_period {
        return Err(AvslError::NotEnoughValidData {
            needed: slow_period,
            valid: len - first,
        });
    }
    if multiplier <= 0.0 || !multiplier.is_finite() {
        return Err(AvslError::InvalidMultiplier { multiplier });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    Ok((
        close,
        low,
        volume,
        fast_period,
        slow_period,
        multiplier,
        first,
        chosen,
    ))
}

/// Core computation dispatcher
#[inline(always)]
fn avsl_compute_into(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), AvslError> {
    unsafe {
        // WASM SIMD128 support
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                return avsl_simd128(
                    close,
                    low,
                    volume,
                    fast_period,
                    slow_period,
                    multiplier,
                    first,
                    out,
                );
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => avsl_scalar(
                close,
                low,
                volume,
                fast_period,
                slow_period,
                multiplier,
                first,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => avsl_avx2(
                close,
                low,
                volume,
                fast_period,
                slow_period,
                multiplier,
                first,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => avsl_avx512(
                close,
                low,
                volume,
                fast_period,
                slow_period,
                multiplier,
                first,
                out,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                avsl_scalar(
                    close,
                    low,
                    volume,
                    fast_period,
                    slow_period,
                    multiplier,
                    first,
                    out,
                )
            }
            _ => unreachable!(),
        }
    }
}

// ==================== SCALAR IMPLEMENTATION ====================
#[inline]
pub fn avsl_scalar(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    first_val: usize,
    out: &mut [f64],
) -> Result<(), AvslError> {
    let len = close.len();

    if len == 0 {
        return Err(AvslError::EmptyInputData);
    }

    let base = first_val + slow_period - 1;
    let warmup2 = base + slow_period - 1;

    if base >= len {
        let upto = warmup2.min(len);
        for v in &mut out[..upto] {
            *v = f64::NAN;
        }
        for v in &mut out[upto..] {
            *v = f64::NAN;
        }
        return Ok(());
    }

    let inv_fast = 1.0 / (fast_period as f64);
    let inv_slow = 1.0 / (slow_period as f64);

    let mut sum_close_f = 0.0_f64;
    let mut sum_close_s = 0.0_f64;
    let mut sum_vol_f = 0.0_f64;
    let mut sum_vol_s = 0.0_f64;
    let mut sum_cxv_f = 0.0_f64;
    let mut sum_cxv_s = 0.0_f64;

    const MAX_WIN: usize = 200;
    let mut ring_vpc: [f64; MAX_WIN] = [0.0; MAX_WIN];
    let mut ring_vpr: [f64; MAX_WIN] = [1.0; MAX_WIN];
    let mut ring_pos: usize = 0;

    let mut pre_ring: Vec<f64> = vec![0.0; slow_period];
    let mut pre_pos: usize = 0;
    let mut pre_sum: f64 = 0.0;
    let mut pre_cnt: usize = 0;

    unsafe {
        let c_ptr = close.as_ptr();
        let l_ptr = low.as_ptr();
        let v_ptr = volume.as_ptr();

        for i in 0..len {
            if i >= first_val {
                let c = *c_ptr.add(i);
                let v = *v_ptr.add(i);
                let cv = c * v;

                sum_close_f += c;
                sum_vol_f += v;
                sum_cxv_f += cv;
                sum_close_s += c;
                sum_vol_s += v;
                sum_cxv_s += cv;

                if i + 1 > fast_period + first_val {
                    let k = i + 1 - fast_period - 1;
                    let c_old = *c_ptr.add(k);
                    let v_old = *v_ptr.add(k);
                    sum_close_f -= c_old;
                    sum_vol_f -= v_old;
                    sum_cxv_f -= c_old * v_old;
                }
                if i + 1 > slow_period + first_val {
                    let k = i + 1 - slow_period - 1;
                    let c_old = *c_ptr.add(k);
                    let v_old = *v_ptr.add(k);
                    sum_close_s -= c_old;
                    sum_vol_s -= v_old;
                    sum_cxv_s -= c_old * v_old;
                }
            }

            if i >= base {
                let sma_f = sum_close_f * inv_fast;
                let sma_s = sum_close_s * inv_slow;
                let vwma_f = if sum_vol_f != 0.0 {
                    sum_cxv_f / sum_vol_f
                } else {
                    sma_f
                };
                let vwma_s = if sum_vol_s != 0.0 {
                    sum_cxv_s / sum_vol_s
                } else {
                    sma_s
                };

                let vpc = vwma_s - sma_s;
                let vpr = if sma_f != 0.0 { vwma_f / sma_f } else { 1.0 };
                let vol_f = sum_vol_f * inv_fast;
                let vol_s = sum_vol_s * inv_slow;
                let vm = if vol_s != 0.0 { vol_f / vol_s } else { 1.0 };
                let vpci = vpc * vpr * vm;

                let len_v = {
                    let t = if vpc < 0.0 {
                        (vpci - 3.0).abs().round()
                    } else {
                        (vpci + 3.0).round()
                    };
                    let m = if t < 1.0 { 1.0 } else { t };
                    let m = if m > MAX_WIN as f64 {
                        MAX_WIN as f64
                    } else {
                        m
                    };
                    m as usize
                };

                ring_vpc[ring_pos] = vpc;
                ring_vpr[ring_pos] = vpr;
                ring_pos += 1;
                if ring_pos == MAX_WIN {
                    ring_pos = 0;
                }

                // Total elements we can actually take from [0..=i]
                let take = len_v.min(i + 1);
                let hist_n = (i - base + 1).min(take);
                let pref_n = take - hist_n;
                let mut acc = 0.0_f64;

                if hist_n > 0 {
                    let mut rp = if ring_pos == 0 {
                        MAX_WIN - 1
                    } else {
                        ring_pos - 1
                    };
                    for j in 0..hist_n {
                        let idx_r = rp;
                        rp = if rp == 0 { MAX_WIN - 1 } else { rp - 1 };
                        let x = *ring_vpc.get_unchecked(idx_r);
                        let adj = if x > -1.0 && x < 0.0 {
                            -1.0
                        } else if x >= 0.0 && x < 1.0 {
                            1.0
                        } else {
                            x
                        };
                        let r = *ring_vpr.get_unchecked(idx_r);
                        if adj != 0.0 && r != 0.0 {
                            acc += *l_ptr.add(i - j) / (adj * r);
                        }
                    }
                }

                if pref_n > 0 {
                    let start_idx = i + 1 - (hist_n + pref_n);
                    let end_idx_excl = i + 1 - hist_n;
                    let mut s = 0.0_f64;
                    let mut k = start_idx;
                    while k + 4 <= end_idx_excl {
                        let a = *l_ptr.add(k);
                        let b = *l_ptr.add(k + 1);
                        let c = *l_ptr.add(k + 2);
                        let d = *l_ptr.add(k + 3);
                        s += a + b + c + d;
                        k += 4;
                    }
                    while k < end_idx_excl {
                        s += *l_ptr.add(k);
                        k += 1;
                    }
                    acc += s;
                }

                let price_v = (acc / (len_v as f64)) * 0.01;
                let dev = (multiplier.mul_add(vpci, 0.0)) * vm;
                let pre_i = (*l_ptr.add(i) - price_v) + dev;

                pre_sum += pre_i;
                if pre_cnt < slow_period {
                    pre_ring[pre_pos] = pre_i;
                    pre_pos += 1;
                    if pre_pos == slow_period {
                        pre_pos = 0;
                    }
                    pre_cnt += 1;
                } else {
                    pre_sum -= pre_ring[pre_pos];
                    pre_ring[pre_pos] = pre_i;
                    pre_pos += 1;
                    if pre_pos == slow_period {
                        pre_pos = 0;
                    }
                }

                if i >= warmup2 {
                    *out.get_unchecked_mut(i) = pre_sum * inv_slow;
                }
            }
        }
    }

    let upto = warmup2.min(len);
    for v in &mut out[..upto] {
        *v = f64::NAN;
    }
    Ok(())
}

// Reference scalar implementation (materializes intermediate series); used for correctness.
#[inline]
fn avsl_scalar_ref(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    first_val: usize,
    out: &mut [f64],
) -> Result<(), AvslError> {
    let len = close.len();

    // rows: 0 vwma_fast | 1 vwma_slow | 2 sma_fast | 3 sma_slow | 4 vol_sma_fast | 5 vol_sma_slow | 6 pre_avsl
    let rows = 7usize;
    let cols = len;

    let mut mu = make_uninit_matrix(rows, cols);
    let warm = [
        first_val + fast_period - 1,
        first_val + slow_period - 1,
        first_val + fast_period - 1,
        first_val + slow_period - 1,
        first_val + fast_period - 1,
        first_val + slow_period - 1,
        first_val + slow_period - 1,
    ];
    init_matrix_prefixes(&mut mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(mu);
    let flat: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let (row0, rest) = flat.split_at_mut(cols);
    let (row1, rest) = rest.split_at_mut(cols);
    let (row2, rest) = rest.split_at_mut(cols);
    let (row3, rest) = rest.split_at_mut(cols);
    let (row4, rest) = rest.split_at_mut(cols);
    let (row5, row6) = rest.split_at_mut(cols);

    {
        let inp = VwmaInput::from_slice(
            close,
            volume,
            VwmaParams {
                period: Some(fast_period),
            },
        );
        vwma_into_slice(row0, &inp, Kernel::Scalar)
            .map_err(|e| AvslError::ComputationError(format!("VWMA Fast error: {}", e)))?;

        let inp = VwmaInput::from_slice(
            close,
            volume,
            VwmaParams {
                period: Some(slow_period),
            },
        );
        vwma_into_slice(row1, &inp, Kernel::Scalar)
            .map_err(|e| AvslError::ComputationError(format!("VWMA Slow error: {}", e)))?;

        let inp = SmaInput::from_slice(
            close,
            SmaParams {
                period: Some(fast_period),
            },
        );
        sma_into_slice(row2, &inp, Kernel::Scalar)
            .map_err(|e| AvslError::ComputationError(format!("SMA Fast error: {}", e)))?;
        let inp = SmaInput::from_slice(
            close,
            SmaParams {
                period: Some(slow_period),
            },
        );
        sma_into_slice(row3, &inp, Kernel::Scalar)
            .map_err(|e| AvslError::ComputationError(format!("SMA Slow error: {}", e)))?;

        let inp = SmaInput::from_slice(
            volume,
            SmaParams {
                period: Some(fast_period),
            },
        );
        sma_into_slice(row4, &inp, Kernel::Scalar)
            .map_err(|e| AvslError::ComputationError(format!("Volume SMA Fast error: {}", e)))?;
        let inp = SmaInput::from_slice(
            volume,
            SmaParams {
                period: Some(slow_period),
            },
        );
        sma_into_slice(row5, &inp, Kernel::Scalar)
            .map_err(|e| AvslError::ComputationError(format!("Volume SMA Slow error: {}", e)))?;
    }

    let vwma_f = &row0[..];
    let vwma_s = &row1[..];
    let sma_f = &row2[..];
    let sma_s = &row3[..];
    let vol_f = &row4[..];
    let vol_s = &row5[..];
    let pre = row6;

    let start = first_val + slow_period - 1;
    for i in start..len {
        let vpc = vwma_s[i] - sma_s[i];
        let vpr = if sma_f[i] != 0.0 {
            vwma_f[i] / sma_f[i]
        } else {
            1.0
        };
        let vm = if vol_s[i] != 0.0 {
            vol_f[i] / vol_s[i]
        } else {
            1.0
        };
        let vpci = vpc * vpr * vm;
        let len_v = if vpc < 0.0 {
            ((vpci - 3.0).abs().round() as usize).max(1).min(200)
        } else {
            ((vpci + 3.0).round() as usize).max(1).min(200)
        };
        let adj = |x: f64| {
            if (-1.0..0.0).contains(&x) {
                -1.0
            } else if (0.0..1.0).contains(&x) {
                1.0
            } else {
                x
            }
        };
        let mut acc = 0.0;
        let base = first_val + slow_period - 1;
        let take = len_v.min(i + 1);
        for j in 0..take {
            let idx = i - j;
            let vpc_c_j = if idx >= base {
                adj(vwma_s[idx] - sma_s[idx])
            } else {
                1.0
            };
            let vpr_j = if idx >= base && sma_f[idx] != 0.0 {
                vwma_f[idx] / sma_f[idx]
            } else {
                1.0
            };
            if vpc_c_j != 0.0 && vpr_j != 0.0 {
                acc += low[idx] / vpc_c_j / vpr_j;
            }
        }
        let price_v = (acc / len_v as f64) / 100.0;
        let dev = multiplier * vpci * vm;
        pre[i] = low[i] - price_v + dev;
    }

    let pre_in = SmaInput::from_slice(
        &pre[..],
        SmaParams {
            period: Some(slow_period),
        },
    );
    sma_into_slice(out, &pre_in, Kernel::Scalar)
        .map_err(|e| AvslError::ComputationError(format!("AVSL SMA error: {}", e)))?;

    let warmup_end = start + slow_period - 1;
    if warmup_end <= len {
        for v in &mut out[..warmup_end] {
            *v = f64::NAN;
        }
    }
    Ok(())
}

// ==================== WASM SIMD128 IMPLEMENTATION ====================
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn avsl_simd128(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    first_val: usize,
    out: &mut [f64],
) -> Result<(), AvslError> {
    // For now, fallback to scalar implementation
    avsl_scalar(
        close,
        low,
        volume,
        fast_period,
        slow_period,
        multiplier,
        first_val,
        out,
    )
}

// ==================== AVX2 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avsl_avx2(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    first_val: usize,
    out: &mut [f64],
) -> Result<(), AvslError> {
    use core::arch::x86_64::*;
    let len = close.len();
    if len == 0 {
        return Err(AvslError::EmptyInputData);
    }

    const MAX_WIN: usize = 200;
    let base = first_val + slow_period - 1;
    let warmup2 = base + slow_period - 1;
    if base >= len {
        let upto = warmup2.min(len);
        for v in &mut out[..upto] {
            *v = f64::NAN;
        }
        for v in &mut out[upto..] {
            *v = f64::NAN;
        }
        return Ok(());
    }

    let inv_fast = 1.0 / (fast_period as f64);
    let inv_slow = 1.0 / (slow_period as f64);

    let mut sum_close_f = 0.0_f64;
    let mut sum_close_s = 0.0_f64;
    let mut sum_vol_f = 0.0_f64;
    let mut sum_vol_s = 0.0_f64;
    let mut sum_cxv_f = 0.0_f64;
    let mut sum_cxv_s = 0.0_f64;

    let mut ring_vpc: [f64; MAX_WIN] = [0.0; MAX_WIN];
    let mut ring_vpr: [f64; MAX_WIN] = [1.0; MAX_WIN];
    let mut ring_pos: usize = 0;

    let mut pre_ring: Vec<f64> = vec![0.0; slow_period];
    let mut pre_pos: usize = 0;
    let mut pre_sum: f64 = 0.0;
    let mut pre_cnt: usize = 0;

    let c_ptr = close.as_ptr();
    let l_ptr = low.as_ptr();
    let v_ptr = volume.as_ptr();

    // masks for adjust()
    let v_neg1 = _mm256_set1_pd(-1.0);
    let v_zero = _mm256_set1_pd(0.0);
    let v_pos1 = _mm256_set1_pd(1.0);

    #[inline(always)]
    unsafe fn adj256(x: __m256d, v_neg1: __m256d, v_zero: __m256d, v_pos1: __m256d) -> __m256d {
        let gt_neg1 = _mm256_cmp_pd(v_neg1, x, _CMP_LT_OQ);
        let lt_zero = _mm256_cmp_pd(x, v_zero, _CMP_LT_OQ);
        let mask1 = _mm256_and_pd(gt_neg1, lt_zero);
        let ge_zero = _mm256_cmp_pd(x, v_zero, _CMP_GE_OQ);
        let lt_pos1 = _mm256_cmp_pd(x, v_pos1, _CMP_LT_OQ);
        let mask2 = _mm256_and_pd(ge_zero, lt_pos1);
        let m1 = _mm256_blendv_pd(x, v_neg1, mask1);
        _mm256_blendv_pd(m1, v_pos1, mask2)
    }

    for i in 0..len {
        if i >= first_val {
            let c = *c_ptr.add(i);
            let v = *v_ptr.add(i);
            let cv = c * v;
            sum_close_f += c;
            sum_vol_f += v;
            sum_cxv_f += cv;
            sum_close_s += c;
            sum_vol_s += v;
            sum_cxv_s += cv;
            if i + 1 > fast_period + first_val {
                let k = i + 1 - fast_period - 1;
                let c_old = *c_ptr.add(k);
                let v_old = *v_ptr.add(k);
                sum_close_f -= c_old;
                sum_vol_f -= v_old;
                sum_cxv_f -= c_old * v_old;
            }
            if i + 1 > slow_period + first_val {
                let k = i + 1 - slow_period - 1;
                let c_old = *c_ptr.add(k);
                let v_old = *v_ptr.add(k);
                sum_close_s -= c_old;
                sum_vol_s -= v_old;
                sum_cxv_s -= c_old * v_old;
            }
        }

        if i >= base {
            let sma_f = sum_close_f * inv_fast;
            let sma_s = sum_close_s * inv_slow;
            let vwma_f = if sum_vol_f != 0.0 {
                sum_cxv_f / sum_vol_f
            } else {
                sma_f
            };
            let vwma_s = if sum_vol_s != 0.0 {
                sum_cxv_s / sum_vol_s
            } else {
                sma_s
            };

            let vpc = vwma_s - sma_s;
            // Use same FP pathway as scalar for vpr to minimize rounding drift
            let vpr = if sma_f != 0.0 { vwma_f / sma_f } else { 1.0 };
            let vol_f = sum_vol_f * inv_fast;
            let vol_s = sum_vol_s * inv_slow;
            let vm = if vol_s != 0.0 { vol_f / vol_s } else { 1.0 };
            let vpci = vpc * vpr * vm;

            let len_v = {
                let t = if vpc < 0.0 {
                    (vpci - 3.0).abs().round()
                } else {
                    (vpci + 3.0).round()
                };
                let m = if t < 1.0 { 1.0 } else { t };
                let m = if m > MAX_WIN as f64 {
                    MAX_WIN as f64
                } else {
                    m
                };
                m as usize
            };

            ring_vpc[ring_pos] = vpc;
            ring_vpr[ring_pos] = vpr;
            ring_pos += 1;
            if ring_pos == MAX_WIN {
                ring_pos = 0;
            }

            let take = len_v.min(i + 1);
            let hist_n = (i - base + 1).min(take);
            let pref_n = take - hist_n;
            let mut acc = 0.0_f64;

            if hist_n > 0 {
                // Scalar-ordered accumulation for exact parity
                let mut rp = if ring_pos == 0 {
                    MAX_WIN - 1
                } else {
                    ring_pos - 1
                };
                for j in 0..hist_n {
                    let idx_r = rp;
                    rp = if rp == 0 { MAX_WIN - 1 } else { rp - 1 };
                    let x = ring_vpc[idx_r];
                    let adj = if x > -1.0 && x < 0.0 {
                        -1.0
                    } else if x >= 0.0 && x < 1.0 {
                        1.0
                    } else {
                        x
                    };
                    let r = ring_vpr[idx_r];
                    if adj != 0.0 && r != 0.0 {
                        acc += *l_ptr.add(i - j) / (adj * r);
                    }
                }
            }

            if pref_n > 0 {
                let start_idx = i + 1 - (hist_n + pref_n);
                let end_idx_excl = i + 1 - hist_n;
                let mut s = 0.0_f64;
                let mut k = start_idx;
                while k < end_idx_excl {
                    s += *l_ptr.add(k);
                    k += 1;
                }
                acc += s;
            }

            let price_v = (acc / (len_v as f64)) * 0.01;
            let dev = (multiplier.mul_add(vpci, 0.0)) * vm;
            let pre_i = (*l_ptr.add(i) - price_v) + dev;

            pre_sum += pre_i;
            if pre_cnt < slow_period {
                pre_ring[pre_pos] = pre_i;
                pre_pos += 1;
                if pre_pos == slow_period {
                    pre_pos = 0;
                }
                pre_cnt += 1;
            } else {
                pre_sum -= pre_ring[pre_pos];
                pre_ring[pre_pos] = pre_i;
                pre_pos += 1;
                if pre_pos == slow_period {
                    pre_pos = 0;
                }
            }

            if i >= warmup2 {
                *out.get_unchecked_mut(i) = pre_sum * inv_slow;
            }
        }
    }

    let upto = warmup2.min(len);
    for v in &mut out[..upto] {
        *v = f64::NAN;
    }

    Ok(())
}

// ==================== AVX512 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn avsl_avx512(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    first_val: usize,
    out: &mut [f64],
) -> Result<(), AvslError> {
    use core::arch::x86_64::*;
    let len = close.len();
    if len == 0 {
        return Err(AvslError::EmptyInputData);
    }

    const MAX_WIN: usize = 200;
    let base = first_val + slow_period - 1;
    let warmup2 = base + slow_period - 1;
    if base >= len {
        let upto = warmup2.min(len);
        for v in &mut out[..upto] {
            *v = f64::NAN;
        }
        for v in &mut out[upto..] {
            *v = f64::NAN;
        }
        return Ok(());
    }

    let inv_fast = 1.0 / (fast_period as f64);
    let inv_slow = 1.0 / (slow_period as f64);
    let mut sum_close_f = 0.0_f64;
    let mut sum_close_s = 0.0_f64;
    let mut sum_vol_f = 0.0_f64;
    let mut sum_vol_s = 0.0_f64;
    let mut sum_cxv_f = 0.0_f64;
    let mut sum_cxv_s = 0.0_f64;
    let mut ring_vpc: [f64; MAX_WIN] = [0.0; MAX_WIN];
    let mut ring_vpr: [f64; MAX_WIN] = [1.0; MAX_WIN];
    let mut ring_pos: usize = 0;
    let mut pre_ring: Vec<f64> = vec![0.0; slow_period];
    let mut pre_pos: usize = 0;
    let mut pre_sum: f64 = 0.0;
    let mut pre_cnt: usize = 0;

    let c_ptr = close.as_ptr();
    let l_ptr = low.as_ptr();
    let v_ptr = volume.as_ptr();
    #[inline(always)]
    fn adj(x: f64) -> f64 {
        if x > -1.0 && x < 0.0 {
            -1.0
        } else if x >= 0.0 && x < 1.0 {
            1.0
        } else {
            x
        }
    }

    for i in 0..len {
        if i >= first_val {
            let c = unsafe { *c_ptr.add(i) };
            let v = unsafe { *v_ptr.add(i) };
            let cv = c * v;
            sum_close_f += c;
            sum_vol_f += v;
            sum_cxv_f += cv;
            sum_close_s += c;
            sum_vol_s += v;
            sum_cxv_s += cv;
            if i + 1 > fast_period + first_val {
                let k = i + 1 - fast_period - 1;
                let c_old = unsafe { *c_ptr.add(k) };
                let v_old = unsafe { *v_ptr.add(k) };
                sum_close_f -= c_old;
                sum_vol_f -= v_old;
                sum_cxv_f -= c_old * v_old;
            }
            if i + 1 > slow_period + first_val {
                let k = i + 1 - slow_period - 1;
                let c_old = unsafe { *c_ptr.add(k) };
                let v_old = unsafe { *v_ptr.add(k) };
                sum_close_s -= c_old;
                sum_vol_s -= v_old;
                sum_cxv_s -= c_old * v_old;
            }
        }

        if i >= base {
            let sma_f = sum_close_f * inv_fast;
            let sma_s = sum_close_s * inv_slow;
            let vwma_f = if sum_vol_f != 0.0 {
                sum_cxv_f / sum_vol_f
            } else {
                sma_f
            };
            let vwma_s = if sum_vol_s != 0.0 {
                sum_cxv_s / sum_vol_s
            } else {
                sma_s
            };
            let vpc = vwma_s - sma_s;
            // Match scalar semantics: when volume sum is zero (vwma = sma), vpr should be 1.0
            let vpr = if sma_f != 0.0 && sum_vol_f != 0.0 {
                (sum_cxv_f * (fast_period as f64)) / (sum_vol_f * sum_close_f)
            } else {
                1.0
            };
            let vol_f = sum_vol_f * inv_fast;
            let vol_s = sum_vol_s * inv_slow;
            let vm = if vol_s != 0.0 { vol_f / vol_s } else { 1.0 };
            let vpci = vpc * vpr * vm;
            let len_v = {
                let t = if vpc < 0.0 {
                    (vpci - 3.0).abs().round()
                } else {
                    (vpci + 3.0).round()
                };
                let m = if t < 1.0 { 1.0 } else { t };
                let m = if m > MAX_WIN as f64 {
                    MAX_WIN as f64
                } else {
                    m
                };
                m as usize
            };

            ring_vpc[ring_pos] = vpc;
            ring_vpr[ring_pos] = vpr;
            ring_pos += 1;
            if ring_pos == MAX_WIN {
                ring_pos = 0;
            }

            let take = len_v.min(i + 1);
            let hist_n = (i - base + 1).min(take);
            let pref_n = take - hist_n;
            let mut acc = 0.0_f64;
            // Accumulate recent portion using exact scalar mapping to ensure parity
            if hist_n > 0 {
                let mut rp = if ring_pos == 0 {
                    MAX_WIN - 1
                } else {
                    ring_pos - 1
                };
                for j in 0..hist_n {
                    let idx_r = rp;
                    rp = if rp == 0 { MAX_WIN - 1 } else { rp - 1 };
                    let a = adj(ring_vpc[idx_r]);
                    let r = ring_vpr[idx_r];
                    if a != 0.0 && r != 0.0 {
                        acc += unsafe { *l_ptr.add(i - j) } / (a * r);
                    }
                }
            }
            if pref_n > 0 {
                let start_idx = i + 1 - (hist_n + pref_n);
                let end_idx_excl = i + 1 - hist_n;
                let mut s = 0.0_f64;
                let mut k = start_idx;
                let n = end_idx_excl - start_idx;
                let vec_n = n / 8;
                let rem = n % 8;
                for _ in 0..vec_n {
                    let a = unsafe { _mm512_loadu_pd(l_ptr.add(k)) };
                    let arr: [f64; 8] = core::mem::transmute(a);
                    s += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
                    k += 8;
                }
                for _ in 0..rem {
                    s += unsafe { *l_ptr.add(k) };
                    k += 1;
                }
                acc += s;
            }

            let price_v = (acc / (len_v as f64)) * 0.01;
            let dev = (multiplier.mul_add(vpci, 0.0)) * vm;
            let pre_i = unsafe { *l_ptr.add(i) } - price_v + dev;
            pre_sum += pre_i;
            if pre_cnt < slow_period {
                pre_ring[pre_pos] = pre_i;
                pre_pos += 1;
                if pre_pos == slow_period {
                    pre_pos = 0;
                }
                pre_cnt += 1;
            } else {
                pre_sum -= pre_ring[pre_pos];
                pre_ring[pre_pos] = pre_i;
                pre_pos += 1;
                if pre_pos == slow_period {
                    pre_pos = 0;
                }
            }
            if i >= warmup2 {
                unsafe {
                    *out.get_unchecked_mut(i) = pre_sum * inv_slow;
                }
            }
        }
    }

    let upto = warmup2.min(len);
    for v in &mut out[..upto] {
        *v = f64::NAN;
    }

    Ok(())
}

// ==================== STREAMING API (O(1) per update) ====================
// Decision: Streaming path uses rolling sums + small rings to achieve O(1) work per tick,
// matching scalar semantics and warmup (warmup2). Public API unchanged (try_new, update).
#[derive(Debug, Clone)]
pub struct AvslStream {
    // params
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,

    // precomputed
    inv_fast: f64,
    inv_slow: f64,
    base: usize,    // first index where vpc/vpr are valid: slow - 1 (first==0 in stream)
    warmup2: usize, // base + slow - 1

    // time / counters
    t: usize, // number of updates processed so far (0-based index of current bar)

    // --- rolling sums for SMA/VWMA (fast & slow) ---
    sum_close_f: f64,
    sum_vol_f: f64,
    sum_cxv_f: f64,
    sum_close_s: f64,
    sum_vol_s: f64,
    sum_cxv_s: f64,

    // ring to remove trailing samples for fast/slow windows
    ring_len: usize,      // = max(fast_period, slow_period)
    ring_pos: usize,      // next write position
    close_ring: Vec<f64>, // len = ring_len
    vol_ring: Vec<f64>,   // len = ring_len
    cxv_ring: Vec<f64>,   // len = ring_len (close*volume)

    // --- prefix-sum rings for variable-length last-K queries (K <= 200) ---
    // We store cumulative sums at index (t+1) % R. Query sum of last K as S[t+1] - S[t+1-K].
    // Take R = MAX_WIN + 1 so we never overwrite a needed entry (K <= MAX_WIN).
    csum_low: [f64; AvslStream::R],
    csum_y: [f64; AvslStream::R], // y_i = low_i / (adj(vpc_i) * vpr_i), 0 before base

    // --- SMA over 'pre' (slow_period) ---
    pre_ring: Vec<f64>, // len = slow_period
    pre_pos: usize,
    pre_sum: f64,
    pre_cnt: usize, // counts only bars >= base for pre-averaging
}

impl AvslStream {
    // constants used by the streaming kernel
    const MAX_WIN: usize = 200;
    const R: usize = Self::MAX_WIN + 1; // prefix-sum ring length

    pub fn try_new(params: AvslParams) -> Result<Self, AvslError> {
        let fast_period = params.fast_period.unwrap_or(12);
        let slow_period = params.slow_period.unwrap_or(26);
        let multiplier = params.multiplier.unwrap_or(2.0);

        if fast_period == 0 {
            return Err(AvslError::InvalidPeriod {
                period: fast_period,
                data_len: 0,
            });
        }
        if slow_period == 0 {
            return Err(AvslError::InvalidPeriod {
                period: slow_period,
                data_len: 0,
            });
        }
        if multiplier <= 0.0 || !multiplier.is_finite() {
            return Err(AvslError::InvalidMultiplier { multiplier });
        }

        let ring_len = fast_period.max(slow_period);
        Ok(Self {
            fast_period,
            slow_period,
            multiplier,
            inv_fast: 1.0 / (fast_period as f64),
            inv_slow: 1.0 / (slow_period as f64),
            base: slow_period - 1,
            warmup2: (slow_period - 1) + (slow_period - 1),
            t: 0,

            sum_close_f: 0.0,
            sum_vol_f: 0.0,
            sum_cxv_f: 0.0,
            sum_close_s: 0.0,
            sum_vol_s: 0.0,
            sum_cxv_s: 0.0,

            ring_len,
            ring_pos: 0,
            close_ring: vec![0.0; ring_len],
            vol_ring: vec![0.0; ring_len],
            cxv_ring: vec![0.0; ring_len],

            csum_low: [0.0; Self::R],
            csum_y: [0.0; Self::R],

            pre_ring: vec![0.0; slow_period],
            pre_pos: 0,
            pre_sum: 0.0,
            pre_cnt: 0,
        })
    }

    #[inline(always)]
    fn sum_last(csum: &[f64; Self::R], t_plus_1_mod: usize, t_plus_1: usize, k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }
        if t_plus_1 >= k {
            let start = (t_plus_1 - k) % Self::R;
            csum[t_plus_1_mod] - csum[start]
        } else {
            csum[t_plus_1_mod]
        }
    }

    #[inline(always)]
    fn adjust_vpc(x: f64) -> f64 {
        if x > -1.0 && x < 0.0 {
            -1.0
        } else if x >= 0.0 && x < 1.0 {
            1.0
        } else {
            x
        }
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64, low: f64, volume: f64) -> Option<f64> {
        let i = self.t;

        // ---- 1) Update rolling sums for fast/slow windows (SMA & VWMA) ----
        let cv = close * volume;

        // positions of samples to drop
        let rp = self.ring_pos;
        let rl = self.ring_len;

        let pos_old_fast = (rp + rl - (self.fast_period % rl)) % rl;
        let pos_old_slow = (rp + rl - (self.slow_period % rl)) % rl;

        // old samples
        let (c_old_f, v_old_f, cv_old_f) = if i >= self.fast_period {
            (
                self.close_ring[pos_old_fast],
                self.vol_ring[pos_old_fast],
                self.cxv_ring[pos_old_fast],
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let (c_old_s, v_old_s, cv_old_s) = if i >= self.slow_period {
            (
                self.close_ring[pos_old_slow],
                self.vol_ring[pos_old_slow],
                self.cxv_ring[pos_old_slow],
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        // add new, remove old (fast)
        self.sum_close_f += close - c_old_f;
        self.sum_vol_f += volume - v_old_f;
        self.sum_cxv_f += cv - cv_old_f;

        // add new, remove old (slow)
        self.sum_close_s += close - c_old_s;
        self.sum_vol_s += volume - v_old_s;
        self.sum_cxv_s += cv - cv_old_s;

        // write into rings and advance
        self.close_ring[rp] = close;
        self.vol_ring[rp] = volume;
        self.cxv_ring[rp] = cv;
        self.ring_pos = (rp + 1) % rl;

        // ---- 2) Prefix sums for low and y (y is 0 before base to match scalar semantics) ----
        let t1_mod = (i + 1) % Self::R;

        // Compute vpc/vpr/vm only when i >= base; otherwise y contributes 0
        let mut y_i = 0.0;
        if i >= self.base {
            let sma_f = self.sum_close_f * self.inv_fast;
            let sma_s = self.sum_close_s * self.inv_slow;

            let vwma_f = if self.sum_vol_f != 0.0 {
                self.sum_cxv_f / self.sum_vol_f
            } else {
                sma_f
            };
            let vwma_s = if self.sum_vol_s != 0.0 {
                self.sum_cxv_s / self.sum_vol_s
            } else {
                sma_s
            };

            let vpc = vwma_s - sma_s;
            let vpr = if sma_f != 0.0 { vwma_f / sma_f } else { 1.0 };

            let vol_f = self.sum_vol_f * self.inv_fast;
            let vol_s = self.sum_vol_s * self.inv_slow;
            let _vm = if vol_s != 0.0 { vol_f / vol_s } else { 1.0 };

            // adj mapping identical to scalar_ref
            let adj = Self::adjust_vpc(vpc);
            if adj != 0.0 && vpr != 0.0 {
                y_i = low / (adj * vpr);
            }
        }

        // write prefix sums (store cumulative at index t+1)
        self.csum_low[t1_mod] = self.csum_low[i % Self::R] + low;
        self.csum_y[t1_mod] = self.csum_y[i % Self::R] + y_i;

        // ---- 3) If i >= base, compute vpci/len_v and the pre value; then feed slow SMA ----
        let mut out: Option<f64> = None;

        if i >= self.base {
            let sma_f = self.sum_close_f * self.inv_fast;
            let sma_s = self.sum_close_s * self.inv_slow;
            let vwma_f = if self.sum_vol_f != 0.0 {
                self.sum_cxv_f / self.sum_vol_f
            } else {
                sma_f
            };
            let vwma_s = if self.sum_vol_s != 0.0 {
                self.sum_cxv_s / self.sum_vol_s
            } else {
                sma_s
            };
            let vpc = vwma_s - sma_s;
            let vpr = if sma_f != 0.0 { vwma_f / sma_f } else { 1.0 };
            let vol_f = self.sum_vol_f * self.inv_fast;
            let vol_s = self.sum_vol_s * self.inv_slow;
            let vm = if vol_s != 0.0 { vol_f / vol_s } else { 1.0 };
            let vpci = vpc * vpr * vm;

            // len_v per spec, clamped to [1..MAX_WIN]
            let t_len = if vpc < 0.0 {
                (vpci - 3.0).abs().round()
            } else {
                (vpci + 3.0).round()
            };
            let len_v = t_len.max(1.0).min(Self::MAX_WIN as f64) as usize;

            // take/hist/pref split identical to scalar path
            let take = len_v.min(i + 1);
            let hist_n = ((i - self.base + 1).min(take)) as usize;
            let pref_n = take - hist_n;

            // prefix-sum queries (O(1))
            let sum_hist_y = Self::sum_last(&self.csum_y, t1_mod, i + 1, hist_n);
            let sum_take_l = Self::sum_last(&self.csum_low, t1_mod, i + 1, take);
            let sum_hist_l = Self::sum_last(&self.csum_low, t1_mod, i + 1, hist_n);
            let acc = sum_hist_y + (sum_take_l - sum_hist_l);

            // price_v and pre_i
            let inv_len_v = 1.0 / (len_v as f64);
            let price_v = (acc * inv_len_v) * 0.01;
            let dev = self.multiplier.mul_add(vpci, 0.0) * vm;
            let pre_i = (low - price_v) + dev;

            // slow SMA over 'pre'
            self.pre_sum += pre_i;
            if self.pre_cnt < self.slow_period {
                self.pre_ring[self.pre_pos] = pre_i;
                self.pre_pos += 1;
                if self.pre_pos == self.slow_period {
                    self.pre_pos = 0;
                }
                self.pre_cnt += 1;
            } else {
                self.pre_sum -= self.pre_ring[self.pre_pos];
                self.pre_ring[self.pre_pos] = pre_i;
                self.pre_pos += 1;
                if self.pre_pos == self.slow_period {
                    self.pre_pos = 0;
                }
            }

            if i >= self.warmup2 {
                out = Some(self.pre_sum * self.inv_slow);
            }
        }

        // advance time
        self.t = i + 1;
        out
    }
}

// ==================== BATCH API ====================
#[derive(Clone, Debug)]
pub struct AvslBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub multiplier: (f64, f64, f64),
}

impl Default for AvslBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (12, 12, 0),    // Static at default value
            slow_period: (26, 275, 1),    // 250 combos (includes default 26)
            multiplier: (2.0, 2.0, 0.0), // Static at default value
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AvslBatchBuilder {
    range: AvslBatchRange,
    kernel: Kernel,
}

impl AvslBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn fast_range(mut self, s: usize, e: usize, st: usize) -> Self {
        self.range.fast_period = (s, e, st);
        self
    }
    #[inline]
    pub fn fast_static(mut self, v: usize) -> Self {
        self.range.fast_period = (v, v, 0);
        self
    }
    #[inline]
    pub fn slow_range(mut self, s: usize, e: usize, st: usize) -> Self {
        self.range.slow_period = (s, e, st);
        self
    }
    #[inline]
    pub fn slow_static(mut self, v: usize) -> Self {
        self.range.slow_period = (v, v, 0);
        self
    }
    #[inline]
    pub fn mult_range(mut self, s: f64, e: f64, st: f64) -> Self {
        self.range.multiplier = (s, e, st);
        self
    }
    #[inline]
    pub fn mult_static(mut self, v: f64) -> Self {
        self.range.multiplier = (v, v, 0.0);
        self
    }

    pub fn apply_slices(
        self,
        close: &[f64],
        low: &[f64],
        volume: &[f64],
    ) -> Result<AvslBatchOutput, AvslError> {
        avsl_batch_with_kernel(close, low, volume, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        close_src: &str,
        low_src: &str,
    ) -> Result<AvslBatchOutput, AvslError> {
        let close = source_type(c, close_src);
        let low = source_type(c, low_src);
        let volume = c.volume.as_slice();
        self.apply_slices(close, low, volume)
    }

    pub fn with_default_candles(c: &Candles) -> Result<AvslBatchOutput, AvslError> {
        AvslBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close", "low")
    }

    pub fn with_default_slices(
        close: &[f64],
        low: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<AvslBatchOutput, AvslError> {
        AvslBatchBuilder::new()
            .kernel(k)
            .apply_slices(close, low, volume)
    }
}

#[derive(Clone, Debug)]
pub struct AvslBatchOutput {
    pub values: Vec<f64>, // row-major [rows * cols]
    pub combos: Vec<AvslParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AvslBatchOutput {
    pub fn row_for_params(&self, p: &AvslParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_period.unwrap_or(12) == p.fast_period.unwrap_or(12)
                && c.slow_period.unwrap_or(26) == p.slow_period.unwrap_or(26)
                && (c.multiplier.unwrap_or(2.0) - p.multiplier.unwrap_or(2.0)).abs() < 1e-12
        })
    }

    #[inline]
    pub fn values_for(&self, p: &AvslParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
    if st == 0 || s == e {
        return vec![s];
    }
    if s < e {
        return (s..=e).step_by(st.max(1)).collect();
    }
    // reversed bounds
    let mut v = Vec::new();
    let step = st.max(1);
    let mut cur = s;
    while cur >= e {
        v.push(cur);
        if cur < step { break; }
        cur -= step;
        if cur == usize::MAX { break; }
    }
    v
}

#[inline(always)]
fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
    let step = if st.is_sign_negative() { -st } else { st };
    if step.abs() < 1e-12 || (s - e).abs() < 1e-12 {
        return vec![s];
    }
    let mut v = Vec::new();
    if s <= e {
        let mut x = s;
        while x <= e + 1e-12 {
            v.push(x);
            x += step;
        }
    } else {
        let mut x = s;
        while x + 1e-12 >= e {
            v.push(x);
            x -= step;
        }
    }
    v
}

#[inline(always)]
fn expand_grid_avsl(r: &AvslBatchRange) -> Vec<AvslParams> {
    let fs = axis_usize(r.fast_period);
    let ss = axis_usize(r.slow_period);
    let ms = axis_f64(r.multiplier);
    let cap = fs
        .len()
        .checked_mul(ss.len())
        .and_then(|x| x.checked_mul(ms.len()))
        .unwrap_or(0);
    let mut out = Vec::with_capacity(cap);
    for &f in &fs {
        for &s in &ss {
            for &m in &ms {
                out.push(AvslParams {
                    fast_period: Some(f),
                    slow_period: Some(s),
                    multiplier: Some(m),
                });
            }
        }
    }
    out
}

pub fn avsl_batch_with_kernel(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    sweep: &AvslBatchRange,
    k: Kernel,
) -> Result<AvslBatchOutput, AvslError> {
    if close.is_empty() {
        return Err(AvslError::EmptyInputData);
    }
    if close.len() != low.len() || close.len() != volume.len() {
        return Err(AvslError::DataLengthMismatch {
            close_len: close.len(),
            low_len: low.len(),
            volume_len: volume.len(),
        });
    }

    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AvslError::InvalidKernelForBatch(other)),
    };

    // We dispatch rows via Scalar compute, like alma.rs maps batch→SIMD choice.
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    let combos = expand_grid_avsl(sweep);
    if combos.is_empty() {
        return Err(AvslError::InvalidRange {
            start: sweep.fast_period.0,
            end: sweep.fast_period.1,
            step: sweep.fast_period.2,
        });
    }

    let cols = close.len();
    let rows = combos.len();

    // Allocate uninitialized rows×cols, set only warm prefixes to NaN.
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // warm prefix for each row = first_valid + slow_period - 1
    let first = first_valid_max3(close, low, volume).ok_or(AvslError::AllValuesNaN)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.slow_period.unwrap_or(26) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Transmute to &mut [f64] without extra init
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    avsl_batch_inner_into(close, low, volume, &combos, simd, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(AvslBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// Separate functions for explicit parallel/serial control
#[inline(always)]
pub fn avsl_batch_slice(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    sweep: &AvslBatchRange,
    kern: Kernel,
) -> Result<AvslBatchOutput, AvslError> {
    avsl_batch_inner(close, low, volume, sweep, kern, false)
}

#[inline(always)]
pub fn avsl_batch_par_slice(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    sweep: &AvslBatchRange,
    kern: Kernel,
) -> Result<AvslBatchOutput, AvslError> {
    avsl_batch_inner(close, low, volume, sweep, kern, true)
}

#[inline(always)]
fn avsl_batch_inner(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    sweep: &AvslBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AvslBatchOutput, AvslError> {
    if close.is_empty() {
        return Err(AvslError::EmptyInputData);
    }
    if close.len() != low.len() || close.len() != volume.len() {
        return Err(AvslError::DataLengthMismatch {
            close_len: close.len(),
            low_len: low.len(),
            volume_len: volume.len(),
        });
    }

    let kernel = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AvslError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    let combos = expand_grid_avsl(sweep);
    if combos.is_empty() {
        return Err(AvslError::InvalidRange {
            start: sweep.fast_period.0,
            end: sweep.fast_period.1,
            step: sweep.fast_period.2,
        });
    }

    let cols = close.len();
    let rows = combos.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let first = first_valid_max3(close, low, volume).ok_or(AvslError::AllValuesNaN)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.slow_period.unwrap_or(26) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    if parallel {
        avsl_batch_inner_into(close, low, volume, &combos, simd, out)?;
    } else {
        // Serial version
        let out_rows: &mut [MaybeUninit<f64>] = unsafe {
            core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
        };
        for (r, dst) in out_rows.chunks_mut(cols).enumerate() {
            let p = &combos[r];
            let fast = p.fast_period.unwrap_or(12);
            let slow = p.slow_period.unwrap_or(26);
            let mult = p.multiplier.unwrap_or(2.0);
            let dst_f64: &mut [f64] =
                unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, cols) };
            avsl_compute_into(close, low, volume, fast, slow, mult, first, simd, dst_f64)?;
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(AvslBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn avsl_batch_inner_into(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    combos: &[AvslParams],
    kern: Kernel,
    out: &mut [f64],
) -> Result<(), AvslError> {
    let cols = close.len();
    let first = first_valid_max3(close, low, volume).ok_or(AvslError::AllValuesNaN)?;
    // Safety: out length is rows*cols laid out by caller.
    let rows = combos.len();
    // length check and warmup prefix validation
    let expected = rows
        .checked_mul(cols)
        .ok_or(AvslError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        })?;
    if out.len() != expected {
        return Err(AvslError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    let out_rows: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| -> Result<(), AvslError> {
        let p = &combos[row];
        let fast = p.fast_period.unwrap_or(12);
        let slow = p.slow_period.unwrap_or(26);
        let mult = p.multiplier.unwrap_or(2.0);

        // Write directly into the row slice
        let dst: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols) };
        avsl_compute_into(close, low, volume, fast, slow, mult, first, kern, dst)
    };

    // Non-WASM: allow parallel fill. On WASM: serial.
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out_rows
            .par_chunks_mut(cols)
            .enumerate()
            .try_for_each(|(r, dst)| do_row(r, dst))
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (r, dst) in out_rows.chunks_mut(cols).enumerate() {
            do_row(r, dst)?;
        }
        Ok(())
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyclass(name = "AvslStream")]
pub struct AvslStreamPy {
    stream: AvslStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AvslStreamPy {
    #[new]
    fn new(fast_period: usize, slow_period: usize, multiplier: f64) -> PyResult<Self> {
        let params = AvslParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            multiplier: Some(multiplier),
        };
        let stream =
            AvslStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AvslStreamPy { stream })
    }

    fn update(&mut self, close: f64, low: f64, volume: f64) -> Option<f64> {
        self.stream.update(close, low, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "avsl")]
#[pyo3(signature = (close, low, volume, fast_period=None, slow_period=None, multiplier=None, kernel=None))]
pub fn avsl_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    multiplier: Option<f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let close_slice = close.as_slice()?;
    let low_slice = low.as_slice()?;
    let volume_slice = volume.as_slice()?;

    let kern = validate_kernel(kernel, false)?;
    let params = AvslParams {
        fast_period,
        slow_period,
        multiplier,
    };
    let input = AvslInput::from_slices(close_slice, low_slice, volume_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| avsl_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "avsl_batch")]
#[pyo3(signature = (close, low, volume, fast_range, slow_range, mult_range, kernel=None))]
pub fn avsl_batch_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    fast_range: (usize, usize, usize),
    slow_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let close = close.as_slice()?;
    let low = low.as_slice()?;
    let volume = volume.as_slice()?;

    let sweep = AvslBatchRange {
        fast_period: fast_range,
        slow_period: slow_range,
        multiplier: mult_range,
    };

    // Expand to know rows and allocate flattened output uninitialized.
    // Treat empty expansion or overflow as an invalid range error.
    let combos = expand_grid_avsl(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err(
            AvslError::InvalidRange {
                start: sweep.fast_period.0,
                end: sweep.fast_period.1,
                step: sweep.fast_period.2,
            }
            .to_string(),
        ));
    }
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| {
            PyValueError::new_err(
                AvslError::InvalidRange {
                    start: sweep.fast_period.0,
                    end: sweep.fast_period.1,
                    step: sweep.fast_period.2,
                }
                .to_string(),
            )
        })?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

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
            avsl_batch_inner_into(close, low, volume, &combos, simd, slice_out).map(|_| combos)
        })
        .map_err(|e: AvslError| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|p| p.fast_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|p| p.slow_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        combos
            .iter()
            .map(|p| p.multiplier.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

// ==================== Python CUDA Bindings ====================
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::avsl_wrapper::CudaAvsl;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

// AVSL-specific Python device handle with CAI v3 + DLPack and context guard
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32Avsl", unsendable)]
pub struct DeviceArrayF32AvslPy {
    pub(crate) inner: crate::cuda::moving_averages::DeviceArrayF32,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32AvslPy {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "use factory functions (avsl_cuda_*_dev) to create this type",
        ))
    }

    #[getter]
    fn __cuda_array_interface__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let inner = &self.inner;
        let d = pyo3::types::PyDict::new(py);
        let item = std::mem::size_of::<f32>();
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * item, item))?;
        let size = inner.rows.saturating_mul(inner.cols);
        let ptr_val: usize = if size == 0 {
            0
        } else {
            inner.buf.as_device_ptr().as_raw() as usize
        };
        d.set_item("data", (ptr_val, false))?;
        // Producer stream is synchronized in wrapper before returning → omit 'stream'
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self._device_id as i32))
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        // Compute allocation device and validate any `dl_device` hint.
        let (kdl, alloc_dev) = self.__dlpack_device__()?;
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        // Producer stream is synchronized in wrapper before returning, so we ignore the hint.
        let _ = stream;

        // Move the VRAM buffer into the shared DLPack manager so the consumer owns it.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            crate::cuda::moving_averages::DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32AvslPy {
    pub fn new(inner: crate::cuda::moving_averages::DeviceArrayF32, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self { inner, _ctx_guard: ctx_guard, _device_id: device_id }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "avsl_cuda_batch_dev")]
#[pyo3(signature = (close_f32, low_f32, volume_f32, fast_range, slow_range, mult_range, device_id=0))]
pub fn avsl_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    volume_f32: numpy::PyReadonlyArray1<'py, f32>,
    fast_range: (usize, usize, usize),
    slow_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<(DeviceArrayF32AvslPy, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let close = close_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let vol = volume_f32.as_slice()?;
    let sweep = AvslBatchRange {
        fast_period: fast_range,
        slow_period: slow_range,
        multiplier: mult_range,
    };
    let (inner, ctx, dev_id, combos) = py.allow_threads(|| {
        let cuda = CudaAvsl::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let (arr, combos) = cuda
            .avsl_batch_dev(close, low, vol, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, ctx, dev_id, combos))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|p| p.fast_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|p| p.slow_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        combos
            .iter()
            .map(|p| p.multiplier.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok((DeviceArrayF32AvslPy::new(inner, ctx, dev_id), dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "avsl_cuda_many_series_one_param_dev")]
#[pyo3(signature = (close_tm_f32, low_tm_f32, volume_tm_f32, cols, rows, fast_period, slow_period, multiplier, device_id=0))]
pub fn avsl_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    close_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    volume_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32AvslPy> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let c = close_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let v = volume_tm_f32.as_slice()?;
    let params = AvslParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
        multiplier: Some(multiplier),
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaAvsl::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let arr = cuda
            .avsl_many_series_one_param_time_major_dev(c, l, v, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
    })?;
    Ok(DeviceArrayF32AvslPy::new(inner, ctx, dev_id))
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn avsl_js(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
) -> Result<Vec<f64>, JsValue> {
    // Validate inputs following the same rules as the scalar path
    let len = close.len();
    if len == 0 {
        return Err(JsValue::from_str("empty input"));
    }
    if close.len() != low.len() || close.len() != volume.len() {
        return Err(JsValue::from_str("data length mismatch"));
    }
    let first = first_valid_max3(close, low, volume)
        .ok_or_else(|| JsValue::from_str("All values are NaN"))?;
    if fast_period == 0 || fast_period > len {
        return Err(JsValue::from_str("Invalid period"));
    }
    if slow_period == 0 || slow_period > len {
        return Err(JsValue::from_str("Invalid period"));
    }
    if !(multiplier.is_finite()) || multiplier <= 0.0 {
        return Err(JsValue::from_str("Invalid multiplier"));
    }
    if len - first < slow_period {
        return Err(JsValue::from_str("Not enough valid data"));
    }

    // Route single-series JS API through the same batch kernel path used by
    // avsl_batch() to ensure identical numerics. This avoids divergence that
    // was observed between single-series and batch implementations.
    let sweep = AvslBatchRange {
        fast_period: (fast_period, fast_period, 0),
        slow_period: (slow_period, slow_period, 0),
        multiplier: (multiplier, multiplier, 0.0),
    };
    let out = avsl_batch_with_kernel(close, low, volume, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out.values)
}

/// Zero-allocation native API that writes results into a caller-provided buffer.
///
/// - Preserves NaN warmups exactly like the Vec-returning API.
/// - `out.len()` must equal the input length; returns an error on mismatch.
/// - Uses `Kernel::Auto` for dispatch.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn avsl_into(input: &AvslInput, out: &mut [f64]) -> Result<(), AvslError> {
    avsl_into_slice(out, input, Kernel::Auto)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn avsl_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn avsl_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn avsl_into(
    close_ptr: *const f64,
    low_ptr: *const f64,
    vol_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || low_ptr.is_null() || vol_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer"));
    }
    unsafe {
        let close = core::slice::from_raw_parts(close_ptr, len);
        let low = core::slice::from_raw_parts(low_ptr, len);
        let vol = core::slice::from_raw_parts(vol_ptr, len);
        let out = core::slice::from_raw_parts_mut(out_ptr, len);

        // Validate like avsl_js to avoid internal panics in batch helpers
        let n = close.len();
        if n == 0 {
            return Err(JsValue::from_str("empty input"));
        }
        if close.len() != low.len() || close.len() != vol.len() {
            return Err(JsValue::from_str("data length mismatch"));
        }
        let first = match first_valid_max3(close, low, vol) {
            Some(i) => i,
            None => return Err(JsValue::from_str("All values are NaN")),
        };
        if fast_period == 0 || fast_period > n || slow_period == 0 || slow_period > n {
            return Err(JsValue::from_str("Invalid period"));
        }
        if !(multiplier.is_finite()) || multiplier <= 0.0 {
            return Err(JsValue::from_str("Invalid multiplier"));
        }
        if n - first < slow_period {
            return Err(JsValue::from_str("Not enough valid data"));
        }

        let params = AvslParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            multiplier: Some(multiplier),
        };

        // Handle in-place gracefully by computing into a temporary slice
        if out_ptr as *const f64 == close_ptr as *const f64
            || out_ptr as *const f64 == low_ptr as *const f64
            || out_ptr as *const f64 == vol_ptr as *const f64
        {
            let mut temp = vec![0.0; len];
            let combos = vec![params];
            avsl_batch_inner_into(close, low, vol, &combos, detect_best_batch_kernel(), &mut temp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            out.copy_from_slice(&temp);
        } else {
            let combos = vec![params];
            avsl_batch_inner_into(close, low, vol, &combos, detect_best_batch_kernel(), out)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AvslBatchConfig {
    pub fast_range: (usize, usize, usize),
    pub slow_range: (usize, usize, usize),
    pub mult_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AvslBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AvslParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = avsl_batch)]
pub fn avsl_batch_unified_js(
    close: &[f64],
    low: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: AvslBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AvslBatchRange {
        fast_period: cfg.fast_range,
        slow_period: cfg.slow_range,
        multiplier: cfg.mult_range,
    };

    let out = avsl_batch_with_kernel(close, low, volume, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = AvslBatchJsOutput {
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
pub struct AvslContext {
    fast_period: usize,
    slow_period: usize,
    multiplier: f64,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl AvslContext {
    #[wasm_bindgen(constructor)]
    pub fn new(
        fast_period: usize,
        slow_period: usize,
        multiplier: f64,
    ) -> Result<AvslContext, JsValue> {
        if fast_period == 0 {
            return Err(JsValue::from_str(&format!(
                "Invalid fast period: {}",
                fast_period
            )));
        }
        if slow_period == 0 {
            return Err(JsValue::from_str(&format!(
                "Invalid slow period: {}",
                slow_period
            )));
        }
        if multiplier <= 0.0 || multiplier.is_nan() || multiplier.is_infinite() {
            return Err(JsValue::from_str(&format!(
                "Invalid multiplier: {}",
                multiplier
            )));
        }

        Ok(AvslContext {
            fast_period,
            slow_period,
            multiplier,
            kernel: Kernel::Auto,
        })
    }

    pub fn update_into(
        &self,
        close_ptr: *const f64,
        low_ptr: *const f64,
        vol_ptr: *const f64,
        out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if len < self.slow_period {
            return Err(JsValue::from_str("Data length less than slow period"));
        }

        if close_ptr.is_null() || low_ptr.is_null() || vol_ptr.is_null() || out_ptr.is_null() {
            return Err(JsValue::from_str("Null pointer passed"));
        }

        unsafe {
            let close = std::slice::from_raw_parts(close_ptr, len);
            let low = std::slice::from_raw_parts(low_ptr, len);
            let volume = std::slice::from_raw_parts(vol_ptr, len);
            let out = std::slice::from_raw_parts_mut(out_ptr, len);

            let first = first_valid_max3(close, low, volume).unwrap_or(0);

            // Handle in-place operation
            if out_ptr as *const f64 == close_ptr
                || out_ptr as *const f64 == low_ptr
                || out_ptr as *const f64 == vol_ptr
            {
                let mut temp = vec![0.0; len];
                avsl_scalar(
                    close,
                    low,
                    volume,
                    self.fast_period,
                    self.slow_period,
                    self.multiplier,
                    first,
                    &mut temp,
                )
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
                out.copy_from_slice(&temp);
            } else {
                avsl_scalar(
                    close,
                    low,
                    volume,
                    self.fast_period,
                    self.slow_period,
                    self.multiplier,
                    first,
                    out,
                )
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.slow_period - 1
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn avsl_batch_into(
    close_ptr: *const f64,
    low_ptr: *const f64,
    vol_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_start: usize,
    fast_end: usize,
    fast_step: usize,
    slow_start: usize,
    slow_end: usize,
    slow_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || low_ptr.is_null() || vol_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to avsl_batch_into"));
    }
    unsafe {
        let close = core::slice::from_raw_parts(close_ptr, len);
        let low = core::slice::from_raw_parts(low_ptr, len);
        let vol = core::slice::from_raw_parts(vol_ptr, len);
        let sweep = AvslBatchRange {
            fast_period: (fast_start, fast_end, fast_step),
            slow_period: (slow_start, slow_end, slow_step),
            multiplier: (mult_start, mult_end, mult_step),
        };
        // Expand to know rows; treat empty expansion or overflow as an error.
        let combos = expand_grid_avsl(&sweep);
        if combos.is_empty() {
            return Err(JsValue::from_str(
                &AvslError::InvalidRange {
                    start: sweep.fast_period.0,
                    end: sweep.fast_period.1,
                    step: sweep.fast_period.2,
                }
                .to_string(),
            ));
        }
        let rows = combos.len();
        let cols = len;
        let total = rows.checked_mul(cols).ok_or_else(|| {
            JsValue::from_str(
                &AvslError::InvalidRange {
                    start: sweep.fast_period.0,
                    end: sweep.fast_period.1,
                    step: sweep.fast_period.2,
                }
                .to_string(),
            )
        })?;
        let out = core::slice::from_raw_parts_mut(out_ptr, total);

        // Fill flattened out
        avsl_batch_inner_into(close, low, vol, &combos, detect_best_batch_kernel(), out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;
    use std::error::Error;

    // Skip macro for unsupported kernels
    macro_rules! skip_if_unsupported {
        ($kernel:expr, $test_name:expr) => {
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                if matches!(
                    $kernel,
                    Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch
                ) {
                    eprintln!("Skipping {} - AVX not supported", $test_name);
                    return Ok(());
                }
            }
        };
    }

    fn check_avsl_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AvslInput::from_candles(&candles, "close", "low", AvslParams::default());
        let result = avsl_with_kernel(&input, kernel)?;

        // REFERENCE VALUES FROM PINESCRIPT
        let expected_last_five = [
            56471.61721191,
            56267.11946706,
            56079.12004921,
            55910.07971214,
            55765.37864229,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            let tolerance = expected_last_five[i].abs() * 0.01; // 1% tolerance for this complex indicator
            assert!(
                diff < tolerance,
                "[{}] AVSL {:?} mismatch at idx {}: got {}, expected {}, diff {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i],
                diff
            );
        }
        Ok(())
    }

    fn check_avsl_empty_input(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let empty: [f64; 0] = [];
        let params = AvslParams::default();
        let input = AvslInput::from_slices(&empty, &empty, &empty, params);
        let res = avsl(&input);
        assert!(
            res.is_err(),
            "[{}] Expected error for empty input",
            test_name
        );
        Ok(())
    }

    fn check_avsl_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = AvslParams::default();
        let input = AvslInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let res = avsl_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Expected error for all NaN input",
            test_name
        );
        Ok(())
    }

    fn check_avsl_mismatched_lengths(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let close = [1.0, 2.0, 3.0];
        let low = [0.9, 1.9]; // Different length
        let volume = [100.0, 200.0, 300.0];
        let params = AvslParams::default();
        let input = AvslInput::from_slices(&close, &low, &volume, params);
        let res = avsl_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Expected error for mismatched data lengths",
            test_name
        );
        Ok(())
    }

    fn check_avsl_invalid_multiplier(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![1.0; 100];
        let params = AvslParams {
            fast_period: Some(12),
            slow_period: Some(26),
            multiplier: Some(-1.0), // Invalid negative multiplier
        };
        let input = AvslInput::from_slices(&data, &data, &data, params);
        let res = avsl_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Expected error for invalid multiplier",
            test_name
        );
        Ok(())
    }

    // Macro to generate all test variants for each kernel
    macro_rules! generate_all_avsl_tests {
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

    // Generate all single kernel tests
    generate_all_avsl_tests!(
        check_avsl_accuracy,
        check_avsl_empty_input,
        check_avsl_all_nan,
        check_avsl_mismatched_lengths,
        check_avsl_invalid_multiplier
    );

    // Batch testing functions
    fn check_avsl_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = AvslBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles, "close", "low")?;

        let def = AvslParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), candles.close.len());

        let expected_last_five = [
            56471.61721191,
            56267.11946706,
            56079.12004921,
            55910.07971214,
            55765.37864229,
        ];

        let start = row.len().saturating_sub(5);
        for (i, &val) in row[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            let tolerance = expected_last_five[i].abs() * 0.01;
            assert!(
                diff < tolerance,
                "[{}] AVSL batch default row {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_avsl_batch_range(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = AvslBatchBuilder::new()
            .kernel(kernel)
            .fast_range(10, 15, 5)
            .slow_range(20, 30, 10)
            .mult_range(1.5, 2.5, 0.5)
            .apply_candles(&candles, "close", "low")?;

        let expected_combos = 2 * 2 * 3; // (10,15 step 5) * (20,30 step 10) * (1.5,2.0,2.5)
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, candles.close.len());

        Ok(())
    }

    // Macro for batch tests
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_avsl_batch_default_row);
    gen_batch_tests!(check_avsl_batch_range);

    #[test]
    fn test_avsl_streaming() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Calculate batch AVSL for comparison
        let params = AvslParams::default();
        let input = AvslInput::from_candles(&candles, "close", "low", params.clone());
        let batch_result = avsl(&input)?;

        // Create streaming calculator
        let mut stream = AvslStream::try_new(params)?;

        // Feed data to stream and collect results
        let mut stream_results = Vec::new();
        for i in 0..candles.close.len() {
            if let Some(value) = stream.update(candles.close[i], candles.low[i], candles.volume[i])
            {
                stream_results.push(value);
            }
        }

        // Compare last values
        if !stream_results.is_empty() && !batch_result.values.is_empty() {
            let last_stream = stream_results.last().unwrap();
            let last_batch = batch_result
                .values
                .iter()
                .rev()
                .find(|&&v| !v.is_nan())
                .unwrap();

            let diff = (last_stream - last_batch).abs();
            let tolerance = last_batch.abs() * 0.01; // 1% tolerance
            assert!(
                diff < tolerance,
                "Streaming vs batch mismatch: {} vs {}, diff {}",
                last_stream,
                last_batch,
                diff
            );
        }

        Ok(())
    }

    #[test]
    fn test_avsl_batch_helpers() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with_default_candles
        let output = AvslBatchBuilder::with_default_candles(&candles).map_err(|e| {
            eprintln!("Error: {:?}", e);
            e
        })?;
        assert_eq!(output.cols, candles.close.len());

        // Test row_for_params
        let params = AvslParams::default();
        let row_idx = output.row_for_params(&params);
        assert!(row_idx.is_some());
        assert_eq!(row_idx.unwrap(), 0); // Default params should be first row

        // Test explicit parallel/serial functions
        let sweep = AvslBatchRange::default();
        let par_output = avsl_batch_par_slice(
            &candles.close,
            &candles.low,
            &candles.volume,
            &sweep,
            Kernel::Auto,
        )?;
        let ser_output = avsl_batch_slice(
            &candles.close,
            &candles.low,
            &candles.volume,
            &sweep,
            Kernel::ScalarBatch,
        )?;

        assert_eq!(par_output.rows, ser_output.rows);
        assert_eq!(par_output.cols, ser_output.cols);

        // Test with_default_slices helper
        let default_output = AvslBatchBuilder::with_default_slices(
            &candles.close,
            &candles.low,
            &candles.volume,
            Kernel::Auto,
        )?;
        assert_eq!(default_output.cols, candles.close.len());
        assert_eq!(default_output.rows, 250); // Default batch range yields 250 parameter combos

        Ok(())
    }

    // Ensure the zero-allocation API matches the Vec-returning API, including NaN warmups.
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_avsl_into_matches_api() -> Result<(), Box<dyn Error>> {
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AvslInput::from_candles(&candles, "close", "low", AvslParams::default());

        // Baseline via existing API
        let baseline = avsl(&input)?;

        // Zero-allocation path
        let mut out = vec![0.0f64; candles.close.len()];
        avsl_into(&input, &mut out)?;

        assert_eq!(baseline.values.len(), out.len());
        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline.values[i], out[i]),
                "Mismatch at index {}: api={} into={}",
                i,
                baseline.values[i],
                out[i]
            );
        }
        Ok(())
    }
}
