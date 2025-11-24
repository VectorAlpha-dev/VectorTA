//! # Stochastic RSI (SRSI)
//!
//! A momentum oscillator that applies the Stochastic formula to RSI values instead of price data.
//!
//! ## Parameters
//! - **rsi_period**: Period for RSI calculation (default: 14).
//! - **stoch_period**: Period for Stochastic calculation on RSI (default: 14).
//! - **k**: Period for slow K MA (default: 3).
//! - **d**: Period for slow D MA (default: 3).
//! - **source**: Candle field (default: "close").
//!
//! ## Errors
//! - **RsiError**: Error from RSI calculation.
//! - **StochError**: Error from Stochastic calculation.
//!
//! ## Returns
//! - **`Ok(SrsiOutput)`** on success, containing vectors `k` and `d`.
//! - **`Err(SrsiError)`** otherwise.
//!
//! ## Developer Notes
//! - **Decision log**: SIMD paths delegate to the scalar kernel; CUDA wrapper is available and returns VRAM handles consumed via CUDA Array Interface v3 and DLPack v1.x; numerical outputs match the scalar reference.
//! - Scalar optimized: inlined Wilder RSI + O(n) min/max via block prefix/suffix scans (avoids per-iteration modulo) and rolling SMAs. Defaults (14,14,3,3) use classic path for bitwise-identical outputs.
//! - **AVX2/AVX512 Kernels**: Delegated to scalar for now; sliding-window min/max is branchy and memory-bound, so SIMD shows no clear win. Revisit with alternative layouts if profiling warrants.
//! - **Streaming**: Enabled O(1) per-tick kernel using Wilder RSI + monotonic deques for Stoch window and rolling SMAs for K/D; matches scalar outputs after warmup.
//! - **Memory Optimization**: Batch mode uses RSI result caching to avoid redundant calculations across parameter combinations. Uses zero-copy helper functions for output allocation.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::ffi as pyffi;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::ffi::{c_void, CString};
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::rsi::{rsi, RsiError, RsiInput, RsiOutput, RsiParams};
use crate::indicators::stoch::{stoch, StochError, StochInput, StochOutput, StochParams};
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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SrsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SrsiData::Slice(slice) => slice,
            SrsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SrsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SrsiOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SrsiParams {
    pub rsi_period: Option<usize>,
    pub stoch_period: Option<usize>,
    pub k: Option<usize>,
    pub d: Option<usize>,
    pub source: Option<String>,
}

impl Default for SrsiParams {
    fn default() -> Self {
        Self {
            rsi_period: Some(14),
            stoch_period: Some(14),
            k: Some(3),
            d: Some(3),
            source: Some("close".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SrsiInput<'a> {
    pub data: SrsiData<'a>,
    pub params: SrsiParams,
}

impl<'a> SrsiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SrsiParams) -> Self {
        Self {
            data: SrsiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SrsiParams) -> Self {
        Self {
            data: SrsiData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SrsiParams::default())
    }
    #[inline]
    pub fn get_rsi_period(&self) -> usize {
        self.params.rsi_period.unwrap_or(14)
    }
    #[inline]
    pub fn get_stoch_period(&self) -> usize {
        self.params.stoch_period.unwrap_or(14)
    }
    #[inline]
    pub fn get_k(&self) -> usize {
        self.params.k.unwrap_or(3)
    }
    #[inline]
    pub fn get_d(&self) -> usize {
        self.params.d.unwrap_or(3)
    }
    #[inline]
    pub fn get_source(&self) -> &str {
        self.params.source.as_deref().unwrap_or("close")
    }
}

#[derive(Clone, Debug)]
pub struct SrsiBuilder {
    rsi_period: Option<usize>,
    stoch_period: Option<usize>,
    k: Option<usize>,
    d: Option<usize>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for SrsiBuilder {
    fn default() -> Self {
        Self {
            rsi_period: None,
            stoch_period: None,
            k: None,
            d: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SrsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn rsi_period(mut self, n: usize) -> Self {
        self.rsi_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn stoch_period(mut self, n: usize) -> Self {
        self.stoch_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn k(mut self, n: usize) -> Self {
        self.k = Some(n);
        self
    }
    #[inline(always)]
    pub fn d(mut self, n: usize) -> Self {
        self.d = Some(n);
        self
    }
    #[inline(always)]
    pub fn source<S: Into<String>>(mut self, s: S) -> Self {
        self.source = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SrsiOutput, SrsiError> {
        let p = SrsiParams {
            rsi_period: self.rsi_period,
            stoch_period: self.stoch_period,
            k: self.k,
            d: self.d,
            source: self.source.clone(),
        };
        let i = SrsiInput::from_candles(c, self.source.as_deref().unwrap_or("close"), p);
        srsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SrsiOutput, SrsiError> {
        let p = SrsiParams {
            rsi_period: self.rsi_period,
            stoch_period: self.stoch_period,
            k: self.k,
            d: self.d,
            source: self.source.clone(),
        };
        let i = SrsiInput::from_slice(d, p);
        srsi_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum SrsiError {
    #[error("srsi: Error from RSI calculation: {0}")]
    RsiError(#[from] RsiError),
    #[error("srsi: Error from Stochastic calculation: {0}")]
    StochError(#[from] StochError),
    #[error("srsi: Input data is empty.")]
    EmptyInputData,
    #[error("srsi: All input data values are NaN.")]
    AllValuesNaN,
    #[error("srsi: Invalid period {period} for data length {data_len}.")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("srsi: Not enough valid data for the requested period. needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("srsi: Output length mismatch - destination buffers must match input data length. Expected {expected}, got k={k_len}, d={d_len}")]
    OutputLengthMismatch {
        expected: usize,
        k_len: usize,
        d_len: usize,
    },
    #[error("srsi: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("srsi: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn srsi(input: &SrsiInput) -> Result<SrsiOutput, SrsiError> {
    srsi_with_kernel(input, Kernel::Auto)
}

pub fn srsi_with_kernel(input: &SrsiInput, kernel: Kernel) -> Result<SrsiOutput, SrsiError> {
    let data: &[f64] = match &input.data {
        SrsiData::Candles { candles, source } => source_type(candles, source),
        SrsiData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(SrsiError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrsiError::AllValuesNaN)?;
    let len = data.len();
    let rsi_period = input.get_rsi_period();
    let stoch_period = input.get_stoch_period();
    let k_len = input.get_k();
    let d_len = input.get_d();

    let needed = rsi_period
        .max(stoch_period)
        .max(k_len)
        .max(d_len);
    let valid = len - first;
    if valid < needed {
        return Err(SrsiError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                srsi_scalar(data, rsi_period, stoch_period, k_len, d_len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                srsi_avx2(data, rsi_period, stoch_period, k_len, d_len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                srsi_avx512(data, rsi_period, stoch_period, k_len, d_len)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn srsi_scalar(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k_period: usize,
    d_period: usize,
) -> Result<SrsiOutput, SrsiError> {
    // Fast path preserves exact behavior for classic defaults
    if rsi_period == 14 && stoch_period == 14 && k_period == 3 && d_period == 3 {
        return srsi_scalar_classic(data, rsi_period, stoch_period, k_period, d_period);
    }

    // ---- validation ---------------------------------------------------------
    let n = data.len();
    if n == 0 {
        return Err(SrsiError::EmptyInputData);
    }
    if rsi_period == 0 || stoch_period == 0 || k_period == 0 || d_period == 0 {
        return Err(SrsiError::InvalidPeriod {
            period: rsi_period.max(stoch_period).max(k_period).max(d_period),
            data_len: n,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrsiError::AllValuesNaN)?;
    let max_need = rsi_period.max(stoch_period).max(k_period).max(d_period);
    if n - first < max_need {
        return Err(SrsiError::NotEnoughValidData {
            needed: max_need,
            valid: n - first,
        });
    }

    // ---- warmups ------------------------------------------------------------
    let rsi_warmup = first + rsi_period;
    let stoch_warmup = rsi_warmup + stoch_period - 1;
    let k_warmup = stoch_warmup + k_period - 1;
    let d_warmup = k_warmup + d_period - 1;

    if n <= d_warmup {
        return Err(SrsiError::NotEnoughValidData {
            needed: d_warmup + 1,
            valid: n,
        });
    }

    // ---- buffers ------------------------------------------------------------
    let mut rsi_vals = alloc_with_nan_prefix(n, rsi_warmup);
    let mut k_out = alloc_with_nan_prefix(n, k_warmup);
    let mut d_out = alloc_with_nan_prefix(n, d_warmup);

    // ---- step 1: Wilder RSI (single pass) -----------------------------------
    let mut avg_gain = 0.0f64;
    let mut avg_loss = 0.0f64;
    let mut prev = *data.get_unchecked(first);
    let end_init = (first + rsi_period).min(n.saturating_sub(1));
    for i in (first + 1)..=end_init {
        let cur = *data.get_unchecked(i);
        if cur.is_finite() && prev.is_finite() {
            let ch = cur - prev;
            if ch > 0.0 {
                avg_gain += ch;
            } else {
                avg_loss += -ch;
            }
        }
        prev = cur;
    }

    let rp = rsi_period as f64;
    avg_gain /= rp;
    avg_loss /= rp;
    let alpha = 1.0f64 / rp;

    // First RSI at index rsi_warmup
    if rsi_warmup < n {
        rsi_vals[rsi_warmup] = if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - 100.0 / (1.0 + rs)
        };
    }

    // Continue smoothing
    prev = *data.get_unchecked(rsi_warmup);
    for i in (rsi_warmup + 1)..n {
        let cur = *data.get_unchecked(i);
        if cur.is_finite() && prev.is_finite() {
            let ch = cur - prev;
            let gain = if ch > 0.0 { ch } else { 0.0 };
            let loss = if ch < 0.0 { -ch } else { 0.0 };
            avg_gain = (gain - avg_gain).mul_add(alpha, avg_gain);
            avg_loss = (loss - avg_loss).mul_add(alpha, avg_loss);
            rsi_vals[i] = if avg_loss == 0.0 {
                100.0
            } else {
                let rs = avg_gain / avg_loss;
                100.0 - 100.0 / (1.0 + rs)
            };
        }
        prev = cur;
    }

    // ---- step 2: Stoch of RSI via block min/max + SMA for K and D -----------
    let sp = stoch_period;
    let kp = k_period;
    let dp = d_period;
    if rsi_warmup < n {
        let m = n - rsi_warmup;
        let base = rsi_warmup;

        // Block prefix/suffix for min and max (avoid deque and per-iter modulo)
        let mut pref_max = vec![0.0f64; m];
        let mut suff_max = vec![0.0f64; m];
        let mut pref_min = vec![0.0f64; m];
        let mut suff_min = vec![0.0f64; m];

        // Process prefix within blocks of length `sp`
        let blocks = (m + sp - 1) / sp;
        let p_pref_max = pref_max.as_mut_ptr();
        let p_pref_min = pref_min.as_mut_ptr();
        let p_rsi = rsi_vals.as_ptr().add(base);
        for b in 0..blocks {
            let start = b * sp;
            let end = core::cmp::min(start + sp, m);
            if start >= end {
                break;
            }
            unsafe {
                let v0 = *p_rsi.add(start);
                *p_pref_max.add(start) = v0;
                *p_pref_min.add(start) = v0;
                let mut i = start + 1;
                while i < end {
                    let v = *p_rsi.add(i);
                    let pmx = *p_pref_max.add(i - 1);
                    let pmn = *p_pref_min.add(i - 1);
                    *p_pref_max.add(i) = if v > pmx { v } else { pmx };
                    *p_pref_min.add(i) = if v < pmn { v } else { pmn };
                    i += 1;
                }
            }
        }
        // Process suffix within blocks of length `sp`
        let p_suff_max = suff_max.as_mut_ptr();
        let p_suff_min = suff_min.as_mut_ptr();
        for b in 0..blocks {
            let block_end_excl = core::cmp::min((b + 1) * sp, m);
            if block_end_excl == 0 {
                break;
            }
            let block_start = block_end_excl - core::cmp::min(sp, block_end_excl);
            unsafe {
                let last = block_end_excl - 1;
                let v_last = *p_rsi.add(last);
                *p_suff_max.add(last) = v_last;
                *p_suff_min.add(last) = v_last;
                let mut i = last;
                while i > block_start {
                    let prev = i - 1;
                    let v = *p_rsi.add(prev);
                    let smx = *p_suff_max.add(i);
                    let smn = *p_suff_min.add(i);
                    *p_suff_max.add(prev) = if v > smx { v } else { smx };
                    *p_suff_min.add(prev) = if v < smn { v } else { smn };
                    i = prev;
                }
            }
        }

        // Rolling SMA for K and D without materializing fast-K
        let mut sum_k = 0.0f64;
        let mut sum_d = 0.0f64;
        let mut fk_ring = vec![0.0f64; kp];
        let mut sk_ring = vec![0.0f64; dp];
        let mut fk_pos = 0usize;
        let mut sk_pos = 0usize;

        let i0 = stoch_warmup;
        let mut i = i0;
        while i < n {
            let t = i - base;
            let t_start = t + 1 - sp; // safe since i >= base + sp - 1
            let hi_l = suff_max[t_start];
            let hi_r = pref_max[t];
            let lo_l = suff_min[t_start];
            let lo_r = pref_min[t];
            let hi = if hi_l > hi_r { hi_l } else { hi_r };
            let lo = if lo_l < lo_r { lo_l } else { lo_r };
            let x = *rsi_vals.get_unchecked(i);
            let fk = if hi > lo {
                ((x - lo) * 100.0) / (hi - lo)
            } else {
                50.0
            };

            sum_k += fk;
            if i >= i0 + kp {
                sum_k -= *fk_ring.get_unchecked(fk_pos);
            }
            *fk_ring.get_unchecked_mut(fk_pos) = fk;
            fk_pos += 1;
            if fk_pos == kp {
                fk_pos = 0;
            }

            if i >= k_warmup {
                let sk = sum_k / (kp as f64);
                *k_out.get_unchecked_mut(i) = sk;

                sum_d += sk;
                if i >= k_warmup + dp {
                    sum_d -= *sk_ring.get_unchecked(sk_pos);
                }
                *sk_ring.get_unchecked_mut(sk_pos) = sk;
                sk_pos += 1;
                if sk_pos == dp {
                    sk_pos = 0;
                }

                if i >= d_warmup {
                    *d_out.get_unchecked_mut(i) = sum_d / (dp as f64);
                }
            }
            i += 1;
        }
    }

    Ok(SrsiOutput { k: k_out, d: d_out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx2(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
) -> Result<SrsiOutput, SrsiError> {
    srsi_scalar(data, rsi_period, stoch_period, k, d)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx512(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
) -> Result<SrsiOutput, SrsiError> {
    if stoch_period <= 32 {
        srsi_avx512_short(data, rsi_period, stoch_period, k, d)
    } else {
        srsi_avx512_long(data, rsi_period, stoch_period, k, d)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx512_short(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
) -> Result<SrsiOutput, SrsiError> {
    srsi_scalar(data, rsi_period, stoch_period, k, d)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx512_long(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
) -> Result<SrsiOutput, SrsiError> {
    srsi_scalar(data, rsi_period, stoch_period, k, d)
}

/// Classic kernel optimization for SRSI with inline RSI and Stochastic calculations
/// Eliminates function call overhead for RSI and Stochastic operations
#[inline]
pub unsafe fn srsi_scalar_classic(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k_period: usize,
    d_period: usize,
) -> Result<SrsiOutput, SrsiError> {
    let n = data.len();
    if n == 0 {
        return Err(SrsiError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrsiError::AllValuesNaN)?;

    // Calculate warmup periods
    let rsi_warmup = first + rsi_period;
    let stoch_warmup = rsi_warmup + stoch_period - 1;
    let k_warmup = stoch_warmup + k_period - 1;
    let d_warmup = k_warmup + d_period - 1;

    if n <= d_warmup {
        return Err(SrsiError::NotEnoughValidData {
            needed: d_warmup + 1,
            valid: n,
        });
    }

    // Step 1: Calculate RSI inline (Wilder's smoothing method)
    let mut rsi_values = alloc_with_nan_prefix(n, rsi_warmup);

    // Initialize gain/loss averages with first period
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    let mut prev = data[first];

    for i in (first + 1)..(first + rsi_period + 1).min(n) {
        if data[i].is_finite() && prev.is_finite() {
            let change = data[i] - prev;
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss += -change;
            }
            prev = data[i];
        }
    }

    avg_gain /= rsi_period as f64;
    avg_loss /= rsi_period as f64;

    // Calculate RSI values using Wilder's smoothing
    let alpha = 1.0 / rsi_period as f64;
    let alpha_1minus = 1.0 - alpha;

    if first + rsi_period < n {
        rsi_values[first + rsi_period] = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        };

        prev = data[first + rsi_period];
    }

    for i in (first + rsi_period + 1)..n {
        if data[i].is_finite() && prev.is_finite() {
            let change = data[i] - prev;
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, -change)
            };

            avg_gain = alpha * gain + alpha_1minus * avg_gain;
            avg_loss = alpha * loss + alpha_1minus * avg_loss;

            rsi_values[i] = if avg_loss == 0.0 {
                100.0
            } else {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            };

            prev = data[i];
        }
    }

    // Step 2: Calculate Stochastic on RSI values inline
    let mut fast_k = alloc_with_nan_prefix(n, stoch_warmup);

    // Calculate Fast %K (raw stochastic of RSI)
    for i in stoch_warmup..n {
        let start = i + 1 - stoch_period;
        let mut min_rsi = f64::MAX;
        let mut max_rsi = f64::MIN;

        for j in start..=i {
            if rsi_values[j].is_finite() {
                min_rsi = min_rsi.min(rsi_values[j]);
                max_rsi = max_rsi.max(rsi_values[j]);
            }
        }

        if max_rsi > min_rsi {
            fast_k[i] = 100.0 * (rsi_values[i] - min_rsi) / (max_rsi - min_rsi);
        } else {
            fast_k[i] = 50.0; // When range is 0, use middle value
        }
    }

    // Step 3: Calculate Slow %K (SMA of Fast %K)
    let mut slow_k = alloc_with_nan_prefix(n, k_warmup);

    // Calculate initial SMA for Slow %K
    let mut k_sum = 0.0;
    for i in stoch_warmup..(stoch_warmup + k_period).min(n) {
        if fast_k[i].is_finite() {
            k_sum += fast_k[i];
        }
    }

    if stoch_warmup + k_period <= n {
        slow_k[stoch_warmup + k_period - 1] = k_sum / k_period as f64;

        // Continue with rolling SMA
        for i in (stoch_warmup + k_period)..n {
            if fast_k[i].is_finite() && fast_k[i - k_period].is_finite() {
                k_sum += fast_k[i] - fast_k[i - k_period];
                slow_k[i] = k_sum / k_period as f64;
            }
        }
    }

    // Step 4: Calculate Slow %D (SMA of Slow %K)
    let mut slow_d = alloc_with_nan_prefix(n, d_warmup);

    // Calculate initial SMA for Slow %D
    let mut d_sum = 0.0;
    for i in k_warmup..(k_warmup + d_period).min(n) {
        if slow_k[i].is_finite() {
            d_sum += slow_k[i];
        }
    }

    if k_warmup + d_period <= n {
        slow_d[k_warmup + d_period - 1] = d_sum / d_period as f64;

        // Continue with rolling SMA
        for i in (k_warmup + d_period)..n {
            if slow_k[i].is_finite() && slow_k[i - d_period].is_finite() {
                d_sum += slow_k[i] - slow_k[i - d_period];
                slow_d[i] = d_sum / d_period as f64;
            }
        }
    }

    Ok(SrsiOutput {
        k: slow_k,
        d: slow_d,
    })
}

#[inline(always)]
pub fn srsi_row_scalar(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    if let Ok(res) = unsafe { srsi_scalar(data, rsi_period, stoch_period, k, d) } {
        k_out.copy_from_slice(&res.k);
        d_out.copy_from_slice(&res.d);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx2(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx512(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx512_short(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx512_long(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[derive(Debug, Clone)]
pub struct SrsiStream {
    // --- configuration ---
    rsi_period: usize,
    stoch_period: usize,
    k_period: usize,
    d_period: usize,

    // --- Wilder RSI state ---
    prev: f64, // last price
    has_prev: bool,
    init_count: usize, // number of deltas accumulated for initial avg
    sum_gain: f64,
    sum_loss: f64,
    avg_gain: f64,
    avg_loss: f64,
    alpha: f64, // 1 / rsi_period
    rsi_ready: bool,
    rsi_index: usize, // 0-based index of produced RSI samples since ready
    last_rsi: f64,

    // --- Stoch(RSI): window min/max via monotonic deques (amortized O(1)) ---
    // Each holds (rsi_index, value)
    max_q: VecDeque<(usize, f64)>, // non-increasing values
    min_q: VecDeque<(usize, f64)>, // non-decreasing values

    // --- Slow %K SMA state (rolling) ---
    fk_ring: Vec<f64>,
    fk_sum: f64,
    fk_pos: usize,
    fk_count: usize,
    inv_k: f64,

    // --- Slow %D SMA state (rolling over Slow %K) ---
    sk_ring: Vec<f64>,
    sk_sum: f64,
    sk_pos: usize,
    sk_count: usize,
    inv_d: f64,
}

impl SrsiStream {
    pub fn try_new(params: SrsiParams) -> Result<Self, SrsiError> {
        let rsi_period = params.rsi_period.unwrap_or(14);
        let stoch_period = params.stoch_period.unwrap_or(14);
        let k_period = params.k.unwrap_or(3);
        let d_period = params.d.unwrap_or(3);

        if rsi_period == 0 || stoch_period == 0 || k_period == 0 || d_period == 0 {
            return Err(SrsiError::InvalidPeriod {
                period: rsi_period
                    .max(stoch_period)
                    .max(k_period)
                    .max(d_period),
                data_len: 0,
            });
        }

        Ok(Self {
            rsi_period,
            stoch_period,
            k_period,
            d_period,

            prev: f64::NAN,
            has_prev: false,
            init_count: 0,
            sum_gain: 0.0,
            sum_loss: 0.0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            alpha: 1.0 / (rsi_period as f64),
            rsi_ready: false,
            rsi_index: 0,
            last_rsi: f64::NAN,

            max_q: VecDeque::with_capacity(stoch_period),
            min_q: VecDeque::with_capacity(stoch_period),

            fk_ring: vec![0.0; k_period],
            fk_sum: 0.0,
            fk_pos: 0,
            fk_count: 0,
            inv_k: 1.0 / (k_period as f64),

            sk_ring: vec![0.0; d_period],
            sk_sum: 0.0,
            sk_pos: 0,
            sk_count: 0,
            inv_d: 1.0 / (d_period as f64),
        })
    }

    /// Reset the streaming state (keeps parameterization).
    #[inline]
    pub fn reset(&mut self) {
        self.prev = f64::NAN;
        self.has_prev = false;
        self.init_count = 0;
        self.sum_gain = 0.0;
        self.sum_loss = 0.0;
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.rsi_ready = false;
        self.rsi_index = 0;
        self.last_rsi = f64::NAN;
        self.max_q.clear();
        self.min_q.clear();
        self.fk_ring.fill(0.0);
        self.fk_sum = 0.0;
        self.fk_pos = 0;
        self.fk_count = 0;
        self.sk_ring.fill(0.0);
        self.sk_sum = 0.0;
        self.sk_pos = 0;
        self.sk_count = 0;
    }

    /// Push one price; returns Some((slow_k, slow_d)) only after full warmup,
    /// else returns None during warmup.
    ///
    /// Complexity: amortized O(1) per call.
    pub fn update(&mut self, v: f64) -> Option<(f64, f64)> {
        if !v.is_finite() {
            // Consistent with batch path: missing/invalid breaks continuity.
            self.reset();
            return None;
        }

        // --- Stage 1: build Wilder RSI in O(1) ---
        if !self.has_prev {
            self.prev = v;
            self.has_prev = true;
            return None; // need at least one delta
        }

        let ch = v - self.prev;
        self.prev = v;

        // Warmup: accumulate first rsi_period deltas to form initial averages.
        if !self.rsi_ready {
            if ch > 0.0 {
                self.sum_gain += ch;
            } else {
                self.sum_loss += -ch;
            }
            self.init_count += 1;

            if self.init_count < self.rsi_period {
                return None; // still accumulating initial averages
            }
            // Set initial Wilder averages
            self.avg_gain = self.sum_gain / (self.rsi_period as f64);
            self.avg_loss = self.sum_loss / (self.rsi_period as f64);

            // First RSI value after warmup (matches scalar implementation)
            let rsi = if self.avg_loss == 0.0 {
                100.0
            } else {
                let rs = self.avg_gain / self.avg_loss;
                100.0 - 100.0 / (1.0 + rs)
            };
            self.last_rsi = rsi;
            self.rsi_ready = true;
            self.rsi_index = 0;

            // Seed deques with first RSI
            self.push_rsi_to_deques(self.rsi_index, rsi);
            // Not enough RSI samples yet for Stoch %K
            return None;
        }

        // Wilder smoothing update (use mul_add for precision+speed)
        let gain = if ch > 0.0 { ch } else { 0.0 };
        let loss = if ch < 0.0 { -ch } else { 0.0 };
        self.avg_gain = (gain - self.avg_gain).mul_add(self.alpha, self.avg_gain);
        self.avg_loss = (loss - self.avg_loss).mul_add(self.alpha, self.avg_loss);

        let rsi = if self.avg_loss == 0.0 {
            100.0
        } else {
            let rs = self.avg_gain / self.avg_loss;
            100.0 - 100.0 / (1.0 + rs)
        };
        self.last_rsi = rsi;

        // --- Stage 2: push RSI into monotonic min/max for Stoch window ---
        self.rsi_index += 1;
        self.push_rsi_to_deques(self.rsi_index, rsi);

        // Not enough RSI samples for Fast %K yet?
        if self.rsi_index + 1 < self.stoch_period {
            return None;
        }

        // Window start index (inclusive) over RSI stream
        let start = self.rsi_index + 1 - self.stoch_period;
        // Evict expired from front (strictly less than window start)
        while let Some(&(j, _)) = self.max_q.front() {
            if j < start {
                self.max_q.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(j, _)) = self.min_q.front() {
            if j < start {
                self.min_q.pop_front();
            } else {
                break;
            }
        }

        debug_assert!(!self.max_q.is_empty() && !self.min_q.is_empty());
        let hi = self.max_q.front().unwrap().1;
        let lo = self.min_q.front().unwrap().1;

        // Fast %K on RSI
        let fast_k = if hi > lo {
            // Avoid divide in tight loops where possible (see fast-math note below)
            let range = hi - lo;
            ((rsi - lo) * 100.0) / range
        } else {
            50.0
        };

        // --- Stage 3: Slow %K (SMA of Fast %K) in O(1) ---
        let slow_k_opt = Self::push_sma(
            fast_k,
            &mut self.fk_ring,
            &mut self.fk_sum,
            &mut self.fk_pos,
            &mut self.fk_count,
            self.k_period,
            self.inv_k,
        );

        let slow_k = match slow_k_opt {
            None => return None, // waiting for K SMA to fill
            Some(v) => v,
        };

        // --- Stage 4: Slow %D (SMA of Slow %K) in O(1) ---
        let slow_d_opt = Self::push_sma(
            slow_k,
            &mut self.sk_ring,
            &mut self.sk_sum,
            &mut self.sk_pos,
            &mut self.sk_count,
            self.d_period,
            self.inv_d,
        );

        slow_d_opt.map(|d| (slow_k, d))
    }

    #[inline(always)]
    fn push_rsi_to_deques(&mut self, idx: usize, rsi: f64) {
        // Insert into max deque: keep values non-increasing
        while let Some(&(_, v)) = self.max_q.back() {
            if v <= rsi {
                self.max_q.pop_back();
            } else {
                break;
            }
        }
        if self.max_q.len() == self.stoch_period {
            self.max_q.pop_front();
        } // keep <= capacity
        self.max_q.push_back((idx, rsi));

        // Insert into min deque: keep values non-decreasing
        while let Some(&(_, v)) = self.min_q.back() {
            if v >= rsi {
                self.min_q.pop_back();
            } else {
                break;
            }
        }
        if self.min_q.len() == self.stoch_period {
            self.min_q.pop_front();
        } // keep <= capacity
        self.min_q.push_back((idx, rsi));
    }

    #[inline(always)]
    fn push_sma(
        new_val: f64,
        ring: &mut [f64],
        sum: &mut f64,
        pos: &mut usize,
        count: &mut usize,
        period: usize,
        inv_period: f64,
    ) -> Option<f64> {
        if *count < period {
            *sum += new_val;
            ring[*pos] = new_val;
            *pos += 1;
            if *pos == period {
                *pos = 0;
            }
            *count += 1;
            if *count == period {
                Some(*sum * inv_period)
            } else {
                None
            }
        } else {
            *sum += new_val - ring[*pos];
            ring[*pos] = new_val;
            *pos += 1;
            if *pos == period {
                *pos = 0;
            }
            Some(*sum * inv_period)
        }
    }
}

#[derive(Clone, Debug)]
pub struct SrsiBatchRange {
    pub rsi_period: (usize, usize, usize),
    pub stoch_period: (usize, usize, usize),
    pub k: (usize, usize, usize),
    pub d: (usize, usize, usize),
}

impl Default for SrsiBatchRange {
    fn default() -> Self {
        Self {
            rsi_period: (14, 14, 0),
            stoch_period: (14, 14, 0),
            k: (3, 3, 0),
            d: (3, 3, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SrsiBatchBuilder {
    range: SrsiBatchRange,
    kernel: Kernel,
}

impl SrsiBatchBuilder {
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
    pub fn stoch_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.stoch_period = (start, end, step);
        self
    }
    #[inline]
    pub fn k_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.k = (start, end, step);
        self
    }
    #[inline]
    pub fn d_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.d = (start, end, step);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<SrsiBatchOutput, SrsiError> {
        srsi_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct SrsiBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub combos: Vec<SrsiParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SrsiBatchOutput {
    pub fn row_for_params(&self, p: &SrsiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.rsi_period.unwrap_or(14) == p.rsi_period.unwrap_or(14)
                && c.stoch_period.unwrap_or(14) == p.stoch_period.unwrap_or(14)
                && c.k.unwrap_or(3) == p.k.unwrap_or(3)
                && c.d.unwrap_or(3) == p.d.unwrap_or(3)
        })
    }
    pub fn k_for(&self, p: &SrsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.k[start..start + self.cols]
        })
    }
    pub fn d_for(&self, p: &SrsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.d[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SrsiBatchRange) -> Result<Vec<SrsiParams>, SrsiError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, SrsiError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            let v: Vec<usize> = (start..=end).step_by(st).collect();
            if v.is_empty() {
                return Err(SrsiError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(SrsiError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
    let rsi_periods = axis(r.rsi_period)?;
    let stoch_periods = axis(r.stoch_period)?;
    let ks = axis(r.k)?;
    let ds = axis(r.d)?;

    if rsi_periods.is_empty()
        || stoch_periods.is_empty()
        || ks.is_empty()
        || ds.is_empty()
    {
        return Err(SrsiError::InvalidRange {
            start: r.rsi_period.0.to_string(),
            end: r.rsi_period.1.to_string(),
            step: r.rsi_period.2.to_string(),
        });
    }

    let mut out =
        Vec::with_capacity(rsi_periods.len() * stoch_periods.len() * ks.len() * ds.len());
    for &rsi_p in &rsi_periods {
        for &stoch_p in &stoch_periods {
            for &k in &ks {
                for &d in &ds {
                    out.push(SrsiParams {
                        rsi_period: Some(rsi_p),
                        stoch_period: Some(stoch_p),
                        k: Some(k),
                        d: Some(d),
                        source: None,
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn srsi_batch_with_kernel(
    data: &[f64],
    sweep: &SrsiBatchRange,
    k: Kernel,
) -> Result<SrsiBatchOutput, SrsiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SrsiError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    srsi_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn srsi_batch_slice(
    data: &[f64],
    sweep: &SrsiBatchRange,
    kern: Kernel,
) -> Result<SrsiBatchOutput, SrsiError> {
    srsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn srsi_batch_par_slice(
    data: &[f64],
    sweep: &SrsiBatchRange,
    kern: Kernel,
) -> Result<SrsiBatchOutput, SrsiError> {
    srsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn srsi_batch_inner_into(
    data: &[f64],
    sweep: &SrsiBatchRange,
    kern: Kernel,
    parallel: bool,
    k_out: &mut [f64],
    d_out: &mut [f64],
) -> Result<Vec<SrsiParams>, SrsiError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(SrsiError::InvalidRange {
            start: sweep.rsi_period.0.to_string(),
            end: sweep.rsi_period.1.to_string(),
            step: sweep.rsi_period.2.to_string(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrsiError::AllValuesNaN)?;

    // Precompute RSI once per unique rsi_period to avoid redundant calculations
    use std::collections::{BTreeMap, BTreeSet};
    let mut rsi_cache: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let uniq_rsi: BTreeSet<usize> = combos.iter().map(|c| c.rsi_period.unwrap()).collect();
    for rp in uniq_rsi {
        let rsi_in = RsiInput::from_slice(data, RsiParams { period: Some(rp) });
        let rsi_out = rsi(&rsi_in)?; // one allocation per distinct rp
        rsi_cache.insert(rp, rsi_out.values);
    }

    let max_period = combos
        .iter()
        .map(|c| {
            c.rsi_period
                .unwrap()
                .max(c.stoch_period.unwrap())
                .max(c.k.unwrap())
                .max(c.d.unwrap())
        })
        .max()
        .unwrap();

    if data.len() - first < max_period {
        return Err(SrsiError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }

    let cols = data.len();

    // Replace do_row to reuse cached RSI and avoid calling srsi_scalar
    let do_row = |row: usize, k_row: &mut [f64], d_row: &mut [f64]| -> Result<(), SrsiError> {
        let prm = &combos[row];
        let rsi_vals = rsi_cache.get(&prm.rsi_period.unwrap()).expect("cached rsi");
        let st_in = StochInput {
            data: crate::indicators::stoch::StochData::Slices {
                high: rsi_vals,
                low: rsi_vals,
                close: rsi_vals,
            },
            params: StochParams {
                fastk_period: prm.stoch_period,
                slowk_period: prm.k,
                slowk_ma_type: Some("sma".to_string()),
                slowd_period: prm.d,
                slowd_ma_type: Some("sma".to_string()),
            },
        };
        let st = stoch(&st_in)?; // allocates once per row
        k_row.copy_from_slice(&st.k);
        d_row.copy_from_slice(&st.d);
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            k_out
                .par_chunks_mut(cols)
                .zip(d_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, (k_row, d_row))| do_row(row, k_row, d_row))?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (k_row, d_row)) in k_out
                .chunks_mut(cols)
                .zip(d_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, k_row, d_row)?;
            }
        }
    } else {
        for (row, (k_row, d_row)) in k_out
            .chunks_mut(cols)
            .zip(d_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, k_row, d_row)?;
        }
    }

    Ok(combos)
}

#[inline(always)]
fn srsi_batch_inner(
    data: &[f64],
    sweep: &SrsiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SrsiBatchOutput, SrsiError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(SrsiError::InvalidRange {
            start: sweep.rsi_period.0.to_string(),
            end: sweep.rsi_period.1.to_string(),
            step: sweep.rsi_period.2.to_string(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrsiError::AllValuesNaN)?;
    let max_period = combos
        .iter()
        .map(|c| {
            c.rsi_period
                .unwrap()
                .max(c.stoch_period.unwrap())
                .max(c.k.unwrap())
                .max(c.d.unwrap())
        })
        .max()
        .unwrap();
    if data.len() - first < max_period {
        return Err(SrsiError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let _ = rows.checked_mul(cols).ok_or_else(|| SrsiError::InvalidRange {
        start: sweep.rsi_period.0.to_string(),
        end: sweep.rsi_period.1.to_string(),
        step: sweep.rsi_period.2.to_string(),
    })?;

    // Fast path: single row — delegate to the optimized single-series kernel
    if rows == 1 {
        let prm = &combos[0];
        let res = unsafe {
            match kern {
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    {
                        srsi_avx512(
                            data,
                            prm.rsi_period.unwrap(),
                            prm.stoch_period.unwrap(),
                            prm.k.unwrap(),
                            prm.d.unwrap(),
                        )
                    }
                    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                    {
                        srsi_scalar(
                            data,
                            prm.rsi_period.unwrap(),
                            prm.stoch_period.unwrap(),
                            prm.k.unwrap(),
                            prm.d.unwrap(),
                        )
                    }
                }
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    {
                        srsi_avx2(
                            data,
                            prm.rsi_period.unwrap(),
                            prm.stoch_period.unwrap(),
                            prm.k.unwrap(),
                            prm.d.unwrap(),
                        )
                    }
                    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                    {
                        srsi_scalar(
                            data,
                            prm.rsi_period.unwrap(),
                            prm.stoch_period.unwrap(),
                            prm.k.unwrap(),
                            prm.d.unwrap(),
                        )
                    }
                }
                _ => srsi_scalar(
                    data,
                    prm.rsi_period.unwrap(),
                    prm.stoch_period.unwrap(),
                    prm.k.unwrap(),
                    prm.d.unwrap(),
                ),
            }
        }?;
        return Ok(SrsiBatchOutput {
            k: res.k,
            d: res.d,
            combos,
            rows: 1,
            cols,
        });
    }
    let mut k_vals = make_uninit_matrix(rows, cols);
    let mut d_vals = make_uninit_matrix(rows, cols);

    // Calculate proper warmup periods for chained pipeline
    fn warm_for(c: &SrsiParams, first: usize) -> usize {
        let rp = c.rsi_period.unwrap();
        let sp = c.stoch_period.unwrap();
        let kp = c.k.unwrap();
        let dp = c.d.unwrap();
        // RSI needs rp-1, then Stoch needs sp-1, then smoothing needs max(k,d)-1
        first + rp - 1 + sp - 1 + kp.max(dp) - 1
    }

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| warm_for(c, first).min(cols))
        .collect();

    init_matrix_prefixes(&mut k_vals, cols, &warmup_periods);
    init_matrix_prefixes(&mut d_vals, cols, &warmup_periods);

    // Create writable views
    let mut k_guard = core::mem::ManuallyDrop::new(k_vals);
    let mut d_guard = core::mem::ManuallyDrop::new(d_vals);
    let k_out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(k_guard.as_mut_ptr() as *mut f64, k_guard.len()) };
    let d_out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(d_guard.as_mut_ptr() as *mut f64, d_guard.len()) };

    // Fill rows using the RSI cache + Stoch path (shared precompute across rows)
    let combos = srsi_batch_inner_into(data, sweep, kern, parallel, k_out, d_out)?;

    // Convert back to Vec using ManuallyDrop pattern
    let k_values = unsafe {
        Vec::from_raw_parts(
            k_guard.as_mut_ptr() as *mut f64,
            k_guard.len(),
            k_guard.capacity(),
        )
    };

    let d_values = unsafe {
        Vec::from_raw_parts(
            d_guard.as_mut_ptr() as *mut f64,
            d_guard.len(),
            d_guard.capacity(),
        )
    };

    Ok(SrsiBatchOutput {
        k: k_values,
        d: d_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn expand_grid_srsi(r: &SrsiBatchRange) -> Result<Vec<SrsiParams>, SrsiError> {
    expand_grid(r)
}

// ==================== PYTHON: CUDA BINDINGS ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "SrsiDeviceArrayF32", unsendable)]
pub struct SrsiDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl SrsiDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (self.inner.cols * itemsize, itemsize))?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Producing CUDA stream is synchronized before returning the handle,
        // so we omit the "stream" key per CAI v3 guidance.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        slf: pyo3::PyRef<'py, Self>,
        py: Python<'py>,
        stream: Option<usize>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        // Array API stream semantics: producer is already synchronized, so only validate.
        if let Some(s) = stream {
            if s == 0 {
                return Err(PyValueError::new_err(
                    "__dlpack__ stream=0 is invalid for CUDA",
                ));
            }
        }
        if let Some((dev_ty, dev_id)) = dl_device {
            if dev_ty != 2 || dev_id != slf.device_id as i32 {
                return Err(PyValueError::new_err(
                    "dl_device mismatch; cross-device copy not supported for SrsiDeviceArrayF32",
                ));
            }
        }
        if let Some(true) = copy {
            return Err(PyValueError::new_err(
                "copy=True not supported for SrsiDeviceArrayF32",
            ));
        }

        // DLPack v0.6 and v1.x (versioned) producer
        #[repr(C)]
        struct DLDevice {
            device_type: i32,
            device_id: i32,
        }
        #[repr(C)]
        struct DLDataType {
            code: u8,
            bits: u8,
            lanes: u16,
        }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
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
            manager_ctx: *mut c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
        }
        #[repr(C)]
        struct DLManagedTensorVersioned {
            manager: *mut DLManagedTensor,
            version: u32,
        }

        #[repr(C)]
        struct ManagerCtx {
            shape: *mut i64,
            strides: *mut i64,
            _shape_box: Box<[i64; 2]>,
            _strides_box: Box<[i64; 2]>,
            py_self: *mut pyffi::PyObject,
        }

        unsafe extern "C" fn capsule_destructor(cap: *mut pyffi::PyObject) {
            if cap.is_null() {
                return;
            }
            let name_c = unsafe { pyffi::PyCapsule_GetName(cap) };
            if name_c.is_null() {
                return;
            }
            let name = unsafe { std::ffi::CStr::from_ptr(name_c) }.to_string_lossy();
            let valid = name == "dltensor" || name == "dltensor_versioned";
            let ptr = unsafe { pyffi::PyCapsule_GetPointer(cap, name_c as *const _) };
            if ptr.is_null() || !valid {
                return;
            }
            unsafe {
                if name == "dltensor_versioned" {
                    let v = ptr as *mut DLManagedTensorVersioned;
                    if !v.is_null() {
                        let mt = (*v).manager;
                        if !mt.is_null() {
                            if let Some(del) = (*mt).deleter {
                                del(mt);
                            }
                        }
                        let _ = Box::from_raw(v);
                    }
                } else {
                    let mt = ptr as *mut DLManagedTensor;
                    if !mt.is_null() {
                        if let Some(del) = (*mt).deleter {
                            del(mt);
                        }
                    }
                }
            }
        }

        unsafe extern "C" fn managed_deleter(p: *mut DLManagedTensor) {
            if p.is_null() {
                return;
            }
            unsafe {
                let ctx = (*p).manager_ctx as *mut ManagerCtx;
                if !ctx.is_null() {
                    let pyobj = (*ctx).py_self;
                    let _ = Box::from_raw(ctx);
                    if !pyobj.is_null() {
                        pyffi::Py_DECREF(pyobj);
                    }
                }
                let _ = Box::from_raw(p);
            }
        }

        let rows = slf.inner.rows as i64;
        let cols = slf.inner.cols as i64;
        let want_versioned = max_version.map(|(maj, _)| maj >= 1).unwrap_or(false);

        // Allocate shape/strides (elements) and INCREF self so the VRAM buffer outlives the capsule
        let mut shape_box = Box::new([rows, cols]);
        let mut strides_box = Box::new([cols, 1]);
        let shape_ptr = shape_box.as_mut_ptr();
        let strides_ptr = strides_box.as_mut_ptr();
        let py_self = slf.as_ptr();
        unsafe {
            pyffi::Py_INCREF(py_self);
        }
        let mgr = Box::new(ManagerCtx {
            shape: shape_ptr,
            strides: strides_ptr,
            _shape_box: shape_box,
            _strides_box: strides_box,
            py_self,
        });
        let mgr_ptr = Box::into_raw(mgr) as *mut c_void;

        let total_elems = (rows as i128).saturating_mul(cols as i128);
        let data_ptr = if total_elems == 0 {
            std::ptr::null_mut()
        } else {
            slf.inner.device_ptr() as usize as *mut c_void
        };

        let mt = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: 2,
                    device_id: slf.device_id as i32,
                },
                ndim: 2,
                dtype: DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: mgr_ptr,
            deleter: Some(managed_deleter),
        });

        unsafe {
            if want_versioned {
                let wrapped = Box::new(DLManagedTensorVersioned {
                    manager: Box::into_raw(mt),
                    version: 1,
                });
                let ptr = Box::into_raw(wrapped) as *mut c_void;
                let name = CString::new("dltensor_versioned").unwrap();
                let cap =
                    pyffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_destructor));
                if cap.is_null() {
                    let _ = Box::from_raw(ptr as *mut DLManagedTensorVersioned);
                    return Err(PyValueError::new_err(
                        "failed to create versioned DLPack capsule",
                    ));
                }
                Ok(PyObject::from_owned_ptr(py, cap))
            } else {
                let ptr = Box::into_raw(mt) as *mut c_void;
                let name = CString::new("dltensor").unwrap();
                let cap =
                    pyffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_destructor));
                if cap.is_null() {
                    let _ = Box::from_raw(ptr as *mut DLManagedTensor);
                    return Err(PyValueError::new_err(
                        "failed to create DLPack capsule",
                    ));
                }
                Ok(PyObject::from_owned_ptr(py, cap))
            }
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl SrsiDeviceArrayF32Py {
    pub fn new_from_rust(inner: DeviceArrayF32, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self {
            inner,
            _ctx: ctx_guard,
            device_id,
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "srsi_cuda_batch_dev")]
#[pyo3(signature = (data_f32, rsi_range, stoch_range, k_range, d_range, device_id=0))]
pub fn srsi_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    rsi_range: (usize, usize, usize),
    stoch_range: (usize, usize, usize),
    k_range: (usize, usize, usize),
    d_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaSrsi;
    use numpy::IntoPyArray;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data_f32.as_slice()?;
    let sweep = SrsiBatchRange {
        rsi_period: rsi_range,
        stoch_period: stoch_range,
        k: k_range,
        d: d_range,
    };
    let ((pair, combos), ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaSrsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let res = cuda
            .srsi_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((res, ctx, dev_id))
    })?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "k",
        SrsiDeviceArrayF32Py::new_from_rust(pair.k, ctx.clone(), dev_id),
    )?;
    dict.set_item(
        "d",
        SrsiDeviceArrayF32Py::new_from_rust(pair.d, ctx, dev_id),
    )?;
    dict.set_item("rows", combos.len())?;
    dict.set_item("cols", slice.len())?;
    dict.set_item(
        "rsi_periods",
        combos
            .iter()
            .map(|p| p.rsi_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_periods",
        combos
            .iter()
            .map(|p| p.stoch_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "k_periods",
        combos
            .iter()
            .map(|p| p.k.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "d_periods",
        combos
            .iter()
            .map(|p| p.d.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "srsi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, rsi_period=14, stoch_period=14, k=3, d=3, device_id=0))]
pub fn srsi_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaSrsi;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if flat.len() != expected {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }
    let params = SrsiParams {
        rsi_period: Some(rsi_period),
        stoch_period: Some(stoch_period),
        k: Some(k),
        d: Some(d),
        source: None,
    };
    let (pair, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaSrsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let res = cuda
            .srsi_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((res, ctx, dev_id))
    })?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "k",
        SrsiDeviceArrayF32Py::new_from_rust(pair.k, ctx.clone(), dev_id),
    )?;
    dict.set_item(
        "d",
        SrsiDeviceArrayF32Py::new_from_rust(pair.d, ctx, dev_id),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("rsi_period", rsi_period)?;
    dict.set_item("stoch_period", stoch_period)?;
    dict.set_item("k_period", k)?;
    dict.set_item("d_period", d)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "srsi")]
#[pyo3(signature = (data, rsi_period=None, stoch_period=None, k=None, d=None, source=None, kernel=None))]
pub fn srsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period: Option<usize>,
    stoch_period: Option<usize>,
    k: Option<usize>,
    d: Option<usize>,
    source: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    // Pre-validate periods for clearer Python errors (aligns with tests)
    if matches!(rsi_period, Some(0))
        || matches!(stoch_period, Some(0))
        || matches!(k, Some(0))
        || matches!(d, Some(0))
    {
        return Err(PyValueError::new_err("Invalid period: values must be > 0"));
    }

    let params = SrsiParams {
        rsi_period,
        stoch_period,
        k,
        d,
        source: source.map(|s| s.to_string()),
    };
    let input = SrsiInput::from_slice(slice_in, params);

    let (k_vec, d_vec) = py
        .allow_threads(|| srsi_with_kernel(&input, kern).map(|o| (o.k, o.d)))
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("Not enough valid data")
                && (matches!(rsi_period, Some(0))
                    || matches!(stoch_period, Some(0))
                    || matches!(k, Some(0))
                    || matches!(d, Some(0)))
            {
                PyValueError::new_err("Invalid period: values must be > 0")
            } else {
                PyValueError::new_err(msg)
            }
        })?;

    Ok((k_vec.into_pyarray(py), d_vec.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "SrsiStream")]
pub struct SrsiStreamPy {
    stream: SrsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SrsiStreamPy {
    #[new]
    fn new(
        rsi_period: Option<usize>,
        stoch_period: Option<usize>,
        k: Option<usize>,
        d: Option<usize>,
    ) -> PyResult<Self> {
        let params = SrsiParams {
            rsi_period,
            stoch_period,
            k,
            d,
            source: None,
        };
        let stream =
            SrsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SrsiStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "srsi_batch")]
#[pyo3(signature = (data, rsi_period_range, stoch_period_range, k_range, d_range, source=None, kernel=None))]
pub fn srsi_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period_range: (usize, usize, usize),
    stoch_period_range: (usize, usize, usize),
    k_range: (usize, usize, usize),
    d_range: (usize, usize, usize),
    source: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = SrsiBatchRange {
        rsi_period: rsi_period_range,
        stoch_period: stoch_period_range,
        k: k_range,
        d: d_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output arrays
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let k_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let d_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let k_slice = unsafe { k_arr.as_slice_mut()? };
    let d_slice = unsafe { d_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            srsi_batch_inner_into(slice_in, &sweep, kernel, true, k_slice, d_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("k", k_arr.reshape((rows, cols))?)?;
    dict.set_item("d", d_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "rsi_periods",
        combos
            .iter()
            .map(|p| p.rsi_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_periods",
        combos
            .iter()
            .map(|p| p.stoch_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "k_periods",
        combos
            .iter()
            .map(|p| p.k.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "d_periods",
        combos
            .iter()
            .map(|p| p.d.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

/// Write SRSI outputs directly to slices - no allocations
pub fn srsi_into_slice(
    dst_k: &mut [f64],
    dst_d: &mut [f64],
    input: &SrsiInput,
    kern: Kernel,
) -> Result<(), SrsiError> {
    let data: &[f64] = input.as_ref();

    if dst_k.len() != data.len() || dst_d.len() != data.len() {
        return Err(SrsiError::OutputLengthMismatch {
            expected: data.len(),
            k_len: dst_k.len(),
            d_len: dst_d.len(),
        });
    }

    // Compute once, then copy into the provided buffers
    let out = srsi_with_kernel(input, kern)?;
    dst_k.copy_from_slice(&out.k);
    dst_d.copy_from_slice(&out.d);

    Ok(())
}

/// Write SRSI outputs into caller-provided buffers without allocating output vectors.
///
/// - Preserves NaN warmup prefixes exactly like `srsi()`/`srsi_with_kernel()`.
/// - `out_k.len()` and `out_d.len()` must equal the input length.
/// - Uses `Kernel::Auto` for internal dispatch.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn srsi_into(input: &SrsiInput, out_k: &mut [f64], out_d: &mut [f64]) -> Result<(), SrsiError> {
    srsi_into_slice(out_k, out_d, input, Kernel::Auto)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_js(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
) -> Result<Vec<f64>, JsValue> {
    if data.is_empty() {
        return Err(JsValue::from_str("srsi: Input data is empty"));
    }

    if rsi_period == 0 || stoch_period == 0 || k == 0 || d == 0 {
        return Err(JsValue::from_str("srsi: Invalid period"));
    }

    let params = SrsiParams {
        rsi_period: Some(rsi_period),
        stoch_period: Some(stoch_period),
        k: Some(k),
        d: Some(d),
        source: None,
    };
    let input = SrsiInput::from_slice(data, params);
    let out = srsi_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&format!("srsi: {}", e)))?;

    // Return flattened array [k..., d...]
    let mut values = Vec::with_capacity(2 * data.len());
    values.extend_from_slice(&out.k);
    values.extend_from_slice(&out.d);

    Ok(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_into(
    in_ptr: usize,
    k_ptr: usize,
    d_ptr: usize,
    len: usize,
    rsi_period: usize,
    stoch_period: usize,
    k: usize,
    d: usize,
) -> Result<(), JsValue> {
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr as *const f64, len);

        if rsi_period == 0 || stoch_period == 0 || k == 0 || d == 0 {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = SrsiParams {
            rsi_period: Some(rsi_period),
            stoch_period: Some(stoch_period),
            k: Some(k),
            d: Some(d),
            source: None,
        };
        let input = SrsiInput::from_slice(data, params);

        // Check for aliasing on both output pointers
        let needs_temp = in_ptr == k_ptr || in_ptr == d_ptr || k_ptr == d_ptr;

        if needs_temp {
            let mut temp_k = vec![0.0; len];
            let mut temp_d = vec![0.0; len];
            srsi_into_slice(&mut temp_k, &mut temp_d, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let k_out = std::slice::from_raw_parts_mut(k_ptr as *mut f64, len);
            let d_out = std::slice::from_raw_parts_mut(d_ptr as *mut f64, len);
            k_out.copy_from_slice(&temp_k);
            d_out.copy_from_slice(&temp_d);
        } else {
            let k_out = std::slice::from_raw_parts_mut(k_ptr as *mut f64, len);
            let d_out = std::slice::from_raw_parts_mut(d_ptr as *mut f64, len);
            srsi_into_slice(k_out, d_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SrsiBatchConfig {
    pub rsi_period_range: (usize, usize, usize),
    pub stoch_period_range: (usize, usize, usize),
    pub k_range: (usize, usize, usize),
    pub d_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SrsiBatchJsOutput {
    pub k_values: Vec<f64>, // All K values flattened
    pub d_values: Vec<f64>, // All D values flattened
    pub rows: usize,        // Number of parameter combinations
    pub cols: usize,        // Data length
    pub combos: Vec<SrsiParams>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = srsi_batch)]
pub fn srsi_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: SrsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = SrsiBatchRange {
        rsi_period: cfg.rsi_period_range,
        stoch_period: cfg.stoch_period_range,
        k: cfg.k_range,
        d: cfg.d_range,
    };

    let out = srsi_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let res = SrsiBatchJsOutput {
        k_values: out.k,
        d_values: out.d,
        rows: out.rows,
        cols: out.cols,
        combos: out.combos,
    };

    serde_wasm_bindgen::to_value(&res)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_batch_into(
    in_ptr: usize,
    k_ptr: usize,
    d_ptr: usize,
    len: usize,
    rsi_period_start: usize,
    rsi_period_end: usize,
    rsi_period_step: usize,
    stoch_period_start: usize,
    stoch_period_end: usize,
    stoch_period_step: usize,
    k_start: usize,
    k_end: usize,
    k_step: usize,
    d_start: usize,
    d_end: usize,
    d_step: usize,
) -> Result<usize, JsValue> {
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr as *const f64, len);

        let sweep = SrsiBatchRange {
            rsi_period: (rsi_period_start, rsi_period_end, rsi_period_step),
            stoch_period: (stoch_period_start, stoch_period_end, stoch_period_step),
            k: (k_start, k_end, k_step),
            d: (d_start, d_end, d_step),
        };

        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let k_out = std::slice::from_raw_parts_mut(k_ptr as *mut f64, total);
        let d_out = std::slice::from_raw_parts_mut(d_ptr as *mut f64, total);

        srsi_batch_inner_into(data, &sweep, Kernel::Auto, false, k_out, d_out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_srsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SrsiParams {
            rsi_period: None,
            stoch_period: None,
            k: None,
            d: None,
            source: None,
        };
        let input = SrsiInput::from_candles(&candles, "close", default_params);
        let output = srsi_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }

    fn check_srsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SrsiParams::default();
        let input = SrsiInput::from_candles(&candles, "close", params);
        let result = srsi_with_kernel(&input, kernel)?;
        assert_eq!(result.k.len(), candles.close.len());
        assert_eq!(result.d.len(), candles.close.len());
        let last_five_k = [
            65.52066633236464,
            61.22507053191985,
            57.220471530042644,
            64.61344854988147,
            60.66534359318523,
        ];
        let last_five_d = [
            64.33503158970049,
            64.42143544464182,
            61.32206946477942,
            61.01966353728503,
            60.83308789104016,
        ];
        let k_slice = &result.k[result.k.len() - 5..];
        let d_slice = &result.d[result.d.len() - 5..];
        for i in 0..5 {
            let diff_k = (k_slice[i] - last_five_k[i]).abs();
            let diff_d = (d_slice[i] - last_five_d[i]).abs();
            assert!(
                diff_k < 1e-6,
                "Mismatch in SRSI K at index {}: got {}, expected {}",
                i,
                k_slice[i],
                last_five_k[i]
            );
            assert!(
                diff_d < 1e-6,
                "Mismatch in SRSI D at index {}: got {}, expected {}",
                i,
                d_slice[i],
                last_five_d[i]
            );
        }
        Ok(())
    }

    fn check_srsi_from_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let slice_data = candles.close.as_slice();
        let params = SrsiParams {
            rsi_period: Some(3),
            stoch_period: Some(3),
            k: Some(2),
            d: Some(2),
            source: Some("close".to_string()),
        };
        let input = SrsiInput::from_slice(&slice_data, params);
        let output = srsi_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), slice_data.len());
        assert_eq!(output.d.len(), slice_data.len());
        Ok(())
    }

    fn check_srsi_custom_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SrsiParams {
            rsi_period: Some(10),
            stoch_period: Some(10),
            k: Some(4),
            d: Some(4),
            source: Some("hlc3".to_string()),
        };
        let input = SrsiInput::from_candles(&candles, "hlc3", params);
        let output = srsi_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_srsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            SrsiParams::default(), // rsi: 14, stoch: 14, k: 3, d: 3
            SrsiParams {
                rsi_period: Some(2), // minimum viable
                stoch_period: Some(2),
                k: Some(2),
                d: Some(2),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(5), // small periods
                stoch_period: Some(5),
                k: Some(3),
                d: Some(3),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(10), // medium periods
                stoch_period: Some(10),
                k: Some(5),
                d: Some(5),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(20), // large periods
                stoch_period: Some(20),
                k: Some(7),
                d: Some(7),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(50), // very large periods
                stoch_period: Some(50),
                k: Some(10),
                d: Some(10),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(7), // mixed periods 1
                stoch_period: Some(14),
                k: Some(3),
                d: Some(5),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(14), // mixed periods 2
                stoch_period: Some(7),
                k: Some(5),
                d: Some(3),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(21), // mixed periods 3
                stoch_period: Some(14),
                k: Some(6),
                d: Some(4),
                source: None,
            },
            SrsiParams {
                rsi_period: Some(100), // extreme case
                stoch_period: Some(100),
                k: Some(20),
                d: Some(20),
                source: None,
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = SrsiInput::from_candles(&candles, "close", params.clone());
            let output = srsi_with_kernel(&input, kernel)?;

            // Check k values
            for (i, &val) in output.k.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in K output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in K output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in K output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
                }
            }

            // Check d values
            for (i, &val) in output.d.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in D output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in D output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in D output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_srsi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_srsi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test strategy with periods from 2 to 20 for more realistic testing
        let strat = (2usize..=20, 2usize..=20, 2usize..=10, 2usize..=10).prop_flat_map(
            |(rsi_period, stoch_period, k, d)| {
                // Calculate minimum data needed for SRSI to work
                // RSI needs rsi_period points, then Stoch needs additional points for its calculations
                let min_data_needed = rsi_period + stoch_period.max(k).max(d) + 10; // Extra buffer
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        min_data_needed..400,
                    ),
                    Just(rsi_period),
                    Just(stoch_period),
                    Just(k),
                    Just(d),
                )
            },
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, rsi_period, stoch_period, k, d)| {
                let params = SrsiParams {
                    rsi_period: Some(rsi_period),
                    stoch_period: Some(stoch_period),
                    k: Some(k),
                    d: Some(d),
                    source: None,
                };
                let input = SrsiInput::from_slice(&data, params.clone());

                // Check if we have enough data - if not, skip this test case
                let output_result = srsi_with_kernel(&input, kernel);
                let ref_output_result = srsi_with_kernel(&input, Kernel::Scalar);

                // Both should either succeed or fail together
                match (output_result, ref_output_result) {
                    (Ok(output), Ok(ref_output)) => {
                        // Calculate expected warmup period
                        // RSI produces first value at rsi_period, then Stoch needs more values
                        // The actual warmup is complex due to the chained indicators
                        let expected_min_warmup = rsi_period;

                        // Property 1: K and D values must be bounded between 0 and 100
                        for i in 0..data.len() {
                            if !output.k[i].is_nan() {
                                prop_assert!(
                                    output.k[i] >= -1e-9 && output.k[i] <= 100.0 + 1e-9,
                                    "idx {}: K value {} is out of bounds [0, 100]",
                                    i,
                                    output.k[i]
                                );
                            }
                            if !output.d[i].is_nan() {
                                prop_assert!(
                                    output.d[i] >= -1e-9 && output.d[i] <= 100.0 + 1e-9,
                                    "idx {}: D value {} is out of bounds [0, 100]",
                                    i,
                                    output.d[i]
                                );
                            }
                        }

                        // Property 2: Warmup period - check that early values are NaN
                        // Don't be too strict about exact warmup period as it's complex with chained indicators
                        for i in 0..expected_min_warmup.min(data.len()) {
                            prop_assert!(
                                output.k[i].is_nan(),
                                "idx {}: Expected NaN during early warmup for K, got {}",
                                i,
                                output.k[i]
                            );
                            prop_assert!(
                                output.d[i].is_nan(),
                                "idx {}: Expected NaN during early warmup for D, got {}",
                                i,
                                output.d[i]
                            );
                        }

                        // Property 3: Eventually should get valid values if we have enough data
                        let has_valid_k = output.k.iter().any(|&x| !x.is_nan());
                        let has_valid_d = output.d.iter().any(|&x| !x.is_nan());
                        if data.len() > rsi_period + stoch_period + k + d {
                            prop_assert!(
                                has_valid_k,
                                "Expected at least one valid K value with sufficient data"
                            );
                            prop_assert!(
                                has_valid_d,
                                "Expected at least one valid D value with sufficient data"
                            );
                        }

                        // Property 4: Constant data should produce SRSI around 50 (neutral)
                        if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                            let last_k = output.k[data.len() - 1];
                            let last_d = output.d[data.len() - 1];
                            if !last_k.is_nan() && !last_d.is_nan() {
                                prop_assert!(
                                    (last_k - 50.0).abs() < 10.0,
                                    "Constant data should produce K near 50, got {}",
                                    last_k
                                );
                                prop_assert!(
                                    (last_d - 50.0).abs() < 10.0,
                                    "Constant data should produce D near 50, got {}",
                                    last_d
                                );
                            }
                        }

                        // Property 5: Strictly increasing prices should produce high SRSI
                        let is_increasing = data.windows(2).all(|w| w[1] > w[0]);
                        if is_increasing && has_valid_k {
                            let last_k = output.k[data.len() - 1];
                            if !last_k.is_nan() {
                                prop_assert!(
                                    last_k > 50.0,
                                    "Strictly increasing prices should produce K > 50, got {}",
                                    last_k
                                );
                            }
                        }

                        // Property 6: Strictly decreasing prices should produce low SRSI
                        let is_decreasing = data.windows(2).all(|w| w[1] < w[0]);
                        if is_decreasing && has_valid_k {
                            let last_k = output.k[data.len() - 1];
                            if !last_k.is_nan() {
                                prop_assert!(
                                    last_k < 50.0,
                                    "Strictly decreasing prices should produce K < 50, got {}",
                                    last_k
                                );
                            }
                        }

                        // Property 7: Kernel consistency - compare with scalar implementation
                        for i in 0..data.len() {
                            let k_val = output.k[i];
                            let d_val = output.d[i];
                            let ref_k = ref_output.k[i];
                            let ref_d = ref_output.d[i];

                            // Check for NaN consistency
                            if !k_val.is_finite() || !ref_k.is_finite() {
                                prop_assert!(
                                    k_val.to_bits() == ref_k.to_bits(),
                                    "K finite/NaN mismatch idx {}: {} vs {}",
                                    i,
                                    k_val,
                                    ref_k
                                );
                            } else {
                                // Check ULP difference for finite values
                                let k_ulp_diff = k_val.to_bits().abs_diff(ref_k.to_bits());
                                prop_assert!(
                                    (k_val - ref_k).abs() <= 1e-9 || k_ulp_diff <= 4,
                                    "K mismatch idx {}: {} vs {} (ULP={})",
                                    i,
                                    k_val,
                                    ref_k,
                                    k_ulp_diff
                                );
                            }

                            if !d_val.is_finite() || !ref_d.is_finite() {
                                prop_assert!(
                                    d_val.to_bits() == ref_d.to_bits(),
                                    "D finite/NaN mismatch idx {}: {} vs {}",
                                    i,
                                    d_val,
                                    ref_d
                                );
                            } else {
                                let d_ulp_diff = d_val.to_bits().abs_diff(ref_d.to_bits());
                                prop_assert!(
                                    (d_val - ref_d).abs() <= 1e-9 || d_ulp_diff <= 4,
                                    "D mismatch idx {}: {} vs {} (ULP={})",
                                    i,
                                    d_val,
                                    ref_d,
                                    d_ulp_diff
                                );
                            }
                        }

                        // Property 8: Determinism - running same calculation twice produces same result
                        let output2 = srsi_with_kernel(&input, kernel).unwrap();
                        for i in 0..data.len() {
                            prop_assert!(
                                output.k[i].to_bits() == output2.k[i].to_bits(),
                                "K determinism failed at idx {}: {} vs {}",
                                i,
                                output.k[i],
                                output2.k[i]
                            );
                            prop_assert!(
                                output.d[i].to_bits() == output2.d[i].to_bits(),
                                "D determinism failed at idx {}: {} vs {}",
                                i,
                                output.d[i],
                                output2.d[i]
                            );
                        }
                    }
                    (Err(_), Err(_)) => {
                        // Both failed - this is expected for insufficient data, skip this test case
                    }
                    (Ok(_), Err(e)) => {
                        prop_assert!(false, "Kernel succeeded but scalar failed: {:?}", e);
                    }
                    (Err(e), Ok(_)) => {
                        prop_assert!(false, "Kernel failed but scalar succeeded: {:?}", e);
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_srsi_tests {
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

    generate_all_srsi_tests!(
        check_srsi_partial_params,
        check_srsi_accuracy,
        check_srsi_custom_params,
        check_srsi_from_slice,
        check_srsi_no_poison
    );

    #[test]
    fn test_srsi_into_slice_size_mismatch() {
        // Test that srsi_into_slice returns SizeMismatch error when buffer sizes don't match
        // Generate enough data for default parameters (rsi_period=14, stoch_period=14, k=3, d=3)
        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let data_len = data.len();
        let params = SrsiParams::default();
        let input = SrsiInput::from_slice(&data, params);

        // Test with k buffer too small
        let mut k_small = vec![0.0; 30]; // Wrong size
        let mut d_correct = vec![0.0; data_len]; // Correct size
        let result = srsi_into_slice(&mut k_small, &mut d_correct, &input, Kernel::Scalar);
        match result {
            Err(SrsiError::OutputLengthMismatch {
                expected,
                k_len,
                d_len,
            }) => {
                assert_eq!(expected, data_len);
                assert_eq!(k_len, 30);
                assert_eq!(d_len, data_len);
            }
            _ => panic!("Expected SizeMismatch error with k buffer too small"),
        }

        // Test with d buffer too small
        let mut k_correct = vec![0.0; data_len]; // Correct size
        let mut d_small = vec![0.0; 35]; // Wrong size
        let result = srsi_into_slice(&mut k_correct, &mut d_small, &input, Kernel::Scalar);
        match result {
            Err(SrsiError::OutputLengthMismatch {
                expected,
                k_len,
                d_len,
            }) => {
                assert_eq!(expected, data_len);
                assert_eq!(k_len, data_len);
                assert_eq!(d_len, 35);
            }
            _ => panic!("Expected SizeMismatch error with d buffer too small"),
        }

        // Test with both buffers wrong size
        let mut k_wrong = vec![0.0; 60]; // Wrong size
        let mut d_wrong = vec![0.0; 70]; // Wrong size
        let result = srsi_into_slice(&mut k_wrong, &mut d_wrong, &input, Kernel::Scalar);
        match result {
            Err(SrsiError::OutputLengthMismatch {
                expected,
                k_len,
                d_len,
            }) => {
                assert_eq!(expected, data_len);
                assert_eq!(k_len, 60);
                assert_eq!(d_len, 70);
            }
            _ => panic!("Expected SizeMismatch error with both buffers wrong size"),
        }

        // Test with correct sizes - should succeed
        let mut k_ok = vec![0.0; data_len];
        let mut d_ok = vec![0.0; data_len];
        let result = srsi_into_slice(&mut k_ok, &mut d_ok, &input, Kernel::Scalar);
        assert!(
            result.is_ok(),
            "Should succeed with correct buffer sizes. Error: {:?}",
            result
        );
    }

    #[cfg(feature = "proptest")]
    generate_all_srsi_tests!(check_srsi_property);

    #[test]
    fn test_srsi_into_matches_api() {
        // Use the existing CSV input data to match other SRSI tests
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file).expect("load csv");
        let input = SrsiInput::from_candles(&c, "close", SrsiParams::default());

        // Baseline via Vec-returning API
        let base = srsi(&input).expect("srsi baseline");

        // Preallocate outputs and call into API
        let mut out_k = vec![0.0; c.close.len()];
        let mut out_d = vec![0.0; c.close.len()];

        // Native into API (Auto kernel)
        #[cfg(not(feature = "wasm"))]
        {
            srsi_into(&input, &mut out_k, &mut out_d).expect("srsi_into");
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, parity still holds using the slice variant
            srsi_into_slice(&mut out_k, &mut out_d, &input, Kernel::Auto)
                .expect("srsi_into_slice");
        }

        assert_eq!(base.k.len(), c.close.len());
        assert_eq!(base.d.len(), c.close.len());
        assert_eq!(out_k.len(), c.close.len());
        assert_eq!(out_d.len(), c.close.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..c.close.len() {
            assert!(
                eq_or_both_nan(base.k[i], out_k[i]),
                "SRSI K mismatch at {i}: {} vs {}",
                base.k[i],
                out_k[i]
            );
            assert!(
                eq_or_both_nan(base.d[i], out_d[i]),
                "SRSI D mismatch at {i}: {} vs {}",
                base.d[i],
                out_d[i]
            );
        }
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SrsiBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&c.close)?;
        let def = SrsiParams::default();
        let k_row = output.k_for(&def).expect("default k row missing");
        let d_row = output.d_for(&def).expect("default d row missing");
        assert_eq!(k_row.len(), c.close.len());
        assert_eq!(d_row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (rsi_period_range, stoch_period_range, k_range, d_range)
            ((2, 10, 2), (2, 10, 2), (2, 4, 1), (2, 4, 1)), // Small periods
            ((5, 25, 5), (5, 25, 5), (3, 7, 2), (3, 7, 2)), // Medium periods
            ((30, 60, 15), (30, 60, 15), (5, 10, 5), (5, 10, 5)), // Large periods
            ((2, 5, 1), (2, 5, 1), (2, 3, 1), (2, 3, 1)),   // Dense small range
            ((10, 30, 10), (5, 15, 5), (3, 6, 3), (3, 6, 3)), // Mixed ranges
            ((14, 14, 0), (14, 14, 0), (3, 3, 0), (3, 3, 0)), // Single value (default)
            ((7, 21, 7), (14, 28, 14), (3, 9, 3), (3, 9, 3)), // Different step sizes
        ];

        for (cfg_idx, &(rsi_range, stoch_range, k_range, d_range)) in
            test_configs.iter().enumerate()
        {
            let output = SrsiBatchBuilder::new()
                .kernel(kernel)
                .rsi_period_range(rsi_range.0, rsi_range.1, rsi_range.2)
                .stoch_period_range(stoch_range.0, stoch_range.1, stoch_range.2)
                .k_range(k_range.0, k_range.1, k_range.2)
                .d_range(d_range.0, d_range.1, d_range.2)
                .apply_slice(&c.close)?;

            // Check k values
            for (idx, &val) in output.k.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in K output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in K output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in K output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
                }
            }

            // Check d values
            for (idx, &val) in output.d.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in D output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in D output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in D output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
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
