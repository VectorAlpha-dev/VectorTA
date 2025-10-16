//! # KDJ (Stochastic Oscillator with MA smoothing)
//!
//! Stochastic Oscillator with K and D lines plus additional J line.
//! J = 3 * K - 2 * D, used to highlight momentum and overbought/oversold conditions.
//!
//! ## Parameters
//! - **fast_k_period**: Window for fast stochastic (default: 9)
//! - **slow_k_period**: K smoothing period (default: 3)
//! - **slow_k_ma_type**: K smoothing MA type (default: "sma")
//! - **slow_d_period**: D smoothing period (default: 3)
//! - **slow_d_ma_type**: D smoothing MA type (default: "sma")
//!
//! ## Returns
//! - **`Ok(KdjOutput)`** on success (`k`, `d`, `j` vectors of length matching input)
//! - **`Err(KdjError)`** on failure
//!
//! ## Developer Status
//! - SIMD: Delegates to scalar. Rolling max/min via deques is sequential/branchy; SIMD shows no consistent win (>5%).
//! - Scalar: Fused single-pass for HH/LL and SMA/EMA smoothing; avoids extra temporaries and dynamic MA where possible.
//! - Batch: Reuses precomputed stochastic per unique `fast_k` across rows to cut duplicate work.
//! - Streaming: Implemented with monotonic deques (amortized O(1)); SMA/EMA smoothing matches scalar warmups
//! - Memory: Good zero-copy usage (alloc_with_nan_prefix, make_uninit_matrix)

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::RollingError;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
// AVec and CACHELINE_ALIGN removed - not needed with current implementation
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
// rayon imported inline where needed
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// ======== Input Data ========

#[derive(Debug, Clone)]
pub enum KdjData<'a> {
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
pub struct KdjInput<'a> {
    pub data: KdjData<'a>,
    pub params: KdjParams,
}

impl<'a> KdjInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: KdjParams) -> Self {
        Self {
            data: KdjData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: KdjParams,
    ) -> Self {
        Self {
            data: KdjData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, KdjParams::default())
    }
    #[inline]
    pub fn get_fast_k_period(&self) -> usize {
        self.params.fast_k_period.unwrap_or(9)
    }
    #[inline]
    pub fn get_slow_k_period(&self) -> usize {
        self.params.slow_k_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slow_k_ma_type(&self) -> &str {
        self.params.slow_k_ma_type.as_deref().unwrap_or("sma")
    }
    #[inline]
    pub fn get_slow_d_period(&self) -> usize {
        self.params.slow_d_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slow_d_ma_type(&self) -> &str {
        self.params.slow_d_ma_type.as_deref().unwrap_or("sma")
    }
}

// ======== Parameters ========

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct KdjParams {
    pub fast_k_period: Option<usize>,
    pub slow_k_period: Option<usize>,
    pub slow_k_ma_type: Option<String>,
    pub slow_d_period: Option<usize>,
    pub slow_d_ma_type: Option<String>,
}

impl Default for KdjParams {
    fn default() -> Self {
        Self {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        }
    }
}

// ======== Output ========

#[derive(Debug, Clone)]
pub struct KdjOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub j: Vec<f64>,
}

// ======== Error Type ========

#[derive(Debug, Error)]
pub enum KdjError {
    #[error("kdj: Empty data provided.")]
    EmptyData,
    #[error("kdj: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kdj: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kdj: All values are NaN.")]
    AllValuesNaN,
    #[error("kdj: Buffer size mismatch: expected = {expected}, got = {got}")]
    BufferSizeMismatch { expected: usize, got: usize },
    #[error("kdj: Rolling error {0}")]
    RollingError(#[from] RollingError),
    #[error("kdj: MA error {0}")]
    MaError(#[from] Box<dyn Error + Send + Sync>),
}

// ======== Main Function/Dispatch ========

#[inline]
pub fn kdj(input: &KdjInput) -> Result<KdjOutput, KdjError> {
    kdj_with_kernel(input, Kernel::Auto)
}

pub fn kdj_with_kernel(input: &KdjInput, kernel: Kernel) -> Result<KdjOutput, KdjError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        KdjData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        KdjData::Slices { high, low, close } => (high, low, close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(KdjError::EmptyData);
    }

    let fast_k_period = input.get_fast_k_period();
    let slow_k_period = input.get_slow_k_period();
    let slow_k_ma_type = input.get_slow_k_ma_type();
    let slow_d_period = input.get_slow_d_period();
    let slow_d_ma_type = input.get_slow_d_ma_type();

    if fast_k_period == 0 || fast_k_period > high.len() {
        return Err(KdjError::InvalidPeriod {
            period: fast_k_period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(KdjError::AllValuesNaN)?;

    if (high.len() - first_valid_idx) < fast_k_period {
        return Err(KdjError::NotEnoughValidData {
            needed: fast_k_period,
            valid: high.len() - first_valid_idx,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
                return kdj_simd128(
                    high,
                    low,
                    close,
                    fast_k_period,
                    slow_k_period,
                    slow_k_ma_type,
                    slow_d_period,
                    slow_d_ma_type,
                    first_valid_idx,
                );
            }
        }

        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kdj_scalar(
                high,
                low,
                close,
                fast_k_period,
                slow_k_period,
                slow_k_ma_type,
                slow_d_period,
                slow_d_ma_type,
                first_valid_idx,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kdj_avx2(
                high,
                low,
                close,
                fast_k_period,
                slow_k_period,
                slow_k_ma_type,
                slow_d_period,
                slow_d_ma_type,
                first_valid_idx,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kdj_avx512(
                high,
                low,
                close,
                fast_k_period,
                slow_k_period,
                slow_k_ma_type,
                slow_d_period,
                slow_d_ma_type,
                first_valid_idx,
            ),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn kdj_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // Allocate outputs up-front and compute directly into them
    let len = high.len();
    let mut k: Vec<f64> = Vec::with_capacity(len);
    let mut d: Vec<f64> = Vec::with_capacity(len);
    let mut j: Vec<f64> = Vec::with_capacity(len);
    unsafe {
        k.set_len(len);
        d.set_len(len);
        j.set_len(len);
    }

    kdj_compute_into_scalar(
        high,
        low,
        close,
        first_valid_idx,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        &mut k,
        &mut d,
        &mut j,
    )?;

    Ok(KdjOutput { k, d, j })
}

// =========== In-place Core API ===========

#[inline]
pub fn kdj_into_slices(
    k_out: &mut [f64],
    d_out: &mut [f64],
    j_out: &mut [f64],
    input: &KdjInput,
    kern: Kernel,
) -> Result<(), KdjError> {
    let (high, low, close) = match &input.data {
        KdjData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        KdjData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(KdjError::EmptyData);
    }
    let len = high.len();
    if k_out.len() != len {
        return Err(KdjError::BufferSizeMismatch {
            expected: len,
            got: k_out.len(),
        });
    }
    if d_out.len() != len {
        return Err(KdjError::BufferSizeMismatch {
            expected: len,
            got: d_out.len(),
        });
    }
    if j_out.len() != len {
        return Err(KdjError::BufferSizeMismatch {
            expected: len,
            got: j_out.len(),
        });
    }

    let fast_k = input.get_fast_k_period();
    if fast_k == 0 || fast_k > len {
        return Err(KdjError::InvalidPeriod {
            period: fast_k,
            data_len: len,
        });
    }

    let first = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(KdjError::AllValuesNaN)?;

    if len - first < fast_k {
        return Err(KdjError::NotEnoughValidData {
            needed: fast_k,
            valid: len - first,
        });
    }

    let slow_k = input.get_slow_k_period();
    let slow_d = input.get_slow_d_period();
    let slow_k_ma = input.get_slow_k_ma_type();
    let slow_d_ma = input.get_slow_d_ma_type();

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => kdj_compute_into_scalar(
            high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, k_out, d_out,
            j_out,
        ),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => {
            // TODO: Implement AVX2-specific compute_into function
            // For now, fall back to scalar
            kdj_compute_into_scalar(
                high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, k_out,
                d_out, j_out,
            )
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            // TODO: Implement AVX512-specific compute_into function
            // For now, fall back to scalar
            kdj_compute_into_scalar(
                high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, k_out,
                d_out, j_out,
            )
        }
        _ => {
            // Fallback for non-x86_64 or when nightly-avx is not enabled
            kdj_compute_into_scalar(
                high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, k_out,
                d_out, j_out,
            )
        }
    }
}

#[inline]
fn kdj_compute_into_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fast_k: usize,
    slow_k: usize,
    slow_k_ma: &str,
    slow_d: usize,
    slow_d_ma: &str,
    k_out: &mut [f64],
    d_out: &mut [f64],
    j_out: &mut [f64],
) -> Result<(), KdjError> {
    use std::collections::VecDeque;

    let len = high.len();
    if len == 0 {
        return Err(KdjError::EmptyData);
    }

    let stoch_warm = first + fast_k - 1;
    let k_warm = stoch_warm + slow_k - 1;
    let d_warm = k_warm + slow_d - 1;

    // Fast paths: fully fused single pass for SMA/SMA and EMA/EMA
    let sma_k = slow_k_ma.eq_ignore_ascii_case("sma");
    let sma_d = slow_d_ma.eq_ignore_ascii_case("sma");
    if sma_k && sma_d {
        // Init NaN prefixes
        for i in 0..k_warm.min(len) {
            k_out[i] = f64::NAN;
        }
        for i in 0..d_warm.min(len) {
            d_out[i] = f64::NAN;
            j_out[i] = f64::NAN;
        }

        let mut maxdq: VecDeque<usize> = VecDeque::with_capacity(fast_k + 1);
        let mut mindq: VecDeque<usize> = VecDeque::with_capacity(fast_k + 1);

        let mut stoch_ring = vec![f64::NAN; slow_k];
        let mut sum_k = 0.0f64;
        let mut cnt_k: usize = 0;

        let mut k_ring = vec![f64::NAN; slow_d];
        let mut sum_d = 0.0f64;
        let mut cnt_d: usize = 0;

        for i in first..len {
            // update max deque
            let hi = unsafe { *high.get_unchecked(i) };
            while let Some(&idx) = maxdq.back() {
                if unsafe { *high.get_unchecked(idx) } <= hi {
                    maxdq.pop_back();
                } else {
                    break;
                }
            }
            maxdq.push_back(i);
            while let Some(&idx) = maxdq.front() {
                if idx + fast_k <= i {
                    maxdq.pop_front();
                } else {
                    break;
                }
            }

            // update min deque
            let lo = unsafe { *low.get_unchecked(i) };
            while let Some(&idx) = mindq.back() {
                if unsafe { *low.get_unchecked(idx) } >= lo {
                    mindq.pop_back();
                } else {
                    break;
                }
            }
            mindq.push_back(i);
            while let Some(&idx) = mindq.front() {
                if idx + fast_k <= i {
                    mindq.pop_front();
                } else {
                    break;
                }
            }

            if i < stoch_warm {
                continue;
            }

            // compute stoch
            let hh = unsafe { *high.get_unchecked(*maxdq.front().unwrap()) };
            let ll = unsafe { *low.get_unchecked(*mindq.front().unwrap()) };
            let denom = hh - ll;
            let stoch_i = if denom == 0.0 || denom.is_nan() {
                f64::NAN
            } else {
                let c = unsafe { *close.get_unchecked(i) };
                100.0 * ((c - ll) / denom)
            };

            // feed K SMA
            let pos_k = i % slow_k;
            let old_st = stoch_ring[pos_k];
            if !old_st.is_nan() {
                sum_k -= old_st;
                cnt_k -= 1;
            }
            stoch_ring[pos_k] = stoch_i;
            if !stoch_i.is_nan() {
                sum_k += stoch_i;
                cnt_k += 1;
            }

            if i >= k_warm {
                let k_val = if cnt_k > 0 {
                    sum_k / (cnt_k as f64)
                } else {
                    f64::NAN
                };
                unsafe { *k_out.get_unchecked_mut(i) = k_val };

                // feed D SMA
                let pos_d = i % slow_d;
                let old_k = k_ring[pos_d];
                if !old_k.is_nan() {
                    sum_d -= old_k;
                    cnt_d -= 1;
                }
                k_ring[pos_d] = k_val;
                if !k_val.is_nan() {
                    sum_d += k_val;
                    cnt_d += 1;
                }

                if i >= d_warm {
                    let d_val = if cnt_d > 0 {
                        sum_d / (cnt_d as f64)
                    } else {
                        f64::NAN
                    };
                    unsafe {
                        *d_out.get_unchecked_mut(i) = d_val;
                        *j_out.get_unchecked_mut(i) = if k_val.is_nan() || d_val.is_nan() {
                            f64::NAN
                        } else {
                            3.0 * k_val - 2.0 * d_val
                        };
                    }
                }
            }
        }
        return Ok(());
    }

    let ema_k = slow_k_ma.eq_ignore_ascii_case("ema");
    let ema_d = slow_d_ma.eq_ignore_ascii_case("ema");
    if ema_k && ema_d {
        // Init NaN prefixes
        for i in 0..k_warm.min(len) {
            k_out[i] = f64::NAN;
        }
        for i in 0..d_warm.min(len) {
            d_out[i] = f64::NAN;
            j_out[i] = f64::NAN;
        }

        let mut maxdq: VecDeque<usize> = VecDeque::with_capacity(fast_k + 1);
        let mut mindq: VecDeque<usize> = VecDeque::with_capacity(fast_k + 1);

        let alpha_k = 2.0 / (slow_k as f64 + 1.0);
        let om_alpha_k = 1.0 - alpha_k;
        let alpha_d = 2.0 / (slow_d as f64 + 1.0);
        let om_alpha_d = 1.0 - alpha_d;

        let mut sum_init_k = 0.0f64;
        let mut cnt_init_k: usize = 0;
        let mut ema_kv = f64::NAN;

        let mut sum_init_d = 0.0f64;
        let mut cnt_init_d: usize = 0;
        let mut ema_dv = f64::NAN;

        for i in first..len {
            // update deques
            let hi = unsafe { *high.get_unchecked(i) };
            while let Some(&idx) = maxdq.back() {
                if unsafe { *high.get_unchecked(idx) } <= hi {
                    maxdq.pop_back();
                } else {
                    break;
                }
            }
            maxdq.push_back(i);
            while let Some(&idx) = maxdq.front() {
                if idx + fast_k <= i {
                    maxdq.pop_front();
                } else {
                    break;
                }
            }

            let lo = unsafe { *low.get_unchecked(i) };
            while let Some(&idx) = mindq.back() {
                if unsafe { *low.get_unchecked(idx) } >= lo {
                    mindq.pop_back();
                } else {
                    break;
                }
            }
            mindq.push_back(i);
            while let Some(&idx) = mindq.front() {
                if idx + fast_k <= i {
                    mindq.pop_front();
                } else {
                    break;
                }
            }

            if i < stoch_warm {
                continue;
            }

            let hh = unsafe { *high.get_unchecked(*maxdq.front().unwrap()) };
            let ll = unsafe { *low.get_unchecked(*mindq.front().unwrap()) };
            let denom = hh - ll;
            let stoch_i = if denom == 0.0 || denom.is_nan() {
                f64::NAN
            } else {
                let c = unsafe { *close.get_unchecked(i) };
                100.0 * ((c - ll) / denom)
            };

            if i <= k_warm {
                if !stoch_i.is_nan() {
                    sum_init_k += stoch_i;
                    cnt_init_k += 1;
                }
                if i == k_warm {
                    ema_kv = if cnt_init_k > 0 {
                        sum_init_k / (cnt_init_k as f64)
                    } else {
                        f64::NAN
                    };
                    unsafe { *k_out.get_unchecked_mut(i) = ema_kv };
                    if !ema_kv.is_nan() {
                        sum_init_d += ema_kv;
                        cnt_init_d += 1;
                    }
                }
                continue;
            }

            // i > k_warm
            if !stoch_i.is_nan() && !ema_kv.is_nan() {
                ema_kv = stoch_i.mul_add(alpha_k, om_alpha_k * ema_kv);
            } else if !stoch_i.is_nan() && ema_kv.is_nan() {
                ema_kv = stoch_i;
            }
            unsafe { *k_out.get_unchecked_mut(i) = ema_kv };

            if i <= d_warm {
                if !ema_kv.is_nan() {
                    sum_init_d += ema_kv;
                    cnt_init_d += 1;
                }
                if i == d_warm {
                    ema_dv = if cnt_init_d > 0 {
                        sum_init_d / (cnt_init_d as f64)
                    } else {
                        f64::NAN
                    };
                    unsafe {
                        *d_out.get_unchecked_mut(i) = ema_dv;
                        *j_out.get_unchecked_mut(i) = if ema_kv.is_nan() || ema_dv.is_nan() {
                            f64::NAN
                        } else {
                            3.0 * ema_kv - 2.0 * ema_dv
                        };
                    }
                }
                continue;
            }

            // i > d_warm
            if !ema_kv.is_nan() && !ema_dv.is_nan() {
                ema_dv = ema_kv.mul_add(alpha_d, om_alpha_d * ema_dv);
            } else if !ema_kv.is_nan() && ema_dv.is_nan() {
                ema_dv = ema_kv;
            }
            unsafe {
                *d_out.get_unchecked_mut(i) = ema_dv;
                *j_out.get_unchecked_mut(i) = if ema_kv.is_nan() || ema_dv.is_nan() {
                    f64::NAN
                } else {
                    3.0 * ema_kv - 2.0 * ema_dv
                };
            }
        }
        return Ok(());
    }

    // Generic MA path: build stochastic once using fused deques, then apply dynamic MA
    let mut stoch = alloc_with_nan_prefix(len, stoch_warm);

    let mut maxdq: VecDeque<usize> = VecDeque::with_capacity(fast_k + 1);
    let mut mindq: VecDeque<usize> = VecDeque::with_capacity(fast_k + 1);

    for i in first..len {
        let hi = unsafe { *high.get_unchecked(i) };
        while let Some(&idx) = maxdq.back() {
            if unsafe { *high.get_unchecked(idx) } <= hi {
                maxdq.pop_back();
            } else {
                break;
            }
        }
        maxdq.push_back(i);
        while let Some(&idx) = maxdq.front() {
            if idx + fast_k <= i {
                maxdq.pop_front();
            } else {
                break;
            }
        }

        let lo = unsafe { *low.get_unchecked(i) };
        while let Some(&idx) = mindq.back() {
            if unsafe { *low.get_unchecked(idx) } >= lo {
                mindq.pop_back();
            } else {
                break;
            }
        }
        mindq.push_back(i);
        while let Some(&idx) = mindq.front() {
            if idx + fast_k <= i {
                mindq.pop_front();
            } else {
                break;
            }
        }

        if i < stoch_warm {
            continue;
        }

        let hh = unsafe { *high.get_unchecked(*maxdq.front().unwrap()) };
        let ll = unsafe { *low.get_unchecked(*mindq.front().unwrap()) };
        let denom = hh - ll;
        let val = if denom == 0.0 || denom.is_nan() {
            f64::NAN
        } else {
            let c = unsafe { *close.get_unchecked(i) };
            100.0 * ((c - ll) / denom)
        };
        unsafe { *stoch.get_unchecked_mut(i) = val };
    }

    let k_vec = ma(slow_k_ma, MaData::Slice(&stoch), slow_k)
        .map_err(|e| KdjError::MaError(e.to_string().into()))?;
    let d_vec = ma(slow_d_ma, MaData::Slice(&k_vec), slow_d)
        .map_err(|e| KdjError::MaError(e.to_string().into()))?;

    k_out.copy_from_slice(&k_vec);
    d_out.copy_from_slice(&d_vec);

    let j_warm = stoch_warm + slow_k - 1 + slow_d - 1;
    for i in 0..j_warm.min(j_out.len()) {
        j_out[i] = f64::NAN;
    }
    for i in j_warm..j_out.len() {
        j_out[i] = if k_out[i].is_nan() || d_out[i].is_nan() {
            f64::NAN
        } else {
            3.0 * k_out[i] - 2.0 * d_out[i]
        };
    }
    Ok(())
}

// =========== SIMD Stubs ===========

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kdj_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    if fast_k_period <= 32 {
        unsafe {
            kdj_avx512_short(
                high,
                low,
                close,
                fast_k_period,
                slow_k_period,
                slow_k_ma_type,
                slow_d_period,
                slow_d_ma_type,
                first_valid_idx,
            )
        }
    } else {
        unsafe {
            kdj_avx512_long(
                high,
                low,
                close,
                fast_k_period,
                slow_k_period,
                slow_k_ma_type,
                slow_d_period,
                slow_d_ma_type,
                first_valid_idx,
            )
        }
    }
}

#[inline]
pub fn kdj_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // TODO: Implement actual AVX2 SIMD optimizations for KDJ calculation
    // AVX2 stub, points to scalar
    kdj_scalar(
        high,
        low,
        close,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        first_valid_idx,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kdj_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // TODO: Implement actual AVX512 SIMD optimizations for KDJ calculation (short period)
    // AVX512 short stub, points to scalar
    kdj_scalar(
        high,
        low,
        close,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        first_valid_idx,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kdj_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // TODO: Implement actual AVX512 SIMD optimizations for KDJ calculation (long period)
    // AVX512 long stub, points to scalar
    kdj_scalar(
        high,
        low,
        close,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        first_valid_idx,
    )
}

// ========== SIMD128 (WASM) ==========

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn kdj_simd128(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // SIMD128 doesn't provide significant benefits for KDJ due to the MA calls
    // and the rolling min/max operations. Delegate to scalar implementation.
    kdj_scalar(
        high,
        low,
        close,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        first_valid_idx,
    )
}

// ========== Builder/Stream ==========

#[derive(Clone, Debug)]
pub struct KdjBuilder {
    fast_k_period: Option<usize>,
    slow_k_period: Option<usize>,
    slow_k_ma_type: Option<String>,
    slow_d_period: Option<usize>,
    slow_d_ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for KdjBuilder {
    fn default() -> Self {
        Self {
            fast_k_period: None,
            slow_k_period: None,
            slow_k_ma_type: None,
            slow_d_period: None,
            slow_d_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KdjBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fast_k_period(mut self, n: usize) -> Self {
        self.fast_k_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_k_period(mut self, n: usize) -> Self {
        self.slow_k_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_k_ma_type<S: Into<String>>(mut self, t: S) -> Self {
        self.slow_k_ma_type = Some(t.into());
        self
    }
    #[inline(always)]
    pub fn slow_d_period(mut self, n: usize) -> Self {
        self.slow_d_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_d_ma_type<S: Into<String>>(mut self, t: S) -> Self {
        self.slow_d_ma_type = Some(t.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KdjOutput, KdjError> {
        let p = KdjParams {
            fast_k_period: self.fast_k_period,
            slow_k_period: self.slow_k_period,
            slow_k_ma_type: self.slow_k_ma_type,
            slow_d_period: self.slow_d_period,
            slow_d_ma_type: self.slow_d_ma_type,
        };
        let i = KdjInput::from_candles(c, p);
        kdj_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<KdjOutput, KdjError> {
        let p = KdjParams {
            fast_k_period: self.fast_k_period,
            slow_k_period: self.slow_k_period,
            slow_k_ma_type: self.slow_k_ma_type,
            slow_d_period: self.slow_d_period,
            slow_d_ma_type: self.slow_d_ma_type,
        };
        let i = KdjInput::from_slices(high, low, close, p);
        kdj_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<KdjStream, KdjError> {
        let p = KdjParams {
            fast_k_period: self.fast_k_period,
            slow_k_period: self.slow_k_period,
            slow_k_ma_type: self.slow_k_ma_type,
            slow_d_period: self.slow_d_period,
            slow_d_ma_type: self.slow_d_ma_type,
        };
        KdjStream::try_new(p)
    }
}

// ========= Stream Processing (drop-in replacement) ==========
// Decision: Streaming uses monotonic deques + O(1) SMA/EMA smoothing; matches scalar warmups.
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct KdjStream {
    // --- config ---
    fast_k_period: usize,
    slow_k_period: usize,
    slow_d_period: usize,
    // lowercased selectors cached as booleans
    k_is_sma: bool,
    k_is_ema: bool,
    d_is_sma: bool,
    d_is_ema: bool,

    // --- sliding window for HH/LL via monotonic deques (amortized O(1)) ---
    i: usize,                      // monotonically increasing update index
    maxdq: VecDeque<(usize, f64)>, // (idx, high) non-increasing
    mindq: VecDeque<(usize, f64)>, // (idx, low) non-decreasing

    // --- stage counters (to reproduce batch warm-ups) ---
    have_fast: bool,      // true once i+1 >= fast_k_period
    stoch_samples: usize, // # of stoch values since have_fast
    k_samples: usize,     // # of K values since K warm-up completed

    // --- smoothing state: SMA for K ---
    stoch_ring: Vec<f64>,
    stoch_pos: usize,
    sum_k: f64,
    cnt_k: usize,
    stoch_filled: bool,

    // --- smoothing state: SMA for D ---
    k_ring: Vec<f64>,
    k_pos: usize,
    sum_d: f64,
    cnt_d: usize,
    k_filled: bool,

    // --- smoothing state: EMA for K ---
    alpha_k: f64,
    om_alpha_k: f64,
    ema_k: f64,
    k_ema_inited: bool,
    init_sum_k: f64,
    init_cnt_k: usize,

    // --- smoothing state: EMA for D ---
    alpha_d: f64,
    om_alpha_d: f64,
    ema_d: f64,
    d_ema_inited: bool,
    init_sum_d: f64,
    init_cnt_d: usize,

    // --- small LUTs for SMA reciprocals (avoid divides) ---
    inv_cnt_k: Vec<f64>,
    inv_cnt_d: Vec<f64>,
}

impl KdjStream {
    pub fn try_new(params: KdjParams) -> Result<Self, KdjError> {
        let fast_k_period = params.fast_k_period.unwrap_or(9);
        let slow_k_period = params.slow_k_period.unwrap_or(3);
        let slow_k_ma_type = params.slow_k_ma_type.unwrap_or_else(|| "sma".to_string());
        let slow_d_period = params.slow_d_period.unwrap_or(3);
        let slow_d_ma_type = params.slow_d_ma_type.unwrap_or_else(|| "sma".to_string());

        if fast_k_period == 0 {
            return Err(KdjError::InvalidPeriod {
                period: fast_k_period,
                data_len: 0,
            });
        }
        if slow_k_period == 0 {
            return Err(KdjError::InvalidPeriod {
                period: slow_k_period,
                data_len: 0,
            });
        }
        if slow_d_period == 0 {
            return Err(KdjError::InvalidPeriod {
                period: slow_d_period,
                data_len: 0,
            });
        }

        let k_is_sma = slow_k_ma_type.eq_ignore_ascii_case("sma");
        let k_is_ema = slow_k_ma_type.eq_ignore_ascii_case("ema");
        let d_is_sma = slow_d_ma_type.eq_ignore_ascii_case("sma");
        let d_is_ema = slow_d_ma_type.eq_ignore_ascii_case("ema");

        // EMA coefficients
        let alpha_k = 2.0 / (slow_k_period as f64 + 1.0);
        let om_alpha_k = 1.0 - alpha_k;
        let alpha_d = 2.0 / (slow_d_period as f64 + 1.0);
        let om_alpha_d = 1.0 - alpha_d;

        // Precompute 1/c for c in 1..=period
        fn build_inv(n: usize) -> Vec<f64> {
            let mut v = vec![f64::NAN; n + 1];
            for c in 1..=n {
                v[c] = 1.0 / (c as f64);
            }
            v
        }

        Ok(Self {
            fast_k_period,
            slow_k_period,
            slow_d_period,
            k_is_sma,
            k_is_ema,
            d_is_sma,
            d_is_ema,

            i: 0,
            maxdq: VecDeque::with_capacity(fast_k_period + 1),
            mindq: VecDeque::with_capacity(fast_k_period + 1),

            have_fast: false,
            stoch_samples: 0,
            k_samples: 0,

            stoch_ring: vec![f64::NAN; slow_k_period],
            stoch_pos: 0,
            sum_k: 0.0,
            cnt_k: 0,
            stoch_filled: false,

            k_ring: vec![f64::NAN; slow_d_period],
            k_pos: 0,
            sum_d: 0.0,
            cnt_d: 0,
            k_filled: false,

            alpha_k,
            om_alpha_k,
            ema_k: f64::NAN,
            k_ema_inited: false,
            init_sum_k: 0.0,
            init_cnt_k: 0,

            alpha_d,
            om_alpha_d,
            ema_d: f64::NAN,
            d_ema_inited: false,
            init_sum_d: 0.0,
            init_cnt_d: 0,

            inv_cnt_k: build_inv(slow_k_period),
            inv_cnt_d: build_inv(slow_d_period),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        let idx = self.i;
        self.i = idx + 1;

        // ---- Update monotonic deques for HH/LL (ignore NaNs) ----
        if !high.is_nan() {
            while let Some(&(_, v)) = self.maxdq.back() {
                if v <= high {
                    self.maxdq.pop_back();
                } else {
                    break;
                }
            }
            self.maxdq.push_back((idx, high));
        }
        if !low.is_nan() {
            while let Some(&(_, v)) = self.mindq.back() {
                if v >= low {
                    self.mindq.pop_back();
                } else {
                    break;
                }
            }
            self.mindq.push_back((idx, low));
        }

        // Expire old entries: keep [idx - fast_k + 1, idx]
        let expire_before = idx + 1 - self.fast_k_period;
        while let Some(&(j, _)) = self.maxdq.front() {
            if j < expire_before {
                self.maxdq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(j, _)) = self.mindq.front() {
            if j < expire_before {
                self.mindq.pop_front();
            } else {
                break;
            }
        }

        // ---- Enough samples to emit a stochastic reading? ----
        if !self.have_fast && (idx + 1) >= self.fast_k_period {
            self.have_fast = true;
        }
        if !self.have_fast {
            return None;
        }

        // Stochastic (%K_fast) for this tick
        let stoch = if close.is_nan() || self.maxdq.is_empty() || self.mindq.is_empty() {
            f64::NAN
        } else {
            let hh = self.maxdq.front().unwrap().1;
            let ll = self.mindq.front().unwrap().1;
            let denom = hh - ll;
            if denom == 0.0 || denom.is_nan() {
                f64::NAN
            } else {
                let inv = 1.0 / denom;
                (close - ll) * (100.0 * inv)
            }
        };
        self.stoch_samples += 1;

        // ---- Smooth into %K_slow ----
        let mut k_val = f64::NAN;
        let k_now_available: bool;

        if self.k_is_sma || (!self.k_is_ema && !self.k_is_sma) {
            // SMA K
            let pos = self.stoch_pos;
            let old = self.stoch_ring[pos];
            if !old.is_nan() {
                self.sum_k -= old;
                self.cnt_k -= 1;
            }
            self.stoch_ring[pos] = stoch;
            self.stoch_pos = (pos + 1) % self.slow_k_period;
            if !stoch.is_nan() {
                self.sum_k += stoch;
                self.cnt_k += 1;
            }
            if !self.stoch_filled && self.stoch_pos == 0 {
                self.stoch_filled = true;
            }

            if self.stoch_filled {
                k_val = if self.cnt_k > 0 {
                    self.sum_k * self.inv_cnt_k[self.cnt_k]
                } else {
                    f64::NAN
                };
                k_now_available = true;
            } else {
                k_now_available = false;
            }
        } else {
            // EMA K
            if !self.k_ema_inited {
                if !stoch.is_nan() {
                    self.init_sum_k += stoch;
                    self.init_cnt_k += 1;
                }
                if self.stoch_samples == self.slow_k_period {
                    self.ema_k = if self.init_cnt_k > 0 {
                        self.init_sum_k * self.inv_cnt_k[self.init_cnt_k]
                    } else {
                        f64::NAN
                    };
                    self.k_ema_inited = true;
                    k_val = self.ema_k;
                    k_now_available = true;
                } else {
                    k_now_available = false;
                }
            } else {
                if !stoch.is_nan() && !self.ema_k.is_nan() {
                    self.ema_k = stoch.mul_add(self.alpha_k, self.om_alpha_k * self.ema_k);
                } else if !stoch.is_nan() && self.ema_k.is_nan() {
                    self.ema_k = stoch;
                }
                k_val = self.ema_k;
                k_now_available = true;
            }
        }

        // ---- Smooth K into %D_slow ----
        if !k_now_available {
            return None;
        }

        let mut d_val = f64::NAN;
        let d_now_available: bool;

        if self.d_is_sma || (!self.d_is_ema && !self.d_is_sma) {
            // SMA D
            let pos = self.k_pos;
            let old_k = self.k_ring[pos];
            if !old_k.is_nan() {
                self.sum_d -= old_k;
                self.cnt_d -= 1;
            }
            self.k_ring[pos] = k_val;
            self.k_pos = (pos + 1) % self.slow_d_period;
            if !k_val.is_nan() {
                self.sum_d += k_val;
                self.cnt_d += 1;
            }
            if !self.k_filled && self.k_pos == 0 {
                self.k_filled = true;
            }

            if self.k_filled {
                d_val = if self.cnt_d > 0 {
                    self.sum_d * self.inv_cnt_d[self.cnt_d]
                } else {
                    f64::NAN
                };
                d_now_available = true;
            } else {
                d_now_available = false;
            }
        } else {
            // EMA D
            if !self.d_ema_inited {
                self.k_samples += 1;
                if !k_val.is_nan() {
                    self.init_sum_d += k_val;
                    self.init_cnt_d += 1;
                }
                if self.k_samples == self.slow_d_period {
                    self.ema_d = if self.init_cnt_d > 0 {
                        self.init_sum_d * self.inv_cnt_d[self.init_cnt_d]
                    } else {
                        f64::NAN
                    };
                    self.d_ema_inited = true;
                    d_val = self.ema_d;
                    d_now_available = true;
                } else {
                    d_now_available = false;
                }
            } else {
                if !k_val.is_nan() && !self.ema_d.is_nan() {
                    self.ema_d = k_val.mul_add(self.alpha_d, self.om_alpha_d * self.ema_d);
                } else if !k_val.is_nan() && self.ema_d.is_nan() {
                    self.ema_d = k_val;
                }
                d_val = self.ema_d;
                d_now_available = true;
            }
        }

        if !self.d_is_ema {
            // If D is SMA, count K samples after consuming one K
            self.k_samples = self.k_samples.saturating_add(1);
        }

        if !d_now_available {
            return None;
        }

        // J = 3*K - 2*D
        let j_val = if k_val.is_nan() || d_val.is_nan() {
            f64::NAN
        } else {
            k_val.mul_add(3.0, -2.0 * d_val)
        };

        Some((k_val, d_val, j_val))
    }
}

// ========== Batch API ==========

#[derive(Clone, Debug)]
pub struct KdjBatchRange {
    pub fast_k_period: (usize, usize, usize),
    pub slow_k_period: (usize, usize, usize),
    pub slow_k_ma_type: (String, String, String),
    pub slow_d_period: (usize, usize, usize),
    pub slow_d_ma_type: (String, String, String),
}

impl Default for KdjBatchRange {
    fn default() -> Self {
        Self {
            fast_k_period: (9, 30, 1),
            slow_k_period: (3, 6, 1),
            slow_k_ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
            slow_d_period: (3, 6, 1),
            slow_d_ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KdjBatchBuilder {
    range: KdjBatchRange,
    kernel: Kernel,
}

impl KdjBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn fast_k_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_k_period = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_k_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_k_period = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_k_ma_type_static<S: Into<String>>(mut self, s: S) -> Self {
        let v = s.into();
        self.range.slow_k_ma_type = (v.clone(), v, "".to_string());
        self
    }
    #[inline]
    pub fn slow_d_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_d_period = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_d_ma_type_static<S: Into<String>>(mut self, s: S) -> Self {
        let v = s.into();
        self.range.slow_d_ma_type = (v.clone(), v, "".to_string());
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<KdjBatchOutput, KdjError> {
        kdj_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<KdjBatchOutput, KdjError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

pub fn kdj_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    k: Kernel,
) -> Result<KdjBatchOutput, KdjError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(KdjError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    kdj_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KdjBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub j: Vec<f64>,
    pub combos: Vec<KdjParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KdjBatchOutput {
    pub fn row_for_params(&self, p: &KdjParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_k_period.unwrap_or(9) == p.fast_k_period.unwrap_or(9)
                && c.slow_k_period.unwrap_or(3) == p.slow_k_period.unwrap_or(3)
                && c.slow_k_ma_type.as_deref().unwrap_or("sma")
                    == p.slow_k_ma_type.as_deref().unwrap_or("sma")
                && c.slow_d_period.unwrap_or(3) == p.slow_d_period.unwrap_or(3)
                && c.slow_d_ma_type.as_deref().unwrap_or("sma")
                    == p.slow_d_ma_type.as_deref().unwrap_or("sma")
        })
    }
    pub fn k_for(&self, p: &KdjParams) -> Option<&[f64]> {
        self.row_for_params(p)
            .map(|row| &self.k[row * self.cols..][..self.cols])
    }
    pub fn d_for(&self, p: &KdjParams) -> Option<&[f64]> {
        self.row_for_params(p)
            .map(|row| &self.d[row * self.cols..][..self.cols])
    }
    pub fn j_for(&self, p: &KdjParams) -> Option<&[f64]> {
        self.row_for_params(p)
            .map(|row| &self.j[row * self.cols..][..self.cols])
    }
}

#[inline(always)]
fn expand_grid(r: &KdjBatchRange) -> Vec<KdjParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_str((start, end, _): (String, String, String)) -> Vec<String> {
        if start == end {
            vec![start]
        } else {
            vec![start, end]
        }
    }
    let fast_k_periods = axis_usize(r.fast_k_period);
    let slow_k_periods = axis_usize(r.slow_k_period);
    let slow_k_ma_types = axis_str(r.slow_k_ma_type.clone());
    let slow_d_periods = axis_usize(r.slow_d_period);
    let slow_d_ma_types = axis_str(r.slow_d_ma_type.clone());
    let mut out = Vec::new();
    for &fkp in &fast_k_periods {
        for &skp in &slow_k_periods {
            for skmt in &slow_k_ma_types {
                for &sdp in &slow_d_periods {
                    for sdmt in &slow_d_ma_types {
                        out.push(KdjParams {
                            fast_k_period: Some(fkp),
                            slow_k_period: Some(skp),
                            slow_k_ma_type: Some(skmt.clone()),
                            slow_d_period: Some(sdp),
                            slow_d_ma_type: Some(sdmt.clone()),
                        });
                    }
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn kdj_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    kern: Kernel,
) -> Result<KdjBatchOutput, KdjError> {
    kdj_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn kdj_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    kern: Kernel,
) -> Result<KdjBatchOutput, KdjError> {
    kdj_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn kdj_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<KdjBatchOutput, KdjError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KdjError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(KdjError::AllValuesNaN)?;

    let max_p = combos
        .iter()
        .map(|c| c.fast_k_period.unwrap())
        .max()
        .unwrap();
    if high.len() - first < max_p {
        return Err(KdjError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();

    // Use make_uninit_matrix instead of vec! for zero-copy
    let mut k_mu = make_uninit_matrix(rows, cols);
    let mut d_mu = make_uninit_matrix(rows, cols);
    let mut j_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each parameter combination
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            let fast_k = c.fast_k_period.unwrap();
            let slow_k = c.slow_k_period.unwrap();
            let slow_d = c.slow_d_period.unwrap();
            // Warmup = first_valid + fast_k - 1 + slow_k - 1 + slow_d - 1
            first + fast_k + slow_k + slow_d - 3
        })
        .collect();

    // Initialize prefixes with NaN
    init_matrix_prefixes(&mut k_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut d_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut j_mu, cols, &warmup_periods);

    // Safe &mut [f64] views over MaybeUninit<f64>
    let mut k_guard = core::mem::ManuallyDrop::new(k_mu);
    let mut d_guard = core::mem::ManuallyDrop::new(d_mu);
    let mut j_guard = core::mem::ManuallyDrop::new(j_mu);

    let k_vals: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(k_guard.as_mut_ptr() as *mut f64, k_guard.len()) };
    let d_vals: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(d_guard.as_mut_ptr() as *mut f64, d_guard.len()) };
    let j_vals: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(j_guard.as_mut_ptr() as *mut f64, j_guard.len()) };

    let chosen = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    // If multiple rows share the same fast_k, precompute stochastic once per unique fast_k and reuse
    let unique_fast: std::collections::BTreeSet<usize> =
        combos.iter().map(|c| c.fast_k_period.unwrap()).collect();

    // Build a cache when reuse exists; otherwise fall back to per-row compute
    let use_stoch_cache = unique_fast.len() < combos.len();
    let mut stoch_cache: std::collections::HashMap<usize, Vec<f64>> =
        std::collections::HashMap::new();
    if use_stoch_cache {
        use std::collections::VecDeque;
        for &fk in &unique_fast {
            let stoch_warm = first + fk - 1;
            let mut stoch = alloc_with_nan_prefix(cols, stoch_warm);
            let mut maxdq: VecDeque<usize> = VecDeque::with_capacity(fk + 1);
            let mut mindq: VecDeque<usize> = VecDeque::with_capacity(fk + 1);
            for i in first..cols {
                let hi = unsafe { *high.get_unchecked(i) };
                while let Some(&idx) = maxdq.back() {
                    if unsafe { *high.get_unchecked(idx) } <= hi {
                        maxdq.pop_back();
                    } else {
                        break;
                    }
                }
                maxdq.push_back(i);
                while let Some(&idx) = maxdq.front() {
                    if idx + fk <= i {
                        maxdq.pop_front();
                    } else {
                        break;
                    }
                }

                let lo = unsafe { *low.get_unchecked(i) };
                while let Some(&idx) = mindq.back() {
                    if unsafe { *low.get_unchecked(idx) } >= lo {
                        mindq.pop_back();
                    } else {
                        break;
                    }
                }
                mindq.push_back(i);
                while let Some(&idx) = mindq.front() {
                    if idx + fk <= i {
                        mindq.pop_front();
                    } else {
                        break;
                    }
                }

                if i < stoch_warm {
                    continue;
                }

                let hh = unsafe { *high.get_unchecked(*maxdq.front().unwrap()) };
                let ll = unsafe { *low.get_unchecked(*mindq.front().unwrap()) };
                let denom = hh - ll;
                let val = if denom == 0.0 || denom.is_nan() {
                    f64::NAN
                } else {
                    let c = unsafe { *close.get_unchecked(i) };
                    100.0 * ((c - ll) / denom)
                };
                unsafe { *stoch.get_unchecked_mut(i) = val };
            }
            stoch_cache.insert(fk, stoch);
        }
    }

    let do_row = |row: usize,
                  out_k: &mut [f64],
                  out_d: &mut [f64],
                  out_j: &mut [f64]|
     -> Result<(), KdjError> {
        let prm = &combos[row];
        let fast_k = prm.fast_k_period.unwrap();
        let slow_k = prm.slow_k_period.unwrap();
        let slow_k_ma = prm.slow_k_ma_type.as_deref().unwrap_or("sma");
        let slow_d = prm.slow_d_period.unwrap();
        let slow_d_ma = prm.slow_d_ma_type.as_deref().unwrap_or("sma");

        if use_stoch_cache {
            let stoch = stoch_cache
                .get(&fast_k)
                .expect("stoch cache missing fast_k");
            let stoch_warm = first + fast_k - 1;
            // Prefer classic kernels for common types
            if slow_k_ma.eq_ignore_ascii_case("sma") && slow_d_ma.eq_ignore_ascii_case("sma") {
                return kdj_classic_sma(stoch, slow_k, slow_d, stoch_warm, out_k, out_d, out_j);
            }
            if slow_k_ma.eq_ignore_ascii_case("ema") && slow_d_ma.eq_ignore_ascii_case("ema") {
                return kdj_classic_ema(stoch, slow_k, slow_d, stoch_warm, out_k, out_d, out_j);
            }
            // Generic MA path
            let k_vec = ma(slow_k_ma, MaData::Slice(stoch), slow_k)
                .map_err(|e| KdjError::MaError(e.to_string().into()))?;
            let d_vec = ma(slow_d_ma, MaData::Slice(&k_vec), slow_d)
                .map_err(|e| KdjError::MaError(e.to_string().into()))?;
            out_k.copy_from_slice(&k_vec);
            out_d.copy_from_slice(&d_vec);
            let j_warm = stoch_warm + slow_k - 1 + slow_d - 1;
            for i in 0..j_warm.min(cols) {
                out_j[i] = f64::NAN;
            }
            for i in j_warm..cols {
                out_j[i] = if out_k[i].is_nan() || out_d[i].is_nan() {
                    f64::NAN
                } else {
                    3.0 * out_k[i] - 2.0 * out_d[i]
                };
            }
            return Ok(());
        }

        // Fall back to original per-row kernel selection
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kdj_row_scalar(
                high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, out_k,
                out_d, out_j,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
                kdj_row_avx2(
                    high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, out_k,
                    out_d, out_j,
                )
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
                kdj_row_avx512(
                    high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, out_k,
                    out_d, out_j,
                )
            },
            _ => kdj_row_scalar(
                high, low, close, first, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, out_k,
                out_d, out_j,
            ),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            k_vals
                .par_chunks_mut(cols)
                .zip(d_vals.par_chunks_mut(cols))
                .zip(j_vals.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, ((ok, od), oj))| do_row(row, ok, od, oj))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, ((ok, od), oj)) in k_vals
                .chunks_mut(cols)
                .zip(d_vals.chunks_mut(cols))
                .zip(j_vals.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, ok, od, oj)?;
            }
        }
    } else {
        for (row, ((ok, od), oj)) in k_vals
            .chunks_mut(cols)
            .zip(d_vals.chunks_mut(cols))
            .zip(j_vals.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, ok, od, oj)?;
        }
    }

    // Move out Vec<f64> without copy
    let k_vec = unsafe {
        Vec::from_raw_parts(
            k_guard.as_mut_ptr() as *mut f64,
            k_guard.len(),
            k_guard.capacity(),
        )
    };
    let d_vec = unsafe {
        Vec::from_raw_parts(
            d_guard.as_mut_ptr() as *mut f64,
            d_guard.len(),
            d_guard.capacity(),
        )
    };
    let j_vec = unsafe {
        Vec::from_raw_parts(
            j_guard.as_mut_ptr() as *mut f64,
            j_guard.len(),
            j_guard.capacity(),
        )
    };

    Ok(KdjBatchOutput {
        k: k_vec,
        d: d_vec,
        j: j_vec,
        combos,
        rows,
        cols,
    })
}

// ========== Row Scalar/AVX ==========

#[inline(always)]
fn kdj_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64],
    out_d: &mut [f64],
    out_j: &mut [f64],
) -> Result<(), KdjError> {
    // Delegate to the new kdj_compute_into_scalar function
    kdj_compute_into_scalar(
        high,
        low,
        close,
        first,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        out_k,
        out_d,
        out_j,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64],
    out_d: &mut [f64],
    out_j: &mut [f64],
) -> Result<(), KdjError> {
    // TODO: Implement actual AVX2 SIMD optimizations for batch row processing
    kdj_row_scalar(
        high,
        low,
        close,
        first,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        out_k,
        out_d,
        out_j,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64],
    out_d: &mut [f64],
    out_j: &mut [f64],
) -> Result<(), KdjError> {
    if fast_k_period <= 32 {
        kdj_row_avx512_short(
            high,
            low,
            close,
            first,
            fast_k_period,
            slow_k_period,
            slow_k_ma_type,
            slow_d_period,
            slow_d_ma_type,
            out_k,
            out_d,
            out_j,
        )
    } else {
        kdj_row_avx512_long(
            high,
            low,
            close,
            first,
            fast_k_period,
            slow_k_period,
            slow_k_ma_type,
            slow_d_period,
            slow_d_ma_type,
            out_k,
            out_d,
            out_j,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64],
    out_d: &mut [f64],
    out_j: &mut [f64],
) -> Result<(), KdjError> {
    // TODO: Implement actual AVX512 SIMD optimizations for batch row processing (short period)
    kdj_row_scalar(
        high,
        low,
        close,
        first,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        out_k,
        out_d,
        out_j,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64],
    out_d: &mut [f64],
    out_j: &mut [f64],
) -> Result<(), KdjError> {
    // TODO: Implement actual AVX512 SIMD optimizations for batch row processing (long period)
    kdj_row_scalar(
        high,
        low,
        close,
        first,
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
        out_k,
        out_d,
        out_j,
    )
}

// ========== Unit Tests ==========
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_kdj_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = KdjParams {
            fast_k_period: None,
            slow_k_period: Some(4),
            slow_k_ma_type: None,
            slow_d_period: None,
            slow_d_ma_type: None,
        };
        let input = KdjInput::from_candles(&candles, partial_params);
        let output = kdj_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        assert_eq!(output.j.len(), candles.close.len());
        Ok(())
    }

    fn check_kdj_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KdjParams::default();
        let input = KdjInput::from_candles(&candles, params);
        let result = kdj_with_kernel(&input, kernel)?;
        let expected_k = [
            58.04341315415984,
            61.56034740940419,
            58.056304282719545,
            56.10961365678364,
            51.43992326447119,
        ];
        let expected_d = [
            49.57659409278555,
            56.81719223571944,
            59.22002161542779,
            58.57542178296905,
            55.20194706799139,
        ];
        let expected_j = [
            74.97705127690843,
            71.04665775677368,
            55.72886961730306,
            51.17799740441281,
            43.91587565743079,
        ];
        let len = result.k.len();
        let start_idx = len - 5;
        for i in 0..5 {
            let k_val = result.k[start_idx + i];
            let d_val = result.d[start_idx + i];
            let j_val = result.j[start_idx + i];
            assert!(
                (k_val - expected_k[i]).abs() < 1e-4,
                "Mismatch in K at index {}: expected {}, got {}",
                i,
                expected_k[i],
                k_val
            );
            assert!(
                (d_val - expected_d[i]).abs() < 1e-4,
                "Mismatch in D at index {}: expected {}, got {}",
                i,
                expected_d[i],
                d_val
            );
            assert!(
                (j_val - expected_j[i]).abs() < 1e-4,
                "Mismatch in J at index {}: expected {}, got {}",
                i,
                expected_j[i],
                j_val
            );
        }
        Ok(())
    }

    fn check_kdj_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KdjInput::with_default_candles(&candles);
        match input.data {
            KdjData::Candles { .. } => {}
            _ => panic!("Expected KdjData::Candles variant"),
        }
        let output = kdj_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        Ok(())
    }

    fn check_kdj_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = KdjParams {
            fast_k_period: Some(0),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] KDJ should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_kdj_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = KdjParams {
            fast_k_period: Some(10),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] KDJ should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_kdj_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = KdjParams {
            fast_k_period: Some(9),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&single_point, &single_point, &single_point, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] KDJ should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_kdj_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = KdjParams::default();
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] KDJ should fail with all-NaN data",
            test_name
        );
        Ok(())
    }

    fn check_kdj_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = KdjParams {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        };
        let first_input = KdjInput::from_candles(&candles, first_params);
        let first_result = kdj_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.k.len(), candles.close.len());

        let second_params = KdjParams {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        };
        let second_input = KdjInput::from_slices(
            &first_result.k,
            &first_result.k,
            &first_result.k,
            second_params,
        );
        let second_result = kdj_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.k.len(), first_result.k.len());
        for i in 50..second_result.k.len() {
            assert!(
                !second_result.k[i].is_nan(),
                "[{}] Expected no NaN in second KDJ at {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_kdj_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KdjParams::default();
        let input = KdjInput::from_candles(&candles, params);
        let result = kdj_with_kernel(&input, kernel)?;
        if result.k.len() > 50 {
            for i in 50..result.k.len() {
                assert!(
                    !result.k[i].is_nan(),
                    "[{}] Expected no NaN in K after index 50 at {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_kdj_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            KdjParams::default(),
            KdjParams {
                fast_k_period: Some(2),
                slow_k_period: Some(2),
                slow_k_ma_type: Some("sma".to_string()),
                slow_d_period: Some(2),
                slow_d_ma_type: Some("sma".to_string()),
            },
            KdjParams {
                fast_k_period: Some(3),
                slow_k_period: Some(3),
                slow_k_ma_type: Some("ema".to_string()),
                slow_d_period: Some(3),
                slow_d_ma_type: Some("ema".to_string()),
            },
            KdjParams {
                fast_k_period: Some(5),
                slow_k_period: Some(2),
                slow_k_ma_type: Some("sma".to_string()),
                slow_d_period: Some(2),
                slow_d_ma_type: Some("sma".to_string()),
            },
            KdjParams {
                fast_k_period: Some(10),
                slow_k_period: Some(5),
                slow_k_ma_type: Some("wma".to_string()),
                slow_d_period: Some(5),
                slow_d_ma_type: Some("wma".to_string()),
            },
            KdjParams {
                fast_k_period: Some(14),
                slow_k_period: Some(3),
                slow_k_ma_type: Some("sma".to_string()),
                slow_d_period: Some(3),
                slow_d_ma_type: Some("sma".to_string()),
            },
            KdjParams {
                fast_k_period: Some(20),
                slow_k_period: Some(4),
                slow_k_ma_type: Some("ema".to_string()),
                slow_d_period: Some(6),
                slow_d_ma_type: Some("ema".to_string()),
            },
            KdjParams {
                fast_k_period: Some(30),
                slow_k_period: Some(6),
                slow_k_ma_type: Some("hma".to_string()),
                slow_d_period: Some(8),
                slow_d_ma_type: Some("hma".to_string()),
            },
            KdjParams {
                fast_k_period: Some(50),
                slow_k_period: Some(10),
                slow_k_ma_type: Some("sma".to_string()),
                slow_d_period: Some(10),
                slow_d_ma_type: Some("sma".to_string()),
            },
            KdjParams {
                fast_k_period: Some(100),
                slow_k_period: Some(20),
                slow_k_ma_type: Some("ema".to_string()),
                slow_d_period: Some(20),
                slow_d_ma_type: Some("ema".to_string()),
            },
            KdjParams {
                fast_k_period: Some(200),
                slow_k_period: Some(30),
                slow_k_ma_type: Some("sma".to_string()),
                slow_d_period: Some(30),
                slow_d_ma_type: Some("sma".to_string()),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = KdjInput::from_candles(&candles, params.clone());
            let output = kdj_with_kernel(&input, kernel)?;

            // Check K values
            for (i, &val) in output.k.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in K \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in K \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in K \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }
            }

            // Check D values
            for (i, &val) in output.d.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in D \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in D \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in D \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }
            }

            // Check J values
            for (i, &val) in output.j.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in J \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in J \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in J \
						 with params: fast_k_period={}, slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.fast_k_period.unwrap_or(9),
						params.slow_k_period.unwrap_or(3),
						params.slow_k_ma_type.as_deref().unwrap_or("sma"),
						params.slow_d_period.unwrap_or(3),
						params.slow_d_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_kdj_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_kdj_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic KDJ parameters and price data
        let strat = (5usize..=21, 2usize..=5, 2usize..=5).prop_flat_map(
            |(fast_k_period, slow_k_period, slow_d_period)| {
                (
                    // Generate base price, volatility, and data length
                    (
                        100f64..10000f64,
                        0.01f64..0.05f64,
                        fast_k_period + 10..400,
                        0u8..100u8,
                    )
                        .prop_flat_map(move |(base_price, volatility, data_len, scenario_type)| {
                            // Generate random changes for price movement
                            (
                                Just(base_price),
                                Just(volatility),
                                Just(data_len),
                                Just(scenario_type),
                                prop::collection::vec((-1f64..1f64), data_len),
                                prop::collection::vec((0.001f64..0.02f64), data_len), // Spread factors
                                prop::collection::vec(prop::bool::ANY, data_len), // Zero spread flags
                            )
                        })
                        .prop_map(
                            move |(
                                base_price,
                                volatility,
                                data_len,
                                scenario_type,
                                price_changes,
                                spread_factors,
                                zero_spread_flags,
                            )| {
                                // Generate synthetic high/low/close data with realistic movement
                                let mut high = Vec::with_capacity(data_len);
                                let mut low = Vec::with_capacity(data_len);
                                let mut close = Vec::with_capacity(data_len);
                                let mut current_price = base_price;

                                // Scenario types: 0-70 = normal, 70-85 = uptrend, 85-95 = downtrend, 95-100 = flat/zero-spread
                                for i in 0..data_len {
                                    let (h, l, c) = if scenario_type >= 95 && i > fast_k_period {
                                        // Flat/zero-spread scenario (5% chance)
                                        (current_price, current_price, current_price)
                                    } else if scenario_type >= 85 && scenario_type < 95 {
                                        // Strong downtrend scenario (10% chance)
                                        current_price = (current_price * 0.99).max(10.0);
                                        let spread = current_price * spread_factors[i] * 0.5;
                                        (
                                            current_price + spread * 0.3,
                                            current_price - spread,
                                            current_price - spread * 0.7,
                                        )
                                    } else if scenario_type >= 70 && scenario_type < 85 {
                                        // Strong uptrend scenario (15% chance)
                                        current_price = current_price * 1.01;
                                        let spread = current_price * spread_factors[i] * 0.5;
                                        (
                                            current_price + spread,
                                            current_price - spread * 0.3,
                                            current_price + spread * 0.7,
                                        )
                                    } else {
                                        // Normal random walk (70% chance)
                                        let change = price_changes[i] * volatility * current_price;
                                        current_price = (current_price + change).max(10.0);

                                        // Occasionally inject zero-spread periods
                                        if zero_spread_flags[i] && i % 10 == 0 {
                                            (current_price, current_price, current_price)
                                        } else {
                                            let spread = current_price * spread_factors[i];
                                            let half_spread = spread / 2.0;
                                            let close_position = (price_changes[i] + 1.0) / 2.0;
                                            let c = current_price - half_spread
                                                + spread * close_position;
                                            (
                                                (current_price + half_spread).max(c),
                                                (current_price - half_spread).min(c),
                                                c,
                                            )
                                        }
                                    };

                                    high.push(h);
                                    low.push(l);
                                    close.push(c);
                                }

                                (high, low, close)
                            },
                        ),
                    Just(fast_k_period),
                    Just(slow_k_period),
                    Just(slow_d_period),
                )
            },
        );

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |((high, low, close), fast_k_period, slow_k_period, slow_d_period)| {
                let params = KdjParams {
                    fast_k_period: Some(fast_k_period),
                    slow_k_period: Some(slow_k_period),
                    slow_k_ma_type: Some("sma".to_string()),
                    slow_d_period: Some(slow_d_period),
                    slow_d_ma_type: Some("sma".to_string()),
                };
                let input = KdjInput::from_slices(&high, &low, &close, params.clone());

                let KdjOutput { k, d, j } = kdj_with_kernel(&input, kernel).unwrap();
                let KdjOutput {
                    k: ref_k,
                    d: ref_d,
                    j: ref_j,
                } = kdj_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length should match input length
                prop_assert_eq!(k.len(), high.len(), "K length mismatch");
                prop_assert_eq!(d.len(), high.len(), "D length mismatch");
                prop_assert_eq!(j.len(), high.len(), "J length mismatch");

                // Property 2: Warmup period - should have NaN values at the beginning
                // KDJ has complex warmup: stoch warmup + slow_k smoothing + slow_d smoothing
                let first_valid_idx = high
                    .iter()
                    .zip(low.iter())
                    .zip(close.iter())
                    .position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
                    .unwrap_or(0);
                let stoch_warmup = first_valid_idx + fast_k_period - 1;
                let k_warmup = stoch_warmup + slow_k_period - 1;
                let d_warmup = k_warmup + slow_d_period - 1;

                // K should be NaN before k_warmup
                for i in 0..k_warmup.min(k.len()) {
                    prop_assert!(
                        k[i].is_nan(),
                        "K[{}] should be NaN during warmup but was {}",
                        i,
                        k[i]
                    );
                }
                // D should be NaN before d_warmup
                for i in 0..d_warmup.min(d.len()) {
                    prop_assert!(
                        d[i].is_nan(),
                        "D[{}] should be NaN during warmup but was {}",
                        i,
                        d[i]
                    );
                }
                // J should be NaN before d_warmup (since J depends on both K and D)
                for i in 0..d_warmup.min(j.len()) {
                    prop_assert!(
                        j[i].is_nan(),
                        "J[{}] should be NaN during warmup but was {}",
                        i,
                        j[i]
                    );
                }

                // Property 3: K and D values should be in [0, 100] range (stochastic oscillator bounds)
                for i in k_warmup..k.len() {
                    if !k[i].is_nan() {
                        prop_assert!(
                            k[i] >= -1e-9 && k[i] <= 100.0 + 1e-9,
                            "K[{}] = {} is outside [0, 100] range",
                            i,
                            k[i]
                        );
                    }
                }
                for i in d_warmup..d.len() {
                    if !d[i].is_nan() {
                        prop_assert!(
                            d[i] >= -1e-9 && d[i] <= 100.0 + 1e-9,
                            "D[{}] = {} is outside [0, 100] range",
                            i,
                            d[i]
                        );
                    }
                }

                // Property 4: J = 3*K - 2*D relationship should hold
                for i in d_warmup..j.len() {
                    if !k[i].is_nan() && !d[i].is_nan() && !j[i].is_nan() {
                        let expected_j = 3.0 * k[i] - 2.0 * d[i];
                        prop_assert!(
                            (j[i] - expected_j).abs() <= 1e-9,
                            "J[{}] = {} but expected {} (3*K - 2*D = 3*{} - 2*{})",
                            i,
                            j[i],
                            expected_j,
                            k[i],
                            d[i]
                        );
                    }
                }

                // Property 5: With zero spread (high=low), stochastic should produce NaN
                // This tests the mathematical edge case where the denominator becomes 0
                for i in stoch_warmup..high.len().min(stoch_warmup + fast_k_period * 2) {
                    // Check windows where all highs equal all lows in the period
                    if i >= fast_k_period {
                        let window_start = i + 1 - fast_k_period;
                        let all_zero_spread =
                            (window_start..=i).all(|j| (high[j] - low[j]).abs() < 1e-10);

                        if all_zero_spread && i >= k_warmup {
                            // When hh - ll = 0, K should be NaN
                            prop_assert!(
                                k[i].is_nan(),
                                "K[{}] should be NaN when high=low in window, but was {}",
                                i,
                                k[i]
                            );
                        }
                    }
                }

                // Property 7: J can exceed [0, 100] bounds (this is valid behavior)
                // J = 3*K - 2*D, so if K=90 and D=30, J=210 which is valid
                let mut j_outside_bounds_found = false;
                for i in d_warmup..j.len() {
                    if !j[i].is_nan() {
                        if j[i] < -1e-9 || j[i] > 100.0 + 1e-9 {
                            j_outside_bounds_found = true;
                            // Verify this is consistent with the formula
                            let expected_j = 3.0 * k[i] - 2.0 * d[i];
                            prop_assert!(
                                (j[i] - expected_j).abs() <= 1e-9,
                                "J[{}] = {} is outside [0,100] but doesn't match formula 3*K - 2*D",
                                i,
                                j[i]
                            );
                        }
                    }
                }
                // Note: We don't require j_outside_bounds_found to be true as it depends on data

                // Property 8: Trend behavior validation
                // Check if we have a strong trend scenario
                let mut trend_sum = 0.0;
                for i in 1..high.len().min(50) {
                    trend_sum += close[i] - close[i - 1];
                }

                if high.len() > d_warmup + 20 {
                    let avg_change = trend_sum / (high.len().min(50) - 1) as f64;
                    let first_close = close[0];

                    // Strong uptrend: average change > 0.5% of initial price
                    if avg_change > first_close * 0.005 {
                        // In strong uptrend, K should tend toward higher values
                        let last_valid_k = k
                            .iter()
                            .rev()
                            .find(|&&x| !x.is_nan())
                            .copied()
                            .unwrap_or(0.0);

                        // We expect K to be above 50 in a strong uptrend (not strict, just tendency)
                        if last_valid_k < 30.0 {
                            // This is a soft check - log but don't fail
                            // Strong uptrends should generally push K higher
                        }
                    }

                    // Strong downtrend: average change < -0.5% of initial price
                    if avg_change < -first_close * 0.005 {
                        // In strong downtrend, K should tend toward lower values
                        let last_valid_k = k
                            .iter()
                            .rev()
                            .find(|&&x| !x.is_nan())
                            .copied()
                            .unwrap_or(100.0);

                        // We expect K to be below 50 in a strong downtrend (not strict, just tendency)
                        if last_valid_k > 70.0 {
                            // This is a soft check - log but don't fail
                            // Strong downtrends should generally push K lower
                        }
                    }
                }

                // Property 6: Kernel consistency - different kernels should produce same results
                // Allow for small floating-point differences (ULP comparison)
                for i in 0..k.len() {
                    let k_bits = k[i].to_bits();
                    let ref_k_bits = ref_k[i].to_bits();
                    let d_bits = d[i].to_bits();
                    let ref_d_bits = ref_d[i].to_bits();
                    let j_bits = j[i].to_bits();
                    let ref_j_bits = ref_j[i].to_bits();

                    if k[i].is_nan() && ref_k[i].is_nan() {
                        // Both NaN is fine
                    } else if !k[i].is_nan() && !ref_k[i].is_nan() {
                        let ulp_diff = if k_bits > ref_k_bits {
                            k_bits - ref_k_bits
                        } else {
                            ref_k_bits - k_bits
                        };
                        prop_assert!(
                            ulp_diff <= 5,
                            "K[{}]: kernel {} gives {} but scalar gives {} (ULP diff: {})",
                            i,
                            kernel as u8,
                            k[i],
                            ref_k[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert!(false, "K[{}]: NaN mismatch between kernels", i);
                    }

                    if d[i].is_nan() && ref_d[i].is_nan() {
                        // Both NaN is fine
                    } else if !d[i].is_nan() && !ref_d[i].is_nan() {
                        let ulp_diff = if d_bits > ref_d_bits {
                            d_bits - ref_d_bits
                        } else {
                            ref_d_bits - d_bits
                        };
                        prop_assert!(
                            ulp_diff <= 5,
                            "D[{}]: kernel {} gives {} but scalar gives {} (ULP diff: {})",
                            i,
                            kernel as u8,
                            d[i],
                            ref_d[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert!(false, "D[{}]: NaN mismatch between kernels", i);
                    }

                    if j[i].is_nan() && ref_j[i].is_nan() {
                        // Both NaN is fine
                    } else if !j[i].is_nan() && !ref_j[i].is_nan() {
                        let ulp_diff = if j_bits > ref_j_bits {
                            j_bits - ref_j_bits
                        } else {
                            ref_j_bits - j_bits
                        };
                        prop_assert!(
                            ulp_diff <= 10, // J has more computation, allow slightly more ULP difference
                            "J[{}]: kernel {} gives {} but scalar gives {} (ULP diff: {})",
                            i,
                            kernel as u8,
                            j[i],
                            ref_j[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert!(false, "J[{}]: NaN mismatch between kernels", i);
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_kdj_tests {
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

    generate_all_kdj_tests!(
        check_kdj_partial_params,
        check_kdj_accuracy,
        check_kdj_default_candles,
        check_kdj_zero_period,
        check_kdj_period_exceeds_length,
        check_kdj_very_small_dataset,
        check_kdj_all_nan,
        check_kdj_reinput,
        check_kdj_nan_handling,
        check_kdj_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_kdj_tests!(check_kdj_property);

    // Batch test, matches alma batch style
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = KdjBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = KdjParams::default();
        let row = output.k_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Just check presence, no golden values in this batch stub.
        for &v in &row[row.len().saturating_sub(5)..] {
            assert!(!v.is_nan(), "[{test}] default-row unexpected NaN at tail");
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        // Each tuple: (fast_k_start, fast_k_end, fast_k_step, slow_k_start, slow_k_end, slow_k_step,
        //              slow_d_start, slow_d_end, slow_d_step, slow_k_ma_type, slow_d_ma_type)
        let test_configs = vec![
            // Small periods with default MA types
            (2, 10, 2, 2, 6, 2, 2, 6, 2, "sma", "sma"),
            // Medium periods with EMA
            (5, 25, 5, 3, 9, 3, 3, 9, 3, "ema", "ema"),
            // Large periods with mixed MA types
            (30, 60, 15, 5, 15, 5, 5, 15, 5, "sma", "ema"),
            // Dense small range with WMA
            (2, 5, 1, 2, 4, 1, 2, 4, 1, "wma", "wma"),
            // Edge case: minimum periods
            (2, 2, 0, 2, 2, 0, 2, 2, 0, "sma", "sma"),
            // Common trading setup
            (9, 15, 3, 3, 6, 3, 3, 6, 3, "sma", "sma"),
            // Large periods with HMA
            (50, 100, 25, 10, 20, 10, 10, 20, 10, "hma", "hma"),
        ];

        for (
            cfg_idx,
            &(
                fk_start,
                fk_end,
                fk_step,
                sk_start,
                sk_end,
                sk_step,
                sd_start,
                sd_end,
                sd_step,
                sk_ma,
                sd_ma,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = KdjBatchBuilder::new()
                .kernel(kernel)
                .fast_k_period_range(fk_start, fk_end, fk_step)
                .slow_k_period_range(sk_start, sk_end, sk_step)
                .slow_k_ma_type_static(sk_ma)
                .slow_d_period_range(sd_start, sd_end, sd_step)
                .slow_d_ma_type_static(sd_ma)
                .apply_candles(&c)?;

            // Check K values
            for (idx, &val) in output.k.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in K with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }
            }

            // Check D values
            for (idx, &val) in output.d.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in D with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }
            }

            // Check J values
            for (idx, &val) in output.j.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in J with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in J with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in J with params: fast_k_period={}, \
						 slow_k_period={}, slow_k_ma_type={}, slow_d_period={}, slow_d_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_k_period.unwrap_or(9),
                        combo.slow_k_period.unwrap_or(3),
                        combo.slow_k_ma_type.as_deref().unwrap_or("sma"),
                        combo.slow_d_period.unwrap_or(3),
                        combo.slow_d_ma_type.as_deref().unwrap_or("sma")
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

// ========== Python Bindings ==========

#[cfg(feature = "python")]
#[pyfunction(name = "kdj")]
#[pyo3(signature = (high, low, close, fast_k_period=9, slow_k_period=3, slow_k_ma_type="sma", slow_d_period=3, slow_d_ma_type="sma", kernel=None))]
pub fn kdj_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    use numpy::PyArray1;

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let params = KdjParams {
        fast_k_period: Some(fast_k_period),
        slow_k_period: Some(slow_k_period),
        slow_k_ma_type: Some(slow_k_ma_type.to_string()),
        slow_d_period: Some(slow_d_period),
        slow_d_ma_type: Some(slow_d_ma_type.to_string()),
    };
    let inp = KdjInput::from_slices(h, l, c, params);
    let kern = validate_kernel(kernel, false)?;

    let (rows, cols) = (1, c.len());
    let k_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let d_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let j_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

    let k_slice = unsafe { k_arr.as_slice_mut()? };
    let d_slice = unsafe { d_arr.as_slice_mut()? };
    let j_slice = unsafe { j_arr.as_slice_mut()? };

    py.allow_threads(|| kdj_into_slices(k_slice, d_slice, j_slice, &inp, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((k_arr, d_arr, j_arr))
}

#[cfg(feature = "python")]
#[pyclass(name = "KdjStream")]
pub struct KdjStreamPy {
    stream: KdjStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KdjStreamPy {
    #[new]
    fn new(
        fast_k_period: usize,
        slow_k_period: usize,
        slow_k_ma_type: &str,
        slow_d_period: usize,
        slow_d_ma_type: &str,
    ) -> PyResult<Self> {
        let params = KdjParams {
            fast_k_period: Some(fast_k_period),
            slow_k_period: Some(slow_k_period),
            slow_k_ma_type: Some(slow_k_ma_type.to_string()),
            slow_d_period: Some(slow_d_period),
            slow_d_ma_type: Some(slow_d_ma_type.to_string()),
        };
        let stream =
            KdjStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KdjStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "kdj_batch")]
#[pyo3(signature = (high, low, close,
                    fast_k_range, slow_k_range, slow_k_ma_type,
                    slow_d_range, slow_d_ma_type, kernel=None))]
pub fn kdj_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fast_k_range: (usize, usize, usize),
    slow_k_range: (usize, usize, usize),
    slow_k_ma_type: &str,
    slow_d_range: (usize, usize, usize),
    slow_d_ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArray1, PyArrayMethods};

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;

    let range = KdjBatchRange {
        fast_k_period: fast_k_range,
        slow_k_period: slow_k_range,
        slow_k_ma_type: (
            slow_k_ma_type.to_string(),
            slow_k_ma_type.to_string(),
            "".to_string(),
        ),
        slow_d_period: slow_d_range,
        slow_d_ma_type: (
            slow_d_ma_type.to_string(),
            slow_d_ma_type.to_string(),
            "".to_string(),
        ),
    };

    let kern = validate_kernel(kernel, true)?;
    let combos;
    let rows;
    let cols = c.len();

    let k_arr = unsafe { PyArray1::<f64>::new(py, [1], false) }; // placeholder shape
    let d_arr = unsafe { PyArray1::<f64>::new(py, [1], false) };
    let j_arr = unsafe { PyArray1::<f64>::new(py, [1], false) };

    let (k_vec, d_vec, j_vec, cmbs, rws) = py
        .allow_threads(|| {
            // Use the batch helper that allocates with make_uninit_matrix and returns Vecs without extra copy.
            let out = kdj_batch_inner(
                h,
                l,
                c,
                &range,
                match kern {
                    Kernel::Avx512Batch => Kernel::Avx512,
                    Kernel::Avx2Batch => Kernel::Avx2,
                    Kernel::ScalarBatch => Kernel::Scalar,
                    Kernel::Auto => detect_best_batch_kernel(),
                    _ => kern,
                },
                true,
            )?;
            Ok::<_, KdjError>((out.k, out.d, out.j, out.combos, out.rows))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    combos = cmbs;
    rows = rws;

    let k_arr = k_vec.into_pyarray(py).reshape((rows, cols))?;
    let d_arr = d_vec.into_pyarray(py).reshape((rows, cols))?;
    let j_arr = j_vec.into_pyarray(py).reshape((rows, cols))?;

    let dict = PyDict::new(py);
    dict.set_item("k", k_arr)?;
    dict.set_item("d", d_arr)?;
    dict.set_item("j", j_arr)?;
    dict.set_item(
        "fast_k_periods",
        combos
            .iter()
            .map(|p| p.fast_k_period.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_k_periods",
        combos
            .iter()
            .map(|p| p.slow_k_period.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_d_periods",
        combos
            .iter()
            .map(|p| p.slow_d_period.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    // Add full combos for API parity with ALMA
    let combo_list = PyList::new(
        py,
        combos.iter().map(|c| {
            let combo_dict = PyDict::new(py);
            combo_dict
                .set_item("fast_k_period", c.fast_k_period.unwrap())
                .unwrap();
            combo_dict
                .set_item("slow_k_period", c.slow_k_period.unwrap())
                .unwrap();
            combo_dict
                .set_item(
                    "slow_k_ma_type",
                    c.slow_k_ma_type.as_ref().unwrap().as_str(),
                )
                .unwrap();
            combo_dict
                .set_item("slow_d_period", c.slow_d_period.unwrap())
                .unwrap();
            combo_dict
                .set_item(
                    "slow_d_ma_type",
                    c.slow_d_ma_type.as_ref().unwrap().as_str(),
                )
                .unwrap();
            combo_dict
        }),
    )?;
    dict.set_item("combos", combo_list)?;

    Ok(dict)
}

// ==================== PYTHON CUDA BINDINGS ====================
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaKdj};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kdj_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, fast_k_range, slow_k_range, slow_k_ma_range, slow_d_range, slow_d_ma_range, device_id=0))]
pub fn kdj_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: PyReadonlyArray1<'_, f32>,
    low_f32: PyReadonlyArray1<'_, f32>,
    close_f32: PyReadonlyArray1<'_, f32>,
    fast_k_range: (usize, usize, usize),
    slow_k_range: (usize, usize, usize),
    slow_k_ma_range: (String, String, String),
    slow_d_range: (usize, usize, usize),
    slow_d_ma_range: (String, String, String),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let h = high_f32.as_slice()?; let l = low_f32.as_slice()?; let c = close_f32.as_slice()?;
    let sweep = KdjBatchRange { fast_k_period: fast_k_range, slow_k_period: slow_k_range, slow_k_ma_type: slow_k_ma_range, slow_d_period: slow_d_range, slow_d_ma_type: slow_d_ma_range };
    let (k, d, j) = py.allow_threads(|| {
        let cuda = CudaKdj::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.kdj_batch_dev(h, l, c, &sweep).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((DeviceArrayF32Py { inner: k }, DeviceArrayF32Py { inner: d }, DeviceArrayF32Py { inner: j }))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kdj_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, cols, rows, fast_k, slow_k, slow_k_ma, slow_d, slow_d_ma, device_id=0))]
pub fn kdj_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: PyReadonlyArray1<'_, f32>,
    low_tm_f32: PyReadonlyArray1<'_, f32>,
    close_tm_f32: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    fast_k: usize,
    slow_k: usize,
    slow_k_ma: String,
    slow_d: usize,
    slow_d_ma: String,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let htm = high_tm_f32.as_slice()?; let ltm = low_tm_f32.as_slice()?; let ctm = close_tm_f32.as_slice()?;
    let params = KdjParams { fast_k_period: Some(fast_k), slow_k_period: Some(slow_k), slow_k_ma_type: Some(slow_k_ma), slow_d_period: Some(slow_d), slow_d_ma_type: Some(slow_d_ma) };
    let (k, d, j) = py.allow_threads(|| {
        let cuda = CudaKdj::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.kdj_many_series_one_param_time_major_dev(htm, ltm, ctm, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((DeviceArrayF32Py { inner: k }, DeviceArrayF32Py { inner: d }, DeviceArrayF32Py { inner: j }))
}

// ========== WASM Bindings ==========

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KdjJsOutput {
    pub values: Vec<f64>, // [k..., d..., j...]
    pub rows: usize,      // 3
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "kdj")]
pub fn kdj_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
) -> Result<JsValue, JsValue> {
    let params = KdjParams {
        fast_k_period: Some(fast_k_period),
        slow_k_period: Some(slow_k_period),
        slow_k_ma_type: Some(slow_k_ma_type.to_string()),
        slow_d_period: Some(slow_d_period),
        slow_d_ma_type: Some(slow_d_ma_type.to_string()),
    };
    let input = KdjInput::from_slices(high, low, close, params);

    let mut k = vec![0.0; close.len()];
    let mut d = vec![0.0; close.len()];
    let mut j = vec![0.0; close.len()];
    kdj_into_slices(&mut k, &mut d, &mut j, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = Vec::with_capacity(3 * close.len());
    values.extend_from_slice(&k);
    values.extend_from_slice(&d);
    values.extend_from_slice(&j);
    let result = KdjJsOutput {
        values,
        rows: 3,
        cols: close.len(),
    };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "kdj_alloc")]
pub fn kdj_alloc(len: usize) -> *mut f64 {
    let mut v: Vec<f64> = Vec::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "kdj_free")]
pub fn kdj_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "kdj_into")]
pub fn kdj_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    k_ptr: *mut f64,
    d_ptr: *mut f64,
    j_ptr: *mut f64,
    len: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
) -> Result<(), JsValue> {
    if [
        high_ptr as usize,
        low_ptr as usize,
        close_ptr as usize,
        k_ptr as usize,
        d_ptr as usize,
        j_ptr as usize,
    ]
    .iter()
    .any(|&p| p == 0)
    {
        return Err(JsValue::from_str("null pointer passed to kdj_into"));
    }
    unsafe {
        let h = std::slice::from_raw_parts(high_ptr, len);
        let l = std::slice::from_raw_parts(low_ptr, len);
        let c = std::slice::from_raw_parts(close_ptr, len);
        let k = std::slice::from_raw_parts_mut(k_ptr, len);
        let d = std::slice::from_raw_parts_mut(d_ptr, len);
        let j = std::slice::from_raw_parts_mut(j_ptr, len);

        let params = KdjParams {
            fast_k_period: Some(fast_k_period),
            slow_k_period: Some(slow_k_period),
            slow_k_ma_type: Some(slow_k_ma_type.to_string()),
            slow_d_period: Some(slow_d_period),
            slow_d_ma_type: Some(slow_d_ma_type.to_string()),
        };
        let input = KdjInput::from_slices(h, l, c, params);
        kdj_into_slices(k, d, j, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KdjBatchConfig {
    pub fast_k_period: (usize, usize, usize),
    pub slow_k_period: (usize, usize, usize),
    pub slow_k_ma_type: String,
    pub slow_d_period: (usize, usize, usize),
    pub slow_d_ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KdjBatchJsOutput {
    pub values: Vec<f64>, // concatenated per combo: [k..., d..., j..., k..., d..., j..., ...]
    pub combos: Vec<KdjParams>,
    pub rows: usize, // combos * 3
    pub cols: usize, // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "kdj_batch")]
pub fn kdj_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: KdjBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = KdjBatchRange {
        fast_k_period: cfg.fast_k_period,
        slow_k_period: cfg.slow_k_period,
        slow_k_ma_type: (
            cfg.slow_k_ma_type.clone(),
            cfg.slow_k_ma_type,
            "".to_string(),
        ),
        slow_d_period: cfg.slow_d_period,
        slow_d_ma_type: (
            cfg.slow_d_ma_type.clone(),
            cfg.slow_d_ma_type,
            "".to_string(),
        ),
    };
    let out = kdj_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = Vec::with_capacity(out.rows * 3 * out.cols);
    for row in 0..out.rows {
        let s = row * out.cols;
        values.extend_from_slice(&out.k[s..s + out.cols]);
        values.extend_from_slice(&out.d[s..s + out.cols]);
        values.extend_from_slice(&out.j[s..s + out.cols]);
    }
    let js = KdjBatchJsOutput {
        values,
        combos: out.combos,
        rows: out.rows * 3,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

// kdj_batch_into removed - use kdj_batch instead

// === Classic Kernel Implementations ===

/// Classic kernel with inline SMA calculations for both K and D smoothing
#[inline]
fn kdj_classic_sma(
    stoch: &[f64],
    slow_k: usize,
    slow_d: usize,
    stoch_warm: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
    j_out: &mut [f64],
) -> Result<(), KdjError> {
    let len = stoch.len();

    // Initialize K output with NaN prefix
    let k_warm = stoch_warm + slow_k - 1;
    for i in 0..k_warm.min(len) {
        k_out[i] = f64::NAN;
    }

    // Calculate %K (SMA of stochastic)
    let mut sum_k = 0.0;
    let mut count_k = 0;

    // Initialize first SMA for K
    for i in stoch_warm..(stoch_warm + slow_k).min(len) {
        if !stoch[i].is_nan() {
            sum_k += stoch[i];
            count_k += 1;
        }
    }

    if count_k > 0 && k_warm < len {
        k_out[k_warm] = sum_k / count_k as f64;

        // Rolling SMA for K
        for i in (k_warm + 1)..len {
            let old_val = stoch[i - slow_k];
            let new_val = stoch[i];
            if !old_val.is_nan() {
                sum_k -= old_val;
                count_k -= 1;
            }
            if !new_val.is_nan() {
                sum_k += new_val;
                count_k += 1;
            }
            k_out[i] = if count_k > 0 {
                sum_k / count_k as f64
            } else {
                f64::NAN
            };
        }
    }

    // Initialize D output with NaN prefix
    let d_warm = k_warm + slow_d - 1;
    for i in 0..d_warm.min(len) {
        d_out[i] = f64::NAN;
    }

    // Calculate %D (SMA of %K)
    let mut sum_d = 0.0;
    let mut count_d = 0;

    // Initialize first SMA for D
    for i in k_warm..(k_warm + slow_d).min(len) {
        if !k_out[i].is_nan() {
            sum_d += k_out[i];
            count_d += 1;
        }
    }

    if count_d > 0 && d_warm < len {
        d_out[d_warm] = sum_d / count_d as f64;

        // Rolling SMA for D
        for i in (d_warm + 1)..len {
            let old_val = k_out[i - slow_d];
            let new_val = k_out[i];
            if !old_val.is_nan() {
                sum_d -= old_val;
                count_d -= 1;
            }
            if !new_val.is_nan() {
                sum_d += new_val;
                count_d += 1;
            }
            d_out[i] = if count_d > 0 {
                sum_d / count_d as f64
            } else {
                f64::NAN
            };
        }
    }

    // Calculate J = 3K - 2D
    for i in 0..d_warm.min(len) {
        j_out[i] = f64::NAN;
    }
    for i in d_warm..len {
        j_out[i] = if k_out[i].is_nan() || d_out[i].is_nan() {
            f64::NAN
        } else {
            3.0 * k_out[i] - 2.0 * d_out[i]
        };
    }

    Ok(())
}

/// Classic kernel with inline EMA calculations for both K and D smoothing
#[inline]
fn kdj_classic_ema(
    stoch: &[f64],
    slow_k: usize,
    slow_d: usize,
    stoch_warm: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
    j_out: &mut [f64],
) -> Result<(), KdjError> {
    let len = stoch.len();

    // Initialize K output with NaN prefix
    let k_warm = stoch_warm + slow_k - 1;
    for i in 0..k_warm.min(len) {
        k_out[i] = f64::NAN;
    }

    // Calculate %K (EMA of stochastic)
    let alpha_k = 2.0 / (slow_k as f64 + 1.0);
    let one_minus_alpha_k = 1.0 - alpha_k;

    // Initialize EMA for K with SMA
    let mut sum_k = 0.0;
    let mut count_k = 0;
    for i in stoch_warm..(stoch_warm + slow_k).min(len) {
        if !stoch[i].is_nan() {
            sum_k += stoch[i];
            count_k += 1;
        }
    }

    if count_k > 0 && k_warm < len {
        let mut ema_k = sum_k / count_k as f64;
        k_out[k_warm] = ema_k;

        // Continue EMA for K
        for i in (k_warm + 1)..len {
            if !stoch[i].is_nan() {
                ema_k = stoch[i].mul_add(alpha_k, one_minus_alpha_k * ema_k);
                k_out[i] = ema_k;
            } else {
                k_out[i] = ema_k; // Carry forward last valid value
            }
        }
    }

    // Initialize D output with NaN prefix
    let d_warm = k_warm + slow_d - 1;
    for i in 0..d_warm.min(len) {
        d_out[i] = f64::NAN;
    }

    // Calculate %D (EMA of %K)
    let alpha_d = 2.0 / (slow_d as f64 + 1.0);
    let one_minus_alpha_d = 1.0 - alpha_d;

    // Initialize EMA for D with SMA
    let mut sum_d = 0.0;
    let mut count_d = 0;
    for i in k_warm..(k_warm + slow_d).min(len) {
        if !k_out[i].is_nan() {
            sum_d += k_out[i];
            count_d += 1;
        }
    }

    if count_d > 0 && d_warm < len {
        let mut ema_d = sum_d / count_d as f64;
        d_out[d_warm] = ema_d;

        // Continue EMA for D
        for i in (d_warm + 1)..len {
            if !k_out[i].is_nan() {
                ema_d = k_out[i].mul_add(alpha_d, one_minus_alpha_d * ema_d);
                d_out[i] = ema_d;
            } else {
                d_out[i] = ema_d; // Carry forward last valid value
            }
        }
    }

    // Calculate J = 3K - 2D
    for i in 0..d_warm.min(len) {
        j_out[i] = f64::NAN;
    }
    for i in d_warm..len {
        j_out[i] = if k_out[i].is_nan() || d_out[i].is_nan() {
            f64::NAN
        } else {
            3.0 * k_out[i] - 2.0 * d_out[i]
        };
    }

    Ok(())
}
