//! # AlphaTrend (AT)
//!
//! A trend-following indicator that creates dynamic support/resistance bands using ATR
//! and either RSI or MFI to determine trend direction. Developed by KivancOzbilgic.
//!
//! ## Parameters
//! - **coeff**: Multiplier for ATR bands (default: 1.0)
//! - **period**: Common period for ATR, RSI/MFI calculations (default: 14)
//! - **no_volume**: Use RSI instead of MFI when true (default: false)
//!
//! ## Returns
//! - **`Ok(AlphaTrendOutput)`** on success, containing `k1` and `k2` vectors of length matching the input.
//! - **`Err(AlphaTrendError)`** otherwise.
//!
//! ## Developer Status
//! - SIMD implemented for TR/HLC3 (AVX2/AVX512), ATR+line kept scalar due to recurrence.
//! - Runtime selection: Auto keeps kernel detection (SIMD measured >5% faster at 100k/1M here).
//! - Row-specific batch kernels: not implemented; opportunity exists to precompute TR and
//!   momentum per unique period across rows. Deferred to a follow-up.
//! - Streaming update: O(1) kernel implemented (ATR via SMA ring, RSI-RMA/MFI sums).
//!   Emits Some((k1,k2)) after warmup+2 bars; matches scalar semantics.
//! - Memory: uses zero-copy/uninit helpers (alloc_with_nan_prefix, make_uninit_matrix).

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
use std::mem::MaybeUninit;
use thiserror::Error;

use crate::indicators::mfi::{mfi_with_kernel, MfiInput, MfiParams};
use crate::indicators::rsi::{rsi_with_kernel, RsiInput, RsiParams};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::alphatrend_wrapper::CudaAlphaTrend;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

impl<'a> AsRef<[f64]> for AlphaTrendInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AlphaTrendData::Slices { close, .. } => close,
            AlphaTrendData::Candles { candles, .. } => &candles.close,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlphaTrendData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AlphaTrendOutput {
    pub k1: Vec<f64>, // Current AlphaTrend line
    pub k2: Vec<f64>, // AlphaTrend[2] - lagged by 2 bars
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AlphaTrendParams {
    pub coeff: Option<f64>,
    pub period: Option<usize>,
    pub no_volume: Option<bool>,
}

impl Default for AlphaTrendParams {
    fn default() -> Self {
        Self {
            coeff: Some(1.0),
            period: Some(14),
            no_volume: Some(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlphaTrendInput<'a> {
    pub data: AlphaTrendData<'a>,
    pub params: AlphaTrendParams,
}

impl<'a> AlphaTrendInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: AlphaTrendParams) -> Self {
        Self {
            data: AlphaTrendData::Candles { candles: c },
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        p: AlphaTrendParams,
    ) -> Self {
        Self {
            data: AlphaTrendData::Slices {
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
        Self::from_candles(c, AlphaTrendParams::default())
    }

    #[inline]
    pub fn get_coeff(&self) -> f64 {
        self.params.coeff.unwrap_or(1.0)
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }

    #[inline]
    pub fn get_no_volume(&self) -> bool {
        self.params.no_volume.unwrap_or(false)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AlphaTrendBuilder {
    coeff: Option<f64>,
    period: Option<usize>,
    no_volume: Option<bool>,
    kernel: Kernel,
}

impl Default for AlphaTrendBuilder {
    fn default() -> Self {
        Self {
            coeff: None,
            period: None,
            no_volume: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AlphaTrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn coeff(mut self, val: f64) -> Self {
        self.coeff = Some(val);
        self
    }

    #[inline(always)]
    pub fn period(mut self, val: usize) -> Self {
        self.period = Some(val);
        self
    }

    #[inline(always)]
    pub fn no_volume(mut self, val: bool) -> Self {
        self.no_volume = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AlphaTrendOutput, AlphaTrendError> {
        let p = AlphaTrendParams {
            coeff: self.coeff,
            period: self.period,
            no_volume: self.no_volume,
        };
        let i = AlphaTrendInput::from_candles(c, p);
        alphatrend_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AlphaTrendOutput, AlphaTrendError> {
        let p = AlphaTrendParams {
            coeff: self.coeff,
            period: self.period,
            no_volume: self.no_volume,
        };
        let i = AlphaTrendInput::from_slices(open, high, low, close, volume, p);
        alphatrend_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AlphaTrendStream, AlphaTrendError> {
        let p = AlphaTrendParams {
            coeff: self.coeff,
            period: self.period,
            no_volume: self.no_volume,
        };
        AlphaTrendStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AlphaTrendError {
    #[error("alphatrend: Input data slice is empty.")]
    EmptyInputData,

    #[error("alphatrend: All values are NaN.")]
    AllValuesNaN,

    #[error("alphatrend: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("alphatrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("alphatrend: Inconsistent data lengths")]
    InconsistentDataLengths,

    #[error("alphatrend: Invalid coefficient: {coeff}")]
    InvalidCoeff { coeff: f64 },

    #[error("alphatrend: RSI calculation failed: {msg}")]
    RsiError { msg: String },

    #[error("alphatrend: MFI calculation failed: {msg}")]
    MfiError { msg: String },
}

#[inline]
pub fn alphatrend(input: &AlphaTrendInput) -> Result<AlphaTrendOutput, AlphaTrendError> {
    alphatrend_with_kernel(input, Kernel::Auto)
}

pub fn alphatrend_with_kernel(
    input: &AlphaTrendInput,
    kernel: Kernel,
) -> Result<AlphaTrendOutput, AlphaTrendError> {
    let (open, high, low, close, volume, coeff, period, no_volume, first, chosen) =
        alphatrend_prepare(input, kernel)?;

    let len = close.len();
    let warm = first + period - 1;

    let mut k1 = alloc_with_nan_prefix(len, warm);
    let mut k2 = alloc_with_nan_prefix(len, warm + 2); // K2 has 2-bar lag warmup

    alphatrend_compute_into(
        open, high, low, close, volume, coeff, period, no_volume, first, chosen, &mut k1, &mut k2,
    )?;

    Ok(AlphaTrendOutput { k1, k2 })
}

#[inline]
pub fn alphatrend_into_slices(
    dst_k1: &mut [f64],
    dst_k2: &mut [f64],
    input: &AlphaTrendInput,
    kern: Kernel,
) -> Result<(), AlphaTrendError> {
    let (open, high, low, close, volume, coeff, period, no_volume, first, chosen) =
        alphatrend_prepare(input, kern)?;

    if dst_k1.len() != close.len() || dst_k2.len() != close.len() {
        return Err(AlphaTrendError::InconsistentDataLengths);
    }

    let warm = first + period - 1;
    let k1_warm_end = warm.min(dst_k1.len());
    let k2_warm_end = (warm + 2).min(dst_k2.len());
    for v in &mut dst_k1[..k1_warm_end] {
        *v = f64::NAN;
    }
    for v in &mut dst_k2[..k2_warm_end] {
        *v = f64::NAN;
    }

    alphatrend_compute_into(
        open, high, low, close, volume, coeff, period, no_volume, first, chosen, dst_k1, dst_k2,
    )?;

    Ok(())
}

#[inline(always)]
fn alphatrend_prepare<'a>(
    input: &'a AlphaTrendInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        f64,
        usize,
        bool,
        usize,
        Kernel,
    ),
    AlphaTrendError,
> {
    let (open, high, low, close, volume) = match &input.data {
        AlphaTrendData::Candles { candles } => (
            &candles.open[..],
            &candles.high[..],
            &candles.low[..],
            &candles.close[..],
            &candles.volume[..],
        ),
        AlphaTrendData::Slices {
            open,
            high,
            low,
            close,
            volume,
        } => (*open, *high, *low, *close, *volume),
    };

    let len = close.len();

    if len == 0 {
        return Err(AlphaTrendError::EmptyInputData);
    }

    if open.len() != len || high.len() != len || low.len() != len || volume.len() != len {
        return Err(AlphaTrendError::InconsistentDataLengths);
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlphaTrendError::AllValuesNaN)?;

    let coeff = input.get_coeff();
    let period = input.get_period();
    let no_volume = input.get_no_volume();

    if period == 0 || period > len {
        return Err(AlphaTrendError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    if len - first < period {
        return Err(AlphaTrendError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    if coeff <= 0.0 || !coeff.is_finite() {
        return Err(AlphaTrendError::InvalidCoeff { coeff });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((
        open, high, low, close, volume, coeff, period, no_volume, first, chosen,
    ))
}

#[inline(always)]
fn alphatrend_compute_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    coeff: f64,
    period: usize,
    no_volume: bool,
    first: usize,
    kernel: Kernel,
    out_k1: &mut [f64],
    out_k2: &mut [f64],
) -> Result<(), AlphaTrendError> {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                return alphatrend_simd128(
                    open, high, low, close, volume, coeff, period, no_volume, first, out_k1, out_k2,
                );
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => alphatrend_scalar(
                open, high, low, close, volume, coeff, period, no_volume, first, out_k1, out_k2,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => alphatrend_avx2(
                open, high, low, close, volume, coeff, period, no_volume, first, out_k1, out_k2,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => alphatrend_avx512(
                open, high, low, close, volume, coeff, period, no_volume, first, out_k1, out_k2,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                alphatrend_scalar(
                    open, high, low, close, volume, coeff, period, no_volume, first, out_k1, out_k2,
                )
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn alphatrend_scalar(
    _open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    coeff: f64,
    period: usize,
    no_volume: bool,
    first_val: usize,
    out_k1: &mut [f64],
    out_k2: &mut [f64],
) -> Result<(), AlphaTrendError> {
    let len = close.len();
    let warmup = first_val + period - 1;

    // true_range: uninit then fill [first_val..)
    let mut tr_mu = make_uninit_matrix(1, len);
    let tr: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(tr_mu.as_mut_ptr() as *mut f64, len) };
    // first bar
    if first_val < len {
        tr[first_val] = high[first_val] - low[first_val];
    }
    for i in (first_val + 1)..len {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Momentum series (RSI if no_volume, else MFI using HLC3)
    let momentum_values: Vec<f64> = if no_volume {
        let rsi_params = RsiParams {
            period: Some(period),
        };
        let rsi_input = RsiInput::from_slice(close, rsi_params);
        rsi_with_kernel(&rsi_input, Kernel::Scalar)
            .map_err(|e| AlphaTrendError::RsiError { msg: e.to_string() })?
            .values
    } else {
        // hlc3 without bulk init
        let mut hlc3_mu = make_uninit_matrix(1, len);
        let hlc3: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(hlc3_mu.as_mut_ptr() as *mut f64, len) };
        for i in 0..len {
            hlc3[i] = (high[i] + low[i] + close[i]) / 3.0;
        }
        let mfi_params = MfiParams {
            period: Some(period),
        };
        let mfi_input = MfiInput::from_slices(hlc3, volume, mfi_params);
        mfi_with_kernel(&mfi_input, Kernel::Scalar)
            .map_err(|e| AlphaTrendError::MfiError { msg: e.to_string() })?
            .values
    };
    // O(n) ATR via sliding window + direct AlphaTrend writes
    if warmup < len {
        let mut sum = 0.0f64;
        for j in first_val..=warmup {
            sum += tr[j];
        }

        // Track previous alpha values for k2 (lag-2)
        let mut prev_alpha = f64::NAN;
        let mut prev1 = f64::NAN;
        let mut prev2 = f64::NAN;

        for i in warmup..len {
            let a = sum / period as f64;
            // up = low - coeff*ATR, down = high + coeff*ATR
            let up_t = low[i] - a * coeff;
            let down_t = high[i] + a * coeff;
            let m_check = momentum_values[i] >= 50.0;

            let cur = if i == warmup {
                if m_check {
                    up_t
                } else {
                    down_t
                }
            } else if m_check {
                // rising regime
                if up_t < prev_alpha {
                    prev_alpha
                } else {
                    up_t
                }
            } else {
                // falling regime
                if down_t > prev_alpha {
                    prev_alpha
                } else {
                    down_t
                }
            };

            out_k1[i] = cur;
            if i >= warmup + 2 {
                out_k2[i] = prev2;
            }

            // advance rings
            prev2 = prev1;
            prev1 = cur;
            prev_alpha = cur;

            // slide window to next bar
            if i + 1 < len {
                sum += tr[i + 1] - tr[i + 1 - period];
            }
        }
    }

    // Minimal prefix clearing (matches alloc prefixes)
    for v in &mut out_k1[..warmup.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut out_k2[..(warmup + 2).min(len)] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn alphatrend_avx2(
    _open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    coeff: f64,
    period: usize,
    no_volume: bool,
    first_val: usize,
    out_k1: &mut [f64],
    out_k2: &mut [f64],
) -> Result<(), AlphaTrendError> {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn mm256_abs_pd(x: __m256d) -> __m256d {
        let sign = _mm256_set1_pd(-0.0);
        _mm256_andnot_pd(sign, x)
    }

    let len = close.len();
    let warmup = first_val + period - 1;
    let p_f = period as f64;

    // --------- TR buffer ----------
    let mut tr_mu = make_uninit_matrix(1, len);
    let tr: &mut [f64] = core::slice::from_raw_parts_mut(tr_mu.as_mut_ptr() as *mut f64, len);

    if first_val < len {
        *tr.get_unchecked_mut(first_val) =
            *high.get_unchecked(first_val) - *low.get_unchecked(first_val);
    }

    let mut i = first_val + 1;
    while i + 4 <= len {
        let hv = _mm256_loadu_pd(high.as_ptr().add(i));
        let lv = _mm256_loadu_pd(low.as_ptr().add(i));
        let pc = _mm256_loadu_pd(close.as_ptr().add(i - 1));

        let hl = _mm256_sub_pd(hv, lv);
        let hc = mm256_abs_pd(_mm256_sub_pd(hv, pc));
        let lc = mm256_abs_pd(_mm256_sub_pd(lv, pc));

        let m1 = _mm256_max_pd(hl, hc);
        let m = _mm256_max_pd(m1, lc);
        _mm256_storeu_pd(tr.as_mut_ptr().add(i), m);
        i += 4;
    }
    while i < len {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let pc = *close.get_unchecked(i - 1);
        let hl = hi - lo;
        let hc = (hi - pc).abs();
        let lc = (lo - pc).abs();
        let m = if hl >= hc { hl } else { hc };
        *tr.get_unchecked_mut(i) = if m >= lc { m } else { lc };
        i += 1;
    }

    // --------- Momentum (RSI/MFI). Vectorize HLC3 if needed ----------
    let momentum_values: Vec<f64> = if no_volume {
        let rsi_params = RsiParams {
            period: Some(period),
        };
        let rsi_input = RsiInput::from_slice(close, rsi_params);
        rsi_with_kernel(&rsi_input, Kernel::Avx2)
            .map_err(|e| AlphaTrendError::RsiError { msg: e.to_string() })?
            .values
    } else {
        let mut hlc3_mu = make_uninit_matrix(1, len);
        let hlc3: &mut [f64] =
            core::slice::from_raw_parts_mut(hlc3_mu.as_mut_ptr() as *mut f64, len);

        let inv3 = _mm256_set1_pd(1.0 / 3.0);
        let mut j = 0usize;
        while j + 4 <= len {
            let hv = _mm256_loadu_pd(high.as_ptr().add(j));
            let lv = _mm256_loadu_pd(low.as_ptr().add(j));
            let cv = _mm256_loadu_pd(close.as_ptr().add(j));
            let s = _mm256_add_pd(_mm256_add_pd(hv, lv), cv);
            let h3 = _mm256_mul_pd(s, inv3);
            _mm256_storeu_pd(hlc3.as_mut_ptr().add(j), h3);
            j += 4;
        }
        while j < len {
            *hlc3.get_unchecked_mut(j) =
                (*high.get_unchecked(j) + *low.get_unchecked(j) + *close.get_unchecked(j))
                    * (1.0 / 3.0);
            j += 1;
        }

        let mfi_params = MfiParams {
            period: Some(period),
        };
        let mfi_input = MfiInput::from_slices(hlc3, volume, mfi_params);
        mfi_with_kernel(&mfi_input, Kernel::Avx2)
            .map_err(|e| AlphaTrendError::MfiError { msg: e.to_string() })?
            .values
    };

    // --------- O(n) ATR + AlphaTrend + K2 (scalar due to recurrence) ----------
    let mut sum = 0.0f64;
    {
        let mut j = first_val;
        while j <= warmup {
            sum += *tr.get_unchecked(j);
            j += 1;
        }
    }

    #[inline(always)]
    fn fast_max(a: f64, b: f64) -> f64 {
        if a >= b {
            a
        } else {
            b
        }
    }
    #[inline(always)]
    fn fast_min(a: f64, b: f64) -> f64 {
        if a <= b {
            a
        } else {
            b
        }
    }

    let mut prev2 = f64::NAN;
    let mut prev1 = f64::NAN;
    let mut prev_alpha = f64::NAN;

    let mut k = warmup;
    while k < len {
        let a = sum / p_f;
        let hi = *high.get_unchecked(k);
        let lo = *low.get_unchecked(k);
        let up = (-coeff).mul_add(a, lo);
        let dn = coeff.mul_add(a, hi);
        let m_ge_50 = *momentum_values.get_unchecked(k) >= 50.0;

        let alpha = if k == warmup {
            if m_ge_50 {
                up
            } else {
                dn
            }
        } else if m_ge_50 {
            fast_max(up, prev_alpha)
        } else {
            fast_min(dn, prev_alpha)
        };

        *out_k1.get_unchecked_mut(k) = alpha;
        if k >= warmup + 2 {
            *out_k2.get_unchecked_mut(k) = prev2;
        }

        prev2 = prev1;
        prev1 = alpha;
        prev_alpha = alpha;

        let nxt = k + 1;
        if nxt < len {
            sum += *tr.get_unchecked(nxt) - *tr.get_unchecked(nxt - period);
        }
        k += 1;
    }

    for v in &mut out_k1[..warmup.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut out_k2[..(warmup + 2).min(len)] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn alphatrend_avx512(
    _open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    coeff: f64,
    period: usize,
    no_volume: bool,
    first_val: usize,
    out_k1: &mut [f64],
    out_k2: &mut [f64],
) -> Result<(), AlphaTrendError> {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn mm512_abs_pd(x: __m512d) -> __m512d {
        let sign = _mm512_set1_pd(-0.0);
        _mm512_andnot_pd(sign, x)
    }

    let len = close.len();
    let warmup = first_val + period - 1;
    let p_f = period as f64;

    // --------- TR buffer ----------
    let mut tr_mu = make_uninit_matrix(1, len);
    let tr: &mut [f64] = core::slice::from_raw_parts_mut(tr_mu.as_mut_ptr() as *mut f64, len);

    if first_val < len {
        *tr.get_unchecked_mut(first_val) =
            *high.get_unchecked(first_val) - *low.get_unchecked(first_val);
    }

    let mut i = first_val + 1;
    while i + 8 <= len {
        let hv = _mm512_loadu_pd(high.as_ptr().add(i));
        let lv = _mm512_loadu_pd(low.as_ptr().add(i));
        let pc = _mm512_loadu_pd(close.as_ptr().add(i - 1));

        let hl = _mm512_sub_pd(hv, lv);
        let hc = mm512_abs_pd(_mm512_sub_pd(hv, pc));
        let lc = mm512_abs_pd(_mm512_sub_pd(lv, pc));

        let m1 = _mm512_max_pd(hl, hc);
        let m = _mm512_max_pd(m1, lc);
        _mm512_storeu_pd(tr.as_mut_ptr().add(i), m);
        i += 8;
    }
    while i < len {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let pc = *close.get_unchecked(i - 1);
        let hl = hi - lo;
        let hc = (hi - pc).abs();
        let lc = (lo - pc).abs();
        let m = if hl >= hc { hl } else { hc };
        *tr.get_unchecked_mut(i) = if m >= lc { m } else { lc };
        i += 1;
    }

    // --------- Momentum (RSI/MFI). Vectorize HLC3 if needed ----------
    let momentum_values: Vec<f64> = if no_volume {
        let rsi_params = RsiParams {
            period: Some(period),
        };
        let rsi_input = RsiInput::from_slice(close, rsi_params);
        rsi_with_kernel(&rsi_input, Kernel::Avx512)
            .map_err(|e| AlphaTrendError::RsiError { msg: e.to_string() })?
            .values
    } else {
        let mut hlc3_mu = make_uninit_matrix(1, len);
        let hlc3: &mut [f64] =
            core::slice::from_raw_parts_mut(hlc3_mu.as_mut_ptr() as *mut f64, len);

        let inv3 = _mm512_set1_pd(1.0 / 3.0);
        let mut j = 0usize;
        while j + 8 <= len {
            let hv = _mm512_loadu_pd(high.as_ptr().add(j));
            let lv = _mm512_loadu_pd(low.as_ptr().add(j));
            let cv = _mm512_loadu_pd(close.as_ptr().add(j));
            let s = _mm512_add_pd(_mm512_add_pd(hv, lv), cv);
            let h3 = _mm512_mul_pd(s, inv3);
            _mm512_storeu_pd(hlc3.as_mut_ptr().add(j), h3);
            j += 8;
        }
        while j < len {
            *hlc3.get_unchecked_mut(j) =
                (*high.get_unchecked(j) + *low.get_unchecked(j) + *close.get_unchecked(j))
                    * (1.0 / 3.0);
            j += 1;
        }

        let mfi_params = MfiParams {
            period: Some(period),
        };
        let mfi_input = MfiInput::from_slices(hlc3, volume, mfi_params);
        mfi_with_kernel(&mfi_input, Kernel::Avx512)
            .map_err(|e| AlphaTrendError::MfiError { msg: e.to_string() })?
            .values
    };

    // --------- O(n) ATR + AlphaTrend + K2 (scalar due to recurrence) ----------
    let mut sum = 0.0f64;
    {
        let mut j = first_val;
        while j <= warmup {
            sum += *tr.get_unchecked(j);
            j += 1;
        }
    }

    #[inline(always)]
    fn fast_max(a: f64, b: f64) -> f64 {
        if a >= b {
            a
        } else {
            b
        }
    }
    #[inline(always)]
    fn fast_min(a: f64, b: f64) -> f64 {
        if a <= b {
            a
        } else {
            b
        }
    }

    let mut prev2 = f64::NAN;
    let mut prev1 = f64::NAN;
    let mut prev_alpha = f64::NAN;

    let mut k = warmup;
    while k < len {
        let a = sum / p_f;
        let hi = *high.get_unchecked(k);
        let lo = *low.get_unchecked(k);
        let up = (-coeff).mul_add(a, lo);
        let dn = coeff.mul_add(a, hi);
        let m_ge_50 = *momentum_values.get_unchecked(k) >= 50.0;

        let alpha = if k == warmup {
            if m_ge_50 {
                up
            } else {
                dn
            }
        } else if m_ge_50 {
            fast_max(up, prev_alpha)
        } else {
            fast_min(dn, prev_alpha)
        };

        *out_k1.get_unchecked_mut(k) = alpha;
        if k >= warmup + 2 {
            *out_k2.get_unchecked_mut(k) = prev2;
        }

        prev2 = prev1;
        prev1 = alpha;
        prev_alpha = alpha;

        let nxt = k + 1;
        if nxt < len {
            sum += *tr.get_unchecked(nxt) - *tr.get_unchecked(nxt - period);
        }
        k += 1;
    }

    for v in &mut out_k1[..warmup.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut out_k2[..(warmup + 2).min(len)] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn alphatrend_simd128(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    coeff: f64,
    period: usize,
    no_volume: bool,
    first_val: usize,
    out_k1: &mut [f64],
    out_k2: &mut [f64],
) -> Result<(), AlphaTrendError> {
    use core::arch::wasm32::*;

    // WASM SIMD128 implementation stub - falls back to scalar for now
    // Future optimization: implement SIMD128-accelerated calculations
    alphatrend_scalar(
        open, high, low, close, volume, coeff, period, no_volume, first_val, out_k1, out_k2,
    )
}

/// Decision: O(1) streaming enabled. ATR as SMA(TR), RSI via Wilder RMA or MFI window sums.
/// Returns Some((k1,k2)) once ATR seeded and 2-bar lag available (> period + 1 bars).
#[derive(Debug, Clone)]
pub struct AlphaTrendStream {
    // --- Params ---
    coeff: f64,
    period: usize,
    inv_period: f64,
    no_volume: bool,

    // --- Rolling ATR (SMA of TR) ---
    tr_ring: Vec<f64>, // length = period
    tr_sum: f64,
    tr_idx: usize,
    tr_filled: usize,

    // --- RSI state (Wilder RMA) ---
    rsi_seeded: bool,
    rsi_init_gains: f64,
    rsi_init_losses: f64,
    rsi_count: usize,
    rsi_avg_gain: f64,
    rsi_avg_loss: f64,

    // --- MFI state (window sums) ---
    mfi_pos_ring: Vec<f64>, // length = period
    mfi_neg_ring: Vec<f64>, // length = period
    mfi_pos_sum: f64,
    mfi_neg_sum: f64,
    mfi_idx: usize,
    mfi_filled: usize,
    prev_tp: f64,

    // --- Previous prices ---
    prev_close: f64,
    have_prev: bool,

    // --- AlphaTrend sticky state & lag ---
    prev_alpha: f64,
    prev1: f64,
    prev2: f64,
    alpha_count: usize,
}

impl AlphaTrendStream {
    pub fn try_new(params: AlphaTrendParams) -> Result<Self, AlphaTrendError> {
        let coeff = params.coeff.unwrap_or(1.0);
        let period = params.period.unwrap_or(14);
        let no_volume = params.no_volume.unwrap_or(false);

        if period == 0 {
            return Err(AlphaTrendError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if coeff <= 0.0 || !coeff.is_finite() {
            return Err(AlphaTrendError::InvalidCoeff { coeff });
        }

        Ok(Self {
            coeff,
            period,
            inv_period: 1.0 / (period as f64),
            no_volume,

            tr_ring: vec![0.0; period],
            tr_sum: 0.0,
            tr_idx: 0,
            tr_filled: 0,

            rsi_seeded: false,
            rsi_init_gains: 0.0,
            rsi_init_losses: 0.0,
            rsi_count: 0,
            rsi_avg_gain: 0.0,
            rsi_avg_loss: 0.0,

            mfi_pos_ring: vec![0.0; period],
            mfi_neg_ring: vec![0.0; period],
            mfi_pos_sum: 0.0,
            mfi_neg_sum: 0.0,
            mfi_idx: 0,
            mfi_filled: 0,
            prev_tp: f64::NAN,

            prev_close: f64::NAN,
            have_prev: false,

            prev_alpha: f64::NAN,
            prev1: f64::NAN,
            prev2: f64::NAN,
            alpha_count: 0,
        })
    }

    /// O(1) update. Returns Some((k1, k2)) once both are defined
    /// (i.e., after alpha_count >= 3 ⇒ after period + 2 bars from start).
    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<(f64, f64)> {
        // Skip malformed bars without mutating state
        if !(high.is_finite() && low.is_finite() && close.is_finite() && volume.is_finite()) {
            return None;
        }
        if high < low {
            return None;
        }

        // ---------- True Range & ATR (SMA) ----------
        // TR = max(high-low, |high-prev_close|, |low-prev_close|)
        let tr = if self.have_prev {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            if hl >= hc {
                if hl >= lc {
                    hl
                } else {
                    lc
                }
            } else {
                if hc >= lc {
                    hc
                } else {
                    lc
                }
            }
        } else {
            high - low
        };

        if self.tr_filled < self.period {
            self.tr_ring[self.tr_idx] = tr;
            self.tr_sum += tr;
            self.tr_filled += 1;
            self.tr_idx = (self.tr_idx + 1) % self.period;
        } else {
            let old = self.tr_ring[self.tr_idx];
            self.tr_ring[self.tr_idx] = tr;
            self.tr_sum += tr - old;
            self.tr_idx = (self.tr_idx + 1) % self.period;
        }
        let atr_ready = self.tr_filled == self.period;
        let atr = if atr_ready {
            self.tr_sum * self.inv_period
        } else {
            f64::NAN
        };

        // ---------- Momentum regime: RSI or MFI (>=50 test without division) ----------
        let mut m_ge_50 = false;

        if self.no_volume {
            // RSI path (Wilder RMA)
            let (gain, loss) = if self.have_prev {
                let d = close - self.prev_close;
                if d >= 0.0 {
                    (d, 0.0)
                } else {
                    (0.0, -d)
                }
            } else {
                (0.0, 0.0)
            };

            if !self.rsi_seeded {
                self.rsi_init_gains += gain;
                self.rsi_init_losses += loss;
                self.rsi_count += 1;
                if self.rsi_count >= self.period {
                    self.rsi_avg_gain = self.rsi_init_gains * self.inv_period;
                    self.rsi_avg_loss = self.rsi_init_losses * self.inv_period;
                    self.rsi_seeded = true;
                }
            } else {
                let n1 = (self.period as f64) - 1.0;
                self.rsi_avg_gain = (self.rsi_avg_gain * n1 + gain) * self.inv_period;
                self.rsi_avg_loss = (self.rsi_avg_loss * n1 + loss) * self.inv_period;
            }

            if self.rsi_seeded {
                if self.rsi_avg_loss == 0.0 {
                    m_ge_50 = self.rsi_avg_gain >= 0.0;
                } else if self.rsi_avg_gain == 0.0 {
                    m_ge_50 = false;
                } else {
                    m_ge_50 = self.rsi_avg_gain >= self.rsi_avg_loss;
                }
            } else {
                m_ge_50 = false;
            }
        } else {
            // MFI path (window sums using Typical Price)
            let tp = (high + low + close) / 3.0;
            if self.have_prev {
                let mf = (tp * volume).max(0.0);
                let (pos, neg) = if tp > self.prev_tp {
                    (mf, 0.0)
                } else if tp < self.prev_tp {
                    (0.0, mf)
                } else {
                    (0.0, 0.0)
                };

                if self.mfi_filled < self.period {
                    self.mfi_pos_sum += pos;
                    self.mfi_neg_sum += neg;
                    self.mfi_pos_ring[self.mfi_idx] = pos;
                    self.mfi_neg_ring[self.mfi_idx] = neg;
                    self.mfi_idx = (self.mfi_idx + 1) % self.period;
                    self.mfi_filled += 1;
                } else {
                    let op = self.mfi_pos_ring[self.mfi_idx];
                    let on = self.mfi_neg_ring[self.mfi_idx];
                    self.mfi_pos_ring[self.mfi_idx] = pos;
                    self.mfi_neg_ring[self.mfi_idx] = neg;
                    self.mfi_pos_sum += pos - op;
                    self.mfi_neg_sum += neg - on;
                    self.mfi_idx = (self.mfi_idx + 1) % self.period;
                }
            }

            if self.mfi_filled == self.period {
                if self.mfi_neg_sum == 0.0 {
                    m_ge_50 = self.mfi_pos_sum >= 0.0;
                } else if self.mfi_pos_sum == 0.0 {
                    m_ge_50 = false;
                } else {
                    m_ge_50 = self.mfi_pos_sum >= self.mfi_neg_sum;
                }
            } else {
                m_ge_50 = false;
            }
            self.prev_tp = tp;
        }

        // ---------- AlphaTrend bands & sticky regime ----------
        let mut emitted = false;
        let mut cur = f64::NAN;

        if atr_ready {
            // up = low - coeff*ATR, down = high + coeff*ATR
            let up = (-self.coeff).mul_add(atr, low);
            let dn = self.coeff.mul_add(atr, high);

            cur = if self.alpha_count == 0 {
                if m_ge_50 {
                    up
                } else {
                    dn
                }
            } else if m_ge_50 {
                if up < self.prev_alpha {
                    self.prev_alpha
                } else {
                    up
                }
            } else {
                if dn > self.prev_alpha {
                    self.prev_alpha
                } else {
                    dn
                }
            };

            self.prev2 = self.prev1;
            self.prev1 = cur;
            self.prev_alpha = cur;
            self.alpha_count += 1;
            emitted = true;
        }

        // advance shared previous values
        self.prev_close = close;
        self.have_prev = true;

        if emitted && self.alpha_count >= 3 {
            Some((cur, self.prev2))
        } else {
            None
        }
    }

    /// Historical warmup for k1 only (consistent with batch path): period - 1
    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period - 1
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "alphatrend")]
#[pyo3(signature = (open, high, low, close, volume, coeff=1.0, period=14, no_volume=false, kernel=None))]
pub fn alphatrend_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    coeff: f64,
    period: usize,
    no_volume: bool,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let open_slice = open.as_slice()?;
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    let kern = validate_kernel(kernel, false)?;
    let params = AlphaTrendParams {
        coeff: Some(coeff),
        period: Some(period),
        no_volume: Some(no_volume),
    };
    let input = AlphaTrendInput::from_slices(
        open_slice,
        high_slice,
        low_slice,
        close_slice,
        volume_slice,
        params,
    );

    let result = py
        .allow_threads(|| alphatrend_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((result.k1.into_pyarray(py), result.k2.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "AlphaTrendStream")]
pub struct AlphaTrendStreamPy {
    stream: AlphaTrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AlphaTrendStreamPy {
    #[new]
    fn new(coeff: f64, period: usize, no_volume: bool) -> PyResult<Self> {
        let params = AlphaTrendParams {
            coeff: Some(coeff),
            period: Some(period),
            no_volume: Some(no_volume),
        };
        let stream =
            AlphaTrendStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AlphaTrendStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close, volume)
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlphaTrendJsOutput {
    pub values: Vec<f64>, // [k1..., k2...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alphatrend_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    coeff: f64,
    period: usize,
    no_volume: bool,
) -> Result<JsValue, JsValue> {
    let params = AlphaTrendParams {
        coeff: Some(coeff),
        period: Some(period),
        no_volume: Some(no_volume),
    };
    let input = AlphaTrendInput::from_slices(open, high, low, close, volume, params);

    let mut k1 = vec![0.0; close.len()];
    let mut k2 = vec![0.0; close.len()];

    alphatrend_into_slices(&mut k1, &mut k2, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Return structured output with metadata
    let mut values = Vec::with_capacity(k1.len() * 2);
    values.extend_from_slice(&k1);
    values.extend_from_slice(&k2);

    let out = AlphaTrendJsOutput {
        values,
        rows: 2,
        cols: close.len(),
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alphatrend_alloc_flat(n: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * n);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alphatrend_free_flat(ptr: *mut f64, n: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 2 * n, 2 * n);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alphatrend_into_flat(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_flat_ptr: *mut f64,
    len: usize,
    coeff: f64,
    period: usize,
    no_volume: bool,
) -> Result<(), JsValue> {
    if [open_ptr, high_ptr, low_ptr, close_ptr, volume_ptr]
        .iter()
        .any(|&p| p.is_null())
        || out_flat_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let (open, high, low, close, volume) = (
            core::slice::from_raw_parts(open_ptr, len),
            core::slice::from_raw_parts(high_ptr, len),
            core::slice::from_raw_parts(low_ptr, len),
            core::slice::from_raw_parts(close_ptr, len),
            core::slice::from_raw_parts(volume_ptr, len),
        );
        let (k1, k2) = (
            core::slice::from_raw_parts_mut(out_flat_ptr, len),
            core::slice::from_raw_parts_mut(out_flat_ptr.add(len), len),
        );
        let params = AlphaTrendParams {
            coeff: Some(coeff),
            period: Some(period),
            no_volume: Some(no_volume),
        };
        let input = AlphaTrendInput::from_slices(open, high, low, close, volume, params);
        alphatrend_into_slices(k1, k2, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(note = "Use alphatrend_alloc_flat/alphatrend_into_flat")]
pub fn alphatrend_alloc(_len: usize) -> *mut f64 {
    core::ptr::null_mut()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(note = "Use alphatrend_free_flat")]
pub fn alphatrend_free(_ptr: *mut f64, _len: usize) {}

// ==================== BATCH PROCESSING ====================

#[derive(Clone, Debug)]
pub struct AlphaTrendBatchRange {
    pub coeff: (f64, f64, f64),        // (start, end, step)
    pub period: (usize, usize, usize), // (start, end, step)
    pub no_volume: bool,
}

impl Default for AlphaTrendBatchRange {
    fn default() -> Self {
        Self {
            coeff: (1.0, 1.0, 0.0),
            period: (14, 240, 1),
            no_volume: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AlphaTrendBatchBuilder {
    range: AlphaTrendBatchRange,
    kernel: Kernel,
}

impl AlphaTrendBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn coeff_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.coeff = (start, end, step);
        self
    }

    #[inline]
    pub fn coeff_static(mut self, val: f64) -> Self {
        self.range.coeff = (val, val, 0.0);
        self
    }

    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline]
    pub fn period_static(mut self, val: usize) -> Self {
        self.range.period = (val, val, 0);
        self
    }

    #[inline]
    pub fn no_volume(mut self, val: bool) -> Self {
        self.range.no_volume = val;
        self
    }

    pub fn apply_candles(self, c: &Candles) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
        alphatrend_batch_with_kernel(c, &self.range, self.kernel)
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
        let len = close.len();
        if open.len() != len || high.len() != len || low.len() != len || volume.len() != len {
            return Err(AlphaTrendError::InconsistentDataLengths);
        }

        let candles = Candles {
            timestamp: vec![0; len],
            open: open.to_vec(),
            high: high.to_vec(),
            low: low.to_vec(),
            close: close.to_vec(),
            volume: volume.to_vec(),
            hl2: vec![],
            hlc3: vec![],
            ohlc4: vec![],
            hlcc4: vec![],
        };

        alphatrend_batch_with_kernel(&candles, &self.range, self.kernel)
    }

    pub fn with_default_candles(c: &Candles) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
        AlphaTrendBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }

    pub fn with_default_slices(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
        AlphaTrendBatchBuilder::new()
            .kernel(k)
            .apply_slices(open, high, low, close, volume)
    }
}

#[derive(Clone, Debug)]
pub struct AlphaTrendBatchOutput {
    pub values_k1: Vec<f64>,
    pub values_k2: Vec<f64>,
    pub combos: Vec<AlphaTrendParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AlphaTrendBatchOutput {
    pub fn row_for_params(&self, p: &AlphaTrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.coeff.unwrap_or(1.0) - p.coeff.unwrap_or(1.0)).abs() < 1e-12
                && c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && c.no_volume.unwrap_or(false) == p.no_volume.unwrap_or(false)
        })
    }

    pub fn values_for(&self, p: &AlphaTrendParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            let end = start + self.cols;
            (&self.values_k1[start..end], &self.values_k2[start..end])
        })
    }
}

#[inline(always)]
fn expand_grid_alphatrend(r: &AlphaTrendBatchRange) -> Vec<AlphaTrendParams> {
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

    let coeffs = axis_f64(r.coeff);
    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(coeffs.len() * periods.len());
    for &c in &coeffs {
        for &p in &periods {
            out.push(AlphaTrendParams {
                coeff: Some(c),
                period: Some(p),
                no_volume: Some(r.no_volume),
            });
        }
    }
    out
}

pub fn alphatrend_batch_with_kernel(
    candles: &Candles,
    sweep: &AlphaTrendBatchRange,
    k: Kernel,
) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(AlphaTrendError::InvalidPeriod {
                period: 0,
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

    alphatrend_batch_inner(candles, sweep, simd, true)
}

#[inline(always)]
pub fn alphatrend_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AlphaTrendBatchRange,
    kern: Kernel,
) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
    alphatrend_batch_inner_from_slices(open, high, low, close, volume, sweep, kern, false)
}

#[inline(always)]
pub fn alphatrend_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AlphaTrendBatchRange,
    kern: Kernel,
) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
    alphatrend_batch_inner_from_slices(open, high, low, close, volume, sweep, kern, true)
}

#[inline(always)]
fn alphatrend_batch_inner_from_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AlphaTrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
    let combos = expand_grid_alphatrend(sweep);
    let cols = close.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(AlphaTrendError::EmptyInputData);
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlphaTrendError::AllValuesNaN)?;
    let warm_k1: Vec<usize> = combos
        .iter()
        .map(|p| first + p.period.unwrap_or(14) - 1)
        .collect();
    let warm_k2: Vec<usize> = warm_k1.iter().map(|&w| w.saturating_add(2)).collect();

    let mut k1_mu = make_uninit_matrix(rows, cols);
    let mut k2_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut k1_mu, cols, &warm_k1);
    init_matrix_prefixes(&mut k2_mu, cols, &warm_k2);

    let mut k1_guard = core::mem::ManuallyDrop::new(k1_mu);
    let mut k2_guard = core::mem::ManuallyDrop::new(k2_mu);
    let out_k1: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(k1_guard.as_mut_ptr() as *mut f64, k1_guard.len())
    };
    let out_k2: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(k2_guard.as_mut_ptr() as *mut f64, k2_guard.len())
    };

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    alphatrend_batch_inner_into_slices(
        open, high, low, close, volume, sweep, actual, parallel, out_k1, out_k2,
    )?;

    let values_k1 = unsafe {
        Vec::from_raw_parts(
            k1_guard.as_mut_ptr() as *mut f64,
            k1_guard.len(),
            k1_guard.capacity(),
        )
    };
    let values_k2 = unsafe {
        Vec::from_raw_parts(
            k2_guard.as_mut_ptr() as *mut f64,
            k2_guard.len(),
            k2_guard.capacity(),
        )
    };
    core::mem::forget(k1_guard);
    core::mem::forget(k2_guard);

    Ok(AlphaTrendBatchOutput {
        values_k1,
        values_k2,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn alphatrend_batch_inner(
    candles: &Candles,
    sweep: &AlphaTrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AlphaTrendBatchOutput, AlphaTrendError> {
    let combos = expand_grid_alphatrend(sweep);
    let cols = candles.close.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(AlphaTrendError::EmptyInputData);
    }

    // Uninit flat matrices
    let mut k1_mu = make_uninit_matrix(rows, cols);
    let mut k2_mu = make_uninit_matrix(rows, cols);

    // Warm prefixes per row
    let first = candles
        .close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlphaTrendError::AllValuesNaN)?;
    let warm_k1: Vec<usize> = combos
        .iter()
        .map(|p| first + p.period.unwrap_or(14) - 1)
        .collect();
    let warm_k2: Vec<usize> = warm_k1.iter().map(|&w| w.saturating_add(2)).collect();

    init_matrix_prefixes(&mut k1_mu, cols, &warm_k1);
    init_matrix_prefixes(&mut k2_mu, cols, &warm_k2);

    // Get &mut [MaybeUninit<f64>] -> &mut [f64]
    let mut k1_guard = core::mem::ManuallyDrop::new(k1_mu);
    let mut k2_guard = core::mem::ManuallyDrop::new(k2_mu);
    let out_k1: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(k1_guard.as_mut_ptr() as *mut f64, k1_guard.len())
    };
    let out_k2: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(k2_guard.as_mut_ptr() as *mut f64, k2_guard.len())
    };

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row =
        |row: usize, k1_row: &mut [f64], k2_row: &mut [f64]| -> Result<(), AlphaTrendError> {
            let p = &combos[row];
            let input = AlphaTrendInput::from_candles(candles, p.clone());
            alphatrend_into_slices(k1_row, k2_row, &input, actual)
        };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        use rayon::prelude::*;

        out_k1
            .par_chunks_mut(cols)
            .zip(out_k2.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(|(row, (k1r, k2r))| do_row(row, k1r, k2r))?;
    } else {
        for (row, (k1r, k2r)) in out_k1
            .chunks_mut(cols)
            .zip(out_k2.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, k1r, k2r)?;
        }
    }

    #[cfg(target_arch = "wasm32")]
    for (row, (k1r, k2r)) in out_k1
        .chunks_mut(cols)
        .zip(out_k2.chunks_mut(cols))
        .enumerate()
    {
        do_row(row, k1r, k2r)?;
    }

    // Reclaim Vec<f64> without copy
    let values_k1 = unsafe {
        Vec::from_raw_parts(
            k1_guard.as_mut_ptr() as *mut f64,
            k1_guard.len(),
            k1_guard.capacity(),
        )
    };
    let values_k2 = unsafe {
        Vec::from_raw_parts(
            k2_guard.as_mut_ptr() as *mut f64,
            k2_guard.len(),
            k2_guard.capacity(),
        )
    };

    Ok(AlphaTrendBatchOutput {
        values_k1,
        values_k2,
        combos,
        rows,
        cols,
    })
}

// New slice-based batch function for direct writes
#[inline(always)]
pub fn alphatrend_batch_inner_into_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AlphaTrendBatchRange,
    kern: Kernel,
    parallel: bool,
    k1_slice: &mut [f64],
    k2_slice: &mut [f64],
) -> Result<(), AlphaTrendError> {
    let combos = expand_grid_alphatrend(sweep);
    let cols = close.len();
    let rows = combos.len();

    if cols == 0 {
        return Err(AlphaTrendError::EmptyInputData);
    }

    if k1_slice.len() != rows * cols || k2_slice.len() != rows * cols {
        return Err(AlphaTrendError::InconsistentDataLengths);
    }

    // Resolve kernel and SIMD mapping for momentum precompute
    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd_kernel = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => detect_best_kernel(),
    };

    // Find first valid index
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlphaTrendError::AllValuesNaN)?;

    // Precompute TR once for the whole dataset
    let mut tr_mu = make_uninit_matrix(1, cols);
    let tr: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(tr_mu.as_mut_ptr() as *mut f64, cols) };
    if first < cols {
        tr[first] = high[first] - low[first];
    }
    for i in (first + 1)..cols {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Optionally precompute HLC3 once if using MFI across rows
    let use_rsi = sweep.no_volume;
    let hlc3_opt: Option<Vec<f64>> = if use_rsi {
        None
    } else {
        let mut hlc3_mu = make_uninit_matrix(1, cols);
        let hlc3: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(hlc3_mu.as_mut_ptr() as *mut f64, cols) };
        for i in 0..cols {
            hlc3[i] = (high[i] + low[i] + close[i]) / 3.0;
        }
        // Reclaim as Vec without copy
        let v = unsafe {
            Vec::from_raw_parts(
                hlc3_mu.as_mut_ptr() as *mut f64,
                hlc3_mu.len(),
                hlc3_mu.capacity(),
            )
        };
        core::mem::forget(hlc3_mu);
        Some(v)
    };

    // Deduplicate periods across combos and precompute momentum per unique period
    use std::collections::HashMap;
    let mut unique_periods: Vec<usize> = combos.iter().map(|p| p.period.unwrap_or(14)).collect();
    unique_periods.sort_unstable();
    unique_periods.dedup();

    let mut momentum_map: HashMap<usize, Vec<f64>> = HashMap::with_capacity(unique_periods.len());
    for &p in &unique_periods {
        if p == 0 || p > cols {
            // Keep behavior consistent with single-row path
            return Err(AlphaTrendError::InvalidPeriod {
                period: p,
                data_len: cols,
            });
        }
        if use_rsi {
            let rsi_params = RsiParams { period: Some(p) };
            let rsi_input = RsiInput::from_slice(close, rsi_params);
            let mv = rsi_with_kernel(&rsi_input, simd_kernel)
                .map_err(|e| AlphaTrendError::RsiError { msg: e.to_string() })?
                .values;
            momentum_map.insert(p, mv);
        } else {
            let hlc3 = hlc3_opt.as_ref().expect("hlc3 precomputed");
            let mfi_params = MfiParams { period: Some(p) };
            let mfi_input = MfiInput::from_slices(hlc3, volume, mfi_params);
            let mv = mfi_with_kernel(&mfi_input, simd_kernel)
                .map_err(|e| AlphaTrendError::MfiError { msg: e.to_string() })?
                .values;
            momentum_map.insert(p, mv);
        }
    }

    // Row compute: streaming ATR + AlphaTrend using shared TR and per-period momentum
    let do_row =
        |row: usize, k1_row: &mut [f64], k2_row: &mut [f64]| -> Result<(), AlphaTrendError> {
            let params = &combos[row];
            let coeff = params.coeff.unwrap_or(1.0);
            if !coeff.is_finite() || coeff <= 0.0 {
                return Err(AlphaTrendError::InvalidCoeff { coeff });
            }
            let period = params.period.unwrap_or(14);
            if period == 0 || period > cols {
                return Err(AlphaTrendError::InvalidPeriod {
                    period,
                    data_len: cols,
                });
            }
            let warmup = first + period - 1;
            if warmup >= cols {
                // nothing to write; prefixes remain NaN as allocated by callers
                return Ok(());
            }

            let mom = momentum_map.get(&period).expect("momentum precomputed");

            // initialize rolling sum for ATR SMA window
            let mut sum = 0.0f64;
            for j in first..=warmup {
                sum += tr[j];
            }

            let mut prev_alpha = f64::NAN;
            let mut prev1 = f64::NAN;
            let mut prev2 = f64::NAN;

            for i in warmup..cols {
                let a = sum / period as f64;
                let up = low[i] - a * coeff;
                let dn = high[i] + a * coeff;
                let m_ge_50 = mom[i] >= 50.0;

                let cur = if i == warmup {
                    if m_ge_50 {
                        up
                    } else {
                        dn
                    }
                } else if m_ge_50 {
                    if up < prev_alpha {
                        prev_alpha
                    } else {
                        up
                    }
                } else {
                    if dn > prev_alpha {
                        prev_alpha
                    } else {
                        dn
                    }
                };

                k1_row[i] = cur;
                if i >= warmup + 2 {
                    k2_row[i] = prev2;
                }

                prev2 = prev1;
                prev1 = cur;
                prev_alpha = cur;

                // slide ATR window if next exists
                if i + 1 < cols {
                    sum += tr[i + 1] - tr[i + 1 - period];
                }
            }
            Ok(())
        };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        use rayon::prelude::*;
        k1_slice
            .par_chunks_mut(cols)
            .zip(k2_slice.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(|(row, (k1r, k2r))| do_row(row, k1r, k2r))?;
    } else {
        for (row, (k1r, k2r)) in k1_slice
            .chunks_mut(cols)
            .zip(k2_slice.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, k1r, k2r)?;
        }
    }

    #[cfg(target_arch = "wasm32")]
    for (row, (k1r, k2r)) in k1_slice
        .chunks_mut(cols)
        .zip(k2_slice.chunks_mut(cols))
        .enumerate()
    {
        do_row(row, k1r, k2r)?;
    }

    Ok(())
}

// ==================== ENHANCED PYTHON BINDINGS ====================

#[cfg(feature = "python")]
#[pyfunction(name = "alphatrend_batch")]
#[pyo3(signature = (open, high, low, close, volume, coeff_range, period_range, no_volume=false, kernel=None))]
pub fn alphatrend_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    coeff_range: (f64, f64, f64),
    period_range: (usize, usize, usize),
    no_volume: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArray1;

    let (o, h, l, c, v) = (
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        volume.as_slice()?,
    );
    let len = c.len();
    if o.len() != len || h.len() != len || l.len() != len || v.len() != len {
        return Err(PyValueError::new_err("Inconsistent data lengths"));
    }

    let sweep = AlphaTrendBatchRange {
        coeff: coeff_range,
        period: period_range,
        no_volume,
    };
    let kern = validate_kernel(kernel, true)?;

    // Preallocate two flat output arrays, then fill in place
    let rows = {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> usize {
            if st == 0 || s == e {
                1
            } else {
                (e - s) / st + 1
            }
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> usize {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                1
            } else {
                ((e - s) / st).floor() as usize + 1
            }
        }
        axis_f64(coeff_range) * axis_usize(period_range)
    };

    let out_k1 = unsafe { PyArray1::<f64>::new(py, [rows * len], false) };
    let out_k2 = unsafe { PyArray1::<f64>::new(py, [rows * len], false) };
    let k1_slice = unsafe { out_k1.as_slice_mut()? };
    let k2_slice = unsafe { out_k2.as_slice_mut()? };

    // Lightweight wrapper that writes directly
    py.allow_threads(|| {
        alphatrend_batch_inner_into_slices(o, h, l, c, v, &sweep, kern, true, k1_slice, k2_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("k1", out_k1.reshape([rows, len])?)?;
    dict.set_item("k2", out_k2.reshape([rows, len])?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", len)?;

    // Add combo parameters
    let combos = expand_grid_alphatrend(&sweep);
    let combo_list = PyList::new(
        py,
        combos.iter().map(|c| {
            let d = PyDict::new(py);
            d.set_item("coeff", c.coeff.unwrap_or(1.0)).unwrap();
            d.set_item("period", c.period.unwrap_or(14)).unwrap();
            d.set_item("no_volume", c.no_volume.unwrap_or(false))
                .unwrap();
            d
        }),
    )?;
    dict.set_item("combos", combo_list)?;

    Ok(dict.into())
}

// ==================== CUDA PYTHON BINDINGS ====================

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "alphatrend_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, volume_f32, coeff_range, period_range, no_volume=false, device_id=0))]
pub fn alphatrend_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: PyReadonlyArray1<'py, f32>,
    low_f32: PyReadonlyArray1<'py, f32>,
    close_f32: PyReadonlyArray1<'py, f32>,
    volume_f32: PyReadonlyArray1<'py, f32>,
    coeff_range: (f64, f64, f64),
    period_range: (usize, usize, usize),
    no_volume: bool,
    device_id: usize,
)-> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    use crate::cuda::cuda_available;
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let (h, l, c, v) = (
        high_f32.as_slice()?,
        low_f32.as_slice()?,
        close_f32.as_slice()?,
        volume_f32.as_slice()?,
    );
    if h.len() != l.len() || h.len() != c.len() || h.len() != v.len() {
        return Err(PyValueError::new_err("Inconsistent data lengths"));
    }
    let sweep = AlphaTrendBatchRange { coeff: coeff_range, period: period_range, no_volume };
    let (batch, coeffs_vec, periods_vec) = py.allow_threads(|| {
        let cuda = CudaAlphaTrend::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda.alphatrend_batch_dev(h, l, c, v, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let coeffs: Vec<f64> = out.combos.iter().map(|p| p.coeff.unwrap_or(1.0)).collect();
        let periods: Vec<u64> = out.combos.iter().map(|p| p.period.unwrap_or(14) as u64).collect();
        Ok::<_, PyErr>((out, coeffs, periods))
    })?;

    let dict = PyDict::new(py);
    dict.set_item("k1", Py::new(py, DeviceArrayF32Py { inner: batch.k1 })?)?;
    dict.set_item("k2", Py::new(py, DeviceArrayF32Py { inner: batch.k2 })?)?;
    dict.set_item("coeffs", coeffs_vec.into_pyarray(py))?;
    dict.set_item("periods", periods_vec.into_pyarray(py))?;
    dict.set_item("rows", batch.k1.rows)?;
    dict.set_item("cols", batch.k1.cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "alphatrend_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, volume_tm_f32, cols, rows, coeff=1.0, period=14, no_volume=false, device_id=0))]
pub fn alphatrend_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: PyReadonlyArray1<'py, f32>,
    low_tm_f32: PyReadonlyArray1<'py, f32>,
    close_tm_f32: PyReadonlyArray1<'py, f32>,
    volume_tm_f32: PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    coeff: f64,
    period: usize,
    no_volume: bool,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let (h, l, c, v) = (
        high_tm_f32.as_slice()?,
        low_tm_f32.as_slice()?,
        close_tm_f32.as_slice()?,
        volume_tm_f32.as_slice()?,
    );
    if h.len() != cols * rows || l.len() != cols * rows || c.len() != cols * rows || v.len() != cols * rows {
        return Err(PyValueError::new_err("Inconsistent time-major shapes"));
    }
    let (k1, k2) = py.allow_threads(|| {
        let cuda = CudaAlphaTrend::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.alphatrend_many_series_one_param_time_major_dev(h, l, c, v, cols, rows, coeff, period, no_volume)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((DeviceArrayF32Py { inner: k1 }, DeviceArrayF32Py { inner: k2 }))
}

// ==================== ENHANCED WASM BINDINGS ====================

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlphaTrendBatchJsOutput {
    pub values: Vec<f64>, // flattened, rows = combos.len() * 2
    pub combos: Vec<AlphaTrendParams>,
    pub rows: usize, // 2 * number_of_combos
    pub cols: usize, // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = alphatrend_batch)]
pub fn alphatrend_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    coeff_start: f64,
    coeff_end: f64,
    coeff_step: f64,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    no_volume: bool,
) -> Result<JsValue, JsValue> {
    let sweep = AlphaTrendBatchRange {
        coeff: (coeff_start, coeff_end, coeff_step),
        period: (period_start, period_end, period_step),
        no_volume,
    };
    let combos = expand_grid_alphatrend(&sweep);
    let rows = combos.len();
    let cols = close.len();

    // Flat buffers, filled in place
    let mut k1 = vec![f64::NAN; rows * cols];
    let mut k2 = vec![f64::NAN; rows * cols];

    alphatrend_batch_inner_into_slices(
        open,
        high,
        low,
        close,
        volume,
        &sweep,
        detect_best_batch_kernel(),
        true,
        &mut k1,
        &mut k2,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Stack k1 then k2 per combo
    let mut values = Vec::with_capacity(rows * 2 * cols);
    for r in 0..rows {
        let base = r * cols;
        values.extend_from_slice(&k1[base..base + cols]);
        values.extend_from_slice(&k2[base..base + cols]);
    }

    let js = AlphaTrendBatchJsOutput {
        values,
        combos,
        rows: rows * 2,
        cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alphatrend_batch_into_flat(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    coeff_start: f64,
    coeff_end: f64,
    coeff_step: f64,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    no_volume: bool,
) -> Result<usize, JsValue> {
    if [open_ptr, high_ptr, low_ptr, close_ptr, volume_ptr]
        .iter()
        .any(|&p| p.is_null())
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let (open, high, low, close, volume) = (
            core::slice::from_raw_parts(open_ptr, len),
            core::slice::from_raw_parts(high_ptr, len),
            core::slice::from_raw_parts(low_ptr, len),
            core::slice::from_raw_parts(close_ptr, len),
            core::slice::from_raw_parts(volume_ptr, len),
        );
        let sweep = AlphaTrendBatchRange {
            coeff: (coeff_start, coeff_end, coeff_step),
            period: (period_start, period_end, period_step),
            no_volume,
        };
        let combos = expand_grid_alphatrend(&sweep);
        let rows = combos.len();
        let cols = len;

        // Interpret out_ptr as two stacked blocks per combo
        let k1 = core::slice::from_raw_parts_mut(out_ptr, rows * cols);
        let k2 = core::slice::from_raw_parts_mut(out_ptr.add(rows * cols), rows * cols);

        alphatrend_batch_inner_into_slices(
            open,
            high,
            low,
            close,
            volume,
            &sweep,
            detect_best_batch_kernel(),
            false,
            k1,
            k2,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows) // caller knows rows*2 and cols
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlphaTrendBatchConfig {
    pub coeff_range: (f64, f64, f64),
    pub period_range: (usize, usize, usize),
    pub no_volume: bool,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = alphatrend_batch_unified)]
pub fn alphatrend_batch_unified_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AlphaTrendBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AlphaTrendBatchRange {
        coeff: config.coeff_range,
        period: config.period_range,
        no_volume: config.no_volume,
    };

    let output =
        alphatrend_batch_slice(open, high, low, close, volume, &sweep, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Flatten output similar to alphatrend_batch_js
    let rows2 = output.rows * 2;
    let cols = output.cols;
    let mut values = Vec::with_capacity(rows2 * cols);
    for r in 0..output.rows {
        let base = r * cols;
        values.extend_from_slice(&output.values_k1[base..base + cols]);
        values.extend_from_slice(&output.values_k2[base..base + cols]);
    }

    let js_output = AlphaTrendBatchJsOutput {
        values,
        combos: output.combos,
        rows: rows2,
        cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alphatrend_into(
    in_ptr: *const f64,
    out_k1_ptr: *mut f64,
    out_k2_ptr: *mut f64,
    len: usize,
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    volume_ptr: *const f64,
    coeff: f64,
    period: usize,
    no_volume: bool,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || out_k1_ptr.is_null()
        || out_k2_ptr.is_null()
        || open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || volume_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer passed to alphatrend_into"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(in_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let params = AlphaTrendParams {
            coeff: Some(coeff),
            period: Some(period),
            no_volume: Some(no_volume),
        };
        let input = AlphaTrendInput::from_slices(open, high, low, close, volume, params);

        let out_k1 = std::slice::from_raw_parts_mut(out_k1_ptr, len);
        let out_k2 = std::slice::from_raw_parts_mut(out_k2_ptr, len);

        alphatrend_into_slices(out_k1, out_k2, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct AlphaTrendContext {
    coeff: f64,
    period: usize,
    no_volume: bool,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl AlphaTrendContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(coeff: f64, period: usize, no_volume: bool) -> Result<AlphaTrendContext, JsValue> {
        if period == 0 {
            return Err(JsValue::from_str("Invalid period: 0"));
        }
        if coeff <= 0.0 || !coeff.is_finite() {
            return Err(JsValue::from_str(&format!(
                "Invalid coefficient: {}",
                coeff
            )));
        }

        Ok(AlphaTrendContext {
            coeff,
            period,
            no_volume,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        open_ptr: *const f64,
        high_ptr: *const f64,
        low_ptr: *const f64,
        close_ptr: *const f64,
        volume_ptr: *const f64,
        out_k1_ptr: *mut f64,
        out_k2_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if len < self.period {
            return Err(JsValue::from_str("Data length less than period"));
        }

        unsafe {
            let open = std::slice::from_raw_parts(open_ptr, len);
            let high = std::slice::from_raw_parts(high_ptr, len);
            let low = std::slice::from_raw_parts(low_ptr, len);
            let close = std::slice::from_raw_parts(close_ptr, len);
            let volume = std::slice::from_raw_parts(volume_ptr, len);
            let out_k1 = std::slice::from_raw_parts_mut(out_k1_ptr, len);
            let out_k2 = std::slice::from_raw_parts_mut(out_k2_ptr, len);

            let params = AlphaTrendParams {
                coeff: Some(self.coeff),
                period: Some(self.period),
                no_volume: Some(self.no_volume),
            };
            let input = AlphaTrendInput::from_slices(open, high, low, close, volume, params);

            // Handle in-place updates
            if close_ptr == out_k1_ptr || close_ptr == out_k2_ptr {
                let mut temp_k1 = vec![0.0; len];
                let mut temp_k2 = vec![0.0; len];

                alphatrend_into_slices(&mut temp_k1, &mut temp_k2, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;

                out_k1.copy_from_slice(&temp_k1);
                out_k2.copy_from_slice(&temp_k2);
            } else {
                alphatrend_into_slices(out_k1, out_k2, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.period - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn check_alphatrend_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlphaTrendInput::from_candles(&candles, AlphaTrendParams::default());
        let result = alphatrend_with_kernel(&input, kernel)?;

        // K1 Reference values from PineScript
        let expected_k1 = [
            60243.00,
            60243.00,
            60138.92857143,
            60088.42857143,
            59937.21428571,
        ];

        // K2 Reference values from PineScript
        let expected_k2 = [
            60542.42857143,
            60454.14285714,
            60243.00,
            60243.00,
            60138.92857143,
        ];

        let start = result.k1.len().saturating_sub(5);

        // Check K1 values
        for (i, &val) in result.k1[start..].iter().enumerate() {
            let diff = (val - expected_k1[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] AlphaTrend K1 {:?} mismatch at idx {}: got {}, expected {} (diff: {})",
                test_name,
                kernel,
                i,
                val,
                expected_k1[i],
                diff
            );
        }

        // Check K2 values
        for (i, &val) in result.k2[start..].iter().enumerate() {
            let diff = (val - expected_k2[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] AlphaTrend K2 {:?} mismatch at idx {}: got {}, expected {} (diff: {})",
                test_name,
                kernel,
                i,
                val,
                expected_k2[i],
                diff
            );
        }

        Ok(())
    }

    fn check_alphatrend_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = AlphaTrendParams {
            coeff: None,
            period: None,
            no_volume: None,
        };
        let input = AlphaTrendInput::from_candles(&candles, default_params);
        let output = alphatrend_with_kernel(&input, kernel)?;
        assert_eq!(output.k1.len(), candles.close.len());
        assert_eq!(output.k2.len(), candles.close.len());

        Ok(())
    }

    fn check_alphatrend_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlphaTrendInput::with_default_candles(&candles);
        let output = alphatrend_with_kernel(&input, kernel)?;
        assert_eq!(output.k1.len(), candles.close.len());
        assert_eq!(output.k2.len(), candles.close.len());

        Ok(())
    }

    fn check_alphatrend_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let open = vec![10.0, 20.0, 30.0];
        let high = vec![12.0, 22.0, 32.0];
        let low = vec![8.0, 18.0, 28.0];
        let close = vec![11.0, 21.0, 31.0];
        let volume = vec![100.0, 200.0, 300.0];

        let params = AlphaTrendParams {
            coeff: Some(1.0),
            period: Some(0),
            no_volume: Some(false),
        };
        let input = AlphaTrendInput::from_slices(&open, &high, &low, &close, &volume, params);
        let res = alphatrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AlphaTrend should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_alphatrend_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let empty: [f64; 0] = [];
        let params = AlphaTrendParams::default();
        let input = AlphaTrendInput::from_slices(&empty, &empty, &empty, &empty, &empty, params);
        let res = alphatrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AlphaTrend should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_alphatrend_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = AlphaTrendParams::default();
        let input = AlphaTrendInput::from_slices(
            &nan_data, &nan_data, &nan_data, &nan_data, &nan_data, params,
        );
        let res = alphatrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AlphaTrend should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_alphatrend_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let data_small = [10.0, 20.0, 30.0];
        let params = AlphaTrendParams {
            coeff: Some(1.0),
            period: Some(10),
            no_volume: Some(false),
        };
        let input = AlphaTrendInput::from_slices(
            &data_small,
            &data_small,
            &data_small,
            &data_small,
            &data_small,
            params,
        );
        let res = alphatrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AlphaTrend should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_alphatrend_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let single_point = [42.0];
        let params = AlphaTrendParams {
            coeff: Some(1.0),
            period: Some(14),
            no_volume: Some(false),
        };
        let input = AlphaTrendInput::from_slices(
            &single_point,
            &single_point,
            &single_point,
            &single_point,
            &single_point,
            params,
        );
        let res = alphatrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AlphaTrend should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_alphatrend_invalid_coeff(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let data = vec![1.0; 20]; // Create a vec with 20 elements of value 1.0
        let params = AlphaTrendParams {
            coeff: Some(-1.0),
            period: Some(14),
            no_volume: Some(false),
        };
        let input = AlphaTrendInput::from_slices(&data, &data, &data, &data, &data, params);
        let res = alphatrend_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlphaTrendError::InvalidCoeff { .. })),
            "[{}] AlphaTrend should fail with invalid coefficient",
            test_name
        );
        Ok(())
    }

    fn check_alphatrend_inconsistent_lengths(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let open = vec![10.0, 20.0, 30.0];
        let high = vec![12.0, 22.0]; // Different length
        let low = vec![8.0, 18.0, 28.0];
        let close = vec![11.0, 21.0, 31.0];
        let volume = vec![100.0, 200.0, 300.0];

        let params = AlphaTrendParams::default();
        let input = AlphaTrendInput::from_slices(&open, &high, &low, &close, &volume, params);
        let res = alphatrend_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlphaTrendError::InconsistentDataLengths)),
            "[{}] AlphaTrend should fail with inconsistent data lengths",
            test_name
        );
        Ok(())
    }

    fn check_alphatrend_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = AlphaTrendParams {
            coeff: Some(1.0),
            period: Some(14),
            no_volume: Some(false),
        };
        let first_input = AlphaTrendInput::from_candles(&candles, first_params);
        let first_result = alphatrend_with_kernel(&first_input, kernel)?;

        // Use K1 output as close price for second run
        let second_params = AlphaTrendParams {
            coeff: Some(1.0),
            period: Some(14),
            no_volume: Some(true), // Use RSI since we only have one series
        };
        // Create synthetic OHLCV from K1 values
        let k1 = &first_result.k1;
        let synthetic_high: Vec<f64> = k1
            .iter()
            .map(|&v| if v.is_nan() { v } else { v + 10.0 })
            .collect();
        let synthetic_low: Vec<f64> = k1
            .iter()
            .map(|&v| if v.is_nan() { v } else { v - 10.0 })
            .collect();
        let synthetic_volume = vec![1000.0; k1.len()];

        let second_input = AlphaTrendInput::from_slices(
            k1,
            &synthetic_high,
            &synthetic_low,
            k1,
            &synthetic_volume,
            second_params,
        );
        let second_result = alphatrend_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.k1.len(), first_result.k1.len());
        assert_eq!(second_result.k2.len(), first_result.k2.len());

        Ok(())
    }

    fn check_alphatrend_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlphaTrendInput::from_candles(
            &candles,
            AlphaTrendParams {
                coeff: Some(1.0),
                period: Some(14),
                no_volume: Some(false),
            },
        );
        let res = alphatrend_with_kernel(&input, kernel)?;
        assert_eq!(res.k1.len(), candles.close.len());
        assert_eq!(res.k2.len(), candles.close.len());

        // Check that values after warmup are not NaN
        if res.k1.len() > 240 {
            for (i, &val) in res.k1[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN in K1 at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_alphatrend_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        // Validate O(1) streaming behavior: returns Some((k1,k2)) after warmup+2
        let params = AlphaTrendParams {
            coeff: Some(1.0),
            period: Some(14),
            no_volume: Some(false),
        };

        let mut stream = AlphaTrendStream::try_new(params)?;
        let warmup = stream.get_warmup_period();

        // Feed some data points
        for i in 0..30 {
            let high = 100.0 + i as f64 + 2.0;
            let low = 100.0 + i as f64 - 2.0;
            let close = 100.0 + i as f64;
            let volume = 1000.0 + i as f64 * 10.0;

            let result = stream.update(high, low, close, volume);
            if i + 1 >= warmup + 3 {
                // Expect values once k2 lag is available
                let some = result.expect("streaming should emit after warmup+2");
                assert!(
                    some.0.is_finite() && some.1.is_finite(),
                    "[{}] Non-finite streaming outputs at i={}",
                    test_name,
                    i
                );
            } else {
                assert!(
                    result.is_none(),
                    "[{}] Should not emit before warmup+2 at i={}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_alphatrend_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            AlphaTrendParams::default(),
            AlphaTrendParams {
                coeff: Some(0.5),
                period: Some(7),
                no_volume: Some(false),
            },
            AlphaTrendParams {
                coeff: Some(2.0),
                period: Some(21),
                no_volume: Some(true),
            },
            AlphaTrendParams {
                coeff: Some(1.5),
                period: Some(10),
                no_volume: Some(false),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = AlphaTrendInput::from_candles(&candles, params.clone());
            let output = alphatrend_with_kernel(&input, kernel)?;

            for (i, &val) in output.k1.iter().chain(output.k2.iter()).enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: coeff={}, period={}, no_volume={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.coeff.unwrap_or(1.0),
                        params.period.unwrap_or(14),
                        params.no_volume.unwrap_or(false)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: coeff={}, period={}, no_volume={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.coeff.unwrap_or(1.0),
                        params.period.unwrap_or(14),
                        params.no_volume.unwrap_or(false)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: coeff={}, period={}, no_volume={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.coeff.unwrap_or(1.0),
                        params.period.unwrap_or(14),
                        params.no_volume.unwrap_or(false)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_alphatrend_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_alphatrend_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;

        let strat = (1usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
                0.1f64..5.0f64, // coeff
                any::<bool>(),  // no_volume
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(close_data, period, coeff, no_volume)| {
                // Generate synthetic OHLCV data from close prices
                let high: Vec<f64> = close_data.iter().map(|&c| c + 5.0).collect();
                let low: Vec<f64> = close_data.iter().map(|&c| c - 5.0).collect();
                let open = close_data.clone();
                let volume = vec![1000.0; close_data.len()];

                let params = AlphaTrendParams {
                    coeff: Some(coeff),
                    period: Some(period),
                    no_volume: Some(no_volume),
                };
                let input =
                    AlphaTrendInput::from_slices(&open, &high, &low, &close_data, &volume, params);

                let result = alphatrend_with_kernel(&input, kernel).unwrap();
                let ref_result = alphatrend_with_kernel(&input, Kernel::Scalar).unwrap();

                // Verify K1 values
                for i in 0..close_data.len() {
                    let y = result.k1[i];
                    let r = ref_result.k1[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "K1 finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "K1 mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }

                // Verify K2 values
                for i in 0..close_data.len() {
                    let y = result.k2[i];
                    let r = ref_result.k2[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "K2 finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "K2 mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    // Macro to generate tests for all kernel variants
    macro_rules! generate_all_alphatrend_tests {
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

    // Generate all kernel-specific tests
    generate_all_alphatrend_tests!(
        check_alphatrend_accuracy,
        check_alphatrend_partial_params,
        check_alphatrend_default_candles,
        check_alphatrend_zero_period,
        check_alphatrend_empty_input,
        check_alphatrend_all_nan,
        check_alphatrend_period_exceeds_length,
        check_alphatrend_very_small_dataset,
        check_alphatrend_invalid_coeff,
        check_alphatrend_inconsistent_lengths,
        check_alphatrend_reinput,
        check_alphatrend_nan_handling,
        check_alphatrend_streaming,
        check_alphatrend_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_alphatrend_tests!(check_alphatrend_property);

    // Batch testing functions
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let sweep = AlphaTrendBatchRange::default();
        let output = alphatrend_batch_with_kernel(&c, &sweep, kernel)?;

        let def = AlphaTrendParams::default();
        let row = output.row_for_params(&def).expect("default row missing");

        assert_eq!(output.cols, c.close.len());

        // Check that we got values for the default parameters
        let k1_start = row * output.cols;
        let k2_start = row * output.cols;

        // Verify last 5 K1 values match expected
        let expected_k1 = [
            60243.00,
            60243.00,
            60138.92857143,
            60088.42857143,
            59937.21428571,
        ];

        let start = output.cols - 5;
        for (i, &expected) in expected_k1.iter().enumerate() {
            let actual = output.values_k1[k1_start + start + i];
            assert!(
                (actual - expected).abs() < 1e-6,
                "[{test}] default-row K1 mismatch at idx {i}: {actual} vs {expected}"
            );
        }
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let sweep = AlphaTrendBatchRange {
            coeff: (1.0, 2.0, 0.5),
            period: (10, 20, 5),
            no_volume: false,
        };

        let output = alphatrend_batch_with_kernel(&c, &sweep, kernel)?;

        // Calculate expected number of combinations
        let coeff_count = ((2.0 - 1.0) / 0.5) as usize + 1; // 3 values
        let period_count = ((20 - 10) / 5) as usize + 1; // 3 values
        let expected_combos = coeff_count * period_count; // 9 total

        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (1.0, 1.0, 0.0, 10, 15, 5, false),
            (0.5, 2.0, 0.5, 14, 14, 0, true),
            (1.5, 1.5, 0.0, 7, 21, 7, false),
        ];

        for (cfg_idx, &(c_start, c_end, c_step, p_start, p_end, p_step, no_vol)) in
            test_configs.iter().enumerate()
        {
            let sweep = AlphaTrendBatchRange {
                coeff: (c_start, c_end, c_step),
                period: (p_start, p_end, p_step),
                no_volume: no_vol,
            };

            let output = alphatrend_batch_with_kernel(&c, &sweep, kernel)?;

            for (idx, &val) in output
                .values_k1
                .iter()
                .chain(output.values_k2.iter())
                .enumerate()
            {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    let combo = if row < output.combos.len() {
                        &output.combos[row]
                    } else {
                        &output.combos[row - output.combos.len()]
                    };

                    panic!(
                        "[{}] Config {}: Found poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: coeff={}, period={}, no_volume={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.coeff.unwrap_or(1.0),
                        combo.period.unwrap_or(14),
                        combo.no_volume.unwrap_or(false)
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

    // Macro for batch tests
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
}
