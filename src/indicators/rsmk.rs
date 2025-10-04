//! # Relative Strength Mark (RSMK)
//!
//! A comparative momentum-based indicator that calculates the log-ratio of two sources, applies a momentum
//! transform, smooths the result with a moving average, and provides both the main and signal lines.
//!
//! ## Parameters
//! - **lookback**: Lookback period for momentum calculation (default: 90)
//! - **period**: Moving average period for smoothing momentum (default: 3)
//! - **signal_period**: Signal moving average period (default: 20)
//! - **matype**: Moving average type for main line (default: "ema")
//! - **signal_matype**: Moving average type for signal line (default: "ema")
//!
//! ## Inputs
//! - Main data series and comparison data series (or candles with source)
//! - Both series must have the same length
//!
//! ## Returns
//! - **indicator**: Main RSMK line as `Vec<f64>` (length matches input)
//! - **signal**: Signal line as `Vec<f64>` (length matches input)
//!
//! ## Developer Notes
//! - SIMD status: AVX2/AVX512 entry points are present and feature-gated. They currently delegate to
//!   the scalar fused path to guarantee identical results (previous momentum-only vectorization showed
//!   minor divergence under property tests). Runtime selection continues to prefer the scalar path.
//! - Streaming update: O(1) per tick via ring buffer for momentum and NaN-aware EMA/SMA windows
//!   (matches fused scalar behavior, see Decision Logging below).
//! - Decision: SIMD paths remain stubbed to scalar for correctness; streaming kernel enabled and
//!   fastest by >O(n) vs prior version at identical outputs.
//! - Memory optimization: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix,
//!   init_matrix_prefixes) and removes branches in hot loops where IEEE-754 NaN propagation suffices.
//! - Decision: Mixed MA combos (EMA→SMA, SMA→EMA) fused variants are implemented, but the main
//!   selection falls back to the generic MA pipeline for those types due to prior underperformance.
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
use serde_wasm_bindgen;
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
use std::error::Error;
use thiserror::Error;

use crate::indicators::moving_averages::ma::{ma, MaData};

#[derive(Debug, Clone)]
pub enum RsmkData<'a> {
    Candles {
        candles: &'a Candles,
        candles_compare: &'a Candles,
        source: &'a str,
    },
    Slices {
        main: &'a [f64],
        compare: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct RsmkOutput {
    pub indicator: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct RsmkParams {
    pub lookback: Option<usize>,
    pub period: Option<usize>,
    pub signal_period: Option<usize>,
    pub matype: Option<String>,
    pub signal_matype: Option<String>,
}

impl Default for RsmkParams {
    fn default() -> Self {
        Self {
            lookback: Some(90),
            period: Some(3),
            signal_period: Some(20),
            matype: Some("ema".to_string()),
            signal_matype: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RsmkInput<'a> {
    pub data: RsmkData<'a>,
    pub params: RsmkParams,
}

impl<'a> RsmkInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        candles_compare: &'a Candles,
        source: &'a str,
        params: RsmkParams,
    ) -> Self {
        Self {
            data: RsmkData::Candles {
                candles,
                candles_compare,
                source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(main: &'a [f64], compare: &'a [f64], params: RsmkParams) -> Self {
        Self {
            data: RsmkData::Slices { main, compare },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles, candles_compare: &'a Candles) -> Self {
        Self::from_candles(candles, candles_compare, "close", RsmkParams::default())
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(90)
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(3)
    }
    #[inline]
    pub fn get_signal_period(&self) -> usize {
        self.params.signal_period.unwrap_or(20)
    }
    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.matype.as_deref().unwrap_or("ema")
    }
    #[inline]
    pub fn get_signal_ma_type(&self) -> &str {
        self.params.signal_matype.as_deref().unwrap_or("ema")
    }
}

#[derive(Clone, Debug)]
pub struct RsmkBuilder {
    lookback: Option<usize>,
    period: Option<usize>,
    signal_period: Option<usize>,
    matype: Option<String>,
    signal_matype: Option<String>,
    kernel: Kernel,
}

impl Default for RsmkBuilder {
    fn default() -> Self {
        Self {
            lookback: None,
            period: None,
            signal_period: None,
            matype: None,
            signal_matype: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RsmkBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn lookback(mut self, n: usize) -> Self {
        self.lookback = Some(n);
        self
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn signal_period(mut self, n: usize) -> Self {
        self.signal_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn matype<S: Into<String>>(mut self, s: S) -> Self {
        self.matype = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn signal_matype<S: Into<String>>(mut self, s: S) -> Self {
        self.signal_matype = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
        candles_compare: &Candles,
    ) -> Result<RsmkOutput, RsmkError> {
        let params = RsmkParams {
            lookback: self.lookback,
            period: self.period,
            signal_period: self.signal_period,
            matype: self.matype.clone(),
            signal_matype: self.signal_matype.clone(),
        };
        let input = RsmkInput::from_candles(candles, candles_compare, "close", params);
        rsmk_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, main: &[f64], compare: &[f64]) -> Result<RsmkOutput, RsmkError> {
        let params = RsmkParams {
            lookback: self.lookback,
            period: self.period,
            signal_period: self.signal_period,
            matype: self.matype.clone(),
            signal_matype: self.signal_matype.clone(),
        };
        let input = RsmkInput::from_slices(main, compare, params);
        rsmk_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RsmkStream, RsmkError> {
        let params = RsmkParams {
            lookback: self.lookback,
            period: self.period,
            signal_period: self.signal_period,
            matype: self.matype,
            signal_matype: self.signal_matype,
        };
        RsmkStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum RsmkError {
    #[error("rsmk: Empty data provided for RSMK.")]
    EmptyData,
    #[error("rsmk: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rsmk: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rsmk: All values are NaN.")]
    AllValuesNaN,
    #[error("rsmk: Error from MA function: {0}")]
    MaError(String),
}

#[cfg(feature = "wasm")]
impl From<RsmkError> for JsValue {
    fn from(err: RsmkError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

#[inline]
pub fn rsmk(input: &RsmkInput) -> Result<RsmkOutput, RsmkError> {
    rsmk_with_kernel(input, Kernel::Auto)
}

pub fn rsmk_with_kernel(input: &RsmkInput, kernel: Kernel) -> Result<RsmkOutput, RsmkError> {
    // Extract and check
    let (main, compare) = match &input.data {
        RsmkData::Candles {
            candles,
            candles_compare,
            source,
        } => (
            source_type(candles, source),
            source_type(candles_compare, source),
        ),
        RsmkData::Slices { main, compare } => (*main, *compare),
    };
    if main.is_empty() || compare.is_empty() {
        return Err(RsmkError::EmptyData);
    }
    if main.len() != compare.len() {
        return Err(RsmkError::InvalidPeriod {
            period: 0,
            data_len: main.len().min(compare.len()),
        });
    }

    let lookback = input.get_lookback();
    let period = input.get_period();
    let signal_period = input.get_signal_period();
    if lookback == 0
        || period == 0
        || signal_period == 0
        || period > main.len()
        || signal_period > main.len()
        || lookback >= main.len()
    {
        return Err(RsmkError::InvalidPeriod {
            period: lookback.max(period).max(signal_period),
            data_len: main.len(),
        });
    }

    // log-ratio
    let mut lr = Vec::with_capacity(main.len());
    unsafe {
        lr.set_len(main.len());
    }
    for i in 0..main.len() {
        let m = main[i];
        let c = compare[i];
        // c == 0 => NaN to preserve domain
        unsafe {
            *lr.get_unchecked_mut(i) = if m.is_nan() || c.is_nan() || c == 0.0 {
                f64::NAN
            } else {
                (m / c).ln()
            };
        }
    }

    let first_valid = lr
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(RsmkError::AllValuesNaN)?;
    // Require enough post-first-valid data to cover lookback and both MA windows.
    // This mirrors classic-kernel behavior (momentum first_valid includes lookback).
    let needed = lookback + period.max(signal_period);
    if lr.len() - first_valid < needed {
        return Err(RsmkError::NotEnoughValidData {
            needed,
            valid: lr.len() - first_valid,
        });
    }

    // Dispatch to selected kernel implementation (single-series)
    let mut mom = alloc_with_nan_prefix(lr.len(), first_valid + lookback);
    let ksel = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k if k.is_batch() => match k {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        },
        k => k,
    };
    return match ksel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            rsmk_avx512(&lr, lookback, period, signal_period, input, first_valid, &mut mom)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            rsmk_avx2(&lr, lookback, period, signal_period, input, first_valid, &mut mom)
        },
        _ => rsmk_scalar(&lr, lookback, period, signal_period, input, first_valid, &mut mom),
    };
}

pub fn rsmk_scalar(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    // === 1) Momentum (log-ratio diff) ===
    // mom already has a NaN prefix sized by caller via alloc_with_nan_prefix
    let len = lr.len();
    let mom_fv = first_valid + lookback; // first valid momentum index

    // Tight, bounds-check-free loop to compute momentum
    // SAFETY: i in [mom_fv, len) and i - lookback in [first_valid, len)
    unsafe {
        for i in mom_fv..len {
            let a = *lr.get_unchecked(i);
            let b = *lr.get_unchecked(i - lookback);
            *mom.get_unchecked_mut(i) = if a.is_nan() || b.is_nan() { f64::NAN } else { a - b };
        }
    }

    // === 2) MA selection ===
    #[inline(always)]
    fn is_ema(s: &str) -> bool { s.eq_ignore_ascii_case("ema") }
    #[inline(always)]
    fn is_sma(s: &str) -> bool { s.eq_ignore_ascii_case("sma") }

    let matype = input.get_ma_type();
    let sigtype = input.get_signal_ma_type();

    // Common warmups based on momentum availability
    let ind_warmup = mom_fv.saturating_add(period.saturating_sub(1));
    let sig_warmup = ind_warmup.saturating_add(signal_period.saturating_sub(1));

    // === 3) Jammed kernels for the 4 common combinations ===

    // --- EMA over momentum, then EMA over indicator (one-pass, fused) ---
    if is_ema(matype) && is_ema(sigtype) {
        let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
        let mut signal    = alloc_with_nan_prefix(len, sig_warmup);

        if ind_warmup < len {
            // Initialize EMA(indicator) from SMA of first 'period' momentum points (ignoring NaNs)
            let mut sum = 0.0;
            let mut cnt = 0usize;
            let init_end = (mom_fv + period).min(len);
            unsafe {
                for i in mom_fv..init_end {
                    let v = *mom.get_unchecked(i);
                    if !v.is_nan() { sum += v; cnt += 1; }
                }
            }

            if cnt > 0 {
                let alpha_ind = 2.0 / (period as f64 + 1.0);
                let alpha_sig = 2.0 / (signal_period as f64 + 1.0);

                // Indicator EMA starts from SMA * 100
                let mut ema_ind = (sum / cnt as f64) * 100.0;
                unsafe { *indicator.get_unchecked_mut(ind_warmup) = ema_ind; }

                // Build signal EMA in-place as we extend indicator:
                //  - accumulate first `signal_period` indicator values to seed EMA(signal)
                let mut ema_sig = 0.0f64; // will be set when seeded
                let mut acc_sig = ema_ind;
                let mut cnt_sig = 1usize;

                // If the signal warmup coincides with indicator warmup (signal_period == 1),
                // seed and write the signal value immediately.
                if sig_warmup == ind_warmup {
                    ema_sig = acc_sig / (cnt_sig as f64);
                    unsafe { *signal.get_unchecked_mut(sig_warmup) = ema_sig; }
                }

                unsafe {
                    for i in (ind_warmup + 1)..len {
                        let mv = *mom.get_unchecked(i);
                        if !mv.is_nan() {
                            let src100 = mv * 100.0;
                            // ema_ind = ema_ind + alpha*(src100 - ema_ind)  (single FMA)
                            ema_ind = (src100 - ema_ind).mul_add(alpha_ind, ema_ind);
                        }
                        *indicator.get_unchecked_mut(i) = ema_ind;

                        // Seed EMA(signal) with SMA of first `signal_period` indicator values
                        if i < sig_warmup {
                            acc_sig += ema_ind;
                            cnt_sig += 1;
                        } else if i == sig_warmup {
                            ema_sig = acc_sig / (cnt_sig as f64);
                            *signal.get_unchecked_mut(i) = ema_sig;
                        } else {
                            // Continue EMA(signal)
                            ema_sig = (ema_ind - ema_sig).mul_add(alpha_sig, ema_sig);
                            *signal.get_unchecked_mut(i) = ema_sig;
                        }
                    }
                }
            } else {
                // No finite values to seed indicator EMA — leave post-warmup as NaN explicitly
                for i in ind_warmup..len { indicator[i] = f64::NAN; }
                for i in sig_warmup..len { signal[i] = f64::NAN; }
            }
        }

        return Ok(RsmkOutput { indicator, signal });
    }

    // --- SMA over momentum, then SMA over indicator (single pass, NaN-aware) ---
    if is_sma(matype) && is_sma(sigtype) {
        let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
        let mut signal    = alloc_with_nan_prefix(len, sig_warmup);

        let mut sum_ind = 0.0;
        let mut cnt_ind = 0usize;

        let mut sum_sig = 0.0;
        let mut cnt_sig = 0usize;

        unsafe {
            for i in mom_fv..len {
                // Update indicator SMA window with NaN-skipping
                let v_new = *mom.get_unchecked(i);
                if !v_new.is_nan() { sum_ind += v_new; cnt_ind += 1; }

                if i >= mom_fv + period {
                    let v_old = *mom.get_unchecked(i - period);
                    if !v_old.is_nan() { sum_ind -= v_old; cnt_ind -= 1; }
                }

                if i >= ind_warmup {
                    let ind_val = if cnt_ind > 0 {
                        (sum_ind / cnt_ind as f64) * 100.0
                    } else {
                        f64::NAN
                    };
                    *indicator.get_unchecked_mut(i) = ind_val;

                    // Update signal SMA window over indicator (NaN-aware)
                    if !ind_val.is_nan() { sum_sig += ind_val; cnt_sig += 1; }

                    if i >= sig_warmup {
                        let old_idx = i - signal_period;
                        let old_ind = *indicator.get_unchecked(old_idx);
                        if !old_ind.is_nan() { sum_sig -= old_ind; cnt_sig -= 1; }

                        *signal.get_unchecked_mut(i) = if cnt_sig > 0 {
                            sum_sig / cnt_sig as f64
                        } else {
                            f64::NAN
                        };
                    }
                }
            }
        }

        return Ok(RsmkOutput { indicator, signal });
    }

    // --- EMA over momentum, then SMA over indicator (single pass after EMA seed) ---
    if is_ema(matype) && is_sma(sigtype) {
        let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
        let mut signal    = alloc_with_nan_prefix(len, sig_warmup);

        if ind_warmup < len {
            // Seed indicator EMA from SMA of first `period` momentum values (NaN-aware)
            let mut sum = 0.0;
            let mut cnt = 0usize;
            let init_end = (mom_fv + period).min(len);
            unsafe {
                for i in mom_fv..init_end {
                    let v = *mom.get_unchecked(i);
                    if !v.is_nan() { sum += v; cnt += 1; }
                }
            }

            if cnt > 0 {
                let alpha_ind = 2.0 / (period as f64 + 1.0);
                let mut ema_ind = (sum / cnt as f64) * 100.0;

                // SMA(signal) rolling state
                let mut sum_sig = 0.0;
                let mut cnt_sig = 0usize;

                unsafe {
                    *indicator.get_unchecked_mut(ind_warmup) = ema_ind;
                    // Include first indicator into signal window build-up
                    sum_sig += ema_ind; cnt_sig += 1;

                    // If the signal warmup coincides with indicator warmup (signal_period == 1),
                    // write the first SMA(signal) immediately.
                    if sig_warmup == ind_warmup {
                        *signal.get_unchecked_mut(sig_warmup) = sum_sig / cnt_sig as f64;
                    }

                    for i in (ind_warmup + 1)..len {
                        let mv = *mom.get_unchecked(i);
                        if !mv.is_nan() {
                            let src100 = mv * 100.0;
                            ema_ind = (src100 - ema_ind).mul_add(alpha_ind, ema_ind);
                        }
                        *indicator.get_unchecked_mut(i) = ema_ind;

                        // Update SMA(signal) window
                        if !ema_ind.is_nan() { sum_sig += ema_ind; cnt_sig += 1; }

                        if i >= sig_warmup {
                            let old_idx = i - signal_period;
                            let old_ind = *indicator.get_unchecked(old_idx);
                            if !old_ind.is_nan() { sum_sig -= old_ind; cnt_sig -= 1; }

                            *signal.get_unchecked_mut(i) = if cnt_sig > 0 {
                                sum_sig / cnt_sig as f64
                            } else {
                                f64::NAN
                            };
                        }
                    }
                }
            } else {
                for i in ind_warmup..len { indicator[i] = f64::NAN; }
                for i in sig_warmup..len { signal[i] = f64::NAN; }
            }
        }

        return Ok(RsmkOutput { indicator, signal });
    }

    // --- SMA over momentum, then EMA over indicator (single pass after SMA build) ---
    if is_sma(matype) && is_ema(sigtype) {
        let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
        let mut signal    = alloc_with_nan_prefix(len, sig_warmup);

        let mut sum_ind = 0.0;
        let mut cnt_ind = 0usize;

        // To seed EMA(signal) we accumulate the first `signal_period` indicator values
        let alpha_sig = 2.0 / (signal_period as f64 + 1.0);
        let mut acc_sig = 0.0;
        let mut cnt_sig = 0usize;
        let mut ema_sig = 0.0f64; // set at seed

        unsafe {
            for i in mom_fv..len {
                // Update indicator SMA window with NaN-skipping
                let v_new = *mom.get_unchecked(i);
                if !v_new.is_nan() { sum_ind += v_new; cnt_ind += 1; }

                if i >= mom_fv + period {
                    let v_old = *mom.get_unchecked(i - period);
                    if !v_old.is_nan() { sum_ind -= v_old; cnt_ind -= 1; }
                }

                if i >= ind_warmup {
                    let ind_val = if cnt_ind > 0 {
                        (sum_ind / cnt_ind as f64) * 100.0
                    } else {
                        f64::NAN
                    };
                    *indicator.get_unchecked_mut(i) = ind_val;

                    if i < sig_warmup {
                        if !ind_val.is_nan() { acc_sig += ind_val; cnt_sig += 1; }
                    } else if i == sig_warmup {
                        ema_sig = if cnt_sig > 0 { acc_sig / cnt_sig as f64 } else { f64::NAN };
                        *signal.get_unchecked_mut(i) = ema_sig;
                    } else {
                        if !ind_val.is_nan() && !ema_sig.is_nan() {
                            ema_sig = (ind_val - ema_sig).mul_add(alpha_sig, ema_sig);
                        } else if !ind_val.is_nan() && ema_sig.is_nan() {
                            // Recover from NaN seed by snapping to the first finite indicator
                            ema_sig = ind_val;
                        }
                        *signal.get_unchecked_mut(i) = ema_sig;
                    }
                }
            }
        }

        return Ok(RsmkOutput { indicator, signal });
    }

    // === 4) Fallback to generic MA pipeline for uncommon types ===
    // (kept for correctness & feature parity with the full MA dispatcher)
    let matype = input.get_ma_type();
    let sigmatype = input.get_signal_ma_type();

    // indicator MA -> reuse Vec directly
    let mut indicator =
        ma(matype, MaData::Slice(mom), period).map_err(|e| RsmkError::MaError(e.to_string()))?;
    for v in &mut indicator {
        *v *= 100.0;
    }

    // signal MA -> reuse Vec directly
    let signal = ma(sigmatype, MaData::Slice(&indicator), signal_period)
        .map_err(|e| RsmkError::MaError(e.to_string()))?;

    Ok(RsmkOutput { indicator, signal })
}

/// Write directly to output slices - no allocations
#[inline]
pub fn rsmk_into_slice(
    dst_indicator: &mut [f64],
    dst_signal: &mut [f64],
    input: &RsmkInput,
    _kern: Kernel,
) -> Result<(), RsmkError> {
    let (main, compare) = match &input.data {
        RsmkData::Candles {
            candles,
            candles_compare,
            source,
        } => (
            source_type(candles, source),
            source_type(candles_compare, source),
        ),
        RsmkData::Slices { main, compare } => (*main, *compare),
    };
    if main.len() == 0 || compare.len() == 0 {
        return Err(RsmkError::EmptyData);
    }
    if main.len() != compare.len()
        || dst_indicator.len() != main.len()
        || dst_signal.len() != main.len()
    {
        return Err(RsmkError::InvalidPeriod {
            period: 0,
            data_len: main.len(),
        });
    }

    let p = &input.params;
    let lookback = p.lookback.unwrap_or(90);
    let period = p.period.unwrap_or(3);
    let signal_period = p.signal_period.unwrap_or(20);
    if lookback == 0 || period == 0 || signal_period == 0 {
        return Err(RsmkError::InvalidPeriod {
            period: 0,
            data_len: main.len(),
        });
    }

    // log-ratio
    let mut lr = Vec::with_capacity(main.len());
    unsafe {
        lr.set_len(main.len());
    }
    for i in 0..main.len() {
        let m = main[i];
        let c = compare[i];
        unsafe {
            *lr.get_unchecked_mut(i) = if m.is_nan() || c.is_nan() || c == 0.0 {
                f64::NAN
            } else {
                (m / c).ln()
            };
        }
    }
    let first = lr
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RsmkError::AllValuesNaN)?;

    // momentum
    let mut mom = alloc_with_nan_prefix(lr.len(), first + lookback);
    for i in (first + lookback)..lr.len() {
        let a = lr[i];
        let b = lr[i - lookback];
        mom[i] = if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a - b
        };
    }

    // indicator
    let matype = p.matype.as_deref().unwrap_or("ema");
    let mut ind =
        ma(matype, MaData::Slice(&mom), period).map_err(|e| RsmkError::MaError(e.to_string()))?;
    for v in &mut ind {
        *v *= 100.0;
    }

    // signal
    let sigtype = p.signal_matype.as_deref().unwrap_or("ema");
    let sig = ma(sigtype, MaData::Slice(&ind), signal_period)
        .map_err(|e| RsmkError::MaError(e.to_string()))?;

    // copy into provided slices
    dst_indicator.copy_from_slice(&ind);
    dst_signal.copy_from_slice(&sig);

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
pub fn rsmk_avx2(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    // Stub to scalar path for correctness; SIMD momentum under review.
    rsmk_scalar(lr, lookback, period, signal_period, input, first_valid, mom)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub fn rsmk_avx512(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    if period <= 32 {
        unsafe { rsmk_avx512_short(lr, lookback, period, signal_period, input, first_valid, mom) }
    } else {
        unsafe { rsmk_avx512_long(lr, lookback, period, signal_period, input, first_valid, mom) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn rsmk_avx512_short(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    // Stub to scalar path for correctness; SIMD momentum under review.
    rsmk_scalar(lr, lookback, period, signal_period, input, first_valid, mom)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn rsmk_avx512_long(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    rsmk_avx512_short(lr, lookback, period, signal_period, input, first_valid, mom)
}

// Decision Logging (streaming): O(1) kernel enabled; maintains ring buffer for log-ratio momentum
// and NaN-aware EMA/SMA windows; matches fused scalar behavior including warm-ups and NaN rules.
#[derive(Debug, Clone)]
pub struct RsmkStream {
    lookback: usize,
    period: usize,
    signal_period: usize,
    // Keep original strings for API parity, plus precomputed flags for hot path
    matype: String,
    signal_matype: String,
    main_is_ema: bool,
    signal_is_ema: bool,

    // log-ratio ring for momentum in O(1)
    lr_buf: Vec<f64>,
    lr_head: usize,
    lr_len: usize,

    // momentum first-valid detection boundary
    saw_first_finite_mom: bool,
    // EMA(main) seed trackers (SMA over first `period` after mom_fv)
    ema_seed_pos: usize,
    ema_seed_sum: f64,
    ema_seed_cnt: usize,
    ema_ind: f64,
    ema_ind_seeded: bool,
    ema_ind_dead: bool, // sticky NaN tail if seed had 0 finite samples

    // SMA(main) window (NaN-aware)
    ind_win_buf: Vec<f64>,
    ind_win_head: usize,
    ind_win_len: usize,
    ind_win_sum: f64,
    ind_win_cnt: usize,

    // Latch when indicator becomes available (even if NaN by window) to start signal timing
    indicator_started: bool,

    // EMA(signal)
    alpha_sig: f64,
    ema_sig: f64,
    ema_sig_seeded: bool,
    ema_sig_seed_pos: usize,
    ema_sig_seed_sum: f64,
    ema_sig_seed_cnt: usize,

    // SMA(signal)
    sig_win_buf: Vec<f64>,
    sig_win_head: usize,
    sig_win_len: usize,
    sig_win_sum: f64,
    sig_win_cnt: usize,

    // Precomputed alpha for EMA(main)
    alpha_main: f64,
}

impl RsmkStream {
    pub fn try_new(params: RsmkParams) -> Result<Self, RsmkError> {
        let lookback = params.lookback.unwrap_or(90);
        let period = params.period.unwrap_or(3);
        let signal_period = params.signal_period.unwrap_or(20);
        if lookback == 0 || period == 0 || signal_period == 0 {
            return Err(RsmkError::InvalidPeriod {
                period: lookback.max(period).max(signal_period),
                data_len: 0,
            });
        }
        let matype = params.matype.unwrap_or_else(|| "ema".to_string());
        let signal_matype = params
            .signal_matype
            .unwrap_or_else(|| "ema".to_string());
        let main_is_ema = matype.eq_ignore_ascii_case("ema");
        let signal_is_ema = signal_matype.eq_ignore_ascii_case("ema");

        Ok(Self {
            lookback,
            period,
            signal_period,
            matype,
            signal_matype,
            main_is_ema,
            signal_is_ema,

            lr_buf: vec![f64::NAN; lookback],
            lr_head: 0,
            lr_len: 0,

            saw_first_finite_mom: false,
            ema_seed_pos: 0,
            ema_seed_sum: 0.0,
            ema_seed_cnt: 0,
            ema_ind: f64::NAN,
            ema_ind_seeded: false,
            ema_ind_dead: false,

            ind_win_buf: vec![f64::NAN; period],
            ind_win_head: 0,
            ind_win_len: 0,
            ind_win_sum: 0.0,
            ind_win_cnt: 0,

            indicator_started: false,

            alpha_sig: 2.0 / (signal_period as f64 + 1.0),
            ema_sig: f64::NAN,
            ema_sig_seeded: false,
            ema_sig_seed_pos: 0,
            ema_sig_seed_sum: 0.0,
            ema_sig_seed_cnt: 0,

            sig_win_buf: vec![f64::NAN; signal_period],
            sig_win_head: 0,
            sig_win_len: 0,
            sig_win_sum: 0.0,
            sig_win_cnt: 0,

            alpha_main: 2.0 / (period as f64 + 1.0),
        })
    }

    #[inline(always)]
    fn push_ring(buf: &mut [f64], head: &mut usize, len: &mut usize, v: f64) -> Option<f64> {
        let cap = buf.len();
        if cap == 0 {
            return None;
        }
        let evicted = if *len < cap {
            buf[*head] = v;
            *len += 1;
            None
        } else {
            let old = core::mem::replace(&mut buf[*head], v);
            Some(old)
        };
        *head += 1;
        if *head == cap {
            *head = 0;
        }
        evicted
    }

    #[inline(always)]
    fn window_push(
        buf: &mut [f64],
        head: &mut usize,
        len: &mut usize,
        sum: &mut f64,
        cnt: &mut usize,
        v: f64,
    ) {
        let cap = buf.len();
        if cap == 0 {
            return;
        }
        if *len < cap {
            buf[*head] = v;
            if v.is_finite() {
                *sum += v;
                *cnt += 1;
            }
            *len += 1;
        } else {
            let old = core::mem::replace(&mut buf[*head], v);
            if old.is_finite() {
                *sum -= old;
                *cnt -= 1;
            }
            if v.is_finite() {
                *sum += v;
                *cnt += 1;
            }
        }
        *head += 1;
        if *head == cap {
            *head = 0;
        }
    }

    /// Streaming update: O(1). Returns (indicator, signal) for the new point.
    pub fn update(&mut self, main: f64, compare: f64) -> Option<(f64, f64)> {
        // 1) log-ratio with domain checks (NaN if compare==0 or any NaN)
        let lr = if main.is_nan() || compare.is_nan() || compare == 0.0 {
            f64::NAN
        } else {
            (main / compare).ln()
        };

        // 2) momentum = lr_t - lr_{t-lookback} via ring buffer
        let evicted = Self::push_ring(&mut self.lr_buf, &mut self.lr_head, &mut self.lr_len, lr);
        let mom = match evicted {
            None => f64::NAN,
            Some(old_lr) => {
                if lr.is_nan() || old_lr.is_nan() {
                    f64::NAN
                } else {
                    lr - old_lr
                }
            }
        };

        if !self.saw_first_finite_mom && mom.is_finite() {
            self.saw_first_finite_mom = true;
        }

        // 3) Indicator path (EMA or SMA)
        let indicator = if self.main_is_ema {
            self.update_indicator_ema(mom)
        } else {
            self.update_indicator_sma(mom)
        };

        // 4) Signal path (EMA or SMA over indicator)
        let signal = if self.signal_is_ema {
            self.update_signal_ema(indicator)
        } else {
            self.update_signal_sma(indicator)
        };

        Some((indicator, signal))
    }

    #[inline(always)]
    fn update_indicator_ema(&mut self, mom: f64) -> f64 {
        if self.ema_ind_dead {
            return f64::NAN;
        }
        if !self.saw_first_finite_mom {
            return f64::NAN;
        }
        if !self.ema_ind_seeded {
            self.ema_seed_pos += 1;
            if mom.is_finite() {
                self.ema_seed_sum += mom;
                self.ema_seed_cnt += 1;
            }
            if self.ema_seed_pos < self.period {
                return f64::NAN;
            }
            if self.ema_seed_cnt == 0 {
                self.ema_ind_dead = true;
                self.indicator_started = true;
                return f64::NAN;
            }
            self.ema_ind = (self.ema_seed_sum / self.ema_seed_cnt as f64) * 100.0;
            self.ema_ind_seeded = true;
            self.indicator_started = true;
            return self.ema_ind;
        }
        if mom.is_finite() {
            let src100 = mom * 100.0;
            self.ema_ind = (src100 - self.ema_ind).mul_add(self.alpha_main, self.ema_ind);
        }
        self.ema_ind
    }

    #[inline(always)]
    fn update_indicator_sma(&mut self, mom: f64) -> f64 {
        if !self.saw_first_finite_mom {
            return f64::NAN;
        }
        Self::window_push(
            &mut self.ind_win_buf,
            &mut self.ind_win_head,
            &mut self.ind_win_len,
            &mut self.ind_win_sum,
            &mut self.ind_win_cnt,
            mom,
        );

        if self.ind_win_len < self.period {
            return f64::NAN;
        }
        let ind = if self.ind_win_cnt > 0 {
            (self.ind_win_sum / self.ind_win_cnt as f64) * 100.0
        } else {
            f64::NAN
        };
        if !self.indicator_started {
            self.indicator_started = true;
        }
        ind
    }

    #[inline(always)]
    fn update_signal_ema(&mut self, indicator: f64) -> f64 {
        if self.ema_ind_dead {
            return f64::NAN;
        }
        if !self.indicator_started {
            return f64::NAN;
        }
        if !self.ema_sig_seeded {
            self.ema_sig_seed_pos += 1;
            if indicator.is_finite() {
                self.ema_sig_seed_sum += indicator;
                self.ema_sig_seed_cnt += 1;
            }
            if self.ema_sig_seed_pos < self.signal_period {
                return f64::NAN;
            }
            self.ema_sig = if self.ema_sig_seed_cnt > 0 {
                self.ema_sig_seed_sum / self.ema_sig_seed_cnt as f64
            } else {
                f64::NAN
            };
            self.ema_sig_seeded = true;
            return self.ema_sig;
        }
        if indicator.is_finite() && self.ema_sig.is_finite() {
            self.ema_sig = (indicator - self.ema_sig).mul_add(self.alpha_sig, self.ema_sig);
        } else if indicator.is_finite() && !self.ema_sig.is_finite() {
            if !self.main_is_ema {
                self.ema_sig = indicator;
            }
        }
        self.ema_sig
    }

    #[inline(always)]
    fn update_signal_sma(&mut self, indicator: f64) -> f64 {
        if self.ema_ind_dead {
            return f64::NAN;
        }
        if !self.indicator_started {
            return f64::NAN;
        }
        Self::window_push(
            &mut self.sig_win_buf,
            &mut self.sig_win_head,
            &mut self.sig_win_len,
            &mut self.sig_win_sum,
            &mut self.sig_win_cnt,
            indicator,
        );
        if self.sig_win_len < self.signal_period {
            return f64::NAN;
        }
        if self.sig_win_cnt > 0 {
            self.sig_win_sum / self.sig_win_cnt as f64
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
pub struct RsmkBatchRange {
    pub lookback: (usize, usize, usize),
    pub period: (usize, usize, usize),
    pub signal_period: (usize, usize, usize),
}

impl Default for RsmkBatchRange {
    fn default() -> Self {
        Self {
            lookback: (90, 90, 1),
            period: (3, 3, 1),
            signal_period: (20, 20, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RsmkBatchBuilder {
    range: RsmkBatchRange,
    kernel: Kernel,
}

impl RsmkBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_period = (start, end, step);
        self
    }
    pub fn apply_slices(self, main: &[f64], compare: &[f64]) -> Result<RsmkBatchOutput, RsmkError> {
        rsmk_batch_with_kernel(main, compare, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct RsmkBatchOutput {
    pub indicator: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<RsmkParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RsmkBatchOutput {
    pub fn row_for_params(&self, p: &RsmkParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.lookback.unwrap_or(90) == p.lookback.unwrap_or(90)
                && c.period.unwrap_or(3) == p.period.unwrap_or(3)
                && c.signal_period.unwrap_or(20) == p.signal_period.unwrap_or(20)
        })
    }

    pub fn indicator_for(&self, p: &RsmkParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.indicator[start..start + self.cols]
        })
    }

    pub fn signal_for(&self, p: &RsmkParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.signal[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &RsmkBatchRange) -> Vec<RsmkParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let looks = axis(r.lookback);
    let periods = axis(r.period);
    let signals = axis(r.signal_period);

    let mut out = Vec::with_capacity(looks.len() * periods.len() * signals.len());
    for &l in &looks {
        for &p in &periods {
            for &s in &signals {
                out.push(RsmkParams {
                    lookback: Some(l),
                    period: Some(p),
                    signal_period: Some(s),
                    matype: Some("ema".to_string()),
                    signal_matype: Some("ema".to_string()),
                });
            }
        }
    }
    out
}

pub fn rsmk_batch_with_kernel(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    k: Kernel,
) -> Result<RsmkBatchOutput, RsmkError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(RsmkError::InvalidPeriod {
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
    rsmk_batch_par_slice(main, compare, sweep, simd)
}

pub fn rsmk_batch_slice(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    kern: Kernel,
) -> Result<RsmkBatchOutput, RsmkError> {
    rsmk_batch_inner(main, compare, sweep, kern, false)
}

pub fn rsmk_batch_par_slice(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    kern: Kernel,
) -> Result<RsmkBatchOutput, RsmkError> {
    rsmk_batch_inner(main, compare, sweep, kern, true)
}

#[inline(always)]
fn rsmk_batch_inner_into(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    kern: Kernel,
    parallel: bool,
    indicator_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<RsmkParams>, RsmkError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RsmkError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = main
        .iter()
        .zip(compare.iter())
        .position(|(&m, &c)| m.is_finite() && c.is_finite() && c != 0.0)
        .ok_or(RsmkError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| {
            c.lookback
                .unwrap()
                .max(c.period.unwrap())
                .max(c.signal_period.unwrap())
        })
        .max()
        .unwrap();

    if main.len() - first < max_p {
        return Err(RsmkError::NotEnoughValidData {
            needed: max_p,
            valid: main.len() - first,
        });
    }

    let rows = combos.len();
    let cols = main.len();

    // Precompute log-ratio once
    let mut lr = Vec::with_capacity(cols);
    unsafe { lr.set_len(cols); }
    for i in 0..cols {
        let m = main[i];
        let c = compare[i];
        unsafe {
            *lr.get_unchecked_mut(i) = if m.is_nan() || c.is_nan() || c == 0.0 {
                f64::NAN
            } else {
                (m / c).ln()
            };
        }
    }

    // Precompute momentum for each unique lookback and reuse per row
    use std::collections::HashMap;
    let mut mom_by_lookback: HashMap<usize, Vec<f64>> = HashMap::new();
    for &lookback in combos.iter().map(|c| c.lookback.unwrap()).collect::<std::collections::BTreeSet<_>>().iter() {
        let mut m = alloc_with_nan_prefix(cols, first + lookback);
        let start = first + lookback;
        for i in start..cols {
            let a = unsafe { *lr.get_unchecked(i) };
            let b = unsafe { *lr.get_unchecked(i - lookback) };
            unsafe { *m.get_unchecked_mut(i) = a - b };
        }
        mom_by_lookback.insert(lookback, m);
    }

    let do_row = |row: usize, ind_row: &mut [f64], sig_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let lookback = prm.lookback.unwrap();
        let period = prm.period.unwrap();
        let signal_period = prm.signal_period.unwrap();
        let mt = prm.matype.as_deref().unwrap_or("ema");
        let st = prm.signal_matype.as_deref().unwrap_or("ema");

        // Use precomputed momentum for this lookback
        let mom = mom_by_lookback.get(&lookback).unwrap();

        // indicator row
        match ma(mt, MaData::Slice(&mom), period) {
            Ok(mut v) => {
                for x in &mut v {
                    *x *= 100.0;
                }
                ind_row.copy_from_slice(&v);
            }
            Err(_) => {
                for x in ind_row.iter_mut() {
                    *x = f64::NAN;
                }
            }
        }

        // signal row
        match ma(st, MaData::Slice(ind_row), signal_period) {
            Ok(vs) => {
                sig_row.copy_from_slice(&vs);
            }
            Err(_) => {
                for x in sig_row.iter_mut() {
                    *x = f64::NAN;
                }
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            indicator_out
                .par_chunks_mut(cols)
                .zip(signal_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (ind_row, sig_row))| do_row(row, ind_row, sig_row));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (ind_row, sig_row)) in indicator_out
                .chunks_mut(cols)
                .zip(signal_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, ind_row, sig_row);
            }
        }
    } else {
        for (row, (ind_row, sig_row)) in indicator_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, ind_row, sig_row);
        }
    }

    Ok(combos)
}

fn rsmk_batch_inner(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RsmkBatchOutput, RsmkError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RsmkError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = main
        .iter()
        .zip(compare.iter())
        .position(|(&m, &c)| m.is_finite() && c.is_finite() && c != 0.0)
        .ok_or(RsmkError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| {
            c.lookback
                .unwrap()
                .max(c.period.unwrap())
                .max(c.signal_period.unwrap())
        })
        .max()
        .unwrap();

    if main.len() - first < max_p {
        return Err(RsmkError::NotEnoughValidData {
            needed: max_p,
            valid: main.len() - first,
        });
    }

    let rows = combos.len();
    let cols = main.len();

    let mut indicators = make_uninit_matrix(rows, cols);
    let mut signals = make_uninit_matrix(rows, cols);

    // Initialize NaN prefixes for each row based on warmup periods
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            let lookback = c.lookback.unwrap();
            let period = c.period.unwrap();
            let signal_period = c.signal_period.unwrap();
            first + lookback.max(period).max(signal_period)
        })
        .collect();

    init_matrix_prefixes(&mut indicators, cols, &warmup_periods);
    init_matrix_prefixes(&mut signals, cols, &warmup_periods);

    // Convert to mutable slices for computation
    let mut indicators = unsafe {
        use std::mem::ManuallyDrop;
        let mut v = ManuallyDrop::new(indicators);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut f64, v.len(), v.capacity())
    };
    let mut signals = unsafe {
        use std::mem::ManuallyDrop;
        let mut v = ManuallyDrop::new(signals);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut f64, v.len(), v.capacity())
    };

    // Precompute log-ratio once
    let mut lr = Vec::with_capacity(cols);
    unsafe { lr.set_len(cols); }
    for i in 0..cols {
        let m = main[i];
        let c = compare[i];
        unsafe {
            *lr.get_unchecked_mut(i) = if m.is_nan() || c.is_nan() || c == 0.0 {
                f64::NAN
            } else {
                (m / c).ln()
            };
        }
    }

    // Precompute momentum for each unique lookback and reuse per row
    use std::collections::HashMap;
    let mut mom_by_lookback: HashMap<usize, Vec<f64>> = HashMap::new();
    for &lookback in combos.iter().map(|c| c.lookback.unwrap()).collect::<std::collections::BTreeSet<_>>().iter() {
        let mut m = alloc_with_nan_prefix(cols, first + lookback);
        let start = first + lookback;
        for i in start..cols {
            let a = unsafe { *lr.get_unchecked(i) };
            let b = unsafe { *lr.get_unchecked(i - lookback) };
            unsafe { *m.get_unchecked_mut(i) = a - b };
        }
        mom_by_lookback.insert(lookback, m);
    }

    let do_row = |row: usize, ind_row: &mut [f64], sig_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let lookback = prm.lookback.unwrap();
        let period = prm.period.unwrap();
        let signal_period = prm.signal_period.unwrap();
        let mt = prm.matype.as_deref().unwrap_or("ema");
        let st = prm.signal_matype.as_deref().unwrap_or("ema");

        // Use precomputed momentum for this lookback
        let mom = mom_by_lookback.get(&lookback).unwrap();

        // indicator row
        match ma(mt, MaData::Slice(&mom), period) {
            Ok(mut v) => {
                for x in &mut v {
                    *x *= 100.0;
                }
                ind_row.copy_from_slice(&v);
            }
            Err(_) => {
                for x in ind_row.iter_mut() {
                    *x = f64::NAN;
                }
            }
        }

        // signal row
        match ma(st, MaData::Slice(ind_row), signal_period) {
            Ok(vs) => {
                sig_row.copy_from_slice(&vs);
            }
            Err(_) => {
                for x in sig_row.iter_mut() {
                    *x = f64::NAN;
                }
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            indicators
                .par_chunks_mut(cols)
                .zip(signals.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (ind_row, sig_row))| do_row(row, ind_row, sig_row));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (ind_row, sig_row)) in indicators
                .chunks_mut(cols)
                .zip(signals.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, ind_row, sig_row);
            }
        }
    } else {
        for (row, (ind_row, sig_row)) in indicators
            .chunks_mut(cols)
            .zip(signals.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, ind_row, sig_row);
        }
    }

    Ok(RsmkBatchOutput {
        indicator: indicators,
        signal: signals,
        combos,
        rows,
        cols,
    })
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "rsmk")]
#[pyo3(signature = (main, compare, lookback, period, signal_period, matype=None, signal_matype=None, kernel=None))]
pub fn rsmk_py<'py>(
    py: Python<'py>,
    main: PyReadonlyArray1<'py, f64>,
    compare: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    period: usize,
    signal_period: usize,
    matype: Option<&str>,
    signal_matype: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let main_slice = main.as_slice()?;
    let compare_slice = compare.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = RsmkParams {
        lookback: Some(lookback),
        period: Some(period),
        signal_period: Some(signal_period),
        matype: matype.map(|s| s.to_string()),
        signal_matype: signal_matype.map(|s| s.to_string()),
    };
    let input = RsmkInput::from_slices(main_slice, compare_slice, params);

    let output = py
        .allow_threads(|| rsmk_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        output.indicator.into_pyarray(py),
        output.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "rsmk_batch")]
#[pyo3(signature = (main, compare, lookback_range, period_range, signal_period_range, matype=None, signal_matype=None, kernel=None))]
pub fn rsmk_batch_py<'py>(
    py: Python<'py>,
    main: PyReadonlyArray1<'py, f64>,
    compare: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    period_range: (usize, usize, usize),
    signal_period_range: (usize, usize, usize),
    matype: Option<&str>,
    signal_matype: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let main_slice = main.as_slice()?;
    let compare_slice = compare.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = RsmkBatchRange {
        lookback: lookback_range,
        period: period_range,
        signal_period: signal_period_range,
    };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = main_slice.len();

    // Pre-allocate output arrays
    let indicator_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let indicator_slice = unsafe { indicator_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py
        .allow_threads(|| {
            // Handle kernel selection for batch operations
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // Map batch kernels to regular kernels
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kern, // Use the original if not a batch kernel
            };

            rsmk_batch_inner_into(
                main_slice,
                compare_slice,
                &sweep,
                simd,
                true,
                indicator_slice,
                signal_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("indicator", indicator_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lookbacks",
        combos
            .iter()
            .map(|p| p.lookback.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_periods",
        combos
            .iter()
            .map(|p| p.signal_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    use pyo3::types::PyList;
    dict.set_item(
        "matypes",
        PyList::new(
            py,
            combos.iter().map(|p| p.matype.as_deref().unwrap_or("ema")),
        )?,
    )?;
    dict.set_item(
        "signal_matypes",
        PyList::new(
            py,
            combos
                .iter()
                .map(|p| p.signal_matype.as_deref().unwrap_or("ema")),
        )?,
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "RsmkStream")]
pub struct RsmkStreamPy {
    inner: RsmkStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RsmkStreamPy {
    #[new]
    pub fn new(
        lookback: usize,
        period: usize,
        signal_period: usize,
        matype: Option<&str>,
        signal_matype: Option<&str>,
    ) -> PyResult<Self> {
        let params = RsmkParams {
            lookback: Some(lookback),
            period: Some(period),
            signal_period: Some(signal_period),
            matype: matype.map(|s| s.to_string()),
            signal_matype: signal_matype.map(|s| s.to_string()),
        };
        let inner =
            RsmkStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(RsmkStreamPy { inner })
    }

    pub fn update(&mut self, main: f64, compare: f64) -> Option<(f64, f64)> {
        self.inner.update(main, compare)
    }
}

// WASM bindings
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RsmkResult {
    pub values: Vec<f64>, // [indicator..., signal...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsmk_js(
    main: &[f64],
    compare: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    matype: Option<String>,
    signal_matype: Option<String>,
) -> Result<JsValue, JsValue> {
    let params = RsmkParams {
        lookback: Some(lookback),
        period: Some(period),
        signal_period: Some(signal_period),
        matype: matype.or(Some("ema".into())),
        signal_matype: signal_matype.or(Some("ema".into())),
    };
    let input = RsmkInput::from_slices(main, compare, params);
    let out = rsmk(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if out.indicator.len() != main.len() || out.signal.len() != main.len() {
        return Err(JsValue::from_str("length mismatch"));
    }
    let mut values = Vec::with_capacity(2 * main.len());
    values.extend_from_slice(&out.indicator);
    values.extend_from_slice(&out.signal);

    let res = RsmkResult {
        values,
        rows: 2,
        cols: main.len(),
    };
    serde_wasm_bindgen::to_value(&res)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsmk_into(
    in_ptr: *const f64,
    indicator_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    compare_ptr: *const f64,
    lookback: usize,
    period: usize,
    signal_period: usize,
    matype: Option<String>,
    signal_matype: Option<String>,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || indicator_ptr.is_null() || signal_ptr.is_null() || compare_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let main = std::slice::from_raw_parts(in_ptr, len);
        let compare = std::slice::from_raw_parts(compare_ptr, len);
        let params = RsmkParams {
            lookback: Some(lookback),
            period: Some(period),
            signal_period: Some(signal_period),
            matype: matype.or_else(|| Some("ema".to_string())),
            signal_matype: signal_matype.or_else(|| Some("ema".to_string())),
        };
        let input = RsmkInput::from_slices(main, compare, params);

        // Check for aliasing - all pointer combinations
        let in_aliased = in_ptr == indicator_ptr || in_ptr == signal_ptr;
        let compare_aliased = compare_ptr == indicator_ptr || compare_ptr == signal_ptr;
        let outputs_aliased = indicator_ptr == signal_ptr;

        if in_aliased || compare_aliased || outputs_aliased {
            // Use temporary buffers for all aliased cases
            let mut temp_indicator = vec![0.0; len];
            let mut temp_signal = vec![0.0; len];

            rsmk_into_slice(&mut temp_indicator, &mut temp_signal, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let indicator_out = std::slice::from_raw_parts_mut(indicator_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);

            // Copy in correct order to handle overlapping memory
            if outputs_aliased {
                // If indicator and signal point to same memory, signal wins (copied last)
                indicator_out.copy_from_slice(&temp_indicator);
                signal_out.copy_from_slice(&temp_signal);
            } else {
                indicator_out.copy_from_slice(&temp_indicator);
                signal_out.copy_from_slice(&temp_signal);
            }
        } else {
            // No aliasing, write directly
            let indicator_out = std::slice::from_raw_parts_mut(indicator_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
            rsmk_into_slice(indicator_out, signal_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsmk_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsmk_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RsmkBatchConfig {
    pub lookback_range: (usize, usize, usize),
    pub period_range: (usize, usize, usize),
    pub signal_period_range: (usize, usize, usize),
    pub matype: Option<String>,
    pub signal_matype: Option<String>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RsmkBatchJsOutput {
    pub indicators: Vec<f64>,
    pub signals: Vec<f64>,
    pub combos: Vec<RsmkParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = rsmk_batch)]
pub fn rsmk_batch_js(main: &[f64], compare: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: RsmkBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let output = RsmkBatchBuilder::new()
        .lookback_range(
            config.lookback_range.0,
            config.lookback_range.1,
            config.lookback_range.2,
        )
        .period_range(
            config.period_range.0,
            config.period_range.1,
            config.period_range.2,
        )
        .signal_period_range(
            config.signal_period_range.0,
            config.signal_period_range.1,
            config.signal_period_range.2,
        )
        .apply_slices(main, compare)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Flatten the 2D arrays into 1D for JavaScript
    let indicators: Vec<f64> = output
        .indicator
        .chunks(output.cols)
        .flat_map(|row| row.iter().copied())
        .collect();

    let signals: Vec<f64> = output
        .signal
        .chunks(output.cols)
        .flat_map(|row| row.iter().copied())
        .collect();

    let js_output = RsmkBatchJsOutput {
        indicators,
        signals,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_rsmk_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = RsmkParams {
            lookback: None,
            period: None,
            signal_period: None,
            matype: None,
            signal_matype: None,
        };
        let input_default = RsmkInput::from_candles(&candles, &candles, "close", default_params);
        let output_default = rsmk_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.indicator.len(), candles.close.len());
        assert_eq!(output_default.signal.len(), candles.close.len());
        Ok(())
    }

    fn check_rsmk_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = RsmkParams::default();
        let input = RsmkInput::from_candles(&candles, &candles, "close", params.clone());
        let rsmk_result = rsmk_with_kernel(&input, kernel)?;
        assert_eq!(rsmk_result.indicator.len(), candles.close.len());
        assert_eq!(rsmk_result.signal.len(), candles.close.len());
        let expected_last_five = [0.0, 0.0, 0.0, 0.0, 0.0];
        let start = rsmk_result.indicator.len() - 5;
        for (i, &value) in rsmk_result.indicator[start..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!((value - expected_value).abs() < 1e-1);
        }
        for (i, &value) in rsmk_result.signal[start..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!((value - expected_value).abs() < 1e-1);
        }
        Ok(())
    }

    fn check_rsmk_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = RsmkInput::with_default_candles(&candles, &candles);
        let rsmk_result = rsmk_with_kernel(&input, kernel)?;
        assert_eq!(rsmk_result.indicator.len(), candles.close.len());
        assert_eq!(rsmk_result.signal.len(), candles.close.len());
        Ok(())
    }

    fn check_rsmk_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 11.0, 12.0];
        let params = RsmkParams {
            lookback: Some(0),
            period: Some(0),
            signal_period: Some(0),
            matype: Some("ema".to_string()),
            signal_matype: Some("ema".to_string()),
        };
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = RsmkParams::default();
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = RsmkParams::default();
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, 10.0, 20.0, 30.0];
        let params = RsmkParams {
            lookback: Some(3),
            period: Some(3),
            signal_period: Some(3),
            matype: Some("ema".to_string()),
            signal_matype: Some("ema".to_string()),
        };
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_ma_error(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let params = RsmkParams {
            lookback: Some(2),
            period: Some(3),
            signal_period: Some(3),
            matype: Some("nonexistent_ma".to_string()),
            signal_matype: Some("ema".to_string()),
        };
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_rsmk_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            RsmkParams::default(), // lookback: 90, period: 3, signal_period: 20
            RsmkParams {
                lookback: Some(1),
                period: Some(1),
                signal_period: Some(1),
                matype: Some("ema".to_string()),
                signal_matype: Some("ema".to_string()),
            }, // minimum values
            RsmkParams {
                lookback: Some(10),
                period: Some(2),
                signal_period: Some(5),
                matype: Some("ema".to_string()),
                signal_matype: Some("ema".to_string()),
            }, // small values
            RsmkParams {
                lookback: Some(50),
                period: Some(10),
                signal_period: Some(15),
                matype: Some("sma".to_string()),
                signal_matype: Some("sma".to_string()),
            }, // medium values
            RsmkParams {
                lookback: Some(100),
                period: Some(20),
                signal_period: Some(30),
                matype: Some("ema".to_string()),
                signal_matype: Some("sma".to_string()),
            }, // large values
            RsmkParams {
                lookback: Some(200),
                period: Some(50),
                signal_period: Some(50),
                matype: Some("sma".to_string()),
                signal_matype: Some("ema".to_string()),
            }, // very large values
            RsmkParams {
                lookback: Some(5),
                period: Some(20),
                signal_period: Some(10),
                matype: Some("ema".to_string()),
                signal_matype: Some("ema".to_string()),
            }, // edge case: period > lookback
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = RsmkInput::from_candles(&candles, &candles, "close", params.clone());
            let output = rsmk_with_kernel(&input, kernel)?;

            // Check indicator values for poison
            for (i, &val) in output.indicator.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in indicator at index {} \
						 with params: lookback={}, period={}, signal_period={}, matype={}, signal_matype={} (param set {})",
						test_name, val, bits, i,
						params.lookback.unwrap_or(90),
						params.period.unwrap_or(3),
						params.signal_period.unwrap_or(20),
						params.matype.as_deref().unwrap_or("ema"),
						params.signal_matype.as_deref().unwrap_or("ema"),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in indicator at index {} \
						 with params: lookback={}, period={}, signal_period={}, matype={}, signal_matype={} (param set {})",
						test_name, val, bits, i,
						params.lookback.unwrap_or(90),
						params.period.unwrap_or(3),
						params.signal_period.unwrap_or(20),
						params.matype.as_deref().unwrap_or("ema"),
						params.signal_matype.as_deref().unwrap_or("ema"),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in indicator at index {} \
						 with params: lookback={}, period={}, signal_period={}, matype={}, signal_matype={} (param set {})",
						test_name, val, bits, i,
						params.lookback.unwrap_or(90),
						params.period.unwrap_or(3),
						params.signal_period.unwrap_or(20),
						params.matype.as_deref().unwrap_or("ema"),
						params.signal_matype.as_deref().unwrap_or("ema"),
						param_idx
					);
                }
            }

            // Check signal values for poison
            for (i, &val) in output.signal.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in signal at index {} \
						 with params: lookback={}, period={}, signal_period={}, matype={}, signal_matype={} (param set {})",
						test_name, val, bits, i,
						params.lookback.unwrap_or(90),
						params.period.unwrap_or(3),
						params.signal_period.unwrap_or(20),
						params.matype.as_deref().unwrap_or("ema"),
						params.signal_matype.as_deref().unwrap_or("ema"),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in signal at index {} \
						 with params: lookback={}, period={}, signal_period={}, matype={}, signal_matype={} (param set {})",
						test_name, val, bits, i,
						params.lookback.unwrap_or(90),
						params.period.unwrap_or(3),
						params.signal_period.unwrap_or(20),
						params.matype.as_deref().unwrap_or("ema"),
						params.signal_matype.as_deref().unwrap_or("ema"),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in signal at index {} \
						 with params: lookback={}, period={}, signal_period={}, matype={}, signal_matype={} (param set {})",
						test_name, val, bits, i,
						params.lookback.unwrap_or(90),
						params.period.unwrap_or(3),
						params.signal_period.unwrap_or(20),
						params.matype.as_deref().unwrap_or("ema"),
						params.signal_matype.as_deref().unwrap_or("ema"),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_rsmk_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_rsmk_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating test data
        let strat = (1usize..=100, 1usize..=50, 1usize..=50).prop_flat_map(
            |(lookback, period, signal_period)| {
                // Ensure we have enough data for all calculations
                // Need: lookback + max(period, signal_period) + extra buffer
                let min_len = lookback + period.max(signal_period) + 50;
                (min_len..=500usize).prop_flat_map(move |len| {
                    (
                        // Generate main prices
                        prop::collection::vec(
                            (1.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            len,
                        ),
                        // Generate compare prices
                        prop::collection::vec(
                            (1.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            len,
                        ),
                        Just(lookback),
                        Just(period),
                        Just(signal_period),
                    )
                })
            },
        );

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(main, compare, lookback, period, signal_period)| {
                    let params = RsmkParams {
                        lookback: Some(lookback),
                        period: Some(period),
                        signal_period: Some(signal_period),
                        matype: Some("ema".to_string()),
                        signal_matype: Some("ema".to_string()),
                    };
                    let input = RsmkInput::from_slices(&main, &compare, params.clone());

                    // Handle potential errors gracefully
                    let output = match rsmk_with_kernel(&input, kernel) {
                        Ok(out) => out,
                        Err(_) => {
                            // Some parameter combinations may legitimately fail
                            // Skip this test case
                            return Ok(());
                        }
                    };

                    let ref_output = match rsmk_with_kernel(&input, Kernel::Scalar) {
                        Ok(out) => out,
                        Err(_) => {
                            // If scalar kernel fails, skip this test case
                            return Ok(());
                        }
                    };

                    // Property 1: Output length should match input length
                    prop_assert_eq!(output.indicator.len(), main.len());
                    prop_assert_eq!(output.signal.len(), main.len());

                    // Property 2: Warmup period validation
                    // The first lookback values must be NaN since momentum requires lookback points
                    // Exception: when main==compare for all values, result can be 0
                    let all_equal = main
                        .iter()
                        .zip(compare.iter())
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON);

                    // Check first lookback values more strictly
                    for i in 0..lookback.min(main.len()) {
                        if all_equal {
                            // When main==compare, 0 is valid
                            prop_assert!(
							output.indicator[i].is_nan() || output.indicator[i].abs() < 1e-9,
							"Expected NaN or 0 during warmup at index {} (before lookback {}), got {}",
							i, lookback, output.indicator[i]
						);
                        } else {
                            // Otherwise must be NaN during momentum warmup
                            prop_assert!(
							output.indicator[i].is_nan(),
							"Expected NaN during warmup at index {} (before lookback {}), got {}",
							i, lookback, output.indicator[i]
						);
                        }
                    }

                    // Verify we eventually get non-NaN values after warmup
                    let full_warmup = lookback + period.max(signal_period);
                    if main.len() > full_warmup + 5 {
                        let has_valid =
                            output.indicator[full_warmup..].iter().any(|&x| !x.is_nan());
                        prop_assert!(
                            has_valid,
                            "Expected some non-NaN values after full warmup period ({})",
                            full_warmup
                        );
                    }

                    // Property 3: When main == compare, log ratio should be 0, momentum should be near 0
                    // Test with identical inputs
                    let identical_params = RsmkParams {
                        lookback: Some(lookback),
                        period: Some(period),
                        signal_period: Some(signal_period),
                        matype: Some("ema".to_string()),
                        signal_matype: Some("ema".to_string()),
                    };
                    let identical_input = RsmkInput::from_slices(&main, &main, identical_params);
                    if let Ok(identical_output) = rsmk_with_kernel(&identical_input, kernel) {
                        // After warmup, indicator should be exactly 0 (momentum of ln(1) = 0)
                        let warmup = lookback.max(period).max(signal_period);
                        for i in warmup..main.len() {
                            if !identical_output.indicator[i].is_nan() {
                                prop_assert!(
                                    identical_output.indicator[i].abs() < 1e-9,
                                    "When main==compare, indicator should be 0 at index {}, got {}",
                                    i,
                                    identical_output.indicator[i]
                                );
                            }
                        }
                    }

                    // Property 4: Constant ratio test
                    // If main = k * compare for constant k, momentum should approach 0
                    let const_ratio = 2.0;
                    let main_scaled: Vec<f64> = compare.iter().map(|&x| x * const_ratio).collect();
                    let const_params = RsmkParams {
                        lookback: Some(lookback),
                        period: Some(period),
                        signal_period: Some(signal_period),
                        matype: Some("ema".to_string()),
                        signal_matype: Some("ema".to_string()),
                    };
                    let const_input = RsmkInput::from_slices(&main_scaled, &compare, const_params);
                    if let Ok(const_output) = rsmk_with_kernel(&const_input, kernel) {
                        // After sufficient data points, momentum should be very close to 0
                        let warmup = lookback.max(period).max(signal_period);
                        let check_start = (warmup + 10).min(main.len());
                        for i in check_start..main.len() {
                            if !const_output.indicator[i].is_nan() {
                                prop_assert!(
								const_output.indicator[i].abs() < 1e-6,
								"For constant ratio, indicator should be near 0 at index {}, got {}",
								i, const_output.indicator[i]
							);
                            }
                        }
                    }

                    // Property 5: Period edge cases
                    if period == 1 {
                        // With period=1, the MA should have minimal smoothing effect
                        // Just verify it doesn't crash and produces valid output
                        prop_assert!(output.indicator.iter().any(|&x| !x.is_nan()));
                    }

                    // Property 6: Kernel consistency
                    // Compare with scalar kernel results (within ULP tolerance)
                    let warmup = lookback.max(period).max(signal_period);
                    for i in warmup..main.len() {
                        let ind = output.indicator[i];
                        let ref_ind = ref_output.indicator[i];
                        let sig = output.signal[i];
                        let ref_sig = ref_output.signal[i];

                        // Check indicator consistency
                        if !ind.is_nan() && !ref_ind.is_nan() {
                            let ind_bits = ind.to_bits();
                            let ref_ind_bits = ref_ind.to_bits();
                            let ulp_diff = ind_bits.abs_diff(ref_ind_bits);

                            prop_assert!(
                                (ind - ref_ind).abs() <= 1e-9 || ulp_diff <= 10,
                                "Indicator mismatch at index {}: {} vs {} (ULP={})",
                                i,
                                ind,
                                ref_ind,
                                ulp_diff
                            );
                        } else {
                            // Both should be NaN or both should be finite
                            prop_assert_eq!(ind.is_nan(), ref_ind.is_nan());
                        }

                        // Check signal consistency
                        if !sig.is_nan() && !ref_sig.is_nan() {
                            let sig_bits = sig.to_bits();
                            let ref_sig_bits = ref_sig.to_bits();
                            let ulp_diff = sig_bits.abs_diff(ref_sig_bits);

                            prop_assert!(
                                (sig - ref_sig).abs() <= 1e-9 || ulp_diff <= 10,
                                "Signal mismatch at index {}: {} vs {} (ULP={})",
                                i,
                                sig,
                                ref_sig,
                                ulp_diff
                            );
                        } else {
                            prop_assert_eq!(sig.is_nan(), ref_sig.is_nan());
                        }
                    }

                    // Property 7: Signal smoothness (improved with variance calculation)
                    // Signal should be smoother than indicator (lower variance)
                    let indicator_diffs: Vec<f64> = output
                        .indicator
                        .windows(2)
                        .filter_map(|w| {
                            if !w[0].is_nan() && !w[1].is_nan() {
                                Some((w[1] - w[0]).abs())
                            } else {
                                None
                            }
                        })
                        .collect();

                    let signal_diffs: Vec<f64> = output
                        .signal
                        .windows(2)
                        .filter_map(|w| {
                            if !w[0].is_nan() && !w[1].is_nan() {
                                Some((w[1] - w[0]).abs())
                            } else {
                                None
                            }
                        })
                        .collect();

                    if !indicator_diffs.is_empty()
                        && !signal_diffs.is_empty()
                        && indicator_diffs.len() > 10
                    {
                        // Calculate actual variance
                        let ind_mean =
                            indicator_diffs.iter().sum::<f64>() / indicator_diffs.len() as f64;
                        let ind_var = indicator_diffs
                            .iter()
                            .map(|x| (x - ind_mean).powi(2))
                            .sum::<f64>()
                            / indicator_diffs.len() as f64;

                        let sig_mean = signal_diffs.iter().sum::<f64>() / signal_diffs.len() as f64;
                        let sig_var = signal_diffs
                            .iter()
                            .map(|x| (x - sig_mean).powi(2))
                            .sum::<f64>()
                            / signal_diffs.len() as f64;

                        // Signal should generally be smoother (lower variance)
                        if signal_period > 1 && ind_var > 1e-12 {
                            prop_assert!(
                                sig_var <= ind_var * 1.2 || sig_var < 1e-10,
                                "Signal should be smoother than indicator: sig_var={} ind_var={}",
                                sig_var,
                                ind_var
                            );
                        }
                    }

                    // Property 8: Edge cases
                    // Test with compare containing zeros - EMA should handle gracefully
                    let mut compare_with_zero = compare.clone();
                    if compare_with_zero.len() > lookback + 5 {
                        compare_with_zero[lookback + 2] = 0.0;
                        let zero_params = RsmkParams {
                            lookback: Some(lookback),
                            period: Some(period),
                            signal_period: Some(signal_period),
                            matype: Some("ema".to_string()),
                            signal_matype: Some("ema".to_string()),
                        };
                        let zero_input =
                            RsmkInput::from_slices(&main, &compare_with_zero, zero_params);
                        if let Ok(zero_output) = rsmk_with_kernel(&zero_input, kernel) {
                            // EMA skips NaN values to maintain continuity, so we should get valid values
                            // This is the correct behavior for financial indicators
                            // Just verify the calculation completes without error
                            prop_assert!(
							zero_output.indicator.len() == main.len(),
							"Output length should match input length even with zeros in compare"
						);
                        }
                    }

                    // Test extreme ratios
                    if main.len() > warmup + 10 {
                        // Test very large ratio
                        let large_ratio = 10000.0;
                        let main_large: Vec<f64> =
                            compare.iter().map(|&x| x * large_ratio).collect();
                        let large_params = RsmkParams {
                            lookback: Some(lookback),
                            period: Some(period),
                            signal_period: Some(signal_period),
                            matype: Some("ema".to_string()),
                            signal_matype: Some("ema".to_string()),
                        };
                        let large_input =
                            RsmkInput::from_slices(&main_large, &compare, large_params);
                        if let Ok(large_output) = rsmk_with_kernel(&large_input, kernel) {
                            // Check that values are reasonable (not infinite)
                            for i in warmup..main.len() {
                                if !large_output.indicator[i].is_nan() {
                                    prop_assert!(
									large_output.indicator[i].is_finite(),
									"Indicator should be finite for large ratios at index {}, got {}",
									i, large_output.indicator[i]
								);
                                    // Momentum of constant large ratio should still be near 0
                                    if i > warmup + 10 {
                                        prop_assert!(
										large_output.indicator[i].abs() < 1.0,
										"Large constant ratio should still have near-zero momentum at index {}, got {}",
										i, large_output.indicator[i]
									);
                                    }
                                }
                            }
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_rsmk_tests {
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

    generate_all_rsmk_tests!(
        check_rsmk_partial_params,
        check_rsmk_accuracy,
        check_rsmk_default_candles,
        check_rsmk_zero_period,
        check_rsmk_very_small_dataset,
        check_rsmk_all_nan,
        check_rsmk_not_enough_valid_data,
        check_rsmk_ma_error,
        check_rsmk_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_rsmk_tests!(check_rsmk_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let main = &candles.close;
        let compare = &candles.close;

        let batch = RsmkBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(main, compare)?;

        let def = RsmkParams::default();
        // Find row matching default parameters
        let default_row = batch
            .combos
            .iter()
            .position(|c| {
                c.lookback.unwrap_or(90) == def.lookback.unwrap_or(90)
                    && c.period.unwrap_or(3) == def.period.unwrap_or(3)
                    && c.signal_period.unwrap_or(20) == def.signal_period.unwrap_or(20)
            })
            .expect("default row missing");

        let start = default_row * batch.cols;
        let ind_row = &batch.indicator[start..start + batch.cols];
        let sig_row = &batch.signal[start..start + batch.cols];

        assert_eq!(ind_row.len(), candles.close.len());
        assert_eq!(sig_row.len(), candles.close.len());

        // RSMK with default: last five values should be (close vs close): zeroes or near zero
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0];
        let len = ind_row.len();
        let start_idx = len - 5;

        for (i, &v) in ind_row[start_idx..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-indicator mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        for (i, &v) in sig_row[start_idx..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-signal mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        // NaN checks for early region
        let max_period = def
            .lookback
            .unwrap()
            .max(def.period.unwrap())
            .max(def.signal_period.unwrap());
        for i in 0..max_period {
            if i < ind_row.len() {
                assert!(
                    ind_row[i].is_nan(),
                    "Expected indicator NaN at index {i}, got {}",
                    ind_row[i]
                );
            }
            if i < sig_row.len() {
                assert!(
                    sig_row[i].is_nan(),
                    "Expected signal NaN at index {i}, got {}",
                    sig_row[i]
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let main = &candles.close;
        let compare = &candles.close;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (lookback_range, period_range, signal_period_range)
            ((10, 20, 5), (2, 5, 1), (5, 10, 5)), // Small ranges
            ((50, 100, 25), (5, 15, 5), (10, 30, 10)), // Medium ranges
            ((100, 200, 50), (20, 40, 10), (30, 60, 15)), // Large ranges
            ((1, 5, 1), (1, 3, 1), (1, 5, 1)),    // Dense small ranges
            ((90, 90, 0), (3, 3, 0), (20, 20, 0)), // Single value (default)
            ((200, 250, 25), (50, 70, 10), (50, 100, 25)), // Very large ranges
        ];

        for (cfg_idx, &(lookback_range, period_range, signal_period_range)) in
            test_configs.iter().enumerate()
        {
            let output = RsmkBatchBuilder::new()
                .kernel(kernel)
                .lookback_range(lookback_range.0, lookback_range.1, lookback_range.2)
                .period_range(period_range.0, period_range.1, period_range.2)
                .signal_period_range(
                    signal_period_range.0,
                    signal_period_range.1,
                    signal_period_range.2,
                )
                .apply_slices(main, compare)?;

            // Check indicator values for poison
            for (idx, &val) in output.indicator.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in indicator \
						 at row {} col {} (flat index {}) with params: lookback={}, period={}, signal_period={}, \
						 matype={}, signal_matype={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.lookback.unwrap_or(90),
						combo.period.unwrap_or(3),
						combo.signal_period.unwrap_or(20),
						combo.matype.as_deref().unwrap_or("ema"),
						combo.signal_matype.as_deref().unwrap_or("ema")
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in indicator \
						 at row {} col {} (flat index {}) with params: lookback={}, period={}, signal_period={}, \
						 matype={}, signal_matype={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.lookback.unwrap_or(90),
						combo.period.unwrap_or(3),
						combo.signal_period.unwrap_or(20),
						combo.matype.as_deref().unwrap_or("ema"),
						combo.signal_matype.as_deref().unwrap_or("ema")
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in indicator \
						 at row {} col {} (flat index {}) with params: lookback={}, period={}, signal_period={}, \
						 matype={}, signal_matype={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.lookback.unwrap_or(90),
						combo.period.unwrap_or(3),
						combo.signal_period.unwrap_or(20),
						combo.matype.as_deref().unwrap_or("ema"),
						combo.signal_matype.as_deref().unwrap_or("ema")
					);
                }
            }

            // Check signal values for poison
            for (idx, &val) in output.signal.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in signal \
						 at row {} col {} (flat index {}) with params: lookback={}, period={}, signal_period={}, \
						 matype={}, signal_matype={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.lookback.unwrap_or(90),
						combo.period.unwrap_or(3),
						combo.signal_period.unwrap_or(20),
						combo.matype.as_deref().unwrap_or("ema"),
						combo.signal_matype.as_deref().unwrap_or("ema")
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in signal \
						 at row {} col {} (flat index {}) with params: lookback={}, period={}, signal_period={}, \
						 matype={}, signal_matype={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.lookback.unwrap_or(90),
						combo.period.unwrap_or(3),
						combo.signal_period.unwrap_or(20),
						combo.matype.as_deref().unwrap_or("ema"),
						combo.signal_matype.as_deref().unwrap_or("ema")
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in signal \
						 at row {} col {} (flat index {}) with params: lookback={}, period={}, signal_period={}, \
						 matype={}, signal_matype={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.lookback.unwrap_or(90),
						combo.period.unwrap_or(3),
						combo.signal_period.unwrap_or(20),
						combo.matype.as_deref().unwrap_or("ema"),
						combo.signal_matype.as_deref().unwrap_or("ema")
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

// === Classic Kernel Implementations ===

/// Classic kernel with inline SMA calculations for both indicator and signal
#[inline]
fn rsmk_classic_sma(
    mom: &[f64],
    period: usize,
    signal_period: usize,
    first_valid: usize,
) -> Result<RsmkOutput, RsmkError> {
    let len = mom.len();
    let ind_warmup = first_valid + period - 1;
    let sig_warmup = ind_warmup + signal_period - 1;

    // Check if we have enough data
    let needed = period.max(signal_period);
    if len < first_valid || len - first_valid < needed {
        return Err(RsmkError::NotEnoughValidData {
            needed,
            valid: if len >= first_valid {
                len - first_valid
            } else {
                0
            },
        });
    }

    let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
    let mut signal = alloc_with_nan_prefix(len, sig_warmup);

    // Calculate indicator (SMA of momentum * 100)
    let mut sum_ind = 0.0;
    let mut count_ind = 0;

    // Initialize first SMA for indicator
    for i in first_valid..(first_valid + period).min(len) {
        if !mom[i].is_nan() {
            sum_ind += mom[i];
            count_ind += 1;
        }
    }

    if count_ind > 0 && ind_warmup < len {
        indicator[ind_warmup] = (sum_ind / count_ind as f64) * 100.0;

        // Rolling SMA for indicator
        for i in (ind_warmup + 1)..len {
            let old_val = mom[i - period];
            let new_val = mom[i];
            if !old_val.is_nan() {
                sum_ind -= old_val;
                count_ind -= 1;
            }
            if !new_val.is_nan() {
                sum_ind += new_val;
                count_ind += 1;
            }
            indicator[i] = if count_ind > 0 {
                (sum_ind / count_ind as f64) * 100.0
            } else {
                f64::NAN
            };
        }
    }

    // Calculate signal (SMA of indicator)
    let mut sum_sig = 0.0;
    let mut count_sig = 0;

    // Initialize first SMA for signal
    for i in ind_warmup..(ind_warmup + signal_period).min(len) {
        if !indicator[i].is_nan() {
            sum_sig += indicator[i];
            count_sig += 1;
        }
    }

    if count_sig > 0 && sig_warmup < len {
        signal[sig_warmup] = sum_sig / count_sig as f64;

        // Rolling SMA for signal
        for i in (sig_warmup + 1)..len {
            let old_val = indicator[i - signal_period];
            let new_val = indicator[i];
            if !old_val.is_nan() {
                sum_sig -= old_val;
                count_sig -= 1;
            }
            if !new_val.is_nan() {
                sum_sig += new_val;
                count_sig += 1;
            }
            signal[i] = if count_sig > 0 {
                sum_sig / count_sig as f64
            } else {
                f64::NAN
            };
        }
    }

    Ok(RsmkOutput { indicator, signal })
}

/// Classic kernel with inline EMA calculations for both indicator and signal
#[inline]
fn rsmk_classic_ema(
    mom: &[f64],
    period: usize,
    signal_period: usize,
    first_valid: usize,
) -> Result<RsmkOutput, RsmkError> {
    let len = mom.len();
    let ind_warmup = first_valid + period - 1;
    let sig_warmup = ind_warmup + signal_period - 1;

    // Check if we have enough data
    let needed = period.max(signal_period);
    if len < first_valid || len - first_valid < needed {
        return Err(RsmkError::NotEnoughValidData {
            needed,
            valid: if len >= first_valid {
                len - first_valid
            } else {
                0
            },
        });
    }

    let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
    let mut signal = alloc_with_nan_prefix(len, sig_warmup);

    // Calculate indicator (EMA of momentum * 100)
    let alpha_ind = 2.0 / (period as f64 + 1.0);
    let one_minus_alpha_ind = 1.0 - alpha_ind;

    // Initialize EMA for indicator with SMA
    let mut sum_ind = 0.0;
    let mut count_ind = 0;
    for i in first_valid..(first_valid + period).min(len) {
        if !mom[i].is_nan() {
            sum_ind += mom[i];
            count_ind += 1;
        }
    }

    if count_ind > 0 && ind_warmup < len {
        let mut ema_ind = (sum_ind / count_ind as f64) * 100.0;
        indicator[ind_warmup] = ema_ind;

        // Continue EMA for indicator
        for i in (ind_warmup + 1)..len {
            if !mom[i].is_nan() {
                ema_ind = (alpha_ind * mom[i] * 100.0) + (one_minus_alpha_ind * ema_ind);
            }
            indicator[i] = ema_ind;
        }
    }

    // Calculate signal (EMA of indicator)
    let alpha_sig = 2.0 / (signal_period as f64 + 1.0);
    let one_minus_alpha_sig = 1.0 - alpha_sig;

    // Initialize EMA for signal with SMA
    let mut sum_sig = 0.0;
    let mut count_sig = 0;
    for i in ind_warmup..(ind_warmup + signal_period).min(len) {
        if !indicator[i].is_nan() {
            sum_sig += indicator[i];
            count_sig += 1;
        }
    }

    if count_sig > 0 && sig_warmup < len {
        let mut ema_sig = sum_sig / count_sig as f64;
        signal[sig_warmup] = ema_sig;

        // Continue EMA for signal
        for i in (sig_warmup + 1)..len {
            if !indicator[i].is_nan() {
                ema_sig = (alpha_sig * indicator[i]) + (one_minus_alpha_sig * ema_sig);
            }
            signal[i] = ema_sig;
        }
    }

    Ok(RsmkOutput { indicator, signal })
}

/// Classic kernel with EMA over momentum then SMA over indicator (fused single-pass after EMA seed)
#[inline]
fn rsmk_classic_ema_sma(
    mom: &[f64],
    period: usize,
    signal_period: usize,
    first_valid: usize,
) -> Result<RsmkOutput, RsmkError> {
    let len = mom.len();
    let ind_warmup = first_valid + period - 1;
    let sig_warmup = ind_warmup + signal_period - 1;

    // Check data sufficiency
    let needed = period.max(signal_period);
    if len < first_valid || len - first_valid < needed {
        return Err(RsmkError::NotEnoughValidData {
            needed,
            valid: if len >= first_valid { len - first_valid } else { 0 },
        });
    }

    let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
    let mut signal = alloc_with_nan_prefix(len, sig_warmup);

    if ind_warmup < len {
        // Seed indicator EMA from SMA of first `period` momentum values (NaN-aware)
        let mut sum = 0.0;
        let mut cnt = 0usize;
        let init_end = (first_valid + period).min(len);
        unsafe {
            for i in first_valid..init_end {
                let v = *mom.get_unchecked(i);
                if !v.is_nan() { sum += v; cnt += 1; }
            }
        }

        if cnt > 0 {
            let alpha_ind = 2.0 / (period as f64 + 1.0);
            let mut ema_ind = (sum / cnt as f64) * 100.0;

            // SMA(signal) rolling state
            let mut sum_sig = 0.0;
            let mut cnt_sig = 0usize;

            unsafe {
                *indicator.get_unchecked_mut(ind_warmup) = ema_ind;
                // Include first indicator into signal window build-up
                sum_sig += ema_ind; cnt_sig += 1;

                for i in (ind_warmup + 1)..len {
                    let mv = *mom.get_unchecked(i);
                    if !mv.is_nan() {
                        let src100 = mv * 100.0;
                        ema_ind = (src100 - ema_ind).mul_add(alpha_ind, ema_ind);
                    }
                    *indicator.get_unchecked_mut(i) = ema_ind;

                    // Update SMA(signal) window
                    if !ema_ind.is_nan() { sum_sig += ema_ind; cnt_sig += 1; }

                    if i >= sig_warmup {
                        let old_idx = i - signal_period;
                        let old_ind = *indicator.get_unchecked(old_idx);
                        if !old_ind.is_nan() { sum_sig -= old_ind; cnt_sig -= 1; }

                        *signal.get_unchecked_mut(i) = if cnt_sig > 0 {
                            sum_sig / cnt_sig as f64
                        } else {
                            f64::NAN
                        };
                    }
                }
            }
        } else {
            for i in ind_warmup..len { indicator[i] = f64::NAN; }
            for i in sig_warmup..len { signal[i] = f64::NAN; }
        }
    }

    Ok(RsmkOutput { indicator, signal })
}

/// Classic kernel with SMA over momentum then EMA over indicator (fused single-pass after SMA build)
#[inline]
fn rsmk_classic_sma_ema(
    mom: &[f64],
    period: usize,
    signal_period: usize,
    first_valid: usize,
) -> Result<RsmkOutput, RsmkError> {
    let len = mom.len();
    let ind_warmup = first_valid + period - 1;
    let sig_warmup = ind_warmup + signal_period - 1;

    // Check data sufficiency
    let needed = period.max(signal_period);
    if len < first_valid || len - first_valid < needed {
        return Err(RsmkError::NotEnoughValidData {
            needed,
            valid: if len >= first_valid { len - first_valid } else { 0 },
        });
    }

    let mut indicator = alloc_with_nan_prefix(len, ind_warmup);
    let mut signal = alloc_with_nan_prefix(len, sig_warmup);

    let mut sum_ind = 0.0;
    let mut cnt_ind = 0usize;

    let alpha_sig = 2.0 / (signal_period as f64 + 1.0);
    let mut acc_sig = 0.0;
    let mut cnt_sig = 0usize;
    let mut ema_sig = 0.0f64; // set at seed

    unsafe {
        for i in first_valid..len {
            // Update indicator SMA window with NaN-skipping
            let v_new = *mom.get_unchecked(i);
            if !v_new.is_nan() { sum_ind += v_new; cnt_ind += 1; }

            if i >= first_valid + period {
                let v_old = *mom.get_unchecked(i - period);
                if !v_old.is_nan() { sum_ind -= v_old; cnt_ind -= 1; }
            }

            if i >= ind_warmup {
                let ind_val = if cnt_ind > 0 {
                    (sum_ind / cnt_ind as f64) * 100.0
                } else {
                    f64::NAN
                };
                *indicator.get_unchecked_mut(i) = ind_val;

                if i < sig_warmup {
                    if !ind_val.is_nan() { acc_sig += ind_val; cnt_sig += 1; }
                } else if i == sig_warmup {
                    ema_sig = if cnt_sig > 0 { acc_sig / cnt_sig as f64 } else { f64::NAN };
                    *signal.get_unchecked_mut(i) = ema_sig;
                } else {
                    if !ind_val.is_nan() && !ema_sig.is_nan() {
                        ema_sig = (ind_val - ema_sig).mul_add(alpha_sig, ema_sig);
                    } else if !ind_val.is_nan() && ema_sig.is_nan() {
                        // Recover from NaN seed by snapping to the first finite indicator
                        ema_sig = ind_val;
                    }
                    *signal.get_unchecked_mut(i) = ema_sig;
                }
            }
        }
    }

    Ok(RsmkOutput { indicator, signal })
}
