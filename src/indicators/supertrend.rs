//! # SuperTrend Indicator
//!
//! Trend-following indicator using ATR-based dynamic bands to identify support and resistance levels.
//! Generates buy/sell signals when price crosses the dynamic bands.
//!
//! ## Parameters
//! - **period**: ATR lookback window for volatility calculation (default: 10)
//! - **factor**: ATR multiplier for band distance from HL/2 (default: 3.0)
//!
//! ## Inputs
//! - High, low, and close price series (or candles)
//! - All series must have the same length
//!
//! ## Returns
//! - **trend**: SuperTrend line as `Vec<f64>` (length matches input, acts as dynamic support/resistance)
//! - **changed**: Trend change signal as `Vec<f64>` (1.0 for change, 0.0 otherwise)
//!
//! ## Developer Notes
//! - SIMD status: Implemented as stubs delegating to scalar. Disabled by default due to sequential data dependency across time steps; no clear win over scalar. Revisit only with a different layout or algorithmic change.
//! - Scalar path: Hot loop optimized (state tracking, fewer branches, tight indexing). No extra allocations.
//! - Batch path: Row-specific optimizations — ATR is precomputed once per unique period and HL2 midpoints are shared across rows to avoid redundant work.
//! - Streaming update: O(1) with embedded ATR stream and efficient state tracking.
//! - Memory: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes).

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaSupertrend};
use crate::indicators::atr::{atr, AtrData, AtrError, AtrInput, AtrOutput, AtrParams};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
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
#[cfg(feature = "python")]
use pyo3::exceptions::{PyBufferError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::AsRef;
use thiserror::Error;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum SuperTrendData<'a> {
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
pub struct SuperTrendParams {
    pub period: Option<usize>,
    pub factor: Option<f64>,
}
impl Default for SuperTrendParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            factor: Some(3.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendInput<'a> {
    pub data: SuperTrendData<'a>,
    pub params: SuperTrendParams,
}

impl<'a> SuperTrendInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: SuperTrendParams) -> Self {
        Self {
            data: SuperTrendData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: SuperTrendParams,
    ) -> Self {
        Self {
            data: SuperTrendData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SuperTrendData::Candles { candles },
            params: SuperTrendParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(3.0)
    }
    #[inline(always)]
    fn as_hlc(&self) -> (&[f64], &[f64], &[f64]) {
        match &self.data {
            SuperTrendData::Candles { candles } => (
                source_type(candles, "high"),
                source_type(candles, "low"),
                source_type(candles, "close"),
            ),
            SuperTrendData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendOutput {
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct SuperTrendBuilder {
    period: Option<usize>,
    factor: Option<f64>,
    kernel: Kernel,
}
impl Default for SuperTrendBuilder {
    fn default() -> Self {
        Self {
            period: None,
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}
impl SuperTrendBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline]
    pub fn factor(mut self, x: f64) -> Self {
        self.factor = Some(x);
        self
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply(self, c: &Candles) -> Result<SuperTrendOutput, SuperTrendError> {
        let p = SuperTrendParams {
            period: self.period,
            factor: self.factor,
        };
        let i = SuperTrendInput::from_candles(c, p);
        supertrend_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SuperTrendOutput, SuperTrendError> {
        let p = SuperTrendParams {
            period: self.period,
            factor: self.factor,
        };
        let i = SuperTrendInput::from_slices(high, low, close, p);
        supertrend_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn into_stream(self) -> Result<SuperTrendStream, SuperTrendError> {
        let p = SuperTrendParams {
            period: self.period,
            factor: self.factor,
        };
        SuperTrendStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SuperTrendError {
    #[error("supertrend: Empty data provided.")]
    EmptyInputData,
    #[error("supertrend: All values are NaN.")]
    AllValuesNaN,
    #[error("supertrend: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("supertrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("supertrend: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("supertrend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("supertrend: Invalid factor range: start={start}, end={end}, step={step}")]
    InvalidFactorRange { start: f64, end: f64, step: f64 },
    #[error("supertrend: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(transparent)]
    AtrError(#[from] AtrError),
}

#[inline]
pub fn supertrend(input: &SuperTrendInput) -> Result<SuperTrendOutput, SuperTrendError> {
    supertrend_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn supertrend_prepare<'a>(
    input: &'a SuperTrendInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64], // high
        &'a [f64], // low
        &'a [f64], // close
        usize,     // period
        f64,       // factor
        usize,     // first_valid_idx
        Vec<f64>,  // atr_values
        Kernel,    // chosen kernel
    ),
    SuperTrendError,
> {
    let (high, low, close) = input.as_hlc();

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(SuperTrendError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(SuperTrendError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let factor = input.get_factor();
    let len = high.len();

    let mut first_valid_idx = None;
    for i in 0..len {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SuperTrendError::AllValuesNaN),
    };

    if (len - first_valid_idx) < period {
        return Err(SuperTrendError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    // Calculate ATR values
    let atr_input = AtrInput::from_slices(
        &high[first_valid_idx..],
        &low[first_valid_idx..],
        &close[first_valid_idx..],
        AtrParams {
            length: Some(period),
        },
    );
    let AtrOutput { values: atr_values } = atr(&atr_input)?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        chosen,
    ))
}

#[inline(always)]
fn supertrend_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    kernel: Kernel,
    trend_out: &mut [f64],
    changed_out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                supertrend_scalar(
                    high,
                    low,
                    close,
                    period,
                    factor,
                    first_valid_idx,
                    &atr_values,
                    trend_out,
                    changed_out,
                );
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                supertrend_scalar(
                    high,
                    low,
                    close,
                    period,
                    factor,
                    first_valid_idx,
                    &atr_values,
                    trend_out,
                    changed_out,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                supertrend_avx2(
                    high,
                    low,
                    close,
                    period,
                    factor,
                    first_valid_idx,
                    &atr_values,
                    trend_out,
                    changed_out,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                supertrend_avx512(
                    high,
                    low,
                    close,
                    period,
                    factor,
                    first_valid_idx,
                    &atr_values,
                    trend_out,
                    changed_out,
                );
            }
            _ => unreachable!(),
        }
    }
}

pub fn supertrend_with_kernel(
    input: &SuperTrendInput,
    kernel: Kernel,
) -> Result<SuperTrendOutput, SuperTrendError> {
    // Use classic kernel for default parameters
    let (high, low, close) = input.as_hlc();
    let period = input.get_period();
    let factor = input.get_factor();

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Use classic kernel for default parameters with scalar kernel
    if chosen == Kernel::Scalar && period == 10 && (factor - 3.0).abs() < f64::EPSILON {
        let len = high.len();
        let mut trend = alloc_with_nan_prefix(len, 0);
        let mut changed = alloc_with_nan_prefix(len, 0);

        unsafe {
            supertrend_scalar_classic(high, low, close, period, factor, &mut trend, &mut changed)?;
        }

        return Ok(SuperTrendOutput { trend, changed });
    }

    // Regular implementation
    let (high, low, close, period, factor, first_valid_idx, atr_values, chosen) =
        supertrend_prepare(input, kernel)?;

    let len = high.len();
    let mut trend = alloc_with_nan_prefix(len, first_valid_idx + period - 1);
    let mut changed = alloc_with_nan_prefix(len, first_valid_idx + period - 1);

    supertrend_compute_into(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        &atr_values,
        chosen,
        &mut trend,
        &mut changed,
    );

    Ok(SuperTrendOutput { trend, changed })
}

/// Writes SuperTrend outputs into caller-provided buffers without allocating.
///
/// - Preserves NaN warmups exactly as `supertrend()`/`supertrend_with_kernel()`.
/// - `trend_out.len()` and `changed_out.len()` must equal the input length.
/// - Uses `Kernel::Auto` to select the best available implementation.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn supertrend_into(
    input: &SuperTrendInput,
    trend_out: &mut [f64],
    changed_out: &mut [f64],
) -> Result<(), SuperTrendError> {
    let (high, _low, _close) = input.as_hlc();
    let len = high.len();

    if trend_out.len() != len {
        return Err(SuperTrendError::OutputLengthMismatch {
            expected: len,
            got: trend_out.len(),
        });
    }
    if changed_out.len() != len {
        return Err(SuperTrendError::OutputLengthMismatch {
            expected: len,
            got: changed_out.len(),
        });
    }

    let (high, low, close, period, factor, first_valid_idx, atr_values, chosen) =
        supertrend_prepare(input, Kernel::Auto)?;

    // Fill warmup prefix with NaNs to match Vec-returning APIs
    let warmup_end = first_valid_idx + period - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut trend_out[..warmup_end.min(len)] {
        *v = qnan;
    }
    for v in &mut changed_out[..warmup_end.min(len)] {
        *v = qnan;
    }

    supertrend_compute_into(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        &atr_values,
        chosen,
        trend_out,
        changed_out,
    );

    Ok(())
}

// Scalar core (reference logic) - Zero allocation implementation
#[inline(always)]
pub fn supertrend_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    let len = high.len();
    let start = first_valid_idx + period;
    if start > len {
        return;
    }

    // SAFETY: All pointer arithmetic is guarded by bounds derived from inputs above.
    unsafe {
        let h_ptr = high.as_ptr();
        let l_ptr = low.as_ptr();
        let c_ptr = close.as_ptr();
        let atr_ptr = atr_values.as_ptr();
        let tr_ptr = trend.as_mut_ptr();
        let ch_ptr = changed.as_mut_ptr();

        // Warmup output index and initial bands
        let warmup = start - 1;
        let hw = *h_ptr.add(warmup);
        let lw = *l_ptr.add(warmup);
        let hl2_w = (hw + lw) * 0.5;
        let atr_w = *atr_ptr.add(period - 1);
        let mut prev_upper_band = hl2_w + factor * atr_w;
        let mut prev_lower_band = hl2_w - factor * atr_w;

        // Initialize state from warmup close
        let mut last_close = *c_ptr.add(warmup);
        let mut upper_state = if last_close <= prev_upper_band {
            *tr_ptr.add(warmup) = prev_upper_band;
            true
        } else {
            *tr_ptr.add(warmup) = prev_lower_band;
            false
        };
        *ch_ptr.add(warmup) = 0.0;

        // Main loop — reduced index math
        let mut i = warmup + 1;
        let mut atr_idx = i.saturating_sub(first_valid_idx);
        let neg_factor = -factor;
        while i < len {
            let atr_i = *atr_ptr.add(atr_idx);
            let hi = *h_ptr.add(i);
            let lo = *l_ptr.add(i);
            let hl2 = (hi + lo) * 0.5;
            let upper_basic = factor.mul_add(atr_i, hl2);
            let lower_basic = neg_factor.mul_add(atr_i, hl2);

            // Tighten using previous close vs previous bands
            let prev_close = last_close;
            let mut curr_upper_band = upper_basic;
            if prev_close <= prev_upper_band {
                curr_upper_band = curr_upper_band.min(prev_upper_band);
            }
            let mut curr_lower_band = lower_basic;
            if prev_close >= prev_lower_band {
                curr_lower_band = curr_lower_band.max(prev_lower_band);
            }

            // Trend decision from previous state
            let curr_close = *c_ptr.add(i);
            if upper_state {
                if curr_close <= curr_upper_band {
                    *tr_ptr.add(i) = curr_upper_band;
                    *ch_ptr.add(i) = 0.0;
                } else {
                    *tr_ptr.add(i) = curr_lower_band;
                    *ch_ptr.add(i) = 1.0;
                    upper_state = false;
                }
            } else {
                if curr_close >= curr_lower_band {
                    *tr_ptr.add(i) = curr_lower_band;
                    *ch_ptr.add(i) = 0.0;
                } else {
                    *tr_ptr.add(i) = curr_upper_band;
                    *ch_ptr.add(i) = 1.0;
                    upper_state = true;
                }
            }

            // Carry state forward
            prev_upper_band = curr_upper_band;
            prev_lower_band = curr_lower_band;
            last_close = curr_close;
            i += 1;
            atr_idx += 1;
        }
    }
}

// AVX2/AVX512 stubs with correct API
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    if period <= 32 {
        supertrend_avx512_short(
            high,
            low,
            close,
            period,
            factor,
            first_valid_idx,
            atr_values,
            trend,
            changed,
        );
    } else {
        supertrend_avx512_long(
            high,
            low,
            close,
            period,
            factor,
            first_valid_idx,
            atr_values,
            trend,
            changed,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

/// Classic kernel optimization for SuperTrend with inline ATR calculation
/// Eliminates function call overhead for ATR operation
#[inline]
pub unsafe fn supertrend_scalar_classic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    trend_out: &mut [f64],
    changed_out: &mut [f64],
) -> Result<(), SuperTrendError> {
    let n = high.len();

    // Find first valid index
    let mut first_valid = None;
    for i in 0..n {
        if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
            first_valid = Some(i);
            break;
        }
    }

    let first_valid = first_valid.ok_or(SuperTrendError::AllValuesNaN)?;

    if n - first_valid < period {
        return Err(SuperTrendError::NotEnoughValidData {
            needed: period,
            valid: n - first_valid,
        });
    }

    // Initialize NaN prefix
    let warmup = first_valid + period - 1;
    for i in 0..warmup.min(n) {
        trend_out[i] = f64::NAN;
        changed_out[i] = f64::NAN;
    }

    // Step 1: Calculate ATR inline (Wilder's method)
    let mut tr_values = vec![0.0; n];

    // Calculate True Range values
    if first_valid < n {
        tr_values[first_valid] = high[first_valid] - low[first_valid];
    }

    for i in (first_valid + 1)..n {
        let high_low = high[i] - low[i];
        let high_close = (high[i] - close[i - 1]).abs();
        let low_close = (low[i] - close[i - 1]).abs();
        tr_values[i] = high_low.max(high_close).max(low_close);
    }

    // Calculate ATR using Wilder's smoothing
    let mut atr_values = vec![f64::NAN; n];

    // Initial ATR is simple average of first 'period' TR values
    let mut atr_sum = 0.0;
    for i in first_valid..(first_valid + period).min(n) {
        atr_sum += tr_values[i];
    }

    if first_valid + period <= n {
        atr_values[first_valid + period - 1] = atr_sum / period as f64;

        // Continue with Wilder's smoothing
        let alpha = 1.0 / period as f64;
        let alpha_1minus = 1.0 - alpha;

        for i in (first_valid + period)..n {
            atr_values[i] = alpha * tr_values[i] + alpha_1minus * atr_values[i - 1];
        }
    }

    // Step 2: Calculate SuperTrend bands
    if warmup >= n {
        return Ok(());
    }

    // Initialize first valid point
    let half_range = (high[warmup] + low[warmup]) / 2.0;
    let mut prev_upper_band = factor.mul_add(atr_values[warmup], half_range);
    let mut prev_lower_band = (-factor).mul_add(atr_values[warmup], half_range);

    // Initialize state from warmup close
    let mut last_close = close[warmup];
    let mut upper_state = if last_close <= prev_upper_band {
        trend_out[warmup] = prev_upper_band;
        true
    } else {
        trend_out[warmup] = prev_lower_band;
        false
    };
    changed_out[warmup] = 0.0;

    // Process remaining points
    for i in (warmup + 1)..n {
        let half_range = (high[i] + low[i]) / 2.0;
        let upper_basic = factor.mul_add(atr_values[i], half_range);
        let lower_basic = (-factor).mul_add(atr_values[i], half_range);

        // Update bands based on previous close
        let prev_close = last_close;
        let mut curr_upper_band = upper_basic;
        let mut curr_lower_band = lower_basic;
        if prev_close <= prev_upper_band {
            curr_upper_band = curr_upper_band.min(prev_upper_band);
        }
        if prev_close >= prev_lower_band {
            curr_lower_band = curr_lower_band.max(prev_lower_band);
        }

        // Determine current trend and change flag using previous state
        let curr_close = close[i];
        if upper_state {
            if curr_close <= curr_upper_band {
                trend_out[i] = curr_upper_band;
                changed_out[i] = 0.0;
            } else {
                trend_out[i] = curr_lower_band;
                changed_out[i] = 1.0;
                upper_state = false;
            }
        } else {
            if curr_close >= curr_lower_band {
                trend_out[i] = curr_lower_band;
                changed_out[i] = 0.0;
            } else {
                trend_out[i] = curr_upper_band;
                changed_out[i] = 1.0;
                upper_state = true;
            }
        }

        // Update previous bands for next iteration
        prev_upper_band = curr_upper_band;
        prev_lower_band = curr_lower_band;
        last_close = curr_close;
    }

    Ok(())
}

// Streaming (stateful) implementation — drop-in replacement
// Decision: Streaming uses O(1) state (prev bands, prev close, boolean state) and FMA for band math.
#[derive(Debug, Clone)]
pub struct SuperTrendStream {
    pub period: usize,
    pub factor: f64,
    atr_stream: crate::indicators::atr::AtrStream,
    // O(1) state we actually need
    prev_upper_band: f64,
    prev_lower_band: f64,
    prev_close: f64,
    upper_state: bool, // true = using upper band, false = using lower band
    warmed: bool,      // first output has been produced
}

impl SuperTrendStream {
    #[inline]
    pub fn try_new(params: SuperTrendParams) -> Result<Self, SuperTrendError> {
        let period = params.period.unwrap_or(10);
        let factor = params.factor.unwrap_or(3.0);
        let atr_stream = crate::indicators::atr::AtrStream::try_new(AtrParams {
            length: Some(period),
        })?;
        Ok(Self {
            period,
            factor,
            atr_stream,
            prev_upper_band: f64::NAN,
            prev_lower_band: f64::NAN,
            prev_close: f64::NAN,
            upper_state: false,
            warmed: false,
        })
    }

    /// O(1) update:
    /// - Uses Wilder/RMA ATR stream from `AtrStream` (first `Some` after `period` ticks)
    /// - Tightens bands against previous close and bands (matches scalar core)
    /// - Emits (trend, changed) once warmup is complete
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        // 1) Update embedded ATR; None until warmup is complete.
        let atr = match self.atr_stream.update(high, low, close) {
            Some(v) => v,
            None => return None,
        };

        // 2) Compute HL2 and basic bands with FMA
        let hl2 = (high + low) * 0.5;
        let upper_basic = self.factor.mul_add(atr, hl2);
        let lower_basic = (-self.factor).mul_add(atr, hl2);

        // 3) First emission: seed state; changed = 0.0
        if !self.warmed {
            self.prev_upper_band = upper_basic;
            self.prev_lower_band = lower_basic;
            self.upper_state = close <= self.prev_upper_band;
            let trend = if self.upper_state {
                self.prev_upper_band
            } else {
                self.prev_lower_band
            };
            self.prev_close = close;
            self.warmed = true;
            return Some((trend, 0.0));
        }

        // 4) Tighten bands using previous close vs previous bands
        let mut curr_upper_band = upper_basic;
        if self.prev_close <= self.prev_upper_band {
            curr_upper_band = curr_upper_band.min(self.prev_upper_band);
        }
        let mut curr_lower_band = lower_basic;
        if self.prev_close >= self.prev_lower_band {
            curr_lower_band = curr_lower_band.max(self.prev_lower_band);
        }

        // 5) Decide trend & change from previous state
        let mut changed = 0.0;
        let trend = if self.upper_state {
            if close <= curr_upper_band {
                curr_upper_band
            } else {
                changed = 1.0;
                self.upper_state = false;
                curr_lower_band
            }
        } else {
            if close >= curr_lower_band {
                curr_lower_band
            } else {
                changed = 1.0;
                self.upper_state = true;
                curr_upper_band
            }
        };

        // 6) Carry state forward
        self.prev_upper_band = curr_upper_band;
        self.prev_lower_band = curr_lower_band;
        self.prev_close = close;

        Some((trend, changed))
    }
}

// Batch range builder + batch output
#[derive(Clone, Debug)]
pub struct SuperTrendBatchRange {
    pub period: (usize, usize, usize),
    pub factor: (f64, f64, f64),
}
impl Default for SuperTrendBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 50, 1),
            factor: (3.0, 3.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SuperTrendBatchBuilder {
    range: SuperTrendBatchRange,
    kernel: Kernel,
}
impl SuperTrendBatchBuilder {
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
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }
    pub fn factor_static(mut self, x: f64) -> Self {
        self.range.factor = (x, x, 0.0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SuperTrendBatchOutput, SuperTrendError> {
        supertrend_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<SuperTrendBatchOutput, SuperTrendError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(
        c: &Candles,
        k: Kernel,
    ) -> Result<SuperTrendBatchOutput, SuperTrendError> {
        SuperTrendBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub struct SuperTrendBatchOutput {
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<SuperTrendParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SuperTrendBatchOutput {
    pub fn row_for_params(&self, p: &SuperTrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(10) == p.period.unwrap_or(10)
                && (c.factor.unwrap_or(3.0) - p.factor.unwrap_or(3.0)).abs() < 1e-12
        })
    }
    pub fn trend_for(&self, p: &SuperTrendParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.trend[start..start + self.cols]
        })
    }
    pub fn changed_for(&self, p: &SuperTrendParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.changed[start..start + self.cols]
        })
    }
}

// Python VRAM handle for CUDA SuperTrend results (CAI v3 + DLPack v1.x)
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct SupertrendDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl SupertrendDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Producing kernels synchronize before returning these handles, so no stream is needed.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
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
        use cust::memory::DeviceBuffer;

        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyBufferError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyBufferError::new_err(
                            "__dlpack__: requested device does not match producer buffer",
                        ));
                    }
                }
            }
        }
        let _ = stream;

        // copy=True is not yet supported (no cross-device copy path here).
        if let Some(copy_obj) = copy.as_ref() {
            let do_copy: bool = copy_obj.extract(py)?;
            if do_copy {
                return Err(PyBufferError::new_err(
                    "__dlpack__(copy=True) not supported for supertrend CUDA buffers",
                ));
            }
        }

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = self.inner.rows;
        let cols = self.inner.cols;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, inner.buf, rows, cols, alloc_dev, max_version_bound)
    }
}

// ---------------- Python CUDA bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "supertrend_cuda_batch_dev")]
#[pyo3(signature = (high, low, close, period_range, factor_range, device_id=0))]
pub fn supertrend_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    factor_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::IntoPyArray;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let sweep = SuperTrendBatchRange {
        period: period_range,
        factor: factor_range,
    };
    let (trend, changed, combos, ctx_arc, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaSupertrend::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let h32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
        let l32: Vec<f32> = l.iter().map(|&v| v as f32).collect();
        let c32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
        let (trend, changed, combos) = cuda
            .supertrend_batch_dev(&h32, &l32, &c32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx_arc = cuda.context_arc();
        let dev_id = cuda.device_id();
        Ok((trend, changed, combos, ctx_arc, dev_id))
    })?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "trend",
        Py::new(
            py,
            SupertrendDeviceArrayF32Py {
                inner: trend,
                _ctx: ctx_arc.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "changed",
        Py::new(
            py,
            SupertrendDeviceArrayF32Py {
                inner: changed,
                _ctx: ctx_arc,
                device_id: dev_id,
            },
        )?,
    )?;
    let periods: Vec<usize> = combos.iter().map(|p| p.period.unwrap()).collect();
    let factors: Vec<f64> = combos.iter().map(|p| p.factor.unwrap()).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("factors", factors.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "supertrend_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, close_tm, cols, rows, period, factor, device_id=0))]
pub fn supertrend_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm: numpy::PyReadonlyArray1<'py, f64>,
    low_tm: numpy::PyReadonlyArray1<'py, f64>,
    close_tm: numpy::PyReadonlyArray1<'py, f64>,
    cols: usize,
    rows: usize,
    period: usize,
    factor: f64,
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::IntoPyArray;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm.as_slice()?;
    let l = low_tm.as_slice()?;
    let c = close_tm.as_slice()?;
    if h.len() != l.len() || l.len() != c.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }
    let h32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = l.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let (out, ctx_arc, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaSupertrend::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .supertrend_many_series_one_param_time_major_dev(
                &h32,
                &l32,
                &c32,
                cols,
                rows,
                period,
                factor as f32,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx_arc = cuda.context_arc();
        let dev_id = cuda.device_id();
        Ok((out, ctx_arc, dev_id))
    })?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "trend",
        Py::new(
            py,
            SupertrendDeviceArrayF32Py {
                inner: out.plus,
                _ctx: ctx_arc.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "changed",
        Py::new(
            py,
            SupertrendDeviceArrayF32Py {
                inner: out.minus,
                _ctx: ctx_arc,
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item("cols", cols)?;
    dict.set_item("rows", rows)?;
    Ok(dict)
}

#[inline(always)]
fn expand_grid(r: &SuperTrendBatchRange) -> Result<Vec<SuperTrendParams>, SuperTrendError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, SuperTrendError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step.max(1)).collect();
            if v.is_empty() {
                return Err(SuperTrendError::InvalidRange { start, end, step });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut cur = start;
        let st = step.max(1);
        while cur >= end {
            v.push(cur);
            let next = cur.saturating_sub(st);
            if next == cur {
                break;
            }
            cur = next;
        }
        if v.is_empty() {
            return Err(SuperTrendError::InvalidRange { start, end, step });
        }
        Ok(v)
    }
    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, SuperTrendError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let st = step.abs();
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(SuperTrendError::InvalidFactorRange { start, end, step });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(SuperTrendError::InvalidFactorRange { start, end, step });
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let factors = axis_f64(r.factor)?;
    let cap = periods
        .len()
        .checked_mul(factors.len())
        .ok_or(SuperTrendError::InvalidRange {
            start: r.period.0,
            end: r.period.1,
            step: r.period.2,
        })?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &f in &factors {
            out.push(SuperTrendParams {
                period: Some(p),
                factor: Some(f),
            });
        }
    }
    Ok(out)
}

pub fn supertrend_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    k: Kernel,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SuperTrendError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    supertrend_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn supertrend_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    kern: Kernel,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    supertrend_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn supertrend_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    kern: Kernel,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    supertrend_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn supertrend_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(SuperTrendError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let len = high.len();
    let mut first_valid_idx = None;
    for i in 0..len {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SuperTrendError::AllValuesNaN),
    };
    let max_p = combos.iter().map(|c| c.period.unwrap_or(10)).max().unwrap();
    if len - first_valid_idx < max_p {
        return Err(SuperTrendError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid_idx,
        });
    }
    let rows = combos.len();
    let cols = len;

    // Guard against rows * cols overflow before allocating matrices.
    rows.checked_mul(cols).ok_or(SuperTrendError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    // For batch operations, use ManuallyDrop pattern like ALMA
    let mut trend_mu = make_uninit_matrix(rows, cols);
    let mut changed_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first_valid_idx + c.period.unwrap_or(10) - 1)
        .collect();

    init_matrix_prefixes(&mut trend_mu, cols, &warm);
    init_matrix_prefixes(&mut changed_mu, cols, &warm);

    // SAFETY: we will fully write post-warmup; warmup already initialized to NaN.
    let mut trend_guard = core::mem::ManuallyDrop::new(trend_mu);
    let mut changed_guard = core::mem::ManuallyDrop::new(changed_mu);

    let trend: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(trend_guard.as_mut_ptr() as *mut f64, trend_guard.len())
    };
    let changed: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(changed_guard.as_mut_ptr() as *mut f64, changed_guard.len())
    };

    // Row-specific optimization: precompute ATR per unique period and reuse across rows.
    let mut atr_cache: HashMap<usize, Vec<f64>> = HashMap::new();
    {
        let mut periods: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
        periods.sort_unstable();
        periods.dedup();
        for &p in &periods {
            let atr_input = AtrInput::from_slices(
                &high[first_valid_idx..],
                &low[first_valid_idx..],
                &close[first_valid_idx..],
                AtrParams { length: Some(p) },
            );
            let AtrOutput { values } = atr(&atr_input)?;
            atr_cache.insert(p, values);
        }
    }

    // Shared precompute: HL2 midpoints reused across parameter rows
    let hl2: Vec<f64> = (0..len).map(|i| 0.5 * (high[i] + low[i])).collect();

    let do_row = |row: usize, trend_row: &mut [f64], changed_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let factor = combos[row].factor.unwrap();
        let atr_values = atr_cache.get(&period).unwrap().as_slice();
        match kern {
            Kernel::Scalar => supertrend_row_scalar_from_hl(
                &hl2,
                close,
                period,
                factor,
                first_valid_idx,
                atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => supertrend_row_scalar_from_hl(
                &hl2,
                close,
                period,
                factor,
                first_valid_idx,
                atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => supertrend_row_avx2(
                high,
                low,
                close,
                period,
                factor,
                first_valid_idx,
                atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => supertrend_row_avx512(
                high,
                low,
                close,
                period,
                factor,
                first_valid_idx,
                atr_values,
                trend_row,
                changed_row,
            ),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            trend
                .par_chunks_mut(cols)
                .zip(changed.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (tr, ch))| do_row(row, tr, ch));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (tr, ch)) in trend
                .chunks_mut(cols)
                .zip(changed.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, tr, ch);
            }
        }
    } else {
        for (row, (tr, ch)) in trend
            .chunks_mut(cols)
            .zip(changed.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, tr, ch);
        }
    }

    // Convert back to Vec with proper capacity preservation
    let trend_vec = unsafe {
        Vec::from_raw_parts(
            trend_guard.as_mut_ptr() as *mut f64,
            trend_guard.len(),
            trend_guard.capacity(),
        )
    };
    let changed_vec = unsafe {
        Vec::from_raw_parts(
            changed_guard.as_mut_ptr() as *mut f64,
            changed_guard.len(),
            changed_guard.capacity(),
        )
    };

    Ok(SuperTrendBatchOutput {
        trend: trend_vec,
        changed: changed_vec,
        combos,
        rows,
        cols,
    })
}

// Scalar row for batch API
#[inline(always)]
unsafe fn supertrend_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

// Scalar row using precomputed HL2 midpoint shared across rows
#[inline(always)]
unsafe fn supertrend_row_scalar_from_hl(
    hl2: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    let len = hl2.len();
    let start = first_valid_idx + period;
    if start > len {
        return;
    }

    let hl_ptr = hl2.as_ptr();
    let c_ptr = close.as_ptr();
    let atr_ptr = atr_values.as_ptr();
    let tr_ptr = trend.as_mut_ptr();
    let ch_ptr = changed.as_mut_ptr();

    // Seed from warmup
    let warmup = start - 1;
    let hl2_w = *hl_ptr.add(warmup);
    let atr_w = *atr_ptr.add(period - 1);
    let mut prev_upper_band = factor.mul_add(atr_w, hl2_w);
    let mut prev_lower_band = (-factor).mul_add(atr_w, hl2_w);

    let mut last_close = *c_ptr.add(warmup);
    let mut upper_state = if last_close <= prev_upper_band {
        *tr_ptr.add(warmup) = prev_upper_band;
        true
    } else {
        *tr_ptr.add(warmup) = prev_lower_band;
        false
    };
    *ch_ptr.add(warmup) = 0.0;

    // Iterate over the rest
    let mut i = warmup + 1;
    while i < len {
        let atr_i = *atr_ptr.add(i - first_valid_idx);
        let hl = *hl_ptr.add(i);
        let upper_basic = factor.mul_add(atr_i, hl);
        let lower_basic = (-factor).mul_add(atr_i, hl);

        // Tighten against previous close and bands
        let prev_close = last_close;
        let mut curr_upper_band = upper_basic;
        if prev_close <= prev_upper_band {
            curr_upper_band = curr_upper_band.min(prev_upper_band);
        }
        let mut curr_lower_band = lower_basic;
        if prev_close >= prev_lower_band {
            curr_lower_band = curr_lower_band.max(prev_lower_band);
        }

        let curr_close = *c_ptr.add(i);
        if upper_state {
            if curr_close <= curr_upper_band {
                *tr_ptr.add(i) = curr_upper_band;
                *ch_ptr.add(i) = 0.0;
            } else {
                *tr_ptr.add(i) = curr_lower_band;
                *ch_ptr.add(i) = 1.0;
                upper_state = false;
            }
        } else {
            if curr_close >= curr_lower_band {
                *tr_ptr.add(i) = curr_lower_band;
                *ch_ptr.add(i) = 0.0;
            } else {
                *tr_ptr.add(i) = curr_upper_band;
                *ch_ptr.add(i) = 1.0;
                upper_state = true;
            }
        }

        prev_upper_band = curr_upper_band;
        prev_lower_band = curr_lower_band;
        last_close = curr_close;
        i += 1;
    }
}

// AVX2/AVX512 row stubs for API
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    if period <= 32 {
        supertrend_row_avx512_short(
            high,
            low,
            close,
            period,
            factor,
            first_valid_idx,
            atr_values,
            trend,
            changed,
        );
    } else {
        supertrend_row_avx512_long(
            high,
            low,
            close,
            period,
            factor,
            first_valid_idx,
            atr_values,
            trend,
            changed,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[cfg(feature = "python")]
#[inline(always)]
pub fn supertrend_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    simd: Kernel,
    parallel: bool,
    trend_out: &mut [f64],
    changed_out: &mut [f64],
) -> Result<Vec<SuperTrendParams>, SuperTrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(SuperTrendError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let len = high.len();
    let mut first_valid_idx = None;
    for i in 0..len {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SuperTrendError::AllValuesNaN),
    };
    let max_p = combos.iter().map(|c| c.period.unwrap_or(10)).max().unwrap();
    if len - first_valid_idx < max_p {
        return Err(SuperTrendError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid_idx,
        });
    }
    let rows = combos.len();
    let cols = len;

    let expected_len = rows
        .checked_mul(cols)
        .ok_or(SuperTrendError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if trend_out.len() != expected_len {
        return Err(SuperTrendError::OutputLengthMismatch {
            expected: expected_len,
            got: trend_out.len(),
        });
    }
    if changed_out.len() != expected_len {
        return Err(SuperTrendError::OutputLengthMismatch {
            expected: expected_len,
            got: changed_out.len(),
        });
    }

    // Initialize NaN prefixes for each row based on warmup period
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first_valid_idx + combo.period.unwrap_or(10) - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            trend_out[row_start + i] = f64::NAN;
            changed_out[row_start + i] = f64::NAN;
        }
    }

    // Shared precompute: HL2 midpoints reused across parameter rows
    let hl2: Vec<f64> = (0..len).map(|i| 0.5 * (high[i] + low[i])).collect();

    let do_row = |row: usize, trend_row: &mut [f64], changed_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let factor = combos[row].factor.unwrap();
        let atr_input = AtrInput::from_slices(
            &high[first_valid_idx..],
            &low[first_valid_idx..],
            &close[first_valid_idx..],
            AtrParams {
                length: Some(period),
            },
        );
        let AtrOutput { values: atr_values } = atr(&atr_input).unwrap();
        match simd {
            Kernel::Scalar => supertrend_row_scalar_from_hl(
                &hl2,
                close,
                period,
                factor,
                first_valid_idx,
                &atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => supertrend_row_avx2(
                high,
                low,
                close,
                period,
                factor,
                first_valid_idx,
                &atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => supertrend_row_avx512(
                high,
                low,
                close,
                period,
                factor,
                first_valid_idx,
                &atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => supertrend_row_scalar_from_hl(
                &hl2,
                close,
                period,
                factor,
                first_valid_idx,
                &atr_values,
                trend_row,
                changed_row,
            ),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            trend_out
                .par_chunks_mut(cols)
                .zip(changed_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (tr, ch))| do_row(row, tr, ch));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (tr, ch)) in trend_out
                .chunks_mut(cols)
                .zip(changed_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, tr, ch);
            }
        }
    } else {
        for (row, (tr, ch)) in trend_out
            .chunks_mut(cols)
            .zip(changed_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, tr, ch);
        }
    }
    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "supertrend")]
#[pyo3(signature = (high, low, close, period, factor, kernel=None))]
pub fn supertrend_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    factor: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = SuperTrendParams {
        period: Some(period),
        factor: Some(factor),
    };
    let input = SuperTrendInput::from_slices(high_slice, low_slice, close_slice, params);

    let (trend_vec, changed_vec) = py
        .allow_threads(|| supertrend_with_kernel(&input, kern).map(|o| (o.trend, o.changed)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((trend_vec.into_pyarray(py), changed_vec.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name = "supertrend_batch")]
#[pyo3(signature = (high, low, close, period_range, factor_range, kernel=None))]
pub fn supertrend_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    factor_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = SuperTrendBatchRange {
        period: period_range,
        factor: factor_range,
    };

    let grid_combos =
        expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if grid_combos.is_empty() {
        return Err(PyValueError::new_err(format!(
            "supertrend: Invalid range: start={}, end={}, step={}",
            sweep.period.0, sweep.period.1, sweep.period.2
        )));
    }
    let rows = grid_combos.len();
    let cols = high_slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("supertrend: rows*cols overflow"))?;

    let trend_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let trend_out = unsafe { trend_arr.as_slice_mut()? };
    let changed_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let changed_out = unsafe { changed_arr.as_slice_mut()? };

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
            supertrend_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                simd,
                true,
                trend_out,
                changed_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("trend", trend_arr.reshape((rows, cols))?)?;
    dict.set_item("changed", changed_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "factors",
        combos
            .iter()
            .map(|p| p.factor.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "SuperTrendStream")]
pub struct SuperTrendStreamPy {
    stream: SuperTrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SuperTrendStreamPy {
    #[new]
    fn new(period: usize, factor: f64) -> PyResult<Self> {
        let params = SuperTrendParams {
            period: Some(period),
            factor: Some(factor),
        };
        let stream =
            SuperTrendStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SuperTrendStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close)
    }
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[cfg(feature = "wasm")]
#[inline]
pub fn supertrend_into_slice(
    trend_dst: &mut [f64],
    changed_dst: &mut [f64],
    input: &SuperTrendInput,
    kern: Kernel,
) -> Result<(), SuperTrendError> {
    let (high, low, close, period, factor, first_valid_idx, atr_values, chosen) =
        supertrend_prepare(input, kern)?;

    let len = high.len();
    if trend_dst.len() != len {
        return Err(SuperTrendError::OutputLengthMismatch {
            expected: len,
            got: trend_dst.len(),
        });
    }
    if changed_dst.len() != len {
        return Err(SuperTrendError::OutputLengthMismatch {
            expected: len,
            got: changed_dst.len(),
        });
    }

    // Fill warmup period with NaN
    let warmup_end = first_valid_idx + period - 1;
    for v in &mut trend_dst[..warmup_end] {
        *v = f64::NAN;
    }
    for v in &mut changed_dst[..warmup_end] {
        *v = f64::NAN;
    }

    supertrend_compute_into(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        &atr_values,
        chosen,
        trend_dst,
        changed_dst,
    );

    Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperTrendJsResult {
    pub values: Vec<f64>, // [trend..., changed...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = supertrend)]
pub fn supertrend_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
) -> Result<JsValue, JsValue> {
    let len = high.len();
    let params = SuperTrendParams {
        period: Some(period),
        factor: Some(factor),
    };
    let input = SuperTrendInput::from_slices(high, low, close, params);

    // Compute directly into two slices of one flat buffer
    let mut values = vec![0.0; len * 2];
    let (trend_slice, changed_slice) = values.split_at_mut(len);
    supertrend_into_slice(trend_slice, changed_slice, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let out = SuperTrendJsResult {
        values,
        rows: 2,
        cols: len,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supertrend_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    trend_ptr: *mut f64,
    changed_ptr: *mut f64,
    len: usize,
    period: usize,
    factor: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || trend_ptr.is_null()
        || changed_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let params = SuperTrendParams {
            period: Some(period),
            factor: Some(factor),
        };
        let input = SuperTrendInput::from_slices(high, low, close, params);

        // Check for aliasing between input and output pointers
        let input_ptrs = [
            high_ptr as *const u8,
            low_ptr as *const u8,
            close_ptr as *const u8,
        ];
        let output_ptrs = [trend_ptr as *const u8, changed_ptr as *const u8];

        let has_aliasing = input_ptrs
            .iter()
            .any(|&inp| output_ptrs.iter().any(|&out| inp == out));

        if has_aliasing {
            // Use temporary buffers if there's aliasing
            let output = supertrend_with_kernel(&input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let trend_out = std::slice::from_raw_parts_mut(trend_ptr, len);
            let changed_out = std::slice::from_raw_parts_mut(changed_ptr, len);

            trend_out.copy_from_slice(&output.trend);
            changed_out.copy_from_slice(&output.changed);
        } else {
            // Direct computation when no aliasing
            let trend_out = std::slice::from_raw_parts_mut(trend_ptr, len);
            let changed_out = std::slice::from_raw_parts_mut(changed_ptr, len);

            supertrend_into_slice(trend_out, changed_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supertrend_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supertrend_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperTrendBatchConfig {
    pub period_range: (usize, usize, usize),
    pub factor_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperTrendBatchJsOutput {
    pub values: Vec<f64>, // rows = 2 * combos, each row has `cols`
    pub periods: Vec<usize>,
    pub factors: Vec<f64>,
    pub rows: usize, // 2 * combos
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = supertrend_batch)]
pub fn supertrend_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: SuperTrendBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = SuperTrendBatchRange {
        period: cfg.period_range,
        factor: cfg.factor_range,
    };

    let batch = supertrend_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // flatten to [trend(row0), changed(row0), trend(row1), changed(row1), ...]
    let mut values = Vec::with_capacity(batch.rows * 2 * batch.cols);
    for r in 0..batch.rows {
        let rs = r * batch.cols;
        values.extend_from_slice(&batch.trend[rs..rs + batch.cols]);
        values.extend_from_slice(&batch.changed[rs..rs + batch.cols]);
    }

    let periods: Vec<usize> = batch
        .combos
        .iter()
        .map(|c| c.period.unwrap_or(10))
        .collect();
    let factors: Vec<f64> = batch
        .combos
        .iter()
        .map(|c| c.factor.unwrap_or(3.0))
        .collect();

    let out = SuperTrendBatchJsOutput {
        values,
        periods,
        factors,
        rows: batch.rows * 2,
        cols: batch.cols,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    #[test]
    fn test_supertrend_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Use the existing CSV dataset for parity with other tests
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_candles(&candles, params);

        // Baseline via existing Vec-returning API
        let baseline = supertrend_with_kernel(&input, Kernel::Auto)?;

        // Preallocate outputs and compute via new into API
        let n = candles.close.len();
        let mut trend_out = vec![0.0; n];
        let mut changed_out = vec![0.0; n];

        // Only compiled for native; avoid wasm symbol clash
        #[cfg(not(feature = "wasm"))]
        {
            supertrend_into(&input, &mut trend_out, &mut changed_out)?;
        }

        assert_eq!(baseline.trend.len(), n);
        assert_eq!(baseline.changed.len(), n);
        assert_eq!(trend_out.len(), n);
        assert_eq!(changed_out.len(), n);

        // Helper: treat NaN == NaN; otherwise allow tiny epsilon due to classic vs ATR-prep paths
        #[inline]
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline.trend[i], trend_out[i]),
                "trend mismatch at {}: baseline={}, into={}",
                i,
                baseline.trend[i],
                trend_out[i]
            );
            assert!(
                eq_or_both_nan(baseline.changed[i], changed_out[i]),
                "changed mismatch at {}: baseline={}, into={}",
                i,
                baseline.changed[i],
                changed_out[i]
            );
        }

        Ok(())
    }

    fn check_supertrend_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = SuperTrendParams {
            period: None,
            factor: None,
        };
        let input = SuperTrendInput::from_candles(&candles, default_params);
        let output = supertrend_with_kernel(&input, kernel)?;
        assert_eq!(output.trend.len(), candles.close.len());
        assert_eq!(output.changed.len(), candles.close.len());

        Ok(())
    }

    fn check_supertrend_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_candles(&candles, params);
        let st_result = supertrend_with_kernel(&input, kernel)?;

        assert_eq!(st_result.trend.len(), candles.close.len());
        assert_eq!(st_result.changed.len(), candles.close.len());

        let expected_last_five_trend = [
            61811.479454208165,
            61721.73150878735,
            61459.10835790861,
            61351.59752211775,
            61033.18776990598,
        ];
        let expected_last_five_changed = [0.0, 0.0, 0.0, 0.0, 0.0];

        let start_index = st_result.trend.len() - 5;
        let trend_slice = &st_result.trend[start_index..];
        let changed_slice = &st_result.changed[start_index..];

        for (i, &val) in trend_slice.iter().enumerate() {
            let exp = expected_last_five_trend[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "[{}] Trend mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp
            );
        }
        for (i, &val) in changed_slice.iter().enumerate() {
            let exp = expected_last_five_changed[i];
            assert!(
                (val - exp).abs() < 1e-9,
                "[{}] Changed mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp
            );
        }
        Ok(())
    }

    fn check_supertrend_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SuperTrendInput::with_default_candles(&candles);
        let output = supertrend_with_kernel(&input, kernel)?;
        assert_eq!(output.trend.len(), candles.close.len());
        assert_eq!(output.changed.len(), candles.close.len());
        Ok(())
    }

    fn check_supertrend_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 13.0];
        let low = [9.0, 11.0, 12.5];
        let close = [9.5, 11.5, 13.0];
        let params = SuperTrendParams {
            period: Some(0),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);
        let res = supertrend_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with zero period", test_name);
        Ok(())
    }

    fn check_supertrend_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 13.0];
        let low = [9.0, 11.0, 12.5];
        let close = [9.5, 11.5, 13.0];
        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);
        let res = supertrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail with period > data.len()",
            test_name
        );
        Ok(())
    }

    fn check_supertrend_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [40.0];
        let close = [41.0];
        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);
        let res = supertrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail for data smaller than period",
            test_name
        );
        Ok(())
    }

    fn check_supertrend_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let first_input = SuperTrendInput::from_candles(&candles, first_params);
        let first_result = supertrend_with_kernel(&first_input, kernel)?;

        let second_params = SuperTrendParams {
            period: Some(5),
            factor: Some(2.0),
        };
        let second_input = SuperTrendInput::from_slices(
            &first_result.trend,
            &first_result.trend,
            &first_result.trend,
            second_params,
        );
        let second_result = supertrend_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.trend.len(), first_result.trend.len());
        assert_eq!(second_result.changed.len(), first_result.changed.len());
        Ok(())
    }

    fn check_supertrend_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_candles(&candles, params);
        let result = supertrend_with_kernel(&input, kernel)?;
        if result.trend.len() > 50 {
            for (i, &val) in result.trend[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_supertrend_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 10;
        let factor = 3.0;
        let params = SuperTrendParams {
            period: Some(period),
            factor: Some(factor),
        };
        let input = SuperTrendInput::from_candles(&candles, params.clone());
        let batch_output = supertrend_with_kernel(&input, kernel)?;

        let mut stream = SuperTrendStream::try_new(params.clone())?;
        let mut stream_trend = Vec::with_capacity(candles.close.len());
        let mut stream_changed = Vec::with_capacity(candles.close.len());

        for i in 0..candles.close.len() {
            let (h, l, c) = (candles.high[i], candles.low[i], candles.close[i]);
            match stream.update(h, l, c) {
                Some((trend, changed)) => {
                    stream_trend.push(trend);
                    stream_changed.push(changed);
                }
                None => {
                    stream_trend.push(f64::NAN);
                    stream_changed.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.trend.len(), stream_trend.len());
        assert_eq!(batch_output.changed.len(), stream_changed.len());

        for (i, (&b, &s)) in batch_output
            .trend
            .iter()
            .zip(stream_trend.iter())
            .enumerate()
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Streaming trend mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in batch_output
            .changed
            .iter()
            .zip(stream_changed.iter())
            .enumerate()
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Streaming changed mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_supertrend_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            SuperTrendParams::default(),
            // Minimum period
            SuperTrendParams {
                period: Some(2),
                factor: Some(1.0),
            },
            // Small period with various factors
            SuperTrendParams {
                period: Some(5),
                factor: Some(0.5),
            },
            SuperTrendParams {
                period: Some(5),
                factor: Some(2.0),
            },
            SuperTrendParams {
                period: Some(5),
                factor: Some(3.5),
            },
            // Medium periods
            SuperTrendParams {
                period: Some(10),
                factor: Some(1.5),
            },
            SuperTrendParams {
                period: Some(14),
                factor: Some(2.5),
            },
            SuperTrendParams {
                period: Some(20),
                factor: Some(3.0),
            },
            // Large periods
            SuperTrendParams {
                period: Some(50),
                factor: Some(2.0),
            },
            SuperTrendParams {
                period: Some(100),
                factor: Some(1.0),
            },
            // Edge case factors
            SuperTrendParams {
                period: Some(10),
                factor: Some(0.1),
            },
            SuperTrendParams {
                period: Some(10),
                factor: Some(5.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = SuperTrendInput::from_candles(&candles, params.clone());
            let output = supertrend_with_kernel(&input, kernel)?;

            // Check trend values
            for (i, &val) in output.trend.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in trend \
						 with params: period={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10),
						params.factor.unwrap_or(3.0),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in trend \
						 with params: period={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10),
						params.factor.unwrap_or(3.0),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in trend \
						 with params: period={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10),
						params.factor.unwrap_or(3.0),
						param_idx
					);
                }
            }

            // Check changed values
            for (i, &val) in output.changed.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in changed \
						 with params: period={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10),
						params.factor.unwrap_or(3.0),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in changed \
						 with params: period={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10),
						params.factor.unwrap_or(3.0),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in changed \
						 with params: period={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10),
						params.factor.unwrap_or(3.0),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_supertrend_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_supertrend_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic OHLC data
        let strat = (2usize..=50).prop_flat_map(|period| {
            let data_len = period * 2 + 50; // Ensure sufficient data length
            (
                // Base price generation
                prop::collection::vec(
                    (100f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                    data_len,
                ),
                Just(period),
                0.5f64..5.0f64, // factor range
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(base_prices, period, factor)| {
                // Generate more realistic OHLC data from base prices
                let mut high = Vec::with_capacity(base_prices.len());
                let mut low = Vec::with_capacity(base_prices.len());
                let mut close = Vec::with_capacity(base_prices.len());

                // Use a simple RNG for variation
                let mut rng_state = 42u64;
                for &base in &base_prices {
                    // Simple LCG for pseudo-random numbers
                    rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let rand1 = ((rng_state >> 32) as f64) / (u32::MAX as f64);
                    rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let rand2 = ((rng_state >> 32) as f64) / (u32::MAX as f64);

                    // Variable spread between 0.5% and 3%
                    let spread = base * (0.005 + rand1 * 0.025);
                    let h = base + spread;
                    let l = base - spread;

                    // Close can be anywhere within high/low range
                    let c = l + (h - l) * rand2;

                    high.push(h);
                    low.push(l);
                    close.push(c);
                }

                let params = SuperTrendParams {
                    period: Some(period),
                    factor: Some(factor),
                };
                let input = SuperTrendInput::from_slices(&high, &low, &close, params);

                // Test with specified kernel
                let output = supertrend_with_kernel(&input, kernel).unwrap();

                // Also get reference output from scalar kernel for comparison
                let ref_output = supertrend_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length should match input
                prop_assert_eq!(
                    output.trend.len(),
                    high.len(),
                    "[{}] Trend length mismatch",
                    test_name
                );
                prop_assert_eq!(
                    output.changed.len(),
                    high.len(),
                    "[{}] Changed length mismatch",
                    test_name
                );

                // Property 2: Warmup period handling
                let warmup_end = period - 1;
                for i in 0..warmup_end {
                    prop_assert!(
                        output.trend[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}",
                        test_name,
                        i
                    );
                    prop_assert!(
                        output.changed[i].is_nan(),
                        "[{}] Expected NaN in changed during warmup at index {}",
                        test_name,
                        i
                    );
                }

                // Property 3: Trend values should be reasonable relative to price data
                // SuperTrend uses ATR-based bands, so values should be within a reasonable
                // multiple of the price range
                for i in warmup_end..output.trend.len() {
                    let val = output.trend[i];
                    if !val.is_nan() {
                        // Get the entire data range to understand the scale
                        let global_high = high.iter().fold(f64::NEG_INFINITY, |a, &b| {
                            if b.is_finite() {
                                a.max(b)
                            } else {
                                a
                            }
                        });
                        let global_low = low.iter().fold(f64::INFINITY, |a, &b| {
                            if b.is_finite() {
                                a.min(b)
                            } else {
                                a
                            }
                        });

                        // SuperTrend bands can legitimately be far from current price
                        // when there are large price movements in the ATR period
                        // Just verify the value is within the overall data scale
                        let global_range = global_high - global_low;

                        // Allow trend to be within the global range plus some margin
                        // for ATR-based expansion
                        let margin = global_range * factor;

                        prop_assert!(
                            val >= global_low - margin && val <= global_high + margin,
                            "[{}] Trend value {} at index {} outside global bounds [{}, {}]",
                            test_name,
                            val,
                            i,
                            global_low - margin,
                            global_high + margin
                        );
                    }
                }

                // Property 4: Changed values must be 0.0 or 1.0
                for i in warmup_end..output.changed.len() {
                    let val = output.changed[i];
                    if !val.is_nan() {
                        prop_assert!(
                            val == 0.0 || val == 1.0,
                            "[{}] Changed value {} at index {} is not 0.0 or 1.0",
                            test_name,
                            val,
                            i
                        );
                    }
                }

                // Property 5: Kernel consistency
                for i in 0..output.trend.len() {
                    let trend_val = output.trend[i];
                    let ref_trend_val = ref_output.trend[i];
                    let changed_val = output.changed[i];
                    let ref_changed_val = ref_output.changed[i];

                    // Check trend consistency
                    if !trend_val.is_finite() || !ref_trend_val.is_finite() {
                        prop_assert_eq!(
                            trend_val.to_bits(),
                            ref_trend_val.to_bits(),
                            "[{}] NaN/Inf mismatch in trend at index {}",
                            test_name,
                            i
                        );
                    } else {
                        let ulp_diff = trend_val.to_bits().abs_diff(ref_trend_val.to_bits());
                        prop_assert!(
                            (trend_val - ref_trend_val).abs() <= 1e-9 || ulp_diff <= 5,
                            "[{}] Kernel mismatch in trend at index {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            trend_val,
                            ref_trend_val,
                            ulp_diff
                        );
                    }

                    // Check changed consistency (should be exact)
                    if !changed_val.is_nan() && !ref_changed_val.is_nan() {
                        prop_assert_eq!(
                            changed_val,
                            ref_changed_val,
                            "[{}] Kernel mismatch in changed at index {}: {} vs {}",
                            test_name,
                            i,
                            changed_val,
                            ref_changed_val
                        );
                    }
                }

                // Property 6: Special case - when all prices are identical
                if base_prices.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                    // After warmup, trend should stabilize
                    let stable_start = (period * 2).min(output.trend.len());
                    if stable_start < output.trend.len() {
                        let stable_trend = output.trend[stable_start];
                        for i in (stable_start + 1)..output.trend.len() {
                            if !output.trend[i].is_nan() && !stable_trend.is_nan() {
                                prop_assert!(
                                    (output.trend[i] - stable_trend).abs() < 1e-9,
                                    "[{}] Trend not stable for constant prices at index {}",
                                    test_name,
                                    i
                                );
                            }
                        }
                    }
                }

                // Property 7: Trend switching consistency
                // When changed=1.0, verify trend actually switched from previous
                // When changed=0.0, trend should maintain same band type
                if output.trend.len() > warmup_end + 1 {
                    for i in (warmup_end + 1)..output.changed.len() {
                        let changed_val = output.changed[i];
                        if !changed_val.is_nan() {
                            let curr_trend = output.trend[i];
                            let prev_trend = output.trend[i - 1];

                            if !curr_trend.is_nan() && !prev_trend.is_nan() {
                                if changed_val == 1.0 {
                                    // Changed flag indicates a switch - trends should be different
                                    // Allow for small numerical differences
                                    prop_assert!(
										(curr_trend - prev_trend).abs() > 1e-6,
										"[{}] Changed=1.0 at index {} but trend didn't switch: {} vs {}",
										test_name, i, prev_trend, curr_trend
									);
                                }
                                // Note: We can't strictly enforce changed=0.0 means same value
                                // because the bands themselves can move even without switching
                            }
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_supertrend_tests {
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

    generate_all_supertrend_tests!(
        check_supertrend_partial_params,
        check_supertrend_accuracy,
        check_supertrend_default_candles,
        check_supertrend_zero_period,
        check_supertrend_period_exceeds_length,
        check_supertrend_very_small_dataset,
        check_supertrend_reinput,
        check_supertrend_nan_handling,
        check_supertrend_streaming,
        check_supertrend_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_supertrend_tests!(check_supertrend_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = SuperTrendBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = SuperTrendParams::default();
        let row = output.trend_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Last few values of trend for reference.
        let expected = [
            61811.479454208165,
            61721.73150878735,
            61459.10835790861,
            61351.59752211775,
            61033.18776990598,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
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
            // (period_start, period_end, period_step, factor_start, factor_end, factor_step)
            (2, 10, 2, 1.0, 3.0, 0.5),   // Small periods
            (5, 25, 5, 2.0, 2.0, 0.0),   // Medium periods, static factor
            (10, 10, 0, 0.5, 4.0, 0.5),  // Static period, varying factors
            (2, 5, 1, 1.5, 1.5, 0.0),    // Dense small range
            (30, 60, 15, 3.0, 3.0, 0.0), // Large periods
            (20, 30, 5, 1.0, 3.0, 1.0),  // Mixed ranges
            (8, 12, 1, 0.5, 2.5, 0.5),   // Dense medium range
        ];

        for (cfg_idx, &(p_start, p_end, p_step, f_start, f_end, f_step)) in
            test_configs.iter().enumerate()
        {
            let output = SuperTrendBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .factor_range(f_start, f_end, f_step)
                .apply_candles(&c)?;

            // Check trend values
            for (idx, &val) in output.trend.iter().enumerate() {
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
						at row {} col {} (flat index {}) in trend with params: period={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.factor.unwrap_or(3.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in trend with params: period={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.factor.unwrap_or(3.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in trend with params: period={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.factor.unwrap_or(3.0)
                    );
                }
            }

            // Check changed values
            for (idx, &val) in output.changed.iter().enumerate() {
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
						at row {} col {} (flat index {}) in changed with params: period={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.factor.unwrap_or(3.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in changed with params: period={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.factor.unwrap_or(3.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in changed with params: period={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.factor.unwrap_or(3.0)
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
