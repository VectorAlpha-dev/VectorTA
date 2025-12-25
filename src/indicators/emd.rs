//! # Empirical Mode Decomposition (EMD)
//!
//! Decision log:
//! - Scalar path optimized: single tight loop, constant-time ring updates, cached prior prices; ~30–55% faster at 100k.
//! - Single-series SIMD kept as stubs: second-order IIR recursion is loop-carried, so AVX2/AVX512 across time is not beneficial.
//! - Batch path: precompute mid-price once and reuse across rows; cross-row SIMD deferred; runtime selection remains scalar.
//! - CUDA path: batch and many-series kernels enabled; PTX wrapper matches scalar within f32 tolerance and exposes Python interop via CUDA Array Interface v3 + DLPack v1.x using the shared helpers in `utilities::dlpack_cuda`.
//!
//! Implements the Empirical Mode Decomposition indicator with band-pass filtering and moving averages,
//! yielding upperband, middleband, and lowerband outputs.
//!
//! ## Parameters
//! - **period**: Window for the band-pass filter (default: 20)
//! - **delta**: Band-pass phase parameter (default: 0.5)
//! - **fraction**: Peak/valley scaling factor (default: 0.1)
//!
//! ## Inputs
//! - **high**: High prices
//! - **low**: Low prices
//! - **close**: Close prices (primary input for filtering)
//! - **volume**: Volume data (used in some calculations)
//!
//! ## Returns
//! - **`Ok(EmdOutput)`** containing:
//!   - `upperband`: Upper band values (Vec<f64>)
//!   - `middleband`: Middle band values (Vec<f64>)
//!   - `lowerband`: Lower band values (Vec<f64>)
//! - Output lengths match input data length with NaN padding for warmup period
//!
//! ## Developer Notes
//! - **AVX2 kernel**: STUB - calls scalar implementation (lines 459-469)
//! - **AVX512 kernel**: STUB - calls scalar implementation (lines 491-501, 505-514)
//! - **Streaming**: Implemented O(1) ring updates; no modulo in hot path; precomputed reciprocals and coefficients.
//! - **Memory optimization**: ✅ Uses alloc_with_nan_prefix (zero-copy) for all three outputs (lines 339-341, 444-446)
//! - **Batch operations**: ✅ Implemented with parallel processing support

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaEmd, CudaEmdBatchResult, DeviceArrayF32Triple};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EmdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EmdData::Candles { candles } => source_type(candles, "close"),
            EmdData::Slices { close, .. } => close,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmdData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EmdOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EmdParams {
    pub period: Option<usize>,
    pub delta: Option<f64>,
    pub fraction: Option<f64>,
}

impl Default for EmdParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            delta: Some(0.5),
            fraction: Some(0.1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmdInput<'a> {
    pub data: EmdData<'a>,
    pub params: EmdParams,
}

impl<'a> EmdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: EmdParams) -> Self {
        Self {
            data: EmdData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: EmdParams,
    ) -> Self {
        Self {
            data: EmdData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, EmdParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_delta(&self) -> f64 {
        self.params.delta.unwrap_or(0.5)
    }
    #[inline]
    pub fn get_fraction(&self) -> f64 {
        self.params.fraction.unwrap_or(0.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EmdBuilder {
    period: Option<usize>,
    delta: Option<f64>,
    fraction: Option<f64>,
    kernel: Kernel,
}

impl Default for EmdBuilder {
    fn default() -> Self {
        Self {
            period: None,
            delta: None,
            fraction: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EmdBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn delta(mut self, d: f64) -> Self {
        self.delta = Some(d);
        self
    }
    #[inline(always)]
    pub fn fraction(mut self, f: f64) -> Self {
        self.fraction = Some(f);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EmdOutput, EmdError> {
        let p = EmdParams {
            period: self.period,
            delta: self.delta,
            fraction: self.fraction,
        };
        let i = EmdInput::from_candles(c, p);
        emd_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmdOutput, EmdError> {
        let p = EmdParams {
            period: self.period,
            delta: self.delta,
            fraction: self.fraction,
        };
        let i = EmdInput::from_slices(high, low, close, volume, p);
        emd_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EmdStream, EmdError> {
        let p = EmdParams {
            period: self.period,
            delta: self.delta,
            fraction: self.fraction,
        };
        EmdStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EmdError {
    #[error("emd: Invalid input length (empty input data)")]
    EmptyInputData,

    #[error("emd: All values are NaN.")]
    AllValuesNaN,

    #[error("emd: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("emd: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("emd: Invalid delta: {delta}")]
    InvalidDelta { delta: f64 },

    #[error("emd: Invalid fraction: {fraction}")]
    InvalidFraction { fraction: f64 },

    #[error("emd: Invalid input length: expected = {expected}, actual = {actual}")]
    InvalidInputLength { expected: usize, actual: usize },

    #[error("emd: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    // Batch/domain errors
    #[error("emd: Invalid range (usize): start={start}, end={end}, step={step}")]
    InvalidRangeU { start: usize, end: usize, step: usize },

    #[error("emd: Invalid range (float): start={start}, end={end}, step={step}")]
    InvalidRangeF { start: f64, end: f64, step: f64 },

    #[error("emd: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn emd(input: &EmdInput) -> Result<EmdOutput, EmdError> {
    emd_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn emd_into_slices(
    ub: &mut [f64],
    mb: &mut [f64],
    lb: &mut [f64],
    input: &EmdInput,
    kernel: Kernel,
) -> Result<(), EmdError> {
    let (high, low, period, delta, fraction, first, chosen) = emd_prepare(input, kernel)?;
    if ub.len() != high.len() || mb.len() != high.len() || lb.len() != high.len() {
        return Err(EmdError::OutputLengthMismatch {
            expected: high.len(),
            got: ub.len().min(mb.len()).min(lb.len()),
        });
    }

    emd_compute_into(
        high, low, period, delta, fraction, first, chosen, ub, mb, lb,
    );

    let up_low_warm = first + 50 - 1;
    let mid_warm = first + 2 * period - 1;
    let ub_len = ub.len();
    let lb_len = lb.len();
    let mb_len = mb.len();
    for v in &mut ub[..up_low_warm.min(ub_len)] {
        *v = f64::NAN;
    }
    for v in &mut lb[..up_low_warm.min(lb_len)] {
        *v = f64::NAN;
    }
    for v in &mut mb[..mid_warm.min(mb_len)] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
fn emd_prepare<'a>(
    input: &'a EmdInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, f64, f64, usize, Kernel), EmdError> {
    let (high, low) = match &input.data {
        EmdData::Candles { candles } => (source_type(candles, "high"), source_type(candles, "low")),
        EmdData::Slices { high, low, .. } => (*high, *low),
    };

    let len = high.len();
    if len == 0 {
        return Err(EmdError::EmptyInputData);
    }
    if low.len() != len {
        return Err(EmdError::InvalidInputLength { expected: len, actual: low.len() });
    }

    let period = input.get_period();
    let delta = input.get_delta();
    let fraction = input.get_fraction();

    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(EmdError::AllValuesNaN)?;

    if period == 0 || period > len {
        return Err(EmdError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let needed = (2 * period).max(50);
    if len - first < needed {
        return Err(EmdError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }
    if delta.is_nan() || delta.is_infinite() {
        return Err(EmdError::InvalidDelta { delta });
    }
    if fraction.is_nan() || fraction.is_infinite() {
        return Err(EmdError::InvalidFraction { fraction });
    }

    let chosen = match kernel {
        // SIMD kernels are stubs that delegate to scalar; avoid dispatch overhead.
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    Ok((high, low, period, delta, fraction, first, chosen))
}

#[inline(always)]
fn emd_compute_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    kernel: Kernel,
    ub: &mut [f64],
    mb: &mut [f64],
    lb: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                emd_scalar_into(high, low, period, delta, fraction, first, ub, mb, lb)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                emd_scalar_into(high, low, period, delta, fraction, first, ub, mb, lb)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                emd_scalar_into(high, low, period, delta, fraction, first, ub, mb, lb)
            }
            _ => unreachable!(),
        }
    }
}

pub fn emd_with_kernel(input: &EmdInput, kernel: Kernel) -> Result<EmdOutput, EmdError> {
    let (high, low, period, delta, fraction, first, chosen) = emd_prepare(input, kernel)?;
    let len = high.len();
    let up_low_warm = first + 50 - 1;
    let mid_warm = first + 2 * period - 1;

    let mut upperband = alloc_with_nan_prefix(len, up_low_warm);
    let mut middleband = alloc_with_nan_prefix(len, mid_warm);
    let mut lowerband = alloc_with_nan_prefix(len, up_low_warm);

    emd_compute_into(
        high,
        low,
        period,
        delta,
        fraction,
        first,
        chosen,
        &mut upperband,
        &mut middleband,
        &mut lowerband,
    );

    Ok(EmdOutput {
        upperband,
        middleband,
        lowerband,
    })
}

/// Compute EMD into caller-provided buffers (no allocations).
/// Preserves NaN warmups exactly like the Vec-returning API and writes results in-place.
/// All output slices must have length equal to the input length.
#[cfg(not(feature = "wasm"))]
pub fn emd_into(
    input: &EmdInput,
    upperband_out: &mut [f64],
    middleband_out: &mut [f64],
    lowerband_out: &mut [f64],
) -> Result<(), EmdError> {
    let (high, low, period, delta, fraction, first, chosen) = emd_prepare(input, Kernel::Auto)?;

    if upperband_out.len() != high.len()
        || middleband_out.len() != high.len()
        || lowerband_out.len() != high.len()
    {
        return Err(EmdError::OutputLengthMismatch {
            expected: high.len(),
            got: upperband_out
                .len()
                .min(middleband_out.len())
                .min(lowerband_out.len()),
        });
    }

    // Prefill warmup prefixes with the same quiet-NaN pattern used by alloc_with_nan_prefix
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let up_low_warm = first + 50 - 1;
    let mid_warm = first + 2 * period - 1;

    let end_u = up_low_warm.min(upperband_out.len());
    for v in &mut upperband_out[..end_u] {
        *v = qnan;
    }
    let end_l = up_low_warm.min(lowerband_out.len());
    for v in &mut lowerband_out[..end_l] {
        *v = qnan;
    }
    let end_m = mid_warm.min(middleband_out.len());
    for v in &mut middleband_out[..end_m] {
        *v = qnan;
    }

    // Compute into provided buffers using the selected kernel
    emd_compute_into(
        high,
        low,
        period,
        delta,
        fraction,
        first,
        chosen,
        upperband_out,
        middleband_out,
        lowerband_out,
    );

    Ok(())
}

#[inline]
pub unsafe fn emd_scalar_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    ub: &mut [f64],
    mb: &mut [f64],
    lb: &mut [f64],
) {
    let len = high.len();
    debug_assert_eq!(low.len(), len);
    debug_assert_eq!(ub.len(), len);
    debug_assert_eq!(mb.len(), len);
    debug_assert_eq!(lb.len(), len);

    // --- constants ---
    let per_up_low = 50usize;
    let per_mid = 2 * period;
    let inv_up_low = 1.0 / (per_up_low as f64);
    let inv_mid = 1.0 / (per_mid as f64);

    // Ehlers band-pass coefficients
    let two_pi = core::f64::consts::PI * 2.0;
    let beta = (two_pi / (period as f64)).cos();
    let gamma = 1.0 / ((two_pi * 2.0 * delta / (period as f64)).cos());
    let alpha = gamma - (gamma * gamma - 1.0).sqrt();
    let half_one_minus_alpha = 0.5 * (1.0 - alpha);
    let beta_times_one_plus_alpha = beta * (1.0 + alpha);

    // --- ring state ---
    let mut sp_ring = vec![0.0f64; per_up_low];
    let mut sv_ring = vec![0.0f64; per_up_low];
    let mut bp_ring = vec![0.0f64; per_mid];

    let mut idx_ul = 0usize;
    let mut idx_mid = 0usize;

    let mut sum_up = 0.0f64;
    let mut sum_low = 0.0f64;
    let mut sum_mb = 0.0f64;

    // --- recursive filter & extrema state ---
    let mut bp_prev1 = 0.0f64;
    let mut bp_prev2 = 0.0f64;
    let mut peak_prev = 0.0f64;
    let mut valley_prev = 0.0f64;

    // keep prior raw prices to avoid reloading (i-2)
    let mut price_prev1 = 0.0f64;
    let mut price_prev2 = 0.0f64;

    // pointers for unchecked access
    let hi_ptr = high.as_ptr();
    let lo_ptr = low.as_ptr();
    let ub_ptr = ub.as_mut_ptr();
    let mb_ptr = mb.as_mut_ptr();
    let lb_ptr = lb.as_mut_ptr();

    // initialize state from the first valid bar
    let mut i = first;
    if i < len {
        let p0 = ((*hi_ptr.add(i)) + (*lo_ptr.add(i))) * 0.5;
        bp_prev1 = p0;
        bp_prev2 = p0;
        peak_prev = p0;
        valley_prev = p0;
        price_prev1 = p0;
        price_prev2 = p0;
    }

    // number of processed bars since `first`
    let mut count = 0usize;

    while i < len {
        // --- price midpoint ---
        let price = ((*hi_ptr.add(i)) + (*lo_ptr.add(i))) * 0.5;

        // --- band-pass recursion (order-2 resonator) ---
        let bp_curr = if count >= 2 {
            half_one_minus_alpha * (price - price_prev2) + beta_times_one_plus_alpha * bp_prev1
                - alpha * bp_prev2
        } else {
            price
        };

        // --- peak / valley detection at previous sample (local extrema) ---
        let mut peak_curr = peak_prev;
        let mut valley_curr = valley_prev;
        if count >= 2 {
            if bp_prev1 > bp_curr && bp_prev1 > bp_prev2 {
                peak_curr = bp_prev1;
            }
            if bp_prev1 < bp_curr && bp_prev1 < bp_prev2 {
                valley_curr = bp_prev1;
            }
        }

        // --- scaled peaks/valleys (fixed 50-sample MA) ---
        let sp = peak_curr * fraction;
        let sv = valley_curr * fraction;

        // --- constant-time ring updates: sum += new - old ---
        let old_sp = *sp_ring.get_unchecked(idx_ul);
        let old_sv = *sv_ring.get_unchecked(idx_ul);
        let old_bp = *bp_ring.get_unchecked(idx_mid);

        *sp_ring.get_unchecked_mut(idx_ul) = sp;
        *sv_ring.get_unchecked_mut(idx_ul) = sv;
        *bp_ring.get_unchecked_mut(idx_mid) = bp_curr;

        sum_up += sp - old_sp;
        sum_low += sv - old_sv;
        sum_mb += bp_curr - old_bp;

        // advance ring indices
        idx_ul += 1;
        if idx_ul == per_up_low {
            idx_ul = 0;
        }
        idx_mid += 1;
        if idx_mid == per_mid {
            idx_mid = 0;
        }

        // --- write outputs once windows are full ---
        let filled = count + 1;
        if filled >= per_up_low {
            *ub_ptr.add(i) = sum_up * inv_up_low;
            *lb_ptr.add(i) = sum_low * inv_up_low;
        }
        if filled >= per_mid {
            *mb_ptr.add(i) = sum_mb * inv_mid;
        }

        // --- shift state ---
        bp_prev2 = bp_prev1;
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
        price_prev2 = price_prev1;
        price_prev1 = price;

        count += 1;
        i += 1;
    }
}

#[inline(always)]
fn emd_compute_from_prices_into(
    prices: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    kernel: Kernel,
    ub: &mut [f64],
    mb: &mut [f64],
    lb: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                emd_scalar_prices_into(prices, period, delta, fraction, first, ub, mb, lb)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                emd_scalar_prices_into(prices, period, delta, fraction, first, ub, mb, lb)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                emd_scalar_prices_into(prices, period, delta, fraction, first, ub, mb, lb)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
unsafe fn emd_scalar_prices_into(
    prices: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    ub: &mut [f64],
    mb: &mut [f64],
    lb: &mut [f64],
) {
    let len = prices.len();
    debug_assert_eq!(ub.len(), len);
    debug_assert_eq!(mb.len(), len);
    debug_assert_eq!(lb.len(), len);

    let per_up_low = 50usize;
    let per_mid = 2 * period;
    let inv_up_low = 1.0 / (per_up_low as f64);
    let inv_mid = 1.0 / (per_mid as f64);

    let two_pi = core::f64::consts::PI * 2.0;
    let beta = (two_pi / (period as f64)).cos();
    let gamma = 1.0 / ((two_pi * 2.0 * delta / (period as f64)).cos());
    let alpha = gamma - (gamma * gamma - 1.0).sqrt();
    let half_one_minus_alpha = 0.5 * (1.0 - alpha);
    let beta_times_one_plus_alpha = beta * (1.0 + alpha);

    let mut sp_ring = vec![0.0f64; per_up_low];
    let mut sv_ring = vec![0.0f64; per_up_low];
    let mut bp_ring = vec![0.0f64; per_mid];
    let mut idx_ul = 0usize;
    let mut idx_mid = 0usize;

    let mut sum_up = 0.0f64;
    let mut sum_low = 0.0f64;
    let mut sum_mb = 0.0f64;

    let mut bp_prev1 = 0.0f64;
    let mut bp_prev2 = 0.0f64;
    let mut peak_prev = 0.0f64;
    let mut valley_prev = 0.0f64;

    let mut price_prev1 = 0.0f64;
    let mut price_prev2 = 0.0f64;

    let pr_ptr = prices.as_ptr();
    let ub_ptr = ub.as_mut_ptr();
    let mb_ptr = mb.as_mut_ptr();
    let lb_ptr = lb.as_mut_ptr();

    let mut i = first;
    if i < len {
        let p0 = *pr_ptr.add(i);
        bp_prev1 = p0;
        bp_prev2 = p0;
        peak_prev = p0;
        valley_prev = p0;
        price_prev1 = p0;
        price_prev2 = p0;
    }

    let mut count = 0usize;
    while i < len {
        let price = *pr_ptr.add(i);

        let bp_curr = if count >= 2 {
            half_one_minus_alpha * (price - price_prev2) + beta_times_one_plus_alpha * bp_prev1
                - alpha * bp_prev2
        } else {
            price
        };

        let mut peak_curr = peak_prev;
        let mut valley_curr = valley_prev;
        if count >= 2 {
            if bp_prev1 > bp_curr && bp_prev1 > bp_prev2 {
                peak_curr = bp_prev1;
            }
            if bp_prev1 < bp_curr && bp_prev1 < bp_prev2 {
                valley_curr = bp_prev1;
            }
        }
        let sp = peak_curr * fraction;
        let sv = valley_curr * fraction;

        let old_sp = *sp_ring.get_unchecked(idx_ul);
        let old_sv = *sv_ring.get_unchecked(idx_ul);
        let old_bp = *bp_ring.get_unchecked(idx_mid);

        *sp_ring.get_unchecked_mut(idx_ul) = sp;
        *sv_ring.get_unchecked_mut(idx_ul) = sv;
        *bp_ring.get_unchecked_mut(idx_mid) = bp_curr;

        sum_up += sp - old_sp;
        sum_low += sv - old_sv;
        sum_mb += bp_curr - old_bp;

        idx_ul += 1;
        if idx_ul == per_up_low {
            idx_ul = 0;
        }
        idx_mid += 1;
        if idx_mid == per_mid {
            idx_mid = 0;
        }

        let filled = count + 1;
        if filled >= per_up_low {
            *ub_ptr.add(i) = sum_up * inv_up_low;
            *lb_ptr.add(i) = sum_low * inv_up_low;
        }
        if filled >= per_mid {
            *mb_ptr.add(i) = sum_mb * inv_mid;
        }

        bp_prev2 = bp_prev1;
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
        price_prev2 = price_prev1;
        price_prev1 = price;

        count += 1;
        i += 1;
    }
}

// Keep the wrapper for backward compatibility
#[inline]
pub unsafe fn emd_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    let per_up_low = 50;
    let per_mid = 2 * period;
    let upperband_warmup = first + per_up_low - 1;
    let middleband_warmup = first + per_mid - 1;

    let mut upperband = alloc_with_nan_prefix(len, upperband_warmup);
    let mut middleband = alloc_with_nan_prefix(len, middleband_warmup);
    let mut lowerband = alloc_with_nan_prefix(len, upperband_warmup);

    emd_scalar_into(
        high,
        low,
        period,
        delta,
        fraction,
        first,
        &mut upperband,
        &mut middleband,
        &mut lowerband,
    );

    Ok(EmdOutput {
        upperband,
        middleband,
        lowerband,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    emd_scalar(high, low, period, delta, fraction, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    if period <= 32 {
        emd_avx512_short(high, low, period, delta, fraction, first, len)
    } else {
        emd_avx512_long(high, low, period, delta, fraction, first, len)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    emd_scalar(high, low, period, delta, fraction, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    emd_scalar(high, low, period, delta, fraction, first, len)
}

#[inline(always)]
pub fn emd_row_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_scalar(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx2(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx512(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx512_short(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx512_long(high, low, period, delta, fraction, first, len) }
}

#[derive(Debug, Clone)]
pub struct EmdStream {
    period: usize,
    delta: f64,
    fraction: f64,
    per_up_low: usize,
    per_mid: usize,
    // reciprocals to avoid per-update division
    inv_up_low: f64,
    inv_mid: f64,
    sum_up: f64,
    sum_low: f64,
    sum_mb: f64,
    sp_ring: Vec<f64>,
    sv_ring: Vec<f64>,
    bp_ring: Vec<f64>,
    idx_up_low: usize,
    idx_mid: usize,
    bp_prev1: f64,
    bp_prev2: f64,
    peak_prev: f64,
    valley_prev: f64,
    price_prev1: f64,
    price_prev2: f64,
    alpha: f64,
    beta: f64,
    // precomputed to avoid redundant ops each tick
    beta_times_one_plus_alpha: f64,
    half_one_minus_alpha: f64,
    initialized: bool,
    count: usize,
}

impl EmdStream {
    pub fn try_new(params: EmdParams) -> Result<Self, EmdError> {
        let period = params.period.unwrap_or(20);
        let delta = params.delta.unwrap_or(0.5);
        let fraction = params.fraction.unwrap_or(0.1);

        if period == 0 {
            return Err(EmdError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if delta.is_nan() || delta.is_infinite() {
            return Err(EmdError::InvalidDelta { delta });
        }
        if fraction.is_nan() || fraction.is_infinite() {
            return Err(EmdError::InvalidFraction { fraction });
        }

        // Precompute constants for performance (Ehlers coefficients)
        let two_pi_over_p = 2.0 * std::f64::consts::PI / (period as f64);
        let beta = (two_pi_over_p).cos();
        let gamma = 1.0 / ((2.0 * delta * two_pi_over_p).cos()); // = 1 / cos(4π·δ/p)
        let alpha = gamma - (gamma * gamma - 1.0).sqrt();
        let half_one_minus_alpha = 0.5 * (1.0 - alpha);
        let beta_times_one_plus_alpha = beta * (1.0 + alpha);
        let per_up_low = 50usize;
        let per_mid = 2 * period;

        Ok(Self {
            period,
            delta,
            fraction,
            per_up_low,
            per_mid,
            inv_up_low: 1.0 / (per_up_low as f64),
            inv_mid: 1.0 / (per_mid as f64),
            sum_up: 0.0,
            sum_low: 0.0,
            sum_mb: 0.0,
            sp_ring: vec![0.0; per_up_low],
            sv_ring: vec![0.0; per_up_low],
            bp_ring: vec![0.0; per_mid],
            idx_up_low: 0,
            idx_mid: 0,
            bp_prev1: 0.0,
            bp_prev2: 0.0,
            peak_prev: 0.0,
            valley_prev: 0.0,
            price_prev1: 0.0,
            price_prev2: 0.0,
            alpha,
            beta,
            beta_times_one_plus_alpha,
            half_one_minus_alpha,
            initialized: false,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
        let price = (high + low) * 0.5;

        if !self.initialized {
            self.bp_prev1 = price;
            self.bp_prev2 = price;
            self.peak_prev = price;
            self.valley_prev = price;
            self.price_prev1 = price;
            self.price_prev2 = price;
            self.initialized = true;
        }
        let bp_curr = if self.count >= 2 {
            // .5*(1-α)*(p[i]-p[i-2]) + β*(1+α)*bp[i-1] - α*bp[i-2]
            self.half_one_minus_alpha * (price - self.price_prev2)
                + self
                    .beta_times_one_plus_alpha
                    .mul_add(self.bp_prev1, -self.alpha * self.bp_prev2)
        } else {
            price
        };
        let mut peak_curr = self.peak_prev;
        let mut valley_curr = self.valley_prev;
        if self.count >= 2 {
            if self.bp_prev1 > bp_curr && self.bp_prev1 > self.bp_prev2 {
                peak_curr = self.bp_prev1;
            }
            if self.bp_prev1 < bp_curr && self.bp_prev1 < self.bp_prev2 {
                valley_curr = self.bp_prev1;
            }
        }
        let sp = peak_curr * self.fraction;
        let sv = valley_curr * self.fraction;

        // constant-time ring maintenance: sum += new - old
        let old_sp = self.sp_ring[self.idx_up_low];
        let old_sv = self.sv_ring[self.idx_up_low];
        let old_bp = self.bp_ring[self.idx_mid];
        self.sum_up += sp - old_sp;
        self.sum_low += sv - old_sv;
        self.sum_mb += bp_curr - old_bp;
        self.sp_ring[self.idx_up_low] = sp;
        self.sv_ring[self.idx_up_low] = sv;
        self.bp_ring[self.idx_mid] = bp_curr;

        // wrap without modulo in the hot path
        self.idx_up_low += 1;
        if self.idx_up_low == self.per_up_low {
            self.idx_up_low = 0;
        }
        self.idx_mid += 1;
        if self.idx_mid == self.per_mid {
            self.idx_mid = 0;
        }
        let mut ub = None;
        let mut lb = None;
        let mut mb = None;
        if self.count + 1 >= self.per_up_low {
            ub = Some(self.sum_up * self.inv_up_low);
            lb = Some(self.sum_low * self.inv_up_low);
        }
        if self.count + 1 >= self.per_mid {
            mb = Some(self.sum_mb * self.inv_mid);
        }
        self.bp_prev2 = self.bp_prev1;
        self.bp_prev1 = bp_curr;
        self.peak_prev = peak_curr;
        self.valley_prev = valley_curr;
        self.price_prev2 = self.price_prev1;
        self.price_prev1 = price;
        self.count += 1;
        (ub, mb, lb)
    }
}

// Batch struct/logic
#[derive(Clone, Debug)]
pub struct EmdBatchRange {
    pub period: (usize, usize, usize),
    pub delta: (f64, f64, f64),
    pub fraction: (f64, f64, f64),
}

impl Default for EmdBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 20, 0),
            delta: (0.5, 0.5, 0.0),
            fraction: (0.1, 0.1, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmdBatchBuilder {
    range: EmdBatchRange,
    kernel: Kernel,
}

impl EmdBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    #[inline]
    pub fn delta_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.delta = (start, end, step);
        self
    }
    #[inline]
    pub fn delta_static(mut self, x: f64) -> Self {
        self.range.delta = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn fraction_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.fraction = (start, end, step);
        self
    }
    #[inline]
    pub fn fraction_static(mut self, x: f64) -> Self {
        self.range.fraction = (x, x, 0.0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmdBatchOutput, EmdError> {
        emd_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<EmdBatchOutput, EmdError> {
        EmdBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close, volume)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<EmdBatchOutput, EmdError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(high, low, close, volume)
    }
}

pub fn emd_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    k: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
    let kernel = match k {
        // SIMD batch kernels are stubs that delegate to scalar; avoid dispatch overhead.
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(EmdError::InvalidKernelForBatch(k));
        }
    };
    emd_batch_par_slice(high, low, sweep, kernel)
}

#[derive(Clone, Debug)]
pub struct EmdBatchOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
    pub combos: Vec<EmdParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_grid(r: &EmdBatchRange) -> Result<Vec<EmdParams>, EmdError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, EmdError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                cur = match cur.checked_add(step) { Some(n) => n, None => break };
            }
        } else {
            // reversed bounds supported
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                match cur.checked_sub(step) { Some(n) => cur = n, None => break }
            }
        }
        if v.is_empty() { Err(EmdError::InvalidRangeU { start, end, step }) } else { Ok(v) }
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, EmdError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += step;
                if !x.is_finite() { break; }
            }
        } else {
            let mut x = start;
            while x >= end - 1e-12 {
                v.push(x);
                x -= step.abs();
                if !x.is_finite() { break; }
            }
        }
        if v.is_empty() { Err(EmdError::InvalidRangeF { start, end, step }) } else { Ok(v) }
    }

    let periods = axis_usize(r.period)?;
    let deltas = axis_f64(r.delta)?;
    let fractions = axis_f64(r.fraction)?;

    let cap = periods
        .len()
        .checked_mul(deltas.len())
        .and_then(|t| t.checked_mul(fractions.len()))
        .ok_or(EmdError::InvalidRangeU { start: 0, end: 0, step: 0 })?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &d in &deltas {
            for &f in &fractions {
                out.push(EmdParams { period: Some(p), delta: Some(d), fraction: Some(f) });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn emd_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    kern: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
    if !kern.is_batch() && kern != Kernel::Auto {
        return Err(EmdError::InvalidKernelForBatch(kern));
    }
    emd_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn emd_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    kern: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
    if !kern.is_batch() && kern != Kernel::Auto {
        return Err(EmdError::InvalidKernelForBatch(kern));
    }
    emd_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn emd_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EmdBatchOutput, EmdError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(EmdError::InvalidRangeU { start: 0, end: 0, step: 0 });
    }

    let len = high.len();
    if low.len() != len {
        return Err(EmdError::InvalidInputLength {
            expected: len,
            actual: low.len(),
        });
    }

    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(EmdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = (2 * max_p).max(50);
    if len - first < needed {
        return Err(EmdError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or(EmdError::InvalidRangeU {
            start: rows,
            end: cols,
            step: usize::MAX,
        })?;

    // Precompute midpoint prices once for all rows to avoid redundant work.
    let prices: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .map(|(&h, &l)| (h + l) * 0.5)
        .collect();

    // allocate uninit matrices
    let mut ub_mu = make_uninit_matrix(rows, cols);
    let mut mb_mu = make_uninit_matrix(rows, cols);
    let mut lb_mu = make_uninit_matrix(rows, cols);

    // warmup prefixes
    let warm_up_low: Vec<usize> = combos.iter().map(|_| first + 50 - 1).collect();
    let warm_mid: Vec<usize> = combos
        .iter()
        .map(|c| first + 2 * c.period.unwrap() - 1)
        .collect();

    init_matrix_prefixes(&mut ub_mu, cols, &warm_up_low);
    init_matrix_prefixes(&mut mb_mu, cols, &warm_mid);
    init_matrix_prefixes(&mut lb_mu, cols, &warm_up_low);

    // Convert to raw parts for parallel access
    let ub_ptr = ub_mu.as_mut_ptr() as *mut f64 as usize;
    let mb_ptr = mb_mu.as_mut_ptr() as *mut f64 as usize;
    let lb_ptr = lb_mu.as_mut_ptr() as *mut f64 as usize;

    let simd = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        },
        Kernel::Avx512 | Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2 | Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Scalar | Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        (0..rows).into_par_iter().for_each(|row| {
            let prm = &combos[row];
            let p = prm.period.unwrap();
            let d = prm.delta.unwrap();
            let f = prm.fraction.unwrap();

            let ub = unsafe {
                std::slice::from_raw_parts_mut((ub_ptr as *mut f64).add(row * cols), cols)
            };
            let mb = unsafe {
                std::slice::from_raw_parts_mut((mb_ptr as *mut f64).add(row * cols), cols)
            };
            let lb = unsafe {
                std::slice::from_raw_parts_mut((lb_ptr as *mut f64).add(row * cols), cols)
            };

            emd_compute_from_prices_into(&prices, p, d, f, first, simd, ub, mb, lb);
        });
        #[cfg(target_arch = "wasm32")]
        {
            let ub_rows =
                unsafe { std::slice::from_raw_parts_mut(ub_ptr as *mut f64, total) };
            let mb_rows =
                unsafe { std::slice::from_raw_parts_mut(mb_ptr as *mut f64, total) };
            let lb_rows =
                unsafe { std::slice::from_raw_parts_mut(lb_ptr as *mut f64, total) };
            for row in 0..rows {
                let prm = &combos[row];
                let p = prm.period.unwrap();
                let d = prm.delta.unwrap();
                let f = prm.fraction.unwrap();

                let ub = &mut ub_rows[row * cols..(row + 1) * cols];
                let mb = &mut mb_rows[row * cols..(row + 1) * cols];
                let lb = &mut lb_rows[row * cols..(row + 1) * cols];

                emd_compute_from_prices_into(&prices, p, d, f, first, simd, ub, mb, lb);
            }
        }
    } else {
        let ub_rows = unsafe { std::slice::from_raw_parts_mut(ub_ptr as *mut f64, total) };
        let mb_rows = unsafe { std::slice::from_raw_parts_mut(mb_ptr as *mut f64, total) };
        let lb_rows = unsafe { std::slice::from_raw_parts_mut(lb_ptr as *mut f64, total) };
        for row in 0..rows {
            let prm = &combos[row];
            let p = prm.period.unwrap();
            let d = prm.delta.unwrap();
            let f = prm.fraction.unwrap();

            let ub = &mut ub_rows[row * cols..(row + 1) * cols];
            let mb = &mut mb_rows[row * cols..(row + 1) * cols];
            let lb = &mut lb_rows[row * cols..(row + 1) * cols];

            emd_compute_from_prices_into(&prices, p, d, f, first, simd, ub, mb, lb);
        }
    }

    let upperband =
        unsafe { Vec::from_raw_parts(ub_mu.as_mut_ptr() as *mut f64, total, total) };
    let middleband =
        unsafe { Vec::from_raw_parts(mb_mu.as_mut_ptr() as *mut f64, total, total) };
    let lowerband =
        unsafe { Vec::from_raw_parts(lb_mu.as_mut_ptr() as *mut f64, total, total) };

    std::mem::forget(ub_mu);
    std::mem::forget(mb_mu);
    std::mem::forget(lb_mu);

    Ok(EmdBatchOutput {
        upperband,
        middleband,
        lowerband,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn emd_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    kern: Kernel,
    parallel: bool,
    upperband_out: &mut [f64],
    middleband_out: &mut [f64],
    lowerband_out: &mut [f64],
) -> Result<Vec<EmdParams>, EmdError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(EmdError::InvalidRangeU { start: 0, end: 0, step: 0 });
    }
    let len = high.len();
    if low.len() != len {
        return Err(EmdError::InvalidInputLength {
            expected: len,
            actual: low.len(),
        });
    }

    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or(EmdError::InvalidRangeU {
            start: rows,
            end: cols,
            step: usize::MAX,
        })?;
    if upperband_out.len() != expected
        || middleband_out.len() != expected
        || lowerband_out.len() != expected
    {
        return Err(EmdError::OutputLengthMismatch {
            expected,
            got: upperband_out
                .len()
                .min(middleband_out.len())
                .min(lowerband_out.len()),
        });
    }

    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(EmdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = (2 * max_p).max(50);
    if len - first < needed {
        return Err(EmdError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    // init NaN prefixes in-place
    {
        let mut ub_mu = unsafe {
            std::slice::from_raw_parts_mut(
                upperband_out.as_mut_ptr() as *mut MaybeUninit<f64>,
                rows * cols,
            )
        };
        let mut mb_mu = unsafe {
            std::slice::from_raw_parts_mut(
                middleband_out.as_mut_ptr() as *mut MaybeUninit<f64>,
                rows * cols,
            )
        };
        let mut lb_mu = unsafe {
            std::slice::from_raw_parts_mut(
                lowerband_out.as_mut_ptr() as *mut MaybeUninit<f64>,
                rows * cols,
            )
        };

        let warm_up_low: Vec<usize> = combos.iter().map(|_| first + 50 - 1).collect();
        let warm_mid: Vec<usize> = combos
            .iter()
            .map(|c| first + 2 * c.period.unwrap() - 1)
            .collect();

        init_matrix_prefixes(&mut ub_mu, cols, &warm_up_low);
        init_matrix_prefixes(&mut mb_mu, cols, &warm_mid);
        init_matrix_prefixes(&mut lb_mu, cols, &warm_up_low);
    }

    // Get raw pointers as usize for parallel access
    let ub_ptr = upperband_out.as_mut_ptr() as usize;
    let mb_ptr = middleband_out.as_mut_ptr() as usize;
    let lb_ptr = lowerband_out.as_mut_ptr() as usize;

    let simd = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        },
        Kernel::Avx512 | Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2 | Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Scalar | Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        (0..rows).into_par_iter().for_each(|row| {
            let prm = &combos[row];
            let p = prm.period.unwrap();
            let d = prm.delta.unwrap();
            let f = prm.fraction.unwrap();

            let ub = unsafe {
                std::slice::from_raw_parts_mut((ub_ptr as *mut f64).add(row * cols), cols)
            };
            let mb = unsafe {
                std::slice::from_raw_parts_mut((mb_ptr as *mut f64).add(row * cols), cols)
            };
            let lb = unsafe {
                std::slice::from_raw_parts_mut((lb_ptr as *mut f64).add(row * cols), cols)
            };

            emd_compute_into(high, low, p, d, f, first, simd, ub, mb, lb);
        });
        #[cfg(target_arch = "wasm32")]
        for row in 0..rows {
            let prm = &combos[row];
            let p = prm.period.unwrap();
            let d = prm.delta.unwrap();
            let f = prm.fraction.unwrap();

            let ub = &mut upperband_out[row * cols..(row + 1) * cols];
            let mb = &mut middleband_out[row * cols..(row + 1) * cols];
            let lb = &mut lowerband_out[row * cols..(row + 1) * cols];

            emd_compute_into(high, low, p, d, f, first, simd, ub, mb, lb);
        }
    } else {
        for row in 0..rows {
            let prm = &combos[row];
            let p = prm.period.unwrap();
            let d = prm.delta.unwrap();
            let f = prm.fraction.unwrap();

            let ub = &mut upperband_out[row * cols..(row + 1) * cols];
            let mb = &mut middleband_out[row * cols..(row + 1) * cols];
            let lb = &mut lowerband_out[row * cols..(row + 1) * cols];

            emd_compute_into(high, low, p, d, f, first, simd, ub, mb, lb);
        }
    }

    Ok(combos)
}

// API parity: this is required for batch indicator discovery/row mapping
impl EmdBatchOutput {
    pub fn row_for_params(&self, p: &EmdParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.delta.unwrap_or(0.5) - p.delta.unwrap_or(0.5)).abs() < 1e-12
                && (c.fraction.unwrap_or(0.1) - p.fraction.unwrap_or(0.1)).abs() < 1e-12
        })
    }
    pub fn bands_for(&self, p: &EmdParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.upperband[start..start + self.cols],
                &self.middleband[start..start + self.cols],
                &self.lowerband[start..start + self.cols],
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    #[cfg(not(feature = "wasm"))]
    fn test_emd_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small but non-trivial OHLC-like input
        let n = 256usize; // >= max(50, 2*default_period)
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        for i in 0..n {
            let base = 100.0 + (i as f64 * 0.01)
                + (2.0 * std::f64::consts::PI * (i as f64) / 17.0).sin() * 3.0
                + (2.0 * std::f64::consts::PI * (i as f64) / 49.0).cos() * 2.0;
            high.push(base + 1.25);
            low.push(base - 1.10);
        }

        // Default params (period=20, delta=0.5, fraction=0.1)
        let params = EmdParams::default();
        let input = EmdInput::from_slices(&high, &low, &[], &[], params);

        // Baseline via Vec-returning API
        let baseline = emd(&input)?;

        // Preallocate output buffers and compute via into()
        let mut ub = vec![0.0; n];
        let mut mb = vec![0.0; n];
        let mut lb = vec![0.0; n];
        emd_into(&input, &mut ub, &mut mb, &mut lb)?;

        // Helper: NaN-equal comparator
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        // Length and element-wise equality (NaN == NaN)
        assert_eq!(baseline.upperband.len(), ub.len());
        assert_eq!(baseline.middleband.len(), mb.len());
        assert_eq!(baseline.lowerband.len(), lb.len());
        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline.upperband[i], ub[i]),
                "upperband mismatch at {}: {:?} vs {:?}",
                i,
                baseline.upperband[i],
                ub[i]
            );
            assert!(
                eq_or_both_nan(baseline.middleband[i], mb[i]),
                "middleband mismatch at {}: {:?} vs {:?}",
                i,
                baseline.middleband[i],
                mb[i]
            );
            assert!(
                eq_or_both_nan(baseline.lowerband[i], lb[i]),
                "lowerband mismatch at {}: {:?} vs {:?}",
                i,
                baseline.lowerband[i],
                lb[i]
            );
        }
        Ok(())
    }

    fn check_emd_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = EmdParams::default();
        let input = EmdInput::from_candles(&candles, params);
        let emd_result = emd_with_kernel(&input, kernel)?;

        let expected_last_five_upper = [
            50.33760237677157,
            50.28850695686447,
            50.23941153695737,
            50.19031611705027,
            48.709744457737344,
        ];
        let expected_last_five_middle = [
            -368.71064280396706,
            -399.11033986231377,
            -421.9368852621732,
            -437.879217150269,
            -447.3257167904511,
        ];
        let expected_last_five_lower = [
            -60.67834136221248,
            -60.93110347122829,
            -61.68154077026321,
            -62.43197806929814,
            -63.18241536833306,
        ];

        let len = candles.close.len();
        let start_idx = len - 5;
        let actual_ub = &emd_result.upperband[start_idx..];
        let actual_mb = &emd_result.middleband[start_idx..];
        let actual_lb = &emd_result.lowerband[start_idx..];
        for i in 0..5 {
            assert!(
                (actual_ub[i] - expected_last_five_upper[i]).abs() < 1e-6,
                "Upperband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_upper[i],
                actual_ub[i]
            );
            assert!(
                (actual_mb[i] - expected_last_five_middle[i]).abs() < 1e-6,
                "Middleband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_middle[i],
                actual_mb[i]
            );
            assert!(
                (actual_lb[i] - expected_last_five_lower[i]).abs() < 1e-6,
                "Lowerband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_lower[i],
                actual_lb[i]
            );
        }
        Ok(())
    }

    fn check_emd_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty_data: [f64; 0] = [];
        let params = EmdParams::default();
        let input =
            EmdInput::from_slices(&empty_data, &empty_data, &empty_data, &empty_data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error on empty data");
        Ok(())
    }

    fn check_emd_all_nans(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = EmdParams::default();
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all-NaN data");
        Ok(())
    }

    fn check_emd_invalid_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = EmdParams {
            period: Some(0),
            ..Default::default()
        };
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for zero period");
        Ok(())
    }

    fn check_emd_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![10.0; 10];
        let params = EmdParams {
            period: Some(20),
            ..Default::default()
        };
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for not enough valid data");
        Ok(())
    }

    fn check_emd_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmdInput::with_default_candles(&candles);
        let result = emd_with_kernel(&input, kernel);
        assert!(
            result.is_ok(),
            "Expected EMD to succeed with default params"
        );
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_emd_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy 1: Variable period with random price data
        let strat1 = (2usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (1f64..1000f64).prop_filter("finite", |x| x.is_finite()),
                    (2 * period).max(50)..400,
                ),
                Just(period),
                (0.1f64..1.0f64).prop_filter("finite", |x| x.is_finite()),
                (0.01f64..0.5f64).prop_filter("finite", |x| x.is_finite()),
            )
        });

        // Strategy 2: Fixed period with varying data lengths
        let strat2 = prop::collection::vec(
            (100f64..10000f64).prop_filter("finite", |x| x.is_finite()),
            100..500,
        )
        .prop_map(|data| (data, 20usize, 0.5f64, 0.1f64));

        // Strategy 3: Trending data (monotonic)
        let strat3 = (100usize..400, prop::bool::ANY).prop_map(|(len, increasing)| {
            let mut data = Vec::with_capacity(len);
            let mut val = 100.0;
            for _ in 0..len {
                data.push(val);
                val += if increasing { 1.0 } else { -1.0 };
            }
            (data, 14usize, 0.5f64, 0.1f64)
        });

        // Strategy 4: Oscillating data (sine-wave patterns)
        let strat4 = (100usize..400, 5usize..50).prop_map(|(len, period_wave)| {
            let mut data = Vec::with_capacity(len);
            for i in 0..len {
                let val = 1000.0
                    + 100.0 * (2.0 * std::f64::consts::PI * i as f64 / period_wave as f64).sin();
                data.push(val);
            }
            (data, 20usize, 0.5f64, 0.1f64)
        });

        // Strategy 5: Real-world-like OHLC data with proper constraints
        let strat5 = (2usize..=30).prop_flat_map(|period| {
            let min_len = (2 * period).max(50);
            prop::collection::vec(
                (
                    50f64..150f64,
                    0.1f64..10f64,
                    -0.5f64..0.5f64,
                    0f64..0.5f64,
                    0f64..0.5f64,
                )
                    .prop_map(
                        |(base, range, close_offset, high_extra, low_extra)| {
                            let open = base;
                            let close = base + range * close_offset;
                            let high = open.max(close) + range * high_extra;
                            let low = open.min(close) - range * low_extra;
                            (high, low, close, base * 1000.0) // volume
                        },
                    ),
                min_len..300,
            )
            .prop_map(move |ohlc_data| {
                let (highs, lows, closes, volumes): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
                    ohlc_data.into_iter().unzip4();
                (highs, lows, closes, volumes, period, 0.5f64, 0.1f64)
            })
        });

        // Helper closure to unzip4
        trait Unzip4<A, B, C, D> {
            fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>);
        }
        impl<A, B, C, D, I: Iterator<Item = (A, B, C, D)>> Unzip4<A, B, C, D> for I {
            fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>) {
                let (mut a_vec, mut b_vec, mut c_vec, mut d_vec) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for (a, b, c, d) in self {
                    a_vec.push(a);
                    b_vec.push(b);
                    c_vec.push(c);
                    d_vec.push(d);
                }
                (a_vec, b_vec, c_vec, d_vec)
            }
        }

        // Combine all strategies
        let combined_strat = prop_oneof![
            strat1.prop_map(|(data, period, delta, fraction)| {
                let high = data.clone();
                let low = data.clone();
                let close = data.clone();
                let volume = vec![1000.0; data.len()];
                (high, low, close, volume, period, delta, fraction)
            }),
            strat2.prop_map(|(data, period, delta, fraction)| {
                let high = data.iter().map(|x| x + 10.0).collect();
                let low = data.iter().map(|x| x - 10.0).collect();
                let close = data.clone();
                let volume = vec![1000.0; data.len()];
                (high, low, close, volume, period, delta, fraction)
            }),
            strat3.prop_map(|(data, period, delta, fraction)| {
                let high = data.iter().map(|x| x + 5.0).collect();
                let low = data.iter().map(|x| x - 5.0).collect();
                let close = data.clone();
                let volume = vec![1000.0; data.len()];
                (high, low, close, volume, period, delta, fraction)
            }),
            strat4.prop_map(|(data, period, delta, fraction)| {
                let high = data.iter().map(|x| x + 20.0).collect();
                let low = data.iter().map(|x| x - 20.0).collect();
                let close = data.clone();
                let volume = vec![5000.0; data.len()];
                (high, low, close, volume, period, delta, fraction)
            }),
            strat5,
        ];

        proptest::test_runner::TestRunner::default().run(
            &combined_strat,
            |(high, low, close, volume, period, delta, fraction)| {
                // Validate input constraints
                for i in 0..high.len() {
                    prop_assert!(
                        high[i] >= low[i],
                        "Invalid OHLC data at index {}: high ({}) < low ({})",
                        i,
                        high[i],
                        low[i]
                    );
                }

                let params = EmdParams {
                    period: Some(period),
                    delta: Some(delta),
                    fraction: Some(fraction),
                };
                let input = EmdInput::from_slices(&high, &low, &close, &volume, params);

                // Test with specified kernel
                let result = emd_with_kernel(&input, kernel).unwrap();
                let upperband = &result.upperband;
                let middleband = &result.middleband;
                let lowerband = &result.lowerband;

                // Test with scalar kernel for comparison
                let ref_result = emd_with_kernel(&input, Kernel::Scalar).unwrap();
                let ref_upperband = &ref_result.upperband;
                let ref_middleband = &ref_result.middleband;
                let ref_lowerband = &ref_result.lowerband;

                // Property 1: Output length matches input
                prop_assert_eq!(upperband.len(), high.len());
                prop_assert_eq!(middleband.len(), high.len());
                prop_assert_eq!(lowerband.len(), high.len());

                // Property 2: Warmup period handling
                // Actual warmup calculation from emd_scalar:
                // upperband_warmup = first + per_up_low - 1 = 0 + 50 - 1 = 49
                // middleband_warmup = first + per_mid - 1 = 0 + 2*period - 1
                let upperband_warmup = (50 - 1).min(high.len());
                let middleband_warmup = ((2 * period) - 1).min(high.len());

                for i in 0..upperband_warmup {
                    prop_assert!(
                        upperband[i].is_nan(),
                        "Upperband should be NaN during warmup at index {}",
                        i
                    );
                    prop_assert!(
                        lowerband[i].is_nan(),
                        "Lowerband should be NaN during warmup at index {}",
                        i
                    );
                }

                for i in 0..middleband_warmup {
                    prop_assert!(
                        middleband[i].is_nan(),
                        "Middleband should be NaN during warmup at index {}",
                        i
                    );
                }

                // Property 3: Band values should be reasonable (after warmup)
                let start_idx = upperband_warmup.max(middleband_warmup) + 1;
                if start_idx < high.len() {
                    // Calculate input data range for bounds checking
                    let input_min = high[start_idx..]
                        .iter()
                        .chain(low[start_idx..].iter())
                        .fold(
                            f64::INFINITY,
                            |a, &b| if b.is_finite() { a.min(b) } else { a },
                        );
                    let input_max = high[start_idx..]
                        .iter()
                        .chain(low[start_idx..].iter())
                        .fold(
                            f64::NEG_INFINITY,
                            |a, &b| if b.is_finite() { a.max(b) } else { a },
                        );

                    if input_min.is_finite() && input_max.is_finite() {
                        let range = input_max - input_min;
                        let center = (input_max + input_min) / 2.0;
                        // Bands should be within reasonable bounds of input data
                        // EMD bands can extend beyond input range due to filtering, but not excessively
                        let bounds_factor = 3.0; // Allow bands to be within 3x range from center
                        let lower_bound = center - bounds_factor * range.max(1.0);
                        let upper_bound = center + bounds_factor * range.max(1.0);

                        for i in start_idx..high.len() {
                            if !upperband[i].is_nan()
                                && !middleband[i].is_nan()
                                && !lowerband[i].is_nan()
                            {
                                // All bands should be finite
                                prop_assert!(
                                    upperband[i].is_finite(),
                                    "Upperband should be finite at index {}",
                                    i
                                );
                                prop_assert!(
                                    middleband[i].is_finite(),
                                    "Middleband should be finite at index {}",
                                    i
                                );
                                prop_assert!(
                                    lowerband[i].is_finite(),
                                    "Lowerband should be finite at index {}",
                                    i
                                );

                                // Bands should be within reasonable bounds
                                prop_assert!(
                                    upperband[i] >= lower_bound && upperband[i] <= upper_bound,
                                    "Upperband {} at index {} outside reasonable bounds [{}, {}]",
                                    upperband[i],
                                    i,
                                    lower_bound,
                                    upper_bound
                                );
                                prop_assert!(
                                    middleband[i] >= lower_bound && middleband[i] <= upper_bound,
                                    "Middleband {} at index {} outside reasonable bounds [{}, {}]",
                                    middleband[i],
                                    i,
                                    lower_bound,
                                    upper_bound
                                );
                                prop_assert!(
                                    lowerband[i] >= lower_bound && lowerband[i] <= upper_bound,
                                    "Lowerband {} at index {} outside reasonable bounds [{}, {}]",
                                    lowerband[i],
                                    i,
                                    lower_bound,
                                    upper_bound
                                );
                            }
                        }
                    }
                }

                // Property 4: Kernel consistency
                // Use more reasonable tolerance for floating-point comparison across SIMD implementations
                let tolerance = 1e-10;
                for i in 0..high.len() {
                    let ub_diff = (upperband[i] - ref_upperband[i]).abs();
                    let mb_diff = (middleband[i] - ref_middleband[i]).abs();
                    let lb_diff = (lowerband[i] - ref_lowerband[i]).abs();

                    if !upperband[i].is_nan() && !ref_upperband[i].is_nan() {
                        prop_assert!(
                            ub_diff < tolerance,
                            "Upperband kernel mismatch at index {}: {} vs {} (diff: {})",
                            i,
                            upperband[i],
                            ref_upperband[i],
                            ub_diff
                        );
                    }
                    if !middleband[i].is_nan() && !ref_middleband[i].is_nan() {
                        prop_assert!(
                            mb_diff < tolerance,
                            "Middleband kernel mismatch at index {}: {} vs {} (diff: {})",
                            i,
                            middleband[i],
                            ref_middleband[i],
                            mb_diff
                        );
                    }
                    if !lowerband[i].is_nan() && !ref_lowerband[i].is_nan() {
                        prop_assert!(
                            lb_diff < tolerance,
                            "Lowerband kernel mismatch at index {}: {} vs {} (diff: {})",
                            i,
                            lowerband[i],
                            ref_lowerband[i],
                            lb_diff
                        );
                    }
                }

                // Property 5: Check for poison values
                for i in 0..high.len() {
                    let ub_bits = upperband[i].to_bits();
                    let mb_bits = middleband[i].to_bits();
                    let lb_bits = lowerband[i].to_bits();

                    prop_assert_ne!(
                        ub_bits,
                        0x1111_1111_1111_1111,
                        "Poison value in upperband at {}",
                        i
                    );
                    prop_assert_ne!(
                        ub_bits,
                        0x2222_2222_2222_2222,
                        "Poison value in upperband at {}",
                        i
                    );
                    prop_assert_ne!(
                        ub_bits,
                        0x3333_3333_3333_3333,
                        "Poison value in upperband at {}",
                        i
                    );

                    prop_assert_ne!(
                        mb_bits,
                        0x1111_1111_1111_1111,
                        "Poison value in middleband at {}",
                        i
                    );
                    prop_assert_ne!(
                        mb_bits,
                        0x2222_2222_2222_2222,
                        "Poison value in middleband at {}",
                        i
                    );
                    prop_assert_ne!(
                        mb_bits,
                        0x3333_3333_3333_3333,
                        "Poison value in middleband at {}",
                        i
                    );

                    prop_assert_ne!(
                        lb_bits,
                        0x1111_1111_1111_1111,
                        "Poison value in lowerband at {}",
                        i
                    );
                    prop_assert_ne!(
                        lb_bits,
                        0x2222_2222_2222_2222,
                        "Poison value in lowerband at {}",
                        i
                    );
                    prop_assert_ne!(
                        lb_bits,
                        0x3333_3333_3333_3333,
                        "Poison value in lowerband at {}",
                        i
                    );
                }

                // Property 6: Edge case - period = 2 (minimum viable)
                if period == 2 {
                    // Should still produce valid output after warmup
                    let min_warmup = (2 * 2).max(50); // max(4, 50) = 50
                    if high.len() > min_warmup {
                        prop_assert!(
                            middleband[min_warmup].is_finite() || middleband[min_warmup].is_nan(),
                            "Period=2 should produce valid or NaN output"
                        );
                    }
                }

                // Property 7: Constant data should produce stable output
                if high.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON)
                    && low.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON)
                    && close.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON)
                {
                    // For constant input, bands should stabilize after warmup
                    let check_start = start_idx + period;
                    if check_start + 5 < high.len() {
                        let ub_stable = &upperband[check_start..check_start + 5];
                        let mb_stable = &middleband[check_start..check_start + 5];
                        let lb_stable = &lowerband[check_start..check_start + 5];

                        for w in ub_stable.windows(2) {
                            if !w[0].is_nan() && !w[1].is_nan() {
                                prop_assert!(
                                    (w[0] - w[1]).abs() < 1e-6,
                                    "Upperband should be stable for constant input"
                                );
                            }
                        }
                        for w in mb_stable.windows(2) {
                            if !w[0].is_nan() && !w[1].is_nan() {
                                prop_assert!(
                                    (w[0] - w[1]).abs() < 1e-6,
                                    "Middleband should be stable for constant input"
                                );
                            }
                        }
                        for w in lb_stable.windows(2) {
                            if !w[0].is_nan() && !w[1].is_nan() {
                                prop_assert!(
                                    (w[0] - w[1]).abs() < 1e-6,
                                    "Lowerband should be stable for constant input"
                                );
                            }
                        }
                    }
                }

                // Property 8: Fraction parameter effect
                // When fraction is very small, upper and lower bands should be close to 0 after warmup
                if fraction < 0.01 && start_idx < high.len() {
                    // Check a few values after warmup
                    let check_end = (start_idx + 10).min(high.len());
                    for i in start_idx..check_end {
                        if !upperband[i].is_nan() && !lowerband[i].is_nan() {
                            // With very small fraction, bands should be near zero
                            // (since they're fraction * peak/valley averages)
                            prop_assert!(
                                upperband[i].abs() < 10.0,
                                "With fraction={}, upperband should be small, got {} at index {}",
                                fraction,
                                upperband[i],
                                i
                            );
                            prop_assert!(
                                lowerband[i].abs() < 10.0,
                                "With fraction={}, lowerband should be small, got {} at index {}",
                                fraction,
                                lowerband[i],
                                i
                            );
                        }
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_emd_property(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_emd_tests {
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

    #[cfg(debug_assertions)]
    fn check_emd_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            EmdParams::default(),
            // Minimum viable parameters
            EmdParams {
                period: Some(2),
                delta: Some(0.1),
                fraction: Some(0.05),
            },
            // Small period variations
            EmdParams {
                period: Some(5),
                delta: Some(0.3),
                fraction: Some(0.1),
            },
            EmdParams {
                period: Some(10),
                delta: Some(0.5),
                fraction: Some(0.15),
            },
            // Medium periods
            EmdParams {
                period: Some(20),
                delta: Some(0.4),
                fraction: Some(0.1),
            },
            EmdParams {
                period: Some(30),
                delta: Some(0.6),
                fraction: Some(0.2),
            },
            // Large periods
            EmdParams {
                period: Some(50),
                delta: Some(0.7),
                fraction: Some(0.25),
            },
            EmdParams {
                period: Some(100),
                delta: Some(0.8),
                fraction: Some(0.3),
            },
            // Edge cases with extreme delta
            EmdParams {
                period: Some(15),
                delta: Some(0.1),
                fraction: Some(0.1),
            },
            EmdParams {
                period: Some(15),
                delta: Some(0.9),
                fraction: Some(0.1),
            },
            // Edge cases with extreme fraction
            EmdParams {
                period: Some(25),
                delta: Some(0.5),
                fraction: Some(0.01),
            },
            EmdParams {
                period: Some(25),
                delta: Some(0.5),
                fraction: Some(0.5),
            },
            // Additional combinations
            EmdParams {
                period: Some(40),
                delta: Some(0.65),
                fraction: Some(0.12),
            },
            EmdParams {
                period: Some(7),
                delta: Some(0.25),
                fraction: Some(0.08),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = EmdInput::from_candles(&candles, params.clone());
            let output = emd_with_kernel(&input, kernel)?;

            // Check upperband array
            for (i, &val) in output.upperband.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in upperband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in upperband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in upperband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }
            }

            // Check middleband array
            for (i, &val) in output.middleband.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in middleband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in middleband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in middleband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }
            }

            // Check lowerband array
            for (i, &val) in output.lowerband.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in lowerband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in lowerband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in lowerband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_emd_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    generate_all_emd_tests!(
        check_emd_accuracy,
        check_emd_empty_data,
        check_emd_all_nans,
        check_emd_invalid_period,
        check_emd_not_enough_valid_data,
        check_emd_default_candles,
        check_emd_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_emd_tests!(check_emd_property);

    #[cfg(test)]
    mod batch_tests {
        use super::*;
        use crate::skip_if_unsupported;
        use crate::utilities::data_loader::read_candles_from_csv;

        fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
            skip_if_unsupported!(kernel, test);

            let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
            let c = read_candles_from_csv(file)?;

            let output = EmdBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

            let def = EmdParams::default();
            let (ub, mb, lb) = output.bands_for(&def).expect("default row missing");

            assert_eq!(ub.len(), c.close.len(), "Upperband length mismatch");
            assert_eq!(mb.len(), c.close.len(), "Middleband length mismatch");
            assert_eq!(lb.len(), c.close.len(), "Lowerband length mismatch");

            // Spot check last values vs. single-batch computation (if desired, could hardcode here)
            let expected_last_five_upper = [
                50.33760237677157,
                50.28850695686447,
                50.23941153695737,
                50.19031611705027,
                48.709744457737344,
            ];
            let expected_last_five_middle = [
                -368.71064280396706,
                -399.11033986231377,
                -421.9368852621732,
                -437.879217150269,
                -447.3257167904511,
            ];
            let expected_last_five_lower = [
                -60.67834136221248,
                -60.93110347122829,
                -61.68154077026321,
                -62.43197806929814,
                -63.18241536833306,
            ];
            let len = ub.len();
            for i in 0..5 {
                assert!(
                    (ub[len - 5 + i] - expected_last_five_upper[i]).abs() < 1e-6,
                    "[{test}] upperband mismatch idx {i}: {} vs {}",
                    ub[len - 5 + i],
                    expected_last_five_upper[i]
                );
                assert!(
                    (mb[len - 5 + i] - expected_last_five_middle[i]).abs() < 1e-6,
                    "[{test}] middleband mismatch idx {i}: {} vs {}",
                    mb[len - 5 + i],
                    expected_last_five_middle[i]
                );
                assert!(
                    (lb[len - 5 + i] - expected_last_five_lower[i]).abs() < 1e-6,
                    "[{test}] lowerband mismatch idx {i}: {} vs {}",
                    lb[len - 5 + i],
                    expected_last_five_lower[i]
                );
            }

            Ok(())
        }

        fn check_batch_param_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
            skip_if_unsupported!(kernel, test);

            let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
            let c = read_candles_from_csv(file)?;

            // Sweep period over 20 and 22, delta over 0.5 and 0.6, fraction over 0.1 and 0.2
            let output = EmdBatchBuilder::new()
                .kernel(kernel)
                .period_range(20, 22, 2)
                .delta_range(0.5, 0.6, 0.1)
                .fraction_range(0.1, 0.2, 0.1)
                .apply_candles(&c)?;

            assert!(
                output.rows == 8,
                "Expected 8 rows (2*2*2 grid), got {}",
                output.rows
            );
            assert_eq!(output.cols, c.close.len());

            // Verify that bands_for returns correct shapes for all combos
            for params in &output.combos {
                let (ub, mb, lb) = output
                    .bands_for(params)
                    .expect("row for params missing in sweep");
                assert_eq!(ub.len(), output.cols);
                assert_eq!(mb.len(), output.cols);
                assert_eq!(lb.len(), output.cols);
            }

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

        #[cfg(debug_assertions)]
        fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
            skip_if_unsupported!(kernel, test);

            let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
            let c = read_candles_from_csv(file)?;

            // Test various parameter sweep configurations
            let test_configs = vec![
                // (period range, delta range, fraction range)
                // Small periods with varying delta and fraction
                (5, 15, 5, 0.1, 0.5, 0.2, 0.05, 0.15, 0.05),
                // Medium periods
                (10, 30, 10, 0.3, 0.7, 0.2, 0.1, 0.2, 0.05),
                // Large periods
                (20, 50, 15, 0.5, 0.8, 0.15, 0.15, 0.3, 0.075),
                // Dense small range
                (8, 12, 1, 0.4, 0.6, 0.1, 0.08, 0.12, 0.02),
                // Single values (no sweep)
                (20, 20, 0, 0.5, 0.5, 0.0, 0.1, 0.1, 0.0),
                // Wide period range with fine delta/fraction
                (5, 40, 5, 0.2, 0.8, 0.1, 0.05, 0.25, 0.05),
                // Edge case: minimum periods with varying other params
                (2, 6, 2, 0.1, 0.9, 0.4, 0.01, 0.5, 0.245),
            ];

            for (
                cfg_idx,
                &(p_start, p_end, p_step, d_start, d_end, d_step, f_start, f_end, f_step),
            ) in test_configs.iter().enumerate()
            {
                let output = EmdBatchBuilder::new()
                    .kernel(kernel)
                    .period_range(p_start, p_end, p_step)
                    .delta_range(d_start, d_end, d_step)
                    .fraction_range(f_start, f_end, f_step)
                    .apply_candles(&c)?;

                // Check upperband matrix
                for (idx, &val) in output.upperband.iter().enumerate() {
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
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in upperband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in upperband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in upperband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }
                }

                // Check middleband matrix
                for (idx, &val) in output.middleband.iter().enumerate() {
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
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in middleband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in middleband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in middleband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }
                }

                // Check lowerband matrix
                for (idx, &val) in output.lowerband.iter().enumerate() {
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
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in lowerband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in lowerband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in lowerband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
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

        gen_batch_tests!(check_batch_default_row);
        gen_batch_tests!(check_batch_param_sweep);
        gen_batch_tests!(check_batch_no_poison);
    }
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "emd")]
#[pyo3(signature = (high, low, period, delta, fraction, kernel=None))]
pub fn emd_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    delta: f64,
    fraction: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let hi = high.as_slice()?;
    let lo = low.as_slice()?;
    if hi.len() != lo.len() {
        return Err(PyValueError::new_err("high and low must have same length"));
    }

    let params = EmdParams {
        period: Some(period),
        delta: Some(delta),
        fraction: Some(fraction),
    };
    let inp = EmdInput::from_slices(hi, lo, &[], &[], params); // close/volume unused
    let kern = validate_kernel(kernel, false)?;

    let ub = unsafe { PyArray1::<f64>::new(py, [hi.len()], false) };
    let mb = unsafe { PyArray1::<f64>::new(py, [hi.len()], false) };
    let lb = unsafe { PyArray1::<f64>::new(py, [hi.len()], false) };

    let ubm = unsafe { ub.as_slice_mut()? };
    let mbm = unsafe { mb.as_slice_mut()? };
    let lbm = unsafe { lb.as_slice_mut()? };

    py.allow_threads(|| emd_into_slices(ubm, mbm, lbm, &inp, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((ub, mb, lb))
}

#[cfg(feature = "python")]
#[pyclass(name = "EmdStream")]
pub struct EmdStreamPy {
    stream: EmdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EmdStreamPy {
    #[new]
    fn new(period: usize, delta: f64, fraction: f64) -> PyResult<Self> {
        let params = EmdParams {
            period: Some(period),
            delta: Some(delta),
            fraction: Some(fraction),
        };
        let stream =
            EmdStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(EmdStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "emd_batch")]
#[pyo3(signature = (high, low, period_range, delta_range, fraction_range, kernel=None))]
pub fn emd_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    delta_range: (f64, f64, f64),
    fraction_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArrayMethods;

    let hi = high.as_slice()?;
    let lo = low.as_slice()?;
    if hi.len() != lo.len() {
        return Err(PyValueError::new_err("high and low must have same length"));
    }

    let sweep = EmdBatchRange {
        period: period_range,
        delta: delta_range,
        fraction: fraction_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = hi.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflow in emd_batch_py"))?;

    let ub = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let mb = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let lb = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let ubm = unsafe { ub.as_slice_mut()? };
    let mbm = unsafe { mb.as_slice_mut()? };
    let lbm = unsafe { lb.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py
        .allow_threads(|| {
            let k = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match k {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            emd_batch_inner_into(hi, lo, &sweep, simd, true, ubm, mbm, lbm)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("upper", ub.reshape((rows, cols))?)?;
    d.set_item("middle", mb.reshape((rows, cols))?)?;
    d.set_item("lower", lb.reshape((rows, cols))?)?;
    d.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "deltas",
        combos
            .iter()
            .map(|p| p.delta.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "fractions",
        combos
            .iter()
            .map(|p| p.fraction.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

// ==================== PYTHON CUDA BINDINGS ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "emd_cuda_batch_dev")]
#[pyo3(signature = (high, low, period_range, delta_range, fraction_range, device_id=0))]
pub fn emd_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f32>,
    low: PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    delta_range: (f64, f64, f64),
    fraction_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let hi = high.as_slice()?;
    let lo = low.as_slice()?;
    if hi.len() != lo.len() {
        return Err(PyValueError::new_err("high and low must have same length"));
    }
    let sweep = EmdBatchRange {
        period: period_range,
        delta: delta_range,
        fraction: fraction_range,
    };
    let (outputs, combos, dev_id) = py.allow_threads(|| {
        let cuda = CudaEmd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let res = cuda
            .emd_batch_dev(hi, lo, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()));
        res.map(|r| (r.outputs, r.combos, dev_id))
    })?;
    let DeviceArrayF32Triple {
        upper,
        middle,
        lower,
    } = outputs;
    let dict = pyo3::types::PyDict::new(py);
    let upper_dev = make_device_array_py(dev_id as usize, upper)?;
    dict.set_item("upperband", Py::new(py, upper_dev)?)?;
    let middle_dev = make_device_array_py(dev_id as usize, middle)?;
    dict.set_item("middleband", Py::new(py, middle_dev)?)?;
    let lower_dev = make_device_array_py(dev_id as usize, lower)?;
    dict.set_item("lowerband", Py::new(py, lower_dev)?)?;
    // Flatten parameters
    let periods: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
    let deltas: Vec<f64> = combos.iter().map(|c| c.delta.unwrap()).collect();
    let fractions: Vec<f64> = combos.iter().map(|c| c.fraction.unwrap()).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("deltas", deltas.into_pyarray(py))?;
    dict.set_item("fractions", fractions.into_pyarray(py))?;
    dict.set_item("rows", combos.len())?;
    dict.set_item("cols", hi.len())?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "emd_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, delta, fraction, device_id=0))]
pub fn emd_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    period: usize,
    delta: f64,
    fraction: f64,
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
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

    // Per-series first_valid indices
    let mut first_valids = vec![0i32; cols];
    for s in 0..cols {
        let mut fv: Option<i32> = None;
        for t in 0..rows {
            let v = flat[t * cols + s];
            if v.is_finite() {
                fv = Some(t as i32);
                break;
            }
        }
        first_valids[s] =
            fv.ok_or_else(|| PyValueError::new_err(format!("series {} has no finite values", s)))?;
    }

    let params = EmdParams {
        period: Some(period),
        delta: Some(delta),
        fraction: Some(fraction),
    };
    let (outputs, dev_id) = py.allow_threads(|| {
        let cuda = CudaEmd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        cuda
            .emd_many_series_one_param_time_major_dev(flat, cols, rows, &params, &first_valids)
            .map(|o| (o, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let DeviceArrayF32Triple {
        upper,
        middle,
        lower,
    } = outputs;
    let dict = pyo3::types::PyDict::new(py);
    let upper_dev = make_device_array_py(dev_id as usize, upper)?;
    dict.set_item("upperband", Py::new(py, upper_dev)?)?;
    let middle_dev = make_device_array_py(dev_id as usize, middle)?;
    dict.set_item("middleband", Py::new(py, middle_dev)?)?;
    let lower_dev = make_device_array_py(dev_id as usize, lower)?;
    dict.set_item("lowerband", Py::new(py, lower_dev)?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("period", period)?;
    dict.set_item("delta", delta)?;
    dict.set_item("fraction", fraction)?;
    Ok(dict)
}

// ############################################
// WASM Bindings
// ############################################

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmdJsOutput {
    pub values: Vec<f64>, // [upper..., middle..., lower...]
    pub rows: usize,      // 3
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "emd_js")]
pub fn emd_js(
    high: &[f64],
    low: &[f64],
    _close: &[f64],
    _volume: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("high and low must have same length"));
    }
    let params = EmdParams {
        period: Some(period),
        delta: Some(delta),
        fraction: Some(fraction),
    };
    let input = EmdInput::from_slices(high, low, &[], &[], params);

    let mut values = vec![f64::NAN; 3 * high.len()];
    let (ub, rest) = values.split_at_mut(high.len());
    let (mb, lb) = rest.split_at_mut(high.len());

    emd_into_slices(ub, mb, lb, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let output = EmdJsOutput {
        values,
        rows: 3,
        cols: high.len(),
    };
    serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    _close_ptr: *const f64,  // unused but kept for API compatibility
    _volume_ptr: *const f64, // unused but kept for API compatibility
    upper_ptr: *mut f64,
    middle_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
    period: usize,
    delta: f64,
    fraction: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || upper_ptr.is_null()
        || middle_ptr.is_null()
        || lower_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        // Check for aliasing - if high_ptr == upper_ptr, we need to copy the data first
        let hi_aliased = high_ptr as *const f64 == upper_ptr as *const f64;
        let lo_aliased = low_ptr as *const f64 == upper_ptr as *const f64
            || low_ptr as *const f64 == middle_ptr as *const f64
            || low_ptr as *const f64 == lower_ptr as *const f64;

        if hi_aliased || lo_aliased {
            // Handle aliasing by computing to temporary buffers first
            let hi = std::slice::from_raw_parts(high_ptr, len);
            let lo = std::slice::from_raw_parts(low_ptr, len);

            let params = EmdParams {
                period: Some(period),
                delta: Some(delta),
                fraction: Some(fraction),
            };
            let input = EmdInput::from_slices(hi, lo, &[], &[], params);
            let output = emd_with_kernel(&input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let ub = std::slice::from_raw_parts_mut(upper_ptr, len);
            let mb = std::slice::from_raw_parts_mut(middle_ptr, len);
            let lb = std::slice::from_raw_parts_mut(lower_ptr, len);
            ub.copy_from_slice(&output.upperband);
            mb.copy_from_slice(&output.middleband);
            lb.copy_from_slice(&output.lowerband);
        } else {
            // No aliasing, can write directly
            let hi = std::slice::from_raw_parts(high_ptr, len);
            let lo = std::slice::from_raw_parts(low_ptr, len);
            let ub = std::slice::from_raw_parts_mut(upper_ptr, len);
            let mb = std::slice::from_raw_parts_mut(middle_ptr, len);
            let lb = std::slice::from_raw_parts_mut(lower_ptr, len);
            let params = EmdParams {
                period: Some(period),
                delta: Some(delta),
                fraction: Some(fraction),
            };
            let input = EmdInput::from_slices(hi, lo, &[], &[], params);
            emd_into_slices(ub, mb, lb, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmdBatchConfig {
    pub period_range: (usize, usize, usize),
    pub delta_range: (f64, f64, f64),
    pub fraction_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmdBatchJsOutput {
    pub upperband: Vec<f64>,    // flattened rows * cols
    pub middleband: Vec<f64>,   // flattened rows * cols
    pub lowerband: Vec<f64>,    // flattened rows * cols
    pub combos: Vec<EmdParams>, // same shape as Rust
    pub rows: usize,            // combos.len()
    pub cols: usize,            // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "emd_batch")]
pub fn emd_batch_unified_js(
    high: &[f64],
    low: &[f64],
    _close: &[f64],
    _volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("high and low must have same length"));
    }

    let cfg: EmdBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = EmdBatchRange {
        period: cfg.period_range,
        delta: cfg.delta_range,
        fraction: cfg.fraction_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows_p = combos.len();
    let cols = high.len();
    let total = rows_p
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows * cols overflow in emd_batch_unified_js"))?;

    // allocate three planes
    let mut ub = vec![f64::NAN; total];
    let mut mb = vec![f64::NAN; total];
    let mut lb = vec![f64::NAN; total];

    emd_batch_inner_into(
        high,
        low,
        &sweep,
        detect_best_kernel(),
        false,
        &mut ub,
        &mut mb,
        &mut lb,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let out = EmdBatchJsOutput {
        upperband: ub,
        middleband: mb,
        lowerband: lb,
        combos,
        rows: rows_p,
        cols,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    upper_ptr: *mut f64,
    middle_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    delta_start: f64,
    delta_end: f64,
    delta_step: f64,
    fraction_start: f64,
    fraction_end: f64,
    fraction_step: f64,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || upper_ptr.is_null()
        || middle_ptr.is_null()
        || lower_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to emd_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = EmdBatchRange {
            period: (period_start, period_end, period_step),
            delta: (delta_start, delta_end, delta_step),
            fraction: (fraction_start, fraction_end, fraction_step),
        };

        // Calculate dimensions
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total_len = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows * cols overflow in emd_batch_into"))?;

        // Get output buffer slices
        let upper_slice = std::slice::from_raw_parts_mut(upper_ptr, total_len);
        let middle_slice = std::slice::from_raw_parts_mut(middle_ptr, total_len);
        let lower_slice = std::slice::from_raw_parts_mut(lower_ptr, total_len);

        // Use the optimized _inner_into function for zero-copy operation
        emd_batch_inner_into(
            high,
            low,
            &sweep,
            detect_best_kernel(),
            false,
            upper_slice,
            middle_slice,
            lower_slice,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
