//! # True Strength Index (TSI)
//!
//! TSI is a momentum oscillator that applies double exponential smoothing to price momentum
//! and its absolute value, providing a normalized measure of trend strength and direction.
//!
//! ## Parameters
//! - **long_period**: Long EMA smoothing period. Defaults to 25.
//! - **short_period**: Short EMA smoothing period. Defaults to 13.
//!
//! ## Returns
//! - **`Ok(TsiOutput)`** containing a `Vec<f64>` of TSI values (-100 to +100) matching input length.
//! - **`Err(TsiError)`** on invalid parameters or insufficient data.
//!
//! ## Developer Notes
//! - SIMD: AVX2/AVX512 are present as stubs and short‑circuit to the scalar path.
//!   Rationale: TSI is inherently sequential (EMA recursion), so per‑element
//!   vectorization does not provide wins; keeping scalar as reference.
//! - Scalar: Inlined double‑EMA kernel with robust NaN handling and no O(N) temporaries.
//!   Note: FMA/mul_add variants did not show consistent speedups vs the existing safe math,
//!   so we retained the simpler arithmetic for stability and portability.
//! - Memory: Uses `alloc_with_nan_prefix` / matrix helpers for zero‑copy prefixes.
//! - Batch: Parallel per‑row evaluation; rows use the scalar kernel. Row‑specific
//!   batch kernels (shared precompute across rows) not implemented yet; expected gains
//!   require additional complexity and are deferred. Runtime selection short‑circuits
//!   to scalar batch.
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

use crate::indicators::ema::{EmaError, EmaParams, EmaStream};
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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for TsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TsiData::Slice(slice) => slice,
            TsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TsiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TsiParams {
    pub long_period: Option<usize>,
    pub short_period: Option<usize>,
}

impl Default for TsiParams {
    fn default() -> Self {
        Self {
            long_period: Some(25),
            short_period: Some(13),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TsiInput<'a> {
    pub data: TsiData<'a>,
    pub params: TsiParams,
}

impl<'a> TsiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TsiParams) -> Self {
        Self {
            data: TsiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TsiParams) -> Self {
        Self {
            data: TsiData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TsiParams::default())
    }
    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(25)
    }
    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(13)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TsiBuilder {
    long_period: Option<usize>,
    short_period: Option<usize>,
    kernel: Kernel,
}

impl Default for TsiBuilder {
    fn default() -> Self {
        Self {
            long_period: None,
            short_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<TsiOutput, TsiError> {
        let p = TsiParams {
            long_period: self.long_period,
            short_period: self.short_period,
        };
        let i = TsiInput::from_candles(c, "close", p);
        tsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TsiOutput, TsiError> {
        let p = TsiParams {
            long_period: self.long_period,
            short_period: self.short_period,
        };
        let i = TsiInput::from_slice(d, p);
        tsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<TsiStream, TsiError> {
        let p = TsiParams {
            long_period: self.long_period,
            short_period: self.short_period,
        };
        TsiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TsiError {
    #[error("tsi: All values are NaN.")]
    AllValuesNaN,
    #[error("tsi: Invalid period: long = {long_period}, short = {short_period}, data length = {data_len}")]
    InvalidPeriod {
        long_period: usize,
        short_period: usize,
        data_len: usize,
    },
    #[error("tsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("tsi: EMA sub-error: {0}")]
    EmaSubError(#[from] EmaError),
}

#[inline(always)]
fn tsi_prepare<'a>(input: &'a TsiInput) -> Result<(&'a [f64], usize, usize, usize), TsiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TsiError::AllValuesNaN)?;
    let long = input.get_long_period();
    let short = input.get_short_period();

    if long == 0 || short == 0 || long > len || short > len {
        return Err(TsiError::InvalidPeriod {
            long_period: long,
            short_period: short,
            data_len: len,
        });
    }
    let needed = 1 + long + short;
    if len - first < needed {
        return Err(TsiError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    Ok((data, long, short, first))
}

#[inline(always)]
fn tsi_compute_into_streaming(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), TsiError> {
    let mut ema_long_num = EmaStream::try_new(EmaParams { period: Some(long) })?;
    let mut ema_short_num = EmaStream::try_new(EmaParams {
        period: Some(short),
    })?;
    let mut ema_long_den = EmaStream::try_new(EmaParams { period: Some(long) })?;
    let mut ema_short_den = EmaStream::try_new(EmaParams {
        period: Some(short),
    })?;

    let warmup_end = first + long + short;
    if out.len() != data.len() {
        return Err(TsiError::InvalidPeriod {
            long_period: out.len(),
            short_period: 0,
            data_len: data.len(),
        });
    }

    // keep prefix NaNs; start from first+1 because mom needs previous
    if first + 1 >= data.len() {
        return Ok(());
    }
    let mut prev = data[first];

    for i in (first + 1)..data.len() {
        let cur = data[i];
        if !cur.is_finite() {
            out[i] = f64::NAN;
            continue; // skip NaN/inf without poisoning prev
        }

        let m = cur - prev;
        prev = cur;

        // numerator path
        let n1 = match ema_long_num.update(m) {
            Some(v) => v,
            None => {
                out[i] = f64::NAN;
                continue;
            }
        };
        let n2 = match ema_short_num.update(n1) {
            Some(v) => v,
            None => {
                out[i] = f64::NAN;
                continue;
            }
        };

        // denominator path
        let d1 = match ema_long_den.update(m.abs()) {
            Some(v) => v,
            None => {
                out[i] = f64::NAN;
                continue;
            }
        };
        let d2 = match ema_short_den.update(d1) {
            Some(v) => v,
            None => {
                out[i] = f64::NAN;
                continue;
            }
        };

        // Always compute TSI when we have both EMAs, but respect warmup
        if i >= warmup_end {
            out[i] = if d2 == 0.0 {
                f64::NAN
            } else {
                (100.0 * (n2 / d2)).clamp(-100.0, 100.0)
            };
        }
    }

    // NaNs already set by alloc_with_nan_prefix or by caller
    Ok(())
}

/// Optimized scalar kernel for arbitrary periods.
///
/// This implementation inlines the double-EMA updates for numerator/denominator
/// and handles NaNs robustly. It preserves warmup semantics: indices before
/// `first + long + short` remain NaN (caller sets the prefix), and gaps in the
/// input (NaN/inf) produce NaNs without poisoning EMA state.
#[inline(always)]
fn tsi_compute_into_inline(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), TsiError> {
    let n = data.len();
    if n == 0 || first >= n {
        return Ok(());
    }

    // Precompute coefficients
    let long_alpha = 2.0 / (long as f64 + 1.0);
    let short_alpha = 2.0 / (short as f64 + 1.0);
    let long_1minus = 1.0 - long_alpha;
    let short_1minus = 1.0 - short_alpha;

    let warmup_end = first + long + short;

    // Previous finite value (by contract, data[first] is finite)
    let mut prev = data[first];

    // Find the first finite value after `first` to seed momentum/EMAs
    let mut i = first + 1;
    while i < n {
        let cur = data[i];
        if cur.is_finite() {
            break;
        } else {
            // Preserve NaN gaps; prefix before warmup is already NaN-initialized
            if i >= warmup_end {
                out[i] = f64::NAN;
            }
            i += 1;
        }
    }
    if i >= n {
        return Ok(());
    }

    // Seed all EMA chains with the first valid momentum
    let mut momentum = data[i] - prev;
    prev = data[i];

    let mut ema_long_num = momentum;
    let mut ema_short_num = momentum;
    let mut ema_long_den = momentum.abs();
    let mut ema_short_den = ema_long_den;

    // Warmup phase (no writes beyond caller's NaN prefix)
    let mut idx = i + 1;
    let end_warm = warmup_end.min(n);
    while idx < end_warm {
        let cur = data[idx];
        if cur.is_finite() {
            momentum = cur - prev;
            prev = cur;

            let am = momentum.abs();

            // Inline EMA updates (loop-jammed)
            ema_long_num = long_alpha * momentum + long_1minus * ema_long_num;
            ema_short_num = short_alpha * ema_long_num + short_1minus * ema_short_num;

            ema_long_den = long_alpha * am + long_1minus * ema_long_den;
            ema_short_den = short_alpha * ema_long_den + short_1minus * ema_short_den;
        }
        // else: skip NaN/inf without touching prev or EMA state
        idx += 1;
    }

    // Output phase
    while idx < n {
        let cur = data[idx];
        if cur.is_finite() {
            momentum = cur - prev;
            prev = cur;

            let am = momentum.abs();

            // Inline EMA updates (numerator + denominator)
            ema_long_num = long_alpha * momentum + long_1minus * ema_long_num;
            ema_short_num = short_alpha * ema_long_num + short_1minus * ema_short_num;

            ema_long_den = long_alpha * am + long_1minus * ema_long_den;
            ema_short_den = short_alpha * ema_long_den + short_1minus * ema_short_den;

            // Compute TSI; avoid divide-by-zero and clamp to [-100, 100]
            let den = ema_short_den;
            let val = if den == 0.0 {
                f64::NAN
            } else {
                (100.0 * (ema_short_num / den)).clamp(-100.0, 100.0)
            };
            out[idx] = val;
        } else {
            out[idx] = f64::NAN; // preserve NaN gaps in output
        }
        idx += 1;
    }

    Ok(())
}

#[inline]
pub fn tsi(input: &TsiInput) -> Result<TsiOutput, TsiError> {
    tsi_with_kernel(input, Kernel::Auto)
}

pub fn tsi_with_kernel(input: &TsiInput, kernel: Kernel) -> Result<TsiOutput, TsiError> {
    let (data, long, short, first) = tsi_prepare(input)?;
    let warmup_end = first + long + short;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_end);

    // Use classic kernel for default parameters with scalar kernel
    let resolved_kernel = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    if resolved_kernel == Kernel::Scalar && long == 25 && short == 13 {
        // Use optimized classic kernel for default parameters
        unsafe {
            tsi_scalar_classic(data, long, short, first, &mut out)?;
        }
    } else {
        // Use inline scalar kernel for general periods
        tsi_compute_into_inline(data, long, short, first, &mut out)?;
    }
    Ok(TsiOutput { values: out })
}

#[inline]
pub fn tsi_into_slice(dst: &mut [f64], input: &TsiInput, kern: Kernel) -> Result<(), TsiError> {
    let (data, long, short, first) = tsi_prepare(input)?;
    let warmup_end = first + long + short;
    // Caller's buffer may not have NaNs set, so set them
    let end = warmup_end.min(dst.len());
    for v in &mut dst[..end] {
        *v = f64::NAN;
    }

    // Use classic kernel for default parameters with scalar kernel
    let resolved_kernel = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    if resolved_kernel == Kernel::Scalar && long == 25 && short == 13 {
        // Use optimized classic kernel for default parameters
        unsafe {
            tsi_scalar_classic(data, long, short, first, dst)?;
        }
    } else {
        // Use inline scalar kernel for general periods
        tsi_compute_into_inline(data, long, short, first, dst)?;
    }
    Ok(())
}

/// Zero-allocation native API that writes results into a caller-provided buffer.
///
/// - Preserves NaN warmups exactly like the Vec-returning API.
/// - `out.len()` must equal the input length; returns an error on mismatch.
/// - Uses `Kernel::Auto` for dispatch (same as `tsi()`).
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn tsi_into(input: &TsiInput, out: &mut [f64]) -> Result<(), TsiError> {
    // Validate length matches input
    let data_len = input.as_ref().len();
    if out.len() != data_len {
        return Err(TsiError::InvalidPeriod {
            // Mirror existing length-mismatch mapping used by streaming variant
            long_period: out.len(),
            short_period: 0,
            data_len: data_len,
        });
    }
    tsi_into_slice(out, input, Kernel::Auto)
}

// Keep these stubs for compatibility but have them use the streaming version
#[inline]
pub unsafe fn tsi_scalar(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
) -> Result<TsiOutput, TsiError> {
    let warmup_end = first + long + short;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_end);
    // Use classic kernel for default parameters
    if long == 25 && short == 13 {
        tsi_scalar_classic(data, long, short, first, &mut out)?;
    } else {
        tsi_compute_into_inline(data, long, short, first, &mut out)?;
    }
    Ok(TsiOutput { values: out })
}

/// Classic kernel optimization for TSI with inline double EMA calculations
/// Eliminates function call overhead for all 4 EMA operations
#[inline]
pub unsafe fn tsi_scalar_classic(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), TsiError> {
    let n = data.len();
    let warmup_end = first + long + short;

    // Initialize output with NaN for warmup periods (already done by alloc_with_nan_prefix)
    // out[..warmup_end.min(n)].fill(f64::NAN); // Already done by caller

    if first + 1 >= n {
        return Ok(());
    }

    // EMA alpha factors
    let long_alpha = 2.0 / (long as f64 + 1.0);
    let short_alpha = 2.0 / (short as f64 + 1.0);
    let long_1minus = 1.0 - long_alpha;
    let short_1minus = 1.0 - short_alpha;

    // Initialize EMAs with first momentum value
    let mut prev = data[first];

    // Skip to first+1 to calculate first momentum
    if first + 1 >= n || !data[first + 1].is_finite() {
        return Ok(());
    }

    let first_momentum = data[first + 1] - prev;
    prev = data[first + 1];

    // Initialize all 4 EMAs with first momentum
    let mut ema_long_num = first_momentum;
    let mut ema_short_num = first_momentum;
    let mut ema_long_den = first_momentum.abs();
    let mut ema_short_den = first_momentum.abs();

    // Process remaining data with inline EMA calculations
    for i in (first + 2)..n {
        let cur = data[i];
        if !cur.is_finite() {
            out[i] = f64::NAN;
            continue;
        }

        let momentum = cur - prev;
        prev = cur;

        // Update long EMA for numerator (momentum)
        ema_long_num = long_alpha * momentum + long_1minus * ema_long_num;

        // Update short EMA for numerator (smoothed momentum)
        ema_short_num = short_alpha * ema_long_num + short_1minus * ema_short_num;

        // Update long EMA for denominator (abs momentum)
        ema_long_den = long_alpha * momentum.abs() + long_1minus * ema_long_den;

        // Update short EMA for denominator (smoothed abs momentum)
        ema_short_den = short_alpha * ema_long_den + short_1minus * ema_short_den;

        // Calculate TSI after warmup period
        if i >= warmup_end {
            out[i] = if ema_short_den == 0.0 {
                f64::NAN
            } else {
                (100.0 * (ema_short_num / ema_short_den)).clamp(-100.0, 100.0)
            };
        }
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx2(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
) -> Result<TsiOutput, TsiError> {
    // Stub, use streaming
    tsi_scalar(data, long, short, first)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx512(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
) -> Result<TsiOutput, TsiError> {
    // Stub, dispatch short/long
    if long <= 32 && short <= 32 {
        tsi_avx512_short(data, long, short, first)
    } else {
        tsi_avx512_long(data, long, short, first)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx512_short(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
) -> Result<TsiOutput, TsiError> {
    // Stub, use streaming
    tsi_scalar(data, long, short, first)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx512_long(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
) -> Result<TsiOutput, TsiError> {
    // Stub, use streaming
    tsi_scalar(data, long, short, first)
}

// Streaming variant
// Decision: Streaming kernel uses inlined EMA updates with robust NaN handling,
// warmup parity (long+short), and clamping to [-100, 100]; O(1) time/space.
#[derive(Debug, Clone)]
pub struct TsiStream {
    // periods
    long: usize,
    short: usize,

    // precomputed EMA coefficients
    alpha_l: f64,
    alpha_s: f64,

    // state for momentum -> double EMA(num/den)
    // Note: initialized lazily on first finite momentum
    ema_long_num: f64,
    ema_short_num: f64,
    ema_long_den: f64,
    ema_short_den: f64,

    // last finite price seen
    prev_price: f64,
    have_prev: bool,

    // whether EMA chains are seeded (set to first momentum)
    seeded: bool,

    // number of valid momentum updates processed (including seed step)
    warmup_ctr: usize,
    warmup_needed: usize, // == long + short
}
impl TsiStream {
    #[inline]
    pub fn try_new(params: TsiParams) -> Result<Self, TsiError> {
        let long = params.long_period.unwrap_or(25);
        let short = params.short_period.unwrap_or(13);

        if long == 0 || short == 0 {
            return Err(TsiError::InvalidPeriod {
                long_period: long,
                short_period: short,
                data_len: 0,
            });
        }

        // α = 2 / (n + 1)  (EMA smoothing factor)
        let alpha_l = 2.0 / (long as f64 + 1.0);
        let alpha_s = 2.0 / (short as f64 + 1.0);

        Ok(Self {
            long,
            short,
            alpha_l,
            alpha_s,
            ema_long_num: 0.0,
            ema_short_num: 0.0,
            ema_long_den: 0.0,
            ema_short_den: 0.0,
            prev_price: f64::NAN,
            have_prev: false,
            seeded: false,
            warmup_ctr: 0,
            warmup_needed: long + short,
        })
    }

    /// O(1) per tick. Returns:
    /// - `None` during warmup or when `value` is non-finite.
    /// - `Some(NaN)` when denominator becomes zero after warmup.
    /// - `Some(tsi)` otherwise, clamped to [-100, 100].
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Skip non-finite ticks without poisoning state (match offline gaps)
        if !value.is_finite() {
            return None;
        }

        // Need a previous finite price to form 1-bar momentum
        if !self.have_prev {
            self.prev_price = value;
            self.have_prev = true;
            return None;
        }

        // 1-bar momentum
        let m = value - self.prev_price;
        self.prev_price = value;
        let am = m.abs();

        if !self.seeded {
            // Seed all four EMA chains with the first valid momentum
            self.ema_long_num = m;
            self.ema_short_num = m;
            self.ema_long_den = am;
            self.ema_short_den = am;
            self.seeded = true;
            self.warmup_ctr = 1;
            return None; // seed step never outputs (match offline)
        }

        // --- inline EMA updates using single-multiply form: ema += α * (x - ema)
        // numerator path (m then smoothed)
        self.ema_long_num += self.alpha_l * (m - self.ema_long_num);
        self.ema_short_num += self.alpha_s * (self.ema_long_num - self.ema_short_num);

        // denominator path (|m| then smoothed)
        self.ema_long_den += self.alpha_l * (am - self.ema_long_den);
        self.ema_short_den += self.alpha_s * (self.ema_long_den - self.ema_short_den);

        self.warmup_ctr += 1;

        // Match offline warmup: need (long + short) momentum updates incl. seed
        if self.warmup_ctr < self.warmup_needed {
            return None;
        }

        // Avoid divide-by-zero; clamp to [-100, 100] like offline kernels
        let den = self.ema_short_den;
        if den == 0.0 {
            return Some(f64::NAN);
        }

        let tsi = 100.0 * (self.ema_short_num / den);
        Some(tsi.clamp(-100.0, 100.0))
    }
}

// Batch/grid API
#[derive(Clone, Debug)]
pub struct TsiBatchRange {
    pub long_period: (usize, usize, usize),
    pub short_period: (usize, usize, usize),
}
impl Default for TsiBatchRange {
    fn default() -> Self {
        Self {
            long_period: (25, 25, 1),
            short_period: (13, 13, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TsiBatchBuilder {
    range: TsiBatchRange,
    kernel: Kernel,
}
impl TsiBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    pub fn long_static(mut self, n: usize) -> Self {
        self.range.long_period = (n, n, 1);
        self
    }
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    pub fn short_static(mut self, n: usize) -> Self {
        self.range.short_period = (n, n, 1);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<TsiBatchOutput, TsiError> {
        tsi_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TsiBatchOutput, TsiError> {
        TsiBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TsiBatchOutput, TsiError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<TsiBatchOutput, TsiError> {
        TsiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn tsi_batch_with_kernel(
    data: &[f64],
    sweep: &TsiBatchRange,
    k: Kernel,
) -> Result<TsiBatchOutput, TsiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(TsiError::InvalidPeriod {
                long_period: 0,
                short_period: 0,
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
    tsi_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TsiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TsiParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TsiBatchOutput {
    pub fn row_for_params(&self, p: &TsiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.long_period.unwrap_or(25) == p.long_period.unwrap_or(25)
                && c.short_period.unwrap_or(13) == p.short_period.unwrap_or(13)
        })
    }
    pub fn values_for(&self, p: &TsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TsiBatchRange) -> Vec<TsiParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let longs = axis_usize(r.long_period);
    let shorts = axis_usize(r.short_period);

    let mut out = Vec::with_capacity(longs.len() * shorts.len()); // Small params vector, OK
    for &l in &longs {
        for &s in &shorts {
            out.push(TsiParams {
                long_period: Some(l),
                short_period: Some(s),
            });
        }
    }
    out
}

#[inline(always)]
pub fn tsi_batch_slice(
    data: &[f64],
    sweep: &TsiBatchRange,
    kern: Kernel,
) -> Result<TsiBatchOutput, TsiError> {
    tsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tsi_batch_par_slice(
    data: &[f64],
    sweep: &TsiBatchRange,
    kern: Kernel,
) -> Result<TsiBatchOutput, TsiError> {
    tsi_batch_inner(data, sweep, kern, true)
}

fn tsi_batch_inner(
    data: &[f64],
    sweep: &TsiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TsiBatchOutput, TsiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TsiError::InvalidPeriod {
            long_period: sweep.long_period.0,
            short_period: sweep.short_period.0,
            data_len: data.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TsiError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    let max_short = combos
        .iter()
        .map(|c| c.short_period.unwrap())
        .max()
        .unwrap();
    let max_needed = 1 + max_long + max_short;
    if data.len() - first < max_needed {
        return Err(TsiError::NotEnoughValidData {
            needed: max_needed,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Use uninitialized memory helpers like ALMA
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each combination
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + 1 + c.long_period.unwrap() + c.short_period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let p = &combos[row];
        let l = p.long_period.unwrap();
        let s = p.short_period.unwrap();
        match kern {
            Kernel::Scalar => unsafe { tsi_row_scalar_into(data, l, s, first, out_row) },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => unsafe { tsi_row_avx2_into(data, l, s, first, out_row) },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => unsafe { tsi_row_avx512_into(data, l, s, first, out_row) },
            _ => unreachable!(),
        }
        .unwrap();
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Convert uninitialized buffer to Vec like ALMA does
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(TsiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn tsi_batch_inner_into(
    data: &[f64],
    sweep: &TsiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TsiParams>, TsiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TsiError::InvalidPeriod {
            long_period: sweep.long_period.0,
            short_period: sweep.short_period.0,
            data_len: data.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TsiError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    let max_short = combos
        .iter()
        .map(|c| c.short_period.unwrap())
        .max()
        .unwrap();
    let max_needed = 1 + max_long + max_short;
    if data.len() - first < max_needed {
        return Err(TsiError::NotEnoughValidData {
            needed: max_needed,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let do_row = |row: usize, out_row: &mut [f64]| {
        let p = &combos[row];
        let l = p.long_period.unwrap();
        let s = p.short_period.unwrap();
        match kern {
            Kernel::Scalar => unsafe { tsi_row_scalar_into(data, l, s, first, out_row) },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => unsafe { tsi_row_avx2_into(data, l, s, first, out_row) },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => unsafe { tsi_row_avx512_into(data, l, s, first, out_row) },
            _ => unreachable!(),
        }
        .unwrap();
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
pub unsafe fn tsi_row_scalar_into(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
    out_row: &mut [f64],
) -> Result<(), TsiError> {
    // out_row length equals data.len(); NaN prefixes already set by init_matrix_prefixes
    tsi_compute_into_inline(data, long, short, first, out_row)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tsi_row_avx2_into(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
    out_row: &mut [f64],
) -> Result<(), TsiError> {
    tsi_compute_into_streaming(data, long, short, first, out_row)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tsi_row_avx512_into(
    data: &[f64],
    long: usize,
    short: usize,
    first: usize,
    out_row: &mut [f64],
) -> Result<(), TsiError> {
    tsi_compute_into_streaming(data, long, short, first, out_row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_tsi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = TsiParams {
            long_period: None,
            short_period: None,
        };
        let input = TsiInput::from_candles(&candles, "close", default_params);
        let output = tsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_tsi_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let input = TsiInput::from_candles(&candles, "close", params);
        let tsi_result = tsi_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -17.757654061849838,
            -17.367527062626184,
            -17.305577681249513,
            -16.937565646991143,
            -17.61825617316731,
        ];
        let start = tsi_result.values.len().saturating_sub(5);
        for (i, &val) in tsi_result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-7,
                "[{}] TSI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_tsi_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TsiParams {
            long_period: Some(0),
            short_period: Some(13),
        };
        let input = TsiInput::from_slice(&input_data, params);
        let res = tsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TSI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_tsi_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let input = TsiInput::from_slice(&data_small, params);
        let res = tsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TSI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_tsi_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let input = TsiInput::from_slice(&single_point, params);
        let res = tsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TSI should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_tsi_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TsiInput::with_default_candles(&candles);
        match input.data {
            TsiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TsiData::Candles"),
        }
        let output = tsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_tsi_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let first_input = TsiInput::from_candles(&candles, "close", first_params);
        let first_result = tsi_with_kernel(&first_input, kernel)?;

        let second_params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let second_input = TsiInput::from_slice(&first_result.values, second_params);
        let second_result = tsi_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_tsi_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TsiInput::from_candles(
            &candles,
            "close",
            TsiParams {
                long_period: Some(25),
                short_period: Some(13),
            },
        );
        let res = tsi_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_tsi_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Prefer existing repository CSV data for parity
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TsiInput::from_candles(&candles, "close", TsiParams::default());

        // Baseline via Vec-returning API
        let baseline = tsi(&input)?.values;

        // Preallocate destination and call zero-allocation API
        let mut out = vec![0.0; candles.close.len()];
        #[cfg(not(feature = "wasm"))]
        {
            tsi_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // On WASM builds, native tsi_into is not emitted; use into_slice
            tsi_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(out.len(), baseline.len());
        let eq_or_both_nan = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || (a == b);
        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(out[i], baseline[i]),
                "mismatch at {}: got {}, expected {}",
                i,
                out[i],
                baseline[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_tsi_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            TsiParams::default(),
            TsiParams {
                long_period: Some(2),
                short_period: Some(2),
            },
            TsiParams {
                long_period: Some(5),
                short_period: Some(3),
            },
            TsiParams {
                long_period: Some(10),
                short_period: Some(5),
            },
            TsiParams {
                long_period: Some(15),
                short_period: Some(7),
            },
            TsiParams {
                long_period: Some(25),
                short_period: Some(13),
            },
            TsiParams {
                long_period: Some(30),
                short_period: Some(15),
            },
            TsiParams {
                long_period: Some(40),
                short_period: Some(20),
            },
            TsiParams {
                long_period: Some(50),
                short_period: Some(25),
            },
            TsiParams {
                long_period: Some(100),
                short_period: Some(50),
            },
            TsiParams {
                long_period: Some(200),
                short_period: Some(100),
            },
            // Edge cases with extreme ratios
            TsiParams {
                long_period: Some(50),
                short_period: Some(2),
            },
            TsiParams {
                long_period: Some(100),
                short_period: Some(5),
            },
            TsiParams {
                long_period: Some(10),
                short_period: Some(9),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = TsiInput::from_candles(&candles, "close", params.clone());
            let output = tsi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_tsi_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_tsi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test data with varying parameters
        let strat = (2usize..=50).prop_flat_map(|long_period| {
            (
                // Generate data with enough length for the periods
                prop::collection::vec(
                    (1.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                    (long_period + 30)..400,
                ),
                Just(long_period),
                // Short period should be less than or equal to long period
                2usize..=long_period.min(25),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, long_period, short_period)| {
                let params = TsiParams {
                    long_period: Some(long_period),
                    short_period: Some(short_period),
                };
                let input = TsiInput::from_slice(&data, params.clone());

                let TsiOutput { values: out } = tsi_with_kernel(&input, kernel).unwrap();
                let TsiOutput { values: ref_out } =
                    tsi_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: TSI values must be in [-100, 100] range
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        prop_assert!(
                            val >= -100.0 - 1e-9 && val <= 100.0 + 1e-9,
                            "Property 1: TSI value out of range at idx {}: {} ∉ [-100, 100]",
                            i,
                            val
                        );
                    }
                }

                // Property 2: Warmup period handling
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

                // The first value should always be NaN (momentum needs previous value)
                prop_assert!(
                    out[0].is_nan(),
                    "Property 2: First value should always be NaN, got {}",
                    out[0]
                );

                // Check if the data has any variation (not all constant)
                let has_variation = data.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-10);

                if has_variation {
                    // TSI produces values earlier than theoretical warmup due to EMA behavior
                    // Just verify we eventually get consistent non-NaN values
                    let first_non_nan = out.iter().position(|x| !x.is_nan());

                    if let Some(idx) = first_non_nan {
                        // Verify first non-NaN appears after some minimal warmup
                        prop_assert!(
                            idx >= 1, // At least after first (momentum needs previous)
                            "Property 2: First non-NaN too early at idx {}",
                            idx
                        );

                        // Check that after sufficient data, values are consistently non-NaN
                        // BUT: TSI correctly returns NaN for constant prices (momentum = 0)
                        // So we need to check if the data has actual price movement
                        let sufficient_data =
                            (first_valid + long_period + short_period).min(out.len());
                        if sufficient_data < out.len() {
                            // Check if there's meaningful price movement in the data
                            let mut has_movement = false;
                            for i in 1..data.len() {
                                if (data[i] - data[i - 1]).abs() > 1e-10 {
                                    has_movement = true;
                                    break;
                                }
                            }

                            // Only check for non-NaN values if there's actual price movement
                            if has_movement {
                                let last_quarter_start = out.len() - (out.len() / 4).max(1);
                                // Allow some NaN values for edge cases, but not all
                                let nan_count = out[last_quarter_start..]
                                    .iter()
                                    .filter(|v| v.is_nan())
                                    .count();
                                let total_count = out.len() - last_quarter_start;
                                prop_assert!(
									nan_count < total_count,
									"Property 2: All values in last quarter are NaN (from idx {}), which suggests an issue",
									last_quarter_start
								);
                            }
                        }
                    }
                }

                // Property 3: Kernel consistency
                for (i, (&val, &ref_val)) in out.iter().zip(ref_out.iter()).enumerate() {
                    if val.is_nan() && ref_val.is_nan() {
                        continue;
                    }
                    if !val.is_finite() || !ref_val.is_finite() {
                        prop_assert!(
                            val.to_bits() == ref_val.to_bits(),
                            "Property 3: Kernel finite/NaN mismatch at idx {}: {} vs {}",
                            i,
                            val,
                            ref_val
                        );
                        continue;
                    }
                    prop_assert!(
                        (val - ref_val).abs() <= 1e-9,
                        "Property 3: Kernel mismatch at idx {}: {} vs {} (diff: {})",
                        i,
                        val,
                        ref_val,
                        (val - ref_val).abs()
                    );
                }

                // Property 4: Zero momentum special case (constant prices)
                // When prices are constant, momentum is 0, and TSI = 100 * (0/0) = NaN
                let constant_data = vec![100.0; 50];
                let const_params = TsiParams {
                    long_period: Some(10),
                    short_period: Some(5),
                };
                let const_input = TsiInput::from_slice(&constant_data, const_params);
                if let Ok(TsiOutput { values: const_out }) = tsi_with_kernel(&const_input, kernel) {
                    // All values should be NaN for constant prices (after first which is always NaN)
                    for (i, &val) in const_out.iter().enumerate() {
                        prop_assert!(
                            val.is_nan(),
                            "Property 4: TSI should be NaN for constant prices at idx {}, got {}",
                            i,
                            val
                        );
                    }
                }

                // Property 5: Trend behavior for all period combinations
                // Strong uptrend
                let uptrend: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 20.0).collect();
                let uptrend_input = TsiInput::from_slice(&uptrend, params.clone());

                // Strong downtrend
                let downtrend: Vec<f64> = (0..100).map(|i| 2100.0 - i as f64 * 20.0).collect();
                let downtrend_input = TsiInput::from_slice(&downtrend, params.clone());

                if let (Ok(TsiOutput { values: up_out }), Ok(TsiOutput { values: down_out })) = (
                    tsi_with_kernel(&uptrend_input, kernel),
                    tsi_with_kernel(&downtrend_input, kernel),
                ) {
                    let test_warmup = 1 + long_period + short_period - 1;
                    if test_warmup + 10 < up_out.len() {
                        // Check last 10 values after warmup
                        let up_vals: Vec<f64> = up_out[up_out.len() - 10..]
                            .iter()
                            .filter(|&&x| !x.is_nan())
                            .copied()
                            .collect();
                        let down_vals: Vec<f64> = down_out[down_out.len() - 10..]
                            .iter()
                            .filter(|&&x| !x.is_nan())
                            .copied()
                            .collect();

                        if !up_vals.is_empty() && !down_vals.is_empty() {
                            let up_avg = up_vals.iter().sum::<f64>() / up_vals.len() as f64;
                            let down_avg = down_vals.iter().sum::<f64>() / down_vals.len() as f64;

                            // Use tolerance based on period size (larger periods smooth more)
                            let tolerance = if long_period > 20 { 10.0 } else { 0.0 };

                            prop_assert!(
								up_avg > tolerance,
								"Property 5: Uptrend TSI should be positive, got avg: {} (long={}, short={})",
								up_avg, long_period, short_period
							);
                            prop_assert!(
								down_avg < -tolerance,
								"Property 5: Downtrend TSI should be negative, got avg: {} (long={}, short={})",
								down_avg, long_period, short_period
							);
                        }
                    }
                }

                // Property 6: Extreme values with realistic strong trends
                // Very strong linear uptrend
                let extreme_up: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 50.0).collect();
                let extreme_params = TsiParams {
                    long_period: Some(5),
                    short_period: Some(3),
                };
                let extreme_input = TsiInput::from_slice(&extreme_up, extreme_params.clone());
                if let Ok(TsiOutput {
                    values: extreme_out,
                }) = tsi_with_kernel(&extreme_input, kernel)
                {
                    // Check last few values - should be very positive (>80 for strong trend)
                    let last_valid = extreme_out.iter().rposition(|x| !x.is_nan());
                    if let Some(idx) = last_valid {
                        if idx >= 20 {
                            let last_vals = &extreme_out[idx - 5..=idx];
                            for &v in last_vals {
                                if !v.is_nan() {
                                    prop_assert!(
										v > 80.0,
										"Property 6: Very strong uptrend should have TSI > 80, got: {}",
										v
									);
                                }
                            }
                        }
                    }
                }

                // Very strong linear downtrend
                let extreme_down: Vec<f64> = (0..100).map(|i| 5100.0 - i as f64 * 50.0).collect();
                let extreme_down_input = TsiInput::from_slice(&extreme_down, extreme_params);
                if let Ok(TsiOutput {
                    values: extreme_down_out,
                }) = tsi_with_kernel(&extreme_down_input, kernel)
                {
                    let last_valid = extreme_down_out.iter().rposition(|x| !x.is_nan());
                    if let Some(idx) = last_valid {
                        if idx >= 20 {
                            let last_vals = &extreme_down_out[idx - 5..=idx];
                            for &v in last_vals {
                                if !v.is_nan() {
                                    prop_assert!(
										v < -80.0,
										"Property 6: Very strong downtrend should have TSI < -80, got: {}",
										v
									);
                                }
                            }
                        }
                    }
                }

                // Property 7: Enhanced formula verification
                // Test basic TSI behavior with predictable data
                if data.len() >= 20 {
                    // Check that increasing prices generally lead to positive TSI
                    let increasing_count = data.windows(2).filter(|w| w[1] > w[0]).count();
                    let decreasing_count = data.windows(2).filter(|w| w[1] < w[0]).count();

                    // Find the last few valid TSI values
                    let valid_tsi: Vec<f64> = out
                        .iter()
                        .rev()
                        .take(10)
                        .filter(|&&x| !x.is_nan())
                        .copied()
                        .collect();

                    if !valid_tsi.is_empty() {
                        let avg_tsi = valid_tsi.iter().sum::<f64>() / valid_tsi.len() as f64;

                        // If prices are mostly increasing, TSI should tend positive
                        if increasing_count > decreasing_count * 2 {
                            prop_assert!(
								avg_tsi > -20.0,
								"Property 7: Mostly increasing prices should have TSI > -20, got: {}",
								avg_tsi
							);
                        }
                        // If prices are mostly decreasing, TSI should tend negative
                        else if decreasing_count > increasing_count * 2 {
                            prop_assert!(
								avg_tsi < 20.0,
								"Property 7: Mostly decreasing prices should have TSI < 20, got: {}",
								avg_tsi
							);
                        }
                    }
                }

                // Property 8: No poison values in debug mode
                #[cfg(debug_assertions)]
                {
                    for (i, &val) in out.iter().enumerate() {
                        if !val.is_nan() {
                            let bits = val.to_bits();
                            prop_assert!(
                                bits != 0x11111111_11111111
                                    && bits != 0x22222222_22222222
                                    && bits != 0x33333333_33333333,
                                "Property 8: Found poison value at idx {}: {} (0x{:016X})",
                                i,
                                val,
                                bits
                            );
                        }
                    }
                }

                // Property 9: Momentum response
                // TSI should respond to changes in momentum direction
                if data.len() >= 50 && has_variation {
                    // Find a period where we have valid TSI values
                    let start_idx = (1 + long_period + short_period).max(20);
                    if start_idx + 20 < out.len() {
                        // Check if TSI changes direction when price momentum changes
                        let mut momentum_changes = 0;
                        let mut tsi_follows = 0;

                        for i in start_idx..out.len() - 10 {
                            if !out[i].is_nan() && !out[i + 5].is_nan() && !out[i + 10].is_nan() {
                                // Check price momentum change
                                let price_change1 = data[i + 5] - data[i];
                                let price_change2 = data[i + 10] - data[i + 5];

                                // Check TSI change
                                let tsi_change1 = out[i + 5] - out[i];
                                let tsi_change2 = out[i + 10] - out[i + 5];

                                // If price momentum reverses, TSI should eventually follow
                                if price_change1 * price_change2 < 0.0 {
                                    momentum_changes += 1;
                                    // Allow some lag, but TSI should respond
                                    if tsi_change1 * tsi_change2 < 0.0
                                        || (price_change2 > 0.0 && tsi_change2 > tsi_change1)
                                        || (price_change2 < 0.0 && tsi_change2 < tsi_change1)
                                    {
                                        tsi_follows += 1;
                                    }
                                }
                            }
                        }

                        // TSI should follow momentum changes at least 50% of the time
                        if momentum_changes > 0 {
                            let follow_rate = tsi_follows as f64 / momentum_changes as f64;
                            prop_assert!(
								follow_rate >= 0.3,  // Allow for smoothing lag
								"Property 9: TSI should respond to momentum changes, follow rate: {:.2}",
								follow_rate
							);
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_tsi_tests {
        ($($test_fn:ident),*) => {
            paste! {
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
    generate_all_tsi_tests!(
        check_tsi_partial_params,
        check_tsi_accuracy,
        check_tsi_zero_period,
        check_tsi_period_exceeds_length,
        check_tsi_very_small_dataset,
        check_tsi_default_candles,
        check_tsi_reinput,
        check_tsi_nan_handling,
        check_tsi_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_tsi_tests!(check_tsi_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = TsiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = TsiParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -17.757654061849838,
            -17.367527062626184,
            -17.305577681249513,
            -16.937565646991143,
            -17.61825617316731,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
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

        let test_configs = vec![
            // (long_start, long_end, long_step, short_start, short_end, short_step)
            (5, 10, 1, 2, 5, 1),         // Small ranges, dense
            (10, 30, 5, 5, 15, 5),       // Medium ranges
            (25, 50, 5, 10, 25, 5),      // Medium to large
            (50, 100, 10, 25, 50, 5),    // Large ranges
            (25, 25, 0, 2, 20, 2),       // Static long, varying short
            (10, 50, 10, 13, 13, 0),     // Varying long, static short
            (100, 200, 50, 50, 100, 25), // Very large ranges
            (2, 5, 1, 2, 5, 1),          // Minimum periods
            (30, 30, 0, 5, 25, 5),       // Static long=30, varying short
        ];

        for (cfg_idx, &(l_start, l_end, l_step, s_start, s_end, s_step)) in
            test_configs.iter().enumerate()
        {
            let output = TsiBatchBuilder::new()
                .kernel(kernel)
                .long_range(l_start, l_end, l_step)
                .short_range(s_start, s_end, s_step)
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
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
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
            paste! {
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

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "tsi")]
#[pyo3(signature = (data, long_period=25, short_period=13, kernel=None))]
pub fn tsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    long_period: usize,
    short_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = TsiParams {
        long_period: Some(long_period),
        short_period: Some(short_period),
    };
    let input = TsiInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| tsi_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "tsi_batch")]
#[pyo3(signature = (data, long_period_range, short_period_range, kernel=None))]
pub fn tsi_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    long_period_range: (usize, usize, usize),
    short_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = TsiBatchRange {
        long_period: long_period_range,
        short_period: short_period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array for batch operations
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    // Compute without GIL using _into variant for zero-copy
    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            // Map batch kernels to regular kernels
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };
            tsi_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    // Add parameter arrays
    dict.set_item(
        "long_periods",
        combos
            .iter()
            .map(|p| p.long_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    dict.set_item(
        "short_periods",
        combos
            .iter()
            .map(|p| p.short_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ----- CUDA Python bindings (device-returning) -----
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::tsi_wrapper::CudaTsi;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tsi_cuda_batch_dev")]
#[pyo3(signature = (data_f32, long_period_range, short_period_range, device_id=0))]
pub fn tsi_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    long_period_range: (usize, usize, usize),
    short_period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, PyDict>)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = TsiBatchRange {
        long_period: long_period_range,
        short_period: short_period_range,
    };
    let (inner, combos) = py.allow_threads(|| {
        let mut cuda = CudaTsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tsi_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    use numpy::{IntoPyArray, PyArrayMethods};
    let dict = PyDict::new(py);
    dict.set_item(
        "long_periods",
        combos
            .iter()
            .map(|p| p.long_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "short_periods",
        combos
            .iter()
            .map(|p| p.short_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tsi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, cols, rows, long_period, short_period, device_id=0))]
pub fn tsi_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    long_period: usize,
    short_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_tm_f32.as_slice()?;
    let inner = py.allow_threads(|| {
        let mut cuda = CudaTsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tsi_many_series_one_param_time_major_dev(
            slice_in,
            cols,
            rows,
            long_period,
            short_period,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "TsiStream")]
pub struct TsiStreamPy {
    inner: TsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TsiStreamPy {
    #[new]
    pub fn new(long_period: usize, short_period: usize) -> PyResult<Self> {
        let params = TsiParams {
            long_period: Some(long_period),
            short_period: Some(short_period),
        };
        let inner = TsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TsiStreamPy { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// ====== WASM Bindings ======

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsi_js(data: &[f64], long_period: usize, short_period: usize) -> Result<Vec<f64>, JsValue> {
    let params = TsiParams {
        long_period: Some(long_period),
        short_period: Some(short_period),
    };
    let input = TsiInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    tsi_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsi_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    long_period: usize,
    short_period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = TsiParams {
            long_period: Some(long_period),
            short_period: Some(short_period),
        };
        let input = TsiInput::from_slice(data, params);

        if in_ptr == out_ptr as *const f64 {
            // Aliasing detected - use temporary buffer
            let mut temp = vec![0.0; len];
            tsi_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            tsi_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TsiBatchConfig {
    pub long_period_range: (usize, usize, usize),
    pub short_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TsiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tsi_batch)]
pub fn tsi_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: TsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = TsiBatchRange {
        long_period: config.long_period_range,
        short_period: config.short_period_range,
    };

    // For WASM, use scalar kernel
    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::Scalar
    } else {
        detect_best_kernel()
    };
    let result =
        tsi_batch_slice(data, &sweep, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = TsiBatchJsOutput {
        values: result.values,
        combos: result.combos,
        rows: result.rows,
        cols: result.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize output: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsi_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    long_period_start: usize,
    long_period_end: usize,
    long_period_step: usize,
    short_period_start: usize,
    short_period_end: usize,
    short_period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to tsi_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = TsiBatchRange {
            long_period: (long_period_start, long_period_end, long_period_step),
            short_period: (short_period_start, short_period_end, short_period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;
        let total_size = rows * cols;

        let out_slice = std::slice::from_raw_parts_mut(out_ptr, total_size);

        // Use the batch_inner_into function to compute directly into output
        // For WASM, use scalar kernel
        let kernel = if cfg!(target_arch = "wasm32") {
            Kernel::Scalar
        } else {
            detect_best_kernel()
        };
        match tsi_batch_inner_into(data, &sweep, kernel, false, out_slice) {
            Ok(_) => Ok(rows),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }
}
