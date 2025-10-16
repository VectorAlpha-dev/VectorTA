//! # Kaufmanstop Indicator
//!
//! Adaptive price stop based on the average true range of price (high-low).
//! Calculates stops above or below price using MA of range multiplied by a factor.
//!
//! ## Parameters
//! - **period**: Window size for range average (default: 22)
//! - **mult**: Multiplier for averaged range (default: 2.0)
//! - **direction**: "long" (stop below) or "short" (stop above) (default: "long")
//! - **ma_type**: Moving average type for range (default: "sma")
//!
//! ## Returns
//! - **`Ok(KaufmanstopOutput)`** on success (`values: Vec<f64>` of length matching input)
//! - **`Err(KaufmanstopError)`** on failure
//!
//! ## Developer Status
//! - **SIMD Kernels**: AVX2/AVX512 enabled (AXPY-style) — disabled by default for Auto since scalar is as fast or faster on typical CPUs (memory-bound). Use explicit Avx2/Avx512 kernels only for experimentation.
//! - **Row-Specific Batch**: Not implemented; MA cost dominates and AXPY write per row is cheap. Potential future work: group by (ma_type, period) and reuse `range_ma` across rows.
//! - **Streaming**: O(1) for SMA/EMA; Other MA types fall back to O(n) with correct chronological order
//! - **Memory**: Good zero-copy usage (alloc_with_nan_prefix, make_uninit_matrix)
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
// aligned_vec not needed for this indicator
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum KaufmanstopData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct KaufmanstopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct KaufmanstopParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub direction: Option<String>,
    pub ma_type: Option<String>,
}

impl Default for KaufmanstopParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(2.0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KaufmanstopInput<'a> {
    pub data: KaufmanstopData<'a>,
    pub params: KaufmanstopParams,
}

impl<'a> KaufmanstopInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: KaufmanstopParams) -> Self {
        Self {
            data: KaufmanstopData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: KaufmanstopParams) -> Self {
        Self {
            data: KaufmanstopData::Slices { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, KaufmanstopParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(22)
    }
    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(2.0)
    }
    #[inline]
    pub fn get_direction(&self) -> &str {
        self.params.direction.as_deref().unwrap_or("long")
    }
    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("sma")
    }
}

impl<'a> AsRef<[f64]> for KaufmanstopInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        // Since kaufmanstop uses two data series (high and low),
        // we'll return the high series as the primary reference
        match &self.data {
            KaufmanstopData::Candles { candles } => {
                candles.select_candle_field("high").unwrap_or(&[])
            }
            KaufmanstopData::Slices { high, .. } => high,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KaufmanstopBuilder {
    period: Option<usize>,
    mult: Option<f64>,
    direction: Option<String>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for KaufmanstopBuilder {
    fn default() -> Self {
        Self {
            period: None,
            mult: None,
            direction: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KaufmanstopBuilder {
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
    pub fn mult(mut self, m: f64) -> Self {
        self.mult = Some(m);
        self
    }
    #[inline(always)]
    pub fn direction<S: Into<String>>(mut self, d: S) -> Self {
        self.direction = Some(d.into());
        self
    }
    #[inline(always)]
    pub fn ma_type<S: Into<String>>(mut self, m: S) -> Self {
        self.ma_type = Some(m.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KaufmanstopOutput, KaufmanstopError> {
        let p = KaufmanstopParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
            ma_type: self.ma_type,
        };
        let i = KaufmanstopInput::from_candles(c, p);
        kaufmanstop_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<KaufmanstopOutput, KaufmanstopError> {
        let p = KaufmanstopParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
            ma_type: self.ma_type,
        };
        let i = KaufmanstopInput::from_slices(high, low, p);
        kaufmanstop_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<KaufmanstopStream, KaufmanstopError> {
        let p = KaufmanstopParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
            ma_type: self.ma_type,
        };
        KaufmanstopStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum KaufmanstopError {
    #[error("kaufmanstop: Empty data provided.")]
    EmptyData,
    #[error("kaufmanstop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kaufmanstop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kaufmanstop: All values are NaN.")]
    AllValuesNaN,
}

#[cfg(feature = "wasm")]
impl From<KaufmanstopError> for JsValue {
    fn from(err: KaufmanstopError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

#[inline]
pub fn kaufmanstop(input: &KaufmanstopInput) -> Result<KaufmanstopOutput, KaufmanstopError> {
    kaufmanstop_with_kernel(input, Kernel::Auto)
}

pub fn kaufmanstop_with_kernel(
    input: &KaufmanstopInput,
    kernel: Kernel,
) -> Result<KaufmanstopOutput, KaufmanstopError> {
    let (high, low) = match &input.data {
        KaufmanstopData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| KaufmanstopError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| KaufmanstopError::EmptyData)?;
            (high, low)
        }
        KaufmanstopData::Slices { high, low } => {
            if high.is_empty() || low.is_empty() {
                return Err(KaufmanstopError::EmptyData);
            }
            (*high, *low)
        }
    };

    if high.is_empty() || low.is_empty() {
        return Err(KaufmanstopError::EmptyData);
    }

    let period = input.get_period();
    let mult = input.get_mult();
    let direction = input.get_direction();
    let ma_type = input.get_ma_type();

    if period == 0 || period > high.len() || period > low.len() {
        return Err(KaufmanstopError::InvalidPeriod {
            period,
            data_len: high.len().min(low.len()),
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(KaufmanstopError::AllValuesNaN)?;

    if (high.len() - first_valid_idx) < period {
        return Err(KaufmanstopError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    // Use alloc_with_nan_prefix instead of vec![f64::NAN; ...]
    let mut out = alloc_with_nan_prefix(high.len(), first_valid_idx + period - 1);

    // Dispatch to classic kernels for common MA types
    if ma_type == "sma" || ma_type == "SMA" {
        unsafe {
            kaufmanstop_scalar_classic_sma(
                high,
                low,
                period,
                first_valid_idx,
                mult,
                direction,
                &mut out,
            )?;
        }
        return Ok(KaufmanstopOutput { values: out });
    } else if ma_type == "ema" || ma_type == "EMA" {
        unsafe {
            kaufmanstop_scalar_classic_ema(
                high,
                low,
                period,
                first_valid_idx,
                mult,
                direction,
                &mut out,
            )?;
        }
        return Ok(KaufmanstopOutput { values: out });
    }

    // Original implementation (fallback for other MA types)
    // Use helper function to allocate with NaN prefix
    let mut hl_diff = alloc_with_nan_prefix(high.len(), first_valid_idx);
    for i in first_valid_idx..high.len() {
        if high[i].is_nan() || low[i].is_nan() {
            hl_diff[i] = f64::NAN;
        } else {
            hl_diff[i] = high[i] - low[i];
        }
    }

    let ma_input = MaData::Slice(&hl_diff[first_valid_idx..]);
    let hl_diff_ma = ma(ma_type, ma_input, period).map_err(|_| KaufmanstopError::AllValuesNaN)?;

    // SIMD underperforms vs scalar (memory-bound AXPY); default to Scalar for Auto.
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kaufmanstop_scalar(
                high,
                low,
                &hl_diff_ma,
                period,
                first_valid_idx,
                mult,
                direction,
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kaufmanstop_avx2(
                high,
                low,
                &hl_diff_ma,
                period,
                first_valid_idx,
                mult,
                direction,
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kaufmanstop_avx512(
                high,
                low,
                &hl_diff_ma,
                period,
                first_valid_idx,
                mult,
                direction,
                &mut out,
            ),
            _ => unreachable!(),
        }
    }
    Ok(KaufmanstopOutput { values: out })
}

#[inline]
pub fn kaufmanstop_scalar(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(out.len(), high.len());

    if first >= high.len() {
        return;
    }

    // Select base series and multiplier sign once (avoid per-iteration branch)
    let is_long = direction.eq_ignore_ascii_case("long");
    let base = if is_long { low } else { high };
    let signed_mult = if is_long { -mult } else { mult };

    // Number of outputs we can write safely
    let n = core::cmp::min(range_ma.len(), high.len() - first);
    if n == 0 {
        return;
    }

    // Slice once to eliminate bounds checks in the hot loop
    let rm = &range_ma[..n];
    let base_s = &base[first..first + n];
    let out_s = &mut out[first..first + n];

    // Process in chunks of 4 for a touch of unrolling; tail handled after
    let chunks = rm.len() & !3usize;
    let mut i = 0usize;
    while i < chunks {
        out_s[i + 0] = base_s[i + 0] + rm[i + 0] * signed_mult;
        out_s[i + 1] = base_s[i + 1] + rm[i + 1] * signed_mult;
        out_s[i + 2] = base_s[i + 2] + rm[i + 2] * signed_mult;
        out_s[i + 3] = base_s[i + 3] + rm[i + 3] * signed_mult;
        i += 4;
    }
    while i < rm.len() {
        out_s[i] = base_s[i] + rm[i] * signed_mult;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kaufmanstop_avx2(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(out.len(), high.len());
    if first >= high.len() {
        return;
    }

    let is_long = direction.eq_ignore_ascii_case("long");
    let base = if is_long { low } else { high };
    let signed_mult = if is_long { -mult } else { mult };

    let n = core::cmp::min(range_ma.len(), high.len() - first);
    if n == 0 {
        return;
    }

    unsafe {
        let base_ptr = base.as_ptr().add(first);
        let rm_ptr = range_ma.as_ptr();
        let out_ptr = out.as_mut_ptr().add(first);

        let mut i = 0usize;
        if n >= 4 {
            let m = _mm256_set1_pd(signed_mult);
            let n_vec = n & !3usize;
            while i < n_vec {
                let r = _mm256_loadu_pd(rm_ptr.add(i));
                let b = _mm256_loadu_pd(base_ptr.add(i));
                let prod = _mm256_mul_pd(r, m);
                let res = _mm256_add_pd(b, prod);
                _mm256_storeu_pd(out_ptr.add(i), res);
                i += 4;
            }
        }

        while i < n {
            let r = *rm_ptr.add(i);
            let b = *base_ptr.add(i);
            *out_ptr.add(i) = b + r * signed_mult;
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kaufmanstop_avx512(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe {
            kaufmanstop_avx512_short(high, low, range_ma, period, first, mult, direction, out)
        }
    } else {
        unsafe { kaufmanstop_avx512_long(high, low, range_ma, period, first, mult, direction, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kaufmanstop_avx512_short(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    _period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    kaufmanstop_avx512_impl(high, low, range_ma, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kaufmanstop_avx512_long(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    _period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    kaufmanstop_avx512_impl(high, low, range_ma, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kaufmanstop_avx512_impl(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(out.len(), high.len());
    if first >= high.len() {
        return;
    }

    let is_long = direction.eq_ignore_ascii_case("long");
    let base = if is_long { low } else { high };
    let signed_mult = if is_long { -mult } else { mult };

    let n = core::cmp::min(range_ma.len(), high.len() - first);
    if n == 0 {
        return;
    }

    let base_ptr = base.as_ptr().add(first);
    let rm_ptr = range_ma.as_ptr();
    let out_ptr = out.as_mut_ptr().add(first);

    let mut i = 0usize;
    if n >= 8 {
        let m = _mm512_set1_pd(signed_mult);
        let n_vec = n & !7usize;
        while i < n_vec {
            let r = _mm512_loadu_pd(rm_ptr.add(i));
            let b = _mm512_loadu_pd(base_ptr.add(i));
            let prod = _mm512_mul_pd(r, m);
            let res = _mm512_add_pd(b, prod);
            _mm512_storeu_pd(out_ptr.add(i), res);
            i += 8;
        }
    }

    while i < n {
        let r = *rm_ptr.add(i);
        let b = *base_ptr.add(i);
        *out_ptr.add(i) = b + r * signed_mult;
        i += 1;
    }
}

// Decision: Streaming uses O(1) SMA/EMA; Other MA types fall back to O(n) over
// the current window in chronological order for correctness. SIMD remains disabled
// by default for Auto due to memory-bound AXPY behavior.

#[derive(Debug, Clone)]
pub struct KaufmanstopStream {
    // Params
    period: usize,
    mult: f64,
    direction: String,
    ma_type: String,

    // Ring buffer of high-low ranges
    range_buffer: Vec<f64>,
    buffer_head: usize,
    filled: bool,

    // Cached direction and scale
    is_long: bool,
    signed_mult: f64,

    // Which streaming path to use
    kind: StreamMaKind,

    // --- O(1) SMA state ---
    sum: f64,
    valid_count: u32,
    inv_period: f64, // 1.0 / period (used when no NaNs in window)

    // --- O(1) EMA state ---
    alpha: f64,
    beta: f64,
    ema: f64,
    ema_ready: bool, // becomes true after first seed from SMA
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamMaKind {
    SMA,
    EMA,
    Other, // WMA/SMMA/... -> fallback O(n)
}

impl KaufmanstopStream {
    pub fn try_new(params: KaufmanstopParams) -> Result<Self, KaufmanstopError> {
        let period = params.period.unwrap_or(22);
        if period == 0 {
            return Err(KaufmanstopError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let mult = params.mult.unwrap_or(2.0);
        let direction = params.direction.unwrap_or_else(|| "long".to_string());
        let ma_type = params.ma_type.unwrap_or_else(|| "sma".to_string());

        let is_long = direction.eq_ignore_ascii_case("long");
        let signed_mult = if is_long { -mult } else { mult };

        let kind = if ma_type.eq_ignore_ascii_case("sma") {
            StreamMaKind::SMA
        } else if ma_type.eq_ignore_ascii_case("ema") {
            StreamMaKind::EMA
        } else {
            StreamMaKind::Other
        };

        // EMA coefficients (ignored for SMA/Other)
        let alpha = 2.0 / (period as f64 + 1.0);
        let beta = 1.0 - alpha;

        Ok(Self {
            period,
            mult,
            direction,
            ma_type,
            range_buffer: vec![f64::NAN; period],
            buffer_head: 0,
            filled: false,

            is_long,
            signed_mult,
            kind,

            sum: 0.0,
            valid_count: 0,
            inv_period: 1.0 / period as f64,

            alpha,
            beta,
            ema: f64::NAN,
            ema_ready: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        // Compute current range (may be NaN)
        let new = if high.is_nan() || low.is_nan() {
            f64::NAN
        } else {
            high - low
        };

        // Did we already have a full window before inserting this sample?
        let was_filled = self.filled;

        // If the window is already full, remove the value that is about to be overwritten
        if was_filled {
            let old = self.range_buffer[self.buffer_head];
            if old == old {
                // old.is_nan() == false
                self.sum -= old;
                // valid_count > 0 by construction when old is finite
                self.valid_count -= 1;
            }
        }

        // Write the new value into the ring and add to running stats if finite
        self.range_buffer[self.buffer_head] = new;
        if new == new {
            self.sum += new;
            self.valid_count += 1;
        }

        // Advance ring
        self.buffer_head += 1;
        if self.buffer_head == self.period {
            self.buffer_head = 0;
            if !self.filled {
                self.filled = true; // window just became full
            }
        }

        // Warmup: do not emit until we have a full window of samples (NaNs allowed)
        if !self.filled {
            return None;
        }

        // Compute MA in O(1) for SMA/EMA; otherwise fall back to O(n) for correctness
        let ma_val = match self.kind {
            StreamMaKind::SMA => {
                // If all values in the window are NaN, the SMA is NaN
                if self.valid_count == 0 {
                    f64::NAN
                } else if self.valid_count as usize == self.period {
                    // Hot path (no NaNs in window): multiply by 1/N to avoid slow div
                    self.sum * self.inv_period
                } else {
                    // Mixed window (some NaNs): average over valid samples
                    self.sum / (self.valid_count as f64)
                }
            }
            StreamMaKind::EMA => {
                // Seed EMA with the average of the first full window (matches batch init).
                if !self.ema_ready {
                    if self.valid_count == 0 {
                        // Keep trying to seed until the window has finite data
                        f64::NAN
                    } else {
                        self.ema = self.sum / (self.valid_count as f64);
                        self.ema_ready = true;
                        self.ema
                    }
                } else {
                    // After seeding: only update EMA when the new range is finite
                    if new == new {
                        // ema = beta*ema + alpha*new (use fused multiply-add where available)
                        self.ema = self.ema.mul_add(self.beta, self.alpha * new);
                    }
                    self.ema
                }
            }
            StreamMaKind::Other => {
                // Fallback path (O(n) but correct): evaluate MA over the current window
                // in chronological order (oldest -> newest) using a small scratch buffer.
                let n = self.period;
                let mut tmp = vec![f64::NAN; n];
                // Oldest element in the window is at buffer_head (next to be overwritten)
                let tail = n - self.buffer_head;
                tmp[..tail].copy_from_slice(&self.range_buffer[self.buffer_head..]);
                tmp[tail..].copy_from_slice(&self.range_buffer[..self.buffer_head]);

                ma(&self.ma_type, MaData::Slice(&tmp), n)
                    .ok()
                    .and_then(|v| v.last().copied())
                    .unwrap_or(f64::NAN)
            }
        };

        // Final stop = base ± mult * MA(range)
        let base = if self.is_long { low } else { high };
        Some(ma_val.mul_add(self.signed_mult, base))
    }
}

#[derive(Clone, Debug)]
pub struct KaufmanstopBatchRange {
    pub period: (usize, usize, usize),
    pub mult: (f64, f64, f64),
    // String parameters use (start, end, step) tuple for consistency,
    // but step is always 0.0 and ignored since strings don't have numeric steps
    pub direction: (String, String, f64),
    pub ma_type: (String, String, f64),
}

impl Default for KaufmanstopBatchRange {
    fn default() -> Self {
        Self {
            period: (22, 22, 0),
            mult: (2.0, 2.0, 0.0),
            direction: ("long".to_string(), "long".to_string(), 0.0),
            ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KaufmanstopBatchBuilder {
    range: KaufmanstopBatchRange,
    kernel: Kernel,
}

impl KaufmanstopBatchBuilder {
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
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }
    #[inline]
    pub fn mult_static(mut self, x: f64) -> Self {
        self.range.mult = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn direction_static<S: Into<String>>(mut self, dir: S) -> Self {
        let s = dir.into();
        self.range.direction = (s.clone(), s, 0.0);
        self
    }
    #[inline]
    pub fn ma_type_static<S: Into<String>>(mut self, t: S) -> Self {
        let s = t.into();
        self.range.ma_type = (s.clone(), s, 0.0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
        kaufmanstop_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
        KaufmanstopBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
        let high = c
            .select_candle_field("high")
            .map_err(|_| KaufmanstopError::EmptyData)?;
        let low = c
            .select_candle_field("low")
            .map_err(|_| KaufmanstopError::EmptyData)?;
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
        KaufmanstopBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct KaufmanstopBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<KaufmanstopParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KaufmanstopBatchOutput {
    pub fn row_for_params(&self, p: &KaufmanstopParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(22) == p.period.unwrap_or(22)
                && (c.mult.unwrap_or(2.0) - p.mult.unwrap_or(2.0)).abs() < 1e-12
                && c.direction.as_ref().unwrap_or(&"long".to_string())
                    == p.direction.as_ref().unwrap_or(&"long".to_string())
                && c.ma_type.as_ref().unwrap_or(&"sma".to_string())
                    == p.ma_type.as_ref().unwrap_or(&"sma".to_string())
        })
    }
    pub fn values_for(&self, p: &KaufmanstopParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &KaufmanstopBatchRange) -> Vec<KaufmanstopParams> {
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
    fn axis_string((start, end, _step): (String, String, f64)) -> Vec<String> {
        // For string parameters, we only support single values or start/end pairs
        // The step parameter is ignored for strings
        if start == end {
            return vec![start.clone()];
        }
        vec![start.clone(), end.clone()]
    }
    let periods = axis_usize(r.period);
    let mults = axis_f64(r.mult);
    let directions = axis_string(r.direction.clone());
    let ma_types = axis_string(r.ma_type.clone());
    let mut out =
        Vec::with_capacity(periods.len() * mults.len() * directions.len() * ma_types.len());
    for &p in &periods {
        for &m in &mults {
            for d in &directions {
                for t in &ma_types {
                    out.push(KaufmanstopParams {
                        period: Some(p),
                        mult: Some(m),
                        direction: Some(d.clone()),
                        ma_type: Some(t.clone()),
                    });
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn kaufmanstop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &KaufmanstopBatchRange,
    k: Kernel,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
    let kernel = match k {
        // SIMD underperforms for this indicator; prefer ScalarBatch for Auto.
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(KaufmanstopError::InvalidPeriod {
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
    kaufmanstop_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn kaufmanstop_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &KaufmanstopBatchRange,
    kern: Kernel,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
    kaufmanstop_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn kaufmanstop_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &KaufmanstopBatchRange,
    kern: Kernel,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
    kaufmanstop_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn kaufmanstop_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &KaufmanstopBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
    let combos = expand_grid(sweep);
    let cols = high.len();
    let rows = combos.len();
    if rows == 0 {
        return Err(KaufmanstopError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // Allocate rows×cols with helpers and initialize warm prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(KaufmanstopError::AllValuesNaN)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Compute into that matrix
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let _ = kaufmanstop_batch_inner_into(high, low, sweep, kern, parallel, out)?;

    // Turn back into Vec<f64> (capacity is element-count, 1:1)
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(KaufmanstopBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn kaufmanstop_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &KaufmanstopBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<KaufmanstopParams>, KaufmanstopError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KaufmanstopError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let len = high.len();
    if len == 0 || len != low.len() {
        return Err(KaufmanstopError::EmptyData);
    }

    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(KaufmanstopError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(KaufmanstopError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    // range_buf with prefix-only init
    let mut range_buf = alloc_with_nan_prefix(len, first);
    for i in first..len {
        range_buf[i] = if high[i].is_nan() || low[i].is_nan() {
            f64::NAN
        } else {
            high[i] - low[i]
        };
    }

    let rows = combos.len();
    let cols = len;
    assert_eq!(out.len(), rows * cols, "out size must be rows*cols");

    // Reinterpret as MaybeUninit for parity with ALMA's writer
    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let c = &combos[row];
        let period = c.period.unwrap();
        let mult = c.mult.unwrap();
        let direction = c.direction.as_ref().unwrap();
        let ma_type = c.ma_type.as_ref().unwrap();

        // MA for this row over the valid tail
        let ma_input = MaData::Slice(&range_buf[first..]);
        let hl_diff_ma = match ma(ma_type, ma_input, period) {
            Ok(v) => v,
            Err(_) => {
                // Fill this row with NaNs
                let dst = std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols);
                for x in dst.iter_mut() {
                    *x = f64::NAN;
                }
                return;
            }
        };

        let dst = std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols);
        match actual {
            Kernel::Scalar | Kernel::ScalarBatch => {
                kaufmanstop_row_scalar(high, low, &hl_diff_ma, period, first, mult, direction, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                kaufmanstop_row_avx2(high, low, &hl_diff_ma, period, first, mult, direction, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                kaufmanstop_row_avx512(high, low, &hl_diff_ma, period, first, mult, direction, dst)
            }
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn kaufmanstop_row_scalar(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    kaufmanstop_scalar(high, low, range_ma, period, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx2(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    kaufmanstop_avx2(high, low, range_ma, period, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx512(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    if period <= 32 {
        kaufmanstop_row_avx512_short(high, low, range_ma, period, first, mult, direction, out);
    } else {
        kaufmanstop_row_avx512_long(high, low, range_ma, period, first, mult, direction, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx512_short(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    kaufmanstop_avx512_impl(high, low, range_ma, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx512_long(
    high: &[f64],
    low: &[f64],
    range_ma: &[f64],
    period: usize,
    first: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) {
    kaufmanstop_avx512_impl(high, low, range_ma, first, mult, direction, out)
}

#[inline(always)]
pub fn expand_grid_wrapper(r: &KaufmanstopBatchRange) -> Vec<KaufmanstopParams> {
    expand_grid(r)
}

#[cfg(feature = "python")]
#[pyfunction(name = "kaufmanstop")]
#[pyo3(signature = (high, low, period=22, mult=2.0, direction="long", ma_type="sma", kernel=None))]
pub fn kaufmanstop_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    mult: f64,
    direction: &str,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err(
            "High and low arrays must have the same length",
        ));
    }

    let kern = validate_kernel(kernel, false)?;
    let params = KaufmanstopParams {
        period: Some(period),
        mult: Some(mult),
        direction: Some(direction.to_string()),
        ma_type: Some(ma_type.to_string()),
    };
    let input = KaufmanstopInput::from_slices(high_slice, low_slice, params);

    // Perform computation within allow_threads for better performance
    let result = py.allow_threads(|| kaufmanstop_with_kernel(&input, kern));

    match result {
        Ok(output) => Ok(output.values.into_pyarray(py)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "KaufmanstopStream")]
pub struct KaufmanstopStreamPy {
    stream: KaufmanstopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KaufmanstopStreamPy {
    #[new]
    fn new(period: usize, mult: f64, direction: &str, ma_type: &str) -> PyResult<Self> {
        let params = KaufmanstopParams {
            period: Some(period),
            mult: Some(mult),
            direction: Some(direction.to_string()),
            ma_type: Some(ma_type.to_string()),
        };
        let stream =
            KaufmanstopStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KaufmanstopStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "kaufmanstop_batch")]
#[pyo3(signature = (high, low, period_range, mult_range=(2.0, 2.0, 0.0), direction="long", ma_type="sma", kernel=None))]
pub fn kaufmanstop_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    direction: &str,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err(
            "High and low arrays must have the same length",
        ));
    }

    let sweep = KaufmanstopBatchRange {
        period: period_range,
        mult: mult_range,
        direction: (direction.to_string(), direction.to_string(), 0.0),
        ma_type: (ma_type.to_string(), ma_type.to_string(), 0.0),
    };

    // Predict rows/cols before allocating
    let combos_preview = expand_grid(&sweep);
    let rows = combos_preview.len();
    let cols = h.len();

    // Preallocate NumPy 1D buffer then fill
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    let combos = py
        .allow_threads(|| {
            let simd = match match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            } {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            kaufmanstop_batch_inner_into(h, l, &sweep, simd, true, out_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    // Columnar metadata like ALMA
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|c| c.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "mults",
        combos
            .iter()
            .map(|c| c.mult.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    // String arrays need to be Python lists, not NumPy arrays
    dict.set_item(
        "directions",
        combos
            .iter()
            .map(|c| c.direction.as_deref().unwrap())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "ma_types",
        combos
            .iter()
            .map(|c| c.ma_type.as_deref().unwrap())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

// ==================== PYTHON: CUDA BINDINGS (zero-copy) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kaufmanstop_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, period_range, mult_range=(2.0, 2.0, 0.0), direction="long", ma_type="sma", device_id=0))]
pub fn kaufmanstop_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    direction: &str,
    ma_type: &str,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaKaufmanstop;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err("High and low arrays must have same length"));
    }
    let sweep = KaufmanstopBatchRange {
        period: period_range,
        mult: mult_range,
        direction: (direction.to_string(), direction.to_string(), 0.0),
        ma_type: (ma_type.to_string(), ma_type.to_string(), 0.0),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaKaufmanstop::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (dev, _combos) = cuda
            .kaufmanstop_batch_dev(h, l, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>(dev)
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kaufmanstop_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, period, mult=2.0, direction="long", ma_type="sma", device_id=0))]
pub fn kaufmanstop_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    mult: f64,
    direction: &str,
    ma_type: &str,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;
    use crate::cuda::cuda_available;
    use crate::cuda::CudaKaufmanstop;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let rows = high_tm_f32.shape()[0];
    let cols = high_tm_f32.shape()[1];
    if low_tm_f32.shape()[0] != rows || low_tm_f32.shape()[1] != cols {
        return Err(PyValueError::new_err("high/low shapes must match"));
    }
    let params = KaufmanstopParams {
        period: Some(period),
        mult: Some(mult),
        direction: Some(direction.to_string()),
        ma_type: Some(ma_type.to_string()),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaKaufmanstop::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda
            .kaufmanstop_many_series_one_param_time_major_dev(h, l, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

// ================== Core Helper Functions ==================
/// Core helper that writes directly to an output slice with no allocations.
/// This is the foundation for all WASM APIs.
pub fn kaufmanstop_into_slice(
    dst: &mut [f64],
    input: &KaufmanstopInput,
    kern: Kernel,
) -> Result<(), KaufmanstopError> {
    // Get data slices from input
    let (high, low) = match &input.data {
        KaufmanstopData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| KaufmanstopError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| KaufmanstopError::EmptyData)?;
            (high, low)
        }
        KaufmanstopData::Slices { high, low } => (*high, *low),
    };

    // Validate inputs
    if high.is_empty() || low.is_empty() {
        return Err(KaufmanstopError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(KaufmanstopError::InvalidPeriod {
            period: 0,
            data_len: high.len(),
        });
    }
    if dst.len() != high.len() {
        return Err(KaufmanstopError::InvalidPeriod {
            period: 0,
            data_len: high.len(),
        });
    }

    let period = input.get_period();
    let mult = input.get_mult();
    let direction = input.get_direction();
    let ma_type = input.get_ma_type();

    if period == 0 || period > high.len() || period > low.len() {
        return Err(KaufmanstopError::InvalidPeriod {
            period,
            data_len: high.len().min(low.len()),
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(KaufmanstopError::AllValuesNaN)?;

    if (high.len() - first_valid_idx) < period {
        return Err(KaufmanstopError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    // Calculate high-low differences in a temporary buffer using zero-copy allocation
    let mut hl_diff = alloc_with_nan_prefix(high.len(), first_valid_idx);
    for i in first_valid_idx..high.len() {
        if high[i].is_nan() || low[i].is_nan() {
            hl_diff[i] = f64::NAN;
        } else {
            hl_diff[i] = high[i] - low[i];
        }
    }

    // Calculate moving average of the differences
    let ma_input = MaData::Slice(&hl_diff[first_valid_idx..]);
    let hl_diff_ma = ma(ma_type, ma_input, period).map_err(|_| KaufmanstopError::AllValuesNaN)?;

    // SIMD underperforms vs scalar (memory-bound AXPY); default to Scalar for Auto.
    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    // Apply the kaufmanstop formula
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kaufmanstop_scalar(
                high,
                low,
                &hl_diff_ma,
                period,
                first_valid_idx,
                mult,
                direction,
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kaufmanstop_avx2(
                high,
                low,
                &hl_diff_ma,
                period,
                first_valid_idx,
                mult,
                direction,
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kaufmanstop_avx512(
                high,
                low,
                &hl_diff_ma,
                period,
                first_valid_idx,
                mult,
                direction,
                dst,
            ),
            _ => unreachable!(),
        }
    }

    // Fill warmup period with NaN
    let warmup_end = first_valid_idx + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Safe API that allocates a new vector and returns it.
/// `high` and `low` are JavaScript Float64Arrays
pub fn kaufmanstop_js(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    direction: &str,
    ma_type: &str,
) -> Result<Vec<f64>, JsError> {
    let params = KaufmanstopParams {
        period: Some(period),
        mult: Some(mult),
        direction: Some(direction.to_string()),
        ma_type: Some(ma_type.to_string()),
    };
    let input = KaufmanstopInput::from_slices(high, low, params);

    match kaufmanstop(&input) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsError::new(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Fast API that writes directly to pre-allocated memory.
/// Performs aliasing checks between input and output pointers.
pub fn kaufmanstop_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    mult: f64,
    direction: &str,
    ma_type: &str,
) -> Result<(), JsError> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsError::new("Null pointer passed to kaufmanstop_into"));
    }

    // SAFETY: We verify pointers are non-null and trust the caller for:
    // 1. Valid memory allocation
    // 2. Proper length
    // 3. No data races
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);

        // Aliasing detection: Check if output overlaps with either input
        let high_start = high_ptr as usize;
        let high_end = high_start + len * std::mem::size_of::<f64>();
        let low_start = low_ptr as usize;
        let low_end = low_start + len * std::mem::size_of::<f64>();
        let out_start = out_ptr as usize;
        let out_end = out_start + len * std::mem::size_of::<f64>();

        // Check if output overlaps with high
        let overlaps_high = (out_start < high_end) && (high_start < out_end);
        // Check if output overlaps with low
        let overlaps_low = (out_start < low_end) && (low_start < out_end);

        if overlaps_high || overlaps_low {
            // Overlap detected, use safe path with allocation
            let params = KaufmanstopParams {
                period: Some(period),
                mult: Some(mult),
                direction: Some(direction.to_string()),
                ma_type: Some(ma_type.to_string()),
            };
            let input = KaufmanstopInput::from_slices(high, low, params);
            let result = kaufmanstop(&input).map_err(|e| JsError::new(&e.to_string()))?;
            out.copy_from_slice(&result.values);
        } else {
            // No overlap, use direct write
            let params = KaufmanstopParams {
                period: Some(period),
                mult: Some(mult),
                direction: Some(direction.to_string()),
                ma_type: Some(ma_type.to_string()),
            };
            let input = KaufmanstopInput::from_slices(high, low, params);
            kaufmanstop_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsError::new(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Allocates memory for a f64 array of the given length.
/// Returns a pointer that must be freed with kaufmanstop_free.
pub fn kaufmanstop_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Frees memory allocated by kaufmanstop_alloc.
/// SAFETY: ptr must have been allocated by kaufmanstop_alloc with the same length.
pub unsafe fn kaufmanstop_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    Vec::from_raw_parts(ptr, len, len);
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KaufmanstopBatchMeta {
    pub combos: Vec<KaufmanstopParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Batch computation with parameter sweep.
/// Returns JavaScript object with { values: Float64Array, combos: Array, rows: number, cols: number }
pub fn kaufmanstop_batch_js(
    high: &[f64],
    low: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
    direction: &str,
    ma_type: &str,
) -> Result<JsValue, JsError> {
    let sweep = KaufmanstopBatchRange {
        period: (period_start, period_end, period_step),
        mult: (mult_start, mult_end, mult_step),
        direction: (direction.to_string(), direction.to_string(), 0.0),
        ma_type: (ma_type.to_string(), ma_type.to_string(), 0.0),
    };

    match kaufmanstop_batch_slice(high, low, &sweep, Kernel::Auto) {
        Ok(output) => {
            let meta = KaufmanstopBatchMeta {
                combos: output.combos,
                rows: output.rows,
                cols: output.cols,
            };

            // Create JS object manually
            let js_object = js_sys::Object::new();

            // Add values as Float64Array
            let values_array = js_sys::Float64Array::from(&output.values[..]);
            js_sys::Reflect::set(&js_object, &"values".into(), &values_array.into())
                .map_err(|e| JsError::new(&format!("Failed to set values: {:?}", e)))?;

            // Add metadata
            let meta_value = serde_wasm_bindgen::to_value(&meta)?;
            let combos = js_sys::Reflect::get(&meta_value, &"combos".into())
                .map_err(|e| JsError::new(&format!("Failed to get combos: {:?}", e)))?;
            js_sys::Reflect::set(&js_object, &"combos".into(), &combos)
                .map_err(|e| JsError::new(&format!("Failed to set combos: {:?}", e)))?;

            let rows = js_sys::Reflect::get(&meta_value, &"rows".into())
                .map_err(|e| JsError::new(&format!("Failed to get rows: {:?}", e)))?;
            js_sys::Reflect::set(&js_object, &"rows".into(), &rows)
                .map_err(|e| JsError::new(&format!("Failed to set rows: {:?}", e)))?;

            let cols = js_sys::Reflect::get(&meta_value, &"cols".into())
                .map_err(|e| JsError::new(&format!("Failed to get cols: {:?}", e)))?;
            js_sys::Reflect::set(&js_object, &"cols".into(), &cols)
                .map_err(|e| JsError::new(&format!("Failed to set cols: {:?}", e)))?;

            Ok(js_object.into())
        }
        Err(e) => Err(JsError::new(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Unified batch API that accepts a config object with ranges.
pub fn kaufmanstop_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsError> {
    #[derive(Deserialize)]
    struct BatchConfig {
        period_range: Option<(usize, usize, usize)>,
        mult_range: Option<(f64, f64, f64)>,
        direction: Option<String>,
        ma_type: Option<String>,
    }

    let config: BatchConfig = serde_wasm_bindgen::from_value(config)?;

    let sweep = KaufmanstopBatchRange {
        period: config.period_range.unwrap_or((22, 22, 0)),
        mult: config.mult_range.unwrap_or((2.0, 2.0, 0.0)),
        direction: config.direction.map(|d| (d.clone(), d, 0.0)).unwrap_or((
            "long".to_string(),
            "long".to_string(),
            0.0,
        )),
        ma_type: config.ma_type.map(|t| (t.clone(), t, 0.0)).unwrap_or((
            "sma".to_string(),
            "sma".to_string(),
            0.0,
        )),
    };

    match kaufmanstop_batch_slice(high, low, &sweep, Kernel::Auto) {
        Ok(output) => {
            let meta = KaufmanstopBatchMeta {
                combos: output.combos,
                rows: output.rows,
                cols: output.cols,
            };

            // Create JS object
            let js_object = js_sys::Object::new();

            // Add values as Float64Array
            let values_array = js_sys::Float64Array::from(&output.values[..]);
            js_sys::Reflect::set(&js_object, &"values".into(), &values_array.into())
                .map_err(|e| JsError::new(&format!("Failed to set values: {:?}", e)))?;

            // Add metadata
            let meta_value = serde_wasm_bindgen::to_value(&meta)?;
            let combos = js_sys::Reflect::get(&meta_value, &"combos".into())
                .map_err(|e| JsError::new(&format!("Failed to get combos: {:?}", e)))?;
            js_sys::Reflect::set(&js_object, &"combos".into(), &combos)
                .map_err(|e| JsError::new(&format!("Failed to set combos: {:?}", e)))?;

            let rows = js_sys::Reflect::get(&meta_value, &"rows".into())
                .map_err(|e| JsError::new(&format!("Failed to get rows: {:?}", e)))?;
            js_sys::Reflect::set(&js_object, &"rows".into(), &rows)
                .map_err(|e| JsError::new(&format!("Failed to set rows: {:?}", e)))?;

            let cols = js_sys::Reflect::get(&meta_value, &"cols".into())
                .map_err(|e| JsError::new(&format!("Failed to get cols: {:?}", e)))?;
            js_sys::Reflect::set(&js_object, &"cols".into(), &cols)
                .map_err(|e| JsError::new(&format!("Failed to set cols: {:?}", e)))?;

            Ok(js_object.into())
        }
        Err(e) => Err(JsError::new(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Fast batch API that writes to pre-allocated memory.
pub fn kaufmanstop_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
    direction: &str,
    ma_type: &str,
) -> Result<JsValue, JsError> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsError::new(
            "Null pointer passed to kaufmanstop_batch_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = KaufmanstopBatchRange {
            period: (period_start, period_end, period_step),
            mult: (mult_start, mult_end, mult_step),
            direction: (direction.to_string(), direction.to_string(), 0.0),
            ma_type: (ma_type.to_string(), ma_type.to_string(), 0.0),
        };

        let combos_preview = expand_grid(&sweep);
        let rows = combos_preview.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        let _ =
            kaufmanstop_batch_inner_into(high, low, &sweep, detect_best_batch_kernel(), false, out)
                .map_err(|e| JsError::new(&e.to_string()))?;

        let meta = KaufmanstopBatchMeta {
            combos: combos_preview,
            rows,
            cols,
        };
        serde_wasm_bindgen::to_value(&meta).map_err(Into::into)
    }
}

/// Optimized KaufmanStop calculation with inline SMA
#[inline]
pub unsafe fn kaufmanstop_scalar_classic_sma(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) -> Result<(), KaufmanstopError> {
    let start_idx = first_valid_idx + period - 1;
    let is_long = direction.eq_ignore_ascii_case("long");

    // Calculate initial SMA of high-low range
    let mut sum = 0.0;
    let mut valid_count = 0;
    for i in 0..period {
        let idx = first_valid_idx + i;
        if !high[idx].is_nan() && !low[idx].is_nan() {
            sum += high[idx] - low[idx];
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return Err(KaufmanstopError::AllValuesNaN);
    }

    let mut sma = sum / valid_count as f64;

    // Calculate first KaufmanStop value
    if is_long {
        out[start_idx] = low[start_idx] - sma * mult;
    } else {
        out[start_idx] = high[start_idx] + sma * mult;
    }

    // Rolling window calculations
    for i in (start_idx + 1)..high.len() {
        // Update SMA with rolling window
        let old_idx = i - period;
        let new_idx = i;

        // Remove old value
        if !high[old_idx].is_nan() && !low[old_idx].is_nan() {
            let old_range = high[old_idx] - low[old_idx];
            sum -= old_range;
            valid_count -= 1;
        }

        // Add new value
        if !high[new_idx].is_nan() && !low[new_idx].is_nan() {
            let new_range = high[new_idx] - low[new_idx];
            sum += new_range;
            valid_count += 1;
        }

        if valid_count > 0 {
            sma = sum / valid_count as f64;
        } else {
            sma = f64::NAN;
        }

        // Calculate KaufmanStop value
        if is_long {
            out[i] = low[i] - sma * mult;
        } else {
            out[i] = high[i] + sma * mult;
        }
    }

    Ok(())
}

/// Optimized KaufmanStop calculation with inline EMA
#[inline]
pub unsafe fn kaufmanstop_scalar_classic_ema(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    mult: f64,
    direction: &str,
    out: &mut [f64],
) -> Result<(), KaufmanstopError> {
    let start_idx = first_valid_idx + period - 1;
    let is_long = direction.eq_ignore_ascii_case("long");
    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;

    // Calculate initial SMA for EMA startup
    let mut sum = 0.0;
    let mut valid_count = 0;
    for i in 0..period {
        let idx = first_valid_idx + i;
        if !high[idx].is_nan() && !low[idx].is_nan() {
            sum += high[idx] - low[idx];
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return Err(KaufmanstopError::AllValuesNaN);
    }

    let mut ema = sum / valid_count as f64;

    // Calculate first KaufmanStop value
    if is_long {
        out[start_idx] = low[start_idx] - ema * mult;
    } else {
        out[start_idx] = high[start_idx] + ema * mult;
    }

    // EMA and KaufmanStop calculations
    for i in (start_idx + 1)..high.len() {
        // Update EMA with new range value
        if !high[i].is_nan() && !low[i].is_nan() {
            let range = high[i] - low[i];
            ema = alpha * range + beta * ema;
        }
        // If current values are NaN, EMA continues with previous value

        // Calculate KaufmanStop value
        if is_long {
            out[i] = low[i] - ema * mult;
        } else {
            out[i] = high[i] + ema * mult;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_kaufmanstop_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KaufmanstopInput::with_default_candles(&candles);
        let output = kaufmanstop_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_kaufmanstop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KaufmanstopInput::with_default_candles(&candles);
        let result = kaufmanstop_with_kernel(&input, kernel)?;
        let expected_last_five = [
            56711.545454545456,
            57132.72727272727,
            57015.72727272727,
            57137.18181818182,
            56516.09090909091,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] Kaufmanstop {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_kaufmanstop_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = KaufmanstopParams {
            period: Some(0),
            mult: None,
            direction: None,
            ma_type: None,
        };
        let input = KaufmanstopInput::from_slices(&high, &low, params);
        let res = kaufmanstop_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Kaufmanstop should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_kaufmanstop_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = KaufmanstopParams {
            period: Some(10),
            mult: None,
            direction: None,
            ma_type: None,
        };
        let input = KaufmanstopInput::from_slices(&high, &low, params);
        let res = kaufmanstop_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Kaufmanstop should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_kaufmanstop_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [41.0];
        let params = KaufmanstopParams {
            period: Some(22),
            mult: None,
            direction: None,
            ma_type: None,
        };
        let input = KaufmanstopInput::from_slices(&high, &low, params);
        let res = kaufmanstop_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Kaufmanstop should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_kaufmanstop_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KaufmanstopInput::with_default_candles(&candles);
        let res = kaufmanstop_with_kernel(&input, kernel)?;
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
    fn check_kaufmanstop_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = candles.select_candle_field("high").unwrap();
        let low = candles.select_candle_field("low").unwrap();

        // Create stream
        let params = KaufmanstopParams {
            period: Some(22),
            mult: Some(2.0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let mut stream = KaufmanstopStream::try_new(params)?;

        // Feed data to stream
        let mut stream_results = Vec::new();
        for i in 0..high.len() {
            if let Some(val) = stream.update(high[i], low[i]) {
                stream_results.push(val);
            } else {
                stream_results.push(f64::NAN);
            }
        }

        // Compare with batch computation
        let input = KaufmanstopInput::with_default_candles(&candles);
        let batch_result = kaufmanstop_with_kernel(&input, kernel)?;

        // Stream should match batch after warmup period
        let warmup = 22 + 21; // period + (period - 1)
        for i in warmup..high.len() {
            let diff = (stream_results[i] - batch_result.values[i]).abs();
            assert!(
                diff < 1e-10 || (stream_results[i].is_nan() && batch_result.values[i].is_nan()),
                "[{}] Stream vs batch mismatch at index {}: {} vs {}",
                test_name,
                i,
                stream_results[i],
                batch_result.values[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_kaufmanstop_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations for Kaufmanstop
        let test_params = vec![
            KaufmanstopParams::default(), // period: 22, mult: 2.0, direction: "long", ma_type: "sma"
            KaufmanstopParams {
                period: Some(2),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // minimum period
            KaufmanstopParams {
                period: Some(5),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // small period
            KaufmanstopParams {
                period: Some(10),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // small-medium period
            KaufmanstopParams {
                period: Some(22),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // default period
            KaufmanstopParams {
                period: Some(50),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // medium period
            KaufmanstopParams {
                period: Some(100),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // large period
            KaufmanstopParams {
                period: Some(200),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // very large period
            KaufmanstopParams {
                period: Some(22),
                mult: Some(0.1),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // very small multiplier
            KaufmanstopParams {
                period: Some(22),
                mult: Some(0.5),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // small multiplier
            KaufmanstopParams {
                period: Some(22),
                mult: Some(1.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // multiplier 1.0
            KaufmanstopParams {
                period: Some(22),
                mult: Some(3.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // large multiplier
            KaufmanstopParams {
                period: Some(22),
                mult: Some(5.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // very large multiplier
            KaufmanstopParams {
                period: Some(22),
                mult: Some(10.0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // extreme multiplier
            KaufmanstopParams {
                period: Some(22),
                mult: Some(2.0),
                direction: Some("short".to_string()),
                ma_type: Some("sma".to_string()),
            }, // short direction
            KaufmanstopParams {
                period: Some(22),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("ema".to_string()),
            }, // ema type
            KaufmanstopParams {
                period: Some(22),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("smma".to_string()),
            }, // smma type
            KaufmanstopParams {
                period: Some(22),
                mult: Some(2.0),
                direction: Some("long".to_string()),
                ma_type: Some("wma".to_string()),
            }, // wma type
            KaufmanstopParams {
                period: Some(14),
                mult: Some(1.5),
                direction: Some("short".to_string()),
                ma_type: Some("ema".to_string()),
            }, // mixed params 1
            KaufmanstopParams {
                period: Some(30),
                mult: Some(2.5),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            }, // mixed params 2
            KaufmanstopParams {
                period: Some(7),
                mult: Some(0.85),
                direction: Some("short".to_string()),
                ma_type: Some("wma".to_string()),
            }, // mixed params 3
            KaufmanstopParams {
                period: Some(3),
                mult: Some(4.0),
                direction: Some("long".to_string()),
                ma_type: Some("ema".to_string()),
            }, // edge case: very small period, large mult
            KaufmanstopParams {
                period: Some(150),
                mult: Some(0.25),
                direction: Some("short".to_string()),
                ma_type: Some("smma".to_string()),
            }, // edge case: large period, small mult
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = KaufmanstopInput::from_candles(&candles, params.clone());
            let output = kaufmanstop_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, direction={}, ma_type={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(22),
                        params.mult.unwrap_or(2.0),
                        params.direction.as_deref().unwrap_or("long"),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, direction={}, ma_type={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(22),
                        params.mult.unwrap_or(2.0),
                        params.direction.as_deref().unwrap_or("long"),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, direction={}, ma_type={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(22),
                        params.mult.unwrap_or(2.0),
                        params.direction.as_deref().unwrap_or("long"),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_kaufmanstop_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_kaufmanstop_tests {
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

    generate_all_kaufmanstop_tests!(
        check_kaufmanstop_partial_params,
        check_kaufmanstop_accuracy,
        check_kaufmanstop_zero_period,
        check_kaufmanstop_period_exceeds_length,
        check_kaufmanstop_very_small_dataset,
        check_kaufmanstop_nan_handling,
        check_kaufmanstop_streaming,
        check_kaufmanstop_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_kaufmanstop_tests!(check_kaufmanstop_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KaufmanstopBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let def = KaufmanstopParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            56711.545454545456,
            57132.72727272727,
            57015.72727272727,
            57137.18181818182,
            56516.09090909091,
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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations for Kaufmanstop
        let test_configs = vec![
            // (period_start, period_end, period_step, mult_start, mult_end, mult_step, direction, ma_type)
            (2, 10, 2, 2.0, 2.0, 0.0, "long", "sma"), // Small periods, fixed mult
            (10, 50, 10, 2.0, 2.0, 0.0, "long", "sma"), // Medium periods, fixed mult
            (20, 100, 20, 2.0, 2.0, 0.0, "long", "sma"), // Large periods, fixed mult
            (22, 22, 0, 0.5, 3.0, 0.5, "long", "sma"), // Fixed period, varying mult
            (22, 22, 0, 1.0, 5.0, 1.0, "short", "sma"), // Fixed period, varying mult, short
            (5, 20, 5, 1.0, 3.0, 1.0, "long", "sma"), // Mixed period and mult sweep
            (10, 30, 10, 1.5, 2.5, 0.5, "short", "sma"), // Mixed sweep, short direction
            (2, 5, 1, 0.1, 0.5, 0.1, "long", "ema"),  // Dense small range, small mult, ema
            (50, 150, 50, 3.0, 5.0, 1.0, "short", "wma"), // Large periods, large mult, wma
            (3, 15, 3, 0.25, 2.0, 0.25, "long", "smma"), // Small-medium periods, varied mult, smma
            (14, 14, 0, 0.5, 4.0, 0.5, "short", "ema"), // Fixed period, wide mult range, ema
            (100, 200, 100, 1.0, 1.0, 0.0, "long", "sma"), // Very large periods, fixed mult
        ];

        for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step, dir, ma_type)) in
            test_configs.iter().enumerate()
        {
            let mut builder = KaufmanstopBatchBuilder::new()
                .kernel(kernel)
                .direction_static(dir)
                .ma_type_static(ma_type);

            // Configure period range
            if p_step > 0 {
                builder = builder.period_range(p_start, p_end, p_step);
            } else {
                builder = builder.period_static(p_start);
            }

            // Configure mult range
            if m_step > 0.0 {
                builder = builder.mult_range(m_start, m_end, m_step);
            } else {
                builder = builder.mult_static(m_start);
            }

            let output = builder.apply_candles(&c)?;

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
						 at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(22),
                        combo.mult.unwrap_or(2.0),
                        combo.direction.as_deref().unwrap_or("long"),
                        combo.ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(22),
                        combo.mult.unwrap_or(2.0),
                        combo.direction.as_deref().unwrap_or("long"),
                        combo.ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(22),
                        combo.mult.unwrap_or(2.0),
                        combo.direction.as_deref().unwrap_or("long"),
                        combo.ma_type.as_deref().unwrap_or("sma")
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_kaufmanstop_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Kaufmanstop Property Test Design:
        //
        // This test validates the Kaufmanstop indicator which calculates adaptive price stops
        // based on the moving average of the high-low range, multiplied by a factor.
        //
        // Key design decisions:
        // 1. MA Type Selection: Only simple MA types (sma, ema, wma) are tested to avoid
        //    complex warmup requirements of DEMA/TEMA which need 2x or 3x the period.
        //
        // 2. Warmup Handling: The kaufmanstop delegates warmup to the underlying MA function.
        //    Each MA type has different warmup characteristics:
        //    - SMA: Requires full period for first output
        //    - EMA: Can produce output immediately but needs warmup for stability
        //    - WMA: Requires full period for weighted calculation
        //
        // 3. Data Generation: Synthetic OHLC data includes:
        //    - Normal volatility periods (0.1% to 5%)
        //    - Occasional very small range periods to test edge cases
        //    - Proper OHLC constraints (high >= low)
        //    - Realistic price trends and random walk
        //
        // 4. Properties Tested:
        //    - Warmup validation
        //    - Kernel consistency (SIMD vs scalar)
        //    - Stop level bounds (long below low, short above high)
        //    - Mathematical formula verification
        //    - Edge cases (period=1, small ranges, small/large multipliers)

        // Strategy to generate realistic OHLC data with proper constraints
        let strat = (2usize..=50) // Period range
            .prop_flat_map(|period| {
                (
                    100.0f64..5000.0f64, // Base price range
                    (period + 20)..400,  // Data length
                    0.001f64..0.05f64,   // Volatility factor (0.1% to 5%)
                    -0.01f64..0.01f64,   // Trend factor
                    Just(period),
                    0.1f64..5.0f64,                              // Multiplier range
                    prop::sample::select(vec!["long", "short"]), // Direction
                    prop::sample::select(vec!["sma", "ema", "wma"]), // MA types
                )
            })
            .prop_map(
                |(base_price, data_len, volatility, trend, period, mult, direction, ma_type)| {
                    // Generate synthetic OHLC data with realistic constraints
                    let mut high = Vec::with_capacity(data_len);
                    let mut low = Vec::with_capacity(data_len);
                    let mut price = base_price;

                    for i in 0..data_len {
                        // Add trend and random walk
                        price *= 1.0 + trend + (i as f64 * 0.0001);
                        let noise = ((i * 17 + 11) % 100) as f64 / 100.0 - 0.5;
                        price *= 1.0 + noise * volatility;

                        // Create high/low with proper constraints
                        // Occasionally create very small or zero range periods (flat price)
                        let range = if i > data_len - 20 && i % 3 == 0 {
                            // Last 20 points, every 3rd point has very small range
                            price * 0.00001 // Very small range for testing
                        } else {
                            price * volatility * 2.0 // Normal range
                        };

                        high.push(price + range / 2.0);
                        low.push(price - range / 2.0);
                    }

                    (high, low, period, mult, direction, ma_type)
                },
            );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high, low, period, mult, direction, ma_type)| {
                let params = KaufmanstopParams {
                    period: Some(period),
                    mult: Some(mult),
                    direction: Some(direction.to_string()),
                    ma_type: Some(ma_type.to_string()),
                };
                let input = KaufmanstopInput::from_slices(&high, &low, params.clone());

                // Get results from the kernel being tested
                let result = kaufmanstop_with_kernel(&input, kernel);
                prop_assert!(
                    result.is_ok(),
                    "Kaufmanstop computation failed: {:?}",
                    result
                );
                let KaufmanstopOutput { values: out } = result.unwrap();

                // Get reference results from scalar kernel
                let ref_result = kaufmanstop_with_kernel(&input, Kernel::Scalar);
                prop_assert!(
                    ref_result.is_ok(),
                    "Reference computation failed: {:?}",
                    ref_result
                );
                let KaufmanstopOutput { values: ref_out } = ref_result.unwrap();

                // Both outputs should have the same length as input
                prop_assert_eq!(out.len(), high.len());
                prop_assert_eq!(ref_out.len(), high.len());

                // Find first valid index
                let first_valid_idx = high
                    .iter()
                    .zip(low.iter())
                    .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
                    .unwrap_or(0);

                // Property 1: Warmup period validation
                // The MA function itself handles warmup, and kaufmanstop writes the MA output
                // starting from first_valid_idx. The actual warmup depends on the MA type.
                // For simple MAs like SMA, EMA, WMA with period N:
                // - SMA needs N points to produce first output
                // - EMA can produce output from first point but may need warmup for stability
                // - WMA needs N points
                // So we expect NaN values before first_valid_idx at minimum
                for i in 0..first_valid_idx {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN before first valid index at {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Also check that we eventually get non-NaN values after sufficient data
                let expected_first_valid = first_valid_idx + period - 1;
                if expected_first_valid < out.len() - 10 {
                    let has_valid = out[expected_first_valid..expected_first_valid + 10]
                        .iter()
                        .any(|&v| !v.is_nan());
                    prop_assert!(
                        has_valid,
                        "Expected at least one valid value after warmup period"
                    );
                }

                // Property 2: Kernel consistency
                // Different kernel implementations should produce nearly identical results
                for i in first_valid_idx..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/infinite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 8,
                        "Kernel mismatch at index {}: {} vs {} (ULP diff: {})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 3: Stop level bounds
                // For valid indices, check that stops respect direction
                for i in first_valid_idx..out.len() {
                    if out[i].is_nan() || high[i].is_nan() || low[i].is_nan() {
                        continue;
                    }

                    let stop = out[i];

                    if direction == "long" {
                        // Long stops should be below the low (with some tolerance for calculation)
                        prop_assert!(
                            stop <= low[i] + 1e-6,
                            "Long stop {} should be below low {} at index {}",
                            stop,
                            low[i],
                            i
                        );
                    } else {
                        // Short stops should be above the high (with some tolerance)
                        prop_assert!(
                            stop >= high[i] - 1e-6,
                            "Short stop {} should be above high {} at index {}",
                            stop,
                            high[i],
                            i
                        );
                    }
                }

                // Property 4: Mathematical definition verification
                // The Kaufmanstop formula is: stop = price +/- (MA(high-low) * mult)
                // We can't easily recalculate the exact MA here due to different MA types,
                // but we can verify the stop distance is reasonable given the multiplier.
                let check_start = first_valid_idx + period;
                if out.len() > check_start + 10 {
                    // Calculate range manually for verification
                    let mut range_values = Vec::new();
                    for i in first_valid_idx..high.len() {
                        if high[i].is_nan() || low[i].is_nan() {
                            range_values.push(f64::NAN);
                        } else {
                            range_values.push(high[i] - low[i]);
                        }
                    }

                    // Verify stop distances are within reasonable bounds
                    // The MA of ranges should be roughly similar to recent ranges,
                    // so the stop distance should be approximately mult * typical_range
                    for i in (check_start + 5)..(check_start + 10).min(out.len()) {
                        if out[i].is_nan() {
                            continue;
                        }

                        // Calculate typical range in the recent window
                        let window_start = i.saturating_sub(period);
                        let recent_ranges: Vec<f64> = (window_start..i)
                            .filter_map(|j| {
                                if !high[j].is_nan() && !low[j].is_nan() {
                                    Some(high[j] - low[j])
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if recent_ranges.is_empty() {
                            continue;
                        }

                        let avg_recent_range =
                            recent_ranges.iter().sum::<f64>() / recent_ranges.len() as f64;
                        let max_distance = avg_recent_range * mult * 2.0; // Allow 2x for MA smoothing effects

                        if direction == "long" {
                            let distance = low[i] - out[i];
                            prop_assert!(
                                distance >= 0.0 && distance <= max_distance,
                                "Long stop distance {} out of bounds at index {} (max: {})",
                                distance,
                                i,
                                max_distance
                            );
                        } else {
                            let distance = out[i] - high[i];
                            prop_assert!(
                                distance >= 0.0 && distance <= max_distance,
                                "Short stop distance {} out of bounds at index {} (max: {})",
                                distance,
                                i,
                                max_distance
                            );
                        }
                    }
                }

                // Property 5: Edge cases
                // Test period = 1 case
                if period == 1 && out.len() > first_valid_idx {
                    // With period=1, the MA of range is just the current range
                    for i in first_valid_idx..out.len().min(first_valid_idx + 10) {
                        if out[i].is_nan() || high[i].is_nan() || low[i].is_nan() {
                            continue;
                        }

                        let expected_range = high[i] - low[i];
                        if direction == "long" {
                            let expected = low[i] - expected_range * mult;
                            prop_assert!(
                                (out[i] - expected).abs() <= 1e-6,
                                "Period=1 long stop mismatch at {}: {} vs {}",
                                i,
                                out[i],
                                expected
                            );
                        } else {
                            let expected = high[i] + expected_range * mult;
                            prop_assert!(
                                (out[i] - expected).abs() <= 1e-6,
                                "Period=1 short stop mismatch at {}: {} vs {}",
                                i,
                                out[i],
                                expected
                            );
                        }
                    }
                }

                // Property 6: Very small range scenario
                // When the price range is very small (near-flat price), stops should be very close to price
                let const_start = out.len().saturating_sub(20);
                let small_range_indices: Vec<usize> = (const_start..out.len())
                    .filter(|&i| {
                        let range = high[i] - low[i];
                        let avg_price = (high[i] + low[i]) / 2.0;
                        range < avg_price * 0.0001 // Very small range relative to price
                    })
                    .collect();

                if !small_range_indices.is_empty() && const_start >= first_valid_idx + period {
                    for &i in &small_range_indices {
                        if out[i].is_nan() {
                            continue;
                        }

                        // With very small range at current point, but MA includes previous normal ranges,
                        // the stop distance will be based on the average which includes larger ranges
                        // Calculate the average range in the lookback window
                        let window_start = i.saturating_sub(period - 1);
                        let window_ranges: Vec<f64> =
                            (window_start..=i).map(|j| high[j] - low[j]).collect();
                        let avg_window_range =
                            window_ranges.iter().sum::<f64>() / window_ranges.len() as f64;
                        let expected_max_distance = avg_window_range * mult * 2.0; // Allow 2x for MA smoothing

                        if direction == "long" {
                            let distance = low[i] - out[i];
                            prop_assert!(
								distance >= 0.0 && distance <= expected_max_distance,
								"Small range long stop distance {} exceeds expected max {} at index {}",
								distance, expected_max_distance, i
							);
                        } else {
                            let distance = out[i] - high[i];
                            prop_assert!(
								distance >= 0.0 && distance <= expected_max_distance,
								"Small range short stop distance {} exceeds expected max {} at index {}",
								distance, expected_max_distance, i
							);
                        }
                    }
                }

                // Property 7: Small multiplier test
                // With very small multiplier, stops should be very close to price
                let test_start = first_valid_idx + period;
                if mult < 0.2 && out.len() > test_start + 5 {
                    for i in (test_start + 2)..(test_start + 5).min(out.len()) {
                        if out[i].is_nan() || high[i].is_nan() || low[i].is_nan() {
                            continue;
                        }

                        let max_distance = (high[i] - low[i]) * 2.0; // Generous bound
                        if direction == "long" {
                            let distance = low[i] - out[i];
                            prop_assert!(
                                distance <= max_distance,
                                "Small mult long stop too far at {}: distance {}",
                                i,
                                distance
                            );
                        } else {
                            let distance = out[i] - high[i];
                            prop_assert!(
                                distance <= max_distance,
                                "Small mult short stop too far at {}: distance {}",
                                i,
                                distance
                            );
                        }
                    }
                }

                // Property 8: Large multiplier test
                // With very large multiplier, stops should still be reasonable (not infinite or extreme)
                if mult > 3.0 && out.len() > test_start + 5 {
                    for i in (test_start + 2)..(test_start + 5).min(out.len()) {
                        if out[i].is_nan() || high[i].is_nan() || low[i].is_nan() {
                            continue;
                        }

                        // Even with large multiplier, stop distance shouldn't exceed
                        // a reasonable multiple of the price itself
                        // With large multipliers (up to 5.0), stops could be up to mult * average_range away
                        // In volatile conditions, average_range could be 10-15% of price
                        // So mult * 0.15 * price is a reasonable upper bound
                        let price = (high[i] + low[i]) / 2.0;
                        let max_reasonable_distance = price * (mult * 0.15).min(1.0); // Scale with multiplier, cap at 100% of price

                        if direction == "long" {
                            let distance = low[i] - out[i];
                            prop_assert!(
								distance < max_reasonable_distance,
								"Large mult long stop unreasonably far at {}: distance {} ({}% of price {})",
								i, distance, (distance / price * 100.0), price
							);
                            // Also ensure stop doesn't go negative for long positions
                            prop_assert!(
                                out[i] > 0.0,
                                "Large mult long stop went negative at {}: {}",
                                i,
                                out[i]
                            );
                        } else {
                            let distance = out[i] - high[i];
                            prop_assert!(
								distance < max_reasonable_distance,
								"Large mult short stop unreasonably far at {}: distance {} ({}% of price {})",
								i, distance, (distance / price * 100.0), price
							);
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }
}
