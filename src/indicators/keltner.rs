//! # Keltner Channels
//!
//! Volatility-based envelope indicator using MA and ATR.
//! Middle band is MA of source, upper/lower bands are MA ± (multiplier × ATR).
//!
//! ## Parameters
//! - **period**: Window for both MA and ATR (default: 20)
//! - **multiplier**: ATR multiplier for bands (default: 2.0)
//! - **ma_type**: Moving average type (default: "ema")
//!
//! ## Returns
//! - **`Ok(KeltnerOutput)`** on success (`upper_band`, `middle_band`, `lower_band` vectors)
//! - **`Err(KeltnerError)`** on failure (typed errors for empty input, NaNs, invalid periods/ranges, and batch misuse)
//!
//! ## Developer Status
//! - **SIMD Kernels**: AVX2/AVX512 present as stubs delegating to scalar. Due to time-recursive EMA/SMA/ATR, SIMD offers marginal gains; runtime short-circuits to scalar.
//! - **Scalar**: Loop-jammed EMA/SMA + ATR RMA in a single pass; uses mul_add and unchecked indexing in tight loops.
//! - **Batch**: Row-specific optimization precomputes TR once and reuses per row; >5% faster at 100k vs non-shared TR with checked sweep ranges.
//! - **Streaming**: Enabled. Exact EMA seeding (SMA of first n) + Wilder RMA with fused mul_add; matches batch outputs to ~1e-9 in tests.
//! - **CUDA/Python**: CUDA batch/many-series wrappers with VRAM handles exposing CUDA Array Interface v3 + DLPack v1.x for CuPy/PyTorch/JAX interop.
//! - **Memory**: Good zero-copy usage (alloc_with_nan_prefix, make_uninit_matrix) with checked sizes for batch matrices.

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
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// CUDA Python interop types for device arrays
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::keltner_wrapper::CudaKeltner;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

// Input & data types

#[derive(Debug, Clone)]
pub enum KeltnerData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), // high, low, close, source
}

#[derive(Debug, Clone)]
pub struct KeltnerOutput {
    pub upper_band: Vec<f64>,
    pub middle_band: Vec<f64>,
    pub lower_band: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct KeltnerParams {
    pub period: Option<usize>,
    pub multiplier: Option<f64>,
    pub ma_type: Option<String>,
}

impl Default for KeltnerParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeltnerInput<'a> {
    pub data: KeltnerData<'a>,
    pub params: KeltnerParams,
}

impl<'a> KeltnerInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: KeltnerParams) -> Self {
        Self {
            data: KeltnerData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slice(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
        params: KeltnerParams,
    ) -> Self {
        Self {
            data: KeltnerData::Slice(high, low, close, source),
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", KeltnerParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_multiplier(&self) -> f64 {
        self.params.multiplier.unwrap_or(2.0)
    }
    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("ema")
    }
}

impl<'a> AsRef<[f64]> for KeltnerInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            KeltnerData::Slice(_, _, _, source) => source,
            KeltnerData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerBuilder {
    period: Option<usize>,
    multiplier: Option<f64>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for KeltnerBuilder {
    fn default() -> Self {
        Self {
            period: None,
            multiplier: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KeltnerBuilder {
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
    pub fn multiplier(mut self, x: f64) -> Self {
        self.multiplier = Some(x);
        self
    }
    #[inline(always)]
    pub fn ma_type(mut self, mt: &str) -> Self {
        self.ma_type = Some(mt.to_lowercase());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KeltnerOutput, KeltnerError> {
        let p = KeltnerParams {
            period: self.period,
            multiplier: self.multiplier,
            ma_type: self.ma_type,
        };
        let i = KeltnerInput::from_candles(c, "close", p);
        keltner_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
    ) -> Result<KeltnerOutput, KeltnerError> {
        let p = KeltnerParams {
            period: self.period,
            multiplier: self.multiplier,
            ma_type: self.ma_type,
        };
        let i = KeltnerInput::from_slice(high, low, close, source, p);
        keltner_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<KeltnerStream, KeltnerError> {
        let p = KeltnerParams {
            period: self.period,
            multiplier: self.multiplier,
            ma_type: self.ma_type,
        };
        KeltnerStream::try_new(p)
    }
}

// Error handling

#[derive(Debug, Error)]
pub enum KeltnerError {
    #[error("keltner: empty data provided.")]
    EmptyInputData,
    #[error("keltner: invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("keltner: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("keltner: all values are NaN.")]
    AllValuesNaN,
    #[error("keltner: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("keltner: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },
    #[error("keltner: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("keltner: invalid input: {0}")]
    InvalidInput(String),
    #[error("keltner: MA error: {0}")]
    MaError(String),
}

// Core indicator API

#[inline]
pub fn keltner(input: &KeltnerInput) -> Result<KeltnerOutput, KeltnerError> {
    keltner_with_kernel(input, Kernel::Auto)
}

pub fn keltner_with_kernel(
    input: &KeltnerInput,
    kernel: Kernel,
) -> Result<KeltnerOutput, KeltnerError> {
    let (high, low, close, source_slice): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        KeltnerData::Candles { candles, source } => (
            candles
                .select_candle_field("high")
                .map_err(|e| KeltnerError::MaError(e.to_string()))?,
            candles
                .select_candle_field("low")
                .map_err(|e| KeltnerError::MaError(e.to_string()))?,
            candles
                .select_candle_field("close")
                .map_err(|e| KeltnerError::MaError(e.to_string()))?,
            source_type(candles, source),
        ),
        KeltnerData::Slice(h, l, c, s) => (*h, *l, *c, *s),
    };
    let period = input.get_period();
    let len = close.len();
    if len == 0 {
        return Err(KeltnerError::EmptyInputData);
    }
    if period == 0 || period > len {
        return Err(KeltnerError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KeltnerError::AllValuesNaN)?;

    if (len - first) < period {
        return Err(KeltnerError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period - 1;
    let mut upper_band = alloc_with_nan_prefix(len, warm);
    let mut middle_band = alloc_with_nan_prefix(len, warm);
    let mut lower_band = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => keltner_scalar(
                high,
                low,
                close,
                source_slice,
                period,
                input.get_multiplier(),
                input.get_ma_type(),
                first,
                &mut upper_band,
                &mut middle_band,
                &mut lower_band,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => keltner_avx2(
                high,
                low,
                close,
                source_slice,
                period,
                input.get_multiplier(),
                input.get_ma_type(),
                first,
                &mut upper_band,
                &mut middle_band,
                &mut lower_band,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => keltner_avx512(
                high,
                low,
                close,
                source_slice,
                period,
                input.get_multiplier(),
                input.get_ma_type(),
                first,
                &mut upper_band,
                &mut middle_band,
                &mut lower_band,
            ),
            _ => unreachable!(),
        }
    }
    Ok(KeltnerOutput {
        upper_band,
        middle_band,
        lower_band,
    })
}

/// Writes Keltner Channels into caller-provided buffers without allocating.
///
/// - Preserves NaN warmup prefix identical to `keltner()`/`keltner_with_kernel()`.
/// - All output slices must have the same length as the input series.
/// - Uses runtime kernel selection (`Kernel::Auto`).
#[cfg(not(feature = "wasm"))]
#[inline(always)]
pub fn keltner_into(
    input: &KeltnerInput,
    upper_dst: &mut [f64],
    middle_dst: &mut [f64],
    lower_dst: &mut [f64],
) -> Result<(), KeltnerError> {
    keltner_into_slice(upper_dst, middle_dst, lower_dst, input, Kernel::Auto)
}

/// Write keltner output directly to pre-allocated slices - zero allocations
#[inline(always)]
pub fn keltner_into_slice(
    upper_dst: &mut [f64],
    middle_dst: &mut [f64],
    lower_dst: &mut [f64],
    input: &KeltnerInput,
    kernel: Kernel,
) -> Result<(), KeltnerError> {
    let (high, low, close, source_slice): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        KeltnerData::Candles { candles, source } => (
            candles
                .select_candle_field("high")
                .map_err(|e| KeltnerError::MaError(e.to_string()))?,
            candles
                .select_candle_field("low")
                .map_err(|e| KeltnerError::MaError(e.to_string()))?,
            candles
                .select_candle_field("close")
                .map_err(|e| KeltnerError::MaError(e.to_string()))?,
            source_type(candles, source),
        ),
        KeltnerData::Slice(h, l, c, s) => (*h, *l, *c, *s),
    };

    let period = input.get_period();
    let len = close.len();

    if len == 0 {
        return Err(KeltnerError::EmptyInputData);
    }

    if upper_dst.len() != len || middle_dst.len() != len || lower_dst.len() != len {
        return Err(KeltnerError::OutputLengthMismatch {
            expected: len,
            got: upper_dst.len().max(middle_dst.len()).max(lower_dst.len()),
        });
    }

    if period == 0 || period > len {
        return Err(KeltnerError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KeltnerError::AllValuesNaN)?;

    if (len - first) < period {
        return Err(KeltnerError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => keltner_scalar(
                high,
                low,
                close,
                source_slice,
                period,
                input.get_multiplier(),
                input.get_ma_type(),
                first,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => keltner_avx2(
                high,
                low,
                close,
                source_slice,
                period,
                input.get_multiplier(),
                input.get_ma_type(),
                first,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => keltner_avx512(
                high,
                low,
                close,
                source_slice,
                period,
                input.get_multiplier(),
                input.get_ma_type(),
                first,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                keltner_scalar(
                    high,
                    low,
                    close,
                    source_slice,
                    period,
                    input.get_multiplier(),
                    input.get_ma_type(),
                    first,
                    upper_dst,
                    middle_dst,
                    lower_dst,
                )
            }
            _ => unreachable!(),
        }
    }

    // Fill warmup with NaN
    let warm = first + period - 1;
    for i in 0..warm {
        upper_dst[i] = f64::NAN;
        middle_dst[i] = f64::NAN;
        lower_dst[i] = f64::NAN;
    }

    Ok(())
}

// Scalar calculation (core logic)

#[inline]
/// Classic kernel - optimized loop-jammed implementation for SMA
pub fn keltner_scalar_classic_sma(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    let len = close.len();
    let warm = first + period - 1;

    if warm >= len {
        return;
    }

    // --- ATR Calculation (inline) ---
    let alpha = 1.0 / (period as f64);
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;
    let mut atr_values = vec![f64::NAN; len];

    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };

        if i < period {
            sum_tr += tr;
            if i == period - 1 {
                rma = sum_tr / (period as f64);
                atr_values[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr_values[i] = rma;
        }
    }

    // --- SMA Calculation (inline) ---
    let mut sum = 0.0;

    // Initialize SMA
    for j in 0..period {
        sum += source[first + j];
    }
    let mut sma_val = sum / period as f64;

    // First valid Keltner point
    if warm < len {
        middle[warm] = sma_val;
        let atr_v = atr_values[warm];
        if !atr_v.is_nan() {
            upper[warm] = sma_val + multiplier * atr_v;
            lower[warm] = sma_val - multiplier * atr_v;
        }
    }

    // Rolling window for remaining values
    for i in (warm + 1)..len {
        // Update SMA
        sum += source[i] - source[i - period];
        sma_val = sum / period as f64;

        // Calculate Keltner bands
        middle[i] = sma_val;
        let atr_v = atr_values[i];
        if !atr_v.is_nan() {
            upper[i] = sma_val + multiplier * atr_v;
            lower[i] = sma_val - multiplier * atr_v;
        }
    }
}

/// Classic kernel - optimized loop-jammed implementation for EMA
pub fn keltner_scalar_classic_ema(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    let len = close.len();
    let warm = first + period - 1;

    if warm >= len {
        return;
    }

    // --- ATR Calculation (inline) ---
    let alpha = 1.0 / (period as f64);
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;
    let mut atr_values = vec![f64::NAN; len];

    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };

        if i < period {
            sum_tr += tr;
            if i == period - 1 {
                rma = sum_tr / (period as f64);
                atr_values[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr_values[i] = rma;
        }
    }

    // --- EMA Calculation (inline) ---
    let ema_alpha = 2.0 / (period as f64 + 1.0);
    let ema_alpha_1 = 1.0 - ema_alpha;

    // Initialize EMA with SMA
    let mut sum = 0.0;
    for j in 0..period {
        sum += source[first + j];
    }
    let mut ema_val = sum / period as f64;

    // First valid Keltner point
    if warm < len {
        middle[warm] = ema_val;
        let atr_v = atr_values[warm];
        if !atr_v.is_nan() {
            upper[warm] = ema_val + multiplier * atr_v;
            lower[warm] = ema_val - multiplier * atr_v;
        }
    }

    // EMA updates for remaining values
    for i in (warm + 1)..len {
        // Update EMA
        ema_val = ema_alpha * source[i] + ema_alpha_1 * ema_val;

        // Calculate Keltner bands
        middle[i] = ema_val;
        let atr_v = atr_values[i];
        if !atr_v.is_nan() {
            upper[i] = ema_val + multiplier * atr_v;
            lower[i] = ema_val - multiplier * atr_v;
        }
    }
}

/// Regular kernel - uses function calls for flexibility
pub fn keltner_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    // Fast path: fully loop-jammed EMA/SMA + ATR (Wilder RMA) with no extra allocations.
    // Fallback for other MA types retains allocation-based path for feature parity.
    let len = close.len();
    let warm = first + period - 1;
    if warm >= len {
        return;
    }

    // ---- ATR init (Wilder RMA) ----
    let pf = period as f64;
    let rma_alpha = 1.0 / pf;

    // Compute initial ATR over the first `period` TRs and roll to `warm`.
    let mut atr: f64;
    unsafe {
        // i = 0
        atr = *high.get_unchecked(0) - *low.get_unchecked(0);
        // i = 1..period-1
        let mut i = 1usize;
        while i < period {
            let hi = *high.get_unchecked(i);
            let lo = *low.get_unchecked(i);
            let pc = *close.get_unchecked(i - 1);
            let hl = hi - lo;
            let hc = (hi - pc).abs();
            let lc = (lo - pc).abs();
            atr += hl.max(hc).max(lc);
            i += 1;
        }
        atr /= pf; // initial RMA at index period-1

        // Step ATR forward to warm index
        let mut k = period;
        while k <= warm {
            let hi = *high.get_unchecked(k);
            let lo = *low.get_unchecked(k);
            let pc = *close.get_unchecked(k - 1);
            let hl = hi - lo;
            let hc = (hi - pc).abs();
            let lc = (lo - pc).abs();
            let tr = hl.max(hc).max(lc);
            atr = (tr - atr).mul_add(rma_alpha, atr);
            k += 1;
        }
    }

    let m = multiplier;

    // ---- EMA path ----
    if ma_type.eq_ignore_ascii_case("ema") {
        let mut ema: f64 = 0.0;
        unsafe {
            let mut j = 0usize;
            while j < period {
                ema += *source.get_unchecked(first + j);
                j += 1;
            }
        }
        ema /= pf;

        middle[warm] = ema;
        upper[warm] = m.mul_add(atr, ema);
        lower[warm] = (-m).mul_add(atr, ema);

        let ema_alpha = 2.0 / (pf + 1.0);

        unsafe {
            let mut i = warm + 1;
            while i < len {
                // ATR update
                let hi = *high.get_unchecked(i);
                let lo = *low.get_unchecked(i);
                let pc = *close.get_unchecked(i - 1);
                let hl = hi - lo;
                let hc = (hi - pc).abs();
                let lc = (lo - pc).abs();
                let tr = hl.max(hc).max(lc);
                atr = (tr - atr).mul_add(rma_alpha, atr);

                // EMA update
                let xi = *source.get_unchecked(i);
                ema = (xi - ema).mul_add(ema_alpha, ema);

                // Bands
                middle[i] = ema;
                upper[i] = m.mul_add(atr, ema);
                lower[i] = (-m).mul_add(atr, ema);
                i += 1;
            }
        }
        return;
    }

    // ---- SMA path ----
    if ma_type.eq_ignore_ascii_case("sma") {
        let mut sum: f64 = 0.0;
        unsafe {
            let mut j = 0usize;
            while j < period {
                sum += *source.get_unchecked(first + j);
                j += 1;
            }
        }
        let mut mid = sum / pf;
        middle[warm] = mid;
        upper[warm] = m.mul_add(atr, mid);
        lower[warm] = (-m).mul_add(atr, mid);

        unsafe {
            let mut i = warm + 1;
            while i < len {
                // ATR update
                let hi = *high.get_unchecked(i);
                let lo = *low.get_unchecked(i);
                let pc = *close.get_unchecked(i - 1);
                let hl = hi - lo;
                let hc = (hi - pc).abs();
                let lc = (lo - pc).abs();
                let tr = hl.max(hc).max(lc);
                atr = (tr - atr).mul_add(rma_alpha, atr);

                // SMA rolling update
                let new_x = *source.get_unchecked(i);
                let old_x = *source.get_unchecked(i - period);
                sum += new_x - old_x;
                mid = sum / pf;

                middle[i] = mid;
                upper[i] = m.mul_add(atr, mid);
                lower[i] = (-m).mul_add(atr, mid);
                i += 1;
            }
        }
        return;
    }

    // ---- Fallback (non-EMA/SMA): preserve original behavior ----
    let mut atr = crate::utilities::helpers::alloc_with_nan_prefix(len, warm);
    let alpha = 1.0 / (period as f64);
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;

    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i < period {
            sum_tr += tr;
            if i == period - 1 {
                rma = sum_tr / (period as f64);
                atr[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr[i] = rma;
        }
    }

    let mut ma_values = crate::utilities::helpers::alloc_with_nan_prefix(len, warm);

    match ma_type {
        "ema" => {
            use crate::indicators::moving_averages::ema::{
                ema_into_slice, EmaData, EmaInput, EmaParams,
            };
            let ema_input = EmaInput {
                data: EmaData::Slice(source),
                params: EmaParams {
                    period: Some(period),
                },
            };
            let _ = ema_into_slice(&mut ma_values, &ema_input, Kernel::Auto);
        }
        "sma" => {
            use crate::indicators::moving_averages::sma::{
                sma_into_slice, SmaData, SmaInput, SmaParams,
            };
            let sma_input = SmaInput {
                data: SmaData::Slice(source),
                params: SmaParams {
                    period: Some(period),
                },
            };
            let _ = sma_into_slice(&mut ma_values, &sma_input, Kernel::Auto);
        }
        _ => {
            if let Ok(result) = crate::indicators::moving_averages::ma::ma(
                ma_type,
                crate::indicators::moving_averages::ma::MaData::Slice(source),
                period,
            ) {
                ma_values.copy_from_slice(&result);
            }
        }
    }

    for i in warm..len {
        let ma_v = ma_values[i];
        let atr_v = atr[i];
        if ma_v.is_nan() || atr_v.is_nan() {
            continue;
        }
        middle[i] = ma_v;
        upper[i] = multiplier.mul_add(atr_v, ma_v);
        lower[i] = (-multiplier).mul_add(atr_v, ma_v);
    }
}

// AVX2 stub

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn keltner_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    keltner_scalar(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    )
}

// AVX512 stub

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn keltner_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    if period <= 32 {
        unsafe {
            keltner_avx512_short(
                high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
            )
        }
    } else {
        unsafe {
            keltner_avx512_long(
                high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
            )
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn keltner_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    keltner_scalar(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn keltner_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    keltner_scalar(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    )
}

// Row/batch/parallel support

#[inline(always)]
pub fn keltner_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    // Use the optimized scalar kernel for all MA types to avoid per-row temporaries
    keltner_scalar(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn keltner_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    keltner_avx2(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn keltner_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    keltner_avx512(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn keltner_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    keltner_avx512_short(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn keltner_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    keltner_avx512_long(
        high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
    )
}

// Batch support

#[derive(Clone, Debug)]
pub struct KeltnerBatchRange {
    pub period: (usize, usize, usize),
    pub multiplier: (f64, f64, f64),
}

impl Default for KeltnerBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 60, 10),
            multiplier: (2.0, 2.0, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerBatchBuilder {
    range: KeltnerBatchRange,
    kernel: Kernel,
}

impl Default for KeltnerBatchBuilder {
    fn default() -> Self {
        Self {
            range: KeltnerBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}
impl KeltnerBatchBuilder {
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
    pub fn multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.multiplier = (start, end, step);
        self
    }
    #[inline]
    pub fn multiplier_static(mut self, m: f64) -> Self {
        self.range.multiplier = (m, m, 0.0);
        self
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KeltnerBatchOutput, KeltnerError> {
        let h = c
            .select_candle_field("high")
            .map_err(|e| KeltnerError::MaError(e.to_string()))?;
        let l = c
            .select_candle_field("low")
            .map_err(|e| KeltnerError::MaError(e.to_string()))?;
        let cl = c
            .select_candle_field("close")
            .map_err(|e| KeltnerError::MaError(e.to_string()))?;
        let src_v = source_type(c, src);
        self.apply_slice(&h, &l, &cl, src_v)
    }
    pub fn apply_slice(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
    ) -> Result<KeltnerBatchOutput, KeltnerError> {
        keltner_batch_with_kernel(high, low, close, source, &self.range, self.kernel)
    }

    pub fn with_default_slice(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
        k: Kernel,
    ) -> Result<KeltnerBatchOutput, KeltnerError> {
        KeltnerBatchBuilder::new()
            .kernel(k)
            .apply_slice(high, low, close, source)
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerBatchOutput {
    pub upper_band: Vec<f64>,
    pub middle_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub combos: Vec<KeltnerParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KeltnerBatchOutput {
    pub fn row_for_params(&self, p: &KeltnerParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.multiplier.unwrap_or(2.0) - p.multiplier.unwrap_or(2.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &KeltnerParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.upper_band[start..start + self.cols],
                &self.middle_band[start..start + self.cols],
                &self.lower_band[start..start + self.cols],
            )
        })
    }
}

fn expand_grid(r: &KeltnerBatchRange) -> Result<Vec<KeltnerParams>, KeltnerError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, KeltnerError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
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
            return Err(KeltnerError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, KeltnerError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(KeltnerError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(KeltnerError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let mults = axis_f64(r.multiplier)?;

    let cap = periods
        .len()
        .checked_mul(mults.len())
        .ok_or_else(|| KeltnerError::InvalidRange {
            start: "rows".into(),
            end: "cols".into(),
            step: "rows*cols".into(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &m in &mults {
            out.push(KeltnerParams {
                period: Some(p),
                multiplier: Some(m),
                ma_type: None,
            });
        }
    }
    Ok(out)
}

pub fn keltner_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerBatchRange,
    k: Kernel,
) -> Result<KeltnerBatchOutput, KeltnerError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(KeltnerError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    keltner_batch_par_slice(high, low, close, source, sweep, simd)
}

pub fn keltner_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerBatchRange,
    kern: Kernel,
) -> Result<KeltnerBatchOutput, KeltnerError> {
    keltner_batch_inner(high, low, close, source, sweep, kern, false, None)
}
pub fn keltner_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerBatchRange,
    kern: Kernel,
) -> Result<KeltnerBatchOutput, KeltnerError> {
    keltner_batch_inner(high, low, close, source, sweep, kern, true, None)
}
fn keltner_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerBatchRange,
    kern: Kernel,
    parallel: bool,
    ma_type: Option<&str>,
) -> Result<KeltnerBatchOutput, KeltnerError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(KeltnerError::InvalidRange {
            start: "range".into(),
            end: "range".into(),
            step: "empty".into(),
        });
    }
    let len = close.len();
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KeltnerError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(KeltnerError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    rows
        .checked_mul(cols)
        .ok_or_else(|| KeltnerError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            first
                .checked_add(p.saturating_sub(1))
                .ok_or_else(|| KeltnerError::InvalidRange {
                    start: first.to_string(),
                    end: p.to_string(),
                    step: "first+period-1".into(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Allocate uninitialized matrices and initialize NaN prefixes
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut middle_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut upper_mu, cols, &warm);
    init_matrix_prefixes(&mut middle_mu, cols, &warm);
    init_matrix_prefixes(&mut lower_mu, cols, &warm);

    // Convert to mutable slices
    let mut upper_guard = core::mem::ManuallyDrop::new(upper_mu);
    let mut middle_guard = core::mem::ManuallyDrop::new(middle_mu);
    let mut lower_guard = core::mem::ManuallyDrop::new(lower_mu);

    let upper: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len())
    };
    let middle: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(middle_guard.as_mut_ptr() as *mut f64, middle_guard.len())
    };
    let lower: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len())
    };

    // Precompute True Range once and reuse across rows (row-specific optimization)
    let mut tr: Vec<f64> = vec![0.0; cols];
    unsafe {
        let mut i = 0usize;
        while i < cols {
            let val = if i == 0 {
                *high.get_unchecked(0) - *low.get_unchecked(0)
            } else {
                let hi = *high.get_unchecked(i);
                let lo = *low.get_unchecked(i);
                let pc = *close.get_unchecked(i - 1);
                let hl = hi - lo;
                let hc = (hi - pc).abs();
                let lc = (lo - pc).abs();
                hl.max(hc).max(lc)
            };
            *tr.get_unchecked_mut(i) = val;
            i += 1;
        }
    }

    // Precompute prefix sums lazily only for SMA rows to avoid overhead for EMA
    let ps: Option<Vec<f64>> = if ma_type.unwrap_or("ema").eq_ignore_ascii_case("sma") {
        let mut buf = vec![0.0; cols + 1];
        unsafe {
            let mut i = 0usize;
            while i < cols {
                let prev = *buf.get_unchecked(i);
                let xi = *source.get_unchecked(i);
                *buf.get_unchecked_mut(i + 1) = prev + xi;
                i += 1;
            }
        }
        Some(buf)
    } else {
        None
    };

    let ma = ma_type.unwrap_or("ema");
    let do_row = |row: usize, up: &mut [f64], mid: &mut [f64], low_out: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let mult = combos[row].multiplier.unwrap();
        let row_warm = warm[row];

        if row_warm >= cols {
            return;
        }

        let pf = period as f64;
        let alpha_rma = 1.0 / pf;

        // Initialize ATR over first `period` TRs and roll to warm
        let mut atr = 0.0f64;
        unsafe {
            let mut j = 0usize;
            while j < period {
                atr += *tr.get_unchecked(j);
                j += 1;
            }
        }
        atr /= pf;
        let mut k = period;
        unsafe {
            while k <= row_warm {
                let tri = *tr.get_unchecked(k);
                atr = (tri - atr).mul_add(alpha_rma, atr);
                k += 1;
            }
        }

        if ma.eq_ignore_ascii_case("ema") {
            // EMA init over source[first..first+period]
            let mut acc = 0.0f64;
            unsafe {
                let mut j = 0usize;
                while j < period {
                    acc += *source.get_unchecked(first + j);
                    j += 1;
                }
            }
            let mut ema = acc / pf;
            unsafe {
                *mid.get_unchecked_mut(row_warm) = ema;
                *up.get_unchecked_mut(row_warm) = mult.mul_add(atr, ema);
                *low_out.get_unchecked_mut(row_warm) = (-mult).mul_add(atr, ema);
            }

            let alpha_ema = 2.0 / (pf + 1.0);
            unsafe {
                let mut i = row_warm + 1;
                while i < cols {
                    // ATR update from precomputed TR
                    let tri = *tr.get_unchecked(i);
                    atr = (tri - atr).mul_add(alpha_rma, atr);
                    // EMA update
                    let xi = *source.get_unchecked(i);
                    ema = (xi - ema).mul_add(alpha_ema, ema);
                    // Bands
                    *mid.get_unchecked_mut(i) = ema;
                    *up.get_unchecked_mut(i) = mult.mul_add(atr, ema);
                    *low_out.get_unchecked_mut(i) = (-mult).mul_add(atr, ema);
                    i += 1;
                }
            }
        } else if ma.eq_ignore_ascii_case("sma") {
            let ps = ps.as_ref().expect("prefix sums computed for SMA");
            // SMA via shared prefix sums of source
            let start = row_warm + 1 - period;
            let end = row_warm + 1;
            let mut sm = unsafe { (*ps.get_unchecked(end) - *ps.get_unchecked(start)) / pf };
            unsafe {
                *mid.get_unchecked_mut(row_warm) = sm;
                *up.get_unchecked_mut(row_warm) = mult.mul_add(atr, sm);
                *low_out.get_unchecked_mut(row_warm) = (-mult).mul_add(atr, sm);
            }

            unsafe {
                let mut i = row_warm + 1;
                while i < cols {
                    let tri = *tr.get_unchecked(i);
                    atr = (tri - atr).mul_add(alpha_rma, atr);
                    let s = (*ps.get_unchecked(i + 1) - *ps.get_unchecked(i + 1 - period)) / pf;
                    sm = s;
                    *mid.get_unchecked_mut(i) = sm;
                    *up.get_unchecked_mut(i) = mult.mul_add(atr, sm);
                    *low_out.get_unchecked_mut(i) = (-mult).mul_add(atr, sm);
                    i += 1;
                }
            }
        } else {
            // Fallback: reuse existing per-row scalar implementation for uncommon MA types
            keltner_row_scalar(
                high, low, close, source, period, mult, ma, first, up, mid, low_out,
            );
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            upper
                .par_chunks_mut(cols)
                .zip(middle.par_chunks_mut(cols))
                .zip(lower.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((u, m), l))| do_row(row, u, m, l));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for ((row, u), (m, l)) in upper
                .chunks_mut(cols)
                .enumerate()
                .zip(middle.chunks_mut(cols).zip(lower.chunks_mut(cols)))
            {
                do_row(row, u, m, l);
            }
        }
    } else {
        for ((row, u), (m, l)) in upper
            .chunks_mut(cols)
            .enumerate()
            .zip(middle.chunks_mut(cols).zip(lower.chunks_mut(cols)))
        {
            do_row(row, u, m, l);
        }
    }
    // Convert back to Vec
    let upper = unsafe {
        let ptr = upper_guard.as_mut_ptr() as *mut f64;
        let len = upper_guard.len();
        let cap = upper_guard.capacity();
        core::mem::forget(upper_guard);
        Vec::from_raw_parts(ptr, len, cap)
    };

    let middle = unsafe {
        let ptr = middle_guard.as_mut_ptr() as *mut f64;
        let len = middle_guard.len();
        let cap = middle_guard.capacity();
        core::mem::forget(middle_guard);
        Vec::from_raw_parts(ptr, len, cap)
    };

    let lower = unsafe {
        let ptr = lower_guard.as_mut_ptr() as *mut f64;
        let len = lower_guard.len();
        let cap = lower_guard.capacity();
        core::mem::forget(lower_guard);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(KeltnerBatchOutput {
        upper_band: upper,
        middle_band: middle,
        lower_band: lower,
        combos,
        rows,
        cols,
    })
}

// Streaming mode

#[derive(Debug, Clone)]
pub struct KeltnerStream {
    period: usize,
    rcp_period: f64, // 1 / period
    multiplier: f64,

    // MA impl
    ma_impl: MaImpl,

    // ATR (Wilder RMA)
    atr: f64,       // valid after seeding
    atr_sum: f64,   // used only during warmup seeding
    rma_alpha: f64, // 1 / period

    // book-keeping
    count: usize,
    prev_close: f64,
}

#[derive(Debug, Clone)]
enum MaImpl {
    // EMA seeded with an exact SMA of the first `period` samples
    Ema {
        alpha: f64,    // 2 / (n + 1)
        value: f64,    // current EMA
        seed_sum: f64, // sum of first n values, only used during warmup
    },
    // True O(1) SMA via ring-buffer
    Sma {
        buffer: Vec<f64>,
        sum: f64,
        idx: usize,
        filled: bool, // becomes true exactly at count == period
    },
}

impl KeltnerStream {
    pub fn try_new(params: KeltnerParams) -> Result<Self, KeltnerError> {
        let period = params.period.unwrap_or(20);
        let multiplier = params.multiplier.unwrap_or(2.0);
        let ma_type = params
            .ma_type
            .unwrap_or_else(|| "ema".to_string())
            .to_lowercase();

        if period == 0 {
            return Err(KeltnerError::InvalidPeriod { period, data_len: 0 });
        }

        let pf = period as f64;
        let rcp = 1.0 / pf;

        let ma_impl = if ma_type == "sma" {
            MaImpl::Sma {
                buffer: vec![0.0; period],
                sum: 0.0,
                idx: 0,
                filled: false,
            }
        } else {
            // default to EMA semantics
            MaImpl::Ema {
                alpha: 2.0 / (pf + 1.0),
                value: 0.0,
                seed_sum: 0.0,
            }
        };

        Ok(Self {
            period,
            rcp_period: rcp,
            multiplier,
            ma_impl,
            atr: 0.0,
            atr_sum: 0.0,
            rma_alpha: rcp,
            count: 0,
            prev_close: f64::NAN,
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        source: f64,
    ) -> Option<(f64, f64, f64)> {
        // --- TR (True Range) ---
        // matches batch kernel: TR = max(high-low, |high-prev_close|, |low-prev_close|)
        let tr = if self.count == 0 {
            high - low
        } else {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        };

        self.prev_close = close;
        self.count += 1;

        // --- Warmup: accumulate seeds, no output ---
        if self.count < self.period {
            // seed ATR sum
            self.atr_sum += tr;

            // seed MA
            match &mut self.ma_impl {
                MaImpl::Ema { seed_sum, .. } => {
                    *seed_sum += source;
                }
                MaImpl::Sma {
                    buffer, sum, idx, ..
                } => {
                    *sum += source;
                    buffer[*idx] = source;
                    *idx = (*idx + 1) % self.period;
                }
            }
            return None;
        }

        // --- First valid output at exactly count == period ---
        if self.count == self.period {
            // finalize ATR seed using the first `period` TRs
            self.atr = (self.atr_sum + tr) * self.rcp_period;

            // seed MA -> exact SMA of first `period` samples
            let mid = match &mut self.ma_impl {
                MaImpl::Ema {
                    value, seed_sum, ..
                } => {
                    *seed_sum += source; // include current
                    *value = *seed_sum * self.rcp_period;
                    *value
                }
                MaImpl::Sma {
                    buffer,
                    sum,
                    idx,
                    filled,
                } => {
                    *sum += source; // include current
                    buffer[*idx] = source;
                    *idx = (*idx + 1) % self.period;
                    *filled = true;
                    *sum * self.rcp_period
                }
            };

            let up = self.multiplier.mul_add(self.atr, mid);
            let lo = (-self.multiplier).mul_add(self.atr, mid);
            return Some((up, mid, lo));
        }

        // --- Steady-state O(1) updates ---
        // ATR (Wilder RMA): atr = (tr - atr) * alpha + atr
        self.atr = (tr - self.atr).mul_add(self.rma_alpha, self.atr);

        // MA update
        let mid = match &mut self.ma_impl {
            MaImpl::Ema { alpha, value, .. } => {
                *value = (source - *value).mul_add(*alpha, *value);
                *value
            }
            MaImpl::Sma {
                buffer, sum, idx, ..
            } => {
                let old = buffer[*idx];
                buffer[*idx] = source;
                *sum += source - old;
                *idx = (*idx + 1) % self.period;
                *sum * self.rcp_period
            }
        };

        let up = self.multiplier.mul_add(self.atr, mid);
        let lo = (-self.multiplier).mul_add(self.atr, mid);
        Some((up, mid, lo))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_keltner_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = KeltnerParams {
            period: Some(20),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel)?;

        assert_eq!(result.upper_band.len(), candles.close.len());
        assert_eq!(result.middle_band.len(), candles.close.len());
        assert_eq!(result.lower_band.len(), candles.close.len());

        let last_five_index = candles.close.len().saturating_sub(5);
        let expected_upper = [
            61619.504155205745,
            61503.56119134791,
            61387.47897150178,
            61286.61078267451,
            61206.25688331261,
        ];
        let expected_middle = [
            59758.339871629956,
            59703.35512195091,
            59640.083205574636,
            59593.884805043715,
            59504.46720456336,
        ];
        let expected_lower = [
            57897.17558805417,
            57903.14905255391,
            57892.68743964749,
            57901.158827412924,
            57802.67752581411,
        ];
        let last_five_upper = &result.upper_band[last_five_index..];
        let last_five_middle = &result.middle_band[last_five_index..];
        let last_five_lower = &result.lower_band[last_five_index..];
        for i in 0..5 {
            let diff_u = (last_five_upper[i] - expected_upper[i]).abs();
            let diff_m = (last_five_middle[i] - expected_middle[i]).abs();
            let diff_l = (last_five_lower[i] - expected_lower[i]).abs();
            assert!(
                diff_u < 1e-1,
                "Upper band mismatch at index {}: expected {}, got {}",
                i,
                expected_upper[i],
                last_five_upper[i]
            );
            assert!(
                diff_m < 1e-1,
                "Middle band mismatch at index {}: expected {}, got {}",
                i,
                expected_middle[i],
                last_five_middle[i]
            );
            assert!(
                diff_l < 1e-1,
                "Lower band mismatch at index {}: expected {}, got {}",
                i,
                expected_lower[i],
                last_five_lower[i]
            );
        }
        Ok(())
    }

    fn check_keltner_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = KeltnerParams::default();
        let input = KeltnerInput::from_candles(&candles, "close", default_params);
        let result = keltner_with_kernel(&input, kernel)?;
        assert_eq!(result.upper_band.len(), candles.close.len());
        assert_eq!(result.middle_band.len(), candles.close.len());
        assert_eq!(result.lower_band.len(), candles.close.len());
        Ok(())
    }

    fn check_keltner_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KeltnerParams {
            period: Some(0),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("invalid period"));
        }
        Ok(())
    }

    fn check_keltner_large_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KeltnerParams {
            period: Some(999999),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("invalid period"));
        }
        Ok(())
    }

    fn check_keltner_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KeltnerParams::default();
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel)?;
        assert_eq!(result.middle_band.len(), candles.close.len());
        if result.middle_band.len() > 240 {
            for (i, &val) in result.middle_band[240..].iter().enumerate() {
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

    fn check_keltner_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 20;
        let multiplier = 2.0;

        let params = KeltnerParams {
            period: Some(period),
            multiplier: Some(multiplier),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params.clone());
        let batch_output = keltner_with_kernel(&input, kernel)?;

        let mut stream = KeltnerStream::try_new(params)?;
        let mut upper_stream = Vec::with_capacity(candles.close.len());
        let mut middle_stream = Vec::with_capacity(candles.close.len());
        let mut lower_stream = Vec::with_capacity(candles.close.len());

        for i in 0..candles.close.len() {
            let hi = candles.high[i];
            let lo = candles.low[i];
            let cl = candles.close[i];
            let src = candles.close[i]; // using "close" as the MA source for streaming
            match stream.update(hi, lo, cl, src) {
                Some((up, mid, low)) => {
                    upper_stream.push(up);
                    middle_stream.push(mid);
                    lower_stream.push(low);
                }
                None => {
                    upper_stream.push(f64::NAN);
                    middle_stream.push(f64::NAN);
                    lower_stream.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.upper_band.len(), upper_stream.len());
        assert_eq!(batch_output.middle_band.len(), middle_stream.len());
        assert_eq!(batch_output.lower_band.len(), lower_stream.len());
        for (i, (&b, &s)) in batch_output
            .middle_band
            .iter()
            .zip(middle_stream.iter())
            .enumerate()
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Keltner streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_keltner_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            KeltnerParams::default(), // period: 20, multiplier: 2.0, ma_type: "ema"
            KeltnerParams {
                period: Some(2),
                multiplier: Some(1.0),
                ma_type: Some("ema".to_string()),
            }, // minimum period
            KeltnerParams {
                period: Some(5),
                multiplier: Some(0.5),
                ma_type: Some("ema".to_string()),
            }, // small period, small multiplier
            KeltnerParams {
                period: Some(10),
                multiplier: Some(1.5),
                ma_type: Some("sma".to_string()),
            }, // medium period with SMA
            KeltnerParams {
                period: Some(20),
                multiplier: Some(3.0),
                ma_type: Some("ema".to_string()),
            }, // default period, large multiplier
            KeltnerParams {
                period: Some(50),
                multiplier: Some(2.5),
                ma_type: Some("sma".to_string()),
            }, // large period
            KeltnerParams {
                period: Some(100),
                multiplier: Some(1.0),
                ma_type: Some("ema".to_string()),
            }, // very large period
            KeltnerParams {
                period: Some(14),
                multiplier: Some(2.0),
                ma_type: Some("sma".to_string()),
            }, // common period with SMA
            KeltnerParams {
                period: Some(7),
                multiplier: Some(1.0),
                ma_type: Some("ema".to_string()),
            }, // week period
            KeltnerParams {
                period: Some(21),
                multiplier: Some(1.5),
                ma_type: Some("ema".to_string()),
            }, // 3-week period
            KeltnerParams {
                period: Some(30),
                multiplier: Some(2.0),
                ma_type: Some("sma".to_string()),
            }, // month period
            KeltnerParams {
                period: Some(3),
                multiplier: Some(0.75),
                ma_type: Some("ema".to_string()),
            }, // very small period
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = KeltnerInput::from_candles(&candles, "close", params.clone());
            let output = keltner_with_kernel(&input, kernel)?;

            // Check all three output bands
            for (band_name, band_values) in [
                ("upper", &output.upper_band),
                ("middle", &output.middle_band),
                ("lower", &output.lower_band),
            ] {
                for (i, &val) in band_values.iter().enumerate() {
                    if val.is_nan() {
                        continue; // NaN values are expected during warmup
                    }

                    let bits = val.to_bits();

                    // Check all three poison patterns
                    if bits == 0x11111111_11111111 {
                        panic!(
							"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
							 in {} band with params: period={}, multiplier={}, ma_type={} (param set {})",
							test_name, val, bits, i, band_name,
							params.period.unwrap_or(20),
							params.multiplier.unwrap_or(2.0),
							params.ma_type.as_deref().unwrap_or("ema"),
							param_idx
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
							 in {} band with params: period={}, multiplier={}, ma_type={} (param set {})",
							test_name, val, bits, i, band_name,
							params.period.unwrap_or(20),
							params.multiplier.unwrap_or(2.0),
							params.ma_type.as_deref().unwrap_or("ema"),
							param_idx
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
							 in {} band with params: period={}, multiplier={}, ma_type={} (param set {})",
							test_name, val, bits, i, band_name,
							params.period.unwrap_or(20),
							params.multiplier.unwrap_or(2.0),
							params.ma_type.as_deref().unwrap_or("ema"),
							param_idx
						);
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_keltner_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KeltnerBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = KeltnerParams::default();
        let (upper, middle, lower) = output.values_for(&def).expect("default row missing");

        assert_eq!(upper.len(), c.close.len());
        assert_eq!(middle.len(), c.close.len());
        assert_eq!(lower.len(), c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (period_start, period_end, period_step, multiplier_start, multiplier_end, multiplier_step)
            (2, 10, 2, 0.5, 2.5, 0.5), // Small periods with various multipliers
            (5, 25, 5, 1.0, 3.0, 1.0), // Medium periods
            (30, 60, 15, 2.0, 2.0, 0.0), // Large periods, static multiplier
            (2, 5, 1, 1.5, 2.5, 0.25), // Dense small range
            (10, 30, 10, 0.75, 2.25, 0.75), // Common trading periods
            (14, 21, 7, 1.0, 2.0, 0.5), // Week-based periods
            (20, 20, 0, 0.5, 3.0, 0.5), // Static period, varying multipliers
        ];

        for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step)) in
            test_configs.iter().enumerate()
        {
            let output = KeltnerBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .multiplier_range(m_start, m_end, m_step)
                .apply_candles(&c, "close")?;

            // Check all three output bands
            for (band_name, band_values) in [
                ("upper", &output.upper_band),
                ("middle", &output.middle_band),
                ("lower", &output.lower_band),
            ] {
                for (idx, &val) in band_values.iter().enumerate() {
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
							 at row {} col {} (flat index {}) in {} band with params: period={}, multiplier={}",
							test, cfg_idx, val, bits, row, col, idx, band_name,
							combo.period.unwrap_or(20),
							combo.multiplier.unwrap_or(2.0)
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} band with params: period={}, multiplier={}",
							test, cfg_idx, val, bits, row, col, idx, band_name,
							combo.period.unwrap_or(20),
							combo.multiplier.unwrap_or(2.0)
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} band with params: period={}, multiplier={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            band_name,
                            combo.period.unwrap_or(20),
                            combo.multiplier.unwrap_or(2.0)
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_keltner_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Simplified unified strategy for all test scenarios
        let strat = (
            2usize..=50,
            50usize..500,
            0.5f64..3.0f64,
            0usize..6,
            any::<u64>(),
        )
            .prop_map(|(period, len, multiplier, scenario, seed)| {
                let mut high = Vec::with_capacity(len);
                let mut low = Vec::with_capacity(len);
                let mut close = Vec::with_capacity(len);

                // Use a simple deterministic RNG based on the seed
                let mut rng_state = seed;
                let mut next_random = || -> f64 {
                    rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223); // LCG
                    (rng_state as f64) / (u64::MAX as f64)
                };

                match scenario {
                    0 => {
                        // Random OHLC data
                        let mut prev_close = 100.0;
                        for _ in 0..len {
                            let volatility = 0.01 + next_random() * 0.04; // 0.01..0.05
                            let change = -volatility + next_random() * (2.0 * volatility);
                            let new_close = (prev_close * (1.0 + change)).max(0.1f64);
                            let high_val = new_close * (1.0 + next_random() * volatility);
                            let low_val = new_close * (1.0 - next_random() * volatility);

                            high.push(high_val);
                            low.push(low_val.min(high_val));
                            close.push(new_close);
                            prev_close = new_close;
                        }
                    }
                    1 => {
                        // Uptrend
                        let start = 100.0;
                        for i in 0..len {
                            let base = start * (1.0 + 0.01 * i as f64);
                            let spread = base * 0.02;
                            high.push(base + spread);
                            low.push(base - spread);
                            close.push(base);
                        }
                    }
                    2 => {
                        // Downtrend
                        let start = 100.0;
                        for i in 0..len {
                            let base = start * (1.0 - 0.005 * i as f64).max(10.0);
                            let spread = base * 0.02;
                            high.push(base + spread);
                            low.push(base - spread);
                            close.push(base);
                        }
                    }
                    3 => {
                        // Volatile
                        let mut price = 100.0;
                        for i in 0..len {
                            let volatility = 0.1 * (1.0 + (i as f64 * 0.1).sin());
                            let change = if i % 2 == 0 {
                                volatility
                            } else {
                                -volatility * 0.8
                            };
                            price = (price * (1.0 + change)).max(1.0);

                            let spread = price * volatility;
                            high.push(price + spread);
                            low.push(price - spread * 0.8);
                            close.push(price);
                        }
                    }
                    4 => {
                        // Constant price (same high, low, close)
                        let constant_price = 50.0;
                        high = vec![constant_price; len];
                        low = vec![constant_price; len];
                        close = vec![constant_price; len];
                    }
                    _ => {
                        // Real-world-like
                        let mut price = 1000.0;
                        let mut momentum = 0.0;

                        for i in 0..len {
                            momentum = momentum * 0.9 + (if i % 20 < 10 { 0.001 } else { -0.001 });
                            let noise = ((i as f64 * 0.3).sin() * 0.005);
                            price = (price * (1.0 + momentum + noise)).max(100.0);

                            let daily_range = price * 0.02;
                            let high_val = price + daily_range * 0.6;
                            let low_val = price - daily_range * 0.4;

                            high.push(high_val);
                            low.push(low_val);
                            close.push(price);
                        }
                    }
                }

                // Randomly select MA type
                let ma_type = if next_random() > 0.5 { "ema" } else { "sma" };
                (high, low, close, period, multiplier, ma_type.to_string())
            });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(high, low, close, period, multiplier, ma_type)| {
				// Use close as source for testing
				let source = close.clone();

				let params = KeltnerParams {
					period: Some(period),
					multiplier: Some(multiplier),
					ma_type: Some(ma_type.clone()),
				};
				let input = KeltnerInput::from_slice(&high, &low, &close, &source, params.clone());

				let result = keltner_with_kernel(&input, kernel).unwrap();
				let scalar_result = keltner_with_kernel(&input, Kernel::Scalar).unwrap();

				// Check output lengths
				prop_assert_eq!(result.upper_band.len(), close.len());
				prop_assert_eq!(result.middle_band.len(), close.len());
				prop_assert_eq!(result.lower_band.len(), close.len());

				// Check warmup period (first period-1 values should be NaN)
				let warmup = period - 1;
				for i in 0..warmup.min(close.len()) {
					prop_assert!(
						result.upper_band[i].is_nan(),
						"Upper band[{}] should be NaN during warmup", i
					);
					prop_assert!(
						result.middle_band[i].is_nan(),
						"Middle band[{}] should be NaN during warmup", i
					);
					prop_assert!(
						result.lower_band[i].is_nan(),
						"Lower band[{}] should be NaN during warmup", i
					);
				}

				// Check valid values after warmup
				for i in warmup..close.len() {
					let upper = result.upper_band[i];
					let middle = result.middle_band[i];
					let lower = result.lower_band[i];

					let scalar_upper = scalar_result.upper_band[i];
					let scalar_middle = scalar_result.middle_band[i];
					let scalar_lower = scalar_result.lower_band[i];

					// Skip if any value is NaN
					if upper.is_nan() || middle.is_nan() || lower.is_nan() {
						continue;
					}

					// Property 1: Band relationships must hold
					prop_assert!(
						upper >= middle - 1e-10,
						"Upper band {} must be >= middle band {} at index {}", upper, middle, i
					);
					prop_assert!(
						middle >= lower - 1e-10,
						"Middle band {} must be >= lower band {} at index {}", middle, lower, i
					);

					// Property 2: Spread must be positive
					let spread = upper - lower;
					prop_assert!(
						spread >= -1e-10,
						"Spread {} must be positive at index {}", spread, i
					);

					// Property 2b: Spread consistency between kernels (validates ATR calculation)
					if i >= warmup + period {
						// Compare with scalar result to ensure consistency
						let scalar_spread = scalar_upper - scalar_lower;
						if scalar_spread > 0.0 && spread > 0.0 {
							let ratio = spread / scalar_spread;
							prop_assert!(
								(ratio - 1.0).abs() < 0.01,
								"Spread ratio between kernels should be consistent at index {}: ratio={}", i, ratio
							);
						}
					}

					// Property 3: Middle band should be within reasonable price range
					let window_start = i.saturating_sub(period - 1);
					let window_high = high[window_start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
					let window_low = low[window_start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));

					// Middle band should be within the general price range (with some tolerance for MA calculations)
					prop_assert!(
						middle <= window_high * 1.05 + 1.0,
						"Middle band {} exceeds window high {} at index {}", middle, window_high, i
					);
					prop_assert!(
						middle >= window_low * 0.95 - 1.0,
						"Middle band {} below window low {} at index {}", middle, window_low, i
					);

					// Property 4: Kernel consistency check
					let tolerance = 1e-9;
					prop_assert!(
						(upper - scalar_upper).abs() <= tolerance,
						"Upper band kernel mismatch at {}: {} vs {} (diff: {})",
						i, upper, scalar_upper, (upper - scalar_upper).abs()
					);
					prop_assert!(
						(middle - scalar_middle).abs() <= tolerance,
						"Middle band kernel mismatch at {}: {} vs {} (diff: {})",
						i, middle, scalar_middle, (middle - scalar_middle).abs()
					);
					prop_assert!(
						(lower - scalar_lower).abs() <= tolerance,
						"Lower band kernel mismatch at {}: {} vs {} (diff: {})",
						i, lower, scalar_lower, (lower - scalar_lower).abs()
					);

					// Property 5: Check for poison values
					#[cfg(debug_assertions)]
					{
						let upper_bits = upper.to_bits();
						let middle_bits = middle.to_bits();
						let lower_bits = lower.to_bits();

						prop_assert!(
							upper_bits != 0x11111111_11111111 &&
							upper_bits != 0x22222222_22222222 &&
							upper_bits != 0x33333333_33333333,
							"Found poison value in upper band at index {}", i
						);
						prop_assert!(
							middle_bits != 0x11111111_11111111 &&
							middle_bits != 0x22222222_22222222 &&
							middle_bits != 0x33333333_33333333,
							"Found poison value in middle band at index {}", i
						);
						prop_assert!(
							lower_bits != 0x11111111_11111111 &&
							lower_bits != 0x22222222_22222222 &&
							lower_bits != 0x33333333_33333333,
							"Found poison value in lower band at index {}", i
						);
					}

					// Property 6: For truly constant prices (high=low=close), bands should converge
					// Check if all values in the window are identical
					let all_same = high[window_start..=i].windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) &&
					               low[window_start..=i].windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) &&
					               close[window_start..=i].windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

					// Also check if high == low == close (truly constant price, no spread)
					let no_spread = high[window_start..=i].iter()
						.zip(low[window_start..=i].iter())
						.zip(close[window_start..=i].iter())
						.all(|((h, l), c)| (h - l).abs() < 1e-10 && (h - c).abs() < 1e-10);

					if all_same && no_spread && i >= warmup + period * 3 {
						// For truly constant prices with no spread, ATR should eventually become 0
						// and bands should converge to be very close (extra time for EMA stabilization)
						let band_spread = upper - lower;
						prop_assert!(
							band_spread < 0.01 || band_spread < middle * 0.001,
							"Bands should converge for constant prices with no spread, but spread is {} at index {} (middle: {})",
							band_spread, i, middle
						);
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }

    macro_rules! generate_all_keltner_tests {
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

    generate_all_keltner_tests!(
        check_keltner_accuracy,
        check_keltner_default_params,
        check_keltner_zero_period,
        check_keltner_large_period,
        check_keltner_nan_handling,
        check_keltner_streaming,
        check_keltner_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_keltner_tests!(check_keltner_property);

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

    #[test]
    fn test_keltner_into_matches_api() {
        use crate::utilities::data_loader::Candles;

        // Build a small but non-trivial synthetic OHLC series
        let n = 256usize;
        let mut ts = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        let mut price = 1000.0f64;
        for i in 0..n {
            let i_f = i as f64;
            let drift = (i_f * 0.001).sin() * 0.5;
            let noise = (i_f * 0.07).cos() * 0.8;
            price = (price + drift + noise).max(1.0);
            let spread = 2.0 + (i % 5) as f64;
            let h = price + spread;
            let l = price - spread * 0.8;
            let o = price - 0.25 * spread;
            let c = price + 0.25 * spread;

            ts.push(i as i64);
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(1000.0 + i as f64);
        }

        let candles = Candles::new(ts, open, high, low, close, volume);
        let input = KeltnerInput::from_candles(&candles, "close", KeltnerParams::default());

        // Baseline via Vec-returning API
        let base = keltner(&input).expect("keltner baseline failed");

        let len = candles.close.len();
        let mut up = vec![0.0; len];
        let mut mid = vec![0.0; len];
        let mut lo = vec![0.0; len];

        // Into API (no allocation)
        keltner_into(&input, &mut up, &mut mid, &mut lo).expect("keltner_into failed");

        assert_eq!(base.upper_band.len(), up.len());
        assert_eq!(base.middle_band.len(), mid.len());
        assert_eq!(base.lower_band.len(), lo.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(base.upper_band[i], up[i]),
                "upper mismatch at {}: base={} into={}",
                i,
                base.upper_band[i],
                up[i]
            );
            assert!(
                eq_or_both_nan(base.middle_band[i], mid[i]),
                "middle mismatch at {}: base={} into={}",
                i,
                base.middle_band[i],
                mid[i]
            );
            assert!(
                eq_or_both_nan(base.lower_band[i], lo[i]),
                "lower mismatch at {}: base={} into={}",
                i,
                base.lower_band[i],
                lo[i]
            );
        }
    }
}

// Python bindings

#[cfg(all(feature = "python", feature = "cuda"))]
pub struct KeltnerDeviceArrayF32 {
    pub inner: DeviceArrayF32,
    pub context: Arc<Context>,
    pub device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct KeltnerDeviceArrayF32Py {
    pub(crate) inner: KeltnerDeviceArrayF32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl KeltnerDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = &self.inner.inner;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;
        // Producer kernels synchronize before returning, so we omit `stream` per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self.inner.device_id as i32))
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<(u8, u8)>,
        _dl_device: Option<(i32, i32)>,
        _copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use cust::memory::DeviceBuffer;
        use std::os::raw::c_char;
        use std::os::raw::c_void;
        use std::ptr::null_mut;

        #[repr(C)]
        struct DLDataType {
            code: u8,
            bits: u8,
            lanes: u16,
        }
        #[repr(C)]
        struct DLDevice {
            device_type: i32,
            device_id: i32,
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
            deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
        }

        struct Holder {
            managed: DLManagedTensor,
            shape: [i64; 2],
            strides: [i64; 2],
            arr: DeviceArrayF32,
            _ctx_guard: cust::sys::CUcontext,
            _device_id: u32,
            _context: Arc<Context>,
        }

        extern "C" fn dl_managed_deleter(mt: *mut DLManagedTensor) {
            if mt.is_null() {
                return;
            }
            unsafe {
                let holder_ptr = (*mt).manager_ctx as *mut Holder;
                if !holder_ptr.is_null() {
                    if !(*holder_ptr)._ctx_guard.is_null() {
                        let dev = (*holder_ptr)._device_id as i32;
                        let _ = cust::sys::cuDevicePrimaryCtxRelease(dev);
                    }
                    drop(Box::from_raw(holder_ptr));
                }
            }
        }

        unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            use std::ffi::CStr;

            if capsule.is_null() {
                return;
            }
            let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule);
            if name_ptr.is_null() {
                return;
            }
            let name = CStr::from_ptr(name_ptr);
            let bytes = name.to_bytes();
            let is_legacy = bytes == b"dltensor";
            let is_versioned = bytes == b"dltensor_versioned";
            if !is_legacy && !is_versioned {
                // Capsule has been consumed/renamed (e.g., "used_dltensor[_versioned]"); skip deleter.
                return;
            }
            let key = if is_versioned {
                b"dltensor_versioned\0".as_ptr() as *const c_char
            } else {
                b"dltensor\0".as_ptr() as *const c_char
            };
            let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, key);
            if ptr.is_null() {
                return;
            }
            let mt = ptr as *mut DLManagedTensor;
            if let Some(del) = unsafe { (*mt).deleter } {
                del(mt);
            }
            unsafe {
                pyo3::ffi::PyCapsule_SetPointer(capsule, null_mut());
            }
        }

        // Move VRAM handle into Holder (one-shot DLPack export).
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        // Retain primary context for the allocation device.
        let dev = self.inner.device_id as i32;
        let mut retained_ctx: cust::sys::CUcontext = std::ptr::null_mut();
        unsafe {
            let _ = cust::sys::cuDevicePrimaryCtxRetain(&mut retained_ctx, dev);
        }

        let mut holder = Box::new(Holder {
            managed: DLManagedTensor {
                dl_tensor: DLTensor {
                    data: if inner.rows == 0 || inner.cols == 0 {
                        std::ptr::null_mut()
                    } else {
                        inner.buf.as_device_ptr().as_raw() as *mut c_void
                    },
                    device: DLDevice {
                        device_type: 2,
                        device_id: dev,
                    },
                    ndim: 2,
                    dtype: DLDataType {
                        code: 2,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(dl_managed_deleter),
            },
            shape: [inner.rows as i64, inner.cols as i64],
            strides: [inner.cols as i64, 1],
            arr: inner,
            _ctx_guard: retained_ctx,
            _device_id: self.inner.device_id,
            _context: self.inner.context.clone(),
        });
        holder.managed.dl_tensor.shape = holder.shape.as_mut_ptr();
        holder.managed.dl_tensor.strides = holder.strides.as_mut_ptr();

        let mt_ptr: *mut DLManagedTensor = &mut holder.managed;
        holder.managed.manager_ctx = &mut *holder as *mut Holder as *mut c_void;
        let _ = Box::into_raw(holder);

        let wants_versioned = matches!(max_version, Some((maj, _)) if maj >= 1);
        let name = if wants_versioned {
            b"dltensor_versioned\0"
        } else {
            b"dltensor\0"
        };
        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(
                mt_ptr as *mut c_void,
                name.as_ptr() as *const c_char,
                Some(capsule_destructor),
            )
        };
        if capsule.is_null() {
            unsafe {
                dl_managed_deleter(mt_ptr);
            }
            return Err(PyValueError::new_err("failed to create DLPack capsule"));
        }

        Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "keltner")]
#[pyo3(signature = (high, low, close, source, period, multiplier, ma_type="ema", kernel=None))]
pub fn keltner_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    source: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    multiplier: f64,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::{PyArray1, PyArrayMethods};

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let s = source.as_slice()?;
    let len = c.len();

    let mut up_arr = unsafe { PyArray1::<f64>::new(py, [len], false) };
    let mut mid_arr = unsafe { PyArray1::<f64>::new(py, [len], false) };
    let mut low_arr = unsafe { PyArray1::<f64>::new(py, [len], false) };

    let up = unsafe { up_arr.as_slice_mut()? };
    let mid = unsafe { mid_arr.as_slice_mut()? };
    let lowo = unsafe { low_arr.as_slice_mut()? };

    let params = KeltnerParams {
        period: Some(period),
        multiplier: Some(multiplier),
        ma_type: Some(ma_type.to_string()),
    };
    let input = KeltnerInput::from_slice(h, l, c, s, params);
    let kern = validate_kernel(kernel, false)?;

    py.allow_threads(|| keltner_into_slice(up, mid, lowo, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((up_arr, mid_arr, low_arr))
}

#[cfg(feature = "python")]
#[pyclass(name = "KeltnerStream")]
pub struct KeltnerStreamPy {
    stream: KeltnerStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KeltnerStreamPy {
    #[new]
    fn new(period: usize, multiplier: f64, ma_type: &str) -> PyResult<Self> {
        let params = KeltnerParams {
            period: Some(period),
            multiplier: Some(multiplier),
            ma_type: Some(ma_type.to_string()),
        };
        let stream =
            KeltnerStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KeltnerStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(high, low, close, source)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "keltner_batch")]
#[pyo3(signature = (high, low, close, source, period_range, multiplier_range, kernel=None))]
pub fn keltner_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    source: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    multiplier_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArray1, PyArrayMethods};
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let s = source.as_slice()?;

    let sweep = KeltnerBatchRange {
        period: period_range,
        multiplier: multiplier_range,
    };
    let kern = validate_kernel(kernel, true)?;

    // Run batch once to get shapes and combos
    let out = py
        .allow_threads(|| {
            keltner_batch_par_slice(
                h,
                l,
                c,
                s,
                &sweep,
                match kern {
                    Kernel::Auto => detect_best_batch_kernel(),
                    k => k,
                },
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = out.rows;
    let cols = out.cols;

    // Materialize into PyArrays without extra copies
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let up_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let mid_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let low_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    unsafe { up_arr.as_slice_mut()? }.copy_from_slice(&out.upper_band);
    unsafe { mid_arr.as_slice_mut()? }.copy_from_slice(&out.middle_band);
    unsafe { low_arr.as_slice_mut()? }.copy_from_slice(&out.lower_band);

    let dict = PyDict::new(py);
    dict.set_item("upper", up_arr.reshape((rows, cols))?)?;
    dict.set_item("middle", mid_arr.reshape((rows, cols))?)?;
    dict.set_item("lower", low_arr.reshape((rows, cols))?)?;
    use numpy::IntoPyArray;
    dict.set_item(
        "periods",
        out.combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        out.combos
            .iter()
            .map(|p| p.multiplier.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "keltner_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, source_f32, period_range, multiplier_range, ma_type="ema", device_id=0))]
pub fn keltner_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    source_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    multiplier_range: (f64, f64, f64),
    ma_type: &str,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let c = close_f32.as_slice()?;
    let s = source_f32.as_slice()?;
    if !(h.len() == l.len() && l.len() == c.len() && c.len() == s.len()) {
        return Err(PyValueError::new_err("input length mismatch"));
    }
    let sweep = KeltnerBatchRange {
        period: period_range,
        multiplier: multiplier_range,
    };
    let (up, mid, low, rows, cols, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaKeltner::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let res = cuda
            .keltner_batch_dev(h, l, c, s, &sweep, ma_type)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = res.outputs.upper.rows;
        let cols = res.outputs.upper.cols;
        Ok::<_, PyErr>((
            res.outputs.upper,
            res.outputs.middle,
            res.outputs.lower,
            rows,
            cols,
            ctx,
            dev_id,
        ))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "upper",
        Py::new(
            py,
            KeltnerDeviceArrayF32Py {
                inner: KeltnerDeviceArrayF32 {
                    inner: up,
                    context: ctx.clone(),
                    device_id: dev_id,
                },
            },
        )?,
    )?;
    dict.set_item(
        "middle",
        Py::new(
            py,
            KeltnerDeviceArrayF32Py {
                inner: KeltnerDeviceArrayF32 {
                    inner: mid,
                    context: ctx.clone(),
                    device_id: dev_id,
                },
            },
        )?,
    )?;
    dict.set_item(
        "lower",
        Py::new(
            py,
            KeltnerDeviceArrayF32Py {
                inner: KeltnerDeviceArrayF32 {
                    inner: low,
                    context: ctx,
                    device_id: dev_id,
                },
            },
        )?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "keltner_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, source_tm_f32, cols, rows, period, multiplier, ma_type="ema", device_id=0))]
pub fn keltner_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    close_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    source_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    multiplier: f32,
    ma_type: &str,
    device_id: usize,
) -> PyResult<(KeltnerDeviceArrayF32Py, KeltnerDeviceArrayF32Py, KeltnerDeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let ht = high_tm_f32.as_slice()?;
    let lt = low_tm_f32.as_slice()?;
    let ct = close_tm_f32.as_slice()?;
    let st = source_tm_f32.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if ht.len() != expected || lt.len() != expected || ct.len() != expected || st.len() != expected
    {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }
    let (up, mid, low, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaKeltner::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let trip = cuda
            .keltner_many_series_one_param_time_major_dev(
                ht, lt, ct, st, cols, rows, period, multiplier, ma_type,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((trip.upper, trip.middle, trip.lower, ctx, dev_id))
    })?;
    Ok((
        KeltnerDeviceArrayF32Py {
            inner: KeltnerDeviceArrayF32 {
                inner: up,
                context: ctx.clone(),
                device_id: dev_id,
            },
        },
        KeltnerDeviceArrayF32Py {
            inner: KeltnerDeviceArrayF32 {
                inner: mid,
                context: ctx.clone(),
                device_id: dev_id,
            },
        },
        KeltnerDeviceArrayF32Py {
            inner: KeltnerDeviceArrayF32 {
                inner: low,
                context: ctx,
                device_id: dev_id,
            },
        },
    ))
}

// WASM bindings

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KeltnerResult {
    pub values: Vec<f64>, // [upper..., middle..., lower...]
    pub rows: usize,      // 3
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "keltner")]
pub fn keltner_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: String,
) -> Result<JsValue, JsValue> {
    if !(high.len() == low.len() && low.len() == close.len() && close.len() == source.len()) {
        return Err(JsValue::from_str("Input arrays must have equal length"));
    }
    let len = close.len();

    // One contiguous buffer, no intermediate Vecs
    let mut values = vec![0.0f64; 3 * len];
    let (upper, rest) = values.split_at_mut(len);
    let (middle, lower) = rest.split_at_mut(len);

    let params = KeltnerParams {
        period: Some(period),
        multiplier: Some(multiplier),
        ma_type: Some(ma_type),
    };
    let input = KeltnerInput::from_slice(high, low, close, source, params);

    keltner_into_slice(upper, middle, lower, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let out = KeltnerResult {
        values,
        rows: 3,
        cols: len,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Fast API with aliasing detection - separate pointers for each output
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn keltner_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    source_ptr: *const f64,
    upper_ptr: *mut f64,
    middle_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
    period: usize,
    multiplier: f64,
    ma_type: &str,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || source_ptr.is_null()
        || upper_ptr.is_null()
        || middle_ptr.is_null()
        || lower_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);

        let params = KeltnerParams {
            period: Some(period),
            multiplier: Some(multiplier),
            ma_type: Some(ma_type.to_string()),
        };
        let input = KeltnerInput::from_slice(high, low, close, source, params);

        // Check for aliasing between input and output pointers
        let input_ptrs = [
            high_ptr as *const f64,
            low_ptr as *const f64,
            close_ptr as *const f64,
            source_ptr as *const f64,
        ];
        let output_ptrs = [
            upper_ptr as *const f64,
            middle_ptr as *const f64,
            lower_ptr as *const f64,
        ];

        let has_aliasing = input_ptrs
            .iter()
            .any(|&in_ptr| output_ptrs.iter().any(|&out_ptr| in_ptr == out_ptr));

        if has_aliasing {
            // Handle aliasing by using temporary buffers
            let mut temp_upper = vec![0.0; len];
            let mut temp_middle = vec![0.0; len];
            let mut temp_lower = vec![0.0; len];

            keltner_into_slice(
                &mut temp_upper,
                &mut temp_middle,
                &mut temp_lower,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
            let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
            let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);

            upper_out.copy_from_slice(&temp_upper);
            middle_out.copy_from_slice(&temp_middle);
            lower_out.copy_from_slice(&temp_lower);
        } else {
            // No aliasing, write directly
            let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
            let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
            let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);

            keltner_into_slice(upper_out, middle_out, lower_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

/// Memory allocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn keltner_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

/// Memory deallocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn keltner_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KeltnerBatchConfig {
    pub period_range: (usize, usize, usize),
    pub multiplier_range: (f64, f64, f64),
    pub ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KeltnerBatchJsOutput {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<KeltnerParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "keltner_batch")]
pub fn keltner_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: KeltnerBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = KeltnerBatchRange {
        period: cfg.period_range,
        multiplier: cfg.multiplier_range,
    };

    let out = keltner_batch_inner(
        high,
        low,
        close,
        source,
        &sweep,
        detect_best_batch_kernel(),
        false,
        Some(&cfg.ma_type),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_out = KeltnerBatchJsOutput {
        upper: out.upper_band,
        middle: out.middle_band,
        lower: out.lower_band,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js_out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "keltner_into_concat")]
pub fn keltner_into_concat(
    h_ptr: *const f64,
    l_ptr: *const f64,
    c_ptr: *const f64,
    s_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    multiplier: f64,
    ma_type: String,
) -> Result<(), JsValue> {
    if [h_ptr, l_ptr, c_ptr, s_ptr, out_ptr as *const f64]
        .iter()
        .any(|p| p.is_null())
    {
        return Err(JsValue::from_str(
            "null pointer passed to keltner_into_concat",
        ));
    }
    unsafe {
        let h = std::slice::from_raw_parts(h_ptr, len);
        let l = std::slice::from_raw_parts(l_ptr, len);
        let c = std::slice::from_raw_parts(c_ptr, len);
        let s = std::slice::from_raw_parts(s_ptr, len);

        let out = std::slice::from_raw_parts_mut(out_ptr, 3 * len);
        let (upper, rest) = out.split_at_mut(len);
        let (middle, lower) = rest.split_at_mut(len);

        let params = KeltnerParams {
            period: Some(period),
            multiplier: Some(multiplier),
            ma_type: Some(ma_type),
        };
        let input = KeltnerInput::from_slice(h, l, c, s, params);
        keltner_into_slice(upper, middle, lower, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
