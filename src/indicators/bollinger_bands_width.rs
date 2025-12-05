//! # Bollinger Bands Width (BBW)
//!
//! Decision note: Streaming path now uses an O(1) rolling-sum/sumsq kernel for stddev with SMA/EMA; SIMD remains disabled by default due to underperformance vs scalar.
//!
//! Bollinger Bands Width (sometimes called Bandwidth) shows the relative distance between
//! the upper and lower Bollinger Bands compared to the middle band.
//! It is typically calculated as: `(upper_band - lower_band) / middle_band`
//!
//! ## Parameters
//! - **period**: Underlying MA window (default: 20)
//! - **devup**: Upward multiplier (default: 2.0)
//! - **devdn**: Downward multiplier (default: 2.0)
//! - **matype**: MA type as string (default: "sma")
//! - **devtype**: 0 = stddev, 1 = mean_ad, 2 = median_ad (default: 0)
//!
//! ## Returns
//! - **`Ok(BollingerBandsWidthOutput)`**: Vec<f64> of same length as input
//! - **`Err(BollingerBandsWidthError)`** otherwise
//!
//! ## Developer Notes
//! - SIMD implemented for initial-window reduction only; loop-carried deps limit gains.
//! - Runtime selection short-circuits to Scalar for Auto: AVX2/AVX512 underperform for typical periods.
//! - Memory optimization: ✅ Uses alloc_with_nan_prefix (zero-copy)
//! - Batch operations: ✅ Implemented with parallel processing support

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
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::bollinger_bands_width_wrapper::CudaBbw;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};

impl<'a> AsRef<[f64]> for BollingerBandsWidthInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            BollingerBandsWidthData::Slice(s) => s,
            BollingerBandsWidthData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

/// O(1) streaming Bollinger Band Width for devtype=0 (stddev) with SMA/EMA middle.
/// Exact population-σ math, matching classic scalar kernels.
/// NaN/non-finite input resets state and returns None on that tick.
#[derive(Clone, Debug)]
pub struct BollingerBandsWidthStream {
    period: usize,
    u_plus_d: f64,
    kind: BBWMiddle,
    devtype: usize, // currently only 0 (stddev) supported in O(1) path
    // ring buffer
    buf: Box<[f64]>,
    head: usize,   // next position to overwrite (oldest sample)
    filled: usize, // number of valid samples in buf, capped at period
    // rolling sums
    sum: f64,
    sumsq: f64,
    // EMA state (only when kind == Ema)
    ema: f64,
    alpha: f64,
    beta: f64,
    ema_seeded: bool, // true after first full window
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BBWMiddle {
    Sma,
    Ema,
}

impl BollingerBandsWidthStream {
    #[inline(always)]
    pub fn new(period: usize, devup: f64, devdn: f64, matype: &str, devtype: usize) -> Self {
        assert!(period > 0);
        let kind = if matype.eq_ignore_ascii_case("ema") {
            BBWMiddle::Ema
        } else {
            BBWMiddle::Sma
        };
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            u_plus_d: devup + devdn,
            kind,
            devtype,
            buf: vec![0.0; period].into_boxed_slice(),
            head: 0,
            filled: 0,
            sum: 0.0,
            sumsq: 0.0,
            ema: 0.0,
            alpha,
            beta: 1.0 - alpha,
            ema_seeded: false,
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        self.sum = 0.0;
        self.sumsq = 0.0;
        self.ema = 0.0;
        self.ema_seeded = false;
        // Buffer contents can remain; `filled` gates validity.
    }

    /// Feed one sample. Returns None until warm-up completes (period samples).
    /// On devtype != 0, this returns None (O(1) path supports stddev only).
    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        if !x.is_finite() {
            self.reset();
            return None;
        }
        if self.devtype != 0 {
            return None;
        }

        if self.filled == self.period {
            // overwrite oldest value
            let old = self.buf[self.head];
            self.sum += x - old;
            self.sumsq += x * x - old * old;
            self.buf[self.head] = x;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
        } else {
            // still warming
            self.buf[self.head] = x;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            self.sum += x;
            self.sumsq += x * x;
            self.filled += 1;
            if self.filled < self.period {
                return None;
            }
        }

        debug_assert!(self.filled == self.period);
        let inv_n = 1.0 / (self.period as f64);
        let mu = self.sum * inv_n;
        let mut var_w = (self.sumsq * inv_n) - mu * mu;
        if var_w < 0.0 {
            var_w = 0.0;
        }

        let (mid, var_about_mid) = match self.kind {
            BBWMiddle::Sma => (mu, var_w),
            BBWMiddle::Ema => {
                if !self.ema_seeded {
                    // First full window: seed EMA from SMA and use window variance
                    self.ema = mu;
                    self.ema_seeded = true;
                    (self.ema, var_w)
                } else {
                    // After seeding, update EMA each tick and use parallel variance
                    self.ema = self.alpha * x + self.beta * self.ema;
                    let diff = mu - self.ema;
                    (self.ema, var_w + diff * diff)
                }
            }
        };

        let std = fast_sqrt64(var_about_mid);
        Some((self.u_plus_d * std) / mid)
    }
}

#[inline(always)]
fn fast_sqrt64(x: f64) -> f64 {
    // Exact path only; fast-math approximations are intentionally not used by default.
    if x <= 0.0 {
        0.0
    } else {
        x.sqrt()
    }
}

#[derive(Debug, Clone)]
pub enum BollingerBandsWidthData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct BollingerBandsWidthOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct BollingerBandsWidthParams {
    pub period: Option<usize>,
    pub devup: Option<f64>,
    pub devdn: Option<f64>,
    pub matype: Option<String>,
    pub devtype: Option<usize>,
}

impl Default for BollingerBandsWidthParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            devup: Some(2.0),
            devdn: Some(2.0),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BollingerBandsWidthInput<'a> {
    pub data: BollingerBandsWidthData<'a>,
    pub params: BollingerBandsWidthParams,
}

impl<'a> BollingerBandsWidthInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: BollingerBandsWidthParams) -> Self {
        Self {
            data: BollingerBandsWidthData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: BollingerBandsWidthParams) -> Self {
        Self {
            data: BollingerBandsWidthData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", BollingerBandsWidthParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_devup(&self) -> f64 {
        self.params.devup.unwrap_or(2.0)
    }
    #[inline]
    pub fn get_devdn(&self) -> f64 {
        self.params.devdn.unwrap_or(2.0)
    }
    #[inline]
    pub fn get_matype(&self) -> String {
        self.params
            .matype
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
    #[inline]
    pub fn get_devtype(&self) -> usize {
        self.params.devtype.unwrap_or(0)
    }
}

#[derive(Debug, Error)]
pub enum BollingerBandsWidthError {
    #[error("bbw: Empty data provided.")]
    EmptyInputData,
    #[error("bbw: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("bbw: All values are NaN.")]
    AllValuesNaN,
    #[error("bbw: Underlying MA or Deviation function failed: {0}")]
    UnderlyingFunctionFailed(String),
    #[error("bbw: MA calculation error: {0}")]
    MaError(String),
    #[error("bbw: Deviation calculation error: {0}")]
    DeviationError(String),
    #[error("bbw: Not enough valid data for period: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("bbw: output slice length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("bbw: invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange { start: i64, end: i64, step: i64 },
    #[error("bbw: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
pub struct BollingerBandsWidthBuilder {
    period: Option<usize>,
    devup: Option<f64>,
    devdn: Option<f64>,
    matype: Option<String>,
    devtype: Option<usize>,
    kernel: Kernel,
}

impl Default for BollingerBandsWidthBuilder {
    fn default() -> Self {
        Self {
            period: None,
            devup: None,
            devdn: None,
            matype: None,
            devtype: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BollingerBandsWidthBuilder {
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
    pub fn devup(mut self, x: f64) -> Self {
        self.devup = Some(x);
        self
    }
    #[inline(always)]
    pub fn devdn(mut self, x: f64) -> Self {
        self.devdn = Some(x);
        self
    }
    #[inline(always)]
    pub fn matype(mut self, x: &str) -> Self {
        self.matype = Some(x.to_string());
        self
    }
    #[inline(always)]
    pub fn devtype(mut self, x: usize) -> Self {
        self.devtype = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
        let p = BollingerBandsWidthParams {
            period: self.period,
            devup: self.devup,
            devdn: self.devdn,
            matype: self.matype,
            devtype: self.devtype,
        };
        let i = BollingerBandsWidthInput::from_candles(c, "close", p);
        bollinger_bands_width_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(
        self,
        d: &[f64],
    ) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
        let p = BollingerBandsWidthParams {
            period: self.period,
            devup: self.devup,
            devdn: self.devdn,
            matype: self.matype,
            devtype: self.devtype,
        };
        let i = BollingerBandsWidthInput::from_slice(d, p);
        bollinger_bands_width_with_kernel(&i, self.kernel)
    }
}

#[inline]
pub fn bollinger_bands_width(
    input: &BollingerBandsWidthInput,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    bollinger_bands_width_with_kernel(input, Kernel::Auto)
}

pub fn bollinger_bands_width_with_kernel(
    input: &BollingerBandsWidthInput,
    kernel: Kernel,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(BollingerBandsWidthError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(BollingerBandsWidthError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(BollingerBandsWidthError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(BollingerBandsWidthError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    // Calculate warmup period for zero-copy allocation
    let warmup_period = first_valid_idx + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_period);

    bollinger_bands_width_compute_into(data, input, &mut out, kernel)?;
    Ok(BollingerBandsWidthOutput { values: out })
}

/// Compute Bollinger Bands Width directly into a pre-allocated buffer.
/// This is the zero-copy variant used by Python/WASM bindings.
pub fn bollinger_bands_width_compute_into(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    out: &mut [f64],
    kernel: Kernel,
) -> Result<(), BollingerBandsWidthError> {
    if data.is_empty() {
        return Err(BollingerBandsWidthError::EmptyInputData);
    }
    if data.len() != out.len() {
        return Err(BollingerBandsWidthError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(BollingerBandsWidthError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(BollingerBandsWidthError::AllValuesNaN),
    };
    if (data.len() - first_valid_idx) < period {
        return Err(BollingerBandsWidthError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    // Note: The caller is responsible for pre-filling NaN values in the warmup period
    // using alloc_with_nan_prefix or similar methods

    // SIMD path underperforms here; prefer Scalar for Auto.
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                bollinger_bands_width_scalar_into(data, input, first_valid_idx, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                bollinger_bands_width_avx2_into(data, input, first_valid_idx, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                bollinger_bands_width_avx512_into(data, input, first_valid_idx, out)
            }
            _ => unreachable!(),
        }
    }
}

/// Compute Bollinger Bands Width into a caller-provided buffer (no allocations).
///
/// - Preserves NaN warmups exactly like the Vec-returning API by pre-filling the
///   warmup prefix with a quiet-NaN (same bit pattern as `alloc_with_nan_prefix`).
/// - `out.len()` must equal the input length; otherwise an error is returned.
#[cfg(not(feature = "wasm"))]
pub fn bollinger_bands_width_into(
    input: &BollingerBandsWidthInput,
    out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
    let data = input.as_ref();

    if data.is_empty() {
        return Err(BollingerBandsWidthError::EmptyInputData);
    }
    if out.len() != data.len() {
        return Err(BollingerBandsWidthError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(BollingerBandsWidthError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(BollingerBandsWidthError::AllValuesNaN),
    };
    if (data.len() - first_valid_idx) < period {
        return Err(BollingerBandsWidthError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    // Prefill warmup prefix with quiet-NaN to match alloc_with_nan_prefix pattern
    let warmup = first_valid_idx + period - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..warmup] {
        *v = qnan;
    }

    // Compute into provided buffer using the existing kernel dispatcher (Auto → Scalar here)
    bollinger_bands_width_compute_into(data, input, out, Kernel::Auto)
}

/// Write Bollinger Bands Width directly to output slice - no allocations.
/// This function follows the alma.rs pattern for WASM bindings.
#[inline]
pub fn bollinger_bands_width_into_slice(
    dst: &mut [f64],
    input: &BollingerBandsWidthInput,
    kern: Kernel,
) -> Result<(), BollingerBandsWidthError> {
    let data = input.as_ref();

    if data.is_empty() {
        return Err(BollingerBandsWidthError::EmptyInputData);
    }

    if dst.len() != data.len() {
        return Err(BollingerBandsWidthError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(BollingerBandsWidthError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(BollingerBandsWidthError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(BollingerBandsWidthError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    // Calculate warmup period
    let warmup_period = first_valid_idx + period - 1;

    // Fill warmup period with NaN
    for v in &mut dst[..warmup_period] {
        *v = f64::NAN;
    }

    // Compute the indicator values into the output slice
    // SIMD path underperforms here; prefer Scalar for Auto.
    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                bollinger_bands_width_scalar_into(data, input, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                bollinger_bands_width_avx2_into(data, input, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                bollinger_bands_width_avx512_into(data, input, first_valid_idx, dst)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn bollinger_bands_width_scalar(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    let period = input.get_period();
    let warmup_period = first_valid_idx + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
    bollinger_bands_width_scalar_into(data, input, first_valid_idx, &mut out)?;
    Ok(BollingerBandsWidthOutput { values: out })
}

#[inline]
pub unsafe fn bollinger_bands_width_scalar_into(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
    let period = input.get_period();
    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let matype = input.get_matype();
    let devtype = input.get_devtype();

    // Dispatch to classic kernels for common cases
    if devtype == 0 {
        // Population standard deviation
        if matype.eq_ignore_ascii_case("sma") {
            return bollinger_bands_width_scalar_classic_sma(
                data,
                period,
                devup,
                devdn,
                first_valid_idx,
                out,
            );
        } else if matype.eq_ignore_ascii_case("ema") {
            return bollinger_bands_width_scalar_classic_ema(
                data,
                period,
                devup,
                devdn,
                first_valid_idx,
                out,
            );
        }
    }

    // Fall back to general implementation
    let ma_data = match &input.data {
        BollingerBandsWidthData::Candles { candles, source } => {
            crate::indicators::moving_averages::ma::MaData::Candles { candles, source }
        }
        BollingerBandsWidthData::Slice(slice) => {
            crate::indicators::moving_averages::ma::MaData::Slice(slice)
        }
    };
    let middle = crate::indicators::moving_averages::ma::ma(&matype, ma_data, period)
        .map_err(|e| BollingerBandsWidthError::UnderlyingFunctionFailed(e.to_string()))?;
    let dev_input = crate::indicators::deviation::DevInput::from_slice(
        data,
        crate::indicators::deviation::DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );
    let dev = crate::indicators::deviation::deviation(&dev_input)
        .map_err(|e| BollingerBandsWidthError::UnderlyingFunctionFailed(e.to_string()))?;
    let dev_values = &dev.values; // <- consistent

    let start = first_valid_idx + period - 1;
    let u_plus_d = devup + devdn;
    let len = data.len();
    let mut i = start;
    while i < len {
        let m = *middle.get_unchecked(i);
        let d = *dev_values.get_unchecked(i);
        *out.get_unchecked_mut(i) = (u_plus_d * d) / m;
        i += 1;
    }
    Ok(())
}

/// Optimized Bollinger Bands Width calculation with inline SMA and standard deviation
#[inline]
pub unsafe fn bollinger_bands_width_scalar_classic_sma(
    data: &[f64],
    period: usize,
    devup: f64,
    devdn: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
    debug_assert!(period > 0 && first_valid_idx + period - 1 < data.len());
    let n = period;
    let inv_n = 1.0 / (n as f64);
    let start = first_valid_idx + n - 1;
    let u_plus_d = devup + devdn;

    // Initial window sums
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64; // sum of squares
    for i in 0..n {
        let x = *data.get_unchecked(first_valid_idx + i);
        sum += x;
        sq_sum = x.mul_add(x, sq_sum);
    }

    // First output
    let mut mean = sum * inv_n;
    let mut var = (sq_sum * inv_n) - mean * mean;
    if var < 0.0 {
        var = 0.0;
    }
    let mut std = var.sqrt();
    let mut mid = mean;
    *out.get_unchecked_mut(start) = (u_plus_d * std) / mid;

    // Rolling updates
    let len = data.len();
    let mut i = start + 1;
    while i < len {
        let new_v = *data.get_unchecked(i);
        let old_v = *data.get_unchecked(i - n);
        sum += new_v - old_v;
        sq_sum = new_v.mul_add(new_v, sq_sum - old_v * old_v);
        mean = sum * inv_n;
        var = (sq_sum * inv_n) - mean * mean;
        if var < 0.0 {
            var = 0.0;
        }
        std = var.sqrt();
        mid = mean;
        *out.get_unchecked_mut(i) = (u_plus_d * std) / mid;
        i += 1;
    }

    Ok(())
}

/// Optimized Bollinger Bands Width calculation with inline EMA and standard deviation
#[inline]
pub unsafe fn bollinger_bands_width_scalar_classic_ema(
    data: &[f64],
    period: usize,
    devup: f64,
    devdn: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
    debug_assert!(period > 0 && first_valid_idx + period - 1 < data.len());
    let n = period;
    let inv_n = 1.0 / (n as f64);
    let start = first_valid_idx + n - 1;
    let u_plus_d = devup + devdn;
    let alpha = 2.0 / (n as f64 + 1.0);
    let beta = 1.0 - alpha;

    // Initial sums over window
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64; // sum of squares
    for i in 0..n {
        let x = *data.get_unchecked(first_valid_idx + i);
        sum += x;
        sq_sum = x.mul_add(x, sq_sum);
    }
    // Seed EMA from initial SMA
    let mut ema = sum * inv_n;
    let mut mu_w = ema;
    let mut var_w = (sq_sum * inv_n) - mu_w * mu_w;
    if var_w < 0.0 {
        var_w = 0.0;
    }
    let mut var_about_ema = var_w;
    let mut std = var_about_ema.sqrt();
    *out.get_unchecked_mut(start) = (u_plus_d * std) / ema;

    // Rolling updates
    let len = data.len();
    let mut i = start + 1;
    while i < len {
        let new_v = *data.get_unchecked(i);
        let old_v = *data.get_unchecked(i - n);

        sum += new_v - old_v;
        sq_sum = new_v.mul_add(new_v, sq_sum - old_v * old_v);
        mu_w = sum * inv_n;

        ema = alpha * new_v + beta * ema;

        var_w = (sq_sum * inv_n) - mu_w * mu_w;
        if var_w < 0.0 {
            var_w = 0.0;
        }
        let diff = mu_w - ema;
        var_about_ema = var_w + diff * diff;

        std = if var_about_ema > 0.0 {
            var_about_ema.sqrt()
        } else {
            0.0
        };
        *out.get_unchecked_mut(i) = (u_plus_d * std) / ema;
        i += 1;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    bollinger_bands_width_scalar(data, input, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512_into(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
    use core::arch::x86_64::*;

    let period = input.get_period();
    let devtype = input.get_devtype();
    let matype = input.get_matype();

    if !(devtype == 0 && (matype.eq_ignore_ascii_case("sma") || matype.eq_ignore_ascii_case("ema")))
    {
        return bollinger_bands_width_scalar_into(data, input, first_valid_idx, out);
    }

    let n = period;
    let inv_n = 1.0 / (n as f64);
    let start = first_valid_idx + n - 1;
    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let u_plus_d = devup + devdn;

    // AVX-512 accumulate initial window
    let mut v_sum = _mm512_setzero_pd();
    let mut v_sumsq = _mm512_setzero_pd();
    let base = first_valid_idx;
    let mut idx = base;
    let end_simd = base + (n & !7);
    while idx < end_simd {
        let px = data.as_ptr().add(idx);
        let x = _mm512_loadu_pd(px);
        v_sum = _mm512_add_pd(v_sum, x);
        let x2 = _mm512_mul_pd(x, x);
        v_sumsq = _mm512_add_pd(v_sumsq, x2);
        idx += 8;
    }
    let mut buf = [0.0f64; 8];
    _mm512_storeu_pd(buf.as_mut_ptr(), v_sum);
    let mut sum = buf.iter().sum::<f64>();
    _mm512_storeu_pd(buf.as_mut_ptr(), v_sumsq);
    let mut sumsq = buf.iter().sum::<f64>();
    while idx < base + n {
        let x = *data.get_unchecked(idx);
        sum += x;
        sumsq = x.mul_add(x, sumsq);
        idx += 1;
    }

    // Seed middle/variance
    let mut mu_w = sum * inv_n;
    let mut ema = mu_w;
    let mut var_w = (sumsq * inv_n) - mu_w * mu_w;
    if var_w < 0.0 {
        var_w = 0.0;
    }
    let mut mid = if matype.eq_ignore_ascii_case("sma") {
        mu_w
    } else {
        ema
    };
    let mut var_about_mid = if matype.eq_ignore_ascii_case("sma") {
        var_w
    } else {
        let diff = mu_w - ema;
        var_w + diff * diff
    };
    let mut std = if var_about_mid > 0.0 {
        var_about_mid.sqrt()
    } else {
        0.0
    };
    *out.get_unchecked_mut(start) = (u_plus_d * std) / mid;

    // Streaming phase (scalar updates)
    let alpha = 2.0 / (n as f64 + 1.0);
    let beta = 1.0 - alpha;
    let len = data.len();
    let mut i = start + 1;
    while i < len {
        let new_v = *data.get_unchecked(i);
        let old_v = *data.get_unchecked(i - n);
        sum += new_v - old_v;
        sumsq = new_v.mul_add(new_v, sumsq - old_v * old_v);
        mu_w = sum * inv_n;
        if matype.eq_ignore_ascii_case("ema") {
            ema = alpha * new_v + beta * ema;
        } else {
            ema = mu_w;
        }
        var_w = (sumsq * inv_n) - mu_w * mu_w;
        if var_w < 0.0 {
            var_w = 0.0;
        }
        if matype.eq_ignore_ascii_case("sma") {
            mid = mu_w;
            var_about_mid = var_w;
        } else {
            mid = ema;
            let diff = mu_w - ema;
            var_about_mid = var_w + diff * diff;
        }
        std = if var_about_mid > 0.0 {
            var_about_mid.sqrt()
        } else {
            0.0
        };
        *out.get_unchecked_mut(i) = (u_plus_d * std) / mid;
        i += 1;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx2(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    bollinger_bands_width_scalar(data, input, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx2_into(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
    use core::arch::x86_64::*;

    let period = input.get_period();
    let devtype = input.get_devtype();
    let matype = input.get_matype();

    if !(devtype == 0 && (matype.eq_ignore_ascii_case("sma") || matype.eq_ignore_ascii_case("ema")))
    {
        return bollinger_bands_width_scalar_into(data, input, first_valid_idx, out);
    }

    let n = period;
    let inv_n = 1.0 / (n as f64);
    let start = first_valid_idx + n - 1;
    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let u_plus_d = devup + devdn;

    // AVX2 accumulate initial window
    let mut v_sum = _mm256_setzero_pd();
    let mut v_sumsq = _mm256_setzero_pd();
    let base = first_valid_idx;
    let mut idx = base;
    let end_simd = base + (n & !3);
    while idx < end_simd {
        let px = data.as_ptr().add(idx);
        let x = _mm256_loadu_pd(px);
        v_sum = _mm256_add_pd(v_sum, x);
        let x2 = _mm256_mul_pd(x, x);
        v_sumsq = _mm256_add_pd(v_sumsq, x2);
        idx += 4;
    }
    let mut buf = [0.0f64; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), v_sum);
    let mut sum = buf[0] + buf[1] + buf[2] + buf[3];
    _mm256_storeu_pd(buf.as_mut_ptr(), v_sumsq);
    let mut sumsq = buf[0] + buf[1] + buf[2] + buf[3];
    while idx < base + n {
        let x = *data.get_unchecked(idx);
        sum += x;
        sumsq = x.mul_add(x, sumsq);
        idx += 1;
    }

    // Seed middle/variance
    let mut mu_w = sum * inv_n;
    let mut ema = mu_w;
    let mut var_w = (sumsq * inv_n) - mu_w * mu_w;
    if var_w < 0.0 {
        var_w = 0.0;
    }
    let mut mid = if matype.eq_ignore_ascii_case("sma") {
        mu_w
    } else {
        ema
    };
    let mut var_about_mid = if matype.eq_ignore_ascii_case("sma") {
        var_w
    } else {
        let diff = mu_w - ema;
        var_w + diff * diff
    };
    let mut std = if var_about_mid > 0.0 {
        var_about_mid.sqrt()
    } else {
        0.0
    };
    *out.get_unchecked_mut(start) = (u_plus_d * std) / mid;

    // Streaming phase (scalar updates)
    let alpha = 2.0 / (n as f64 + 1.0);
    let beta = 1.0 - alpha;
    let len = data.len();
    let mut i = start + 1;
    while i < len {
        let new_v = *data.get_unchecked(i);
        let old_v = *data.get_unchecked(i - n);
        sum += new_v - old_v;
        sumsq = new_v.mul_add(new_v, sumsq - old_v * old_v);
        mu_w = sum * inv_n;
        if matype.eq_ignore_ascii_case("ema") {
            ema = alpha * new_v + beta * ema;
        } else {
            ema = mu_w;
        }
        var_w = (sumsq * inv_n) - mu_w * mu_w;
        if var_w < 0.0 {
            var_w = 0.0;
        }
        if matype.eq_ignore_ascii_case("sma") {
            mid = mu_w;
            var_about_mid = var_w;
        } else {
            mid = ema;
            let diff = mu_w - ema;
            var_about_mid = var_w + diff * diff;
        }
        std = if var_about_mid > 0.0 {
            var_about_mid.sqrt()
        } else {
            0.0
        };
        *out.get_unchecked_mut(i) = (u_plus_d * std) / mid;
        i += 1;
    }

    Ok(())
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512_short(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    bollinger_bands_width_avx512(data, input, first_valid_idx)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512_long(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    bollinger_bands_width_avx512(data, input, first_valid_idx)
}

#[doc(hidden)]
#[inline(always)]
pub fn bollinger_bands_width_row_scalar(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    let period = input.get_period();
    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let matype = input.get_matype();
    let devtype = input.get_devtype();
    let ma_data = match &input.data {
        BollingerBandsWidthData::Candles { candles, source } => {
            crate::indicators::moving_averages::ma::MaData::Candles { candles, source }
        }
        BollingerBandsWidthData::Slice(slice) => {
            crate::indicators::moving_averages::ma::MaData::Slice(slice)
        }
    };
    let middle = crate::indicators::moving_averages::ma::ma(&matype, ma_data, period).unwrap();
    let dev_input = crate::indicators::deviation::DevInput::from_slice(
        data,
        crate::indicators::deviation::DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );
    let dev = crate::indicators::deviation::deviation(&dev_input).unwrap();
    let dev_values = &dev.values; // <- consistent
    for i in (first_valid_idx + period - 1)..data.len() {
        let m = middle[i];
        let u = m + devup * dev_values[i];
        let l = m - devdn * dev_values[i];
        out[i] = (u - l) / m;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[doc(hidden)]
#[inline(always)]
pub fn bollinger_bands_width_row_avx2(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    bollinger_bands_width_row_scalar(data, input, first_valid_idx, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[doc(hidden)]
#[inline(always)]
pub fn bollinger_bands_width_row_avx512(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    bollinger_bands_width_row_scalar(data, input, first_valid_idx, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[doc(hidden)]
#[inline(always)]
pub fn bollinger_bands_width_row_avx512_short(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    bollinger_bands_width_row_avx512(data, input, first_valid_idx, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[doc(hidden)]
#[inline(always)]
pub fn bollinger_bands_width_row_avx512_long(
    data: &[f64],
    input: &BollingerBandsWidthInput,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    bollinger_bands_width_row_avx512(data, input, first_valid_idx, out)
}

#[derive(Clone, Debug)]
pub struct BollingerBandsWidthBatchRange {
    pub period: (usize, usize, usize),
    pub devup: (f64, f64, f64),
    pub devdn: (f64, f64, f64),
}

impl Default for BollingerBandsWidthBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 60, 1),
            devup: (2.0, 2.0, 0.0),
            devdn: (2.0, 2.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BollingerBandsWidthBatchBuilder {
    range: BollingerBandsWidthBatchRange,
    kernel: Kernel,
}

impl BollingerBandsWidthBatchBuilder {
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
    pub fn devup_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.devup = (start, end, step);
        self
    }
    #[inline]
    pub fn devdn_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.devdn = (start, end, step);
        self
    }
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
        bollinger_bands_width_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
        BollingerBandsWidthBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct BollingerBandsWidthBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<BollingerBandsWidthParams>,
    pub rows: usize,
    pub cols: usize,
}
impl BollingerBandsWidthBatchOutput {
    pub fn row_for_params(&self, p: &BollingerBandsWidthParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.devup.unwrap_or(2.0) - p.devup.unwrap_or(2.0)).abs() < 1e-12
                && (c.devdn.unwrap_or(2.0) - p.devdn.unwrap_or(2.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &BollingerBandsWidthParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_checked(
    r: &BollingerBandsWidthBatchRange,
) -> Result<Vec<BollingerBandsWidthParams>, BollingerBandsWidthError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, BollingerBandsWidthError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(step) { Some(n) => x = n, None => break }
            }
        } else {
            // reversed bounds supported
            let mut x = start as i64;
            let step_i = step as i64;
            while x >= end as i64 {
                v.push(x as usize);
                x -= step_i;
            }
        }
        if v.is_empty() {
            return Err(BollingerBandsWidthError::InvalidRange {
                start: start as i64,
                end: end as i64,
                step: step as i64,
            });
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, BollingerBandsWidthError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if step > 0.0 {
            if start <= end {
                let mut x = start;
                while x <= end + 1e-12 {
                    v.push(x);
                    x += step;
                }
            }
        } else {
            if start >= end {
                let mut x = start;
                while x >= end - 1e-12 {
                    v.push(x);
                    x += step; // step < 0
                }
            }
        }
        if v.is_empty() {
            return Err(BollingerBandsWidthError::InvalidRange {
                start: start as i64,
                end: end as i64,
                step: step as i64,
            });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let devups = axis_f64(r.devup)?;
    let devdns = axis_f64(r.devdn)?;

    let cap = periods
        .len()
        .checked_mul(devups.len())
        .and_then(|v| v.checked_mul(devdns.len()))
        .ok_or(BollingerBandsWidthError::InvalidRange {
            start: periods.len() as i64,
            end: devups.len() as i64,
            step: devdns.len() as i64,
        })?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &u in &devups {
            for &d in &devdns {
                out.push(BollingerBandsWidthParams {
                    period: Some(p),
                    devup: Some(u),
                    devdn: Some(d),
                    matype: Some("sma".to_string()),
                    devtype: Some(0),
                });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn bollinger_bands_width_batch_with_kernel(
    data: &[f64],
    sweep: &BollingerBandsWidthBatchRange,
    k: Kernel,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(BollingerBandsWidthError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    bollinger_bands_width_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn bollinger_bands_width_batch_slice(
    data: &[f64],
    sweep: &BollingerBandsWidthBatchRange,
    kern: Kernel,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
    bollinger_bands_width_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn bollinger_bands_width_batch_par_slice(
    data: &[f64],
    sweep: &BollingerBandsWidthBatchRange,
    kern: Kernel,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
    bollinger_bands_width_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn bollinger_bands_width_batch_inner(
    data: &[f64],
    sweep: &BollingerBandsWidthBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    // Check for empty data and return AllValuesNaN for consistency
    if cols == 0 {
        return Err(BollingerBandsWidthError::AllValuesNaN);
    }

    // Step 1: Allocate uninitialized matrix (checked rows*cols)
    let _ = rows
        .checked_mul(cols)
        .ok_or(BollingerBandsWidthError::InvalidRange { start: rows as i64, end: cols as i64, step: 0 })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Step 2: Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| data.iter().position(|x| !x.is_nan()).unwrap_or(0) + c.period.unwrap() - 1)
        .collect();

    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Step 4: Convert to mutable slice for computation
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let values_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Step 5: Compute into the buffer
    bollinger_bands_width_batch_inner_into(data, sweep, kern, parallel, values_slice)?;

    // Step 6: Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(BollingerBandsWidthBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

/// Compute batch Bollinger Bands Width directly into a pre-allocated buffer.
/// This is the zero-copy variant used by Python bindings for batch operations.
#[inline(always)]
pub fn bollinger_bands_width_batch_inner_into(
    data: &[f64],
    sweep: &BollingerBandsWidthBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<BollingerBandsWidthParams>, BollingerBandsWidthError> {
    let combos = expand_grid_checked(sweep)?;
    if combos.is_empty() {
        return Err(BollingerBandsWidthError::InvalidRange { start: 0, end: 0, step: 0 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BollingerBandsWidthError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(BollingerBandsWidthError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let cols = data.len();
    let rows = combos.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(BollingerBandsWidthError::InvalidRange { start: rows as i64, end: cols as i64, step: 0 })?;
    if out.len() != expected {
        return Err(BollingerBandsWidthError::OutputLengthMismatch { expected, got: out.len() });
    }

    // Group combinations by (period, matype, devtype) to avoid redundant calculations
    use std::collections::HashMap;
    let mut groups: HashMap<(usize, String, usize), Vec<(usize, f64, f64)>> = HashMap::new();

    for (idx, combo) in combos.iter().enumerate() {
        let key = (
            combo.period.unwrap(),
            combo.matype.as_ref().unwrap_or(&"sma".to_string()).clone(),
            combo.devtype.unwrap_or(0),
        );
        groups.entry(key).or_insert_with(Vec::new).push((
            idx,
            combo.devup.unwrap(),
            combo.devdn.unwrap(),
        ));
    }

    // Process each unique (period, matype, devtype) group
    for ((period, matype, devtype), indices) in groups {
        // Compute MA and deviation once for this group
        let ma_data = crate::indicators::moving_averages::ma::MaData::Slice(data);
        let middle = crate::indicators::moving_averages::ma::ma(&matype, ma_data, period)
            .map_err(|e| BollingerBandsWidthError::MaError(e.to_string()))?;

        let dev_input = crate::indicators::deviation::DevInput::from_slice(
            data,
            crate::indicators::deviation::DevParams {
                period: Some(period),
                devtype: Some(devtype),
            },
        );
        let dev_values = crate::indicators::deviation::deviation(&dev_input)
            .map_err(|e| BollingerBandsWidthError::DeviationError(e.to_string()))?;

        // Precompute dev/middle ratio for this group once
        let time_start = first + period - 1;
        let mut ratio: Vec<f64> = vec![f64::NAN; cols];
        for i in time_start..cols {
            let m = middle[i];
            let d = dev_values.values[i];
            ratio[i] = d / m;
        }

        // Now compute BBW for each (devup, devdn) combination in this group
        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;

                // Precompute a fast row -> (devup, devdn) map for this group
                let mut row_params: Vec<Option<(f64, f64)>> = vec![None; combos.len()];
                for &(idx, u, d) in &indices {
                    row_params[idx] = Some((u, d));
                }

                out.par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(row_idx, out_row)| {
                        if let Some((u, d)) = row_params[row_idx] {
                            let u_plus_d = u + d;
                            // Vectorized scale where available
                            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                            {
                                match kern {
                                    Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
                                        use core::arch::x86_64::*;
                                        let k = _mm512_set1_pd(u_plus_d);
                                        let mut i = time_start;
                                        while i + 8 <= cols {
                                            let vr = _mm512_loadu_pd(ratio.as_ptr().add(i));
                                            let vout = _mm512_mul_pd(k, vr);
                                            _mm512_storeu_pd(out_row.as_mut_ptr().add(i), vout);
                                            i += 8;
                                        }
                                        while i < cols {
                                            *out_row.get_unchecked_mut(i) =
                                                u_plus_d * *ratio.get_unchecked(i);
                                            i += 1;
                                        }
                                    },
                                    Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
                                        use core::arch::x86_64::*;
                                        let k = _mm256_set1_pd(u_plus_d);
                                        let mut i = time_start;
                                        while i + 4 <= cols {
                                            let vr = _mm256_loadu_pd(ratio.as_ptr().add(i));
                                            let vout = _mm256_mul_pd(k, vr);
                                            _mm256_storeu_pd(out_row.as_mut_ptr().add(i), vout);
                                            i += 4;
                                        }
                                        while i < cols {
                                            *out_row.get_unchecked_mut(i) =
                                                u_plus_d * *ratio.get_unchecked(i);
                                            i += 1;
                                        }
                                    },
                                    _ => {
                                        for i in time_start..cols {
                                            out_row[i] = u_plus_d * ratio[i];
                                        }
                                    }
                                }
                            }
                            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                            {
                                for i in time_start..cols {
                                    out_row[i] = u_plus_d * ratio[i];
                                }
                            }
                        }
                    });
            }

            #[cfg(target_arch = "wasm32")]
            {
                for &(idx, devup, devdn) in &indices {
                    let row_off = idx * cols;
                    let end = row_off + cols;
                    let out_row = &mut out[row_off..end];
                    let u_plus_d = devup + devdn;
                    for i in time_start..cols {
                        out_row[i] = u_plus_d * ratio[i];
                    }
                }
            }
        } else {
            for &(idx, devup, devdn) in &indices {
                let row_off = idx * cols;
                let end = row_off + cols;
                let out_row = &mut out[row_off..end];
                let u_plus_d = devup + devdn;
                // Vectorized scale where available
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                {
                    match kern {
                        Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
                            use core::arch::x86_64::*;
                            let k = _mm512_set1_pd(u_plus_d);
                            let mut i = time_start;
                            while i + 8 <= cols {
                                let vr = _mm512_loadu_pd(ratio.as_ptr().add(i));
                                let vout = _mm512_mul_pd(k, vr);
                                _mm512_storeu_pd(out_row.as_mut_ptr().add(i), vout);
                                i += 8;
                            }
                            while i < cols {
                                *out_row.get_unchecked_mut(i) = u_plus_d * *ratio.get_unchecked(i);
                                i += 1;
                            }
                        },
                        Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
                            use core::arch::x86_64::*;
                            let k = _mm256_set1_pd(u_plus_d);
                            let mut i = time_start;
                            while i + 4 <= cols {
                                let vr = _mm256_loadu_pd(ratio.as_ptr().add(i));
                                let vout = _mm256_mul_pd(k, vr);
                                _mm256_storeu_pd(out_row.as_mut_ptr().add(i), vout);
                                i += 4;
                            }
                            while i < cols {
                                *out_row.get_unchecked_mut(i) = u_plus_d * *ratio.get_unchecked(i);
                                i += 1;
                            }
                        },
                        _ => {
                            for i in time_start..cols {
                                out_row[i] = u_plus_d * ratio[i];
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                {
                    for i in time_start..cols {
                        out_row[i] = u_plus_d * ratio[i];
                    }
                }
            }
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;
    use paste::paste;

    #[test]
    fn test_bollinger_bands_width_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Small but non-trivial input (positive values to avoid divide-by-zero in middle band)
        let data: Vec<f64> = (0..256)
            .map(|i| ((i as f64).sin() + 2.0) * (1.0 + (i as f64).cos() * 0.05))
            .collect();

        let input = BollingerBandsWidthInput::from_slice(&data, BollingerBandsWidthParams::default());

        // Baseline via existing API
        let baseline = bollinger_bands_width_with_kernel(&input, Kernel::Auto)?.values;

        // Preallocate output and call the new into API
        let mut out = vec![0.0; data.len()];
        #[cfg(not(feature = "wasm"))]
        {
            bollinger_bands_width_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, fall back to the existing into_slice wrapper with Auto kernel
            bollinger_bands_width_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), out.len());

        // Helper: treat NaN == NaN as equal; otherwise require exact equality
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: baseline={:?}, into={:?}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }

    fn check_bbw_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = BollingerBandsWidthParams {
            period: Some(22),
            devup: Some(2.2),
            devdn: None,
            matype: Some("ema".to_string()),
            devtype: None,
        };
        let input = BollingerBandsWidthInput::from_candles(&candles, "hl2", partial_params);
        let output = bollinger_bands_width_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_bbw_default(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BollingerBandsWidthInput::with_default_candles(&candles);
        let output = bollinger_bands_width_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_bbw_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsWidthParams {
            period: Some(0),
            ..Default::default()
        };
        let input = BollingerBandsWidthInput::from_slice(&data, params);
        let result = bollinger_bands_width_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for zero period",
            test_name
        );
        Ok(())
    }

    fn check_bbw_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsWidthParams {
            period: Some(10),
            ..Default::default()
        };
        let input = BollingerBandsWidthInput::from_slice(&data, params);
        let result = bollinger_bands_width_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for period > data.len()",
            test_name
        );
        Ok(())
    }

    fn check_bbw_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let input =
            BollingerBandsWidthInput::from_slice(&data, BollingerBandsWidthParams::default());
        let result = bollinger_bands_width_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for small data",
            test_name
        );
        Ok(())
    }

    fn check_bbw_nan_check(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BollingerBandsWidthInput::with_default_candles(&candles);
        let result = bollinger_bands_width_with_kernel(&input, kernel)?;
        let check_index = 240;
        if result.values.len() > check_index {
            for i in check_index..result.values.len() {
                // at least some values after check_index should not be NaN
                if !result.values[i].is_nan() {
                    return Ok(());
                }
            }
            panic!(
                "All BBWidth values from index {} onward are NaN.",
                check_index
            );
        }
        Ok(())
    }

    // Batch grid test: only parity, doesn't check numerical values
    fn check_batch_default_row(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = BollingerBandsWidthBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = BollingerBandsWidthParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_bollinger_bands_width_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to increase coverage
        let test_params = vec![
            // Default parameters
            BollingerBandsWidthParams::default(),
            // Small period
            BollingerBandsWidthParams {
                period: Some(5),
                devup: Some(1.0),
                devdn: Some(1.0),
                matype: Some("sma".to_string()),
                devtype: Some(0),
            },
            // Large period
            BollingerBandsWidthParams {
                period: Some(50),
                devup: Some(3.0),
                devdn: Some(3.0),
                matype: Some("ema".to_string()),
                devtype: Some(1),
            },
            // Asymmetric deviations
            BollingerBandsWidthParams {
                period: Some(15),
                devup: Some(2.5),
                devdn: Some(1.5),
                matype: Some("wma".to_string()),
                devtype: Some(2),
            },
            // Edge case parameters
            BollingerBandsWidthParams {
                period: Some(2),
                devup: Some(0.5),
                devdn: Some(0.5),
                matype: Some("sma".to_string()),
                devtype: Some(0),
            },
        ];

        // Test with different sources too
        let sources = vec!["close", "hl2", "hlc3", "ohlc4"];

        for params in test_params {
            for &source in &sources {
                let input =
                    BollingerBandsWidthInput::from_candles(&candles, source, params.clone());
                let output = bollinger_bands_width_with_kernel(&input, kernel)?;

                // Check every value for poison patterns
                for (i, &val) in output.values.iter().enumerate() {
                    // Skip NaN values as they're expected in the warmup period
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params: period={}, devup={}, devdn={}, source={}",
                            test_name, val, bits, i,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params: period={}, devup={}, devdn={}, source={}",
                            test_name, val, bits, i,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params: period={}, devup={}, devdn={}, source={}",
                            test_name, val, bits, i,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_bollinger_bands_width_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations with diverse parameter ranges
        let batch_configs = vec![
            // Wide range of periods with standard deviations
            (2, 50, 5, 1.0, 3.0, 0.5, 1.0, 3.0, 0.5),
            // Small periods with varying deviations
            (5, 15, 2, 0.5, 3.5, 0.25, 0.5, 3.5, 0.25),
            // Large periods with small deviation range
            (40, 100, 10, 1.5, 2.5, 0.1, 1.5, 2.5, 0.1),
            // Asymmetric deviations
            (10, 30, 5, 1.0, 4.0, 1.0, 0.5, 2.0, 0.5),
            // Edge case: very small periods
            (2, 5, 1, 0.1, 5.0, 0.5, 0.1, 5.0, 0.5),
        ];

        let sources = vec!["close", "hl2", "ohlc4"];

        for (
            period_start,
            period_end,
            period_step,
            devup_start,
            devup_end,
            devup_step,
            devdn_start,
            devdn_end,
            devdn_step,
        ) in batch_configs
        {
            for &source in &sources {
                let output = BollingerBandsWidthBatchBuilder::new()
                    .kernel(kernel)
                    .period_range(period_start, period_end, period_step)
                    .devup_range(devup_start, devup_end, devup_step)
                    .devdn_range(devdn_start, devdn_end, devdn_step)
                    .apply_candles(&c, source)?;

                // Check every value in the entire batch matrix for poison patterns
                for (idx, &val) in output.values.iter().enumerate() {
                    // Skip NaN values as they're expected in warmup periods
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();
                    let row = idx / output.cols;
                    let col = idx % output.cols;

                    // Get the parameters for this row
                    let params = &output.combos[row];

                    // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, devup={}, devdn={}, source={}",
                            test, val, bits, row, col, idx,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, devup={}, devdn={}, source={}",
                            test, val, bits, row, col, idx,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, devup={}, devdn={}, source={}",
                            test, val, bits, row, col, idx,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_bollinger_bands_width_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        use proptest::prelude::*;

        // Strategy 1: Random price data with realistic period ranges
        let random_data_strat =
            (10usize..=30, 1.0f64..=3.0, 1.0f64..=3.0).prop_flat_map(|(period, devup, devdn)| {
                let len = period * 3..400;
                (
                    prop::collection::vec(
                        (10f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    Just(period),
                    Just(devup),
                    Just(devdn),
                    Just("random"),
                )
            });

        // Strategy 2: Constant data (BBW should be near 0)
        let constant_data_strat =
            (10usize..=25, 1.0f64..=2.5, 1.0f64..=2.5).prop_flat_map(|(period, devup, devdn)| {
                let len = period * 2..200;
                (
                    prop::collection::vec(Just(100.0f64), len),
                    Just(period),
                    Just(devup),
                    Just(devdn),
                    Just("constant"),
                )
            });

        // Strategy 3: High volatility data (alternating high/low)
        let volatile_data_strat =
            (10usize..=25, 1.5f64..=2.5, 1.5f64..=2.5).prop_flat_map(|(period, devup, devdn)| {
                let len = period * 3..300;
                (
                    prop::collection::vec(0f64..1000f64, len).prop_map(|v| {
                        // Create alternating high/low pattern for high volatility
                        v.into_iter()
                            .enumerate()
                            .map(|(i, val)| if i % 2 == 0 { val + 500.0 } else { val })
                            .collect()
                    }),
                    Just(period),
                    Just(devup),
                    Just(devdn),
                    Just("volatile"),
                )
            });

        // Strategy 4: Edge cases with small periods
        let edge_case_strat =
            (2usize..=5, 0.5f64..=4.0, 0.5f64..=4.0).prop_flat_map(|(period, devup, devdn)| {
                let len = period * 4..100;
                (
                    prop::collection::vec(
                        (50f64..500f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    Just(period),
                    Just(devup),
                    Just(devdn),
                    Just("edge"),
                )
            });

        // Combine all strategies
        let combined_strat = prop_oneof![
            random_data_strat,
            constant_data_strat,
            volatile_data_strat,
            edge_case_strat,
        ];

        proptest::test_runner::TestRunner::default()
            .run(
                &combined_strat,
                |(data, period, devup, devdn, data_type)| {
                    let params = BollingerBandsWidthParams {
                        period: Some(period),
                        devup: Some(devup),
                        devdn: Some(devdn),
                        matype: Some("sma".to_string()),
                        devtype: Some(0), // standard deviation
                    };
                    let input = BollingerBandsWidthInput::from_slice(&data, params.clone());

                    // Test with the specified kernel
                    let result = bollinger_bands_width_with_kernel(&input, kernel);
                    prop_assert!(result.is_ok(), "BBW calculation failed: {:?}", result);
                    let out = result.unwrap().values;

                    // Test with scalar reference kernel for comparison
                    let ref_input = BollingerBandsWidthInput::from_slice(&data, params);
                    let ref_result = bollinger_bands_width_with_kernel(&ref_input, Kernel::Scalar);
                    prop_assert!(ref_result.is_ok(), "Reference BBW calculation failed");
                    let ref_out = ref_result.unwrap().values;

                    // Property 1: Output length should match input
                    prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                    // Property 2: Warmup period - first (period-1) values should be NaN
                    for i in 0..(period - 1) {
                        prop_assert!(
                            out[i].is_nan(),
                            "Expected NaN during warmup at index {}, got {}",
                            i,
                            out[i]
                        );
                    }

                    // Property 3: Non-NaN values should be non-negative (BBW is a width)
                    for (i, &val) in out.iter().enumerate() {
                        if !val.is_nan() {
                            prop_assert!(
                                val >= 0.0,
                                "BBW must be non-negative at index {}: got {}",
                                i,
                                val
                            );
                        }
                    }

                    // Property 4: For constant data, BBW should be near 0
                    if data_type == "constant" {
                        for (i, &val) in out.iter().enumerate().skip(period - 1) {
                            prop_assert!(
                                val.abs() < 1e-6,
                                "BBW for constant data should be near 0 at index {}: got {}",
                                i,
                                val
                            );
                        }
                    }

                    // Property 5: Kernel consistency - compare with scalar reference
                    for i in 0..out.len() {
                        let y = out[i];
                        let r = ref_out[i];

                        // Both should be NaN or both should be finite
                        if y.is_nan() && r.is_nan() {
                            continue;
                        }

                        prop_assert!(
                            y.is_finite() == r.is_finite(),
                            "Finite/NaN mismatch at index {}: kernel={}, scalar={}",
                            i,
                            y,
                            r
                        );

                        if y.is_finite() && r.is_finite() {
                            // Check ULP difference for floating point accuracy
                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();
                            let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                                "Kernel mismatch at index {}: {} vs {} (ULP={}, diff={})",
                                i,
                                y,
                                r,
                                ulp_diff,
                                (y - r).abs()
                            );
                        }
                    }

                    // Property 6: BBW should respond to volatility appropriately
                    if data_type == "volatile" && out.len() > period * 2 {
                        // For highly volatile data, BBW should be consistently higher
                        let valid_values: Vec<f64> = out
                            .iter()
                            .skip(period - 1)
                            .copied()
                            .filter(|&v| v.is_finite())
                            .collect();

                        if valid_values.len() > 10 {
                            let avg_bbw =
                                valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                            // For alternating high/low pattern, BBW should be substantial
                            prop_assert!(
                                avg_bbw > 0.1,
                                "BBW should be substantial for volatile data: avg={}",
                                avg_bbw
                            );
                        }
                    }

                    // Property 7: Mathematical relationship - BBW scales with deviation multipliers
                    // BBW = (devup + devdn) * stddev / ma
                    // So if we double devup and devdn, BBW should approximately double
                    if data_type == "random" && out.len() > period * 2 {
                        // Run the same data with doubled deviations
                        let params_double = BollingerBandsWidthParams {
                            period: Some(period),
                            devup: Some(devup * 2.0),
                            devdn: Some(devdn * 2.0),
                            matype: Some("sma".to_string()),
                            devtype: Some(0),
                        };
                        let input_double =
                            BollingerBandsWidthInput::from_slice(&data, params_double);
                        let result_double =
                            bollinger_bands_width_with_kernel(&input_double, kernel);

                        if let Ok(out_double) = result_double {
                            let out_double = out_double.values;

                            // Compare ratios for valid values
                            for i in (period - 1)..out.len().min(period * 3) {
                                if out[i].is_finite() && out_double[i].is_finite() && out[i] > 1e-6
                                {
                                    let ratio = out_double[i] / out[i];
                                    // The ratio should be approximately 2.0 (within 10% tolerance)
                                    prop_assert!(
                                        (ratio - 2.0).abs() < 0.2,
                                        "BBW scaling issue at index {}: ratio={} (expected ~2.0)",
                                        i,
                                        ratio
                                    );
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

    macro_rules! generate_all_bbw_tests {
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

    generate_all_bbw_tests!(
        check_bbw_partial_params,
        check_bbw_default,
        check_bbw_zero_period,
        check_bbw_period_exceeds_length,
        check_bbw_very_small_dataset,
        check_bbw_nan_check,
        check_bollinger_bands_width_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_bbw_tests!(check_bollinger_bands_width_property);

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

#[cfg(feature = "python")]
#[pyfunction(name = "bollinger_bands_width")]
#[pyo3(signature = (data, period, devup, devdn, matype=None, devtype=None, kernel=None))]
pub fn bollinger_bands_width_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    devup: f64,
    devdn: f64,
    matype: Option<&str>,
    devtype: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use crate::utilities::kernel_validation::validate_kernel;
    use numpy::{IntoPyArray, PyArrayMethods};
    use pyo3::exceptions::PyValueError;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = BollingerBandsWidthParams {
        period: Some(period),
        devup: Some(devup),
        devdn: Some(devdn),
        matype: matype.map(|s| s.to_string()),
        devtype: devtype,
    };
    let bbw_in = BollingerBandsWidthInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| bollinger_bands_width_with_kernel(&bbw_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "BollingerBandsWidthStream")]
pub struct BollingerBandsWidthStreamPy {
    inner: BollingerBandsWidthStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BollingerBandsWidthStreamPy {
    #[new]
    fn new(
        period: usize,
        devup: f64,
        devdn: f64,
        matype: Option<&str>,
        devtype: Option<usize>,
    ) -> PyResult<Self> {
        if period == 0 {
            return Err(PyValueError::new_err("period must be > 0"));
        }
        let mt = matype.unwrap_or("sma");
        let dt = devtype.unwrap_or(0);
        Ok(Self {
            inner: BollingerBandsWidthStream::new(period, devup, devdn, mt, dt),
        })
    }

    /// O(1) update; returns None until warm-up completes.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "bollinger_bands_width_batch")]
#[pyo3(signature = (data, period_range, devup_range, devdn_range, kernel=None))]
pub fn bollinger_bands_width_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    devup_range: (f64, f64, f64),
    devdn_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;
    let slice_in = data.as_slice()?;

    let sweep = BollingerBandsWidthBatchRange {
        period: period_range,
        devup: devup_range,
        devdn: devdn_range,
    };

    let combos = expand_grid_checked(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("bbw: output size overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Initialize NaN prefixes for each row based on warmup period
    let first = slice_in
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("All values NaN"))?;
    for (r, p) in combos.iter().enumerate() {
        let warm = first + p.period.unwrap() - 1;
        let start = r * cols;
        slice_out[start..start + warm.min(cols)].fill(f64::NAN);
    }

    let kern = validate_kernel(kernel, true)?;
    let combos = py
        .allow_threads(|| {
            let k = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            // Map batch kernel -> simd scalar/batch like alma.rs
            let simd = match k {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            bollinger_bands_width_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "devups",
        combos
            .iter()
            .map(|p| p.devup.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "devdns",
        combos
            .iter()
            .map(|p| p.devdn.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "bollinger_bands_width_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, devup_range, devdn_range, device_id=0))]
pub fn bollinger_bands_width_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    devup_range: (f64, f64, f64),
    devdn_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, PyDict>)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = BollingerBandsWidthBatchRange {
        period: period_range,
        devup: devup_range,
        devdn: devdn_range,
    };

    let (inner, dev_id, combos) = py.allow_threads(|| {
        let cuda = CudaBbw::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let (arr, meta) = cuda
            .bbw_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, dev_id, meta))
    })?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|(p, _)| *p as u64).collect();
    let uplusd: Vec<f64> = combos.iter().map(|(_, k)| *k as f64).collect();
    let uplusd_len = uplusd.len();
    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("u_plus_d", uplusd.into_pyarray(py))?;
    // For SMA/stddev-only CUDA path, expose fixed metadata
    dict.set_item("ma_types", PyList::new(py, vec!["sma"; uplusd_len])?)?;
    dict.set_item("devtypes", vec![0u64; uplusd_len].into_pyarray(py))?;

    let handle = make_device_array_py(dev_id as usize, inner)?;
    Ok((handle, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "bollinger_bands_width_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, cols, rows, period, devup, devdn, device_id=0))]
pub fn bollinger_bands_width_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    devup: f64,
    devdn: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_tm_f32.as_slice()?;
    let (inner, dev_id) = py.allow_threads(|| {
        let cuda = CudaBbw::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let arr = cuda
            .bbw_many_series_one_param_time_major_dev(
                slice_in,
                cols,
                rows,
                period,
                devup as f32,
                devdn as f32,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, dev_id))
    })?;
    make_device_array_py(dev_id as usize, inner)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bbw_alloc(len: usize) -> *mut f64 {
    let mut v: Vec<f64> = Vec::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bbw_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_js(
    data: &[f64],
    period: usize,
    devup: f64,
    devdn: f64,
    matype: Option<String>,
    devtype: Option<usize>,
) -> Result<Vec<f64>, JsValue> {
    let params = BollingerBandsWidthParams {
        period: Some(period),
        devup: Some(devup),
        devdn: Some(devdn),
        matype: matype.or_else(|| Some("sma".to_string())),
        devtype: devtype.or(Some(0)),
    };
    let input = BollingerBandsWidthInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    bollinger_bands_width_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    devup_start: f64,
    devup_end: f64,
    devup_step: f64,
    devdn_start: f64,
    devdn_end: f64,
    devdn_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = BollingerBandsWidthBatchRange {
        period: (period_start, period_end, period_step),
        devup: (devup_start, devup_end, devup_step),
        devdn: (devdn_start, devdn_end, devdn_step),
    };

    bollinger_bands_width_batch_inner(data, &sweep, detect_best_batch_kernel(), false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    devup_start: f64,
    devup_end: f64,
    devup_step: f64,
    devdn_start: f64,
    devdn_end: f64,
    devdn_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = BollingerBandsWidthBatchRange {
        period: (period_start, period_end, period_step),
        devup: (devup_start, devup_end, devup_step),
        devdn: (devdn_start, devdn_end, devdn_step),
    };

    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len() * 3);

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
        metadata.push(combo.devup.unwrap());
        metadata.push(combo.devdn.unwrap());
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BollingerBandsWidthBatchConfig {
    pub period_range: (usize, usize, usize),
    pub devup_range: (f64, f64, f64),
    pub devdn_range: (f64, f64, f64),
    pub matype: Option<String>,
    pub devtype: Option<usize>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BollingerBandsWidthBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<BollingerBandsWidthParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = bollinger_bands_width_batch)]
pub fn bollinger_bands_width_batch_unified_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    // Deserialize the configuration object from JavaScript
    let config: BollingerBandsWidthBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = BollingerBandsWidthBatchRange {
        period: config.period_range,
        devup: config.devup_range,
        devdn: config.devdn_range,
    };

    // Run the existing core logic
    let mut output =
        bollinger_bands_width_batch_inner(data, &sweep, detect_best_batch_kernel(), false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Update combos with matype and devtype
    for combo in &mut output.combos {
        combo.matype = config.matype.clone().or_else(|| Some("sma".to_string()));
        combo.devtype = config.devtype.or(Some(0));
    }

    // Create the structured output
    let js_output = BollingerBandsWidthBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Fast API for WASM bindings - handles aliasing
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    devup: f64,
    devdn: f64,
    matype: Option<String>,
    devtype: Option<usize>,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let params = BollingerBandsWidthParams {
            period: Some(period),
            devup: Some(devup),
            devdn: Some(devdn),
            matype: matype.or_else(|| Some("sma".to_string())),
            devtype: devtype.or(Some(0)),
        };
        let input = BollingerBandsWidthInput::from_slice(data, params);
        if core::ptr::eq(in_ptr, out_ptr as *const f64) {
            let mut tmp = vec![0.0; len];
            bollinger_bands_width_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            out.copy_from_slice(&tmp);
        } else {
            bollinger_bands_width_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

// Fast batch API for WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    devup_start: f64,
    devup_end: f64,
    devup_step: f64,
    devdn_start: f64,
    devdn_end: f64,
    devdn_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to bollinger_bands_width_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = BollingerBandsWidthBatchRange {
            period: (period_start, period_end, period_step),
            devup: (devup_start, devup_end, devup_step),
            devdn: (devdn_start, devdn_end, devdn_step),
        };

        let combos = expand_grid_checked(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("bbw: output size overflow"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        // Initialize NaN prefixes for each row based on warmup period
        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(JsValue::from_str("All values NaN"))?;
        for (r, combo) in combos.iter().enumerate() {
            let warm = first + combo.period.unwrap() - 1;
            let start = r * cols;
            out[start..start + warm.min(cols)].fill(f64::NAN);
        }

        bollinger_bands_width_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
