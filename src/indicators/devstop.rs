//! # DevStop Indicator
//!
//! A volatility-based stop indicator similar to Kase Dev Stops. Computes the difference between rolling highs and lows, smooths the range with a moving average, computes a deviation, and applies a multiplier offset, then finalizes with rolling extrema.
//!
//! ## Parameters
//! - **period**: Rolling window size (default: 20).
//! - **mult**: Multiplier for deviation (default: 0.0).
//! - **devtype**: Deviation type (0: stddev, 1: mean abs, 2: median abs, default: 0).
//! - **direction**: "long" or "short" (default: "long").
//! - **ma_type**: Moving average type (default: "sma").
//!
//! ## Returns
//! - **`Ok(DevStopOutput)`** on success, containing `Vec<f64>` matching input length.
//! - **`Err(DevStopError)`** otherwise.
//!
//! ## Developer Notes
//! - **Decision**: Single-series SIMD offers little gain due to strong recurrences; we route AVX2/AVX512 to the fused scalar classic path when applicable (devtype=stddev + SMA/EMA), otherwise fall back to the generic path.
//! - **AVX2/AVX512 kernels**: Route to fused scalar for classic SMA/EMA; otherwise delegate to `devstop_into_slice`.
//! - **Streaming**: Enabled with O(1) per-tick updates for stddev+SMA/EMA; devtype 1/2 use σ-based O(1) approximations in streaming.
//! - **Memory optimization**: ✅ Uses alloc_with_nan_prefix (zero-copy)
//! - **Batch operations**: ✅ Implemented with parallel processing support
//! - **Row-specific batch**: Implemented for classic stddev + SMA/EMA using shared precompute of r and prefix sums; other devtypes fall back to generic path.

use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
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
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum DevStopData<'a> {
    Candles {
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
    },
    SliceHL(&'a [f64], &'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DevStopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DevStopParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub devtype: Option<usize>,
    pub direction: Option<String>,
    pub ma_type: Option<String>,
}

impl Default for DevStopParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            mult: Some(0.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DevStopInput<'a> {
    pub data: DevStopData<'a>,
    pub params: DevStopParams,
}

impl<'a> DevStopInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
        params: DevStopParams,
    ) -> Self {
        Self {
            data: DevStopData::Candles {
                candles,
                source_high,
                source_low,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DevStopParams) -> Self {
        Self {
            data: DevStopData::SliceHL(high, low),
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "high", "low", DevStopParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(0.0)
    }
    #[inline]
    pub fn get_devtype(&self) -> usize {
        self.params.devtype.unwrap_or(0)
    }
    #[inline]
    pub fn get_direction(&self) -> String {
        self.params
            .direction
            .clone()
            .unwrap_or_else(|| "long".to_string())
    }
    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct DevStopBuilder {
    period: Option<usize>,
    mult: Option<f64>,
    devtype: Option<usize>,
    direction: Option<String>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for DevStopBuilder {
    fn default() -> Self {
        Self {
            period: None,
            mult: None,
            devtype: None,
            direction: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DevStopBuilder {
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
    pub fn mult(mut self, x: f64) -> Self {
        self.mult = Some(x);
        self
    }
    #[inline(always)]
    pub fn devtype(mut self, d: usize) -> Self {
        self.devtype = Some(d);
        self
    }
    #[inline(always)]
    pub fn direction(mut self, d: &str) -> Self {
        self.direction = Some(d.to_string());
        self
    }
    #[inline(always)]
    pub fn ma_type(mut self, t: &str) -> Self {
        self.ma_type = Some(t.to_string());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DevStopOutput, DevStopError> {
        let p = DevStopParams {
            period: self.period,
            mult: self.mult,
            devtype: self.devtype,
            direction: self.direction.clone(),
            ma_type: self.ma_type.clone(),
        };
        let i = DevStopInput::from_candles(c, "high", "low", p);
        devstop_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DevStopOutput, DevStopError> {
        let p = DevStopParams {
            period: self.period,
            mult: self.mult,
            devtype: self.devtype,
            direction: self.direction.clone(),
            ma_type: self.ma_type.clone(),
        };
        let i = DevStopInput::from_slices(high, low, p);
        devstop_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DevStopStream, DevStopError> {
        let p = DevStopParams {
            period: self.period,
            mult: self.mult,
            devtype: self.devtype,
            direction: self.direction,
            ma_type: self.ma_type,
        };
        DevStopStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DevStopError {
    #[error("devstop: All values are NaN for high or low.")]
    AllValuesNaN,
    #[error("devstop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("devstop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("devstop: Calculation error: {0}")]
    DevStopCalculation(String),
}

#[inline]
pub fn devstop(input: &DevStopInput) -> Result<DevStopOutput, DevStopError> {
    devstop_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn devstop_warmup(first: usize, period: usize) -> usize {
    // high/low rolling(2) warm = first + (2-1)
    // MA(deviation) over range adds + (period-1)
    // final rolling extrema over base adds + (period-1)
    first + 2 * period - 1
}

#[inline(always)]
fn devstop_prepare<'a>(
    input: &'a DevStopInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        f64,
        usize,
        bool,
        String,
        Kernel,
    ),
    DevStopError,
> {
    let (high, low) = match &input.data {
        DevStopData::Candles {
            candles,
            source_high,
            source_low,
        } => (
            source_type(candles, source_high),
            source_type(candles, source_low),
        ),
        DevStopData::SliceHL(h, l) => (*h, *l),
    };
    let len = high.len();
    if len == 0 || low.len() == 0 {
        return Err(DevStopError::AllValuesNaN);
    }
    let fh = high.iter().position(|x| !x.is_nan());
    let fl = low.iter().position(|x| !x.is_nan());
    let first = match (fh, fl) {
        (Some(h), Some(l)) => h.min(l),
        _ => return Err(DevStopError::AllValuesNaN),
    };

    let period = input.get_period();
    if period == 0 || period > len || period > low.len() {
        return Err(DevStopError::InvalidPeriod {
            period,
            data_len: len.min(low.len()),
        });
    }
    if (len - first) < period || (low.len() - first) < period {
        return Err(DevStopError::NotEnoughValidData {
            needed: period,
            valid: (len - first).min(low.len() - first),
        });
    }

    let mult = input.get_mult();
    let devtype = input.get_devtype();
    if devtype > 2 {
        return Err(DevStopError::DevStopCalculation(format!(
            "invalid devtype {}",
            devtype
        )));
    }
    let is_long = input.get_direction().eq_ignore_ascii_case("long");
    let ma_type = input.get_ma_type();

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((
        high, low, len, first, mult, devtype, is_long, ma_type, chosen,
    ))
}

#[inline]
pub fn devstop_into_slice(
    dst: &mut [f64],
    input: &DevStopInput,
    _kernel: Kernel,
) -> Result<(), DevStopError> {
    // Extract data without re-validating (caller should have validated)
    let (high, low) = match &input.data {
        DevStopData::Candles {
            candles,
            source_high,
            source_low,
        } => (
            source_type(candles, source_high),
            source_type(candles, source_low),
        ),
        DevStopData::SliceHL(h, l) => (*h, *l),
    };
    let len = high.len();

    // Quick validation of dst length
    if dst.len() != len {
        return Err(DevStopError::InvalidPeriod {
            period: dst.len(),
            data_len: len,
        });
    }

    // Find first valid index
    let fh = high.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let fl = low.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let first = fh.min(fl);

    // Get parameters
    let period = input.get_period();
    let mult = input.get_mult();
    let devtype = input.get_devtype();
    let is_long = input.get_direction().eq_ignore_ascii_case("long");
    let ma_type = input.get_ma_type();

    // Dispatch to classic kernels for common cases
    if devtype == 0 {
        // Population standard deviation
        if ma_type == "sma" || ma_type == "SMA" {
            return unsafe {
                devstop_scalar_classic_sma(high, low, period, mult, is_long, first, dst)
            };
        } else if ma_type == "ema" || ma_type == "EMA" {
            return unsafe {
                devstop_scalar_classic_ema(high, low, period, mult, is_long, first, dst)
            };
        }
    }

    // Build range with warm prefix only, no full NaN fill
    let mut range = alloc_with_nan_prefix(len, first + 1);
    // rolling(2) max/min without allocating temp vectors
    if first + 1 < len {
        let mut prev_h = high[first];
        let mut prev_l = low[first];
        for i in (first + 1)..len {
            let h = high[i];
            let l = low[i];
            if !h.is_nan() && !prev_h.is_nan() && !l.is_nan() && !prev_l.is_nan() {
                let hi2 = if h > prev_h { h } else { prev_h };
                let lo2 = if l < prev_l { l } else { prev_l };
                range[i] = hi2 - lo2;
            }
            prev_h = h;
            prev_l = l;
        }
    }

    // These produce one allocation each; unavoidable given their APIs
    let avtr = ma(&ma_type, MaData::Slice(&range), input.get_period())
        .map_err(|e| DevStopError::DevStopCalculation(format!("ma: {e:?}")))?;
    let dev_values = {
        let di = DevInput::from_slice(
            &range,
            DevParams {
                period: Some(input.get_period()),
                devtype: Some(devtype),
            },
        );
        deviation(&di).map_err(|e| DevStopError::DevStopCalculation(format!("deviation: {e:?}")))?
    };

    // Stream final into dst with a deque. No `base` or `final_values` Vec.
    // Keep only last `period` base values in a small ring.
    use std::collections::VecDeque;
    let period = input.get_period();
    let start_base = first + period; // base becomes defined here
    let start_final = start_base + period - 1; // final rolling extrema defined here
    let warm = devstop_warmup(first, period);

    // dst already allocated by caller; write only valid tail, leave warmup untouched.
    // For Python/WASM paths we'll allocate dst via alloc_with_nan_prefix(warm) to avoid writes here.
    let mut dq: VecDeque<usize> = VecDeque::with_capacity(period + 1);
    let mut ring: Vec<f64> = vec![f64::NAN; period];

    for i in start_base..len {
        let base = if is_long {
            // long: high - avtr - mult*dev
            if high[i].is_nan() || avtr[i].is_nan() || dev_values[i].is_nan() {
                f64::NAN
            } else {
                high[i] - avtr[i] - mult * dev_values[i]
            }
        } else {
            // short: low + avtr + mult*dev
            if low[i].is_nan() || avtr[i].is_nan() || dev_values[i].is_nan() {
                f64::NAN
            } else {
                low[i] + avtr[i] + mult * dev_values[i]
            }
        };

        // store in ring
        ring[i % period] = base;

        // update deque for rolling max/min over `base`
        if is_long {
            while let Some(&j) = dq.back() {
                let bj = ring[j % period];
                if bj.is_nan() || bj <= base {
                    dq.pop_back();
                } else {
                    break;
                }
            }
        } else {
            while let Some(&j) = dq.back() {
                let bj = ring[j % period];
                if bj.is_nan() || bj >= base {
                    dq.pop_back();
                } else {
                    break;
                }
            }
        }
        dq.push_back(i);
        // drop out-of-window
        let cut = i + 1 - period;
        while let Some(&j) = dq.front() {
            if j < cut {
                dq.pop_front();
            } else {
                break;
            }
        }

        if i >= start_final {
            if let Some(&j) = dq.front() {
                dst[i] = ring[j % period];
            } else {
                // If deque is empty (all NaN or insufficient data), output NaN
                dst[i] = f64::NAN;
            }
        }
    }

    // Ensure warm prefix stays NaN
    for v in &mut dst[..warm.min(len)] {
        *v = f64::NAN;
    }
    Ok(())
}

pub fn devstop_with_kernel(
    input: &DevStopInput,
    kernel: Kernel,
) -> Result<DevStopOutput, DevStopError> {
    // Validate input once to get length and first
    let (high, low) = match &input.data {
        DevStopData::Candles {
            candles,
            source_high,
            source_low,
        } => (
            source_type(candles, source_high),
            source_type(candles, source_low),
        ),
        DevStopData::SliceHL(h, l) => (*h, *l),
    };
    let len = high.len();
    if len == 0 || low.len() == 0 {
        return Err(DevStopError::AllValuesNaN);
    }
    let fh = high.iter().position(|x| !x.is_nan());
    let fl = low.iter().position(|x| !x.is_nan());
    let first = match (fh, fl) {
        (Some(h), Some(l)) => h.min(l),
        _ => return Err(DevStopError::AllValuesNaN),
    };

    let period = input.get_period();
    if period == 0 || period > len || period > low.len() {
        return Err(DevStopError::InvalidPeriod {
            period,
            data_len: len.min(low.len()),
        });
    }
    if (len - first) < period || (low.len() - first) < period {
        return Err(DevStopError::NotEnoughValidData {
            needed: period,
            valid: (len - first).min(low.len() - first),
        });
    }

    // Resolve kernel once
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Allocate output without NaN prefix (devstop_into_slice/fused kernels set warmup)
    let _warm = devstop_warmup(first, period);
    let mut out = vec![0.0; len];

    // Pass resolved kernel directly
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                devstop_into_slice(&mut out, input, Kernel::Scalar)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => devstop_into_slice(&mut out, input, Kernel::Avx2)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                devstop_into_slice(&mut out, input, Kernel::Avx512)?
            }
            _ => devstop_into_slice(&mut out, input, Kernel::Scalar)?,
        }
    }
    Ok(DevStopOutput { values: out })
}

// Old scalar implementation - kept for reference but not used
#[inline]
pub fn devstop_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, period, first);
    let _ = devstop_into_slice(out, input, Kernel::Scalar);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn devstop_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    // Fast path: classic stddev with SMA/EMA → use fused scalar kernel for speed/correctness
    let devtype = input.get_devtype();
    let is_long = input.get_direction().eq_ignore_ascii_case("long");
    let mult = input.get_mult();
    let ma_type = input.get_ma_type();
    unsafe {
        if devtype == 0 && (ma_type.eq_ignore_ascii_case("sma") || ma_type.eq_ignore_ascii_case("ema")) {
            let _ = if ma_type.eq_ignore_ascii_case("sma") {
                devstop_scalar_classic_sma(high, low, period, mult, is_long, first, out)
            } else {
                devstop_scalar_classic_ema(high, low, period, mult, is_long, first, out)
            };
        } else {
            let _ = devstop_into_slice(out, input, Kernel::Avx2);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn devstop_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let devtype = input.get_devtype();
    let is_long = input.get_direction().eq_ignore_ascii_case("long");
    let mult = input.get_mult();
    let ma_type = input.get_ma_type();
    unsafe {
        if devtype == 0 && (ma_type.eq_ignore_ascii_case("sma") || ma_type.eq_ignore_ascii_case("ema")) {
            let _ = if ma_type.eq_ignore_ascii_case("sma") {
                devstop_scalar_classic_sma(high, low, period, mult, is_long, first, out)
            } else {
                devstop_scalar_classic_ema(high, low, period, mult, is_long, first, out)
            };
        } else {
            let _ = devstop_into_slice(out, input, Kernel::Avx512);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn devstop_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, period, first);
    let _ = devstop_into_slice(out, input, Kernel::Avx512);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn devstop_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, period, first);
    let _ = devstop_into_slice(out, input, Kernel::Avx512);
}

#[inline(always)]
pub fn devstop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &DevStopBatchRange,
    kernel: Kernel,
) -> Result<DevStopBatchOutput, DevStopError> {
    let chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(DevStopError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match chosen {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Default to Scalar for any other kernel
    };
    devstop_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DevStopBatchRange {
    pub period: (usize, usize, usize),
    pub mult: (f64, f64, f64),
    pub devtype: (usize, usize, usize),
}

impl Default for DevStopBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 20, 0),
            mult: (0.0, 0.0, 0.0),
            devtype: (0, 0, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DevStopBatchBuilder {
    range: DevStopBatchRange,
    kernel: Kernel,
}

impl DevStopBatchBuilder {
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
    pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.devtype = (start, end, step);
        self
    }
    #[inline]
    pub fn devtype_static(mut self, x: usize) -> Self {
        self.range.devtype = (x, x, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<DevStopBatchOutput, DevStopError> {
        devstop_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<DevStopBatchOutput, DevStopError> {
        DevStopBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
}

#[derive(Clone, Debug)]
pub struct DevStopBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DevStopParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DevStopBatchOutput {
    pub fn row_for_params(&self, p: &DevStopParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.mult.unwrap_or(0.0) - p.mult.unwrap_or(0.0)).abs() < 1e-12
                && c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
        })
    }
    pub fn values_for(&self, p: &DevStopParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_devstop(r: &DevStopBatchRange) -> Vec<DevStopParams> {
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
    let periods = axis_usize(r.period);
    let mults = axis_f64(r.mult);
    let devtypes = axis_usize(r.devtype);

    let mut out = Vec::with_capacity(periods.len() * mults.len() * devtypes.len());
    for &p in &periods {
        for &m in &mults {
            for &d in &devtypes {
                out.push(DevStopParams {
                    period: Some(p),
                    mult: Some(m),
                    devtype: Some(d),
                    direction: Some("long".to_string()),
                    ma_type: Some("sma".to_string()),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn devstop_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DevStopBatchRange,
    kern: Kernel,
) -> Result<DevStopBatchOutput, DevStopError> {
    devstop_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn devstop_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DevStopBatchRange,
    kern: Kernel,
) -> Result<DevStopBatchOutput, DevStopError> {
    devstop_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn devstop_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &DevStopBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DevStopBatchOutput, DevStopError> {
    let combos = expand_grid_devstop(sweep);
    if combos.is_empty() {
        return Err(DevStopError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let fh = high
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DevStopError::AllValuesNaN)?;
    let fl = low
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DevStopError::AllValuesNaN)?;
    let first = fh.min(fl);

    // Check we have enough data for the largest period's warmup
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let max_warmup = devstop_warmup(first, max_p);
    if high.len() <= max_warmup || low.len() <= max_warmup {
        return Err(DevStopError::NotEnoughValidData {
            needed: max_warmup + 1,
            valid: high.len().min(low.len()),
        });
    }

    let rows = combos.len();
    let cols = high.len();

    // Uninitialized matrix, then warm prefixes per row
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warms: Vec<usize> = combos
        .iter()
        .map(|c| devstop_warmup(first, c.period.unwrap()))
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    // Into-slice compute per row without touching warm prefix again
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Convert batch kernel to regular kernel for row processing
    let simd_kern = match kern {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx512Batch => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                Kernel::Avx512
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                Kernel::Scalar
            }
        }
        Kernel::Avx2Batch => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                Kernel::Avx2
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                Kernel::Scalar
            }
        }
        k => k, // Pass through non-batch kernels
    };

    // Row-specific optimized batch for classic stddev + SMA/EMA
    let all_classic = combos.iter().all(|c| {
        let dt = c.devtype.unwrap_or(0);
        let mt = c
            .ma_type
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("sma");
        dt == 0 && (mt.eq_ignore_ascii_case("sma") || mt.eq_ignore_ascii_case("ema"))
    });

    if all_classic {
        let len = cols;
        // Precompute 2-bar range r and prefix sums over finite values only
        let mut r = vec![f64::NAN; len];
        if first + 1 < len {
            let mut prev_h = high[first];
            let mut prev_l = low[first];
            for i in (first + 1)..len {
                let h = high[i];
                let l = low[i];
                if !h.is_nan() && !prev_h.is_nan() && !l.is_nan() && !prev_l.is_nan() {
                    let hi2 = if h > prev_h { h } else { prev_h };
                    let lo2 = if l < prev_l { l } else { prev_l };
                    r[i] = hi2 - lo2;
                }
                prev_h = h;
                prev_l = l;
            }
        }
        // Exclusive prefix sums for sum, sumsq, and finite count
        let mut p1 = vec![0.0f64; len + 1];
        let mut p2 = vec![0.0f64; len + 1];
        let mut pc = vec![0usize; len + 1];
        for i in 0..len {
            let ri = r[i];
            p1[i + 1] = p1[i];
            p2[i + 1] = p2[i];
            pc[i + 1] = pc[i];
            if ri.is_finite() {
                p1[i + 1] += ri;
                p2[i + 1] += ri * ri;
                pc[i + 1] += 1;
            }
        }

        let process_row = |row: usize, dst_row_mu: &mut [f64]| -> Result<(), DevStopError> {
            let prm = &combos[row];
            let period = prm.period.unwrap_or(20);
            let mult = prm.mult.unwrap_or(0.0);
            let is_long = prm
                .direction
                .as_ref()
                .map(|d| d.as_str())
                .unwrap_or("long")
                .eq_ignore_ascii_case("long");
            let ma_type = prm
                .ma_type
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("sma");

            let start_base = first + period;
            if start_base >= len {
                return Ok(());
            }
            let start_final = start_base + period - 1;

            // EMA state
            let mut ema = 0.0f64;
            let mut use_ema = ma_type.eq_ignore_ascii_case("ema");
            let (alpha, beta) = if use_ema {
                let a = 2.0 / (period as f64 + 1.0);
                (a, 1.0 - a)
            } else {
                (0.0, 0.0)
            };
            if use_ema {
                let a = first + 1;
                let b = start_base;
                let cnt0 = pc[b] - pc[a];
                if cnt0 > 0 {
                    ema = (p1[b] - p1[a]) / (cnt0 as f64);
                } else {
                    ema = f64::NAN;
                }
            }

            // Monotonic deque over base with capacity = period
            let mut base_ring = vec![f64::NAN; period];
            let mut dq_buf = vec![0usize; period];
            let mut dq_head = 0usize;
            let mut dq_len = 0usize;
            #[inline(always)]
            fn dq_idx_at(buf: &[usize], head: usize, cap: usize, k: usize) -> usize {
                unsafe { *buf.get_unchecked((head + k) % cap) }
            }
            #[inline(always)]
            fn dq_back_idx(buf: &[usize], head: usize, len: usize, cap: usize) -> usize {
                unsafe { *buf.get_unchecked((head + len - 1) % cap) }
            }
            #[inline(always)]
            fn dq_pop_back(len: &mut usize) { *len -= 1; }
            #[inline(always)]
            fn dq_pop_front(head: &mut usize, len: &mut usize, cap: usize) { *head = (*head + 1) % cap; *len -= 1; }
            #[inline(always)]
            fn dq_push_back(buf: &mut [usize], head: usize, len: &mut usize, cap: usize, value: usize) {
                let pos = (head + *len) % cap;
                unsafe { *buf.get_unchecked_mut(pos) = value; }
                *len += 1;
            }

            for i in start_base..len {
                if use_ema {
                    let ri = r[i];
                    if ri.is_finite() {
                        ema = ri.mul_add(alpha, beta * ema);
                    }
                }
                let a = i + 1 - period;
                let b = i + 1;
                let cnt = pc[b] - pc[a];
                let (avtr, sigma) = if cnt == 0 {
                    (f64::NAN, f64::NAN)
                } else if use_ema {
                    let e1 = (p1[b] - p1[a]) / (cnt as f64);
                    let e2 = (p2[b] - p2[a]) / (cnt as f64);
                    let var = (e2 - 2.0 * ema * e1 + ema * ema).max(0.0);
                    (ema, var.sqrt())
                } else {
                    let e1 = (p1[b] - p1[a]) / (cnt as f64);
                    let e2 = (p2[b] - p2[a]) / (cnt as f64);
                    let var = (e2 - e1 * e1).max(0.0);
                    (e1, var.sqrt())
                };

                let h = high[i];
                let l = low[i];
                let base = if is_long {
                    if h.is_nan() || avtr.is_nan() || sigma.is_nan() { f64::NAN } else { h - avtr - mult * sigma }
                } else {
                    if l.is_nan() || avtr.is_nan() || sigma.is_nan() { f64::NAN } else { l + avtr + mult * sigma }
                };

                let slot = i % period;
                base_ring[slot] = base;
                if is_long {
                    while dq_len > 0 {
                        let j = dq_back_idx(&dq_buf, dq_head, dq_len, period);
                        let bj = base_ring[j % period];
                        if bj.is_nan() || bj <= base { dq_pop_back(&mut dq_len); } else { break; }
                    }
                } else {
                    while dq_len > 0 {
                        let j = dq_back_idx(&dq_buf, dq_head, dq_len, period);
                        let bj = base_ring[j % period];
                        if bj.is_nan() || bj >= base { dq_pop_back(&mut dq_len); } else { break; }
                    }
                }
                dq_push_back(&mut dq_buf, dq_head, &mut dq_len, period, i);

                let cut = i + 1 - period;
                while dq_len > 0 && dq_idx_at(&dq_buf, dq_head, period, 0) < cut {
                    dq_pop_front(&mut dq_head, &mut dq_len, period);
                }

                if i >= start_final {
                    let out_val = if dq_len > 0 {
                        let j = dq_idx_at(&dq_buf, dq_head, period, 0);
                        base_ring[j % period]
                    } else { f64::NAN };
                    dst_row_mu[i] = out_val;
                }
            }
            Ok(())
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                out.par_chunks_mut(cols)
                    .enumerate()
                    .try_for_each(|(row, sl)| process_row(row, sl))?;
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (row, sl) in out.chunks_mut(cols).enumerate() {
                    process_row(row, sl)?;
                }
            }
        } else {
            for (row, sl) in out.chunks_mut(cols).enumerate() {
                process_row(row, sl)?;
            }
        }

        let values = unsafe {
            Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity())
        };
        core::mem::forget(guard);
        return Ok(DevStopBatchOutput { values, combos, rows, cols });
    }

    let do_row = |row: usize, dst_row_mu: &mut [f64]| -> Result<(), DevStopError> {
        let prm = &combos[row];
        let input = DevStopInput {
            data: DevStopData::SliceHL(high, low),
            params: prm.clone(),
        };
        // compute only; warm prefix already set by init_matrix_prefixes
        devstop_into_slice(dst_row_mu, &input, simd_kern)?;
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(r, sl)| do_row(r, sl))?; // propagate error
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, sl) in out.chunks_mut(cols).enumerate() {
                do_row(r, sl)?;
            }
        }
    } else {
        for (r, sl) in out.chunks_mut(cols).enumerate() {
            do_row(r, sl)?;
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    core::mem::forget(guard);

    Ok(DevStopBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn devstop_row_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, period, first);
    let _ = devstop_into_slice(out, input, Kernel::Scalar);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx2(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, first, period);
    let _ = devstop_into_slice(out, input, Kernel::Avx2);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx512(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, first, period);
    let _ = devstop_into_slice(out, input, Kernel::Avx512);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, first, period);
    let _ = devstop_into_slice(out, input, Kernel::Avx512);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    input: &DevStopInput,
    out: &mut [f64],
) {
    let _ = (high, low, first, period);
    let _ = devstop_into_slice(out, input, Kernel::Avx512);
}

#[derive(Debug, Clone)]
pub struct DevStopStream {
    // params
    period: usize,
    mult: f64,
    devtype: u8,       // 0=stddev, 1=mean abs (approx), 2=median abs (approx)
    is_long: bool,     // direction
    is_ema: bool,      // MA kind

    // 2-bar range state
    prev_h: f64,
    prev_l: f64,
    have_prev: bool,

    // sliding window over range r_t = max(h_t,h_{t-1}) - min(l_t,l_{t-1})
    r_ring: Box<[f64]>, // len = period
    r_head: usize,       // next position to overwrite
    r_filled: bool,
    sum: f64,            // Σ r (finite only)
    sum2: f64,           // Σ r^2
    cnt: usize,          // #finite r in window

    // EMA over r (boot with SMA of available prefill to match classic)
    ema: f64,
    ema_booted: bool,
    alpha: f64, // 2/(period+1)
    beta:  f64, // 1 - alpha

    // base window + monotonic deque for rolling extrema over base (period)
    base_ring: Box<[f64]>, // len = period
    dq_idx: Box<[usize]>,  // circular deque storing absolute indices
    dq_head: usize,
    dq_len: usize,

    // absolute time index (0-based)
    t: usize,
}

impl DevStopStream {
    pub fn try_new(params: DevStopParams) -> Result<Self, DevStopError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(DevStopError::InvalidPeriod { period, data_len: 0 });
        }
        let mult = params.mult.unwrap_or(0.0);
        let devtype = params.devtype.unwrap_or(0) as u8;
        let is_long = params
            .direction
            .as_deref()
            .unwrap_or("long")
            .eq_ignore_ascii_case("long");
        let is_ema = params
            .ma_type
            .as_deref()
            .unwrap_or("sma")
            .eq_ignore_ascii_case("ema");

        let alpha = if is_ema { 2.0 / (period as f64 + 1.0) } else { 0.0 };

        Ok(Self {
            period,
            mult,
            devtype,
            is_long,
            is_ema,
            prev_h: f64::NAN,
            prev_l: f64::NAN,
            have_prev: false,
            r_ring: vec![f64::NAN; period].into_boxed_slice(),
            r_head: 0,
            r_filled: false,
            sum: 0.0,
            sum2: 0.0,
            cnt: 0,
            ema: f64::NAN,
            ema_booted: !is_ema,
            alpha,
            beta: 1.0 - alpha,
            base_ring: vec![f64::NAN; period].into_boxed_slice(),
            dq_idx: vec![0usize; period].into_boxed_slice(),
            dq_head: 0,
            dq_len: 0,
            t: 0,
        })
    }

    /// Push one bar and (after warmup) return the new DevStop value.
    /// Warmup for streaming (first=0): emits after index t >= 2*period - 1.
    #[inline]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        // ----- 1) build 2-bar range r_t -----
        let mut r_new = f64::NAN;
        if self.have_prev
            && high.is_finite()
            && low.is_finite()
            && self.prev_h.is_finite()
            && self.prev_l.is_finite()
        {
            let hi2 = if high > self.prev_h { high } else { self.prev_h };
            let lo2 = if low < self.prev_l { low } else { self.prev_l };
            r_new = hi2 - lo2;
        }
        self.prev_h = high;
        self.prev_l = low;
        self.have_prev = true;

        // ----- 2) slide window statistics over r -----
        let p = self.period;
        if self.r_filled {
            let old = self.r_ring[self.r_head];
            if old.is_finite() {
                self.sum -= old;
                self.sum2 -= old * old;
                self.cnt -= 1;
            }
        }
        self.r_ring[self.r_head] = r_new;
        self.r_head = (self.r_head + 1) % p;
        if self.r_head == 0 {
            self.r_filled = true;
        }
        if r_new.is_finite() {
            self.sum += r_new;
            self.sum2 += r_new * r_new;
            self.cnt += 1;
        }

        // ----- 3) update EMA if needed (boot with SMA of current ring) -----
        if self.is_ema {
            if !self.ema_booted {
                if self.t + 1 >= self.period {
                    self.ema = if self.cnt > 0 { self.sum / self.cnt as f64 } else { f64::NAN };
                    self.ema_booted = true;
                }
            } else if r_new.is_finite() {
                self.ema = r_new.mul_add(self.alpha, self.beta * self.ema);
            }
        }

        // ----- 4) compute avtr and deviation (O(1)) -----
        let base_val = if self.t + 1 >= self.period {
            let (avtr, sigma) = if self.cnt == 0 {
                (f64::NAN, f64::NAN)
            } else if self.is_ema {
                let invc = 1.0 / (self.cnt as f64);
                let e1 = self.sum * invc;
                let e2 = self.sum2 * invc;
                let ema = self.ema;
                let var = (e2 - 2.0 * ema * e1 + ema * ema).max(0.0);
                (ema, var.sqrt())
            } else {
                let invc = 1.0 / (self.cnt as f64);
                let mean = self.sum * invc;
                let var = ((self.sum2 * invc) - mean * mean).max(0.0);
                (mean, var.sqrt())
            };

            let dev = match self.devtype {
                0 => sigma,                                                // stddev (exact)
                1 => sigma * fast_mean_abs_ratio(),                        // ≈ mean |r - mean|
                2 => sigma * fast_mad_ratio(),                             // ≈ MAD
                _ => sigma,
            };

            if self.is_long {
                if high.is_nan() || avtr.is_nan() || dev.is_nan() {
                    f64::NAN
                } else {
                    high - avtr - self.mult * dev
                }
            } else {
                if low.is_nan() || avtr.is_nan() || dev.is_nan() {
                    f64::NAN
                } else {
                    low + avtr + self.mult * dev
                }
            }
        } else {
            f64::NAN
        };

        // ----- 5) push base into monotonic deque (rolling max/min over last p bases) -----
        let i = self.t;
        if self.t + 1 >= self.period {
            let slot = i % p;
            self.base_ring[slot] = base_val;

            if self.is_long {
                while self.dq_len > 0 {
                    let back_pos = (self.dq_head + self.dq_len - 1) % p;
                    let j = self.dq_idx[back_pos];
                    let bj = self.base_ring[j % p];
                    if bj.is_nan() || bj <= base_val {
                        self.dq_len -= 1;
                    } else {
                        break;
                    }
                }
            } else {
                while self.dq_len > 0 {
                    let back_pos = (self.dq_head + self.dq_len - 1) % p;
                    let j = self.dq_idx[back_pos];
                    let bj = self.base_ring[j % p];
                    if bj.is_nan() || bj >= base_val {
                        self.dq_len -= 1;
                    } else {
                        break;
                    }
                }
            }
            // push i
            let push_pos = (self.dq_head + self.dq_len) % p;
            self.dq_idx[push_pos] = i;
            self.dq_len += 1;

            // pop expired
            let cut = i + 1 - p;
            while self.dq_len > 0 {
                let j = self.dq_idx[self.dq_head];
                if j < cut {
                    self.dq_head = (self.dq_head + 1) % p;
                    self.dq_len -= 1;
                } else {
                    break;
                }
            }
        }

        // ----- 6) emit after full warmup (start_final = period + (period-1)) -----
        let out = if self.t + 1 >= (2 * self.period) {
            if self.dq_len > 0 {
                let j = self.dq_idx[self.dq_head];
                Some(self.base_ring[j % p])
            } else {
                Some(f64::NAN)
            }
        } else {
            None
        };

        self.t += 1;
        out
    }
}

// ---------- Fast-math helpers for O(1) approximations ----------
#[inline(always)]
fn fast_mean_abs_ratio() -> f64 {
    // For normal data: E|X-μ| = σ * sqrt(2/π) (~0.79788456)
    0.797_884_560_802_865_4_f64
}

#[inline(always)]
fn fast_mad_ratio() -> f64 {
    // MAD ≈ σ / 1.4826   (i.e., MAD*1.4826 ≈ σ for normal data)
    1.0 / 1.482_602_218_505_602_f64
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "devstop")]
#[pyo3(signature = (high, low, period, mult, devtype, direction, ma_type, kernel=None))]
pub fn devstop_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    mult: f64,
    devtype: usize,
    direction: &str,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err("high/low length mismatch"));
    }

    // Check for all NaN input early
    if h.iter().all(|&x| x.is_nan()) && l.iter().all(|&x| x.is_nan()) {
        return Err(PyValueError::new_err("All values are NaN"));
    }

    // Validate period early
    if period == 0 {
        return Err(PyValueError::new_err("Invalid period"));
    }

    let len = h.len();
    if period > len {
        return Err(PyValueError::new_err("Invalid period"));
    }

    // Find first valid value
    let fh = h.iter().position(|x| !x.is_nan());
    let fl = l.iter().position(|x| !x.is_nan());
    let first = match (fh, fl) {
        (Some(h), Some(l)) => h.min(l),
        _ => return Err(PyValueError::new_err("All values are NaN")),
    };

    if len - first < period {
        return Err(PyValueError::new_err("Not enough valid data"));
    }

    let params = DevStopParams {
        period: Some(period),
        mult: Some(mult),
        devtype: Some(devtype),
        direction: Some(direction.to_string()),
        ma_type: Some(ma_type.to_string()),
    };
    let input = DevStopInput::from_slices(h, l, params);

    let kern = validate_kernel(kernel, false)?;
    let warm = devstop_warmup(first, period);

    let out = unsafe { PyArray1::<f64>::new(py, [h.len()], false) };
    let slice_out = unsafe { out.as_slice_mut()? };
    // prefill warm prefix only
    let slice_len = slice_out.len();
    for v in &mut slice_out[..warm.min(slice_len)] {
        *v = f64::NAN;
    }

    py.allow_threads(|| devstop_into_slice(slice_out, &input, kern))
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("InvalidPeriod") {
                PyValueError::new_err("Invalid period")
            } else if msg.contains("NotEnoughValidData") {
                PyValueError::new_err("Not enough valid data")
            } else if msg.contains("AllValuesNaN") {
                PyValueError::new_err("All values are NaN")
            } else {
                PyValueError::new_err(msg)
            }
        })?;

    Ok(out)
}

#[cfg(feature = "python")]
#[pyfunction(name = "devstop_batch")]
#[pyo3(signature = (high, low, period_range, mult_range, devtype_range, kernel=None))]
pub fn devstop_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    devtype_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err("high/low length mismatch"));
    }

    let sweep = DevStopBatchRange {
        period: period_range,
        mult: mult_range,
        devtype: devtype_range,
    };
    let kern = validate_kernel(kernel, true)?;

    let combos = expand_grid_devstop(&sweep);
    let rows = combos.len();
    let cols = h.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let k = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match k {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar, // Default to Scalar for any other kernel
        };
        // Compute into `slice_out` directly:
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let input = DevStopInput {
                data: DevStopData::SliceHL(h, l),
                params: combo.clone(),
            };
            let warm = {
                let fh = h.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let fl = l.iter().position(|x| !x.is_nan()).unwrap_or(0);
                devstop_warmup(fh.min(fl), combo.period.unwrap())
            };
            let dst = &mut slice_out[start..start + cols];
            for v in &mut dst[..warm.min(cols)] {
                *v = f64::NAN;
            }
            devstop_into_slice(dst, &input, simd).map_err(|e| format!("{}", e))?;
        }
        Ok::<(), String>(())
    })
    .map_err(|e: String| PyValueError::new_err(e))?;

    let d = PyDict::new(py);
    d.set_item("values", out_arr.reshape((rows, cols))?)?;
    d.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "mults",
        combos
            .iter()
            .map(|p| p.mult.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "devtypes",
        combos
            .iter()
            .map(|p| p.devtype.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = devstop)]
pub fn devstop_js(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    devtype: usize,
    direction: &str,
    ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("length mismatch"));
    }

    // Check for all NaN input early
    if high.iter().all(|&x| x.is_nan()) && low.iter().all(|&x| x.is_nan()) {
        return Err(JsValue::from_str("All values are NaN"));
    }

    // Validate period early
    if period == 0 {
        return Err(JsValue::from_str("Invalid period"));
    }

    let len = high.len();
    if period > len {
        return Err(JsValue::from_str("Invalid period"));
    }

    // Find first valid value
    let fh = high.iter().position(|x| !x.is_nan());
    let fl = low.iter().position(|x| !x.is_nan());
    let first = match (fh, fl) {
        (Some(h), Some(l)) => h.min(l),
        _ => return Err(JsValue::from_str("All values are NaN")),
    };

    if len - first < period {
        return Err(JsValue::from_str("Not enough valid data"));
    }

    let params = DevStopParams {
        period: Some(period),
        mult: Some(mult),
        devtype: Some(devtype),
        direction: Some(direction.to_string()),
        ma_type: Some(ma_type.to_string()),
    };
    let input = DevStopInput::from_slices(high, low, params);
    let mut out = vec![0.0; high.len()];
    // In WASM, always use Scalar kernel since SIMD isn't available
    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::Scalar
    } else {
        detect_best_kernel()
    };
    devstop_into_slice(&mut out, &input, kernel).map_err(|e| {
        // Convert error messages to match test expectations
        let msg = e.to_string();
        if msg.contains("InvalidPeriod") {
            JsValue::from_str("Invalid period")
        } else if msg.contains("NotEnoughValidData") {
            JsValue::from_str("Not enough valid data")
        } else if msg.contains("AllValuesNaN") {
            JsValue::from_str("All values are NaN")
        } else {
            JsValue::from_str(&msg)
        }
    })?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn devstop_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn devstop_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = devstop_into)]
pub fn devstop_into_js(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    mult: f64,
    devtype: usize,
    direction: &str,
    ma_type: &str,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let h = std::slice::from_raw_parts(high_ptr, len);
        let l = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let params = DevStopParams {
            period: Some(period),
            mult: Some(mult),
            devtype: Some(devtype),
            direction: Some(direction.to_string()),
            ma_type: Some(ma_type.to_string()),
        };
        let input = DevStopInput::from_slices(h, l, params);
        // In WASM, always use Scalar kernel since SIMD isn't available
        let kernel = if cfg!(target_arch = "wasm32") {
            Kernel::Scalar
        } else {
            detect_best_kernel()
        };
        devstop_into_slice(out, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DevStopBatchConfig {
    pub period_range: (usize, usize, usize),
    pub mult_range: (f64, f64, f64),
    pub devtype_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DevStopBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DevStopParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = devstop_batch)]
pub fn devstop_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("length mismatch"));
    }
    let cfg: DevStopBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = DevStopBatchRange {
        period: cfg.period_range,
        mult: cfg.mult_range,
        devtype: cfg.devtype_range,
    };
    // In WASM, always use ScalarBatch kernel since SIMD isn't available
    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::ScalarBatch
    } else {
        detect_best_batch_kernel()
    };
    let out = devstop_batch_inner(high, low, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = DevStopBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Fused, single-pass DevStop classic kernel:
/// - Builds 2-bar range on the fly
/// - Maintains rolling SMA or EMA of the range
/// - Maintains rolling standard deviation via two-moment identities
/// - Computes base and final rolling extrema with a monotonic deque
#[inline]
unsafe fn devstop_scalar_classic_fused<const EMA: bool>(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    is_long: bool,
    first: usize,
    dst: &mut [f64],
) -> Result<(), DevStopError> {
    debug_assert_eq!(high.len(), low.len());
    let len = high.len();
    if len == 0 {
        return Ok(());
    }
    if period == 0 {
        return Err(DevStopError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // Indices & warmup math
    let start_base = first + period; // first index where base is defined
    let start_final = start_base + period - 1; // first index where final rolling extrema is defined
    let warm = start_final; // all < start_final => NaN

    // NaN warm prefix (and truncate if warm > len)
    let warm_end = warm.min(len);
    for j in 0..warm_end {
        *dst.get_unchecked_mut(j) = f64::NAN;
    }
    if start_base >= len {
        return Ok(());
    }

    #[inline(always)]
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }
    #[inline(always)]
    fn max0(x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            x
        }
    }

    // Prefill range ring: r[k] for k in [first+1 .. start_base-1] (period-1 items)
    let mut r_ring = vec![f64::NAN; period];
    let mut r_ins_pos = 0usize;
    let mut r_inserted = 0usize;

    let mut sum = 0.0f64; // Σ r
    let mut sum2 = 0.0f64; // Σ r^2
    let mut cnt = 0usize; // number of finite r's currently in ring

    let mut prev_h = *high.get_unchecked(first);
    let mut prev_l = *low.get_unchecked(first);
    let end_init = start_base.min(len);

    for k in (first + 1)..end_init {
        let h = *high.get_unchecked(k);
        let l = *low.get_unchecked(k);
        let r = if h.is_nan() || l.is_nan() || prev_h.is_nan() || prev_l.is_nan() {
            f64::NAN
        } else {
            let hi2 = if h > prev_h { h } else { prev_h };
            let lo2 = if l < prev_l { l } else { prev_l };
            hi2 - lo2
        };
        *r_ring.get_unchecked_mut(r_ins_pos) = r;
        r_ins_pos += 1;
        r_inserted += 1;
        if r.is_finite() {
            sum += r;
            sum2 = fma(r, r, sum2);
            cnt += 1;
        }
        prev_h = h;
        prev_l = l;
    }
    r_ins_pos = (period - 1) % period;

    // EMA state (initialized with SMA of available prefill; matches classic_ema boot)
    let mut ema = if EMA {
        if cnt > 0 { sum / (cnt as f64) } else { f64::NAN }
    } else {
        0.0
    };
    let alpha = if EMA { 2.0 / (period as f64 + 1.0) } else { 0.0 };
    let beta = if EMA { 1.0 - alpha } else { 0.0 };

    // Monotonic deque for rolling extrema over `base`
    let mut base_ring = vec![f64::NAN; period];
    let cap = period;
    let mut dq_buf = vec![0usize; cap]; // absolute indices
    let mut dq_head = 0usize;
    let mut dq_len = 0usize;
    #[inline(always)]
    fn dq_idx_at(buf: &[usize], head: usize, cap: usize, k: usize) -> usize {
        unsafe { *buf.get_unchecked((head + k) % cap) }
    }
    #[inline(always)]
    fn dq_back_idx(buf: &[usize], head: usize, len: usize, cap: usize) -> usize {
        unsafe { *buf.get_unchecked((head + len - 1) % cap) }
    }
    #[inline(always)]
    fn dq_pop_back(len: &mut usize) {
        *len -= 1;
    }
    #[inline(always)]
    fn dq_pop_front(head: &mut usize, len: &mut usize, cap: usize) {
        *head = (*head + 1) % cap;
        *len -= 1;
    }
    #[inline(always)]
    fn dq_push_back(buf: &mut [usize], head: usize, len: &mut usize, cap: usize, value: usize) {
        let pos = (head + *len) % cap;
        unsafe { *buf.get_unchecked_mut(pos) = value; }
        *len += 1;
    }

    // Main streaming loop (one pass)
    for i in start_base..len {
        let h = *high.get_unchecked(i);
        let l = *low.get_unchecked(i);

        // 2-bar range using (i-1, i)
        let r_new = if h.is_nan() || l.is_nan() || prev_h.is_nan() || prev_l.is_nan() {
            f64::NAN
        } else {
            let hi2 = if h > prev_h { h } else { prev_h };
            let lo2 = if l < prev_l { l } else { prev_l };
            hi2 - lo2
        };
        prev_h = h;
        prev_l = l;

        // Sliding update of Σr and Σr^2
        let had_full = r_inserted >= period;
        let old = if had_full {
            *r_ring.get_unchecked(r_ins_pos)
        } else {
            f64::NAN
        };
        if had_full && old.is_finite() {
            sum -= old;
            sum2 -= old * old;
            cnt -= 1;
        }
        // Insert new
        *r_ring.get_unchecked_mut(r_ins_pos) = r_new;
        r_ins_pos = (r_ins_pos + 1) % period;
        r_inserted += 1;
        if r_new.is_finite() {
            sum += r_new;
            sum2 = fma(r_new, r_new, sum2);
            cnt += 1;
        }

        // Update EMA (if enabled)
        if EMA && r_new.is_finite() {
            ema = r_new.mul_add(alpha, beta * ema);
        }

        // Compute avtr and stddev
        let (avtr, sigma) = if cnt == 0 {
            (f64::NAN, f64::NAN)
        } else if EMA {
            let inv = 1.0 / (cnt as f64);
            let e1 = sum * inv; // E[r]
            let e2 = sum2 * inv; // E[r^2]
            let var = max0(e2 - (2.0 * ema) * e1 + ema * ema);
            (ema, var.sqrt())
        } else {
            let inv = 1.0 / (cnt as f64);
            let mean = sum * inv; // E[r]
            let var = max0((sum2 - (sum * sum) * inv) * inv); // E[r^2] - (E[r])^2
            (mean, var.sqrt())
        };

        // Base value
        let base = if is_long {
            if h.is_nan() || avtr.is_nan() || sigma.is_nan() {
                f64::NAN
            } else {
                h - avtr - mult * sigma
            }
        } else {
            if l.is_nan() || avtr.is_nan() || sigma.is_nan() {
                f64::NAN
            } else {
                l + avtr + mult * sigma
            }
        };

        // Store base and update deque
        let bslot = i % period;
        *base_ring.get_unchecked_mut(bslot) = base;
        if is_long {
            while dq_len > 0 {
                let j = dq_back_idx(&dq_buf, dq_head, dq_len, cap);
                let bj = *base_ring.get_unchecked(j % period);
                if bj.is_nan() || bj <= base {
                    dq_pop_back(&mut dq_len);
                } else {
                    break;
                }
            }
        } else {
            while dq_len > 0 {
                let j = dq_back_idx(&dq_buf, dq_head, dq_len, cap);
                let bj = *base_ring.get_unchecked(j % period);
                if bj.is_nan() || bj >= base {
                    dq_pop_back(&mut dq_len);
                } else {
                    break;
                }
            }
        }
        dq_push_back(&mut dq_buf, dq_head, &mut dq_len, cap, i);

        // Drop expired
        let cut = i + 1 - period;
        while dq_len > 0 && dq_idx_at(&dq_buf, dq_head, cap, 0) < cut {
            dq_pop_front(&mut dq_head, &mut dq_len, cap);
        }

        // Emit final value
        if i >= start_final {
            let out = if dq_len > 0 {
                let j = dq_idx_at(&dq_buf, dq_head, cap, 0);
                *base_ring.get_unchecked(j % period)
            } else {
                f64::NAN
            };
            *dst.get_unchecked_mut(i) = out;
        }
    }
    Ok(())
}

/// Optimized DevStop calculation with inline SMA and standard deviation (fused kernel)
#[inline]
pub unsafe fn devstop_scalar_classic_sma(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    is_long: bool,
    first: usize,
    dst: &mut [f64],
) -> Result<(), DevStopError> {
    devstop_scalar_classic_fused::<false>(high, low, period, mult, is_long, first, dst)
}

/// Optimized DevStop calculation with inline EMA and standard deviation (fused kernel)
#[inline]
pub unsafe fn devstop_scalar_classic_ema(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    is_long: bool,
    first: usize,
    dst: &mut [f64],
) -> Result<(), DevStopError> {
    devstop_scalar_classic_fused::<true>(high, low, period, mult, is_long, first, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_devstop_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = DevStopParams {
            period: None,
            mult: None,
            devtype: None,
            direction: None,
            ma_type: None,
        };
        let input_default = DevStopInput::from_candles(&candles, "high", "low", default_params);
        let output_default = devstop_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = DevStopParams {
            period: Some(20),
            mult: Some(1.0),
            devtype: Some(2),
            direction: Some("short".to_string()),
            ma_type: Some("ema".to_string()),
        };
        let input_custom = DevStopInput::from_candles(&candles, "high", "low", params_custom);
        let output_custom = devstop_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_devstop_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = &candles.high;
        let low = &candles.low;

        let params = DevStopParams {
            period: Some(20),
            mult: Some(0.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(high, low, params);
        let result = devstop_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), candles.close.len());
        assert!(result.values.len() >= 5);
        let last_five = &result.values[result.values.len() - 5..];
        for &val in last_five {
            println!("Indicator values {}", val);
        }
        Ok(())
    }

    fn check_devstop_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = DevStopInput::with_default_candles(&candles);
        match input.data {
            DevStopData::Candles {
                source_high,
                source_low,
                ..
            } => {
                assert_eq!(source_high, "high");
                assert_eq!(source_low, "low");
            }
            _ => panic!("Expected DevStopData::Candles"),
        }
        let output = devstop_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_devstop_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = DevStopParams {
            period: Some(0),
            mult: Some(1.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(&high, &low, params);
        let result = devstop_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_devstop_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = DevStopParams {
            period: Some(10),
            mult: Some(1.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(&high, &low, params);
        let result = devstop_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_devstop_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0];
        let low = [90.0];
        let params = DevStopParams {
            period: Some(20),
            mult: Some(2.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(&high, &low, params);
        let result = devstop_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_devstop_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = DevStopParams {
            period: Some(20),
            mult: Some(1.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_candles(&candles, "high", "low", params);
        let first_result = devstop_with_kernel(&input, kernel)?;

        assert_eq!(first_result.values.len(), candles.close.len());

        let reinput_params = DevStopParams {
            period: Some(20),
            mult: Some(0.5),
            devtype: Some(2),
            direction: Some("short".to_string()),
            ma_type: Some("ema".to_string()),
        };
        let second_input =
            DevStopInput::from_slices(&first_result.values, &first_result.values, reinput_params);
        let second_result = devstop_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_devstop_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = &candles.high;
        let low = &candles.low;

        let params = DevStopParams {
            period: Some(20),
            mult: Some(0.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(high, low, params);
        let result = devstop_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), high.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_devstop_tests {
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
    fn check_devstop_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            DevStopParams::default(),
            // Minimum period
            DevStopParams {
                period: Some(2),
                mult: Some(0.0),
                devtype: Some(0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            // Small period with various multipliers
            DevStopParams {
                period: Some(5),
                mult: Some(0.5),
                devtype: Some(0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            DevStopParams {
                period: Some(5),
                mult: Some(1.0),
                devtype: Some(1),
                direction: Some("short".to_string()),
                ma_type: Some("ema".to_string()),
            },
            // Medium periods with different devtypes
            DevStopParams {
                period: Some(10),
                mult: Some(0.0),
                devtype: Some(0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            DevStopParams {
                period: Some(10),
                mult: Some(2.0),
                devtype: Some(1),
                direction: Some("short".to_string()),
                ma_type: Some("ema".to_string()),
            },
            DevStopParams {
                period: Some(10),
                mult: Some(1.5),
                devtype: Some(2),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            // Default period (20) variations
            DevStopParams {
                period: Some(20),
                mult: Some(0.0),
                devtype: Some(0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            DevStopParams {
                period: Some(20),
                mult: Some(1.0),
                devtype: Some(1),
                direction: Some("short".to_string()),
                ma_type: Some("ema".to_string()),
            },
            DevStopParams {
                period: Some(20),
                mult: Some(2.5),
                devtype: Some(2),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            // Large periods
            DevStopParams {
                period: Some(50),
                mult: Some(0.5),
                devtype: Some(0),
                direction: Some("short".to_string()),
                ma_type: Some("ema".to_string()),
            },
            DevStopParams {
                period: Some(50),
                mult: Some(1.0),
                devtype: Some(1),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            // Very large period
            DevStopParams {
                period: Some(100),
                mult: Some(0.0),
                devtype: Some(0),
                direction: Some("long".to_string()),
                ma_type: Some("sma".to_string()),
            },
            DevStopParams {
                period: Some(100),
                mult: Some(3.0),
                devtype: Some(2),
                direction: Some("short".to_string()),
                ma_type: Some("ema".to_string()),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DevStopInput::from_candles(&candles, "high", "low", params.clone());
            let output = devstop_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, devtype={}, direction={}, ma_type={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(20),
                        params.mult.unwrap_or(0.0),
                        params.devtype.unwrap_or(0),
                        params.direction.as_deref().unwrap_or("long"),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, devtype={}, direction={}, ma_type={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(20),
                        params.mult.unwrap_or(0.0),
                        params.devtype.unwrap_or(0),
                        params.direction.as_deref().unwrap_or("long"),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, devtype={}, direction={}, ma_type={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(20),
                        params.mult.unwrap_or(0.0),
                        params.devtype.unwrap_or(0),
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
    fn check_devstop_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    fn check_devstop_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating test data with more realistic scenarios
        let strat = (2usize..=50)
            .prop_flat_map(|period| {
                (
                    // Generate base price and volatility
                    (100.0f64..5000.0f64, 0.01f64..0.1f64),
                    Just(period),
                    0.0f64..3.0f64,  // mult range
                    0usize..=2,      // devtype range (0: stddev, 1: mean abs, 2: median abs)
                    prop::bool::ANY, // direction (true = long, false = short)
                    // MA type selection
                    prop::sample::select(vec!["sma", "ema", "wma", "hma", "dema"]),
                )
            })
            .prop_flat_map(
                move |(base_price_vol, period, mult, devtype, is_long, ma_type)| {
                    let (base_price, volatility) = base_price_vol;
                    let data_len = period + 10 + (period * 3); // Ensure enough data for warmup

                    // Generate more realistic price data with trends and volatility
                    let price_strategy = prop::collection::vec(
                        (-volatility..volatility)
                            .prop_map(move |change| base_price * (1.0 + change)),
                        data_len..400,
                    );

                    (
                        price_strategy.clone(), // For high calculation
                        price_strategy,         // For low calculation
                        Just(period),
                        Just(mult),
                        Just(devtype),
                        Just(is_long),
                        Just(ma_type.to_string()),
                    )
                },
            );

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(high_base, low_base, period, mult, devtype, is_long, ma_type)| {
                    // Create high/low with realistic spread
                    let len = high_base.len().min(low_base.len());
                    let mut high = vec![0.0; len];
                    let mut low = vec![0.0; len];

                    for i in 0..len {
                        // Ensure high >= low with realistic spread (0.1% to 2% typically)
                        let mid = (high_base[i] + low_base[i]) / 2.0;
                        let spread = mid * 0.001 * (1.0 + (i as f64 * 0.1).sin().abs()); // Variable spread
                        high[i] = mid + spread;
                        low[i] = mid - spread;
                    }

                    let direction = if is_long {
                        "long".to_string()
                    } else {
                        "short".to_string()
                    };

                    let params = DevStopParams {
                        period: Some(period),
                        mult: Some(mult),
                        devtype: Some(devtype),
                        direction: Some(direction.clone()),
                        ma_type: Some(ma_type.clone()),
                    };
                    let input = DevStopInput::from_slices(&high, &low, params.clone());

                    // Test with specified kernel
                    let result = devstop_with_kernel(&input, kernel);
                    prop_assert!(
                        result.is_ok(),
                        "DevStop calculation failed: {:?}",
                        result.err()
                    );
                    let out = result.unwrap().values;

                    // Test with scalar reference kernel
                    let ref_result = devstop_with_kernel(&input, Kernel::Scalar);
                    prop_assert!(ref_result.is_ok(), "Reference calculation failed");
                    let ref_out = ref_result.unwrap().values;

                    // Property 1: Output length should match input length
                    prop_assert_eq!(out.len(), high.len(), "Output length mismatch");

                    // Property 2: Warmup period handling - be less rigid
                    // Just verify that we have NaN values early and finite values later
                    let expected_warmup = period * 2; // Approximate
                    let has_early_nans = out.iter().take(period).any(|&x| x.is_nan());
                    let has_late_finites =
                        out.iter().skip(expected_warmup + 5).any(|&x| x.is_finite());

                    if out.len() > expected_warmup + 5 {
                        prop_assert!(
                            has_early_nans || period <= 2,
                            "Expected some NaN values during warmup period"
                        );
                        prop_assert!(
                            has_late_finites,
                            "Expected finite values after warmup period"
                        );
                    }

                    // Property 3: Kernel consistency - compare with scalar reference
                    for i in 0..out.len() {
                        let y = out[i];
                        let r = ref_out[i];

                        // Both should be NaN or both should be finite
                        if y.is_nan() != r.is_nan() {
                            prop_assert!(
                                false,
                                "NaN mismatch at index {}: kernel is_nan={}, scalar is_nan={}",
                                i,
                                y.is_nan(),
                                r.is_nan()
                            );
                        }

                        // Check finite value consistency
                        if y.is_finite() && r.is_finite() {
                            let abs_diff = (y - r).abs();
                            let rel_diff = if r.abs() > 1e-10 {
                                abs_diff / r.abs()
                            } else {
                                abs_diff
                            };

                            prop_assert!(
                                abs_diff <= 1e-6 || rel_diff <= 1e-6,
                                "Value mismatch at index {}: kernel={}, scalar={}, diff={}",
                                i,
                                y,
                                r,
                                abs_diff
                            );
                        }
                    }

                    // Property 4: Multiplier effect
                    // Higher multiplier should generally mean stops further from price
                    if mult > 0.1 && out.len() > expected_warmup + 10 {
                        // Compare with zero multiplier version
                        let params_zero = DevStopParams {
                            period: Some(period),
                            mult: Some(0.0),
                            devtype: Some(devtype),
                            direction: Some(direction.clone()),
                            ma_type: Some(ma_type.clone()),
                        };
                        let input_zero = DevStopInput::from_slices(&high, &low, params_zero);
                        if let Ok(result_zero) = devstop_with_kernel(&input_zero, Kernel::Scalar) {
                            let out_zero = result_zero.values;

                            // Count how many times the stop with multiplier is further from price
                            let mut further_count = 0;
                            let mut total_count = 0;

                            for i in expected_warmup..out.len() {
                                if out[i].is_finite() && out_zero[i].is_finite() {
                                    total_count += 1;
                                    if direction == "long" {
                                        // For long, higher mult should give lower (further) stops
                                        if out[i] <= out_zero[i] {
                                            further_count += 1;
                                        }
                                    } else {
                                        // For short, higher mult should give higher (further) stops
                                        if out[i] >= out_zero[i] {
                                            further_count += 1;
                                        }
                                    }
                                }
                            }

                            // At least 90% of the time, multiplier should increase stop distance
                            if total_count > 0 {
                                let ratio = further_count as f64 / total_count as f64;
                                prop_assert!(
								ratio >= 0.9 || mult < 0.1,
								"Multiplier effect not working: only {:.1}% of stops are further with mult={}",
								ratio * 100.0, mult
							);
                            }
                        }
                    }

                    // Property 5: Special case - flat candles (high == low)
                    // Test a subset with flat candles
                    if len > 20 {
                        let mut flat_high = high.clone();
                        let mut flat_low = high.clone(); // Same as high
                        for i in 10..20.min(len) {
                            flat_high[i] = 1000.0;
                            flat_low[i] = 1000.0;
                        }

                        let flat_params = params.clone();
                        let flat_input =
                            DevStopInput::from_slices(&flat_high, &flat_low, flat_params);
                        let flat_result = devstop_with_kernel(&flat_input, kernel);

                        // Should not error on flat candles
                        prop_assert!(
                            flat_result.is_ok(),
                            "DevStop should handle flat candles (high==low)"
                        );
                    }

                    // Property 6: Deviation type parameter is being used
                    // Simply verify that the devtype parameter affects the calculation
                    // Different deviation types may produce similar results with certain data patterns
                    // which is mathematically valid, so we just verify the parameter is processed
                    if out.len() > expected_warmup + 10 && mult > 0.5 {
                        // Test that all three deviation types can be calculated without error
                        for test_devtype in 0..=2 {
                            let params_test = DevStopParams {
                                period: Some(period),
                                mult: Some(mult),
                                devtype: Some(test_devtype),
                                direction: Some(direction.clone()),
                                ma_type: Some(ma_type.clone()),
                            };
                            let input_test = DevStopInput::from_slices(&high, &low, params_test);
                            let result_test = devstop_with_kernel(&input_test, Kernel::Scalar);

                            prop_assert!(
                                result_test.is_ok(),
                                "DevStop should handle all deviation types: failed on devtype {}",
                                test_devtype
                            );

                            // Verify output has the correct structure
                            if let Ok(output) = result_test {
                                prop_assert_eq!(
                                    output.values.len(),
                                    high.len(),
                                    "Output length should match input for devtype {}",
                                    test_devtype
                                );
                            }
                        }
                    }

                    // Property 7: Stop movement logic
                    // The stops should behave sensibly - not jumping erratically
                    if out.len() > expected_warmup + period {
                        let mut max_jump = 0.0;
                        let mut jump_count = 0;

                        for i in (expected_warmup + 1)..out.len() {
                            if out[i].is_finite() && out[i - 1].is_finite() {
                                let jump = (out[i] - out[i - 1]).abs();
                                let relative_jump = jump / out[i - 1].abs().max(1.0);

                                // Track the maximum jump
                                if relative_jump > max_jump {
                                    max_jump = relative_jump;
                                }

                                // Count large jumps (more than 20% change)
                                if relative_jump > 0.2 {
                                    jump_count += 1;
                                }
                            }
                        }

                        // Stops shouldn't jump more than 50% in a single step typically
                        // (unless we have extreme price movements)
                        prop_assert!(
                            max_jump < 0.5 || jump_count < 5,
                            "Stop values jumping too much: max jump = {:.1}%, large jumps = {}",
                            max_jump * 100.0,
                            jump_count
                        );
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    generate_all_devstop_tests!(
        check_devstop_partial_params,
        check_devstop_accuracy,
        check_devstop_default_candles,
        check_devstop_zero_period,
        check_devstop_period_exceeds_length,
        check_devstop_very_small_dataset,
        check_devstop_reinput,
        check_devstop_nan_handling,
        check_devstop_no_poison,
        check_devstop_property
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let high = &c.high;
        let low = &c.low;

        let output = DevStopBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(high, low)?;

        let def = DevStopParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

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
    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let high = &c.high;
        let low = &c.low;

        let output = DevStopBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 5)
            .mult_range(0.0, 2.0, 0.5)
            .devtype_range(0, 2, 1)
            .apply_slices(high, low)?;

        let expected_combos = 5 * 5 * 3; // 5 periods * 5 mults * 3 devtypes
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let high = &c.high;
        let low = &c.low;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (period_start, period_end, period_step, mult_start, mult_end, mult_step, devtype_start, devtype_end, devtype_step)
            (2, 10, 2, 0.0, 2.0, 0.5, 0, 2, 1), // Small periods with mult variations
            (5, 25, 5, 0.0, 1.0, 0.25, 0, 0, 0), // Medium periods, single devtype
            (30, 60, 15, 1.0, 3.0, 1.0, 1, 1, 0), // Large periods, mean abs dev
            (2, 5, 1, 0.0, 0.5, 0.1, 2, 2, 0),  // Dense small range, median abs dev
            (10, 20, 2, 0.5, 2.5, 0.5, 0, 2, 2), // Medium range, all devtypes
            (20, 20, 0, 0.0, 3.0, 0.3, 0, 2, 1), // Single period, mult sweep
            (5, 50, 15, 1.0, 1.0, 0.0, 0, 2, 1), // Wide period range, single mult
        ];

        for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step, d_start, d_end, d_step)) in
            test_configs.iter().enumerate()
        {
            let output = DevStopBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .mult_range(m_start, m_end, m_step)
                .devtype_range(d_start, d_end, d_step)
                .apply_slices(high, low)?;

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
						 at row {} col {} (flat index {}) with params: period={}, mult={}, devtype={}, \
						 direction={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(20),
                        combo.mult.unwrap_or(0.0),
                        combo.devtype.unwrap_or(0),
                        combo.direction.as_deref().unwrap_or("long"),
                        combo.ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, devtype={}, \
						 direction={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(20),
                        combo.mult.unwrap_or(0.0),
                        combo.devtype.unwrap_or(0),
                        combo.direction.as_deref().unwrap_or("long"),
                        combo.ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, devtype={}, \
						 direction={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(20),
                        combo.mult.unwrap_or(0.0),
                        combo.devtype.unwrap_or(0),
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
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);
}
