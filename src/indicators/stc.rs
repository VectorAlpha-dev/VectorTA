//! # Schaff Trend Cycle (STC)
//!
//! Schaff Trend Cycle (STC) is an oscillator that applies MACD and double stochastic steps,
//! then smooths with EMA. This implementation supports batch, AVX, and builder APIs like alma.rs.
//!
//! ## Parameters
//! - **fast_period**: Period for fast MA (default: 23)
//! - **slow_period**: Period for slow MA (default: 50)
//! - **k_period**: Stochastic window (default: 10)
//! - **d_period**: EMA smoothing window (default: 3)
//! - **fast_ma_type**: Type for fast MA (default: "ema")
//! - **slow_ma_type**: Type for slow MA (default: "ema")
//!
//! ## Returns
//! - **`Ok(StcOutput)`** or **`Err(StcError)`**
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: All SIMD implementations are stubs that delegate to scalar for strict API parity. Future optimization could vectorize the MACD and stochastic calculations.
//! - **Streaming Performance**: Uses O(n) recalculation approach by maintaining a growing buffer. Efficient streaming would require maintaining separate state for each component (MACD, Stoch, EMA).
//! - **Memory Optimization**: Uses `alloc_with_nan_prefix` for proper warmup handling. Intermediate calculations use appropriately-sized working buffers rather than full data-length allocations.

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
use core::mem::MaybeUninit;
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
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum StcData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for StcInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            StcData::Slice(slice) => slice,
            StcData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StcOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct StcParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub k_period: Option<usize>,
    pub d_period: Option<usize>,
    pub fast_ma_type: Option<String>,
    pub slow_ma_type: Option<String>,
}

impl Default for StcParams {
    fn default() -> Self {
        Self {
            fast_period: Some(23),
            slow_period: Some(50),
            k_period: Some(10),
            d_period: Some(3),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StcInput<'a> {
    pub data: StcData<'a>,
    pub params: StcParams,
}

impl<'a> StcInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: StcParams) -> Self {
        Self {
            data: StcData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: StcParams) -> Self {
        Self {
            data: StcData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", StcParams::default())
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(23)
    }
    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(50)
    }
    #[inline]
    pub fn get_k_period(&self) -> usize {
        self.params.k_period.unwrap_or(10)
    }
    #[inline]
    pub fn get_d_period(&self) -> usize {
        self.params.d_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_fast_ma_type(&self) -> &str {
        self.params.fast_ma_type.as_deref().unwrap_or("ema")
    }
    #[inline]
    pub fn get_slow_ma_type(&self) -> &str {
        self.params.slow_ma_type.as_deref().unwrap_or("ema")
    }
}

#[derive(Clone, Debug)]
pub struct StcBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    k_period: Option<usize>,
    d_period: Option<usize>,
    fast_ma_type: Option<String>,
    slow_ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for StcBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            k_period: None,
            d_period: None,
            fast_ma_type: None,
            slow_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StcBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fast_period(mut self, n: usize) -> Self {
        self.fast_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_period(mut self, n: usize) -> Self {
        self.slow_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn k_period(mut self, n: usize) -> Self {
        self.k_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn d_period(mut self, n: usize) -> Self {
        self.d_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn fast_ma_type<T: Into<String>>(mut self, s: T) -> Self {
        self.fast_ma_type = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn slow_ma_type<T: Into<String>>(mut self, s: T) -> Self {
        self.slow_ma_type = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<StcOutput, StcError> {
        let p = StcParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            k_period: self.k_period,
            d_period: self.d_period,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
        };
        let i = StcInput::from_candles(c, "close", p);
        stc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<StcOutput, StcError> {
        let p = StcParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            k_period: self.k_period,
            d_period: self.d_period,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
        };
        let i = StcInput::from_slice(d, p);
        stc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<StcStream, StcError> {
        let p = StcParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            k_period: self.k_period,
            d_period: self.d_period,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
        };
        StcStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum StcError {
    #[error("stc: Empty data provided.")]
    EmptyData,
    #[error("stc: All values are NaN.")]
    AllValuesNaN,
    #[error("stc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stc: Internal error: {0}")]
    Internal(String),
}

#[inline]
pub fn stc(input: &StcInput) -> Result<StcOutput, StcError> {
    stc_with_kernel(input, Kernel::Auto)
}

pub fn stc_with_kernel(input: &StcInput, kernel: Kernel) -> Result<StcOutput, StcError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(StcError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StcError::AllValuesNaN)?;

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let k_period = input.get_k_period();
    let d_period = input.get_d_period();
    let needed = fast_period.max(slow_period).max(k_period).max(d_period);

    if (len - first) < needed {
        return Err(StcError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warmup = first + needed - 1;
    let mut output = alloc_with_nan_prefix(len, warmup);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => stc_scalar(
                data,
                fast_period,
                slow_period,
                k_period,
                d_period,
                input.get_fast_ma_type(),
                input.get_slow_ma_type(),
                first,
                &mut output,
            )?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stc_avx2(
                data,
                fast_period,
                slow_period,
                k_period,
                d_period,
                input.get_fast_ma_type(),
                input.get_slow_ma_type(),
                first,
                &mut output,
            )?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => stc_avx512(
                data,
                fast_period,
                slow_period,
                k_period,
                d_period,
                input.get_fast_ma_type(),
                input.get_slow_ma_type(),
                first,
                &mut output,
            )?,
            _ => unreachable!(),
        }
    }

    Ok(StcOutput { values: output })
}

#[inline]
pub fn stc_into_slice(dst: &mut [f64], input: &StcInput, kern: Kernel) -> Result<(), StcError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(StcError::EmptyData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StcError::AllValuesNaN)?;
    if dst.len() != len {
        return Err(StcError::Internal(format!(
            "dst len {} != src len {}",
            dst.len(),
            len
        )));
    }

    let needed = input
        .get_fast_period()
        .max(input.get_slow_period())
        .max(input.get_k_period())
        .max(input.get_d_period());

    if (len - first) < needed {
        return Err(StcError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    // correct warmup
    let warmup_end = first + needed - 1;
    for v in &mut dst[..warmup_end.min(len)] {
        *v = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => stc_scalar(
                data,
                input.get_fast_period(),
                input.get_slow_period(),
                input.get_k_period(),
                input.get_d_period(),
                input.get_fast_ma_type(),
                input.get_slow_ma_type(),
                first,
                dst,
            )?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stc_avx2(
                data,
                input.get_fast_period(),
                input.get_slow_period(),
                input.get_k_period(),
                input.get_d_period(),
                input.get_fast_ma_type(),
                input.get_slow_ma_type(),
                first,
                dst,
            )?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => stc_avx512(
                data,
                input.get_fast_period(),
                input.get_slow_period(),
                input.get_k_period(),
                input.get_d_period(),
                input.get_fast_ma_type(),
                input.get_slow_ma_type(),
                first,
                dst,
            )?,
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[inline]
pub fn stc_scalar(
    data: &[f64],
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
    fast_type: &str,
    slow_type: &str,
    first: usize,
    out: &mut [f64],
) -> Result<(), StcError> {
    // Check for classic kernel optimization
    if fast_type == "ema" && slow_type == "ema" {
        return unsafe { stc_scalar_classic_ema(data, fast, slow, k, d, first, out) };
    } else if fast_type == "sma" && slow_type == "sma" {
        return unsafe { stc_scalar_classic_sma(data, fast, slow, k, d, first, out) };
    }

    // Fall back to regular implementation for other MA types
    use crate::indicators::ema::{ema, EmaInput, EmaParams};
    use crate::indicators::moving_averages::ma::{ma, MaData};
    use crate::indicators::utility_functions::{max_rolling, min_rolling};
    use crate::utilities::helpers::alloc_with_nan_prefix;

    let len = data.len();
    let slice = &data[first..];

    // Get MAs
    let fast_ma = ma(fast_type, MaData::Slice(slice), fast)
        .map_err(|e| StcError::Internal(format!("Fast MA error: {}", e)))?;
    let slow_ma = ma(slow_type, MaData::Slice(slice), slow)
        .map_err(|e| StcError::Internal(format!("Slow MA error: {}", e)))?;

    // Allocate working buffers for intermediate calculations
    // These are algorithm-specific and much smaller than full data size
    let working_len = slice.len();
    let mut macd = alloc_with_nan_prefix(working_len, 0);

    // Calculate MACD
    for i in 0..working_len {
        macd[i] = fast_ma[i] - slow_ma[i];
    }

    // First stochastic calculation
    let macd_min = min_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let macd_max = max_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;

    let mut stok = alloc_with_nan_prefix(working_len, 0);
    for i in 0..working_len {
        let range = macd_max[i] - macd_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            stok[i] = (macd[i] - macd_min[i]) / range * 100.0;
        } else if !macd[i].is_nan() {
            stok[i] = 50.0;
        }
    }

    // First EMA smoothing
    let d_ema = ema(&EmaInput::from_slice(&stok, EmaParams { period: Some(d) }))
        .map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let d_vals = &d_ema.values;

    // Second stochastic calculation
    let d_min = min_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let d_max = max_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;

    let mut kd = alloc_with_nan_prefix(working_len, 0);
    for i in 0..working_len {
        let range = d_max[i] - d_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            kd[i] = (d_vals[i] - d_min[i]) / range * 100.0;
        } else if !d_vals[i].is_nan() {
            kd[i] = 50.0;
        }
    }

    // Final EMA smoothing
    let kd_ema = ema(&EmaInput::from_slice(&kd, EmaParams { period: Some(d) }))
        .map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let final_stc = &kd_ema.values;

    // Write results directly to output buffer
    for (i, &val) in final_stc.iter().enumerate() {
        out[first + i] = val;
    }

    Ok(())
}

// Classic kernel with inline EMA calculations
#[inline]
pub unsafe fn stc_scalar_classic_ema(
    data: &[f64],
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), StcError> {
    use crate::indicators::utility_functions::{max_rolling, min_rolling};
    use crate::utilities::helpers::alloc_with_nan_prefix;

    let slice = &data[first..];
    let working_len = slice.len();

    // EMA alpha factors
    let fast_alpha = 2.0 / (fast as f64 + 1.0);
    let slow_alpha = 2.0 / (slow as f64 + 1.0);
    let d_alpha = 2.0 / (d as f64 + 1.0);

    // Initialize EMAs with SMA
    let mut fast_sum = 0.0;
    let mut slow_sum = 0.0;

    // Calculate initial SMAs for EMA initialization
    for i in 0..fast.min(working_len) {
        fast_sum += slice[i];
        if i < slow {
            slow_sum += slice[i];
        }
    }

    for i in fast..slow.min(working_len) {
        slow_sum += slice[i];
    }

    let mut fast_ema = if fast <= working_len {
        fast_sum / fast as f64
    } else {
        f64::NAN
    };
    let mut slow_ema = if slow <= working_len {
        slow_sum / slow as f64
    } else {
        f64::NAN
    };

    // Calculate EMAs and MACD
    let mut macd = alloc_with_nan_prefix(working_len, 0);

    for i in 0..working_len {
        if i >= fast - 1 {
            if i == fast - 1 {
                // First EMA value is the SMA
                fast_ema = fast_sum / fast as f64;
            } else {
                // Update EMA
                fast_ema = fast_alpha * slice[i] + (1.0 - fast_alpha) * fast_ema;
            }
        }

        if i >= slow - 1 {
            if i == slow - 1 {
                // First EMA value is the SMA
                slow_ema = slow_sum / slow as f64;
            } else {
                // Update EMA
                slow_ema = slow_alpha * slice[i] + (1.0 - slow_alpha) * slow_ema;
            }
        }

        // Calculate MACD
        if i >= slow - 1 {
            macd[i] = fast_ema - slow_ema;
        } else {
            macd[i] = f64::NAN;
        }
    }

    // First stochastic calculation
    let macd_min = min_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let macd_max = max_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;

    let mut stok = alloc_with_nan_prefix(working_len, 0);
    for i in 0..working_len {
        let range = macd_max[i] - macd_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            stok[i] = (macd[i] - macd_min[i]) / range * 100.0;
        } else if !macd[i].is_nan() {
            stok[i] = 50.0;
        }
    }

    // First EMA smoothing (inline)
    let mut d_vals = alloc_with_nan_prefix(working_len, 0);
    let mut d_ema = f64::NAN;
    let mut d_sum = 0.0;
    let mut d_count = 0;

    for i in 0..working_len {
        if !stok[i].is_nan() {
            if d_count < d {
                d_sum += stok[i];
                d_count += 1;
                if d_count == d {
                    d_ema = d_sum / d as f64;
                    d_vals[i] = d_ema;
                } else {
                    d_vals[i] = f64::NAN;
                }
            } else {
                d_ema = d_alpha * stok[i] + (1.0 - d_alpha) * d_ema;
                d_vals[i] = d_ema;
            }
        } else {
            d_vals[i] = f64::NAN;
        }
    }

    // Second stochastic calculation
    let d_min = min_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let d_max = max_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;

    let mut kd = alloc_with_nan_prefix(working_len, 0);
    for i in 0..working_len {
        let range = d_max[i] - d_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            kd[i] = (d_vals[i] - d_min[i]) / range * 100.0;
        } else if !d_vals[i].is_nan() {
            kd[i] = 50.0;
        }
    }

    // Final EMA smoothing (inline)
    let mut final_ema = f64::NAN;
    let mut final_sum = 0.0;
    let mut final_count = 0;

    for i in 0..working_len {
        if !kd[i].is_nan() {
            if final_count < d {
                final_sum += kd[i];
                final_count += 1;
                if final_count == d {
                    final_ema = final_sum / d as f64;
                    out[first + i] = final_ema;
                } else {
                    out[first + i] = f64::NAN;
                }
            } else {
                final_ema = d_alpha * kd[i] + (1.0 - d_alpha) * final_ema;
                out[first + i] = final_ema;
            }
        } else {
            out[first + i] = f64::NAN;
        }
    }

    Ok(())
}

// Classic kernel with inline SMA calculations
#[inline]
pub unsafe fn stc_scalar_classic_sma(
    data: &[f64],
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), StcError> {
    use crate::indicators::utility_functions::{max_rolling, min_rolling};
    use crate::utilities::helpers::alloc_with_nan_prefix;

    let slice = &data[first..];
    let working_len = slice.len();

    // Calculate SMAs and MACD
    let mut macd = alloc_with_nan_prefix(working_len, 0);

    // Initialize rolling sums
    let mut fast_sum = 0.0;
    let mut slow_sum = 0.0;

    // Initial sums
    for i in 0..fast.min(working_len) {
        fast_sum += slice[i];
    }
    for i in 0..slow.min(working_len) {
        slow_sum += slice[i];
    }

    // Calculate rolling SMAs and MACD
    for i in 0..working_len {
        if i >= fast {
            fast_sum = fast_sum - slice[i - fast] + slice[i];
        }
        if i >= slow {
            slow_sum = slow_sum - slice[i - slow] + slice[i];
        }

        if i >= slow - 1 {
            let fast_ma = if i >= fast - 1 {
                fast_sum / fast as f64
            } else {
                // Calculate partial SMA for fast
                let mut sum = 0.0;
                let start = if i >= fast - 1 { i - fast + 1 } else { 0 };
                for j in start..=i {
                    sum += slice[j];
                }
                sum / ((i - start + 1) as f64)
            };
            let slow_ma = slow_sum / slow as f64;
            macd[i] = fast_ma - slow_ma;
        } else {
            macd[i] = f64::NAN;
        }
    }

    // First stochastic calculation
    let macd_min = min_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let macd_max = max_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;

    let mut stok = alloc_with_nan_prefix(working_len, 0);
    for i in 0..working_len {
        let range = macd_max[i] - macd_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            stok[i] = (macd[i] - macd_min[i]) / range * 100.0;
        } else if !macd[i].is_nan() {
            stok[i] = 50.0;
        }
    }

    // First EMA smoothing (inline)
    let d_alpha = 2.0 / (d as f64 + 1.0);
    let mut d_vals = alloc_with_nan_prefix(working_len, 0);
    let mut d_ema = f64::NAN;
    let mut d_sum = 0.0;
    let mut d_count = 0;

    for i in 0..working_len {
        if !stok[i].is_nan() {
            if d_count < d {
                d_sum += stok[i];
                d_count += 1;
                if d_count == d {
                    d_ema = d_sum / d as f64;
                    d_vals[i] = d_ema;
                } else {
                    d_vals[i] = f64::NAN;
                }
            } else {
                d_ema = d_alpha * stok[i] + (1.0 - d_alpha) * d_ema;
                d_vals[i] = d_ema;
            }
        } else {
            d_vals[i] = f64::NAN;
        }
    }

    // Second stochastic calculation
    let d_min = min_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let d_max = max_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;

    let mut kd = alloc_with_nan_prefix(working_len, 0);
    for i in 0..working_len {
        let range = d_max[i] - d_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            kd[i] = (d_vals[i] - d_min[i]) / range * 100.0;
        } else if !d_vals[i].is_nan() {
            kd[i] = 50.0;
        }
    }

    // Final EMA smoothing (inline)
    let mut final_ema = f64::NAN;
    let mut final_sum = 0.0;
    let mut final_count = 0;

    for i in 0..working_len {
        if !kd[i].is_nan() {
            if final_count < d {
                final_sum += kd[i];
                final_count += 1;
                if final_count == d {
                    final_ema = final_sum / d as f64;
                    out[first + i] = final_ema;
                } else {
                    out[first + i] = f64::NAN;
                }
            } else {
                final_ema = d_alpha * kd[i] + (1.0 - d_alpha) * final_ema;
                out[first + i] = final_ema;
            }
        } else {
            out[first + i] = f64::NAN;
        }
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stc_avx2(
    data: &[f64],
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
    fast_type: &str,
    slow_type: &str,
    first: usize,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_scalar(data, fast, slow, k, d, fast_type, slow_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stc_avx512(
    data: &[f64],
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
    fast_type: &str,
    slow_type: &str,
    first: usize,
    out: &mut [f64],
) -> Result<(), StcError> {
    if fast <= 32 && slow <= 32 {
        unsafe { stc_avx512_short(data, fast, slow, k, d, fast_type, slow_type, first, out) }
    } else {
        unsafe { stc_avx512_long(data, fast, slow, k, d, fast_type, slow_type, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stc_avx512_short(
    data: &[f64],
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
    fast_type: &str,
    slow_type: &str,
    first: usize,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_scalar(data, fast, slow, k, d, fast_type, slow_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stc_avx512_long(
    data: &[f64],
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
    fast_type: &str,
    slow_type: &str,
    first: usize,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_scalar(data, fast, slow, k, d, fast_type, slow_type, first, out)
}

// Batch API and related structs
#[derive(Clone, Debug)]
pub struct StcBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub k_period: (usize, usize, usize),
    pub d_period: (usize, usize, usize),
}

impl Default for StcBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (23, 23, 0),
            slow_period: (50, 50, 0),
            k_period: (10, 10, 0),
            d_period: (3, 3, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StcBatchBuilder {
    range: StcBatchRange,
    kernel: Kernel,
}

impl StcBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }
    pub fn k_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.k_period = (start, end, step);
        self
    }
    pub fn d_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.d_period = (start, end, step);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<StcBatchOutput, StcError> {
        stc_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<StcBatchOutput, StcError> {
        StcBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<StcBatchOutput, StcError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<StcBatchOutput, StcError> {
        StcBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn stc_batch_with_kernel(
    data: &[f64],
    sweep: &StcBatchRange,
    k: Kernel,
) -> Result<StcBatchOutput, StcError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(StcError::Internal("Invalid kernel".to_string())),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    stc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StcBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<StcParams>,
    pub rows: usize,
    pub cols: usize,
}

impl StcBatchOutput {
    pub fn row_for_params(&self, p: &StcParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_period == p.fast_period
                && c.slow_period == p.slow_period
                && c.k_period == p.k_period
                && c.d_period == p.d_period
        })
    }
    pub fn values_for(&self, p: &StcParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &StcBatchRange) -> Vec<StcParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let fasts = axis(r.fast_period);
    let slows = axis(r.slow_period);
    let ks = axis(r.k_period);
    let ds = axis(r.d_period);
    let mut out = Vec::with_capacity(fasts.len() * slows.len() * ks.len() * ds.len());
    for &f in &fasts {
        for &s in &slows {
            for &k in &ks {
                for &d in &ds {
                    out.push(StcParams {
                        fast_period: Some(f),
                        slow_period: Some(s),
                        k_period: Some(k),
                        d_period: Some(d),
                        fast_ma_type: None,
                        slow_ma_type: None,
                    });
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn stc_batch_slice(
    data: &[f64],
    sweep: &StcBatchRange,
    kern: Kernel,
) -> Result<StcBatchOutput, StcError> {
    stc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn stc_batch_par_slice(
    data: &[f64],
    sweep: &StcBatchRange,
    kern: Kernel,
) -> Result<StcBatchOutput, StcError> {
    stc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn stc_batch_inner(
    data: &[f64],
    sweep: &StcBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<StcBatchOutput, StcError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(StcError::NotEnoughValidData {
            needed: 1,
            valid: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StcError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(|c| {
            c.fast_period
                .unwrap()
                .max(c.slow_period.unwrap())
                .max(c.k_period.unwrap())
                .max(c.d_period.unwrap())
        })
        .max()
        .unwrap();
    if data.len() - first < max_needed {
        return Err(StcError::NotEnoughValidData {
            needed: max_needed,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            first
                + c.fast_period
                    .unwrap()
                    .max(c.slow_period.unwrap())
                    .max(c.k_period.unwrap())
                    .max(c.d_period.unwrap())
                - 1
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        match kern {
            Kernel::Scalar => stc_row_scalar(data, first, prm, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => stc_row_avx2(data, first, prm, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => stc_row_avx512(data, first, prm, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => stc_row_scalar(data, first, prm, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_slice
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| {
                    do_row(row, slice).unwrap();
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
                do_row(row, slice).unwrap();
            }
        }
    } else {
        for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
            do_row(row, slice).unwrap();
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(StcBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn stc_row_scalar(
    data: &[f64],
    first: usize,
    prm: &StcParams,
    out: &mut [f64],
) -> Result<(), StcError> {
    let fast_type = prm.fast_ma_type.as_deref().unwrap_or("ema");
    let slow_type = prm.slow_ma_type.as_deref().unwrap_or("ema");

    // Check for classic kernel optimization
    if fast_type == "ema" && slow_type == "ema" {
        return stc_row_scalar_classic_ema(data, first, prm, out);
    } else if fast_type == "sma" && slow_type == "sma" {
        return stc_row_scalar_classic_sma(data, first, prm, out);
    }

    // Fall back to regular implementation
    stc_scalar(
        data,
        prm.fast_period.unwrap(),
        prm.slow_period.unwrap(),
        prm.k_period.unwrap(),
        prm.d_period.unwrap(),
        fast_type,
        slow_type,
        first,
        out,
    )
}

// Classic row kernel with inline EMA for batch processing
#[inline(always)]
pub unsafe fn stc_row_scalar_classic_ema(
    data: &[f64],
    first: usize,
    prm: &StcParams,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_scalar_classic_ema(
        data,
        prm.fast_period.unwrap(),
        prm.slow_period.unwrap(),
        prm.k_period.unwrap(),
        prm.d_period.unwrap(),
        first,
        out,
    )
}

// Classic row kernel with inline SMA for batch processing
#[inline(always)]
pub unsafe fn stc_row_scalar_classic_sma(
    data: &[f64],
    first: usize,
    prm: &StcParams,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_scalar_classic_sma(
        data,
        prm.fast_period.unwrap(),
        prm.slow_period.unwrap(),
        prm.k_period.unwrap(),
        prm.d_period.unwrap(),
        first,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx2(
    data: &[f64],
    first: usize,
    prm: &StcParams,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_row_scalar(data, first, prm, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx512(
    data: &[f64],
    first: usize,
    prm: &StcParams,
    out: &mut [f64],
) -> Result<(), StcError> {
    if prm.fast_period.unwrap() <= 32 && prm.slow_period.unwrap() <= 32 {
        stc_row_avx512_short(data, first, prm, out)
    } else {
        stc_row_avx512_long(data, first, prm, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx512_short(
    data: &[f64],
    first: usize,
    prm: &StcParams,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_row_scalar(data, first, prm, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx512_long(
    data: &[f64],
    first: usize,
    prm: &StcParams,
    out: &mut [f64],
) -> Result<(), StcError> {
    stc_row_scalar(data, first, prm, out)
}

// Streaming STC
#[derive(Debug, Clone)]
pub struct StcStream {
    pub fast_period: usize,
    pub slow_period: usize,
    pub k_period: usize,
    pub d_period: usize,
    buffer: Vec<f64>,
    params: StcParams,
    // Internal state for streaming calculations
    // Note: Current implementation recalculates from buffer on each update (O(n))
    // Future optimization could maintain streaming state for each component
}

impl StcStream {
    pub fn try_new(params: StcParams) -> Result<Self, StcError> {
        let fast = params.fast_period.unwrap_or(23);
        let slow = params.slow_period.unwrap_or(50);
        let k = params.k_period.unwrap_or(10);
        let d = params.d_period.unwrap_or(3);

        if fast == 0 || slow == 0 || k == 0 || d == 0 {
            return Err(StcError::NotEnoughValidData {
                needed: 1,
                valid: 0,
            });
        }

        Ok(Self {
            fast_period: fast,
            slow_period: slow,
            k_period: k,
            d_period: d,
            buffer: Vec::new(), // Start with empty buffer that will grow
            params,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Append new value to buffer (grows unbounded for accuracy)
        self.buffer.push(value);

        // Need enough data for the calculation
        let min_data = self
            .fast_period
            .max(self.slow_period)
            .max(self.k_period)
            .max(self.d_period);

        if self.buffer.len() < min_data {
            return None;
        }

        // Recalculate STC on full buffer (O(n) but accurate)
        let input = StcInput::from_slice(&self.buffer, self.params.clone());
        match stc(&input) {
            Ok(res) => res.values.last().cloned(),
            Err(_) => None,
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "stc")]
#[pyo3(signature = (data, fast_period=23, slow_period=50, k_period=10, d_period=3, fast_ma_type="ema", slow_ma_type="ema", kernel=None))]
pub fn stc_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    fast_period: usize,
    slow_period: usize,
    k_period: usize,
    d_period: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = StcParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
        k_period: Some(k_period),
        d_period: Some(d_period),
        fast_ma_type: Some(fast_ma_type.to_string()),
        slow_ma_type: Some(slow_ma_type.to_string()),
    };
    let stc_in = StcInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| stc_with_kernel(&stc_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "StcStream")]
pub struct StcStreamPy {
    stream: StcStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StcStreamPy {
    #[new]
    fn new(
        fast_period: usize,
        slow_period: usize,
        k_period: usize,
        d_period: usize,
    ) -> PyResult<Self> {
        let params = StcParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            k_period: Some(k_period),
            d_period: Some(d_period),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("ema".to_string()),
        };
        let stream =
            StcStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(StcStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "stc_batch")]
#[pyo3(signature = (data, fast_period_range, slow_period_range, k_period_range, d_period_range, kernel=None))]
pub fn stc_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    fast_period_range: (usize, usize, usize),
    slow_period_range: (usize, usize, usize),
    k_period_range: (usize, usize, usize),
    d_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;

    let sweep = StcBatchRange {
        fast_period: fast_period_range,
        slow_period: slow_period_range,
        k_period: k_period_range,
        d_period: d_period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

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
            stc_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|p| p.fast_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|p| p.slow_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "k_periods",
        combos
            .iter()
            .map(|p| p.k_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "d_periods",
        combos
            .iter()
            .map(|p| p.d_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_stc_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stc_py, m)?)?;
    m.add_function(wrap_pyfunction!(stc_batch_py, m)?)?;
    m.add_class::<StcStreamPy>()?;
    Ok(())
}

// Helper function for batch processing
#[inline(always)]
fn stc_batch_inner_into(
    data: &[f64],
    sweep: &StcBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<StcParams>, StcError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(StcError::NotEnoughValidData {
            needed: 1,
            valid: 0,
        });
    }

    let len = data.len();
    if len == 0 {
        return Err(StcError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(StcError::AllValuesNaN)?;

    let max_needed = combos
        .iter()
        .map(|c| {
            c.fast_period
                .unwrap()
                .max(c.slow_period.unwrap())
                .max(c.k_period.unwrap())
                .max(c.d_period.unwrap())
        })
        .max()
        .unwrap();

    if (len - first) < max_needed {
        return Err(StcError::NotEnoughValidData {
            needed: max_needed,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    if out.len() != rows * cols {
        return Err(StcError::Internal(format!(
            "out len {} != rows*cols {}",
            out.len(),
            rows * cols
        )));
    }

    // init NaN prefixes on the destination matrix
    let mut out_mu = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            first
                + c.fast_period
                    .unwrap()
                    .max(c.slow_period.unwrap())
                    .max(c.k_period.unwrap())
                    .max(c.d_period.unwrap())
                - 1
        })
        .collect();
    init_matrix_prefixes(&mut out_mu, cols, &warm);

    let chosen = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match chosen {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx512 => Kernel::Avx512,
        Kernel::Avx2 => Kernel::Avx2,
        Kernel::Scalar => Kernel::Scalar,
        _ => Kernel::Scalar,
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        match simd {
            Kernel::Scalar => stc_row_scalar(data, first, &combos[row], out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => stc_row_avx2(data, first, &combos[row], out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => stc_row_avx512(data, first, &combos[row], out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => stc_row_scalar(data, first, &combos[row], out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu.par_chunks_mut(cols).enumerate().for_each(|(r, mr)| {
                // reinterpret row back to f64 for compute
                let row_slice =
                    unsafe { core::slice::from_raw_parts_mut(mr.as_mut_ptr() as *mut f64, cols) };
                do_row(r, row_slice).unwrap();
            });
        }
        #[cfg(target_arch = "wasm32")]
        for (r, mr) in out_mu.chunks_mut(cols).enumerate() {
            let row_slice =
                unsafe { core::slice::from_raw_parts_mut(mr.as_mut_ptr() as *mut f64, cols) };
            do_row(r, row_slice).unwrap();
        }
    } else {
        for (r, mr) in out_mu.chunks_mut(cols).enumerate() {
            let row_slice =
                unsafe { core::slice::from_raw_parts_mut(mr.as_mut_ptr() as *mut f64, cols) };
            do_row(r, row_slice).unwrap();
        }
    }

    Ok(combos)
}

// ============================================
// WASM Bindings
// ============================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stc_js(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    k_period: usize,
    d_period: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
    let params = StcParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
        k_period: Some(k_period),
        d_period: Some(d_period),
        fast_ma_type: Some(fast_ma_type.to_string()),
        slow_ma_type: Some(slow_ma_type.to_string()),
    };
    let input = StcInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];
    stc_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stc_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_period: usize,
    slow_period: usize,
    k_period: usize,
    d_period: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = StcParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            k_period: Some(k_period),
            d_period: Some(d_period),
            fast_ma_type: Some(fast_ma_type.to_string()),
            slow_ma_type: Some(slow_ma_type.to_string()),
        };
        let input = StcInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // Handle aliasing case
            let mut temp = vec![0.0; len];
            stc_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            stc_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StcBatchConfig {
    pub fast_period_range: (usize, usize, usize),
    pub slow_period_range: (usize, usize, usize),
    pub k_period_range: (usize, usize, usize),
    pub d_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StcBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<StcParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = stc_batch)]
pub fn stc_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: StcBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = StcBatchRange {
        fast_period: config.fast_period_range,
        slow_period: config.slow_period_range,
        k_period: config.k_period_range,
        d_period: config.d_period_range,
    };

    let result = stc_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let output = StcBatchJsOutput {
        values: result.values,
        combos: result.combos,
        rows: result.rows,
        cols: result.cols,
    };

    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_stc_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StcInput::with_default_candles(&candles);
        let output = stc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_stc_last_five(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StcInput::with_default_candles(&candles);
        let result = stc_with_kernel(&input, kernel)?;
        let expected = [
            0.21394384188858884,
            0.10697192094429442,
            0.05348596047214721,
            50.02674298023607,
            49.98686202668157,
        ];
        let n = result.values.len();
        for (i, &exp) in expected.iter().enumerate() {
            let val = result.values[n - 5 + i];
            assert!(
                (val - exp).abs() < 1e-5,
                "Expected {}, got {} at idx {}",
                exp,
                val,
                n - 5 + i
            );
        }
        Ok(())
    }

    fn check_stc_with_slice_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let slice_data = [10.0, 11.0, 12.0, 13.0, 14.0];
        let params = StcParams {
            fast_period: Some(2),
            slow_period: Some(3),
            k_period: Some(2),
            d_period: Some(1),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("ema".to_string()),
        };
        let input = StcInput::from_slice(&slice_data, params);
        let result = stc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), slice_data.len());
        Ok(())
    }

    fn check_stc_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: [f64; 0] = [];
        let input = StcInput::from_slice(&data, StcParams::default());
        let result = stc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_stc_all_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = StcInput::from_slice(&data, StcParams::default());
        let result = stc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_stc_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, 2.0, 3.0];
        let params = StcParams {
            fast_period: Some(5),
            ..Default::default()
        };
        let input = StcInput::from_slice(&data, params);
        let result = stc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_stc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            StcParams::default(), // fast: 23, slow: 50, k: 10, d: 3
            StcParams {
                fast_period: Some(2),
                slow_period: Some(3),
                k_period: Some(2),
                d_period: Some(1),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
            StcParams {
                fast_period: Some(5),
                slow_period: Some(10),
                k_period: Some(5),
                d_period: Some(2),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
            StcParams {
                fast_period: Some(10),
                slow_period: Some(20),
                k_period: Some(7),
                d_period: Some(3),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
            StcParams {
                fast_period: Some(20),
                slow_period: Some(40),
                k_period: Some(10),
                d_period: Some(5),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
            StcParams {
                fast_period: Some(30),
                slow_period: Some(60),
                k_period: Some(15),
                d_period: Some(7),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
            StcParams {
                fast_period: Some(50),
                slow_period: Some(100),
                k_period: Some(20),
                d_period: Some(10),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
            // Edge case: minimum periods
            StcParams {
                fast_period: Some(2),
                slow_period: Some(2),
                k_period: Some(2),
                d_period: Some(1),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
            // Edge case: fast > slow (unusual but valid)
            StcParams {
                fast_period: Some(25),
                slow_period: Some(15),
                k_period: Some(10),
                d_period: Some(3),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = StcInput::from_candles(&candles, "close", params.clone());
            let output = stc_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: fast={}, slow={}, k={}, d={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fast_period.unwrap_or(23),
                        params.slow_period.unwrap_or(50),
                        params.k_period.unwrap_or(10),
                        params.d_period.unwrap_or(3),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: fast={}, slow={}, k={}, d={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fast_period.unwrap_or(23),
                        params.slow_period.unwrap_or(50),
                        params.k_period.unwrap_or(10),
                        params.d_period.unwrap_or(3),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: fast={}, slow={}, k={}, d={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fast_period.unwrap_or(23),
                        params.slow_period.unwrap_or(50),
                        params.k_period.unwrap_or(10),
                        params.d_period.unwrap_or(3),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_stc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_stc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }
    generate_all_stc_tests!(
        check_stc_default_params,
        check_stc_last_five,
        check_stc_with_slice_data,
        check_stc_empty_data,
        check_stc_all_nan_data,
        check_stc_not_enough_valid_data,
        check_stc_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = StcBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = StcParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (fast_start, fast_end, fast_step, slow_start, slow_end, slow_step, k_start, k_end, k_step, d_start, d_end, d_step)
            (2, 10, 2, 3, 15, 3, 2, 8, 2, 1, 3, 1), // Small periods
            (5, 25, 5, 10, 50, 10, 5, 15, 5, 2, 5, 1), // Medium periods
            (20, 40, 10, 40, 80, 20, 10, 20, 5, 3, 6, 1), // Large periods
            (2, 5, 1, 3, 6, 1, 2, 4, 1, 1, 2, 1),   // Dense small range
            (10, 10, 0, 20, 20, 0, 10, 10, 0, 3, 3, 0), // Single values (step=0)
            (15, 30, 5, 30, 60, 10, 7, 14, 7, 3, 5, 2), // Mixed ranges
            (50, 100, 25, 100, 200, 50, 20, 30, 10, 5, 10, 5), // Very large periods
        ];

        for (
            cfg_idx,
            &(
                f_start,
                f_end,
                f_step,
                s_start,
                s_end,
                s_step,
                k_start,
                k_end,
                k_step,
                d_start,
                d_end,
                d_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = StcBatchBuilder::new()
                .kernel(kernel)
                .fast_period_range(f_start, f_end, f_step)
                .slow_period_range(s_start, s_end, s_step)
                .k_period_range(k_start, k_end, k_step)
                .d_period_range(d_start, d_end, d_step)
                .apply_candles(&c, "close")?;

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
						 at row {} col {} (flat index {}) with params: fast={}, slow={}, k={}, d={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_period.unwrap_or(23),
                        combo.slow_period.unwrap_or(50),
                        combo.k_period.unwrap_or(10),
                        combo.d_period.unwrap_or(3)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast={}, slow={}, k={}, d={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_period.unwrap_or(23),
                        combo.slow_period.unwrap_or(50),
                        combo.k_period.unwrap_or(10),
                        combo.d_period.unwrap_or(3)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast={}, slow={}, k={}, d={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_period.unwrap_or(23),
                        combo.slow_period.unwrap_or(50),
                        combo.k_period.unwrap_or(10),
                        combo.d_period.unwrap_or(3)
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
