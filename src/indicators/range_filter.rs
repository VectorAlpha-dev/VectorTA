//! # Range Filter [DW]
//!
//! The Range Filter is an experimental study designed to filter out minor price action for a clearer view of trends.
//! Inspired by the QQE's volatility filter, this filter applies the process directly to price rather than to a smoothed RSI.
//!
//! ## Parameters
//! - **range_size**: Multiplier for the range size (default: 2.618)
//! - **range_period**: Period for calculating Average Change (default: 14)
//! - **smooth_range**: Whether to smooth the range (default: true)
//! - **smooth_period**: Period for smoothing the range (default: 27)
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **filter**: Vector of filtered price values with NaN prefix during warmup
//! - **high_band**: Vector of upper band values
//! - **low_band**: Vector of lower band values
//!
//! ## Developer Notes
//! - Decision: SIMD kept as stubs. The core loop is inherently sequential (stateful EMAs + clamp), so AVX2/AVX512 do not provide a clear win here. Runtime selection still routes through the stubs which delegate to the scalar path.
//! - Batch: minor row-specific optimization only when multiple parameter rows are requested (precomputes abs-change once). It is gated to avoid overhead for the common single-row case used in benches/tests.
//! - Streaming: Implemented with O(1) update performance (maintains running EMAs)
//! - CUDA: batch and many-series wrappers share warmup/NaN semantics with the scalar path and return VRAM handles used by Python via CUDA Array Interface v3 and DLPack v1.x.
//! - Zero-copy Memory: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::{PyBufferError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaRangeFilter};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyUntypedArrayMethods;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

impl<'a> AsRef<[f64]> for RangeFilterInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RangeFilterData::Slice(slice) => slice,
            RangeFilterData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RangeFilterData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RangeFilterOutput {
    pub filter: Vec<f64>,
    pub high_band: Vec<f64>,
    pub low_band: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct RangeFilterParams {
    pub range_size: Option<f64>,
    pub range_period: Option<usize>,
    pub smooth_range: Option<bool>,
    pub smooth_period: Option<usize>,
}

impl Default for RangeFilterParams {
    fn default() -> Self {
        Self {
            range_size: Some(2.618),
            range_period: Some(14),
            smooth_range: Some(true),
            smooth_period: Some(27),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeFilterInput<'a> {
    pub data: RangeFilterData<'a>,
    pub params: RangeFilterParams,
}

impl<'a> RangeFilterInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: RangeFilterParams) -> Self {
        Self {
            data: RangeFilterData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: RangeFilterParams) -> Self {
        Self {
            data: RangeFilterData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", RangeFilterParams::default())
    }

    #[inline]
    pub fn get_range_size(&self) -> f64 {
        self.params.range_size.unwrap_or(2.618)
    }

    #[inline]
    pub fn get_range_period(&self) -> usize {
        self.params.range_period.unwrap_or(14)
    }

    #[inline]
    pub fn get_smooth_range(&self) -> bool {
        self.params.smooth_range.unwrap_or(true)
    }

    #[inline]
    pub fn get_smooth_period(&self) -> usize {
        self.params.smooth_period.unwrap_or(27)
    }
}

#[derive(Clone, Debug)]
pub struct RangeFilterBuilder {
    range_size: Option<f64>,
    range_period: Option<usize>,
    smooth_range: Option<bool>,
    smooth_period: Option<usize>,
    kernel: Kernel,
}

impl Default for RangeFilterBuilder {
    fn default() -> Self {
        Self {
            range_size: None,
            range_period: None,
            smooth_range: None,
            smooth_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RangeFilterBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range_size(mut self, x: f64) -> Self {
        self.range_size = Some(x);
        self
    }

    #[inline(always)]
    pub fn range_period(mut self, n: usize) -> Self {
        self.range_period = Some(n);
        self
    }

    #[inline(always)]
    pub fn smooth_range(mut self, b: bool) -> Self {
        self.smooth_range = Some(b);
        self
    }

    #[inline(always)]
    pub fn smooth_period(mut self, n: usize) -> Self {
        self.smooth_period = Some(n);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<RangeFilterOutput, RangeFilterError> {
        let p = RangeFilterParams {
            range_size: self.range_size,
            range_period: self.range_period,
            smooth_range: self.smooth_range,
            smooth_period: self.smooth_period,
        };
        let i = RangeFilterInput::from_candles(c, "close", p);
        range_filter_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<RangeFilterOutput, RangeFilterError> {
        let p = RangeFilterParams {
            range_size: self.range_size,
            range_period: self.range_period,
            smooth_range: self.smooth_range,
            smooth_period: self.smooth_period,
        };
        let i = RangeFilterInput::from_slice(d, p);
        range_filter_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum RangeFilterError {
    #[error("range_filter: Input data slice is empty.")]
    EmptyInputData,

    #[error("range_filter: All values are NaN.")]
    AllValuesNaN,

    #[error("range_filter: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("range_filter: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("range_filter: Invalid range_size: {range_size}")]
    InvalidRangeSize { range_size: f64 },

    #[error("range_filter: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("range_filter: Invalid batch range: start={start}, end={end}, step={step}")]
    InvalidRange { start: f64, end: f64, step: f64 },

    #[error("range_filter: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

// Batch processing structures
#[derive(Clone, Debug)]
pub struct RangeFilterBatchRange {
    pub range_size: (f64, f64, f64),
    pub range_period: (usize, usize, usize),
    pub smooth_range: Option<bool>,
    pub smooth_period: Option<usize>,
}

impl Default for RangeFilterBatchRange {
    fn default() -> Self {
        Self {
            range_size: (2.618, 2.867, 0.001),
            range_period: (14, 14, 0),
            smooth_range: Some(true),
            smooth_period: Some(27),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RangeFilterBatchOutput {
    pub filter_values: Vec<f64>,
    pub high_band_values: Vec<f64>,
    pub low_band_values: Vec<f64>,
    pub combos: Vec<RangeFilterParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RangeFilterBatchOutput {
    pub fn row_for_params(&self, p: &RangeFilterParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.range_size.unwrap_or(2.618) - p.range_size.unwrap_or(2.618)).abs() < 1e-12
                && c.range_period.unwrap_or(14) == p.range_period.unwrap_or(14)
                && c.smooth_range.unwrap_or(true) == p.smooth_range.unwrap_or(true)
                && c.smooth_period.unwrap_or(27) == p.smooth_period.unwrap_or(27)
        })
    }

    pub fn filter_values_for(&self, p: &RangeFilterParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.filter_values[start..start + self.cols]
        })
    }

    pub fn high_band_values_for(&self, p: &RangeFilterParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.high_band_values[start..start + self.cols]
        })
    }

    pub fn low_band_values_for(&self, p: &RangeFilterParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.low_band_values[start..start + self.cols]
        })
    }

    pub fn triple_for(&self, p: &RangeFilterParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            let end = start + self.cols;
            (
                &self.filter_values[start..end],
                &self.high_band_values[start..end],
                &self.low_band_values[start..end],
            )
        })
    }
}

#[derive(Clone, Debug)]
pub struct RangeFilterBatchBuilder {
    range: RangeFilterBatchRange,
    kernel: Kernel,
}

impl Default for RangeFilterBatchBuilder {
    fn default() -> Self {
        Self {
            range: RangeFilterBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl RangeFilterBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, r: RangeFilterBatchRange) -> Self {
        self.range = r;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<RangeFilterBatchOutput, RangeFilterError> {
        range_filter_batch_slice(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice_par(self, data: &[f64]) -> Result<RangeFilterBatchOutput, RangeFilterError> {
        range_filter_batch_par_slice(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<RangeFilterBatchOutput, RangeFilterError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    #[inline]
    pub fn with_default_candles(c: &Candles) -> Result<RangeFilterBatchOutput, RangeFilterError> {
        RangeFilterBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }

    #[inline]
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<RangeFilterBatchOutput, RangeFilterError> {
        RangeFilterBatchBuilder::new().kernel(k).apply_slice(data)
    }
}

#[inline(always)]
pub fn range_filter(input: &RangeFilterInput) -> Result<RangeFilterOutput, RangeFilterError> {
    range_filter_with_kernel(input, Kernel::Auto)
}

/// Writes Range Filter outputs into caller-provided buffers without allocating.
///
/// - Preserves NaN warmup prefixes exactly as the Vec-returning API.
/// - All output slice lengths must equal the input length.
/// - Dispatches with `Kernel::Auto` and uses the same compute path as `range_filter()`.
#[inline]
pub fn range_filter_into(
    input: &RangeFilterInput,
    out_filter: &mut [f64],
    out_high: &mut [f64],
    out_low: &mut [f64],
) -> Result<(), RangeFilterError> {
    range_filter_into_slice(out_filter, out_high, out_low, input, Kernel::Auto)
}

#[inline]
pub fn range_filter_into_slice(
    dst_filter: &mut [f64],
    dst_high: &mut [f64],
    dst_low: &mut [f64],
    input: &RangeFilterInput,
    kern: Kernel,
) -> Result<(), RangeFilterError> {
    let (data, range_size, range_period, smooth_range, smooth_period, first, chosen) =
        range_filter_prepare(input, kern)?;

    let n = data.len();
    if dst_filter.len() != n {
        return Err(RangeFilterError::OutputLengthMismatch {
            expected: n,
            got: dst_filter.len(),
        });
    }
    if dst_high.len() != n {
        return Err(RangeFilterError::OutputLengthMismatch {
            expected: n,
            got: dst_high.len(),
        });
    }
    if dst_low.len() != n {
        return Err(RangeFilterError::OutputLengthMismatch {
            expected: n,
            got: dst_low.len(),
        });
    }

    // Compute the values first
    range_filter_compute_into(
        data,
        range_size,
        range_period,
        smooth_range,
        smooth_period,
        first,
        chosen,
        dst_filter,
        dst_high,
        dst_low,
    )?;

    // Then set the warmup prefix to NaN (after computation to match standard behavior)
    let warmup_end = first + range_period.max(if smooth_range { smooth_period } else { 0 });
    for i in 0..warmup_end.min(n) {
        dst_filter[i] = f64::NAN;
        dst_high[i] = f64::NAN;
        dst_low[i] = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
pub fn range_filter_with_kernel(
    input: &RangeFilterInput,
    kernel: Kernel,
) -> Result<RangeFilterOutput, RangeFilterError> {
    let (data, range_size, range_period, smooth_range, smooth_period, first, chosen) =
        range_filter_prepare(input, kernel)?;

    // Calculate warmup based on periods involved
    let warmup_end = first + range_period.max(if smooth_range { smooth_period } else { 0 });

    let mut filter = alloc_with_nan_prefix(data.len(), warmup_end);
    let mut high_band = alloc_with_nan_prefix(data.len(), warmup_end);
    let mut low_band = alloc_with_nan_prefix(data.len(), warmup_end);

    range_filter_compute_into(
        data,
        range_size,
        range_period,
        smooth_range,
        smooth_period,
        first,
        chosen,
        &mut filter,
        &mut high_band,
        &mut low_band,
    )?;

    // Ensure warmup prefix remains NaN, matching documented behavior and into_slice()
    let n = data.len();
    for i in 0..warmup_end.min(n) {
        filter[i] = f64::NAN;
        high_band[i] = f64::NAN;
        low_band[i] = f64::NAN;
    }

    Ok(RangeFilterOutput {
        filter,
        high_band,
        low_band,
    })
}

#[inline(always)]
fn range_filter_prepare<'a>(
    input: &'a RangeFilterInput,
    kernel: Kernel,
) -> Result<(&'a [f64], f64, usize, bool, usize, usize, Kernel), RangeFilterError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(RangeFilterError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RangeFilterError::AllValuesNaN)?;
    let range_size = input.get_range_size();
    if !range_size.is_finite() || range_size <= 0.0 {
        return Err(RangeFilterError::InvalidRangeSize { range_size });
    }
    let range_period = input.get_range_period();
    let smooth_range = input.get_smooth_range();
    let smooth_period = input.get_smooth_period();
    if range_period == 0 || range_period > len {
        return Err(RangeFilterError::InvalidPeriod {
            period: range_period,
            data_len: len,
        });
    }

    if smooth_range && (smooth_period == 0 || smooth_period > len) {
        return Err(RangeFilterError::InvalidPeriod {
            period: smooth_period,
            data_len: len,
        });
    }

    let needed = range_period.max(if smooth_range { smooth_period } else { 0 });
    if len - first < needed {
        return Err(RangeFilterError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    Ok((
        data,
        range_size,
        range_period,
        smooth_range,
        smooth_period,
        first,
        chosen,
    ))
}

#[inline(always)]
fn range_filter_compute_into(
    data: &[f64],
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
    first: usize,
    kernel: Kernel,
    filter: &mut [f64],
    high_band: &mut [f64],
    low_band: &mut [f64],
) -> Result<(), RangeFilterError> {
    match kernel {
        Kernel::Scalar => range_filter_scalar(
            data,
            range_size,
            range_period,
            smooth_range,
            smooth_period,
            first,
            filter,
            high_band,
            low_band,
        ),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            range_filter_avx2(
                data,
                range_size,
                range_period,
                smooth_range,
                smooth_period,
                first,
                filter,
                high_band,
                low_band,
            )
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            range_filter_avx512(
                data,
                range_size,
                range_period,
                smooth_range,
                smooth_period,
                first,
                filter,
                high_band,
                low_band,
            )
        },
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx512 => range_filter_scalar(
            data,
            range_size,
            range_period,
            smooth_range,
            smooth_period,
            first,
            filter,
            high_band,
            low_band,
        ),
        _ => range_filter_scalar(
            data,
            range_size,
            range_period,
            smooth_range,
            smooth_period,
            first,
            filter,
            high_band,
            low_band,
        ),
    }
}

#[inline]
pub fn range_filter_scalar(
    data: &[f64],
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
    first: usize,
    filter: &mut [f64],
    high_band: &mut [f64],
    low_band: &mut [f64],
) -> Result<(), RangeFilterError> {
    let n = data.len();
    if n == 0 {
        return Ok(());
    }

    
    let alpha_ac = 2.0 / (range_period as f64 + 1.0);
    let one_minus_alpha_ac = 1.0 - alpha_ac;
    let alpha_range = if smooth_range {
        2.0 / (smooth_period as f64 + 1.0)
    } else {
        0.0
    };
    let one_minus_alpha_range = 1.0 - alpha_range;

    
    let mut ac_ema: f64 = 0.0;
    let mut ac_initialized = false;
    let mut range_ema: f64 = 0.0;
    let mut range_initialized = false;

    
    let mut prev_filter = if first < n { data[first] } else { f64::NAN };
    let mut prev_price = prev_filter;

    
    if first + 1 >= n {
        return Ok(());
    }

    
    if !smooth_range {
        
        for i in (first + 1)..n {
            let price = data[i];

            
            let d = price - prev_price;
            let abs_change = d.abs();
            if !abs_change.is_nan() {
                if !ac_initialized {
                    ac_ema = abs_change;
                    ac_initialized = true;
                } else {
                    ac_ema = alpha_ac * abs_change + one_minus_alpha_ac * ac_ema;
                }
            }

            if !ac_initialized {
                prev_price = price;
                continue;
            }

            let range = ac_ema * range_size;

            
            let min_b = price - range;
            let max_b = price + range;
            let current = prev_filter.clamp(min_b, max_b);

            filter[i] = current;
            high_band[i] = current + range;
            low_band[i] = current - range;

            prev_filter = current;
            prev_price = price;
        }
    } else {
        
        for i in (first + 1)..n {
            let price = data[i];

            
            let d = price - prev_price;
            let abs_change = d.abs();
            if !abs_change.is_nan() {
                if !ac_initialized {
                    ac_ema = abs_change;
                    ac_initialized = true;
                } else {
                    ac_ema = alpha_ac * abs_change + one_minus_alpha_ac * ac_ema;
                }
            }

            if !ac_initialized {
                prev_price = price;
                continue;
            }

            let mut range = ac_ema * range_size;
            if !range_initialized {
                range_ema = range;
                range_initialized = true;
            } else {
                range_ema = alpha_range * range + one_minus_alpha_range * range_ema;
            }
            range = range_ema;

            
            let min_b = price - range;
            let max_b = price + range;
            let current = prev_filter.clamp(min_b, max_b);

            filter[i] = current;
            high_band[i] = current + range;
            low_band[i] = current - range;

            prev_filter = current;
            prev_price = price;
        }
    }

    Ok(())
}

#[inline]
fn range_filter_scalar_with_abs_change(
    data: &[f64],
    abs_change: &[f64],
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
    first: usize,
    filter: &mut [f64],
    high_band: &mut [f64],
    low_band: &mut [f64],
) -> Result<(), RangeFilterError> {
    let n = data.len();
    if n == 0 {
        return Ok(());
    }

    
    let alpha_ac = 2.0 / (range_period as f64 + 1.0);
    let one_minus_alpha_ac = 1.0 - alpha_ac;
    let alpha_range = if smooth_range {
        2.0 / (smooth_period as f64 + 1.0)
    } else {
        0.0
    };
    let one_minus_alpha_range = 1.0 - alpha_range;

    let mut ac_ema: f64 = 0.0;
    let mut ac_initialized = false;
    let mut range_ema: f64 = 0.0;
    let mut range_initialized = false;

    
    let mut prev_filter = if first < n { data[first] } else { f64::NAN };

    if first + 1 >= n {
        return Ok(());
    }

    if !smooth_range {
        for i in (first + 1)..n {
            let price = data[i];

            
            let ac = abs_change[i];
            if !ac.is_nan() {
                if !ac_initialized {
                    ac_ema = ac;
                    ac_initialized = true;
                } else {
                    ac_ema = alpha_ac * ac + one_minus_alpha_ac * ac_ema;
                }
            }

            if !ac_initialized {
                continue;
            }

            let range = ac_ema * range_size;

            let min_b = price - range;
            let max_b = price + range;
            let current = prev_filter.clamp(min_b, max_b);

            filter[i] = current;
            high_band[i] = current + range;
            low_band[i] = current - range;

            prev_filter = current;
        }
    } else {
        for i in (first + 1)..n {
            let price = data[i];

            
            let ac = abs_change[i];
            if !ac.is_nan() {
                if !ac_initialized {
                    ac_ema = ac;
                    ac_initialized = true;
                } else {
                    ac_ema = alpha_ac * ac + one_minus_alpha_ac * ac_ema;
                }
            }

            if !ac_initialized {
                continue;
            }

            let mut range = ac_ema * range_size;
            if !range_initialized {
                range_ema = range;
                range_initialized = true;
            } else {
                range_ema = alpha_range * range + one_minus_alpha_range * range_ema;
            }
            range = range_ema;

            let min_b = price - range;
            let max_b = price + range;
            let current = prev_filter.clamp(min_b, max_b);

            filter[i] = current;
            high_band[i] = current + range;
            low_band[i] = current - range;

            prev_filter = current;
        }
    }

    Ok(())
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn range_filter_avx2(
    data: &[f64],
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
    first: usize,
    filter: &mut [f64],
    high_band: &mut [f64],
    low_band: &mut [f64],
) -> Result<(), RangeFilterError> {
    
    
    range_filter_scalar(
        data,
        range_size,
        range_period,
        smooth_range,
        smooth_period,
        first,
        filter,
        high_band,
        low_band,
    )
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn range_filter_avx512(
    data: &[f64],
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
    first: usize,
    filter: &mut [f64],
    high_band: &mut [f64],
    low_band: &mut [f64],
) -> Result<(), RangeFilterError> {
    
    
    range_filter_scalar(
        data,
        range_size,
        range_period,
        smooth_range,
        smooth_period,
        first,
        filter,
        high_band,
        low_band,
    )
}


#[inline(always)]
fn expand_grid(r: &RangeFilterBatchRange) -> Result<Vec<RangeFilterParams>, RangeFilterError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, RangeFilterError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            loop {
                v.push(cur);
                match cur.checked_add(step) {
                    Some(next) if next <= end => {
                        cur = next;
                    }
                    Some(_) => break,
                    None => {
                        return Err(RangeFilterError::InvalidRange {
                            start: start as f64,
                            end: end as f64,
                            step: step as f64,
                        });
                    }
                }
            }
        } else {
            let mut cur = start;
            loop {
                v.push(cur);
                match cur.checked_sub(step) {
                    Some(next) if next >= end => {
                        cur = next;
                    }
                    Some(_) => break,
                    None => {
                        return Err(RangeFilterError::InvalidRange {
                            start: start as f64,
                            end: end as f64,
                            step: step as f64,
                        });
                    }
                }
            }
        }

        if v.is_empty() {
            return Err(RangeFilterError::InvalidRange {
                start: start as f64,
                end: end as f64,
                step: step as f64,
            });
        }

        Ok(v)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, RangeFilterError> {
        let eps = 1e-12;
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(RangeFilterError::InvalidRange { start, end, step });
        }
        if step.abs() < eps || (start - end).abs() < eps {
            return Ok(vec![start]);
        }

        let mut v = Vec::new();
        let delta = end - start;
        let dir = if delta >= 0.0 { 1.0 } else { -1.0 };
        let mut step_eff = step;
        if step_eff.signum() == 0.0 {
            step_eff = dir * step.abs();
        } else if step_eff.signum() != dir {
            step_eff = dir * step.abs();
        }

        let mut x = start;
        if dir > 0.0 {
            while x <= end + eps {
                v.push(x);
                x += step_eff;
            }
        } else {
            while x >= end - eps {
                v.push(x);
                x += step_eff;
            }
        }

        if v.is_empty() {
            return Err(RangeFilterError::InvalidRange { start, end, step });
        }

        Ok(v)
    }

    let range_sizes = axis_f64(r.range_size)?;
    let range_periods = axis_usize(r.range_period)?;

    let combos_len = range_sizes
        .len()
        .checked_mul(range_periods.len())
        .ok_or(RangeFilterError::InvalidRange {
            start: r.range_size.0,
            end: r.range_size.1,
            step: r.range_size.2,
        })?;

    let mut out = Vec::with_capacity(combos_len);
    for &rs in &range_sizes {
        for &rp in &range_periods {
            out.push(RangeFilterParams {
                range_size: Some(rs),
                range_period: Some(rp),
                smooth_range: r.smooth_range,
                smooth_period: r.smooth_period,
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn range_filter_batch_slice(
    data: &[f64],
    sweep: &RangeFilterBatchRange,
    kern: Kernel,
) -> Result<RangeFilterBatchOutput, RangeFilterError> {
    range_filter_batch_inner(data, sweep, kern, false)
}

pub fn range_filter_batch_with_kernel(
    data: &[f64],
    sweep: &RangeFilterBatchRange,
    k: Kernel,
) -> Result<RangeFilterBatchOutput, RangeFilterError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(RangeFilterError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    range_filter_batch_inner(data, sweep, simd, true)
}

#[inline(always)]
pub fn range_filter_batch_par_slice(
    data: &[f64],
    sweep: &RangeFilterBatchRange,
    kern: Kernel,
) -> Result<RangeFilterBatchOutput, RangeFilterError> {
    range_filter_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn range_filter_batch_inner(
    data: &[f64],
    sweep: &RangeFilterBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RangeFilterBatchOutput, RangeFilterError> {
    let combos = expand_grid(sweep)?;
    let cols = data.len();
    let rows = combos.len();

    if cols == 0 {
        return Err(RangeFilterError::AllValuesNaN);
    }

    
    let _ = rows
        .checked_mul(cols)
        .ok_or(RangeFilterError::InvalidRange {
            start: sweep.range_size.0,
            end: sweep.range_size.1,
            step: sweep.range_size.2,
        })?;
    let mut filter_mu = make_uninit_matrix(rows, cols);
    let mut high_mu = make_uninit_matrix(rows, cols);
    let mut low_mu = make_uninit_matrix(rows, cols);

    
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RangeFilterError::AllValuesNaN)?;
    let warms: Vec<usize> = combos
        .iter()
        .map(|c| {
            let rp = c.range_period.unwrap_or(14);
            let sp = if c.smooth_range.unwrap_or(true) {
                c.smooth_period.unwrap_or(27)
            } else {
                0
            };
            first + rp.max(sp)
        })
        .collect();

    init_matrix_prefixes(&mut filter_mu, cols, &warms);
    init_matrix_prefixes(&mut high_mu, cols, &warms);
    init_matrix_prefixes(&mut low_mu, cols, &warms);

    
    let mut f_guard = core::mem::ManuallyDrop::new(filter_mu);
    let mut h_guard = core::mem::ManuallyDrop::new(high_mu);
    let mut l_guard = core::mem::ManuallyDrop::new(low_mu);

    let filter: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(f_guard.as_mut_ptr() as *mut f64, f_guard.len()) };
    let high: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(h_guard.as_mut_ptr() as *mut f64, h_guard.len()) };
    let low: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(l_guard.as_mut_ptr() as *mut f64, l_guard.len()) };

    range_filter_batch_inner_into(data, &combos, kern, parallel, filter, high, low)?;

    
    let filter_values = unsafe {
        Vec::from_raw_parts(
            f_guard.as_mut_ptr() as *mut f64,
            f_guard.len(),
            f_guard.capacity(),
        )
    };
    let high_values = unsafe {
        Vec::from_raw_parts(
            h_guard.as_mut_ptr() as *mut f64,
            h_guard.len(),
            h_guard.capacity(),
        )
    };
    let low_values = unsafe {
        Vec::from_raw_parts(
            l_guard.as_mut_ptr() as *mut f64,
            l_guard.len(),
            l_guard.capacity(),
        )
    };

    Ok(RangeFilterBatchOutput {
        filter_values: filter_values,
        high_band_values: high_values,
        low_band_values: low_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn range_filter_batch_inner_into(
    data: &[f64],
    combos: &[RangeFilterParams],
    kern: Kernel,
    parallel: bool,
    filter_out: &mut [f64],
    high_out: &mut [f64],
    low_out: &mut [f64],
) -> Result<(), RangeFilterError> {
    if combos.is_empty() {
        return Err(RangeFilterError::InvalidRange {
            start: 0.0,
            end: 0.0,
            step: 0.0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RangeFilterError::AllValuesNaN)?;
    let cols = data.len();

    let expected = combos
        .len()
        .checked_mul(cols)
        .ok_or(RangeFilterError::InvalidRange {
            start: 0.0,
            end: 0.0,
            step: 0.0,
        })?;
    if filter_out.len() != expected {
        return Err(RangeFilterError::OutputLengthMismatch {
            expected,
            got: filter_out.len(),
        });
    }
    if high_out.len() != expected {
        return Err(RangeFilterError::OutputLengthMismatch {
            expected,
            got: high_out.len(),
        });
    }
    if low_out.len() != expected {
        return Err(RangeFilterError::OutputLengthMismatch {
            expected,
            got: low_out.len(),
        });
    }

    
    
    let max_needed = combos
        .iter()
        .map(|c| {
            let rp = c.range_period.unwrap_or(14);
            let sp = if c.smooth_range.unwrap_or(true) {
                c.smooth_period.unwrap_or(27)
            } else {
                0
            };
            rp.max(sp)
        })
        .max()
        .unwrap_or(0);

    let valid = cols - first;
    if valid < max_needed {
        return Err(RangeFilterError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    
    for c in combos {
        let rp = c.range_period.unwrap_or(14);
        if rp == 0 || rp > cols {
            return Err(RangeFilterError::InvalidPeriod {
                period: rp,
                data_len: cols,
            });
        }
        let sr = c.smooth_range.unwrap_or(true);
        let sp = c.smooth_period.unwrap_or(27);
        if sr && (sp == 0 || sp > cols) {
            return Err(RangeFilterError::InvalidPeriod {
                period: sp,
                data_len: cols,
            });
        }
    }

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    
    let chosen_single = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => actual, 
    };

    
    
    let abs_change: Option<Vec<f64>> = {
        
        if combos.len() > 1 && first + 1 < cols {
            let mut diffs = vec![f64::NAN; cols];
            let mut prev = data[first];
            for i in (first + 1)..cols {
                let p = data[i];
                let d = p - prev;
                diffs[i] = if d < 0.0 { -d } else { d };
                prev = p;
            }
            Some(diffs)
        } else {
            None
        }
    };

    let do_row = |row: usize,
                  f_row: &mut [f64],
                  h_row: &mut [f64],
                  l_row: &mut [f64]|
     -> Result<(), RangeFilterError> {
        let p = &combos[row];
        
        let range_size = p.range_size.unwrap_or(2.618);
        let range_period = p.range_period.unwrap_or(14);
        let smooth_range = p.smooth_range.unwrap_or(true);
        let smooth_period = p.smooth_period.unwrap_or(27);

        
        if let (Kernel::Scalar, Some(ref diffs)) = (chosen_single, &abs_change) {
            range_filter_scalar_with_abs_change(
                data,
                diffs,
                range_size,
                range_period,
                smooth_range,
                smooth_period,
                first,
                f_row,
                h_row,
                l_row,
            )
        } else {
            
            range_filter_compute_into(
                data,
                range_size,
                range_period,
                smooth_range,
                smooth_period,
                first,
                chosen_single,
                f_row,
                h_row,
                l_row,
            )
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            use std::sync::Mutex;
            let err = Mutex::new(None);
            filter_out
                .par_chunks_mut(cols)
                .zip(high_out.par_chunks_mut(cols))
                .zip(low_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((f_row, h_row), l_row))| {
                    if err.lock().unwrap().is_none() {
                        if let Err(e) = do_row(row, f_row, h_row, l_row) {
                            *err.lock().unwrap() = Some(e);
                        }
                    }
                });
            if let Some(e) = err.into_inner().unwrap() {
                return Err(e);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((f_row, h_row), l_row))) in filter_out
                .chunks_mut(cols)
                .zip(high_out.chunks_mut(cols))
                .zip(low_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, f_row, h_row, l_row)?;
            }
        }
    } else {
        for (row, (((f_row, h_row), l_row))) in filter_out
            .chunks_mut(cols)
            .zip(high_out.chunks_mut(cols))
            .zip(low_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, f_row, h_row, l_row)?;
        }
    }

    
    for (row, p) in combos.iter().enumerate() {
        let rp = p.range_period.unwrap_or(14);
        let sp = if p.smooth_range.unwrap_or(true) {
            p.smooth_period.unwrap_or(27)
        } else {
            0
        };
        let warm_end = (first + rp.max(sp)).min(cols);
        let offset = row * cols;
        for i in 0..warm_end {
            let idx = offset + i;
            filter_out[idx] = f64::NAN;
            high_out[idx] = f64::NAN;
            low_out[idx] = f64::NAN;
        }
    }

    Ok(())
}


#[cfg(feature = "python")]
#[pyfunction(name = "range_filter")]
#[pyo3(signature = (data, range_size=2.618, range_period=14, smooth_range=true, smooth_period=27, kernel=None))]
pub fn range_filter_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = RangeFilterParams {
        range_size: Some(range_size),
        range_period: Some(range_period),
        smooth_range: Some(smooth_range),
        smooth_period: Some(smooth_period),
    };

    let input = RangeFilterInput::from_slice(slice_in, params);

    let (f, h, l) = py
        .allow_threads(|| {
            range_filter_with_kernel(&input, kern).map(|o| (o.filter, o.high_band, o.low_band))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((f.into_pyarray(py), h.into_pyarray(py), l.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name = "range_filter_batch")]
#[pyo3(signature = (data,
                    range_size_start=2.618, range_size_end=2.618, range_size_step=0.1,
                    range_period_start=14, range_period_end=14, range_period_step=1,
                    smooth_range=true, smooth_period=27, kernel=None))]
pub fn range_filter_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    range_size_start: f64,
    range_size_end: f64,
    range_size_step: f64,
    range_period_start: usize,
    range_period_end: usize,
    range_period_step: usize,
    smooth_range: bool,
    smooth_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;

    let sweep = RangeFilterBatchRange {
        range_size: (range_size_start, range_size_end, range_size_step),
        range_period: (range_period_start, range_period_end, range_period_step),
        smooth_range: Some(smooth_range),
        smooth_period: Some(smooth_period),
    };

    
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("range_filter_batch_py: rows*cols overflowed"))?;
    let f_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let h_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let l_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let kern = validate_kernel(kernel, true)?;

    
    let f_slice = unsafe { f_arr.as_slice_mut() }.unwrap();
    let h_slice = unsafe { h_arr.as_slice_mut() }.unwrap();
    let l_slice = unsafe { l_arr.as_slice_mut() }.unwrap();

    py.allow_threads(|| {
        
        range_filter_batch_inner_into(slice_in, &combos, kern, true, f_slice, h_slice, l_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("filter", f_arr.reshape((rows, cols))?)?;
    dict.set_item("high", h_arr.reshape((rows, cols))?)?;
    dict.set_item("low", l_arr.reshape((rows, cols))?)?;
    
    dict.set_item(
        "range_sizes",
        combos
            .iter()
            .map(|c| c.range_size.unwrap_or(2.618))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "range_periods",
        combos
            .iter()
            .map(|c| c.range_period.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_range",
        combos
            .iter()
            .map(|c| c.smooth_range.unwrap_or(true))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_periods",
        combos
            .iter()
            .map(|c| c.smooth_period.unwrap_or(27) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}


#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RangeFilterJsResult {
    pub values: Vec<f64>, 
    pub rows: usize,      
    pub cols: usize,      
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn range_filter_js(
    data: &[f64],
    range_size: Option<f64>,
    range_period: Option<usize>,
    smooth_range: Option<bool>,
    smooth_period: Option<usize>,
) -> Result<JsValue, JsValue> {
    let len = data.len();

    let params = RangeFilterParams {
        range_size: range_size.or(Some(2.618)),
        range_period: range_period.or(Some(14)),
        smooth_range: smooth_range.or(Some(true)),
        smooth_period: smooth_period.or(Some(27)),
    };
    let input = RangeFilterInput::from_slice(data, params);

    let result = range_filter(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

    
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("filter"),
        &serde_wasm_bindgen::to_value(&result.filter).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("high_band"),
        &serde_wasm_bindgen::to_value(&result.high_band).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("low_band"),
        &serde_wasm_bindgen::to_value(&result.low_band).unwrap(),
    )?;

    Ok(obj.into())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn range_filter_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn range_filter_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RangeFilterBatchConfig {
    pub range_size: (f64, f64, f64),
    pub range_period: (usize, usize, usize),
    pub smooth_range: bool,
    pub smooth_period: usize,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RangeFilterBatchJsOutput {
    pub filter: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub combos: Vec<RangeFilterParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = range_filter_batch_unified)]
pub fn range_filter_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: RangeFilterBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    
    if data.is_empty() {
        return Err(JsValue::from_str("Input data slice is empty"));
    }

    let sweep = RangeFilterBatchRange {
        range_size: cfg.range_size,
        range_period: cfg.range_period,
        smooth_range: Some(cfg.smooth_range),
        smooth_period: Some(cfg.smooth_period),
    };

    let out = range_filter_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    
    let mut values = Vec::with_capacity(out.rows * out.cols * 3);
    values.extend_from_slice(&out.filter_values);
    values.extend_from_slice(&out.high_band_values);
    values.extend_from_slice(&out.low_band_values);

    
    let obj = js_sys::Object::new();

    
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("values"),
        &serde_wasm_bindgen::to_value(&values).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("filter"),
        &serde_wasm_bindgen::to_value(&out.filter_values).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("high_band"),
        &serde_wasm_bindgen::to_value(&out.high_band_values).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("low_band"),
        &serde_wasm_bindgen::to_value(&out.low_band_values).unwrap(),
    )?;

    
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;

    Ok(obj.into())
}


#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct RangeFilterDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>, 
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl RangeFilterDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
            .as_device_ptr()
            .as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        // Producing stream is synchronized before return; omit explicit stream
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
                    "__dlpack__(copy=True) not supported for range_filter CUDA buffers",
                ));
            }
        }

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = self.rows;
        let cols = self.cols;
        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "range_filter_cuda_batch_dev")]
#[pyo3(signature = (data_f32,
                    range_size_start=2.618, range_size_end=2.618, range_size_step=0.1,
                    range_period_start=14, range_period_end=14, range_period_step=1,
                    smooth_range=true, smooth_period=27, device_id=0))]
pub fn range_filter_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    range_size_start: f64,
    range_size_end: f64,
    range_size_step: f64,
    range_period_start: usize,
    range_period_end: usize,
    range_period_step: usize,
    smooth_range: bool,
    smooth_period: usize,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let sweep = RangeFilterBatchRange {
        range_size: (range_size_start, range_size_end, range_size_step),
        range_period: (range_period_start, range_period_end, range_period_step),
        smooth_range: Some(smooth_range),
        smooth_period: Some(smooth_period),
    };
    let slice_in: &[f32] = data_f32.as_slice()?;
    let (dev_trio, combos) = py.allow_threads(|| {
        let cuda =
            CudaRangeFilter::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.range_filter_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "filter",
        Py::new(
            py,
            RangeFilterDeviceArrayF32Py {
                buf: Some(dev_trio.filter),
                rows: dev_trio.rows,
                cols: dev_trio.cols,
                _ctx: dev_trio.ctx.clone(),
                device_id: dev_trio.device_id,
            },
        )?,
    )?;
    dict.set_item(
        "high",
        Py::new(
            py,
            RangeFilterDeviceArrayF32Py {
                buf: Some(dev_trio.high),
                rows: dev_trio.rows,
                cols: dev_trio.cols,
                _ctx: dev_trio.ctx.clone(),
                device_id: dev_trio.device_id,
            },
        )?,
    )?;
    dict.set_item(
        "low",
        Py::new(
            py,
            RangeFilterDeviceArrayF32Py {
                buf: Some(dev_trio.low),
                rows: dev_trio.rows,
                cols: dev_trio.cols,
                _ctx: dev_trio.ctx.clone(),
                device_id: dev_trio.device_id,
            },
        )?,
    )?;
    // Metadata
    dict.set_item("rows", combos.len())?;
    dict.set_item("cols", slice_in.len())?;
    use numpy::IntoPyArray;
    dict.set_item(
        "range_sizes",
        combos
            .iter()
            .map(|c| c.range_size.unwrap_or(2.618))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "range_periods",
        combos
            .iter()
            .map(|c| c.range_period.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_range",
        combos
            .iter()
            .map(|c| c.smooth_range.unwrap_or(true))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_periods",
        combos
            .iter()
            .map(|c| c.smooth_period.unwrap_or(27) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "range_filter_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, range_size=2.618, range_period=14, smooth_range=true, smooth_period=27, device_id=0))]
pub fn range_filter_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let flat = data_tm_f32.as_slice()?;
    let params = RangeFilterParams {
        range_size: Some(range_size),
        range_period: Some(range_period),
        smooth_range: Some(smooth_range),
        smooth_period: Some(smooth_period),
    };
    let dev_trio = py.allow_threads(|| {
        let cuda =
            CudaRangeFilter::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.range_filter_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "filter",
        Py::new(
            py,
            RangeFilterDeviceArrayF32Py {
                buf: Some(dev_trio.filter),
                rows: dev_trio.rows,
                cols: dev_trio.cols,
                _ctx: dev_trio.ctx.clone(),
                device_id: dev_trio.device_id,
            },
        )?,
    )?;
    dict.set_item(
        "high",
        Py::new(
            py,
            RangeFilterDeviceArrayF32Py {
                buf: Some(dev_trio.high),
                rows: dev_trio.rows,
                cols: dev_trio.cols,
                _ctx: dev_trio.ctx.clone(),
                device_id: dev_trio.device_id,
            },
        )?,
    )?;
    dict.set_item(
        "low",
        Py::new(
            py,
            RangeFilterDeviceArrayF32Py {
                buf: Some(dev_trio.low),
                rows: dev_trio.rows,
                cols: dev_trio.cols,
                _ctx: dev_trio.ctx.clone(),
                device_id: dev_trio.device_id,
            },
        )?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("range_size", range_size)?;
    dict.set_item("range_period", range_period)?;
    dict.set_item("smooth_range", smooth_range)?;
    dict.set_item("smooth_period", smooth_period)?;
    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = range_filter_batch)]
pub fn range_filter_batch_js(
    data: &[f64],
    range_size_start: f64,
    range_size_end: f64,
    range_size_step: f64,
    range_period_start: usize,
    range_period_end: usize,
    range_period_step: usize,
    smooth_range: bool,
    smooth_period: usize,
) -> Result<JsValue, JsValue> {
    let sweep = RangeFilterBatchRange {
        range_size: (range_size_start, range_size_end, range_size_step),
        range_period: (range_period_start, range_period_end, range_period_step),
        smooth_range: Some(smooth_range),
        smooth_period: Some(smooth_period),
    };

    let output = range_filter_batch_slice(data, &sweep, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create a JS-friendly output structure
    let js_output = serde_json::json!({
        "filter": output.filter_values,
        "high": output.high_band_values,
        "low": output.low_band_values,
        "rows": output.rows,
        "cols": output.cols,
        "combos": output.combos.iter().map(|c| {
            serde_json::json!({
                "range_size": c.range_size,
                "range_period": c.range_period,
                "smooth_range": c.smooth_range,
                "smooth_period": c.smooth_period
            })
        }).collect::<Vec<_>>()
    });

    serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = range_filter_batch_into)]
pub fn range_filter_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64, // expects rows * cols * 3 elements
    len: usize,
    range_size_start: f64,
    range_size_end: f64,
    range_size_step: f64,
    range_period_start: usize,
    range_period_end: usize,
    range_period_step: usize,
    smooth_range: bool,
    smooth_period: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to range_filter_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = RangeFilterBatchRange {
            range_size: (range_size_start, range_size_end, range_size_step),
            range_period: (range_period_start, range_period_end, range_period_step),
            smooth_range: Some(smooth_range),
            smooth_period: Some(smooth_period),
        };

        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        // Output is organized as [filter_values..., high_values..., low_values...]
        let total = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| {
                JsValue::from_str("range_filter_batch_into: rows*cols*3 overflowed usize")
            })?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (filter_out, rest) = out.split_at_mut(rows * cols);
        let (high_out, low_out) = rest.split_at_mut(rows * cols);

        range_filter_batch_inner_into(
            data,
            &combos,
            detect_best_kernel(),
            false,
            filter_out,
            high_out,
            low_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = range_filter_into_flat)]
pub fn range_filter_into_flat(
    data_ptr: *const f64,
    out_ptr: *mut f64, // expects length = 3*len
    len: usize,
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 3 * len);

        let params = RangeFilterParams {
            range_size: Some(range_size),
            range_period: Some(range_period),
            smooth_range: Some(smooth_range),
            smooth_period: Some(smooth_period),
        };
        let input = RangeFilterInput::from_slice(data, params);

        // Compute into three temporary views inside the single buffer without extra copies
        let (f, rest) = out.split_at_mut(len);
        let (h, l) = rest.split_at_mut(len);
        range_filter_into_slice(f, h, l, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// Streaming support
// Decision: Streaming uses O(1) state with prev_price + FMA EMA; branchless clamp for stability.
#[derive(Debug, Clone)]
pub struct RangeFilterStream {
    range_size: f64,
    range_period: usize,
    smooth_range: bool,
    smooth_period: usize,

    // EMA coefficients (precomputed)
    alpha_ac: f64,
    one_minus_alpha_ac: f64,
    alpha_range: f64,
    one_minus_alpha_range: f64,

    // Running EMA state
    ac_ema: f64,
    ac_initialized: bool,

    range_ema: f64,
    range_initialized: bool,

    // Previous samples
    prev_price: f64,
    have_prev_price: bool,

    // Filter state
    prev_filter: f64,
    filter_initialized: bool,
}

impl RangeFilterStream {
    #[inline(always)]
    pub fn try_new(params: RangeFilterParams) -> Result<Self, RangeFilterError> {
        let range_size = params.range_size.unwrap_or(2.618);
        if !(range_size.is_finite() && range_size > 0.0) {
            return Err(RangeFilterError::InvalidRangeSize { range_size });
        }

        let range_period = params.range_period.unwrap_or(14);
        if range_period == 0 {
            return Err(RangeFilterError::InvalidPeriod {
                period: range_period,
                data_len: 0,
            });
        }

        let smooth_range = params.smooth_range.unwrap_or(true);
        let smooth_period = params.smooth_period.unwrap_or(27);
        if smooth_range && smooth_period == 0 {
            return Err(RangeFilterError::InvalidPeriod {
                period: smooth_period,
                data_len: 0,
            });
        }

        // EMA coefficients
        let alpha_ac = 2.0 / (range_period as f64 + 1.0);
        let alpha_range = if smooth_range {
            2.0 / (smooth_period as f64 + 1.0)
        } else {
            0.0
        };

        Ok(Self {
            range_size,
            range_period,
            smooth_range,
            smooth_period,

            alpha_ac,
            one_minus_alpha_ac: 1.0 - alpha_ac,
            alpha_range,
            one_minus_alpha_range: 1.0 - alpha_range,

            ac_ema: 0.0,
            ac_initialized: false,

            range_ema: 0.0,
            range_initialized: false,

            prev_price: f64::NAN,
            have_prev_price: false,

            prev_filter: f64::NAN,
            filter_initialized: false,
        })
    }

    #[inline(always)]
    fn ema_step(prev: f64, x: f64, alpha: f64) -> f64 {
        // FMA form: prev + alpha*(x - prev) → one rounding on FMA hardware
        (x - prev).mul_add(alpha, prev)
    }

    #[inline(always)]
    fn clamp_branchless(x: f64, lo: f64, hi: f64) -> f64 {
        // Equivalent to clamp without panic and with predictable control flow
        hi.min(lo.max(x))
    }

    /// Push a new price. Returns (filter, high_band, low_band) once enough state exists,
    /// otherwise None during warmup (needs at least 2 non-NaN prices).
    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<(f64, f64, f64)> {
        if !price.is_finite() {
            return None;
        }

        if !self.have_prev_price {
            self.prev_price = price;
            self.have_prev_price = true;
            return None; // need a delta
        }

        // --- Update Average Change EMA (seed on first difference) ---
        let abs_change = (price - self.prev_price).abs();
        if !self.ac_initialized {
            self.ac_ema = abs_change;
            self.ac_initialized = true;
        } else {
            self.ac_ema = Self::ema_step(self.ac_ema, abs_change, self.alpha_ac);
        }

        // Not enough info yet (shouldn't happen after ac_initialized, but keep symmetry)
        if !self.ac_initialized {
            self.prev_price = price;
            return None;
        }

        
        let mut range = self.ac_ema * self.range_size;
        if self.smooth_range {
            if !self.range_initialized {
                self.range_ema = range;
                self.range_initialized = true;
            } else {
                self.range_ema = Self::ema_step(self.range_ema, range, self.alpha_range);
            }
            range = self.range_ema;
        }

        
        if !self.filter_initialized {
            
            self.prev_filter = price;
            self.filter_initialized = true;
        }

        
        let lo = price - range;
        let hi = price + range;
        let current = Self::clamp_branchless(self.prev_filter, lo, hi);

        
        self.prev_filter = current;
        self.prev_price = price;

        Some((current, current + range, current - range))
    }

    /// Returns the last (filter, high, low) if initialized.
    #[inline(always)]
    pub fn current_value(&self) -> Option<(f64, f64, f64)> {
        if !self.filter_initialized || !self.ac_initialized {
            return None;
        }
        let range = if self.smooth_range && self.range_initialized {
            self.range_ema
        } else {
            self.ac_ema * self.range_size
        };
        Some((
            self.prev_filter,
            self.prev_filter + range,
            self.prev_filter - range,
        ))
    }
}


impl RangeFilterBuilder {
    #[inline(always)]
    pub fn into_stream(self) -> Result<RangeFilterStream, RangeFilterError> {
        let params = RangeFilterParams {
            range_size: self.range_size,
            range_period: self.range_period,
            smooth_range: self.smooth_range,
            smooth_period: self.smooth_period,
        };
        RangeFilterStream::try_new(params)
    }
}


#[cfg(feature = "python")]
#[pyclass(name = "RangeFilterStream")]
pub struct RangeFilterStreamPy {
    stream: RangeFilterStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RangeFilterStreamPy {
    #[new]
    fn new(
        range_size: f64,
        range_period: usize,
        smooth_range: bool,
        smooth_period: usize,
    ) -> PyResult<Self> {
        let params = RangeFilterParams {
            range_size: Some(range_size),
            range_period: Some(range_period),
            smooth_range: Some(smooth_range),
            smooth_period: Some(smooth_period),
        };
        let stream =
            RangeFilterStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(RangeFilterStreamPy { stream })
    }

    fn update(&mut self, price: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(price)
    }

    fn current_value(&self) -> Option<(f64, f64, f64)> {
        self.stream.current_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::{read_candles_from_csv, Candles};
    use paste::paste;
    use std::error::Error;

    
    fn check_range_filter_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let input = RangeFilterInput::with_default_candles(&candles);
        let result = range_filter_with_kernel(&input, kernel)?;

        
        let n = result.filter.len();
        let last_5_filter: Vec<f64> = result.filter[(n - 5)..].to_vec();
        let last_5_high: Vec<f64> = result.high_band[(n - 5)..].to_vec();
        let last_5_low: Vec<f64> = result.low_band[(n - 5)..].to_vec();

        let expected_filter = vec![
            59_589.73987817684,
            59_589.73987817684,
            59_589.73987817684,
            59_589.73987817684,
            59_589.73987817684,
        ];

        let expected_high = vec![
            60_935.63924911415,
            60_906.58379951138,
            60_874.2002431993,
            60_838.79850154794,
            60_810.879398758305,
        ];

        let expected_low = vec![
            58_243.84050723953,
            58_272.8959568423,
            58_305.27951315438,
            58_340.68125480574,
            58_368.60035759538,
        ];

        let tolerance = 1e-10; 

        
        println!("Actual Filter values: {:?}", last_5_filter);
        println!("Actual High Band values: {:?}", last_5_high);
        println!("Actual Low Band values: {:?}", last_5_low);

        
        for (i, &val) in last_5_filter.iter().enumerate() {
            let diff = (val - expected_filter[i]).abs();
            assert!(
                diff < tolerance,
                "[{}] Filter[{}] mismatch: expected {}, got {} (diff: {})",
                test,
                i,
                expected_filter[i],
                val,
                diff
            );
        }

        
        for (i, &val) in last_5_high.iter().enumerate() {
            let diff = (val - expected_high[i]).abs();
            assert!(
                diff < tolerance,
                "[{}] High Band[{}] mismatch: expected {}, got {} (diff: {})",
                test,
                i,
                expected_high[i],
                val,
                diff
            );
        }

        
        for (i, &val) in last_5_low.iter().enumerate() {
            let diff = (val - expected_low[i]).abs();
            assert!(
                diff < tolerance,
                "[{}] Low Band[{}] mismatch: expected {}, got {} (diff: {})",
                test,
                i,
                expected_low[i],
                val,
                diff
            );
        }

        Ok(())
    }

    
    fn check_range_filter_default_candles(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = RangeFilterBuilder::new().kernel(kernel).apply(&c)?;
        assert_eq!(out.filter.len(), c.close.len());
        assert_eq!(out.high_band.len(), c.close.len());
        assert_eq!(out.low_band.len(), c.close.len());
        Ok(())
    }

    
    fn check_range_filter_empty_input(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let params = RangeFilterParams::default();
        let input = RangeFilterInput::from_slice(&[], params);
        let result = range_filter_with_kernel(&input, kernel);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RangeFilterError::EmptyInputData
        ));
        Ok(())
    }

    
    fn check_range_filter_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let data = vec![f64::NAN; 10];
        let params = RangeFilterParams::default();
        let input = RangeFilterInput::from_slice(&data, params);
        let result = range_filter_with_kernel(&input, kernel);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RangeFilterError::AllValuesNaN
        ));
        Ok(())
    }

    
    fn check_range_filter_invalid_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = RangeFilterParams {
            range_period: Some(10), 
            ..Default::default()
        };
        let input = RangeFilterInput::from_slice(&data, params);
        let result = range_filter_with_kernel(&input, kernel);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RangeFilterError::InvalidPeriod { .. }
        ));
        Ok(())
    }

    
    fn check_range_filter_into_slice(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let data = (0..256).map(|i| i as f64).collect::<Vec<_>>();
        let params = RangeFilterParams::default();
        let input = RangeFilterInput::from_slice(&data, params);

        let mut f = vec![999.0; data.len()];
        let mut h = vec![999.0; data.len()];
        let mut l = vec![999.0; data.len()];

        range_filter_into_slice(&mut f, &mut h, &mut l, &input, kernel)?;

        
        let warmup = 27; 
        for i in 0..warmup {
            assert!(
                f[i].is_nan(),
                "[{}] filter[{}] should be NaN during warmup",
                test,
                i
            );
            assert!(
                h[i].is_nan(),
                "[{}] high[{}] should be NaN during warmup",
                test,
                i
            );
            assert!(
                l[i].is_nan(),
                "[{}] low[{}] should be NaN during warmup",
                test,
                i
            );
        }

        
        for i in warmup..data.len() {
            assert!(
                !f[i].is_nan(),
                "[{}] filter[{}] should not be NaN after warmup",
                test,
                i
            );
            assert!(
                !h[i].is_nan(),
                "[{}] high[{}] should not be NaN after warmup",
                test,
                i
            );
            assert!(
                !l[i].is_nan(),
                "[{}] low[{}] should not be NaN after warmup",
                test,
                i
            );
        }
        Ok(())
    }

    
    fn check_range_filter_kernel_parity(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let data = (0..100)
            .map(|i| (i as f64).sin() * 100.0 + 500.0)
            .collect::<Vec<_>>();
        let params = RangeFilterParams::default();
        let input = RangeFilterInput::from_slice(&data, params);

        
        let scalar_result = range_filter_with_kernel(&input, Kernel::Scalar)?;

        
        let kernel_result = range_filter_with_kernel(&input, kernel)?;

        
        for i in 0..data.len() {
            if scalar_result.filter[i].is_nan() {
                assert!(
                    kernel_result.filter[i].is_nan(),
                    "[{}] filter[{}] NaN mismatch",
                    test,
                    i
                );
            } else {
                let diff = (scalar_result.filter[i] - kernel_result.filter[i]).abs();
                assert!(
                    diff < 1e-10,
                    "[{}] filter[{}] mismatch: {} vs {} (diff: {})",
                    test,
                    i,
                    scalar_result.filter[i],
                    kernel_result.filter[i],
                    diff
                );
            }
        }
        Ok(())
    }

    
    fn check_range_filter_streaming(test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = (0..100)
            .map(|i| (i as f64).sin() * 100.0 + 500.0)
            .collect::<Vec<_>>();
        let params = RangeFilterParams::default();

        
        let input = RangeFilterInput::from_slice(&data, params.clone());
        let batch_result = range_filter(&input)?;

        
        let mut stream = RangeFilterStream::try_new(params)?;
        let mut streaming_filter = Vec::new();
        let mut streaming_high = Vec::new();
        let mut streaming_low = Vec::new();

        for &price in &data {
            if let Some((f, h, l)) = stream.update(price) {
                streaming_filter.push(f);
                streaming_high.push(h);
                streaming_low.push(l);
            } else {
                streaming_filter.push(f64::NAN);
                streaming_high.push(f64::NAN);
                streaming_low.push(f64::NAN);
            }
        }

        
        
        let start = 30; 
        for i in start..streaming_filter.len().min(batch_result.filter.len()) {
            
            assert!(
                !streaming_filter[i].is_nan(),
                "[{}] Stream filter[{}] is NaN",
                test,
                i
            );
            assert!(
                !streaming_high[i].is_nan(),
                "[{}] Stream high[{}] is NaN",
                test,
                i
            );
            assert!(
                !streaming_low[i].is_nan(),
                "[{}] Stream low[{}] is NaN",
                test,
                i
            );

            
            assert!(
                streaming_high[i] >= streaming_filter[i],
                "[{}] High band should be >= filter at [{}]",
                test,
                i
            );
            assert!(
                streaming_filter[i] >= streaming_low[i],
                "[{}] Filter should be >= low band at [{}]",
                test,
                i
            );
        }
        Ok(())
    }

    
    fn check_range_filter_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = RangeFilterBuilder::new().kernel(kernel).apply(&c)?;

        for &v in out
            .filter
            .iter()
            .chain(out.high_band.iter())
            .chain(out.low_band.iter())
        {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                "[{}] poison pattern leaked: 0x{:016X}",
                test,
                b
            );
        }
        Ok(())
    }

    
    fn check_rf_partial_params(_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let params = RangeFilterParams {
            range_size: None,
            range_period: None,
            smooth_range: None,
            smooth_period: None,
        };
        let input = RangeFilterInput::from_candles(&c, "close", params);
        let out = range_filter_with_kernel(&input, k)?;
        assert_eq!(out.filter.len(), c.close.len());
        Ok(())
    }

    fn check_rf_default_candles(_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = RangeFilterInput::with_default_candles(&c);
        if let RangeFilterData::Candles { source, .. } = input.data {
            assert_eq!(source, "close");
        }
        let out = range_filter_with_kernel(&input, k)?;
        assert_eq!(out.filter.len(), c.close.len());
        Ok(())
    }

    fn check_rf_zero_or_bad_period(_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        let d = [1.0, 2.0, 3.0];
        for p in [0usize, 10usize] {
            let params = RangeFilterParams {
                range_size: Some(2.618),
                range_period: Some(p),
                smooth_range: Some(true),
                smooth_period: Some(27),
            };
            let input = RangeFilterInput::from_slice(&d, params);
            assert!(range_filter_with_kernel(&input, k).is_err());
        }
        Ok(())
    }

    fn check_rf_invalid_range_size(_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        let d = [1.0, 2.0, 3.0, 4.0];
        for rs in [0.0, f64::NAN, f64::INFINITY, -1.0] {
            let params = RangeFilterParams {
                range_size: Some(rs),
                range_period: Some(2),
                smooth_range: Some(false),
                smooth_period: Some(1),
            };
            let input = RangeFilterInput::from_slice(&d, params);
            assert!(matches!(
                range_filter_with_kernel(&input, k),
                Err(RangeFilterError::InvalidRangeSize { .. })
            ));
        }
        Ok(())
    }

    fn check_rf_nan_handling(_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = RangeFilterInput::from_candles(&c, "close", RangeFilterParams::default());
        let out = range_filter_with_kernel(&input, k)?;
        
        let first_non_nan = out
            .filter
            .iter()
            .position(|v| !v.is_nan())
            .unwrap_or(out.filter.len());
        
        assert!(
            first_non_nan > 0,
            "Should have warmup period with NaN values"
        );
        
        assert!(out.filter[..first_non_nan].iter().all(|v| v.is_nan()));
        assert!(out.high_band[..first_non_nan].iter().all(|v| v.is_nan()));
        assert!(out.low_band[..first_non_nan].iter().all(|v| v.is_nan()));
        
        if first_non_nan < out.filter.len() {
            assert!(out.filter[first_non_nan..].iter().all(|v| v.is_finite()));
        }
        Ok(())
    }

    fn check_rf_streaming_parity(_name: &str, _k: Kernel) -> Result<(), Box<dyn Error>> {
        
        
        let data = (0..100)
            .map(|i| (i as f64).sin() * 100.0 + 500.0)
            .collect::<Vec<_>>();
        let p = RangeFilterParams::default();

        let mut s = RangeFilterStream::try_new(p)?;
        let mut stream_filter = Vec::new();
        let mut stream_high = Vec::new();
        let mut stream_low = Vec::new();

        for &price in &data {
            if let Some((f, h, l)) = s.update(price) {
                stream_filter.push(f);
                stream_high.push(h);
                stream_low.push(l);
            } else {
                stream_filter.push(f64::NAN);
                stream_high.push(f64::NAN);
                stream_low.push(f64::NAN);
            }
        }

        
        let start = 30; 
        for i in start..stream_filter.len() {
            
            assert!(!stream_filter[i].is_nan(), "Stream filter[{}] is NaN", i);
            assert!(!stream_high[i].is_nan(), "Stream high[{}] is NaN", i);
            assert!(!stream_low[i].is_nan(), "Stream low[{}] is NaN", i);

            
            assert!(
                stream_high[i] >= stream_filter[i],
                "High band should be >= filter at [{}]",
                i
            );
            assert!(
                stream_filter[i] >= stream_low[i],
                "Filter should be >= low band at [{}]",
                i
            );
        }
        Ok(())
    }

    fn check_rf_batch_default_row(_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = RangeFilterBatchBuilder::new()
            .kernel(k)
            .apply_candles(&c, "close")?;
        let def = RangeFilterParams::default();
        let row = out.filter_values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[test]
    fn test_range_filter_into_matches_api() -> Result<(), Box<dyn Error>> {
        
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let input = RangeFilterInput::with_default_candles(&candles);

        
        let baseline = range_filter(&input)?;

        
        let n = candles.close.len();
        let mut f = vec![0.0; n];
        let mut h = vec![0.0; n];
        let mut l = vec![0.0; n];
        range_filter_into(&input, &mut f, &mut h, &mut l)?;

        assert_eq!(baseline.filter.len(), n);
        assert_eq!(baseline.high_band.len(), n);
        assert_eq!(baseline.low_band.len(), n);

        
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline.filter[i], f[i]),
                "filter[{}]: {:?} vs {:?}",
                i,
                baseline.filter[i],
                f[i]
            );
            assert!(
                eq_or_both_nan(baseline.high_band[i], h[i]),
                "high[{}]: {:?} vs {:?}",
                i,
                baseline.high_band[i],
                h[i]
            );
            assert!(
                eq_or_both_nan(baseline.low_band[i], l[i]),
                "low[{}]: {:?} vs {:?}",
                i,
                baseline.low_band[i],
                l[i]
            );
        }

        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_range_filter_property(
        _name: &str,
        _k: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        let strat = (1usize..=64).prop_flat_map(|rp| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    rp..400,
                ),
                Just(rp),
                any::<bool>(),
                1usize..=64,
                (0.5f64..5.0f64),
            )
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, rp, smooth, sp, rs)| {
                let p = RangeFilterParams {
                    range_size: Some(rs),
                    range_period: Some(rp),
                    smooth_range: Some(smooth),
                    smooth_period: Some(sp),
                };
                let input = RangeFilterInput::from_slice(&data, p);
                let a = range_filter_with_kernel(&input, Kernel::Scalar).unwrap();

                
                let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warm = first + rp.max(if smooth { sp } else { 0 });
                for i in warm..data.len() {
                    prop_assert!(a.low_band[i] <= a.filter[i] && a.filter[i] <= a.high_band[i]);
                    prop_assert!(a.filter[i].is_finite());
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    fn check_range_filter_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        
        let first = range_filter_with_kernel(&RangeFilterInput::with_default_candles(&c), kernel)?;

        
        let params = RangeFilterParams::default();
        let second =
            range_filter_with_kernel(&RangeFilterInput::from_slice(&first.filter, params), kernel)?;

        
        assert_eq!(second.filter.len(), first.filter.len());

        
        let first_valid = first
            .filter
            .iter()
            .position(|v| !v.is_nan())
            .unwrap_or(first.filter.len());

        
        
        let second_valid = second
            .filter
            .iter()
            .position(|v| !v.is_nan())
            .unwrap_or(second.filter.len());

        
        assert!(
            second_valid > 0,
            "[{}] Should have warmup NaNs in reinput",
            test
        );

        
        for i in 0..second_valid {
            assert!(
                second.filter[i].is_nan(),
                "[{}] reinput should be NaN at {} but got {}",
                test,
                i,
                second.filter[i]
            );
        }
        Ok(())
    }

    
    macro_rules! generate_all_range_filter_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]()  {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        };
    }

    generate_all_range_filter_tests!(
        check_range_filter_accuracy,
        check_range_filter_default_candles,
        check_range_filter_empty_input,
        check_range_filter_all_nan,
        check_range_filter_invalid_period,
        check_range_filter_into_slice,
        check_range_filter_kernel_parity,
        check_range_filter_streaming,
        check_range_filter_no_poison,
        check_rf_partial_params,
        check_rf_default_candles,
        check_rf_zero_or_bad_period,
        check_rf_invalid_range_size,
        check_rf_nan_handling,
        check_rf_streaming_parity,
        check_rf_batch_default_row,
        check_range_filter_reinput
    );

    #[cfg(feature = "proptest")]
    generate_all_range_filter_tests!(check_range_filter_property);

    
    fn check_range_filter_batch_default(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = RangeFilterBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&c.close)?;

        let def = RangeFilterParams::default();
        let row = output.filter_values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    fn check_range_filter_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let data = (0..50)
            .map(|i| (i as f64).sin() * 100.0 + 500.0)
            .collect::<Vec<_>>();

        let sweep = RangeFilterBatchRange {
            range_size: (2.0, 3.0, 0.5),
            range_period: (10, 20, 5),
            smooth_range: Some(true),
            smooth_period: Some(15),
        };

        let output = range_filter_batch_inner(&data, &sweep, kernel, false)?;

        
        assert_eq!(output.rows, 9);
        assert_eq!(output.cols, data.len());

        Ok(())
    }

    fn check_range_filter_batch_parallel(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        #[cfg(target_arch = "wasm32")]
        {
            
            eprintln!("[{}] skipped (parallel not supported on WASM)", test);
            return Ok(());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let data = (0..100)
                .map(|i| (i as f64).sin() * 100.0 + 500.0)
                .collect::<Vec<_>>();

            let sweep = RangeFilterBatchRange::default();

            
            let seq = range_filter_batch_slice(&data, &sweep, kernel)?;
            let par = range_filter_batch_par_slice(&data, &sweep, kernel)?;

            
            assert_eq!(seq.filter_values.len(), par.filter_values.len());
            for i in 0..seq.filter_values.len() {
                if seq.filter_values[i].is_nan() {
                    assert!(
                        par.filter_values[i].is_nan(),
                        "Filter NaN mismatch at {}",
                        i
                    );
                } else {
                    assert!(
                        (seq.filter_values[i] - par.filter_values[i]).abs() < 1e-10,
                        "Filter value mismatch at {}: {} vs {}",
                        i,
                        seq.filter_values[i],
                        par.filter_values[i]
                    );
                }
            }

            assert_eq!(seq.high_band_values.len(), par.high_band_values.len());
            for i in 0..seq.high_band_values.len() {
                if seq.high_band_values[i].is_nan() {
                    assert!(
                        par.high_band_values[i].is_nan(),
                        "High band NaN mismatch at {}",
                        i
                    );
                } else {
                    assert!(
                        (seq.high_band_values[i] - par.high_band_values[i]).abs() < 1e-10,
                        "High band value mismatch at {}: {} vs {}",
                        i,
                        seq.high_band_values[i],
                        par.high_band_values[i]
                    );
                }
            }

            assert_eq!(seq.low_band_values.len(), par.low_band_values.len());
            for i in 0..seq.low_band_values.len() {
                if seq.low_band_values[i].is_nan() {
                    assert!(
                        par.low_band_values[i].is_nan(),
                        "Low band NaN mismatch at {}",
                        i
                    );
                } else {
                    assert!(
                        (seq.low_band_values[i] - par.low_band_values[i]).abs() < 1e-10,
                        "Low band value mismatch at {}: {} vs {}",
                        i,
                        seq.low_band_values[i],
                        par.low_band_values[i]
                    );
                }
            }
        }

        Ok(())
    }

    
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }

                #[test]
                fn [<$fn_name _auto>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto>]), Kernel::Auto);
                }
            }
        };
    }

    fn check_range_filter_batch_no_poison(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = RangeFilterBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&c.close)?;

        
        for v in out
            .filter_values
            .iter()
            .chain(&out.high_band_values)
            .chain(&out.low_band_values)
        {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                "[{}] Batch poison pattern leaked: 0x{:016X}",
                test,
                b
            );
        }
        Ok(())
    }

    gen_batch_tests!(check_range_filter_batch_default);
    gen_batch_tests!(check_range_filter_batch_sweep);
    gen_batch_tests!(check_range_filter_batch_parallel);
    gen_batch_tests!(check_range_filter_batch_no_poison);
}
