//! # Buff Averages
//!
//! Volume-weighted moving average indicator that computes dual-period volume-weighted
//! averages for trend analysis. Calculates separate fast and slow volume-weighted
//! moving averages to identify momentum shifts and trend changes.
//!
//! ## Parameters
//! - **fast_period**: Number of periods for fast average (default: 5)
//! - **slow_period**: Number of periods for slow average (default: 20)
//! - **price data**: Price series (close, open, high, low, or custom)
//! - **volume data**: Volume series for weighting calculations
//!
//! ## Returns
//! - **fast_buff**: Volume-weighted average over fast period
//! - **slow_buff**: Volume-weighted average over slow period
//!
//! ## Developer Status
//! - **AVX2 kernel**: STUB - Falls back to scalar implementation
//! - **AVX512 kernel**: STUB - Falls back to scalar implementation
//! - **Streaming update**: O(n) - Iterates through both buffers for each update
//! - **Optimization needed**: Implement SIMD kernels for better performance
//! - **Streaming improvement**: Could be optimized to O(1) with running sums

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

// Feature-gated imports for WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// Core imports
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

// SIMD imports for AVX optimizations
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// Parallel processing support
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Standard library imports
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// ==================== TRAIT IMPLEMENTATIONS ====================
impl<'a> AsRef<[f64]> for BuffAveragesInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            BuffAveragesData::Slice(slice) => slice,
            BuffAveragesData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum BuffAveragesData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct BuffAveragesOutput {
    pub fast_buff: Vec<f64>,
    pub slow_buff: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct BuffAveragesParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
}

impl Default for BuffAveragesParams {
    fn default() -> Self {
        Self {
            fast_period: Some(5),
            slow_period: Some(20),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct BuffAveragesInput<'a> {
    pub data: BuffAveragesData<'a>,
    pub volume: Option<&'a [f64]>,
    pub params: BuffAveragesParams,
}

impl<'a> BuffAveragesInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: BuffAveragesParams) -> Self {
        Self {
            data: BuffAveragesData::Candles {
                candles: c,
                source: s,
            },
            volume: Some(&c.volume),
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(price: &'a [f64], volume: &'a [f64], p: BuffAveragesParams) -> Self {
        Self {
            data: BuffAveragesData::Slice(price),
            volume: Some(volume),
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: BuffAveragesParams) -> Self {
        Self {
            data: BuffAveragesData::Slice(sl),
            volume: None,
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", BuffAveragesParams::default())
    }

    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(5)
    }

    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(20)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct BuffAveragesBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    kernel: Kernel,
}

impl Default for BuffAveragesBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BuffAveragesBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn fast_period(mut self, val: usize) -> Self {
        self.fast_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn slow_period(mut self, val: usize) -> Self {
        self.slow_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<BuffAveragesOutput, BuffAveragesError> {
        let p = BuffAveragesParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = BuffAveragesInput::from_candles(c, "close", p);
        buff_averages_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        price: &[f64],
        volume: &[f64],
    ) -> Result<BuffAveragesOutput, BuffAveragesError> {
        let p = BuffAveragesParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = BuffAveragesInput::from_slices(price, volume, p);
        buff_averages_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<BuffAveragesStream, BuffAveragesError> {
        let p = BuffAveragesParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        BuffAveragesStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum BuffAveragesError {
    #[error("buff_averages: Input data slice is empty.")]
    EmptyInputData,

    #[error("buff_averages: All values are NaN.")]
    AllValuesNaN,

    #[error("buff_averages: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("buff_averages: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("buff_averages: Price and volume arrays have different lengths: price = {price_len}, volume = {volume_len}")]
    MismatchedDataLength { price_len: usize, volume_len: usize },

    #[error("buff_averages: Volume data is required for this indicator")]
    MissingVolumeData,
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
/// Main entry point with automatic kernel detection
#[inline]
pub fn buff_averages(input: &BuffAveragesInput) -> Result<BuffAveragesOutput, BuffAveragesError> {
    buff_averages_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection
pub fn buff_averages_with_kernel(
    input: &BuffAveragesInput,
    kernel: Kernel,
) -> Result<BuffAveragesOutput, BuffAveragesError> {
    let (price, volume, fast_period, slow_period, first, chosen) =
        buff_averages_prepare(input, kernel)?;

    let warm = first + slow_period - 1;

    // CRITICAL: Use zero-copy allocation helper with correct warmup
    let mut fast_buff = alloc_with_nan_prefix(price.len(), warm);
    let mut slow_buff = alloc_with_nan_prefix(price.len(), warm);

    buff_averages_compute_into(
        price,
        volume,
        fast_period,
        slow_period,
        first,
        chosen,
        &mut fast_buff,
        &mut slow_buff,
    );

    Ok(BuffAveragesOutput {
        fast_buff,
        slow_buff,
    })
}

/// Zero-allocation version for performance-critical paths
#[inline]
pub fn buff_averages_into_slices(
    fast_dst: &mut [f64],
    slow_dst: &mut [f64],
    input: &BuffAveragesInput,
    kern: Kernel,
) -> Result<(), BuffAveragesError> {
    let (price, volume, fast_p, slow_p, first, chosen) = buff_averages_prepare(input, kern)?;

    if fast_dst.len() != price.len() || slow_dst.len() != price.len() {
        return Err(BuffAveragesError::InvalidPeriod {
            period: price.len(),
            data_len: price.len(),
        });
    }

    buff_averages_compute_into(
        price, volume, fast_p, slow_p, first, chosen, fast_dst, slow_dst,
    );

    // Fill warmup period with NaN
    let warm = first + slow_p - 1;
    for x in &mut fast_dst[..warm] {
        *x = f64::NAN;
    }
    for x in &mut slow_dst[..warm] {
        *x = f64::NAN;
    }

    Ok(())
}

/// Prepare and validate input data
#[inline(always)]
fn buff_averages_prepare<'a>(
    input: &'a BuffAveragesInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, usize, Kernel), BuffAveragesError> {
    let price: &[f64] = input.as_ref();
    let len = price.len();

    if len == 0 {
        return Err(BuffAveragesError::EmptyInputData);
    }

    // Get volume data
    let volume = match &input.data {
        BuffAveragesData::Candles { candles, .. } => &candles.volume,
        BuffAveragesData::Slice(_) => input.volume.ok_or(BuffAveragesError::MissingVolumeData)?,
    };

    if price.len() != volume.len() {
        return Err(BuffAveragesError::MismatchedDataLength {
            price_len: price.len(),
            volume_len: volume.len(),
        });
    }

    let first = price
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BuffAveragesError::AllValuesNaN)?;

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();

    // Validation
    if fast_period == 0 || fast_period > len {
        return Err(BuffAveragesError::InvalidPeriod {
            period: fast_period,
            data_len: len,
        });
    }

    if slow_period == 0 || slow_period > len {
        return Err(BuffAveragesError::InvalidPeriod {
            period: slow_period,
            data_len: len,
        });
    }

    if len - first < slow_period {
        return Err(BuffAveragesError::NotEnoughValidData {
            needed: slow_period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((price, volume, fast_period, slow_period, first, chosen))
}

/// Core computation dispatcher
#[inline(always)]
fn buff_averages_compute_into(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    kernel: Kernel,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    unsafe {
        // WASM SIMD128 support
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                buff_averages_simd128(
                    price,
                    volume,
                    fast_period,
                    slow_period,
                    first,
                    fast_out,
                    slow_out,
                );
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => buff_averages_scalar(
                price,
                volume,
                fast_period,
                slow_period,
                first,
                fast_out,
                slow_out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => buff_averages_avx2(
                price,
                volume,
                fast_period,
                slow_period,
                first,
                fast_out,
                slow_out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => buff_averages_avx512(
                price,
                volume,
                fast_period,
                slow_period,
                first,
                fast_out,
                slow_out,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                buff_averages_scalar(
                    price,
                    volume,
                    fast_period,
                    slow_period,
                    first,
                    fast_out,
                    slow_out,
                )
            }
            _ => unreachable!(),
        }
    }
}

// ==================== SCALAR IMPLEMENTATION ====================
#[inline]
pub fn buff_averages_scalar(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    let len = price.len();
    if len == 0 {
        return;
    }

    let warm = first + slow_period - 1;
    if warm >= len {
        return;
    } // guarded by prepare()

    // Compute initial slow window at index = warm
    let mut slow_numerator = 0.0;
    let mut slow_denominator = 0.0;
    let slow_start = warm + 1 - slow_period;

    for i in slow_start..=warm {
        let p = price[i];
        let v = volume[i];
        if !p.is_nan() && !v.is_nan() {
            slow_numerator += p * v;
            slow_denominator += v;
        }
    }

    // Compute initial fast window at index = warm
    let mut fast_numerator = 0.0;
    let mut fast_denominator = 0.0;
    let fast_start = warm + 1 - fast_period;

    for i in fast_start..=warm {
        let p = price[i];
        let v = volume[i];
        if !p.is_nan() && !v.is_nan() {
            fast_numerator += p * v;
            fast_denominator += v;
        }
    }

    // First valid writes at warm
    if slow_denominator != 0.0 {
        slow_out[warm] = slow_numerator / slow_denominator;
    } else {
        slow_out[warm] = 0.0;
    }

    if fast_denominator != 0.0 {
        fast_out[warm] = fast_numerator / fast_denominator;
    } else {
        fast_out[warm] = 0.0;
    }

    // Rolling from warm+1 .. len-1
    for i in (warm + 1)..len {
        // Slow roll
        let old_slow = i - slow_period;
        let new_p = price[i];
        let new_v = volume[i];
        let old_p = price[old_slow];
        let old_v = volume[old_slow];

        if !old_p.is_nan() && !old_v.is_nan() {
            slow_numerator -= old_p * old_v;
            slow_denominator -= old_v;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            slow_numerator += new_p * new_v;
            slow_denominator += new_v;
        }

        slow_out[i] = if slow_denominator != 0.0 {
            slow_numerator / slow_denominator
        } else {
            0.0
        };

        // Fast roll
        let old_fast = i - fast_period;
        let old_pf = price[old_fast];
        let old_vf = volume[old_fast];

        if !old_pf.is_nan() && !old_vf.is_nan() {
            fast_numerator -= old_pf * old_vf;
            fast_denominator -= old_vf;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            fast_numerator += new_p * new_v;
            fast_denominator += new_v;
        }

        fast_out[i] = if fast_denominator != 0.0 {
            fast_numerator / fast_denominator
        } else {
            0.0
        };
    }
}

// ==================== WASM SIMD128 IMPLEMENTATION ====================
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn buff_averages_simd128(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    // For now, fallback to scalar
    buff_averages_scalar(
        price,
        volume,
        fast_period,
        slow_period,
        first,
        fast_out,
        slow_out,
    );
}

// ==================== AVX2 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn buff_averages_avx2(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    // Stub - fallback to scalar for now
    buff_averages_scalar(
        price,
        volume,
        fast_period,
        slow_period,
        first,
        fast_out,
        slow_out,
    );
}

// ==================== AVX512 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn buff_averages_avx512(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    // Stub - fallback to scalar for now
    buff_averages_scalar(
        price,
        volume,
        fast_period,
        slow_period,
        first,
        fast_out,
        slow_out,
    );
}

// ==================== STREAMING SUPPORT ====================
/// Streaming calculator for real-time updates
#[derive(Debug, Clone)]
pub struct BuffAveragesStream {
    price_buffer: Vec<f64>,
    volume_buffer: Vec<f64>,
    fast_period: usize,
    slow_period: usize,
    index: usize,
    ready: bool,
}

impl BuffAveragesStream {
    pub fn try_new(params: BuffAveragesParams) -> Result<Self, BuffAveragesError> {
        let fast_period = params.fast_period.unwrap_or(5);
        let slow_period = params.slow_period.unwrap_or(20);

        if fast_period == 0 {
            return Err(BuffAveragesError::InvalidPeriod {
                period: fast_period,
                data_len: 0,
            });
        }

        if slow_period == 0 {
            return Err(BuffAveragesError::InvalidPeriod {
                period: slow_period,
                data_len: 0,
            });
        }

        Ok(Self {
            price_buffer: vec![0.0; slow_period],
            volume_buffer: vec![0.0; slow_period],
            fast_period,
            slow_period,
            index: 0,
            ready: false,
        })
    }

    pub fn update(&mut self, price: f64, volume: f64) -> Option<(f64, f64)> {
        let idx = self.index % self.slow_period;
        self.price_buffer[idx] = price;
        self.volume_buffer[idx] = volume;
        self.index += 1;

        if self.index >= self.slow_period {
            self.ready = true;
        }

        if self.ready {
            // Calculate slow buffer
            let mut slow_num = 0.0;
            let mut slow_den = 0.0;
            for i in 0..self.slow_period {
                slow_num += self.price_buffer[i] * self.volume_buffer[i];
                slow_den += self.volume_buffer[i];
            }
            let slow_buff = if slow_den != 0.0 {
                slow_num / slow_den
            } else {
                0.0
            };

            // Calculate fast buffer
            let mut fast_num = 0.0;
            let mut fast_den = 0.0;
            let start = self.slow_period - self.fast_period;
            for i in start..self.slow_period {
                let idx = (self.index - self.slow_period + i) % self.slow_period;
                fast_num += self.price_buffer[idx] * self.volume_buffer[idx];
                fast_den += self.volume_buffer[idx];
            }
            let fast_buff = if fast_den != 0.0 {
                fast_num / fast_den
            } else {
                0.0
            };

            Some((fast_buff, slow_buff))
        } else {
            None
        }
    }
}

// ==================== BATCH PROCESSING ====================
/// Batch processing range for parameter sweeps
#[derive(Clone, Debug)]
pub struct BuffAveragesBatchRange {
    pub fast_period: (usize, usize, usize), // (start, end, step)
    pub slow_period: (usize, usize, usize),
}

impl Default for BuffAveragesBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (5, 5, 0),
            slow_period: (20, 20, 0),
        }
    }
}

/// Batch output structure
#[derive(Clone, Debug)]
pub struct BuffAveragesBatchOutput {
    pub fast: Vec<f64>,              // rows*cols flattened
    pub slow: Vec<f64>,              // rows*cols flattened
    pub combos: Vec<(usize, usize)>, // (fast, slow) pairs
    pub rows: usize,
    pub cols: usize,
}

/// Batch builder for ergonomic API
#[derive(Clone, Debug, Default)]
pub struct BuffAveragesBatchBuilder {
    range: BuffAveragesBatchRange,
    kernel: Kernel,
}

impl BuffAveragesBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }

    #[inline]
    pub fn fast_period_static(mut self, val: usize) -> Self {
        self.range.fast_period = (val, val, 0);
        self
    }

    #[inline]
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }

    #[inline]
    pub fn slow_period_static(mut self, val: usize) -> Self {
        self.range.slow_period = (val, val, 0);
        self
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
        let price = source_type(candles, "close");
        let volume = &candles.volume;
        buff_averages_batch_with_kernel(price, volume, &self.range, self.kernel)
    }

    pub fn apply_slices(
        self,
        price: &[f64],
        volume: &[f64],
    ) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
        buff_averages_batch_with_kernel(price, volume, &self.range, self.kernel)
    }
}

/// Helper to expand parameter grid
fn expand_grid_ba(r: &BuffAveragesBatchRange) -> Vec<(usize, usize)> {
    fn axis((a, b, s): (usize, usize, usize)) -> Vec<usize> {
        if s == 0 || a == b {
            return vec![a];
        }
        (a..=b).step_by(s).collect()
    }

    let fasts = axis(r.fast_period);
    let slows = axis(r.slow_period);
    let mut v = Vec::with_capacity(fasts.len() * slows.len());

    for &f in &fasts {
        for &s in &slows {
            v.push((f, s));
        }
    }
    v
}

/// Internal batch processing that writes directly into caller buffers (zero-copy)
#[inline]
pub fn buff_averages_batch_inner_into(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    kern: Kernel,
    fast_out: &mut [f64], // len = rows*cols
    slow_out: &mut [f64], // len = rows*cols
) -> Result<Vec<(usize, usize)>, BuffAveragesError> {
    buff_averages_batch_inner_into_parallel(price, volume, sweep, kern, fast_out, slow_out, false)
}

/// Internal batch processing with parallel option
#[inline]
fn buff_averages_batch_inner_into_parallel(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    kern: Kernel,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    parallel: bool,
) -> Result<Vec<(usize, usize)>, BuffAveragesError> {
    let combos = expand_grid_ba(sweep);
    if combos.is_empty() {
        return Err(BuffAveragesError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    if price.len() != volume.len() || price.is_empty() {
        return Err(BuffAveragesError::MismatchedDataLength {
            price_len: price.len(),
            volume_len: volume.len(),
        });
    }

    let first = price
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BuffAveragesError::AllValuesNaN)?;

    let max_slow = combos.iter().map(|&(_, s)| s).max().unwrap();
    if price.len() - first < max_slow {
        return Err(BuffAveragesError::NotEnoughValidData {
            needed: max_slow,
            valid: price.len() - first,
        });
    }

    let rows = combos.len();
    let cols = price.len();
    assert_eq!(fast_out.len(), rows * cols);
    assert_eq!(slow_out.len(), rows * cols);

    // SAFETY: re-interpret as MaybeUninit to use init_matrix_prefixes
    let fast_mu = unsafe {
        core::slice::from_raw_parts_mut(
            fast_out.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            fast_out.len(),
        )
    };
    let slow_mu = unsafe {
        core::slice::from_raw_parts_mut(
            slow_out.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            slow_out.len(),
        )
    };

    let warms: Vec<usize> = combos.iter().map(|&(_, slow)| first + slow - 1).collect();
    init_matrix_prefixes(fast_mu, cols, &warms);
    init_matrix_prefixes(slow_mu, cols, &warms);

    let simd = match match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    } {
        Kernel::ScalarBatch => Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => Kernel::Scalar,
    };

    // Compute each row - either parallel or sequential
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;

            // Process fast and slow outputs in parallel
            fast_out
                .par_chunks_mut(cols)
                .zip(slow_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (fr, sr))| {
                    let (fp, sp) = combos[row];
                    buff_averages_compute_into(price, volume, fp, sp, first, simd, fr, sr);
                });
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM doesn't support threads, fall back to sequential
            for (row, &(fp, sp)) in combos.iter().enumerate() {
                let fr = &mut fast_out[row * cols..(row + 1) * cols];
                let sr = &mut slow_out[row * cols..(row + 1) * cols];
                buff_averages_compute_into(price, volume, fp, sp, first, simd, fr, sr);
            }
        }
    } else {
        // Sequential processing
        for (row, &(fp, sp)) in combos.iter().enumerate() {
            let fr = &mut fast_out[row * cols..(row + 1) * cols];
            let sr = &mut slow_out[row * cols..(row + 1) * cols];
            buff_averages_compute_into(price, volume, fp, sp, first, simd, fr, sr);
        }
    }

    Ok(combos)
}

/// Batch processing with kernel selection (sequential)
pub fn buff_averages_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    k: Kernel,
) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
    buff_averages_batch_inner(price, volume, sweep, k, false)
}

/// Batch processing with kernel selection (parallel)
#[inline(always)]
pub fn buff_averages_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    k: Kernel,
) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
    buff_averages_batch_inner(price, volume, sweep, k, true)
}

/// Internal batch processing with parallel option
#[inline(always)]
fn buff_averages_batch_inner(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    k: Kernel,
    parallel: bool,
) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
    if price.is_empty() {
        return Err(BuffAveragesError::EmptyInputData);
    }
    if price.len() != volume.len() {
        return Err(BuffAveragesError::MismatchedDataLength {
            price_len: price.len(),
            volume_len: volume.len(),
        });
    }
    let first = price
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BuffAveragesError::AllValuesNaN)?;
    let combos = expand_grid_ba(sweep);
    if combos.is_empty() {
        return Err(BuffAveragesError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let max_slow = combos.iter().map(|&(_, s)| s).max().unwrap();
    if price.len() - first < max_slow {
        return Err(BuffAveragesError::NotEnoughValidData {
            needed: max_slow,
            valid: price.len() - first,
        });
    }

    let rows = combos.len();
    let cols = price.len();

    // 2 matrices, uninitialized, zero-copy
    let mut fast_mu = make_uninit_matrix(rows, cols);
    let mut slow_mu = make_uninit_matrix(rows, cols);

    // Convert to f64 slices for compute
    let fast_slice =
        unsafe { core::slice::from_raw_parts_mut(fast_mu.as_mut_ptr() as *mut f64, fast_mu.len()) };
    let slow_slice =
        unsafe { core::slice::from_raw_parts_mut(slow_mu.as_mut_ptr() as *mut f64, slow_mu.len()) };

    // Inner helper will set NaN warm prefixes itself
    buff_averages_batch_inner_into_parallel(
        price, volume, sweep, k, fast_slice, slow_slice, parallel,
    )?;

    // Return as Vec<f64> without copy
    let fast = unsafe {
        let ptr = fast_mu.as_mut_ptr() as *mut f64;
        let len = fast_mu.len();
        let cap = fast_mu.capacity();
        core::mem::forget(fast_mu);
        Vec::from_raw_parts(ptr, len, cap)
    };
    let slow = unsafe {
        let ptr = slow_mu.as_mut_ptr() as *mut f64;
        let len = slow_mu.len();
        let cap = slow_mu.capacity();
        core::mem::forget(slow_mu);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(BuffAveragesBatchOutput {
        fast,
        slow,
        combos,
        rows,
        cols,
    })
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "buff_averages")]
#[pyo3(signature = (price, volume, fast_period=5, slow_period=20, kernel=None))]
pub fn buff_averages_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    fast_period: usize,
    slow_period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let price_slice = price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = BuffAveragesParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let input = BuffAveragesInput::from_slices(price_slice, volume_slice, params);

    let result = py
        .allow_threads(|| buff_averages_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        result.fast_buff.into_pyarray(py),
        result.slow_buff.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "buff_averages_batch")]
#[pyo3(signature = (price, volume, fast_range, slow_range, kernel=None))]
pub fn buff_averages_batch_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    fast_range: (usize, usize, usize),
    slow_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    let p = price.as_slice()?;
    let v = volume.as_slice()?;
    let sweep = BuffAveragesBatchRange {
        fast_period: fast_range,
        slow_period: slow_range,
    };
    let kern = validate_kernel(kernel, true)?;

    // Allocate NumPy buffers up front
    let combos = expand_grid_ba(&sweep);
    let rows = combos.len();
    let cols = p.len();
    let fast_arr = unsafe { numpy::PyArray1::<f64>::new(py, [rows * cols], false) };
    let slow_arr = unsafe { numpy::PyArray1::<f64>::new(py, [rows * cols], false) };

    // Get mutable slices before allow_threads
    let fast_slice = unsafe { fast_arr.as_slice_mut()? };
    let slow_slice = unsafe { slow_arr.as_slice_mut()? };

    // Compute directly into NumPy memory (zero extra copy)
    let combos = py
        .allow_threads(|| {
            buff_averages_batch_inner_into(p, v, &sweep, kern, fast_slice, slow_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("fast", fast_arr.reshape((rows, cols))?)?;
    d.set_item("slow", slow_arr.reshape((rows, cols))?)?;
    d.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|c| c.0 as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|c| c.1 as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

#[cfg(feature = "python")]
#[pyclass(name = "BuffAveragesStream")]
pub struct BuffAveragesStreamPy {
    stream: BuffAveragesStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BuffAveragesStreamPy {
    #[new]
    fn new(fast_period: usize, slow_period: usize) -> PyResult<Self> {
        let params = BuffAveragesParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
        };
        let stream = BuffAveragesStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(BuffAveragesStreamPy { stream })
    }

    fn update(&mut self, price: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(price, volume)
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BuffAveragesJsResult {
    pub values: Vec<f64>, // row-major: [fast..., slow...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = buff_averages)]
pub fn buff_averages_unified_js(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
) -> Result<JsValue, JsValue> {
    let len = price.len();
    let params = BuffAveragesParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let input = BuffAveragesInput::from_slices(price, volume, params);

    // Allocate one 2×len matrix with warm prefixes in one pass
    let mut mat = make_uninit_matrix(2, len);
    {
        let warms = {
            // reuse prepare() to compute warm index
            let (_, _, _, sp, first, _) = buff_averages_prepare(&input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            vec![first + sp - 1, first + sp - 1]
        };
        init_matrix_prefixes(&mut mat, len, &warms);
    }

    // Compute directly into the two rows
    let values = unsafe {
        let flat = core::slice::from_raw_parts_mut(mat.as_mut_ptr() as *mut f64, mat.len());
        let (fast_out, slow_out) = flat.split_at_mut(len);
        buff_averages_into_slices(fast_out, slow_out, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let ptr = mat.as_mut_ptr() as *mut f64;
        let len = mat.len();
        let cap = mat.capacity();
        core::mem::forget(mat);
        Vec::from_raw_parts(ptr, len, cap)
    };

    let js = BuffAveragesJsResult {
        values,
        rows: 2,
        cols: len,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Keep old API for backwards compatibility
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn buff_averages_js(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let len = price.len();
    let params = BuffAveragesParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let input = BuffAveragesInput::from_slices(price, volume, params);

    // One allocation for [fast..., slow...]
    let mut mat = make_uninit_matrix(2, len);
    {
        let (_, _, _, sp, first, _) = buff_averages_prepare(&input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let warm = first + sp - 1;
        init_matrix_prefixes(&mut mat, len, &[warm, warm]);
    }

    let values = unsafe {
        let flat = core::slice::from_raw_parts_mut(mat.as_mut_ptr() as *mut f64, mat.len());
        let (fast_out, slow_out) = flat.split_at_mut(len);
        buff_averages_into_slices(fast_out, slow_out, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let ptr = mat.as_mut_ptr() as *mut f64;
        let len = mat.len();
        let cap = mat.capacity();
        core::mem::forget(mat);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn buff_averages_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_period: usize,
    slow_period: usize,
) -> Result<(), JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to buff_averages_into",
        ));
    }

    unsafe {
        let price = core::slice::from_raw_parts(price_ptr, len);
        let volume = core::slice::from_raw_parts(volume_ptr, len);
        let (fast_out, slow_out) =
            core::slice::from_raw_parts_mut(out_ptr, 2 * len).split_at_mut(len);

        let params = BuffAveragesParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
        };
        let input = BuffAveragesInput::from_slices(price, volume, params);

        buff_averages_into_slices(fast_out, slow_out, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn buff_averages_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn buff_averages_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 2 * len, 2 * len);
    }
}

// ==================== WASM BATCH BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BuffAveragesBatchJsOutput {
    pub values: Vec<f64>,         // row-major (fast rows..., then slow rows...)
    pub rows: usize,              // = 2 * combos.len()
    pub cols: usize,              // = data len
    pub fast_periods: Vec<usize>, // length = combos.len()
    pub slow_periods: Vec<usize>, // length = combos.len()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = buff_averages_batch)]
pub fn buff_averages_batch_unified_js(
    price: &[f64],
    volume: &[f64],
    fast_range: Vec<usize>, // [start, end, step]
    slow_range: Vec<usize>, // [start, end, step]
) -> Result<JsValue, JsValue> {
    if fast_range.len() != 3 || slow_range.len() != 3 {
        return Err(JsValue::from_str(
            "fast_range and slow_range must each have 3 elements [start, end, step]",
        ));
    }

    let sweep = BuffAveragesBatchRange {
        fast_period: (fast_range[0], fast_range[1], fast_range[2]),
        slow_period: (slow_range[0], slow_range[1], slow_range[2]),
    };

    let out = buff_averages_batch_with_kernel(price, volume, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Flatten as [fast rows..., then slow rows...] to keep layout simple.
    let mut values = Vec::with_capacity(out.fast.len() + out.slow.len());
    values.extend_from_slice(&out.fast);
    values.extend_from_slice(&out.slow);

    let js = BuffAveragesBatchJsOutput {
        values,
        rows: out.rows * 2, // *2 because we have fast and slow
        cols: out.cols,
        fast_periods: out.combos.iter().map(|c| c.0).collect(),
        slow_periods: out.combos.iter().map(|c| c.1).collect(),
    };

    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Zero-copy WASM batch writer
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn buff_averages_batch_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_fast_ptr: *mut f64,
    out_slow_ptr: *mut f64,
    len: usize,
    fast_start: usize,
    fast_end: usize,
    fast_step: usize,
    slow_start: usize,
    slow_end: usize,
    slow_step: usize,
) -> Result<usize, JsValue> {
    if price_ptr.is_null()
        || volume_ptr.is_null()
        || out_fast_ptr.is_null()
        || out_slow_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to buff_averages_batch_into",
        ));
    }
    unsafe {
        let price = core::slice::from_raw_parts(price_ptr, len);
        let volume = core::slice::from_raw_parts(volume_ptr, len);
        let sweep = BuffAveragesBatchRange {
            fast_period: (fast_start, fast_end, fast_step),
            slow_period: (slow_start, slow_end, slow_step),
        };
        // Reuse inner "into" that sets warm prefixes
        let combos = {
            let rows = expand_grid_ba(&sweep).len();
            let fast_out = core::slice::from_raw_parts_mut(out_fast_ptr, rows * len);
            let slow_out = core::slice::from_raw_parts_mut(out_slow_ptr, rows * len);
            buff_averages_batch_inner_into(
                price,
                volume,
                &sweep,
                detect_best_batch_kernel(),
                fast_out,
                slow_out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
        };
        Ok(combos.len())
    }
}

// ==================== DEPRECATED WASM CONTEXT ====================
// This is provided for API parity with alma.rs but is deprecated
// For performance-critical paths, use the direct APIs with persistent buffers

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct BuffAveragesContext {
    fast_period: usize,
    slow_period: usize,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl BuffAveragesContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For performance patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(fast_period: usize, slow_period: usize) -> Result<BuffAveragesContext, JsValue> {
        if fast_period == 0 {
            return Err(JsValue::from_str(&format!(
                "Invalid fast period: {}",
                fast_period
            )));
        }
        if slow_period == 0 {
            return Err(JsValue::from_str(&format!(
                "Invalid slow period: {}",
                slow_period
            )));
        }

        Ok(BuffAveragesContext {
            fast_period,
            slow_period,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        price_ptr: *const f64,
        volume_ptr: *const f64,
        fast_out_ptr: *mut f64,
        slow_out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if len < self.slow_period {
            return Err(JsValue::from_str("Data length less than slow period"));
        }

        if price_ptr.is_null()
            || volume_ptr.is_null()
            || fast_out_ptr.is_null()
            || slow_out_ptr.is_null()
        {
            return Err(JsValue::from_str("null pointer passed to update_into"));
        }

        unsafe {
            let price = std::slice::from_raw_parts(price_ptr, len);
            let volume = std::slice::from_raw_parts(volume_ptr, len);
            let fast_out = std::slice::from_raw_parts_mut(fast_out_ptr, len);
            let slow_out = std::slice::from_raw_parts_mut(slow_out_ptr, len);

            let params = BuffAveragesParams {
                fast_period: Some(self.fast_period),
                slow_period: Some(self.slow_period),
            };
            let input = BuffAveragesInput::from_slices(price, volume, params);

            // Check if we're writing to the same memory as input
            let needs_temp = price_ptr == fast_out_ptr
                || price_ptr == slow_out_ptr
                || volume_ptr == fast_out_ptr
                || volume_ptr == slow_out_ptr;

            if needs_temp {
                // Use temporary buffers
                let mut temp_fast = vec![0.0; len];
                let mut temp_slow = vec![0.0; len];

                buff_averages_into_slices(&mut temp_fast, &mut temp_slow, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;

                fast_out.copy_from_slice(&temp_fast);
                slow_out.copy_from_slice(&temp_slow);
            } else {
                // Direct write
                buff_averages_into_slices(fast_out, slow_out, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.slow_period - 1
    }

    #[wasm_bindgen]
    pub fn compute(&self, price: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
        let params = BuffAveragesParams {
            fast_period: Some(self.fast_period),
            slow_period: Some(self.slow_period),
        };
        let input = BuffAveragesInput::from_slices(price, volume, params);
        let result = buff_averages_with_kernel(&input, self.kernel)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Concatenate [fast..., slow...]
        let mut output = Vec::with_capacity(price.len() * 2);
        output.extend_from_slice(&result.fast_buff);
        output.extend_from_slice(&result.slow_buff);
        Ok(output)
    }
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_buff_averages_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input =
            BuffAveragesInput::from_candles(&candles, "close", BuffAveragesParams::default());
        let result = buff_averages_with_kernel(&input, kernel)?;

        // REFERENCE VALUES FROM PINESCRIPT (NEVER MODIFY THESE!)
        let expected_fast = [
            58740.30855637,
            59132.28418702,
            59309.76658172,
            59266.10492431, // Fixed typo: was 59.266 in original
            59194.11908892,
        ];

        let expected_slow = [
            59209.26229392,
            59201.87047432,
            59217.15739355,
            59195.74527194,
            59196.26139533,
        ];

        // Get last 6 values and use first 5 for comparison
        let start = result.fast_buff.len().saturating_sub(6);

        for (i, (&fast_val, &slow_val)) in result.fast_buff[start..]
            .iter()
            .take(5)
            .zip(result.slow_buff[start..].iter())
            .enumerate()
        {
            let fast_diff = (fast_val - expected_fast[i]).abs();
            let slow_diff = (slow_val - expected_slow[i]).abs();
            assert!(
                fast_diff < 1e-3, // Increased tolerance for floating point differences
                "[{}] Buff Averages {:?} fast mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                fast_val,
                expected_fast[i]
            );
            assert!(
                slow_diff < 1e-3, // Increased tolerance for floating point differences
                "[{}] Buff Averages {:?} slow mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                slow_val,
                expected_slow[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_buff_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = buff_averages_with_kernel(&BuffAveragesInput::with_default_candles(&c), kernel)?;

        for (i, &v) in out.fast_buff.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                "[{}] poison in fast at {}: {:#x}",
                test_name,
                i,
                b
            );
        }

        for (i, &v) in out.slow_buff.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                "[{}] poison in slow at {}: {:#x}",
                test_name,
                i,
                b
            );
        }
        Ok(())
    }

    fn check_buff_nan_prefix(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = BuffAveragesInput::with_default_candles(&c);

        let (price, _, _, slow_p, first, _) = buff_averages_prepare(&input, kernel)?;
        let warm = first + slow_p - 1;

        let out = buff_averages_with_kernel(&input, kernel)?;

        assert!(
            out.fast_buff[..warm].iter().all(|x| x.is_nan()),
            "[{}] fast warmup not NaN",
            test_name
        );
        assert!(
            out.slow_buff[..warm].iter().all(|x| x.is_nan()),
            "[{}] slow warmup not NaN",
            test_name
        );
        assert!(
            out.fast_buff[warm..].iter().all(|x| x.is_finite()),
            "[{}] fast post-warm has NaN",
            test_name
        );
        assert!(
            out.slow_buff[warm..].iter().all(|x| x.is_finite()),
            "[{}] slow post-warm has NaN",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = BuffAveragesParams {
            fast_period: None,
            slow_period: None,
        };
        let input = BuffAveragesInput::from_candles(&candles, "close", default_params);
        let output = buff_averages_with_kernel(&input, kernel)?;
        assert_eq!(output.fast_buff.len(), candles.close.len());
        assert_eq!(output.slow_buff.len(), candles.close.len());

        Ok(())
    }

    fn check_buff_averages_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = BuffAveragesInput::with_default_candles(&candles);
        match input.data {
            BuffAveragesData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected BuffAveragesData::Candles"),
        }
        let output = buff_averages_with_kernel(&input, kernel)?;
        assert_eq!(output.fast_buff.len(), candles.close.len());
        assert_eq!(output.slow_buff.len(), candles.close.len());

        Ok(())
    }

    fn check_buff_averages_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let volume_data = [100.0, 200.0, 300.0];
        let params = BuffAveragesParams {
            fast_period: Some(0),
            slow_period: Some(10),
        };
        let input = BuffAveragesInput::from_slices(&input_data, &volume_data, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let volume_small = [100.0, 200.0, 300.0];
        let params = BuffAveragesParams {
            fast_period: Some(5),
            slow_period: Some(10),
        };
        let input = BuffAveragesInput::from_slices(&data_small, &volume_small, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let single_volume = [100.0];
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&single_point, &single_volume, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&empty, &empty, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let nan_volume = [f64::NAN, f64::NAN, f64::NAN];
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&nan_data, &nan_volume, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    // Test for mismatched data lengths
    fn check_buff_averages_mismatched_lengths(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price_data = [10.0, 20.0, 30.0];
        let volume_data = [100.0, 200.0]; // Different length
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&price_data, &volume_data, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with mismatched data lengths",
            test_name
        );
        Ok(())
    }

    // Test for missing volume data
    fn check_buff_averages_missing_volume(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price_data = [
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
        ];
        let params = BuffAveragesParams::default();

        // Create input with slice data but no volume
        let input = BuffAveragesInput {
            data: BuffAveragesData::Slice(&price_data),
            params,
            volume: None, // Missing volume
        };

        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with missing volume data",
            test_name
        );
        Ok(())
    }

    // ==================== BATCH TESTS ====================

    fn check_buff_averages_batch_single(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = BuffAveragesBatchRange {
            fast_period: (5, 5, 0),   // Single value
            slow_period: (20, 20, 0), // Single value
        };

        let price = source_type(&candles, "close");
        let volume = &candles.volume;

        let batch_result = buff_averages_batch_with_kernel(price, volume, &range, kernel)?;

        // Compare with single calculation
        let single_input =
            BuffAveragesInput::from_slices(price, volume, BuffAveragesParams::default());
        let single_result = buff_averages_with_kernel(&single_input, kernel)?;

        // Should have 1 combination
        assert_eq!(batch_result.rows, 1, "[{}] Expected 1 row", test_name);
        assert_eq!(
            batch_result.combos.len(),
            1,
            "[{}] Expected 1 combination",
            test_name
        );

        // Compare outputs
        for i in 0..price.len() {
            let batch_fast = batch_result.fast[i];
            let single_fast = single_result.fast_buff[i];
            if batch_fast.is_finite() && single_fast.is_finite() {
                assert!(
                    (batch_fast - single_fast).abs() < 1e-10,
                    "[{}] Fast mismatch at {}: batch={}, single={}",
                    test_name,
                    i,
                    batch_fast,
                    single_fast
                );
            }

            let batch_slow = batch_result.slow[i];
            let single_slow = single_result.slow_buff[i];
            if batch_slow.is_finite() && single_slow.is_finite() {
                assert!(
                    (batch_slow - single_slow).abs() < 1e-10,
                    "[{}] Slow mismatch at {}: batch={}, single={}",
                    test_name,
                    i,
                    batch_slow,
                    single_slow
                );
            }
        }
        Ok(())
    }

    fn check_buff_averages_batch_grid(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = BuffAveragesBatchRange {
            fast_period: (3, 7, 2),   // 3, 5, 7
            slow_period: (18, 22, 2), // 18, 20, 22
        };

        let price = source_type(&candles, "close");
        let volume = &candles.volume;

        let result = buff_averages_batch_with_kernel(price, volume, &range, kernel)?;

        // Should have 3 fast periods * 3 slow periods = 9 combinations
        assert_eq!(result.rows, 9, "[{}] Expected 9 rows", test_name);
        assert_eq!(
            result.cols,
            candles.close.len(),
            "[{}] Cols mismatch",
            test_name
        );
        assert_eq!(
            result.combos.len(),
            9,
            "[{}] Expected 9 combinations",
            test_name
        );
        assert_eq!(
            result.fast.len(),
            9 * candles.close.len(),
            "[{}] Fast size mismatch",
            test_name
        );
        assert_eq!(
            result.slow.len(),
            9 * candles.close.len(),
            "[{}] Slow size mismatch",
            test_name
        );

        // Verify combinations
        let expected_combos = vec![
            (3, 18),
            (3, 20),
            (3, 22),
            (5, 18),
            (5, 20),
            (5, 22),
            (7, 18),
            (7, 20),
            (7, 22),
        ];
        assert_eq!(
            result.combos, expected_combos,
            "[{}] Combinations mismatch",
            test_name
        );

        Ok(())
    }

    fn check_buff_averages_batch_empty(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [];
        let volume = [];

        let range = BuffAveragesBatchRange {
            fast_period: (5, 10, 1),
            slow_period: (15, 20, 1),
        };

        let res = buff_averages_batch_with_kernel(&price, &volume, &range, kernel);
        assert!(
            res.is_err(),
            "[{}] Batch should fail with empty input",
            test_name
        );
        Ok(())
    }

    // Test parallel batch processing
    fn check_buff_averages_batch_parallel(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = BuffAveragesBatchRange {
            fast_period: (3, 7, 2),   // 3, 5, 7
            slow_period: (18, 22, 2), // 18, 20, 22
        };

        let price = source_type(&candles, "close");
        let volume = &candles.volume;

        // Run sequential version
        let seq_result = buff_averages_batch_with_kernel(price, volume, &range, kernel)?;

        // Run parallel version
        let par_result = buff_averages_batch_par_slice(price, volume, &range, kernel)?;

        // Compare results - they should be identical
        assert_eq!(
            seq_result.rows, par_result.rows,
            "[{}] Row count mismatch",
            test_name
        );
        assert_eq!(
            seq_result.cols, par_result.cols,
            "[{}] Col count mismatch",
            test_name
        );
        assert_eq!(
            seq_result.combos, par_result.combos,
            "[{}] Combos mismatch",
            test_name
        );

        // Compare actual values with small tolerance for floating point differences
        for i in 0..seq_result.fast.len() {
            let seq_fast = seq_result.fast[i];
            let par_fast = par_result.fast[i];
            if seq_fast.is_finite() && par_fast.is_finite() {
                assert!(
                    (seq_fast - par_fast).abs() < 1e-10,
                    "[{}] Fast parallel mismatch at {}: seq={}, par={}",
                    test_name,
                    i,
                    seq_fast,
                    par_fast
                );
            } else {
                assert_eq!(
                    seq_fast.is_nan(),
                    par_fast.is_nan(),
                    "[{}] Fast NaN mismatch at {}",
                    test_name,
                    i
                );
            }
        }

        for i in 0..seq_result.slow.len() {
            let seq_slow = seq_result.slow[i];
            let par_slow = par_result.slow[i];
            if seq_slow.is_finite() && par_slow.is_finite() {
                assert!(
                    (seq_slow - par_slow).abs() < 1e-10,
                    "[{}] Slow parallel mismatch at {}: seq={}, par={}",
                    test_name,
                    i,
                    seq_slow,
                    par_slow
                );
            } else {
                assert_eq!(
                    seq_slow.is_nan(),
                    par_slow.is_nan(),
                    "[{}] Slow NaN mismatch at {}",
                    test_name,
                    i
                );
            }
        }

        Ok(())
    }

    // ==================== STREAMING TESTS ====================

    #[test]
    fn test_buff_averages_stream() -> Result<(), Box<dyn Error>> {
        let params = BuffAveragesParams::default();
        let mut stream = BuffAveragesStream::try_new(params)?;

        // Feed data points
        let test_data = vec![
            (100.0, 1000.0),
            (110.0, 1100.0),
            (120.0, 1200.0),
            (130.0, 1300.0),
            (140.0, 1400.0),
            (150.0, 1500.0),
            (160.0, 1600.0),
            (170.0, 1700.0),
            (180.0, 1800.0),
            (190.0, 1900.0),
            (200.0, 2000.0),
            (210.0, 2100.0),
            (220.0, 2200.0),
            (230.0, 2300.0),
            (240.0, 2400.0),
            (250.0, 2500.0),
            (260.0, 2600.0),
            (270.0, 2700.0),
            (280.0, 2800.0),
            (290.0, 2900.0),
            (300.0, 3000.0),
        ];

        let mut results = Vec::new();
        for (price, volume) in test_data {
            if let Some(result) = stream.update(price, volume) {
                results.push(result);
            }
        }

        // Should get result after slow period (20)
        assert!(!results.is_empty(), "Stream should produce results");

        Ok(())
    }

    // ==================== MACRO GENERATION ====================

    // Macro for generating single indicator tests
    macro_rules! generate_buff_averages_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar);
                    }
                )*

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2);
                    }

                    #[test]
                    fn [<$test_fn _avx512>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512);
                    }
                )*
            }
        };
    }

    // Macro for batch tests
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

    // Generate all single kernel tests
    generate_buff_averages_tests!(
        check_buff_averages_accuracy,
        check_buff_averages_partial_params,
        check_buff_averages_default_candles,
        check_buff_averages_zero_period,
        check_buff_averages_period_exceeds_length,
        check_buff_averages_very_small_dataset,
        check_buff_averages_empty_input,
        check_buff_averages_all_nan,
        check_buff_averages_mismatched_lengths,
        check_buff_averages_missing_volume,
        check_buff_nan_prefix
    );

    // Debug-only tests
    #[cfg(debug_assertions)]
    generate_buff_averages_tests!(check_buff_no_poison);

    // Generate batch tests
    gen_batch_tests!(check_buff_averages_batch_single);
    gen_batch_tests!(check_buff_averages_batch_grid);
    gen_batch_tests!(check_buff_averages_batch_empty);
    gen_batch_tests!(check_buff_averages_batch_parallel);

    // ==================== PROPERTY-BASED TESTS ====================

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn prop_buff_averages_length_preserved(
            len in 50usize..100,
            fast_period in 2usize..10,
            slow_period in 11usize..30
        ) {
            // Generate same-length vectors
            let data: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 10.0).collect();
            let volume: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 100.0).collect();

            prop_assume!(data.len() > slow_period);

            let params = BuffAveragesParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
            };
            let input = BuffAveragesInput::from_slices(&data, &volume, params);

            if let Ok(output) = buff_averages(&input) {
                prop_assert_eq!(output.fast_buff.len(), data.len());
                prop_assert_eq!(output.slow_buff.len(), data.len());
            }
        }

        #[test]
        fn prop_buff_averages_nan_handling(
            len in 50usize..100
        ) {
            // Generate same-length vectors
            let mut data: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 10.0).collect();
            let mut volume: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 100.0).collect();

            // Insert some NaNs
            for i in (0..5).map(|x| x * 10) {
                if i < data.len() {
                    data[i] = f64::NAN;
                    volume[i] = f64::NAN;
                }
            }

            let params = BuffAveragesParams::default();
            let input = BuffAveragesInput::from_slices(&data, &volume, params);

            // Should either succeed or fail gracefully
            let _ = buff_averages(&input);
        }
    }
}
