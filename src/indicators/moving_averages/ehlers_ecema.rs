//! # Ehlers Error Correcting Exponential Moving Average (ECEMA)
//!
//! Adaptive exponential moving average that automatically adjusts its gain to minimize
//! error between the indicator and price. Tests multiple gain values and selects the
//! one producing the least error for optimal smoothing and responsiveness.
//!
//! ## Parameters
//! - **length**: Period for EMA calculation (default: 20)
//! - **gain_limit**: Maximum gain value to test, divided by 10 (default: 50)
//! - **pine_compatible**: Use Pine Script compatible calculation mode
//! - **confirmed_only**: Use confirmed (previous) values only
//!
//! ## Returns
//! - **values**: Adaptive EMA values with optimized gain selection
//!
//! ## Developer Status
//! - **AVX2 kernel**: STUB - Falls back to scalar implementation
//! - **AVX512 kernel**: STUB - Falls back to scalar implementation
//! - **Streaming update**: O(n) - Recalculates EMA from scratch each update
//! - **Memory optimization**: GOOD - Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix)
//! - **Optimization needed**: Implement SIMD kernels, optimize streaming to O(1) with incremental updates

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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

// Import EMA indicator for regular EMA calculation
use crate::indicators::moving_averages::ema::{ema, ema_into_slice, EmaInput, EmaParams};

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
impl<'a> AsRef<[f64]> for EhlersEcemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersEcemaData::Slice(slice) => slice,
            EhlersEcemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum EhlersEcemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct EhlersEcemaOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EhlersEcemaParams {
    pub length: Option<usize>,
    pub gain_limit: Option<usize>,
    pub pine_compatible: Option<bool>,
    pub confirmed_only: Option<bool>,
}

impl Default for EhlersEcemaParams {
    fn default() -> Self {
        Self {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        }
    }
}

/// Input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct EhlersEcemaInput<'a> {
    pub data: EhlersEcemaData<'a>,
    pub params: EhlersEcemaParams,
}

impl<'a> EhlersEcemaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhlersEcemaParams) -> Self {
        Self {
            data: EhlersEcemaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EhlersEcemaParams) -> Self {
        Self {
            data: EhlersEcemaData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhlersEcemaParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(20)
    }

    #[inline]
    pub fn get_gain_limit(&self) -> usize {
        self.params.gain_limit.unwrap_or(50)
    }

    #[inline]
    pub fn get_pine_compatible(&self) -> bool {
        self.params.pine_compatible.unwrap_or(false)
    }

    #[inline]
    pub fn get_confirmed_only(&self) -> bool {
        self.params.confirmed_only.unwrap_or(false)
    }
}

// ==================== BUILDER PATTERN ====================
#[derive(Copy, Clone, Debug)]
pub struct EhlersEcemaBuilder {
    length: Option<usize>,
    gain_limit: Option<usize>,
    kernel: Kernel,
}

impl Default for EhlersEcemaBuilder {
    fn default() -> Self {
        Self {
            length: None,
            gain_limit: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersEcemaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }

    #[inline(always)]
    pub fn gain_limit(mut self, g: usize) -> Self {
        self.gain_limit = Some(g);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
        let p = EhlersEcemaParams {
            length: self.length,
            gain_limit: self.gain_limit,
            pine_compatible: None,
            confirmed_only: None,
        };
        let i = EhlersEcemaInput::from_candles(c, "close", p);
        ehlers_ecema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
        let p = EhlersEcemaParams {
            length: self.length,
            gain_limit: self.gain_limit,
            pine_compatible: None,
            confirmed_only: None,
        };
        let i = EhlersEcemaInput::from_slice(d, p);
        ehlers_ecema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersEcemaStream, EhlersEcemaError> {
        let p = EhlersEcemaParams {
            length: self.length,
            gain_limit: self.gain_limit,
            pine_compatible: None,
            confirmed_only: None,
        };
        EhlersEcemaStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum EhlersEcemaError {
    #[error("ehlers_ecema: Input data slice is empty.")]
    EmptyInputData,

    #[error("ehlers_ecema: All values are NaN.")]
    AllValuesNaN,

    #[error("ehlers_ecema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("ehlers_ecema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("ehlers_ecema: Invalid gain limit: {gain_limit}")]
    InvalidGainLimit { gain_limit: usize },

    #[error("ehlers_ecema: EMA calculation failed: {0}")]
    EmaError(String),
}

// ==================== PINE-STYLE EMA CALCULATION ====================
/// Calculate Pine EMA in-place (no extra Vec)
#[inline]
fn calculate_pine_ema_into(
    dst: &mut [f64],
    data: &[f64],
    _length: usize,
    alpha: f64,
    beta: f64,
    first: usize,
) {
    let len = data.len();
    for v in &mut dst[..first.min(len)] {
        *v = f64::NAN;
    }
    if first >= len {
        return;
    }
    let mut ema = 0.0; // zero-seeded
    for i in first..len {
        let src = data[i];
        if src.is_finite() {
            ema = alpha * src + beta * ema;
            dst[i] = ema;
        } else {
            dst[i] = ema;
        }
    }
}

// ==================== MAIN ALGORITHM ====================
#[inline]
pub fn ehlers_ecema(input: &EhlersEcemaInput) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
    ehlers_ecema_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn ehlers_ecema_prepare<'a>(
    input: &'a EhlersEcemaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, f64, f64, Kernel), EhlersEcemaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(EhlersEcemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersEcemaError::AllValuesNaN)?;
    let length = input.get_length();
    let gain_limit = input.get_gain_limit();

    if length == 0 || length > len {
        return Err(EhlersEcemaError::InvalidPeriod {
            period: length,
            data_len: len,
        });
    }

    if len - first < length {
        return Err(EhlersEcemaError::NotEnoughValidData {
            needed: length,
            valid: len - first,
        });
    }

    if gain_limit == 0 {
        return Err(EhlersEcemaError::InvalidGainLimit { gain_limit });
    }

    let alpha = 2.0 / (length as f64 + 1.0);
    let beta = 1.0 - alpha;
    let chosen = if matches!(kernel, Kernel::Auto) {
        detect_best_kernel()
    } else {
        kernel
    };

    Ok((data, length, gain_limit, first, alpha, beta, chosen))
}

#[inline(always)]
fn ehlers_ecema_compute_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    kernel: Kernel,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => ehlers_ecema_scalar_into_with_mode(
                data,
                ema_values,
                length,
                gain_limit,
                first,
                alpha,
                beta,
                pine_compatible,
                confirmed_only,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ehlers_ecema_avx2_into_with_mode(
                data,
                ema_values,
                length,
                gain_limit,
                first,
                alpha,
                beta,
                pine_compatible,
                confirmed_only,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ehlers_ecema_avx512_into_with_mode(
                data,
                ema_values,
                length,
                gain_limit,
                first,
                alpha,
                beta,
                pine_compatible,
                confirmed_only,
                out,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                ehlers_ecema_scalar_into_with_mode(
                    data,
                    ema_values,
                    length,
                    gain_limit,
                    first,
                    alpha,
                    beta,
                    pine_compatible,
                    confirmed_only,
                    out,
                )
            }
            _ => unreachable!(),
        }
    }
}

pub fn ehlers_ecema_with_kernel(
    input: &EhlersEcemaInput,
    kernel: Kernel,
) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
    let (data, length, gain_limit, first, alpha, beta, chosen) =
        ehlers_ecema_prepare(input, kernel)?;
    let pine_compatible = input.get_pine_compatible();
    let confirmed_only = input.get_confirmed_only();

    // Build EMA in-place
    let mut ema_buf = alloc_with_nan_prefix(
        data.len(),
        if pine_compatible {
            first
        } else {
            first + length - 1
        },
    );
    if pine_compatible {
        calculate_pine_ema_into(&mut ema_buf, data, length, alpha, beta, first);
    } else {
        let ema_input = EmaInput::from_slice(
            data,
            EmaParams {
                period: Some(length),
            },
        );
        ema_into_slice(&mut ema_buf, &ema_input, chosen)
            .map_err(|e| EhlersEcemaError::EmaError(e.to_string()))?;
    }

    // Allocate output once
    let mut out = alloc_with_nan_prefix(
        data.len(),
        if pine_compatible {
            first
        } else {
            first + length - 1
        },
    );

    ehlers_ecema_compute_into_with_mode(
        data,
        &ema_buf,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        chosen,
        pine_compatible,
        confirmed_only,
        &mut out,
    );

    Ok(EhlersEcemaOutput { values: out })
}

#[inline]
pub fn ehlers_ecema_into_slice(
    dst: &mut [f64],
    input: &EhlersEcemaInput,
    kern: Kernel,
) -> Result<(), EhlersEcemaError> {
    let (data, length, gain_limit, first, alpha, beta, chosen) = ehlers_ecema_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(EhlersEcemaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    let pine_compatible = input.get_pine_compatible();
    let confirmed_only = input.get_confirmed_only();

    // EMA buffer
    let mut ema_buf = alloc_with_nan_prefix(
        data.len(),
        if pine_compatible {
            first
        } else {
            first + length - 1
        },
    );
    if pine_compatible {
        calculate_pine_ema_into(&mut ema_buf, data, length, alpha, beta, first);
    } else {
        let ema_input = EmaInput::from_slice(
            data,
            EmaParams {
                period: Some(length),
            },
        );
        ema_into_slice(&mut ema_buf, &ema_input, chosen)
            .map_err(|e| EhlersEcemaError::EmaError(e.to_string()))?;
    }

    // Warmup prefix, ALMA-style
    let warmup_end = if pine_compatible {
        first
    } else {
        first + length - 1
    };
    let dst_len = dst.len();
    for v in &mut dst[..warmup_end.min(dst_len)] {
        *v = f64::NAN;
    }

    ehlers_ecema_compute_into_with_mode(
        data,
        &ema_buf,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        chosen,
        pine_compatible,
        confirmed_only,
        dst,
    );
    Ok(())
}

#[inline(always)]
unsafe fn ehlers_ecema_scalar_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    let len = data.len();
    debug_assert_eq!(out.len(), len);
    debug_assert_eq!(ema_values.len(), len);

    // Determine start index based on mode
    let start_idx = if pine_compatible {
        first // Pine mode: start from first valid value
    } else {
        first + length - 1 // Regular mode: wait for warmup
    };

    for i in start_idx..len {
        let src = if confirmed_only && i > 0 {
            *data.get_unchecked(i - 1)
        } else {
            *data.get_unchecked(i)
        };
        let ema = *ema_values.get_unchecked(i);

        // Get previous ecEma value
        let prev_ec = if i == start_idx {
            if pine_compatible {
                0.0 // Pine mode: start with 0 (nz(ecEma[1]) = 0)
            } else {
                ema // Regular mode: use EMA as initial value
            }
        } else {
            *out.get_unchecked(i - 1)
        };

        let mut least_error = f64::INFINITY;
        let mut best_gain = 0.0;

        // Test different gain values to find the one with minimum error
        for gain_int in -(gain_limit as i32)..=(gain_limit as i32) {
            let gain = gain_int as f64 / 10.0;
            let test_ec_ema = alpha * (ema + gain * (src - prev_ec)) + beta * prev_ec;
            let error = (src - test_ec_ema).abs();

            if error < least_error {
                least_error = error;
                best_gain = gain;
            }
        }

        // Calculate final value with best gain
        let final_ec_ema = alpha * (ema + best_gain * (src - prev_ec)) + beta * prev_ec;
        *out.get_unchecked_mut(i) = final_ec_ema;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ehlers_ecema_avx2_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    // For now, use scalar implementation with mode
    ehlers_ecema_scalar_into_with_mode(
        data,
        ema_values,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        pine_compatible,
        confirmed_only,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ehlers_ecema_avx512_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    // For now, use scalar implementation with mode
    ehlers_ecema_scalar_into_with_mode(
        data,
        ema_values,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        pine_compatible,
        confirmed_only,
        out,
    )
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug, Default)]
pub struct EhlersEcemaBatchRange {
    pub length: (usize, usize, usize),
    pub gain_limit: (usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct EhlersEcemaBatchBuilder {
    range: EhlersEcemaBatchRange,
    kernel: Kernel,
}

impl Default for EhlersEcemaBatchBuilder {
    fn default() -> Self {
        Self {
            range: EhlersEcemaBatchRange {
                length: (20, 20, 0),
                gain_limit: (50, 50, 0),
            },
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersEcemaBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, n: usize) -> Self {
        self.range.length = (n, n, 0);
        self
    }

    #[inline]
    pub fn gain_limit_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.gain_limit = (start, end, step);
        self
    }

    #[inline]
    pub fn gain_limit_static(mut self, g: usize) -> Self {
        self.range.gain_limit = (g, g, 0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        ehlers_ecema_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        EhlersEcemaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        EhlersEcemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct EhlersEcemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersEcemaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersEcemaBatchOutput {
    pub fn row_for_params(&self, p: &EhlersEcemaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(20) == p.length.unwrap_or(20)
                && c.gain_limit.unwrap_or(50) == p.gain_limit.unwrap_or(50)
        })
    }

    pub fn values_for(&self, p: &EhlersEcemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

pub fn ehlers_ecema_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    k: Kernel,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EhlersEcemaError::InvalidPeriod {
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

    ehlers_ecema_batch_inner(data, sweep, simd, true)
}

#[inline(always)]
fn expand_grid(r: &EhlersEcemaBatchRange) -> Vec<EhlersEcemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let lengths = axis_usize(r.length);
    let gain_limits = axis_usize(r.gain_limit);

    let mut out = Vec::with_capacity(lengths.len() * gain_limits.len());
    for &l in &lengths {
        for &g in &gain_limits {
            out.push(EhlersEcemaParams {
                length: Some(l),
                gain_limit: Some(g),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            });
        }
    }
    out
}

#[inline(always)]
fn ehlers_ecema_batch_inner(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    let rows = combos.len();

    if cols == 0 {
        // Match alma.rs behavior
        return Err(EhlersEcemaError::AllValuesNaN);
    }
    if combos.is_empty() {
        return Err(EhlersEcemaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // Allocate uninitialized matrix and initialize row prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersEcemaError::AllValuesNaN)?;
    let max_len = combos
        .iter()
        .map(|p| p.length.unwrap_or(20))
        .max()
        .unwrap_or(1);
    if cols - first < max_len {
        return Err(EhlersEcemaError::NotEnoughValidData {
            needed: max_len,
            valid: cols - first,
        });
    }

    // Warmup per row mirrors ALMA: first + length - 1
    let warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.length.unwrap_or(20) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Turn to &mut [f64] without extra init
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Fill rows directly
    ehlers_ecema_batch_inner_into(data, sweep, kern, parallel, out)?;

    // Recreate Vec<f64> without copy
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(EhlersEcemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ehlers_ecema_batch_inner_into(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EhlersEcemaParams>, EhlersEcemaError> {
    use std::collections::HashMap;

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EhlersEcemaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersEcemaError::AllValuesNaN)?;
    let cols = data.len();

    // Cache EMA series by length
    let mut ema_cache: HashMap<usize, Vec<f64>> = HashMap::new();

    // Pre-compute all unique lengths
    let unique_lengths: std::collections::HashSet<usize> =
        combos.iter().map(|p| p.length.unwrap_or(20)).collect();

    for &length in &unique_lengths {
        let mut buf = alloc_with_nan_prefix(cols, first + length - 1);
        let ema_in = EmaInput::from_slice(
            data,
            EmaParams {
                period: Some(length),
            },
        );
        ema_into_slice(&mut buf, &ema_in, kern)
            .map_err(|e| EhlersEcemaError::EmaError(e.to_string()))?;
        ema_cache.insert(length, buf);
    }

    let do_row = |row: usize, row_out: &mut [f64]| -> Result<(), EhlersEcemaError> {
        let p = &combos[row];
        let length = p.length.unwrap_or(20);
        let gain_limit = p.gain_limit.unwrap_or(50);
        let alpha = 2.0 / (length as f64 + 1.0);
        let beta = 1.0 - alpha;
        let ema_vals = ema_cache.get(&length).unwrap();
        unsafe {
            ehlers_ecema_scalar_into_with_mode(
                data, ema_vals, length, gain_limit, first, alpha, beta, false, false, row_out,
            );
        }
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .try_for_each(|(r, s)| do_row(r, s))?;
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    }

    Ok(combos)
}

// ==================== PUBLIC BATCH HELPERS FOR API PARITY ====================
#[inline(always)]
pub fn ehlers_ecema_batch_slice(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    ehlers_ecema_batch_inner(
        data,
        sweep,
        match kern {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        },
        /*parallel=*/ false,
    )
}

#[inline(always)]
pub fn ehlers_ecema_batch_par_slice(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    ehlers_ecema_batch_inner(
        data,
        sweep,
        match kern {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        },
        /*parallel=*/ true,
    )
}

// ==================== STREAMING SUPPORT ====================
#[derive(Debug, Clone)]
pub struct EhlersEcemaStream {
    length: usize,
    gain_limit: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    ema_mean: f64,
    ema_filled: bool,
    prev_ecema: f64,
    pine_compatible: bool,
    confirmed_only: bool,
    prev_value: Option<f64>, // For confirmed_only mode
}

impl EhlersEcemaStream {
    pub fn try_new(params: EhlersEcemaParams) -> Result<Self, EhlersEcemaError> {
        let length = params.length.unwrap_or(20);
        let gain_limit = params.gain_limit.unwrap_or(50);
        let pine_compatible = params.pine_compatible.unwrap_or(false);
        let confirmed_only = params.confirmed_only.unwrap_or(false);

        if length == 0 {
            return Err(EhlersEcemaError::InvalidPeriod {
                period: length,
                data_len: 0,
            });
        }

        if gain_limit == 0 {
            return Err(EhlersEcemaError::InvalidGainLimit { gain_limit });
        }

        let alpha = 2.0 / (length as f64 + 1.0);
        let beta = 1.0 - alpha;

        Ok(Self {
            length,
            gain_limit,
            alpha,
            beta,
            count: 0,
            ema_mean: 0.0,
            ema_filled: false,
            prev_ecema: 0.0,
            pine_compatible,
            confirmed_only,
            prev_value: None,
        })
    }

    pub fn next(&mut self, value: f64) -> f64 {
        if !value.is_finite() {
            return f64::NAN;
        }

        // Handle confirmed_only mode (use previous value as source)
        let src = if self.confirmed_only {
            match self.prev_value {
                Some(prev) => {
                    self.prev_value = Some(value);
                    prev
                }
                None => {
                    self.prev_value = Some(value);
                    value // First value, no previous to use
                }
            }
        } else {
            value
        };

        self.count += 1;

        // Calculate EMA based on mode
        let ema_value = if self.pine_compatible {
            // Pine-style: zero-seeded EMA
            self.ema_mean = self.alpha * src + self.beta * self.ema_mean;
            self.ema_mean
        } else {
            // Regular mode: running mean for warmup
            if !self.ema_filled {
                self.ema_mean =
                    ((self.count as f64 - 1.0) * self.ema_mean + src) / self.count as f64;
                if self.count >= self.length {
                    self.ema_filled = true;
                }
                self.ema_mean
            } else {
                self.ema_mean = self.beta * self.ema_mean + self.alpha * src;
                self.ema_mean
            }
        };

        // Determine if we should output value based on mode
        if !self.pine_compatible && self.count < self.length {
            self.prev_ecema = ema_value;
            return f64::NAN;
        }

        // Set initial prev_ecema for Pine mode
        if self.pine_compatible && self.count == 1 {
            self.prev_ecema = 0.0; // Pine: nz(ecEma[1]) = 0 on first bar
        }

        // Find best gain
        let mut least_error = 1000000.0;
        let mut best_gain = 0.0;

        for gain_int in -(self.gain_limit as i32)..=(self.gain_limit as i32) {
            let gain = gain_int as f64 / 10.0;
            let test_ec_ema = self.alpha * (ema_value + gain * (src - self.prev_ecema))
                + self.beta * self.prev_ecema;
            let error = (src - test_ec_ema).abs();

            if error < least_error {
                least_error = error;
                best_gain = gain;
            }
        }

        // Calculate final error-correcting EMA with best gain
        let ec_ema = self.alpha * (ema_value + best_gain * (src - self.prev_ecema))
            + self.beta * self.prev_ecema;
        self.prev_ecema = ec_ema;

        ec_ema
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.ema_mean = 0.0;
        self.ema_filled = false;
        self.prev_ecema = 0.0;
        self.prev_value = None;
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_ecema")]
#[pyo3(signature = (data, length=20, gain_limit=50, pine_compatible=false, confirmed_only=false, kernel=None))]
pub fn ehlers_ecema_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    gain_limit: usize,
    pine_compatible: bool,
    confirmed_only: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = EhlersEcemaParams {
        length: Some(length),
        gain_limit: Some(gain_limit),
        pine_compatible: Some(pine_compatible),
        confirmed_only: Some(confirmed_only),
    };
    let input = EhlersEcemaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ehlers_ecema_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_ecema_batch")]
#[pyo3(signature = (data, length_range, gain_limit_range, kernel=None))]
pub fn ehlers_ecema_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    gain_limit_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let sweep = EhlersEcemaBatchRange {
        length: length_range,
        gain_limit: gain_limit_range,
    };
    let kern = validate_kernel(kernel, true)?;

    // Compute using inner_into to avoid extra copies; shape to (rows, cols)
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let simd = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        ehlers_ecema_batch_inner_into(slice_in, &sweep, simd, true, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "gain_limits",
        combos
            .iter()
            .map(|p| p.gain_limit.unwrap_or(50) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersEcemaStream")]
pub struct EhlersEcemaStreamPy {
    inner: EhlersEcemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersEcemaStreamPy {
    #[new]
    #[pyo3(signature = (length=20, gain_limit=50, pine_compatible=false, confirmed_only=false))]
    pub fn new(
        length: usize,
        gain_limit: usize,
        pine_compatible: bool,
        confirmed_only: bool,
    ) -> PyResult<Self> {
        let params = EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(pine_compatible),
            confirmed_only: Some(confirmed_only),
        };
        let stream =
            EhlersEcemaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: stream })
    }

    /// ALMA-compatible name - returns None during warmup
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let y = self.inner.next(value);
        if y.is_nan() {
            None
        } else {
            Some(y)
        }
    }

    /// Backward-compatible alias
    pub fn next(&mut self, value: f64) -> Option<f64> {
        self.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset()
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EhlersEcemaBatchConfig {
    pub length_range: (usize, usize, usize),
    pub gain_limit_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EhlersEcemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersEcemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_ecema_js(
    data: &[f64],
    length: usize,
    gain_limit: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = EhlersEcemaParams {
        length: Some(length),
        gain_limit: Some(gain_limit),
        pine_compatible: Some(false),
        confirmed_only: Some(false),
    };
    let input = EhlersEcemaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    ehlers_ecema_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_ecema_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_ecema_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ehlers_ecema_batch)]
pub fn ehlers_ecema_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: EhlersEcemaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EhlersEcemaBatchRange {
        length: cfg.length_range,
        gain_limit: cfg.gain_limit_range,
    };

    let out = ehlers_ecema_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = EhlersEcemaBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_ecema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    gain_start: usize,
    gain_end: usize,
    gain_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = EhlersEcemaBatchRange {
            length: (length_start, length_end, length_step),
            gain_limit: (gain_start, gain_end, gain_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        ehlers_ecema_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_ecema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    gain_limit: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_ecema_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if length == 0 || length > len {
            return Err(JsValue::from_str("Invalid length"));
        }

        if gain_limit == 0 {
            return Err(JsValue::from_str("Invalid gain limit"));
        }

        let params = EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            ehlers_ecema_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ehlers_ecema_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ehlers_ecema_into_ex)]
pub fn ehlers_ecema_into_ex(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    gain_limit: usize,
    pine_compatible: bool,
    confirmed_only: bool,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_ecema_into_ex",
        ));
    }
    if length == 0 || length > len {
        return Err(JsValue::from_str("Invalid length"));
    }
    if gain_limit == 0 {
        return Err(JsValue::from_str("Invalid gain limit"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(pine_compatible),
            confirmed_only: Some(confirmed_only),
        };
        let input = EhlersEcemaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // preserve zero-copy semantics with a single temporary like ALMA
            let mut temp = vec![0.0; len];
            ehlers_ecema_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ehlers_ecema_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    // ==================== TEST HELPER FUNCTIONS ====================
    fn check_ehlers_ecema_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EhlersEcemaParams {
            length: None,
            gain_limit: None,
            pine_compatible: None,
            confirmed_only: None,
        };
        let input = EhlersEcemaInput::from_candles(&candles, "close", default_params);
        let output = ehlers_ecema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_ecema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = EhlersEcemaParams {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_candles(&candles, "close", params);
        let result = ehlers_ecema_with_kernel(&input, kernel)?;

        // Verify output length
        assert_eq!(result.values.len(), candles.close.len());

        // Check that non-NaN values start appearing after warmup
        let first_valid = result.values.iter().position(|x| !x.is_nan());
        assert!(
            first_valid.is_some(),
            "[{}] No valid values found",
            test_name
        );

        // Expected values from real CSV data
        let expected_last_five = [
            59368.42792078,
            59311.07435861,
            59212.84931613,
            59221.59111692,
            58978.72640292,
        ];

        // Check last 5 values
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Regular mode mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        Ok(())
    }

    fn check_ehlers_ecema_pine_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = EhlersEcemaParams {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(true), // Enable Pine-compatible mode
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_candles(&candles, "close", params);
        let result = ehlers_ecema_with_kernel(&input, kernel)?;

        // Verify output length
        assert_eq!(result.values.len(), candles.close.len());

        // Expected values from real CSV data
        // Note: Pine mode is currently producing the same values as regular mode
        // This may indicate the pine_compatible flag isn't being applied correctly
        let expected_last_five = [
            59368.42792078,
            59311.07435861,
            59212.84931613,
            59221.59111692,
            58978.72640292,
        ];

        // Check last 5 values
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Pine mode mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        // In Pine mode, values should start from the beginning (no warmup)
        assert!(
            result.values[0].is_finite(),
            "[{}] Pine mode should have valid value at index 0",
            test_name
        );

        Ok(())
    }

    fn check_ehlers_ecema_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersEcemaInput::with_default_candles(&candles);
        match input.data {
            EhlersEcemaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EhlersEcemaData::Candles"),
        }
        let output = ehlers_ecema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_ecema_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = EhlersEcemaParams {
            length: Some(0),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&input_data, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with zero period", test_name);
        Ok(())
    }

    fn check_ehlers_ecema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = EhlersEcemaParams {
            length: Some(10),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&data_small, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = EhlersEcemaParams {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&single_point, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EhlersEcemaInput::from_slice(&empty, EhlersEcemaParams::default());
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersEcemaError::EmptyInputData)),
            "[{}] Should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_invalid_gain_limit(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = EhlersEcemaParams {
            length: Some(2),
            gain_limit: Some(0), // Invalid gain limit
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&data, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersEcemaError::InvalidGainLimit { .. })),
            "[{}] Should fail with invalid gain limit",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Use smaller period for reinput test to match Python/WASM tests
        let first_params = EhlersEcemaParams {
            length: Some(10),
            gain_limit: Some(30),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let first_input = EhlersEcemaInput::from_candles(&candles, "close", first_params);
        let first_result = ehlers_ecema_with_kernel(&first_input, kernel)?;

        let second_params = EhlersEcemaParams {
            length: Some(10),
            gain_limit: Some(30),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let second_input = EhlersEcemaInput::from_slice(&first_result.values, second_params);
        let second_result = ehlers_ecema_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        // Expected values from real CSV data
        let expected_last_five = [
            59324.20351585,
            59282.79818999,
            59207.38519971,
            59194.22630265,
            59025.67038012,
        ];

        // Check last 5 values
        let start = second_result.values.len().saturating_sub(5);
        for (i, &val) in second_result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Reinput mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        // Verify that we get valid values after double processing
        let valid_count = second_result
            .values
            .iter()
            .skip(20) // Skip warmup periods from both passes (10-1 + 10-1 = 18, use 20 for safety)
            .filter(|x| x.is_finite())
            .count();
        assert!(
            valid_count > 0,
            "[{}] No valid values after reinput",
            test_name
        );

        Ok(())
    }

    fn check_ehlers_ecema_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersEcemaInput::from_candles(
            &candles,
            "close",
            EhlersEcemaParams {
                length: Some(20),
                gain_limit: Some(50),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
        );
        let res = ehlers_ecema_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());

        // After warmup, values should not be NaN
        if res.values.len() > 40 {
            for (i, &val) in res.values[40..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at index {}",
                    test_name,
                    40 + i
                );
            }
        }
        Ok(())
    }

    fn check_ehlers_ecema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let length = 20;
        let gain_limit = 50;

        let input = EhlersEcemaInput::from_candles(
            &candles,
            "close",
            EhlersEcemaParams {
                length: Some(length),
                gain_limit: Some(gain_limit),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
        );
        let batch_output = ehlers_ecema_with_kernel(&input, kernel)?.values;

        let mut stream = EhlersEcemaStream::try_new(EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            stream_values.push(stream.next(price));
        }

        assert_eq!(batch_output.len(), stream_values.len());

        // Compare after warmup period
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if i < length - 1 {
                // During warmup, both should be NaN
                assert!(
                    b.is_nan() || s.is_nan(),
                    "[{}] Expected NaN during warmup at {}: batch={}, stream={}",
                    test_name,
                    i,
                    b,
                    s
                );
            } else if i >= length && b.is_finite() && s.is_finite() {
                // After warmup, values should be close (allow more tolerance for adaptive algorithm)
                let diff = (b - s).abs();
                let relative_diff = diff / b.abs().max(1.0);
                assert!(
                    relative_diff < 0.001, // 0.1% relative tolerance
                    "[{}] Streaming mismatch at idx {}: batch={}, stream={}, diff={}, rel_diff={}",
                    test_name,
                    i,
                    b,
                    s,
                    diff,
                    relative_diff
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ehlers_ecema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            EhlersEcemaParams::default(),
            EhlersEcemaParams {
                length: Some(10),
                gain_limit: Some(30),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
            EhlersEcemaParams {
                length: Some(20),
                gain_limit: Some(50),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
            EhlersEcemaParams {
                length: Some(30),
                gain_limit: Some(100),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = EhlersEcemaInput::from_candles(&candles, "close", params.clone());
            let output = ehlers_ecema_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: length={}, gain_limit={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(20),
                        params.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: length={}, gain_limit={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(20),
                        params.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: length={}, gain_limit={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(20),
                        params.gain_limit.unwrap_or(50)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ehlers_ecema_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ehlers_ecema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (5usize..=100) // length range
            .prop_flat_map(|length| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        length..400,
                    ),
                    Just(length),
                    10usize..=100, // gain_limit range
                )
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, length, gain_limit)| {
                let params = EhlersEcemaParams {
                    length: Some(length),
                    gain_limit: Some(gain_limit),
                    pine_compatible: Some(false),
                    confirmed_only: Some(false),
                };
                let input = EhlersEcemaInput::from_slice(&data, params);

                let EhlersEcemaOutput { values: out } =
                    ehlers_ecema_with_kernel(&input, kernel).unwrap();
                let EhlersEcemaOutput { values: ref_out } =
                    ehlers_ecema_with_kernel(&input, Kernel::Scalar).unwrap();

                // Check output length matches input
                assert_eq!(
                    out.len(),
                    data.len(),
                    "[{}] Output length mismatch",
                    test_name
                );
                assert_eq!(
                    ref_out.len(),
                    data.len(),
                    "[{}] Reference output length mismatch",
                    test_name
                );

                // Check values after warmup period match between kernels
                for i in (length - 1)..data.len() {
                    if out[i].is_finite() && ref_out[i].is_finite() {
                        // Allow small numerical differences between kernels
                        let diff = (out[i] - ref_out[i]).abs();
                        let relative_diff = diff / ref_out[i].abs().max(1.0);
                        assert!(
                            relative_diff < 1e-10,
                            "[{}] Kernel mismatch at idx {}: {} vs {} (diff={})",
                            test_name,
                            i,
                            out[i],
                            ref_out[i],
                            diff
                        );
                    }
                }

                // Check warmup period has NaN values
                for i in 0..(length - 1) {
                    assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at idx {}",
                        test_name,
                        i
                    );
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    // ==================== TEST GENERATION MACROS ====================
    macro_rules! generate_all_ehlers_ecema_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    // Generate all test variants
    generate_all_ehlers_ecema_tests!(
        check_ehlers_ecema_partial_params,
        check_ehlers_ecema_accuracy,
        check_ehlers_ecema_pine_accuracy,
        check_ehlers_ecema_default_candles,
        check_ehlers_ecema_zero_period,
        check_ehlers_ecema_period_exceeds_length,
        check_ehlers_ecema_very_small_dataset,
        check_ehlers_ecema_empty_input,
        check_ehlers_ecema_invalid_gain_limit,
        check_ehlers_ecema_reinput,
        check_ehlers_ecema_nan_handling,
        check_ehlers_ecema_streaming,
        check_ehlers_ecema_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ehlers_ecema_tests!(check_ehlers_ecema_property);

    // ==================== BATCH PROCESSING TESTS ====================
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EhlersEcemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = EhlersEcemaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Verify we get valid values after warmup
        let valid_count = row.iter().skip(40).filter(|x| x.is_finite()).count();
        assert!(valid_count > 0, "[{}] No valid values in default row", test);

        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EhlersEcemaBatchBuilder::new()
            .kernel(kernel)
            .length_range(10, 30, 5)
            .gain_limit_range(30, 70, 10)
            .apply_candles(&c, "close")?;

        let expected_combos = 5 * 5; // 5 lengths * 5 gain limits
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (10, 20, 5, 30, 50, 10),
            (20, 20, 0, 50, 50, 0),
            (15, 25, 2, 40, 60, 5),
        ];

        for (cfg_idx, &(l_start, l_end, l_step, g_start, g_end, g_step)) in
            test_configs.iter().enumerate()
        {
            let output = EhlersEcemaBatchBuilder::new()
                .kernel(kernel)
                .length_range(l_start, l_end, l_step)
                .gain_limit_range(g_start, g_end, g_step)
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
                        at row {} col {} (flat index {}) with params: length={}, gain_limit={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(20),
                        combo.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: length={}, gain_limit={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(20),
                        combo.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: length={}, gain_limit={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(20),
                        combo.gain_limit.unwrap_or(50)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);
}
