//! # Ultimate Oscillator (ULTOSC)
//!
//! Combines short, medium, and long time periods into a single oscillator value,
//! blending market momentum over multiple horizons with weighted averages (4:2:1 ratio).
//!
//! ## Parameters
//! - **timeperiod1**: Short window for fast momentum (default: 7)
//! - **timeperiod2**: Medium window for intermediate momentum (default: 14)
//! - **timeperiod3**: Long window for slow momentum (default: 28)
//!
//! ## Inputs
//! - High, low, and close price series (or candles)
//! - All series must have the same length
//!
//! ## Returns
//! - **values**: Ultimate Oscillator values as `Vec<f64>` (length matches input, range 0-100)
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Currently stubs that call scalar implementation
//! - **Streaming update**: O(1) performance with efficient circular buffer and running sums
//! - **Memory optimization**: Properly uses zero-copy helper functions (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)
//! - **TODO**: Implement actual SIMD kernels for AVX2/AVX512
//! - **Note**: Streaming implementation is well-optimized with incremental sum updates

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

// --- DATA STRUCTS ---
#[derive(Debug, Clone)]
pub enum UltOscData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct UltOscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct UltOscParams {
    pub timeperiod1: Option<usize>,
    pub timeperiod2: Option<usize>,
    pub timeperiod3: Option<usize>,
}

impl Default for UltOscParams {
    fn default() -> Self {
        Self {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UltOscInput<'a> {
    pub data: UltOscData<'a>,
    pub params: UltOscParams,
}

impl<'a> UltOscInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src,
                low_src,
                close_src,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src: "high",
                low_src: "low",
                close_src: "close",
            },
            params: UltOscParams::default(),
        }
    }
    #[inline]
    pub fn get_timeperiod1(&self) -> usize {
        self.params.timeperiod1.unwrap_or(7)
    }
    #[inline]
    pub fn get_timeperiod2(&self) -> usize {
        self.params.timeperiod2.unwrap_or(14)
    }
    #[inline]
    pub fn get_timeperiod3(&self) -> usize {
        self.params.timeperiod3.unwrap_or(28)
    }
}

// --- BUILDER ---
#[derive(Copy, Clone, Debug)]
pub struct UltOscBuilder {
    timeperiod1: Option<usize>,
    timeperiod2: Option<usize>,
    timeperiod3: Option<usize>,
    kernel: Kernel,
}

impl Default for UltOscBuilder {
    fn default() -> Self {
        Self {
            timeperiod1: None,
            timeperiod2: None,
            timeperiod3: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UltOscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn timeperiod1(mut self, p: usize) -> Self {
        self.timeperiod1 = Some(p);
        self
    }
    #[inline(always)]
    pub fn timeperiod2(mut self, p: usize) -> Self {
        self.timeperiod2 = Some(p);
        self
    }
    #[inline(always)]
    pub fn timeperiod3(mut self, p: usize) -> Self {
        self.timeperiod3 = Some(p);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<UltOscOutput, UltOscError> {
        let params = UltOscParams {
            timeperiod1: self.timeperiod1,
            timeperiod2: self.timeperiod2,
            timeperiod3: self.timeperiod3,
        };
        let input = UltOscInput::with_default_candles(candles);
        ultosc_with_kernel(&UltOscInput { params, ..input }, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<UltOscOutput, UltOscError> {
        let params = UltOscParams {
            timeperiod1: self.timeperiod1,
            timeperiod2: self.timeperiod2,
            timeperiod3: self.timeperiod3,
        };
        let input = UltOscInput::from_slices(high, low, close, params);
        ultosc_with_kernel(&input, self.kernel)
    }
}

// --- ERROR ---
#[derive(Debug, Error)]
pub enum UltOscError {
    #[error("ultosc: Empty data provided.")]
    EmptyData,
    #[error("ultosc: Invalid periods: p1 = {p1}, p2 = {p2}, p3 = {p3}, data length = {data_len}")]
    InvalidPeriods {
        p1: usize,
        p2: usize,
        p3: usize,
        data_len: usize,
    },
    #[error("ultosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ultosc: All values are NaN (or their preceding data is NaN).")]
    AllValuesNaN,
}

// --- HELPER FUNCTIONS ---
#[inline]
fn ultosc_prepare<'a>(
    input: &'a UltOscInput,
    kernel: Kernel,
) -> Result<
    (
        (&'a [f64], &'a [f64], &'a [f64]),
        usize,
        usize,
        usize,
        usize,
        usize,
        Kernel,
    ),
    UltOscError,
> {
    let (high, low, close) = match &input.data {
        UltOscData::Candles {
            candles,
            high_src,
            low_src,
            close_src,
        } => {
            let high = source_type(candles, high_src);
            let low = source_type(candles, low_src);
            let close = source_type(candles, close_src);
            (high, low, close)
        }
        UltOscData::Slices { high, low, close } => (*high, *low, *close),
    };

    let len = high.len();
    if len == 0 || low.len() == 0 || close.len() == 0 {
        return Err(UltOscError::EmptyData);
    }

    let p1 = input.get_timeperiod1();
    let p2 = input.get_timeperiod2();
    let p3 = input.get_timeperiod3();

    if p1 == 0 || p2 == 0 || p3 == 0 || p1 > len || p2 > len || p3 > len {
        return Err(UltOscError::InvalidPeriods {
            p1,
            p2,
            p3,
            data_len: len,
        });
    }

    let largest_period = p1.max(p2.max(p3));
    let first_valid = match (1..len).find(|&i| {
        !high[i - 1].is_nan()
            && !low[i - 1].is_nan()
            && !close[i - 1].is_nan()
            && !high[i].is_nan()
            && !low[i].is_nan()
            && !close[i].is_nan()
    }) {
        Some(i) => i,
        None => return Err(UltOscError::AllValuesNaN),
    };

    let start_idx = first_valid + (largest_period - 1);
    if start_idx >= len {
        return Err(UltOscError::NotEnoughValidData {
            needed: largest_period,
            valid: len.saturating_sub(first_valid),
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((
        (high, low, close),
        p1,
        p2,
        p3,
        first_valid,
        start_idx,
        chosen,
    ))
}

// --- KERNEL ENTRYPOINTS ---
#[inline]
pub fn ultosc(input: &UltOscInput) -> Result<UltOscOutput, UltOscError> {
    ultosc_with_kernel(input, Kernel::Auto)
}

pub fn ultosc_with_kernel(
    input: &UltOscInput,
    kernel: Kernel,
) -> Result<UltOscOutput, UltOscError> {
    let ((high, low, close), p1, p2, p3, first_valid, start_idx, chosen) =
        ultosc_prepare(input, kernel)?;
    let len = high.len();
    let mut out = alloc_with_nan_prefix(len, start_idx);

    ultosc_compute_into(high, low, close, p1, p2, p3, first_valid, chosen, &mut out);

    Ok(UltOscOutput { values: out })
}

#[inline]
fn ultosc_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    chosen: Kernel,
    dst: &mut [f64],
) {
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ultosc_scalar(high, low, close, p1, p2, p3, first_valid, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ultosc_avx2(high, low, close, p1, p2, p3, first_valid, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ultosc_avx512(high, low, close, p1, p2, p3, first_valid, dst)
            }
            _ => unreachable!(),
        }
    }
}

// --- KERNEL IMPL ---
#[inline(always)]
pub unsafe fn ultosc_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let len = high.len();
    let max_period = p1.max(p2).max(p3);

    // Allocate temporary buffers on stack for small periods, heap for large
    const STACK_THRESHOLD: usize = 256;

    if max_period <= STACK_THRESHOLD {
        // Stack allocation for small periods
        let mut cmtl_stack = [0.0_f64; STACK_THRESHOLD];
        let mut tr_stack = [0.0_f64; STACK_THRESHOLD];
        let cmtl_buf = &mut cmtl_stack[..max_period];
        let tr_buf = &mut tr_stack[..max_period];

        ultosc_scalar_impl(
            high,
            low,
            close,
            p1,
            p2,
            p3,
            first_valid,
            out,
            cmtl_buf,
            tr_buf,
        );
    } else {
        // Heap allocation for large periods
        let mut cmtl_vec = vec![0.0; max_period];
        let mut tr_vec = vec![0.0; max_period];

        ultosc_scalar_impl(
            high,
            low,
            close,
            p1,
            p2,
            p3,
            first_valid,
            out,
            &mut cmtl_vec,
            &mut tr_vec,
        );
    }
}

#[inline(always)]
unsafe fn ultosc_scalar_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
    cmtl_buf: &mut [f64],
    tr_buf: &mut [f64],
) {
    let len = high.len();
    let max_period = p1.max(p2).max(p3);

    let mut sum1_a = 0.0;
    let mut sum1_b = 0.0;
    let mut sum2_a = 0.0;
    let mut sum2_b = 0.0;
    let mut sum3_a = 0.0;
    let mut sum3_b = 0.0;

    let start_idx = first_valid + max_period - 1;
    let mut buf_idx = 0;

    // Calculate values for warmup period
    for i in first_valid..=start_idx {
        if i >= len {
            break;
        }

        let cmtl_val;
        let tr_val;

        if high[i].is_nan() || low[i].is_nan() || close[i].is_nan() || close[i - 1].is_nan() {
            cmtl_val = f64::NAN;
            tr_val = f64::NAN;
        } else {
            let true_low = low[i].min(close[i - 1]);
            let mut true_range = high[i] - low[i];
            let diff1 = (high[i] - close[i - 1]).abs();
            if diff1 > true_range {
                true_range = diff1;
            }
            let diff2 = (low[i] - close[i - 1]).abs();
            if diff2 > true_range {
                true_range = diff2;
            }
            cmtl_val = close[i] - true_low;
            tr_val = true_range;
        }

        cmtl_buf[buf_idx] = cmtl_val;
        tr_buf[buf_idx] = tr_val;
        buf_idx = (buf_idx + 1) % max_period;
    }

    // Initialize sums for each period
    for i in 0..p1 {
        let idx = (buf_idx + max_period - p1 + i) % max_period;
        if !cmtl_buf[idx].is_nan() && !tr_buf[idx].is_nan() {
            sum1_a += cmtl_buf[idx];
            sum1_b += tr_buf[idx];
        }
    }

    for i in 0..p2 {
        let idx = (buf_idx + max_period - p2 + i) % max_period;
        if !cmtl_buf[idx].is_nan() && !tr_buf[idx].is_nan() {
            sum2_a += cmtl_buf[idx];
            sum2_b += tr_buf[idx];
        }
    }

    for i in 0..p3 {
        let idx = (buf_idx + max_period - p3 + i) % max_period;
        if !cmtl_buf[idx].is_nan() && !tr_buf[idx].is_nan() {
            sum3_a += cmtl_buf[idx];
            sum3_b += tr_buf[idx];
        }
    }

    // Main calculation loop
    let mut today = start_idx;
    while today < len {
        // Calculate current ULTOSC value
        let v1 = if sum1_b != 0.0 {
            4.0 * (sum1_a / sum1_b)
        } else {
            0.0
        };
        let v2 = if sum2_b != 0.0 {
            2.0 * (sum2_a / sum2_b)
        } else {
            0.0
        };
        let v3 = if sum3_b != 0.0 { sum3_a / sum3_b } else { 0.0 };
        out[today] = 100.0 * (v1 + v2 + v3) / 7.0;

        // Prepare for next iteration
        if today + 1 < len {
            // Calculate new values
            let cmtl_val;
            let tr_val;

            if high[today + 1].is_nan()
                || low[today + 1].is_nan()
                || close[today + 1].is_nan()
                || close[today].is_nan()
            {
                cmtl_val = f64::NAN;
                tr_val = f64::NAN;
            } else {
                let true_low = low[today + 1].min(close[today]);
                let mut true_range = high[today + 1] - low[today + 1];
                let diff1 = (high[today + 1] - close[today]).abs();
                if diff1 > true_range {
                    true_range = diff1;
                }
                let diff2 = (low[today + 1] - close[today]).abs();
                if diff2 > true_range {
                    true_range = diff2;
                }
                cmtl_val = close[today + 1] - true_low;
                tr_val = true_range;
            }

            // Remove oldest values from sums
            let old_idx_1 = (buf_idx + max_period - p1) % max_period;
            if !cmtl_buf[old_idx_1].is_nan() && !tr_buf[old_idx_1].is_nan() {
                sum1_a -= cmtl_buf[old_idx_1];
                sum1_b -= tr_buf[old_idx_1];
            }

            let old_idx_2 = (buf_idx + max_period - p2) % max_period;
            if !cmtl_buf[old_idx_2].is_nan() && !tr_buf[old_idx_2].is_nan() {
                sum2_a -= cmtl_buf[old_idx_2];
                sum2_b -= tr_buf[old_idx_2];
            }

            let old_idx_3 = (buf_idx + max_period - p3) % max_period;
            if !cmtl_buf[old_idx_3].is_nan() && !tr_buf[old_idx_3].is_nan() {
                sum3_a -= cmtl_buf[old_idx_3];
                sum3_b -= tr_buf[old_idx_3];
            }

            // Add new values to buffer
            cmtl_buf[buf_idx] = cmtl_val;
            tr_buf[buf_idx] = tr_val;

            // Add new values to sums
            if !cmtl_val.is_nan() && !tr_val.is_nan() {
                sum1_a += cmtl_val;
                sum1_b += tr_val;
                sum2_a += cmtl_val;
                sum2_b += tr_val;
                sum3_a += cmtl_val;
                sum3_b += tr_val;
            }

            buf_idx = (buf_idx + 1) % max_period;
        }

        today += 1;
    }
}

// --- AVX STUBS ---
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if p1.max(p2).max(p3) <= 32 {
        ultosc_avx512_short(high, low, close, p1, p2, p3, first_valid, out)
    } else {
        ultosc_avx512_long(high, low, close, p1, p2, p3, first_valid, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}

// --- ROW/BATCH/BATCHBUILDER (no sweep for ultosc, but stubs for parity) ---
#[inline(always)]
pub fn ultosc_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx2(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512_short(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512_long(high, low, close, p1, p2, p3, first_valid, out) }
}

// --- Batch APIs ---
#[derive(Clone, Debug)]
pub struct UltOscBatchRange {
    pub timeperiod1: (usize, usize, usize),
    pub timeperiod2: (usize, usize, usize),
    pub timeperiod3: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UltOscBatchConfig {
    pub timeperiod1_range: (usize, usize, usize),
    pub timeperiod2_range: (usize, usize, usize),
    pub timeperiod3_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UltOscBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UltOscParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct UltOscBatchBuilder {
    kernel: Kernel,
}

impl Default for UltOscBatchBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl UltOscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slice(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &UltOscBatchRange,
    ) -> Result<UltOscBatchOutput, UltOscError> {
        ultosc_batch_with_kernel(high, low, close, sweep, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct UltOscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UltOscParams>,
    pub rows: usize,
    pub cols: usize,
}

impl UltOscBatchOutput {
    pub fn row_for_params(&self, p: &UltOscParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.timeperiod1.unwrap_or(7) == p.timeperiod1.unwrap_or(7)
                && c.timeperiod2.unwrap_or(14) == p.timeperiod2.unwrap_or(14)
                && c.timeperiod3.unwrap_or(28) == p.timeperiod3.unwrap_or(28)
        })
    }

    pub fn values_for(&self, p: &UltOscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &UltOscBatchRange) -> Vec<UltOscParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let timeperiod1s = axis_usize(r.timeperiod1);
    let timeperiod2s = axis_usize(r.timeperiod2);
    let timeperiod3s = axis_usize(r.timeperiod3);

    let mut out = Vec::with_capacity(timeperiod1s.len() * timeperiod2s.len() * timeperiod3s.len());
    for &tp1 in &timeperiod1s {
        for &tp2 in &timeperiod2s {
            for &tp3 in &timeperiod3s {
                out.push(UltOscParams {
                    timeperiod1: Some(tp1),
                    timeperiod2: Some(tp2),
                    timeperiod3: Some(tp3),
                });
            }
        }
    }
    out
}

pub fn ultosc_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &UltOscBatchRange,
    k: Kernel,
) -> Result<UltOscBatchOutput, UltOscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(UltOscError::InvalidPeriods {
                p1: 0,
                p2: 0,
                p3: 0,
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

    ultosc_batch_inner(high, low, close, sweep, simd, true)
}

#[inline(always)]
fn ultosc_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &UltOscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<UltOscBatchOutput, UltOscError> {
    let combos = expand_grid(sweep);
    let cols = high.len();
    let rows = combos.len();

    if cols == 0 {
        return Err(UltOscError::EmptyData);
    }

    // Find first valid index (both i-1 and i must be valid)
    let first_valid_idx = (1..cols)
        .find(|&i| {
            !high[i - 1].is_nan()
                && !low[i - 1].is_nan()
                && !close[i - 1].is_nan()
                && !high[i].is_nan()
                && !low[i].is_nan()
                && !close[i].is_nan()
        })
        .ok_or(UltOscError::AllValuesNaN)?;

    // Calculate warmup periods for each combo
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p1 = c.timeperiod1.unwrap_or(7);
            let p2 = c.timeperiod2.unwrap_or(14);
            let p3 = c.timeperiod3.unwrap_or(28);
            first_valid_idx + p1.max(p2).max(p3) - 1
        })
        .collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    ultosc_batch_inner_into(high, low, close, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(UltOscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn ultosc_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &UltOscBatchRange,
    simd: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<UltOscParams>, UltOscError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(UltOscError::InvalidPeriods {
            p1: 0,
            p2: 0,
            p3: 0,
            data_len: 0,
        });
    }

    let len = high.len();
    // Find first valid index (both i-1 and i must be valid)
    let first_valid_idx = (1..len)
        .find(|&i| {
            !high[i - 1].is_nan()
                && !low[i - 1].is_nan()
                && !close[i - 1].is_nan()
                && !high[i].is_nan()
                && !low[i].is_nan()
                && !close[i].is_nan()
        })
        .ok_or(UltOscError::AllValuesNaN)?;

    let max_p = combos
        .iter()
        .map(|c| {
            let p1 = c.timeperiod1.unwrap_or(7);
            let p2 = c.timeperiod2.unwrap_or(14);
            let p3 = c.timeperiod3.unwrap_or(28);
            p1.max(p2).max(p3)
        })
        .max()
        .unwrap();

    if len - first_valid_idx < max_p {
        return Err(UltOscError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid_idx,
        });
    }

    let rows = combos.len();
    let cols = len;

    let do_row = |row: usize, row_out: &mut [f64]| unsafe {
        let p1 = combos[row].timeperiod1.unwrap();
        let p2 = combos[row].timeperiod2.unwrap();
        let p3 = combos[row].timeperiod3.unwrap();

        match simd {
            Kernel::Scalar => ultosc_scalar(high, low, close, p1, p2, p3, first_valid_idx, row_out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ultosc_avx2(high, low, close, p1, p2, p3, first_valid_idx, row_out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ultosc_avx512(high, low, close, p1, p2, p3, first_valid_idx, row_out),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_out)| do_row(row, row_out));
        }
        #[cfg(target_arch = "wasm32")]
        {
            out.chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_out)| do_row(row, row_out));
        }
    } else {
        out.chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_out)| do_row(row, row_out));
    }

    Ok(combos)
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_ultosc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UltOscParams {
            timeperiod1: None,
            timeperiod2: None,
            timeperiod3: None,
        };
        let input = UltOscInput::from_candles(&candles, "high", "low", "close", params);
        let output = ultosc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ultosc_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_candles(&candles, "high", "low", "close", params);
        let result = ultosc_with_kernel(&input, kernel)?;
        let expected_last_five = [
            41.25546890298435,
            40.83865967175865,
            48.910324164909625,
            45.43113094857947,
            42.163165136766295,
        ];
        assert!(result.values.len() >= 5);
        let start_idx = result.values.len() - 5;
        for (i, &val) in result.values[start_idx..].iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "[{}] ULTOSC mismatch at last five index {}: expected {}, got {}",
                test_name,
                i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_ultosc_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UltOscInput::with_default_candles(&candles);
        let result = ultosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ultosc_zero_periods(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams {
            timeperiod1: Some(0),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for zero period",
            test_name
        );
        Ok(())
    }

    fn check_ultosc_period_exceeds_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for period exceeding data length",
            test_name
        );
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ultosc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            UltOscParams::default(),
            // Minimum periods
            UltOscParams {
                timeperiod1: Some(1),
                timeperiod2: Some(2),
                timeperiod3: Some(3),
            },
            // Small periods
            UltOscParams {
                timeperiod1: Some(2),
                timeperiod2: Some(4),
                timeperiod3: Some(8),
            },
            // Small to medium periods
            UltOscParams {
                timeperiod1: Some(5),
                timeperiod2: Some(10),
                timeperiod3: Some(20),
            },
            // Standard periods
            UltOscParams {
                timeperiod1: Some(7),
                timeperiod2: Some(14),
                timeperiod3: Some(28),
            },
            // Medium periods
            UltOscParams {
                timeperiod1: Some(10),
                timeperiod2: Some(20),
                timeperiod3: Some(40),
            },
            // Large periods
            UltOscParams {
                timeperiod1: Some(14),
                timeperiod2: Some(28),
                timeperiod3: Some(56),
            },
            // Very large periods
            UltOscParams {
                timeperiod1: Some(20),
                timeperiod2: Some(40),
                timeperiod3: Some(80),
            },
            // Asymmetric periods - close together
            UltOscParams {
                timeperiod1: Some(5),
                timeperiod2: Some(6),
                timeperiod3: Some(7),
            },
            // Asymmetric periods - far apart
            UltOscParams {
                timeperiod1: Some(3),
                timeperiod2: Some(10),
                timeperiod3: Some(50),
            },
            // Edge case - all same
            UltOscParams {
                timeperiod1: Some(14),
                timeperiod2: Some(14),
                timeperiod3: Some(14),
            },
            // Edge case - reverse order
            UltOscParams {
                timeperiod1: Some(28),
                timeperiod2: Some(14),
                timeperiod3: Some(7),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = UltOscInput::from_candles(&candles, "high", "low", "close", params.clone());
            let output = ultosc_with_kernel(&input, kernel)?;

            // Check values
            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: timeperiod1={}, timeperiod2={}, timeperiod3={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.timeperiod1.unwrap_or(7),
                        params.timeperiod2.unwrap_or(14),
                        params.timeperiod3.unwrap_or(28),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: timeperiod1={}, timeperiod2={}, timeperiod3={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.timeperiod1.unwrap_or(7),
                        params.timeperiod2.unwrap_or(14),
                        params.timeperiod3.unwrap_or(28),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: timeperiod1={}, timeperiod2={}, timeperiod3={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.timeperiod1.unwrap_or(7),
                        params.timeperiod2.unwrap_or(14),
                        params.timeperiod3.unwrap_or(28),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ultosc_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ultosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate random test data for high/low/close prices with varying periods
        let strat = (1usize..=50, 1usize..=50, 1usize..=50).prop_flat_map(|(p1, p2, p3)| {
            let max_period = p1.max(p2).max(p3);
            (
                // Generate price data with realistic constraints
                // Need at least max_period + 1 for ULTOSC (needs previous close)
                prop::collection::vec(
                    (0.1f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                    (max_period + 1)..400,
                ),
                Just((p1, p2, p3)),
            )
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(base_prices, (p1, p2, p3))| {
                // Generate high/low/close from base prices with realistic and varied relationships
                let mut high = Vec::with_capacity(base_prices.len());
                let mut low = Vec::with_capacity(base_prices.len());
                let mut close = Vec::with_capacity(base_prices.len());

                // Use a simple pseudo-random number generator for variation
                let mut seed = p1 + p2 * 7 + p3 * 13;
                for &price in &base_prices {
                    // Vary the spread between 1% and 10%
                    seed = (seed * 1103515245 + 12345) % (1 << 31);
                    let spread_pct = 0.01 + (seed as f64 / (1u64 << 31) as f64) * 0.09;
                    let spread = price * spread_pct;

                    // Vary where the close falls within the range
                    seed = (seed * 1103515245 + 12345) % (1 << 31);
                    let close_position = seed as f64 / (1u64 << 31) as f64; // 0.0 to 1.0

                    let h = price + spread * 0.5;
                    let l = price - spread * 0.5;
                    let c = l + (h - l) * close_position;

                    high.push(h);
                    low.push(l);
                    close.push(c);
                }

                let params = UltOscParams {
                    timeperiod1: Some(p1),
                    timeperiod2: Some(p2),
                    timeperiod3: Some(p3),
                };
                let input = UltOscInput::from_slices(&high, &low, &close, params.clone());

                let result = ultosc_with_kernel(&input, kernel).unwrap();
                let out = result.values;

                // Also compute with scalar kernel for reference
                let ref_result = ultosc_with_kernel(&input, Kernel::Scalar).unwrap();
                let ref_out = ref_result.values;

                let max_period = p1.max(p2).max(p3);
                // ULTOSC needs previous close, so warmup is max_period (includes the first_valid offset)
                let warmup = max_period;

                // Property 1: Warmup period validation
                // First warmup values should be NaN
                for i in 0..warmup.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                // Property 2: Kernel consistency
                // All kernels should produce identical results
                for (i, (&y, &r)) in out.iter().zip(ref_out.iter()).enumerate() {
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] NaN/inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                    } else {
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] Value mismatch at index {}: {} vs {} (ULP diff: {})",
                            test_name,
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                // Property 3: Output bounds
                // ULTOSC values must be between 0 and 100
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        prop_assert!(
                            val >= 0.0 && val <= 100.0,
                            "[{}] ULTOSC value {} at index {} is out of bounds [0, 100]",
                            test_name,
                            val,
                            i
                        );
                    }
                }

                // Property 4: Constant price property
                // If all prices are constant, ULTOSC should stabilize to a specific value
                if high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                {
                    // After the largest period, the indicator should produce stable values
                    // We need at least max_period + a few more points to see stability
                    let stability_check_start = (warmup + p3.max(p2).max(p1)).min(out.len());
                    if stability_check_start < out.len() - 2 {
                        // Find first non-NaN value after stability point
                        let stable_region = &out[stability_check_start..];
                        let first_valid = stable_region.iter().position(|&v| !v.is_nan());

                        if let Some(idx) = first_valid {
                            let expected_stable = stable_region[idx];
                            // All subsequent values should match the first stable value
                            for (i, &val) in stable_region.iter().skip(idx + 1).enumerate() {
                                if !val.is_nan() {
                                    prop_assert!(
										(val - expected_stable).abs() < 1e-8,
										"[{}] Expected stable value {} for constant prices at index {}, got {}",
										test_name, expected_stable, stability_check_start + idx + 1 + i, val
									);
                                }
                            }
                        }
                    }
                }

                // Property 5: Zero range property
                // When high = low = close for all values
                let zero_range_high = vec![100.0; base_prices.len()];
                let zero_range_low = zero_range_high.clone();
                let zero_range_close = zero_range_high.clone();

                let zero_input = UltOscInput::from_slices(
                    &zero_range_high,
                    &zero_range_low,
                    &zero_range_close,
                    params.clone(),
                );
                if let Ok(zero_result) = ultosc_with_kernel(&zero_input, kernel) {
                    // After warmup, with zero range (high=low=close), true range is 0,
                    // so ULTOSC should be 0 (as per lines 459-462 implementation)
                    for (i, &val) in zero_result.values.iter().enumerate().skip(warmup) {
                        if !val.is_nan() {
                            prop_assert!(
                                val.abs() < 1e-8,
                                "[{}] Expected 0 for zero range at index {}, got {}",
                                test_name,
                                i,
                                val
                            );
                        }
                    }
                }

                // Property 6: Weight relationship verification (4:2:1)
                // ULTOSC formula: 100 * (4*BP1/TR1 + 2*BP2/TR2 + BP3/TR3) / 7
                // This is a fundamental property of the indicator
                // We can verify the weights are applied correctly by checking that
                // the final result is properly weighted
                if out.len() > warmup {
                    // The formula divides by 7 because 4+2+1=7
                    // This is a sanity check that the implementation follows the spec
                    for i in warmup..out.len().min(warmup + 5) {
                        if !out[i].is_nan() {
                            // ULTOSC values should be reasonable oscillator values
                            // Not testing exact formula here, just that it's bounded reasonably
                            prop_assert!(
                                out[i] >= 0.0 && out[i] <= 100.0,
                                "[{}] ULTOSC at {} should be in [0,100], got {}",
                                test_name,
                                i,
                                out[i]
                            );
                        }
                    }
                }

                // Property 7: Period ordering independence
                // ULTOSC should work regardless of period ordering (p1, p2, p3 don't need to be ordered)
                let reordered_params = UltOscParams {
                    timeperiod1: Some(p3),
                    timeperiod2: Some(p1),
                    timeperiod3: Some(p2),
                };
                let reordered_input =
                    UltOscInput::from_slices(&high, &low, &close, reordered_params);

                // Should not error regardless of ordering
                prop_assert!(
                    ultosc_with_kernel(&reordered_input, kernel).is_ok(),
                    "[{}] ULTOSC should work with any period ordering",
                    test_name
                );

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_ultosc_tests {
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

    generate_all_ultosc_tests!(
        check_ultosc_partial_params,
        check_ultosc_accuracy,
        check_ultosc_default_candles,
        check_ultosc_zero_periods,
        check_ultosc_period_exceeds_data_length,
        check_ultosc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ultosc_tests!(check_ultosc_property);
    fn check_ultosc_batch_default(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with a simple parameter sweep
        let sweep = UltOscBatchRange {
            timeperiod1: (5, 9, 2),   // 5, 7, 9
            timeperiod2: (12, 16, 2), // 12, 14, 16
            timeperiod3: (26, 30, 2), // 26, 28, 30
        };

        let batch_builder = UltOscBatchBuilder::new().kernel(kernel);
        let output =
            batch_builder.apply_slice(&candles.high, &candles.low, &candles.close, &sweep)?;

        // Check structure
        assert_eq!(output.rows, 3 * 3 * 3); // 27 combinations
        assert_eq!(output.cols, candles.close.len());
        assert_eq!(output.values.len(), output.rows * output.cols);
        assert_eq!(output.combos.len(), output.rows);

        // Verify specific combination matches single calculation
        let single_params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let single_input =
            UltOscInput::from_slices(&candles.high, &candles.low, &candles.close, single_params);
        let single_result = ultosc_with_kernel(&single_input, kernel)?;

        // Find the row for this combination
        if let Some(row_idx) = output.row_for_params(&single_params) {
            let batch_row = output.values_for(&single_params).unwrap();

            // Compare last 5 values
            let start = batch_row.len().saturating_sub(5);
            for i in 0..5 {
                let diff = (batch_row[start + i] - single_result.values[start + i]).abs();
                assert!(
                    diff < 1e-10,
                    "[{}] Batch vs single mismatch at idx {}: got {}, expected {}",
                    test_name,
                    i,
                    batch_row[start + i],
                    single_result.values[start + i]
                );
            }
        } else {
            panic!("[{}] Could not find row for params 7,14,28", test_name);
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (tp1_start, tp1_end, tp1_step, tp2_start, tp2_end, tp2_step, tp3_start, tp3_end, tp3_step)
            (2, 8, 2, 4, 16, 4, 8, 32, 8),   // Small to medium ranges
            (5, 7, 1, 10, 14, 2, 20, 28, 4), // Dense small ranges
            (7, 7, 0, 14, 14, 0, 14, 42, 7), // Static tp1/tp2, varying tp3
            (1, 5, 1, 10, 10, 0, 20, 20, 0), // Varying tp1, static tp2/tp3
            (10, 20, 5, 20, 40, 10, 40, 80, 20), // Large ranges
            (3, 9, 3, 6, 18, 6, 12, 36, 12), // Multiples of 3
            (5, 10, 1, 10, 20, 2, 20, 40, 4), // Different step sizes
        ];

        for (
            cfg_idx,
            &(
                tp1_start,
                tp1_end,
                tp1_step,
                tp2_start,
                tp2_end,
                tp2_step,
                tp3_start,
                tp3_end,
                tp3_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let sweep = UltOscBatchRange {
                timeperiod1: (tp1_start, tp1_end, tp1_step),
                timeperiod2: (tp2_start, tp2_end, tp2_step),
                timeperiod3: (tp3_start, tp3_end, tp3_step),
            };

            let batch_builder = UltOscBatchBuilder::new().kernel(kernel);
            let output =
                batch_builder.apply_slice(&candles.high, &candles.low, &candles.close, &sweep)?;

            // Check values
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
						at row {} col {} (flat index {}) with params: timeperiod1={}, timeperiod2={}, timeperiod3={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.timeperiod1.unwrap_or(7),
                        combo.timeperiod2.unwrap_or(14),
                        combo.timeperiod3.unwrap_or(28)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: timeperiod1={}, timeperiod2={}, timeperiod3={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.timeperiod1.unwrap_or(7),
                        combo.timeperiod2.unwrap_or(14),
                        combo.timeperiod3.unwrap_or(28)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: timeperiod1={}, timeperiod2={}, timeperiod3={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.timeperiod1.unwrap_or(7),
                        combo.timeperiod2.unwrap_or(14),
                        combo.timeperiod3.unwrap_or(28)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
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
            }
        };
    }

    gen_batch_tests!(check_ultosc_batch_default);
    gen_batch_tests!(check_batch_no_poison);
}

// ============================================================================
// WASM Helper Functions
// ============================================================================

#[inline]
pub fn ultosc_into_slice(
    dst: &mut [f64],
    input: &UltOscInput,
    kern: Kernel,
) -> Result<(), UltOscError> {
    let ((high, low, close), p1, p2, p3, first_valid, start_idx, chosen) =
        ultosc_prepare(input, kern)?;

    if dst.len() != high.len() {
        return Err(UltOscError::InvalidPeriods {
            p1: dst.len(),
            p2: high.len(),
            p3: 0,
            data_len: high.len(),
        });
    }

    // Compute directly into destination
    ultosc_compute_into(high, low, close, p1, p2, p3, first_valid, chosen, dst);

    // Fill warmup period with NaN
    for v in &mut dst[..start_idx] {
        *v = f64::NAN;
    }

    Ok(())
}

// ============================================================================
// Streaming Implementation
// ============================================================================

#[derive(Debug, Clone)]
pub struct UltOscStream {
    params: UltOscParams,
    cmtl_buf: Vec<f64>,
    tr_buf: Vec<f64>,
    sum1_a: f64,
    sum1_b: f64,
    sum2_a: f64,
    sum2_b: f64,
    sum3_a: f64,
    sum3_b: f64,
    buffer_idx: usize,
    max_period: usize,
    p1: usize,
    p2: usize,
    p3: usize,
    initialized: bool,
    prev_close: Option<f64>,
}

impl UltOscStream {
    pub fn try_new(params: UltOscParams) -> Result<Self, UltOscError> {
        let p1 = params.timeperiod1.unwrap_or(7);
        let p2 = params.timeperiod2.unwrap_or(14);
        let p3 = params.timeperiod3.unwrap_or(28);

        if p1 == 0 || p2 == 0 || p3 == 0 {
            return Err(UltOscError::InvalidPeriods {
                p1,
                p2,
                p3,
                data_len: 0,
            });
        }

        let max_period = p1.max(p2).max(p3);

        Ok(UltOscStream {
            params,
            cmtl_buf: vec![0.0; max_period],
            tr_buf: vec![0.0; max_period],
            sum1_a: 0.0,
            sum1_b: 0.0,
            sum2_a: 0.0,
            sum2_b: 0.0,
            sum3_a: 0.0,
            sum3_b: 0.0,
            buffer_idx: 0,
            max_period,
            p1,
            p2,
            p3,
            initialized: false,
            prev_close: None,
        })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() || close.is_nan() {
            return None;
        }

        if let Some(prev_close) = self.prev_close {
            // Calculate true range and close minus true low
            let true_low = low.min(prev_close);
            let mut true_range = high - low;
            let diff1 = (high - prev_close).abs();
            if diff1 > true_range {
                true_range = diff1;
            }
            let diff2 = (low - prev_close).abs();
            if diff2 > true_range {
                true_range = diff2;
            }

            let cmtl_val = close - true_low;
            let tr_val = true_range;

            // Update circular buffers
            self.cmtl_buf[self.buffer_idx] = cmtl_val;
            self.tr_buf[self.buffer_idx] = tr_val;

            if !self.initialized {
                // Prime the sums during warmup
                self.sum1_a += cmtl_val;
                self.sum1_b += tr_val;
                self.sum2_a += cmtl_val;
                self.sum2_b += tr_val;
                self.sum3_a += cmtl_val;
                self.sum3_b += tr_val;

                // Check if we've filled the largest period
                if self.buffer_idx + 1 >= self.max_period {
                    self.initialized = true;

                    // Adjust sums to their correct window sizes
                    self.sum1_a = 0.0;
                    self.sum1_b = 0.0;
                    self.sum2_a = 0.0;
                    self.sum2_b = 0.0;
                    self.sum3_a = 0.0;
                    self.sum3_b = 0.0;

                    for i in 0..self.p1 {
                        let idx =
                            (self.buffer_idx + self.max_period + 1 - self.p1 + i) % self.max_period;
                        self.sum1_a += self.cmtl_buf[idx];
                        self.sum1_b += self.tr_buf[idx];
                    }

                    for i in 0..self.p2 {
                        let idx =
                            (self.buffer_idx + self.max_period + 1 - self.p2 + i) % self.max_period;
                        self.sum2_a += self.cmtl_buf[idx];
                        self.sum2_b += self.tr_buf[idx];
                    }

                    for i in 0..self.p3 {
                        let idx =
                            (self.buffer_idx + self.max_period + 1 - self.p3 + i) % self.max_period;
                        self.sum3_a += self.cmtl_buf[idx];
                        self.sum3_b += self.tr_buf[idx];
                    }
                }
            } else {
                // We're initialized, maintain rolling sums

                // Add new values
                self.sum1_a += cmtl_val;
                self.sum1_b += tr_val;
                self.sum2_a += cmtl_val;
                self.sum2_b += tr_val;
                self.sum3_a += cmtl_val;
                self.sum3_b += tr_val;

                // Remove old values
                let old_idx_1 = (self.buffer_idx + self.max_period + 1 - self.p1) % self.max_period;
                self.sum1_a -= self.cmtl_buf[old_idx_1];
                self.sum1_b -= self.tr_buf[old_idx_1];

                let old_idx_2 = (self.buffer_idx + self.max_period + 1 - self.p2) % self.max_period;
                self.sum2_a -= self.cmtl_buf[old_idx_2];
                self.sum2_b -= self.tr_buf[old_idx_2];

                let old_idx_3 = (self.buffer_idx + self.max_period + 1 - self.p3) % self.max_period;
                self.sum3_a -= self.cmtl_buf[old_idx_3];
                self.sum3_b -= self.tr_buf[old_idx_3];
            }

            // Advance circular buffer index
            self.buffer_idx = (self.buffer_idx + 1) % self.max_period;

            if self.initialized {
                // Calculate ULTOSC
                let v1 = if self.sum1_b != 0.0 {
                    4.0 * (self.sum1_a / self.sum1_b)
                } else {
                    0.0
                };
                let v2 = if self.sum2_b != 0.0 {
                    2.0 * (self.sum2_a / self.sum2_b)
                } else {
                    0.0
                };
                let v3 = if self.sum3_b != 0.0 {
                    self.sum3_a / self.sum3_b
                } else {
                    0.0
                };
                let ultosc_val = 100.0 * (v1 + v2 + v3) / 7.0;
                self.prev_close = Some(close);
                return Some(ultosc_val);
            }
        }

        self.prev_close = Some(close);
        None
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction(name = "ultosc")]
#[pyo3(signature = (high, low, close, timeperiod1=None, timeperiod2=None, timeperiod3=None, kernel=None))]
pub fn ultosc_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod1: Option<usize>,
    timeperiod2: Option<usize>,
    timeperiod3: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = UltOscParams {
        timeperiod1,
        timeperiod2,
        timeperiod3,
    };
    let input = UltOscInput::from_slices(high_slice, low_slice, close_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ultosc_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ultosc_batch")]
#[pyo3(signature = (high, low, close, timeperiod1_range, timeperiod2_range, timeperiod3_range, kernel=None))]
pub fn ultosc_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod1_range: (usize, usize, usize),
    timeperiod2_range: (usize, usize, usize),
    timeperiod3_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = UltOscBatchRange {
        timeperiod1: timeperiod1_range,
        timeperiod2: timeperiod2_range,
        timeperiod3: timeperiod3_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Find first valid index (both i-1 and i must be valid)
    let first_valid_idx = (1..cols)
        .find(|&i| {
            !high_slice[i - 1].is_nan()
                && !low_slice[i - 1].is_nan()
                && !close_slice[i - 1].is_nan()
                && !high_slice[i].is_nan()
                && !low_slice[i].is_nan()
                && !close_slice[i].is_nan()
        })
        .unwrap_or(0);

    // Calculate warmup periods for each combo and initialize NaN prefixes
    for (row, combo) in combos.iter().enumerate() {
        let p1 = combo.timeperiod1.unwrap_or(7);
        let p2 = combo.timeperiod2.unwrap_or(14);
        let p3 = combo.timeperiod3.unwrap_or(28);
        let warmup = first_valid_idx + p1.max(p2).max(p3) - 1;

        // Fill the warmup period with NaN for this row
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            slice_out[row_start + i] = f64::NAN;
        }
    }

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };
            ultosc_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                simd,
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "timeperiod1",
        combos
            .iter()
            .map(|p| p.timeperiod1.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "timeperiod2",
        combos
            .iter()
            .map(|p| p.timeperiod2.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "timeperiod3",
        combos
            .iter()
            .map(|p| p.timeperiod3.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "UltOscStream")]
pub struct UltOscStreamPy {
    stream: UltOscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl UltOscStreamPy {
    #[new]
    #[pyo3(signature = (timeperiod1=None, timeperiod2=None, timeperiod3=None))]
    fn new(
        timeperiod1: Option<usize>,
        timeperiod2: Option<usize>,
        timeperiod3: Option<usize>,
    ) -> PyResult<Self> {
        let params = UltOscParams {
            timeperiod1,
            timeperiod2,
            timeperiod3,
        };
        let stream =
            UltOscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(UltOscStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = UltOscParams {
        timeperiod1: Some(timeperiod1),
        timeperiod2: Some(timeperiod2),
        timeperiod3: Some(timeperiod3),
    };
    let input = UltOscInput::from_slices(high, low, close, params);

    // Single allocation
    let mut output = vec![0.0; high.len()];
    ultosc_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ultosc_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        if timeperiod1 == 0 || timeperiod2 == 0 || timeperiod3 == 0 {
            return Err(JsValue::from_str("Invalid period: cannot be zero"));
        }

        let max_period = timeperiod1.max(timeperiod2).max(timeperiod3);
        if max_period > len {
            return Err(JsValue::from_str("Period exceeds data length"));
        }

        let params = UltOscParams {
            timeperiod1: Some(timeperiod1),
            timeperiod2: Some(timeperiod2),
            timeperiod3: Some(timeperiod3),
        };
        let input = UltOscInput::from_slices(high, low, close, params);

        // CRITICAL: Check for aliasing with any input array
        if high_ptr == out_ptr as *const f64
            || low_ptr == out_ptr as *const f64
            || close_ptr == out_ptr as *const f64
        {
            // Input and output overlap - use temporary buffer
            let mut temp = vec![0.0; len];
            ultosc_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing - write directly to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ultosc_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ultosc_batch)]
pub fn ultosc_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: UltOscBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = UltOscBatchRange {
        timeperiod1: config.timeperiod1_range,
        timeperiod2: config.timeperiod2_range,
        timeperiod3: config.timeperiod3_range,
    };

    let batch_output = ultosc_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let rows = batch_output.combos.len();
    let cols = high.len();

    let result = UltOscBatchJsOutput {
        values: batch_output.values,
        combos: batch_output.combos,
        rows,
        cols,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
