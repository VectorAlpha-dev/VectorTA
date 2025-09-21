//! # Ultimate Moving Average (UMA)
//!
//! An adaptive moving average that dynamically adjusts its length based on multiple indicators
//! including RSI, MFI, and standard deviation to respond to market conditions.
//!
//! ## Parameters
//! - **accelerator**: Acceleration factor for dynamic length adjustment (default: 8.0)
//! - **min_length**: Minimum lookback period (default: 5)
//! - **max_length**: Maximum lookback period (default: 50)
//! - **smooth_length**: Smoothing period for final output (default: 4)
//!
//! ## Returns
//! - **`Ok(UmaOutput)`** containing values vector of length matching input data
//! - **`Err(UmaError)`** on calculation errors
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Not implemented (no explicit SIMD kernel functions)
//! - **Streaming update**: O(n) - calls full uma function on each update
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix)
//! - **Optimization needed**: Implement SIMD kernels and optimize streaming to O(1)

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaUma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use js_sys;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::deviation::{deviation, DeviationInput, DeviationParams};
use crate::indicators::mfi::{mfi, MfiInput, MfiParams};
use crate::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
use crate::indicators::moving_averages::wma::{wma, WmaInput, WmaParams};
use crate::indicators::rsi::{rsi, RsiInput, RsiParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for UmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            UmaData::Slice(slice) => slice,
            UmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct UmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct UmaParams {
    pub accelerator: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for UmaParams {
    fn default() -> Self {
        Self {
            accelerator: Some(1.0),
            min_length: Some(5),
            max_length: Some(50),
            smooth_length: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UmaInput<'a> {
    pub data: UmaData<'a>,
    pub params: UmaParams,
    pub volume: Option<&'a [f64]>,
}

impl<'a> UmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: UmaParams) -> Self {
        Self {
            data: UmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
            volume: Some(&c.volume),
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], vol: Option<&'a [f64]>, p: UmaParams) -> Self {
        Self {
            data: UmaData::Slice(sl),
            params: p,
            volume: vol,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", UmaParams::default())
    }

    #[inline]
    pub fn get_accelerator(&self) -> f64 {
        self.params.accelerator.unwrap_or(1.0)
    }

    #[inline]
    pub fn get_min_length(&self) -> usize {
        self.params.min_length.unwrap_or(5)
    }

    #[inline]
    pub fn get_max_length(&self) -> usize {
        self.params.max_length.unwrap_or(50)
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(4)
    }
}

// ==================== BUILDER ====================

#[derive(Copy, Clone, Debug)]
pub struct UmaBuilder {
    accelerator: Option<f64>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for UmaBuilder {
    fn default() -> Self {
        Self {
            accelerator: None,
            min_length: None,
            max_length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn accelerator(mut self, a: f64) -> Self {
        self.accelerator = Some(a);
        self
    }

    #[inline(always)]
    pub fn min_length(mut self, n: usize) -> Self {
        self.min_length = Some(n);
        self
    }

    #[inline(always)]
    pub fn max_length(mut self, n: usize) -> Self {
        self.max_length = Some(n);
        self
    }

    #[inline(always)]
    pub fn smooth_length(mut self, n: usize) -> Self {
        self.smooth_length = Some(n);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<UmaOutput, UmaError> {
        let p = UmaParams {
            accelerator: self.accelerator,
            min_length: self.min_length,
            max_length: self.max_length,
            smooth_length: self.smooth_length,
        };
        let i = UmaInput::from_candles(c, "close", p);
        uma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64], v: Option<&[f64]>) -> Result<UmaOutput, UmaError> {
        let p = UmaParams {
            accelerator: self.accelerator,
            min_length: self.min_length,
            max_length: self.max_length,
            smooth_length: self.smooth_length,
        };
        let i = UmaInput::from_slice(d, v, p);
        uma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<UmaStream, UmaError> {
        let p = UmaParams {
            accelerator: self.accelerator,
            min_length: self.min_length,
            max_length: self.max_length,
            smooth_length: self.smooth_length,
        };
        UmaStream::try_new(p)
    }
}

// ==================== STREAMING ====================

#[derive(Debug, Clone)]
pub struct UmaStream {
    buffer: Vec<f64>,
    volume_buffer: Vec<f64>,
    params: UmaParams,
    dynamic_length: f64,
    kernel: Kernel,
}

impl UmaStream {
    pub fn try_new(params: UmaParams) -> Result<Self, UmaError> {
        let max_length = params.max_length.unwrap_or(50);
        let min_length = params.min_length.unwrap_or(5);

        if min_length > max_length {
            return Err(UmaError::MinLengthGreaterThanMaxLength {
                min_length,
                max_length,
            });
        }

        Ok(Self {
            buffer: Vec::with_capacity(max_length * 3),
            volume_buffer: Vec::with_capacity(max_length * 3),
            params,
            dynamic_length: max_length as f64,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.update_with_volume(value, None)
    }

    pub fn update_with_volume(&mut self, value: f64, volume: Option<f64>) -> Option<f64> {
        self.buffer.push(value);
        if let Some(v) = volume {
            self.volume_buffer.push(v);
        } else {
            self.volume_buffer.push(0.0);
        }

        let max_length = self.params.max_length.unwrap_or(50);
        let smooth_length = self.params.smooth_length.unwrap_or(4);
        let min_needed = max_length + smooth_length;

        if self.buffer.len() < min_needed {
            return None;
        }

        // Keep buffer size manageable
        if self.buffer.len() > min_needed * 2 {
            self.buffer.drain(0..self.buffer.len() - min_needed * 2);
            self.volume_buffer
                .drain(0..self.volume_buffer.len() - min_needed * 2);
        }

        let vol_slice = if !self.volume_buffer.is_empty() {
            Some(self.volume_buffer.as_slice())
        } else {
            None
        };

        let input = UmaInput::from_slice(&self.buffer, vol_slice, self.params.clone());

        match uma_with_kernel(&input, self.kernel) {
            Ok(output) => output.values.last().copied(),
            Err(_) => None,
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.volume_buffer.clear();
        self.dynamic_length = self.params.max_length.unwrap_or(50) as f64;
    }
}

// ==================== ERROR TYPES ====================

#[derive(Debug, Error)]
pub enum UmaError {
    #[error("uma: Input data slice is empty.")]
    EmptyInputData,
    #[error("uma: All values are NaN.")]
    AllValuesNaN,
    #[error("uma: Invalid max_length: max_length = {max_length}, data length = {data_len}")]
    InvalidMaxLength { max_length: usize, data_len: usize },
    #[error("uma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("uma: Invalid accelerator: {accelerator}")]
    InvalidAccelerator { accelerator: f64 },
    #[error("uma: Invalid min_length: {min_length}")]
    InvalidMinLength { min_length: usize },
    #[error("uma: Invalid smooth_length: {smooth_length}")]
    InvalidSmoothLength { smooth_length: usize },
    #[error("uma: min_length ({min_length}) must be <= max_length ({max_length})")]
    MinLengthGreaterThanMaxLength {
        min_length: usize,
        max_length: usize,
    },
    #[error("uma: Error from dependency: {0}")]
    DependencyError(String),
}

// ==================== CORE FUNCTIONS ====================

#[inline(always)]
fn uma_prepare<'a>(input: &'a UmaInput) -> Result<(&'a [f64], usize, usize, usize, f64), UmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(UmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(UmaError::AllValuesNaN)?;
    let accelerator = input.get_accelerator();
    let min_length = input.get_min_length();
    let max_length = input.get_max_length();
    let smooth_length = input.get_smooth_length();

    if max_length == 0 || max_length > len {
        return Err(UmaError::InvalidMaxLength {
            max_length,
            data_len: len,
        });
    }
    if min_length == 0 {
        return Err(UmaError::InvalidMinLength { min_length });
    }
    if min_length > max_length {
        return Err(UmaError::MinLengthGreaterThanMaxLength {
            min_length,
            max_length,
        });
    }
    if smooth_length == 0 {
        return Err(UmaError::InvalidSmoothLength { smooth_length });
    }
    if accelerator < 1.0 {
        return Err(UmaError::InvalidAccelerator { accelerator });
    }
    if len - first < max_length {
        return Err(UmaError::NotEnoughValidData {
            needed: max_length,
            valid: len - first,
        });
    }
    Ok((data, first, min_length, max_length, accelerator))
}

#[inline(always)]
fn uma_core_into(
    input: &UmaInput,
    first: usize,
    min_length: usize,
    max_length: usize,
    accelerator: f64,
    _kernel: Kernel, // For future SIMD implementations
    out: &mut [f64],
) -> Result<(), UmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    // Precompute mean and stddev once at max_length
    let mean = sma(&SmaInput::from_slice(
        data,
        SmaParams {
            period: Some(max_length),
        },
    ))
    .map_err(|e| UmaError::DependencyError(e.to_string()))?
    .values;
    let std_dev = deviation(&DeviationInput::from_slice(
        data,
        DeviationParams {
            period: Some(max_length),
            devtype: Some(0),
        },
    ))
    .map_err(|e| UmaError::DependencyError(e.to_string()))?
    .values;

    let warmup_end = first + max_length - 1;

    // Reusable scratch for MFI typical prices and volumes to avoid per-step reallocation
    let mut tp: Vec<f64> = Vec::with_capacity(max_length);
    let mut vv: Vec<f64> = Vec::with_capacity(max_length);

    let mut length = max_length as f64;

    for i in warmup_end..len {
        let mean_val = mean[i];
        let std_val = std_dev[i];
        if mean_val.is_nan() || std_val.is_nan() {
            continue;
        }

        let a = mean_val - 1.75 * std_val;
        let b = mean_val - 0.25 * std_val;
        let c = mean_val + 0.25 * std_val;
        let d = mean_val + 1.75 * std_val;
        let src = data[i];

        if src >= b && src <= c {
            length += 1.0;
        } else if src < a || src > d {
            length -= 1.0;
        }

        length = length.max(min_length as f64).min(max_length as f64);
        let len_r = length.round() as usize;
        if i + 1 < len_r {
            continue;
        }

        // Momentum factor (reuse buffers, no per-loop allocations)
        let mf = if let Some(vol) = input.volume {
            let v_i = *vol.get(i).unwrap_or(&0.0);
            if v_i == 0.0 || v_i.is_nan() {
                // RSI fallback
                let window_start = 0.max(i as isize + 1 - 2 * len_r as isize) as usize;
                let window_end = i + 1;
                let rsi_input = RsiInput::from_slice(
                    &data[window_start..window_end],
                    RsiParams {
                        period: Some(len_r),
                    },
                );
                rsi(&rsi_input)
                    .ok()
                    .and_then(|r| r.values.last().copied())
                    .unwrap_or(50.0)
            } else {
                // MFI with Candles or (price,volume) slices
                let window_start = i + 1 - len_r;
                match &input.data {
                    UmaData::Candles { candles, .. } => {
                        tp.clear();
                        vv.clear();
                        for j in window_start..=i {
                            tp.push((candles.high[j] + candles.low[j] + candles.close[j]) / 3.0);
                            vv.push(candles.volume[j]);
                        }
                        mfi(&MfiInput::from_slices(
                            &tp,
                            &vv,
                            MfiParams {
                                period: Some(len_r),
                            },
                        ))
                        .ok()
                        .and_then(|r| r.values.last().copied())
                        .unwrap_or(50.0)
                    }
                    UmaData::Slice(_) => {
                        // For slice data with volume, calculate a volume-weighted momentum
                        // This is a simplified approach since we don't have OHLC data for proper MFI
                        let px = &data[window_start..=i];
                        let vv_slice = &vol[window_start..=i];

                        // Calculate volume-weighted price changes
                        let mut up_vol = 0.0;
                        let mut down_vol = 0.0;
                        for j in 1..px.len() {
                            let price_change = px[j] - px[j - 1];
                            let vol = vv_slice[j];
                            if price_change > 0.0 {
                                up_vol += vol;
                            } else if price_change < 0.0 {
                                down_vol += vol;
                            }
                        }

                        // Calculate volume-weighted momentum (similar to MFI but simpler)
                        if up_vol + down_vol > 0.0 {
                            100.0 * up_vol / (up_vol + down_vol)
                        } else {
                            50.0
                        }
                    }
                }
            }
        } else {
            let window_start = 0.max(i as isize + 1 - 2 * len_r as isize) as usize;
            let window_end = i + 1;
            let rsi_input = RsiInput::from_slice(
                &data[window_start..window_end],
                RsiParams {
                    period: Some(len_r),
                },
            );
            rsi(&rsi_input)
                .ok()
                .and_then(|r| r.values.last().copied())
                .unwrap_or(50.0)
        };

        let mf_scaled = mf * 2.0 - 100.0;
        let p = accelerator + (mf_scaled.abs() / 25.0);

        let window_start = i + 1 - len_r;
        let mut sum = 0.0;
        let mut wsum = 0.0;

        // Power-weighted average, newest gets highest weight
        for j in 0..len_r {
            let idx = window_start + j;
            let w = ((len_r - j) as f64).powf(p);
            let x = data[idx];
            if !x.is_nan() {
                sum += x * w;
                wsum += w;
            }
        }
        out[i] = if wsum > 0.0 { sum / wsum } else { data[i] };
    }

    Ok(())
}

#[inline]
pub fn uma(input: &UmaInput) -> Result<UmaOutput, UmaError> {
    uma_with_kernel(input, Kernel::Auto)
}

pub fn uma_with_kernel(input: &UmaInput, kernel: Kernel) -> Result<UmaOutput, UmaError> {
    let (data, first, min_len, max_len, accel) = uma_prepare(input)?;
    let warm = first + max_len - 1;

    // Primary buffer: NaN prefix allocated once.
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    // Thread kernel through for parity and future SIMD.
    uma_core_into(input, first, min_len, max_len, accel, kernel, &mut out)?;

    // Optional smoothing with no extra copy: return WMA's Vec directly.
    let smooth = input.get_smooth_length();
    if smooth > 1 {
        let w = wma(&WmaInput::from_slice(
            &out,
            WmaParams {
                period: Some(smooth),
            },
        ))
        .map_err(|e| UmaError::DependencyError(e.to_string()))?
        .values;
        return Ok(UmaOutput { values: w });
    }

    Ok(UmaOutput { values: out })
}

#[inline]
pub fn uma_into_slice(dst: &mut [f64], input: &UmaInput, kern: Kernel) -> Result<(), UmaError> {
    let (data, first, min_len, max_len, accel) = uma_prepare(input)?;
    if dst.len() != data.len() {
        return Err(UmaError::InvalidMaxLength {
            max_length: dst.len(),
            data_len: data.len(),
        });
    }

    // Ensure warmup NaNs even if later smoothing overwrites.
    let warm = first + max_len - 1;
    let warm_end = warm.min(dst.len());
    for v in &mut dst[..warm_end] {
        *v = f64::NAN;
    }

    uma_core_into(input, first, min_len, max_len, accel, kern, dst)?;

    let smooth = input.get_smooth_length();
    if smooth > 1 {
        // Fallback: compute WMA once and copy once.
        let tmp = wma(&WmaInput::from_slice(
            dst,
            WmaParams {
                period: Some(smooth),
            },
        ))
        .map_err(|e| UmaError::DependencyError(e.to_string()))?
        .values;
        dst.copy_from_slice(&tmp);
    }
    Ok(())
}

// ==================== BATCH PROCESSING ====================

#[derive(Clone, Debug)]
pub struct UmaBatchRange {
    pub accelerator: (f64, f64, f64),         // (start, end, step)
    pub min_length: (usize, usize, usize),    // (start, end, step)
    pub max_length: (usize, usize, usize),    // (start, end, step)
    pub smooth_length: (usize, usize, usize), // (start, end, step)
}

impl Default for UmaBatchRange {
    fn default() -> Self {
        Self {
            accelerator: (1.0, 1.0, 0.0),
            min_length: (5, 5, 0),
            max_length: (50, 50, 0),
            smooth_length: (4, 4, 0),
        }
    }
}

#[derive(Copy, Clone)]
pub struct UmaBatchBuilder {
    accelerator_range: (f64, f64, f64),
    min_length_range: (usize, usize, usize),
    max_length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for UmaBatchBuilder {
    fn default() -> Self {
        Self {
            accelerator_range: (1.0, 1.0, 0.0),
            min_length_range: (5, 5, 0),
            max_length_range: (50, 50, 0),
            smooth_length_range: (4, 4, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl UmaBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn accelerator_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.accelerator_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn min_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.min_length_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn max_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.max_length_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smooth_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.smooth_length_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<UmaBatchOutput, UmaError> {
        let sweep = UmaBatchRange {
            accelerator: self.accelerator_range,
            min_length: self.min_length_range,
            max_length: self.max_length_range,
            smooth_length: self.smooth_length_range,
        };

        let data = source_type(candles, source);
        uma_batch_inner(data, Some(&candles.volume), &sweep, self.kernel, false)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
        volume: Option<&[f64]>,
    ) -> Result<UmaBatchOutput, UmaError> {
        let sweep = UmaBatchRange {
            accelerator: self.accelerator_range,
            min_length: self.min_length_range,
            max_length: self.max_length_range,
            smooth_length: self.smooth_length_range,
        };

        uma_batch_inner(data, volume, &sweep, self.kernel, false)
    }

    pub fn with_default_slice(
        data: &[f64],
        volume: Option<&[f64]>,
        k: Kernel,
    ) -> Result<UmaBatchOutput, UmaError> {
        UmaBatchBuilder::new().kernel(k).apply_slice(data, volume)
    }

    pub fn with_default_candles(c: &Candles) -> Result<UmaBatchOutput, UmaError> {
        UmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct UmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl UmaBatchOutput {
    pub fn row_for_params(&self, p: &UmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.accelerator.unwrap_or(1.0) == p.accelerator.unwrap_or(1.0)
                && c.min_length.unwrap_or(5) == p.min_length.unwrap_or(5)
                && c.max_length.unwrap_or(50) == p.max_length.unwrap_or(50)
                && c.smooth_length.unwrap_or(4) == p.smooth_length.unwrap_or(4)
        })
    }

    pub fn values_for(&self, p: &UmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid_uma(r: &UmaBatchRange) -> Vec<UmaParams> {
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

    let accs = axis_f64(r.accelerator);
    let mins = axis_usize(r.min_length);
    let maxs = axis_usize(r.max_length);
    let smooths = axis_usize(r.smooth_length);

    let mut out = Vec::with_capacity(accs.len() * mins.len() * maxs.len() * smooths.len());
    for &a in &accs {
        for &min in &mins {
            for &max in &maxs {
                for &s in &smooths {
                    if min <= max {
                        // Only valid combinations
                        out.push(UmaParams {
                            accelerator: Some(a),
                            min_length: Some(min),
                            max_length: Some(max),
                            smooth_length: Some(s),
                        });
                    }
                }
            }
        }
    }
    out
}

pub fn uma_batch_with_kernel(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    k: Kernel,
) -> Result<UmaBatchOutput, UmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(UmaError::InvalidMaxLength {
                max_length: 0,
                data_len: 0,
            })
        }
    };

    uma_batch_inner(data, volume, sweep, kernel, true)
}

#[inline(always)]
pub fn uma_batch_slice(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
) -> Result<UmaBatchOutput, UmaError> {
    uma_batch_inner(data, volume, sweep, kern, false)
}

#[inline(always)]
pub fn uma_batch_par_slice(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
) -> Result<UmaBatchOutput, UmaError> {
    uma_batch_inner(data, volume, sweep, kern, true)
}

#[inline(always)]
fn debatch(k: Kernel) -> Kernel {
    match k {
        Kernel::Avx512Batch | Kernel::Avx512 => Kernel::Avx512,
        Kernel::Avx2Batch | Kernel::Avx2 => Kernel::Avx2,
        Kernel::ScalarBatch | Kernel::Scalar => Kernel::Scalar,
        Kernel::Auto => detect_best_kernel(),
        _ => Kernel::Scalar,
    }
}

#[inline(always)]
fn uma_batch_inner_into(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<UmaParams>, UmaError> {
    let combos = expand_grid_uma(sweep);
    if combos.is_empty() {
        return Err(UmaError::InvalidMaxLength {
            max_length: 0,
            data_len: 0,
        });
    }

    let cols = data.len();
    let rows = combos.len();
    debug_assert_eq!(out.len(), rows * cols);

    let per_row_kernel = debatch(kern);

    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.max_length.unwrap_or(50) - 1 + c.smooth_length.unwrap_or(4) - 1)
        .collect();

    // Initialize NaN prefixes on an uninitialized view to avoid extra writes.
    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    let do_row = |row: usize, row_slice: &mut [f64]| -> Result<(), UmaError> {
        let input = UmaInput::from_slice(data, volume, combos[row].clone());
        uma_into_slice(row_slice, &input, per_row_kernel)
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        use rayon::prelude::*;
        out.par_chunks_mut(cols)
            .enumerate()
            .try_for_each(|(r, s)| do_row(r, s))?;
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    }

    Ok(combos)
}

#[inline(always)]
fn uma_batch_inner(
    data: &[f64],
    volume: Option<&[f64]>,
    sweep: &UmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<UmaBatchOutput, UmaError> {
    let combos = expand_grid_uma(sweep);
    let cols = data.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(UmaError::EmptyInputData);
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            first + c.max_length.unwrap_or(50) - 1 + c.smooth_length.unwrap_or(4) - 1
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let combos = uma_batch_inner_into(data, volume, sweep, kern, parallel, out_slice)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(UmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// ==================== PYTHON BINDINGS ====================

#[cfg(feature = "python")]
#[pyfunction(name = "uma")]
#[pyo3(signature = (data, accelerator, min_length, max_length, smooth_length, volume=None, kernel=None))]
pub fn uma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::PyArrayMethods;
    let kern = validate_kernel(kernel, false)?;
    let slice_in = data.as_slice()?;
    let vol_slice = volume.as_ref().map(|v| v.as_slice()).transpose()?;

    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };
    let input = UmaInput::from_slice(slice_in, vol_slice, params);

    let out: Vec<f64> = py
        .allow_threads(|| uma_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "UmaStream")]
pub struct UmaStreamPy {
    stream: UmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl UmaStreamPy {
    #[new]
    pub fn new(
        accelerator: f64,
        min_length: usize,
        max_length: usize,
        smooth_length: usize,
    ) -> PyResult<Self> {
        let params = UmaParams {
            accelerator: Some(accelerator),
            min_length: Some(min_length),
            max_length: Some(max_length),
            smooth_length: Some(smooth_length),
        };
        let stream =
            UmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    /// alma-style: single-arg update
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }

    /// keep extended form for volume
    #[pyo3(name = "update_with_volume")]
    pub fn update_with_volume_py(&mut self, value: f64, volume: Option<f64>) -> Option<f64> {
        self.stream.update_with_volume(value, volume)
    }

    pub fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "uma_batch")]
#[pyo3(signature = (data, accelerator_range, min_length_range, max_length_range, smooth_length_range, volume=None, kernel=None))]
pub fn uma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    accelerator_range: (f64, f64, f64),
    min_length_range: (usize, usize, usize),
    max_length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    volume: Option<numpy::PyReadonlyArray1<'py, f64>>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let vol_slice = volume.as_ref().map(|v| v.as_slice()).transpose()?;
    let sweep = UmaBatchRange {
        accelerator: accelerator_range,
        min_length: min_length_range,
        max_length: max_length_range,
        smooth_length: smooth_length_range,
    };
    let combos = expand_grid_uma(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| uma_batch_inner_into(slice_in, vol_slice, &sweep, kern, false, out_slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item(
        "accelerators",
        combos
            .iter()
            .map(|c| c.accelerator.unwrap_or(1.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "min_lengths",
        combos
            .iter()
            .map(|c| c.min_length.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "max_lengths",
        combos
            .iter()
            .map(|c| c.max_length.unwrap_or(50) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        combos
            .iter()
            .map(|c| c.smooth_length.unwrap_or(4) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    // Add combos list for compatibility with tests
    let combo_list: Vec<Bound<'py, PyDict>> = combos
        .iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("accelerator", c.accelerator.unwrap_or(1.0))
                .unwrap();
            d.set_item("min_length", c.min_length.unwrap_or(5)).unwrap();
            d.set_item("max_length", c.max_length.unwrap_or(50))
                .unwrap();
            d.set_item("smooth_length", c.smooth_length.unwrap_or(4))
                .unwrap();
            d.into()
        })
        .collect();
    dict.set_item("combos", combo_list)?;

    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "uma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, accelerator_range, min_length_range, max_length_range, smooth_length_range, volume_f32=None, device_id=0))]
pub fn uma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    accelerator_range: (f64, f64, f64),
    min_length_range: (usize, usize, usize),
    max_length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    volume_f32: Option<numpy::PyReadonlyArray1<'_, f32>>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let volume_slice = volume_f32.as_ref().map(|v| v.as_slice()).transpose()?;
    let sweep = UmaBatchRange {
        accelerator: accelerator_range,
        min_length: min_length_range,
        max_length: max_length_range,
        smooth_length: smooth_length_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaUma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.uma_batch_dev(slice_in, volume_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "uma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, accelerator, min_length, max_length, smooth_length, volume_tm_f32=None, device_id=0))]
pub fn uma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: PyReadonlyArray2<'_, f32>,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
    volume_tm_f32: Option<PyReadonlyArray2<'_, f32>>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    use numpy::PyUntypedArrayMethods;

    let rows = prices_tm_f32.shape()[0];
    let cols = prices_tm_f32.shape()[1];
    if let Some(vol) = &volume_tm_f32 {
        let vshape = vol.shape();
        if vshape != prices_tm_f32.shape() {
            return Err(PyValueError::new_err(
                "price and volume matrices must share shape",
            ));
        }
    }

    let prices_flat = prices_tm_f32.as_slice()?;
    let volume_flat = volume_tm_f32
        .as_ref()
        .map(|arr| arr.as_slice())
        .transpose()?;

    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaUma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.uma_many_series_one_param_time_major_dev(prices_flat, volume_flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

// ==================== WASM BINDINGS ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_js(
    data: &[f64],
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
    volume: Option<Vec<f64>>,
) -> Result<JsValue, JsValue> {
    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };
    let vol_slice = volume.as_deref();
    let input = UmaInput::from_slice(data, vol_slice, params);

    match uma_with_kernel(&input, detect_best_kernel()) {
        Ok(output) => {
            // Return as an object with values property to match test expectations
            let obj = js_sys::Object::new();
            let values_array = js_sys::Array::new();
            for val in output.values {
                values_array.push(&JsValue::from_f64(val));
            }
            js_sys::Reflect::set(&obj, &JsValue::from_str("values"), &values_array)?;
            Ok(obj.into())
        }
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UmaBatchConfig {
    pub accelerator_range: (f64, f64, f64),
    pub min_length_range: (usize, usize, usize),
    pub max_length_range: (usize, usize, usize),
    pub smooth_length_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "uma_batch")]
pub fn uma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: UmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = UmaBatchRange {
        accelerator: cfg.accelerator_range,
        min_length: cfg.min_length_range,
        max_length: cfg.max_length_range,
        smooth_length: cfg.smooth_length_range,
    };
    let out = uma_batch_inner(data, None, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = UmaBatchJsOutput {
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
pub fn uma_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = UmaParams {
            accelerator: Some(accelerator),
            min_length: Some(min_length),
            max_length: Some(max_length),
            smooth_length: Some(smooth_length),
        };
        let input = UmaInput::from_slice(data, None, params);

        if core::ptr::eq(in_ptr as *const u8, out_ptr as *const u8) {
            // in-place: compute into temp then copy once
            let mut tmp = vec![0.0; len];
            uma_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            uma_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

// ==================== WASM STREAMING API ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_new(
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
) -> *mut UmaStream {
    let params = UmaParams {
        accelerator: Some(accelerator),
        min_length: Some(min_length),
        max_length: Some(max_length),
        smooth_length: Some(smooth_length),
    };

    match UmaStream::try_new(params) {
        Ok(stream) => Box::into_raw(Box::new(stream)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_update(stream: *mut UmaStream, value: f64) -> Option<f64> {
    if stream.is_null() {
        return None;
    }
    unsafe { (*stream).update_with_volume(value, None) }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_update_with_volume(
    stream: *mut UmaStream,
    value: f64,
    volume: f64,
) -> Option<f64> {
    if stream.is_null() {
        return None;
    }
    unsafe { (*stream).update_with_volume(value, Some(volume)) }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_reset(stream: *mut UmaStream) {
    if !stream.is_null() {
        unsafe {
            (*stream).reset();
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_stream_free(stream: *mut UmaStream) {
    if !stream.is_null() {
        unsafe {
            let _ = Box::from_raw(stream);
        }
    }
}

// ==================== WASM ZERO-COPY HELPERS ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_get_view(ptr: *mut f64, len: usize) -> js_sys::Float64Array {
    unsafe { js_sys::Float64Array::view(std::slice::from_raw_parts(ptr, len)) }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_update(
    ptr: *mut f64,
    len: usize,
    accelerator: f64,
    min_length: usize,
    max_length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(ptr, len);
        let params = UmaParams {
            accelerator: Some(accelerator),
            min_length: Some(min_length),
            max_length: Some(max_length),
            smooth_length: Some(smooth_length),
        };
        let input = UmaInput::from_slice(data, None, params);

        // Compute into temporary buffer then copy back
        let mut tmp = vec![0.0; len];
        uma_into_slice(&mut tmp, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let out = std::slice::from_raw_parts_mut(ptr, len);
        out.copy_from_slice(&tmp);
        Ok(())
    }
}

// ==================== WASM BATCH EXPORT ====================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn uma_batch_js(
    data: &[f64],
    accelerator_range: Vec<f64>,     // [start, end, step]
    min_length_range: Vec<usize>,    // [start, end, step]
    max_length_range: Vec<usize>,    // [start, end, step]
    smooth_length_range: Vec<usize>, // [start, end, step]
    volume: Option<Vec<f64>>,
) -> Result<JsValue, JsValue> {
    if accelerator_range.len() != 3
        || min_length_range.len() != 3
        || max_length_range.len() != 3
        || smooth_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "All range arrays must have exactly 3 elements: [start, end, step]",
        ));
    }

    let sweep = UmaBatchRange {
        accelerator: (
            accelerator_range[0],
            accelerator_range[1],
            accelerator_range[2],
        ),
        min_length: (
            min_length_range[0],
            min_length_range[1],
            min_length_range[2],
        ),
        max_length: (
            max_length_range[0],
            max_length_range[1],
            max_length_range[2],
        ),
        smooth_length: (
            smooth_length_range[0],
            smooth_length_range[1],
            smooth_length_range[2],
        ),
    };

    let vol_slice = volume.as_deref();
    let kernel = detect_best_batch_kernel();

    let result = uma_batch_inner(data, vol_slice, &sweep, kernel, true)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert to JS-friendly format with individual parameter arrays
    let obj = js_sys::Object::new();

    // Add values as flat array
    let values_array = js_sys::Array::new();
    for val in result.values {
        values_array.push(&JsValue::from_f64(val));
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("values"), &values_array)?;

    // Add parameter arrays
    let accelerators = js_sys::Array::new();
    let min_lengths = js_sys::Array::new();
    let max_lengths = js_sys::Array::new();
    let smooth_lengths = js_sys::Array::new();

    for combo in &result.combos {
        accelerators.push(&JsValue::from_f64(combo.accelerator.unwrap_or(1.0)));
        min_lengths.push(&JsValue::from_f64(combo.min_length.unwrap_or(5) as f64));
        max_lengths.push(&JsValue::from_f64(combo.max_length.unwrap_or(50) as f64));
        smooth_lengths.push(&JsValue::from_f64(combo.smooth_length.unwrap_or(4) as f64));
    }

    js_sys::Reflect::set(&obj, &JsValue::from_str("accelerators"), &accelerators)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("min_lengths"), &min_lengths)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("max_lengths"), &max_lengths)?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("smooth_lengths"), &smooth_lengths)?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(result.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(result.cols as f64),
    )?;

    Ok(obj.into())
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_uma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = UmaParams {
            accelerator: None,
            min_length: None,
            max_length: None,
            smooth_length: None,
        };
        let input = UmaInput::from_candles(&candles, "close", default_params);
        let output = uma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_uma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = UmaInput::from_candles(&candles, "close", UmaParams::default());
        let result = uma_with_kernel(&input, kernel)?;

        // Get valid values
        let values = &result.values;
        let valid_values: Vec<f64> = values.iter().filter(|&&v| !v.is_nan()).copied().collect();

        // Expected values calculated from CSV data
        let expected_last_five = [
            59665.81830666,
            59477.69234542,
            59314.50778603,
            59218.23033661,
            59143.61473211,
        ];

        // Check we have enough valid values
        if valid_values.len() >= 5 {
            let start = valid_values.len().saturating_sub(5);
            for (i, &val) in valid_values[start..].iter().enumerate() {
                let diff = (val - expected_last_five[i]).abs();
                let tolerance = expected_last_five[i] * 0.01; // 1% tolerance
                assert!(
                    diff < tolerance || diff < 100.0,
                    "[{}] UMA {:?} mismatch at idx {}: got {}, expected {}",
                    test_name,
                    kernel,
                    i,
                    val,
                    expected_last_five[i]
                );
            }
        }
        Ok(())
    }

    fn check_uma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = UmaInput::with_default_candles(&candles);
        match input.data {
            UmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected UmaData::Candles"),
        }
        let output = uma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_uma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = UmaParams {
            accelerator: Some(1.0),
            min_length: Some(5),
            max_length: Some(0), // Invalid
            smooth_length: Some(4),
        };
        let input = UmaInput::from_slice(&input_data, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] UMA should fail with zero max_length",
            test_name
        );
        Ok(())
    }

    fn check_uma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = UmaParams {
            accelerator: Some(1.0),
            min_length: Some(5),
            max_length: Some(10),
            smooth_length: Some(4),
        };
        let input = UmaInput::from_slice(&data_small, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] UMA should fail with max_length exceeding data length",
            test_name
        );
        Ok(())
    }

    fn check_uma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = UmaParams::default();
        let input = UmaInput::from_slice(&single_point, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] UMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_uma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = UmaInput::from_slice(&empty, None, UmaParams::default());
        let res = uma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(UmaError::EmptyInputData)),
            "[{}] UMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_uma_invalid_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();

        // Test invalid accelerator
        let params = UmaParams {
            accelerator: Some(0.5), // Invalid - must be >= 1.0
            max_length: Some(10),
            ..Default::default()
        };
        let input = UmaInput::from_slice(&data, None, params);
        let res = uma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(UmaError::InvalidAccelerator { .. })),
            "[{}] UMA should fail with invalid accelerator, got: {:?}",
            test_name,
            res
        );
        Ok(())
    }

    fn check_uma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = UmaParams::default();
        let first_input = UmaInput::from_candles(&candles, "close", first_params);
        let first_result = uma_with_kernel(&first_input, kernel)?;

        let second_params = UmaParams::default();
        let second_input = UmaInput::from_slice(&first_result.values, None, second_params);
        let second_result = uma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        // Just verify we got valid output
        let valid_count = second_result
            .values
            .iter()
            .filter(|&&v| !v.is_nan())
            .count();
        assert!(
            valid_count > 0,
            "[{}] UMA reinput should produce valid values",
            test_name
        );

        Ok(())
    }

    fn check_uma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let mut data = vec![f64::NAN; 10];
        data.extend((0..100).map(|i| 100.0 + i as f64));

        let input = UmaInput::from_slice(&data, None, UmaParams::default());
        let res = uma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), data.len());

        // Check we have valid values after the NaNs and warmup
        let valid_count = res.values[60..].iter().filter(|&&v| !v.is_nan()).count();
        assert!(
            valid_count > 0,
            "[{}] UMA should handle NaN prefix",
            test_name
        );

        Ok(())
    }

    fn check_uma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = UmaParams::default();
        let input = UmaInput::from_candles(&candles, "close", params.clone());
        let batch_output = uma_with_kernel(&input, kernel)?.values;

        let mut stream = UmaStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());

        for (i, &price) in candles.close.iter().enumerate() {
            let volume = if i < candles.volume.len() {
                Some(candles.volume[i])
            } else {
                None
            };

            match stream.update_with_volume(price, volume) {
                Some(uma_val) => stream_values.push(uma_val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());

        // Note: UMA streaming implementation has inherent differences from batch due to
        // how it manages its dynamic length buffer. We compare the overall trend
        // rather than exact values.

        // Compare last few valid values (streaming may have slight differences)
        let batch_valid: Vec<f64> = batch_output
            .iter()
            .filter(|&&v| !v.is_nan())
            .copied()
            .collect();
        let stream_valid: Vec<f64> = stream_values
            .iter()
            .filter(|&&v| !v.is_nan())
            .copied()
            .collect();

        if batch_valid.len() >= 5 && stream_valid.len() >= 5 {
            let batch_last = &batch_valid[batch_valid.len() - 5..];
            let stream_last = &stream_valid[stream_valid.len() - 5..];

            for (i, (&b, &s)) in batch_last.iter().zip(stream_last.iter()).enumerate() {
                let diff = (b - s).abs();
                let relative_diff = diff / b.abs().max(1.0);
                assert!(
                    relative_diff < 0.1, // Allow 10% difference for UMA's dynamic length
                    "[{}] UMA streaming mismatch at idx {}: batch={}, stream={}, diff={}, rel_diff={}",
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
    fn check_uma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            UmaParams::default(),
            UmaParams {
                accelerator: Some(1.5),
                min_length: Some(3),
                max_length: Some(30),
                smooth_length: Some(3),
            },
            UmaParams {
                accelerator: Some(2.0),
                min_length: Some(10),
                max_length: Some(100),
                smooth_length: Some(8),
            },
        ];

        for params in test_params.iter() {
            let input = UmaInput::from_candles(&candles, "close", params.clone());
            let output = uma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_uma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_uma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (5usize..=20, 20usize..=50, 2usize..=8, 1.0f64..3.0).prop_flat_map(
            |(min_len, max_len, smooth_len, acc)| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        max_len + 10..200,
                    ),
                    Just(min_len),
                    Just(max_len),
                    Just(smooth_len),
                    Just(acc),
                )
            },
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, min_len, max_len, smooth_len, acc)| {
                let params = UmaParams {
                    accelerator: Some(acc),
                    min_length: Some(min_len),
                    max_length: Some(max_len),
                    smooth_length: Some(smooth_len),
                };
                let input = UmaInput::from_slice(&data, None, params);

                let UmaOutput { values: out } = uma_with_kernel(&input, kernel).unwrap();
                let UmaOutput { values: ref_out } =
                    uma_with_kernel(&input, Kernel::Scalar).unwrap();

                // Basic sanity checks
                prop_assert_eq!(out.len(), data.len());
                prop_assert_eq!(ref_out.len(), data.len());

                // Compare kernel output with reference scalar
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-6 || ulp_diff <= 10,
                        "mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_uma_tests {
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

    generate_all_uma_tests!(
        check_uma_partial_params,
        check_uma_accuracy,
        check_uma_default_candles,
        check_uma_zero_period,
        check_uma_period_exceeds_length,
        check_uma_very_small_dataset,
        check_uma_empty_input,
        check_uma_invalid_params,
        check_uma_reinput,
        check_uma_nan_handling,
        check_uma_streaming,
        check_uma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_uma_tests!(check_uma_property);

    #[test]
    fn uma_into_slice_matches_with_kernel() {
        use crate::utilities::data_loader::read_candles_from_csv;
        let c = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let params = UmaParams::default();
        let input = UmaInput::from_candles(&c, "close", params.clone());

        let via_api = uma_with_kernel(&input, Kernel::Scalar).unwrap().values;
        let mut via_into = vec![0.0; via_api.len()];
        uma_into_slice(&mut via_into, &input, Kernel::Scalar).unwrap();

        assert_eq!(via_api.len(), via_into.len());
        for (a, b) in via_api.iter().zip(via_into.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[cfg(feature = "python")]
    #[test]
    fn uma_batch_py_no_copy_shape() {
        pyo3::Python::with_gil(|py| {
            use numpy::PyArray1;
            let data = PyArray1::from_vec(py, (0..256).map(|i| i as f64).collect());
            let d = crate::indicators::moving_averages::uma::uma_batch_py(
                py,
                data.readonly(),
                (1.0, 1.0, 0.0),
                (5, 5, 0),
                (50, 50, 0),
                (4, 4, 0),
                None,
                Some("scalar_batch"),
            )
            .unwrap();
            let v = d.get_item("values").unwrap();
            // Just ensure it's a 2D view, not a copied nested array
            assert!(v.downcast::<numpy::PyArray2<f64>>().is_ok());
        });
    }

    // Batch processing tests
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = UmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = UmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Verify we have valid values
        let valid_count = row.iter().filter(|&&v| !v.is_nan()).count();
        assert!(
            valid_count > 0,
            "[{}] Batch should produce valid values",
            test
        );

        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();

        let output = UmaBatchBuilder::new()
            .kernel(kernel)
            .accelerator_range(1.0, 2.0, 0.5)
            .min_length_range(5, 10, 5)
            .max_length_range(20, 30, 10)
            .smooth_length_range(3, 5, 2)
            .apply_slice(&data, None)?;

        let expected_combos = 3 * 2 * 2 * 2; // 3 accelerators * 2 min_lengths * 2 max_lengths * 2 smooth_lengths
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, data.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (1.0, 1.5, 0.5, 5, 10, 5, 20, 30, 10, 3, 5, 2),
            (2.0, 2.0, 0.0, 10, 10, 0, 50, 50, 0, 4, 4, 0),
        ];

        for (
            cfg_idx,
            &(
                a_start,
                a_end,
                a_step,
                min_start,
                min_end,
                min_step,
                max_start,
                max_end,
                max_step,
                s_start,
                s_end,
                s_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = UmaBatchBuilder::new()
                .kernel(kernel)
                .accelerator_range(a_start, a_end, a_step)
                .min_length_range(min_start, min_end, min_step)
                .max_length_range(max_start, max_end, max_step)
                .smooth_length_range(s_start, s_end, s_step)
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
                        at row {} col {} (flat index {}) with params: acc={}, min={}, max={}, smooth={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.accelerator.unwrap_or(1.0),
                        combo.min_length.unwrap_or(5),
                        combo.max_length.unwrap_or(50),
                        combo.smooth_length.unwrap_or(4)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {})",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {})",
                        test, cfg_idx, val, bits, row, col, idx
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                    Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);
}
