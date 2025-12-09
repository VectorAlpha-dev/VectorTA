//! # WaveTrend Indicator
//!
//! A momentum oscillator that identifies overbought and oversold conditions using a combination
//! of EMAs, absolute deviations, and SMAs to produce smooth wave-like signals.
//!
//! ## Parameters
//! - **channel_length**: EMA period for initial smoothing (default: 9)
//! - **average_length**: EMA period for transformed channel index (default: 12)
//! - **ma_length**: SMA period for final smoothing of WT1 (default: 3)
//! - **factor**: Scaling factor for normalization (default: 0.015)
//!
//! ## Inputs
//! - Data series as slice or candles with source
//!
//! ## Returns
//! - **wt1**: Primary WaveTrend line as `Vec<f64>` (length matches input)
//! - **wt2**: Secondary WaveTrend line (SMA of WT1) as `Vec<f64>`
//! - **wt_diff**: Difference (WT2 - WT1) as `Vec<f64>`
//!
//! ## Developer Notes
//! - **SIMD status (single-series)**: SIMD underperforms scalar due to sequential EMA dependencies. Runtime selection short-circuits AVX2/AVX512 to the scalar fast path to preserve best performance. SIMD code remains for future evolution.
//! - **SIMD status (batch)**: SIMD-across-rows currently disabled by default (short-circuited to scalar batch) because it underperforms for typical grids; revisit with shared-stage execution if needed.
//! - **CUDA status**: FP32 CUDA kernels are enabled for one-series × many-params and many-series × one-param. VRAM handles are exposed to Python with CUDA Array Interface v3 and DLPack v1.x; context lifetime is managed via primary-context RAII and wrapper-owned context guards so frees occur on a valid device context.
//! - **Streaming update**: O(1) and mirrors batch semantics by inserting one value per bar into the WT2 SMA window (finite or NaN) and emitting only when the last `ma_length` positions are all finite.
//! - **Memory optimization**: Properly uses zero-copy helper functions (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)
//! - **Note**: Streaming implementation maintains separate state for each calculation stage
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::wavetrend::CudaWavetrend;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
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

use crate::indicators::moving_averages::ema::{ema, EmaError, EmaInput, EmaParams};
use crate::indicators::moving_averages::sma::{sma, SmaError, SmaInput, SmaParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for WavetrendInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WavetrendData::Slice(slice) => slice,
            WavetrendData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WavetrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WavetrendOutput {
    pub wt1: Vec<f64>,
    pub wt2: Vec<f64>,
    pub wt_diff: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WavetrendParams {
    pub channel_length: Option<usize>,
    pub average_length: Option<usize>,
    pub ma_length: Option<usize>,
    pub factor: Option<f64>,
}

impl Default for WavetrendParams {
    fn default() -> Self {
        Self {
            channel_length: Some(9),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WavetrendInput<'a> {
    pub data: WavetrendData<'a>,
    pub params: WavetrendParams,
}

impl<'a> WavetrendInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: WavetrendParams) -> Self {
        Self {
            data: WavetrendData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: WavetrendParams) -> Self {
        Self {
            data: WavetrendData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hlc3", WavetrendParams::default())
    }
    #[inline]
    pub fn get_channel_length(&self) -> usize {
        self.params.channel_length.unwrap_or(9)
    }
    #[inline]
    pub fn get_average_length(&self) -> usize {
        self.params.average_length.unwrap_or(12)
    }
    #[inline]
    pub fn get_ma_length(&self) -> usize {
        self.params.ma_length.unwrap_or(3)
    }
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(0.015)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WavetrendBuilder {
    channel_length: Option<usize>,
    average_length: Option<usize>,
    ma_length: Option<usize>,
    factor: Option<f64>,
    kernel: Kernel,
}

impl Default for WavetrendBuilder {
    fn default() -> Self {
        Self {
            channel_length: None,
            average_length: None,
            ma_length: None,
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WavetrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn channel_length(mut self, n: usize) -> Self {
        self.channel_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn average_length(mut self, n: usize) -> Self {
        self.average_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn ma_length(mut self, n: usize) -> Self {
        self.ma_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn factor(mut self, f: f64) -> Self {
        self.factor = Some(f);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<WavetrendOutput, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        let i = WavetrendInput::from_candles(c, "hlc3", p);
        wavetrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<WavetrendOutput, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        let i = WavetrendInput::from_slice(d, p);
        wavetrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<WavetrendStream, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        WavetrendStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum WavetrendError {
    #[error("wavetrend: Empty data provided.")]
    EmptyInputData,
    #[error("wavetrend: Empty data provided.")]
    EmptyData,
    #[error("wavetrend: All values are NaN.")]
    AllValuesNaN,
    #[error("wavetrend: Invalid channel_length = {channel_length}, data length = {data_len}")]
    InvalidChannelLen {
        channel_length: usize,
        data_len: usize,
    },
    #[error("wavetrend: Invalid average_length = {average_length}, data length = {data_len}")]
    InvalidAverageLen {
        average_length: usize,
        data_len: usize,
    },
    #[error("wavetrend: Invalid ma_length = {ma_length}, data length = {data_len}")]
    InvalidMaLen { ma_length: usize, data_len: usize },
    #[error("wavetrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("wavetrend: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("wavetrend: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputSliceLengthMismatch { expected: usize, got: usize },
    #[error("wavetrend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },
    #[error("wavetrend: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("wavetrend: EMA error {0}")]
    EmaError(#[from] EmaError),
    #[error("wavetrend: SMA error {0}")]
    SmaError(#[from] SmaError),
}

#[inline]
pub fn wavetrend(input: &WavetrendInput) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_with_kernel(input, Kernel::Auto)
}

pub fn wavetrend_with_kernel(
    input: &WavetrendInput,
    kernel: Kernel,
) -> Result<WavetrendOutput, WavetrendError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(WavetrendError::EmptyInputData);
    }
    let channel_len = input.get_channel_length();
    let average_len = input.get_average_length();
    let ma_len = input.get_ma_length();
    let factor = input.get_factor();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;
    let needed = *[channel_len, average_len, ma_len].iter().max().unwrap();
    let valid = data.len() - first;

    if channel_len == 0 || channel_len > data.len() {
        return Err(WavetrendError::InvalidChannelLen {
            channel_length: channel_len,
            data_len: data.len(),
        });
    }
    if average_len == 0 || average_len > data.len() {
        return Err(WavetrendError::InvalidAverageLen {
            average_length: average_len,
            data_len: data.len(),
        });
    }
    if ma_len == 0 || ma_len > data.len() {
        return Err(WavetrendError::InvalidMaLen {
            ma_length: ma_len,
            data_len: data.len(),
        });
    }
    if valid < needed {
        return Err(WavetrendError::NotEnoughValidData { needed, valid });
    }

    // Auto: select best available kernel (AVX512 → AVX2 → Scalar)
    // Note: Single-series AVX2/AVX512 underperform scalar for this indicator due to
    // sequential EMA dependencies. We short-circuit SIMD to Scalar to preserve best
    // performance while keeping SIMD code present for future evolution.
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        // For single-series, SIMD underperforms; force scalar regardless of build features
        Kernel::Avx2 | Kernel::Avx512 => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                wavetrend_scalar(data, channel_len, average_len, ma_len, factor, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                wavetrend_avx2(data, channel_len, average_len, ma_len, factor, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wavetrend_avx512(data, channel_len, average_len, ma_len, factor, first)
            }
            _ => unreachable!(),
        }
    }
}

fn wavetrend_kernel_dispatch(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
    kernel: Kernel,
) -> Result<WavetrendOutput, WavetrendError> {
    let warmup_period = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    let mut wt1_final = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut wt2_final = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut diff_final = alloc_with_nan_prefix(data.len(), warmup_period);

    wavetrend_compute_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
        &mut wt1_final,
        &mut wt2_final,
        &mut diff_final,
        kernel,
    )?;

    Ok(WavetrendOutput {
        wt1: wt1_final,
        wt2: wt2_final,
        wt_diff: diff_final,
    })
}

pub fn wavetrend_scalar(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_kernel_dispatch(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        Kernel::Scalar,
    )
}

use std::collections::VecDeque;
// AVX2 stub - points to scalar
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx2(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    // Fused single-pass AVX2 variant (uses FMA via mul_add); maintains scalar semantics.
    let warmup_period = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    let mut wt1_out = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut wt2_out = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut diff_out = alloc_with_nan_prefix(data.len(), warmup_period);

    wavetrend_fused_avx2_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
        &mut wt1_out,
        &mut wt2_out,
        &mut diff_out,
    );

    Ok(WavetrendOutput {
        wt1: wt1_out,
        wt2: wt2_out,
        wt_diff: diff_out,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn wavetrend_fused_avx2_into(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
    warmup_period: usize,
    dst_wt1: &mut [f64],
    dst_wt2: &mut [f64],
    dst_wt_diff: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    // EMA coefficients
    let alpha_ch = 2.0 / (channel_len as f64 + 1.0);
    let beta_ch = 1.0 - alpha_ch;
    let alpha_avg = 2.0 / (average_len as f64 + 1.0);
    let beta_avg = 1.0 - alpha_avg;

    // EMA states
    let mut esa_state: f64 = f64::NAN;
    let mut de_state: f64 = f64::NAN;
    let mut wt1_state: f64 = f64::NAN;
    let mut esa_seeded = false;
    let mut de_seeded = false;
    let mut wt1_seeded = false;

    // WT2 ring buffer (window over positions)
    let mut ring_vals = vec![f64::NAN; ma_len];
    let mut ring_mask = vec![0u8; ma_len];
    let mut head = 0usize;
    let mut sma_sum = 0.0f64;
    let mut sma_count = 0usize;
    let inv_ma = 1.0 / (ma_len as f64);

    for idx in first..n {
        let x = data[idx];
        let mut wt1_i = f64::NAN;
        let mut wt2_i = f64::NAN;

        if x.is_finite() {
            // ESA = alpha_ch*x + beta_ch*esa_prev  (use mul_add for FMA)
            if !esa_seeded {
                esa_state = x;
                esa_seeded = true;
            } else {
                esa_state = x.mul_add(alpha_ch, beta_ch * esa_state);
            }

            let abs_diff = (x - esa_state).abs();
            if !de_seeded {
                de_state = abs_diff;
                de_seeded = true;
            } else {
                de_state = abs_diff.mul_add(alpha_ch, beta_ch * de_state);
            }

            let den = factor * de_state;
            if den != 0.0 && den.is_finite() && esa_state.is_finite() {
                let ci = (x - esa_state) / den;
                if ci.is_finite() {
                    if !wt1_seeded {
                        wt1_state = ci;
                        wt1_seeded = true;
                    } else {
                        wt1_state = ci.mul_add(alpha_avg, beta_avg * wt1_state);
                    }
                    wt1_i = wt1_state;
                }
            }
        }

        // ma_len > 0 is guaranteed by validated params
        if ring_mask[head] != 0 {
            sma_sum -= ring_vals[head];
            sma_count -= 1;
        }
        if wt1_i.is_finite() {
            ring_vals[head] = wt1_i;
            ring_mask[head] = 1;
            sma_sum += wt1_i;
            sma_count += 1;
        } else {
            ring_vals[head] = f64::NAN;
            ring_mask[head] = 0;
        }
        head += 1;
        if head == ma_len {
            head = 0;
        }
        if sma_count == ma_len {
            wt2_i = sma_sum * inv_ma;
        }

        if idx >= warmup_period {
            dst_wt1[idx] = wt1_i;
            dst_wt2[idx] = wt2_i;
            dst_wt_diff[idx] = if wt1_i.is_finite() && wt2_i.is_finite() {
                wt2_i - wt1_i
            } else {
                f64::NAN
            };
        }
    }
}

// AVX512 stub logic for short/long
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    if channel_len <= 32 {
        wavetrend_avx512_short(data, channel_len, average_len, ma_len, factor, first)
    } else {
        wavetrend_avx512_long(data, channel_len, average_len, ma_len, factor, first)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512_short(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_kernel_dispatch(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        Kernel::Avx512,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512_long(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_kernel_dispatch(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        Kernel::Avx512,
    )
}

#[inline(always)]
fn wavetrend_prepare<'a>(
    input: &'a WavetrendInput,
) -> Result<(&'a [f64], usize, usize, usize, f64, usize, usize), WavetrendError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(WavetrendError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;
    let channel_len = input.get_channel_length();
    let average_len = input.get_average_length();
    let ma_len = input.get_ma_length();
    let factor = input.get_factor();

    // Validate parameters
    if channel_len == 0 || channel_len > data.len() {
        return Err(WavetrendError::InvalidChannelLen {
            channel_length: channel_len,
            data_len: data.len(),
        });
    }
    if average_len == 0 || average_len > data.len() {
        return Err(WavetrendError::InvalidAverageLen {
            average_length: average_len,
            data_len: data.len(),
        });
    }
    if ma_len == 0 || ma_len > data.len() {
        return Err(WavetrendError::InvalidMaLen {
            ma_length: ma_len,
            data_len: data.len(),
        });
    }

    let max_period = channel_len.max(average_len).max(ma_len);
    if data.len() - first < max_period {
        return Err(WavetrendError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }

    // Calculate warmup period
    let warmup_period = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    Ok((
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
    ))
}

#[inline(always)]
fn wavetrend_compute_into(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
    warmup_period: usize,
    dst_wt1: &mut [f64],
    dst_wt2: &mut [f64],
    dst_wt_diff: &mut [f64],
    kernel: Kernel,
) -> Result<(), WavetrendError> {
    // Fast path: single-pass, loop‑jammed scalar kernel.
    // Caller guarantees NaN prefix is already initialized in dst_* slices.
    if matches!(kernel.to_non_batch(), Kernel::Scalar) {
        let n = data.len();
        if n == 0 {
            return Ok(());
        }

        // Precompute EMA coefficients
        let alpha_ch = 2.0 / (channel_len as f64 + 1.0);
        let beta_ch = 1.0 - alpha_ch;
        let alpha_avg = 2.0 / (average_len as f64 + 1.0);
        let beta_avg = 1.0 - alpha_avg;

        // EMA states (seed lazily on first finite value)
        let mut esa_state: f64 = f64::NAN;
        let mut de_state: f64 = f64::NAN;
        let mut wt1_state: f64 = f64::NAN;
        let mut esa_seeded = false;
        let mut de_seeded = false;
        let mut wt1_seeded = false;

        // WT2 = SMA(ma_len) over WT1, implemented as a sliding window over positions.
        // Track finite-membership explicitly to mirror sma_compute_into semantics and
        // avoid repeated is_finite checks on the leaving element.
        let mut ring_vals = vec![f64::NAN; ma_len];
        let mut ring_mask = vec![0u8; ma_len]; // 1 = finite member present at slot, 0 = NaN
        let mut head = 0usize;
        let mut sma_sum = 0.0f64;
        let mut sma_count = 0usize; // number of finite WT1s in the last ma_len positions

        // Process from the first finite input onward; indices < first are already NaN in dst_*.
        for idx in first..n {
            let x = data[idx];

            // Defaults (NaN during warmup or invalid stages)
            let mut wt1_i = f64::NAN;
            let mut wt2_i = f64::NAN;

            // ─────────── ESA and DE EMAs ───────────
            if x.is_finite() {
                // ESA = EMA(channel_len) on price
                if !esa_seeded {
                    esa_state = x;
                    esa_seeded = true;
                } else {
                    esa_state = alpha_ch * x + beta_ch * esa_state;
                }

                // DE = EMA(channel_len) on |price − ESA|
                let abs_diff = (x - esa_state).abs();
                if !de_seeded {
                    de_state = abs_diff;
                    de_seeded = true;
                } else {
                    de_state = alpha_ch * abs_diff + beta_ch * de_state;
                }

                // ─────────── CI and WT1 EMA(average_len) ───────────
                let den = factor * de_state;
                if den != 0.0 && den.is_finite() && esa_state.is_finite() {
                    let ci = (x - esa_state) / den;
                    if ci.is_finite() {
                        if !wt1_seeded {
                            wt1_state = ci;
                            wt1_seeded = true;
                        } else {
                            wt1_state = alpha_avg * ci + beta_avg * wt1_state;
                        }
                        wt1_i = wt1_state;
                    }
                }
            }
            // Else: input NaN → produce NaN this step and DO NOT advance EMA states (skip), matching ema_compute_into behavior.

            // ─────────── WT2 = SMA(ma_len) over WT1 (window over positions, not over finite samples) ───────────
            if ma_len > 0 {
                // Remove the value leaving the window (slot `head`)
                if ring_mask[head] != 0 {
                    sma_sum -= ring_vals[head];
                    sma_count -= 1;
                }
                // Insert current WT1 (finite or NaN) into the window
                if wt1_i.is_finite() {
                    ring_vals[head] = wt1_i;
                    ring_mask[head] = 1;
                    sma_sum += wt1_i;
                    sma_count += 1;
                } else {
                    ring_vals[head] = f64::NAN;
                    ring_mask[head] = 0;
                }
                head += 1;
                if head == ma_len {
                    head = 0;
                }

                // Emit WT2 only when the last `ma_len` positions are all finite (count == ma_len),
                // exactly mirroring sma_compute_into's behavior.
                if sma_count == ma_len {
                    wt2_i = sma_sum / (ma_len as f64);
                }
            }

            // ─────────── Commit outputs only after warmup (prefix already NaN) ───────────
            if idx >= warmup_period {
                dst_wt1[idx] = wt1_i;
                dst_wt2[idx] = wt2_i;
                dst_wt_diff[idx] = if wt1_i.is_finite() && wt2_i.is_finite() {
                    wt2_i - wt1_i
                } else {
                    f64::NAN
                };
            }
        }

        return Ok(());
    }

    // Note: Caller is responsible for NaN prefix initialization
    // This avoids double work when using alloc_with_nan_prefix or init_matrix_prefixes

    let data_valid = &data[first..];
    let simd_kernel = kernel.to_non_batch();

    // We need working space for intermediate calculations
    // Use stack allocation for small periods, heap for large
    if data_valid.len() <= STACK_LIMIT {
        // Stack allocation for small data
        let mut esa_buf = [0.0f64; STACK_LIMIT];
        let mut de_buf = [0.0f64; STACK_LIMIT];
        let mut ci_buf = [0.0f64; STACK_LIMIT];
        let mut wt1_buf = [0.0f64; STACK_LIMIT];
        let mut wt2_buf = [0.0f64; STACK_LIMIT];

        let esa = &mut esa_buf[..data_valid.len()];
        let de = &mut de_buf[..data_valid.len()];
        let ci = &mut ci_buf[..data_valid.len()];
        let wt1 = &mut wt1_buf[..data_valid.len()];
        let wt2 = &mut wt2_buf[..data_valid.len()];

        wavetrend_core_computation(
            data_valid,
            channel_len,
            average_len,
            ma_len,
            factor,
            esa,
            de,
            ci,
            wt1,
            wt2,
            simd_kernel,
        )?;

        // Copy results to output starting from warmup_period
        for i in 0..data_valid.len() {
            let out_idx = i + first;
            if out_idx >= warmup_period {
                dst_wt1[out_idx] = wt1[i];
                dst_wt2[out_idx] = wt2[i];
                if !wt1[i].is_nan() && !wt2[i].is_nan() {
                    dst_wt_diff[out_idx] = wt2[i] - wt1[i];
                } else {
                    dst_wt_diff[out_idx] = f64::NAN;
                }
            }
        }
    } else {
        // Heap allocation for large data
        let mut esa = vec![0.0; data_valid.len()];
        let mut de = vec![0.0; data_valid.len()];
        let mut ci = vec![0.0; data_valid.len()];
        let mut wt1 = vec![0.0; data_valid.len()];
        let mut wt2 = vec![0.0; data_valid.len()];

        wavetrend_core_computation(
            data_valid,
            channel_len,
            average_len,
            ma_len,
            factor,
            &mut esa,
            &mut de,
            &mut ci,
            &mut wt1,
            &mut wt2,
            simd_kernel,
        )?;

        // Copy results to output starting from warmup_period
        for i in 0..data_valid.len() {
            let out_idx = i + first;
            if out_idx >= warmup_period {
                dst_wt1[out_idx] = wt1[i];
                dst_wt2[out_idx] = wt2[i];
                if !wt1[i].is_nan() && !wt2[i].is_nan() {
                    dst_wt_diff[out_idx] = wt2[i] - wt1[i];
                } else {
                    dst_wt_diff[out_idx] = f64::NAN;
                }
            }
        }
    }

    Ok(())
}

// Stack allocation limit for intermediate buffers
const STACK_LIMIT: usize = 512;

#[inline(always)]
fn wavetrend_core_computation(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    esa: &mut [f64],
    de: &mut [f64],
    ci: &mut [f64],
    wt1: &mut [f64],
    wt2: &mut [f64],
    kernel: Kernel,
) -> Result<(), WavetrendError> {
    // Stage 1: ESA = EMA(channel_length) on price
    ema_compute_into(data, channel_len, esa);

    // Stage 2: DE = EMA(channel_length) on |price - ESA|
    // We need a temporary buffer for the absolute differences
    // Then compute EMA into de
    if data.len() <= STACK_LIMIT {
        let mut abs_diff_buf = [0.0f64; STACK_LIMIT];
        let abs_diff = &mut abs_diff_buf[..data.len()];
        compute_abs_diff(abs_diff, data, esa, kernel);
        ema_compute_into(abs_diff, channel_len, de);
    } else {
        let mut abs_diff = vec![0.0; data.len()];
        compute_abs_diff(&mut abs_diff, data, esa, kernel);
        ema_compute_into(&abs_diff, channel_len, de);
    }

    // Stage 3: CI = (price - ESA) / (factor * DE)
    compute_ci(ci, data, esa, de, factor, kernel);

    // Stage 4: WT1 = EMA(average_length) on CI
    ema_compute_into(ci, average_len, wt1);

    // Stage 5: WT2 = SMA(ma_length) on WT1
    sma_compute_into(wt1, ma_len, wt2);

    Ok(())
}

#[inline(always)]
fn compute_abs_diff(out: &mut [f64], data: &[f64], esa: &[f64], kernel: Kernel) {
    let simd = kernel.to_non_batch();
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        match simd {
            Kernel::Avx512 => unsafe {
                absdiff_vec_avx512(out, data, esa);
                return;
            },
            Kernel::Avx2 => unsafe {
                absdiff_vec_avx2(out, data, esa);
                return;
            },
            _ => {}
        }
    }

    for i in 0..out.len() {
        out[i] = (data[i] - esa[i]).abs();
    }
}

#[inline(always)]
fn compute_ci(out: &mut [f64], data: &[f64], esa: &[f64], de: &[f64], factor: f64, kernel: Kernel) {
    let simd = kernel.to_non_batch();
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        match simd {
            Kernel::Avx512 => unsafe {
                ci_vec_avx512(out, data, esa, de, factor);
                return;
            },
            Kernel::Avx2 => unsafe {
                ci_vec_avx2(out, data, esa, de, factor);
                return;
            },
            _ => {}
        }
    }

    for i in 0..out.len() {
        let den = factor * de[i];
        if den != 0.0 && !data[i].is_nan() && !esa[i].is_nan() && !de[i].is_nan() {
            out[i] = (data[i] - esa[i]) / den;
        } else {
            out[i] = f64::NAN;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn absdiff_vec_avx2(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let n = dst.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let pd = dst.as_mut_ptr();
    let sign = _mm256_set1_pd(-0.0f64);
    let mut i = 0usize;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(pa.add(i));
        let vb = _mm256_loadu_pd(pb.add(i));
        let vd = _mm256_sub_pd(va, vb);
        let vabs = _mm256_andnot_pd(sign, vd);
        _mm256_storeu_pd(pd.add(i), vabs);
        i += 4;
    }
    while i < n {
        *pd.add(i) = (*pa.add(i) - *pb.add(i)).abs();
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ci_vec_avx2(dst: &mut [f64], data: &[f64], esa: &[f64], de: &[f64], factor: f64) {
    let n = dst.len();
    let px = data.as_ptr();
    let pe = esa.as_ptr();
    let pd = de.as_ptr();
    let pr = dst.as_mut_ptr();

    let vf = _mm256_set1_pd(factor);
    let vzero = _mm256_set1_pd(0.0);
    let vnan = _mm256_set1_pd(f64::NAN);

    let mut i = 0usize;
    while i + 4 <= n {
        let vx = _mm256_loadu_pd(px.add(i));
        let ve = _mm256_loadu_pd(pe.add(i));
        let vd = _mm256_loadu_pd(pd.add(i));

        let vnum = _mm256_sub_pd(vx, ve);
        let vden = _mm256_mul_pd(vf, vd);
        let vci = _mm256_div_pd(vnum, vden);

        let ord_x = _mm256_cmp_pd(vx, vx, _CMP_ORD_Q);
        let ord_e = _mm256_cmp_pd(ve, ve, _CMP_ORD_Q);
        let ord_d = _mm256_cmp_pd(vd, vd, _CMP_ORD_Q);
        let ord_all = _mm256_and_pd(ord_x, _mm256_and_pd(ord_e, ord_d));
        let den_zero = _mm256_cmp_pd(vden, vzero, _CMP_EQ_OQ);
        let valid = _mm256_andnot_pd(den_zero, ord_all);

        let vres = _mm256_blendv_pd(vnan, vci, valid);
        _mm256_storeu_pd(pr.add(i), vres);
        i += 4;
    }
    while i < n {
        let x = *px.add(i);
        let e = *pe.add(i);
        let d = *pd.add(i);
        let den = factor * d;
        *pr.add(i) = if den != 0.0 && x.is_finite() && e.is_finite() && d.is_finite() {
            (x - e) / den
        } else {
            f64::NAN
        };
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn absdiff_vec_avx512(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let n = dst.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let pd = dst.as_mut_ptr();
    let sign = _mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64);
    let sign_pd = _mm512_castsi512_pd(sign);

    let mut i = 0usize;
    while i + 8 <= n {
        let va = _mm512_loadu_pd(pa.add(i));
        let vb = _mm512_loadu_pd(pb.add(i));
        let vd = _mm512_sub_pd(va, vb);
        let vabs = _mm512_andnot_pd(sign_pd, vd);
        _mm512_storeu_pd(pd.add(i), vabs);
        i += 8;
    }
    while i < n {
        *pd.add(i) = (*pa.add(i) - *pb.add(i)).abs();
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ci_vec_avx512(dst: &mut [f64], data: &[f64], esa: &[f64], de: &[f64], factor: f64) {
    let n = dst.len();
    let px = data.as_ptr();
    let pe = esa.as_ptr();
    let pdv = de.as_ptr();
    let pr = dst.as_mut_ptr();

    let vf = _mm512_set1_pd(factor);
    let vzero = _mm512_set1_pd(0.0);
    let vnan = _mm512_set1_pd(f64::NAN);

    let mut i = 0usize;
    while i + 8 <= n {
        let vx = _mm512_loadu_pd(px.add(i));
        let ve = _mm512_loadu_pd(pe.add(i));
        let vd = _mm512_loadu_pd(pdv.add(i));

        let vnum = _mm512_sub_pd(vx, ve);
        let vden = _mm512_mul_pd(vf, vd);
        let vci = _mm512_div_pd(vnum, vden);

        let ord_x = _mm512_cmp_pd_mask(vx, vx, _CMP_ORD_Q);
        let ord_e = _mm512_cmp_pd_mask(ve, ve, _CMP_ORD_Q);
        let ord_d = _mm512_cmp_pd_mask(vd, vd, _CMP_ORD_Q);
        let ord_all = ord_x & ord_e & ord_d;
        let den_zero = _mm512_cmp_pd_mask(vden, vzero, _CMP_EQ_OQ);
        let valid = ord_all & (!den_zero);

        let vres = _mm512_mask_mov_pd(vnan, valid, vci);
        _mm512_storeu_pd(pr.add(i), vres);
        i += 8;
    }
    while i < n {
        let x = *px.add(i);
        let e = *pe.add(i);
        let d = *pdv.add(i);
        let den = factor * d;
        *pr.add(i) = if den != 0.0 && x.is_finite() && e.is_finite() && d.is_finite() {
            (x - e) / den
        } else {
            f64::NAN
        };
        i += 1;
    }
}

// Helper function for in-place EMA computation
#[inline(always)]
fn ema_compute_into(data: &[f64], period: usize, out: &mut [f64]) {
    if period == 0 || data.is_empty() {
        return;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;

    // Find first valid value
    let mut ema_val = f64::NAN;
    for i in 0..data.len() {
        if !data[i].is_nan() {
            if ema_val.is_nan() {
                ema_val = data[i];
            } else {
                ema_val = alpha * data[i] + beta * ema_val;
            }
            out[i] = ema_val;
        } else {
            out[i] = f64::NAN;
        }
    }
}

// Helper function for in-place SMA computation
#[inline(always)]
fn sma_compute_into(data: &[f64], period: usize, out: &mut [f64]) {
    if period == 0 || data.is_empty() {
        return;
    }

    let mut sum = 0.0;
    let mut count = 0;

    // Initialize with NaN
    for i in 0..out.len() {
        out[i] = f64::NAN;
    }

    // Calculate SMA
    for i in 0..data.len() {
        if !data[i].is_nan() {
            sum += data[i];
            count += 1;

            if i >= period {
                if !data[i - period].is_nan() {
                    sum -= data[i - period];
                    count -= 1;
                }
            }

            // We have enough values when count reaches period (and stays at period for a window)
            if count >= period {
                out[i] = sum / period as f64;
            }
        }
    }
}

/// Write wavetrend results directly to output slices - no allocations
#[inline]
pub fn wavetrend_into_slice(
    dst_wt1: &mut [f64],
    dst_wt2: &mut [f64],
    dst_wt_diff: &mut [f64],
    input: &WavetrendInput,
    kern: Kernel,
) -> Result<(), WavetrendError> {
    // Prepare and validate parameters
    let (data, channel_len, average_len, ma_len, factor, first, warmup_period) =
        wavetrend_prepare(input)?;

    // Validate output slice lengths
    if dst_wt1.len() != data.len() {
        return Err(WavetrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_wt1.len(),
        });
    }
    if dst_wt2.len() != data.len() {
        return Err(WavetrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_wt2.len(),
        });
    }
    if dst_wt_diff.len() != data.len() {
        return Err(WavetrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_wt_diff.len(),
        });
    }

    // Initialize NaN prefix for warmup period
    for i in 0..warmup_period.min(data.len()) {
        dst_wt1[i] = f64::NAN;
        dst_wt2[i] = f64::NAN;
        dst_wt_diff[i] = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        other => other,
    };

    // Compute directly into output slices
    wavetrend_compute_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
        dst_wt1,
        dst_wt2,
        dst_wt_diff,
        chosen,
    )?;

    Ok(())
}

#[derive(Clone, Debug)]
pub struct WavetrendStream {
    // ─────────── user-provided parameters (never change after construction) ───────────
    pub channel_length: usize,
    pub average_length: usize,
    pub ma_length: usize,
    pub factor: f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 1: “ESA” = EMA(channel_length) on price
    //   We seed immediately: last_esa = first finite price. α_ch = 2/(channel_length+1).
    esa_buf: VecDeque<f64>, // not used for seeding, but we keep it for capacity hint
    last_esa: Option<f64>,
    alpha_ch: f64,
    // NEW: cache beta for the channel EMA
    beta_ch: f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 2: “DE” = EMA(channel_length) on |price − ESA|
    //   Seed immediately: last_de = first |price₀ − esa₀|. Then recurse with α_ch again.
    de_buf: VecDeque<f64>, // not used for seeding, but kept for symmetry
    last_de: Option<f64>,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 3: “WT1” = EMA(average_length) on CI = (price − ESA)/(factor·DE)
    //   Seed immediately as soon as we get the very first valid CI. α_avg = 2/(average_length+1).
    ci_buf: VecDeque<f64>, // not used for seeding, but kept for capacity hint
    last_wt1: Option<f64>,
    alpha_avg: f64,
    // NEW: cache beta for the avg EMA
    beta_avg: f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 4: “WT2” = SMA(ma_length) on the most recent WT1 values
    //   We keep a sliding window of length ma_length in wt1_buf and a running_sum.
    wt1_buf: VecDeque<f64>, // ring of length ≤ ma_length; push every bar (finite or NaN)
    running_sum: f64,
    // NEW: number of finite WT1s in the last ma_length positions (O(1) update)
    sma_count: usize,
    // NEW: 1.0 / ma_length to avoid a divide in steady-state
    inv_ma: f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // history: so that streaming index = batch index. Every time update(...) is called,
    // we push the raw `price` so that the test harness can compare indexes directly.
    pub history: Vec<f64>,
}

impl WavetrendStream {
    pub fn try_new(p: WavetrendParams) -> Result<Self, WavetrendError> {
        let channel_length = p.channel_length.unwrap_or(9);
        let average_length = p.average_length.unwrap_or(12);
        let ma_length = p.ma_length.unwrap_or(3);
        let factor = p.factor.unwrap_or(0.015);

        // Validate that no period is zero
        if channel_length == 0 {
            return Err(WavetrendError::InvalidChannelLen {
                channel_length,
                data_len: 0,
            });
        }
        if average_length == 0 {
            return Err(WavetrendError::InvalidAverageLen {
                average_length,
                data_len: 0,
            });
        }
        if ma_length == 0 {
            return Err(WavetrendError::InvalidMaLen {
                ma_length,
                data_len: 0,
            });
        }

        // Precompute smoothing constants:
        //   α_ch  = 2 / (channel_length + 1)
        //   α_avg = 2 / (average_length + 1)
        let alpha_ch = 2.0 / (channel_length as f64 + 1.0);
        let alpha_avg = 2.0 / (average_length as f64 + 1.0);

        Ok(Self {
            channel_length,
            average_length,
            ma_length,
            factor,

            esa_buf: VecDeque::with_capacity(channel_length),
            last_esa: None,
            alpha_ch,
            beta_ch: 1.0 - alpha_ch,

            de_buf: VecDeque::with_capacity(channel_length),
            last_de: None,

            ci_buf: VecDeque::with_capacity(average_length),
            last_wt1: None,
            alpha_avg,
            beta_avg: 1.0 - alpha_avg,

            wt1_buf: VecDeque::with_capacity(ma_length),
            running_sum: 0.0,
            sma_count: 0,
            inv_ma: 1.0 / (ma_length as f64),

            history: Vec::new(),
        })
    }

    /// O(1) streaming update that mirrors the batch semantics exactly:
    /// - ESA = EMA(price, channel_length)
    /// - DE = EMA(|price-ESA|, channel_length)
    /// - WT1 = EMA(CI, average_length) where CI = (price-ESA) / (factor*DE)
    /// - WT2 = SMA over the last `ma_length` positions; emit only when all are finite
    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<(f64, f64, f64)> {
        // Keep 1:1 index mapping with batch path
        self.history.push(price);

        // We'll produce a single position into the SMA window no matter what.
        let mut wt1_val = f64::NAN;

        // ───── ESA & DE update only when input is finite (match scalar EMA behavior) ─────
        if price.is_finite() {
            // ESA
            if let Some(prev) = self.last_esa {
                let new_esa = ema_step(prev, price, self.alpha_ch, self.beta_ch);
                self.last_esa = Some(new_esa);
            } else {
                self.last_esa = Some(price);
            }

            // DE = EMA( |price - ESA|, channel_len )
            if let Some(esa_now) = self.last_esa {
                let abs_diff = fast_abs_f64(price - esa_now);
                if let Some(prev_de) = self.last_de {
                    let new_de = ema_step(prev_de, abs_diff, self.alpha_ch, self.beta_ch);
                    self.last_de = Some(new_de);
                } else {
                    self.last_de = Some(abs_diff);
                }
            }

            // CI and WT1
            if let (Some(esa_now), Some(de_now)) = (self.last_esa, self.last_de) {
                let den = self.factor * de_now;
                if den != 0.0 && den.is_finite() && esa_now.is_finite() {
                    let ci = (price - esa_now) / den;
                    if ci.is_finite() {
                        if let Some(prev_wt1) = self.last_wt1 {
                            let new_wt1 = ema_step(prev_wt1, ci, self.alpha_avg, self.beta_avg);
                            self.last_wt1 = Some(new_wt1);
                        } else {
                            self.last_wt1 = Some(ci);
                        }
                        if let Some(v) = self.last_wt1 {
                            wt1_val = v;
                        }
                    }
                }
            }
        }
        // If price is NaN, we intentionally do NOT advance ESA/DE/WT1,
        // but we still push a position (NaN) into the WT2 window below.

        // ───── WT2 = SMA over positions (ring implemented with VecDeque) ────────────────
        // Maintain fixed window length of `ma_length` after startup
        if self.wt1_buf.len() == self.ma_length {
            // Remove oldest position
            if let Some(leaving) = self.wt1_buf.pop_front() {
                if leaving.is_finite() {
                    self.running_sum -= leaving;
                    if self.sma_count > 0 {
                        self.sma_count -= 1;
                    }
                }
            }
        }
        // Insert current position (finite WT1 or NaN)
        self.wt1_buf.push_back(wt1_val);
        if wt1_val.is_finite() {
            self.running_sum += wt1_val;
            self.sma_count += 1;
        }

        // Emit only when (a) window has exactly ma_length positions, and (b) all are finite
        if self.wt1_buf.len() == self.ma_length && self.sma_count == self.ma_length {
            let wt1 = wt1_val; // guaranteed finite here
            let wt2 = self.running_sum * self.inv_ma;
            let diff = wt2 - wt1;
            Some((wt1, wt2, diff))
        } else {
            None
        }
    }
}

#[inline(always)]
fn ema_step(prev: f64, x: f64, alpha: f64, beta: f64) -> f64 {
    x.mul_add(alpha, beta * prev)
}

#[inline(always)]
fn fast_abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
}

// ---------------- CUDA Python VRAM handle (CAI v3 + DLPack v1.x) ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "WavetrendDeviceArrayF32", unsendable)]
pub struct WavetrendDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl WavetrendDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Producer kernels synchronize before returning, so no stream key is required.
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
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        // Compute target device and validate any requested dl_device.
        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err(
                            "dl_device mismatch for __dlpack__",
                        ));
                    }
                }
            }
        }
        let _ = stream;

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let dummy = cust::memory::DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[derive(Clone, Debug)]
pub struct WavetrendBatchRange {
    pub channel_length: (usize, usize, usize),
    pub average_length: (usize, usize, usize),
    pub ma_length: (usize, usize, usize),
    pub factor: (f64, f64, f64),
}

impl Default for WavetrendBatchRange {
    fn default() -> Self {
        Self {
            channel_length: (6, 12, 3),
            average_length: (9, 15, 3),
            ma_length: (2, 4, 1),
            factor: (0.010, 0.020, 0.005),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WavetrendBatchBuilder {
    range: WavetrendBatchRange,
    kernel: Kernel,
}

impl WavetrendBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn channel_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.channel_length = (start, end, step);
        self
    }
    pub fn channel_static(mut self, x: usize) -> Self {
        self.range.channel_length = (x, x, 0);
        self
    }
    pub fn avg_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.average_length = (start, end, step);
        self
    }
    pub fn avg_static(mut self, x: usize) -> Self {
        self.range.average_length = (x, x, 0);
        self
    }
    pub fn ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma_length = (start, end, step);
        self
    }
    pub fn ma_static(mut self, x: usize) -> Self {
        self.range.ma_length = (x, x, 0);
        self
    }
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }
    pub fn factor_static(mut self, x: f64) -> Self {
        self.range.factor = (x, x, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<WavetrendBatchOutput, WavetrendError> {
        wavetrend_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<WavetrendBatchOutput, WavetrendError> {
        WavetrendBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<WavetrendBatchOutput, WavetrendError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WavetrendBatchOutput, WavetrendError> {
        WavetrendBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "hlc3")
    }
}

pub fn wavetrend_batch_with_kernel(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    k: Kernel,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    // Prefer optimized scalar batch for Auto to avoid slower SIMD rows.
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        // SIMD-across-rows currently underperforms for this indicator; short-circuit to scalar batch
        Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(WavetrendError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    wavetrend_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct WavetrendBatchOutput {
    pub wt1: Vec<f64>,
    pub wt2: Vec<f64>,
    pub wt_diff: Vec<f64>,
    pub combos: Vec<WavetrendParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WavetrendBatchOutput {
    pub fn row_for_params(&self, p: &WavetrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.channel_length.unwrap_or(9) == p.channel_length.unwrap_or(9)
                && c.average_length.unwrap_or(12) == p.average_length.unwrap_or(12)
                && c.ma_length.unwrap_or(3) == p.ma_length.unwrap_or(3)
                && (c.factor.unwrap_or(0.015) - p.factor.unwrap_or(0.015)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &WavetrendParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.wt1[start..start + self.cols],
                &self.wt2[start..start + self.cols],
                &self.wt_diff[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &WavetrendBatchRange) -> Result<Vec<WavetrendParams>, WavetrendError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, WavetrendError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            return Ok((start..=end).step_by(st).collect());
        }
        // reversed bounds
        let st = step.max(1) as isize;
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(WavetrendError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, WavetrendError> {
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
                return Err(WavetrendError::InvalidRange {
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
            return Err(WavetrendError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let chs = axis_usize(r.channel_length)?;
    let avgs = axis_usize(r.average_length)?;
    let mas = axis_usize(r.ma_length)?;
    let factors = axis_f64(r.factor)?;

    let cap = chs
        .len()
        .checked_mul(avgs.len())
        .and_then(|x| x.checked_mul(mas.len()))
        .and_then(|x| x.checked_mul(factors.len()))
        .ok_or_else(|| WavetrendError::InvalidRange {
            start: "cap".into(),
            end: "overflow".into(),
            step: "mul".into(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &c in &chs {
        for &a in &avgs {
            for &m in &mas {
                for &f in &factors {
                    out.push(WavetrendParams {
                        channel_length: Some(c),
                        average_length: Some(a),
                        ma_length: Some(m),
                        factor: Some(f),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn wavetrend_batch_slice(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    wavetrend_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn wavetrend_batch_par_slice(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    wavetrend_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn wavetrend_batch_inner(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(WavetrendError::InvalidRange {
            start: "range".into(),
            end: "range".into(),
            step: "empty".into(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;
    let max_ch = combos
        .iter()
        .map(|c| c.channel_length.unwrap())
        .max()
        .unwrap();
    let max_avg = combos
        .iter()
        .map(|c| c.average_length.unwrap())
        .max()
        .unwrap();
    let max_ma = combos.iter().map(|c| c.ma_length.unwrap()).max().unwrap();
    let max_p = *[max_ch, max_avg, max_ma].iter().max().unwrap();
    if data.len() - first < max_p {
        return Err(WavetrendError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let _ = rows.checked_mul(cols).ok_or_else(|| WavetrendError::InvalidRange {
        start: rows.to_string(),
        end: cols.to_string(),
        step: "rows*cols".into(),
    })?;

    // Calculate warmup periods for each parameter combination
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            first + c.channel_length.unwrap() - 1 + c.average_length.unwrap() - 1
                + c.ma_length.unwrap()
                - 1
        })
        .collect();

    // Use helper functions for batch allocation
    let mut wt1_mu = make_uninit_matrix(rows, cols);
    let mut wt2_mu = make_uninit_matrix(rows, cols);
    let mut wt_diff_mu = make_uninit_matrix(rows, cols);

    // Initialize NaN prefixes
    init_matrix_prefixes(&mut wt1_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut wt2_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut wt_diff_mu, cols, &warmup_periods);

    // Convert to mutable slices for computation
    let mut wt1_guard = core::mem::ManuallyDrop::new(wt1_mu);
    let mut wt2_guard = core::mem::ManuallyDrop::new(wt2_mu);
    let mut wt_diff_guard = core::mem::ManuallyDrop::new(wt_diff_mu);

    let wt1: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt1_guard.as_mut_ptr() as *mut f64, wt1_guard.len())
    };
    let wt2: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt2_guard.as_mut_ptr() as *mut f64, wt2_guard.len())
    };
    let wt_diff: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt_diff_guard.as_mut_ptr() as *mut f64, wt_diff_guard.len())
    };

    let do_row = |row: usize, w1: &mut [f64], w2: &mut [f64], wd: &mut [f64]| unsafe {
        let p = &combos[row];
        let row_kernel = match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => wavetrend_row_avx512(
                data,
                first,
                p.channel_length.unwrap(),
                p.average_length.unwrap(),
                p.ma_length.unwrap(),
                p.factor.unwrap_or(0.015),
                w1,
                w2,
                wd,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => wavetrend_row_avx2(
                data,
                first,
                p.channel_length.unwrap(),
                p.average_length.unwrap(),
                p.ma_length.unwrap(),
                p.factor.unwrap_or(0.015),
                w1,
                w2,
                wd,
            ),
            _ => wavetrend_row_scalar(
                data,
                first,
                p.channel_length.unwrap(),
                p.average_length.unwrap(),
                p.ma_length.unwrap(),
                p.factor.unwrap_or(0.015),
                w1,
                w2,
                wd,
            ),
        };
        if let Err(e) = row_kernel {
            panic!("wavetrend row error: {:?}", e);
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            wt1.par_chunks_mut(cols)
                .zip(wt2.par_chunks_mut(cols))
                .zip(wt_diff.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((w1, w2), wd))| do_row(row, w1, w2, wd));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((w1, w2), wd))) in wt1
                .chunks_mut(cols)
                .zip(wt2.chunks_mut(cols))
                .zip(wt_diff.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, w1, w2, wd);
            }
        }
    } else {
        for (row, (((w1, w2), wd))) in wt1
            .chunks_mut(cols)
            .zip(wt2.chunks_mut(cols))
            .zip(wt_diff.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, w1, w2, wd);
        }
    }

    // Convert back to owned vectors
    let wt1_vec = unsafe {
        Vec::from_raw_parts(
            wt1_guard.as_mut_ptr() as *mut f64,
            wt1_guard.len(),
            wt1_guard.capacity(),
        )
    };
    let wt2_vec = unsafe {
        Vec::from_raw_parts(
            wt2_guard.as_mut_ptr() as *mut f64,
            wt2_guard.len(),
            wt2_guard.capacity(),
        )
    };
    let wt_diff_vec = unsafe {
        Vec::from_raw_parts(
            wt_diff_guard.as_mut_ptr() as *mut f64,
            wt_diff_guard.len(),
            wt_diff_guard.capacity(),
        )
    };

    Ok(WavetrendBatchOutput {
        wt1: wt1_vec,
        wt2: wt2_vec,
        wt_diff: wt_diff_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn wavetrend_batch_inner_into(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
    parallel: bool,
    out_wt1: &mut [f64],
    out_wt2: &mut [f64],
    out_wt_diff: &mut [f64],
) -> Result<Vec<WavetrendParams>, WavetrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(WavetrendError::InvalidRange {
            start: "range".into(),
            end: "range".into(),
            step: "empty".into(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;
    let max_ch = combos
        .iter()
        .map(|c| c.channel_length.unwrap())
        .max()
        .unwrap();
    let max_avg = combos
        .iter()
        .map(|c| c.average_length.unwrap())
        .max()
        .unwrap();
    let max_ma = combos.iter().map(|c| c.ma_length.unwrap()).max().unwrap();
    let max_p = *[max_ch, max_avg, max_ma].iter().max().unwrap();
    if data.len() - first < max_p {
        return Err(WavetrendError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let _ = rows.checked_mul(cols).ok_or_else(|| WavetrendError::InvalidRange {
        start: rows.to_string(),
        end: cols.to_string(),
        step: "rows*cols".into(),
    })?;

    // Initialize NaN prefixes for each row based on warmup period
    // Since _batch_inner_into receives external buffers, we must manually initialize
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.channel_length.unwrap() - 1 + combo.average_length.unwrap() - 1
            + combo.ma_length.unwrap()
            - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out_wt1[row_start + i] = f64::NAN;
            out_wt2[row_start + i] = f64::NAN;
            out_wt_diff[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, w1: &mut [f64], w2: &mut [f64], wd: &mut [f64]| unsafe {
        let p = &combos[row];
        let r = wavetrend_row_scalar(
            data,
            first,
            p.channel_length.unwrap(),
            p.average_length.unwrap(),
            p.ma_length.unwrap(),
            p.factor.unwrap_or(0.015),
            w1,
            w2,
            wd,
        );
        if let Err(e) = r {
            panic!("wavetrend row error: {:?}", e);
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_wt1
                .par_chunks_mut(cols)
                .zip(out_wt2.par_chunks_mut(cols))
                .zip(out_wt_diff.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((w1, w2), wd))| do_row(row, w1, w2, wd));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((w1, w2), wd))) in out_wt1
                .chunks_mut(cols)
                .zip(out_wt2.chunks_mut(cols))
                .zip(out_wt_diff.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, w1, w2, wd);
            }
        }
    } else {
        for (row, (((w1, w2), wd))) in out_wt1
            .chunks_mut(cols)
            .zip(out_wt2.chunks_mut(cols))
            .zip(out_wt_diff.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, w1, w2, wd);
        }
    }
    Ok(combos)
}

#[inline(always)]
unsafe fn wavetrend_row_scalar(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Scalar,
    )
}

// AVX2/AVX512 batch row stubs - always point to scalar
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx2(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx2,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx512,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512_short(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx512,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512_long(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx512,
    )
}

#[inline(always)]
unsafe fn wavetrend_row_with_kernel(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
    kernel: Kernel,
) -> Result<(), WavetrendError> {
    debug_assert_eq!(wt1.len(), data.len());
    debug_assert_eq!(wt2.len(), data.len());
    debug_assert_eq!(wd.len(), data.len());

    let warmup = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    wavetrend_compute_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup,
        wt1,
        wt2,
        wd,
        kernel,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_wavetrend_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = WavetrendParams {
            channel_length: None,
            average_length: None,
            ma_length: None,
            factor: None,
        };
        let input = WavetrendInput::from_candles(&candles, "hlc3", default_params);
        let output = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(output.wt1.len(), candles.close.len());
        Ok(())
    }

    fn check_wavetrend_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WavetrendInput::from_candles(&candles, "hlc3", WavetrendParams::default());
        let result = wavetrend_with_kernel(&input, kernel)?;
        let len = result.wt1.len();
        let expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ];
        let expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ];
        for (i, &val) in result.wt1[len - 5..].iter().enumerate() {
            let diff = (val - expected_wt1[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT1 mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_wt1[i]
            );
        }
        for (i, &val) in result.wt2[len - 5..].iter().enumerate() {
            let diff = (val - expected_wt2[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT2 mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_wt2[i]
            );
        }
        let last_five_diff = &result.wt_diff[len - 5..];
        for i in 0..5 {
            let expected = expected_wt2[i] - expected_wt1[i];
            let diff = (last_five_diff[i] - expected).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT_DIFF mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                last_five_diff[i],
                expected
            );
        }
        Ok(())
    }

    fn check_wavetrend_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WavetrendInput::with_default_candles(&candles);
        match input.data {
            WavetrendData::Candles { source, .. } => assert_eq!(source, "hlc3"),
            _ => panic!("Expected WavetrendData::Candles"),
        }
        let output = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(output.wt1.len(), candles.close.len());
        Ok(())
    }

    fn check_wavetrend_zero_channel(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(0),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        };
        let input = WavetrendInput::from_slice(&input_data, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with zero channel_length",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_channel_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(10),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        };
        let input = WavetrendInput::from_slice(&data_small, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with channel_length exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = WavetrendParams::default();
        let input = WavetrendInput::from_slice(&single_point, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WavetrendInput::from_candles(
            &candles,
            "hlc3",
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(12),
                ma_length: Some(3),
                factor: Some(0.015),
            },
        );
        let res = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(res.wt1.len(), candles.close.len());
        if res.wt1.len() > 240 {
            for (i, &val) in res.wt1[240..].iter().enumerate() {
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

    #[cfg(debug_assertions)]
    fn check_wavetrend_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            WavetrendParams::default(), // channel=9, average=12, ma=3, factor=0.015
            WavetrendParams {
                channel_length: Some(1),
                average_length: Some(1),
                ma_length: Some(1),
                factor: Some(0.001),
            }, // minimum viable parameters
            WavetrendParams {
                channel_length: Some(2),
                average_length: Some(2),
                ma_length: Some(2),
                factor: Some(0.005),
            }, // very small parameters
            WavetrendParams {
                channel_length: Some(5),
                average_length: Some(7),
                ma_length: Some(3),
                factor: Some(0.01),
            }, // small parameters
            WavetrendParams {
                channel_length: Some(10),
                average_length: Some(15),
                ma_length: Some(5),
                factor: Some(0.02),
            }, // medium parameters
            WavetrendParams {
                channel_length: Some(20),
                average_length: Some(25),
                ma_length: Some(7),
                factor: Some(0.025),
            }, // medium-large parameters
            WavetrendParams {
                channel_length: Some(30),
                average_length: Some(40),
                ma_length: Some(10),
                factor: Some(0.03),
            }, // large parameters
            WavetrendParams {
                channel_length: Some(50),
                average_length: Some(60),
                ma_length: Some(15),
                factor: Some(0.04),
            }, // very large parameters
            WavetrendParams {
                channel_length: Some(100),
                average_length: Some(120),
                ma_length: Some(20),
                factor: Some(0.05),
            }, // extreme parameters
            WavetrendParams {
                channel_length: Some(7),
                average_length: Some(11),
                ma_length: Some(3),
                factor: Some(0.013),
            }, // prime numbers
            WavetrendParams {
                channel_length: Some(13),
                average_length: Some(17),
                ma_length: Some(5),
                factor: Some(0.017),
            }, // more primes
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(3),
                ma_length: Some(12),
                factor: Some(0.015),
            }, // inverted typical sizes
            WavetrendParams {
                channel_length: Some(15),
                average_length: Some(15),
                ma_length: Some(15),
                factor: Some(0.015),
            }, // all equal
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(12),
                ma_length: Some(3),
                factor: Some(0.0001),
            }, // very small factor
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(12),
                ma_length: Some(3),
                factor: Some(1.0),
            }, // large factor
            WavetrendParams {
                channel_length: Some(3),
                average_length: Some(5),
                ma_length: Some(1),
                factor: Some(0.008),
            }, // fibonacci-like
            WavetrendParams {
                channel_length: Some(8),
                average_length: Some(13),
                ma_length: Some(2),
                factor: Some(0.021),
            }, // more fibonacci
            WavetrendParams {
                channel_length: Some(21),
                average_length: Some(34),
                ma_length: Some(8),
                factor: Some(0.034),
            }, // larger fibonacci
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = WavetrendInput::from_candles(&candles, "hlc3", params.clone());
            let output = wavetrend_with_kernel(&input, kernel)?;

            // Check wt1 output
            for (i, &val) in output.wt1.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }
            }

            // Check wt2 output
            for (i, &val) in output.wt2.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }
            }

            // Check wt_diff output
            for (i, &val) in output.wt_diff.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
						test_name, val, bits, i,
						params.channel_length.unwrap_or(9),
						params.average_length.unwrap_or(12),
						params.ma_length.unwrap_or(3),
						params.factor.unwrap_or(0.015),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_wavetrend_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    fn check_wavetrend_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let channel_length = 9;
        let average_length = 12;
        let ma_length = 3;
        let factor = 0.015;

        let input = WavetrendInput::from_candles(
            &candles,
            "hlc3",
            WavetrendParams {
                channel_length: Some(channel_length),
                average_length: Some(average_length),
                ma_length: Some(ma_length),
                factor: Some(factor),
            },
        );
        let full_output = wavetrend_with_kernel(&input, kernel)?;

        let mut stream = WavetrendStream::try_new(WavetrendParams {
            channel_length: Some(channel_length),
            average_length: Some(average_length),
            ma_length: Some(ma_length),
            factor: Some(factor),
        })?;

        let mut wt1_stream = Vec::with_capacity(candles.hlc3.len());
        let mut wt2_stream = Vec::with_capacity(candles.hlc3.len());
        let mut diff_stream = Vec::with_capacity(candles.hlc3.len());
        for &price in &candles.hlc3 {
            match stream.update(price) {
                Some((wt1, wt2, diff)) => {
                    wt1_stream.push(wt1);
                    wt2_stream.push(wt2);
                    diff_stream.push(diff);
                }
                None => {
                    wt1_stream.push(f64::NAN);
                    wt2_stream.push(f64::NAN);
                    diff_stream.push(f64::NAN);
                }
            }
        }

        let mut first_non_nan = None;
        for (i, &b) in full_output.wt1.iter().enumerate() {
            if !b.is_nan() {
                first_non_nan = Some(i);
                break;
            }
        }
        let start = first_non_nan.unwrap_or(0);
        assert_eq!(full_output.wt1.len(), wt1_stream.len());
        for (i, (&b, &s)) in full_output
            .wt1
            .iter()
            .zip(wt1_stream.iter())
            .enumerate()
            .skip(start)
        {
            if b.is_nan() || s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt1 f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in full_output.wt2.iter().zip(wt2_stream.iter()).enumerate() {
            if b.is_nan() || s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt2 f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in full_output
            .wt_diff
            .iter()
            .zip(diff_stream.iter())
            .enumerate()
        {
            if b.is_nan() || s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
				diff < 1e-9,
				"[{}] Wavetrend streaming wt_diff f64 mismatch at idx {}: full={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_wavetrend_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy: Generate reasonable parameter combinations and data
        let strat = (2usize..=30, 2usize..=30, 1usize..=10, 0.001f64..1.0f64).prop_flat_map(
            |(channel_len, average_len, ma_len, factor)| {
                // Ensure data length is always larger than the warmup period
                // warmup = first + channel_len - 1 + average_len - 1 + ma_len - 1
                // Since first is 0 for non-NaN data, warmup = channel_len + average_len + ma_len - 3
                // Add extra buffer to ensure we have valid data after warmup
                let min_len = channel_len + average_len + ma_len + 20;
                (min_len..400).prop_flat_map(move |data_len| {
                    (
                        prop::collection::vec(
                            (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        Just(channel_len),
                        Just(average_len),
                        Just(ma_len),
                        Just(factor),
                    )
                })
            },
        );

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(data, channel_len, average_len, ma_len, factor)| {
                    let params = WavetrendParams {
                        channel_length: Some(channel_len),
                        average_length: Some(average_len),
                        ma_length: Some(ma_len),
                        factor: Some(factor),
                    };
                    let input = WavetrendInput::from_slice(&data, params);

                    // Run with the specified kernel and scalar reference
                    let output = wavetrend_with_kernel(&input, kernel).unwrap();
                    let ref_output = wavetrend_with_kernel(&input, Kernel::Scalar).unwrap();

                    // Find first valid index (after warmup)
                    let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                    let expected_warmup =
                        first_valid + channel_len - 1 + average_len - 1 + ma_len - 1;

                    // Property 1: WT_DIFF = WT2 - WT1 for all valid indices
                    // Make sure we don't exceed the data length
                    for i in expected_warmup.min(data.len())..data.len() {
                        if output.wt1[i].is_finite() && output.wt2[i].is_finite() {
                            let expected_diff = output.wt2[i] - output.wt1[i];
                            let actual_diff = output.wt_diff[i];
                            prop_assert!(
                                (actual_diff - expected_diff).abs() <= 1e-9,
                                "WT_DIFF mismatch at idx {}: expected {}, got {}",
                                i,
                                expected_diff,
                                actual_diff
                            );
                        }
                    }

                    // Property 2: WT2 should be smoother than WT1 (it's an SMA of WT1)
                    // Check variance when we have enough valid data points
                    let valid_start = expected_warmup.min(data.len());
                    let valid_wt1: Vec<f64> = output.wt1[valid_start..]
                        .iter()
                        .filter(|&&x| x.is_finite())
                        .copied()
                        .collect();
                    let valid_wt2: Vec<f64> = output.wt2[valid_start..]
                        .iter()
                        .filter(|&&x| x.is_finite())
                        .copied()
                        .collect();

                    // Only check smoothness if we have sufficient data and ma_len > 1
                    if valid_wt1.len() > 10 && valid_wt2.len() > 10 && ma_len > 1 {
                        // WT2 should generally change less drastically than WT1
                        let mut wt1_changes = 0.0;
                        let mut wt2_changes = 0.0;
                        for i in 1..valid_wt1.len().min(valid_wt2.len()) {
                            wt1_changes += (valid_wt1[i] - valid_wt1[i - 1]).abs();
                            wt2_changes += (valid_wt2[i] - valid_wt2[i - 1]).abs();
                        }
                        // WT2 should have less total change (smoother)
                        if wt1_changes > 1e-6 {
                            // Only check if there's actual movement
                            prop_assert!(
                                wt2_changes <= wt1_changes * 1.1, // Allow 10% tolerance
                                "WT2 should be smoother: wt1_changes={}, wt2_changes={}",
                                wt1_changes,
                                wt2_changes
                            );
                        }
                    }

                    // Property 3: Constant price should lead to stable oscillator
                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9)
                        && data.len() > valid_start + 10
                    {
                        // After sufficient warmup, oscillator should stabilize near zero
                        let last_10_wt1: Vec<f64> = output.wt1[output.wt1.len() - 10..]
                            .iter()
                            .filter(|&&x| x.is_finite())
                            .copied()
                            .collect();
                        if last_10_wt1.len() >= 5 {
                            let avg_wt1: f64 =
                                last_10_wt1.iter().sum::<f64>() / last_10_wt1.len() as f64;
                            prop_assert!(
                                avg_wt1.abs() <= 1.0, // Oscillator should be near zero for constant prices
                                "Constant price should give near-zero oscillator: avg_wt1={}",
                                avg_wt1
                            );
                        }
                    }

                    // Property 3b: Factor scaling relationship
                    // CI = (price - ESA) / (factor * DE), so doubling factor should roughly halve CI/WT1 values
                    if factor < 0.5 && valid_start < data.len() {
                        // Only test when we can double without exceeding 1.0
                        let params_double = WavetrendParams {
                            channel_length: Some(channel_len),
                            average_length: Some(average_len),
                            ma_length: Some(ma_len),
                            factor: Some(factor * 2.0),
                        };
                        let input_double = WavetrendInput::from_slice(&data, params_double);
                        let output_double = wavetrend_with_kernel(&input_double, kernel).unwrap();

                        // Check relationship for a few valid points after warmup
                        let check_end = data.len().min(valid_start + 20);
                        let mut checked_count = 0;
                        for i in valid_start..check_end {
                            if output.wt1[i].is_finite()
                                && output_double.wt1[i].is_finite()
                                && output.wt1[i].abs() > 0.1
                            {
                                // Only check non-near-zero values
                                let ratio = output_double.wt1[i] / output.wt1[i];
                                // The relationship should be roughly inverse (doubling factor ~halves WT1)
                                // Allow generous tolerance as the relationship is affected by EMA smoothing
                                prop_assert!(
								(ratio - 0.5).abs() <= 0.35, // Allow 35% tolerance due to EMA effects
								"Factor doubling should roughly halve WT1 at idx {}: original={}, doubled={}, ratio={}",
								i, output.wt1[i], output_double.wt1[i], ratio
							);
                                checked_count += 1;
                                if checked_count >= 5 {
                                    // Check at most 5 points
                                    break;
                                }
                            }
                        }
                    }

                    // Property 4: Special case - when ma_len = 1, WT2 should equal WT1
                    if ma_len == 1 {
                        for i in valid_start..data.len() {
                            if output.wt1[i].is_finite() && output.wt2[i].is_finite() {
                                prop_assert!(
                                    (output.wt1[i] - output.wt2[i]).abs() <= 1e-9,
                                    "When ma_len=1, WT2 should equal WT1 at idx {}: wt1={}, wt2={}",
                                    i,
                                    output.wt1[i],
                                    output.wt2[i]
                                );
                            }
                        }
                    }

                    // Property 5: Kernel consistency - compare with scalar reference
                    for i in 0..data.len() {
                        let wt1 = output.wt1[i];
                        let wt1_ref = ref_output.wt1[i];
                        let wt2 = output.wt2[i];
                        let wt2_ref = ref_output.wt2[i];
                        let diff = output.wt_diff[i];
                        let diff_ref = ref_output.wt_diff[i];

                        // Check NaN consistency
                        if wt1.is_nan() || wt1_ref.is_nan() {
                            prop_assert!(
                                wt1.is_nan() && wt1_ref.is_nan(),
                                "NaN mismatch for WT1 at idx {}: kernel={:?}, ref={:?}",
                                i,
                                wt1,
                                wt1_ref
                            );
                        } else {
                            // Check ULP difference for finite values
                            let wt1_bits = wt1.to_bits();
                            let wt1_ref_bits = wt1_ref.to_bits();
                            let ulp_diff = wt1_bits.abs_diff(wt1_ref_bits);
                            prop_assert!(
                                (wt1 - wt1_ref).abs() <= 1e-9 || ulp_diff <= 4,
                                "WT1 mismatch at idx {}: kernel={}, ref={} (ULP={})",
                                i,
                                wt1,
                                wt1_ref,
                                ulp_diff
                            );
                        }

                        // Same checks for WT2 and WT_DIFF
                        if wt2.is_nan() || wt2_ref.is_nan() {
                            prop_assert!(
                                wt2.is_nan() && wt2_ref.is_nan(),
                                "NaN mismatch for WT2 at idx {}: kernel={:?}, ref={:?}",
                                i,
                                wt2,
                                wt2_ref
                            );
                        } else {
                            let wt2_bits = wt2.to_bits();
                            let wt2_ref_bits = wt2_ref.to_bits();
                            let ulp_diff = wt2_bits.abs_diff(wt2_ref_bits);
                            prop_assert!(
                                (wt2 - wt2_ref).abs() <= 1e-9 || ulp_diff <= 4,
                                "WT2 mismatch at idx {}: kernel={}, ref={} (ULP={})",
                                i,
                                wt2,
                                wt2_ref,
                                ulp_diff
                            );
                        }

                        if diff.is_nan() || diff_ref.is_nan() {
                            prop_assert!(
                                diff.is_nan() && diff_ref.is_nan(),
                                "NaN mismatch for WT_DIFF at idx {}: kernel={:?}, ref={:?}",
                                i,
                                diff,
                                diff_ref
                            );
                        } else {
                            let diff_bits = diff.to_bits();
                            let diff_ref_bits = diff_ref.to_bits();
                            let ulp_diff = diff_bits.abs_diff(diff_ref_bits);
                            prop_assert!(
                                (diff - diff_ref).abs() <= 1e-9 || ulp_diff <= 4,
                                "WT_DIFF mismatch at idx {}: kernel={}, ref={} (ULP={})",
                                i,
                                diff,
                                diff_ref,
                                ulp_diff
                            );
                        }
                    }

                    // Property 6: Warmup period validation
                    // Values before warmup should be NaN
                    for i in 0..expected_warmup.min(data.len()) {
                        prop_assert!(
                            output.wt1[i].is_nan(),
                            "WT1 should be NaN during warmup at idx {}: got {}",
                            i,
                            output.wt1[i]
                        );
                        prop_assert!(
                            output.wt2[i].is_nan(),
                            "WT2 should be NaN during warmup at idx {}: got {}",
                            i,
                            output.wt2[i]
                        );
                        prop_assert!(
                            output.wt_diff[i].is_nan(),
                            "WT_DIFF should be NaN during warmup at idx {}: got {}",
                            i,
                            output.wt_diff[i]
                        );
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_wavetrend_tests {
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

    generate_all_wavetrend_tests!(
        check_wavetrend_partial_params,
        check_wavetrend_accuracy,
        check_wavetrend_default_candles,
        check_wavetrend_zero_channel,
        check_wavetrend_channel_exceeds_length,
        check_wavetrend_very_small_dataset,
        check_wavetrend_nan_handling,
        check_wavetrend_streaming,
        check_wavetrend_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_wavetrend_tests!(check_wavetrend_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = WavetrendBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hlc3")?;

        let def = WavetrendParams::default();
        let (wt1_row, wt2_row, diff_row) = output.values_for(&def).expect("default row missing");

        assert_eq!(wt1_row.len(), c.close.len());
        assert_eq!(wt2_row.len(), c.close.len());
        assert_eq!(diff_row.len(), c.close.len());

        let expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ];
        let expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ];

        let start = wt1_row.len().saturating_sub(5);
        for (i, &v) in wt1_row[start..].iter().enumerate() {
            assert!(
                (v - expected_wt1[i]).abs() < 1e-8,
                "[{test}] default-row WT1 mismatch at idx {i}: {v} vs {expected}",
                test = test,
                i = i,
                v = v,
                expected = expected_wt1[i]
            );
        }
        for (i, &v) in wt2_row[start..].iter().enumerate() {
            assert!(
                (v - expected_wt2[i]).abs() < 1e-6,
                "[{test}] default-row WT2 mismatch at idx {i}: {v} vs {expected}",
                test = test,
                i = i,
                v = v,
                expected = expected_wt2[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (channel_start, channel_end, channel_step, avg_start, avg_end, avg_step, ma_start, ma_end, ma_step, factor_start, factor_end, factor_step)
            (2, 10, 2, 3, 12, 3, 1, 5, 1, 0.005, 0.015, 0.005), // Small ranges
            (5, 25, 5, 10, 30, 5, 2, 8, 2, 0.01, 0.03, 0.01),   // Medium ranges
            (20, 60, 10, 25, 75, 10, 5, 15, 5, 0.02, 0.05, 0.015), // Large ranges
            (2, 5, 1, 2, 5, 1, 1, 3, 1, 0.001, 0.005, 0.001),   // Dense small range
            (10, 30, 10, 15, 45, 15, 3, 9, 3, 0.015, 0.045, 0.015), // Medium spaced
            (50, 100, 25, 60, 120, 30, 10, 20, 5, 0.03, 0.06, 0.03), // Very large ranges
            (9, 9, 0, 12, 12, 0, 3, 3, 0, 0.015, 0.015, 0.0),   // Static defaults
            (1, 3, 1, 1, 3, 1, 1, 2, 1, 0.001, 0.003, 0.001),   // Minimum ranges
        ];

        for (cfg_idx, config) in test_configs.iter().enumerate() {
            let output = WavetrendBatchBuilder::new()
                .kernel(kernel)
                .channel_range(config.0, config.1, config.2)
                .avg_range(config.3, config.4, config.5)
                .ma_range(config.6, config.7, config.8)
                .factor_range(config.9, config.10, config.11)
                .apply_candles(&c, "hlc3")?;

            // Check wt1 output
            for (idx, &val) in output.wt1.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }
            }

            // Check wt2 output
            for (idx, &val) in output.wt2.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }
            }

            // Check wt_diff output
            for (idx, &val) in output.wt_diff.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.channel_length.unwrap_or(9),
						combo.average_length.unwrap_or(12),
						combo.ma_length.unwrap_or(3),
						combo.factor.unwrap_or(0.015)
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

#[cfg(feature = "python")]
#[pyfunction(name = "wavetrend")]
#[pyo3(signature = (data, channel_length, average_length, ma_length, factor, kernel=None))]
pub fn wavetrend_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    channel_length: usize,
    average_length: usize,
    ma_length: usize,
    factor: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = WavetrendParams {
        channel_length: Some(channel_length),
        average_length: Some(average_length),
        ma_length: Some(ma_length),
        factor: Some(factor),
    };
    let input = WavetrendInput::from_slice(slice_in, params);

    let (wt1_vec, wt2_vec, wt_diff_vec) = py
        .allow_threads(|| wavetrend_with_kernel(&input, kern).map(|o| (o.wt1, o.wt2, o.wt_diff)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        wt1_vec.into_pyarray(py),
        wt2_vec.into_pyarray(py),
        wt_diff_vec.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "WavetrendStream")]
pub struct WavetrendStreamPy {
    stream: WavetrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WavetrendStreamPy {
    #[new]
    fn new(
        channel_length: usize,
        average_length: usize,
        ma_length: usize,
        factor: f64,
    ) -> PyResult<Self> {
        let params = WavetrendParams {
            channel_length: Some(channel_length),
            average_length: Some(average_length),
            ma_length: Some(ma_length),
            factor: Some(factor),
        };
        let stream =
            WavetrendStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(WavetrendStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "wavetrend_batch")]
#[pyo3(signature = (data, channel_length_range, average_length_range, ma_length_range, factor_range, kernel=None))]
pub fn wavetrend_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    channel_length_range: (usize, usize, usize),
    average_length_range: (usize, usize, usize),
    ma_length_range: (usize, usize, usize),
    factor_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = WavetrendBatchRange {
        channel_length: channel_length_range,
        average_length: average_length_range,
        ma_length: ma_length_range,
        factor: factor_range,
    };

    let combos = expand_grid(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    // Allocate three arrays for the three outputs
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow for wavetrend_batch"))?;
    let wt1_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let wt2_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let wt_diff_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let slice_wt1 = unsafe { wt1_arr.as_slice_mut()? };
    let slice_wt2 = unsafe { wt2_arr.as_slice_mut()? };
    let slice_wt_diff = unsafe { wt_diff_arr.as_slice_mut()? };

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
                _ => unreachable!(),
            };
            wavetrend_batch_inner_into(
                slice_in,
                &sweep,
                simd,
                true,
                slice_wt1,
                slice_wt2,
                slice_wt_diff,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("wt1", wt1_arr.reshape((rows, cols))?)?;
    dict.set_item("wt2", wt2_arr.reshape((rows, cols))?)?;
    dict.set_item("wt_diff", wt_diff_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "channel_lengths",
        combos
            .iter()
            .map(|p| p.channel_length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "average_lengths",
        combos
            .iter()
            .map(|p| p.average_length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_lengths",
        combos
            .iter()
            .map(|p| p.ma_length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "factors",
        combos
            .iter()
            .map(|p| p.factor.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wavetrend_cuda_batch_dev")]
#[pyo3(signature = (data_f32, channel_length_range, average_length_range, ma_length_range, factor_range, device_id=0))]
pub fn wavetrend_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    channel_length_range: (usize, usize, usize),
    average_length_range: (usize, usize, usize),
    ma_length_range: (usize, usize, usize),
    factor_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = WavetrendBatchRange {
        channel_length: channel_length_range,
        average_length: average_length_range,
        ma_length: ma_length_range,
        factor: factor_range,
    };

    let (batch, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaWavetrend::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        cuda.wavetrend_batch_dev(slice_in, &sweep)
            .map(|b| (b, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "wt1",
        Py::new(
            py,
            WavetrendDeviceArrayF32Py {
                inner: batch.wt1,
                _ctx: ctx.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "wt2",
        Py::new(
            py,
            WavetrendDeviceArrayF32Py {
                inner: batch.wt2,
                _ctx: ctx.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "wt_diff",
        Py::new(
            py,
            WavetrendDeviceArrayF32Py {
                inner: batch.wt_diff,
                _ctx: ctx,
                device_id: dev_id,
            },
        )?,
    )?;

    // Return axis values (unique per dimension) to match ALMA-style metadata expectations
    let (c0, c1, cstep) = channel_length_range;
    let (a0, a1, astep) = average_length_range;
    let (m0, m1, mstep) = ma_length_range;
    let (f0, f1, fstep) = factor_range;
    let channel_axis: Vec<usize> = if cstep == 0 {
        vec![c0]
    } else {
        (c0..=c1).step_by(cstep).collect()
    };
    let average_axis: Vec<usize> = if astep == 0 {
        vec![a0]
    } else {
        (a0..=a1).step_by(astep).collect()
    };
    let ma_axis: Vec<usize> = if mstep == 0 {
        vec![m0]
    } else {
        (m0..=m1).step_by(mstep).collect()
    };
    let mut factor_axis: Vec<f64> = Vec::new();
    if fstep.abs() < f64::EPSILON || (f0 - f1).abs() < f64::EPSILON {
        factor_axis.push(f0);
    } else {
        let mut v = f0;
        while v <= f1 + fstep.abs() * 1e-12 {
            factor_axis.push(v);
            v += fstep;
        }
    }

    dict.set_item("channel_lengths", channel_axis.into_pyarray(py))?;
    dict.set_item("average_lengths", average_axis.into_pyarray(py))?;
    dict.set_item("ma_lengths", ma_axis.into_pyarray(py))?;
    dict.set_item("factors", factor_axis.into_pyarray(py))?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wavetrend_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, channel_length, average_length, ma_length, factor, device_id=0))]
pub fn wavetrend_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    channel_length: usize,
    average_length: usize,
    ma_length: usize,
    factor: f64,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array (rows x cols)"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;

    let params = WavetrendParams {
        channel_length: Some(channel_length),
        average_length: Some(average_length),
        ma_length: Some(ma_length),
        factor: Some(factor),
    };

    let (wt1, wt2, wt_diff, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaWavetrend::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        cuda.wavetrend_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map(|(a, b, c)| (a, b, c, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "wt1",
        Py::new(
            py,
            WavetrendDeviceArrayF32Py {
                inner: wt1,
                _ctx: ctx.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "wt2",
        Py::new(
            py,
            WavetrendDeviceArrayF32Py {
                inner: wt2,
                _ctx: ctx.clone(),
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item(
        "wt_diff",
        Py::new(
            py,
            WavetrendDeviceArrayF32Py {
                inner: wt_diff,
                _ctx: ctx,
                device_id: dev_id,
            },
        )?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("channel_length", channel_length)?;
    dict.set_item("average_length", average_length)?;
    dict.set_item("ma_length", ma_length)?;
    dict.set_item("factor", factor)?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wavetrend_js(
    data: &[f64],
    channel_length: usize,
    average_length: usize,
    ma_length: usize,
    factor: f64,
) -> Result<Vec<f64>, JsValue> {
    let params = WavetrendParams {
        channel_length: Some(channel_length),
        average_length: Some(average_length),
        ma_length: Some(ma_length),
        factor: Some(factor),
    };
    let input = WavetrendInput::from_slice(data, params);

    // Single allocation for flattened output [wt1..., wt2..., wt_diff...]
    let mut output = vec![0.0; data.len() * 3];
    let (wt1_part, rest) = output.split_at_mut(data.len());
    let (wt2_part, wt_diff_part) = rest.split_at_mut(data.len());

    wavetrend_into_slice(wt1_part, wt2_part, wt_diff_part, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wavetrend_into(
    in_ptr: *const f64,
    wt1_ptr: *mut f64,
    wt2_ptr: *mut f64,
    wt_diff_ptr: *mut f64,
    len: usize,
    channel_length: usize,
    average_length: usize,
    ma_length: usize,
    factor: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || wt1_ptr.is_null() || wt2_ptr.is_null() || wt_diff_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = WavetrendParams {
            channel_length: Some(channel_length),
            average_length: Some(average_length),
            ma_length: Some(ma_length),
            factor: Some(factor),
        };
        let input = WavetrendInput::from_slice(data, params);

        // Check if any output pointer aliases with input
        let needs_temp = in_ptr as *const u8 == wt1_ptr as *const u8
            || in_ptr as *const u8 == wt2_ptr as *const u8
            || in_ptr as *const u8 == wt_diff_ptr as *const u8;

        if needs_temp {
            // Use temporary buffer if any output aliases input
            let mut temp = vec![0.0; len * 3];
            let (temp_wt1, rest) = temp.split_at_mut(len);
            let (temp_wt2, temp_wt_diff) = rest.split_at_mut(len);

            wavetrend_into_slice(temp_wt1, temp_wt2, temp_wt_diff, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy from temp to output pointers
            let wt1_out = std::slice::from_raw_parts_mut(wt1_ptr, len);
            let wt2_out = std::slice::from_raw_parts_mut(wt2_ptr, len);
            let wt_diff_out = std::slice::from_raw_parts_mut(wt_diff_ptr, len);

            wt1_out.copy_from_slice(temp_wt1);
            wt2_out.copy_from_slice(temp_wt2);
            wt_diff_out.copy_from_slice(temp_wt_diff);
        } else {
            // Direct computation into output slices
            let wt1_out = std::slice::from_raw_parts_mut(wt1_ptr, len);
            let wt2_out = std::slice::from_raw_parts_mut(wt2_ptr, len);
            let wt_diff_out = std::slice::from_raw_parts_mut(wt_diff_ptr, len);

            wavetrend_into_slice(wt1_out, wt2_out, wt_diff_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wavetrend_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wavetrend_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WavetrendBatchConfig {
    pub channel_length_range: (usize, usize, usize),
    pub average_length_range: (usize, usize, usize),
    pub ma_length_range: (usize, usize, usize),
    pub factor_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WavetrendBatchJsOutput {
    pub wt1_values: Vec<f64>,
    pub wt2_values: Vec<f64>,
    pub wt_diff_values: Vec<f64>,
    pub channel_lengths: Vec<usize>,
    pub average_lengths: Vec<usize>,
    pub ma_lengths: Vec<usize>,
    pub factors: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = wavetrend_batch)]
pub fn wavetrend_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: WavetrendBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let sweep = WavetrendBatchRange {
        channel_length: (
            config.channel_length_range.0,
            config.channel_length_range.1,
            config.channel_length_range.2,
        ),
        average_length: (
            config.average_length_range.0,
            config.average_length_range.1,
            config.average_length_range.2,
        ),
        ma_length: (
            config.ma_length_range.0,
            config.ma_length_range.1,
            config.ma_length_range.2,
        ),
        factor: (
            config.factor_range.0,
            config.factor_range.1,
            config.factor_range.2,
        ),
    };

    let batch_output = wavetrend_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = WavetrendBatchJsOutput {
        wt1_values: batch_output.wt1,
        wt2_values: batch_output.wt2,
        wt_diff_values: batch_output.wt_diff,
        channel_lengths: batch_output
            .combos
            .iter()
            .map(|p| p.channel_length.unwrap())
            .collect(),
        average_lengths: batch_output
            .combos
            .iter()
            .map(|p| p.average_length.unwrap())
            .collect(),
        ma_lengths: batch_output
            .combos
            .iter()
            .map(|p| p.ma_length.unwrap())
            .collect(),
        factors: batch_output
            .combos
            .iter()
            .map(|p| p.factor.unwrap())
            .collect(),
        rows: batch_output.combos.len(),
        cols: data.len(),
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
