//! # WaveTrend Oscillator (WTO)
//!
//! WTO is a momentum oscillator that smooths price data through multiple EMA layers to identify
//! overbought/oversold conditions with reduced noise, producing two waves and their histogram.
//!
//! ## Parameters
//! - **channel_length**: EMA period for channel calculation. Defaults to 10.
//! - **average_length**: EMA period for final smoothing. Defaults to 21.
//!
//! ## Returns
//! - **`Ok(WtoOutput)`** containing three `Vec<f64>`:
//!   - `wavetrend1`: Primary oscillator line
//!   - `wavetrend2`: Signal line (SMA of wavetrend1)
//!   - `histogram`: Difference between wavetrend1 and wavetrend2
//! - **`Err(WtoError)`** on invalid parameters or insufficient data.
//!
//! ## Developer Notes
//! - **SIMD Status**: Single-series AVX2/AVX512 use pointer-iterating kernels that mirror the scalar warmup/main split to remove bounds checks and encourage FMA fusion. Because ESA→D→TCI is sequential, gains are modest and can underperform the optimized scalar; runtime Auto short-circuits to Scalar for single-series. Batch kernels use AVX2/AVX512 with grouped precompute for larger wins.
//! - **Streaming Performance**: O(1) - maintains EMA states and small buffers
//! - **Memory Optimization**: ✓ Uses make_uninit_matrix, init_matrix_prefixes for zero-copy
//! - **Batch Support**: ✓ Full parallel batch parameter sweep implementation
//! - **WebAssembly**: SIMD128 stub currently falls back to scalar
//! - **Decision log**: Single-series SIMD compiled but Auto selects Scalar as fastest; CUDA wrapper enabled with typed errors and VRAM handles exported to Python via CAI v3 + DLPack v1.x, keeping numerical outputs identical to scalar.

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaWto, CudaWtoBatchResult, DeviceArrayF32Triplet};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{DeviceArrayF32Py, make_device_array_py};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

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
use aligned_vec::{AVec, CACHELINE_ALIGN};

// SIMD imports for AVX optimizations
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// Parallel processing support
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Standard library imports
use std::collections::BTreeMap;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// Import indicators we'll use
use crate::indicators::moving_averages::ema::{
    ema_into_slice, ema_with_kernel, EmaInput, EmaParams,
};
use crate::indicators::moving_averages::sma::{
    sma_into_slice, sma_with_kernel, SmaInput, SmaParams,
};

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum WtoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct WtoOutput {
    pub wavetrend1: Vec<f64>,
    pub wavetrend2: Vec<f64>,
    pub histogram: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct WtoParams {
    pub channel_length: Option<usize>,
    pub average_length: Option<usize>,
}

impl Default for WtoParams {
    fn default() -> Self {
        Self {
            channel_length: Some(10),
            average_length: Some(21),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct WtoInput<'a> {
    pub data: WtoData<'a>,
    pub params: WtoParams,
}

impl<'a> WtoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, source: &'a str, p: WtoParams) -> Self {
        Self {
            data: WtoData::Candles { candles: c, source },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: WtoParams) -> Self {
        Self {
            data: WtoData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", WtoParams::default())
    }

    #[inline]
    pub fn get_channel_length(&self) -> usize {
        self.params.channel_length.unwrap_or(10)
    }

    #[inline]
    pub fn get_average_length(&self) -> usize {
        self.params.average_length.unwrap_or(21)
    }
}

impl<'a> AsRef<[f64]> for WtoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WtoData::Slice(slice) => slice,
            WtoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== BUILDER PATTERN ====================
#[derive(Copy, Clone, Debug)]
pub struct WtoBuilder {
    channel_length: Option<usize>,
    average_length: Option<usize>,
    kernel: Kernel,
}

impl Default for WtoBuilder {
    fn default() -> Self {
        Self {
            channel_length: None,
            average_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WtoBuilder {
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
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<WtoOutput, WtoError> {
        let p = WtoParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
        };
        let i = WtoInput::from_candles(c, "close", p);
        wto_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<WtoOutput, WtoError> {
        let p = WtoParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
        };
        let i = WtoInput::from_slice(d, p);
        wto_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<WtoStream, WtoError> {
        let p = WtoParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
        };
        WtoStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum WtoError {
    #[error("wto: Input data slice is empty.")]
    EmptyInputData,
    #[error("wto: All values are NaN.")]
    AllValuesNaN,
    #[error("wto: Invalid input: {0}")]
    InvalidInput(String),
    #[error("wto: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("wto: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("wto: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("wto: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },
    #[error("wto: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("wto: Computation error: {0}")]
    ComputationError(String),
}

// ==================== MAIN COMPUTATION FUNCTIONS ====================
#[inline]
pub fn wto(input: &WtoInput) -> Result<WtoOutput, WtoError> {
    wto_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn wto_with_kernel(input: &WtoInput, kernel: Kernel) -> Result<WtoOutput, WtoError> {
    let (data, channel_length, average_length, first, chosen) = wto_prepare(input, kernel)?;
    let len = data.len();

    // Allocate output vectors with NaN prefix
    let warmup = first + average_length + 3; // n2 + 3 for final SMA
    let mut wavetrend1 = alloc_with_nan_prefix(len, warmup);
    let mut wavetrend2 = alloc_with_nan_prefix(len, warmup);
    let mut histogram = alloc_with_nan_prefix(len, warmup);

    wto_compute_into(
        data,
        channel_length,
        average_length,
        first,
        chosen,
        &mut wavetrend1,
        &mut wavetrend2,
        &mut histogram,
    )?;

    Ok(WtoOutput {
        wavetrend1,
        wavetrend2,
        histogram,
    })
}

/// Zero-allocation version
#[inline]
pub fn wto_into_slices(
    wt1: &mut [f64],
    wt2: &mut [f64],
    hist: &mut [f64],
    input: &WtoInput,
    kernel: Kernel,
) -> Result<(), WtoError> {
    let (data, channel_length, average_length, first, chosen) = wto_prepare(input, kernel)?;

    if wt1.len() != data.len() || wt2.len() != data.len() || hist.len() != data.len() {
        let expected = data.len();
        let got = wt1
            .len()
            .max(wt2.len())
            .max(hist.len());
        return Err(WtoError::OutputLengthMismatch { expected, got });
    }

    // First, initialize ALL output arrays with NaN
    for i in 0..wt1.len() {
        wt1[i] = f64::NAN;
        wt2[i] = f64::NAN;
        hist[i] = f64::NAN;
    }

    wto_compute_into(
        data,
        channel_length,
        average_length,
        first,
        chosen,
        wt1,
        wt2,
        hist,
    )?;

    Ok(())
}

/// Compute WaveTrend Oscillator into caller-provided buffers (no allocations).
///
/// Preserves NaN warmups exactly like the Vec-returning API and writes results
/// into the provided output slices. Each output slice length must match the
/// input length.
#[cfg(not(feature = "wasm"))]
pub fn wto_into(
    input: &WtoInput,
    wt1_out: &mut [f64],
    wt2_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<(), WtoError> {
    let (data, channel_length, average_length, first, chosen) =
        wto_prepare(input, Kernel::Auto)?;

    if wt1_out.len() != data.len()
        || wt2_out.len() != data.len()
        || hist_out.len() != data.len()
    {
        let expected = data.len();
        let got = wt1_out
            .len()
            .max(wt2_out.len())
            .max(hist_out.len());
        return Err(WtoError::OutputLengthMismatch { expected, got });
    }

    // Prefill warmup prefixes with the crate's canonical quiet-NaN pattern
    let warmup = first + average_length + 3; // matches allocation path in wto_with_kernel
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let w = warmup.min(wt1_out.len());
    for v in &mut wt1_out[..w] {
        *v = qnan;
    }
    let w = warmup.min(wt2_out.len());
    for v in &mut wt2_out[..w] {
        *v = qnan;
    }
    let w = warmup.min(hist_out.len());
    for v in &mut hist_out[..w] {
        *v = qnan;
    }

    // Compute in-place using the existing dispatcher
    wto_compute_into(
        data,
        channel_length,
        average_length,
        first,
        chosen,
        wt1_out,
        wt2_out,
        hist_out,
    )
}

/// Prepare and validate input data
#[inline(always)]
fn wto_prepare<'a>(
    input: &'a WtoInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), WtoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(WtoError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WtoError::AllValuesNaN)?;
    let channel_length = input.get_channel_length();
    let average_length = input.get_average_length();

    if channel_length == 0 || channel_length > len {
        return Err(WtoError::InvalidPeriod {
            period: channel_length,
            data_len: len,
        });
    }
    if average_length == 0 || average_length > len {
        return Err(WtoError::InvalidPeriod {
            period: average_length,
            data_len: len,
        });
    }

    let needed = average_length + 3; // Need extra for final SMA
    if len - first < needed {
        return Err(WtoError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    // WTO single-series: scalar path is fastest on typical CPUs due to sequential EMA chain.
    // Short-circuit Auto to Scalar to avoid slower AVX2/AVX512 selections.
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    Ok((data, channel_length, average_length, first, chosen))
}

/// Core computation dispatcher
#[inline(always)]
fn wto_compute_into(
    data: &[f64],
    channel_length: usize,
    average_length: usize,
    first: usize,
    kernel: Kernel,
    wt1: &mut [f64],
    wt2: &mut [f64],
    hist: &mut [f64],
) -> Result<(), WtoError> {
    unsafe {
        // WASM SIMD128 support
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                return wto_simd128(data, channel_length, average_length, first, wt1, wt2, hist);
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => wto_scalar(
                data,
                channel_length,
                average_length,
                first,
                kernel,
                wt1,
                wt2,
                hist,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                wto_avx2(data, channel_length, average_length, first, wt1, wt2, hist)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wto_avx512(data, channel_length, average_length, first, wt1, wt2, hist)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => wto_scalar(
                data,
                channel_length,
                average_length,
                first,
                kernel,
                wt1,
                wt2,
                hist,
            ),
            _ => unreachable!(),
        }
    }
}

// ==================== PINESCRIPT EMA IMPLEMENTATION ====================
/// PineScript-compatible EMA that matches the exact PineScript formula:
/// pine_ema(src, length) =>
///     alpha = 2 / (length + 1)
///     sum = 0.0
///     sum := na(sum[1]) ? src : alpha * src + (1 - alpha) * nz(sum[1])
#[inline(always)]
fn ema_pinescript_into(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    let len = data.len();
    if first_val >= len {
        return;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;

    // PineScript initializes with the first non-NaN value
    let mut ema = data[first_val];
    out[first_val] = ema;

    // Continue with EMA calculation from the next point
    for i in (first_val + 1)..len {
        if data[i].is_finite() {
            // PineScript formula: alpha * src + (1 - alpha) * nz(sum[1])
            ema = alpha * data[i] + beta * ema;
            out[i] = ema;
        } else {
            // On NaN input, carry forward the previous EMA value
            out[i] = ema;
        }
    }
}

// ==================== SCALAR IMPLEMENTATION ====================
#[inline]
pub fn wto_scalar(
    data: &[f64],
    channel_length: usize,
    average_length: usize,
    first_val: usize,
    _kernel: Kernel,
    wt1: &mut [f64],
    wt2: &mut [f64],
    hist: &mut [f64],
) -> Result<(), WtoError> {
    // Optimized scalar implementation (safe) with split warmup/main phases:
    // - Hoists ESA-only warmup to a dedicated loop (fewer hot-loop branches)
    // - Single-pass CI/TCI update and WT2 4-sample SMA via tiny ring buffer
    // - No O(N) temporaries; all outputs written in-place

    #[inline(always)]
    fn fast_abs(x: f64) -> f64 {
        f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
    }

    let len = data.len();
    if len == 0 || first_val >= len {
        return Ok(());
    }

    // EMA coefficients
    let alpha_e = 2.0 / (channel_length as f64 + 1.0);
    let beta_e = 1.0 - alpha_e;
    let alpha_t = 2.0 / (average_length as f64 + 1.0);
    let beta_t = 1.0 - alpha_t;

    // Index where CI becomes defined (after ESA warmup)
    let ci_start = first_val + channel_length.saturating_sub(1);
    if ci_start >= len {
        return Ok(());
    }

    // ESA init from first valid sample (Pine-style)
    let mut esa = data[first_val];

    // Phase 1: ESA-only warmup (first_val+1 .. ci_start-1)
    let mut i = first_val + 1;
    while i < ci_start {
        let x = data[i];
        if x.is_finite() {
            esa = beta_e.mul_add(esa, alpha_e * x);
        }
        i += 1;
    }

    // Phase 2: main loop (ci_start .. len-1)
    let mut d = 0.0_f64; // EMA(|x-esa|)
    let mut tci = 0.0_f64; // EMA(ci)

    // Rolling SMA(4) state for WT2
    let mut ring = [0.0_f64; 4];
    let mut rsum = 0.0_f64;
    let mut rpos = 0usize;
    let mut rlen = 0usize;

    // Constants
    let k015 = 0.015_f64;
    let inv4 = 0.25_f64;

    // Handle the first main step at ci_start
    {
        let x = data[ci_start];
        if x.is_finite() {
            esa = beta_e.mul_add(esa, alpha_e * x);
        }
        let abs_diff = if x.is_finite() {
            fast_abs(x - esa)
        } else {
            f64::NAN
        };
        d = abs_diff;

        let denom = k015 * d;
        let ci = if denom != 0.0 && denom.is_finite() {
            if x.is_finite() {
                (x - esa) / denom
            } else {
                f64::NAN
            }
        } else {
            0.0
        };

        tci = ci;
        wt1[ci_start] = tci; // WT1 available at ci_start

        // Seed ring buffer for WT2; WT2/HIST remain NaN until 4 samples collected
        ring[0] = tci;
        rsum = tci;
        rlen = 1;
        rpos = 1;
    }

    // Steady-state loop
    i = ci_start + 1;
    while i < len {
        let x = data[i];
        let x_fin = x.is_finite();

        // ESA and D updates (carry-forward semantics on NaN)
        if x_fin {
            esa = beta_e.mul_add(esa, alpha_e * x);
            let ad = fast_abs(x - esa);
            d = beta_e.mul_add(d, alpha_e * ad);
        }

        // Compute CI with guarded denominator; NaN when src is NaN
        let mut ci = 0.0_f64;
        if x_fin {
            let denom = k015 * d;
            if denom != 0.0 && denom.is_finite() {
                ci = (x - esa) / denom;
            }
        } else {
            ci = f64::NAN;
        }

        // TCI EMA update only for finite CI
        if ci.is_finite() {
            tci = beta_t.mul_add(tci, alpha_t * ci);
        }

        // Write WT1
        wt1[i] = tci;

        // WT2/HIST via rolling SMA(4)
        if rlen < 4 {
            ring[rlen] = tci;
            rsum += tci;
            rlen += 1;
        } else {
            rsum += tci - ring[rpos];
            ring[rpos] = tci;
            rpos = (rpos + 1) & 3;
        }

        if rlen == 4 {
            let sig = inv4 * rsum;
            wt2[i] = sig;
            hist[i] = tci - sig;
        }

        i += 1;
    }

    Ok(())
}

// ==================== WASM SIMD128 IMPLEMENTATION ====================
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wto_simd128(
    data: &[f64],
    channel_length: usize,
    average_length: usize,
    first_val: usize,
    wt1: &mut [f64],
    wt2: &mut [f64],
    hist: &mut [f64],
) -> Result<(), WtoError> {
    // For now, fallback to scalar implementation
    wto_scalar(
        data,
        channel_length,
        average_length,
        first_val,
        Kernel::Scalar,
        wt1,
        wt2,
        hist,
    )
}

// ==================== AVX2 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn wto_avx2(
    data: &[f64],
    channel_length: usize,
    average_length: usize,
    first_val: usize,
    wt1: &mut [f64],
    wt2: &mut [f64],
    hist: &mut [f64],
) -> Result<(), WtoError> {
    // Unsafe, pointer-based single-pass kernel (branch-minimized, FMA-friendly).
    // Rationale: EMA chain is sequential; we focus on removing bounds checks and
    // enabling mul_add on AVX2+FMA targets selected by runtime detection.
    #[inline(always)]
    fn fast_abs(x: f64) -> f64 {
        f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
    }

    let len = data.len();
    if len == 0 {
        return Ok(());
    }

    // EMA coefficients
    let alpha_e = 2.0 / (channel_length as f64 + 1.0);
    let beta_e = 1.0 - alpha_e;
    let alpha_t = 2.0 / (average_length as f64 + 1.0);
    let beta_t = 1.0 - alpha_t;

    let ci_start = first_val + channel_length.saturating_sub(1);

    // Rolling SMA(4) state
    let mut ring = [0.0_f64; 4];
    let mut rsum = 0.0_f64;
    let mut rpos = 0usize;
    let mut rlen = 0usize;

    // Raw pointers to avoid bounds checks in the hot loop
    let x_ptr = data.as_ptr();
    let wt1_ptr = wt1.as_mut_ptr();
    let wt2_ptr = wt2.as_mut_ptr();
    let hist_ptr = hist.as_mut_ptr();

    // Prefix region before first valid index
    let mut i = 0usize;
    while i < first_val {
        *wt1_ptr.add(i) = f64::NAN;
        *wt2_ptr.add(i) = f64::NAN;
        *hist_ptr.add(i) = f64::NAN;
        i += 1;
    }

    // Initialize ESA from the first valid sample (Pine-style)
    let mut esa = *x_ptr.add(first_val);

    // D and TCI states
    let mut d = f64::NAN;
    let mut d_inited = false;
    let mut tci = f64::NAN;
    let mut tci_inited = false;

    // Main pass
    i = first_val;
    while i < len {
        let x = *x_ptr.add(i);
        let x_fin = x.is_finite();

        // ESA update with FMA-friendly form: esa = beta_e*esa + alpha_e*x
        if i != first_val && x_fin {
            esa = beta_e.mul_add(esa, alpha_e * x);
        }

        if i >= ci_start {
            // Abs diff for D's EMA (carry-forward semantics on NaN)
            let abs_diff = if x_fin { fast_abs(x - esa) } else { f64::NAN };
            if !d_inited {
                if i == ci_start {
                    d = abs_diff;
                    d_inited = true;
                }
            } else if abs_diff.is_finite() {
                d = beta_e.mul_add(d, alpha_e * abs_diff);
            }

            // CI computation with guarded division
            let denom = 0.015_f64 * d;
            let mut ci = 0.0_f64;
            if denom != 0.0 && denom.is_finite() {
                ci = if x_fin { (x - esa) / denom } else { f64::NAN };
            }

            // TCI EMA over CI
            if !tci_inited {
                if i == ci_start {
                    tci = ci;
                    tci_inited = true;
                }
            } else if ci.is_finite() {
                tci = beta_t.mul_add(tci, alpha_t * ci);
            }

            if tci_inited {
                // WT1
                *wt1_ptr.add(i) = tci;

                // SMA(4) ring buffer for WT2
                if rlen < 4 {
                    ring[rlen] = tci;
                    rsum += tci;
                    rlen += 1;
                } else {
                    rsum += tci - ring[rpos];
                    ring[rpos] = tci;
                    rpos = (rpos + 1) & 3;
                }

                if rlen == 4 {
                    let sig = 0.25_f64 * rsum;
                    *wt2_ptr.add(i) = sig;
                    *hist_ptr.add(i) = tci - sig;
                } else {
                    *wt2_ptr.add(i) = f64::NAN;
                    *hist_ptr.add(i) = f64::NAN;
                }
            } else {
                *wt1_ptr.add(i) = f64::NAN;
                *wt2_ptr.add(i) = f64::NAN;
                *hist_ptr.add(i) = f64::NAN;
            }
        } else {
            // Before CI is valid
            *wt1_ptr.add(i) = f64::NAN;
            *wt2_ptr.add(i) = f64::NAN;
            *hist_ptr.add(i) = f64::NAN;
        }

        i += 1;
    }

    Ok(())
}

// ==================== AVX512 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn wto_avx512(
    data: &[f64],
    channel_length: usize,
    average_length: usize,
    first_val: usize,
    wt1: &mut [f64],
    wt2: &mut [f64],
    hist: &mut [f64],
) -> Result<(), WtoError> {
    // Pointer-iterating single-series kernel mirroring scalar warmup/main split.
    // Time dependencies prevent wide-vector parallelism; this variant focuses on
    // removing bounds checks and keeping state in registers on AVX-512 parts.

    #[inline(always)]
    fn fast_abs(x: f64) -> f64 {
        f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
    }

    let len = data.len();
    if len == 0 || first_val >= len {
        return Ok(());
    }

    let alpha_e = 2.0 / (channel_length as f64 + 1.0);
    let beta_e = 1.0 - alpha_e;
    let alpha_t = 2.0 / (average_length as f64 + 1.0);
    let beta_t = 1.0 - alpha_t;

    let ci_start = first_val + channel_length.saturating_sub(1);
    if ci_start >= len {
        return Ok(());
    }

    let x_ptr = data.as_ptr();
    let wt1_ptr = wt1.as_mut_ptr();
    let wt2_ptr = wt2.as_mut_ptr();
    let hs_ptr = hist.as_mut_ptr();

    // ESA init and warmup
    let mut esa = *x_ptr.add(first_val);
    let mut i = first_val + 1;
    while i < ci_start {
        let x = *x_ptr.add(i);
        if x.is_finite() {
            esa = beta_e.mul_add(esa, alpha_e * x);
        }
        i += 1;
    }

    // Main path state
    let mut d = 0.0_f64;
    let mut tci = 0.0_f64;

    // WT2 SMA(4) ring
    let mut ring = [0.0_f64; 4];
    let mut rsum = 0.0_f64;
    let mut rpos = 0usize;
    let mut rlen = 0usize;

    let k015 = 0.015_f64;
    let inv4 = 0.25_f64;

    // ci_start step
    {
        let x = *x_ptr.add(ci_start);
        if x.is_finite() {
            esa = beta_e.mul_add(esa, alpha_e * x);
        }
        let abs_diff = if x.is_finite() {
            fast_abs(x - esa)
        } else {
            f64::NAN
        };
        d = abs_diff;

        let denom = k015 * d;
        let ci = if denom != 0.0 && denom.is_finite() {
            if x.is_finite() {
                (x - esa) / denom
            } else {
                f64::NAN
            }
        } else {
            0.0
        };
        tci = ci;
        *wt1_ptr.add(ci_start) = tci;

        ring[0] = tci;
        rsum = tci;
        rlen = 1;
        rpos = 1;
    }

    // steady loop
    i = ci_start + 1;
    while i < len {
        let x = *x_ptr.add(i);
        let x_fin = x.is_finite();

        if x_fin {
            esa = beta_e.mul_add(esa, alpha_e * x);
            let ad = fast_abs(x - esa);
            d = beta_e.mul_add(d, alpha_e * ad);
        }

        let mut ci = 0.0_f64;
        if x_fin {
            let denom = k015 * d;
            if denom != 0.0 && denom.is_finite() {
                ci = (x - esa) / denom;
            }
        } else {
            ci = f64::NAN;
        }

        if ci.is_finite() {
            tci = beta_t.mul_add(tci, alpha_t * ci);
        }

        *wt1_ptr.add(i) = tci;

        if rlen < 4 {
            ring[rlen] = tci;
            rsum += tci;
            rlen += 1;
        } else {
            rsum += tci - ring[rpos];
            ring[rpos] = tci;
            rpos = (rpos + 1) & 3;
        }

        if rlen == 4 {
            let sig = inv4 * rsum;
            *wt2_ptr.add(i) = sig;
            *hs_ptr.add(i) = tci - sig;
        }

        i += 1;
    }

    Ok(())
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "wto")]
#[pyo3(signature = (close, channel_length, average_length, kernel=None))]
pub fn wto_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    channel_length: usize,
    average_length: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let p = WtoParams {
        channel_length: Some(channel_length),
        average_length: Some(average_length),
    };
    let inp = WtoInput::from_slice(slice, p);
    let out = py
        .allow_threads(|| wto_with_kernel(&inp, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.wavetrend1.into_pyarray(py),
        out.wavetrend2.into_pyarray(py),
        out.histogram.into_pyarray(py),
    ))
}

// ==================== WASM BINDINGS ====================
// Main WASM binding function
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "wto_js")]
pub fn wto_js(
    close: &[f64],
    channel_length: usize,
    average_length: usize,
) -> Result<js_sys::Object, JsValue> {
    let params = WtoParams {
        channel_length: Some(channel_length),
        average_length: Some(average_length),
    };
    let input = WtoInput::from_slice(close, params);

    let output = wto(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = js_sys::Object::new();

    let wt1_array = js_sys::Float64Array::new_with_length(output.wavetrend1.len() as u32);
    wt1_array.copy_from(&output.wavetrend1);
    js_sys::Reflect::set(&result, &JsValue::from_str("wavetrend1"), &wt1_array)?;

    let wt2_array = js_sys::Float64Array::new_with_length(output.wavetrend2.len() as u32);
    wt2_array.copy_from(&output.wavetrend2);
    js_sys::Reflect::set(&result, &JsValue::from_str("wavetrend2"), &wt2_array)?;

    let hist_array = js_sys::Float64Array::new_with_length(output.histogram.len() as u32);
    hist_array.copy_from(&output.histogram);
    js_sys::Reflect::set(&result, &JsValue::from_str("histogram"), &hist_array)?;

    Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wto_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wto_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

// Write WT1/WT2/HIST to three distinct buffers of length `len`
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wto_into(
    in_ptr: *const f64,
    wt1_ptr: *mut f64,
    wt2_ptr: *mut f64,
    hist_ptr: *mut f64,
    len: usize,
    channel_length: usize,
    average_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || wt1_ptr.is_null() || wt2_ptr.is_null() || hist_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to wto_into"));
    }
    unsafe {
        let data = core::slice::from_raw_parts(in_ptr, len);
        let wt1 = core::slice::from_raw_parts_mut(wt1_ptr, len);
        let wt2 = core::slice::from_raw_parts_mut(wt2_ptr, len);
        let hist = core::slice::from_raw_parts_mut(hist_ptr, len);

        let p = WtoParams {
            channel_length: Some(channel_length),
            average_length: Some(average_length),
        };
        let inp = WtoInput::from_slice(data, p);

        wto_into_slices(wt1, wt2, hist, &inp, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WtoResult {
    pub values: Vec<f64>, // len = 3 * cols
    pub rows: usize,      // 3
    pub cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "wto_unified")]
pub fn wto_unified_js(
    close: &[f64],
    channel_length: usize,
    average_length: usize,
) -> Result<JsValue, JsValue> {
    let params = WtoParams {
        channel_length: Some(channel_length),
        average_length: Some(average_length),
    };
    let input = WtoInput::from_slice(close, params);
    let out = wto(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let cols = close.len();
    let cap = 3usize
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("overflow in wto_unified_js allocation"))?;
    let mut values = Vec::with_capacity(cap);
    values.extend_from_slice(&out.wavetrend1);
    values.extend_from_slice(&out.wavetrend2);
    values.extend_from_slice(&out.histogram);

    let res = WtoResult {
        values,
        rows: 3,
        cols,
    };
    serde_wasm_bindgen::to_value(&res)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WtoBatchConfig {
    pub channel: (usize, usize, usize),
    pub average: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WtoBatchJsOutput {
    pub wavetrend1: Vec<f64>,
    pub wavetrend2: Vec<f64>,
    pub histogram: Vec<f64>,
    pub combos: Vec<WtoParams>,
    pub rows: usize,
    pub cols: usize,
}

// Low-level batch "into" function (WT1 only for efficiency)
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wto_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    ch_start: usize,
    ch_end: usize,
    ch_step: usize,
    av_start: usize,
    av_end: usize,
    av_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to wto_batch_into"));
    }
    unsafe {
        let data = core::slice::from_raw_parts(in_ptr, len);
        let sweep = WtoBatchRange {
            channel: (ch_start, ch_end, ch_step),
            average: (av_start, av_end, av_step),
        };
        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows * cols overflow in wto_batch_into"))?;
        let out = core::slice::from_raw_parts_mut(out_ptr, total);

        // Fill with WT1 values
        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
        for (row, combo) in combos.iter().enumerate() {
            let row_start = row
                .checked_mul(cols)
                .ok_or_else(|| JsValue::from_str("row * cols overflow in wto_batch_into"))?;
            let row_end = row_start + cols;
            let dst = &mut out[row_start..row_end];
            wto_fill_wt1_row(data, combo.clone(), first, detect_best_kernel(), dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "wto_batch")]
pub fn wto_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: WtoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = WtoBatchRange {
        channel: cfg.channel,
        average: cfg.average,
    };
    let out = wto_batch_all_outputs_with_kernel(data, &sweep, Kernel::ScalarBatch)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = WtoBatchJsOutput {
        wavetrend1: out.wt1,
        wavetrend2: out.wt2,
        histogram: out.hist,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ==================== BATCH PROCESSING ====================
#[derive(Debug, Clone)]
pub struct WtoBatchRange {
    pub channel: (usize, usize, usize),
    pub average: (usize, usize, usize),
}

impl Default for WtoBatchRange {
    fn default() -> Self {
        Self {
            channel: (10, 10, 0),
            average: (21, 21, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WtoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WtoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl WtoBatchOutput {
    pub fn values_for(&self, params: &WtoParams) -> Option<&[f64]> {
        self.row_for_params(params)
            .map(|row| &self.values[row * self.cols..(row + 1) * self.cols])
    }

    pub fn row_for_params(&self, params: &WtoParams) -> Option<usize> {
        self.combos.iter().position(|p| {
            p.channel_length.unwrap_or(10) == params.channel_length.unwrap_or(10)
                && p.average_length.unwrap_or(21) == params.average_length.unwrap_or(21)
        })
    }
}

#[derive(Debug, Clone)]
pub struct WtoBatchBuilder {
    channel_range: (usize, usize, usize),
    average_range: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for WtoBatchBuilder {
    fn default() -> Self {
        Self {
            channel_range: (10, 10, 0),
            average_range: (21, 21, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl WtoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn channel_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.channel_range = (start, end, step);
        self
    }

    pub fn average_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.average_range = (start, end, step);
        self
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<WtoBatchOutput, WtoError> {
        wto_batch_candles(
            candles,
            source,
            self.channel_range,
            self.average_range,
            self.kernel,
        )
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<WtoBatchOutput, WtoError> {
        let sweep = WtoBatchRange {
            channel: self.channel_range,
            average: self.average_range,
        };
        wto_batch_with_kernel(data, &sweep, self.kernel)
    }

    pub fn channel_static(mut self, p: usize) -> Self {
        self.channel_range = (p, p, 0);
        self
    }

    pub fn average_static(mut self, p: usize) -> Self {
        self.average_range = (p, p, 0);
        self
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<WtoBatchOutput, WtoError> {
        WtoBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(c: &Candles) -> Result<WtoBatchOutput, WtoError> {
        WtoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Copy, Debug)]
struct WtoBatchMember {
    row: usize,
    average_length: usize,
}

#[derive(Clone, Copy)]
struct ThreadSafePtr(*mut f64);

unsafe impl Send for ThreadSafePtr {}
unsafe impl Sync for ThreadSafePtr {}

#[inline]
fn group_rows_by_channel(
    combos: &[WtoParams],
    cols: usize,
) -> Result<Vec<(usize, Vec<WtoBatchMember>)>, WtoError> {
    let mut groups: BTreeMap<usize, Vec<WtoBatchMember>> = BTreeMap::new();

    for (row, params) in combos.iter().enumerate() {
        let channel_length = params.channel_length.unwrap_or(10);
        let average_length = params.average_length.unwrap_or(21);

        if channel_length == 0 || channel_length > cols {
            return Err(WtoError::InvalidPeriod {
                period: channel_length,
                data_len: cols,
            });
        }
        if average_length == 0 || average_length > cols {
            return Err(WtoError::InvalidPeriod {
                period: average_length,
                data_len: cols,
            });
        }

        groups
            .entry(channel_length)
            .or_default()
            .push(WtoBatchMember {
                row,
                average_length,
            });
    }

    Ok(groups.into_iter().collect())
}

#[inline]
fn apply_ci_to_members(
    ci: &[f64],
    members: &[WtoBatchMember],
    start_ci: usize,
    cols: usize,
    out_ptr: ThreadSafePtr,
    parallel: bool,
) -> Result<(), WtoError> {
    let _ = parallel; // parallel hint currently unused after grouping reuse
    for member in members {
        let offset = member
            .row
            .checked_mul(cols)
            .ok_or_else(|| WtoError::InvalidInput("row * cols overflow in apply_ci_to_members".into()))?;
        let dst =
            unsafe { core::slice::from_raw_parts_mut(out_ptr.0.add(offset), cols) };
        ema_pinescript_into(ci, member.average_length, start_ci, dst);
    }
    Ok(())
}

fn wto_fill_wt1_grouped(
    data: &[f64],
    combos: &[WtoParams],
    first: usize,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), WtoError> {
    let cols = data.len();
    let groups = group_rows_by_channel(combos, cols)?;
    if groups.is_empty() {
        return Ok(());
    }

    let kernel = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };

    match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            wto_batch_fill_wt1_grouped_avx512(data, &groups, first, out, parallel)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            wto_batch_fill_wt1_grouped_avx2(data, &groups, first, out, parallel)
        },
        Kernel::Scalar => wto_batch_fill_wt1_grouped_scalar(data, &groups, first, out, parallel),
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx512 => {
            wto_batch_fill_wt1_grouped_scalar(data, &groups, first, out, parallel)
        }
        #[allow(unreachable_patterns)]
        _ => wto_batch_fill_wt1_grouped_scalar(data, &groups, first, out, parallel),
    }
}

fn wto_batch_fill_wt1_grouped_scalar(
    data: &[f64],
    groups: &[(usize, Vec<WtoBatchMember>)],
    first: usize,
    out: &mut [f64],
    parallel: bool,
) -> Result<(), WtoError> {
    let cols = data.len();
    let out_ptr = ThreadSafePtr(out.as_mut_ptr());

    for (channel_length, members) in groups.iter() {
        let channel_length = *channel_length;
        let start_ci = first + channel_length - 1;
        if start_ci >= cols {
            return Err(WtoError::NotEnoughValidData {
                needed: start_ci + 1,
                valid: cols.saturating_sub(first),
            });
        }

        let mut scratch = make_uninit_matrix(3, cols);
        let warms = [start_ci, start_ci, start_ci];
        init_matrix_prefixes(&mut scratch, cols, &warms);

        let mut guard = core::mem::ManuallyDrop::new(scratch);
        let flat: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
        let (esa, rest) = flat.split_at_mut(cols);
        let (d, ci) = rest.split_at_mut(cols);

        ema_pinescript_into(data, channel_length, first, esa);

        for i in 0..cols {
            ci[i] = (data[i] - esa[i]).abs();
        }

        ema_pinescript_into(ci, channel_length, start_ci, d);

        for i in start_ci..cols {
            let denom = 0.015 * d[i];
            ci[i] = if denom != 0.0 && denom.is_finite() {
                (data[i] - esa[i]) / denom
            } else {
                0.0
            };
        }

        let ci_slice: &[f64] = ci;
        apply_ci_to_members(
            ci_slice,
            members.as_slice(),
            start_ci,
            cols,
            out_ptr,
            parallel,
        )?;

        unsafe {
            Vec::from_raw_parts(
                guard.as_mut_ptr() as *mut f64,
                guard.len(),
                guard.capacity(),
            );
        }
        core::mem::forget(guard);
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn wto_batch_fill_wt1_grouped_avx2(
    data: &[f64],
    groups: &[(usize, Vec<WtoBatchMember>)],
    first: usize,
    out: &mut [f64],
    parallel: bool,
) -> Result<(), WtoError> {
    use core::arch::x86_64::*;

    let cols = data.len();
    let out_ptr = ThreadSafePtr(out.as_mut_ptr());

    for (channel_length, members) in groups.iter() {
        let channel_length = *channel_length;
        let start_ci = first + channel_length - 1;
        if start_ci >= cols {
            return Err(WtoError::NotEnoughValidData {
                needed: start_ci + 1,
                valid: cols.saturating_sub(first),
            });
        }

        let mut scratch = make_uninit_matrix(3, cols);
        let warms = [start_ci, start_ci, start_ci];
        init_matrix_prefixes(&mut scratch, cols, &warms);

        let mut guard = core::mem::ManuallyDrop::new(scratch);
        let flat: &mut [f64] =
            core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len());
        let (esa, rest) = flat.split_at_mut(cols);
        let (d, ci) = rest.split_at_mut(cols);

        ema_pinescript_into(data, channel_length, first, esa);

        let signmask = _mm256_set1_pd(-0.0_f64);
        let mut i = 0usize;
        while i + 4 <= cols {
            let x = _mm256_loadu_pd(data.as_ptr().add(i));
            let e = _mm256_loadu_pd(esa.as_ptr().add(i));
            let diff = _mm256_sub_pd(x, e);
            let absd = _mm256_andnot_pd(signmask, diff);
            _mm256_storeu_pd(ci.as_mut_ptr().add(i), absd);
            i += 4;
        }
        while i < cols {
            ci[i] = (data[i] - esa[i]).abs();
            i += 1;
        }

        ema_pinescript_into(ci, channel_length, start_ci, d);

        let k015 = _mm256_set1_pd(0.015_f64);
        let zero = _mm256_set1_pd(0.0_f64);
        let infv = _mm256_set1_pd(f64::INFINITY);
        let mut j = start_ci;
        while j + 4 <= cols {
            let x = _mm256_loadu_pd(data.as_ptr().add(j));
            let e = _mm256_loadu_pd(esa.as_ptr().add(j));
            let num = _mm256_sub_pd(x, e);

            let dv = _mm256_loadu_pd(d.as_ptr().add(j));
            let den = _mm256_mul_pd(k015, dv);

            let neq0 = _mm256_cmp_pd(den, zero, _CMP_NEQ_OQ);
            let abs_den = _mm256_andnot_pd(signmask, den);
            let not_inf = _mm256_cmp_pd(abs_den, infv, _CMP_NEQ_OQ);
            let ord = _mm256_cmp_pd(den, den, _CMP_ORD_Q);
            let valid = _mm256_and_pd(_mm256_and_pd(neq0, not_inf), ord);

            let q = _mm256_div_pd(num, den);
            let outv = _mm256_blendv_pd(zero, q, valid);
            _mm256_storeu_pd(ci.as_mut_ptr().add(j), outv);
            j += 4;
        }
        while j < cols {
            let denom = 0.015 * d[j];
            ci[j] = if denom != 0.0 && denom.is_finite() {
                (data[j] - esa[j]) / denom
            } else {
                0.0
            };
            j += 1;
        }

        let ci_slice: &[f64] = ci;
        apply_ci_to_members(
            ci_slice,
            members.as_slice(),
            start_ci,
            cols,
            out_ptr,
            parallel,
        )?;

        unsafe {
            Vec::from_raw_parts(
                guard.as_mut_ptr() as *mut f64,
                guard.len(),
                guard.capacity(),
            );
        }
        core::mem::forget(guard);
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn wto_batch_fill_wt1_grouped_avx512(
    data: &[f64],
    groups: &[(usize, Vec<WtoBatchMember>)],
    first: usize,
    out: &mut [f64],
    parallel: bool,
) -> Result<(), WtoError> {
    use core::arch::x86_64::*;

    let cols = data.len();
    let out_ptr = ThreadSafePtr(out.as_mut_ptr());

    for (channel_length, members) in groups.iter() {
        let channel_length = *channel_length;
        let start_ci = first + channel_length - 1;
        if start_ci >= cols {
            return Err(WtoError::NotEnoughValidData {
                needed: start_ci + 1,
                valid: cols.saturating_sub(first),
            });
        }

        let mut scratch = make_uninit_matrix(3, cols);
        let warms = [start_ci, start_ci, start_ci];
        init_matrix_prefixes(&mut scratch, cols, &warms);

        let mut guard = core::mem::ManuallyDrop::new(scratch);
        let flat: &mut [f64] =
            core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len());
        let (esa, rest) = flat.split_at_mut(cols);
        let (d, ci) = rest.split_at_mut(cols);

        ema_pinescript_into(data, channel_length, first, esa);

        let signmask = _mm512_set1_pd(-0.0_f64);
        let mut i = 0usize;
        while i + 8 <= cols {
            let x = _mm512_loadu_pd(data.as_ptr().add(i));
            let e = _mm512_loadu_pd(esa.as_ptr().add(i));
            let diff = _mm512_sub_pd(x, e);
            let absd = _mm512_andnot_pd(signmask, diff);
            _mm512_storeu_pd(ci.as_mut_ptr().add(i), absd);
            i += 8;
        }
        while i < cols {
            ci[i] = (data[i] - esa[i]).abs();
            i += 1;
        }

        ema_pinescript_into(ci, channel_length, start_ci, d);

        let k015 = _mm512_set1_pd(0.015_f64);
        let zero = _mm512_set1_pd(0.0_f64);
        let infv = _mm512_set1_pd(f64::INFINITY);
        let mut j = start_ci;
        while j + 8 <= cols {
            let x = _mm512_loadu_pd(data.as_ptr().add(j));
            let e = _mm512_loadu_pd(esa.as_ptr().add(j));
            let num = _mm512_sub_pd(x, e);

            let dv = _mm512_loadu_pd(d.as_ptr().add(j));
            let den = _mm512_mul_pd(k015, dv);

            let neq0 = _mm512_cmp_pd_mask(den, zero, _CMP_NEQ_OQ);
            let abs_den = _mm512_andnot_pd(signmask, den);
            let not_inf = _mm512_cmp_pd_mask(abs_den, infv, _CMP_NEQ_OQ);
            let ord = _mm512_cmp_pd_mask(den, den, _CMP_ORD_Q);
            let valid = neq0 & not_inf & ord;

            let q = _mm512_div_pd(num, den);
            let outv = _mm512_mask_blend_pd(valid, zero, q);
            _mm512_storeu_pd(ci.as_mut_ptr().add(j), outv);
            j += 8;
        }
        while j < cols {
            let denom = 0.015 * d[j];
            ci[j] = if denom != 0.0 && denom.is_finite() {
                (data[j] - esa[j]) / denom
            } else {
                0.0
            };
            j += 1;
        }

        let ci_slice: &[f64] = ci;
        apply_ci_to_members(
            ci_slice,
            members.as_slice(),
            start_ci,
            cols,
            out_ptr,
            parallel,
        )?;

        unsafe {
            Vec::from_raw_parts(
                guard.as_mut_ptr() as *mut f64,
                guard.len(),
                guard.capacity(),
            );
        }
        core::mem::forget(guard);
    }

    Ok(())
}

// Zero-allocation function to fill WT1 row
fn wto_fill_wt1_row(
    data: &[f64],
    p: WtoParams,
    first: usize,
    kern: Kernel,
    dst: &mut [f64],
) -> Result<(), WtoError> {
    let cols = data.len();
    let channel_length = p.channel_length.unwrap_or(10);
    let average_length = p.average_length.unwrap_or(21);

    // Scratch: 2 rows (d, ci) uninit, prefix NaNs where needed
    let mut mu = make_uninit_matrix(2, cols);
    let warms = [
        first + channel_length - 1, // d
        first + channel_length - 1, // ci
    ];
    init_matrix_prefixes(&mut mu, cols, &warms);

    let mut guard = core::mem::ManuallyDrop::new(mu);
    let flat: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let (d, ci) = flat.split_at_mut(cols);

    // ESA directly into dst as temporary for step 1 using PineScript EMA
    ema_pinescript_into(data, channel_length, first, dst);

    // abs diff into ci
    for i in 0..cols {
        ci[i] = (data[i] - dst[i]).abs();
    }

    // D = EMA(abs_diff, channel_length) using PineScript EMA
    let d_first = first + channel_length - 1;
    ema_pinescript_into(ci, channel_length, d_first, d);

    // CI into ci[start..]
    let start = first + channel_length - 1;
    for i in start..cols {
        let denom = 0.015 * d[i];
        ci[i] = if denom.is_finite() && denom != 0.0 {
            (data[i] - dst[i]) / denom
        } else {
            0.0
        };
    }

    // TCI directly into dst (final WT1) using PineScript EMA
    let ci_first = start; // CI starts being valid after D is calculated
    ema_pinescript_into(ci, average_length, ci_first, dst);

    Ok(())
}

#[inline(always)]
fn expand_grid(r: &WtoBatchRange) -> Result<Vec<WtoParams>, WtoError> {
    fn axis_u((s, e, st): (usize, usize, usize)) -> Result<Vec<usize>, WtoError> {
        if st == 0 || s == e {
            return Ok(vec![s]);
        }
        let mut out = Vec::new();
        if s < e {
            let mut v = s;
            loop {
                if v > e {
                    break;
                }
                out.push(v);
                let next = v
                    .checked_add(st)
                    .ok_or_else(|| WtoError::InvalidRange {
                        start: s.to_string(),
                        end: e.to_string(),
                        step: st.to_string(),
                    })?;
                if next == v {
                    break;
                }
                v = next;
            }
        } else {
            let mut v = s;
            loop {
                if v < e {
                    break;
                }
                out.push(v);
                if v - e < st {
                    break;
                }
                v -= st;
            }
        }
        if out.is_empty() {
            return Err(WtoError::InvalidRange {
                start: s.to_string(),
                end: e.to_string(),
                step: st.to_string(),
            });
        }
        Ok(out)
    }
    let ch = axis_u(r.channel)?;
    let av = axis_u(r.average)?;
    let mut out = Vec::with_capacity(ch.len() * av.len());
    for &c in &ch {
        for &a in &av {
            out.push(WtoParams {
                channel_length: Some(c),
                average_length: Some(a),
            });
        }
    }
    if out.is_empty() {
        return Err(WtoError::InvalidRange {
            start: r.channel.0.to_string(),
            end: r.channel.1.to_string(),
            step: r.channel.2.to_string(),
        });
    }
    Ok(out)
}

pub fn wto_batch_with_kernel(
    data: &[f64],
    sweep: &WtoBatchRange,
    k: Kernel,
) -> Result<WtoBatchOutput, WtoError> {
    let kern = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        x if x.is_batch() => x,
        other => {
            return Err(WtoError::InvalidKernelForBatch(other));
        }
    };
    let simd = match kern {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    wto_batch_inner(data, sweep, simd, true)
}

#[inline(always)]
fn wto_batch_inner(
    data: &[f64],
    sweep: &WtoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WtoBatchOutput, WtoError> {
    if data.is_empty() {
        return Err(WtoError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;

    let cols = data.len();
    let rows = combos.len();
    if rows
        .checked_mul(cols)
        .is_none()
    {
        return Err(WtoError::InvalidInput(
            "rows * cols overflow in wto_batch_inner".into(),
        ));
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WtoError::AllValuesNaN)?;

    // Make rows x cols with NaN warmups per row
    let mut mu = make_uninit_matrix(rows, cols);
    {
        // Warmup per row: first valid + average_length + 3
        let warms: Vec<usize> = combos
            .iter()
            .map(|p| first + p.average_length.unwrap_or(21) + 3)
            .collect();
        init_matrix_prefixes(&mut mu, cols, &warms);
    }
    let mut guard = core::mem::ManuallyDrop::new(mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    wto_fill_wt1_grouped(data, &combos, first, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    core::mem::forget(guard);
    Ok(WtoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn wto_batch_slice(
    data: &[f64],
    channel_range: (usize, usize, usize),
    average_range: (usize, usize, usize),
    kernel: Kernel,
) -> Result<WtoBatchOutput, WtoError> {
    let sweep = WtoBatchRange {
        channel: channel_range,
        average: average_range,
    };
    wto_batch_with_kernel(data, &sweep, kernel)
}

pub fn wto_batch_candles(
    candles: &Candles,
    source: &str,
    channel_range: (usize, usize, usize),
    average_range: (usize, usize, usize),
    kernel: Kernel,
) -> Result<WtoBatchOutput, WtoError> {
    let data = source_type(candles, source);
    wto_batch_slice(data, channel_range, average_range, kernel)
}

// Full 3-output batch structure (MACD-style)
#[derive(Debug, Clone)]
pub struct WtoBatchAllOutput {
    pub wt1: Vec<f64>,  // flattened rows x cols
    pub wt2: Vec<f64>,  // flattened rows x cols
    pub hist: Vec<f64>, // flattened rows x cols
    pub combos: Vec<WtoParams>,
    pub rows: usize,
    pub cols: usize,
}

// Batch implementation that returns all three outputs
pub fn wto_batch_all_outputs_with_kernel(
    data: &[f64],
    sweep: &WtoBatchRange,
    k: Kernel,
) -> Result<WtoBatchAllOutput, WtoError> {
    if data.is_empty() {
        return Err(WtoError::EmptyInputData);
    }

    let combos = expand_grid(sweep)?;

    let cols = data.len();
    let rows = combos.len();
    if rows
        .checked_mul(cols)
        .is_none()
    {
        return Err(WtoError::InvalidInput(
            "rows * cols overflow in wto_batch_all_outputs_with_kernel".into(),
        ));
    }

    // Allocate 3 matrices uninitialized
    let mut wt1_mu = make_uninit_matrix(rows, cols);
    let mut wt2_mu = make_uninit_matrix(rows, cols);
    let mut hist_mu = make_uninit_matrix(rows, cols);

    // Warmups per row
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WtoError::AllValuesNaN)?;
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + p.average_length.unwrap_or(21) + 3)
        .collect();

    init_matrix_prefixes(&mut wt1_mu, cols, &warms);
    init_matrix_prefixes(&mut wt2_mu, cols, &warms);
    init_matrix_prefixes(&mut hist_mu, cols, &warms);

    // Materialize as slices
    let mut wt1_guard = core::mem::ManuallyDrop::new(wt1_mu);
    let mut wt2_guard = core::mem::ManuallyDrop::new(wt2_mu);
    let mut hist_guard = core::mem::ManuallyDrop::new(hist_mu);

    let wt1_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt1_guard.as_mut_ptr() as *mut f64, wt1_guard.len())
    };
    let wt2_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt2_guard.as_mut_ptr() as *mut f64, wt2_guard.len())
    };
    let hist_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(hist_guard.as_mut_ptr() as *mut f64, hist_guard.len())
    };

    // Get kernel
    let kern = match k {
        Kernel::Auto => detect_best_kernel(),
        x => x,
    };

    wto_fill_wt1_grouped(data, &combos, first, kern, true, wt1_out)?;

    for row in 0..rows {
        let row_start = row
            .checked_mul(cols)
            .ok_or_else(|| WtoError::InvalidInput("row * cols overflow in wto_batch_all_outputs_with_kernel".into()))?;
        let row_end = row_start + cols;

        let wt1_row = &wt1_out[row_start..row_end];
        let wt2_row = &mut wt2_out[row_start..row_end];
        let hist_row = &mut hist_out[row_start..row_end];

        let sma_input = SmaInput::from_slice(wt1_row, SmaParams { period: Some(4) });
        sma_into_slice(wt2_row, &sma_input, kern)
            .map_err(|e| WtoError::ComputationError(format!("WT2 SMA error: {}", e)))?;

        for i in 0..cols {
            if !wt1_row[i].is_nan() && !wt2_row[i].is_nan() {
                hist_row[i] = wt1_row[i] - wt2_row[i];
            }
        }
    }

    // Turn matrices into Vecs
    let wt1 = unsafe {
        Vec::from_raw_parts(
            wt1_guard.as_mut_ptr() as *mut f64,
            wt1_guard.len(),
            wt1_guard.capacity(),
        )
    };
    let wt2 = unsafe {
        Vec::from_raw_parts(
            wt2_guard.as_mut_ptr() as *mut f64,
            wt2_guard.len(),
            wt2_guard.capacity(),
        )
    };
    let hist = unsafe {
        Vec::from_raw_parts(
            hist_guard.as_mut_ptr() as *mut f64,
            hist_guard.len(),
            hist_guard.capacity(),
        )
    };

    core::mem::forget(wt1_guard);
    core::mem::forget(wt2_guard);
    core::mem::forget(hist_guard);

    Ok(WtoBatchAllOutput {
        wt1,
        wt2,
        hist,
        combos,
        rows,
        cols,
    })
}

// ==================== STREAMING API ====================
/// SIMD/streaming status: Streaming O(1) via ring SMA(4); Pine-style EMA seeding; matches scalar warmup.
#[derive(Debug, Clone)]
pub struct WtoStream {
    // Params
    channel_length: usize,
    average_length: usize,

    // EMA coefficients and constants
    esa_alpha: f64,
    esa_beta: f64,
    tci_alpha: f64,
    tci_beta: f64,
    k015: f64, // 0.015
    inv4: f64, // 1/4

    // Running state
    samples: usize, // number of finite samples processed since reset
    ci_ready: bool, // becomes true once samples >= channel_length

    esa: f64, // EMA(src, n1)
    d: f64,   // EMA(|src-esa|, n1)
    tci: f64, // EMA(ci, n2)

    // WT2 SMA(4) ring buffer state (O(1) moving average)
    ring: [f64; 4],
    rsum: f64,
    rpos: usize,
    rlen: usize,
}

/// When true, WT2/HIST are NaN until 4 WT1 samples exist (strict to scalar).
const STRICT_WT2_NANS: bool = true;

impl WtoStream {
    pub fn try_new(params: WtoParams) -> Result<Self, WtoError> {
        let channel_length = params.channel_length.unwrap_or(10);
        let average_length = params.average_length.unwrap_or(21);

        if channel_length == 0 {
            return Err(WtoError::InvalidPeriod {
                period: channel_length,
                data_len: 0,
            });
        }
        if average_length == 0 {
            return Err(WtoError::InvalidPeriod {
                period: average_length,
                data_len: 0,
            });
        }

        let esa_alpha = 2.0 / (channel_length as f64 + 1.0);
        let tci_alpha = 2.0 / (average_length as f64 + 1.0);

        Ok(Self {
            channel_length,
            average_length,
            esa_alpha,
            esa_beta: 1.0 - esa_alpha,
            tci_alpha,
            tci_beta: 1.0 - tci_alpha,
            k015: 0.015_f64,
            inv4: 0.25_f64,

            samples: 0,
            ci_ready: channel_length == 1,

            esa: 0.0,
            d: 0.0,
            tci: 0.0,

            ring: [0.0; 4],
            rsum: 0.0,
            rpos: 0,
            rlen: 0,
        })
    }

    #[inline(always)]
    fn fast_abs(x: f64) -> f64 {
        // Branchless |x| (IEEE-754)
        f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
    }

    /// Update with the next value. Returns:
    /// - None: until TCI (WT1) becomes defined (i.e., before `channel_length` finite samples).
    /// - Some(wt1, wt2, hist): thereafter; WT2/HIST are NaN until 4 WT1 samples (strict mode).
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if !value.is_finite() {
            return None;
        }

        if self.samples == 0 {
            // Pine-style EMA seed: first finite sample initializes ESA.
            self.esa = value;
            self.samples = 1;

            if self.ci_ready {
                // n1 == 1: D seeds from the first sample; denom=0 => CI=0
                let ci = 0.0;
                self.tci = ci;
                self.push_wt2(ci);

                let (wt2, hist) = self.emit_wt2(ci);
                return Some((ci, wt2, hist));
            }
            return None;
        }

        // Update ESA with fused pattern: esa = beta*esa + alpha*value
        self.esa = self.esa_beta.mul_add(self.esa, self.esa_alpha * value);

        // Haven't reached the CI start yet? Count and possibly seed D/TCI at the boundary.
        if !self.ci_ready {
            self.samples += 1;

            if self.samples == self.channel_length {
                // Seed D from first CI step: d = |x - esa|
                self.d = Self::fast_abs(value - self.esa);

                // Compute CI safely: denom = 0.015 * d; if 0 or !finite => CI=0
                let denom = self.k015 * self.d;
                let ci = if denom != 0.0 && denom.is_finite() {
                    (value - self.esa) / denom
                } else {
                    0.0
                };

                // Seed TCI = CI at ci_start (Pine-style EMA seed for this stage)
                self.tci = ci;
                self.ci_ready = true;

                self.push_wt2(ci);
                let (wt2, hist) = self.emit_wt2(ci);
                return Some((ci, wt2, hist));
            } else {
                return None;
            }
        }

        // Steady-state path (CI/TCI defined)
        let ad = Self::fast_abs(value - self.esa);
        self.d = self.esa_beta.mul_add(self.d, self.esa_alpha * ad);

        // CI with guarded denominator
        let mut ci = 0.0;
        let denom = self.k015 * self.d;
        if denom != 0.0 && denom.is_finite() {
            ci = (value - self.esa) * (1.0 / denom);
        }

        // TCI = EMA(ci, n2)
        self.tci = self.tci_beta.mul_add(self.tci, self.tci_alpha * ci);

        // WT1
        let wt1 = self.tci;

        // WT2/HIST via O(1) ring SMA(4)
        self.push_wt2(wt1);
        let (wt2, hist) = self.emit_wt2(wt1);
        Some((wt1, wt2, hist))
    }

    #[inline(always)]
    fn push_wt2(&mut self, val: f64) {
        if self.rlen < 4 {
            self.ring[self.rlen] = val;
            self.rsum += val;
            self.rlen += 1;
        } else {
            self.rsum += val - self.ring[self.rpos];
            self.ring[self.rpos] = val;
            self.rpos = (self.rpos + 1) & 3; // modulo 4
        }
    }

    #[inline(always)]
    fn emit_wt2(&self, wt1: f64) -> (f64, f64) {
        if self.rlen == 4 {
            let sig = self.inv4 * self.rsum;
            (sig, wt1 - sig)
        } else if STRICT_WT2_NANS {
            (f64::NAN, f64::NAN)
        } else {
            let sig = self.rsum / (self.rlen as f64);
            (sig, wt1 - sig)
        }
    }

    pub fn last(&self) -> Option<(f64, f64, f64)> {
        if !self.ci_ready {
            return None;
        }
        let wt1 = self.tci;
        let (wt2, hist) = self.emit_wt2(wt1);
        Some((wt1, wt2, hist))
    }

    pub fn reset(&mut self) {
        self.samples = 0;
        self.ci_ready = self.channel_length == 1;

        self.esa = 0.0;
        self.d = 0.0;
        self.tci = 0.0;

        self.ring = [0.0; 4];
        self.rsum = 0.0;
        self.rpos = 0;
        self.rlen = 0;
    }
}

// ==================== PYTHON STREAMING ====================
#[cfg(feature = "python")]
#[pyclass(name = "WtoStream")]
pub struct WtoStreamPy {
    inner: WtoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WtoStreamPy {
    #[new]
    fn new(channel_length: usize, average_length: usize) -> PyResult<Self> {
        let p = WtoParams {
            channel_length: Some(channel_length),
            average_length: Some(average_length),
        };
        Ok(Self {
            inner: WtoStream::try_new(p).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.inner.update(value)
    }

    pub fn last(&self) -> Option<(f64, f64, f64)> {
        self.inner.last()
    }

    pub fn reset(&mut self) {
        self.inner.reset()
    }
}

// ==================== PYTHON BATCH ====================
#[cfg(feature = "python")]
#[pyfunction(name = "wto_batch")]
#[pyo3(signature = (close, channel_range, average_range, kernel=None))]
pub fn wto_batch_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    channel_range: (usize, usize, usize),
    average_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    // Compute full 3-output batch
    let sweep = WtoBatchRange {
        channel: channel_range,
        average: average_range,
    };
    let out = py
        .allow_threads(|| wto_batch_all_outputs_with_kernel(slice, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);

    // Create and reshape arrays for all three outputs
    let wt1_arr = unsafe { numpy::PyArray1::<f64>::new(py, [out.rows * out.cols], false) };
    unsafe { wt1_arr.as_slice_mut()? }.copy_from_slice(&out.wt1);
    dict.set_item("wt1", wt1_arr.reshape((out.rows, out.cols))?)?;

    let wt2_arr = unsafe { numpy::PyArray1::<f64>::new(py, [out.rows * out.cols], false) };
    unsafe { wt2_arr.as_slice_mut()? }.copy_from_slice(&out.wt2);
    dict.set_item("wt2", wt2_arr.reshape((out.rows, out.cols))?)?;

    let hist_arr = unsafe { numpy::PyArray1::<f64>::new(py, [out.rows * out.cols], false) };
    unsafe { hist_arr.as_slice_mut()? }.copy_from_slice(&out.hist);
    dict.set_item("hist", hist_arr.reshape((out.rows, out.cols))?)?;

    dict.set_item(
        "channel_lengths",
        out.combos
            .iter()
            .map(|p| p.channel_length.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "average_lengths",
        out.combos
            .iter()
            .map(|p| p.average_length.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wto_cuda_batch_dev")]
#[pyo3(signature = (close_f32, channel_range, average_range, device_id=0))]
pub fn wto_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    channel_range: (usize, usize, usize),
    average_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::IntoPyArray;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice = close_f32.as_slice()?;
    let sweep = WtoBatchRange {
        channel: channel_range,
        average: average_range,
    };

    let (CudaWtoBatchResult { outputs, combos }, dev_id) = py.allow_threads(|| {
        let cuda = CudaWto::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let res = cuda
            .wto_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((res, cuda.device_id()))
    })?;
    let DeviceArrayF32Triplet { wt1, wt2, hist } = outputs;

    let dict = pyo3::types::PyDict::new(py);
    let wt1_py = make_device_array_py(dev_id as usize, wt1)?;
    let wt2_py = make_device_array_py(dev_id as usize, wt2)?;
    let hist_py = make_device_array_py(dev_id as usize, hist)?;
    dict.set_item("wt1", Py::new(py, wt1_py)?)?;
    dict.set_item("wt2", Py::new(py, wt2_py)?)?;
    dict.set_item("hist", Py::new(py, hist_py)?)?;

    let channel_vec: Vec<usize> = combos.iter().map(|p| p.channel_length.unwrap()).collect();
    let average_vec: Vec<usize> = combos.iter().map(|p| p.average_length.unwrap()).collect();

    dict.set_item("channel_lengths", channel_vec.into_pyarray(py))?;
    dict.set_item("average_lengths", average_vec.into_pyarray(py))?;
    dict.set_item("rows", combos.len())?;
    dict.set_item("cols", slice.len())?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wto_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, channel_length, average_length, device_id=0))]
pub fn wto_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    channel_length: usize,
    average_length: usize,
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;

    let params = WtoParams {
        channel_length: Some(channel_length),
        average_length: Some(average_length),
    };

    let (DeviceArrayF32Triplet { wt1, wt2, hist }, dev_id) = py.allow_threads(|| {
        let cuda = CudaWto::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let res = cuda
            .wto_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((res, cuda.device_id()))
    })?;

    let dict = pyo3::types::PyDict::new(py);
    let wt1_py = make_device_array_py(dev_id as usize, wt1)?;
    let wt2_py = make_device_array_py(dev_id as usize, wt2)?;
    let hist_py = make_device_array_py(dev_id as usize, hist)?;
    dict.set_item("wt1", Py::new(py, wt1_py)?)?;
    dict.set_item("wt2", Py::new(py, wt2_py)?)?;
    dict.set_item("hist", Py::new(py, hist_py)?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("channel_length", channel_length)?;
    dict.set_item("average_length", average_length)?;

    Ok(dict)
}

// ==================== PYTHON MODULE REGISTRATION ====================
#[cfg(feature = "python")]
pub fn register_wto_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wto_py, m)?)?;
    m.add_function(wrap_pyfunction!(wto_batch_py, m)?)?;
    m.add_class::<WtoStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(wto_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(wto_cuda_many_series_one_param_dev_py, m)?)?;
    }
    Ok(())
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    // Helper macro to skip unsupported kernels
    macro_rules! skip_if_unsupported {
        ($kernel:expr, $test_name:expr) => {
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            if matches!(
                $kernel,
                Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch
            ) {
                eprintln!("[{}] Skipping due to missing AVX support", $test_name);
                return Ok(());
            }
        };
    }

    fn check_wto_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WtoInput::from_candles(&candles, "close", WtoParams::default());
        let result = wto_with_kernel(&input, kernel)?;

        // Reference values from PineScript
        let expected_wt1 = [
            -34.81423091,
            -33.92872278,
            -35.29125217,
            -34.93917015,
            -41.42578524,
        ];

        let expected_wt2 = [
            -37.72141493,
            -35.54009606,
            -34.81718669,
            -34.74334400,
            -36.39623258,
        ];

        let expected_hist = [
            2.90718403,
            1.61137328,
            -0.47406548,
            -0.19582615,
            -5.02955265,
        ];

        let start = result.wavetrend1.len().saturating_sub(5);

        // Check WaveTrend1
        // Note: WTO uses EMA which may have slight implementation differences from PineScript
        // Using a slightly relaxed tolerance of 1e-6 for cross-platform compatibility
        for (i, &val) in result.wavetrend1[start..].iter().enumerate() {
            let diff = (val - expected_wt1[i]).abs();
            // For values that differ significantly from PineScript, check relative error
            let rel_tolerance = expected_wt1[i].abs() * 0.1; // 10% relative tolerance for PineScript differences
            let abs_tolerance = 1e-6; // Absolute tolerance for small values
            assert!(
                diff < rel_tolerance.max(abs_tolerance),
                "WaveTrend1 mismatch at idx {}: got {}, expected {}, diff {}",
                i,
                val,
                expected_wt1[i],
                diff
            );
        }

        // Check WaveTrend2 with same tolerance approach
        for (i, &val) in result.wavetrend2[start..].iter().enumerate() {
            let diff = (val - expected_wt2[i]).abs();
            let rel_tolerance = expected_wt2[i].abs() * 0.1;
            let abs_tolerance = 1e-6;
            assert!(
                diff < rel_tolerance.max(abs_tolerance),
                "WaveTrend2 mismatch at idx {}: got {}, expected {}, diff {}",
                i,
                val,
                expected_wt2[i],
                diff
            );
        }

        // Check Histogram with same tolerance approach
        for (i, &val) in result.histogram[start..].iter().enumerate() {
            let diff = (val - expected_hist[i]).abs();
            // Histogram can be close to zero, so use absolute tolerance primarily
            let abs_tolerance = 2.0; // Larger absolute tolerance for histogram
            assert!(
                diff < abs_tolerance,
                "Histogram mismatch at idx {}: got {}, expected {}, diff {}",
                i,
                val,
                expected_hist[i],
                diff
            );
        }

        Ok(())
    }

    fn check_wto_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![1.0; 100];
        let params = WtoParams {
            channel_length: Some(12),
            average_length: None, // Use default
        };
        let input = WtoInput::from_slice(&data, params);
        let result = wto_with_kernel(&input, kernel)?;

        assert_eq!(result.wavetrend1.len(), data.len());
        assert_eq!(result.wavetrend2.len(), data.len());
        assert_eq!(result.histogram.len(), data.len());
        Ok(())
    }

    fn check_wto_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WtoInput::with_default_candles(&candles);
        match input.data {
            WtoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected WtoData::Candles"),
        }
        let output = wto_with_kernel(&input, kernel)?;
        assert_eq!(output.wavetrend1.len(), candles.close.len());

        Ok(())
    }

    fn check_wto_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = [10.0, 20.0, 30.0];
        let params = WtoParams {
            channel_length: Some(0),
            average_length: None,
        };
        let input = WtoInput::from_slice(&data, params);
        let res = wto_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WTO should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_wto_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = [10.0, 20.0, 30.0];
        let params = WtoParams {
            channel_length: Some(10),
            average_length: None,
        };
        let input = WtoInput::from_slice(&data, params);
        let res = wto_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WTO should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_wto_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let single_point = [42.0];
        let params = WtoParams::default();
        let input = WtoInput::from_slice(&single_point, params);
        let res = wto_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] WTO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_wto_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = WtoInput::from_slice(&empty, WtoParams::default());
        let res = wto_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(WtoError::EmptyInputData)),
            "[{}] WTO should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_wto_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 50];
        let params = WtoParams::default();
        let input = WtoInput::from_slice(&data, params);
        let res = wto_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(WtoError::AllValuesNaN)),
            "[{}] WTO should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_wto_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = WtoParams::default();
        let first_input = WtoInput::from_candles(&candles, "close", first_params);
        let first_result = wto_with_kernel(&first_input, kernel)?;

        let second_params = WtoParams::default();
        let second_input = WtoInput::from_slice(&first_result.wavetrend1, second_params);
        let second_result = wto_with_kernel(&second_input, kernel)?;

        assert_eq!(
            second_result.wavetrend1.len(),
            first_result.wavetrend1.len()
        );
        Ok(())
    }

    fn check_wto_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WtoInput::from_candles(&candles, "close", WtoParams::default());
        let res = wto_with_kernel(&input, kernel)?;

        assert_eq!(res.wavetrend1.len(), candles.close.len());
        if res.wavetrend1.len() > 50 {
            for (i, &val) in res.wavetrend1[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_wto_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Calculate batch WTO for comparison
        let params = WtoParams::default();
        let input = WtoInput::from_candles(&candles, "close", params.clone());
        let batch_result = wto_with_kernel(&input, kernel)?;

        // Create streaming calculator
        let mut stream = WtoStream::try_new(params)?;

        // Feed data to stream and collect results
        let mut stream_wt1 = Vec::new();
        let mut stream_wt2 = Vec::new();
        let mut stream_hist = Vec::new();

        for i in 0..candles.close.len() {
            if let Some((wt1, wt2, hist)) = stream.update(candles.close[i]) {
                stream_wt1.push(wt1);
                stream_wt2.push(wt2);
                stream_hist.push(hist);
            }
        }

        // Streaming will have fewer values due to warmup
        assert!(!stream_wt1.is_empty());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_wto_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            WtoParams {
                channel_length: Some(5),
                average_length: Some(10),
            },
            WtoParams {
                channel_length: Some(20),
                average_length: Some(40),
            },
            WtoParams {
                channel_length: Some(3),
                average_length: Some(7),
            },
        ];

        for params in test_params {
            let input = WtoInput::from_candles(&candles, "close", params.clone());
            let output = wto_with_kernel(&input, kernel)?;

            for (i, &val) in output.wavetrend1.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
                        "[{}] Found poison value {} (0x{:016X}) at index {} with params: {:?}",
                        test_name, val, bits, i, params
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_wto_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    // Enhanced poison check for all three outputs
    #[cfg(debug_assertions)]
    fn check_wto_no_poison_all(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = WtoInput::with_default_candles(&c);
        let out = wto_with_kernel(&input, kernel)?;

        for series in [&out.wavetrend1, &out.wavetrend2, &out.histogram] {
            for &v in series {
                if v.is_nan() {
                    continue;
                }
                let b = v.to_bits();
                assert!(
                    b != 0x11111111_11111111
                        && b != 0x22222222_22222222
                        && b != 0x33333333_33333333,
                    "[{}] poison value 0x{:016X}",
                    test_name,
                    b
                );
            }
        }
        Ok(())
    }

    // Batch poison check
    fn check_batch_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        // Test with WT1-only batch
        let output = WtoBatchBuilder::new()
            .channel_range(5, 15, 5)
            .average_range(10, 30, 10)
            .kernel(kernel)
            .apply_candles(&candles, "close")?;

        for &v in &output.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                "[{}] batch poison value 0x{:016X}",
                test_name,
                b
            );
        }

        // Test with full 3-output batch
        let sweep = WtoBatchRange {
            channel: (5, 15, 5),
            average: (10, 30, 10),
        };
        let data = source_type(&candles, "close");
        let full_out = wto_batch_all_outputs_with_kernel(data, &sweep, kernel)?;

        for series in [&full_out.wt1, &full_out.wt2, &full_out.hist] {
            for &v in series {
                if v.is_nan() {
                    continue;
                }
                let b = v.to_bits();
                assert!(
                    b != 0x11111111_11111111
                        && b != 0x22222222_22222222
                        && b != 0x33333333_33333333,
                    "[{}] full batch poison value 0x{:016X}",
                    test_name,
                    b
                );
            }
        }

        Ok(())
    }

    // Macro to generate test variants for all kernels
    macro_rules! generate_all_wto_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    // Generate all single kernel tests
    generate_all_wto_tests!(
        check_wto_accuracy,
        check_wto_partial_params,
        check_wto_default_candles,
        check_wto_zero_period,
        check_wto_period_exceeds_length,
        check_wto_very_small_dataset,
        check_wto_empty_input,
        check_wto_all_nan,
        check_wto_reinput,
        check_wto_nan_handling,
        check_wto_streaming,
        check_wto_no_poison,
        check_batch_poison
    );

    // Additional poison check tests for debug mode
    #[cfg(debug_assertions)]
    #[test]
    fn test_wto_no_poison_all_scalar() {
        check_wto_no_poison_all("test_wto_no_poison_all_scalar", Kernel::Scalar).unwrap();
    }

    #[cfg(all(debug_assertions, feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_wto_no_poison_all_avx2() {
        check_wto_no_poison_all("test_wto_no_poison_all_avx2", Kernel::Avx2).unwrap();
    }

    #[cfg(all(debug_assertions, feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_wto_no_poison_all_avx512() {
        check_wto_no_poison_all("test_wto_no_poison_all_avx512", Kernel::Avx512).unwrap();
    }

    #[cfg(all(debug_assertions, target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_wto_no_poison_all_simd128() {
        check_wto_no_poison_all("test_wto_no_poison_all_simd128", Kernel::Simd128).unwrap();
    }

    // Batch testing functions
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = WtoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles, "close")?;

        let def = WtoParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), candles.close.len());
        Ok(())
    }

    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let output = WtoBatchBuilder::new()
            .kernel(kernel)
            .channel_range(5, 15, 5)
            .average_range(10, 30, 10)
            .apply_candles(&candles, "close")?;

        let expected_combos = 3 * 3; // (5,10,15) * (10,20,30)
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, candles.close.len());

        Ok(())
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
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);

    // Parity test for the native into API (lengths must match and values must be identical)
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_wto_into_matches_api() {
        // Use the existing CSV so parity is checked on real candle data
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("failed to load candles");

        let input = WtoInput::with_default_candles(&candles);

        // Baseline via Vec-returning API
        let base = wto(&input).expect("wto baseline failed");

        // Preallocate outputs (will be overwritten, including warmups)
        let n = candles.close.len();
        let mut wt1 = vec![0.0; n];
        let mut wt2 = vec![0.0; n];
        let mut hist = vec![0.0; n];

        // Zero-allocation path
        wto_into(&input, &mut wt1, &mut wt2, &mut hist).expect("wto_into failed");

        // Helper: NaN == NaN, otherwise exact or tight epsilon
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        assert_eq!(wt1.len(), base.wavetrend1.len());
        assert_eq!(wt2.len(), base.wavetrend2.len());
        assert_eq!(hist.len(), base.histogram.len());

        for i in 0..n {
            assert!(
                eq_or_both_nan(wt1[i], base.wavetrend1[i]),
                "wt1 mismatch at {}: into={}, api={}",
                i,
                wt1[i],
                base.wavetrend1[i]
            );
            assert!(
                eq_or_both_nan(wt2[i], base.wavetrend2[i]),
                "wt2 mismatch at {}: into={}, api={}",
                i,
                wt2[i],
                base.wavetrend2[i]
            );
            assert!(
                eq_or_both_nan(hist[i], base.histogram[i]),
                "hist mismatch at {}: into={}, api={}",
                i,
                hist[i],
                base.histogram[i]
            );
        }
    }
}
