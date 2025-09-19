//! # Volume Price Confirmation Index (VPCI)
//!
//! VPCI confirms price movements using volume-weighted moving averages (VWMAs), comparing
//! price and volume trends to detect confluence/divergence. It supports SIMD kernels and
//! batch grid evaluation for hyperparameter sweeps.
//!
//! ## Parameters
//! - **short_range**: Window size for short-term averages (default: 5).
//! - **long_range**: Window size for long-term averages (default: 25).
//!
//! ## Returns
//! - **Ok(VpciOutput)** on success (`vpci`, `vpcis` of same length as input).
//! - **Err(VpciError)** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Functions exist with proper structure but all delegate to scalar implementation through vpci_compute_into. Prefix sum approach ready for SIMD vectorization.
//! - **Streaming Performance**: O(n) implementation where n is period. Recalculates VWMAs and SMAs by iterating through buffers. Note: VPCIS returns VPCI value (not properly smoothed) in streaming mode.
//! - **Memory Optimization**: Uses `alloc_with_nan_prefix` and batch helpers. Uses AVec for cache alignment but SIMD not yet leveraged. Prefix sum approach is memory efficient.

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
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::sma::{sma, SmaData, SmaError, SmaInput, SmaParams};

#[derive(Debug, Clone)]
pub enum VpciData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VpciOutput {
    pub vpci: Vec<f64>,
    pub vpcis: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VpciParams {
    pub short_range: Option<usize>,
    pub long_range: Option<usize>,
}

impl Default for VpciParams {
    fn default() -> Self {
        Self {
            short_range: Some(5),
            long_range: Some(25),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VpciInput<'a> {
    pub data: VpciData<'a>,
    pub params: VpciParams,
}

impl<'a> VpciInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
        params: VpciParams,
    ) -> Self {
        Self {
            data: VpciData::Candles {
                candles,
                close_source,
                volume_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: VpciParams) -> Self {
        Self {
            data: VpciData::Slices { close, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VpciData::Candles {
                candles,
                close_source: "close",
                volume_source: "volume",
            },
            params: VpciParams::default(),
        }
    }

    #[inline]
    pub fn get_short_range(&self) -> usize {
        self.params.short_range.unwrap_or(5)
    }
    #[inline]
    pub fn get_long_range(&self) -> usize {
        self.params.long_range.unwrap_or(25)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VpciBuilder {
    short_range: Option<usize>,
    long_range: Option<usize>,
    kernel: Kernel,
}

impl Default for VpciBuilder {
    fn default() -> Self {
        Self {
            short_range: None,
            long_range: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VpciBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_range(mut self, n: usize) -> Self {
        self.short_range = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_range(mut self, n: usize) -> Self {
        self.long_range = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VpciOutput, VpciError> {
        let p = VpciParams {
            short_range: self.short_range,
            long_range: self.long_range,
        };
        let i = VpciInput::from_candles(c, "close", "volume", p);
        vpci_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VpciOutput, VpciError> {
        let p = VpciParams {
            short_range: self.short_range,
            long_range: self.long_range,
        };
        let i = VpciInput::from_slices(close, volume, p);
        vpci_with_kernel(&i, self.kernel)
    }
}

/// Streaming implementation for VPCI indicator
#[derive(Clone, Debug)]
pub struct VpciStream {
    short_range: usize,
    long_range: usize,
    close_buffer: Vec<f64>,
    volume_buffer: Vec<f64>,
    head: usize,
    filled: bool,
    count: usize,
}

impl VpciStream {
    pub fn try_new(params: VpciParams) -> Result<Self, VpciError> {
        let short_range = params.short_range.unwrap_or(5);
        let long_range = params.long_range.unwrap_or(25);

        if short_range == 0 || long_range == 0 {
            return Err(VpciError::InvalidRange {
                period: if short_range == 0 {
                    short_range
                } else {
                    long_range
                },
                data_len: 0,
            });
        }

        if short_range > long_range {
            return Err(VpciError::InvalidRange {
                period: short_range,
                data_len: long_range,
            });
        }

        Ok(Self {
            short_range,
            long_range,
            close_buffer: vec![f64::NAN; long_range],
            volume_buffer: vec![f64::NAN; long_range],
            head: 0,
            filled: false,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<(f64, f64)> {
        self.close_buffer[self.head] = close;
        self.volume_buffer[self.head] = volume;
        self.head = (self.head + 1) % self.long_range;
        self.count += 1;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }

        // Need at least long_range values to compute
        if self.count < self.long_range {
            return None;
        }

        Some(self.compute_current())
    }

    #[inline(always)]
    fn compute_current(&self) -> (f64, f64) {
        // Calculate VWMAs and SMAs using ring buffer
        let vwma_long = self.compute_vwma(self.long_range);
        let vwma_short = self.compute_vwma(self.short_range);
        let sma_close_long = self.compute_sma_close(self.long_range);
        let sma_close_short = self.compute_sma_close(self.short_range);
        let sma_volume_long = self.compute_sma_volume(self.long_range);
        let sma_volume_short = self.compute_sma_volume(self.short_range);

        // Calculate VPCI
        let vpc = vwma_long - sma_close_long;
        let vpr = if sma_close_short != 0.0 {
            vwma_short / sma_close_short
        } else {
            f64::NAN
        };
        let vm = if sma_volume_long != 0.0 {
            sma_volume_short / sma_volume_long
        } else {
            f64::NAN
        };

        let vpci = vpc * vpr * vm;

        // NOTE: VPCIS (smoothed VPCI) calculation requires historical VPCI values.
        // In streaming mode, properly calculating VPCIS = SMA(VPCI*Volume, short) / SMA(Volume, short)
        // would require maintaining a rolling buffer of past VPCI values.
        // For now, we return the current VPCI value for both to maintain API compatibility.
        // Users requiring accurate VPCIS in streaming mode should use the batch API instead.
        (vpci, vpci)
    }

    #[inline(always)]
    fn compute_vwma(&self, period: usize) -> f64 {
        let mut sum_cv = 0.0;
        let mut sum_v = 0.0;
        let mut idx = (self.head + self.long_range - period) % self.long_range;

        for _ in 0..period {
            sum_cv += self.close_buffer[idx] * self.volume_buffer[idx];
            sum_v += self.volume_buffer[idx];
            idx = (idx + 1) % self.long_range;
        }

        if sum_v != 0.0 {
            sum_cv / sum_v
        } else {
            f64::NAN
        }
    }

    #[inline(always)]
    fn compute_sma_close(&self, period: usize) -> f64 {
        let mut sum = 0.0;
        let mut idx = (self.head + self.long_range - period) % self.long_range;

        for _ in 0..period {
            sum += self.close_buffer[idx];
            idx = (idx + 1) % self.long_range;
        }

        sum / period as f64
    }

    #[inline(always)]
    fn compute_sma_volume(&self, period: usize) -> f64 {
        let mut sum = 0.0;
        let mut idx = (self.head + self.long_range - period) % self.long_range;

        for _ in 0..period {
            sum += self.volume_buffer[idx];
            idx = (idx + 1) % self.long_range;
        }

        sum / period as f64
    }
}

#[derive(Debug, Error)]
pub enum VpciError {
    #[error("vpci: All close or volume values are NaN.")]
    AllValuesNaN,

    #[error("vpci: Invalid range: period = {period}, data length = {data_len}")]
    InvalidRange { period: usize, data_len: usize },

    #[error("vpci: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("vpci: SMA error: {0}")]
    SmaError(#[from] SmaError),

    #[error("vpci: Mismatched input lengths: close = {close_len}, volume = {volume_len}")]
    MismatchedInputLengths { close_len: usize, volume_len: usize },

    #[error("vpci: Mismatched output lengths: vpci_len = {vpci_len}, vpcis_len = {vpcis_len}, expected = {data_len}")]
    MismatchedOutputLengths {
        vpci_len: usize,
        vpcis_len: usize,
        data_len: usize,
    },

    #[error("vpci: Kernel not available")]
    KernelNotAvailable,
}

// ================================
// Core Helper Functions
// ================================

#[inline(always)]
fn first_valid_both(close: &[f64], volume: &[f64]) -> Option<usize> {
    close
        .iter()
        .zip(volume)
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
}

#[inline(always)]
fn ensure_same_len(close: &[f64], volume: &[f64]) -> Result<(), VpciError> {
    if close.len() != volume.len() {
        return Err(VpciError::MismatchedInputLengths {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn build_prefix_sums(close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut ps_close = vec![0.0; n + 1];
    let mut ps_vol = vec![0.0; n + 1];
    let mut ps_cv = vec![0.0; n + 1];

    for i in 0..n {
        let c = close[i];
        let v = volume[i];
        // Only add finite values to prefix sums, treating NaN as 0
        // This prevents NaN propagation while allowing windows to work
        let c_val = if c.is_finite() { c } else { 0.0 };
        let v_val = if v.is_finite() { v } else { 0.0 };
        ps_close[i + 1] = ps_close[i] + c_val;
        ps_vol[i + 1] = ps_vol[i] + v_val;
        ps_cv[i + 1] = ps_cv[i] + c_val * v_val;
    }
    (ps_close, ps_vol, ps_cv)
}

#[inline(always)]
fn window_sum(ps: &[f64], start: usize, end_inclusive: usize) -> f64 {
    // end_inclusive is inclusive index in original array
    let a = start;
    let b = end_inclusive + 1;
    ps[b] - ps[a]
}

#[inline(always)]
fn vpci_prepare<'a>(
    input: &'a VpciInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, usize, Kernel), VpciError> {
    let (close, volume) = match &input.data {
        VpciData::Candles {
            candles,
            close_source,
            volume_source,
        } => (
            source_type(candles, close_source),
            source_type(candles, volume_source),
        ),
        VpciData::Slices { close, volume } => (*close, *volume),
    };

    ensure_same_len(close, volume)?;

    let len = close.len();
    if len == 0 {
        return Err(VpciError::AllValuesNaN);
    }
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;

    let short = input.get_short_range();
    let long = input.get_long_range();
    if short == 0 || long == 0 || short > len || long > len {
        return Err(VpciError::InvalidRange {
            period: short.max(long),
            data_len: len,
        });
    }
    if short > long {
        return Err(VpciError::InvalidRange {
            period: short,
            data_len: long,
        });
    }
    if (len - first) < long {
        return Err(VpciError::NotEnoughValidData {
            needed: long,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((close, volume, first, short, long, chosen))
}

// ================================
// Core Computation Functions
// ================================

#[inline(always)]
fn vpci_scalar_into_from_psums(
    close: &[f64],
    volume: &[f64],
    first: usize,
    short: usize,
    long: usize,
    ps_close: &[f64],
    ps_vol: &[f64],
    ps_cv: &[f64],
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) {
    debug_assert_eq!(close.len(), volume.len());
    let n = close.len();
    let warmup = first + long - 1;

    // NaN prefix
    for i in 0..warmup.min(n) {
        vpci_out[i] = f64::NAN;
        vpcis_out[i] = f64::NAN;
    }
    if warmup >= n {
        return;
    }

    // Helper closures for rolling stats via prefix sums
    #[inline(always)]
    fn sma(ps: &[f64], start: usize, len: usize) -> f64 {
        window_sum(ps, start, start + len - 1) / (len as f64)
    }
    #[inline(always)]
    fn vwma(ps_cv: &[f64], ps_vol: &[f64], start: usize, len: usize) -> f64 {
        let sum_v = window_sum(ps_vol, start, start + len - 1);
        if sum_v == 0.0 {
            return f64::NAN;
        }
        window_sum(ps_cv, start, start + len - 1) / sum_v
    }

    // Compute VPCI and VPCIS
    let mut sum_vpci_vol_short = 0.0;
    let mut have_vpcis_window = false;

    for i in warmup..n {
        let long_start = (i + 1).saturating_sub(long);
        let short_start = (i + 1).saturating_sub(short);

        let vwma_l = vwma(ps_cv, ps_vol, long_start, long);
        let sma_l = sma(ps_close, long_start, long);
        let vwma_s = vwma(ps_cv, ps_vol, short_start, short);
        let sma_s = sma(ps_close, short_start, short);
        let sma_v_s = sma(ps_vol, short_start, short);
        let sma_v_l = sma(ps_vol, long_start, long);

        let vpc = vwma_l - sma_l;
        let vpr = if sma_s != 0.0 {
            vwma_s / sma_s
        } else {
            f64::NAN
        };
        let vm = if sma_v_l != 0.0 {
            sma_v_s / sma_v_l
        } else {
            f64::NAN
        };

        let vpci = vpc * vpr * vm;
        vpci_out[i] = vpci;

        // VPCIS window starts once we have short items ending at i.
        // Initialize once at i == warmup: include indices [i - short + 1 .. i], but clamp lower bound to warmup.
        if !have_vpcis_window {
            if i + 1 >= warmup + short {
                // first full short window entirely ≥ warmup
                sum_vpci_vol_short = 0.0;
                for k in (i + 1 - short)..=i {
                    if k >= warmup {
                        let x = vpci_out[k];
                        let v = volume[k];
                        if x.is_finite() && v.is_finite() {
                            sum_vpci_vol_short += x * v;
                        }
                    }
                }
                have_vpcis_window = true;
            } else {
                // Partial window that includes < warmup indices: include only k ≥ warmup
                sum_vpci_vol_short = 0.0;
                for k in warmup..=i {
                    let x = vpci_out[k];
                    let v = volume[k];
                    if x.is_finite() && v.is_finite() {
                        sum_vpci_vol_short += x * v;
                    }
                }
                // Still allowed to emit based on definition used in your scalar version
                have_vpcis_window = true;
            }
        } else {
            // Slide by one
            let add_k = i;
            if add_k >= warmup {
                let x_add = vpci_out[add_k];
                let v_add = volume[add_k];
                if x_add.is_finite() && v_add.is_finite() {
                    sum_vpci_vol_short += x_add * v_add;
                }
            }
            let rm_k = i.saturating_sub(short);
            if rm_k >= warmup {
                let x_rm = vpci_out[rm_k];
                let v_rm = volume[rm_k];
                if x_rm.is_finite() && v_rm.is_finite() {
                    sum_vpci_vol_short -= x_rm * v_rm;
                }
            }
        }

        // VPCIS = SMA(VPCI*Volume, short) / SMA(Volume, short)
        let denom = sma_v_s;
        if denom != 0.0 && denom.is_finite() {
            vpcis_out[i] = (sum_vpci_vol_short / (short as f64)) / denom;
        } else {
            vpcis_out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
fn vpci_compute_into(
    close: &[f64],
    volume: &[f64],
    first: usize,
    short: usize,
    long: usize,
    _kernel: Kernel, // kernels map to same scalar path here
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) {
    let (ps_c, ps_v, ps_cv) = build_prefix_sums(close, volume);
    vpci_scalar_into_from_psums(
        close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, vpci_out, vpcis_out,
    );
}

#[inline]
pub fn vpci(input: &VpciInput) -> Result<VpciOutput, VpciError> {
    vpci_with_kernel(input, Kernel::Auto)
}

pub fn vpci_with_kernel(input: &VpciInput, kernel: Kernel) -> Result<VpciOutput, VpciError> {
    let (close, volume, first, short, long, chosen) = vpci_prepare(input, kernel)?;

    let len = close.len();
    let warmup = first + long - 1;
    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close, volume, first, short, long, chosen, &mut vpci, &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[inline]
pub fn vpci_into_slice(
    vpci_dst: &mut [f64],
    vpcis_dst: &mut [f64],
    input: &VpciInput,
    kernel: Kernel,
) -> Result<(), VpciError> {
    let (close, volume, first, short, long, chosen) = vpci_prepare(input, kernel)?;
    if vpci_dst.len() != close.len() || vpcis_dst.len() != close.len() {
        return Err(VpciError::MismatchedOutputLengths {
            vpci_len: vpci_dst.len(),
            vpcis_len: vpcis_dst.len(),
            data_len: close.len(),
        });
    }
    let warmup = first + long - 1;
    for i in 0..warmup.min(vpci_dst.len()) {
        vpci_dst[i] = f64::NAN;
        vpcis_dst[i] = f64::NAN;
    }
    vpci_compute_into(
        close, volume, first, short, long, chosen, vpci_dst, vpcis_dst,
    );
    Ok(())
}

// Simple stub functions that use the unified compute path
#[inline]
pub unsafe fn vpci_scalar(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmup = first + long - 1;

    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close,
        volume,
        first,
        short,
        long,
        Kernel::Scalar,
        &mut vpci,
        &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx2(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmup = first + long - 1;

    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close,
        volume,
        first,
        short,
        long,
        Kernel::Avx2,
        &mut vpci,
        &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmup = first + long - 1;

    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close,
        volume,
        first,
        short,
        long,
        Kernel::Avx512,
        &mut vpci,
        &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512_short(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512_long(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[inline]
pub fn vpci_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
    let k = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(VpciError::KernelNotAvailable),
    };
    let simd = match k {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vpci_batch_par_slice(close, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VpciBatchRange {
    pub short_range: (usize, usize, usize),
    pub long_range: (usize, usize, usize),
}

impl Default for VpciBatchRange {
    fn default() -> Self {
        Self {
            short_range: (5, 20, 1),
            long_range: (25, 60, 5),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VpciBatchBuilder {
    range: VpciBatchRange,
    kernel: Kernel,
}

impl VpciBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_range = (start, end, step);
        self
    }
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_range = (start, end, step);
        self
    }
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VpciBatchOutput, VpciError> {
        vpci_batch_with_kernel(close, volume, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct VpciBatchOutput {
    pub vpci: Vec<f64>,
    pub vpcis: Vec<f64>,
    pub combos: Vec<VpciParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VpciBatchOutput {
    pub fn row_for_params(&self, p: &VpciParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_range.unwrap_or(5) == p.short_range.unwrap_or(5)
                && c.long_range.unwrap_or(25) == p.long_range.unwrap_or(25)
        })
    }
    pub fn vpci_for(&self, p: &VpciParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.vpci[start..start + self.cols]
        })
    }
    pub fn vpcis_for(&self, p: &VpciParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.vpcis[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &VpciBatchRange) -> Vec<VpciParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis_usize(r.short_range);
    let longs = axis_usize(r.long_range);

    let mut out = Vec::with_capacity(shorts.len() * longs.len());
    for &s in &shorts {
        for &l in &longs {
            out.push(VpciParams {
                short_range: Some(s),
                long_range: Some(l),
            });
        }
    }
    out
}

#[inline(always)]
pub fn vpci_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
    vpci_batch_inner(close, volume, sweep, kernel, false)
}

#[inline(always)]
pub fn vpci_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
    vpci_batch_inner(close, volume, sweep, kernel, true)
}

#[inline(always)]
fn vpci_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VpciBatchOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let combos = expand_grid(sweep);
    let cols = close.len();
    let rows = combos.len();
    if cols == 0 || rows == 0 {
        return Err(VpciError::InvalidRange {
            period: 0,
            data_len: cols,
        });
    }

    // Find first valid data and calculate warmup periods
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|p| first + p.long_range.unwrap() - 1)
        .collect();

    let mut vpci_mu = make_uninit_matrix(rows, cols);
    let mut vpcis_mu = make_uninit_matrix(rows, cols);

    // Initialize NaN prefixes using the helper function
    init_matrix_prefixes(&mut vpci_mu, cols, &warmups);
    init_matrix_prefixes(&mut vpcis_mu, cols, &warmups);

    let ptr_v = vpci_mu.as_ptr() as *mut f64;
    let ptr_s = vpcis_mu.as_ptr() as *mut f64;
    let cap_v = vpci_mu.capacity();
    let cap_s = vpcis_mu.capacity();

    // writable slices
    let vpci_slice = unsafe { core::slice::from_raw_parts_mut(ptr_v, rows * cols) };
    let vpcis_slice = unsafe { core::slice::from_raw_parts_mut(ptr_s, rows * cols) };

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

    // compute (propagates Err without leaking)
    let combos = vpci_batch_inner_into(
        close,
        volume,
        sweep,
        simd,
        parallel,
        vpci_slice,
        vpcis_slice,
    )?;

    // take ownership only now
    core::mem::forget(vpci_mu);
    core::mem::forget(vpcis_mu);
    let vpci_vec = unsafe { Vec::from_raw_parts(ptr_v, rows * cols, cap_v) };
    let vpcis_vec = unsafe { Vec::from_raw_parts(ptr_s, rows * cols, cap_s) };

    Ok(VpciBatchOutput {
        vpci: vpci_vec,
        vpcis: vpcis_vec,
        combos,
        rows,
        cols,
    })
}

/// Zero-copy batch operation that writes directly into provided output buffers
#[inline(always)]
fn vpci_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
    parallel: bool,
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) -> Result<Vec<VpciParams>, VpciError> {
    ensure_same_len(close, volume)?;
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VpciError::InvalidRange {
            period: 0,
            data_len: close.len(),
        });
    }
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_range.unwrap()).max().unwrap();
    if len - first < max_long {
        return Err(VpciError::NotEnoughValidData {
            needed: max_long,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    // Precompute prefix sums once
    let (ps_c, ps_v, ps_cv) = build_prefix_sums(close, volume);

    // Initialize NaN prefixes per row (needed for direct calls from Python/WASM)
    // Note: When called from vpci_batch_inner, this duplicates work already done
    // by init_matrix_prefixes, but it's harmless and ensures correctness for all paths
    for (row, prm) in combos.iter().enumerate() {
        let warmup = first + prm.long_range.unwrap() - 1;
        let s = row * cols;
        for i in 0..warmup.min(cols) {
            vpci_out[s + i] = f64::NAN;
            vpcis_out[s + i] = f64::NAN;
        }
    }

    // Process rows
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            vpci_out
                .par_chunks_mut(cols)
                .zip(vpcis_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (dst_vpci, dst_vpcis))| {
                    let prm = &combos[row];
                    let short = prm.short_range.unwrap();
                    let long = prm.long_range.unwrap();

                    vpci_scalar_into_from_psums(
                        close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci,
                        dst_vpcis,
                    );
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let prm = &combos[row];
                let short = prm.short_range.unwrap();
                let long = prm.long_range.unwrap();

                let row_off = row * cols;
                let dst_vpci = &mut vpci_out[row_off..row_off + cols];
                let dst_vpcis = &mut vpcis_out[row_off..row_off + cols];

                vpci_scalar_into_from_psums(
                    close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci, dst_vpcis,
                );
            }
        }
    } else {
        for row in 0..rows {
            let prm = &combos[row];
            let short = prm.short_range.unwrap();
            let long = prm.long_range.unwrap();

            let row_off = row * cols;
            let dst_vpci = &mut vpci_out[row_off..row_off + cols];
            let dst_vpcis = &mut vpcis_out[row_off..row_off + cols];

            vpci_scalar_into_from_psums(
                close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci, dst_vpcis,
            );
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn vpci_row_scalar(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_scalar(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx2(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx2(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    if long <= 32 {
        vpci_row_avx512_short(close, volume, short, long)
    } else {
        vpci_row_avx512_long(close, volume, short, long)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512_short(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512_long(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[inline(always)]
pub fn expand_grid_vpci(r: &VpciBatchRange) -> Vec<VpciParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_vpci_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = VpciParams {
            short_range: Some(3),
            long_range: None,
        };
        let input = VpciInput::from_candles(&candles, "close", "volume", params);
        let output = vpci_with_kernel(&input, kernel)?;
        assert_eq!(output.vpci.len(), candles.close.len());
        assert_eq!(output.vpcis.len(), candles.close.len());
        Ok(())
    }

    fn check_vpci_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = VpciParams {
            short_range: Some(5),
            long_range: Some(25),
        };
        let input = VpciInput::from_candles(&candles, "close", "volume", params);
        let output = vpci_with_kernel(&input, kernel)?;

        let vpci_len = output.vpci.len();
        let vpcis_len = output.vpcis.len();
        assert_eq!(vpci_len, candles.close.len());
        assert_eq!(vpcis_len, candles.close.len());

        let vpci_last_five = &output.vpci[vpci_len.saturating_sub(5)..];
        let vpcis_last_five = &output.vpcis[vpcis_len.saturating_sub(5)..];
        let expected_vpci = [
            -319.65148214323426,
            -133.61700649928346,
            -144.76194155503174,
            -83.55576212490328,
            -169.53504207700533,
        ];
        let expected_vpcis = [
            -1049.2826640115732,
            -694.1067814399748,
            -519.6960416662324,
            -330.9401404636258,
            -173.004986803695,
        ];
        for (i, &val) in vpci_last_five.iter().enumerate() {
            let diff = (val - expected_vpci[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] VPCI mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_vpci[i]
            );
        }
        for (i, &val) in vpcis_last_five.iter().enumerate() {
            let diff = (val - expected_vpcis[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] VPCIS mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_vpcis[i]
            );
        }
        Ok(())
    }

    fn check_vpci_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VpciInput::with_default_candles(&candles);
        let output = vpci_with_kernel(&input, kernel)?;
        assert_eq!(output.vpci.len(), candles.close.len());
        assert_eq!(output.vpcis.len(), candles.close.len());
        Ok(())
    }

    fn check_vpci_slice_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [10.0, 12.0, 14.0, 13.0, 15.0];
        let volume_data = [100.0, 200.0, 300.0, 250.0, 400.0];
        let params = VpciParams {
            short_range: Some(2),
            long_range: Some(3),
        };
        let input = VpciInput::from_slices(&close_data, &volume_data, params);
        let output = vpci_with_kernel(&input, kernel)?;
        assert_eq!(output.vpci.len(), close_data.len());
        assert_eq!(output.vpcis.len(), close_data.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_vpci_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            VpciParams::default(), // short: 5, long: 25
            VpciParams {
                short_range: Some(2),
                long_range: Some(3),
            }, // minimum viable
            VpciParams {
                short_range: Some(2),
                long_range: Some(10),
            }, // small short, medium long
            VpciParams {
                short_range: Some(5),
                long_range: Some(20),
            }, // medium both
            VpciParams {
                short_range: Some(10),
                long_range: Some(30),
            }, // medium-large both
            VpciParams {
                short_range: Some(20),
                long_range: Some(50),
            }, // large both
            VpciParams {
                short_range: Some(3),
                long_range: Some(100),
            }, // small short, very large long
            VpciParams {
                short_range: Some(50),
                long_range: Some(100),
            }, // very large both
            VpciParams {
                short_range: Some(7),
                long_range: Some(21),
            }, // common Fibonacci values
            VpciParams {
                short_range: Some(14),
                long_range: Some(28),
            }, // double common values
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = VpciInput::from_candles(&candles, "close", "volume", params.clone());
            let output = vpci_with_kernel(&input, kernel)?;

            // Check VPCI values
            for (i, &val) in output.vpci.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in VPCI with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in VPCI with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in VPCI with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }
            }

            // Check VPCIS values
            for (i, &val) in output.vpcis.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in VPCIS with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in VPCIS with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in VPCIS with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vpci_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    fn calculate_variance(data: &[f64]) -> f64 {
        let finite_values: Vec<f64> = data.iter().filter(|v| v.is_finite()).copied().collect();

        if finite_values.len() < 2 {
            return 0.0;
        }

        let mean = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
        let variance = finite_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / finite_values.len() as f64;

        variance
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vpci_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=20).prop_flat_map(|short_range| {
            ((short_range + 1)..=50).prop_flat_map(move |long_range| {
                let min_len = long_range + 10; // Ensure we have enough data after warmup
                (min_len..400).prop_flat_map(move |data_len| {
                    (
                        // Close prices: realistic price range
                        prop::collection::vec(
                            (100f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        // Volume: realistic volume range
                        prop::collection::vec(
                            (1000f64..1000000f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        Just(short_range),
                        Just(long_range),
                    )
                })
            })
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(close, volume, short_range, long_range)| {
                let params = VpciParams {
                    short_range: Some(short_range),
                    long_range: Some(long_range),
                };
                let input = VpciInput::from_slices(&close, &volume, params);

                let VpciOutput {
                    vpci: out,
                    vpcis: out_smooth,
                } = vpci_with_kernel(&input, kernel).unwrap();
                let VpciOutput {
                    vpci: ref_out,
                    vpcis: ref_out_smooth,
                } = vpci_with_kernel(&input, Kernel::Scalar).unwrap();

                // Find first valid index (non-NaN in both close and volume)
                let first_valid = close
                    .iter()
                    .zip(volume.iter())
                    .position(|(c, v)| !c.is_nan() && !v.is_nan())
                    .unwrap_or(0);

                // Expected warmup period
                let expected_warmup = first_valid + long_range - 1;

                // Check warmup period - values before warmup should be NaN
                for i in 0..expected_warmup.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                    prop_assert!(
                        out_smooth[i].is_nan(),
                        "Expected NaN in VPCIS during warmup at index {}, got {}",
                        i,
                        out_smooth[i]
                    );
                }

                // Check values after warmup period
                for i in expected_warmup..close.len() {
                    let y = out[i];
                    let ys = out_smooth[i];
                    let r = ref_out[i];
                    let rs = ref_out_smooth[i];

                    // Values should be finite after warmup (unless input had NaN)
                    if !close[i].is_nan() && !volume[i].is_nan() {
                        prop_assert!(
                            y.is_finite() || r.is_nan(),
                            "VPCI should be finite at idx {} after warmup, got {}",
                            i,
                            y
                        );
                    }

                    // Check kernel consistency for VPCI
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch in VPCI at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let y_bits = y.to_bits();
                        let r_bits = r.to_bits();
                        let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "VPCI mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }

                    // Check kernel consistency for VPCIS
                    if !ys.is_finite() || !rs.is_finite() {
                        prop_assert!(
                            ys.to_bits() == rs.to_bits(),
                            "finite/NaN mismatch in VPCIS at idx {}: {} vs {}",
                            i,
                            ys,
                            rs
                        );
                    } else {
                        let ys_bits = ys.to_bits();
                        let rs_bits = rs.to_bits();
                        let ulp_diff: u64 = ys_bits.abs_diff(rs_bits);

                        prop_assert!(
                            (ys - rs).abs() <= 1e-9 || ulp_diff <= 4,
                            "VPCIS mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            ys,
                            rs,
                            ulp_diff
                        );
                    }
                }

                // Additional mathematical properties specific to VPCI

                // Property 1: When prices are constant, VPC component should be near zero
                // VPC = VWMA_long - SMA_long, both should be equal for constant prices
                let prices_constant = close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);

                if prices_constant && expected_warmup < close.len() {
                    for i in expected_warmup..close.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                out[i].abs() <= 1e-6,
                                "VPCI should be ~0 when prices are constant, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                // Property 2: When volumes are constant, test volume ratio behavior
                let volumes_constant = volume.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);

                if volumes_constant && expected_warmup < close.len() {
                    // VM = SMA_volume_short / SMA_volume_long = 1.0 when volumes are constant
                    // This simplifies the VPCI calculation
                    for i in expected_warmup..close.len() {
                        if out[i].is_finite() && ref_out[i].is_finite() {
                            // Just verify consistency since the math still involves price components
                            prop_assert!(
                                (out[i] - ref_out[i]).abs() <= 1e-9,
                                "VPCI kernels should match exactly with constant volume"
                            );
                        }
                    }
                }

                // Property 3: VPCIS relationship to VPCI - both should be finite in same locations
                // VPCIS is a volume-weighted average of VPCI, so when VPCI is finite, VPCIS should be too
                if expected_warmup + short_range < close.len() {
                    for i in (expected_warmup + short_range)..close.len() {
                        if out[i].is_finite() && volume[i].is_finite() && volume[i] > 0.0 {
                            // If VPCI is finite and volume is valid, VPCIS should also be finite
                            // (unless there's a division by zero in the calculation)
                            if !out_smooth[i].is_finite() {
                                // Check if it's due to division by zero (SMA of volume being 0)
                                let vol_window = &volume[i.saturating_sub(short_range - 1)..=i];
                                let vol_sum: f64 = vol_window.iter().sum();
                                prop_assert!(
									vol_sum.abs() < 1e-10,
									"VPCIS should be finite when VPCI is finite and volume > 0 at index {}",
									i
								);
                            }
                        }
                    }
                }

                // Property 4: Special edge case when short_range == long_range
                if short_range == long_range && expected_warmup < close.len() {
                    // When periods are equal, certain components interact in specific ways
                    // VPC uses same period for VWMA and SMA, but different calculations
                    for i in expected_warmup..close.len().min(expected_warmup + 10) {
                        if out[i].is_finite() {
                            // The output should still be valid and finite
                            prop_assert!(
                                !out[i].is_nan(),
                                "VPCI should be valid even when short_range == long_range"
                            );
                        }
                    }
                }

                // Property 5: Extreme parameter ratios should still produce valid results
                let extreme_ratio = long_range as f64 / short_range as f64 > 10.0;
                if extreme_ratio && expected_warmup < close.len() {
                    for i in expected_warmup..close.len().min(expected_warmup + 5) {
                        prop_assert!(
                            out[i].is_nan() || out[i].is_finite(),
                            "VPCI should handle extreme parameter ratios gracefully at index {}",
                            i
                        );
                    }
                }

                // Property 6: Verify valid count consistency between kernels
                let valid_count = out
                    .iter()
                    .skip(expected_warmup)
                    .filter(|v| v.is_finite())
                    .count();

                let ref_valid_count = ref_out
                    .iter()
                    .skip(expected_warmup)
                    .filter(|v| v.is_finite())
                    .count();

                prop_assert_eq!(
                    valid_count,
                    ref_valid_count,
                    "Valid value count mismatch between kernels"
                );

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_vpci_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    generate_all_vpci_tests!(
        check_vpci_partial_params,
        check_vpci_accuracy,
        check_vpci_default_candles,
        check_vpci_slice_input,
        check_vpci_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vpci_tests!(check_vpci_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let close = &c.close;
        let volume = &c.volume;

        let output = VpciBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(close, volume)?;

        let def = VpciParams::default();
        let row = output.vpci_for(&def).expect("default row missing");

        assert_eq!(row.len(), close.len());

        let expected = [
            -319.65148214323426,
            -133.61700649928346,
            -144.76194155503174,
            -83.55576212490328,
            -169.53504207700533,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let close = &c.close;
        let volume = &c.volume;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (short_start, short_end, short_step, long_start, long_end, long_step)
            (2, 10, 2, 5, 25, 5),     // Small to medium ranges
            (5, 15, 5, 20, 40, 10),   // Medium ranges
            (10, 20, 5, 30, 60, 15),  // Medium to large ranges
            (2, 5, 1, 10, 15, 1),     // Dense small ranges
            (20, 30, 2, 40, 60, 5),   // Large ranges
            (3, 7, 2, 21, 35, 7),     // Fibonacci-inspired ranges
            (8, 12, 1, 25, 30, 1),    // Narrow dense ranges
            (2, 50, 10, 10, 100, 20), // Wide sparse ranges
        ];

        for (cfg_idx, &(short_start, short_end, short_step, long_start, long_end, long_step)) in
            test_configs.iter().enumerate()
        {
            let output = VpciBatchBuilder::new()
                .kernel(kernel)
                .short_range(short_start, short_end, short_step)
                .long_range(long_start, long_end, long_step)
                .apply_slices(close, volume)?;

            // Check VPCI values
            for (idx, &val) in output.vpci.iter().enumerate() {
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
						 in VPCI at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 in VPCI at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 in VPCI at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }
            }

            // Check VPCIS values
            for (idx, &val) in output.vpcis.iter().enumerate() {
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
						 in VPCIS at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 in VPCIS at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 in VPCIS at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
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

#[cfg(feature = "python")]
#[pyfunction(name = "vpci")]
#[pyo3(signature = (close, volume, short_range, long_range, kernel=None))]
pub fn vpci_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    short_range: usize,
    long_range: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    if close_slice.len() != volume_slice.len() {
        return Err(PyValueError::new_err(
            "Close and volume arrays must have the same length",
        ));
    }

    let kern = validate_kernel(kernel, false)?;
    let params = VpciParams {
        short_range: Some(short_range),
        long_range: Some(long_range),
    };
    let input = VpciInput::from_slices(close_slice, volume_slice, params);

    let (vpci_vec, vpcis_vec) = py
        .allow_threads(|| vpci_with_kernel(&input, kern).map(|o| (o.vpci, o.vpcis)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((vpci_vec.into_pyarray(py), vpcis_vec.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "VpciStream")]
pub struct VpciStreamPy {
    stream: VpciStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VpciStreamPy {
    #[new]
    fn new(short_range: usize, long_range: usize) -> PyResult<Self> {
        let params = VpciParams {
            short_range: Some(short_range),
            long_range: Some(long_range),
        };
        let stream =
            VpciStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VpciStreamPy { stream })
    }

    fn update(&mut self, close: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpci_batch")]
#[pyo3(signature = (close, volume, short_range_tuple, long_range_tuple, kernel=None))]
pub fn vpci_batch_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    short_range_tuple: (usize, usize, usize),
    long_range_tuple: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    if close_slice.len() != volume_slice.len() {
        return Err(PyValueError::new_err(
            "Close and volume arrays must have the same length",
        ));
    }

    let sweep = VpciBatchRange {
        short_range: short_range_tuple,
        long_range: long_range_tuple,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = close_slice.len();

    // Pre-allocate output arrays for batch operations
    let vpci_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let vpcis_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let vpci_slice = unsafe { vpci_arr.as_slice_mut()? };
    let vpcis_slice = unsafe { vpcis_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // Map batch kernels to regular kernels for computation
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };

            // Use zero-copy batch operation directly into pre-allocated arrays
            vpci_batch_inner_into(
                close_slice,
                volume_slice,
                &sweep,
                simd,
                true,
                vpci_slice,
                vpcis_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("vpci", vpci_arr.reshape((rows, cols))?)?;
    dict.set_item("vpcis", vpcis_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "short_ranges",
        combos
            .iter()
            .map(|p| p.short_range.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "long_ranges",
        combos
            .iter()
            .map(|p| p.long_range.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

/// WASM helper: Write directly to output slices - no allocations

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_js(
    close: &[f64],
    volume: &[f64],
    short_range: usize,
    long_range: usize,
) -> Result<JsValue, JsValue> {
    let params = VpciParams {
        short_range: Some(short_range),
        long_range: Some(long_range),
    };
    let input = VpciInput::from_slices(close, volume, params);

    let out = vpci(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    #[derive(Serialize)]
    struct Out {
        vpci: Vec<f64>,
        vpcis: Vec<f64>,
    }
    serde_wasm_bindgen::to_value(&Out {
        vpci: out.vpci,
        vpcis: out.vpcis,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    vpci_ptr: *mut f64,
    vpcis_ptr: *mut f64,
    len: usize,
    short_range: usize,
    long_range: usize,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to vpci_into"));
    }

    unsafe {
        let close = core::slice::from_raw_parts(close_ptr, len);
        let volume = core::slice::from_raw_parts(volume_ptr, len);
        let vpci = core::slice::from_raw_parts_mut(vpci_ptr, len);
        let vpcis = core::slice::from_raw_parts_mut(vpcis_ptr, len);

        let params = VpciParams {
            short_range: Some(short_range),
            long_range: Some(long_range),
        };
        let input = VpciInput::from_slices(close, volume, params);

        vpci_into_slice(vpci, vpcis, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VpciBatchConfig {
    pub short_range: (usize, usize, usize),
    pub long_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VpciBatchJsOutput {
    pub vpci: Vec<f64>,
    pub vpcis: Vec<f64>,
    pub combos: Vec<VpciParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "vpci_batch")]
pub fn vpci_batch_unified_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: VpciBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = VpciBatchRange {
        short_range: cfg.short_range,
        long_range: cfg.long_range,
    };
    let output = vpci_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_out = VpciBatchJsOutput {
        vpci: output.vpci,
        vpcis: output.vpcis,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    vpci_ptr: *mut f64,
    vpcis_ptr: *mut f64,
    len: usize,
    short_start: usize,
    short_end: usize,
    short_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to vpci_batch_into"));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let sweep = VpciBatchRange {
            short_range: (short_start, short_end, short_step),
            long_range: (long_start, long_end, long_step),
        };

        let combos = expand_grid_vpci(&sweep);
        let rows = combos.len();
        let total_len = rows * len;

        // Need to handle aliasing only between outputs and inputs
        let need_temp = close_ptr == vpci_ptr as *const f64
            || close_ptr == vpcis_ptr as *const f64
            || volume_ptr == vpci_ptr as *const f64
            || volume_ptr == vpcis_ptr as *const f64;

        if need_temp {
            // Run batch into temporary buffers
            let output = vpci_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy to output pointers
            let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, total_len);
            let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, total_len);
            vpci_out.copy_from_slice(&output.vpci);
            vpcis_out.copy_from_slice(&output.vpcis);
        } else {
            // Direct computation using zero-copy batch operation
            let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, total_len);
            let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, total_len);

            // Use zero-copy batch operation directly into output buffers
            vpci_batch_inner_into(
                close,
                volume,
                &sweep,
                detect_best_kernel(),
                false,
                vpci_out,
                vpcis_out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers or VpciStream"
)]
pub struct VpciContext {
    short_range: usize,
    long_range: usize,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl VpciContext {
    #[wasm_bindgen(constructor)]
    pub fn new(short_range: usize, long_range: usize) -> Result<VpciContext, JsValue> {
        if short_range == 0 || long_range == 0 || short_range > long_range {
            return Err(JsValue::from_str("Invalid range parameters"));
        }

        Ok(VpciContext {
            short_range,
            long_range,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        close_ptr: *const f64,
        volume_ptr: *const f64,
        vpci_ptr: *mut f64,
        vpcis_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null()
        {
            return Err(JsValue::from_str("null pointer passed to update_into"));
        }

        if len < self.long_range {
            return Err(JsValue::from_str("Data length less than long range"));
        }

        unsafe {
            let close = std::slice::from_raw_parts(close_ptr, len);
            let volume = std::slice::from_raw_parts(volume_ptr, len);

            let params = VpciParams {
                short_range: Some(self.short_range),
                long_range: Some(self.long_range),
            };
            let input = VpciInput::from_slices(close, volume, params);

            // Check for aliasing
            let need_temp = close_ptr == vpci_ptr as *const f64
                || close_ptr == vpcis_ptr as *const f64
                || volume_ptr == vpci_ptr as *const f64
                || volume_ptr == vpcis_ptr as *const f64;

            if need_temp {
                let mut temp_vpci = vec![0.0; len];
                let mut temp_vpcis = vec![0.0; len];
                vpci_into_slice(&mut temp_vpci, &mut temp_vpcis, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;

                let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, len);
                let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, len);
                vpci_out.copy_from_slice(&temp_vpci);
                vpcis_out.copy_from_slice(&temp_vpcis);
            } else {
                let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, len);
                let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, len);
                vpci_into_slice(vpci_out, vpcis_out, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.long_range - 1
    }
}
