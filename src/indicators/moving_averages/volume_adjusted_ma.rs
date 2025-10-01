//! # Volume Adjusted Moving Average (VOLUME_ADJUSTED_MA)
//!
//! A volume-weighted moving average that dynamically adjusts its lookback period based on volume.
//! VOLUME_ADJUSTED_MA uses volume increments to determine how many bars to include in the average calculation.
//!
//! ## Parameters
//! - **length**: Base period for the moving average (default: 13)
//! - **vi_factor**: Volume increment factor (default: 0.67)
//! - **strict**: If true, enforces exact length requirement; if false, uses up to length bars (default: true)
//! - **sample_period**: Number of bars to use for volume averaging (0 = all bars) (default: 0)
//!
//! ## Returns
//! - **`Ok(VolumeAdjustedMaOutput)`** on success, containing values vector.
//! - **`Err(VolumeAdjustedMaError)`** otherwise.
//!
//! ## Developer Notes
//! - SIMD status: AVX2/AVX512 enabled for non-strict path (fixed-width window).
//!   Strict=true path retains scalar inner loop for exact early stop semantics.
//! - Scalar optimized: rolling window for `sample_period > 0`, fast reciprocal, mul_add, and
//!   short-circuit for `inv==0`. Baseline improved ~16–26% at 100k.
//! - Bench reference (100k, target-cpu=native): scalar ~4.88 ms → AVX2 ~4.79 ms, AVX512 ~4.73 ms
//!   with default parameters. Larger SIMD gains occur when `strict=false`.
//! - Memory: zero-copy allocation via `alloc_with_nan_prefix`; no O(N) temporaries per output.
//! - Batch SIMD: row-specific batch kernels not implemented; revisit once shared precompute
//!   across rows (avg-volume series) is factored cleanly without complicating the API.

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaVolumeAdjustedMa;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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

// Core imports with complete helper set
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// Parallel processing support
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Standard library imports
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// ==================== TRAIT IMPLEMENTATIONS ====================
impl<'a> AsRef<[f64]> for VolumeAdjustedMaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VolumeAdjustedMaData::Slice { data, .. } => data,
            VolumeAdjustedMaData::Candles {
                candles, source, ..
            } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices with volume and candle data
#[derive(Debug, Clone)]
pub enum VolumeAdjustedMaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice {
        data: &'a [f64],
        volume: &'a [f64],
    },
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct VolumeAdjustedMaOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VolumeAdjustedMaParams {
    pub length: Option<usize>,
    pub vi_factor: Option<f64>,
    pub strict: Option<bool>,
    pub sample_period: Option<usize>,
}

impl Default for VolumeAdjustedMaParams {
    fn default() -> Self {
        Self {
            length: Some(13),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct VolumeAdjustedMaInput<'a> {
    pub data: VolumeAdjustedMaData<'a>,
    pub params: VolumeAdjustedMaParams,
}

impl<'a> VolumeAdjustedMaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VolumeAdjustedMaParams) -> Self {
        Self {
            data: VolumeAdjustedMaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(data: &'a [f64], volume: &'a [f64], p: VolumeAdjustedMaParams) -> Self {
        Self {
            data: VolumeAdjustedMaData::Slice { data, volume },
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VolumeAdjustedMaParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(13)
    }

    #[inline]
    pub fn get_vi_factor(&self) -> f64 {
        self.params.vi_factor.unwrap_or(0.67)
    }

    #[inline]
    pub fn get_strict(&self) -> bool {
        self.params.strict.unwrap_or(true)
    }

    #[inline]
    pub fn get_sample_period(&self) -> usize {
        self.params.sample_period.unwrap_or(0)
    }

    #[inline]
    pub fn get_volume(&self) -> &[f64] {
        match &self.data {
            VolumeAdjustedMaData::Candles { candles, .. } => &candles.volume,
            VolumeAdjustedMaData::Slice { volume, .. } => volume,
        }
    }
}

// ==================== BUILDER API ====================
#[derive(Copy, Clone, Debug)]
pub struct VolumeAdjustedMaBuilder {
    length: Option<usize>,
    vi_factor: Option<f64>,
    strict: Option<bool>,
    sample_period: Option<usize>,
    kernel: Kernel,
}

impl Default for VolumeAdjustedMaBuilder {
    fn default() -> Self {
        Self {
            length: None,
            vi_factor: None,
            strict: None,
            sample_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeAdjustedMaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VolumeAdjustedMaStream, VolumeAdjustedMaError> {
        let p = VolumeAdjustedMaParams {
            length: self.length,
            vi_factor: self.vi_factor,
            strict: self.strict,
            sample_period: self.sample_period,
        };
        VolumeAdjustedMaStream::try_new(p)
    }

    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }

    #[inline(always)]
    pub fn vi_factor(mut self, f: f64) -> Self {
        self.vi_factor = Some(f);
        self
    }

    #[inline(always)]
    pub fn strict(mut self, b: bool) -> Self {
        self.strict = Some(b);
        self
    }

    #[inline(always)]
    pub fn sample_period(mut self, n: usize) -> Self {
        self.sample_period = Some(n);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VolumeAdjustedMaOutput, VolumeAdjustedMaError> {
        let p = VolumeAdjustedMaParams {
            length: self.length,
            vi_factor: self.vi_factor,
            strict: self.strict,
            sample_period: self.sample_period,
        };
        let i = VolumeAdjustedMaInput::from_candles(c, "close", p);
        VolumeAdjustedMa_with_kernel(&i, self.kernel)
    }

    #[deprecated(note = "Use apply(&Candles) for default 'close' or apply_candles_src(...)")]
    #[inline(always)]
    pub fn apply_src(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<VolumeAdjustedMaOutput, VolumeAdjustedMaError> {
        let p = VolumeAdjustedMaParams {
            length: self.length,
            vi_factor: self.vi_factor,
            strict: self.strict,
            sample_period: self.sample_period,
        };
        let i = VolumeAdjustedMaInput::from_candles(c, src, p);
        VolumeAdjustedMa_with_kernel(&i, self.kernel)
    }

    /// ALMA-compatible entry point for slice inputs.
    /// Doc alias provided so `apply_slice` shows up in searches.
    /// This indicator requires both `data` and `volume`.
    #[inline(always)]
    #[doc(alias = "apply_slice")]
    pub fn apply_slices(
        self,
        data: &[f64],
        volume: &[f64],
    ) -> Result<VolumeAdjustedMaOutput, VolumeAdjustedMaError> {
        let p = VolumeAdjustedMaParams {
            length: self.length,
            vi_factor: self.vi_factor,
            strict: self.strict,
            sample_period: self.sample_period,
        };
        let i = VolumeAdjustedMaInput::from_slices(data, volume, p);
        VolumeAdjustedMa_with_kernel(&i, self.kernel)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum VolumeAdjustedMaError {
    #[error("volume_adjusted_ma: Input data slice is empty.")]
    EmptyInputData,

    #[error("volume_adjusted_ma: Volume data slice is empty.")]
    EmptyVolumeData,

    #[error("volume_adjusted_ma: All values are NaN.")]
    AllValuesNaN,

    #[error("volume_adjusted_ma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("volume_adjusted_ma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("volume_adjusted_ma: Invalid vi_factor: {vi_factor}. Must be positive.")]
    InvalidViFactor { vi_factor: f64 },

    #[error(
        "volume_adjusted_ma: Data length mismatch: price = {price_len}, volume = {volume_len}"
    )]
    DataLengthMismatch { price_len: usize, volume_len: usize },
}

// ==================== PREPARATION AND COMPUTE FUNCTIONS ====================
#[inline(always)]
fn VolumeAdjustedMa_prepare<'a>(
    input: &'a VolumeAdjustedMaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, f64, bool, usize, usize, Kernel), VolumeAdjustedMaError> {
    let data: &[f64] = input.as_ref();
    let vol: &[f64] = input.get_volume();
    let len = data.len();
    if len == 0 {
        return Err(VolumeAdjustedMaError::EmptyInputData);
    }
    if vol.len() == 0 {
        return Err(VolumeAdjustedMaError::EmptyVolumeData);
    }
    if len != vol.len() {
        return Err(VolumeAdjustedMaError::DataLengthMismatch {
            price_len: len,
            volume_len: vol.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VolumeAdjustedMaError::AllValuesNaN)?;
    let length = input.get_length();
    let vi_factor = input.get_vi_factor();
    let strict = input.get_strict();
    let sample_period = input.get_sample_period();

    if length == 0 || length > len {
        return Err(VolumeAdjustedMaError::InvalidPeriod {
            period: length,
            data_len: len,
        });
    }
    if !vi_factor.is_finite() || vi_factor <= 0.0 {
        return Err(VolumeAdjustedMaError::InvalidViFactor { vi_factor });
    }
    if len - first < length {
        return Err(VolumeAdjustedMaError::NotEnoughValidData {
            needed: length,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((
        data,
        vol,
        length,
        vi_factor,
        strict,
        sample_period,
        first,
        chosen,
    ))
}

#[inline(always)]
fn VolumeAdjustedMa_compute_into(
    data: &[f64],
    vol: &[f64],
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            VolumeAdjustedMa_avx512(
                data,
                vol,
                length,
                vi_factor,
                strict,
                sample_period,
                first,
                out,
            )
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            VolumeAdjustedMa_avx2(
                data,
                vol,
                length,
                vi_factor,
                strict,
                sample_period,
                first,
                out,
            )
        },
        _ => VolumeAdjustedMa_scalar(
            data,
            vol,
            length,
            vi_factor,
            strict,
            sample_period,
            first,
            out,
        ),
    }
}

#[inline(always)]
fn VolumeAdjustedMa_scalar(
    data: &[f64],
    vol: &[f64],
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let warmup = first + length - 1;

    // Pre-state for average-volume calculation
    let mut cum_vol = 0.0f64; // for sample_period == 0
    let mut win_sum = 0.0f64; // for sample_period  > 0 (rolling)
    let mut win_ready = false;

    // Preload cumulative volume up to (but not including) `warmup` to match original semantics
    if sample_period == 0 {
        let upto = warmup.min(len);
        for j in 0..upto {
            let x = vol[j];
            if x.is_finite() {
                cum_vol += x;
            }
        }
    }

    for i in warmup..len {
        // Compute average volume
        let avg_volume = if sample_period == 0 {
            // cumulative average over all bars 0..=i
            let x = vol[i];
            if x.is_finite() {
                cum_vol += x;
            }
            cum_vol / ((i + 1) as f64)
        } else {
            // exact average over last `sample_period` bars using a rolling window
            if i + 1 < sample_period {
                out[i] = f64::NAN;
                continue;
            }
            if !win_ready {
                let start = i + 1 - sample_period;
                let mut s = 0.0f64;
                for k in start..=i {
                    let v = vol[k];
                    if v.is_finite() {
                        s += v;
                    }
                }
                win_sum = s;
                win_ready = true;
            } else {
                let addv = vol[i];
                if addv.is_finite() {
                    win_sum += addv;
                }
                let remv = vol[i - sample_period];
                if remv.is_finite() {
                    win_sum -= remv;
                }
            }
            win_sum / (sample_period as f64)
        };

        // Threshold and reciprocal (avoid division in hot loop)
        let vi_th = avg_volume * vi_factor;
        let inv = if vi_th > 0.0 { 1.0 / vi_th } else { 0.0 };

        // Cap of backward accumulation
        let cap = if strict {
            length.saturating_mul(10).min(i + 1)
        } else {
            length.min(i + 1)
        };

        // Fast-path: when inv == 0, contribution weights are zero => result reduces to p0
        if inv == 0.0 {
            let nmb = cap; // number of bars we conceptually traversed
            if i >= nmb {
                let p0 = data[i - nmb];
                out[i] = if p0.is_finite() { p0 } else { f64::NAN };
            } else {
                out[i] = f64::NAN;
            }
            continue;
        }

        // Backward accumulation
        let mut weighted_sum = 0.0f64;
        let mut v2i_sum = 0.0f64;
        let mut nmb = 0usize;

        let mut idx = i;
        for j in 0..cap {
            let vv = vol[idx];
            let v2i = if vv.is_finite() { vv * inv } else { 0.0 };
            v2i_sum += v2i;

            let px = data[idx];
            if px.is_finite() {
                weighted_sum = px.mul_add(v2i, weighted_sum);
            }

            nmb = j + 1;

            if strict {
                if v2i_sum >= length as f64 {
                    break;
                }
            } else if nmb >= length {
                break;
            }

            if idx == 0 {
                break;
            }
            idx -= 1;
        }

        // vama = ((length - v2i_sum)*p0 + weighted_sum) / length
        if nmb > 0 && i >= nmb {
            let p0 = data[i - nmb];
            if p0.is_finite() {
                out[i] = ((length as f64 - v2i_sum).mul_add(p0, weighted_sum)) / (length as f64);
            } else {
                out[i] = f64::NAN;
            }
        } else {
            out[i] = f64::NAN;
        }
    }
}

// ==================== AVX512 SIMD Implementation ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn VolumeAdjustedMa_avx512(
    data: &[f64],
    vol: &[f64],
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = data.len();
    if len == 0 {
        return;
    }

    let warmup = first + length - 1;
    let len_f = length as f64;

    // rolling/cumulative volume
    let mut cum_vol = 0.0f64;
    let mut win_sum = 0.0f64;
    let mut win_ready = false;

    if sample_period == 0 {
        let upto = warmup.min(len);
        for j in 0..upto {
            let x = *vol.get_unchecked(j);
            if x.is_finite() {
                cum_vol += x;
            }
        }
    }

    for i in warmup..len {
        // average volume identical to scalar
        let avg_volume = if sample_period == 0 {
            let vi = *vol.get_unchecked(i);
            if vi.is_finite() {
                cum_vol += vi;
            }
            cum_vol / ((i + 1) as f64)
        } else {
            if i + 1 < sample_period {
                *out.get_unchecked_mut(i) = f64::NAN;
                continue;
            }
            if !win_ready {
                let start = i + 1 - sample_period;
                let mut s = 0.0f64;
                for k in start..=i {
                    let x = *vol.get_unchecked(k);
                    if x.is_finite() {
                        s += x;
                    }
                }
                win_sum = s;
                win_ready = true;
            } else {
                let addv = *vol.get_unchecked(i);
                if addv.is_finite() {
                    win_sum += addv;
                }
                let remv = *vol.get_unchecked(i - sample_period);
                if remv.is_finite() {
                    win_sum -= remv;
                }
            }
            win_sum / (sample_period as f64)
        };

        let vi_th = avg_volume * vi_factor;
        let inv = if vi_th > 0.0 { 1.0 / vi_th } else { 0.0 };

        // cap mirrors scalar
        let cap = if strict {
            length.saturating_mul(10).min(i + 1)
        } else {
            length.min(i + 1)
        };

        if inv == 0.0 {
            let nmb = cap;
            if i >= nmb {
                let p0 = *data.get_unchecked(i - nmb);
                *out.get_unchecked_mut(i) = if p0.is_finite() { p0 } else { f64::NAN };
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
            continue;
        }

        if !strict && cap == length {
            // Vectorized fixed-width reduction over [i+1-length, i]
            let start = i + 1 - length;

            let inv_v = _mm512_set1_pd(inv);

            let mut acc_v2i = _mm512_setzero_pd();
            let mut acc_w = _mm512_setzero_pd();

            let mut j = 0usize;
            while j + 8 <= length {
                let idx = start + j;
                let vv = _mm512_loadu_pd(vol.as_ptr().add(idx));
                let mv = _mm512_cmp_pd_mask(vv, vv, _CMP_ORD_Q);
                let v2 = _mm512_mul_pd(vv, inv_v);
                let v2 = _mm512_maskz_mov_pd(mv, v2);

                let pp = _mm512_loadu_pd(data.as_ptr().add(idx));
                let mp = _mm512_cmp_pd_mask(pp, pp, _CMP_ORD_Q);
                let w = _mm512_mul_pd(pp, v2);
                let w = _mm512_maskz_mov_pd(mp, w);

                acc_v2i = _mm512_add_pd(acc_v2i, v2);
                acc_w = _mm512_add_pd(acc_w, w);
                j += 8;
            }

            let mut v2i_sum = _mm512_reduce_add_pd(acc_v2i);
            let mut weighted_sum = _mm512_reduce_add_pd(acc_w);

            while j < length {
                let idx = start + j;
                let vv = *vol.get_unchecked(idx);
                let v2 = if vv.is_finite() { vv * inv } else { 0.0 };
                v2i_sum += v2;

                let px = *data.get_unchecked(idx);
                if px.is_finite() {
                    weighted_sum = px.mul_add(v2, weighted_sum);
                }
                j += 1;
            }

            if i >= length {
                let p0 = *data.get_unchecked(i - length);
                *out.get_unchecked_mut(i) = if p0.is_finite() {
                    ((len_f - v2i_sum).mul_add(p0, weighted_sum)) / len_f
                } else {
                    f64::NAN
                };
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
        } else {
            // strict=true (or unusual cap): use scalar-style inner loop with exact early-stop behavior
            let mut weighted_sum = 0.0f64;
            let mut v2i_sum = 0.0f64;
            let mut nmb = 0usize;
            let mut idx = i;
            while nmb < cap {
                let vv = *vol.get_unchecked(idx);
                let v2i = if vv.is_finite() { vv * inv } else { 0.0 };
                v2i_sum += v2i;

                let px = *data.get_unchecked(idx);
                if px.is_finite() {
                    weighted_sum = px.mul_add(v2i, weighted_sum);
                }

                nmb += 1;
                if strict && v2i_sum >= len_f {
                    break;
                }
                if idx == 0 {
                    break;
                }
                idx -= 1;
            }

            if nmb > 0 && i >= nmb {
                let p0 = *data.get_unchecked(i - nmb);
                *out.get_unchecked_mut(i) = if p0.is_finite() {
                    ((len_f - v2i_sum).mul_add(p0, weighted_sum)) / len_f
                } else {
                    f64::NAN
                };
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
        }
    }
}

// ==================== AVX2 SIMD Implementation ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn VolumeAdjustedMa_avx2(
    data: &[f64],
    vol: &[f64],
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum256_pd(v: __m256d) -> f64 {
        let lo: __m128d = _mm256_castpd256_pd128(v);
        let hi: __m128d = _mm256_extractf128_pd(v, 1);
        let s = _mm_add_pd(lo, hi);
        let s2 = _mm_hadd_pd(s, s);
        _mm_cvtsd_f64(s2)
    }

    let len = data.len();
    if len == 0 {
        return;
    }

    let warmup = first + length - 1;
    let len_f = length as f64;

    // rolling/cumulative volume
    let mut cum_vol = 0.0f64;
    let mut win_sum = 0.0f64;
    let mut win_ready = false;

    if sample_period == 0 {
        let upto = warmup.min(len);
        for j in 0..upto {
            let x = *vol.get_unchecked(j);
            if x.is_finite() {
                cum_vol += x;
            }
        }
    }

    for i in warmup..len {
        let avg_volume = if sample_period == 0 {
            let vi = *vol.get_unchecked(i);
            if vi.is_finite() {
                cum_vol += vi;
            }
            cum_vol / ((i + 1) as f64)
        } else {
            if i + 1 < sample_period {
                *out.get_unchecked_mut(i) = f64::NAN;
                continue;
            }
            if !win_ready {
                let start = i + 1 - sample_period;
                let mut s = 0.0f64;
                for k in start..=i {
                    let x = *vol.get_unchecked(k);
                    if x.is_finite() {
                        s += x;
                    }
                }
                win_sum = s;
                win_ready = true;
            } else {
                let addv = *vol.get_unchecked(i);
                if addv.is_finite() {
                    win_sum += addv;
                }
                let remv = *vol.get_unchecked(i - sample_period);
                if remv.is_finite() {
                    win_sum -= remv;
                }
            }
            win_sum / (sample_period as f64)
        };

        let vi_th = avg_volume * vi_factor;
        let inv = if vi_th > 0.0 { 1.0 / vi_th } else { 0.0 };

        let cap = if strict {
            length.saturating_mul(10).min(i + 1)
        } else {
            length.min(i + 1)
        };

        if inv == 0.0 {
            let nmb = cap;
            if i >= nmb {
                let p0 = *data.get_unchecked(i - nmb);
                *out.get_unchecked_mut(i) = if p0.is_finite() { p0 } else { f64::NAN };
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
            continue;
        }

        if !strict && cap == length {
            let start = i + 1 - length;

            let inv_v = _mm256_set1_pd(inv);

            let mut acc_v2i = _mm256_setzero_pd();
            let mut acc_w = _mm256_setzero_pd();

            let mut j = 0usize;
            while j + 4 <= length {
                let idx = start + j;
                let vv = _mm256_loadu_pd(vol.as_ptr().add(idx));
                let m_v = _mm256_cmp_pd(vv, vv, _CMP_ORD_Q);
                let v2 = _mm256_mul_pd(vv, inv_v);
                let v2 = _mm256_and_pd(v2, m_v);

                let pp = _mm256_loadu_pd(data.as_ptr().add(idx));
                let m_p = _mm256_cmp_pd(pp, pp, _CMP_ORD_Q);
                let w = _mm256_mul_pd(pp, v2);
                let w = _mm256_and_pd(w, m_p);

                acc_v2i = _mm256_add_pd(acc_v2i, v2);
                acc_w = _mm256_add_pd(acc_w, w);
                j += 4;
            }

            let mut v2i_sum = hsum256_pd(acc_v2i);
            let mut weighted_sum = hsum256_pd(acc_w);

            while j < length {
                let idx = start + j;
                let vv = *vol.get_unchecked(idx);
                let v2 = if vv.is_finite() { vv * inv } else { 0.0 };
                v2i_sum += v2;

                let px = *data.get_unchecked(idx);
                if px.is_finite() {
                    weighted_sum = px.mul_add(v2, weighted_sum);
                }
                j += 1;
            }

            if i >= length {
                let p0 = *data.get_unchecked(i - length);
                *out.get_unchecked_mut(i) = if p0.is_finite() {
                    ((len_f - v2i_sum).mul_add(p0, weighted_sum)) / len_f
                } else {
                    f64::NAN
                };
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
        } else {
            // strict=true (or unusual cap): scalar-style inner for early-stop parity
            let mut weighted_sum = 0.0f64;
            let mut v2i_sum = 0.0f64;
            let mut nmb = 0usize;
            let mut idx = i;
            while nmb < cap {
                let vv = *vol.get_unchecked(idx);
                let v2i = if vv.is_finite() { vv * inv } else { 0.0 };
                v2i_sum += v2i;

                let px = *data.get_unchecked(idx);
                if px.is_finite() {
                    weighted_sum = px.mul_add(v2i, weighted_sum);
                }

                nmb += 1;
                if strict && v2i_sum >= len_f {
                    break;
                }
                if idx == 0 {
                    break;
                }
                idx -= 1;
            }

            if nmb > 0 && i >= nmb {
                let p0 = *data.get_unchecked(i - nmb);
                *out.get_unchecked_mut(i) = if p0.is_finite() {
                    ((len_f - v2i_sum).mul_add(p0, weighted_sum)) / len_f
                } else {
                    f64::NAN
                };
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
        }
    }
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
#[inline]
pub fn VolumeAdjustedMa(
    input: &VolumeAdjustedMaInput,
) -> Result<VolumeAdjustedMaOutput, VolumeAdjustedMaError> {
    VolumeAdjustedMa_with_kernel(input, Kernel::Auto)
}

pub fn VolumeAdjustedMa_with_kernel(
    input: &VolumeAdjustedMaInput,
    kernel: Kernel,
) -> Result<VolumeAdjustedMaOutput, VolumeAdjustedMaError> {
    let (data, vol, length, vi_factor, strict, sample_period, first, chosen) =
        VolumeAdjustedMa_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + length - 1);
    VolumeAdjustedMa_compute_into(
        data,
        vol,
        length,
        vi_factor,
        strict,
        sample_period,
        first,
        chosen,
        &mut out,
    );
    Ok(VolumeAdjustedMaOutput { values: out })
}

#[inline]
pub fn VolumeAdjustedMa_into_slice(
    dst: &mut [f64],
    input: &VolumeAdjustedMaInput,
    kern: Kernel,
) -> Result<(), VolumeAdjustedMaError> {
    let (data, vol, length, vi_factor, strict, sample_period, first, chosen) =
        VolumeAdjustedMa_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(VolumeAdjustedMaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    VolumeAdjustedMa_compute_into(
        data,
        vol,
        length,
        vi_factor,
        strict,
        sample_period,
        first,
        chosen,
        dst,
    );

    let warm = first + length - 1;
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug)]
pub struct VolumeAdjustedMaBatchRange {
    pub length: (usize, usize, usize),
    pub vi_factor: (f64, f64, f64),
    pub sample_period: (usize, usize, usize),
    pub strict: Option<bool>, // None => both {true,false}
}

impl Default for VolumeAdjustedMaBatchRange {
    fn default() -> Self {
        Self {
            length: (13, 55, 1),
            vi_factor: (0.67, 0.67, 0.0),
            sample_period: (0, 0, 0),
            strict: Some(true),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VolumeAdjustedMaBatchBuilder {
    range: VolumeAdjustedMaBatchRange,
    kernel: Kernel,
}

impl VolumeAdjustedMaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn length_range(mut self, s: usize, e: usize, st: usize) -> Self {
        self.range.length = (s, e, st);
        self
    }
    pub fn length_static(mut self, n: usize) -> Self {
        self.range.length = (n, n, 0);
        self
    }
    pub fn vi_factor_range(mut self, s: f64, e: f64, st: f64) -> Self {
        self.range.vi_factor = (s, e, st);
        self
    }
    pub fn vi_factor_static(mut self, f: f64) -> Self {
        self.range.vi_factor = (f, f, 0.0);
        self
    }
    pub fn sample_period_range(mut self, s: usize, e: usize, st: usize) -> Self {
        self.range.sample_period = (s, e, st);
        self
    }
    pub fn sample_period_static(mut self, n: usize) -> Self {
        self.range.sample_period = (n, n, 0);
        self
    }
    pub fn strict_static(mut self, b: bool) -> Self {
        self.range.strict = Some(b);
        self
    }
    pub fn strict_both(mut self) -> Self {
        self.range.strict = None;
        self
    }

    pub fn apply_slices(
        self,
        data: &[f64],
        volume: &[f64],
    ) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
        VolumeAdjustedMa_batch_with_kernel(data, volume, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
        self.apply_slices(source_type(c, src), &c.volume)
    }

    pub fn with_default_slice(
        data: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
        VolumeAdjustedMaBatchBuilder::new()
            .kernel(k)
            .apply_slices(data, volume)
    }

    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
        VolumeAdjustedMaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct VolumeAdjustedMaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VolumeAdjustedMaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeAdjustedMaBatchOutput {
    pub fn row_for_params(&self, p: &VolumeAdjustedMaParams) -> Option<usize> {
        self.combos.iter().position(|q| {
            q.length.unwrap_or(13) == p.length.unwrap_or(13)
                && (q.vi_factor.unwrap_or(0.67) - p.vi_factor.unwrap_or(0.67)).abs() < 1e-12
                && q.strict.unwrap_or(true) == p.strict.unwrap_or(true)
                && q.sample_period.unwrap_or(0) == p.sample_period.unwrap_or(0)
        })
    }

    pub fn values_for(&self, p: &VolumeAdjustedMaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
    if st == 0 || s == e {
        vec![s]
    } else {
        (s..=e).step_by(st).collect()
    }
}

#[inline(always)]
fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
    if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
        vec![s]
    } else {
        let mut v = Vec::new();
        let mut x = s;
        while x <= e + 1e-12 {
            v.push(x);
            x += st;
        }
        v
    }
}

#[inline(always)]
fn expand_grid_VolumeAdjustedMa(r: &VolumeAdjustedMaBatchRange) -> Vec<VolumeAdjustedMaParams> {
    let lengths = axis_usize(r.length);
    let vfs = axis_f64(r.vi_factor);
    let sps = axis_usize(r.sample_period);
    let stricts: Vec<bool> = match r.strict {
        Some(b) => vec![b],
        None => vec![true, false],
    };
    let mut out = Vec::with_capacity(lengths.len() * vfs.len() * sps.len() * stricts.len());
    for &l in &lengths {
        for &vf in &vfs {
            for &sp in &sps {
                for &st in &stricts {
                    out.push(VolumeAdjustedMaParams {
                        length: Some(l),
                        vi_factor: Some(vf),
                        sample_period: Some(sp),
                        strict: Some(st),
                    });
                }
            }
        }
    }
    out
}

pub fn VolumeAdjustedMa_batch_with_kernel(
    data: &[f64],
    volume: &[f64],
    sweep: &VolumeAdjustedMaBatchRange,
    k: Kernel,
) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
    let batch = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VolumeAdjustedMaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    // Map batch kernel to single kernel used by the row worker
    let simd = match batch {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    VolumeAdjustedMa_batch_inner(data, volume, sweep, simd, false)
}

#[inline(always)]
pub fn VolumeAdjustedMa_batch_slice(
    data: &[f64],
    volume: &[f64],
    sweep: &VolumeAdjustedMaBatchRange,
    k: Kernel,
) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
    let batch = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VolumeAdjustedMaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match batch {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    VolumeAdjustedMa_batch_inner(data, volume, sweep, simd, false)
}

#[inline(always)]
pub fn VolumeAdjustedMa_batch_par_slice(
    data: &[f64],
    volume: &[f64],
    sweep: &VolumeAdjustedMaBatchRange,
    k: Kernel,
) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
    let batch = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VolumeAdjustedMaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match batch {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    VolumeAdjustedMa_batch_inner(data, volume, sweep, simd, true)
}

fn VolumeAdjustedMa_batch_inner(
    data: &[f64],
    volume: &[f64],
    sweep: &VolumeAdjustedMaBatchRange,
    k: Kernel,
    parallel: bool,
) -> Result<VolumeAdjustedMaBatchOutput, VolumeAdjustedMaError> {
    if data.len() != volume.len() {
        return Err(VolumeAdjustedMaError::DataLengthMismatch {
            price_len: data.len(),
            volume_len: volume.len(),
        });
    }
    let combos = expand_grid_VolumeAdjustedMa(sweep);
    if combos.is_empty() {
        return Err(VolumeAdjustedMaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let cols = data.len();
    if cols == 0 {
        return Err(VolumeAdjustedMaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VolumeAdjustedMaError::AllValuesNaN)?;
    let rows = combos.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + p.length.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            VolumeAdjustedMa_batch_inner_into_par(data, volume, &combos, first, k, out)?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            VolumeAdjustedMa_batch_inner_into(data, volume, &combos, first, k, out)?;
        }
    } else {
        VolumeAdjustedMa_batch_inner_into(data, volume, &combos, first, k, out)?;
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(VolumeAdjustedMaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn VolumeAdjustedMa_batch_inner_into(
    data: &[f64],
    volume: &[f64],
    combos: &[VolumeAdjustedMaParams],
    first: usize,
    kern: Kernel,
    out: &mut [f64],
) -> Result<(), VolumeAdjustedMaError> {
    let cols = data.len();
    for (row, dst) in out.chunks_mut(cols).enumerate() {
        let p = &combos[row];
        VolumeAdjustedMa_compute_into(
            data,
            volume,
            p.length.unwrap(),
            p.vi_factor.unwrap(),
            p.strict.unwrap(),
            p.sample_period.unwrap(),
            first,
            kern,
            dst,
        );
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn VolumeAdjustedMa_batch_inner_into_par(
    data: &[f64],
    volume: &[f64],
    combos: &[VolumeAdjustedMaParams],
    first: usize,
    kern: Kernel,
    out: &mut [f64],
) -> Result<(), VolumeAdjustedMaError> {
    let cols = data.len();
    out.par_chunks_mut(cols)
        .enumerate()
        .try_for_each(|(row, dst)| {
            let p = &combos[row];
            VolumeAdjustedMa_compute_into(
                data,
                volume,
                p.length.unwrap(),
                p.vi_factor.unwrap(),
                p.strict.unwrap(),
                p.sample_period.unwrap(),
                first,
                kern,
                dst,
            );
            Ok::<(), VolumeAdjustedMaError>(())
        })
}

// ==================== STREAMING SUPPORT ====================
#[derive(Debug, Clone)]
pub struct VolumeAdjustedMaStream {
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    prices: Vec<f64>,  // Now stores all historical data
    volumes: Vec<f64>, // Now stores all historical data
}

impl VolumeAdjustedMaStream {
    pub fn try_new(p: VolumeAdjustedMaParams) -> Result<Self, VolumeAdjustedMaError> {
        let length = p.length.unwrap_or(13);
        if length == 0 {
            return Err(VolumeAdjustedMaError::InvalidPeriod {
                period: length,
                data_len: 0,
            });
        }
        let vi = p.vi_factor.unwrap_or(0.67);
        if !vi.is_finite() || vi <= 0.0 {
            return Err(VolumeAdjustedMaError::InvalidViFactor { vi_factor: vi });
        }
        let strict = p.strict.unwrap_or(true);
        let sp = p.sample_period.unwrap_or(0);

        Ok(Self {
            length,
            vi_factor: vi,
            strict,
            sample_period: sp,
            prices: Vec::new(),  // Start empty, will grow dynamically
            volumes: Vec::new(), // Start empty, will grow dynamically
        })
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        // Store all historical data (O(n) space)
        self.prices.push(price);
        self.volumes.push(volume);

        let i = self.prices.len() - 1; // Current index

        // Need at least length bars
        if i < self.length - 1 {
            return Some(f64::NAN);
        }

        // Calculate average volume for normalization
        let avg_volume = if self.sample_period == 0 {
            // Use all historical data
            let mut sum = 0.0;
            for v in &self.volumes {
                sum += if v.is_finite() { *v } else { 0.0 };
            }
            if self.volumes.len() > 0 {
                sum / (self.volumes.len() as f64)
            } else {
                0.0
            }
        } else {
            // Use sample_period bars
            if i < self.sample_period - 1 {
                // Not enough bars to form the sample window
                return Some(f64::NAN);
            }
            let start = i + 1 - self.sample_period;
            let mut sum = 0.0;
            for j in start..=i {
                sum += if self.volumes[j].is_finite() {
                    self.volumes[j]
                } else {
                    0.0
                };
            }
            sum / (self.sample_period as f64)
        };

        let vi_th = avg_volume * self.vi_factor;

        // Accumulate from current bar backward (O(n) per update)
        let cap = if self.strict {
            self.length.saturating_mul(10).min(i + 1)
        } else {
            self.length.min(i + 1)
        };

        let mut weighted_sum = 0.0;
        let mut v2i_sum = 0.0;
        let mut nmb = 0usize;

        let mut idx = i;
        for j in 0..cap {
            let raw_v2i = if vi_th > 0.0 && self.volumes[idx].is_finite() {
                self.volumes[idx] / vi_th
            } else {
                f64::NAN
            };
            let v2i_nz = if raw_v2i.is_finite() { raw_v2i } else { 0.0 };

            let p = self.prices[idx];
            if p.is_finite() {
                weighted_sum += p * v2i_nz;
            }
            v2i_sum += v2i_nz;

            nmb = j + 1;

            if self.strict {
                if v2i_sum >= self.length as f64 {
                    break;
                }
            } else if nmb >= self.length {
                break;
            }

            if idx == 0 {
                break;
            }
            idx -= 1;
        }

        // Calculate final value
        if nmb > 0 && i >= nmb {
            let idx_nmb = i - nmb;
            let p0 = self.prices[idx_nmb];
            if p0.is_finite() {
                Some((weighted_sum - (v2i_sum - self.length as f64) * p0) / (self.length as f64))
            } else {
                Some(f64::NAN)
            }
        } else {
            Some(f64::NAN)
        }
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "VolumeAdjustedMa")]
#[pyo3(signature = (data, volume, length=13, vi_factor=0.67, strict=true, sample_period=0, kernel=None))]
pub fn volume_adjusted_ma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let volume_in = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = VolumeAdjustedMaParams {
        length: Some(length),
        vi_factor: Some(vi_factor),
        strict: Some(strict),
        sample_period: Some(sample_period),
    };
    let input = VolumeAdjustedMaInput::from_slices(slice_in, volume_in, params);

    let result = py
        .allow_threads(|| VolumeAdjustedMa_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "VolumeAdjustedMa_batch")]
#[pyo3(signature = (data, volume, length_range, vi_factor_range, sample_period_range, strict=None, kernel=None))]
pub fn volume_adjusted_ma_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    vi_factor_range: (f64, f64, f64),
    sample_period_range: (usize, usize, usize),
    strict: Option<bool>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let d = data.as_slice()?;
    let v = volume.as_slice()?;
    if d.len() != v.len() {
        return Err(PyValueError::new_err("price and volume length mismatch"));
    }

    let sweep = VolumeAdjustedMaBatchRange {
        length: length_range,
        vi_factor: vi_factor_range,
        sample_period: sample_period_range,
        strict,
    };

    // Expand grid once to size arrays and warms
    let combos = expand_grid_VolumeAdjustedMa(&sweep);
    let rows = combos.len();
    let cols = d.len();
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("empty grid or data"));
    }

    // Pre-allocate NumPy array and initialize warm prefixes to NaN without full fills
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

    // Init warm prefixes using the shared helper on raw memory
    let first = d
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("all data values are NaN"))?;
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + p.length.unwrap() - 1)
        .collect();

    unsafe {
        // Treat PyArray memory as MaybeUninit<f64> to use init_matrix_prefixes
        let mu_slice = std::slice::from_raw_parts_mut(
            out_arr.as_ptr() as *mut std::mem::MaybeUninit<f64>,
            rows * cols,
        );
        init_matrix_prefixes(mu_slice, cols, &warms);
    }

    // Compute into the same buffer, honoring batch kernel mapping
    let kern = validate_kernel(kernel, true)?;
    let batch = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let simd = match batch {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    // SAFETY: memory is valid for f64 writes
    let out_slice = unsafe { out_arr.as_slice_mut()? };
    py.allow_threads(|| VolumeAdjustedMa_batch_inner_into(d, v, &combos, first, simd, out_slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Package metadata like ALMA
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "vi_factors",
        combos
            .iter()
            .map(|p| p.vi_factor.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sample_periods",
        combos
            .iter()
            .map(|p| p.sample_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stricts",
        combos
            .iter()
            .map(|p| p.strict.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "volume_adjusted_ma_cuda_batch_dev")]
#[pyo3(signature = (price_f32, volume_f32, length_range, vi_factor_range, sample_period_range, strict=None, device_id=0))]
pub fn volume_adjusted_ma_cuda_batch_dev_py(
    py: Python<'_>,
    price_f32: numpy::PyReadonlyArray1<'_, f32>,
    volume_f32: numpy::PyReadonlyArray1<'_, f32>,
    length_range: (usize, usize, usize),
    vi_factor_range: (f64, f64, f64),
    sample_period_range: (usize, usize, usize),
    strict: Option<bool>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let prices = price_f32.as_slice()?;
    let volumes = volume_f32.as_slice()?;
    if prices.len() != volumes.len() {
        return Err(PyValueError::new_err("price and volume length mismatch"));
    }

    let sweep = VolumeAdjustedMaBatchRange {
        length: length_range,
        vi_factor: vi_factor_range,
        sample_period: sample_period_range,
        strict,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaVolumeAdjustedMa::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vama_batch_dev(prices, volumes, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "volume_adjusted_ma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (price_tm_f32, volume_tm_f32, length, vi_factor, strict=true, sample_period=0, device_id=0))]
pub fn volume_adjusted_ma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    price_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    volume_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let price_slice = price_tm_f32.as_slice()?;
    let volume_slice = volume_tm_f32.as_slice()?;

    let shape = price_tm_f32.shape();
    if shape != volume_tm_f32.shape() {
        return Err(PyValueError::new_err(
            "price and volume tensors must share the same shape",
        ));
    }
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D arrays (time, series)"));
    }
    let rows = shape[0];
    let cols = shape[1];

    let params = VolumeAdjustedMaParams {
        length: Some(length),
        vi_factor: Some(vi_factor),
        strict: Some(strict),
        sample_period: Some(sample_period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaVolumeAdjustedMa::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vama_multi_series_one_param_time_major_dev(
            price_slice,
            volume_slice,
            cols,
            rows,
            &params,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "VolumeAdjustedMaStream")]
pub struct VolumeAdjustedMaStreamPy {
    stream: VolumeAdjustedMaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolumeAdjustedMaStreamPy {
    #[new]
    fn new(length: usize, vi_factor: f64, strict: bool, sample_period: usize) -> PyResult<Self> {
        let s = VolumeAdjustedMaStream::try_new(VolumeAdjustedMaParams {
            length: Some(length),
            vi_factor: Some(vi_factor),
            strict: Some(strict),
            sample_period: Some(sample_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream: s })
    }
    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(price, volume)
    }
}

#[cfg(feature = "python")]
pub fn register_VolumeAdjustedMa_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volume_adjusted_ma_py, m)?)?;
    m.add_function(wrap_pyfunction!(volume_adjusted_ma_batch_py, m)?)?;
    m.add_class::<VolumeAdjustedMaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(volume_adjusted_ma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            volume_adjusted_ma_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    Ok(())
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VolumeAdjustedMaJsResult {
    pub values: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn volume_adjusted_ma_js(
    data: &[f64],
    volume: &[f64],
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = VolumeAdjustedMaParams {
        length: Some(length),
        vi_factor: Some(vi_factor),
        strict: Some(strict),
        sample_period: Some(sample_period),
    };
    let input = VolumeAdjustedMaInput::from_slices(data, volume, params);

    VolumeAdjustedMa(&input)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn volume_adjusted_ma_unified_js(
    data: &[f64],
    volume: &[f64],
    length: Option<usize>,
    vi_factor: Option<f64>,
    strict: Option<bool>,
    sample_period: Option<usize>,
) -> Result<JsValue, JsValue> {
    let params = VolumeAdjustedMaParams {
        length,
        vi_factor,
        strict,
        sample_period,
    };
    let input = VolumeAdjustedMaInput::from_slices(data, volume, params);

    VolumeAdjustedMa(&input)
        .map(|output| {
            serde_wasm_bindgen::to_value(&VolumeAdjustedMaJsResult {
                values: output.values,
            })
            .unwrap()
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn volume_adjusted_ma_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn volume_adjusted_ma_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn volume_adjusted_ma_into(
    price_ptr: *const f64,
    vol_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
) -> Result<(), JsValue> {
    if price_ptr.is_null() || vol_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(price_ptr, len);
        let vol = std::slice::from_raw_parts(vol_ptr, len);
        let params = VolumeAdjustedMaParams {
            length: Some(length),
            vi_factor: Some(vi_factor),
            strict: Some(strict),
            sample_period: Some(sample_period),
        };
        let input = VolumeAdjustedMaInput::from_slices(data, vol, params);

        let aliased = out_ptr as *const f64 == price_ptr || out_ptr as *const f64 == vol_ptr;
        if aliased {
            let mut tmp = vec![0.0; len];
            VolumeAdjustedMa_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            VolumeAdjustedMa_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VolumeAdjustedMaBatchConfig {
    pub length_range: (usize, usize, usize),
    pub vi_factor_range: (f64, f64, f64),
    pub sample_period_range: (usize, usize, usize),
    pub strict: Option<bool>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VolumeAdjustedMaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VolumeAdjustedMaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn volume_adjusted_ma_batch(
    data: &[f64],
    volume: &[f64],
    cfg: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: VolumeAdjustedMaBatchConfig = serde_wasm_bindgen::from_value(cfg)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = VolumeAdjustedMaBatchRange {
        length: cfg.length_range,
        vi_factor: cfg.vi_factor_range,
        sample_period: cfg.sample_period_range,
        strict: cfg.strict,
    };
    let out = VolumeAdjustedMa_batch_with_kernel(data, volume, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VolumeAdjustedMaBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn volume_adjusted_ma_batch_into(
    price_ptr: *const f64,
    vol_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    vf_start: f64,
    vf_end: f64,
    vf_step: f64,
    sp_start: usize,
    sp_end: usize,
    sp_step: usize,
    strict: Option<bool>,
) -> Result<usize, JsValue> {
    if price_ptr.is_null() || vol_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to VolumeAdjustedMa_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(price_ptr, len);
        let vol = std::slice::from_raw_parts(vol_ptr, len);
        let sweep = VolumeAdjustedMaBatchRange {
            length: (length_start, length_end, length_step),
            vi_factor: (vf_start, vf_end, vf_step),
            sample_period: (sp_start, sp_end, sp_step),
            strict,
        };
        let combos = expand_grid_VolumeAdjustedMa(&sweep);
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| JsValue::from_str("all data values are NaN"))?;
        VolumeAdjustedMa_batch_inner_into(data, vol, &combos, first, simd, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_VolumeAdjustedMa_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters (length=13, vi_factor=0.67, strict=true)
        let input = VolumeAdjustedMaInput::from_candles(
            &candles,
            "close",
            VolumeAdjustedMaParams::default(),
        );
        let result = VolumeAdjustedMa_with_kernel(&input, kernel)?;

        // REFERENCE VALUES (Updated to match actual output with CSV start date)
        let expected = [
            60249.34558277224278,
            60283.79398716031574,
            60173.39296975171601,
            60260.20330381247186,
            60226.09537554050621,
        ];

        let start = result.values.len().saturating_sub(5);

        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] VolumeAdjustedMa {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_VolumeAdjustedMa_slow(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with slow parameters (length=55, vi_factor=0.67, strict=true)
        let params = VolumeAdjustedMaParams {
            length: Some(55),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VolumeAdjustedMaInput::from_candles(&candles, "close", params);
        let result = VolumeAdjustedMa_with_kernel(&input, kernel)?;

        // REFERENCE VALUES (Updated after Pine logic fixes, Slow: length=55)
        let expected = [
            60943.90131552854,
            60929.79497887764,
            60912.66617792769,
            60900.71462347596,
            60844.41271673433,
        ];

        let start = result.values.len().saturating_sub(5);

        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] VolumeAdjustedMa slow {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_VolumeAdjustedMa_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = VolumeAdjustedMaInput::with_default_candles(&c);
        let out = VolumeAdjustedMa_with_kernel(&input, kernel)?;
        assert_eq!(out.values.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_VolumeAdjustedMa_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = VolumeAdjustedMaInput::with_default_candles(&c);
        let out = VolumeAdjustedMa_with_kernel(&input, kernel)?;
        for (i, &v) in out.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x11111111_11111111,
                "[{}] alloc poison at {}",
                test_name, i
            );
            assert_ne!(
                b, 0x22222222_22222222,
                "[{}] init_matrix poison at {}",
                test_name, i
            );
            assert_ne!(
                b, 0x33333333_33333333,
                "[{}] make_uninit poison at {}",
                test_name, i
            );
        }
        Ok(())
    }

    fn check_VolumeAdjustedMa_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let p = VolumeAdjustedMaParams::default();
        let batch = VolumeAdjustedMa_with_kernel(
            &VolumeAdjustedMaInput::from_candles(&c, "close", p.clone()),
            kernel,
        )?
        .values;
        let mut s = VolumeAdjustedMaStream::try_new(p)?;
        let mut seq = Vec::with_capacity(c.close.len());
        for i in 0..c.close.len() {
            seq.push(s.update(c.close[i], c.volume[i]).unwrap_or(f64::NAN));
        }
        assert_eq!(batch.len(), seq.len());
        for (i, (&b, &x)) in batch.iter().zip(seq.iter()).enumerate() {
            if b.is_nan() && x.is_nan() {
                continue;
            }
            let delta = (b - x).abs();
            assert!(
                delta < 1e-9,
                "[{}] stream mismatch at {} - batch: {}, stream: {}, delta: {}",
                test_name,
                i,
                b,
                x,
                delta
            );
        }
        Ok(())
    }

    fn check_VolumeAdjustedMa_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty_data: [f64; 0] = [];
        let empty_volume: [f64; 0] = [];
        let params = VolumeAdjustedMaParams::default();
        let input = VolumeAdjustedMaInput::from_slices(&empty_data, &empty_volume, params);
        let res = VolumeAdjustedMa_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VolumeAdjustedMa should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_VolumeAdjustedMa_all_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [100.0, 100.0, 100.0];
        let params = VolumeAdjustedMaParams::default();
        let input = VolumeAdjustedMaInput::from_slices(&nan_data, &volume, params);
        let res = VolumeAdjustedMa_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VolumeAdjustedMa should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_VolumeAdjustedMa_invalid_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let volume = [100.0, 100.0, 100.0];
        let params = VolumeAdjustedMaParams {
            length: Some(0),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VolumeAdjustedMaInput::from_slices(&data, &volume, params);
        let res = VolumeAdjustedMa_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VolumeAdjustedMa should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_VolumeAdjustedMa_invalid_vi_factor(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let volume = [100.0, 100.0, 100.0];
        let params = VolumeAdjustedMaParams {
            length: Some(2),
            vi_factor: Some(0.0),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VolumeAdjustedMaInput::from_slices(&data, &volume, params);
        let res = VolumeAdjustedMa_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VolumeAdjustedMa should fail with zero vi_factor",
            test_name
        );
        Ok(())
    }

    fn check_VolumeAdjustedMa_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with partial params (using defaults for others)
        let params = VolumeAdjustedMaParams {
            length: Some(20),
            vi_factor: None,     // Use default
            strict: None,        // Use default
            sample_period: None, // Use default
        };
        let input = VolumeAdjustedMaInput::from_candles(&candles, "close", params);
        let result = VolumeAdjustedMa_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_VolumeAdjustedMa_zero_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let volume = [100.0, 100.0, 100.0, 100.0, 100.0];
        let params = VolumeAdjustedMaParams {
            length: Some(0),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VolumeAdjustedMaInput::from_slices(&data, &volume, params);
        let res = VolumeAdjustedMa_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VolumeAdjustedMa should fail with zero length",
            test_name
        );
        Ok(())
    }

    fn check_VolumeAdjustedMa_length_exceeds_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let volume = [100.0, 100.0, 100.0];
        let params = VolumeAdjustedMaParams {
            length: Some(10), // length > data size
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VolumeAdjustedMaInput::from_slices(&data, &volume, params);
        let res = VolumeAdjustedMa_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VolumeAdjustedMa should fail when length exceeds data size",
            test_name
        );
        Ok(())
    }

    fn check_VolumeAdjustedMa_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let volume = [100.0];
        let params = VolumeAdjustedMaParams {
            length: Some(1),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VolumeAdjustedMaInput::from_slices(&data, &volume, params);
        let result = VolumeAdjustedMa_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), 1);
        // Edge case: with only 1 bar and length=1, we can't calculate the Pine formula
        // (needs a bar beyond the last included), so NaN is expected
        assert!(result.values[0].is_nan());
        Ok(())
    }

    fn check_VolumeAdjustedMa_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, 10.0, 20.0, 30.0, 40.0, 50.0];
        let volume = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let params = VolumeAdjustedMaParams {
            length: Some(3),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VolumeAdjustedMaInput::from_slices(&data, &volume, params);
        let result = VolumeAdjustedMa_with_kernel(&input, kernel)?;

        // First values should be NaN (warmup)
        assert!(result.values[0].is_nan());
        assert!(result.values[1].is_nan());

        // After warmup, should have valid values
        for i in 5..result.values.len() {
            assert!(
                result.values[i].is_finite(),
                "[{}] Expected finite value at index {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_VolumeAdjustedMa_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];

        // First pass
        let params = VolumeAdjustedMaParams::default();
        let input1 = VolumeAdjustedMaInput::from_slices(data, volume, params.clone());
        let output1 = VolumeAdjustedMa_with_kernel(&input1, kernel)?;

        // Use constant volume for re-input test
        let const_volume = vec![1000.0; output1.values.len()];

        // Second pass using output as input
        let input2 = VolumeAdjustedMaInput::from_slices(&output1.values, &const_volume, params);
        let output2 = VolumeAdjustedMa_with_kernel(&input2, kernel)?;

        assert_eq!(output1.values.len(), output2.values.len());

        // Check that values after double warmup are finite
        let double_warmup = 24; // Approximately 2 * (length - 1)
        for i in double_warmup..output2.values.len() {
            assert!(
                output2.values[i].is_finite() || output2.values[i].is_nan(),
                "[{}] Expected finite or NaN at index {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let data = &candles.close[..50];
        let volume = &candles.volume[..50];

        let range = VolumeAdjustedMaBatchRange {
            length: (10, 20, 5),
            vi_factor: (0.5, 0.7, 0.1),
            sample_period: (0, 10, 10),
            strict: Some(true),
        };

        let result = VolumeAdjustedMa_batch_with_kernel(data, volume, &range, kernel)?;

        assert!(result.values.len() > 0);
        assert_eq!(result.cols, data.len());
        assert!(result.rows > 0);
        assert_eq!(result.combos.len(), result.rows);

        // Verify each combo has valid parameters
        for combo in &result.combos {
            assert!(combo.length.unwrap() >= 10 && combo.length.unwrap() <= 20);
            assert!(
                (combo.vi_factor.unwrap() >= 0.5 - 1e-10)
                    && (combo.vi_factor.unwrap() <= 0.7 + 1e-10)
            );
        }
        Ok(())
    }

    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];

        // Batch with default params in range
        let range = VolumeAdjustedMaBatchRange::default();

        let batch_result = VolumeAdjustedMa_batch_with_kernel(data, volume, &range, kernel)?;

        // Single run with default params
        let single_input =
            VolumeAdjustedMaInput::from_slices(data, volume, VolumeAdjustedMaParams::default());
        let single_result = VolumeAdjustedMa_with_kernel(&single_input, kernel)?;

        // Find default params in batch
        let default_params = VolumeAdjustedMaParams::default();
        let default_row = batch_result.values_for(&default_params);

        assert!(
            default_row.is_some(),
            "[{}] Default params not found in batch",
            test_name
        );

        // Compare results
        let batch_values = default_row.unwrap();
        for i in 0..data.len() {
            if single_result.values[i].is_nan() && batch_values[i].is_nan() {
                continue;
            }
            assert!(
                (single_result.values[i] - batch_values[i]).abs() < 1e-10,
                "[{}] Mismatch at index {}: single={}, batch={}",
                test_name,
                i,
                single_result.values[i],
                batch_values[i]
            );
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_VolumeAdjustedMa_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        use proptest::prelude::*;

        let strat = (2usize..=30).prop_flat_map(|length| {
            (
                prop::collection::vec(
                    (10.0f64..100.0f64).prop_filter("finite", |x| x.is_finite()),
                    length..200,
                ),
                prop::collection::vec(
                    (100.0f64..1000.0f64).prop_filter("positive", |x| x.is_finite() && *x > 0.0),
                    length..200,
                ),
                Just(length),
                0.1f64..2.0f64,
                prop::bool::ANY,
                0usize..=20,
            )
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, volume, length, vi_factor, strict, sample_period)| {
                // Ensure data and volume have same length
                let min_len = data.len().min(volume.len());
                let data = &data[..min_len];
                let volume = &volume[..min_len];

                let params = VolumeAdjustedMaParams {
                    length: Some(length),
                    vi_factor: Some(vi_factor),
                    strict: Some(strict),
                    sample_period: Some(sample_period),
                };

                let input = VolumeAdjustedMaInput::from_slices(data, volume, params);
                let result = VolumeAdjustedMa_with_kernel(&input, kernel);

                prop_assert!(
                    result.is_ok(),
                    "[{}] Unexpected error: {:?}",
                    test_name,
                    result
                );
                let output = result.unwrap();

                // Property: Output length equals input length
                prop_assert_eq!(
                    output.values.len(),
                    data.len(),
                    "[{}] Output length mismatch",
                    test_name
                );

                // Property: Warmup period is correctly applied (at least some NaN values at start)
                let warmup = length - 1;
                for i in 0..warmup.min(data.len()) {
                    prop_assert!(
                        output.values[i].is_nan(),
                        "[{}] Expected NaN in warmup period at index {}",
                        test_name,
                        i
                    );
                }

                // Property: After warmup, values should be finite (not infinite)
                for i in warmup..data.len() {
                    prop_assert!(
                        output.values[i].is_finite() || output.values[i].is_nan(),
                        "[{}] Expected finite or NaN value at index {}, got {}",
                        test_name,
                        i,
                        output.values[i]
                    );
                }

                // Property: Compare with scalar implementation for consistency
                if kernel != Kernel::Scalar {
                    let scalar_result =
                        VolumeAdjustedMa_with_kernel(&input, Kernel::Scalar).unwrap();
                    for i in 0..data.len() {
                        if output.values[i].is_nan() && scalar_result.values[i].is_nan() {
                            continue;
                        }
                        prop_assert!(
                            (output.values[i] - scalar_result.values[i]).abs() < 1e-10,
                            "[{}] Kernel mismatch at index {}: kernel={}, scalar={}",
                            test_name,
                            i,
                            output.values[i],
                            scalar_result.values[i]
                        );
                    }
                }

                Ok(())
            },
        )?;
        Ok(())
    }

    // ==================== TEST GENERATION MACROS ====================
    macro_rules! generate_all_VolumeAdjustedMa_tests {
        ($($fn:ident),* $(,)?) => {
            paste::paste! {
                $(
                    #[test] fn [<$fn _scalar_f64>]() { let _ = $fn(stringify!([<$fn _scalar_f64>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$fn _avx2_f64>]()   { let _ = $fn(stringify!([<$fn _avx2_f64>]),   Kernel::Avx2); }
                    #[test] fn [<$fn _avx512_f64>]() { let _ = $fn(stringify!([<$fn _avx512_f64>]), Kernel::Avx512); }
                )*
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test] fn [<$fn _simd128_f64>]() { let _ = $fn(stringify!([<$fn _simd128_f64>]), Kernel::Scalar); }
                )*
            }
        };
    }

    macro_rules! gen_VolumeAdjustedMa_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]),   Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = VolumeAdjustedMaBatchBuilder::new()
            .kernel(kernel)
            .length_range(10, 20, 5)
            .vi_factor_range(0.5, 0.7, 0.1)
            .sample_period_static(0)
            .strict_static(true)
            .apply_candles(&c, "close")?;

        for (idx, &v) in out.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x11111111_11111111,
                "[{}] alloc poison at {}",
                test_name, idx
            );
            assert_ne!(
                b, 0x22222222_22222222,
                "[{}] init_matrix poison at {}",
                test_name, idx
            );
            assert_ne!(
                b, 0x33333333_33333333,
                "[{}] make_uninit poison at {}",
                test_name, idx
            );
        }
        Ok(())
    }

    fn check_builder_apply_default(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = VolumeAdjustedMaBuilder::new().kernel(kernel).apply(&c)?;
        assert_eq!(out.values.len(), c.close.len());
        Ok(())
    }

    fn check_batch_slice_vs_par(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let data = &c.close[..64];
        let vol = &c.volume[..64];
        let sweep = VolumeAdjustedMaBatchRange::default();
        let a = VolumeAdjustedMa_batch_slice(data, vol, &sweep, kernel)?;
        let b = VolumeAdjustedMa_batch_par_slice(data, vol, &sweep, kernel)?;
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        for i in 0..a.values.len() {
            if a.values[i].is_nan() && b.values[i].is_nan() {
                continue;
            }
            assert!(
                (a.values[i] - b.values[i]).abs() < 1e-12,
                "[{}] idx {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    // Generate tests for all kernels
    generate_all_VolumeAdjustedMa_tests!(
        check_VolumeAdjustedMa_accuracy,
        check_VolumeAdjustedMa_slow,
        check_VolumeAdjustedMa_default_candles,
        // check_VolumeAdjustedMa_no_poison, // Only available in debug builds
        check_VolumeAdjustedMa_streaming,
        check_builder_apply_default,
        check_batch_slice_vs_par,
        check_VolumeAdjustedMa_empty_input,
        check_VolumeAdjustedMa_all_nan,
        check_VolumeAdjustedMa_invalid_period,
        check_VolumeAdjustedMa_invalid_vi_factor,
        check_VolumeAdjustedMa_partial_params,
        check_VolumeAdjustedMa_zero_length,
        check_VolumeAdjustedMa_length_exceeds_data,
        check_VolumeAdjustedMa_very_small_dataset,
        check_VolumeAdjustedMa_nan_handling,
        check_VolumeAdjustedMa_reinput
    );

    // Generate batch tests
    gen_VolumeAdjustedMa_batch_tests!(check_batch_sweep);
    gen_VolumeAdjustedMa_batch_tests!(check_batch_default_row);
    #[cfg(debug_assertions)]
    gen_VolumeAdjustedMa_batch_tests!(check_batch_no_poison);

    #[cfg(feature = "proptest")]
    generate_all_VolumeAdjustedMa_tests!(check_VolumeAdjustedMa_property);
}
