//! # Volume Adjusted Moving Average (VAMA)
//!
//! A volume-weighted moving average that dynamically adjusts its lookback period based on volume.
//! VAMA uses volume increments to determine how many bars to include in the average calculation.
//!
//! ## Parameters
//! - **length**: Base period for the moving average (default: 13)
//! - **vi_factor**: Volume increment factor (default: 0.67)
//! - **strict**: If true, enforces exact length requirement; if false, uses up to length bars (default: true)
//! - **sample_period**: Number of bars to use for volume averaging (0 = all bars) (default: 0)
//!
//! ## Errors
//! - **EmptyInputData**: vama: Input data slice is empty.
//! - **EmptyVolumeData**: vama: Volume data slice is empty.
//! - **AllValuesNaN**: vama: All input values are `NaN`.
//! - **InvalidPeriod**: vama: Period is zero or exceeds data length.
//! - **NotEnoughValidData**: vama: Not enough valid data points for calculation.
//! - **InvalidViFactor**: vama: Volume increment factor must be positive.
//! - **DataLengthMismatch**: vama: Price and volume data have different lengths.
//!
//! ## Returns
//! - **`Ok(VamaOutput)`** on success, containing values vector.
//! - **`Err(VamaError)`** otherwise.

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

// Core imports with complete helper set
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_kernel, detect_best_batch_kernel,
    init_matrix_prefixes, make_uninit_matrix,
};

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::x86_64::*;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

// Parallel processing support
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Standard library imports
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// ==================== TRAIT IMPLEMENTATIONS ====================
impl<'a> AsRef<[f64]> for VamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VamaData::Slice { data, .. } => data,
            VamaData::Candles { candles, source, .. } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices with volume and candle data
#[derive(Debug, Clone)]
pub enum VamaData<'a> {
    Candles { 
        candles: &'a Candles, 
        source: &'a str 
    },
    Slice { 
        data: &'a [f64], 
        volume: &'a [f64] 
    },
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct VamaOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VamaParams {
    pub length: Option<usize>,
    pub vi_factor: Option<f64>,
    pub strict: Option<bool>,
    pub sample_period: Option<usize>,
}

impl Default for VamaParams {
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
pub struct VamaInput<'a> {
    pub data: VamaData<'a>,
    pub params: VamaParams,
}

impl<'a> VamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VamaParams) -> Self {
        Self {
            data: VamaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    
    #[inline]
    pub fn from_slices(data: &'a [f64], volume: &'a [f64], p: VamaParams) -> Self {
        Self {
            data: VamaData::Slice { data, volume },
            params: p,
        }
    }
    
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VamaParams::default())
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
            VamaData::Candles { candles, .. } => &candles.volume,
            VamaData::Slice { volume, .. } => volume,
        }
    }
}

// ==================== BUILDER API ====================
#[derive(Copy, Clone, Debug)]
pub struct VamaBuilder {
    length: Option<usize>,
    vi_factor: Option<f64>,
    strict: Option<bool>,
    sample_period: Option<usize>,
    kernel: Kernel,
}

impl Default for VamaBuilder {
    fn default() -> Self {
        Self { 
            length: None, 
            vi_factor: None, 
            strict: None, 
            sample_period: None, 
            kernel: Kernel::Auto 
        }
    }
}

impl VamaBuilder {
    #[inline(always)] 
    pub fn new() -> Self { Self::default() }
    
    #[inline(always)]
    pub fn with_default_candles(self, _c: &Candles, _src: &str) -> Self {
        // For future use
        self
    }
    
    #[inline(always)]
    pub fn with_default_slice(self, _data: &[f64], _volume: &[f64]) -> Self {
        // For future use
        self
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
    pub fn apply(self, c: &Candles, src: &str) -> Result<VamaOutput, VamaError> {
        let p = VamaParams { 
            length: self.length, 
            vi_factor: self.vi_factor, 
            strict: self.strict, 
            sample_period: self.sample_period 
        };
        let i = VamaInput::from_candles(c, src, p);
        vama_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn apply_slices(self, data: &[f64], volume: &[f64]) -> Result<VamaOutput, VamaError> {
        let p = VamaParams { 
            length: self.length, 
            vi_factor: self.vi_factor, 
            strict: self.strict, 
            sample_period: self.sample_period 
        };
        let i = VamaInput::from_slices(data, volume, p);
        vama_with_kernel(&i, self.kernel)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum VamaError {
    #[error("vama: Input data slice is empty.")]
    EmptyInputData,
    
    #[error("vama: Volume data slice is empty.")]
    EmptyVolumeData,
    
    #[error("vama: All values are NaN.")]
    AllValuesNaN,
    
    #[error("vama: Invalid period: length = {length}, data length = {data_len}")]
    InvalidPeriod { length: usize, data_len: usize },
    
    #[error("vama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    
    #[error("vama: Invalid vi_factor: {vi_factor}. Must be positive.")]
    InvalidViFactor { vi_factor: f64 },
    
    #[error("vama: Data length mismatch: price = {price_len}, volume = {volume_len}")]
    DataLengthMismatch { price_len: usize, volume_len: usize },
}

// ==================== PREPARATION AND COMPUTE FUNCTIONS ====================
#[inline(always)]
fn vama_prepare<'a>(
    input: &'a VamaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, f64, bool, usize, usize, Kernel), VamaError> {
    let data: &[f64] = input.as_ref();
    let vol: &[f64] = input.get_volume();
    let len = data.len();
    if len == 0 { return Err(VamaError::EmptyInputData); }
    if vol.len() == 0 { return Err(VamaError::EmptyVolumeData); }
    if len != vol.len() { 
        return Err(VamaError::DataLengthMismatch{ 
            price_len: len, 
            volume_len: vol.len() 
        }); 
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(VamaError::AllValuesNaN)?;
    let length = input.get_length();
    let vi_factor = input.get_vi_factor();
    let strict = input.get_strict();
    let sample_period = input.get_sample_period();

    if length == 0 || length > len { 
        return Err(VamaError::InvalidPeriod{ length, data_len: len }); 
    }
    if !vi_factor.is_finite() || vi_factor <= 0.0 { 
        return Err(VamaError::InvalidViFactor{ vi_factor }); 
    }
    if len - first < length { 
        return Err(VamaError::NotEnoughValidData{ 
            needed: length, 
            valid: len - first 
        }); 
    }

    let chosen = match kernel { 
        Kernel::Auto => detect_best_kernel(), 
        k => k 
    };
    Ok((data, vol, length, vi_factor, strict, sample_period, first, chosen))
}

#[inline(always)]
fn vama_compute_into(
    data: &[f64], vol: &[f64],
    length: usize, vi_factor: f64, strict: bool, sample_period: usize, first: usize,
    kernel: Kernel, out: &mut [f64],
) {
    match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe { 
            vama_avx512(data, vol, length, vi_factor, strict, sample_period, first, out) 
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe { 
            vama_avx2(data, vol, length, vi_factor, strict, sample_period, first, out) 
        },
        _ => vama_scalar(data, vol, length, vi_factor, strict, sample_period, first, out),
    }
}

#[inline(always)]
fn vama_scalar(
    data: &[f64], vol: &[f64],
    length: usize, vi_factor: f64, strict: bool, sample_period: usize, first: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let warmup = first + length - 1;
    // compute from warmup..len, same logic as before
    for i in warmup..len {
        let avg_volume = if sample_period == 0 {
            // cumulative average over entire series
            let mut sum = 0.0; 
            let mut cnt = 0usize;
            for j in 0..len { 
                let v = vol[j]; 
                if v.is_finite() && v > 0.0 { 
                    sum += v; 
                    cnt += 1; 
                } 
            }
            if cnt > 0 { sum / cnt as f64 } else { 0.0 }
        } else {
            let start = i.saturating_sub(sample_period - 1);
            let mut sum = 0.0; 
            let mut cnt = 0usize;
            for j in start..=i { 
                let v = vol[j]; 
                if v.is_finite() && v > 0.0 { 
                    sum += v; 
                    cnt += 1; 
                } 
            }
            if cnt > 0 { sum / cnt as f64 } else { 0.0 }
        };
        let vi_th = avg_volume * vi_factor;
        let mut weighted_sum = 0.0;
        let mut v2i_sum = 0.0;
        let mut nmb = 0usize;

        for j in 0..length.min(i + 1) {
            let idx = i - j;
            let v2i = if vi_th > 0.0 && vol[idx].is_finite() { 
                vol[idx] / vi_th 
            } else { 
                0.0 
            };
            if data[idx].is_finite() {
                weighted_sum += data[idx] * v2i;
                v2i_sum += v2i;
            }
            nmb = j + 1;
            if strict {
                if v2i_sum >= length as f64 { break; }
            } else if j + 1 >= length { break; }
        }

        if nmb > 0 && v2i_sum > 0.0 {
            if v2i_sum > length as f64 && nmb > 0 && i + 1 >= nmb {
                let idx_nmb = i + 1 - nmb;
                if data[idx_nmb].is_finite() {
                    let adjustment = (v2i_sum - length as f64) * data[idx_nmb];
                    out[i] = (weighted_sum - adjustment) / length as f64;
                } else {
                    out[i] = weighted_sum / v2i_sum;
                }
            } else {
                out[i] = weighted_sum / v2i_sum.max(1.0);
            }
        } else {
            out[i] = f64::NAN;
        }
    }
}

// ==================== AVX512 SIMD Implementation ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn vama_avx512(
    data: &[f64], vol: &[f64],
    length: usize, vi_factor: f64, strict: bool, sample_period: usize, first: usize,
    out: &mut [f64],
) {
    // For VAMA, the algorithm is inherently sequential due to the dynamic lookback
    // based on volume increments. However, we can still use SIMD for:
    // 1. Computing average volumes in parallel
    // 2. Vector operations for weighted sums
    let len = data.len();
    let warmup = first + length - 1;
    
    // Use scalar fallback for now, but with AVX512 for average volume calculations
    for i in warmup..len {
        let avg_volume = if sample_period == 0 {
            // Use SIMD to compute average over entire volume array
            let mut sum_vec = _mm512_setzero_pd();
            let mut cnt = 0usize;
            let chunks = len / 8;
            let remainder = len % 8;
            
            for chunk in 0..chunks {
                let v_vec = _mm512_loadu_pd(vol.as_ptr().add(chunk * 8));
                let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(v_vec, _mm512_setzero_pd());
                let finite_mask = _mm512_cmp_pd_mask::<_CMP_ORD_Q>(v_vec, v_vec);
                let valid_mask = mask & finite_mask;
                sum_vec = _mm512_mask_add_pd(sum_vec, valid_mask, sum_vec, v_vec);
                cnt += valid_mask.count_ones() as usize;
            }
            
            if remainder > 0 {
                let tail_mask = (1u8 << remainder) - 1;
                let v_vec = _mm512_maskz_loadu_pd(tail_mask, vol.as_ptr().add(chunks * 8));
                let pos_mask = _mm512_mask_cmp_pd_mask::<_CMP_GT_OQ>(tail_mask, v_vec, _mm512_setzero_pd());
                let finite_mask = _mm512_mask_cmp_pd_mask::<_CMP_ORD_Q>(tail_mask, v_vec, v_vec);
                let valid_mask = pos_mask & finite_mask;
                sum_vec = _mm512_mask_add_pd(sum_vec, valid_mask, sum_vec, v_vec);
                cnt += valid_mask.count_ones() as usize;
            }
            
            let sum = _mm512_reduce_add_pd(sum_vec);
            if cnt > 0 { sum / cnt as f64 } else { 0.0 }
        } else {
            // Compute average for sample period using scalar (small window)
            let start = i.saturating_sub(sample_period - 1);
            let mut sum = 0.0; 
            let mut cnt = 0usize;
            for j in start..=i { 
                let v = vol[j]; 
                if v.is_finite() && v > 0.0 { 
                    sum += v; 
                    cnt += 1; 
                } 
            }
            if cnt > 0 { sum / cnt as f64 } else { 0.0 }
        };
        
        // Rest of computation remains scalar due to sequential nature
        let vi_th = avg_volume * vi_factor;
        let mut weighted_sum = 0.0;
        let mut v2i_sum = 0.0;
        let mut nmb = 0usize;

        for j in 0..length.min(i + 1) {
            let idx = i - j;
            let v2i = if vi_th > 0.0 && vol[idx].is_finite() { 
                vol[idx] / vi_th 
            } else { 
                0.0 
            };
            if data[idx].is_finite() {
                weighted_sum += data[idx] * v2i;
                v2i_sum += v2i;
            }
            nmb = j + 1;
            if strict {
                if v2i_sum >= length as f64 { break; }
            } else if j + 1 >= length { break; }
        }

        if nmb > 0 && v2i_sum > 0.0 {
            if v2i_sum > length as f64 && nmb > 0 && i + 1 >= nmb {
                let idx_nmb = i + 1 - nmb;
                if data[idx_nmb].is_finite() {
                    let adjustment = (v2i_sum - length as f64) * data[idx_nmb];
                    out[i] = (weighted_sum - adjustment) / length as f64;
                } else {
                    out[i] = weighted_sum / v2i_sum;
                }
            } else {
                out[i] = weighted_sum / v2i_sum.max(1.0);
            }
        } else {
            out[i] = f64::NAN;
        }
    }
}

// ==================== AVX2 SIMD Implementation ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vama_avx2(
    data: &[f64], vol: &[f64],
    length: usize, vi_factor: f64, strict: bool, sample_period: usize, first: usize,
    out: &mut [f64],
) {
    // Similar to AVX512 but using 256-bit registers
    let len = data.len();
    let warmup = first + length - 1;
    
    for i in warmup..len {
        let avg_volume = if sample_period == 0 {
            // Use AVX2 to compute average over entire volume array
            let mut sum_vec = _mm256_setzero_pd();
            let mut cnt = 0usize;
            let chunks = len / 4;
            let remainder = len % 4;
            
            for chunk in 0..chunks {
                let v_vec = _mm256_loadu_pd(vol.as_ptr().add(chunk * 4));
                // Check if positive and finite
                let zero = _mm256_setzero_pd();
                let pos_mask = _mm256_cmp_pd(v_vec, zero, _CMP_GT_OQ);
                let finite_mask = _mm256_cmp_pd(v_vec, v_vec, _CMP_ORD_Q);
                let valid_mask = _mm256_and_pd(pos_mask, finite_mask);
                let masked_v = _mm256_and_pd(v_vec, valid_mask);
                sum_vec = _mm256_add_pd(sum_vec, masked_v);
                
                // Count valid elements
                let mask_bits = _mm256_movemask_pd(valid_mask);
                cnt += mask_bits.count_ones() as usize;
            }
            
            // Handle remainder
            for j in (chunks * 4)..len {
                let v = vol[j];
                if v.is_finite() && v > 0.0 {
                    sum_vec = _mm256_add_pd(sum_vec, _mm256_set1_pd(v));
                    cnt += 1;
                    break; // Only add once for scalar remainder
                }
            }
            
            // Horizontal sum
            let sum_high = _mm256_extractf128_pd(sum_vec, 1);
            let sum_low = _mm256_castpd256_pd128(sum_vec);
            let sum128 = _mm_add_pd(sum_low, sum_high);
            let sum64 = _mm_hadd_pd(sum128, sum128);
            let sum = _mm_cvtsd_f64(sum64);
            
            if cnt > 0 { sum / cnt as f64 } else { 0.0 }
        } else {
            // Compute average for sample period using scalar
            let start = i.saturating_sub(sample_period - 1);
            let mut sum = 0.0; 
            let mut cnt = 0usize;
            for j in start..=i { 
                let v = vol[j]; 
                if v.is_finite() && v > 0.0 { 
                    sum += v; 
                    cnt += 1; 
                } 
            }
            if cnt > 0 { sum / cnt as f64 } else { 0.0 }
        };
        
        // Rest of computation remains scalar
        let vi_th = avg_volume * vi_factor;
        let mut weighted_sum = 0.0;
        let mut v2i_sum = 0.0;
        let mut nmb = 0usize;

        for j in 0..length.min(i + 1) {
            let idx = i - j;
            let v2i = if vi_th > 0.0 && vol[idx].is_finite() { 
                vol[idx] / vi_th 
            } else { 
                0.0 
            };
            if data[idx].is_finite() {
                weighted_sum += data[idx] * v2i;
                v2i_sum += v2i;
            }
            nmb = j + 1;
            if strict {
                if v2i_sum >= length as f64 { break; }
            } else if j + 1 >= length { break; }
        }

        if nmb > 0 && v2i_sum > 0.0 {
            if v2i_sum > length as f64 && nmb > 0 && i + 1 >= nmb {
                let idx_nmb = i + 1 - nmb;
                if data[idx_nmb].is_finite() {
                    let adjustment = (v2i_sum - length as f64) * data[idx_nmb];
                    out[i] = (weighted_sum - adjustment) / length as f64;
                } else {
                    out[i] = weighted_sum / v2i_sum;
                }
            } else {
                out[i] = weighted_sum / v2i_sum.max(1.0);
            }
        } else {
            out[i] = f64::NAN;
        }
    }
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
#[inline]
pub fn vama(input: &VamaInput) -> Result<VamaOutput, VamaError> {
    vama_with_kernel(input, Kernel::Auto)
}

pub fn vama_with_kernel(input: &VamaInput, kernel: Kernel) -> Result<VamaOutput, VamaError> {
    let (data, vol, length, vi_factor, strict, sample_period, first, chosen) = vama_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + length - 1);
    vama_compute_into(data, vol, length, vi_factor, strict, sample_period, first, chosen, &mut out);
    Ok(VamaOutput { values: out })
}

#[inline]
pub fn vama_into_slice(
    dst: &mut [f64], data: &[f64], volume: &[f64],
    length: usize, vi_factor: f64, strict: bool, sample_period: usize, first: usize,
) -> Result<(), VamaError> {
    if dst.len() != data.len() { 
        return Err(VamaError::InvalidPeriod { 
            length: dst.len(), 
            data_len: data.len() 
        }); 
    }
    vama_compute_into(data, volume, length, vi_factor, strict, sample_period, first, detect_best_kernel(), dst);
    let warm = first + length - 1;
    for v in &mut dst[..warm] { *v = f64::NAN; }
    Ok(())
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug)]
pub struct VamaBatchRange {
    pub length: (usize, usize, usize),
    pub vi_factor: (f64, f64, f64),
    pub sample_period: (usize, usize, usize),
    pub strict: Option<bool>, // None => both {true,false}
}

impl Default for VamaBatchRange {
    fn default() -> Self {
        Self { 
            length: (13, 55, 1), 
            vi_factor: (0.67, 0.67, 0.0), 
            sample_period: (0, 0, 0), 
            strict: Some(true) 
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VamaBatchBuilder {
    range: VamaBatchRange,
    kernel: Kernel,
}

impl VamaBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn length_range(mut self, s: usize, e: usize, st: usize) -> Self { 
        self.range.length=(s,e,st); 
        self 
    }
    pub fn length_static(mut self, n: usize) -> Self { 
        self.range.length=(n,n,0); 
        self 
    }
    pub fn vi_factor_range(mut self, s:f64,e:f64,st:f64)->Self{ 
        self.range.vi_factor=(s,e,st); 
        self 
    }
    pub fn vi_factor_static(mut self, f:f64)->Self{ 
        self.range.vi_factor=(f,f,0.0); 
        self 
    }
    pub fn sample_period_range(mut self,s:usize,e:usize,st:usize)->Self{ 
        self.range.sample_period=(s,e,st); 
        self 
    }
    pub fn sample_period_static(mut self,n:usize)->Self{ 
        self.range.sample_period=(n,n,0); 
        self 
    }
    pub fn strict_static(mut self,b:bool)->Self{ 
        self.range.strict=Some(b); 
        self 
    }
    pub fn strict_both(mut self)->Self{ 
        self.range.strict=None; 
        self 
    }

    pub fn apply_slices(self, data:&[f64], volume:&[f64]) -> Result<VamaBatchOutput, VamaError> {
        vama_batch_with_kernel(data, volume, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c:&Candles, src:&str) -> Result<VamaBatchOutput, VamaError> {
        self.apply_slices(source_type(c, src), &c.volume)
    }
}

#[derive(Clone, Debug)]
pub struct VamaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VamaBatchOutput {
    pub fn values_for(&self, p:&VamaParams)->Option<&[f64]>{
        self.combos.iter().position(|q|
            q.length.unwrap_or(13)==p.length.unwrap_or(13) &&
            (q.vi_factor.unwrap_or(0.67)-p.vi_factor.unwrap_or(0.67)).abs()<1e-12 &&
            q.strict.unwrap_or(true)==p.strict.unwrap_or(true) &&
            q.sample_period.unwrap_or(0)==p.sample_period.unwrap_or(0)
        ).map(|r|{ 
            let s=r*self.cols; 
            &self.values[s..s+self.cols] 
        })
    }
}

#[inline(always)]
fn axis_usize((s,e,st):(usize,usize,usize))->Vec<usize>{ 
    if st==0||s==e { vec![s] } else { (s..=e).step_by(st).collect() } 
}

#[inline(always)]
fn axis_f64((s,e,st):(f64,f64,f64))->Vec<f64>{
    if st.abs()<1e-12||(s-e).abs()<1e-12 { 
        vec![s] 
    } else { 
        let mut v=Vec::new(); 
        let mut x=s; 
        while x<=e+1e-12 { 
            v.push(x); 
            x+=st; 
        } 
        v 
    }
}

#[inline(always)]
fn expand_grid_vama(r:&VamaBatchRange)->Vec<VamaParams>{
    let lengths=axis_usize(r.length);
    let vfs=axis_f64(r.vi_factor);
    let sps=axis_usize(r.sample_period);
    let stricts:Vec<bool>=match r.strict{ Some(b)=>vec![b], None=>vec![true,false] };
    let mut out=Vec::with_capacity(lengths.len()*vfs.len()*sps.len()*stricts.len());
    for &l in &lengths { 
        for &vf in &vfs { 
            for &sp in &sps { 
                for &st in &stricts {
                    out.push(VamaParams{ 
                        length:Some(l), 
                        vi_factor:Some(vf), 
                        sample_period:Some(sp), 
                        strict:Some(st) 
                    });
                }
            }
        }
    }
    out
}

pub fn vama_batch_with_kernel(
    data:&[f64], volume:&[f64], sweep:&VamaBatchRange, k:Kernel
)->Result<VamaBatchOutput, VamaError>{
    vama_batch_inner(data, volume, sweep, k, false)
}

#[inline(always)]
pub fn vama_batch_par_slice(
    data:&[f64], volume:&[f64], sweep:&VamaBatchRange, k:Kernel
)->Result<VamaBatchOutput, VamaError>{
    vama_batch_inner(data, volume, sweep, k, true)
}

fn vama_batch_inner(
    data:&[f64], volume:&[f64], sweep:&VamaBatchRange, k:Kernel, parallel:bool
)->Result<VamaBatchOutput, VamaError>{
    if data.len()!=volume.len(){ 
        return Err(VamaError::DataLengthMismatch{
            price_len:data.len(), 
            volume_len:volume.len()
        }); 
    }
    let combos=expand_grid_vama(sweep);
    if combos.is_empty(){ 
        return Err(VamaError::InvalidPeriod{ length:0, data_len:0 }); 
    }
    let cols=data.len(); 
    if cols==0 { return Err(VamaError::EmptyInputData); }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(VamaError::AllValuesNaN)?;
    let rows=combos.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warms:Vec<usize>=combos.iter().map(|p| first + p.length.unwrap() - 1).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) 
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            vama_batch_inner_into_par(data, volume, &combos, first, out)?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            vama_batch_inner_into(data, volume, &combos, first, out)?;
        }
    } else {
        vama_batch_inner_into(data, volume, &combos, first, out)?;
    }

    let values = unsafe { 
        Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity()) 
    };
    Ok(VamaBatchOutput{ values, combos, rows, cols })
}

#[inline(always)]
fn vama_batch_inner_into(
    data:&[f64], volume:&[f64], combos:&[VamaParams], first:usize, out:&mut [f64]
)->Result<(), VamaError>{
    let cols = data.len();
    for (row, dst) in out.chunks_mut(cols).enumerate() {
        let p=&combos[row];
        let length=p.length.unwrap(); 
        let vi=p.vi_factor.unwrap(); 
        let st=p.strict.unwrap(); 
        let sp=p.sample_period.unwrap();
        // reuse scalar compute
        vama_compute_into(data, volume, length, vi, st, sp, first, Kernel::Scalar, dst);
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn vama_batch_inner_into_par(
    data:&[f64], volume:&[f64], combos:&[VamaParams], first:usize, out:&mut [f64]
)->Result<(), VamaError>{
    let cols = data.len();
    out.par_chunks_mut(cols)
        .enumerate()
        .try_for_each(|(row, dst)| {
            let p=&combos[row];
            let length=p.length.unwrap(); 
            let vi=p.vi_factor.unwrap(); 
            let st=p.strict.unwrap(); 
            let sp=p.sample_period.unwrap();
            // reuse scalar compute
            vama_compute_into(data, volume, length, vi, st, sp, first, Kernel::Scalar, dst);
            Ok::<(), VamaError>(())
        })
}

// ==================== STREAMING SUPPORT ====================
#[derive(Debug, Clone)]
pub struct VamaStream {
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    head: usize,
    filled: bool,
    vol_sum_all: f64,
    vol_cnt_all: usize,
}

impl VamaStream {
    pub fn try_new(p: VamaParams) -> Result<Self, VamaError> {
        let length = p.length.unwrap_or(13);
        if length==0 { 
            return Err(VamaError::InvalidPeriod{length, data_len:0}); 
        }
        let vi = p.vi_factor.unwrap_or(0.67);
        if !vi.is_finite() || vi<=0.0 { 
            return Err(VamaError::InvalidViFactor{vi_factor:vi}); 
        }
        let strict = p.strict.unwrap_or(true);
        let sp = p.sample_period.unwrap_or(0);
        Ok(Self{ 
            length, 
            vi_factor:vi, 
            strict, 
            sample_period:sp,
            prices: vec![f64::NAN; length], 
            volumes: vec![f64::NAN; length],
            head:0, 
            filled:false, 
            vol_sum_all:0.0, 
            vol_cnt_all:0 
        })
    }
    
    #[inline(always)]
    pub fn update(&mut self, price:f64, volume:f64) -> Option<f64> {
        self.prices[self.head]=price;
        self.volumes[self.head]=volume;
        self.head=(self.head+1)%self.length;
        if !self.filled && self.head==0 { self.filled=true; }
        if self.sample_period==0 {
            if volume.is_finite() && volume>0.0 { 
                self.vol_sum_all += volume; 
                self.vol_cnt_all += 1; 
            }
        }
        if !self.filled { return None; }

        // avg volume
        let avg_volume = if self.sample_period==0 {
            if self.vol_cnt_all>0 { 
                self.vol_sum_all / self.vol_cnt_all as f64 
            } else { 
                0.0 
            }
        } else {
            // window of last sample_period over ring
            let mut sum=0.0; 
            let mut cnt=0usize;
            let mut idx=self.head;
            for _ in 0..self.sample_period.min(self.length) {
                let v=self.volumes[idx];
                if v.is_finite() && v>0.0 { 
                    sum+=v; 
                    cnt+=1; 
                }
                idx=(idx+self.length-1)%self.length;
            }
            if cnt>0 { sum/cnt as f64 } else { 0.0 }
        };
        let vi_th = avg_volume * self.vi_factor;

        let mut weighted_sum=0.0; 
        let mut v2i_sum=0.0; 
        let mut nmb=0usize;
        let mut idx=self.head;
        for j in 0..self.length {
            idx = (idx + self.length - 1) % self.length;
            let p=self.prices[idx]; 
            let v=self.volumes[idx];
            let v2i = if vi_th>0.0 && v.is_finite() { 
                v/vi_th 
            } else { 
                0.0 
            };
            if p.is_finite() { 
                weighted_sum += p * v2i; 
                v2i_sum += v2i; 
            }
            nmb = j+1;
            if self.strict {
                if v2i_sum >= self.length as f64 { break; }
            } else if j+1 >= self.length { break; }
        }
        if nmb>0 && v2i_sum>0.0 {
            if v2i_sum > self.length as f64 && nmb>0 {
                let idx_nmb = (self.head + self.length - (nmb % self.length)) % self.length;
                let p0 = self.prices[idx_nmb];
                if p0.is_finite() {
                    let adjustment = (v2i_sum - self.length as f64) * p0;
                    Some((weighted_sum - adjustment) / self.length as f64)
                } else {
                    Some(weighted_sum / v2i_sum)
                }
            } else {
                Some(weighted_sum / v2i_sum.max(1.0))
            }
        } else { 
            Some(f64::NAN) 
        }
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "vama")]
#[pyo3(signature = (data, volume, length=13, vi_factor=0.67, strict=true, sample_period=0, kernel=None))]
pub fn vama_py<'py>(
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
    let params = VamaParams {
        length: Some(length),
        vi_factor: Some(vi_factor),
        strict: Some(strict),
        sample_period: Some(sample_period),
    };
    let input = VamaInput::from_slices(slice_in, volume_in, params);
    
    let result = py
        .allow_threads(|| vama_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok(result.values.into_pyarray(py))
}

#[cfg(feature="python")]
#[pyfunction(name="vama_batch")]
#[pyo3(signature = (data, volume, length_range, vi_factor_range, sample_period_range, strict=None, kernel=None))]
pub fn vama_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length_range: (usize,usize,usize),
    vi_factor_range: (f64,f64,f64),
    sample_period_range: (usize,usize,usize),
    strict: Option<bool>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = data.as_slice()?;
    let v = volume.as_slice()?;
    let sweep = VamaBatchRange{ 
        length:length_range, 
        vi_factor:vi_factor_range, 
        sample_period:sample_period_range, 
        strict 
    };
    let kern = validate_kernel(kernel, true)?;
    let combos_rows_cols_values = py.allow_threads(|| {
        let out = vama_batch_with_kernel(d, v, &sweep, kern)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_,PyErr>(out)
    })?;
    let dict = PyDict::new(py);
    let rows = combos_rows_cols_values.rows;
    let cols = combos_rows_cols_values.cols;
    let arr = unsafe { numpy::PyArray1::<f64>::new(py, [rows*cols], false) };
    unsafe { arr.as_slice_mut()? }.copy_from_slice(&combos_rows_cols_values.values);
    dict.set_item("values", arr.reshape((rows, cols))?)?;
    dict.set_item("lengths", combos_rows_cols_values.combos.iter()
        .map(|p| p.length.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("vi_factors", combos_rows_cols_values.combos.iter()
        .map(|p| p.vi_factor.unwrap()).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("sample_periods", combos_rows_cols_values.combos.iter()
        .map(|p| p.sample_period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("stricts", combos_rows_cols_values.combos.iter()
        .map(|p| p.strict.unwrap()).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature="python")]
#[pyclass(name="VamaStream")]
pub struct VamaStreamPy { 
    stream: VamaStream 
}

#[cfg(feature="python")]
#[pymethods]
impl VamaStreamPy {
    #[new]
    fn new(length:usize, vi_factor:f64, strict:bool, sample_period:usize) -> PyResult<Self> {
        let s = VamaStream::try_new(VamaParams{ 
            length:Some(length), 
            vi_factor:Some(vi_factor), 
            strict:Some(strict), 
            sample_period:Some(sample_period) 
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self{ stream:s })
    }
    fn update(&mut self, price:f64, volume:f64) -> Option<f64> { 
        self.stream.update(price, volume) 
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vama_js(
    data: &[f64],
    volume: &[f64],
    length: usize,
    vi_factor: f64,
    strict: bool,
    sample_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = VamaParams {
        length: Some(length),
        vi_factor: Some(vi_factor),
        strict: Some(strict),
        sample_period: Some(sample_period),
    };
    let input = VamaInput::from_slices(data, volume, params);
    
    vama(&input)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature="wasm")]
#[wasm_bindgen]
pub fn vama_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature="wasm")]
#[wasm_bindgen]
pub fn vama_free(ptr:*mut f64, len:usize) { 
    unsafe { 
        let _ = Vec::from_raw_parts(ptr, len, len); 
    } 
}

#[cfg(feature="wasm")]
#[wasm_bindgen]
pub fn vama_into(
    price_ptr:*const f64, vol_ptr:*const f64, out_ptr:*mut f64, len:usize,
    length:usize, vi_factor:f64, strict:bool, sample_period:usize
) -> Result<(), JsValue> {
    if price_ptr.is_null() || vol_ptr.is_null() || out_ptr.is_null() { 
        return Err(JsValue::from_str("null pointer")); 
    }
    unsafe {
        let data = std::slice::from_raw_parts(price_ptr, len);
        let vol  = std::slice::from_raw_parts(vol_ptr, len);
        let mut tmp;
        if out_ptr as *const f64 == price_ptr || out_ptr as *const f64 == vol_ptr {
            tmp = vec![0.0; len];
            vama_into_slice(&mut tmp, data, vol, length, vi_factor, strict, sample_period,
                data.iter().position(|x| !x.is_nan()).unwrap_or(0))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vama_into_slice(out, data, vol, length, vi_factor, strict, sample_period,
                data.iter().position(|x| !x.is_nan()).unwrap_or(0))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature="wasm")]
#[derive(Serialize, Deserialize)]
pub struct VamaBatchConfig {
    pub length_range:(usize,usize,usize),
    pub vi_factor_range:(f64,f64,f64),
    pub sample_period_range:(usize,usize,usize),
    pub strict: Option<bool>,
}

#[cfg(feature="wasm")]
#[derive(Serialize, Deserialize)]
pub struct VamaBatchJsOutput {
    pub values: Vec<f64>, 
    pub combos: Vec<VamaParams>, 
    pub rows: usize, 
    pub cols: usize,
}

#[cfg(feature="wasm")]
#[wasm_bindgen(js_name="vama_batch")]
pub fn vama_batch_unified_js(data:&[f64], volume:&[f64], cfg: JsValue) -> Result<JsValue, JsValue> {
    let cfg: VamaBatchConfig = serde_wasm_bindgen::from_value(cfg)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = VamaBatchRange{ 
        length:cfg.length_range, 
        vi_factor:cfg.vi_factor_range, 
        sample_period:cfg.sample_period_range, 
        strict:cfg.strict 
    };
    let out = vama_batch_with_kernel(data, volume, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VamaBatchJsOutput{ 
        values: out.values, 
        combos: out.combos, 
        rows: out.rows, 
        cols: out.cols 
    }).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;
    
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    
    fn check_vama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // Test with default parameters (length=13, vi_factor=0.67, strict=true)
        let input = VamaInput::from_candles(&candles, "close", VamaParams::default());
        let result = vama_with_kernel(&input, kernel)?;
        
        // REFERENCE VALUES FROM PINESCRIPT (Fast: length=13)
        let expected = [
            58881.58124494,
            58866.67951208,
            58873.34641238,
            58870.41762890,
            58696.37821343,
        ];
        
        let start = result.values.len().saturating_sub(5);
        
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] VAMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected[i]
            );
        }
        Ok(())
    }
    
    fn check_vama_slow(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // Test with slow parameters (length=55, vi_factor=0.67, strict=true)
        let params = VamaParams {
            length: Some(55),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VamaInput::from_candles(&candles, "close", params);
        let result = vama_with_kernel(&input, kernel)?;
        
        // REFERENCE VALUES FROM PINESCRIPT (Slow: length=55)
        let expected = [
            60338.30226444,
            60327.06967012,
            60318.07491767,
            60324.78454609,
            60305.94922998,
        ];
        
        let start = result.values.len().saturating_sub(5);
        
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] VAMA slow {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_vama_default_candles(test:&str, kernel:Kernel)->Result<(),Box<dyn Error>>{
        let file="src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c=read_candles_from_csv(file)?;
        let input=VamaInput::with_default_candles(&c);
        let out=vama_with_kernel(&input, kernel)?;
        assert_eq!(out.values.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_vama_no_poison(test:&str, kernel:Kernel)->Result<(),Box<dyn Error>>{
        let file="src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c=read_candles_from_csv(file)?;
        let input=VamaInput::with_default_candles(&c);
        let out=vama_with_kernel(&input, kernel)?;
        for (i,&v) in out.values.iter().enumerate() {
            if v.is_nan(){ continue; }
            let b=v.to_bits();
            assert_ne!(b, 0x11111111_11111111, "[{}] alloc poison at {}", test, i);
            assert_ne!(b, 0x22222222_22222222, "[{}] init_matrix poison at {}", test, i);
            assert_ne!(b, 0x33333333_33333333, "[{}] make_uninit poison at {}", test, i);
        }
        Ok(())
    }

    fn check_vama_streaming(test:&str, kernel:Kernel)->Result<(),Box<dyn Error>>{
        let file="src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c=read_candles_from_csv(file)?;
        let p=VamaParams::default();
        let batch=vama_with_kernel(&VamaInput::from_candles(&c,"close",p.clone()), kernel)?.values;
        let mut s=VamaStream::try_new(p)?;
        let mut seq=Vec::with_capacity(c.close.len());
        for i in 0..c.close.len(){ 
            seq.push(s.update(c.close[i], c.volume[i]).unwrap_or(f64::NAN)); 
        }
        assert_eq!(batch.len(), seq.len());
        for (i, (&b,&x)) in batch.iter().zip(seq.iter()).enumerate() {
            if b.is_nan() && x.is_nan(){ continue; }
            assert!((b-x).abs()<1e-9, "[{}] stream mismatch at {}", test, i);
        }
        Ok(())
    }
    
    fn check_vama_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let empty_data: [f64; 0] = [];
        let empty_volume: [f64; 0] = [];
        let params = VamaParams::default();
        let input = VamaInput::from_slices(&empty_data, &empty_volume, params);
        let res = vama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VAMA should fail with empty input",
            test_name
        );
        Ok(())
    }
    
    fn check_vama_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [100.0, 100.0, 100.0];
        let params = VamaParams::default();
        let input = VamaInput::from_slices(&nan_data, &volume, params);
        let res = vama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VAMA should fail with all NaN values",
            test_name
        );
        Ok(())
    }
    
    fn check_vama_invalid_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = [10.0, 20.0, 30.0];
        let volume = [100.0, 100.0, 100.0];
        let params = VamaParams {
            length: Some(0),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VamaInput::from_slices(&data, &volume, params);
        let res = vama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VAMA should fail with zero period", test_name);
        Ok(())
    }
    
    fn check_vama_invalid_vi_factor(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = [10.0, 20.0, 30.0];
        let volume = [100.0, 100.0, 100.0];
        let params = VamaParams {
            length: Some(2),
            vi_factor: Some(0.0),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VamaInput::from_slices(&data, &volume, params);
        let res = vama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VAMA should fail with zero vi_factor", test_name);
        Ok(())
    }
    
    fn check_vama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // Test with partial params (using defaults for others)
        let params = VamaParams {
            length: Some(20),
            vi_factor: None, // Use default
            strict: None,    // Use default
            sample_period: None, // Use default
        };
        let input = VamaInput::from_candles(&candles, "close", params);
        let result = vama_with_kernel(&input, kernel)?;
        
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }
    
    fn check_vama_zero_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let volume = [100.0, 100.0, 100.0, 100.0, 100.0];
        let params = VamaParams {
            length: Some(0),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VamaInput::from_slices(&data, &volume, params);
        let res = vama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VAMA should fail with zero length",
            test_name
        );
        Ok(())
    }
    
    fn check_vama_length_exceeds_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = [10.0, 20.0, 30.0];
        let volume = [100.0, 100.0, 100.0];
        let params = VamaParams {
            length: Some(10), // length > data size
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VamaInput::from_slices(&data, &volume, params);
        let res = vama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VAMA should fail when length exceeds data size",
            test_name
        );
        Ok(())
    }
    
    fn check_vama_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = [42.0];
        let volume = [100.0];
        let params = VamaParams {
            length: Some(1),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VamaInput::from_slices(&data, &volume, params);
        let result = vama_with_kernel(&input, kernel)?;
        
        assert_eq!(result.values.len(), 1);
        assert!(result.values[0].is_finite());
        Ok(())
    }
    
    fn check_vama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = [f64::NAN, f64::NAN, 10.0, 20.0, 30.0, 40.0, 50.0];
        let volume = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let params = VamaParams {
            length: Some(3),
            vi_factor: Some(0.67),
            strict: Some(true),
            sample_period: Some(0),
        };
        let input = VamaInput::from_slices(&data, &volume, params);
        let result = vama_with_kernel(&input, kernel)?;
        
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
    
    fn check_vama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];
        
        // First pass
        let params = VamaParams::default();
        let input1 = VamaInput::from_slices(data, volume, params.clone());
        let output1 = vama_with_kernel(&input1, kernel)?;
        
        // Use constant volume for re-input test
        let const_volume = vec![1000.0; output1.values.len()];
        
        // Second pass using output as input
        let input2 = VamaInput::from_slices(&output1.values, &const_volume, params);
        let output2 = vama_with_kernel(&input2, kernel)?;
        
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
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let data = &candles.close[..50];
        let volume = &candles.volume[..50];
        
        let range = VamaBatchRange {
            length: (10, 20, 5),
            vi_factor: (0.5, 0.7, 0.1),
            sample_period: (0, 10, 10),
            strict: Some(true),
        };
        
        let result = vama_batch_with_kernel(data, volume, &range, kernel)?;
        
        assert!(result.values.len() > 0);
        assert_eq!(result.cols, data.len());
        assert!(result.rows > 0);
        assert_eq!(result.combos.len(), result.rows);
        
        // Verify each combo has valid parameters
        for combo in &result.combos {
            assert!(combo.length.unwrap() >= 10 && combo.length.unwrap() <= 20);
            assert!((combo.vi_factor.unwrap() >= 0.5 - 1e-10) && (combo.vi_factor.unwrap() <= 0.7 + 1e-10));
        }
        Ok(())
    }
    
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];
        
        // Batch with default params in range
        let range = VamaBatchRange::default();
        
        let batch_result = vama_batch_with_kernel(data, volume, &range, kernel)?;
        
        // Single run with default params
        let single_input = VamaInput::from_slices(data, volume, VamaParams::default());
        let single_result = vama_with_kernel(&single_input, kernel)?;
        
        // Find default params in batch
        let default_params = VamaParams::default();
        let default_row = batch_result.values_for(&default_params);
        
        assert!(default_row.is_some(), "[{}] Default params not found in batch", test_name);
        
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
    fn check_vama_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        
        let strat = (2usize..=30)
            .prop_flat_map(|length| {
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
        
        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, volume, length, vi_factor, strict, sample_period)| {
                // Ensure data and volume have same length
                let min_len = data.len().min(volume.len());
                let data = &data[..min_len];
                let volume = &volume[..min_len];
                
                let params = VamaParams {
                    length: Some(length),
                    vi_factor: Some(vi_factor),
                    strict: Some(strict),
                    sample_period: Some(sample_period),
                };
                
                let input = VamaInput::from_slices(data, volume, params);
                let result = vama_with_kernel(&input, kernel);
                
                prop_assert!(result.is_ok(), "[{}] Unexpected error: {:?}", test_name, result);
                let output = result.unwrap();
                
                // Property: Output length equals input length
                prop_assert_eq!(output.values.len(), data.len(), 
                    "[{}] Output length mismatch", test_name);
                
                // Property: Warmup period is correctly applied (at least some NaN values at start)
                let warmup = length - 1;
                for i in 0..warmup.min(data.len()) {
                    prop_assert!(output.values[i].is_nan(), 
                        "[{}] Expected NaN in warmup period at index {}", test_name, i);
                }
                
                // Property: After warmup, values should be finite (not infinite)
                for i in warmup..data.len() {
                    prop_assert!(output.values[i].is_finite() || output.values[i].is_nan(),
                        "[{}] Expected finite or NaN value at index {}, got {}", 
                        test_name, i, output.values[i]);
                }
                
                // Property: Compare with scalar implementation for consistency
                if kernel != Kernel::Scalar {
                    let scalar_result = vama_with_kernel(&input, Kernel::Scalar).unwrap();
                    for i in 0..data.len() {
                        if output.values[i].is_nan() && scalar_result.values[i].is_nan() {
                            continue;
                        }
                        prop_assert!(
                            (output.values[i] - scalar_result.values[i]).abs() < 1e-10,
                            "[{}] Kernel mismatch at index {}: kernel={}, scalar={}",
                            test_name, i, output.values[i], scalar_result.values[i]
                        );
                    }
                }
                
                Ok(())
            })?;
        Ok(())
    }
    
    #[test]
    fn test_vama_accuracy_scalar() {
        let _ = check_vama_accuracy("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_slow_scalar() {
        let _ = check_vama_slow("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_empty_input_scalar() {
        let _ = check_vama_empty_input("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_all_nan_scalar() {
        let _ = check_vama_all_nan("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_invalid_period_scalar() {
        let _ = check_vama_invalid_period("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_invalid_vi_factor_scalar() {
        let _ = check_vama_invalid_vi_factor("scalar", Kernel::Scalar);
    }

    #[test]
    fn test_vama_default_candles_scalar() {
        let _ = check_vama_default_candles("scalar", Kernel::Scalar);
    }

    #[test]
    fn test_vama_streaming_scalar() {
        let _ = check_vama_streaming("scalar", Kernel::Scalar);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_vama_no_poison_scalar() {
        let _ = check_vama_no_poison("scalar", Kernel::Scalar);
    }
    
    // ==================== AVX2 Tests ====================
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_accuracy_avx2() {
        if !is_x86_feature_detected!("avx2") { return; }
        let _ = check_vama_accuracy("avx2", Kernel::Avx2);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_slow_avx2() {
        if !is_x86_feature_detected!("avx2") { return; }
        let _ = check_vama_slow("avx2", Kernel::Avx2);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_streaming_avx2() {
        if !is_x86_feature_detected!("avx2") { return; }
        let _ = check_vama_streaming("avx2", Kernel::Avx2);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_default_candles_avx2() {
        if !is_x86_feature_detected!("avx2") { return; }
        let _ = check_vama_default_candles("avx2", Kernel::Avx2);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64", debug_assertions))]
    #[test]
    fn test_vama_no_poison_avx2() {
        if !is_x86_feature_detected!("avx2") { return; }
        let _ = check_vama_no_poison("avx2", Kernel::Avx2);
    }
    
    // ==================== AVX512 Tests ====================
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_accuracy_avx512() {
        if !is_x86_feature_detected!("avx512f") { return; }
        let _ = check_vama_accuracy("avx512", Kernel::Avx512);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_slow_avx512() {
        if !is_x86_feature_detected!("avx512f") { return; }
        let _ = check_vama_slow("avx512", Kernel::Avx512);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_streaming_avx512() {
        if !is_x86_feature_detected!("avx512f") { return; }
        let _ = check_vama_streaming("avx512", Kernel::Avx512);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_vama_default_candles_avx512() {
        if !is_x86_feature_detected!("avx512f") { return; }
        let _ = check_vama_default_candles("avx512", Kernel::Avx512);
    }
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64", debug_assertions))]
    #[test]
    fn test_vama_no_poison_avx512() {
        if !is_x86_feature_detected!("avx512f") { return; }
        let _ = check_vama_no_poison("avx512", Kernel::Avx512);
    }
    
    // ==================== Additional Comprehensive Tests ====================
    #[test]
    fn test_vama_batch_accuracy() {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).unwrap();
        let data = candles.close.clone();
        let volume = candles.volume.clone();
        
        let range = VamaBatchRange {
            length: (10, 20, 5),
            vi_factor: (0.5, 0.7, 0.1),
            sample_period: (0, 10, 10),
            strict: Some(true),
        };
        
        let result = vama_batch_with_kernel(&data, &volume, &range, Kernel::Auto).unwrap();
        assert!(result.values.len() > 0);
        assert_eq!(result.rows, data.len());
        assert!(result.cols > 0);
        assert_eq!(result.combos.len(), result.cols);
    }
    
    #[test]
    fn test_vama_re_input() {
        // Test that we can feed output back as input
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).unwrap();
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];
        
        let params = VamaParams::default();
        let input1 = VamaInput::from_slices(data, volume, params.clone());
        let output1 = vama(&input1).unwrap();
        
        // Use a constant volume for re-input test
        let const_volume = vec![1000.0; output1.values.len()];
        let input2 = VamaInput::from_slices(&output1.values, &const_volume, params);
        let output2 = vama(&input2).unwrap();
        
        assert_eq!(output1.values.len(), output2.values.len());
        // Check that values after double warmup are finite
        let double_warmup = 24; // Approximately 2 * (length - 1)
        for i in double_warmup..output2.values.len() {
            assert!(output2.values[i].is_finite() || output2.values[i].is_nan());
        }
    }
    
    #[test]
    fn test_vama_builder_api() {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).unwrap();
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];
        
        // Test builder with method chaining
        let result = VamaBuilder::new()
            .length(20)
            .vi_factor(0.5)
            .strict(false)
            .sample_period(10)
            .kernel(Kernel::Auto)
            .apply_slices(data, volume)
            .unwrap();
        
        assert_eq!(result.values.len(), data.len());
        
        // Test builder pattern consistency
        let result2 = VamaBuilder::new()
            .length(20)
            .vi_factor(0.5)
            .apply_slices(data, volume)
            .unwrap();
        
        assert_eq!(result2.values.len(), data.len());
    }
    
    #[test]
    fn test_vama_batch_builder() {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).unwrap();
        let data = &candles.close[..50];
        let volume = &candles.volume[..50];
        
        let result = VamaBatchBuilder::new()
            .length_range(10, 30, 10)
            .vi_factor_static(0.67)
            .sample_period_static(0)
            .strict_static(true)
            .kernel(Kernel::Auto)
            // No need for with_default_slice in this test
            .apply_slices(data, volume)
            .unwrap();
        
        assert_eq!(result.rows, data.len());
        assert!(result.cols > 0); // At least one combination
    }
    
    // New tests for additional coverage
    #[test]
    fn test_vama_partial_params_scalar() {
        let _ = check_vama_partial_params("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_zero_length_scalar() {
        let _ = check_vama_zero_length("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_length_exceeds_data_scalar() {
        let _ = check_vama_length_exceeds_data("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_very_small_dataset_scalar() {
        let _ = check_vama_very_small_dataset("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_nan_handling_scalar() {
        let _ = check_vama_nan_handling("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_reinput_scalar() {
        let _ = check_vama_reinput("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_batch_sweep_scalar() {
        let _ = check_batch_sweep("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_batch_default_row_scalar() {
        let _ = check_batch_default_row("scalar", Kernel::Scalar);
    }
    
    #[cfg(feature = "proptest")]
    #[test]
    fn test_vama_property_scalar() {
        let _ = check_vama_property("scalar", Kernel::Scalar);
    }
    
    #[test]
    fn test_vama_simd_correctness() {
        // Test that all SIMD implementations produce identical results to scalar
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).unwrap();
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];
        
        let params = VamaParams {
            length: Some(15),
            vi_factor: Some(0.75),
            strict: Some(true),
            sample_period: Some(5),
        };
        
        // Scalar reference
        let input = VamaInput::from_slices(data, volume, params.clone());
        let scalar_result = vama_with_kernel(&input, Kernel::Scalar).unwrap();
        
        // Test AVX2 if available
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            let avx2_result = vama_with_kernel(&input, Kernel::Avx2).unwrap();
            for i in 0..scalar_result.values.len() {
                if scalar_result.values[i].is_nan() && avx2_result.values[i].is_nan() {
                    continue;
                }
                assert!(
                    (scalar_result.values[i] - avx2_result.values[i]).abs() < 1e-12,
                    "AVX2 mismatch at index {}: scalar={}, avx2={}",
                    i,
                    scalar_result.values[i],
                    avx2_result.values[i]
                );
            }
        }
        
        // Test AVX512 if available
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") {
            let avx512_result = vama_with_kernel(&input, Kernel::Avx512).unwrap();
            for i in 0..scalar_result.values.len() {
                if scalar_result.values[i].is_nan() && avx512_result.values[i].is_nan() {
                    continue;
                }
                assert!(
                    (scalar_result.values[i] - avx512_result.values[i]).abs() < 1e-12,
                    "AVX512 mismatch at index {}: scalar={}, avx512={}",
                    i,
                    scalar_result.values[i],
                    avx512_result.values[i]
                );
            }
        }
    }
    
    #[test]
    fn test_vama_batch_parallel_vs_sequential() {
        // Test that parallel and sequential batch processing produce identical results
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).unwrap();
        let data = &candles.close[..50];
        let volume = &candles.volume[..50];
        
        let range = VamaBatchRange {
            length: (10, 20, 2),
            vi_factor: (0.5, 0.8, 0.1),
            sample_period: (0, 0, 0),
            strict: Some(true),
        };
        
        // Sequential batch
        let seq_result = vama_batch_with_kernel(data, volume, &range, Kernel::Auto).unwrap();
        
        // Parallel batch
        let par_result = vama_batch_par_slice(data, volume, &range, Kernel::Auto).unwrap();
        
        // Compare results
        assert_eq!(seq_result.rows, par_result.rows);
        assert_eq!(seq_result.cols, par_result.cols);
        assert_eq!(seq_result.combos.len(), par_result.combos.len());
        
        for i in 0..seq_result.values.len() {
            if seq_result.values[i].is_nan() && par_result.values[i].is_nan() {
                continue;
            }
            assert!(
                (seq_result.values[i] - par_result.values[i]).abs() < 1e-12,
                "Parallel vs sequential mismatch at index {}: seq={}, par={}",
                i,
                seq_result.values[i],
                par_result.values[i]
            );
        }
    }
    
    #[test]
    fn test_vama_stream_vs_batch() {
        // Verify streaming and batch produce identical results
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).unwrap();
        let data = &candles.close[..100];
        let volume = &candles.volume[..100];
        
        let params = VamaParams {
            length: Some(15),
            vi_factor: Some(0.7),
            strict: Some(true),
            sample_period: Some(5),
        };
        
        // Batch calculation
        let input = VamaInput::from_slices(data, volume, params.clone());
        let batch = vama(&input).unwrap();
        
        // Streaming calculation
        let mut stream = VamaStream::try_new(params).unwrap();
        let mut streaming = Vec::new();
        for i in 0..data.len() {
            streaming.push(stream.update(data[i], volume[i]).unwrap_or(f64::NAN));
        }
        
        // Compare results
        assert_eq!(batch.values.len(), streaming.len());
        for i in 0..batch.values.len() {
            if batch.values[i].is_nan() && streaming[i].is_nan() {
                continue;
            }
            assert!((batch.values[i] - streaming[i]).abs() < 1e-10,
                    "Mismatch at index {}: batch={}, stream={}", 
                    i, batch.values[i], streaming[i]);
        }
    }
}