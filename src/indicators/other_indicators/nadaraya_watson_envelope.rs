//! # Nadaraya-Watson Envelope (NWE)
//!
//! A non-parametric regression envelope indicator using Gaussian kernel weights
//! to estimate price trends and create upper/lower bands based on mean absolute error.
//!
//! ## Parameters
//! - **bandwidth**: Gaussian kernel bandwidth for smoothing (default: 8.0)
//! - **multiplier**: Band width multiplier for MAE (default: 3.0)
//! - **lookback**: Maximum lookback period for regression (default: 500)
//!
//! ## Errors
//! - **EmptyInputData**: Input data slice is empty
//! - **AllValuesNaN**: All input values are `NaN`
//! - **InvalidBandwidth**: Bandwidth must be positive
//! - **InvalidMultiplier**: Multiplier must be non-negative
//! - **InvalidLookback**: Lookback period must be positive
//! - **NotEnoughValidData**: Not enough valid data points for calculation
//!
//! ## Returns
//! - **`Ok(NweOutput)`** on success, containing `upper` and `lower` envelope vectors
//! - **`Err(NweError)`** otherwise

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
    alloc_with_nan_prefix, detect_best_kernel, detect_best_batch_kernel,
    init_matrix_prefixes, make_uninit_matrix,
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
impl<'a> AsRef<[f64]> for NweInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NweData::Slice(slice) => slice,
            NweData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum NweData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

/// Output structure containing upper and lower envelope values
#[derive(Debug, Clone)]
pub struct NweOutput {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct NweParams {
    pub bandwidth: Option<f64>,
    pub multiplier: Option<f64>,
    pub lookback: Option<usize>,
}

impl Default for NweParams {
    fn default() -> Self {
        Self {
            bandwidth: Some(8.0),
            multiplier: Some(3.0),
            lookback: Some(500),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct NweInput<'a> {
    pub data: NweData<'a>,
    pub params: NweParams,
}

impl<'a> NweInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: NweParams) -> Self {
        Self {
            data: NweData::Candles { candles: c, source: s },
            params: p,
        }
    }
    
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: NweParams) -> Self {
        Self {
            data: NweData::Slice(sl),
            params: p,
        }
    }
    
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", NweParams::default())
    }
    
    #[inline]
    pub fn get_bandwidth(&self) -> f64 {
        self.params.bandwidth.unwrap_or(8.0)
    }
    
    #[inline]
    pub fn get_multiplier(&self) -> f64 {
        self.params.multiplier.unwrap_or(3.0)
    }
    
    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(500)
    }
}

// ==================== BUILDER PATTERN ====================
#[derive(Copy, Clone, Debug)]
pub struct NweBuilder {
    bandwidth: Option<f64>,
    multiplier: Option<f64>,
    lookback: Option<usize>,
    kernel: Kernel,
}

impl Default for NweBuilder {
    fn default() -> Self {
        Self {
            bandwidth: None,
            multiplier: None,
            lookback: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NweBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline(always)]
    pub fn bandwidth(mut self, h: f64) -> Self {
        self.bandwidth = Some(h);
        self
    }
    
    #[inline(always)]
    pub fn multiplier(mut self, m: f64) -> Self {
        self.multiplier = Some(m);
        self
    }
    
    #[inline(always)]
    pub fn lookback(mut self, l: usize) -> Self {
        self.lookback = Some(l);
        self
    }
    
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<NweOutput, NweError> {
        let p = NweParams {
            bandwidth: self.bandwidth,
            multiplier: self.multiplier,
            lookback: self.lookback,
        };
        let i = NweInput::from_candles(c, "close", p);
        nadaraya_watson_envelope_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<NweOutput, NweError> {
        let p = NweParams {
            bandwidth: self.bandwidth,
            multiplier: self.multiplier,
            lookback: self.lookback,
        };
        let i = NweInput::from_slice(d, p);
        nadaraya_watson_envelope_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn into_stream(self) -> Result<NweStream, NweError> {
        let p = NweParams { 
            bandwidth: self.bandwidth, 
            multiplier: self.multiplier, 
            lookback: self.lookback 
        };
        NweStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum NweError {
    #[error("nadaraya_watson_envelope: Input data slice is empty")]
    EmptyInputData,
    
    #[error("nadaraya_watson_envelope: All values are NaN")]
    AllValuesNaN,
    
    #[error("nadaraya_watson_envelope: Invalid bandwidth: {bandwidth}")]
    InvalidBandwidth { bandwidth: f64 },
    
    #[error("nadaraya_watson_envelope: Invalid multiplier: {multiplier}")]
    InvalidMultiplier { multiplier: f64 },
    
    #[error("nadaraya_watson_envelope: Invalid lookback: {lookback}")]
    InvalidLookback { lookback: usize },
    
    #[error("nadaraya_watson_envelope: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    
    #[error("nadaraya_watson_envelope: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

// ==================== HELPER FUNCTIONS ====================

/// Gaussian kernel function for weighted regression
#[inline(always)]
fn gaussian_kernel(x: f64, bandwidth: f64) -> f64 {
    (-x * x / (2.0 * bandwidth * bandwidth)).exp()
}

// ==================== CORE CALCULATION ====================

/// Prepare calculation parameters with proper warmup computation
#[inline]
fn nwe_prepare<'a>(input: &'a NweInput) -> Result<(&'a [f64], f64, f64, usize, usize, usize, Vec<f64>, f64), NweError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 { 
        return Err(NweError::EmptyInputData); 
    }

    let bandwidth = input.get_bandwidth();
    if bandwidth <= 0.0 || bandwidth.is_nan() { 
        return Err(NweError::InvalidBandwidth { bandwidth }); 
    }
    
    let multiplier = input.get_multiplier();
    if multiplier < 0.0 || multiplier.is_nan() { 
        return Err(NweError::InvalidMultiplier { multiplier }); 
    }
    
    let lookback = input.get_lookback();
    if lookback == 0 { 
        return Err(NweError::InvalidLookback { lookback }); 
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(NweError::AllValuesNaN)?;
    const MAE_LEN: usize = 499;
    let warm_out = first + lookback - 1;
    let warm_total = warm_out + MAE_LEN - 1;

    // If there can never be a valid output, fail like alma.rs does
    if len <= warm_out { 
        return Err(NweError::NotEnoughValidData { needed: lookback, valid: len - first }); 
    }

    // Precompute Gaussian weights
    let mut w = Vec::with_capacity(lookback);
    let mut den = 0.0;
    for k in 0..lookback {
        let wk = (-(k as f64) * (k as f64) / (2.0 * bandwidth * bandwidth)).exp();
        w.push(wk);
        den += wk;
    }

    Ok((data, bandwidth, multiplier, lookback, warm_out, warm_total, w, den))
}

/// Zero-copy NWE calculation into provided slices
#[inline]
pub fn nadaraya_watson_envelope_into_slices(
    input: &NweInput,
    upper_out: &mut [f64],
    lower_out: &mut [f64],
) -> Result<(), NweError> {
    let (data, _bw, mult, lookback, warm_out, warm_total, w, den) = nwe_prepare(input)?;
    let len = data.len();
    if upper_out.len() != len || lower_out.len() != len {
        return Err(NweError::NotEnoughValidData { needed: len, valid: upper_out.len().min(lower_out.len()) });
    }

    // Work buffers (no full NaN fill)
    let mut out = alloc_with_nan_prefix(len, warm_out);
    let mut resid = alloc_with_nan_prefix(len, warm_out);

    // Compute endpoint NWE into `out` starting at warm_out
    for t in warm_out..len {
        let mut num = 0.0;
        let mut any_nan = false;
        // fixed window [t-(lookback-1) .. t]
        for k in 0..lookback {
            let x = data[t - k];
            if x.is_nan() { any_nan = true; break; }
            num += x * w[k];
        }
        if !any_nan { out[t] = num / den; } // else keep NaN
    }

    // Residuals from warm_out
    for t in warm_out..len {
        let x = data[t];
        let y = out[t];
        if !x.is_nan() && !y.is_nan() {
            resid[t] = (x - y).abs();
        }
    }

    // Upper/Lower outputs with NaN prefix up to warm_total
    for v in &mut upper_out[..warm_total.min(len)] { *v = f64::from_bits(0x7ff8_0000_0000_0000); }
    for v in &mut lower_out[..warm_total.min(len)] { *v = f64::from_bits(0x7ff8_0000_0000_0000); }

    // Strict SMA_499 over residuals with NaN propagation
    const MAE_LEN: usize = 499;
    if warm_total >= len { return Ok(()); }

    let mut sum = 0.0;
    let mut nan_count = 0usize;

    // Prime window [warm_out .. warm_out + MAE_LEN - 2]
    let start = warm_out;
    let prime_end = (start + MAE_LEN - 1).min(len);
    for t in start..prime_end {
        let r = resid[t];
        if r.is_nan() { nan_count += 1; } else { sum += r; }
    }

    for t in warm_total..len {
        // include current residual
        let r_cur = resid[t];
        if r_cur.is_nan() { nan_count += 1; } else { sum += r_cur; }

        // window start to drop
        let s = t + 1 - MAE_LEN;
        let mae = if nan_count == 0 { (sum / (MAE_LEN as f64)) * mult } else { f64::NAN };

        let y = out[t];
        if !y.is_nan() && !mae.is_nan() {
            upper_out[t] = y + mae;
            lower_out[t] = y - mae;
        }

        // slide: remove resid[s]
        let r_old = resid[s];
        if r_old.is_nan() { nan_count -= 1; } else { sum -= r_old; }
    }

    Ok(())
}

/// Zero-copy NWE calculation without redundant prefix writes
#[inline]
pub fn nadaraya_watson_envelope_into_slices_no_prefix(
    input: &NweInput,
    upper_out: &mut [f64],
    lower_out: &mut [f64],
) -> Result<usize, NweError> {
    let (data, _bw, mult, lookback, warm_out, warm_total, w, den) = nwe_prepare(input)?;
    let len = data.len();
    if upper_out.len() != len || lower_out.len() != len {
        return Err(NweError::NotEnoughValidData { needed: len, valid: upper_out.len().min(lower_out.len()) });
    }

    let mut out = alloc_with_nan_prefix(len, warm_out);
    let mut resid = alloc_with_nan_prefix(len, warm_out);

    // endpoint regression
    for t in warm_out..len {
        let mut num = 0.0;
        let mut bad = false;
        for k in 0..lookback {
            let x = data[t - k];
            if x.is_nan() { bad = true; break; }
            num += x * w[k];
        }
        if !bad { out[t] = num / den; }
    }

    // residuals
    for t in warm_out..len {
        let x = data[t];
        let y = out[t];
        if !x.is_nan() && !y.is_nan() { resid[t] = (x - y).abs(); }
    }

    // sliding MAE
    const MAE_LEN: usize = 499;
    if warm_total >= len { return Ok(warm_total); }

    let mut sum = 0.0;
    let mut nan_c = 0usize;
    let start = warm_out;
    let prime_end = (start + MAE_LEN - 1).min(len);
    for t in start..prime_end { 
        let r=resid[t]; 
        if r.is_nan(){nan_c+=1}else{sum+=r}; 
    }

    for t in warm_total..len {
        let r_cur = resid[t];
        if r_cur.is_nan(){nan_c+=1}else{sum+=r_cur}

        let mae = if nan_c == 0 { (sum / (MAE_LEN as f64)) * mult } else { f64::NAN };
        let y = out[t];
        if !y.is_nan() && !mae.is_nan() {
            upper_out[t] = y + mae;
            lower_out[t] = y - mae;
        } else {
            upper_out[t] = f64::NAN;
            lower_out[t] = f64::NAN;
        }

        let s = t + 1 - MAE_LEN;
        let r_old = resid[s];
        if r_old.is_nan(){nan_c-=1}else{sum-=r_old}
    }

    Ok(warm_total)
}

/// Main NWE calculation (non-repainting endpoint method)
#[inline]
pub fn nadaraya_watson_envelope(input: &NweInput) -> Result<NweOutput, NweError> {
    // compute warm length once
    let len = input.as_ref().len();
    let (_,_,_,_,_,warm_total,_,_) = nwe_prepare(input)?;
    let mut upper = alloc_with_nan_prefix(len, warm_total);
    let mut lower = alloc_with_nan_prefix(len, warm_total);
    let _ = nadaraya_watson_envelope_into_slices_no_prefix(input, &mut upper, &mut lower)?;
    Ok(NweOutput { upper, lower })
}

pub fn nadaraya_watson_envelope_with_kernel(
    input: &NweInput, 
    _kernel: Kernel  // unused; Pine uses only Gaussian
) -> Result<NweOutput, NweError> {
    nadaraya_watson_envelope(input)
}

// ==================== STREAMING SUPPORT ====================
pub struct NweStream {
    lookback: usize,
    weights: Vec<f64>,
    den: f64,

    // price ring
    ring: Vec<f64>,
    head: usize,
    filled: bool,

    // residual ring for MAE
    mae_len: usize,
    resid_ring: Vec<f64>,
    resid_head: usize,
    resid_filled: bool,
    resid_sum: f64,
    resid_nan_count: usize,

    multiplier: f64,
}

impl NweStream {
    pub fn try_new(params: NweParams) -> Result<Self, NweError> {
        let bandwidth = params.bandwidth.unwrap_or(8.0);
        let multiplier = params.multiplier.unwrap_or(3.0);
        let lookback = params.lookback.unwrap_or(500);
        
        if bandwidth <= 0.0 || bandwidth.is_nan() {
            return Err(NweError::InvalidBandwidth { bandwidth });
        }
        
        if multiplier < 0.0 || multiplier.is_nan() {
            return Err(NweError::InvalidMultiplier { multiplier });
        }
        
        if lookback == 0 {
            return Err(NweError::InvalidLookback { lookback });
        }
        
        let mut weights = vec![0.0; lookback];
        for k in 0..lookback {
            weights[k] = gaussian_kernel(k as f64, bandwidth);
        }
        let den: f64 = weights.iter().sum();

        Ok(Self {
            lookback,
            weights,
            den,
            ring: vec![f64::NAN; lookback],
            head: 0,
            filled: false,
            mae_len: 499,
            resid_ring: vec![f64::NAN; 499],
            resid_head: 0,
            resid_filled: false,
            resid_sum: 0.0,
            resid_nan_count: 499, // start as all NaN
            multiplier,
        })
    }
    
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        // push price into ring
        self.ring[self.head] = value;
        self.head = (self.head + 1) % self.lookback;
        if !self.filled && self.head == 0 { 
            self.filled = true; 
        }

        // endpoint regression when filled
        let y = if self.filled {
            let mut num = 0.0;
            for k in 0..self.lookback {
                // newest at head-1 maps to weights[0]
                let idx = (self.head + self.lookback - 1 - k) % self.lookback;
                let x = self.ring[idx];
                if x.is_nan() { 
                    num = f64::NAN; 
                    break; 
                }
                num += x * self.weights[k];
            }
            if num.is_nan() { 
                f64::NAN 
            } else { 
                num / self.den 
            }
        } else { 
            f64::NAN 
        };

        // update residual ring
        let resid = if !value.is_nan() && !y.is_nan() { 
            (value - y).abs() 
        } else { 
            f64::NAN 
        };
        
        // remove old
        let old = self.resid_ring[self.resid_head];
        if old.is_nan() { 
            self.resid_nan_count = self.resid_nan_count.saturating_sub(1);
        } else { 
            self.resid_sum -= old; 
        }
        
        // insert new
        self.resid_ring[self.resid_head] = resid;
        if resid.is_nan() { 
            self.resid_nan_count += 1;
        } else { 
            self.resid_sum += resid; 
        }
        
        self.resid_head = (self.resid_head + 1) % self.mae_len;
        if !self.resid_filled && self.resid_head == 0 { 
            self.resid_filled = true; 
        }

        if self.filled && self.resid_filled && self.resid_nan_count == 0 && !y.is_nan() {
            let mae = (self.resid_sum / (self.mae_len as f64)) * self.multiplier;
            Some((y + mae, y - mae))
        } else {
            None
        }
    }
    
    pub fn reset(&mut self) {
        self.ring.fill(f64::NAN);
        self.head = 0;
        self.filled = false;
        self.resid_ring.fill(f64::NAN);
        self.resid_head = 0;
        self.resid_filled = false;
        self.resid_sum = 0.0;
        self.resid_nan_count = self.mae_len;
    }
}

// ==================== BATCH PROCESSING ====================
#[derive(Debug, Clone)]
pub struct NweBatchRange {
    pub bandwidth: (f64, f64, f64),    // (start, end, step)
    pub multiplier: (f64, f64, f64),   // (start, end, step)
    pub lookback: (usize, usize, usize), // (start, end, step)
}

#[derive(Debug, Clone)]
pub struct NweBatchOutput {
    pub values_upper: Vec<f64>,  // Flattened upper values (row-major)
    pub values_lower: Vec<f64>,  // Flattened lower values (row-major)
    pub combos: Vec<NweParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NweBatchOutput {
    pub fn row_for_params(&self, p: &NweParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.bandwidth.unwrap_or(8.0) == p.bandwidth.unwrap_or(8.0)
                && c.multiplier.unwrap_or(3.0) == p.multiplier.unwrap_or(3.0)
                && c.lookback.unwrap_or(500) == p.lookback.unwrap_or(500)
        })
    }
    
    pub fn values_upper_for(&self, p: &NweParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values_upper[start..start + self.cols]
        })
    }
    
    pub fn values_lower_for(&self, p: &NweParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values_lower[start..start + self.cols]
        })
    }
}

// ==================== BATCH BUILDER ====================
#[derive(Copy, Clone, Debug)]
pub struct NweBatchBuilder {
    bandwidth: (f64, f64, f64),
    multiplier: (f64, f64, f64),
    lookback: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for NweBatchBuilder {
    fn default() -> Self {
        Self {
            bandwidth: (8.0, 8.0, 0.0),
            multiplier: (3.0, 3.0, 0.0),
            lookback: (500, 500, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl NweBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline(always)]
    pub fn bandwidth_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.bandwidth = (start, end, step);
        self
    }
    
    #[inline(always)]
    pub fn bandwidth_static(mut self, value: f64) -> Self {
        self.bandwidth = (value, value, 0.0);
        self
    }
    
    #[inline(always)]
    pub fn multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.multiplier = (start, end, step);
        self
    }
    
    #[inline(always)]
    pub fn multiplier_static(mut self, value: f64) -> Self {
        self.multiplier = (value, value, 0.0);
        self
    }
    
    #[inline(always)]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.lookback = (start, end, step);
        self
    }
    
    #[inline(always)]
    pub fn lookback_static(mut self, value: usize) -> Self {
        self.lookback = (value, value, 0);
        self
    }
    
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    #[inline(always)]
    pub fn apply_candles(self, c: &Candles, source: &str) -> Result<NweBatchOutput, NweError> {
        let data = source_type(c, source);
        self.apply_slice(data)
    }
    
    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<NweBatchOutput, NweError> {
        let sweep = NweBatchRange {
            bandwidth: self.bandwidth,
            multiplier: self.multiplier,
            lookback: self.lookback,
        };
        nwe_batch_with_kernel(data, &sweep, self.kernel)
    }
    
    #[inline(always)]
    pub fn with_default_candles(c: &Candles) -> Result<NweBatchOutput, NweError> {
        Self::new().apply_candles(c, "close")
    }
    
    #[inline(always)]
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<NweBatchOutput, NweError> {
        Self::new().kernel(k).apply_slice(data)
    }
}

// Helper function to expand parameter grid
#[inline(always)]
fn expand_grid(r: &NweBatchRange) -> Vec<NweParams> {
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
    
    let bandwidths = axis_f64(r.bandwidth);
    let multipliers = axis_f64(r.multiplier);
    let lookbacks = axis_usize(r.lookback);
    
    let mut out = Vec::with_capacity(bandwidths.len() * multipliers.len() * lookbacks.len());
    for &b in &bandwidths {
        for &m in &multipliers {
            for &l in &lookbacks {
                out.push(NweParams {
                    bandwidth: Some(b),
                    multiplier: Some(m),
                    lookback: Some(l),
                });
            }
        }
    }
    out
}

/// Helper for batch processing - compute into pre-allocated slices
#[inline(always)]
fn nwe_batch_inner_into(
    data: &[f64],
    sweep: &NweBatchRange,
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<Vec<NweParams>, NweError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() { 
        return Err(NweError::NotEnoughValidData { needed: 1, valid: 0 }); 
    }
    let rows = combos.len();
    let cols = data.len();

    // Prepare per-row warm prefixes
    let mut warm_upper = Vec::with_capacity(rows);
    for prm in &combos {
        // reuse prepare to compute warm points
        let tmp = NweInput::from_slice(data, prm.clone());
        match nwe_prepare(&tmp) {
            Ok((_d, _bw, _m, _lookback, _warm_out, warm_total, _w, _den)) => {
                warm_upper.push(warm_total.min(cols));
            }
            Err(_) => {
                // If prepare fails for this combo, use full warmup
                warm_upper.push(cols);
            }
        }
    }

    // Initialize NaN prefixes in-place
    let out_upper_mu = unsafe { 
        core::slice::from_raw_parts_mut(out_upper.as_mut_ptr() as *mut MaybeUninit<f64>, out_upper.len()) 
    };
    let out_lower_mu = unsafe { 
        core::slice::from_raw_parts_mut(out_lower.as_mut_ptr() as *mut MaybeUninit<f64>, out_lower.len()) 
    };
    init_matrix_prefixes(out_upper_mu, cols, &warm_upper);
    init_matrix_prefixes(out_lower_mu, cols, &warm_upper);

    // For each row compute into borrowed slices (no extra allocs for results)
    for (row, prm) in combos.iter().enumerate() {
        let start = row * cols;
        let u_row = &mut out_upper[start..start + cols];
        let l_row = &mut out_lower[start..start + cols];
        let input = NweInput::from_slice(data, prm.clone());
        // Ignore errors for individual rows (they'll remain NaN)
        let _ = nadaraya_watson_envelope_into_slices(&input, u_row, l_row);
    }

    Ok(combos)
}

pub fn nadaraya_watson_envelope_batch_with_kernel(
    data: &[f64],
    sweep: &NweBatchRange,
    kernel: Kernel,
) -> Result<NweBatchOutput, NweError> {
    nwe_batch_with_kernel(data, sweep, kernel)
}

/// Parallel batch processing using rayon
#[cfg(not(target_arch = "wasm32"))]
pub fn nwe_batch_par_slice(
    data: &[f64],
    sweep: &NweBatchRange,
    k: Kernel,
) -> Result<NweBatchOutput, NweError> {
    let _batch_k = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(NweError::InvalidPeriod { period: 0, data_len: 0 }),
    };

    use rayon::prelude::*;
    
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 { 
        return Err(NweError::EmptyInputData); 
    }

    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);

    // Compute warmup per row using the existing prepare
    let warms: Vec<usize> = combos.iter().map(|p| {
        let tmp = NweInput::from_slice(data, p.clone());
        match nwe_prepare(&tmp) {
            Ok((_d, _bw, _m, _lb, _warm_out, warm_total, _w, _den)) => warm_total.min(cols),
            Err(_) => cols, // keep fully NaN if invalid
        }
    }).collect();

    // Write NaN prefixes once, in place
    init_matrix_prefixes(&mut upper_mu, cols, &warms);
    init_matrix_prefixes(&mut lower_mu, cols, &warms);

    // Convert MU to f64 slices for writing
    let mut upper_guard = core::mem::ManuallyDrop::new(upper_mu);
    let mut lower_guard = core::mem::ManuallyDrop::new(lower_mu);
    let upper_slice: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len()) 
    };
    let lower_slice: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len()) 
    };

    // Process combinations in parallel
    upper_slice.par_chunks_mut(cols)
        .zip(lower_slice.par_chunks_mut(cols))
        .zip(combos.par_iter())
        .for_each(|((u_row, l_row), prm)| {
            let input = NweInput::from_slice(data, prm.clone());
            // Ignore errors for individual rows (they'll remain NaN)
            let _ = nadaraya_watson_envelope_into_slices(&input, u_row, l_row);
        });

    // Reclaim Vecs without copy
    let values_upper = unsafe { 
        Vec::from_raw_parts(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len(), upper_guard.capacity()) 
    };
    let values_lower = unsafe { 
        Vec::from_raw_parts(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len(), lower_guard.capacity()) 
    };

    Ok(NweBatchOutput { values_upper, values_lower, combos, rows, cols })
}

/// Sequential batch for WASM (no rayon)
#[cfg(target_arch = "wasm32")]
pub fn nwe_batch_par_slice(
    data: &[f64],
    sweep: &NweBatchRange,
    kernel: Kernel,
) -> Result<NweBatchOutput, NweError> {
    // Fall back to sequential on WASM
    nwe_batch_with_kernel(data, sweep, kernel)
}

pub fn nwe_batch_with_kernel(
    data: &[f64],
    sweep: &NweBatchRange,
    k: Kernel,
) -> Result<NweBatchOutput, NweError> {
    let _batch_k = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(NweError::InvalidPeriod { period: 0, data_len: 0 }),
    };

    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 { 
        return Err(NweError::EmptyInputData); 
    }

    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);

    // Convert MU to f64 slices for writing
    let mut upper_guard = core::mem::ManuallyDrop::new(upper_mu);
    let mut lower_guard = core::mem::ManuallyDrop::new(lower_mu);
    let upper_slice: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len()) 
    };
    let lower_slice: &mut [f64] = unsafe { 
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len()) 
    };

    let combos_final = nwe_batch_inner_into(data, sweep, upper_slice, lower_slice)?;

    // Reclaim Vecs without copy
    let values_upper = unsafe { 
        Vec::from_raw_parts(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len(), upper_guard.capacity()) 
    };
    let values_lower = unsafe { 
        Vec::from_raw_parts(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len(), lower_guard.capacity()) 
    };

    Ok(NweBatchOutput { values_upper, values_lower, combos: combos_final, rows, cols })
}

/// Sequential batch slice alias (mirrors alma_batch_slice)
#[inline(always)]
pub fn nwe_batch_slice(
    data: &[f64],
    sweep: &NweBatchRange,
    k: Kernel,
) -> Result<NweBatchOutput, NweError> {
    nwe_batch_with_kernel(data, sweep, k)
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "nadaraya_watson_envelope")]
#[pyo3(signature = (data, bandwidth=8.0, multiplier=3.0, lookback=500, kernel=None))]
pub fn nadaraya_watson_envelope_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    bandwidth: f64,
    multiplier: f64,
    lookback: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice_in = data.as_slice()?;
    let _kern = validate_kernel(kernel, false)?; // validate but don't use (NWE doesn't use kernels)

    let params = NweParams { 
        bandwidth: Some(bandwidth), 
        multiplier: Some(multiplier), 
        lookback: Some(lookback) 
    };
    let input = NweInput::from_slice(slice_in, params);

    // Compute with zero-copy into_slices internally, but return as PyArrays
    let len = slice_in.len();
    // Use zero-copy allocator instead of vec![0.0; len] to avoid full zero-fill
    let mut upper = alloc_with_nan_prefix(len, 0);
    let mut lower = alloc_with_nan_prefix(len, 0);
    
    py.allow_threads(|| {
        nadaraya_watson_envelope_into_slices(&input, &mut upper, &mut lower)
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        upper.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "nadaraya_watson_envelope_batch")]
#[pyo3(signature = (
    data, 
    bandwidth_range=(8.0, 8.0, 0.0),
    multiplier_range=(3.0, 3.0, 0.0),
    lookback_range=(500, 500, 0),
    kernel=None
))]
pub fn nadaraya_watson_envelope_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    bandwidth_range: (f64, f64, f64),
    multiplier_range: (f64, f64, f64),
    lookback_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    
    let sweep = NweBatchRange {
        bandwidth: bandwidth_range,
        multiplier: multiplier_range,
        lookback: lookback_range,
    };
    
    let result = py
        .allow_threads(|| nadaraya_watson_envelope_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    let dict = PyDict::new(py);
    
    // Extract parameter arrays from combos
    let bandwidths: Vec<f64> = result.combos.iter().map(|c| c.bandwidth.unwrap_or(8.0)).collect();
    let multipliers: Vec<f64> = result.combos.iter().map(|c| c.multiplier.unwrap_or(3.0)).collect();
    let lookbacks: Vec<usize> = result.combos.iter().map(|c| c.lookback.unwrap_or(500)).collect();
    
    // Reshape flattened arrays into 2D
    dict.set_item("upper", result.values_upper.into_pyarray(py).reshape((result.rows, result.cols))?)?;
    dict.set_item("lower", result.values_lower.into_pyarray(py).reshape((result.rows, result.cols))?)?;
    dict.set_item("bandwidths", bandwidths.into_pyarray(py))?;
    dict.set_item("multipliers", multipliers.into_pyarray(py))?;
    dict.set_item("lookbacks", lookbacks.into_pyarray(py))?;
    
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "NweStream")]
pub struct NweStreamPy {
    inner: NweStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NweStreamPy {
    #[new]
    #[pyo3(signature = (bandwidth=8.0, multiplier=3.0, lookback=500))]
    pub fn new(bandwidth: f64, multiplier: f64, lookback: usize) -> PyResult<Self> {
        let params = NweParams {
            bandwidth: Some(bandwidth),
            multiplier: Some(multiplier),
            lookback: Some(lookback),
        };
        
        let inner = NweStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(Self { inner })
    }
    
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.inner.update(value)
    }
    
    pub fn reset(&mut self) {
        self.inner.reset()
    }
}

#[cfg(feature = "python")]
pub fn register_nadaraya_watson_envelope_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nadaraya_watson_envelope_py, m)?)?;
    m.add_function(wrap_pyfunction!(nadaraya_watson_envelope_batch_py, m)?)?;
    m.add_class::<NweStreamPy>()?;
    Ok(())
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NweJsResult {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NweJsFlat {
    pub values: Vec<f64>, // [upper..., lower...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = nadaraya_watson_envelope)]
pub fn nadaraya_watson_envelope_unified_js(
    data: &[f64],
    bandwidth: f64,
    multiplier: f64,
    lookback: usize,
) -> Result<JsValue, JsValue> {
    let params = NweParams {
        bandwidth: Some(bandwidth),
        multiplier: Some(multiplier),
        lookback: Some(lookback),
    };
    let input = NweInput::from_slice(data, params);
    
    let result = nadaraya_watson_envelope(&input)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let js_result = NweJsResult {
        upper: result.upper,
        lower: result.lower,
    };
    
    serde_wasm_bindgen::to_value(&js_result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nadaraya_watson_envelope_js(
    data: &[f64],
    bandwidth: f64,
    multiplier: f64,
    lookback: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = NweParams {
        bandwidth: Some(bandwidth),
        multiplier: Some(multiplier),
        lookback: Some(lookback),
    };
    let input = NweInput::from_slice(data, params);
    
    let result = nadaraya_watson_envelope(&input)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    // Flatten as [upper..., lower...]
    let mut output = Vec::with_capacity(data.len() * 2);
    output.extend_from_slice(&result.upper);
    output.extend_from_slice(&result.lower);
    
    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = nadaraya_watson_envelope_flat)]
pub fn nadaraya_watson_envelope_flat_js(
    data: &[f64],
    bandwidth: f64,
    multiplier: f64,
    lookback: usize,
) -> Result<JsValue, JsValue> {
    let params = NweParams {
        bandwidth: Some(bandwidth),
        multiplier: Some(multiplier),
        lookback: Some(lookback),
    };
    let input = NweInput::from_slice(data, params);

    let mut upper = vec![0.0; data.len()];
    let mut lower = vec![0.0; data.len()];
    nadaraya_watson_envelope_into_slices(&input, &mut upper, &mut lower)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = Vec::with_capacity(2 * data.len());
    values.extend_from_slice(&upper);
    values.extend_from_slice(&lower);

    serde_wasm_bindgen::to_value(&NweJsFlat { values, rows: 2, cols: data.len() })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = nadaraya_watson_envelope_into_flat)]
pub fn nadaraya_watson_envelope_into_flat(
    data_ptr: *const f64,
    out_ptr: *mut f64, // length = 2*len
    len: usize,
    bandwidth: f64,
    multiplier: f64,
    lookback: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() { 
        return Err(JsValue::from_str("null pointer")); 
    }
    unsafe {
        let data = core::slice::from_raw_parts(data_ptr, len);
        let out = core::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (upper, lower) = out.split_at_mut(len);

        let params = NweParams { 
            bandwidth: Some(bandwidth), 
            multiplier: Some(multiplier), 
            lookback: Some(lookback) 
        };
        let input = NweInput::from_slice(data, params);
        nadaraya_watson_envelope_into_slices(&input, upper, lower)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nadaraya_watson_envelope_into(
    data_ptr: *const f64,
    upper_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
    bandwidth: f64,
    multiplier: f64,
    lookback: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || upper_ptr.is_null() || lower_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to nadaraya_watson_envelope_into"));
    }
    
    unsafe {
        let data = core::slice::from_raw_parts(data_ptr, len);
        let upper_out = core::slice::from_raw_parts_mut(upper_ptr, len);
        let lower_out = core::slice::from_raw_parts_mut(lower_ptr, len);
        
        let params = NweParams {
            bandwidth: Some(bandwidth),
            multiplier: Some(multiplier),
            lookback: Some(lookback),
        };
        let input = NweInput::from_slice(data, params);
        
        nadaraya_watson_envelope_into_slices(&input, upper_out, lower_out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nadaraya_watson_envelope_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nadaraya_watson_envelope_free(ptr: *mut f64, len: usize) {
    unsafe { let _ = Vec::from_raw_parts(ptr, 2 * len, 2 * len); }
}

// ==================== WASM CONTEXT FOR STATEFUL OPERATIONS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct NweContext {
    stream: NweStream,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl NweContext {
    #[wasm_bindgen(constructor)]
    pub fn new(bandwidth: f64, multiplier: f64, lookback: usize) -> Result<NweContext, JsValue> {
        let params = NweParams {
            bandwidth: Some(bandwidth),
            multiplier: Some(multiplier),
            lookback: Some(lookback),
        };
        
        let stream = NweStream::try_new(params)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(NweContext { stream })
    }
    
    #[wasm_bindgen]
    pub fn update(&mut self, value: f64) -> Option<Vec<f64>> {
        self.stream.update(value).map(|(upper, lower)| vec![upper, lower])
    }
    
    #[wasm_bindgen]
    pub fn update_batch(&mut self, values: &[f64]) -> Vec<f64> {
        let mut results = Vec::with_capacity(values.len() * 2);
        
        for &value in values {
            if let Some((upper, lower)) = self.stream.update(value) {
                results.push(upper);
                results.push(lower);
            } else {
                results.push(f64::NAN);
                results.push(f64::NAN);
            }
        }
        
        results
    }
    
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.stream.reset();
    }
}

// ==================== WASM BATCH BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NweBatchJsOutput {
    pub upper: Vec<f64>,         // Flattened 2D array
    pub lower: Vec<f64>,         // Flattened 2D array
    pub rows: usize,             // Number of parameter combinations
    pub cols: usize,             // Data length
    pub bandwidths: Vec<f64>,
    pub multipliers: Vec<f64>,
    pub lookbacks: Vec<usize>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = nadaraya_watson_envelope_batch)]
pub fn nadaraya_watson_envelope_batch_unified_js(
    data: &[f64],
    bandwidth_range: Vec<f64>,  // [start, end, step]
    multiplier_range: Vec<f64>, // [start, end, step]
    lookback_range: Vec<usize>, // [start, end, step]
) -> Result<JsValue, JsValue> {
    if bandwidth_range.len() != 3 || multiplier_range.len() != 3 || lookback_range.len() != 3 {
        return Err(JsValue::from_str("All ranges must have exactly 3 elements [start, end, step]"));
    }
    
    let sweep = NweBatchRange {
        bandwidth: (bandwidth_range[0], bandwidth_range[1], bandwidth_range[2]),
        multiplier: (multiplier_range[0], multiplier_range[1], multiplier_range[2]),
        lookback: (lookback_range[0], lookback_range[1], lookback_range[2]),
    };
    
    let result = nadaraya_watson_envelope_batch_with_kernel(data, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    // Extract parameter arrays from combos
    let bandwidths: Vec<f64> = result.combos.iter().map(|c| c.bandwidth.unwrap_or(8.0)).collect();
    let multipliers: Vec<f64> = result.combos.iter().map(|c| c.multiplier.unwrap_or(3.0)).collect();
    let lookbacks: Vec<usize> = result.combos.iter().map(|c| c.lookback.unwrap_or(500)).collect();
    
    let js_output = NweBatchJsOutput {
        upper: result.values_upper,
        lower: result.values_lower,
        rows: result.rows,
        cols: result.cols,
        bandwidths,
        multipliers,
        lookbacks,
    };
    
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;
    use std::error::Error;
    
    // Test function implementations
    fn check_nwe_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = NweParams {
            bandwidth: None,
            multiplier: None,
            lookback: None,
        };
        let input = NweInput::from_candles(&candles, "close", params);
        let output = nadaraya_watson_envelope_with_kernel(&input, kernel)?;
        assert_eq!(output.upper.len(), candles.close.len());
        assert_eq!(output.lower.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_nwe_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = NweInput::from_candles(&candles, "close", NweParams::default());
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel)?;
        
        // Reference values from PineScript non-repainting mode
        let expected_upper = [62141.41569185, 62108.86018850, 62069.70106389, 62045.52821051, 61980.68541380];
        let expected_lower = [56560.88881720, 56530.68399610, 56490.03377396, 56465.39492722, 56394.51167599];
        
        let len = result.upper.len();
        let start = len.saturating_sub(5);
        
        for (i, (&upper, &lower)) in result.upper[start..].iter()
            .zip(result.lower[start..].iter())
            .enumerate() 
        {
            let diff_upper = (upper - expected_upper[i]).abs();
            let diff_lower = (lower - expected_lower[i]).abs();
            assert!(
                diff_upper < 1e-6,
                "[{}] NWE {:?} upper mismatch at idx {}: got {}, expected {}",
                test_name, kernel, i, upper, expected_upper[i]
            );
            assert!(
                diff_lower < 1e-6,
                "[{}] NWE {:?} lower mismatch at idx {}: got {}, expected {}",
                test_name, kernel, i, lower, expected_lower[i]
            );
        }
        Ok(())
    }
    
    fn check_nwe_warmup_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        // Create data with exactly 1000 points to test warmup
        let data = (0..1000).map(|i| 50000.0 + (i as f64).sin() * 100.0).collect::<Vec<_>>();
        let input = NweInput::from_slice(&data, NweParams::default());
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel)?;
        
        // With defaults: lookback=500, mae_len=499
        // First non-NaN should be at index 997 (lookback-1 + mae_len-1)
        const WARMUP: usize = 499 + 498;
        
        // All values before warmup should be NaN
        for i in 0..WARMUP {
            assert!(result.upper[i].is_nan(), "[{}] Upper should be NaN at {} during warmup", test_name, i);
            assert!(result.lower[i].is_nan(), "[{}] Lower should be NaN at {} during warmup", test_name, i);
        }
        
        // First valid value should be at WARMUP index
        if data.len() > WARMUP {
            assert!(!result.upper[WARMUP].is_nan(), "[{}] Upper should not be NaN at {}", test_name, WARMUP);
            assert!(!result.lower[WARMUP].is_nan(), "[{}] Lower should not be NaN at {}", test_name, WARMUP);
        }
        
        Ok(())
    }
    
    fn check_nwe_zero_bandwidth(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0];
        let params = NweParams {
            bandwidth: Some(0.0),
            multiplier: Some(3.0),
            lookback: Some(500),
        };
        let input = NweInput::from_slice(&data, params);
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel);
        assert!(matches!(result, Err(NweError::InvalidBandwidth { .. })));
        Ok(())
    }
    
    fn check_nwe_negative_multiplier(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0];
        let params = NweParams {
            bandwidth: Some(8.0),
            multiplier: Some(-1.0),
            lookback: Some(500),
        };
        let input = NweInput::from_slice(&data, params);
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel);
        assert!(matches!(result, Err(NweError::InvalidMultiplier { .. })));
        Ok(())
    }
    
    fn check_nwe_zero_lookback(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0];
        let params = NweParams {
            bandwidth: Some(8.0),
            multiplier: Some(3.0),
            lookback: Some(0),
        };
        let input = NweInput::from_slice(&data, params);
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel);
        assert!(matches!(result, Err(NweError::InvalidLookback { .. })));
        Ok(())
    }
    
    fn check_nwe_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input = NweInput::from_slice(&[], NweParams::default());
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel);
        assert!(matches!(result, Err(NweError::EmptyInputData)));
        Ok(())
    }
    
    fn check_nwe_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 10];
        let input = NweInput::from_slice(&data, NweParams::default());
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel);
        assert!(matches!(result, Err(NweError::AllValuesNaN)));
        Ok(())
    }
    
    fn check_nwe_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        // Single point should succeed but produce all NaN
        let data = vec![42.0];
        let input = NweInput::from_slice(&data, NweParams::default());
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel)?;
        assert!(result.upper[0].is_nan());
        assert!(result.lower[0].is_nan());
        
        Ok(())
    }
    
    fn check_nwe_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        
        // Test with default parameters and default source (close)
        let input = NweInput::with_default_candles(&candles);
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel)?;
        
        assert_eq!(result.upper.len(), candles.close.len());
        assert_eq!(result.lower.len(), candles.close.len());
        
        // Find first valid output after warmup
        let first_valid = result.upper.iter()
            .position(|x| !x.is_nan())
            .expect("[{}] No valid upper values found");
        
        // Verify upper > lower for valid values
        assert!(result.upper[first_valid] > result.lower[first_valid],
            "[{}] Upper not greater than lower at first valid index", test_name);
        
        Ok(())
    }
    
    fn check_nwe_lookback_exceeds_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = NweParams {
            bandwidth: Some(8.0),
            multiplier: Some(3.0),
            lookback: Some(10), // lookback > data length
        };
        
        let input = NweInput::from_slice(&data, params);
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel);
        
        // Should return error for insufficient data
        assert!(matches!(result, Err(NweError::NotEnoughValidData { .. })),
            "[{}] Expected NotEnoughValidData error when lookback > data length", test_name);
        
        Ok(())
    }
    
    fn check_nwe_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        // Create sufficient data for non-NaN output
        let data = (0..1100).map(|i| 50000.0 + (i as f64 * 0.1).sin() * 1000.0).collect::<Vec<_>>();
        
        // Use smaller parameters for faster warmup
        let params = NweParams {
            bandwidth: Some(2.0),
            multiplier: Some(2.0),
            lookback: Some(50),
        };
        
        // First pass
        let input1 = NweInput::from_slice(&data, params.clone());
        let result1 = nadaraya_watson_envelope_with_kernel(&input1, kernel)?;
        
        // Extract upper values as new input
        let non_nan_upper: Vec<f64> = result1.upper.iter()
            .filter(|&&x| !x.is_nan())
            .copied()
            .collect();
        
        if non_nan_upper.len() > 100 {
            // Second pass on upper envelope
            let input2 = NweInput::from_slice(&non_nan_upper, params);
            let result2 = nadaraya_watson_envelope_with_kernel(&input2, kernel)?;
            
            // Should produce some non-NaN values
            assert!(result2.upper.iter().any(|&x| !x.is_nan()));
        }
        
        Ok(())
    }
    
    fn check_nwe_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        // Create data with some NaN values
        let mut data = vec![42.0; 1100];
        data[100] = f64::NAN;
        data[200] = f64::NAN;
        data[300] = f64::NAN;
        
        let params = NweParams {
            bandwidth: Some(2.0),
            multiplier: Some(1.0),
            lookback: Some(50),
        };
        
        let input = NweInput::from_slice(&data, params);
        let result = nadaraya_watson_envelope_with_kernel(&input, kernel)?;
        
        // Should handle NaN values gracefully
        assert_eq!(result.upper.len(), data.len());
        assert_eq!(result.lower.len(), data.len());
        
        Ok(())
    }
    
    fn check_nwe_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        // Streaming doesn't use kernel variants, so just test once
        
        // Need enough data for non-NaN output
        let data = (0..1100).map(|i| 50000.0 + (i as f64 * 0.1).sin() * 1000.0).collect::<Vec<_>>();
        
        let params = NweParams {
            bandwidth: Some(8.0),
            multiplier: Some(3.0),
            lookback: Some(500),
        };
        
        // Batch calculation
        let input = NweInput::from_slice(&data, params.clone());
        let batch_result = nadaraya_watson_envelope(&input)?;
        
        // Streaming calculation
        let mut stream = NweStream::try_new(params)?;
        let mut stream_upper = Vec::new();
        let mut stream_lower = Vec::new();
        
        for &value in &data {
            if let Some((upper, lower)) = stream.update(value) {
                stream_upper.push(upper);
                stream_lower.push(lower);
            }
        }
        
        // Find first non-NaN in batch
        let batch_start = batch_result.upper.iter()
            .position(|&x| !x.is_nan())
            .unwrap_or(batch_result.upper.len());
        
        // Compare where both have values
        if !stream_upper.is_empty() && batch_start < batch_result.upper.len() {
            let batch_end = batch_result.upper.len();
            let stream_end = stream_upper.len();
            let compare_len = stream_end.min(batch_end - batch_start);
            
            if compare_len > 0 {
                // Compare last few values (streaming catches up to batch)
                let batch_slice = &batch_result.upper[batch_end - compare_len..];
                let stream_slice = &stream_upper[stream_end - compare_len..];
                
                for (i, (&b, &s)) in batch_slice.iter().zip(stream_slice.iter()).enumerate() {
                    let diff = (b - s).abs();
                    assert!(
                        diff < 1e-6 || (b.is_nan() && s.is_nan()),
                        "[{}] Streaming mismatch at {}: batch={}, stream={}", 
                        test_name, i, b, s
                    );
                }
            }
        }
        
        Ok(())
    }
    
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        
        let output = NweBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        
        let def = NweParams::default();
        let upper_row = output.values_upper_for(&def).expect("default upper row missing");
        let lower_row = output.values_lower_for(&def).expect("default lower row missing");
        
        assert_eq!(upper_row.len(), c.close.len());
        assert_eq!(lower_row.len(), c.close.len());
        
        let expected_upper = [62141.41569185, 62108.86018850, 62069.70106389, 62045.52821051, 61980.68541380];
        let expected_lower = [56560.88881720, 56530.68399610, 56490.03377396, 56465.39492722, 56394.51167599];
        
        let start = upper_row.len().saturating_sub(5);
        for (i, &val) in upper_row[start..].iter().enumerate() {
            let diff = (val - expected_upper[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Batch upper mismatch at {}: {} vs {}",
                test_name, i, val, expected_upper[i]
            );
        }
        
        for (i, &val) in lower_row[start..].iter().enumerate() {
            let diff = (val - expected_lower[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Batch lower mismatch at {}: {} vs {}",
                test_name, i, val, expected_lower[i]
            );
        }
        
        Ok(())
    }
    
    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        // Use smaller dataset for speed
        let data = (0..1100).map(|i| 50000.0 + (i as f64 * 0.1).sin() * 1000.0).collect::<Vec<_>>();
        
        let output = NweBatchBuilder::new()
            .kernel(kernel)
            .bandwidth_range(6.0, 10.0, 2.0)  // 3 values
            .multiplier_range(2.0, 4.0, 1.0)  // 3 values
            .lookback_range(400, 500, 100)    // 2 values
            .apply_slice(&data)?;
        
        // Should have 3 * 3 * 2 = 18 combinations
        assert_eq!(output.rows, 18);
        assert_eq!(output.cols, data.len());
        assert_eq!(output.combos.len(), 18);
        
        Ok(())
    }
    
    // Macro for generating test variants
    macro_rules! generate_all_nwe_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar);
                    }
                )*
                
                // These kernels don't affect NWE (it doesn't use SIMD yet)
                // but we test them for consistency
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
        }
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
        }
    }
    
    #[cfg(debug_assertions)]
    fn check_nwe_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let res = nadaraya_watson_envelope_with_kernel(&NweInput::with_default_candles(&c), kernel)?;
        for (i, &v) in res.upper.iter().chain(res.lower.iter()).enumerate() {
            if v.is_nan() { continue; }
            let b = v.to_bits();
            assert!(b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                    "[{}] poison at {}", test_name, i);
        }
        Ok(())
    }
    
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let sweep = NweBatchRange {
            bandwidth: (8.0, 10.0, 2.0),
            multiplier: (2.0, 3.0, 1.0),
            lookback: (400, 500, 100),
        };
        let res = nwe_batch_with_kernel(source_type(&c, "close"), &sweep, kernel)?;
        
        for &v in res.values_upper.iter().chain(res.values_lower.iter()) {
            if v.is_nan() { continue; }
            let b = v.to_bits();
            assert!(b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                    "[{}] batch poison", test_name);
        }
        Ok(())
    }
    
    #[cfg(debug_assertions)]
    fn check_batch_par_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let sweep = NweBatchRange { 
            bandwidth:(8.0,10.0,2.0), 
            multiplier:(2.0,3.0,1.0), 
            lookback:(400,500,100) 
        };
        let res = nwe_batch_par_slice(source_type(&c, "close"), &sweep, kernel)?;
        for &v in res.values_upper.iter().chain(res.values_lower.iter()) {
            if v.is_nan() { continue; }
            let b = v.to_bits();
            assert!(b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333, 
                "[{}] batch-par poison", test_name);
        }
        Ok(())
    }
    
    #[cfg(feature = "proptest")]
    fn check_nwe_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);
        
        proptest!(|(
            data in prop::collection::vec(0.1f64..10000.0, 10..1000),
            bandwidth in 0.1f64..50.0,
            multiplier in 0.1f64..10.0,
            lookback in 2usize..100
        )| {
            // Property: upper envelope should always be >= lower envelope when both are valid
            let params = NweParams {
                bandwidth: Some(bandwidth),
                multiplier: Some(multiplier),
                lookback: Some(lookback.min(data.len())),
            };
            
            let input = NweInput::from_slice(&data, params);
            if let Ok(result) = nadaraya_watson_envelope_with_kernel(&input, kernel) {
                for (i, (&u, &l)) in result.upper.iter().zip(result.lower.iter()).enumerate() {
                    if !u.is_nan() && !l.is_nan() {
                        prop_assert!(
                            u >= l,
                            "Upper[{}] = {} should be >= Lower[{}] = {}",
                            i, u, i, l
                        );
                    }
                }
            }
        });
        
        Ok(())
    }
    
    // Generate all test variants
    generate_all_nwe_tests!(
        check_nwe_partial_params,
        check_nwe_accuracy,
        check_nwe_warmup_period,
        check_nwe_zero_bandwidth,
        check_nwe_negative_multiplier,
        check_nwe_zero_lookback,
        check_nwe_empty_input,
        check_nwe_all_nan,
        check_nwe_very_small_dataset,
        check_nwe_default_candles,
        check_nwe_lookback_exceeds_data,
        check_nwe_reinput,
        check_nwe_nan_handling,
        check_nwe_streaming
    );
    
    // Add poison check tests only in debug builds
    #[cfg(debug_assertions)]
    generate_all_nwe_tests!(
        check_nwe_no_poison,
        check_batch_no_poison,
        check_batch_par_no_poison
    );
    
    // Add property-based tests when proptest feature is enabled
    #[cfg(feature = "proptest")]
    generate_all_nwe_tests!(
        check_nwe_property
    );
    
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
}