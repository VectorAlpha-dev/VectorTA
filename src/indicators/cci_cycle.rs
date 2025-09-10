//! # CCI Cycle
//!
//! CCI Cycle is a momentum oscillator that applies cycle analysis to the Commodity Channel Index.
//! It uses double exponential smoothing, RMA smoothing, and stochastic normalization to identify cycles.
//!
//! ## Parameters
//! - **length**: Period for CCI calculation (default: 10)
//! - **factor**: Smoothing factor for cycle calculation (default: 0.5)
//!
//! ## Errors
//! - **EmptyInputData**: cci_cycle: Input data slice is empty.
//! - **AllValuesNaN**: cci_cycle: All input values are `NaN`.
//! - **InvalidPeriod**: cci_cycle: Period is zero or exceeds data length.
//! - **NotEnoughValidData**: cci_cycle: Not enough valid data points for calculation.
//!
//! ## Returns
//! - **`Ok(CciCycleOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(CciCycleError)`** otherwise.

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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, 
    init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

// Import other indicators with into_slice functions
use crate::indicators::cci::{cci, cci_into_slice, CciInput, CciParams};
use crate::indicators::moving_averages::ema::{ema, ema_into_slice, EmaInput, EmaParams};
use crate::indicators::moving_averages::smma::{smma, smma_into_slice, SmmaInput, SmmaParams};

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
impl<'a> AsRef<[f64]> for CciCycleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CciCycleData::Slice(slice) => slice,
            CciCycleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum CciCycleData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct CciCycleOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CciCycleParams {
    pub length: Option<usize>,
    pub factor: Option<f64>,
}

impl Default for CciCycleParams {
    fn default() -> Self {
        Self {
            length: Some(10),
            factor: Some(0.5),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct CciCycleInput<'a> {
    pub data: CciCycleData<'a>,
    pub params: CciCycleParams,
}

impl<'a> CciCycleInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CciCycleParams) -> Self {
        Self {
            data: CciCycleData::Candles { candles: c, source: s },
            params: p,
        }
    }
    
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CciCycleParams) -> Self {
        Self {
            data: CciCycleData::Slice(sl),
            params: p,
        }
    }
    
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CciCycleParams::default())
    }
    
    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(10)
    }
    
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(0.5)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct CciCycleBuilder {
    length: Option<usize>,
    factor: Option<f64>,
    kernel: Kernel,
}

impl Default for CciCycleBuilder {
    fn default() -> Self {
        Self {
            length: None,
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CciCycleBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline(always)]
    pub fn length(mut self, val: usize) -> Self {
        self.length = Some(val);
        self
    }
    
    #[inline(always)]
    pub fn factor(mut self, val: f64) -> Self {
        self.factor = Some(val);
        self
    }
    
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CciCycleOutput, CciCycleError> {
        let p = CciCycleParams {
            length: self.length,
            factor: self.factor,
        };
        let i = CciCycleInput::from_candles(c, "close", p);
        cci_cycle_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CciCycleOutput, CciCycleError> {
        let p = CciCycleParams {
            length: self.length,
            factor: self.factor,
        };
        let i = CciCycleInput::from_slice(d, p);
        cci_cycle_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn into_stream(self) -> Result<CciCycleStream, CciCycleError> {
        let p = CciCycleParams {
            length: self.length,
            factor: self.factor,
        };
        CciCycleStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum CciCycleError {
    #[error("cci_cycle: Input data slice is empty.")]
    EmptyInputData,
    
    #[error("cci_cycle: All values are NaN.")]
    AllValuesNaN,
    
    #[error("cci_cycle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    
    #[error("cci_cycle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    
    #[error("cci_cycle: CCI calculation failed: {0}")]
    CciError(String),
    
    #[error("cci_cycle: EMA calculation failed: {0}")]
    EmaError(String),
    
    #[error("cci_cycle: SMMA calculation failed: {0}")]
    SmmaError(String),
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
/// Main entry point with automatic kernel detection
#[inline]
pub fn cci_cycle(input: &CciCycleInput) -> Result<CciCycleOutput, CciCycleError> {
    cci_cycle_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection - using zero-copy pipeline
pub fn cci_cycle_with_kernel(input: &CciCycleInput, kernel: Kernel) -> Result<CciCycleOutput, CciCycleError> {
    let (data, length, factor, first, chosen) = cci_cycle_prepare(input, kernel)?;

    // Scratch for: CCI -> (later) double_ema -> (later) pf
    let mut work = alloc_with_nan_prefix(data.len(), first + length - 1);

    // 1) CCI into `work`
    let ci = CciInput::from_slice(data, CciParams { period: Some(length) });
    cci_into_slice(&mut work, &ci, chosen).map_err(|e| CciCycleError::CciError(e.to_string()))?;

    // 2) EMA(short/long) over CCI
    let half = (length + 1) / 2;
    let mut ema_s = alloc_with_nan_prefix(data.len(), first + half - 1);
    let mut ema_l = alloc_with_nan_prefix(data.len(), first + length - 1);

    let eis = EmaInput::from_slice(&work, EmaParams { period: Some(half) });
    ema_into_slice(&mut ema_s, &eis, chosen).map_err(|e| CciCycleError::EmaError(e.to_string()))?;

    let eil = EmaInput::from_slice(&work, EmaParams { period: Some(length) });
    ema_into_slice(&mut ema_l, &eil, chosen).map_err(|e| CciCycleError::EmaError(e.to_string()))?;

    // Final output with conservative warmup only
    let mut out = alloc_with_nan_prefix(data.len(), first + length * 4);

    // 3) Compute remaining steps using `work` as reusable scratch
    cci_cycle_compute_from_parts(
        data, length, factor, first, chosen,
        &ema_s, &ema_l, &mut work, &mut out,
    )?;

    Ok(CciCycleOutput { values: out })
}

/// Zero-allocation version for WASM and performance-critical paths
#[inline]
pub fn cci_cycle_into_slice(dst: &mut [f64], input: &CciCycleInput, kern: Kernel) -> Result<(), CciCycleError> {
    let (data, length, factor, first, chosen) = cci_cycle_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(CciCycleError::InvalidPeriod { period: dst.len(), data_len: data.len() });
    }

    // Scratch: CCI -> (later) double_ema -> (later) pf
    let mut work = alloc_with_nan_prefix(dst.len(), first + length - 1);

    // 1) CCI
    let ci = CciInput::from_slice(data, CciParams { period: Some(length) });
    cci_into_slice(&mut work, &ci, chosen).map_err(|e| CciCycleError::CciError(e.to_string()))?;

    // 2) EMA(short/long)
    let half = (length + 1) / 2;
    let mut ema_s = alloc_with_nan_prefix(dst.len(), first + half - 1);
    let mut ema_l = alloc_with_nan_prefix(dst.len(), first + length - 1);

    let eis = EmaInput::from_slice(&work, EmaParams { period: Some(half) });
    ema_into_slice(&mut ema_s, &eis, chosen).map_err(|e| CciCycleError::EmaError(e.to_string()))?;

    let eil = EmaInput::from_slice(&work, EmaParams { period: Some(length) });
    ema_into_slice(&mut ema_l, &eil, chosen).map_err(|e| CciCycleError::EmaError(e.to_string()))?;

    // Warmup NaNs only for the visible prefix
    let warm = (first + length * 4).min(dst.len());
    for v in &mut dst[..warm] { *v = f64::NAN; }

    // 3) Finish using `work` as pf buffer
    cci_cycle_compute_from_parts(
        data, length, factor, first, chosen,
        &ema_s, &ema_l, &mut work, dst,
    )?;

    Ok(())
}

/// Prepare and validate input data
#[inline(always)]
fn cci_cycle_prepare<'a>(
    input: &'a CciCycleInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, usize, Kernel), CciCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    
    if len == 0 {
        return Err(CciCycleError::EmptyInputData);
    }
    
    let first = data.iter().position(|x| !x.is_nan())
        .ok_or(CciCycleError::AllValuesNaN)?;
    
    let length = input.get_length();
    let factor = input.get_factor();
    
    // Validation
    if length == 0 || length > len {
        return Err(CciCycleError::InvalidPeriod { 
            period: length, 
            data_len: len 
        });
    }
    
    if len - first < length * 2 {
        return Err(CciCycleError::NotEnoughValidData {
            needed: length * 2,
            valid: len - first,
        });
    }
    
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    
    Ok((data, length, factor, first, chosen))
}

/// Compute from precomputed parts - avoids temporary Vecs and UB
#[inline(always)]
fn cci_cycle_compute_from_parts(
    data: &[f64],
    length: usize,
    factor: f64,
    first: usize,
    kernel: Kernel,
    ema_short: &[f64],
    ema_long:  &[f64],
    work: &mut [f64],   // reused: holds double_ema, then pf
    out:  &mut [f64],
) -> Result<(), CciCycleError> {
    let len = data.len();
    let de_warm = first + length - 1;

    // A) double_ema into `work` (overwrite old CCI), NaN only up to warm
    for i in 0..de_warm { work[i] = f64::NAN; }
    for i in de_warm..len {
        let s = ema_short[i];
        let l = ema_long[i];
        if !s.is_nan() && !l.is_nan() { work[i] = 2.0 * s - l; } else { work[i] = f64::NAN; }
    }

    // B) SMMA over `work` -> `ccis`
    let smma_p = ((length as f64).sqrt().round() as usize).max(1);
    let sm_warm = first + smma_p - 1;
    let mut ccis = alloc_with_nan_prefix(len, sm_warm);
    {
        let si = SmmaInput::from_slice(&work, SmmaParams { period: Some(smma_p) });
        smma_into_slice(&mut ccis, &si, kernel).map_err(|e| CciCycleError::SmmaError(e.to_string()))?;
    }

    // C) First pass: stochastic on `ccis`, EMA-like smoothing into `work` (pf)
    let stoch_warm = first + length - 1;
    for i in 0..stoch_warm { work[i] = f64::NAN; }

    let mut prev_f1 = f64::NAN;
    let mut prev_pf = f64::NAN;

    for i in stoch_warm..len {
        let x = ccis[i];
        if x.is_nan() {
            work[i] = f64::NAN;
            prev_f1 = f64::NAN;
            continue;
        }

        let start = i + 1 - length;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &v in &ccis[start..=i] {
            if !v.is_nan() {
                if v < mn { mn = v; }
                if v > mx { mx = v; }
            }
        }

        let cur_f1 = if mn.is_finite() {
            let range = mx - mn;
            if range > 0.0 {
                ((x - mn) / range) * 100.0
            } else if prev_f1.is_nan() {
                50.0
            } else {
                prev_f1
            }
        } else {
            f64::NAN
        };

        let pf_i = if cur_f1.is_nan() {
            f64::NAN
        } else if prev_pf.is_nan() {
            cur_f1
        } else {
            prev_pf + factor * (cur_f1 - prev_pf)
        };

        work[i] = pf_i;
        prev_f1 = cur_f1;
        prev_pf = pf_i;
    }

    // D) Second pass: normalize `work` (pf) into `out`
    for i in 0..len {
        let p = work[i];
        if p.is_nan() {
            out[i] = f64::NAN;
            continue;
        }
        let start = i.saturating_sub(length - 1);
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &v in &work[start..=i] {
            if !v.is_nan() {
                if v < mn { mn = v; }
                if v > mx { mx = v; }
            }
        }
        if !mn.is_finite() {
            out[i] = f64::NAN;
            continue;
        }
        let range = mx - mn;
        if range > 0.0 {
            let f2 = ((p - mn) / range) * 100.0;
            let prev = if i > 0 { out[i - 1] } else { f64::NAN };
            out[i] = if prev.is_nan() { f2 } else { prev + factor * (f2 - prev) };
        } else {
            out[i] = if i > 0 { out[i - 1] } else { 50.0 };
        }
    }

    Ok(())
}

// ==================== STREAMING SUPPORT ====================
/// Streaming calculator for real-time updates
#[derive(Debug, Clone)]
pub struct CciCycleStream {
    buffer: Vec<f64>,
    cci_buffer: Vec<f64>,
    ema_short_state: f64,
    ema_long_state: f64,
    smma_state: f64,
    pf_state: f64,
    pff_state: f64,
    length: usize,
    factor: f64,
    index: usize,
    ready: bool,
}

impl CciCycleStream {
    pub fn try_new(params: CciCycleParams) -> Result<Self, CciCycleError> {
        let length = params.length.unwrap_or(10);
        let factor = params.factor.unwrap_or(0.5);
        
        if length == 0 {
            return Err(CciCycleError::InvalidPeriod { 
                period: length, 
                data_len: 0 
            });
        }
        
        Ok(Self {
            buffer: vec![0.0; length * 2],
            cci_buffer: vec![0.0; length],
            ema_short_state: f64::NAN,
            ema_long_state: f64::NAN,
            smma_state: f64::NAN,
            pf_state: f64::NAN,
            pff_state: f64::NAN,
            length,
            factor,
            index: 0,
            ready: false,
        })
    }
    
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // This would require implementing streaming versions of CCI, EMA, SMMA
        // For now, return None until enough data collected
        let buffer_len = self.buffer.len();
        self.buffer[self.index % buffer_len] = value;
        self.index += 1;
        
        if self.index >= self.length * 4 {
            self.ready = true;
        }
        
        if self.ready {
            // Would need to implement incremental calculation
            None // Placeholder
        } else {
            None
        }
    }
}

// ==================== BATCH PROCESSING ====================
/// Batch processing for parameter sweeps
#[derive(Clone, Debug)]
pub struct CciCycleBatchRange {
    pub length: (usize, usize, usize), // (start, end, step)
    pub factor: (f64, f64, f64),
}

impl Default for CciCycleBatchRange {
    fn default() -> Self {
        Self {
            length: (10, 100, 10),
            factor: (0.5, 0.5, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CciCycleBatchBuilder {
    range: CciCycleBatchRange,
    kernel: Kernel,
}

impl CciCycleBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
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
    pub fn length_static(mut self, val: usize) -> Self {
        self.range.length = (val, val, 0);
        self
    }
    
    #[inline]
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }
    
    #[inline]
    pub fn factor_static(mut self, val: f64) -> Self {
        self.range.factor = (val, val, 0.0);
        self
    }
    
    pub fn apply_slice(self, data: &[f64]) -> Result<CciCycleBatchOutput, CciCycleError> {
        cci_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }
    
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CciCycleBatchOutput, CciCycleError> {
        let data = source_type(c, src);
        cci_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }
    
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CciCycleBatchOutput, CciCycleError> {
        CciCycleBatchBuilder::new().kernel(k).apply_slice(data)
    }
    
    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<CciCycleBatchOutput, CciCycleError> {
        CciCycleBatchBuilder::new().kernel(k).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct CciCycleBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CciCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CciCycleBatchOutput {
    pub fn row_for_params(&self, p: &CciCycleParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(10) == p.length.unwrap_or(10)
                && (c.factor.unwrap_or(0.5) - p.factor.unwrap_or(0.5)).abs() < 1e-12
        })
    }
    
    pub fn values_for(&self, p: &CciCycleParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

/// Helper to expand parameter grid
#[inline(always)]
fn expand_grid(r: &CciCycleBatchRange) -> Vec<CciCycleParams> {
    fn axis_usize((s,e,st):(usize,usize,usize))->Vec<usize>{
        if st==0 || s==e { return vec![s]; }
        (s..=e).step_by(st).collect()
    }
    fn axis_f64((s,e,st):(f64,f64,f64))->Vec<f64>{
        if st.abs()<1e-12 || (s-e).abs()<1e-12 { return vec![s]; }
        let mut v=Vec::new(); 
        let mut x=s;
        while x<=e+1e-12 { 
            v.push(x); 
            x+=st; 
        }
        v
    }
    let lens = axis_usize(r.length);
    let facts = axis_f64(r.factor);
    let mut out=Vec::with_capacity(lens.len()*facts.len());
    for &l in &lens { 
        for &f in &facts {
            out.push(CciCycleParams{length:Some(l), factor:Some(f)});
        }
    }
    out
}

/// Batch processing with kernel selection
pub fn cci_cycle_batch_with_kernel(
    data: &[f64],
    sweep: &CciCycleBatchRange,
    k: Kernel,
) -> Result<CciCycleBatchOutput, CciCycleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CciCycleError::InvalidPeriod { period: 0, data_len: 0 }),
    };

    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    if cols==0 { return Err(CciCycleError::AllValuesNaN); }

    // Workspace: rows×cols uninit + warmup prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos.iter().map(|p| {
        let length = p.length.unwrap();
        // conservative warmup like single path
        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
        first + length*4
    }).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Get &mut [f64] over the uninit matrix
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len())
    };

    // Fill rows
    let do_row = |row: usize, dst: &mut [f64]| -> Result<(), CciCycleError> {
        let prm = combos[row].clone();
        let inp = CciCycleInput::from_slice(data, prm);
        // Map batch kernel to per-row kernel
        let rk = match kernel {
            Kernel::ScalarBatch => Kernel::Scalar,
            Kernel::Avx2Batch   => Kernel::Avx2,
            Kernel::Avx512Batch => Kernel::Avx512,
            _ => Kernel::Scalar,
        };
        cci_cycle_into_slice(dst, &inp, rk)
    };

    #[cfg(not(target_arch="wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(cols)
           .enumerate()
           .try_for_each(|(r, s)| do_row(r, s))?;
    }
    #[cfg(target_arch="wasm32")]
    {
        for (r, slice) in out.chunks_mut(cols).enumerate() { 
            do_row(r, slice)?; 
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity())
    };
    core::mem::forget(guard);
    
    Ok(CciCycleBatchOutput { values, combos, rows, cols })
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "cci_cycle")]
#[pyo3(signature = (data, length=10, factor=0.5, kernel=None))]
pub fn cci_cycle_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    factor: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = CciCycleParams {
        length: Some(length),
        factor: Some(factor),
    };
    let input = CciCycleInput::from_slice(slice_in, params);
    
    let result_vec: Vec<f64> = py
        .allow_threads(|| cci_cycle_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CciCycleStream")]
pub struct CciCycleStreamPy {
    stream: CciCycleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CciCycleStreamPy {
    #[new]
    fn new(length: usize, factor: f64) -> PyResult<Self> {
        let params = CciCycleParams {
            length: Some(length),
            factor: Some(factor),
        };
        let stream = CciCycleStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CciCycleStreamPy { stream })
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cci_cycle_batch")]
#[pyo3(signature = (data, length_range, factor_range, kernel=None))]
pub fn cci_cycle_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    factor_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let sweep = CciCycleBatchRange { length: length_range, factor: factor_range };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch_k = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        // Reuse Rust batch inner via into_slice per row
        let do_row = |row: usize, dst: &mut [f64]| -> Result<(), CciCycleError> {
            let prm = combos[row].clone();
            let inp = CciCycleInput::from_slice(slice_in, prm);
            let rk = match batch_k {
                Kernel::ScalarBatch => Kernel::Scalar,
                Kernel::Avx2Batch   => Kernel::Avx2,
                Kernel::Avx512Batch => Kernel::Avx512,
                _ => Kernel::Scalar,
            };
            cci_cycle_into_slice(dst, &inp, rk)
        };
        #[cfg(not(target_arch="wasm32"))]
        {
            use rayon::prelude::*;
            slice_out.par_chunks_mut(cols)
                     .enumerate()
                     .try_for_each(|(r, s)| do_row(r, s))
        }
        #[cfg(target_arch="wasm32")]
        {
            for (r, s) in slice_out.chunks_mut(cols).enumerate() { 
                do_row(r, s)?; 
            }
            Ok::<(), CciCycleError>(())
        }
    }).map_err(|e: CciCycleError| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("lengths", combos.iter().map(|p| p.length.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("factors", combos.iter().map(|p| p.factor.unwrap()).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(dict)
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_cycle_js(
    data: &[f64],
    length: usize,
    factor: f64,
) -> Result<Vec<f64>, JsValue> {
    let params = CciCycleParams {
        length: Some(length),
        factor: Some(factor),
    };
    let input = CciCycleInput::from_slice(data, params);
    
    let mut output = alloc_with_nan_prefix(data.len(), 0); // no full zeroing
    cci_cycle_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_cycle_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_cycle_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_cycle_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    factor: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cci_cycle_into"));
    }
    
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        
        let params = CciCycleParams {
            length: Some(length),
            factor: Some(factor),
        };
        let input = CciCycleInput::from_slice(data, params);
        
        if in_ptr == out_ptr {
            let mut temp = alloc_with_nan_prefix(len, 0);
            cci_cycle_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            cci_cycle_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        
        Ok(())
    }
}

// WASM Batch support structures
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CciCycleBatchConfig {
    pub length_range: (usize, usize, usize),
    pub factor_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CciCycleBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CciCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cci_cycle_batch)]
pub fn cci_cycle_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: CciCycleBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = CciCycleBatchRange { length: cfg.length_range, factor: cfg.factor_range };
    let out = cci_cycle_batch_with_kernel(data, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = CciCycleBatchJsOutput { values: out.values, combos: out.combos, rows: out.rows, cols: out.cols };
    serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
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
    
    // ==================== TEST GENERATION MACROS ====================
    macro_rules! generate_all_cci_cycle_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar)
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2)
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx512>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512)
                    }
                )*
            }
        };
    }
    
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] 
                fn [<$fn_name _scalar>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] 
                fn [<$fn_name _avx2>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] 
                fn [<$fn_name _avx512>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch)
                }
            }
        };
    }
    
    fn check_cci_cycle_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = CciCycleInput::from_candles(&candles, "close", CciCycleParams::default());
        let result = cci_cycle_with_kernel(&input, kernel)?;
        
        // REFERENCE VALUES FROM PINESCRIPT
        let expected_last_five = [
            9.25177192,
            20.49219826,
            35.42917181,
            55.57843075,
            77.78921538,
        ];
        
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] CCI_CYCLE {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    
    fn check_cci_cycle_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = cci_cycle_with_kernel(&CciCycleInput::with_default_candles(&c), kernel)?.values;
        for (i, &v) in out.iter().enumerate() {
            if v.is_nan() { continue; }
            let b = v.to_bits();
            assert_ne!(b, 0x11111111_11111111, "[{}] alloc_with_nan_prefix poison at {}", test_name, i);
            assert_ne!(b, 0x22222222_22222222, "[{}] init_matrix_prefixes poison at {}", test_name, i);
            assert_ne!(b, 0x33333333_33333333, "[{}] make_uninit_matrix poison at {}", test_name, i);
        }
        Ok(())
    }
    
    fn check_cci_cycle_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let default_params = CciCycleParams {
            length: None,
            factor: None,
        };
        let input = CciCycleInput::from_candles(&candles, "close", default_params);
        let output = cci_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_cci_cycle_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = CciCycleInput::with_default_candles(&candles);
        match input.data {
            CciCycleData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CciCycleData::Candles"),
        }
        let output = cci_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_cci_cycle_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CciCycleParams {
            length: Some(0),
            factor: None,
        };
        let input = CciCycleInput::from_slice(&input_data, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] CCI_CYCLE should fail with zero period", test_name);
        Ok(())
    }
    
    fn check_cci_cycle_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CciCycleParams {
            length: Some(10),
            factor: None,
        };
        let input = CciCycleInput::from_slice(&data_small, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    
    fn check_cci_cycle_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CciCycleParams::default();
        let input = CciCycleInput::from_slice(&single_point, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    
    fn check_cci_cycle_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = CciCycleParams::default();
        let input = CciCycleInput::from_slice(&empty, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with empty input",
            test_name
        );
        Ok(())
    }
    
    fn check_cci_cycle_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = CciCycleParams::default();
        let input = CciCycleInput::from_slice(&nan_data, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with all NaN values",
            test_name
        );
        Ok(())
    }
    
    fn check_cci_cycle_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // First calculation
        let input = CciCycleInput::from_candles(&candles, "close", CciCycleParams::default());
        let output1 = cci_cycle_with_kernel(&input, kernel)?;
        
        // Use output as input for second calculation
        let input2 = CciCycleInput::from_slice(&output1.values, CciCycleParams::default());
        let output2 = cci_cycle_with_kernel(&input2, kernel)?;
        
        assert_eq!(output2.values.len(), output1.values.len());
        
        // Check that we get valid output
        let non_nan_count = output2.values.iter().filter(|&&v| !v.is_nan()).count();
        assert!(non_nan_count > 0, "[{}] Reinput produced all NaN values", test_name);
        
        Ok(())
    }
    
    fn check_cci_cycle_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        // Data with more valid values to meet minimum requirements
        // CCI Cycle needs at least length*2 valid data points
        let data_with_nans = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, f64::NAN, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0
        ];
        
        let params = CciCycleParams {
            length: Some(5),
            factor: Some(0.5),
        };
        let input = CciCycleInput::from_slice(&data_with_nans, params.clone());
        let result = cci_cycle_with_kernel(&input, kernel);
        
        // The indicator should handle data with some NaN values
        assert!(result.is_ok(), "[{}] Should handle data with some NaN values", test_name);
        
        if let Ok(output) = result {
            assert_eq!(output.values.len(), data_with_nans.len());
            // Should have some valid output values despite NaN in input
            let valid_count = output.values.iter().filter(|&&v| !v.is_nan()).count();
            assert!(valid_count > 0, "[{}] Should produce some valid values", test_name);
        }
        
        // Test with too many NaNs (should fail)
        let mostly_nans = vec![f64::NAN; 20];
        let input2 = CciCycleInput::from_slice(&mostly_nans, params);
        let result2 = cci_cycle_with_kernel(&input2, kernel);
        assert!(result2.is_err(), "[{}] Should fail with all NaN values", test_name);
        
        Ok(())
    }
    
    fn check_cci_cycle_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        // Note: Streaming doesn't use kernel selection, but we keep the parameter for consistency
        let params = CciCycleParams {
            length: Some(10),
            factor: Some(0.5),
        };
        
        let stream_result = CciCycleStream::try_new(params.clone());
        assert!(stream_result.is_ok(), "[{}] Stream creation should succeed", test_name);
        
        let mut stream = stream_result?;
        
        // Feed some data points
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                             11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
        
        for &value in &test_data {
            let _ = stream.update(value);
        }
        
        // After warmup, stream should produce values
        // Note: Current implementation returns None, this is a placeholder test
        // Real implementation would need streaming versions of dependencies
        
        Ok(())
    }
    
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        
        let output = CciCycleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles, "close")?;
        
        // Use the new values_for() method to check default parameters
        let default_params = CciCycleParams::default();
        let row = output.values_for(&default_params);
        
        assert!(row.is_some(), "[{}] Default parameters not found in batch output", test_name);
        
        if let Some(values) = row {
            assert_eq!(values.len(), candles.close.len());
            
            // Check that we have some non-NaN values
            let non_nan_count = values.iter().filter(|&&v| !v.is_nan()).count();
            assert!(non_nan_count > 0, "[{}] Default row has no valid values", test_name);
        }
        
        assert_eq!(output.cols, candles.close.len());
        
        Ok(())
    }
    
    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let data = vec![1.0; 100]; // Simple test data
        
        let output = CciCycleBatchBuilder::new()
            .kernel(kernel)
            .length_range(10, 20, 5)
            .factor_range(0.3, 0.7, 0.2)
            .apply_slice(&data)?;
        
        // Should have 3 lengths * 3 factors = 9 combinations
        assert_eq!(output.combos.len(), 9, "[{}] Unexpected number of parameter combinations", test_name);
        assert_eq!(output.rows, 9);
        assert_eq!(output.cols, 100);
        assert_eq!(output.values.len(), 900);
        
        Ok(())
    }
    
    // ==================== MACRO INVOCATIONS ====================
    generate_all_cci_cycle_tests!(
        check_cci_cycle_accuracy,
        check_cci_cycle_no_poison,
        check_cci_cycle_partial_params,
        check_cci_cycle_default_candles,
        check_cci_cycle_zero_period,
        check_cci_cycle_period_exceeds_length,
        check_cci_cycle_very_small_dataset,
        check_cci_cycle_empty_input,
        check_cci_cycle_all_nan,
        check_cci_cycle_reinput,
        check_cci_cycle_nan_handling,
        check_cci_cycle_streaming
    );
    
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    
    // Property-based tests
    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_cci_cycle_no_panic(data: Vec<f64>, length in 1usize..100) {
            let params = CciCycleParams {
                length: Some(length),
                factor: Some(0.5),
            };
            let input = CciCycleInput::from_slice(&data, params);
            let _ = cci_cycle(&input);
        }
        
        #[test]
        fn test_cci_cycle_length_preservation(size in 10usize..100) {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let params = CciCycleParams::default();
            let input = CciCycleInput::from_slice(&data, params);
            
            if let Ok(output) = cci_cycle(&input) {
                prop_assert_eq!(output.values.len(), size);
            }
        }
    }
}