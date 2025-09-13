//! # Polynomial Regression Bands (PRB)
//!
//! Calculates polynomial regression with optional smoothing and standard deviation bands.
//!
//! ## Parameters
//! - **data**: Input price data
//! - **smooth_data**: Apply Super Smoother Filter (default: true)
//! - **smooth_period**: SSF period (default: 10)
//! - **regression_period**: Lookback period (default: 100)
//! - **polynomial_order**: Order of polynomial (default: 2)
//! - **ndev**: Standard deviation multiplier for bands (default: 2.0)
//!
//! ## Returns
//! - `values`: Vec<f64> - Main regression line
//! - `upper_band`: Vec<f64> - Upper band values
//! - `lower_band`: Vec<f64> - Lower band values
//!
//! ## Developer Status
//! **AVX2**: Stub (calls scalar)
//! **AVX512**: Stub (calls scalar)
//! **Streaming**: O(n²) - Full regression recalc on each update
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`

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
use aligned_vec::{AVec, CACHELINE_ALIGN};

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

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both candle data and raw slices
#[derive(Debug, Clone)]
pub enum PrbData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

/// Output structure containing polynomial regression values and bands
#[derive(Debug, Clone)]
pub struct PrbOutput {
    pub values: Vec<f64>,      // Main polynomial regression line
    pub upper_band: Vec<f64>,   // Upper band (regression + ndev * stdev)
    pub lower_band: Vec<f64>,   // Lower band (regression - ndev * stdev)
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct PrbParams {
    pub smooth_data: Option<bool>,
    pub smooth_period: Option<usize>,
    pub regression_period: Option<usize>,
    pub polynomial_order: Option<usize>,
    pub regression_offset: Option<i32>,
    pub ndev: Option<f64>,  // Standard deviation multiplier for bands
    pub equ_from: Option<usize>,  // "Forecast From _ Bars Ago" parameter for Pine Script parity
}

impl Default for PrbParams {
    fn default() -> Self {
        Self {
            smooth_data: Some(true),
            smooth_period: Some(10),
            regression_period: Some(100),
            polynomial_order: Some(2),
            regression_offset: Some(0),
            ndev: Some(2.0),
            equ_from: Some(0),  // Default to 0 for standard behavior
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct PrbInput<'a> {
    pub data: PrbData<'a>,
    pub params: PrbParams,
}

impl<'a> AsRef<[f64]> for PrbInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PrbData::Candles { candles, source } => source_type(candles, source),
            PrbData::Slice(slice) => slice,
        }
    }
}

impl<'a> PrbInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PrbParams) -> Self {
        Self {
            data: PrbData::Candles { candles: c, source: s },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PrbParams) -> Self {
        Self {
            data: PrbData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PrbParams::default())
    }

    #[inline]
    pub fn get_smooth_data(&self) -> bool {
        self.params.smooth_data.unwrap_or(true)
    }

    #[inline]
    pub fn get_smooth_period(&self) -> usize {
        self.params.smooth_period.unwrap_or(10)
    }

    #[inline]
    pub fn get_regression_period(&self) -> usize {
        self.params.regression_period.unwrap_or(100)
    }

    #[inline]
    pub fn get_polynomial_order(&self) -> usize {
        self.params.polynomial_order.unwrap_or(2)
    }

    #[inline]
    pub fn get_regression_offset(&self) -> i32 {
        self.params.regression_offset.unwrap_or(0)
    }
    
    pub fn get_ndev(&self) -> f64 {
        self.params.ndev.unwrap_or(2.0)
    }
    
    #[inline]
    pub fn get_equ_from(&self) -> usize {
        self.params.equ_from.unwrap_or(0)
    }
}

// ==================== BUILDER PATTERN ====================
#[derive(Copy, Clone, Debug)]
pub struct PrbBuilder {
    smooth_data: Option<bool>,
    smooth_period: Option<usize>,
    regression_period: Option<usize>,
    polynomial_order: Option<usize>,
    regression_offset: Option<i32>,
    ndev: Option<f64>,
    equ_from: Option<usize>,
    kernel: Kernel,
}

impl Default for PrbBuilder {
    fn default() -> Self {
        Self {
            smooth_data: None,
            smooth_period: None,
            regression_period: None,
            polynomial_order: None,
            regression_offset: None,
            ndev: None,
            equ_from: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PrbBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline(always)]
    pub fn smooth_data(mut self, s: bool) -> Self {
        self.smooth_data = Some(s);
        self
    }
    
    #[inline(always)]
    pub fn smooth_period(mut self, p: usize) -> Self {
        self.smooth_period = Some(p);
        self
    }
    
    #[inline(always)]
    pub fn regression_period(mut self, p: usize) -> Self {
        self.regression_period = Some(p);
        self
    }
    
    #[inline(always)]
    pub fn polynomial_order(mut self, o: usize) -> Self {
        self.polynomial_order = Some(o);
        self
    }
    
    #[inline(always)]
    pub fn regression_offset(mut self, o: i32) -> Self {
        self.regression_offset = Some(o);
        self
    }
    
    #[inline(always)]
    pub fn ndev(mut self, n: f64) -> Self {
        self.ndev = Some(n);
        self
    }
    
    #[inline(always)]
    pub fn equ_from(mut self, e: usize) -> Self {
        self.equ_from = Some(e);
        self
    }
    
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PrbOutput, PrbError> {
        let p = PrbParams {
            smooth_data: self.smooth_data,
            smooth_period: self.smooth_period,
            regression_period: self.regression_period,
            polynomial_order: self.polynomial_order,
            regression_offset: self.regression_offset,
            ndev: self.ndev,
            equ_from: self.equ_from,
        };
        let i = PrbInput::from_candles(c, "close", p);
        prb_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PrbOutput, PrbError> {
        let p = PrbParams {
            smooth_data: self.smooth_data,
            smooth_period: self.smooth_period,
            regression_period: self.regression_period,
            polynomial_order: self.polynomial_order,
            regression_offset: self.regression_offset,
            ndev: self.ndev,
            equ_from: self.equ_from,
        };
        let i = PrbInput::from_slice(d, p);
        prb_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn into_stream(self) -> Result<PrbStream, PrbError> {
        let p = PrbParams {
            smooth_data: self.smooth_data,
            smooth_period: self.smooth_period,
            regression_period: self.regression_period,
            polynomial_order: self.polynomial_order,
            regression_offset: self.regression_offset,
            ndev: self.ndev,
            equ_from: self.equ_from,
        };
        PrbStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum PrbError {
    #[error("prb: Input data slice is empty.")]
    EmptyInputData,
    
    #[error("prb: All values are NaN.")]
    AllValuesNaN,
    
    #[error("prb: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    
    #[error("prb: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    
    #[error("prb: Invalid polynomial order: {order} (must be >= 1)")]
    InvalidOrder { order: usize },
    
    #[error("prb: Invalid smooth period: {period} (must be >= 2)")]
    InvalidSmoothPeriod { period: usize },
    
    #[error("prb: Matrix is singular and cannot be decomposed")]
    SingularMatrix,
}

// ==================== STREAMING SUPPORT ====================
pub struct PrbStream {
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
    buffer: Vec<f64>,
    smooth_buffer: Vec<f64>,
    ssf_y1: f64,
    ssf_y2: f64,
    ssf_c1: f64,
    ssf_c2: f64,
    ssf_c3: f64,
    filled: bool,
}

impl PrbStream {
    pub fn try_new(params: PrbParams) -> Result<Self, PrbError> {
        let smooth_data = params.smooth_data.unwrap_or(true);
        let smooth_period = params.smooth_period.unwrap_or(10);
        let regression_period = params.regression_period.unwrap_or(100);
        let polynomial_order = params.polynomial_order.unwrap_or(2);
        let regression_offset = params.regression_offset.unwrap_or(0);
        let ndev = params.ndev.unwrap_or(2.0);
        
        if polynomial_order < 1 {
            return Err(PrbError::InvalidOrder { order: polynomial_order });
        }
        
        if smooth_data && smooth_period < 2 {
            return Err(PrbError::InvalidSmoothPeriod { period: smooth_period });
        }
        
        if regression_period == 0 {
            return Err(PrbError::InvalidPeriod { period: regression_period, data_len: 0 });
        }
        
        // Calculate SSF coefficients
        let pi = std::f64::consts::PI;
        let omega = 2.0 * pi / (smooth_period as f64);
        let a = (-std::f64::consts::SQRT_2 * pi / (smooth_period as f64)).exp();
        let b = 2.0 * a * ((std::f64::consts::SQRT_2 / 2.0) * omega).cos();
        let c3 = -a * a;
        let c2 = b;
        let c1 = 1.0 - c2 - c3;
        
        Ok(Self {
            smooth_data,
            smooth_period,
            regression_period,
            polynomial_order,
            regression_offset,
            ndev,
            buffer: Vec::with_capacity(regression_period),
            smooth_buffer: Vec::with_capacity(regression_period),
            ssf_y1: 0.0,
            ssf_y2: 0.0,
            ssf_c1: c1,
            ssf_c2: c2,
            ssf_c3: c3,
            filled: false,
        })
    }
    
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        // Apply SSF if enabled
        let smoothed = if self.smooth_data {
            let y = self.ssf_c1 * value + self.ssf_c2 * self.ssf_y1 + self.ssf_c3 * self.ssf_y2;
            self.ssf_y2 = self.ssf_y1;
            self.ssf_y1 = y;
            y
        } else {
            value
        };
        
        // Add to buffer
        if self.buffer.len() < self.regression_period {
            self.buffer.push(value);
            self.smooth_buffer.push(smoothed);
            // Set filled when we have enough data
            if self.buffer.len() == self.regression_period {
                self.filled = true;
            }
        } else {
            self.buffer.rotate_left(1);
            self.buffer[self.regression_period - 1] = value;
            self.smooth_buffer.rotate_left(1);
            self.smooth_buffer[self.regression_period - 1] = smoothed;
        }
        
        if !self.filled {
            return None;
        }
        
        // Calculate regression
        let mut x_vals = vec![0.0; self.regression_period];
        for i in 0..self.regression_period {
            x_vals[i] = (i + 1) as f64;
        }
        
        let coefficients = calculate_regression_coefficients(
            &x_vals,
            &self.smooth_buffer,
            self.polynomial_order
        ).ok()?;
        
        let x_position = (self.regression_period as f64) - (self.regression_offset as f64);
        let regression_value = evaluate_polynomial(&coefficients, x_position);
        
        // Calculate standard deviation of the window
        let mean = self.smooth_buffer.iter().sum::<f64>() / self.regression_period as f64;
        let variance = self.smooth_buffer.iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>() / self.regression_period as f64;
        let stdev = variance.sqrt();
        
        // Calculate bands
        let upper = regression_value + self.ndev * stdev;
        let lower = regression_value - self.ndev * stdev;
        
        Some((regression_value, upper, lower))
    }
}

// ==================== CORE COMPUTATION ====================

/// 2-Pole Super Smoother Filter (SSF) implementation
/// This matches the Pine Script formula exactly with proper fallback mechanism
#[inline]
fn ssf_filter(data: &[f64], period: usize, first: usize) -> Vec<f64> {
    let len = data.len();
    if len == 0 { return Vec::new(); }

    let mut out = alloc_with_nan_prefix(len, first);

    let pi = std::f64::consts::PI;
    let omega = 2.0 * pi / (period as f64);
    let a = (-std::f64::consts::SQRT_2 * pi / (period as f64)).exp();
    let b = 2.0 * a * ((std::f64::consts::SQRT_2 / 2.0) * omega).cos();
    let c3 = -a * a;
    let c2 = b;
    let c1 = 1.0 - c2 - c3;

    let mut y1 = f64::NAN; // SSF[1]
    let mut y2 = f64::NAN; // SSF[2]

    for i in first..len {
        // Pine's fallback: nz(SSF[1], x) and nz(SSF[2], nz(SSF[1], x))
        let prev1 = if y1.is_nan() { data[i] } else { y1 };
        let prev2 = if y2.is_nan() { prev1 } else { y2 };
        let y = c1 * data[i] + c2 * prev1 + c3 * prev2;
        out[i] = y;
        y2 = y1;
        y1 = y;
    }
    out
}

/// LU Decomposition for solving linear systems
/// Returns (L, U) matrices
fn lu_decomposition(matrix: &[f64], size: usize) -> Result<(Vec<f64>, Vec<f64>), PrbError> {
    let mut l = vec![0.0; size * size];
    let mut u = vec![0.0; size * size];
    
    // Copy first row to U
    for j in 0..size {
        u[j] = matrix[j];
    }
    
    // Check for zero pivot
    if u[0].abs() < 1e-10 {
        return Err(PrbError::SingularMatrix);
    }
    
    // Compute first column of L
    for i in 1..size {
        l[i * size] = matrix[i * size] / u[0];
    }
    
    // Set diagonal of L to 1
    for i in 0..size {
        l[i * size + i] = 1.0;
    }
    
    // Compute remaining elements
    for i in 1..size {
        for j in i..size {
            // Compute U[i][j]
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[i * size + k] * u[k * size + j];
            }
            u[i * size + j] = matrix[i * size + j] - sum;
            
            // Compute L[j][i] for j > i
            if j > i {
                let mut sum = 0.0;
                for k in 0..i {
                    sum += l[j * size + k] * u[k * size + i];
                }
                if u[i * size + i].abs() < 1e-10 {
                    return Err(PrbError::SingularMatrix);
                }
                l[j * size + i] = (matrix[j * size + i] - sum) / u[i * size + i];
            }
        }
    }
    
    Ok((l, u))
}

/// Forward substitution for lower triangular system Ly = b
fn forward_substitution(l: &[f64], b: &[f64], size: usize) -> Vec<f64> {
    let mut y = vec![0.0; size];
    
    for i in 0..size {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * size + j] * y[j];
        }
        y[i] = sum / l[i * size + i];
    }
    
    y
}

/// Backward substitution for upper triangular system Ux = y
fn backward_substitution(u: &[f64], y: &[f64], size: usize) -> Vec<f64> {
    let mut x = vec![0.0; size];
    
    for i in (0..size).rev() {
        let mut sum = y[i];
        for j in (i + 1)..size {
            sum -= u[i * size + j] * x[j];
        }
        x[i] = sum / u[i * size + i];
    }
    
    x
}

/// Workspace for reusable buffers
struct PrbWorkspace {
    // sizes depend on order; grow on demand
    x_power_sums: Vec<f64>, // len = 2*order + 1
    xy_sums: Vec<f64>,      // len = order + 1
    matrix: Vec<f64>,       // (order+1)^2
    l: Vec<f64>,            // (order+1)^2
    u: Vec<f64>,            // (order+1)^2
    y: Vec<f64>,            // (order+1)
    coeffs: Vec<f64>,       // (order+1)
    x_vals: Vec<f64>,       // regression_period
}

impl PrbWorkspace {
    fn ensure(&mut self, order: usize, reg_p: usize) {
        let ms = order + 1;
        let sq = ms * ms;
        if self.x_power_sums.len() < 2 * order + 1 { self.x_power_sums.resize(2 * order + 1, 0.0); }
        if self.xy_sums.len() < ms { self.xy_sums.resize(ms, 0.0); }
        if self.matrix.len() < sq { self.matrix.resize(sq, 0.0); }
        if self.l.len() < sq { self.l.resize(sq, 0.0); }
        if self.u.len() < sq { self.u.resize(sq, 0.0); }
        if self.y.len() < ms { self.y.resize(ms, 0.0); }
        if self.coeffs.len() < ms { self.coeffs.resize(ms, 0.0); }
        if self.x_vals.len() < reg_p {
            self.x_vals.resize(reg_p, 0.0);
            for i in 0..reg_p { self.x_vals[i] = (i + 1) as f64; }
        }
    }
}

/// In-place solver that uses slices, not new Vecs
#[inline]
fn poly_coeffs_into(
    x_vals: &[f64],           // len = n
    y_window: &[f64],         // len = n
    order: usize,
    x_power_sums: &mut [f64], // len >= 2*order+1
    xy_sums: &mut [f64],      // len >= order+1
    matrix: &mut [f64],       // len >= (order+1)^2
    l: &mut [f64], u: &mut [f64],
    y: &mut [f64],
    coeffs: &mut [f64],
) -> Result<(), PrbError> {
    let n = x_vals.len();
    let m = order + 1;

    // power sums
    for p in 0..=(2 * order) {
        let mut s = 0.0;
        for i in 0..n { s += x_vals[i].powi(p as i32); }
        x_power_sums[p] = s;
    }
    // xy sums
    for p in 0..=order {
        let mut s = 0.0;
        for i in 0..n { s += x_vals[i].powi(p as i32) * y_window[i]; }
        xy_sums[p] = s;
    }
    // build matrix
    for i in 0..m {
        for j in 0..m {
            matrix[i*m + j] = x_power_sums[i + j];
        }
    }
    // LU
    let (l2, u2) = lu_decomposition(matrix, m)?;
    l[..m*m].copy_from_slice(&l2);
    u[..m*m].copy_from_slice(&u2);

    // forward/backward
    for i in 0..m { y[i] = xy_sums[i]; }
    let yy = forward_substitution(l, y, m);
    let xx = backward_substitution(u, &yy, m);
    coeffs[..m].copy_from_slice(&xx);
    Ok(())
}

/// Calculate polynomial regression coefficients (kept for compatibility)
fn calculate_regression_coefficients(
    x_vals: &[f64],
    y_vals: &[f64],
    order: usize,
) -> Result<Vec<f64>, PrbError> {
    let n = x_vals.len();
    let matrix_size = order + 1;
    
    // Calculate x power sums for matrix
    let mut x_power_sums = vec![0.0; 2 * order + 1];
    for p in 0..=(2 * order) {
        let mut sum = 0.0;
        for i in 0..n {
            sum += x_vals[i].powi(p as i32);
        }
        x_power_sums[p] = sum;
    }
    
    // Calculate x^p * y sums for result vector
    let mut xy_sums = vec![0.0; order + 1];
    for p in 0..=order {
        let mut sum = 0.0;
        for i in 0..n {
            sum += x_vals[i].powi(p as i32) * y_vals[i];
        }
        xy_sums[p] = sum;
    }
    
    // Build coefficient matrix
    let mut matrix = vec![0.0; matrix_size * matrix_size];
    for i in 0..matrix_size {
        for j in 0..matrix_size {
            matrix[i * matrix_size + j] = x_power_sums[i + j];
        }
    }
    
    // Solve using LU decomposition
    let (l, u) = lu_decomposition(&matrix, matrix_size)?;
    let y = forward_substitution(&l, &xy_sums, matrix_size);
    let coefficients = backward_substitution(&u, &y, matrix_size);
    
    Ok(coefficients)
}

/// Evaluate polynomial at given x value
#[inline]
fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for (i, &coef) in coefficients.iter().enumerate() {
        result += coef * x.powi(i as i32);
    }
    result
}

/// Main PRB computation function
#[inline(always)]
fn prb_compute_into(
    data: &[f64],
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
    equ_from: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), PrbError> {
    // Dispatch to appropriate kernel implementation
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
            unsafe {
                prb_simd128(data, smooth_data, smooth_period, regression_period, 
                           polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)?;
            }
            return Ok(());
        }
    }
    
    match kernel {
        Kernel::Scalar | Kernel::ScalarBatch => {
            prb_scalar(data, smooth_data, smooth_period, regression_period,
                      polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)?
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => {
            unsafe {
                prb_avx2(data, smooth_data, smooth_period, regression_period,
                        polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)?
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            unsafe {
                prb_avx512(data, smooth_data, smooth_period, regression_period,
                          polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)?
            }
        }
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            prb_scalar(data, smooth_data, smooth_period, regression_period,
                      polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)?
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Scalar implementation - zero-copy and allocation-lean
#[inline]
fn prb_scalar(
    data: &[f64],
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
    equ_from: usize,
    first: usize,
    out: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), PrbError> {
    let len = data.len();

    let smoothed_buf;
    let smoothed: &[f64] = if smooth_data {
        smoothed_buf = ssf_filter(data, smooth_period, first);
        &smoothed_buf
    } else {
        data
    };

    let mut ws = PrbWorkspace {
        x_power_sums: Vec::new(), xy_sums: Vec::new(), matrix: Vec::new(),
        l: Vec::new(), u: Vec::new(), y: Vec::new(), coeffs: Vec::new(),
        x_vals: Vec::new(),
    };
    ws.ensure(polynomial_order, regression_period);

    let warmup = first + regression_period - 1 + equ_from;
    // Adjust x_pos to include equ_from for Pine Script parity
    let x_pos = (regression_period as f64) - (regression_offset as f64) + (equ_from as f64);

    for i in warmup..len {
        // Window ends equ_from bars ago (Pine Script's "Forecast From" behavior)
        let start = i + 1 - regression_period - equ_from;
        let y_win = &smoothed[start..start + regression_period];

        poly_coeffs_into(
            &ws.x_vals[..regression_period],
            y_win,
            polynomial_order,
            &mut ws.x_power_sums, &mut ws.xy_sums,
            &mut ws.matrix, &mut ws.l, &mut ws.u,
            &mut ws.y, &mut ws.coeffs,
        )?;

        let reg = evaluate_polynomial(&ws.coeffs[..=polynomial_order], x_pos);

        // one-pass variance not required; two-pass is fine and allocation-free
        let mean = y_win.iter().sum::<f64>() / (regression_period as f64);
        let mut s = 0.0;
        for &v in y_win { let d = v - mean; s += d * d; }
        let stdev = (s / regression_period as f64).sqrt();

        out[i] = reg;
        out_upper[i] = reg + ndev * stdev;
        out_lower[i] = reg - ndev * stdev;
    }
    Ok(())
}

// AVX2 implementation
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn prb_avx2(
    data: &[f64],
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
    equ_from: usize,
    first: usize,
    out: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), PrbError> {
    // For now, fall back to scalar implementation
    // Full SIMD implementation would require vectorizing the matrix operations
    prb_scalar(data, smooth_data, smooth_period, regression_period,
              polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)
}

// AVX512 implementation
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn prb_avx512(
    data: &[f64],
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
    equ_from: usize,
    first: usize,
    out: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), PrbError> {
    // For now, fall back to scalar implementation
    // Full SIMD implementation would require vectorizing the matrix operations
    prb_scalar(data, smooth_data, smooth_period, regression_period,
              polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)
}

// WASM SIMD128 implementation
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn prb_simd128(
    data: &[f64],
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
    equ_from: usize,
    first: usize,
    out: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), PrbError> {
    // For now, fall back to scalar implementation
    // Full SIMD implementation would require vectorizing the matrix operations
    prb_scalar(data, smooth_data, smooth_period, regression_period,
              polynomial_order, regression_offset, ndev, equ_from, first, out, out_upper, out_lower)
}

// ==================== PUBLIC API ====================

/// Entry point with automatic kernel selection
#[inline]
pub fn prb(input: &PrbInput) -> Result<PrbOutput, PrbError> {
    prb_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection
pub fn prb_with_kernel(input: &PrbInput, kernel: Kernel) -> Result<PrbOutput, PrbError> {
    // Validate input
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(PrbError::EmptyInputData);
    }
    
    let first = data.iter().position(|x| !x.is_nan()).ok_or(PrbError::AllValuesNaN)?;
    
    // Get parameters
    let smooth_data = input.get_smooth_data();
    let smooth_period = input.get_smooth_period();
    let regression_period = input.get_regression_period();
    let polynomial_order = input.get_polynomial_order();
    let regression_offset = input.get_regression_offset();
    let ndev = input.get_ndev();
    let equ_from = input.get_equ_from();
    
    // Validate parameters
    if polynomial_order < 1 {
        return Err(PrbError::InvalidOrder { order: polynomial_order });
    }
    
    if smooth_data && smooth_period < 2 {
        return Err(PrbError::InvalidSmoothPeriod { period: smooth_period });
    }
    
    if regression_period == 0 || regression_period > len {
        return Err(PrbError::InvalidPeriod {
            period: regression_period,
            data_len: len,
        });
    }
    
    let warmup = first + regression_period - 1 + equ_from;
    if warmup >= len {
        return Err(PrbError::NotEnoughValidData {
            needed: regression_period,
            valid: len - first,
        });
    }
    
    // Select kernel
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    
    // Allocate output buffers with NaN prefix
    let mut values = alloc_with_nan_prefix(len, warmup);
    let mut upper_band = alloc_with_nan_prefix(len, warmup);
    let mut lower_band = alloc_with_nan_prefix(len, warmup);
    
    // Compute PRB with bands
    prb_compute_into(
        data,
        smooth_data,
        smooth_period,
        regression_period,
        polynomial_order,
        regression_offset,
        ndev,
        equ_from,
        first,
        chosen,
        &mut values,
        &mut upper_band,
        &mut lower_band,
    )?;
    
    Ok(PrbOutput { values, upper_band, lower_band })
}

/// Write PRB output directly into pre-allocated slices for zero-copy operation
#[inline]
pub fn prb_into_slice(
    dst_main: &mut [f64],
    dst_upper: &mut [f64],
    dst_lower: &mut [f64],
    input: &PrbInput,
    kern: Kernel,
) -> Result<(), PrbError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 { 
        return Err(PrbError::EmptyInputData); 
    }
    if dst_main.len() != len || dst_upper.len() != len || dst_lower.len() != len {
        return Err(PrbError::InvalidPeriod { period: dst_main.len(), data_len: len });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(PrbError::AllValuesNaN)?;
    let smooth_data = input.get_smooth_data();
    let smooth_period = input.get_smooth_period();
    let regression_period = input.get_regression_period();
    let polynomial_order = input.get_polynomial_order();
    let regression_offset = input.get_regression_offset();
    let ndev = input.get_ndev();
    let equ_from = input.get_equ_from();

    if polynomial_order < 1 { 
        return Err(PrbError::InvalidOrder { order: polynomial_order }); 
    }
    if smooth_data && smooth_period < 2 { 
        return Err(PrbError::InvalidSmoothPeriod { period: smooth_period }); 
    }
    if regression_period == 0 || regression_period > len {
        return Err(PrbError::InvalidPeriod { period: regression_period, data_len: len });
    }

    let warmup = first + regression_period - 1 + equ_from;
    for v in &mut dst_main[..warmup] { *v = f64::NAN; }
    for v in &mut dst_upper[..warmup] { *v = f64::NAN; }
    for v in &mut dst_lower[..warmup] { *v = f64::NAN; }

    let chosen = match kern { 
        Kernel::Auto => detect_best_kernel(), 
        k => k 
    };
    
    prb_compute_into(
        data, smooth_data, smooth_period, regression_period, polynomial_order,
        regression_offset, ndev, equ_from, first, chosen, dst_main, dst_upper, dst_lower
    )
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug)]
pub struct PrbBatchRange {
    pub smooth_period: (usize, usize, usize),
    pub regression_period: (usize, usize, usize),
    pub polynomial_order: (usize, usize, usize),
    pub regression_offset: (i32, i32, i32),
}

impl Default for PrbBatchRange {
    fn default() -> Self {
        Self {
            smooth_period: (10, 10, 0),
            regression_period: (100, 100, 0),
            polynomial_order: (2, 2, 0),
            regression_offset: (0, 0, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrbBatchOutput {
    pub values: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub combos: Vec<PrbParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PrbBatchOutput {
    pub fn row_for_params(&self, p: &PrbParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.smooth_period.unwrap_or(10) == p.smooth_period.unwrap_or(10)
                && c.regression_period.unwrap_or(100) == p.regression_period.unwrap_or(100)
                && c.polynomial_order.unwrap_or(2) == p.polynomial_order.unwrap_or(2)
                && c.regression_offset.unwrap_or(0) == p.regression_offset.unwrap_or(0)
        })
    }

    pub fn values_for(&self, p: &PrbParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct PrbBatchBuilder {
    range: PrbBatchRange,
    kernel: Kernel,
    smooth_data: bool,
}

impl PrbBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn smooth_data(mut self, s: bool) -> Self {
        self.smooth_data = s;
        self
    }

    #[inline]
    pub fn smooth_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth_period = (start, end, step);
        self
    }

    #[inline]
    pub fn regression_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.regression_period = (start, end, step);
        self
    }

    #[inline]
    pub fn polynomial_order_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.polynomial_order = (start, end, step);
        self
    }

    #[inline]
    pub fn regression_offset_range(mut self, start: i32, end: i32, step: i32) -> Self {
        self.range.regression_offset = (start, end, step);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<PrbBatchOutput, PrbError> {
        prb_batch_with_kernel(data, &self.range, self.kernel, self.smooth_data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PrbBatchOutput, PrbError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
}

pub fn prb_batch_with_kernel(
    data: &[f64],
    sweep: &PrbBatchRange,
    kernel: Kernel,
    smooth_data: bool,
) -> Result<PrbBatchOutput, PrbError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PrbError::InvalidPeriod { period: 0, data_len: 0 }),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    prb_batch_par_slice(data, sweep, simd, smooth_data)
}

#[inline(always)]
pub fn prb_batch_slice(
    data: &[f64],
    sweep: &PrbBatchRange,
    kern: Kernel,
    smooth_data: bool,
) -> Result<PrbBatchOutput, PrbError> {
    prb_batch_inner(data, sweep, kern, smooth_data, false)
}

#[inline(always)]
pub fn prb_batch_par_slice(
    data: &[f64],
    sweep: &PrbBatchRange,
    kern: Kernel,
    smooth_data: bool,
) -> Result<PrbBatchOutput, PrbError> {
    prb_batch_inner(data, sweep, kern, smooth_data, true)
}

#[inline(always)]
fn prb_batch_inner(
    data: &[f64],
    sweep: &PrbBatchRange,
    kern: Kernel,
    smooth_data: bool,
    parallel: bool,
) -> Result<PrbBatchOutput, PrbError> {
    use core::mem::ManuallyDrop;
    let combos = expand_grid(sweep, smooth_data);
    let cols = data.len();
    let rows = combos.len();
    if cols == 0 { return Err(PrbError::AllValuesNaN); }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(PrbError::AllValuesNaN)?;
    let warm: Vec<usize> = combos.iter().map(|c| first + c.regression_period.unwrap() - 1).collect();

    // Three output matrices
    let mut mu_main = make_uninit_matrix(rows, cols);
    let mut mu_up   = make_uninit_matrix(rows, cols);
    let mut mu_lo   = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut mu_main, cols, &warm);
    init_matrix_prefixes(&mut mu_up,   cols, &warm);
    init_matrix_prefixes(&mut mu_lo,   cols, &warm);

    let mut g_main = ManuallyDrop::new(mu_main);
    let mut g_up   = ManuallyDrop::new(mu_up);
    let mut g_lo   = ManuallyDrop::new(mu_lo);

    let out_main: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(g_main.as_mut_ptr() as *mut f64, g_main.len()) };
    let out_up:   &mut [f64] = unsafe { core::slice::from_raw_parts_mut(g_up.as_mut_ptr()   as *mut f64, g_up.len()) };
    let out_lo:   &mut [f64] = unsafe { core::slice::from_raw_parts_mut(g_lo.as_mut_ptr()   as *mut f64, g_lo.len()) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            let main_ptr = out_main.as_mut_ptr() as usize;
            let up_ptr = out_up.as_mut_ptr() as usize;
            let lo_ptr = out_lo.as_mut_ptr() as usize;
            
            (0..rows).into_par_iter().try_for_each(|row| -> Result<(), PrbError> {
                let c = &combos[row];
                unsafe {
                    let r_main = std::slice::from_raw_parts_mut(
                        (main_ptr as *mut f64).add(row * cols),
                        cols
                    );
                    let r_up = std::slice::from_raw_parts_mut(
                        (up_ptr as *mut f64).add(row * cols),
                        cols
                    );
                    let r_lo = std::slice::from_raw_parts_mut(
                        (lo_ptr as *mut f64).add(row * cols),
                        cols
                    );
                    
                    prb_compute_into(
                        data,
                        smooth_data,
                        c.smooth_period.unwrap_or(10),
                        c.regression_period.unwrap_or(100),
                        c.polynomial_order.unwrap_or(2),
                        c.regression_offset.unwrap_or(0),
                        c.ndev.unwrap_or(2.0),
                        c.equ_from.unwrap_or(0),
                        first,
                        kern,
                        r_main, r_up, r_lo
                    )
                }
            })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let c = &combos[row];
                let r_main = &mut out_main[row * cols .. (row + 1) * cols];
                let r_up   = &mut out_up  [row * cols .. (row + 1) * cols];
                let r_lo   = &mut out_lo  [row * cols .. (row + 1) * cols];

                prb_compute_into(
                    data,
                    smooth_data,
                    c.smooth_period.unwrap_or(10),
                    c.regression_period.unwrap_or(100),
                    c.polynomial_order.unwrap_or(2),
                    c.regression_offset.unwrap_or(0),
                    c.ndev.unwrap_or(2.0),
                    c.equ_from.unwrap_or(0),
                    first,
                    kern,
                    r_main, r_up, r_lo
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let c = &combos[row];
            let r_main = &mut out_main[row * cols .. (row + 1) * cols];
            let r_up   = &mut out_up  [row * cols .. (row + 1) * cols];
            let r_lo   = &mut out_lo  [row * cols .. (row + 1) * cols];

            prb_compute_into(
                data,
                smooth_data,
                c.smooth_period.unwrap_or(10),
                c.regression_period.unwrap_or(100),
                c.polynomial_order.unwrap_or(2),
                c.regression_offset.unwrap_or(0),
                c.ndev.unwrap_or(2.0),
                c.equ_from.unwrap_or(0),
                first,
                kern,
                r_main, r_up, r_lo
            )?;
        }
    }

    let values = unsafe { Vec::from_raw_parts(g_main.as_mut_ptr() as *mut f64, g_main.len(), g_main.capacity()) };
    let upper_band = unsafe { Vec::from_raw_parts(g_up.as_mut_ptr()   as *mut f64, g_up.len(),   g_up.capacity()) };
    let lower_band = unsafe { Vec::from_raw_parts(g_lo.as_mut_ptr()   as *mut f64, g_lo.len(),   g_lo.capacity()) };

    Ok(PrbBatchOutput { values, upper_band, lower_band, combos, rows, cols })
}

#[inline(always)]
fn expand_grid(r: &PrbBatchRange, smooth_flag: bool) -> Vec<PrbParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_i32((start, end, step): (i32, i32, i32)) -> Vec<i32> {
        if step == 0 || start == end { return vec![start]; }
        let mut v = Vec::new(); let mut x = start;
        while x <= end { v.push(x); x += step; }
        v
    }

    let sps = axis_usize(r.smooth_period);
    let rps = axis_usize(r.regression_period);
    let pos = axis_usize(r.polynomial_order);
    let ros = axis_i32(r.regression_offset);

    let mut out = Vec::with_capacity(sps.len()*rps.len()*pos.len()*ros.len());
    for &sp in &sps {
        for &rp in &rps {
            for &po in &pos {
                for &ro in &ros {
                    out.push(PrbParams {
                        smooth_data: Some(smooth_flag),
                        smooth_period: Some(sp),
                        regression_period: Some(rp),
                        polynomial_order: Some(po),
                        regression_offset: Some(ro),
                        ndev: Some(2.0),  // Default ndev for batch processing
                        equ_from: Some(0),  // Default equ_from for batch processing
                    });
                }
            }
        }
    }
    out
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "prb")]
#[pyo3(signature = (data, smooth_data, smooth_period, regression_period, polynomial_order, regression_offset, ndev, kernel=None))]
pub fn prb_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice_in = data.as_slice()?;
    let params = PrbParams {
        smooth_data: Some(smooth_data),
        smooth_period: Some(smooth_period),
        regression_period: Some(regression_period),
        polynomial_order: Some(polynomial_order),
        regression_offset: Some(regression_offset),
        ndev: Some(ndev),
        equ_from: Some(0),  // Default value for Python bindings
    };
    let input = PrbInput::from_slice(slice_in, params);
    let kern = validate_kernel(kernel, false)?;
    let (m,u,l) = py.allow_threads(|| {
        prb_with_kernel(&input, kern).map(|o| (o.values, o.upper_band, o.lower_band))
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((m.into_pyarray(py), u.into_pyarray(py), l.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "prb_batch")]
pub fn prb_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    smooth_data: bool,
    smooth_period_start: usize,
    smooth_period_end: usize,
    smooth_period_step: usize,
    regression_period_start: usize,
    regression_period_end: usize,
    regression_period_step: usize,
    polynomial_order_start: usize,
    polynomial_order_end: usize,
    polynomial_order_step: usize,
    regression_offset_start: i32,
    regression_offset_end: i32,
    regression_offset_step: i32,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    
    let sweep = PrbBatchRange {
        smooth_period: (smooth_period_start, smooth_period_end, smooth_period_step),
        regression_period: (regression_period_start, regression_period_end, regression_period_step),
        polynomial_order: (polynomial_order_start, polynomial_order_end, polynomial_order_step),
        regression_offset: (regression_offset_start, regression_offset_end, regression_offset_step),
    };
    
    let out = prb_batch_with_kernel(slice_in, &sweep, kern, smooth_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = out.rows;
    let cols = out.cols;

    let dict = PyDict::new(py);
    
    // Convert to PyArray and reshape
    use ndarray::Array2;
    let values_arr = Array2::from_shape_vec((rows, cols), out.values)
        .map_err(|e| PyValueError::new_err(format!("Failed to reshape values: {}", e)))?;
    let upper_arr = Array2::from_shape_vec((rows, cols), out.upper_band)
        .map_err(|e| PyValueError::new_err(format!("Failed to reshape upper: {}", e)))?;
    let lower_arr = Array2::from_shape_vec((rows, cols), out.lower_band)
        .map_err(|e| PyValueError::new_err(format!("Failed to reshape lower: {}", e)))?;
    
    dict.set_item("values", values_arr.into_pyarray(py))?;   // shape (rows, cols)
    dict.set_item("upper",  upper_arr.into_pyarray(py))?;    // shape (rows, cols)
    dict.set_item("lower",  lower_arr.into_pyarray(py))?;    // shape (rows, cols)

    // axes + metadata, same names you already use
    dict.set_item("smooth_periods",
        out.combos.iter().map(|p| p.smooth_period.unwrap_or(10) as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("regression_periods",
        out.combos.iter().map(|p| p.regression_period.unwrap_or(100) as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("polynomial_orders",
        out.combos.iter().map(|p| p.polynomial_order.unwrap_or(2) as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("regression_offsets",
        out.combos.iter().map(|p| p.regression_offset.unwrap_or(0)).collect::<Vec<_>>().into_pyarray(py))?;

    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PrbStreamPy {
    stream: PrbStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PrbStreamPy {
    #[new]
    fn new(
        smooth_data: Option<bool>,
        smooth_period: Option<usize>,
        regression_period: Option<usize>,
        polynomial_order: Option<usize>,
        regression_offset: Option<i32>,
        ndev: Option<f64>,
    ) -> PyResult<Self> {
        let params = PrbParams {
            smooth_data,
            smooth_period,
            regression_period,
            polynomial_order,
            regression_offset,
            ndev,
            equ_from: Some(0),  // Default value for Python streaming API
        };
        let stream = PrbStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PrbStreamPy { stream })
    }
    
    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(value)
    }
}

// ==================== WASM BINDINGS ====================

// WASM result structure for returning multiple arrays
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct PrbJsResult {
    values: Vec<f64>, // [main..., upper..., lower...] flattened
    rows: usize,      // 3 (main, upper, lower)
    cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl PrbJsResult {
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }
    
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "prb")]
pub fn prb_js(
    data: &[f64],
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
) -> Result<PrbJsResult, JsValue> {
    let params = PrbParams {
        smooth_data: Some(smooth_data),
        smooth_period: Some(smooth_period),
        regression_period: Some(regression_period),
        polynomial_order: Some(polynomial_order),
        regression_offset: Some(regression_offset),
        ndev: Some(ndev),
        equ_from: Some(0),  // Default value for Python bindings
    };
    let input = PrbInput::from_slice(data, params);
    
    let output = prb(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    // Flatten the three arrays into one
    let mut values = output.values;
    values.extend(output.upper_band);
    values.extend(output.lower_band);
    
    Ok(PrbJsResult {
        values,
        rows: 3,
        cols: data.len(),
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "prb_batch")]
pub fn prb_batch_js(
    data: &[f64],
    smooth_data: bool,
    smooth_period_start: usize,
    smooth_period_end: usize,
    smooth_period_step: usize,
    regression_period_start: usize,
    regression_period_end: usize,
    regression_period_step: usize,
    polynomial_order_start: usize,
    polynomial_order_end: usize,
    polynomial_order_step: usize,
    regression_offset_start: i32,
    regression_offset_end: i32,
    regression_offset_step: i32,
) -> Result<JsValue, JsValue> {
    let sweep = PrbBatchRange {
        smooth_period: (smooth_period_start, smooth_period_end, smooth_period_step),
        regression_period: (regression_period_start, regression_period_end, regression_period_step),
        polynomial_order: (polynomial_order_start, polynomial_order_end, polynomial_order_step),
        regression_offset: (regression_offset_start, regression_offset_end, regression_offset_step),
    };

    let out = prb_batch_slice(data, &sweep, detect_best_kernel(), smooth_data)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // concatenate [values rows..., upper rows..., lower rows...] in row-major
    let mut values = Vec::with_capacity(out.values.len() + out.upper_band.len() + out.lower_band.len());
    values.extend_from_slice(&out.values);
    values.extend_from_slice(&out.upper_band);
    values.extend_from_slice(&out.lower_band);

    let flat = PrbBatchFlatJs {
        values,
        rows: 3 * out.rows,
        cols: out.cols,
        combos: out.combos,
    };
    serde_wasm_bindgen::to_value(&flat).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// WASM pointer helpers for zero-copy operations
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn prb_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn prb_free(ptr: *mut f64, len: usize) {
    unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn prb_into(
    in_ptr: *const f64,
    out_main: *mut f64,
    out_upper: *mut f64,
    out_lower: *mut f64,
    len: usize,
    smooth_data: bool,
    smooth_period: usize,
    regression_period: usize,
    polynomial_order: usize,
    regression_offset: i32,
    ndev: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_main.is_null() || out_upper.is_null() || out_lower.is_null() {
        return Err(JsValue::from_str("null pointer passed to prb_into"));
    }
    unsafe {
        let data = core::slice::from_raw_parts(in_ptr, len);
        let mut m = core::slice::from_raw_parts_mut(out_main, len);
        let mut u = core::slice::from_raw_parts_mut(out_upper, len);
        let mut l = core::slice::from_raw_parts_mut(out_lower, len);
        let params = PrbParams {
            smooth_data: Some(smooth_data),
            smooth_period: Some(smooth_period),
            regression_period: Some(regression_period),
            polynomial_order: Some(polynomial_order),
            regression_offset: Some(regression_offset),
            ndev: Some(ndev),
            equ_from: Some(0),
        };
        let input = PrbInput::from_slice(data, params);
        prb_into_slice(&mut m, &mut u, &mut l, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PrbBatchJsOutput {
    pub values: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<PrbParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PrbBatchFlatJs {
    pub values: Vec<f64>, // concatenated blocks row-major, with 3 outputs stacked per combo
    pub rows: usize,      // = 3 * combos_len
    pub cols: usize,      // = data.len()
    pub combos: Vec<PrbParams>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "prb_batch_unified")]
pub fn prb_batch_unified_js(data: &[f64], config: JsValue, smooth_data: bool) -> Result<JsValue, JsValue> {
    #[derive(Serialize, Deserialize)]
    struct Cfg {
        smooth_period: (usize, usize, usize),
        regression_period: (usize, usize, usize),
        polynomial_order: (usize, usize, usize),
        regression_offset: (i32, i32, i32),
    }
    let cfg: Cfg = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = PrbBatchRange {
        smooth_period: cfg.smooth_period,
        regression_period: cfg.regression_period,
        polynomial_order: cfg.polynomial_order,
        regression_offset: cfg.regression_offset,
    };
    let out = prb_batch_inner(data, &sweep, detect_best_kernel(), smooth_data, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = PrbBatchJsOutput {
        values: out.values,
        upper: out.upper_band,
        lower: out.lower_band,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;
    
    fn check_prb_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(100),
            polynomial_order: Some(2),
            regression_offset: Some(0),
            ndev: Some(2.0),
            equ_from: Some(0),
        };
        let input = PrbInput::from_candles(&candles, "close", params);
        let result = prb_with_kernel(&input, kernel)?;
        
        // Reference values from actual CSV data (last 5 non-NaN values)
        let expected_last_five = [
            59083.04826441,
            58900.06593477,
            58722.13172976,
            58575.33291206,
            58376.00589983,
        ];
        
        // Get last 5 non-NaN values
        let non_nan_values: Vec<f64> = result.values.iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();
        
        assert!(non_nan_values.len() >= 5, "[{}] Should have at least 5 non-NaN values", test_name);
        
        let start = non_nan_values.len() - 5;
        for i in 0..5 {
            let actual = non_nan_values[start + i];
            let expected_val = expected_last_five[i];
            let diff = (actual - expected_val).abs();
            let tolerance = expected_val.abs() * 0.01; // 1% tolerance
            
            assert!(
                diff < tolerance,
                "[{}] PRB {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                actual,
                expected_val
            );
        }
        Ok(())
    }
    
    fn check_prb_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = PrbParams {
            smooth_data: None,
            smooth_period: None,
            regression_period: None,
            polynomial_order: None,
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let input = PrbInput::from_candles(&candles, "close", params);
        let output = prb_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_prb_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(0),
            polynomial_order: None,
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let input = PrbInput::from_slice(&input_data, params);
        let res = prb_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] PRB should fail with zero period", test_name);
        Ok(())
    }
    
    fn check_prb_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(10),
            polynomial_order: None,
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let input = PrbInput::from_slice(&data_small, params);
        let res = prb_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PRB should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    
    fn check_prb_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(100),
            polynomial_order: None,
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let input = PrbInput::from_slice(&single_point, params);
        let res = prb_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] PRB should fail with insufficient data", test_name);
        Ok(())
    }
    
    fn check_prb_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = PrbInput::from_slice(&empty, PrbParams::default());
        let res = prb_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(PrbError::EmptyInputData)),
            "[{}] PRB should fail with empty input",
            test_name
        );
        Ok(())
    }
    
    fn check_prb_invalid_smooth_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = PrbParams {
            smooth_data: Some(true),
            smooth_period: Some(1),
            regression_period: Some(2),
            polynomial_order: None,
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let input = PrbInput::from_slice(&data, params);
        let res = prb_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(PrbError::InvalidSmoothPeriod { .. })),
            "[{}] PRB should fail with invalid smooth period",
            test_name
        );
        Ok(())
    }
    
    fn check_prb_invalid_order(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(2),
            polynomial_order: Some(0),
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let input = PrbInput::from_slice(&data, params);
        let res = prb_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(PrbError::InvalidOrder { .. })),
            "[{}] PRB should fail with invalid polynomial order",
            test_name
        );
        Ok(())
    }
    
    fn check_prb_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let first_params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(50),
            polynomial_order: Some(2),
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let first_input = PrbInput::from_candles(&candles, "close", first_params);
        let first_result = prb_with_kernel(&first_input, kernel)?;
        
        let second_params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(50),
            polynomial_order: Some(2),
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let second_input = PrbInput::from_slice(&first_result.values, second_params);
        let second_result = prb_with_kernel(&second_input, kernel)?;
        
        assert_eq!(second_result.values.len(), first_result.values.len());
        
        Ok(())
    }
    
    fn check_prb_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = PrbInput::from_candles(
            &candles,
            "close",
            PrbParams {
                smooth_data: Some(false),
                smooth_period: None,
                regression_period: Some(50),
                polynomial_order: Some(2),
                regression_offset: None,
                ndev: None,
                equ_from: None,
            },
        );
        let res = prb_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
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
    
    fn check_prb_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(10),
            polynomial_order: Some(2),
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        
        let mut stream = PrbStream::try_new(params)?;
        
        // Feed data points
        for i in 1..=15 {
            let val = i as f64 * 10.0;
            let result = stream.update(val);
            if i >= 10 {
                assert!(result.is_some(), "[{}] Stream should produce output after warmup", test_name);
            } else {
                assert!(result.is_none(), "[{}] Stream should not produce output during warmup", test_name);
            }
        }
        
        Ok(())
    }
    
    fn check_prb_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = PrbParams {
            smooth_data: Some(false),
            smooth_period: None,
            regression_period: Some(50),
            polynomial_order: Some(2),
            regression_offset: None,
            ndev: None,
            equ_from: None,
        };
        let input = PrbInput::from_candles(&candles, "close", params);
        let output = prb_with_kernel(&input, kernel)?;
        
        #[cfg(debug_assertions)]
        {
            // Check all three output arrays for poison values
            for arr in [&output.values[..], &output.upper_band[..], &output.lower_band[..]] {
                for (idx, &val) in arr.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }
                    
                    let bits = val.to_bits();
                    
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                            test_name, val, bits, idx
                        );
                    }
                    
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                            test_name, val, bits, idx
                        );
                    }
                    
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                            test_name, val, bits, idx
                        );
                    }
                }
            }
        }
        
        Ok(())
    }
    
    // Batch tests for API parity
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = PrbBatchBuilder::new()
            .kernel(kernel)
            .smooth_data(false)
            .apply_candles(&c, "close")?;
        // default row should exist and match length
        let def = PrbParams { smooth_data: Some(false), ..PrbParams::default() };
        let row = out.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = PrbBatchBuilder::new()
            .kernel(kernel)
            .smooth_data(false)
            .smooth_period_range(8, 12, 2)
            .regression_period_range(50, 60, 5)
            .polynomial_order_range(1, 3, 1)
            .regression_offset_range(0, 2, 1)
            .apply_candles(&c, "close")?;
        let expected = 3 * 3 * 3 * 3;
        assert_eq!(out.combos.len(), expected);
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = PrbBatchBuilder::new()
            .kernel(kernel)
            .smooth_data(false)
            .regression_period_range(9, 12, 1)
            .apply_candles(&c, "close")?;
        for arr in [&out.values[..], &out.upper_band[..], &out.lower_band[..]] {
            for (idx, &v) in arr.iter().enumerate() {
                if v.is_nan() { continue; }
                let b = v.to_bits();
                assert!(b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                    "[{}] poison at flat index {}", test, idx);
            }
        }
        Ok(())
    }
    
    // Macro to generate tests for all kernel variants
    macro_rules! generate_all_prb_tests {
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
    
    generate_all_prb_tests!(
        check_prb_accuracy,
        check_prb_partial_params,
        check_prb_zero_period,
        check_prb_period_exceeds_length,
        check_prb_very_small_dataset,
        check_prb_empty_input,
        check_prb_invalid_smooth_period,
        check_prb_invalid_order,
        check_prb_reinput,
        check_prb_nan_handling,
        check_prb_streaming,
        check_prb_no_poison
    );
    
    // Batch test generation macro
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);
}
