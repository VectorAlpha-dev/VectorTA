//! # Ehlers Predictive Moving Average (Ehlers PMA)
//!
//! A predictive moving average developed by John Ehlers that uses double smoothing
//! with weighted moving averages to create a predictive signal and trigger line.
//!
//! ## Algorithm
//! The indicator computes two weighted moving averages (WMAs) with fixed weights:
//! 1. First WMA: 7-period weighted average with weights [7,6,5,4,3,2,1]/28
//! 2. Second WMA: Same weighted average applied to the first WMA  
//! 3. Predict: 2 × WMA1 - WMA2 (extrapolation)
//! 4. Trigger: 4-period weighted average of Predict with weights [4,3,2,1]/10
//!
//! ## Parameters
//! This indicator has no configurable parameters. The weights and periods are fixed
//! as per Ehlers' original specification.
//!
//! ## Errors
//! - **EmptyInputData**: Input data slice is empty
//! - **AllValuesNaN**: All input data values are `NaN`
//! - **NotEnoughValidData**: Insufficient valid data points (minimum 13 required)
//!
//! ## Returns
//! - **`Ok(EhlersPmaOutput)`** containing:
//!   - `predict`: Leading indicator line
//!   - `trigger`: Smoothed signal line for crossover detection
//!
//! ## Example
//! ```rust
//! use my_project::other_indicators::ehlers_pma::*;
//! 
//! let data = vec![100.0; 20]; // Sample data
//! let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
//! let output = ehlers_pma(&input).unwrap();
//! assert_eq!(output.predict.len(), data.len());
//! assert_eq!(output.trigger.len(), data.len());
//! ```

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EhlersPmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersPmaOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EhlersPmaParams;

impl Default for EhlersPmaParams {
    #[inline]
    fn default() -> Self { Self }
}

#[derive(Debug, Clone)]
pub struct EhlersPmaInput<'a> {
    pub data: EhlersPmaData<'a>,
    pub params: EhlersPmaParams,
}

impl<'a> EhlersPmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhlersPmaParams) -> Self {
        Self {
            data: EhlersPmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EhlersPmaParams) -> Self {
        Self {
            data: EhlersPmaData::Slice(sl),
            params: p,
        }
    }
    
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhlersPmaParams::default())
    }
}

impl<'a> AsRef<[f64]> for EhlersPmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersPmaData::Slice(s) => s,
            EhlersPmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersPmaBuilder {
    kernel: Kernel,
}

impl Default for EhlersPmaBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersPmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EhlersPmaOutput, EhlersPmaError> {
        let i = EhlersPmaInput::from_candles(c, "close", EhlersPmaParams::default());
        ehlers_pma_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhlersPmaOutput, EhlersPmaError> {
        let i = EhlersPmaInput::from_slice(d, EhlersPmaParams::default());
        ehlers_pma_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersPmaStream, EhlersPmaError> {
        EhlersPmaStream::try_new(EhlersPmaParams::default())
    }
}

#[derive(Debug, Error)]
pub enum EhlersPmaError {
    #[error("ehlers_pma: Input data slice is empty.")]
    EmptyInputData,
    
    #[error("ehlers_pma: All values are NaN.")]
    AllValuesNaN,
    
    #[error("ehlers_pma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    
    #[error("ehlers_pma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

/// Computes the Ehlers Predictive Moving Average with automatic kernel selection.
///
/// This is the main entry point for calculating the indicator. It automatically
/// detects and uses the best available SIMD kernel for your CPU architecture.
///
/// # Arguments
/// * `input` - Input data containing price series and parameters
///
/// # Returns
/// * `Ok(EhlersPmaOutput)` - Predict and trigger arrays on success
/// * `Err(EhlersPmaError)` - Error if input validation fails
#[inline]
pub fn ehlers_pma(input: &EhlersPmaInput) -> Result<EhlersPmaOutput, EhlersPmaError> {
    ehlers_pma_with_kernel(input, Kernel::Auto)
}

/// Computes the Ehlers Predictive Moving Average with explicit kernel selection.
///
/// This variant allows manual selection of the computation kernel for testing
/// or when specific performance characteristics are required.
///
/// # Arguments
/// * `input` - Input data containing price series and parameters  
/// * `kernel` - Explicit kernel selection (Scalar, AVX2, AVX512, or Auto)
///
/// # Returns
/// * `Ok(EhlersPmaOutput)` - Predict and trigger arrays on success
/// * `Err(EhlersPmaError)` - Error if input validation fails
pub fn ehlers_pma_with_kernel(input: &EhlersPmaInput, kernel: Kernel) -> Result<EhlersPmaOutput, EhlersPmaError> {
    let data = input.as_ref();
    let len = data.len();
    
    if len == 0 {
        return Err(EhlersPmaError::EmptyInputData);
    }
    
    let first_valid = data.iter().position(|x| !x.is_nan()).ok_or(EhlersPmaError::AllValuesNaN)?;
    
    // TradingView parity: src is effectively data[1] on historical bars.
    // With a 1-bar lag, first WMA7 needs indices [i-1..i-7] ⇒ i ≥ first_valid + 7
    // Then WMA7(WMA7) ⇒ i ≥ first_valid + 13
    const MIN_REQUIRED: usize = 14;
    if len - first_valid < MIN_REQUIRED {
        return Err(EhlersPmaError::NotEnoughValidData {
            needed: MIN_REQUIRED,
            valid: len - first_valid,
        });
    }
    
    // warmups with 1-bar lag
    let warm_wma1 = first_valid + 7;
    let warm_wma2 = first_valid + 13;
    let warm_predict = warm_wma2;
    let warm_trigger = warm_wma2 + 3;

    // allocate with minimal writes
    let mut wma1    = alloc_with_nan_prefix(len, warm_wma1);
    let mut wma2    = alloc_with_nan_prefix(len, warm_wma2);
    let mut predict = alloc_with_nan_prefix(len, warm_predict);
    let mut trigger = alloc_with_nan_prefix(len, warm_trigger);

    // Determine which kernel to use
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Compute using appropriate kernel
    unsafe {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        match chosen {
            Kernel::Avx512 | Kernel::Avx512Batch =>
                ehlers_pma_avx512(data, &mut wma1, &mut wma2, &mut predict, &mut trigger, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Avx2   | Kernel::Avx2Batch   =>
                ehlers_pma_avx2  (data, &mut wma1, &mut wma2, &mut predict, &mut trigger, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Scalar | Kernel::ScalarBatch =>
                ehlers_pma_scalar(data, &mut wma1, &mut wma2, &mut predict, &mut trigger, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Auto => unreachable!(),
        }
        
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            // Accept ScalarBatch explicitly in no-AVX builds too
            let _ = chosen; // keep var used
            ehlers_pma_scalar(data, &mut wma1, &mut wma2, &mut predict, &mut trigger, 
                              warm_wma1, warm_wma2, warm_predict, warm_trigger);
        }
    }

    Ok(EhlersPmaOutput { predict, trigger })
}

// Scalar kernel implementation
#[inline]
pub fn ehlers_pma_scalar(
    data: &[f64],
    wma1: &mut [f64],
    wma2: &mut [f64],
    predict: &mut [f64],
    trigger: &mut [f64],
    warm_wma1: usize,
    warm_wma2: usize,
    warm_predict: usize,
    warm_trigger: usize,
) {
    let len = data.len();
    
    // WMA7 on src_lag: src[t] = data[t-1]
    for i in warm_wma1..len {
        // uses data[i-1..i-7]
        wma1[i] = (7.0 * data[i-1] + 
                   6.0 * data[i-2] +
                   5.0 * data[i-3] +
                   4.0 * data[i-4] +
                   3.0 * data[i-5] +
                   2.0 * data[i-6] +
                   1.0 * data[i-7]) / 28.0;
    }

    // WMA7 on WMA7
    for i in warm_wma2..len {
        wma2[i] = (7.0 * wma1[i] +
                   6.0 * wma1[i-1] +
                   5.0 * wma1[i-2] +
                   4.0 * wma1[i-3] +
                   3.0 * wma1[i-4] +
                   2.0 * wma1[i-5] +
                   1.0 * wma1[i-6]) / 28.0;
    }

    // Predict = 2*WMA1 - WMA2
    for i in warm_predict..len {
        predict[i] = 2.0 * wma1[i] - wma2[i];
    }

    // Trigger = WMA4(predict) with weights [4,3,2,1]/10
    // Only calculate trigger if we have enough data
    let warm_trigger_safe = warm_trigger.min(len);
    for i in warm_trigger_safe..len {
        trigger[i] = (4.0 * predict[i] +
                      3.0 * predict[i-1] +
                      2.0 * predict[i-2] +
                      1.0 * predict[i-3]) / 10.0;
    }
}

// AVX2 kernel stub - falls back to scalar for now
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn ehlers_pma_avx2(
    data: &[f64],
    wma1: &mut [f64],
    wma2: &mut [f64],
    predict: &mut [f64],
    trigger: &mut [f64],
    warm_wma1: usize,
    warm_wma2: usize,
    warm_predict: usize,
    warm_trigger: usize,
) {
    // TODO: Implement AVX2 optimized version
    // For now, fall back to scalar
    ehlers_pma_scalar(data, wma1, wma2, predict, trigger,
                      warm_wma1, warm_wma2, warm_predict, warm_trigger)
}

// AVX512 kernel stub - falls back to scalar for now
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ehlers_pma_avx512(
    data: &[f64],
    wma1: &mut [f64],
    wma2: &mut [f64],
    predict: &mut [f64],
    trigger: &mut [f64],
    warm_wma1: usize,
    warm_wma2: usize,
    warm_predict: usize,
    warm_trigger: usize,
) {
    // TODO: Implement AVX512 optimized version
    // For now, fall back to scalar
    ehlers_pma_scalar(data, wma1, wma2, predict, trigger,
                      warm_wma1, warm_wma2, warm_predict, warm_trigger)
}

// Kernel-aware zero-copy flat output
#[inline]
pub fn ehlers_pma_into_flat_with_kernel(
    out: &mut [f64],
    input: &EhlersPmaInput,
    kernel: Kernel,
) -> Result<(usize, usize), EhlersPmaError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 { return Err(EhlersPmaError::EmptyInputData); }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(EhlersPmaError::AllValuesNaN)?;
    const MIN_REQUIRED: usize = 14;
    if len - first < MIN_REQUIRED {
        return Err(EhlersPmaError::NotEnoughValidData { needed: MIN_REQUIRED, valid: len - first });
    }

    let rows = 2usize;
    let cols = len;
    if out.len() != rows * cols {
        return Err(EhlersPmaError::InvalidPeriod { period: rows * cols, data_len: out.len() });
    }

    let mut tmp_mu = make_uninit_matrix(2, len);
    // wma1 starts at first+6, wma2 at first+12
    init_matrix_prefixes(&mut tmp_mu, len, &[first + 7, first + 13]);
    let (wma1_mu, wma2_mu) = tmp_mu.split_at_mut(len);
    let wma1 = unsafe { core::slice::from_raw_parts_mut(wma1_mu.as_mut_ptr() as *mut f64, len) };
    let wma2 = unsafe { core::slice::from_raw_parts_mut(wma2_mu.as_mut_ptr() as *mut f64, len) };

    let (predict_flat, trigger_flat) = out.split_at_mut(cols);

    let warm_wma1    = first + 7;
    let warm_wma2    = first + 13;
    let warm_predict = warm_wma2;
    let warm_trigger = warm_wma2 + 3;

    // init warm prefixes once, but ensure we don't exceed array bounds
    for v in &mut predict_flat[..warm_predict.min(len)] { *v = f64::NAN; }
    for v in &mut trigger_flat[..warm_trigger.min(len)] { *v = f64::NAN; }

    let chosen = match kernel { Kernel::Auto => detect_best_kernel(), k => k };

    unsafe {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        match chosen {
            Kernel::Avx512 | Kernel::Avx512Batch =>
                ehlers_pma_avx512(data, wma1, wma2, predict_flat, trigger_flat, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Avx2   | Kernel::Avx2Batch   =>
                ehlers_pma_avx2  (data, wma1, wma2, predict_flat, trigger_flat, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Scalar | Kernel::ScalarBatch =>
                ehlers_pma_scalar(data, wma1, wma2, predict_flat, trigger_flat, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Auto => unreachable!(),
        }

        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            let _ = chosen; // keep var used
            ehlers_pma_scalar(
                data, wma1, wma2, predict_flat, trigger_flat,
                warm_wma1, warm_wma2, warm_predict, warm_trigger
            );
        }
    }

    Ok((rows, cols))
}

// Flat "into" API for multi-output (rows=2)
#[inline]
pub fn ehlers_pma_into_flat(
    out: &mut [f64],
    input: &EhlersPmaInput,
) -> Result<(usize, usize), EhlersPmaError> {
    ehlers_pma_into_flat_with_kernel(out, input, Kernel::Auto)
}

// Kernel-aware zero-copy "into_slices"
#[inline]
pub fn ehlers_pma_into_slices_with_kernel(
    predict: &mut [f64],
    trigger: &mut [f64],
    input: &EhlersPmaInput,
    kernel: Kernel,
) -> Result<(), EhlersPmaError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 { return Err(EhlersPmaError::EmptyInputData); }
    if predict.len() != len || trigger.len() != len {
        return Err(EhlersPmaError::InvalidPeriod { period: len, data_len: predict.len().min(trigger.len()) });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(EhlersPmaError::AllValuesNaN)?;
    const MIN_REQUIRED: usize = 14;
    if len - first < MIN_REQUIRED {
        return Err(EhlersPmaError::NotEnoughValidData { needed: MIN_REQUIRED, valid: len - first });
    }

    let warm_wma1    = first + 7;
    let warm_wma2    = first + 13;
    let warm_predict = warm_wma2;
    let warm_trigger = warm_wma2 + 3;

    // temporaries for wma1/wma2
    let mut tmp_mu = make_uninit_matrix(2, len);
    init_matrix_prefixes(&mut tmp_mu, len, &[warm_wma1, warm_wma2]);
    let (wma1_mu, wma2_mu) = tmp_mu.split_at_mut(len);
    let wma1 = unsafe { core::slice::from_raw_parts_mut(wma1_mu.as_mut_ptr() as *mut f64, len) };
    let wma2 = unsafe { core::slice::from_raw_parts_mut(wma2_mu.as_mut_ptr() as *mut f64, len) };

    // warm prefixes
    for v in &mut predict[..warm_predict] { *v = f64::NAN; }
    for v in &mut trigger[..warm_trigger] { *v = f64::NAN; }

    let chosen = match kernel { Kernel::Auto => detect_best_kernel(), k => k };

    unsafe {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        match chosen {
            Kernel::Avx512 | Kernel::Avx512Batch =>
                ehlers_pma_avx512(data, wma1, wma2, predict, trigger, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Avx2   | Kernel::Avx2Batch   =>
                ehlers_pma_avx2  (data, wma1, wma2, predict, trigger, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Scalar | Kernel::ScalarBatch =>
                ehlers_pma_scalar(data, wma1, wma2, predict, trigger, warm_wma1, warm_wma2, warm_predict, warm_trigger),
            Kernel::Auto => unreachable!(),
        }

        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            let _ = chosen; // keep var used
            ehlers_pma_scalar(
                data, wma1, wma2, predict, trigger,
                warm_wma1, warm_wma2, warm_predict, warm_trigger
            );
        }
    }
    Ok(())
}

// Zero-copy "into_slices"
#[inline]
pub fn ehlers_pma_into_slices(
    predict: &mut [f64],
    trigger: &mut [f64],
    input: &EhlersPmaInput,
) -> Result<(), EhlersPmaError> {
    ehlers_pma_into_slices_with_kernel(predict, trigger, input, Kernel::Auto)
}

// Streaming implementation
#[derive(Debug, Clone)]
pub struct EhlersPmaStream {
    // feed WMA7 with prev tick to replicate TV's [1] lag
    prev: Option<f64>,
    wma1_buffer: Vec<f64>,     // holds 7 values of src_lag
    predict_buffer: Vec<f64>,  // holds 4 values of predict
    wma1_values: Vec<f64>,     // holds 7 values of wma1
}

impl EhlersPmaStream {
    pub fn try_new(_params: EhlersPmaParams) -> Result<Self, EhlersPmaError> {
        Ok(Self {
            prev: None,
            wma1_buffer: Vec::with_capacity(7),
            predict_buffer: Vec::with_capacity(4),
            wma1_values: Vec::with_capacity(7),
        })
    }
    
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        // src_lag = previous tick
        let src_lag = match self.prev {
            None => {
                self.prev = Some(value);
                return None; // need one tick to establish lag
            }
            Some(p) => { self.prev = Some(value); p }
        };

        // maintain 7-wide buffer of src_lag
        if self.wma1_buffer.len() < 7 { self.wma1_buffer.push(src_lag); }
        else { self.wma1_buffer.rotate_left(1); self.wma1_buffer[6] = src_lag; }
        if self.wma1_buffer.len() < 7 { return None; }

        let wma1 = (7.0*self.wma1_buffer[6] + 6.0*self.wma1_buffer[5] + 5.0*self.wma1_buffer[4]
                  + 4.0*self.wma1_buffer[3] + 3.0*self.wma1_buffer[2] + 2.0*self.wma1_buffer[1]
                  + 1.0*self.wma1_buffer[0]) / 28.0;

        if self.wma1_values.len() < 7 { self.wma1_values.push(wma1); }
        else { self.wma1_values.rotate_left(1); self.wma1_values[6] = wma1; }
        if self.wma1_values.len() < 7 { return None; }

        let wma2 = (7.0*self.wma1_values[6] + 6.0*self.wma1_values[5] + 5.0*self.wma1_values[4]
                  + 4.0*self.wma1_values[3] + 3.0*self.wma1_values[2] + 2.0*self.wma1_values[1]
                  + 1.0*self.wma1_values[0]) / 28.0;

        let predict = 2.0 * wma1 - wma2;

        if self.predict_buffer.len() < 4 { self.predict_buffer.push(predict); }
        else { self.predict_buffer.rotate_left(1); self.predict_buffer[3] = predict; }
        if self.predict_buffer.len() < 4 { return Some((predict, f64::NAN)); }

        let trigger = (4.0*self.predict_buffer[3] + 3.0*self.predict_buffer[2]
                     + 2.0*self.predict_buffer[1] + 1.0*self.predict_buffer[0]) / 10.0;

        Some((predict, trigger))
    }
    
    pub fn reset(&mut self) {
        self.prev = None;
        self.wma1_buffer.clear();
        self.predict_buffer.clear();
        self.wma1_values.clear();
    }
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_pma")]
#[pyo3(signature = (data, kernel=None))]
pub fn ehlers_pma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = EhlersPmaInput::from_slice(slice, EhlersPmaParams::default());

    let out = py
        .allow_threads(|| ehlers_pma_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((out.predict.into_pyarray(py), out.trigger.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_pma_flat")]
#[pyo3(signature = (data, kernel=None))]
pub fn ehlers_pma_flat_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArray1;
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let rows = 2usize;
    let cols = slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let input = EhlersPmaInput::from_slice(slice, EhlersPmaParams::default());
    py.allow_threads(|| ehlers_pma_into_flat_with_kernel(out_slice, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    // add names for downstream parity with multi-output indicators
    dict.set_item("lines", vec!["predict", "trigger"])?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_pma_batch")]
#[pyo3(signature = (data, _period_range=(0,0,0), _offset_range=(0.0,0.0,0.0), _sigma_range=(0.0,0.0,0.0), kernel=None))]
pub fn ehlers_pma_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    _period_range: (usize,usize,usize),
    _offset_range: (f64,f64,f64),
    _sigma_range: (f64,f64,f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    ehlers_pma_flat_py(py, data, kernel)
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersPmaStream")]
pub struct EhlersPmaStreamPy {
    stream: EhlersPmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersPmaStreamPy {
    #[new]
    #[pyo3(signature = (period=None, offset=None, sigma=None))]
    fn new(period: Option<usize>, offset: Option<f64>, sigma: Option<f64>) -> PyResult<Self> {
        // Ignore the parameters - they're just for API parity
        let _ = (period, offset, sigma);
        let stream = EhlersPmaStream::try_new(EhlersPmaParams::default())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }
    
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
    
    fn reset(&mut self) {
        self.stream.reset();
    }
}

// WASM bindings
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EhlersPmaJsOutput {
    pub values: Vec<f64>, // [predict..., trigger...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ehlers_pma)]
pub fn ehlers_pma_js(data: &[f64]) -> Result<JsValue, JsValue> {
    let input = EhlersPmaInput::from_slice(data, EhlersPmaParams::default());
    let mut values = vec![0.0_f64; 2 * data.len()];
    let (rows, cols) = ehlers_pma_into_flat(&mut values, &input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let result = EhlersPmaJsOutput { values, rows, cols };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_pma_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_pma_free(ptr: *mut f64, len: usize) {
    unsafe { let _ = Vec::from_raw_parts(ptr, 2*len, 2*len); }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ehlers_pma_into)]
pub fn ehlers_pma_into_js(in_ptr: *const f64, out_ptr: *mut f64, len: usize) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = core::slice::from_raw_parts(in_ptr, len);
        let out  = core::slice::from_raw_parts_mut(out_ptr, 2 * len);

        // In-place guard like alma_into
        if core::ptr::eq(in_ptr, out_ptr as *const f64) {
            let mut tmp = vec![0.0f64; 2 * len];
            let input = EhlersPmaInput::from_slice(data, EhlersPmaParams::default());
            ehlers_pma_into_flat(&mut tmp, &input).map_err(|e| JsValue::from_str(&e.to_string()))?;
            out.copy_from_slice(&tmp);
            return Ok(());
        }

        let input = EhlersPmaInput::from_slice(data, EhlersPmaParams::default());
        ehlers_pma_into_flat(out, &input).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    // Comprehensive test functions following ALMA pattern
    fn check_ehlers_pma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EhlersPmaInput::from_candles(&candles, "close", EhlersPmaParams::default());
        
        let out = ehlers_pma_with_kernel(&input, kernel)?;
        
        // Reference values from PineScript using close source:
        let expected_predict_last_five = [
            59161.97066327,
            59240.51785714,
            59260.29846939,
            59225.19005102,
            59192.78443878,
        ];
        let expected_trigger_last_five = [
            59020.56403061,
            59141.96938776,
            59214.56709184,
            59232.46619898,
            59220.78227041,
        ];
        
        // Check last 5 predict values
        let start = out.predict.len().saturating_sub(5);
        for (i, &val) in out.predict[start..].iter().enumerate() {
            let diff = (val - expected_predict_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] Ehlers PMA predict {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_predict_last_five[i]
            );
        }
        
        // Check last 5 trigger values
        for (i, &val) in out.trigger[start..].iter().enumerate() {
            let diff = (val - expected_trigger_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] Ehlers PMA trigger {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_trigger_last_five[i]
            );
        }
        
        Ok(())
    }

    fn check_ehlers_pma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersPmaInput::with_default_candles(&candles);
        match input.data {
            EhlersPmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EhlersPmaData::Candles"),
        }
        let output = ehlers_pma_with_kernel(&input, kernel)?;
        assert_eq!(output.predict.len(), candles.close.len());
        assert_eq!(output.trigger.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_pma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EhlersPmaInput::from_slice(&empty, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersPmaError::EmptyInputData)),
            "[{}] Ehlers PMA should fail with empty input", 
            test_name
        );
        Ok(())
    }

    fn check_ehlers_pma_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 20];
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersPmaError::AllValuesNaN)),
            "[{}] Ehlers PMA should fail with all NaN values", 
            test_name
        );
        Ok(())
    }

    fn check_ehlers_pma_insufficient_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersPmaError::NotEnoughValidData { .. })),
            "[{}] Ehlers PMA should fail with insufficient data", 
            test_name
        );
        Ok(())
    }
    
    fn check_ehlers_pma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let input = EhlersPmaInput::from_slice(&single_point, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            res.is_err(), 
            "[{}] Ehlers PMA should fail with single data point", 
            test_name
        );
        
        let two_points = [42.0, 43.0];
        let input2 = EhlersPmaInput::from_slice(&two_points, EhlersPmaParams::default());
        let res2 = ehlers_pma_with_kernel(&input2, kernel);
        assert!(
            res2.is_err(), 
            "[{}] Ehlers PMA should fail with only two data points", 
            test_name
        );
        Ok(())
    }

    fn check_ehlers_pma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersPmaInput::from_candles(&candles, "hl2", EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel)?;
        
        assert_eq!(res.predict.len(), candles.close.len());
        assert_eq!(res.trigger.len(), candles.close.len());
        
        // Check that values after warmup are valid
        if res.predict.len() > 20 {
            for (i, &val) in res.predict[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN in predict at index {}",
                    test_name,
                    20 + i
                );
            }
        }
        if res.trigger.len() > 20 {
            for (i, &val) in res.trigger[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN in trigger at index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_ehlers_pma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // Use HL2 as input
        let hl2: Vec<f64> = candles.high.iter()
            .zip(candles.low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        // Batch calculation
        let input = EhlersPmaInput::from_slice(&hl2, EhlersPmaParams::default());
        let batch_output = ehlers_pma_with_kernel(&input, kernel)?;

        // Streaming calculation
        let mut stream = EhlersPmaStream::try_new(EhlersPmaParams::default())?;
        let mut stream_predict = Vec::with_capacity(hl2.len());
        let mut stream_trigger = Vec::with_capacity(hl2.len());

        for &value in &hl2 {
            match stream.update(value) {
                Some((p, t)) => {
                    stream_predict.push(p);
                    stream_trigger.push(t);
                }
                None => {
                    stream_predict.push(f64::NAN);
                    stream_trigger.push(f64::NAN);
                }
            }
        }

        // Compare results
        assert_eq!(batch_output.predict.len(), stream_predict.len());
        assert_eq!(batch_output.trigger.len(), stream_trigger.len());

        for (i, ((&bp, &bt), (&sp, &st))) in batch_output.predict.iter()
            .zip(batch_output.trigger.iter())
            .zip(stream_predict.iter().zip(stream_trigger.iter()))
            .enumerate() 
        {
            if bp.is_nan() && sp.is_nan() {
                continue;
            }
            if bt.is_nan() && st.is_nan() {
                continue;
            }
            
            let predict_diff = (bp - sp).abs();
            let trigger_diff = (bt - st).abs();
            
            assert!(
                predict_diff < 1e-9,
                "[{}] Predict streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, bp, sp, predict_diff
            );
            assert!(
                trigger_diff < 1e-9,
                "[{}] Trigger streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, bt, st, trigger_diff
            );
        }
        Ok(())
    }

    fn check_ehlers_pma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // First pass with HL2
        let hl2: Vec<f64> = candles.high.iter()
            .zip(candles.low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        let first_input = EhlersPmaInput::from_slice(&hl2, EhlersPmaParams::default());
        let first_result = ehlers_pma_with_kernel(&first_input, kernel)?;

        // Second pass using predict as input
        let second_input = EhlersPmaInput::from_slice(&first_result.predict, EhlersPmaParams::default());
        let second_result = ehlers_pma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.predict.len(), first_result.predict.len());
        assert_eq!(second_result.trigger.len(), first_result.trigger.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ehlers_pma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = EhlersPmaInput::from_candles(&candles, "hl2", EhlersPmaParams::default());
        let output = ehlers_pma_with_kernel(&input, kernel)?;

        for (arr_name, arr) in [("predict", &output.predict[..]), ("trigger", &output.trigger[..])] {
            for (i, &val) in arr.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }
                let bits = val.to_bits();
                
                if bits == 0x1111_1111_1111_1111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in {} at index {}",
                        test_name, val, bits, arr_name, i
                    );
                }
                if bits == 0x2222_2222_2222_2222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in {} at index {}",
                        test_name, val, bits, arr_name, i
                    );
                }
                if bits == 0x3333_3333_3333_3333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in {} at index {}",
                        test_name, val, bits, arr_name, i
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ehlers_pma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_ehlers_pma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = prop::collection::vec(
            (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
            14..400, // Minimum 14 values required with 1-bar lag
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |data| {
                let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
                
                let result = ehlers_pma_with_kernel(&input, kernel).unwrap();
                let ref_result = ehlers_pma_with_kernel(&input, Kernel::Scalar).unwrap();
                
                // Check that lengths match
                prop_assert_eq!(result.predict.len(), data.len());
                prop_assert_eq!(result.trigger.len(), data.len());
                
                // Compare with reference scalar implementation
                for i in 0..data.len() {
                    let p = result.predict[i];
                    let t = result.trigger[i];
                    let ref_p = ref_result.predict[i];
                    let ref_t = ref_result.trigger[i];
                    
                    if !p.is_finite() || !ref_p.is_finite() {
                        prop_assert_eq!(p.to_bits(), ref_p.to_bits(), 
                            "Predict finite/NaN mismatch at idx {}: {} vs {}", i, p, ref_p);
                        continue;
                    }
                    if !t.is_finite() || !ref_t.is_finite() {
                        prop_assert_eq!(t.to_bits(), ref_t.to_bits(), 
                            "Trigger finite/NaN mismatch at idx {}: {} vs {}", i, t, ref_t);
                        continue;
                    }
                    
                    let p_ulp_diff = p.to_bits().abs_diff(ref_p.to_bits());
                    let t_ulp_diff = t.to_bits().abs_diff(ref_t.to_bits());
                    
                    prop_assert!(
                        (p - ref_p).abs() <= 1e-9 || p_ulp_diff <= 4,
                        "Predict mismatch idx {}: {} vs {} (ULP={})", i, p, ref_p, p_ulp_diff
                    );
                    prop_assert!(
                        (t - ref_t).abs() <= 1e-9 || t_ulp_diff <= 4,
                        "Trigger mismatch idx {}: {} vs {} (ULP={})", i, t, ref_t, t_ulp_diff
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    // Test generation macro
    macro_rules! generate_all_ehlers_pma_tests {
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

    fn check_ehlers_pma_invalid_output_len(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = vec![1.0; 20];
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let mut p = vec![0.0; 19];
        let mut t = vec![0.0; 20];
        let res = ehlers_pma_into_slices(&mut p, &mut t, &input);
        assert!(matches!(res, Err(EhlersPmaError::InvalidPeriod { .. })),
            "[{}] expected InvalidPeriod", test_name);
        Ok(())
    }

    // Generate all kernel-specific tests
    generate_all_ehlers_pma_tests!(
        check_ehlers_pma_accuracy,
        check_ehlers_pma_default_candles,
        check_ehlers_pma_empty_input,
        check_ehlers_pma_all_nan,
        check_ehlers_pma_insufficient_data,
        check_ehlers_pma_very_small_dataset,
        check_ehlers_pma_nan_handling,
        check_ehlers_pma_streaming,
        check_ehlers_pma_reinput,
        check_ehlers_pma_no_poison,
        check_ehlers_pma_invalid_output_len
    );

    #[cfg(feature = "proptest")]
    generate_all_ehlers_pma_tests!(check_ehlers_pma_property);
    
    // Additional targeted tests for specific edge cases not covered above
    #[test]
    fn test_ehlers_pma_basic() {
        let data = vec![
            59161.97066327, 59240.51785714, 59260.29846939,
            59225.19005102, 59192.78443878, 59200.0, 59180.0,
            59220.0, 59250.0, 59230.0, 59210.0, 59240.0,
            59260.0, 59280.0, 59270.0, 59250.0, 59300.0  // Added 17th value for trigger test
        ];
        
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let result = ehlers_pma(&input).unwrap();
        
        // Check that we have output
        assert_eq!(result.predict.len(), data.len());
        assert_eq!(result.trigger.len(), data.len());
        
        // First 13 values should be NaN for predict (with 1-bar lag)
        for i in 0..13 {
            assert!(result.predict[i].is_nan());
        }
        
        // First 16 values should be NaN for trigger
        for i in 0..16 {
            assert!(result.trigger[i].is_nan());
        }
        
        // Values after warmup should be valid
        assert!(!result.predict[13].is_nan());
        assert!(!result.trigger[16].is_nan());
    }
    
    #[test]
    fn test_ehlers_pma_into_flat() {
        let data = vec![
            59161.97066327, 59240.51785714, 59260.29846939,
            59225.19005102, 59192.78443878, 59200.0, 59180.0,
            59220.0, 59250.0, 59230.0, 59210.0, 59240.0,
            59260.0, 59280.0, 59270.0, 59250.0, 59300.0  // Added 17th value for trigger test
        ];
        
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let mut output = vec![0.0; data.len() * 2];
        let (rows, cols) = ehlers_pma_into_flat(&mut output, &input).unwrap();
        
        assert_eq!(rows, 2);
        assert_eq!(cols, data.len());
        
        // Check first half is predict, second half is trigger
        let (predict_flat, trigger_flat) = output.split_at(data.len());
        
        // First 13 values should be NaN for predict (with 1-bar lag)
        for i in 0..13 {
            assert!(predict_flat[i].is_nan());
        }
        
        // First 16 values should be NaN for trigger
        for i in 0..16 {
            assert!(trigger_flat[i].is_nan());
        }
    }
    
    #[test]
    fn check_ehlers_pma_into_slices_noalloc() -> Result<(), Box<dyn std::error::Error>> {
        use crate::utilities::data_loader::read_candles_from_csv;
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = EhlersPmaInput::with_default_candles(&c);

        let batch = ehlers_pma(&input).unwrap();

        let mut p = vec![0.0; c.close.len()];
        let mut t = vec![0.0; c.close.len()];
        ehlers_pma_into_slices(&mut p, &mut t, &input).unwrap();

        assert_eq!(p.len(), batch.predict.len());
        assert_eq!(t.len(), batch.trigger.len());
        for i in 0..p.len() {
            let (a,b) = (p[i], batch.predict[i]);
            if a.is_nan() || b.is_nan() { assert_eq!(a.to_bits(), b.to_bits()); } else { assert!((a-b).abs() < 1e-12); }
            let (a2,b2) = (t[i], batch.trigger[i]);
            if a2.is_nan() || b2.is_nan() { assert_eq!(a2.to_bits(), b2.to_bits()); } else { assert!((a2-b2).abs() < 1e-12); }
        }
        Ok(())
    }
}