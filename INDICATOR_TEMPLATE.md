# Comprehensive Indicator Implementation Template

This template provides complete API parity with existing indicators like ALMA.rs. Every section must be filled out to ensure consistency across all indicators.

## Input Requirements

To create a new indicator, provide:
1. **Indicator Name**: `{INDICATOR_NAME}` (e.g., "macd", "stochastic")
2. **Display Name**: `{DISPLAY_NAME}` (e.g., "MACD", "Stochastic Oscillator")
3. **PineScript Implementation**: Complete algorithm logic
4. **Parameters**: List all parameters with defaults
5. **Reference Values**: 5 accurate test values from the PineScript implementation
6. **Description**: Brief description of what the indicator does
7. **Calculation Type**: Single value or multiple outputs (e.g., MACD has signal and histogram)

## Complete Rust Implementation Template

```rust
//! # {DISPLAY_NAME} ({INDICATOR_NAME_UPPER})
//!
//! {DESCRIPTION}
//!
//! ## Parameters
//! - **{param1}**: {param1_description} (default: {default1})
//! - **{param2}**: {param2_description} (default: {default2})
//!
//! ## Errors
//! - **EmptyInputData**: {indicator_name}: Input data slice is empty.
//! - **AllValuesNaN**: {indicator_name}: All input values are `NaN`.
//! - **InvalidPeriod**: {indicator_name}: Period is zero or exceeds data length.
//! - **NotEnoughValidData**: {indicator_name}: Not enough valid data points for calculation.
//! {ADDITIONAL_ERRORS}
//!
//! ## Returns
//! - **`Ok({IndicatorName}Output)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err({IndicatorName}Error)`** otherwise.

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

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

// ==================== TRAIT IMPLEMENTATIONS ====================
impl<'a> AsRef<[f64]> for {IndicatorName}Input<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            {IndicatorName}Data::Slice(slice) => slice,
            {IndicatorName}Data::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum {IndicatorName}Data<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct {IndicatorName}Output {
    pub values: Vec<f64>,
    // Add additional fields for multi-output indicators
    // pub signal: Vec<f64>,
    // pub histogram: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct {IndicatorName}Params {
    pub {param1}: Option<{param1_type}>,
    pub {param2}: Option<{param2_type}>,
    // Add all parameters here
}

impl Default for {IndicatorName}Params {
    fn default() -> Self {
        Self {
            {param1}: Some({default1}),
            {param2}: Some({default2}),
            // Set all defaults
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct {IndicatorName}Input<'a> {
    pub data: {IndicatorName}Data<'a>,
    pub params: {IndicatorName}Params,
}

impl<'a> {IndicatorName}Input<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: {IndicatorName}Params) -> Self {
        Self {
            data: {IndicatorName}Data::Candles { candles: c, source: s },
            params: p,
        }
    }
    
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: {IndicatorName}Params) -> Self {
        Self {
            data: {IndicatorName}Data::Slice(sl),
            params: p,
        }
    }
    
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", {IndicatorName}Params::default())
    }
    
    #[inline]
    pub fn get_{param1}(&self) -> {param1_type} {
        self.params.{param1}.unwrap_or({default1})
    }
    
    #[inline]
    pub fn get_{param2}(&self) -> {param2_type} {
        self.params.{param2}.unwrap_or({default2})
    }
    // Add getters for all parameters
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct {IndicatorName}Builder {
    {param1}: Option<{param1_type}>,
    {param2}: Option<{param2_type}>,
    kernel: Kernel,
}

impl Default for {IndicatorName}Builder {
    fn default() -> Self {
        Self {
            {param1}: None,
            {param2}: None,
            kernel: Kernel::Auto,
        }
    }
}

impl {IndicatorName}Builder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline(always)]
    pub fn {param1}(mut self, val: {param1_type}) -> Self {
        self.{param1} = Some(val);
        self
    }
    
    #[inline(always)]
    pub fn {param2}(mut self, val: {param2_type}) -> Self {
        self.{param2} = Some(val);
        self
    }
    
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<{IndicatorName}Output, {IndicatorName}Error> {
        let p = {IndicatorName}Params {
            {param1}: self.{param1},
            {param2}: self.{param2},
        };
        let i = {IndicatorName}Input::from_candles(c, "close", p);
        {indicator_name}_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<{IndicatorName}Output, {IndicatorName}Error> {
        let p = {IndicatorName}Params {
            {param1}: self.{param1},
            {param2}: self.{param2},
        };
        let i = {IndicatorName}Input::from_slice(d, p);
        {indicator_name}_with_kernel(&i, self.kernel)
    }
    
    #[inline(always)]
    pub fn into_stream(self) -> Result<{IndicatorName}Stream, {IndicatorName}Error> {
        let p = {IndicatorName}Params {
            {param1}: self.{param1},
            {param2}: self.{param2},
        };
        {IndicatorName}Stream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum {IndicatorName}Error {
    #[error("{indicator_name}: Input data slice is empty.")]
    EmptyInputData,
    
    #[error("{indicator_name}: All values are NaN.")]
    AllValuesNaN,
    
    #[error("{indicator_name}: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    
    #[error("{indicator_name}: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    
    // Add indicator-specific errors here
    // #[error("{indicator_name}: Invalid {param}: {value}")]
    // Invalid{Param} { value: f64 },
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
/// Main entry point with automatic kernel detection
#[inline]
pub fn {indicator_name}(input: &{IndicatorName}Input) -> Result<{IndicatorName}Output, {IndicatorName}Error> {
    {indicator_name}_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection
pub fn {indicator_name}_with_kernel(input: &{IndicatorName}Input, kernel: Kernel) -> Result<{IndicatorName}Output, {IndicatorName}Error> {
    let (data, {param1}, {param2}, first, chosen) = {indicator_name}_prepare(input, kernel)?;
    
    // CRITICAL: Use zero-copy allocation helper
    let mut out = alloc_with_nan_prefix(data.len(), first + {warmup_period});
    
    {indicator_name}_compute_into(data, {param1}, {param2}, first, chosen, &mut out);
    
    Ok({IndicatorName}Output { values: out })
}

/// Zero-allocation version for WASM and performance-critical paths
#[inline]
pub fn {indicator_name}_into_slice(dst: &mut [f64], input: &{IndicatorName}Input, kern: Kernel) -> Result<(), {IndicatorName}Error> {
    let (data, {param1}, {param2}, first, chosen) = {indicator_name}_prepare(input, kern)?;
    
    if dst.len() != data.len() {
        return Err({IndicatorName}Error::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    
    {indicator_name}_compute_into(data, {param1}, {param2}, first, chosen, dst);
    
    // Fill warmup period with NaN
    let warmup_end = first + {warmup_period};
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    
    Ok(())
}

/// Prepare and validate input data
#[inline(always)]
fn {indicator_name}_prepare<'a>(
    input: &'a {IndicatorName}Input,
    kernel: Kernel,
) -> Result<(&'a [f64], {param1_type}, {param2_type}, usize, Kernel), {IndicatorName}Error> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    
    if len == 0 {
        return Err({IndicatorName}Error::EmptyInputData);
    }
    
    let first = data.iter().position(|x| !x.is_nan())
        .ok_or({IndicatorName}Error::AllValuesNaN)?;
    
    let {param1} = input.get_{param1}();
    let {param2} = input.get_{param2}();
    
    // Validation
    if {param1} == 0 || {param1} > len {
        return Err({IndicatorName}Error::InvalidPeriod { 
            period: {param1}, 
            data_len: len 
        });
    }
    
    if len - first < {param1} {
        return Err({IndicatorName}Error::NotEnoughValidData {
            needed: {param1},
            valid: len - first,
        });
    }
    
    // Add additional parameter validation here
    
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    
    Ok((data, {param1}, {param2}, first, chosen))
}

/// Core computation dispatcher
#[inline(always)]
fn {indicator_name}_compute_into(
    data: &[f64],
    {param1}: {param1_type},
    {param2}: {param2_type},
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        // WASM SIMD128 support
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                {indicator_name}_simd128(data, {param1}, {param2}, first, out);
                return;
            }
        }
        
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                {indicator_name}_scalar(data, {param1}, {param2}, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                {indicator_name}_avx2(data, {param1}, {param2}, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                {indicator_name}_avx512(data, {param1}, {param2}, first, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                {indicator_name}_scalar(data, {param1}, {param2}, first, out)
            }
            _ => unreachable!(),
        }
    }
}

// ==================== SCALAR IMPLEMENTATION ====================
#[inline]
pub fn {indicator_name}_scalar(
    data: &[f64],
    {param1}: {param1_type},
    {param2}: {param2_type},
    first_val: usize,
    out: &mut [f64],
) {
    // TODO: Implement scalar algorithm based on PineScript
    // This is where the main calculation logic goes
    
    for i in (first_val + {warmup_period})..data.len() {
        // Implement calculation here
        out[i] = 0.0; // Placeholder
    }
}

// ==================== WASM SIMD128 IMPLEMENTATION ====================
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn {indicator_name}_simd128(
    data: &[f64],
    {param1}: {param1_type},
    {param2}: {param2_type},
    first_val: usize,
    out: &mut [f64],
) {
    use core::arch::wasm32::*;
    
    // TODO: Implement SIMD128 optimized version
    // For now, fallback to scalar
    {indicator_name}_scalar(data, {param1}, {param2}, first_val, out);
}

// ==================== AVX2 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn {indicator_name}_avx2(
    data: &[f64],
    {param1}: {param1_type},
    {param2}: {param2_type},
    first_val: usize,
    out: &mut [f64],
) {
    // TODO: Implement AVX2 optimized version
    // For now, fallback to scalar
    {indicator_name}_scalar(data, {param1}, {param2}, first_val, out);
}

// ==================== AVX512 IMPLEMENTATION ====================
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn {indicator_name}_avx512(
    data: &[f64],
    {param1}: {param1_type},
    {param2}: {param2_type},
    first_val: usize,
    out: &mut [f64],
) {
    // TODO: Implement AVX512 optimized version
    // For now, fallback to scalar
    {indicator_name}_scalar(data, {param1}, {param2}, first_val, out);
}

// ==================== STREAMING SUPPORT ====================
/// Streaming calculator for real-time updates
#[derive(Debug, Clone)]
pub struct {IndicatorName}Stream {
    buffer: Vec<f64>,
    {param1}: {param1_type},
    {param2}: {param2_type},
    index: usize,
    ready: bool,
}

impl {IndicatorName}Stream {
    pub fn try_new(params: {IndicatorName}Params) -> Result<Self, {IndicatorName}Error> {
        let {param1} = params.{param1}.unwrap_or({default1});
        let {param2} = params.{param2}.unwrap_or({default2});
        
        if {param1} == 0 {
            return Err({IndicatorName}Error::InvalidPeriod { 
                period: {param1}, 
                data_len: 0 
            });
        }
        
        Ok(Self {
            buffer: vec![0.0; {param1}],
            {param1},
            {param2},
            index: 0,
            ready: false,
        })
    }
    
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.index % self.{param1}] = value;
        self.index += 1;
        
        if self.index >= self.{param1} {
            self.ready = true;
        }
        
        if self.ready {
            // TODO: Calculate streaming value
            Some(0.0) // Placeholder
        } else {
            None
        }
    }
}

// ==================== BATCH PROCESSING ====================
/// Batch processing for parameter sweeps
#[derive(Clone, Debug)]
pub struct {IndicatorName}BatchRange {
    pub {param1}: (usize, usize, usize), // (start, end, step)
    pub {param2}: ({param2_type}, {param2_type}, {param2_type}),
}

impl Default for {IndicatorName}BatchRange {
    fn default() -> Self {
        Self {
            {param1}: ({default1}, {default1} * 10, 1),
            {param2}: ({default2}, {default2}, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct {IndicatorName}BatchBuilder {
    range: {IndicatorName}BatchRange,
    kernel: Kernel,
}

impl {IndicatorName}BatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    #[inline]
    pub fn {param1}_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.{param1} = (start, end, step);
        self
    }
    
    #[inline]
    pub fn {param1}_static(mut self, val: usize) -> Self {
        self.range.{param1} = (val, val, 0);
        self
    }
    
    pub fn apply_slice(self, data: &[f64]) -> Result<{IndicatorName}BatchOutput, {IndicatorName}Error> {
        {indicator_name}_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct {IndicatorName}BatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<{IndicatorName}Params>,
    pub rows: usize,
    pub cols: usize,
}

pub fn {indicator_name}_batch_with_kernel(
    data: &[f64],
    sweep: &{IndicatorName}BatchRange,
    k: Kernel,
) -> Result<{IndicatorName}BatchOutput, {IndicatorName}Error> {
    // TODO: Implement batch processing
    // See alma.rs for reference implementation
    unimplemented!("Batch processing not yet implemented")
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "{indicator_name}")]
#[pyo3(signature = (data, {param1}, {param2}, kernel=None))]
pub fn {indicator_name}_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    {param1}: {param1_type},
    {param2}: {param2_type},
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = {IndicatorName}Params {
        {param1}: Some({param1}),
        {param2}: Some({param2}),
    };
    let input = {IndicatorName}Input::from_slice(slice_in, params);
    
    let result_vec: Vec<f64> = py
        .allow_threads(|| {indicator_name}_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "{IndicatorName}Stream")]
pub struct {IndicatorName}StreamPy {
    stream: {IndicatorName}Stream,
}

#[cfg(feature = "python")]
#[pymethods]
impl {IndicatorName}StreamPy {
    #[new]
    fn new({param1}: {param1_type}, {param2}: {param2_type}) -> PyResult<Self> {
        let params = {IndicatorName}Params {
            {param1}: Some({param1}),
            {param2}: Some({param2}),
        };
        let stream = {IndicatorName}Stream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok({IndicatorName}StreamPy { stream })
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "{indicator_name}_batch")]
#[pyo3(signature = (data, {param1}_range, {param2}_range, kernel=None))]
pub fn {indicator_name}_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    {param1}_range: (usize, usize, usize),
    {param2}_range: ({param2_type}, {param2_type}, {param2_type}),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    // TODO: Implement batch Python bindings
    // See alma.rs for reference
    unimplemented!("Batch Python bindings not yet implemented")
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn {indicator_name}_js(
    data: &[f64],
    {param1}: {param1_type},
    {param2}: {param2_type},
) -> Result<Vec<f64>, JsValue> {
    let params = {IndicatorName}Params {
        {param1}: Some({param1}),
        {param2}: Some({param2}),
    };
    let input = {IndicatorName}Input::from_slice(data, params);
    
    let mut output = vec![0.0; data.len()];
    {indicator_name}_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn {indicator_name}_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn {indicator_name}_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn {indicator_name}_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    {param1}: {param1_type},
    {param2}: {param2_type},
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to {indicator_name}_into"));
    }
    
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        
        let params = {IndicatorName}Params {
            {param1}: Some({param1}),
            {param2}: Some({param2}),
        };
        let input = {IndicatorName}Input::from_slice(data, params);
        
        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            {indicator_name}_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            {indicator_name}_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        
        Ok(())
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
    
    fn check_{indicator_name}_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = {IndicatorName}Input::from_candles(&candles, "close", {IndicatorName}Params::default());
        let result = {indicator_name}_with_kernel(&input, kernel)?;
        
        // REFERENCE VALUES FROM PINESCRIPT
        let expected_last_five = [
            {ref_val_1},
            {ref_val_2},
            {ref_val_3},
            {ref_val_4},
            {ref_val_5},
        ];
        
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] {INDICATOR_NAME_UPPER} {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    
    fn check_{indicator_name}_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let default_params = {IndicatorName}Params {
            {param1}: None,
            {param2}: None,
        };
        let input = {IndicatorName}Input::from_candles(&candles, "close", default_params);
        let output = {indicator_name}_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_{indicator_name}_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = {IndicatorName}Input::with_default_candles(&candles);
        match input.data {
            {IndicatorName}Data::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected {IndicatorName}Data::Candles"),
        }
        let output = {indicator_name}_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_{indicator_name}_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = {IndicatorName}Params {
            {param1}: Some(0),
            {param2}: None,
        };
        let input = {IndicatorName}Input::from_slice(&input_data, params);
        let res = {indicator_name}_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] {INDICATOR_NAME_UPPER} should fail with zero period", test_name);
        Ok(())
    }
    
    fn check_{indicator_name}_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = {IndicatorName}Params {
            {param1}: Some(10),
            {param2}: None,
        };
        let input = {IndicatorName}Input::from_slice(&data_small, params);
        let res = {indicator_name}_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] {INDICATOR_NAME_UPPER} should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    
    fn check_{indicator_name}_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = {IndicatorName}Params::default();
        let input = {IndicatorName}Input::from_slice(&single_point, params);
        let res = {indicator_name}_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] {INDICATOR_NAME_UPPER} should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    
    fn check_{indicator_name}_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = {IndicatorName}Params::default();
        let input = {IndicatorName}Input::from_slice(&empty, params);
        let res = {indicator_name}_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] {INDICATOR_NAME_UPPER} should fail with empty input",
            test_name
        );
        Ok(())
    }
    
    fn check_{indicator_name}_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = {IndicatorName}Params::default();
        let input = {IndicatorName}Input::from_slice(&nan_data, params);
        let res = {indicator_name}_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] {INDICATOR_NAME_UPPER} should fail with all NaN values",
            test_name
        );
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_accuracy_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_accuracy("{indicator_name}_accuracy_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_accuracy_avx2() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_accuracy("{indicator_name}_accuracy_avx2", Kernel::Avx2)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_accuracy_avx512() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_accuracy("{indicator_name}_accuracy_avx512", Kernel::Avx512)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_partial_params_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_partial_params("{indicator_name}_partial_params_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_default_candles_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_default_candles("{indicator_name}_default_candles_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_zero_period_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_zero_period("{indicator_name}_zero_period_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_period_exceeds_length_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_period_exceeds_length("{indicator_name}_period_exceeds_length_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_very_small_dataset_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_very_small_dataset("{indicator_name}_very_small_dataset_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_empty_input_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_empty_input("{indicator_name}_empty_input_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    #[test]
    fn test_{indicator_name}_all_nan_scalar() -> Result<(), Box<dyn Error>> {
        check_{indicator_name}_all_nan("{indicator_name}_all_nan_scalar", Kernel::Scalar)?;
        Ok(())
    }
    
    // Property-based tests
    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_{indicator_name}_no_panic(data: Vec<f64>, {param1} in 1usize..100) {
            let params = {IndicatorName}Params {
                {param1}: Some({param1}),
                {param2}: Some({default2}),
            };
            let input = {IndicatorName}Input::from_slice(&data, params);
            let _ = {indicator_name}(&input);
        }
        
        #[test]
        fn test_{indicator_name}_length_preservation(size in 10usize..100) {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let params = {IndicatorName}Params::default();
            let input = {IndicatorName}Input::from_slice(&data, params);
            
            if let Ok(output) = {indicator_name}(&input) {
                prop_assert_eq!(output.values.len(), size);
            }
        }
    }
}
```

## Registration Checklist

After creating the indicator file, complete these registration steps:

### 1. Module Registration

**In `src/new_indicators/mod.rs`:**
```rust
pub mod {indicator_name};
pub use {indicator_name}::{
    {indicator_name}, {IndicatorName}Input, {IndicatorName}Output, {IndicatorName}Params
};
```

**In `src/indicators/mod.rs` (for moving averages):**
```rust
// Add to the moving_averages re-export
pub use moving_averages::{
    // ... existing indicators ...
    {indicator_name},
};
```

### 2. Python Bindings Registration

**In `src/bindings/python.rs`:**
```rust
// Add import at top
use crate::indicators::{indicator_name}::{
    {indicator_name}_py, {indicator_name}_batch_py, {IndicatorName}StreamPy
};

// In the module function, add:
m.add_function(wrap_pyfunction!({indicator_name}_py, m)?)?;
m.add_function(wrap_pyfunction!({indicator_name}_batch_py, m)?)?;
m.add_class::<{IndicatorName}StreamPy>()?;
```

### 3. WASM Bindings Registration

**In `src/bindings/wasm.rs`:**
```rust
// Export the functions (they're already marked with #[wasm_bindgen])
// No additional registration needed if using #[wasm_bindgen] attribute
```

### 4. Benchmark Registration

**In `benches/indicator_benchmark.rs`:**
```rust
// Add to imports
use my_project::indicators::{indicator_name}::{
    {indicator_name}_with_kernel, {IndicatorName}BatchBuilder, {IndicatorName}Input
};

// Add wrapper macros
make_kernel_wrappers!(
    {indicator_name}, {indicator_name}_with_kernel, {IndicatorName}InputS; 
    Scalar,Avx2,Avx512
);

// Add to single benchmarks
{indicator_name} => {IndicatorName}InputS; None;
{indicator_name}_scalar,
{indicator_name}_avx2,
{indicator_name}_avx512

// If batch is supported:
{indicator_name}_batch => {IndicatorName}InputS; Some(232);
{indicator_name}_batch_scalarbatch,
{indicator_name}_batch_avx2batch,
{indicator_name}_batch_avx512batch
```

**In `benchmarks/criterion_comparable_benchmark.py`:**
```python
# Add to indicator list
INDICATORS = [
    # ... existing indicators ...
    ("{indicator_name}", {default1}, {default2}),  # ({param1}, {param2})
]
```

### 5. Generate Test Files

Run the test generation script:
```bash
python scripts\generate_binding_tests.py {indicator_name}
```

This creates:
- `tests/python/test_{indicator_name}.py`
- `tests/wasm/test_{indicator_name}.js`

### 6. Update Documentation

**In README.md:**
- Increment indicator count
- Add to indicator list if maintaining one

## Python Test Template

```python
"""
Python binding tests for {INDICATOR_NAME_UPPER} indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built", allow_module_level=True)

class Test{IndicatorName}:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_{indicator_name}_accuracy(self, test_data):
        """Test {INDICATOR_NAME_UPPER} matches expected values from Rust tests"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['{indicator_name}']
        
        result = ta_indicators.{indicator_name}(
            close,
            {param1}=expected['default_params']['{param1}'],
            {param2}=expected['default_params']['{param2}']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-8,
            msg="{INDICATOR_NAME_UPPER} last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('{indicator_name}', result, 'close', expected['default_params'])
    
    def test_{indicator_name}_zero_period(self):
        """Test {INDICATOR_NAME_UPPER} fails with zero period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.{indicator_name}(input_data, {param1}=0)
    
    def test_{indicator_name}_empty_input(self):
        """Test {INDICATOR_NAME_UPPER} fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.{indicator_name}(empty)
```

## WASM Test Template

```javascript
/**
 * WASM binding tests for {INDICATOR_NAME_UPPER} indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose, 
    EXPECTED_OUTPUTS 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module');
        throw error;
    }
    
    testData = loadTestData();
});

test('{INDICATOR_NAME_UPPER} accuracy', async () => {
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.{indicator_name};
    
    const result = wasm.{indicator_name}_js(
        close,
        expected.defaultParams.{param1},
        expected.defaultParams.{param2}
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "{INDICATOR_NAME_UPPER} last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('{indicator_name}', result, 'close', expected.defaultParams);
});

test('{INDICATOR_NAME_UPPER} zero period', () => {
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.{indicator_name}_js(inputData, 0, {default2});
    }, /Invalid period/);
});

test('{INDICATOR_NAME_UPPER} empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.{indicator_name}_js(empty, {default1}, {default2});
    }, /empty/i);
});
```

## Critical Implementation Notes

### 1. Memory Management (MANDATORY)
- **ALWAYS** use `alloc_with_nan_prefix()` for output vectors
- **NEVER** use `vec![f64::NAN; data.len()]` for outputs
- Use `make_uninit_matrix()` for batch operations
- Small intermediate buffers (< period size) can use regular Vec

### 2. Error Handling
- All error cases must be handled
- Use thiserror for error definitions
- Return descriptive error messages

### 3. SIMD Optimizations
- Start with scalar implementation
- Add SIMD only after scalar works correctly
- Use feature gates properly for AVX2/AVX512
- Always provide WASM SIMD128 path

### 4. Testing Requirements
- Accuracy test with 5 reference values (MANDATORY)
- Edge case tests (empty, NaN, invalid params)
- Property-based tests for robustness
- Kernel-specific tests (scalar, AVX2, AVX512)

### 5. API Consistency
- Follow exact naming patterns
- Implement all builder methods
- Support both slice and candle inputs
- Provide streaming support where applicable

## Example Usage

Given this PineScript:
```pinescript
//@version=5
indicator("Custom Indicator")
custom(src, period, multiplier) =>
    sma_val = ta.sma(src, period)
    std_val = ta.stdev(src, period)
    upper = sma_val + multiplier * std_val
    upper

result = custom(close, 20, 2.0)
plot(result)
```

With reference values:
- 59286.72, 59273.53, 59204.37, 59155.93, 59026.92

You would fill in:
- `{indicator_name}`: custom
- `{IndicatorName}`: Custom
- `{INDICATOR_NAME_UPPER}`: CUSTOM
- `{param1}`: period (usize)
- `{param2}`: multiplier (f64)
- `{default1}`: 20
- `{default2}`: 2.0
- Reference values in the test

## Validation Checklist

Before considering an indicator complete:
- [ ] All placeholders replaced
- [ ] Scalar implementation complete
- [ ] All tests pass
- [ ] Python bindings registered and tested
- [ ] WASM bindings registered and tested
- [ ] Benchmarks added
- [ ] Documentation complete
- [ ] Zero-copy memory patterns used
- [ ] Error handling comprehensive
- [ ] API matches existing indicators exactly