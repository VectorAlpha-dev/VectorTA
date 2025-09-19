//! # NET MyRSI
//!
//! Combines Ehlers' MyRSI with Noise Elimination Technique for smoother RSI calculation.
//!
//! ## Parameters
//! - **data**: Input price data
//! - **period**: MyRSI calculation period (default: 14)
//!
//! ## Returns
//! - `Vec<f64>` - NET MyRSI values matching input length
//!
//! ## Developer Status
//! **AVX2**: Not implemented (no SIMD functions found)
//! **AVX512**: Not implemented (no SIMD functions found)
//! **Streaming**: O(n) - Requires full window for NET calculation
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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
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
impl<'a> AsRef<[f64]> for NetMyrsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NetMyrsiData::Slice(slice) => slice,
            NetMyrsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum NetMyrsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct NetMyrsiOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct NetMyrsiParams {
    pub period: Option<usize>,
}

impl Default for NetMyrsiParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct NetMyrsiInput<'a> {
    pub data: NetMyrsiData<'a>,
    pub params: NetMyrsiParams,
}

impl<'a> NetMyrsiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: NetMyrsiParams) -> Self {
        Self {
            data: NetMyrsiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: NetMyrsiParams) -> Self {
        Self {
            data: NetMyrsiData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", NetMyrsiParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct NetMyrsiBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for NetMyrsiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NetMyrsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, val: usize) -> Self {
        self.period = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<NetMyrsiOutput, NetMyrsiError> {
        let p = NetMyrsiParams {
            period: self.period,
        };
        let i = NetMyrsiInput::from_candles(c, "close", p);
        net_myrsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<NetMyrsiOutput, NetMyrsiError> {
        let p = NetMyrsiParams {
            period: self.period,
        };
        let i = NetMyrsiInput::from_slice(d, p);
        net_myrsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<NetMyrsiStream, NetMyrsiError> {
        let p = NetMyrsiParams {
            period: self.period,
        };
        NetMyrsiStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum NetMyrsiError {
    #[error("net_myrsi: Input data slice is empty.")]
    EmptyInputData,

    #[error("net_myrsi: All values are NaN.")]
    AllValuesNaN,

    #[error("net_myrsi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("net_myrsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// ==================== CORE COMPUTATION ====================
/// Helper function to compute NET MyRSI into a slice
#[inline(always)]
fn net_myrsi_compute_into(data: &[f64], period: usize, first: usize, out: &mut [f64], _k: Kernel) {
    // preserve out[..first+period-1] NaNs
    let mut tmp = alloc_with_nan_prefix(data.len(), first + period - 1);
    compute_myrsi_from(data, period, first, &mut tmp);
    compute_net_from(&tmp, period, first, out);
}

/// MyRSI calculation based on John F. Ehlers' formula
#[inline(always)]
fn compute_myrsi_from(data: &[f64], period: usize, first: usize, out_myrsi: &mut [f64]) {
    let start = first + period; // need period+1 actual values
    let len = data.len();

    for i in start..len {
        let mut cu = 0.0;
        let mut cd = 0.0;
        for j in 0..period {
            let newer = data[i - j];
            let older = data[i - j - 1];
            let diff = newer - older;
            if diff > 0.0 {
                cu += diff;
            } else if diff < 0.0 {
                cd += -diff;
            }
        }
        let sum = cu + cd;
        out_myrsi[i] = if sum != 0.0 { (cu - cd) / sum } else { 0.0 };
    }
}

/// NET (Noise Elimination Technique) calculation
#[inline(always)]
fn compute_net_from(myrsi: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let start = first + period - 1;
    let len = myrsi.len();
    let denom = (period * (period - 1)) as f64 / 2.0;

    for idx in start..len {
        let mut num = 0.0;
        for i in 1..period {
            for k in 0..i {
                let vi = myrsi[idx - i];
                let vk = myrsi[idx - k];
                let d = vi - vk;
                if d > 0.0 {
                    num -= 1.0;
                } else if d < 0.0 {
                    num += 1.0;
                }
            }
        }
        out[idx] = num / denom;
    }
}

/// Preparation function to validate inputs and parameters
#[inline(always)]
fn net_myrsi_prepare<'a>(
    input: &'a NetMyrsiInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), NetMyrsiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(NetMyrsiError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NetMyrsiError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(NetMyrsiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // Need period+1 values for MyRSI (to compare consecutive pairs)
    if len - first < period + 1 {
        return Err(NetMyrsiError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let chosen = if matches!(kernel, Kernel::Auto) {
        detect_best_kernel()
    } else {
        kernel
    };

    Ok((data, period, first, chosen))
}

// ==================== MAIN ENTRY POINTS ====================
/// Main entry point with automatic kernel selection
#[inline]
pub fn net_myrsi(input: &NetMyrsiInput) -> Result<NetMyrsiOutput, NetMyrsiError> {
    net_myrsi_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection
pub fn net_myrsi_with_kernel(
    input: &NetMyrsiInput,
    kernel: Kernel,
) -> Result<NetMyrsiOutput, NetMyrsiError> {
    let (data, period, first, chosen) = net_myrsi_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);
    net_myrsi_compute_into(data, period, first, &mut out, chosen);
    Ok(NetMyrsiOutput { values: out })
}

/// Zero-allocation version for WASM and performance-critical paths
#[inline]
pub fn net_myrsi_into_slice(
    dst: &mut [f64],
    input: &NetMyrsiInput,
    kern: Kernel,
) -> Result<(), NetMyrsiError> {
    let (data, period, first, chosen) = net_myrsi_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(NetMyrsiError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    net_myrsi_compute_into(data, period, first, dst, chosen);

    // enforce warmup NaNs like alma_into_slice
    let warm = first + period - 1;
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }

    Ok(())
}

// ==================== STREAMING SUPPORT ====================
#[derive(Debug, Clone)]
pub struct NetMyrsiStream {
    period: usize,
    price: Vec<f64>, // len = period + 1
    myrsi: Vec<f64>, // len = period
    head: usize,
    filled_prices: bool,
    filled_myrsi: bool,
}

impl NetMyrsiStream {
    pub fn try_new(params: NetMyrsiParams) -> Result<Self, NetMyrsiError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(NetMyrsiError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        Ok(Self {
            period,
            price: vec![f64::NAN; period + 1],
            myrsi: vec![f64::NAN; period],
            head: 0,
            filled_prices: false,
            filled_myrsi: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // price ring
        self.price[self.head % (self.period + 1)] = value;
        if !self.filled_prices && self.head + 1 >= self.period + 1 {
            self.filled_prices = true;
        }

        // compute MyRSI when we have period+1 prices
        if self.filled_prices {
            let mut cu = 0.0;
            let mut cd = 0.0;
            for j in 0..self.period {
                let newer = self.price[(self.head - j) % (self.period + 1)];
                let older = self.price[(self.head - j - 1) % (self.period + 1)];
                let diff = newer - older;
                if diff > 0.0 {
                    cu += diff;
                } else if diff < 0.0 {
                    cd += -diff;
                }
            }
            let s = cu + cd;
            let r = if s != 0.0 { (cu - cd) / s } else { 0.0 };
            self.myrsi[self.head % self.period] = r;

            if !self.filled_myrsi && self.head + 1 >= self.period * 2 {
                self.filled_myrsi = true;
            }
        }

        // compute NET once we have period MyRSI values
        let out = if self.filled_myrsi {
            let denom = (self.period * (self.period - 1)) as f64 / 2.0;
            let mut num = 0.0;
            for i in 1..self.period {
                for k in 0..i {
                    let vi = self.myrsi[(self.head - i) % self.period];
                    let vk = self.myrsi[(self.head - k) % self.period];
                    let d = vi - vk;
                    if d > 0.0 {
                        num -= 1.0;
                    } else if d < 0.0 {
                        num += 1.0;
                    }
                }
            }
            Some(num / denom)
        } else {
            None
        };

        self.head = self.head.wrapping_add(1);
        out
    }
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug)]
pub struct NetMyrsiBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for NetMyrsiBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 50, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NetMyrsiBatchBuilder {
    range: NetMyrsiBatchRange,
    kernel: Kernel,
}

impl NetMyrsiBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<NetMyrsiBatchOutput, NetMyrsiError> {
        net_myrsi_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<NetMyrsiBatchOutput, NetMyrsiError> {
        self.apply_slice(source_type(c, src))
    }

    pub fn with_default_candles(c: &Candles) -> Result<NetMyrsiBatchOutput, NetMyrsiError> {
        NetMyrsiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct NetMyrsiBatchOutput {
    pub values: Vec<f64>, // row-major (rows * cols)
    pub combos: Vec<NetMyrsiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NetMyrsiBatchOutput {
    pub fn row_for_params(&self, p: &NetMyrsiParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }

    pub fn values_for(&self, p: &NetMyrsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|r| {
            let s = r * self.cols;
            &self.values[s..s + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_period((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

pub fn net_myrsi_batch_with_kernel(
    data: &[f64],
    sweep: &NetMyrsiBatchRange,
    k: Kernel,
) -> Result<NetMyrsiBatchOutput, NetMyrsiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(NetMyrsiError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };

    let periods = expand_grid_period(sweep.period);
    let combos: Vec<NetMyrsiParams> = periods
        .into_iter()
        .map(|p| NetMyrsiParams { period: Some(p) })
        .collect();
    net_myrsi_batch_inner(data, &combos, kernel)
}

#[inline(always)]
fn net_myrsi_batch_inner(
    data: &[f64],
    combos: &[NetMyrsiParams],
    kern: Kernel,
) -> Result<NetMyrsiBatchOutput, NetMyrsiError> {
    if data.is_empty() {
        return Err(NetMyrsiError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NetMyrsiError::AllValuesNaN)?;
    let cols = data.len();
    let rows = combos.len();

    // validate all periods upfront
    let max_needed = combos
        .iter()
        .map(|c| c.period.unwrap_or(14) + 1)
        .max()
        .unwrap();
    if cols - first < max_needed {
        return Err(NetMyrsiError::NotEnoughValidData {
            needed: max_needed,
            valid: cols - first,
        });
    }

    // allocate row-major matrix with poison-safe helpers
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // row-specific warmup (first + period - 1)
    let warms: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap_or(14) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    // produce mutable &[f64] view
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // per-row compute (scalar; kern kept for parity)
    for (row, slice) in out.chunks_mut(cols).enumerate() {
        let p = combos[row].period.unwrap_or(14);
        // compute into slice directly through helper
        net_myrsi_compute_into(data, p, first, slice, kern);
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(NetMyrsiBatchOutput {
        values,
        combos: combos.to_vec(),
        rows,
        cols,
    })
}

#[inline(always)]
fn net_myrsi_batch_inner_into(
    data: &[f64],
    combos: &[NetMyrsiParams],
    kern: Kernel,
    out: &mut [f64],
) -> Result<(), NetMyrsiError> {
    if data.is_empty() {
        return Err(NetMyrsiError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NetMyrsiError::AllValuesNaN)?;
    let cols = data.len();
    let rows = combos.len();

    // upfront validation
    let max_needed = combos
        .iter()
        .map(|c| c.period.unwrap_or(14) + 1)
        .max()
        .unwrap();
    if cols - first < max_needed {
        return Err(NetMyrsiError::NotEnoughValidData {
            needed: max_needed,
            valid: cols - first,
        });
    }

    // single reusable scratch row
    let mut tmp_mu = make_uninit_matrix(1, cols);

    for (row, dst) in out.chunks_mut(cols).enumerate() {
        let p = combos[row].period.unwrap_or(14);
        let warm = first + p - 1;
        // Initialize dst prefix with NaNs
        for i in 0..warm {
            dst[i] = f64::NAN;
        }
        init_matrix_prefixes(&mut tmp_mu, cols, &[warm]);
        let tmp: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(tmp_mu.as_mut_ptr() as *mut f64, cols) };
        compute_myrsi_from(data, p, first, tmp);
        compute_net_from(tmp, p, first, dst);
    }

    let _ = tmp_mu; // drop
    Ok(())
}

// Legacy batch function for compatibility
#[cfg(not(target_arch = "wasm32"))]
pub fn net_myrsi_batch(
    data: &[Vec<f64>],
    params: &NetMyrsiParams,
    kernel: Option<Kernel>,
) -> Result<Vec<Vec<f64>>, NetMyrsiError> {
    let kern = kernel.unwrap_or_else(detect_best_batch_kernel);

    let results: Result<Vec<_>, _> = data
        .par_iter()
        .map(|series| {
            let input = NetMyrsiInput::from_slice(series, params.clone());
            net_myrsi_with_kernel(&input, kern).map(|o| o.values)
        })
        .collect();

    results
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "net_myrsi")]
pub fn net_myrsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = NetMyrsiParams {
        period: Some(period),
    };
    let input = NetMyrsiInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| net_myrsi_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "net_myrsi_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn net_myrsi_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;

    let periods = expand_grid_period(period_range);
    let combos: Vec<NetMyrsiParams> = periods
        .into_iter()
        .map(|p| NetMyrsiParams { period: Some(p) })
        .collect();
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let actual = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        // no SIMD specialization needed, keep signature parity
        net_myrsi_batch_inner_into(slice_in, &combos, actual, out_slice)
    })
    .map_err(|e: NetMyrsiError| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "NetMyrsiStream")]
pub struct NetMyrsiStreamPy {
    stream: NetMyrsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NetMyrsiStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = NetMyrsiParams {
            period: Some(period),
        };
        let stream =
            NetMyrsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn net_myrsi_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let input = NetMyrsiInput::from_slice(
        data,
        NetMyrsiParams {
            period: Some(period),
        },
    );
    let mut out = vec![f64::NAN; data.len()];
    net_myrsi_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn net_myrsi_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn net_myrsi_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn net_myrsi_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = NetMyrsiInput::from_slice(
            data,
            NetMyrsiParams {
                period: Some(period),
            },
        );

        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            net_myrsi_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            net_myrsi_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NetMyrsiBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NetMyrsiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NetMyrsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = net_myrsi_batch)]
pub fn net_myrsi_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: NetMyrsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = NetMyrsiBatchRange {
        period: cfg.period_range,
    };
    let out = net_myrsi_batch_with_kernel(data, &sweep, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&NetMyrsiBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn net_myrsi_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to net_myrsi_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let periods = expand_grid_period((period_start, period_end, period_step));
        let combos: Vec<NetMyrsiParams> = periods
            .into_iter()
            .map(|p| NetMyrsiParams { period: Some(p) })
            .collect();
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        net_myrsi_batch_inner_into(data, &combos, detect_best_kernel(), out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_net_myrsi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = NetMyrsiParams { period: None };
        let input = NetMyrsiInput::from_candles(&candles, "close", default_params);
        let output = net_myrsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_net_myrsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = NetMyrsiInput::from_candles(&candles, "close", NetMyrsiParams::default());
        let result = net_myrsi_with_kernel(&input, kernel)?;

        // Reference values from PineScript
        let expected_last_five = [0.64835165, 0.49450549, 0.29670330, 0.07692308, -0.07692308];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] NET_MYRSI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_net_myrsi_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = NetMyrsiInput::with_default_candles(&candles);
        match input.data {
            NetMyrsiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected NetMyrsiData::Candles"),
        }
        let output = net_myrsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_net_myrsi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = NetMyrsiParams { period: Some(0) };
        let input = NetMyrsiInput::from_slice(&input_data, params);
        let res = net_myrsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NET_MYRSI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_net_myrsi_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = NetMyrsiParams { period: Some(10) };
        let input = NetMyrsiInput::from_slice(&data_small, params);
        let res = net_myrsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NET_MYRSI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_net_myrsi_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = vec![10.0, 20.0, 30.0, 15.0, 25.0];
        let params = NetMyrsiParams { period: Some(3) };
        let input = NetMyrsiInput::from_slice(&data_small, params);
        let result = net_myrsi_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data_small.len());
        Ok(())
    }

    fn check_net_myrsi_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: Vec<f64> = vec![];
        let params = NetMyrsiParams::default();
        let input = NetMyrsiInput::from_slice(&data, params);
        let res = net_myrsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NET_MYRSI should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_net_myrsi_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 30];
        let params = NetMyrsiParams::default();
        let input = NetMyrsiInput::from_slice(&data, params);
        let res = net_myrsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NET_MYRSI should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_net_myrsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = NetMyrsiParams { period: Some(14) };
        let first_input = NetMyrsiInput::from_candles(&candles, "close", first_params);
        let first_result = net_myrsi_with_kernel(&first_input, kernel)?;

        let second_params = NetMyrsiParams { period: Some(14) };
        let second_input = NetMyrsiInput::from_slice(&first_result.values, second_params);
        let second_result = net_myrsi_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        // Check that second pass produces valid output
        let non_nan_count = second_result.values.iter().filter(|v| !v.is_nan()).count();
        assert!(
            non_nan_count > 0,
            "[{}] Second pass should produce non-NaN values",
            test_name
        );

        Ok(())
    }

    fn check_net_myrsi_warmup_nans(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first = candles.close.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let p = 14;
        let out = net_myrsi_with_kernel(
            &NetMyrsiInput::from_candles(&candles, "close", NetMyrsiParams { period: Some(p) }),
            kernel,
        )?
        .values;
        let warm = first + p - 1;
        assert!(
            out[..warm].iter().all(|v| v.is_nan()),
            "[{}] Warmup NaNs should not be overwritten",
            test_name
        );
        Ok(())
    }

    fn check_net_myrsi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        for _ in 0..20 {
            data.push(data[data.len() - 1] + 1.0);
        }

        // Insert NaN in the middle
        data[15] = f64::NAN;

        let params = NetMyrsiParams { period: Some(14) };
        let input = NetMyrsiInput::from_slice(&data, params);
        let result = net_myrsi_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), data.len());

        // For NET MyRSI, NaN in middle doesn't necessarily propagate to same index
        // due to the lookback nature of the calculation
        // Just verify that we handle NaN gracefully and produce output
        let non_nan_count = result.values.iter().filter(|v| !v.is_nan()).count();
        assert!(
            non_nan_count > 0,
            "[{}] Should produce some non-NaN values despite NaN input",
            test_name
        );

        Ok(())
    }

    fn check_net_myrsi_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let period = 14;

        let batch = NetMyrsiInput::from_candles(
            &c,
            "close",
            NetMyrsiParams {
                period: Some(period),
            },
        );
        let b = net_myrsi_with_kernel(&batch, kernel)?.values;

        let mut s = NetMyrsiStream::try_new(NetMyrsiParams {
            period: Some(period),
        })?;
        let mut stream = Vec::with_capacity(c.close.len());
        for &px in &c.close {
            stream.push(s.update(px).unwrap_or(f64::NAN));
        }

        let first_valid = b.iter().position(|v| !v.is_nan()).unwrap();
        for i in first_valid..b.len() {
            if !b[i].is_nan() && !stream[i].is_nan() {
                assert!(
                    (b[i] - stream[i]).abs() < 1e-10,
                    "[{test}] idx {i}: {} vs {}",
                    b[i],
                    stream[i]
                );
            }
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_net_myrsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            NetMyrsiParams::default(),
            NetMyrsiParams { period: Some(10) },
            NetMyrsiParams { period: Some(20) },
        ];

        for params in test_params {
            let input = NetMyrsiInput::from_candles(&candles, "close", params.clone());
            let output = net_myrsi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }
                let bits = val.to_bits();
                assert_ne!(
                    bits, 0x11111111_11111111,
                    "[{}] alloc_with_nan_prefix poison at {} with params {:?}",
                    test_name, i, params
                );
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_net_myrsi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_net_myrsi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period + 2..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = NetMyrsiParams {
                    period: Some(period),
                };
                let input = NetMyrsiInput::from_slice(&data, params);

                let NetMyrsiOutput { values: out } = net_myrsi_with_kernel(&input, kernel).unwrap();
                let NetMyrsiOutput { values: ref_out } =
                    net_myrsi_with_kernel(&input, Kernel::Scalar).unwrap();

                assert_eq!(out.len(), ref_out.len());
                assert_eq!(out.len(), data.len());

                // Find first valid index
                let first_valid = ref_out
                    .iter()
                    .position(|v| !v.is_nan())
                    .unwrap_or(ref_out.len());

                // All values before first valid should be NaN
                for i in 0..first_valid {
                    assert!(out[i].is_nan(), "Expected NaN at index {}", i);
                }

                // Compare valid values
                for i in first_valid..out.len() {
                    if ref_out[i].is_nan() {
                        assert!(out[i].is_nan(), "Expected NaN at index {}", i);
                    } else {
                        let diff = (out[i] - ref_out[i]).abs();
                        let rel_diff = diff / ref_out[i].abs().max(1e-10);
                        assert!(
                            rel_diff < 1e-10,
                            "Mismatch at {}: kernel={}, scalar={}, diff={}",
                            i,
                            out[i],
                            ref_out[i],
                            diff
                        );
                    }
                }

                Ok(())
            })
            .map_err(|e| format!("Property test failed: {}", e).into())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_net_myrsi_property(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_net_myrsi_tests {
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

    generate_all_net_myrsi_tests!(
        check_net_myrsi_partial_params,
        check_net_myrsi_accuracy,
        check_net_myrsi_default_candles,
        check_net_myrsi_zero_period,
        check_net_myrsi_period_exceeds_length,
        check_net_myrsi_very_small_dataset,
        check_net_myrsi_empty_input,
        check_net_myrsi_all_nan,
        check_net_myrsi_reinput,
        check_net_myrsi_warmup_nans,
        check_net_myrsi_nan_handling,
        check_net_myrsi_streaming,
        check_net_myrsi_no_poison,
        check_net_myrsi_property
    );

    // Batch processing tests
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = NetMyrsiBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 20, 1)
            .apply_candles(&c, "close")?;
        let def = NetMyrsiParams::default();
        let row = out.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = NetMyrsiBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 20, 5)
            .apply_candles(&c, "close")?;
        for (idx, &v) in out.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let bits = v.to_bits();
            assert_ne!(
                bits, 0x1111_1111_1111_1111,
                "[{test}] alloc_with_nan_prefix poison at {idx}"
            );
            assert_ne!(
                bits, 0x2222_2222_2222_2222,
                "[{test}] init_matrix_prefixes poison at {idx}"
            );
            assert_ne!(
                bits, 0x3333_3333_3333_3333,
                "[{test}] make_uninit_matrix poison at {idx}"
            );
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = NetMyrsiBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 20, 5)
            .apply_candles(&c, "close")?;
        assert_eq!(out.rows, 3); // 10, 15, 20
        assert_eq!(out.cols, c.close.len());
        assert_eq!(out.values.len(), out.rows * out.cols);

        // Verify parameter combinations
        assert_eq!(out.combos.len(), 3);
        assert_eq!(out.combos[0].period, Some(10));
        assert_eq!(out.combos[1].period, Some(15));
        assert_eq!(out.combos[2].period, Some(20));
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                    Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);
}
