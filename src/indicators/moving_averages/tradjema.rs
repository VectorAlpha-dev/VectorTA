//! # True Range Adjusted Exponential Moving Average (TRADJEMA)
//!
//! The True Range Adjusted EMA adjusts the exponential moving average smoothing factor
//! based on the normalized true range, providing a dynamic response to volatility.
//!
//! ## Parameters
//! - **length**: The lookback period for calculations (default: 40)
//! - **mult**: Multiplier for the adjustment factor (default: 10.0)
//!
//! ## Returns
//! - **`Ok(TradjemaOutput)`** on success, containing a Vec<f64> of length matching the input.
//! - **`Err(TradjemaError)`** otherwise.
//!
//! ## Developer Status
//! - **AVX2 kernel**: STUB - Falls back to scalar implementation
//! - **AVX512 kernel**: STUB - Falls back to scalar implementation
//! - **Streaming update**: O(n) - Recalculates min/max over TR buffer each update
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix) ✓
//! - **Optimization needed**: Implement SIMD kernels for vectorized processing
//! - **Streaming improvement**: Could optimize min/max tracking to reduce O(n) updates

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
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
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

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both candle data and raw slices
#[derive(Debug, Clone)]
pub enum TradjemaData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct TradjemaOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TradjemaParams {
    pub length: Option<usize>,
    pub mult: Option<f64>,
}

impl Default for TradjemaParams {
    fn default() -> Self {
        Self {
            length: Some(40),
            mult: Some(10.0),
        }
    }
}

/// Input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct TradjemaInput<'a> {
    pub data: TradjemaData<'a>,
    pub params: TradjemaParams,
}

impl<'a> TradjemaInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: TradjemaParams) -> Self {
        Self {
            data: TradjemaData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: TradjemaParams,
    ) -> Self {
        Self {
            data: TradjemaData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, TradjemaParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(40)
    }

    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(10.0)
    }
}

// ==================== BUILDER PATTERN ====================
#[derive(Copy, Clone, Debug)]
pub struct TradjemaBuilder {
    length: Option<usize>,
    mult: Option<f64>,
    kernel: Kernel,
}

impl Default for TradjemaBuilder {
    fn default() -> Self {
        Self {
            length: None,
            mult: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TradjemaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }

    #[inline(always)]
    pub fn mult(mut self, m: f64) -> Self {
        self.mult = Some(m);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<TradjemaOutput, TradjemaError> {
        let p = TradjemaParams {
            length: self.length,
            mult: self.mult,
        };
        let i = TradjemaInput::from_candles(c, p);
        tradjema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<TradjemaOutput, TradjemaError> {
        let p = TradjemaParams {
            length: self.length,
            mult: self.mult,
        };
        let i = TradjemaInput::from_slices(high, low, close, p);
        tradjema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<TradjemaStream, TradjemaError> {
        let p = TradjemaParams {
            length: self.length,
            mult: self.mult,
        };
        TradjemaStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum TradjemaError {
    #[error("tradjema: Input data slice is empty.")]
    EmptyInputData,

    #[error("tradjema: All values are NaN.")]
    AllValuesNaN,

    #[error("tradjema: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },

    #[error("tradjema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("tradjema: OHLC data length mismatch")]
    MissingData,

    #[error("tradjema: Invalid multiplier: {mult}")]
    InvalidMult { mult: f64 },
}

// ==================== MAIN COMPUTATION ====================
#[inline(always)]
fn tradjema_prepare<'a>(
    input: &'a TradjemaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, f64, Kernel), TradjemaError> {
    let (high, low, close) = match &input.data {
        TradjemaData::Candles { candles } => {
            let h = candles
                .select_candle_field("high")
                .map_err(|_| TradjemaError::EmptyInputData)?;
            let l = candles
                .select_candle_field("low")
                .map_err(|_| TradjemaError::EmptyInputData)?;
            let c = candles
                .select_candle_field("close")
                .map_err(|_| TradjemaError::EmptyInputData)?;
            (h, l, c)
        }
        TradjemaData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(TradjemaError::MissingData);
            }
            (*high, *low, *close)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(TradjemaError::EmptyInputData);
    }

    let first = close
        .iter()
        .position(|v| !v.is_nan())
        .ok_or(TradjemaError::AllValuesNaN)?;
    let length = input.get_length();
    if length < 2 || length > len {
        return Err(TradjemaError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if len - first < length {
        return Err(TradjemaError::NotEnoughValidData {
            needed: length,
            valid: len - first,
        });
    }

    let mult = input.get_mult();
    if mult <= 0.0 || !mult.is_finite() {
        return Err(TradjemaError::InvalidMult { mult });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((high, low, close, length, first, mult, chosen))
}

#[inline]
pub fn tradjema(input: &TradjemaInput) -> Result<TradjemaOutput, TradjemaError> {
    tradjema_with_kernel(input, Kernel::Auto)
}

pub fn tradjema_with_kernel(
    input: &TradjemaInput,
    kernel: Kernel,
) -> Result<TradjemaOutput, TradjemaError> {
    let (h, l, c, length, first, mult, chosen) = tradjema_prepare(input, kernel)?;
    let warm = first + length - 1;
    let mut out = alloc_with_nan_prefix(c.len(), warm);
    tradjema_compute_into(h, l, c, length, mult, first, chosen, &mut out);
    Ok(TradjemaOutput { values: out })
}

#[inline]
pub fn tradjema_into_slice(
    dst: &mut [f64],
    input: &TradjemaInput,
    kern: Kernel,
) -> Result<(), TradjemaError> {
    let (h, l, c, length, first, mult, chosen) = tradjema_prepare(input, kern)?;
    if dst.len() != c.len() {
        return Err(TradjemaError::InvalidLength {
            length: dst.len(),
            data_len: c.len(),
        });
    }
    tradjema_compute_into(h, l, c, length, mult, first, chosen, dst);

    let warm = first + length - 1;
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
fn tradjema_compute_into_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
    first: usize,
    out: &mut [f64],
) {
    let alpha = 2.0 / (length as f64 + 1.0);
    let warm = first + length - 1;

    // Ring buffer of TRs, no memmoves
    let mut tr_buf = vec![0.0; length];
    let mut head = 0usize;

    // Seed window [first .. first+length)
    for k in 0..length {
        let idx = first + k;
        let tr = if idx == first {
            high[idx] - low[idx]
        } else {
            let hl = high[idx] - low[idx];
            let hc = (high[idx] - close[idx - 1]).abs();
            let lc = (low[idx] - close[idx - 1]).abs();
            hl.max(hc).max(lc)
        };
        tr_buf[k] = tr;
    }
    head = length - 1;

    // Compute min/max over initial window
    let mut tr_low = tr_buf[0];
    let mut tr_high = tr_buf[0];
    for &v in &tr_buf[1..] {
        if v < tr_low {
            tr_low = v;
        }
        if v > tr_high {
            tr_high = v;
        }
    }

    // Pine-compatible 1-bar lag
    let src_at = |i: usize| close[i - 1];

    let current_tr0 = tr_buf[head];
    let tr_adj0 = if tr_high != tr_low {
        (current_tr0 - tr_low) / (tr_high - tr_low)
    } else {
        0.0
    };
    let mut y = alpha * (1.0 + tr_adj0 * mult) * (src_at(warm) - 0.0);
    out[warm] = y;

    // Main loop
    for i in (warm + 1)..out.len() {
        // next TR
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        let tr_new = hl.max(hc).max(lc);

        // advance ring; pop old, push new
        head = (head + 1) % length;
        let tr_old = std::mem::replace(&mut tr_buf[head], tr_new);

        // update min/max with bounded work; full rescan only when necessary
        if tr_old <= tr_low || tr_old >= tr_high {
            // outgoing was an extremum: recompute min/max over window
            tr_low = tr_buf[0];
            tr_high = tr_buf[0];
            for &v in &tr_buf[1..] {
                if v < tr_low {
                    tr_low = v;
                }
                if v > tr_high {
                    tr_high = v;
                }
            }
        } else {
            // outgoing not an extremum: only compare new value
            if tr_new < tr_low {
                tr_low = tr_new;
            }
            if tr_new > tr_high {
                tr_high = tr_new;
            }
        }

        let tr_adj = if tr_high != tr_low {
            (tr_new - tr_low) / (tr_high - tr_low)
        } else {
            0.0
        };
        let a = alpha * (1.0 + tr_adj * mult);
        y += a * (src_at(i) - y);
        out[i] = y;
    }
}

#[inline(always)]
fn tradjema_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
    first: usize,
    kern: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kern, Kernel::Scalar | Kernel::ScalarBatch) {
                tradjema_compute_into_scalar(high, low, close, length, mult, first, out);
                return;
            }
        }
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                tradjema_compute_into_scalar(high, low, close, length, mult, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                tradjema_compute_into_scalar(high, low, close, length, mult, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                tradjema_compute_into_scalar(high, low, close, length, mult, first, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                tradjema_compute_into_scalar(high, low, close, length, mult, first, out)
            }
            _ => unreachable!(),
        }
    }
}

// ==================== STREAMING IMPLEMENTATION ====================
#[derive(Debug, Clone)]
pub struct TradjemaStream {
    length: usize,
    mult: f64,
    tr_buffer: Vec<f64>,
    idx: usize,
    filled: bool,
    prev_close: f64,
    tradjema: f64,
    alpha: f64,
}

impl TradjemaStream {
    pub fn try_new(params: TradjemaParams) -> Result<Self, TradjemaError> {
        let length = params.length.unwrap_or(40);
        let mult = params.mult.unwrap_or(10.0);

        if length < 2 {
            return Err(TradjemaError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        if mult <= 0.0 || !mult.is_finite() {
            return Err(TradjemaError::InvalidMult { mult });
        }

        Ok(Self {
            length,
            mult,
            tr_buffer: vec![0.0; length],
            idx: 0,
            filled: false,
            prev_close: f64::NAN,
            tradjema: f64::NAN,
            alpha: 2.0 / (length as f64 + 1.0),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        // Calculate true range with previous close
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        };

        // Update ring buffer
        self.tr_buffer[self.idx] = tr;
        self.idx = (self.idx + 1) % self.length;

        // Not filled yet
        if !self.filled && self.idx != 0 {
            self.prev_close = close; // keep prev_close for lagged src
            return None;
        }

        // First bar with full window
        if !self.filled && self.idx == 0 {
            self.filled = true;

            // Min/max over full window
            let mut tr_low = self.tr_buffer[0];
            let mut tr_high = self.tr_buffer[0];
            for &v in &self.tr_buffer[1..] {
                if v < tr_low {
                    tr_low = v;
                }
                if v > tr_high {
                    tr_high = v;
                }
            }
            let curr_tr = self.tr_buffer[self.length - 1];
            let tr_adj = if tr_high != tr_low {
                (curr_tr - tr_low) / (tr_high - tr_low)
            } else {
                0.0
            };

            let adjusted_alpha = self.alpha * (1.0 + tr_adj * self.mult);
            let src = if self.prev_close.is_nan() {
                close
            } else {
                self.prev_close
            }; // 1-bar lag
            self.tradjema = 0.0 + adjusted_alpha * (src - 0.0); // Pine seed

            self.prev_close = close;
            return Some(self.tradjema);
        }

        // Subsequent bars
        let mut tr_low = self.tr_buffer[0];
        let mut tr_high = self.tr_buffer[0];
        for &v in &self.tr_buffer[1..] {
            if v < tr_low {
                tr_low = v;
            }
            if v > tr_high {
                tr_high = v;
            }
        }
        let tr_adj = if tr_high != tr_low {
            (tr - tr_low) / (tr_high - tr_low)
        } else {
            0.0
        };

        let adjusted_alpha = self.alpha * (1.0 + tr_adj * self.mult);
        let src = self.prev_close; // lagged source
        self.tradjema += adjusted_alpha * (src - self.tradjema);

        self.prev_close = close;
        Some(self.tradjema)
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "tradjema")]
#[pyo3(signature = (high, low, close, length, mult, kernel=None))]
pub fn tradjema_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    mult: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (h, l, c) = (high.as_slice()?, low.as_slice()?, close.as_slice()?);
    if h.len() != l.len() || l.len() != c.len() {
        return Err(PyValueError::new_err(
            "All OHLC arrays must have the same length",
        ));
    }
    let kern = validate_kernel(kernel, false)?;
    let input = TradjemaInput::from_slices(
        h,
        l,
        c,
        TradjemaParams {
            length: Some(length),
            mult: Some(mult),
        },
    );

    let values = py
        .allow_threads(|| tradjema_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "tradjema_batch")]
#[pyo3(signature = (high, low, close, length_range, mult_range, kernel=None))]
pub fn tradjema_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArray1;

    let (h, l, c) = (high.as_slice()?, low.as_slice()?, close.as_slice()?);
    if h.len() != l.len() || l.len() != c.len() {
        return Err(PyValueError::new_err(
            "All OHLC arrays must have the same length",
        ));
    }

    let sweep = TradjemaBatchRange {
        length: length_range,
        mult: mult_range,
    };
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("Empty parameter grid"));
    }
    let rows = combos.len();
    let cols = c.len();

    // Create uninitialized NumPy buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Warm-prefix NaNs per row without copying full rows
    let first = c
        .iter()
        .position(|v| !v.is_nan())
        .ok_or_else(|| PyValueError::new_err("All values are NaN"))?;
    for (row, prm) in combos.iter().enumerate() {
        let length = prm.length.unwrap_or(40);
        let warm = first + length - 1;
        let row_slice = &mut slice_out[row * cols..(row + 1) * cols];
        for v in &mut row_slice[..warm] {
            *v = f64::NAN;
        }
    }

    let kern = validate_kernel(kernel, true)?;
    let simd = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };

    // Fill from warm onward
    let combos = py
        .allow_threads(|| tradjema_batch_inner_into(h, l, c, &sweep, simd, true, slice_out))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(40) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "mults",
        combos
            .iter()
            .map(|p| p.mult.unwrap_or(10.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "TradjemaStream")]
pub struct TradjemaStreamPy {
    inner: TradjemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TradjemaStreamPy {
    #[new]
    fn new(length: usize, mult: f64) -> PyResult<Self> {
        TradjemaStream::try_new(TradjemaParams {
            length: Some(length),
            mult: Some(mult),
        })
        .map(|inner| Self { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.inner.update(high, low, close)
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tradjema_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    mult: f64,
) -> Result<(), JsValue> {
    if [high_ptr, low_ptr, close_ptr, out_ptr]
        .iter()
        .any(|p| p.is_null())
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let h = std::slice::from_raw_parts(high_ptr, len);
        let l = std::slice::from_raw_parts(low_ptr, len);
        let c = std::slice::from_raw_parts(close_ptr, len);

        let params = TradjemaParams {
            length: Some(length),
            mult: Some(mult),
        };
        let input = TradjemaInput::from_slices(h, l, c, params);

        if (out_ptr as *const f64) == close_ptr
            || (out_ptr as *const f64) == high_ptr
            || (out_ptr as *const f64) == low_ptr
        {
            // aliasing: compute into temp then copy
            let mut tmp = vec![f64::NAN; len];
            tradjema_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            tradjema_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tradjema_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
    out_ptr: *mut f64,
) -> Result<usize, JsValue> {
    if [high_ptr, low_ptr, close_ptr].iter().any(|p| p.is_null()) || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer passed"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        // Build sweep once (same semantics as expand_grid)
        let sweep = TradjemaBatchRange {
            length: (length_start, length_end, length_step),
            mult: (mult_start, mult_end, mult_step),
        };
        let combos = expand_grid(&sweep);
        if combos.is_empty() {
            return Err(JsValue::from_str("Empty parameter grid"));
        }
        let rows = combos.len();
        let cols = len;

        // Directly fill caller's buffer
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Pre-initialize only warm prefixes to NaN to match ALMA's layout contract
        // (cannot use make_uninit_matrix/init_matrix_prefixes on a foreign buffer)
        let first = close
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| JsValue::from_str("All values are NaN"))?;
        for (row, prm) in combos.iter().enumerate() {
            let length = prm.length.unwrap_or(40);
            let warm = first + length - 1;
            let row_slice = &mut out[row * cols..(row + 1) * cols];
            for v in &mut row_slice[..warm] {
                *v = f64::NAN;
            }
        }

        // Compute in-place for all rows
        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        tradjema_batch_inner_into(high, low, close, &sweep, simd, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tradjema_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tradjema_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tradjema_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
) -> Result<Vec<f64>, JsValue> {
    // Check for empty input first
    if close.is_empty() {
        return Err(JsValue::from_str("Input data slice is empty"));
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(JsValue::from_str("length mismatch"));
    }
    // Validate + compute warm
    if length < 2 || length > close.len() {
        return Err(JsValue::from_str("Invalid length"));
    }
    if !(mult.is_finite()) || mult <= 0.0 {
        return Err(JsValue::from_str("Invalid mult"));
    }
    let first = close
        .iter()
        .position(|v| !v.is_nan())
        .ok_or_else(|| JsValue::from_str("All values are NaN"))?;
    if close.len() - first < length {
        return Err(JsValue::from_str("Not enough valid data"));
    }
    let warm = first + length - 1;

    // Allocate only the warm prefix as NaN, rest uninitialized then written
    let mut out = alloc_with_nan_prefix(close.len(), warm);

    // Compute in place
    tradjema_compute_into(
        high,
        low,
        close,
        length,
        mult,
        first,
        detect_best_kernel(),
        &mut out,
    );

    Ok(out)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TradjemaBatchConfig {
    pub length_range: (usize, usize, usize),
    pub mult_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TradjemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TradjemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "tradjema_batch")]
pub fn tradjema_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: TradjemaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = TradjemaBatchRange {
        length: cfg.length_range,
        mult: cfg.mult_range,
    };

    // Check if arrays are empty
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(JsValue::from_str("Input arrays are empty"));
    }

    let out = tradjema_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = TradjemaBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug)]
pub struct TradjemaBatchRange {
    pub length: (usize, usize, usize),
    pub mult: (f64, f64, f64),
}

impl Default for TradjemaBatchRange {
    fn default() -> Self {
        Self {
            length: (40, 40, 0),
            mult: (10.0, 10.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TradjemaBatchBuilder {
    range: TradjemaBatchRange,
    kernel: Kernel,
}

impl TradjemaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default_candles(c: &Candles) -> Result<TradjemaBatchOutput, TradjemaError> {
        TradjemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
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
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<TradjemaBatchOutput, TradjemaError> {
        tradjema_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<TradjemaBatchOutput, TradjemaError> {
        let high = c
            .select_candle_field("high")
            .map_err(|_| TradjemaError::EmptyInputData)?;
        let low = c
            .select_candle_field("low")
            .map_err(|_| TradjemaError::EmptyInputData)?;
        let close = c
            .select_candle_field("close")
            .map_err(|_| TradjemaError::EmptyInputData)?;
        self.apply_slices(high, low, close)
    }
}

#[derive(Clone, Debug)]
pub struct TradjemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TradjemaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl TradjemaBatchOutput {
    pub fn row_for_params(&self, p: &TradjemaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(40) == p.length.unwrap_or(40)
                && (c.mult.unwrap_or(10.0) - p.mult.unwrap_or(10.0)).abs() < 1e-9
        })
    }

    pub fn values_for(&self, p: &TradjemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TradjemaBatchRange) -> Vec<TradjemaParams> {
    let (l_start, l_end, l_step) = r.length;
    let (m_start, m_end, m_step) = r.mult;

    // Handle single value case - when both ranges have only one value
    // This handles: step=0 OR start==end for both dimensions
    let length_is_single = l_step == 0 || l_start >= l_end;
    let mult_is_single = m_step == 0.0 || m_start >= m_end;

    if length_is_single && mult_is_single {
        return vec![TradjemaParams {
            length: Some(l_start),
            mult: Some(m_start),
        }];
    }

    let mut combos = Vec::new();

    // Generate all combinations
    let mut length = l_start;
    loop {
        let mut mult = m_start;
        loop {
            combos.push(TradjemaParams {
                length: Some(length),
                mult: Some(mult),
            });

            // Check if we should continue with mult
            if mult_is_single || mult >= m_end {
                break;
            }
            mult += m_step;
            // Only include end value if we haven't gone past it
            if mult > m_end {
                break; // Don't include values past the end
            }
        }

        // Check if we should continue with length
        if length_is_single || length >= l_end {
            break;
        }
        length += l_step;
        // Only include end value if we haven't gone past it
        if length > l_end {
            break; // Don't include values past the end
        }
    }

    combos
}

pub fn tradjema_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &TradjemaBatchRange,
    k: Kernel,
) -> Result<TradjemaBatchOutput, TradjemaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        // Also handle non-batch kernels by converting them
        Kernel::Scalar => Kernel::ScalarBatch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        Kernel::Avx512 => Kernel::Avx512Batch,
        _ => Kernel::ScalarBatch, // Default to scalar batch
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Default fallback
    };

    tradjema_batch_inner(high, low, close, sweep, simd, true)
}

#[inline(always)]
fn tradjema_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &TradjemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TradjemaParams>, TradjemaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TradjemaError::InvalidLength {
            length: 0,
            data_len: 0,
        });
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(TradjemaError::MissingData);
    }

    let cols = close.len();
    let first = close
        .iter()
        .position(|v| !v.is_nan())
        .ok_or(TradjemaError::AllValuesNaN)?;

    // prefix NaNs already handled by caller via init_matrix_prefixes
    let do_row = |row: usize, dst: &mut [f64]| {
        let p = &combos[row];
        let length = p.length.unwrap_or(40);
        let mult = p.mult.unwrap_or(10.0);
        if length < 2 {
            return;
        }
        tradjema_compute_into(high, low, close, length, mult, first, kern, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
        #[cfg(target_arch = "wasm32")]
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

fn tradjema_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &TradjemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TradjemaBatchOutput, TradjemaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TradjemaError::InvalidLength {
            length: 0,
            data_len: 0,
        });
    }
    let rows = combos.len();
    let cols = close.len();

    // allocate uninit matrix and initialize NaN prefixes per row
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let first = close
        .iter()
        .position(|v| !v.is_nan())
        .ok_or(TradjemaError::AllValuesNaN)?;
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + p.length.unwrap_or(40) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    // get &mut [f64] from MaybeUninit backing
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // kern is already a regular (non-batch) kernel passed from tradjema_batch_with_kernel
    let combos = tradjema_batch_inner_into(high, low, close, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(TradjemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_tradjema_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let close = candles.select_candle_field("close")?;

        let default_params = TradjemaParams {
            length: None,
            mult: None,
        };
        let input = TradjemaInput::from_slices(high, low, close, default_params);
        let output = tradjema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_tradjema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let close = candles.select_candle_field("close")?;

        let input = TradjemaInput::from_slices(high, low, close, TradjemaParams::default());
        let result = tradjema_with_kernel(&input, kernel)?;

        // Verify output length matches input
        assert_eq!(result.values.len(), candles.close.len());

        // Check that valid values start after warmup period
        let warmup = 39; // length - 1
        for i in 0..warmup {
            assert!(
                result.values[i].is_nan(),
                "[{}] Expected NaN during warmup at index {}",
                test_name,
                i
            );
        }

        // Check that we have valid values after warmup
        for i in warmup..result.values.len() {
            assert!(
                !result.values[i].is_nan(),
                "[{}] Expected valid value after warmup at index {}",
                test_name,
                i
            );
        }

        // Verify the last 5 values match expected reference values
        let expected_last_five = [
            59395.39322263,
            59388.09683228,
            59373.08371503,
            59350.75110897,
            59323.14225348,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] TRADJEMA accuracy mismatch at last_5[{}]: got {:.8}, expected {:.8}, diff={:.10}",
                test_name,
                i,
                val,
                expected_last_five[i],
                diff
            );
        }

        Ok(())
    }

    fn check_tradjema_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TradjemaInput::with_default_candles(&candles);
        let output = tradjema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_tradjema_zero_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = vec![10.0, 20.0, 30.0];

        // Test length = 0
        let params = TradjemaParams {
            length: Some(0),
            mult: None,
        };
        let input = TradjemaInput::from_slices(&input_data, &input_data, &input_data, params);
        let res = tradjema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRADJEMA should fail with zero length",
            test_name
        );

        // Test length = 1 (minimum is 2)
        let params = TradjemaParams {
            length: Some(1),
            mult: None,
        };
        let input = TradjemaInput::from_slices(&input_data, &input_data, &input_data, params);
        let res = tradjema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRADJEMA should fail with length=1 (minimum is 2)",
            test_name
        );

        Ok(())
    }

    fn check_tradjema_length_exceeds_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = vec![10.0, 20.0, 30.0];
        let params = TradjemaParams {
            length: Some(10),
            mult: None,
        };
        let input = TradjemaInput::from_slices(&data_small, &data_small, &data_small, params);
        let res = tradjema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRADJEMA should fail with length exceeding data",
            test_name
        );
        Ok(())
    }

    fn check_tradjema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = vec![42.0];
        let params = TradjemaParams {
            length: Some(40),
            mult: None,
        };
        let input = TradjemaInput::from_slices(&single_point, &single_point, &single_point, params);
        let res = tradjema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRADJEMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_tradjema_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: Vec<f64> = vec![];
        let input = TradjemaInput::from_slices(&empty, &empty, &empty, TradjemaParams::default());
        let res = tradjema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(TradjemaError::EmptyInputData)),
            "[{}] TRADJEMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_tradjema_invalid_mult(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test negative mult
        let params = TradjemaParams {
            length: Some(2),
            mult: Some(-10.0),
        };
        let input = TradjemaInput::from_slices(&data, &data, &data, params);
        let res = tradjema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(TradjemaError::InvalidMult { .. })),
            "[{}] TRADJEMA should fail with negative mult",
            test_name
        );

        // Test NaN mult
        let params = TradjemaParams {
            length: Some(2),
            mult: Some(f64::NAN),
        };
        let input = TradjemaInput::from_slices(&data, &data, &data, params);
        let res = tradjema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(TradjemaError::InvalidMult { .. })),
            "[{}] TRADJEMA should fail with NaN mult",
            test_name
        );

        Ok(())
    }

    fn check_tradjema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let close = candles.select_candle_field("close")?;

        // First pass
        let first_params = TradjemaParams {
            length: Some(20),
            mult: Some(5.0),
        };
        let first_input = TradjemaInput::from_slices(high, low, close, first_params);
        let first_result = tradjema_with_kernel(&first_input, kernel)?;

        // Use output as input for second pass (using close for all OHLC)
        let second_params = TradjemaParams {
            length: Some(20),
            mult: Some(5.0),
        };
        let second_input = TradjemaInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = tradjema_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_tradjema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let close = candles.select_candle_field("close")?;

        let input = TradjemaInput::from_slices(
            high,
            low,
            close,
            TradjemaParams {
                length: Some(40),
                mult: Some(10.0),
            },
        );
        let res = tradjema_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());

        // Check NaN handling after warmup
        if res.values.len() > 50 {
            for (i, &val) in res.values[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_tradjema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let close = candles.select_candle_field("close")?;

        let length = 40;
        let mult = 10.0;

        let input = TradjemaInput::from_slices(
            high,
            low,
            close,
            TradjemaParams {
                length: Some(length),
                mult: Some(mult),
            },
        );
        let batch_output = tradjema_with_kernel(&input, kernel)?.values;

        let mut stream = TradjemaStream::try_new(TradjemaParams {
            length: Some(length),
            mult: Some(mult),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            match stream.update(high[i], low[i], close[i]) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] TRADJEMA streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_tradjema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let close = candles.select_candle_field("close")?;

        let test_params = vec![
            TradjemaParams::default(),
            TradjemaParams {
                length: Some(10),
                mult: Some(5.0),
            },
            TradjemaParams {
                length: Some(20),
                mult: Some(7.5),
            },
            TradjemaParams {
                length: Some(50),
                mult: Some(15.0),
            },
            TradjemaParams {
                length: Some(100),
                mult: Some(20.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = TradjemaInput::from_slices(high, low, close, params.clone());
            let output = tradjema_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: length={}, mult={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(40),
                        params.mult.unwrap_or(10.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: length={}, mult={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(40),
                        params.mult.unwrap_or(10.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: length={}, mult={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(40),
                        params.mult.unwrap_or(10.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_tradjema_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_tradjema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=100).prop_flat_map(|length| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    length..400,
                ),
                Just(length),
                0.1f64..50.0f64,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, length, mult)| {
                let params = TradjemaParams {
                    length: Some(length),
                    mult: Some(mult),
                };

                // Use same data for high/low/close for simplicity
                let input = TradjemaInput::from_slices(&data, &data, &data, params);

                let TradjemaOutput { values: out } = tradjema_with_kernel(&input, kernel).unwrap();
                let TradjemaOutput { values: ref_out } =
                    tradjema_with_kernel(&input, Kernel::Scalar).unwrap();

                for i in (length - 1)..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_tradjema_tests {
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

    generate_all_tradjema_tests!(
        check_tradjema_partial_params,
        check_tradjema_accuracy,
        check_tradjema_default_candles,
        check_tradjema_zero_length,
        check_tradjema_length_exceeds_data,
        check_tradjema_very_small_dataset,
        check_tradjema_empty_input,
        check_tradjema_invalid_mult,
        check_tradjema_reinput,
        check_tradjema_nan_handling,
        check_tradjema_streaming,
        check_tradjema_no_poison
    );

    fn check_tradjema_into_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let f = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(f)?;
        let (h, l, cl) = (
            c.select_candle_field("high")?,
            c.select_candle_field("low")?,
            c.select_candle_field("close")?,
        );
        let input = TradjemaInput::from_slices(h, l, cl, TradjemaParams::default());
        let mut dst = vec![0.0; cl.len()];
        tradjema_into_slice(&mut dst, &input, kernel)?;
        let first = cl.iter().position(|v| !v.is_nan()).unwrap();
        let warm = first + input.get_length() - 1;
        assert!(
            dst[..warm].iter().all(|v| v.is_nan()),
            "[{}] warmup prefix must be NaN",
            test_name
        );
        Ok(())
    }

    generate_all_tradjema_tests!(check_tradjema_into_slice);

    #[cfg(feature = "proptest")]
    generate_all_tradjema_tests!(check_tradjema_property);

    // Batch processing tests
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TradjemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = TradjemaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TradjemaBatchBuilder::new()
            .kernel(kernel)
            .length_range(20, 50, 10)
            .mult_range(5.0, 15.0, 5.0)
            .apply_candles(&c)?;

        let expected_combos = 4 * 3; // 4 lengths * 3 mults
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (10, 30, 10, 5.0, 15.0, 5.0),
            (40, 40, 0, 10.0, 10.0, 0.0),
            (20, 60, 20, 7.5, 12.5, 2.5),
        ];

        for (cfg_idx, &(l_start, l_end, l_step, m_start, m_end, m_step)) in
            test_configs.iter().enumerate()
        {
            let output = TradjemaBatchBuilder::new()
                .kernel(kernel)
                .length_range(l_start, l_end, l_step)
                .mult_range(m_start, m_end, m_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
                        "[{}] Config {}: Found poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: length={}, mult={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(40),
                        combo.mult.unwrap_or(10.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                    Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_expand_grid_single_value() {
        // Test case that's failing in WASM
        let range = TradjemaBatchRange {
            length: (2, 2, 0),
            mult: (10.0, 10.0, 0.0),
        };
        let combos = expand_grid(&range);
        assert!(
            !combos.is_empty(),
            "expand_grid should not return empty for single value"
        );
        assert_eq!(combos.len(), 1, "Should have exactly one combo");
        assert_eq!(combos[0].length, Some(2));
        assert_eq!(combos[0].mult, Some(10.0));

        // Another test case from the failing tests
        let range2 = TradjemaBatchRange {
            length: (40, 40, 0),
            mult: (10.0, 10.0, 0.0),
        };
        let combos2 = expand_grid(&range2);
        assert!(
            !combos2.is_empty(),
            "expand_grid should not return empty for single value (40,40,0)"
        );
        assert_eq!(
            combos2.len(),
            1,
            "Should have exactly one combo for (40,40,0)"
        );
        assert_eq!(combos2[0].length, Some(40));
        assert_eq!(combos2[0].mult, Some(10.0));
    }
}
