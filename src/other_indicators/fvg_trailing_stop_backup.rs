//! # FVG Trailing Stop
//!
//! FVG Trailing Stop is a trend-following indicator that identifies Fair Value Gaps (FVGs) in price action
//! and uses them to create dynamic trailing stop levels. It combines FVG detection with smoothed channel
//! extremes to generate adaptive support and resistance levels.
//!
//! ## Parameters
//! - **unmitigated_fvg_lookback**: Number of FVGs to track in lookback window (default: 5)
//! - **smoothing_length**: Period for SMA smoothing of levels (default: 9)
//! - **reset_on_cross**: Whether to reset trailing stop on cross (default: false)
//!
//! ## Errors
//! - **EmptyInputData**: fvg_trailing_stop: Input data slice is empty.
//! - **AllValuesNaN**: fvg_trailing_stop: All input values are `NaN`.
//! - **InvalidPeriod**: fvg_trailing_stop: Period is zero or exceeds data length.
//! - **NotEnoughValidData**: fvg_trailing_stop: Not enough valid data points for calculation.
//!
//! ## Returns
//! - **`Ok(FvgTrailingStopOutput)`** on success, containing:
//!   - `upper`: Upper channel boundary (NaN when lower is active)
//!   - `lower`: Lower channel boundary (NaN when upper is active)
//!   - `upper_ts`: Upper trailing stop (NaN when lower is active)
//!   - `lower_ts`: Lower trailing stop (NaN when upper is active)
//! - **`Err(FvgTrailingStopError)`** otherwise.

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

// Standard library imports
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// ==================== DATA STRUCTURES ====================
/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct FvgTrailingStopOutput {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper_ts: Vec<f64>,
    pub lower_ts: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct FvgTrailingStopParams {
    pub unmitigated_fvg_lookback: Option<usize>,
    pub smoothing_length: Option<usize>,
    pub reset_on_cross: Option<bool>,
}

impl Default for FvgTrailingStopParams {
    fn default() -> Self {
        Self {
            unmitigated_fvg_lookback: Some(5),
            smoothing_length: Some(9),
            reset_on_cross: Some(false),
        }
    }
}

/// Data source for FVG Trailing Stop - either Candles or direct slices
#[derive(Debug, Clone)]
pub enum FvgTrailingStopData<'a> {
    Candles(&'a Candles),
    Slices { high: &'a [f64], low: &'a [f64], close: &'a [f64] },
}

/// Helper function to find first valid OHLC data point
#[inline]
fn first_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    for i in 0..high.len() {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            return i;
        }
    }
    usize::MAX
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct FvgTrailingStopInput<'a> {
    pub data: FvgTrailingStopData<'a>,
    pub params: FvgTrailingStopParams,
}

impl<'a> FvgTrailingStopInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: FvgTrailingStopParams) -> Self {
        Self {
            data: FvgTrailingStopData::Candles(candles),
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FvgTrailingStopParams,
    ) -> Self {
        Self {
            data: FvgTrailingStopData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FvgTrailingStopParams::default())
    }

    pub fn get_lookback(&self) -> usize {
        self.params.unmitigated_fvg_lookback.unwrap_or(5)
    }

    pub fn get_smoothing(&self) -> usize {
        self.params.smoothing_length.unwrap_or(9)
    }

    pub fn get_reset_on_cross(&self) -> bool {
        self.params.reset_on_cross.unwrap_or(false)
    }

    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            FvgTrailingStopData::Candles(c) => (&c.high, &c.low, &c.close),
            FvgTrailingStopData::Slices { high, low, close } => (high, low, close),
        }
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum FvgTrailingStopError {
    #[error("fvg_trailing_stop: Input data slice is empty.")]
    EmptyInputData,
    
    #[error("fvg_trailing_stop: All values are NaN.")]
    AllValuesNaN,
    
    #[error("fvg_trailing_stop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    
    #[error("fvg_trailing_stop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// ==================== CORE COMPUTATION ====================
#[inline]
fn fvg_ts_prepare<'a>(
    input: &'a FvgTrailingStopInput,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, bool), FvgTrailingStopError> {
    let (high, low, close) = input.as_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FvgTrailingStopError::EmptyInputData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(FvgTrailingStopError::InvalidPeriod { period: len, data_len: len });
    }
    let first = first_valid_ohlc(high, low, close);
    if first == usize::MAX {
        return Err(FvgTrailingStopError::AllValuesNaN);
    }
    let lookback = input.get_lookback();
    let smoothing_len = input.get_smoothing();
    let reset_on_cross = input.get_reset_on_cross();
    Ok((high, low, close, lookback, smoothing_len, reset_on_cross))
}

#[inline]
fn fvg_ts_compute_into(
    high: &[f64], low: &[f64], close: &[f64],
    lookback: usize, smoothing_len: usize, reset_on_cross: bool,
    upper: &mut [f64], lower: &mut [f64], upper_ts: &mut [f64], lower_ts: &mut [f64],
) {
    let len = high.len();
    // State
    let mut bull_lvls: VecDeque<f64> = VecDeque::with_capacity(lookback);
    let mut bear_lvls: VecDeque<f64> = VecDeque::with_capacity(lookback);
    let mut last_bull_non_na: Option<usize> = None;
    let mut last_bear_non_na: Option<usize> = None;
    let mut bull_hist: Vec<f64> = Vec::with_capacity(len);
    let mut bear_hist: Vec<f64> = Vec::with_capacity(len);
    let mut os: Option<i8> = None;
    let mut ts: Option<f64> = None;
    let mut ts_prev: Option<f64> = None;

    for i in 0..len {
        // defaults stay NaN; warm prefix already written by allocator
        // FVG detection
        if i >= 2 && !high[i-2].is_nan() && !low[i-2].is_nan() && !close[i-1].is_nan() {
            if low[i] > high[i-2] && close[i-1] > high[i-2] {
                bull_lvls.push_back(high[i-2]);
                if bull_lvls.len() > lookback { bull_lvls.pop_front(); }
            }
            if high[i] < low[i-2] && close[i-1] < low[i-2] {
                bear_lvls.push_back(low[i-2]);
                if bear_lvls.len() > lookback { bear_lvls.pop_front(); }
            }
        }
        // mitigation
        let c = close[i];
        bull_lvls.retain(|&lvl| c >= lvl);
        bear_lvls.retain(|&lvl| c <= lvl);

        let bull_avg = if bull_lvls.is_empty() { f64::NAN } else { bull_lvls.iter().sum::<f64>() / (bull_lvls.len() as f64) };
        let bear_avg = if bear_lvls.is_empty() { f64::NAN } else { bear_lvls.iter().sum::<f64>() / (bear_lvls.len() as f64) };
        if !bull_avg.is_nan() { last_bull_non_na = Some(i); }
        if !bear_avg.is_nan() { last_bear_non_na = Some(i); }

        // progressive SMA fallbacks
        let bull_bs = if bull_avg.is_nan() {
            match last_bull_non_na { Some(last) => ((i - last).max(1)).min(smoothing_len), None => 1 }
        } else { 1 };
        let bear_bs = if bear_avg.is_nan() {
            match last_bear_non_na { Some(last) => ((i - last).max(1)).min(smoothing_len), None => 1 }
        } else { 1 };

        let bull_sma = if bull_avg.is_nan() && i + 1 >= bull_bs {
            let mut sum = 0.0; for j in (i + 1 - bull_bs)..=i { sum += close[j]; } sum / bull_bs as f64
        } else { f64::NAN };
        let bear_sma = if bear_avg.is_nan() && i + 1 >= bear_bs {
            let mut sum = 0.0; for j in (i + 1 - bear_bs)..=i { sum += close[j]; } sum / bear_bs as f64
        } else { f64::NAN };

        let x_bull = if !bull_avg.is_nan() { bull_avg } else { bull_sma };
        let x_bear = if !bear_avg.is_nan() { bear_avg } else { bear_sma };
        bull_hist.push(x_bull);
        bear_hist.push(x_bear);

        // fixed-window SMA over x-series; NaN if any NaN in window
        let mut bull_disp = f64::NAN;
        let mut bear_disp = f64::NAN;
        if i + 1 >= smoothing_len {
            let start = i + 1 - smoothing_len;
            let mut ok = true; let mut s = 0.0;
            for j in start..=i { let v = bull_hist[j]; if v.is_nan() { ok = false; break; } s += v; }
            if ok { bull_disp = s / smoothing_len as f64; }
            let mut ok2 = true; let mut s2 = 0.0;
            for j in start..=i { let v = bear_hist[j]; if v.is_nan() { ok2 = false; break; } s2 += v; }
            if ok2 { bear_disp = s2 / smoothing_len as f64; }
        }

        let prev_os = os;
        let next_os = if !bear_disp.is_nan() && c > bear_disp { Some(1) }
                      else if !bull_disp.is_nan() && c < bull_disp { Some(-1) }
                      else { os };
        os = next_os;

        if let (Some(cur), Some(prev)) = (os, prev_os) {
            if cur == 1 && prev != 1      { ts = Some(bull_disp); }
            else if cur == -1 && prev != -1 { ts = Some(bear_disp); }
            else if cur == 1 {
                if let Some(t) = ts { ts = Some(bull_disp.max(t)); }
            } else if cur == -1 {
                if let Some(t) = ts { ts = Some(bear_disp.min(t)); }
            }
        } else {
            if os == Some(1)  { if let Some(t) = ts { ts = Some(bull_disp.max(t)); } }
            if os == Some(-1) { if let Some(t) = ts { ts = Some(bear_disp.min(t)); } }
        }

        if reset_on_cross {
            if os == Some(1) {
                if let Some(t) = ts {
                    if c < t { ts = None; }
                } else if !bear_disp.is_nan() && c > bear_disp { ts = Some(bull_disp); }
            } else if os == Some(-1) {
                if let Some(t) = ts {
                    if c > t { ts = None; }
                } else if !bull_disp.is_nan() && c < bull_disp { ts = Some(bear_disp); }
            }
        }

        let show = ts.is_some() || ts_prev.is_some();
        let ts_nz = ts.or(ts_prev);

        if os == Some(1) && show {
            lower[i]    = bull_disp;
            lower_ts[i] = ts_nz.unwrap_or(f64::NAN);
        } else if os == Some(-1) && show {
            upper[i]    = bear_disp;
            upper_ts[i] = ts_nz.unwrap_or(f64::NAN);
        }
        ts_prev = ts;
    }
}

// ==================== MAIN ALGORITHM ====================
#[inline]
pub fn fvg_trailing_stop(input: &FvgTrailingStopInput) -> Result<FvgTrailingStopOutput, FvgTrailingStopError> {
    fvg_trailing_stop_with_kernel(input, Kernel::Auto)
}

pub fn fvg_trailing_stop_with_kernel(
    input: &FvgTrailingStopInput,
    _kernel: Kernel, // kept for parity; scalar path
) -> Result<FvgTrailingStopOutput, FvgTrailingStopError> {
    let (h, l, c, lookback, smoothing_len, reset_on_cross) = fvg_ts_prepare(input)?;
    let len = h.len();
    let first = {
        // recompute once here; consistent with prepare
        let f = first_valid_ohlc(h, l, c);
        debug_assert_ne!(f, usize::MAX);
        f
    };
    // conservative warm prefix: need 2 bars for FVG check + smoothing window
    let warm = (first + 2 + smoothing_len.saturating_sub(1)).min(len);

    let mut upper    = alloc_with_nan_prefix(len, warm);
    let mut lower    = alloc_with_nan_prefix(len, warm);
    let mut upper_ts = alloc_with_nan_prefix(len, warm);
    let mut lower_ts = alloc_with_nan_prefix(len, warm);

    fvg_ts_compute_into(h, l, c, lookback, smoothing_len, reset_on_cross,
                        &mut upper, &mut lower, &mut upper_ts, &mut lower_ts);

    Ok(FvgTrailingStopOutput { upper, lower, upper_ts, lower_ts })
}

#[inline]
pub fn fvg_trailing_stop_into_slices(
    upper: &mut [f64], lower: &mut [f64], upper_ts: &mut [f64], lower_ts: &mut [f64],
    input: &FvgTrailingStopInput, _kern: Kernel,
) -> Result<(), FvgTrailingStopError> {
    let (h, l, c, lookback, smoothing_len, reset_on_cross) = fvg_ts_prepare(input)?;
    let len = h.len();
    if upper.len()!=len || lower.len()!=len || upper_ts.len()!=len || lower_ts.len()!=len {
        return Err(FvgTrailingStopError::InvalidPeriod { period: len, data_len: len });
    }
    // caller may preseed warm with NaNs; we just compute
    fvg_ts_compute_into(h, l, c, lookback, smoothing_len, reset_on_cross,
                        upper, lower, upper_ts, lower_ts);
    Ok(())
}

// ==================== BATCH OPERATIONS ====================
#[derive(Clone, Debug)]
pub struct FvgTsBatchRange {
    pub lookback: (usize, usize, usize),
    pub smoothing: (usize, usize, usize),
    pub reset_on_cross: (bool, bool), // false..true toggle
}

impl Default for FvgTsBatchRange {
    fn default() -> Self { 
        Self { 
            lookback: (5, 5, 0), 
            smoothing: (9, 9, 0), 
            reset_on_cross: (false, false) 
        } 
    }
}

#[derive(Clone, Debug)]
pub struct FvgTsBatchOutput {
    pub values: Vec<f64>,   // rows x cols, flattened
    pub combos: Vec<FvgTrailingStopParams>,
    pub rows: usize,        // = 4 * combos.len()
    pub cols: usize,
}

#[inline]
fn expand_grid_ts(r: &FvgTsBatchRange) -> Vec<FvgTrailingStopParams> {
    let mut v = Vec::new();
    let looks = if r.lookback.2 == 0 { vec![r.lookback.0] } else { (r.lookback.0..=r.lookback.1).step_by(r.lookback.2).collect() };
    let smooths = if r.smoothing.2 == 0 { vec![r.smoothing.0] } else { (r.smoothing.0..=r.smoothing.1).step_by(r.smoothing.2).collect() };
    let resets = if r.reset_on_cross.0 == r.reset_on_cross.1 { vec![r.reset_on_cross.0] } else { vec![false, true] };
    for &lb in &looks { 
        for &sm in &smooths { 
            for &rs in &resets {
                v.push(FvgTrailingStopParams{ 
                    unmitigated_fvg_lookback: Some(lb), 
                    smoothing_length: Some(sm), 
                    reset_on_cross: Some(rs) 
                });
            }
        }
    }
    v
}

pub fn fvg_trailing_stop_batch_with_kernel(
    high: &[f64], low: &[f64], close: &[f64], sweep: &FvgTsBatchRange, _kern: Kernel
) -> Result<FvgTsBatchOutput, FvgTrailingStopError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FvgTrailingStopError::EmptyInputData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(FvgTrailingStopError::InvalidPeriod { period: len, data_len: len });
    }
    let first = first_valid_ohlc(high, low, close);
    if first == usize::MAX { 
        return Err(FvgTrailingStopError::AllValuesNaN); 
    }

    let combos = expand_grid_ts(sweep);
    let cols = len;
    let rows = combos.len() * 4;

    // per-combo warm prefix (same for the 4 rows of that combo)
    let mut warm_vec = Vec::with_capacity(rows);
    for prm in &combos {
        let sm = prm.smoothing_length.unwrap_or(9);
        let warm = (first + 2 + sm.saturating_sub(1)).min(cols);
        warm_vec.extend_from_slice(&[warm, warm, warm, warm]);
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm_vec);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len())
    };

    for (ci, prm) in combos.iter().enumerate() {
        let base = ci * 4 * cols;
        let (first_half, second_half) = out[base..base + 4*cols].split_at_mut(2*cols);
        let (u, l) = first_half.split_at_mut(cols);
        let (uts, lts) = second_half.split_at_mut(cols);
        fvg_ts_compute_into(
            high, low, close,
            prm.unmitigated_fvg_lookback.unwrap(), 
            prm.smoothing_length.unwrap(), 
            prm.reset_on_cross.unwrap_or(false),
            u, l, uts, lts
        );
    }

    let values = unsafe {
        Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity())
    };
    core::mem::forget(guard);
    Ok(FvgTsBatchOutput { values, combos, rows, cols })
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "fvg_trailing_stop")]
#[pyo3(signature = (high, low, close, unmitigated_fvg_lookback, smoothing_length, reset_on_cross, kernel=None))]
pub fn fvg_trailing_stop_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low:  PyReadonlyArray1<'py, f64>,
    close:PyReadonlyArray1<'py, f64>,
    unmitigated_fvg_lookback: usize,
    smoothing_length: usize,
    reset_on_cross: bool,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::IntoPyArray;
    let (h,l,c) = (high.as_slice()?, low.as_slice()?, close.as_slice()?);
    let kern = validate_kernel(kernel, false)?;
    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
        smoothing_length: Some(smoothing_length),
        reset_on_cross: Some(reset_on_cross),
    };
    let input = FvgTrailingStopInput::from_slices(h,l,c,params);
    let out = py.allow_threads(|| fvg_trailing_stop_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.upper.into_pyarray(py),
        out.lower.into_pyarray(py),
        out.upper_ts.into_pyarray(py),
        out.lower_ts.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "fvg_trailing_stop_batch")]
#[pyo3(signature = (high, low, close, lookback_range, smoothing_range, reset_toggle, kernel=None))]
pub fn fvg_trailing_stop_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low:  PyReadonlyArray1<'py, f64>,
    close:PyReadonlyArray1<'py, f64>,
    lookback_range:(usize,usize,usize),
    smoothing_range:(usize,usize,usize),
    reset_toggle:(bool,bool),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let (h,l,c) = (high.as_slice()?, low.as_slice()?, close.as_slice()?);
    let sweep = FvgTsBatchRange{ lookback:lookback_range, smoothing:smoothing_range, reset_on_cross:reset_toggle };
    let kern = validate_kernel(kernel, true)?;
    let out = py.allow_threads(|| fvg_trailing_stop_batch_with_kernel(h,l,c,&sweep,kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    // reshape helpers: rows = 4*combos, cols = len
    let values = unsafe { PyArray1::new(py, [out.values.len()], false) };
    unsafe { values.as_slice_mut()? }.copy_from_slice(&out.values);
    dict.set_item("values", values.reshape((out.rows, out.cols))?)?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    dict.set_item("lookbacks", out.combos.iter().map(|p| p.unmitigated_fvg_lookback.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("smoothings", out.combos.iter().map(|p| p.smoothing_length.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("resets", out.combos.iter().map(|p| p.reset_on_cross.unwrap_or(false)).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(dict)
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FvgTsJsOutput { 
    pub values: Vec<f64>, 
    pub rows: usize, 
    pub cols: usize 
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "fvgTrailingStop")]
pub fn fvg_trailing_stop_js(
    high:&[f64], low:&[f64], close:&[f64],
    unmitigated_fvg_lookback:usize, smoothing_length:usize, reset_on_cross:bool
) -> Result<JsValue, JsValue> {
    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
        smoothing_length: Some(smoothing_length),
        reset_on_cross: Some(reset_on_cross),
    };
    let input = FvgTrailingStopInput::from_slices(high, low, close, params);
    // allocate flattened 4 x len once
    let len = high.len();
    let mut buf_mu = make_uninit_matrix(4, len);
    let first = first_valid_ohlc(high, low, close);
    let warm = if first != usize::MAX {
        (first + 2 + smoothing_length.saturating_sub(1)).min(len)
    } else {
        0
    };
    init_matrix_prefixes(&mut buf_mu, len, &[warm, warm, warm, warm]);
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let (first_half, second_half) = out.split_at_mut(2*len);
    let (u, l) = first_half.split_at_mut(len);
    let (uts, lts) = second_half.split_at_mut(len);
    fvg_trailing_stop_into_slices(u,l,uts,lts,&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let values = unsafe { Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity()) };
    core::mem::forget(guard);
    let js = FvgTsJsOutput { values, rows:4, cols:len };
    serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn fvg_trailing_stop_into_flat(
    high_ptr:*const f64, low_ptr:*const f64, close_ptr:*const f64,
    out_ptr:*mut f64, len:usize,
    unmitigated_fvg_lookback:usize, smoothing_length:usize, reset_on_cross:bool
) -> Result<(), JsValue> {
    if [high_ptr as usize, low_ptr as usize, close_ptr as usize, out_ptr as usize].iter().any(|&p| p==0) {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let h = core::slice::from_raw_parts(high_ptr, len);
        let l = core::slice::from_raw_parts(low_ptr, len);
        let c = core::slice::from_raw_parts(close_ptr, len);
        let out = core::slice::from_raw_parts_mut(out_ptr, 4*len);
        let (first_half, second_half) = out.split_at_mut(2*len);
        let (u, lw) = first_half.split_at_mut(len);
        let (uts, lts) = second_half.split_at_mut(len);
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(unmitigated_fvg_lookback),
            smoothing_length: Some(smoothing_length),
            reset_on_cross: Some(reset_on_cross),
        };
        let input = FvgTrailingStopInput::from_slices(h,l,c,params);
        fvg_trailing_stop_into_slices(u,lw,uts,lts,&input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

// ==================== BUILDER PATTERN ====================
#[derive(Copy, Clone, Debug)]
pub struct FvgTrailingStopBuilder {
    unmitigated_fvg_lookback: Option<usize>,
    smoothing_length: Option<usize>,
    reset_on_cross: Option<bool>,
    kernel: Kernel,
}

impl Default for FvgTrailingStopBuilder {
    fn default() -> Self {
        Self {
            unmitigated_fvg_lookback: None,
            smoothing_length: None,
            reset_on_cross: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FvgTrailingStopBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn lookback(mut self, n: usize) -> Self {
        self.unmitigated_fvg_lookback = Some(n);
        self
    }
    
    pub fn smoothing(mut self, n: usize) -> Self {
        self.smoothing_length = Some(n);
        self
    }
    
    pub fn reset_on_cross(mut self, reset: bool) -> Self {
        self.reset_on_cross = Some(reset);
        self
    }
    
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    
    pub fn apply(&self, candles: &Candles) -> Result<FvgTrailingStopOutput, FvgTrailingStopError> {
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: self.unmitigated_fvg_lookback,
            smoothing_length: self.smoothing_length,
            reset_on_cross: self.reset_on_cross,
        };
        let input = FvgTrailingStopInput::from_candles(candles, params);
        fvg_trailing_stop_with_kernel(&input, self.kernel)
    }
}

// ==================== STREAMING SUPPORT ====================
#[cfg(feature = "python")]
#[pyclass]
pub struct FvgTrailingStopStreamPy;

// Minimal streaming implementation to satisfy Python bindings
// Full streaming support can be added later if needed