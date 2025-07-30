//! # Weighted Close Price (WCLPRICE)
//!
//! Computes `(high + low + 2*close) / 4` for each index. NaN if any input field is NaN at index.
//!
//! ## Parameters
//! - None (uses all of high, low, close)
//!
//! ## Errors
//! - **EmptyData**: Input is empty
//! - **AllValuesNaN**: All values are NaN in any required field
//!
//! ## Returns
//! - **Ok(WclpriceOutput)** on success, contains a `Vec<f64>`
//! - **Err(WclpriceError)** otherwise

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// Note: Unlike ALMA, WCLPRICE doesn't implement AsRef<[f64]> because it needs multiple slices
// This is intentional as WCLPRICE requires high, low, and close data

#[derive(Debug, Clone)]
pub enum WclpriceData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct WclpriceOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct WclpriceParams;

impl Default for WclpriceParams {
	fn default() -> Self {
		Self
	}
}

#[derive(Debug, Clone)]
pub struct WclpriceInput<'a> {
	pub data: WclpriceData<'a>,
	pub params: WclpriceParams,
}

impl<'a> WclpriceInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles) -> Self {
		Self {
			data: WclpriceData::Candles { candles },
			params: WclpriceParams::default(),
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
		Self {
			data: WclpriceData::Slices { high, low, close },
			params: WclpriceParams::default(),
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct WclpriceBuilder {
	kernel: Kernel,
}
impl Default for WclpriceBuilder {
	fn default() -> Self {
		Self { kernel: Kernel::Auto }
	}
}
impl WclpriceBuilder {
	#[inline]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn apply(self, candles: &Candles) -> Result<WclpriceOutput, WclpriceError> {
		let i = WclpriceInput::from_candles(candles);
		wclprice_with_kernel(&i, self.kernel)
	}
	#[inline]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<WclpriceOutput, WclpriceError> {
		let i = WclpriceInput::from_slices(high, low, close);
		wclprice_with_kernel(&i, self.kernel)
	}
	#[inline]
	pub fn into_stream(self) -> WclpriceStream {
		WclpriceStream::default()
	}
}

#[derive(Debug, Error)]
pub enum WclpriceError {
	#[error("wclprice: Empty data provided.")]
	EmptyData,
	#[error("wclprice: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn wclprice(input: &WclpriceInput) -> Result<WclpriceOutput, WclpriceError> {
	wclprice_with_kernel(input, Kernel::Auto)
}

pub fn wclprice_with_kernel(input: &WclpriceInput, kernel: Kernel) -> Result<WclpriceOutput, WclpriceError> {
	let (high, low, close) = match &input.data {
		WclpriceData::Candles { candles } => {
			let high = candles.select_candle_field("high").unwrap();
			let low = candles.select_candle_field("low").unwrap();
			let close = candles.select_candle_field("close").unwrap();
			(high, low, close)
		}
		WclpriceData::Slices { high, low, close } => (*high, *low, *close),
	};

	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(WclpriceError::EmptyData);
	}
	let len = high.len().min(low.len()).min(close.len());
	if len == 0 {
		return Err(WclpriceError::EmptyData);
	}
	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(WclpriceError::AllValuesNaN)?;

	let mut out = alloc_with_nan_prefix(len, first);
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => wclprice_scalar(high, low, close, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => wclprice_avx2(high, low, close, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => wclprice_avx512(high, low, close, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(WclpriceOutput { values: out })
}

#[inline]
pub fn wclprice_into_slice(dst: &mut [f64], input: &WclpriceInput, kern: Kernel) -> Result<(), WclpriceError> {
	let (high, low, close) = match &input.data {
		WclpriceData::Candles { candles } => {
			let high = candles.select_candle_field("high").unwrap();
			let low = candles.select_candle_field("low").unwrap();
			let close = candles.select_candle_field("close").unwrap();
			(high, low, close)
		}
		WclpriceData::Slices { high, low, close } => (*high, *low, *close),
	};

	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(WclpriceError::EmptyData);
	}
	
	let len = high.len().min(low.len()).min(close.len());
	if len == 0 {
		return Err(WclpriceError::EmptyData);
	}
	
	if dst.len() != len {
		return Err(WclpriceError::EmptyData); // Should have a better error, but maintaining compatibility
	}
	
	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(WclpriceError::AllValuesNaN)?;

	// Fill warmup with NaN
	for v in &mut dst[..first] {
		*v = f64::NAN;
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => wclprice_scalar(high, low, close, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => wclprice_avx2(high, low, close, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => wclprice_avx512(high, low, close, first, dst),
			_ => unreachable!(),
		}
	}
	
	Ok(())
}

#[inline]
pub fn wclprice_scalar(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	let len = high.len().min(low.len()).min(close.len());
	for i in first_valid..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		if h.is_nan() || l.is_nan() || c.is_nan() {
			out[i] = f64::NAN;
		} else {
			out[i] = (h + l + 2.0 * c) / 4.0;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	unsafe { wclprice_avx512_short(high, low, close, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx2(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_short(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_long(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_scalar(high, low, close, first_valid, out)
}

#[inline]
pub fn wclprice_row_scalar(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx2(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_avx2(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_avx512(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_short(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_avx512_short(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_long(high: &[f64], low: &[f64], close: &[f64], first_valid: usize, out: &mut [f64]) {
	wclprice_avx512_long(high, low, close, first_valid, out)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchRange; // No parameters

impl Default for WclpriceBatchRange {
	fn default() -> Self {
		Self
	}
}

#[derive(Clone, Debug, Default)]
pub struct WclpriceBatchBuilder {
	kernel: Kernel,
}
impl WclpriceBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<WclpriceBatchOutput, WclpriceError> {
		wclprice_batch_with_kernel(high, low, close, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
		let high = c.select_candle_field("high").unwrap();
		let low = c.select_candle_field("low").unwrap();
		let close = c.select_candle_field("close").unwrap();
		self.apply_slices(high, low, close)
	}
	pub fn with_default_candles(c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
		WclpriceBatchBuilder::new().apply_candles(c)
	}
}

pub fn wclprice_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	k: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => Kernel::ScalarBatch,
	};
	wclprice_batch_par_slice(high, low, close, kernel)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<WclpriceParams>,
	pub rows: usize,
	pub cols: usize,
}
impl WclpriceBatchOutput {
	pub fn values_for(&self, _params: &WclpriceParams) -> Option<&[f64]> {
		if self.rows == 1 {
			Some(&self.values[..self.cols])
		} else {
			None
		}
	}
}

#[inline(always)]
pub fn wclprice_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kern: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
	wclprice_batch_inner(high, low, close, kern, false)
}
#[inline(always)]
pub fn wclprice_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kern: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
	wclprice_batch_inner(high, low, close, kern, true)
}
#[inline(always)]
fn wclprice_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kern: Kernel,
	_parallel: bool,
) -> Result<WclpriceBatchOutput, WclpriceError> {
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(WclpriceError::EmptyData);
	}
	let len = high.len().min(low.len()).min(close.len());
	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(WclpriceError::AllValuesNaN)?;
	let mut values = alloc_with_nan_prefix(len, first);
	unsafe {
		match kern {
			Kernel::ScalarBatch | Kernel::Scalar => wclprice_row_scalar(high, low, close, first, &mut values),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2Batch | Kernel::Avx2 => wclprice_row_avx2(high, low, close, first, &mut values),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512Batch | Kernel::Avx512 => wclprice_row_avx512(high, low, close, first, &mut values),
			_ => unreachable!(),
		}
	}
	let combos = expand_grid(&WclpriceBatchRange);
	Ok(WclpriceBatchOutput {
		values,
		combos,
		rows: 1,
		cols: len,
	})
}

#[inline(always)]
fn expand_grid(_r: &WclpriceBatchRange) -> Vec<WclpriceParams> {
	vec![WclpriceParams]
}

#[inline(always)]
fn wclprice_batch_inner_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kern: Kernel,
	_parallel: bool,
	out: &mut [f64],
) -> Result<Vec<WclpriceParams>, WclpriceError> {
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(WclpriceError::EmptyData);
	}
	let len = high.len().min(low.len()).min(close.len());
	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(WclpriceError::AllValuesNaN)?;
	
	// Since WCLPRICE has no parameters, we only have one row
	// Fill the output slice with NaN up to first valid index
	if first > 0 {
		out[..first].fill(f64::NAN);
	}
	
	unsafe {
		match kern {
			Kernel::ScalarBatch | Kernel::Scalar => wclprice_row_scalar(high, low, close, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2Batch | Kernel::Avx2 => wclprice_row_avx2(high, low, close, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512Batch | Kernel::Avx512 => wclprice_row_avx512(high, low, close, first, out),
			_ => unreachable!(),
		}
	}
	
	Ok(vec![WclpriceParams])
}

#[derive(Debug, Clone)]
pub struct WclpriceStream;
impl Default for WclpriceStream {
	fn default() -> Self {
		Self
	}
}
impl WclpriceStream {
	#[inline(always)]
	pub fn update(&mut self, h: f64, l: f64, c: f64) -> Option<f64> {
		if h.is_nan() || l.is_nan() || c.is_nan() {
			None
		} else {
			Some((h + l + 2.0 * c) / 4.0)
		}
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "wclprice")]
#[pyo3(signature = (high, low, close, kernel=None))]
pub fn wclprice_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let len = high_slice.len().min(low_slice.len()).min(close_slice.len());
	let kern = validate_kernel(kernel, false)?;

	let input = WclpriceInput::from_slices(high_slice, low_slice, close_slice);

	// Allocate output array with uninitialized memory
	let output = unsafe { PyArray1::<f64>::new(py, [len], false) };
	let slice_out = unsafe { output.as_slice_mut()? };
	
	py.allow_threads(|| {
		wclprice_into_slice(slice_out, &input, kern)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(output.to_owned())
}

#[cfg(feature = "python")]
#[pyclass(name = "WclpriceStream")]
pub struct WclpriceStreamPy {
	stream: WclpriceStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WclpriceStreamPy {
	#[new]
	fn new() -> PyResult<Self> {
		Ok(WclpriceStreamPy {
			stream: WclpriceStream::default(),
		})
	}

	fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		self.stream.update(high, low, close)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "wclprice_batch")]
#[pyo3(signature = (high, low, close, kernel=None))]
pub fn wclprice_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	// WCLPRICE has no parameters, so there's only one combination
	let rows = 1;
	let cols = high_slice.len().min(low_slice.len()).min(close_slice.len());

	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	let combos = py.allow_threads(|| {
		let batch_kernel = match kern {
			Kernel::Auto => detect_best_batch_kernel(),
			k => k,
		};
		
		// Map batch kernels to regular kernels
		let simd = match batch_kernel {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			_ => batch_kernel,
		};

		// Write directly to the output buffer - zero copy
		wclprice_batch_inner_into(high_slice, low_slice, close_slice, simd, true, slice_out)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let result = PyDict::new(py);
	result.set_item("values", out_arr)?;
	result.set_item("rows", rows)?;
	result.set_item("cols", cols)?;

	// Create params list (empty for WCLPRICE)
	let params_list = PyList::empty(py);
	let params = PyDict::new(py);
	params_list.append(params)?;
	result.set_item("params", params_list)?;

	Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_js(high: &[f64], low: &[f64], close: &[f64]) -> Result<Vec<f64>, JsValue> {
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(JsValue::from_str("wclprice: Empty data provided"));
	}
	
	let input = WclpriceInput::from_slices(high, low, close);
	let mut output = vec![0.0; high.len().min(low.len()).min(close.len())];
	
	wclprice_into_slice(&mut output, &input, detect_best_kernel())
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		let input = WclpriceInput::from_slices(high, low, close);
		
		// Check for aliasing with any input pointer
		if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			wclprice_into_slice(&mut temp, &input, detect_best_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			wclprice_into_slice(out, &input, detect_best_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WclpriceBatchConfig {
	// WCLPRICE has no parameters, so this is empty
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WclpriceBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<WclpriceParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = wclprice_batch)]
pub fn wclprice_batch_js(high: &[f64], low: &[f64], close: &[f64], _config: JsValue) -> Result<JsValue, JsValue> {
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(JsValue::from_str("wclprice: Empty data provided"));
	}
	
	// Use the proper batch infrastructure
	let output = wclprice_batch_inner(high, low, close, detect_best_kernel(), false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = WclpriceBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wclprice_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		// WCLPRICE has no parameters, so only 1 row
		let rows = 1;
		
		// Check for aliasing with any input pointer
		if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			wclprice_batch_inner_into(high, low, close, detect_best_kernel(), false, &mut temp)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			wclprice_batch_inner_into(high, low, close, detect_best_kernel(), false, out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_wclprice_slices(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high = vec![59230.0, 59220.0, 59077.0, 59160.0, 58717.0];
		let low = vec![59222.0, 59211.0, 59077.0, 59143.0, 58708.0];
		let close = vec![59225.0, 59210.0, 59080.0, 59150.0, 58710.0];
		let input = WclpriceInput::from_slices(&high, &low, &close);
		let output = wclprice_with_kernel(&input, kernel)?;
		let expected = vec![59225.5, 59212.75, 59078.5, 59150.75, 58711.25];
		for (i, &v) in output.values.iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-2,
				"[{test}] mismatch at {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}
	fn check_wclprice_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let input = WclpriceInput::from_candles(&candles);
		let output = wclprice_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_wclprice_empty_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high: [f64; 0] = [];
		let low: [f64; 0] = [];
		let close: [f64; 0] = [];
		let input = WclpriceInput::from_slices(&high, &low, &close);
		let res = wclprice_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail with empty data", test);
		Ok(())
	}
	fn check_wclprice_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high = vec![f64::NAN, f64::NAN];
		let low = vec![f64::NAN, f64::NAN];
		let close = vec![f64::NAN, f64::NAN];
		let input = WclpriceInput::from_slices(&high, &low, &close);
		let res = wclprice_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail with all NaN", test);
		Ok(())
	}
	fn check_wclprice_partial_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high = vec![f64::NAN, 59000.0];
		let low = vec![f64::NAN, 58950.0];
		let close = vec![f64::NAN, 58975.0];
		let input = WclpriceInput::from_slices(&high, &low, &close);
		let output = wclprice_with_kernel(&input, kernel)?;
		assert!(output.values[0].is_nan());
		assert!((output.values[1] - (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0).abs() < 1e-8);
		Ok(())
	}
	macro_rules! generate_all_wclprice_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                   #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }
	generate_all_wclprice_tests!(
		check_wclprice_slices,
		check_wclprice_candles,
		check_wclprice_empty_data,
		check_wclprice_all_nan,
		check_wclprice_partial_nan
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = WclpriceBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let row = output.values_for(&WclpriceParams).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}
	macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        }
    }
	gen_batch_tests!(check_batch_default_row);
}
