//! # Volume Price Trend (VPT)
//!
//! Exact match for Jesse's implementation (shifted array approach).
//!
//! ## Parameters
//! None (uses price/volume arrays).
//!
//! ## Errors
//! - **EmptyData**: vpt: Input price or volume data is empty or mismatched.
//! - **AllValuesNaN**: vpt: All input price or volume values are NaN.
//! - **NotEnoughValidData**: vpt: Fewer than 2 valid price/volume points.
//!
//! ## Returns
//! - **Ok(VptOutput)** with output array.
//! - **Err(VptError)** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum VptData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slices { price: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct VptOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VptParams;

#[derive(Debug, Clone)]
pub struct VptInput<'a> {
	pub data: VptData<'a>,
	pub params: VptParams,
}

impl<'a> VptInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
		Self {
			data: VptData::Candles { candles, source },
			params: VptParams::default(),
		}
	}

	#[inline]
	pub fn from_slices(price: &'a [f64], volume: &'a [f64]) -> Self {
		Self {
			data: VptData::Slices { price, volume },
			params: VptParams::default(),
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: VptData::Candles {
				candles,
				source: "close",
			},
			params: VptParams::default(),
		}
	}
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VptBuilder {
	kernel: Kernel,
}

impl VptBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self { kernel: Kernel::Auto }
	}

	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<VptOutput, VptError> {
		let i = VptInput::with_default_candles(c);
		vpt_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
		let i = VptInput::from_slices(price, volume);
		vpt_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> VptStream {
		VptStream::default()
	}
}

#[derive(Debug, Error)]
pub enum VptError {
	#[error("vpt: Empty data provided.")]
	EmptyData,
	#[error("vpt: All price/volume values are NaN.")]
	AllValuesNaN,
	#[error("vpt: Not enough valid data (fewer than 2 valid points).")]
	NotEnoughValidData,
}

#[inline]
pub fn vpt(input: &VptInput) -> Result<VptOutput, VptError> {
	vpt_with_kernel(input, Kernel::Auto)
}

pub fn vpt_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
	let (price, volume) = match &input.data {
		VptData::Candles { candles, source } => {
			let price = source_type(candles, source);
			let vol = candles.select_candle_field("volume").map_err(|_| VptError::EmptyData)?;
			(price, vol)
		}
		VptData::Slices { price, volume } => (*price, *volume),
	};

	if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
		return Err(VptError::EmptyData);
	}

	let valid_count = price
		.iter()
		.zip(volume.iter())
		.filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
		.count();

	if valid_count == 0 {
		return Err(VptError::AllValuesNaN);
	}
	if valid_count < 2 {
		return Err(VptError::NotEnoughValidData);
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => vpt_scalar(price, volume),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vpt_avx2(price, volume),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vpt_avx512(price, volume),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn vpt_scalar(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
	let n = price.len();
	let mut res = alloc_with_nan_prefix(n, 1);
	
	// VPT uses "shifted array approach": output[i] = vpt_val[i] + vpt_val[i-1]
	let mut prev_vpt_val = f64::NAN;
	
	for i in 1..n {
		let p0 = price[i - 1];
		let p1 = price[i];
		let v1 = volume[i];
		
		// Calculate current VPT value
		let vpt_val = if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
			f64::NAN
		} else {
			v1 * ((p1 - p0) / p0)
		};
		
		// Output is current VPT value + previous VPT value (shifted array approach)
		res[i] = if vpt_val.is_nan() || prev_vpt_val.is_nan() {
			f64::NAN
		} else {
			vpt_val + prev_vpt_val
		};
		
		// Save current VPT value for next iteration
		prev_vpt_val = vpt_val;
	}
	
	Ok(VptOutput { values: res })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx2(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
	// For API parity only; reuses scalar logic.
	vpt_scalar(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
	// For API parity only; reuses scalar logic.
	vpt_scalar(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_short(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
	vpt_avx512(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_long(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
	vpt_avx512(price, volume)
}

#[inline]
pub fn vpt_indicator(input: &VptInput) -> Result<VptOutput, VptError> {
	vpt(input)
}

#[inline]
pub fn vpt_indicator_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
	vpt_with_kernel(input, kernel)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx2(input: &VptInput) -> Result<VptOutput, VptError> {
	unsafe {
		let (price, volume) = match &input.data {
			VptData::Candles { candles, source } => {
				let price = source_type(candles, source);
				let vol = candles.select_candle_field("volume").unwrap();
				(price, vol)
			}
			VptData::Slices { price, volume } => (*price, *volume),
		};
		vpt_avx2(price, volume)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512(input: &VptInput) -> Result<VptOutput, VptError> {
	unsafe {
		let (price, volume) = match &input.data {
			VptData::Candles { candles, source } => {
				let price = source_type(candles, source);
				let vol = candles.select_candle_field("volume").unwrap();
				(price, vol)
			}
			VptData::Slices { price, volume } => (*price, *volume),
		};
		vpt_avx512(price, volume)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_short(input: &VptInput) -> Result<VptOutput, VptError> {
	unsafe {
		let (price, volume) = match &input.data {
			VptData::Candles { candles, source } => {
				let price = source_type(candles, source);
				let vol = candles.select_candle_field("volume").unwrap();
				(price, vol)
			}
			VptData::Slices { price, volume } => (*price, *volume),
		};
		vpt_avx512_short(price, volume)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_long(input: &VptInput) -> Result<VptOutput, VptError> {
	unsafe {
		let (price, volume) = match &input.data {
			VptData::Candles { candles, source } => {
				let price = source_type(candles, source);
				let vol = candles.select_candle_field("volume").unwrap();
				(price, vol)
			}
			VptData::Slices { price, volume } => (*price, *volume),
		};
		vpt_avx512_long(price, volume)
	}
}

#[inline]
pub fn vpt_indicator_scalar(input: &VptInput) -> Result<VptOutput, VptError> {
	unsafe {
		let (price, volume) = match &input.data {
			VptData::Candles { candles, source } => {
				let price = source_type(candles, source);
				let vol = candles.select_candle_field("volume").unwrap();
				(price, vol)
			}
			VptData::Slices { price, volume } => (*price, *volume),
		};
		vpt_scalar(price, volume)
	}
}

#[inline]
pub fn vpt_expand_grid() -> Vec<VptParams> {
	// VPT has no parameters, return single empty params
	// Using array instead of vec! to avoid allocation
	[VptParams].to_vec()
}

/// Write VPT directly to output slice - no allocations
pub fn vpt_into_slice(dst: &mut [f64], price: &[f64], volume: &[f64], kern: Kernel) -> Result<(), VptError> {
	if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
		return Err(VptError::EmptyData);
	}
	
	if dst.len() != price.len() {
		return Err(VptError::EmptyData); // Using EmptyData as we don't have InvalidLength
	}
	
	let valid_count = price
		.iter()
		.zip(volume.iter())
		.filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
		.count();
	
	if valid_count == 0 {
		return Err(VptError::AllValuesNaN);
	}
	if valid_count < 2 {
		return Err(VptError::NotEnoughValidData);
	}
	
	// Use the row version which writes directly to output
	unsafe {
		match kern {
			Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => vpt_row_scalar(price, volume, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vpt_row_avx2(price, volume, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vpt_row_avx512(price, volume, dst),
			_ => vpt_row_scalar(price, volume, dst),
		}
	}
	
	Ok(())
}

pub fn vpt_batch_inner_into(
	price: &[f64],
	volume: &[f64],
	_range: &VptBatchRange,
	kern: Kernel,
	_parallel: bool,
	out: &mut [f64],
) -> Result<Vec<VptParams>, VptError> {
	if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
		return Err(VptError::EmptyData);
	}

	// VPT has no parameters, so only one "combo"
	let combos = vec![VptParams];
	
	// Single row output
	match kern {
		Kernel::Scalar | Kernel::ScalarBatch => unsafe { vpt_row_scalar(price, volume, out) },
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 | Kernel::Avx2Batch => unsafe { vpt_row_avx2(price, volume, out) },
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 | Kernel::Avx512Batch => unsafe { vpt_row_avx512(price, volume, out) },
		_ => unsafe { vpt_row_scalar(price, volume, out) },
	}

	Ok(combos)
}

#[derive(Clone, Debug, Default)]
pub struct VptStream {
	last_price: f64,
	last_vpt: f64,
	is_initialized: bool,
}

impl VptStream {
	#[inline]
	pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
		if !self.is_initialized {
			self.last_price = price;
			self.last_vpt = f64::NAN;
			self.is_initialized = true;
			return None;
		}
		if self.last_price.is_nan() || self.last_price == 0.0 || price.is_nan() || volume.is_nan() {
			self.last_price = price;
			self.last_vpt = f64::NAN;
			return Some(f64::NAN);
		}
		let vpt_val = volume * ((price - self.last_price) / self.last_price);
		let out = if self.last_vpt.is_nan() {
			f64::NAN
		} else {
			vpt_val + self.last_vpt
		};
		self.last_price = price;
		self.last_vpt = vpt_val;
		Some(out)
	}
}

#[derive(Clone, Debug, Default)]
pub struct VptBatchRange;

#[derive(Clone, Debug, Default)]
pub struct VptBatchBuilder {
	kernel: Kernel,
}

impl VptBatchBuilder {
	pub fn new() -> Self {
		Self { kernel: Kernel::Auto }
	}

	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptBatchOutput, VptError> {
		vpt_batch_with_kernel(price, volume, self.kernel)
	}

	pub fn with_default_slices(price: &[f64], volume: &[f64], k: Kernel) -> Result<VptBatchOutput, VptError> {
		VptBatchBuilder::new().kernel(k).apply_slices(price, volume)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VptBatchOutput, VptError> {
		let price = source_type(c, src);
		let volume = c.select_candle_field("volume").map_err(|_| VptError::EmptyData)?;
		self.apply_slices(price, volume)
	}

	pub fn with_default_candles(c: &Candles) -> Result<VptBatchOutput, VptError> {
		VptBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn vpt_batch_with_kernel(price: &[f64], volume: &[f64], k: Kernel) -> Result<VptBatchOutput, VptError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => Kernel::ScalarBatch,
	};
	vpt_batch_par_slice(price, volume, kernel)
}

#[derive(Clone, Debug)]
pub struct VptBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<VptParams>,
	pub rows: usize,
	pub cols: usize,
}

impl VptBatchOutput {
	pub fn row_for_params(&self, _p: &VptParams) -> Option<usize> {
		Some(0)
	}

	pub fn values_for(&self, _p: &VptParams) -> Option<&[f64]> {
		Some(&self.values[..])
	}
}

#[inline(always)]
pub fn vpt_batch_slice(price: &[f64], volume: &[f64], kern: Kernel) -> Result<VptBatchOutput, VptError> {
	vpt_batch_inner(price, volume, kern, false)
}

#[inline(always)]
pub fn vpt_batch_par_slice(price: &[f64], volume: &[f64], kern: Kernel) -> Result<VptBatchOutput, VptError> {
	vpt_batch_inner(price, volume, kern, true)
}

#[inline(always)]
fn vpt_batch_inner(price: &[f64], volume: &[f64], kern: Kernel, _parallel: bool) -> Result<VptBatchOutput, VptError> {
	let combos = vpt_expand_grid();
	let rows = 1;
	let cols = price.len();

	let output = match kern {
		Kernel::Scalar | Kernel::ScalarBatch => unsafe { vpt_scalar(price, volume)? },
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 | Kernel::Avx2Batch => unsafe { vpt_avx2(price, volume)? },
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 | Kernel::Avx512Batch => unsafe { vpt_avx512(price, volume)? },
		_ => unreachable!(),
	};

	Ok(VptBatchOutput {
		values: output.values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn vpt_row_scalar(price: &[f64], volume: &[f64], out: &mut [f64]) {
	let n = price.len();
	
	// First value is always NaN
	out[0] = f64::NAN;
	
	// VPT uses "shifted array approach": output[i] = vpt_val[i] + vpt_val[i-1]
	let mut prev_vpt_val = f64::NAN;
	
	for i in 1..n {
		let p0 = price[i - 1];
		let p1 = price[i];
		let v1 = volume[i];
		
		// Calculate current VPT value
		let vpt_val = if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
			f64::NAN
		} else {
			v1 * ((p1 - p0) / p0)
		};
		
		// Output is current VPT value + previous VPT value (shifted array approach)
		out[i] = if vpt_val.is_nan() || prev_vpt_val.is_nan() {
			f64::NAN
		} else {
			vpt_val + prev_vpt_val
		};
		
		// Save current VPT value for next iteration
		prev_vpt_val = vpt_val;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2(price: &[f64], volume: &[f64], out: &mut [f64]) {
	vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512(price: &[f64], volume: &[f64], out: &mut [f64]) {
	vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_short(price: &[f64], volume: &[f64], out: &mut [f64]) {
	vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_long(price: &[f64], volume: &[f64], out: &mut [f64]) {
	vpt_row_scalar(price, volume, out)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpt")]
#[pyo3(signature = (price, volume, kernel=None))]
pub fn vpt_py<'py>(
	py: Python<'py>,
	price: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let price_slice = price.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let input = VptInput::from_slices(price_slice, volume_slice);

	let result_vec: Vec<f64> = py
		.allow_threads(|| vpt_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VptStream")]
pub struct VptStreamPy {
	stream: VptStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VptStreamPy {
	#[new]
	fn new() -> PyResult<Self> {
		Ok(VptStreamPy {
			stream: VptStream::default(),
		})
	}

	fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
		self.stream.update(price, volume)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpt_batch")]
#[pyo3(signature = (price, volume, kernel=None))]
pub fn vpt_batch_py<'py>(
	py: Python<'py>,
	price: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let price_slice = price.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	// VPT has no parameters, so single row output
	let rows = 1;
	let cols = price_slice.len();

	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	let _combos = py
		.allow_threads(|| {
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};
			vpt_batch_inner_into(price_slice, volume_slice, &VptBatchRange, kernel, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	
	// No parameters for VPT, but include empty list for consistency
	dict.set_item("params", Vec::<f64>::new().into_pyarray(py))?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_js(price: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
	let mut output = vec![0.0; price.len()];
	
	vpt_into_slice(&mut output, price, volume, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_into(
	price_ptr: *const f64,
	volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<(), JsValue> {
	if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let price = std::slice::from_raw_parts(price_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		
		// Check if either input aliases with output
		if price_ptr == out_ptr || volume_ptr == out_ptr {
			// Need temporary buffer for aliasing
			let mut temp = vec![0.0; len];
			vpt_into_slice(&mut temp, price, volume, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, write directly
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			vpt_into_slice(out, price, volume, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VptBatchConfig {
	// VPT has no parameters, so empty config
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VptBatchJsOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vpt_batch)]
pub fn vpt_batch_js(price: &[f64], volume: &[f64], _config: JsValue) -> Result<JsValue, JsValue> {
	// VPT has no parameters, so batch returns single row
	let output = vpt_batch_with_kernel(price, volume, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = VptBatchJsOutput {
		values: output.values,
		rows: 1,
		cols: price.len(),
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_batch_into(
	price_ptr: *const f64,
	volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<usize, JsValue> {
	if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let price = std::slice::from_raw_parts(price_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		let out = std::slice::from_raw_parts_mut(out_ptr, len);
		
		// VPT has no parameters, so just compute once
		vpt_batch_inner_into(price, volume, &VptBatchRange, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		// Return number of parameter combinations (always 1 for VPT)
		Ok(1)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_vpt_basic_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = VptInput::from_candles(&candles, "close");
		let output = vpt_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_vpt_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price = [1.0, 1.1, 1.05, 1.2, 1.3];
		let volume = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
		let input = VptInput::from_slices(&price, &volume);
		let output = vpt_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), price.len());
		Ok(())
	}

	fn check_vpt_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price = [100.0];
		let volume = [500.0];
		let input = VptInput::from_slices(&price, &volume);
		let result = vpt_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_vpt_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price: [f64; 0] = [];
		let volume: [f64; 0] = [];
		let input = VptInput::from_slices(&price, &volume);
		let result = vpt_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_vpt_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price = [f64::NAN, f64::NAN, f64::NAN];
		let volume = [f64::NAN, f64::NAN, f64::NAN];
		let input = VptInput::from_slices(&price, &volume);
		let result = vpt_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_vpt_accuracy_from_csv(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = VptInput::from_candles(&candles, "close");
		let output = vpt_with_kernel(&input, kernel)?;

		let expected_last_five = [
			-0.40358334248536065,
			-0.16292768139917702,
			-0.4792942916867958,
			-0.1188231211518107,
			-3.3492674990910025,
		];

		assert!(output.values.len() >= 5);
		let start_index = output.values.len() - 5;
		for (i, &value) in output.values[start_index..].iter().enumerate() {
			let expected_value = expected_last_five[i];
			assert!(
				(value - expected_value).abs() < 1e-3,
				"VPT mismatch at final bars, index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		Ok(())
	}

	macro_rules! generate_all_vpt_tests {
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
            }
        }
    }

	generate_all_vpt_tests!(
		check_vpt_basic_candles,
		check_vpt_basic_slices,
		check_vpt_not_enough_data,
		check_vpt_empty_data,
		check_vpt_all_nan,
		check_vpt_accuracy_from_csv
	);
}
