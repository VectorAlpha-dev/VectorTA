//! # On Balance Volume (OBV)
//!
//! OBV is a cumulative volume-based indicator. Adds or subtracts volume based on price movement direction.
//!
//! ## Parameters
//! - *(none)*
//!
//! ## Errors
//! - **EmptyData**: obv: Input data slice is empty.
//! - **DataLengthMismatch**: obv: Mismatch in data lengths (close vs. volume).
//! - **AllValuesNaN**: obv: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(ObvOutput)`** on success, containing a `Vec<f64>` of length matching input.
//! - **`Err(ObvError)`** otherwise.

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
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// Input data enum
#[derive(Debug, Clone)]
pub enum ObvData<'a> {
	Candles { candles: &'a Candles },
	Slices { close: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct ObvOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ObvParams;

#[derive(Debug, Clone)]
pub struct ObvInput<'a> {
	pub data: ObvData<'a>,
	pub params: ObvParams,
}

impl<'a> ObvInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: ObvParams) -> Self {
		Self {
			data: ObvData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: ObvParams) -> Self {
		Self {
			data: ObvData::Slices { close, volume },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, ObvParams::default())
	}
}

#[derive(Copy, Clone, Debug)]
pub struct ObvBuilder {
	kernel: Kernel,
}

impl Default for ObvBuilder {
	fn default() -> Self {
		Self { kernel: Kernel::Auto }
	}
}

impl ObvBuilder {
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
	pub fn apply(self, candles: &Candles) -> Result<ObvOutput, ObvError> {
		let i = ObvInput::from_candles(candles, ObvParams::default());
		obv_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<ObvOutput, ObvError> {
		let i = ObvInput::from_slices(close, volume, ObvParams::default());
		obv_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> ObvStream {
		ObvStream::new()
	}
}

// Error type
#[derive(Debug, Error)]
pub enum ObvError {
	#[error("obv: Empty data provided.")]
	EmptyData,
	#[error("obv: Data length mismatch: close_len = {close_len}, volume_len = {volume_len}")]
	DataLengthMismatch { close_len: usize, volume_len: usize },
	#[error("obv: All values are NaN.")]
	AllValuesNaN,
}

impl From<Box<dyn std::error::Error>> for ObvError {
	fn from(_: Box<dyn std::error::Error>) -> Self {
		ObvError::EmptyData
	}
}

#[inline]
pub fn obv(input: &ObvInput) -> Result<ObvOutput, ObvError> {
	obv_with_kernel(input, Kernel::Auto)
}

pub fn obv_with_kernel(input: &ObvInput, kernel: Kernel) -> Result<ObvOutput, ObvError> {
	let (close, volume) = match &input.data {
		ObvData::Candles { candles } => {
			let close = source_type(candles, "close");
			let volume = source_type(candles, "volume");
			(close, volume)
		}
		ObvData::Slices { close, volume } => (*close, *volume),
	};

	if close.is_empty() || volume.is_empty() {
		return Err(ObvError::EmptyData);
	}
	if close.len() != volume.len() {
		return Err(ObvError::DataLengthMismatch {
			close_len: close.len(),
			volume_len: volume.len(),
		});
	}
	let first = close
		.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.ok_or(ObvError::AllValuesNaN)?;

	let mut out = alloc_with_nan_prefix(close.len(), first);
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => obv_scalar(close, volume, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => obv_avx2(close, volume, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => obv_avx512(close, volume, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(ObvOutput { values: out })
}

#[inline]
pub fn obv_scalar(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	// 1) start OBV at zero on the first valid bar:
	let mut prev_obv = 0.0;
	let mut prev_close = close[first_valid];
	out[first_valid] = 0.0;

	// 2) accumulate ±volume thereafter
	for i in (first_valid + 1)..close.len() {
		if close[i] > prev_close {
			prev_obv += volume[i];
		} else if close[i] < prev_close {
			prev_obv -= volume[i];
		}
		out[i] = prev_obv;
		prev_close = close[i];
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx2(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	obv_scalar(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	obv_scalar(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512_short(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	obv_scalar(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512_long(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	obv_scalar(close, volume, first_valid, out)
}

#[inline(always)]
pub unsafe fn obv_row_scalar(
	close: &[f64],
	volume: &[f64],
	first: usize,
	_period: usize,
	_stride: usize,
	_w_ptr: *const f64,
	_inv_n: f64,
	out: &mut [f64],
) {
	obv_scalar(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx2(
	close: &[f64],
	volume: &[f64],
	first: usize,
	_period: usize,
	_stride: usize,
	_w_ptr: *const f64,
	_inv_n: f64,
	out: &mut [f64],
) {
	obv_scalar(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512(
	close: &[f64],
	volume: &[f64],
	first: usize,
	_period: usize,
	_stride: usize,
	_w_ptr: *const f64,
	_inv_n: f64,
	out: &mut [f64],
) {
	obv_scalar(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512_short(
	close: &[f64],
	volume: &[f64],
	first: usize,
	_period: usize,
	_stride: usize,
	_w_ptr: *const f64,
	_inv_n: f64,
	out: &mut [f64],
) {
	obv_scalar(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512_long(
	close: &[f64],
	volume: &[f64],
	first: usize,
	_period: usize,
	_stride: usize,
	_w_ptr: *const f64,
	_inv_n: f64,
	out: &mut [f64],
) {
	obv_scalar(close, volume, first, out)
}

#[derive(Clone, Debug)]
pub struct ObvStream {
	prev_close: Option<f64>,
	prev_obv: Option<f64>,
	initialized: bool,
}

impl ObvStream {
	pub fn new() -> Self {
		Self {
			prev_close: None,
			prev_obv: None,
			initialized: false,
		}
	}
	#[inline(always)]
	pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
		if !self.initialized && !close.is_nan() && !volume.is_nan() {
			self.prev_close = Some(close);
			self.prev_obv = Some(0.0);  // OBV starts at 0, not first volume
			self.initialized = true;
			return Some(0.0);
		}
		if !self.initialized {
			return None;
		}
		let mut obv = self.prev_obv.unwrap();
		let prev = self.prev_close.unwrap();
		if close > prev {
			obv += volume;
		} else if close < prev {
			obv -= volume;
		}
		self.prev_obv = Some(obv);
		self.prev_close = Some(close);
		Some(obv)
	}
}

#[derive(Clone, Debug)]
pub struct ObvBatchRange {
	pub reserved: usize,
}

impl Default for ObvBatchRange {
	fn default() -> Self {
		Self { reserved: 1 }
	}
}

#[derive(Clone, Debug, Default)]
pub struct ObvBatchBuilder {
	kernel: Kernel,
}

impl ObvBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<ObvBatchOutput, ObvError> {
		obv_batch_with_kernel(close, volume, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<ObvBatchOutput, ObvError> {
		let close = source_type(c, "close");
		let volume = source_type(c, "volume");
		self.apply_slices(close, volume)
	}
	pub fn with_default_candles(c: &Candles) -> Result<ObvBatchOutput, ObvError> {
		ObvBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub struct ObvBatchOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

pub fn obv_batch_with_kernel(close: &[f64], volume: &[f64], kernel: Kernel) -> Result<ObvBatchOutput, ObvError> {
	let chosen = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => Kernel::ScalarBatch,
	};
	obv_batch_par_slice(close, volume, chosen)
}

#[inline(always)]
pub fn obv_batch_slice(close: &[f64], volume: &[f64], kern: Kernel) -> Result<ObvBatchOutput, ObvError> {
	obv_batch_inner(close, volume, kern, false)
}

#[inline(always)]
pub fn obv_batch_par_slice(close: &[f64], volume: &[f64], kern: Kernel) -> Result<ObvBatchOutput, ObvError> {
	obv_batch_inner(close, volume, kern, true)
}

#[inline(always)]
fn obv_batch_inner(close: &[f64], volume: &[f64], kern: Kernel, _parallel: bool) -> Result<ObvBatchOutput, ObvError> {
	if close.is_empty() || volume.is_empty() {
		return Err(ObvError::EmptyData);
	}
	if close.len() != volume.len() {
		return Err(ObvError::DataLengthMismatch {
			close_len: close.len(),
			volume_len: volume.len(),
		});
	}
	let first = close
		.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.ok_or(ObvError::AllValuesNaN)?;

	let mut out = alloc_with_nan_prefix(close.len(), first);
	unsafe {
		match kern {
			Kernel::ScalarBatch | Kernel::Scalar => {
				obv_row_scalar(close, volume, first, 0, 0, std::ptr::null(), 0.0, &mut out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2Batch | Kernel::Avx2 => obv_row_avx2(close, volume, first, 0, 0, std::ptr::null(), 0.0, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512Batch | Kernel::Avx512 => obv_row_avx512(close, volume, first, 0, 0, std::ptr::null(), 0.0, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(ObvBatchOutput {
		values: out,
		rows: 1,
		cols: close.len(),
	})
}

#[inline(always)]
fn obv_batch_inner_into(
	close: &[f64],
	volume: &[f64],
	kern: Kernel,
	out: &mut [f64],
) -> Result<(), ObvError> {
	if close.is_empty() || volume.is_empty() {
		return Err(ObvError::EmptyData);
	}
	if close.len() != volume.len() {
		return Err(ObvError::DataLengthMismatch {
			close_len: close.len(),
			volume_len: volume.len(),
		});
	}
	let first = close
		.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.ok_or(ObvError::AllValuesNaN)?;

	unsafe {
		match kern {
			Kernel::ScalarBatch | Kernel::Scalar => {
				obv_row_scalar(close, volume, first, 0, 0, std::ptr::null(), 0.0, out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2Batch | Kernel::Avx2 => {
				obv_row_avx2(close, volume, first, 0, 0, std::ptr::null(), 0.0, out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512Batch | Kernel::Avx512 => {
				obv_row_avx512(close, volume, first, 0, 0, std::ptr::null(), 0.0, out)
			}
			_ => unreachable!(),
		}
	}
	Ok(())
}

#[inline(always)]
fn expand_grid(_r: &ObvBatchRange) -> Vec<ObvParams> {
	vec![ObvParams::default()]
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	fn check_obv_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close: [f64; 0] = [];
		let volume: [f64; 0] = [];
		let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
		let result = obv_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for empty data");
		Ok(())
	}
	fn check_obv_data_length_mismatch(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close = [1.0, 2.0, 3.0];
		let volume = [100.0, 200.0];
		let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
		let result = obv_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for mismatched data length");
		Ok(())
	}
	fn check_obv_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close = [f64::NAN, f64::NAN];
		let volume = [f64::NAN, f64::NAN];
		let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
		let result = obv_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for all NaN data");
		Ok(())
	}
	fn check_obv_csv_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close = source_type(&candles, "close");
		let volume = source_type(&candles, "volume");
		let input = ObvInput::from_candles(&candles, ObvParams::default());
		let obv_result = obv_with_kernel(&input, kernel)?;
		assert_eq!(obv_result.values.len(), close.len());
		let last_five_expected = [
			-329661.6180239202,
			-329767.87639284023,
			-329889.94421654026,
			-329801.35075036023,
			-330218.2007503602,
		];
		let start_idx = obv_result.values.len() - 5;
		let result_tail = &obv_result.values[start_idx..];
		for (i, &val) in result_tail.iter().enumerate() {
			let exp_val = last_five_expected[i];
			let diff = (val - exp_val).abs();
			assert!(
				diff < 1e-6,
				"OBV mismatch at tail index {}: expected {}, got {}",
				i,
				exp_val,
				val
			);
		}
		Ok(())
	}

	macro_rules! generate_all_obv_tests {
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

	generate_all_obv_tests!(
		check_obv_empty_data,
		check_obv_data_length_mismatch,
		check_obv_all_nan,
		check_obv_csv_accuracy
	);
}

/// Helper function for WASM bindings - writes directly to output slice with zero allocations
pub fn obv_into_slice(dst: &mut [f64], close: &[f64], volume: &[f64], kern: Kernel) -> Result<(), ObvError> {
	if close.is_empty() || volume.is_empty() {
		return Err(ObvError::EmptyData);
	}
	if close.len() != volume.len() {
		return Err(ObvError::DataLengthMismatch {
			close_len: close.len(),
			volume_len: volume.len(),
		});
	}
	if dst.len() != close.len() {
		return Err(ObvError::DataLengthMismatch {
			close_len: close.len(),
			volume_len: dst.len(),
		});
	}
	
	let first = close
		.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.ok_or(ObvError::AllValuesNaN)?;
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	// Fill NaN prefix
	for v in &mut dst[..first] {
		*v = f64::NAN;
	}
	
	// Compute OBV directly into destination
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => obv_scalar(close, volume, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => obv_avx2(close, volume, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => obv_avx512(close, volume, first, dst),
			_ => unreachable!(),
		}
	}
	
	Ok(())
}

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "obv")]
#[pyo3(signature = (close, volume, kernel=None))]
pub fn obv_py<'py>(
	py: Python<'py>,
	close: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let input = ObvInput::from_slices(close_slice, volume_slice, ObvParams::default());
	
	let result_vec: Vec<f64> = py
		.allow_threads(|| obv_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ObvStream")]
pub struct ObvStreamPy {
	stream: ObvStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ObvStreamPy {
	#[new]
	pub fn new() -> PyResult<Self> {
		Ok(ObvStreamPy {
			stream: ObvStream::new(),
		})
	}
	
	pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
		self.stream.update(close, volume)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "obv_batch")]
#[pyo3(signature = (close, volume, kernel=None))]
pub fn obv_batch_py<'py>(
	py: Python<'py>,
	close: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	// OBV has no parameters, so batch is just single calculation
	let rows = 1;
	let cols = close_slice.len();
	
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	
	py.allow_threads(|| {
		let kernel = match kern {
			Kernel::Auto => detect_best_batch_kernel(),
			k => k,
		};
		
		obv_batch_inner_into(close_slice, volume_slice, kernel, slice_out)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	// No parameters to return for OBV
	
	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_js(close: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
	let mut output = vec![0.0; close.len()];
	
	obv_into_slice(&mut output, close, volume, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_into(
	close_ptr: *const f64,
	volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<(), JsValue> {
	if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer passed to obv_into"));
	}
	
	unsafe {
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		
		// Check for aliasing - OBV can't operate in-place on either input
		if close_ptr == out_ptr || volume_ptr == out_ptr {
			// Need temporary buffer if output aliases with either input
			let mut temp = vec![0.0; len];
			obv_into_slice(&mut temp, close, volume, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			obv_into_slice(out, close, volume, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ObvBatchJsOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = obv_batch)]
pub fn obv_batch_js(close: &[f64], volume: &[f64]) -> Result<JsValue, JsValue> {
	// OBV has no parameters, so batch returns single row
	let mut output = vec![0.0; close.len()];
	
	obv_into_slice(&mut output, close, volume, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = ObvBatchJsOutput {
		values: output,
		rows: 1,
		cols: close.len(),
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
