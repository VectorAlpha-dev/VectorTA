//! # Negative Volume Index (NVI)
//!
//! The NVI (Negative Volume Index) focuses on days when the volume decreases from the previous day.
//! This implementation follows Tulip Indicators and does not take any parameters.
//!
//! ## Errors
//! - **EmptyData**: nvi: Input data slice(s) is empty.
//! - **AllCloseValuesNaN**: nvi: All close input values are `NaN`.
//! - **AllVolumeValuesNaN**: nvi: All volume input values are `NaN`.
//! - **NotEnoughValidData**: nvi: Fewer than 2 valid (non-`NaN`) data points after the first valid index.
//!
//! ## Returns
//! - **`Ok(NviOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(NviError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum NviData<'a> {
	Candles {
		candles: &'a Candles,
		close_source: &'a str,
	},
	Slices {
		close: &'a [f64],
		volume: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct NviOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct NviParams; // no params

#[derive(Debug, Clone)]
pub struct NviInput<'a> {
	pub data: NviData<'a>,
	pub params: NviParams,
}

impl<'a> NviInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, close_source: &'a str, params: NviParams) -> Self {
		Self {
			data: NviData::Candles { candles, close_source },
			params,
		}
	}
	#[inline]
	pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: NviParams) -> Self {
		Self {
			data: NviData::Slices { close, volume },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "close", NviParams)
	}
}

#[derive(Debug, Error)]
pub enum NviError {
	#[error("nvi: Empty data provided.")]
	EmptyData,
	#[error("nvi: All close values are NaN.")]
	AllCloseValuesNaN,
	#[error("nvi: All volume values are NaN.")]
	AllVolumeValuesNaN,
	#[error("nvi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[derive(Copy, Clone, Debug, Default)]
pub struct NviBuilder {
	kernel: Kernel,
}
impl NviBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<NviOutput, NviError> {
		let i = NviInput::with_default_candles(c);
		nvi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, close: &[f64], volume: &[f64]) -> Result<NviOutput, NviError> {
		let i = NviInput::from_slices(close, volume, NviParams);
		nvi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<NviStream, NviError> {
		NviStream::try_new()
	}
}

#[derive(Debug, Clone)]
pub struct NviStream {
	prev_close: Option<f64>,
	prev_volume: Option<f64>,
	nvi_val: f64,
	started: bool,
}
impl NviStream {
	pub fn try_new() -> Result<Self, NviError> {
		Ok(Self {
			prev_close: None,
			prev_volume: None,
			nvi_val: 1000.0,
			started: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
		if !self.started && !close.is_nan() && !volume.is_nan() {
			self.prev_close = Some(close);
			self.prev_volume = Some(volume);
			self.started = true;
			return Some(self.nvi_val);
		}
		if !self.started {
			return None;
		}
		let prev_c = self.prev_close?;
		let prev_v = self.prev_volume?;
		let mut new_nvi = self.nvi_val;
		if volume < prev_v {
			let pct = (close - prev_c) / prev_c;
			new_nvi += new_nvi * pct;
		}
		self.nvi_val = new_nvi;
		self.prev_close = Some(close);
		self.prev_volume = Some(volume);
		Some(self.nvi_val)
	}
}

#[inline]
pub fn nvi(input: &NviInput) -> Result<NviOutput, NviError> {
	nvi_with_kernel(input, Kernel::Auto)
}
pub fn nvi_with_kernel(input: &NviInput, kernel: Kernel) -> Result<NviOutput, NviError> {
	let (close, volume): (&[f64], &[f64]) = match &input.data {
		NviData::Candles { candles, close_source } => {
			let close = source_type(candles, close_source);
			let volume = candles.select_candle_field("volume").map_err(|_| NviError::EmptyData)?;
			(close, volume)
		}
		NviData::Slices { close, volume } => (*close, *volume),
	};

	if close.is_empty() || volume.is_empty() {
		return Err(NviError::EmptyData);
	}
	let first_valid_idx = close
		.iter()
		.zip(volume.iter())
		.position(|(&c, &v)| !c.is_nan() && !v.is_nan())
		.ok_or_else(|| {
			if close.iter().all(|&c| c.is_nan()) {
				NviError::AllCloseValuesNaN
			} else {
				NviError::AllVolumeValuesNaN
			}
		})?;
	if (close.len() - first_valid_idx) < 2 {
		return Err(NviError::NotEnoughValidData {
			needed: 2,
			valid: close.len() - first_valid_idx,
		});
	}
	let mut out = alloc_with_nan_prefix(close.len(), first_valid_idx);
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => nvi_scalar(close, volume, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => nvi_avx2(close, volume, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => nvi_avx512(close, volume, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(NviOutput { values: out })
}

#[inline]
pub fn nvi_into_slice(
	dst: &mut [f64],
	close: &[f64],
	volume: &[f64],
	kern: Kernel,
) -> Result<(), NviError> {
	if close.is_empty() || volume.is_empty() {
		return Err(NviError::EmptyData);
	}
	
	if dst.len() != close.len() || dst.len() != volume.len() {
		return Err(NviError::NotEnoughValidData {
			needed: close.len(),
			valid: dst.len(),
		});
	}
	
	let first_valid_idx = close
		.iter()
		.zip(volume.iter())
		.position(|(&c, &v)| !c.is_nan() && !v.is_nan())
		.ok_or_else(|| {
			if close.iter().all(|&c| c.is_nan()) {
				NviError::AllCloseValuesNaN
			} else {
				NviError::AllVolumeValuesNaN
			}
		})?;
		
	if (close.len() - first_valid_idx) < 2 {
		return Err(NviError::NotEnoughValidData {
			needed: 2,
			valid: close.len() - first_valid_idx,
		});
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => nvi_scalar(close, volume, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => nvi_avx2(close, volume, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => nvi_avx512(close, volume, first_valid_idx, dst),
			_ => unreachable!(),
		}
	}
	
	// Fill warmup period with NaN
	for v in &mut dst[..first_valid_idx] {
		*v = f64::NAN;
	}
	
	Ok(())
}

pub fn nvi_scalar(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	assert!(
		close.len() == volume.len() && volume.len() == out.len(),
		"Input slices must all have the same length."
	);
	let len = close.len();
	if len == 0 || first_valid >= len {
		return;
	}

	// Start NVI at first valid index
	let mut nvi_val = 1000.0;
	if first_valid < len {
		out[first_valid] = nvi_val;
	}
	
	if first_valid + 1 >= len {
		return;
	}

	// Track previous day's close & volume for bar-to-bar comparison
	let mut prev_close = close[first_valid];
	let mut prev_volume = volume[first_valid];

	// For each subsequent bar
	for i in (first_valid + 1)..len {
		// 3a) Only update when volume has decreased from the prior bar
		if volume[i] < prev_volume {
			// Percentage change in price from yesterday --> today
			let pct_change = (close[i] - prev_close) / prev_close; // :contentReference[oaicite:4]{index=4}
														  // Apply Fosback formula: NVI_t = NVI_{t-1} + (pct_change)*NVI_{t-1}
			nvi_val += nvi_val * pct_change; // :contentReference[oaicite:5]{index=5}
		}
		// 3b) Otherwise, carry forward the same NVI
		out[i] = nvi_val;

		// 3c) Update “previous bar” references
		prev_close = close[i];
		prev_volume = volume[i];
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx2(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	nvi_scalar(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	nvi_scalar(close, volume, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512_short(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
	nvi_scalar(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512_long(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
	nvi_scalar(close, volume, first, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_nvi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = NviInput::with_default_candles(&candles);
		let output = nvi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_nvi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = NviInput::with_default_candles(&candles);
		let result = nvi_with_kernel(&input, kernel)?;
		let expected_last_five = [
			154243.6925373456,
			153973.11239019397,
			153973.11239019397,
			154275.63921207888,
			154275.63921207888,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-5,
				"[{}] NVI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_nvi_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close_data: [f64; 0] = [];
		let volume_data: [f64; 0] = [];
		let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
		let res = nvi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] NVI should fail with empty data", test_name);
		Ok(())
	}

	fn check_nvi_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close_data = [f64::NAN, 100.0];
		let volume_data = [f64::NAN, 120.0];
		let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
		let res = nvi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] NVI should fail with not enough valid data",
			test_name
		);
		Ok(())
	}

	fn check_nvi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close = candles.select_candle_field("close")?;
		let volume = candles.select_candle_field("volume")?;
		let input = NviInput::from_slices(close, volume, NviParams);
		let batch_output = nvi_with_kernel(&input, kernel)?.values;
		let mut stream = NviStream::try_new()?;
		
		// Find first valid index for proper comparison
		let first_valid = close
			.iter()
			.zip(volume.iter())
			.position(|(&c, &v)| !c.is_nan() && !v.is_nan())
			.unwrap_or(0);
		
		// Use alloc_with_nan_prefix for zero-copy allocation
		let mut stream_values = alloc_with_nan_prefix(close.len(), first_valid);
		
		for (i, (&c, &v)) in close.iter().zip(volume.iter()).enumerate() {
			if let Some(nvi_val) = stream.update(c, v) {
				stream_values[i] = nvi_val;
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
				"[{}] NVI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_nvi_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }

	generate_all_nvi_tests!(
		check_nvi_partial_params,
		check_nvi_accuracy,
		check_nvi_empty_data,
		check_nvi_not_enough_valid_data,
		check_nvi_streaming
	);
}

#[cfg(feature = "python")]
#[pyclass(name = "NviStream")]
pub struct NviStreamPy {
	stream: NviStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NviStreamPy {
	#[new]
	fn new() -> PyResult<Self> {
		let stream = NviStream::try_new()
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(NviStreamPy { stream })
	}

	fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
		self.stream.update(close, volume)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "nvi")]
#[pyo3(signature = (close, volume, kernel=None))]
pub fn nvi_py<'py>(
	py: Python<'py>,
	close: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let input = NviInput::from_slices(close_slice, volume_slice, NviParams);
	
	let result_vec: Vec<f64> = py.allow_threads(|| {
		nvi_with_kernel(&input, kern)
			.map(|o| o.values)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_js(close: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
	let mut output = vec![0.0; close.len()];
	
	nvi_into_slice(&mut output, close, volume, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_into(
	close_ptr: *const f64,
	volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<(), JsValue> {
	if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		
		// Check for aliasing with either input pointer
		if close_ptr == out_ptr as *const f64 || volume_ptr == out_ptr as *const f64 {
			// Handle aliasing by using a temporary buffer
			let mut temp = vec![0.0; len];
			nvi_into_slice(&mut temp, close, volume, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			nvi_into_slice(out, close, volume, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}
