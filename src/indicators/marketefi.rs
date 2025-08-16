//! # Market Facilitation Index (marketefi)
//!
//! Market Facilitation Index (marketefi) measures price movement efficiency relative to trading volume.
//!
//! ## Parameters
//! - No adjustable parameters; calculation is direct.
//!
//! ## Errors
//! - **EmptyData**: marketefi: Input data slice is empty.
//! - **MismatchedDataLength**: marketefi: `high`, `low`, and `volume` slices do not have the same length.
//! - **AllValuesNaN**: marketefi: All input data values are `NaN`.
//! - **NotEnoughValidData**: marketefi: No calculable values remain after the first valid index.
//! - **ZeroOrNaNVolume**: marketefi: Volume is zero or NaN at a valid index.
//!
//! ## Returns
//! - **`Ok(MarketefiOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the first valid index.
//! - **`Err(MarketefiError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum MarketefiData<'a> {
	Candles {
		candles: &'a Candles,
		source_high: &'a str,
		source_low: &'a str,
		source_volume: &'a str,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		volume: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct MarketefiParams;

impl Default for MarketefiParams {
	fn default() -> Self {
		Self
	}
}

#[derive(Debug, Clone)]
pub struct MarketefiInput<'a> {
	pub data: MarketefiData<'a>,
	pub params: MarketefiParams,
}

impl<'a> MarketefiInput<'a> {
	#[inline]
	pub fn from_candles(
		candles: &'a Candles,
		source_high: &'a str,
		source_low: &'a str,
		source_volume: &'a str,
		params: MarketefiParams,
	) -> Self {
		Self {
			data: MarketefiData::Candles {
				candles,
				source_high,
				source_low,
				source_volume,
			},
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], volume: &'a [f64], params: MarketefiParams) -> Self {
		Self {
			data: MarketefiData::Slices { high, low, volume },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "high", "low", "volume", MarketefiParams::default())
	}
}

#[derive(Debug, Clone)]
pub struct MarketefiOutput {
	pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MarketefiBuilder {
	kernel: Kernel,
}

impl MarketefiBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<MarketefiOutput, MarketefiError> {
		let i = MarketefiInput::with_default_candles(c);
		marketefi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], volume: &[f64]) -> Result<MarketefiOutput, MarketefiError> {
		let i = MarketefiInput::from_slices(high, low, volume, MarketefiParams::default());
		marketefi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> MarketefiStream {
		MarketefiStream::new()
	}
}

#[derive(Debug, Error)]
pub enum MarketefiError {
	#[error("marketefi: Empty data provided.")]
	EmptyData,
	#[error("marketefi: Mismatched data length among high, low, and volume.")]
	MismatchedDataLength,
	#[error("marketefi: All values are NaN.")]
	AllValuesNaN,
	#[error("marketefi: Not enough valid data to calculate.")]
	NotEnoughValidData,
	#[error("marketefi: Zero or NaN volume at a valid index.")]
	ZeroOrNaNVolume,
}

#[cfg(feature = "wasm")]
impl From<MarketefiError> for JsValue {
	fn from(err: MarketefiError) -> Self {
		JsValue::from_str(&err.to_string())
	}
}

#[inline]
pub fn marketefi(input: &MarketefiInput) -> Result<MarketefiOutput, MarketefiError> {
	marketefi_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn marketefi_into_slice(
	dst: &mut [f64],
	input: &MarketefiInput,
	kern: Kernel,
) -> Result<(), MarketefiError> {
	let (high, low, volume) = match &input.data {
		MarketefiData::Candles {
			candles,
			source_high,
			source_low,
			source_volume,
		} => (
			source_type(candles, source_high),
			source_type(candles, source_low),
			source_type(candles, source_volume),
		),
		MarketefiData::Slices { high, low, volume } => (*high, *low, *volume),
	};

	if high.is_empty() || low.is_empty() || volume.is_empty() {
		return Err(MarketefiError::EmptyData);
	}
	if high.len() != low.len() || low.len() != volume.len() {
		return Err(MarketefiError::MismatchedDataLength);
	}
	if dst.len() != high.len() {
		return Err(MarketefiError::MismatchedDataLength);
	}

	let len = high.len();
	let first = (0..len)
		.find(|&i| {
			let h = high[i];
			let l = low[i];
			let v = volume[i];
			!(h.is_nan() || l.is_nan() || v.is_nan())
		})
		.ok_or(MarketefiError::AllValuesNaN)?;


	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => marketefi_scalar(high, low, volume, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => marketefi_avx2(high, low, volume, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => marketefi_avx512(high, low, volume, first, dst),
			_ => unreachable!(),
		}
	}

	let valid_count = dst[first..].iter().filter(|v| !v.is_nan()).count();
	if valid_count == 0 {
		return Err(MarketefiError::NotEnoughValidData);
	}
	if dst[first..].iter().all(|&val| val.is_nan()) {
		return Err(MarketefiError::ZeroOrNaNVolume);
	}

	Ok(())
}

pub fn marketefi_with_kernel(input: &MarketefiInput, kernel: Kernel) -> Result<MarketefiOutput, MarketefiError> {
	let (high, low, volume) = match &input.data {
		MarketefiData::Candles {
			candles,
			source_high,
			source_low,
			source_volume,
		} => (
			source_type(candles, source_high),
			source_type(candles, source_low),
			source_type(candles, source_volume),
		),
		MarketefiData::Slices { high, low, volume } => (*high, *low, *volume),
	};

	if high.is_empty() || low.is_empty() || volume.is_empty() {
		return Err(MarketefiError::EmptyData);
	}
	if high.len() != low.len() || low.len() != volume.len() {
		return Err(MarketefiError::MismatchedDataLength);
	}
	let len = high.len();
	let first = (0..len)
		.find(|&i| {
			let h = high[i];
			let l = low[i];
			let v = volume[i];
			!(h.is_nan() || l.is_nan() || v.is_nan())
		})
		.ok_or(MarketefiError::AllValuesNaN)?;
	let mut out = alloc_with_nan_prefix(len, first);
	// Fill remaining values with NaN for binding compatibility
	for i in first..out.len() {
		out[i] = f64::NAN;
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => marketefi_scalar(high, low, volume, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => marketefi_avx2(high, low, volume, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => marketefi_avx512(high, low, volume, first, &mut out),
			_ => unreachable!(),
		}
	}
	let valid_count = out[first..].iter().filter(|v| !v.is_nan()).count();
	if valid_count == 0 {
		return Err(MarketefiError::NotEnoughValidData);
	}
	if out[first..].iter().all(|&val| val.is_nan()) {
		return Err(MarketefiError::ZeroOrNaNVolume);
	}
	Ok(MarketefiOutput { values: out })
}

#[inline]
pub fn marketefi_scalar(high: &[f64], low: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	for i in first_valid..high.len() {
		let h = high[i];
		let l = low[i];
		let v = volume[i];
		if h.is_nan() || l.is_nan() || v.is_nan() || v == 0.0 {
			out[i] = f64::NAN;
		} else {
			out[i] = (h - l) / v;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx512(high: &[f64], low: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	unsafe { marketefi_avx512_short(high, low, volume, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx2(high: &[f64], low: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx512_short(high: &[f64], low: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn marketefi_avx512_long(high: &[f64], low: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first_valid, out)
}

// Row/batch interface

#[inline(always)]
pub fn marketefi_row_scalar(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx2(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx512(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx512_short(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn marketefi_row_avx512_long(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
	marketefi_scalar(high, low, volume, first, out)
}

#[derive(Clone, Debug)]
pub struct MarketefiBatchRange; // No params, just 1 row.

impl Default for MarketefiBatchRange {
	fn default() -> Self {
		Self
	}
}

#[derive(Clone, Debug, Default)]
pub struct MarketefiBatchBuilder {
	kernel: Kernel,
}

impl MarketefiBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		volume: &[f64],
	) -> Result<MarketefiBatchOutput, MarketefiError> {
		marketefi_batch_with_kernel(high, low, volume, self.kernel)
	}
	pub fn with_default_candles(c: &Candles) -> Result<MarketefiBatchOutput, MarketefiError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let volume = source_type(c, "volume");
		MarketefiBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_slices(high, low, volume)
	}
}

pub fn marketefi_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	volume: &[f64],
	kernel: Kernel,
) -> Result<MarketefiBatchOutput, MarketefiError> {
	let k = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		x if x.is_batch() => x,
		_ => Kernel::ScalarBatch,
	};
	marketefi_batch_par_slice(high, low, volume, k)
}

#[derive(Clone, Debug)]
pub struct MarketefiBatchOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

#[inline(always)]
pub fn marketefi_batch_slice(
	high: &[f64],
	low: &[f64],
	volume: &[f64],
	kernel: Kernel,
) -> Result<MarketefiBatchOutput, MarketefiError> {
	marketefi_batch_inner(high, low, volume, kernel, false)
}

#[inline(always)]
pub fn marketefi_batch_par_slice(
	high: &[f64],
	low: &[f64],
	volume: &[f64],
	kernel: Kernel,
) -> Result<MarketefiBatchOutput, MarketefiError> {
	marketefi_batch_inner(high, low, volume, kernel, true)
}

#[inline(always)]
fn marketefi_batch_inner(
	high: &[f64],
	low: &[f64],
	volume: &[f64],
	kernel: Kernel,
	_parallel: bool,
) -> Result<MarketefiBatchOutput, MarketefiError> {
	let len = high.len();
	let first = (0..len)
		.find(|&i| {
			let h = high[i];
			let l = low[i];
			let v = volume[i];
			!(h.is_nan() || l.is_nan() || v.is_nan())
		})
		.ok_or(MarketefiError::AllValuesNaN)?;
	let mut out = alloc_with_nan_prefix(len, first);
	// Fill remaining values with NaN for binding compatibility
	for i in first..out.len() {
		out[i] = f64::NAN;
	}
	unsafe {
		match kernel {
			Kernel::ScalarBatch | Kernel::Scalar => marketefi_row_scalar(high, low, volume, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2Batch | Kernel::Avx2 => marketefi_row_avx2(high, low, volume, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512Batch | Kernel::Avx512 => marketefi_row_avx512(high, low, volume, first, &mut out),
			_ => marketefi_row_scalar(high, low, volume, first, &mut out),
		}
	}
	Ok(MarketefiBatchOutput {
		values: out,
		rows: 1,
		cols: len,
	})
}

#[inline(always)]
pub fn expand_grid(_: &MarketefiBatchRange) -> Vec<MarketefiParams> {
	vec![MarketefiParams]
}

// Streaming (single-point rolling)
#[derive(Debug, Clone)]
pub struct MarketefiStream;

impl MarketefiStream {
	pub fn new() -> Self {
		Self
	}
	pub fn update(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
		if high.is_nan() || low.is_nan() || volume.is_nan() || volume == 0.0 {
			None
		} else {
			Some((high - low) / volume)
		}
	}
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "marketefi")]
#[pyo3(signature = (high, low, volume, kernel=None))]
pub fn marketefi_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	volume: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let input = MarketefiInput::from_slices(high_slice, low_slice, volume_slice, MarketefiParams::default());

	let result_vec: Vec<f64> = py
		.allow_threads(|| marketefi_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MarketefiStream")]
pub struct MarketefiStreamPy {
	stream: MarketefiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MarketefiStreamPy {
	#[new]
	fn new() -> PyResult<Self> {
		Ok(MarketefiStreamPy {
			stream: MarketefiStream::new(),
		})
	}

	fn update(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
		self.stream.update(high, low, volume)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "marketefi_batch")]
#[pyo3(signature = (high, low, volume, kernel=None))]
pub fn marketefi_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	volume: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	// Since marketefi has no parameters, there's only one row
	let result = py
		.allow_threads(|| marketefi_batch_with_kernel(high_slice, low_slice, volume_slice, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", result.values.into_pyarray(py).reshape((result.rows, result.cols))?)?;

	Ok(dict)
}

// Unit tests
#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use paste::paste;
	#[cfg(feature = "proptest")]
	use proptest::prelude::*;

	fn check_marketefi_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MarketefiInput::with_default_candles(&candles);
		let res = marketefi_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		let expected_last_five = [
			2.8460112192104607,
			3.020938522420525,
			3.0474861329079292,
			3.691017115591989,
			2.247810963176202,
		];
		let start = res.values.len() - 5;
		for (i, &v) in res.values[start..].iter().enumerate() {
			let exp = expected_last_five[i];
			assert!(
				(v - exp).abs() < 1e-6,
				"[{}] marketefi mismatch at {}: got {}, exp {}",
				test,
				start + i,
				v,
				exp
			);
		}
		Ok(())
	}

	fn check_marketefi_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high = [f64::NAN, 2.0, 3.0];
		let low = [f64::NAN, 1.0, 2.0];
		let vol = [f64::NAN, 1.0, 1.0];
		let input = MarketefiInput::from_slices(&high, &low, &vol, MarketefiParams::default());
		let res = marketefi_with_kernel(&input, kernel)?;
		assert!(res.values[0].is_nan());
		assert_eq!(res.values[1], 1.0 / 1.0);
		Ok(())
	}

	fn check_marketefi_empty_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let input = MarketefiInput::from_slices(&[], &[], &[], MarketefiParams::default());
		let res = marketefi_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_marketefi_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high = [3.0, 4.0, 5.0];
		let low = [2.0, 3.0, 3.0];
		let vol = [1.0, 2.0, 2.0];
		let mut stream = MarketefiStream::new();
		let mut vals = Vec::new();
		for i in 0..high.len() {
			vals.push(stream.update(high[i], low[i], vol[i]).unwrap_or(f64::NAN));
		}
		let input = MarketefiInput::from_slices(&high, &low, &vol, MarketefiParams::default());
		let res = marketefi_with_kernel(&input, kernel)?;
		for (a, b) in vals.iter().zip(res.values.iter()) {
			if a.is_nan() && b.is_nan() {
				continue;
			}
			assert!((a - b).abs() < 1e-8);
		}
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_marketefi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		// Strategy for generating test data
		// Generate realistic market data with varying scenarios
		let strat = (50usize..400, 0usize..7, any::<u64>())
			.prop_map(|(len, scenario, seed)| {
				// LCG-based deterministic RNG for reproducible tests
				let mut rng_state = seed.wrapping_mul(1664525).wrapping_add(1013904223);
				let mut next_f64 = || {
					rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
					(rng_state as f64) / (u64::MAX as f64)
				};
				
				let mut high = Vec::with_capacity(len);
				let mut low = Vec::with_capacity(len);
				let mut volume = Vec::with_capacity(len);
				
				match scenario {
					0 => {
						// Random realistic market data
						for _ in 0..len {
							let base = 50.0 + next_f64() * 450.0;
							let spread = 0.1 + next_f64() * 10.0;
							high.push(base + spread);
							low.push(base);
							volume.push(100.0 + next_f64() * 10000.0);
						}
					},
					1 => {
						// Constant prices (high = low) - should produce 0.0
						let price = 100.0 + next_f64() * 200.0;
						for _ in 0..len {
							high.push(price);
							low.push(price);
							volume.push(1000.0 + next_f64() * 1000.0);
						}
					},
					2 => {
						// Trending market with increasing volatility
						let mut base = 100.0;
						for i in 0..len {
							let trend = 0.5 * (i as f64 / len as f64);
							base += trend;
							let volatility = 0.5 + (i as f64 / len as f64) * 5.0;
							high.push(base + volatility);
							low.push(base - volatility * 0.5);
							volume.push(500.0 + next_f64() * 5000.0 + i as f64 * 10.0);
						}
					},
					3 => {
						// Small volumes (edge case testing)
						for _ in 0..len {
							let base = 50.0 + next_f64() * 100.0;
							let spread = 0.1 + next_f64() * 5.0;
							high.push(base + spread);
							low.push(base);
							volume.push(0.001 + next_f64() * 1.0); // Very small volumes
						}
					},
					4 => {
						// Large volumes and price ranges
						for _ in 0..len {
							let base = 1000.0 + next_f64() * 9000.0;
							let spread = 10.0 + next_f64() * 100.0;
							high.push(base + spread);
							low.push(base);
							volume.push(1e6 + next_f64() * 1e7); // Large volumes
						}
					},
					5 => {
						// Zero volumes mixed with valid volumes
						for i in 0..len {
							let base = 100.0 + next_f64() * 100.0;
							let spread = 1.0 + next_f64() * 5.0;
							high.push(base + spread);
							low.push(base);
							// Every 5th element has zero volume
							if i % 5 == 0 {
								volume.push(0.0);
							} else {
								volume.push(100.0 + next_f64() * 1000.0);
							}
						}
					},
					_ => {
						// Inverted prices (high < low) - data quality issue but should handle
						for _ in 0..len {
							let base = 100.0 + next_f64() * 200.0;
							let spread = 1.0 + next_f64() * 10.0;
							// Occasionally invert high and low
							if next_f64() < 0.3 {
								high.push(base - spread);  // high is lower
								low.push(base);             // low is higher
							} else {
								high.push(base + spread);
								low.push(base);
							}
							volume.push(500.0 + next_f64() * 5000.0);
						}
					},
				}
				
				(high, low, volume, scenario)
			});
		
		proptest::test_runner::TestRunner::default()
			.run(&strat, |(high, low, volume, scenario)| {
				let input = MarketefiInput::from_slices(&high, &low, &volume, MarketefiParams::default());
				
				// Get output from the kernel being tested
				let output = marketefi_with_kernel(&input, kernel)?;
				
				// Get reference output from scalar kernel
				let ref_output = marketefi_with_kernel(&input, Kernel::Scalar)?;
				
				// Property 1: Output length must match input length
				prop_assert_eq!(output.values.len(), high.len(), 
					"Output length mismatch: got {}, expected {}", output.values.len(), high.len());
				
				// Property 2: First valid index behavior - all values before first valid index should be NaN
				let first_valid = (0..high.len()).find(|&i| {
					!high[i].is_nan() && !low[i].is_nan() && !volume[i].is_nan()
				});
				
				if let Some(first) = first_valid {
					for i in 0..first {
						prop_assert!(output.values[i].is_nan(), 
							"Expected NaN before first valid index {} but got {} at index {}", 
							first, output.values[i], i);
					}
				}
				
				// Property 3: Mathematical accuracy - verify calculation
				for i in 0..high.len() {
					let expected = if high[i].is_nan() || low[i].is_nan() || volume[i].is_nan() || volume[i] == 0.0 {
						f64::NAN
					} else {
						(high[i] - low[i]) / volume[i]
					};
					
					let actual = output.values[i];
					
					if expected.is_nan() {
						prop_assert!(actual.is_nan(), 
							"Expected NaN at index {} but got {}", i, actual);
					} else {
						prop_assert!((actual - expected).abs() < 1e-10, 
							"Calculation mismatch at index {}: expected {}, got {}", i, expected, actual);
					}
				}
				
				// Property 4: Kernel consistency - all kernels must produce identical results
				for i in 0..output.values.len() {
					let out_val = output.values[i];
					let ref_val = ref_output.values[i];
					
					if out_val.is_nan() && ref_val.is_nan() {
						continue;
					}
					
					prop_assert!((out_val - ref_val).abs() < 1e-10,
						"Kernel mismatch at index {}: kernel={}, reference={}", i, out_val, ref_val);
				}
				
				// Property 5: Special case - when high equals low, result should be 0.0 (for non-zero volume)
				if scenario == 1 {
					for i in 0..output.values.len() {
						if !high[i].is_nan() && !low[i].is_nan() && !volume[i].is_nan() && volume[i] != 0.0 {
							prop_assert!((output.values[i] - 0.0).abs() < 1e-10,
								"When high=low, expected 0.0 but got {} at index {}", output.values[i], i);
						}
					}
				}
				
				// Property 6: Zero volume should produce NaN
				if scenario == 5 {
					for i in 0..output.values.len() {
						if volume[i] == 0.0 {
							prop_assert!(output.values[i].is_nan(),
								"Expected NaN for zero volume at index {} but got {}", i, output.values[i]);
						}
					}
				}
				
				// Property 7: Inverted prices (high < low) should produce negative values
				if scenario == 6 {
					for i in 0..output.values.len() {
						if !high[i].is_nan() && !low[i].is_nan() && !volume[i].is_nan() && volume[i] != 0.0 {
							if high[i] < low[i] {
								prop_assert!(output.values[i] < 0.0,
									"Expected negative value when high < low at index {}, but got {}", i, output.values[i]);
							}
						}
					}
				}
				
				// Property 8: Check for poison values
				for (i, &val) in output.values.iter().enumerate() {
					if val.is_nan() {
						continue;
					}
					
					let bits = val.to_bits();
					
					prop_assert!(bits != 0x11111111_11111111,
						"Found alloc_with_nan_prefix poison value at index {}", i);
					prop_assert!(bits != 0x22222222_22222222,
						"Found init_matrix_prefixes poison value at index {}", i);
					prop_assert!(bits != 0x33333333_33333333,
						"Found make_uninit_matrix poison value at index {}", i);
				}
				
				Ok(())
			})?;
		
		Ok(())
	}

	macro_rules! generate_all_marketefi_tests {
        ($($test_fn:ident),*) => {
            paste! {
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
	generate_all_marketefi_tests!(
		check_marketefi_accuracy,
		check_marketefi_nan_handling,
		check_marketefi_empty_data,
		check_marketefi_streaming,
		check_marketefi_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_marketefi_tests!(check_marketefi_property);

	#[test]
	fn test_marketefi_into_slice() -> Result<(), Box<dyn Error>> {
		let high = vec![100.0, 105.0, 110.0, 108.0, 112.0];
		let low = vec![95.0, 98.0, 102.0, 104.0, 106.0];
		let volume = vec![1000.0, 1500.0, 2000.0, 1200.0, 1800.0];
		
		let input = MarketefiInput::from_slices(&high, &low, &volume, MarketefiParams::default());
		
		// Test with pre-allocated buffer
		let mut dst = vec![0.0; high.len()];
		marketefi_into_slice(&mut dst, &input, Kernel::Scalar)?;
		
		// Compare with regular marketefi function
		let output = marketefi(&input)?;
		
		assert_eq!(dst.len(), output.values.len());
		for i in 0..dst.len() {
			if dst[i].is_nan() && output.values[i].is_nan() {
				continue;
			}
			assert!((dst[i] - output.values[i]).abs() < 1e-10, 
				"Mismatch at index {}: into_slice={}, regular={}", i, dst[i], output.values[i]);
		}
		
		// Verify the calculation is correct
		for i in 0..high.len() {
			let expected = (high[i] - low[i]) / volume[i];
			assert!((dst[i] - expected).abs() < 1e-10, 
				"Incorrect calculation at index {}: got={}, expected={}", i, dst[i], expected);
		}
		
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_marketefi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Since marketefi has no parameters, we just test with default params
		let params = MarketefiParams::default();
		let input = MarketefiInput::from_candles(&candles, "high", "low", "volume", params.clone());
		let output = marketefi_with_kernel(&input, kernel)?;
		
		for (i, &val) in output.values.iter().enumerate() {
			if val.is_nan() {
				continue; // NaN values are expected during warmup
			}
			
			let bits = val.to_bits();
			
			// Check all three poison patterns
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}
			
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}
			
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_marketefi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let out = MarketefiBatchBuilder::new().kernel(kernel).apply_slices(
			source_type(&candles, "high"),
			source_type(&candles, "low"),
			source_type(&candles, "volume"),
		)?;
		let expected_last_five = [
			2.8460112192104607,
			3.020938522420525,
			3.0474861329079292,
			3.691017115591989,
			2.247810963176202,
		];
		let row = &out.values;
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			let exp = expected_last_five[i];
			assert!(
				(v - exp).abs() < 1e-8,
				"[{test}] batch row mismatch at {i}: {v} vs {exp}"
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Since marketefi has no parameters, we only have one configuration to test
		let output = MarketefiBatchBuilder::new()
			.kernel(kernel)
			.apply_slices(
				source_type(&c, "high"),
				source_type(&c, "low"), 
				source_type(&c, "volume")
			)?;
		
		for (idx, &val) in output.values.iter().enumerate() {
			if val.is_nan() {
				continue;
			}
			
			let bits = val.to_bits();
			let row = idx / output.cols;
			let col = idx % output.cols;
			
			// Check all three poison patterns with detailed context
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
					at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}
			
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) \
					at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}
			
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) \
					at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! gen_batch_tests {
		($fn_name:ident) => {
			paste! {
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
				}
			}
		};
	}
	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn marketefi_js(high: &[f64], low: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
	let input = MarketefiInput::from_slices(high, low, volume, MarketefiParams::default());
	
	let mut output = vec![0.0; high.len()];
	
	marketefi_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn marketefi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn marketefi_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn marketefi_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to marketefi_into"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		
		let input = MarketefiInput::from_slices(high, low, volume, MarketefiParams::default());
		
		// Check for aliasing - if any input pointer equals output pointer
		if high_ptr == out_ptr || low_ptr == out_ptr || volume_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			marketefi_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			marketefi_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MarketefiBatchConfig {
	// No parameters for marketefi, so empty config
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MarketefiBatchJsOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = marketefi_batch)]
pub fn marketefi_batch_js(
	high: &[f64],
	low: &[f64],
	volume: &[f64],
	_config: JsValue,
) -> Result<JsValue, JsValue> {
	// Since marketefi has no parameters, batch returns single row
	let result = marketefi_batch_with_kernel(high, low, volume, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let output = MarketefiBatchJsOutput {
		values: result.values,
		rows: result.rows,
		cols: result.cols,
	};
	
	serde_wasm_bindgen::to_value(&output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
