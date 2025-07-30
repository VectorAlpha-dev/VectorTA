//! # Balance of Power (BOP)
//!
//! (Close - Open) / (High - Low)
//!
//! If (High - Low) <= 0.0, output is 0.0.
//!
//! ## Parameters
//! None currently required; see `BopParams` for future extensibility.
//!
//! ## Errors
//! - **EmptyData**: bop: No data provided.
//! - **InconsistentLengths**: bop: Input arrays have different lengths.
//!
//! ## Returns
//! - **`Ok(BopOutput)`** on success, containing a `Vec<f64>` with the BOP values.
//! - **`Err(BopError)`** otherwise.
//!
//! ## Example
//! ```
//! use my_project::indicators::bop::{bop, BopInput, BopParams};
//! let open = [1.0, 2.0];
//! let high = [2.0, 3.0];
//! let low = [0.5, 1.0];
//! let close = [1.5, 2.5];
//! let input = BopInput::from_slices(&open, &high, &low, &close, BopParams::default());
//! let out = bop(&input).unwrap();
//! assert!((out.values[0] - 0.5).abs() < 1e-12);
//! ```

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
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum BopData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		open: &'a [f64],
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
	},
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BopParams {}

#[derive(Debug, Clone)]
pub struct BopInput<'a> {
	pub data: BopData<'a>,
	pub params: BopParams,
}

impl<'a> BopInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: BopParams) -> Self {
		Self {
			data: BopData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(open: &'a [f64], high: &'a [f64], low: &'a [f64], close: &'a [f64], params: BopParams) -> Self {
		Self {
			data: BopData::Slices { open, high, low, close },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: BopData::Candles { candles },
			params: BopParams::default(),
		}
	}
}

#[derive(Debug, Clone)]
pub struct BopOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum BopError {
	#[error("bop: Data is empty.")]
	EmptyData,
	#[error("bop: Data is empty.")]
	DataIsEmpty,
	#[error("bop: Inconsistent lengths.")]
	InconsistentLengths,
	#[error("bop: Input lengths mismatch - open: {open_len}, high: {high_len}, low: {low_len}, close: {close_len}")]
	InputLengthsMismatch {
		open_len: usize,
		high_len: usize,
		low_len: usize,
		close_len: usize,
	},
	#[error("bop: Candle field error: {0}")]
	CandleFieldError(String),
}

#[derive(Copy, Clone, Debug)]
pub struct BopBuilder {
	kernel: Kernel,
}

impl Default for BopBuilder {
	fn default() -> Self {
		Self { kernel: Kernel::Auto }
	}
}

impl BopBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<BopOutput, BopError> {
		let i = BopInput::from_candles(c, BopParams::default());
		bop_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Result<BopOutput, BopError> {
		let i = BopInput::from_slices(open, high, low, close, BopParams::default());
		bop_with_kernel(&i, self.kernel)
	}
	/// Create a streaming BOP calculator.
	///
	/// Note: BOP stream is kernel-agnostic; always uses scalar implementation.
	#[inline(always)]
	pub fn into_stream(self) -> Result<BopStream, BopError> {
		BopStream::try_new()
	}
}

#[inline]
pub fn bop(input: &BopInput) -> Result<BopOutput, BopError> {
	bop_with_kernel(input, Kernel::Auto)
}

pub fn bop_with_kernel(input: &BopInput, kernel: Kernel) -> Result<BopOutput, BopError> {
	let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
		BopData::Candles { candles } => {
			let open = candles
				.select_candle_field("open")
				.map_err(|e| BopError::CandleFieldError(e.to_string()))?;
			let high = candles
				.select_candle_field("high")
				.map_err(|e| BopError::CandleFieldError(e.to_string()))?;
			let low = candles
				.select_candle_field("low")
				.map_err(|e| BopError::CandleFieldError(e.to_string()))?;
			let close = candles
				.select_candle_field("close")
				.map_err(|e| BopError::CandleFieldError(e.to_string()))?;
			(open, high, low, close)
		}
		BopData::Slices { open, high, low, close } => (open, high, low, close),
	};

	let len = open.len();
	if len == 0 {
		return Err(BopError::EmptyData);
	}
	if len != high.len() || len != low.len() || len != close.len() {
		return Err(BopError::InconsistentLengths);
	}

	// Calculate the warmup period - first index where all arrays have valid values
	let warmup_period = (0..len)
		.find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.unwrap_or(len);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	// Use zero-copy allocation with NaN prefix
	let mut out = alloc_with_nan_prefix(len, warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => bop_scalar(open, high, low, close, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => bop_avx2(open, high, low, close, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => bop_avx512(open, high, low, close, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(BopOutput { values: out })
}

#[inline]
pub unsafe fn bop_scalar(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	// Find the first valid index where all arrays have non-NaN values
	let first_valid = (0..open.len())
		.find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.unwrap_or(open.len());

	// Only compute values starting from the first valid index
	for i in first_valid..open.len() {
		let denom = high[i] - low[i];
		out[i] = if denom <= 0.0 {
			0.0
		} else {
			(close[i] - open[i]) / denom
		};
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx2(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	bop_scalar(open, high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx512(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	bop_scalar(open, high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx512_short(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	bop_scalar(open, high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx512_long(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	bop_scalar(open, high, low, close, out)
}

#[inline]
pub fn bop_row_scalar(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	unsafe { bop_scalar(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx2(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	unsafe { bop_avx2(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx512(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	unsafe { bop_avx512(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx512_short(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	unsafe { bop_avx512_short(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx512_long(open: &[f64], high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
	unsafe { bop_avx512_long(open, high, low, close, out) }
}

// ---- Batch/Streaming structs and functions ----

#[derive(Clone, Debug)]
pub struct BopStream {
	pub last: Option<f64>,
}

impl BopStream {
	pub fn try_new() -> Result<Self, BopError> {
		Ok(Self { last: None })
	}
	#[inline(always)]
	pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> f64 {
		let denom = high - low;
		let val = if denom <= 0.0 { 0.0 } else { (close - open) / denom };
		self.last = Some(val);
		val
	}
}

// ---- Batch processing API ----

/// Batch parameter range for BOP.
///
/// Note: BOP has no parameters, so this struct is intentionally empty.
/// It exists to maintain API consistency with other indicators.
#[derive(Clone, Debug)]
pub struct BopBatchRange {
	// Intentionally empty - BOP has no parameters to sweep
}

impl Default for BopBatchRange {
	fn default() -> Self {
		Self {}
	}
}

#[derive(Clone, Debug, Default)]
pub struct BopBatchBuilder {
	range: BopBatchRange,
	kernel: Kernel,
}

impl BopBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply_slices(
		self,
		open: &[f64],
		high: &[f64],
		low: &[f64],
		close: &[f64],
	) -> Result<BopBatchOutput, BopError> {
		bop_batch_with_kernel(open, high, low, close, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<BopBatchOutput, BopError> {
		let open = source_type(c, "open");
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		self.apply_slices(open, high, low, close)
	}
	pub fn with_default_candles(c: &Candles) -> Result<BopBatchOutput, BopError> {
		BopBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

#[derive(Clone, Debug)]
pub struct BopBatchOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

pub fn bop_batch_with_kernel(
	open: &[f64],
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kernel: Kernel,
) -> Result<BopBatchOutput, BopError> {
	let len = open.len();
	if len == 0 || high.len() != len || low.len() != len || close.len() != len {
		return Err(BopError::InconsistentLengths);
	}

	// Calculate the warmup period - first index where all arrays have valid values
	let warmup_period = (0..len)
		.find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.unwrap_or(len);

	let chosen = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	};

	// Use zero-copy allocation with NaN prefix
	let mut values = alloc_with_nan_prefix(len, warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => bop_scalar(open, high, low, close, &mut values),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => bop_avx2(open, high, low, close, &mut values),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => bop_avx512(open, high, low, close, &mut values),
			_ => unreachable!(),
		}
	}
	Ok(BopBatchOutput {
		values,
		rows: 1,
		cols: len,
	})
}

/// Internal function that writes directly into a provided output slice for zero-copy Python bindings
#[inline(always)]
fn bop_batch_inner_into(
	open: &[f64],
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kernel: Kernel,
	out: &mut [f64],
) -> Result<(), BopError> {
	let len = open.len();
	if len == 0 || high.len() != len || low.len() != len || close.len() != len {
		return Err(BopError::InconsistentLengths);
	}
	if out.len() != len {
		return Err(BopError::InconsistentLengths);
	}

	// Calculate the warmup period - first index where all arrays have valid values
	let warmup_period = (0..len)
		.find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.unwrap_or(len);

	// Fill the warmup period with NaN
	out[..warmup_period].fill(f64::NAN);

	let chosen = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => bop_scalar(open, high, low, close, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => bop_avx2(open, high, low, close, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => bop_avx512(open, high, low, close, out),
			_ => unreachable!(),
		}
	}
	Ok(())
}

#[inline(always)]
pub fn bop_batch_slice(
	open: &[f64],
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kern: Kernel,
) -> Result<BopBatchOutput, BopError> {
	bop_batch_with_kernel(open, high, low, close, kern)
}

#[inline(always)]
pub fn bop_batch_par_slice(
	open: &[f64],
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kern: Kernel,
) -> Result<BopBatchOutput, BopError> {
	// BOP is cheap, so just run the regular batch; use rayon for real batch ops if needed.
	bop_batch_with_kernel(open, high, low, close, kern)
}

#[inline(always)]
fn expand_grid(_r: &BopBatchRange) -> Vec<BopParams> {
	vec![BopParams {}]
}

// ---- Unit tests ----

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_bop_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = BopInput::with_default_candles(&candles);
		let output = bop_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_bop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = BopInput::with_default_candles(&candles);
		let bop_result = bop_with_kernel(&input, kernel)?;

		let expected_last_five = [
			0.045454545454545456,
			-0.32398753894080995,
			-0.3844086021505376,
			0.3547400611620795,
			-0.5336179295624333,
		];
		let start_index = bop_result.values.len().saturating_sub(5);
		let result_last_five = &bop_result.values[start_index..];
		for (i, &v) in result_last_five.iter().enumerate() {
			assert!(
				(v - expected_last_five[i]).abs() < 1e-10,
				"[{}] BOP mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				v,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_bop_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = BopInput::with_default_candles(&candles);
		match input.data {
			BopData::Candles { .. } => {}
			_ => panic!("Expected BopData::Candles"),
		}
		let output = bop_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_bop_with_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let empty: [f64; 0] = [];
		let params = BopParams::default();
		let input = BopInput::from_slices(&empty, &empty, &empty, &empty, params);
		let result = bop_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] Expected an error for empty data", test_name);
		Ok(())
	}

	fn check_bop_with_inconsistent_lengths(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let open = [1.0, 2.0, 3.0];
		let high = [1.5, 2.5];
		let low = [0.8, 1.8, 2.8];
		let close = [1.2, 2.2, 3.2];
		let params = BopParams::default();
		let input = BopInput::from_slices(&open, &high, &low, &close, params);
		let result = bop_with_kernel(&input, kernel);
		assert!(
			result.is_err(),
			"[{}] Expected an error for inconsistent input lengths",
			test_name
		);
		Ok(())
	}

	fn check_bop_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let open = [10.0];
		let high = [12.0];
		let low = [9.5];
		let close = [11.0];
		let params = BopParams::default();
		let input = BopInput::from_slices(&open, &high, &low, &close, params);
		let result = bop_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), 1);
		assert!((result.values[0] - 0.4).abs() < 1e-10);
		Ok(())
	}

	fn check_bop_with_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_input = BopInput::with_default_candles(&candles);
		let first_result = bop_with_kernel(&first_input, kernel)?;

		let dummy = vec![0.0; first_result.values.len()];
		let second_input = BopInput::from_slices(&dummy, &dummy, &dummy, &first_result.values, BopParams::default());
		let second_result = bop_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		for (i, &val) in second_result.values.iter().enumerate() {
			assert!(
				(val - 0.0).abs() < f64::EPSILON,
				"[{}] Expected BOP=0.0 for dummy data at idx {}, got {}",
				test_name,
				i,
				val
			);
		}
		Ok(())
	}

	fn check_bop_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = BopInput::with_default_candles(&candles);
		let bop_result = bop_with_kernel(&input, kernel)?;
		if bop_result.values.len() > 240 {
			for i in 240..bop_result.values.len() {
				assert!(!bop_result.values[i].is_nan(), "[{}] Found NaN at idx {}", test_name, i);
			}
		}
		Ok(())
	}

	fn check_bop_streaming(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		// streaming BOP is trivial (no state), just check that update matches scalar formula
		let open = [10.0, 5.0, 6.0, 10.0, 11.0];
		let high = [15.0, 6.0, 9.0, 20.0, 13.0];
		let low = [10.0, 5.0, 4.0, 10.0, 11.0];
		let close = [14.0, 6.0, 7.0, 12.0, 12.0];
		let mut s = BopStream::try_new()?;
		for i in 0..open.len() {
			let val = s.update(open[i], high[i], low[i], close[i]);
			let denom = high[i] - low[i];
			let expected = if denom <= 0.0 {
				0.0
			} else {
				(close[i] - open[i]) / denom
			};
			assert!((val - expected).abs() < 1e-12, "stream mismatch");
		}
		Ok(())
	}

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_bop_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with default parameters
		let input = BopInput::with_default_candles(&candles);
		let output = bop_with_kernel(&input, kernel)?;

		// Check every value for poison patterns
		for (i, &val) in output.values.iter().enumerate() {
			// Skip NaN values as they're expected in the warmup period
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_bop_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	macro_rules! generate_all_bop_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

	generate_all_bop_tests!(
		check_bop_partial_params,
		check_bop_accuracy,
		check_bop_default_candles,
		check_bop_with_empty_data,
		check_bop_with_inconsistent_lengths,
		check_bop_very_small_dataset,
		check_bop_with_slice_data_reinput,
		check_bop_nan_handling,
		check_bop_streaming,
		check_bop_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let open = source_type(&c, "open");
		let high = source_type(&c, "high");
		let low = source_type(&c, "low");
		let close = source_type(&c, "close");

		let batch_output = BopBatchBuilder::new()
			.kernel(kernel)
			.apply_slices(open, high, low, close)?;

		assert_eq!(batch_output.cols, c.close.len());
		assert_eq!(batch_output.rows, 1);

		// Confirm that batch output matches scalar indicator
		let input = BopInput::from_slices(open, high, low, close, BopParams::default());
		let scalar = bop_with_kernel(&input, kernel)?;

		for (i, &v) in batch_output.values.iter().enumerate() {
			assert!(
				(v - scalar.values[i]).abs() < 1e-12,
				"[{test}] batch value mismatch at idx {i}: {v} vs {scalar:?}"
			);
		}
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
				}
			}
		};
	}
	gen_batch_tests!(check_batch_default_row);

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let open = source_type(&c, "open");
		let high = source_type(&c, "high");
		let low = source_type(&c, "low");
		let close = source_type(&c, "close");

		// BOP has no parameters, so we just test the single batch row
		let output = BopBatchBuilder::new()
			.kernel(kernel)
			.apply_slices(open, high, low, close)?;

		// Check every value in the entire batch matrix for poison patterns
		for (idx, &val) in output.values.iter().enumerate() {
			// Skip NaN values as they're expected in warmup periods
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();
			let row = idx / output.cols;
			let col = idx % output.cols;

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "bop")]
#[pyo3(signature = (open, high, low, close, *, kernel=None))]
pub fn bop_py<'py>(
	py: Python<'py>,
	open: numpy::PyReadonlyArray1<'py, f64>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	// Zero-copy, read-only views
	let open_slice = open.as_slice()?;
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;

	// Validate kernel before allow_threads
	let kern = validate_kernel(kernel, false)?;

	// Create input structure
	let params = BopParams::default();
	let input = BopInput::from_slices(open_slice, high_slice, low_slice, close_slice, params);

	// Get Vec<f64> from Rust function and convert to NumPy with zero-copy
	let result_vec: Vec<f64> = py
		.allow_threads(|| bop_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "BopStream")]
pub struct BopStreamPy {
	stream: BopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BopStreamPy {
	#[new]
	fn new() -> PyResult<Self> {
		let stream = BopStream::try_new().map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(BopStreamPy { stream })
	}

	/// Updates the stream with new OHLC values and returns the calculated BOP value.
	fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> f64 {
		self.stream.update(open, high, low, close)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "bop_batch")]
#[pyo3(signature = (open, high, low, close, *, kernel=None))]
pub fn bop_batch_py<'py>(
	py: Python<'py>,
	open: numpy::PyReadonlyArray1<'py, f64>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::{PyDict, PyList};

	// Zero-copy, read-only views
	let open_slice = open.as_slice()?;
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;

	// Validate kernel before allow_threads
	let kern = validate_kernel(kernel, true)?;

	// For BOP batch, we have only 1 row since BOP has no parameters
	let rows = 1;
	let cols = open_slice.len();

	// Pre-allocate output array (OK for batch operations)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Compute without GIL - write directly to pre-allocated array
	py.allow_threads(|| -> Result<(), BopError> {
		// Use the zero-copy inner function that writes directly to output
		bop_batch_inner_into(open_slice, high_slice, low_slice, close_slice, kern, slice_out)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item("rows", rows)?;
	dict.set_item("cols", cols)?;

	// BOP has no parameters, so we return empty arrays - matching ALMA pattern
	dict.set_item("params", Vec::<f64>::new().into_pyarray(py))?;

	Ok(dict)
}

/// Write directly to output slice - no allocations
#[inline]
pub fn bop_into_slice(
	dst: &mut [f64],
	open: &[f64],
	high: &[f64],
	low: &[f64],
	close: &[f64],
	kern: Kernel,
) -> Result<(), BopError> {
	// Validate inputs
	let len = open.len();
	if len == 0 {
		return Err(BopError::DataIsEmpty);
	}
	if high.len() != len || low.len() != len || close.len() != len {
		return Err(BopError::InputLengthsMismatch {
			open_len: len,
			high_len: high.len(),
			low_len: low.len(),
			close_len: close.len(),
		});
	}
	if dst.len() != len {
		return Err(BopError::InputLengthsMismatch {
			open_len: len,
			high_len: high.len(),
			low_len: low.len(),
			close_len: dst.len(),
		});
	}

	// Find warmup period
	let warmup_period = (0..len)
		.find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.unwrap_or(len);

	// Select kernel
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	// Compute directly into output
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => bop_scalar(open, high, low, close, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => bop_avx2(open, high, low, close, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => bop_avx512(open, high, low, close, dst),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				bop_scalar(open, high, low, close, dst)
			}
			_ => unreachable!(),
		}
	}

	// Fill warmup with NaN
	for v in &mut dst[..warmup_period] {
		*v = f64::NAN;
	}

	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bop_js(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Result<Vec<f64>, JsValue> {
	// Single allocation pattern
	let mut output = vec![0.0; open.len()];

	bop_into_slice(&mut output, open, high, low, close, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bop_into(
	open_ptr: *const f64,
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<(), JsValue> {
	if open_ptr.is_null() || high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let open = std::slice::from_raw_parts(open_ptr, len);
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);

		// Check for aliasing with any input
		if open_ptr == out_ptr || high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
			// For aliasing case, we need to be careful about which input is aliased
			// BOP formula: (close - open) / (high - low)
			// We can reuse the output buffer by reading values before overwriting
			
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			
			// Find warmup period
			let warmup_period = (0..len)
				.find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
				.unwrap_or(len);
			
			// Fill warmup with NaN first
			for v in &mut out[..warmup_period] {
				*v = f64::NAN;
			}
			
			// Compute BOP values starting from first valid index
			for i in warmup_period..len {
				let denom = high[i] - low[i];
				out[i] = if denom <= 0.0 {
					0.0
				} else {
					(close[i] - open[i]) / denom
				};
			}
		} else {
			// No aliasing, compute directly
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			bop_into_slice(out, open, high, low, close, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bop_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bop_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bop_batch_js(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Result<Vec<f64>, JsValue> {
	// BOP has no parameters, so batch processing is just the regular calculation
	// Use single allocation pattern
	let mut output = vec![0.0; open.len()];
	
	bop_into_slice(&mut output, open, high, low, close, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bop_batch_into(
	open_ptr: *const f64,
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
) -> Result<usize, JsValue> {
	if open_ptr.is_null() || high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let open = std::slice::from_raw_parts(open_ptr, len);
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let out = std::slice::from_raw_parts_mut(out_ptr, len);

		// BOP has no parameters, so batch is just single calculation
		bop_into_slice(out, open, high, low, close, Kernel::Auto)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(1) // Always 1 row for BOP (no parameters)
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bop_batch_metadata_js() -> Result<Vec<f64>, JsValue> {
	// BOP has no parameters, return empty metadata array
	// This maintains the same structure as ALMA for uniform treatment
	Ok(Vec::new())
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BopBatchConfig {
	// BOP has no parameters, but we keep this for API consistency
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BopBatchJsOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = bop_batch)]
pub fn bop_batch_unified_js(
	open: &[f64],
	high: &[f64],
	low: &[f64],
	close: &[f64],
	_config: JsValue, // Unused but kept for API consistency
) -> Result<JsValue, JsValue> {
	// Run the BOP calculation
	let output =
		bop_batch_with_kernel(open, high, low, close, Kernel::Scalar).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// Create the structured output
	let js_output = BopBatchJsOutput {
		values: output.values,
		rows: output.rows,
		cols: output.cols,
	};

	// Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
