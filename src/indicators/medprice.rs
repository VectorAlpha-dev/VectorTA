//! # Median Price (MEDPRICE)
//!
//! The median price is calculated as `(high + low) / 2.0` for each data point.
//! This indicator uses the provided high and low price sources and returns a
//! vector of median prices. Leading `NaN` values will be produced until the
//! first valid (non-`NaN`) values of both `high` and `low` are encountered.
//!
//! ## Parameters
//! *None*
//!
//! ## Errors
//! - **EmptyData**: medprice: Input data slices are empty.
//! - **DifferentLength**: medprice: `high` and `low` data slices have different lengths.
//! - **AllValuesNaN**: medprice: All input data values (high or low) are `NaN`.
//!
//! ## Returns
//! - **`Ok(MedpriceOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN` until the first valid high/low pair is encountered.
//! - **`Err(MedpriceError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

/// Source data for medprice indicator.
#[derive(Debug, Clone)]
pub enum MedpriceData<'a> {
	Candles {
		candles: &'a Candles,
		high_source: &'a str,
		low_source: &'a str,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct MedpriceOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct MedpriceParams;

#[derive(Debug, Clone)]
pub struct MedpriceInput<'a> {
	pub data: MedpriceData<'a>,
	pub params: MedpriceParams,
}

impl<'a> MedpriceInput<'a> {
	#[inline]
	pub fn from_candles(
		candles: &'a Candles,
		high_source: &'a str,
		low_source: &'a str,
		params: MedpriceParams,
	) -> Self {
		Self {
			data: MedpriceData::Candles {
				candles,
				high_source,
				low_source,
			},
			params,
		}
	}

	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MedpriceParams) -> Self {
		Self {
			data: MedpriceData::Slices { high, low },
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "high", "low", MedpriceParams::default())
	}

	#[inline]
	pub fn get_high_low(&self) -> (&[f64], &[f64]) {
		match &self.data {
			MedpriceData::Candles {
				candles,
				high_source,
				low_source,
			} => (source_type(candles, high_source), source_type(candles, low_source)),
			MedpriceData::Slices { high, low } => (high, low),
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct MedpriceBuilder {
	kernel: Kernel,
}

impl Default for MedpriceBuilder {
	fn default() -> Self {
		Self { kernel: Kernel::Auto }
	}
}

impl MedpriceBuilder {
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
	pub fn apply(self, candles: &Candles) -> Result<MedpriceOutput, MedpriceError> {
		let input = MedpriceInput::with_default_candles(candles);
		medprice_with_kernel(&input, self.kernel)
	}

	#[inline]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MedpriceOutput, MedpriceError> {
		let input = MedpriceInput::from_slices(high, low, MedpriceParams::default());
		medprice_with_kernel(&input, self.kernel)
	}

	#[inline]
	pub fn into_stream(self) -> Result<MedpriceStream, MedpriceError> {
		MedpriceStream::try_new()
	}
}

#[derive(Debug, Error)]
pub enum MedpriceError {
	#[error("medprice: Empty data provided.")]
	EmptyData,
	#[error("medprice: Different lengths for high ({high_len}) and low ({low_len}).")]
	DifferentLength { high_len: usize, low_len: usize },
	#[error("medprice: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn medprice(input: &MedpriceInput) -> Result<MedpriceOutput, MedpriceError> {
	medprice_with_kernel(input, Kernel::Auto)
}

pub fn medprice_with_kernel(input: &MedpriceInput, kernel: Kernel) -> Result<MedpriceOutput, MedpriceError> {
	let (high, low) = input.get_high_low();

	if high.is_empty() || low.is_empty() {
		return Err(MedpriceError::EmptyData);
	}
	if high.len() != low.len() {
		return Err(MedpriceError::DifferentLength {
			high_len: high.len(),
			low_len: low.len(),
		});
	}

	let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
		Some(idx) => idx,
		None => return Err(MedpriceError::AllValuesNaN),
	};

	let mut out = vec![f64::NAN; high.len()];

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => medprice_scalar(high, low, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => medprice_avx512(high, low, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(MedpriceOutput { values: out })
}

#[inline]
pub fn medprice_scalar(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	for i in first..high.len() {
		if high[i].is_nan() || low[i].is_nan() {
			continue;
		}
		out[i] = (high[i] + low[i]) * 0.5;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medprice_avx2(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	// AVX2 stub, just call scalar
	medprice_scalar(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medprice_avx512(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	// AVX512 stub, just call scalar
	medprice_scalar(high, low, first, out)
}

// Row functions
#[inline(always)]
pub unsafe fn medprice_row_scalar(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	medprice_scalar(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx2(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	medprice_avx2(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	medprice_avx512(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512_short(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	medprice_avx512(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512_long(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
	medprice_avx512(high, low, first, out)
}

// Streaming (single-point) stateful
#[derive(Debug, Clone)]
pub struct MedpriceStream {
	started: bool,
}

impl MedpriceStream {
	pub fn try_new() -> Result<Self, MedpriceError> {
		Ok(Self { started: false })
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		if high.is_nan() || low.is_nan() {
			return None;
		}
		Some((high + low) * 0.5)
	}
}

// Batch/grid sweep for "expand_grid" compatibility (for future-proof API parity)
#[derive(Clone, Debug)]
pub struct MedpriceBatchRange {
	pub dummy: (usize, usize, usize), // for compatibility
}
impl Default for MedpriceBatchRange {
	fn default() -> Self {
		Self { dummy: (0, 0, 0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct MedpriceBatchBuilder {
	kernel: Kernel,
	range: MedpriceBatchRange,
}

impl MedpriceBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<MedpriceBatchOutput, MedpriceError> {
		medprice_batch_with_kernel(high, low, self.kernel)
	}
	pub fn apply_candles(
		self,
		c: &Candles,
		high_src: &str,
		low_src: &str,
	) -> Result<MedpriceBatchOutput, MedpriceError> {
		let high = source_type(c, high_src);
		let low = source_type(c, low_src);
		self.apply_slice(high, low)
	}
}

pub fn medprice_batch_with_kernel(high: &[f64], low: &[f64], k: Kernel) -> Result<MedpriceBatchOutput, MedpriceError> {
	medprice_batch_par_slice(high, low, k)
}

#[derive(Clone, Debug)]
pub struct MedpriceBatchOutput {
	pub values: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

#[inline(always)]
fn expand_grid(_r: &MedpriceBatchRange) -> Vec<MedpriceParams> {
	vec![MedpriceParams::default()]
}

#[inline(always)]
pub fn medprice_batch_slice(high: &[f64], low: &[f64], kern: Kernel) -> Result<MedpriceBatchOutput, MedpriceError> {
	medprice_batch_inner(high, low, kern, false)
}

#[inline(always)]
pub fn medprice_batch_par_slice(high: &[f64], low: &[f64], kern: Kernel) -> Result<MedpriceBatchOutput, MedpriceError> {
	medprice_batch_inner(high, low, kern, true)
}

#[inline(always)]
fn medprice_batch_inner(
	high: &[f64],
	low: &[f64],
	kern: Kernel,
	_parallel: bool,
) -> Result<MedpriceBatchOutput, MedpriceError> {
	if high.is_empty() || low.is_empty() {
		return Err(MedpriceError::EmptyData);
	}
	if high.len() != low.len() {
		return Err(MedpriceError::DifferentLength {
			high_len: high.len(),
			low_len: low.len(),
		});
	}

	let first = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
		Some(idx) => idx,
		None => return Err(MedpriceError::AllValuesNaN),
	};

	let rows = 1;
	let cols = high.len();
	let mut values = vec![f64::NAN; rows * cols];
	unsafe {
		medprice_row_scalar(high, low, first, &mut values);
	}

	Ok(MedpriceBatchOutput { values, rows, cols })
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_medprice_with_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MedpriceInput::with_default_candles(&candles);
		let output = medprice_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_medprice_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MedpriceInput::from_candles(&candles, "high", "low", MedpriceParams);
		let result = medprice_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len(), "Output length mismatch");
		let expected_last_five = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
		assert!(result.values.len() >= 5, "Not enough data for comparison");
		let start_index = result.values.len() - 5;
		let actual_last_five = &result.values[start_index..];
		for (i, &val) in actual_last_five.iter().enumerate() {
			let expected = expected_last_five[i];
			assert!(
				(val - expected).abs() < 1e-1,
				"Mismatch at last five index {}: expected {}, got {}",
				i,
				expected,
				val
			);
		}
		Ok(())
	}

	fn check_medprice_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [];
		let low = [];
		let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
		let result = medprice_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for empty data");
		Ok(())
	}

	fn check_medprice_different_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0];
		let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
		let result = medprice_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for different slice lengths");
		Ok(())
	}

	fn check_medprice_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, f64::NAN, f64::NAN];
		let low = [f64::NAN, f64::NAN, f64::NAN];
		let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
		let result = medprice_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for all NaN data");
		Ok(())
	}

	fn check_medprice_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, 100.0, 110.0];
		let low = [f64::NAN, 80.0, 90.0];
		let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
		let result = medprice_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), 3);
		assert!(result.values[0].is_nan());
		assert_eq!(result.values[1], 90.0);
		assert_eq!(result.values[2], 100.0);
		Ok(())
	}

	fn check_medprice_late_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [100.0, 110.0, f64::NAN];
		let low = [80.0, 90.0, f64::NAN];
		let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
		let result = medprice_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), 3);
		assert_eq!(result.values[0], 90.0);
		assert_eq!(result.values[1], 100.0);
		assert!(result.values[2].is_nan());
		Ok(())
	}

	fn check_medprice_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [100.0, 110.0, 120.0];
		let low = [80.0, 90.0, 100.0];
		let mut stream = MedpriceStream::try_new()?;
		let mut values = Vec::with_capacity(high.len());
		for (&h, &l) in high.iter().zip(low.iter()) {
			values.push(stream.update(h, l));
		}
		assert_eq!(values[0], Some(90.0));
		assert_eq!(values[1], Some(100.0));
		assert_eq!(values[2], Some(110.0));
		Ok(())
	}

	fn check_medprice_batch(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [100.0, 110.0, 120.0];
		let low = [80.0, 90.0, 100.0];
		let builder = MedpriceBatchBuilder::new().kernel(kernel);
		let batch = builder.apply_slice(&high, &low)?;
		assert_eq!(batch.values.len(), high.len());
		assert_eq!(batch.rows, 1);
		assert_eq!(batch.cols, 3);
		assert_eq!(batch.values, vec![90.0, 100.0, 110.0]);
		Ok(())
	}

	macro_rules! generate_all_medprice_tests {
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

	generate_all_medprice_tests!(
		check_medprice_with_default_candles,
		check_medprice_accuracy,
		check_medprice_empty_data,
		check_medprice_different_length,
		check_medprice_all_values_nan,
		check_medprice_nan_handling,
		check_medprice_late_nan_handling,
		check_medprice_streaming,
		check_medprice_batch
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let high = source_type(&c, "high");
		let low = source_type(&c, "low");

		let output = MedpriceBatchBuilder::new().kernel(kernel).apply_slice(high, low)?;

		assert_eq!(output.rows, 1, "[{test}] batch output should have one row");
		assert_eq!(output.cols, high.len(), "[{test}] batch cols mismatch");
		assert_eq!(output.values.len(), output.cols, "[{test}] values shape mismatch");

		let last_expected = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
		let start = output.values.len().saturating_sub(5);
		for (i, &val) in output.values[start..].iter().enumerate() {
			assert!(
				(val - last_expected[i]).abs() < 1e-1,
				"[{test}] batch last-five mismatch idx {i}: got {val}, expected {}",
				last_expected[i]
			);
		}
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
				}
			}
		};
	}

	gen_batch_tests!(check_batch_default_row);
}
