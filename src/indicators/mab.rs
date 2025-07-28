//! # Moving Average Bands (MAB)
//!
//! Calculates upper, middle, and lower bands based on fast and slow moving averages and the rolling standard deviation of their difference over the fast window.
//!
//! ## Parameters
//! - **fast_period**: Fast MA window (default: 10)
//! - **slow_period**: Slow MA window (default: 50)
//! - **devup**: Upper band multiplier (default: 1.0)
//! - **devdn**: Lower band multiplier (default: 1.0)
//! - **fast_ma_type**: Fast MA type ("sma" or "ema", default: "sma")
//! - **slow_ma_type**: Slow MA type ("sma" or "ema", default: "sma")
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are NaN.
//! - **InvalidPeriod**: Fast/slow period is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data for the required periods.
//! - **EmptyData**: Input slice is empty.
//!
//! ## Returns
//! - `Ok(MabOutput)` with `.upperband`, `.middleband`, `.lowerband` (all Vec<f64>)
//! - `Err(MabError)` otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MabInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			MabData::Slice(slice) => slice,
			MabData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum MabData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MabOutput {
	pub upperband: Vec<f64>,
	pub middleband: Vec<f64>,
	pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MabParams {
	pub fast_period: Option<usize>,
	pub slow_period: Option<usize>,
	pub devup: Option<f64>,
	pub devdn: Option<f64>,
	pub fast_ma_type: Option<String>,
	pub slow_ma_type: Option<String>,
}

impl Default for MabParams {
	fn default() -> Self {
		Self {
			fast_period: Some(10),
			slow_period: Some(50),
			devup: Some(1.0),
			devdn: Some(1.0),
			fast_ma_type: Some("sma".to_string()),
			slow_ma_type: Some("sma".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct MabInput<'a> {
	pub data: MabData<'a>,
	pub params: MabParams,
}

impl<'a> MabInput<'a> {
	pub fn from_candles(candles: &'a Candles, source: &'a str, params: MabParams) -> Self {
		Self {
			data: MabData::Candles { candles, source },
			params,
		}
	}

	pub fn from_slice(slice: &'a [f64], params: MabParams) -> Self {
		Self {
			data: MabData::Slice(slice),
			params,
		}
	}

	pub fn with_default_params(data: MabData<'a>) -> Self {
		Self {
			data,
			params: MabParams::default(),
		}
	}

	pub fn get_fast_period(&self) -> usize {
		self.params.fast_period.unwrap_or(10)
	}

	pub fn get_slow_period(&self) -> usize {
		self.params.slow_period.unwrap_or(50)
	}

	pub fn get_devup(&self) -> f64 {
		self.params.devup.unwrap_or(1.0)
	}

	pub fn get_devdn(&self) -> f64 {
		self.params.devdn.unwrap_or(1.0)
	}

	pub fn get_fast_ma_type(&self) -> &str {
		self.params.fast_ma_type.as_ref().map(|s| s.as_str()).unwrap_or("sma")
	}

	pub fn get_slow_ma_type(&self) -> &str {
		self.params.slow_ma_type.as_ref().map(|s| s.as_str()).unwrap_or("sma")
	}
}

#[derive(Error, Debug)]
pub enum MabError {
	#[error("mab: All values are NaN.")]
	AllValuesNaN,
	#[error("mab: Invalid period: fast={fast} slow={slow} len={len}")]
	InvalidPeriod { fast: usize, slow: usize, len: usize },
	#[error("mab: Not enough valid data: need={need} valid={valid}")]
	NotEnoughValidData { need: usize, valid: usize },
	#[error("mab: Input data slice is empty.")]
	EmptyData,
	#[error("mab: Insufficient length: upper={upper_len} middle={middle_len} lower={lower_len} expected={expected}")]
	InvalidLength {
		upper_len: usize,
		middle_len: usize,
		lower_len: usize,
		expected: usize,
	},
}

#[inline(always)]
fn mab_validate(input: &MabInput) -> Result<usize, MabError> {
	let data = input.as_ref();
	if data.is_empty() {
		return Err(MabError::EmptyData);
	}
	let fast = input.get_fast_period();
	let slow = input.get_slow_period();
	if fast == 0 || slow == 0 || fast > data.len() || slow > data.len() {
		return Err(MabError::InvalidPeriod {
			fast,
			slow,
			len: data.len(),
		});
	}

	let first_valid = data.iter().position(|&x| !x.is_nan()).ok_or(MabError::AllValuesNaN)?;
	let max_period = fast.max(slow);
	if data.len() - first_valid < max_period {
		return Err(MabError::NotEnoughValidData {
			need: max_period,
			valid: data.len() - first_valid,
		});
	}
	Ok(first_valid)
}

#[inline(always)]
fn mab_prepare<'a>(input: &'a MabInput, kernel: Kernel) -> Result<(&'a [f64], usize, Kernel, usize, usize, f64, f64), MabError> {
	let first_valid = mab_validate(input)?;
	let data = input.as_ref();
	let fast = input.get_fast_period();
	let slow = input.get_slow_period();
	let devup = input.get_devup();
	let devdn = input.get_devdn();
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		_ => kernel,
	};
	let warmup = first_valid + fast.max(slow) - 1;
	Ok((data, warmup, chosen, fast, slow, devup, devdn))
}

pub fn mab(input: &MabInput) -> Result<MabOutput, MabError> {
	let (_data, _warmup, kernel, _fast, _slow, _devup, _devdn) = mab_prepare(input, Kernel::Auto)?;
	mab_with_kernel(input, kernel)
}

fn mab_with_kernel(input: &MabInput, kernel: Kernel) -> Result<MabOutput, MabError> {
	let (data, warmup, chosen, fast_period, slow_period, devup, devdn) = mab_prepare(input, kernel)?;
	let first_valid = mab_validate(input)?;

	// Get MA types
	let fast_ma_type = input.get_fast_ma_type();
	let slow_ma_type = input.get_slow_ma_type();

	// Create inputs for the MAs
	use crate::indicators::sma::{sma, SmaInput, SmaParams};
	use crate::indicators::ema::{ema, EmaInput, EmaParams};

	// Compute fast MA
	let fast_ma = match fast_ma_type {
		"ema" => {
			let params = EmaParams {
				period: Some(fast_period),
			};
			let input = EmaInput::from_slice(data, params);
			let result = ema(&input).map_err(|_| MabError::NotEnoughValidData {
				need: fast_period,
				valid: data.len() - first_valid,
			})?;
			result.values
		}
		_ => {
			// Default to SMA
			let params = SmaParams {
				period: Some(fast_period),
			};
			let input = SmaInput::from_slice(data, params);
			let result = sma(&input).map_err(|_| MabError::NotEnoughValidData {
				need: fast_period,
				valid: data.len() - first_valid,
			})?;
			result.values
		}
	};

	// Compute slow MA
	let slow_ma = match slow_ma_type {
		"ema" => {
			let params = EmaParams {
				period: Some(slow_period),
			};
			let input = EmaInput::from_slice(data, params);
			let result = ema(&input).map_err(|_| MabError::NotEnoughValidData {
				need: slow_period,
				valid: data.len() - first_valid,
			})?;
			result.values
		}
		_ => {
			// Default to SMA
			let params = SmaParams {
				period: Some(slow_period),
			};
			let input = SmaInput::from_slice(data, params);
			let result = sma(&input).map_err(|_| MabError::NotEnoughValidData {
				need: slow_period,
				valid: data.len() - first_valid,
			})?;
			result.values
		}
	};

	// Use alloc_with_nan_prefix for zero-copy allocation
	let mut upperband = alloc_with_nan_prefix(data.len(), warmup);
	let mut middleband = alloc_with_nan_prefix(data.len(), warmup);
	let mut lowerband = alloc_with_nan_prefix(data.len(), warmup);

	mab_compute_into(
		&fast_ma,
		&slow_ma,
		fast_period,
		devup,
		devdn,
		first_valid,
		chosen,
		&mut upperband,
		&mut middleband,
		&mut lowerband,
	);

	Ok(MabOutput {
		upperband,
		middleband,
		lowerband,
	})
}

#[inline]
fn mab_compute_into(
	fast_ma: &[f64],
	slow_ma: &[f64],
	fast_period: usize,
	devup: f64,
	devdn: f64,
	first_valid: usize,
	kernel: Kernel,
	upper: &mut [f64],
	mid: &mut [f64],
	lower: &mut [f64],
) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => mab_scalar(
				fast_ma,
				slow_ma,
				fast_period,
				devup,
				devdn,
				first_valid,
				upper,
				mid,
				lower,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => mab_avx2(
				fast_ma,
				slow_ma,
				fast_period,
				devup,
				devdn,
				first_valid,
				upper,
				mid,
				lower,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => mab_avx512(
				fast_ma,
				slow_ma,
				fast_period,
				devup,
				devdn,
				first_valid,
				upper,
				mid,
				lower,
			),
			_ => unreachable!(),
		}
	}
}

#[inline(always)]
pub unsafe fn mab_scalar(
	fast_ma: &[f64],
	slow_ma: &[f64],
	fast_period: usize,
	devup: f64,
	devdn: f64,
	first: usize,
	upper: &mut [f64],
	mid: &mut [f64],
	lower: &mut [f64],
) {
	// Compute warmup for MAB specifically
	let warmup = fast_ma
		.iter()
		.take_while(|&&x| x.is_nan())
		.count()
		.max(slow_ma.iter().take_while(|&&x| x.is_nan()).count());

	// Main computation
	let mut diffs: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, fast_period);
	diffs.resize(fast_period, 0.0);

	for i in warmup..fast_ma.len() {
		let diff = fast_ma[i] - slow_ma[i];
		diffs[i % fast_period] = diff;

		if i >= warmup + fast_period - 1 {
			// Compute mean of diffs
			let mean = diffs.iter().sum::<f64>() / fast_period as f64;

			// Compute standard deviation
			let variance = diffs.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / fast_period as f64;
			let std_dev = variance.sqrt();

			mid[i] = fast_ma[i];
			upper[i] = fast_ma[i] + devup * std_dev;
			lower[i] = fast_ma[i] - devdn * std_dev;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mab_avx2(
	fast_ma: &[f64],
	slow_ma: &[f64],
	fast_period: usize,
	devup: f64,
	devdn: f64,
	first: usize,
	upper: &mut [f64],
	mid: &mut [f64],
	lower: &mut [f64],
) {
	// Compute warmup for MAB specifically
	let warmup = fast_ma
		.iter()
		.take_while(|&&x| x.is_nan())
		.count()
		.max(slow_ma.iter().take_while(|&&x| x.is_nan()).count());

	// Main computation
	let mut diffs: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, fast_period);
	diffs.resize(fast_period, 0.0);

	let devup_vec = _mm256_set1_pd(devup);
	let devdn_vec = _mm256_set1_pd(devdn);
	let period_f64 = fast_period as f64;
	let inv_period = _mm256_set1_pd(1.0 / period_f64);

	for i in warmup..fast_ma.len() {
		let diff = fast_ma[i] - slow_ma[i];
		diffs[i % fast_period] = diff;

		if i >= warmup + fast_period - 1 {
			// Compute mean of diffs using AVX2
			let mut sum_vec = _mm256_setzero_pd();
			let mut j = 0;

			// Process 4 elements at a time
			while j + 3 < fast_period {
				let diff_vec = _mm256_loadu_pd(&diffs[j]);
				sum_vec = _mm256_add_pd(sum_vec, diff_vec);
				j += 4;
			}

			// Sum the vector elements
			let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum_vec);
			let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

			// Handle remaining elements
			while j < fast_period {
				sum += diffs[j];
				j += 1;
			}

			let mean = sum / period_f64;
			let mean_vec = _mm256_set1_pd(mean);

			// Compute variance using AVX2
			let mut variance_vec = _mm256_setzero_pd();
			j = 0;

			while j + 3 < fast_period {
				let diff_vec = _mm256_loadu_pd(&diffs[j]);
				let centered = _mm256_sub_pd(diff_vec, mean_vec);
				let squared = _mm256_mul_pd(centered, centered);
				variance_vec = _mm256_add_pd(variance_vec, squared);
				j += 4;
			}

			// Sum the variance vector
			let var_array = std::mem::transmute::<__m256d, [f64; 4]>(variance_vec);
			let mut variance = var_array[0] + var_array[1] + var_array[2] + var_array[3];

			// Handle remaining elements
			while j < fast_period {
				let centered = diffs[j] - mean;
				variance += centered * centered;
				j += 1;
			}

			variance /= period_f64;
			let std_dev = variance.sqrt();
			let std_dev_vec = _mm256_set1_pd(std_dev);

			// Compute bands
			let fast_ma_val = _mm256_set1_pd(fast_ma[i]);
			let upper_val = _mm256_add_pd(fast_ma_val, _mm256_mul_pd(devup_vec, std_dev_vec));
			let lower_val = _mm256_sub_pd(fast_ma_val, _mm256_mul_pd(devdn_vec, std_dev_vec));

			// Store results
			mid[i] = fast_ma[i];
			upper[i] = _mm256_cvtsd_f64(upper_val);
			lower[i] = _mm256_cvtsd_f64(lower_val);
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn mab_avx512(
	fast_ma: &[f64],
	slow_ma: &[f64],
	fast_period: usize,
	devup: f64,
	devdn: f64,
	first: usize,
	upper: &mut [f64],
	mid: &mut [f64],
	lower: &mut [f64],
) {
	// Compute warmup for MAB specifically
	let warmup = fast_ma
		.iter()
		.take_while(|&&x| x.is_nan())
		.count()
		.max(slow_ma.iter().take_while(|&&x| x.is_nan()).count());

	// Main computation
	let mut diffs: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, fast_period);
	diffs.resize(fast_period, 0.0);

	let devup_vec = _mm512_set1_pd(devup);
	let devdn_vec = _mm512_set1_pd(devdn);
	let period_f64 = fast_period as f64;
	let inv_period = _mm512_set1_pd(1.0 / period_f64);

	for i in warmup..fast_ma.len() {
		let diff = fast_ma[i] - slow_ma[i];
		diffs[i % fast_period] = diff;

		if i >= warmup + fast_period - 1 {
			// Compute mean of diffs using AVX512
			let mut sum_vec = _mm512_setzero_pd();
			let mut j = 0;

			// Process 8 elements at a time
			while j + 7 < fast_period {
				let diff_vec = _mm512_loadu_pd(&diffs[j]);
				sum_vec = _mm512_add_pd(sum_vec, diff_vec);
				j += 8;
			}

			// Sum the vector elements
			let sum = _mm512_reduce_add_pd(sum_vec) + diffs[j..fast_period].iter().sum::<f64>();
			let mean = sum / period_f64;
			let mean_vec = _mm512_set1_pd(mean);

			// Compute variance using AVX512
			let mut variance_vec = _mm512_setzero_pd();
			j = 0;

			while j + 7 < fast_period {
				let diff_vec = _mm512_loadu_pd(&diffs[j]);
				let centered = _mm512_sub_pd(diff_vec, mean_vec);
				let squared = _mm512_mul_pd(centered, centered);
				variance_vec = _mm512_add_pd(variance_vec, squared);
				j += 8;
			}

			// Sum the variance and handle remainder
			let mut variance = _mm512_reduce_add_pd(variance_vec);
			while j < fast_period {
				let centered = diffs[j] - mean;
				variance += centered * centered;
				j += 1;
			}

			variance /= period_f64;
			let std_dev = variance.sqrt();
			let std_dev_vec = _mm512_set1_pd(std_dev);

			// Compute bands
			let fast_ma_val = _mm512_set1_pd(fast_ma[i]);
			let upper_val = _mm512_add_pd(fast_ma_val, _mm512_mul_pd(devup_vec, std_dev_vec));
			let lower_val = _mm512_sub_pd(fast_ma_val, _mm512_mul_pd(devdn_vec, std_dev_vec));

			// Store results
			mid[i] = fast_ma[i];
			upper[i] = _mm512_reduce_add_pd(upper_val) / 8.0;
			lower[i] = _mm512_reduce_add_pd(lower_val) / 8.0;
		}
	}
}

// Stream implementation
pub struct MabStream {
	fast_buffer: Vec<f64>,
	slow_buffer: Vec<f64>,
	diffs_buffer: Vec<f64>,
	fast_index: usize,
	slow_index: usize,
	diff_index: usize,
	count: usize,
	fast_period: usize,
	slow_period: usize,
	devup: f64,
	devdn: f64,
	fast_ma_type: String,
	slow_ma_type: String,
	fast_sum: f64,
	slow_sum: f64,
	fast_ma: f64,
	slow_ma: f64,
	ema_fast: f64,
	ema_slow: f64,
	kernel: Kernel,
}

impl MabStream {
	pub fn try_new(params: MabParams) -> Result<Self, String> {
		let fast_period = params.fast_period.unwrap_or(10);
		let slow_period = params.slow_period.unwrap_or(50);
		let devup = params.devup.unwrap_or(1.0);
		let devdn = params.devdn.unwrap_or(1.0);
		let fast_ma_type = params.fast_ma_type.unwrap_or_else(|| "sma".to_string());
		let slow_ma_type = params.slow_ma_type.unwrap_or_else(|| "sma".to_string());

		if fast_period == 0 || slow_period == 0 {
			return Err("Period cannot be zero".to_string());
		}

		Ok(Self {
			fast_buffer: vec![0.0; fast_period],
			slow_buffer: vec![0.0; slow_period],
			diffs_buffer: vec![0.0; fast_period],
			fast_index: 0,
			slow_index: 0,
			diff_index: 0,
			count: 0,
			fast_period,
			slow_period,
			devup,
			devdn,
			fast_ma_type,
			slow_ma_type,
			fast_sum: 0.0,
			slow_sum: 0.0,
			fast_ma: 0.0,
			slow_ma: 0.0,
			ema_fast: 0.0,
			ema_slow: 0.0,
			kernel: detect_best_kernel(),
		})
	}

	pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
		if value.is_nan() {
			return None;
		}

		self.count += 1;

		// Update fast MA
		match self.fast_ma_type.as_str() {
			"ema" => {
				if self.count == 1 {
					self.ema_fast = value;
					self.fast_ma = value;
				} else {
					let k = 2.0 / (self.fast_period as f64 + 1.0);
					self.ema_fast = value * k + self.ema_fast * (1.0 - k);
					self.fast_ma = self.ema_fast;
				}
			}
			_ => {
				// SMA
				if self.count <= self.fast_period {
					self.fast_buffer[self.fast_index] = value;
					self.fast_sum += value;
					if self.count == self.fast_period {
						self.fast_ma = self.fast_sum / self.fast_period as f64;
					}
				} else {
					let old_value = self.fast_buffer[self.fast_index];
					self.fast_buffer[self.fast_index] = value;
					self.fast_sum += value - old_value;
					self.fast_ma = self.fast_sum / self.fast_period as f64;
				}
				self.fast_index = (self.fast_index + 1) % self.fast_period;
			}
		}

		// Update slow MA
		match self.slow_ma_type.as_str() {
			"ema" => {
				if self.count == 1 {
					self.ema_slow = value;
					self.slow_ma = value;
				} else {
					let k = 2.0 / (self.slow_period as f64 + 1.0);
					self.ema_slow = value * k + self.ema_slow * (1.0 - k);
					self.slow_ma = self.ema_slow;
				}
			}
			_ => {
				// SMA
				if self.count <= self.slow_period {
					self.slow_buffer[self.slow_index] = value;
					self.slow_sum += value;
					if self.count == self.slow_period {
						self.slow_ma = self.slow_sum / self.slow_period as f64;
					}
				} else {
					let old_value = self.slow_buffer[self.slow_index];
					self.slow_buffer[self.slow_index] = value;
					self.slow_sum += value - old_value;
					self.slow_ma = self.slow_sum / self.slow_period as f64;
				}
				self.slow_index = (self.slow_index + 1) % self.slow_period;
			}
		}

		// We need both MAs to be ready
		let max_period = self.fast_period.max(self.slow_period);
		if self.count < max_period {
			return None;
		}

		// Calculate diff and update buffer
		let diff = self.fast_ma - self.slow_ma;
		if self.count <= max_period + self.fast_period - 1 {
			self.diffs_buffer[self.diff_index] = diff;
			self.diff_index = (self.diff_index + 1) % self.fast_period;
		} else {
			self.diffs_buffer[self.diff_index] = diff;
			self.diff_index = (self.diff_index + 1) % self.fast_period;
		}

		// We need fast_period diffs to calculate std dev
		if self.count < max_period + self.fast_period - 1 {
			return None;
		}

		// Calculate mean and std dev
		let mean = self.diffs_buffer.iter().sum::<f64>() / self.fast_period as f64;
		let variance = self
			.diffs_buffer
			.iter()
			.map(|&d| (d - mean).powi(2))
			.sum::<f64>()
			/ self.fast_period as f64;
		let std_dev = variance.sqrt();

		// Calculate bands
		let upper = self.fast_ma + self.devup * std_dev;
		let middle = self.fast_ma;
		let lower = self.fast_ma - self.devdn * std_dev;

		Some((upper, middle, lower))
	}
}

// Batch implementation
#[derive(Clone, Debug)]
pub struct MabBatchRange {
	pub fast_period: (usize, usize, usize),
	pub slow_period: (usize, usize, usize),
	pub devup: (f64, f64, f64),
	pub devdn: (f64, f64, f64),
	pub fast_ma_type: (String, String, String),
	pub slow_ma_type: (String, String, String),
}

impl Default for MabBatchRange {
	fn default() -> Self {
		Self {
			fast_period: (10, 12, 1),
			slow_period: (50, 50, 0),
			devup: (1.0, 1.0, 0.0),
			devdn: (1.0, 1.0, 0.0),
			fast_ma_type: ("sma".to_string(), "sma".to_string(), String::new()),
			slow_ma_type: ("sma".to_string(), "sma".to_string(), String::new()),
		}
	}
}

#[derive(Clone, Debug)]
pub struct MabBatchOutput {
	pub upperbands: Vec<f64>,
	pub middlebands: Vec<f64>,
	pub lowerbands: Vec<f64>,
	pub combos: Vec<MabParams>,
	pub rows: usize,
	pub cols: usize,
}
impl MabBatchOutput {
	pub fn row_for_params(&self, p: &MabParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.fast_period == p.fast_period
				&& c.slow_period == p.slow_period
				&& c.devup == p.devup
				&& c.devdn == p.devdn
				&& c.fast_ma_type == p.fast_ma_type
				&& c.slow_ma_type == p.slow_ma_type
		})
	}

	pub fn upper_slice(&self, row: usize) -> Option<&[f64]> {
		if row < self.rows {
			let start = row * self.cols;
			let end = start + self.cols;
			Some(&self.upperbands[start..end])
		} else {
			None
		}
	}

	pub fn middle_slice(&self, row: usize) -> Option<&[f64]> {
		if row < self.rows {
			let start = row * self.cols;
			let end = start + self.cols;
			Some(&self.middlebands[start..end])
		} else {
			None
		}
	}

	pub fn lower_slice(&self, row: usize) -> Option<&[f64]> {
		if row < self.rows {
			let start = row * self.cols;
			let end = start + self.cols;
			Some(&self.lowerbands[start..end])
		} else {
			None
		}
	}
}

fn expand_grid(p: &MabBatchRange) -> Vec<MabParams> {
	let mut combos = vec![];

	// Generate all fast periods
	let mut fast_periods = vec![];
	if p.fast_period.2 == 0 {
		fast_periods.push(p.fast_period.0);
	} else {
		let mut period = p.fast_period.0;
		while period <= p.fast_period.1 {
			fast_periods.push(period);
			period += p.fast_period.2;
		}
	}

	// Generate all slow periods
	let mut slow_periods = vec![];
	if p.slow_period.2 == 0 {
		slow_periods.push(p.slow_period.0);
	} else {
		let mut period = p.slow_period.0;
		while period <= p.slow_period.1 {
			slow_periods.push(period);
			period += p.slow_period.2;
		}
	}

	// Generate all devup values
	let mut devups = vec![];
	if p.devup.2 == 0.0 {
		devups.push(p.devup.0);
	} else {
		let mut dev = p.devup.0;
		while dev <= p.devup.1 {
			devups.push(dev);
			dev += p.devup.2;
		}
	}

	// Generate all devdn values
	let mut devdns = vec![];
	if p.devdn.2 == 0.0 {
		devdns.push(p.devdn.0);
	} else {
		let mut dev = p.devdn.0;
		while dev <= p.devdn.1 {
			devdns.push(dev);
			dev += p.devdn.2;
		}
	}

	// Create all combinations
	for &fast in &fast_periods {
		for &slow in &slow_periods {
			for &devup in &devups {
				for &devdn in &devdns {
					combos.push(MabParams {
						fast_period: Some(fast),
						slow_period: Some(slow),
						devup: Some(devup),
						devdn: Some(devdn),
						fast_ma_type: Some(p.fast_ma_type.0.clone()),
						slow_ma_type: Some(p.slow_ma_type.0.clone()),
					});
				}
			}
		}
	}

	combos
}

pub fn mab_batch(input: &[f64], sweep: &MabBatchRange) -> Result<MabBatchOutput, MabError> {
	mab_batch_inner(input, sweep, Kernel::Auto, false)
}

fn mab_batch_inner(
	input: &[f64],
	sweep: &MabBatchRange,
	kernel: Kernel,
	parallel: bool,
) -> Result<MabBatchOutput, MabError> {
	let kernel = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};

	let combos = expand_grid(sweep);
	let rows = combos.len();
	let cols = input.len();

	// Calculate warmup periods for each combination
	let first = input.iter().position(|x| !x.is_nan()).unwrap_or(0);
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|p| {
			let fast = p.fast_period.unwrap();
			let slow = p.slow_period.unwrap();
			first + fast.max(slow) - 1
		})
		.collect();

	// Use the zero-copy allocation pattern
	let mut upper_buf = make_uninit_matrix(rows, cols);
	let mut middle_buf = make_uninit_matrix(rows, cols);
	let mut lower_buf = make_uninit_matrix(rows, cols);

	init_matrix_prefixes(&mut upper_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut middle_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut lower_buf, cols, &warmup_periods);

	// Convert to mutable slices for computation
	let upper_slice = unsafe {
		std::slice::from_raw_parts_mut(upper_buf.as_mut_ptr() as *mut f64, rows * cols)
	};
	let middle_slice = unsafe {
		std::slice::from_raw_parts_mut(middle_buf.as_mut_ptr() as *mut f64, rows * cols)
	};
	let lower_slice = unsafe {
		std::slice::from_raw_parts_mut(lower_buf.as_mut_ptr() as *mut f64, rows * cols)
	};

	mab_batch_inner_into(input, sweep, simd, parallel, upper_slice, middle_slice, lower_slice)?;

	// Convert MaybeUninit to f64
	let upperbands = unsafe {
		let mut v = Vec::with_capacity(rows * cols);
		v.set_len(rows * cols);
		std::ptr::copy_nonoverlapping(upper_buf.as_ptr() as *const f64, v.as_mut_ptr(), rows * cols);
		v
	};
	let middlebands = unsafe {
		let mut v = Vec::with_capacity(rows * cols);
		v.set_len(rows * cols);
		std::ptr::copy_nonoverlapping(middle_buf.as_ptr() as *const f64, v.as_mut_ptr(), rows * cols);
		v
	};
	let lowerbands = unsafe {
		let mut v = Vec::with_capacity(rows * cols);
		v.set_len(rows * cols);
		std::ptr::copy_nonoverlapping(lower_buf.as_ptr() as *const f64, v.as_mut_ptr(), rows * cols);
		v
	};

	Ok(MabBatchOutput {
		upperbands,
		middlebands,
		lowerbands,
		combos,
		rows,
		cols,
	})
}

fn mab_batch_inner_into(
	input: &[f64],
	sweep: &MabBatchRange,
	kernel: Kernel,
	parallel: bool,
	upper_out: &mut [f64],
	middle_out: &mut [f64],
	lower_out: &mut [f64],
) -> Result<Vec<MabParams>, MabError> {
	let combos = expand_grid(sweep);
	let rows = combos.len();
	let cols = input.len();

	// Verify output slices have correct length
	if upper_out.len() != rows * cols || middle_out.len() != rows * cols || lower_out.len() != rows * cols {
		return Err(MabError::InvalidLength {
			upper_len: upper_out.len(),
			middle_len: middle_out.len(),
			lower_len: lower_out.len(),
			expected: rows * cols,
		});
	}

	// Split the output slices into chunks for parallel processing
	let upper_chunks = upper_out.par_chunks_mut(cols);
	let middle_chunks = middle_out.par_chunks_mut(cols);
	let lower_chunks = lower_out.par_chunks_mut(cols);

	// Process in parallel
	upper_chunks
		.zip(middle_chunks)
		.zip(lower_chunks)
		.enumerate()
		.for_each(|(row, ((upper_row, middle_row), lower_row))| {
			let params = &combos[row];
			let mab_params = MabParams {
				fast_period: params.fast_period,
				slow_period: params.slow_period,
				devup: params.devup,
				devdn: params.devdn,
				fast_ma_type: params.fast_ma_type.clone(),
				slow_ma_type: params.slow_ma_type.clone(),
			};
			let mab_input = MabInput::from_slice(input, mab_params);

			// Process this row
			if let Ok(output) = mab_with_kernel(&mab_input, kernel) {
				upper_row.copy_from_slice(&output.upperband);
				middle_row.copy_from_slice(&output.middleband);
				lower_row.copy_from_slice(&output.lowerband);
			}
		});

	Ok(combos)
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "mab")]
#[pyo3(signature = (data, fast_period=10, slow_period=50, devup=1.0, devdn=1.0, fast_ma_type="sma", slow_ma_type="sma", kernel=None))]
pub fn mab_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	fast_period: usize,
	slow_period: usize,
	devup: f64,
	devdn: f64,
	fast_ma_type: &str,
	slow_ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	let slice_in = data.as_slice()?;
	let params = MabParams {
		fast_period: Some(fast_period),
		slow_period: Some(slow_period),
		devup: Some(devup),
		devdn: Some(devdn),
		fast_ma_type: Some(fast_ma_type.to_string()),
		slow_ma_type: Some(slow_ma_type.to_string()),
	};
	let input = MabInput::from_slice(slice_in, params);

	let chosen_kernel = validate_kernel(kernel, false)?;

	let result = py
		.allow_threads(|| match chosen_kernel {
			Kernel::Auto => mab(&input),
			k => mab_with_kernel(&input, k),
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy conversion to NumPy arrays
	Ok((
		result.upperband.into_pyarray(py),
		result.middleband.into_pyarray(py),
		result.lowerband.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "MabStream")]
pub struct MabStreamPy {
	stream: MabStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MabStreamPy {
	#[new]
	fn new(
		fast_period: usize,
		slow_period: usize,
		devup: f64,
		devdn: f64,
		fast_ma_type: &str,
		slow_ma_type: &str,
	) -> PyResult<Self> {
		let params = MabParams {
			fast_period: Some(fast_period),
			slow_period: Some(slow_period),
			devup: Some(devup),
			devdn: Some(devdn),
			fast_ma_type: Some(fast_ma_type.to_string()),
			slow_ma_type: Some(slow_ma_type.to_string()),
		};
		let stream = MabStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(MabStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "mab_batch")]
#[pyo3(signature = (data, fast_period_range, slow_period_range, devup_range=(1.0, 1.0, 0.0), devdn_range=(1.0, 1.0, 0.0), fast_ma_type="sma", slow_ma_type="sma", kernel=None))]
pub fn mab_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	fast_period_range: (usize, usize, usize),
	slow_period_range: (usize, usize, usize),
	devup_range: (f64, f64, f64),
	devdn_range: (f64, f64, f64),
	fast_ma_type: &str,
	slow_ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;

	let sweep = MabBatchRange {
		fast_period: fast_period_range,
		slow_period: slow_period_range,
		devup: devup_range,
		devdn: devdn_range,
		fast_ma_type: (fast_ma_type.to_string(), fast_ma_type.to_string(), "".to_string()),
		slow_ma_type: (slow_ma_type.to_string(), slow_ma_type.to_string(), "".to_string()),
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output arrays for batch operations
	let upper_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let middle_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let lower_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	
	let slice_upper = unsafe { upper_arr.as_slice_mut()? };
	let slice_middle = unsafe { middle_arr.as_slice_mut()? };
	let slice_lower = unsafe { lower_arr.as_slice_mut()? };

	let kern = validate_kernel(kernel, true)?;

	// Initialize NaN prefixes outside of allow_threads
	// Calculate warmup periods for each parameter combination
	let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|p| {
			let fast = p.fast_period.unwrap();
			let slow = p.slow_period.unwrap();
			first + fast.max(slow) - 1
		})
		.collect();
	
	// Initialize NaN prefixes for all three output arrays
	for (row, &warmup) in warmup_periods.iter().enumerate() {
		let row_start = row * cols;
		let row_end = row_start + warmup.min(cols);
		for i in row_start..row_end {
			slice_upper[i] = f64::NAN;
			slice_middle[i] = f64::NAN;
			slice_lower[i] = f64::NAN;
		}
	}

	let combos = py
		.allow_threads(|| {
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};
			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => unreachable!(),
			};

			// Use zero-copy _into function to write directly to pre-allocated arrays
			mab_batch_inner_into(slice_in, &sweep, simd, true, slice_upper, slice_middle, slice_lower)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("upperbands", upper_arr.reshape((rows, cols))?)?;
	dict.set_item("middlebands", middle_arr.reshape((rows, cols))?)?;
	dict.set_item("lowerbands", lower_arr.reshape((rows, cols))?)?;
	
	// Extract parameter arrays using zero-copy
	dict.set_item(
		"fast_periods",
		combos
			.iter()
			.map(|p| p.fast_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"slow_periods",
		combos
			.iter()
			.map(|p| p.slow_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"devups",
		combos
			.iter()
			.map(|p| p.devup.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"devdns",
		combos
			.iter()
			.map(|p| p.devdn.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

// ========== WASM Bindings ==========

#[cfg(feature = "wasm")]
pub fn mab_into_slice(
	upper_dst: &mut [f64],
	middle_dst: &mut [f64],
	lower_dst: &mut [f64],
	input: &MabInput,
	kern: Kernel,
) -> Result<(), MabError> {
	let data = input.as_ref();
	let (_, warmup, _, _, _, _, _) = mab_prepare(input, kern)?;

	if upper_dst.len() != data.len() || middle_dst.len() != data.len() || lower_dst.len() != data.len() {
		return Err(MabError::InvalidLength {
			upper_len: upper_dst.len(),
			middle_len: middle_dst.len(),
			lower_len: lower_dst.len(),
			expected: data.len(),
		});
	}

	let output = mab_with_kernel(input, kern)?;

	upper_dst.copy_from_slice(&output.upperband);
	middle_dst.copy_from_slice(&output.middleband);
	lower_dst.copy_from_slice(&output.lowerband);

	// Fill warmup with NaN (already done by alloc_with_nan_prefix)
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mab_js(
	data: &[f64],
	fast_period: usize,
	slow_period: usize,
	devup: f64,
	devdn: f64,
	fast_ma_type: &str,
	slow_ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
	let params = MabParams {
		fast_period: Some(fast_period),
		slow_period: Some(slow_period),
		devup: Some(devup),
		devdn: Some(devdn),
		fast_ma_type: Some(fast_ma_type.to_string()),
		slow_ma_type: Some(slow_ma_type.to_string()),
	};
	let input = MabInput::from_slice(data, params);

	// Allocate output arrays
	let mut upper = vec![0.0; data.len()];
	let mut middle = vec![0.0; data.len()];
	let mut lower = vec![0.0; data.len()];

	mab_into_slice(&mut upper, &mut middle, &mut lower, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	// Return flattened array: [upper..., middle..., lower...]
	let mut result = Vec::with_capacity(data.len() * 3);
	result.extend_from_slice(&upper);
	result.extend_from_slice(&middle);
	result.extend_from_slice(&lower);

	Ok(result)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MabBatchConfig {
	pub fast_period_range: (usize, usize, usize),
	pub slow_period_range: (usize, usize, usize),
	pub devup_range: (f64, f64, f64),
	pub devdn_range: (f64, f64, f64),
	pub fast_ma_type: String,
	pub slow_ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MabBatchJsOutput {
	pub upperbands: Vec<f64>,
	pub middlebands: Vec<f64>,
	pub lowerbands: Vec<f64>,
	pub combos: Vec<MabParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = mab_batch)]
pub fn mab_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: MabBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = MabBatchRange {
		fast_period: config.fast_period_range,
		slow_period: config.slow_period_range,
		devup: config.devup_range,
		devdn: config.devdn_range,
		fast_ma_type: (config.fast_ma_type.clone(), config.fast_ma_type.clone(), "".to_string()),
		slow_ma_type: (config.slow_ma_type.clone(), config.slow_ma_type.clone(), "".to_string()),
	};

	let output = mab_batch_inner(data, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = MabBatchJsOutput {
		upperbands: output.upperbands,
		middlebands: output.middlebands,
		lowerbands: output.lowerbands,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mab_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mab_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mab_into(
	in_ptr: *const f64,
	upper_ptr: *mut f64,
	middle_ptr: *mut f64,
	lower_ptr: *mut f64,
	len: usize,
	fast_period: usize,
	slow_period: usize,
	devup: f64,
	devdn: f64,
	fast_ma_type: &str,
	slow_ma_type: &str,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || upper_ptr.is_null() || middle_ptr.is_null() || lower_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = MabParams {
			fast_period: Some(fast_period),
			slow_period: Some(slow_period),
			devup: Some(devup),
			devdn: Some(devdn),
			fast_ma_type: Some(fast_ma_type.to_string()),
			slow_ma_type: Some(slow_ma_type.to_string()),
		};
		let input = MabInput::from_slice(data, params);

		// Check for aliasing between input and any output
		let need_temp = in_ptr == upper_ptr || in_ptr == middle_ptr || in_ptr == lower_ptr;

		if need_temp {
			// Use temporary buffers
			let mut temp_upper = vec![0.0; len];
			let mut temp_middle = vec![0.0; len];
			let mut temp_lower = vec![0.0; len];

			mab_into_slice(&mut temp_upper, &mut temp_middle, &mut temp_lower, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;

			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);

			upper_out.copy_from_slice(&temp_upper);
			middle_out.copy_from_slice(&temp_middle);
			lower_out.copy_from_slice(&temp_lower);
		} else {
			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);

			mab_into_slice(upper_out, middle_out, lower_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mab_batch_into(
	in_ptr: *const f64,
	upper_ptr: *mut f64,
	middle_ptr: *mut f64,
	lower_ptr: *mut f64,
	len: usize,
	fast_period_start: usize,
	fast_period_end: usize,
	fast_period_step: usize,
	slow_period_start: usize,
	slow_period_end: usize,
	slow_period_step: usize,
	devup_start: f64,
	devup_end: f64,
	devup_step: f64,
	devdn_start: f64,
	devdn_end: f64,
	devdn_step: f64,
	fast_ma_type: &str,
	slow_ma_type: &str,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || upper_ptr.is_null() || middle_ptr.is_null() || lower_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer passed to mab_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = MabBatchRange {
			fast_period: (fast_period_start, fast_period_end, fast_period_step),
			slow_period: (slow_period_start, slow_period_end, slow_period_step),
			devup: (devup_start, devup_end, devup_step),
			devdn: (devdn_start, devdn_end, devdn_step),
			fast_ma_type: (fast_ma_type.to_string(), fast_ma_type.to_string(), "".to_string()),
			slow_ma_type: (slow_ma_type.to_string(), slow_ma_type.to_string(), "".to_string()),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let upper_out = std::slice::from_raw_parts_mut(upper_ptr, rows * cols);
		let middle_out = std::slice::from_raw_parts_mut(middle_ptr, rows * cols);
		let lower_out = std::slice::from_raw_parts_mut(lower_ptr, rows * cols);

		mab_batch_inner_into(data, &sweep, Kernel::Auto, false, upper_out, middle_out, lower_out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}