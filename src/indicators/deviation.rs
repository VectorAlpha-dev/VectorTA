//! # Rolling Deviation Indicator
//!
//! Computes rolling Standard Deviation, Mean Absolute Deviation, or Median Absolute Deviation.
//! This indicator provides three different measures of variability in a data series:
//!
//! ## Deviation Types
//! 
//! ### Standard Deviation (devtype = 0)
//! Measures the amount of variation or dispersion of a set of values. A low standard deviation
//! indicates that values tend to be close to the mean, while a high standard deviation indicates
//! that values are spread out over a wider range.
//!
//! Formula: σ = √(Σ(x - μ)² / n)
//!
//! ### Mean Absolute Deviation (devtype = 1)
//! The average of the absolute deviations from the mean. It's more robust to outliers than
//! standard deviation but less commonly used in financial analysis.
//!
//! Formula: MAD = Σ|x - μ| / n
//!
//! ### Median Absolute Deviation (devtype = 2)
//! The median of the absolute deviations from the median. This is the most robust measure
//! against outliers and is useful for identifying anomalies in data.
//!
//! Formula: MedAD = median(|x - median(x)|)
//!
//! ## Parameters
//! - **period**: Window size (number of data points). Must be >= 2 for standard deviation.
//! - **devtype**: Type of deviation: 0 = Standard Deviation, 1 = Mean Absolute, 2 = Median Absolute.
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are `NaN`.
//! - **InvalidPeriod**: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: Not enough valid data points for the requested `period`.
//! - **InvalidDevType**: Invalid devtype (must be 0, 1, or 2).
//!
//! ## Returns
//! - **`Ok(DeviationOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(DeviationError)`** otherwise.
//!
//! ## Example
//! ```rust
//! use rust_backtester::indicators::deviation::{deviation, DeviationInput, DeviationParams};
//! 
//! let data = vec![10.0, 20.0, 30.0, 25.0, 35.0, 28.0, 33.0, 29.0, 31.0, 30.0];
//! let params = DeviationParams {
//!     period: Some(5),
//!     devtype: Some(0), // Standard deviation
//! };
//! let input = DeviationInput::from_slice(&data, params);
//! let output = deviation(&input).unwrap();
//! ```

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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DeviationInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			DeviationData::Slice(slice) => slice,
			DeviationData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

/// Data type for deviation indicator.
#[derive(Debug, Clone)]
pub enum DeviationData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

/// Input for deviation indicator.
#[derive(Debug, Clone)]
pub struct DeviationInput<'a> {
	pub data: DeviationData<'a>,
	pub params: DeviationParams,
}

impl<'a> DeviationInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: DeviationParams) -> Self {
		Self {
			data: DeviationData::Candles { candles: c, source: s },
			params: p,
		}
	}
	
	#[inline]
	pub fn from_slice(data: &'a [f64], params: DeviationParams) -> Self {
		Self {
			data: DeviationData::Slice(data),
			params,
		}
	}
	
	#[inline]
	pub fn with_defaults(data: &'a [f64]) -> Self {
		Self {
			data: DeviationData::Slice(data),
			params: DeviationParams::default(),
		}
	}
	
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", DeviationParams::default())
	}
	
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(9)
	}
	
	#[inline]
	pub fn get_devtype(&self) -> usize {
		self.params.devtype.unwrap_or(0)
	}
	
	#[inline]
	fn as_slice(&self) -> &[f64] {
		self.as_ref()
	}
}

/// Output for deviation indicator.
#[derive(Debug, Clone)]
pub struct DeviationOutput {
	pub values: Vec<f64>,
}

/// Parameters for deviation indicator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DeviationParams {
	pub period: Option<usize>,
	pub devtype: Option<usize>,
}
impl Default for DeviationParams {
	fn default() -> Self {
		Self {
			period: Some(9),
			devtype: Some(0),
		}
	}
}

/// Builder for deviation indicator.
#[derive(Copy, Clone, Debug)]
pub struct DeviationBuilder {
	period: Option<usize>,
	devtype: Option<usize>,
	kernel: Kernel,
}
impl Default for DeviationBuilder {
	fn default() -> Self {
		Self {
			period: None,
			devtype: None,
			kernel: Kernel::Auto,
		}
	}
}
impl DeviationBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn period(mut self, n: usize) -> Self {
		self.period = Some(n);
		self
	}
	#[inline(always)]
	pub fn devtype(mut self, d: usize) -> Self {
		self.devtype = Some(d);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles, s: &str) -> Result<DeviationOutput, DeviationError> {
		let p = DeviationParams {
			period: self.period,
			devtype: self.devtype,
		};
		let i = DeviationInput::from_candles(c, s, p);
		deviation_with_kernel(&i, self.kernel)
	}
	
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<DeviationOutput, DeviationError> {
		let p = DeviationParams {
			period: self.period,
			devtype: self.devtype,
		};
		let i = DeviationInput::from_slice(d, p);
		deviation_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<DeviationStream, DeviationError> {
		let p = DeviationParams {
			period: self.period,
			devtype: self.devtype,
		};
		DeviationStream::try_new(p)
	}
}

/// Deviation indicator errors.
#[derive(Debug, Error)]
pub enum DeviationError {
	#[error("deviation: All values are NaN.")]
	AllValuesNaN,
	#[error("deviation: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("deviation: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("deviation: Invalid devtype (must be 0, 1, or 2). devtype={devtype}")]
	InvalidDevType { devtype: usize },
	#[error("deviation: Calculation error: {0}")]
	CalculationError(String),
}

#[inline(always)]
pub fn deviation(input: &DeviationInput) -> Result<DeviationOutput, DeviationError> {
	deviation_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn deviation_into_slice(dst: &mut [f64], input: &DeviationInput, kern: Kernel) -> Result<(), DeviationError> {
	let data = input.as_slice();
	let period = input.get_period();
	let devtype = input.get_devtype();
	
	if dst.len() != data.len() {
		return Err(DeviationError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}
	
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DeviationError::AllValuesNaN)?;
	
	if period == 0 || period > data.len() {
		return Err(DeviationError::InvalidPeriod { period, data_len: data.len() });
	}
	if (data.len() - first) < period {
		return Err(DeviationError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first,
		});
	}
	if !(0..=2).contains(&devtype) {
		return Err(DeviationError::InvalidDevType { devtype });
	}
	
	// Initialize output with NaN prefix
	let warmup = first + period - 1;
	for i in 0..warmup {
		dst[i] = f64::NAN;
	}
	
	// Compute directly into dst
	match devtype {
		0 => standard_deviation_rolling_into(data, period, first, dst)?,
		1 => mean_absolute_deviation_rolling_into(data, period, first, dst)?,
		2 => median_absolute_deviation_rolling_into(data, period, first, dst)?,
		_ => unreachable!(),
	}
	
	Ok(())
}

#[inline(always)]
pub fn deviation_with_kernel(input: &DeviationInput, kernel: Kernel) -> Result<DeviationOutput, DeviationError> {
	let data = input.as_slice();
	let period = input.get_period();
	
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DeviationError::AllValuesNaN)?;
		
	let len = data.len();
	let mut out = alloc_with_nan_prefix(len, first + period - 1);
	
	// Only compute from warmup period onwards
	let warmup = first + period - 1;
	if warmup < len {
		// Create a temporary input with the kernel
		deviation_into_slice(&mut out, input, kernel)?;
	}
	
	Ok(DeviationOutput { values: out })
}

#[inline(always)]
pub fn deviation_scalar(data: &[f64], period: usize, first: usize, devtype: usize) -> Result<Vec<f64>, DeviationError> {
	match devtype {
		0 => standard_deviation_rolling(data, period).map_err(|e| DeviationError::CalculationError(e.to_string())),
		1 => mean_absolute_deviation_rolling(data, period).map_err(|e| DeviationError::CalculationError(e.to_string())),
		2 => {
			median_absolute_deviation_rolling(data, period).map_err(|e| DeviationError::CalculationError(e.to_string()))
		}
		_ => Err(DeviationError::InvalidDevType { devtype }),
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx512(data: &[f64], period: usize, first: usize, devtype: usize, out: &mut [f64]) {
	deviation_scalar(data, period, first, devtype)
		.map(|v| out.copy_from_slice(&v))
		.ok();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx2(data: &[f64], period: usize, first: usize, devtype: usize, out: &mut [f64]) {
	deviation_scalar(data, period, first, devtype)
		.map(|v| out.copy_from_slice(&v))
		.ok();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx512_short(data: &[f64], period: usize, first: usize, devtype: usize, out: &mut [f64]) {
	deviation_avx512(data, period, first, devtype, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx512_long(data: &[f64], period: usize, first: usize, devtype: usize, out: &mut [f64]) {
	deviation_avx512(data, period, first, devtype, out);
}

/// Batch param sweep for deviation.
#[derive(Clone, Debug)]
pub struct DeviationBatchRange {
	pub period: (usize, usize, usize),
	pub devtype: (usize, usize, usize),
}
impl Default for DeviationBatchRange {
	fn default() -> Self {
		Self {
			period: (9, 60, 1),
			devtype: (0, 2, 1),
		}
	}
}

/// Builder for batch deviation.
#[derive(Clone, Debug, Default)]
pub struct DeviationBatchBuilder {
	range: DeviationBatchRange,
	kernel: Kernel,
}
impl DeviationBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.period = (start, end, step);
		self
	}
	#[inline]
	pub fn period_static(mut self, p: usize) -> Self {
		self.range.period = (p, p, 0);
		self
	}
	#[inline]
	pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.devtype = (start, end, step);
		self
	}
	#[inline]
	pub fn devtype_static(mut self, d: usize) -> Self {
		self.range.devtype = (d, d, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<DeviationBatchOutput, DeviationError> {
		deviation_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DeviationBatchOutput, DeviationError> {
		DeviationBatchBuilder::new().kernel(k).apply_slice(data)
	}
}

pub fn deviation_batch_with_kernel(
	data: &[f64],
	sweep: &DeviationBatchRange,
	k: Kernel,
) -> Result<DeviationBatchOutput, DeviationError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(DeviationError::InvalidPeriod { period: 0, data_len: 0 });
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	deviation_batch_par_slice(data, sweep, simd)
}

/// Output for batch deviation grid.
#[derive(Clone, Debug)]
pub struct DeviationBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<DeviationParams>,
	pub rows: usize,
	pub cols: usize,
}
impl DeviationBatchOutput {
	pub fn row_for_params(&self, p: &DeviationParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(9) == p.period.unwrap_or(9) && c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
		})
	}
	pub fn values_for(&self, p: &DeviationParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &DeviationBatchRange) -> Vec<DeviationParams> {
	// Calculate capacity upfront
	let period_count = if r.period.2 == 0 || r.period.0 == r.period.1 {
		1
	} else {
		((r.period.1 - r.period.0) / r.period.2) + 1
	};
	
	let devtype_count = if r.devtype.2 == 0 || r.devtype.0 == r.devtype.1 {
		1
	} else {
		((r.devtype.1 - r.devtype.0) / r.devtype.2) + 1
	};
	
	let mut out = Vec::with_capacity(period_count * devtype_count);
	
	// Generate periods inline
	if r.period.2 == 0 || r.period.0 == r.period.1 {
		let p = r.period.0;
		// Generate devtypes inline
		if r.devtype.2 == 0 || r.devtype.0 == r.devtype.1 {
			out.push(DeviationParams {
				period: Some(p),
				devtype: Some(r.devtype.0),
			});
		} else {
			let mut d = r.devtype.0;
			while d <= r.devtype.1 {
				out.push(DeviationParams {
					period: Some(p),
					devtype: Some(d),
				});
				d += r.devtype.2;
			}
		}
	} else {
		let mut p = r.period.0;
		while p <= r.period.1 {
			// Generate devtypes inline
			if r.devtype.2 == 0 || r.devtype.0 == r.devtype.1 {
				out.push(DeviationParams {
					period: Some(p),
					devtype: Some(r.devtype.0),
				});
			} else {
				let mut d = r.devtype.0;
				while d <= r.devtype.1 {
					out.push(DeviationParams {
						period: Some(p),
						devtype: Some(d),
					});
					d += r.devtype.2;
				}
			}
			p += r.period.2;
		}
	}
	out
}

#[inline(always)]
pub fn deviation_batch_slice(
	data: &[f64],
	sweep: &DeviationBatchRange,
	kern: Kernel,
) -> Result<DeviationBatchOutput, DeviationError> {
	deviation_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn deviation_batch_par_slice(
	data: &[f64],
	sweep: &DeviationBatchRange,
	kern: Kernel,
) -> Result<DeviationBatchOutput, DeviationError> {
	deviation_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn deviation_batch_inner(
	data: &[f64],
	sweep: &DeviationBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DeviationBatchOutput, DeviationError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DeviationError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DeviationError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(DeviationError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory like alma.rs
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to regular Vec for processing
	let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
	let values_slice: &mut [f64] = unsafe {
		std::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
	};
	
	let do_row = |row: usize, out_row: &mut [f64]| {
		let period = combos[row].period.unwrap();
		let devtype = combos[row].devtype.unwrap();
		let warmup = warmup_periods[row];
		
		// Create input for this row
		let params = DeviationParams {
			period: Some(period),
			devtype: Some(devtype),
		};
		let input = DeviationInput::from_slice(data, params);
		
		// Compute directly into the row, skipping NaN prefix
		match devtype {
			0 => standard_deviation_rolling_into(data, period, first, out_row).ok(),
			1 => mean_absolute_deviation_rolling_into(data, period, first, out_row).ok(),
			2 => median_absolute_deviation_rolling_into(data, period, first, out_row).ok(),
			_ => None,
		};
	};
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			values_slice
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	
	// Convert to owned Vec for output
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(DeviationBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline]
fn standard_deviation_rolling_into(data: &[f64], period: usize, first: usize, out: &mut [f64]) -> Result<(), DeviationError> {
	if period < 2 {
		return Err(DeviationError::InvalidPeriod { period, data_len: data.len() });
	}
	
	let first_valid_idx = first;
	if data.len() - first_valid_idx < period {
		return Err(DeviationError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first_valid_idx,
		});
	}
	
	let mut sum = 0.0;
	let mut sumsq = 0.0;
	for &val in &data[first_valid_idx..(first_valid_idx + period)] {
		sum += val;
		sumsq += val * val;
	}
	
	let idx = first_valid_idx + period - 1;
	let mean = sum / (period as f64);
	let var = (sumsq / (period as f64)) - mean * mean;
	out[idx] = var.sqrt();
	
	for i in (idx + 1)..data.len() {
		let val_in = data[i];
		let val_out = data[i - period];
		sum += val_in - val_out;
		sumsq += val_in * val_in - val_out * val_out;
		let mean = sum / (period as f64);
		let var = (sumsq / (period as f64)) - mean * mean;
		out[i] = var.sqrt();
	}
	Ok(())
}

#[inline]
fn standard_deviation_rolling(data: &[f64], period: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
	if period < 2 {
		return Err("Period must be >= 2 for standard deviation.".into());
	}
	let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err("All values are NaN.".into()),
	};
	if data.len() - first_valid_idx < period {
		return Err(format!(
			"Not enough valid data: need {}, but only {} valid from index {}.",
			period,
			data.len() - first_valid_idx,
			first_valid_idx
		)
		.into());
	}
	let mut result = alloc_with_nan_prefix(data.len(), period - 1);
	let mut sum = 0.0;
	let mut sumsq = 0.0;
	for &val in &data[first_valid_idx..(first_valid_idx + period)] {
		sum += val;
		sumsq += val * val;
	}
	let mut idx = first_valid_idx + period - 1;
	let mean = sum / (period as f64);
	let var = (sumsq / (period as f64)) - mean * mean;
	result[idx] = var.sqrt();
	for i in (idx + 1)..data.len() {
		let val_in = data[i];
		let val_out = data[i - period];
		sum += val_in - val_out;
		sumsq += val_in * val_in - val_out * val_out;
		let mean = sum / (period as f64);
		let var = (sumsq / (period as f64)) - mean * mean;
		result[i] = var.sqrt();
	}
	Ok(result)
}

#[inline]
fn mean_absolute_deviation_rolling_into(data: &[f64], period: usize, first: usize, out: &mut [f64]) -> Result<(), DeviationError> {
	let first_valid_idx = first;
	if data.len() - first_valid_idx < period {
		return Err(DeviationError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first_valid_idx,
		});
	}
	
	let start_window_end = first_valid_idx + period - 1;
	for i in start_window_end..data.len() {
		let window_start = i + 1 - period;
		let window = &data[window_start..=i];
		let mean = window.iter().sum::<f64>() / (period as f64);
		let abs_sum = window.iter().fold(0.0, |acc, &x| acc + (x - mean).abs());
		out[i] = abs_sum / (period as f64);
	}
	Ok(())
}

#[inline]
fn mean_absolute_deviation_rolling(data: &[f64], period: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
	let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err("All values are NaN.".into()),
	};
	if data.len() - first_valid_idx < period {
		return Err(format!(
			"Not enough valid data: need {}, but only {} valid from index {}.",
			period,
			data.len() - first_valid_idx,
			first_valid_idx
		)
		.into());
	}
	let mut result = alloc_with_nan_prefix(data.len(), period - 1);
	let start_window_end = first_valid_idx + period - 1;
	for i in start_window_end..data.len() {
		let window_start = i + 1 - period;
		if window_start < first_valid_idx {
			continue;
		}
		let window = &data[window_start..=i];
		let mean = window.iter().sum::<f64>() / (period as f64);
		let abs_sum = window.iter().fold(0.0, |acc, &x| acc + (x - mean).abs());
		result[i] = abs_sum / (period as f64);
	}
	Ok(result)
}

#[inline]
fn median_absolute_deviation_rolling_into(data: &[f64], period: usize, first: usize, out: &mut [f64]) -> Result<(), DeviationError> {
	let first_valid_idx = first;
	if data.len() - first_valid_idx < period {
		return Err(DeviationError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first_valid_idx,
		});
	}
	
	let start_window_end = first_valid_idx + period - 1;
	
	// Pre-allocate buffer for absolute deviations
	const STACK_SIZE: usize = 256;
	let mut stack_buffer: [f64; STACK_SIZE] = [0.0; STACK_SIZE];
	let mut heap_buffer: Vec<f64> = if period > STACK_SIZE {
		vec![0.0; period]
	} else {
		Vec::new()
	};
	
	for i in start_window_end..data.len() {
		let window_start = i + 1 - period;
		let window = &data[window_start..=i];
		let median = find_median(window);
		
		// Use pre-allocated buffer for absolute deviations
		let abs_devs = if period <= STACK_SIZE {
			&mut stack_buffer[..period]
		} else {
			&mut heap_buffer[..period]
		};
		
		for (j, &x) in window.iter().enumerate() {
			abs_devs[j] = (x - median).abs();
		}
		
		out[i] = find_median(abs_devs);
	}
	Ok(())
}

#[inline]
fn median_absolute_deviation_rolling(data: &[f64], period: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
	let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err("All values are NaN.".into()),
	};
	if data.len() - first_valid_idx < period {
		return Err(format!(
			"Not enough valid data: need {}, but only {} valid from index {}.",
			period,
			data.len() - first_valid_idx,
			first_valid_idx
		)
		.into());
	}
	let mut result = alloc_with_nan_prefix(data.len(), period - 1);
	let start_window_end = first_valid_idx + period - 1;
	
	// Pre-allocate buffer for absolute deviations
	const STACK_SIZE: usize = 256;
	let mut stack_buffer: [f64; STACK_SIZE] = [0.0; STACK_SIZE];
	let mut heap_buffer: Vec<f64> = if period > STACK_SIZE {
		vec![0.0; period]
	} else {
		Vec::new()
	};
	
	for i in start_window_end..data.len() {
		let window_start = i + 1 - period;
		if window_start < first_valid_idx {
			continue;
		}
		let window = &data[window_start..=i];
		let median = find_median(window);
		
		// Use pre-allocated buffer for absolute deviations
		let abs_devs = if period <= STACK_SIZE {
			&mut stack_buffer[..period]
		} else {
			&mut heap_buffer[..period]
		};
		
		for (j, &x) in window.iter().enumerate() {
			abs_devs[j] = (x - median).abs();
		}
		
		result[i] = find_median(abs_devs);
	}
	Ok(result)
}

#[inline]
fn find_median(slice: &[f64]) -> f64 {
	if slice.is_empty() {
		return f64::NAN;
	}
	// Use stack allocation for small windows, heap for large
	const STACK_SIZE: usize = 256;
	let len = slice.len();
	
	if len <= STACK_SIZE {
		// Stack allocation for small windows
		let mut sorted: [f64; STACK_SIZE] = [0.0; STACK_SIZE];
		sorted[..len].copy_from_slice(slice);
		let sorted_slice = &mut sorted[..len];
		sorted_slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
		
		let mid = len / 2;
		if len % 2 == 0 {
			(sorted_slice[mid - 1] + sorted_slice[mid]) / 2.0
		} else {
			sorted_slice[mid]
		}
	} else {
		// For large windows, we have to allocate
		let mut sorted = slice.to_vec();
		sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
		let mid = sorted.len() / 2;
		if sorted.len() % 2 == 0 {
			(sorted[mid - 1] + sorted[mid]) / 2.0
		} else {
			sorted[mid]
		}
	}
}

/// Streaming rolling deviation.
#[derive(Debug, Clone)]
pub struct DeviationStream {
	period: usize,
	devtype: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}
impl DeviationStream {
	pub fn try_new(params: DeviationParams) -> Result<Self, DeviationError> {
		let period = params.period.unwrap_or(9);
		if period == 0 {
			return Err(DeviationError::InvalidPeriod { period, data_len: 0 });
		}
		let devtype = params.devtype.unwrap_or(0);
		if !(0..=2).contains(&devtype) {
			return Err(DeviationError::InvalidDevType { devtype });
		}
		Ok(Self {
			period,
			devtype,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(match self.devtype {
			0 => self.std_dev_ring(),
			1 => self.mean_abs_dev_ring(),
			2 => self.median_abs_dev_ring(),
			_ => f64::NAN,
		})
	}
	#[inline(always)]
	fn std_dev_ring(&self) -> f64 {
		let n = self.period as f64;
		let sum: f64 = self.buffer.iter().sum();
		let mean = sum / n;
		let sumsq: f64 = self.buffer.iter().map(|&x| x * x).sum();
		let var = (sumsq / n) - mean * mean;
		var.sqrt()
	}
	#[inline(always)]
	fn mean_abs_dev_ring(&self) -> f64 {
		let n = self.period as f64;
		let sum: f64 = self.buffer.iter().sum();
		let mean = sum / n;
		let abs_sum = self.buffer.iter().fold(0.0, |acc, &x| acc + (x - mean).abs());
		abs_sum / n
	}
	#[inline(always)]
	fn median_abs_dev_ring(&self) -> f64 {
		let median = find_median(&self.buffer);
		let mut abs_devs: Vec<f64> = self.buffer.iter().map(|&x| (x - median).abs()).collect();
		find_median(&abs_devs)
	}
}

/// Row-level kernel API
#[inline(always)]
pub unsafe fn deviation_row_scalar(
	data: &[f64],
	first: usize,
	period: usize,
	_stride: usize,
	devtype: usize,
	out: &mut [f64],
) {
	// Initialize with NaN up to warmup period
	let warmup = first + period - 1;
	for i in 0..warmup.min(out.len()) {
		out[i] = f64::NAN;
	}
	
	// Compute directly into out
	match devtype {
		0 => standard_deviation_rolling_into(data, period, first, out).ok(),
		1 => mean_absolute_deviation_rolling_into(data, period, first, out).ok(), 
		2 => median_absolute_deviation_rolling_into(data, period, first, out).ok(),
		_ => None,
	};
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx2(
	data: &[f64],
	first: usize,
	period: usize,
	stride: usize,
	devtype: usize,
	out: &mut [f64],
) {
	deviation_row_scalar(data, first, period, stride, devtype, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx512(
	data: &[f64],
	first: usize,
	period: usize,
	stride: usize,
	devtype: usize,
	out: &mut [f64],
) {
	if period <= 32 {
		deviation_row_avx512_short(data, first, period, stride, devtype, out);
	} else {
		deviation_row_avx512_long(data, first, period, stride, devtype, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx512_short(
	data: &[f64],
	first: usize,
	period: usize,
	stride: usize,
	devtype: usize,
	out: &mut [f64],
) {
	deviation_row_scalar(data, first, period, stride, devtype, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx512_long(
	data: &[f64],
	first: usize,
	period: usize,
	stride: usize,
	devtype: usize,
	out: &mut [f64],
) {
	deviation_row_scalar(data, first, period, stride, devtype, out);
}

#[inline(always)]
pub fn deviation_expand_grid(r: &DeviationBatchRange) -> Vec<DeviationParams> {
	expand_grid(r)
}

pub use DeviationError as DevError;
pub use DeviationInput as DevInput;
pub use DeviationParams as DevParams;

use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};
impl Index<usize> for DeviationOutput {
	type Output = f64;
	fn index(&self, idx: usize) -> &Self::Output {
		&self.values[idx]
	}
}
impl IndexMut<usize> for DeviationOutput {
	fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
		&mut self.values[idx]
	}
}
impl<'a> IntoIterator for &'a DeviationOutput {
	type Item = &'a f64;
	type IntoIter = Iter<'a, f64>;
	fn into_iter(self) -> Self::IntoIter {
		self.values.iter()
	}
}
impl<'a> IntoIterator for &'a mut DeviationOutput {
	type Item = &'a mut f64;
	type IntoIter = IterMut<'a, f64>;
	fn into_iter(self) -> Self::IntoIter {
		self.values.iter_mut()
	}
}
impl DeviationOutput {
	pub fn last(&self) -> Option<&f64> {
		self.values.last()
	}
	pub fn len(&self) -> usize {
		self.values.len()
	}
	pub fn is_empty(&self) -> bool {
		self.values.is_empty()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;

	fn check_deviation_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0, 5.0];
		let input = DeviationInput::with_defaults(&data);
		let output = deviation_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), data.len());
		Ok(())
	}
	fn check_deviation_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0, 5.0];
		let params = DeviationParams {
			period: Some(3),
			devtype: Some(0),
		};
		let input = DeviationInput::from_slice(&data, params);
		let result = deviation_with_kernel(&input, kernel)?;
		let expected = 0.816496580927726;
		for &val in &result.values[2..] {
			assert!(
				(val - expected).abs() < 1e-12,
				"[{test}] deviation mismatch: got {}, expected {}",
				val,
				expected
			);
		}
		Ok(())
	}
	fn check_deviation_default_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
		let input = DeviationInput::with_defaults(&data);
		let output = deviation_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), data.len());
		Ok(())
	}
	fn check_deviation_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0];
		let params = DeviationParams {
			period: Some(0),
			devtype: Some(0),
		};
		let input = DeviationInput::from_slice(&data, params);
		let res = deviation_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{test}] deviation should error with zero period");
		Ok(())
	}
	fn check_deviation_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0];
		let params = DeviationParams {
			period: Some(10),
			devtype: Some(0),
		};
		let input = DeviationInput::from_slice(&data, params);
		let res = deviation_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{test}] deviation should error if period > data.len()");
		Ok(())
	}
	fn check_deviation_very_small_dataset(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let single = [42.0];
		let params = DeviationParams {
			period: Some(9),
			devtype: Some(0),
		};
		let input = DeviationInput::from_slice(&single, params);
		let res = deviation_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{test}] deviation should error if not enough data");
		Ok(())
	}
	fn check_deviation_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
		let params = DeviationParams {
			period: Some(3),
			devtype: Some(0),
		};
		let input = DeviationInput::from_slice(&data, params);
		let res = deviation_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), data.len());
		for (i, &v) in res.values.iter().enumerate().skip(4) {
			assert!(!v.is_nan(), "[{test}] Unexpected NaN at out-index {}", i);
		}
		Ok(())
	}
	fn check_deviation_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0, 5.0];
		let period = 3;
		let devtype = 0;
		let input = DeviationInput::from_slice(
			&data,
			DeviationParams {
				period: Some(period),
				devtype: Some(devtype),
			},
		);
		let batch_output = deviation_with_kernel(&input, kernel)?.values;
		let mut stream = DeviationStream::try_new(DeviationParams {
			period: Some(period),
			devtype: Some(devtype),
		})?;
		let mut stream_values = Vec::with_capacity(data.len());
		for &val in &data {
			match stream.update(val) {
				Some(out_val) => stream_values.push(out_val),
				None => stream_values.push(f64::NAN),
			}
		}
		assert_eq!(batch_output.len(), stream_values.len());
		for (i, (b, s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			assert!(
				(b - s).abs() < 1e-9,
				"[{test}] streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				i,
				b,
				s,
				(b - s).abs()
			);
		}
		Ok(())
	}
	fn check_deviation_mean_absolute(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0, 5.0];
		let params = DeviationParams {
			period: Some(3),
			devtype: Some(1),
		};
		let input = DeviationInput::from_slice(&data, params);
		let result = deviation_with_kernel(&input, kernel)?;
		let expected = 2.0 / 3.0;
		for &val in &result.values[2..] {
			assert!(
				(val - expected).abs() < 1e-12,
				"[{test}] mean abs deviation mismatch: got {}, expected {}",
				val,
				expected
			);
		}
		Ok(())
	}
	fn check_deviation_median_absolute(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0, 5.0];
		let params = DeviationParams {
			period: Some(3),
			devtype: Some(2),
		};
		let input = DeviationInput::from_slice(&data, params);
		let result = deviation_with_kernel(&input, kernel)?;
		let expected = 1.0;
		for &val in &result.values[2..] {
			assert!(
				(val - expected).abs() < 1e-12,
				"[{test}] median abs deviation mismatch: got {}, expected {}",
				val,
				expected
			);
		}
		Ok(())
	}
	fn check_deviation_invalid_devtype(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0];
		let params = DeviationParams {
			period: Some(2),
			devtype: Some(999),
		};
		let input = DeviationInput::from_slice(&data, params);
		let result = deviation_with_kernel(&input, kernel);
		assert!(
			matches!(result, Err(DeviationError::InvalidDevType { .. })),
			"[{test}] invalid devtype should error"
		);
		Ok(())
	}

	macro_rules! generate_all_deviation_tests {
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
	generate_all_deviation_tests!(
		check_deviation_partial_params,
		check_deviation_accuracy,
		check_deviation_default_params,
		check_deviation_zero_period,
		check_deviation_period_exceeds_length,
		check_deviation_very_small_dataset,
		check_deviation_nan_handling,
		check_deviation_streaming,
		check_deviation_mean_absolute,
		check_deviation_median_absolute,
		check_deviation_invalid_devtype
	);
	
	#[cfg(test)]
	mod deviation_property_tests {
		use super::*;
		use proptest::prelude::*;
		
		proptest! {
			#[test]
			fn deviation_property_test(
				data in prop::collection::vec(prop::num::f64::ANY, 10..=1000),
				period in 2usize..=50,
				devtype in 0usize..=2
			) {
				// Skip if all data is NaN
				if data.iter().all(|x| x.is_nan()) {
					return Ok(());
				}
				
				// Skip if not enough valid data
				let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				if data.len() - first_valid < period {
					return Ok(());
				}
				
				let params = DeviationParams {
					period: Some(period),
					devtype: Some(devtype),
				};
				let input = DeviationInput::from_slice(&data, params);
				
				// Test that it doesn't panic
				let result = deviation(&input);
				
				// If successful, verify output length matches input
				if let Ok(output) = result {
					prop_assert_eq!(output.values.len(), data.len());
					
					// Verify NaN prefix
					for i in 0..(first_valid + period - 1).min(data.len()) {
						prop_assert!(output.values[i].is_nan());
					}
					
					// Verify non-NaN values after warmup (for valid devtypes)
					if devtype <= 2 {
						for i in (first_valid + period - 1)..data.len() {
							prop_assert!(!output.values[i].is_nan() || data[i].is_nan());
						}
					}
				}
			}
		}
	}

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0, 5.0];
		let output = DeviationBatchBuilder::new().kernel(kernel).apply_slice(&data)?;
		let def = DeviationParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), data.len());
		let single = DeviationInput::from_slice(&data, def.clone());
		let single_out = deviation_with_kernel(&single, kernel)?.values;
		for (i, (r, s)) in row.iter().zip(single_out.iter()).enumerate() {
			if r.is_nan() && s.is_nan() {
				continue;
			}
			assert!(
				(r - s).abs() < 1e-12,
				"[{test}] default-row batch mismatch at idx {}: {} vs {}",
				i,
				r,
				s
			);
		}
		Ok(())
	}
	fn check_batch_varying_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [1.0, 2.0, 3.0, 4.0, 5.0];
		let batch_output = DeviationBatchBuilder::new()
			.period_range(2, 3, 1)
			.devtype_range(0, 2, 1)
			.kernel(kernel)
			.apply_slice(&data)?;
		assert!(batch_output.rows >= 3, "[{test}] Not enough batch rows");
		for params in &batch_output.combos {
			let single = DeviationInput::from_slice(&data, params.clone());
			let single_out = deviation_with_kernel(&single, kernel)?.values;
			let row = batch_output.values_for(params).unwrap();
			for (i, (r, s)) in row.iter().zip(single_out.iter()).enumerate() {
				if r.is_nan() && s.is_nan() {
					continue;
				}
				assert!(
					(r - s).abs() < 1e-12,
					"[{test}] batch grid row mismatch at idx {}: {} vs {}",
					i,
					r,
					s
				);
			}
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
	gen_batch_tests!(check_batch_varying_params);
}

#[cfg(feature = "python")]
#[pyfunction(name = "deviation")]
#[pyo3(signature = (data, period, devtype, kernel=None))]
pub fn deviation_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	devtype: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = DeviationParams {
		period: Some(period),
		devtype: Some(devtype),
	};
	let input = DeviationInput::from_slice(slice_in, params);

	// Allocate output array first to avoid intermediate Vec
	let len = slice_in.len();
	let out_arr = unsafe { PyArray1::<f64>::new(py, len, false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	py.allow_threads(|| {
		// Use the into_slice version to write directly to numpy array
		deviation_into_slice(slice_out, &input, kern)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "DeviationStream")]
pub struct DeviationStreamPy {
	stream: DeviationStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DeviationStreamPy {
	#[new]
	fn new(period: usize, devtype: usize) -> PyResult<Self> {
		let params = DeviationParams {
			period: Some(period),
			devtype: Some(devtype),
		};
		let stream = DeviationStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(DeviationStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "deviation_batch")]
#[pyo3(signature = (data, period_range, devtype_range, kernel=None))]
pub fn deviation_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	devtype_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = DeviationBatchRange {
		period: period_range,
		devtype: devtype_range,
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	py.allow_threads(|| {
		let kernel = match kern {
			Kernel::Auto => detect_best_batch_kernel(),
			k => k,
		};
		let simd = match kernel {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			_ => kernel,
		};
		
		// Get batch result and copy to output
		let result = deviation_batch_inner(slice_in, &sweep, simd, true)?;
		slice_out.copy_from_slice(&result.values);
		Ok(())
	})
	.map_err(|e: DeviationError| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"devtypes",
		combos.iter()
			.map(|p| p.devtype.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = deviation_js)]
pub fn deviation_js(data: &[f64], period: usize, devtype: usize) -> Result<Vec<f64>, JsValue> {
	let params = DeviationParams {
		period: Some(period),
		devtype: Some(devtype),
	};
	let input = DeviationInput::from_slice(data, params);
	
	// Use helper for zero allocation - calculate warmup period
	let first_valid_idx = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);
	let warmup = first_valid_idx + period - 1;
	let mut output = alloc_with_nan_prefix(data.len(), warmup);
	
	deviation_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct DeviationBatchResult {
	values: Vec<f64>, // flattened array
	combos: usize,    // number of parameter combinations
	cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl DeviationBatchResult {
	#[wasm_bindgen(getter)]
	pub fn values(&self) -> Vec<f64> {
		self.values.clone()
	}

	#[wasm_bindgen(getter)]
	pub fn combos(&self) -> usize {
		self.combos
	}

	#[wasm_bindgen(getter)]
	pub fn cols(&self) -> usize {
		self.cols
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DeviationBatchConfig {
	pub period_range: (usize, usize, usize),
	pub devtype_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn deviation_batch(data: &[f64], config: JsValue) -> Result<DeviationBatchResult, JsValue> {
	let config: DeviationBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = DeviationBatchRange {
		period: config.period_range,
		devtype: config.devtype_range,
	};
	
	let result = deviation_batch_with_kernel(data, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(DeviationBatchResult {
		values: result.values,
		combos: result.rows,
		cols: result.cols,
	})
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn deviation_batch_metadata(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	devtype_start: usize,
	devtype_end: usize,
	devtype_step: usize,
) -> Vec<f64> {
	let sweep = DeviationBatchRange {
		period: (period_start, period_end, period_step),
		devtype: (devtype_start, devtype_end, devtype_step),
	};
	
	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len() * 2);
	
	for combo in combos {
		metadata.push(combo.period.unwrap() as f64);
		metadata.push(combo.devtype.unwrap() as f64);
	}
	
	metadata
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn deviation_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn deviation_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = deviation_into)]
pub unsafe fn deviation_into(data_ptr: *const f64, len: usize, period: usize, devtype: usize, out_ptr: *mut f64) -> Result<(), JsValue> {
	let data = std::slice::from_raw_parts(data_ptr, len);
	
	// Check for aliasing (if input and output point to same memory)
	let in_ptr = data_ptr as usize;
	let out_ptr_addr = out_ptr as usize;
	let size = len * std::mem::size_of::<f64>();
	
	let params = DeviationParams {
		period: Some(period),
		devtype: Some(devtype),
	};
	let input = DeviationInput::from_slice(data, params);
	
	if in_ptr == out_ptr_addr {
		// Same memory location, need temporary buffer
		let mut temp = vec![0.0; len];
		deviation_into_slice(&mut temp, &input, Kernel::Auto)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		let out = std::slice::from_raw_parts_mut(out_ptr, len);
		out.copy_from_slice(&temp);
	} else {
		// Different memory locations, safe to write directly
		let out = std::slice::from_raw_parts_mut(out_ptr, len);
		deviation_into_slice(out, &input, Kernel::Auto)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
	}
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn deviation_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	devtype_start: usize,
	devtype_end: usize,
	devtype_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to deviation_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = DeviationBatchRange {
			period: (period_start, period_end, period_step),
			devtype: (devtype_start, devtype_end, devtype_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		if rows == 0 {
			return Err(JsValue::from_str("No valid parameter combinations"));
		}

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		// Use the batch inner function directly into the output slice
		let warmup_periods: Vec<usize> = combos.iter()
			.map(|p| {
				let first_valid_idx = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);
				first_valid_idx + p.period.unwrap_or(1) - 1
			})
			.collect();

		// Initialize with NaN using a more direct approach
		for (row, &warmup) in warmup_periods.iter().enumerate() {
			let row_start = row * cols;
			for i in 0..warmup.min(cols) {
				out[row_start + i] = f64::NAN;
			}
		}

		// Process each parameter combination
		for (row, p) in combos.iter().enumerate() {
			let input = DeviationInput::from_slice(data, p.clone());
			let row_start = row * cols;
			let row_slice = &mut out[row_start..row_start + cols];
			
			// Only compute from warmup onwards to avoid overwriting NaN prefix
			let warmup = warmup_periods[row];
			if warmup < cols {
				deviation_into_slice(row_slice, &input, Kernel::Auto)
					.map_err(|e| JsValue::from_str(&e.to_string()))?;
			}
		}

		Ok(rows)
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct DeviationStream {
	inner: crate::indicators::deviation::DeviationStream,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl DeviationStream {
	#[wasm_bindgen(constructor)]
	pub fn new(period: usize, devtype: usize) -> Result<DeviationStream, JsValue> {
		let params = DeviationParams {
			period: Some(period),
			devtype: Some(devtype),
		};
		let inner = crate::indicators::deviation::DeviationStream::try_new(params)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		Ok(DeviationStream { inner })
	}
	
	#[wasm_bindgen]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.inner.update(value)
	}
}
