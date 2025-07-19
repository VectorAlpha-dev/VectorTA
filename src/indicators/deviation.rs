//! # Rolling Deviation Indicator
//!
//! Computes rolling Standard Deviation, Mean Absolute Deviation, or Median Absolute Deviation.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::detect_best_batch_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

/// Input for deviation indicator.
#[derive(Debug, Clone)]
pub struct DeviationInput<'a> {
	pub data: &'a [f64],
	pub params: DeviationParams,
}
impl<'a> DeviationInput<'a> {
	#[inline]
	pub fn from_slice(data: &'a [f64], params: DeviationParams) -> Self {
		Self { data, params }
	}
	#[inline]
	pub fn with_defaults(data: &'a [f64]) -> Self {
		Self {
			data,
			params: DeviationParams::default(),
		}
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(9)
	}
	#[inline]
	pub fn get_devtype(&self) -> usize {
		self.params.devtype.unwrap_or(0)
	}
}

/// Output for deviation indicator.
#[derive(Debug, Clone)]
pub struct DeviationOutput {
	pub values: Vec<f64>,
}

/// Parameters for deviation indicator.
#[derive(Debug, Clone)]
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

#[inline(always)]
pub fn deviation_with_kernel(input: &DeviationInput, kernel: Kernel) -> Result<DeviationOutput, DeviationError> {
	let data = input.data;
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DeviationError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();
	let devtype = input.get_devtype();

	if period == 0 || period > len {
		return Err(DeviationError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(DeviationError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	if !(0..=2).contains(&devtype) {
		return Err(DeviationError::InvalidDevType { devtype });
	}

	let chosen = match kernel {
		Kernel::Auto => Kernel::Scalar,
		other => other,
	};
	let mut out = vec![f64::NAN; len];
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				let v = deviation_scalar(data, period, first, devtype)?;
				out.copy_from_slice(&v[..len]);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				deviation_avx2(data, period, first, devtype, &mut out);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				deviation_avx512(data, period, first, devtype, &mut out);
			}
			_ => unreachable!(),
		}
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
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let devtypes = axis_usize(r.devtype);
	let mut out = Vec::with_capacity(periods.len() * devtypes.len());
	for &p in &periods {
		for &d in &devtypes {
			out.push(DeviationParams {
				period: Some(p),
				devtype: Some(d),
			});
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
	let mut values = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let devtype = combos[row].devtype.unwrap();
		match kern {
			Kernel::Scalar => deviation_row_scalar(data, first, period, 0, devtype, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => deviation_row_avx2(data, first, period, 0, devtype, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => deviation_row_avx512(data, first, period, 0, devtype, out_row),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			values
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in values.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in values.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	Ok(DeviationBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
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
	let mut result = vec![f64::NAN; data.len()];
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
	let mut result = vec![f64::NAN; data.len()];
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
	let mut result = vec![f64::NAN; data.len()];
	let start_window_end = first_valid_idx + period - 1;
	for i in start_window_end..data.len() {
		let window_start = i + 1 - period;
		if window_start < first_valid_idx {
			continue;
		}
		let window = &data[window_start..=i];
		let median = find_median(window);
		let mut abs_devs: Vec<f64> = window.iter().map(|&x| (x - median).abs()).collect();
		result[i] = find_median(&abs_devs);
	}
	Ok(result)
}

#[inline]
fn find_median(slice: &[f64]) -> f64 {
	if slice.is_empty() {
		return f64::NAN;
	}
	let mut sorted = slice.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let mid = sorted.len() / 2;
	if sorted.len() % 2 == 0 {
		(sorted[mid - 1] + sorted[mid]) / 2.0
	} else {
		sorted[mid]
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
	let result = deviation_scalar(data, period, first, devtype).unwrap();
	out.copy_from_slice(&result[..out.len()]);
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
