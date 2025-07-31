//! # Linear Regression Angle (LRA)
//!
//! Computes the angle (in degrees) of the linear regression line for a given period.
//! Follows ALMA-style API for compatibility, SIMD-stubbed, with streaming and batch mode.
//!
//! ## Parameters
//! - **period**: Window size (number of data points), defaults to 14.
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are `NaN`.
//! - **InvalidPeriod**: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data for `period`.
//!
//! ## Returns
//! - **`Ok(Linearreg_angleOutput)`** on success with `.values` field.
//! - **`Err(Linearreg_angleError)`** otherwise.

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
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum Linearreg_angleData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for Linearreg_angleInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			Linearreg_angleData::Slice(slice) => slice,
			Linearreg_angleData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct Linearreg_angleParams {
	pub period: Option<usize>,
}

impl Default for Linearreg_angleParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleInput<'a> {
	pub data: Linearreg_angleData<'a>,
	pub params: Linearreg_angleParams,
}

impl<'a> Linearreg_angleInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: Linearreg_angleParams) -> Self {
		Self {
			data: Linearreg_angleData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: Linearreg_angleParams) -> Self {
		Self {
			data: Linearreg_angleData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", Linearreg_angleParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct Linearreg_angleBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for Linearreg_angleBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl Linearreg_angleBuilder {
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
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
		let p = Linearreg_angleParams { period: self.period };
		let i = Linearreg_angleInput::from_candles(c, "close", p);
		linearreg_angle_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
		let p = Linearreg_angleParams { period: self.period };
		let i = Linearreg_angleInput::from_slice(d, p);
		linearreg_angle_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<Linearreg_angleStream, Linearreg_angleError> {
		let p = Linearreg_angleParams { period: self.period };
		Linearreg_angleStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum Linearreg_angleError {
	#[error("linearreg_angle: All values are NaN.")]
	AllValuesNaN,
	#[error("linearreg_angle: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("linearreg_angle: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("linearreg_angle: Empty data slice.")]
	EmptyData,
}

#[cfg(feature = "wasm")]
impl From<Linearreg_angleError> for JsValue {
	fn from(err: Linearreg_angleError) -> Self {
		JsValue::from_str(&err.to_string())
	}
}

#[inline]
pub fn linearreg_angle(input: &Linearreg_angleInput) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
	linearreg_angle_with_kernel(input, Kernel::Auto)
}

pub fn linearreg_angle_with_kernel(
	input: &Linearreg_angleInput,
	kernel: Kernel,
) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(Linearreg_angleError::EmptyData);
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(Linearreg_angleError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(Linearreg_angleError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(Linearreg_angleError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut out = alloc_with_nan_prefix(len, first + period - 1);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => linearreg_angle_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => linearreg_angle_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => linearreg_angle_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(Linearreg_angleOutput { values: out })
}

#[inline]
pub fn linearreg_angle_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	let sum_x = (period * (period - 1)) as f64 * 0.5;
	let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
	let divisor = sum_x * sum_x - (period as f64) * sum_x_sqr;
	let n = data.len();

	// Use sliding window approach to avoid allocating arrays proportional to input size
	for i in (first_valid + period - 1)..n {
		let start = i + 1 - period;
		let end = i + 1;
		
		// Calculate sum_y for the window
		let mut sum_y = 0.0;
		for j in start..end {
			sum_y += data[j];
		}
		
		// Calculate sum_xy = sum(i * y[i]) for the window
		// We need sum_kd = sum((absolute_index) * data[absolute_index]) for window
		let mut sum_kd = 0.0;
		for j in start..end {
			sum_kd += (j as f64) * data[j];
		}
		
		let sum_xy = (i as f64) * sum_y - sum_kd;
		let slope = ((period as f64) * sum_xy - sum_x * sum_y) / divisor;
		out[i] = slope.atan() * (180.0 / PI);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { linearreg_angle_avx512_short(data, period, first_valid, out) }
	} else {
		unsafe { linearreg_angle_avx512_long(data, period, first_valid, out) }
	}
}

#[inline]
pub fn linearreg_angle_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	linearreg_angle_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	linearreg_angle_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	linearreg_angle_scalar(data, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	sum_x: f64,
	sum_x_sqr: f64,
	divisor: f64,
	count: usize,  // Track total values seen for correct indexing
}

impl Linearreg_angleStream {
	pub fn try_new(params: Linearreg_angleParams) -> Result<Self, Linearreg_angleError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(Linearreg_angleError::InvalidPeriod { period, data_len: 0 });
		}

		let sum_x = (period * (period - 1)) as f64 * 0.5;
		let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
		let divisor = sum_x * sum_x - (period as f64) * sum_x_sqr;

		Ok(Self {
			period,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
			sum_x,
			sum_x_sqr,
			divisor,
			count: 0,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;
		self.count += 1;

		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		
		// Calculate sum_y and sum_kd for the window
		let mut sum_y = 0.0;
		let mut sum_kd = 0.0;
		
		// The buffer contains the last 'period' values in circular order
		// We need to calculate the proper indices for the linear regression
		let start_idx = self.count - self.period;
		
		for j in 0..self.period {
			let buf_idx = (self.head + j) % self.period;
			let actual_idx = start_idx + j;
			sum_y += self.buffer[buf_idx];
			sum_kd += (actual_idx as f64) * self.buffer[buf_idx];
		}
		
		let current_idx = self.count - 1;
		let sum_xy = (current_idx as f64) * sum_y - sum_kd;
		let slope = ((self.period as f64) * sum_xy - self.sum_x * sum_y) / self.divisor;
		Some(slope.atan() * (180.0 / PI))
	}
}

#[derive(Clone, Debug)]
pub struct Linearreg_angleBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for Linearreg_angleBatchRange {
	fn default() -> Self {
		Self { period: (14, 60, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct Linearreg_angleBatchBuilder {
	range: Linearreg_angleBatchRange,
	kernel: Kernel,
}

impl Linearreg_angleBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
		linearreg_angle_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
		Linearreg_angleBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
		Linearreg_angleBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

pub fn linearreg_angle_batch_with_kernel(
	data: &[f64],
	sweep: &Linearreg_angleBatchRange,
	k: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(Linearreg_angleError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	linearreg_angle_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct Linearreg_angleBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<Linearreg_angleParams>,
	pub rows: usize,
	pub cols: usize,
}

impl Linearreg_angleBatchOutput {
	pub fn row_for_params(&self, p: &Linearreg_angleParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &Linearreg_angleParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &Linearreg_angleBatchRange) -> Vec<Linearreg_angleParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(Linearreg_angleParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn linearreg_angle_batch_slice(
	data: &[f64],
	sweep: &Linearreg_angleBatchRange,
	kern: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
	linearreg_angle_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linearreg_angle_batch_par_slice(
	data: &[f64],
	sweep: &Linearreg_angleBatchRange,
	kern: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
	linearreg_angle_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linearreg_angle_batch_inner(
	data: &[f64],
	sweep: &Linearreg_angleBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(Linearreg_angleError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(Linearreg_angleError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(Linearreg_angleError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory helpers to avoid allocation
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each row
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => linearreg_angle_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => linearreg_angle_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => linearreg_angle_row_avx512(data, first, period, out_row),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	
	// Convert uninitialized memory to Vec
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	core::mem::forget(buf_guard);
	
	Ok(Linearreg_angleBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn linearreg_angle_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_angle_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_angle_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		linearreg_angle_row_avx512_short(data, first, period, out)
	} else {
		linearreg_angle_row_avx512_long(data, first, period, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_angle_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_angle_scalar(data, period, first, out)
}

#[inline(always)]
fn expand_grid_lra(r: &Linearreg_angleBatchRange) -> Vec<Linearreg_angleParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	axis_usize(r.period)
		.into_iter()
		.map(|p| Linearreg_angleParams { period: Some(p) })
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_lra_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = Linearreg_angleParams { period: None };
		let input = Linearreg_angleInput::from_candles(&candles, "close", default_params);
		let output = linearreg_angle_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_lra_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = Linearreg_angleParams { period: Some(14) };
		let input = Linearreg_angleInput::from_candles(&candles, "close", params);
		let result = linearreg_angle_with_kernel(&input, kernel)?;

		let expected_last_five = [
			-89.30491945492733,
			-89.28911257342405,
			-89.1088041965075,
			-86.58419429159467,
			-87.77085937059316,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-5,
				"[{}] LRA {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_lra_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = Linearreg_angleParams { period: Some(0) };
		let input = Linearreg_angleInput::from_slice(&input_data, params);
		let res = linearreg_angle_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] LRA should fail with zero period", test_name);
		Ok(())
	}

	fn check_lra_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = Linearreg_angleParams { period: Some(10) };
		let input = Linearreg_angleInput::from_slice(&data_small, params);
		let res = linearreg_angle_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] LRA should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_lra_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = Linearreg_angleParams { period: Some(14) };
		let input = Linearreg_angleInput::from_slice(&single_point, params);
		let res = linearreg_angle_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] LRA should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_lra_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = Linearreg_angleParams { period: Some(14) };
		let first_input = Linearreg_angleInput::from_candles(&candles, "close", first_params);
		let first_result = linearreg_angle_with_kernel(&first_input, kernel)?;

		let second_params = Linearreg_angleParams { period: Some(14) };
		let second_input = Linearreg_angleInput::from_slice(&first_result.values, second_params);
		let second_result = linearreg_angle_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_lra_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations for linearreg_angle
		let test_params = vec![
			Linearreg_angleParams::default(),                    // period: 14 (default)
			Linearreg_angleParams { period: Some(2) },          // minimum viable period
			Linearreg_angleParams { period: Some(3) },          // very small period
			Linearreg_angleParams { period: Some(5) },          // small period
			Linearreg_angleParams { period: Some(7) },          // small period
			Linearreg_angleParams { period: Some(10) },         // small-medium period
			Linearreg_angleParams { period: Some(14) },         // default period
			Linearreg_angleParams { period: Some(20) },         // medium period
			Linearreg_angleParams { period: Some(30) },         // medium period
			Linearreg_angleParams { period: Some(50) },         // medium-large period
			Linearreg_angleParams { period: Some(100) },        // large period
			Linearreg_angleParams { period: Some(200) },        // very large period
			Linearreg_angleParams { period: Some(500) },        // extremely large period
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = Linearreg_angleInput::from_candles(&candles, "close", params.clone());
			let output = linearreg_angle_with_kernel(&input, kernel)?;
			
			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14),
						param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_lra_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_lra_tests {
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
	generate_all_lra_tests!(
		check_lra_partial_params,
		check_lra_accuracy,
		check_lra_zero_period,
		check_lra_period_exceeds_length,
		check_lra_very_small_dataset,
		check_lra_reinput,
		check_lra_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = Linearreg_angleBatchBuilder::new()
			.kernel(kernel)
			.apply_candles(&c, "close")?;

		let def = Linearreg_angleParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			-89.30491945492733,
			-89.28911257342405,
			-89.1088041965075,
			-86.58419429159467,
			-87.77085937059316,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-5,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	fn check_batch_grid_search(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let batch = Linearreg_angleBatchBuilder::new()
			.kernel(kernel)
			.period_range(10, 16, 2)
			.apply_candles(&c, "close")?;

		// Should have periods: 10, 12, 14, 16
		let periods = [10, 12, 14, 16];
		assert_eq!(batch.rows, 4);

		for (ix, p) in periods.iter().enumerate() {
			let param = Linearreg_angleParams { period: Some(*p) };
			let row_idx = batch.row_for_params(&param);
			assert_eq!(row_idx, Some(ix), "Batch grid missing period {p}");
			let row = batch.values_for(&param).expect("Missing row for period");
			assert_eq!(row.len(), batch.cols, "Row len mismatch for period {p}");
		}
		Ok(())
	}

	fn check_batch_period_static(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let batch = Linearreg_angleBatchBuilder::new()
			.kernel(kernel)
			.period_static(14)
			.apply_candles(&c, "close")?;

		assert_eq!(batch.rows, 1);
		let param = Linearreg_angleParams { period: Some(14) };
		let row = batch.values_for(&param).expect("Missing static row");
		assert_eq!(row.len(), batch.cols);

		// Check a value
		let last = *row.last().unwrap();
		let expected = -87.77085937059316;
		assert!(
			(last - expected).abs() < 1e-5,
			"Static period row last val mismatch: got {last}, want {expected}"
		);

		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations for linearreg_angle
		let test_configs = vec![
			// (period_start, period_end, period_step)
			(2, 10, 2),        // Small periods, step 2
			(5, 15, 1),        // Dense small range
			(10, 50, 10),      // Medium periods, step 10
			(20, 100, 20),     // Large periods, step 20
			(50, 200, 50),     // Very large periods
			(14, 14, 0),       // Single period (default)
			(2, 5, 1),         // Very small dense range
			(100, 500, 100),   // Extremely large periods
			(7, 21, 7),        // Weekly-based periods
			(30, 90, 30),      // Monthly-based periods
		];
		
		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let mut builder = Linearreg_angleBatchBuilder::new()
				.kernel(kernel);
			
			// Configure period range
			if p_step > 0 {
				builder = builder.period_range(p_start, p_end, p_step);
			} else {
				builder = builder.period_static(p_start);
			}
			
			let output = builder.apply_candles(&c, "close")?;
			
			for (idx, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];
				
				// Check all three poison patterns with detailed context
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, 
						combo.period.unwrap_or(14)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
				}
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
	gen_batch_tests!(check_batch_grid_search);
	gen_batch_tests!(check_batch_period_static);
	gen_batch_tests!(check_batch_no_poison);
}

/// Write linearreg_angle directly to output slice - no allocations
pub fn linearreg_angle_into_slice(
	dst: &mut [f64],
	input: &Linearreg_angleInput,
	kern: Kernel,
) -> Result<(), Linearreg_angleError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(Linearreg_angleError::EmptyData);
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(Linearreg_angleError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(Linearreg_angleError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(Linearreg_angleError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	if dst.len() != len {
		return Err(Linearreg_angleError::InvalidPeriod {
			period: dst.len(),
			data_len: len,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => linearreg_angle_scalar(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => linearreg_angle_avx2(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => linearreg_angle_avx512(data, period, first, dst),
			_ => unreachable!(),
		}
	}

	Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_angle")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn linearreg_angle_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = Linearreg_angleParams { period: Some(period) };
	let linearreg_angle_in = Linearreg_angleInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| linearreg_angle_with_kernel(&linearreg_angle_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "Linearreg_angleStream")]
pub struct Linearreg_angleStreamPy {
	stream: Linearreg_angleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl Linearreg_angleStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = Linearreg_angleParams { period: Some(period) };
		let stream = Linearreg_angleStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(Linearreg_angleStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_angle_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn linearreg_angle_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = Linearreg_angleBatchRange { period: period_range };

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	let kern = validate_kernel(kernel, true)?;

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
			linearreg_angle_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos
			.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[inline(always)]
fn linearreg_angle_batch_inner_into(
	data: &[f64],
	sweep: &Linearreg_angleBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<Linearreg_angleParams>, Linearreg_angleError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(Linearreg_angleError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(Linearreg_angleError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(Linearreg_angleError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => linearreg_angle_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => linearreg_angle_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => linearreg_angle_row_avx512(data, first, period, out_row),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	Ok(combos)
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = Linearreg_angleParams { period: Some(period) };
	let input = Linearreg_angleInput::from_slice(data, params);

	let mut output = vec![0.0; data.len()];
	linearreg_angle_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = Linearreg_angleParams { period: Some(period) };
		let input = Linearreg_angleInput::from_slice(data, params);

		if in_ptr == out_ptr {
			// Aliasing detected - use temp buffer
			let mut temp = vec![0.0; len];
			linearreg_angle_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing - write directly
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			linearreg_angle_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct Linearreg_angleBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct Linearreg_angleBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<Linearreg_angleParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = linearreg_angle_batch)]
pub fn linearreg_angle_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: Linearreg_angleBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = Linearreg_angleBatchRange {
		period: config.period_range,
	};

	let output = linearreg_angle_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = Linearreg_angleBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
