//! # Linear Regression Intercept (LINEARREG_INTERCEPT)
//!
//! Calculates the y-value of the linear regression line at the last point
//! of each regression window. Effectively gives the "intercept" if the last bar
//! in each window is the reference point.
//!
//! ## Parameters
//! - **period**: Window size (number of data points), default 14
//!
//! ## Errors
//! - **AllValuesNaN**: linearreg_intercept: All input data values are `NaN`.
//! - **InvalidPeriod**: linearreg_intercept: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: linearreg_intercept: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(LinearRegInterceptOutput)`** on success, containing a `Vec<f64>` matching the input.
//! - **`Err(LinearRegInterceptError)`** otherwise.

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
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// ---- Input data structures ----

impl<'a> AsRef<[f64]> for LinearRegInterceptInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			LinearRegInterceptData::Slice(slice) => slice,
			LinearRegInterceptData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum LinearRegInterceptData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptParams {
	pub period: Option<usize>,
}

impl Default for LinearRegInterceptParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptInput<'a> {
	pub data: LinearRegInterceptData<'a>,
	pub params: LinearRegInterceptParams,
}

impl<'a> LinearRegInterceptInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: LinearRegInterceptParams) -> Self {
		Self {
			data: LinearRegInterceptData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: LinearRegInterceptParams) -> Self {
		Self {
			data: LinearRegInterceptData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", LinearRegInterceptParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

// ---- Builder pattern ----

#[derive(Copy, Clone, Debug)]
pub struct LinearRegInterceptBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for LinearRegInterceptBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl LinearRegInterceptBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
		let p = LinearRegInterceptParams { period: self.period };
		let i = LinearRegInterceptInput::from_candles(c, "close", p);
		linearreg_intercept_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
		let p = LinearRegInterceptParams { period: self.period };
		let i = LinearRegInterceptInput::from_slice(d, p);
		linearreg_intercept_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<LinearRegInterceptStream, LinearRegInterceptError> {
		let p = LinearRegInterceptParams { period: self.period };
		LinearRegInterceptStream::try_new(p)
	}
}

// ---- Error type ----

#[derive(Debug, Error)]
pub enum LinearRegInterceptError {
	#[error("linearreg_intercept: All values are NaN.")]
	AllValuesNaN,
	#[error("linearreg_intercept: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("linearreg_intercept: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

// ---- Main entrypoints ----

#[inline]
pub fn linearreg_intercept(
	input: &LinearRegInterceptInput,
) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
	linearreg_intercept_with_kernel(input, Kernel::Auto)
}

pub fn linearreg_intercept_with_kernel(
	input: &LinearRegInterceptInput,
	kernel: Kernel,
) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
	let data: &[f64] = input.as_ref();

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(LinearRegInterceptError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(LinearRegInterceptError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(LinearRegInterceptError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let mut out = alloc_with_nan_prefix(len, period - 1);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => linearreg_intercept_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => linearreg_intercept_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => linearreg_intercept_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(LinearRegInterceptOutput { values: out })
}

/// Write directly to output slice - no allocations (WASM helper)
#[inline]
pub fn linearreg_intercept_into_slice(
	dst: &mut [f64], 
	input: &LinearRegInterceptInput, 
	kern: Kernel
) -> Result<(), LinearRegInterceptError> {
	let data: &[f64] = input.as_ref();
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(LinearRegInterceptError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(LinearRegInterceptError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(LinearRegInterceptError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	
	if dst.len() != data.len() {
		return Err(LinearRegInterceptError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => linearreg_intercept_scalar(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => linearreg_intercept_avx2(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => linearreg_intercept_avx512(data, period, first, dst),
			_ => unreachable!(),
		}
	}
	
	// Fill warmup with NaN
	for v in &mut dst[..period - 1] {
		*v = f64::NAN;
	}
	
	Ok(())
}

#[inline]
pub fn linearreg_intercept_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	let n = period as f64;
	let mut sum_x = 0.0;
	let mut sum_x2 = 0.0;
	for i in 0..period {
		let xi = (i + 1) as f64;
		sum_x += xi;
		sum_x2 += xi * xi;
	}
	let denom = n * sum_x2 - sum_x * sum_x;
	if denom.abs() < f64::EPSILON {
		return;
	}
	let bd = 1.0 / denom;

	let mut sum_y = 0.0;
	let mut sum_xy = 0.0;
	for i in 0..(period - 1) {
		let val = data[first_val + i];
		let xi = (i + 1) as f64;
		sum_y += val;
		sum_xy += val * xi;
	}

	let p_idx = period as f64;

	for i in (first_val + period - 1)..data.len() {
		let val = data[i];
		sum_y += val;
		sum_xy += val * p_idx;

		let b = (n * sum_xy - sum_x * sum_y) * bd;
		let a = (sum_y - b * sum_x) / n;
		out[i] = a + b;

		let remove_idx = i as isize - (period as isize) + 1;
		if remove_idx >= 0 && (remove_idx as usize) < data.len() {
			let old_val = data[remove_idx as usize];
			sum_xy -= sum_y;
			sum_y -= old_val;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_intercept_avx512(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { linearreg_intercept_avx512_short(data, period, first_val, out) }
	} else {
		unsafe { linearreg_intercept_avx512_long(data, period, first_val, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_intercept_avx2(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	unsafe { linearreg_intercept_scalar(data, period, first_val, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn linearreg_intercept_avx512_short(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	linearreg_intercept_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn linearreg_intercept_avx512_long(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	linearreg_intercept_scalar(data, period, first_val, out)
}

// ---- Streaming struct ----

#[derive(Debug, Clone)]
pub struct LinearRegInterceptStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	sum_x: f64,
	sum_x2: f64,
	n: f64,
	bd: f64,
	sum_y: f64,
	sum_xy: f64,
}

impl LinearRegInterceptStream {
	pub fn try_new(params: LinearRegInterceptParams) -> Result<Self, LinearRegInterceptError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(LinearRegInterceptError::InvalidPeriod { period, data_len: 0 });
		}
		let mut sum_x = 0.0;
		let mut sum_x2 = 0.0;
		for i in 0..period {
			let xi = (i + 1) as f64;
			sum_x += xi;
			sum_x2 += xi * xi;
		}
		let n = period as f64;
		let denom = n * sum_x2 - sum_x * sum_x;
		if denom.abs() < f64::EPSILON {
			return Err(LinearRegInterceptError::InvalidPeriod { period, data_len: 0 });
		}
		let bd = 1.0 / denom;

		Ok(Self {
			period,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
			sum_x,
			sum_x2,
			n,
			bd,
			sum_y: 0.0,
			sum_xy: 0.0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		let tail_idx = self.head;
		let prev = self.buffer[tail_idx];
		self.buffer[tail_idx] = value;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
			// first fill: recalc everything
			self.sum_y = self.buffer.iter().sum();
			self.sum_xy = self.buffer.iter().enumerate().map(|(i, v)| v * ((i + 1) as f64)).sum();
		} else if self.filled {
			self.sum_y += value - prev;
			self.sum_xy += value * (self.period as f64) - self.sum_y;
		} else {
			self.sum_y += value;
			self.sum_xy += value * (self.head as f64);
		}

		if !self.filled {
			return None;
		}

		let b = (self.n * self.sum_xy - self.sum_x * self.sum_y) * self.bd;
		let a = (self.sum_y - b * self.sum_x) / self.n;
		Some(a + b)
	}
}

// ---- Batch range & builder ----

#[derive(Clone, Debug)]
pub struct LinearRegInterceptBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for LinearRegInterceptBatchRange {
	fn default() -> Self {
		Self { period: (14, 200, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct LinearRegInterceptBatchBuilder {
	range: LinearRegInterceptBatchRange,
	kernel: Kernel,
}

impl LinearRegInterceptBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
		linearreg_intercept_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(
		data: &[f64],
		k: Kernel,
	) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
		LinearRegInterceptBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(
		self,
		c: &Candles,
		src: &str,
	) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
		LinearRegInterceptBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

pub fn linearreg_intercept_batch_with_kernel(
	data: &[f64],
	sweep: &LinearRegInterceptBatchRange,
	k: Kernel,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(LinearRegInterceptError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	linearreg_intercept_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct LinearRegInterceptBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<LinearRegInterceptParams>,
	pub rows: usize,
	pub cols: usize,
}
impl LinearRegInterceptBatchOutput {
	pub fn row_for_params(&self, p: &LinearRegInterceptParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &LinearRegInterceptParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

// ---- Batch helpers ----

#[inline(always)]
fn expand_grid(r: &LinearRegInterceptBatchRange) -> Vec<LinearRegInterceptParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(LinearRegInterceptParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn linearreg_intercept_batch_slice(
	data: &[f64],
	sweep: &LinearRegInterceptBatchRange,
	kern: Kernel,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
	linearreg_intercept_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linearreg_intercept_batch_par_slice(
	data: &[f64],
	sweep: &LinearRegInterceptBatchRange,
	kern: Kernel,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
	linearreg_intercept_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linearreg_intercept_batch_inner_into(
	data: &[f64],
	sweep: &LinearRegInterceptBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<LinearRegInterceptParams>, LinearRegInterceptError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(LinearRegInterceptError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(LinearRegInterceptError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(LinearRegInterceptError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let cols = data.len();

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => linearreg_intercept_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => linearreg_intercept_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => linearreg_intercept_row_avx512(data, first, period, out_row),
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

	Ok(combos)
}

#[inline(always)]
fn linearreg_intercept_batch_inner(
	data: &[f64],
	sweep: &LinearRegInterceptBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(LinearRegInterceptError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(LinearRegInterceptError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(LinearRegInterceptError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	let mut buf_mu = make_uninit_matrix(rows, cols);

	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	let mut values = unsafe {
		let ptr = buf_mu.as_mut_ptr() as *mut f64;
		std::mem::forget(buf_mu);
		Vec::from_raw_parts(ptr, rows * cols, rows * cols)
	};
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => linearreg_intercept_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => linearreg_intercept_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => linearreg_intercept_row_avx512(data, first, period, out_row),
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

	Ok(LinearRegInterceptBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn linearreg_intercept_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_intercept_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_intercept_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_intercept_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn linearreg_intercept_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_intercept_avx512(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn linearreg_intercept_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_intercept_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn linearreg_intercept_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_intercept_avx512_long(data, period, first, out)
}

#[inline(always)]
fn expand_grid_reg(r: &LinearRegInterceptBatchRange) -> Vec<LinearRegInterceptParams> {
	expand_grid(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_linreg_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = LinearRegInterceptParams { period: None };
		let input = LinearRegInterceptInput::from_candles(&candles, "close", default_params);
		let output = linearreg_intercept_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_linreg_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = LinearRegInterceptInput::from_candles(&candles, "close", LinearRegInterceptParams::default());
		let result = linearreg_intercept_with_kernel(&input, kernel)?;
		let expected_last_five = [
			60000.91428571429,
			59947.142857142855,
			59754.57142857143,
			59318.4,
			59321.91428571429,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] LinReg {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_linreg_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = LinearRegInterceptInput::with_default_candles(&candles);
		match input.data {
			LinearRegInterceptData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected LinearRegInterceptData::Candles"),
		}
		let output = linearreg_intercept_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_linreg_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = LinearRegInterceptParams { period: Some(0) };
		let input = LinearRegInterceptInput::from_slice(&input_data, params);
		let res = linearreg_intercept_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] LinReg should fail with zero period", test_name);
		Ok(())
	}

	fn check_linreg_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = LinearRegInterceptParams { period: Some(10) };
		let input = LinearRegInterceptInput::from_slice(&data_small, params);
		let res = linearreg_intercept_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] LinReg should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_linreg_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = LinearRegInterceptParams { period: Some(14) };
		let input = LinearRegInterceptInput::from_slice(&single_point, params);
		let res = linearreg_intercept_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] LinReg should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_linreg_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = LinearRegInterceptParams { period: Some(14) };
		let first_input = LinearRegInterceptInput::from_candles(&candles, "close", first_params);
		let first_result = linearreg_intercept_with_kernel(&first_input, kernel)?;
		let second_params = LinearRegInterceptParams { period: Some(14) };
		let second_input = LinearRegInterceptInput::from_slice(&first_result.values, second_params);
		let second_result = linearreg_intercept_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());

		let start = second_result
			.values
			.iter()
			.position(|v| !v.is_nan())
			.unwrap_or(second_result.values.len());

		for (i, v) in second_result.values[start..].iter().enumerate() {
			assert!(
				!v.is_nan(),
				"[{}] Unexpected NaN at index {} after reinput",
				test_name,
				start + i
			);
		}
		Ok(())
	}

	fn check_linreg_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input =
			LinearRegInterceptInput::from_candles(&candles, "close", LinearRegInterceptParams { period: Some(14) });
		let res = linearreg_intercept_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 40 {
			for (i, &val) in res.values[40..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					40 + i
				);
			}
		}
		Ok(())
	}

	macro_rules! generate_all_linreg_tests {
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

	generate_all_linreg_tests!(
		check_linreg_partial_params,
		check_linreg_accuracy,
		check_linreg_default_candles,
		check_linreg_zero_period,
		check_linreg_period_exceeds_length,
		check_linreg_very_small_dataset,
		check_linreg_reinput,
		check_linreg_nan_handling
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = LinearRegInterceptBatchBuilder::new()
			.kernel(kernel)
			.apply_candles(&c, "close")?;

		let def = LinearRegInterceptParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			60000.91428571429,
			59947.142857142855,
			59754.57142857143,
			59318.4,
			59321.91428571429,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
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
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_intercept")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn linearreg_intercept_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = LinearRegInterceptParams { period: Some(period) };
	let input = LinearRegInterceptInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| linearreg_intercept_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "LinearRegInterceptStream")]
pub struct LinearRegInterceptStreamPy {
	stream: LinearRegInterceptStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LinearRegInterceptStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = LinearRegInterceptParams { period: Some(period) };
		let stream = LinearRegInterceptStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(LinearRegInterceptStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_intercept_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn linearreg_intercept_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = LinearRegInterceptBatchRange { period: period_range };

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
			linearreg_intercept_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

// ============= WASM API =============

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_intercept_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = LinearRegInterceptParams { period: Some(period) };
	let input = LinearRegInterceptInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];
	linearreg_intercept_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_intercept_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_intercept_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_intercept_into(
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
		let params = LinearRegInterceptParams { period: Some(period) };
		let input = LinearRegInterceptInput::from_slice(data, params);
		
		if in_ptr == out_ptr {  // CRITICAL: Aliasing check
			let mut temp = vec![0.0; len];
			linearreg_intercept_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			linearreg_intercept_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LinearRegInterceptBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LinearRegInterceptBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<LinearRegInterceptParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = linearreg_intercept_batch)]
pub fn linearreg_intercept_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: LinearRegInterceptBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = LinearRegInterceptBatchRange {
		period: config.period_range,
	};
	
	let batch_output = linearreg_intercept_batch_with_kernel(data, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let result = LinearRegInterceptBatchJsOutput {
		values: batch_output.values,
		combos: batch_output.combos,
		rows: batch_output.values.len() / data.len(),
		cols: data.len(),
	};
	
	serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_intercept_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let sweep = LinearRegInterceptBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let total_size = rows * len;
		
		if in_ptr == out_ptr {
			let mut temp = vec![0.0; total_size];
			linearreg_intercept_batch_inner_into(data, &sweep, Kernel::Auto, true, &mut temp)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
			linearreg_intercept_batch_inner_into(data, &sweep, Kernel::Auto, true, out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}
