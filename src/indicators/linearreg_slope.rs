//! # Linear Regression Slope
//!
//! Computes the slope (coefficient `b`) of the linear regression line over a moving window.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 14.
//!
//! ## Errors
//! - **EmptyData**: linearreg_slope: Input data slice is empty.
//! - **InvalidPeriod**: linearreg_slope: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: linearreg_slope: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
//! - **AllValuesNaN**: linearreg_slope: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(LinearRegSlopeOutput)`** on success, containing a `Vec<f64>` matching the input length, with leading `NaN`s until the window is filled.
//! - **`Err(LinearRegSlopeError)`** otherwise.

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
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for LinearRegSlopeInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			LinearRegSlopeData::Slice(slice) => slice,
			LinearRegSlopeData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum LinearRegSlopeData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinearRegSlopeOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct LinearRegSlopeParams {
	pub period: Option<usize>,
}

impl Default for LinearRegSlopeParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct LinearRegSlopeInput<'a> {
	pub data: LinearRegSlopeData<'a>,
	pub params: LinearRegSlopeParams,
}

impl<'a> LinearRegSlopeInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: LinearRegSlopeParams) -> Self {
		Self {
			data: LinearRegSlopeData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: LinearRegSlopeParams) -> Self {
		Self {
			data: LinearRegSlopeData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", LinearRegSlopeParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct LinearRegSlopeBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for LinearRegSlopeBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl LinearRegSlopeBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<LinearRegSlopeOutput, LinearRegSlopeError> {
		let p = LinearRegSlopeParams { period: self.period };
		let i = LinearRegSlopeInput::from_candles(c, "close", p);
		linearreg_slope_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<LinearRegSlopeOutput, LinearRegSlopeError> {
		let p = LinearRegSlopeParams { period: self.period };
		let i = LinearRegSlopeInput::from_slice(d, p);
		linearreg_slope_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<LinearRegSlopeStream, LinearRegSlopeError> {
		let p = LinearRegSlopeParams { period: self.period };
		LinearRegSlopeStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum LinearRegSlopeError {
	#[error("linearreg_slope: Empty data provided.")]
	EmptyData,
	#[error("linearreg_slope: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("linearreg_slope: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("linearreg_slope: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn linearreg_slope(input: &LinearRegSlopeInput) -> Result<LinearRegSlopeOutput, LinearRegSlopeError> {
	linearreg_slope_with_kernel(input, Kernel::Auto)
}

pub fn linearreg_slope_with_kernel(
	input: &LinearRegSlopeInput,
	kernel: Kernel,
) -> Result<LinearRegSlopeOutput, LinearRegSlopeError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(LinearRegSlopeError::EmptyData);
	}
	let period = input.get_period();
	if period == 0 || period > data.len() {
		return Err(LinearRegSlopeError::InvalidPeriod {
			period,
			data_len: data.len(),
		});
	}
	let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err(LinearRegSlopeError::AllValuesNaN),
	};
	if (data.len() - first_valid_idx) < period {
		return Err(LinearRegSlopeError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first_valid_idx,
		});
	}
	let mut out = alloc_with_nan_prefix(data.len(), first_valid_idx + period - 1);
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => linearreg_slope_scalar(data, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => linearreg_slope_avx2(data, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => linearreg_slope_avx512(data, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(LinearRegSlopeOutput { values: out })
}

#[inline]
pub fn linearreg_slope_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	let n = period as f64;
	let sum_x = (period - 1) as f64 * n / 2.0;
	let sum_x2 = (period - 1) as f64 * n * (2.0 * (period - 1) as f64 + 1.0) / 6.0;
	let denominator = n * sum_x2 - sum_x * sum_x;
	
	if denominator.abs() < f64::EPSILON {
		// Degenerate case - all slopes are NaN
		for i in (first + period - 1)..data.len() {
			out[i] = f64::NAN;
		}
		return;
	}

	for i in (first + period - 1)..data.len() {
		let start = i + 1 - period;
		
		// Calculate sum_y and sum_xy directly for this window
		let mut sum_y = 0.0;
		let mut sum_xy = 0.0;
		
		for j in 0..period {
			let y = data[start + j];
			sum_y += y;
			sum_xy += (j as f64) * y;
		}
		
		let numerator = n * sum_xy - sum_x * sum_y;
		out[i] = numerator / denominator;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_slope_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	unsafe { linearreg_slope_avx512_short(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_slope_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	linearreg_slope_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_slope_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	linearreg_slope_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_slope_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	linearreg_slope_scalar(data, period, first_valid, out)
}

pub fn linearreg_slope_batch_with_kernel(
	data: &[f64],
	sweep: &LinearRegSlopeBatchRange,
	kernel: Kernel,
) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
	let k = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(LinearRegSlopeError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match k {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	linearreg_slope_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct LinearRegSlopeBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for LinearRegSlopeBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct LinearRegSlopeBatchBuilder {
	range: LinearRegSlopeBatchRange,
	kernel: Kernel,
}

impl LinearRegSlopeBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
		linearreg_slope_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
		LinearRegSlopeBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
		LinearRegSlopeBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct LinearRegSlopeBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<LinearRegSlopeParams>,
	pub rows: usize,
	pub cols: usize,
}
impl LinearRegSlopeBatchOutput {
	pub fn row_for_params(&self, p: &LinearRegSlopeParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &LinearRegSlopeParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &LinearRegSlopeBatchRange) -> Vec<LinearRegSlopeParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(LinearRegSlopeParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn linearreg_slope_batch_slice(
	data: &[f64],
	sweep: &LinearRegSlopeBatchRange,
	kern: Kernel,
) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
	linearreg_slope_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linearreg_slope_batch_par_slice(
	data: &[f64],
	sweep: &LinearRegSlopeBatchRange,
	kern: Kernel,
) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
	linearreg_slope_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linearreg_slope_batch_inner(
	data: &[f64],
	sweep: &LinearRegSlopeBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<LinearRegSlopeBatchOutput, LinearRegSlopeError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(LinearRegSlopeError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(LinearRegSlopeError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(LinearRegSlopeError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory with proper warmup periods
	let mut buf_mu = make_uninit_matrix(rows, cols);
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
	};
	
	linearreg_slope_batch_inner_into(data, sweep, kern, parallel, out)?;
	
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(LinearRegSlopeBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn linearreg_slope_batch_inner_into(
	data: &[f64],
	sweep: &LinearRegSlopeBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<LinearRegSlopeParams>, LinearRegSlopeError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(LinearRegSlopeError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let first = data.iter().position(|x| !x.is_nan()).ok_or(LinearRegSlopeError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(LinearRegSlopeError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	
	let rows = combos.len();
	let cols = data.len();
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };
	
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
		
		match kern {
			Kernel::Scalar => linearreg_slope_row_scalar(data, first, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => linearreg_slope_row_avx2(data, first, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => linearreg_slope_row_avx512(data, first, period, dst),
			_ => unreachable!(),
		}
	};
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out_uninit
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}
		
		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	
	Ok(combos)
}

#[inline(always)]
unsafe fn linearreg_slope_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_slope_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_slope_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_slope_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_slope_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		linearreg_slope_row_avx512_short(data, first, period, out);
	} else {
		linearreg_slope_row_avx512_long(data, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_slope_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_slope_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_slope_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	linearreg_slope_scalar(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct LinearRegSlopeStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	n: f64,
	sum_x: f64,
	sum_x2: f64,
}

impl LinearRegSlopeStream {
	pub fn try_new(params: LinearRegSlopeParams) -> Result<Self, LinearRegSlopeError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(LinearRegSlopeError::InvalidPeriod { period, data_len: 0 });
		}
		let n = period as f64;
		let sum_x = (period - 1) as f64 * n / 2.0;
		let sum_x2 = (period - 1) as f64 * n * (2.0 * (period - 1) as f64 + 1.0) / 6.0;
		Ok(Self {
			period,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
			n,
			sum_x,
			sum_x2,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		// Update buffer
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}

		// Calculate sum_y and sum_xy for current window
		let mut sum_y = 0.0;
		let mut sum_xy = 0.0;
		let mut idx = self.head;
		
		for x in 0..self.period {
			let y = self.buffer[idx];
			sum_y += y;
			sum_xy += (x as f64) * y;
			idx = (idx + 1) % self.period;
		}

		let numerator = self.n * sum_xy - self.sum_x * sum_y;
		let denominator = self.n * self.sum_x2 - self.sum_x * self.sum_x;
		Some(if denominator.abs() < f64::EPSILON {
			f64::NAN
		} else {
			numerator / denominator
		})
	}
}

#[inline(always)]
fn expand_grid_stream(_r: &LinearRegSlopeBatchRange) -> Vec<LinearRegSlopeParams> {
	vec![LinearRegSlopeParams::default()]
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_slope")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn linearreg_slope_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = LinearRegSlopeParams {
		period: Some(period),
	};
	let linearreg_slope_in = LinearRegSlopeInput::from_slice(slice_in, params);
	
	let result_vec: Vec<f64> = py
		.allow_threads(|| linearreg_slope_with_kernel(&linearreg_slope_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_slope_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn linearreg_slope_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;
	
	let slice_in = data.as_slice()?;
	
	let sweep = LinearRegSlopeBatchRange {
		period: period_range,
	};
	
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
			linearreg_slope_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let result = PyDict::new(py);
	result.set_item("values", out_arr)?;
	result.set_item("rows", rows)?;
	result.set_item("cols", cols)?;
	
	let combo_list = PyList::empty(py);
	for combo in combos {
		let combo_dict = PyDict::new(py);
		combo_dict.set_item("period", combo.period.unwrap_or(14))?;
		combo_list.append(combo_dict)?;
	}
	result.set_item("combos", combo_list)?;
	
	Ok(result.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "LinearRegSlopeStream")]
pub struct LinearRegSlopeStreamPy {
	stream: LinearRegSlopeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LinearRegSlopeStreamPy {
	#[new]
	pub fn new(period: usize) -> PyResult<Self> {
		let params = LinearRegSlopeParams {
			period: Some(period),
		};
		let stream = LinearRegSlopeStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(Self { stream })
	}
	
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

/// Write directly to output slice - no allocations
pub fn linearreg_slope_into_slice(
	dst: &mut [f64],
	input: &LinearRegSlopeInput,
	kern: Kernel,
) -> Result<(), LinearRegSlopeError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(LinearRegSlopeError::EmptyData);
	}
	let period = input.get_period();
	if period == 0 || period > data.len() {
		return Err(LinearRegSlopeError::InvalidPeriod {
			period,
			data_len: data.len(),
		});
	}
	if dst.len() != data.len() {
		return Err(LinearRegSlopeError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}
	
	let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err(LinearRegSlopeError::AllValuesNaN),
	};
	if (data.len() - first_valid_idx) < period {
		return Err(LinearRegSlopeError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first_valid_idx,
		});
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => linearreg_slope_scalar(data, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => linearreg_slope_avx2(data, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => linearreg_slope_avx512(data, period, first_valid_idx, dst),
			_ => unreachable!(),
		}
	}
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_slope_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = LinearRegSlopeParams { period: Some(period) };
	let input = LinearRegSlopeInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];
	linearreg_slope_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_slope_into(
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
		let params = LinearRegSlopeParams { period: Some(period) };
		let input = LinearRegSlopeInput::from_slice(data, params);
		
		if in_ptr == out_ptr {
			// Handle aliasing
			let mut temp = vec![0.0; len];
			linearreg_slope_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			linearreg_slope_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_slope_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_slope_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LinearRegSlopeBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LinearRegSlopeBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<LinearRegSlopeParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = linearreg_slope_batch)]
pub fn linearreg_slope_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: LinearRegSlopeBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = LinearRegSlopeBatchRange {
		period: config.period_range,
	};
	
	let output = linearreg_slope_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = LinearRegSlopeBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_linearreg_slope_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = LinearRegSlopeParams { period: None };
		let input = LinearRegSlopeInput::from_candles(&candles, "close", default_params);
		let output = linearreg_slope_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_linearreg_slope_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [100.0, 98.0, 95.0, 90.0, 85.0, 80.0, 78.0, 77.0, 79.0, 81.0];
		let params = LinearRegSlopeParams { period: Some(5) };
		let input = LinearRegSlopeInput::from_slice(&input_data, params);
		let result = linearreg_slope_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), input_data.len());
		for val in &result.values[4..] {
			assert!(!val.is_nan(), "Expected valid slope values after period-1 index");
		}
		Ok(())
	}

	fn check_linearreg_slope_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = LinearRegSlopeParams { period: Some(0) };
		let input = LinearRegSlopeInput::from_slice(&input_data, params);
		let res = linearreg_slope_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] linearreg_slope should fail with zero period",
			test_name
		);
		Ok(())
	}

	fn check_linearreg_slope_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = LinearRegSlopeParams { period: Some(10) };
		let input = LinearRegSlopeInput::from_slice(&data_small, params);
		let res = linearreg_slope_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] linearreg_slope should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_linearreg_slope_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = LinearRegSlopeParams { period: Some(14) };
		let input = LinearRegSlopeInput::from_slice(&single_point, params);
		let res = linearreg_slope_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] linearreg_slope should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_linearreg_slope_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
		let first_params = LinearRegSlopeParams { period: Some(3) };
		let first_input = LinearRegSlopeInput::from_slice(&input_data, first_params);
		let first_result = linearreg_slope_with_kernel(&first_input, kernel)?;
		let second_params = LinearRegSlopeParams { period: Some(3) };
		let second_input = LinearRegSlopeInput::from_slice(&first_result.values, second_params);
		let second_result = linearreg_slope_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_linearreg_slope_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = LinearRegSlopeInput::from_candles(&candles, "close", LinearRegSlopeParams { period: Some(14) });
		let res = linearreg_slope_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 240 {
			for (i, &val) in res.values[240..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					240 + i
				);
			}
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_linearreg_slope_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			LinearRegSlopeParams::default(),                      // period: 14
			LinearRegSlopeParams { period: Some(2) },             // minimum period
			LinearRegSlopeParams { period: Some(3) },             // very small period
			LinearRegSlopeParams { period: Some(5) },             // small period
			LinearRegSlopeParams { period: Some(7) },             // week period
			LinearRegSlopeParams { period: Some(10) },            // common small period
			LinearRegSlopeParams { period: Some(14) },            // default period (explicit)
			LinearRegSlopeParams { period: Some(20) },            // medium period
			LinearRegSlopeParams { period: Some(21) },            // 3-week period
			LinearRegSlopeParams { period: Some(30) },            // month period
			LinearRegSlopeParams { period: Some(50) },            // large period
			LinearRegSlopeParams { period: Some(100) },           // very large period
			LinearRegSlopeParams { period: Some(200) },           // extra large period
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = LinearRegSlopeInput::from_candles(&candles, "close", params.clone());
			let output = linearreg_slope_with_kernel(&input, kernel)?;
			
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
	fn check_linearreg_slope_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_linearreg_slope_tests {
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
	generate_all_linearreg_slope_tests!(
		check_linearreg_slope_partial_params,
		check_linearreg_slope_accuracy,
		check_linearreg_slope_zero_period,
		check_linearreg_slope_period_exceeds_length,
		check_linearreg_slope_very_small_dataset,
		check_linearreg_slope_reinput,
		check_linearreg_slope_nan_handling,
		check_linearreg_slope_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = LinearRegSlopeBatchBuilder::new()
			.kernel(kernel)
			.apply_candles(&c, "close")?;
		let def = LinearRegSlopeParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step)
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods
			(30, 60, 15),    // Large periods
			(2, 5, 1),       // Dense small range
			(10, 30, 10),    // Common trading periods
			(14, 21, 7),     // Week-based periods
			(14, 14, 0),     // Static period (default)
			(50, 150, 25),   // Large period sweep
			(3, 15, 3),      // Small to medium range
		];
		
		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = LinearRegSlopeBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c, "close")?;
			
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
	gen_batch_tests!(check_batch_no_poison);
}
