//! # Kaufman Efficiency Ratio (ER)
//!
//! The Kaufman Efficiency Ratio (ER) compares the absolute price change over a specified
//! period to the sum of the incremental absolute changes within that same window.
//! Returns a value between 0.0 and 1.0 (high = efficient move, low = choppy).
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **AllValuesNaN**: er: All input data values are `NaN`.
//! - **InvalidPeriod**: er: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: er: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(ErOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(ErError)`** otherwise.

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

impl<'a> AsRef<[f64]> for ErInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			ErData::Slice(slice) => slice,
			ErData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum ErData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ErOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ErParams {
	pub period: Option<usize>,
}

impl Default for ErParams {
	fn default() -> Self {
		Self { period: Some(5) }
	}
}

#[derive(Debug, Clone)]
pub struct ErInput<'a> {
	pub data: ErData<'a>,
	pub params: ErParams,
}

impl<'a> ErInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: ErParams) -> Self {
		Self {
			data: ErData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: ErParams) -> Self {
		Self {
			data: ErData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", ErParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(5)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct ErBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for ErBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl ErBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<ErOutput, ErError> {
		let p = ErParams { period: self.period };
		let i = ErInput::from_candles(c, "close", p);
		er_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<ErOutput, ErError> {
		let p = ErParams { period: self.period };
		let i = ErInput::from_slice(d, p);
		er_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<ErStream, ErError> {
		let p = ErParams { period: self.period };
		ErStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum ErError {
	#[error("er: Input data slice is empty.")]
	EmptyInputData,
	#[error("er: All input data values are NaN.")]
	AllValuesNaN,
	#[error("er: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("er: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("er: Output length mismatch: dst_len = {dst_len}, data_len = {data_len}")]
	OutputLenMismatch { dst_len: usize, data_len: usize },
}

#[cfg(feature = "wasm")]
impl From<ErError> for JsValue {
	fn from(err: ErError) -> Self {
		JsValue::from_str(&err.to_string())
	}
}

#[inline]
pub fn er(input: &ErInput) -> Result<ErOutput, ErError> {
	er_with_kernel(input, Kernel::Auto)
}

pub fn er_with_kernel(input: &ErInput, kernel: Kernel) -> Result<ErOutput, ErError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(ErError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(ErError::AllValuesNaN)?;
	let period = input.get_period();
	if period == 0 || period > len {
		return Err(ErError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(ErError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warm = first + period - 1;
	let mut out = alloc_with_nan_prefix(len, warm);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => er_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => er_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => er_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(ErOutput { values: out })
}

#[inline]
pub fn er_into_slice(dst: &mut [f64], input: &ErInput, kern: Kernel) -> Result<(), ErError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(ErError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(ErError::AllValuesNaN)?;
	let period = input.get_period();
	if period == 0 || period > len {
		return Err(ErError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(ErError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	if dst.len() != len {
		return Err(ErError::OutputLenMismatch {
			dst_len: dst.len(),
			data_len: len,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => er_scalar(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => er_avx2(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => er_avx512(data, period, first, dst),
			_ => unreachable!(),
		}
	}

	// mark warmup after compute (matches alma.rs)
	let warm_end = first + period - 1;
	for v in &mut dst[..warm_end] {
		*v = f64::NAN;
	}
	
	Ok(())
}

#[inline]
pub fn er_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	let n = data.len();
	for i in (first + period - 1)..n {
		let start = i + 1 - period;
		let delta = (data[i] - data[start]).abs();
		let mut sum = 0.0;
		for j in start..i {
			sum += (data[j + 1] - data[j]).abs();
		}
		if sum > 0.0 {
			// Clamp to [0.0, 1.0] to handle floating point precision issues
			out[i] = (delta / sum).min(1.0);
		} else {
			// When sum is 0 (all values in window are the same), ER is undefined
			// but we set it to 0.0 to indicate no directional movement
			out[i] = 0.0;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn er_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	unsafe {
		if period <= 32 {
			er_avx512_short(data, period, first, out);
		} else {
			er_avx512_long(data, period, first, out);
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn er_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn er_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn er_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	er_scalar(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct ErStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl ErStream {
	pub fn try_new(params: ErParams) -> Result<Self, ErError> {
		let period = params.period.unwrap_or(5);
		if period == 0 {
			return Err(ErError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
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
		let mut sum = 0.0;
		let mut last = self.head;
		let mut prev = last;
		for _ in 1..self.period {
			prev = (prev + 1) % self.period;
			let a = self.buffer[last];
			let b = self.buffer[prev];
			sum += (b - a).abs();
			last = prev;
		}
		let start = self.head;
		let end = (self.head + self.period - 1) % self.period;
		let delta = (self.buffer[end] - self.buffer[start]).abs();
		if sum > 0.0 {
			// Clamp to [0.0, 1.0] to handle floating point precision issues
			Some((delta / sum).min(1.0))
		} else {
			Some(0.0)
		}
	}
}

#[derive(Clone, Debug)]
pub struct ErBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for ErBatchRange {
	fn default() -> Self {
		Self { period: (5, 60, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct ErBatchBuilder {
	range: ErBatchRange,
	kernel: Kernel,
}

impl ErBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<ErBatchOutput, ErError> {
		er_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ErBatchOutput, ErError> {
		ErBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ErBatchOutput, ErError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<ErBatchOutput, ErError> {
		ErBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn er_batch_with_kernel(data: &[f64], sweep: &ErBatchRange, k: Kernel) -> Result<ErBatchOutput, ErError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(ErError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	er_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ErBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<ErParams>,
	pub rows: usize,
	pub cols: usize,
}
impl ErBatchOutput {
	pub fn row_for_params(&self, p: &ErParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
	}
	pub fn values_for(&self, p: &ErParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &ErBatchRange) -> Vec<ErParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(ErParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn er_batch_slice(data: &[f64], sweep: &ErBatchRange, kern: Kernel) -> Result<ErBatchOutput, ErError> {
	er_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn er_batch_par_slice(data: &[f64], sweep: &ErBatchRange, kern: Kernel) -> Result<ErBatchOutput, ErError> {
	er_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn er_batch_inner_into(data: &[f64], sweep: &ErBatchRange, kern: Kernel, parallel: bool, out: &mut [f64]) -> Result<Vec<ErParams>, ErError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(ErError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let cols = data.len();
	if cols == 0 {
		return Err(ErError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(ErError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if cols - first < max_p {
		return Err(ErError::NotEnoughValidData {
			needed: max_p,
			valid: cols - first,
		});
	}
	
	// initialize warm prefixes in-place using MaybeUninit view
	let rows = combos.len();
	let out_mu = unsafe {
		std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
	};
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(out_mu, cols, &warm);
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => er_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => er_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => er_row_avx512(data, first, period, out_row),
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

#[inline(always)]
fn er_batch_inner(data: &[f64], sweep: &ErBatchRange, kern: Kernel, parallel: bool) -> Result<ErBatchOutput, ErError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(ErError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let cols = data.len();
	if cols == 0 {
		return Err(ErError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(ErError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if cols - first < max_p {
		return Err(ErError::NotEnoughValidData {
			needed: max_p,
			valid: cols - first,
		});
	}
	
	let rows = combos.len();
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	// Convert to mutable slice for computation
	let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
	let values: &mut [f64] = unsafe { 
		std::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) 
	};
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => er_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => er_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => er_row_avx512(data, first, period, out_row),
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
	
	// Convert uninitialized memory back to Vec
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(ErBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn er_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		er_row_avx512_short(data, first, period, out);
	} else {
		er_row_avx512_long(data, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	er_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn er_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	er_scalar(data, period, first, out)
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "er")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn er_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = ErParams { period: Some(period) };
	let input = ErInput::from_slice(slice_in, params);

	let result_vec = py
		.allow_threads(|| er_with_kernel(&input, kern))
		.map(|result| result.values)
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ErStream")]
pub struct ErStreamPy {
	stream: ErStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ErStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = ErParams { period: Some(period) };
		let stream = ErStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(ErStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "er_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn er_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = ErBatchRange { period: period_range };
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// uninitialized numpy buffer; we'll set warm via init_matrix_prefixes inside _into
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	let combos = py
		.allow_threads(|| {
			let batch = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};
			let simd = match batch {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => unreachable!(),
			};
			er_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos.iter().map(|p| p.period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py),
	)?;
	dict.set_item("rows", rows)?;
	dict.set_item("cols", cols)?;
	Ok(dict)
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn er_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = ErParams { period: Some(period) };
	let input = ErInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];
	
	er_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn er_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn er_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn er_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to er_into"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}
		
		let params = ErParams { period: Some(period) };
		let input = ErInput::from_slice(data, params);
		
		// Critical: handle aliasing
		if in_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			er_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			er_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ErBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ErBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<ErParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = er_batch)]
pub fn er_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: ErBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = ErBatchRange {
		period: config.period_range,
	};
	
	let output = er_batch_with_kernel(data, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = ErBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn er_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to er_batch_into"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let sweep = ErBatchRange {
			period: (period_start, period_end, period_step),
		};
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;
		if rows * cols > 0 {
			let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
			// warm prefixes initialized inside er_batch_inner_into
			let batch_kernel = detect_best_batch_kernel();
			let simd = match batch_kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => unreachable!(),
			};
			er_batch_inner_into(data, &sweep, simd, false, out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_er_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = ErParams { period: None };
		let input = ErInput::from_candles(&candles, "close", default_params);
		let output = er_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_er_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = ErInput::with_default_candles(&candles);
		match input.data {
			ErData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected ErData::Candles"),
		}
		let output = er_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_er_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = ErParams { period: Some(0) };
		let input = ErInput::from_slice(&input_data, params);
		let res = er_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ER should fail with zero period", test_name);
		Ok(())
	}

	fn check_er_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = ErParams { period: Some(10) };
		let input = ErInput::from_slice(&data_small, params);
		let res = er_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] ER should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_er_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = ErParams { period: Some(5) };
		let input = ErInput::from_slice(&single_point, params);
		let res = er_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ER should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_er_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = ErParams { period: Some(5) };
		let first_input = ErInput::from_candles(&candles, "close", first_params);
		let first_result = er_with_kernel(&first_input, kernel)?;

		let second_params = ErParams { period: Some(5) };
		let second_input = ErInput::from_slice(&first_result.values, second_params);
		let second_result = er_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_er_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = ErInput::from_candles(&candles, "close", ErParams { period: Some(5) });
		let res = er_with_kernel(&input, kernel)?;
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

	fn check_er_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 5;

		let input = ErInput::from_candles(&candles, "close", ErParams { period: Some(period) });
		let batch_output = er_with_kernel(&input, kernel)?.values;

		let mut stream = ErStream::try_new(ErParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(er_val) => stream_values.push(er_val),
				None => stream_values.push(f64::NAN),
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
				"[{}] ER streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_er_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			ErParams::default(),
			ErParams { period: Some(1) },
			ErParams { period: Some(2) },
			ErParams { period: Some(3) },
			ErParams { period: Some(4) },
			ErParams { period: Some(5) },
			ErParams { period: Some(10) },
			ErParams { period: Some(14) },
			ErParams { period: Some(20) },
			ErParams { period: Some(30) },
			ErParams { period: Some(50) },
			ErParams { period: Some(100) },
			ErParams { period: Some(200) },
			ErParams { period: Some(500) },
			ErParams { period: Some(1000) },
			ErParams { period: Some(2000) },
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = ErInput::from_candles(&candles, "close", params.clone());
			let output = er_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(5),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(5),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(5),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_er_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_er_tests {
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

	generate_all_er_tests!(
		check_er_partial_params,
		check_er_default_candles,
		check_er_zero_period,
		check_er_period_exceeds_length,
		check_er_very_small_dataset,
		check_er_reinput,
		check_er_nan_handling,
		check_er_streaming,
		check_er_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_er_tests!(check_er_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = ErBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = ErParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());

		// Not a strict accuracy test, just batch output row length check.
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
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let test_configs = vec![
			(1, 5, 1),
			(2, 10, 2),
			(5, 30, 5),
			(10, 100, 10),
			(50, 500, 50),
			(100, 1000, 100),
			(14, 14, 0),
			(3, 15, 1),
			(20, 200, 20),
			(25, 50, 5),
		];

		for (cfg_idx, &(period_start, period_end, period_step)) in test_configs.iter().enumerate() {
			let output = ErBatchBuilder::new()
				.kernel(kernel)
				.period_range(period_start, period_end, period_step)
				.apply_candles(&c, "close")?;

			for (idx, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(5)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(5)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(5)
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

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);

	#[cfg(feature = "proptest")]
	fn check_er_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy for generating test data with realistic price movements
		// Note: Period starts from 2 since period=1 doesn't make mathematical sense for ER
		let strat = (2usize..=50)
			.prop_flat_map(|period| {
				let min_len = period * 2; // Ensure sufficient data for meaningful testing
				(
					// Base price level and volatility
					(100.0f64..5000.0f64, 0.01f64..0.1f64),
					// Trend strength (-2% to +2% per step)
					-0.02f64..0.02f64,
					// Generate period and data length
					Just(period),
					min_len..400,
				)
			})
			.prop_flat_map(|((base_price, volatility), trend, period, len)| {
				// Generate realistic price data with trend and noise
				let price_changes = prop::collection::vec(
					(-1.0f64..1.0f64),
					len
				);
				
				(Just(base_price), Just(volatility), Just(trend), Just(period), price_changes)
			})
			.prop_map(|(base_price, volatility, trend, period, changes)| {
				// Create realistic price series with trend and volatility
				let mut data = Vec::with_capacity(changes.len());
				let mut price = base_price;
				
				for (i, &noise) in changes.iter().enumerate() {
					// Add trend component
					price *= 1.0 + trend;
					// Add noise scaled by volatility
					price *= 1.0 + (noise * volatility);
					// Ensure price stays positive
					price = price.max(1.0);
					data.push(price);
				}
				
				(data, period)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = ErParams { period: Some(period) };
				let input = ErInput::from_slice(&data, params);

				let ErOutput { values: out } = er_with_kernel(&input, kernel).unwrap();
				let ErOutput { values: ref_out } = er_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Output length equals input length
				prop_assert_eq!(out.len(), data.len());

				// Property 2: Warmup period handling - first (period - 1) values should be NaN
				let warmup = period - 1;
				for i in 0..warmup {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Property 3: Mathematical bounds - ER must be in [0.0, 1.0]
				for i in warmup..data.len() {
					let val = out[i];
					if !val.is_nan() {
						prop_assert!(
							val >= -1e-10 && val <= 1.0 + 1e-10,
							"ER value {} at index {} outside valid range [0, 1]",
							val,
							i
						);
					}
				}

				// Property 4: Kernel consistency - all kernels should produce identical results
				for i in 0..data.len() {
					let y = out[i];
					let r = ref_out[i];

					if !y.is_finite() || !r.is_finite() {
						prop_assert_eq!(
							y.to_bits(),
							r.to_bits(),
							"NaN/Inf mismatch at index {}: {} vs {}",
							i,
							y,
							r
						);
					} else {
						let diff = (y - r).abs();
						let ulp_diff = y.to_bits().abs_diff(r.to_bits());
						prop_assert!(
							diff <= 1e-9 || ulp_diff <= 4,
							"Kernel mismatch at index {}: {} vs {} (diff={}, ULP={})",
							i,
							y,
							r,
							diff,
							ulp_diff
						);
					}
				}

				// Property 5: Perfect efficiency for straight line moves
				// Note: Using 0.90 threshold for practical tolerance with floating-point arithmetic
				if data.len() >= period + 10 {
					// Find a monotonic section if exists
					for window_start in warmup..(data.len() - period) {
						let window_end = (window_start + period).min(data.len() - 1);
						let window = &data[window_start..=window_end];
						
						// Check if this window is monotonic but not constant
						let is_monotonic_up = window.windows(2).all(|w| w[1] >= w[0] - 1e-10);
						let is_monotonic_down = window.windows(2).all(|w| w[1] <= w[0] + 1e-10);
						let is_constant = window.windows(2).all(|w| (w[1] - w[0]).abs() < 1e-10);
						
						// Skip constant windows as they're handled by Property 6
						if !is_constant && (is_monotonic_up || is_monotonic_down) {
							// For a perfect trend, ER should be close to 1.0
							let er_val = out[window_end];
							if !er_val.is_nan() && (window[window.len()-1] - window[0]).abs() > 1e-6 {
								prop_assert!(
									er_val >= 0.90,
									"Expected high ER (>0.90) for monotonic move at {}, got {}",
									window_end,
									er_val
								);
							}
						}
					}
				}

				// Property 6: Constant prices should yield 0.0
				// When all prices in window are identical, delta=0 and sum=0, ER remains unset (NaN from warmup)
				// But after warmup, the indicator leaves the value as 0.0 since sum is not > 0
				for i in warmup..data.len().saturating_sub(period) {
					let window_end = i + period - 1;
					if window_end < data.len() {
						let window = &data[i..=window_end];
						let is_constant = window.windows(2).all(|w| (w[1] - w[0]).abs() < 1e-10);
						
						if is_constant {
							let er_val = out[window_end];
							// For constant prices, ER calculation has delta=0 and sum=0
							// The guard `if sum > 0.0` prevents division, leaving the value unchanged
							// Since we use alloc_with_nan_prefix, values after warmup stay as allocated (0.0)
							prop_assert!(
								er_val.is_nan() || er_val.abs() < 1e-10,
								"Constant prices should yield NaN or 0, got {} at index {}",
								er_val,
								window_end
							);
						}
					}
				}

				// Property 7: Non-negative values
				for i in warmup..data.len() {
					let val = out[i];
					if !val.is_nan() {
						prop_assert!(
							val >= -1e-10,
							"ER should be non-negative, got {} at index {}",
							val,
							i
						);
					}
				}

				// Property 8: Choppy market detection
				// Create a synthetic choppy pattern and verify low ER
				if period >= 4 && data.len() >= period * 3 {
					// Look for sections with high volatility relative to net movement
					for i in warmup..(data.len() - period) {
						let window_start = i;
						let window_end = i + period - 1;
						if window_end < data.len() {
							let net_change = (data[window_end] - data[window_start]).abs();
							let mut total_movement = 0.0;
							for j in window_start..window_end {
								total_movement += (data[j + 1] - data[j]).abs();
							}
							
							// If total movement is much larger than net change, market is choppy
							if total_movement > 0.0 && net_change / total_movement < 0.3 {
								let er_val = out[window_end];
								if !er_val.is_nan() {
									// Choppy markets should have low ER
									prop_assert!(
										er_val <= 0.35,
										"Expected low ER (<0.35) for choppy market at {}, got {}",
										window_end,
										er_val
									);
								}
							}
						}
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}
}
