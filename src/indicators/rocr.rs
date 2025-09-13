//! # Rate of Change Ratio (ROCR)
//!
//! Measures the ratio current/past, centered at 1.0 (>1 increase, <1 decrease).
//!
//! ## Parameters
//! - **data**: Input price data
//! - **period**: Lookback window (default: 10)
//!
//! ## Returns
//! - `Vec<f64>` - ROCR values centered at 1.0, matching input length
//!
//! ## Developer Status
//! **AVX2**: Stub (calls scalar)
//! **AVX512**: Has short/long variants but all stubs
//! **Streaming**: O(1) - Simple ring buffer lookup
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, 
	init_matrix_prefixes, make_uninit_matrix
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 10;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for RocrInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			RocrData::Slice(slice) => slice,
			RocrData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum RocrData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocrOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct RocrParams {
	pub period: Option<usize>,
}

impl Default for RocrParams {
	fn default() -> Self {
		Self { period: Some(DEFAULT_PERIOD) }
	}
}

#[derive(Debug, Clone)]
pub struct RocrInput<'a> {
	pub data: RocrData<'a>,
	pub params: RocrParams,
}

impl<'a> RocrInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: RocrParams) -> Self {
		Self {
			data: RocrData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: RocrParams) -> Self {
		Self {
			data: RocrData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", RocrParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(DEFAULT_PERIOD)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct RocrBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for RocrBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl RocrBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<RocrOutput, RocrError> {
		let p = RocrParams { period: self.period };
		let i = RocrInput::from_candles(c, "close", p);
		rocr_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<RocrOutput, RocrError> {
		let p = RocrParams { period: self.period };
		let i = RocrInput::from_slice(d, p);
		rocr_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<RocrStream, RocrError> {
		let p = RocrParams { period: self.period };
		RocrStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum RocrError {
	#[error("rocr: Empty data provided.")]
	EmptyData,
	#[error("rocr: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("rocr: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("rocr: All values are NaN.")]
	AllValuesNaN,
}

#[inline(always)]
fn rocr_prepare<'a>(
	input: &'a RocrInput,
	kern: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), RocrError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(RocrError::EmptyData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocrError::AllValuesNaN)?;
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(RocrError::InvalidPeriod { period, data_len: len });
	}
	if len - first < period {
		return Err(RocrError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};
	Ok((data, period, first, chosen))
}

#[inline]
pub fn rocr(input: &RocrInput) -> Result<RocrOutput, RocrError> {
	rocr_with_kernel(input, Kernel::Auto)
}

pub fn rocr_with_kernel(input: &RocrInput, kernel: Kernel) -> Result<RocrOutput, RocrError> {
	let (data, period, first, chosen) = rocr_prepare(input, kernel)?;
	let mut out = alloc_with_nan_prefix(data.len(), first + period);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => rocr_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rocr_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rocr_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(RocrOutput { values: out })
}

/// Write ROCR directly to output slice - zero allocations (for WASM optimization)
pub fn rocr_into_slice(dst: &mut [f64], input: &RocrInput, kern: Kernel) -> Result<(), RocrError> {
	let (data, period, first, chosen) = rocr_prepare(input, kern)?;
	if dst.len() != data.len() {
		return Err(RocrError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => rocr_scalar(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rocr_avx2(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rocr_avx512(data, period, first, dst),
			_ => unreachable!(),
		}
	}

	// Warmup prefix = NaN, inclusive of first..first+period
	let warm = first + period;
	for v in &mut dst[..warm] {
		*v = f64::NAN;
	}
	Ok(())
}

#[inline]
pub fn rocr_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	for i in (first_val + period)..data.len() {
		let past = data[i - period];
		out[i] = if past == 0.0 || past.is_nan() { 0.0 } else { data[i] / past };
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocr_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocr_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocr_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocr_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

pub struct RocrStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl RocrStream {
	pub fn try_new(params: RocrParams) -> Result<Self, RocrError> {
		let period = params.period.unwrap_or(DEFAULT_PERIOD);
		if period == 0 {
			return Err(RocrError::InvalidPeriod { period, data_len: 0 });
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
		let out = if self.filled {
			let prev = self.buffer[self.head];
			if prev == 0.0 || prev.is_nan() { 0.0 } else { value / prev }
		} else {
			// do not emit until filled
			self.buffer[self.head] = value;
			self.head = (self.head + 1) % self.period;
			if !self.filled && self.head == 0 { self.filled = true; }
			return None;
		};

		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 { self.filled = true; }
		Some(out)
	}
}

#[derive(Clone, Debug)]
pub struct RocrBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for RocrBatchRange {
	fn default() -> Self {
		Self { period: (9, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct RocrBatchBuilder {
	range: RocrBatchRange,
	kernel: Kernel,
}

impl RocrBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<RocrBatchOutput, RocrError> {
		rocr_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RocrBatchOutput, RocrError> {
		RocrBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RocrBatchOutput, RocrError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<RocrBatchOutput, RocrError> {
		RocrBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn rocr_batch_with_kernel(data: &[f64], sweep: &RocrBatchRange, k: Kernel) -> Result<RocrBatchOutput, RocrError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(RocrError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	rocr_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RocrBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RocrParams>,
	pub rows: usize,
	pub cols: usize,
}
impl RocrBatchOutput {
	pub fn row_for_params(&self, p: &RocrParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
	}

	pub fn values_for(&self, p: &RocrParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &RocrBatchRange) -> Vec<RocrParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(RocrParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn rocr_batch_slice(data: &[f64], sweep: &RocrBatchRange, kern: Kernel) -> Result<RocrBatchOutput, RocrError> {
	rocr_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn rocr_batch_par_slice(data: &[f64], sweep: &RocrBatchRange, kern: Kernel) -> Result<RocrBatchOutput, RocrError> {
	rocr_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rocr_batch_inner(
	data: &[f64],
	sweep: &RocrBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<RocrBatchOutput, RocrError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RocrError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocrError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(RocrError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	// Use uninitialized memory for better performance
	let mut values_uninit = make_uninit_matrix(rows, cols);
	
	// Initialize NaN prefixes for each row based on period
	let warmup_periods: Vec<usize> = combos.iter()
		.map(|c| first + c.period.unwrap())
		.collect();
	init_matrix_prefixes(&mut values_uninit, cols, &warmup_periods);
	
	// Convert to mutable slice without copying - using ManuallyDrop pattern from ALMA
	let mut buf_guard = core::mem::ManuallyDrop::new(values_uninit);
	let values: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len()
		)
	};

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();

		match kern {
			Kernel::Scalar => rocr_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rocr_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rocr_row_avx512(data, first, period, out_row),
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

	// Convert ManuallyDrop back to Vec without copying
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(RocrBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn rocr_batch_inner_into(
	data: &[f64],
	sweep: &RocrBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<RocrParams>, RocrError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RocrError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocrError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(RocrError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let cols = data.len();

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();

		match kern {
			Kernel::Scalar => rocr_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rocr_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rocr_row_avx512(data, first, period, out_row),
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
unsafe fn rocr_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	for i in (first + period)..data.len() {
		let current = data[i];
		let past = data[i - period];
		out[i] = if past == 0.0 || past.is_nan() {
			0.0
		} else {
			current / past
		};
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rocr_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocr_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocr_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocr_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[inline(always)]
pub fn expand_grid_rocr(r: &RocrBatchRange) -> Vec<RocrParams> {
	expand_grid(r)
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "rocr")]
#[pyo3(signature = (data, period=DEFAULT_PERIOD, kernel=None))]
pub fn rocr_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = RocrParams {
		period: Some(period),
	};
	let input = RocrInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| rocr_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "RocrStream")]
pub struct RocrStreamPy {
	stream: RocrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RocrStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = RocrParams {
			period: Some(period),
		};
		let stream = RocrStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(RocrStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "rocr_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn rocr_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;
	use std::mem::MaybeUninit;

	let slice_in = data.as_slice()?;
	let sweep = RocrBatchRange {
		period: period_range,
	};

	// Build combos up front to compute warmprefix per row
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	let kern = validate_kernel(kernel, true)?;
	let simd = match match kern {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	} {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};

	// Allocate uninitialized Python array, then prefill warm prefixes via helper
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(cols);
	let warms: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

	// Cast to MaybeUninit and initialize only warm prefixes with NaN (zero extra copies)
	unsafe {
		let mu: &mut [MaybeUninit<f64>] = std::slice::from_raw_parts_mut(
			out_arr.as_ptr() as *mut MaybeUninit<f64>,
			rows * cols
		);
		init_matrix_prefixes(mu, cols, &warms);
	}

	// Now compute rows directly into the same buffer
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	let combos_result = py.allow_threads(|| rocr_batch_inner_into(slice_in, &sweep, simd, true, slice_out))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos_result.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocr_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = RocrParams {
		period: Some(period),
	};
	let input = RocrInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];  // Single allocation
	rocr_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocr_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocr_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocr_into(
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
		let params = RocrParams {
			period: Some(period),
		};
		let input = RocrInput::from_slice(data, params);
		
		if in_ptr == out_ptr {  // CRITICAL: Aliasing check
			// In-place operation - use temporary buffer
			let mut temp = vec![0.0; len];
			rocr_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// Direct write to output buffer
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			rocr_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

// Batch API for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RocrBatchConfig {
	pub period_range: (usize, usize, usize), // (start, end, step)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RocrBatchJsOutput {
	pub values: Vec<f64>,       // Flattened array
	pub combos: Vec<RocrParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = rocr_batch)]
pub fn rocr_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: RocrBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = RocrBatchRange {
		period: config.period_range,
	};

	let output = rocr_batch_with_kernel(data, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = RocrBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocr_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to rocr_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		
		// Calculate number of combinations
		let period_count = if period_step > 0 {
			((period_end.saturating_sub(period_start)) / period_step) + 1
		} else {
			1
		};
		
		let total_elements = period_count * len;
		let out = std::slice::from_raw_parts_mut(out_ptr, total_elements);
		
		// Build batch range
		let sweep = RocrBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		// Compute batch directly into output buffer
		let simd = match detect_best_batch_kernel() {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			_ => Kernel::Scalar,
		};
		let combos = rocr_batch_inner_into(data, &sweep, simd, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(combos.len())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_rocr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = RocrParams { period: None };
		let input = RocrInput::from_candles(&candles, "close", default_params);
		let output = rocr_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_rocr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocrInput::from_candles(&candles, "close", RocrParams { period: Some(10) });
		let result = rocr_with_kernel(&input, kernel)?;
		let expected_last_five = [
			0.9977448290950706,
			0.9944380965183492,
			0.9967247986764135,
			0.9950545846019277,
			0.984954072979463,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] ROCR {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_rocr_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocrInput::with_default_candles(&candles);
		match input.data {
			RocrData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected RocrData::Candles"),
		}
		let output = rocr_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_rocr_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = RocrParams { period: Some(0) };
		let input = RocrInput::from_slice(&input_data, params);
		let res = rocr_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ROCR should fail with zero period", test_name);
		Ok(())
	}

	fn check_rocr_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = RocrParams { period: Some(10) };
		let input = RocrInput::from_slice(&data_small, params);
		let res = rocr_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] ROCR should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_rocr_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = RocrParams { period: Some(9) };
		let input = RocrInput::from_slice(&single_point, params);
		let res = rocr_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ROCR should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_rocr_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = RocrParams { period: Some(14) };
		let first_input = RocrInput::from_candles(&candles, "close", first_params);
		let first_result = rocr_with_kernel(&first_input, kernel)?;

		let second_params = RocrParams { period: Some(14) };
		let second_input = RocrInput::from_slice(&first_result.values, second_params);
		let second_result = rocr_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 28..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] ROCR Slice Reinput {:?} unexpected NaN at idx {}",
				test_name,
				kernel,
				i,
			);
		}
		Ok(())
	}

	fn check_rocr_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocrInput::from_candles(&candles, "close", RocrParams { period: Some(9) });
		let res = rocr_with_kernel(&input, kernel)?;
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

	fn check_rocr_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 9;

		let input = RocrInput::from_candles(&candles, "close", RocrParams { period: Some(period) });
		let batch_output = rocr_with_kernel(&input, kernel)?.values;

		let mut stream = RocrStream::try_new(RocrParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(rocr_val) => stream_values.push(rocr_val),
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
				"[{}] ROCR streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_rocr_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			RocrParams::default(),                    // period: 10
			RocrParams { period: Some(1) },          // minimum viable
			RocrParams { period: Some(5) },          // small
			RocrParams { period: Some(20) },         // medium
			RocrParams { period: Some(50) },         // large
			RocrParams { period: Some(100) },        // very large
			RocrParams { period: Some(2) },          // edge case: smallest meaningful period
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = RocrInput::from_candles(&candles, "close", params.clone());
			let output = rocr_with_kernel(&input, kernel)?;
			
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
						test_name, val, bits, i, params.period.unwrap_or(10), param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(10), param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(10), param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_rocr_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_rocr_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// ROCR is for price data, typically positive but can be zero
		// Test with realistic price ranges including edge cases
		let strat = (1usize..=64)
			.prop_flat_map(|period| {
				(
					prop::collection::vec(
						prop::strategy::Union::new(vec![
							// Most values: normal price range
							(0.1f64..10000f64).boxed(),
							// Occasional zeros (about 5% chance)
							prop::strategy::Just(0.0).boxed(),
							// Extreme small values for ratio testing
							(1e-10f64..1e-5f64).boxed(),
							// Large values for overflow testing
							(1e5f64..1e8f64).boxed(),
						]).prop_filter("finite values", |x| x.is_finite() && *x >= 0.0),
						period..400,
					),
					Just(period),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = RocrParams {
					period: Some(period),
				};
				let input = RocrInput::from_slice(&data, params);

				let RocrOutput { values: out } = rocr_with_kernel(&input, kernel).unwrap();
				let RocrOutput { values: ref_out } = rocr_with_kernel(&input, Kernel::Scalar).unwrap();

				// Check warmup period - first 'period' values should be NaN
				for i in 0..period {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Verify ROCR calculation for valid indices
				for i in period..data.len() {
					let current = data[i];
					let past = data[i - period];
					
					// Calculate expected value according to ROCR formula
					let expected = if past == 0.0 || past.is_nan() {
						0.0
					} else {
						current / past
					};

					let y = out[i];
					let r = ref_out[i];

					// Verify the mathematical formula
					if !y.is_nan() {
						// Use relative tolerance for large values, absolute for small
						let tolerance = if expected.abs() > 1.0 {
							expected.abs() * 1e-9
						} else {
							1e-9
						};
						
						prop_assert!(
							(y - expected).abs() <= tolerance,
							"ROCR formula mismatch at idx {}: got {}, expected {} (current={}, past={})",
							i, y, expected, current, past
						);

						// ROCR should be non-negative (prices are non-negative)
						prop_assert!(
							y >= 0.0,
							"ROCR should be non-negative at idx {}: got {}",
							i, y
						);
						
						// Test edge case: when current is 0, ROCR should be 0
						if current == 0.0 && past != 0.0 {
							prop_assert!(
								y == 0.0,
								"ROCR should be 0 when current=0 at idx {}: got {}",
								i, y
							);
						}
						
						// Test edge case: when past is 0, ROCR should be 0
						if past == 0.0 {
							prop_assert!(
								y == 0.0,
								"ROCR should be 0 when past=0 at idx {}: got {}",
								i, y
							);
						}
						
						// Verify no overflow/underflow with extreme ratios
						prop_assert!(
							y.is_finite(),
							"ROCR should be finite at idx {}: got {} (current={}, past={})",
							i, y, current, past
						);
					}

					// Special case: period = 1 means comparing to previous value
					if period == 1 && i > 0 && data[i - 1] != 0.0 {
						let expected_simple = data[i] / data[i - 1];
						if !y.is_nan() {
							let tolerance = if expected_simple.abs() > 1.0 {
								expected_simple.abs() * 1e-9
							} else {
								1e-9
							};
							prop_assert!(
								(y - expected_simple).abs() <= tolerance,
								"Period=1 mismatch at idx {}: got {}, expected {}",
								i, y, expected_simple
							);
						}
					}

					// Special case: constant non-zero data should return 1.0
					// Only check this for windows with at least 2 elements
					if i >= period && period > 1 {
						let window = &data[i - period + 1..=i];
						// Check if all values in the window are approximately equal
						let first_val = window[0];
						let is_constant = first_val != 0.0 && window.iter()
							.all(|&v| (v - first_val).abs() <= 1e-10 * first_val.abs().max(1.0));
						
						if is_constant {
							prop_assert!(
								(y - 1.0).abs() <= 1e-9,
								"Constant data should yield ROCR=1.0 at idx {}: got {}",
								i, y
							);
						}
					}

					// Verify kernel consistency
					let y_bits = y.to_bits();
					let r_bits = r.to_bits();

					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"finite/NaN mismatch idx {}: {} vs {}",
							i, y, r
						);
						continue;
					}

					let ulp_diff: u64 = y_bits.abs_diff(r_bits);

					// Allow slightly more ULP difference for extreme values
					let max_ulp = if y.abs() > 1e6 || y.abs() < 1e-6 { 8 } else { 4 };
					
					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= max_ulp,
						"Kernel mismatch idx {}: {} vs {} (ULP={})",
						i, y, r, ulp_diff
					);
				}
				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_rocr_tests {
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

	generate_all_rocr_tests!(
		check_rocr_partial_params,
		check_rocr_accuracy,
		check_rocr_default_candles,
		check_rocr_zero_period,
		check_rocr_period_exceeds_length,
		check_rocr_very_small_dataset,
		check_rocr_reinput,
		check_rocr_nan_handling,
		check_rocr_streaming,
		check_rocr_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_rocr_tests!(check_rocr_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = RocrBatchBuilder::new()
			.period_static(10)
			.kernel(kernel)
			.apply_candles(&c, "close")?;

		let def = RocrParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			0.9977448290950706,
			0.9944380965183492,
			0.9967247986764135,
			0.9950545846019277,
			0.984954072979463,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-8,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods  
			(30, 60, 15),    // Large periods
			(2, 5, 1),       // Dense small range
			(10, 50, 10),    // Wide range with large step
			(1, 3, 1),       // Minimum periods
			(100, 200, 50),  // Very large periods
		];
		
		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = RocrBatchBuilder::new()
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
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(10)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(10)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(10)
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
									 Kernel::Auto);
				}
			}
		};
	}
	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}
