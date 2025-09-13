//! # Rate of Change Percentage (ROCP)
//!
//! The Rate of Change Percentage (ROCP) calculates the relative change in value
//! between the current price and the price `period` bars ago, without the
//! extra `* 100` factor used by ROC:
//!
//! \[ ROCP[i] = (price[i] - price[i - period]) / price[i - period] \]
//!
//! This indicator is centered around 0 and can be positive or negative.
//!
//! ## Parameters
//! - **period**: The lookback window (number of data points). Defaults to 10.
//!
//! ## Errors
//! - **AllValuesNaN**: rocp: All input data values are `NaN`.
//! - **InvalidPeriod**: rocp: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: rocp: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//!
//! ## Returns
//! - **`Ok(RocpOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the moving window is filled.
//! - **`Err(RocpError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
	init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for RocpInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			RocpData::Slice(slice) => slice,
			RocpData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum RocpData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocpOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct RocpParams {
	pub period: Option<usize>,
}

impl Default for RocpParams {
	fn default() -> Self {
		Self { period: Some(10) }
	}
}

#[derive(Debug, Clone)]
pub struct RocpInput<'a> {
	pub data: RocpData<'a>,
	pub params: RocpParams,
}

impl<'a> RocpInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: RocpParams) -> Self {
		Self {
			data: RocpData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: RocpParams) -> Self {
		Self {
			data: RocpData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", RocpParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(10)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct RocpBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for RocpBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl RocpBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<RocpOutput, RocpError> {
		let p = RocpParams { period: self.period };
		let i = RocpInput::from_candles(c, "close", p);
		rocp_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<RocpOutput, RocpError> {
		let p = RocpParams { period: self.period };
		let i = RocpInput::from_slice(d, p);
		rocp_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<RocpStream, RocpError> {
		let p = RocpParams { period: self.period };
		RocpStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum RocpError {
	#[error("rocp: Input data slice is empty.")]
	EmptyInputData,
	#[error("rocp: All values are NaN.")]
	AllValuesNaN,
	#[error("rocp: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("rocp: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn rocp(input: &RocpInput) -> Result<RocpOutput, RocpError> {
	rocp_with_kernel(input, Kernel::Auto)
}

pub fn rocp_with_kernel(input: &RocpInput, kernel: Kernel) -> Result<RocpOutput, RocpError> {
	let data: &[f64] = match &input.data {
		RocpData::Candles { candles, source } => source_type(candles, source),
		RocpData::Slice(sl) => sl,
	};

	let len = data.len();
	if len == 0 {
		return Err(RocpError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocpError::AllValuesNaN)?;
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(RocpError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(RocpError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let mut out = alloc_with_nan_prefix(len, first + period);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => rocp_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rocp_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rocp_avx512(data, period, first, &mut out),
			_ => rocp_scalar(data, period, first, &mut out),
		}
	}

	Ok(RocpOutput { values: out })
}

#[inline]
pub fn rocp_into_slice(dst: &mut [f64], input: &RocpInput, kern: Kernel) -> Result<(), RocpError> {
	let data: &[f64] = match &input.data {
		RocpData::Candles { candles, source } => source_type(candles, source),
		RocpData::Slice(sl) => sl,
	};

	let len = data.len();
	if len == 0 {
		return Err(RocpError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocpError::AllValuesNaN)?;
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(RocpError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(RocpError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	if dst.len() != len {
		return Err(RocpError::InvalidPeriod {
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
			Kernel::Scalar | Kernel::ScalarBatch => rocp_scalar(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rocp_avx2(data, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rocp_avx512(data, period, first, dst),
			_ => rocp_scalar(data, period, first, dst),
		}
	}

	// Prefix-only warmup NaNs (ROCP first output at index first + period)
	for v in &mut dst[..(first + period)] {
		*v = f64::NAN;
	}

	Ok(())
}

#[inline]
pub fn rocp_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	for i in (first_val + period)..data.len() {
		let prev = data[i - period];
		out[i] = (data[i] - prev) / prev;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocp_avx512(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { rocp_avx512_short(data, period, first_val, out) }
	} else {
		unsafe { rocp_avx512_long(data, period, first_val, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocp_avx2(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	// AVX2 stub (uses scalar for now, but keeps API parity)
	rocp_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocp_avx512_short(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	// AVX512 short stub (uses scalar for now)
	rocp_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocp_avx512_long(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	// AVX512 long stub (uses scalar for now)
	rocp_scalar(data, period, first_val, out)
}

#[derive(Debug, Clone)]
pub struct RocpStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl RocpStream {
	pub fn try_new(params: RocpParams) -> Result<Self, RocpError> {
		let period = params.period.unwrap_or(10);
		if period == 0 {
			return Err(RocpError::InvalidPeriod { period, data_len: 0 });
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
		let prev = self.buffer[self.head];
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some((value - prev) / prev)
	}
}

#[derive(Clone, Debug)]
pub struct RocpBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for RocpBatchRange {
	fn default() -> Self {
		Self { period: (9, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct RocpBatchBuilder {
	range: RocpBatchRange,
	kernel: Kernel,
}

impl RocpBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<RocpBatchOutput, RocpError> {
		rocp_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RocpBatchOutput, RocpError> {
		RocpBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RocpBatchOutput, RocpError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<RocpBatchOutput, RocpError> {
		RocpBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn rocp_batch_with_kernel(data: &[f64], sweep: &RocpBatchRange, k: Kernel) -> Result<RocpBatchOutput, RocpError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(RocpError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	rocp_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RocpBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RocpParams>,
	pub rows: usize,
	pub cols: usize,
}
impl RocpBatchOutput {
	pub fn row_for_params(&self, p: &RocpParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
	}

	pub fn values_for(&self, p: &RocpParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &RocpBatchRange) -> Vec<RocpParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(RocpParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn rocp_batch_slice(data: &[f64], sweep: &RocpBatchRange, kern: Kernel) -> Result<RocpBatchOutput, RocpError> {
	rocp_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn rocp_batch_par_slice(data: &[f64], sweep: &RocpBatchRange, kern: Kernel) -> Result<RocpBatchOutput, RocpError> {
	rocp_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rocp_batch_inner(
	data: &[f64],
	sweep: &RocpBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<RocpBatchOutput, RocpError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RocpError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocpError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(RocpError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory operations like alma.rs
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Initialize NaN prefixes based on warmup periods
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap())
		.collect();
	
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Use ManuallyDrop for safer ownership transfer
	let mut guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len())
	};
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => rocp_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rocp_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rocp_row_avx512(data, first, period, out_row),
			_ => rocp_row_scalar(data, first, period, out_row),
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

	// SAFETY: Convert the buffer back to Vec
	let values = unsafe {
		Vec::from_raw_parts(
			guard.as_mut_ptr() as *mut f64,
			guard.len(),
			guard.capacity(),
		)
	};

	Ok(RocpBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn rocp_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	for i in (first + period)..data.len() {
		let prev = data[i - period];
		out[i] = (data[i] - prev) / prev;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rocp_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocp_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		rocp_row_avx512_short(data, first, period, out);
	} else {
		rocp_row_avx512_long(data, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocp_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocp_row_scalar(data, first, period, out)
}

// No changes to tests required; batch and scalar logic are unified.

#[inline(always)]
pub fn rocp_batch_inner_into(
	data: &[f64],
	sweep: &RocpBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<RocpParams>, RocpError> {
	use core::mem::MaybeUninit;

	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RocpError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocpError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(RocpError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let cols = data.len();

	// 1) Work in uninitialized space
	let out_mu: &mut [MaybeUninit<f64>] = unsafe {
		core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
	};

	// 2) Warmup NaN prefixes per row (ROCP warm = first + period)
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
	init_matrix_prefixes(out_mu, cols, &warm);

	// 3) Row writer: take MU row, view as f64, compute into it
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		let dst: &mut [f64] = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => rocp_row_scalar(data, first, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rocp_row_avx2(data, first, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rocp_row_avx512(data, first, period, dst),
			_ => rocp_row_scalar(data, first, period, dst),
		}
	};

	// 4) Iterate by MU rows
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
			out_mu.par_chunks_mut(cols).enumerate().for_each(|(r, s)| do_row(r, s));
		}
		#[cfg(target_arch = "wasm32")]
		for (r, s) in out_mu.chunks_mut(cols).enumerate() { do_row(r, s); }
	} else {
		for (r, s) in out_mu.chunks_mut(cols).enumerate() { do_row(r, s); }
	}

	Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "rocp")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn rocp_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = RocpParams { period: Some(period) };
	let rocp_in = RocpInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| rocp_with_kernel(&rocp_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "RocpStream")]
pub struct RocpStreamPy {
	stream: RocpStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RocpStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = RocpParams { period: Some(period) };
		let stream = RocpStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(RocpStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "rocp_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn rocp_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;

	let sweep = RocpBatchRange {
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
				other if other.is_batch() => other,
				_ => return Err(RocpError::InvalidPeriod { period: 0, data_len: 0 }),
			};

			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => unreachable!(),
			};

			rocp_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

#[cfg(feature = "python")]
pub fn register_rocp_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(rocp_py, m)?)?;
	m.add_function(wrap_pyfunction!(rocp_batch_py, m)?)?;
	m.add_class::<RocpStreamPy>()?;
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocp_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = RocpParams { period: Some(period) };
	let input = RocpInput::from_slice(data, params);

	let mut output = vec![0.0; data.len()];
	rocp_into_slice(&mut output, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocp_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocp_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocp_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to rocp_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = RocpParams { period: Some(period) };
		let input = RocpInput::from_slice(data, params);

		if in_ptr == out_ptr {
			// Handle aliasing case
			let mut temp = vec![0.0; len];
			rocp_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			rocp_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RocpBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RocpBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RocpParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = rocp_batch)]
pub fn rocp_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: RocpBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = RocpBatchRange {
		period: config.period_range,
	};

	let output = rocp_batch_inner(data, &sweep, detect_best_kernel(), false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = RocpBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rocp_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to rocp_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = RocpBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let total_size = rows * len;

		let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

		let kernel = detect_best_batch_kernel();
		let simd = match kernel {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			Kernel::ScalarBatch | Kernel::Scalar => Kernel::Scalar,
			_ => Kernel::Scalar,
		};

		rocp_batch_inner_into(data, &sweep, simd, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_rocp_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = RocpParams { period: None };
		let input = RocpInput::from_candles(&candles, "close", default_params);
		let output = rocp_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_rocp_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = RocpInput::from_candles(&candles, "close", RocpParams { period: Some(10) });
		let result = rocp_with_kernel(&input, kernel)?;

		let expected_last_five = [
			-0.0022551709049293996,
			-0.005561903481650759,
			-0.003275201323586514,
			-0.004945415398072297,
			-0.015045927020537019,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-9,
				"[{}] ROCP {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_rocp_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocpInput::with_default_candles(&candles);
		match input.data {
			RocpData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected RocpData::Candles"),
		}
		let output = rocp_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_rocp_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = RocpParams { period: Some(0) };
		let input = RocpInput::from_slice(&input_data, params);
		let res = rocp_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ROCP should fail with zero period", test_name);
		Ok(())
	}

	fn check_rocp_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = RocpParams { period: Some(10) };
		let input = RocpInput::from_slice(&data_small, params);
		let res = rocp_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] ROCP should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_rocp_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = RocpParams { period: Some(9) };
		let input = RocpInput::from_slice(&single_point, params);
		let res = rocp_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ROCP should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_rocp_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = RocpParams { period: Some(14) };
		let first_input = RocpInput::from_candles(&candles, "close", first_params);
		let first_result = rocp_with_kernel(&first_input, kernel)?;

		let second_params = RocpParams { period: Some(14) };
		let second_input = RocpInput::from_slice(&first_result.values, second_params);
		let second_result = rocp_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 28..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] ROCP Slice Reinput {:?} mismatch at idx {}: got NaN",
				test_name,
				kernel,
				i
			);
		}
		Ok(())
	}

	fn check_rocp_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocpInput::from_candles(&candles, "close", RocpParams { period: Some(9) });
		let res = rocp_with_kernel(&input, kernel)?;
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

	fn check_rocp_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 9;

		let input = RocpInput::from_candles(&candles, "close", RocpParams { period: Some(period) });
		let batch_output = rocp_with_kernel(&input, kernel)?.values;

		let mut stream = RocpStream::try_new(RocpParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(rocp_val) => stream_values.push(rocp_val),
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
				"[{}] ROCP streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_rocp_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			RocpParams::default(),
			RocpParams { period: Some(2) },    // minimum
			RocpParams { period: Some(5) },    // small
			RocpParams { period: Some(7) },    // small
			RocpParams { period: Some(9) },    // typical
			RocpParams { period: Some(14) },   // medium
			RocpParams { period: Some(20) },   // medium
			RocpParams { period: Some(50) },   // large
			RocpParams { period: Some(100) },  // very large
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = RocpInput::from_candles(&candles, "close", params.clone());
			let output = rocp_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

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
	fn check_rocp_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	macro_rules! generate_all_rocp_tests {
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

	generate_all_rocp_tests!(
		check_rocp_partial_params,
		check_rocp_accuracy,
		check_rocp_default_candles,
		check_rocp_zero_period,
		check_rocp_period_exceeds_length,
		check_rocp_very_small_dataset,
		check_rocp_reinput,
		check_rocp_nan_handling,
		check_rocp_streaming,
		check_rocp_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = RocpBatchBuilder::new()
			.period_static(10)
			.kernel(kernel)
			.apply_candles(&c, "close")?;

		let def = RocpParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			-0.0022551709049293996,
			-0.005561903481650759,
			-0.003275201323586514,
			-0.004945415398072297,
			-0.015045927020537019,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-9,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let test_configs = vec![
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods
			(30, 60, 15),    // Large periods
			(2, 5, 1),       // Dense small range
			(10, 50, 10),    // Wide range
			(7, 21, 7),      // Weekly periods
			(14, 28, 14),    // Bi-weekly periods
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = RocpBatchBuilder::new()
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
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
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

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_rocp_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate random period values and corresponding data vectors
		// Use non-negative values to simulate real financial data (prices >= 0)
		let strat = (1usize..=64)
			.prop_flat_map(|period| {
				(
					prop::collection::vec(
						(0.01f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						period..400,
					),
					Just(period),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = RocpParams {
					period: Some(period),
				};
				let input = RocpInput::from_slice(&data, params);

				// Calculate ROCP with the specified kernel
				let RocpOutput { values: out } = rocp_with_kernel(&input, kernel).unwrap();
				
				// Calculate reference with scalar kernel for consistency check
				let RocpOutput { values: ref_out } = rocp_with_kernel(&input, Kernel::Scalar).unwrap();

				// Find first valid index
				let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				
				// Verify outputs starting from warmup period
				for i in (first_valid + period)..data.len() {
					let prev_value = data[i - period];
					let curr_value = data[i];
					let y = out[i];
					let r = ref_out[i];

					// Mathematical correctness: ROCP = (current - previous) / previous
					if prev_value != 0.0 && prev_value.is_finite() && curr_value.is_finite() {
						let expected = (curr_value - prev_value) / prev_value;
						
						// Check mathematical correctness (with tolerance for floating-point)
						prop_assert!(
							y.is_nan() || (y - expected).abs() <= 1e-10,
							"idx {}: ROCP mismatch. Got {}, expected {}, curr={}, prev={}",
							i, y, expected, curr_value, prev_value
						);
					} else if prev_value == 0.0 {
						// Division by zero should produce infinity or NaN
						prop_assert!(
							!y.is_finite(),
							"idx {}: Expected non-finite value when dividing by zero, got {}",
							i, y
						);
					}

					// Special case: constant data (using relative difference for better scale handling)
					let is_constant = data[first_valid..=i].windows(2).all(|w| {
						let max_val = w[0].max(w[1]);
						if max_val > 0.0 {
							(w[0] - w[1]).abs() / max_val < 1e-10
						} else {
							w[0] == w[1]
						}
					});
					
					if is_constant && data[first_valid].is_finite() && data[first_valid] != 0.0 {
						// If all values are the same, ROCP should be 0
						prop_assert!(
							y.abs() <= 1e-10,
							"Constant data should produce ROCP=0, got {} at idx {}",
							y, i
						);
					}
					
					// Special case: monotonic sequences
					// For strictly increasing data, ROCP should be positive
					let is_increasing = i >= first_valid + period && 
						(first_valid..=i).all(|j| j == first_valid || data[j] > data[j-1]);
					
					if is_increasing && y.is_finite() {
						prop_assert!(
							y > -1e-10,  // Allow small negative due to floating point
							"Strictly increasing data should produce positive ROCP, got {} at idx {}",
							y, i
						);
					}
					
					// For strictly decreasing data, ROCP should be negative
					let is_decreasing = i >= first_valid + period && 
						(first_valid..=i).all(|j| j == first_valid || data[j] < data[j-1]);
					
					if is_decreasing && y.is_finite() {
						prop_assert!(
							y < 1e-10,  // Allow small positive due to floating point
							"Strictly decreasing data should produce negative ROCP, got {} at idx {}",
							y, i
						);
					}

					// Kernel consistency check
					let y_bits = y.to_bits();
					let r_bits = r.to_bits();

					if !y.is_finite() || !r.is_finite() {
						// NaN/Inf should match exactly
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"finite/NaN mismatch idx {}: {} vs {}",
							i, y, r
						);
						continue;
					}

					// ULP comparison for floating-point accuracy
					let ulp_diff: u64 = y_bits.abs_diff(r_bits);

					prop_assert!(
						(y - r).abs() <= 1e-10 || ulp_diff <= 4,
						"Kernel mismatch idx {}: {} vs {} (ULP={})",
						i, y, r, ulp_diff
					);

				}

				// Verify warmup period contains NaN
				for i in 0..(first_valid + period).min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at idx {}, got {}",
						i, out[i]
					);
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	#[cfg(feature = "proptest")]
	generate_all_rocp_tests!(check_rocp_property);
}
