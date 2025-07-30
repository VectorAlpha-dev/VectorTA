//! # Chaikin's Volatility (CVI)
//!
//! Chaikin's Volatility (CVI) measures the volatility of a financial instrument by calculating
//! the percentage difference between two exponentially smoothed averages of the trading range
//! (high-low) over a given period. A commonly used default period is 10. Higher values for the
//! period will smooth out short-term fluctuations, while lower values will track rapid changes
//! more closely.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 10.
//!
//! ## Errors
//! - **EmptyData**: cvi: Input data (high/low) is empty.
//! - **InvalidPeriod**: cvi: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: cvi: Fewer than `2*period - 1` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: cvi: All input high/low values are `NaN`.
//!
//! ## Returns
//! - **`Ok(CviOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the first calculable index (at `2*period - 1` from the first
//!   valid data point).
//! - **`Err(CviError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for CviInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			CviData::Slices { high, .. } => high,
			CviData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum CviData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct CviOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CviParams {
	pub period: Option<usize>,
}

impl Default for CviParams {
	fn default() -> Self {
		Self { period: Some(10) }
	}
}

#[derive(Debug, Clone)]
pub struct CviInput<'a> {
	pub data: CviData<'a>,
	pub params: CviParams,
}

impl<'a> CviInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, source: &'a str, params: CviParams) -> Self {
		Self {
			data: CviData::Candles { candles, source },
			params,
		}
	}

	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: CviParams) -> Self {
		Self {
			data: CviData::Slices { high, low },
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "hl2", CviParams::default())
	}

	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(10)
	}
}

#[derive(Debug, Copy, Clone)]
pub struct CviBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for CviBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl CviBuilder {
	#[inline]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline]
	pub fn period(mut self, n: usize) -> Self {
		self.period = Some(n);
		self
	}
	#[inline]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline]
	pub fn apply(self, candles: &Candles) -> Result<CviOutput, CviError> {
		let params = CviParams { period: self.period };
		let input = CviInput::from_candles(candles, "hl2", params);
		cvi_with_kernel(&input, self.kernel)
	}

	#[inline]
	pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<CviOutput, CviError> {
		let params = CviParams { period: self.period };
		let input = CviInput::from_slices(high, low, params);
		cvi_with_kernel(&input, self.kernel)
	}

	#[inline]
	pub fn into_stream(self, initial_high: f64, initial_low: f64) -> Result<CviStream, CviError> {
		let params = CviParams { period: self.period };
		CviStream::try_new(params, initial_high, initial_low)
	}
}

#[derive(Debug, Error)]
pub enum CviError {
	#[error("cvi: Empty data provided for CVI.")]
	EmptyData,
	#[error("cvi: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("cvi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("cvi: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn cvi(input: &CviInput) -> Result<CviOutput, CviError> {
	cvi_with_kernel(input, Kernel::Auto)
}

pub fn cvi_with_kernel(input: &CviInput, kernel: Kernel) -> Result<CviOutput, CviError> {
	let (high, low) = match &input.data {
		CviData::Candles { candles, source: _ } => {
			if candles.high.is_empty() || candles.low.is_empty() {
				return Err(CviError::EmptyData);
			}
			(&candles.high[..], &candles.low[..])
		}
		CviData::Slices { high, low } => {
			if high.is_empty() || low.is_empty() {
				return Err(CviError::EmptyData);
			}
			(*high, *low)
		}
	};

	if high.len() != low.len() {
		return Err(CviError::EmptyData);
	}

	let period = input.get_period();
	if period == 0 || period > high.len() {
		return Err(CviError::InvalidPeriod {
			period,
			data_len: high.len(),
		});
	}

	let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
		Some(idx) => idx,
		None => return Err(CviError::AllValuesNaN),
	};

	let needed = 2 * period - 1;
	if (high.len() - first_valid_idx) < needed {
		return Err(CviError::NotEnoughValidData {
			needed,
			valid: high.len() - first_valid_idx,
		});
	}

	let mut cvi_values = alloc_with_nan_prefix(high.len(), first_valid_idx + needed - 1);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => cvi_scalar(high, low, period, first_valid_idx, &mut cvi_values),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => cvi_avx2(high, low, period, first_valid_idx, &mut cvi_values),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => cvi_avx512(high, low, period, first_valid_idx, &mut cvi_values),
			_ => unreachable!(),
		}
	}

	Ok(CviOutput { values: cvi_values })
}

#[inline]
pub fn cvi_scalar(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	let alpha = 2.0 / (period as f64 + 1.0);
	let mut val = high[first_valid_idx] - low[first_valid_idx];
	let mut lag_buffer = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
	lag_buffer.resize(period, 0.0);
	lag_buffer[0] = val;
	let mut head = 1;

	let needed = 2 * period - 1;
	for i in (first_valid_idx + 1)..(first_valid_idx + needed) {
		let range = high[i] - low[i];
		val += (range - val) * alpha;
		lag_buffer[head] = val;
		head = (head + 1) % period;
	}
	for i in (first_valid_idx + needed)..high.len() {
		let range = high[i] - low[i];
		val += (range - val) * alpha;
		let old = lag_buffer[head];
		out[i] = 100.0 * (val - old) / old;
		lag_buffer[head] = val;
		head = (head + 1) % period;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cvi_avx2(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	cvi_scalar(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cvi_avx512(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { cvi_avx512_short(high, low, period, first_valid_idx, out) }
	} else {
		unsafe { cvi_avx512_long(high, low, period, first_valid_idx, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cvi_avx512_short(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	cvi_scalar(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cvi_avx512_long(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	cvi_scalar(high, low, period, first_valid_idx, out)
}

#[inline(always)]
pub fn cvi_row_scalar(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	cvi_scalar(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx2(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	cvi_avx2(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx512(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	cvi_avx512(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx512_short(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	unsafe { cvi_avx512_short(high, low, period, first_valid_idx, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx512_long(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	unsafe { cvi_avx512_long(high, low, period, first_valid_idx, out) }
}

#[derive(Clone, Debug)]
pub struct CviBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for CviBatchRange {
	fn default() -> Self {
		Self { period: (10, 20, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct CviBatchBuilder {
	range: CviBatchRange,
	kernel: Kernel,
}

impl CviBatchBuilder {
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
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<CviBatchOutput, CviError> {
		cvi_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn apply_candles(self, candles: &Candles) -> Result<CviBatchOutput, CviError> {
		self.apply_slices(&candles.high, &candles.low)
	}
}

pub fn cvi_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &CviBatchRange,
	k: Kernel,
) -> Result<CviBatchOutput, CviError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(CviError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	cvi_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CviBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CviParams>,
	pub rows: usize,
	pub cols: usize,
}
impl CviBatchOutput {
	pub fn row_for_params(&self, p: &CviParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
	}

	pub fn values_for(&self, p: &CviParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &CviBatchRange) -> Vec<CviParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(CviParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn cvi_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &CviBatchRange,
	kern: Kernel,
) -> Result<CviBatchOutput, CviError> {
	cvi_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn cvi_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &CviBatchRange,
	kern: Kernel,
) -> Result<CviBatchOutput, CviError> {
	cvi_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn cvi_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &CviBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<CviBatchOutput, CviError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(CviError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first_valid_idx = (0..high.len())
		.find(|&i| !high[i].is_nan() && !low[i].is_nan())
		.ok_or(CviError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let needed = 2 * max_p - 1;
	if high.len() - first_valid_idx < needed {
		return Err(CviError::NotEnoughValidData {
			needed,
			valid: high.len() - first_valid_idx,
		});
	}

	let rows = combos.len();
	let cols = high.len();

	let mut buf_mu = make_uninit_matrix(rows, cols);
	let warmup_periods: Vec<usize> = combos.iter().map(|c| {
		let period = c.period.unwrap();
		first_valid_idx + 2 * period - 2
	}).collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => cvi_row_scalar(high, low, period, first_valid_idx, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => cvi_row_avx2(high, low, period, first_valid_idx, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => cvi_row_avx512(high, low, period, first_valid_idx, out_row),
			_ => unreachable!(),
		}
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

	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	core::mem::forget(buf_guard);

	Ok(CviBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn cvi_batch_inner_into(
	high: &[f64],
	low: &[f64],
	sweep: &CviBatchRange,
	kern: Kernel,
	parallel: bool,
	output: &mut [f64],
) -> Result<Vec<CviParams>, CviError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(CviError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first_valid_idx = (0..high.len())
		.find(|&i| !high[i].is_nan() && !low[i].is_nan())
		.ok_or(CviError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let needed = 2 * max_p - 1;
	if high.len() - first_valid_idx < needed {
		return Err(CviError::NotEnoughValidData {
			needed,
			valid: high.len() - first_valid_idx,
		});
	}

	let rows = combos.len();
	let cols = high.len();

	let do_row = |row: usize, out_row: &mut [f64]| {
		let cvi_params = &combos[row];
		let params = cvi_params.clone();
		let cvi_in = CviInput::from_slices(high, low, params);
		if let Ok(output) = cvi_with_kernel(&cvi_in, kern) {
			out_row.copy_from_slice(&output.values);
		}
	};

	#[cfg(not(target_arch = "wasm32"))]
	{
		if parallel {
			output.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| {
				do_row(row, slice);
			});
		} else {
			for (row, slice) in output.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	}
	#[cfg(target_arch = "wasm32")]
	{
		for (row, slice) in output.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	Ok(combos)
}

// For streaming: not a natural fit for CVI, but provide parity.
#[derive(Debug, Clone)]
pub struct CviStream {
	period: usize,
	alpha: f64,
	lag_buffer: Vec<f64>,
	head: usize,
	filled: bool,
	state_val: f64,
}

impl CviStream {
	pub fn try_new(params: CviParams, initial_high: f64, initial_low: f64) -> Result<Self, CviError> {
		let period = params.period.unwrap_or(10);
		if period == 0 {
			return Err(CviError::InvalidPeriod { period, data_len: 0 });
		}
		let alpha = 2.0 / (period as f64 + 1.0);
		let val = initial_high - initial_low;
		let mut lag_buffer = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
		lag_buffer.resize(period, 0.0);
		lag_buffer[0] = val;
		// Convert AVec to Vec
		let lag_buffer_vec: Vec<f64> = lag_buffer.into_iter().copied().collect();
		Ok(Self {
			period,
			alpha,
			lag_buffer: lag_buffer_vec,
			head: 1,
			filled: false,
			state_val: val,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		let range = high - low;
		self.state_val += (range - self.state_val) * self.alpha;
		
		self.lag_buffer[self.head] = self.state_val;
		self.head = (self.head + 1) % self.period;
		
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		
		if !self.filled {
			return None;
		}
		
		let old_idx = self.head; // The oldest value is at the current head position
		let old = self.lag_buffer[old_idx];
		Some(100.0 * (self.state_val - old) / old)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "cvi")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn cvi_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = CviParams { period: Some(period) };
	let cvi_in = CviInput::from_slices(high_slice, low_slice, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| cvi_with_kernel(&cvi_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CviStream")]
pub struct CviStreamPy {
	stream: CviStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CviStreamPy {
	#[new]
	fn new(period: usize, initial_high: f64, initial_low: f64) -> PyResult<Self> {
		let params = CviParams { period: Some(period) };
		let stream = CviStream::try_new(params, initial_high, initial_low).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(CviStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "cvi_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn cvi_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;

	let sweep = CviBatchRange { period: period_range };
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = high_slice.len();

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
			cvi_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos
			.iter()
			.map(|p| p.period.unwrap_or(10) as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict.into())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_cvi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = CviParams { period: None };
		let input_default = CviInput::from_candles(&candles, "close", default_params);
		let output_default = cvi_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());
		Ok(())
	}

	fn check_cvi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = CviParams { period: Some(5) };
		let input = CviInput::from_candles(&candles, "close", params);
		let cvi_result = cvi_with_kernel(&input, kernel)?;

		let expected_last_five_cvi = [
			-52.96320026271643,
			-64.39616778235792,
			-59.4830094380472,
			-52.4690724045071,
			-11.858704179539174,
		];
		assert!(cvi_result.values.len() >= 5);
		let start_index = cvi_result.values.len() - 5;
		let result_last_five = &cvi_result.values[start_index..];
		for (i, &val) in result_last_five.iter().enumerate() {
			let expected = expected_last_five_cvi[i];
			assert!(
				(val - expected).abs() < 1e-6,
				"[{}] CVI mismatch at index {}: expected {}, got {}",
				test_name,
				i,
				expected,
				val
			);
		}
		Ok(())
	}

	fn check_cvi_input_with_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = CviInput::with_default_candles(&candles);
		match input.data {
			CviData::Candles { .. } => {}
			_ => panic!("Expected CviData::Candles variant"),
		}
		Ok(())
	}

	fn check_cvi_with_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = CviParams { period: Some(0) };
		let input = CviInput::from_slices(&high, &low, params);

		let result = cvi_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] Expected an error for zero period", test_name);
		Ok(())
	}

	fn check_cvi_with_period_exceeding_data_length(
		test_name: &str,
		kernel: Kernel,
	) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = CviParams { period: Some(10) };
		let input = CviInput::from_slices(&high, &low, params);

		let result = cvi_with_kernel(&input, kernel);
		assert!(
			result.is_err(),
			"[{}] Expected an error for period > data.len()",
			test_name
		);
		Ok(())
	}

	fn check_cvi_very_small_data_set(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [5.0];
		let low = [2.0];
		let params = CviParams { period: Some(10) };
		let input = CviInput::from_slices(&high, &low, params);

		let result = cvi_with_kernel(&input, kernel);
		assert!(
			result.is_err(),
			"[{}] Expected error for data smaller than period",
			test_name
		);
		Ok(())
	}

	fn check_cvi_with_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, 20.0, 30.0];
		let low = [5.0, 15.0, f64::NAN];
		let input = CviInput::from_slices(&high, &low, CviParams { period: Some(2) });

		let result = cvi_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] Expected an error due to trailing NaN", test_name);
		Ok(())
	}

	fn check_cvi_slice_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 12.0, 12.5, 12.2, 13.0, 14.0, 15.0, 16.0, 16.5, 17.0, 17.5, 18.0];
		let low = [9.0, 10.0, 11.5, 11.0, 12.0, 13.5, 14.0, 14.5, 15.5, 16.0, 16.5, 17.0];
		let first_input = CviInput::from_slices(&high, &low, CviParams { period: Some(3) });
		let first_result = cvi_with_kernel(&first_input, kernel)?;
		let second_input = CviInput::from_slices(&first_result.values, &low, CviParams { period: Some(3) });
		let second_result = cvi_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), low.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_cvi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations for CVI
		let test_params = vec![
			CviParams::default(),                    // period: 10
			CviParams { period: Some(2) },          // minimum viable period
			CviParams { period: Some(5) },          // small period
			CviParams { period: Some(10) },         // default
			CviParams { period: Some(14) },         // common period
			CviParams { period: Some(20) },         // medium period
			CviParams { period: Some(50) },         // large period
			CviParams { period: Some(100) },        // very large period
			CviParams { period: Some(200) },        // extreme period
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			// Test with high/low data
			let input = CviInput::from_candles(&candles, "high", params.clone());
			let output = cvi_with_kernel(&input, kernel)?;
			
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
						params.period.unwrap_or(10), param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10), param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(10), param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_cvi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_cvi_tests {
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

	generate_all_cvi_tests!(
		check_cvi_partial_params,
		check_cvi_accuracy,
		check_cvi_input_with_default_candles,
		check_cvi_with_zero_period,
		check_cvi_with_period_exceeding_data_length,
		check_cvi_very_small_data_set,
		check_cvi_with_nan_data,
		check_cvi_slice_reinput,
		check_cvi_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = CviBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = CviParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
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

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations for CVI
		let test_configs = vec![
			(2, 10, 2),      // Small periods with step 2
			(5, 25, 5),      // Medium periods with step 5
			(10, 50, 10),    // Common periods
			(2, 5, 1),       // Dense small range
			(30, 60, 15),    // Large periods
			(14, 21, 7),     // Common trading periods
			(50, 100, 25),   // Very large periods
		];
		
		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = CviBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c)?;
			
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
						combo.period.unwrap_or(10)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(10)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(10)
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}
}

/// Helper function to find the first valid index where both high and low are not NaN
#[inline]
fn find_first_valid_idx(high: &[f64], low: &[f64]) -> Option<usize> {
	(0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan())
}

/// Core helper function that writes directly to an output slice.
/// This is used by WASM bindings to enable zero-copy operations.
#[inline(always)]
pub fn cvi_into_slice(output: &mut [f64], input: &CviInput, kernel: Kernel) -> Result<(), CviError> {
	let (high, low) = match &input.data {
		CviData::Candles { candles, source } => {
			let data = source_type(candles, source);
			(data, data) // For CVI we need high/low separately, this is a limitation
		}
		CviData::Slices { high, low } => (*high, *low),
	};

	let period = input.params.period.unwrap_or(10);

	// Validate inputs
	if high.is_empty() || low.is_empty() {
		return Err(CviError::EmptyData);
	}
	if period == 0 || period > high.len() {
		return Err(CviError::InvalidPeriod { period, data_len: high.len() });
	}
	if high.len() != low.len() || output.len() != high.len() {
		return Err(CviError::EmptyData);
	}

	// Find first valid index
	let first_valid_idx = match find_first_valid_idx(high, low) {
		Some(idx) => idx,
		None => return Err(CviError::AllValuesNaN),
	};

	let warmup = period - 1;
	let min_data_points = warmup + period;

	// Check we have enough valid data
	if high.len() - first_valid_idx < min_data_points {
		return Err(CviError::NotEnoughValidData { 
			needed: min_data_points, 
			valid: high.len() - first_valid_idx 
		});
	}

	// Fill prefix with NaN
	let out_start = first_valid_idx + warmup + period - 1;
	for i in 0..out_start {
		output[i] = f64::NAN;
	}

	// Use kernel-specific implementation
	match kernel {
		Kernel::Scalar => cvi_scalar(high, low, period, first_valid_idx, output),
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 => unsafe { cvi_avx2(high, low, period, first_valid_idx, output) },
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 => unsafe { cvi_avx512(high, low, period, first_valid_idx, output) },
		Kernel::Auto => match detect_best_kernel() {
			Kernel::Scalar => cvi_scalar(high, low, period, first_valid_idx, output),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => unsafe { cvi_avx2(high, low, period, first_valid_idx, output) },
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => unsafe { cvi_avx512(high, low, period, first_valid_idx, output) },
			_ => cvi_scalar(high, low, period, first_valid_idx, output),
		},
		_ => return Err(CviError::InvalidPeriod { period, data_len: high.len() }),
	}

	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_js(high: &[f64], low: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = CviParams { period: Some(period) };
	let input = CviInput::from_slices(high, low, params);

	// Use the main cvi function which already uses alloc_with_nan_prefix
	let output = cvi(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to cvi_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);

		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = CviParams { period: Some(period) };
		let input = CviInput::from_slices(high, low, params);

		// Check for aliasing with output
		let aliased = high_ptr == out_ptr || low_ptr == out_ptr;

		if aliased {
			// Use the main cvi function which uses alloc_with_nan_prefix for aliasing case
			let result = cvi(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&result.values);
		} else {
			// Direct write if not aliased
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			cvi_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CviBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CviBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CviParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cvi_batch)]
pub fn cvi_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: CviBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = CviBatchRange {
		period: config.period_range,
	};

	let output = cvi_batch_inner(high, low, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = CviBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to cvi_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);

		let sweep = CviBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		cvi_batch_inner_into(high, low, &sweep, Kernel::Auto, false, out).map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}
