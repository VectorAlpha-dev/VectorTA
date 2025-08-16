//! # Midpoint Indicator
//!
//! Calculates the midpoint of the highest and lowest value over a given window (`period`).
//! Returns a vector matching the input size, with leading NaNs for incomplete windows.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: midpoint: All input data values are `NaN`.
//! - **InvalidPeriod**: midpoint: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: midpoint: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(MidpointOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(MidpointError)`** otherwise.

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

// --- INPUT/OUTPUT TYPES ---

#[derive(Debug, Clone)]
pub enum MidpointData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for MidpointInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			MidpointData::Slice(slice) => slice,
			MidpointData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub struct MidpointOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MidpointParams {
	pub period: Option<usize>,
}

impl Default for MidpointParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct MidpointInput<'a> {
	pub data: MidpointData<'a>,
	pub params: MidpointParams,
}

impl<'a> MidpointInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: MidpointParams) -> Self {
		Self {
			data: MidpointData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: MidpointParams) -> Self {
		Self {
			data: MidpointData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", MidpointParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

// --- BUILDER ---

#[derive(Copy, Clone, Debug)]
pub struct MidpointBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for MidpointBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl MidpointBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<MidpointOutput, MidpointError> {
		let p = MidpointParams { period: self.period };
		let i = MidpointInput::from_candles(c, "close", p);
		midpoint_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<MidpointOutput, MidpointError> {
		let p = MidpointParams { period: self.period };
		let i = MidpointInput::from_slice(d, p);
		midpoint_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<MidpointStream, MidpointError> {
		let p = MidpointParams { period: self.period };
		MidpointStream::try_new(p)
	}
}

// --- ERROR ---

#[derive(Debug, Error)]
pub enum MidpointError {
	#[error("midpoint: All values are NaN.")]
	AllValuesNaN,
	#[error("midpoint: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("midpoint: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[cfg(feature = "wasm")]
impl From<MidpointError> for JsValue {
	fn from(err: MidpointError) -> Self {
		JsValue::from_str(&err.to_string())
	}
}

// --- INDICATOR API ---

#[inline]
pub fn midpoint(input: &MidpointInput) -> Result<MidpointOutput, MidpointError> {
	midpoint_with_kernel(input, Kernel::Auto)
}

pub fn midpoint_with_kernel(input: &MidpointInput, kernel: Kernel) -> Result<MidpointOutput, MidpointError> {
	let data: &[f64] = input.as_ref();

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(MidpointError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(MidpointError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(MidpointError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let mut out = alloc_with_nan_prefix(len, first + period - 1);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => midpoint_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => midpoint_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => midpoint_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(MidpointOutput { values: out })
}

#[inline]
pub fn midpoint_into_slice(out: &mut [f64], input: &MidpointInput, kernel: Kernel) -> Result<(), MidpointError> {
	let data: &[f64] = input.as_ref();

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(MidpointError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(MidpointError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(MidpointError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	if out.len() != len {
		return Err(MidpointError::InvalidPeriod { period: out.len(), data_len: len });
	}

	// Initialize NaN prefix
	for i in 0..(first + period - 1) {
		out[i] = f64::NAN;
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => midpoint_scalar(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => midpoint_avx2(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => midpoint_avx512(data, period, first, out),
			_ => unreachable!(),
		}
	}

	Ok(())
}

// --- SIMD STUBS ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	unsafe { midpoint_avx512_long(data, period, first, out) }
}

#[inline]
pub fn midpoint_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	for i in (first + period - 1)..data.len() {
		let window = &data[(i + 1 - period)..=i];
		let mut highest = f64::MIN;
		let mut lowest = f64::MAX;
		for &val in window {
			if val > highest {
				highest = val;
			}
			if val < lowest {
				lowest = val;
			}
		}
		out[i] = (highest + lowest) / 2.0;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn midpoint_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}

// --- BATCH / STREAMING ---

#[derive(Debug, Clone)]
pub struct MidpointStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl MidpointStream {
	pub fn try_new(params: MidpointParams) -> Result<Self, MidpointError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(MidpointError::InvalidPeriod { period, data_len: 0 });
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
		Some(self.calc_midpoint())
	}
	#[inline(always)]
	fn calc_midpoint(&self) -> f64 {
		let mut highest = f64::MIN;
		let mut lowest = f64::MAX;
		let mut idx = self.head;
		for _ in 0..self.period {
			let v = self.buffer[idx];
			if v > highest {
				highest = v;
			}
			if v < lowest {
				lowest = v;
			}
			idx = (idx + 1) % self.period;
		}
		(highest + lowest) / 2.0
	}
}

// --- BATCH API (Parameter Sweep) ---

#[derive(Clone, Debug)]
pub struct MidpointBatchRange {
	pub period: (usize, usize, usize),
}
impl Default for MidpointBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}
#[derive(Clone, Debug, Default)]
pub struct MidpointBatchBuilder {
	range: MidpointBatchRange,
	kernel: Kernel,
}
impl MidpointBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.period = (start, end, step);
		self
	}
	pub fn period_static(mut self, p: usize) -> Self {
		self.range.period = (p, p, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<MidpointBatchOutput, MidpointError> {
		midpoint_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MidpointBatchOutput, MidpointError> {
		MidpointBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MidpointBatchOutput, MidpointError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<MidpointBatchOutput, MidpointError> {
		MidpointBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

pub fn midpoint_batch_with_kernel(
	data: &[f64],
	sweep: &MidpointBatchRange,
	k: Kernel,
) -> Result<MidpointBatchOutput, MidpointError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(MidpointError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	midpoint_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MidpointBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<MidpointParams>,
	pub rows: usize,
	pub cols: usize,
}
impl MidpointBatchOutput {
	pub fn row_for_params(&self, p: &MidpointParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &MidpointParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

// --- BATCH INNER ---

#[inline(always)]
fn expand_grid(r: &MidpointBatchRange) -> Vec<MidpointParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(MidpointParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn midpoint_batch_slice(
	data: &[f64],
	sweep: &MidpointBatchRange,
	kern: Kernel,
) -> Result<MidpointBatchOutput, MidpointError> {
	midpoint_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn midpoint_batch_par_slice(
	data: &[f64],
	sweep: &MidpointBatchRange,
	kern: Kernel,
) -> Result<MidpointBatchOutput, MidpointError> {
	midpoint_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn midpoint_batch_inner(
	data: &[f64],
	sweep: &MidpointBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<MidpointBatchOutput, MidpointError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MidpointError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(MidpointError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(MidpointError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory allocation like ALMA
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each combo
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to regular slice
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) 
	};
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => midpoint_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => midpoint_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => midpoint_row_avx512(data, first, period, out_row),
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
	
	// Convert uninitialized buffer back to Vec
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity()
		)
	};
	core::mem::forget(buf_guard);
	
	Ok(MidpointBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

// --- ROW KERNELS ---

#[inline(always)]
unsafe fn midpoint_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn midpoint_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	midpoint_scalar(data, period, first, out)
}

// --- TESTS ---

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	#[cfg(feature = "proptest")]
	use proptest::prelude::*;

	fn check_midpoint_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = MidpointParams { period: None };
		let input = MidpointInput::from_candles(&candles, "close", default_params);
		let output = midpoint_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_midpoint_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MidpointInput::from_candles(&candles, "close", MidpointParams::default());
		let result = midpoint_with_kernel(&input, kernel)?;
		let expected_last_five = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] MIDPOINT {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}
	fn check_midpoint_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MidpointInput::with_default_candles(&candles);
		match input.data {
			MidpointData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected MidpointData::Candles"),
		}
		let output = midpoint_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_midpoint_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = MidpointParams { period: Some(0) };
		let input = MidpointInput::from_slice(&input_data, params);
		let res = midpoint_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MIDPOINT should fail with zero period", test_name);
		Ok(())
	}
	fn check_midpoint_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = MidpointParams { period: Some(10) };
		let input = MidpointInput::from_slice(&data_small, params);
		let res = midpoint_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] MIDPOINT should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_midpoint_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = MidpointParams { period: Some(9) };
		let input = MidpointInput::from_slice(&single_point, params);
		let res = midpoint_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] MIDPOINT should fail with insufficient data",
			test_name
		);
		Ok(())
	}
	fn check_midpoint_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = MidpointParams { period: Some(14) };
		let first_input = MidpointInput::from_candles(&candles, "close", first_params);
		let first_result = midpoint_with_kernel(&first_input, kernel)?;
		let second_params = MidpointParams { period: Some(14) };
		let second_input = MidpointInput::from_slice(&first_result.values, second_params);
		let second_result = midpoint_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}
	fn check_midpoint_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MidpointInput::from_candles(&candles, "close", MidpointParams::default());
		let res = midpoint_with_kernel(&input, kernel)?;
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
	fn check_midpoint_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 14;
		let input = MidpointInput::from_candles(&candles, "close", MidpointParams { period: Some(period) });
		let batch_output = midpoint_with_kernel(&input, kernel)?.values;
		let mut stream = MidpointStream::try_new(MidpointParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(mid_val) => stream_values.push(mid_val),
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
				"[{}] MIDPOINT streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_midpoint_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			MidpointParams::default(),  // period: 14
			MidpointParams { period: Some(2) },  // minimum viable
			MidpointParams { period: Some(5) },  // small
			MidpointParams { period: Some(7) },  // small
			MidpointParams { period: Some(10) }, // small-medium
			MidpointParams { period: Some(20) }, // medium
			MidpointParams { period: Some(30) }, // medium-large
			MidpointParams { period: Some(50) }, // large
			MidpointParams { period: Some(100) }, // very large
			MidpointParams { period: Some(200) }, // extra large
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = MidpointInput::from_candles(&candles, "close", params.clone());
			let output = midpoint_with_kernel(&input, kernel)?;
			
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
						test_name, val, bits, i, params.period.unwrap_or(14), param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(14), param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(14), param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_midpoint_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_midpoint_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		// Strategy for generating test data with multiple scenarios
		let strat = (1usize..50, 50usize..400, 0usize..9, any::<u64>())
			.prop_map(|(period, len, scenario, seed)| {
				// LCG-based deterministic RNG
				let mut lcg = seed;
				let mut rng = || {
					lcg = lcg.wrapping_mul(1103515245).wrapping_add(12345);
					((lcg / 65536) % 1000000) as f64 / 10000.0 - 50.0
				};
				
				let data = match scenario {
					0 => {
						// Random data
						(0..len).map(|_| rng()).collect()
					}
					1 => {
						// Constant value
						let val = rng();
						vec![val; len]
					}
					2 => {
						// Monotonic increasing
						let start = rng();
						let step = rng().abs() / 100.0;
						(0..len).map(|i| start + (i as f64) * step).collect()
					}
					3 => {
						// Monotonic decreasing
						let start = rng();
						let step = rng().abs() / 100.0;
						(0..len).map(|i| start - (i as f64) * step).collect()
					}
					4 => {
						// Extreme ranges
						(0..len).map(|i| {
							if i % 2 == 0 { 1000.0 + rng() } else { -1000.0 + rng() }
						}).collect()
					}
					5 => {
						// Sine wave pattern
						let amplitude = rng().abs() + 10.0;
						let offset = rng();
						(0..len).map(|i| {
							offset + amplitude * (i as f64 * 0.1).sin()
						}).collect()
					}
					6 => {
						// Large values (realistic for financial data)
						(0..len).map(|_| rng() * 1e6).collect()
					}
					7 => {
						// Small values (penny stocks, fractional shares)
						(0..len).map(|_| rng() * 1e-3).collect()
					}
					8 => {
						// Mixed scale values (diversified portfolio)
						(0..len).map(|i| {
							if i % 3 == 0 { rng() * 1e6 } 
							else if i % 3 == 1 { rng() * 1e-3 } 
							else { rng() }
						}).collect()
					}
					_ => {
						// Triangle wave pattern
						let amplitude = rng().abs() + 10.0;
						let period_len = 20;
						(0..len).map(|i| {
							let phase = (i % period_len) as f64 / period_len as f64;
							if phase < 0.5 {
								amplitude * (2.0 * phase)
							} else {
								amplitude * (2.0 - 2.0 * phase)
							}
						}).collect()
					}
				};
				
				(data, period, scenario)
			});
		
		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period, scenario)| {
				let params = MidpointParams { period: Some(period) };
				let input = MidpointInput::from_slice(&data, params);
				
				let result = midpoint_with_kernel(&input, kernel)?;
				let scalar_result = midpoint_with_kernel(&input, Kernel::Scalar)?;
				
				// Adaptive tolerance based on value magnitude
				let tolerance = |expected: f64| -> f64 {
					// Use relative tolerance for large values, absolute for small
					(expected.abs() * 1e-12).max(1e-10)
				};
				
				// Property 1: Output length matches input
				prop_assert_eq!(result.values.len(), data.len());
				
				// Find first non-NaN value
				let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				let warmup_end = first + period - 1;
				
				// Property 2: First valid index behavior
				for i in 0..warmup_end.min(result.values.len()) {
					prop_assert!(
						result.values[i].is_nan(),
						"Expected NaN at index {} during warmup (warmup_end={})",
						i, warmup_end
					);
				}
				
				// Property 3: Mathematical accuracy
				for i in warmup_end..data.len() {
					let window = &data[(i + 1 - period)..=i];
					let mut highest = f64::MIN;
					let mut lowest = f64::MAX;
					
					for &val in window {
						if val > highest {
							highest = val;
						}
						if val < lowest {
							lowest = val;
						}
					}
					
					let expected = (highest + lowest) / 2.0;
					let actual = result.values[i];
					let tol = tolerance(expected);
					
					prop_assert!(
						(actual - expected).abs() < tol,
						"Mathematical accuracy failed at index {}: expected {}, got {}, tolerance {}",
						i, expected, actual, tol
					);
				}
				
				// Property 4: Kernel consistency
				for i in 0..result.values.len() {
					let kernel_val = result.values[i];
					let scalar_val = scalar_result.values[i];
					
					if kernel_val.is_nan() && scalar_val.is_nan() {
						continue;
					}
					
					let tol = tolerance(scalar_val);
					prop_assert!(
						(kernel_val - scalar_val).abs() < tol,
						"Kernel consistency failed at index {}: kernel={}, scalar={}, tolerance={}",
						i, kernel_val, scalar_val, tol
					);
				}
				
				// Property 5: Special case - period = 1
				if period == 1 {
					for i in first..data.len() {
						let tol = tolerance(data[i]);
						prop_assert!(
							(result.values[i] - data[i]).abs() < tol,
							"Period=1 should equal input at index {}: {} vs {}, tolerance {}",
							i, result.values[i], data[i], tol
						);
					}
				}
				
				// Property 6: Special case - constant data
				if !data.is_empty() {
					let first_val = data[first];
					let val_tol = tolerance(first_val);
					if data.windows(2).all(|w| (w[0] - w[1]).abs() < val_tol) {
						for i in warmup_end..data.len() {
							prop_assert!(
								(result.values[i] - first_val).abs() < val_tol,
								"Constant data should produce constant output at index {}",
								i
							);
						}
					}
				}
				
				// Property 7: Window with identical values
				for i in warmup_end..data.len() {
					let window = &data[(i + 1 - period)..=i];
					if !window.is_empty() {
						let window_val = window[0];
						let win_tol = tolerance(window_val);
						if window.windows(2).all(|w| (w[0] - w[1]).abs() < win_tol) {
							prop_assert!(
								(result.values[i] - window_val).abs() < win_tol,
								"Window with identical values should produce that value at index {}",
								i
							);
						}
					}
				}
				
				// Property 8: Monotonic data midpoint verification
				if scenario == 2 || scenario == 3 {
					// For monotonic data, midpoint should be exactly between first and last of window
					for i in warmup_end..data.len() {
						let window_start = data[i + 1 - period];
						let window_end = data[i];
						let expected_midpoint = (window_start + window_end) / 2.0;
						let tol = tolerance(expected_midpoint);
						
						prop_assert!(
							(result.values[i] - expected_midpoint).abs() < tol,
							"Monotonic data midpoint mismatch at index {}: expected {}, got {}, tolerance {}",
							i, expected_midpoint, result.values[i], tol
						);
					}
				}
				
				// Property 9: Poison detection
				#[cfg(debug_assertions)]
				{
					for (i, &val) in result.values.iter().enumerate() {
						if val.is_nan() {
							continue;
						}
						
						let bits = val.to_bits();
						prop_assert!(
							bits != 0x11111111_11111111 && 
							bits != 0x22222222_22222222 && 
							bits != 0x33333333_33333333,
							"Found poison value at index {}: {} (0x{:016X})",
							i, val, bits
						);
					}
				}
				
				Ok(())
			})?;
		
		Ok(())
	}

	macro_rules! generate_all_midpoint_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                   #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }
	generate_all_midpoint_tests!(
		check_midpoint_partial_params,
		check_midpoint_accuracy,
		check_midpoint_default_candles,
		check_midpoint_zero_period,
		check_midpoint_period_exceeds_length,
		check_midpoint_very_small_dataset,
		check_midpoint_reinput,
		check_midpoint_nan_handling,
		check_midpoint_streaming,
		check_midpoint_no_poison
	);
	
	#[cfg(feature = "proptest")]
	generate_all_midpoint_tests!(check_midpoint_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = MidpointBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = MidpointParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
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
			(10, 20, 2),     // Dense medium range
			(50, 100, 10),   // Large range with bigger step
			(14, 14, 0),     // Single value (default)
			(3, 7, 1),       // Very dense small range
		];
		
		for (cfg_idx, &(period_start, period_end, period_step)) in test_configs.iter().enumerate() {
			let output = MidpointBatchBuilder::new()
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
				
				// Check all three poison patterns with detailed context
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(14)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(14)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(14)
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
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }
	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}

// --- BATCH INNER INTO ---

#[inline(always)]
pub fn midpoint_batch_inner_into(
	data: &[f64],
	sweep: &MidpointBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<MidpointParams>, MidpointError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MidpointError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(MidpointError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(MidpointError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	if out.len() != rows * cols {
		return Err(MidpointError::InvalidPeriod { period: out.len(), data_len: rows * cols });
	}
	
	// Initialize output with NaN
	out.fill(f64::NAN);
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => midpoint_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => midpoint_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => midpoint_row_avx512(data, first, period, out_row),
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

// --- PYTHON BINDINGS ---

#[cfg(feature = "python")]
#[pyfunction(name = "midpoint")]
#[pyo3(signature = (data, period=None, kernel=None))]
pub fn midpoint_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: Option<usize>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};
	
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = MidpointParams { period };
	let input = MidpointInput::from_slice(slice_in, params);
	
	let result_vec: Vec<f64> = py
		.allow_threads(|| midpoint_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MidpointStream")]
pub struct MidpointStreamPy {
	stream: MidpointStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MidpointStreamPy {
	#[new]
	fn new(period: Option<usize>) -> PyResult<Self> {
		let params = MidpointParams { period };
		let stream = MidpointStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(MidpointStreamPy { stream })
	}
	
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "midpoint_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn midpoint_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = MidpointBatchRange {
		period: period_range,
	};
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();
	
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	
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
			midpoint_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

// --- WASM BINDINGS ---

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midpoint_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = MidpointParams {
		period: Some(period),
	};
	let input = MidpointInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];  // Single allocation
	midpoint_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midpoint_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midpoint_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn midpoint_into(
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
		let params = MidpointParams { period: Some(period) };
		let input = MidpointInput::from_slice(data, params);
		
		if in_ptr == out_ptr {  // CRITICAL: Aliasing check
			let mut temp = vec![0.0; len];
			midpoint_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			midpoint_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MidpointBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MidpointBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<MidpointParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = midpoint_batch)]
pub fn midpoint_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: MidpointBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = MidpointBatchRange {
		period: config.period_range,
	};
	
	let result = midpoint_batch_slice(data, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let output = MidpointBatchJsOutput {
		values: result.values,
		combos: result.combos,
		rows: result.rows,
		cols: result.cols,
	};
	
	serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}
