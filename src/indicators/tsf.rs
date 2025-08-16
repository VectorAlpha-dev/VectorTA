//! # Time Series Forecast (TSF)
//!
//! TSF computes a linear regression forecast over a rolling window, returning the value projected for the next step. Parameters and features match alma.rs, with streaming, batch, and AVX function stubs, builder patterns, and validation. AVX/AVX512 are stubs mapped to scalar, but present for API parity.
//!
//! ## Parameters
//! - **period**: Regression window size (default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: tsf: All input data values are `NaN`.
//! - **InvalidPeriod**: tsf: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: tsf: Not enough valid data points for the requested period.
//!
//! ## Returns
//! - **`Ok(TsfOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(TsfError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
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
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for TsfInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			TsfData::Slice(slice) => slice,
			TsfData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum TsfData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TsfOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TsfParams {
	pub period: Option<usize>,
}

impl Default for TsfParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct TsfInput<'a> {
	pub data: TsfData<'a>,
	pub params: TsfParams,
}

impl<'a> TsfInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: TsfParams) -> Self {
		Self {
			data: TsfData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: TsfParams) -> Self {
		Self {
			data: TsfData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", TsfParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct TsfBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for TsfBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl TsfBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<TsfOutput, TsfError> {
		let p = TsfParams { period: self.period };
		let i = TsfInput::from_candles(c, "close", p);
		tsf_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<TsfOutput, TsfError> {
		let p = TsfParams { period: self.period };
		let i = TsfInput::from_slice(d, p);
		tsf_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<TsfStream, TsfError> {
		let p = TsfParams { period: self.period };
		TsfStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum TsfError {
	#[error("tsf: Input data slice is empty.")]
	EmptyInputData,
	#[error("tsf: All values are NaN.")]
	AllValuesNaN,
	#[error("tsf: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("tsf: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn tsf(input: &TsfInput) -> Result<TsfOutput, TsfError> {
	tsf_with_kernel(input, Kernel::Auto)
}

pub fn tsf_with_kernel(input: &TsfInput, kernel: Kernel) -> Result<TsfOutput, TsfError> {
	let data: &[f64] = match &input.data {
		TsfData::Candles { candles, source } => source_type(candles, source),
		TsfData::Slice(sl) => sl,
	};

	let len = data.len();
	if len == 0 {
		return Err(TsfError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(TsfError::AllValuesNaN)?;
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(TsfError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(TsfError::NotEnoughValidData {
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
			Kernel::Scalar | Kernel::ScalarBatch => tsf_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => tsf_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => tsf_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(TsfOutput { values: out })
}

#[inline]
pub fn tsf_into_slice(dst: &mut [f64], input: &TsfInput, kern: Kernel) -> Result<(), TsfError> {
	let data: &[f64] = match &input.data {
		TsfData::Candles { candles, source } => source_type(candles, source),
		TsfData::Slice(sl) => sl,
	};

	let len = data.len();
	if len == 0 {
		return Err(TsfError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(TsfError::AllValuesNaN)?;
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(TsfError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(TsfError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	if dst.len() != data.len() {
		return Err(TsfError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	match chosen {
		Kernel::Scalar | Kernel::ScalarBatch => tsf_scalar(data, period, first, dst),
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 | Kernel::Avx2Batch => unsafe { tsf_avx2(data, period, first, dst) },
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 | Kernel::Avx512Batch => unsafe { tsf_avx512(data, period, first, dst) },
		#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
		Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => tsf_scalar(data, period, first, dst),
		_ => unreachable!(),
	}

	// Fill warmup with NaN
	let warmup_end = first + period - 1;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	Ok(())
}

#[inline]
pub fn tsf_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	// Precompute ∑ x and ∑ x² for x = 0..period-1
	let sum_x = (0..period).map(|x| x as f64).sum::<f64>();
	let sum_x_sqr = (0..period).map(|x| (x as f64) * (x as f64)).sum::<f64>();
	let divisor = (period as f64 * sum_x_sqr) - (sum_x * sum_x);

	// We only start writing output once we have 'period' non‐NaN points
	// at indices [first_val .. first_val + period - 2], so the first valid index is:
	// i = first_val + period - 1
	for i in (first_val + period - 1)..data.len() {
		let mut sum_xy = 0.0;
		let mut sum_y = 0.0;

		// --- CORRECTION HERE ---
		// j = 0 should correspond to the oldest point in the window,
		// i.e. data[i - (period - 1)].  When j = period - 1, that is data[i].
		for j in 0..period {
			let idx = i - (period - 1) + j;
			let val = data[idx];
			sum_y += val;
			sum_xy += (j as f64) * val;
		}

		let m = ((period as f64) * sum_xy - sum_x * sum_y) / divisor;
		let b = (sum_y - m * sum_x) / (period as f64);
		out[i] = b + m * (period as f64);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tsf_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	unsafe {
		if period <= 32 {
			tsf_avx512_short(data, period, first_valid, out);
		} else {
			tsf_avx512_long(data, period, first_valid, out);
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tsf_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	unsafe { tsf_scalar(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsf_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	tsf_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsf_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	tsf_scalar(data, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct TsfStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	sum_x: f64,
	sum_x_sqr: f64,
	divisor: f64,
}

impl TsfStream {
	pub fn try_new(params: TsfParams) -> Result<Self, TsfError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(TsfError::InvalidPeriod { period, data_len: 0 });
		}

		// Precompute ∑ x and ∑ x² for x = 0..period-1
		let sum_x = (0..period).map(|x| x as f64).sum::<f64>();
		let sum_x_sqr = (0..period).map(|x| (x as f64) * (x as f64)).sum::<f64>();
		let divisor = (period as f64 * sum_x_sqr) - (sum_x * sum_x);

		Ok(Self {
			period,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
			sum_x,
			sum_x_sqr,
			divisor,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		// Write the newest value at buffer[head], then advance head.
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;

		// Once head wraps to 0, we know the ring has filled at least once.
		if !self.filled && self.head == 0 {
			self.filled = true;
		}

		// Until we've filled 'period' values, we return None.
		if !self.filled {
			return None;
		}

		// Once filled, compute the regression forecast via dot_ring()
		Some(self.dot_ring())
	}

	#[inline(always)]
	fn dot_ring(&self) -> f64 {
		// This loop already uses the correct chronological order:
		//   j = 0 → oldest (at index = head)
		//   j = period-1 → newest (at index = head + period - 1 mod period)
		let mut sum_xy = 0.0;
		let mut sum_y = 0.0;
		let mut idx = self.head; // head always points at the oldest element

		for j in 0..self.period {
			let val = self.buffer[idx];
			sum_y += val;
			sum_xy += (j as f64) * val;
			idx = (idx + 1) % self.period;
		}

		let m = ((self.period as f64) * sum_xy - self.sum_x * sum_y) / self.divisor;
		let b = (sum_y - m * self.sum_x) / (self.period as f64);
		b + m * (self.period as f64)
	}
}

#[derive(Clone, Debug)]
pub struct TsfBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for TsfBatchRange {
	fn default() -> Self {
		Self { period: (14, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct TsfBatchBuilder {
	range: TsfBatchRange,
	kernel: Kernel,
}

impl TsfBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<TsfBatchOutput, TsfError> {
		tsf_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TsfBatchOutput, TsfError> {
		TsfBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TsfBatchOutput, TsfError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<TsfBatchOutput, TsfError> {
		TsfBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn tsf_batch_with_kernel(data: &[f64], sweep: &TsfBatchRange, k: Kernel) -> Result<TsfBatchOutput, TsfError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(TsfError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	tsf_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TsfBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TsfParams>,
	pub rows: usize,
	pub cols: usize,
}
impl TsfBatchOutput {
	pub fn row_for_params(&self, p: &TsfParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}

	pub fn values_for(&self, p: &TsfParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &TsfBatchRange) -> Vec<TsfParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(TsfParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn tsf_batch_slice(data: &[f64], sweep: &TsfBatchRange, kern: Kernel) -> Result<TsfBatchOutput, TsfError> {
	tsf_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tsf_batch_par_slice(data: &[f64], sweep: &TsfBatchRange, kern: Kernel) -> Result<TsfBatchOutput, TsfError> {
	tsf_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn tsf_batch_inner(
	data: &[f64],
	sweep: &TsfBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<TsfBatchOutput, TsfError> {
	// Build the list of TsfParams to run over
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TsfError::InvalidPeriod { period: 0, data_len: 0 });
	}

	// Find first non‐NaN index
	let first = data.iter().position(|x| !x.is_nan()).ok_or(TsfError::AllValuesNaN)?;
	// Compute the maximum period required by any combo
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(TsfError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	let mut sum_xs = vec![0.0; rows];
	let mut sum_x_sq = vec![0.0; rows];
	let mut divisors = vec![0.0; rows];

	// Precompute ∑x and (∑x²) and divisor for each "period = combos[row].period"
	for (row, prm) in combos.iter().enumerate() {
		let period = prm.period.unwrap();
		let sum_x = (0..period).map(|x| x as f64).sum::<f64>();
		let sum_x2 = (0..period).map(|x| (x as f64) * (x as f64)).sum::<f64>();
		let divisor = (period as f64 * sum_x2) - (sum_x * sum_x);
		sum_xs[row] = sum_x;
		sum_x_sq[row] = sum_x2;
		divisors[row] = divisor;
	}

	// Use uninitialized memory like ALMA
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Compute warmup periods for each row
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Use ManuallyDrop pattern for safe conversion
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	// Closure that computes one row into out_row
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let sum_x = sum_xs[row];
		let divisor = divisors[row];

		match kern {
			Kernel::Scalar => tsf_row_scalar(data, first, period, sum_x, divisor, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => tsf_row_avx2(data, first, period, sum_x, divisor, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => tsf_row_avx512(data, first, period, sum_x, divisor, out_row),
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

	// Convert back to Vec using ManuallyDrop pattern
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(TsfBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

// Helper function for Python batch bindings - writes directly to the provided buffer
#[inline(always)]
fn tsf_batch_inner_into(
	data: &[f64],
	sweep: &TsfBatchRange,
	kern: Kernel,
	parallel: bool,
	output: &mut [f64],
) -> Result<Vec<TsfParams>, TsfError> {
	// Build the list of TsfParams to run over
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TsfError::InvalidPeriod { period: 0, data_len: 0 });
	}

	// Find first non‐NaN index
	let first = data.iter().position(|x| !x.is_nan()).ok_or(TsfError::AllValuesNaN)?;
	// Compute the maximum period required by any combo
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(TsfError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	let mut sum_xs = vec![0.0; rows];
	let mut sum_x_sq = vec![0.0; rows];
	let mut divisors = vec![0.0; rows];

	// Precompute ∑x and (∑x²) and divisor for each "period = combos[row].period"
	for (row, prm) in combos.iter().enumerate() {
		let period = prm.period.unwrap();
		let sum_x = (0..period).map(|x| x as f64).sum::<f64>();
		let sum_x2 = (0..period).map(|x| (x as f64) * (x as f64)).sum::<f64>();
		let divisor = (period as f64 * sum_x2) - (sum_x * sum_x);
		sum_xs[row] = sum_x;
		sum_x_sq[row] = sum_x2;
		divisors[row] = divisor;
	}

	// Initialize output with NaN
	output.fill(f64::NAN);

	// Closure that computes one row into out_row
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let sum_x = sum_xs[row];
		let divisor = divisors[row];

		match kern {
			Kernel::Scalar => tsf_row_scalar(data, first, period, sum_x, divisor, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => tsf_row_avx2(data, first, period, sum_x, divisor, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => tsf_row_avx512(data, first, period, sum_x, divisor, out_row),
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			output
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in output.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in output.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	Ok(combos)
}

// 2) tsf_row_scalar (fixed indexing so j=0→oldest, j=period−1→newest)
#[inline(always)]
unsafe fn tsf_row_scalar(data: &[f64], first: usize, period: usize, sum_x: f64, divisor: f64, out: &mut [f64]) {
	// Loop i from (first + period − 1) .. end.  At i we compute a regression over
	//   indices [i−(period−1) .. i], with x = 0 at data[i−(period−1)], x = period−1 at data[i].
	for i in (first + period - 1)..data.len() {
		let mut sum_xy = 0.0;
		let mut sum_y = 0.0;

		// Correct order: “j = 0” hits data[i − (period − 1)] (oldest),
		// “j = period − 1” hits data[i] (most recent).
		for j in 0..period {
			let idx = i - (period - 1) + j;
			let val = data[idx];
			sum_y += val;
			sum_xy += (j as f64) * val;
		}

		let m = ((period as f64) * sum_xy - sum_x * sum_y) / divisor;
		let b = (sum_y - m * sum_x) / (period as f64);
		out[i] = b + m * (period as f64);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx2(data: &[f64], first: usize, period: usize, sum_x: f64, divisor: f64, out: &mut [f64]) {
	tsf_row_scalar(data, first, period, sum_x, divisor, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx512(data: &[f64], first: usize, period: usize, sum_x: f64, divisor: f64, out: &mut [f64]) {
	if period <= 32 {
		tsf_row_avx512_short(data, first, period, sum_x, divisor, out);
	} else {
		tsf_row_avx512_long(data, first, period, sum_x, divisor, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx512_short(data: &[f64], first: usize, period: usize, sum_x: f64, divisor: f64, out: &mut [f64]) {
	tsf_row_scalar(data, first, period, sum_x, divisor, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx512_long(data: &[f64], first: usize, period: usize, sum_x: f64, divisor: f64, out: &mut [f64]) {
	tsf_row_scalar(data, first, period, sum_x, divisor, out)
}

#[inline(always)]
pub fn expand_grid_tsf(r: &TsfBatchRange) -> Vec<TsfParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(TsfParams { period: Some(p) });
	}
	out
}

// ================================
// Python Bindings
// ================================

#[cfg(feature = "python")]
#[pyfunction(name = "tsf")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn tsf_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = TsfParams {
		period: Some(period),
	};
	let tsf_in = TsfInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| tsf_with_kernel(&tsf_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TsfStream")]
pub struct TsfStreamPy {
	stream: TsfStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TsfStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = TsfParams {
			period: Some(period),
		};
		let stream = TsfStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(TsfStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "tsf_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn tsf_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = TsfBatchRange {
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
			tsf_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos.iter().map(|p| p.period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py)
	)?;

	Ok(dict)
}

// ================================
// WASM Bindings
// ================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = TsfParams {
		period: Some(period),
	};
	let input = TsfInput::from_slice(data, params);

	let mut output = vec![0.0; data.len()];

	tsf_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_into(
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
		let params = TsfParams {
			period: Some(period),
		};
		let input = TsfInput::from_slice(data, params);

		if in_ptr == out_ptr {
			// Handle aliasing case
			let mut temp = vec![0.0; len];
			tsf_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			tsf_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TsfBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TsfBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TsfParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tsf_batch)]
pub fn tsf_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: TsfBatchConfig =
		serde_wasm_bindgen::from_value(config)
			.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = TsfBatchRange {
		period: config.period_range,
	};

	let output = tsf_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = TsfBatchJsOutput {
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
pub fn tsf_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to tsf_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = TsfBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		if rows == 0 {
			return Err(JsValue::from_str("No valid parameter combinations"));
		}

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		let kernel = detect_best_batch_kernel();
		let simd = match kernel {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			_ => unreachable!(),
		};

		tsf_batch_inner_into(data, &sweep, simd, true, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_tsf_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = TsfParams { period: None };
		let input = TsfInput::from_candles(&candles, "close", default_params);
		let output = tsf_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_tsf_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = TsfInput::from_candles(&candles, "close", TsfParams::default());
		let result = tsf_with_kernel(&input, kernel)?;
		let expected_last_five = [
			58846.945054945056,
			58818.83516483516,
			58854.57142857143,
			59083.846153846156,
			58962.25274725275,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] TSF {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_tsf_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = TsfInput::with_default_candles(&candles);
		match input.data {
			TsfData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected TsfData::Candles"),
		}
		let output = tsf_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_tsf_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = TsfParams { period: Some(0) };
		let input = TsfInput::from_slice(&input_data, params);
		let res = tsf_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TSF should fail with zero period", test_name);
		Ok(())
	}

	fn check_tsf_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = TsfParams { period: Some(10) };
		let input = TsfInput::from_slice(&data_small, params);
		let res = tsf_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] TSF should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_tsf_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = TsfParams { period: Some(9) };
		let input = TsfInput::from_slice(&single_point, params);
		let res = tsf_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TSF should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_tsf_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = TsfParams { period: Some(14) };
		let first_input = TsfInput::from_candles(&candles, "close", first_params);
		let first_result = tsf_with_kernel(&first_input, kernel)?;
		let second_params = TsfParams { period: Some(14) };
		let second_input = TsfInput::from_slice(&first_result.values, second_params);
		let second_result = tsf_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_tsf_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = TsfInput::from_candles(&candles, "close", TsfParams { period: Some(14) });
		let res = tsf_with_kernel(&input, kernel)?;
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

	fn check_tsf_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 14;
		let input = TsfInput::from_candles(&candles, "close", TsfParams { period: Some(period) });
		let batch_output = tsf_with_kernel(&input, kernel)?.values;

		let mut stream = TsfStream::try_new(TsfParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(tsf_val) => stream_values.push(tsf_val),
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
				"[{}] TSF streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_tsf_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			TsfParams::default(), // period: 14
			TsfParams { period: Some(1) },
			TsfParams { period: Some(2) },
			TsfParams { period: Some(5) },
			TsfParams { period: Some(7) },
			TsfParams { period: Some(10) },
			TsfParams { period: Some(20) },
			TsfParams { period: Some(30) },
			TsfParams { period: Some(50) },
			TsfParams { period: Some(100) },
			TsfParams { period: Some(200) },
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = TsfInput::from_candles(&candles, "close", params.clone());
			let output = tsf_with_kernel(&input, kernel)?;

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
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
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
						params.period.unwrap_or(14),
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
						params.period.unwrap_or(14),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_tsf_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_tsf_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate test strategies with various period values and data patterns
		let strat = (1usize..=64)
			.prop_flat_map(|period| {
				(
					// Generate data vectors with various patterns
					prop::collection::vec(
						(-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						period.max(10)..400,
					),
					Just(period),
					// Trend factor for generating linear trends
					-100.0f64..100.0f64,
					// Intercept for linear data
					-1e5f64..1e5f64,
					// Flag to test special patterns
					prop::bool::ANY,
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(mut data, period, trend, intercept, use_special_pattern)| {
				// Optionally apply special patterns for testing
				if use_special_pattern {
					let pattern_choice = data.len() % 3;
					match pattern_choice {
						0 => {
							// Create perfectly linear data
							for (i, val) in data.iter_mut().enumerate() {
								*val = trend * i as f64 + intercept;
							}
						}
						1 => {
							// Create constant data
							let constant = data.first().copied().unwrap_or(42.0);
							data.fill(constant);
						}
						_ => {
							// Keep random data but add a trend
							for (i, val) in data.iter_mut().enumerate() {
								*val += trend * i as f64;
							}
						}
					}
				}

				let params = TsfParams {
					period: Some(period),
				};
				let input = TsfInput::from_slice(&data, params);

				// Run TSF with specified kernel and scalar reference
				let TsfOutput { values: out } = tsf_with_kernel(&input, kernel).unwrap();
				let TsfOutput { values: ref_out } = tsf_with_kernel(&input, Kernel::Scalar).unwrap();

				// Verify warmup period
				for i in 0..(period - 1) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Test properties after warmup
				for i in (period - 1)..data.len() {
					let y = out[i];
					let r = ref_out[i];

					// Property 1: Kernel consistency
					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"finite/NaN mismatch idx {}: {} vs {}",
							i, y, r
						);
						continue;
					}

					let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"Kernel mismatch idx {}: {} vs {} (ULP={})",
						i, y, r, ulp_diff
					);

					// Property 2: Special cases
					if period == 1 {
						// Period=1 should return the current value
						prop_assert!(
							(y - data[i]).abs() <= f64::EPSILON,
							"Period=1 should return current value at idx {}: {} vs {}",
							i, y, data[i]
						);
					}

					// Property 3: Constant data
					if i >= period && data[i - period + 1..=i].windows(2).all(|w| (w[0] - w[1]).abs() <= 1e-10) {
						let expected = data[i];
						prop_assert!(
							(y - expected).abs() <= 1e-6,
							"Constant data should forecast constant at idx {}: {} vs {}",
							i, y, expected
						);
					}

					// Property 4: Linear trend verification
					// For perfectly linear data, the forecast should be exact
					if i >= period + 1 {
						let window = &data[i - period + 1..=i];
						let is_linear = window.windows(2).enumerate().all(|(j, w)| {
							if j == 0 { return true; }
							let diff1 = w[1] - w[0];
							let diff0 = window[1] - window[0];
							(diff1 - diff0).abs() <= 1e-10
						});

						if is_linear && window.len() >= 2 {
							// For linear data, forecast should continue the trend
							let slope = (window[window.len() - 1] - window[0]) / (window.len() - 1) as f64;
							let expected = window[window.len() - 1] + slope;
							prop_assert!(
								(y - expected).abs() <= 1e-6,
								"Linear forecast mismatch at idx {}: {} vs {} (slope={})",
								i, y, expected, slope
							);
						}
					}

					// Property 5: Finite output for finite input
					prop_assert!(
						y.is_finite(),
						"Output should be finite at idx {}: {}",
						i, y
					);
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_tsf_tests {
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

	generate_all_tsf_tests!(
		check_tsf_partial_params,
		check_tsf_accuracy,
		check_tsf_default_candles,
		check_tsf_zero_period,
		check_tsf_period_exceeds_length,
		check_tsf_very_small_dataset,
		check_tsf_reinput,
		check_tsf_nan_handling,
		check_tsf_streaming,
		check_tsf_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_tsf_tests!(check_tsf_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = TsfBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = TsfParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			58846.945054945056,
			58818.83516483516,
			58854.57142857143,
			59083.846153846156,
			58962.25274725275,
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

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step)
			(2, 10, 2),       // Small periods
			(5, 25, 5),       // Medium periods
			(20, 50, 10),     // Large periods
			(2, 5, 1),        // Dense small range
			(14, 14, 0),      // Single value (default period)
			(30, 60, 15),     // Mixed range
			(50, 100, 25),    // Very large periods
			(100, 200, 50),   // Extra large periods
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = TsfBatchBuilder::new()
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
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
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
						combo.period.unwrap_or(14)
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
	gen_batch_tests!(check_batch_no_poison);
}
