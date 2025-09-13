//! # Normalized Average True Range (NATR)
//!
//! Normalizes ATR by closing price as a percentage for cross-asset volatility comparison.
//!
//! ## Parameters
//! - **high**: High price data
//! - **low**: Low price data
//! - **close**: Close price data
//! - **period**: ATR calculation period (default: 14)
//!
//! ## Returns
//! - `Vec<f64>` - NATR values (percentage) matching input length
//!
//! ## Developer Status
//! **AVX2**: Stub (calls scalar)
//! **AVX512**: Has short/long variants but all stubs
//! **Streaming**: O(1) - Uses exponential smoothing
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`

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
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub enum NatrData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct NatrOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct NatrParams {
	pub period: Option<usize>,
}

impl Default for NatrParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct NatrInput<'a> {
	pub data: NatrData<'a>,
	pub params: NatrParams,
}

impl<'a> NatrInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: NatrParams) -> Self {
		Self {
			data: NatrData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: NatrParams) -> Self {
		Self {
			data: NatrData::Slices { high, low, close },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: NatrData::Candles { candles },
			params: NatrParams::default(),
		}
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct NatrBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for NatrBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl NatrBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<NatrOutput, NatrError> {
		let p = NatrParams { period: self.period };
		let i = NatrInput::from_candles(c, p);
		natr_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<NatrOutput, NatrError> {
		let p = NatrParams { period: self.period };
		let i = NatrInput::from_slices(high, low, close, p);
		natr_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<NatrStream, NatrError> {
		let p = NatrParams { period: self.period };
		NatrStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum NatrError {
	#[error("natr: Empty data provided for NATR.")]
	EmptyData,
	#[error("natr: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("natr: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("natr: All values are NaN.")]
	AllValuesNaN,
	#[error("natr: Mismatched lengths: expected = {expected}, actual = {actual}")]
	MismatchedLength { expected: usize, actual: usize },
}

#[inline]
pub fn natr(input: &NatrInput) -> Result<NatrOutput, NatrError> {
	natr_with_kernel(input, Kernel::Auto)
}

pub fn natr_with_kernel(input: &NatrInput, kernel: Kernel) -> Result<NatrOutput, NatrError> {
	let (high, low, close) = match &input.data {
		NatrData::Candles { candles } => {
			let high = source_type(candles, "high");
			let low = source_type(candles, "low");
			let close = source_type(candles, "close");
			(high, low, close)
		}
		NatrData::Slices { high, low, close } => (*high, *low, *close),
	};

	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(NatrError::EmptyData);
	}

	let len_h = high.len();
	let len_l = low.len();
	let len_c = close.len();
	if len_h != len_l || len_h != len_c {
		return Err(NatrError::MismatchedLength {
			expected: len_h,
			actual: if len_l != len_h { len_l } else { len_c },
		});
	}
	let len = len_h;

	let period = input.get_period();
	if period == 0 || period > len {
		return Err(NatrError::InvalidPeriod { period, data_len: len });
	}

	let first_valid_idx = {
		let first_valid_idx_h = high.iter().position(|&x| !x.is_nan());
		let first_valid_idx_l = low.iter().position(|&x| !x.is_nan());
		let first_valid_idx_c = close.iter().position(|&x| !x.is_nan());

		match (first_valid_idx_h, first_valid_idx_l, first_valid_idx_c) {
			(Some(h), Some(l), Some(c)) => Some(h.max(l).max(c)),
			_ => None,
		}
	};

	let first_valid_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(NatrError::AllValuesNaN),
	};

	if (len - first_valid_idx) < period {
		return Err(NatrError::NotEnoughValidData {
			needed: period,
			valid: len - first_valid_idx,
		});
	}

	let mut out = alloc_with_nan_prefix(len, first_valid_idx + period - 1);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => natr_scalar(high, low, close, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => natr_avx2(high, low, close, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => natr_avx512(high, low, close, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(NatrOutput { values: out })
}

#[inline]
pub fn natr_scalar(high: &[f64], low: &[f64], close: &[f64], period: usize, first: usize, out: &mut [f64]) {
	let mut sum_tr = 0.0;
	let mut prev_atr = 0.0;
	let mut count_since_first = 0usize;

	for i in first..out.len() {
		let tr = if i == first {
			high[i] - low[i]
		} else {
			let tr_curr = high[i] - low[i];
			let tr_prev_close_high = (high[i] - close[i - 1]).abs();
			let tr_prev_close_low = (low[i] - close[i - 1]).abs();
			tr_curr.max(tr_prev_close_high).max(tr_prev_close_low)
		};

		if count_since_first < period {
			sum_tr += tr;
			if count_since_first == period - 1 {
				prev_atr = sum_tr / (period as f64);
				let c = close[i];
				if c.is_finite() && c != 0.0 {
					out[i] = (prev_atr / c) * 100.0;
				} else {
					out[i] = f64::NAN;
				}
			}
		} else {
			let new_atr = ((prev_atr * ((period - 1) as f64)) + tr) / (period as f64);
			prev_atr = new_atr;

			let c = close[i];
			if c.is_finite() && c != 0.0 {
				out[i] = (new_atr / c) * 100.0;
			} else {
				out[i] = f64::NAN;
			}
		}

		count_since_first += 1;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn natr_avx512(high: &[f64], low: &[f64], close: &[f64], period: usize, first: usize, out: &mut [f64]) {
	unsafe {
		if period <= 32 {
			natr_avx512_short(high, low, close, period, first, out);
		} else {
			natr_avx512_long(high, low, close, period, first, out);
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn natr_avx2(high: &[f64], low: &[f64], close: &[f64], period: usize, first: usize, out: &mut [f64]) {
	natr_scalar(high, low, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn natr_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	out: &mut [f64],
) {
	natr_scalar(high, low, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn natr_avx512_long(high: &[f64], low: &[f64], close: &[f64], period: usize, first: usize, out: &mut [f64]) {
	natr_scalar(high, low, close, period, first, out);
}

#[derive(Debug, Clone)]
pub struct NatrStream {
	period: usize,
	tr_buffer: Vec<f64>,
	close_buffer: Vec<f64>,
	sum_tr: f64,
	prev_atr: f64,
	head: usize,
	filled: bool,
	count: usize,
}

impl NatrStream {
	pub fn try_new(params: NatrParams) -> Result<Self, NatrError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(NatrError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			// initialize buffers with zeros to avoid propagating NaNs during
			// the warm-up phase
			tr_buffer: vec![0.0; period],
			close_buffer: vec![0.0; period],
			sum_tr: 0.0,
			prev_atr: 0.0,
			head: 0,
			filled: false,
			count: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		let tr = if self.count == 0 {
			high - low
		} else {
			let tr_curr = high - low;
			let tr_prev_close_high = (high - self.close_buffer[(self.head + self.period - 1) % self.period]).abs();
			let tr_prev_close_low = (low - self.close_buffer[(self.head + self.period - 1) % self.period]).abs();
			tr_curr.max(tr_prev_close_high).max(tr_prev_close_low)
		};

		self.sum_tr += tr - self.tr_buffer[self.head];
		self.tr_buffer[self.head] = tr;
		self.close_buffer[self.head] = close;
		self.head = (self.head + 1) % self.period;

		self.count += 1;

		if !self.filled {
			if self.count == self.period {
				self.prev_atr = self.sum_tr / (self.period as f64);
				self.filled = true;
				if close.is_finite() && close != 0.0 {
					return Some((self.prev_atr / close) * 100.0);
				} else {
					return Some(f64::NAN);
				}
			}
			return None;
		} else {
			let new_atr = ((self.prev_atr * ((self.period - 1) as f64)) + tr) / (self.period as f64);
			self.prev_atr = new_atr;
			if close.is_finite() && close != 0.0 {
				return Some((new_atr / close) * 100.0);
			} else {
				return Some(f64::NAN);
			}
		}
	}
}

#[derive(Clone, Debug)]
pub struct NatrBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for NatrBatchRange {
	fn default() -> Self {
		Self { period: (14, 30, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct NatrBatchBuilder {
	range: NatrBatchRange,
	kernel: Kernel,
}

impl NatrBatchBuilder {
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
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<NatrBatchOutput, NatrError> {
		natr_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<NatrBatchOutput, NatrError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		self.apply_slices(high, low, close)
	}
	pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<NatrBatchOutput, NatrError> {
		NatrBatchBuilder::new().kernel(k).apply_candles(c)
	}
}

pub fn natr_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &NatrBatchRange,
	k: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(NatrError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	natr_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct NatrBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<NatrParams>,
	pub rows: usize,
	pub cols: usize,
}
impl NatrBatchOutput {
	pub fn row_for_params(&self, p: &NatrParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &NatrParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &NatrBatchRange) -> Vec<NatrParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(NatrParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn natr_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &NatrBatchRange,
	kern: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
	natr_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn natr_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &NatrBatchRange,
	kern: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
	natr_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn natr_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &NatrBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<NatrBatchOutput, NatrError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(NatrError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let len_h = high.len();
	let len_l = low.len();
	let len_c = close.len();
	if len_h != len_l || len_h != len_c {
		return Err(NatrError::MismatchedLength {
			expected: len_h,
			actual: if len_l != len_h { len_l } else { len_c },
		});
	}
	let len = len_h;
	
	let first = high
		.iter()
		.position(|x| !x.is_nan())
		.unwrap_or(0)
		.max(low.iter().position(|x| !x.is_nan()).unwrap_or(0))
		.max(close.iter().position(|x| !x.is_nan()).unwrap_or(0));
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(NatrError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	
	let mut buf_mu = make_uninit_matrix(rows, cols);
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
			Kernel::Scalar => natr_row_scalar(high, low, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => natr_row_avx2(high, low, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => natr_row_avx512(high, low, close, first, period, out_row),
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
	
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(NatrBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn natr_batch_inner_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &NatrBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<NatrParams>, NatrError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(NatrError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let len_h = high.len();
	let len_l = low.len();
	let len_c = close.len();
	if len_h != len_l || len_h != len_c {
		return Err(NatrError::MismatchedLength {
			expected: len_h,
			actual: if len_l != len_h { len_l } else { len_c },
		});
	}
	let len = len_h;
	
	let first = high
		.iter()
		.position(|x| !x.is_nan())
		.unwrap_or(0)
		.max(low.iter().position(|x| !x.is_nan()).unwrap_or(0))
		.max(close.iter().position(|x| !x.is_nan()).unwrap_or(0));
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(NatrError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;

	// Initialize warmup positions with NaN for each row
	for (row, combo) in combos.iter().enumerate() {
		let period = combo.period.unwrap();
		let warmup_end = first + period - 1;
		let row_start = row * cols;
		for i in 0..warmup_end.min(cols) {
			out[row_start + i] = f64::NAN;
		}
	}

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => natr_row_scalar(high, low, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => natr_row_avx2(high, low, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => natr_row_avx512(high, low, close, first, period, out_row),
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
unsafe fn natr_row_scalar(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	natr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn natr_row_avx2(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	natr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		natr_row_avx512_short(high, low, close, first, period, out);
	} else {
		natr_row_avx512_long(high, low, close, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	out: &mut [f64],
) {
	natr_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	out: &mut [f64],
) {
	natr_scalar(high, low, close, period, first, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	#[cfg(feature = "proptest")]
	use proptest::prelude::*;

	fn check_natr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = NatrParams { period: None };
		let input_default = NatrInput::from_candles(&candles, default_params);
		let output_default = natr_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());
		let params_period_7 = NatrParams { period: Some(7) };
		let input_period_7 = NatrInput::from_candles(&candles, params_period_7);
		let output_period_7 = natr_with_kernel(&input_period_7, kernel)?;
		assert_eq!(output_period_7.values.len(), candles.close.len());
		Ok(())
	}

	fn check_natr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close_prices = candles.select_candle_field("close").unwrap();
		let params = NatrParams { period: Some(14) };
		let input = NatrInput::from_candles(&candles, params.clone());
		let natr_result = natr_with_kernel(&input, kernel)?;
		assert_eq!(natr_result.values.len(), close_prices.len());
		let expected_last_five = [
			1.5465877404905772,
			1.4773840355794576,
			1.4201627494720954,
			1.3556212509014807,
			1.3836271128536142,
		];
		let start_index = natr_result.values.len() - 5;
		let result_last_five = &natr_result.values[start_index..];
		for (i, &value) in result_last_five.iter().enumerate() {
			let expected_value = expected_last_five[i];
			assert!(
				(value - expected_value).abs() < 1e-8,
				"NATR mismatch at index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		let period = params.period.unwrap();
		for i in 0..(period - 1) {
			assert!(natr_result.values[i].is_nan());
		}
		Ok(())
	}

	fn check_natr_with_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 10.0, 15.0];
		let close = [7.0, 14.0, 25.0];
		let params = NatrParams { period: Some(0) };
		let input = NatrInput::from_slices(&high, &low, &close, params);
		let result = natr_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_natr_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 10.0, 15.0];
		let close = [7.0, 14.0, 25.0];
		let params = NatrParams { period: Some(10) };
		let input = NatrInput::from_slices(&high, &low, &close, params);
		let result = natr_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_natr_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [42.0];
		let low = [40.0];
		let close = [41.0];
		let params = NatrParams { period: Some(14) };
		let input = NatrInput::from_slices(&high, &low, &close, params);
		let result = natr_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_natr_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, f64::NAN];
		let low = [f64::NAN, f64::NAN];
		let close = [f64::NAN, f64::NAN];
		let params = NatrParams { period: Some(2) };
		let input = NatrInput::from_slices(&high, &low, &close, params);
		let result = natr_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_natr_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, 10.0];
		let low = [f64::NAN, 5.0];
		let close = [f64::NAN, 7.0];
		let params = NatrParams { period: Some(5) };
		let input = NatrInput::from_slices(&high, &low, &close, params);
		let result = natr_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_natr_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = NatrParams { period: Some(14) };
		let first_input = NatrInput::from_candles(&candles, first_params);
		let first_result = natr_with_kernel(&first_input, kernel)?;
		assert_eq!(first_result.values.len(), candles.close.len());

		let second_params = NatrParams { period: Some(14) };
		let second_input = NatrInput::from_slices(
			&first_result.values,
			&first_result.values,
			&first_result.values,
			second_params,
		);
		let second_result = natr_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());

		for i in 28..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"Expected no NaN after index 28, but found NaN at index {}",
				i
			);
		}
		Ok(())
	}

	fn check_natr_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = NatrParams { period: Some(14) };
		let input = NatrInput::from_candles(&candles, params);
		let natr_result = natr_with_kernel(&input, kernel)?;
		assert_eq!(natr_result.values.len(), candles.close.len());
		if natr_result.values.len() > 30 {
			for i in 30..natr_result.values.len() {
				assert!(
					!natr_result.values[i].is_nan(),
					"Expected no NaN after index 30, but found NaN at index {}",
					i
				);
			}
		}
		Ok(())
	}

	fn check_natr_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = NatrInput::with_default_candles(&candles);
		match input.data {
			NatrData::Candles { .. } => {}
			_ => panic!("Expected NatrData::Candles variant"),
		}
		let output = natr_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_natr_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 14;
		let high = &candles.high;
		let low = &candles.low;
		let close = &candles.close;
		let input = NatrInput::from_slices(high, low, close, NatrParams { period: Some(period) });
		let batch_output = natr_with_kernel(&input, kernel)?.values;

		let mut stream = NatrStream::try_new(NatrParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(close.len());
		for ((&h, &l), &c) in high.iter().zip(low.iter()).zip(close.iter()) {
			match stream.update(h, l, c) {
				Some(natr_val) => stream_values.push(natr_val),
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
				"[{}] NATR streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_natr_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			NatrParams::default(),                    // period: 14
			NatrParams { period: Some(2) },          // minimum viable period
			NatrParams { period: Some(5) },          // small period
			NatrParams { period: Some(7) },          // small period
			NatrParams { period: Some(10) },         // small-medium period
			NatrParams { period: Some(20) },         // medium period
			NatrParams { period: Some(30) },         // medium-large period
			NatrParams { period: Some(50) },         // large period
			NatrParams { period: Some(100) },        // very large period
			NatrParams { period: Some(200) },        // extra large period
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = NatrInput::from_candles(&candles, params.clone());
			let output = natr_with_kernel(&input, kernel)?;

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
	fn check_natr_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = NatrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = NatrParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods
			(30, 60, 15),    // Large periods
			(2, 5, 1),       // Dense small range
			(10, 20, 2),     // Mid-range with small step
			(50, 100, 10),   // Large range
			(14, 14, 0),     // Single period (default)
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = NatrBatchBuilder::new()
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
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	fn check_natr_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		// Strategy: generate period and length, then create realistic OHLC data
		let strat = (2usize..=50, 50usize..=400, 0usize..=2)
			.prop_flat_map(|(period, len, scenario)| {
				// Generate base close prices with different scenarios
				let close_strategy = match scenario {
					0 => {
						// Normal range prices
						prop::collection::vec(
							(1.0f64..1000.0f64).prop_filter("finite", |x| x.is_finite()),
							len,
						).boxed()
					},
					1 => {
						// Very small prices (edge case)
						prop::collection::vec(
							(0.01f64..1.0f64).prop_filter("finite", |x| x.is_finite()),
							len,
						).boxed()
					},
					_ => {
						// Constant prices
						(1.0f64..100.0f64).prop_map(move |val| vec![val; len]).boxed()
					}
				};
				
				(close_strategy, Just(period), Just(len), Just(scenario))
			})
			.prop_flat_map(|(close_prices, period, len, scenario)| {
				// Generate high/low based on close with realistic spreads
				let mut high_vec = Vec::with_capacity(len);
				let mut low_vec = Vec::with_capacity(len);
				
				// Use a deterministic approach for generating high/low
				for (i, &close) in close_prices.iter().enumerate() {
					if scenario == 2 {
						// Constant prices scenario
						high_vec.push(close);
						low_vec.push(close);
					} else {
						// Create volatility that varies over time (expanded range: 0.1% to 20%)
						let volatility_factor = 0.001 + 0.20 * ((i * 7919) % 100) as f64 / 100.0;
						let spread = close * volatility_factor;
						
						// Ensure high >= close >= low
						let high = close + spread * 0.5;
						let low = close - spread * 0.5;
						
						high_vec.push(high);
						low_vec.push(low.max(0.001)); // Ensure positive but allow very small values
					}
				}
				
				(Just(high_vec), Just(low_vec), Just(close_prices), Just(period), Just(scenario))
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(high, low, close, period, scenario)| {
				let params = NatrParams { period: Some(period) };
				let input = NatrInput::from_slices(&high, &low, &close, params);

				// Test with specified kernel
				let result = natr_with_kernel(&input, kernel)?;
				
				// Test with scalar kernel as reference
				let ref_result = natr_with_kernel(&input, Kernel::Scalar)?;

				// Property 1: Output length matches input length
				prop_assert_eq!(result.values.len(), high.len());
				prop_assert_eq!(result.values.len(), low.len());
				prop_assert_eq!(result.values.len(), close.len());

				// Property 2: Warmup period handling - first (period - 1) values should be NaN
				for i in 0..(period - 1) {
					prop_assert!(
						result.values[i].is_nan(),
						"Expected NaN at index {} during warmup, got {}",
						i,
						result.values[i]
					);
				}

				// Property 3: NATR values should be non-negative after warmup (TR is always positive)
				for i in period..result.values.len() {
					if result.values[i].is_finite() {
						prop_assert!(
							result.values[i] >= 0.0,
							"NATR should be non-negative at index {}: got {}",
							i,
							result.values[i]
						);
						
						// NATR values should be reasonable (with expanded volatility, could be higher)
						prop_assert!(
							result.values[i] < 10000.0,
							"NATR seems unreasonably high at index {}: got {}",
							i,
							result.values[i]
						);
					}
				}

				// Property 4: Kernel consistency - different SIMD implementations should produce same results
				for i in 0..result.values.len() {
					let val = result.values[i];
					let ref_val = ref_result.values[i];
					
					if val.is_nan() && ref_val.is_nan() {
						continue;
					}
					
					if val.is_finite() && ref_val.is_finite() {
						// Allow small tolerance for floating point differences
						let diff = (val - ref_val).abs();
						let tolerance = (ref_val.abs() * 1e-10).max(1e-10);
						prop_assert!(
							diff <= tolerance,
							"Kernel mismatch at index {}: {} vs {} (diff: {})",
							i,
							val,
							ref_val,
							diff
						);
					} else {
						// Both should have same finite/infinite status
						prop_assert_eq!(
							val.is_finite(),
							ref_val.is_finite(),
							"Finite status mismatch at index {}: {} vs {}",
							i,
							val,
							ref_val
						);
					}
				}

				// Property 5: Test special case - when all prices are constant (scenario 2)
				// NATR should be exactly 0 after initial warmup
				if scenario == 2 {
					// Check that high == low == close for all values (constant prices)
					let is_constant = high.iter().zip(&low).zip(&close)
						.all(|((h, l), c)| (*h - *l).abs() < 1e-10 && (*h - *c).abs() < 1e-10);
					
					if is_constant && result.values.len() > period + 5 {
						// After the initial period, NATR should be 0 for constant prices
						// Start checking a few indices after warmup to allow stabilization
						for i in (period + 5)..result.values.len() {
							if result.values[i].is_finite() {
								prop_assert!(
									result.values[i].abs() < 1e-10,
									"NATR should be 0 for constant prices at index {}, got {}",
									i,
									result.values[i]
								);
							}
						}
					}
				}

				// Property 6: Test edge case with very small prices (scenario 1)
				if scenario == 1 {
					// Verify that NATR still works correctly with small prices
					for i in period..result.values.len() {
						if result.values[i].is_finite() && close[i] > 0.0 {
							// NATR should still be reasonable even with small prices
							prop_assert!(
								result.values[i] >= 0.0 && result.values[i] < 100000.0,
								"NATR out of bounds with small prices at index {}: got {}",
								i,
								result.values[i]
							);
						}
					}
				}

				// Property 7: Division by zero handling
				// Check if any close prices are near zero
				if close.iter().any(|&c| c.abs() < 1e-10) {
					// NATR should return 0 when close is 0, not NaN or infinity
					for (i, &c) in close.iter().enumerate() {
						if c.abs() < 1e-10 && i >= period - 1 {
							prop_assert!(
								result.values[i] == 0.0 || result.values[i].is_nan(),
								"NATR should be 0 or NaN when close is 0, got {} at index {}",
								result.values[i],
								i
							);
						}
					}
				}

				// Property 8: Check poison values aren't present
				#[cfg(debug_assertions)]
				{
					for (i, &val) in result.values.iter().enumerate() {
						if val.is_finite() {
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
				}

				Ok(())
			})?;

		Ok(())
	}

	macro_rules! generate_all_natr_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

	#[cfg(feature = "proptest")]
	generate_all_natr_tests!(check_natr_property);

	generate_all_natr_tests!(
		check_natr_partial_params,
		check_natr_accuracy,
		check_natr_with_zero_period,
		check_natr_period_exceeds_length,
		check_natr_very_small_dataset,
		check_natr_all_values_nan,
		check_natr_not_enough_valid_data,
		check_natr_slice_data_reinput,
		check_natr_nan_check,
		check_natr_default_candles,
		check_natr_streaming,
		check_natr_no_poison
	);

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

#[cfg(feature = "python")]
#[pyfunction(name = "natr")]
#[pyo3(signature = (high, low, close, period, kernel=None))]
pub fn natr_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = NatrParams { period: Some(period) };
	let input = NatrInput::from_slices(high_slice, low_slice, close_slice, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| natr_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "natr_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
pub fn natr_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
		return Err(PyValueError::new_err("natr: Mismatched input lengths"));
	}
	let cols = high_slice.len();
	
	let kern = validate_kernel(kernel, true)?;
	let sweep = NatrBatchRange { period: period_range };
	let combos = expand_grid(&sweep);
	let rows = combos.len();

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
				_ => kernel,
			};
			natr_batch_inner_into(high_slice, low_slice, close_slice, &sweep, simd, true, slice_out)
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
#[pyclass(name = "NatrStream")]
pub struct NatrStreamPy {
	stream: NatrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NatrStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = NatrParams { period: Some(period) };
		let stream = NatrStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(NatrStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		self.stream.update(high, low, close)
	}
}

/// Write directly to output slice - no allocations
pub fn natr_into_slice(
	dst: &mut [f64], 
	input: &NatrInput, 
	kern: Kernel
) -> Result<(), NatrError> {
	let (high, low, close, period) = match &input.data {
		NatrData::Candles { candles } => (
			candles.high.as_slice(),
			candles.low.as_slice(),
			candles.close.as_slice(),
			input.params.period.unwrap_or(14),
		),
		NatrData::Slices { high, low, close } => (
			*high,
			*low,
			*close,
			input.params.period.unwrap_or(14),
		),
	};

	let len = high.len().min(low.len()).min(close.len());
	
	if dst.len() != len {
		return Err(NatrError::MismatchedLength { 
			expected: len, 
			actual: dst.len() 
		});
	}

	// Validate parameters
	if len == 0 {
		return Err(NatrError::EmptyData);
	}
	if period == 0 {
		return Err(NatrError::InvalidPeriod { period, data_len: len });
	}
	if period > len {
		return Err(NatrError::InvalidPeriod { period, data_len: len });
	}

	// Find first valid index
	let first_valid_idx = {
		let first_valid_idx_h = high.iter().position(|&x| !x.is_nan());
		let first_valid_idx_l = low.iter().position(|&x| !x.is_nan());
		let first_valid_idx_c = close.iter().position(|&x| !x.is_nan());

		match (first_valid_idx_h, first_valid_idx_l, first_valid_idx_c) {
			(Some(h), Some(l), Some(c)) => Some(h.max(l).max(c)),
			_ => None,
		}
	};

	let first_valid_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(NatrError::AllValuesNaN),
	};

	if (len - first_valid_idx) < period {
		return Err(NatrError::NotEnoughValidData {
			needed: period,
			valid: len - first_valid_idx,
		});
	}

	// Choose kernel
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Compute NATR directly into dst
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => natr_scalar(high, low, close, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => natr_avx2(high, low, close, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => natr_avx512(high, low, close, period, first_valid_idx, dst),
			_ => unreachable!(),
		}
	}

	// Fill warmup with NaN
	for v in &mut dst[..(first_valid_idx + period - 1)] {
		*v = f64::NAN;
	}

	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_js(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	if high.len() != low.len() || high.len() != close.len() {
		return Err(JsValue::from_str("natr: Mismatched input lengths"));
	}
	let params = NatrParams { period: Some(period) };
	let input = NatrInput::from_slices(high, low, close, params);
	let mut output = vec![0.0; high.len()];
	natr_into_slice(&mut output, &input, detect_best_kernel())
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let params = NatrParams { period: Some(period) };
		let input = NatrInput::from_slices(high, low, close, params);
		
		// Check if any input pointer equals output pointer (aliasing)
		if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			natr_into_slice(&mut temp, &input, detect_best_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			natr_into_slice(out, &input, detect_best_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NatrBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NatrBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<NatrParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = natr_batch)]
pub fn natr_batch_unified_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	config: JsValue
) -> Result<JsValue, JsValue> {
	if high.len() != low.len() || high.len() != close.len() {
		return Err(JsValue::from_str("natr: Mismatched input lengths"));
	}
	let cfg: NatrBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = NatrBatchRange { period: cfg.period_range };

	let out = natr_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js = NatrBatchJsOutput { values: out.values, combos: out.combos, rows: out.rows, cols: out.cols };
	serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn natr_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to natr_batch_into"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		let sweep = NatrBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;
		
		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
		
		natr_batch_inner_into(high, low, close, &sweep, detect_best_kernel(), false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(rows)
	}
}
