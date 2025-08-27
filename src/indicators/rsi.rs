//! # Relative Strength Index (RSI)
//!
//! A momentum oscillator measuring recent price changes’ speed and magnitude.
//! RSI oscillates between 0 and 100. Typical period: 14 bars.
//!
//! ## Parameters
//! - **period**: Window size (number of bars, default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: rsi: All input values are `NaN`.
//! - **InvalidPeriod**: rsi: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: rsi: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(RsiOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(RsiError)`** otherwise.

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
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
use paste::paste;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for RsiInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			RsiData::Slice(slice) => slice,
			RsiData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum RsiData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RsiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct RsiParams {
	pub period: Option<usize>,
}

impl Default for RsiParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct RsiInput<'a> {
	pub data: RsiData<'a>,
	pub params: RsiParams,
}

impl<'a> RsiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: RsiParams) -> Self {
		Self {
			data: RsiData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: RsiParams) -> Self {
		Self {
			data: RsiData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", RsiParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct RsiBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for RsiBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl RsiBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<RsiOutput, RsiError> {
		let p = RsiParams { period: self.period };
		let i = RsiInput::from_candles(c, "close", p);
		rsi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<RsiOutput, RsiError> {
		let p = RsiParams { period: self.period };
		let i = RsiInput::from_slice(d, p);
		rsi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<RsiStream, RsiError> {
		let p = RsiParams { period: self.period };
		RsiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum RsiError {
	#[error("rsi: Input data slice is empty.")]
	EmptyInputData,
	#[error("rsi: All values are NaN.")]
	AllValuesNaN,
	#[error("rsi: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("rsi: Length mismatch: destination length = {dst}, source length = {src}")]
	LengthMismatch { dst: usize, src: usize },
	#[error("rsi: Unsupported kernel type for batch operation")]
	UnsupportedKernel,
}

#[inline]
pub fn rsi(input: &RsiInput) -> Result<RsiOutput, RsiError> {
	rsi_with_kernel(input, Kernel::Auto)
}

pub fn rsi_with_kernel(input: &RsiInput, kernel: Kernel) -> Result<RsiOutput, RsiError> {
	let data: &[f64] = match &input.data {
		RsiData::Candles { candles, source } => source_type(candles, source),
		RsiData::Slice(sl) => sl,
	};

	let len = data.len();
	if len == 0 {
		return Err(RsiError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RsiError::AllValuesNaN)?;
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(RsiError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(RsiError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut out = alloc_with_nan_prefix(len, first + period);
	rsi_compute_into(data, period, first, chosen, &mut out);
	Ok(RsiOutput { values: out })
}

#[inline(always)]
fn rsi_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => rsi_compute_into_scalar(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rsi_compute_into_scalar(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rsi_compute_into_scalar(data, period, first, out),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub fn rsi_into_slice(dst: &mut [f64], input: &RsiInput, kern: Kernel) -> Result<(), RsiError> {
	let data: &[f64] = match &input.data {
		RsiData::Candles { candles, source } => source_type(candles, source),
		RsiData::Slice(sl) => sl,
	};

	let len = data.len();
	if len == 0 {
		return Err(RsiError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RsiError::AllValuesNaN)?;
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(RsiError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(RsiError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	if dst.len() != data.len() {
		return Err(RsiError::LengthMismatch {
			dst: dst.len(),
			src: data.len(),
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	rsi_compute_into(data, period, first, chosen, dst);
	
	// Fill warmup period with NaN
	let warmup_end = first + period;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	Ok(())
}

#[inline(always)]
unsafe fn rsi_compute_into_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	let len = data.len();
	let inv_period = 1.0 / period as f64;
	let beta = 1.0 - inv_period;

	let mut avg_gain = 0.0;
	let mut avg_loss = 0.0;
	let mut has_nan = false;

	for i in (first + 1)..=(first + period) {
		let delta = data[i] - data[i - 1];
		if !delta.is_finite() { 
			has_nan = true;
			break;  // Any NaN in warmup invalidates the calculation
		}
		if delta > 0.0 {
			avg_gain += delta;
		} else if delta < 0.0 {
			avg_loss += -delta;
		}
	}
	
	let initial_rsi = if has_nan {
		avg_gain = f64::NAN;  // Poison the averages so NaN propagates
		avg_loss = f64::NAN;
		f64::NAN  // If any NaN in warmup, initial RSI is NaN
	} else {
		avg_gain *= inv_period;
		avg_loss *= inv_period;
		if avg_gain + avg_loss == 0.0 {
			50.0
		} else {
			100.0 * avg_gain / (avg_gain + avg_loss)
		}
	};
	out[first + period] = initial_rsi;

	for i in (first + period + 1)..len {
		let delta = data[i] - data[i - 1];
		let gain = if delta > 0.0 { delta } else { 0.0 };
		let loss = if delta < 0.0 { -delta } else { 0.0 };
		avg_gain = inv_period * gain + beta * avg_gain;
		avg_loss = inv_period * loss + beta * avg_loss;
		let rsi = if avg_gain + avg_loss == 0.0 {
			50.0
		} else {
			100.0 * avg_gain / (avg_gain + avg_loss)
		};
		out[i] = rsi;
	}
}

// --- Batch grid/range support ---

#[derive(Clone, Debug)]
pub struct RsiBatchRange {
	pub period: (usize, usize, usize),
}
impl Default for RsiBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}
#[derive(Clone, Debug, Default)]
pub struct RsiBatchBuilder {
	range: RsiBatchRange,
	kernel: Kernel,
}
impl RsiBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<RsiBatchOutput, RsiError> {
		rsi_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RsiBatchOutput, RsiError> {
		RsiBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RsiBatchOutput, RsiError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<RsiBatchOutput, RsiError> {
		RsiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn rsi_batch_with_kernel(data: &[f64], sweep: &RsiBatchRange, k: Kernel) -> Result<RsiBatchOutput, RsiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(RsiError::UnsupportedKernel),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	rsi_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RsiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RsiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl RsiBatchOutput {
	pub fn row_for_params(&self, p: &RsiParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &RsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &RsiBatchRange) -> Vec<RsiParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(RsiParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn rsi_batch_slice(data: &[f64], sweep: &RsiBatchRange, kern: Kernel) -> Result<RsiBatchOutput, RsiError> {
	rsi_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn rsi_batch_par_slice(data: &[f64], sweep: &RsiBatchRange, kern: Kernel) -> Result<RsiBatchOutput, RsiError> {
	rsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rsi_batch_inner(
	data: &[f64],
	sweep: &RsiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<RsiBatchOutput, RsiError> {
	if data.is_empty() {
		return Err(RsiError::EmptyInputData);
	}
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RsiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(RsiError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(RsiError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory for performance
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap())
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to mutable slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values: &mut [f64] = 
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => rsi_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rsi_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rsi_row_avx512(data, first, period, out_row),
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
	
	// Convert back to Vec for return
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(RsiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn rsi_batch_inner_into(
	data: &[f64],
	sweep: &RsiBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<RsiParams>, RsiError> {
	if data.is_empty() {
		return Err(RsiError::EmptyInputData);
	}
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RsiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(RsiError::AllValuesNaN)?;
	let rows = combos.len();
	let cols = data.len();
	
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if cols - first < max_p {
		return Err(RsiError::NotEnoughValidData {
			needed: max_p,
			valid: cols - first,
		});
	}
	if out.len() != rows * cols {
		return Err(RsiError::InvalidPeriod { period: out.len(), data_len: rows * cols });
	}

	// Treat caller buffer as uninit and initialize ONLY warmup prefixes per row.
	let out_mu: &mut [MaybeUninit<f64>] = unsafe {
		core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
	};
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
	init_matrix_prefixes(out_mu, cols, &warm);

	// Reinterpret as f64 for compute
	let values: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64, out_mu.len())
	};

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => rsi_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rsi_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rsi_row_avx512(data, first, period, out_row),
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

	Ok(combos)
}

#[inline(always)]
unsafe fn rsi_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	let len = data.len();
	let inv_period = 1.0 / period as f64;
	let beta = 1.0 - inv_period;
	let mut avg_gain = 0.0;
	let mut avg_loss = 0.0;
	let mut has_nan = false;

	for i in (first + 1)..=(first + period) {
		let delta = data[i] - data[i - 1];
		if !delta.is_finite() { 
			has_nan = true;
			break;  // Any NaN in warmup invalidates the calculation
		}
		if delta > 0.0 {
			avg_gain += delta;
		} else if delta < 0.0 {
			avg_loss += -delta;
		}
	}
	
	let initial_rsi = if has_nan {
		avg_gain = f64::NAN;  // Poison the averages so NaN propagates
		avg_loss = f64::NAN;
		f64::NAN  // If any NaN in warmup, initial RSI is NaN
	} else {
		avg_gain *= inv_period;
		avg_loss *= inv_period;
		if avg_gain + avg_loss == 0.0 {
			50.0
		} else {
			100.0 * avg_gain / (avg_gain + avg_loss)
		}
	};
	out[first + period] = initial_rsi;

	for i in (first + period + 1)..len {
		let delta = data[i] - data[i - 1];
		let gain = if delta > 0.0 { delta } else { 0.0 };
		let loss = if delta < 0.0 { -delta } else { 0.0 };
		avg_gain = inv_period * gain + beta * avg_gain;
		avg_loss = inv_period * loss + beta * avg_loss;
		let rsi = if avg_gain + avg_loss == 0.0 {
			50.0
		} else {
			100.0 * avg_gain / (avg_gain + avg_loss)
		};
		out[i] = rsi;
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsi_row_scalar(data, first, period, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		rsi_row_avx512_short(data, first, period, out)
	} else {
		rsi_row_avx512_long(data, first, period, out)
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsi_row_scalar(data, first, period, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsi_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsi_row_scalar(data, first, period, out)
}

// --- Streaming RSI (for parity with alma.rs) ---

#[derive(Debug, Clone)]
pub struct RsiStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	avg_gain: f64,
	avg_loss: f64,
	prev: f64,
	first: bool,
}
impl RsiStream {
	pub fn try_new(params: RsiParams) -> Result<Self, RsiError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(RsiError::InvalidPeriod { period, data_len: 0 });
		}
		let mut buffer = Vec::with_capacity(period);
		unsafe {
			buffer.set_len(period);
			for i in 0..period {
				*buffer.get_unchecked_mut(i) = f64::NAN;
			}
		}
		Ok(Self {
			period,
			buffer,
			head: 0,
			filled: false,
			avg_gain: 0.0,
			avg_loss: 0.0,
			prev: f64::NAN,
			first: true,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		if self.first {
			self.prev = value;
			self.first = false;
			return None;
		}
		let delta = value - self.prev;
		self.prev = value;
		self.buffer[self.head] = delta;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
			let mut gain = 0.0;
			let mut loss = 0.0;
			for &d in &self.buffer[..self.period] {
				if d > 0.0 {
					gain += d;
				} else {
					loss += -d;
				}
			}
			self.avg_gain = gain / self.period as f64;
			self.avg_loss = loss / self.period as f64;
			if self.avg_gain + self.avg_loss == 0.0 {
				return Some(50.0);
			} else {
				return Some(100.0 * self.avg_gain / (self.avg_gain + self.avg_loss));
			}
		}
		if !self.filled {
			return None;
		}
		let gain = if delta > 0.0 { delta } else { 0.0 };
		let loss = if delta < 0.0 { -delta } else { 0.0 };
		let inv_period = 1.0 / self.period as f64;
		let beta = 1.0 - inv_period;
		self.avg_gain = inv_period * gain + beta * self.avg_gain;
		self.avg_loss = inv_period * loss + beta * self.avg_loss;
		if self.avg_gain + self.avg_loss == 0.0 {
			Some(50.0)
		} else {
			Some(100.0 * self.avg_gain / (self.avg_gain + self.avg_loss))
		}
	}
}

// ---- Tests ----

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_rsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let partial_params = RsiParams { period: None };
		let input = RsiInput::from_candles(&candles, "close", partial_params);
		let result = rsi_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		Ok(())
	}
	fn check_rsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = RsiInput::from_candles(&candles, "close", RsiParams { period: Some(14) });
		let result = rsi_with_kernel(&input, kernel)?;
		let expected_last_five = [43.42, 42.68, 41.62, 42.86, 39.01];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-2,
				"[{}] RSI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}
	fn check_rsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = RsiInput::with_default_candles(&candles);
		match input.data {
			RsiData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected RsiData::Candles"),
		}
		let output = rsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_rsi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = RsiParams { period: Some(0) };
		let input = RsiInput::from_slice(&input_data, params);
		let res = rsi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] RSI should fail with zero period", test_name);
		Ok(())
	}
	fn check_rsi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = RsiParams { period: Some(10) };
		let input = RsiInput::from_slice(&data_small, params);
		let res = rsi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] RSI should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_rsi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = RsiParams { period: Some(14) };
		let input = RsiInput::from_slice(&single_point, params);
		let res = rsi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] RSI should fail with insufficient data", test_name);
		Ok(())
	}
	fn check_rsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = RsiParams { period: Some(14) };
		let first_input = RsiInput::from_candles(&candles, "close", first_params);
		let first_result = rsi_with_kernel(&first_input, kernel)?;
		let second_params = RsiParams { period: Some(5) };
		let second_input = RsiInput::from_slice(&first_result.values, second_params);
		let second_result = rsi_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		if second_result.values.len() > 240 {
			for i in 240..second_result.values.len() {
				assert!(!second_result.values[i].is_nan(), "Found NaN in RSI at {}", i);
			}
		}
		Ok(())
	}
	fn check_rsi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = RsiInput::from_candles(&candles, "close", RsiParams { period: Some(14) });
		let res = rsi_with_kernel(&input, kernel)?;
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
	fn check_rsi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 14;
		let input = RsiInput::from_candles(&candles, "close", RsiParams { period: Some(period) });
		let batch_output = rsi_with_kernel(&input, kernel)?.values;

		let mut stream = RsiStream::try_new(RsiParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(rsi_val) => stream_values.push(rsi_val),
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
				diff < 1e-6,
				"[{}] RSI streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_rsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			RsiParams::default(),
			RsiParams { period: Some(2) },    // minimum
			RsiParams { period: Some(5) },    // small
			RsiParams { period: Some(7) },    // small
			RsiParams { period: Some(10) },   // small
			RsiParams { period: Some(14) },   // default/typical
			RsiParams { period: Some(20) },   // medium
			RsiParams { period: Some(30) },   // medium
			RsiParams { period: Some(50) },   // large
			RsiParams { period: Some(100) },  // very large
			RsiParams { period: Some(200) },  // extremely large
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = RsiInput::from_candles(&candles, "close", params.clone());
			let output = rsi_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

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
	fn check_rsi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_rsi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate test data with realistic market conditions
		let strat = (2usize..=100)
			.prop_flat_map(|period| {
				(
					// Price data between -1e6 and 1e6 (filtered for finite values)
					prop::collection::vec(
						(-1e6f64..1e6f64).prop_filter("finite price", |x| x.is_finite() && x.abs() > 1e-10),
						period + 10..400,
					),
					Just(period),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = RsiParams { period: Some(period) };
				let input = RsiInput::from_slice(&data, params);

				// Get results from the kernel being tested
				let RsiOutput { values: out } = rsi_with_kernel(&input, kernel)?;
				
				// Get reference results from scalar kernel for comparison
				let RsiOutput { values: ref_out } = rsi_with_kernel(&input, Kernel::Scalar)?;

				// Find first non-NaN index
				let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				let warmup_end = first_valid + period;

				// Property 1: RSI values must be in range [0, 100]
				for (i, &val) in out.iter().enumerate() {
					if !val.is_nan() {
						prop_assert!(
							val >= 0.0 && val <= 100.0,
							"[{}] RSI value {} at index {} is out of range [0, 100]",
							test_name, val, i
						);
					}
				}

				// Property 2: Warmup period handling
				for i in 0..warmup_end.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"[{}] Expected NaN during warmup at index {}, got {}",
						test_name, i, out[i]
					);
				}
				
				// First non-NaN should be at warmup_end
				if warmup_end < out.len() {
					prop_assert!(
						!out[warmup_end].is_nan(),
						"[{}] Expected non-NaN at index {} (warmup_end), got NaN",
						test_name, warmup_end
					);
				}

				// Property 3: Kernel consistency
				for i in 0..out.len() {
					let y = out[i];
					let r = ref_out[i];
					
					if y.is_nan() && r.is_nan() {
						continue;
					}
					
					prop_assert!(
						(y - r).abs() < 1e-9,
						"[{}] Kernel mismatch at index {}: {} vs {} (diff: {})",
						test_name, i, y, r, (y - r).abs()
					);
				}

				// Property 4: Constant prices should yield RSI = 50
				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && warmup_end < out.len() {
					for i in warmup_end..out.len() {
						prop_assert!(
							(out[i] - 50.0).abs() < 1e-9,
							"[{}] Constant prices should yield RSI=50, got {} at index {}",
							test_name, out[i], i
						);
					}
				}

				// Property 5: Monotonic increasing prices should trend toward 100
				// Use period-adaptive thresholds for more realistic testing
				let strictly_increasing = data.windows(2).all(|w| w[1] > w[0] + 1e-10);
				if strictly_increasing && out.len() > warmup_end + 10 {
					let last_rsi = out[out.len() - 1];
					let high_threshold = if period <= 5 {
						60.0
					} else if period <= 20 {
						65.0
					} else {
						70.0
					};
					prop_assert!(
						last_rsi > high_threshold,
						"[{}] Strictly increasing prices should yield RSI > {} (period={}), got {}",
						test_name, high_threshold, period, last_rsi
					);
				}

				// Property 6: Monotonic decreasing prices should trend toward 0
				// Use period-adaptive thresholds for more realistic testing
				let strictly_decreasing = data.windows(2).all(|w| w[1] < w[0] - 1e-10);
				if strictly_decreasing && out.len() > warmup_end + 10 {
					let last_rsi = out[out.len() - 1];
					let low_threshold = if period <= 5 {
						40.0
					} else if period <= 20 {
						35.0
					} else {
						30.0
					};
					prop_assert!(
						last_rsi < low_threshold,
						"[{}] Strictly decreasing prices should yield RSI < {} (period={}), got {}",
						test_name, low_threshold, period, last_rsi
					);
				}

				// Property 7: No poison values (in debug mode)
				#[cfg(debug_assertions)]
				{
					for (i, &val) in out.iter().enumerate() {
						if val.is_nan() {
							continue;
						}
						
						let bits = val.to_bits();
						prop_assert!(
							bits != 0x11111111_11111111 && 
							bits != 0x22222222_22222222 && 
							bits != 0x33333333_33333333,
							"[{}] Found poison value {} (0x{:016X}) at index {}",
							test_name, val, bits, i
						);
					}
				}

				// Property 8: Oscillating prices should keep RSI in middle range
				// Check if prices alternate between increases and decreases
				let mut oscillating = true;
				let mut prev_delta = 0.0;
				for window in data.windows(2) {
					let delta = window[1] - window[0];
					if prev_delta != 0.0 && delta != 0.0 {
						// Check if signs are different (oscillating)
						if (delta > 0.0 && prev_delta > 0.0) || (delta < 0.0 && prev_delta < 0.0) {
							oscillating = false;
							break;
						}
					}
					prev_delta = delta;
				}
				
				// If we have oscillating prices with sufficient data after warmup
				if oscillating && out.len() > warmup_end + 10 && prev_delta != 0.0 {
					// RSI should stay in middle range (roughly 40-60)
					let last_quarter_start = out.len() - (out.len() - warmup_end) / 4;
					for i in last_quarter_start..out.len() {
						if !out[i].is_nan() {
							prop_assert!(
								out[i] >= 35.0 && out[i] <= 65.0,
								"[{}] Oscillating prices should keep RSI in [35, 65] range, got {} at index {}",
								test_name, out[i], i
							);
						}
					}
				}

				// Property 9: Mathematical consistency check for a few values
				if warmup_end + 5 < out.len() {
					// Manually calculate RSI for one point to verify
					let idx = warmup_end + 3;
					let mut avg_gain = 0.0;
					let mut avg_loss = 0.0;
					
					// Initial average calculation
					for j in (first_valid + 1)..=(first_valid + period) {
						let delta = data[j] - data[j - 1];
						if delta > 0.0 {
							avg_gain += delta;
						} else {
							avg_loss += -delta;
						}
					}
					avg_gain /= period as f64;
					avg_loss /= period as f64;
					
					// Update averages using EMA
					let inv_period = 1.0 / period as f64;
					let beta = 1.0 - inv_period;
					for j in (first_valid + period + 1)..=idx {
						let delta = data[j] - data[j - 1];
						let gain = if delta > 0.0 { delta } else { 0.0 };
						let loss = if delta < 0.0 { -delta } else { 0.0 };
						avg_gain = inv_period * gain + beta * avg_gain;
						avg_loss = inv_period * loss + beta * avg_loss;
					}
					
					let expected_rsi = if avg_gain + avg_loss == 0.0 {
						50.0
					} else {
						100.0 * avg_gain / (avg_gain + avg_loss)
					};
					
					prop_assert!(
						(out[idx] - expected_rsi).abs() < 1e-9,
						"[{}] RSI calculation mismatch at index {}: got {}, expected {}",
						test_name, idx, out[idx], expected_rsi
					);
				}

				Ok(())
			})?;

		Ok(())
	}

	macro_rules! generate_all_rsi_tests {
        ($($test_fn:ident),*) => {
            paste! {
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

	fn check_rsi_error_variants(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		// Test LengthMismatch error
		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
		let mut dst = vec![0.0; 3]; // Wrong size
		let params = RsiParams { period: Some(2) };
		let input = RsiInput::from_slice(&data, params);
		
		match rsi_into_slice(&mut dst, &input, kernel) {
			Err(RsiError::LengthMismatch { dst: 3, src: 5 }) => {
				// Expected error
			}
			other => panic!("[{}] Expected LengthMismatch error, got {:?}", test_name, other),
		}
		
		// Test UnsupportedKernel error for batch operations
		let sweep = RsiBatchRange { period: (14, 14, 0) };
		match rsi_batch_with_kernel(&data, &sweep, Kernel::Scalar) {
			Err(RsiError::UnsupportedKernel) => {
				// Expected error  
			}
			other => panic!("[{}] Expected UnsupportedKernel error, got {:?}", test_name, other),
		}
		
		Ok(())
	}

	generate_all_rsi_tests!(
		check_rsi_partial_params,
		check_rsi_accuracy,
		check_rsi_default_candles,
		check_rsi_zero_period,
		check_rsi_period_exceeds_length,
		check_rsi_very_small_dataset,
		check_rsi_reinput,
		check_rsi_nan_handling,
		check_rsi_streaming,
		check_rsi_no_poison,
		check_rsi_error_variants
	);

	#[cfg(feature = "proptest")]
	generate_all_rsi_tests!(check_rsi_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = RsiBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = RsiParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [43.42, 42.68, 41.62, 42.86, 39.01];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-2,
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

		let test_configs = vec![
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods
			(30, 60, 15),    // Large periods
			(2, 5, 1),       // Dense small range
			(7, 21, 7),      // Common RSI periods (weekly)
			(10, 50, 10),    // Extended range
			(14, 28, 14),    // Double default period
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = RsiBatchBuilder::new()
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
			paste! {
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

// --- Python Bindings ---

#[cfg(feature = "python")]
#[pyfunction(name = "rsi")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn rsi_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = RsiParams {
		period: Some(period),
	};
	let input = RsiInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| rsi_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "RsiStream")]
pub struct RsiStreamPy {
	stream: RsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RsiStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = RsiParams {
			period: Some(period),
		};
		let stream = RsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(RsiStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "rsi_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn rsi_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = RsiBatchRange {
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
				_ => kernel,
			};
			rsi_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsi_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = RsiParams { period: Some(period) };
	let input = RsiInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];
	
	rsi_into_slice(&mut output, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsi_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rsi_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to rsi_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = RsiParams { period: Some(period) };
		let input = RsiInput::from_slice(data, params);

		if in_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			rsi_into_slice(&mut temp, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			rsi_into_slice(out, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RsiBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RsiBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RsiParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = rsi_batch)]
pub fn rsi_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: RsiBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = RsiBatchRange {
		period: config.period_range,
	};

	let output = rsi_batch_with_kernel(data, &sweep, detect_best_batch_kernel())
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = RsiBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
