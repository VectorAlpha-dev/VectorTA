//! # Directional Indicator (DI)
//!
//! Calculates both +DI (plus directional indicator) and -DI (minus directional indicator)
//! using the same approach as Wilder's DMI, measuring trend strength and direction
//! by comparing upward and downward price movements over a specified period.
//!
//! ## Parameters
//! - **period**: The smoothing window size. Defaults to 14.
//!
//! ## Errors
//! - **EmptyData**: di: Input data slice is empty.
//! - **InvalidPeriod**: di: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: di: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: di: All high/low/close values are `NaN`.
//!
//! ## Returns
//! - **`Ok(DiOutput)`** on success, containing two `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the calculation window is filled.
//! - **`Err(DiError)`** otherwise.

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
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DiData<'a> {
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
pub struct DiOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DiParams {
	pub period: Option<usize>,
}

impl Default for DiParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct DiInput<'a> {
	pub data: DiData<'a>,
	pub params: DiParams,
}

impl<'a> DiInput<'a> {
	#[inline(always)]
	pub fn from_candles(candles: &'a Candles, params: DiParams) -> Self {
		Self {
			data: DiData::Candles { candles },
			params,
		}
	}
	#[inline(always)]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: DiParams) -> Self {
		Self {
			data: DiData::Slices { high, low, close },
			params,
		}
	}
	#[inline(always)]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: DiData::Candles { candles },
			params: DiParams::default(),
		}
	}
	#[inline(always)]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct DiBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for DiBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl DiBuilder {
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
	pub fn apply(self, candles: &Candles) -> Result<DiOutput, DiError> {
		let params = DiParams { period: self.period };
		let input = DiInput::from_candles(candles, params);
		di_with_kernel(&input, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<DiOutput, DiError> {
		let params = DiParams { period: self.period };
		let input = DiInput::from_slices(high, low, close, params);
		di_with_kernel(&input, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<DiStream, DiError> {
		let params = DiParams { period: self.period };
		DiStream::try_new(params)
	}
}

#[derive(Debug, Error)]
pub enum DiError {
	#[error("di: Empty data provided for DI.")]
	EmptyData,
	#[error("di: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("di: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("di: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn di(input: &DiInput) -> Result<DiOutput, DiError> {
	di_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn di_prepare<'a>(input: &'a DiInput<'a>, kernel: Kernel) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), DiError> {
	let (high, low, close) = match &input.data {
		DiData::Candles { candles } => {
			let h = source_type(candles, "high");
			let l = source_type(candles, "low");
			let c = source_type(candles, "close");
			(h, l, c)
		}
		DiData::Slices { high, low, close } => (*high, *low, *close),
	};
	let n = high.len();
	if n == 0 || low.len() != n || close.len() != n {
		return Err(DiError::EmptyData);
	}
	let period = input.get_period();
	if period == 0 || period > n {
		return Err(DiError::InvalidPeriod { period, data_len: n });
	}
	let first_valid_idx = (0..n).find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()));
	let first_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(DiError::AllValuesNaN),
	};
	if (n - first_idx) < period {
		return Err(DiError::NotEnoughValidData {
			needed: period,
			valid: n - first_idx,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((high, low, close, period, first_idx, chosen))
}

pub fn di_with_kernel(input: &DiInput, kernel: Kernel) -> Result<DiOutput, DiError> {
	let (high, low, close, period, first_idx, chosen) = di_prepare(input, kernel)?;

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => di_scalar(high, low, close, period, first_idx),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => di_avx2(high, low, close, period, first_idx),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => di_avx512(high, low, close, period, first_idx),
			_ => unreachable!(),
		}
	}
}

// Stubs for non-nightly builds
#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
#[inline(always)]
pub unsafe fn di_avx2_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) {
	di_scalar_into(high, low, close, period, first_idx, out_plus, out_minus)
}

#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
#[inline(always)]
pub unsafe fn di_avx512_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) {
	di_scalar_into(high, low, close, period, first_idx, out_plus, out_minus)
}

#[inline(always)]
fn di_compute_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
	kernel: Kernel,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => di_scalar_into(high, low, close, period, first_idx, out_plus, out_minus),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => di_avx2_into(high, low, close, period, first_idx, out_plus, out_minus),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => di_avx512_into(high, low, close, period, first_idx, out_plus, out_minus),
			_ => unreachable!(),
		}
	}
}

/// Write DI results directly to output slices - no allocations
pub fn di_into_slice(
	dst_plus: &mut [f64],
	dst_minus: &mut [f64],
	input: &DiInput,
	kern: Kernel,
) -> Result<(), DiError> {
	let (high, low, close, period, first_idx, chosen) = di_prepare(input, kern)?;

	let n = high.len();
	if dst_plus.len() != n || dst_minus.len() != n {
		return Err(DiError::InvalidPeriod {
			period: n,
			data_len: dst_plus.len().min(dst_minus.len()),
		});
	}

	di_compute_into(high, low, close, period, first_idx, chosen, dst_plus, dst_minus);
	
	// Fill warmup period with NaN
	let warmup_end = first_idx + period - 1;
	for v in &mut dst_plus[..warmup_end] {
		*v = f64::NAN;
	}
	for v in &mut dst_minus[..warmup_end] {
		*v = f64::NAN;
	}
	
	Ok(())
}

#[inline(always)]
pub unsafe fn di_scalar_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) {
	let mut prev_high = high[first_idx];
	let mut prev_low = low[first_idx];
	let mut prev_close = close[first_idx];
	let mut plus_dm_sum = 0.0;
	let mut minus_dm_sum = 0.0;
	let mut tr_sum = 0.0;

	for i in (first_idx + 1)..(first_idx + period) {
		let diff_p = high[i] - prev_high;
		let diff_m = prev_low - low[i];
		prev_high = high[i];
		prev_low = low[i];
		let tr = true_range(high[i], low[i], prev_close);
		prev_close = close[i];
		if diff_p > 0.0 && diff_p > diff_m {
			plus_dm_sum += diff_p;
		}
		if diff_m > 0.0 && diff_m > diff_p {
			minus_dm_sum += diff_m;
		}
		tr_sum += tr;
	}

	let mut idx = first_idx + period - 1;
	let mut current_plus_dm = plus_dm_sum;
	let mut current_minus_dm = minus_dm_sum;
	let mut current_tr = tr_sum;

	out_plus[idx] = if current_tr == 0.0 {
		0.0
	} else {
		(current_plus_dm / current_tr) * 100.0
	};
	out_minus[idx] = if current_tr == 0.0 {
		0.0
	} else {
		(current_minus_dm / current_tr) * 100.0
	};
	idx += 1;

	while idx < high.len() {
		let diff_p = high[idx] - prev_high;
		let diff_m = prev_low - low[idx];
		prev_high = high[idx];
		prev_low = low[idx];
		let tr = true_range(high[idx], low[idx], prev_close);
		prev_close = close[idx];
		if diff_p > 0.0 && diff_p > diff_m {
			current_plus_dm = current_plus_dm - (current_plus_dm / (period as f64)) + diff_p;
		} else {
			current_plus_dm = current_plus_dm - (current_plus_dm / (period as f64));
		}
		if diff_m > 0.0 && diff_m > diff_p {
			current_minus_dm = current_minus_dm - (current_minus_dm / (period as f64)) + diff_m;
		} else {
			current_minus_dm = current_minus_dm - (current_minus_dm / (period as f64));
		}
		current_tr = current_tr - (current_tr / (period as f64)) + tr;
		out_plus[idx] = if current_tr == 0.0 {
			0.0
		} else {
			(current_plus_dm / current_tr) * 100.0
		};
		out_minus[idx] = if current_tr == 0.0 {
			0.0
		} else {
			(current_minus_dm / current_tr) * 100.0
		};
		idx += 1;
	}
}

#[inline(always)]
pub unsafe fn di_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
) -> Result<DiOutput, DiError> {
	let n = high.len();
	let mut plus_di = alloc_with_nan_prefix(n, first_idx + period - 1);
	let mut minus_di = alloc_with_nan_prefix(n, first_idx + period - 1);
	di_scalar_into(high, low, close, period, first_idx, &mut plus_di, &mut minus_di);
	Ok(DiOutput {
		plus: plus_di,
		minus: minus_di,
	})
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_avx2_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) {
	di_scalar_into(high, low, close, period, first_idx, out_plus, out_minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
) -> Result<DiOutput, DiError> {
	di_scalar(high, low, close, period, first_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_avx512_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) {
	di_scalar_into(high, low, close, period, first_idx, out_plus, out_minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
) -> Result<DiOutput, DiError> {
	di_scalar(high, low, close, period, first_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
) -> Result<DiOutput, DiError> {
	di_avx512(high, low, close, period, first_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first_idx: usize,
) -> Result<DiOutput, DiError> {
	di_avx512(high, low, close, period, first_idx)
}

// Streaming (single-point update)
#[derive(Debug, Clone)]
pub struct DiStream {
	period: usize,
	buffer_high: Vec<f64>,
	buffer_low: Vec<f64>,
	buffer_close: Vec<f64>,
	head: usize,
	filled: bool,
	prev_high: f64,
	prev_low: f64,
	prev_close: f64,
	plus_dm: f64,
	minus_dm: f64,
	tr: f64,
}

impl DiStream {
	pub fn try_new(params: DiParams) -> Result<Self, DiError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(DiError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buffer_high: vec![f64::NAN; period],
			buffer_low: vec![f64::NAN; period],
			buffer_close: vec![f64::NAN; period],
			head: 0,
			filled: false,
			prev_high: f64::NAN,
			prev_low: f64::NAN,
			prev_close: f64::NAN,
			plus_dm: 0.0,
			minus_dm: 0.0,
			tr: 0.0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
		self.buffer_high[self.head] = high;
		self.buffer_low[self.head] = low;
		self.buffer_close[self.head] = close;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
		}

		if !self.filled {
			self.prev_high = high;
			self.prev_low = low;
			self.prev_close = close;
			return None;
		}

		let mut plus_dm_sum = 0.0;
		let mut minus_dm_sum = 0.0;
		let mut tr_sum = 0.0;
		let mut prev_h = self.buffer_high[self.head];
		let mut prev_l = self.buffer_low[self.head];
		let mut prev_c = self.buffer_close[self.head];
		for i in 0..self.period {
			let idx = (self.head + i) % self.period;
			let h = self.buffer_high[idx];
			let l = self.buffer_low[idx];
			let c = self.buffer_close[idx];
			let diff_p = h - prev_h;
			let diff_m = prev_l - l;
			if diff_p > 0.0 && diff_p > diff_m {
				plus_dm_sum += diff_p;
			}
			if diff_m > 0.0 && diff_m > diff_p {
				minus_dm_sum += diff_m;
			}
			tr_sum += true_range(h, l, prev_c);
			prev_h = h;
			prev_l = l;
			prev_c = c;
		}
		let plus = if tr_sum == 0.0 {
			0.0
		} else {
			(plus_dm_sum / tr_sum) * 100.0
		};
		let minus = if tr_sum == 0.0 {
			0.0
		} else {
			(minus_dm_sum / tr_sum) * 100.0
		};
		Some((plus, minus))
	}
}

// Batch Range and Builder
#[derive(Clone, Debug)]
pub struct DiBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for DiBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct DiBatchBuilder {
	range: DiBatchRange,
	kernel: Kernel,
}

impl DiBatchBuilder {
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
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<DiBatchOutput, DiError> {
		di_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<DiBatchOutput, DiError> {
		let h = source_type(c, "high");
		let l = source_type(c, "low");
		let cl = source_type(c, "close");
		self.apply_slices(h, l, cl)
	}
	pub fn with_default_candles(c: &Candles) -> Result<DiBatchOutput, DiError> {
		DiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub fn di_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DiBatchRange,
	k: Kernel,
) -> Result<DiBatchOutput, DiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(DiError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	di_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DiBatchOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
	pub combos: Vec<DiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl DiBatchOutput {
	pub fn row_for_params(&self, p: &DiParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn plus_for(&self, p: &DiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.plus[start..start + self.cols]
		})
	}
	pub fn minus_for(&self, p: &DiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.minus[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &DiBatchRange) -> Vec<DiParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(DiParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn di_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DiBatchRange,
	kern: Kernel,
) -> Result<DiBatchOutput, DiError> {
	di_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn di_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DiBatchRange,
	kern: Kernel,
) -> Result<DiBatchOutput, DiError> {
	di_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn di_batch_inner_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DiBatchRange,
	kern: Kernel,
	parallel: bool,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) -> Result<Vec<DiParams>, DiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let n = high.len();
	if n == 0 || low.len() != n || close.len() != n {
		return Err(DiError::EmptyData);
	}
	let first = (0..n)
		.find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
		.ok_or(DiError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if n - first < max_p {
		return Err(DiError::NotEnoughValidData {
			needed: max_p,
			valid: n - first,
		});
	}

	let cols = n;
	
	// Initialize NaN prefixes for each row
	for (row, combo) in combos.iter().enumerate() {
		let warmup = first + combo.period.unwrap() - 1;
		let row_start = row * cols;
		for i in 0..warmup {
			out_plus[row_start + i] = f64::NAN;
			out_minus[row_start + i] = f64::NAN;
		}
	}

	let do_row = |row: usize, out_plus: &mut [f64], out_minus: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let result = di_row_scalar(high, low, close, period, first, out_plus, out_minus);
		debug_assert!(result.is_ok());
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out_plus.par_chunks_mut(cols)
				.zip(out_minus.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (pl, mi))| do_row(row, pl, mi));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (pl, mi)) in out_plus.chunks_mut(cols).zip(out_minus.chunks_mut(cols)).enumerate() {
				do_row(row, pl, mi);
			}
		}
	} else {
		for (row, (pl, mi)) in out_plus.chunks_mut(cols).zip(out_minus.chunks_mut(cols)).enumerate() {
			do_row(row, pl, mi);
		}
	}

	Ok(combos)
}

#[inline(always)]
fn di_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DiBatchOutput, DiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let n = high.len();
	if n == 0 || low.len() != n || close.len() != n {
		return Err(DiError::EmptyData);
	}
	let first = (0..n)
		.find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
		.ok_or(DiError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if n - first < max_p {
		return Err(DiError::NotEnoughValidData {
			needed: max_p,
			valid: n - first,
		});
	}

	let rows = combos.len();
	let cols = n;
	
	// Calculate warmup periods for each parameter combination
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	
	// Allocate uninitialized matrices
	let mut plus_mu = make_uninit_matrix(rows, cols);
	let mut minus_mu = make_uninit_matrix(rows, cols);
	
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut plus_mu, cols, &warmup_periods);
	init_matrix_prefixes(&mut minus_mu, cols, &warmup_periods);
	
	// Convert to mutable slices for computation
	let mut plus_guard = core::mem::ManuallyDrop::new(plus_mu);
	let mut minus_guard = core::mem::ManuallyDrop::new(minus_mu);
	let plus: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(plus_guard.as_mut_ptr() as *mut f64, plus_guard.len()) };
	let minus: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(minus_guard.as_mut_ptr() as *mut f64, minus_guard.len()) };

	let do_row = |row: usize, out_plus: &mut [f64], out_minus: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let result = di_row_scalar(high, low, close, period, first, out_plus, out_minus);
		debug_assert!(result.is_ok());
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			plus.par_chunks_mut(cols)
				.zip(minus.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (pl, mi))| do_row(row, pl, mi));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (pl, mi)) in plus.chunks_mut(cols).zip(minus.chunks_mut(cols)).enumerate() {
				do_row(row, pl, mi);
			}
		}
	} else {
		for (row, (pl, mi)) in plus.chunks_mut(cols).zip(minus.chunks_mut(cols)).enumerate() {
			do_row(row, pl, mi);
		}
	}

	// Convert back to Vec for output
	let plus = unsafe {
		Vec::from_raw_parts(
			plus_guard.as_mut_ptr() as *mut f64,
			plus_guard.len(),
			plus_guard.capacity(),
		)
	};
	let minus = unsafe {
		Vec::from_raw_parts(
			minus_guard.as_mut_ptr() as *mut f64,
			minus_guard.len(),
			minus_guard.capacity(),
		)
	};

	Ok(DiBatchOutput {
		plus,
		minus,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn di_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) -> Result<(), DiError> {
	let n = high.len();
	let mut prev_high = high[first];
	let mut prev_low = low[first];
	let mut prev_close = close[first];
	let mut plus_dm_sum = 0.0;
	let mut minus_dm_sum = 0.0;
	let mut tr_sum = 0.0;

	for i in (first + 1)..(first + period) {
		let diff_p = high[i] - prev_high;
		let diff_m = prev_low - low[i];
		prev_high = high[i];
		prev_low = low[i];
		let tr = true_range(high[i], low[i], prev_close);
		prev_close = close[i];
		if diff_p > 0.0 && diff_p > diff_m {
			plus_dm_sum += diff_p;
		}
		if diff_m > 0.0 && diff_m > diff_p {
			minus_dm_sum += diff_m;
		}
		tr_sum += tr;
	}

	let mut idx = first + period - 1;
	let mut current_plus_dm = plus_dm_sum;
	let mut current_minus_dm = minus_dm_sum;
	let mut current_tr = tr_sum;

	out_plus[idx] = if current_tr == 0.0 {
		0.0
	} else {
		(current_plus_dm / current_tr) * 100.0
	};
	out_minus[idx] = if current_tr == 0.0 {
		0.0
	} else {
		(current_minus_dm / current_tr) * 100.0
	};
	idx += 1;

	while idx < n {
		let diff_p = high[idx] - prev_high;
		let diff_m = prev_low - low[idx];
		prev_high = high[idx];
		prev_low = low[idx];
		let tr = true_range(high[idx], low[idx], prev_close);
		prev_close = close[idx];
		if diff_p > 0.0 && diff_p > diff_m {
			current_plus_dm = current_plus_dm - (current_plus_dm / (period as f64)) + diff_p;
		} else {
			current_plus_dm = current_plus_dm - (current_plus_dm / (period as f64));
		}
		if diff_m > 0.0 && diff_m > diff_p {
			current_minus_dm = current_minus_dm - (current_minus_dm / (period as f64)) + diff_m;
		} else {
			current_minus_dm = current_minus_dm - (current_minus_dm / (period as f64));
		}
		current_tr = current_tr - (current_tr / (period as f64)) + tr;
		out_plus[idx] = if current_tr == 0.0 {
			0.0
		} else {
			(current_plus_dm / current_tr) * 100.0
		};
		out_minus[idx] = if current_tr == 0.0 {
			0.0
		} else {
			(current_minus_dm / current_tr) * 100.0
		};
		idx += 1;
	}
	Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) -> Result<(), DiError> {
	di_row_scalar(high, low, close, period, first, out_plus, out_minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) -> Result<(), DiError> {
	di_row_scalar(high, low, close, period, first, out_plus, out_minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) -> Result<(), DiError> {
	di_row_avx512(high, low, close, period, first, out_plus, out_minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn di_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) -> Result<(), DiError> {
	di_row_avx512(high, low, close, period, first, out_plus, out_minus)
}

// Utility
#[inline(always)]
fn true_range(current_high: f64, current_low: f64, prev_close: f64) -> f64 {
	let mut tr1 = current_high - current_low;
	let tr2 = (current_high - prev_close).abs();
	let tr3 = (current_low - prev_close).abs();
	if tr2 > tr1 {
		tr1 = tr2;
	}
	if tr3 > tr1 {
		tr1 = tr3;
	}
	tr1
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_di_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = DiParams { period: None };
		let input_default = DiInput::from_candles(&candles, default_params);
		let output_default = di_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.plus.len(), candles.close.len());
		assert_eq!(output_default.minus.len(), candles.close.len());
		let params_period_10 = DiParams { period: Some(10) };
		let input_period_10 = DiInput::from_candles(&candles, params_period_10);
		let output_period_10 = di_with_kernel(&input_period_10, kernel)?;
		assert_eq!(output_period_10.plus.len(), candles.close.len());
		assert_eq!(output_period_10.minus.len(), candles.close.len());
		Ok(())
	}
	fn check_di_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = DiParams { period: Some(14) };
		let input = DiInput::from_candles(&candles, params);
		let di_result = di_with_kernel(&input, kernel)?;
		assert_eq!(di_result.plus.len(), candles.close.len());
		assert_eq!(di_result.minus.len(), candles.close.len());
		let test_plus = [
			10.99067007335658,
			11.306993269828585,
			10.948661818939213,
			10.683207768215592,
			9.802180952619183,
		];
		let test_minus = [
			28.06728094177839,
			27.331240567633152,
			27.759989125359493,
			26.951434842917386,
			30.748897303623057,
		];
		if di_result.plus.len() > 5 {
			let plus_tail = &di_result.plus[di_result.plus.len() - 5..];
			let minus_tail = &di_result.minus[di_result.minus.len() - 5..];
			for i in 0..5 {
				assert!(
					(plus_tail[i] - test_plus[i]).abs() < 1e-6,
					"Mismatch in +DI at tail index {}: expected {}, got {}",
					i,
					test_plus[i],
					plus_tail[i]
				);
				assert!(
					(minus_tail[i] - test_minus[i]).abs() < 1e-6,
					"Mismatch in -DI at tail index {}: expected {}, got {}",
					i,
					test_minus[i],
					minus_tail[i]
				);
			}
		}
		Ok(())
	}
	fn check_di_with_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 11.0, 12.0];
		let low = [9.0, 8.0, 7.0];
		let close = [9.5, 10.0, 11.0];
		let params = DiParams { period: Some(0) };
		let input = DiInput::from_slices(&high, &low, &close, params);
		let result = di_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}
	fn check_di_with_period_exceeding_data_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 11.0, 12.0];
		let low = [9.0, 8.0, 7.0];
		let close = [9.5, 10.0, 11.0];
		let params = DiParams { period: Some(10) };
		let input = DiInput::from_slices(&high, &low, &close, params);
		let result = di_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}
	fn check_di_very_small_data_set(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [42.0];
		let low = [41.0];
		let close = [41.5];
		let params = DiParams { period: Some(14) };
		let input = DiInput::from_slices(&high, &low, &close, params);
		let result = di_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}
	fn check_di_with_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = DiParams { period: Some(14) };
		let first_input = DiInput::from_candles(&candles, first_params);
		let first_result = di_with_kernel(&first_input, kernel)?;
		assert_eq!(first_result.plus.len(), candles.close.len());
		assert_eq!(first_result.minus.len(), candles.close.len());
		let second_params = DiParams { period: Some(14) };
		let second_input = DiInput::from_slices(&first_result.plus, &first_result.minus, &candles.close, second_params);
		let second_result = di_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.plus.len(), first_result.plus.len());
		assert_eq!(second_result.minus.len(), first_result.minus.len());
		Ok(())
	}
	fn check_di_accuracy_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = DiParams { period: Some(14) };
		let input = DiInput::from_candles(&candles, params);
		let di_result = di_with_kernel(&input, kernel)?;
		assert_eq!(di_result.plus.len(), candles.close.len());
		assert_eq!(di_result.minus.len(), candles.close.len());
		if di_result.plus.len() > 40 {
			for i in 40..di_result.plus.len() {
				assert!(!di_result.plus[i].is_nan());
				assert!(!di_result.minus[i].is_nan());
			}
		}
		Ok(())
	}
	#[cfg(debug_assertions)]
	fn check_di_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			DiParams::default(),                    // period: 14
			DiParams { period: Some(2) },           // minimum viable
			DiParams { period: Some(5) },           // small
			DiParams { period: Some(7) },           // small
			DiParams { period: Some(10) },          // small-medium
			DiParams { period: Some(20) },          // medium
			DiParams { period: Some(30) },          // medium-large
			DiParams { period: Some(50) },          // large
			DiParams { period: Some(100) },         // very large
			DiParams { period: Some(200) },         // extremely large
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = DiInput::from_candles(&candles, params.clone());
			let output = di_with_kernel(&input, kernel)?;
			
			// Check plus values
			for (i, &val) in output.plus.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in plus output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in plus output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in plus output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
			
			// Check minus values
			for (i, &val) in output.minus.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in minus output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in minus output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in minus output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_di_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}
	macro_rules! generate_all_di_tests {
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
	generate_all_di_tests!(
		check_di_partial_params,
		check_di_accuracy,
		check_di_with_zero_period,
		check_di_with_period_exceeding_data_length,
		check_di_very_small_data_set,
		check_di_with_slice_data_reinput,
		check_di_accuracy_nan_check,
		check_di_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = DiBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = DiParams::default();
		let row = output.plus_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		// Sanity: tail slice should match scalar indicator with default params.
		let scalar = DiInput::from_candles(&c, DiParams::default());
		let scalar_out = di_with_kernel(&scalar, Kernel::Scalar)?;
		let plus_tail = &row[row.len() - 5..];
		let scalar_tail = &scalar_out.plus[scalar_out.plus.len() - 5..];
		for (i, (&a, &b)) in plus_tail.iter().zip(scalar_tail.iter()).enumerate() {
			assert!(
				(a - b).abs() < 1e-8,
				"[{test}] batch/scalar plus mismatch idx={i}: {a} vs {b}"
			);
		}
		Ok(())
	}

	fn check_batch_period_range(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = DiBatchBuilder::new()
			.kernel(kernel)
			.period_range(10, 20, 5)
			.apply_candles(&c)?;

		assert_eq!(output.rows, 3);
		assert_eq!(output.cols, c.close.len());

		let periods = [10, 15, 20];
		for (i, p) in periods.iter().enumerate() {
			let param = DiParams { period: Some(*p) };
			let plus = output.plus_for(&param).expect("plus missing");
			let minus = output.minus_for(&param).expect("minus missing");
			assert_eq!(plus.len(), c.close.len());
			assert_eq!(minus.len(), c.close.len());
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
			(10, 10, 0),     // Static period (small)
			(14, 14, 0),     // Static period (default)
			(50, 50, 0),     // Static period (large)
			(7, 21, 7),      // Medium range with larger step
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = DiBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c)?;

			// Check plus values
			for (idx, &val) in output.plus.iter().enumerate() {
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
						at row {} col {} (flat index {}) in plus output with params: period={}",
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
						at row {} col {} (flat index {}) in plus output with params: period={}",
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
						at row {} col {} (flat index {}) in plus output with params: period={}",
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

			// Check minus values
			for (idx, &val) in output.minus.iter().enumerate() {
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
						at row {} col {} (flat index {}) in minus output with params: period={}",
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
						at row {} col {} (flat index {}) in minus output with params: period={}",
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
						at row {} col {} (flat index {}) in minus output with params: period={}",
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
	gen_batch_tests!(check_batch_period_range);
	gen_batch_tests!(check_batch_no_poison);
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "di")]
#[pyo3(signature = (high, low, close, period, kernel=None))]
pub fn di_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = DiParams { period: Some(period) };
	let input = DiInput::from_slices(high_slice, low_slice, close_slice, params);
	
	let (plus_vec, minus_vec) = py.allow_threads(|| {
		di_with_kernel(&input, kern)
			.map(|o| (o.plus, o.minus))
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok((
		plus_vec.into_pyarray(py),
		minus_vec.into_pyarray(py)
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "DiStream")]
pub struct DiStreamPy {
	inner: DiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DiStreamPy {
	#[new]
	pub fn new(period: usize) -> PyResult<Self> {
		let params = DiParams {
			period: Some(period),
		};
		let inner = DiStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(DiStreamPy { inner })
	}

	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
		self.inner.update(high, low, close)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "di_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
pub fn di_batch_py<'py>(
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
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = DiBatchRange { period: period_range };
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = high_slice.len();
	
	// Pre-allocate output arrays for batch operations
	let plus_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let minus_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let plus_slice = unsafe { plus_arr.as_slice_mut()? };
	let minus_slice = unsafe { minus_arr.as_slice_mut()? };
	
	let combos = py.allow_threads(|| {
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
		di_batch_inner_into(high_slice, low_slice, close_slice, &sweep, simd, true, plus_slice, minus_slice)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("plus", plus_arr.reshape((rows, cols))?)?;
	dict.set_item("minus", minus_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	
	Ok(dict)
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct DiJsOutput {
	plus: Vec<f64>,
	minus: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl DiJsOutput {
	#[wasm_bindgen(getter)]
	pub fn plus(&self) -> Vec<f64> {
		self.plus.clone()
	}
	
	#[wasm_bindgen(getter)]
	pub fn minus(&self) -> Vec<f64> {
		self.minus.clone()
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn di_js(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<DiJsOutput, JsValue> {
	let params = DiParams { period: Some(period) };
	let input = DiInput::from_slices(high, low, close, params);
	
	let mut output_plus = vec![0.0; high.len()];
	let mut output_minus = vec![0.0; high.len()];
	
	di_into_slice(&mut output_plus, &mut output_minus, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(DiJsOutput {
		plus: output_plus,
		minus: output_minus,
	})
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn di_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn di_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn di_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	plus_ptr: *mut f64,
	minus_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || plus_ptr.is_null() || minus_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let params = DiParams { period: Some(period) };
		let input = DiInput::from_slices(high, low, close, params);
		
		// Check for aliasing - any input could be the same as any output
		let has_aliasing = high_ptr as *const f64 == plus_ptr as *const f64 ||
						  high_ptr as *const f64 == minus_ptr as *const f64 ||
						  low_ptr as *const f64 == plus_ptr as *const f64 ||
						  low_ptr as *const f64 == minus_ptr as *const f64 ||
						  close_ptr as *const f64 == plus_ptr as *const f64 ||
						  close_ptr as *const f64 == minus_ptr as *const f64 ||
						  plus_ptr == minus_ptr;
		
		if has_aliasing {
			// Use temporary buffers
			let mut temp_plus = vec![0.0; len];
			let mut temp_minus = vec![0.0; len];
			di_into_slice(&mut temp_plus, &mut temp_minus, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let out_plus = std::slice::from_raw_parts_mut(plus_ptr, len);
			let out_minus = std::slice::from_raw_parts_mut(minus_ptr, len);
			out_plus.copy_from_slice(&temp_plus);
			out_minus.copy_from_slice(&temp_minus);
		} else {
			let out_plus = std::slice::from_raw_parts_mut(plus_ptr, len);
			let out_minus = std::slice::from_raw_parts_mut(minus_ptr, len);
			di_into_slice(out_plus, out_minus, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DiBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DiBatchJsOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
	pub periods: Vec<usize>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "di_batch")]
pub fn di_batch_js(high: &[f64], low: &[f64], close: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: DiBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = DiBatchRange { period: config.period_range };
	
	let output = di_batch_inner(high, low, close, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = DiBatchJsOutput {
		plus: output.plus,
		minus: output.minus,
		periods: output.combos.iter().map(|p| p.period.unwrap()).collect(),
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn di_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	plus_ptr: *mut f64,
	minus_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || plus_ptr.is_null() || minus_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		let sweep = DiBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let total_size = rows * len;
		
		let out_plus = std::slice::from_raw_parts_mut(plus_ptr, total_size);
		let out_minus = std::slice::from_raw_parts_mut(minus_ptr, total_size);
		
		di_batch_inner_into(high, low, close, &sweep, Kernel::Auto, false, out_plus, out_minus)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(rows)
	}
}
