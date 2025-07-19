//! # Pearson's Correlation Coefficient of High vs. Low (CORREL_HL)
//!
//! Measures the strength and direction of the linear relationship between
//! the `high` and `low` fields of candle data over a rolling window of length `period`.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 9.
//!
//! ## Errors
//! - **EmptyData**: correl_hl: The `high` or `low` arrays are empty.
//! - **InvalidPeriod**: correl_hl: `period` is zero or exceeds the data length.
//! - **DataLengthMismatch**: correl_hl: `high` and `low` arrays must have the same length.
//! - **NotEnoughValidData**: correl_hl: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: correl_hl: All `high` or `low` values are `NaN`.
//!
//! ## Returns
//! - **`Ok(CorrelHlOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the rolling window is filled.
//! - **`Err(CorrelHlError)`** otherwise.

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
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CorrelHlData<'a> {
	Candles { candles: &'a Candles },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct CorrelHlOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CorrelHlParams {
	pub period: Option<usize>,
}

impl Default for CorrelHlParams {
	fn default() -> Self {
		Self { period: Some(9) }
	}
}

#[derive(Debug, Clone)]
pub struct CorrelHlInput<'a> {
	pub data: CorrelHlData<'a>,
	pub params: CorrelHlParams,
}

impl<'a> CorrelHlInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: CorrelHlParams) -> Self {
		Self {
			data: CorrelHlData::Candles { candles },
			params,
		}
	}

	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: CorrelHlParams) -> Self {
		Self {
			data: CorrelHlData::Slices { high, low },
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, CorrelHlParams::default())
	}

	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(9)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct CorrelHlBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for CorrelHlBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl CorrelHlBuilder {
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
	pub fn apply(self, candles: &Candles) -> Result<CorrelHlOutput, CorrelHlError> {
		let params = CorrelHlParams { period: self.period };
		let input = CorrelHlInput::from_candles(candles, params);
		correl_hl_with_kernel(&input, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<CorrelHlOutput, CorrelHlError> {
		let params = CorrelHlParams { period: self.period };
		let input = CorrelHlInput::from_slices(high, low, params);
		correl_hl_with_kernel(&input, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<CorrelHlStream, CorrelHlError> {
		let params = CorrelHlParams { period: self.period };
		CorrelHlStream::try_new(params)
	}
}

#[derive(Debug, Error)]
pub enum CorrelHlError {
	#[error("correl_hl: Empty data provided (high or low).")]
	EmptyData,
	#[error("correl_hl: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("correl_hl: Data length mismatch between high and low.")]
	DataLengthMismatch,
	#[error("correl_hl: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("correl_hl: All values are NaN in high or low.")]
	AllValuesNaN,
}

#[inline]
pub fn correl_hl(input: &CorrelHlInput) -> Result<CorrelHlOutput, CorrelHlError> {
	correl_hl_with_kernel(input, Kernel::Auto)
}

pub fn correl_hl_with_kernel(input: &CorrelHlInput, kernel: Kernel) -> Result<CorrelHlOutput, CorrelHlError> {
	let (high, low) = match &input.data {
		CorrelHlData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|_e| CorrelHlError::EmptyData)?;
			let low = candles
				.select_candle_field("low")
				.map_err(|_e| CorrelHlError::EmptyData)?;
			(high, low)
		}
		CorrelHlData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() {
		return Err(CorrelHlError::EmptyData);
	}

	if high.len() != low.len() {
		return Err(CorrelHlError::DataLengthMismatch);
	}

	let period = input.get_period();
	if period == 0 || period > high.len() {
		return Err(CorrelHlError::InvalidPeriod {
			period,
			data_len: high.len(),
		});
	}

	let first_valid_idx = match high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !h.is_nan() && !l.is_nan())
	{
		Some(idx) => idx,
		None => return Err(CorrelHlError::AllValuesNaN),
	};

	if (high.len() - first_valid_idx) < period {
		return Err(CorrelHlError::NotEnoughValidData {
			needed: period,
			valid: high.len() - first_valid_idx,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first_valid_idx + period - 1;
	let mut out = alloc_with_nan_prefix(high.len(), warmup_period);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => correl_hl_scalar(high, low, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => correl_hl_avx2(high, low, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => correl_hl_avx512(high, low, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(CorrelHlOutput { values: out })
}

#[inline]
pub fn correl_hl_scalar(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	let mut sum_h = 0.0;
	let mut sum_h2 = 0.0;
	let mut sum_l = 0.0;
	let mut sum_l2 = 0.0;
	let mut sum_hl = 0.0;

	let pf = period as f64;

	#[inline(always)]
	fn corr_from_sums(sum_h: f64, sum_h2: f64, sum_l: f64, sum_l2: f64, sum_hl: f64, period: f64) -> f64 {
		let cov = sum_hl - (sum_h * sum_l / period);
		let var_h = sum_h2 - (sum_h * sum_h / period);
		let var_l = sum_l2 - (sum_l * sum_l / period);

		if var_h <= 0.0 || var_l <= 0.0 {
			0.0
		} else {
			cov / (var_h.sqrt() * var_l.sqrt())
		}
	}

	for i in first..(first + period) {
		let h = high[i];
		let l = low[i];
		sum_h += h;
		sum_h2 += h * h;
		sum_l += l;
		sum_l2 += l * l;
		sum_hl += h * l;
	}

	out[first + period - 1] = corr_from_sums(sum_h, sum_h2, sum_l, sum_l2, sum_hl, pf);

	for i in (first + period)..high.len() {
		let old_idx = i - period;
		let new_idx = i;

		let old_h = high[old_idx];
		let old_l = low[old_idx];
		let new_h = high[new_idx];
		let new_l = low[new_idx];

		if old_h.is_nan() || old_l.is_nan() || new_h.is_nan() || new_l.is_nan() {
			sum_h = 0.0;
			sum_h2 = 0.0;
			sum_l = 0.0;
			sum_l2 = 0.0;
			sum_hl = 0.0;
			for j in (i - period + 1)..=i {
				let hh = high[j];
				let ll = low[j];
				sum_h += hh;
				sum_h2 += hh * hh;
				sum_l += ll;
				sum_l2 += ll * ll;
				sum_hl += hh * ll;
			}
		} else {
			sum_h += new_h - old_h;
			sum_h2 += new_h * new_h - old_h * old_h;
			sum_l += new_l - old_l;
			sum_l2 += new_l * new_l - old_l * old_l;
			sum_hl += new_h * new_l - old_h * old_l;
		}

		out[i] = corr_from_sums(sum_h, sum_h2, sum_l, sum_l2, sum_hl, pf);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn correl_hl_avx2(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	correl_hl_scalar(high, low, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn correl_hl_avx512(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { correl_hl_avx512_short(high, low, period, first, out) }
	} else {
		unsafe { correl_hl_avx512_long(high, low, period, first, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn correl_hl_avx512_short(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	correl_hl_scalar(high, low, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn correl_hl_avx512_long(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	correl_hl_scalar(high, low, period, first, out)
}

#[derive(Debug, Clone)]
pub struct CorrelHlStream {
	period: usize,
	buffer_high: Vec<f64>,
	buffer_low: Vec<f64>,
	head: usize,
	filled: bool,
}

impl CorrelHlStream {
	pub fn try_new(params: CorrelHlParams) -> Result<Self, CorrelHlError> {
		let period = params.period.unwrap_or(9);
		if period == 0 {
			return Err(CorrelHlError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buffer_high: vec![f64::NAN; period],
			buffer_low: vec![f64::NAN; period],
			head: 0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, h: f64, l: f64) -> Option<f64> {
		self.buffer_high[self.head] = h;
		self.buffer_low[self.head] = l;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(self.dot_ring())
	}

	#[inline(always)]
	fn dot_ring(&self) -> f64 {
		let mut sum_h = 0.0;
		let mut sum_h2 = 0.0;
		let mut sum_l = 0.0;
		let mut sum_l2 = 0.0;
		let mut sum_hl = 0.0;
		for i in 0..self.period {
			let h = self.buffer_high[(self.head + i) % self.period];
			let l = self.buffer_low[(self.head + i) % self.period];
			sum_h += h;
			sum_h2 += h * h;
			sum_l += l;
			sum_l2 += l * l;
			sum_hl += h * l;
		}
		let pf = self.period as f64;
		let cov = sum_hl - (sum_h * sum_l / pf);
		let var_h = sum_h2 - (sum_h * sum_h / pf);
		let var_l = sum_l2 - (sum_l * sum_l / pf);
		if var_h <= 0.0 || var_l <= 0.0 {
			0.0
		} else {
			cov / (var_h.sqrt() * var_l.sqrt())
		}
	}
}

#[derive(Clone, Debug)]
pub struct CorrelHlBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for CorrelHlBatchRange {
	fn default() -> Self {
		Self { period: (9, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct CorrelHlBatchBuilder {
	range: CorrelHlBatchRange,
	kernel: Kernel,
}

impl CorrelHlBatchBuilder {
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

	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<CorrelHlBatchOutput, CorrelHlError> {
		correl_hl_batch_with_kernel(high, low, &self.range, self.kernel)
	}

	pub fn apply_candles(self, c: &Candles) -> Result<CorrelHlBatchOutput, CorrelHlError> {
		let high = c.select_candle_field("high").map_err(|_| CorrelHlError::EmptyData)?;
		let low = c.select_candle_field("low").map_err(|_| CorrelHlError::EmptyData)?;
		self.apply_slices(high, low)
	}
}

pub fn expand_grid(r: &CorrelHlBatchRange) -> Vec<CorrelHlParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(CorrelHlParams { period: Some(p) });
	}
	out
}

#[derive(Clone, Debug)]
pub struct CorrelHlBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CorrelHlParams>,
	pub rows: usize,
	pub cols: usize,
}

impl CorrelHlBatchOutput {
	pub fn row_for_params(&self, p: &CorrelHlParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
	}

	pub fn values_for(&self, p: &CorrelHlParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

pub fn correl_hl_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &CorrelHlBatchRange,
	k: Kernel,
) -> Result<CorrelHlBatchOutput, CorrelHlError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(CorrelHlError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	correl_hl_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn correl_hl_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &CorrelHlBatchRange,
	kern: Kernel,
) -> Result<CorrelHlBatchOutput, CorrelHlError> {
	correl_hl_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn correl_hl_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &CorrelHlBatchRange,
	kern: Kernel,
) -> Result<CorrelHlBatchOutput, CorrelHlError> {
	correl_hl_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn correl_hl_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &CorrelHlBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<CorrelHlBatchOutput, CorrelHlError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(CorrelHlError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !h.is_nan() && !l.is_nan())
		.ok_or(CorrelHlError::AllValuesNaN)?;

	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p {
		return Err(CorrelHlError::NotEnoughValidData {
			needed: max_p,
			valid: high.len() - first,
		});
	}

	let rows = combos.len();
	let cols = high.len();

	// Calculate warmup periods for each row
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();

	// Allocate uninitialized matrix
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Initialize NaN prefixes
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Convert to mutable slice for computation
	let mut buf_guard = ManuallyDrop::new(buf_mu);
	let values_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => correl_hl_row_scalar(high, low, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => correl_hl_row_avx2(high, low, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => correl_hl_row_avx512(high, low, first, period, out_row),
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

	// Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(CorrelHlBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn correl_hl_row_scalar(high: &[f64], low: &[f64], first: usize, period: usize, out: &mut [f64]) {
	correl_hl_scalar(high, low, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn correl_hl_row_avx2(high: &[f64], low: &[f64], first: usize, period: usize, out: &mut [f64]) {
	correl_hl_avx2(high, low, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correl_hl_row_avx512(high: &[f64], low: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		correl_hl_row_avx512_short(high, low, first, period, out)
	} else {
		correl_hl_row_avx512_long(high, low, first, period, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correl_hl_row_avx512_short(high: &[f64], low: &[f64], first: usize, period: usize, out: &mut [f64]) {
	correl_hl_avx512_short(high, low, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correl_hl_row_avx512_long(high: &[f64], low: &[f64], first: usize, period: usize, out: &mut [f64]) {
	correl_hl_avx512_long(high, low, period, first, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_correl_hl_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = CorrelHlParams { period: None };
		let input = CorrelHlInput::from_candles(&candles, params);
		let output = correl_hl_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_correl_hl_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = CorrelHlParams { period: Some(5) };
		let input = CorrelHlInput::from_candles(&candles, params);
		let result = correl_hl_with_kernel(&input, kernel)?;
		let expected = [
			0.04589155420456278,
			0.6491664099299647,
			0.9691259236943873,
			0.9915438003818791,
			0.8460608423095615,
		];
		let start_index = result.values.len() - 5;
		for (i, &val) in result.values[start_index..].iter().enumerate() {
			let exp = expected[i];
			let diff = (val - exp).abs();
			assert!(
				diff < 1e-7,
				"[{}] Value mismatch at index {}: expected {}, got {}",
				test_name,
				i,
				exp,
				val
			);
		}
		Ok(())
	}

	fn check_correl_hl_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 2.0, 3.0];
		let low = [1.0, 2.0, 3.0];
		let params = CorrelHlParams { period: Some(0) };
		let input = CorrelHlInput::from_slices(&high, &low, params);
		let result = correl_hl_with_kernel(&input, kernel);
		assert!(
			result.is_err(),
			"[{}] correl_hl should fail with zero period",
			test_name
		);
		Ok(())
	}

	fn check_correl_hl_period_exceeds_length(
		test_name: &str,
		kernel: Kernel,
	) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 2.0, 3.0];
		let low = [1.0, 2.0, 3.0];
		let params = CorrelHlParams { period: Some(10) };
		let input = CorrelHlInput::from_slices(&high, &low, params);
		let result = correl_hl_with_kernel(&input, kernel);
		assert!(
			result.is_err(),
			"[{}] correl_hl should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_correl_hl_data_length_mismatch(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 2.0, 3.0];
		let low = [1.0, 2.0];
		let params = CorrelHlParams { period: Some(2) };
		let input = CorrelHlInput::from_slices(&high, &low, params);
		let result = correl_hl_with_kernel(&input, kernel);
		assert!(
			result.is_err(),
			"[{}] correl_hl should fail on length mismatch",
			test_name
		);
		Ok(())
	}

	fn check_correl_hl_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, f64::NAN, f64::NAN];
		let low = [f64::NAN, f64::NAN, f64::NAN];
		let params = CorrelHlParams { period: Some(2) };
		let input = CorrelHlInput::from_slices(&high, &low, params);
		let result = correl_hl_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] correl_hl should fail on all NaN", test_name);
		Ok(())
	}

	fn check_correl_hl_from_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = CorrelHlParams { period: Some(9) };
		let input = CorrelHlInput::from_candles(&candles, params);
		let output = correl_hl_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_correl_hl_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 2.0, 3.0, 4.0, 5.0];
		let low = [0.5, 1.0, 1.5, 2.0, 2.5];
		let params = CorrelHlParams { period: Some(2) };
		let first_input = CorrelHlInput::from_slices(&high, &low, params.clone());
		let first_result = correl_hl_with_kernel(&first_input, kernel)?;
		let second_input = CorrelHlInput::from_slices(&first_result.values, &low, params);
		let second_result = correl_hl_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), low.len());
		Ok(())
	}

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_correl_hl_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with default parameters
		let input = CorrelHlInput::from_candles(&candles, CorrelHlParams::default());
		let output = correl_hl_with_kernel(&input, kernel)?;

		// Check every value for poison patterns
		for (i, &val) in output.values.iter().enumerate() {
			// Skip NaN values as they're expected in the warmup period
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_correl_hl_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	macro_rules! generate_all_correl_hl_tests {
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

	generate_all_correl_hl_tests!(
		check_correl_hl_partial_params,
		check_correl_hl_accuracy,
		check_correl_hl_zero_period,
		check_correl_hl_period_exceeds_length,
		check_correl_hl_data_length_mismatch,
		check_correl_hl_all_nan,
		check_correl_hl_from_candles,
		check_correl_hl_reinput,
		check_correl_hl_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = CorrelHlBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = CorrelHlParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test batch with multiple parameter combinations
		let output = CorrelHlBatchBuilder::new()
			.kernel(kernel)
			.period_range(5, 20, 5) // Test with periods 5, 10, 15, 20
			.apply_candles(&c)?;

		// Check every value in the entire batch matrix for poison patterns
		for (idx, &val) in output.values.iter().enumerate() {
			// Skip NaN values as they're expected in warmup periods
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();
			let row = idx / output.cols;
			let col = idx % output.cols;

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
				}
			}
		};
	}
	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}
