//! # Normalized Average True Range (NATR)
//!
//! A volatility indicator that normalizes the Average True Range (ATR) by the
//! closing price, expressed as a percentage. NATR is useful for comparing volatility
//! across different assets or time periods where price scales differ.
//!
//! ## Parameters
//! - **period**: The number of data points to consider for the ATR calculation (Wilder's method). Defaults to 14.
//!
//! ## Errors
//! - **EmptyData**: natr: Input data slice is empty.
//! - **InvalidPeriod**: natr: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: natr: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: natr: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(NatrOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the ATR window is filled.
//! - **`Err(NatrError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

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

	let period = input.get_period();
	let len = high.len().min(low.len()).min(close.len());
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

	let mut out = vec![f64::NAN; len];

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
					out[i] = 0.0;
				}
			}
		} else {
			let new_atr = ((prev_atr * ((period - 1) as f64)) + tr) / (period as f64);
			prev_atr = new_atr;

			let c = close[i];
			if c.is_finite() && c != 0.0 {
				out[i] = (new_atr / c) * 100.0;
			} else {
				out[i] = 0.0;
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
					return Some(0.0);
				}
			}
			return None;
		} else {
			let new_atr = ((self.prev_atr * ((self.period - 1) as f64)) + tr) / (self.period as f64);
			self.prev_atr = new_atr;
			if close.is_finite() && close != 0.0 {
				return Some((new_atr / close) * 100.0);
			} else {
				return Some(0.0);
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
	let first = high
		.iter()
		.position(|x| !x.is_nan())
		.unwrap_or(0)
		.max(low.iter().position(|x| !x.is_nan()).unwrap_or(0))
		.max(close.iter().position(|x| !x.is_nan()).unwrap_or(0));
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let len = high.len().min(low.len()).min(close.len());
	if len - first < max_p {
		return Err(NatrError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	let mut values = vec![f64::NAN; rows * cols];

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
	Ok(NatrBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
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
            }
        }
    }

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
		check_natr_streaming
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
}
